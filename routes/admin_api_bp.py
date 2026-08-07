"""Extracted from app.py — admin_api_bp (see route list below)."""
from __future__ import annotations
import logging
import time
import os
import threading
from datetime import datetime
from flask import Blueprint, jsonify, request
from extensions import limiter
from app import DASHBOARD_CACHE, CACHE_TTL
logger = logging.getLogger(__name__)

admin_api_bp = Blueprint("admin_api_bp", __name__)


# ── Lazy shims to app.py internals (resolved at request time) ──
def _cache_key(*a, **k):
    from app import _cache_key as _fn
    return _fn(*a, **k)

def _page_html_tmp_path(*a, **k):
    from app import _page_html_tmp_path as _fn
    return _fn(*a, **k)

def _touch_value_cache_bust(*a, **k):
    from app import _touch_value_cache_bust as _fn
    return _fn(*a, **k)

def get_league_ctx_from_cache(*a, **k):
    from app import get_league_ctx_from_cache as _fn
    return _fn(*a, **k)


@admin_api_bp.route("/api/prewarm-league")
@limiter.limit("60 per minute")
def api_prewarm_league():
    """Warm a league's context cache so a later switch to it renders without the
    cold Sleeper fetch (build_league_context is the dominant switch latency).

    The league switcher calls this in the background for the viewer's other
    leagues after a page loads. It only builds the shared context that any page
    render for that league would build anyway — no per-viewer data is returned
    (just ok/cached), so it's safe to call speculatively. Returns immediately
    when the context is already warm.
    """
    platform = (request.args.get("platform") or "sleeper").strip().lower()
    league_id = (request.args.get("league_id") or "").strip()
    try:
        season = int(request.args.get("season") or datetime.now().year)
    except (TypeError, ValueError):
        season = datetime.now().year
    if not league_id:
        return jsonify({"ok": False, "error": "league_id required"}), 400

    key = _cache_key(platform, season, league_id)
    entry = DASHBOARD_CACHE.get(key)
    if entry and (time.time() - entry.get("ts", 0) <= CACHE_TTL):
        return jsonify({"ok": True, "cached": True})
    try:
        get_league_ctx_from_cache(platform, league_id, season)
    except Exception:
        logger.debug("prewarm-league failed", exc_info=True)
        return jsonify({"ok": False}), 200
    return jsonify({"ok": True, "cached": False})


@admin_api_bp.route("/api/refresh-league", methods=["POST"])
@limiter.limit("4 per minute")
def api_refresh_league():
    """Force-expire a league context so the next request rebuilds it from source."""
    payload = request.get_json(silent=True) or {}
    platform = (payload.get("platform") or "sleeper").strip().lower()
    league_id = (payload.get("league_id") or "").strip()
    season = int(payload.get("season") or datetime.now().year)
    if not league_id:
        return jsonify({"error": "league_id required"}), 400
    key = _cache_key(platform, season, league_id)
    if key in DASHBOARD_CACHE:
        DASHBOARD_CACHE[key]["ts"] = 0       # expire context cache
        DASHBOARD_CACHE[key]["page_html"] = {}  # clear rendered HTML so pages re-render fresh
    # Also remove /tmp files so other gunicorn workers don't serve stale HTML
    for page in ("dashboard", "activity", "teams", "graphs", "standings", "weekly"):
        try:
            path = _page_html_tmp_path(platform, season, league_id, page)
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
    # Drop the archetype engine's memoized sim state + suggestion results for this
    # league, so strategy suggestions reflect the roster immediately after a refresh
    # instead of serving a cached result for the length of its TTL.
    try:
        from dashboard_services.archetype_engine import invalidate_league_caches
        invalidate_league_caches(platform, league_id, season)
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
    return jsonify({"ok": True})


@admin_api_bp.route("/api/flush-value-cache", methods=["POST"])
@limiter.limit("10 per minute")
def api_flush_value_cache():
    """
    Clear the in-memory model value cache so the next request fetches fresh data
    from the DB. Useful right after a cron run without restarting the app.

    Caller must pass the correct CRON_SECRET (same env var used by the cron job).
    """
    secret = os.environ.get("CRON_SECRET", "")
    provided = (request.get_json(force=True, silent=True) or {}).get("secret", "")
    # Require the secret to be set AND match - when CRON_SECRET is unset the
    # old `if secret and …` guard would pass any request (short-circuit on falsy).
    if not secret or provided != secret:
        return jsonify({"error": "unauthorized"}), 403

    global _MODEL_VALUE_CACHE, _MODEL_VALUE_CACHE_TS
    _MODEL_VALUE_CACHE    = None
    _MODEL_VALUE_CACHE_TS = 0
    # Bump the shared marker so the OTHER gunicorn workers (which each hold their
    # own in-memory copy) also bust on their next read -- otherwise this POST only
    # clears the single worker that handled it and the rest serve stale values
    # until their 15-min TTL lapses.
    _touch_value_cache_bust()
    # Drop the memoized DB current-values table too so trade eval/suggestions and
    # rookie rankings reload fresh values instead of waiting out its TTL.
    try:
        from dashboard_services.player_value_history import clear_current_values_cache
        clear_current_values_cache()
    except Exception:
        logger.debug("[flush-value-cache] current-values cache clear failed", exc_info=True)
    # Also drop the advanced-metrics daily caches (value table, position ranks,
    # metric leaderboards) so the page/modals serve freshly rebuilt values.
    try:
        from data_building.advanced_metrics import clear_daily_caches
        clear_daily_caches()
    except Exception:
        logger.debug("[flush-value-cache] adv-metrics cache clear failed", exc_info=True)
    return jsonify({"ok": True, "message": "Model value + advanced-metrics caches cleared - next request will reload from DB."})


@admin_api_bp.route("/api/run-daily-cron", methods=["POST"])
@limiter.limit("5 per hour")
def api_run_daily_cron():
    """
    Trigger a full cron_daily run in a background thread.

    Caller must pass the correct CRON_SECRET:
        curl -X POST /api/run-daily-cron -H 'Content-Type: application/json' \\
             -d '{"secret": "<CRON_SECRET>"}'

    Optional: pass "force": true to delete model_values.json first so all
    freshness guards are bypassed and values are fully rebuilt from scratch.

    Returns immediately; the cron runs in the background (check server logs).
    """
    secret   = os.environ.get("CRON_SECRET", "")
    body     = request.get_json(force=True, silent=True) or {}
    provided = body.get("secret", "")
    if not secret or provided != secret:
        return jsonify({"error": "unauthorized"}), 403

    force = bool(body.get("force", False))

    def _run_cron(force_rebuild: bool):
        from utils.paths import DATA_DIR
        try:
            if force_rebuild:
                # Remove model_values.json so freshness guards are all bypassed
                _mv = DATA_DIR / "model_values.json"
                if _mv.exists():
                    _mv.unlink()
                    logger.info("[run-daily-cron] Deleted %s to force full rebuild", _mv)
                # The board anchors the top-5 average to 999.9 fresh each run (no
                # basket EMA state to reset). Deleting model_values.json above also
                # means this forced run rebuilds unclamped, then subsequent daily
                # runs apply the ±10% per-player move clamp from there.
                # Set env var so cron_daily.main() also bypasses freshness guards
                os.environ["CRON_FORCE_REBUILD"] = "1"
            from cron_daily import main as _cron_main
            logger.info("[run-daily-cron] Starting (force=%s)", force_rebuild)
            _cron_main()
            # Flush the in-memory cache so the fresh values are served immediately
            global _MODEL_VALUE_CACHE, _MODEL_VALUE_CACHE_TS
            _MODEL_VALUE_CACHE    = None
            _MODEL_VALUE_CACHE_TS = 0
            # Also bust the sibling workers (see /api/flush-value-cache).
            _touch_value_cache_bust()
            try:
                from dashboard_services.player_value_history import clear_current_values_cache
                clear_current_values_cache()
            except Exception:
                logger.debug("[run-daily-cron] current-values cache clear failed", exc_info=True)
            try:
                from data_building.advanced_metrics import clear_daily_caches
                clear_daily_caches()
            except Exception:
                logger.debug("[run-daily-cron] adv-metrics cache clear failed", exc_info=True)
            logger.info("[run-daily-cron] Completed - cache flushed")
        except Exception as _e:
            logger.error("[run-daily-cron] Failed: %s", _e, exc_info=True)

    threading.Thread(target=_run_cron, args=(force,), daemon=True).start()
    return jsonify({
        "ok":     True,
        "force":  force,
        "message": "Daily cron triggered in background - check server logs for progress.",
    })


@admin_api_bp.route("/api/debug-values")
@limiter.limit("30 per minute")
def api_debug_values():
    """
    Diagnostic endpoint for value provenance.

    Default: the top-20 players by value_1qb with their WLS/calibration columns,
    so you can confirm whether the WLS trade calibration is landing in the DB
    (calibration_backing = trade weight behind WLS; a weight near 0 means WLS is
    effectively off and the value is the vendor/engine blend).

    ?player=<sleeper_id or name>: full provenance for one player — the
    player_values row (incl. calibration_backing), the FantasyCalc /
    DynastyProcess vendor values, and the last 14 daily history points.
    """
    _COLS = (
        "player_id, position, value_1qb, value_sf, "
        "calibrated_value_1qb, calibrated_value_sf, "
        "calibration_backing, calibration_backing_sf, "
        "calibration_source, calibration_weight, last_updated"
    )

    def _num(v):
        return float(v) if v is not None else None

    def _row_dict(r):
        return {
            "player_id":              str(r["player_id"]),
            "position":               str(r["position"] or ""),
            "value_1qb":              _num(r["value_1qb"]),
            "value_sf":               _num(r["value_sf"]),
            "calibrated_value_1qb":   _num(r["calibrated_value_1qb"]),
            "calibrated_value_sf":    _num(r["calibrated_value_sf"]),
            "calibration_backing":    _num(r["calibration_backing"]),
            "calibration_backing_sf": _num(r["calibration_backing_sf"]),
            "calibration_source":     str(r["calibration_source"] or ""),
            "calibration_weight":     _num(r["calibration_weight"]),
            "last_updated":           r["last_updated"].isoformat() if r["last_updated"] else None,
            "coalesce_gives":         (_num(r["calibrated_value_1qb"]) if r["calibrated_value_1qb"] is not None else _num(r["value_1qb"])),
        }

    try:
        from dashboard_services.db import get_conn as _gc

        # ── Optional single-player provenance (?player=id|name) ──────────────
        _player = (request.args.get("player") or "").strip()
        lookup = None
        if _player:
            from utils.utils import load_players_index as _lpi, normalize_name as _nn
            _idx = _lpi() or {}
            _ids: list = []
            if _player.isdigit() and _player in _idx:
                _ids = [_player]
            else:
                _q = _nn(_player)
                for _pid, _meta in _idx.items():
                    _nm = _nn((_meta or {}).get("name") or (_meta or {}).get("full_name") or "")
                    if _nm and (_nm == _q or (_q and _q in _nm)):
                        _ids.append(str(_pid))
                _ids = _ids[:5]

            entries = []
            if _ids:
                _ph = ",".join(["%s"] * len(_ids))
                with _gc() as _conn:
                    _prows = _conn.execute(
                        f"SELECT {_COLS} FROM player_values WHERE player_id IN ({_ph})",
                        tuple(_ids),
                    ).fetchall()
                    _pv = {str(r["player_id"]): _row_dict(r) for r in _prows}
                    _hist: dict = {}
                    _hrows = _conn.execute(
                        f"SELECT player_id, as_of_date, value, sf_value "
                        f"FROM player_value_history "
                        f"WHERE player_id IN ({_ph}) AND source = 'model' "
                        f"ORDER BY as_of_date DESC LIMIT 400",
                        tuple(_ids),
                    ).fetchall()
                    for _hr in _hrows:
                        _hid = str(_hr["player_id"])
                        _bucket = _hist.setdefault(_hid, [])
                        if len(_bucket) < 14:
                            _bucket.append({
                                "as_of_date": str(_hr["as_of_date"]),
                                "value":    _num(_hr["value"]),
                                "sf_value": _num(_hr["sf_value"]),
                            })

                # Vendor values (best-effort) for side-by-side comparison.
                _fc: dict = {}
                _dp_by_name: dict = {}
                try:
                    from data_building.external_data.external_values_scraper import (
                        load_fantasycalc_api_values, load_dynastyprocess_values,
                    )
                    for _r in (load_fantasycalc_api_values() or []):
                        _sid = str(_r.get("sleeper_id") or "").strip()
                        if _sid:
                            _fc[_sid] = _r.get("value")
                    for _r in (load_dynastyprocess_values() or []):
                        _nm = _nn(str(_r.get("player") or _r.get("name") or ""))
                        if _nm:
                            _dp_by_name[_nm] = (_r.get("value") or _r.get("value_1qb")
                                                or _r.get("dynasty_value"))
                except Exception:
                    logger.debug("[debug-values] vendor load skipped", exc_info=True)

                for _id in _ids:
                    _meta = _idx.get(_id) or {}
                    _nmk = _nn(_meta.get("name") or _meta.get("full_name") or "")
                    entries.append({
                        "player_id":            _id,
                        "name":                 _meta.get("name") or _meta.get("full_name") or "",
                        "player_values":        _pv.get(_id),
                        "fantasycalc_value":    _fc.get(_id),
                        "dynastyprocess_value": _dp_by_name.get(_nmk),
                        "history":              _hist.get(_id, []),
                    })
            lookup = {"query": _player, "matched": entries}

        with _gc() as _conn:
            rows = _conn.execute(
                f"""
                SELECT {_COLS}
                FROM player_values
                WHERE value_1qb IS NOT NULL AND value_1qb > 0
                  AND (position IS NULL OR position != 'PICK')
                ORDER BY value_1qb DESC NULLS LAST
                LIMIT 20
                """
            ).fetchall()
        from pathlib import Path as _Path
        import time as _time
        from utils.paths import DATA_DIR
        _mv_path = DATA_DIR / "model_values.json"
        _mv_mtime = (
            datetime.fromtimestamp(_mv_path.stat().st_mtime).isoformat()
            if _mv_path.exists() else "missing"
        )
        _cache_age = round(_time.time() - _MODEL_VALUE_CACHE_TS) if _MODEL_VALUE_CACHE_TS else None
        _basket_state = None
        _headroom_state = None
        try:
            from data_building.value_model_training import _load_state, _BASKET_STATE_KEY, _HEADROOM_STATE_KEY
            _basket_state = _load_state(_BASKET_STATE_KEY)
            _headroom_state = _load_state(_HEADROOM_STATE_KEY)
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
        _resp = {
            "model_values_json_mtime": _mv_mtime,
            "in_memory_cache_age_seconds": _cache_age,
            "in_memory_cache_size": len(_MODEL_VALUE_CACHE) if _MODEL_VALUE_CACHE else 0,
            "pipeline_state": {
                "basket_1qb": _basket_state,
                "headroom_1qb": _headroom_state,
                "note": "basket near 999.9 means _1qb_scale≈1.0 → top players capped at 999.9; reset via POST /api/run-daily-cron with force:true",
            },
            "top_players": [_row_dict(r) for r in rows],
        }
        if lookup is not None:
            _resp["lookup"] = lookup
        return jsonify(_resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
