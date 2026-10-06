"""Internal APIs: admin ops, analytics, and the UI audit hub.

Extracted from app.py to shrink the monolith (admin_api_bp + analytics_bp +
ui_audit_bp merged)."""
from __future__ import annotations
import hmac
import logging
import time
import os
import threading
from datetime import (
    datetime,
    timezone,
)
from flask import (
    Blueprint,
    Response,
    jsonify,
    redirect,
    request,
    session,
    url_for,
)
from extensions import limiter


def _dashboard_cache():
    # Lazy import: this module is imported by app.py during app creation,
    # so a module-level `from app import` would be circular.
    from app import DASHBOARD_CACHE

    return DASHBOARD_CACHE


import html
from dashboard_services.admin_auth import ADMIN_SESSION_KEY, _configured_password, is_admin, mark_admin_session, verify_admin_password

logger = logging.getLogger(__name__)

internal_bp = Blueprint("internal", __name__)


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

def _touch_league_bust(*a, **k):
    from app import _touch_league_bust as _fn
    return _fn(*a, **k)

def _league_ctx_cache_valid(*a, **k):
    from app import _league_ctx_cache_valid as _fn
    return _fn(*a, **k)

def get_league_ctx_from_cache(*a, **k):
    from app import get_league_ctx_from_cache as _fn
    return _fn(*a, **k)


# In-memory model value cache (also bust across gunicorn workers via the shared
# marker in _touch_value_cache_bust). Initialized here so readers like
# /api/debug-values never hit a NameError on a fresh worker before the first
# flush; previously these names only came into existence via `global` the first
# time /api/flush-value-cache ran.
_MODEL_VALUE_CACHE = None
_MODEL_VALUE_CACHE_TS = 0


@internal_bp.route("/api/prewarm-league")
@limiter.limit("60 per minute")
def api_prewarm_league():
    """Warm a league's context cache so a later switch to it renders without the
    cold Sleeper fetch (build_league_context is the dominant switch latency).

    The league switcher calls this in the background for the viewer's other
    leagues after a page loads. It only builds the shared context that any page
    render for that league would build anyway -- no per-viewer data is returned
    (just ok/cached), so it's safe to call speculatively. Returns immediately
    when the context is already warm.

    ESPN is intentionally skipped: a full context prewarm fetches weekly box
    scores and contended with the live page / player modal. Switch navigates
    to ESPN immediately (no refresh-league wait) and reuses any warm context;
    roster-freshness polling expires it if rosters changed.
    """
    platform = (request.args.get("platform") or "sleeper").strip().lower()
    league_id = (request.args.get("league_id") or "").strip()
    try:
        season = int(request.args.get("season") or datetime.now().year)
    except (TypeError, ValueError):
        season = datetime.now().year
    if not league_id:
        return jsonify({"ok": False, "error": "league_id required"}), 400
    if platform == "espn":
        return jsonify({"ok": True, "skipped": True, "reason": "espn"})

    key = _cache_key(platform, season, league_id)
    entry = _dashboard_cache().get(key)
    if _league_ctx_cache_valid(entry, platform, season, league_id):
        return jsonify({"ok": True, "cached": True})
    try:
        get_league_ctx_from_cache(platform, league_id, season)
    except Exception:
        logger.debug("prewarm-league failed", exc_info=True)
        return jsonify({"ok": False}), 200
    return jsonify({"ok": True, "cached": False})


def _refresh_league_authorized(
    *,
    platform: str,
    season: int,
    league_id: str,
    provided_secret: str,
    last_league_id: str,
    member_id,
    account_id,
) -> bool:
    """Pure auth decision for /api/refresh-league (session values passed in).

    Order: CRON_SECRET ops bypass, then "currently viewing this league", then
    verified league membership via the session's platform identity, then via
    the Google account: the league itself linked to the account, or a linked
    platform identity that is a verified member. (The session only carries a
    Sleeper identity after the explicit username flow, so Google-signed-in
    callers would otherwise be rejected despite the league being theirs.)
    """
    secret = os.environ.get("CRON_SECRET", "")
    if secret and provided_secret and hmac.compare_digest(provided_secret, secret):
        return True
    if str(last_league_id or "") == str(league_id):
        return True
    from dashboard_services.subscriptions import viewer_is_league_member

    if member_id and viewer_is_league_member(member_id, league_id, platform, season):
        return True
    if account_id:
        try:
            from dashboard_services.accounts import (
                list_account_platform_ids,
                list_user_leagues,
            )

            plat = (platform or "").strip().lower()
            # The league itself linked to this Google account.
            try:
                season_i = int(season) if season not in (None, "") else None
            except (TypeError, ValueError):
                season_i = None
            for lg in list_user_leagues(int(account_id)):
                if str(lg.get("platform") or "").lower() != plat:
                    continue
                if str(lg.get("league_id")) != str(league_id):
                    continue
                lg_season = lg.get("season")
                if season_i is None or lg_season in (None, "") or int(lg_season) == season_i:
                    return True
            # A linked platform identity that is a verified member (Sleeper).
            if plat == "sleeper":
                for pid in list_account_platform_ids(int(account_id), "sleeper"):
                    if viewer_is_league_member(pid, league_id, platform, season):
                        return True
        except Exception:
            logger.warning(
                "[refresh-league] account identity fallback failed", exc_info=True
            )
    return False


@internal_bp.route("/api/refresh-league", methods=["POST"])
@limiter.limit("10 per minute")
def api_refresh_league():
    """Force-expire a league context so the next request rebuilds it from source.

    Allowed when:
      - the request includes a valid ``CRON_SECRET`` (ops / automation), or
      - the caller is currently viewing this league (``last_league_id`` match), or
      - the caller is a verified member of the league, or
      - the caller's Google account has this league linked, or a linked
        platform identity that is a verified member (Sleeper).
    """
    payload = request.get_json(silent=True) or {}
    platform = (payload.get("platform") or "sleeper").strip().lower()
    league_id = (payload.get("league_id") or "").strip()
    try:
        season = int(payload.get("season") or datetime.now().year)
    except (TypeError, ValueError):
        season = datetime.now().year
    if not league_id:
        return jsonify({"error": "league_id required"}), 400

    if not _refresh_league_authorized(
        platform=platform,
        season=season,
        league_id=league_id,
        provided_secret=str(payload.get("secret") or ""),
        last_league_id=str(session.get("last_league_id") or ""),
        member_id=session.get("viewer_user_id") or session.get("viewer_username"),
        account_id=session.get("account_id"),
    ):
        return jsonify({"error": "forbidden"}), 403

    key = _cache_key(platform, season, league_id)
    _dc = _dashboard_cache()
    if key in _dc:
        # Mark for forced rebuild WITHOUT zeroing ts: the stale-fallback in
        # get_league_ctx_from_cache needs the old ts to serve last-known-good
        # data if the rebuild fails. Zeroing ts made time.time() - 0 exceed
        # the stale window, turning rebuild failures into HTTP 500s.
        _dc[key]["force_refresh"] = True
        _dc[key]["page_html"] = {}  # clear rendered HTML so pages re-render fresh
    # Sibling gunicorn workers keep their own DASHBOARD_CACHE; bump a shared
    # marker so their next read rebuilds too (otherwise Refresh only expires
    # the worker that handled the POST).
    try:
        _touch_league_bust(platform, season, league_id)
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
    # Clear this worker's provider payloads immediately. Sibling workers do the
    # same just before rebuilding when they observe the shared bust marker.
    if platform == "sleeper":
        try:
            from utils.utils import clear_league_provider_cache_for_league
            clear_league_provider_cache_for_league(league_id)
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
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
    # ESPN: drop process-cached League / globals for THIS league only so
    # post-draft rosters aren't stuck on empty pre-draft shells -- without
    # wiping every other ESPN room on the worker (player modal / live page).
    if platform == "espn":
        try:
            from dashboard_services.providers.espn_api import clear_espn_league_caches
            clear_espn_league_caches(league_id, season)
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
    # Draft grades are peer-relative to this league. Drop cached grade payloads
    # so a switch into this room rebuilds Value/Starters against its teams.
    try:
        from app import _DRAFT_GRADES_CACHE
        for _k in list(_DRAFT_GRADES_CACHE):
            if (
                isinstance(_k, tuple) and len(_k) >= 3
                and str(_k[0]) == str(platform) and str(_k[1]) == str(league_id)
            ):
                _DRAFT_GRADES_CACHE.pop(_k, None)
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
    return jsonify({"ok": True})


@internal_bp.route("/api/flush-value-cache", methods=["POST"])
@limiter.limit("10 per minute")
def api_flush_value_cache():
    """
    Clear the in-memory model value cache so the next request fetches fresh data
    from the DB. Useful right after a cron run without restarting the app.

    Caller must pass the correct CRON_SECRET (same env var used by the cron job).
    """
    secret = os.environ.get("CRON_SECRET", "")
    provided = str((request.get_json(force=True, silent=True) or {}).get("secret", "") or "")
    # Require the secret to be set AND match - when CRON_SECRET is unset the
    # old `if secret and …` guard would pass any request (short-circuit on falsy).
    if not secret or not provided or not hmac.compare_digest(provided, secret):
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


@internal_bp.route("/api/cron/pipeline-health", methods=["POST"])
@limiter.limit("60 per minute")
def api_cron_pipeline_health():
    """Ingest one cron step's pipeline health (called by cron_daily).

    The cron container's disk is invisible to the web container on Render, so
    the cron POSTs each step's status here and the web persists it to its own
    CACHE_DIR/pipeline_health.json, which /api/health/pipeline reads back.

    Caller must pass the correct CRON_SECRET (same env var used by the cron
    job). Fail closed when CRON_SECRET is unset.
    """
    payload = request.get_json(force=True, silent=True) or {}
    secret = os.environ.get("CRON_SECRET", "")
    provided = str(payload.get("secret") or "")
    # Require the secret to be set AND match - when CRON_SECRET is unset the
    # old `if secret and ...` guard would pass any request (short-circuit on falsy).
    if not secret or not provided or not hmac.compare_digest(provided, secret):
        return jsonify({"error": "unauthorized"}), 403
    step = str(payload.get("step") or "").strip()[:120]
    status = str(payload.get("status") or "").strip()
    if not step or status not in ("ok", "error", "timeout", "skipped"):
        return jsonify({"error": "step and a valid status (ok/error/timeout/skipped) are required"}), 400
    from utils.pipeline_health import write_step_health
    data = write_step_health(step, status)
    return jsonify({"ok": True, "step": step, "status": status, "steps": len(data)})


@internal_bp.route("/api/run-daily-cron", methods=["POST"])
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
    provided = str(body.get("secret", "") or "")
    if not secret or not provided or not hmac.compare_digest(provided, secret):
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


@internal_bp.route("/api/debug-values")
@limiter.limit("30 per minute")
def api_debug_values():
    """
    Diagnostic endpoint for value provenance.

    Auth: same CRON_SECRET gate as the sibling admin endpoints
    (/api/flush-value-cache, /api/run-daily-cron). Pass the secret as
    ``?secret=``, an ``X-Cron-Secret`` header, or JSON ``{"secret": ...}``.
    Fails closed with 403 when CRON_SECRET is unset or does not match.

    Default: the top-20 players by value_1qb with their WLS/calibration columns,
    so you can confirm whether the WLS trade calibration is landing in the DB
    (calibration_backing = trade weight behind WLS; a weight near 0 means WLS is
    effectively off and the value is the vendor/engine blend).

    ?player=<sleeper_id or name>: full provenance for one player -- the
    player_values row (incl. calibration_backing), the FantasyCalc /
    DynastyProcess vendor values, and the last 14 daily history points.
    """
    # CRON_SECRET check, identical to the sibling admin endpoints: require the
    # secret to be set AND match -- when CRON_SECRET is unset the old
    # `if secret and ...` guard would pass any request (short-circuit on falsy).
    # GET has no JSON body by convention, so ?secret= and X-Cron-Secret are
    # accepted in addition to the JSON body form the POST siblings use.
    secret = os.environ.get("CRON_SECRET", "")
    provided = str(
        request.args.get("secret")
        or request.headers.get("X-Cron-Secret")
        or (request.get_json(silent=True) or {}).get("secret")
        or ""
    )
    if not secret or not provided or not hmac.compare_digest(provided, secret):
        return jsonify({"error": "unauthorized"}), 403

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
    except Exception:
        logger.exception("[api_debug_values] Error")
        return jsonify({"error": "Internal error"}), 500


# ────────────────────────────────────────────────────────────────────────────
# Merged from routes/analytics_bp.py: Admin product-analytics dashboard (Phase 1).
# ────────────────────────────────────────────────────────────────────────────

# ── Inline SVG charts ─────────────────────────────────────────────────────────

def _bars_svg(pairs, width=680, height=190, bar_color="#4f8ff7", max_bar_w=48):
    """pairs: list of (label, value). Returns an inline SVG bar chart.

    Bars are capped at max_bar_w px and centered in their slots, so a sparse
    series (e.g. a single day of data) does not render as one giant bar.
    Every nonzero value gets a label above its bar, the x axis labels every
    point (rotated when the series is dense), and the gridlines carry y-axis
    tick labels."""
    if not pairs:
        return '<p class="muted">No data yet.</p>'
    maxv = max(v for _, v in pairs) or 1
    n = len(pairs)
    rotate = n > 15
    pad_l, pad_r, pad_t, pad_b = 34, 10, 24, 58 if rotate else 30
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    slot = plot_w / n
    bw = min(slot * 0.64, max_bar_w)
    parts = ['<svg viewBox="0 0 %d %d" class="chart" role="img">' % (width, height)]
    for frac in (0.0, 0.5, 1.0):
        gy = pad_t + plot_h * (1 - frac)
        cls = "grid base" if frac == 0.0 else "grid"
        parts.append(
            '<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="%s"/>'
            % (pad_l, gy, width - pad_r, gy, cls)
        )
        parts.append(
            '<text x="%d" y="%.1f" text-anchor="end" class="ytick">%d</text>'
            % (pad_l - 5, gy + 3.5, round(maxv * frac))
        )
    step = 1 if rotate else max(1, n // 12)
    for i, (label, value) in enumerate(pairs):
        frac = (value / maxv) if maxv else 0
        bh = max(frac * plot_h, 2 if value else 0)
        x = pad_l + i * slot + (slot - bw) / 2
        y = pad_t + plot_h - bh
        parts.append(
            '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="2" fill="%s">'
            '<title>%s: %s</title></rect>'
            % (x, y, bw, bh, bar_color, html.escape(str(label)), value)
        )
        if value:
            parts.append(
                '<text x="%.1f" y="%.1f" text-anchor="middle" class="vallab">%s</text>'
                % (x + bw / 2, y - 5, value)
            )
        if i % step == 0 or i == n - 1:
            cx = x + bw / 2
            if rotate:
                parts.append(
                    '<text x="%.1f" y="%d" text-anchor="end" class="axis" '
                    'transform="rotate(-45 %.1f %d)">%s</text>'
                    % (cx, height - 8, cx, height - 8, html.escape(str(label)))
                )
            else:
                parts.append(
                    '<text x="%.1f" y="%d" text-anchor="middle" class="axis">%s</text>'
                    % (cx, height - 10, html.escape(str(label)))
                )
    parts.append("</svg>")
    return "".join(parts)


def _pct(a, b):
    """Conversion percent a/b as a short string; 'n/a' when b is 0."""
    if not b:
        return "n/a"
    return "%.1f%%" % (100.0 * a / b)


# ── Page sections ─────────────────────────────────────────────────────────────

def _section(title, body, note=""):
    note_html = '<p class="muted">%s</p>' % html.escape(note) if note else ""
    return (
        '<section class="card"><h2>%s</h2>%s%s</section>'
        % (html.escape(title), note_html, body)
    )


def _feature_table(rows):
    """rows: [{week, event, count}] -> HTML table, weeks as rows, events as cols."""
    if not rows:
        return '<p class="muted">No data yet.</p>'
    events = sorted({r["event"] for r in rows})
    weeks = sorted({r["week"] for r in rows}, reverse=True)
    lookup = {(r["week"], r["event"]): r["count"] for r in rows}
    head = "".join("<th>%s</th>" % html.escape(e) for e in events)
    body_rows = []
    for w in weeks:
        cells = "".join(
            "<td>%d</td>" % lookup.get((w, e), 0) for e in events
        )
        body_rows.append("<tr><th scope='row'>%s</th>%s</tr>" % (html.escape(w), cells))
    return (
        '<div class="tablewrap"><table><thead><tr><th>Week</th>%s</tr></thead>'
        "<tbody>%s</tbody></table></div>" % (head, "".join(body_rows))
    )


def _retention_table(rows):
    if not rows:
        return '<p class="muted">No data yet.</p>'
    body_rows = []
    for r in rows:
        body_rows.append(
            "<tr><th scope='row'>%s</th><td>%d</td><td>%d</td><td>%s%%</td></tr>"
            % (html.escape(r["week"]), r["active"], r["returned"], r["rate"])
        )
    return (
        '<div class="tablewrap"><table><thead><tr>'
        "<th>Week</th><th>Active users</th><th>Returned from prior week</th>"
        "<th>Return rate</th></tr></thead><tbody>%s</tbody></table></div>"
        % "".join(body_rows)
    )


_BREAKDOWN_ROWS = [
    ("Raw distinct identities (old definition)", "headline"),
    ("Signed-in accounts", "signed_in"),
    ("Anonymous sessions", "anon_sessions"),
    ("Anonymous: one-and-done (1 pageview)", "anon_one_and_done"),
    ("Anonymous: engaged (2+ pageviews, never signed in)", "anon_engaged"),
    ("Anonymous sessions that also signed in (counted twice)", "anon_linked_sessions"),
    ("DAU (current definition: signed-in + engaged anonymous)", "realistic_preview"),
    ("Total pageviews", "total_pageviews"),
]


def _breakdown_html(breakdown, top_paths):
    """DAU decomposition table + top one-and-done paths line.

    Reconciles the chart's realistic DAU against the raw old-definition
    count. breakdown: {"utc": {...}, "ny": {...}} or None when the
    breakdown queries failed (the rest of the page must still render).
    """
    if not breakdown:
        return '<p class="muted">Breakdown unavailable.</p>'
    utc = breakdown.get("utc") or {}
    ny = breakdown.get("ny") or {}
    body_rows = []
    for label, key in _BREAKDOWN_ROWS:
        body_rows.append(
            "<tr><th scope='row'>%s</th><td>%d</td><td>%d</td></tr>"
            % (html.escape(label), int(utc.get(key, 0) or 0), int(ny.get(key, 0) or 0))
        )
    table = (
        '<div class="tablewrap"><table><thead><tr>'
        "<th>Metric</th><th>UTC day</th><th>New York day (chart day)</th>"
        "</tr></thead><tbody>%s</tbody></table></div>" % "".join(body_rows)
    )
    paths_html = ""
    if top_paths:
        listed = ", ".join(
            "%s: %d" % (html.escape(str(p["path"])), int(p["count"]))
            for p in top_paths
        )
        paths_html = (
            '<p class="muted">Top one-and-done paths (New York day): %s</p>' % listed
        )
    return table + paths_html


def _funnel_html(f):
    stages = [
        ("Visitors", f["visitors"], None),
        ("Signups", f["signups"], f["visitors"]),
        ("League linked", f["linked"], f["signups"]),
        ("PRO", f["pro"], f["linked"]),
    ]
    cards = []
    for name, count, base in stages:
        conv = "" if base is None else '<div class="conv">%s of prior step</div>' % _pct(count, base)
        cards.append(
            '<div class="funnel-step"><div class="funnel-num">%d</div>'
            '<div class="funnel-name">%s</div>%s</div>'
            % (count, html.escape(name), conv)
        )
    return '<div class="funnel">%s</div>' % "".join(cards)


# ── Admin password login ───────────────────────────────────────────────────

_LOGIN_STYLE = """
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; padding: 24px; background: #f6f7f9; color: #1c2333; }
  .wrap { max-width: 420px; margin: 60px auto; }
  .card { background: #fff; border: 1px solid #e3e6ec; border-radius: 12px;
          padding: 24px; }
  h1 { font-size: 20px; margin: 0 0 6px; }
  .sub { color: #5b6478; font-size: 13px; margin: 0 0 18px; }
  label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; }
  input[type=password] { width: 100%; box-sizing: border-box; font-size: 15px;
      padding: 10px 12px; border: 1px solid #d4d9e3; border-radius: 8px; }
  button { margin-top: 14px; width: 100%; font-size: 15px; font-weight: 600;
      padding: 11px; border: 0; border-radius: 8px; background: #4f8ff7;
      color: #fff; cursor: pointer; }
  button:disabled { background: #a9b4c9; cursor: default; }
  .error { background: #fdecec; border: 1px solid #f3b8b8; color: #a33;
          border-radius: 8px; padding: 10px 12px; font-size: 13px;
          margin-bottom: 14px; }
  .note { background: #fff8e6; border: 1px solid #f0d98c; border-radius: 8px;
          padding: 10px 12px; font-size: 13px; margin-bottom: 14px; }
"""


def _login_page(error="", password_configured=True):
    error_html = (
        '<div class="error">%s</div>' % html.escape(error) if error else ""
    )
    if password_configured:
        form = """
  <form method="post">
    <label for="password">Admin password</label>
    <input type="password" id="password" name="password" autocomplete="current-password" autofocus>
    <button type="submit">Log in</button>
  </form>"""
    else:
        form = (
            '<div class="note">No admin password is configured. '
            "Set the ADMIN_PASSWORD environment variable to enable password login.</div>"
            '<form method="post"><button type="submit" disabled>Log in</button></form>'
        )
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Admin login</title>
<style>%s</style>
</head>
<body>
<div class="wrap"><div class="card">
  <h1>Admin login</h1>
  <p class="sub">Sign in to view the analytics dashboard.</p>
  %s%s
</div></div>
</body>
</html>""" % (_LOGIN_STYLE, error_html, form)


@internal_bp.route("/admin/login", methods=["GET"])
def admin_login_form():
    if is_admin():
        return redirect("/admin/analytics")
    return Response(
        _login_page(password_configured=bool(_configured_password())),
        mimetype="text/html",
    )


@internal_bp.route("/admin/login", methods=["POST"])
@limiter.limit("10 per minute")
def admin_login_submit():
    if is_admin():
        return redirect("/admin/analytics")
    provided = request.form.get("password") or ""
    if verify_admin_password(provided):
        mark_admin_session()
        return redirect("/admin/analytics")
    # Generic message on purpose: do not distinguish a missing password
    # from a wrong one, or an unconfigured ADMIN_PASSWORD.
    return Response(
        _login_page(
            error="Incorrect password.",
            password_configured=bool(_configured_password()),
        ),
        mimetype="text/html",
    )


# ── Route ─────────────────────────────────────────────────────────────────────

def _stat_table(rows):
    """rows: list of (label, value) already-formatted strings."""
    if not rows:
        return ""
    trs = "".join(
        "<tr><th scope='row'>%s</th><td>%s</td></tr>"
        % (html.escape(str(label)), html.escape(str(value)))
        for label, value in rows
    )
    return '<div class="tablewrap"><table><tbody>%s</tbody></table></div>' % trs


def _rank_table(headers, rows):
    """A stats table with one label column plus numeric columns."""
    if not rows:
        return '<p class="muted">No data yet.</p>'
    ths = "".join("<th>%s</th>" % html.escape(str(h)) for h in headers)
    trs = []
    for r in rows:
        cells = "<th scope='row'>%s</th>" % html.escape(str(r[0])) + "".join(
            "<td>%s</td>" % format(v, ",") for v in r[1:]
        )
        trs.append("<tr>%s</tr>" % cells)
    return (
        '<div class="tablewrap"><table><thead><tr>%s</tr></thead>'
        "<tbody>%s</tbody></table></div>" % (ths, "".join(trs))
    )


def _paywall_html(summary) -> str:
    if summary is None:
        return "<p class='muted'>Paywall data unavailable.</p>"
    parts = [
        "<p class='funnel-line'>"
        f"<strong>{summary['total_views']:,}</strong> paywall views from "
        f"<strong>{summary['viewers']:,}</strong> viewers &rarr; "
        f"<strong>{summary['checkout_viewers']:,}</strong> reached checkout "
        f"({summary['checkout_pct']}%) &rarr; "
        f"<strong>{summary['subscribed_viewers']:,}</strong> subscribed "
        f"({summary['subscribed_pct']}%)"
        "</p>"
    ]
    parts.append(
        "<h3 class='subhead'>Views by surface</h3>"
        + _rank_table(
            ["Surface", "Views", "Viewers"],
            [(r["surface"], r["views"], r["viewers"]) for r in summary.get("by_surface", [])],
        )
    )
    if summary.get("by_metric"):
        parts.append(
            "<h3 class='subhead'>Views by locked metric</h3>"
            + _rank_table(
                ["Metric", "Views", "Viewers"],
                [(r["metric"], r["views"], r["viewers"]) for r in summary["by_metric"]],
            )
        )
    return "".join(parts)


def _traffic_html(sources, landings) -> str:
    if sources is None and landings is None:
        return "<p class='muted'>Traffic source data unavailable.</p>"
    parts = []
    if sources is not None:
        parts.append(
            "<h3 class='subhead'>By referrer</h3>"
            + _rank_table(
                ["Source", "Sessions", "Engaged", "Signed in"],
                [(r["source"], r["sessions"], r["engaged"], r["signed_in"]) for r in sources],
            )
        )
    if landings is not None:
        parts.append(
            "<h3 class='subhead'>Top landing pages</h3>"
            + _rank_table(
                ["Landing path", "Sessions", "Engaged"],
                [(r["path"], r["sessions"], r["engaged"]) for r in landings],
            )
        )
    return "".join(parts)


def _activation_html(cohort) -> str:
    if cohort is None:
        return "<p class='muted'>Activation cohort unavailable.</p>"
    parts = [
        _stat_table([
            ("Signups", f"{cohort['signups']:,}"),
            ("Linked within 24 hours", f"{cohort['linked_24h']:,} ({cohort['pct_24h']}%)"),
            ("Linked within 7 days", f"{cohort['linked_7d']:,} ({cohort['pct_7d']}%)"),
            ("Linked at any point", f"{cohort['linked_ever']:,} ({cohort['pct_ever']}%)"),
        ])
    ]
    if cohort.get("by_provider"):
        parts.append(
            "<h3 class='subhead'>First league provider (linked within 7 days)</h3>"
            + _rank_table(
                ["Provider", "Accounts"],
                [(r["provider"], r["count"]) for r in cohort["by_provider"]],
            )
        )
    return "".join(parts)


def _revenue_html(rev) -> str:
    if rev is None:
        return "<p class='muted'>Revenue data unavailable.</p>"
    return _stat_table([
        ("Checkout started (unique people)", f"{rev['checkout_identities']:,}"),
        ("PRO subscribed (events)", f"{rev['subscribed_events']:,}"),
        ("PRO subscribed (unique people)", f"{rev['subscribed_identities']:,}"),
        ("Checkout to subscribed", f"{rev['checkout_conversion_pct']}%"),
        ("PRO cancelled (events)", f"{rev['cancelled_events']:,}"),
        ("Currently active PRO subscriptions", f"{rev['active_pro_subscriptions']:,}"),
        ("New PRO subscriptions (window)", f"{rev['new_pro_subscriptions']:,}"),
    ])


def _feature_ranking_table(rankings) -> str:
    if rankings is None:
        return "<p class='muted'>Feature ranking unavailable.</p>"
    if rankings:
        return _rank_table(
            ["Feature", "Uses", "Unique users"],
            [(r["event"], r["uses"], r["users"]) for r in rankings],
        )
    return "<p class='muted'>No feature events yet.</p>"


@internal_bp.route("/api/analytics/paywall", methods=["POST"])
@limiter.limit("60 per minute")
def api_paywall_viewed():
    """Client beacon: a paywall or upsell nudge was displayed.

    Body: {"surface": <one of PAYWALL_SURFACES>, "metric"?: str,
    "path": "/..."}. Always 204: telemetry must never break the page,
    and invalid payloads are dropped silently rather than recorded.
    """
    import dashboard_services.analytics as _a

    try:
        payload = request.get_json(silent=True) or {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        return "", 204
    surface = payload.get("surface")
    path = payload.get("path")
    metric = payload.get("metric")
    if surface not in _a.PAYWALL_SURFACES:
        return "", 204
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or len(path) > 512
    ):
        return "", 204
    props: dict = {"surface": surface}
    if isinstance(metric, str) and metric.strip():
        props["metric"] = metric.strip()[:64]
    try:
        _a.track_event(
            _a.EVENT_PAYWALL_VIEWED,
            account_id=_a.account_id_from_session(),
            session_id=_a.ensure_anon_session_id(),
            path=path,
            props=props,
        )
    except Exception:
        pass
    return "", 204


@internal_bp.route("/admin/analytics")
def admin_analytics():
    if not is_admin():
        return Response("Not found", status=404)
    try:
        from dashboard_services import analytics as _a

        dau = _a.dau_last_30_days()
        wau = _a.wau_last_12_weeks()
        signups = _a.signups_per_day()
        usage = _a.feature_usage_by_week(8)
        retention = _a.week_over_week_return()
        funnel = _a.funnel_last_30_days()
        has_events = _a.events_table_ready()
    except Exception:
        logger.exception("[analytics] admin page data failed")
        return Response(
            "<h1>Analytics unavailable</h1>"
            "<p>The stats page hit a database error. Check the server logs.</p>",
            status=500, mimetype="text/html",
        )

    dau_pairs = _a.fill_daily_gaps(dau, "date", "users", 30, today=_a.ny_today())
    wau_pairs = _a.fill_weekly_gaps(wau, "week", "users", 12, today=_a.ny_today())
    signup_pairs = _a.fill_daily_gaps(signups, "date", "signups", 30)

    # DAU breakdown (reconciliation): fetched separately so a failure here
    # can never take down the rest of the page.
    try:
        breakdown = _a.dau_breakdown()
        breakdown_paths = _a.one_and_done_top_paths(5)
    except Exception:
        logger.exception("[analytics] dau breakdown failed")
        breakdown = None
        breakdown_paths = []

    # Owner metrics: each fetched in isolation (same pattern as the
    # breakdown) so one failing query degrades only its own section.
    try:
        feature_ranking = _a.feature_usage_ranking()
    except Exception:
        logger.exception("[analytics] feature ranking failed")
        feature_ranking = None
    try:
        signed_in_retention = _a.account_retention()
    except Exception:
        logger.exception("[analytics] account retention failed")
        signed_in_retention = None
    try:
        traffic = _a.traffic_sources()
        landings = _a.top_landing_paths()
    except Exception:
        logger.exception("[analytics] traffic sources failed")
        traffic = None
        landings = None
    try:
        paywall = _a.paywall_summary()
    except Exception:
        logger.exception("[analytics] paywall summary failed")
        paywall = None
    try:
        cohort = _a.activation_cohort()
    except Exception:
        logger.exception("[analytics] activation cohort failed")
        cohort = None
    try:
        revenue = _a.revenue_summary()
    except Exception:
        logger.exception("[analytics] revenue summary failed")
        revenue = None

    empty_note = (
        "Event collection just started, so these charts fill in over the coming days. "
        "Signups come from the existing accounts table and are available now."
        if not has_events else ""
    )

    # Self-exclusion status: show what is filtering the numbers on this page.
    status_bits = []
    excluded_ids = sorted(_a.excluded_account_ids())
    if excluded_ids:
        status_bits.append(
            "Excluding account%s %s from all numbers below."
            % ("s" if len(excluded_ids) != 1 else "",
               ", ".join(str(i) for i in excluded_ids))
        )
    if session.get(ADMIN_SESSION_KEY):
        status_bits.append("Admin session: your visits are not recorded.")
    status_html = (
        '<p class="sub">%s</p>' % " ".join(html.escape(b) for b in status_bits)
        if status_bits else ""
    )

    body = "".join([
        _section("Daily active users", _bars_svg(dau_pairs),
                 "Signed-in accounts plus anonymous visitors with 2+ pages, "
                 "New York day, last 30 days."),
        _section("DAU breakdown: today",
                 _breakdown_html(breakdown, breakdown_paths),
                 "What today's DAU is made of. The chart above now uses the "
                 "realistic definition on a New York day; the raw row is the "
                 "old definition for comparison."),
        _section("Weekly active users", _bars_svg(wau_pairs, bar_color="#34c98e"),
                 "Same definition per week (signed-in accounts plus engaged "
                 "anonymous visitors), last 12 weeks."),
        _section("Signups per day", _bars_svg(signup_pairs, bar_color="#f5a623"),
                 "New accounts from the accounts table, last 30 days."),
        _section("Most used features (last 30 days)",
                 _feature_ranking_table(feature_ranking),
                 "Non-pageview events ranked by uses. Unique users counts "
                 "accounts when signed in, else sessions."),
        _section("Feature usage by week", _feature_table(usage),
                 "Explicit product events per week, last 8 weeks. Pageviews excluded."),
        _section("Week-over-week return", _retention_table(retention),
                 "Share of each week's active users who were also active the prior week."),
        _section("Signed-in retention",
                 _retention_table(signed_in_retention)
                 if signed_in_retention is not None
                 else '<p class="muted">Signed-in retention unavailable.</p>',
                 "Same return rate for signed-in accounts only (no anonymous "
                 "sessions), New York weeks, last 9 weeks."),
        _section("Funnel: visitor to PRO", _funnel_html(funnel),
                 "Period totals for the last 30 days. Not a strict cohort funnel."),
        _section("Activation: signup to league linked (cohort)",
                 _activation_html(cohort),
                 "Accounts created in the last 30 days and how quickly they "
                 "linked a first league. Unlike the funnel above, every stage "
                 "counts the same accounts. Caveat: league rows that predate "
                 "the added_at backfill carry the migration date, so old links "
                 "can read as linked on that date."),
        _section("Revenue", _revenue_html(revenue),
                 "Last 30 days. Subscription counts come from the per-league "
                 "subscription table. MRR is not shown: stored subscription "
                 "rows carry no price; the amount charged lives in Stripe."),
        _section("Paywalls", _paywall_html(paywall),
                 "Last 30 days. One view is one plan-modal open or one inline "
                 "nudge render. Conversion counts a viewer whose checkout or "
                 "subscription came after their first view."),
        _section("Traffic sources", _traffic_html(traffic, landings),
                 "First pageview of each session in the last 30 days. "
                 "Referrers are recorded host-only; sessions with no external "
                 "referrer are direct. Engaged means 2+ pageviews and never "
                 "signed in."),
    ])

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Product Analytics</title>
<style>
  :root { color-scheme: light; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; padding: 24px; background: #f6f7f9; color: #1c2333; }
  .wrap { max-width: 1060px; margin: 0 auto; }
  h1 { font-size: 24px; margin: 0 0 4px; }
  .sub { color: #5b6478; margin: 0 0 20px; font-size: 13px; }
  .card { background: #fff; border: 1px solid #e3e6ec; border-radius: 12px;
          padding: 18px 20px; margin-bottom: 18px; }
  .card h2 { font-size: 16px; margin: 0 0 10px; }
  .card h3.subhead { font-size: 13px; margin: 16px 0 6px; }
  .funnel-line { font-size: 14px; margin: 0 0 8px; }
  .muted { color: #7a8398; font-size: 12px; }
  .chart { width: 100%%; height: auto; display: block; }
  .chart rect { fill: #4f8ff7; }
  .chart text.vallab { font-size: 11px; fill: #5b6478; font-weight: 600; }
  .chart text.axis { font-size: 10px; fill: #7a8398; }
  .chart text.ytick { font-size: 10px; fill: #a0a8bb; }
  .chart line.grid { stroke: #edf0f5; stroke-width: 1; }
  .chart line.grid.base { stroke: #dfe3ea; }
  .tablewrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%%; font-size: 13px; }
  th, td { border: 1px solid #e8ebf1; padding: 7px 10px; text-align: right; }
  th[scope=row], thead th { text-align: left; background: #f2f4f8; }
  td { text-align: right; }
  .funnel { display: flex; gap: 12px; flex-wrap: wrap; }
  .funnel-step { flex: 1 1 160px; background: #f2f4f8; border-radius: 10px;
                padding: 14px; text-align: center; }
  .funnel-num { font-size: 28px; font-weight: 700; }
  .funnel-name { font-size: 13px; color: #5b6478; margin-top: 2px; }
  .funnel .conv { font-size: 12px; color: #2f7d4f; margin-top: 6px; font-weight: 600; }
  .notice { background: #fff8e6; border: 1px solid #f0d98c; border-radius: 10px;
            padding: 12px 16px; margin-bottom: 18px; font-size: 13px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Product Analytics</h1>
  <p class="sub">First-party usage stats. Generated %s.</p>
  %s
  %s
  %s
</div>
</body>
</html>""" % (
        html.escape(generated),
        ('<div class="notice">%s</div>' % html.escape(empty_note)) if empty_note else "",
        status_html,
        body,
    )
    return Response(page, mimetype="text/html")


# ────────────────────────────────────────────────────────────────────────────
# Merged from routes/ui_audit_bp.py: Local UI/UX audit hub -- only registered when ``UI_AUDIT=1``.
# ────────────────────────────────────────────────────────────────────────────

def _hub_body() -> str:
    # Lazy import: utils.ui_audit_fixture pulls pandas, which isn't installed
    # in the "Python lint & syntax" CI job. Only the ui-audit routes need it.
    from utils.ui_audit_fixture import (
        UI_AUDIT_LEAGUE_ID,
        _DEFAULT_PLATFORM,
        all_audit_hrefs,
        league_page_href,
    )

    public_links: list[tuple[str, str]] = []
    league_links: list[tuple[str, str]] = []
    for href, label in all_audit_hrefs():
        if href.startswith(f"/{_DEFAULT_PLATFORM}/"):
            league_links.append((href, label))
        else:
            public_links.append((href, label))

    def _section(title: str, links: list[tuple[str, str]]) -> str:
        rows = "".join(
            f'<li><a href="{html.escape(h)}">{html.escape(lbl)}</a>'
            f' <span class="ui-audit-path">{html.escape(h)}</span></li>'
            for h, lbl in links
        )
        return (
            f'<section class="card central ui-audit-section">'
            f'<div class="card-header"><h2>{html.escape(title)}</h2></div>'
            f'<div class="card-body"><ul class="ui-audit-links">{rows}</ul></div>'
            f"</section>"
        )

    dash = league_page_href("dashboard")
    return (
        '<style>'
        ".ui-audit-hero{margin:0 0 1rem;color:var(--text-muted);font-size:15px;line-height:1.5}"
        ".ui-audit-links{list-style:none;margin:0;padding:0;display:grid;gap:10px}"
        ".ui-audit-links a{font-weight:600}"
        ".ui-audit-path{display:block;font-size:12px;color:var(--text-muted);font-family:monospace}"
        ".ui-audit-actions{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 1.25rem}"
        ".ui-audit-actions .btn{min-height:44px}"
        "</style>"
        '<div class="ui-audit-hero">'
        "<p>Deterministic mock league <strong>UI Audit Dynasty</strong> "
        f"(<code>{UI_AUDIT_LEAGUE_ID}</code>) -- week 11, 10 teams, in-season. "
        "No live Sleeper calls.</p>"
        '<div class="ui-audit-actions">'
        f'<a class="btn btn-primary" href="{url_for("internal.bootstrap")}">'
        "Bootstrap signed-in session</a>"
        f'<a class="btn btn-secondary" href="{html.escape(dash)}">Open dashboard</a>'
        "</div></div>"
        + _section("Public & account pages", public_links)
        + _section("League pages (mock data)", league_links)
    )


@internal_bp.route("/ui-audit")
def hub():
    from utils.ui_audit_fixture import ui_audit_enabled

    if not ui_audit_enabled():
        return ("UI audit mode is off. Set UI_AUDIT=1 and restart.", 404)
    from app import render_page

    body = _hub_body()
    return render_page(
        "UI Audit Hub",
        None,
        "",
        body,
        description="Local UI/UX walkthrough catalog for BR Fantasy.",
    )


@internal_bp.route("/ui-audit/bootstrap")
def bootstrap():
    from utils.ui_audit_fixture import (
        bootstrap_viewer_session,
        league_page_href,
        ui_audit_enabled,
    )

    if not ui_audit_enabled():
        return ("UI audit mode is off.", 404)
    bootstrap_viewer_session(session)
    session.modified = True
    nxt = request.args.get("next") or league_page_href("dashboard")
    if not nxt.startswith("/"):
        nxt = league_page_href("dashboard")
    return redirect(nxt)
