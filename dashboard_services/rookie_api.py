"""
Rookie prospect API blueprint.

Endpoints:
    GET  /api/prospects/rankings?year=2026&pos=WR&league_type=1qb
    GET  /api/prospects/player/<player_id>
    GET  /api/prospects/active-class
    POST /api/prospects/prospects        (add/update one or more prospects)
    POST /api/prospects/refresh          (triggers pipeline re-run)
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from flask import Blueprint, jsonify, request

log = logging.getLogger(__name__)

rookie_bp = Blueprint("prospects", __name__, url_prefix="/api/prospects")

# In-memory cache so we don't re-run the pipeline on every page load.
# Invalidated on refresh or on first hit per process.
_cache: Dict[Any, List[Dict[str, Any]]] = {}


def _nfl_draft_complete(draft_year: int) -> bool:
    from data_building.rookie_pipeline.pipeline import is_draft_complete
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            return is_draft_complete(draft_year, conn)
    except Exception:
        return is_draft_complete(draft_year)


def _get_rankings(draft_year: int) -> List[Dict[str, Any]]:
    draft_done = _nfl_draft_complete(draft_year)
    cache_key = (draft_year, draft_done)
    if cache_key not in _cache:
        from data_building.rookie_pipeline.pipeline import get_rookie_rankings_from_db
        _cache[cache_key] = get_rookie_rankings_from_db(draft_year, filter_undrafted=draft_done)
    return _cache[cache_key]


def _safe_float(v, default=None):
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _row_to_dict(row: Dict) -> Dict:
    """Serialise a row dict to JSON-safe types."""
    out = {}
    for k, v in row.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


@rookie_bp.route("/active-class")
def active_class():
    from data_building.rookie_pipeline.pipeline import get_active_rookie_class
    year = get_active_rookie_class()
    return jsonify({"draft_class_year": year})


@rookie_bp.route("/rankings")
def rankings():
    try:
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class
        from data_building.rookie_pipeline.value_translation import format_draft_capital

        year = request.args.get("year", type=int) or get_active_rookie_class()
        pos  = (request.args.get("pos") or "").upper() or None
        league_type = (request.args.get("league_type") or "1qb").lower()
        league_size = request.args.get("league_size", type=int) or 10

        rows = _get_rankings(year)

        # Optional server-side position filter (client can also filter)
        if pos:
            rows = [r for r in rows if (r.get("position") or "").upper() == pos]

        total_players = len(rows)

        # Build response list with value field chosen by league settings
        result = []
        for r in rows:
            d = _row_to_dict(r)

            # Choose value based on settings
            if league_type == "sf":
                val_key = "rookie_sf_value" if league_size == 10 else f"rookie_sf_value_{league_size}"
                d["display_value"] = d.get(val_key) or d.get("rookie_sf_value")
            else:
                val_key = "rookie_value" if league_size == 10 else f"rookie_value_{league_size}"
                d["display_value"] = d.get(val_key) or d.get("rookie_value")

            # Draft capital label
            d["draft_capital_label"] = format_draft_capital(
                d.get("projected_round"),
                d.get("projected_pick"),
                d.get("projected_pick_low"),
                d.get("projected_pick_high"),
            )
            
            # Add headshot URL as espnHeadshot for modal compatibility
            if d.get("headshot_url"):
                d["espnHeadshot"] = d["headshot_url"]
            
            result.append(d)

        # Overlay dynasty rookie ADP for both SF and 1QB formats.
        # Returns adp_rank (1QB) and sf_adp_rank (SF) so the client can pick the
        # right field without needing a separate fetch per format.
        try:
            from dashboard_services.db import get_conn as _gc
            _ADP_SQL = """
                SELECT player_id,
                       SUM(avg_pick * sample_size) / NULLIF(SUM(sample_size), 0) AS avg_pick
                FROM draft_adp
                WHERE season = %s AND draft_type = 'rookie' AND is_superflex = %s
                GROUP BY player_id
                HAVING SUM(sample_size) >= 1
                ORDER BY avg_pick ASC
            """
            with _gc() as _conn:
                _sf_rows  = _conn.execute(_ADP_SQL, (year, True)).fetchall()
                _qb1_rows = _conn.execute(_ADP_SQL, (year, False)).fetchall()
            _sf_map  = {str(r["player_id"]): float(r["avg_pick"]) for r in _sf_rows}
            _qb1_map = {str(r["player_id"]): float(r["avg_pick"]) for r in _qb1_rows}
            for d in result:
                sid = str(d.get("sleeper_id") or "")
                if sid:
                    if sid in _sf_map:
                        d["sf_adp_rank"] = _sf_map[sid]
                    if sid in _qb1_map:
                        d["adp_rank"] = _qb1_map[sid]
        except Exception:
            pass

        # FC fallback: for each format, fill in any prospects still missing ADP.
        # Results are cached in-memory per process (keyed by date) to avoid live
        # network calls on every request.
        _FC_CACHE: dict = getattr(_get_rankings, "_fc_cache", {})
        if not hasattr(_get_rankings, "_fc_cache"):
            _get_rankings._fc_cache = _FC_CACHE  # type: ignore[attr-defined]

        def _get_fc_data(is_sf_flag: bool):
            from datetime import date as _date
            import requests as _req, json as _json
            _key = ("sf" if is_sf_flag else "1qb", _date.today().isoformat())
            if _key in _FC_CACHE:
                return _FC_CACHE[_key]
            try:
                from utils.paths import DATA_DIR
                _cache_file = DATA_DIR / f"fc_dynasty_rookie_adp_{'sf' if is_sf_flag else '1qb'}_{_key[1]}.json"
                if _cache_file.exists():
                    data = _json.loads(_cache_file.read_text())
                else:
                    _num_qbs = 2 if is_sf_flag else 1
                    _resp = _req.get(
                        f"https://fantasycalc.com/api/values/current?numQbs={_num_qbs}&type=1&ppr=0.5",
                        timeout=10, headers={"User-Agent": "fantasy-dashboard/1.0"},
                    )
                    data = _resp.json() if _resp.ok else []
                    try:
                        _cache_file.write_text(_json.dumps(data))
                    except Exception:
                        pass
                _FC_CACHE[_key] = data
                return data
            except Exception:
                return []

        def _apply_fc_adp(prospects, adp_field: str, is_sf_flag: bool):
            _fc_data = _get_fc_data(is_sf_flag)
            _fc_by_sid = {}
            for _entry in (_fc_data or []):
                _p = _entry.get("player") or {}
                _sid = str(_p.get("sleeperId") or "")
                if _sid and _p.get("rosterPosition") != "P":
                    _fc_by_sid[_sid] = _entry.get("overallRank") or 9999
            ranked = sorted(
                [(d, _fc_by_sid.get(str(d.get("sleeper_id") or ""), 9999)) for d in prospects],
                key=lambda x: x[1],
            )
            for _rank, (_d, _fc_rank) in enumerate(ranked, start=1):
                if _fc_rank < 9999:
                    _d[adp_field] = float(_rank)

        _needs_sf  = [d for d in result if d.get("sf_adp_rank")  is None]
        _needs_qb1 = [d for d in result if d.get("adp_rank") is None]
        if _needs_sf:
            try:
                _apply_fc_adp(_needs_sf, "sf_adp_rank", True)
            except Exception:
                pass
        if _needs_qb1:
            try:
                _apply_fc_adp(_needs_qb1, "adp_rank", False)
            except Exception:
                pass

        # Overlay values from the main player_values DB for linked prospects,
        # so the prospects page shows the same numbers as the /players page.
        try:
            from dashboard_services.player_value_history import load_current_values_from_db
            _pv_rows = load_current_values_from_db() or []
            _pv_map = {str(r.get("id") or ""): r for r in _pv_rows}
            _val_keys = ("value", "sf_value",
                         "value_8", "value_12", "value_14",
                         "sf_value_8", "sf_value_12", "sf_value_14")
            for d in result:
                _sid = str(d.get("sleeper_id") or "")
                if _sid and _sid in _pv_map:
                    _pv = _pv_map[_sid]
                    for _vk in _val_keys:
                        if _pv.get(_vk) is not None:
                            d[_vk] = float(_pv[_vk])
        except Exception:
            pass

        # Sort: tier ascending, then display_value descending within each tier
        result.sort(key=lambda x: (x.get("tier") or 99, -(x.get("display_value") or 0)))

        return jsonify({
            "draft_class_year": year,
            "total_players": total_players,
            "rankings": result,
        })

    except Exception as exc:
        log.exception("[rookie_api] /rankings error")
        return jsonify({"error": str(exc)}), 500


@rookie_bp.route("/player/<player_id>")
def player_detail(player_id: str):
    try:
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class
        year = request.args.get("year", type=int) or get_active_rookie_class()
        rows = _get_rankings(year)
        row  = next((r for r in rows if r["player_id"] == player_id), None)
        if not row:
            return jsonify({"error": "Player not found"}), 404
        
        player_data = _row_to_dict(row)
        
        # Add headshot URL as espnHeadshot for modal compatibility
        if player_data.get("headshot_url"):
            player_data["espnHeadshot"] = player_data["headshot_url"]
        
        return jsonify(player_data)
    except Exception as exc:
        log.exception("[rookie_api] /player error")
        return jsonify({"error": str(exc)}), 500


@rookie_bp.route("/prospects", methods=["POST"])
def add_prospects():
    """
    Add or update one or more prospects with their full data.

    Accepts a single prospect object or {"prospects": [...]}.

    Required fields per prospect:  name, position, draft_class_year
    Optional fields:               player_id, school, age, height_inches,
                                   weight_lbs, early_declare, transfer_history,
                                   headshot_url, seasons, athleticism

    seasons[] fields:
        season, games_played,
        pass_yards, pass_tds, pass_attempts, completions, interceptions,
        rush_attempts, rush_yards, rush_tds,
        receptions, targets, receiving_yards, receiving_tds,
        yds_per_carry, yds_per_reception, yds_per_attempt,
        completion_pct, td_int_ratio, dominator_rating,
        market_share_yards, market_share_tds,
        team, conference, team_pass_rate

    athleticism fields:
        forty_yard, vertical_inches, broad_jump_in, three_cone,
        short_shuttle, bench_reps, speed_score, ras_score

    Returns: {"added": N, "prospects": [scored_row, ...]}
    Each scored row includes all component scores, values, tier, and rank.
    """
    try:
        body = request.json or {}

        # Accept single prospect dict or {"prospects": [...]}
        if "prospects" in body:
            incoming = body["prospects"]
        elif "name" in body:
            incoming = [body]
        else:
            return jsonify({"error": 'Expected a prospect object or {"prospects": [...]}'}), 400

        if not incoming:
            return jsonify({"error": "No prospects provided"}), 400

        from data_building.rookie_pipeline.ingestion import normalize_prospect
        from data_building.rookie_pipeline.prospect_model import score_prospect
        from data_building.rookie_pipeline.mock_draft_consensus import build_mock_draft_consensus
        from data_building.rookie_pipeline.value_translation import translate_score_to_value, format_draft_capital
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class

        def _make_player_id(name: str, draft_year: int) -> str:
            slug = re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")
            return f"ROOKIE_{draft_year}_{slug}"

        scored_rows = []

        for raw in incoming:
            if not raw.get("name"):
                return jsonify({"error": "Each prospect must have a 'name'"}), 400
            if not raw.get("position"):
                return jsonify({"error": f"Prospect '{raw['name']}' is missing 'position'"}), 400

            draft_year = int(raw.get("draft_class_year") or get_active_rookie_class())
            raw["draft_class_year"] = draft_year

            if not raw.get("player_id"):
                raw["player_id"] = _make_player_id(raw["name"], draft_year)

            prospect = normalize_prospect(raw)

            # Fetch any existing mock draft consensus for this player
            consensus_map = build_mock_draft_consensus(draft_year)
            dc = consensus_map.get(prospect["player_id"])

            # Score and translate to dynasty values
            scores = score_prospect(prospect, dc)
            values = translate_score_to_value(scores, prospect, dc)

            # Build a flat row matching the shape returned by _merge_inmemory_result
            ath = prospect.get("athleticism") or {}
            row: Dict[str, Any] = {
                "player_id":                     prospect["player_id"],
                "draft_class_year":              draft_year,
                "name":                          prospect.get("name"),
                "position":                      prospect.get("position"),
                "school":                        prospect.get("school"),
                "age":                           prospect.get("age"),
                "height_inches":                 prospect.get("height_inches"),
                "weight_lbs":                    prospect.get("weight_lbs"),
                "early_declare":                 prospect.get("early_declare"),
                "transfer_history":              prospect.get("transfer_history"),
                "overall_rank":                  None,   # filled after re-sort below
                "position_rank":                 None,
                "prospect_score":                scores.get("prospect_score"),
                "rookie_value":                  values.get("rookie_value"),
                "rookie_sf_value":               values.get("rookie_sf_value"),
                "rookie_value_8":                values.get("rookie_value_8"),
                "rookie_value_12":               values.get("rookie_value_12"),
                "rookie_value_14":               values.get("rookie_value_14"),
                "rookie_sf_value_8":             values.get("rookie_sf_value_8"),
                "rookie_sf_value_12":            values.get("rookie_sf_value_12"),
                "rookie_sf_value_14":            values.get("rookie_sf_value_14"),
                "tier":                          values.get("tier"),
                "tier_label":                    values.get("tier_label"),
                "key_reasons":                   scores.get("key_reasons"),
                "production_score":              scores.get("production_score"),
                "efficiency_score":              scores.get("efficiency_score"),
                "age_score":                     scores.get("age_score"),
                "breakout_profile_score":        scores.get("breakout_profile_score"),
                "athleticism_score":             scores.get("athleticism_score"),
                "competition_score":             scores.get("competition_score"),
                "environment_adjustment":        scores.get("environment_adjustment"),
                "durability_score":              scores.get("durability_score"),
                "projected_draft_capital_score": scores.get("projected_draft_capital_score"),
                "fantasy_translation_score":     scores.get("fantasy_translation_score"),
                "confidence_score":              scores.get("confidence_score"),
                "calculated_at":                 None,
                "projected_round":               dc.get("projected_round") if dc else None,
                "projected_pick":                dc.get("projected_pick") if dc else None,
                "projected_pick_low":            dc.get("projected_pick_low") if dc else None,
                "projected_pick_high":           dc.get("projected_pick_high") if dc else None,
                "num_mocks_used":                dc.get("num_mocks_used") if dc else None,
                "consensus_confidence":          dc.get("consensus_confidence") if dc else None,
                "forty_yard":                    ath.get("forty_yard"),
                "ras_score":                     ath.get("ras_score"),
            }
            row["draft_capital_label"] = format_draft_capital(
                row["projected_round"], row["projected_pick"],
                row["projected_pick_low"], row["projected_pick_high"],
            )

            # Merge into the in-memory rankings cache for this year,
            # replacing any existing entry with the same player_id.
            current = _get_rankings(draft_year)
            current = [r for r in current if r.get("player_id") != prospect["player_id"]]
            current.append(row)

            # Re-sort by prospect_score and re-assign overall + position ranks
            current.sort(key=lambda x: x.get("prospect_score") or 0.0, reverse=True)
            pos_counters: Dict[str, int] = {}
            for i, r in enumerate(current):
                r["overall_rank"] = i + 1
                pos = (r.get("position") or "UNK").upper()
                pos_counters[pos] = pos_counters.get(pos, 0) + 1
                r["position_rank"] = pos_counters[pos]

            _cache[draft_year] = current

            # Retrieve the newly ranked row for the response
            updated = next((r for r in current if r["player_id"] == prospect["player_id"]), row)
            scored_rows.append(_row_to_dict(updated))

            # Persist to DB (best-effort — non-fatal if DB is unavailable)
            try:
                from data_building.rookie_pipeline.pipeline import upsert_prospects, upsert_rankings
                from dashboard_services.db import get_conn
                with get_conn() as conn:
                    upsert_prospects([prospect], conn)
                    upsert_rankings([scores], [values], conn)
                    conn.commit()
                log.info("[rookie_api] Persisted prospect %s to DB", prospect["player_id"])
            except Exception as db_exc:
                log.warning("[rookie_api] DB upsert skipped (DB unavailable): %s", db_exc)

        return jsonify({"added": len(scored_rows), "prospects": scored_rows})

    except Exception as exc:
        log.exception("[rookie_api] POST /prospects error")
        return jsonify({"error": str(exc)}), 500


@rookie_bp.route("/comparables/<player_id>")
def comparables(player_id: str):
    """Return historical prospects at the same position with a similar prospect score."""
    try:
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class
        from dashboard_services.db import get_conn

        year = request.args.get("year", type=int) or get_active_rookie_class()
        rows = _get_rankings(year)
        prospect = next((r for r in rows if r["player_id"] == player_id), None)

        if not prospect:
            return jsonify({"comparables": []})

        position = (prospect.get("position") or "").upper()
        score = float(prospect.get("prospect_score") or 0)
        band = 5.0  # ±16 points prospect_score

        try:
            with get_conn() as conn:
                db_rows = conn.execute(
                    """
                    SELECT player_id, name, position, draft_class_year, school,
                           prospect_score, tier, tier_label, overall_rank, position_rank,
                           actual_pick, actual_round, actual_nfl_team, headshot_url
                    FROM historical_prospect_grades
                    WHERE position = %s
                      AND prospect_score BETWEEN %s AND %s
                      AND draft_class_year < %s
                    ORDER BY ABS(prospect_score - %s) ASC, draft_class_year DESC
                    LIMIT 5
                    """,
                    (position, score - band, score + band, year, score),
                ).fetchall()

            result = []
            for r in db_rows:
                result.append({
                    "player_id":        r["player_id"],
                    "name":             r["name"],
                    "position":         r["position"],
                    "draft_class_year": r["draft_class_year"],
                    "school":           r["school"],
                    "prospect_score":   float(r["prospect_score"] or 0),
                    "tier":             r["tier"],
                    "tier_label":       r["tier_label"],
                    "overall_rank":     r["overall_rank"],
                    "position_rank":    r["position_rank"],
                    "actual_pick":      r["actual_pick"],
                    "actual_round":     r["actual_round"],
                    "actual_nfl_team":  r["actual_nfl_team"],
                    "headshot_url":     r["headshot_url"],
                })
        except Exception as db_exc:
            log.warning("[rookie_api] comparables DB error: %s", db_exc)
            result = []

        return jsonify({"comparables": result})

    except Exception as exc:
        log.exception("[rookie_api] /comparables error")
        return jsonify({"error": str(exc)}), 500


@rookie_bp.route("/by-sleeper/<sleeper_id>")
def by_sleeper(sleeper_id: str):
    """Return prospect data for a player identified by their Sleeper player ID."""
    try:
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class
        from data_building.rookie_pipeline.value_translation import format_draft_capital
        from dashboard_services.db import get_conn

        # Check active class rankings first (in-memory)
        year = get_active_rookie_class()
        for y in [year, year - 1]:
            rows = _get_rankings(y)
            row = next((r for r in rows if str(r.get("sleeper_id") or "") == str(sleeper_id)), None)
            if row:
                d = _row_to_dict(row)
                d["draft_capital_label"] = format_draft_capital(
                    d.get("projected_round"), d.get("projected_pick"),
                    d.get("projected_pick_low"), d.get("projected_pick_high"),
                )
                if d.get("headshot_url"):
                    d["espnHeadshot"] = d["headshot_url"]
                return jsonify(d)

        # Fallback: query DB directly
        try:
            with get_conn() as conn:
                row = conn.execute(
                    """
                    SELECT rp.*, rr.prospect_score, rr.tier, rr.tier_label,
                           rr.overall_rank, rr.position_rank,
                           rr.production_score, rr.efficiency_score, rr.age_score,
                           rr.breakout_profile_score, rr.athleticism_score,
                           rr.competition_score, rr.projected_draft_capital_score,
                           rr.confidence_score, rr.key_reasons,
                           rr.rookie_value, rr.rookie_sf_value,
                           rr.rookie_value_8, rr.rookie_value_12, rr.rookie_value_14,
                           rr.rookie_sf_value_8, rr.rookie_sf_value_12, rr.rookie_sf_value_14,
                           rmc.projected_round, rmc.projected_pick,
                           rmc.projected_pick_low, rmc.projected_pick_high,
                           rmc.num_mocks_used,
                           rpa.forty_yard, rpa.ras_score
                    FROM rookie_prospects rp
                    JOIN rookie_rankings rr ON rp.player_id = rr.player_id
                    LEFT JOIN rookie_mock_draft_consensus rmc ON rp.player_id = rmc.player_id
                    LEFT JOIN rookie_prospect_athleticism rpa ON rp.player_id = rpa.player_id
                    WHERE rp.sleeper_id = %s
                    ORDER BY rr.draft_class_year DESC
                    LIMIT 1
                    """,
                    (str(sleeper_id),),
                ).fetchone()

                if row:
                    d = dict(row)
                    d["draft_capital_label"] = format_draft_capital(
                        d.get("projected_round"), d.get("projected_pick"),
                        d.get("projected_pick_low"), d.get("projected_pick_high"),
                    )
                    if d.get("headshot_url"):
                        d["espnHeadshot"] = d["headshot_url"]
                    return jsonify(_row_to_dict(d))
        except Exception as db_exc:
            log.warning("[rookie_api] by-sleeper DB error: %s", db_exc)

        return jsonify({"error": "Prospect not found for sleeper_id"}), 404

    except Exception as exc:
        log.exception("[rookie_api] /by-sleeper error")
        return jsonify({"error": str(exc)}), 500


@rookie_bp.route("/link-sleeper", methods=["POST"])
def link_sleeper():
    """Link a prospect's rookie player_id to their Sleeper player ID and optionally promote to player_values."""
    try:
        body = request.json or {}
        player_id = body.get("player_id", "").strip()
        sleeper_id = body.get("sleeper_id", "").strip()

        if not player_id or not sleeper_id:
            return jsonify({"error": "player_id and sleeper_id are required"}), 400

        from dashboard_services.db import get_conn
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class
        from data_building.rookie_pipeline.value_translation import format_draft_capital

        # Update DB
        try:
            with get_conn() as conn:
                conn.execute(
                    "UPDATE rookie_prospects SET sleeper_id = %s, updated_at = NOW() WHERE player_id = %s",
                    (sleeper_id, player_id),
                )
                conn.commit()
        except Exception as db_exc:
            log.warning("[rookie_api] link-sleeper DB error: %s", db_exc)
            return jsonify({"error": f"DB update failed: {db_exc}"}), 500

        # Update in-memory cache
        year = get_active_rookie_class()
        for y in [year, year - 1]:
            rows = _get_rankings(y)
            for r in rows:
                if r.get("player_id") == player_id:
                    r["sleeper_id"] = sleeper_id
                    break

        # Optionally promote: insert into player_values so the player appears in the main system
        promote = body.get("promote", True)
        promoted = False
        if promote:
            try:
                rows = _get_rankings(year)
                row = next((r for r in rows if r["player_id"] == player_id), None)
                if not row:
                    for y in [year - 1]:
                        row = next((r for r in _get_rankings(y) if r["player_id"] == player_id), None)
                        if row:
                            break

                if row:
                    val_1qb = float(row.get("rookie_value") or 0)
                    val_sf = float(row.get("rookie_sf_value") or 0)
                    pos = row.get("position", "")
                    name = row.get("name", "")
                    pos_rank = row.get("position_rank")
                    pos_rank_label = f"{pos}{pos_rank}" if pos and pos_rank else ""

                    with get_conn() as conn:
                        conn.execute(
                            """
                            INSERT INTO player_values
                                (player_id, value_1qb, value_sf, calibrated_value_1qb, calibrated_value_sf,
                                 position, pos_rank, pos_rank_label, last_updated)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_DATE)
                            ON CONFLICT (player_id) DO UPDATE SET
                                value_1qb = EXCLUDED.value_1qb,
                                value_sf = EXCLUDED.value_sf,
                                calibrated_value_1qb = EXCLUDED.calibrated_value_1qb,
                                calibrated_value_sf = EXCLUDED.calibrated_value_sf,
                                position = EXCLUDED.position,
                                pos_rank = EXCLUDED.pos_rank,
                                pos_rank_label = EXCLUDED.pos_rank_label,
                                last_updated = EXCLUDED.last_updated
                            """,
                            (sleeper_id, val_1qb, val_sf, val_1qb, val_sf,
                             pos, pos_rank, pos_rank_label),
                        )
                        # Seed a value history row so the chart has at least one point
                        conn.execute(
                            """
                            INSERT INTO player_value_history
                                (as_of_date, player_id, name, position, value, source)
                            VALUES (CURRENT_DATE, %s, %s, %s, %s, 'model')
                            ON CONFLICT (as_of_date, player_id, source) DO UPDATE SET
                                value = EXCLUDED.value
                            """,
                            (sleeper_id, name, pos, val_1qb),
                        )
                        conn.commit()
                    promoted = True
            except Exception as prom_exc:
                log.warning("[rookie_api] link-sleeper promote error: %s", prom_exc)

        return jsonify({"ok": True, "player_id": player_id, "sleeper_id": sleeper_id, "promoted": promoted})

    except Exception as exc:
        log.exception("[rookie_api] /link-sleeper error")
        return jsonify({"error": str(exc)}), 500


@rookie_bp.route("/auto-link/<player_id>")
def auto_link(player_id: str):
    """Auto-match a prospect to their Sleeper ID via players_index.json name lookup, then promote."""
    try:
        from utils.utils import load_players_index
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class
        from dashboard_services.db import get_conn

        year = get_active_rookie_class()
        row = None
        for y in [year, year - 1]:
            row = next((r for r in _get_rankings(y) if r.get("player_id") == player_id), None)
            if row:
                break

        if not row:
            return jsonify({"ok": False, "error": "Prospect not found"}), 404

        if row.get("sleeper_id"):
            return jsonify({"ok": True, "sleeper_id": row["sleeper_id"], "already_linked": True})

        prospect_name = row.get("name", "")
        if not prospect_name:
            return jsonify({"ok": False, "error": "Prospect has no name"}), 400

        players_index = load_players_index() or {}

        def _norm(n: str) -> str:
            n = n.lower()
            n = re.sub(r"['\.\-]", "", n)
            n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
            return re.sub(r"\s+", " ", n).strip()

        norm_prospect = _norm(prospect_name)
        sleeper_id = None
        for sid, pdata in players_index.items():
            if _norm(pdata.get("name", "")) == norm_prospect:
                sleeper_id = sid
                break

        if not sleeper_id:
            return jsonify({"ok": False, "error": f"No match for '{prospect_name}'"})

        try:
            with get_conn() as conn:
                conn.execute(
                    "UPDATE rookie_prospects SET sleeper_id = %s, updated_at = NOW() WHERE player_id = %s",
                    (sleeper_id, player_id),
                )
                conn.commit()
        except Exception as db_exc:
            log.warning("[rookie_api] auto-link DB error: %s", db_exc)
            return jsonify({"error": f"DB update failed: {db_exc}"}), 500

        for y in [year, year - 1]:
            for r in _get_rankings(y):
                if r.get("player_id") == player_id:
                    r["sleeper_id"] = sleeper_id
                    break

        promoted = False
        try:
            val_1qb = float(row.get("rookie_value") or 0)
            val_sf = float(row.get("rookie_sf_value") or 0)
            pos = row.get("position", "")
            name = row.get("name", "")
            pos_rank = row.get("position_rank")
            pos_rank_label = f"{pos}{pos_rank}" if pos and pos_rank else ""

            with get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO player_values
                        (player_id, value_1qb, value_sf, calibrated_value_1qb, calibrated_value_sf,
                         position, pos_rank, pos_rank_label, last_updated)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_DATE)
                    ON CONFLICT (player_id) DO UPDATE SET
                        value_1qb = EXCLUDED.value_1qb,
                        value_sf = EXCLUDED.value_sf,
                        calibrated_value_1qb = EXCLUDED.calibrated_value_1qb,
                        calibrated_value_sf = EXCLUDED.calibrated_value_sf,
                        position = EXCLUDED.position,
                        pos_rank = EXCLUDED.pos_rank,
                        pos_rank_label = EXCLUDED.pos_rank_label,
                        last_updated = EXCLUDED.last_updated
                    """,
                    (sleeper_id, val_1qb, val_sf, val_1qb, val_sf, pos, pos_rank, pos_rank_label),
                )
                conn.execute(
                    """
                    INSERT INTO player_value_history
                        (as_of_date, player_id, name, position, value, source)
                    VALUES (CURRENT_DATE, %s, %s, %s, %s, 'model')
                    ON CONFLICT (as_of_date, player_id, source) DO UPDATE SET
                        value = EXCLUDED.value
                    """,
                    (sleeper_id, name, pos, val_1qb),
                )
                conn.commit()
            promoted = True
        except Exception as prom_exc:
            log.warning("[rookie_api] auto-link promote error: %s", prom_exc)

        return jsonify({"ok": True, "sleeper_id": sleeper_id, "already_linked": False, "promoted": promoted})

    except Exception as exc:
        log.exception("[rookie_api] /auto-link error")
        return jsonify({"error": str(exc)}), 500


@rookie_bp.route("/draft-status", methods=["GET"])
def draft_status():
    """Check if the draft is complete for a given year."""
    try:
        import datetime as _dt
        year = request.args.get("year", type=int)
        if year is None:
            from data_building.rookie_pipeline.pipeline import get_active_rookie_class
            year = get_active_rookie_class()

        from data_building.rookie_pipeline.pipeline import is_draft_complete
        from dashboard_services.db import get_conn

        draft_date = None
        with get_conn() as conn:
            draft_complete = is_draft_complete(year, conn)
            try:
                row = conn.execute(
                    "SELECT draft_date FROM rookie_active_class WHERE draft_class_year = %s",
                    (year,),
                ).fetchone()
                if row and row["draft_date"]:
                    d = row["draft_date"]
                    if isinstance(d, str):
                        d = _dt.datetime.strptime(d[:10], "%Y-%m-%d").date()
                    draft_date = d
            except Exception:
                pass

        if draft_date is None:
            draft_date = _dt.date(year, 4, 26)  # typical end-of-draft fallback

        today = _dt.date.today()
        days_since = (today - draft_date).days if draft_complete else None

        return jsonify({
            "draft_year":       year,
            "draft_complete":   draft_complete,
            "days_since_draft": days_since,
        })
    except Exception as exc:
        log.exception("[rookie_api] /draft-status error")
        return jsonify({"error": str(exc)}), 500


@rookie_bp.route("/refresh", methods=["POST"])
def refresh():
    """Re-run the pipeline and bust the in-memory cache."""
    try:
        from data_building.rookie_pipeline.pipeline import (
            get_active_rookie_class, run_rookie_pipeline,
        )
        year = request.json.get("year") if request.json else None
        if year is None:
            year = get_active_rookie_class()
        year = int(year)

        _cache.pop(year, None)
        run_rookie_pipeline(year)
        _cache.pop(year, None)  # force fresh DB read on next request

        return jsonify({"status": "ok", "draft_class_year": year})
    except Exception as exc:
        log.exception("[rookie_api] /refresh error")
        return jsonify({"error": str(exc)}), 500


def register_rookie_routes(app):
    app.register_blueprint(rookie_bp)
    return app
