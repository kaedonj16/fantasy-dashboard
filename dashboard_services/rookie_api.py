"""
Rookie prospect API blueprint.

Endpoints:
    GET  /api/rookies/rankings?year=2026&pos=WR&league_type=1qb
    GET  /api/rookies/player/<player_id>
    GET  /api/rookies/active-class
    POST /api/rookies/prospects        (add/update one or more prospects)
    POST /api/rookies/refresh          (triggers pipeline re-run)
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from flask import Blueprint, jsonify, request

log = logging.getLogger(__name__)

rookie_bp = Blueprint("rookies", __name__, url_prefix="/api/rookies")

# In-memory cache so we don't re-run the pipeline on every page load.
# Invalidated on refresh or on first hit per process.
_cache: Dict[int, List[Dict[str, Any]]] = {}


def _get_rankings(draft_year: int) -> List[Dict[str, Any]]:
    if draft_year not in _cache:
        from data_building.rookie_pipeline.pipeline import get_rookie_rankings_from_db
        _cache[draft_year] = get_rookie_rankings_from_db(draft_year)
    return _cache[draft_year]


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

        # Position filter
        if pos:
            rows = [r for r in rows if (r.get("position") or "").upper() == pos]

        # Limit to top 60
        rows = rows[:60]

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
            result.append(d)

        # Sort: tier ascending, then display_value descending within each tier
        result.sort(key=lambda x: (x.get("tier") or 99, -(x.get("display_value") or 0)))

        return jsonify({"draft_class_year": year, "count": len(result), "rankings": result})

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
        return jsonify(_row_to_dict(row))
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
