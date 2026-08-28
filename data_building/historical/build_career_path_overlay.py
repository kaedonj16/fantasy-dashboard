"""Build cache/player_history/career_path_overlay.json from usage_rows.

Stdlib + committed JSON only. Does not rewrite historical_profile_aggregates.json.
Does not import pandas. Request paths must not call this — cron / CLI rebuild.
"""
from __future__ import annotations

import json
from typing import Any

from dashboard_services.historical.aggregates_store import (
    CAREER_PATH_OVERLAY_PATH,
    PROFILE_PATH,
)
from dashboard_services.historical.career_profiles import build_career_path_overlay
from dashboard_services.historical.definitions import DRAFT_CAPITAL_ORDER
from dashboard_services.historical.finishes import (
    assign_all_scoring_finishes,
    attach_prior_career_features,
)
from dashboard_services.historical.seasons import (
    canonicalize_usage_row,
    identity_from_players_index_entry,
    row_appeared,
)
from utils.paths import CACHE_DIR, PLAYER_HISTORY_DIR

DEFAULT_SEASONS = tuple(range(2018, 2026))


def _load_usage_rows(season: int) -> list[dict]:
    path = PLAYER_HISTORY_DIR / f"usage_rows_{season}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _identity_map() -> dict[str, dict]:
    path = CACHE_DIR / "players_index.json"
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        index = {}
    if not isinstance(index, dict):
        return {}
    out: dict[str, dict] = {}
    for pid, meta in index.items():
        out[str(pid)] = identity_from_players_index_entry(meta or {})
    return out


def _canonicalize_season(season: int, identity_map: dict[str, dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for raw in _load_usage_rows(season):
        pid = str(raw.get("sleeper_id") or raw.get("player_id") or raw.get("id") or "")
        row = canonicalize_usage_row(raw, season, identity_map.get(pid) or {})
        if row is None:
            continue
        key = row["sleeper_id"]
        if key in seen or not row_appeared(row):
            continue
        seen.add(key)
        out.append(row)
    return out


def _stamp_capital_from_profiles(rows: list[dict], by_player: dict) -> None:
    """Draft capital is career-constant; copy it from live preseason profiles."""
    if not isinstance(by_player, dict):
        return
    for row in rows:
        if row.get("draft_capital_bucket") in DRAFT_CAPITAL_ORDER:
            continue
        pid = str(row.get("sleeper_id") or "")
        cap = (by_player.get(pid) or {}).get("draft_capital_bucket")
        if cap in DRAFT_CAPITAL_ORDER:
            row["draft_capital_bucket"] = cap


def rebuild_career_path_overlay(*, write: bool = True) -> dict[str, Any]:
    identity_map = _identity_map()
    combined: list[dict] = []
    for season in DEFAULT_SEASONS:
        combined.extend(assign_all_scoring_finishes(_canonicalize_season(season, identity_map)))
    featured = attach_prior_career_features(combined)
    try:
        aggs = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        aggs = {}
    by_player = ((aggs.get("preseason_profiles") or {}).get("by_player") or {})
    _stamp_capital_from_profiles(featured, by_player)
    overlay = build_career_path_overlay(featured)
    if write:
        CAREER_PATH_OVERLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
        CAREER_PATH_OVERLAY_PATH.write_text(
            json.dumps(overlay, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return overlay


if __name__ == "__main__":
    payload = rebuild_career_path_overlay(write=True)
    counts = payload.get("prior_top12_count") or {}
    bounce = payload.get("bounce_back") or {}
    wr_n = (bounce.get("WR") or {}).get("n_bounce_back")
    print(
        "wrote "
        + str(CAREER_PATH_OVERLAY_PATH)
        + " n_players="
        + str(payload.get("n_players"))
        + " btj="
        + str(counts.get("11631"))
        + " wr_bounce_n="
        + str(wr_n)
    )
