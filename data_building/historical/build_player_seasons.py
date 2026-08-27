"""I/O: build the canonical historical player-season warehouse.

Reads committed ``cache/player_history/usage_rows_{season}.json`` (both
legacy-totals and sleeper-averages schemas), joins identity from
``players_index`` + ``draft_history.parquet``, writes parquet. Request
paths must not call this — it is a cron / CLI rebuild.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from utils.paths import PLAYER_HISTORY_DIR, PLAYER_INVESTMENT_DIR

from dashboard_services.historical.finishes import (
    assign_all_scoring_finishes,
    attach_prior_career_features,
)
from dashboard_services.historical.seasons import (
    canonicalize_usage_row,
    coverage_counts,
    identity_from_players_index_entry,
    row_appeared,
)
from data_building.external_data.player_history import (
    load_player_history_df,
    save_combined_player_history_df,
    save_player_history_df,
    usage_rows_json_path_for_season,
)

DEFAULT_SEASONS = tuple(range(2018, 2026))
COVERAGE_PATH = PLAYER_HISTORY_DIR / "historical_coverage.json"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_usage_rows(season: int) -> list[dict]:
    path = usage_rows_json_path_for_season(season)
    if not path.exists():
        return []
    data = _load_json(path)
    return data if isinstance(data, list) else []


def load_players_index() -> dict:
    from utils.utils import load_players_index as _load
    return _load() or {}


def gsis_crosswalk_from_legacy_usage(seasons: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022)) -> dict[str, str]:
    """sleeper_id → gsis_id from committed 2018–2022 usage_rows (no nflverse)."""
    mapping: dict[str, str] = {}
    for season in seasons:
        for raw in load_usage_rows(season):
            pid = str(raw.get("id") or raw.get("sleeper_id") or "")
            usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
            gsis = usage.get("gsis_id") or raw.get("gsis_id")
            if pid and gsis and str(gsis).strip() and pid not in mapping:
                mapping[pid] = str(gsis).strip()
    return mapping


def load_draft_identity() -> dict[str, dict]:
    """sleeper_id → draft year/round/pick from committed draft_history.parquet."""
    path = PLAYER_INVESTMENT_DIR / "draft_history.parquet"
    if not path.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_parquet(path)
    except Exception as exc:
        print(f"[historical] draft_history parquet unreadable ({exc})")
        return {}
    out: dict[str, dict] = {}
    if df is None or df.empty or "sleeper_id" not in df.columns:
        return {}
    for _, row in df.iterrows():
        sid = str(row.get("sleeper_id") or "").strip()
        if not sid:
            continue
        undrafted = False
        rnd = row.get("draft_round")
        try:
            if rnd is not None and float(rnd) == 0:
                undrafted = True
        except (TypeError, ValueError):
            pass
        out[sid] = {
            "draft_year": None if _isna(row.get("draft_year")) else row.get("draft_year"),
            "nfl_draft_round": None if _isna(row.get("draft_round")) else row.get("draft_round"),
            "nfl_draft_pick": None if _isna(row.get("draft_pick")) else row.get("draft_pick"),
            "undrafted": undrafted,
        }
    return out


def _isna(value: Any) -> bool:
    if value is None:
        return True
    try:
        import pandas as pd
        return bool(pd.isna(value))
    except Exception:
        return False


def build_identity_map(
    players_index: Optional[dict] = None,
    draft_identity: Optional[dict] = None,
    gsis_map: Optional[dict] = None,
) -> dict[str, dict]:
    players_index = players_index if players_index is not None else load_players_index()
    draft_identity = draft_identity if draft_identity is not None else load_draft_identity()
    gsis_map = gsis_map if gsis_map is not None else gsis_crosswalk_from_legacy_usage()
    out: dict[str, dict] = {}
    for pid, meta in (players_index or {}).items():
        ident = identity_from_players_index_entry(meta or {})
        extra = draft_identity.get(str(pid)) or {}
        if extra.get("draft_year") is not None:
            ident["draft_year"] = extra["draft_year"]
        ident["nfl_draft_round"] = extra.get("nfl_draft_round")
        ident["nfl_draft_pick"] = extra.get("nfl_draft_pick")
        ident["undrafted"] = extra.get("undrafted", False)
        gsis = gsis_map.get(str(pid))
        if gsis:
            ident["gsis_id"] = gsis
        out[str(pid)] = ident
    # Players who appear in usage but not in the current players_index
    # (retired) still get gsis / draft when the maps have them.
    for pid, gsis in gsis_map.items():
        ident = out.setdefault(pid, {})
        ident.setdefault("gsis_id", gsis)
        extra = draft_identity.get(pid) or {}
        for key in ("draft_year", "nfl_draft_round", "nfl_draft_pick", "undrafted"):
            if extra.get(key) is not None and ident.get(key) is None:
                ident[key] = extra[key]
    return out


def canonicalize_season(
    season: int,
    usage_rows: Optional[list] = None,
    identity_map: Optional[dict] = None,
) -> list[dict]:
    rows = usage_rows if usage_rows is not None else load_usage_rows(season)
    ident = identity_map if identity_map is not None else build_identity_map()
    out = []
    seen = set()
    for raw in rows:
        pid = str(raw.get("sleeper_id") or raw.get("player_id") or raw.get("id") or "")
        row = canonicalize_usage_row(raw, season, ident.get(pid) or {})
        if row is None:
            continue
        key = row["sleeper_id"]
        if key in seen:
            continue
        if not row_appeared(row):
            continue
        seen.add(key)
        out.append(row)
    return out


def add_finishes(rows: list[dict]) -> list[dict]:
    return assign_all_scoring_finishes(rows)


def rebuild_historical_warehouse(
    seasons: tuple[int, ...] = DEFAULT_SEASONS,
    *,
    write: bool = True,
) -> dict:
    """Canonicalize → positional finishes → prior-career features → parquet.

    ``write=False`` is for tests. Coverage is always returned.
    """
    import pandas as pd

    identity_map = build_identity_map()
    by_season: dict[int, list[dict]] = {}
    coverage: dict[str, Any] = {"seasons": {}, "identity": {"n": len(identity_map)}}

    for season in seasons:
        raw = load_usage_rows(season)
        rows = canonicalize_season(season, raw, identity_map)
        rows = add_finishes(rows)
        by_season[season] = rows
        coverage["seasons"][str(season)] = {
            "usage_rows": len(raw),
            "canonical_rows": len(rows),
            "fields": coverage_counts(rows)["fields"],
        }

    combined = []
    for season in seasons:
        combined.extend(by_season.get(season) or [])
    featured = attach_prior_career_features(combined)

    featured_by_season: dict[int, list[dict]] = {s: [] for s in seasons}
    for row in featured:
        featured_by_season.setdefault(int(row["season"]), []).append(row)

    saved = []
    if write:
        frames = []
        for season in seasons:
            season_rows = featured_by_season.get(season) or []
            if not season_rows:
                continue
            df = pd.DataFrame(season_rows)
            save_player_history_df(df, season)
            frames.append(df)
            saved.append(season)
        if frames:
            all_df = pd.concat(frames, ignore_index=True)
            save_combined_player_history_df(all_df)
        coverage["written_seasons"] = saved
        coverage["combined_rows"] = len(featured)
        COVERAGE_PATH.write_text(json.dumps(coverage, indent=2, default=str), encoding="utf-8")
        print(f"[historical] wrote seasons {saved}; coverage → {COVERAGE_PATH}")
    else:
        coverage["written_seasons"] = []
        coverage["combined_rows"] = len(featured)

    coverage["rows"] = featured
    return coverage


def load_canonical_history(season: Optional[int] = None):
    """Request-path reader: precomputed parquet only, never scans usage JSON."""
    return load_player_history_df(season)


if __name__ == "__main__":
    rebuild_historical_warehouse()
