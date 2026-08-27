"""I/O: cache NGS/snap overlays and join them onto warehouse rows.

Request paths never call this. Rebuild reads committed JSON under
``cache/player_history/`` so cron does not need ``nfl_data_py``.

Cache refresh (optional, needs network):

    python -m data_building.historical.build_usage_efficiency
"""
from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path
from typing import Any, Optional

from utils.paths import PLAYER_HISTORY_DIR

from dashboard_services.historical.usage import apply_efficiency_overlay, normalize_snap_pct

NGS_RECEIVING_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "nextgen_stats/ngs_receiving.parquet"
)
NGS_PASSING_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "nextgen_stats/ngs_passing.parquet"
)
NGS_RUSHING_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "nextgen_stats/ngs_rushing.parquet"
)
SNAP_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "snap_counts/snap_counts_{season}.parquet"
)
IDS_URL = "https://github.com/dynastyprocess/data/raw/master/files/db_playerids.csv"

DEFAULT_SEASONS = tuple(range(2018, 2026))


def nflverse_cache_path(season: int) -> Path:
    return PLAYER_HISTORY_DIR / f"nflverse_metrics_{season}.json"


def snap_cache_path(season: int) -> Path:
    return PLAYER_HISTORY_DIR / f"snap_counts_{season}.json"


def _f(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or math.isinf(number):
        return None
    return number


def _sleeper_id(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    try:
        text = str(int(float(value)))
    except (TypeError, ValueError):
        text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return None
    return text


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "fantasy-dashboard-historical"})
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as out:
        out.write(resp.read())
    return dest


def _load_id_crosswalk() -> tuple[dict[str, str], dict[str, str]]:
    """gsis→sleeper and pfr→sleeper from DynastyProcess ids (same file nfl_data_py uses)."""
    import pandas as pd

    dest = Path("/tmp/db_playerids.csv")
    if not dest.exists():
        _download(IDS_URL, dest)
    df = pd.read_csv(dest)
    gsis_map: dict[str, str] = {}
    pfr_map: dict[str, str] = {}
    for _, row in df.iterrows():
        sleeper = _sleeper_id(row.get("sleeper_id"))
        if not sleeper:
            continue
        gsis = str(row.get("gsis_id") or "").strip()
        if gsis and gsis.lower() != "nan":
            gsis_map.setdefault(gsis, sleeper)
        pfr = str(row.get("pfr_id") or "").strip()
        if pfr and pfr.lower() != "nan":
            pfr_map.setdefault(pfr, sleeper)
    return gsis_map, pfr_map


def _ngs_from_frame(df, gsis_map: dict[str, str], mapping: tuple[tuple[str, str], ...], season: int) -> dict[str, dict]:
    import pandas as pd

    if df is None or getattr(df, "empty", True):
        return {}
    work = df[(df["season"] == season) & (df.get("season_type", "REG") == "REG")]
    if "week" in work.columns:
        work = work[work["week"] == 0]
    out: dict[str, dict] = {}
    for _, row in work.iterrows():
        gsis = str(row.get("player_gsis_id") or "").strip()
        pid = gsis_map.get(gsis)
        if not pid:
            continue
        cols: dict[str, float] = {}
        for src, dst in mapping:
            val = _f(row.get(src))
            if val is not None:
                cols[dst] = val
        if cols:
            out.setdefault(pid, {}).update(cols)
    return out


def build_ngs_cache_for_season(
    season: int,
    gsis_map: dict[str, str],
    *,
    receiving_df=None,
    passing_df=None,
    rushing_df=None,
) -> dict[str, dict]:
    """Same column names as ``nflverse_metrics`` NGS builders."""
    combined: dict[str, dict] = {}
    rec = _ngs_from_frame(
        receiving_df,
        gsis_map,
        (
            ("avg_separation", "ngs_avg_separation"),
            ("avg_cushion", "ngs_avg_cushion"),
            ("avg_intended_air_yards", "ngs_avg_intended_air_yards"),
            ("percent_share_of_intended_air_yards", "ngs_pct_share_intended_air_yards"),
            ("avg_yac", "ngs_avg_yac"),
            ("avg_expected_yac", "ngs_avg_expected_yac"),
            ("avg_yac_above_expectation", "ngs_avg_yac_above_expectation"),
            ("catch_percentage", "ngs_catch_pct"),
        ),
        season,
    )
    for pid, cols in rec.items():
        if "ngs_avg_intended_air_yards" in cols:
            cols["adot"] = cols["ngs_avg_intended_air_yards"]
            cols["avg_depth_of_target"] = cols["ngs_avg_intended_air_yards"]
        sep, cushion = cols.get("ngs_avg_separation"), cols.get("ngs_avg_cushion")
        if sep is not None and cushion is not None:
            cols["ngs_created_separation"] = round(float(sep) - float(cushion), 2)
        combined.setdefault(pid, {}).update(cols)
    pas = _ngs_from_frame(
        passing_df,
        gsis_map,
        (
            ("avg_time_to_throw", "ngs_avg_time_to_throw"),
            ("aggressiveness", "ngs_aggressiveness"),
            ("avg_completed_air_yards", "ngs_avg_completed_air_yards"),
            ("completion_percentage_above_expectation", "ngs_cpoe"),
        ),
        season,
    )
    for pid, cols in pas.items():
        combined.setdefault(pid, {}).update(cols)
    rush = _ngs_from_frame(
        rushing_df,
        gsis_map,
        (
            ("rush_yards_over_expected_per_att", "ngs_rush_yards_over_expected_per_att"),
            ("efficiency", "ngs_rush_efficiency"),
            ("avg_time_to_los", "ngs_avg_time_to_los"),
            ("percent_attempts_gte_eight_defenders", "ngs_percent_attempts_gte_eight_defenders"),
        ),
        season,
    )
    for pid, cols in rush.items():
        combined.setdefault(pid, {}).update(cols)
    return combined


def build_snap_cache_for_season(season: int, pfr_map: dict[str, str], snap_df=None) -> dict[str, dict]:
    """sleeper_id → {snap_pct, snaps}. Missing stays absent, never a fake 0."""
    if snap_df is None or getattr(snap_df, "empty", True):
        return {}
    work = snap_df
    if "game_type" in work.columns:
        work = work[work["game_type"] == "REG"]
    if "season" in work.columns:
        work = work[work["season"] == season]
    out: dict[str, dict] = {}
    if "pfr_player_id" not in work.columns:
        return {}
    for pfr_id, grp in work.groupby("pfr_player_id"):
        pid = pfr_map.get(str(pfr_id))
        if not pid:
            continue
        off = grp["offense_snaps"] if "offense_snaps" in grp.columns else None
        pct = grp["offense_pct"] if "offense_pct" in grp.columns else None
        total_snaps = float(off.sum()) if off is not None else None
        mean_pct = float(pct.mean()) if pct is not None and len(pct) else None
        snap_pct = normalize_snap_pct(mean_pct)
        cols = {}
        if snap_pct is not None:
            cols["snap_pct"] = snap_pct
        if total_snaps is not None and total_snaps > 0:
            cols["snaps"] = round(total_snaps, 1)
        if cols:
            out[pid] = cols
    return out


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def refresh_efficiency_cache(seasons: tuple[int, ...] = DEFAULT_SEASONS) -> dict:
    """Download NGS + snap parquets, write per-season JSON keyed by sleeper_id."""
    import pandas as pd

    gsis_map, pfr_map = _load_id_crosswalk()
    tmp = Path("/tmp/historical_nflverse")
    tmp.mkdir(parents=True, exist_ok=True)
    rec = pd.read_parquet(_download(NGS_RECEIVING_URL, tmp / "ngs_receiving.parquet"))
    pas = pd.read_parquet(_download(NGS_PASSING_URL, tmp / "ngs_passing.parquet"))
    rush = pd.read_parquet(_download(NGS_RUSHING_URL, tmp / "ngs_rushing.parquet"))
    written = {"nflverse": [], "snaps": []}
    for season in seasons:
        ngs = build_ngs_cache_for_season(
            season, gsis_map, receiving_df=rec, passing_df=pas, rushing_df=rush
        )
        _write_json(nflverse_cache_path(season), ngs)
        written["nflverse"].append({"season": season, "players": len(ngs)})
        snap_path = tmp / f"snap_counts_{season}.parquet"
        try:
            _download(SNAP_URL.format(season=season), snap_path)
            snaps = build_snap_cache_for_season(season, pfr_map, pd.read_parquet(snap_path))
        except Exception as exc:
            print(f"[historical] snap cache {season} skipped ({exc})")
            snaps = {}
        _write_json(snap_cache_path(season), snaps)
        written["snaps"].append({"season": season, "players": len(snaps)})
        print(f"[historical] efficiency cache {season}: ngs={len(ngs)} snaps={len(snaps)}")
    return written


def load_json_map(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return {str(k): (v if isinstance(v, dict) else {}) for k, v in data.items()}


def load_efficiency_overlay(season: int) -> dict[str, dict]:
    """sleeper_id → overlay dict. Empty when cache files are absent."""
    merged: dict[str, dict] = {}
    for part in (load_json_map(nflverse_cache_path(season)), load_json_map(snap_cache_path(season))):
        for pid, cols in part.items():
            merged.setdefault(pid, {}).update(cols)
    return merged


def overlay_season_rows(rows: list[dict], overlay: Optional[dict[str, dict]] = None) -> list[dict]:
    overlay = overlay if overlay is not None else (
        load_efficiency_overlay(int(rows[0]["season"])) if rows else {}
    )
    return [
        apply_efficiency_overlay(row, overlay.get(str(row.get("sleeper_id") or "")))
        for row in rows
    ]


if __name__ == "__main__":
    refresh_efficiency_cache()
