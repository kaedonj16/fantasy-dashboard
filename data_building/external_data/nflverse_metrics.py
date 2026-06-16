"""
Build redistributable advanced receiving metrics from the nflverse ecosystem:

  - NGS (Next Gen Stats) receiving tracking metrics via nfl_data_py.import_ngs_data
      avg_separation, avg_cushion, avg_intended_air_yards,
      pct_share_of_intended_air_yards, avg_yac, avg_expected_yac,
      avg_yac_above_expectation, catch_percentage
  - FTN charting (nflverse subset, 2022+) via nfl_data_py.import_ftn_data joined to
    play-by-play, to derive:
      drop_rate              = drops / catchable targets   (percent)
      contested_catch_rate   = contested receptions / contested targets (percent)

Unlike PFF/SumerSports, these sources are redistributable, so the derived
columns are safe to display publicly. They populate the same drop_rate /
contested_catch_rate / avg_depth_of_target columns the PFF importer uses (plus
new ngs_* columns), and the season reader coalesces all snapshot rows so these
merge with the computed + PFF snapshots for the same season.

nfl_data_py is an optional dependency; every builder degrades to {} if it (or the
underlying data) is unavailable, mirroring pfr_snap_counts.py.
"""

from __future__ import annotations

from typing import Dict, Optional

# NGS / FTN player-tracking data only exists from these seasons on.
NGS_FLOOR = 2016
FTN_FLOOR = 2022

_GSIS_TO_SLEEPER: Optional[Dict[str, str]] = None


def _gsis_to_sleeper() -> Dict[str, str]:
    """Crosswalk nflverse gsis_id -> Sleeper id using nfl_data_py.import_ids().

    Cached for the life of the process. Returns {} if unavailable.
    """
    global _GSIS_TO_SLEEPER
    if _GSIS_TO_SLEEPER is not None:
        return _GSIS_TO_SLEEPER

    mapping: Dict[str, str] = {}
    try:
        import nfl_data_py as nfl  # optional dependency
        ids = nfl.import_ids()
        for _, row in ids.iterrows():
            gsis = row.get("gsis_id")
            sleeper = row.get("sleeper_id")
            if gsis is None or sleeper is None:
                continue
            gsis = str(gsis).strip()
            # sleeper_id often arrives as a float (e.g. 4034.0); normalise.
            try:
                sleeper = str(int(float(sleeper)))
            except (ValueError, TypeError):
                sleeper = str(sleeper).strip()
            if gsis and sleeper and sleeper.lower() != "nan":
                mapping[gsis] = sleeper
    except Exception as e:
        print(f"[nflverse_metrics] id crosswalk unavailable ({e})")

    _GSIS_TO_SLEEPER = mapping
    return mapping


def _f(v) -> Optional[float]:
    """Coerce to float, treating NaN/None/blank as None."""
    if v is None:
        return None
    try:
        f = float(v)
    except (ValueError, TypeError):
        return None
    if f != f:  # NaN
        return None
    return f


def build_ngs_receiving_for_season(season: int) -> Dict[str, Dict[str, float]]:
    """Return {sleeper_id: {ngs columns}} for a season's NGS receiving data.

    Uses the NGS season-summary rows (week == 0). Also mirrors NGS intended air
    yards into avg_depth_of_target so the existing aDOT tile renders without PFF.
    """
    if season < NGS_FLOOR:
        print(f"[nflverse_metrics] {season} < NGS floor ({NGS_FLOOR}); skipping NGS")
        return {}

    try:
        import nfl_data_py as nfl
        df = nfl.import_ngs_data(stat_type="receiving", years=[season])
    except Exception as e:
        print(f"[nflverse_metrics] NGS receiving unavailable for {season} ({e})")
        return {}

    if df is None or df.empty:
        return {}

    # Keep only the regular-season summary rows (week 0 = season aggregate).
    df = df[(df["season_type"] == "REG") & (df["week"] == 0)]
    crosswalk = _gsis_to_sleeper()

    out: Dict[str, Dict[str, float]] = {}
    for _, r in df.iterrows():
        gsis = str(r.get("player_gsis_id") or "").strip()
        pid = crosswalk.get(gsis)
        if not pid:
            continue

        intended_ay = _f(r.get("avg_intended_air_yards"))
        row: Dict[str, float] = {}
        for src, dst in (
            ("avg_separation", "ngs_avg_separation"),
            ("avg_cushion", "ngs_avg_cushion"),
            ("avg_intended_air_yards", "ngs_avg_intended_air_yards"),
            ("percent_share_of_intended_air_yards", "ngs_pct_share_intended_air_yards"),
            ("avg_yac", "ngs_avg_yac"),
            ("avg_expected_yac", "ngs_avg_expected_yac"),
            ("avg_yac_above_expectation", "ngs_avg_yac_above_expectation"),
            ("catch_percentage", "ngs_catch_pct"),
        ):
            val = _f(r.get(src))
            if val is not None:
                row[dst] = val

        # Fill the shared aDOT column from NGS when present (yards scale matches).
        if intended_ay is not None:
            row["avg_depth_of_target"] = intended_ay

        if row:
            out[pid] = row
    return out


def build_ftn_charting_for_season(season: int) -> Dict[str, Dict[str, float]]:
    """Return {sleeper_id: {drop_rate, contested_catch_rate}} from FTN charting.

    FTN charting is play-level with no player id, so we join it to play-by-play
    on (game_id, play_id) to attribute each charted flag to the targeted
    receiver. Rates are percentages to match the existing column scale.
    """
    if season < FTN_FLOOR:
        print(f"[nflverse_metrics] {season} < FTN floor ({FTN_FLOOR}); skipping FTN")
        return {}

    try:
        import nfl_data_py as nfl
        ftn = nfl.import_ftn_data([season])
        pbp = nfl.import_pbp_data(
            [season],
            columns=[
                "game_id", "play_id", "season_type",
                "receiver_player_id", "complete_pass", "pass_attempt",
            ],
            downcast=True,
        )
    except Exception as e:
        print(f"[nflverse_metrics] FTN/pbp unavailable for {season} ({e})")
        return {}

    if ftn is None or ftn.empty or pbp is None or pbp.empty:
        return {}

    try:
        merged = ftn.merge(
            pbp,
            left_on=["nflverse_game_id", "nflverse_play_id"],
            right_on=["game_id", "play_id"],
            how="inner",
        )
    except Exception as e:
        print(f"[nflverse_metrics] FTN/pbp merge failed for {season} ({e})")
        return {}

    merged = merged[merged["season_type"] == "REG"]
    merged = merged[merged["receiver_player_id"].notna()]

    crosswalk = _gsis_to_sleeper()

    # Aggregate per targeted receiver (gsis id).
    agg: Dict[str, Dict[str, float]] = {}
    for _, r in merged.iterrows():
        gsis = str(r.get("receiver_player_id") or "").strip()
        if not gsis:
            continue
        bucket = agg.setdefault(gsis, {"catchable": 0.0, "drops": 0.0,
                                       "contested": 0.0, "contested_caught": 0.0})
        catchable = _f(r.get("is_catchable_ball")) or 0.0
        drop = _f(r.get("is_drop")) or 0.0
        contested = _f(r.get("is_contested_ball")) or 0.0
        complete = _f(r.get("complete_pass")) or 0.0
        bucket["catchable"] += catchable
        bucket["drops"] += drop
        bucket["contested"] += contested
        if contested and complete:
            bucket["contested_caught"] += 1.0

    out: Dict[str, Dict[str, float]] = {}
    for gsis, b in agg.items():
        pid = crosswalk.get(gsis)
        if not pid:
            continue
        row: Dict[str, float] = {}
        if b["catchable"] > 0:
            row["drop_rate"] = round(b["drops"] / b["catchable"] * 100.0, 1)
        if b["contested"] > 0:
            row["contested_catch_rate"] = round(
                b["contested_caught"] / b["contested"] * 100.0, 1)
        if row:
            out[pid] = row
    return out


def build_pbp_metrics_for_season(season: int) -> Dict[str, Dict[str, float]]:
    """Return {sleeper_id: {...}} of EPA-family + breakaway/explosive metrics.

    All derived from open nflverse play-by-play, so fully public-safe:
      QB:      passing_epa, epa_per_play, cpoe, sack_rate, scramble_rate, success_rate
      Rusher:  rushing_epa, breakaway_percentage, explosive_runs_10_plus
      Receiver: receiving_epa
    Rates are stored as percentages to match the existing column conventions.
    """
    try:
        import nfl_data_py as nfl
        pbp = nfl.import_pbp_data(
            [season],
            columns=[
                "game_id", "play_id", "season_type", "play_type",
                "epa", "qb_epa", "cpoe", "success",
                "sack", "qb_scramble", "rush_attempt", "pass_attempt",
                "qb_dropback", "rushing_yards",
                "passer_player_id", "rusher_player_id", "receiver_player_id",
            ],
            downcast=True,
        )
    except Exception as e:
        print(f"[nflverse_metrics] pbp unavailable for {season} ({e})")
        return {}

    if pbp is None or pbp.empty:
        return {}

    pbp = pbp[pbp["season_type"] == "REG"]
    crosswalk = _gsis_to_sleeper()
    out: Dict[str, Dict[str, float]] = {}

    def _emit(gsis, cols):
        pid = crosswalk.get(str(gsis).strip())
        if pid and cols:
            out.setdefault(pid, {}).update(cols)

    # --- Passing (QB) ---
    passes = pbp[pbp["passer_player_id"].notna()]
    for gsis, g in passes.groupby("passer_player_id"):
        dropbacks = float(g["qb_dropback"].sum()) if "qb_dropback" in g else float(len(g))
        cols: Dict[str, float] = {}
        pe = _f(g["epa"].sum())
        if pe is not None:
            cols["passing_epa"] = round(pe, 1)
        qbepa = _f(g["qb_epa"].mean())
        if qbepa is not None:
            cols["epa_per_play"] = round(qbepa, 3)
        cpoe = _f(g["cpoe"].mean())
        if cpoe is not None:
            cols["cpoe"] = round(cpoe, 1)
        if dropbacks > 0:
            cols["sack_rate"] = round(float(g["sack"].sum()) / dropbacks * 100, 1)
            cols["scramble_rate"] = round(float(g["qb_scramble"].sum()) / dropbacks * 100, 1)
        sr = _f(g["success"].mean())
        if sr is not None:
            cols["success_rate"] = round(sr * 100, 1)
        _emit(gsis, cols)

    # --- Rushing ---
    rushes = pbp[pbp["rusher_player_id"].notna()]
    for gsis, g in rushes.groupby("rusher_player_id"):
        cols = {}
        re_ = _f(g["epa"].sum())
        if re_ is not None:
            cols["rushing_epa"] = round(re_, 1)
        ry = g["rushing_yards"].fillna(0)
        total_ry = float(ry.sum())
        if total_ry > 0:
            breakaway = float(ry[ry >= 15].sum())
            cols["breakaway_percentage"] = round(breakaway / total_ry * 100, 1)
        cols["explosive_runs_10_plus"] = int((ry >= 10).sum())
        _emit(gsis, cols)

    # --- Receiving ---
    recs = pbp[pbp["receiver_player_id"].notna()]
    for gsis, g in recs.groupby("receiver_player_id"):
        rce = _f(g["epa"].sum())
        if rce is not None:
            _emit(gsis, {"receiving_epa": round(rce, 1)})

    return out


def build_nflverse_metrics_for_season(season: int) -> Dict[str, Dict[str, float]]:
    """Merge NGS + FTN + pbp-derived metrics into one {sleeper_id: {columns}} map."""
    combined: Dict[str, Dict[str, float]] = {}
    for part in (build_ngs_receiving_for_season(season),
                 build_ftn_charting_for_season(season),
                 build_pbp_metrics_for_season(season)):
        for pid, cols in part.items():
            combined.setdefault(pid, {}).update(cols)
    return combined
