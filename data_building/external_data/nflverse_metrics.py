"""
Build redistributable advanced metrics from the nflverse ecosystem:

  - NGS receiving (2016+) via nfl_data_py.import_ngs_data
      avg_separation, avg_cushion, intended air yards, YAC / xYAC, catch %
  - NGS passing (2016+)
      time to throw, aggressiveness, completed air yards, CPOE, air yards to sticks
  - NGS rushing (2016+)
      RYOE, efficiency, time to LOS, 8+ defender rate
  - FTN charting (2022+) joined to play-by-play
      drop_rate, contested_catch_rate, adj completion,
      play-action / out-of-pocket / blitz splits, stacked-box EPA
  - Play-by-play EPA family
      passing/rushing/receiving EPA, CPOE, success rates, sack/scramble,
      PACR/RACR, QB hit rate, explosive pass rate, EPA per rush/target

Unlike PFF/SumerSports, these sources are redistributable, so the derived
columns are safe to display publicly. They populate the same drop_rate /
contested_catch_rate / avg_depth_of_target columns the PFF importer uses (plus
new ngs_* columns), and the season reader coalesces all snapshot rows so these
merge with the computed + PFF snapshots for the same season.

nfl_data_py is an optional dependency; every builder degrades to {} if it (or the
underlying data) is unavailable, mirroring pfr_snap_counts.py.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

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


def season_team_by_sleeper(season: int) -> Dict[str, str]:
    """{sleeper_id: team} for the team each player was ROSTERED on in `season`,
    from nfl_data_py seasonal rosters (weekly rosters as a fallback).

    Used to stamp the correct HISTORICAL team on a season's usage rows, instead of
    the player's current players_index team — otherwise a trade rewrites the past
    (a player's prior-season row shows their new team) and breaks vacated-opportunity
    detection. Returns {} when nfl_data_py or the data is unavailable, so callers can
    fall back to the current-team behavior.
    """
    xwalk = _gsis_to_sleeper()
    if not xwalk:
        return {}
    out: Dict[str, str] = {}
    try:
        import nfl_data_py as nfl  # optional dependency
        df = None
        for fn in ("import_seasonal_rosters", "import_weekly_rosters", "import_rosters"):
            f = getattr(nfl, fn, None)
            if f is None:
                continue
            try:
                df = f([season])
            except Exception:
                df = None
            if df is not None and not df.empty:
                break
        if df is None or df.empty:
            return {}
        # Weekly rosters carry a week column — keep the latest week per player so a
        # mid-season move resolves to the team they finished the year on.
        if "week" in df.columns:
            df = df.sort_values("week").groupby("player_id", as_index=False).last()
        for _, row in df.iterrows():
            gsis = row.get("player_id")
            if gsis is None:
                gsis = row.get("gsis_id")
            team = row.get("team")
            if gsis is None or team is None:
                continue
            sid = xwalk.get(str(gsis).strip())
            if not sid:
                continue
            t = str(team).strip()
            if t and t.lower() != "nan":
                out[sid] = t
    except Exception as e:
        print(f"[nflverse_metrics] season roster teams unavailable ({e})")
    return out


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


def _flag(v) -> float:
    """Treat bools / 0-1 flags as 0.0 or 1.0; missing/NaN → 0.0."""
    f = _f(v)
    if f is None:
        return 0.0
    return 1.0 if f else 0.0


def _rate_pct(numer: float, denom: float, digits: int = 1) -> Optional[float]:
    """(numer / denom) * 100, or None when denom is empty."""
    if denom and denom > 0:
        return round(float(numer) / float(denom) * 100.0, digits)
    return None


def _created_separation(sep: Optional[float], cushion: Optional[float]) -> Optional[float]:
    """Yards of separation created vs the pre-snap cushion (sep − cushion)."""
    if sep is None or cushion is None:
        return None
    return round(float(sep) - float(cushion), 2)


def _apply_created_separation(row: Dict[str, float]) -> None:
    created = _created_separation(
        row.get("ngs_avg_separation"), row.get("ngs_avg_cushion"))
    if created is not None:
        row["ngs_created_separation"] = created


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

        _apply_created_separation(row)
        if row:
            out[pid] = row
    return out


def build_ngs_passing_for_season(season: int) -> Dict[str, Dict[str, float]]:
    """Return {sleeper_id: {ngs passing columns}} for a season's NGS passing data.

    Time-to-throw, aggressiveness, and air-yards differentials are the closest
    open substitutes for PFF time-to-throw / big-time-throw style QB tiles.
    """
    if season < NGS_FLOOR:
        return {}

    try:
        import nfl_data_py as nfl
        df = nfl.import_ngs_data(stat_type="passing", years=[season])
    except Exception as e:
        print(f"[nflverse_metrics] NGS passing unavailable for {season} ({e})")
        return {}

    if df is None or df.empty:
        return {}

    df = df[(df["season_type"] == "REG") & (df["week"] == 0)]
    crosswalk = _gsis_to_sleeper()

    out: Dict[str, Dict[str, float]] = {}
    for _, r in df.iterrows():
        pid = crosswalk.get(str(r.get("player_gsis_id") or "").strip())
        if not pid:
            continue
        row: Dict[str, float] = {}
        for src, dst in (
            ("avg_time_to_throw", "ngs_avg_time_to_throw"),
            ("aggressiveness", "ngs_aggressiveness"),
            ("avg_completed_air_yards", "ngs_avg_completed_air_yards"),
            ("avg_air_yards_differential", "ngs_avg_air_yards_differential"),
            ("avg_air_yards_to_sticks", "ngs_avg_air_yards_to_sticks"),
            ("completion_percentage_above_expectation", "ngs_cpoe"),
            ("max_completed_air_distance", "ngs_max_completed_air_distance"),
        ):
            val = _f(r.get(src))
            if val is not None:
                row[dst] = round(val, 2)
        if row:
            out[pid] = row
    return out


def build_ngs_rushing_for_season(season: int) -> Dict[str, Dict[str, float]]:
    """Return {sleeper_id: {...}} of NGS rushing metrics (RYOE, efficiency).

    A free 'creation' metric for backs — the closest open equivalent to PFF's
    elusive rating (yards generated beyond what blocking/situation expected).
    """
    if season < NGS_FLOOR:
        return {}

    try:
        import nfl_data_py as nfl
        df = nfl.import_ngs_data(stat_type="rushing", years=[season])
    except Exception as e:
        print(f"[nflverse_metrics] NGS rushing unavailable for {season} ({e})")
        return {}

    if df is None or df.empty:
        return {}

    df = df[(df["season_type"] == "REG") & (df["week"] == 0)]
    crosswalk = _gsis_to_sleeper()

    out: Dict[str, Dict[str, float]] = {}
    for _, r in df.iterrows():
        pid = crosswalk.get(str(r.get("player_gsis_id") or "").strip())
        if not pid:
            continue
        row: Dict[str, float] = {}
        for src, dst in (
            ("rush_yards_over_expected", "ngs_rush_yards_over_expected"),
            ("rush_yards_over_expected_per_att", "ngs_rush_yards_over_expected_per_att"),
            ("efficiency", "ngs_rush_efficiency"),
            ("avg_time_to_los", "ngs_avg_time_to_los"),
            ("percent_attempts_gte_eight_defenders",
             "ngs_percent_attempts_gte_eight_defenders"),
        ):
            val = _f(r.get(src))
            if val is not None:
                row[dst] = round(val, 2)
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
                "receiver_player_id", "passer_player_id", "rusher_player_id",
                "complete_pass", "pass_attempt", "rush_attempt",
                "epa", "qb_dropback",
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

    reg = merged[merged["season_type"] == "REG"]
    crosswalk = _gsis_to_sleeper()
    out: Dict[str, Dict[str, float]] = {}

    # --- Per targeted receiver: drop rate + contested catch rate ---
    recs = reg[reg["receiver_player_id"].notna()]
    agg: Dict[str, Dict[str, float]] = {}
    for _, r in recs.iterrows():
        gsis = str(r.get("receiver_player_id") or "").strip()
        if not gsis:
            continue
        bucket = agg.setdefault(gsis, {"catchable": 0.0, "drops": 0.0,
                                       "contested": 0.0, "contested_caught": 0.0})
        bucket["catchable"] += _f(r.get("is_catchable_ball")) or 0.0
        bucket["drops"] += _f(r.get("is_drop")) or 0.0
        contested = _f(r.get("is_contested_ball")) or 0.0
        bucket["contested"] += contested
        if contested and (_f(r.get("complete_pass")) or 0.0):
            bucket["contested_caught"] += 1.0
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
            out.setdefault(pid, {}).update(row)

    # --- Per passer: adjusted completion rate ---
    # (completions + drops) / (attempts - throwaways): PFF-style accuracy.
    passers = reg[reg["passer_player_id"].notna()]
    pagg: Dict[str, Dict[str, float]] = {}
    for _, r in passers.iterrows():
        gsis = str(r.get("passer_player_id") or "").strip()
        if not gsis:
            continue
        b = pagg.setdefault(gsis, {
            "att": 0.0, "cmp": 0.0, "drops": 0.0, "throwaways": 0.0,
            "plays": 0.0, "pa": 0.0, "oop": 0.0, "blitz": 0.0,
            "pa_epa_sum": 0.0, "pa_epa_n": 0.0,
            "blitz_epa_sum": 0.0, "blitz_epa_n": 0.0,
        })
        b["att"] += _f(r.get("pass_attempt")) or 0.0
        b["cmp"] += _f(r.get("complete_pass")) or 0.0
        b["drops"] += _f(r.get("is_drop")) or 0.0
        b["throwaways"] += _f(r.get("is_throw_away")) or 0.0
        dropback = _f(r.get("qb_dropback")) or _f(r.get("pass_attempt")) or 0.0
        if dropback:
            b["plays"] += 1.0
            pa = _flag(r.get("is_play_action"))
            oop = _flag(r.get("is_qb_out_of_pocket"))
            blitz = _flag(r.get("is_blitz"))
            b["pa"] += pa
            b["oop"] += oop
            b["blitz"] += blitz
            epa = _f(r.get("epa"))
            if epa is not None:
                if pa:
                    b["pa_epa_sum"] += epa
                    b["pa_epa_n"] += 1.0
                if blitz:
                    b["blitz_epa_sum"] += epa
                    b["blitz_epa_n"] += 1.0
    for gsis, b in pagg.items():
        pid = crosswalk.get(gsis)
        if not pid:
            continue
        row: Dict[str, float] = {}
        denom = b["att"] - b["throwaways"]
        if denom > 0:
            row["adjusted_completion_rate"] = round(
                (b["cmp"] + b["drops"]) / denom * 100.0, 1)
        if b["plays"] > 0:
            pa_rate = _rate_pct(b["pa"], b["plays"])
            if pa_rate is not None:
                row["play_action_rate"] = pa_rate
            oop_rate = _rate_pct(b["oop"], b["plays"])
            if oop_rate is not None:
                row["out_of_pocket_rate"] = oop_rate
            blitz_rate = _rate_pct(b["blitz"], b["plays"])
            if blitz_rate is not None:
                row["blitz_rate_faced"] = blitz_rate
        if b["pa_epa_n"] > 0:
            row["play_action_epa"] = round(b["pa_epa_sum"] / b["pa_epa_n"], 3)
        if b["blitz_epa_n"] > 0:
            row["epa_vs_blitz"] = round(b["blitz_epa_sum"] / b["blitz_epa_n"], 3)
        if row:
            out.setdefault(pid, {}).update(row)

    # --- Per rusher: EPA vs 8+ in the box ---
    rushers = reg[reg["rusher_player_id"].notna()]
    ragg: Dict[str, Dict[str, float]] = {}
    for _, r in rushers.iterrows():
        gsis = str(r.get("rusher_player_id") or "").strip()
        if not gsis:
            continue
        b = ragg.setdefault(gsis, {"stacked_epa_sum": 0.0, "stacked_epa_n": 0.0})
        box = _f(r.get("n_defense_box"))
        epa = _f(r.get("epa"))
        if box is not None and box >= 8 and epa is not None:
            b["stacked_epa_sum"] += epa
            b["stacked_epa_n"] += 1.0
    for gsis, b in ragg.items():
        pid = crosswalk.get(gsis)
        if not pid or b["stacked_epa_n"] <= 0:
            continue
        out.setdefault(pid, {})["epa_vs_stacked_box"] = round(
            b["stacked_epa_sum"] / b["stacked_epa_n"], 3)

    return out


def build_pbp_metrics_for_season(season: int) -> Dict[str, Dict[str, float]]:
    """Return {sleeper_id: {...}} of EPA-family + breakaway/explosive metrics.

    All derived from open nflverse play-by-play, so fully public-safe:
      QB:      passing_epa, epa_per_play, cpoe, sack_rate, scramble_rate,
               success_rate, qb_hit_rate, explosive_pass_rate, pacr
      Rusher:  rushing_epa, rushing_epa_per_att, rushing_success_rate,
               breakaway_percentage, explosive_runs_10_plus
      Receiver: receiving_epa, receiving_epa_per_target, receiving_success_rate, racr
    Rates are stored as percentages to match the existing column conventions.
    """
    try:
        import nfl_data_py as nfl
        pbp = nfl.import_pbp_data(
            [season],
            columns=[
                "game_id", "play_id", "season_type", "play_type",
                "epa", "qb_epa", "cpoe", "success",
                "sack", "qb_scramble", "qb_hit", "rush_attempt", "pass_attempt",
                "qb_dropback", "rushing_yards", "air_yards", "yards_gained",
                "complete_pass", "yards_after_catch",
                "passing_yards", "pass_touchdown", "interception",
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
            if "qb_hit" in g.columns:
                hit_rate = _rate_pct(float(g["qb_hit"].fillna(0).sum()), dropbacks)
                if hit_rate is not None:
                    cols["qb_hit_rate"] = hit_rate
        sr = _f(g["success"].mean())
        if sr is not None:
            cols["success_rate"] = round(sr * 100, 1)
        # Standard NFL passer rating from box score (free, all seasons).
        att = float(g["pass_attempt"].fillna(0).sum())
        if att >= 1:
            cmp_ = float(g["complete_pass"].fillna(0).sum())
            yds = float(g["passing_yards"].fillna(0).sum())
            td = float(g["pass_touchdown"].fillna(0).sum())
            ints = float(g["interception"].fillna(0).sum())

            def _clamp(x):
                return min(max(x, 0.0), 2.375)

            a = _clamp((cmp_ / att - 0.3) * 5)
            b = _clamp((yds / att - 3) * 0.25)
            c = _clamp((td / att) * 20)
            d = _clamp(2.375 - (ints / att) * 25)
            cols["nfl_passer_rating"] = round((a + b + c + d) / 6 * 100, 1)
            explosive = float(
                ((g["complete_pass"].fillna(0) > 0) &
                 (g["passing_yards"].fillna(0) >= 16)).sum()
            )
            exp_rate = _rate_pct(explosive, att)
            if exp_rate is not None:
                cols["explosive_pass_rate"] = exp_rate
            if "air_yards" in g.columns:
                air = float(g["air_yards"].fillna(0).sum())
                if air > 0:
                    cols["pacr"] = round(yds / air, 3)
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
        sr = _f(g["success"].mean())
        if sr is not None:
            cols["rushing_success_rate"] = round(sr * 100, 1)
        epa_att = _f(g["epa"].mean())
        if epa_att is not None:
            cols["rushing_epa_per_att"] = round(epa_att, 3)
        _emit(gsis, cols)

    # --- Receiving ---
    recs = pbp[pbp["receiver_player_id"].notna()]
    for gsis, g in recs.groupby("receiver_player_id"):
        cols = {}
        rce = _f(g["epa"].sum())
        if rce is not None:
            cols["receiving_epa"] = round(rce, 1)
        # Yards after catch (yards_after_catch is populated on completions).
        receptions = float(g["complete_pass"].fillna(0).sum())
        total_yac = float(g["yards_after_catch"].fillna(0).sum())
        if receptions > 0:
            cols["yards_after_catch"] = round(total_yac, 0)
            cols["yards_after_catch_per_reception"] = round(total_yac / receptions, 1)
        sr = _f(g["success"].mean())
        if sr is not None:
            cols["receiving_success_rate"] = round(sr * 100, 1)
        epa_tgt = _f(g["epa"].mean())
        if epa_tgt is not None:
            cols["receiving_epa_per_target"] = round(epa_tgt, 3)
        air = float(g["air_yards"].fillna(0).sum()) if "air_yards" in g.columns else 0.0
        rec_yds = 0.0
        if "yards_gained" in g.columns:
            rec_yds = float(
                g.loc[g["complete_pass"].fillna(0) > 0, "yards_gained"].fillna(0).sum())
        elif receptions > 0:
            rec_air = float(
                g.loc[g["complete_pass"].fillna(0) > 0, "air_yards"].fillna(0).sum()
            ) if "air_yards" in g.columns else 0.0
            rec_yds = rec_air + total_yac
        if air > 0:
            cols["racr"] = round(rec_yds / air, 3)
        _emit(gsis, cols)

    return out


def build_nflverse_metrics_for_season(season: int) -> Dict[str, Dict[str, float]]:
    """Merge NGS + FTN + pbp-derived metrics into one {sleeper_id: {columns}} map."""
    combined: Dict[str, Dict[str, float]] = {}
    for part in (build_ngs_receiving_for_season(season),
                 build_ngs_passing_for_season(season),
                 build_ngs_rushing_for_season(season),
                 build_ftn_charting_for_season(season),
                 build_pbp_metrics_for_season(season)):
        for pid, cols in part.items():
            combined.setdefault(pid, {}).update(cols)
    for row in combined.values():
        _apply_created_separation(row)
    return combined


# Per-week metric values plus the volume weights needed to re-aggregate the
# rate metrics over an arbitrary week range (same "store the components, derive
# over the range" shape the usage weekly metrics use for yards/touch etc.).
# Totals (sum over a range): passing_epa, rushing_epa, receiving_epa,
# yards_after_catch, explosive_runs_10_plus, ngs_rush_yards_over_expected.
# Everything else is a rate averaged over the range, weighted by the matching
# volume column (w_dropbacks / w_carries / w_targets / w_receptions).
def build_nflverse_weekly_metrics_for_season(
    season: int,
) -> Dict[Tuple[str, int], Dict[str, float]]:
    """Return {(sleeper_id, week): {metric cols + weight cols}} per game week.

    Mirrors build_nflverse_metrics_for_season but at week granularity so the
    advanced metrics can be filtered by week. NGS publishes native per-week rows
    (week > 0); play-by-play and FTN are play-level and grouped by (player, week).
    """
    crosswalk = _gsis_to_sleeper()
    out: Dict[Tuple[str, int], Dict[str, float]] = {}

    def _row(pid: str, week) -> Dict[str, float]:
        try:
            wk = int(week)
        except (TypeError, ValueError):
            return {}
        return out.setdefault((pid, wk), {})

    # ---------- NGS receiving (native weekly rows) ----------
    if season >= NGS_FLOOR:
        try:
            import nfl_data_py as nfl
            df = nfl.import_ngs_data(stat_type="receiving", years=[season])
            df = df[(df["season_type"] == "REG") & (df["week"] > 0)]
            for _, r in df.iterrows():
                pid = crosswalk.get(str(r.get("player_gsis_id") or "").strip())
                if not pid:
                    continue
                row = _row(pid, r.get("week"))
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
                        row[dst] = round(val, 2)
                iay = _f(r.get("avg_intended_air_yards"))
                if iay is not None:
                    row["avg_depth_of_target"] = round(iay, 2)
                _apply_created_separation(row)
        except Exception as e:
            print(f"[nflverse_metrics] weekly NGS receiving unavailable for {season} ({e})")

    # ---------- NGS passing (native weekly rows) ----------
    if season >= NGS_FLOOR:
        try:
            import nfl_data_py as nfl
            df = nfl.import_ngs_data(stat_type="passing", years=[season])
            df = df[(df["season_type"] == "REG") & (df["week"] > 0)]
            for _, r in df.iterrows():
                pid = crosswalk.get(str(r.get("player_gsis_id") or "").strip())
                if not pid:
                    continue
                row = _row(pid, r.get("week"))
                for src, dst in (
                    ("avg_time_to_throw", "ngs_avg_time_to_throw"),
                    ("aggressiveness", "ngs_aggressiveness"),
                    ("avg_completed_air_yards", "ngs_avg_completed_air_yards"),
                    ("avg_air_yards_differential", "ngs_avg_air_yards_differential"),
                    ("avg_air_yards_to_sticks", "ngs_avg_air_yards_to_sticks"),
                    ("completion_percentage_above_expectation", "ngs_cpoe"),
                    ("max_completed_air_distance", "ngs_max_completed_air_distance"),
                ):
                    val = _f(r.get(src))
                    if val is not None:
                        row[dst] = round(val, 2)
        except Exception as e:
            print(f"[nflverse_metrics] weekly NGS passing unavailable for {season} ({e})")

    # ---------- NGS rushing (native weekly rows) ----------
    if season >= NGS_FLOOR:
        try:
            import nfl_data_py as nfl
            df = nfl.import_ngs_data(stat_type="rushing", years=[season])
            df = df[(df["season_type"] == "REG") & (df["week"] > 0)]
            for _, r in df.iterrows():
                pid = crosswalk.get(str(r.get("player_gsis_id") or "").strip())
                if not pid:
                    continue
                row = _row(pid, r.get("week"))
                ryoe = _f(r.get("rush_yards_over_expected"))
                if ryoe is not None:
                    row["ngs_rush_yards_over_expected"] = round(ryoe, 2)
                rpa = _f(r.get("rush_yards_over_expected_per_att"))
                if rpa is not None:
                    row["ngs_rush_yards_over_expected_per_att"] = round(rpa, 2)
                eff = _f(r.get("efficiency"))
                if eff is not None:
                    row["ngs_rush_efficiency"] = round(eff, 2)
                tlos = _f(r.get("avg_time_to_los"))
                if tlos is not None:
                    row["ngs_avg_time_to_los"] = round(tlos, 2)
                eight = _f(r.get("percent_attempts_gte_eight_defenders"))
                if eight is not None:
                    row["ngs_percent_attempts_gte_eight_defenders"] = round(eight, 1)
        except Exception as e:
            print(f"[nflverse_metrics] weekly NGS rushing unavailable for {season} ({e})")

    # ---------- play-by-play (EPA family) + FTN charting ----------
    try:
        import nfl_data_py as nfl
        pbp = nfl.import_pbp_data(
            [season],
            columns=[
                "game_id", "play_id", "week", "season_type", "play_type",
                "epa", "qb_epa", "cpoe", "success",
                "sack", "qb_scramble", "qb_hit", "rush_attempt", "pass_attempt",
                "qb_dropback", "rushing_yards", "air_yards", "yards_gained",
                "complete_pass", "yards_after_catch",
                "passing_yards", "pass_touchdown", "interception",
                "passer_player_id", "rusher_player_id", "receiver_player_id",
            ],
            downcast=True,
        )
    except Exception as e:
        print(f"[nflverse_metrics] weekly pbp unavailable for {season} ({e})")
        pbp = None

    if pbp is not None and not pbp.empty:
        pbp = pbp[pbp["season_type"] == "REG"]

        # --- Passing (QB) per player+week ---
        for (gsis, week), g in pbp[pbp["passer_player_id"].notna()].groupby(
                ["passer_player_id", "week"]):
            pid = crosswalk.get(str(gsis).strip())
            if not pid:
                continue
            cols = _row(pid, week)
            dropbacks = float(g["qb_dropback"].sum()) if "qb_dropback" in g else float(len(g))
            att = float(g["pass_attempt"].fillna(0).sum())
            cols["w_dropbacks"] = dropbacks
            cols["w_pass_att"] = att
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
                if "qb_hit" in g.columns:
                    hit_rate = _rate_pct(float(g["qb_hit"].fillna(0).sum()), dropbacks)
                    if hit_rate is not None:
                        cols["qb_hit_rate"] = hit_rate
            sr = _f(g["success"].mean())
            if sr is not None:
                cols["success_rate"] = round(sr * 100, 1)
            if att >= 1:
                cmp_ = float(g["complete_pass"].fillna(0).sum())
                yds = float(g["passing_yards"].fillna(0).sum())
                td = float(g["pass_touchdown"].fillna(0).sum())
                ints = float(g["interception"].fillna(0).sum())

                def _clamp(x):
                    return min(max(x, 0.0), 2.375)

                a = _clamp((cmp_ / att - 0.3) * 5)
                b = _clamp((yds / att - 3) * 0.25)
                c = _clamp((td / att) * 20)
                d = _clamp(2.375 - (ints / att) * 25)
                cols["nfl_passer_rating"] = round((a + b + c + d) / 6 * 100, 1)
                explosive = float(
                    ((g["complete_pass"].fillna(0) > 0) &
                     (g["passing_yards"].fillna(0) >= 16)).sum()
                )
                exp_rate = _rate_pct(explosive, att)
                if exp_rate is not None:
                    cols["explosive_pass_rate"] = exp_rate
                if "air_yards" in g.columns:
                    air = float(g["air_yards"].fillna(0).sum())
                    cols["w_pass_air_yards"] = air
                    if air > 0:
                        cols["pacr"] = round(yds / air, 3)

        # --- Rushing per player+week ---
        for (gsis, week), g in pbp[pbp["rusher_player_id"].notna()].groupby(
                ["rusher_player_id", "week"]):
            pid = crosswalk.get(str(gsis).strip())
            if not pid:
                continue
            cols = _row(pid, week)
            cols["w_carries"] = float(g["rush_attempt"].fillna(0).sum()) or float(len(g))
            re_ = _f(g["epa"].sum())
            if re_ is not None:
                cols["rushing_epa"] = round(re_, 1)
            ry = g["rushing_yards"].fillna(0)
            total_ry = float(ry.sum())
            if total_ry > 0:
                cols["breakaway_percentage"] = round(float(ry[ry >= 15].sum()) / total_ry * 100, 1)
            cols["explosive_runs_10_plus"] = int((ry >= 10).sum())
            sr = _f(g["success"].mean())
            if sr is not None:
                cols["rushing_success_rate"] = round(sr * 100, 1)
            epa_att = _f(g["epa"].mean())
            if epa_att is not None:
                cols["rushing_epa_per_att"] = round(epa_att, 3)

        # --- Receiving per player+week ---
        for (gsis, week), g in pbp[pbp["receiver_player_id"].notna()].groupby(
                ["receiver_player_id", "week"]):
            pid = crosswalk.get(str(gsis).strip())
            if not pid:
                continue
            cols = _row(pid, week)
            cols["w_targets"] = float(len(g))
            receptions = float(g["complete_pass"].fillna(0).sum())
            cols["w_receptions"] = receptions
            rce = _f(g["epa"].sum())
            if rce is not None:
                cols["receiving_epa"] = round(rce, 1)
            total_yac = float(g["yards_after_catch"].fillna(0).sum())
            if receptions > 0:
                cols["yards_after_catch"] = round(total_yac, 0)
                cols["yards_after_catch_per_reception"] = round(total_yac / receptions, 1)
            sr = _f(g["success"].mean())
            if sr is not None:
                cols["receiving_success_rate"] = round(sr * 100, 1)
            epa_tgt = _f(g["epa"].mean())
            if epa_tgt is not None:
                cols["receiving_epa_per_target"] = round(epa_tgt, 3)
            air = float(g["air_yards"].fillna(0).sum()) if "air_yards" in g.columns else 0.0
            cols["w_rec_air_yards"] = air
            rec_yds = 0.0
            if "yards_gained" in g.columns:
                rec_yds = float(
                    g.loc[g["complete_pass"].fillna(0) > 0, "yards_gained"].fillna(0).sum())
            elif receptions > 0 and "air_yards" in g.columns:
                rec_yds = float(
                    g.loc[g["complete_pass"].fillna(0) > 0, "air_yards"].fillna(0).sum()
                ) + total_yac
            if air > 0:
                cols["racr"] = round(rec_yds / air, 3)

        # --- FTN charting (drop / contested / adjusted completion) per player+week ---
        if season >= FTN_FLOOR:
            try:
                ftn = nfl.import_ftn_data([season])
                # FTN has its own 'week' column; drop it before merging so we
                # use pbp's week without pandas creating week_x / week_y suffixes.
                ftn_trimmed = ftn.drop(columns=["week"], errors="ignore")
                merged = ftn_trimmed.merge(
                    pbp[["game_id", "play_id", "week", "receiver_player_id",
                         "passer_player_id", "rusher_player_id",
                         "complete_pass", "pass_attempt", "epa", "qb_dropback"]],
                    left_on=["nflverse_game_id", "nflverse_play_id"],
                    right_on=["game_id", "play_id"],
                    how="inner",
                )
                for (gsis, week), g in merged[merged["receiver_player_id"].notna()].groupby(
                        ["receiver_player_id", "week"]):
                    pid = crosswalk.get(str(gsis).strip())
                    if not pid:
                        continue
                    cols = _row(pid, week)
                    catchable = float(g["is_catchable_ball"].fillna(0).sum())
                    drops = float(g["is_drop"].fillna(0).sum())
                    contested = float(g["is_contested_ball"].fillna(0).sum())
                    contested_caught = float(
                        ((g["is_contested_ball"].fillna(0) > 0) &
                         (g["complete_pass"].fillna(0) > 0)).sum())
                    if catchable > 0:
                        cols["drop_rate"] = round(drops / catchable * 100, 1)
                    if contested > 0:
                        cols["contested_catch_rate"] = round(contested_caught / contested * 100, 1)
                for (gsis, week), g in merged[merged["passer_player_id"].notna()].groupby(
                        ["passer_player_id", "week"]):
                    pid = crosswalk.get(str(gsis).strip())
                    if not pid:
                        continue
                    cols = _row(pid, week)
                    att = float(g["pass_attempt"].fillna(0).sum())
                    throwaways = float(g["is_throw_away"].fillna(0).sum()) if "is_throw_away" in g.columns else 0.0
                    denom = att - throwaways
                    if denom > 0:
                        cmp_ = float(g["complete_pass"].fillna(0).sum())
                        drops = float(g["is_drop"].fillna(0).sum()) if "is_drop" in g.columns else 0.0
                        cols["adjusted_completion_rate"] = round((cmp_ + drops) / denom * 100, 1)
                    # Match season FTN: situation rates/EPA are dropback-only.
                    db = g[g["qb_dropback"].fillna(0) > 0] if "qb_dropback" in g.columns else g
                    dropbacks = float(len(db))
                    if dropbacks > 0:
                        if "is_play_action" in db.columns:
                            pa_rate = _rate_pct(float(db["is_play_action"].fillna(0).sum()), dropbacks)
                            if pa_rate is not None:
                                cols["play_action_rate"] = pa_rate
                            pa_mask = db["is_play_action"].fillna(0) > 0
                            if pa_mask.any() and "epa" in db.columns:
                                pa_epa = _f(db.loc[pa_mask, "epa"].mean())
                                if pa_epa is not None:
                                    cols["play_action_epa"] = round(pa_epa, 3)
                        if "is_qb_out_of_pocket" in db.columns:
                            oop_rate = _rate_pct(
                                float(db["is_qb_out_of_pocket"].fillna(0).sum()), dropbacks)
                            if oop_rate is not None:
                                cols["out_of_pocket_rate"] = oop_rate
                        if "is_blitz" in db.columns:
                            blitz_rate = _rate_pct(float(db["is_blitz"].fillna(0).sum()), dropbacks)
                            if blitz_rate is not None:
                                cols["blitz_rate_faced"] = blitz_rate
                            blitz_mask = db["is_blitz"].fillna(0) > 0
                            if blitz_mask.any() and "epa" in db.columns:
                                blitz_epa = _f(db.loc[blitz_mask, "epa"].mean())
                                if blitz_epa is not None:
                                    cols["epa_vs_blitz"] = round(blitz_epa, 3)
                if "rusher_player_id" in merged.columns and "n_defense_box" in merged.columns:
                    for (gsis, week), g in merged[merged["rusher_player_id"].notna()].groupby(
                            ["rusher_player_id", "week"]):
                        pid = crosswalk.get(str(gsis).strip())
                        if not pid:
                            continue
                        stacked = g["n_defense_box"].fillna(0) >= 8
                        if stacked.any() and "epa" in g.columns:
                            stacked_epa = _f(g.loc[stacked, "epa"].mean())
                            if stacked_epa is not None:
                                _row(pid, week)["epa_vs_stacked_box"] = round(stacked_epa, 3)
            except Exception as e:
                print(f"[nflverse_metrics] weekly FTN unavailable for {season} ({e})")

    # Drop any (pid, week) buckets that ended up with only weight columns and no
    # actual metric value (e.g. a player who only appears as a rusher weight).
    _weight_only = {
        "w_dropbacks", "w_pass_att", "w_carries", "w_targets", "w_receptions",
        "w_pass_air_yards", "w_rec_air_yards",
    }
    for row in out.values():
        _apply_created_separation(row)
    return {k: v for k, v in out.items() if v and (set(v.keys()) - _weight_only)}
