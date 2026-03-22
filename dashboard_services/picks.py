# picks.py

from __future__ import annotations

import math
import pandas as pd
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, Tuple, List, Union

from dashboard_services.api import get_nfl_state
from data_building.paths import DATA_DIR

DYNASTYPROCESS_VALUES_PATH = DATA_DIR / f"dynastyprocess_values_{date.today().isoformat()}.csv"
FANTASYCALC_VALUES_PATH = DATA_DIR / f"fantasycalc_api_values_{date.today().isoformat()}.csv"

# FantasyCalc examples handled:
#   2026 Pick 1.01
#   2026 1.01
#   2027 1st
#   2027 1st (Early)
_FC_NAME_RE = re.compile(
    r"(?P<year>\d{4})\s+"
    r"(?:Pick\s+)?"
    r"(?P<round>\d+)"
    r"(?:\.(?P<pos_in_round>\d+))?"
    r"(?:st|nd|rd|th)?"
    r"(?:\s*\((?P<bucket>Early|Mid|Late)\))?",
    re.IGNORECASE,
)

# DynastyProcess examples handled:
#   2026 Pick 1.01
#   2026 1st
#   2027 Early 1st
#   2027 Mid 2nd
#   2027 Late 3rd
_DP_NAME_RE = re.compile(
    r"(?P<year>\d{4})\s+"
    r"(?:"
    r"(?:Pick\s+(?P<round_dp>\d+)\.(?P<pos_in_round>\d+))"
    r"|"
    r"(?:(?P<bucket>Early|Mid|Late)\s+)?(?P<round_fc>\d+)(?:st|nd|rd|th)"
    r")",
    re.IGNORECASE,
)

PickKey = Tuple[str, int, int, Union[int, str]]


def _normalize_bucket_label(raw: str) -> str:
    if not raw:
        return ""
    s = str(raw).strip().lower()
    if s.startswith("e"):
        return "early"
    if s.startswith("m"):
        return "mid"
    if s.startswith("l"):
        return "late"
    return ""


def _bucket_for_pick_in_round(pos_in_round: int, picks_per_round: int = 10) -> str:
    if pos_in_round is None:
        return ""
    if picks_per_round <= 0:
        return "mid"

    third = picks_per_round / 3.0
    if pos_in_round <= math.ceil(third):
        return "early"
    elif pos_in_round <= math.ceil(2 * third):
        return "mid"
    else:
        return "late"


def _is_current_year_draft_complete(current_year: int) -> bool:
    """
    Approximation:
    - offseason -> upcoming rookie picks should still exist and can be exact slots
    - pre / regular / post -> current rookie draft has effectively passed
    """
    state = get_nfl_state() or {}

    try:
        season = int(state.get("season", current_year))
    except Exception:
        season = current_year

    season_type = str(state.get("season_type", "")).lower().strip()

    return season == current_year and season_type in {"pre", "regular", "post"}


def _use_exact_slots_for_year(year: int, current_year: int, draft_done: bool) -> bool:
    """
    Only the upcoming draft year should use exact pick slots,
    and only while we are still in the offseason before that draft.
    """
    return year == current_year and not draft_done


def _remap_pick_to_league_size(
    rnd: int,
    pos_in_round: int,
    source_picks_per_round: int,
    league_teams: int,
) -> Tuple[int, int]:
    """
    Convert a market/source draft slot into the equivalent slot for this league size
    while preserving absolute pick order.

    Example:
      source: 12-team, 1.12 -> overall 12
      league: 10-team -> 2.02
    """
    if rnd <= 0 or pos_in_round <= 0:
        return rnd, pos_in_round
    if source_picks_per_round <= 0 or league_teams <= 0:
        return rnd, pos_in_round

    overall_pick = (rnd - 1) * source_picks_per_round + pos_in_round
    new_rnd = ((overall_pick - 1) // league_teams) + 1
    new_pos = ((overall_pick - 1) % league_teams) + 1
    return new_rnd, new_pos


def _build_pick_key(
    *,
    year: int,
    rnd: int,
    current_year: int,
    draft_done: bool,
    pos_in_round: int | None = None,
    bucket: str = "",
    bucket_round_size: int = 10,
) -> PickKey | None:
    """
    Rules:
    - past years: excluded elsewhere
    - current year before draft completes: use exact slot if available
    - all other years: use early/mid/late bucket
    - if future pick has no bucket and no exact slot, default to 'early'
      so generic values like '2027 1st' still appear
    """
    if _use_exact_slots_for_year(year, current_year, draft_done):
        if pos_in_round is None:
            return None
        return ("slot", year, rnd, int(pos_in_round))

    if not bucket and pos_in_round is not None:
        bucket = _bucket_for_pick_in_round(pos_in_round, bucket_round_size)

    bucket = _normalize_bucket_label(bucket)

    # Important for values like "2027 1st" from FantasyCalc
    if not bucket:
        bucket = "early"

    return ("bucket", year, rnd, bucket)


def _pick_key_sort_value(key: PickKey):
    kind, year, rnd, detail = key
    if kind == "slot":
        return year, rnd, 0, int(detail)
    bucket_order = {"early": 0, "mid": 1, "late": 2}
    return year, rnd, 1, bucket_order.get(str(detail), 9)


def _pick_key_to_output_string(key: PickKey) -> str:
    kind, year, rnd, detail = key
    if kind == "slot":
        return f"{year}_{rnd}_{int(detail):02d}"
    return f"{year}_{rnd}_{detail}"


def load_pick_value_table(
    fantasycalc_csv: Path = FANTASYCALC_VALUES_PATH,
    dynastyprocess_csv: Path = DYNASTYPROCESS_VALUES_PATH,
    league_teams: int = 10,
    source_picks_per_round: int = 12,
    w_fc: float = 0.55,
    w_dp: float = 0.45,
    current_year: int | None = None,
) -> Dict[str, float]:
    """
    Build a draft pick value table by merging FantasyCalc + DynastyProcess.

    Output key styles:
      offseason upcoming draft:
        2026_1_01
        2026_2_02

      future drafts / unknown exact order:
        2027_1_early
        2027_2_mid
        2028_1_late

    Parameters:
      league_teams:
        number of teams in the user's league; used for exact-slot remapping and bucketing

      source_picks_per_round:
        number of picks per round in the source market values.
        Keep at 12 if your source pick market is effectively 12-team.
    """

    if current_year is None:
        current_year = date.today().year

    draft_done = _is_current_year_draft_complete(current_year)

    fc_vals: Dict[PickKey, List[float]] = defaultdict(list)
    dp_vals: Dict[PickKey, List[float]] = defaultdict(list)

    # ------------------ FantasyCalc ------------------
    try:
        df_fc = pd.read_csv(fantasycalc_csv)
    except FileNotFoundError:
        df_fc = pd.DataFrame()

    if not df_fc.empty:
        value_col = "value"

        if value_col not in df_fc.columns:
            print(f"[load_pick_value_table] FantasyCalc missing '{value_col}' column, skipping")
        else:
            pos_series = df_fc.get("position")
            if pos_series is not None:
                mask_players = pos_series.astype(str).str.upper() != "PICK"
                df_fc_players = df_fc.loc[mask_players]
            else:
                df_fc_players = df_fc

            max_raw_fc = float(pd.to_numeric(df_fc_players[value_col], errors="coerce").max() or 0.0)
            fc_scale = (999.9 / max_raw_fc) if max_raw_fc > 0 else 0.0

            if "position" in df_fc.columns:
                df_fc_picks = df_fc.loc[df_fc["position"].astype(str).str.upper() == "PICK"].copy()
            else:
                df_fc_picks = pd.DataFrame()

            for row in df_fc_picks.itertuples(index=False):
                name = getattr(row, "name", "")
                m = _FC_NAME_RE.search(str(name))
                if not m:
                    continue

                try:
                    year = int(m.group("year"))
                    rnd = int(m.group("round"))
                    raw_val = float(getattr(row, value_col))

                    pos_in_round_raw = m.group("pos_in_round")
                    pos_in_round = int(pos_in_round_raw) if pos_in_round_raw is not None else None
                    bucket = _normalize_bucket_label(m.group("bucket"))
                except Exception:
                    continue

                if year < current_year:
                    continue

                if year == current_year and draft_done:
                    continue

                if rnd not in (1, 2, 3):
                    continue

                # Remap exact slots to actual league size for the upcoming draft
                if pos_in_round is not None and _use_exact_slots_for_year(year, current_year, draft_done):
                    rnd, pos_in_round = _remap_pick_to_league_size(
                        rnd=rnd,
                        pos_in_round=pos_in_round,
                        source_picks_per_round=source_picks_per_round,
                        league_teams=league_teams,
                    )

                key = _build_pick_key(
                    year=year,
                    rnd=rnd,
                    current_year=current_year,
                    draft_done=draft_done,
                    pos_in_round=pos_in_round,
                    bucket=bucket,
                    bucket_round_size=league_teams,
                )
                if key is None:
                    continue

                scaled_val = raw_val * fc_scale
                fc_vals[key].append(scaled_val)

    # ------------------ DynastyProcess ------------------
    try:
        df_dp = pd.read_csv(dynastyprocess_csv)
    except FileNotFoundError:
        df_dp = pd.DataFrame()

    if not df_dp.empty:
        cols = df_dp.columns
        pos_col = "pos" if "pos" in cols else "position" if "position" in cols else None
        name_col = "player" if "player" in cols else "Player" if "Player" in cols else None
        value_col = "value_1qb"

        if not pos_col or not name_col:
            print("[load_pick_value_table] DynastyProcess: no pos/player columns, skipping")
        elif value_col not in cols:
            print(f"[load_pick_value_table] DynastyProcess missing '{value_col}' column, skipping")
        else:
            mask_players = df_dp[pos_col].astype(str).str.upper() != "PICK"
            df_dp_players = df_dp.loc[mask_players]
            max_raw_dp = float(pd.to_numeric(df_dp_players[value_col], errors="coerce").max() or 0.0)
            dp_scale = (999.9 / max_raw_dp) if max_raw_dp > 0 else 0.0

            df_dp_picks = df_dp.loc[df_dp[pos_col].astype(str).str.upper() == "PICK"].copy()

            for row in df_dp_picks.itertuples(index=False):
                name = getattr(row, name_col)
                m = _DP_NAME_RE.search(str(name))
                if not m:
                    continue

                try:
                    year = int(m.group("year"))
                    rnd = int(m.group("round_dp") or m.group("round_fc"))
                    pos_in_round_raw = m.group("pos_in_round")
                    pos_in_round = int(pos_in_round_raw) if pos_in_round_raw is not None else None
                    bucket = _normalize_bucket_label(m.group("bucket"))
                    raw_val = float(getattr(row, value_col))
                except Exception:
                    continue

                if year < current_year:
                    continue

                if year == current_year and draft_done:
                    continue

                if rnd not in (1, 2, 3):
                    continue

                # Remap exact slots to actual league size for the upcoming draft
                if pos_in_round is not None and _use_exact_slots_for_year(year, current_year, draft_done):
                    rnd, pos_in_round = _remap_pick_to_league_size(
                        rnd=rnd,
                        pos_in_round=pos_in_round,
                        source_picks_per_round=source_picks_per_round,
                        league_teams=league_teams,
                    )

                key = _build_pick_key(
                    year=year,
                    rnd=rnd,
                    current_year=current_year,
                    draft_done=draft_done,
                    pos_in_round=pos_in_round,
                    bucket=bucket,
                    bucket_round_size=league_teams,
                )
                if key is None:
                    continue

                scaled_val = raw_val * dp_scale
                dp_vals[key].append(scaled_val)

    # ------------------ Merge FC + DP ------------------
    all_keys = set(fc_vals.keys()) | set(dp_vals.keys())
    if not all_keys:
        print("[load_pick_value_table] no pick values found from either CSV")
        return {}

    if not fc_vals:
        w_fc, w_dp = 0.0, 1.0
    elif not dp_vals:
        w_fc, w_dp = 1.0, 0.0

    final: Dict[str, float] = {}

    for key in sorted(all_keys, key=_pick_key_sort_value):
        fc_list = fc_vals.get(key, [])
        dp_list = dp_vals.get(key, [])

        fc_val = (sum(fc_list) / len(fc_list)) if fc_list else None
        dp_val = (sum(dp_list) / len(dp_list)) if dp_list else None

        if fc_val is not None and dp_val is not None:
            val = w_fc * fc_val + w_dp * dp_val
        elif fc_val is not None:
            val = fc_val
        elif dp_val is not None:
            val = dp_val
        else:
            continue

        key_str = _pick_key_to_output_string(key)
        final[key_str] = round(float(val), 1)

    print(
        f"[load_pick_value_table] built {len(final)} pick values "
        f"(current_year={current_year}, draft_done={draft_done}, "
        f"league_teams={league_teams}, source_picks_per_round={source_picks_per_round})"
    )

    return final