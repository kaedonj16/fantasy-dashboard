# picks.py

from __future__ import annotations

import math
import pandas as pd
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, Tuple, List

from dashboard_services.api import get_nfl_state
from data_building.paths import DATA_DIR

DYNASTYPROCESS_VALUES_PATH = DATA_DIR / f"dynastyprocess_values_{date.today().isoformat()}.csv"
FANTASYCALC_VALUES_PATH = DATA_DIR / f"fantasycalc_api_values_{date.today().isoformat()}.csv"

# Precompile regex patterns once (used in load_pick_value_table)
_FC_NAME_RE = re.compile(
    r"(?P<year>\d{4})\s+"
    r"(?P<round>\d+)(?:st|nd|rd|th)"
    r"(?:\s*\((?P<bucket>Early|Mid|Late)\))?",
    re.IGNORECASE,
)

_DP_NAME_RE = re.compile(
    r"(?P<year>\d{4})\s+"
    r"(?:(?:Pick\s+(?P<round_dp>\d+)\.(?P<pos_in_round>\d+))|"
    r"(?P<round_fc>\d+)(?:st|nd|rd|th))",
    re.IGNORECASE,
)


def _normalize_bucket_label(raw: str) -> str:
    if not raw:
        return ""
    s = str(raw).strip().lower()
    if s.startswith("e"):
        return "_early"
    if s.startswith("m"):
        return "_mid"
    if s.startswith("l"):
        return "_late"
    return s


def _bucket_for_pick_in_round(pos_in_round: int, picks_per_round: int = 10) -> str:
    if pos_in_round is None:
        return ""
    if picks_per_round <= 0:
        return "_mid"
    third = picks_per_round / 3.0
    if pos_in_round <= math.ceil(third):
        return "_early"
    elif pos_in_round <= math.ceil(2 * third):
        return "_mid"
    else:
        return "_late"


def _is_current_year_draft_complete(current_year: int) -> bool:
    """
    Approximation:
    - offseason -> current rookie picks should still exist
    - pre / regular / post -> rookie draft has effectively passed
    """
    state = get_nfl_state() or {}

    try:
        season = int(state.get("season", current_year))
    except Exception:
        season = current_year

    season_type = str(state.get("season_type", "")).lower().strip()

    return season == current_year and season_type in {"pre", "regular", "post"}


def load_pick_value_table(
        fantasycalc_csv: Path = FANTASYCALC_VALUES_PATH,
        dynastyprocess_csv: Path = DYNASTYPROCESS_VALUES_PATH,
        picks_per_round: int = 10,
        w_fc: float = 0.55,
        w_dp: float = 0.45,
        current_year: int | None = None,
) -> Dict[str, float]:
    """
    Build a draft pick value table by merging FantasyCalc + DynastyProcess.

    Values are scaled from each source's raw CSV values into the 0–999.9 range,
    using non-pick player rows in that same CSV to determine the scale factor.

    Returns keys like:
      {
        "2026_1_early": 510.5,
        "2026_1_mid":   440.2,
        "2026_2_late":  210.7,
      }
    """

    if current_year is None:
        current_year = date.today().year

    draft_done = _is_current_year_draft_complete(current_year)

    fc_vals: Dict[Tuple[int, int, str], List[float]] = defaultdict(list)
    dp_vals: Dict[Tuple[int, int, str], List[float]] = defaultdict(list)

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
                    bucket = _normalize_bucket_label(m.group("bucket"))
                    raw_val = float(getattr(row, value_col))
                except Exception:
                    continue

                # Drop past-year picks always
                if year < current_year:
                    continue

                # Drop current-year picks only after the draft window has passed
                if year == current_year and draft_done:
                    continue

                if rnd not in (1, 2, 3):
                    continue

                scaled_val = raw_val * fc_scale
                key = (year, rnd, bucket)
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
                    rnd = int(m.group("round"))
                    pos_in_round = int(m.group("pos_in_round"))
                    raw_val = float(getattr(row, value_col))
                except Exception:
                    continue

                # Drop past-year picks always
                if year < current_year:
                    continue

                # Drop current-year picks only after the draft window has passed
                if year == current_year and draft_done:
                    continue

                if rnd not in (1, 2, 3):
                    continue

                bucket = _bucket_for_pick_in_round(pos_in_round, picks_per_round)
                scaled_val = raw_val * dp_scale
                key = (year, rnd, bucket)
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

    for year, rnd, bucket in sorted(all_keys):
        fc_list = fc_vals.get((year, rnd, bucket), [])
        dp_list = dp_vals.get((year, rnd, bucket), [])

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

        key_str = f"{year}_{rnd}_{bucket}"
        final[key_str] = round(float(val), 1)

    print(
        f"[load_pick_value_table] built {len(final)} pick values "
        f"(current_year={current_year}, draft_done={draft_done})"
    )

    return final
