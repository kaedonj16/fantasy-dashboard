# picks.py

from __future__ import annotations

import math
import pandas as pd
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, Tuple, List

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DYNASTYPROCESS_VALUES_PATH = DATA_DIR / f"dynastyprocess_values_{date.today().isoformat()}.csv"
FANTASYCALC_VALUES_PATH = DATA_DIR / f"fantasycalc_api_values_{date.today().isoformat()}.csv"


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

    IMPORTANT: Values are scaled directly from the raw CSV values into
    the 0–999.9 range, using the *players in that CSV* to determine the
    scale factor. Example: if max value in FC is 10,000, then 5105
    becomes ~510.5.

    Returns:
      {
        "2026_1_early": 510.5,
        "2026_1_mid":   440.2,
        "2026_2_late":  210.7,
        ...
      }
    """

    if current_year is None:
        current_year = date.today().year

    # (year, round, bucket) -> list of *scaled* values from each source
    fc_vals: Dict[Tuple[int, int, str], List[float]] = defaultdict(list)
    dp_vals: Dict[Tuple[int, int, str], List[float]] = defaultdict(list)

    # ------------------ FantasyCalc ------------------
    try:
        df_fc = pd.read_csv(fantasycalc_csv)
    except FileNotFoundError:
        df_fc = pd.DataFrame()

    if not df_fc.empty:
        # Pick a value column
        value_col = "value"

        # Scale factor: max player value in this CSV -> ~999.9
        # (players + picks share same scale; we use players to define it)
        # If you want to include picks in the max, change the mask.
        pos_series = df_fc.get("position")
        if pos_series is not None:
            mask_players = pos_series.astype(str).str.upper() != "PICK"
            df_fc_players = df_fc[mask_players].copy()
        else:
            df_fc_players = df_fc.copy()

        max_raw_fc = float(df_fc_players[value_col].max() or 0.0)
        fc_scale = (999.9 / max_raw_fc) if max_raw_fc > 0 else 0.0

        # Now only keep PICK rows to build pick values
        df_fc_picks = df_fc[df_fc["position"].astype(str).str.upper() == "PICK"].copy()
        # Names like: "2026 1st (Early)"
        name_re = re.compile(
            r"(?P<year>\d{4})\s+"
            r"(?P<round>\d+)(?:st|nd|rd|th)"
            r"(?:\s*\((?P<bucket>Early|Mid|Late)\))?",
            re.IGNORECASE
        )

        for _, row in df_fc_picks.iterrows():
            m = name_re.search(str(row.get("name", "")))
            if not m:
                continue
            try:
                year = int(m.group("year"))
                rnd = int(m.group("round"))
                bucket = _normalize_bucket_label(m.group("bucket"))
                raw_val = float(row[value_col])
            except Exception:
                continue

            # Drop current season + only rounds 1–3
            if year == current_year:
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
        pos_col = "pos" if "pos" in df_dp.columns else "position" if "position" in df_dp.columns else None
        name_col = "player" if "player" in df_dp.columns else "Player" if "Player" in df_dp.columns else None

        if not pos_col or not name_col:
            print("[load_pick_value_table] DynastyProcess: no pos/player columns, skipping")
        else:
            # Pick a value column
            value_col = "value_1qb"

            # Scale factor based on *players* in DP CSV
            mask_players = df_dp[pos_col].astype(str).str.upper() != "PICK"
            df_dp_players = df_dp[mask_players].copy()
            max_raw_dp = float(df_dp_players[value_col].max() or 0.0)
            dp_scale = (999.9 / max_raw_dp) if max_raw_dp > 0 else 0.0

            # Now only use rows where pos == "PICK"
            df_dp_picks = df_dp[df_dp[pos_col].astype(str).str.upper() == "PICK"].copy()
            # Names like: "2025 Pick 1.01"
            name_re = re.compile(
                r"(?P<year>\d{4})\s+"
                r"(?:(?:Pick\s+(?P<round_dp>\d+)\.(?P<pos_in_round>\d+))|"
                r"(?P<round_fc>\d+)(?:st|nd|rd|th))",
                re.IGNORECASE
            )

            for _, row in df_dp_picks.iterrows():
                m = name_re.search(str(row[name_col]))
                if not m:
                    continue
                try:
                    year = int(m.group("year"))
                    rnd = int(m.group("round"))
                    pos_in_round = int(m.group("pos_in_round"))
                    raw_val = float(row[value_col])
                except Exception:
                    continue

                # Drop current season + only rounds 1–3
                if year == current_year:
                    continue
                if rnd not in (1, 2, 3):
                    continue

                bucket = _bucket_for_pick_in_round(pos_in_round, picks_per_round)
                scaled_val = raw_val * dp_scale
                key = (year, rnd, bucket)
                dp_vals[key].append(scaled_val)

    # ------------------ Merge FC + DP (already in 0–999 range) ------------------

    all_keys = set(fc_vals.keys()) | set(dp_vals.keys())
    if not all_keys:
        print("[load_pick_value_table] no pick values found from either CSV")
        return {}

    # Normalize weights if only one source present
    if not fc_vals:
        w_fc, w_dp = 0.0, 1.0
    elif not dp_vals:
        w_fc, w_dp = 1.0, 0.0

    final: Dict[str, float] = {}

    for (year, rnd, bucket) in sorted(all_keys):
        fc_list = fc_vals.get((year, rnd, bucket), [])
        dp_list = dp_vals.get((year, rnd, bucket), [])

        fc_val = sum(fc_list) / len(fc_list) if fc_list else None
        dp_val = sum(dp_list) / len(dp_list) if dp_list else None

        if fc_val is not None and dp_val is not None:
            val = w_fc * fc_val + w_dp * dp_val
        elif fc_val is not None:
            val = fc_val
        elif dp_val is not None:
            val = dp_val
        else:
            continue

        key_str = f"{year}_{rnd}{bucket}"
        final[key_str] = round(float(val), 1)
    return final
