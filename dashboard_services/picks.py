# picks.py

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, Tuple, List
from pathlib import Path
from datetime import date

import pandas as pd

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
        return "early"
    if s.startswith("m"):
        return "mid"
    if s.startswith("l"):
        return "late"
    return s


def _bucket_for_pick_in_round(pos_in_round: int, picks_per_round: int = 12) -> str:
    if picks_per_round <= 0:
        return "mid"
    third = picks_per_round / 3.0
    if pos_in_round <= math.ceil(third):
        return "early"
    elif pos_in_round <= math.ceil(2 * third):
        return "mid"
    else:
        return "late"


def load_pick_value_table(
    fantasycalc_csv: Path = FANTASYCALC_VALUES_PATH,
    dynastyprocess_csv: Path = DYNASTYPROCESS_VALUES_PATH,
    picks_per_round: int = 12,
    w_fc: float = 0.5,
    w_dp: float = 0.5,
    current_year: int | None = None,
) -> Dict[str, float]:
    """
    Build a draft pick value table by merging FantasyCalc + DynastyProcess.

    Returns values on the SAME 0–800-ish scale as your player model,
    anchored so that an early 1st ends up around mid-tier (~540).

    Keys look like:
      "2026_1_early", "2027_2_mid", "2028_3_late", ...
    """

    if current_year is None:
        current_year = date.today().year

    # We'll explicitly keep only current_year+1 .. current_year+3
    allowed_years = {current_year + 1, current_year + 2, current_year + 3}

    fc_vals: Dict[Tuple[int, int, str], List[float]] = defaultdict(list)
    dp_vals: Dict[Tuple[int, int, str], List[float]] = defaultdict(list)

    # ------------------ FantasyCalc ------------------
    try:
        df_fc = pd.read_csv(fantasycalc_csv)
    except FileNotFoundError:
        df_fc = pd.DataFrame()

    if not df_fc.empty:
        # only rows where FC says it's a pick
        df_fc_picks = df_fc[df_fc["position"].astype(str).str.upper() == "PICK"].copy()

        if "name" not in df_fc_picks.columns:
            print("[load_pick_value_table] FantasyCalc: 'name' column not found, skipping")
        else:
            # pick a numeric value column
            value_col = None
            for cand in ["value", "combined_value", "fc_value", "points", "score"]:
                if cand in df_fc_picks.columns:
                    value_col = cand
                    break

            if not value_col:
                print("[load_pick_value_table] FantasyCalc: no numeric value column, skipping")
            else:
                # full format: "2026 1st (Early)"
                name_re_full = re.compile(
                    r"(?P<year>\d{4})\s+"
                    r"(?P<round>\d+)(?:st|nd|rd|th)"
                    r"\s*\((?P<bucket>Early|Mid|Late)\)",
                    re.IGNORECASE,
                )
                # simpler format: "2027 1st"  (no bucket -> we will treat as 'mid')
                name_re_simple = re.compile(
                    r"(?P<year>\d{4})\s+"
                    r"(?P<round>\d+)(?:st|nd|rd|th)\b",
                    re.IGNORECASE,
                )

                for _, row in df_fc_picks.iterrows():
                    raw_name = str(row["name"])

                    m = name_re_full.search(raw_name)
                    bucket = None
                    if m:
                        year = int(m.group("year"))
                        rnd = int(m.group("round"))
                        bucket = _normalize_bucket_label(m.group("bucket"))
                    else:
                        m2 = name_re_simple.search(raw_name)
                        if not m2:
                            continue
                        year = int(m2.group("year"))
                        rnd = int(m2.group("round"))
                        # no explicit bucket -> treat as mid for that round
                        bucket = "mid"

                    try:
                        val = float(row[value_col])
                    except Exception:
                        continue

                    # --- FILTERS: only 2026–2028 and rounds 1–3 ---
                    if year not in allowed_years:
                        continue
                    if rnd not in (1, 2, 3):
                        continue

                    key = (year, rnd, bucket)
                    fc_vals[key].append(val)

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
            # keep only rows that are picks
            df_dp_picks = df_dp[df_dp[pos_col].astype(str).str.upper() == "PICK"].copy()

            value_col = None
            for cand in ["value_1qb", "value", "dp_value", "points", "score"]:
                if cand in df_dp_picks.columns:
                    value_col = cand
                    break

            if not value_col:
                print("[load_pick_value_table] DynastyProcess: no numeric value column, skipping")
            else:
                # Example DP string: "2025 Pick 1.01"
                name_re = re.compile(
                    r"(?P<year>\d{4})\s+Pick\s+(?P<round>\d+)\.(?P<pos_in_round>\d+)",
                    re.IGNORECASE,
                )

                for _, row in df_dp_picks.iterrows():
                    raw_name = str(row[name_col])
                    m = name_re.search(raw_name)
                    if not m:
                        continue

                    try:
                        year = int(m.group("year"))
                        rnd = int(m.group("round"))
                        pos_in_round = int(m.group("pos_in_round"))
                        val = float(row[value_col])
                    except Exception:
                        continue

                    # --- FILTERS: only 2026–2028 and rounds 1–3 ---
                    if year not in allowed_years:
                        continue
                    if rnd not in (1, 2, 3):
                        continue

                    bucket = _bucket_for_pick_in_round(pos_in_round, picks_per_round)
                    key = (year, rnd, bucket)
                    dp_vals[key].append(val)

    # ------------------ Merge into RAW pick values ------------------
    raw_by_key: Dict[Tuple[int, int, str], float] = {}

    all_keys = set(fc_vals.keys()) | set(dp_vals.keys())
    if not all_keys:
        print("[load_pick_value_table] no pick values found from either CSV")
        return {}

    # Normalize weights if only one source present
    if not fc_vals:
        w_fc, w_dp = 0.0, 1.0
    elif not dp_vals:
        w_fc, w_dp = 1.0, 0.0

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

        raw_by_key[(year, rnd, bucket)] = float(val)

    if not raw_by_key:
        return {}

    # ------------------ Anchor-based normalization into 0–800-ish ------------------

    # 1) Find the raw value of "early 1st" picks across allowed years
    early_first_vals = [
        v for (year, rnd, bucket), v in raw_by_key.items()
        if rnd == 1 and bucket == "early"
    ]

    if early_first_vals:
        anchor_raw = sum(early_first_vals) / len(early_first_vals)
    else:
        all_raw_vals = list(raw_by_key.values())
        anchor_raw = float(pd.Series(all_raw_vals).median())

    ANCHOR_TARGET = 540.0  # map early 1sts to ~540

    if anchor_raw <= 0:
        scale = 0.1
    else:
        scale = ANCHOR_TARGET / anchor_raw

    normalized: Dict[str, float] = {}
    MAX_PICK_VALUE = 800.0  # cap so picks don't outrank elite elite players

    for (year, rnd, bucket), raw in raw_by_key.items():
        scaled = raw * scale
        scaled = max(0.0, min(scaled, MAX_PICK_VALUE))
        key_str = f"{year}_{rnd}_{bucket}"
        normalized[key_str] = round(scaled, 1)

    # Helpful debug – you can leave / remove this as you like
    years_seen = sorted({y for (y, _, _) in raw_by_key.keys()})
    print("[load_pick_value_table] years in pick table:", years_seen)

    return normalized
