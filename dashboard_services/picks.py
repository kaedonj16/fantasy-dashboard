# picks.py

from __future__ import annotations

import json
import glob
import logging
from datetime import date
from typing import Dict

from utils.paths import DATA_DIR

logger = logging.getLogger(__name__)


def load_pick_value_table(
    league_teams: int = 10,
    current_year: int | None = None,
    **_kwargs,
) -> Dict[str, float]:
    """
    Build a draft pick value table from WLS trade data.

    Reads from pick_values_wls_latest.json (slot picks like 2026_1_01
    and bucket picks like 2026_1_early) and normalizes with
    market_calibration_scale.json.

    Keys: "{season}_{round}_{slot_or_bucket}"
      e.g. "2026_1_01", "2026_1_early", "2027_2_mid"

    Values are in the same normalized 0–999.9 scale as player values.
    """
    if current_year is None:
        current_year = date.today().year

    final: Dict[str, float] = {}

    # Primary source: WLS latest file (has both slot and bucket picks)
    wls_path = DATA_DIR / "pick_values_wls_latest.json"
    if not wls_path.exists():
        # Fall back to most recent dated file
        candidates = sorted(glob.glob(str(DATA_DIR / "pick_values_wls_*.json")), reverse=True)
        candidates = [c for c in candidates if "latest" not in c]
        if candidates:
            wls_path = candidates[0]

    # Dynasty rookie drafts are at most 5 rounds; cap to avoid noise from
    # later rounds (WLS tracks up to 50 rounds of slot data).
    _MAX_ROUND = 5

    if wls_path.exists():
        try:
            wls_data = json.loads(wls_path.read_text())
            wls_1qb = wls_data.get("1qb", {})
            wls_final: Dict[str, float] = {}
            for key, val in wls_1qb.items():
                if not val or float(val) <= 0:
                    continue
                parts = key.split("_")
                # Skip bare YYYY_R keys (no slot/bucket suffix)
                if len(parts) < 3:
                    continue
                try:
                    if int(parts[1]) > _MAX_ROUND:
                        continue
                except (ValueError, IndexError):
                    pass
                wls_final[key] = float(val)
            if wls_final:
                # Enforce monotonic ordering for slot picks: within each (year, round),
                # slot picks must decrease in value as slot number increases.
                # WLS data can have noise that makes e.g. 1.11 > 1.02 which is impossible.
                from collections import defaultdict
                slot_groups: Dict[str, list] = defaultdict(list)
                bucket_entries: Dict[str, float] = {}
                for key, val in wls_final.items():
                    parts = key.split("_")
                    try:
                        int(parts[2])  # slot picks have numeric third part
                        slot_groups[f"{parts[0]}_{parts[1]}"].append((int(parts[2]), key, val))
                    except (ValueError, IndexError):
                        bucket_entries[key] = val

                fixed: Dict[str, float] = {}
                for group_key, slots in slot_groups.items():
                    slots.sort(key=lambda x: x[0])  # sort by slot number
                    running_max = float("inf")
                    for slot_num, key, val in slots:
                        capped = min(val, running_max)
                        fixed[key] = round(capped, 1)
                        running_max = capped
                fixed.update(bucket_entries)
                return fixed
        except Exception:
            logger.warning("picks: failed to build pick table from market calibration file", exc_info=True)

    # Fallback: DB bucket picks (trade_intel_pick_stats)
    if not final:
        try:
            from dashboard_services.db import get_conn
            with get_conn() as conn:
                rows = conn.execute("""
                    SELECT pick_season, pick_round, pick_bucket,
                           weighted_market_value_1qb
                    FROM trade_intel_pick_stats
                    WHERE season = (
                        SELECT MAX(season) FROM trade_intel_pick_stats
                        WHERE trade_count >= 10
                    )
                      AND trade_count >= 10
                      AND weighted_market_value_1qb IS NOT NULL
                    ORDER BY pick_season, pick_round, pick_bucket
                """).fetchall()
            for r in rows:
                key = f"{r['pick_season']}_{r['pick_round']}_{r['pick_bucket']}"
                final[key] = float(r['weighted_market_value_1qb'])
        except Exception:
            logger.warning("picks: failed to load pick values from trade_intel_pick_stats", exc_info=True)

    # Apply normalization scale
    try:
        _scale_path = DATA_DIR / "market_calibration_scale.json"
        if _scale_path.exists():
            _scale = float(json.loads(_scale_path.read_text()).get("scale_1qb", 1.0))
            if _scale and _scale != 1.0:
                final = {k: round(v * _scale, 1) for k, v in final.items()}
    except Exception:
        logger.warning("picks: failed to apply normalization scale", exc_info=True)

    return final
