# picks.py

from __future__ import annotations

import json
import glob
from datetime import date
from typing import Dict

from utils.paths import DATA_DIR


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
            for key, val in wls_1qb.items():
                if not val or float(val) <= 0:
                    continue
                parts = key.split("_")
                try:
                    if len(parts) >= 2 and int(parts[1]) > _MAX_ROUND:
                        continue
                except (ValueError, IndexError):
                    pass
                final[key] = float(val)
        except Exception:
            pass

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
            pass

    # Apply normalization scale
    try:
        _scale_path = DATA_DIR / "market_calibration_scale.json"
        if _scale_path.exists():
            _scale = float(json.loads(_scale_path.read_text()).get("scale_1qb", 1.0))
            if _scale and _scale != 1.0:
                final = {k: round(v * _scale, 1) for k, v in final.items()}
    except Exception:
        pass

    return final
