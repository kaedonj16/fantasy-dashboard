# picks.py

from __future__ import annotations

from datetime import date
from typing import Dict

from utils.paths import DATA_DIR


def load_pick_value_table(
    league_teams: int = 10,
    current_year: int | None = None,
    **_kwargs,
) -> Dict[str, float]:
    """
    Build a draft pick value table from trade intel market data.

    Pick values are derived from actual trades using the same proportional
    attribution and decay-weighted median as player values (computed by
    data_building/trade_intel/analytics.py and stored in trade_intel_pick_stats).

    Keys: "{pick_season}_{pick_round}_{pick_bucket}"
      e.g. "2026_1_early", "2027_2_mid", "2026_3_late"

    Values are in the same normalized 0–999.9 scale as player values after
    market_calibration_scale.json is applied.
    """
    if current_year is None:
        current_year = date.today().year

    final: Dict[str, float] = {}

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

    # Apply the same normalization scale used for player market values.
    try:
        import json as _json
        _scale_path = DATA_DIR / "market_calibration_scale.json"
        if _scale_path.exists():
            _scale = float(_json.loads(_scale_path.read_text()).get("scale_1qb", 1.0))
            if _scale and _scale != 1.0:
                final = {k: round(v * _scale, 1) for k, v in final.items()}
    except Exception:
        pass

    return final
