# picks.py

from __future__ import annotations

import json
import glob
import logging
from datetime import date
from typing import Dict, Optional

from utils.paths import DATA_DIR

logger = logging.getLogger(__name__)


def _load_fc_slot_pick_values(current_year: int) -> Dict[str, float]:
    """
    Load FantasyCalc slot pick values for the current draft year, normalized
    to the model's 0-999.9 scale using the top-player ratio.

    Returns keys like "2026_1_01", values in model scale.
    """
    try:
        import pandas as pd
        import json as _json
        from pathlib import Path

        fc_path = DATA_DIR / "fantasycalc_api_values.csv"
        model_path = DATA_DIR / "model_values.json"
        if not fc_path.exists() or not model_path.exists():
            return {}

        fc = pd.read_csv(fc_path)
        model_data = _json.loads(model_path.read_text())

        # Model scale top value
        model_players = [p for p in model_data
                         if str(p.get("position", "")).upper() not in ("PICK", "")]
        if not model_players:
            return {}
        model_top = max(float(p.get("value") or 0) for p in model_players)
        if model_top <= 0:
            return {}

        # FC top player value (non-picks)
        fc_players = fc[~fc["name"].str.contains("Pick|1st|2nd|3rd|4th", na=False)]
        fc_top = float(fc_players["value"].max())
        if fc_top <= 0:
            return {}

        ratio = model_top / fc_top

        # Current-year slot picks from FC: sleeper_id "DP_R_S" = round R+1, slot S+1
        year_str = str(current_year)
        fc_slots = fc[
            fc["sleeper_id"].str.startswith("DP_", na=False) &
            fc["name"].str.contains(year_str, na=False)
        ]

        result: Dict[str, float] = {}
        for _, row in fc_slots.iterrows():
            sid = str(row["sleeper_id"])  # e.g. DP_0_0
            parts = sid.split("_")
            if len(parts) != 3:
                continue
            try:
                rnd = int(parts[1]) + 1
                slot = int(parts[2]) + 1
                key = f"{year_str}_{rnd}_{slot:02d}"
                result[key] = round(float(row["value"]) * ratio, 1)
            except (ValueError, TypeError):
                continue

        return result
    except Exception:
        return {}


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

                # Override current-year slot picks with FC-normalized values so
                # they are in the same 0-999.9 model scale as player values.
                # WLS uses its own trade-frequency units (top pick ~458) which
                # are not comparable to the model scale (top player ~999).
                fc_slots = _load_fc_slot_pick_values(current_year)
                if fc_slots:
                    for k, v in fc_slots.items():
                        if v > 0:
                            fixed[k] = v
                    # Re-enforce monotonicity after FC overlay
                    from collections import defaultdict as _dd2
                    slot_groups2: Dict[str, list] = _dd2(list)
                    for key, val in list(fixed.items()):
                        p2 = key.split("_")
                        try:
                            int(p2[2])
                            slot_groups2[f"{p2[0]}_{p2[1]}"].append((int(p2[2]), key, val))
                        except (ValueError, IndexError):
                            pass
                    for gk, sl in slot_groups2.items():
                        sl.sort(key=lambda x: x[0])
                        rm = float("inf")
                        for sn, k2, v2 in sl:
                            capped2 = min(v2, rm)
                            fixed[k2] = round(capped2, 1)
                            rm = capped2

                # Override future-year bucket picks with model_values.json values.
                # WLS bucket values are in WLS units (not 0-999.9 model scale);
                # model_values.json has pre-calibrated bucket picks for 2027+.
                _BUCKET_KW = {"early", "mid", "late"}
                try:
                    import json as _mj
                    _mp = DATA_DIR / "model_values.json"
                    if _mp.exists():
                        _mdata = _mj.loads(_mp.read_text())
                        for _mp_entry in _mdata:
                            if str(_mp_entry.get("position", "")).upper() != "PICK":
                                continue
                            _mid = str(_mp_entry.get("id") or "")
                            _mparts = _mid.split("_")
                            if len(_mparts) != 3:
                                continue
                            _mbkt = _mparts[2].lower()
                            if _mbkt not in _BUCKET_KW:
                                continue
                            try:
                                _myr = int(_mparts[0])
                            except ValueError:
                                continue
                            # Only override future years (not current-year — those
                            # have FC-normalized slot picks already)
                            if _myr <= current_year:
                                continue
                            _mv = float(_mp_entry.get("value") or 0)
                            if _mv > 0:
                                fixed[_mid] = _mv
                except Exception:
                    pass

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
