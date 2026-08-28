"""I/O: parquet warehouse → small JSON profile aggregates.

Request paths must read ``historical_profile_aggregates.json``, never scan
parquet per player. Rebuild from cron after the warehouse step.
Phase 2–9: rates, comps, ADP, signals, board profiles, walk-forward verdict.
"""
from __future__ import annotations

import json
import math
from typing import Any, Optional

from utils.paths import PLAYER_HISTORY_DIR

from dashboard_services.historical.career_profiles import assemble_profile_aggregates
from data_building.external_data.player_history import load_player_history_df
from data_building.historical.build_adp import attach_historical_adp

PROFILE_PATH = PLAYER_HISTORY_DIR / "historical_profile_aggregates.json"


def _to_native(value: Any) -> Any:
    if value is None:
        return None
    try:
        import pandas as pd
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            return _to_native(value.item())
        except (ValueError, AttributeError):
            pass
    return value


def records_from_warehouse(season: Optional[int] = None) -> list[dict]:
    """Load precomputed parquet into plain dicts (NaN → None)."""
    df = load_player_history_df(season)
    if df is None or getattr(df, "empty", True):
        return []
    records = []
    for raw in df.to_dict(orient="records"):
        records.append({key: _to_native(val) for key, val in raw.items()})
    return records


def rebuild_historical_profiles(
    rows: Optional[list[dict]] = None,
    *,
    write: bool = True,
    scoring: str = "ppr",
    attach_adp: Optional[bool] = None,
) -> dict:
    """Assemble Phase 2–9 aggregates. ``write=False`` for tests.

    ADP is joined from committed frozen snapshots when loading the warehouse.
    Injected test rows skip that join unless ``attach_adp=True``.
    """
    records = rows if rows is not None else records_from_warehouse()
    if attach_adp is None:
        attach_adp = rows is None
    if attach_adp:
        records = attach_historical_adp(records)
    payload = assemble_profile_aggregates(records, scoring=scoring)
    if write:
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROFILE_PATH.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        print(
            f"[historical] profiles {payload.get('n_player_seasons')} seasons "
            f"{payload.get('season_range')} → {PROFILE_PATH}"
        )
    payload["written_path"] = str(PROFILE_PATH) if write else None
    return payload


def rebuild_cohort_index_overlay(
    rows: Optional[list[dict]] = None,
    *,
    write: bool = True,
) -> dict:
    """Write the compact observation index without rewriting profile rates.

    Used when the warehouse parquet exists (or rows are injected) but a full
    profile rebuild is not required. Request paths still never scan parquet.
    """
    from dashboard_services.historical.cohorts import build_cohort_index

    overlay_path = PLAYER_HISTORY_DIR / "cohort_index.json"
    records = rows if rows is not None else records_from_warehouse()
    if records:
        records = attach_historical_adp(records)
    index = build_cohort_index(records)
    if write:
        overlay_path.write_text(
            json.dumps(index, separators=(",", ":"), default=str),
            encoding="utf-8",
        )
        print(
            f"[historical] cohort index {index.get('n')} observations "
            f"({index.get('n_with_trajectory')} with trajectory) → {overlay_path}"
        )
    index["written_path"] = str(overlay_path) if write else None
    return index


def load_historical_profiles() -> dict:
    """Request-path reader: precomputed JSON only."""
    if not PROFILE_PATH.exists():
        return {}
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    rebuild_historical_profiles()
