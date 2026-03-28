# dashboard_services/value_exports.py
import csv
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from data_building.player_value import build_value_table_for_usage
from utils.utils import load_relevant_index

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ENGINE_VALUES_CSV = DATA_DIR / f"engine_values_{date.today().isoformat()}.csv"


def _is_blank(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False


def export_engine_values(out_csv: Path = ENGINE_VALUES_CSV) -> None:
    players_index = load_relevant_index() or {}

    # Generate both 1QB and Superflex engine values, with confidence scores from the 1QB run
    value_table_1qb, confidence_table = build_value_table_for_usage(league_type="1QB", include_confidence=True)
    value_table_1qb = value_table_1qb or {}
    confidence_table = confidence_table or {}
    value_table_sf = build_value_table_for_usage(league_type="Superflex") or {}

    rows = []
    skipped = {
        "missing_meta": 0,
        "blank_name": 0,
        "blank_position": 0,
        "blank_team": 0,
        "blank_value": 0,
    }

    # Combine both value tables
    all_pids = set(value_table_1qb.keys()) | set(value_table_sf.keys())

    for pid in all_pids:
        meta = players_index.get(str(pid), {}) or {}
        if not meta:
            skipped["missing_meta"] += 1
            continue

        name = meta.get("name") or ""
        position = meta.get("pos") or meta.get("position") or ""
        team = meta.get("team") or ""

        if _is_blank(name):
            skipped["blank_name"] += 1
            continue
        if _is_blank(position):
            skipped["blank_position"] += 1
            continue
        if _is_blank(team):
            skipped["blank_team"] += 1
            continue

        val_1qb = value_table_1qb.get(pid)
        val_sf = value_table_sf.get(pid)

        if val_1qb is None and val_sf is None:
            skipped["blank_value"] += 1
            continue

        try:
            engine_value_1qb = float(val_1qb) if val_1qb is not None else 0.0
            engine_value_sf = float(val_sf) if val_sf is not None else 0.0
        except Exception:
            skipped["blank_value"] += 1
            continue

        rows.append(
            {
                "player_id": str(pid),
                "name": str(name).strip(),
                "position": str(position).strip(),
                "team": str(team).strip(),
                "engine_value": round(engine_value_1qb, 1),
                "sf_engine_value": round(engine_value_sf, 1),
                "value_confidence": confidence_table.get(str(pid), confidence_table.get(pid, "")),
            }
        )

    rows.sort(key=lambda r: (r["position"], -r["engine_value"], r["name"]))

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    today = date.today()
    yesterday = today - timedelta(days=1)

    stem = out_csv.stem
    suffix = out_csv.suffix or ".csv"

    yesterday_file = None
    try:
        tail = stem.split("_")[-1]
        date.fromisoformat(tail)
        base = "_".join(stem.split("_")[:-1])
        yesterday_file = out_csv.parent / f"{base}_{yesterday.isoformat()}{suffix}"
    except Exception:
        yesterday_file = None

    if yesterday_file and yesterday_file.exists():
        print(f"[engine_values] Removing yesterday's value file: {yesterday_file.name}")
        try:
            yesterday_file.unlink()
        except Exception as e:
            print(f"[engine_values] Failed to remove yesterday's file: {e}")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["player_id", "name", "position", "team", "engine_value", "sf_engine_value", "value_confidence"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    export_engine_values()
