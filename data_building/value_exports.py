# dashboard_services/value_exports.py
import csv
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from utils.utils import load_relevant_index
from data_building.player_value import build_value_table_for_usage

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
    value_table = build_value_table_for_usage() or {}

    print(f"[engine_values] relevant index size: {len(players_index)}")
    print(f"[engine_values] value table size: {len(value_table)}")

    rows = []
    skipped = {
        "missing_meta": 0,
        "blank_name": 0,
        "blank_position": 0,
        "blank_team": 0,
        "blank_value": 0,
    }

    for pid, val in value_table.items():
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
        if val is None:
            skipped["blank_value"] += 1
            continue

        try:
            engine_value = float(val)
        except Exception:
            skipped["blank_value"] += 1
            continue

        rows.append(
            {
                "player_id": str(pid),
                "name": str(name).strip(),
                "position": str(position).strip(),
                "team": str(team).strip(),
                "engine_value": round(engine_value, 1),
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
            fieldnames=["player_id", "name", "position", "team", "engine_value"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[engine_values] skipped={skipped}")
    print(f"[engine_values] Wrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    export_engine_values()
