# dashboard_services/value_exports.py
import csv
from datetime import date, timedelta
from pathlib import Path

from dashboard_services.utils import load_relevant_index
from data_building.player_value import build_value_table_for_usage

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ENGINE_VALUES_CSV = DATA_DIR / f"engine_values_{date.today().isoformat()}.csv"

import csv
from datetime import date, timedelta
from pathlib import Path


def export_engine_values(out_csv: Path = ENGINE_VALUES_CSV) -> None:
    players_index = load_relevant_index()
    value_table = build_value_table_for_usage()

    rows = []
    for pid, val in value_table.items():
        meta = players_index.get(str(pid), {}) or {}
        if not meta:
            continue

        rows.append(
            {
                "player_id": str(pid),
                "name": meta.get("name") or "",
                "position": meta.get("pos") or meta.get("position") or "",
                "team": meta.get("team") or "",
                "engine_value": val,
            }
        )

    # Stable output ordering (nice for git / debugging)
    rows.sort(key=lambda r: (r["position"], r["engine_value"]), reverse=True)

    out_csv = Path(out_csv)
    dirname = out_csv.parent
    dirname.mkdir(parents=True, exist_ok=True)

    # If your out_csv is dated like engine_values_YYYY-MM-DD.csv,
    # remove yesterday's version in the same directory.
    today = date.today()
    yesterday = today - timedelta(days=1)

    stem = out_csv.stem  # e.g. "engine_values_2025-12-04" OR "engine_values"
    suffix = out_csv.suffix or ".csv"

    # Detect and replace a trailing ISO date in the stem.
    # If no date is present, we won't try to delete anything.
    yesterday_file = None
    try:
        tail = stem.split("_")[-1]
        # If tail looks like YYYY-MM-DD, treat as dated file pattern
        date.fromisoformat(tail)
        base = "_".join(stem.split("_")[:-1])  # "engine_values"
        yesterday_file = dirname / f"{base}_{yesterday.isoformat()}{suffix}"
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

    print(f"[engine_values] Wrote {len(rows)} rows -> {out_csv}")


if __name__ == '__main__':
    export_engine_values()
