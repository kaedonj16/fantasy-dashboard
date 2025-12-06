# dashboard_services/value_exports.py
import csv
from datetime import date, timedelta
from pathlib import Path

from dashboard_services.data_building.player_value import build_value_table_for_usage
from dashboard_services.utils import load_relevant_index

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ENGINE_VALUES_CSV = DATA_DIR / f"engine_values_{date.today().isoformat()}.csv"


def export_engine_values(
        out_csv: Path = ENGINE_VALUES_CSV,
) -> None:
    players_index = load_relevant_index()
    value_table = build_value_table_for_usage()

    rows = []
    for pid, val in value_table.items():
        meta = players_index.get(pid, {})
        rows.append(
            {
                "player_id": pid,
                "name": meta.get("name"),
                "position": meta.get("pos") or meta.get("position"),
                "team": meta.get("team"),
                "engine_value": val,
            }
        )

    today = date.today()
    yesterday = today - timedelta(days=1)

    # Example file naming:
    # values_2025-12-04.csv
    # override if you use static file names
    dirname = out_csv.parent
    pattern = f"engine_values_{yesterday.isoformat()}.csv"
    yesterday_file = dirname / pattern

    if yesterday_file.exists():
        print(f"[engine_values] Removing yesterday's value file: {yesterday_file.name}")
        try:
            yesterday_file.unlink()
        except Exception as e:
            print(f"[engine_values] Failed to remove yesterday's file: {e}")

    dirname.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["player_id", "name", "position", "team", "engine_value"],
        )
        writer.writeheader()
        writer.writerows(rows)



if __name__ == '__main__':
    export_engine_values()
