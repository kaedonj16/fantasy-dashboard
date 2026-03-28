# dashboard_services/value_exports.py
import csv
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from data_building.player_value import build_value_table_for_usage
from utils.utils import load_relevant_index

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ENGINE_VALUES_CSV = DATA_DIR / f"engine_values_{date.today().isoformat()}.csv"

LEAGUE_SIZES = [8, 10, 12, 14]


def _is_blank(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False


def export_engine_values(out_csv: Path = ENGINE_VALUES_CSV) -> None:
    players_index = load_relevant_index() or {}

    # Run the engine model for each league size and each format (1QB + Superflex).
    # The 10-team run also generates confidence scores.
    engine_by_size: dict[int, dict[str, float]] = {}
    sf_engine_by_size: dict[int, dict[str, float]] = {}
    confidence_table: dict[str, Any] = {}

    for n in LEAGUE_SIZES:
        if n == 10:
            vt, conf = build_value_table_for_usage(league_type="1QB", include_confidence=True, num_teams=n)
            confidence_table = conf or {}
        else:
            vt = build_value_table_for_usage(league_type="1QB", include_confidence=False, num_teams=n)
        engine_by_size[n] = vt or {}
        sf_engine_by_size[n] = build_value_table_for_usage(league_type="Superflex", num_teams=n) or {}

    rows = []
    skipped = {
        "missing_meta": 0,
        "blank_name": 0,
        "blank_position": 0,
        "blank_team": 0,
        "blank_value": 0,
    }

    # Union of all player IDs across all runs
    all_pids: set[str] = set()
    for vt in engine_by_size.values():
        all_pids |= set(vt.keys())
    for vt in sf_engine_by_size.values():
        all_pids |= set(vt.keys())

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

        # Check at least the default 10-team value exists
        if engine_by_size[10].get(pid) is None and sf_engine_by_size[10].get(pid) is None:
            skipped["blank_value"] += 1
            continue

        row: dict[str, Any] = {
            "player_id": str(pid),
            "name": str(name).strip(),
            "position": str(position).strip(),
            "team": str(team).strip(),
            # Default 10-team columns kept for backward compatibility
            "engine_value": round(float(engine_by_size[10].get(pid) or 0.0), 1),
            "sf_engine_value": round(float(sf_engine_by_size[10].get(pid) or 0.0), 1),
            "value_confidence": confidence_table.get(str(pid), confidence_table.get(pid, "")),
        }

        # Per-league-size columns
        for n in LEAGUE_SIZES:
            row[f"engine_value_{n}"] = round(float(engine_by_size[n].get(pid) or 0.0), 1)
            row[f"sf_engine_value_{n}"] = round(float(sf_engine_by_size[n].get(pid) or 0.0), 1)

        rows.append(row)

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

    fieldnames = [
        "player_id", "name", "position", "team",
        "engine_value", "sf_engine_value", "value_confidence",
    ]
    for n in LEAGUE_SIZES:
        fieldnames += [f"engine_value_{n}", f"sf_engine_value_{n}"]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[engine_values] Wrote {len(rows)} players to {out_csv.name} "
          f"(league sizes: {LEAGUE_SIZES})")


if __name__ == "__main__":
    export_engine_values()
