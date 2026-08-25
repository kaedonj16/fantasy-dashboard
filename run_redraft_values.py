"""Safely preview or write trade-calibrated true-redraft values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    from data_building.trade_intel.league_types import LeagueType
    from data_building.trade_intel.trade_value_model import (
        MIN_REDRAFT_NATIVE_TRADES,
        _detect_season,
        run_trade_value_model,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int)
    parser.add_argument("--league-size", type=int, choices=(8, 10, 12, 14), action="append")
    parser.add_argument("--write", action="store_true", help="Write qualified segments; default is dry-run")
    parser.add_argument("--format", choices=("1qb", "sf", "both"), default="both")
    parser.add_argument("--refresh-priors", action="store_true")
    parser.add_argument("--minimum-native-trades", type=int, default=MIN_REDRAFT_NATIVE_TRADES)
    parser.add_argument("--json-report", type=Path)
    args = parser.parse_args()

    if args.refresh_priors:
        from data_building.update_player_values_with_rankings import update_player_values_with_rankings
        print(f"Refreshed {update_player_values_with_rankings()} player values")

    season = args.season or _detect_season()
    results = []
    write_formats = ("1qb", "sf") if args.format == "both" else (args.format,)
    for league_size in args.league_size or [10, 12]:
        result = run_trade_value_model(
            season=season,
            league_type=LeagueType.REDRAFT,
            league_size=league_size,
            dry_run=not args.write,
            min_native_trades=args.minimum_native_trades,
            write_formats=write_formats,
        )
        results.append(result)
        printable = {k: v for k, v in result.items() if k not in {"rows", "priors"}}
        print(json.dumps(printable, indent=2, default=str))

    if args.json_report:
        args.json_report.write_text(json.dumps(results, indent=2, default=str))
        print(f"Wrote report to {args.json_report}")


if __name__ == "__main__":
    main()
