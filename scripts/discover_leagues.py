"""
Standalone league discovery job.

Expands the trade-intel league pool by BFS-crawling the Sleeper API from
known starting points (trending players → league IDs → owner user IDs →
their leagues). No trade crawling — discovery only.

Memory: run_discovery caps seeds, in-flight HTTP, and the BFS frontier
independently of --target, and stream-scans user-leagues/rosters instead of
parsing full JSON. Safe for the 512Mi starter cron; do not raise --target
thinking it controls RSS — it only caps how many leagues we insert.

Usage
-----
    python scripts/discover_leagues.py                  # discover up to 1000 new leagues
    python scripts/discover_leagues.py --target 500
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Running `python scripts/discover_leagues.py` puts scripts/ on sys.path, not the
# project root, so `import data_building` fails with ModuleNotFoundError. Add the
# repo root (parent of scripts/) explicitly so the package imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sleeper league discovery — expands the trade-intel pool.")
    parser.add_argument("--target", type=int, default=1000,
                        help="Max new leagues to discover this run. Default 1000.")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    from data_building.trade_intel.league_discovery import run_discovery

    logger.info("League discovery: targeting %d new leagues...", args.target)
    discovered = run_discovery(target=args.target)
    logger.info("League discovery complete: %d new leagues added.", discovered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
