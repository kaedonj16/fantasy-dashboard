"""
Standalone league discovery job.

Expands the trade-intel league pool by BFS-crawling the Sleeper API from
known starting points (trending players → league IDs → owner user IDs →
their leagues). No trade crawling — discovery only.

Usage
-----
    python scripts/discover_leagues.py                  # discover up to 2000 new leagues
    python scripts/discover_leagues.py --target 5000
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sleeper league discovery — expands the trade-intel pool.")
    parser.add_argument("--target", type=int, default=2000,
                        help="Max new leagues to discover this run. Default 2000.")
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
