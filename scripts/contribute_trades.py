"""
Opt-in trade contribution CLI (backend only).

Sets the per-user consent flag on trade_intel_users and/or contributes an
opted-in user's league trades to the trade-intel pool. Nothing runs unless
the user has explicitly opted in.

Usage
-----
    python scripts/contribute_trades.py --user-id <sleeper_user_id> --opt-in
    python scripts/contribute_trades.py --user-id <sleeper_user_id> --opt-out
    python scripts/contribute_trades.py --user-id <sleeper_user_id> --run
    python scripts/contribute_trades.py --user-id <sleeper_user_id> --opt-in --run
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Running `python scripts/contribute_trades.py` puts scripts/ on sys.path, not
# the project root, so `import data_building` fails with ModuleNotFoundError.
# Add the repo root (parent of scripts/) explicitly, like discover_leagues.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Opt-in trade contribution: manage consent and contribute "
                    "an opted-in user's league trades to the trade-intel pool."
    )
    parser.add_argument(
        "--user-id", required=True,
        help="trade_intel_users.user_id (Sleeper user id) to act on.",
    )
    consent = parser.add_mutually_exclusive_group()
    consent.add_argument(
        "--opt-in", action="store_true",
        help="Set contrib_opt_in=TRUE for this user.",
    )
    consent.add_argument(
        "--opt-out", action="store_true",
        help="Set contrib_opt_in=FALSE for this user.",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Contribute the user's leagues' trades (requires opt-in).",
    )
    args = parser.parse_args()

    if not (args.opt_in or args.opt_out or args.run):
        parser.error("nothing to do: pass --opt-in / --opt-out and/or --run")

    from dotenv import load_dotenv
    load_dotenv()

    from data_building.trade_intel.contrib import (
        contribute_user_leagues,
        set_opt_in,
    )

    if args.opt_in:
        set_opt_in(args.user_id, True)
        logger.info("Opted in user %s for trade contribution.", args.user_id)
    elif args.opt_out:
        set_opt_in(args.user_id, False)
        logger.info("Opted out user %s from trade contribution.", args.user_id)

    if args.run:
        result = contribute_user_leagues(args.user_id)
        print(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
