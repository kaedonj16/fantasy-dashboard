"""Standalone opt-in trade contribution job.

Pulls completed trades from a user's / account's leagues into the
trade-intel pool (``data_building.trade_intel.contrib``). Nothing is
contributed unless the id has explicitly opted in.

Two id spaces:

* ``--user-id``: a Sleeper user id (the original path). Contributes that
  user's Sleeper leagues via ``contribute_user_leagues``.
* ``--account-id``: an internal account id. Contributes the account's
  CONNECTED ESPN / Yahoo / MFL / Fleaflicker leagues via
  ``contribute_account_leagues`` (credentials resolve from the stored
  league links; this is the only path that reaches non-Sleeper platforms).

Usage
-----
    # Sleeper user: opt in, then contribute
    python scripts/contribute_trades.py --user-id 123456 --opt-in
    python scripts/contribute_trades.py --user-id 123456

    # Account: opt in, then contribute (optionally narrow the platforms)
    python scripts/contribute_trades.py --account-id 42 --opt-in
    python scripts/contribute_trades.py --account-id 42 --providers espn,yahoo

    # Opt back out
    python scripts/contribute_trades.py --account-id 42 --opt-out
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Running `python scripts/contribute_trades.py` puts scripts/ on sys.path, not
# the project root, so `import data_building` fails with ModuleNotFoundError.
# Add the repo root (parent of scripts/) explicitly so imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Opt-in trade contribution into the trade-intel pool."
    )
    id_group = parser.add_mutually_exclusive_group(required=True)
    id_group.add_argument("--user-id", help="Sleeper user id (Sleeper leagues only).")
    id_group.add_argument("--account-id", type=int,
                          help="Internal account id (connected ESPN/Yahoo/MFL/Fleaflicker leagues).")
    consent = parser.add_mutually_exclusive_group()
    consent.add_argument("--opt-in", action="store_true",
                         help="Record opt-in for the given id and stop.")
    consent.add_argument("--opt-out", action="store_true",
                         help="Record opt-out for the given id and stop.")
    parser.add_argument("--season", type=int, default=None,
                        help="Season to contribute (default: current season).")
    parser.add_argument("--providers", default=None,
                        help="Comma-separated provider list (default: sleeper for "
                             "--user-id, espn,yahoo,mfl,fleaflicker for --account-id).")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    from data_building.trade_intel import contrib

    providers = (
        [p.strip().lower() for p in args.providers.split(",") if p.strip()]
        if args.providers else None
    )

    if args.user_id:
        if args.opt_in or args.opt_out:
            contrib.set_opt_in(args.user_id, opted_in=args.opt_in)
            logger.info("user_id=%s opt_in=%s recorded.", args.user_id, args.opt_in)
            return 0
        result = contrib.contribute_user_leagues(
            args.user_id,
            season=args.season,
            providers=providers or (contrib.PROVIDER_SLEEPER,),
        )
    else:
        if args.opt_in or args.opt_out:
            contrib.set_account_opt_in(args.account_id, opted_in=args.opt_in)
            logger.info("account_id=%s opt_in=%s recorded.", args.account_id, args.opt_in)
            return 0
        result = contrib.contribute_account_leagues(
            args.account_id,
            season=args.season,
            providers=providers or contrib.ACCOUNT_PROVIDERS,
        )

    logger.info("Contribution complete: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
