#!/usr/bin/env python3
"""Retroactively match ScoreZone TD clips for finished games (one-off runs).

The live clip pass inside the store poll only matches a TD within minutes
of the play, so a game that finished before the matcher existed (or while
the poller was down) can never gain Watch buttons. This backfill rebuilds
each game's wanted TDs from ESPN's archived summary (which still carries
per-play wallclock), reads the official accounts' tweets by time range,
runs the exact same precision gate, and stores the matches in
redzone_play_clips. Safe to re-run: plays that already have a clip are
skipped and the store upsert is first-match-wins.

Usage:
    X_BEARER_TOKEN=... python scripts/backfill_scorezone_clips.py \
        --date 2026-10-29 [--dry-run]
    X_BEARER_TOKEN=... python scripts/backfill_scorezone_clips.py \
        --game 20261029_CAR@GB

X reads are pay-per-use ($0.005/post); the run prints its tweets-read
total and the estimated read cost. Exit 2 when X_BEARER_TOKEN is unset,
1 when no games resolve or every game fails, 0 otherwise.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# `python scripts/backfill_scorezone_clips.py` puts scripts/ on sys.path,
# not the repo root, so the project packages don't import. Add the repo
# root explicitly, matching the other cron scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("backfill-scorezone-clips")


def _season_for_date(yyyymmdd: str) -> int:
    """NFL season for a game date: Jan/Feb games belong to the prior year."""
    year = int(yyyymmdd[:4])
    month = int(yyyymmdd[4:6])
    return year - 1 if month <= 2 else year


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill ScoreZone TD clips for finished games.")
    parser.add_argument(
        "--date", action="append", default=[], metavar="YYYY-MM-DD",
        help="backfill every final game on this date (repeatable)")
    parser.add_argument(
        "--game", action="append", default=[], metavar="GAME_ID",
        help="payload game id, e.g. 20261029_CAR@GB (repeatable)")
    parser.add_argument(
        "--season", type=int, default=None,
        help="season key for the clip store (default: derived from the "
             "first target's date)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="match and report only; write nothing")
    args = parser.parse_args(argv)

    bearer = os.environ.get("X_BEARER_TOKEN") or ""
    if not bearer:
        print("[backfill-clips] X_BEARER_TOKEN is not set; cannot read X. "
              "Set it in the environment and re-run.", file=sys.stderr)
        return 2

    from utils.scorezone_clips import (
        XClipFetcher,
        games_final_on_date,
        run_clip_backfill,
    )

    game_ids: list[str] = list(args.game)
    for date_str in args.date:
        try:
            game_ids.extend(games_final_on_date(date_str))
        except Exception as exc:
            logger.warning(
                "[backfill-clips] date lookup failed %s: %s", date_str, exc)
    # Dedupe, keep order.
    seen: set[str] = set()
    game_ids = [g for g in game_ids if not (g in seen or seen.add(g))]
    if not game_ids:
        print("[backfill-clips] no final games resolved from --date/--game.",
              file=sys.stderr)
        return 1

    season = args.season
    if season is None:
        first_date = game_ids[0].split("_", 1)[0]
        season = _season_for_date(first_date) if first_date.isdigit() else 0
    if not season:
        print("[backfill-clips] could not derive a season; pass --season.",
              file=sys.stderr)
        return 1

    try:
        from dashboard_services.api import get_nfl_players, get_nfl_state

        nfl_players = get_nfl_players() or {}
        week = int((get_nfl_state() or {}).get("week") or 1)
    except Exception as exc:
        logger.warning("[backfill-clips] player index load failed: %s", exc)
        nfl_players, week = {}, 1
    if not nfl_players:
        logger.warning(
            "[backfill-clips] WARNING: empty player index; no TD scorers "
            "can be named, so nothing will match")

    stats = run_clip_backfill(
        season,
        game_ids,
        fetcher=XClipFetcher(bearer),
        nfl_players=nfl_players,
        dry_run=args.dry_run,
        week=week,
    )
    failed = 0
    for g in stats["games"]:
        if g["error"]:
            failed += 1
        logger.info(
            "[backfill-clips] game=%s wanted=%s tweets=%s matched=%s "
            "stored=%s%s",
            g["game_id"], g["wanted"], g["tweets"], g["matched"],
            g["stored"], f" error={g['error']}" if g["error"] else "",
        )
    logger.info(
        "[backfill-clips] done: games=%s wanted=%s tweets_read=%s "
        "matched=%s stored=%s est_read_cost=$%.2f%s",
        len(stats["games"]), stats["wanted"], stats["tweets"],
        stats["matched"], stats["stored"], stats["read_cost_usd"],
        " (dry run, nothing written)" if args.dry_run else "",
    )
    return 1 if failed and failed == len(stats["games"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
