#!/usr/bin/env python3
"""
Live advanced-metrics refresh — rebuild the snapshot as soon as a game finishes.

The daily cron (``cron_daily.py``) builds the full Advanced Metrics pipeline once
a day. That means a Sunday slate does not reach the Advanced Metrics view until
the next morning's run, and MNF not until the day after. This script closes that
gap: run frequently on game days, it rebuilds the Sleeper-derived snapshot the
moment a new game goes final so the immediately-available stats (targets,
carries, receptions, yardage, red-zone usage, role scores, efficiency rates)
populate within minutes.

What it does each run:
  1. Read NFL state. Outside the regular / post season it exits (the daily cron
     handles the prior completed season).
  2. Detect the current week's finished games. If none have finished, or the set
     of finished games has not changed since the last successful rebuild, it
     skips the heavy work (idempotent, cheap to run often).
  3. Rebuild the base Advanced Metrics snapshot over weeks 1..current, forcing a
     refetch of the in-progress week's box scores (a populated Sleeper cache
     otherwise freezes on its first partial fetch and never sees later finals).
     This writes today's snapshot row via the same UPSERT path the daily cron
     uses, so the two never conflict.
  4. Best-effort: refresh the per-week nflverse metrics so the provider-sourced
     columns (NGS / FTN / EPA) fill in as soon as nflverse publishes them, rather
     than waiting for the single daily run. nflverse publishes a completed week's
     data on the Tue–Wed after the games, so this pull is gated to those weekdays
     and throttled to hourly within them — no all-week download storm — and any
     provider outage is a no-op that never erases the last good data.

Designed to be safe to run on any schedule: on a non-game day, or when nothing
has finished since the last run, it does almost nothing.

Usage:
    python -m scripts.refresh_live_advanced_metrics
    python -m scripts.refresh_live_advanced_metrics --force   # rebuild regardless
    python -m scripts.refresh_live_advanced_metrics --no-nflverse
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

from dashboard_services.api import get_nfl_state
from utils.paths import CACHE_DIR
from utils.utils import (
    finished_game_ids_for_week,
    load_players_index,
    resolve_adv_metrics_completed_week,
)


STATE_PATH = Path(CACHE_DIR) / "live_advanced_metrics_state.json"

# Don't re-pull nflverse (NGS/FTN/EPA) more than this often. The provider data
# lags the games by design; hammering it every few minutes buys nothing.
NFLVERSE_THROTTLE_SEC = 60 * 60

# nflverse publishes a completed week's NGS / FTN / play-by-play data on the
# Tuesday–Wednesday after the games. Outside that window a pull just re-downloads
# unchanged parquet, so gate it to those weekdays (local time; the cron runs with
# TZ=America/New_York). Monday=0 .. Sunday=6, so Tue=1, Wed=2. --force overrides.
NFLVERSE_PUBLISH_WEEKDAYS = frozenset({1, 2})


def _load_state() -> dict:
    try:
        with open(STATE_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(state: dict) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(state, f)
        tmp.replace(STATE_PATH)
    except OSError as e:
        print(f"[live-adv] could not persist state: {e}")


def _flush_app_caches() -> None:
    """Bust the running app's in-memory daily caches so the freshly rebuilt
    metrics are served on the next request instead of after the date-keyed
    entries roll over. Mirrors the daily cron's flush; skipped silently when
    APP_URL / CRON_SECRET are not configured, and never fatal."""
    app_url = os.environ.get("APP_URL", "").rstrip("/")
    cron_secret = os.environ.get("CRON_SECRET", "")
    if not app_url or not cron_secret:
        print("[live-adv] cache flush skipped — APP_URL or CRON_SECRET not set")
        return
    try:
        import urllib.request
        body = json.dumps({"secret": cron_secret}).encode()
        req = urllib.request.Request(
            f"{app_url}/api/flush-value-cache",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[live-adv] cache flush: HTTP {resp.status}")
    except Exception as e:
        print(f"[live-adv] cache flush failed (non-fatal): {e}")


def _refresh_nflverse_weekly(season: int, players_index: dict) -> int:
    """Best-effort per-week nflverse upsert. Returns rows written (0 on failure)."""
    try:
        from data_building.advanced_metrics import init_advanced_metrics_db
        from scripts.sync_nflverse_metrics import upsert_weekly_season
        init_advanced_metrics_db()
        return upsert_weekly_season(season, players_index)
    except Exception as e:
        import traceback
        print(f"[live-adv] nflverse weekly refresh failed (non-fatal): {e}")
        traceback.print_exc()
        return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild Advanced Metrics as soon as a game finishes.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if no new game has finished since the last run.")
    parser.add_argument("--no-nflverse", action="store_true",
                        help="Skip the throttled nflverse (NGS/FTN/EPA) weekly refresh.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    nfl_state = get_nfl_state() or {}
    season_type = str(nfl_state.get("season_type", "")).lower().strip()
    if season_type not in ("reg", "post"):
        print(f"[live-adv] season_type={season_type or 'unknown'} — not in season; "
              "daily cron handles the prior season. Nothing to do.")
        return 0

    season = int(nfl_state.get("season") or datetime.now().year)
    current_week = int(nfl_state.get("week") or nfl_state.get("display_week") or 0)
    if current_week < 1:
        print("[live-adv] no current week yet; nothing to do.")
        return 0

    finished = finished_game_ids_for_week(season, current_week)
    completed_week = resolve_adv_metrics_completed_week(season, current_week)
    print(f"[live-adv] season={season} week={current_week} "
          f"finished_games={len(finished)} completed_week={completed_week}")

    state = _load_state()
    today = date.today().isoformat()
    last_key = state.get("finished_key")
    # A stable fingerprint of "which games are final" for this week. When it is
    # unchanged AND we already built today, there is nothing new to fold in.
    finished_key = f"{season}:{current_week}:" + ",".join(finished)
    already_built_today = state.get("last_build_date") == today

    did_work = False

    # The live refresh exists to fold the IN-PROGRESS week's finals in fast.
    # When that week has no finals yet (completed_week < current_week), the daily
    # cron's baseline already covers every fully-finished week, so the snapshot
    # rebuild has nothing to add — but the throttled nflverse pull below still
    # runs so provider columns fill in on the Tue/Wed after a week completes.
    in_progress_finals = completed_week == current_week
    should_build = args.force or (
        in_progress_finals and (finished_key != last_key or not already_built_today))
    if not should_build:
        if not in_progress_finals:
            print("[live-adv] no finals in the in-progress week yet; "
                  "daily cron covers finished weeks. Skipping snapshot rebuild.")
        else:
            print("[live-adv] no new finals since last build; skipping snapshot rebuild.")
    else:
        from data_building.advanced_metrics import build_advanced_metrics_snapshot
        players_index = load_players_index() or {}
        # Force a refetch of the in-progress week so a just-final game lands now.
        force_weeks = [current_week] if completed_week == current_week else None
        summary = build_advanced_metrics_snapshot(
            season, completed_week,
            players_index=players_index,
            force_weeks=force_weeks,
        )
        print(f"[live-adv] snapshot rebuilt: {summary}")

        # Refresh the per-week usage rows too, forcing the in-progress week. This
        # is what powers the Advanced Metrics *week filter*: without it the week
        # dropdown offers the current week but its usage data stays frozen at the
        # pre-game fetch, so the view shows only the last completed week.
        try:
            from data_building.weekly_metrics import build_weekly_metrics
            wk_rows = build_weekly_metrics(
                season, weeks=[current_week], force_weeks=[current_week])
            print(f"[live-adv] weekly usage rows for week {current_week}: {wk_rows}")
        except Exception as e:
            import traceback
            print(f"[live-adv] weekly usage refresh failed (non-fatal): {e}")
            traceback.print_exc()

        state["last_build_date"] = today
        state["finished_key"] = finished_key
        state["last_build_ts"] = time.time()
        did_work = True

    # Provider (nflverse) refresh so NGS/FTN/EPA columns fill in ASAP once
    # nflverse publishes. Gated to the Tue–Wed publish window (outside it a pull
    # just re-downloads unchanged parquet) and throttled to hourly within it, so
    # it picks up the data as it lands without an all-week download storm.
    if not args.no_nflverse:
        weekday = datetime.now().weekday()
        in_publish_window = weekday in NFLVERSE_PUBLISH_WEEKDAYS
        last_nflverse = float(state.get("last_nflverse_ts") or 0)
        throttle_elapsed = (time.time() - last_nflverse) >= NFLVERSE_THROTTLE_SEC
        if not (args.force or in_publish_window):
            print("[live-adv] nflverse refresh skipped — outside the Tue–Wed "
                  "publish window (use --force to override).")
        elif args.force or throttle_elapsed:
            players_index = load_players_index() or {}
            wn = _refresh_nflverse_weekly(season, players_index)
            print(f"[live-adv] nflverse weekly metrics: {wn} player-weeks for season {season}")
            state["last_nflverse_ts"] = time.time()
            if wn:
                did_work = True
        else:
            wait = int((NFLVERSE_THROTTLE_SEC - (time.time() - last_nflverse)) / 60)
            print(f"[live-adv] nflverse refresh throttled (~{wait} min until next).")

    _save_state(state)

    # Only bust the app caches when we actually wrote new data.
    if did_work:
        _flush_app_caches()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
