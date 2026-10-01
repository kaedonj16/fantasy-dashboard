#!/usr/bin/env python3
"""
Wednesday breakout grading and calibration proposal email.

Loads the season's graded weekly breakout calls, builds the weekly
report (see data_building/breakout_engine/weekly_report.py: how the
recently graded calls did, the season record so far, and a concrete
suggested-changes plan), prints it, and emails it to Kaedon. Runs on a
Wednesday cron (see render.yaml: 13:00 UTC, Wednesday morning US
Eastern) so the plan lands right after the week's grading wave.

Delivery uses the app's existing provider-independent sender,
utils.email_delivery.send_email (Brevo when BREVO_API_KEY is set, SMTP
otherwise). Recipient: BREAKOUT_REPORT_EMAIL when set, else the sending
account (EMAIL_USER, then BREVO_SENDER_EMAIL). When no recipient or no
provider is configured the script prints the report, says so, and exits
0: a missing mail config must never fail the cron.

The loader below mirrors scripts/calibrate_weekly_breakouts.py's
load_graded_calls (same table-exists guard, same ordering) instead of
importing it: that script imports python-dotenv unconditionally at
module level, which the test environment does not carry, and this
script's module must stay importable there. Keep the two in sync.

Usage:
    python scripts/send_weekly_breakout_report.py
    python scripts/send_weekly_breakout_report.py --season 2026
    python scripts/send_weekly_breakout_report.py --dry-run
"""

import argparse
import html as html_lib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # local .env loading is a convenience; env vars still apply
    pass


def _default_season() -> int:
    try:
        from data_building.sleeper_data import get_nfl_state
        state = get_nfl_state() or {}
        season = state.get("season")
        if season:
            return int(season)
    except Exception:
        pass
    from datetime import date
    return date.today().year


def load_graded_calls(season: int) -> list:
    """Every persisted grade row for the season, ALL scoring versions.

    Read-only: when the grades table does not exist yet the season simply
    has no graded calls, so this returns [] rather than creating anything.
    Mirror of scripts/calibrate_weekly_breakouts.load_graded_calls.
    Fail-soft: when the database is unreachable (DATABASE_URL unset,
    connection failure, missing table module) the Wednesday cron must
    still print the honest zero-data report and exit 0, so any read
    failure degrades to [] here.
    """
    try:
        from dashboard_services.db import get_conn
        from data_building.breakout_engine.weekly_grading import GRADES_TABLE

        with get_conn() as conn:
            found = conn.execute(
                "SELECT 1 AS x FROM information_schema.tables "
                "WHERE table_name = %s",
                (GRADES_TABLE,),
            ).fetchone()
            if not found:
                return []
            rows = conn.execute(
                f"SELECT * FROM {GRADES_TABLE} WHERE season = %s "
                f"ORDER BY scoring_version, as_of_week, player_id",
                (int(season),),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _latest_week(season: int):
    """The season's most recent stored week, or None when unknowable.
    Fail-soft: the report's reason line simply omits it."""
    try:
        from data_building.breakout_engine.weekly_store import (
            latest_scored_week,
        )
        return latest_scored_week(int(season))
    except Exception:
        return None


def build_email(season: int, rows: list, latest_week=None) -> tuple:
    """(subject, plain-text body) for already-loaded grade rows. Pure."""
    from data_building.breakout_engine.weekly_report import (
        build_weekly_report,
        render_email,
    )
    report = build_weekly_report(rows, season, latest_week=latest_week)
    return render_email(report)


def resolve_recipient() -> str:
    """Where the report goes: the dedicated override first, then the
    sending account (the report is for the account owner)."""
    for key in ("BREAKOUT_REPORT_EMAIL", "EMAIL_USER", "BREVO_SENDER_EMAIL"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    return ""


def deliver(subject: str, body: str, recipient: str):
    """Send via the app's existing email path. Returns the SendResult."""
    from utils.email_delivery import send_email

    html_body = (
        '<pre style="font-family: ui-monospace, Menlo, Consolas, '
        'monospace; font-size: 13px; line-height: 1.5; white-space: '
        f'pre-wrap;">{html_lib.escape(body)}</pre>'
    )
    return send_email(
        recipient, subject, html_body, text=body,
        tags=["breakout-weekly-report"],
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None,
                        help="Season to report on (default: current NFL "
                             "season)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the report without sending email")
    args = parser.parse_args(argv)

    season = args.season if args.season is not None else _default_season()
    rows = load_graded_calls(season)
    subject, body = build_email(season, rows, _latest_week(season))
    print(f"Subject: {subject}")
    print()
    print(body)

    if args.dry_run:
        print()
        print("[dry-run] report printed only; no email sent.")
        return 0

    recipient = resolve_recipient()
    if not recipient:
        print()
        print("email not configured, report printed only: no recipient "
              "(set BREAKOUT_REPORT_EMAIL).")
        return 0

    from utils.email_delivery import is_configured
    if not is_configured():
        print()
        print("email not configured, report printed only: no Brevo API "
              "key and no SMTP credentials are set.")
        return 0

    result = deliver(subject, body, recipient)
    print()
    if result.ok:
        print(f"Report emailed to {recipient} via {result.provider}.")
    else:
        print(f"Email send failed ({result.provider}: "
              f"{result.error or 'unknown error'}); report printed only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
