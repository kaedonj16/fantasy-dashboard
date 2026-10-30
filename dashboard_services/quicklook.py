"""Pure helpers for the Player Rankings hover quick-look card.

Kept free of app.py imports so the tier / trend / kickoff semantics can be
unit-tested without the full stack. The route in app.py
(/api/player-quicklook/<id>) supplies the data; these helpers decide how
it is labeled.

Matchup tiers are the Schedule Rankings page's own scale
(dashboard_services/pages/schedule_page.py): ease ranks run 1 = easiest,
Elite is the top 25% of defenses, Brutal the bottom 25%, in that page's
chip colors.
"""

from datetime import datetime

# (label, hex) per tier, in rank order (easiest first).
TIER_ELITE = ("Elite", "#22c55e")
TIER_GOOD = ("Good", "#84cc16")
TIER_TOUGH = ("Tough", "#f59e0b")
TIER_BRUTAL = ("Brutal", "#ef4444")


def tier_for_rank(rank, total):
    """Schedule Rankings tier for an ease rank (1 = easiest) out of total.

    Quartiles of the ranked pool: top 25% Elite, then Good, Tough, and the
    bottom 25% Brutal. Returns (label, hex) or None for unusable input.
    """
    try:
        rank = int(rank)
        total = int(total)
    except (TypeError, ValueError):
        return None
    if rank < 1 or total < 1 or rank > total:
        return None
    share = rank / total
    if share <= 0.25:
        return TIER_ELITE
    if share <= 0.50:
        return TIER_GOOD
    if share <= 0.75:
        return TIER_TOUGH
    return TIER_BRUTAL


def pct_change_7d(points):
    """Percent change over the last 7 days of a dated value series.

    points: [(iso_date_str, value), ...] ascending, already scaled to the
    current value (the value-history reader's contract). Compares the last
    point with the latest point at least 7 days older; returns None when
    the series cannot support the claim (too short, too compressed, or a
    non-positive base) so callers render nothing instead of a made-up move.
    """
    pts = [(d, v) for d, v in points if v is not None and v > 0]
    if len(pts) < 2:
        return None
    try:
        last_date = datetime.strptime(str(pts[-1][0])[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    base = None
    for d, v in reversed(pts[:-1]):
        try:
            dd = datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue
        if (last_date - dd).days >= 7:
            base = v
            break
    if base is None:
        # The whole series spans less than a week-plus: only claim a 7d
        # move when the series itself covers at least 5 days, measured
        # from its first point.
        try:
            first_date = datetime.strptime(str(pts[0][0])[:10], "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return None
        if (last_date - first_date).days < 5:
            return None
        base = pts[0][1]
    if not base:
        return None
    return round((pts[-1][1] / base - 1.0) * 100.0, 1)


def format_kickoff(game_date, game_time):
    """'Sun 1:00 PM' from a schedule row's YYYYMMDD date + Tank01 time.

    game_time arrives as '1:00p' / '8:15p' / '4:25p' (or occasionally a
    fuller '1:00 PM'). Returns '' when either half is unusable so the
    caller can fall back to the date alone.
    """
    s = str(game_date or "").strip()
    if len(s) != 8 or not s.isdigit():
        return ""
    try:
        day = datetime.strptime(s, "%Y%m%d").strftime("%a")
    except ValueError:
        return ""
    t = str(game_time or "").strip()
    if not t:
        return day
    tl = t.lower().replace(" ", "")
    meridiem = ""
    if tl.endswith("pm") or tl.endswith("p"):
        meridiem = "PM"
        tl = tl[:-2] if tl.endswith("pm") else tl[:-1]
    elif tl.endswith("am") or tl.endswith("a"):
        meridiem = "AM"
        tl = tl[:-2] if tl.endswith("am") else tl[:-1]
    if ":" not in tl:
        return day
    return f"{day} {tl} {meridiem}".strip()
