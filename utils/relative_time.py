"""Human-relative timestamp formatting ("Just now", "2d ago", "May 12").

Extracted from app.py; ``now`` is injectable so the boundaries are testable.
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")


def rel_time(dt, now: datetime = None) -> str:
    """Human-relative timestamp: 'Just now', '5m ago', 'Today 3:42 PM',
    'Yesterday', '3d ago', '2w ago', then 'May 12' beyond a month."""
    now = (now or datetime.now(EASTERN)).astimezone(EASTERN)
    dt_et = dt.astimezone(EASTERN)
    diff = now - dt_et
    secs = diff.total_seconds()
    if secs < 60:
        return "Just now"
    if secs < 3600:
        mins = int(secs // 60)
        return f"{mins}m ago"
    today = now.date()
    if dt_et.date() == today:
        hour = dt_et.strftime("%I").lstrip("0") or "12"
        return f"Today {hour}:{dt_et.strftime('%M %p')}"
    if dt_et.date() == (now - timedelta(days=1)).date():
        return "Yesterday"
    days = (today - dt_et.date()).days
    if days < 7:
        return f"{days}d ago"
    if days < 30:
        return f"{days // 7}w ago"
    return dt_et.strftime("%b %d")
