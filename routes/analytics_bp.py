"""Admin product-analytics dashboard (Phase 1).

``GET /admin/analytics`` renders a server-side summary of first-party usage:
DAU/WAU, signups, feature usage, week-over-week return, and the
visitor -> signup -> league linked -> PRO funnel.

``GET``/``POST /admin/login`` is the password login for browser use: it checks
the submitted password against the ``ADMIN_PASSWORD`` env var and, on success,
marks the session admin (the same session flag the ``ADMIN_KEY`` flow sets).

Gated by :func:`dashboard_services.admin_auth.is_admin`; non-admins get a
404 so the page's existence is not revealed. Charts are dependency-free
inline SVG generated in Python (same spirit as the NFL team graphs).
"""
from __future__ import annotations

import html
import logging
from datetime import datetime, timezone

from flask import Blueprint, Response, redirect, request, session

from dashboard_services.admin_auth import (
    ADMIN_SESSION_KEY,
    _configured_password,
    is_admin,
    mark_admin_session,
    verify_admin_password,
)
from extensions import limiter

analytics_bp = Blueprint("analytics_bp", __name__)
logger = logging.getLogger(__name__)


# ── Inline SVG charts ─────────────────────────────────────────────────────────

def _bars_svg(pairs, width=680, height=190, bar_color="#4f8ff7", max_bar_w=48):
    """pairs: list of (label, value). Returns an inline SVG bar chart.

    Bars are capped at max_bar_w px and centered in their slots, so a sparse
    series (e.g. a single day of data) does not render as one giant bar.
    Every nonzero value gets a label above its bar, the x axis labels every
    point (rotated when the series is dense), and the gridlines carry y-axis
    tick labels."""
    if not pairs:
        return '<p class="muted">No data yet.</p>'
    maxv = max(v for _, v in pairs) or 1
    n = len(pairs)
    rotate = n > 15
    pad_l, pad_r, pad_t, pad_b = 34, 10, 24, 58 if rotate else 30
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    slot = plot_w / n
    bw = min(slot * 0.64, max_bar_w)
    parts = ['<svg viewBox="0 0 %d %d" class="chart" role="img">' % (width, height)]
    for frac in (0.0, 0.5, 1.0):
        gy = pad_t + plot_h * (1 - frac)
        cls = "grid base" if frac == 0.0 else "grid"
        parts.append(
            '<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="%s"/>'
            % (pad_l, gy, width - pad_r, gy, cls)
        )
        parts.append(
            '<text x="%d" y="%.1f" text-anchor="end" class="ytick">%d</text>'
            % (pad_l - 5, gy + 3.5, round(maxv * frac))
        )
    step = 1 if rotate else max(1, n // 12)
    for i, (label, value) in enumerate(pairs):
        frac = (value / maxv) if maxv else 0
        bh = max(frac * plot_h, 2 if value else 0)
        x = pad_l + i * slot + (slot - bw) / 2
        y = pad_t + plot_h - bh
        parts.append(
            '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="2" fill="%s">'
            '<title>%s: %s</title></rect>'
            % (x, y, bw, bh, bar_color, html.escape(str(label)), value)
        )
        if value:
            parts.append(
                '<text x="%.1f" y="%.1f" text-anchor="middle" class="vallab">%s</text>'
                % (x + bw / 2, y - 5, value)
            )
        if i % step == 0 or i == n - 1:
            cx = x + bw / 2
            if rotate:
                parts.append(
                    '<text x="%.1f" y="%d" text-anchor="end" class="axis" '
                    'transform="rotate(-45 %.1f %d)">%s</text>'
                    % (cx, height - 8, cx, height - 8, html.escape(str(label)))
                )
            else:
                parts.append(
                    '<text x="%.1f" y="%d" text-anchor="middle" class="axis">%s</text>'
                    % (cx, height - 10, html.escape(str(label)))
                )
    parts.append("</svg>")
    return "".join(parts)


def _pct(a, b):
    """Conversion percent a/b as a short string; 'n/a' when b is 0."""
    if not b:
        return "n/a"
    return "%.1f%%" % (100.0 * a / b)


# ── Page sections ─────────────────────────────────────────────────────────────

def _section(title, body, note=""):
    note_html = '<p class="muted">%s</p>' % html.escape(note) if note else ""
    return (
        '<section class="card"><h2>%s</h2>%s%s</section>'
        % (html.escape(title), note_html, body)
    )


def _feature_table(rows):
    """rows: [{week, event, count}] -> HTML table, weeks as rows, events as cols."""
    if not rows:
        return '<p class="muted">No data yet.</p>'
    events = sorted({r["event"] for r in rows})
    weeks = sorted({r["week"] for r in rows}, reverse=True)
    lookup = {(r["week"], r["event"]): r["count"] for r in rows}
    head = "".join("<th>%s</th>" % html.escape(e) for e in events)
    body_rows = []
    for w in weeks:
        cells = "".join(
            "<td>%d</td>" % lookup.get((w, e), 0) for e in events
        )
        body_rows.append("<tr><th scope='row'>%s</th>%s</tr>" % (html.escape(w), cells))
    return (
        '<div class="tablewrap"><table><thead><tr><th>Week</th>%s</tr></thead>'
        "<tbody>%s</tbody></table></div>" % (head, "".join(body_rows))
    )


def _retention_table(rows):
    if not rows:
        return '<p class="muted">No data yet.</p>'
    body_rows = []
    for r in rows:
        body_rows.append(
            "<tr><th scope='row'>%s</th><td>%d</td><td>%d</td><td>%s%%</td></tr>"
            % (html.escape(r["week"]), r["active"], r["returned"], r["rate"])
        )
    return (
        '<div class="tablewrap"><table><thead><tr>'
        "<th>Week</th><th>Active users</th><th>Returned from prior week</th>"
        "<th>Return rate</th></tr></thead><tbody>%s</tbody></table></div>"
        % "".join(body_rows)
    )


_BREAKDOWN_ROWS = [
    ("Raw distinct identities (old definition)", "headline"),
    ("Signed-in accounts", "signed_in"),
    ("Anonymous sessions", "anon_sessions"),
    ("Anonymous: one-and-done (1 pageview)", "anon_one_and_done"),
    ("Anonymous: engaged (2+ pageviews, never signed in)", "anon_engaged"),
    ("Anonymous sessions that also signed in (counted twice)", "anon_linked_sessions"),
    ("DAU (current definition: signed-in + engaged anonymous)", "realistic_preview"),
    ("Total pageviews", "total_pageviews"),
]


def _breakdown_html(breakdown, top_paths):
    """DAU decomposition table + top one-and-done paths line.

    Reconciles the chart's realistic DAU against the raw old-definition
    count. breakdown: {"utc": {...}, "ny": {...}} or None when the
    breakdown queries failed (the rest of the page must still render).
    """
    if not breakdown:
        return '<p class="muted">Breakdown unavailable.</p>'
    utc = breakdown.get("utc") or {}
    ny = breakdown.get("ny") or {}
    body_rows = []
    for label, key in _BREAKDOWN_ROWS:
        body_rows.append(
            "<tr><th scope='row'>%s</th><td>%d</td><td>%d</td></tr>"
            % (html.escape(label), int(utc.get(key, 0) or 0), int(ny.get(key, 0) or 0))
        )
    table = (
        '<div class="tablewrap"><table><thead><tr>'
        "<th>Metric</th><th>UTC day</th><th>New York day (chart day)</th>"
        "</tr></thead><tbody>%s</tbody></table></div>" % "".join(body_rows)
    )
    paths_html = ""
    if top_paths:
        listed = ", ".join(
            "%s: %d" % (html.escape(str(p["path"])), int(p["count"]))
            for p in top_paths
        )
        paths_html = (
            '<p class="muted">Top one-and-done paths (New York day): %s</p>' % listed
        )
    return table + paths_html


def _funnel_html(f):
    stages = [
        ("Visitors", f["visitors"], None),
        ("Signups", f["signups"], f["visitors"]),
        ("League linked", f["linked"], f["signups"]),
        ("PRO", f["pro"], f["linked"]),
    ]
    cards = []
    for name, count, base in stages:
        conv = "" if base is None else '<div class="conv">%s of prior step</div>' % _pct(count, base)
        cards.append(
            '<div class="funnel-step"><div class="funnel-num">%d</div>'
            '<div class="funnel-name">%s</div>%s</div>'
            % (count, html.escape(name), conv)
        )
    return '<div class="funnel">%s</div>' % "".join(cards)


# ── Admin password login ───────────────────────────────────────────────────

_LOGIN_STYLE = """
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; padding: 24px; background: #f6f7f9; color: #1c2333; }
  .wrap { max-width: 420px; margin: 60px auto; }
  .card { background: #fff; border: 1px solid #e3e6ec; border-radius: 12px;
          padding: 24px; }
  h1 { font-size: 20px; margin: 0 0 6px; }
  .sub { color: #5b6478; font-size: 13px; margin: 0 0 18px; }
  label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; }
  input[type=password] { width: 100%; box-sizing: border-box; font-size: 15px;
      padding: 10px 12px; border: 1px solid #d4d9e3; border-radius: 8px; }
  button { margin-top: 14px; width: 100%; font-size: 15px; font-weight: 600;
      padding: 11px; border: 0; border-radius: 8px; background: #4f8ff7;
      color: #fff; cursor: pointer; }
  button:disabled { background: #a9b4c9; cursor: default; }
  .error { background: #fdecec; border: 1px solid #f3b8b8; color: #a33;
          border-radius: 8px; padding: 10px 12px; font-size: 13px;
          margin-bottom: 14px; }
  .note { background: #fff8e6; border: 1px solid #f0d98c; border-radius: 8px;
          padding: 10px 12px; font-size: 13px; margin-bottom: 14px; }
"""


def _login_page(error="", password_configured=True):
    error_html = (
        '<div class="error">%s</div>' % html.escape(error) if error else ""
    )
    if password_configured:
        form = """
  <form method="post">
    <label for="password">Admin password</label>
    <input type="password" id="password" name="password" autocomplete="current-password" autofocus>
    <button type="submit">Log in</button>
  </form>"""
    else:
        form = (
            '<div class="note">No admin password is configured. '
            "Set the ADMIN_PASSWORD environment variable to enable password login.</div>"
            '<form method="post"><button type="submit" disabled>Log in</button></form>'
        )
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Admin login</title>
<style>%s</style>
</head>
<body>
<div class="wrap"><div class="card">
  <h1>Admin login</h1>
  <p class="sub">Sign in to view the analytics dashboard.</p>
  %s%s
</div></div>
</body>
</html>""" % (_LOGIN_STYLE, error_html, form)


@analytics_bp.route("/admin/login", methods=["GET"])
def admin_login_form():
    if is_admin():
        return redirect("/admin/analytics")
    return Response(
        _login_page(password_configured=bool(_configured_password())),
        mimetype="text/html",
    )


@analytics_bp.route("/admin/login", methods=["POST"])
@limiter.limit("10 per minute")
def admin_login_submit():
    if is_admin():
        return redirect("/admin/analytics")
    provided = request.form.get("password") or ""
    if verify_admin_password(provided):
        mark_admin_session()
        return redirect("/admin/analytics")
    # Generic message on purpose: do not distinguish a missing password
    # from a wrong one, or an unconfigured ADMIN_PASSWORD.
    return Response(
        _login_page(
            error="Incorrect password.",
            password_configured=bool(_configured_password()),
        ),
        mimetype="text/html",
    )


# ── Route ─────────────────────────────────────────────────────────────────────

@analytics_bp.route("/admin/analytics")
def admin_analytics():
    if not is_admin():
        return Response("Not found", status=404)
    try:
        from dashboard_services import analytics as _a

        dau = _a.dau_last_30_days()
        wau = _a.wau_last_12_weeks()
        signups = _a.signups_per_day()
        usage = _a.feature_usage_by_week(8)
        retention = _a.week_over_week_return()
        funnel = _a.funnel_last_30_days()
        has_events = _a.events_table_ready()
    except Exception:
        logger.exception("[analytics] admin page data failed")
        return Response(
            "<h1>Analytics unavailable</h1>"
            "<p>The stats page hit a database error. Check the server logs.</p>",
            status=500, mimetype="text/html",
        )

    dau_pairs = _a.fill_daily_gaps(dau, "date", "users", 30, today=_a.ny_today())
    wau_pairs = _a.fill_weekly_gaps(wau, "week", "users", 12, today=_a.ny_today())
    signup_pairs = _a.fill_daily_gaps(signups, "date", "signups", 30)

    # DAU breakdown (reconciliation): fetched separately so a failure here
    # can never take down the rest of the page.
    try:
        breakdown = _a.dau_breakdown()
        breakdown_paths = _a.one_and_done_top_paths(5)
    except Exception:
        logger.exception("[analytics] dau breakdown failed")
        breakdown = None
        breakdown_paths = []

    empty_note = (
        "Event collection just started, so these charts fill in over the coming days. "
        "Signups come from the existing accounts table and are available now."
        if not has_events else ""
    )

    # Self-exclusion status: show what is filtering the numbers on this page.
    status_bits = []
    excluded_ids = sorted(_a.excluded_account_ids())
    if excluded_ids:
        status_bits.append(
            "Excluding account%s %s from all numbers below."
            % ("s" if len(excluded_ids) != 1 else "",
               ", ".join(str(i) for i in excluded_ids))
        )
    if session.get(ADMIN_SESSION_KEY):
        status_bits.append("Admin session: your visits are not recorded.")
    status_html = (
        '<p class="sub">%s</p>' % " ".join(html.escape(b) for b in status_bits)
        if status_bits else ""
    )

    body = "".join([
        _section("Daily active users", _bars_svg(dau_pairs),
                 "Signed-in accounts plus anonymous visitors with 2+ pages, "
                 "New York day, last 30 days."),
        _section("DAU breakdown: today",
                 _breakdown_html(breakdown, breakdown_paths),
                 "What today's DAU is made of. The chart above now uses the "
                 "realistic definition on a New York day; the raw row is the "
                 "old definition for comparison."),
        _section("Weekly active users", _bars_svg(wau_pairs, bar_color="#34c98e"),
                 "Same definition per week (signed-in accounts plus engaged "
                 "anonymous visitors), last 12 weeks."),
        _section("Signups per day", _bars_svg(signup_pairs, bar_color="#f5a623"),
                 "New accounts from the accounts table, last 30 days."),
        _section("Feature usage by week", _feature_table(usage),
                 "Explicit product events per week, last 8 weeks. Pageviews excluded."),
        _section("Week-over-week return", _retention_table(retention),
                 "Share of each week's active users who were also active the prior week."),
        _section("Funnel: visitor to PRO", _funnel_html(funnel),
                 "Period totals for the last 30 days. Not a strict cohort funnel."),
    ])

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Product Analytics</title>
<style>
  :root { color-scheme: light; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; padding: 24px; background: #f6f7f9; color: #1c2333; }
  .wrap { max-width: 1060px; margin: 0 auto; }
  h1 { font-size: 24px; margin: 0 0 4px; }
  .sub { color: #5b6478; margin: 0 0 20px; font-size: 13px; }
  .card { background: #fff; border: 1px solid #e3e6ec; border-radius: 12px;
          padding: 18px 20px; margin-bottom: 18px; }
  .card h2 { font-size: 16px; margin: 0 0 10px; }
  .muted { color: #7a8398; font-size: 12px; }
  .chart { width: 100%%; height: auto; display: block; }
  .chart rect { fill: #4f8ff7; }
  .chart text.vallab { font-size: 11px; fill: #5b6478; font-weight: 600; }
  .chart text.axis { font-size: 10px; fill: #7a8398; }
  .chart text.ytick { font-size: 10px; fill: #a0a8bb; }
  .chart line.grid { stroke: #edf0f5; stroke-width: 1; }
  .chart line.grid.base { stroke: #dfe3ea; }
  .tablewrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%%; font-size: 13px; }
  th, td { border: 1px solid #e8ebf1; padding: 7px 10px; text-align: right; }
  th[scope=row], thead th { text-align: left; background: #f2f4f8; }
  td { text-align: right; }
  .funnel { display: flex; gap: 12px; flex-wrap: wrap; }
  .funnel-step { flex: 1 1 160px; background: #f2f4f8; border-radius: 10px;
                padding: 14px; text-align: center; }
  .funnel-num { font-size: 28px; font-weight: 700; }
  .funnel-name { font-size: 13px; color: #5b6478; margin-top: 2px; }
  .funnel .conv { font-size: 12px; color: #2f7d4f; margin-top: 6px; font-weight: 600; }
  .notice { background: #fff8e6; border: 1px solid #f0d98c; border-radius: 10px;
            padding: 12px 16px; margin-bottom: 18px; font-size: 13px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Product Analytics</h1>
  <p class="sub">First-party usage stats. Generated %s.</p>
  %s
  %s
  %s
</div>
</body>
</html>""" % (
        html.escape(generated),
        ('<div class="notice">%s</div>' % html.escape(empty_note)) if empty_note else "",
        status_html,
        body,
    )
    return Response(page, mimetype="text/html")
