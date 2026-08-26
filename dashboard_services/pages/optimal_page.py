"""Optimal lineup tracker.

Moved from app.py so Weekly Hub can import it without keeping the HTML builder
in the Flask monolith. Player-index lookup is lazy-imported from app.
"""
from __future__ import annotations

import html
import logging
from collections import defaultdict
from datetime import datetime

from flask import request

from utils.optimal_lineup import compute_optimal_lineup as _compute_optimal_lineup

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMAL LINEUP TRACKER
# ══════════════════════════════════════════════════════════════════════════════

_OPT_POS_COLORS = {"QB": "#3b82f6", "RB": "#22c55e", "WR": "#f59e0b", "TE": "#8b5cf6"}


def _opt_player_badges(players_out: list) -> str:
    parts = []
    for p in players_out:
        if p["actual_start"] == p["optimal_start"]:
            badge_style = "background:var(--surface);border:1px solid var(--border);"
            icon = ""
        elif p["actual_start"] and not p["optimal_start"]:
            badge_style = "background:#ef444420;border:1px solid #ef4444;color:#ef4444;"
            icon = "&#9660; "
        else:
            badge_style = "background:#22c55e20;border:1px solid #22c55e;color:#22c55e;"
            icon = "&#9650; "
        pos_color = _OPT_POS_COLORS.get(p["pos"], "#6b7280")
        parts.append(
            f"<span style='padding:4px 8px;border-radius:6px;font-size:12px;{badge_style}'>"
            f"<span style='color:{pos_color};font-weight:700;'>{p['pos']}</span> "
            f"{html.escape(p['name'])} {icon}<b>{p['pts']:.1f}</b></span>"
        )
    return "".join(parts)


def _opt_pos_breakdown(weeks_data: list) -> str:
    pos_left: dict = {}
    pos_misses: dict = {}
    for wd in weeks_data:
        for p in wd["players"]:
            if not p["actual_start"] and p["optimal_start"]:
                bucket = p["pos"] if p["pos"] in ("QB", "RB", "WR", "TE") else "OTHER"
                pos_left[bucket] = pos_left.get(bucket, 0.0) + p["pts"]
                pos_misses[bucket] = pos_misses.get(bucket, 0) + 1
    if not pos_left:
        return ""
    rows_html = ""
    for pos in ["QB", "RB", "WR", "TE", "OTHER"]:
        if pos not in pos_left:
            continue
        left = round(pos_left[pos], 1)
        misses = pos_misses[pos]
        pos_color = _OPT_POS_COLORS.get(pos, "#6b7280")
        label = "K/DEF/FLEX" if pos == "OTHER" else pos
        rows_html += (
            f"<tr>"
            f"<td style='padding:8px 12px;'><span style='font-weight:700;color:{pos_color};'>{label}</span></td>"
            f"<td style='padding:8px 12px;font-weight:600;color:#ef4444;'>+{left}</td>"
            f"<td style='padding:8px 12px;color:var(--muted);'>{misses}×</td>"
            f"</tr>"
        )
    return (
        "<div class='card' style='margin-top:16px;'>"
        "<div class='card-header'><h3>Points Left by Position</h3></div>"
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr style='border-bottom:1px solid var(--border);'>"
        "<th style='padding:8px 12px;text-align:left;font-size:11px;color:var(--muted);'>POS</th>"
        "<th style='padding:8px 12px;text-align:left;font-size:11px;color:var(--muted);'>LEFT ON BENCH</th>"
        "<th style='padding:8px 12px;text-align:left;font-size:11px;color:var(--muted);'>MISSED STARTS</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
        "</div>"
    )


def _opt_recurring_mistakes(weeks_data: list) -> str:
    """Repeat offenders: individual players the viewer left on the bench in a
    week where the optimal lineup would have started them. A player benched more
    than once is a pattern worth naming ("you've benched Player X 3x"), so this
    ranks by times-benched then points forfeited. Single one-off misses are
    already covered by the week-by-week table, so require >= 2 occurrences."""
    agg: dict = defaultdict(lambda: {"name": "", "pos": "", "weeks": [], "pts": 0.0})
    for wd in weeks_data:
        wk = wd.get("week")
        for p in wd["players"]:
            if p["optimal_start"] and not p["actual_start"]:
                rec = agg[p["pid"]]
                rec["name"] = p["name"]
                rec["pos"] = p["pos"]
                rec["weeks"].append(wk)
                rec["pts"] += p["pts"]
    repeats = [r for r in agg.values() if len(r["weeks"]) >= 2]
    if not repeats:
        return ""
    repeats.sort(key=lambda r: (-len(r["weeks"]), -r["pts"]))
    rows_html = ""
    for r in repeats[:8]:
        n = len(r["weeks"])
        pos_color = _OPT_POS_COLORS.get(r["pos"], "#6b7280")
        wk_list = ", ".join(f"W{w}" for w in sorted(w for w in r["weeks"] if w))
        rows_html += (
            f"<tr style='border-bottom:1px solid var(--border);'>"
            f"<td style='padding:9px 12px;'>"
            f"<span style='font-weight:700;color:{pos_color};margin-right:6px;'>{r['pos']}</span>"
            f"<span style='font-weight:600;'>{html.escape(r['name'])}</span></td>"
            f"<td style='padding:9px 12px;font-weight:700;'>{n}&times;</td>"
            f"<td style='padding:9px 12px;color:#ef4444;font-weight:600;'>+{round(r['pts'], 1)}</td>"
            f"<td style='padding:9px 12px;color:var(--muted);font-size:12px;'>{wk_list}</td>"
            f"</tr>"
        )
    return (
        "<div class='card' style='margin-top:16px;overflow:auto;'>"
        "<div class='card-header'><h3>Recurring Mistakes</h3></div>"
        "<div style='padding:0 12px 4px;font-size:12px;color:var(--muted);'>"
        "Players you benched in a week the optimal lineup would have started them, "
        "twice or more this season.</div>"
        "<table style='width:100%;min-width:420px;border-collapse:collapse;'>"
        "<thead><tr style='border-bottom:1px solid var(--border);'>"
        "<th style='padding:8px 12px;text-align:left;font-size:11px;color:var(--muted);'>PLAYER</th>"
        "<th style='padding:8px 12px;text-align:left;font-size:11px;color:var(--muted);'>BENCHED</th>"
        "<th style='padding:8px 12px;text-align:left;font-size:11px;color:var(--muted);'>PTS LOST</th>"
        "<th style='padding:8px 12px;text-align:left;font-size:11px;color:var(--muted);'>WEEKS</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
        "</div>"
    )


def build_optimal_body(ctx):
    from app import get_players_index_global
    from dashboard_services.platform_api import get_matchups as _gm

    platform = ctx.get("platform") or "sleeper"
    season = ctx.get("season") or datetime.now().year
    league_id = ctx.get("league_id") or ""
    viewer_rid = str((ctx.get("viewer") or {}).get("viewer_roster_id") or "")
    roster_positions = ctx.get("roster_positions") or []
    settings = (ctx.get("league") or {}).get("settings") or {}
    playoff_start = int(settings.get("playoff_week_start") or 14)
    # current_week lives on ctx directly; build_league_context never exposes a
    # "state" key, so reading ctx.get("state") here would pin current_week to 0
    # and make the page render "No completed weeks yet" all season.
    current_week = int(ctx.get("current_week") or 0)
    players_idx = get_players_index_global() or {}
    roster_map = ctx.get("roster_map") or {}
    rosters = ctx.get("rosters") or []

    view = request.args.get("view", "user")
    period = request.args.get("period", "season")
    try:
        sel_week = int(request.args.get("week", 0))
    except (ValueError, TypeError):
        sel_week = 0

    max_week = min(current_week - 1, playoff_start - 1)
    if max_week < 1:
        return ("<div style='padding:32px;text-align:center;color:var(--muted);'>"
                "<p>No completed weeks yet. Optimal lineup data will appear here once the season starts.</p>"
                "</div>")

    if sel_week < 1 or sel_week > max_week:
        sel_week = max_week

    # Optimal Lineup renders inside the Matchups page's "optimal" tab, so its
    # internal view/period/week nav points back there (with ?tab=optimal) rather
    # than to a standalone /optimal page.
    base_url = f"/{platform}/{season}/{league_id}/weekly?tab=optimal"

    # ── Navigation ────────────────────────────────────────────────────────────
    def _tab(label, tv, tp, active):
        wk = f"&week={sel_week}" if tp == "weekly" else ""
        cls = "opt-tab active" if active else "opt-tab"
        return f"<a class='{cls}' href='{base_url}&view={tv}&period={tp}{wk}'>{label}</a>"

    week_opts = "".join(
        f"<option value='{w}'{' selected' if w == sel_week else ''}>Week {w}</option>"
        for w in range(1, max_week + 1)
    )
    week_picker = (
        f"<select class='opt-week-select' "
        f"onchange=\"window.location='{base_url}&view={view}&period=weekly&week='+this.value\">"
        f"{week_opts}</select>"
    ) if period == "weekly" else ""

    nav_html = (
        "<div class='opt-nav'>"
        f"  <div class='opt-tab-group'>"
        f"    {_tab('My Team', 'user', period, view == 'user')}"
        f"    {_tab('League', 'league', period, view == 'league')}"
        f"  </div>"
        f"  <div class='opt-tab-group'>"
        f"    {_tab('Season', view, 'season', period == 'season')}"
        f"    {_tab('Weekly', view, 'weekly', period == 'weekly')}"
        f"  </div>"
        f"  {week_picker}"
        "</div>"
    )

    # ── Fetch only the weeks this view actually needs ─────────────────────────
    # Weekly views need exactly one week; season views need all completed weeks.
    weeks_to_fetch = [sel_week] if period == "weekly" else list(range(1, max_week + 1))

    def _fetch_opt_week(w):
        try:
            return w, _gm(platform, league_id, w, season) or []
        except Exception:
            return w, []

    from concurrent.futures import ThreadPoolExecutor as _OPT_TPE
    with _OPT_TPE(max_workers=min(len(weeks_to_fetch), 8)) as _opt_pool:
        all_matchups: dict = dict(_opt_pool.map(_fetch_opt_week, weeks_to_fetch))

    def _roster_weeks(rid: str) -> list:
        weeks = []
        for w in range(1, max_week + 1):
            try:
                row = next((m for m in all_matchups[w] if str(m.get("roster_id")) == str(rid)), None)
                if not row:
                    continue
                starters_raw = [str(p) for p in (row.get("starters") or []) if p and str(p) != "0"]
                all_pids = [str(p) for p in (row.get("players") or []) if p and str(p) != "0"]
                pts_map = {str(k): float(v or 0) for k, v in (row.get("players_points") or {}).items()}
                if not starters_raw or not all_pids:
                    continue
                actual_pts = float(row.get("points") or 0)
                pos_map = {pid: (players_idx.get(pid) or {}).get("pos") or "" for pid in all_pids}
                opt_set, opt_pts = _compute_optimal_lineup(pts_map, pos_map, roster_positions, all_pids)
                left = round(max(opt_pts - actual_pts, 0), 1)
                starter_set = set(starters_raw)
                players_out = sorted([
                    {
                        "pid": pid,
                        "name": (players_idx.get(pid) or {}).get("name") or pid,
                        "pos": ((players_idx.get(pid) or {}).get("pos") or "").upper(),
                        "pts": float(pts_map.get(pid) or 0),
                        "actual_start": pid in starter_set,
                        "optimal_start": pid in opt_set,
                    }
                    for pid in all_pids
                ], key=lambda p: -p["pts"])
                weeks.append({"week": w, "actual_pts": round(actual_pts, 2),
                              "opt_pts": round(opt_pts, 2), "left": left, "players": players_out})
            except Exception:
                continue
        return weeks

    def _wk_table_row(wd: dict) -> str:
        left_color = "#ef4444" if wd["left"] > 15 else ("#f59e0b" if wd["left"] > 7 else "#22c55e")
        badges = _opt_player_badges(wd["players"])
        return (
            f"<tr style='cursor:pointer;' onclick='this.nextElementSibling.style.display="
            f"this.nextElementSibling.style.display===\"none\"?\"table-row\":\"none\"'>"
            f"<td style='padding:10px 12px;font-weight:600;'>Week {wd['week']}</td>"
            f"<td style='padding:10px 12px;'>{wd['actual_pts']}</td>"
            f"<td style='padding:10px 12px;'>{wd['opt_pts']}</td>"
            f"<td style='padding:10px 12px;font-weight:700;color:{left_color};'>+{wd['left']}</td>"
            f"</tr>"
            f"<tr style='display:none;background:var(--surface2);'>"
            f"<td colspan='4' style='padding:12px;'>"
            f"<div style='font-size:12px;color:var(--muted);margin-bottom:8px;'>"
            f"<span style='color:#ef4444'>&#9660; Should have benched</span> &nbsp; "
            f"<span style='color:#22c55e'>&#9650; Should have started</span></div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:6px;'>{badges}</div>"
            f"</td></tr>"
        )

    _TH = "<th style='padding:10px 12px;text-align:left;font-size:11px;color:var(--muted);'>"
    _no_data = ("<div class='card'><div class='card-body' style='padding:24px;text-align:center;"
                "color:var(--muted);'>No data available.</div></div>")

    # ── League season view ────────────────────────────────────────────────────
    if view == "league" and period == "season":
        team_rows = []
        for r in rosters:
            rid = str(r.get("roster_id") or "")
            if not rid:
                continue
            tname = roster_map.get(rid) or f"Team {rid}"
            weeks = _roster_weeks(rid)
            if not weeks:
                continue
            s_actual = sum(w["actual_pts"] for w in weeks)
            s_opt = sum(w["opt_pts"] for w in weeks)
            s_left = round(sum(w["left"] for w in weeks), 1)
            eff = round((s_actual / s_opt * 100) if s_opt else 0, 1)
            team_rows.append({"name": tname, "actual": round(s_actual, 1),
                              "opt": round(s_opt, 1), "left": s_left, "eff": eff})
        team_rows.sort(key=lambda x: -x["eff"])

        rows_html = ""
        for i, tr in enumerate(team_rows):
            eff_color = "#22c55e" if tr["eff"] >= 92 else ("#f59e0b" if tr["eff"] >= 86 else "#ef4444")
            rows_html += (
                f"<tr style='border-bottom:1px solid var(--border);'>"
                f"<td style='padding:10px 12px;color:var(--muted);font-weight:600;'>#{i + 1}</td>"
                f"<td style='padding:10px 12px;font-weight:600;'>{html.escape(tr['name'])}</td>"
                f"<td style='padding:10px 12px;font-weight:700;color:{eff_color};'>{tr['eff']}%</td>"
                f"<td style='padding:10px 12px;'>{tr['actual']}</td>"
                f"<td style='padding:10px 12px;'>{tr['opt']}</td>"
                f"<td style='padding:10px 12px;color:#ef4444;font-weight:600;'>+{tr['left']}</td>"
                f"</tr>"
            )
        body_html = (
            f"<div class='card' style='overflow:auto;'>"
            f"<div class='card-header'><h3>League Lineup Efficiency - Season</h3></div>"
            f"<table style='width:100%;min-width:520px;border-collapse:collapse;'>"
            f"<thead><tr style='border-bottom:2px solid var(--border);'>"
            f"{_TH}#</th>{_TH}TEAM</th>{_TH}EFFICIENCY</th>"
            f"{_TH}ACTUAL PTS</th>{_TH}OPTIMAL PTS</th>{_TH}LEFT ON BENCH</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table></div>"
        ) if team_rows else _no_data
        return nav_html + body_html

    # ── League weekly view ────────────────────────────────────────────────────
    if view == "league" and period == "weekly":
        wk_rows = []
        for r in rosters:
            rid = str(r.get("roster_id") or "")
            if not rid:
                continue
            tname = roster_map.get(rid) or f"Team {rid}"
            row = next((m for m in all_matchups.get(sel_week, []) if str(m.get("roster_id")) == str(rid)), None)
            if not row:
                continue
            try:
                starters_raw = [str(p) for p in (row.get("starters") or []) if p and str(p) != "0"]
                all_pids = [str(p) for p in (row.get("players") or []) if p and str(p) != "0"]
                pts_map = {str(k): float(v or 0) for k, v in (row.get("players_points") or {}).items()}
                if not starters_raw or not all_pids:
                    continue
                actual_pts = float(row.get("points") or 0)
                pos_map = {pid: (players_idx.get(pid) or {}).get("pos") or "" for pid in all_pids}
                opt_set, opt_pts = _compute_optimal_lineup(pts_map, pos_map, roster_positions, all_pids)
                left = round(max(opt_pts - actual_pts, 0), 1)
                starter_set = set(starters_raw)
                players_out = sorted([
                    {"pid": pid, "name": (players_idx.get(pid) or {}).get("name") or pid,
                     "pos": ((players_idx.get(pid) or {}).get("pos") or "").upper(),
                     "pts": float(pts_map.get(pid) or 0),
                     "actual_start": pid in starter_set, "optimal_start": pid in opt_set}
                    for pid in all_pids
                ], key=lambda p: -p["pts"])
                wk_rows.append({"name": tname, "actual": round(actual_pts, 2),
                                "opt": round(opt_pts, 2), "left": left, "players": players_out})
            except Exception:
                continue
        wk_rows.sort(key=lambda x: -x["left"])

        rows_html = ""
        for tr in wk_rows:
            left_color = "#ef4444" if tr["left"] > 15 else ("#f59e0b" if tr["left"] > 7 else "#22c55e")
            badges = _opt_player_badges(tr["players"])
            rows_html += (
                f"<tr style='cursor:pointer;border-bottom:1px solid var(--border);' "
                f"onclick='this.nextElementSibling.style.display="
                f"this.nextElementSibling.style.display===\"none\"?\"table-row\":\"none\"'>"
                f"<td style='padding:10px 12px;font-weight:600;'>{html.escape(tr['name'])}</td>"
                f"<td style='padding:10px 12px;'>{tr['actual']}</td>"
                f"<td style='padding:10px 12px;'>{tr['opt']}</td>"
                f"<td style='padding:10px 12px;font-weight:700;color:{left_color};'>+{tr['left']}</td>"
                f"</tr>"
                f"<tr style='display:none;background:var(--surface2);'>"
                f"<td colspan='4' style='padding:12px;'>"
                f"<div style='font-size:12px;color:var(--muted);margin-bottom:8px;'>"
                f"<span style='color:#ef4444'>&#9660; Should have benched</span> &nbsp; "
                f"<span style='color:#22c55e'>&#9650; Should have started</span></div>"
                f"<div style='display:flex;flex-wrap:wrap;gap:6px;'>{badges}</div>"
                f"</td></tr>"
            )
        body_html = (
            f"<div class='card' style='overflow:auto;'>"
            f"<div class='card-header'><h3>Week {sel_week} - League Efficiency</h3></div>"
            f"<table style='width:100%;min-width:520px;border-collapse:collapse;'>"
            f"<thead><tr style='border-bottom:2px solid var(--border);'>"
            f"{_TH}TEAM</th>{_TH}ACTUAL</th>{_TH}OPTIMAL</th>{_TH}LEFT ON BENCH</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table></div>"
        ) if wk_rows else _no_data
        return nav_html + body_html

    # ── User views ────────────────────────────────────────────────────────────
    if not viewer_rid:
        return (nav_html +
                "<div class='card central'><div class='card-body' style='padding:24px;text-align:center'>"
                "<p style='color:var(--muted)'>Sign in to see your optimal lineup history.</p>"
                "</div></div>")

    weeks_data = _roster_weeks(viewer_rid)
    if not weeks_data:
        return (nav_html +
                "<div class='card central'><div class='card-body' style='padding:24px;text-align:center'>"
                "<p style='color:var(--muted)'>No lineup data found.</p></div></div>")

    # ── User weekly view ──────────────────────────────────────────────────────
    if period == "weekly":
        wd = next((w for w in weeks_data if w["week"] == sel_week), None)
        if not wd:
            return nav_html + _no_data
        left_color = "#ef4444" if wd["left"] > 15 else ("#f59e0b" if wd["left"] > 7 else "#22c55e")
        badges = _opt_player_badges(wd["players"])
        pos_html = _opt_pos_breakdown([wd])
        summary = (
            f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));"
            f"gap:12px;margin-bottom:16px;'>"
            f"<div class='card' style='padding:16px;text-align:center;'>"
            f"<div style='font-size:22px;font-weight:700;'>{wd['actual_pts']}</div>"
            f"<div style='font-size:11px;color:var(--muted);margin-top:4px;'>ACTUAL PTS</div></div>"
            f"<div class='card' style='padding:16px;text-align:center;'>"
            f"<div style='font-size:22px;font-weight:700;'>{wd['opt_pts']}</div>"
            f"<div style='font-size:11px;color:var(--muted);margin-top:4px;'>OPTIMAL PTS</div></div>"
            f"<div class='card' style='padding:16px;text-align:center;'>"
            f"<div style='font-size:22px;font-weight:700;color:{left_color};'>+{wd['left']}</div>"
            f"<div style='font-size:11px;color:var(--muted);margin-top:4px;'>LEFT ON BENCH</div></div>"
            f"</div>"
        )
        detail = (
            f"<div class='card'>"
            f"<div class='card-header'><h3>Week {sel_week} - Player Breakdown</h3></div>"
            f"<div style='padding:12px;'>"
            f"<div style='font-size:12px;color:var(--muted);margin-bottom:8px;'>"
            f"<span style='color:#ef4444'>&#9660; Should have benched</span> &nbsp; "
            f"<span style='color:#22c55e'>&#9650; Should have started</span></div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:6px;'>{badges}</div>"
            f"</div></div>"
        )
        return nav_html + summary + detail + pos_html

    # ── User season view ──────────────────────────────────────────────────────
    season_actual = sum(w["actual_pts"] for w in weeks_data)
    season_optimal = sum(w["opt_pts"] for w in weeks_data)
    season_left = round(sum(w["left"] for w in weeks_data), 1)
    efficiency = round((season_actual / season_optimal * 100) if season_optimal else 0, 1)
    worst_week = max(weeks_data, key=lambda x: x["left"])
    best_week = min(weeks_data, key=lambda x: x["left"])

    summary_html = (
        f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));"
        f"gap:12px;margin-bottom:20px;'>"
        f"<div class='card' style='padding:16px;text-align:center;'>"
        f"<div style='font-size:24px;font-weight:700;color:var(--accent);'>{season_left}</div>"
        f"<div style='font-size:11px;color:var(--muted);margin-top:4px;'>PTS LEFT ON BENCH</div></div>"
        f"<div class='card' style='padding:16px;text-align:center;'>"
        f"<div style='font-size:24px;font-weight:700;color:var(--text);'>{efficiency}%</div>"
        f"<div style='font-size:11px;color:var(--muted);margin-top:4px;'>LINEUP EFFICIENCY</div></div>"
        f"<div class='card' style='padding:16px;text-align:center;'>"
        f"<div style='font-size:24px;font-weight:700;color:#ef4444;'>Wk {worst_week['week']}</div>"
        f"<div style='font-size:11px;color:var(--muted);margin-top:4px;'>WORST (+{worst_week['left']} left)</div></div>"
        f"<div class='card' style='padding:16px;text-align:center;'>"
        f"<div style='font-size:24px;font-weight:700;color:#22c55e;'>Wk {best_week['week']}</div>"
        f"<div style='font-size:11px;color:var(--muted);margin-top:4px;'>BEST (+{best_week['left']} left)</div></div>"
        f"</div>"
    )

    rows_html = "".join(_wk_table_row(wd) for wd in reversed(weeks_data))
    table_html = (
        f"<div class='card' style='overflow:auto;'>"
        f"<div class='card-header' style='display:flex;justify-content:space-between;align-items:center;'>"
        f"<h3>Week-by-Week Breakdown</h3>"
        f"<span style='font-size:12px;color:var(--muted);'>Click row to expand</span></div>"
        f"<table style='width:100%;border-collapse:collapse;'>"
        f"<thead><tr style='border-bottom:1px solid var(--border);'>"
        f"{_TH}WEEK</th>{_TH}ACTUAL PTS</th>{_TH}OPTIMAL PTS</th>{_TH}LEFT ON BENCH</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table></div>"
    )

    pos_html = _opt_pos_breakdown(weeks_data)
    recurring_html = _opt_recurring_mistakes(weeks_data)

    return nav_html + summary_html + table_html + pos_html + recurring_html


# /<...>/optimal (redirect) is served by routes/league_pages_bp.py.
