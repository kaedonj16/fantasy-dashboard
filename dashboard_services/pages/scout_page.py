"""Opponent scouting report for the Weekly Hub Scout tab.

Moved from app.py so a smoke test can render a player row (including projected
PPG) without importing Flask. Live value-table lookup is lazy-imported from
app at request time.
"""
from __future__ import annotations

import html
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def platform_sign_in_hint(platform: str) -> str:
    """How to identify a team on this provider, used in unsigned-in empty states."""
    p = str(platform or "sleeper").strip().lower()
    return {
        "espn": "your ESPN team name",
        "yahoo": "your Yahoo team name",
        "mfl": "your MFL team name",
    }.get(p, "your Sleeper username")


def _live_model_value_table():
    """Live dynasty values. Overridable in tests so they never import Flask."""
    from app import get_model_value_table_cached
    return get_model_value_table_cached() or []


def _week_proj_points(week_map, pid, scoring=None, pos="") -> "float | None":
    """Numeric weekly projection for a player, scored for the league.

    Trusts Sleeper's published total for plain PPR/half/std leagues and
    recomputes from the raw stat line for custom scoring.
    """
    from utils.fantasy_scoring import weekly_projection_points
    return weekly_projection_points(week_map, pid, scoring, pos)


def build_scout_body(ctx: dict) -> str:
    viewer = ctx.get("viewer") or {}
    viewer_roster_id = str(viewer.get("viewer_roster_id") or "")
    platform = ctx.get("platform") or "sleeper"
    league_id = ctx.get("league_id") or ""
    season = ctx.get("season") or datetime.now().year
    current_week = ctx.get("current_week") or 0

    _sign_in_hint = platform_sign_in_hint(platform)
    _NOT_SIGNED_IN = (
        "<div class='card' style='text-align:center;padding:40px;'>"
        "<h2 style='margin-bottom:8px;'>Sign in to view your scouting report</h2>"
        f"<p style='color:var(--muted);'>Enter {_sign_in_hint} in the menu to unlock opponent scouting.</p>"
        "</div>"
    )

    if not viewer_roster_id:
        return _NOT_SIGNED_IN

    if ctx.get("offseason_mode"):
        return (
            "<div class='card' style='text-align:center;padding:40px;'>"
            "<h2 style='margin-bottom:8px;'>Scouting report is available during the regular season</h2>"
            "<p style='color:var(--muted);'>Check back once the season starts.</p>"
            "</div>"
        )

    rosters = ctx.get("rosters") or []
    roster_map = ctx.get("roster_map") or {}
    standings_map = ctx.get("standings_map") or {}
    model_value_table = ctx.get("model_value_table") or []
    try:
        live = _live_model_value_table() or []
        if live:
            model_value_table = live
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
    players_index = ctx.get("players_index") or {}
    matchups_by_week = ctx.get("matchups_by_week") or {}
    statuses = ctx.get("statuses") or {}
    proj_by_roster = ctx.get("proj_by_roster") or {}
    week_proj_map: dict = {}
    try:
        pw = ctx.get("proj_by_week") or {}
        week_proj_map = pw.get(current_week) or pw.get(str(current_week)) or {}
        if not isinstance(week_proj_map, dict):
            week_proj_map = {}
        if not week_proj_map:
            from utils.utils import load_week_projection
            week_proj_map = load_week_projection(int(season), int(current_week)) or {}
    except Exception:
        week_proj_map = {}

    values_by_id = {str(r.get("id") or ""): r for r in model_value_table if r.get("id")}

    # Find viewer's matchup for current week
    current_matchups = matchups_by_week.get(current_week) or []
    viewer_matchup = None
    opponent_roster_id = None
    opponent_team_block = None

    for m in current_matchups:
        # Live hub matchups use left/right; older/tour shapes used team1/team2.
        t1 = m.get("left") or m.get("team1") or {}
        t2 = m.get("right") or m.get("team2") or {}
        if str(t1.get("roster_id")) == viewer_roster_id:
            viewer_matchup = m
            opponent_roster_id = str(t2.get("roster_id"))
            opponent_team_block = t2
            break
        elif str(t2.get("roster_id")) == viewer_roster_id:
            viewer_matchup = m
            opponent_roster_id = str(t1.get("roster_id"))
            opponent_team_block = t1
            break

    if not viewer_matchup or not opponent_roster_id:
        return (
            f"<div class='card' style='text-align:center;padding:40px;'>"
            f"<h2 style='margin-bottom:8px;'>No matchup found for Week {current_week}</h2>"
            f"<p style='color:var(--muted);'>Your current week matchup could not be determined.</p>"
            f"</div>"
        )

    opponent_roster = next((r for r in rosters if str(r.get("roster_id")) == opponent_roster_id), None)
    if not opponent_roster:
        return "<div class='card'>Opponent roster not found.</div>"

    opp_name = html.escape(roster_map.get(opponent_roster_id, f"Roster {opponent_roster_id}"))
    opp_standing = standings_map.get(opponent_roster_id) or {}
    opp_wins = int(opp_standing.get("wins") or 0)
    opp_losses = int(opp_standing.get("losses") or 0)
    opp_pf = float(opp_standing.get("pf") or 0)
    opp_pa = float(opp_standing.get("pa") or 0)
    opp_rec_cls = "color-win" if opp_wins > opp_losses else ("color-loss" if opp_losses > opp_wins else "")

    # Starters from matchup block (dicts with pid) or roster id strings.
    starter_pids = set()
    for s in opponent_team_block.get("starters") or []:
        if not s:
            continue
        if isinstance(s, dict):
            pid = s.get("pid") or s.get("player_id")
        else:
            pid = s
        if pid:
            starter_pids.add(str(pid))
    opp_pts = opponent_team_block.get("pts_total")

    # Projected score for opponent
    opp_proj = proj_by_roster.get((current_week, opponent_roster_id))

    # Injury statuses
    status_by_pid = (statuses.get(current_week) or {}).get("statuses", {}) or {}

    # Build player list grouped by position
    _POS_ORDER = ["QB", "RB", "WR", "TE", "K", "DEF"]
    _POS_CLS = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "K": "K", "DEF": "DEF"}
    _INJ_CLS = {"Q": "inj-q", "D": "inj-d", "O": "inj-o", "IR": "inj-o", "Sus": "inj-o"}
    _INJ_LABEL = {"Q": "Q", "D": "D", "O": "O", "IR": "IR", "Questionable": "Q", "Doubtful": "D", "Out": "O",
                  "Suspended": "SUS"}

    all_pids = [str(p) for p in (opponent_roster.get("players") or [])]
    starters_by_pos: dict = {}
    bench_players = []

    for pid in all_pids:
        v = values_by_id.get(pid) or {}
        meta = players_index.get(pid) or {}
        pos = (v.get("position") or meta.get("pos") or "?").upper()
        val = float(v.get("value") or 0)
        name = v.get("name") or meta.get("name") or f"Player {pid}"
        team = (v.get("team") or meta.get("team") or "").upper()
        pos_rank = v.get("pos_rank_label") or ""
        is_starter = pid in starter_pids
        raw_inj = str(status_by_pid.get(pid) or "")
        inj_key = raw_inj if raw_inj in _INJ_CLS else None
        proj_ppg = _week_proj_points(week_proj_map, pid, ctx.get("raw_scoring_settings"), pos)

        entry = {
            "pid": pid, "name": name, "pos": pos, "team": team,
            "value": val, "pos_rank": pos_rank, "is_starter": is_starter,
            "inj_key": inj_key, "proj_ppg": proj_ppg,
        }
        if is_starter:
            starters_by_pos.setdefault(pos, []).append(entry)
        else:
            bench_players.append(entry)

    # Compute position group values and league-wide averages for strength/weakness
    _STARTER_COUNTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
    pos_group_vals = {}
    for pos, players in starters_by_pos.items():
        players.sort(key=lambda x: x["value"], reverse=True)
        pos_group_vals[pos] = sum(p["value"] for p in players)

    # League-wide average top-N value per position
    league_pos_avgs = {}
    for pos, top_n in _STARTER_COUNTS.items():
        sums = []
        for r in rosters:
            r_pids = [str(p) for p in (r.get("players") or [])]
            r_vals = sorted(
                [float((values_by_id.get(p) or {}).get("value") or 0) for p in r_pids
                 if (values_by_id.get(p) or {}).get("position", "").upper() == pos],
                reverse=True,
            )
            sums.append(sum(r_vals[:top_n]))
        league_pos_avgs[pos] = (sum(sums) / len(sums)) if sums else 0

    # Build strength/weakness summary
    strengths, weaknesses = [], []
    for pos in ["QB", "RB", "WR", "TE"]:
        opp_val = pos_group_vals.get(pos) or 0
        avg = league_pos_avgs.get(pos) or 1
        delta_pct = ((opp_val - avg) / avg) * 100 if avg else 0
        if delta_pct >= 12:
            strengths.append((pos, delta_pct, opp_val))
        elif delta_pct <= -12:
            weaknesses.append((pos, delta_pct, opp_val))

    strengths.sort(key=lambda x: -x[1])
    weaknesses.sort(key=lambda x: x[1])

    # Build HTML
    def _player_row(p, show_bench_label=False):
        pc = _POS_CLS.get(p["pos"], "pos-k")
        inj_html = ""
        if p.get("inj_key"):
            ic = _INJ_CLS.get(p["inj_key"], "")
            il = _INJ_LABEL.get(p["inj_key"], p["inj_key"])
            inj_html = f"<span class='inj-badge {ic}'>{il}</span>"
        pr_html = f"<span class='scout-pos-rank'>{html.escape(p.get('pos_rank', ''))}</span>" if p.get(
            "pos_rank") else ""
        ppg_html = (
            f"<span class='scout-ppg'>{p['proj_ppg']:.1f} proj</span>"
            if p.get("proj_ppg") is not None
            else "<span class='scout-ppg scout-ppg-miss' title='Week projection unavailable'>Proj unavailable</span>"
        )
        val_html = f"<span class='scout-val'>{p['value']:.0f}</span>" if p["value"] else ""
        bench_cls = " scout-bench" if show_bench_label else ""
        return (
            f"<div class='scout-player-row{bench_cls}'>"
            f"<span class='pos-badge {pc}'>{p['pos']}</span>"
            f"<span class='scout-player-name'>{html.escape(p['name'])}</span>"
            f"<span class='scout-team'>{html.escape(p['team'])}</span>"
            f"{pr_html}{inj_html}{ppg_html}{val_html}"
            f"</div>"
        )

    # Starters section
    starters_html = ""
    for pos in _POS_ORDER:
        for p in starters_by_pos.get(pos, []):
            starters_html += _player_row(p)
    # Handle any FLEX/unknown positions
    for pos, players in starters_by_pos.items():
        if pos not in _POS_ORDER:
            for p in players:
                starters_html += _player_row(p)

    bench_players.sort(key=lambda x: x["value"], reverse=True)
    bench_html = "".join(_player_row(p, show_bench_label=True) for p in bench_players[:10])

    # Strengths / weaknesses cards
    def _sw_chip(pos, delta_pct, val, is_strength):
        color = "var(--color-win)" if is_strength else "var(--color-loss)"
        arrow = "▲" if is_strength else "▼"
        pc = _POS_CLS.get(pos, "pos-k")
        return (
            f"<div class='scout-sw-chip'>"
            f"<span class='pos-badge {pc}'>{pos}</span>"
            f"<span class='scout-sw-val' style='color:{color};'>{arrow} {abs(delta_pct):.0f}% vs avg</span>"
            f"</div>"
        )

    sw_html = ""
    if strengths or weaknesses:
        s_chips = "".join(_sw_chip(pos, d, v, True) for pos, d, v in
                          strengths) or "<span style='color:var(--muted);font-size:0.85em;'>None notable</span>"
        w_chips = "".join(_sw_chip(pos, d, v, False) for pos, d, v in
                          weaknesses) or "<span style='color:var(--muted);font-size:0.85em;'>None notable</span>"
        sw_html = (
            f"<div class='main-two-col' style='margin-bottom:14px;'>"
            f"<div class='card'>"
            f"<div class='card-header'><h2>Their Strengths</h2></div>"
            f"<div class='card-body scout-sw-list'>{s_chips}</div>"
            f"</div>"
            f"<div class='card'>"
            f"<div class='card-header'><h2>Their Weaknesses</h2></div>"
            f"<div class='card-body scout-sw-list'>{w_chips}</div>"
            f"</div>"
            f"</div>"
        )

    proj_html = ""
    if opp_proj is not None:
        proj_html = f"<span class='scout-proj'>Proj: <strong>{opp_proj:.1f}</strong></span>"
    pts_html = ""
    if opp_pts is not None:
        pts_html = f"<span class='scout-proj' style='color:var(--muted);'>Score: <strong>{opp_pts:.1f}</strong></span>"

    return (
            f"<style>"
            f".scout-header-row{{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:4px;}}"
            f".scout-opp-name{{font-size:1.3em;font-weight:700;}}"
            f".scout-record{{font-size:0.9em;}}"
            f".scout-proj{{font-size:0.9em;background:var(--card);border:1px solid var(--border);border-radius:6px;padding:3px 8px;}}"
            f".scout-stat-row{{display:flex;gap:20px;flex-wrap:wrap;margin-top:6px;margin-bottom:16px;}}"
            f".scout-stat{{display:flex;flex-direction:column;gap:1px;}}"
            f".scout-stat-lbl{{font-size:0.7em;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;}}"
            f".scout-stat-val{{font-size:1em;font-weight:600;}}"
            f".scout-player-row{{display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid var(--border);font-size:0.88em;}}"
            f".scout-player-row:last-child{{border-bottom:none;}}"
            f".scout-bench{{opacity:.7;}}"
            f".scout-player-name{{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:500;}}"
            f".scout-team{{font-size:0.78em;color:var(--muted);min-width:28px;}}"
            f".scout-pos-rank{{font-size:0.75em;color:var(--muted);}}"
            f".scout-ppg{{font-size:0.75em;color:var(--muted);min-width:52px;text-align:right;}}"
            f".scout-ppg-miss{{min-width:88px;font-style:italic;}}"
            f".scout-proj-stamp{{font-size:0.75em;color:var(--muted);font-weight:500;margin-top:2px;}}"
            f".scout-val{{font-weight:600;color:var(--accent);font-size:0.85em;min-width:32px;text-align:right;}}"
            f".scout-sw-list{{display:flex;flex-direction:column;gap:8px;padding-top:4px;}}"
            f".scout-sw-chip{{display:flex;align-items:center;gap:8px;}}"
            f".scout-sw-val{{font-size:0.88em;font-weight:600;}}"
            f".inj-badge{{font-size:0.7em;font-weight:700;padding:1px 5px;border-radius:4px;line-height:1.4;}}"
            f".inj-q{{background:#fef08a;color:#713f12;}}"
            f".inj-d{{background:#fed7aa;color:#7c2d12;}}"
            f".inj-o{{background:#fecaca;color:#7f1d1d;}}"
            f"</style>"
            f"<div class='scout-header-row'>"
            f"<span class='scout-opp-name'>{opp_name}</span>"
            f"<span class='scout-record {opp_rec_cls}'>{opp_wins}-{opp_losses}</span>"
            f"{proj_html}{pts_html}"
            f"</div>"
            f"<div class='scout-stat-row'>"
            f"<div class='scout-stat'><span class='scout-stat-lbl'>Points For</span><span class='scout-stat-val'>{opp_pf:.1f}</span></div>"
            f"<div class='scout-stat'><span class='scout-stat-lbl'>Points Against</span><span class='scout-stat-val'>{opp_pa:.1f}</span></div>"
            f"</div>"
            f"{sw_html}"
            f"<div class='main-two-col'>"
            f"<div class='card'>"
            f"<div class='card-header'><h2>Week {current_week} Starters</h2>"
            f"<div class='scout-proj-stamp'>Week {current_week} Sleeper proj</div></div>"
            "<div class='card-body'>" + (
                    starters_html or "<p style='color:var(--muted);font-size:0.85em;'>Lineup not yet set.</p>") + "</div>"
                                                                                                                  "</div>"
                                                                                                                  "<div class='card'>"
                                                                                                                  f"<div class='card-header'><h2>Bench (Top 10)</h2></div>"
                                                                                                                  "<div class='card-body'>" + (
                    bench_html or "<p style='color:var(--muted);font-size:0.85em;'>No bench data.</p>") + "</div>"
                                                                                                          "</div>"
                                                                                                          "</div>"
    )
