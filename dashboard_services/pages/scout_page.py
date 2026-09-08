"""Opponent scouting report for the Weekly Hub Scout tab.

Weekly-focused, not dynasty. The old version leaned on the dynasty trade-value
table (big 400/1000 numbers) for both the player rows and the strength/weakness
math, which is meaningless in a weekly matchup and left real starters blank
whenever that table lacked a name. This version scouts *this week*:

  1. "Where the matchup is won" - your projected starters vs theirs, by
     position, so you can see where you're favored and where you're the dog.
  2. "Their starters" - the opponent's lineup ranked by this week's projection,
     with injury flags, so the biggest threats are on top.

Names come from the same players map the matchup preview uses, so rows never
render blank. Kept Flask-free (and bs4-free) so the slim unit/lint job can
render it.
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
        "fleaflicker": "your Fleaflicker team name",
    }.get(p, "your Sleeper username")


def _week_proj_points(week_map, pid, scoring=None, pos="") -> "float | None":
    """Numeric weekly projection for a player, scored for the league.

    Trusts Sleeper's published total for plain PPR/half/std leagues and
    recomputes from the raw stat line for custom scoring.
    """
    from utils.fantasy_scoring import weekly_projection_points
    return weekly_projection_points(week_map, pid, scoring, pos)


# Positions we break the matchup down by, in display order. K/DEF are appended
# only when a lineup actually starts them (some leagues don't).
_POS_ORDER = ["QB", "RB", "WR", "TE", "K", "DEF"]
_PRIMARY_POS = ["QB", "RB", "WR", "TE"]
_INJ_CLS = {"Q": "inj-q", "D": "inj-d", "O": "inj-o", "IR": "inj-o", "Sus": "inj-o"}
_INJ_LABEL = {
    "Q": "Q", "D": "D", "O": "O", "IR": "IR", "Sus": "SUS",
    "Questionable": "Q", "Doubtful": "D", "Out": "O", "Suspended": "SUS",
}
_INJ_SITTING = {"O", "IR", "Sus", "Out", "Suspended"}  # very likely not playing


def _resolve_player(pid: str, players_map: dict, players_index: dict) -> tuple[str, str, str]:
    """(name, pos, team) for a pid, using the same sources as the matchup view.

    Mirrors utils.from_players_map without importing it (that module pulls bs4,
    which the slim test job lacks). Falls back to a readable D/ST label for the
    2-3 letter team codes platforms use for defenses.
    """
    info = players_map.get(pid) or players_index.get(pid) or {}
    name = info.get("name") or info.get("full_name") or ""
    if not name:
        first = info.get("first_name") or ""
        last = info.get("last_name") or ""
        name = " ".join(x for x in (first, last) if x)
    pos = (info.get("pos") or info.get("position") or "").upper()
    team = (info.get("team") or "").upper()

    is_def_code = pid.isalpha() and 2 <= len(pid) <= 3
    if not name and is_def_code:
        name, pos, team = f"{pid} D/ST", "DEF", pid
    if not pos and is_def_code:
        pos, team = "DEF", pid
    if not name:
        name = f"Player {pid}"
    return name, (pos or "?"), team


def _starter_pids(team_block: dict) -> list[str]:
    """Starter pids from a matchup team block (dicts with pid, or bare ids)."""
    out: list[str] = []
    for s in team_block.get("starters") or []:
        if not s:
            continue
        pid = s.get("pid") or s.get("player_id") if isinstance(s, dict) else s
        if pid:
            out.append(str(pid))
    return out


def _build_starters(pids, players_map, players_index, week_proj_map, status_by_pid, scoring):
    """List of starter entries: pos, name, team, proj (float|None), injury key."""
    entries = []
    for pid in pids:
        name, pos, team = _resolve_player(pid, players_map, players_index)
        raw_inj = str(status_by_pid.get(pid) or "")
        inj_key = raw_inj if raw_inj in _INJ_CLS else None
        entries.append({
            "pid": pid,
            "name": name,
            "pos": pos,
            "team": team,
            "proj": _week_proj_points(week_proj_map, pid, scoring, pos),
            "inj_key": inj_key,
            "inj_raw": raw_inj,
        })
    return entries


def build_scout_body(ctx: dict) -> str:
    viewer = ctx.get("viewer") or {}
    viewer_roster_id = str(viewer.get("viewer_roster_id") or "")
    platform = ctx.get("platform") or "sleeper"
    season = ctx.get("season") or datetime.now().year
    current_week = ctx.get("current_week") or 0

    _sign_in_hint = platform_sign_in_hint(platform)
    if not viewer_roster_id:
        return (
            "<div class='card' style='text-align:center;padding:40px;'>"
            "<h2 style='margin-bottom:8px;'>Sign in to view your scouting report</h2>"
            f"<p style='color:var(--muted);'>Enter {_sign_in_hint} in the menu to unlock opponent scouting.</p>"
            "</div>"
        )

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
    players_index = ctx.get("players_index") or {}
    players_map = ctx.get("players_map") or {}
    matchups_by_week = ctx.get("matchups_by_week") or {}
    statuses = ctx.get("statuses") or {}
    scoring = ctx.get("raw_scoring_settings")

    week_proj_map: dict = {}
    try:
        from utils.week_proj import week_proj_map_from_bundles
        week_proj_map = week_proj_map_from_bundles(ctx.get("proj_by_week") or {}, current_week)
        if not week_proj_map:
            from utils.utils import load_week_projection
            week_proj_map = load_week_projection(int(season), int(current_week)) or {}
    except Exception:
        week_proj_map = {}

    # Find the viewer's matchup for the current week; capture both sides.
    from utils.matchup_schedule import resolve_matchup_week
    matchup_week = resolve_matchup_week(current_week, matchups_by_week)
    current_matchups = (
        matchups_by_week.get(matchup_week)
        or matchups_by_week.get(str(matchup_week))
        or []
    )
    viewer_block = opponent_block = None
    opponent_roster_id = None
    for m in current_matchups:
        # Live hub uses left/right; older/tour shapes used team1/team2.
        t1 = m.get("left") or m.get("team1") or {}
        t2 = m.get("right") or m.get("team2") or {}
        if str(t1.get("roster_id")) == viewer_roster_id:
            viewer_block, opponent_block = t1, t2
            opponent_roster_id = str(t2.get("roster_id"))
            break
        if str(t2.get("roster_id")) == viewer_roster_id:
            viewer_block, opponent_block = t2, t1
            opponent_roster_id = str(t1.get("roster_id"))
            break

    if not opponent_block or not opponent_roster_id:
        return (
            f"<div class='card' style='text-align:center;padding:40px;'>"
            f"<h2 style='margin-bottom:8px;'>No matchup found for Week {current_week}</h2>"
            f"<p style='color:var(--muted);'>Your current week matchup could not be determined.</p>"
            f"</div>"
        )

    opponent_roster = next((r for r in rosters if str(r.get("roster_id")) == opponent_roster_id), None)

    opp_name = html.escape(roster_map.get(opponent_roster_id, f"Roster {opponent_roster_id}"))
    opp_standing = standings_map.get(opponent_roster_id)
    if not isinstance(opp_standing, dict):
        opp_standing = standings_map.get(str(opponent_roster_id))
    if not isinstance(opp_standing, dict):
        opp_standing = {}
    opp_settings = (opponent_roster or {}).get("settings") or {}
    opp_wins = int(opp_standing.get("wins") or opp_settings.get("wins") or 0)
    opp_losses = int(opp_standing.get("losses") or opp_settings.get("losses") or 0)
    opp_rec_cls = "color-win" if opp_wins > opp_losses else ("color-loss" if opp_losses > opp_wins else "")

    status_by_pid = (statuses.get(current_week) or {}).get("statuses", {}) or {}

    opp_starters = _build_starters(
        _starter_pids(opponent_block), players_map, players_index,
        week_proj_map, status_by_pid, scoring,
    )
    you_starters = _build_starters(
        _starter_pids(viewer_block or {}), players_map, players_index,
        week_proj_map, status_by_pid, scoring,
    )

    have_proj = any(p["proj"] is not None for p in opp_starters + you_starters)

    def _pos_total(starters: list, pos: str) -> "float | None":
        vals = [p["proj"] for p in starters if p["pos"] == pos and p["proj"] is not None]
        return sum(vals) if vals else None

    you_total = sum(p["proj"] for p in you_starters if p["proj"] is not None)
    opp_total = sum(p["proj"] for p in opp_starters if p["proj"] is not None)

    # ── Section 1: positional edge ────────────────────────────────────────────
    positions = [p for p in _POS_ORDER if p in _PRIMARY_POS]
    for pos in ("K", "DEF"):
        if any(p["pos"] == pos for p in opp_starters + you_starters):
            positions.append(pos)

    def _edge_row(pos: str) -> str:
        you_p = _pos_total(you_starters, pos)
        opp_p = _pos_total(opp_starters, pos)
        if you_p is None and opp_p is None:
            return ""
        yv, ov = (you_p or 0.0), (opp_p or 0.0)
        total = yv + ov
        you_share = (yv / total * 100.0) if total > 0 else 50.0
        delta = yv - ov
        if abs(delta) < 1.0:
            pill_cls, pill = "edge-even", "Even"
        elif delta > 0:
            pill_cls, pill = "edge-you", f"You +{delta:.1f}"
        else:
            pill_cls, pill = "edge-them", f"Them +{abs(delta):.1f}"
        pc = pos if pos in _POS_ORDER else "K"
        you_txt = f"{yv:.1f}" if you_p is not None else "—"
        opp_txt = f"{ov:.1f}" if opp_p is not None else "—"
        return (
            f"<div class='scout-edge'>"
            f"<span class='pos-badge {pc}'>{pos}</span>"
            f"<span class='scout-edge-you'>{you_txt}</span>"
            f"<span class='scout-edge-bar'>"
            f"<span class='scout-edge-fill' style='width:{you_share:.0f}%;'></span>"
            f"</span>"
            f"<span class='scout-edge-opp'>{opp_txt}</span>"
            f"<span class='scout-edge-pill {pill_cls}'>{pill}</span>"
            f"</div>"
        )

    edge_rows = "".join(_edge_row(pos) for pos in positions)

    if have_proj and edge_rows:
        total_delta = you_total - opp_total
        if abs(total_delta) < 1.0:
            head_cls, head_txt = "edge-even", "Dead even"
        elif total_delta > 0:
            head_cls, head_txt = "edge-you", f"You favored by {total_delta:.1f}"
        else:
            head_cls, head_txt = "edge-them", f"Underdog by {abs(total_delta):.1f}"
        edge_section = (
            f"<div class='card scout-card'>"
            f"<div class='card-header scout-edge-header'>"
            f"<h2>Where the matchup is won</h2>"
            f"<span class='scout-edge-pill {head_cls}'>{head_txt}</span>"
            f"</div>"
            f"<div class='card-body'>"
            f"<div class='scout-edge scout-edge-key'>"
            f"<span class='pos-badge' style='visibility:hidden;'>·</span>"
            f"<span class='scout-edge-you'>You {you_total:.1f}</span>"
            f"<span class='scout-edge-bar' style='visibility:hidden;'></span>"
            f"<span class='scout-edge-opp'>{opp_total:.1f} Them</span>"
            f"<span class='scout-edge-pill' style='visibility:hidden;'>·</span>"
            f"</div>"
            f"{edge_rows}"
            f"</div></div>"
        )
    else:
        edge_section = (
            "<div class='card scout-card'><div class='card-body' "
            "style='color:var(--muted);font-size:0.9em;padding:18px;'>"
            "Weekly projections aren't available yet, so the positional edge "
            "can't be computed. Check back closer to kickoff.</div></div>"
        )

    # ── Section 2: opponent threat report ─────────────────────────────────────
    def _proj_sort_key(p):
        return p["proj"] if p["proj"] is not None else -1.0

    ranked = sorted(opp_starters, key=_proj_sort_key, reverse=True)
    top_pid = ranked[0]["pid"] if ranked and ranked[0]["proj"] is not None else None
    banged_up = [p for p in opp_starters if p["inj_raw"] in _INJ_SITTING]

    def _threat_row(p: dict) -> str:
        pc = p["pos"] if p["pos"] in _POS_ORDER else "K"
        inj_html = ""
        if p["inj_key"]:
            ic = _INJ_CLS.get(p["inj_key"], "")
            il = _INJ_LABEL.get(p["inj_key"], p["inj_key"])
            inj_html = f"<span class='inj-badge {ic}'>{il}</span>"
        tag_html = ""
        if p["pid"] == top_pid:
            tag_html = "<span class='scout-tag scout-tag-threat'>Top threat</span>"
        if p["proj"] is not None:
            ppg_html = f"<span class='scout-ppg'>{p['proj']:.1f} proj</span>"
        else:
            ppg_html = (
                "<span class='scout-ppg scout-ppg-miss' "
                "title='Week projection unavailable'>Proj unavailable</span>"
            )
        team_html = f"<span class='scout-team'>{html.escape(p['team'])}</span>" if p["team"] else ""
        return (
            f"<div class='scout-player-row'>"
            f"<span class='pos-badge {pc}'>{p['pos']}</span>"
            f"<span class='scout-player-name'>{html.escape(p['name'])}</span>"
            f"{team_html}{tag_html}{inj_html}{ppg_html}"
            f"</div>"
        )

    threat_rows = "".join(_threat_row(p) for p in ranked)
    if not threat_rows:
        threat_rows = "<p style='color:var(--muted);font-size:0.9em;'>Lineup not yet set.</p>"

    inj_note = ""
    if banged_up:
        names = ", ".join(html.escape(p["name"]) for p in banged_up[:3])
        more = f" +{len(banged_up) - 3} more" if len(banged_up) > 3 else ""
        inj_note = (
            f"<div class='scout-inj-note'>"
            f"⚠ {names}{more} may not play — a chance to gain ground."
            f"</div>"
        )

    threat_section = (
        f"<div class='card scout-card'>"
        f"<div class='card-header scout-edge-header'>"
        f"<h2>Their starters</h2>"
        f"<span class='scout-proj-stamp'>Week {current_week} Sleeper proj</span>"
        f"</div>"
        f"<div class='card-body'>{inj_note}{threat_rows}</div>"
        f"</div>"
    )

    return _SCOUT_STYLE + (
        f"<div class='scout-report'>"
        f"<div class='scout-header-row'>"
        f"<span class='scout-opp-name'>{opp_name}</span>"
        f"<span class='scout-record {opp_rec_cls}'>{opp_wins}-{opp_losses}</span>"
        f"</div>"
        f"{edge_section}"
        f"{threat_section}"
        f"</div>"
    )


_SCOUT_STYLE = (
    "<style>"
    ".scout-report{display:flex;flex-direction:column;gap:14px;}"
    ".scout-header-row{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;}"
    ".scout-opp-name{font-size:1.3em;font-weight:700;}"
    ".scout-record{font-size:0.95em;font-weight:600;}"
    ".scout-record.color-win{color:var(--win);}"
    ".scout-record.color-loss{color:var(--loss);}"
    ".scout-card{margin:0;}"
    ".scout-edge-header{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;}"
    ".scout-edge{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid var(--border);}"
    ".scout-edge:last-child{border-bottom:none;}"
    ".scout-edge-key{font-size:0.72em;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;font-weight:600;padding-bottom:2px;}"
    ".scout-edge-you,.scout-edge-opp{min-width:40px;font-weight:600;font-size:0.9em;}"
    ".scout-edge-you{text-align:right;}"
    ".scout-edge-opp{text-align:left;}"
    ".scout-edge-bar{flex:1;height:8px;border-radius:4px;background:color-mix(in srgb,var(--loss) 55%,transparent);overflow:hidden;min-width:60px;}"
    ".scout-edge-fill{display:block;height:100%;background:var(--win);border-radius:4px 0 0 4px;}"
    ".scout-edge-pill{font-size:0.75em;font-weight:700;padding:2px 8px;border-radius:999px;white-space:nowrap;min-width:64px;text-align:center;}"
    ".edge-you{background:color-mix(in srgb,var(--win) 16%,transparent);color:var(--win);}"
    ".edge-them{background:color-mix(in srgb,var(--loss) 16%,transparent);color:var(--loss);}"
    ".edge-even{background:color-mix(in srgb,var(--muted) 16%,transparent);color:var(--muted);}"
    ".scout-player-row{display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid var(--border);font-size:0.9em;}"
    ".scout-player-row:last-child{border-bottom:none;}"
    ".scout-player-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:600;}"
    ".scout-team{font-size:0.78em;color:var(--muted);min-width:30px;}"
    ".scout-ppg{font-size:0.82em;font-weight:600;min-width:66px;text-align:right;}"
    ".scout-ppg-miss{min-width:92px;font-style:italic;font-weight:400;color:var(--muted);}"
    ".scout-proj-stamp{font-size:0.72em;color:var(--muted);font-weight:500;}"
    ".scout-tag{font-size:0.68em;font-weight:700;padding:2px 7px;border-radius:999px;text-transform:uppercase;letter-spacing:.03em;}"
    ".scout-tag-threat{background:color-mix(in srgb,var(--warning) 20%,transparent);color:color-mix(in srgb,var(--warning) 85%,var(--text));}"
    ".scout-inj-note{font-size:0.82em;color:color-mix(in srgb,var(--warning) 85%,var(--text));background:color-mix(in srgb,var(--warning) 10%,transparent);border:1px solid color-mix(in srgb,var(--warning) 30%,var(--border));border-radius:6px;padding:7px 10px;margin-bottom:10px;}"
    ".inj-badge{font-size:0.7em;font-weight:700;padding:1px 5px;border-radius:4px;line-height:1.4;}"
    ".inj-q{background:#fef08a;color:#713f12;}"
    ".inj-d{background:#fed7aa;color:#7c2d12;}"
    ".inj-o{background:#fecaca;color:#7f1d1d;}"
    "</style>"
)
