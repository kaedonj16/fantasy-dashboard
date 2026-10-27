"""Opponent scouting report for the Weekly Hub Scout tab.

Weekly-focused, not dynasty. The old version leaned on the dynasty trade-value
table (big 400/1000 numbers) for both the player rows and the strength/weakness
math, which is meaningless in a weekly matchup and left real starters blank
whenever that table lacked a name. This version scouts *this week* as a
gameplan:

  1. "Gameplan" - the verdict (favored or dog, with the slide's own win
     probability), the derived calls from the positional edges, and the
     always-open positional edge by slot group (QB/RB/WR/TE/FLEX, plus K/DEF
     when the league starts them).
  2. "Their threats" - the opponent's lineup ranked by this week's
     projection (top 4, the rest behind a show-all toggle), with injury
     flags and a boom/bust profile, plus their soft spots, bench pivots and
     a status watch.

The boom/bust profile is a real distribution off a player's game-by-game
scores (floor, ceiling, a steady/volatile label), not a guess: it reuses
``utils.consistency`` on the same weekly Sleeper stat files the start/sit page
scores. Early in a season, when the current year has too few games, it leans on
last season and molds toward this one as games play (blended_consistency_profile).

Names come from the same players map the matchup preview uses, so rows never
render blank. Kept Flask-free (and bs4-free) so the slim unit/lint job can
render it.
"""
from __future__ import annotations

import html
import logging
from datetime import datetime

from utils.consistency import BLEND_FULL_SEASON, blended_consistency_profile

logger = logging.getLogger(__name__)

# Season weekly-points cache: {(season, scoring_sig): (loaded_at, {pid: [pts]})}.
# Rendering the Scout tab shouldn't re-glob and re-score a season's stat files on
# every request, so hold the scored map briefly (mirrors the start/sit loader).
_WEEKLY_PTS_CACHE: dict = {}
_WEEKLY_PTS_TTL = 3600.0


def _last_final_week(ctx: dict, fallback: int) -> int:
    """Last provider-finalized week from ctx's df_weekly (duck-typed, no
    pandas import: this module renders in the slim test job). Used only to
    mirror the matchup slide's projection-lookup inputs for win probability."""
    df = ctx.get("df_weekly")
    try:
        if (
            df is not None
            and not getattr(df, "empty", True)
            and {"week", "finalized"}.issubset(set(getattr(df, "columns", [])))
        ):
            finals = [
                int(row["week"])
                for row in df.to_dict("records")
                if row.get("finalized") is True and row.get("week") is not None
            ]
            if finals:
                return max(finals)
    except Exception:
        pass
    return int(fallback or 0)


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


def _stamped_scoring(ctx: dict) -> dict:
    """League scoring settings, alias-stamped the way the stat scorer expects."""
    raw = ctx.get("raw_scoring_settings") or ctx.get("scoring_settings") or {}
    if not raw:
        return {}
    try:
        from utils.league_scoring import stamp_scoring_aliases
        return stamp_scoring_aliases(raw)
    except Exception:
        return raw


def _season_weekly_points(season: int, scoring: dict) -> dict:
    """{pid: [weekly fantasy points]} for a season's played games, league-scored.

    Flask-free twin of app._load_season_weekly_points: globs the cached Sleeper
    weekly stat files and scores each line with ``score_stats``. Cached by
    (season, scoring signature); never raises.
    """
    import glob
    import hashlib
    import json
    import os
    import time
    from pathlib import Path

    sig = hashlib.md5(
        json.dumps(scoring or {}, sort_keys=True, default=str).encode()
    ).hexdigest()[:10]
    key = (int(season), sig)
    hit = _WEEKLY_PTS_CACHE.get(key)
    if hit and time.time() - hit[0] < _WEEKLY_PTS_TTL:
        return hit[1]

    out: dict = {}
    try:
        from utils.fantasy_scoring import score_stats, stat_row_is_appearance
        # scout_page.py -> dashboard_services/pages/ -> repo root is parents[2].
        base = Path(__file__).resolve().parents[2] / "cache" / "sleeper_stats"
        pattern = os.path.join(str(base), f"sleeper_stats_s{int(season)}_w*.json")
        for wf in glob.glob(pattern):
            try:
                with open(wf) as f:
                    week_stats = json.load(f)
            except Exception:
                continue
            if not isinstance(week_stats, dict):
                continue
            for pid, st in week_stats.items():
                if not isinstance(st, dict):
                    continue
                if not stat_row_is_appearance(st):
                    continue  # DNP row: not a game (twin parity with app.py).
                try:
                    out.setdefault(str(pid), []).append(
                        round(float(score_stats(st, scoring or {})), 2)
                    )
                except Exception:
                    continue
    except Exception:
        logger.debug("scout: weekly points load failed", exc_info=True)

    _WEEKLY_PTS_CACHE[key] = (time.time(), out)
    return out


def _resolve_weekly_points(ctx: dict, season: int) -> tuple[dict, dict, "int | None"]:
    """(current_map, prior_map, prior_season) of {pid: [weekly pts]}.

    Tests (and any caller that already computed them) can inject
    ``weekly_pts_map`` / ``prior_pts_map`` / ``prior_season`` on the ctx. Absent
    that, load from the cached stat files: the current season, plus last season
    (or the one before) while the current year is still a thin sample, so early
    in the year the profile leans on real history instead of one or two games.
    """
    if "weekly_pts_map" in ctx:
        return (
            ctx.get("weekly_pts_map") or {},
            ctx.get("prior_pts_map") or {},
            ctx.get("prior_season"),
        )

    scoring = _stamped_scoring(ctx)
    cur = _season_weekly_points(season, scoring)
    prior: dict = {}
    prior_season: "int | None" = None
    if max((len(v) for v in cur.values()), default=0) < BLEND_FULL_SEASON:
        for py in (int(season) - 1, int(season) - 2):
            pm = _season_weekly_points(py, scoring)
            if max((len(v) for v in pm.values()), default=0) >= 3:
                prior, prior_season = pm, py
                break
    return cur, prior, prior_season


# Accordion header for the positional-edge section (kept short so it reads well
# in a collapsed row next to the favored/underdog pill).
_EDGE_TITLE = "Positional Edge"


# Boom/bust profile label -> (css kind, short display). "Boom or bust" and
# "Volatile" both read as risk; "Steady" is the reassuring one.
_PROFILE_KIND = {
    "Boom or bust": ("boom", "Boom/bust"),
    "Volatile": ("volatile", "Volatile"),
    "Balanced": ("balanced", "Balanced"),
    "Steady": ("steady", "Steady"),
    "Small sample": ("small", "New/limited"),
}


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

    # Boom/bust profile per opponent starter, from real weekly scores. Best-effort:
    # if the stat files or math are unavailable, rows just render without a chip.
    try:
        cur_pts, prior_pts, prior_season = _resolve_weekly_points(ctx, int(season))
        for p in opp_starters:
            p["profile"] = blended_consistency_profile(
                cur_pts.get(p["pid"]) or [],
                prior_pts.get(p["pid"]) or [],
                p["pos"],
                prior_season=prior_season,
            )
    except Exception:
        logger.debug("scout: consistency profiles unavailable", exc_info=True)
        for p in opp_starters:
            p.setdefault("profile", None)

    have_proj = any(p["proj"] is not None for p in opp_starters + you_starters)

    you_total = sum(p["proj"] for p in you_starters if p["proj"] is not None)
    opp_total = sum(p["proj"] for p in opp_starters if p["proj"] is not None)

    # ── Slot grouping: edges are by lineup slot (QB/RB/WR/TE/FLEX/K/DEF), so
    # a flexed RB counts at FLEX, not RB, matching the matchup preview. ──────
    slot_order: list = []
    try:
        from dashboard_services.matchups import _canonical_slot_order
        slot_order = _canonical_slot_order(ctx.get("roster_positions") or [])
    except Exception:
        slot_order = []

    def _slot_group(slot: str) -> str:
        s = (slot or "").upper()
        if s in ("FLEX", "SUPER_FLEX", "SF", "OP", "RB/WR/TE", "W/R/T", "RB_WR", "WR_TE"):
            return "FLEX"
        if s in ("DEF", "D/ST", "DST"):
            return "DEF"
        return s

    def _with_slots(entries: list) -> list:
        """Pair each starter entry with its lineup-slot group (best effort).

        Without roster_positions the group is just the player's position.
        """
        if not slot_order:
            return [(e, e["pos"]) for e in entries]
        try:
            from dashboard_services.matchups import _realign_starters_to_slots
            aligned = _realign_starters_to_slots(list(entries), slot_order)
        except Exception:
            aligned = list(entries)
        out = []
        for i, e in enumerate(aligned):
            if e is None:
                continue
            slot = slot_order[i] if i < len(slot_order) else e["pos"]
            out.append((e, _slot_group(slot)))
        return out

    you_slotted = _with_slots(you_starters)
    opp_slotted = _with_slots(opp_starters)

    def _group_total(slotted: list, group: str) -> "float | None":
        vals = [e["proj"] for e, g in slotted if g == group and e["proj"] is not None]
        return sum(vals) if vals else None

    _GROUP_ORDER = ["QB", "RB", "WR", "TE", "FLEX", "K", "DEF"]
    edge_data: list = []  # (group, you_val|None, opp_val|None, delta)
    for g in _GROUP_ORDER:
        if not any(gr == g for _, gr in you_slotted + opp_slotted):
            continue
        yv = _group_total(you_slotted, g)
        ov = _group_total(opp_slotted, g)
        if yv is None and ov is None:
            continue
        edge_data.append((g, yv, ov, (yv or 0.0) - (ov or 0.0)))

    # ── Win probability: the exact model the matchup carousel uses, fed this
    # matchup's blocks and the slide's own projection lookups. ───────────────
    win_prob: "float | None" = None
    if have_proj:
        try:
            from dashboard_services.matchups import (
                build_slide_proj_lookups,
                compute_win_prob,
            )
            _wp_statuses = (
                (statuses.get(matchup_week) or statuses.get(current_week) or {})
                .get("statuses", {}) or {}
            )
            _wp_map, _wp_pid_proj, _wp_frac = build_slide_proj_lookups(
                season, matchup_week, _last_final_week(ctx, matchup_week),
                ctx.get("proj_by_week") or {}, ctx.get("team_game_lookup") or {},
                scoring_settings=ctx.get("raw_scoring_settings")
                or ctx.get("scoring_settings"),
            )
            _p = compute_win_prob(
                viewer_block, opponent_block, _wp_statuses, _wp_map,
                frac_lookup=_wp_frac,
            )
            if _p is not None:
                win_prob = float(_p)
        except Exception:
            logger.debug("scout: win prob unavailable", exc_info=True)

    # ── Verdict ───────────────────────────────────────────────────────────────
    you_standing = standings_map.get(viewer_roster_id)
    if not isinstance(you_standing, dict):
        you_standing = standings_map.get(str(viewer_roster_id))
    if not isinstance(you_standing, dict):
        you_standing = {}
    you_roster = next(
        (r for r in rosters if str(r.get("roster_id")) == viewer_roster_id), None)
    you_settings = (you_roster or {}).get("settings") or {}
    you_wins = int(you_standing.get("wins") or you_settings.get("wins") or 0)
    you_losses = int(you_standing.get("losses") or you_settings.get("losses") or 0)

    same_div = False
    try:
        from utils.standings_divisions import div_map_for_ctx
        _dm = div_map_for_ctx(ctx) or {}
        def _div_of(rid: str):
            for key in (rid, str(rid)):
                if key in _dm:
                    return _dm[key]
            try:
                return _dm.get(int(rid))
            except (TypeError, ValueError):
                return None
        _dv, _do = _div_of(viewer_roster_id), _div_of(opponent_roster_id or "")
        same_div = _dv is not None and _dv == _do
    except Exception:
        same_div = False

    total_delta = you_total - opp_total
    verdict_dog = total_delta < 0
    if have_proj:
        chip = ""
        if win_prob is not None:
            chip = (
                f"<span class='scout-wchip{' is-dog' if win_prob < 0.5 else ''}'>"
                f"{win_prob * 100:.0f}% win prob</span>"
            )
        if abs(total_delta) < 1.0:
            verdict_line = f"Dead even with {opp_name} {chip}"
        elif total_delta > 0:
            verdict_line = (
                f"You are favored by <em>{total_delta:.1f}</em> vs {opp_name} {chip}")
        else:
            verdict_line = (
                f"You are underdogs by <em>{abs(total_delta):.1f}</em> vs {opp_name} {chip}")
        verdict_sub = (
            f"{you_total:.1f} vs {opp_total:.1f} proj · {opp_name} "
            f"{opp_wins}-{opp_losses}, you {you_wins}-{you_losses} · "
            f"Week {matchup_week}" + (" · divisional" if same_div else "")
        )
        wpbar = (
            f"<div class='scout-wpbar'><span style='width:{win_prob * 100:.0f}%;'></span></div>"
            if win_prob is not None else ""
        )
    else:
        verdict_line = (
            f"{opp_name} <span class='scout-record {opp_rec_cls}'>"
            f"{opp_wins}-{opp_losses}</span>")
        verdict_sub = f"Week {matchup_week} · projections not available yet"
        wpbar = ""

    # ── Gameplan calls: derived from the slot edges ───────────────────────────
    def _group_names(slotted: list, group: str, limit: int = 2) -> list:
        es = [e for e, g in slotted if g == group and e["proj"] is not None]
        es.sort(key=lambda e: e["proj"], reverse=True)
        return es[:limit]

    def _join_names(entries: list) -> str:
        names = [html.escape(e["name"]) for e in entries]
        if len(names) <= 2:
            return " and ".join(names)
        return ", ".join(names)

    # Largest single-slot (1 starter vs 1 starter) edge, for the callout.
    _single_deltas = {}
    for g, yv, ov, delta in edge_data:
        if len(_group_names(you_slotted, g, 9)) == 1 and len(_group_names(opp_slotted, g, 9)) == 1:
            _single_deltas[g] = delta
    _best_single = max(_single_deltas.values(), default=None)

    plan_rows: list = []
    _used_groups: set = set()
    pos_edges = sorted((d for d in edge_data if d[3] >= 1.0), key=lambda d: -d[3])
    neg_edges = sorted((d for d in edge_data if d[3] <= -1.0), key=lambda d: d[3])
    for g, yv, ov, delta in pos_edges[:2]:
        yn = _group_names(you_slotted, g)
        on = _group_names(opp_slotted, g)
        if len(yn) == 1 and len(on) == 1:
            ev = (f"<b>{html.escape(yn[0]['name'])} +{delta:.1f} over "
                  f"{html.escape(on[0]['name'])}</b>")
            if _best_single is not None and delta == _best_single:
                ev += " <span>· biggest single-slot edge</span>"
        else:
            ev = (f"<b>+{delta:.1f} edge</b> <span>· {_join_names(yn)} vs "
                  f"{_join_names(on)}</span>")
        plan_rows.append(("win", f"Win it at {g}", ev))
        _used_groups.add(g)
    if neg_edges:
        g, yv, ov, delta = neg_edges[0]
        yn = _group_names(you_slotted, g)
        on = _group_names(opp_slotted, g)
        if len(yn) == 1 and len(on) == 1:
            ev = (f"<b>{html.escape(on[0]['name'])} +{abs(delta):.1f} over "
                  f"{html.escape(yn[0]['name'])}</b>")
            if len(neg_edges) == 1:
                ev += " <span>· the one slot they take</span>"
        else:
            ev = (f"<b>{abs(delta):.1f} edge to them</b> <span>· {_join_names(on)} vs "
                  f"{_join_names(yn)}</span>")
        plan_rows.append(("risk", f"Survive at {g}", ev))
        _used_groups.add(g)

    # ── Positional Edge (always open, by slot) ────────────────────────────────
    def _edge_row(g: str, yv, ov, delta: float) -> str:
        yv2, ov2 = (yv or 0.0), (ov or 0.0)
        total = yv2 + ov2
        you_share = (yv2 / total * 100.0) if total > 0 else 50.0
        if abs(delta) < 1.0:
            pill_cls, pill = "edge-even", "Even"
        elif delta > 0:
            pill_cls, pill = "edge-you", f"You +{delta:.1f}"
        else:
            pill_cls, pill = "edge-them", f"Them +{abs(delta):.1f}"
        you_txt = f"{yv:.1f}" if yv is not None else "–"
        opp_txt = f"{ov:.1f}" if ov is not None else "–"
        return (
            f"<div class='scout-edge'>"
            f"<span class='pos-badge {g}'>{g}</span>"
            f"<span class='scout-edge-you'>{you_txt}</span>"
            f"<span class='scout-edge-bar'>"
            f"<span class='scout-edge-fill' style='width:{you_share:.0f}%;'></span>"
            f"</span>"
            f"<span class='scout-edge-opp'>{opp_txt}</span>"
            f"<span class='scout-edge-pill {pill_cls}'>{pill}</span>"
            f"</div>"
        )

    if have_proj and edge_data:
        if abs(total_delta) < 1.0:
            head_cls, head_txt = "edge-even", "Dead even"
        elif total_delta > 0:
            head_cls, head_txt = "edge-you", f"You favored by {total_delta:.1f}"
        else:
            head_cls, head_txt = "edge-them", f"Underdog by {abs(total_delta):.1f}"
        edge_section = (
            f"<div class='scout-edge-sec'>"
            f"<div class='scout-edge-head'><b>{_EDGE_TITLE}</b>"
            f"<span class='scout-edge-pill {head_cls}'>{head_txt}</span></div>"
            + "".join(_edge_row(g, yv, ov, d) for g, yv, ov, d in edge_data)
            + "</div>"
        )
    else:
        edge_section = (
            f"<div class='scout-edge-sec'>"
            f"<div class='scout-edge-head'><b>{_EDGE_TITLE}</b></div>"
            "<div class='scout-noproj'>"
            "Weekly projections aren't available yet, so the positional edge "
            "can't be computed. Check back closer to kickoff.</div></div>"
        )

    # ── Section 2: opponent threat report ─────────────────────────────────────
    def _proj_sort_key(p):
        return p["proj"] if p["proj"] is not None else -1.0

    ranked = sorted(opp_starters, key=_proj_sort_key, reverse=True)
    banged_up = [p for p in opp_starters if p["inj_raw"] in _INJ_SITTING]

    def _profile_html(prof: "dict | None") -> str:
        """Boom/bust chip + floor-ceiling range, with the rates in the tooltip."""
        if not prof:
            return ""
        kind, short = _PROFILE_KIND.get(prof.get("label", ""), ("balanced", prof.get("label", "")))
        floor, ceiling = prof.get("floor"), prof.get("ceiling")
        range_html = ""
        if floor is not None and ceiling is not None:
            range_html = f"<span class='scout-range'>{floor:.0f}–{ceiling:.0f}</span>"
        boom = int(round((prof.get("boom_rate") or 0) * 100))
        bust = int(round((prof.get("bust_rate") or 0) * 100))
        season_note = ""
        if prof.get("blended") is False and prof.get("season"):
            season_note = f", {prof['season']} data"
        title = (
            f"{prof.get('label', '')}: floor {floor}, ceiling {ceiling} "
            f"(20th-80th pct of weekly scores){season_note}. "
            f"Boom {boom}% / bust {bust}% of games."
        )
        return (
            f"<span class='scout-profile scout-prof-{kind}' title='{html.escape(title)}'>"
            f"{html.escape(short)}</span>{range_html}"
        )

    def _threat_row(rank: int, p: dict) -> str:
        pc = p["pos"] if p["pos"] in _POS_ORDER or p["pos"] == "FLEX" else "K"
        inj_html = ""
        if p["inj_key"]:
            ic = _INJ_CLS.get(p["inj_key"], "")
            il = _INJ_LABEL.get(p["inj_key"], p["inj_key"])
            inj_html = f"<span class='inj-badge {ic}'>{il}</span>"
        if p["proj"] is not None:
            ppg_html = f"<span class='scout-ppg'>{p['proj']:.1f} proj</span>"
        else:
            ppg_html = (
                "<span class='scout-ppg scout-ppg-miss' "
                "title='Week projection unavailable'>Proj unavailable</span>"
            )
        team_html = f"<span class='scout-team'>{html.escape(p['team'])}</span>" if p["team"] else ""
        return (
            f"<div class='scout-trow'>"
            f"<span class='scout-rk'>{rank}</span>"
            f"<span class='pos-badge {pc}'>{p['pos']}</span>"
            f"<span class='scout-tid'>"
            f"<span class='scout-tname'>{html.escape(p['name'])}</span>"
            f"{team_html}{inj_html}"
            f"</span>"
            f"<span class='scout-prof-wrap'>{_profile_html(p.get('profile'))}</span>"
            f"{ppg_html}"
            f"</div>"
        )

    # Top 4 threats up front; the rest behind a client-side show-all toggle.
    threat_rows = "".join(_threat_row(i + 1, p) for i, p in enumerate(ranked[:4]))
    extra_rows = "".join(
        _threat_row(i + 5, p) for i, p in enumerate(ranked[4:]))
    if extra_rows:
        threat_rows += (
            f"<div class='scout-threat-extra' hidden>{extra_rows}</div>"
            f"<button type='button' class='scout-showall' data-showall "
            f"data-more='Show all {len(ranked)} starters' "
            f"data-less='Show fewer starters'>Show all {len(ranked)} starters</button>"
        )
    if not threat_rows:
        threat_rows = "<p style='color:var(--muted);font-size:0.9em;'>Lineup not yet set.</p>"

    # Third gameplan call when the edges alone don't fill three: watch their
    # top threat at a slot the calls haven't covered.
    if len(plan_rows) < 3:
        for p in ranked:
            if p["proj"] is None or p["pos"] in _used_groups:
                continue
            plan_rows.append((
                "risk", f"Watch their {p['pos']}",
                f"<b>{html.escape(p['name'])} {p['proj']:.1f} proj</b> "
                f"<span>· their top threat at the slot</span>",
            ))
            _used_groups.add(p["pos"])
            break
    plan_html = "".join(
        f"<div class='scout-plan-row pr-{kind}'>"
        f"<div class='scout-plan-k {kind}'>{label}</div><p>{ev}</p></div>"
        for kind, label, ev in plan_rows
    )

    # One-line volatility read: only when the lineup leans clearly one way, and
    # only off players with a real (non-small-sample) profile.
    read_note = ""
    profiled = [p for p in opp_starters if p.get("profile") and not p["profile"].get("small_sample")]
    if len(profiled) >= 4:
        risky = sum(1 for p in profiled if p["profile"]["label"] in ("Boom or bust", "Volatile"))
        steady = sum(1 for p in profiled if p["profile"]["label"] == "Steady")
        if risky >= max(3, len(profiled) // 2):
            read_note = (
                f"<div class='scout-read'>High-variance lineup: {risky} of their "
                f"{len(profiled)} starters run hot or cold, so a quiet week from them is in play.</div>"
            )
        elif steady >= max(3, len(profiled) // 2):
            read_note = (
                f"<div class='scout-read'>Steady lineup: {steady} of their "
                f"{len(profiled)} starters are consistent, so don't count on them cratering.</div>"
            )

    # ── Soft spots + pivots ───────────────────────────────────────────────────
    def _slot_label(entry: dict) -> str:
        """Slot label for a starter, e.g. RB2 for their second RB-slot back."""
        for e, g in opp_slotted:
            if e is entry:
                same = [x for x, gg in opp_slotted if gg == g]
                if len(same) > 1:
                    return f"{g}{same.index(e) + 1}"
                return g
        return entry["pos"]

    soft_rows: list = []  # (tag|None, text_html)
    opp_by_proj = sorted(
        (p for p in opp_starters if p["proj"] is not None),
        key=lambda p: p["proj"],
    )
    you_min = min(
        (p["proj"] for p in you_starters if p["proj"] is not None), default=None)
    if opp_by_proj:
        low = opp_by_proj[0]
        where = (
            "is the lowest starter on either side."
            if you_min is None or low["proj"] <= you_min
            else "is their lowest starter."
        )
        soft_rows.append((
            f"{_slot_label(low)} · {low['proj']:.1f}",
            f"{html.escape(low['name'])} {where}",
        ))
    if len(opp_by_proj) >= 2:
        second = opp_by_proj[1]
        label = (second.get("profile") or {}).get("label", "")
        if label in ("Boom or bust", "Volatile"):
            txt = f"{html.escape(second['name'])} runs hot or cold."
        else:
            txt = f"{html.escape(second['name'])} is next-lowest at {second['proj']:.1f}."
        soft_rows.append((f"{_slot_label(second)} · {second['proj']:.1f}", txt))

    # Their bench: a real pivot (out-projects a starter) or a thin one.
    opp_roster = next(
        (r for r in rosters if str(r.get("roster_id")) == opponent_roster_id), None)
    _opp_starter_pid_set = {p["pid"] for p in opp_starters}
    bench_proj: list = []
    for pid in (opp_roster or {}).get("players") or []:
        pid = str(pid)
        if pid in _opp_starter_pid_set:
            continue
        bname, bpos, _bteam = _resolve_player(pid, players_map, players_index)
        bproj = _week_proj_points(week_proj_map, pid, scoring, bpos)
        if bproj is not None:
            bench_proj.append((bname, bproj))
    if bench_proj:
        bname, bmax = max(bench_proj, key=lambda x: x[1])
        if opp_by_proj and bmax > opp_by_proj[0]["proj"]:
            soft_rows.append((
                "Bench · live",
                f"{html.escape(bname)} ({bmax:.1f}) out-projects their "
                f"{_slot_label(opp_by_proj[0])} starter. They have a move to make.",
            ))
        else:
            soft_rows.append((
                "Bench · thin",
                f"No bench player over {bmax:.1f} proj, no real pivot.",
            ))

    # Status watch: a starter who may not play, and what his absence does to
    # the slot edge if we can price it.
    if banged_up:
        names = ", ".join(html.escape(p["name"]) for p in banged_up[:3])
        more = f" +{len(banged_up) - 3} more" if len(banged_up) > 3 else ""
        sit_txt = f"{names}{more} may not play."
        top_sit = max(banged_up, key=lambda p: p["proj"] or 0.0)
        if top_sit["proj"]:
            for e, g in opp_slotted:
                if e is top_sit:
                    _gd = next((d for gg, _y, _o, d in edge_data if gg == g), None)
                    if _gd is not None:
                        sit_txt += (
                            f" If {html.escape(top_sit['name'])} sits, your {g} "
                            f"edge goes to about +{_gd + top_sit['proj']:.0f}. "
                            f"No move needed."
                        )
                    break
        soft_rows.append(("Status", sit_txt))

    # The read: their path vs your cushion, from the edges and top threats.
    top_threats = [p for p in ranked if p["proj"] is not None][:2]
    best_edge = max(edge_data, key=lambda d: d[3]) if edge_data else None
    read_txt = ""
    if top_threats and best_edge is not None:
        t_names = " and ".join(html.escape(p["name"]) for p in top_threats)
        if win_prob is not None and win_prob < 0.5:
            read_txt = (
                f"Your path: {t_names} stay near their floors, and your "
                f"{best_edge[0]} hits its ceiling.")
        elif win_prob is not None:
            read_txt = (
                f"They need {t_names} at their ceilings to flip this. "
                f"Your {best_edge[0]} edge is the cushion.")
        else:
            read_txt = f"It swings on {t_names} against your {best_edge[0]}."
    if read_txt:
        soft_rows.append((None, f"<b>The read:</b> {read_txt}"))

    soft_html = ""
    if soft_rows:
        rows_html = "".join(
            f"<div class='scout-soft-row'>"
            + (f"<span class='scout-soft-tag'>{html.escape(tag)}</span> " if tag else "")
            + f"<span>{txt}</span></div>"
            for tag, txt in soft_rows
        )
        soft_html = (
            "<div class='scout-soft-sec'>"
            "<b class='scout-soft-title'>Soft spots + your pivots</b>"
            f"{rows_html}</div>"
        )

    threat_stamp = (
        f"<div class='scout-proj-stamp' style='margin-bottom:8px;'>"
        f"Week {current_week} Sleeper proj · boom/bust from weekly scores</div>"
    )

    gameplan_card = (
        f"<div class='card scout-card'>"
        f"<div class='card-header'><h2>Gameplan</h2></div>"
        f"<div class='card-body'>"
        f"<div class='scout-verdict{' is-dog' if verdict_dog else ''}'>{verdict_line}</div>"
        f"<div class='scout-verdict-sub'>{html.escape(verdict_sub)}</div>"
        f"{wpbar}"
        + (f"<div class='scout-plan'>{plan_html}</div>" if plan_html else "")
        + f"{edge_section}"
        f"</div></div>"
    )
    threat_section = (
        f"<div class='card scout-card'>"
        f"<div class='card-header'><h2>Their threats</h2>"
        f"<span class='scout-threat-sub'>ranked by projection · floor-ceiling</span></div>"
        f"<div class='card-body'>{threat_stamp}{read_note}{threat_rows}{soft_html}</div>"
        f"</div>"
    )

    return _SCOUT_STYLE + (
        f"<div class='scout-report'>"
        f"{gameplan_card}"
        f"{threat_section}"
        f"</div>"
    ) + _SCOUT_JS


_SCOUT_STYLE = (
    "<style>"
    ".scout-report{display:flex;flex-direction:column;gap:14px;}"
    ".scout-record{font-size:0.95em;font-weight:600;}"
    ".scout-record.color-win{color:var(--win);}"
    ".scout-record.color-loss{color:var(--loss);}"
    ".scout-card{margin:0;}"
    # Gameplan card: verdict + win prob, the derived calls, then the edge.
    ".scout-verdict{font-size:1.18em;font-weight:800;}"
    ".scout-verdict em{font-style:normal;color:var(--win);}"
    ".scout-verdict.is-dog em{color:var(--loss);}"
    ".scout-wchip{display:inline-block;background:var(--win);color:#fff;font-weight:800;font-size:0.72em;border-radius:999px;padding:2px 10px;vertical-align:2px;margin-left:6px;white-space:nowrap;}"
    ".scout-wchip.is-dog{background:var(--loss);}"
    ".scout-verdict-sub{color:var(--muted);font-size:0.88em;margin-top:2px;}"
    ".scout-wpbar{height:6px;border-radius:3px;background:color-mix(in srgb,var(--loss) 45%,transparent);overflow:hidden;margin-top:10px;}"
    ".scout-wpbar span{display:block;height:100%;background:var(--win);}"
    ".scout-plan{margin-top:10px;}"
    ".scout-plan-row{display:flex;gap:12px;align-items:baseline;border:1px solid var(--border);border-radius:10px;padding:9px 12px;margin-bottom:8px;font-size:0.94em;}"
    ".scout-plan-row:last-child{margin-bottom:0;}"
    ".scout-plan-row.pr-win{background:color-mix(in srgb,var(--win) 5%,transparent);border-color:color-mix(in srgb,var(--win) 25%,var(--border));}"
    ".scout-plan-row.pr-risk{background:color-mix(in srgb,var(--warning) 6%,transparent);border-color:color-mix(in srgb,var(--warning) 28%,var(--border));}"
    ".scout-plan-k{font-size:0.72em;font-weight:800;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);flex:0 0 108px;}"
    ".scout-plan-k.win{color:var(--win);}"
    ".scout-plan-k.risk{color:color-mix(in srgb,var(--warning) 85%,var(--text));}"
    ".scout-plan-row p{margin:0;}"
    ".scout-plan-row p span{color:var(--muted);}"
    ".scout-edge-sec{border-top:1px solid var(--border);margin-top:12px;padding-top:10px;}"
    ".scout-edge-head{display:flex;align-items:center;gap:10px;}"
    ".scout-edge-head b{font-size:0.98em;}"
    ".scout-edge-head .scout-edge-pill{margin-left:auto;}"
    ".scout-noproj{color:var(--muted);font-size:0.9em;padding:10px 0 4px;}"
    ".scout-edge{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid var(--border);}"
    ".scout-edge:last-child{border-bottom:none;}"
    ".scout-edge-you,.scout-edge-opp{min-width:40px;font-weight:600;font-size:0.9em;}"
    ".scout-edge-you{text-align:right;}"
    ".scout-edge-opp{text-align:left;}"
    ".scout-edge-bar{flex:1;height:10px;border-radius:5px;background:color-mix(in srgb,var(--loss) 55%,transparent);overflow:hidden;min-width:60px;}"
    ".scout-edge-fill{display:block;height:100%;background:var(--win);border-radius:4px 0 0 4px;}"
    ".scout-edge-pill{font-size:0.75em;font-weight:700;padding:2px 8px;border-radius:999px;white-space:nowrap;min-width:64px;text-align:center;}"
    ".edge-you{background:color-mix(in srgb,var(--win) 16%,transparent);color:var(--win);}"
    ".edge-them{background:color-mix(in srgb,var(--loss) 16%,transparent);color:var(--loss);}"
    ".edge-even{background:color-mix(in srgb,var(--muted) 16%,transparent);color:var(--muted);}"
    # Threat rows: rank, badge, name + team, profile chip + range, projection.
    ".scout-threat-sub{color:var(--muted);font-size:0.8em;margin-left:auto;}"
    ".scout-trow{display:flex;align-items:center;gap:9px;flex-wrap:wrap;padding:7px 0;border-bottom:1px solid var(--border);font-size:0.94em;}"
    ".scout-trow:last-child{border-bottom:none;}"
    ".scout-rk{color:var(--warning);font-weight:800;font-size:0.8em;width:14px;flex:0 0 auto;}"
    ".scout-tid{flex:1 1 auto;min-width:0;display:flex;align-items:center;gap:8px;}"
    ".scout-tname{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:700;}"
    ".scout-team{font-size:0.78em;color:var(--muted);flex-shrink:0;}"
    ".scout-ppg{font-size:0.88em;font-weight:800;min-width:66px;text-align:right;white-space:nowrap;}"
    ".scout-ppg-miss{min-width:92px;font-style:italic;font-weight:400;color:var(--muted);}"
    ".scout-proj-stamp{font-size:0.72em;color:var(--muted);font-weight:500;}"
    ".scout-showall{display:block;margin:8px auto 0;background:none;border:0;font:inherit;font-size:0.9em;font-weight:700;color:var(--accent);cursor:pointer;padding:4px 8px;}"
    ".scout-read{font-size:0.84em;color:var(--text);background:var(--card-soft,color-mix(in srgb,var(--muted) 8%,transparent));border:1px solid var(--border);border-radius:6px;padding:7px 10px;margin-bottom:10px;}"
    ".scout-prof-wrap{display:inline-flex;align-items:center;gap:5px;justify-content:flex-end;min-width:118px;}"
    ".scout-profile{font-size:0.68em;font-weight:700;padding:2px 7px;border-radius:999px;white-space:nowrap;cursor:default;}"
    ".scout-prof-steady{background:color-mix(in srgb,var(--win) 16%,transparent);color:var(--win);}"
    ".scout-prof-balanced{background:color-mix(in srgb,var(--muted) 16%,transparent);color:var(--muted);}"
    ".scout-prof-volatile{background:color-mix(in srgb,var(--warning) 20%,transparent);color:color-mix(in srgb,var(--warning) 85%,var(--text));}"
    ".scout-prof-boom{background:color-mix(in srgb,var(--loss) 16%,transparent);color:var(--loss);}"
    ".scout-prof-small{background:color-mix(in srgb,var(--muted) 12%,transparent);color:var(--muted);font-style:italic;font-weight:500;}"
    ".scout-range{font-size:0.78em;color:var(--muted);min-width:38px;text-align:right;white-space:nowrap;}"
    ".inj-badge{font-size:0.7em;font-weight:700;padding:1px 5px;border-radius:4px;line-height:1.4;}"
    ".inj-q{background:#fef08a;color:#713f12;}"
    ".inj-d{background:#fed7aa;color:#7c2d12;}"
    ".inj-o{background:#fecaca;color:#7f1d1d;}"
    # Soft spots + pivots, closing with the read.
    ".scout-soft-sec{border-top:1px solid var(--border);margin-top:10px;padding-top:8px;}"
    ".scout-soft-title{font-size:0.98em;}"
    ".scout-soft-row{padding:7px 0;border-bottom:1px solid var(--border);font-size:0.94em;}"
    ".scout-soft-row:last-child{border-bottom:none;}"
    ".scout-soft-row span{color:var(--muted);}"
    ".scout-soft-row b{color:var(--text);}"
    ".scout-soft-tag{display:inline-block;background:var(--card-soft,#f1f5f9);border:1px solid var(--border);border-radius:6px;padding:0 6px;font-weight:800;font-size:0.8em;color:var(--text);}"
    # Narrow screens (phones): identity takes the first line so names are never
    # truncated; profile/proj metrics drop to a second line under the name.
    "@media (max-width:560px){"
    ".scout-tid{flex-basis:100%;}"
    ".scout-tname{white-space:normal;overflow:visible;text-overflow:clip;}"
    ".scout-prof-wrap{min-width:0;justify-content:flex-start;margin-left:62px;}"
    ".scout-ppg{margin-left:auto;}"
    ".scout-plan-k{flex-basis:92px;}"
    "}"
    "</style>"
)


# Threats show-all toggle: expands the starters past the top 4 in place.
# Inline so scout_page stays self-contained; binds once per report.
_SCOUT_JS = (
    "<script>(function(){"
    "var reports=document.querySelectorAll('.scout-report');"
    "for(var i=0;i<reports.length;i++){"
    "var r=reports[i];if(r.__scoutBound)continue;r.__scoutBound=true;"
    "var btn=r.querySelector('[data-showall]');"
    "var extra=r.querySelector('.scout-threat-extra');"
    "if(!btn||!extra)continue;"
    "btn.addEventListener('click',function(){"
    "var show=extra.hidden;extra.hidden=!show;"
    "btn.textContent=show?btn.getAttribute('data-less'):btn.getAttribute('data-more');"
    "});"
    "}"
    "})();</script>"
)
