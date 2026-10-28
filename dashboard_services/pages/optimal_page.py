"""Lineup tab for the Weekly Hub: This Week (current vs projected-optimal)
plus the historical Lineup Lab (completed weeks) as the History view."""
from __future__ import annotations

import html
import logging
from datetime import datetime
from flask import request

from utils.optimal_lineup import analyze_lineup, analyze_team_week

logger = logging.getLogger(__name__)


def verified_completed_weeks(df_weekly, *, season_complete=False, matchups_by_week=None):
    """Return provider-verified completed weeks, retaining playoff weeks."""
    if (df_weekly is not None and not getattr(df_weekly, "empty", True)
            and {"week", "finalized"}.issubset(df_weekly.columns)):
        # ``to_dict`` also keeps this helper testable in the lightweight CI job,
        # which intentionally does not install pandas.  Do not use truthiness:
        # only an explicit provider-finalized value makes a week complete.
        records = df_weekly.to_dict("records")
        return sorted({int(row["week"]) for row in records
                       if row.get("finalized") is True and row.get("week") is not None})
    if season_complete:
        return sorted(int(w) for w in (matchups_by_week or {}).keys())
    return []


def _esc(value) -> str:
    return html.escape(str(value or ""), quote=True)


def _player(pid, players, score) -> str:
    if not pid:
        return '<span class="opt-empty">Empty slot</span>'
    info = players.get(str(pid)) or {}
    name = info.get("name") or str(pid)
    pos = (info.get("pos") or "—").upper()
    return (f'<span class="opt-player"><span class="opt-pos opt-pos-{_esc(pos.lower())}">{_esc(pos)}</span>'
            f'<button type="button" class="opt-player-name player-clickable" data-player-id="{_esc(pid)}" '
            f'data-player-name="{_esc(name)}">{_esc(name)}</button>'
            f'<strong class="opt-score">{score:.1f}</strong></span>')


def _comparison(data, players, *, open_=False) -> str:
    scores = data["scores"]
    changed = {i for g in data.get("groups", []) for i in g["slots"]}
    rows = []
    for i, slot in enumerate(data["slots"]):
        actual, optimal = data["actual_assignment"][i], data["optimal_assignment"][i]
        a_score = scores.get(actual, 0.0) if actual else 0.0
        o_score = scores.get(optimal, 0.0) if optimal else 0.0
        delta = o_score - a_score
        rows.append(
            f'<div class="opt-lineup-row{" is-changed" if i in changed else ""}">'
            f'<div class="opt-slot">{_esc(slot.replace("_", "/"))}</div>'
            f'<div class="opt-side"><span class="opt-side-label">Actual</span>{_player(actual, players, a_score)}</div>'
            f'<div class="opt-arrow" aria-hidden="true">→</div>'
            f'<div class="opt-side"><span class="opt-side-label">Optimal after results</span>{_player(optimal, players, o_score)}</div>'
            f'<div class="opt-gain">{"%+.1f" % delta if i in changed else "—"}</div></div>')
    explanation = ""
    for group in data.get("groups", []):
        incoming = ", ".join((players.get(p) or {}).get("name") or p for p in group["incoming"]) or "empty slot"
        outgoing = ", ".join((players.get(p) or {}).get("name") or p for p in group["outgoing"]) or "empty slot"
        label = "Grouped FLEX/Superflex change" if len(group["slots"]) > 1 else "Lineup change"
        explanation += (f'<div class="opt-group-note"><strong>{label}:</strong> {_esc(incoming)} replaced '
                        f'{_esc(outgoing)} → Missed gain {group["gain"]:.1f}</div>')
    bench = [p for p in data["pids"] if p not in set(data["actual_assignment"])]
    bench_html = "".join(f'<div class="opt-bench-player">{_player(p, players, scores[p])}</div>' for p in bench)
    return (f'<div class="opt-comparison">{"".join(rows)}{explanation}'
            f'<details class="opt-bench"><summary>Show bench</summary><div class="opt-bench-grid">{bench_html}</div></details></div>')


def _metric(value, label, cls=""):
    return f'<div class="card opt-metric {cls}"><strong>{value}</strong><span>{label}</span></div>'


def _build_optimal_history(ctx):
    from app import get_players_index_global
    from dashboard_services.platform_api import get_matchups

    platform = ctx.get("platform") or "sleeper"
    season = int(ctx.get("season") or datetime.now().year)
    league_id = ctx.get("league_id") or ""
    viewer_rid = str((ctx.get("viewer") or {}).get("viewer_roster_id") or "")
    slots = ctx.get("roster_positions") or []
    raw_players = get_players_index_global() or {}
    players = {}
    rosters, roster_map = ctx.get("rosters") or [], ctx.get("roster_map") or {}

    # Finalized rows are the source of truth; this works for historical seasons and playoffs.
    df = ctx.get("df_weekly")
    completed = verified_completed_weeks(
        df, season_complete=bool(ctx.get("season_complete")),
        matchups_by_week=ctx.get("matchups_by_week"),
    )
    if not completed:
        return '<div class="opt-empty-state">No verified completed weeks yet.</div>'

    explicit_view = request.args.get("view")
    explicit_period = request.args.get("period")
    view = explicit_view if explicit_view in {"user", "league"} else "user"
    period = explicit_period if explicit_period in {"weekly", "season"} else "weekly"
    try:
        selected = int(request.args.get("week", completed[-1]))
    except (TypeError, ValueError):
        selected = completed[-1]
    if selected not in completed:
        selected = completed[-1]
    base = f'/{platform}/{season}/{league_id}/weekly?tab=optimal'

    def tab(label, key, val, active):
        qview = val if key == "view" else view
        qperiod = val if key == "period" else period
        # Keep the last weekly selection in Season URLs so returning to Weekly
        # (or switching team/league views) does not unexpectedly jump weeks.
        wk = f'&week={selected}'
        return f'<a class="opt-tab{" active" if active else ""}" href="{base}&view={qview}&period={qperiod}{wk}">{label}</a>'
    options = "".join(f'<option value="{w}"{" selected" if w == selected else ""}>Week {w}</option>' for w in completed)
    nav = (f'<h1 class="opt-page-title page-title" style="margin-bottom:12px;">Lineup Lab</h1>'
           f'<nav class="opt-nav" aria-label="Lineup analysis controls"><div class="opt-tab-group">'
           f'{tab("My Team", "view", "user", view == "user")}{tab("League", "view", "league", view == "league")}</div>'
           f'<div class="opt-tab-group">{tab("Weekly", "period", "weekly", period == "weekly")}'
           f'{tab("Season", "period", "season", period == "season")}</div>'
           + (f'<label class="sr-only" for="optWeek">Completed week</label><select id="optWeek" class="opt-week-select" '
              f'data-base-url="{base}&view={view}&period=weekly">{options}</select>' if period == "weekly" else "")
           + '</nav><p class="opt-method">Optimal lineup uses final results and your league’s roster rules.</p>')

    weeks_to_fetch = [selected] if period == "weekly" else completed
    matchup_cache = ctx.setdefault("optimal_matchups_by_week", {})
    for week in weeks_to_fetch:
        if week not in matchup_cache:
            try:
                matchup_cache[week] = get_matchups(platform, league_id, week, season) or []
            except (LookupError, ValueError, RuntimeError, OSError) as exc:
                logger.warning("lineup matchup unavailable platform=%s league=%s week=%s: %s", platform, league_id, week, exc)
                matchup_cache[week] = None

    def analyze(rid, week):
        out = analyze_team_week(matchup_cache.get(week), rid, raw_players, slots, week=week)
        players.update(out.get("players") or {})
        return out

    def incomplete(d):
        reason = d.get("reason") or "one or more player scores or positions are unavailable"
        details = []
        for field, label in (("missing_scores", "missing score"),
                             ("unknown_positions", "missing position")):
            for pid in d.get(field) or []:
                name = (players.get(str(pid)) or {}).get("name") or str(pid)
                details.append(f"{name}: {label}")
        detail_html = (f'<ul>{"".join(f"<li>{_esc(item)}</li>" for item in details)}</ul>'
                       if details else "")
        return (f'<div class="card opt-incomplete"><strong>Incomplete scoring data</strong>'
                f'<span>{_esc(reason)}. Efficiency and missed points are withheld.</span>{detail_html}</div>')

    def weekly_panel(d, title=""):
        if not d.get("complete"):
            return incomplete(d)
        eff = "—" if d["efficiency"] is None else f'{d["efficiency"]:.1f}%'
        official = d.get("official_total")
        adjustment = ""
        if official is not None and abs(float(official) - d["actual"]) > .005:
            adjustment = (f'<div class="opt-adjustment">Official provider total: <strong>{float(official):.2f}</strong>. '
                          f'It differs from the player starter sum by {float(official)-d["actual"]:+.2f} due to provider scoring adjustments.</div>')
        return ((f'<h3 class="opt-section-title">{_esc(title)}</h3>' if title else "") + '<div class="opt-summary">'
                + _metric(f'{d["actual"]:.2f}', "Actual starter points")
                + _metric(f'{d["optimal"]:.2f}', "Optimal points")
                + _metric(f'{d["missed"]:.2f}', "Missed points", "is-accent")
                + _metric(eff, "Lineup efficiency") + f'</div>{adjustment}{_comparison(d, players)}')

    if view == "user" and not viewer_rid:
        return nav + '<div class="opt-empty-state">Sign in to see your lineup history.</div>'

    if view == "user":
        data = [analyze(viewer_rid, w) for w in weeks_to_fetch]
        if period == "weekly":
            return nav + weekly_panel(data[0], f'Week {selected} lineup')
        good = [d for d in data if d.get("complete")]
        actual, optimal = sum(d["actual"] for d in good), sum(d["optimal"] for d in good)
        missed = sum(d["missed"] for d in good)
        eff = f'{actual / optimal * 100:.1f}%' if optimal > 0 else "—"
        summary = ('<div class="opt-summary">' + _metric(eff, "Lineup efficiency")
                   + _metric(f'{missed:.1f}', "Total missed points", "is-accent")
                   + _metric(f'{missed / len(good):.1f}' if good else "—", "Average missed / week")
                   + _metric(f'{len(good)} / {len(data)}', "Weeks analyzed") + '</div>')
        history = ""
        for d in reversed(data):
            label = (f'Actual {d["actual"]:.1f} · Optimal {d["optimal"]:.1f} · Missed {d["missed"]:.1f} · '
                     f'{d["efficiency"]:.1f}%' if d.get("complete") and d.get("efficiency") is not None else "Incomplete scoring data")
            history += f'<details class="card opt-week"><summary><strong>Week {d["week"]}</strong><span>{label}</span></summary>{weekly_panel(d)}</details>'
        return nav + summary + f'<div class="opt-history">{history}</div>'

    teams = []
    for roster in rosters:
        rid = str(roster.get("roster_id") or "")
        if not rid:
            continue
        ds = [analyze(rid, w) for w in weeks_to_fetch]
        good = [d for d in ds if d.get("complete")]
        actual, optimal = sum(d["actual"] for d in good), sum(d["optimal"] for d in good)
        teams.append({"rid": rid, "name": roster_map.get(rid) or f'Team {rid}', "data": ds, "good": good,
                      "actual": actual, "optimal": optimal, "missed": sum(d["missed"] for d in good),
                      "eff": actual / optimal * 100 if optimal > 0 else None})
    teams.sort(key=lambda t: (t["eff"] is None, -(t["eff"] or 0)))
    cards = ""
    for rank, team in enumerate(teams, 1):
        eff_num = team["eff"]
        eff = "—" if eff_num is None else f'{eff_num:.1f}%'
        bar_w = 0 if eff_num is None else max(0, min(100, eff_num))
        eff_warn = eff_num is not None and eff_num < 85
        missed_hi = team["missed"] > 0.005
        rank_tint = f" opt-rank-{rank}" if rank <= 3 else ""
        cards += (f'<details class="card opt-team{" is-viewer" if team["rid"] == viewer_rid else ""}"><summary>'
                  f'<span class="opt-rank{rank_tint}">#{rank}</span>'
                  f'<span class="opt-team-identity"><strong class="opt-team-name">{_esc(team["name"])}</strong>'
                  f'<span class="opt-team-sub">{team["actual"]:.1f} of {team["optimal"]:.1f} pts</span></span>'
                  f'<span class="opt-team-hero{" is-warn" if eff_warn else ""}">'
                  f'<span class="opt-team-label">Efficiency</span>'
                  f'<span class="opt-team-effnum">{eff}</span>'
                  f'<span class="opt-team-bar" aria-hidden="true"><span style="width:{bar_w:.1f}%"></span></span></span>'
                  f'<span class="opt-team-stats">'
                  f'<span class="opt-team-stat"><span class="opt-team-label">Actual</span><span class="opt-team-value">{team["actual"]:.1f}</span></span>'
                  f'<span class="opt-team-stat"><span class="opt-team-label">Optimal</span><span class="opt-team-value">{team["optimal"]:.1f}</span></span>'
                  f'<span class="opt-team-stat"><span class="opt-team-label">Missed</span>'
                  f'<span class="opt-team-value{" is-missed" if missed_hi else ""}">{team["missed"]:.1f}</span></span></span>'
                  f'<span class="opt-team-weeks"><span class="opt-team-label">Weeks counted</span> '
                  f'<span class="opt-team-value">{len(team["good"])} of {len(team["data"])} weeks</span></span></summary>'
                  + ''.join(weekly_panel(d, f'Week {d["week"]}') for d in reversed(team["data"])) + '</details>')
    heading = ('<div class="opt-leaderboard-head">Efficiency leaderboard · sorted highest first</div>'
               '<div class="opt-leaderboard-columns" aria-hidden="true"><span>Rank</span><span>Team</span>'
               '<span>Efficiency</span><span>Actual</span><span>Optimal</span><span>Missed</span></div>')
    return nav + f'<div class="opt-leaderboard">{heading}{cards}</div>'


# ── This Week: current starters vs the projected-optimal lineup ──────────────

_POS_PLURAL = {"QB": "QBs", "RB": "RBs", "WR": "WRs", "TE": "TEs",
               "K": "kickers", "DEF": "defenses"}


def _lw_last_final_week(ctx, fallback):
    """Last provider-finalized week from ctx's df_weekly (duck-typed, mirrors
    the slide projection-lookup inputs)."""
    df = ctx.get("df_weekly")
    try:
        if (df is not None and not getattr(df, "empty", True)
                and {"week", "finalized"}.issubset(set(getattr(df, "columns", [])))):
            finals = [int(r["week"]) for r in df.to_dict("records")
                      if r.get("finalized") is True and r.get("week") is not None]
            if finals:
                return max(finals)
    except Exception:
        pass
    return int(fallback or 0)


def _lw_ordinal(n):
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _lw_block_pids(block):
    out = []
    for s in (block or {}).get("starters") or []:
        pid = s.get("pid") if isinstance(s, dict) else s
        if pid not in (None, "", "0"):
            out.append(str(pid))
    return out


def _lw_player_info(ctx, wanted_pids=()):
    """pid -> {name, pos, team} merged from ctx maps; the global players
    index is consulted only when a wanted pid is still unnamed."""
    info = {}

    def _add(pid, name, pos, team):
        pid = str(pid)
        if not name:
            return
        cur = info.setdefault(pid, {})
        if name and not cur.get("name"):
            cur["name"] = name
        if pos and not cur.get("pos"):
            cur["pos"] = str(pos).upper()
        if team and not cur.get("team"):
            cur["team"] = str(team).upper()

    for pid, p in (ctx.get("players_index") or {}).items():
        if isinstance(p, dict):
            _add(pid, p.get("name"), p.get("pos"), p.get("team") or p.get("nfl"))
    for pid, p in (ctx.get("players_map") or {}).items():
        if isinstance(p, dict):
            nm = p.get("full_name") or " ".join(
                x for x in [p.get("first_name"), p.get("last_name")] if x) or p.get("name")
            _add(pid, nm, p.get("position") or p.get("pos"), p.get("team"))
    if any(not (info.get(str(pid)) or {}).get("name") for pid in wanted_pids):
        try:
            from app import get_players_index_global
            for pid, p in (get_players_index_global() or {}).items():
                if isinstance(p, dict):
                    _add(pid, p.get("name"), p.get("pos"), p.get("team") or p.get("nfl"))
        except Exception:
            pass
    return info


def _lw_stable_optimal_assignment(slots, actual_asg, solver_asg, positions):
    """Re-seat the solver's optimal starter SET for display.

    assign_optimal_lineup maximizes points, not slot stability, so its raw
    assignment can rotate continuing starters across slots (Walker RB1 to
    RB2, Lamb WR to FLEX) depending on roster iteration order. Diffed
    slot-by-slot, that rotation reads as phantom swaps with wrong deltas.
    This returns a legal assignment of the same set that keeps every
    continuing starter in his actual slot whenever one exists (exact search
    maximizing kept slots); it falls back to the solver's own assignment if
    no stable seating is found."""
    from functools import lru_cache

    from utils.lineups import canonicalize_slot, slot_eligible_positions

    solver_set = {p for p in solver_asg if p}
    actual_set = {p for p in actual_asg if p}
    if not solver_set:
        return list(solver_asg)
    if solver_set == actual_set:
        return list(actual_asg)
    players = [p for p in actual_asg if p in solver_set]
    players += [p for p in solver_asg if p and p not in actual_set]
    if len(players) != len(slots):
        return list(solver_asg)
    pos = {p: canonicalize_slot(positions.get(p)) for p in players}
    eligible = [
        [j for j, p in enumerate(players)
         if pos[p] and pos[p] in slot_eligible_positions(slot)]
        for slot in slots
    ]
    if any(not e for e in eligible):
        return list(solver_asg)

    @lru_cache(maxsize=None)
    def _best(i, used):
        if i == len(slots):
            return 0
        best = -1
        for j in eligible[i]:
            if used & (1 << j):
                continue
            rest = _best(i + 1, used | (1 << j))
            if rest < 0:
                continue
            val = rest + (1 if actual_asg[i] == players[j] else 0)
            if val > best:
                best = val
        return best

    if _best(0, 0) < 0:
        return list(solver_asg)
    out, used = [], 0
    for i in range(len(slots)):
        for j in eligible[i]:
            if used & (1 << j):
                continue
            rest = _best(i + 1, used | (1 << j))
            if rest < 0:
                continue
            if rest + (1 if actual_asg[i] == players[j] else 0) == _best(i, used):
                out.append(players[j])
                used |= 1 << j
                break
    return out if len(out) == len(slots) else list(solver_asg)


def _build_lineup_this_week(ctx):
    """Current vs projected-optimal for the current matchup week.

    Returns (html, swap_count); ("", None) when the view cannot be built
    honestly (no matchup, no projections, no complete optimal)."""
    from dashboard_services.matchups import (
        build_slide_proj_lookups,
        compute_win_prob,
        resolve_team_game,
    )
    from utils.matchup_schedule import resolve_display_week

    viewer_rid = str((ctx.get("viewer") or {}).get("viewer_roster_id") or "")
    if not viewer_rid:
        return "", None
    season = int(ctx.get("season") or datetime.now().year)
    current_week = int(ctx.get("current_week") or 1)
    matchups_by_week = ctx.get("matchups_by_week") or {}
    # The same week the hub's carousel slides show for this ctx (shared
    # helper, clamped, with the completed/offseason branch), so the label
    # and the priced matchup can never point at different weeks.
    week = resolve_display_week(
        current_week, matchups_by_week, int(ctx.get("weeks") or 0),
        season_complete=bool(ctx.get("season_complete")),
        offseason_mode=bool(ctx.get("offseason_mode")),
        last_final_week=_lw_last_final_week(ctx, 0))
    rows = matchups_by_week.get(week) or matchups_by_week.get(str(week)) or []

    viewer_block = opponent_block = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        for a_key, b_key in (("left", "right"), ("team1", "team2")):
            a, b = row.get(a_key), row.get(b_key)
            if not isinstance(a, dict) or not isinstance(b, dict):
                continue
            if str(a.get("roster_id")) == viewer_rid:
                viewer_block, opponent_block = a, b
            elif str(b.get("roster_id")) == viewer_rid:
                viewer_block, opponent_block = b, a
        if viewer_block is not None:
            break
    if viewer_block is None or opponent_block is None:
        return "", None
    opponent_rid = str(opponent_block.get("roster_id") or "")

    rosters = ctx.get("rosters") or []
    viewer_roster = next(
        (r for r in rosters if str(r.get("roster_id")) == viewer_rid), None)
    # Taxi-squad players cannot be started without a roster move, so they
    # are never suggested here (the same roster["taxi"] exclusion the digest
    # and push start decisions apply). Starters are never on taxi; the
    # starter-append guard below stays as a harmless safety net.
    taxi_pids = {str(p) for p in (viewer_roster or {}).get("taxi") or []
                 if str(p) not in ("", "0")}
    roster_pids = [str(p) for p in (viewer_roster or {}).get("players") or []
                   if str(p) not in ("", "0") and str(p) not in taxi_pids]
    starter_pids = _lw_block_pids(viewer_block)
    if not roster_pids or not starter_pids:
        return "", None
    for pid in starter_pids:
        if pid not in roster_pids:
            roster_pids.append(pid)

    info = _lw_player_info(ctx, roster_pids)
    positions = {pid: (info.get(pid) or {}).get("pos", "") for pid in roster_pids}

    # The slide's own projection lookups, so the numbers match the carousel.
    raw_scoring = (ctx.get("raw_scoring_settings") or ctx.get("scoring_settings")
                   or ctx.get("scoring") or {})
    week_proj_map, pid_proj, frac_lookup = build_slide_proj_lookups(
        season, week, _lw_last_final_week(ctx, week),
        ctx.get("proj_by_week") or {}, ctx.get("team_game_lookup") or {},
        scoring_settings=raw_scoring)
    proj = {}
    for pid in roster_pids:
        try:
            v = pid_proj(pid, positions.get(pid, ""))
        except Exception:
            v = None
        if v is not None:
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = 0.0
            if v > 0:
                proj[pid] = v
    if not proj or not any(proj.get(p, 0) > 0 for p in starter_pids):
        return "", None

    slots_spec = ctx.get("roster_positions") or []
    data = analyze_lineup(proj, positions, slots_spec, roster_pids, starter_pids)
    if not data.get("complete"):
        keep = [p for p in roster_pids if p in proj and positions.get(p)]
        data = analyze_lineup(
            proj, positions, slots_spec, keep,
            [p for p in starter_pids if p in set(keep)])
        roster_pids = keep
    if not data.get("complete"):
        return "", None

    actual, optimal = float(data["actual"]), float(data["optimal"])
    missed = max(0.0, optimal - actual)
    slots = data["slots"]
    actual_asg = data["actual_assignment"]
    # Display the optimal SET seated stably (continuing starters keep their
    # slots), never the solver's raw rotation.
    optimal_asg = _lw_stable_optimal_assignment(
        slots, actual_asg, data["optimal_assignment"], positions)
    starter_set = {p for p in actual_asg if p}
    optimal_set = {p for p in optimal_asg if p}
    changed = {i for i, (a, o) in enumerate(zip(actual_asg, optimal_asg)) if a != o}

    # Win probability, current vs optimal starters, via the carousel function.
    statuses = ctx.get("statuses") or {}
    status_by_pid = ((statuses.get(week) or {}).get("statuses")
                     or (statuses.get(current_week) or {}).get("statuses") or {})
    wp_cur = wp_opt = None
    try:
        wp_cur = compute_win_prob(viewer_block, opponent_block, status_by_pid,
                                  week_proj_map, frac_lookup=frac_lookup)
        orig_by_pid = {}
        for s in viewer_block.get("starters") or []:
            if isinstance(s, dict) and s.get("pid") is not None:
                orig_by_pid[str(s["pid"])] = s
        opt_entries = []
        for pid in optimal_asg:
            if not pid:
                continue
            if pid in orig_by_pid:
                opt_entries.append(orig_by_pid[pid])
            else:
                pinfo = info.get(pid) or {}
                opt_entries.append({"pid": pid, "pos": pinfo.get("pos", ""),
                                    "nfl": pinfo.get("team", "")})
        opt_block = dict(viewer_block)
        opt_block["starters"] = opt_entries
        wp_opt = compute_win_prob(opt_block, opponent_block, status_by_pid,
                                  week_proj_map, frac_lookup=frac_lookup)
    except Exception:
        wp_cur = wp_opt = None

    # Swap reasons: projection delta, plus the site's points-allowed table
    # when the incoming player draws one of the softest matchups at his spot.
    fpts_against = {}
    try:
        from app import _compute_fpts_against
        fpts_against = _compute_fpts_against(season, raw_scoring) or {}
    except Exception:
        fpts_against = {}
    team_game_lookup = ctx.get("team_game_lookup") or {}

    def _matchup_note(pid):
        if not fpts_against:
            return ""
        pinfo = info.get(pid) or {}
        pos, team = pinfo.get("pos", ""), pinfo.get("team", "")
        if not pos or not team:
            return ""
        try:
            from utils.nfl import canon_team
            game = resolve_team_game(canon_team(team), team_game_lookup)
        except Exception:
            game = None
        if not game:
            return ""
        opp = game.get("away") if game.get("home") == team else game.get("home")
        if not opp:
            return ""
        ranked = sorted(
            ((t, (vals or {}).get(pos)) for t, vals in fpts_against.items()),
            key=lambda x: (x[1] is not None, x[1] or 0.0), reverse=True)
        ranked = [t for t, v in ranked if v is not None]
        if opp not in ranked:
            return ""
        rank = ranked.index(opp) + 1
        if rank > 10:
            return ""
        return (f"{opp} allows the {_lw_ordinal(rank)}-most points to "
                f"{_POS_PLURAL.get(pos, pos + 's')}")

    # Paired moves from the starter-set difference: who actually enters the
    # lineup and who actually leaves it. A bench player seated (in the
    # stabilized display) in a departing starter's slot pairs with him
    # directly; leftovers, where the entry forced a rotation, pair off by
    # projection. The deltas telescope: they always sum to Optimal minus
    # Current, so the banner, the +Available metric, the tab chip and the
    # table cannot disagree.
    incoming = sorted(optimal_set - starter_set, key=lambda p: -proj.get(p, 0.0))
    outgoing = sorted(starter_set - optimal_set, key=lambda p: -proj.get(p, 0.0))
    pair_out = {}
    used_out = set()
    for inn in incoming:
        seat = optimal_asg.index(inn)
        a = actual_asg[seat]
        if a not in optimal_set and a not in used_out:
            pair_out[inn] = a
            used_out.add(a)
    rest_in = [p for p in incoming if p not in pair_out]
    rest_out = [p for p in outgoing if p not in used_out]
    for inn, out in zip(rest_in, rest_out):
        pair_out[inn] = out

    swaps = []
    for inn in incoming:
        out = pair_out.get(inn)
        delta = proj.get(inn, 0.0) - (proj.get(out, 0.0) if out else 0.0)
        in_name = (info.get(inn) or {}).get("name") or inn
        out_name = ((info.get(out) or {}).get("name") or out) if out else "Empty slot"
        if delta >= 0:
            reason = f"{in_name} projects {delta:.1f} more than {out_name}."
        else:
            reason = (f"{in_name} projects {abs(delta):.1f} fewer than {out_name}, "
                      f"but the full set of moves gains {missed:.1f} overall.")
        note = _matchup_note(inn)
        if note:
            reason += f" {note}."
        elif 0 <= delta < 1.5:
            reason += " Close call."
        swaps.append({"slot": slots[optimal_asg.index(inn)], "out": out,
                      "in": inn, "delta": delta, "reason": reason})
    swaps.sort(key=lambda s: -s["delta"])

    roster_map = ctx.get("roster_map") or {}
    opp_name = roster_map.get(opponent_rid) or f"Roster {opponent_rid}"
    sub = (f"<div class='lw-sub'>Week {week} vs {_esc(opp_name)} · "
           f"projections, not final results</div>")

    wp_metric = ""
    if wp_cur is not None:
        wp_txt = f"{wp_cur * 100:.0f}%"
        if wp_opt is not None and swaps:
            wp_txt += f" → {wp_opt * 100:.0f}%"
        wp_metric = (f"<div class='lw-metric'><b>{wp_txt}</b>"
                     f"<span>Win prob</span></div>")
    if swaps:
        metrics = (
            "<div class='lw-metrics'>"
            f"<div class='lw-metric'><b>{actual:.1f}</b><span>Current proj</span></div>"
            f"<div class='lw-metric'><b>{optimal:.1f}</b><span>Optimal proj</span></div>"
            f"<div class='lw-metric acc'><b>+{missed:.1f}</b><span>Available</span></div>"
            f"{wp_metric}</div>"
        )
    else:
        # Already optimal: the Optimal and Available tiles would only
        # repeat the Current number, so the band collapses to the two
        # tiles that say something (current projection, win probability).
        metrics = (
            "<div class='lw-metrics lw-metrics-solo'>"
            f"<div class='lw-metric'><b>{actual:.1f}</b><span>Current proj</span></div>"
            f"{wp_metric}</div>"
        )

    if swaps:
        swaps_html = "".join(
            f"<div class='lw-swap'><span class='lw-d'>+{s['delta']:.1f}</span>"
            f"<span class='lw-out'>{_esc((info.get(s['out']) or {}).get('name') or s['out'] or 'Empty slot')}</span>"
            f"<span class='lw-arrow'>→</span>"
            f"<span class='lw-in'>{_esc((info.get(s['in']) or {}).get('name') or s['in'])}</span>"
            f"<span class='lw-why'>{_esc(s['reason'])}</span></div>"
            for s in swaps)
    else:
        swaps_html = ("<div class='lw-noswap'>No swaps needed. Your current "
                      "starters are the projected-optimal lineup.</div>")

    def _cell(pid):
        if not pid:
            return "Empty slot"
        nm = (info.get(pid) or {}).get("name") or pid
        return f"{nm} · {proj.get(pid, 0.0):.1f}"

    if swaps:
        rows_html = ""
        for i, slot in enumerate(slots):
            a, o = actual_asg[i], optimal_asg[i]
            is_swap = i in changed and a != o
            if is_swap:
                d = proj.get(o, 0.0) - (proj.get(a, 0.0) if a else 0.0)
                pill = f"<span class='lw-spill'>Swap +{d:.1f}</span>"
            else:
                pill = "<span class='lw-keep'>Keep</span>"
            row_cls = " class='lw-sw'" if is_swap else ""
            rows_html += (
                f"<tr{row_cls}>"
                f"<td>{_esc(str(slot).replace('_', '/'))}</td>"
                f"<td>{_esc(_cell(a))}</td><td>{_esc(_cell(o))}</td>"
                f"<td>{pill}</td></tr>")
        body = (f"<div class='lw-wrap'><table class='lw-table'><thead><tr>"
                f"<th>Slot</th><th>Current</th><th>Optimal</th><th></th></tr></thead>"
                f"<tbody>{rows_html}</tbody></table></div>")
    else:
        # Already optimal: a Current-vs-Optimal table would print the same
        # lineup twice, so show one compact starters list with a Total
        # footer instead. On desktop the hub CSS stretches the list to
        # fill the panel (rows share the height evenly).
        list_rows = ""
        for i, slot in enumerate(slots):
            a = actual_asg[i]
            nm = ((info.get(a) or {}).get("name") or a) if a else "Empty slot"
            pts = f"{proj.get(a, 0.0):.1f}" if a else ""
            list_rows += (
                f"<div class='lw-strow'>"
                f"<span class='lw-stslot'>{_esc(str(slot).replace('_', '/'))}</span>"
                f"<span class='lw-stname'>{_esc(nm)}</span>"
                f"<span class='lw-stpts'>{pts}</span></div>")
        list_rows += (
            f"<div class='lw-strow lw-strow-total'>"
            f"<span class='lw-stslot'>Total</span><span class='lw-stname'></span>"
            f"<span class='lw-stpts'>{actual:.1f}</span></div>")
        body = f"<div class='lw-wrap'><div class='lw-stlist'>{list_rows}</div></div>"

    html_out = f"<div class='lw-thisweek'>{sub}{metrics}{swaps_html}{body}</div>"
    return html_out, len(swaps)


def build_optimal_body(ctx):
    """Lineup tab body: This Week / History segmented switch (v12 layout).

    This Week defaults on, except when the request carries explicit history
    params (view/period), which is how Lineup Lab links and fragment loads
    arrive. When This Week cannot be built, History renders alone."""
    try:
        tw_html, swap_count = _build_lineup_this_week(ctx)
    except Exception:
        logger.exception("lineup this-week build failed")
        tw_html, swap_count = "", None
    history_html = _build_optimal_history(ctx)
    if not tw_html:
        return history_html
    ctx["lineup_swap_count"] = swap_count
    try:
        args = request.args
        history_default = bool(args.get("view") or args.get("period"))
    except Exception:
        history_default = False
    tw_active = not history_default
    seg = (
        "<div class='opt-seg'>"
        f"<button type='button' class='opt-seg-btn{' active' if tw_active else ''}' "
        f"data-opt-seg='thisweek'>This Week</button>"
        f"<button type='button' class='opt-seg-btn{' active' if not tw_active else ''}' "
        f"data-opt-seg='history'>History</button>"
        "</div>"
    )
    return (
        seg
        + f"<div class='opt-seg-panel' data-opt-panel='thisweek'"
          f"{'' if tw_active else ' hidden'}>{tw_html}</div>"
        + f"<div class='opt-seg-panel' data-opt-panel='history'"
          f"{'' if not tw_active else ' hidden'}>{history_html}</div>"
    )
