"""Historical lineup analysis for the Matchups page Lineup tab."""
from __future__ import annotations

import html
import logging
from datetime import datetime
from flask import request

from utils.optimal_lineup import analyze_lineup

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


def build_optimal_body(ctx):
    from app import get_players_index_global
    from dashboard_services.platform_api import get_matchups

    platform = ctx.get("platform") or "sleeper"
    season = int(ctx.get("season") or datetime.now().year)
    league_id = ctx.get("league_id") or ""
    viewer_rid = str((ctx.get("viewer") or {}).get("viewer_roster_id") or "")
    slots = ctx.get("roster_positions") or []
    players = get_players_index_global() or {}
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
        wk = f'&week={selected}' if qperiod == "weekly" else ""
        return f'<a class="opt-tab{" active" if active else ""}" href="{base}&view={qview}&period={qperiod}{wk}">{label}</a>'
    options = "".join(f'<option value="{w}"{" selected" if w == selected else ""}>Week {w}</option>' for w in completed)
    nav = (f'<nav class="opt-nav" aria-label="Lineup analysis controls"><div class="opt-tab-group">'
           f'{tab("My Team", "view", "user", view == "user")}{tab("League", "view", "league", view == "league")}</div>'
           f'<div class="opt-tab-group">{tab("Weekly", "period", "weekly", period == "weekly")}'
           f'{tab("Season", "period", "season", period == "season")}</div>'
           + (f'<label class="sr-only" for="optWeek">Completed week</label><select id="optWeek" class="opt-week-select" '
              f'data-base-url="{base}&view={view}&period=weekly">{options}</select>' if period == "weekly" else "")
           + '</nav><p class="opt-method">Optimal lineup uses final results and your league’s roster rules.</p>')

    weeks_to_fetch = [selected] if period == "weekly" else completed
    matchup_cache = ctx.get("optimal_matchups_by_week") or {}
    for week in weeks_to_fetch:
        if week not in matchup_cache:
            try:
                matchup_cache[week] = get_matchups(platform, league_id, week, season) or []
            except (LookupError, ValueError, RuntimeError, OSError) as exc:
                logger.warning("lineup matchup unavailable platform=%s league=%s week=%s: %s", platform, league_id, week, exc)
                matchup_cache[week] = None

    def analyze(rid, week):
        rows = matchup_cache.get(week)
        if rows is None:
            return {"week": week, "complete": False, "reason": "matchup unavailable"}
        row = next((m for m in rows if str(m.get("roster_id")) == str(rid)), None)
        if not row:
            return {"week": week, "complete": False, "reason": "historical roster unavailable"}
        pids = [str(p) for p in (row.get("players") or []) if p is not None and str(p) != "0"]
        starters = [str(p) if p is not None else "0" for p in (row.get("starters") or [])]
        raw_scores = {str(k): v for k, v in (row.get("players_points") or {}).items()}
        positions = {p: (players.get(p) or {}).get("pos") for p in pids}
        out = analyze_lineup(raw_scores, positions, slots, pids, starters, row.get("points"))
        out.update({"week": week, "pids": pids,
                    "scores": {p: (None if v is None else float(v)) for p, v in raw_scores.items()}})
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
        eff = "—" if team["eff"] is None else f'{team["eff"]:.1f}%'
        cards += (f'<details class="card opt-team{" is-viewer" if team["rid"] == viewer_rid else ""}"><summary>'
                  f'<span class="opt-rank">#{rank}</span><strong>{_esc(team["name"])}</strong><span>{eff}</span>'
                  f'<span>{team["actual"]:.1f} actual</span><span>{team["optimal"]:.1f} optimal</span>'
                  f'<span>{team["missed"]:.1f} missed</span><span>{len(team["good"])}/{len(team["data"])} weeks</span></summary>'
                  + ''.join(weekly_panel(d, f'Week {d["week"]}') for d in reversed(team["data"])) + '</details>')
    return nav + '<div class="opt-leaderboard-head">Efficiency leaderboard · sorted highest first</div>' + cards
