"""Weekly recap HTML builder.

Moved from app.py so the Flask monolith can keep shrinking. Helpers that still
live in app.py are lazy-imported inside the builder (request time).
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _rank_movement(current: list[dict], prior: list[dict]) -> dict[str, int]:
    """Return places gained by roster, using only two already-capped snapshots."""
    old = {str(row["rid"]): rank for rank, row in enumerate(prior, 1)}
    return {
        str(row["rid"]): old[str(row["rid"])] - rank
        for rank, row in enumerate(current, 1)
        if str(row["rid"]) in old
    }


def _matchup_badges(matchups: list[dict]) -> dict[int, list[str]]:
    """Only the non-duplicative scoreboard distinction belongs on a row."""
    if not matchups:
        return {}
    highest = max(range(len(matchups)), key=lambda i: matchups[i]["w_pts"] + matchups[i]["l_pts"])
    return {highest: ["Highest-Scoring Matchup"]}


def _top_performers_by_roster(matchups: list[dict]) -> dict[str, list[dict]]:
    """Return each roster's highest-scoring actual weekly starter(s).

    The normalized Matchups data supplies provider-authoritative, league-scored
    totals. Missing values are not coerced to zero, while real zero and negative
    totals remain eligible. The historical marker blocks current-roster fallback
    data from rewriting an old lineup.
    """
    out: dict[str, list[dict]] = {}
    for matchup in matchups or []:
        for team in (matchup.get("left") or {}, matchup.get("right") or {}):
            rid = str(team.get("roster_id") or "")
            if not rid or team.get("lineup_is_historical") is not True:
                continue
            scored = []
            for player in team.get("starters") or []:
                pts = player.get("pts")
                if isinstance(pts, (int, float)) and not isinstance(pts, bool):
                    scored.append({
                        "pid": str(player.get("pid") or ""),
                        "name": str(player.get("name") or "Unknown player"),
                        "pts": float(pts),
                    })
            if not scored:
                continue
            high = max(player["pts"] for player in scored)
            out[rid] = sorted(
                (player for player in scored if player["pts"] == high),
                key=lambda player: (player["name"].casefold(), player["pid"]),
            )
    return out


def _weekly_efficiency_rows(efficiency_data: dict, selected_week: int) -> list[dict]:
    """Return complete weekly efficiency rows in deterministic rank order.

    The inputs come exclusively from ``season_efficiency``, which delegates to
    the shared legal-lineup optimizer. Missing/incomplete weeks are omitted
    instead of being presented as zero-efficiency performances.
    """
    rows = []
    for rid, season_row in (efficiency_data.get("by_rid") or {}).items():
        week_row = next(
            (row for row in (season_row.get("weeks") or [])
             if row.get("week") == selected_week),
            None,
        )
        if not week_row or week_row.get("eff") is None or week_row.get("optimal") is None:
            continue
        actual = float(week_row["actual"])
        optimal = float(week_row["optimal"])
        if optimal <= 0:
            continue
        # Clamp presentation inputs at their mathematical bounds so provider
        # rounding cannot surface -0.0 points left or >100% efficiency.
        points_left = max(0.0, optimal - actual)
        efficiency = min(100.0, max(0.0, actual / optimal * 100.0))
        rows.append({"rid": str(rid), "actual": actual, "optimal": optimal,
                     "eff": efficiency, "missed": points_left})
    return sorted(rows, key=lambda row: (-row["eff"], -row["actual"], row["rid"]))


def build_recap_body(ctx: dict, selected_week: Optional[int] = None) -> str:
    from app import (  # noqa: E402  (lazy: avoids a circular import at module load)
        _build_lineup_analysis_html,
        _build_next_week_ctx,
        _build_recap_preview_df,
        build_standings_as_of_week,
        _mock_lineup_analysis_html,
        has_premium_for_viewer,
        html,
        json,
        rank_mark,
        session,
        team_crest,
    )

    df_weekly = ctx.get("df_weekly")
    roster_map = ctx.get("roster_map") or {}
    users = ctx.get("users") or []
    league = ctx.get("league") or {}
    settings = league.get("settings") or {}
    playoff_start = int(settings.get("playoff_week_start") or 14)
    _platform = ctx.get("platform") or "sleeper"
    _season = ctx.get("season") or ""
    _league_id = ctx.get("league_id") or ""
    history_url = f"/{_platform}/{_season}/{_league_id}/history" if _league_id else ""

    if df_weekly is None:
        history_link = f'<a class="recap-history-link" href="{history_url}">History</a>' if history_url else ""
        return (f'<main class="weekly-recap"><div class="recap-page-header"><h2>Weekly Recap</h2>'
                f'{history_link}</div><div class="card recap-data-unavailable" role="status">'
                'Weekly results could not be loaded. Try again shortly; sample results are not shown '
                'when league data is unavailable.</div></main>')

    # ── Preview mode: no finalized weeks yet → use mock data ───────────────
    preview_mode = False
    has_real_finalized = (
            df_weekly is not None
            and not df_weekly.empty
            and "finalized" in df_weekly.columns
            and bool((df_weekly["finalized"] == True).any())
    )
    if not has_real_finalized:
        preview_mode = True
        # Build mock data from real team names if available, else defaults
        if roster_map:
            team_names = [str(n) for n in roster_map.values()][:10]
            # Map mock roster_ids to real ones so avatars resolve
            real_rids = list(roster_map.keys())
            df_weekly = _build_recap_preview_df(team_names)
            # Overwrite roster_id with real ones to enable avatar lookup
            for i, rid in enumerate(df_weekly["roster_id"].tolist()):
                if i < len(real_rids):
                    df_weekly.at[i, "roster_id"] = str(real_rids[i % len(real_rids)])
        else:
            team_names = ["Dynasty Kings", "Gridiron Ghosts", "Blitz Brigade",
                          "Redzone Rebels", "Endzone Elite", "Pocket Protectors"]
            df_weekly = _build_recap_preview_df(team_names)
            roster_map = {str(i + 1): n for i, n in enumerate(team_names)}

    # Resolve pictures through the same roster-identity resolver used everywhere
    # else. It handles provider team art, owner art, then a generated crest.
    from dashboard_services.api import team_avatar
    roster_by_rid = {str(r.get("roster_id")): r for r in (ctx.get("rosters") or [])}
    avatar_by_rid = {
        rid: team_avatar(_platform, roster, users) or ""
        for rid, roster in roster_by_rid.items()
    }
    # avatar by owner name
    owner_avatar: dict = {}
    for u in users:
        name = u.get("display_name") or u.get("username") or ""
        ava = u.get("metadata", {}).get("avatar") or u.get("avatar") or ""
        if name and ava:
            if not ava.startswith("http"):
                ava = f"https://sleepercdn.com/avatars/thumbs/{ava}"
            owner_avatar[name] = ava

    # team name by roster_id
    team_by_rid: dict = {str(rid): name for rid, name in roster_map.items()}

    fin_df = df_weekly[df_weekly["finalized"] == True].copy()

    available_weeks = sorted(fin_df["week"].unique().tolist())
    reg_weeks = [w for w in available_weeks if w < playoff_start]

    if selected_week is None or selected_week not in available_weeks:
        selected_week = available_weeks[-1]

    week_df = fin_df[fin_df["week"] == selected_week].copy()
    scored_week_df = week_df[week_df["points"].notna()].copy()

    normalized_weeks = ctx.get("matchups_by_week") or {}
    normalized_matchups = (
        normalized_weeks.get(selected_week)
        or normalized_weeks.get(str(selected_week))
        or []
    )
    top_performers = {} if preview_mode else _top_performers_by_roster(normalized_matchups)

    # ── Matchup pairs ──────────────────────────────────────────────────────
    matchups: list[dict] = []
    for _, grp in scored_week_df.groupby("matchup_id"):
        if len(grp) != 2:
            continue
        grp = grp.sort_values("points", ascending=False)
        w_row, l_row = grp.iloc[0], grp.iloc[1]
        margin = float(w_row["points"]) - float(l_row["points"])
        matchups.append({
            "winner": w_row["owner"],
            "loser": l_row["owner"],
            "w_rid": str(w_row.get("roster_id", "")),
            "l_rid": str(l_row.get("roster_id", "")),
            "w_pts": float(w_row["points"]),
            "l_pts": float(l_row["points"]),
            "margin": margin,
            "tied": margin == 0,
        })
    matchups.sort(key=lambda x: -x["margin"])

    # ── Highlights ─────────────────────────────────────────────────────────
    if scored_week_df.empty:
        return (f'<main class="weekly-recap"><h2>Week {selected_week} Recap</h2>'
                '<div class="card recap-data-unavailable" role="status">Final scores are unavailable for this week.</div></main>')
    high_row = scored_week_df.loc[scored_week_df["points"].idxmax()]
    low_row = scored_week_df.loc[scored_week_df["points"].idxmin()]
    decisive_matchups = [m for m in matchups if not m.get("tied")]
    blowout = decisive_matchups[0] if decisive_matchups else None
    closest = matchups[-1] if matchups else None

    league_avg = float(scored_week_df["points"].mean())
    league_total = float(scored_week_df["points"].sum())

    # Season high/low context
    from dashboard_services.recap_calculations import season_high_through
    season_high = season_high_through(
        fin_df[["week", "points"]].to_dict("records"), selected_week, float(high_row["points"]),
    )

    def ava_img(owner_name, rid="", size=32):
        ava = avatar_by_rid.get(str(rid)) or owner_avatar.get(owner_name, "")
        # Data-URI crests from team_avatar() are fragile as <img> src (CSP,
        # encoding); render the reliable inline SVG crest instead.
        if ava.startswith("data:"):
            ava = ""
        if ava:
            # visibility:hidden (not display:none) preserves the grid slot so a
            # broken avatar can't shift the team name into the 34px column.
            return f"<img src='{ava}' alt='' loading='lazy' decoding='async' style='width:{size}px;height:{size}px;border-radius:50%;object-fit:cover;flex-shrink:0;' onerror=\"this.style.visibility='hidden'\">"
        return team_crest(team_by_rid.get(rid) or owner_name or "?", size)

    def team_name(owner, rid=""):
        return html.escape(team_by_rid.get(rid) or owner or "–")

    def team_link(owner, rid="", inner=None, extra_class=""):
        """Wrap content so a click opens the fantasy team modal (delegated
        .team-clickable handler in app.js). Falls back to plain content when no
        roster id is known."""
        rid_s = str(rid or "")
        body = inner if inner is not None else team_name(owner, rid)
        if not rid_s:
            return body
        tn = html.escape(team_by_rid.get(rid) or owner or "", quote=True)
        cls = ("team-clickable " + extra_class).strip()
        return (f'<span class="{cls}" role="button" tabindex="0" '
                f'data-roster-id="{html.escape(rid_s, quote=True)}" '
                f'data-team-name="{tn}" style="cursor:pointer;">{body}</span>')

    # Fetch the cached, shared legal-lineup analysis once. The same selected
    # week dataset drives this page section, its award cards, and the share card.
    efficiency_rows = []
    efficiency_data = {"state": "preview"}
    if not preview_mode:
        try:
            from dashboard_services.season_efficiency import compute_league_season_efficiency
            efficiency_data = compute_league_season_efficiency(
                {**ctx, "efficiency_weeks": [int(selected_week)]})
            efficiency_rows = _weekly_efficiency_rows(efficiency_data, int(selected_week))
        except Exception as exc:
            logger.warning("weekly recap efficiency failed league=%s season=%s week=%s: %s",
                           ctx.get("league_id"), ctx.get("season"), selected_week, exc)
            efficiency_data = {"state": "loading_failure"}
            efficiency_rows = []
    best_efficiency = efficiency_rows[0] if efficiency_rows else None
    most_left = (sorted(efficiency_rows,
                        key=lambda row: (-row["missed"], -row["actual"], row["rid"]))[0]
                 if efficiency_rows else None)

    # ── Week selector ──────────────────────────────────────────────────────
    week_opts = "".join(
        f"<option value='{w}' {'selected' if w == selected_week else ''}>"
        f"{'Playoffs · ' if w >= playoff_start else ''}Week {w}</option>"
        for w in reversed(available_weeks)
    )
    history_banner = ""

    # Data for the client-drawn shareable recap card (static/app.js paints it
    # onto a canvas and hands it to the native share sheet).
    _card_matchups = sorted(matchups, key=lambda x: -x["w_pts"])[:6]
    _recap_share = {
        "league": str(league.get("name") or "League"),
        "week": int(selected_week),
        "season": str(_season),
        "games": [{"w": m["winner"], "l": m["loser"],
                   "ws": round(m["w_pts"], 1), "ls": round(m["l_pts"], 1)}
                  for m in _card_matchups],
        "top": {"team": str(high_row["owner"]), "pts": round(float(high_row["points"]), 1)},
        "blowout": ({"team": blowout["winner"], "margin": round(blowout["margin"], 1)}
                    if blowout else None),
        "closest": ({"team": closest["winner"], "margin": round(closest["margin"], 1)}
                    if closest else None),
        "best_lineup": ({"team": team_by_rid.get(best_efficiency["rid"], "Unknown team"),
                         "efficiency": round(best_efficiency["eff"], 1),
                         "actual": round(best_efficiency["actual"], 1),
                         "optimal": round(best_efficiency["optimal"], 1)}
                        if best_efficiency else None),
        "most_left": ({"team": team_by_rid.get(most_left["rid"], "Unknown team"),
                       "efficiency": round(most_left["eff"], 1),
                       "points_left": round(most_left["missed"], 1)}
                      if most_left else None),
    }
    _recap_share_json = json.dumps(_recap_share).replace("</", "<\\/")

    week_selector = f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;flex-wrap:wrap;">
  <div style="flex:1;min-width:160px;">
    <h2 style="margin:0;font-size:20px;">Week {selected_week} Recap</h2>
  </div>
  <select onchange="window.location.search='?week='+this.value"
          style="padding:5px 10px;border-radius:6px;border:1px solid var(--border);
                 background:var(--card);color:var(--text);font-size:13px;cursor:pointer;">
    {week_opts}
  </select>
  {f'<a class="recap-history-link" href="{history_url}">History</a>' if history_url else ''}
  <button type="button" id="recapShareBtn"
          style="display:flex;align-items:center;gap:5px;padding:5px 12px;border-radius:6px;
                 border:1px solid var(--border);background:var(--card);color:var(--text);
                 font-size:13px;cursor:pointer;font-weight:600;">
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg> Share
  </button>
  <script type="application/json" id="recapShareData">{_recap_share_json}</script>
</div>"""

    # ── Headline cards ─────────────────────────────────────────────────────
    def _scorer_opp(row):
        """The other team in this scorer's matchup, so the scorer cards can show
        who they beat/lost to (matching the two-row matchup cards) instead of
        leaving the card half-empty."""
        try:
            grp = week_df[week_df["matchup_id"] == row.get("matchup_id")]
            others = grp[grp["owner"] != row["owner"]]
            if others.empty:
                return None
            o = others.iloc[0]
            return {"owner": o["owner"], "rid": str(o.get("roster_id", "")),
                    "pts": float(o["points"])}
        except Exception:
            return None

    def scorer_card(icon, label, name, pts, rid, sub, accent, opp=None, medal_rank=None):
        # The week's HIGH SCORER earns a gold medal (a weekly award); the other
        # cards keep their semantic icon. Icon/medal sits in an accent chip.
        chip_inner = (rank_mark(medal_rank, size=15, wrap=False)
                      if medal_rank else f'<i class="{icon}" aria-hidden="true"></i>')
        header = f"""
  <div class="rc-award-h">
    <span class="rc-award-chip">{chip_inner}</span>
    <span class="rc-award-lbl">{label}</span>
  </div>"""
        if opp:
            diff = abs(pts - opp["pts"])
            result = "Tied" if pts == opp["pts"] else f"{'Won' if pts > opp['pts'] else 'Lost'} by {diff:.2f}"
            body = f"""
  <div style="display:flex;flex-direction:column;gap:8px;">
    <div style="display:flex;align-items:center;gap:10px;">
      {ava_img(name, rid, 34)}
      <div style="flex:1;min-width:0;">
        <div style="font-size:14px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{team_link(name, rid)}</div>
      </div>
      <div style="font-size:23px;font-weight:800;color:{accent};flex-shrink:0;letter-spacing:-.5px;font-variant-numeric:tabular-nums;">{pts:.2f}</div>
    </div>
    <div style="height:1px;background:var(--border);"></div>
    <div style="display:flex;align-items:center;gap:10px;opacity:0.5;">
      {ava_img(opp["owner"], opp["rid"], 34)}
      <div style="flex:1;min-width:0;">
        <div style="font-size:14px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{team_name(opp["owner"], opp["rid"])}</div>
      </div>
      <div style="font-size:23px;font-weight:800;flex-shrink:0;letter-spacing:-.5px;font-variant-numeric:tabular-nums;">{opp["pts"]:.2f}</div>
    </div>
  </div>
  <div class="rc-award-foot"><span style="color:{accent};">{html.escape(sub)}</span> &middot; {result}</div>"""
        else:
            body = f"""
  <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
    <div style="display:flex;align-items:center;gap:10px;min-width:0;">
      {ava_img(name, rid, 42)}
      <div style="min-width:0;">
        <div style="font-weight:700;font-size:15px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{team_link(name, rid)}</div>
        <div style="font-size:12px;color:var(--muted);">@{html.escape(name)}</div>
      </div>
    </div>
    <div style="text-align:right;flex-shrink:0;">
      <div style="font-size:30px;font-weight:800;color:{accent};letter-spacing:-.5px;line-height:1;font-variant-numeric:tabular-nums;">{pts:.2f}</div>
      <div style="font-size:11px;color:{accent};font-weight:600;margin-top:4px;">{html.escape(sub)}</div>
    </div>
  </div>"""
        return f"""
<div class="card rc-award" style="--rc-accent:{accent};">{header}{body}
</div>"""

    def matchup_card(icon, label, m, accent="var(--accent)"):
        w_team = team_name(m["winner"], m["w_rid"])
        l_team = team_name(m["loser"], m["l_rid"])
        return f"""
<div class="card rc-award" style="--rc-accent:{accent};">
  <div class="rc-award-h">
    <span class="rc-award-chip"><i class="{icon}" aria-hidden="true"></i></span>
    <span class="rc-award-lbl">{label}</span>
  </div>
  <div style="display:flex;flex-direction:column;gap:8px;">
    <div style="display:flex;align-items:center;gap:10px;">
      {ava_img(m["winner"], m["w_rid"], 34)}
      <div style="flex:1;min-width:0;">
        <div style="font-size:14px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{w_team}</div>
      </div>
      <div style="font-size:23px;font-weight:800;color:{accent};flex-shrink:0;letter-spacing:-.5px;">{m['w_pts']:.1f}</div>
    </div>
    <div style="height:1px;background:var(--border);"></div>
    <div style="display:flex;align-items:center;gap:10px;opacity:0.5;">
      {ava_img(m["loser"], m["l_rid"], 34)}
      <div style="flex:1;min-width:0;">
        <div style="font-size:14px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{l_team}</div>
      </div>
      <div style="font-size:23px;font-weight:800;flex-shrink:0;letter-spacing:-.5px;">{m['l_pts']:.1f}</div>
    </div>
  </div>
  <div class="rc-award-foot">margin {m['margin']:.1f} &middot; <a class="recap-view-matchup" href="/{_platform}/{_season}/{_league_id}/weekly?week={selected_week}">View matchup <span aria-hidden="true">&rsaquo;</span></a></div>
</div>"""

    high_sub = "Season high" if season_high else f"+{float(high_row['points']) - league_avg:.1f} vs avg"
    low_sub = f"{float(low_row['points']) - league_avg:.1f} vs avg"

    cards_html = f"""
<style>
  .rc-awards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-bottom:20px; }}
  @media (max-width:900px) {{ .rc-awards {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
  @media (max-width:350px) {{ .rc-awards {{ grid-template-columns:1fr; }} }}
  .rc-award {{ position:relative; overflow:hidden; padding:11px 12px 10px; display:flex;
               flex-direction:column; gap:8px; min-width:0; }}
  .rc-award::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:var(--rc-accent); }}
  .rc-award-h {{ display:flex; align-items:center; gap:8px; }}
  .rc-award-chip {{ width:24px; height:24px; border-radius:7px; display:grid; place-items:center;
                    flex:0 0 auto; font-size:12px; color:var(--rc-accent);
                    background:color-mix(in srgb, var(--rc-accent) 16%, transparent); }}
  .rc-award-lbl {{ font-size:10px; font-weight:800; letter-spacing:.07em; text-transform:uppercase; color:var(--muted); }}
  .rc-award-foot {{ font-size:11px; font-weight:600; color:var(--muted); }}
  .recap-view-matchup {{ display:inline-flex; align-items:center; gap:3px; font-size:12px;
                         font-weight:700; color:var(--accent); text-decoration:none; white-space:nowrap; }}
  .recap-view-matchup:hover {{ text-decoration:underline; }}
  .recap-matchup-score .recap-view-matchup {{ margin-top:6px; }}
  .recap-team .team-clickable {{ border-radius:6px; }}
  .recap-team-name .team-clickable:hover {{ text-decoration:underline; }}
  .st-name .team-clickable:hover {{ text-decoration:underline; }}
</style>
<section class="recap-section"><div class="recap-section-heading"><h2>Week at a Glance</h2></div>
<div class="rc-awards">
  {scorer_card("fa-solid fa-fire", "HIGH SCORER", high_row["owner"],
               float(high_row["points"]), str(high_row.get("roster_id", "")),
               high_sub, "var(--win)", medal_rank=1)}
  {scorer_card("fa-solid fa-arrow-trend-down", "LOW SCORER", low_row["owner"],
               float(low_row["points"]), str(low_row.get("roster_id", "")),
               low_sub, "var(--loss)" )}
  {matchup_card("fa-solid fa-trophy", "BIGGEST WIN", blowout, "var(--accent)") if blowout else '<div class="card rc-award"><div class="rc-award-h"><span class="rc-award-lbl">BIGGEST WIN</span></div><div class="rc-award-foot">No decisive result</div></div>'}
  {matchup_card("fa-solid fa-bolt", "CLOSEST GAME", closest, "var(--warning)") if closest else ""}
</div></section>"""

    # ── Scoreboard ─────────────────────────────────────────────────────────
    def top_performer_html(rid: str) -> str:
        performers = top_performers.get(str(rid)) or []
        if not performers:
            return ('<div class="recap-top-performer">'
                    '<span class="recap-top-label"><span class="recap-top-label-long">Top performer: </span>'
                    '<span class="recap-top-label-short">Top</span></span>'
                    '<span class="recap-top-performer-players"><span class="recap-top-performer-name">N/A</span></span>'
                    '</div>')
        links = []
        for player in performers:
            safe_name = html.escape(player["name"])
            safe_name_attr = html.escape(player["name"], quote=True)
            safe_pid = html.escape(player["pid"], quote=True)
            if player["pid"]:
                links.append(
                    f'<span class="player-clickable recap-top-performer-name" tabindex="0" role="button" '
                    f'data-player-id="{safe_pid}" data-player-name="{safe_name_attr}" '
                    f'data-wl-star-pid="{safe_pid}" '
                    f'data-league-id="{html.escape(str(_league_id), quote=True)}" '
                    f'data-platform="{html.escape(str(_platform), quote=True)}" '
                    f'data-season="{html.escape(str(_season), quote=True)}" '
                    f'aria-label="Open {safe_name_attr} player details">{safe_name}</span>'
                )
            else:
                links.append(f'<span class="recap-top-performer-name">{safe_name}</span>')
        names = '<span class="recap-top-performer-players">' + ' / '.join(links) + '</span>'
        return (
            f'<div class="recap-top-performer"><span class="recap-top-label">'
            f'<span class="recap-top-label-long">Top performer: </span>'
            f'<span class="recap-top-label-short">Top</span></span>{names}'
            f'<span class="recap-top-performer-points"><span class="recap-top-points-separator"> &middot; </span>'
            f'{performers[0]["pts"]:.1f} pts</span></div>'
        )

    badge_map = _matchup_badges(matchups)

    def matchup_result_row(m, matchup_index):
        w_team = team_name(m["winner"], m["w_rid"])
        l_team = team_name(m["loser"], m["l_rid"])
        result = "Tied" if m.get("tied") else f"Won by {m['margin']:.2f}"
        def team_block(owner, rid, name, points, side, winner=False):
            ava = team_link(owner, rid, ava_img(owner, rid, 36))
            nm = team_link(owner, rid, name, extra_class="recap-team-name-link")
            return f"""<div class="recap-team recap-team--{side}{' recap-team--winner' if winner else ''}">
              <div class="recap-team-identity">{ava}
                <div class="recap-team-copy"><div class="recap-team-name">{nm}</div>
                <div class="recap-team-manager">@{html.escape(owner)}</div></div>
                <strong class="recap-team-score">{points:.2f}</strong></div>
              {top_performer_html(rid)}
            </div>"""
        view_mu = (f'<a class="recap-view-matchup" '
                   f'href="/{_platform}/{_season}/{_league_id}/weekly?week={selected_week}">'
                   f'View matchup <span aria-hidden="true">&rsaquo;</span></a>')
        return f"""
<article class="recap-matchup-row">
  <div class="recap-matchup-badges">{''.join(f'<span>{label}</span>' for label in badge_map.get(matchup_index, []))}</div>
  <div class="recap-matchup-main">
    {team_block(m['winner'], m['w_rid'], w_team, m['w_pts'], 'left', not m.get('tied'))}
    <div class="recap-matchup-score"><div class="recap-matchup-scoreline">{m['w_pts']:.2f} <span>–</span> {m['l_pts']:.2f}</div>
      <div class="recap-matchup-margin">{result}</div>{view_mu}</div>
    <div class="recap-matchup-vs" aria-hidden="true"><span>VS</span></div>
    {team_block(m['loser'], m['l_rid'], l_team, m['l_pts'], 'right')}
    <div class="recap-matchup-footer">
      <span class="recap-matchup-margin">{result}</span>{view_mu}
    </div>
  </div>
</article>"""


    scoreboard_rows = "".join(matchup_result_row(m, i) for i, m in enumerate(matchups))
    scoreboard_html = f"""
<section class="recap-section recap-scoreboard-section">
<div class="recap-section-heading recap-scoreboard-header"><h2>Scoreboard</h2>
    <span class="recap-scoreboard-summary">
      <span class="recap-summary-desktop">Avg: {league_avg:.1f} &nbsp;·&nbsp; Total: {league_total:.1f}</span>
      <span class="recap-summary-mobile">Avg {league_avg:.1f} &nbsp;·&nbsp; Total {league_total:,.1f}</span>
    </span>
  </div>
<div class="card recap-scoreboard-card">{scoreboard_rows}</div></section>"""

    # ── Lineup efficiency ─────────────────────────────────────────────────
    def efficiency_team_row(row, rank=None):
        rid = row["rid"]
        rank_html = f'<span class="recap-eff-rank">{rank}</span>' if rank is not None else ""
        return f"""<div class="recap-eff-row">
          {rank_html}{ava_img("", rid, 34)}
          <div class="recap-eff-team"><div class="recap-eff-name">{team_link("", rid)}</div>
            <div class="recap-eff-detail">{row['actual']:.1f} / {row['optimal']:.1f} possible</div></div>
          <strong class="recap-eff-percent">{row['eff']:.0f}%</strong>
        </div>"""

    efficiency_help = ("Lineup efficiency measures how many of your optimal possible points you actually "
                       "started. Optimal points use the highest-scoring legal lineup from your roster that week.")
    if efficiency_rows:
        top_three_html = "".join(
            efficiency_team_row(row, rank) for rank, row in enumerate(efficiency_rows[:3], 1)
        )
        toughest_html = efficiency_team_row(most_left)
        toughest_insight = ("Perfect lineup" if most_left["missed"] < 0.05 else
                            f'<strong>{most_left["missed"]:.1f} pts</strong><span>left on the table</span>')
        partial_note = ('<div class="recap-eff-partial" role="status">Showing teams with complete historical data; '
                        'this is not a full-league ranking.</div>'
                        if efficiency_data.get("state") == "partial" else "")
        efficiency_content = f"""
<div class="recap-eff-grid">
  <div class="card recap-eff-card recap-eff-card--top"><h3>Top 3 Managers</h3>{top_three_html}</div>
  <div class="card recap-eff-card recap-eff-card--bench"><h3>Toughest Bench</h3>{toughest_html}
    <div class="recap-eff-insight">{toughest_insight}</div></div>
</div>{partial_note}"""
    else:
        state = efficiency_data.get("state")
        unavailable = ("No completed week is available yet." if state == "no_completed_week" else
                       "Lineup data is temporarily loading. Please try again shortly." if state == "loading_failure" else
                       "Historical roster, score, or position data is incomplete for this week.")
        efficiency_content = ('<div class="card recap-lineup-unavailable" role="status">'
                              f'{unavailable}</div>')
    efficiency_html = f"""
<section class="recap-section recap-efficiency">
  <div class="recap-eff-heading"><div><h2>Lineup Efficiency
    <button type="button" class="recap-eff-info" data-tooltip="{html.escape(efficiency_help, quote=True)}"
      aria-label="About lineup efficiency">i</button></h2>
    <p>Actual points vs. optimal lineup points (higher is better).</p></div></div>
  {efficiency_content}
</section>"""

    # ── Shared historical standings + power snapshot ───────────────────────
    # Preview mode replaces the empty provider frame with deterministic sample
    # rows. Feed that same frame into the shared historical builder rather than
    # accidentally asking it to index the original zero-column DataFrame.
    recap_ctx = dict(ctx)
    recap_ctx["df_weekly"] = fin_df
    historical_ctx = build_standings_as_of_week(recap_ctx, selected_week)
    from dashboard_services.ai.context_builders import build_power_rankings_context
    from utils.standings_divisions import resolve_divisions
    division_info = resolve_divisions(historical_ctx) or {}
    division_by_rid = division_info.get("by_rid") or {}
    division_names = division_info.get("names") or {}
    def _standings_rows(capped_ctx):
        frame = capped_ctx["df_weekly"].copy()
        frame["win"] = frame["points"] > frame["points_against"]
        frame["tie"] = frame["points"] == frame["points_against"]
        rows = []
        for rid, grp in frame.groupby("roster_id"):
            rows.append({
                "rid": str(rid), "owner": grp["owner"].iloc[0],
                "wins": int(grp["win"].sum()),
                "ties": int(grp["tie"].sum()),
                "losses": len(grp) - int(grp["win"].sum()) - int(grp["tie"].sum()),
                "pf": float(grp["points"].sum()),
                "division": division_by_rid.get(int(rid)) if str(rid).isdigit() else None,
            })
        rows.sort(key=lambda x: (x.get("division") or 9999, -x["wins"], -x["pf"]))
        return rows

    standings_rows_data = _standings_rows(historical_ctx)
    prior_standings = []
    prior_power_teams = []
    if selected_week > 1:
        prior_ctx = build_standings_as_of_week(recap_ctx, selected_week - 1)
        prior_standings = _standings_rows(prior_ctx)
        prior_power_teams = (build_power_rankings_context(prior_ctx) or {}).get("teams") or []
    from dashboard_services.recap_calculations import scoped_rank_movement
    standings_movement = scoped_rank_movement(standings_rows_data, prior_standings)

    def standing_row(rank, s):
        bar_pct = s["pf"] / max(r["pf"] for r in standings_rows_data) * 100 if standings_rows_data else 0
        lead = " lead" if rank == 1 else ""
        move = standings_movement.get(s["rid"])
        movement = (f'<span class="st-movement up">↑{move}</span>' if move and move > 0 else
                    f'<span class="st-movement down">↓{abs(move)}</span>' if move and move < 0 else
                    '<span class="st-movement flat">—</span>' if prior_standings else '')
        return f"""
<div class="st-row{' recap-viewer-team' if str(session.get('viewer_roster_id') or '') == s['rid'] else ''}">
  <div class="st-rank{lead}">{rank}</div>
  {ava_img(s["owner"], s["rid"], 28)}
  <div class="st-main">
    <div class="st-name">{team_link(s['owner'], s['rid'])}</div>
    <div class="st-bar"><div class="st-fill" style="width:{bar_pct:.0f}%;"></div></div>
  </div>
  <div class="st-rec">
    <div class="wl">{s['wins']}-{s['losses']}{f"-{s['ties']}" if s['ties'] else ''}</div>
    <div class="pf">{s['pf']:.1f} PF</div>
  </div>
  {movement}
</div>"""

    standing_parts, division_rank = [], {}
    last_div = object()
    for s in standings_rows_data:
        div = s.get("division")
        if div != last_div and div:
            standing_parts.append(f'<div class="st-division">{html.escape(division_names.get(div) or f"Division {div}")}</div>')
        peers = [x for x in standings_rows_data if x.get("division") == div] if div else standings_rows_data
        rank = peers.index(s) + 1
        division_rank[s["rid"]] = rank if div else None
        standing_parts.append(standing_row(rank, s))
        last_div = div
    standing_rows_html = "".join(standing_parts)
    standings_html = f"""
<div class="card" style="overflow:hidden;">
  <div class="card-header">
    <h3>Standings</h3>
    <a href="/{_platform}/{_season}/{_league_id}/standings?week={selected_week}" style="font-size:12px;color:var(--accent);">View full standings</a>
  </div>
  {standing_rows_html}
</div>"""

    power_teams = (build_power_rankings_context(historical_ctx) or {}).get("teams") or []
    prior_power_rank = {str(p.get("roster_id")): i for i, p in enumerate(prior_power_teams, 1)}
    power_deltas = {
        i: prior_power_rank.get(str(p.get("roster_id")), i) - i
        for i, p in enumerate(power_teams, 1)
        if str(p.get("roster_id")) in prior_power_rank
    }
    power_rows = []
    for i, p in enumerate(power_teams, 1):
        rid = str(p.get("roster_id") or "")
        p["power_rank"] = i
        score = p.get("power_score")
        score_text = f"{float(score):.1f}" if score is not None else ""
        delta = power_deltas.get(i)
        move = (f"<span class='st-movement {'up' if delta > 0 else 'down'}'>{'↑' if delta > 0 else '↓'}{abs(delta)}</span>"
                if delta else ("<span class='st-movement flat'>—</span>" if prior_power_teams else ""))
        power_rows.append(f'<div class="st-row{" recap-viewer-team" if str(session.get("viewer_roster_id") or "") == rid else ""}"><div class="st-rank{(" lead" if i == 1 else "")}">{i}</div>'
                          f'{ava_img(team_by_rid.get(rid, ""), rid, 28)}<div class="st-main"><div class="st-name">'
                          f'{team_link(team_by_rid.get(rid, ""), rid)}</div></div><div class="st-rec"><div class="wl">{score_text}</div></div>{move}</div>')
    power_html = f'<div class="card" style="overflow:hidden"><div class="card-header"><h3>Power Rankings</h3>' \
                 f'<a href="/{_platform}/{_season}/{_league_id}/teams" style="font-size:12px;color:var(--accent)">View power rankings</a></div>{"".join(power_rows)}</div>'
    standings_html = f'<section class="recap-section recap-changes"><div class="recap-section-heading"><h2>Standings &amp; Power Rankings</h2><small>Through Week {selected_week}</small></div><div class="recap-rank-grid">{standings_html}{power_html}</div></section>'

    power_by_rid = {str(p.get("roster_id")): p for p in power_teams}
    recap_team_context = {}
    for overall_rank, s in enumerate(sorted(standings_rows_data, key=lambda x: (-x["wins"], -x["pf"])), 1):
        rid, power = s["rid"], power_by_rid.get(s["rid"], {})
        item = {"standing_rank": overall_rank, "division_rank": division_rank.get(rid),
                "division_name": division_names.get(s.get("division")), "points_for": round(s["pf"], 1)}
        if power:
            item.update(power_rank=power.get("power_rank"), power_score=power.get("power_score"))
        recap_team_context[rid] = {k: v for k, v in item.items() if v is not None}

    # ── AI weekly storyline column + next-week game-of-the-week ────────────
    if preview_mode:
        from dashboard_services.ai.weekly_recap import get_weekly_ai_recap_preview
        ai_column_html, next_week_html = get_weekly_ai_recap_preview()
    else:
        # Build a next-week preview only when this recap is for the latest
        # finalized week overall (so the "upcoming" game is genuinely upcoming and
        # its availability is current), and that next week isn't already played.
        next_week = selected_week + 1
        next_week_ctx = None
        from dashboard_services.recap_calculations import upcoming_week_applicable
        if upcoming_week_applicable(selected_week, available_weeks):
            next_week_ctx = _build_next_week_ctx(
                ctx, next_week, playoff_start, _league_id, _season, _platform, team_by_rid,
            )

        from dashboard_services.ai.weekly_recap import get_weekly_ai_recap, get_weekly_ai_recap_teaser
        _has_prem = has_premium_for_viewer(
            session.get("viewer_username"), session.get("viewer_user_id"),
            _league_id, _platform, _season,
        )
        if not _has_prem:
            ai_column_html, next_week_html = get_weekly_ai_recap_teaser()
        else:
            ai_column_html, next_week_html = get_weekly_ai_recap(
                df_weekly=df_weekly,
                matchups_by_week=ctx.get("matchups_by_week") or {},
                selected_week=selected_week,
                team_by_rid=team_by_rid,
                league=league,
                league_id=ctx.get("league_id") or "",
                season=ctx.get("season") or "",
                next_week_ctx=next_week_ctx,
                team_context=recap_team_context,
                platform=_platform,
            )

    # ── Lineup analysis: busts, sleepers, coaching mistakes ────────────────
    if preview_mode:
        lineup_html = _mock_lineup_analysis_html(team_names)
    else:
        lineup_html = _build_lineup_analysis_html(
            ctx.get("matchups_by_week") or {},
            selected_week,
            team_by_rid,
            owner_avatar,
            ctx.get("roster_positions") or [],
            str(session.get("viewer_roster_id") or ""),
        )

    if lineup_html:
        lineup_html = (f'<section class="recap-section recap-decisions">'
                       f'<div class="recap-section-heading"><h2>Lineup Review</h2></div>'
                       f'{lineup_html}</section>')

    preview_banner = ""
    if preview_mode:
        preview_banner = """
<div class="recap-preview-watermark" style="position:relative;margin-bottom:16px;">
  <div style="display:flex;align-items:center;gap:10px;padding:12px 16px;
              border:1px solid var(--accent);border-radius:8px;background:rgba(99,102,241,0.08);">
    <span style="font-size:18px;">👁️</span>
    <div style="font-size:13px;color:var(--text);">
      <strong>Preview week</strong> -- this is sample data, not your league’s results.
      Your real weekly recap will appear here after Week 1 completes.
    </div>
  </div>
  <div aria-hidden="true" style="pointer-events:none;position:absolute;inset:0;overflow:hidden;border-radius:8px;">
    <div style="position:absolute;top:50%;left:-10%;right:-10%;transform:rotate(-12deg);
                text-align:center;font-size:42px;font-weight:800;letter-spacing:.18em;
                color:rgba(99,102,241,0.18);text-transform:uppercase;">SAMPLE</div>
  </div>
</div>"""

    story_html = (f'<section class="recap-section recap-story"><div class="recap-section-heading">'
                  f'<h2>Weekly Story</h2></div>{ai_column_html}</section>') if ai_column_html else ""
    up_next_html = ""
    weekly_url = f"/{_platform}/{_season}/{_league_id}/weekly?week={selected_week + 1}"
    if next_week_html:
        up_next_html = (f'<section class="recap-section recap-up-next"><div class="recap-section-heading">'
                        f'<h2>Up Next — Week {selected_week + 1}</h2></div>{next_week_html}'
                        f'<a class="recap-next-link" href="{weekly_url}">View matchup</a></section>')
    elif not preview_mode and selected_week < available_weeks[-1]:
        up_next_html = (f'<section class="recap-section recap-up-next"><div class="recap-section-heading">'
                        f'<h2>Up Next — Week {selected_week + 1}</h2></div><div class="card recap-next-history">'
                        f'This is a historical recap. <a href="{weekly_url}">View Week {selected_week + 1} matchups</a>.'
                        f'</div></section>')
    elif not preview_mode and selected_week >= playoff_start + 2 and history_url:
        up_next_html = (f'<section class="recap-section recap-up-next"><div class="recap-section-heading">'
                        f'<h2>Season Complete</h2></div><div class="card recap-next-history">'
                        f'<a href="{history_url}">Open season history and recap</a>.</div></section>')

    return ('<main class="weekly-recap">' + week_selector + preview_banner + history_banner
            + scoreboard_html + efficiency_html + cards_html + story_html + lineup_html + standings_html
            + up_next_html + '</main>')
