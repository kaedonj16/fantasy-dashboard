"""Weekly recap HTML builder.

Moved from app.py so the Flask monolith can keep shrinking. Helpers that still
live in app.py are lazy-imported inside the builder (request time).
"""
from __future__ import annotations

from typing import Optional

def build_recap_body(ctx: dict, selected_week: Optional[int] = None) -> str:
    from app import (  # noqa: E402  (lazy: avoids a circular import at module load)
        _build_lineup_analysis_html,
        _build_next_week_ctx,
        _build_recap_preview_df,
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
    if not reg_weeks:
        reg_weeks = available_weeks

    if selected_week is None or selected_week not in available_weeks:
        selected_week = reg_weeks[-1]

    week_df = fin_df[fin_df["week"] == selected_week].copy()

    # ── Matchup pairs ──────────────────────────────────────────────────────
    matchups: list[dict] = []
    for _, grp in week_df.groupby("matchup_id"):
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
        })
    matchups.sort(key=lambda x: -x["margin"])

    # ── Highlights ─────────────────────────────────────────────────────────
    high_row = week_df.loc[week_df["points"].idxmax()]
    low_row = week_df.loc[week_df["points"].idxmin()]
    blowout = matchups[0] if matchups else None
    closest = matchups[-1] if matchups else None

    league_avg = float(week_df["points"].mean()) if not week_df.empty else 0
    league_total = float(week_df["points"].sum()) if not week_df.empty else 0

    # Season high/low context
    fin_max = float(fin_df["points"].max())
    season_high = float(high_row["points"]) >= fin_max

    def ava_img(owner_name, rid="", size=32):
        ava = owner_avatar.get(owner_name, "")
        if ava:
            return f"<img src='{ava}' alt='' loading='lazy' decoding='async' style='width:{size}px;height:{size}px;border-radius:50%;object-fit:cover;flex-shrink:0;' onerror=\"this.style.display='none'\">"
        return team_crest(team_by_rid.get(rid) or owner_name or "?", size)

    def team_name(owner, rid=""):
        return html.escape(team_by_rid.get(rid) or owner or "–")

    # ── Week selector ──────────────────────────────────────────────────────
    week_opts = "".join(
        f"<option value='{w}' {'selected' if w == selected_week else ''}>Week {w}</option>"
        for w in reversed(reg_weeks)
    )
    history_banner = ""
    if history_url:
        history_banner = f"""
<div id="historyRecapBanner" style="display:flex;align-items:center;gap:10px;padding:11px 16px;
     margin-bottom:16px;border-radius:8px;background:var(--surface2);border:1px solid var(--border);">
  <i class="fa-solid fa-trophy" style="font-size:13px;color:var(--accent);flex-shrink:0;"></i>
  <span style="font-size:13px;color:var(--text);flex:1;">
    Want the full season breakdown? View it on the
    <a href="{history_url}" style="color:var(--accent);font-weight:600;text-decoration:none;">History page</a>.
  </span>
  <button onclick="this.parentElement.style.display='none'"
          style="background:none;border:none;color:var(--muted);cursor:pointer;padding:0 2px;
                 font-size:16px;line-height:1;flex-shrink:0;" aria-label="Dismiss">&#x2715;</button>
</div>"""

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
            result = f"{'Won' if pts > opp['pts'] else 'Lost'} by {diff:.1f}"
            body = f"""
  <div style="display:flex;flex-direction:column;gap:8px;">
    <div style="display:flex;align-items:center;gap:10px;">
      {ava_img(name, rid, 34)}
      <div style="flex:1;min-width:0;">
        <div style="font-size:14px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{team_name(name, rid)}</div>
      </div>
      <div style="font-size:23px;font-weight:800;color:{accent};flex-shrink:0;letter-spacing:-.5px;font-variant-numeric:tabular-nums;">{pts:.2f}</div>
    </div>
    <div style="height:1px;background:var(--border);"></div>
    <div style="display:flex;align-items:center;gap:10px;opacity:0.5;">
      {ava_img(opp["owner"], opp["rid"], 34)}
      <div style="flex:1;min-width:0;">
        <div style="font-size:14px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{team_name(opp["owner"], opp["rid"])}</div>
      </div>
      <div style="font-size:23px;font-weight:800;flex-shrink:0;letter-spacing:-.5px;font-variant-numeric:tabular-nums;">{opp["pts"]:.1f}</div>
    </div>
  </div>
  <div class="rc-award-foot"><span style="color:{accent};">{html.escape(sub)}</span> &middot; {result}</div>"""
        else:
            body = f"""
  <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
    <div style="display:flex;align-items:center;gap:10px;min-width:0;">
      {ava_img(name, rid, 42)}
      <div style="min-width:0;">
        <div style="font-weight:700;font-size:15px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{team_name(name, rid)}</div>
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
  <div class="rc-award-foot">margin {m['margin']:.1f}</div>
</div>"""

    high_sub = "Season high" if season_high else f"+{float(high_row['points']) - league_avg:.1f} vs avg"
    low_sub = f"{float(low_row['points']) - league_avg:.1f} vs avg"

    cards_html = f"""
<style>
  .rc-awards {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin-bottom:20px; }}
  @media (max-width:640px) {{ .rc-awards {{ grid-template-columns:1fr; }} }}
  .rc-award {{ position:relative; overflow:hidden; padding:15px 17px 14px; display:flex;
               flex-direction:column; gap:12px; min-width:0; }}
  .rc-award::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:var(--rc-accent); }}
  .rc-award-h {{ display:flex; align-items:center; gap:8px; }}
  .rc-award-chip {{ width:24px; height:24px; border-radius:7px; display:grid; place-items:center;
                    flex:0 0 auto; font-size:12px; color:var(--rc-accent);
                    background:color-mix(in srgb, var(--rc-accent) 16%, transparent); }}
  .rc-award-lbl {{ font-size:10px; font-weight:800; letter-spacing:.07em; text-transform:uppercase; color:var(--muted); }}
  .rc-award-foot {{ font-size:11px; font-weight:600; color:var(--muted); }}
</style>
<div class="rc-awards">
  {scorer_card("fa-solid fa-fire", "HIGH SCORER", high_row["owner"],
               float(high_row["points"]), str(high_row.get("roster_id", "")),
               high_sub, "var(--win)", _scorer_opp(high_row), medal_rank=1)}
  {scorer_card("fa-solid fa-arrow-trend-down", "LOW SCORER", low_row["owner"],
               float(low_row["points"]), str(low_row.get("roster_id", "")),
               low_sub, "var(--loss)", _scorer_opp(low_row))}
  {matchup_card("fa-solid fa-trophy", "BIGGEST WIN", blowout, "var(--accent)") if blowout else ""}
  {matchup_card("fa-solid fa-bolt", "CLOSEST GAME", closest, "var(--warning)") if closest else ""}
</div>"""

    # ── Scoreboard ─────────────────────────────────────────────────────────
    def matchup_result_row(m):
        w_team = team_name(m["winner"], m["w_rid"])
        l_team = team_name(m["loser"], m["l_rid"])
        margin_color = "var(--loss)" if m["margin"] > 50 else ("var(--warning)" if m["margin"] > 20 else "var(--win)")
        return f"""
<div style="display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid var(--border);">
  <div style="flex:1;min-width:0;display:flex;align-items:center;gap:8px;">
    {ava_img(m["winner"], m["w_rid"], 30)}
    <div style="min-width:0;">
      <div style="font-weight:700;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{w_team}</div>
      <div style="font-size:10px;color:var(--muted);">@{html.escape(m["winner"])}</div>
    </div>
  </div>
  <div style="text-align:center;flex-shrink:0;min-width:110px;">
    <div style="font-size:15px;font-weight:800;">{m['w_pts']:.2f} <span style="color:var(--muted);font-weight:400;font-size:12px;">–</span> {m['l_pts']:.2f}</div>
    <div style="font-size:10px;color:{margin_color};font-weight:700;">+{m['margin']:.2f}</div>
  </div>
  <div style="flex:1;min-width:0;display:flex;align-items:center;gap:8px;justify-content:flex-end;">
    <div style="min-width:0;text-align:right;">
      <div style="font-weight:600;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{l_team}</div>
      <div style="font-size:10px;color:var(--muted);">@{html.escape(m["loser"])}</div>
    </div>
    {ava_img(m["loser"], m["l_rid"], 30)}
  </div>
</div>"""

    scoreboard_rows = "".join(matchup_result_row(m) for m in matchups)
    scoreboard_html = f"""
<div class="card" style="overflow:hidden;">
  <div class="card-header">
    <h3>Scoreboard</h3>
    <span style="font-size:12px;color:var(--muted);">
      Avg: {league_avg:.1f} &nbsp;·&nbsp; Total: {league_total:.1f}
    </span>
  </div>
  {scoreboard_rows}
</div>"""

    # ── Season standings snapshot ──────────────────────────────────────────
    # Compute cumulative wins/losses/PF through selected_week
    cum_df = fin_df[fin_df["week"] <= selected_week].copy()
    cum_df["win"] = cum_df["points"] > cum_df["points_against"]
    standings_rows_data = []
    for rid, grp in cum_df.groupby("roster_id"):
        owner = grp["owner"].iloc[0]
        wins = int(grp["win"].sum())
        losses = len(grp) - wins
        pf = float(grp["points"].sum())
        standings_rows_data.append({
            "rid": str(rid), "owner": owner,
            "wins": wins, "losses": losses, "pf": pf,
        })
    standings_rows_data.sort(key=lambda x: (-x["wins"], -x["pf"]))

    def standing_row(rank, s):
        bar_pct = s["pf"] / max(r["pf"] for r in standings_rows_data) * 100 if standings_rows_data else 0
        lead = " lead" if rank == 1 else ""
        return f"""
<div class="st-row">
  <div class="st-rank{lead}">{rank}</div>
  {ava_img(s["owner"], s["rid"], 28)}
  <div class="st-main">
    <div class="st-name">{team_name(s['owner'], s['rid'])}</div>
    <div class="st-bar"><div class="st-fill" style="width:{bar_pct:.0f}%;"></div></div>
  </div>
  <div class="st-rec">
    <div class="wl">{s['wins']}-{s['losses']}</div>
    <div class="pf">{s['pf']:.1f} PF</div>
  </div>
</div>"""

    standing_rows_html = "".join(standing_row(i + 1, s) for i, s in enumerate(standings_rows_data))
    standings_html = f"""
<div class="card" style="overflow:hidden;">
  <div class="card-header">
    <h3>Standings</h3>
    <span style="font-size:12px;color:var(--muted);">Through week {selected_week}</span>
  </div>
  {standing_rows_html}
</div>"""

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
        if selected_week == available_weeks[-1] and next_week not in available_weeks:
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
        )

    preview_banner = ""
    if preview_mode:
        preview_banner = """
<div class="recap-preview-watermark" style="position:relative;margin-bottom:16px;">
  <div style="display:flex;align-items:center;gap:10px;padding:12px 16px;
              border:1px solid var(--accent);border-radius:8px;background:rgba(99,102,241,0.08);">
    <span style="font-size:18px;">👁️</span>
    <div style="font-size:13px;color:var(--text);">
      <strong>Preview week</strong> — this is sample data, not your league’s results.
      Your real weekly recap will appear here after Week 1 completes.
    </div>
  </div>
  <div aria-hidden="true" style="pointer-events:none;position:absolute;inset:0;overflow:hidden;border-radius:8px;">
    <div style="position:absolute;top:50%;left:-10%;right:-10%;transform:rotate(-12deg);
                text-align:center;font-size:42px;font-weight:800;letter-spacing:.18em;
                color:rgba(99,102,241,0.18);text-transform:uppercase;">SAMPLE</div>
  </div>
</div>"""

    scoreboard_and_recap = f"""
<div class="recap-scoreboard-grid" style="display:grid;grid-template-columns:3fr 2fr;gap:16px;margin-bottom:20px;align-items:stretch;">
  {scoreboard_html.replace("margin-bottom:20px;", "")}
  {ai_column_html.replace("margin-bottom:20px;", "")}
</div>"""

    return (preview_banner + history_banner + week_selector + cards_html
            + scoreboard_and_recap + (next_week_html or "") + lineup_html + standings_html)

