from __future__ import annotations

import json
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Any, Optional

from .api import get_matchups, get_users, get_rosters, _avatar_from_users, get_nfl_state
from .utils import fetch_week_from_tank01, get_week_projections_cached, write_json


def get_owner_id(rosters: Optional[list[dict]] = None, roster_id: Optional[str] = None) -> Optional[str]:
    return next((r["owner_id"] for r in rosters if str(r.get("roster_id")) == str(roster_id)), None)


def build_matchup_preview(league_id: str, week: int, roster_map: Dict[str, str],
                          players_map: Dict[str, Dict[str, str]]) -> List[dict]:
    mlist = get_matchups(league_id, week) or []
    by_mid: Dict[Any, List[dict]] = {}
    for m in mlist:
        mid = m.get("matchup_id")
        by_mid.setdefault(mid, []).append(m)

    def _from_players_map(pid: str) -> Dict[str, str]:
        info = players_map.get(pid) if players_map else None
        if info:
            name = info.get("name") or pid
            nfl = info.get("team") or "FA"
            pos = info.get("pos") or (info.get("fantasy_positions", [""])[0] if info.get("fantasy_positions") else "")
            return {"name": name, "nfl": nfl, "pos": pos}
        if pid.isalpha() and 2 <= len(pid) <= 3:
            return {"name": f"{pid} D/ST", "nfl": pid, "pos": "DEF"}
        return {"name": pid, "nfl": "FA", "pos": ""}

    def _pinfo(pid: str, pts_map: Dict[str, float]) -> dict:
        base = _from_players_map(pid)
        pts = pts_map.get(pid) if pts_map else None
        return {"pid": pid, "name": base["name"], "pos": base["pos"], "nfl": base["nfl"], "pts": pts}

    def _team_block(row: dict) -> dict:
        rid = str(row.get("roster_id"))
        users = get_users(league_id)
        rosters = get_rosters(league_id)
        starters_raw = [s for s in (row.get("starters") or []) if s]
        pts_map = {str(k): v for k, v in (row.get("players_points") or {}).items()}
        s_infos: List[dict] = [_pinfo(str(pid), pts_map) for pid in starters_raw]
        pts_total = float(row["points"]) if isinstance(row.get("points"), (int, float)) else None
        wins, losses = 0, 0
        for r in rosters:
            if str(r.get("roster_id")) == rid:
                wins = r.get("settings", {}).get("wins", 0)
                losses = r.get("settings", {}).get("losses", 0)
        username = next((u.get("display_name") for u in users if u["user_id"] == get_owner_id(rosters, rid)), None)
        return {
            "name": roster_map.get(rid, f"Roster {rid}"),
            "starters": s_infos,
            "pts_total": pts_total,
            "avatar": _avatar_from_users(users, get_owner_id(rosters, rid)),
            "record": f"{wins}-{losses}",
            "username": username,
        }

    out = []
    for mid, rows in by_mid.items():
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: str(r.get("roster_id")))
        left = _team_block(rows_sorted[0])
        right = _team_block(rows_sorted[1]) if len(rows_sorted) > 1 else {"name": "TBD", "avatar": None, "starters": [],
                                                                          "pts_total": None}
        out.append({"matchup_id": mid, "left": left, "right": right})
    return out[:5]


def render_matchup_carousel_weeks(slides_by_week: dict[int, str]) -> str:
    opts = "".join(
        f"<option value='{w}'{' selected' if w == get_nfl_state().get('week') else ''}>Week {w}</option>"
        for w in sorted(slides_by_week.keys())
    )
    json_slides = json.dumps({str(k): v for k, v in slides_by_week.items()})
    carousel_script = f"""
    (function(){{
      const slidesByWeek = {json_slides};
      const weekSel = document.getElementById('mWeek');
      const track   = document.getElementById('mTrack');
      const prevBtn = document.getElementById('mPrev');
      const nextBtn = document.getElementById('mNext');
      function setWeek(w){{
        const html = slidesByWeek[w] || "<div class='m-empty'>No matchups</div>";
        track.innerHTML = html;
        idx = 0; cacheSlides(); update();
      }}
      let idx = 0; let slides = [];
      function cacheSlides() {{ slides = track.querySelectorAll('.m-slide'); }}
      function update() {{
        const w = track.clientWidth;
        track.scrollTo({{left: idx * w, behavior: 'smooth'}});
        if (prevBtn) prevBtn.disabled = (idx === 0);
        if (nextBtn) nextBtn.disabled = (idx >= Math.max(0, slides.length - 1));
      }}
      prevBtn && prevBtn.addEventListener('click', ()=>{{ idx = Math.max(0, idx - 1); update(); }});
      nextBtn && nextBtn.addEventListener('click', ()=>{{ idx = Math.min(Math.max(0, slides.length - 1), idx + 1); update(); }});
      window.addEventListener('resize', update);
      weekSel && weekSel.addEventListener('change', (e)=> setWeek(e.target.value));
      cacheSlides(); update();
    }})();
    """
    return f"""
    <div class="card central" data-section='matchups'>
      <div class="m-nav">
        <h2>Matchup Preview</h2>
        <div class="m-controls">
          <select id="mWeek" class="search">{opts}</select>
          <button class="m-btn" id="mPrev">‹ Prev</button>
          <button class="m-btn" id="mNext">Next ›</button>
        </div>
      </div>
      <div class="m-carousel">
        <div class="m-track" id="mTrack">
          {slides_by_week.get(get_nfl_state().get('week'), "<div class='m-empty'>No matchups</div>")}
        </div>
      </div>
    </div>
    <script>{carousel_script}</script>
    """


def add_bye_weeks_to_players():
    player_path = Path("cache/players_index.json")
    team_path = Path("cache/teams_index.json")
    with open(player_path, 'r') as file:
        players = json.load(file)
    with open(team_path, 'r') as file:
        teams = json.load(file)
    for player_id, player_data in players.items():
        team_abv = player_data.get("team")
        if team_abv in teams:
            player_data["byeWeek"] = teams[team_abv]["byeWeek"]

    write_json(player_path, players)


def _render_matchup_slide(m: dict, w: int, proj_week: int, season: int) -> str:
    """One slide with rows like:
       [Left Name] [Left Pts/Proj] [Right Pts/Proj] [Right Name]
    """
    projections = {}
    players = {}
    teams = {}
    proj = w > proj_week

    if proj:
        projection_path = Path(f"cache/projections_s{season}_w{w}.json")
        player_path = Path("cache/players_index.json")
        team_path = Path("cache/teams_index.json")
        if projection_path.exists():
            with open(projection_path, 'r') as file:
                projections = json.load(file)
        else:
            get_week_projections_cached(season, w, fetch_week_from_tank01)
        with open(player_path, 'r') as file:
            players = json.load(file)
        with open(team_path, 'r') as file:
            teams = json.load(file)

    def team_head(t, proj):
        points = (f"{t['pts_total']:.2f}" if isinstance(t.get("pts_total"), (int, float)) else "—")
        if proj:
            val = 0.0
            starters = t['starters']
            for p in starters:
                pid = p.get("pid")
                player = players.get(pid)
                if player is None:
                    player = teams.get(pid)
                if player:
                    val += projections.get(pid, 0.0)
                points = f"{val:.2f}"
        ava = t.get("avatar") or ""
        img = f"<img class='avatar' src='{ava}' onerror=\"this.style.display='none'\">" if ava else ""
        if proj:
            return f"""
        <div class="m-team">
          {img}
          <div>
          <div class="name left">{t['name']}</div>
          <div>
          <div>{t['record']} • @{t['username']}</div>
          </div>
          </div>
          <div style="display: grid;grid-template-columns: 1;">
            <span class="num">0.0</span>
            <span class="proj" style='opacity: 0.4; text-align:center;'">{points}</span>
          </div>
        </div>
        """
        else:
            return f"""
        <div class="m-team">
          {img}
          <div>
          <div class="name left">{t['name']}</div>
          <div>
          <div>{t['record']} • @{t['username']}</div>
          </div>
          </div>
          <span class="num">{points}</span>
        </div>
        """

    def team_head_2nd(t, proj):
        points = (f"{t['pts_total']:.2f}" if isinstance(t.get("pts_total"), (int, float)) else "—")
        print(t)
        if proj:
            val = 0.0
            starters = t['starters']
            for p in starters:
                pid = p.get("pid")
                player = players.get(pid)
                if player is None:
                    player = teams.get(pid)
                if player:
                    val += projections.get(pid, 0.0)
                points = f"{val:.2f}"
        ava = t.get("avatar") or ""
        img = f"<img class='avatar' src='{ava}' onerror=\"this.style.display='none'\">" if ava else ""

        if proj:
            return f"""
        <div class="m-team">
          <div style="display: grid;grid-template-columns: 1;">
            <span class="num">0.0</span>
            <span class="proj" style='opacity: 0.4; text-align:center;'">{points}</span>
          </div>
          <div class = "right">
          <div class="name">{t['name']}</div>
          <div>@{t['username']} • {t['record']}</div>
          </div>
          {img}
        </div>
        """
        else:
            return f"""
        <div class="m-team">
         <span class="num">{points}</span>
          <div class = "right">
          <div class="name">{t['name']}</div>
          <div>@{t['username']} • {t['record']}</div>
          </div>
          {img}
        </div>
        """

    # Build 10 rows, pairing i-th starter from each side

    rows_html = []
    for L, R in zip_longest(m["left"].get("starters", []),
                            m["right"].get("starters", []), fillvalue=None):
        # Helper to format one player cell and score
        def player_bits(p, side: str, left: bool):
            if not p:
                return "", ""
            name = p.get("name", "")
            nfl = p.get("nfl", "")
            pos = p.get("pos")
            val = p.get("pts")
            player = None
            # show **actual points** if present, else projection if present
            if players != {} and projections != {}:
                pid = p.get("pid")
                player = players.get(pid)
                if player is None:
                    player = teams.get(pid)
                if player:
                    val = projections.get(pid)
                    if val == 0.0 and player.get("byeWeek") == w:
                        val = "BYE"
            # Compute score clearly without nested conditional expressions
            if isinstance(val, (int, float)):
                score = f"{val:.1f}"
            elif player and player.get("byeWeek") == w:
                score = "BYE"
            else:
                score = 0.0
            # name on the outside, small NFL tag
            if (left):
                if proj:
                    if score == "BYE":
                        cell = (
                            f"<div class='p {side}' style='opacity:0.4;><span class='pos-badge {pos}'>{pos}</span><span class='pname'>{name}</span>"
                            f"<span class='meta'>&nbsp;{nfl}</span></div>")
                    else:
                        cell = (
                            f"<div class='p {side}'><span class='pos-badge {pos}'>{pos}</span><span class='pname'>{name}</span>"
                            f"<span class='meta'>&nbsp;{nfl}</span></div>")
                else:
                    cell = (
                        f"<div class='p {side}'><span class='pos-badge {pos}'>{pos}</span><span class='pname'>{name}</span>"
                        f"<span class='meta'>&nbsp;{nfl}</span></div>")
            else:
                if score == "BYE":
                    cell = (
                        f"<div class='p {side}' style='justify-content: flex-end; opacity:0.4;'><span class='meta'>&nbsp;{nfl}</span>"
                        f"<span class='pname'>{name}</span> <span class='pos-badge {pos}'>{pos}</span></div>")
                else:
                    cell = (
                        f"<div class='p {side}' style='justify-content: flex-end;'><span class='meta'>&nbsp;{nfl}</span>"
                        f"<span class='pname'>{name}</span> <span class='pos-badge {pos}'>{pos}</span></div>")
            return cell, score

        left_cell, left_score = player_bits(L, "left", True)
        right_cell, right_score = player_bits(R, "right", False)
        if left_score == "BYE":
            points = (f"<span class='num mid l' style='opacity: 0.4;'>{left_score}</span>"
                      f"<span class='num mid r'>{right_score}</span>")
        elif right_score == "BYE":
            points = (f"<span class='num mid l'>{left_score}</span>"
                      f"<span class='num mid r' style='opacity: 0.4;'>{right_score}</span>")
        elif float(left_score) > float(right_score):
            points = (f"<span class='num mid l more'>{left_score}</span>"
                      f"<span class='num mid r'>{right_score}</span>")
        elif float(left_score) < float(right_score):
            points = (f"<span class='num mid l'>{left_score}</span>"
                      f"<span class='num mid r more'>{right_score}</span>")
        else:
            points = (f"<span class='num mid l'>{left_score}</span>"
                      f"<span class='num mid r'>{right_score}</span>")

        rows_html.append(
            f"""<div class="m-row">
                  {left_cell}
                  {points}
                  {right_cell}
                </div>"""
        )

    return f"""
    <div class="m-slide">
      <div class="m-head">
        {team_head(m['left'], proj)}
        <div class="m-vs">vs</div>
        {team_head_2nd(m['right'], proj)}
      </div>
      <div class="m-body">
        <div class="m-combo">
          {''.join(rows_html)}
        </div>
      </div>
    </div>
    """
