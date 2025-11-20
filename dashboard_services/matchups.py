from __future__ import annotations

import json
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Any, Optional

from .api import get_matchups, get_users, get_rosters, _avatar_from_users, get_nfl_state
from .utils import write_json

STATUS_NOT_STARTED = "not_started"
STATUS_IN_PROGRESS = "in_progress"
STATUS_FINAL = "final"


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


def render_matchup_carousel_weeks(slides_by_week: dict[int, str], dashboard: bool) -> str:
    json_slides = json.dumps({str(k): v for k, v in slides_by_week.items()})
    carousel_script = f"""
    (function(){{
      const slidesByWeek = {json_slides};
      const weekSel = document.getElementById('hubWeek') || document.getElementById('mWeek');
      const track   = document.getElementById('mTrack');
      const prevBtn = document.getElementById('mPrev');
      const nextBtn = document.getElementById('mNext');

      let idx = 0;
      let slides = [];

      function cacheSlides() {{
        slides = track.querySelectorAll('.m-slide');
      }}

      function update() {{
        const w = track.clientWidth;
        track.scrollTo({{ left: idx * w, behavior: 'smooth' }});
        if (prevBtn) prevBtn.disabled = (idx === 0);
        if (nextBtn) nextBtn.disabled = (idx >= Math.max(0, slides.length - 1));
      }}

      function setWeek(w) {{
        const html = slidesByWeek[w] || "<div class='m-empty'>No matchups</div>";
        track.innerHTML = html;
        idx = 0;
        cacheSlides();
        update();
      }}

      // Hook into hubWeek if it exists
      if (weekSel) {{
        setWeek(weekSel.value);
        weekSel.addEventListener('change', (e) => setWeek(e.target.value));
      }} else {{
        const keys = Object.keys(slidesByWeek);
        if (keys.length) {{
          setWeek(keys[0]);
        }}
      }}

      prevBtn && prevBtn.addEventListener('click', () => {{
        idx = Math.max(0, idx - 1); 
        update();
      }});

      nextBtn && nextBtn.addEventListener('click', () => {{
        idx = Math.min(Math.max(0, slides.length - 1), idx + 1);
        update();
      }});

      window.addEventListener('resize', update);
    }})();
    """

    central = "central" if dashboard else ""
    style = "max-width:800px;" if not dashboard else ""
    return f"""
        <div class="card {central}" data-section='matchups' style='{style} margin-bottom:30px;'>
        <div class="m-nav">
          <h2>Matchup Preview</h2>
          <div class="m-controls">
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

    # return f""" <div class="card central" data-section='matchups' style='margin-bottom:30px;'> <div class="m-nav"> <h2>Matchup Preview</h2> <div class="m-controls"> <select id="mWeek" class="search">{opts}</select> <button class="m-btn" id="mPrev">‹ Prev</button> <button class="m-btn" id="mNext">Next ›</button> </div> </div> <div class="m-carousel"> <div class="m-track" id="mTrack"> {slides_by_week.get(get_nfl_state().get('week'), "<div class='m-empty'>No matchups</div>")} </div> </div> </div> <script>{carousel_script}</script> """


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


def team_live_totals(team: dict,
                     status_by_pid: dict[str, str],
                     projections: dict,
                     ) -> tuple[float, float]:
    """
    actual_total:
        sum of all actual points for starters (p['pts'])
    live_proj_total:
        - players not started  -> use projection
        - players started/finished -> use actual
    """
    actual_total = 0.0
    live_proj_total = 0.0

    for p in team.get("starters", []):
        pid = p.get("pid")

        actual = p.get("pts") or 0.0
        actual_total += actual

        status = status_by_pid.get(pid, STATUS_NOT_STARTED)
        proj_val = projections.get(pid, 0.0)

        if status is STATUS_NOT_STARTED:
            live_proj_total += proj_val
        else:
            live_proj_total += actual

    return actual_total, live_proj_total


def compute_team_projections_for_weeks(
        matchups_by_week: dict[int, list[dict]],
        status_by_pid: dict[str, str],
        projections: dict,
        roster_map: dict[str, str]  # roster_id -> owner
) -> dict[tuple[int, str], float]:
    """
    Returns {(week, roster_id): live_proj_total}
    """
    proj_by_roster: dict[tuple[int, str], float] = {}

    # build reverse map owner -> roster_id if needed
    owner_to_rid = {owner: rid for rid, owner in roster_map.items()}

    for week, matchups in matchups_by_week.items():
        for m in matchups:
            for side in ("left", "right"):
                team = m[side]

                # however you can get roster_id – adjust as needed:
                rid = team.get("roster_id")
                if rid is None:
                    rid = owner_to_rid.get(team["name"])

                if rid is None:
                    continue

                _, live_proj_total = team_live_totals(
                    team, status_by_pid.get(week).get("statuses"), projections.get(week).get("projections")
                )
                proj_by_roster[(week, str(rid))] = live_proj_total

    return proj_by_roster


def render_matchup_slide(
        m: dict,
        w: int,
        proj_week: int,
        status_by_pid: dict[str, str],
        projections: dict[str, float],
        players: dict,
        teams: dict,
) -> str:
    """One slide with rows like:
       [Left Name] [Left Pts/Proj] [Right Pts/Proj] [Right Name]
    """
    # proj flag: you can still use this to change team headers if you want
    proj = w > proj_week

    # ---- 1) Team headers ----
    def team_head(t, proj_mode: bool):
        """
        Left team header.

        proj_mode == False:
            - behaves like your original: uses t['pts_total']
        proj_mode == True:
            - top: actual total (from starters' pts)
            - bottom: live projection (actual for started/finished, proj for not started)
        """
        ava = t.get("avatar") or ""
        img = f"<img class='avatar' src='{ava}' onerror=\"this.style.display='none'\">" if ava else ""

        if not proj_mode:
            points = f"{t['pts_total']:.2f}" if isinstance(t.get("pts_total"), (int, float)) else "—"
            return f"""
        <div class="m-team">
          {img}
          <div>
            <div class="name left">{t['name']}</div>
            <div>{t['record']} • @{t['username']}</div>
          </div>
          <span class="num">{points}</span>
        </div>
        """

        actual_total, live_proj_total = team_live_totals(t, status_by_pid, projections.get(w).get("projections"))

        return f"""
        <div class="m-team">
          {img}
          <div>
            <div class="name left">{t['name']}</div>
            <div>{t['record']} • @{t['username']}</div>
          </div>
          <div style="display:grid;grid-template-columns:1;">
            <span class="num">{actual_total:.1f}</span>
            <span class="proj" style="opacity:0.4;text-align:center;">{live_proj_total:.1f}</span>
          </div>
        </div>
        """

    def team_head_2nd(t, proj_mode: bool):
        """
        Right team header – mirrored layout.
        """
        ava = t.get("avatar") or ""
        img = f"<img class='avatar' src='{ava}' onerror=\"this.style.display='none'\">" if ava else ""

        if not proj_mode:
            points = f"{t['pts_total']:.2f}" if isinstance(t.get("pts_total"), (int, float)) else "—"
            return f"""
        <div class="m-team">
          <span class="num">{points}</span>
          <div class="right">
            <div class="name">{t['name']}</div>
            <div>@{t['username']} • {t['record']}</div>
          </div>
          {img}
        </div>
        """

        actual_total, live_proj_total = team_live_totals(t, status_by_pid, projections.get(w).get("projections"))
        return f"""
        <div class="m-team">
          <div style="display:grid;grid-template-columns:1;">
            <span class="num">{actual_total:.1f}</span>
            <span class="proj" style="opacity:0.4;text-align:center;">{live_proj_total:.1f}</span>
          </div>
          <div class="right">
            <div class="name">{t['name']}</div>
            <div>@{t['username']} • {t['record']}</div>
          </div>
          {img}
        </div>
        """

    # ---- 2) Per-player bits with status + proj/actual rules ----
    def player_bits(p, side: str, left_side: bool):
        if not p:
            return "", 0.0, None, False  # cell_html, actual_val, proj_val, is_bye

        pid = p.get("pid")
        name = p.get("name", "")
        nfl = p.get("nfl", "")
        pos = p.get("pos")

        # Actual fantasy points (from Sleeper)
        actual = p.get("pts") or 0.0

        # Projection from Tank01 for this player/team
        proj_val = None
        is_bye = False

        player_index = players.get(pid) or teams.get(pid)
        if player_index:
            proj_val = projections.get(w, {}).get("projections", {}).get(pid, 0.0)
            if proj_val == 0.0 and player_index.get("byeWeek") == w:
                is_bye = True

        # Status
        status = status_by_pid.get(pid, STATUS_NOT_STARTED)
        if status == "BYE":
            is_bye = True

        # Rule 1: not started -> 0.0 actual, muted projection
        if status == STATUS_NOT_STARTED and not is_bye:
            display_actual = 0.0
            display_proj = proj_val if proj_val is not None else 0.0

        # Rule 2: in progress -> only actual
        elif status == STATUS_IN_PROGRESS and not is_bye:
            display_actual = actual
            display_proj = None

        # Rule 3: final (even 0) -> only actual
        elif status == STATUS_FINAL and not is_bye:
            display_actual = actual
            display_proj = None

        # BYE
        elif is_bye:
            display_actual = 0.0
            display_proj = None

        else:
            # Fallback – treat as not started
            display_actual = 0.0 if actual is None else actual
            display_proj = proj_val if proj_val is not None else 0.0

        # Player cell HTML
        if left_side:
            if is_bye:
                cell = (
                    f"<div class='p {side}' style='opacity:0.4;'>"
                    f"<span class='pos-badge {pos}'>{pos}</span>"
                    f"<span class='pname'>{name}</span>"
                    f"<span class='meta'>&nbsp;{nfl}</span>"
                    f"</div>"
                )
            else:
                cell = (
                    f"<div class='p {side}'>"
                    f"<span class='pos-badge {pos}'>{pos}</span>"
                    f"<span class='pname'>{name}</span>"
                    f"<span class='meta'>&nbsp;{nfl}</span>"
                    f"</div>"
                )
        else:
            if is_bye:
                cell = (
                    f"<div class='p {side}' style='justify-content:flex-end; opacity:0.4;'>"
                    f"<span class='meta'>&nbsp;{nfl}</span>"
                    f"<span class='pname'>{name}</span>"
                    f" <span class='pos-badge {pos}'>{pos}</span>"
                    f"</div>"
                )
            else:
                cell = (
                    f"<div class='p {side}' style='justify-content:flex-end;'>"
                    f"<span class='meta'>&nbsp;{nfl}</span>"
                    f"<span class='pname'>{name}</span>"
                    f" <span class='pos-badge {pos}'>{pos}</span>"
                    f"</div>"
                )

        return cell, float(display_actual), display_proj, is_bye

    # ---- 3) Build rows ----
    rows_html = []

    for L, R in zip_longest(m["left"].get("starters", []),
                            m["right"].get("starters", []),
                            fillvalue=None):
        left_cell, left_actual, left_proj, left_is_bye = player_bits(L, "left", True)
        right_cell, right_actual, right_proj, right_is_bye = player_bits(R, "right", False)

        # Compare actual for highlighting
        la = 0.0 if left_is_bye else left_actual
        ra = 0.0 if right_is_bye else right_actual

        left_more = la > ra
        right_more = ra > la

        def score_stack(actual_val, proj_val, side: str, is_bye: bool, more: bool) -> str:
            if is_bye:
                return (
                    "<div class='num-stack' style='display:grid'>"
                    f"<span class='num mid {side}' style='opacity:0.4;'>BYE</span>"
                    "</div>"
                )
            if proj_val is None:
                cls = f"num mid {side}" + (" more" if more else "")
                return (
                    "<div class='num-stack' style='display:grid'>"
                    f"<span class='{cls}'>{actual_val:.1f}</span>"
                    "</div>"
                )

            cls_actual = f"num mid {side}" + (" more" if more else "")
            return (
                "<div class='num-stack' style='display:grid'>"
                f"<span class='{cls_actual}'>{actual_val:.1f}</span>"
                f"<span class='num mid {side} proj' style='opacity:0.4;'>{proj_val:.1f}</span>"
                "</div>"
            )

        left_points_html = score_stack(left_actual, left_proj, "l", left_is_bye, left_more)
        right_points_html = score_stack(right_actual, right_proj, "r", right_is_bye, right_more)

        points = f"{left_points_html}{right_points_html}"

        rows_html.append(
            f"""<div class="m-row">
                  {left_cell}
                  {points}
                  {right_cell}
                </div>"""
        )

    # ---- 4) Final slide HTML ----
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
