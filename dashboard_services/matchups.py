from __future__ import annotations

import json
from datetime import datetime, date
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Any, Optional

from dashboard_services.api import (
    get_matchups,
    get_users,
    get_rosters,
    avatar_from_users,
    get_nfl_state,
    get_nfl_scores_for_date,
    build_team_game_lookup,
)
from dashboard_services.data_building.value_model_training import normalize_name
from dashboard_services.utils import write_json, load_week_schedule, load_teams_index, load_week_stats

STATUS_NOT_STARTED = "not_started"
STATUS_IN_PROGRESS = "in_progress"
STATUS_FINAL = "final"


def get_owner_id(rosters: Optional[list[dict]] = None, roster_id: Optional[str] = None) -> Optional[str]:
    return next((r["owner_id"] for r in rosters if str(r.get("roster_id")) == str(roster_id)), None)


def build_matchup_preview(
    league_id: str,
    week: int,
    roster_map: Dict[str, str],
    players_map: Dict[str, Dict[str, str]],
) -> List[dict]:
    mlist = get_matchups(league_id, week) or []
    if not mlist:
        return []

    # Pre-fetch users/rosters once instead of per team
    users = get_users(league_id) or []
    rosters = get_rosters(league_id) or []

    # Precompute maps for fast lookup
    owner_id_by_rid: Dict[str, Optional[str]] = {}
    record_by_rid: Dict[str, tuple[int, int]] = {}
    for r in rosters:
        rid_str = str(r.get("roster_id"))
        owner_id_by_rid[rid_str] = r.get("owner_id")
        settings = r.get("settings") or {}
        record_by_rid[rid_str] = (settings.get("wins", 0), settings.get("losses", 0))

    username_by_owner: Dict[str, Optional[str]] = {
        u["user_id"]: u.get("display_name") for u in users if "user_id" in u
    }

    # Cache avatars per owner_id, since avatar_from_users may scan the list
    avatar_cache: Dict[Optional[str], Any] = {}

    def get_avatar(owner_id: Optional[str]) -> Any:
        if owner_id not in avatar_cache:
            avatar_cache[owner_id] = avatar_from_users(users, owner_id) if owner_id is not None else None
        return avatar_cache[owner_id]

    by_mid: Dict[Any, List[dict]] = {}
    for m in mlist:
        mid = m.get("matchup_id")
        by_mid.setdefault(mid, []).append(m)

    def _from_players_map(pid: str) -> Dict[str, str]:
        if players_map:
            info = players_map.get(pid)
        else:
            info = None
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
        starters_raw = [s for s in (row.get("starters") or []) if s]
        pts_map = {str(k): v for k, v in (row.get("players_points") or {}).items()}
        s_infos: List[dict] = [_pinfo(str(pid), pts_map) for pid in starters_raw]
        pts_total = float(row["points"]) if isinstance(row.get("points"), (int, float)) else None

        wins, losses = record_by_rid.get(rid, (0, 0))
        owner_id = owner_id_by_rid.get(rid)
        username = username_by_owner.get(owner_id)

        return {
            "name": roster_map.get(rid, f"Roster {rid}"),
            "starters": s_infos,
            "pts_total": pts_total,
            "avatar": get_avatar(owner_id),
            "record": f"{wins}-{losses}",
            "username": username,
        }

    out = []
    for mid, rows in by_mid.items():
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: str(r.get("roster_id")))
        left = _team_block(rows_sorted[0])
        right = (
            _team_block(rows_sorted[1])
            if len(rows_sorted) > 1
            else {"name": "TBD", "avatar": None, "starters": [], "pts_total": None}
        )
        out.append({"matchup_id": mid, "left": left, "right": right})
    return out[:5]


def render_matchup_carousel_weeks(slides_by_week: dict[int, str], dashboard: bool) -> str:
    json_slides = json.dumps({str(k): v for k, v in slides_by_week.items()})
    # Call get_nfl_state once instead of inside the f-string
    nfl_state = get_nfl_state() or {}
    current_week = nfl_state.get("week")

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
            {slides_by_week.get(current_week, "<div class='m-empty'>No matchups</div>")}
          </div>
        </div>
      </div>
      <script>{carousel_script}</script>
    """


def add_bye_weeks_to_players():
    player_path = Path("cache/players_index.json")
    team_path = Path("cache/teams_index.json")
    with open(player_path, "r") as file:
        players = json.load(file)
    with open(team_path, "r") as file:
        teams = json.load(file)
    for player_id, player_data in players.items():
        team_abv = player_data.get("team")
        if team_abv in teams:
            player_data["byeWeek"] = teams[team_abv]["byeWeek"]

    write_json(player_path, players)


def team_live_totals(
    team: dict,
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

    starters = team.get("starters") or []
    for p in starters:
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
    roster_map: dict[str, str],  # roster_id -> owner
) -> dict[tuple[int, str], float]:
    """
    Returns {(week, roster_id): live_proj_total}
    """
    proj_by_roster: dict[tuple[int, str], float] = {}

    # build reverse map owner -> roster_id if needed
    owner_to_rid = {owner: rid for rid, owner in roster_map.items()}

    for week, matchups in matchups_by_week.items():
        # pull week-level containers once per week
        week_status_container = status_by_pid.get(week)
        week_proj_container = projections.get(week)

        # keep semantics the same: if these are missing or shaped differently,
        # team_live_totals will still see whatever comes through
        week_statuses = week_status_container.get("statuses") if isinstance(week_status_container, dict) else {}
        week_projections = week_proj_container.get("projections") if isinstance(week_proj_container, dict) else {}

        for m in matchups:
            for side in ("left", "right"):
                team = m[side]

                rid = team.get("roster_id")
                if rid is None:
                    rid = owner_to_rid.get(team["name"])
                if rid is None:
                    continue

                _, live_proj_total = team_live_totals(
                    team,
                    week_statuses,
                    week_projections,
                )
                proj_by_roster[(week, str(rid))] = live_proj_total

    return proj_by_roster


def build_team_schedule_lookup(games: List[dict]) -> Dict[str, dict]:
    """
    Given a list of game dicts from Tank01 getNFLGamesForWeek,
    build a lookup: team_abv -> that week's game dict.

    Each team appears at most once per week, so mapping is safe.
    """
    lookup: Dict[str, dict] = {}
    for g in games:
        home = (g["home"] or "").upper()
        away = (g["away"] or "").upper()

        if home:
            lookup[home] = g
        if away:
            lookup[away] = g

    return lookup


def parse_game_datetime(game_time_str: str) -> datetime:
    """
    Convert Tank01 game date/time into a real datetime object.

    game_date: "20251204"
    game_time_str: "8:15p" or "1:00a" (Tank01 style)
    """
    time_str = game_time_str.strip().lower()

    # Add missing "m"
    if time_str.endswith("a") or time_str.endswith("p"):
        time_str += "m"  # "8:15p" → "8:15pm"

    dt = datetime.strptime(f"{time_str}", "%I:%M%p")
    return dt


def build_defense_rankings(teams_index: dict) -> dict:
    """
    Returns a dictionary ranking all teams by defensive metrics:
      - opp_pass_yds_pg (lower = better)
      - opp_rush_yds_pg (lower = better)

    Output:
      {
        "ATL": {"opp_pass_yds_pg": 14, "opp_rush_yds_pg": 17},
        "DAL": {...},
        ...
      }
    """
    pass_list = []
    rush_list = []

    for abbr, info in teams_index.items():
        opp_pass = info.get("opp_pass_yds_pg")
        opp_rush = info.get("opp_rush_yds_pg")

        if opp_pass is not None:
            pass_list.append((abbr, float(opp_pass)))
        if opp_rush is not None:
            rush_list.append((abbr, float(opp_rush)))

    pass_sorted = sorted(pass_list, key=lambda x: x[1])
    rush_sorted = sorted(rush_list, key=lambda x: x[1])

    rankings = {abbr: {} for abbr in teams_index.keys()}

    for rank, (abbr, _) in enumerate(pass_sorted, start=1):
        rankings[abbr]["opp_pass_yds_pg"] = rank

    for rank, (abbr, _) in enumerate(rush_sorted, start=1):
        rankings[abbr]["opp_rush_yds_pg"] = rank

    return rankings


def format_player_stats(
    teams_stats: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    team: str,
    pos: str,
    player: str,
) -> Optional[str]:
    """
    Returns a compact stat line with no player name, e.g.:
      "4 rec, 61 yds, 1 td"
    Drops any 0-value TD lines.
    """
    team_data = teams_stats.get(team)
    if not team_data:
        return None

    pos_data = team_data.get(pos)
    if not pos_data:
        return None

    player_stats = pos_data.get(normalize_name(player))
    if not player_stats:
        return None

    def phrase(v: int, singular: str, plural: str) -> str:
        return f"{v} {singular if v == 1 else plural}"

    parts: list[str] = []

    if pos == "QB":
        py = player_stats.get("pass_yds", 0)
        ptd = player_stats.get("pass_td", 0)
        ints = player_stats.get("int", 0)
        ra = player_stats.get("rush_att", 0)
        ry = player_stats.get("rush_yds", 0)
        rtd = player_stats.get("rush_td", 0)

        if py:
            parts.append(phrase(py, "pass yd", "pass yds"))
        if ptd > 0:
            parts.append(phrase(ptd, "td", "tds"))
        if ints:
            parts.append(phrase(ints, "int", "ints"))
        if ra:
            parts.append(phrase(ra, "car", "car"))
        if ry:
            parts.append(phrase(ry, "rush yd", "rush yds"))
        if rtd > 0:
            parts.append(phrase(rtd, "rush td", "rush tds"))

    elif pos in {"RB", "WR", "TE"}:
        ra = player_stats.get("rush_att", 0)
        ry = player_stats.get("rush_yds", 0)
        rtd = player_stats.get("rush_td", 0)
        rec = player_stats.get("rec", 0)
        rec_yds = player_stats.get("rec_yds", 0)
        rec_td = player_stats.get("rec_td", 0)

        if rec:
            parts.append(phrase(rec, "rec", "rec"))
        if rec_yds:
            parts.append(phrase(rec_yds, "yd", "yds"))
        if rec_td > 0:
            parts.append(phrase(rec_td, "td", "tds"))

        if ra:
            parts.append(phrase(ra, "car", "car"))
        if ry:
            parts.append(phrase(ry, "yd", "yds"))
        if rtd > 0:
            parts.append(phrase(rtd, "td", "tds"))

    else:
        for k, v in player_stats.items():
            if isinstance(v, int) and v != 0:
                parts.append(f"{k}={v}")

    if not parts:
        return "no stats"

    return ", ".join(parts)


def render_matchup_slide(
    season: str,
    m: dict,
    w: int,
    proj_week: int,
    status_by_pid: dict[str, str],
    projections: dict[str, float],
    players: dict,
    teams: dict,
    team_game_lookup: dict,
) -> str:
    """One slide with rows like:
       [Left Name] [Left Pts/Proj] [Right Pts/Proj] [Right Name]
    """
    proj = w > proj_week
    defense_ranks = build_defense_rankings(load_teams_index())
    week_stats = load_week_stats(season, w)

    def team_head(t, proj_mode: bool):
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

        actual_total, live_proj_total = team_live_totals(
            t, status_by_pid, projections.get(w).get("projections")
        )

        return f"""
        <div class="m-team">
          {img}
          <div>
            <div class="name left">{t['name']}</div>
            <div>{t['record']} • @{t['username']}</div>
          </div>
          <div style="display:grid;grid-template-columns:1;justify-items: center;">
            <span class="num">{actual_total:.1f}</span>
            <span class="proj" style="opacity:0.4;text-align:center;">{live_proj_total:.1f}</span>
          </div>
        </div>
        """

    def team_head_2nd(t, proj_mode: bool):
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

        actual_total, live_proj_total = team_live_totals(
            t, status_by_pid, projections.get(w).get("projections")
        )
        return f"""
        <div class="m-team">
          <div style="display:grid;grid-template-columns:1;justify-items: center;">
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

    def format_team_game_line(team_abv: str, game: dict, pos: str) -> str:
        if not team_abv or not game:
            return ""

        home = str(game.get("home") or "").upper()
        away = str(game.get("away") or "").upper()
        t_up = team_abv.upper()
        if t_up not in (home, away):
            return ""

        is_home = (t_up == home)
        opp = away if is_home else home

        status_code = str(game.get("gameStatusCode") or "0")  # '0' scheduled, '1' live, '2' final
        game_date = str(game.get("gameDate") or "")  # '20251204'
        game_time = str(game.get("gameTime") or "")  # '8:15p'

        today_str = date.today().strftime("%Y%m%d")
        if game_date < today_str:
            status_code = "2"
        elif game_date == today_str:
            if parse_game_datetime(game_time) > datetime.now():
                status_code = "2"

        if status_code == "0":
            dow = ""
            if game_date:
                try:
                    dt = datetime.strptime(game_date, "%Y%m%d")
                    dow = dt.strftime("%a")
                except ValueError:
                    pass

            display_time = game_time
            if display_time.endswith("p"):
                display_time = display_time[:-1] + " pm"
            elif display_time.endswith("a"):
                display_time = display_time[:-1] + " am"

            opp_ranks = defense_ranks.get(opp, {})
            if pos in ["QB", "WR", "TE"]:
                opp_rank = opp_ranks.get("opp_pass_yds_pg")
            else:
                opp_rank = opp_ranks.get("opp_rush_yds_pg")

            prefix = "@ " + opp + (f" (#{opp_rank})" if opp_rank is not None else "") if not is_home else \
                     "vs " + opp + (f" (#{opp_rank})" if opp_rank is not None else "")
            return " ".join(x for x in [dow, display_time, prefix] if x).strip()

        line_score = game.get("lineScore") or {}
        period = line_score.get("period", "")
        clock = game.get("gameClock", "")
        extra = " ".join(x for x in [period, clock] if x).strip()

        score_str = ""

        if status_code == "1":
            game_date_std = datetime.strptime(game_date, "%Y%m%d").strftime("%Y%m%d")
            scores_body = get_nfl_scores_for_date(game_date_std)
            team_game_lookup_live = build_team_game_lookup(scores_body)
            game_live = team_game_lookup_live.get(team_abv)
            if is_home:
                my_pts = game_live.get("homePts")
                opp_pts = game_live.get("awayPts")
            else:
                my_pts = game_live.get("awayPts")
                opp_pts = game_live.get("homePts")

            if my_pts is not None and opp_pts is not None:
                score_str = f"{my_pts}-{opp_pts}"
            return f"<span class='live-dot'></span>{score_str} {extra}".strip()

        if status_code == "2":
            game_date_std = datetime.strptime(game_date, "%Y%m%d").strftime("%Y%m%d")
            scores_body = get_nfl_scores_for_date(game_date_std)
            team_game_lookup_live = build_team_game_lookup(scores_body)
            game_final = team_game_lookup_live.get(team_abv)
            prefix = "@ " + opp if not is_home else "vs " + opp
            if is_home:
                my_pts = game_final.get("homePts")
                opp_pts = game_final.get("awayPts")
            else:
                my_pts = game_final.get("awayPts")
                opp_pts = game_final.get("homePts")

            if my_pts is not None and opp_pts is not None:
                score_str = f"{my_pts}-{opp_pts}"

            if score_str:
                return f"Final {prefix} {score_str}"
            return "Final"

        return ""

    def player_bits(
        p,
        side: str,
        left_side: bool,
        team_schedule_lookup: dict[str, dict] | None = None,
    ):
        if not p:
            return "", 0.0, None, False, None

        pid = p.get("pid")
        name = p.get("name", "")
        nfl = p.get("nfl", "")
        pos = p.get("pos")

        actual = p.get("pts") or 0.0

        proj_val = None
        is_bye = False

        player_index = players.get(pid) or teams.get(pid)
        if player_index:
            proj_val = projections.get(w, {}).get("projections", {}).get(pid, 0.0)
            if proj_val == 0.0 and player_index.get("byeWeek") == w:
                is_bye = True

        status = status_by_pid.get(pid, STATUS_NOT_STARTED)
        if status == "BYE":
            is_bye = True

        if status == STATUS_NOT_STARTED and not is_bye:
            display_actual = 0.0
            display_proj = proj_val if proj_val is not None else 0.0
        elif status == STATUS_IN_PROGRESS and not is_bye:
            display_actual = actual
            display_proj = None
        elif status == STATUS_FINAL and not is_bye:
            display_actual = actual
            display_proj = None
        elif is_bye:
            display_actual = 0.0
            display_proj = None
        else:
            display_actual = 0.0 if actual is None else actual
            display_proj = proj_val if proj_val is not None else 0.0

        game_line = ""
        stats = None
        if nfl:
            team_code = str(nfl).upper()
            game = None

            if team_game_lookup:
                game = team_game_lookup.get(team_code)
                stats = format_player_stats(
                    week_stats,
                    team_code,
                    pos,
                    "ken walker" if name == "Kenneth Walker" else name,
                )

            if game is None and team_schedule_lookup:
                game = team_schedule_lookup.get(team_code)

            if game:
                game_line = format_team_game_line(team_code, game, pos)
                stats = format_player_stats(
                    week_stats,
                    nfl,
                    pos,
                    "ken walker" if name == "Kenneth Walker" else name,
                )

        meta_content = f"&nbsp;{nfl}"

        if left_side:
            if is_bye:
                cell = (
                    f"<div class='p {side}' style='opacity:0.4;'>"
                    f"<span class='pos-badge {pos}'>{pos}</span>"
                    f"<span class='pname'>{name}</span>"
                    f"<span class='meta'>{meta_content}</span>"
                    f"</div>"
                )
            else:
                cell = (
                    f"<div class='p {side}'>"
                    f"<span class='pos-badge {pos}'>{pos}</span>"
                    f"<div style='display: flex;flex-direction: column;'>"
                    f"<div><span class='pname'>{name} </span>"
                    f"<span class='meta'> {meta_content}</span></div>"
                    f"<span class='meta'>{game_line}</span></div>"
                    f"</div>"
                )
        else:
            if is_bye:
                cell = (
                    f"<div class='p {side}' style='justify-content:flex-end; opacity:0.4;'>"
                    f"<span class='meta'>{meta_content}</span>"
                    f"<span class='pname'>{name}</span>"
                    f"<span class='pos-badge {pos}'>{pos}</span>"
                    f"</div>"
                )
            else:
                cell = (
                    f"<div class='p {side}' style='justify-content:flex-end;'>"
                    f"<div style='display:flex;flex-direction:column-reverse;'>"
                    f"<span class='meta'>{game_line}</span>"
                    f"<div><span class='meta'>{meta_content} </span>"
                    f"<span class='pname'> {name}</span></div></div>"
                    f"<span class='pos-badge {pos}'>{pos}</span>"
                    f"</div>"
                )

        return cell, float(display_actual), display_proj, is_bye, (stats if stats else None)

    rows_html: List[str] = []

    team_schedule_lookup = build_team_schedule_lookup(load_week_schedule(season, w))

    for L, R in zip_longest(
        m["left"].get("starters", []),
        m["right"].get("starters", []),
        fillvalue=None,
    ):
        left_cell, left_actual, left_proj, left_is_bye, left_stats = player_bits(
            L, "left", True, team_schedule_lookup
        )
        right_cell, right_actual, right_proj, right_is_bye, right_stats = player_bits(
            R, "right", False, team_schedule_lookup
        )

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

        def stat_stack(stats, side: str) -> str:
            if stats is None:
                return "<div></div>"
            if side == "left":
                return (
                    "<div class='p right' style='display: grid;'>"
                    f"<span class='meta' style='display:flex;justify-content:flex-end'>{stats}</span></div>"
                )
            return (
                "<div class='p left'>"
                f"<span class='meta'>{stats}</span></div>"
            )

        left_points_html = score_stack(left_actual, left_proj, "l", left_is_bye, left_more)
        right_points_html = score_stack(right_actual, right_proj, "r", right_is_bye, right_more)
        points = f"{left_points_html}{right_points_html}"

        rows_html.append(
            f"""<div class="m-row">
                  {left_cell}
                  {stat_stack(left_stats, "left")}
                  {points}
                  {stat_stack(right_stats, "right")}
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
