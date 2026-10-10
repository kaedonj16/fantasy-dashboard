"""Teams page builder (deep team analytics: roster grades, intel, archetypes,
value trends / Beat the Market, and in-season schedule difficulty).

Playoff odds / Power Rankings live on Standings; draft grades live in Draft
Room / Draft History -- not as Teams sidebar tabs.

Moved verbatim from app.py to shrink the monolith. The heavy app.py internals it
uses are lazy-imported from app inside the function (resolved at request time),
so importing this module at start-up never triggers a circular import.
"""
import html
import json
import logging
import math
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Union

from utils.pick_slots import pick_label as _pk_pick_label
from utils.pick_slots import pick_value_from_table as _pk_pick_value_from_table

logger = logging.getLogger(__name__)


def roster_shape_label(pos_vals: Dict[str, List[float]], is_sf: bool) -> str:
    """Descriptive roster-construction archetype from positional value shares.

    The draft room labels a *draft* by pick order; a standing roster has no
    draft order, so this classifies the same recognizable shapes (WR Factory,
    Hero RB, Zero RB, ...) from where a team's dynasty value actually sits.
    Purely descriptive - it names the build, it does not grade it.
    """
    rb = sorted(pos_vals.get("RB", []), reverse=True)
    qbv = sum(pos_vals.get("QB", []))
    rbv = sum(rb)
    wrv = sum(pos_vals.get("WR", []))
    tev = sum(pos_vals.get("TE", []))
    total = qbv + rbv + wrv + tev
    if total <= 0:
        return ""
    qs, rs, ws, ts = qbv / total, rbv / total, wrv / total, tev / total
    top_rb_share = (rb[0] / rbv) if rbv > 0 and rb else 0.0   # concentration in the RB room
    top_rb_of_total = (rb[0] / total) if rb else 0.0          # is that back an elite anchor
    # Ordered specific -> generic; first match wins.
    if is_sf and qs >= 0.28:
        return "Konami Code"
    # TE Premium: TE isn't merely present, it's a genuine strength -- the TE room
    # holds a high share of value AND at least matches the RB room. This stops an
    # RB-heavy team that happens to own one elite TE (e.g. Bowers) from reading as
    # "TE Premium" when it's really a Robust RB build.
    if ts >= 0.18 and ts >= rs:
        return "TE Premium"
    if rs <= 0.15 and ws >= 0.38:
        return "Zero RB"
    # Hero RB: one elite back carries a thin RB room, with a WR-forward rest.
    if top_rb_share >= 0.55 and top_rb_of_total >= 0.24 and ws >= rs:
        return "Hero RB"
    if ws >= 0.45:
        return "WR Factory"
    if rs >= 0.38:
        return "Robust RB"
    return "Balanced"


_DW_SUFFIX_RE = re.compile(r"^(II|III|IV|V|Jr|Sr)$", re.IGNORECASE)


def _dw_short_parts(name):
    """Name words with generational suffixes stripped (same regex approach as
    PR #2442's JS _sosLastName: pop trailing II/III/IV/V/Jr/Sr tokens, dots
    optional, so "Kenneth Walker III" shortens to "Walker", not "III")."""
    parts = str(name or "").strip().split()
    while len(parts) > 1 and _DW_SUFFIX_RE.match(parts[-1].replace(".", "")):
        parts.pop()
    return parts


_dw_players_index_cache = None


def _dw_players_index():
    """Module-level lazy players_index (espnHeadshot URLs for drawer rows)."""
    global _dw_players_index_cache
    if _dw_players_index_cache is None:
        try:
            from utils.utils import load_players_index
            _dw_players_index_cache = load_players_index() or {}
        except Exception:
            _dw_players_index_cache = {}
    return _dw_players_index_cache


def _dw_hires_headshot(url, width=128):
    """Mirror of app.js _hiResHeadshot: route ESPN headshots through the
    combiner at the requested width so drawer rows match the player modal."""
    if not url:
        return ""
    if "/combiner/" in url:
        return url
    m = re.search(r"espncdn\.com(/i/headshots/[^?]+\.(?:png|jpg|jpeg))", url, re.I)
    if not m:
        return url
    return f"https://a.espncdn.com/combiner/i?img={m.group(1)}&w={width}&scale=crop&cquality=100"


def build_teams_body(ctx: dict) -> str:
    """
    Teams page:
      - One card per team
      - Within each card:
          * positional strength table (value + z-score + bar)
          * each position row can expand to show that position's players + values
      - Positional Index summary per team in header
    """
    from app import (  # noqa: E402  (lazy: avoids a circular import at module load)
        _playoff_sim_cached, _safe_int, _team_pick_value,
        team_avatar,
        build_historical_pick_slot_map, count_roster_positions, get_roster_positions,
        has_draft_ended, load_pick_value_table, _TEAMS_JS_V,
        _TEAMS_JS_FILE,
        _league_is_redraft,
    )
    from dashboard_services.ai.context_builders import (
        ctx_scoring_type, league_format_value_lookup, redraft_window_label,
    )
    rosters = ctx["rosters"]  # Sleeper /rosters
    roster_map = ctx["roster_map"]  # mapping roster_id -> team name
    users = ctx["users"]
    platform = ctx["platform"]
    picks_by_roster = ctx.get("picks_by_roster") or {}
    league_id = str(ctx.get("league_id") or "")
    current_season = _safe_int((ctx.get("league") or {}).get("season"), datetime.now().year)
    _is_redraft = bool(_league_is_redraft(ctx) or ctx_scoring_type(ctx) == "redraft")
    # Redraft leagues and hosts without a pick feed do not invent future picks.
    if _is_redraft or not ctx.get("draft_capital_available", True):
        picks_by_roster = {}

    # Projected draft slots for next year's picks, from projected final
    # standings this season (fewest average final wins picks first). Feeds the
    # expandable PICKS detail row on each team card.
    _pk_proj_year = current_season + 1
    _pk_slot_by_original: dict = {}
    _pk_final_slots: dict = {}
    _pk_value_tbl: dict = {}
    # Exact slots for the upcoming draft: its order is already cemented by
    # last season's final standings (same source _team_pick_value uses).
    try:
        _pk_final_slots = build_historical_pick_slot_map(
            platform=platform,
            root_league_id=league_id,
            current_season=current_season,
            source_season=current_season - 1,
        ) or {}
    except Exception:
        logger.debug("teams: final pick slots failed", exc_info=True)
    _pk_odds = []
    try:
        _pk_odds = _playoff_sim_cached(ctx, platform, block=False) or []
        if _pk_odds:
            _pk_order = sorted(
                _pk_odds,
                key=lambda r: (
                    float(r.get("avg_final_wins") or r.get("wins") or 0),
                    float(r.get("playoff_pct") or 0),
                ),
            )
            _pk_slot_by_original = {
                str(r.get("roster_id")): i + 1 for i, r in enumerate(_pk_order)
            }
    except Exception:
        logger.debug("teams: pick slot projection failed", exc_info=True)
    try:
        from dashboard_services.picks import load_pick_value_table as _lpvt_teams
        _pk_value_tbl = dict(_lpvt_teams(league_teams=len(rosters) or 10) or {})
    except Exception:
        logger.debug("teams: pick value table failed", exc_info=True)
    
    viewer = ctx.get("viewer") or {}
    viewer_roster_id = viewer.get("viewer_roster_id")

    # ----------------- Load value table -----------------
    # Expected rows like {id, name, position, team, value, search_name}
    model_vals = ctx.get("model_value_table") or []

    # map sleeper_id -> row. League-type values (1QB/SF, size, redraft) + TE
    # premium come from the shared lookup so My Leagues and this page rank the
    # same rooms the same way.
    _rp_early = ctx.get("roster_positions") or []
    from utils.lineup_slots import is_superflex_lineup
    from utils.value_helpers import format_rank_label_key, row_format_rank_label
    _is_sf_early = is_superflex_lineup(_rp_early)
    _scoring = "redraft" if _is_redraft else "dynasty"
    # Pos ranks like RB23 must match the league format (redraft vs dynasty, SF vs
    # 1QB). Hardcoding dynasty pos_rank_label leaked dynasty ranks into redraft.
    _rank_label_key = format_rank_label_key(is_redraft=_is_redraft, is_sf=_is_sf_early)
    by_id: Dict[str, Dict] = league_format_value_lookup(ctx)

    name_to_rank_label: Dict[str, str] = {}
    name_to_age: Dict[str, Union[float, None]] = {}

    for obj in model_vals:
        if not isinstance(obj, dict):
            continue
        safe_name = str(obj.get("search_name") or "").strip().lower()
        if not safe_name:
            continue
        pos_lbl = (
            row_format_rank_label(obj, _rank_label_key)
            or obj.get("position")
            or obj.get("pos")
            or ""
        )
        name_to_rank_label[safe_name] = str(pos_lbl)
        age_val = obj.get("age")
        if age_val is not None:
            try:
                name_to_age[safe_name] = float(age_val)
            except Exception:
                name_to_age[safe_name] = None

    CORE_POS = {"QB", "RB", "WR", "TE"}
    POS_ORDER = ["QB", "RB", "WR", "TE"]

    # ----------------- Roster → position → players (for dropdowns) -----------------
    roster_pos_players: Dict[int, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

    for r in rosters:
        rid = r.get("roster_id")
        if rid is None:
            continue
        try:
            rid_int = int(rid)
        except Exception:
            continue

        for pid in (r.get("players") or []):
            p = by_id.get(str(pid))
            if not p:
                continue
            pos = str(p.get("position") or p.get("pos") or "").upper()
            if pos == "PICK":
                continue
            if pos not in CORE_POS:
                continue  # only core positions in dropdown

            roster_pos_players[rid_int][pos].append(p)

    # sort each position bucket by value (high → low)
    for rid, pos_map in roster_pos_players.items():
        for pos, plist in pos_map.items():
            plist.sort(key=lambda x: float(x.get("value", 0.0)), reverse=True)

    # value-weighted average age per team per position (top 8 players)
    team_pos_age: Dict[int, Dict[str, Optional[float]]] = defaultdict(dict)
    for _rid, _pos_map in roster_pos_players.items():
        for _pos in POS_ORDER:
            _plist = _pos_map.get(_pos, [])
            _age_vals = []
            for _p in _plist[:8]:
                _nm = str(_p.get("search_name") or "").strip().lower()
                _a = name_to_age.get(_nm)
                _v = float(_p.get("value") or 0)
                if _a is not None and _v > 0:
                    _age_vals.append((_a, _v))
            if _age_vals:
                _tv = sum(v for _, v in _age_vals)
                team_pos_age[_rid][_pos] = round(sum(a * v for a, v in _age_vals) / _tv, 1)
            else:
                team_pos_age[_rid][_pos] = None

    # ----------------- Build per-team position value buckets (for strength table) -----------------
    team_meta: Dict[int, Dict] = {}  # name, avatar
    team_pos_values: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for r in rosters:
        rid = r.get("roster_id")
        if rid is None:
            continue

        display_name = roster_map.get(str(rid)) if isinstance(roster_map, dict) else str(rid)
        avatar = team_avatar(platform, r, users)
        team_meta[rid] = {
            "name": display_name,
            "avatar": avatar,
        }

        for pid in (r.get("players") or []):
            row = by_id.get(str(pid))
            if not row:
                continue
            pos = str(row.get("position") or row.get("pos") or "").upper()
            try:
                val = float(row.get("value") or 0.0)
            except Exception:
                val = 0.0
            if val <= 0:
                continue
            team_pos_values[rid][pos].append(val)

    # ensure every team has all core pos keys for the table
    for rid in team_meta.keys():
        for pos in POS_ORDER:
            team_pos_values[rid].setdefault(pos, [])

    # ----------------- Compute per-team draft capital value -----------------
    pick_by_key: Dict[str, float] = load_pick_value_table() or {}
    team_pick_value: Dict[int, float] = {}
    for r in rosters:
        rid = r.get("roster_id")
        if rid is None:
            continue
        team_pick_value[int(rid)] = _team_pick_value(
            picks_by_roster.get(str(rid), []), pick_by_key,
            platform=platform, league_id=league_id, season=current_season,
        )

    # ----------------- Compute per-team positional strength + league baselines -----------------
    # Rank uses weighted_pos_strength (same helper as My Leagues); the
    # per-position ranks feed the drawer payload below.
    from utils.roster_strength import rank_rosters_by_position
    slot_counts = count_roster_positions(
        ctx.get("roster_positions") or get_roster_positions() or []
    )
    team_pos_strength, pos_rank = rank_rosters_by_position(
        team_pos_values, slot_counts, positions=POS_ORDER,
    )

    league_pos_avg: Dict[str, float] = {}
    league_pos_std: Dict[str, float] = {}

    for pos in POS_ORDER:
        series = [team_pos_strength[rid][pos] for rid in team_meta.keys()]
        if not series:
            league_pos_avg[pos] = 0.0
            league_pos_std[pos] = 0.0
            continue
        mean = sum(series) / len(series)
        var = sum((x - mean) ** 2 for x in series) / len(series)
        std = math.sqrt(var)
        league_pos_avg[pos] = mean
        league_pos_std[pos] = std

    # ----------------- Z-scores & positional index -----------------
    team_pos_index: Dict[int, float] = {}

    LINEUP_WEIGHTS = {
        "QB": slot_counts.get("QB") or 1,
        "RB": slot_counts.get("RB") or 2,
        "WR": slot_counts.get("WR") or 2,
        "TE": slot_counts.get("TE") or 1,
        "FLEX": slot_counts.get("FLEX") or 1,
    }
    weight_sum = sum(LINEUP_WEIGHTS[pos] for pos in POS_ORDER if LINEUP_WEIGHTS.get(pos, 0) > 0) or 1.0

    for rid in team_meta.keys():
        idx_num = 0.0

        for pos in POS_ORDER:
            team_strength = team_pos_strength[rid][pos]
            mu = league_pos_avg[pos]
            sigma = league_pos_std[pos]
            if sigma > 0:
                z = (team_strength - mu) / sigma
            else:
                z = 0.0

            w = LINEUP_WEIGHTS.get(pos, 0)
            idx_num += w * z

        team_pos_index[rid] = idx_num / weight_sum

    # ----------------- Helper: players under a position row -----------------
    def render_pos_players(rid: int, pos_code: str) -> str:
        plist = roster_pos_players.get(rid, {}).get(pos_code, [])
        if not plist:
            return "<div style='color:#64748b;font-size:13px;'>No players at this position.</div>"

        chip = _POS_CHIP.get(pos_code, "#64748b")
        index = _dw_players_index()

        rows_html = []
        for p in plist:
            name = str(p.get("name") or "")
            name_raw = p.get('search_name', '')
            name_key = str(name_raw or "").strip().lower()

            rank_label = name_to_rank_label.get(
                name_key,
                p.get('position', '')
            )
            age = name_to_age.get(name_key)
            age_txt = f"{age:.1f} yrs" if age is not None else ""

            try:
                val = float(p.get("value") or 0.0)
            except Exception:
                val = 0.0
            val_txt = f"{val:.1f}" if val > 0 else ""

            # Short display-name pieces: initials come from the first two
            # suffix-stripped words; the title uses the short (suffix-stripped)
            # last name, e.g. "Walker" for "Kenneth Walker III".
            short_parts = _dw_short_parts(name)
            initials = "".join(w[0] for w in short_parts[:2]).upper()
            short_name = short_parts[-1] if short_parts else ""

            # ESPN combiner headshot at 128px, exactly like the player modal
            # (_hiResHeadshot); tapping a row opens the modal, so the photo
            # must not visibly change. Omitted when unknown (initials show).
            player_id = str(p.get("id", ""))
            raw_hs = str((index.get(player_id) or {}).get("espnHeadshot") or "").strip()
            hs_url = _dw_hires_headshot(raw_hs, 128)
            if hs_url:
                headshot = (
                    '<img src="'
                    + html.escape(hs_url)
                    + '" alt="" loading="lazy" decoding="async" onerror="this.remove()">'
                )
            else:
                headshot = ""

            # Build meta parts (rank, team, age)
            meta_parts = [str(rank_label or ""), str(p.get('team') or "")]
            if age_txt:
                meta_parts.append(age_txt)
            meta_str = " · ".join(filter(None, meta_parts))

            position = p.get('position', '')
            years_exp = p.get('years_exp')
            team_abbr = str(p.get('team') or "")
            rows_html.append(
                '<div class="td-prow">'
                f'  <span class="td-hs" style="--ring:{html.escape(chip)};" title="{html.escape(short_name)} · {html.escape(str(rank_label or ""))}">'
                f'    <span class="td-hs-init">{html.escape(initials)}</span>'
                f'    {headshot}'
                f'    <span class="td-hs-tm">{html.escape(team_abbr)}</span>'
                '  </span>'
                '  <div class="td-pinfo">'
                f'    <div class="td-pnm player-clickable" data-player-id="{html.escape(player_id)}" data-player-name="{html.escape(name)}" data-position="{html.escape(str(position))}" data-years-exp="{html.escape(str(years_exp))}" data-value="{html.escape(str(val))}" data-breakout-check="true">{html.escape(name)}</div>'
                f'    <div class="td-pmeta">{html.escape(meta_str)}</div>'
                '  </div>'
                '  <div class="td-pstat">'
                f'    <div class="td-pval">{html.escape(val_txt)}</div>'
                '    <span class="td-tag-slot"></span>'
                '  </div>'
                '</div>'
            )

        return "".join(rows_html)

    # Pre-compute roster grades for all teams
    from dashboard_services.ai.context_builders import calculate_roster_grade as _calc_grade

    _n_teams = len(team_meta)
    _offseason = ctx.get("offseason_mode", False)
    _rp_list = ctx.get("roster_positions") or []
    from utils.lineup_slots import is_superflex_lineup
    _is_sf = is_superflex_lineup(_rp_list)
    _redraft_key = "redraft_value_sf" if _is_sf else "redraft_value_1qb"

    # ── Compute dynasty totals, redraft totals, and dynasty/redraft ratios per team ──
    _team_dynasty_total: Dict[int, float] = {}
    _team_redraft_total: Dict[int, float] = {}
    _team_dr_ratio: Dict[int, float] = {}
    for _r in rosters:
        _rid = _r.get("roster_id")
        if _rid is None:
            continue
        _pairs: List[tuple] = []
        for _pid in (_r.get("players") or []):
            _row = by_id.get(str(_pid))
            if not _row:
                continue
            _pos = str(_row.get("position") or "").upper()
            if _pos not in CORE_POS:
                continue
            _dval = float(_row.get("value") or 0)
            _rval = float(_row.get(_redraft_key) or 0)
            _pairs.append((_dval, _rval))
        # Sort by dynasty value to get consistent top-8
        _pairs.sort(reverse=True)
        _team_dynasty_total[_rid] = sum(d for d, _ in _pairs[:8])
        _team_redraft_total[_rid] = sum(rv for _, rv in _pairs[:8])
        _ratios = [d / max(rv, 1) for d, rv in _pairs[:10] if d > 50 or rv > 50]
        _team_dr_ratio[_rid] = round(sum(_ratios) / len(_ratios), 3) if _ratios else 1.0

    # ── Percentile helpers ──
    def _make_pct_fn(totals: Dict[int, float]):
        _sorted = sorted(totals.values())
        _n = max(len(_sorted) - 1, 1)
        def _pct(rid: int) -> float:
            t = totals.get(rid, 0.0)
            return sum(1 for v in _sorted if v < t) / _n
        return _pct

    _dynasty_pct = _make_pct_fn(_team_dynasty_total)
    _redraft_pct = _make_pct_fn(_team_redraft_total)

    def _grade_for_roster(r_id: int) -> dict:
        roster_obj = next((r for r in rosters if r.get("roster_id") == r_id), {})
        flat_players = []
        for pid in roster_obj.get("players") or []:
            row = by_id.get(str(pid))
            if not row:
                continue
            pos = str(row.get("position") or row.get("pos") or "").upper()
            if pos not in CORE_POS:
                continue
            val = float(row.get("value") or 0.0)
            nm = str(row.get("name") or "").strip().lower()
            age = name_to_age.get(nm)
            flat_players.append({"position": pos, "value": val, "age": age})
        flat_players.sort(key=lambda x: x["value"], reverse=True)
        picks = picks_by_roster.get(str(r_id), [])
        p_ranks = {pos: pos_rank[pos].get(r_id, _n_teams) for pos in POS_ORDER}
        return _calc_grade(
            flat_players, picks,
            position_ranks=p_ranks,
            num_teams=_n_teams,
            dynasty_pct_val=_dynasty_pct(r_id),
            redraft_pct_val=_redraft_pct(r_id),
            dr_ratio=_team_dr_ratio.get(r_id, 1.0),
            scoring_type=_scoring,
        )

    team_grades = {rid: _grade_for_roster(rid) for rid in team_meta}
    if _is_redraft:
        _po_by_rid = {
            str((row or {}).get("roster_id")): (row or {}).get("playoff_pct")
            for row in (_pk_odds or [])
        }
        for _rid, _g in team_grades.items():
            _g["win_window"] = redraft_window_label(
                playoff_pct=_po_by_rid.get(str(_rid)),
                redraft_pct=_redraft_pct(_rid),
            )

    # ----------------- Build HTML cards -----------------
    cards_html = []
    _drawer_data = {}

    # Competitive-window accent colors (mirror the window legend below) and the
    # per-position chip colors used across the reworked cards.
    _WINDOW_COLORS = {
        "Contend": "#22c55e", "Bubble": "#f59e0b", "Out": "#94a3b8",
        "Contender": "#22c55e", "Win-Now": "#f59e0b", "Aging Contender": "#84cc16",
        "Contender Window": "#3b82f6", "2-3 Year Window": "#6366f1", "Rising": "#8b5cf6",
        "Holding Pattern": "#94a3b8", "Retooling": "#f97316", "Rebuilding": "#ef4444",
        "Full Rebuild": "#b91c1c",
    }
    _POS_CHIP = {"QB": "#3b82f6", "RB": "#22c55e", "WR": "#f59e0b", "TE": "#8b5cf6"}

    # League-best positional totals for the drawer's vs-league strength bars
    # (fill width = team total / best in league; tick = league average).
    _dw_pos_max = {}
    for _p in POS_ORDER:
        _totals = [sum(team_pos_values[_rid].get(_p, [])) for _rid in team_meta.keys()]
        _dw_pos_max[_p] = round(max(_totals, default=0.0), 1)

    for _card_idx, (rid, meta) in enumerate(team_meta.items()):
        name = meta["name"]
        avatar = meta.get("avatar") or ""
        img_html = (
            f"<img class='avatar' src='{avatar}' alt='' loading='lazy' decoding='async' onerror=\"this.style.display='none'\">"
            if avatar else ""
        )

        # Positional detail for the team drawer: per-position aggregates plus the
        # server-rendered player rows (render_pos_players). The expandable
        # in-card position table was removed; the drawer is the detail surface.
        _drawer_pos = []
        for pos in POS_ORDER:
            vals = team_pos_values[rid][pos]
            count = len(vals)
            total = sum(vals)

            rank = pos_rank[pos].get(rid, 0)
            _pos_age = team_pos_age.get(int(rid), {}).get(pos)
            _age_txt = f"{_pos_age:.1f}" if _pos_age is not None else ""

            _drawer_pos.append({
                "pos": pos,
                "chip": _POS_CHIP.get(pos, "#64748b"),
                "total": round(total, 1),
                "pos_max": _dw_pos_max.get(pos, 0.0),
                "pos_avg": round(league_pos_avg.get(pos, 0.0), 1),
                "count": count,
                "age": _age_txt,
                "rank": rank,
                "num_teams": _n_teams,
                "strength": "STRONG" if rank and rank <= max(1, _n_teams // 3) else "",
                "players_html": render_pos_players(rid, pos),
            })


        # ── Value-by-position mix bar (a compact stacked bar + legend that
        # replaces the per-card Plotly chart: same figures, a fraction of the
        # height and no chart dependency). "Picks" reads as CAP (draft capital).
        _mix_labels = ["QB", "RB", "WR", "TE"]
        _mix_colors = ["#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6"]
        _mix_values = [
            round(sum(team_pos_values[rid].get("QB", [])), 1),
            round(sum(team_pos_values[rid].get("RB", [])), 1),
            round(sum(team_pos_values[rid].get("WR", [])), 1),
            round(sum(team_pos_values[rid].get("TE", [])), 1),
        ]
        if not _is_redraft and ctx.get("draft_capital_available", True):
            _mix_labels.append("CAP")
            _mix_colors.append("#c92c68")
            _mix_values.append(round(team_pick_value.get(rid, 0.0), 1))
        _mix_total = sum(_mix_values)
        _mix_segs = "".join(
            f"<div class='tsc-mix-seg' style='flex:{v:.2f};background:{c};'></div>"
            for v, c in zip(_mix_values, _mix_colors) if v and v > 0
        )
        _mix_legend = " ".join(
            f"<span class='tsc-mix-leg'><span class='tsc-mix-dot' style='background:{c};'></span>"
            f"<span class='tsc-mix-k'>{lbl}</span> {v:,.0f}</span>"
            for lbl, v, c in zip(_mix_labels, _mix_values, _mix_colors)
        )
        _mix_html = (
            "<div class='tsc-mix'>"
            "  <div class='tsc-mix-head'>"
            "    <span class='tsc-mix-lbl'>Value by position</span>"
            f"    <span class='tsc-mix-total'>{_mix_total:,.1f}</span>"
            "  </div>"
            f"  <div class='tsc-mix-bar'>{_mix_segs}</div>"
            f"  <div class='tsc-mix-legend'>{_mix_legend}</div>"
            "</div>"
        )

        _gdata = team_grades.get(rid, {})
        _grade = _gdata.get("grade", "?")
        _win_window = _gdata.get("win_window", "")
        _grade_cls = "grade-a" if _grade.startswith("A") else "grade-b" if _grade.startswith("B") else "grade-c" if _grade.startswith("C") else "grade-d"
        _grade_chip = (
            f"<span class='tsc-grade {_grade_cls}' title='{html.escape(_win_window)}'>{_grade}</span>"
            if _grade and _grade != "?" else ""
        )

        # Numeric sort keys for client-side sorting
        _grade_num = {"A+":12,"A":11,"A-":10,"B+":9,"B":8,"B-":7,"C+":6,"C":5,"C-":4,"D+":3,"D":2,"D-":1,"F":0}.get(_grade, 0)
        _archetype_num = {
            "Contend":          1,
            "Bubble":           2,
            "Long Shot":        3,
            "Contender":        1,
            "Win-Now":          2,
            "Aging Contender":  3,
            "Contender Window": 4,
            "2-3 Year Window":  5,
            "Rising":           6,
            "Holding Pattern":  7,
            "Retooling":        8,
            "Rebuilding":       9,
            "Full Rebuild":     10,
        }.get(_win_window, 7)
        _window_cls = {
            "Contend":          "wt-contend",
            "Bubble":           "wt-bubble",
            "Long Shot":        "wt-long-shot",
            "Contender":        "wt-contender",
            "Win-Now":          "wt-win-now",
            "Aging Contender":  "wt-aging-contender",
            "Contender Window": "wt-contender-window",
            "2-3 Year Window":  "wt-2yr",
            "Rising":           "wt-rising",
            "Holding Pattern":  "wt-holding",
            "Retooling":        "wt-retooling",
            "Rebuilding":       "wt-rebuilding",
            "Full Rebuild":     "wt-full-rebuild",
        }.get(_win_window, "wt-holding")
        _pos_idx = team_pos_index[rid]
        _is_viewer = str(rid) == str(viewer_roster_id or "")

        # ── Compact card chrome (Mock 4) ──────────────────────────────────────
        _win_color = _WINDOW_COLORS.get(_win_window, "#94a3b8")
        _initials = ("".join(w[0] for w in str(name).split()[:2]).upper() or "?")[:2]
        if img_html:
            _avatar_html = (
                "<span class='tsc-avatar-wrap'>"
                f"<img class='tsc-avatar' src='{avatar}' alt='' loading='lazy' decoding='async' "
                "onerror=\"this.style.visibility='hidden'\">"
                "</span>"
            )
        else:
            _avatar_html = (
                f"<span class='tsc-avatar-wrap tsc-avatar-mono'>{html.escape(_initials)}</span>"
            )
        _you_pill = "<span class='tsc-you'>YOU</span>" if _is_viewer else ""
        _status_label = html.escape(_win_window) if _win_window else "Unranked"

        # Positional Index: diverging bar around the league average (z = 0).
        # Capped at +/-2 sigma, which fills a full half-track (Mock 4 style).
        _pi_dir = "up" if _pos_idx >= 0 else "dn"
        _pi_w = min(abs(_pos_idx) / 2.0, 1.0) * 50.0
        _pi_left = 50.0 if _pi_dir == "up" else 50.0 - _pi_w
        _index_html = (
            "<div class='tsc-pi' title='Positional Index: how far this team&apos;s starting-lineup "
            "strength sits above or below the league average, in standard deviations.'>"
            "  <div class='tsc-pi-head'>"
            "    <span class='tsc-pi-lbl'>Positional Index</span>"
            f"    <span class='tsc-pi-num {_pi_dir}'>{_pos_idx:+.2f}</span>"
            "  </div>"
            "  <div class='tsc-pi-track'>"
            "    <span class='tsc-pi-mid'></span>"
            f"    <span class='tsc-pi-fill {_pi_dir}' style='left:{_pi_left:.0f}%;width:{_pi_w:.0f}%;'></span>"
            "  </div>"
            "</div>"
        )

        card_html = (
            f"<div class='card team-strength-card {_window_cls}' data-br-moment='draftgrade' data-sort-grade='{_grade_num}' data-sort-posindex='{_pos_idx:.4f}' data-sort-archetype='{_archetype_num}' data-roster-id='{rid}' data-original-index='{_card_idx}'" + (" data-viewer='1'" if _is_viewer else "") + ">"
            "  <div class='tsc-head'>"
            f"    {_avatar_html}"
            "    <div class='tsc-idtext'>"
            f"      <div class='tsc-namerow'><span class='tsc-name'>{html.escape(str(name))}</span>{_you_pill}</div>"
            f"      <div class='tsc-status'><span class='tsc-dot' style='background:{_win_color};'></span>{_status_label}</div>"
            "    </div>"
            f"    {_grade_chip}"
            "  </div>"
            f"  {_index_html}"
            f"  {_mix_html}"
            f"  <button class='tsc-details' type='button' data-roster-id='{rid}'>View details <span aria-hidden='true'>&rarr;</span></button>"
            "</div>"
        )

        cards_html.append(card_html)

        # Drawer payload: everything the team drawer renders without a fetch.
        _drawer_data[str(rid)] = {
            "name": str(name),
            "avatar": avatar,
            "initials": _initials,
            "is_viewer": _is_viewer,
            "grade": _grade,
            "grade_cls": _grade_cls,
            "window": _win_window,
            "win_color": _win_color,
            "pos_index": f"{_pos_idx:+.2f}",
            "pos_index_dir": _pi_dir,
            "positions": _drawer_pos,
        }

    all_cards_html = "".join(
        cards_html) or "<div class='card'><div class='card-body'><p>No teams found.</p></div></div>"
    # Drawer data for teams.js: per-team header + positional detail.
    _drawer_json = json.dumps(_drawer_data).replace("</", "<\\/")

    # ---------- League analytics section (lazy-loaded) ----------
    platform_js = platform
    season_js = current_season

    # Detect league type (sf vs 1qb) from this league's roster positions
    _rp_list = list(ctx.get("roster_positions") or get_roster_positions() or [])
    from utils.lineup_slots import is_superflex_lineup
    _is_sf = is_superflex_lineup(_rp_list)
    _league_type_js = "sf" if _is_sf else "1qb"
    _league_size_js = int(len(rosters)) if rosters else 10

    _offseason_mode_js = bool(ctx.get("offseason_mode", False))
    _draft_ended_js = has_draft_ended(league_id, platform, current_season)
    # Teams analytics JS moved to static/teams.js; pass its inputs as JSON.
    _teams_cfg_json = json.dumps({
        "platform": platform_js,
        "leagueId": league_id,
        "season": season_js,
        "leagueType": _league_type_js,
        "leagueSize": _league_size_js,
        "viewerRosterId": str(viewer_roster_id or ""),
        "offseasonMode": _offseason_mode_js,
        "draftEnded": _draft_ended_js,
    })

    # Skeleton markup reused by the lazy-loaded analytics panels.
    _analytics_skeleton = (
        '<div class="analytics-skeleton"><div class="sk-shimmer sk-line" style="width:60%"></div>'
        '<div class="sk-shimmer sk-line sk-line--w75" style="margin-top:10px"></div>'
        '<div class="sk-shimmer sk-line sk-line--w50" style="margin-top:10px"></div>'
        '<div class="sk-shimmer sk-line sk-line--w60" style="margin-top:10px"></div></div>'
    )

    _teams_foot_scripts = f"""
    <script>window.__teamsCfg = {_teams_cfg_json};</script>
    <script src="/static/{_TEAMS_JS_FILE}?v={_TEAMS_JS_V}" defer></script>
    """

    # ---------- Window legend ----------
    if _is_redraft:
        _window_section = """
          <div class="wl-section-label" style="margin-top:10px;">This season</div>
          <div class="wl-row"><span class="wl-dot" style="background:#22c55e;"></span><strong class="wl-label">Contend</strong><span class="wl-desc">Playoff favorite &mdash; 70%+ odds to make the playoffs</span></div>
          <div class="wl-row"><span class="wl-dot" style="background:#f59e0b;"></span><strong class="wl-label">Bubble</strong><span class="wl-desc">Live but not locked &mdash; 35&ndash;70% playoff odds</span></div>
          <div class="wl-row"><span class="wl-dot" style="background:#94a3b8;"></span><strong class="wl-label">Long Shot</strong><span class="wl-desc">Under 35% playoff odds &mdash; still alive, not mathematically eliminated</span></div>
          <div class="wl-grade-note">This label is playoff odds, not the letter grade. A clean draft can still sit mid-pack.</div>
        """
        _grade_note = "Grade is this-season roster construction (starters + value). It is not playoff odds."
        _grade_a_desc = "Elite this-season roster: top starter rooms and projected scoring"
        _grade_d_desc = "Weak this-season roster: thin starters and low projected scoring"
        _sort_archetype_label = "Odds"
    else:
        _window_section = """
          <div class="wl-section-label" style="margin-top:10px;">Competitive Windows</div>
          <div class="wl-row"><span class="wl-dot" style="background:#22c55e;"></span><strong class="wl-label">Contender</strong><span class="wl-desc">Elite dynasty + strong scoring projection, premier roster right now</span></div>
          <div class="wl-row"><span class="wl-dot" style="background:#f59e0b;"></span><strong class="wl-label">Win-Now</strong><span class="wl-desc">Elite scoring with aging stars, peak years are here and window is open</span></div>
          <div class="wl-row"><span class="wl-dot" style="background:#84cc16;"></span><strong class="wl-label">Aging Contender</strong><span class="wl-desc">Strong roster projecting well, but franchise age is trending up</span></div>
          <div class="wl-row"><span class="wl-dot" style="background:#3b82f6;"></span><strong class="wl-label">Contender Window</strong><span class="wl-desc">Elite dynasty value with young or prime core, window opening soon</span></div>
          <div class="wl-row"><span class="wl-dot" style="background:#6366f1;"></span><strong class="wl-label">2-3 Year Window</strong><span class="wl-desc">Strong future value building toward contention over the next few seasons</span></div>
          <div class="wl-row"><span class="wl-dot" style="background:#8b5cf6;"></span><strong class="wl-label">Rising</strong><span class="wl-desc">Young, future-heavy roster beginning to accumulate dynasty value</span></div>
          <div class="wl-row"><span class="wl-dot" style="background:#94a3b8;"></span><strong class="wl-label">Holding Pattern</strong><span class="wl-desc">Average across all metrics, direction not yet clear</span></div>
          <div class="wl-row"><span class="wl-dot" style="background:#f97316;"></span><strong class="wl-label">Retooling</strong><span class="wl-desc">Selling aging core, accumulating capital to reset for the future</span></div>
          <div class="wl-row"><span class="wl-dot" style="background:#ef4444;"></span><strong class="wl-label">Rebuilding</strong><span class="wl-desc">Below-average dynasty + redraft, active rebuild in progress</span></div>
          <div class="wl-row"><span class="wl-dot" style="background:#dc2626;"></span><strong class="wl-label">Full Rebuild</strong><span class="wl-desc">Stacked with picks, very low current value, all-in on the future</span></div>
        """
        _grade_note = "Grade factors: dynasty value (40%) &middot; projected scoring (25%) &middot; age profile (15%) &middot; elite players (12%) &middot; draft capital (8%)"
        _grade_a_desc = "Elite roster: top dynasty value, strong depth, elite core players"
        _grade_d_desc = "Weak roster: low dynasty value and scoring projection league-wide"
        _sort_archetype_label = "Archetype"
    _window_legend_html = f"""
    <div class="window-legend-wrap">
      <button class="window-legend-toggle" id="windowLegendToggle" aria-expanded="false">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style="flex-shrink:0"><circle cx="7" cy="7" r="6" stroke="currentColor" stroke-width="1.5"/><path d="M7 6.5v3M7 4.5h.01" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
        Legend
        <svg class="wl-chevron" width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2.5 4.5L6 8l3.5-3.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
      </button>
      <div class="window-legend-panel" id="windowLegendPanel">
        <div class="window-legend-grid">
          <div class="wl-section-label">Grades</div>
          <div class="wl-row"><span class="wl-grade grade-a">A</span><strong class="wl-label">A+ to A&minus;</strong><span class="wl-desc">{_grade_a_desc}</span></div>
          <div class="wl-row"><span class="wl-grade grade-b">B</span><strong class="wl-label">B+ to B&minus;</strong><span class="wl-desc">Competitive roster with clear strengths and some positional gaps</span></div>
          <div class="wl-row"><span class="wl-grade grade-c">C</span><strong class="wl-label">C+ to C&minus;</strong><span class="wl-desc">Below-average roster needing reinforcement in multiple areas</span></div>
          <div class="wl-row"><span class="wl-grade grade-d">D</span><strong class="wl-label">D</strong><span class="wl-desc">{_grade_d_desc}</span></div>
          <div class="wl-grade-note">{_grade_note}</div>
          {_window_section}
          <div class="wl-section-label" style="margin-top:10px;">Roster Shapes</div>
          <div class="wl-grade-note" style="margin-top:0;">How a roster's value is built by position &mdash; descriptive, not part of the grade.</div>
          <div class="wl-row"><span class="wl-shape">CAP</span><span class="wl-desc">Draft capital &mdash; the trade value of the team's draft picks</span></div>
          <div class="wl-row"><span class="wl-shape">WR Factory</span><span class="wl-desc">Value concentrated at WR &mdash; a deep, WR-dominant roster</span></div>
          <div class="wl-row"><span class="wl-shape">Robust RB</span><span class="wl-desc">RB-heavy build with a strong, deep backfield</span></div>
          <div class="wl-row"><span class="wl-shape">Hero RB</span><span class="wl-desc">One elite back anchoring a thin RB room, with a WR-forward rest</span></div>
          <div class="wl-row"><span class="wl-shape">Zero RB</span><span class="wl-desc">Minimal RB value, loaded at WR</span></div>
          <div class="wl-row"><span class="wl-shape">TE Premium</span><span class="wl-desc">Heavy investment at TE &mdash; an elite tight end anchors the build</span></div>
          <div class="wl-row"><span class="wl-shape">Konami Code</span><span class="wl-desc">Superflex build with two-plus premium QBs soaking up roster value</span></div>
          <div class="wl-row"><span class="wl-shape">Balanced</span><span class="wl-desc">No single position dominates &mdash; value spread evenly</span></div>
        </div>
      </div>
    </div>
    """

    # ---------- Page shell ----------
    # Full-page tabs (TEAMS | VALUE | SCHEDULE) -- no sidebar. The tab strip sits
    # at the top; each section is a full-width view. Roster Intel lives in the
    # team detail drawer, not as a page tab. See teams.js (tab wiring) and
    # dashboard.css (.teams-ptabs / .teams-pview).
    return f"""
    <div class="teams-page" id="teamsPageLayout" data-active-tab="teams">
      <nav class="teams-ptabs" role="tablist" aria-label="Teams views">
        <button class="on" data-ptab="teams" role="tab" aria-selected="true">TEAMS</button>
        <button data-ptab="value" role="tab" aria-selected="false">VALUE</button>
        <button data-ptab="sched" role="tab" aria-selected="false" id="schedPtabBtn" style="display:none">SCHEDULE</button>
      </nav>
      <section class="teams-pview on" id="v-teams" role="tabpanel">
        <div class="teams-topbar">
          <div class="teams-sort-bar">
            <span style="font-size:13px;color:var(--text-muted);margin-right:8px;">Sort by:</span>
            <div class="otc-main-tabs br-slide-tabs teams-sort-tabs" data-br-slide-tabs>
              <button class="teams-sort-btn otc-main-tab" data-sort="posindex">Positional Index</button>
              <button class="teams-sort-btn otc-main-tab" data-sort="grade">Team Grade</button>
              <button class="teams-sort-btn otc-main-tab" data-sort="archetype">{_sort_archetype_label}</button>
            </div>
            <span id="teamsSortLabel" style="font-size:11px;color:var(--text-muted);margin-left:10px;opacity:0;transition:opacity .2s;"></span>
          </div>
          {_window_legend_html}
        </div>
        <div class="teams-grid" id="teamsGrid">
          {all_cards_html}
        </div>
      </section>
      <section class="teams-pview" id="v-value" role="tabpanel">
        <div id="btmPanel">{_analytics_skeleton}</div>
      </section>
      <section class="teams-pview" id="v-sched" role="tabpanel">
        <div id="sosPanel">{_analytics_skeleton}</div>
      </section>
    </div>
    <div class="td-scrim" id="teamDrawerScrim"></div>
    <aside class="td-drawer" id="teamDrawer" role="dialog" aria-modal="true" aria-label="Team details">
      <div class="td-head" id="teamDrawerHead"></div>
      <div class="td-body" id="teamDrawerBody"></div>
    </aside>
    <script id="teamsDrawerData" type="application/json">{_drawer_json}</script>
    {_teams_foot_scripts}

    <script>
    (function() {{

      // Teams sort bar
      var _sortKey = '';
      function floatViewer() {{
        var rid = (window._viewerRid || '').toString().trim();
        if (!rid) return;
        var grid = document.getElementById('teamsGrid');
        if (!grid) return;
        var viewer = grid.querySelector('.team-strength-card[data-roster-id="' + rid + '"]');
        if (viewer && grid.firstChild !== viewer) grid.insertBefore(viewer, grid.firstChild);
      }}
      function _setSortLabel(text) {{
        var lbl = document.getElementById('teamsSortLabel');
        if (!lbl) return;
        lbl.textContent = text ? ('Sorted by ' + text) : '';
        lbl.style.opacity = text ? '1' : '0';
      }}
      function restoreDefault() {{
        var grid = document.getElementById('teamsGrid');
        if (!grid) return;
        var cards = Array.from(grid.querySelectorAll('.team-strength-card'));
        cards.sort(function(a, b) {{ return Number(a.dataset.originalIndex) - Number(b.dataset.originalIndex); }});
        cards.forEach(function(c) {{ grid.appendChild(c); }});
        floatViewer();
        document.querySelectorAll('.teams-sort-btn').forEach(function(btn) {{ btn.classList.remove('active'); }});
        var _st0 = document.querySelector('.teams-sort-tabs'); if (_st0) _st0.classList.remove('has-active');
        _sortKey = '';
        _setSortLabel('');
      }}
      var _sortKeyLabels = {{ posindex: 'Positional Index', grade: 'Team Grade', archetype: '{_sort_archetype_label}' }};
      function sortTeams(key) {{
        // clicking the active sort deselects it and restores default order
        if (_sortKey === key) {{ restoreDefault(); return; }}
        _sortKey = key;
        var grid = document.getElementById('teamsGrid');
        if (!grid) return;
        var cards = Array.from(grid.querySelectorAll('.team-strength-card'));
        cards.sort(function(a, b) {{
          if (key === 'grade') {{
            return (Number(b.dataset.sortGrade) || 0) - (Number(a.dataset.sortGrade) || 0);
          }} else if (key === 'archetype') {{
            return Number(a.dataset.sortArchetype) - Number(b.dataset.sortArchetype);
          }} else {{
            return Number(b.dataset.sortPosindex) - Number(a.dataset.sortPosindex);
          }}
        }});
        cards.forEach(function(c) {{ grid.appendChild(c); }});
        document.querySelectorAll('.teams-sort-btn').forEach(function(btn) {{
          btn.classList.toggle('active', btn.dataset.sort === key);
        }});
        var _st1 = document.querySelector('.teams-sort-tabs'); if (_st1) _st1.classList.add('has-active');
        _setSortLabel(_sortKeyLabels[key] || key);
      }}
      document.querySelectorAll('.teams-sort-btn').forEach(function(btn) {{
        btn.addEventListener('click', function() {{ sortTeams(btn.dataset.sort); }});
      }});
      // Default: float the viewer's card to the top using the session-injected _viewerRid
      (function() {{
        floatViewer();
      }})();

      // Lazy-render Plotly charts as they scroll into view
      (function() {{
        function renderChart(el) {{
          if (el.dataset.rendered) return;
          el.dataset.rendered = '1';
          try {{
            var trace  = JSON.parse(el.getAttribute('data-chart'));
            var layout = JSON.parse(el.getAttribute('data-layout'));
            el.innerHTML = '';
            Plotly.newPlot(el.id, trace, layout, {{responsive: true, displayModeBar: false}});
          }} catch(e) {{}}
        }}
        function tryRender(el) {{
          if (window.ensurePlotly) {{ window.ensurePlotly().then(function() {{ renderChart(el); }}).catch(function() {{}}); }}
        }}
        var charts = document.querySelectorAll('.team-chart-lazy');
        if ('IntersectionObserver' in window) {{
          var obs = new IntersectionObserver(function(entries) {{
            entries.forEach(function(e) {{
              if (e.isIntersecting) {{ tryRender(e.target); obs.unobserve(e.target); }}
            }});
          }}, {{ rootMargin: '300px' }});
          charts.forEach(function(el) {{ obs.observe(el); }});
        }} else {{
          charts.forEach(tryRender);
        }}
      }})();

      // Window legend toggle
      (function() {{
        var btn   = document.getElementById('windowLegendToggle');
        var panel = document.getElementById('windowLegendPanel');
        if (!btn || !panel) return;
        btn.addEventListener('click', function() {{
          var open = panel.classList.toggle('wl-open');
          btn.setAttribute('aria-expanded', open ? 'true' : 'false');
        }});
      }})();

    }})();
    </script>
    """
