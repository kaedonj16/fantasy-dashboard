"""Teams page builder (deep team analytics: roster grades, intel, archetypes,
playoff odds, draft grades, pick value).

Moved verbatim from app.py to shrink the monolith. The heavy app.py internals it
uses are lazy-imported from app inside the function (resolved at request time),
so importing this module at start-up never triggers a circular import.
"""
import html
import json
import logging
import math
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Union

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
    if ts >= 0.15:
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
        _pk_pick_label, _pk_pick_value_from_table, _playoff_sim_cached, _safe_int,
        _team_pick_value, _weighted_pos_strength, apply_te_premium, avatar_from_users,
        build_historical_pick_slot_map, count_roster_positions, get_roster_positions,
        has_draft_ended, load_pick_value_table, te_premium_from_settings, _TEAMS_JS_V,
    )
    rosters = ctx["rosters"]  # Sleeper /rosters
    roster_map = ctx["roster_map"]  # mapping roster_id -> team name
    users = ctx["users"]
    platform = ctx["platform"]
    picks_by_roster = ctx.get("picks_by_roster") or {}
    league_id = str(ctx.get("league_id") or "")
    current_season = _safe_int((ctx.get("league") or {}).get("season"), datetime.now().year)

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
    try:
        _pk_odds = _playoff_sim_cached(ctx, platform)
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

    name_to_rank_label: Dict[str, str] = {}
    name_to_age: Dict[str, Union[float, None]] = {}

    for obj in model_vals:
        if not isinstance(obj, dict):
            continue
        safe_name = str(obj.get("search_name") or "").strip().lower()
        if not safe_name:
            continue
        pos_lbl = obj.get("pos_rank_label") or obj.get("position") or obj.get("pos") or ""
        name_to_rank_label[safe_name] = str(pos_lbl)
        age_val = obj.get("age")
        if age_val is not None:
            try:
                name_to_age[safe_name] = float(age_val)
            except Exception:
                name_to_age[safe_name] = None

    # map sleeper_id -> row. Apply the league's TE premium up front (on a shallow
    # copy, never the cached row) so every downstream value read — sort, age
    # weighting, positional strength — uses the TE-adjusted value automatically.
    _tep = te_premium_from_settings(ctx.get("scoring_settings"))

    def _te_adj_row(p: dict) -> dict:
        if _tep and str(p.get("position") or p.get("pos") or "").upper() == "TE":
            return {**p, "value": apply_te_premium(p.get("value"), "TE", _tep)}
        return p

    by_id: Dict[str, Dict] = {
        str(p["id"]): _te_adj_row(p)
        for p in model_vals
        if isinstance(p, dict) and p.get("id") is not None
    }

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
        avatar = avatar_from_users(platform, users, str(rid))
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

    pick_series = list(team_pick_value.values())
    _pick_mean = sum(pick_series) / len(pick_series) if pick_series else 0.0
    _pick_var = sum((v - _pick_mean) ** 2 for v in pick_series) / len(pick_series) if pick_series else 0.0
    _pick_std = math.sqrt(_pick_var)
    team_pick_z: Dict[int, float] = {
        rid: ((v - _pick_mean) / _pick_std if _pick_std > 0 else 0.0)
        for rid, v in team_pick_value.items()
    }
    pick_z_min = min(team_pick_z.values()) if team_pick_z else 0.0
    pick_z_max = max(team_pick_z.values()) if team_pick_z else 0.0

    # ----------------- Compute per-team positional strength + league baselines -----------------
    team_pos_strength: Dict[int, Dict[str, float]] = defaultdict(dict)
    slot_counts = count_roster_positions(get_roster_positions())

    for rid, pos_map in team_pos_values.items():
        for pos, vals in pos_map.items():
            team_pos_strength[rid][pos] = _weighted_pos_strength(vals, pos, slot_counts)

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
    team_pos_z: Dict[int, Dict[str, float]] = defaultdict(dict)
    team_pos_index: Dict[int, float] = {}

    LINEUP_WEIGHTS = {
        "QB": slot_counts.get("QB") or 1,
        "RB": slot_counts.get("RB") or 2,
        "WR": slot_counts.get("WR") or 2,
        "TE": slot_counts.get("TE") or 1,
        "FLEX": slot_counts.get("FLEX") or 1,
    }
    weight_sum = sum(LINEUP_WEIGHTS[pos] for pos in POS_ORDER if LINEUP_WEIGHTS.get(pos, 0) > 0) or 1.0

    pos_z_min: Dict[str, float] = {pos: float("inf") for pos in POS_ORDER}
    pos_z_max: Dict[str, float] = {pos: float("-inf") for pos in POS_ORDER}

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
            team_pos_z[rid][pos] = z

            pos_z_min[pos] = min(pos_z_min[pos], z)
            pos_z_max[pos] = max(pos_z_max[pos], z)

            w = LINEUP_WEIGHTS.get(pos, 0)
            idx_num += w * z

        team_pos_index[rid] = idx_num / weight_sum

    for pos in POS_ORDER:
        if pos_z_min[pos] == float("inf"):
            pos_z_min[pos] = 0.0
        if pos_z_max[pos] == float("-inf"):
            pos_z_max[pos] = 0.0

    # ----------------- Positional ranks (per position) -----------------
    # pos_rank[pos][rid] = rank (1 = best at that position)
    pos_rank: Dict[str, Dict[int, int]] = {pos: {} for pos in POS_ORDER}

    for pos in POS_ORDER:
        # rank by z-score (strongest to weakest)
        ranked = sorted(
            team_meta.keys(),
            key=lambda rid: team_pos_z[rid].get(pos, 0.0),
            reverse=True,
        )
        for i, rid in enumerate(ranked, start=1):
            pos_rank[pos][rid] = i

    # ----------------- Helper: players under a position row -----------------
    def render_pos_players(rid: int, pos_code: str) -> str:
        plist = roster_pos_players.get(rid, {}).get(pos_code, [])
        if not plist:
            return "<div style='color:#64748b;font-size:12px;'>No players at this position.</div>"

        rows_html = []
        for p in plist:
            name = p.get("name")
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

            # Build meta parts (rank, team, age)
            meta_parts = [rank_label, p.get('team', '')]
            if age_txt:
                meta_parts.append(age_txt)
            meta_str = " • ".join(filter(None, meta_parts))

            player_id = p.get("id", "")
            position = p.get('position', '')
            years_exp = p.get('years_exp')
            rows_html.append(
                "<div class='player-activity'>"
                "  <div style='display:flex;align-items:center;justify-content:space-between;width:100%'>"
                "    <div style='display: inline-flex;gap: 5px;align-items: center;'>"
                f"      <div style='font-weight:600;cursor:pointer;' class='player-clickable' data-player-id='{player_id}' data-player-name='{name}' data-position='{position}' data-years-exp='{years_exp}' data-value='{val}' data-breakout-check='true'>{name}</div>"
                f"      <div style='color:#64748b;font-size:12px'>"
                f"        {meta_str}"
                "      </div>"
                "    </div>"
                f"    <div class='player-trade-value'>{val_txt}</div>"
                "  </div>"
                "</div>"
            )

        return "".join(rows_html)

    # Pre-compute global chart Y-max so all team cards share the same Y-axis scale
    _chart_all_pos_vals = []
    for _rid in team_meta:
        _chart_all_pos_vals.extend([
            sum(team_pos_values[_rid].get("QB", [])),
            sum(team_pos_values[_rid].get("RB", [])),
            sum(team_pos_values[_rid].get("WR", [])),
            sum(team_pos_values[_rid].get("TE", [])),
            team_pick_value.get(_rid, 0.0),
        ])
    _chart_y_max = round(max(_chart_all_pos_vals) * 1.15, 1) if _chart_all_pos_vals else 100.0

    # Pre-compute roster grades for all teams
    from dashboard_services.ai.context_builders import calculate_roster_grade as _calc_grade

    _n_teams = len(team_meta)
    _offseason = ctx.get("offseason_mode", False)
    _rp_list = ctx.get("roster_positions") or []
    _is_sf = any(str(s).upper() in {"SUPER_FLEX", "SFLEX"} for s in _rp_list)
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
        )

    team_grades = {rid: _grade_for_roster(rid) for rid in team_meta}

    # ----------------- Build HTML cards -----------------
    cards_html = []

    for _card_idx, (rid, meta) in enumerate(team_meta.items()):
        name = meta["name"]
        avatar = meta.get("avatar") or ""
        img_html = (
            f"<img class='avatar' src='{avatar}' alt='' loading='lazy' decoding='async' onerror=\"this.style.display='none'\">"
            if avatar else ""
        )

        z_map = team_pos_z[rid]
        strongest_pos = max(POS_ORDER, key=lambda p: z_map.get(p, 0.0))
        weakest_pos = min(POS_ORDER, key=lambda p: z_map.get(p, 0.0))

        table_rows = []
        for pos in POS_ORDER:
            vals = team_pos_values[rid][pos]
            count = len(vals)
            total = sum(vals)
            strength_score = team_pos_strength[rid][pos]
            z = z_map[pos]

            # bar width scaled within this position across league
            z_min = pos_z_min[pos]
            z_max = pos_z_max[pos]
            if z_max > z_min:
                pct = 10 + 80 * (z - z_min) / (z_max - z_min)  # 10–90%
            else:
                pct = 50.0

            highlight_class = ""
            if pos == strongest_pos:
                highlight_class = " pos-strongest"
            elif pos == weakest_pos:
                highlight_class = " pos-weakest"

            rank = pos_rank[pos].get(rid, 0)
            _pos_age = team_pos_age.get(int(rid), {}).get(pos)
            _age_txt = f"{_pos_age:.1f}" if _pos_age is not None else "–"

            # main row (clickable)
            main_row = (

                "<tr class='pos-row{cls}' data-pos='{pos}'>"
                "  <td class='pos-name'>"
                "    <span class='pos-row-toggle'>▾</span> {pos}"
                "  </td>"
                "  <td class='pos-count'>{count}</td>"
                "  <td class='pos-age'>{age}</td>"
                "  <td class='pos-total'>{total:.1f}</td>"
                "  <td class='pos-avg'>{strength_score:.1f}</td>"
                "  <td class='pos-z'>{z:.2f}</td>"
                "  <td class='pos-bar-cell'>"
                "    <div class='pos-bar-outer'>"
                "      <div class='pos-bar-inner' style='width:{pct:.0f}%;'></div>"
                "    </div>"
                "  </td>"
                "<td class='pos-rank'>#{rank}</td>"
                "</tr>".format(
                    cls=highlight_class,
                    rank=rank,
                    pos=pos,
                    count=count,
                    total=total,
                    z=z,
                    pct=pct,
                    strength_score=strength_score,
                    age=_age_txt,
                )
            )

            # detail row right under it (collapsed by default)
            detail_html = render_pos_players(rid, pos)
            detail_row = (
                f"<tr class='pos-detail-row' data-pos='{pos}' style='display:none;'>"
                "  <td colspan='8'>"
                "    <div class='pos-detail-inner'>"
                f"      {detail_html}"
                "    </div>"
                "  </td>"
                "</tr>"
            )

            table_rows.append(main_row)
            table_rows.append(detail_row)

        # Draft Capital row
        pick_val = team_pick_value.get(rid, 0.0)
        pick_z = team_pick_z.get(rid, 0.0)
        if pick_z_max > pick_z_min:
            pick_pct = 10 + 80 * (pick_z - pick_z_min) / (pick_z_max - pick_z_min)
        else:
            pick_pct = 50.0
        pick_count = len(picks_by_roster.get(str(rid), []))
        table_rows.append(
            "<tr class='pos-row pos-picks-row'>"
            "  <td class='pos-name'>"
            "    <span class='pos-row-toggle'>▾</span> "
            "    <i class='fa-solid fa-clipboard-list' style='font-size:11px;opacity:0.7;'></i> PICKS"
            "  </td>"
            f"  <td class='pos-count'>{pick_count}</td>"
            "  <td class='pos-age'>–</td>"
            f"  <td class='pos-total'>{pick_val:.1f}</td>"
            "  <td class='pos-avg'>–</td>"
            f"  <td class='pos-z'>{pick_z:+.2f}</td>"
            "  <td class='pos-bar-cell'>"
            "    <div class='pos-bar-outer'>"
            f"      <div class='pos-bar-inner' style='width:{pick_pct:.0f}%;background:var(--color-pick,#8b5cf6);'></div>"
            "    </div>"
            "  </td>"
            "  <td class='pos-rank'></td>"
            "</tr>"
        )

        # Expandable pick detail: each future pick with its projected slot
        # ("2027 1.03" from the playoff-odds sim, tagged projected) and value.
        _pk_rows = []
        for _pk in picks_by_roster.get(str(rid), []):
            try:
                _pk_yr = int(_pk.get("season") or 0)
                _pk_rnd = int(_pk.get("round") or 0)
            except (TypeError, ValueError):
                continue
            if not _pk_yr or not _pk_rnd:
                continue
            _pk_orig = str(_pk.get("original_owner") or rid)
            _pk_slot = None
            _pk_is_proj = False
            if _pk_yr == current_season:
                # Upcoming draft: order is final (last season is in the books).
                try:
                    _pk_slot = _pk_final_slots.get(int(_pk_orig))
                except (TypeError, ValueError):
                    _pk_slot = None
            elif _pk_yr == _pk_proj_year:
                _pk_slot = _pk_slot_by_original.get(_pk_orig)
                _pk_is_proj = _pk_slot is not None
            _pk_lbl = _pk_pick_label(_pk_yr, _pk_rnd, _pk_slot)
            _pk_val = _pk_pick_value_from_table(
                _pk_value_tbl, _pk_yr, _pk_rnd, _pk_slot, len(rosters) or 10
            )
            _pk_from = ""
            if _pk_orig != str(rid):
                _pk_from_name = roster_map.get(_pk_orig) or f"Roster {_pk_orig}"
                _pk_from = f"<span class='dc-from'>from {html.escape(str(_pk_from_name))}</span>"
            _pk_badge = "<span class='dc-proj'>projected</span>" if _pk_is_proj else ""
            _pk_rows.append(
                f"<li class='dc-pick'>"
                f"<span class='dc-pick-label'>{html.escape(_pk_lbl)}</span>"
                f"{_pk_from}{_pk_badge}"
                f"<span class='dc-pick-val'>{_pk_val:,.0f}</span>"
                f"</li>"
            )
        _pk_note = (
            f"<div class='dc-note'>{_pk_proj_year} slots are projected from current "
            f"playoff odds; later years use round values.</div>"
            if any("dc-proj" in r for r in _pk_rows) else ""
        )
        _pk_detail = (
            f"<ul class='dc-pick-list'>{''.join(_pk_rows)}</ul>{_pk_note}"
            if _pk_rows else "<div class='dc-none'>No future picks</div>"
        )
        table_rows.append(
            "<tr class='pos-detail-row' data-pos='PICKS' style='display:none;'>"
            "  <td colspan='8'>"
            f"    <div class='pos-detail-inner'>{_pk_detail}</div>"
            "  </td>"
            "</tr>"
        )

        # ── Position value bar chart ──────────────────────────────────────────
        _chart_labels  = ["QB", "RB", "WR", "TE", "Picks"]
        _chart_colors  = ["#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6", "#c92c68"]
        _chart_values  = [
            round(sum(team_pos_values[rid].get("QB", [])), 1),
            round(sum(team_pos_values[rid].get("RB", [])), 1),
            round(sum(team_pos_values[rid].get("WR", [])), 1),
            round(sum(team_pos_values[rid].get("TE", [])), 1),
            round(team_pick_value.get(rid, 0.0), 1),
        ]
        _chart_div_id  = f"teamValueChart_{rid}"
        _chart_data    = json.dumps([{
            "type":          "bar",
            "x":             _chart_labels,
            "y":             _chart_values,
            "marker":        {"color": _chart_colors},
            "hovertemplate": "%{x}: %{y:,.0f}<extra></extra>",
        }])
        _chart_layout  = json.dumps({
            "margin":       {"t": 8, "b": 28, "l": 44, "r": 8},
            "paper_bgcolor":"rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "height":       200,
            "yaxis": {
                "range":      [0, _chart_y_max],
                "tickformat": ".2s",
                "showgrid":   True,
                "gridcolor":  "rgba(100,116,139,0.2)",
                "zeroline":   False,
                "tickfont":   {"size": 11},
            },
            "xaxis": {"showgrid": False, "tickfont": {"size": 12}},
            "showlegend":   False,
            "bargap":       0.3,
        })
        _chart_data_attr  = html.escape(_chart_data,   quote=True)
        _chart_layout_attr = html.escape(_chart_layout, quote=True)
        _chart_html = (
            f'<div id="{_chart_div_id}" class="team-value-chart team-chart-lazy"'
            f' data-chart="{_chart_data_attr}"'
            f' data-layout="{_chart_layout_attr}">'
            f'<div class="team-chart-skeleton"></div>'
            f'</div>'
        )

        _gdata = team_grades.get(rid, {})
        _grade = _gdata.get("grade", "?")
        _win_window = _gdata.get("win_window", "")
        _grade_cls = "grade-a" if _grade.startswith("A") else "grade-b" if _grade.startswith("B") else "grade-c" if _grade.startswith("C") else "grade-d"
        _grade_badge = f"<span class='roster-grade-inline {_grade_cls}' title='{_win_window}'>{_grade}</span>"

        # Numeric sort keys for client-side sorting
        _grade_num = {"A+":12,"A":11,"A-":10,"B+":9,"B":8,"B-":7,"C+":6,"C":5,"C-":4,"D+":3,"D":2,"D-":1,"F":0}.get(_grade, 0)
        _archetype_num = {
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
        _shape = roster_shape_label(team_pos_values[rid], _is_sf)
        _shape_html = (
            f"<span class='tc-shape'>{html.escape(_shape)}</span>"
            "<span class='tc-shape-sep'> &bull; </span>"
            if _shape else ""
        )

        _is_viewer = str(rid) == str(viewer_roster_id or "")
        card_html = (
            f"<div class='card team-strength-card {_window_cls}' data-br-moment='draftgrade' data-sort-grade='{_grade_num}' data-sort-posindex='{_pos_idx:.4f}' data-sort-archetype='{_archetype_num}' data-roster-id='{rid}' data-original-index='{_card_idx}'" + (" data-viewer='1'" if _is_viewer else "") + ">"
            "  <div class='card-header-row'>"
            f"    <div style='display:flex;align-items:center;gap:8px;min-width:0;flex:1;'>{img_html}<h2 class='team-clickable' style='cursor:pointer;' data-roster-id='{rid}' data-team-name='{name}'>{name}</h2>"
            f"<div class='mini-label' style='flex-shrink:0;'>{_shape_html}<span class='grade-window-label'>{_win_window}</span></div></div>"
            "    <div style='display:flex;align-items:center;gap:6px;flex-shrink:0;'>"
            f"      {_grade_badge}"
            + f"      <button class='share-report-btn' title='Share team report card' "
               f"data-roster='{rid}' data-platform='{platform}' data-season='{current_season}' data-league='{league_id}'>"
               "<svg class='share-report-icon' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' aria-hidden='true'><circle cx='18' cy='5' r='3'/><circle cx='6' cy='12' r='3'/><circle cx='18' cy='19' r='3'/><line x1='8.59' y1='13.51' x2='15.42' y2='17.49'/><line x1='15.41' y1='6.51' x2='8.59' y2='10.49'/></svg></button>"
            +
            "      <button class='team-card-toggle' aria-label='Expand card' aria-expanded='false'>"
            "        <svg width='14' height='14' viewBox='0 0 14 14' fill='none'>"
            "          <path d='M3 5l4 4 4-4' stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/>"
            "        </svg>"
            "      </button>"
            "    </div>"
            "  </div>"
            "  <div class='card-body'>"
            f"    {_chart_html}"
            f"    <div class='tc-posindex-row' title='Positional Index: how far this team&apos;s starting-lineup strength sits above or below the league average, in standard deviations.'>"
            f"      <span class='tc-posindex-lbl'>Positional Index</span>"
            f"      <span class='tc-pi-num'>{_pos_idx:+.2f}</span>"
            f"    </div>"
            "    <div class='pos-table-wrap'>"
            "    <table class='pos-strength-table'>"
            "      <thead>"
            "        <tr>"
            "          <th>Pos</th>"
            "          <th>#</th>"
            "          <th>Age</th>"
            "          <th>Value</th>"
            "          <th>Score</th>"
            "          <th>Z</th>"
            "          <th>Strength</th>"
            "          <th>Rank</th>"
            "        </tr>"
            "      </thead>"
            "      <tbody>"
            f"        {''.join(table_rows)}"
            "      </tbody>"
            "    </table>"
            "    </div>"
            "  </div>"
            "</div>"
        )

        cards_html.append(card_html)

    all_cards_html = "".join(
        cards_html) or "<div class='card'><div class='card-body'><p>No teams found.</p></div></div>"

    # ---------- League analytics section (lazy-loaded) ----------
    platform_js = platform
    season_js = current_season

    # Detect league type (sf vs 1qb) from roster positions
    _rp = get_roster_positions()
    _rp_list = list(_rp) if _rp else []
    _is_sf = any(str(s).upper() in {"SUPER_FLEX", "SFLEX"} for s in _rp_list)
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
    <script src="/static/teams.js?v={_TEAMS_JS_V}" defer></script>
    """

    # ---------- Window legend ----------
    _window_legend_html = """
    <div class="window-legend-wrap">
      <button class="window-legend-toggle" id="windowLegendToggle" aria-expanded="false">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style="flex-shrink:0"><circle cx="7" cy="7" r="6" stroke="currentColor" stroke-width="1.5"/><path d="M7 6.5v3M7 4.5h.01" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
        Legend
        <svg class="wl-chevron" width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2.5 4.5L6 8l3.5-3.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
      </button>
      <div class="window-legend-panel" id="windowLegendPanel">
        <div class="window-legend-grid">
          <div class="wl-section-label">Grades</div>
          <div class="wl-row"><span class="wl-grade grade-a">A</span><strong class="wl-label">A+ to A&minus;</strong><span class="wl-desc">Elite roster: top dynasty value, strong depth, elite core players</span></div>
          <div class="wl-row"><span class="wl-grade grade-b">B</span><strong class="wl-label">B+ to B&minus;</strong><span class="wl-desc">Competitive roster with clear strengths and some positional gaps</span></div>
          <div class="wl-row"><span class="wl-grade grade-c">C</span><strong class="wl-label">C+ to C&minus;</strong><span class="wl-desc">Below-average roster needing reinforcement in multiple areas</span></div>
          <div class="wl-row"><span class="wl-grade grade-d">D</span><strong class="wl-label">D</strong><span class="wl-desc">Weak roster: low dynasty value and scoring projection league-wide</span></div>
          <div class="wl-grade-note">Grade factors: dynasty value (40%) &middot; projected scoring (25%) &middot; age profile (15%) &middot; elite players (12%) &middot; draft capital (8%)</div>
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
          <div class="wl-section-label" style="margin-top:10px;">Roster Shapes</div>
          <div class="wl-grade-note" style="margin-top:0;">How a roster's value is built by position &mdash; descriptive, not part of the grade.</div>
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
    # Desktop: two columns — the team-grade cards (sort bar + legend + grid) fill
    # the main column and stay visible, while League Analytics (Value / Roster
    # Intel / Schedule) sits in a right-hand sidebar with its own tab strip.
    # Mobile (<=1180px): the whole thing collapses into a single tabbed card
    # (Teams / Value / Roster Intel / Schedule) — the "Teams" tab shows the grid,
    # the others show the analytics panels. See teams.js (default-tab wiring) and
    # dashboard.css (.teams-page responsive rules).
    return f"""
    <div class="page-layout teams-page" id="teamsPageLayout" data-active-tab="teams">
      <main class="page-main">
        <div class="teams-topbar">
          <div class="teams-sort-bar">
            <span style="font-size:12px;color:var(--text-muted);margin-right:8px;">Sort by:</span>
            <div class="otc-main-tabs br-slide-tabs teams-sort-tabs" data-br-slide-tabs>
              <button class="teams-sort-btn otc-main-tab" data-sort="posindex">Positional Index</button>
              <button class="teams-sort-btn otc-main-tab" data-sort="grade">Team Grade</button>
              <button class="teams-sort-btn otc-main-tab" data-sort="archetype">Archetype</button>
            </div>
            <span id="teamsSortLabel" style="font-size:11px;color:var(--text-muted);margin-left:10px;opacity:0;transition:opacity .2s;"></span>
          </div>
          {_window_legend_html}
        </div>
        <div class="teams-grid" id="teamsGrid">
          {all_cards_html}
        </div>
      </main>

      <aside class="page-sidebar teams-sidebar">
        <div class="card teams-analytics-card" id="teamsAnalyticsCard">
          <div class="card-tabs">
            <div class="tab-strip" id="teamsAnalyticsTabs">
              <button class="tab-btn active teams-tab-mobile" data-tab="teams">Teams</button>
              <button class="tab-btn" data-tab="btm">Value</button>
              <button class="tab-btn" data-tab="roster-intel">Roster Intel</button>
              <button class="tab-btn" data-tab="sos" id="sosTabBtn" style="display:none">Schedule</button>
            </div>
            <div class="tab-panels">
              <div class="tab-panel" data-tab="btm" id="btmPanel">{_analytics_skeleton}</div>
              <div class="tab-panel" data-tab="roster-intel" id="rosterIntelPanel">{_analytics_skeleton}</div>
              <div class="tab-panel" data-tab="sos" id="sosPanel">{_analytics_skeleton}</div>
            </div>
          </div>
        </div>
      </aside>
    </div>
    {_teams_foot_scripts}

    <script>
    (function() {{
      // Click a position row to toggle its detail row
      document.addEventListener('click', function(e) {{
        const row = e.target.closest('.pos-row');
        if (!row) return;
        const detail = row.nextElementSibling;
        if (!detail || !detail.classList.contains('pos-detail-row')) return;

        const isOpen = detail.style.display === '' || detail.style.display === 'table-row';
        detail.style.display = isOpen ? 'none' : 'table-row';

        // rotate the little arrow
        const chevron = row.querySelector('.pos-row-toggle');
        if (chevron) {{
          chevron.style.transform = isOpen ? 'rotate(0deg)' : 'rotate(180deg)';
        }}
      }});

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
      var _sortKeyLabels = {{ posindex: 'Positional Index', grade: 'Team Grade', archetype: 'Archetype' }};
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

      // Mobile collapsible team cards
      document.querySelectorAll('.team-card-toggle').forEach(function(btn) {{
        btn.addEventListener('click', function(e) {{
          e.stopPropagation();
          var card = btn.closest('.team-strength-card');
          if (!card) return;
          var expanded = card.classList.toggle('tc-expanded');
          btn.setAttribute('aria-expanded', expanded ? 'true' : 'false');
        }});
      }});

    }})();
    </script>
    """
