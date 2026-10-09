"""League-context roster grades shared by the Teams page and the dashboard hero.

The Teams page computes grades with full league context (dynasty/redraft
percentiles vs the league, positional ranks, dynasty/redraft ratio) via
``calculate_roster_grade``. The dashboard hero previously used
``get_roster_grade`` from the AI renderer, which calls the same function
WITHOUT league context and falls back to a different blend, so the two
surfaces could disagree (e.g. "B+ Holding Pattern" vs "A+ Contender").

This module mirrors the Teams page computation exactly so both surfaces
always show the same grade and window label.
"""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

_CORE_POS = {"QB", "RB", "WR", "TE"}
_POS_ORDER = ["QB", "RB", "WR", "TE"]


def viewer_league_context_grade(ctx: dict, viewer_roster_id) -> dict:
    """Return the Teams-page grade dict for one roster.

    Mirrors ``dashboard_services/pages/teams_page.py`` (_grade_for_roster and
    its percentile scaffolding) so the dashboard hero strip shows the same
    grade and win-window label as the Teams page. Returns {} on any failure
    so the hero degrades gracefully.
    """
    try:
        return _compute(ctx, viewer_roster_id)
    except Exception:
        logger.debug("league-context grade failed", exc_info=True)
        return {}


def _compute(ctx: dict, viewer_roster_id) -> dict:
    from dashboard_services.ai.context_builders import (
        calculate_roster_grade,
        ctx_scoring_type,
        league_format_value_lookup,
        redraft_window_label,
    )
    from utils.lineup_slots import (
        count_roster_positions,
        get_roster_positions,
        is_superflex_lineup,
    )
    from utils.trade import rank_rosters_by_position

    rosters = ctx.get("rosters") or []
    if not rosters:
        return {}

    try:
        _vid = int(viewer_roster_id)
    except (TypeError, ValueError):
        return {}

    model_vals = ctx.get("model_value_table") or []
    by_id: Dict[str, dict] = league_format_value_lookup(ctx)

    name_to_age: Dict[str, float | None] = {}
    for obj in model_vals:
        if not isinstance(obj, dict):
            continue
        safe_name = str(obj.get("search_name") or "").strip().lower()
        if not safe_name:
            continue
        age_val = obj.get("age")
        if age_val is not None:
            try:
                name_to_age[safe_name] = float(age_val)
            except (TypeError, ValueError):
                name_to_age[safe_name] = None

    _rp = ctx.get("roster_positions") or get_roster_positions() or []
    _is_sf = is_superflex_lineup(_rp)
    _redraft_key = "redraft_value_sf" if _is_sf else "redraft_value_1qb"
    scoring = ctx_scoring_type(ctx)
    _is_redraft = scoring == "redraft"

    # Per-team position value buckets (for positional ranks).
    team_pos_values: Dict[int, Dict[str, list]] = {}
    for r in rosters:
        rid = r.get("roster_id")
        if rid is None:
            continue
        try:
            rid_i = int(rid)
        except (TypeError, ValueError):
            continue
        buckets: Dict[str, list] = {p: [] for p in _POS_ORDER}
        for pid in (r.get("players") or []):
            row = by_id.get(str(pid))
            if not row:
                continue
            pos = str(row.get("position") or row.get("pos") or "").upper()
            try:
                val = float(row.get("value") or 0.0)
            except (TypeError, ValueError):
                val = 0.0
            if val <= 0:
                continue
            if pos in buckets:
                buckets[pos].append(val)
        team_pos_values[rid_i] = buckets

    slot_counts = count_roster_positions(_rp)
    _, pos_rank = rank_rosters_by_position(
        team_pos_values, slot_counts, positions=_POS_ORDER
    )

    # Dynasty/redraft totals per team (top-8 by dynasty value), then percentiles.
    team_dynasty_total: Dict[int, float] = {}
    team_redraft_total: Dict[int, float] = {}
    team_dr_ratio: Dict[int, float] = {}
    for r in rosters:
        rid = r.get("roster_id")
        if rid is None:
            continue
        try:
            rid_i = int(rid)
        except (TypeError, ValueError):
            continue
        pairs = []
        for pid in (r.get("players") or []):
            row = by_id.get(str(pid))
            if not row:
                continue
            pos = str(row.get("position") or "").upper()
            if pos not in _CORE_POS:
                continue
            try:
                dval = float(row.get("value") or 0)
            except (TypeError, ValueError):
                dval = 0.0
            try:
                rval = float(row.get(_redraft_key) or 0)
            except (TypeError, ValueError):
                rval = 0.0
            pairs.append((dval, rval))
        pairs.sort(reverse=True)
        team_dynasty_total[rid_i] = sum(d for d, _ in pairs[:8])
        team_redraft_total[rid_i] = sum(rv for _, rv in pairs[:8])
        ratios = [d / max(rv, 1) for d, rv in pairs[:10] if d > 50 or rv > 50]
        team_dr_ratio[rid_i] = round(sum(ratios) / len(ratios), 3) if ratios else 1.0

    def _make_pct_fn(totals: Dict[int, float]):
        _sorted = sorted(totals.values())
        _n = max(len(_sorted) - 1, 1)

        def _pct(rid: int) -> float:
            t = totals.get(rid, 0.0)
            return sum(1 for v in _sorted if v < t) / _n

        return _pct

    dynasty_pct = _make_pct_fn(team_dynasty_total)
    redraft_pct = _make_pct_fn(team_redraft_total)
    n_teams = len(team_pos_values)

    # Flat player list for the viewer, same shape as the Teams page.
    roster_obj = next(
        (r for r in rosters if str(r.get("roster_id")) == str(viewer_roster_id)),
        {},
    )
    flat_players = []
    for pid in roster_obj.get("players") or []:
        row = by_id.get(str(pid))
        if not row:
            continue
        pos = str(row.get("position") or row.get("pos") or "").upper()
        if pos not in _CORE_POS:
            continue
        try:
            val = float(row.get("value") or 0.0)
        except (TypeError, ValueError):
            val = 0.0
        nm = str(row.get("name") or "").strip().lower()
        flat_players.append({"position": pos, "value": val, "age": name_to_age.get(nm)})
    flat_players.sort(key=lambda x: x["value"], reverse=True)

    picks_by_roster = ctx.get("picks_by_roster") or {}
    picks = picks_by_roster.get(str(_vid), [])
    p_ranks = {pos: pos_rank[pos].get(_vid, n_teams) for pos in _POS_ORDER}

    grade = calculate_roster_grade(
        flat_players,
        picks,
        position_ranks=p_ranks,
        num_teams=n_teams,
        dynasty_pct_val=dynasty_pct(_vid),
        redraft_pct_val=redraft_pct(_vid),
        dr_ratio=team_dr_ratio.get(_vid, 1.0),
        scoring_type=scoring,
    )

    if _is_redraft:
        playoff_pct = None
        for row in (ctx.get("playoff_odds") or []):
            if str((row or {}).get("roster_id")) == str(_vid):
                playoff_pct = (row or {}).get("playoff_pct")
                break
        grade["win_window"] = redraft_window_label(
            playoff_pct=playoff_pct,
            redraft_pct=redraft_pct(_vid),
        )
        grade["scoring_type"] = "redraft"

    return grade
