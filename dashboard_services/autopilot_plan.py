"""Autopilot plan builder: real lineup plans from the site's own engines.

A plan is the diff between the team's current ESPN lineup (fresh,
cache-free read through :mod:`dashboard_services.providers.espn_writes`)
and the optimal lineup from ``utils.lineups.assign_optimal_lineup``, fed
by the blended start score exactly as the Start/Sit pool computes it
(``utils.start_sit_score.compute_start_score`` over the daily
start-score bundles, with the projection cap, return-from-absence shade,
and handcuff lead adjustments the pool applies; the matchup factor comes
from the same rank table the pool uses, neutral on any failure, which is
the pool's own fallback rule). No new formulas live here.

Guardrails applied at plan time (re-checked again at execution):

- Locked players (ESPN ``lineupLocked``) are pinned: their slot instance
  leaves the optimizer entirely, so no move can touch them (guardrail 6).
- Protected players are never benched by a plan (Kaedon's Phase 1
  decision): a protected starter is pinned to his slot. A protected bench
  player may still be promoted; protection binds benching and drops only.
- Starters who cannot be mapped to a canonical player or have no
  start-score bundle are pinned rather than guessed about; unscored bench
  players are simply never promoted.
- ``problem_slots_only`` (the Auto scope setting): only swap groups that
  fill an empty slot or bench an empty/bye/injured problem starter are
  kept (guardrail 5).

Plans are keyed to ESPN's ``scoringPeriodId`` from the roster read, never
the site's week number (plan doc risk note). For ESPN the scoring period
is the NFL week, which is also the start-score bundle week; the site's
own start/sit code treats them interchangeably
(``espn_api._espn_current_scoring_week``).

One open plan per league: persisting a plan supersedes the previous open
plan (guardrail 9). Every plan build writes a "planned" audit entry per
swap group.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set

from dashboard_services import autopilot_store as store
from dashboard_services.providers import espn_writes

logger = logging.getLogger(__name__)

BENCH_SLOT = espn_writes.BENCH_SLOT
IR_SLOT = espn_writes.IR_SLOT

#: ESPN injuryStatus -> the site's status vocabulary (find_lineup_issues /
#: start_sit_pool_excluded speak Sleeper-style statuses).
_ESPN_INJURY_MAP = {
    "ACTIVE": "",
    "PROBABLE": "",
    "QUESTIONABLE": "Questionable",
    "DAY_TO_DAY": "Questionable",
    "DOUBTFUL": "Doubtful",
    "OUT": "Out",
    "INJURY_RESERVE": "IR",
    "SUSPENSION": "Suspended",
    "PUP": "PUP",
    "NFI": "NFI",
}

#: Canonical slot name -> ESPN lineupSlotId (espn_api._ESPN_LINEUP_SLOT_IDS
#: inverted through _ESPN_SLOT_TO_SLEEPER). Used when a slot instance has
#: no current occupant to learn its id from (empty slots).
_CANONICAL_SLOT_IDS = {
    "QB": 0, "RB": 2, "RB_WR": 3, "WR": 4, "WR_TE": 5, "TE": 6,
    "SUPER_FLEX": 7, "DEF": 16, "K": 17, "FLEX": 23,
}

_PLAN_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")


def _slot_canonical_name(slot_id: int) -> str:
    """Canonical slot name for an ESPN lineupSlotId."""
    try:
        from dashboard_services.providers.espn_api import (
            _ESPN_LINEUP_SLOT_IDS,
            _map_espn_slot_name,
        )

        label = _ESPN_LINEUP_SLOT_IDS.get(int(slot_id))
        if label:
            return _map_espn_slot_name(label) or label
    except Exception:
        pass
    for name, sid in _CANONICAL_SLOT_IDS.items():
        if sid == int(slot_id):
            return name
    return ""


def _def_rank(season: int, pos: str, opponent) -> tuple:
    """Matchup rank (1 = easiest) + defense count, pool-identical source.

    Any failure leaves the factor neutral (None, 32), the same fallback
    the Start/Sit pool applies.
    """
    if not opponent:
        return None, 32
    try:
        from app import _matchup_rank_table, _norm_sched_team

        rank_map, total, _ri, _iz = _matchup_rank_table(int(season), pos)
        rank_map = rank_map or {}
        rank = rank_map.get(_norm_sched_team(opponent)) or rank_map.get(opponent)
        return rank, total or 32
    except Exception:
        return None, 32


def _player_score(canon_pid: str, bundles: Dict[str, dict], season: int,
                  live_status: Optional[str]) -> Optional[Dict[str, Any]]:
    """One player's blended start score from his bundle, pool-identical.

    Returns {"score", "pos", "excluded"} or None when there is no bundle
    (no honest score exists). ``excluded`` marks the IR/Out/PUP/NFI pool
    exclusion; such a player scores 0.0 so the optimizer benches him, and
    callers must never promote him from the bench.
    """
    bundle = bundles.get(str(canon_pid))
    if not isinstance(bundle, dict):
        return None
    from utils.start_sit_score import (
        cap_projection,
        compute_start_score,
        lead_adjusted_last4_ppg,
        lead_adjusted_score_inputs,
        return_adjusted_projection,
        start_sit_pool_excluded,
    )

    pos = str(bundle.get("pos") or "").upper()
    if pos not in _PLAN_POSITIONS:
        return None
    status = live_status or bundle.get("injury_status")
    if start_sit_pool_excluded(status):
        return {"score": 0.0, "pos": pos, "excluded": True}

    proj = cap_projection(bundle.get("proj_pts") or 0.0, bundle.get("exp_last4_ppg"))
    proj = return_adjusted_projection(proj, bundle.get("absence_weeks"))
    last4 = bundle.get("last4_ppg")
    recent = bundle.get("recent_ppg") or 0.0
    season_ppg = bundle.get("season_ppg") or 0.0
    usage = bundle.get("usage_delta")
    trend = bundle.get("snap_trend")

    lead_pid = bundle.get("lead_pid")
    if lead_pid:
        lead_bundle = bundles.get(str(lead_pid)) or {}
        lead_status = lead_bundle.get("injury_status")
        lead_bye = (bool(lead_bundle.get("on_bye")) if lead_bundle
                    else bool(bundle.get("on_bye")))
        last4 = lead_adjusted_last4_ppg(
            last4, bundle.get("last4_ppg_lead"), lead_pid,
            lead_status=lead_status, lead_on_bye=lead_bye)
        (proj, recent, season_ppg, usage, trend) = lead_adjusted_score_inputs(
            proj, recent, season_ppg, usage, trend,
            season_ppg_lead=bundle.get("season_ppg_lead"),
            recent_ppg_lead=bundle.get("recent_ppg_lead"),
            usage_delta_lead=bundle.get("usage_delta_lead"),
            snap_trend_lead=bundle.get("snap_trend_lead"),
            lead_pid=lead_pid, lead_status=lead_status, lead_on_bye=lead_bye)

    def_rank, def_total = _def_rank(season, pos, bundle.get("opponent"))
    try:
        score, _factors, _demotion = compute_start_score(
            proj,
            on_bye=bool(bundle.get("on_bye")),
            recent_ppg=recent,
            season_ppg=season_ppg,
            last4_ppg=last4,
            def_rank=def_rank,
            def_total=def_total,
            usage_delta=usage,
            usage_season_avg=bundle.get("usage_season_avg"),
            injury_status=status,
            played_games=bundle.get("played_games"),
            implied_total=bundle.get("implied_total"),
            bust_rate=bundle.get("bust_rate"),
            weather_kind=bundle.get("weather_kind"),
            position=pos,
            oline_index=bundle.get("oline_index"),
            expected_team_plays=bundle.get("expected_team_plays"),
            league_average_plays=bundle.get("league_average_plays"),
            role_confidence=bundle.get("role_confidence"),
            snap_trend=trend,
            low_total_team=bool(bundle.get("low_total_team")),
        )
    except Exception:
        logger.debug("autopilot start score failed pid=%s", canon_pid, exc_info=True)
        return None
    return {"score": float(score or 0.0), "pos": pos, "excluded": False}


def _league_globals_from_payload(payload: dict) -> Dict[str, Any]:
    """Scoring settings + canonical roster positions from the raw payload.

    Mirrors espn_api.get_league_globals, but sourced from the payload the
    plan already read with the acting account's credentials (get_league_
    globals resolves credentials through the Flask session instead, which
    a cron / service caller does not have).
    """
    from dashboard_services.providers.espn_api import (
        expand_espn_lineup_slot_counts,
        normalize_espn_scoring_items,
    )
    from utils.league_scoring import normalize_league_scoring

    settings_node = payload.get("settings") or {}
    scoring_items = (
        (settings_node.get("scoringSettings") or {}).get("scoringItems") or []
    )
    scoring_settings: Dict[str, Any] = {
        "rec": 0.0, "pass_yd": 0.04, "pass_td": 4.0, "pass_int": -2.0,
        "rush_yd": 0.1, "rush_td": 6.0, "rec_yd": 0.1, "rec_td": 6.0,
        "fum_lost": -2.0, "2pt": 2.0, "fg_0_19": 3.0, "fg_20_29": 3.0,
        "fg_30_39": 3.0, "fg_40_49": 4.0, "fg_50p": 5.0, "xpt": 1.0,
    }
    scoring_settings.update(normalize_espn_scoring_items(scoring_items))
    try:
        scoring_settings = normalize_league_scoring("espn", scoring_settings)
    except Exception:
        logger.debug("autopilot normalize_league_scoring failed", exc_info=True)
    slot_counts = (settings_node.get("rosterSettings") or {}).get("lineupSlotCounts")
    roster_positions = expand_espn_lineup_slot_counts(slot_counts)
    return {"scoring_settings": scoring_settings, "roster_positions": roster_positions}


def _group_items(items: List[Dict[str, Any]], by_pid: Dict[int, dict]) -> List[Dict[str, Any]]:
    """Group LINEUP items into swap units (connected via starter slots).

    A direct swap, and any chain that runs through starter slots (bench
    player takes a slot whose occupant reshuffles into another slot whose
    occupant goes to the bench), is one unit: it is the smallest set of
    items that can be accepted or executed independently.
    """
    parent = {it["player_id"]: it["player_id"] for it in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a in items:
        for b in items:
            if a is b:
                continue
            if a["to_slot"] == b["from_slot"] and a["to_slot"] != BENCH_SLOT:
                union(a["player_id"], b["player_id"])
            if a["from_slot"] == b["to_slot"] and a["from_slot"] != BENCH_SLOT:
                union(a["player_id"], b["player_id"])

    groups: Dict[int, List[Dict[str, Any]]] = {}
    for it in items:
        groups.setdefault(find(it["player_id"]), []).append(it)

    swaps = []
    for group_items in groups.values():
        ins, outs = [], []
        delta = 0.0
        for it in group_items:
            info = by_pid[it["player_id"]]
            entry = {
                "player_id": it["player_id"],
                "canon_id": info.get("canon_id"),
                "name": info.get("name"),
                "pos": info.get("pos"),
                "score": info.get("score"),
                "from_slot": it["from_slot"],
                "to_slot": it["to_slot"],
            }
            if it["from_slot"] == BENCH_SLOT and it["to_slot"] != BENCH_SLOT:
                ins.append(entry)
                delta += float(info.get("score") or 0.0)
            elif it["to_slot"] == BENCH_SLOT and it["from_slot"] != BENCH_SLOT:
                outs.append(entry)
                delta -= float(info.get("score") or 0.0)
        swaps.append({
            "items": group_items,
            "in": ins,
            "out": outs,
            "delta": round(delta, 2),
        })
    swaps.sort(key=lambda s: -s["delta"])
    return swaps


def build_plan(account_id, platform, league_id, season, *, transport=None,
               persist: bool = True) -> Dict[str, Any]:
    """Build (and by default persist) the current Autopilot plan.

    Returns a dict with a ``status``: "off" / "paused" / "kill_switch"
    (nothing planned), "no_moves" (plan persisted, lineup already
    optimal), or "planned" (plan persisted with swap groups). The plan
    dict itself is under ``plan`` when one was persisted.
    """
    settings = store.get_settings(account_id, platform, league_id, season)
    mode = settings["mode"]
    if mode == "off":
        return {"status": "off", "swaps": []}
    if store.get_kill_switch(account_id):
        return {"status": "kill_switch", "swaps": []}
    if settings["paused"]:
        return {"status": "paused", "swaps": []}
    if not settings["lineups_enabled"]:
        return {"status": "lineups_disabled", "swaps": []}

    from dashboard_services.providers.espn_api import (
        _espn_to_canon_cached,
        resolve_espn_player_id,
    )
    from utils.lineups import assign_optimal_lineup, find_lineup_issues, starting_slots

    state = espn_writes.read_lineup_state(
        account_id, league_id, season, transport=transport
    )
    payload = state["payload"]
    period = state["scoring_period_id"]
    entries = state["entries"]
    globals_ = _league_globals_from_payload(payload)
    roster_positions = globals_["roster_positions"]
    if not roster_positions:
        return {"status": "no_roster_positions", "swaps": []}
    slots = starting_slots(roster_positions)

    # --- Canonical mapping + scores -------------------------------------
    espn_to_canon = _espn_to_canon_cached()
    canon_of: Dict[int, Optional[str]] = {}
    for e in entries:
        try:
            canon_of[e["player_id"]] = resolve_espn_player_id(
                e["player_id"], espn_to_canon
            )
        except Exception:
            canon_of[e["player_id"]] = None

    canon_pids = [c for c in canon_of.values() if c]
    bundles: Dict[str, dict] = {}
    if canon_pids:
        from data_building.start_score_bundle import (
            load_start_score_bundles_for_scoring,
        )

        bundles = load_start_score_bundles_for_scoring(
            int(season), int(period), globals_["scoring_settings"], canon_pids
        ) or {}

    protected: Set[int] = set(settings["protected_player_ids"] or [])
    by_pid: Dict[int, dict] = {}
    for e in entries:
        canon = canon_of[e["player_id"]]
        live_status = _ESPN_INJURY_MAP.get(
            str(e.get("injury_status") or "").upper(), e.get("injury_status") or ""
        )
        scored = _player_score(canon, bundles, int(season), live_status) if canon else None
        bundle = bundles.get(str(canon)) if canon else None
        by_pid[e["player_id"]] = {
            "entry": e,
            "canon_id": canon,
            "name": e["name"],
            "pos": (scored or {}).get("pos") or (bundle or {}).get("pos") or e.get("position"),
            "score": (scored or {}).get("score"),
            "excluded": bool((scored or {}).get("excluded")),
            "on_bye": bool((bundle or {}).get("on_bye")),
            "team": str((bundle or {}).get("team") or ""),
            "injury_status": live_status or (bundle or {}).get("injury_status") or "",
        }

    # --- Slot instances and the current assignment ----------------------
    # Instances are the ordered starting slots. Each instance learns its
    # ESPN slot id from its current occupant when it has one.
    instances: List[Dict[str, Any]] = []
    by_slot_name: Dict[str, List[dict]] = {}
    for e in entries:
        if e["slot"] in (BENCH_SLOT, IR_SLOT):
            continue
        by_slot_name.setdefault(_slot_canonical_name(e["slot"]), []).append(e)
    for name in slots:
        pool = by_slot_name.get(name) or []
        occupant = pool.pop(0) if pool else None
        instances.append({
            "name": name,
            "espn_slot_id": (occupant["slot"] if occupant is not None
                             else _CANONICAL_SLOT_IDS.get(name)),
            "occupant": occupant,
        })

    def _pinned(entry: dict) -> bool:
        if entry["locked"]:
            return True
        if entry["player_id"] in protected:
            return True
        info = by_pid[entry["player_id"]]
        return info["canon_id"] is None or info["score"] is None

    # --- Problem slots (find_lineup_issues + bye flags) ------------------
    starters_in_order: List[str] = []
    player_info: Dict[str, dict] = {}
    for inst in instances:
        occ = inst["occupant"]
        starters_in_order.append(str(occ["player_id"]) if occ else "0")
        if occ is not None:
            info = by_pid[occ["player_id"]]
            if info["canon_id"]:
                player_info[info["canon_id"]] = {
                    "name": info["name"],
                    "team": info["team"],
                    "injury_status": info["injury_status"],
                }
    # find_lineup_issues keys player_info by the ids in `starters`; the
    # engine speaks canonical ids, so run it in canonical space.
    canon_starters = [
        (by_pid[int(pid)]["canon_id"] or "0") if pid != "0" else "0"
        for pid in starters_in_order
    ]
    issues = find_lineup_issues(canon_starters, player_info)
    problem_instances: Set[int] = set()
    espn_of_canon = {v: k for k, v in canon_of.items() if v}
    for idx, inst in enumerate(instances):
        occ = inst["occupant"]
        if occ is None:
            problem_instances.add(idx)
            continue
        info = by_pid[occ["player_id"]]
        if info["on_bye"]:
            problem_instances.add(idx)
            issues.append({
                "kind": "bye",
                "pid": info["canon_id"] or str(occ["player_id"]),
                "name": info["name"],
                "detail": f"{info['name']} is on bye",
            })
    issue_pids = {str(i.get("pid")) for i in issues}
    problem_pids: Set[int] = {
        espn_of_canon[p] for p in issue_pids if p in espn_of_canon
    }

    # --- Optimize over the unpinned pool --------------------------------
    unpinned_instances = [
        i for i, inst in enumerate(instances)
        if inst["occupant"] is None or not _pinned(inst["occupant"])
    ]
    opt_slots = [instances[i]["name"] for i in unpinned_instances]
    pts_map: Dict[str, float] = {}
    pos_map: Dict[str, str] = {}
    pool_pids: List[str] = []
    for i in unpinned_instances:
        occ = instances[i]["occupant"]
        if occ is not None:
            info = by_pid[occ["player_id"]]
            pts_map[info["canon_id"]] = float(info["score"])
            pos_map[info["canon_id"]] = info["pos"]
            pool_pids.append(info["canon_id"])
    for e in entries:
        if e["slot"] != BENCH_SLOT:
            continue
        info = by_pid[e["player_id"]]
        if info["canon_id"] is None or info["score"] is None:
            continue
        if info["excluded"] or float(info["score"]) <= 0:
            continue  # never promote an excluded or zero-score bench player
        pts_map[info["canon_id"]] = float(info["score"])
        pos_map[info["canon_id"]] = info["pos"]
        pool_pids.append(info["canon_id"])

    target_slot: Dict[int, int] = {e["player_id"]: e["slot"] for e in entries}
    if opt_slots and pool_pids:
        optimal = assign_optimal_lineup(pts_map, pos_map, opt_slots, pool_pids)
        assigned: Dict[str, int] = {}
        for inst_idx, canon in zip(unpinned_instances, optimal["assignment"]):
            if canon:
                assigned[str(canon)] = instances[inst_idx]["espn_slot_id"]
        assigned_espn = {
            espn_of_canon[c]: slot for c, slot in assigned.items() if c in espn_of_canon
        }
        for pid, slot_id in assigned_espn.items():
            target_slot[pid] = slot_id
        pool_espn = {espn_of_canon[c] for c in pool_pids if c in espn_of_canon}
        for pid in pool_espn:
            if pid not in assigned_espn and by_pid[pid]["entry"]["slot"] != IR_SLOT:
                target_slot[pid] = BENCH_SLOT

    # --- Diff into LINEUP items, group into swaps ------------------------
    items: List[Dict[str, Any]] = []
    for e in entries:
        pid = e["player_id"]
        if target_slot.get(pid, e["slot"]) != e["slot"]:
            items.append({
                "player_id": pid,
                "from_slot": e["slot"],
                "to_slot": target_slot[pid],
            })
    swaps = _group_items(items, by_pid)
    # Pure slot reshuffles (no player enters or leaves the starting
    # lineup, zero score change) are churn, not a plan: drop them. Their
    # instances keep their current occupants, which stays legal and does
    # not affect any surviving group.
    swaps = [s for s in swaps if s["in"] or s["out"]]
    items = [it for swap in swaps for it in swap["items"]]

    if settings["problem_slots_only"]:
        kept = []
        for swap in swaps:
            touches_problem = False
            for it in swap["items"]:
                if it["from_slot"] == BENCH_SLOT and it["to_slot"] != BENCH_SLOT:
                    # Promotion: qualifies when it fills an empty instance
                    # or takes a problem starter's slot instance.
                    for idx in unpinned_instances:
                        if instances[idx]["espn_slot_id"] == it["to_slot"] and (
                            idx in problem_instances
                            or (instances[idx]["occupant"] is not None
                                and instances[idx]["occupant"]["player_id"] in problem_pids)
                        ):
                            touches_problem = True
                if it["to_slot"] == BENCH_SLOT and it["player_id"] in problem_pids:
                    touches_problem = True
            if touches_problem:
                kept.append(swap)
        swaps = kept
        items = [it for swap in swaps for it in swap["items"]]

    plan = {
        "team_id": state["team_id"],
        "scoring_period": period,
        "mode": mode,
        "snapshot": state["snapshot"],
        "swaps": swaps,
        "lineup_items": items,
        "issues": issues,
    }
    if not persist:
        return {"status": "planned" if swaps else "no_moves", "plan": plan}

    saved = store.save_plan(account_id, platform, league_id, season, plan)
    for swap in swaps:
        for entry in swap["in"]:
            store.record_audit(
                account_id, platform, league_id, season, "planned",
                plan_id=saved["id"], scoring_period=period, mode=mode,
                player_id=entry["player_id"], player_name=entry["name"],
                from_slot=entry["from_slot"], to_slot=entry["to_slot"],
                score_delta=swap["delta"],
                detail=f"Start score {entry.get('score')} in, "
                       f"{', '.join(o['name'] for o in swap['out'])} out",
            )
    return {
        "status": "planned" if swaps else "no_moves",
        "plan": saved,
    }
