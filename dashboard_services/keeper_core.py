"""Keeper core + keeper-value presence (Wave 2 PR 4, spec 3.2 check 7).

Keeper core = Wave 1's designated keepers UNION projected keepers.
Kyren Williams is the case that turns on the union: he is keeper core
by projection even though he is not one of the five *designated*
keepers in his league.

Projection uses the real machinery — ``utils.draft.project_league_keepers``
— fed the same way the keeper page feeds it when market ADP is missing
(``dashboard_services/pages/keeper_page.py``): ADP falls back to the
player's overall rank by format value, and undrafted cost falls to the
last round. With uniform costs the surplus optimizer keeps each
roster's highest-value players up to the league's keeper limit, which
is the honest projection available without draft-cost data. When the
optimizer cannot run, the fallback is the same top-N-by-value set
computed directly. Fail-soft throughout: no limit and no designations
means no core.

Keeper *value* note: ``keeper_surplus_value`` is denominated in the
abstract pick-value units of ``utils.draft.pick_value`` (its docstring:
"the exact unit is intentionally abstract"), which are NOT
commensurable with trade values. The keeper blend therefore cannot be
a literal numeric floor across scales; it is enforced where the grade
is stamped — a package that surrenders keeper core for a return with
no keeper value cannot be graded a steal (see the guard in the
player-packages path). "Has keeper value" for a player = membership
in the league's keeper core (designated or projected).
"""
from __future__ import annotations

from typing import Optional

_POSITIONS = ("QB", "RB", "WR", "TE")


def keeper_limit(ctx: dict, is_redraft: bool = False) -> Optional[int]:
    """The league's keeper limit, or None when unknown / not a keeper
    format. Sources: the Wave 1 rules record summary on the ctx, then
    league settings."""
    if is_redraft or not isinstance(ctx, dict):
        return None
    rules = ctx.get("_league_rules") or {}
    settings = ctx.get("league_settings") or {}
    for src in (rules, settings):
        val = src.get("max_keepers") if isinstance(src, dict) else None
        try:
            n = int(val)
        except (TypeError, ValueError):
            continue
        if n > 0:
            return n
    return None


def is_keeper_format(ctx: dict, is_redraft: bool = False) -> bool:
    """Keeper/dynasty league? Redraft never is. A known keeper limit or
    known designations make it one; otherwise the league's dynasty
    scoring type does (dynasty rosters are all keepers by definition).
    """
    if is_redraft or not isinstance(ctx, dict):
        return False
    if keeper_limit(ctx) is not None:
        return True
    try:
        from utils.league_rules import keeper_locked_by_roster
        if keeper_locked_by_roster(ctx):
            return True
    except Exception:
        pass
    try:
        from dashboard_services.ai.context_builders import ctx_scoring_type
        return ctx_scoring_type(ctx) == "dynasty"
    except Exception:
        return False


def projected_keepers_by_roster(
    rosters: list,
    values_by_id: dict,
    limit: int,
    league_size: int = 12,
) -> dict:
    """{roster_id: [player_id, ...]} projected keepers via
    ``project_league_keepers`` (value-rank ADP fallback, per the
    keeper page). Fallback: top-``limit`` by format value per roster.
    """
    limit = int(limit or 0)
    if limit <= 0 or not rosters:
        return {}
    try:
        from utils.draft import KeeperCandidate, KeeperRules
        from utils.draft import project_league_keepers as _project
        ranked = sorted(
            ((pid, float((info or {}).get("value") or 0))
             for pid, info in (values_by_id or {}).items()
             if float((info or {}).get("value") or 0) > 0),
            key=lambda kv: -kv[1],
        )
        value_rank = {pid: float(i + 1) for i, (pid, _v) in enumerate(ranked)}
        per_team: dict = {}
        for roster in rosters or []:
            if not isinstance(roster, dict):
                continue
            rid = str(roster.get("roster_id"))
            cands = []
            for pid in (roster.get("players") or []):
                pid = str(pid)
                info = values_by_id.get(pid) or {}
                if float(info.get("value") or 0) <= 0:
                    continue
                cands.append(KeeperCandidate(
                    player_id=pid,
                    name=str(info.get("name") or pid),
                    position=str(info.get("position") or "").upper(),
                    drafted_round=None,
                    years_kept=0,
                    adp_overall=value_rank.get(pid),
                    value=float(info.get("value") or 0),
                ))
            if cands:
                per_team[rid] = cands
        if not per_team:
            return {}
        rules = KeeperRules(
            league_size=max(int(league_size or 12), 1),
            num_rounds=25,
            escalation=0,
            undrafted_round=25,
            one_per_round=False,
        )
        projected = _project(per_team, rules, limit)
        out = {str(rid): [str(p) for p in (ids or [])]
               for rid, ids in (projected or {}).items()}
        if any(out.values()):
            return out
    except Exception:
        pass
    # Fallback: top-limit by format value per roster.
    out = {}
    for roster in rosters or []:
        if not isinstance(roster, dict):
            continue
        rid = str(roster.get("roster_id"))
        vals = sorted(
            ((str(pid), float((values_by_id.get(str(pid)) or {})
                              .get("value") or 0))
             for pid in (roster.get("players") or [])),
            key=lambda kv: -kv[1],
        )
        out[rid] = [pid for pid, v in vals if v > 0][:limit]
    return out


def keeper_core(ctx: dict, rosters: list, values_by_id: dict,
                is_redraft: bool = False,
                league_size: int = 12) -> dict:
    """The keeper-core picture for one league. Returns::

        {"active": bool,           # keeper/dynasty format?
         "core_ids": set[str],     # designated UNION projected
         "designated_ids": set[str],
         "projected_ids": set[str],
         "limit": int | None}

    In a keep-everything dynasty league with no stated limit, the
    projection limit defaults to 8 per roster (a dynasty core is its
    top assets; the guard only needs to know whether the return
    carries any of them).
    """
    result = {"active": False, "core_ids": set(), "designated_ids": set(),
              "projected_ids": set(), "limit": None}
    if not is_keeper_format(ctx, is_redraft=is_redraft):
        return result
    result["active"] = True
    designated: set = set()
    try:
        from utils.league_rules import keeper_locked_by_roster
        for ids in (keeper_locked_by_roster(ctx) or {}).values():
            designated.update(str(p) for p in (ids or ()))
    except Exception:
        designated = set()
    limit = keeper_limit(ctx, is_redraft=is_redraft)
    result["limit"] = limit
    eff_limit = limit if limit is not None else 8
    projected: set = set()
    try:
        by_roster = projected_keepers_by_roster(
            rosters, values_by_id, eff_limit, league_size=league_size)
        for ids in by_roster.values():
            projected.update(str(p) for p in (ids or []))
    except Exception:
        projected = set()
    result["designated_ids"] = designated
    result["projected_ids"] = projected
    result["core_ids"] = designated | projected
    return result


def has_keeper_value(player_id, core: dict) -> bool:
    """Does this player carry keeper value in this league? Membership
    in the keeper core (designated or projected)."""
    if not isinstance(core, dict) or not core.get("active"):
        return False
    return str(player_id) in (core.get("core_ids") or set())


__all__ = [
    "keeper_limit",
    "is_keeper_format",
    "projected_keepers_by_roster",
    "keeper_core",
    "has_keeper_value",
]
