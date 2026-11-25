"""Autopilot execution service: run an approved plan against ESPN.

One entry point, :func:`execute_plan`, used by Approve mode (the user
accepted specific swap groups) and Auto mode (the run itself is the
authorization, so every group in the plan executes). The check sequence
follows the Phase 1 plan Step 5 exactly; every guardrail is re-checked at
execution time against fresh reads, never trusted from plan time:

1. Settings re-read: access gate, mode matches the call, league not
   paused, account kill switch off, lineups enabled.
2. Live roster re-read: if it already equals the plan's intended final
   state, the run is an idempotent no-op ("already_applied"), not an
   error. If it differs from the plan's snapshot in any other way, the
   plan is discarded as superseded (guardrail 9) and nothing is written.
3. Per-group revalidation (guardrail 6): every item's player is still
   rostered at the planned from_slot, is not lineup-locked now, is
   eligible for the target slot, and no group benches a protected
   player. A group that fails revalidation is audited "failed" and
   excluded; the rest still execute. Nothing is ever silently swapped
   for another move.
4. Credentials resolve for the acting account only (inside
   espn_writes); one ROSTER/LINEUP transaction carries all surviving
   groups' items. A rejection stops everything: audit "failed" with
   ESPN's words verbatim, no retry, no fallback (guardrail 8).
5. Verify by re-read (guardrail 7): the post-write roster must equal the
   intended final state for every submitted player. A mismatch is a
   failure and PAUSES the league (spec failure handling); the audit
   shows both states.

Every step writes append-only audit entries (guardrail 10). The return
value reports per-group outcomes for the caller (the UI PR / notifier).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set

from dashboard_services import autopilot_preview, autopilot_store as store
from dashboard_services.providers import espn_writes

logger = logging.getLogger(__name__)

BENCH_SLOT = espn_writes.BENCH_SLOT
IR_SLOT = espn_writes.IR_SLOT


def _intended_final_state(plan: Dict[str, Any], groups: Sequence[dict]) -> Dict[int, int]:
    final = {int(k): int(v) for k, v in (plan.get("snapshot") or {}).items()}
    for group in groups:
        for item in group["items"]:
            final[int(item["player_id"])] = int(item["to_slot"])
    return final


def _audit_group(account_id, platform, league_id, season, plan, action, group,
                 *, detail=None, verification=None, espn_status=None):
    for entry in group["in"] or group["items"]:
        store.record_audit(
            account_id, platform, league_id, season, action,
            plan_id=plan["id"], scoring_period=plan["scoring_period"],
            mode=plan.get("mode"), player_id=entry["player_id"],
            player_name=entry.get("name"),
            from_slot=entry.get("from_slot"), to_slot=entry.get("to_slot"),
            score_delta=group.get("delta"), detail=detail,
            verification=verification, espn_status=espn_status,
        )


def execute_plan(account_id, platform, league_id, season, plan_id, *,
                 accepted_groups: Optional[Sequence[int]] = None,
                 transport=None) -> Dict[str, Any]:
    """Execute a stored plan. Returns a per-group outcome report.

    ``accepted_groups``: indexes into the plan's swap list. Required in
    Approve mode (only accepted groups run; the rest are audited
    "skipped"). Ignored in Auto mode, where the whole plan runs.
    """
    result: Dict[str, Any] = {
        "status": "refused", "plan_id": plan_id, "groups": [], "detail": None,
    }

    # --- Gate + settings (before any credential resolution) -------------
    if not autopilot_preview.autopilot_access(account_id, platform, league_id, season):
        result["detail"] = "Autopilot access gate refused this account/league."
        return result
    settings = store.get_settings(account_id, platform, league_id, season)
    mode = settings["mode"]
    if mode == "off":
        result["detail"] = "Autopilot is off for this league."
        return result
    if store.get_kill_switch(account_id):
        result["detail"] = "The account-wide Autopilot kill switch is on."
        result["status"] = "blocked_kill_switch"
        return result
    if settings["paused"]:
        result["detail"] = "Autopilot is paused for this league."
        result["status"] = "blocked_paused"
        return result
    if not settings["lineups_enabled"]:
        result["detail"] = "Lineups are disabled in this league's Autopilot settings."
        return result

    plan = store.get_plan(plan_id)
    if plan is None or int(plan["account_id"]) != int(account_id) \
            or str(plan["league_id"]) != str(league_id) \
            or str(plan["platform"]) != str(platform):
        result["detail"] = "Plan not found for this account/league."
        return result
    if plan["status"] != "open":
        result["detail"] = f"Plan is {plan['status']}, not open."
        return result

    swaps: List[dict] = plan.get("swaps") or []
    if mode == "approve":
        accepted_idx = set(accepted_groups or [])
        chosen = [g for i, g in enumerate(swaps) if i in accepted_idx]
        skipped = [g for i, g in enumerate(swaps) if i not in accepted_idx]
        for group in skipped:
            _audit_group(account_id, platform, league_id, season, plan,
                         "skipped", group, detail="Not accepted by the user.")
        for group in chosen:
            _audit_group(account_id, platform, league_id, season, plan,
                         "approved", group, detail="Accepted by the user.")
    else:  # auto: the run is the authorization
        chosen = list(swaps)
    if not chosen:
        result["status"] = "nothing_to_do"
        result["detail"] = "No swap groups selected for execution."
        store.set_plan_status(plan["id"], "executed")
        return result

    # --- Fresh roster: idempotency + snapshot guard (guardrail 9) ------
    state = espn_writes.read_lineup_state(
        account_id, league_id, season,
        scoring_period_id=int(plan["scoring_period"]), transport=transport,
    )
    current = state["snapshot"]
    intended_all = _intended_final_state(plan, chosen)
    if all(current.get(pid) == slot for pid, slot in intended_all.items()) \
            and set(current) == set(intended_all):
        store.set_plan_status(plan["id"], "already_applied")
        result["status"] = "already_applied"
        result["detail"] = "The roster already reflects this plan; no write sent."
        result["groups"] = [
            {"index": swaps.index(g), "outcome": "already_applied"} for g in chosen
        ]
        return result
    plan_snapshot = {int(k): int(v) for k, v in (plan.get("snapshot") or {}).items()}
    if current != plan_snapshot:
        store.set_plan_status(plan["id"], "superseded")
        for group in chosen:
            _audit_group(
                account_id, platform, league_id, season, plan, "failed", group,
                detail="Roster changed since the plan was built; plan discarded, replan needed.",
                verification="snapshot_mismatch",
            )
        result["status"] = "superseded"
        result["detail"] = "Roster changed since planning; plan discarded, replan needed."
        return result

    # --- Per-group revalidation (guardrail 6 + protected list) ----------
    entries = {e["player_id"]: e for e in state["entries"]}
    protected: Set[int] = set(settings["protected_player_ids"] or [])
    live_groups: List[dict] = []
    outcomes: List[Dict[str, Any]] = []
    for group in chosen:
        problem = None
        for item in group["items"]:
            entry = entries.get(int(item["player_id"]))
            if entry is None:
                problem = "Player is no longer on the roster."
                break
            if int(entry["slot"]) != int(item["from_slot"]):
                problem = "Player's slot changed since planning."
                break
            if entry["locked"]:
                problem = "Player's game has started; the player is locked."
                break
            if int(item["to_slot"]) != BENCH_SLOT \
                    and int(item["to_slot"]) not in entry["eligible_slots"]:
                problem = "Player is not eligible for the target slot."
                break
            if int(item["to_slot"]) == BENCH_SLOT \
                    and int(item["player_id"]) in protected:
                problem = "Protected players are never benched by Autopilot."
                break
        if problem:
            _audit_group(account_id, platform, league_id, season, plan,
                         "failed", group, detail=problem,
                         verification="revalidation_failed")
            outcomes.append({
                "index": swaps.index(group), "outcome": "failed",
                "detail": problem,
            })
        else:
            live_groups.append(group)

    if not live_groups:
        store.set_plan_status(plan["id"], "failed")
        result["status"] = "failed"
        result["detail"] = "Every selected move failed revalidation; nothing was written."
        result["groups"] = outcomes
        return result

    # --- One transaction for all surviving groups ------------------------
    items = [item for group in live_groups for item in group["items"]]
    try:
        submitted = espn_writes.submit_lineup(
            account_id, league_id, season, int(plan["team_id"]),
            int(plan["scoring_period"]), items, transport=transport,
        )
    except espn_writes.EspnWriteRejected as exc:
        for group in live_groups:
            _audit_group(
                account_id, platform, league_id, season, plan, "failed", group,
                detail=f"ESPN rejected the lineup write: {exc}",
                espn_status=str(exc.espn_status or exc.http_status),
            )
            outcomes.append({
                "index": swaps.index(group), "outcome": "failed",
                "detail": str(exc),
            })
        store.set_plan_status(plan["id"], "failed")
        result["status"] = "rejected"
        result["detail"] = f"ESPN rejected the lineup write: {exc}"
        result["groups"] = outcomes
        return result

    # --- Verify by re-read (guardrail 7) ---------------------------------
    intended = _intended_final_state(plan, live_groups)
    ok, actual = espn_writes.verify_lineup(
        account_id, league_id, season, intended,
        scoring_period_id=int(plan["scoring_period"]), transport=transport,
    )
    if not ok:
        store.set_paused(account_id, platform, league_id, season, True)
        for group in live_groups:
            _audit_group(
                account_id, platform, league_id, season, plan, "failed", group,
                detail="Verification mismatch: the re-read roster does not "
                       "match the intended state. League paused.",
                verification=f"expected {intended}, actual {actual}",
                espn_status=str(submitted.get("status") or ""),
            )
            outcomes.append({
                "index": swaps.index(group), "outcome": "failed",
                "detail": "Verification mismatch; league paused.",
            })
        store.set_plan_status(plan["id"], "failed")
        result["status"] = "verification_failed"
        result["detail"] = "The write was accepted but verification failed; league paused."
        result["groups"] = outcomes
        return result

    for group in live_groups:
        _audit_group(
            account_id, platform, league_id, season, plan, "executed", group,
            detail="Lineup write executed and verified by re-read.",
            verification="verified",
            espn_status=str(submitted.get("status") or ""),
        )
        outcomes.append({"index": swaps.index(group), "outcome": "executed"})

    failed_any = any(o["outcome"] == "failed" for o in outcomes)
    store.set_plan_status(plan["id"], "partial" if failed_any else "executed")
    result["status"] = "partial" if failed_any else "executed"
    result["groups"] = outcomes
    return result
