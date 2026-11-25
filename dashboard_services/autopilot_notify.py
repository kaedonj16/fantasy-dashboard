"""Autopilot notifications: push + email on executed and failed moves.

Events (Phase 1 plan Step 6):
- A run executed moves: what moved, and that it was verified by re-read.
- A run failed (ESPN rejection, verification mismatch, superseded plan,
  revalidation failures): the reason, verbatim from the execute result.
- A scheduled Approve run queued a plan with moves waiting.

Channels are gated by the league's stored per-league preferences
(autopilot_store settings notify_push / notify_email; both default on,
Kaedon's 2026-11-24 decision). Push goes through the existing account
broadcast in utils.push_notifications; email through utils.email to the
account's own address.

Nothing in here may fail an execution: every send is wrapped, failures
are logged, and the functions return a small report instead of raising.
"""
from __future__ import annotations

import html
import logging
from typing import Any, Dict, List, Optional

from dashboard_services import autopilot_store as store

logger = logging.getLogger(__name__)


def _page_url(platform, season, league_id) -> str:
    return f"/{platform}/{season}/{league_id}/autopilot"


def _group_label(plan: Dict[str, Any], index: int) -> str:
    """Human label for one swap group: 'Start A over B'."""
    swaps = (plan or {}).get("swaps") or []
    if index is None or index >= len(swaps):
        return "Lineup move"
    group = swaps[index]
    ins = ", ".join(str(e.get("name") or "a player") for e in group.get("in") or [])
    outs = ", ".join(str(e.get("name") or "a player") for e in group.get("out") or [])
    if ins and outs:
        return f"Start {ins} over {outs}"
    if ins:
        return f"Start {ins}"
    if outs:
        return f"Bench {outs}"
    return "Lineup move"


def _account_email(account_id) -> Optional[str]:
    try:
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            row = conn.execute(
                "SELECT email FROM accounts WHERE id = %s", (int(account_id),)
            ).fetchone()
        if row:
            return str(row["email"] or "").strip() or None
    except Exception:
        logger.debug("autopilot notify: account email lookup failed", exc_info=True)
    return None


def _send(account_id, platform, league_id, season, settings, title, lines) -> Dict[str, Any]:
    """Dispatch one notification over the enabled channels. Never raises."""
    report = {"push": False, "email": False}
    body_text = "\n".join(lines)
    url = _page_url(platform, season, league_id)
    if settings.get("notify_push", True):
        try:
            from utils.push_notifications import _broadcast_account

            sent = _broadcast_account(
                str(account_id), title, body_text, url,
                tag="autopilot", league_id=str(league_id),
                platform=str(platform),
            )
            report["push"] = bool(sent)
        except Exception:
            logger.warning("autopilot push failed", exc_info=True)
    if settings.get("notify_email", True):
        try:
            email = _account_email(account_id)
            if email:
                from utils.email import send_email

                items = "".join(
                    f"<li>{html.escape(line)}</li>" for line in lines
                )
                result = send_email(
                    email,
                    title,
                    f"<p><strong>{html.escape(title)}</strong></p>"
                    f"<ul>{items}</ul>"
                    f"<p>Details and the full audit log are on your "
                    f"Autopilot page.</p>",
                    text=f"{title}\n\n{body_text}",
                    tags=["autopilot"],
                )
                report["email"] = bool(getattr(result, "ok", False))
        except Exception:
            logger.warning("autopilot email failed", exc_info=True)
    return report


def notify_execution(account_id, platform, league_id, season,
                     result: Dict[str, Any], plan: Optional[Dict[str, Any]] = None
                     ) -> Dict[str, Any]:
    """Notify after an execute_plan run (manual Accept or scheduled Auto).

    Silent when nothing executed and nothing failed. Never raises.
    """
    try:
        groups = (result or {}).get("groups") or []
        executed = [g for g in groups if g.get("outcome") == "executed"]
        applied = [g for g in groups if g.get("outcome") == "already_applied"]
        failed = [g for g in groups if g.get("outcome") == "failed"]
        status = (result or {}).get("status")
        if not executed and not failed and status not in (
            "rejected", "verification_failed",
        ):
            return {"push": False, "email": False, "sent": False}
        settings = store.get_settings(account_id, platform, league_id, season)
        plan = plan or {}
        lines: List[str] = []
        for g in executed:
            lines.append(
                f"{_group_label(plan, g.get('index'))}. "
                "Executed and verified by re-reading your roster."
            )
        for g in applied:
            lines.append(
                f"{_group_label(plan, g.get('index'))}. "
                "Already in place; no write was needed."
            )
        for g in failed:
            detail = g.get("detail") or "The move failed."
            lines.append(f"{_group_label(plan, g.get('index'))}. Failed: {detail}")
        if status == "rejected" and not failed and result.get("detail"):
            lines.append(f"ESPN rejected the lineup write: {result['detail']}")
        if status == "verification_failed":
            lines.append(
                "The write was accepted but verification failed, so this "
                "league's Autopilot is paused. Check the Autopilot page."
            )
        if executed or applied:
            title = "Autopilot: lineup updated"
        else:
            title = "Autopilot: a move needs your attention"
        report = _send(account_id, platform, league_id, season, settings, title, lines)
        report["sent"] = True
        return report
    except Exception:
        logger.warning("autopilot notify_execution failed", exc_info=True)
        return {"push": False, "email": False, "sent": False}


def notify_plan_waiting(account_id, platform, league_id, season,
                        plan: Dict[str, Any]) -> Dict[str, Any]:
    """Notify that a scheduled Approve run queued moves. Never raises."""
    try:
        swaps = (plan or {}).get("swaps") or []
        if not swaps:
            return {"push": False, "email": False, "sent": False}
        settings = store.get_settings(account_id, platform, league_id, season)
        lines = [
            f"{_group_label(plan, i)} (+{g.get('delta'):.1f} start score)"
            if isinstance(g.get("delta"), (int, float))
            else _group_label(plan, i)
            for i, g in enumerate(swaps)
        ]
        title = (
            f"Autopilot: {len(swaps)} lineup move"
            f"{'s' if len(swaps) != 1 else ''} waiting for your approval"
        )
        report = _send(account_id, platform, league_id, season, settings, title, lines)
        report["sent"] = True
        return report
    except Exception:
        logger.warning("autopilot notify_plan_waiting failed", exc_info=True)
        return {"push": False, "email": False, "sent": False}
