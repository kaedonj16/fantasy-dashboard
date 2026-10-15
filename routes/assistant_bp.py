"""Assistant API endpoints (Stage 1 grounding spine).

All endpoints require a logged-in account; guests get 401 login_required.
League-context helpers are not needed here directly: the service layer owns
context resolution, so this blueprint stays a thin JSON adapter.
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from dashboard_services.ai.client import AIUnavailableError, AIRateLimitError
from dashboard_services.assistant import service
from dashboard_services.assistant.context import resolve_account_key

logger = logging.getLogger(__name__)

assistant_bp = Blueprint("assistant_bp", __name__)

_CORRECTION_LABELS = {
    "wrong_data": "wrong data",
    "bad_reasoning": "bad reasoning",
    "ignored_situation": "an ignored situation",
    "other": "your correction",
}
_INTENT_LABELS = {
    "start_sit": "start/sit call", "trade_eval": "trade call", "trade_finding": "trade finding",
    "buy_sell_hold": "buy/sell/hold call",
    "player": "player answer", "metric": "metric answer", "leaderboard": "leaderboard answer", "waivers": "waiver answer",
    "this_week": "this-week answer", "clarify": "answer", "out_of_scope": "answer", "remember": "answer",
    "playoffs": "playoff answer", "power_rankings": "power rankings answer",
    "standings": "standings answer", "matchup": "matchup answer", "schedule": "schedule answer",
}


def _require_account():
    key = resolve_account_key()
    if not key:
        return None, (jsonify({"error": "login_required"}), 401)
    return key, None


@assistant_bp.route("/api/assistant/context", methods=["GET"])
def api_assistant_context():
    key, err = _require_account()
    if err:
        return err
    result = service.get_context(
        key,
        platform=request.args.get("platform"),
        league_id=request.args.get("league_id"),
        season=request.args.get("season", type=int),
    )
    return jsonify({"ok": True, "league": result["league"], "caps": result["caps"], "teaser": result["teaser"]})


@assistant_bp.route("/api/assistant/chat", methods=["POST"])
def api_assistant_chat():
    key, err = _require_account()
    if err:
        return err
    body = request.get_json(silent=True) or {}
    message = str(body.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message_required"}), 400
    try:
        result = service.ask(
            key, message,
            platform=body.get("platform"), league_id=body.get("league_id"),
            season=body.get("season"), thread_id=body.get("thread_id"),
            player_id_hint=body.get("player_id"),
        )
    except service.CapReached as exc:
        return jsonify({"error": "cap_reached",
                        "message": "You have used your monthly assistant questions. The cap resets next month.",
                        "caps": exc.caps}), 429
    except (AIUnavailableError, AIRateLimitError):
        return jsonify({"error": "assistant_unavailable",
                        "message": "The assistant is unavailable right now. Please try again shortly."}), 503
    answer = result["answer"]
    payload = {"thread_id": result["thread_id"], "log_id": result["log_id"], "intent": result["intent"],
               "answer": answer, "caps": result["caps"]}
    return jsonify(payload)


@assistant_bp.route("/api/assistant/feedback", methods=["POST"])
def api_assistant_feedback():
    key, err = _require_account()
    if err:
        return err
    body = request.get_json(silent=True) or {}
    if body.get("thumb") not in ("up", "down") or body.get("log_id") is None:
        return jsonify({"error": "invalid_feedback"}), 400
    if not service.set_thumb(body["log_id"], key, body["thumb"]):
        return jsonify({"error": "not_found"}), 404
    return jsonify({"ok": True})


@assistant_bp.route("/api/assistant/correction", methods=["POST"])
def api_assistant_correction():
    key, err = _require_account()
    if err:
        return err
    body = request.get_json(silent=True) or {}
    category = body.get("category")
    if category not in _CORRECTION_LABELS or body.get("log_id") is None:
        return jsonify({"error": "invalid_correction"}), 400
    row = service.get_log(body["log_id"], key)
    if row is None:
        return jsonify({"error": "not_found"}), 404
    result = service.apply_correction(body["log_id"], key, category=category, text=body.get("text"))
    if result is None:
        return jsonify({"error": "not_found"}), 404
    intent_label = _INTENT_LABELS.get(row.get("intent") or "", "answer")
    label = _CORRECTION_LABELS[category]
    # The ack states only what actually happened: a wrong-data correction is
    # re-run against fresh evidence on the spot; anything else is logged and
    # flagged for the weekly review of flagged answers.
    if category == "wrong_data" and result.get("recheck") == "changed":
        ack = (f"Logged under {label} for that {intent_label}. I re-ran it against the "
               "current data and the answer changed, so a revised answer is posted below.")
    elif category == "wrong_data" and result.get("recheck") == "unchanged":
        ack = (f"Logged under {label} for that {intent_label}. I re-ran the numbers and "
               "they still support the original call. Your flag stays on the record for review.")
    elif category == "wrong_data":
        ack = (f"Logged under {label} for that {intent_label}. It is flagged for review "
               "against the data before the same reasoning is reused.")
    elif category == "other":
        ack = f"Logged your correction for that {intent_label}. It goes into the weekly review of flagged answers."
    else:
        ack = f"Logged under {label} for that {intent_label}. It goes into the weekly review of flagged answers."
    return jsonify({"ok": True, "ack": ack, "revision": result.get("revision")})


@assistant_bp.route("/api/assistant/preferences", methods=["GET"])
def api_assistant_preferences_list():
    key, err = _require_account()
    if err:
        return err
    # The panel payload: stored preferences plus fresh inferred
    # proposals and the confirmed risk profile, computed on open.
    return jsonify(service.preference_panel(key))


@assistant_bp.route("/api/assistant/preference-proposals/<int:proposal_id>/confirm", methods=["POST"])
def api_assistant_proposal_confirm(proposal_id):
    key, err = _require_account()
    if err:
        return err
    result = service.confirm_proposal(key, proposal_id)
    if result is None:
        return jsonify({"error": "not_found"}), 404
    return jsonify({"ok": True, **result})


@assistant_bp.route("/api/assistant/preference-proposals/<int:proposal_id>/dismiss", methods=["POST"])
def api_assistant_proposal_dismiss(proposal_id):
    key, err = _require_account()
    if err:
        return err
    if not service.dismiss_proposal(key, proposal_id):
        return jsonify({"error": "not_found"}), 404
    return jsonify({"ok": True})


@assistant_bp.route("/api/assistant/preferences", methods=["POST"])
def api_assistant_preferences_add():
    key, err = _require_account()
    if err:
        return err
    body = request.get_json(silent=True) or {}
    text = str(body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text_required"}), 400
    pref = service.add_preference(key, text, league_id=body.get("league_id"))
    return jsonify({"ok": True, "preference": pref})


@assistant_bp.route("/api/assistant/preferences/<int:pref_id>", methods=["DELETE"])
def api_assistant_preferences_delete(pref_id):
    key, err = _require_account()
    if err:
        return err
    service.delete_preference(key, pref_id)
    return jsonify({"ok": True})


@assistant_bp.route("/api/assistant/preferences", methods=["DELETE"])
def api_assistant_preferences_forget_all():
    key, err = _require_account()
    if err:
        return err
    service.delete_all_preferences(key)
    return jsonify({"ok": True})
