"""Deterministic stub composer for the eval harness (mock mode).

Builds an answer for each frozen question from its evidence pack alone:
the verdict comes from the pack's own fields (pick_player_id, fairness_gap
plus margin.thin, signal, leaderboard order, gate-passing candidates,
digest actions), receipts are sampled from pack numbers, and an
available:false pack gets the backend's no-data pattern (mirroring
dashboard_services.assistant.compose.no_data_answer, reimplemented here so
mock mode never imports the model stack).

Digit discipline: templates interpolate only pack values. Any other number
in the output would be an invented number and the scorer would be right to
fail it, so prose avoids digits entirely (no "top five", no "30-day").
"""
from __future__ import annotations


def fmt(value) -> str:
    """Render a pack number the way prose would: 4850.0 -> '4850'."""
    if value is None:
        return ""
    f = float(value)
    if f == int(f):
        return str(int(f))
    return str(f)


# Mirror of compose._ACT_LINKS: act_link is a {href, label} object (or None),
# assigned deterministically, never a bare string.
_ACT_LINKS = {
    "start_sit": {"href": "/waivers?tab=startsit", "label": "Open Start/Sit"},
    "trade_eval": {"href": "/trade-hub", "label": "Open in Trade Hub"},
    "buy_sell_hold": {"href": "/players", "label": "View player"},
    "player": {"href": "/players", "label": "View player"},
    "metric": {"href": "/advanced-metrics", "label": "Open Advanced Metrics"},
    "waivers": {"href": "/waivers", "label": "Open Waivers"},
    "this_week": {"href": "/", "label": "Open dashboard"},
}

_CTA_LABELS = {"lineup": "Set lineup", "waiver": "View waivers",
               "waivers": "View waivers", "trade": "Build offer"}


def no_data_answer(pack: dict) -> dict:
    """Mirror of compose.no_data_answer (same strings, same shape)."""
    missing = pack.get("missing") or "that data is not tracked"
    nearest = pack.get("nearest") or []
    text = f"That is not available: {missing}."
    if nearest:
        text += f" The nearest tracked options are {', '.join(str(n) for n in nearest[:3])}. Ask about one of those and I can pull the live numbers."
    return {"verdict": None, "text": text, "receipts": [], "caveat": None,
            "followups": [f"What is {n}?" for n in nearest[:2]],
            "act_link": dict(_ACT_LINKS["metric"]), "actions": []}


def _base(intent):
    link = _ACT_LINKS.get(intent)
    return {"caveat": None, "followups": [],
            "act_link": dict(link) if link else None, "actions": []}


def _start_sit(pack):
    players = pack.get("players") or []
    pick = next((p for p in players if p.get("player_id") == pack.get("pick_player_id")), players[0])
    other = next((p for p in players if p is not pick), pick)
    receipts = []
    for pl, label in ((pick, "projection"), (other, "projection")):
        if pl.get("projection") is not None:
            receipts.append({"label": f"{pl['name']} {label}", "value": fmt(pl["projection"])})
    for pl in (pick, other):
        ppg = (pl.get("role") or {}).get("ppg")
        if ppg is not None and len(receipts) < 4:
            receipts.append({"label": f"{pl['name']} points per game", "value": fmt(ppg)})
    ans = _base("start_sit")
    ans.update({
        "verdict": f"START {pick['name']}",
        "text": (f"Start {pick['name']} over {other['name']}. "
                 f"{pick['name']} projects {fmt(pick.get('projection'))} points against "
                 f"{fmt(other.get('projection'))} for {other['name']}."),
        "receipts": receipts[:4],
        "followups": ["What about next week?", "Who else is on my bench?"],
    })
    if (pack.get("margin") or {}).get("thin"):
        ans["caveat"] = "Close call: the margin here is thin."
    return ans


def _trade_eval(pack):
    gap = float(pack.get("fairness_gap") or 0)
    thin = bool((pack.get("margin") or {}).get("thin"))
    verdict = "FAIR" if thin or gap == 0 else ("ACCEPT" if gap > 0 else "DECLINE")
    give_t = (pack.get("give") or {}).get("total_value")
    recv_t = (pack.get("receive") or {}).get("total_value")
    receipts = [{"label": "You give (total value)", "value": fmt(give_t)},
                {"label": "You receive (total value)", "value": fmt(recv_t)},
                {"label": "Fairness gap", "value": fmt(gap)}]
    give_names = ", ".join(p["name"] for p in (pack.get("give") or {}).get("players") or [])
    recv_names = ", ".join(p["name"] for p in (pack.get("receive") or {}).get("players") or [])
    text = (f"{give_names} for {recv_names}. Your side totals {fmt(give_t)} against "
            f"{fmt(recv_t)} coming back, a fairness gap of {fmt(gap)}.")
    unpriced = pack.get("unpriced") or []
    if unpriced:
        text += f" One piece is unpriced: {unpriced[0]}. The rest is judged on the priced players."
    ans = _base("trade_eval")
    ans.update({"verdict": verdict, "text": text, "receipts": receipts,
                "followups": ["What would balance it?", "Who else could I ask for?"]})
    if thin:
        ans["caveat"] = "Close call: the values are nearly even."
    return ans


def _buy_sell_hold(pack):
    player = pack.get("player") or {}
    trend = pack.get("value_trend_30d") or {}
    verdict = str(pack.get("signal") or "hold").upper()
    receipts = []
    if trend.get("delta_pct") is not None:
        receipts.append({"label": "Value trend (percent)", "value": fmt(trend["delta_pct"])})
    if trend.get("current") is not None:
        receipts.append({"label": "Current value", "value": fmt(trend["current"])})
    role = pack.get("role") or {}
    if role.get("ppg") is not None:
        receipts.append({"label": "Points per game", "value": fmt(role["ppg"])})
    if role.get("target_share") is not None and len(receipts) < 4:
        receipts.append({"label": "Target share", "value": fmt(role["target_share"])})
    ans = _base("buy_sell_hold")
    ans.update({
        "verdict": verdict,
        "text": (f"{verdict} on {player.get('name')}. The recent value trend is "
                 f"{fmt(trend.get('delta_pct'))} percent, with current value at "
                 f"{fmt(trend.get('current'))}."),
        "receipts": receipts[:4],
        "followups": ["What is his role right now?", "Who could I trade him for?"],
    })
    if (pack.get("margin") or {}).get("thin"):
        ans["caveat"] = "Close call: the trend is small either way."
    return ans


def _player(pack):
    player = pack.get("player") or {}
    role = pack.get("role") or {}
    receipts = []
    for key, label in (("target_share", "Target share"), ("touches_per_game", "Touches per game"),
                       ("ppg", "Points per game"), ("snap_share", "Snap share")):
        if role.get(key) is not None and len(receipts) < 4:
            receipts.append({"label": label, "value": fmt(role[key])})
    if player.get("value") is not None and len(receipts) < 4:
        receipts.append({"label": "Value", "value": fmt(player["value"])})
    ans = _base("player")
    ans.update({
        "verdict": f"{player.get('name')} update",
        "text": (f"{player.get('name')} is scoring {fmt(role.get('ppg'))} points per game "
                 f"with a target share of {fmt(role.get('target_share'))} for the "
                 f"{player.get('team')} at {player.get('position')}."),
        "receipts": receipts,
        "followups": ["Buy, sell, or hold on him?", "What about next week?"],
    })
    return ans


def _metric(pack):
    top5 = pack.get("leaderboard_top5") or []
    if not top5:
        ans = _base("metric")
        ans.update({"verdict": pack.get("label"), "text": pack.get("definition") or "",
                    "receipts": [], "followups": []})
        return ans
    chosen, seen = [], set()
    ordered = [top5[0]] + [e for e in top5 if e.get("on_my_roster")] + top5[1:]
    for e in ordered:
        if e["player_id"] not in seen and len(chosen) < 4:
            chosen.append(e)
            seen.add(e["player_id"])
    receipts = [{"label": e["name"], "value": fmt(e.get("value"))} for e in chosen]
    top = top5[0]
    text = f"{pack.get('definition')} {top['name']} leads at {fmt(top.get('value'))}."
    mine = next((e for e in top5 if e.get("on_my_roster") and e is not top), None)
    if mine:
        text += f" Your {mine['name']} sits at {fmt(mine.get('value'))}."
    ans = _base("metric")
    ans.update({"verdict": f"{pack.get('label')}: {top['name']} leads", "text": text,
                "receipts": receipts, "followups": ["Who is moving up?", "What is snap share?"]})
    return ans


def _waivers(pack):
    candidates = pack.get("candidates") or []
    ans = _base("waivers")
    if not candidates:
        ans.update({"verdict": None,
                    "text": "The wire is thin this week: nobody available clears your positional gate.",
                    "receipts": [], "followups": ["What should I do with my team this week?"]})
        return ans
    top = candidates[0]
    worst = (pack.get("my_worst_by_position") or {}).get(top.get("position")) or {}
    receipts = [{"label": f"{top['name']} value", "value": fmt(top.get("value"))}]
    if worst.get("value") is not None:
        receipts.append({"label": f"Your worst {top.get('position')} ({worst.get('name')}) value",
                         "value": fmt(worst["value"])})
    if len(candidates) > 1:
        receipts.append({"label": f"{candidates[1]['name']} value",
                         "value": fmt(candidates[1].get("value"))})
    text = (f"Add {top['name']}. The value is {fmt(top.get('value'))}, clear of your weakest "
            f"{top.get('position')} at {fmt(worst.get('value'))}.")
    drops = pack.get("cleared_drops") or []
    if drops:
        text += f" {drops[0]['name']} is a cleared drop if you need the spot."
    ans.update({"verdict": f"ADD {top['name']}", "text": text, "receipts": receipts[:4],
                "followups": ["Who else is available?", "What should I do with my team this week?"]})
    return ans


def _this_week(pack):
    actions = pack.get("actions") or []
    receipts = [{"label": a.get("tag") or "action", "value": fmt(a["key_number"])}
                for a in actions if isinstance(a, dict) and a.get("key_number") is not None][:4]
    text = " ".join(f"{a.get('headline')}. {a.get('detail')}" for a in actions if isinstance(a, dict))
    ans = _base("this_week")
    ans.update({
        "verdict": actions[0]["headline"] if actions else "Your team is set",
        "text": text or "Your team is set for the week.",
        "receipts": receipts,
        "followups": ["Who should I add from waivers?"],
        # Mirror of the composer's action mapping: href + cta_label + key_number.
        "actions": [{"tag": a.get("tag"), "headline": a.get("headline"),
                     "detail": a.get("detail"),
                     "href": a.get("act_link") or a.get("href") or "/",
                     "cta_label": _cta_label(a),
                     "key_number": a.get("key_number")} for a in actions],
    })
    return ans


def _cta_label(action):
    if action.get("cta_label"):
        return str(action["cta_label"])
    tag = str(action.get("tag") or "").lower()
    for key, label in _CTA_LABELS.items():
        if key in tag:
            return label
    return "View"


_BUILDERS = {"start_sit": _start_sit, "trade_eval": _trade_eval,
             "buy_sell_hold": _buy_sell_hold, "player": _player, "metric": _metric,
             "waivers": _waivers, "this_week": _this_week}


def compose_mock(question: dict) -> dict:
    """Build the mock answer for one frozen question entry."""
    pack = question.get("evidence_pack") or {}
    if isinstance(pack, dict) and pack.get("available") is False:
        return no_data_answer(pack)
    builder = _BUILDERS[question["type"]]
    return builder(pack)
