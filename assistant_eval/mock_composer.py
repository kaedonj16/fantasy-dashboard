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
    "start_sit": {"href": "/waivers?tab=startsit", "label": "View in Start/Sit"},
    "trade_eval": {"href": "/trade", "label": "Open in Trade Calculator"},
    "buy_sell_hold": {"href": "/players", "label": "View player"},
    "player": {"href": "/players", "label": "View player"},
    "metric": {"href": "/metrics", "label": "View on Advanced Metrics"},
    "leaderboard": {"href": "/metrics", "label": "View on Advanced Metrics"},
    "waivers": {"href": "/waivers", "label": "View on Waivers"},
    "this_week": {"href": "/", "label": "View dashboard"},
}


# ── Mirrors of compose's deep-link builders ────────────────────────────────
# Same rules as dashboard_services/assistant/compose.py: trade answers stage
# the pack's deal in the calculator (a = receive side, b = give side, the
# params its own share flow mints), start/sit stages the pair in Compare,
# metric presets Advanced Metrics, player answers open the modal. Mentions
# carry pack player ids through to the UI.
from urllib.parse import quote, quote_plus


def _league_base(pack):
    league = (pack or {}).get("league") or {}
    platform, season, league_id = league.get("platform"), league.get("season"), league.get("league_id")
    if platform and season and league_id:
        return f"/{platform}/{season}/{league_id}"
    return None


def _side_ids(players):
    return [str(p["player_id"]) for p in (players or [])
            if isinstance(p, dict) and p.get("player_id")]


def _trade_calc_link(pack):
    pack = pack or {}
    give = _side_ids((pack.get("give") or {}).get("players"))
    recv = _side_ids((pack.get("receive") or {}).get("players"))
    if not give and not recv:
        return None
    base = _league_base(pack)
    path = f"{base}/trade" if base else "/trade"
    params = []
    if recv:
        params.append("a=" + ",".join(quote(pid, safe="") for pid in recv))
    if give:
        params.append("b=" + ",".join(quote(pid, safe="") for pid in give))
    for key, param in (("receive_picks", "ap"), ("give_picks", "bp")):
        ids = [str(p.get("pick_id")) for p in (pack.get(key) or [])
               if isinstance(p, dict) and p.get("pick_id")]
        if ids:
            params.append(param + "=" + ",".join(quote(i, safe="") for i in ids))
    return {"href": path + "?" + "&".join(params), "label": "Open in Trade Calculator"}


def _player_modal_link(player):
    if not isinstance(player, dict) or not player.get("player_id"):
        return None
    href = "/?player=" + quote(str(player["player_id"]), safe="")
    if player.get("name"):
        href += "&player_name=" + quote_plus(str(player["name"]))
    return {"href": href, "label": "View player"}


def _metric_link(pack):
    key = (pack or {}).get("metric_key")
    if not key:
        return None
    base = _league_base(pack)
    href = (f"{base}/metrics" if base else "/metrics") + "?metric=" + quote(str(key), safe="")
    positions = [str(p) for p in (pack.get("positions") or []) if p]
    if len(positions) == 1:
        href += "&pos=" + quote(positions[0], safe="")
    return {"href": href, "label": "Open Advanced Metrics"}


def _start_sit_link(pack):
    players = [p for p in ((pack or {}).get("players") or [])
               if isinstance(p, dict) and p.get("player_id")]
    if len(players) < 2:
        return None
    pick_id = str(pack.get("pick_player_id") or players[0]["player_id"])
    pick = next((p for p in players if str(p["player_id"]) == pick_id), players[0])
    others = [p for p in players if p is not pick]

    def _rank(p):
        for field in ("start_score", "projection", "value"):
            if p.get(field) is not None:
                try:
                    return float(p[field])
                except Exception:
                    return 0.0
        return 0.0

    other = max(others, key=_rank)
    base = _league_base(pack)
    href = (f"{base}/compare" if base else "/compare")
    href += "?p1=" + quote(str(pick["player_id"]), safe="")
    href += "&p2=" + quote(str(other["player_id"]), safe="")
    return {"href": href, "label": "Open head-to-head"}


def _resolve_act_link(intent, pack):
    pack = pack or {}
    if intent == "trade_eval":
        link = _trade_calc_link(pack)
        if link:
            return link
        base = _league_base(pack)
        return {"href": f"{base}/trade" if base else "/trade",
                "label": "Open in Trade Calculator"}
    if intent in ("player", "buy_sell_hold"):
        link = _player_modal_link(pack.get("player"))
        if link:
            return link
    elif intent == "metric":
        link = _metric_link(pack)
        if link:
            return link
    elif intent == "start_sit":
        link = _start_sit_link(pack)
        if link:
            return link
    static = _ACT_LINKS.get(intent)
    return dict(static) if static else None


def _mentions(intent, pack):
    pack = pack or {}
    found, seen = [], set()

    def _add(entry):
        if not isinstance(entry, dict):
            return
        pid = entry.get("player_id") or entry.get("id")
        name = entry.get("name")
        if pid and name and str(pid) not in seen:
            seen.add(str(pid))
            found.append({"player_id": str(pid), "name": str(name)})

    if intent == "trade_eval":
        for side in ("give", "receive"):
            for p in ((pack.get(side) or {}).get("players") or []):
                _add(p)
    elif intent == "start_sit":
        for p in pack.get("players") or []:
            _add(p)
    elif intent in ("player", "buy_sell_hold"):
        _add(pack.get("player"))
    elif intent == "metric":
        for p in pack.get("leaderboard_top5") or []:
            _add(p)
    elif intent == "waivers":
        for p in pack.get("candidates") or []:
            _add(p)
        for p in pack.get("cleared_drops") or []:
            _add(p)
    return found[:8]

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
            "act_link": dict(_ACT_LINKS["metric"]), "actions": [], "mentions": []}


def _board_link(pack):
    """Mirror of compose._board_link: the leaderboard preset link shape."""
    key = (pack or {}).get("metric_key")
    if not key:
        return None
    base = _league_base(pack)
    href = (f"{base}/metrics" if base else "/metrics") + "?metric=" + quote(str(key), safe="")
    positions = [str(p) for p in (pack.get("positions") or []) if p]
    if not positions and pack.get("position_filter"):
        positions = [str(pack["position_filter"])]
    if len(positions) == 1:
        href += "&pos=" + quote(positions[0], safe="")
    return {"href": href, "label": "Open Advanced Metrics"}


def _base(intent, pack):
    return {"caveat": None, "followups": [],
            "act_link": _resolve_act_link(intent, pack), "actions": [],
            "mentions": _mentions(intent, pack)}


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
    ans = _base("start_sit", pack)
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
    # Mirror of the backend's pinned verdict: utils.trade_value
    # .fair_value_band, the trade analyzer's shared fairness definition
    # (fair within max(7% of the larger side's total, 25.0)). Reimplemented
    # here so mock mode never imports the backend stack.
    give_t0 = (pack.get("give") or {}).get("total_value") or 0
    recv_t0 = (pack.get("receive") or {}).get("total_value") or 0
    band = max(max(float(give_t0), float(recv_t0), 1.0) * 0.07, 25.0)
    verdict = "FAIR" if abs(gap) <= band else ("ACCEPT" if gap > 0 else "DECLINE")
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
    ans = _base("trade_eval", pack)
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
    ans = _base("buy_sell_hold", pack)
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
    ans = _base("player", pack)
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
        ans = _base("metric", pack)
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
    ans = _base("metric", pack)
    ans.update({"verdict": f"{pack.get('label')}: {top['name']} leads", "text": text,
                "receipts": receipts, "followups": ["Who is moving up?", "What is snap share?"]})
    return ans


def _waivers(pack):
    candidates = pack.get("candidates") or []
    ans = _base("waivers", pack)
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
    receipts = [{"label": a.get("headline") or a.get("tag") or "action", "value": fmt(a["key_number"])}
                for a in actions if isinstance(a, dict) and a.get("key_number") is not None][:4]
    text = " ".join(f"{a.get('headline')}. {a.get('detail')}" for a in actions if isinstance(a, dict))
    ans = _base("this_week", pack)
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


# -- leaderboard: mirrors of compose's deterministic board rendering -------------

def gated_answer(pack: dict) -> dict:
    """Mirror of compose.gated_answer (same strings, same shape)."""
    label = pack.get("label") or "That metric"
    text = (f"{label} is a PRO metric on BR Fantasy, so its board is not included "
            "on the free tier. Free boards I can pull right now include target share, "
            "air yards share, and snap share.")
    return {"verdict": None, "text": text, "receipts": [], "caveat": None,
            "followups": ["Who leads in target share?", "Who leads in snap share?"],
            "act_link": _board_link(pack) or dict(_ACT_LINKS["leaderboard"]),
            "actions": []}


def _fmt_delta(value) -> str:
    f = float(value)
    if f == 0:
        return "0"
    return ("+" if f > 0 else "-") + fmt(abs(f))


def _board_verdict(pack):
    board = pack.get("board") or []
    if not board:
        return None
    top = board[0]
    kind = pack.get("board_kind")
    label = pack.get("label") or ""
    if kind == "matchup":
        return f"{top.get('team')} gives up the most to {pack.get('position_filter')}s"
    if kind == "intersection":
        return f"{top.get('name')} tops both boards"
    if kind == "movers":
        word = "faller" if pack.get("direction") == "fallers" else "riser"
        return f"{top.get('name')} is the biggest {word} in {label}"
    return f"{top.get('name')} leads {label}"


def _board_line(row, pack):
    kind = pack.get("board_kind")
    rank = row.get("rank")
    name = row.get("name") or ""
    meta = ", ".join(x for x in (row.get("team"), row.get("position")) if x)
    head = f"{rank}. {name}" + (f" ({meta})" if meta else "")
    value = fmt(row.get("value")) if row.get("value") is not None else ""
    if kind == "matchup":
        return (f"{rank}. {row.get('team')}: {value} fantasy points allowed "
                f"per game to {pack.get('position_filter')}")
    if kind == "movers":
        if row.get("delta") is not None:
            return (f"{head}: {_fmt_delta(row['delta'])} to {value} "
                    f"(was {fmt(row.get('baseline_value'))})")
        return f"{head}: {value} {pack.get('label')}"
    if kind == "intersection":
        sec = row.get("secondary") or {}
        return (f"{head}: {value} {pack.get('label')}, "
                f"{fmt(sec.get('value'))} {pack.get('second_label')}")
    line = f"{head}: {value}"
    if kind == "dual":
        sec = row.get("secondary")
        if sec:
            line += (f" - {pack.get('second_label')} {fmt(sec.get('value'))} "
                     f"(#{sec.get('rank')})")
        elif pack.get("second_label"):
            line += f" - {pack.get('second_label')} not qualified"
    return line


def _board_receipt(row, pack):
    kind = pack.get("board_kind")
    label = f"{row.get('rank')}. {row.get('name') or row.get('team')}"
    value = fmt(row.get("value")) if row.get("value") is not None else ""
    if kind == "movers" and row.get("delta") is not None:
        return {"label": label, "value": f"{_fmt_delta(row['delta'])} (now {value})"}
    if kind == "intersection":
        sec = row.get("secondary") or {}
        return {"label": label,
                "value": f"{value} - {pack.get('second_label')} {fmt(sec.get('value'))}"}
    if kind == "dual":
        sec = row.get("secondary")
        if sec:
            return {"label": label,
                    "value": (f"{value} - {pack.get('second_label')} "
                              f"{fmt(sec.get('value'))} (#{sec.get('rank')})")}
    return {"label": label, "value": value}


def _board_intro_template(pack):
    board = pack.get("board") or []
    top = board[0] if board else {}
    kind = pack.get("board_kind")
    label = pack.get("label") or ""
    value = fmt(top.get("value")) if top.get("value") is not None else ""
    if kind == "matchup":
        return (f"{top.get('team')} is the softest matchup for "
                f"{pack.get('position_filter')}s at {value} fantasy points "
                "allowed per game.")
    if kind == "intersection":
        return (f"{pack.get('board_count')} players sit in the top "
                f"{pack.get('requested_n')} of both {label} and "
                f"{pack.get('second_label')}.")
    if kind == "movers":
        if top.get("delta") is not None:
            return (f"{top.get('name')} moved the most in {label} over the "
                    f"recent window, {_fmt_delta(top['delta'])} to {value}.")
        return f"{top.get('name')} leads the {label} board at {value}."
    return (f"{top.get('name')} leads {label} at {value} across "
            f"{pack.get('pool_size')} qualified players.")


def _leaderboard(pack):
    # Mock mode has no model, so the intro is always the deterministic
    # template (the live composer may swap in a model framing sentence that
    # names no board entry; verdict, lines, and receipts are identical).
    board = pack.get("board") or []
    body = "\n".join([_board_intro_template(pack)]
                     + [_board_line(r, pack) for r in board])
    qual = pack.get("qualification_text")
    if qual:
        body += "\n" + qual
    ans = _base("leaderboard", pack)
    ans["act_link"] = _board_link(pack) or ans["act_link"]
    ans.update({"verdict": _board_verdict(pack), "text": body,
                "receipts": [_board_receipt(r, pack) for r in board],
                "followups": ["Who is rising in target share?",
                              "Who leads in air yards share?"]})
    return ans


_BUILDERS = {"start_sit": _start_sit, "trade_eval": _trade_eval,
             "buy_sell_hold": _buy_sell_hold, "player": _player, "metric": _metric,
             "leaderboard": _leaderboard,
             "waivers": _waivers, "this_week": _this_week}


def compose_mock(question: dict) -> dict:
    """Build the mock answer for one frozen question entry."""
    pack = question.get("evidence_pack") or {}
    if isinstance(pack, dict) and pack.get("gated"):
        return gated_answer(pack)
    if isinstance(pack, dict) and pack.get("available") is False:
        return no_data_answer(pack)
    builder = _BUILDERS[question["type"]]
    return builder(pack)
