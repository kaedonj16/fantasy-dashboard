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


def _fu(text: str) -> dict:
    """Mirror of compose.followup_pair: follow-ups are {label, ask}
    pairs. Every string the mock emits is already the user's own
    question, so the label (chip text) and the ask (the message sent
    when the chip is tapped) are the same string."""
    return {"label": text, "ask": text}


# Mirror of compose._ACT_LINKS: act_link is a {href, label} link object, a
# {player_id, player_name, label} modal action (player / buy_sell_hold), or
# None; assigned deterministically, never a bare string.
_ACT_LINKS = {
    "start_sit": {"href": "/waivers?tab=startsit", "label": "View in Start/Sit"},
    "trade_eval": {"href": "/trade", "label": "Open in Trade Calculator"},
    "metric": {"href": "/metrics", "label": "View on Advanced Metrics"},
    "leaderboard": {"href": "/metrics", "label": "View on Advanced Metrics"},
    "waivers": {"href": "/waivers", "label": "View on Waivers"},
    "this_week": {"href": "/", "label": "View dashboard"},
    "trade_finding": {"href": "/trade?tab=suggestions", "label": "Open Trade Suggestions"},
    "breakout_board": {"href": "/breakouts", "label": "View breakout board"},
    "breakout_track_record": {"href": "/breakouts", "label": "View breakout board"},
    "breakout_history": {"href": "/breakouts", "label": "View breakout board"},
    "keeper": {"href": "/keeper", "label": "View Keeper Optimizer"},
    "rookie": {"href": "/prospects", "label": "View Rookie Rankings"},
    "cross_league": {"href": "/portfolio", "label": "View Portfolio"},
    "replacement": {"href": "/trade?tab=suggestions", "label": "Open Trade Suggestions"},
    "handcuff": {"href": "/waivers", "label": "View on Waivers"},
    "watchlist": {"href": "/watchlist", "label": "View Watchlist"},
}


# ── Mirrors of compose's deep-link builders ────────────────────────────────
# Same rules as dashboard_services/assistant/compose.py: trade answers stage
# the pack's deal in the calculator (a = receive side, b = give side, the
# params its own share flow mints), start/sit stages the pair in Compare,
# metric presets Advanced Metrics, player answers open the modal. Mentions
# carry pack player ids through to the UI.
from urllib.parse import quote


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
    return {"player_id": str(player["player_id"]),
            "player_name": str(player.get("name") or ""),
            "label": "View player"}


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
        # Terminal, mirroring compose: modal action or no link at all.
        return _player_modal_link(pack.get("player"))
    elif intent == "keeper":
        base = _league_base(pack)
        return {"href": f"{base}/keeper" if base else "/keeper",
                "label": "View Keeper Optimizer"}
    elif intent == "metric":
        link = _metric_link(pack)
        if link:
            return link
    elif intent == "start_sit":
        link = _start_sit_link(pack)
        if link:
            return link
    elif intent == "keeper":
        base = _league_base(pack)
        return {"href": f"{base}/keeper" if base else "/keeper",
                "label": "View Keeper Optimizer"}
    elif intent == "replacement":
        cands = [c for c in (pack.get("candidates") or []) if isinstance(c, dict)]
        if cands and cands[0].get("location") == "waivers":
            return {"href": "/waivers", "label": "View on Waivers"}
        base = _league_base(pack)
        return {"href": f"{base}/trade?tab=suggestions" if base else "/trade?tab=suggestions",
                "label": "Open Trade Suggestions"}
    elif intent == "handcuff":
        nxt = pack.get("next_up") or {}
        if nxt.get("location") == "waivers":
            return {"href": "/waivers", "label": "View on Waivers"}
        link = _player_modal_link(nxt) or _player_modal_link(pack.get("subject"))
        if link:
            return link
    pack_link = _pack_act_link(intent, pack)
    if pack_link:
        return pack_link
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
        for p in pack.get("stash_candidates") or []:
            _add(p)
        for p in pack.get("streamers") or []:
            _add(p)
        for p in pack.get("drop_ranking") or []:
            _add(p)
        for p in pack.get("cleared_drops") or []:
            _add(p)
        _add(pack.get("asked_drop"))
    elif intent == "replacement":
        _add(pack.get("subject"))
        for p in pack.get("candidates") or []:
            _add(p)
    elif intent == "handcuff":
        _add(pack.get("subject"))
        _add(pack.get("next_up"))
        _add(pack.get("also_next"))
        for p in pack.get("my_options") or []:
            _add(p)
    elif intent == "keeper":
        for p in pack.get("my_keepers") or []:
            _add(p)
        _add(pack.get("asked_player"))
        for p in pack.get("roster_core") or []:
            _add(p)
    elif intent == "watchlist":
        for p in pack.get("players") or []:
            _add(p)
    elif intent == "rookie":
        prospect = pack.get("prospect") or {}
        if prospect.get("sleeper_id") and prospect.get("name"):
            _add({"player_id": prospect["sleeper_id"], "name": prospect["name"]})
    return found[:8]



def _pack_act_link(intent, pack):
    """Mirror of compose._pack_act_link (league-scoped Teams path)."""
    if intent not in ("teams", "pick_ownership") or not isinstance(pack, dict):
        return None
    league = pack.get("league") or {}
    if league.get("platform") and league.get("season") and league.get("league_id"):
        return {"href": f"/{league['platform']}/{league['season']}/{league['league_id']}/teams",
                "label": "View on Teams"}
    return None

_CTA_LABELS = {"lineup": "Set lineup", "waiver": "View waivers",
               "waivers": "View waivers", "trade": "Build offer"}

# Mirror of compose._LEAGUE_LINKS: league-context intents link at the
# league surface, href built from the pack's league fields.
_LEAGUE_LINKS = {
    "playoffs": ("dashboard", "View Playoff Odds"),
    "power_rankings": ("standings", "View Power Rankings"),
    "standings": ("standings", "View standings"),
    "matchup": ("weekly", "View Matchups"),
    "schedule": ("schedule", "Open in Schedule Assistant"),
    "activity": ("activity", "View Activity"),
    "front_office": ("dashboard", "View Front Office"),
    "live": ("scorezone", "Open ScoreZone"),
    "draft": ("draft", "Open Draft Room"),
    "nfl_schedule": ("scorezone", "Open ScoreZone"),
}


def _league_link(intent, pack):
    spec = _LEAGUE_LINKS.get(intent or "")
    league = (pack or {}).get("league") or {}
    if not spec or not league.get("platform") or not league.get("season") or not league.get("league_id"):
        return None
    page, label = spec
    return {"href": f"/{league['platform']}/{league['season']}/{league['league_id']}/{page}",
            "label": label}


# Mirror of compose's unresolved-player decline (same strings).
_UNRESOLVED_PLAYER_MISSING = {"no resolved player", "no resolved players"}

_UNRESOLVED_PLAYER_TEXT = {
    "buy_sell_hold": ("I could not tell which player you mean. Name the player "
                      "and I will make the buy, sell, or hold call from their "
                      "value trend and production."),
    "start_sit": ("I could not tell which players you mean. Name the players "
                  "and I will make the start or sit call."),
    "player": ("I could not tell which player you mean. Name the player and "
               "I will pull their numbers."),
    "breakout_history": ("I could not tell which player you mean. Name the "
                         "player and I will check whether they were a "
                         "breakout call."),
}


# Mirror of compose._MISSING_TEXT: internal markers that are not
# sentences, each mapped to the plain sentence the user reads.
_MISSING_TEXT = {
    "your roster in this league": (
        "I could not find your roster in this league, so that check "
        "cannot run here."),
    "no league context loaded": (
        "I do not have a league loaded for this chat yet. Open the "
        "assistant from one of your league pages and I can use its "
        "context."),
    "no league context": (
        "I do not have a league loaded for this chat yet. Open the "
        "assistant from one of your league pages and I can use its "
        "context."),
    "pro_only": "That board is part of PRO on BR Fantasy.",
    "routes not tracked": (
        "Routes not tracked on the site, so I cannot build that board."),
    "metric not available": "That metric is not available right now.",
    "that data is not tracked": "That data is not tracked on the site.",
    "that metric is not tracked": "That metric is not tracked on the site.",
    "a position to rank defenses against": (
        "Tell me a position and I will rank the defenses against it."),
    "which pick (season and round) you mean": (
        "I could not tell which pick you mean. Name the season and the "
        "round and I will price it."),
    "no signed-in account for this chat": (
        "That needs a signed-in account. Sign in and ask again."),
}


def _missing_text(missing) -> str:
    """Mirror of compose._missing_text (same strings, same shape)."""
    import re as _re
    m = str(missing or "").strip()
    if m in _MISSING_TEXT:
        return _MISSING_TEXT[m]
    if m.startswith("no model value for "):
        name = m[len("no model value for "):].strip()
        return f"{name} does not have a model value yet, so I cannot price him."
    if m.startswith("no age on file for "):
        name = m[len("no age on file for "):].strip()
        return f"I do not have an age on file for {name}."
    if m.startswith("no one behind "):
        rest = m[len("no one behind "):].strip()
        return f"There is no one behind {rest}."
    repl = _re.match(r"no (\w+) replacement for (.+?) in a ", m)
    if repl:
        return (f"No {repl.group(1)} replacement for {repl.group(2)} "
                "turned up in a comparable value band.")
    words = m.split()
    if len(words) >= 4 and "_" not in m and "{" not in m and "}" not in m:
        text = m[0].upper() + m[1:]
        if not text.endswith((".", "!", "?")):
            text += "."
        return text
    return "I do not have that on the site yet."


def _empty_value_board_answer(pack: dict) -> dict:
    """Mirror of compose._empty_value_board_answer (same strings)."""
    kind = (pack or {}).get("board_kind")
    word = "buy-low" if kind == "buy_low" else "sell-high"
    miss = (pack or {}).get("closest_miss") or {}
    receipts = []
    if miss.get("name"):
        reason = miss.get("reason_text") or "he does not clear the screen"
        text = (f"No {word} candidate clears the screen right now. "
                f"The closest miss is {miss['name']}: {reason}.")
        if (miss.get("delta") is not None
                and miss.get("baseline_value") is not None
                and miss.get("value") is not None):
            receipts = [{
                "label": f"{miss['name']} value move",
                "value": (f"{_fmt_delta(miss.get('delta'))} "
                          f"({fmt(miss.get('baseline_value'))} to "
                          f"{fmt(miss.get('value'))})"),
            }]
    else:
        text = (f"No {word} candidate clears the screen right now: no "
                "mover this week pairs a real price move with a "
                "verifiable production line.")
    other = ("Who should I sell high?" if kind == "buy_low"
             else "Who should I buy low?")
    return {"verdict": f"No {word} clears the screen right now",
            "text": text, "receipts": receipts, "caveat": None,
            "followups": [_fu("Who are the biggest movers this week?"),
                          _fu(other)],
            "act_link": _board_link(pack) or dict(_ACT_LINKS["leaderboard"]),
            "actions": [], "mentions": []}


def no_data_answer(pack: dict) -> dict:
    """Mirror of compose.no_data_answer (same strings, same shape)."""
    missing = pack.get("missing") or "that data is not tracked"
    nearest = pack.get("nearest") or []
    if missing in _UNRESOLVED_PLAYER_MISSING:
        text = _UNRESOLVED_PLAYER_TEXT.get(
            (pack or {}).get("type"),
            "I could not tell which player you mean. Name the player and "
            "I will pull the numbers.")
        nearest = []
    else:
        text = _missing_text(missing)
    if nearest:
        text += f" The nearest tracked options are {', '.join(str(n) for n in nearest[:3])}. Ask about one of those and I can pull the live numbers."
    # The link names the surface the question was about: league-context
    # intents use their league link, everything else resolves normally
    # (a keeper no-data lands on the Keeper Optimizer, not Advanced Metrics).
    link = (_league_link((pack or {}).get("type"), pack)
            or _resolve_act_link(str((pack or {}).get("type") or "metric"), pack)
            or dict(_ACT_LINKS["metric"]))
    return {"verdict": None, "text": text, "receipts": [], "caveat": None,
            "followups": [_fu(f"What is {n}?") for n in nearest[:2]],
            "act_link": link, "actions": [], "mentions": []}


def _board_link(pack):
    """Mirror of compose._board_link: the leaderboard preset link shape."""
    if (pack or {}).get("board_kind") in ("buy_low", "sell_high"):
        return {"href": "/top-movers", "label": "View on Top Movers"}
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


def _start_sit_roster(pack):
    # Mirror of the roster-scoped start/sit pack: the verdict names the
    # recommended starter at the slot; the comparison player is the slot's
    # current occupant, or the best bench challenger when the pick already
    # holds the slot.
    players = pack.get("players") or []
    pick = next((p for p in players if p.get("player_id") == pack.get("pick_player_id")), players[0])
    occupant = next((p for p in players if p.get("slot_occupant")), None)
    bench = next((p for p in players if p is not pick and not p.get("current_starter")), None)
    other = occupant if occupant is not None and occupant is not pick else (bench or pick)
    slot = pack.get("slot_label") or pack.get("slot") or "that slot"
    receipts = []
    for pl in (pick, other):
        if pl.get("projection") is not None and all(r["label"] != f"{pl['name']} projection" for r in receipts):
            receipts.append({"label": f"{pl['name']} projection", "value": fmt(pl["projection"])})
    for pl in (pick, other):
        ppg = (pl.get("role") or {}).get("ppg")
        if ppg is not None and len(receipts) < 4 and all(r["label"] != f"{pl['name']} points per game" for r in receipts):
            receipts.append({"label": f"{pl['name']} points per game", "value": fmt(ppg)})
    ans = _base("start_sit", pack)
    if pack.get("question_mode") == "sit" and pack.get("sit_player_id"):
        sit = next((p for p in players if p.get("player_id") == pack.get("sit_player_id")), other)
        text = (f"Sit {sit['name']} at {slot}. Start {pick['name']} in his place: "
                f"{pick['name']} projects {fmt(pick.get('projection'))} points against "
                f"{fmt(sit.get('projection'))} for {sit['name']}.")
    elif occupant is not None and occupant.get("player_id") == pick.get("player_id"):
        text = (f"Keep {pick['name']} at {slot}. {pick['name']} projects "
                f"{fmt(pick.get('projection'))} points, clear of {other['name']} at "
                f"{fmt(other.get('projection'))} on your bench.")
    else:
        text = (f"Start {pick['name']} at {slot}. {pick['name']} projects "
                f"{fmt(pick.get('projection'))} points against {fmt(other.get('projection'))} "
                f"for {other['name']}, who holds the slot now.")
    ans.update({
        "verdict": f"START {pick['name']}",
        "text": text,
        "receipts": receipts[:4],
        "followups": [_fu("What about next week?"), _fu("Who else is on my bench?")],
    })
    if (pack.get("margin") or {}).get("thin"):
        ans["caveat"] = "Close call: the margin here is thin."
    return ans


def _start_sit(pack):
    if pack.get("roster_mode"):
        return _start_sit_roster(pack)
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
        "followups": [_fu("What about next week?"), _fu("Who else is on my bench?")],
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
    for key, side in (("give_picks", "You give"), ("receive_picks", "You receive")):
        for p in pack.get(key) or []:
            if not isinstance(p, dict):
                continue
            owner = p.get("original_owner_team")
            text += f" {side} the {p.get('label')} pick"
            if owner:
                text += f" (originally {owner}'s pick)"
            if p.get("value") is not None:
                text += f" worth {fmt(p.get('value'))}"
            text += "."
    unpriced = pack.get("unpriced") or []
    if unpriced:
        text += f" One piece is unpriced: {unpriced[0]}. The rest is judged on the priced players."
    ans = _base("trade_eval", pack)
    ans.update({"verdict": verdict, "text": text, "receipts": receipts,
                "followups": [_fu("What would balance it?"), _fu("Who else could I ask for?")]})
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
        "followups": [_fu("What is his role right now?"), _fu("Who could I trade him for?")],
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
        "followups": [_fu("Buy, sell, or hold on him?"), _fu("What about next week?")],
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
                "receipts": receipts, "followups": [_fu("Who is moving up?"), _fu("What is snap share?")]})
    return ans


def _waivers(pack):
    candidates = pack.get("candidates") or []
    ans = _base("waivers", pack)
    mode = pack.get("waiver_mode")
    if mode == "drop" and pack.get("drop_ranking"):
        top = pack["drop_ranking"][0]
        receipts = [{"label": f"{top['name']} value", "value": fmt(top.get("value"))}]
        if len(pack["drop_ranking"]) > 1:
            nxt = pack["drop_ranking"][1]
            receipts.append({"label": f"{nxt['name']} value", "value": fmt(nxt.get("value"))})
        text = (f"{top['name']} is your most droppable player. His value is "
                f"{fmt(top.get('value'))}, the lowest on your roster, and you hold "
                f"{fmt(top.get('roster_depth_at_position'))} at {top.get('position')}.")
        ans.update({"verdict": f"DROP {top['name']}", "text": text, "receipts": receipts[:4],
                    "followups": [_fu("Who should I add from waivers?"),
                                  _fu("What should I do with my team this week?")]})
        return ans
    if mode == "stash" and pack.get("stash_candidates"):
        top = pack["stash_candidates"][0]
        receipts = [{"label": f"{top['name']} value", "value": fmt(top.get("value"))}]
        if top.get("age") is not None:
            receipts.append({"label": f"{top['name']} age", "value": fmt(top.get("age"))})
        if len(pack["stash_candidates"]) > 1:
            nxt = pack["stash_candidates"][1]
            receipts.append({"label": f"{nxt['name']} value", "value": fmt(nxt.get("value"))})
        text = (f"Stash {top['name']}. His value is {fmt(top.get('value'))} at age "
                f"{fmt(top.get('age'))}, the strongest stash profile on the wire.")
        ans.update({"verdict": f"STASH {top['name']}", "text": text, "receipts": receipts[:4],
                    "followups": [_fu("Who should I add from waivers?"),
                                  _fu("What should I do with my team this week?")]})
        return ans
    if mode == "stream" and pack.get("streamers"):
        top = pack["streamers"][0]
        receipts = [{"label": f"{top['name']} projection", "value": fmt(top.get("projection"))}]
        if top.get("value") is not None:
            receipts.append({"label": f"{top['name']} value", "value": fmt(top.get("value"))})
        if len(pack["streamers"]) > 1:
            nxt = pack["streamers"][1]
            receipts.append({"label": f"{nxt['name']} projection",
                             "value": fmt(nxt.get("projection"))})
        text = (f"Stream {top['name']} this week. He projects {fmt(top.get('projection'))} "
                f"points {top.get('matchup') or ''}, the best available line at "
                f"{top.get('position')}.").replace("  ", " ")
        ans.update({"verdict": f"STREAM {top['name']}", "text": text, "receipts": receipts[:4],
                    "followups": [_fu("Who should I start at FLEX?"),
                                  _fu("What should I do with my team this week?")]})
        return ans
    if not candidates:
        ans.update({"verdict": None,
                    "text": "The wire is thin this week: nobody available clears your positional gate.",
                    "receipts": [], "followups": [_fu("What should I do with my team this week?")]})
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
                "followups": [_fu("Who else is available?"), _fu("What should I do with my team this week?")]})
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
        "followups": [_fu("Who should I add from waivers?")],
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

def _finding_link(pack, finding):
    """Mirror of compose._finding_link (same URL shape, pack ids only)."""
    league = pack.get("league") or {}
    if not league.get("season") or not league.get("league_id"):
        return None
    recv = [str(p.get("player_id")) for p in finding.get("receive") or []
            if isinstance(p, dict) and p.get("player_id")]
    give = [str(p.get("player_id")) for p in finding.get("give") or []
            if isinstance(p, dict) and p.get("player_id")]
    if not recv or not give:
        return None
    return {"href": f"/{league.get('platform') or 'sleeper'}/{league['season']}/{league['league_id']}"
                    f"/trade?a={','.join(recv)}&b={','.join(give)}",
            "label": "Open in Trade Calculator"}

def _gated_trade_finding_answer(pack):
    """Mirror of compose._gated_trade_finding_answer (same strings)."""
    league = pack.get("league") or {}
    if league.get("season") and league.get("league_id"):
        pricing = (f"/{league.get('platform') or 'sleeper'}/{league['season']}"
                   f"/{league['league_id']}/pricing")
    else:
        pricing = "/pricing"
    return {
        "verdict": "Trade finding is a PRO feature",
        "text": ("Finding trades is part of PRO on BR Fantasy. The scan checks every roster in "
                 "your league for a partner whose needs line up with yours, and your plan does "
                 "not include it, so I cannot run it here. With PRO you can ask me to find a "
                 "trade and I will rank real packages from your league with values on both sides."),
        "receipts": [], "caveat": None,
        "followups": [_fu("What should I do with my team this week?"),
                      _fu("Should I accept a trade I was offered?")],
        "act_link": {"href": pricing, "label": "See PRO plans"},
        "actions": [],
    }

def _gated_front_office_answer(pack):
    """Mirror of compose._gated_front_office_answer (same strings)."""
    league = pack.get("league") or {}
    if league.get("season") and league.get("league_id"):
        pricing = (f"/{league.get('platform') or 'sleeper'}/{league['season']}"
                   f"/{league['league_id']}/pricing")
    else:
        pricing = "/pricing"
    return {
        "verdict": "The Front Office report is a PRO feature",
        "text": ("The Front Office report is part of PRO on BR Fantasy. It computes your "
                 "team's direction, playoff odds, top moves, trade targets, and cut "
                 "candidates, and your plan does not include it, so I cannot pull it here. "
                 "With PRO you can ask me what your front office says and I will walk "
                 "through its computed data."),
        "receipts": [], "caveat": None,
        "followups": [_fu("What should I do with my team this week?"),
                      _fu("Who should I add from waivers?")],
        "act_link": {"href": pricing, "label": "See PRO plans"},
        "actions": [],
    }


def gated_answer(pack: dict) -> dict:
    """Mirror of compose.gated_answer (same strings, same shape)."""
    if (pack or {}).get("type") == "trade_finding":
        return _gated_trade_finding_answer(pack)
    if (pack or {}).get("type") == "front_office":
        return _gated_front_office_answer(pack)
    label = pack.get("label") or "That metric"
    text = (f"{label} is a PRO metric on BR Fantasy, so its board is not included "
            "on the free tier. Free boards I can pull right now include target share, "
            "air yards share, and snap share.")
    return {"verdict": None, "text": text, "receipts": [], "caveat": None,
            "followups": [_fu("Who leads in target share?"), _fu("Who leads in snap share?")],
            "act_link": _board_link(pack) or dict(_ACT_LINKS["leaderboard"]),
            "actions": []}


def _fmt_delta(value) -> str:
    f = float(value)
    if f == 0:
        return "0"
    return ("+" if f > 0 else "-") + fmt(abs(f))


def _join_and(parts):
    """Mirror of compose._join_and."""
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return "".join(parts)
    if len(parts) == 2:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def _board_verdict(pack):
    board = pack.get("board") or []
    if not board:
        return None
    top = board[0]
    kind = pack.get("board_kind")
    label = pack.get("label") or ""
    if kind == "matchup":
        focus = pack.get("focus") or {}
        ranks = focus.get("ranks") or []
        if len(ranks) > 1:
            listed = _join_and(
                [f"#{r.get('rank')} of {r.get('pool_size')} against "
                 f"{r.get('position')}s" for r in ranks])
            return f"{focus.get('name')} defense is {listed}"
        if ranks:
            r = ranks[0]
            return (f"{focus.get('name')} defense is #{r.get('rank')} of "
                    f"{r.get('pool_size')} against {r.get('position')}s")
        boards = pack.get("boards") or []
        if len(boards) > 1:
            parts = []
            for sub in boards:
                sub_board = sub.get("board") or []
                if sub_board:
                    parts.append(f"{sub_board[0].get('team')} gives up the "
                                 f"most to {sub.get('position')}s")
            if parts:
                return "; ".join(parts)
        return f"{top.get('team')} gives up the most to {pack.get('position_filter')}s"
    if kind == "intersection":
        return f"{top.get('name')} tops both boards"
    if kind == "movers":
        word = "faller" if pack.get("direction") == "fallers" else "riser"
        return f"{top.get('name')} is the biggest {word} in {label}"
    if kind == "buy_low":
        return f"{top.get('name')} is the top buy-low"
    if kind == "sell_high":
        return f"{top.get('name')} is the top sell-high"
    if kind == "breadth":
        n = pack.get("breadth_top_n") or pack.get("requested_n") or 10
        word = pack.get("family_word") or ""
        return (f"{top.get('name')} has the most top-{n} "
                f"{word + ' ' if word else ''}finishes")
    if kind == "finishes":
        return (f"{top.get('name')} has the most top-{pack.get('finish_n') or 12} "
                f"finishes")
    if kind == "floor":
        return f"{top.get('name')} has the highest floor"
    if kind == "ceiling":
        return f"{top.get('name')} has the highest ceiling"
    if kind == "value":
        pos = pack.get("position_filter")
        return (f"{top.get('name')} is the most valuable {pos}"
                if pos else f"{top.get('name')} is the most valuable player")
    if kind == "slate":
        return f"{top.get('name')} has the easiest upcoming slate"
    if kind == "gap":
        return f"{top.get('name')} laps his team's No. 2 by the most"
    if kind == "rank_movers":
        return f"{top.get('name')} climbed the most in positional rank"
    if kind == "defenses":
        focus = pack.get("focus") or {}
        if focus:
            return (f"{focus.get('name')} defense is #{focus.get('rank')} of "
                    f"{focus.get('pool_size')} by fantasy points")
        return f"{top.get('name')} is the top defense by fantasy points"
    return f"{top.get('name')} leads {label}"


def _breadth_finishes_text(row):
    """Mirror of compose._breadth_finishes_text."""
    fins = row.get("top_finishes") or []
    parts = [f"{f.get('label')} #{f.get('rank')} ({fmt(f.get('value'))})"
             for f in fins[:5]]
    text = ", ".join(parts)
    if len(fins) > len(parts):
        text += ", and more"
    return text


def _value_stats(row):
    """Mirror of compose._value_stats."""
    if row.get("recent_ppg") is not None:
        return (f"{fmt(row.get('recent_ppg'))} PPG vs "
                f"{fmt(row.get('recent_expected_ppg'))} expected over "
                f"the last {row.get('recent_games')}")
    return (f"{fmt(row.get('ppg'))} PPG vs "
            f"{fmt(row.get('expected_ppg'))} expected")


def _value_fragments(row, pack):
    """Mirror of compose._value_fragments."""
    parts = []
    if row.get("mechanism"):
        parts.append(str(row["mechanism"]))
    anchor = row.get("peer_anchor_name")
    gap = row.get("peer_anchor_gap_pct")
    if anchor and gap is not None:
        window = (f"over the last {row.get('recent_games')}"
                  if row.get("recent_games") else "this season")
        if pack.get("board_kind") == "buy_low":
            parts.append(f"producing like {anchor} {window}, priced "
                         f"{fmt(gap)}% below him")
        else:
            parts.append(f"producing like {anchor} {window}, priced "
                         f"{fmt(gap)}% above him")
    if row.get("slate_note"):
        parts.append(f"{row['slate_note']} (avg matchup rank "
                     f"{fmt(row.get('slate_avg_rank'))})")
    return "; ".join(parts)


def _board_line(row, pack, position=None):
    kind = pack.get("board_kind")
    rank = row.get("rank")
    name = row.get("name") or ""
    meta = ", ".join(x for x in (row.get("team"), row.get("position")) if x)
    head = f"{rank}. {name}" + (f" ({meta})" if meta else "")
    value = fmt(row.get("value")) if row.get("value") is not None else ""
    if kind == "matchup":
        return (f"{rank}. {row.get('team')}: {value} fantasy points allowed "
                f"per game to {position or pack.get('position_filter')}")
    if kind == "movers":
        if row.get("delta") is not None:
            return (f"{head}: {_fmt_delta(row['delta'])} to {value} "
                    f"(was {fmt(row.get('baseline_value'))})")
        return f"{head}: {value} {pack.get('label')}"
    if kind in ("buy_low", "sell_high") and row.get("delta") is not None:
        line = (f"{head}: value {_fmt_delta(row['delta'])} to {value} "
                f"(was {fmt(row.get('baseline_value'))}), "
                f"{_value_stats(row)}")
        frag = _value_fragments(row, pack)
        if frag:
            line += "; " + frag
        return line
    if kind == "intersection":
        sec = row.get("secondary") or {}
        return (f"{head}: {value} {pack.get('label')}, "
                f"{fmt(sec.get('value'))} {pack.get('second_label')}")
    if kind == "breadth":
        n = pack.get("breadth_top_n") or pack.get("requested_n") or 10
        return (f"{head}: top {n} in {value} metrics: "
                f"{_breadth_finishes_text(row)}")
    if kind == "finishes":
        return (f"{head}: {value} top-{pack.get('finish_n') or 12} finishes "
                f"in {row.get('games')} games ({fmt(row.get('ppg'))} PPG)")
    if kind == "floor":
        return (f"{head}: floor {fmt(row.get('floor'))} points "
                f"(ceiling {fmt(row.get('ceiling'))}, "
                f"{fmt(row.get('ppg'))} PPG over {row.get('games')} games)")
    if kind == "ceiling":
        return (f"{head}: ceiling {fmt(row.get('ceiling'))} points "
                f"(floor {fmt(row.get('floor'))}, "
                f"{fmt(row.get('ppg'))} PPG over {row.get('games')} games)")
    if kind == "value":
        line = f"{head}: {value} value"
        if row.get("age") is not None:
            line += f", age {fmt(row.get('age'))}"
        return line
    if kind == "slate":
        return (f"{head}: {value} {pack.get('label')}, slate avg rank "
                f"{fmt(row.get('slate_avg_rank'))} over the next "
                f"{row.get('slate_games')} games")
    if kind == "gap":
        return (f"{head}: {value} {pack.get('label')} vs "
                f"{fmt(row.get('second_value'))} for "
                f"{row.get('second_name')} (gap {fmt(row.get('gap'))})")
    if kind == "rank_movers":
        return (f"{head}: now #{row.get('new_rank')} at "
                f"{row.get('position')} (was #{row.get('old_rank')}, up "
                f"{row.get('climb')}), value "
                f"{fmt(row.get('current_value'))}")
    if kind == "defenses":
        return (f"{head}: {value} points in {row.get('games')} games "
                f"({fmt(row.get('ppg'))} PPG)")
    line = f"{head}: {value}"
    if kind == "dual":
        sec = row.get("secondary")
        if sec:
            line += (f" - {pack.get('second_label')} {fmt(sec.get('value'))} "
                     f"(#{sec.get('rank')})")
        elif pack.get("second_label"):
            line += f" - {pack.get('second_label')} not qualified"
    return line


def _board_receipt(row, pack, position=None):
    kind = pack.get("board_kind")
    label = f"{row.get('rank')}. {row.get('name') or row.get('team')}"
    value = fmt(row.get("value")) if row.get("value") is not None else ""
    if kind == "matchup" and position:
        return {"label": f"{label} vs {position}", "value": value}
    if kind == "movers" and row.get("delta") is not None:
        return {"label": label, "value": f"{_fmt_delta(row['delta'])} (now {value})"}
    if kind in ("buy_low", "sell_high") and row.get("delta") is not None:
        if row.get("recent_ppg") is not None:
            stats = (f"{fmt(row.get('recent_ppg'))} PPG last "
                     f"{row.get('recent_games')} "
                     f"({_fmt_delta(row.get('recent_over_expected'))} "
                     f"vs expected)")
        else:
            stats = (f"{fmt(row.get('ppg'))} PPG "
                     f"({_fmt_delta(row.get('over_expected'))} vs expected)")
        text = f"value {_fmt_delta(row['delta'])} (now {value}); {stats}"
        if row.get("peer_median_value") is not None:
            text += f"; peer median {fmt(row.get('peer_median_value'))}"
        return {"label": label, "value": text}
    if kind == "intersection":
        sec = row.get("secondary") or {}
        return {"label": label,
                "value": f"{value} - {pack.get('second_label')} {fmt(sec.get('value'))}"}
    if kind == "breadth":
        n = pack.get("breadth_top_n") or pack.get("requested_n") or 10
        return {"label": label,
                "value": (f"top {n} in {value} metrics: "
                          f"{_breadth_finishes_text(row)}")}
    if kind == "finishes":
        return {"label": label,
                "value": f"{value} finishes ({fmt(row.get('ppg'))} PPG)"}
    if kind == "floor":
        return {"label": label,
                "value": f"floor {value} (ceiling {fmt(row.get('ceiling'))})"}
    if kind == "ceiling":
        return {"label": label,
                "value": f"ceiling {value} (floor {fmt(row.get('floor'))})"}
    if kind == "value":
        text = value
        if row.get("age") is not None:
            text += f" (age {fmt(row.get('age'))})"
        return {"label": label, "value": text}
    if kind == "slate":
        return {"label": label,
                "value": (f"{value} - slate avg rank "
                          f"{fmt(row.get('slate_avg_rank'))}")}
    if kind == "gap":
        return {"label": label,
                "value": (f"{value} vs {fmt(row.get('second_value'))} "
                          f"({row.get('second_name')})")}
    if kind == "rank_movers":
        return {"label": label,
                "value": (f"#{row.get('old_rank')} to #{row.get('new_rank')} "
                          f"(up {row.get('climb')}, value "
                          f"{fmt(row.get('current_value'))})")}
    if kind == "defenses":
        return {"label": label,
                "value": f"{value} points ({fmt(row.get('ppg'))} PPG)"}
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
        sentences = []
        focus = pack.get("focus") or {}
        ranks = focus.get("ranks") or []
        if len(ranks) > 1:
            listed = _join_and(
                [f"#{r.get('rank')} of {r.get('pool_size')} against "
                 f"{r.get('position')}s ({fmt(r.get('value'))} fantasy "
                 "points allowed per game)" for r in ranks])
            sentences.append(
                f"The {focus.get('name')} defense ranks {listed}.")
        elif ranks:
            r = ranks[0]
            sentences.append(
                f"The {focus.get('name')} defense ranks #{r.get('rank')} of "
                f"{r.get('pool_size')} against {r.get('position')}s, allowing "
                f"{fmt(r.get('value'))} fantasy points per game to "
                f"{r.get('position')}s.")
        boards = pack.get("boards") or []
        if len(boards) > 1:
            for sub in boards:
                sub_board = sub.get("board") or []
                if sub_board:
                    stop = sub_board[0]
                    sentences.append(
                        f"{stop.get('team')} is the softest matchup for "
                        f"{sub.get('position')}s at "
                        f"{fmt(stop.get('value'))} fantasy points "
                        "allowed per game.")
        else:
            sentences.append(
                f"{top.get('team')} is the softest matchup for "
                f"{pack.get('position_filter')}s at {value} fantasy points "
                "allowed per game.")
        return " ".join(sentences)
    if kind == "intersection":
        return (f"{pack.get('board_count')} players sit in the top "
                f"{pack.get('requested_n')} of both {label} and "
                f"{pack.get('second_label')}.")
    if kind == "movers":
        if top.get("delta") is not None:
            return (f"{top.get('name')} moved the most in {label} over the "
                    f"recent window, {_fmt_delta(top['delta'])} to {value}.")
        return f"{top.get('name')} leads the {label} board at {value}."
    if kind == "breadth":
        n = pack.get("breadth_top_n") or pack.get("requested_n") or 10
        word = pack.get("family_word") or ""
        pos = pack.get("position_filter")
        who = f"any other {pos}" if pos else "any other player"
        return (f"{top.get('name')} finishes in the top {n} of {value} of "
                f"the {pack.get('metric_count')} "
                f"{word + ' ' if word else ''}metrics counted here, more "
                f"than {who}.")
    if kind in ("buy_low", "sell_high") and top.get("delta") is not None:
        word = "buy-low" if kind == "buy_low" else "sell-high"
        if top.get("recent_ppg") is not None:
            scoring = (f"{fmt(top.get('recent_ppg'))} PPG against "
                       f"{fmt(top.get('recent_expected_ppg'))} expected "
                       f"over the last {top.get('recent_games')}")
        else:
            scoring = (f"{fmt(top.get('ppg'))} PPG against "
                       f"{fmt(top.get('expected_ppg'))} expected")
        return (f"{top.get('name')} tops the {word} board: value "
                f"{_fmt_delta(top['delta'])} to {value} while scoring "
                f"{scoring}.")
    if kind == "finishes":
        return (f"{top.get('name')} owns the most top-{pack.get('finish_n') or 12} "
                f"weekly finishes with {value} of them.")
    if kind == "floor":
        return (f"{top.get('name')} has the highest floor at "
                f"{fmt(top.get('floor'))} points in his worst week.")
    if kind == "ceiling":
        return (f"{top.get('name')} has the highest ceiling at "
                f"{fmt(top.get('ceiling'))} points in his best week.")
    if kind == "slate":
        return (f"{top.get('name')} has the easiest next-{pack.get('slate_weeks') or 4} "
                f"slate among the {label} leaders, an average matchup rank "
                f"of {fmt(top.get('slate_avg_rank'))}.")
    if kind == "gap":
        return (f"{top.get('name')} beats his own team's No. 2 option by "
                f"{fmt(top.get('gap'))} in {label}, the widest gap on "
                "any team.")
    if kind == "rank_movers":
        return (f"{top.get('name')} climbed {value} spots at "
                f"{top.get('position')} over the last "
                f"{pack.get('window_days') or 30} days, to "
                f"#{top.get('new_rank')}.")
    if kind == "defenses":
        focus = pack.get("focus") or {}
        period = (f"in {pack.get('target_season')}"
                  if pack.get("historical") else "this season")
        if focus:
            text = (f"The {focus.get('name')} defense ranks "
                    f"#{focus.get('rank')} of {focus.get('pool_size')} by "
                    f"fantasy points {period}: {fmt(focus.get('value'))} "
                    f"points in {focus.get('games')} games "
                    f"({fmt(focus.get('ppg'))} per game).")
            if focus.get("value_rank"):
                text += (f" By model trade value it ranks "
                         f"#{focus.get('value_rank')} of "
                         f"{focus.get('value_pool')} defenses.")
            return text
        return (f"{top.get('name')} leads all defenses {period} at {value} "
                f"fantasy points ({fmt(top.get('ppg'))} per game).")
    return (f"{top.get('name')} leads {label} at {value} across "
            f"{pack.get('pool_size')} qualified players.")


def _leaderboard(pack):
    # Mock mode has no model, so the intro is always the deterministic
    # template (the live composer may swap in a model framing sentence that
    # names no board entry; verdict, lines, and receipts are identical).
    board = pack.get("board") or []
    boards = pack.get("boards") or []
    if len(boards) > 1:
        lines = [_board_intro_template(pack)]
        receipts = []
        for sub in boards:
            pos = sub.get("position")
            lines.append(f"Fantasy points allowed to {pos}s:")
            lines += [_board_line(r, pack, pos)
                      for r in sub.get("board") or []]
            if sub.get("qualification_text"):
                lines.append(sub["qualification_text"])
            receipts += [_board_receipt(r, pack, pos)
                         for r in sub.get("board") or []]
        body = "\n".join(lines)
    else:
        body = "\n".join([_board_intro_template(pack)]
                         + [_board_line(r, pack) for r in board])
        qual = pack.get("qualification_text")
        if qual:
            body += "\n" + qual
        receipts = [_board_receipt(r, pack) for r in board]
    ans = _base("leaderboard", pack)
    ans["act_link"] = _board_link(pack) or ans["act_link"]
    ans.update({"verdict": _board_verdict(pack), "text": body,
                "receipts": receipts,
                "followups": [_fu("Who is rising in target share?"),
                              _fu("Who leads in air yards share?")]})
    return ans


def _playoffs(pack):
    me = pack.get("me") or {}
    scen = pack.get("scenario") or {}
    receipts = []
    if me.get("playoff_pct") is not None:
        receipts.append({"label": "Your playoff odds (%)", "value": fmt(me["playoff_pct"])})
    traj = pack.get("odds_trajectory") or {}
    if traj.get("start_pct") is not None and traj.get("start_week") is not None:
        receipts.append({"label": f"Your playoff odds in Week {traj['start_week']} (%)",
                         "value": fmt(traj["start_pct"])})
    if scen.get("wins_to_clinch") is not None:
        receipts.append({"label": "Wins to clinch", "value": fmt(scen["wins_to_clinch"])})
    elif me.get("wins_still_needed") is not None:
        receipts.append({"label": "Wins still needed (projected)",
                         "value": fmt(me["wins_still_needed"])})
    if pack.get("weeks_remaining") is not None:
        receipts.append({"label": "Weeks remaining", "value": fmt(pack["weeks_remaining"])})
    ans = _base("playoffs", pack)
    if pack.get("odds_soft"):
        ans.update({
            "verdict": "Too early to call",
            "text": (f"The odds are soft this early, so read the percentage loosely: the sim puts "
                     f"{me.get('team_name')} at {fmt(me.get('playoff_pct'))} percent with "
                     f"{fmt(pack.get('weeks_remaining'))} weeks left. The wins math is firmer: "
                     f"the projected cut line is {fmt(pack.get('projected_cut_wins'))} wins, so "
                     f"you need about {fmt(me.get('wins_still_needed'))} more wins from here."),
            "receipts": receipts[:4],
            "followups": [_fu("What are my odds of a first-round bye?"), _fu("Who is on the bubble?")],
        })
        return ans
    if scen:
        text = (f"The sim has {me.get('team_name')} at {fmt(me.get('playoff_pct'))} percent. "
                f"{scen.get('summary')}.")
        if scen.get("next_opponent"):
            text += f" Your next game is against {scen['next_opponent']}."
        ans.update({"verdict": scen.get("summary") or "Alive", "text": text,
                    "receipts": receipts[:4],
                    "followups": [_fu("What are my odds of a first-round bye?"),
                                  _fu("How hard is my remaining schedule?")]})
        return ans
    text = (f"{me.get('team_name')} sits at {me.get('record')} with "
            f"{fmt(pack.get('weeks_remaining'))} weeks left. Your projected finish is "
            f"{fmt(me.get('avg_final_wins'))} wins against a projected cut line of "
            f"{fmt(pack.get('projected_cut_wins'))} wins, so the math says about "
            f"{fmt(me.get('wins_still_needed'))} more wins gets you in.")
    if traj.get("delta") is not None:
        text += (f" Your odds have moved from {fmt(traj.get('start_pct'))} percent in Week "
                 f"{traj.get('start_week')} to {fmt(traj.get('current_pct'))} percent in "
                 f"Week {traj.get('current_week')}.")
        movers = pack.get("odds_movers") or {}
        risers = movers.get("risers") or []
        if risers:
            top_riser = risers[0]
            text += (f" The biggest riser in the league is {top_riser.get('team_name')}, "
                     f"up from {fmt(top_riser.get('start_pct'))} to "
                     f"{fmt(top_riser.get('current_pct'))} percent.")
    ans.update({
        "verdict": f"{fmt(me.get('playoff_pct'))}% playoff odds",
        "text": text,
        "receipts": receipts[:4],
        "followups": [_fu("What are my odds of a first-round bye?"), _fu("Who is on the bubble?")],
    })
    return ans


def _power_rankings(pack):
    me = pack.get("me") or {}
    leader = pack.get("leader") or {}
    receipts = []
    if me.get("rank") is not None:
        receipts.append({"label": "Your power rank", "value": fmt(me["rank"])})
    if me.get("power_score") is not None:
        receipts.append({"label": "Your power score", "value": fmt(me["power_score"])})
    if leader.get("power_score") is not None and leader.get("roster_id") != me.get("roster_id"):
        receipts.append({"label": f"{leader.get('team_name')} power score",
                         "value": fmt(leader["power_score"])})
    rows = pack.get("teams") or []
    second = rows[1] if len(rows) > 1 else {}
    if (second.get("power_score") is not None and len(receipts) < 4
            and second.get("roster_id") not in (me.get("roster_id"), leader.get("roster_id"))):
        receipts.append({"label": f"{second.get('team_name')} power score",
                         "value": fmt(second["power_score"])})
    ans = _base("power_rankings", pack)
    if leader.get("roster_id") == me.get("roster_id"):
        verdict = "You are the team to beat"
    else:
        verdict = f"{leader.get('team_name')} is the team to beat"
    text = (f"{leader.get('team_name')} tops the board at {fmt(leader.get('power_score'))}. "
            f"You sit at number {fmt(me.get('rank'))} with a power score of "
            f"{fmt(me.get('power_score'))} and a {me.get('record')} record.")
    ans.update({"verdict": verdict, "text": text, "receipts": receipts[:4],
                "followups": [_fu("What do the standings look like?"), _fu("What are my playoff odds?")]})
    return ans


def _standings(pack):
    me = pack.get("me") or {}
    leader = pack.get("points_leader") or {}
    receipts = []
    if me.get("seed") is not None:
        receipts.append({"label": "Your seed", "value": fmt(me["seed"])})
    if me.get("pf") is not None:
        receipts.append({"label": "Your points for", "value": fmt(me["pf"])})
    if leader.get("pf") is not None and leader.get("team_name") != me.get("team_name"):
        receipts.append({"label": f"{leader.get('team_name')} points for",
                         "value": fmt(leader["pf"])})
    text = (f"You are {me.get('record')} with {fmt(me.get('pf'))} points for. "
            f"{leader.get('team_name')} leads the league in scoring at "
            f"{fmt(leader.get('pf'))} points for.")
    divs = pack.get("division_leaders") or []
    if divs:
        named = " and ".join(f"{d.get('team_name')} in Division {fmt(d.get('division'))}"
                              for d in divs)
        text += f" Division leaders: {named}."
    ans = _base("standings", pack)
    ans.update({"verdict": f"Seed {fmt(me.get('seed'))} at {me.get('record')}", "text": text,
                "receipts": receipts[:4],
                "followups": [_fu("Where am I in the power rankings?"), _fu("What are my playoff odds?")]})
    return ans


def _matchup(pack):
    m = pack.get("matchup") or {}
    a, b = m.get("team_a") or {}, m.get("team_b") or {}
    receipts = []
    if a.get("win_prob_pct") is not None:
        receipts.append({"label": f"{a.get('team_name')} win probability (%)",
                         "value": fmt(a["win_prob_pct"])})
    if a.get("projected") is not None:
        receipts.append({"label": f"{a.get('team_name')} projected", "value": fmt(a["projected"])})
    if b.get("projected") is not None:
        receipts.append({"label": f"{b.get('team_name')} projected", "value": fmt(b["projected"])})
    favored = a if (a.get("win_prob_pct") or 0) >= (b.get("win_prob_pct") or 0) else b
    text = (f"{a.get('team_name')} projects {fmt(a.get('projected'))} against "
            f"{fmt(b.get('projected'))} for {b.get('team_name')}. The win probability is "
            f"{fmt(a.get('win_prob_pct'))} percent for {a.get('team_name')} and "
            f"{fmt(b.get('win_prob_pct'))} percent for {b.get('team_name')}.")
    ans = _base("matchup", pack)
    ans.update({"verdict": f"{favored.get('team_name')} favored", "text": text,
                "receipts": receipts[:4],
                "followups": [_fu("Who should I start at FLEX?"), _fu("What are my playoff odds?")]})
    if (pack.get("margin") or {}).get("thin"):
        ans["caveat"] = "Close call: the win probability is nearly even."
    return ans


def _schedule(pack):
    me = pack.get("me") or {}
    easiest = pack.get("easiest") or {}
    hardest = pack.get("hardest") or {}
    stretch = (pack.get("playoff_stretch") or {}).get("me") or {}
    receipts = []
    if stretch.get("opp_index") is not None:
        receipts.append({"label": "Your playoff stretch opponent index",
                         "value": fmt(stretch["opp_index"])})
    if me.get("opp_index") is not None:
        receipts.append({"label": "Your remaining opponent index", "value": fmt(me["opp_index"])})
    if easiest.get("opp_index") is not None and easiest.get("roster_id") != me.get("roster_id"):
        receipts.append({"label": f"{easiest.get('team_name')} opponent index",
                         "value": fmt(easiest["opp_index"])})
    text = (f"{easiest.get('team_name')} has the softest remaining slate at "
            f"{fmt(easiest.get('opp_index'))} on the opponent strength index, where higher "
            f"means tougher. Yours rates {fmt(me.get('opp_index'))}, and "
            f"{hardest.get('team_name')} has the toughest at {fmt(hardest.get('opp_index'))}.")
    if stretch.get("opp_index") is not None:
        opps = ", ".join(o.get("opponent") or "" for o in stretch.get("opponents") or [])
        text += (f" Your playoff stretch rates {fmt(stretch['opp_index'])}"
                 + (f", against {opps}." if opps else "."))
    ans = _base("schedule", pack)
    ans.update({"verdict": f"{easiest.get('team_name')} has the easiest remaining schedule",
                "text": text, "receipts": receipts[:4],
                "followups": [_fu("What are my playoff odds?"), _fu("Break down my matchup")]})
    return ans


def _side_receipts(f):
    return [{"label": "You receive (total value)", "value": fmt(f.get("receive_total"))},
            {"label": "You give (total value)", "value": fmt(f.get("give_total"))},
            {"label": "Fairness gap", "value": fmt(f.get("fairness_gap"))}]


def _names(players):
    return " and ".join(p["name"] for p in players or [])


def _position_no_fit_answer(pack):
    """Mirror of compose._position_no_fit_answer (same strings)."""
    pos = str((pack or {}).get("give_position") or "").upper()
    target = ((pack or {}).get("target_player") or {}).get("name") or "that player"
    fb = (pack or {}).get("fallback_finding") or {}
    text = f"No {pos} package on your roster gets within range of {target}."
    if (pack or {}).get("partner_scoped") and (pack or {}).get("partner_team"):
        text = (f"No {pos} package on your roster gets you {target} from "
                f"{pack['partner_team']}.")
    receipts = []
    if fb and fb.get("receive_total") is not None and fb.get("give_total") is not None:
        give = fb.get("give") or []
        anchor = max(give, key=lambda p: float(p.get("value") or 0)) if give else {}
        fb_pos = str(anchor.get("position") or "").upper()
        gap = fb.get("fairness_gap")
        gap_clause = ""
        if isinstance(gap, (int, float)):
            gap_clause = f", a fairness gap of {fmt(abs(float(gap)))}"
        text += (f" Closest deal overall is a {fb_pos} deal: you send "
                 f"{_names(give)} to {fb.get('partner_team')} for "
                 f"{target}, totals {fmt(fb.get('receive_total'))} against "
                 f"{fmt(fb.get('give_total'))}{gap_clause}.")
        receipts = _side_receipts(fb)
    ans = _base("trade_finding", pack)
    ans.update({"verdict": f"No {pos} deal for {target}", "text": text,
                "receipts": receipts,
                "followups": [_fu(f"Who can I trade for {target}?"),
                              _fu("What should I do with my team this week?")]})
    return ans


def _trade_finding(pack):
    if pack.get("gated"):
        return gated_answer(pack)
    if pack.get("position_no_fit"):
        return _position_no_fit_answer(pack)
    equiv = (pack.get("pick_equivalents") or {}).get("equivalents") or []
    if equiv:
        sell = pack.get("sell_player") or {}
        top = equiv[0]
        receipts = [{"label": f"{sell.get('name')} value", "value": fmt(sell.get("value"))}]
        for e in equiv[:2]:
            receipts.append({"label": str(e.get("label") or "Pick"),
                             "value": fmt(e.get("value"))})
        text = (f"{sell.get('name')} is valued at {fmt(sell.get('value'))}. That buys "
                f"about a {top.get('label')} at {fmt(top.get('value'))}")
        if len(equiv) > 1:
            text += f", or {equiv[1].get('label')} at {fmt(equiv[1].get('value'))}"
        text += "."
        ans = _base("trade_finding", pack)
        ans.update({"verdict": str(top.get("label") or ""), "text": text,
                    "receipts": receipts[:4],
                    "followups": [_fu("Who else could I target?"),
                                  _fu("What should I do with my team this week?")]})
        return ans
    findings = pack.get("findings") or []
    ans = _base("trade_finding", pack)
    if findings:
        top = findings[0]
        link = _finding_link(pack, top)
        if link:
            ans["act_link"] = link
        recv, give = top.get("receive") or [], top.get("give") or []
        ans.update({
            "verdict": f"Trade target: {_names(recv[:1])} from {top.get('partner_team')}",
            "text": (f"{top.get('partner_team')} fits on both sides. You receive {_names(recv)} "
                     f"worth {fmt(top.get('receive_total'))}; you send {_names(give)} worth "
                     f"{fmt(top.get('give_total'))}, a fairness gap of "
                     f"{fmt(abs(float(top.get('fairness_gap') or 0)))}."),
            "receipts": _side_receipts(top),
            "followups": [_fu("Who else could I target?"), _fu("What can I get for my spare players?")],
        })
        return ans
    blocked = pack.get("partner_blocked") or {}
    if blocked:
        partner = blocked.get("partner_team") or "That team"
        target = (pack.get("target_player") or {}).get("name") or "that target"
        bits = "; ".join(
            f"{a.get('name')}: {a.get('reason')}"
            for a in blocked.get("partner_appeal") or [] if not a.get("appeal"))
        reason = f", because {bits}" if bits else ""
        lead = (f"No clean fit right now. The packages that reach {target} are not ones "
                f"{partner} would take. {partner} would likely pass on the straight "
                f"package: {_names(blocked.get('receive') or [])} for "
                f"{_names(blocked.get('give') or [])} totals "
                f"{fmt(blocked.get('receive_total'))} against "
                f"{fmt(blocked.get('give_total'))}{reason}.")
        counter = pack.get("counter") or {}
        if counter:
            slots = counter.get("slots") or {}
            vu, vd = slots.get("viewer_upgrade") or {}, slots.get("viewer_downgrade") or {}
            pu, pd = slots.get("partner_upgrade") or {}, slots.get("partner_downgrade") or {}
            vd_s = (f"the {vd.get('player')} is a downgrade, a pick in place of a player"
                    if vd.get("position") == "PICK" else
                    f"{vd.get('player')} is a downgrade at {vd.get('position')}, a step "
                    f"below your current starter there")
            pu_s = (f"your {pu.get('player')} is the upgrade, a future pick for his window"
                    if pu.get("position") == "PICK" else
                    f"{pu.get('player')} is the upgrade at {pu.get('position')}")
            pd_s = (f"the {pd.get('player')} is the downgrade, a pick in place of a player"
                    if pd.get("position") == "PICK" else
                    f"{pd.get('player')} is the downgrade at {pd.get('position')}, a "
                    f"startable piece a step below the one they send back")
            read = counter.get("profile_read")
            text = (lead
                    + f" You could try {_names(counter.get('give') or [])} for "
                      f"{_names(counter.get('receive') or [])} instead: totals "
                      f"{fmt(counter.get('receive_total'))} against "
                      f"{fmt(counter.get('give_total'))}, a fairness gap of "
                      f"{fmt(abs(float(counter.get('fairness_gap') or 0)))}."
                    + (f" {read}" if read else "")
                    + f" For you, {vu.get('player')} is the upgrade at {vu.get('position')}; "
                      f"{vd_s}. For {partner}, {pu_s}; {pd_s}.")
            ans.update({
                "verdict": "No clean fit right now", "text": text,
                "receipts": _side_receipts(counter),
                "followups": [_fu("Who is available on waivers?"),
                              _fu("What should I do with my team this week?")],
            })
            return ans
        ans.update({
            "verdict": "No clean fit right now",
            "text": (lead + f" Nothing in range is built from pieces {partner} needs. "
                     "The nearest alternative is the waiver wire."),
            "receipts": _side_receipts(blocked),
            "followups": [_fu("Who is available on waivers?"),
                          _fu("What should I do with my team this week?")],
        })
        return ans
    miss = pack.get("closest_miss") or {}
    if miss:
        ans.update({
            "verdict": "No clean fit right now",
            "text": (f"No clean fit right now. The closest miss is {miss.get('partner_team')}: "
                     f"{_names(miss.get('receive') or [])} for {_names(miss.get('give') or [])} "
                     f"totals {fmt(miss.get('receive_total'))} against "
                     f"{fmt(miss.get('give_total'))}, a fairness gap of "
                     f"{fmt(abs(float(miss.get('fairness_gap') or 0)))}, outside the fair band. "
                     "The nearest alternative is the waiver wire."),
            "receipts": _side_receipts(miss),
            "followups": [_fu("Who is available on waivers?"), _fu("What should I do with my team this week?")],
        })
        return ans
    ans.update({
        "verdict": "No clean fit right now",
        "text": ("No clean fit right now: no roster in your league lines up on both sides "
                 "inside the fair band. The nearest alternative is the waiver wire."),
        "receipts": [],
        "followups": [_fu("Who is available on waivers?"), _fu("What should I do with my team this week?")],
    })
    return ans

def _breakout_board(pack):
    cands = pack.get("candidates") or []
    receipts = [{"label": c["name"], "value": fmt(c.get("breakout_score"))}
                for c in cands[:3] if c.get("breakout_score") is not None]
    top = cands[0] if cands else {}
    text = (f"{top.get('name')} leads the {pack.get('week_label')} breakout calls "
            f"at {fmt(top.get('breakout_score'))}.")
    if len(cands) > 1:
        text += f" {cands[1].get('name')} is next at {fmt(cands[1].get('breakout_score'))}."
    if pack.get("preview") and pack.get("locked_count"):
        text += f" {fmt(pack.get('locked_count'))} more calls are on the full board."
    ans = _base("breakout_board", pack)
    ans.update({"verdict": f"{pack.get('week_label')} breakout calls", "text": text,
                "receipts": receipts[:4],
                "followups": [_fu("How good is BR at breakout calls?")]})
    return ans


def _breakout_track_record(pack):
    o = pack.get("overall") or {}
    receipts = [{"label": "Graded calls", "value": fmt(o.get("graded"))},
                {"label": "Hit rate (percent)", "value": fmt(o.get("hit_rate_pct"))},
                {"label": "Hits", "value": fmt(o.get("hit"))},
                {"label": "Misses", "value": fmt(o.get("miss"))}]
    text = (f"BR has graded {fmt(o.get('graded'))} of {fmt(o.get('calls'))} breakout calls "
            f"this season, with {fmt(o.get('hit'))} hits and a hit rate of "
            f"{fmt(o.get('hit_rate_pct'))} percent.")
    ans = _base("breakout_track_record", pack)
    ans.update({"verdict": f"Hit rate {fmt(o.get('hit_rate_pct'))} percent", "text": text,
                "receipts": receipts,
                "followups": [_fu("Who are this week's breakout calls?")]})
    return ans


def _breakout_history(pack):
    player = pack.get("player") or {}
    name = player.get("name")
    ans = _base("breakout_history", pack)
    if pack.get("was_call"):
        earliest = pack.get("earliest_call") or {}
        text = f"Yes. {name} was first surfaced as a breakout call for {earliest.get('label')}."
        if pack.get("grade"):
            text += f" That call graded as a {pack['grade']}."
        receipts = [{"label": "First breakout call", "value": str(earliest.get("label"))}]
        if earliest.get("breakout_score") is not None:
            receipts.append({"label": "Breakout score", "value": fmt(earliest["breakout_score"])})
        ans.update({"verdict": f"YES, {name} was a breakout call", "text": text,
                    "receipts": receipts,
                    "followups": [_fu("Who are this week's breakout calls?")]})
    else:
        ans.update({"verdict": f"NO, {name} was not a breakout call",
                    "text": f"No. {name} has not been surfaced as a breakout call this season.",
                    "receipts": [], "followups": [_fu("Who are this week's breakout calls?")]})
    return ans


def _compose_sweep(pack):
    """Mirror of compose._compose_sweep (deterministic rendering)."""
    s = pack.get("sweep") or {}
    a, b = s.get("team_a") or {}, s.get("team_b") or {}
    an, bn = a.get("team_name") or "Team A", b.get("team_name") or "Team B"
    aw, bw = s.get("a_wins") or 0, s.get("b_wins") or 0
    if aw > bw:
        verdict = f"{an} wins the category sweep {aw}-{bw}"
        lead = f"{an} takes the sweep {aw} categories to {bw}"
    elif bw > aw:
        verdict = f"{bn} wins the category sweep {bw}-{aw}"
        lead = f"{bn} takes the sweep {bw} categories to {aw}"
    else:
        verdict = f"Dead even across categories at {aw}-{bw}"
        lead = f"It is dead even at {aw} categories apiece"
    if s.get("ties"):
        lead += f", with {s['ties']} tied"
    lines = [lead + "."]
    receipts = []
    for c in s.get("categories") or []:
        winner = {"a": an, "b": bn}.get(c.get("winner"), "Tied")
        lines.append(f"{c.get('category')}: {an} "
                     f"{fmt(c.get('a_value'))} vs {bn} "
                     f"{fmt(c.get('b_value'))} ({winner})")
        if len(receipts) < 4:
            receipts.append({
                "label": f"{c.get('category')} ({an} vs {bn})",
                "value": (f"{fmt(c.get('a_value'))} to "
                          f"{fmt(c.get('b_value'))}")})
    ans = _base("teams", pack)
    ans.update({"verdict": verdict, "text": "\n".join(lines),
                "receipts": receipts,
                "followups": [_fu("Grade every team's WR room"),
                              _fu("Who has the strongest roster?")]})
    return ans


def _teams(pack):
    if pack.get("sweep"):
        return _compose_sweep(pack)
    focus = pack.get("position_focus")
    ranking = pack.get("position_ranking") or []
    ans = _base("teams", pack)
    if focus and ranking:
        first, last = ranking[0], ranking[-1]
        receipts = [{"label": f"Best {focus} room ({first.get('team_name')})",
                     "value": fmt(first.get("strength"))},
                    {"label": f"Weakest {focus} room ({last.get('team_name')})",
                     "value": fmt(last.get("strength"))}]
        text = (f"{last.get('team_name')} has the weakest {focus} room at "
                f"{fmt(last.get('strength'))}. {first.get('team_name')} has the "
                f"strongest at {fmt(first.get('strength'))}.")
        ans.update({"verdict": f"Weakest {focus} room: {last.get('team_name')}",
                    "text": text, "receipts": receipts,
                    "followups": [_fu("Whose roster is strongest?")]})
        return ans
    s = pack.get("strongest") or {}
    w = pack.get("weakest") or {}
    mine = next((t for t in pack.get("teams") or [] if t.get("is_viewer")), {})
    receipts = [{"label": f"Strongest roster ({s.get('team_name')}) grade score",
                 "value": fmt(s.get("grade_score"))}]
    if mine.get("grade_score") is not None:
        receipts.append({"label": "Your grade score", "value": fmt(mine["grade_score"])})
    if w.get("grade_score") is not None:
        receipts.append({"label": f"Weakest roster ({w.get('team_name')}) grade score",
                         "value": fmt(w.get("grade_score"))})
    text = (f"{s.get('team_name')} has the strongest roster in {pack.get('mode_label')} "
            f"mode with a grade score of {fmt(s.get('grade_score'))}.")
    breadth_top = pack.get("most_top_players") or {}
    if breadth_top.get("count") is not None:
        receipts.append({"label": (f"Most top-{pack.get('breadth_n') or 24} "
                                   f"players ({breadth_top.get('team_name')})"),
                         "value": fmt(breadth_top["count"])})
        text += (f" {breadth_top.get('team_name')} rosters the most "
                 f"top-{pack.get('breadth_n') or 24} players with "
                 f"{fmt(breadth_top['count'])}.")
    ans.update({"verdict": f"Strongest roster: {s.get('team_name')}", "text": text,
                "receipts": receipts[:4], "followups": [_fu("Who is weakest at TE?")]})
    return ans


def _pick_ownership(pack):
    pick = pack.get("pick") or {}
    owner = pack.get("current_owner_team")
    receipts = []
    if pick.get("value") is not None:
        receipts.append({"label": f"{pick.get('label')} value", "value": fmt(pick["value"])})
    receipts.append({"label": "Pick", "value": str(pick.get("label"))})
    ans = _base("pick_ownership", pack)
    if pack.get("owned_by_me"):
        ans.update({"verdict": f"You own the {pick.get('label')}",
                    "text": f"You still own your {pick.get('label')}.",
                    "receipts": receipts, "followups": [_fu("What other picks do I own?")]})
    else:
        ans.update({"verdict": f"{owner} owns the {pick.get('label')}",
                    "text": f"Your {pick.get('label')} now belongs to {owner}.",
                    "receipts": receipts, "followups": [_fu("What other picks do I own?")]})
    return ans



def _keeper_receipts(pack):
    """Mirror of compose._required_receipts for keeper packs."""
    rows = []
    if (pack.get("mode") or "") == "dynasty":
        for p in (pack.get("roster_core") or [])[:2]:
            if p.get("value") is not None:
                rows.append({"label": f"{p.get('name')} value", "value": fmt(p["value"])})
        if pack.get("roster_value_total") is not None and len(rows) < 4:
            rows.append({"label": "Roster value total",
                         "value": fmt(pack["roster_value_total"])})
        return rows[:4]
    for p in (pack.get("my_keepers") or [])[:2]:
        if p.get("value") is not None:
            rows.append({"label": f"{p.get('name')} value", "value": fmt(p["value"])})
        if p.get("cost_round") is not None and len(rows) < 4:
            rows.append({"label": f"{p.get('name')} keeper cost (round)",
                         "value": fmt(p["cost_round"])})
    asked = pack.get("asked_player") or {}
    if asked.get("value") is not None and len(rows) < 4:
        rows.append({"label": f"{asked.get('name')} value", "value": fmt(asked["value"])})
    return rows[:4]


def _keeper(pack):
    ans = _base("keeper", pack)
    league_name = (pack.get("league") or {}).get("league_name") or "this league"
    if (pack.get("mode") or "") == "dynasty":
        core = pack.get("roster_core") or []
        top = core[0] if core else {}
        text = (f"In {league_name} there is no keeper decision to make: a dynasty "
                f"roster carries over in full. Your core is led by {top.get('name')} "
                f"at {fmt(top.get('value'))}, the roster totals "
                f"{fmt(pack.get('roster_value_total'))} in value, and the average "
                f"age is {fmt(pack.get('roster_avg_age'))}.")
        ans.update({"verdict": "Dynasty: the whole roster carries over",
                    "text": text, "receipts": _keeper_receipts(pack),
                    "followups": [_fu("Who is first on my cut line?")]})
        return ans
    keepers = pack.get("my_keepers") or []
    asked = pack.get("asked_player") or {}
    listing = "; ".join(
        f"{k['name']} (round {fmt(k.get('cost_round'))}, value {fmt(k.get('value'))})"
        for k in keepers)
    text = f"Your projected keepers in {league_name}: {listing}."
    cut = pack.get("cut_line") or []
    if cut:
        text += (f" The cut line starts with {cut[0]['name']} at "
                 f"{fmt(cut[0].get('value'))}.")
    verdict = f"Your {fmt(pack.get('keeper_limit'))} keepers in {league_name}"
    if asked.get("name"):
        if asked.get("projected_keeper"):
            verdict = f"KEEP {asked['name']}"
            text = (f"Yes. {asked['name']} projects as a keeper at a round "
                    f"{fmt(asked.get('cost_round'))} cost with a value of "
                    f"{fmt(asked.get('value'))}. " + text)
        else:
            verdict = f"NOT A KEEPER: {asked['name']}"
            text = (f"No. {asked['name']} (value {fmt(asked.get('value'))}) sits "
                    f"outside your projected keepers. " + text)
    ans.update({"verdict": verdict, "text": text, "receipts": _keeper_receipts(pack),
                "followups": [_fu("Who is first on my cut line?")]})
    return ans


def _rookie_receipts(pack):
    """Mirror of compose._required_receipts for rookie packs."""
    rows = []
    prospect = pack.get("prospect") or {}
    board = [prospect] if prospect else (pack.get("prospects") or [])
    for p in board[:2]:
        if p.get("prospect_score") is not None:
            rows.append({"label": f"{p.get('name')} prospect score",
                         "value": fmt(p["prospect_score"])})
    if prospect and prospect.get("projected_pick") is not None and len(rows) < 4:
        rows.append({"label": "Projected draft pick",
                     "value": fmt(prospect["projected_pick"])})
    line = prospect.get("college_latest") or {}
    for key, label in (("receiving_yards", "Receiving yards (latest college season)"),
                       ("rush_yards", "Rushing yards (latest college season)"),
                       ("pass_yards", "Passing yards (latest college season)")):
        if line.get(key) is not None and len(rows) < 4:
            rows.append({"label": label, "value": fmt(line[key])})
            break
    return rows[:4]


def _rookie(pack):
    ans = _base("rookie", pack)
    year = fmt(pack.get("draft_year"))
    prospect = pack.get("prospect") or {}
    if prospect:
        line = prospect.get("college_latest") or {}
        segs = []
        if line.get("pass_yards") is not None:
            segs.append(f"{fmt(line['pass_yards'])} passing yards and "
                        f"{fmt(line.get('pass_tds'))} passing touchdowns")
        if line.get("rush_yards") is not None:
            segs.append(f"{fmt(line['rush_yards'])} rushing yards and "
                        f"{fmt(line.get('rush_tds'))} rushing touchdowns")
        if line.get("receiving_yards") is not None:
            segs.append(f"{fmt(line.get('receptions'))} catches for "
                        f"{fmt(line['receiving_yards'])} yards and "
                        f"{fmt(line.get('receiving_tds'))} receiving touchdowns "
                        f"on {fmt(line.get('targets'))} targets")
        text = (f"{prospect['name']} is the {prospect.get('position')}"
                f"{fmt(prospect.get('position_rank'))} and number "
                f"{fmt(prospect.get('overall_rank'))} overall in the {year} class "
                f"out of {prospect.get('school')}, with a prospect score of "
                f"{fmt(prospect.get('prospect_score'))}.")
        if segs:
            text += (f" Latest college season at {prospect.get('school')}: "
                     f"{fmt(line.get('games_played'))} games, "
                     + " and ".join(segs) + ".")
        ans.update({"verdict": f"{prospect['name']}: {prospect.get('position')}"
                               f"{fmt(prospect.get('position_rank'))} in the {year} class",
                    "text": text, "receipts": _rookie_receipts(pack),
                    "followups": [_fu("Who are the top quarterbacks in the class?")]})
        return ans
    board = pack.get("prospects") or []
    top = board[0] if board else {}
    pos = pack.get("position_filter")
    listing = "; ".join(
        f"{p['name']} ({p.get('school')}) at {fmt(p.get('prospect_score'))}"
        for p in board[:5])
    verdict = (f"{year} {pos} class: {top.get('name')} leads" if pos
               else f"{year} class: {top.get('name')} leads")
    ans.update({"verdict": verdict,
                "text": f"The board: {listing}.",
                "receipts": _rookie_receipts(pack),
                "followups": [_fu("Who are the top quarterbacks in the class?")]})
    return ans


def _cross_league(pack):
    ans = _base("cross_league", pack)
    leagues = [s for s in (pack.get("leagues") or []) if isinstance(s, dict)]
    rows = []
    for s in leagues:
        if len(rows) >= 4:
            break
        if s.get("rank") is None:
            continue
        if s.get("record"):
            rows.append({"label": f"{s.get('league_name')} record",
                         "value": str(s["record"])})
        if len(rows) < 4:
            rows.append({"label": f"{s.get('league_name')} rank",
                         "value": fmt(s["rank"])})
    lines = []
    for s in leagues:
        if s.get("rank") is None and not s.get("record"):
            lines.append(f"{s.get('league_name')}: data is unavailable right now.")
            continue
        seg = (f"{s.get('league_name')}: {s.get('record')}, rank "
               f"{fmt(s.get('rank'))} of {fmt(s.get('total_teams'))}")
        qr = (s.get("pos_ranks") or {}).get("QB")
        if qr is not None:
            seg += f"; your QB room ranks {fmt(qr)}"
        lines.append(seg + ".")
    ranked = [s for s in leagues if s.get("rank") and s.get("total_teams")]
    verdict = "Your leagues at a glance"
    if ranked:
        best = min(ranked, key=lambda s: s["rank"] / s["total_teams"])
        verdict = f"Strongest: {best.get('league_name')}"
    ans.update({"verdict": verdict, "text": " ".join(lines), "receipts": rows,
                "followups": [_fu("Which of my teams is strongest?")]})
    return ans


def _replacement(pack):
    ans = _base("replacement", pack)
    subject = pack.get("subject") or {}
    cands = pack.get("candidates") or []
    top = cands[0] if cands else {}
    mode = pack.get("mode") or "younger"
    receipts = []
    if subject.get("value") is not None:
        receipts.append({"label": f"{subject.get('name')} value",
                         "value": fmt(subject.get("value"))})
    if subject.get("age") is not None:
        receipts.append({"label": f"{subject.get('name')} age",
                         "value": fmt(subject.get("age"))})
    if top.get("value") is not None:
        receipts.append({"label": f"{top.get('name')} value",
                         "value": fmt(top.get("value"))})
    if top.get("age") is not None and len(receipts) < 4:
        receipts.append({"label": f"{top.get('name')} age",
                         "value": fmt(top.get("age"))})
    text = (f"{top.get('name')} is the closest {mode} replacement for "
            f"{subject.get('name')}. {subject.get('name')} is valued at "
            f"{fmt(subject.get('value'))} at age {fmt(subject.get('age'))}; "
            f"{top.get('name')} is valued at {fmt(top.get('value'))} at age "
            f"{fmt(top.get('age'))} and sits on {top.get('location')}.")
    ans.update({"verdict": top.get("name"), "text": text, "receipts": receipts[:4],
                "followups": [_fu("What could I get for him in a trade?"),
                              _fu("Who is his handcuff?")]})
    return ans


def _handcuff(pack):
    ans = _base("handcuff", pack)
    subject = pack.get("subject") or {}
    nxt = pack.get("next_up") or {}
    receipts = []
    if nxt.get("value") is not None:
        receipts.append({"label": f"{nxt.get('name')} value",
                         "value": fmt(nxt.get("value"))})
    if subject.get("value") is not None:
        receipts.append({"label": f"{subject.get('name')} value",
                         "value": fmt(subject.get("value"))})
    mine = pack.get("my_options") or []
    if mine and mine[0].get("value") is not None and len(receipts) < 4:
        receipts.append({"label": f"{mine[0].get('name')} value",
                         "value": fmt(mine[0].get("value"))})
    text = (f"{nxt.get('name')} is next up behind {subject.get('name')} for "
            f"{subject.get('team')}. He is valued at {fmt(nxt.get('value'))} and "
            f"sits on {nxt.get('location')}.")
    if mine:
        text += (f" On your roster, {mine[0].get('name')} is your best in-house "
                 f"option at {fmt(mine[0].get('value'))}.")
    ans.update({"verdict": nxt.get("name"), "text": text, "receipts": receipts[:4],
                "followups": [_fu("Who should I start if he is out?"),
                              _fu("Who is available on waivers?")]})
    return ans


def _activity(pack):
    counts = pack.get("counts") or {}
    latest = pack.get("latest_trade") or {}
    teams = [t for t in (latest.get("teams") or []) if isinstance(t, dict)]
    receipts = []
    for t in teams[:2]:
        if t.get("gets_value") is not None:
            receipts.append({"label": f"{t.get('team_name')} received (total value)",
                             "value": fmt(t["gets_value"])})
    if latest.get("value_gap") is not None and len(receipts) < 4:
        receipts.append({"label": "Fairness gap", "value": fmt(latest["value_gap"])})
    if not receipts:
        if counts.get("trades") is not None:
            receipts.append({"label": "Trades this season", "value": fmt(counts["trades"])})
        if counts.get("waiver_adds") is not None and len(receipts) < 4:
            receipts.append({"label": "Waiver adds this season",
                             "value": fmt(counts["waiver_adds"])})
    ans = _base("activity", pack)
    ans["act_link"] = _league_link("activity", pack) or ans["act_link"]
    followups = [_fu("Who should I add from waivers?"),
                 _fu("What should I do with my team this week?")]
    if len(teams) < 2:
        ans.update({"verdict": "No trades yet this season",
                    "text": (f"Your league data shows {fmt(counts.get('trades'))} trades and "
                             f"{fmt(counts.get('waiver_adds'))} waiver adds so far this season."),
                    "receipts": receipts, "followups": followups})
        return ans
    a, b = teams[0], teams[1]
    a_gets = ", ".join(p.get("name") or "" for p in a.get("gets") or [])
    b_gets = ", ".join(p.get("name") or "" for p in b.get("gets") or [])
    lines = [f"The latest deal, in Week {fmt(latest.get('week'))}: "
             f"{a.get('team_name')} received {a_gets} ({fmt(a.get('gets_value'))} in value), "
             f"and {b.get('team_name')} received {b_gets} ({fmt(b.get('gets_value'))} in value)."]
    if latest.get("value_gap") is not None and latest.get("fairness"):
        lines.append(f"The fairness gap is {fmt(latest['value_gap'])}, so the deal reads "
                     f"{latest['fairness']} on value, with {latest.get('ahead_team')} ahead.")
    lines.append(f"There have been {fmt(counts.get('trades'))} trades this season and "
                 f"{fmt(counts.get('trades_this_week'))} this week.")
    adds = pack.get("recent_adds") or []
    if adds:
        first = adds[0]
        lines.append(f"Latest waiver move: {first.get('team_name')} added "
                     f"{', '.join(first.get('adds') or [])}.")
    injuries = pack.get("injuries") or []
    if injuries:
        first = injuries[0]
        lines.append(f"Injury headline: {first.get('player')} ({first.get('status')}) "
                     f"on {first.get('team_name')}.")
    ans.update({"verdict": f"Latest trade: {a.get('team_name')} and {b.get('team_name')}",
                "text": " ".join(lines), "receipts": receipts, "followups": followups})
    return ans


def _watchlist(pack):
    players = pack.get("players") or []
    top = pack.get("top_mover") or {}
    receipts = []
    if top.get("value") is not None:
        receipts.append({"label": f"{top.get('name')} value", "value": fmt(top["value"])})
    if top.get("delta7") is not None and len(receipts) < 4:
        receipts.append({"label": f"{top.get('name')} 7-day move", "value": fmt(top["delta7"])})
    if not receipts and pack.get("count") is not None:
        receipts.append({"label": "Players on your watchlist", "value": fmt(pack["count"])})
    ans = _base("watchlist", pack)
    followups = [_fu("Who should I add from waivers?"),
                 _fu("What should I do with my team this week?")]
    if not players:
        ans.update({"verdict": "Your watchlist is empty",
                    "text": "You have no players on your watchlist yet. Star a player "
                            "and he will show up here with his value moves and injury status.",
                    "receipts": receipts, "followups": followups})
        return ans
    lines = []
    if top:
        lines.append(f"The biggest move on your watchlist is {top.get('name')}, "
                     f"{fmt(top.get('delta7'))} over the last 7 days to {fmt(top.get('value'))}.")
    for p in players:
        line = f"{p.get('name')}: value {fmt(p.get('value'))}"
        if p.get("delta7") is not None:
            line += f", 7-day move {fmt(p['delta7'])}"
        if p.get("injury_status"):
            line += f", injury status {p['injury_status']}"
        if p.get("alert"):
            line += " (alert)"
        lines.append(line + ".")
    verdict = (f"{top.get('name')} is moving on your watchlist" if top
               else f"{fmt(pack.get('count'))} players on your watchlist")
    ans.update({"verdict": verdict, "text": " ".join(lines),
                "receipts": receipts, "followups": followups})
    return ans


def _front_office(pack):
    top = (pack.get("trade_targets") or [None])[0]
    receipts = []
    if pack.get("playoff_pct") is not None:
        receipts.append({"label": "Playoff odds (%)", "value": fmt(pack["playoff_pct"])})
    if top and top.get("get_total") is not None:
        receipts.append({"label": f"{top.get('partner')} trade: you receive (total value)",
                         "value": fmt(top["get_total"])})
    if top and top.get("give_total") is not None and len(receipts) < 4:
        receipts.append({"label": f"{top.get('partner')} trade: you give (total value)",
                         "value": fmt(top["give_total"])})
    ans = _base("front_office", pack)
    ans["act_link"] = _league_link("front_office", pack) or ans["act_link"]
    lines = []
    if pack.get("direction"):
        lines.append(f"Your front office profiles you as a {pack['direction']} team"
                     + (f" at {pack['record']}" if pack.get("record") else "")
                     + (f" with a {fmt(pack.get('playoff_pct'))}% chance to make the playoffs."
                        if pack.get("playoff_pct") is not None else "."))
    if pack.get("weakest_positions"):
        lines.append(f"Your weakest spot is {', '.join(pack['weakest_positions'])}.")
    for m in (pack.get("top_moves") or [])[:1]:
        if m.get("line"):
            lines.append(f"Top move: {m['line']}.")
    if top:
        gets = ", ".join(g.get("name") for g in top.get("gets") or [] if g.get("name"))
        gives = ", ".join(g.get("name") for g in top.get("gives") or [] if g.get("name"))
        lines.append(f"The top trade target is with {top.get('partner')}: you receive "
                     f"{gets} ({fmt(top.get('get_total'))}) for {gives} "
                     f"({fmt(top.get('give_total'))})"
                     + (f", graded {top['fairness_label']}." if top.get("fairness_label") else "."))
    wv = (pack.get("waiver_targets") or [None])[0]
    if wv:
        lines.append(f"Top waiver target: {wv.get('name')} at {fmt(wv.get('value'))}.")
    cuts = [c.get("name") for c in (pack.get("cut_candidates") or []) if c.get("name")]
    if cuts:
        lines.append(f"Cut candidates if you need a roster spot: {', '.join(cuts)}.")
    ans.update({"verdict": f"Front office: {pack.get('direction') or 'your report'}",
                "text": " ".join(lines), "receipts": receipts,
                "followups": [_fu("Find me a trade"), _fu("Who should I add from waivers?")]})
    return ans


_OPEN_STAT_WORDS = (("rec_yd", "receiving yards"), ("rec", "receptions"),
                    ("rec_tgt", "targets"), ("rec_td", "receiving TDs"),
                    ("rush_yd", "rushing yards"), ("rush_att", "carries"),
                    ("rush_td", "rushing TDs"), ("pass_yd", "passing yards"),
                    ("pass_cmp", "completions"), ("pass_att", "attempts"),
                    ("pass_td", "passing TDs"))


def _open_board_line(row, entry):
    meta = ", ".join(x for x in (row.get("team"), row.get("position")) if x)
    head = f"{row.get('rank')}. {row.get('name')}" + (f" ({meta})" if meta else "")
    line = f"{head}: {fmt(row.get('value'))}"
    if entry.get("stat_board"):
        line += f" {(entry.get('label') or '').lower()}"
        parts = []
        for key, word in _OPEN_STAT_WORDS:
            if key == entry.get("stat_key"):
                continue
            if row.get(key) is not None:
                parts.append(f"{fmt(row[key])} {word}")
            if len(parts) >= 3:
                break
        if parts:
            line += f" ({', '.join(parts)})"
    return line


def _open(pack, question):
    """Mirror of the open answering path over frozen tool evidence.

    The frozen pack carries tool entries (production shape); the mock
    derives the verdict, lines, and receipts from the evidence alone,
    the same discipline the live composer runs under: gated evidence
    gets the gated answer, empty evidence the honest decline, and every
    number comes from an evidence field.
    """
    evidence = [e for e in (pack.get("evidence") or []) if isinstance(e, dict)]
    usable = [e for e in evidence if e.get("ok", True) and not e.get("gated")]
    ask = (question or {}).get("question") or ""
    followups = [_fu("What should I do with my team this week?"),
                 _fu("Who should I start at FLEX?")]
    if not usable:
        gated = next((e for e in evidence if e.get("gated")), None)
        if gated is not None:
            label = gated.get("label") or "That metric"
            return {"verdict": None,
                    "text": (f"{label} is a PRO metric on BR Fantasy, so its numbers are not "
                             "included on the free tier. Free boards I can pull right now "
                             "include target share, air yards share, and snap share."),
                    "receipts": [], "caveat": None,
                    "followups": [_fu("Who leads in target share?"),
                                  _fu("Who leads in snap share?")],
                    "act_link": {"href": "/metrics", "label": "View on Advanced Metrics"},
                    "actions": [], "mentions": []}
        nearest = []
        for e in evidence:
            for n in e.get("nearest") or []:
                if n not in nearest:
                    nearest.append(n)
        text = "That is not available from your league data."
        if nearest:
            text += (f" The nearest tracked options are {', '.join(nearest[:3])}. "
                     "Ask about one of those and I can pull the live numbers.")
        return {"verdict": None, "text": text, "receipts": [], "caveat": None,
                "followups": followups, "act_link": None, "actions": [],
                "mentions": []}
    verdict = None
    lines: list = []
    receipts: list = []
    link = None
    mentions: list = []
    seen_mentions = set()
    for e in usable:
        for r in e.get("receipts") or []:
            if len(receipts) < 4:
                receipts.append({"label": str(r.get("label")),
                                 "value": str(r.get("value"))})
        if link is None and e.get("link"):
            link = e["link"]
        for p in e.get("players") or []:
            if p.get("player_id") and p.get("name") \
                    and str(p["player_id"]) not in seen_mentions:
                seen_mentions.add(str(p["player_id"]))
                mentions.append({"player_id": str(p["player_id"]),
                                 "name": str(p["name"])})
        tool = e.get("tool")
        if tool == "top_board":
            board = e.get("board") or []
            if board:
                top = board[0]
                if verdict is None:
                    verdict = f"{top.get('name')} leads {e.get('label')}"
                lines.append(f"{top.get('name')} leads {e.get('label')} at "
                             f"{fmt(top.get('value'))}.")
                if e.get("basis"):
                    lines.append(f"{e['basis']}.")
                lines.extend(_open_board_line(r, e) for r in board)
        elif tool == "metric_definition":
            if verdict is None:
                verdict = e.get("label")
            lines.append(f"{e.get('label')}: {e.get('definition')}")
            leader = e.get("leader") or {}
            if leader.get("name"):
                lines.append(f"{leader['name']} leads it at {fmt(leader.get('value'))}.")
            if e.get("gated_board"):
                lines.append(f"The {e.get('label')} board itself is PRO on BR "
                             "Fantasy, so its numbers are not shown on the free tier.")
        elif tool == "player_profile":
            p = e.get("player") or {}
            if verdict is None and p.get("name"):
                verdict = f"{p['name']} update"
            head = f"{p.get('name')} ({p.get('team')}, {p.get('position')})"
            if p.get("value") is not None:
                head += f" is valued at {fmt(p['value'])}"
            lines.append(head + ".")
            role = e.get("role") or {}
            if role.get("ppg") is not None:
                lines.append(f"He scores {fmt(role['ppg'])} points per game with a "
                             f"{fmt(role.get('target_share'))} target share.")
            if e.get("owner_team"):
                lines.append(f"He is rostered by {e['owner_team']}.")
        elif tool == "player_stats":
            p = e.get("player") or {}
            totals = e.get("totals") or {}
            if verdict is None and p.get("name"):
                verdict = f"{p['name']} update"
            lines.append(f"{p.get('name')} has {fmt(totals.get('fantasy_points'))} "
                         f"fantasy points in {totals.get('games')} games.")
            for key, word in _OPEN_STAT_WORDS:
                if totals.get(key) is not None:
                    lines.append(f"{word.capitalize()}: {fmt(totals[key])}.")
            for g in e.get("recent_games") or []:
                lines.append(f"Week {g.get('week')}: {fmt(g.get('points'))} points.")
            if e.get("best_week_points") is not None:
                lines.append(f"Best week: {fmt(e['best_week_points'])} points. "
                             f"Worst week: {fmt(e.get('worst_week_points'))} points.")
        elif tool == "player_metrics":
            p = e.get("player") or {}
            if verdict is None and p.get("name"):
                verdict = f"{p['name']} update"
            for r in (e.get("metric_ranks") or [])[:3]:
                lines.append(f"{r.get('label')}: {fmt(r.get('value'))} "
                             f"(rank {r.get('rank')}).")
            if e.get("gated_metrics"):
                lines.append("Some of his advanced metrics are PRO on BR "
                             "Fantasy and are not shown on the free tier.")
        elif tool == "team_roster":
            team = e.get("team") or {}
            injuries = e.get("injuries") or []
            if verdict is None:
                if injuries:
                    verdict = (f"{injuries[0]['name']} is "
                               f"{str(injuries[0].get('injury_status') or '').lower()}")
                elif team.get("team_name"):
                    verdict = f"{team['team_name']} rundown"
            lines.append(f"{team.get('team_name')} roster:")
            for p in (e.get("starters") or [])[:5]:
                lines.append(f"{p.get('name')} ({p.get('position')}): "
                             f"{fmt(p.get('value'))}.")
            st = e.get("standing") or {}
            if st:
                lines.append(f"Record {st.get('wins')}-{st.get('losses')} with "
                             f"{fmt(st.get('pf'))} points for.")
            for inj in injuries:
                lines.append(f"{inj['name']} is listed as {inj.get('injury_status')}.")
        elif tool == "league_lookup":
            leader = e.get("points_leader") or e.get("leader") or {}
            me = e.get("me") or {}
            if verdict is None:
                if leader.get("team_name"):
                    verdict = f"{leader['team_name']} on top"
                elif me.get("team_name"):
                    verdict = f"{me['team_name']} check-in"
            if leader.get("team_name"):
                lines.append(f"{leader['team_name']} leads with "
                             f"{fmt(leader.get('pf'))} points for.")
            if me:
                lines.append(f"Your team is {me.get('wins')}-{me.get('losses')} with "
                             f"{fmt(me.get('pf'))} points for, seed {me.get('seed')}.")
            for row in (e.get("table") or [])[:3]:
                lines.append(f"{row.get('team_name')}: {row.get('wins')}-"
                             f"{row.get('losses')}, {fmt(row.get('pf'))} points for.")
        elif tool == "compare_players":
            detail = e.get("players_detail") or []
            if len(detail) == 2:
                a, b = detail
                hi = a if (a.get("value") or 0) >= (b.get("value") or 0) else b
                lo = b if hi is a else a
                if verdict is None:
                    verdict = f"{hi['name']} over {lo['name']}"
                for p in detail:
                    lines.append(f"{p['name']}: value {fmt(p.get('value'))}, "
                                 f"{fmt(p.get('ppg'))} points per game, "
                                 f"{fmt(p.get('target_share'))} target share.")
        elif tool == "value_lookup":
            if e.get("player"):
                p = e["player"]
                move = e.get("move_7d") or {}
                if verdict is None:
                    verdict = f"{p['name']} value check"
                lines.append(f"{p['name']} is valued at {fmt(p.get('value'))}.")
                if move.get("delta") is not None:
                    lines.append(f"His value moved {fmt(move['delta'])} over "
                                 "the last 7 days.")
            else:
                risers = e.get("risers") or []
                fallers = e.get("fallers") or []
                if verdict is None:
                    if "buy" in ask.lower() and fallers:
                        verdict = f"{fallers[0]['name']} is the buy-low"
                    elif risers:
                        verdict = f"{risers[0]['name']} is rising"
                for r in risers[:2]:
                    lines.append(f"{r['name']} is up {fmt(r.get('delta'))} to "
                                 f"{fmt(r.get('value'))} this week.")
                for r in fallers[:2]:
                    lines.append(f"{r['name']} is down {fmt(abs(r.get('delta') or 0))} "
                                 f"to {fmt(r.get('value'))} this week.")
        elif tool == "next_up":
            sub = e.get("subject") or {}
            nxt = e.get("next_up") or {}
            if verdict is None and nxt.get("name"):
                verdict = f"{nxt['name']} is next up"
            lines.append(f"If {sub.get('name')} misses time, {nxt.get('name')} "
                         f"is next up for the {nxt.get('team')}.")
            if nxt.get("value") is not None:
                lines.append(f"{nxt['name']} is valued at {fmt(nxt['value'])} and "
                             f"sits on {nxt.get('location') or 'waivers'}.")
            if sub.get("value") is not None and len(receipts) < 4:
                receipts.append({"label": f"{sub.get('name')} value",
                                 "value": fmt(sub["value"])})
    return {"verdict": verdict, "text": " ".join(lines), "receipts": receipts,
            "caveat": None, "followups": followups, "act_link": link,
            "actions": [], "mentions": mentions[:8]}


def _live(pack):
    mu = pack.get("matchup") or {}
    a, b = mu.get("team_a") or {}, mu.get("team_b") or {}
    receipts = []
    if a.get("current_points") is not None:
        receipts.append({"label": f"{a.get('team_name')} points so far",
                         "value": fmt(a["current_points"])})
    if b.get("current_points") is not None:
        receipts.append({"label": f"{b.get('team_name')} points so far",
                         "value": fmt(b["current_points"])})
    if a.get("win_prob_pct") is not None and len(receipts) < 4:
        receipts.append({"label": f"{a.get('team_name')} win probability (%)",
                         "value": fmt(a["win_prob_pct"])})
    if a.get("projected") is not None and len(receipts) < 4:
        receipts.append({"label": f"{a.get('team_name')} projected final",
                         "value": fmt(a["projected"])})
    state = pack.get("state")
    mine = bool(mu.get("is_mine"))
    lead = (a.get("current_points") or 0) >= (b.get("current_points") or 0)
    if state == "pregame":
        verdict = "No games have started yet"
        text = (f"No games have kicked off for Week {pack.get('week')} yet. "
                f"The projection has {a.get('team_name')} at {fmt(a.get('projected'))} "
                f"and {b.get('team_name')} at {fmt(b.get('projected'))}.")
    else:
        if mine:
            verdict = (f"You {'lead' if lead else 'trail'} "
                       f"{fmt(a.get('current_points'))} to {fmt(b.get('current_points'))}")
        else:
            verdict = (f"{a.get('team_name')} {'leads' if lead else 'trails'} "
                       f"{fmt(a.get('current_points'))} to {fmt(b.get('current_points'))}")
        text = (f"{a.get('team_name')} has {fmt(a.get('current_points'))} points so far "
                f"against {fmt(b.get('current_points'))} for {b.get('team_name')}, with a "
                f"live projection of {fmt(a.get('projected'))} to {fmt(b.get('projected'))} "
                f"and a win probability of {fmt(a.get('win_prob_pct'))} percent.")
        need = pack.get("need") or {}
        rem = need.get("my_remaining") or []
        if mine and rem:
            names = ", ".join(f"{r.get('name')} ({fmt(r.get('projection'))} projected)"
                              for r in rem)
            text += f" You still have {names} to play."
        if mine and (need.get("deficit") or 0) > 0:
            text += (f" You need at least {fmt(need.get('deficit'))} more points "
                     f"than {b.get('team_name')} the rest of the way.")
    ans = _base("live", pack)
    ans.update({"verdict": verdict, "text": text, "receipts": receipts[:4],
                "followups": [_fu("What do I need tonight?"), _fu("Break down my matchup")]})
    return ans


def _draft(pack):
    receipts = []
    cur = pack.get("current_pick") or {}
    if cur.get("pick_no") is not None:
        receipts.append({"label": "Current pick", "value": fmt(cur["pick_no"])})
    nxt = pack.get("my_next_pick") or {}
    if nxt.get("pick_no") is not None and len(receipts) < 4:
        receipts.append({"label": "Your next pick", "value": fmt(nxt["pick_no"])})
    best = (pack.get("best_available") or [None])[0]
    if best and best.get("value") is not None and len(receipts) < 4:
        receipts.append({"label": f"{best.get('name')} value", "value": fmt(best["value"])})
    prep = pack.get("prep") or {}
    for cand in ([prep.get("target")] + list(prep.get("next_up") or [])
                 + list(prep.get("fallers") or [])):
        if len(receipts) >= 4:
            break
        if cand and cand.get("score") is not None:
            receipts.append({"label": f"{cand.get('name')} prospect score",
                             "value": fmt(cand["score"])})
    if not receipts and pack.get("picks_made") is not None:
        receipts.append({"label": "Picks made", "value": fmt(pack["picks_made"])})
    status = pack.get("status")
    if prep:
        target = prep.get("target") or {}
        if target.get("name"):
            verdict = f"Target {target.get('name')} at {prep.get('slot')}"
            text = (f"At the {prep.get('slot')} (pick {fmt(prep.get('overall_pick'))} of the "
                    f"{fmt(prep.get('class_year'))} class), the target is "
                    f"{target.get('name')} ({target.get('position')}, "
                    f"{target.get('school')}, prospect score "
                    f"{fmt(target.get('score'))}).")
            nxt = prep.get("next_up") or []
            if nxt:
                cluster = ", ".join(
                    f"{c.get('name')} ({fmt(c.get('score'))})" for c in nxt)
                text += f" Next in the cluster: {cluster}."
            fallers = prep.get("fallers") or []
            if fallers:
                names = " or ".join(str(c.get("name")) for c in fallers)
                if len(fallers) > 2:
                    names = "any of " + names
                text += (f" If {names} somehow falls to the "
                         f"{prep.get('slot')}, take him instead.")
            if prep.get("assumption"):
                text += f" {prep['assumption']}"
        else:
            verdict = "Draft prep"
            text = (f"The rookie board tracks {fmt(prep.get('board_size'))} names, "
                    f"so nobody is slotted at pick {fmt(prep.get('overall_pick'))} "
                    f"of the {fmt(prep.get('class_year'))} class.")
    elif status == "drafting" and best:
        verdict = str(best.get("name"))
        text = (f"{cur.get('team_name')} is on the clock at pick {fmt(cur.get('pick_no'))} "
                f"of round {fmt(cur.get('round'))}. The best player still available is "
                f"{best.get('name')} ({best.get('position')}, {best.get('team')}), "
                f"valued at {fmt(best.get('value'))}.")
        if nxt:
            text += f" Your next pick is {fmt(nxt.get('pick_no'))}."
    elif status == "pre_draft" and best:
        verdict = str(best.get("name"))
        text = (f"The draft has not started. On current values the best player "
                f"available is {best.get('name')} ({best.get('position')}), "
                f"valued at {fmt(best.get('value'))}.")
    else:
        verdict = "Draft complete"
        mine = pack.get("my_picks") or []
        names = ", ".join(str(p.get("name")) for p in mine)
        text = (f"The draft is complete with {fmt(pack.get('picks_made'))} picks made.")
        if names:
            text += f" Your picks: {names}."
    ans = _base("draft", pack)
    ans.update({"verdict": verdict, "text": text, "receipts": receipts[:4],
                "followups": [_fu("Who is on the clock?"),
                              _fu("What does my roster still need?")]})
    return ans


def _nfl_game_line(g):
    line = (f"{g.get('away_name')} at {g.get('home_name')} "
            f"({g.get('kickoff_label')}")
    if g.get("network"):
        line += f", {g['network']}"
    line += ")"
    if g.get("status") == "live":
        line += " (in progress)"
    elif g.get("status") == "final" and g.get("away_score") is not None \
            and g.get("home_score") is not None:
        line += (f", final: {g.get('away_name')} {g.get('away_score')}, "
                 f"{g.get('home_name')} {g.get('home_score')}")
    return line


def _nfl_kickoff_receipt(g):
    return {"label": f"{g.get('away_name')} at {g.get('home_name')} kickoff",
            "value": f"{g.get('date_label')}, {g.get('kickoff_label')}"}


def _nfl_bucket(g):
    import re as _re
    m = _re.search(r"(\d{1,2}):(\d{2})\s*(AM|PM)",
                   str(g.get("kickoff_label") or ""))
    if not m:
        return "Early"
    mins = (int(m.group(1)) % 12) * 60 + int(m.group(2)) \
        + (720 if m.group(3) == "PM" else 0)
    if mins < 720:
        return "Morning"
    if mins < 960:
        return "Early"
    if mins < 1200:
        return "Late"
    return "Primetime"


def _nfl_schedule(pack):
    # Mirror of compose._compose_nfl_schedule: the pack's games, days,
    # kickoff times and networks render verbatim; nothing else may
    # introduce a number or a team.
    games = [g for g in (pack.get("games") or []) if isinstance(g, dict)]
    scope = pack.get("scope")
    nxt = pack.get("next_game") if isinstance(pack.get("next_game"), dict) \
        else None
    week = pack.get("week")

    def _single_receipts(g):
        rows = [_nfl_kickoff_receipt(g)]
        if g.get("network"):
            rows.append({"label": "Network", "value": g["network"]})
        else:
            rows.append({"label": "Game day", "value": g.get("date_long")})
        return rows

    def _next_sentence():
        if not nxt:
            return ""
        net = f" on {nxt['network']}" if nxt.get("network") else ""
        return (f" The next game is {nxt.get('away_name')} at "
                f"{nxt.get('home_name')} on {nxt.get('date_long')} at "
                f"{nxt.get('kickoff_label')}{net}.")

    ans = _base("nfl_schedule", pack)
    if scope == "team":
        team = pack.get("team") or {}
        name = team.get("name") or team.get("abbr") or "That team"
        followups = [_fu("Who plays tonight?"),
                     _fu("What games are on Sunday?")]
        if games:
            g = games[0]
            home = g.get("home") == team.get("abbr")
            opp = g.get("home_name") if not home else g.get("away_name")
            verb = "host" if home else "visit"
            net = f" on {g['network']}" if g.get("network") else ""
            verdict = f"{name} play {g.get('date_label')}"
            text = (f"The {name} {verb} the {opp} on "
                    f"{g.get('date_long')} at {g.get('kickoff_label')}"
                    f"{net}.")
            receipts = _single_receipts(g)
        elif pack.get("bye"):
            verdict = f"{name} are on a bye in Week {week}"
            text = (f"The {name} are on a bye in Week {week}."
                    + _next_sentence())
            if nxt:
                receipts = [
                    {"label": "Next game kickoff",
                     "value": f"{nxt.get('date_label')}, "
                              f"{nxt.get('kickoff_label')}"},
                    {"label": "Next game",
                     "value": f"{nxt.get('away_name')} at "
                              f"{nxt.get('home_name')}"}]
            else:
                receipts = [{"label": "Bye week", "value": f"Week {week}"}]
        else:
            verdict = f"No {name} game on the loaded schedule"
            text = (f"The {name} have no game in the schedule loaded "
                    f"for this week or next.")
            receipts = []
        ans.update({"verdict": verdict, "text": text, "receipts": receipts,
                    "followups": followups})
        return ans
    if scope == "week":
        verdict = f"Week {week} NFL schedule"
        by_day = {}
        for g in games:
            by_day.setdefault(g.get("date_label") or "", []).append(g)
        lines = [f"Week {week} NFL schedule:"]
        for day_label, rows in by_day.items():
            lines.append(f"{day_label}: "
                         + "; ".join(_nfl_game_line(g) for g in rows))
        receipts = [_nfl_kickoff_receipt(g) for g in games[:4]]
        ans.update({"verdict": verdict, "text": "\n".join(lines),
                    "receipts": receipts,
                    "followups": [_fu("Who plays tonight?"),
                                  _fu("What do I need tonight to win?")]})
        return ans
    day_name = pack.get("day_name") or ""
    rel = {"tonight": "tonight", "today": "today",
           "tomorrow": "tomorrow"}.get(scope)
    followups = [_fu("Who plays this week?"),
                 _fu("What do I need tonight to win?")]
    if not games:
        if rel:
            verdict = f"No NFL game {rel}"
            text = f"There is no NFL game {rel}."
        else:
            verdict = f"No NFL games on {day_name}"
            text = (f"There are no NFL games on {day_name} in the "
                    f"schedule loaded for this week.")
        text += _next_sentence()
        receipts = [{"label": f"Games {rel or 'on ' + day_name}",
                     "value": "0"}]
        if nxt:
            receipts.append(_nfl_kickoff_receipt(nxt))
        ans.update({"verdict": verdict, "text": text, "receipts": receipts,
                    "followups": followups})
        return ans
    if len(games) == 1:
        g = games[0]
        lead = {"tonight": "Tonight's game", "today": "Today's game",
                "tomorrow": "Tomorrow's game"}.get(scope) \
            or f"{day_name}'s game"
        verdict = (f"{g.get('away_name')} at {g.get('home_name')} "
                   f"{rel}" if rel else
                   f"{g.get('away_name')} at {g.get('home_name')} "
                   f"on {day_name}")
        text = f"{lead}: {_nfl_game_line(g)}."
        ans.update({"verdict": verdict, "text": text,
                    "receipts": _single_receipts(g),
                    "followups": followups})
        return ans
    verdict = f"{len(games)} games on {day_name}"
    lines = [f"{day_name}'s NFL games:"]
    for bucket in ("Morning", "Early", "Late", "Primetime"):
        rows = [g for g in games if _nfl_bucket(g) == bucket]
        if rows:
            lines.append(f"{bucket}: "
                         + "; ".join(_nfl_game_line(g) for g in rows))
    ans.update({"verdict": verdict, "text": "\n".join(lines),
                "receipts": [_nfl_kickoff_receipt(g) for g in games[:4]],
                "followups": followups})
    return ans


_BUILDERS = {"start_sit": _start_sit, "trade_eval": _trade_eval,
             "buy_sell_hold": _buy_sell_hold, "player": _player, "metric": _metric,
             "leaderboard": _leaderboard, "trade_finding": _trade_finding,
             "replacement": _replacement, "handcuff": _handcuff,
             "breakout_board": _breakout_board,
             "breakout_track_record": _breakout_track_record,
             "breakout_history": _breakout_history,
             "teams": _teams, "pick_ownership": _pick_ownership,
             "waivers": _waivers, "this_week": _this_week,
             "playoffs": _playoffs, "power_rankings": _power_rankings,
             "standings": _standings, "matchup": _matchup, "schedule": _schedule,
             "keeper": _keeper, "rookie": _rookie, "cross_league": _cross_league,
             "activity": _activity, "watchlist": _watchlist, "front_office": _front_office,
             "live": _live, "draft": _draft, "nfl_schedule": _nfl_schedule}


def compose_mock(question: dict) -> dict:
    """Build the mock answer for one frozen question entry."""
    pack = question.get("evidence_pack") or {}
    if isinstance(pack, dict) and pack.get("gated"):
        return gated_answer(pack)
    if isinstance(pack, dict) and pack.get("available") is False \
            and pack.get("type") == "leaderboard" \
            and pack.get("board_kind") in ("buy_low", "sell_high"):
        return _empty_value_board_answer(pack)
    if isinstance(pack, dict) and pack.get("available") is False:
        return no_data_answer(pack)
    if question["type"] == "open":
        return _open(pack, question)
    builder = _BUILDERS[question["type"]]
    return builder(pack)
