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
    "trade_finding": {"href": "/trade?tab=suggestions", "label": "Open Trade Suggestions"},
    "breakout_board": {"href": "/breakouts", "label": "View breakout board"},
    "breakout_track_record": {"href": "/breakouts", "label": "View breakout board"},
    "breakout_history": {"href": "/breakouts", "label": "View breakout board"},
    "keeper": {"href": "/keeper", "label": "View Keeper Optimizer"},
    "rookie": {"href": "/prospects", "label": "View Rookie Rankings"},
    "cross_league": {"href": "/portfolio", "label": "View Portfolio"},
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
        for p in pack.get("cleared_drops") or []:
            _add(p)
    elif intent == "keeper":
        for p in pack.get("my_keepers") or []:
            _add(p)
        _add(pack.get("asked_player"))
        for p in pack.get("roster_core") or []:
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
}


def _league_link(intent, pack):
    spec = _LEAGUE_LINKS.get(intent or "")
    league = (pack or {}).get("league") or {}
    if not spec or not league.get("platform") or not league.get("season") or not league.get("league_id"):
        return None
    page, label = spec
    return {"href": f"/{league['platform']}/{league['season']}/{league['league_id']}/{page}",
            "label": label}


def no_data_answer(pack: dict) -> dict:
    """Mirror of compose.no_data_answer (same strings, same shape)."""
    missing = pack.get("missing") or "that data is not tracked"
    nearest = pack.get("nearest") or []
    text = f"That is not available: {missing}."
    if nearest:
        text += f" The nearest tracked options are {', '.join(str(n) for n in nearest[:3])}. Ask about one of those and I can pull the live numbers."
    # The link names the surface the question was about: league-context
    # intents use their league link, everything else resolves normally
    # (a keeper no-data lands on the Keeper Optimizer, not Advanced Metrics).
    link = (_league_link((pack or {}).get("type"), pack)
            or _resolve_act_link(str((pack or {}).get("type") or "metric"), pack)
            or dict(_ACT_LINKS["metric"]))
    return {"verdict": None, "text": text, "receipts": [], "caveat": None,
            "followups": [f"What is {n}?" for n in nearest[:2]],
            "act_link": link, "actions": [], "mentions": []}


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
        "followups": ["What should I do with my team this week?",
                      "Should I accept a trade I was offered?"],
        "act_link": {"href": pricing, "label": "See PRO plans"},
        "actions": [],
    }

def gated_answer(pack: dict) -> dict:
    """Mirror of compose.gated_answer (same strings, same shape)."""
    if (pack or {}).get("type") == "trade_finding":
        return _gated_trade_finding_answer(pack)
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


def _playoffs(pack):
    me = pack.get("me") or {}
    scen = pack.get("scenario") or {}
    receipts = []
    if me.get("playoff_pct") is not None:
        receipts.append({"label": "Your playoff odds (%)", "value": fmt(me["playoff_pct"])})
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
            "followups": ["What are my odds of a first-round bye?", "Who is on the bubble?"],
        })
        return ans
    if scen:
        text = (f"The sim has {me.get('team_name')} at {fmt(me.get('playoff_pct'))} percent. "
                f"{scen.get('summary')}.")
        if scen.get("next_opponent"):
            text += f" Your next game is against {scen['next_opponent']}."
        ans.update({"verdict": scen.get("summary") or "Alive", "text": text,
                    "receipts": receipts[:4],
                    "followups": ["What are my odds of a first-round bye?",
                                  "How hard is my remaining schedule?"]})
        return ans
    ans.update({
        "verdict": f"{fmt(me.get('playoff_pct'))}% playoff odds",
        "text": (f"{me.get('team_name')} sits at {me.get('record')} with "
                 f"{fmt(pack.get('weeks_remaining'))} weeks left. Your projected finish is "
                 f"{fmt(me.get('avg_final_wins'))} wins against a projected cut line of "
                 f"{fmt(pack.get('projected_cut_wins'))} wins, so the math says about "
                 f"{fmt(me.get('wins_still_needed'))} more wins gets you in."),
        "receipts": receipts[:4],
        "followups": ["What are my odds of a first-round bye?", "Who is on the bubble?"],
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
                "followups": ["What do the standings look like?", "What are my playoff odds?"]})
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
                "followups": ["Where am I in the power rankings?", "What are my playoff odds?"]})
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
                "followups": ["Who should I start at FLEX?", "What are my playoff odds?"]})
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
                "followups": ["What are my playoff odds?", "Break down my matchup"]})
    return ans


def _side_receipts(f):
    return [{"label": "You receive (total value)", "value": fmt(f.get("receive_total"))},
            {"label": "You give (total value)", "value": fmt(f.get("give_total"))},
            {"label": "Fairness gap", "value": fmt(f.get("fairness_gap"))}]


def _names(players):
    return " and ".join(p["name"] for p in players or [])


def _trade_finding(pack):
    if pack.get("gated"):
        return gated_answer(pack)
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
            "followups": ["Who else could I target?", "What can I get for my spare players?"],
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
            "followups": ["Who is available on waivers?", "What should I do with my team this week?"],
        })
        return ans
    ans.update({
        "verdict": "No clean fit right now",
        "text": ("No clean fit right now: no roster in your league lines up on both sides "
                 "inside the fair band. The nearest alternative is the waiver wire."),
        "receipts": [],
        "followups": ["Who is available on waivers?", "What should I do with my team this week?"],
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
                "followups": ["How good is BR at breakout calls?"]})
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
                "followups": ["Who are this week's breakout calls?"]})
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
                    "followups": ["Who are this week's breakout calls?"]})
    else:
        ans.update({"verdict": f"NO, {name} was not a breakout call",
                    "text": f"No. {name} has not been surfaced as a breakout call this season.",
                    "receipts": [], "followups": ["Who are this week's breakout calls?"]})
    return ans


def _teams(pack):
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
                    "followups": ["Whose roster is strongest?"]})
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
    ans.update({"verdict": f"Strongest roster: {s.get('team_name')}", "text": text,
                "receipts": receipts[:4], "followups": ["Who is weakest at TE?"]})
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
                    "receipts": receipts, "followups": ["What other picks do I own?"]})
    else:
        ans.update({"verdict": f"{owner} owns the {pick.get('label')}",
                    "text": f"Your {pick.get('label')} now belongs to {owner}.",
                    "receipts": receipts, "followups": ["What other picks do I own?"]})
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
                    "followups": ["Who is first on my cut line?"]})
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
                "followups": ["Who is first on my cut line?"]})
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
                    "followups": ["Who are the top quarterbacks in the class?"]})
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
                "followups": ["Who are the top quarterbacks in the class?"]})
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
                "followups": ["Which of my teams is strongest?"]})
    return ans


_BUILDERS = {"start_sit": _start_sit, "trade_eval": _trade_eval,
             "buy_sell_hold": _buy_sell_hold, "player": _player, "metric": _metric,
             "leaderboard": _leaderboard, "trade_finding": _trade_finding,
             "breakout_board": _breakout_board,
             "breakout_track_record": _breakout_track_record,
             "breakout_history": _breakout_history,
             "teams": _teams, "pick_ownership": _pick_ownership,
             "waivers": _waivers, "this_week": _this_week,
             "playoffs": _playoffs, "power_rankings": _power_rankings,
             "standings": _standings, "matchup": _matchup, "schedule": _schedule,
             "keeper": _keeper, "rookie": _rookie, "cross_league": _cross_league}


def compose_mock(question: dict) -> dict:
    """Build the mock answer for one frozen question entry."""
    pack = question.get("evidence_pack") or {}
    if isinstance(pack, dict) and pack.get("gated"):
        return gated_answer(pack)
    if isinstance(pack, dict) and pack.get("available") is False:
        return no_data_answer(pack)
    builder = _BUILDERS[question["type"]]
    return builder(pack)
