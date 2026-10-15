"""Answer composition: pack in, answer out. This is the eval seam.

Pure with respect to tools: it only calls the model (via model.call_model_json,
monkeypatched in tests) and post-checks quality in code.

Grounding is enforced on EVERY surface, not just receipts: numbers in the
verdict, prose, caveat, and follow-ups must all appear in the evidence pack
(or the question itself). A draft that states an unverifiable number is
regenerated once with a stricter instruction; if the retry still fails, the
offending sentences and numbers are stripped rather than shipped. Receipts
whose numbers do not all appear in the pack JSON are dropped.

Citation is a contract, not a hope (live eval 2026-10-14: m1/tw1/tw2 failed
because the model never cited the pack's key numbers). Each intent has
deterministic required receipt rows derived from the pack (side totals and
the fairness gap for trades, the leader and the user's own players for
metrics, each action's key number for the digest, the top candidate and the
user's weakest starter for waivers). If a required number is missing from
every answer surface, the draft is regenerated once with the missing numbers
named; if it is still missing, the deterministic row is injected.

Trade verdicts are pinned in code to the trade analyzer's shared fairness
band (utils.trade_value.fair_value_band, the definition api_trade_eval and
the outcome analyzer share), so the assistant can never contradict the
calculator: the model writes prose around the code-set verdict and every
player in the deal is guaranteed to be named.

Internal language never reaches the user: the words "pack" / "evidence pack"
and follow-ups phrased as instructions to send or provide data are treated
as draft defects (one regeneration, then deterministic cleanup: offending
sentences stripped, follow-ups replaced with user-voiced templates).

Receipt values are humanized: a value the model pasted as a serialized
object renders as its friendly label (a league renders as its team name,
never raw JSON), and rows with empty values ([], {}, None, blank) are
omitted entirely. No data, no row.

The answer dict is the UI contract (static/app.js initAssistant): act_link is
an object {href, label} (or None) assigned deterministically in code, never
by the model; this-week actions carry href, cta_label, and the pack's
key_number so the UI can render them as Next Steps rows.

Act links stage state in the site's own tools where a real mechanism exists:
trade answers open the calculator with the pack's deal prefilled (its native
?a=&b= id params), start/sit stages the pair in Compare (?p1=&p2=), metric
answers preset Advanced Metrics (?metric=&pos=), and player answers open the
player modal (?player=). The answer also carries `mentions` (players the
pack resolved, with ids) so the UI can open the modal from a name in the
text instead of rendering it as plain text.
"""
from __future__ import annotations

import json
import logging
import re
from urllib.parse import quote, quote_plus

from dashboard_services.assistant import PROMPT_VERSION
from dashboard_services.assistant.model import call_model_json

logger = logging.getLogger(__name__)

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": ["string", "null"]},
        "text": {"type": "string"},
        "receipts": {"type": "array", "items": {"type": "object", "properties": {
            "label": {"type": "string"}, "value": {"type": "string"}}, "required": ["label", "value"], "additionalProperties": False}},
        "caveat": {"type": ["string", "null"]},
        "followups": {"type": "array", "items": {"type": "string"}},
        "act_link": {"type": ["string", "null"]},
    },
    "required": ["verdict", "text", "receipts", "caveat", "followups", "act_link"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are the BR Fantasy assistant. Answer from the evidence pack ONLY. "
    "Verdict first, in the verdict field. Every number you state must appear verbatim in the pack; "
    "never use a number from memory. Receipts: 2 to 4 entries, each a label and the pack value as a string. "
    "Cite the key numbers: for a trade, both side totals and the fairness gap; for a metric, the leader "
    "and any of the user's own players on the board; for the weekly digest, each action's key number; "
    "for waivers, the top candidate's value and the user's weakest starter at that position; "
    "for playoffs, the odds and the wins still needed; for standings and power rankings, the user's "
    "own seed or rank and the leader; for a matchup, the win probability and both projected totals; "
    "for schedule, the user's opponent strength index and the easiest team's. "
    "If the pack flags odds_soft, say the odds are soft this early and lead with the wins math. "
    "If fixed_verdict is present in the payload, the verdict field must be exactly that value; "
    "it is set by the site's trade fairness band and is not yours to change. "
    "Caveat only when the pack marks margin.thin true or data stale, otherwise null. "
    "If assumed_context.assumed_league_uncertain is true, state the assumed league in the text. "
    "If a stored preference shapes the answer, name that preference in the text. "
    "Never mention the pack, evidence, tools, or any internal data structure; call it your league data. "
    "Follow-ups must be short questions in the user's voice that they would tap, never instructions "
    "to send, provide, or share anything. "
    "No em dashes. Use plain hyphens. Short, direct sentences. Return JSON only."
)

# Deterministic act links per intent (UI contract: {href, label} or None).
# These are the fallbacks; _resolve_act_link upgrades most intents to a
# deep link that stages state in the target tool (see the builders below).
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


def _league_base(pack: dict) -> str | None:
    """League page prefix (/{platform}/{season}/{league_id}) from the pack."""
    league = (pack or {}).get("league") or {}
    platform, season, league_id = league.get("platform"), league.get("season"), league.get("league_id")
    if platform and season and league_id:
        return f"/{platform}/{season}/{league_id}"
    return None


def _side_ids(players) -> list[str]:
    return [str(p["player_id"]) for p in (players or [])
            if isinstance(p, dict) and p.get("player_id")]


def _trade_calc_link(pack: dict) -> dict | None:
    """Open the trade calculator with this pack's deal staged.

    The calculator loads a deal from id params (loadTradeFromURL in
    static/app.js, the same params its own share flow mints): a/b are
    comma-separated player ids, ap/bp pick ids, and a side's list is what
    that side RECEIVES (side A is the viewer's side). So the pack's receive
    side goes in a, its give side in b. The league URL form opens the
    calculator in the league's own context (its scoring, format, and value
    mode), the same form the Front Office Build around links use. Built in
    code from pack ids only, never from model output. Ids the calculator
    cannot resolve are dropped by its loader, so a stale id degrades to a
    partial deal, never an error.
    """
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
    # Picks: the calculator takes canonical pick ids (2027_1_03) per side in
    # ap/bp, but today's packs carry picks as side-less free text, so there
    # is nothing honest to stage yet. Side-attributed canonical picks land
    # here if the pack ever carries them.
    for key, param in (("receive_picks", "ap"), ("give_picks", "bp")):
        ids = [str(p.get("pick_id")) for p in (pack.get(key) or [])
               if isinstance(p, dict) and p.get("pick_id")]
        if ids:
            params.append(param + "=" + ",".join(quote(i, safe="") for i in ids))
    return {"href": path + "?" + "&".join(params), "label": "Open in Trade Calculator"}


def _player_modal_link(player) -> dict | None:
    """Open the player modal directly: app.js auto-opens it on any page when
    the URL carries ?player=<id>&player_name=<name>."""
    if not isinstance(player, dict) or not player.get("player_id"):
        return None
    href = "/?player=" + quote(str(player["player_id"]), safe="")
    if player.get("name"):
        href += "&player_name=" + quote_plus(str(player["name"]))
    return {"href": href, "label": "View player"}


def _metric_link(pack: dict) -> dict | None:
    """Preset Advanced Metrics: its page reads ?metric= and ?pos= on load."""
    key = (pack or {}).get("metric_key")
    if not key:
        return None
    base = _league_base(pack)
    href = (f"{base}/metrics" if base else "/metrics") + "?metric=" + quote(str(key), safe="")
    positions = [str(p) for p in (pack.get("positions") or []) if p]
    if len(positions) == 1:
        href += "&pos=" + quote(positions[0], safe="")
    return {"href": href, "label": "Open Advanced Metrics"}


def _board_link(pack: dict) -> dict | None:
    """Preset Advanced Metrics for a leaderboard pack, in the same shape the
    metric answers use: /{league}/metrics?metric=<key>, plus &pos= when the
    board is scoped to exactly one position (the pack's positions list, or
    its single position_filter). The AM page reads ?metric= and ?pos= on
    load. Falls back to the static entry when the pack names no metric."""
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


def _start_sit_link(pack: dict) -> dict | None:
    """Stage the decision pair in Compare, which prefills from ?p1=&p2=."""
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


def _resolve_act_link(intent: str, pack: dict) -> dict | None:
    """The answer's act link: a state-staging deep link when the pack has
    what the target tool needs, else the static per-intent fallback."""
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
    league_link = _league_link(intent, pack)
    if league_link:
        return league_link
    static = _ACT_LINKS.get(intent)
    return dict(static) if static else None


def _mentions(intent: str, pack: dict) -> list[dict]:
    """Players the answer names, with ids, so the UI can open the player
    modal from a name instead of rendering it as plain text. Deterministic
    from the pack; only players the pack actually resolved."""
    pack = pack or {}
    found: list[dict] = []
    seen: set[str] = set()

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

# League-context intents link at the league surface for the subject: the
# playoff odds tab lives on the league dashboard, power rankings on the
# standings page, matchups on the weekly hub, schedule strength on the
# Schedule Assistant. The href is built from the pack's league fields.
_LEAGUE_LINKS = {
    "playoffs": ("dashboard", "View Playoff Odds"),
    "power_rankings": ("standings", "View Power Rankings"),
    "standings": ("standings", "View standings"),
    "matchup": ("weekly", "View Matchups"),
    "schedule": ("schedule", "Open in Schedule Assistant"),
}


def _league_link(intent: str, pack: dict) -> dict | None:
    spec = _LEAGUE_LINKS.get(intent or "")
    league = (pack or {}).get("league") or {}
    if not spec or not league.get("platform") or not league.get("season") or not league.get("league_id"):
        return None
    page, label = spec
    return {"href": f"/{league['platform']}/{league['season']}/{league['league_id']}/{page}",
            "label": label}


# CTA labels for this-week action rows, keyed by the pack action's tag; the
# digest vocabulary (lineup / waiver / trade) matches the site's own rows.
_CTA_LABELS = {"lineup": "Set lineup", "waiver": "View waivers", "waivers": "View waivers",
               "trade": "Build offer"}


def _cta_label(action: dict) -> str:
    if action.get("cta_label"):
        return str(action["cta_label"])
    tag = str(action.get("tag") or "").lower()
    for key, label in _CTA_LABELS.items():
        if key in tag:
            return label
    return "View"


# Standalone numeric token (same discipline as the eval scorer): digits glued
# to letters or codes (WR1, W7, 2026season) are labels, not numeric claims.
_NUM_RE = re.compile(r"(?<![A-Za-z0-9])\$?-?\d[\d,]*(?:\.\d+)?%?(?![A-Za-z0-9])")


def _numbers_in(text: str) -> list[str]:
    return _NUM_RE.findall(text or "")


def _norm_num(token: str) -> str:
    return str(token).replace("$", "").replace(",", "").rstrip("%").strip()


def _canon(token: str) -> str:
    """Scorer-style canonical number: sign dropped, 17.90 -> 17.9, 2380.0 -> 2380."""
    s = _norm_num(token)
    try:
        f = abs(float(s))
    except (TypeError, ValueError):
        return s
    return str(int(f)) if f == int(f) else str(f)


def _canon_numbers(text) -> set[str]:
    return {_canon(n) for n in _numbers_in(str(text or ""))}


def _fmt_num(value) -> str:
    """Render a pack number the way prose would: 4850.0 -> '4850', 0.24 -> '0.24'."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(f)) if f == int(f) else str(f)


def _pack_numbers(pack) -> set[str]:
    raw = json.dumps(pack or {}, default=str)
    found = set(_numbers_in(raw))
    # Also accept integer forms of float pack values (12.0 in pack, "12" in receipt).
    extra = set()
    for n in found:
        try:
            f = float(n.rstrip("%"))
            if f == int(f):
                extra.add(str(int(f)))
                extra.add(f"{int(f)}.0")
        except Exception:
            pass
    return found | extra


def _ground_set(pack, question: str | None = None) -> set[str]:
    """Normalized numbers an answer may state: the pack's, plus the question's.

    Normalized (commas and % stripped) with sign variants, so "17.9%" in prose
    grounds against 17.9 in the pack and a pack drop of -20.1 can be stated as
    a 20.1 magnitude. A differently-rounded value (17.9 stated as 18) does NOT
    ground; that is an altered number, not a formatting choice.
    """
    allowed: set[str] = set()
    for token in _pack_numbers(pack):
        norm = _norm_num(token)
        if norm:
            allowed.add(norm)
            allowed.add(norm.lstrip("-"))
            allowed.add("-" + norm.lstrip("-"))
    for token in _numbers_in(question or ""):
        norm = _norm_num(token)
        if norm:
            allowed.add(norm)
            allowed.add(norm.lstrip("-"))
    return allowed


def _receipt_grounded(receipt_value: str, allowed: set[str]) -> bool:
    return all(_norm_num(n) in allowed for n in _numbers_in(receipt_value))


def _ungrounded_numbers(data: dict, allowed: set[str]) -> list[str]:
    """Numbers stated in verdict/prose/caveat/follow-ups that are not allowed."""
    texts = [data.get("verdict") or "", data.get("text") or "", data.get("caveat") or ""]
    texts += [str(f) for f in (data.get("followups") or [])]
    bad: list[str] = []
    for text in texts:
        for token in _numbers_in(str(text)):
            if _norm_num(token) not in allowed and token not in bad:
                bad.append(token)
    return bad


def _strip_ungrounded(data: dict, allowed: set[str]) -> dict:
    """Last-resort cleanup: remove what cannot be verified, keep the rest.

    Prose loses the offending sentences; the verdict and caveat lose the
    offending number tokens; follow-ups carrying one are dropped whole.
    """
    out = dict(data)
    text = str(out.get("text") or "")
    if text:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept = [s for s in sentences
                if all(_norm_num(n) in allowed for n in _numbers_in(s))]
        out["text"] = " ".join(kept).strip()
        if not out["text"]:
            out["text"] = "See the receipts below for the numbers behind this call."
    for field in ("verdict", "caveat"):
        value = out.get(field)
        if value:
            cleaned = str(value)
            for token in _numbers_in(cleaned):
                if _norm_num(token) not in allowed:
                    cleaned = cleaned.replace(token, "")
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
            out[field] = cleaned or None
    out["followups"] = [
        f for f in (out.get("followups") or [])
        if all(_norm_num(n) in allowed for n in _numbers_in(str(f)))
    ]
    return out


# -- receipt values: human-readable or omitted ---------------------------------

_RECEIPT_FRIENDLY_KEYS = ("team_name", "league_name", "name", "label", "title")


def _humanize_receipt_value(raw) -> str | None:
    """Render a receipt value as a human-readable string, or None to drop it.

    A model sometimes pastes a whole pack object into a receipt value (the
    league dict arrived as raw JSON in a live answer) or an empty container
    for a field with no data. Dicts render as their friendly label field, so
    a league renders as its team or league name, never serialized JSON.
    Empty containers, nulls, and blank strings return None and the row is
    omitted entirely: no data, no row.
    """
    if raw is None:
        return None
    obj = raw
    if isinstance(raw, str):
        s = raw.strip()
        if s in ("", "[]", "{}", "null", "None"):
            return None
        if s[:1] in ("{", "["):
            try:
                obj = json.loads(s)
            except Exception:
                return s
            if isinstance(obj, str):
                return obj.strip() or None
            if obj is None:
                return None
        else:
            return s
    if isinstance(obj, dict):
        for key in _RECEIPT_FRIENDLY_KEYS:
            v = obj.get(key)
            if v not in (None, ""):
                return str(v)
        return None
    if isinstance(obj, (list, tuple)):
        parts = []
        for item in obj[:3]:
            h = _humanize_receipt_value(item)
            if h:
                parts.append(h)
        return ", ".join(parts) if parts else None
    return str(obj)


# -- the citation contract: deterministic required receipt rows ------------------

def _required_receipts(intent: str, pack: dict) -> list[dict]:
    """Receipt rows an answer to this intent must carry, derived from the pack.

    These are the numbers the eval scorer (and any reader) checks for. They
    are built from pack fields only, so they are grounded by construction.
    """
    rows: list[dict] = []
    pack = pack or {}
    if intent == "trade_eval":
        give = (pack.get("give") or {}).get("total_value")
        receive = (pack.get("receive") or {}).get("total_value")
        gap = pack.get("fairness_gap")
        if give is not None:
            rows.append({"label": "You give (total value)", "value": _fmt_num(give)})
        if receive is not None:
            rows.append({"label": "You receive (total value)", "value": _fmt_num(receive)})
        if gap is not None:
            rows.append({"label": "Fairness gap", "value": _fmt_num(gap)})
    elif intent == "metric":
        top = [e for e in (pack.get("leaderboard_top5") or [])
               if isinstance(e, dict) and e.get("value") is not None]
        if top:
            ordered = top[:1] + [e for e in top[1:] if e.get("on_my_roster")] \
                + [e for e in top[1:] if not e.get("on_my_roster")]
            for e in ordered[:4]:
                rows.append({"label": str(e.get("name") or "Leader"),
                             "value": _fmt_num(e["value"])})
    elif intent == "this_week":
        for a in (pack.get("actions") or []):
            if not isinstance(a, dict):
                continue
            kn = a.get("key_number")
            if isinstance(kn, dict):
                kn = kn.get("value")
            if kn is None or kn == "":
                continue
            value = _fmt_num(kn) if isinstance(kn, (int, float)) else str(kn)
            # The label names the action's subject (the headline names
            # the player: "Claim Kaleb Johnson"), never the bare tag.
            rows.append({"label": str(a.get("headline") or a.get("tag") or "action"),
                         "value": value})
            if len(rows) >= 4:
                break
    elif intent == "waivers":
        cands = [c for c in (pack.get("candidates") or []) if isinstance(c, dict)]
        if cands:
            top = cands[0]
            if top.get("value") is not None:
                rows.append({"label": f"{top.get('name')} value",
                             "value": _fmt_num(top["value"])})
            worst = (pack.get("my_worst_by_position") or {}).get(top.get("position")) or {}
            if worst.get("value") is not None:
                rows.append({"label": f"Your worst {top.get('position')} ({worst.get('name')}) value",
                             "value": _fmt_num(worst["value"])})
            if len(cands) > 1 and cands[1].get("value") is not None:
                rows.append({"label": f"{cands[1].get('name')} value",
                             "value": _fmt_num(cands[1]["value"])})
    elif intent == "player":
        role = pack.get("role") or {}
        for key, label in (("target_share", "Target share"),
                           ("touches_per_game", "Touches per game"),
                           ("ppg", "Points per game"),
                           ("snap_share", "Snap share")):
            if role.get(key) is not None and len(rows) < 4:
                rows.append({"label": label, "value": _fmt_num(role[key])})
        player = pack.get("player") or {}
        if player.get("value") is not None and len(rows) < 4:
            rows.append({"label": "Value", "value": _fmt_num(player["value"])})
    elif intent == "playoffs":
        me = pack.get("me") or {}
        scen = pack.get("scenario") or {}
        if me.get("playoff_pct") is not None:
            rows.append({"label": "Your playoff odds (%)", "value": _fmt_num(me["playoff_pct"])})
        if scen.get("wins_to_clinch") is not None:
            rows.append({"label": "Wins to clinch", "value": _fmt_num(scen["wins_to_clinch"])})
        elif me.get("wins_still_needed") is not None:
            rows.append({"label": "Wins still needed (projected)", "value": _fmt_num(me["wins_still_needed"])})
        if pack.get("weeks_remaining") is not None:
            rows.append({"label": "Weeks remaining", "value": _fmt_num(pack["weeks_remaining"])})
    elif intent == "power_rankings":
        me = pack.get("me") or {}
        leader = pack.get("leader") or {}
        if me.get("rank") is not None:
            rows.append({"label": "Your power rank", "value": _fmt_num(me["rank"])})
        if me.get("power_score") is not None:
            rows.append({"label": "Your power score", "value": _fmt_num(me["power_score"])})
        if leader.get("power_score") is not None and leader.get("roster_id") != me.get("roster_id"):
            rows.append({"label": f"{leader.get('team_name')} power score",
                         "value": _fmt_num(leader["power_score"])})
    elif intent == "standings":
        me = pack.get("me") or {}
        leader = pack.get("points_leader") or {}
        if me.get("seed") is not None:
            rows.append({"label": "Your seed", "value": _fmt_num(me["seed"])})
        if me.get("pf") is not None:
            rows.append({"label": "Your points for", "value": _fmt_num(me["pf"])})
        if leader.get("pf") is not None and leader.get("team_name") != me.get("team_name"):
            rows.append({"label": f"{leader.get('team_name')} points for",
                         "value": _fmt_num(leader["pf"])})
    elif intent == "matchup":
        m = pack.get("matchup") or {}
        a, b = m.get("team_a") or {}, m.get("team_b") or {}
        if a.get("win_prob_pct") is not None:
            rows.append({"label": f"{a.get('team_name')} win probability (%)",
                         "value": _fmt_num(a["win_prob_pct"])})
        if a.get("projected") is not None:
            rows.append({"label": f"{a.get('team_name')} projected", "value": _fmt_num(a["projected"])})
        if b.get("projected") is not None:
            rows.append({"label": f"{b.get('team_name')} projected", "value": _fmt_num(b["projected"])})
    elif intent == "schedule":
        me = pack.get("me") or {}
        easiest = pack.get("easiest") or {}
        stretch = (pack.get("playoff_stretch") or {}).get("me") or {}
        if stretch.get("opp_index") is not None:
            rows.append({"label": "Your playoff stretch opponent index",
                         "value": _fmt_num(stretch["opp_index"])})
        if me.get("opp_index") is not None:
            rows.append({"label": "Your remaining opponent index", "value": _fmt_num(me["opp_index"])})
        if easiest.get("opp_index") is not None and easiest.get("roster_id") != me.get("roster_id"):
            rows.append({"label": f"{easiest.get('team_name')} opponent index",
                         "value": _fmt_num(easiest["opp_index"])})
    return rows[:4]


_ORDINAL_SLOTS = {"top": 0, "first": 0, "second": 1, "third": 2, "fourth": 3}


def _name_waiver_receipts(pack: dict, receipts: list) -> list:
    """Name the player in rank-only waiver receipt labels.

    Model receipts sometimes describe a candidate by rank alone ("Top
    waiver candidate", "Second waiver option") without naming him. When
    such a label names nobody in the pack and its ordinal matches a
    ranked candidate slot, the label names that candidate instead: a
    receipt about a specific player names the player.
    """
    cands = [c for c in (pack.get("candidates") or [])
             if isinstance(c, dict) and c.get("name")]
    if not cands:
        return receipts
    names = [str(c["name"]) for c in cands]
    out = []
    for r in receipts:
        if isinstance(r, dict):
            label = str(r.get("label") or "")
            low = label.lower()
            first = low.split(" ")[0].strip("#") if low else ""
            idx = _ORDINAL_SLOTS.get(first)
            if (label and idx is not None and idx < len(cands)
                    and not any(n.lower() in low for n in names)
                    and any(w in low for w in ("waiver", "candidate", "option", "pickup"))):
                r = dict(r)
                r["label"] = f"{names[idx]} value"
        out.append(r)
    return out


def _required_numbers(intent: str, pack: dict) -> set[str]:
    out: set[str] = set()
    for row in _required_receipts(intent, pack):
        out |= _canon_numbers(row["value"])
    return out


def _surface_numbers(data: dict, actions: list) -> set[str]:
    """Every number a reader can see, mirroring the eval scorer's surfaces."""
    out: set[str] = set()
    texts = [data.get("verdict") or "", data.get("text") or "", data.get("caveat") or ""]
    for r in (data.get("receipts") or []):
        if isinstance(r, dict):
            texts += [str(r.get("label") or ""), str(r.get("value") or "")]
    texts += [str(f) for f in (data.get("followups") or [])]
    for a in actions or []:
        if isinstance(a, dict):
            texts += [str(a.get("headline") or ""), str(a.get("detail") or "")]
    for t in texts:
        out |= _canon_numbers(t)
    return out


def _missing_key_numbers(intent: str, pack: dict, data: dict, actions: list) -> list[str]:
    required = _required_numbers(intent, pack)
    if not required:
        return []
    missing = required - _surface_numbers(data, actions)

    def _key(s):
        try:
            return (0, float(s))
        except (TypeError, ValueError):
            return (1, 0.0)

    return sorted(missing, key=_key)


def _ensure_required_receipts(intent: str, pack: dict, receipts: list) -> list:
    """Inject any required receipt rows the final answer still lacks.

    Model receipts are kept first; a missing required row is appended while
    there is room (receipts cap at 4), otherwise it replaces the last
    receipt that is not itself a required row.
    """
    required = _required_receipts(intent, pack)
    if not required:
        return receipts
    out = list(receipts)
    have: set[str] = set()
    for r in out:
        have |= _canon_numbers(r.get("value"))
    req_canons: set[str] = set()
    for row in required:
        req_canons |= _canon_numbers(row["value"])
    for row in required:
        nums = _canon_numbers(row["value"])
        if nums & have:
            continue
        if len(out) < 4:
            out.append(row)
            have |= nums
            continue
        for i in range(len(out) - 1, -1, -1):
            if not (_canon_numbers(out[i].get("value")) & req_canons):
                out[i] = row
                have |= nums
                break
    return out


# -- internal language and follow-up voice ---------------------------------------

_INTERNAL_RE = re.compile(r"evidence[\s_-]*pack|\bpack\b", re.IGNORECASE)

# Follow-ups phrased as instructions to the user to supply data. The
# assistant fetches its own data; a follow-up is a question the user taps.
_SYSTEM_ASK_RE = re.compile(r"^\s*(send|provide|share|upload|paste)\b", re.IGNORECASE)

_FOLLOWUP_TEMPLATES = {
    "start_sit": ["What about next week?", "Who else is on my bench?"],
    "trade_eval": ["What would balance this trade?", "Who else could I ask for?"],
    "buy_sell_hold": ["Who could I trade him for?", "What is his role right now?"],
    "player": ["Buy, sell, or hold on him?", "How is his upcoming schedule?"],
    "metric": ["Who is moving up?", "What is snap share?"],
    "leaderboard": ["Who is rising in target share?", "Who leads in air yards share?",
                    "Which defenses give up the most to tight ends?"],
    "waivers": ["Who else is available?", "What should I do with my team this week?"],
    "this_week": ["Who should I start at FLEX?", "Who is the best waiver add?",
                  "Any buy-lows I should target?"],
    "playoffs": ["What are my odds of a first-round bye?", "Who is on the bubble?",
                 "How hard is my remaining schedule?"],
    "power_rankings": ["What do the standings look like?", "What are my playoff odds?",
                       "Break down my matchup"],
    "standings": ["Where am I in the power rankings?", "Who has the most points for?",
                  "What are my playoff odds?"],
    "matchup": ["Who should I start at FLEX?", "How hard is my remaining schedule?",
                "What are my playoff odds?"],
    "schedule": ["What are my playoff odds?", "Break down my matchup",
                 "Who has the hardest remaining schedule?"],
}


def _internal_hits(data: dict) -> list[str]:
    texts = [data.get("verdict") or "", data.get("text") or "", data.get("caveat") or ""]
    texts += [str(f) for f in (data.get("followups") or [])]
    return [str(t) for t in texts if t and _INTERNAL_RE.search(str(t))]


def _strip_internal_language(data: dict) -> dict:
    """Remove internal terms deterministically when a retry still leaks them.

    Prose loses the offending sentences; a verdict or caveat carrying one is
    dropped whole (they are too short to redact safely); follow-ups carrying
    one are dropped and the template fallback in _clean_followups refills.
    """
    out = dict(data)
    text = str(out.get("text") or "")
    if text:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept = [s for s in sentences if not _INTERNAL_RE.search(s)]
        out["text"] = " ".join(kept).strip()
        if not out["text"]:
            out["text"] = "See the receipts below for the numbers behind this call."
    for field in ("verdict", "caveat"):
        value = out.get(field)
        if value and _INTERNAL_RE.search(str(value)):
            out[field] = None
    out["followups"] = [f for f in (out.get("followups") or [])
                        if not _INTERNAL_RE.search(str(f))]
    return out


def _clean_followups(intent: str, followups) -> list[str]:
    """Keep only user-voiced follow-ups; fall back to per-intent templates.

    A follow-up the user taps is a question in their voice ("Who should I
    start at FLEX?"). Anything that reads as system language (internal
    terms) or as an instruction to send/provide data is rejected; when
    nothing survives, the intent's templates are used instead.
    """
    kept = []
    for f in followups or []:
        s = str(f).strip()
        if not s or len(s) > 100:
            continue
        if _INTERNAL_RE.search(s) or _SYSTEM_ASK_RE.search(s):
            continue
        kept.append(s)
    if not kept:
        kept = list(_FOLLOWUP_TEMPLATES.get(intent) or [])
    return kept[:3]


# -- trade verdict: pinned to the analyzer's fairness band ------------------------

def _trade_verdict(pack: dict) -> str | None:
    """The trade call, derived from the pack totals by the shared fair band.

    utils.trade_value.fair_value_band is the definition the trade calculator
    (api_trade_eval) and the trade outcome analyzer share: fair within
    max(7% of the larger side's total, 25.0). ACCEPT when the user gains
    more than the band, DECLINE when they lose more than the band, FAIR
    inside it. The model never chooses this verdict.
    """
    try:
        give = float(((pack or {}).get("give") or {}).get("total_value") or 0)
        receive = float(((pack or {}).get("receive") or {}).get("total_value") or 0)
    except (TypeError, ValueError):
        return None
    if give <= 0 and receive <= 0:
        return None
    try:
        from utils.trade_value import fair_value_band
        band = fair_value_band(max(give, receive, 1.0))
    except Exception:
        return None
    gap = receive - give
    if abs(gap) <= band:
        return "FAIR"
    return "ACCEPT" if gap > 0 else "DECLINE"


def _ensure_deal_names(pack: dict, data: dict) -> dict:
    """Guarantee every player in the deal is named somewhere in the answer.

    If the model's prose dropped a side's player, a deterministic summary
    sentence built from the pack is prepended (names only, no numbers, so
    it cannot break grounding).
    """
    sides: dict[str, list[str]] = {}
    for side in ("give", "receive"):
        sides[side] = [str(p.get("name")) for p in ((pack.get(side) or {}).get("players") or [])
                       if isinstance(p, dict) and p.get("name")]
    all_names = sides["give"] + sides["receive"]
    if not all_names:
        return data
    full = " ".join([str(data.get("verdict") or ""), str(data.get("text") or "")] +
                    [f"{r.get('label')} {r.get('value')}"
                     for r in (data.get("receipts") or []) if isinstance(r, dict)])
    if all(n.lower() in full.lower() for n in all_names):
        return data

    def _join(names):
        if len(names) <= 2:
            return " and ".join(names)
        return ", ".join(names[:-1]) + " and " + names[-1]

    parts = []
    if sides["give"]:
        parts.append(f"You give {_join(sides['give'])}")
    if sides["receive"]:
        parts.append(f"You receive {_join(sides['receive'])}")
    if not parts:
        return data
    out = dict(data)
    sentence = ". ".join(parts) + "."
    out["text"] = (sentence + " " + str(out.get("text") or "")).strip()
    return out


def no_data_answer(pack: dict) -> dict:
    missing = pack.get("missing") or "that data is not tracked"
    nearest = pack.get("nearest") or []
    text = f"That is not available: {missing}."
    if nearest:
        text += f" The nearest tracked options are {', '.join(str(n) for n in nearest[:3])}. Ask about one of those and I can pull the live numbers."
    link = _league_link((pack or {}).get("type"), pack) or dict(_ACT_LINKS["metric"])
    return {"verdict": None, "text": text, "receipts": [], "caveat": None,
            "followups": [f"What is {n}?" for n in nearest[:2]],
            "act_link": link, "actions": [], "mentions": []}


def gated_answer(pack: dict) -> dict:
    """The honest PRO-gated answer: parity with the AM surfaces, which show
    the metric exists but withhold its numbers on the free tier. No board
    numbers anywhere in this answer."""
    label = pack.get("label") or "That metric"
    text = (f"{label} is a PRO metric on BR Fantasy, so its board is not included "
            "on the free tier. Free boards I can pull right now include target share, "
            "air yards share, and snap share.")
    return {"verdict": None, "text": text, "receipts": [], "caveat": None,
            "followups": ["Who leads in target share?", "Who leads in snap share?"],
            "act_link": _board_link(pack) or dict(_ACT_LINKS["leaderboard"]),
            "actions": []}


# -- leaderboard answers: the board is code, the model only frames it -----------

_BOARD_SYSTEM = (
    "You are the BR Fantasy assistant. The site has already computed the ranked board "
    "in the evidence pack; it is rendered separately and must not be reproduced, "
    "reordered, or extended. Write ONE short intro sentence that frames the question "
    "WITHOUT naming any player or team and without restating any rank; then give 2 to 3 "
    "follow-up questions in the user's voice. Every number you state must appear "
    "verbatim in the pack. Never mention the pack or evidence. No em dashes. "
    "Return JSON only."
)


def _fmt_delta(value) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f == 0:
        return "0"
    return ("+" if f > 0 else "-") + _fmt_num(abs(f))


def _board_verdict(pack: dict) -> str | None:
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


def _board_line(row: dict, pack: dict) -> str:
    kind = pack.get("board_kind")
    rank = row.get("rank")
    name = row.get("name") or ""
    meta = ", ".join(x for x in (row.get("team"), row.get("position")) if x)
    head = f"{rank}. {name}" + (f" ({meta})" if meta else "")
    value = _fmt_num(row.get("value")) if row.get("value") is not None else ""
    if kind == "matchup":
        return (f"{rank}. {row.get('team')}: {value} fantasy points allowed "
                f"per game to {pack.get('position_filter')}")
    if kind == "movers":
        if row.get("delta") is not None:
            return (f"{head}: {_fmt_delta(row['delta'])} to {value} "
                    f"(was {_fmt_num(row.get('baseline_value'))})")
        return f"{head}: {value} {pack.get('label')}"
    if kind == "intersection":
        sec = row.get("secondary") or {}
        return (f"{head}: {value} {pack.get('label')}, "
                f"{_fmt_num(sec.get('value'))} {pack.get('second_label')}")
    line = f"{head}: {value}"
    if kind == "dual":
        sec = row.get("secondary")
        if sec:
            line += (f" - {pack.get('second_label')} {_fmt_num(sec.get('value'))} "
                     f"(#{sec.get('rank')})")
        elif pack.get("second_label"):
            line += f" - {pack.get('second_label')} not qualified"
    return line


def _board_receipt(row: dict, pack: dict) -> dict:
    kind = pack.get("board_kind")
    label = f"{row.get('rank')}. {row.get('name') or row.get('team')}"
    value = _fmt_num(row.get("value")) if row.get("value") is not None else ""
    if kind == "movers" and row.get("delta") is not None:
        return {"label": label, "value": f"{_fmt_delta(row['delta'])} (now {value})"}
    if kind == "intersection":
        sec = row.get("secondary") or {}
        return {"label": label,
                "value": f"{value} - {pack.get('second_label')} {_fmt_num(sec.get('value'))}"}
    if kind == "dual":
        sec = row.get("secondary")
        if sec:
            return {"label": label,
                    "value": (f"{value} - {pack.get('second_label')} "
                              f"{_fmt_num(sec.get('value'))} (#{sec.get('rank')})")}
    return {"label": label, "value": value}


def _board_intro_template(pack: dict) -> str:
    board = pack.get("board") or []
    top = board[0] if board else {}
    kind = pack.get("board_kind")
    label = pack.get("label") or ""
    value = _fmt_num(top.get("value")) if top.get("value") is not None else ""
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


def _compose_leaderboard(question: str, pack: dict, *, preferences=None,
                         assumed_context=None, history=None) -> dict:
    """Deterministic board answer. Ranks, order, values, receipts, and the
    qualification line are built from the pack by code; the model may only
    contribute one framing sentence (accepted only when it names no board
    entry, so it cannot restate or contradict the computed board) and the
    follow-up chips."""
    if pack.get("gated"):
        return gated_answer(pack)
    board = pack.get("board") or []
    payload = {
        "intent": "leaderboard",
        "question": question,
        "evidence_pack": pack,
        "preferences": [p.get("text") if isinstance(p, dict) else str(p)
                        for p in (preferences or [])],
        "assumed_context": assumed_context or {},
        "history": (history or [])[-3:],
        "prompt_version": PROMPT_VERSION,
    }
    allowed = _ground_set(pack, question)
    intro = None
    followups: list = []
    try:
        data = _draft(_BOARD_SYSTEM, payload, allowed)
        followups = data.get("followups") or []
        if pack.get("board_kind") != "matchup":
            text = str(data.get("text") or "").strip()
            first = re.split(r"(?<=[.!?])\s+", text)[0] if text else ""
            names = [str(r.get("name") or "") for r in board]
            if (first and not _INTERNAL_RE.search(first)
                    and not any(n and n.lower() in first.lower() for n in names)):
                intro = first
    except Exception:
        logger.info("[assistant] leaderboard draft failed; template intro",
                    exc_info=True)
    if not intro:
        intro = _board_intro_template(pack)
    body = "\n".join([intro] + [_board_line(r, pack) for r in board])
    qual = pack.get("qualification_text")
    if qual:
        body += "\n" + qual
    return {"verdict": _board_verdict(pack), "text": body,
            "receipts": [_board_receipt(r, pack) for r in board],
            "caveat": None,
            "followups": _clean_followups("leaderboard", followups),
            "act_link": _board_link(pack) or dict(_ACT_LINKS["leaderboard"]),
            "actions": []}


_STRICT_SUFFIX = (
    " STRICT MODE: your previous draft stated numbers that are NOT in the evidence pack. "
    "Use only numbers that appear verbatim in the evidence pack or in the question. "
    "If a number you want is not in the pack, leave the number out entirely."
)

_QUALITY_SUFFIX = (
    " QUALITY MODE: your previous draft failed checks. "
    "Cite every key number named in the quality_retry note, in the text or the receipts. "
    "Never use the words pack or evidence pack; call it your league data. "
    "Follow-ups must be short questions in the user's voice, never instructions to send or provide anything."
)


def _draft(system: str, payload: dict, allowed: set[str]) -> dict:
    """One model call plus the grounding enforcement pass.

    An ungrounded draft is regenerated once under the strict suffix; a
    still-ungrounded retry is stripped deterministically. Never returns
    None: a draft with invented numbers never ships as-is.
    """
    data = call_model_json(system, json.dumps(payload, default=str), SCHEMA, "assistant_answer")
    bad = _ungrounded_numbers(data, allowed)
    if not bad:
        return data
    logger.info("[assistant] ungrounded prose numbers, regenerating once: %s", bad)
    retry_payload = dict(payload)
    retry_payload["grounding_retry"] = {
        "banned_numbers": bad,
        "instruction": "Your previous draft used numbers that are not in the evidence pack. "
                       "Do not use them. Restate the answer using only pack numbers.",
    }
    retry = None
    try:
        retry = call_model_json(system + _STRICT_SUFFIX, json.dumps(retry_payload, default=str),
                                SCHEMA, "assistant_answer")
    except Exception:
        retry = None
    if retry is not None and not _ungrounded_numbers(retry, allowed):
        return retry
    if retry is not None:
        data = retry
    data = _strip_ungrounded(data, allowed)
    logger.info("[assistant] stripped ungrounded prose numbers: %s",
                _ungrounded_numbers(data, allowed) or bad)
    return data


def _final_receipts(data: dict, allowed: set[str]) -> list:
    """Model receipts, humanized and grounding-filtered, capped at 4."""
    receipts = []
    for r in (data.get("receipts") or [])[:6]:
        if not isinstance(r, dict):
            continue
        label = str(r.get("label") or "")
        value = _humanize_receipt_value(r.get("value"))
        if value is None:
            continue
        if _receipt_grounded(value, allowed):
            receipts.append({"label": label, "value": value})
        else:
            logger.info("[assistant] dropped ungrounded receipt %s=%s", label, value)
        if len(receipts) >= 4:
            break
    return receipts


def compose_answer(intent: str, question: str, pack: dict, *, preferences=None, assumed_context=None, history=None) -> dict:
    """Compose an answer dict from an evidence pack. Pure eval seam."""
    if isinstance(pack, dict) and pack.get("gated"):
        return gated_answer(pack)
    if isinstance(pack, dict) and pack.get("available") is False:
        return no_data_answer(pack)
    if intent == "leaderboard":
        return _compose_leaderboard(question, pack, preferences=preferences,
                                    assumed_context=assumed_context, history=history)
    # This-week actions are deterministic from the pack; build them first so
    # the citation check sees the same surfaces the scorer will.
    actions = []
    if intent == "this_week":
        default_href = (_ACT_LINKS.get("this_week") or {}).get("href")
        for a in (pack or {}).get("actions") or []:
            if isinstance(a, dict):
                actions.append({
                    "tag": a.get("tag"), "headline": a.get("headline"),
                    "detail": a.get("detail"),
                    "href": a.get("act_link") or a.get("href") or default_href,
                    "cta_label": _cta_label(a),
                    "key_number": a.get("key_number"),
                })
    payload = {
        "intent": intent,
        "question": question,
        "evidence_pack": pack,
        "preferences": [p.get("text") if isinstance(p, dict) else str(p) for p in (preferences or [])],
        "assumed_context": assumed_context or {},
        "history": (history or [])[-3:],
        "prompt_version": PROMPT_VERSION,
    }
    fixed_verdict = _trade_verdict(pack) if intent == "trade_eval" else None
    if fixed_verdict:
        payload["fixed_verdict"] = fixed_verdict
    allowed = _ground_set(pack, question)
    data = _draft(SYSTEM, payload, allowed)
    # Quality pass: key-number citation + internal language. One combined
    # regeneration naming both defects, then deterministic fallbacks below.
    missing = _missing_key_numbers(intent, pack, data, actions)
    internal = _internal_hits(data)
    if missing or internal:
        logger.info("[assistant] quality defects (missing=%s, internal=%s), regenerating once",
                    missing, bool(internal))
        retry_payload = dict(payload)
        retry_payload["quality_retry"] = {
            "missing_key_numbers": missing,
            "banned_terms": ["pack", "evidence pack"],
            "instruction": "Cite every missing key number in the text or receipts. "
                           "Do not use the banned terms. Follow-ups are questions in the user's voice.",
        }
        try:
            data = _draft(SYSTEM + _QUALITY_SUFFIX, retry_payload, allowed)
        except Exception:
            logger.info("[assistant] quality retry failed; keeping first draft", exc_info=True)
        missing = _missing_key_numbers(intent, pack, data, actions)
        internal = _internal_hits(data)
    if internal:
        data = _strip_internal_language(data)
    receipts = _ensure_required_receipts(intent, pack, _final_receipts(data, allowed))
    if intent == "waivers":
        receipts = _name_waiver_receipts(pack, receipts)
    if intent == "trade_eval":
        data = _ensure_deal_names(pack, data)
        if fixed_verdict:
            data = dict(data)
            data["verdict"] = fixed_verdict
    caveat = data.get("caveat")
    margin_thin = bool(((pack or {}).get("margin") or {}).get("thin"))
    if caveat and not margin_thin and not (pack or {}).get("stale"):
        caveat = None
    return {
        "verdict": data.get("verdict"),
        "text": data.get("text") or "",
        "receipts": receipts,
        "caveat": caveat,
        "followups": _clean_followups(intent, data.get("followups")),
        "act_link": _resolve_act_link(intent, pack),
        "actions": actions,
        "mentions": _mentions(intent, pack),
    }


def numbers_shown_from(receipts) -> dict:
    return {r["label"]: r["value"] for r in (receipts or []) if isinstance(r, dict) and r.get("label")}
