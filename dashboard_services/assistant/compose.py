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

Follow-ups are {label, ask} pairs, never bare strings: the label is the
chip text the user sees (an inviting offer is fine there), the ask is the
exact message the chip sends, so the ask must read as the user's own
question. Clicking a chip that says "Want the cheapest buy-in for Zay
Flowers?" sends "What is the cheapest way to buy Zay Flowers?", never the
label. An offer-style string with no ask is rejected, not reworded; the
deterministic templates (also pairs) refill instead.

Receipt values are humanized: a value the model pasted as a serialized
object renders as its friendly label (a league renders as its team name,
never raw JSON), and rows with empty values ([], {}, None, blank) are
omitted entirely. No data, no row.

The answer dict is the UI contract (static/app.js initAssistant): act_link is
an object assigned deterministically in code, never by the model: either a
link {href, label} or a player-modal action {player_id, player_name, label}
(or None). This-week actions carry href, cta_label, and the pack's
key_number so the UI can render them as Next Steps rows.

Act links stage state in the site's own tools where a real mechanism exists:
trade answers open the calculator with the pack's deal prefilled (its native
?a=&b= id params), start/sit stages the pair in Compare (?p1=&p2=), metric
answers preset Advanced Metrics (?metric=&pos=), and player answers open the
player modal in place (the UI calls openPlayerModal, no navigation). The answer also carries `mentions` (players the
pack resolved, with ids) so the UI can open the modal from a name in the
text instead of rendering it as plain text.
"""
from __future__ import annotations

import json
import logging
import re
from urllib.parse import quote

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
        "followups": {"type": "array", "items": {"type": "object", "properties": {
            "label": {"type": "string"}, "ask": {"type": "string"}},
            "required": ["label", "ask"], "additionalProperties": False}},
        "act_link": {"type": ["string", "null"]},
    },
    "required": ["verdict", "text", "receipts", "caveat", "followups", "act_link"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are the BR Fantasy assistant. Answer from the evidence pack ONLY. "
    "Verdict first, in the verdict field. Every number you state must appear verbatim in the pack; "
    "never use a number from memory. Receipts: 2 to 4 entries, each a label and the pack value as a string. "
    "Cite the key numbers: for a trade, both side totals and the fairness gap; for trade finding, the "
    "top finding's receive total, give total, and fairness gap; for a metric, the leader "
    "and any of the user's own players on the board; for the weekly digest, each action's key number; "
    "for waivers, the top candidate's value and the user's weakest starter at that position; "
    "for playoffs, the odds and the wins still needed; for standings and power rankings, the user's "
    "own seed or rank and the leader; for a matchup, the win probability and both projected "
    "totals (a matchup answer is only about the matchup: the verdict leads with the win "
    "probability and the projected score, and it never carries waiver targets or weekly "
    "digest actions); "
    "for schedule, the user's opponent strength index and the easiest team's. "
    "For league activity, cite the most recent trade's two received totals and the fairness "
    "gap, and say plainly whether that deal was fair or lopsided; an activity answer reports "
    "what already happened in the league and never recommends a move. "
    "If the pack flags odds_soft, say the odds are soft this early and lead with the wins math. "
    "For trade finding, present only the findings in the evidence: never invent a partner, a package, "
    "or a player swap, and order by mutual fit, not by how much the user wins. When the evidence marks "
    "no_fit, say plainly that there is no clean fit right now, name the closest miss if one is given, "
    "and point at the nearest alternative instead of forcing a bad trade. When the evidence carries "
    "give_position, the findings already honor it (the user's outgoing side is anchored at that "
    "position): present the top finding as the answer to that ask, and never substitute a deal "
    "built on another position. "
    "for breakout calls, the top candidates and their scores; for team grades, the strongest and "
    "weakest teams and their grade scores; for a draft pick, its value and who owns it. "
    "For keepers, cite each kept player's cost round and value, and the first player on the cut line. "
    "For a replacement, cite the subject's value and age and the top candidate's value, age, and "
    "where he is rostered; name every candidate's location (your roster, the team that rosters "
    "him, or waivers). For a handcuff, cite the next man up and his value, and say whether he is "
    "claimable on waivers or rostered, and by whom. For a stash, cite the top stash candidate's "
    "value and age. For a drop, cite the most droppable player's value and the depth you hold at "
    "his position. For a streamer, cite his projection for this week and his matchup. For worth "
    "in picks, cite the player's value and the closest pick equivalent's label and value. "
    "For rookies, cite the stored prospect score and the overall and position ranks exactly as given; "
    "mention the score version only if the user asks about it. "
    "For cross-league answers, attach the league name to every number you state and compare teams only "
    "by their standing inside their own league (record, rank, positional ranks); never compare raw "
    "points or values across leagues, and name any league whose data is unavailable. "
    "For the watchlist, cite the biggest 7-day mover's current value and 7-day move, and name "
    "any watched player carrying an injury designation. The watchlist is read-only in chat: "
    "never say you added, removed, or starred a player. "
    "For front office answers, cite the playoff odds and the top trade target's two side "
    "totals, and present the computed top moves, targets, and cut candidates as the site's "
    "data in your own words; the report's written memo is not in your evidence, so never "
    "quote or invent memo language. "
    "For start/sit, cite the recommended starter's projection and the projection of the "
    "player he is compared against. When the evidence is roster-scoped (a slot and "
    "decisions from the user's own lineup), the verdict names the recommended starter "
    "at that slot, the text names the current starter he replaces (or that he stays "
    "ahead of) and the other candidates, and a sit question names who sits and who "
    "starts in his place. "
    "If fixed_verdict is present in the payload, the verdict field must be exactly that value; "
    "it is set by the site's trade fairness band and is not yours to change. "
    "Caveat only when the pack marks margin.thin true or data stale, otherwise null. "
    "If assumed_context.assumed_league_uncertain is true, state the assumed league in the text. "
    "If a stored preference shapes the answer, name that preference in the text. "
    "Never mention the pack, evidence, tools, or any internal data structure; call it your league data. "
    "Follow-ups are objects with a label and an ask. The label is the short chip text the user "
    "sees; it may be phrased as an inviting offer, like 'Want the cheapest buy-in for Zay Flowers?'. "
    "The ask is the exact message sent when the chip is tapped, so write it as a natural question "
    "in the user's own voice, like 'What is the cheapest way to buy Zay Flowers?'. An ask is never "
    "an offer from you and never an instruction to send, provide, or share anything. "
    "No em dashes. Use plain hyphens. Short, direct sentences. Return JSON only."
)

# Deterministic act links per intent (UI contract: {href, label} or None).
# These are the fallbacks; _resolve_act_link upgrades most intents to a
# deep link that stages state in the target tool (see the builders below).
# player / buy_sell_hold have NO static fallback on purpose: their act link
# is the modal action from _player_modal_link, and with no resolved player
# there is no button at all (never a trip to the rankings page).
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
    """The 'View player' act link as a modal action, not an href: the UI
    renders it as a button that calls openPlayerModal in place (the same
    path the answer's player mentions use), so the popup stays open behind
    the modal and nothing navigates, least of all to the rankings page."""
    if not isinstance(player, dict) or not player.get("player_id"):
        return None
    return {"player_id": str(player["player_id"]),
            "player_name": str(player.get("name") or ""),
            "label": "View player"}


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
    if (pack or {}).get("board_kind") in ("buy_low", "sell_high"):
        # Value boards come from the site's movers data, so their home is
        # the Top Movers page, not an Advanced Metrics preset.
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


def _finding_link(pack: dict, finding: dict) -> dict | None:
    """Stage one finding in the trade calculator, the way the calculator's
    own share links do: /{platform}/{season}/{league_id}/trade?a=<ids the
    viewer receives>&b=<ids the viewer gives>. Built from pack ids only."""
    league = (pack or {}).get("league") or {}
    season, league_id = league.get("season"), league.get("league_id")
    if not season or not league_id or not isinstance(finding, dict):
        return None
    recv = [str(p.get("player_id")) for p in finding.get("receive") or []
            if isinstance(p, dict) and p.get("player_id")]
    give = [str(p.get("player_id")) for p in finding.get("give") or []
            if isinstance(p, dict) and p.get("player_id")]
    if not recv or not give:
        return None
    return {"href": f"/{league.get('platform') or 'sleeper'}/{season}/{league_id}"
                    f"/trade?a={','.join(recv)}&b={','.join(give)}",
            "label": "Open in Trade Calculator"}


def _finding_act_link(pack: dict) -> dict | None:
    """Answer-level link for trade finding: the top finding staged in the
    calculator, else the league's Suggestions tab."""
    findings = (pack or {}).get("findings") or []
    if findings:
        link = _finding_link(pack, findings[0])
        if link:
            return link
    league = (pack or {}).get("league") or {}
    if league.get("season") and league.get("league_id"):
        return {"href": f"/{league.get('platform') or 'sleeper'}/{league['season']}"
                        f"/{league['league_id']}/trade?tab=suggestions",
                "label": "Open Trade Suggestions"}
    return None

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
        # Terminal: the modal action when the pack resolved a player, else
        # no act link at all. Never fall through to a page navigation.
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
    elif intent == "trade_finding":
        link = _finding_act_link(pack)
        if link:
            return link
    elif intent == "replacement":
        # The next step depends on where the top candidate lives: a free
        # agent is a waiver claim, another roster's player is a trade.
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
    elif intent == "rookie":
        # Prospect ids are pipeline ids, not NFL ids: only a prospect who
        # already has a Sleeper id can open the player modal.
        prospect = pack.get("prospect") or {}
        if prospect.get("sleeper_id") and prospect.get("name"):
            _add({"player_id": prospect["sleeper_id"], "name": prospect["name"]})
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
    "activity": ("activity", "View Activity"),
    "front_office": ("dashboard", "View Front Office"),
    "live": ("scorezone", "Open ScoreZone"),
    "draft": ("draft", "Open Draft Room"),
    "nfl_schedule": ("scorezone", "Open ScoreZone"),
}


def _league_link(intent: str, pack: dict) -> dict | None:
    spec = _LEAGUE_LINKS.get(intent or "")
    league = (pack or {}).get("league") or {}
    if not spec or not league.get("platform") or not league.get("season") or not league.get("league_id"):
        return None
    page, label = spec
    return {"href": f"/{league['platform']}/{league['season']}/{league['league_id']}/{page}",
            "label": label}



def _pack_act_link(intent: str, pack: dict) -> dict | None:
    """League-scoped act link built from the pack, for intents whose page
    lives under the league path (the Teams page has no bare route)."""
    if intent not in ("teams", "pick_ownership"):
        return None
    league = (pack or {}).get("league") or {}
    platform, season, league_id = league.get("platform"), league.get("season"), league.get("league_id")
    if not (platform and season and league_id):
        return None
    return {"href": f"/{platform}/{season}/{league_id}/teams", "label": "View on Teams"}


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


def _allowed_canon(pack, question: str | None = None) -> set[str]:
    """The eval scorer's allowed set, canonicalized: every number in the
    pack JSON plus every number in the question, in _canon form. The
    scorer compares canonical-to-canonical (sign dropped, trailing zeros
    trimmed); the final grounding gate uses this set so its pass/fail
    line is exactly the scorer's."""
    allowed: set[str] = set()
    for token in _numbers_in(json.dumps(pack or {}, default=str)):
        allowed.add(_canon(token))
    for token in _numbers_in(question or ""):
        allowed.add(_canon(token))
    return allowed


def _receipt_grounded(receipt_value: str, allowed: set[str]) -> bool:
    return all(_norm_num(n) in allowed for n in _numbers_in(receipt_value))


def _ungrounded_numbers(data: dict, allowed: set[str]) -> list[str]:
    """Numbers stated in verdict/prose/caveat/follow-ups that are not allowed."""
    texts = [data.get("verdict") or "", data.get("text") or "", data.get("caveat") or ""]
    for f in (data.get("followups") or []):
        texts += _followup_texts(f)
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
        if all(_norm_num(n) in allowed
               for t in _followup_texts(f) for n in _numbers_in(t))
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

def _pack_picks(pack: dict) -> list[dict]:
    """Every pick entry in a trade pack, side-attributed: the unified
    ``picks`` list when the builder carries one, else the per-side
    ``give_picks`` / ``receive_picks`` lists with the side filled in."""
    picks = [p for p in ((pack or {}).get("picks") or []) if isinstance(p, dict)]
    if picks:
        return picks
    out = []
    for side, key in (("give", "give_picks"), ("receive", "receive_picks")):
        for p in (pack or {}).get(key) or []:
            if isinstance(p, dict):
                entry = dict(p)
                entry.setdefault("side", side)
                out.append(entry)
    return out


def _pick_label(pick: dict) -> str | None:
    """The canonical pick label ('2027 1.11'): the builder's label when
    present, else rendered from the canonical pick id (#2618) or the
    season/round/slot fields."""
    pick = pick or {}
    if pick.get("label"):
        return str(pick["label"])
    parts = str(pick.get("pick_id") or "").split("_")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return f"{parts[0]} {int(parts[1])}.{int(parts[2]):02d}"
    season, rnd, slot = pick.get("season"), pick.get("round"), pick.get("slot")
    if season is not None and rnd is not None:
        if slot is not None:
            return f"{season} {int(rnd)}.{int(slot):02d}"
        return f"{season} round {int(rnd)}"
    return None


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
        # Every priced pick in the deal gets a row under its canonical
        # label (live eval te4: the pick was never named by its label).
        for pick in _pack_picks(pack):
            if len(rows) >= 4:
                break
            label = _pick_label(pick)
            if label and pick.get("value") is not None:
                side = pick.get("side")
                rows.append({"label": (f"{label} (pick you {side})"
                                       if side in ("give", "receive") else label),
                             "value": _fmt_num(pick["value"])})
    elif intent == "trade_finding":
        equiv = pack.get("pick_equivalents") or {}
        if equiv.get("equivalents"):
            sell = pack.get("sell_player") or {}
            if sell.get("value") is not None:
                rows.append({"label": f"{sell.get('name')} value",
                             "value": _fmt_num(sell["value"])})
            for e in equiv["equivalents"][:2]:
                if e.get("value") is not None and len(rows) < 4:
                    rows.append({"label": str(e.get("label") or "Pick"),
                                 "value": _fmt_num(e["value"])})
        top = (pack.get("findings") or [None])[0] or pack.get("closest_miss") or {}
        if top.get("receive_total") is not None and len(rows) < 4:
            rows.append({"label": "You receive (total value)", "value": _fmt_num(top["receive_total"])})
        if top.get("give_total") is not None and len(rows) < 4:
            rows.append({"label": "You give (total value)", "value": _fmt_num(top["give_total"])})
        if top.get("fairness_gap") is not None and len(rows) < 4:
            rows.append({"label": "Fairness gap", "value": _fmt_num(top["fairness_gap"])})
    elif intent == "start_sit":
        players = [p for p in (pack.get("players") or []) if isinstance(p, dict)]
        pick = next((p for p in players
                     if str(p.get("player_id")) == str(pack.get("pick_player_id") or "")), None)
        others = [p for p in players if p is not pick]
        # The comparison the answer makes: the slot's current occupant in
        # roster mode, else the runner-up.
        other = (next((p for p in others if p.get("slot_occupant")), None)
                 or next((p for p in others
                          if str(p.get("player_id")) == str(pack.get("sit_player_id") or "")), None)
                 or (others[0] if others else None))
        for p in ([pick] if pick else []) + ([other] if other else []):
            if p.get("projection") is not None:
                rows.append({"label": f"{p.get('name')} projection",
                             "value": _fmt_num(p["projection"])})
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
        mode = pack.get("waiver_mode")
        if mode == "drop":
            for c in (pack.get("drop_ranking") or [])[:2]:
                if isinstance(c, dict) and c.get("value") is not None:
                    rows.append({"label": f"{c.get('name')} value",
                                 "value": _fmt_num(c["value"])})
        elif mode == "stash":
            for c in (pack.get("stash_candidates") or [])[:1]:
                if isinstance(c, dict) and c.get("value") is not None:
                    rows.append({"label": f"{c.get('name')} value",
                                 "value": _fmt_num(c["value"])})
                if isinstance(c, dict) and c.get("age") is not None and len(rows) < 4:
                    rows.append({"label": f"{c.get('name')} age",
                                 "value": _fmt_num(c["age"])})
        elif mode == "stream":
            for c in (pack.get("streamers") or [])[:1]:
                if isinstance(c, dict) and c.get("projection") is not None:
                    rows.append({"label": f"{c.get('name')} projection",
                                 "value": _fmt_num(c["projection"])})
                if isinstance(c, dict) and c.get("value") is not None and len(rows) < 4:
                    rows.append({"label": f"{c.get('name')} value",
                                 "value": _fmt_num(c["value"])})
        cands = [c for c in (pack.get("candidates") or []) if isinstance(c, dict)]
        if cands and len(rows) < 4:
            top = cands[0]
            if top.get("value") is not None:
                rows.append({"label": f"{top.get('name')} value",
                             "value": _fmt_num(top["value"])})
            worst = (pack.get("my_worst_by_position") or {}).get(top.get("position")) or {}
            if worst.get("value") is not None and len(rows) < 4:
                rows.append({"label": f"Your worst {top.get('position')} ({worst.get('name')}) value",
                             "value": _fmt_num(worst["value"])})
            if len(cands) > 1 and cands[1].get("value") is not None and len(rows) < 4:
                rows.append({"label": f"{cands[1].get('name')} value",
                             "value": _fmt_num(cands[1]["value"])})
    elif intent == "replacement":
        subject = pack.get("subject") or {}
        if subject.get("value") is not None:
            rows.append({"label": f"{subject.get('name')} value",
                         "value": _fmt_num(subject["value"])})
        if subject.get("age") is not None and len(rows) < 4:
            rows.append({"label": f"{subject.get('name')} age",
                         "value": _fmt_num(subject["age"])})
        for c in (pack.get("candidates") or [])[:1]:
            if isinstance(c, dict) and c.get("value") is not None and len(rows) < 4:
                rows.append({"label": f"{c.get('name')} value",
                             "value": _fmt_num(c["value"])})
            if isinstance(c, dict) and c.get("age") is not None and len(rows) < 4:
                rows.append({"label": f"{c.get('name')} age",
                             "value": _fmt_num(c["age"])})
    elif intent == "handcuff":
        nxt = pack.get("next_up") or {}
        if nxt.get("value") is not None:
            rows.append({"label": f"{nxt.get('name')} value",
                         "value": _fmt_num(nxt["value"])})
        subject = pack.get("subject") or {}
        if subject.get("value") is not None and len(rows) < 4:
            rows.append({"label": f"{subject.get('name')} value",
                         "value": _fmt_num(subject["value"])})
        for c in (pack.get("my_options") or [])[:1]:
            if isinstance(c, dict) and c.get("value") is not None and len(rows) < 4:
                rows.append({"label": f"{c.get('name')} value",
                             "value": _fmt_num(c["value"])})
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
        traj = pack.get("odds_trajectory") or {}
        if traj.get("start_pct") is not None and traj.get("start_week") is not None:
            rows.append({"label": f"Your playoff odds in Week {traj['start_week']} (%)",
                         "value": _fmt_num(traj["start_pct"])})
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
        # The runner-up's score is part of the headline picture ("who is
        # the team to beat" is answered against the #2 as much as the #1):
        # pin it too, or the citation contract depends on the model
        # volunteering it (live eval pr2: 83.6 never cited).
        ranked = pack.get("teams") or []
        second = ranked[1] if len(ranked) > 1 else {}
        if (isinstance(second, dict) and second.get("power_score") is not None
                and len(rows) < 4
                and second.get("roster_id") not in (me.get("roster_id"),
                                                    leader.get("roster_id"))):
            rows.append({"label": f"{second.get('team_name')} power score",
                         "value": _fmt_num(second["power_score"])})
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
    elif intent == "breakout_board":
        for c in (pack.get("candidates") or [])[:3]:
            if isinstance(c, dict) and c.get("breakout_score") is not None and len(rows) < 4:
                rows.append({"label": str(c.get("name") or "Candidate"),
                             "value": _fmt_num(c["breakout_score"])})
    elif intent == "breakout_track_record":
        overall = pack.get("overall") or {}
        for key, label in (("graded", "Graded calls"), ("hit_rate_pct", "Hit rate (percent)"),
                           ("hit", "Hits"), ("miss", "Misses")):
            if overall.get(key) is not None and len(rows) < 4:
                rows.append({"label": label, "value": _fmt_num(overall[key])})
    elif intent == "breakout_history":
        earliest = pack.get("earliest_call") or {}
        if earliest.get("label"):
            rows.append({"label": "First breakout call", "value": str(earliest["label"])})
        if earliest.get("breakout_score") is not None:
            rows.append({"label": "Breakout score", "value": _fmt_num(earliest["breakout_score"])})
        current = pack.get("current_call") or {}
        if current.get("breakout_score") is not None and len(rows) < 4:
            rows.append({"label": "Current breakout score",
                         "value": _fmt_num(current["breakout_score"])})
    elif intent == "teams":
        breadth_top = pack.get("most_top_players") or {}
        if breadth_top.get("count") is not None:
            rows.append({"label": (f"Most top-{pack.get('breadth_n') or 24} "
                                   f"players ({breadth_top.get('team_name')})"),
                         "value": _fmt_num(breadth_top["count"])})
        ranking = pack.get("position_ranking") or []
        focus = pack.get("position_focus")
        if focus and ranking:
            first, last = ranking[0], ranking[-1]
            if first.get("strength") is not None:
                rows.append({"label": f"Best {focus} room ({first.get('team_name')})",
                             "value": _fmt_num(first["strength"])})
            if last.get("strength") is not None and len(rows) < 4:
                rows.append({"label": f"Weakest {focus} room ({last.get('team_name')})",
                             "value": _fmt_num(last["strength"])})
        else:
            strongest = pack.get("strongest") or {}
            if strongest.get("grade_score") is not None:
                rows.append({"label": f"Strongest roster ({strongest.get('team_name')}) grade score",
                             "value": _fmt_num(strongest["grade_score"])})
            mine = next((t for t in (pack.get("teams") or []) if t.get("is_viewer")), None)
            if mine and mine.get("grade_score") is not None and len(rows) < 4:
                rows.append({"label": "Your grade score", "value": _fmt_num(mine["grade_score"])})
            weakest = pack.get("weakest") or {}
            if weakest.get("grade_score") is not None and len(rows) < 4:
                rows.append({"label": f"Weakest roster ({weakest.get('team_name')}) grade score",
                             "value": _fmt_num(weakest["grade_score"])})
    elif intent == "pick_ownership":
        pick = pack.get("pick") or {}
        if pick.get("value") is not None:
            rows.append({"label": f"{pick.get('label')} value", "value": _fmt_num(pick["value"])})
    elif intent == "keeper":
        if (pack.get("mode") or "") == "dynasty":
            for p in (pack.get("roster_core") or [])[:2]:
                if p.get("value") is not None:
                    rows.append({"label": f"{p.get('name')} value",
                                 "value": _fmt_num(p["value"])})
            if pack.get("roster_value_total") is not None and len(rows) < 4:
                rows.append({"label": "Roster value total",
                             "value": _fmt_num(pack["roster_value_total"])})
        else:
            for p in (pack.get("my_keepers") or [])[:2]:
                if p.get("value") is not None:
                    rows.append({"label": f"{p.get('name')} value",
                                 "value": _fmt_num(p["value"])})
                if p.get("cost_round") is not None and len(rows) < 4:
                    rows.append({"label": f"{p.get('name')} keeper cost (round)",
                                 "value": _fmt_num(p["cost_round"])})
            asked = pack.get("asked_player") or {}
            if asked.get("value") is not None and len(rows) < 4:
                rows.append({"label": f"{asked.get('name')} value",
                             "value": _fmt_num(asked["value"])})
    elif intent == "rookie":
        prospect = pack.get("prospect") or {}
        board = [prospect] if prospect else (pack.get("prospects") or [])
        for p in board[:2]:
            if p.get("prospect_score") is not None:
                rows.append({"label": f"{p.get('name')} prospect score",
                             "value": _fmt_num(p["prospect_score"])})
        if prospect and prospect.get("projected_pick") is not None and len(rows) < 4:
            rows.append({"label": "Projected draft pick",
                         "value": _fmt_num(prospect["projected_pick"])})
        line = prospect.get("college_latest") or {}
        for key, label in (("receiving_yards", "Receiving yards (latest college season)"),
                           ("rush_yards", "Rushing yards (latest college season)"),
                           ("pass_yards", "Passing yards (latest college season)")):
            if line.get(key) is not None and len(rows) < 4:
                rows.append({"label": label, "value": _fmt_num(line[key])})
                break
    elif intent == "activity":
        latest = pack.get("latest_trade") or {}
        for t in (latest.get("teams") or [])[:2]:
            if isinstance(t, dict) and t.get("gets_value") is not None:
                rows.append({"label": f"{t.get('team_name')} received (total value)",
                             "value": _fmt_num(t["gets_value"])})
        if latest.get("value_gap") is not None and len(rows) < 4:
            rows.append({"label": "Fairness gap", "value": _fmt_num(latest["value_gap"])})
        if not rows:
            counts = pack.get("counts") or {}
            if counts.get("trades") is not None:
                rows.append({"label": "Trades this season", "value": _fmt_num(counts["trades"])})
            if counts.get("waiver_adds") is not None and len(rows) < 4:
                rows.append({"label": "Waiver adds this season",
                             "value": _fmt_num(counts["waiver_adds"])})
    elif intent == "watchlist":
        top = pack.get("top_mover") or {}
        if top.get("value") is not None:
            rows.append({"label": f"{top.get('name')} value", "value": _fmt_num(top["value"])})
        if top.get("delta7") is not None and len(rows) < 4:
            rows.append({"label": f"{top.get('name')} 7-day move",
                         "value": _fmt_num(top["delta7"])})
        if not rows and pack.get("count") is not None:
            rows.append({"label": "Players on your watchlist",
                         "value": _fmt_num(pack["count"])})
    elif intent == "front_office":
        if pack.get("playoff_pct") is not None:
            rows.append({"label": "Playoff odds (%)",
                         "value": _fmt_num(pack["playoff_pct"])})
        top = (pack.get("trade_targets") or [None])[0]
        if top and top.get("get_total") is not None:
            rows.append({"label": f"{top.get('partner')} trade: you receive (total value)",
                         "value": _fmt_num(top["get_total"])})
        if top and top.get("give_total") is not None and len(rows) < 4:
            rows.append({"label": f"{top.get('partner')} trade: you give (total value)",
                         "value": _fmt_num(top["give_total"])})
    elif intent == "live":
        mu = pack.get("matchup") or {}
        a, b = mu.get("team_a") or {}, mu.get("team_b") or {}
        if a.get("current_points") is not None:
            rows.append({"label": f"{a.get('team_name')} points so far",
                         "value": _fmt_num(a["current_points"])})
        if b.get("current_points") is not None:
            rows.append({"label": f"{b.get('team_name')} points so far",
                         "value": _fmt_num(b["current_points"])})
        if a.get("win_prob_pct") is not None and len(rows) < 4:
            rows.append({"label": f"{a.get('team_name')} win probability (%)",
                         "value": _fmt_num(a["win_prob_pct"])})
        if a.get("projected") is not None and len(rows) < 4:
            rows.append({"label": f"{a.get('team_name')} projected final",
                         "value": _fmt_num(a["projected"])})
    elif intent == "draft":
        cur = pack.get("current_pick") or {}
        if cur.get("pick_no") is not None:
            rows.append({"label": "Current pick", "value": _fmt_num(cur["pick_no"])})
        nxt = pack.get("my_next_pick") or {}
        if nxt.get("pick_no") is not None and len(rows) < 4:
            rows.append({"label": "Your next pick",
                         "value": _fmt_num(nxt["pick_no"])})
        best = (pack.get("best_available") or [None])[0]
        if best and best.get("value") is not None and len(rows) < 4:
            rows.append({"label": f"{best.get('name')} value",
                         "value": _fmt_num(best["value"])})
        prep = pack.get("prep") or {}
        # Prep receipts pin the slotted target first, then the cluster
        # behind him and the headliner above the slot (the faller).
        for cand in ([prep.get("target")] + list(prep.get("next_up") or [])
                     + list(prep.get("fallers") or [])):
            if len(rows) >= 4:
                break
            if cand and cand.get("score") is not None:
                rows.append({"label": f"{cand.get('name')} prospect score",
                             "value": _fmt_num(cand["score"])})
        if not rows and pack.get("picks_made") is not None:
            rows.append({"label": "Picks made", "value": _fmt_num(pack["picks_made"])})
    elif intent == "cross_league":
        for s in (pack.get("leagues") or []):
            if len(rows) >= 4:
                break
            if not isinstance(s, dict) or s.get("rank") is None:
                continue
            if s.get("record"):
                rows.append({"label": f"{s.get('league_name')} record",
                             "value": str(s["record"])})
            if len(rows) < 4:
                rows.append({"label": f"{s.get('league_name')} rank",
                             "value": _fmt_num(s["rank"])})
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
    for f in (data.get("followups") or []):
        texts += _followup_texts(f)
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

def followup_pair(label: str, ask: str | None = None) -> dict:
    """One follow-up chip: {label, ask}.

    The label is the chip text the user sees; the ask is the exact message
    sent when the chip is tapped. Most chips read as the user's own
    question, so the ask defaults to the label; an offer-style label
    ("Want the cheapest buy-in for Zay Flowers?") must carry a hand-written
    user-voice ask ("What is the cheapest way to buy Zay Flowers?").
    """
    label = str(label)
    return {"label": label, "ask": label if ask is None else str(ask)}


# Deterministic per-intent fallbacks. Every entry is a pair; the strings
# below are already phrased as the user's own questions, so each label is
# also its ask (followup_pair defaults the ask to the label).
_FOLLOWUP_TEMPLATES = {
    "start_sit": [followup_pair("What about next week?"), followup_pair("Who else is on my bench?")],
    "trade_eval": [followup_pair("What would balance this trade?"), followup_pair("Who else could I ask for?")],
    "trade_finding": [followup_pair("Who else could I target?"), followup_pair("What can I get for my spare players?"),
                      followup_pair("What should I do with my team this week?")],
    "buy_sell_hold": [followup_pair("Who could I trade him for?"), followup_pair("What is his role right now?")],
    "player": [followup_pair("Buy, sell, or hold on him?"), followup_pair("How is his upcoming schedule?")],
    "metric": [followup_pair("Who is moving up?"), followup_pair("What is snap share?")],
    "leaderboard": [followup_pair("Who is rising in target share?"), followup_pair("Who leads in air yards share?"),
                    followup_pair("Which defenses give up the most to tight ends?")],
    "waivers": [followup_pair("Who else is available?"), followup_pair("What should I do with my team this week?")],
    "this_week": [followup_pair("Who should I start at FLEX?"), followup_pair("Who is the best waiver add?"),
                  followup_pair("Any buy-lows I should target?")],
    "playoffs": [followup_pair("What are my odds of a first-round bye?"), followup_pair("Who is on the bubble?"),
                 followup_pair("How hard is my remaining schedule?")],
    "power_rankings": [followup_pair("What do the standings look like?"), followup_pair("What are my playoff odds?"),
                       followup_pair("Break down my matchup")],
    "standings": [followup_pair("Where am I in the power rankings?"), followup_pair("Who has the most points for?"),
                  followup_pair("What are my playoff odds?")],
    "matchup": [followup_pair("Who should I start at FLEX?"), followup_pair("How hard is my remaining schedule?"),
                followup_pair("What are my playoff odds?")],
    "schedule": [followup_pair("What are my playoff odds?"), followup_pair("Break down my matchup"),
                 followup_pair("Who has the hardest remaining schedule?")],
    "breakout_board": [followup_pair("How good is BR at breakout calls?"), followup_pair("Who are the breakout calls at RB?")],
    "breakout_track_record": [followup_pair("Who are this week's breakout calls?"), followup_pair("Who was the biggest breakout hit?")],
    "breakout_history": [followup_pair("Who are this week's breakout calls?"), followup_pair("How good is BR at breakout calls?")],
    "teams": [followup_pair("Who is weakest at TE?"), followup_pair("Grade every team's WR room")],
    "pick_ownership": [followup_pair("What other picks do I own?"), followup_pair("Who has the most draft capital?")],
    "keeper": [followup_pair("Who is first on my cut line?"), followup_pair("What does my last keeper spot come down to?")],
    "rookie": [followup_pair("Who are the top quarterbacks in the class?"), followup_pair("Who is the top prospect?"),
               followup_pair("What about the running backs?")],
    "cross_league": [followup_pair("Which of my teams is strongest?"), followup_pair("What should I do with my team this week?")],
    "replacement": [followup_pair("What could I get for him in a trade?"), followup_pair("Who is his handcuff?")],
    "handcuff": [followup_pair("Who should I start if he is out?"), followup_pair("Who is available on waivers?")],
    "activity": [followup_pair("Who should I add from waivers?"), followup_pair("What should I do with my team this week?")],
    "watchlist": [followup_pair("Who should I add from waivers?"), followup_pair("What should I do with my team this week?")],
    "front_office": [followup_pair("Find me a trade"), followup_pair("Who should I add from waivers?")],
    "live": [followup_pair("What do I need tonight?"), followup_pair("Break down my matchup"),
             followup_pair("Who should I start at FLEX?")],
    "draft": [followup_pair("Who is on the clock?"), followup_pair("What does my roster still need?"),
              followup_pair("Who is the best rookie available?")],
    "open": [followup_pair("What should I do with my team this week?"), followup_pair("Who should I start at FLEX?"),
             followup_pair("Any buy-lows I should target?")],
}


def _followup_texts(f) -> list[str]:
    """Every reader-visible string of one follow-up: label + ask for a
    pair, the string itself for a legacy bare-string follow-up."""
    if isinstance(f, dict):
        return [str(f.get("label") or ""), str(f.get("ask") or "")]
    return [str(f or "")]


def _internal_hits(data: dict) -> list[str]:
    texts = [data.get("verdict") or "", data.get("text") or "", data.get("caveat") or ""]
    for f in (data.get("followups") or []):
        texts += _followup_texts(f)
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
                        if not any(_INTERNAL_RE.search(t) for t in _followup_texts(f))]
    return out


# A follow-up phrased as an offer FROM the assistant ("Want the cheapest
# buy-in for ...?", "Want me to ...?") is the assistant's voice, not the
# user's. As a pair it is fine as a label, because the ask field is what
# gets sent; as a bare string there is no user-voiced ask, so it is
# rejected rather than reworded.
_OFFER_RE = re.compile(r"^\s*want\b", re.IGNORECASE)


def _pair_followup(f) -> dict | None:
    """Normalize one follow-up to a {label, ask} pair, or None to drop it.

    Dict pairs keep their label verbatim (an inviting offer is fine
    there) and are kept when the ask is a clean user-voiced question. A
    bare string becomes a pair with ask == label only when it already
    reads as the user's own question; offer-style and system-style
    strings are dropped, never reworded.
    """
    if isinstance(f, dict):
        label = str(f.get("label") or "").strip()
        ask = str(f.get("ask") or "").strip()
    elif isinstance(f, str):
        label = ask = f.strip()
    else:
        return None
    if not label or not ask or len(label) > 100 or len(ask) > 100:
        return None
    if _INTERNAL_RE.search(label) or _INTERNAL_RE.search(ask):
        return None
    if _SYSTEM_ASK_RE.search(label) or _SYSTEM_ASK_RE.search(ask):
        return None
    if _OFFER_RE.search(ask):
        return None
    return {"label": label, "ask": ask}


def _clean_followups(intent: str, followups) -> list[dict]:
    """Keep only follow-ups whose ask is in the user's voice; fall back
    to per-intent templates.

    A follow-up chip sends its ask, and the ask is a question in the
    user's voice ("Who should I start at FLEX?"). Anything whose ask
    reads as system language (internal terms), as an instruction to
    send/provide data, or as an offer from the assistant is rejected;
    when nothing survives, the intent's templates are used instead.
    """
    kept = []
    for f in followups or []:
        pair = _pair_followup(f)
        if pair is not None:
            kept.append(pair)
    if not kept:
        kept = [dict(p) for p in (_FOLLOWUP_TEMPLATES.get(intent) or [])]
    return kept[:3]


# -- trade verdict: pinned to the analyzer's fairness band ------------------------

def _draft_prep_verdict(pack: dict) -> str | None:
    """The draft-prep call, pinned like the trade verdict: the pack's
    slotting already decided who the target at the named pick is (the
    highest-ranked rookie presumed still available there), so the model
    never chooses this verdict. Without the pin, a prep answer named
    the board's No. 1 player as the target at 1.03 (live report
    2026-10-15): he will be gone two picks earlier."""
    prep = (pack or {}).get("prep") or {}
    target = prep.get("target") or {}
    if target.get("name") and prep.get("slot"):
        return f"Target {target['name']} at {prep['slot']}"
    return None


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
    """Guarantee every asset in the deal is named somewhere in the answer.

    If the model's prose dropped a side's player, or never named a pick
    by its canonical label (live eval te4: '2027 1.11' absent from the
    whole answer), a deterministic summary sentence built from the pack
    is prepended. It carries names and pick labels only, both pack
    fields, so it cannot break grounding.
    """
    sides: dict[str, list[str]] = {}
    required_names: list[str] = []
    required_labels: list[str] = []
    for side in ("give", "receive"):
        assets = [str(p.get("name")) for p in ((pack.get(side) or {}).get("players") or [])
                  if isinstance(p, dict) and p.get("name")]
        required_names += assets
        for pick in _pack_picks(pack):
            if pick.get("side") != side:
                continue
            label = _pick_label(pick)
            if label:
                assets.append(f"the {label} pick")
                required_labels.append(label)
        sides[side] = assets
    if not required_names and not required_labels:
        return data
    prose = " ".join([str(data.get("verdict") or ""), str(data.get("text") or ""),
                      str(data.get("caveat") or "")]).lower()
    full = prose + " " + " ".join(
        f"{r.get('label')} {r.get('value')}"
        for r in (data.get("receipts") or []) if isinstance(r, dict)).lower()
    # Player names may live on any surface; a pick's canonical label must
    # be part of the answer's asset naming (the prose), not only a row.
    if (all(n.lower() in full for n in required_names)
            and all(lbl.lower() in prose for lbl in required_labels)):
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


def _required_mentions(intent: str, pack: dict) -> list[tuple[str, str]]:
    """Names the answer must state, as (name, kind) pairs from the pack.

    An answer that never names the team or league it is about reads as a
    dodge, and the eval checks for it. Kinds: the subject team whose
    playoff odds are quoted (live eval po1: the odds shipped without
    naming Caleb's Casting Couch), the sim's next opponent for a playoff
    scenario (po3), every opponent inside a playoff-stretch window (sc2),
    and every league in a cross-league comparison (xl2: one league named,
    the other dropped).
    """
    pack = pack or {}
    out: list[tuple[str, str]] = []
    if intent == "playoffs":
        subject = ((pack.get("me") or {}).get("team_name")
                   or (pack.get("league") or {}).get("team_name"))
        if subject:
            out.append((str(subject), "subject"))
        opp = (pack.get("scenario") or {}).get("next_opponent")
        if opp:
            out.append((str(opp), "opponent"))
    elif intent == "schedule":
        stretch = (pack.get("playoff_stretch") or {}).get("me") or {}
        out += [(str(o["opponent"]), "stretch")
                for o in (stretch.get("opponents") or [])
                if isinstance(o, dict) and o.get("opponent")]
    elif intent == "cross_league":
        out += [(str(s["league_name"]), "league")
                for s in (pack.get("leagues") or [])
                if isinstance(s, dict) and s.get("league_name")]
    elif intent == "matchup":
        m = pack.get("matchup") or {}
        a = (m.get("team_a") or {}).get("team_name")
        b = (m.get("team_b") or {}).get("team_name")
        if a:
            out.append((str(a), "matchup_a"))
        if b:
            out.append((str(b), "matchup_b"))
    elif intent == "activity":
        latest = pack.get("latest_trade") or {}
        out += [(str(t["team_name"]), "activity")
                for t in (latest.get("teams") or [])
                if isinstance(t, dict) and t.get("team_name")]
        moved: list[str] = []
        for t in (latest.get("teams") or []):
            if not isinstance(t, dict):
                continue
            for p in (t.get("gets") or []):
                if isinstance(p, dict) and p.get("name") \
                        and str(p["name"]) not in moved:
                    moved.append(str(p["name"]))
        out += [(n, "activity_player") for n in moved]
    elif intent == "watchlist":
        top = pack.get("top_mover") or {}
        if top.get("name"):
            out.append((str(top["name"]), "watchlist"))
    elif intent == "live":
        mu = pack.get("matchup") or {}
        for side in ("team_a", "team_b"):
            name = (mu.get(side) or {}).get("team_name")
            if name:
                out.append((str(name), "live_team"))
    elif intent == "draft":
        cur = pack.get("current_pick") or {}
        if cur.get("team_name"):
            out.append((str(cur["team_name"]), "draft_clock"))
        best = (pack.get("best_available") or [None])[0]
        if best and best.get("name"):
            out.append((str(best["name"]), "draft_best"))
    elif intent == "front_office":
        top = (pack.get("trade_targets") or [None])[0]
        if top and top.get("partner"):
            out.append((str(top["partner"]), "front_office"))
    elif intent == "open":
        # The open path has no fixed pack subject; the teams its tool
        # evidence is about are the required mentions (live eval ow12:
        # a record answer that named neither the user's team nor the
        # comparison team). Sources: a team_roster entry's team, and a
        # league_lookup entry's own team plus its headline comparison
        # team (leader / points leader / top table row).
        seen: set[str] = set()
        for entry in (pack.get("evidence") or []):
            if not isinstance(entry, dict):
                continue
            names: list[str] = []
            if entry.get("tool") == "team_roster":
                n = (entry.get("team") or {}).get("team_name")
                if n:
                    names.append(str(n))
            elif entry.get("tool") == "league_lookup":
                me = (entry.get("me") or {}).get("team_name")
                if me:
                    names.append(str(me))
                for field in ("leader", "points_leader"):
                    n = (entry.get(field) or {}).get("team_name")
                    if n:
                        names.append(str(n))
                table = entry.get("table") or []
                if table and isinstance(table[0], dict) \
                        and table[0].get("team_name"):
                    names.append(str(table[0]["team_name"]))
            for n in names:
                if n not in seen:
                    seen.add(n)
                    out.append((n, "open_team"))
        out = out[:3]
    return out


def _join_names(names: list[str]) -> str:
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _ensure_required_mentions(intent: str, pack: dict, data: dict, receipts: list) -> dict:
    """Inject a deterministic sentence naming any required team or league
    the draft skipped: the receipt-injection pattern, for names instead
    of numbers. The check runs on the prose surfaces only (verdict, text,
    caveat): a name that survives only inside a receipt label has not
    been said. The sentences carry no numbers, so they can never break
    grounding."""
    required = _required_mentions(intent, pack)
    if not required:
        return data
    full = " ".join([str(data.get("verdict") or ""), str(data.get("text") or ""),
                     str(data.get("caveat") or "")]).lower()
    missing: dict[str, list[str]] = {}
    for name, kind in required:
        if name.lower() not in full:
            missing.setdefault(kind, []).append(name)
    if not missing:
        return data
    sentences = []
    if "matchup_a" in missing or "matchup_b" in missing:
        # A matchup answer names the pairing, not either side alone:
        # one sentence carrying both full names covers either omission.
        pair = [name for name, kind in required
                if kind in ("matchup_a", "matchup_b")]
        if len(pair) == 2:
            sentences.append(f"The matchup is {pair[0]} against {pair[1]}.")
            missing.pop("matchup_a", None)
            missing.pop("matchup_b", None)
    for kind, names in missing.items():
        if kind == "subject":
            sentences.append(f"These are the playoff odds for {names[0]}.")
        elif kind == "opponent":
            sentences.append(f"Your next matchup is against {names[0]}.")
        elif kind == "matchup_a":
            sentences.append(f"One side of the matchup is {names[0]}.")
        elif kind == "matchup_b":
            sentences.append(f"The other side of the matchup is {names[0]}.")
        elif kind == "activity_player":
            sentences.append(f"That deal moved {_join_names(names)}.")
        elif kind == "open_team":
            if len(names) == 1:
                sentences.append(f"That is the picture for {names[0]}.")
            else:
                sentences.append(f"That covers {_join_names(names)}.")
        elif kind == "activity":
            if len(names) == 1:
                sentences.append(f"That deal involved {names[0]}.")
            else:
                sentences.append("That deal was between " + ", ".join(names[:-1])
                                 + f" and {names[-1]}.")
        elif kind == "watchlist":
            sentences.append(f"The biggest move on your watchlist is {names[0]}.")
        elif kind == "front_office":
            sentences.append(f"The top front office trade target is with {names[0]}.")
        elif kind == "league":
            sentences.append(f"That comparison also includes {_join_names(names)}.")
        elif len(names) == 1:
            sentences.append(f"That stretch includes {names[0]}.")
        else:
            sentences.append("That stretch runs through " + _join_names(names) + ".")
    out = dict(data)
    out["text"] = (str(out.get("text") or "").rstrip() + " "
                   + " ".join(sentences)).strip()
    return out


# Builders mark an unresolvable player question with these internal
# markers; the user-facing decline must ask which player they mean, never
# quote the marker (live bug 2026-10-15: "That is not available: no
# resolved player." answered "Who should I buy low?").
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


# Internal `missing` markers that are NOT sentences: fragments and
# shorthand the builders write for logs and tests. Each maps to the
# plain sentence the user reads. (The old generic wrapper quoted the
# marker verbatim: "That is not available: no buy-low candidates right
# now.", live 2026-10-15.)
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
    """Translate a builder's internal `missing` marker into the plain
    sentence the user reads. Exact map first, then the f-string families,
    then sentence-shaped markers pass through as themselves (capitalized,
    punctuated); anything left falls back to a plain sentence and is
    logged, so no raw marker can reach the user."""
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
    repl = re.match(r"no (\w+) replacement for (.+?) in a ", m)
    if repl:
        return (f"No {repl.group(1)} replacement for {repl.group(2)} "
                "turned up in a comparable value band.")
    words = m.split()
    if len(words) >= 4 and "_" not in m and "{" not in m and "}" not in m:
        text = m[0].upper() + m[1:]
        if not text.endswith((".", "!", "?")):
            text += "."
        return text
    logger.warning("[assistant] untranslated missing marker: %r", m)
    return "I do not have that on the site yet."


def no_data_answer(pack: dict) -> dict:
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
            "followups": [followup_pair(f"What is {n}?") for n in nearest[:2]],
            "act_link": link, "actions": [], "mentions": []}


def _gated_trade_finding_answer(pack: dict) -> dict:
    """Deterministic answer for a PRO-gated trade-finding pack.

    Mirrors the site: /api/trade-suggestions returns a 403 paywall before
    any suggestion is built, so a gated pack carries no scan data and this
    answer is produced without a model call. Nothing to ground, nothing to
    leak.
    """
    league = (pack or {}).get("league") or {}
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
        "followups": [followup_pair("What should I do with my team this week?"),
                      followup_pair("Should I accept a trade I was offered?")],
        "act_link": {"href": pricing, "label": "See PRO plans"},
        "actions": [],
    }


def _gated_front_office_answer(pack: dict) -> dict:
    """Deterministic answer for a PRO-gated front office pack.

    Mirrors the site: /api/gm-memo returns a 403 paywall before any report
    data is built, so a gated pack carries no report data and this answer
    is produced without a model call. Nothing to ground, nothing to leak.
    """
    league = (pack or {}).get("league") or {}
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
        "followups": [followup_pair("What should I do with my team this week?"),
                      followup_pair("Who should I add from waivers?")],
        "act_link": {"href": pricing, "label": "See PRO plans"},
        "actions": [],
    }

def _join_asset_names(players) -> str:
    names = [str(p.get("name")) for p in players or []
             if isinstance(p, dict) and p.get("name")]
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _position_no_fit_answer(pack: dict) -> dict:
    """Deterministic answer when an outgoing-position constraint found no
    deal: the user asked which of HER players at one position lands the
    target ("what rb can I trade for A.J. Brown") and no package anchored
    at that position is plausible and in band. Code-composed like the
    no-fit answer so another position's deal can never be dressed up as
    the answer: the text declines the position framing plainly, then
    offers the best overall fit as a labeled fallback. Every number comes
    from the pack's fallback_finding."""
    pos = str((pack or {}).get("give_position") or "").upper()
    target = ((pack or {}).get("target_player") or {}).get("name") or "that player"
    fb = (pack or {}).get("fallback_finding") or {}
    text = f"No {pos} package on your roster gets within range of {target}."
    if (pack or {}).get("partner_scoped") and (pack or {}).get("partner_team"):
        # The constraint arrived as a partner-want follow-up ("he wants
        # a rb"): the decline is scoped to that partner first, and only
        # the labeled fallback below widens past it.
        text = (f"No {pos} package on your roster gets you {target} from "
                f"{pack['partner_team']}.")
    receipts: list[dict] = []
    if fb and fb.get("receive_total") is not None and fb.get("give_total") is not None:
        give = fb.get("give") or []
        anchor = max(give, key=lambda p: float(p.get("value") or 0)) if give else {}
        fb_pos = str(anchor.get("position") or "").upper()
        gap = fb.get("fairness_gap")
        gap_clause = ""
        if isinstance(gap, (int, float)):
            gap_clause = f", a fairness gap of {_fmt_num(abs(float(gap)))}"
        text += (f" Closest deal overall is a {fb_pos} deal: you send "
                 f"{_join_asset_names(give)} to {fb.get('partner_team')} for "
                 f"{target}, totals {_fmt_num(fb.get('receive_total'))} against "
                 f"{_fmt_num(fb.get('give_total'))}{gap_clause}.")
        receipts = [
            {"label": "You receive (total value)", "value": _fmt_num(fb.get("receive_total"))},
            {"label": "You give (total value)", "value": _fmt_num(fb.get("give_total"))},
        ]
        if isinstance(gap, (int, float)):
            receipts.append({"label": "Fairness gap", "value": _fmt_num(gap)})
    return {"verdict": f"No {pos} deal for {target}", "text": text,
            "receipts": receipts,
            "caveat": None,
            "followups": [followup_pair(f"Who can I trade for {target}?"),
                          followup_pair("What should I do with my team this week?")],
            "act_link": _resolve_act_link("trade_finding", pack),
            "actions": [], "mentions": _mentions("trade_finding", pack)}


def _no_fit_answer(pack: dict) -> dict:
    """Deterministic answer for a trade-finding pack with no clean fit.

    The scan already decided there is no fit; the answer is code-composed
    like the gated and no-data answers so the model can never dress the
    closest miss up as a recommendation (live eval tf4: the draft
    proposed the miss as a real trade). The body states plainly that
    there is no clean fit, names the closest miss and why it misses
    (outside the analyzer's fair band), and points at the nearest
    alternative. Every number comes from the pack.
    """
    miss = (pack or {}).get("closest_miss") or {}
    alt = (pack or {}).get("nearest_alternative")
    if alt == "waivers":
        alt_sentence = " The nearest alternative is the waiver wire."
    elif alt:
        alt_sentence = f" The nearest alternative is {alt}."
    else:
        alt_sentence = ""
    if miss and miss.get("receive_total") is not None and miss.get("give_total") is not None:
        gap = miss.get("fairness_gap")
        gap_clause = ""
        if isinstance(gap, (int, float)):
            gap_clause = f", a fairness gap of {_fmt_num(abs(float(gap)))}"
        text = (f"No clean fit right now. The closest miss is {miss.get('partner_team')}: "
                f"{_join_asset_names(miss.get('receive'))} for "
                f"{_join_asset_names(miss.get('give'))} totals "
                f"{_fmt_num(miss.get('receive_total'))} against "
                f"{_fmt_num(miss.get('give_total'))}{gap_clause}, outside the fair band."
                + alt_sentence)
    elif miss:
        text = (f"No clean fit right now. The closest miss is with "
                f"{miss.get('partner_team')}, and it falls outside the fair band."
                + alt_sentence)
    else:
        text = ("No clean fit right now: no roster in your league lines up on both "
                "sides inside the fair band." + alt_sentence)
    return {"verdict": "No clean fit right now", "text": text,
            "receipts": _required_receipts("trade_finding", pack),
            "caveat": None,
            "followups": [followup_pair("Who is available on waivers?"),
                          followup_pair("What should I do with my team this week?")],
            "act_link": _resolve_act_link("trade_finding", pack),
            "actions": [], "mentions": _mentions("trade_finding", pack)}

def _empty_value_board_answer(pack: dict) -> dict:
    """Deterministic answer for a genuinely empty buy-low / sell-high
    board (the tf4 no-fit pattern): plain copy, no model call, and the
    closest miss named with the gate he failed, so an honest empty can
    never be dressed up as a recommendation or leak the builder's
    internal marker. Every number comes from the pack's closest_miss."""
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
                          f"({_fmt_num(miss.get('baseline_value'))} to "
                          f"{_fmt_num(miss.get('value'))})"),
            }]
    else:
        text = (f"No {word} candidate clears the screen right now: no "
                "mover this week pairs a real price move with a "
                "verifiable production line.")
    other = ("Who should I sell high?" if kind == "buy_low"
             else "Who should I buy low?")
    return {"verdict": f"No {word} clears the screen right now",
            "text": text, "receipts": receipts, "caveat": None,
            "followups": [followup_pair("Who are the biggest movers this week?"),
                          followup_pair(other)],
            "act_link": _board_link(pack) or dict(_ACT_LINKS["leaderboard"]),
            "actions": [], "mentions": []}


def gated_answer(pack: dict) -> dict:
    """The honest PRO-gated answer: parity with the AM surfaces, which show
    the metric exists but withhold its numbers on the free tier. No board
    numbers anywhere in this answer. Trade-finding packs carry their own
    gated answer (the Suggestions paywall)."""
    if (pack or {}).get("type") == "trade_finding":
        return _gated_trade_finding_answer(pack)
    if (pack or {}).get("type") == "front_office":
        return _gated_front_office_answer(pack)
    label = pack.get("label") or "That metric"
    text = (f"{label} is a PRO metric on BR Fantasy, so its board is not included "
            "on the free tier. Free boards I can pull right now include target share, "
            "air yards share, and snap share.")
    return {"verdict": None, "text": text, "receipts": [], "caveat": None,
            "followups": [followup_pair("Who leads in target share?"), followup_pair("Who leads in snap share?")],
            "act_link": _board_link(pack) or dict(_ACT_LINKS["leaderboard"]),
            "actions": []}


# -- leaderboard answers: the board is code, the model only frames it -----------

_BOARD_SYSTEM = (
    "You are the BR Fantasy assistant. The site has already computed the ranked board "
    "in the evidence pack; it is rendered separately and must not be reproduced, "
    "reordered, or extended. Write ONE short intro sentence that frames the question "
    "WITHOUT naming any player or team and without restating any rank; then give 2 to 3 "
    "follow-ups as label/ask objects: the label is the short chip text, the ask is the "
    "question in the user's own voice that is sent when the chip is tapped. Every number you state must appear "
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


def _join_and(parts: list[str]) -> str:
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return "".join(parts)
    if len(parts) == 2:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def _matchup_focus_ranks(focus: dict) -> list[dict]:
    return focus.get("ranks") or []


def _board_verdict(pack: dict) -> str | None:
    board = pack.get("board") or []
    if not board:
        return None
    top = board[0]
    kind = pack.get("board_kind")
    label = pack.get("label") or ""
    if kind == "matchup":
        focus = pack.get("focus") or {}
        ranks = _matchup_focus_ranks(focus)
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


def _breadth_finishes_text(row: dict) -> str:
    """A breadth row's finishes, the receipts' substance: each metric he
    places in, with his rank and value there. Capped at five for
    readability; the tail is "and more" (no count, so no ungrounded
    number can ride in)."""
    fins = row.get("top_finishes") or []
    parts = [f"{f.get('label')} #{f.get('rank')} ({_fmt_num(f.get('value'))})"
             for f in fins[:5]]
    text = ", ".join(parts)
    if len(fins) > len(parts):
        text += ", and more"
    return text


def _value_stats(row: dict) -> str:
    """The production line a value-board row leads with: the recent
    window when the pack carries one, else the season line."""
    if row.get("recent_ppg") is not None:
        return (f"{_fmt_num(row.get('recent_ppg'))} PPG vs "
                f"{_fmt_num(row.get('recent_expected_ppg'))} expected over "
                f"the last {row.get('recent_games')}")
    return (f"{_fmt_num(row.get('ppg'))} PPG vs "
            f"{_fmt_num(row.get('expected_ppg'))} expected")


def _value_fragments(row: dict, pack: dict) -> str:
    """Why-line fragments for a value-board row, built only from fields
    the pack carries: the mechanism, the peer anchor, the slate note."""
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
                         f"{_fmt_num(gap)}% below him")
        else:
            parts.append(f"producing like {anchor} {window}, priced "
                         f"{_fmt_num(gap)}% above him")
    if row.get("slate_note"):
        parts.append(f"{row['slate_note']} (avg matchup rank "
                     f"{_fmt_num(row.get('slate_avg_rank'))})")
    return "; ".join(parts)


_STAT_EXTRA_WORDS = (("rec_yd", "receiving yards"), ("rec", "receptions"),
                     ("rec_tgt", "targets"), ("rec_td", "receiving TDs"),
                     ("rush_yd", "rushing yards"), ("rush_att", "carries"),
                     ("rush_td", "rushing TDs"), ("pass_yd", "passing yards"),
                     ("pass_cmp", "completions"), ("pass_att", "attempts"),
                     ("pass_td", "passing TDs"))


def _stat_extras(row: dict, pack: dict) -> str:
    """The rest of a stat-board row's family line, in parentheses: a
    receiving-yards row also shows his receptions, targets, and TDs.
    The ranked stat itself is skipped (the line already leads with it)."""
    if not (pack or {}).get("stat_board"):
        return ""
    parts = []
    for key, word in _STAT_EXTRA_WORDS:
        if key == pack.get("stat_key"):
            continue
        if row.get(key) is not None:
            parts.append(f"{_fmt_num(row[key])} {word}")
        if len(parts) >= 3:
            break
    return f" ({', '.join(parts)})" if parts else ""


def _board_line(row: dict, pack: dict, position: str | None = None) -> str:
    kind = pack.get("board_kind")
    rank = row.get("rank")
    name = row.get("name") or ""
    meta = ", ".join(x for x in (row.get("team"), row.get("position")) if x)
    head = f"{rank}. {name}" + (f" ({meta})" if meta else "")
    value = _fmt_num(row.get("value")) if row.get("value") is not None else ""
    if pack.get("stat_board"):
        return f"{head}: {value} {(pack.get('label') or '').lower()}" \
               f"{_stat_extras(row, pack)}"
    if kind == "matchup":
        return (f"{rank}. {row.get('team')}: {value} fantasy points allowed "
                f"per game to {position or pack.get('position_filter')}")
    if kind == "movers":
        if row.get("delta") is not None:
            return (f"{head}: {_fmt_delta(row['delta'])} to {value} "
                    f"(was {_fmt_num(row.get('baseline_value'))})")
        return f"{head}: {value} {pack.get('label')}"
    if kind in ("buy_low", "sell_high") and row.get("delta") is not None:
        line = (f"{head}: value {_fmt_delta(row['delta'])} to {value} "
                f"(was {_fmt_num(row.get('baseline_value'))}), "
                f"{_value_stats(row)}")
        frag = _value_fragments(row, pack)
        if frag:
            line += "; " + frag
        return line
    if kind == "intersection":
        sec = row.get("secondary") or {}
        return (f"{head}: {value} {pack.get('label')}, "
                f"{_fmt_num(sec.get('value'))} {pack.get('second_label')}")
    if kind == "breadth":
        n = pack.get("breadth_top_n") or pack.get("requested_n") or 10
        return (f"{head}: top {n} in {value} metrics: "
                f"{_breadth_finishes_text(row)}")
    if kind == "finishes":
        return (f"{head}: {value} top-{pack.get('finish_n') or 12} finishes "
                f"in {row.get('games')} games ({_fmt_num(row.get('ppg'))} PPG)")
    if kind == "floor":
        return (f"{head}: floor {_fmt_num(row.get('floor'))} points "
                f"(ceiling {_fmt_num(row.get('ceiling'))}, "
                f"{_fmt_num(row.get('ppg'))} PPG over {row.get('games')} games)")
    if kind == "ceiling":
        return (f"{head}: ceiling {_fmt_num(row.get('ceiling'))} points "
                f"(floor {_fmt_num(row.get('floor'))}, "
                f"{_fmt_num(row.get('ppg'))} PPG over {row.get('games')} games)")
    if kind == "value":
        line = f"{head}: {value} value"
        if row.get("age") is not None:
            line += f", age {_fmt_num(row.get('age'))}"
        return line
    if kind == "slate":
        return (f"{head}: {value} {pack.get('label')}, slate avg rank "
                f"{_fmt_num(row.get('slate_avg_rank'))} over the next "
                f"{row.get('slate_games')} games")
    if kind == "gap":
        return (f"{head}: {value} {pack.get('label')} vs "
                f"{_fmt_num(row.get('second_value'))} for "
                f"{row.get('second_name')} (gap {_fmt_num(row.get('gap'))})")
    if kind == "rank_movers":
        return (f"{head}: now #{row.get('new_rank')} at "
                f"{row.get('position')} (was #{row.get('old_rank')}, up "
                f"{row.get('climb')}), value "
                f"{_fmt_num(row.get('current_value'))}")
    if kind == "defenses":
        return (f"{head}: {value} points in {row.get('games')} games "
                f"({_fmt_num(row.get('ppg'))} PPG)")
    line = f"{head}: {value}"
    if kind == "dual":
        sec = row.get("secondary")
        if sec:
            line += (f" - {pack.get('second_label')} {_fmt_num(sec.get('value'))} "
                     f"(#{sec.get('rank')})")
        elif pack.get("second_label"):
            line += f" - {pack.get('second_label')} not qualified"
    return line


def _board_receipt(row: dict, pack: dict, position: str | None = None) -> dict:
    kind = pack.get("board_kind")
    label = f"{row.get('rank')}. {row.get('name') or row.get('team')}"
    value = _fmt_num(row.get("value")) if row.get("value") is not None else ""
    if pack.get("stat_board"):
        return {"label": label, "value": f"{value}{_stat_extras(row, pack)}"}
    if kind == "matchup" and position:
        # One receipt list serves every sub-board of a multi-position
        # answer; the position keeps same-rank labels distinct.
        return {"label": f"{label} vs {position}", "value": value}
    if kind == "movers" and row.get("delta") is not None:
        return {"label": label, "value": f"{_fmt_delta(row['delta'])} (now {value})"}
    if kind in ("buy_low", "sell_high") and row.get("delta") is not None:
        if row.get("recent_ppg") is not None:
            stats = (f"{_fmt_num(row.get('recent_ppg'))} PPG last "
                     f"{row.get('recent_games')} "
                     f"({_fmt_delta(row.get('recent_over_expected'))} "
                     f"vs expected)")
        else:
            stats = (f"{_fmt_num(row.get('ppg'))} PPG "
                     f"({_fmt_delta(row.get('over_expected'))} vs expected)")
        text = f"value {_fmt_delta(row['delta'])} (now {value}); {stats}"
        if row.get("peer_median_value") is not None:
            text += f"; peer median {_fmt_num(row.get('peer_median_value'))}"
        return {"label": label, "value": text}
    if kind == "intersection":
        sec = row.get("secondary") or {}
        return {"label": label,
                "value": f"{value} - {pack.get('second_label')} {_fmt_num(sec.get('value'))}"}
    if kind == "breadth":
        n = pack.get("breadth_top_n") or pack.get("requested_n") or 10
        return {"label": label,
                "value": (f"top {n} in {value} metrics: "
                          f"{_breadth_finishes_text(row)}")}
    if kind == "finishes":
        return {"label": label,
                "value": f"{value} finishes ({_fmt_num(row.get('ppg'))} PPG)"}
    if kind == "floor":
        return {"label": label,
                "value": f"floor {value} (ceiling {_fmt_num(row.get('ceiling'))})"}
    if kind == "ceiling":
        return {"label": label,
                "value": f"ceiling {value} (floor {_fmt_num(row.get('floor'))})"}
    if kind == "value":
        text = value
        if row.get("age") is not None:
            text += f" (age {_fmt_num(row.get('age'))})"
        return {"label": label, "value": text}
    if kind == "slate":
        return {"label": label,
                "value": (f"{value} - slate avg rank "
                          f"{_fmt_num(row.get('slate_avg_rank'))}")}
    if kind == "gap":
        return {"label": label,
                "value": (f"{value} vs {_fmt_num(row.get('second_value'))} "
                          f"({row.get('second_name')})")}
    if kind == "rank_movers":
        return {"label": label,
                "value": (f"#{row.get('old_rank')} to #{row.get('new_rank')} "
                          f"(up {row.get('climb')}, value "
                          f"{_fmt_num(row.get('current_value'))})")}
    if kind == "defenses":
        return {"label": label,
                "value": f"{value} points ({_fmt_num(row.get('ppg'))} PPG)"}
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
        sentences = []
        focus = pack.get("focus") or {}
        ranks = _matchup_focus_ranks(focus)
        if len(ranks) > 1:
            listed = _join_and(
                [f"#{r.get('rank')} of {r.get('pool_size')} against "
                 f"{r.get('position')}s ({_fmt_num(r.get('value'))} fantasy "
                 "points allowed per game)" for r in ranks])
            sentences.append(
                f"The {focus.get('name')} defense ranks {listed}.")
        elif ranks:
            r = ranks[0]
            sentences.append(
                f"The {focus.get('name')} defense ranks #{r.get('rank')} of "
                f"{r.get('pool_size')} against {r.get('position')}s, allowing "
                f"{_fmt_num(r.get('value'))} fantasy points per game to "
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
                        f"{_fmt_num(stop.get('value'))} fantasy points "
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
            scoring = (f"{_fmt_num(top.get('recent_ppg'))} PPG against "
                       f"{_fmt_num(top.get('recent_expected_ppg'))} expected "
                       f"over the last {top.get('recent_games')}")
        else:
            scoring = (f"{_fmt_num(top.get('ppg'))} PPG against "
                       f"{_fmt_num(top.get('expected_ppg'))} expected")
        return (f"{top.get('name')} tops the {word} board: value "
                f"{_fmt_delta(top['delta'])} to {value} while scoring "
                f"{scoring}.")
    if kind == "finishes":
        return (f"{top.get('name')} owns the most top-{pack.get('finish_n') or 12} "
                f"weekly finishes with {value} of them.")
    if kind == "floor":
        return (f"{top.get('name')} has the highest floor at "
                f"{_fmt_num(top.get('floor'))} points in his worst week.")
    if kind == "ceiling":
        return (f"{top.get('name')} has the highest ceiling at "
                f"{_fmt_num(top.get('ceiling'))} points in his best week.")
    if kind == "slate":
        return (f"{top.get('name')} has the easiest next-{pack.get('slate_weeks') or 4} "
                f"slate among the {label} leaders, an average matchup rank "
                f"of {_fmt_num(top.get('slate_avg_rank'))}.")
    if kind == "gap":
        return (f"{top.get('name')} beats his own team's No. 2 option by "
                f"{_fmt_num(top.get('gap'))} in {label}, the widest gap on "
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
                    f"fantasy points {period}: {_fmt_num(focus.get('value'))} "
                    f"points in {focus.get('games')} games "
                    f"({_fmt_num(focus.get('ppg'))} per game).")
            if focus.get("value_rank"):
                text += (f" By model trade value it ranks "
                         f"#{focus.get('value_rank')} of "
                         f"{focus.get('value_pool')} defenses.")
            return text
        return (f"{top.get('name')} leads all defenses {period} at {value} "
                f"fantasy points ({_fmt_num(top.get('ppg'))} per game).")
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
        return _enforce_grounding(gated_answer(pack), "leaderboard", pack, question)
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
    boards = pack.get("boards") or []
    if len(boards) > 1:
        # Multi-board pack (defenses vs several positions): one headed
        # section per sub-board, each with its own basis line; receipts
        # run across the sub-boards in order, position-tagged.
        lines = [intro]
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
        body = "\n".join([intro] + [_board_line(r, pack) for r in board])
        qual = pack.get("qualification_text")
        if qual:
            body += "\n" + qual
        receipts = [_board_receipt(r, pack) for r in board]
    return _enforce_grounding(
        {"verdict": _board_verdict(pack), "text": body,
         "receipts": receipts,
         "caveat": None,
         "followups": _clean_followups("leaderboard", followups),
         "act_link": _board_link(pack) or dict(_ACT_LINKS["leaderboard"]),
         "actions": []},
        "leaderboard", pack, question)


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
    """Model receipts, humanized and grounding-filtered, capped at 4.

    Both halves of a row are checked: the value against the pack numbers
    and the LABEL against the canonical allowed set. The label check is
    the one that was missing: the eval scorer scans receipt labels, and
    a count riding a label ("Top 5 target share", "Keeper 1 value")
    shipped unchecked while its value was grounded (live eval round 2:
    m1 invented '5', kp1 invented '1' and '2', prose fully clean).
    """
    allowed_canon = {_canon(t) for t in allowed}
    receipts = []
    for r in (data.get("receipts") or [])[:6]:
        if not isinstance(r, dict):
            continue
        label = str(r.get("label") or "")
        value = _humanize_receipt_value(r.get("value"))
        if value is None:
            continue
        if not _receipt_grounded(value, allowed):
            logger.info("[assistant] dropped ungrounded receipt %s=%s", label, value)
        elif _canon_numbers(label) - allowed_canon:
            logger.info("[assistant] dropped receipt with ungrounded label %s=%s",
                        label, value)
        else:
            receipts.append({"label": label, "value": value})
        if len(receipts) >= 4:
            break
    return receipts


def _enforce_grounding(answer: dict, intent: str, pack: dict, question: str) -> dict:
    """Final parity gate: no number the eval scorer would call invented
    ships on ANY reader-visible surface of the finished answer.

    The draft-time checks scan verdict/text/caveat/follow-ups and filter
    receipt values, but surfaces assembled after the draft (receipt
    labels, injected rows, action rows) historically escaped every
    check; live eval round 2 shipped invented numbers there with the
    prose fully clean. This gate re-checks the complete answer under
    the scorer's own rule (canonical numbers vs the pack plus the
    question) and resolves deterministically: prose loses offending
    sentences, verdict and caveat lose offending tokens, and
    follow-ups, receipts, and actions carrying an offending number are
    dropped whole. A clean answer passes through untouched.
    """
    allowed_canon = _allowed_canon(pack, question)

    def _bad(text) -> bool:
        return bool(_canon_numbers(text) - allowed_canon)

    out = dict(answer)
    prose_bad = any(_bad(out.get(k)) for k in ("verdict", "text", "caveat")) or \
        any(_bad(t) for f in (out.get("followups") or [])
            for t in _followup_texts(f))
    if prose_bad:
        # The stripper's norm-based check is a superset of the canonical
        # one (same numeric values, stricter spelling), so it always
        # resolves what this gate flags.
        stripped = _strip_ungrounded(
            {"verdict": out.get("verdict"), "text": out.get("text"),
             "caveat": out.get("caveat"),
             "followups": out.get("followups") or []},
            _ground_set(pack, question))
        for k in ("verdict", "text", "caveat", "followups"):
            out[k] = stripped.get(k)
        logger.info("[assistant] final grounding gate stripped prose numbers")
    receipts = []
    for r in (out.get("receipts") or []):
        if isinstance(r, dict) and (_bad(r.get("label")) or _bad(r.get("value"))):
            logger.info("[assistant] final grounding gate dropped receipt %s",
                        r.get("label"))
            continue
        receipts.append(r)
    out["receipts"] = receipts
    actions = []
    for a in (out.get("actions") or []):
        if isinstance(a, dict) and (_bad(a.get("headline")) or _bad(a.get("detail"))):
            logger.info("[assistant] final grounding gate dropped action %s",
                        a.get("headline"))
            continue
        actions.append(a)
    out["actions"] = actions
    if not out.get("followups"):
        out["followups"] = _clean_followups(intent, [])
    return out


# Appended to SYSTEM only when the caller carries a confirmed risk
# profile, so profile-less requests stay byte-identical to before. The
# profile shapes framing and tie-break language inside close calls; it
# never changes a number, a receipt, or a verdict the data decides.
_PROFILE_SUFFIX = (
    " The payload risk_profile holds stances this user has confirmed about how they play. "
    "Use a stance only to shape framing and tie-break language when the call is close "
    "(margin.thin true). It never overrides a clear margin and never adds a number. "
    "When a stance shapes the answer, name it in the text."
)


def _compose_sweep(pack: dict) -> dict:
    """Deterministic category-sweep answer (Teams pack with a sweep):
    the categories and their winners are computed in the pack, so the
    text and receipts only render them; no model draft can miscount a
    category."""
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
                     f"{_fmt_num(c.get('a_value'))} vs {bn} "
                     f"{_fmt_num(c.get('b_value'))} ({winner})")
        if len(receipts) < 4:
            receipts.append({
                "label": f"{c.get('category')} ({an} vs {bn})",
                "value": (f"{_fmt_num(c.get('a_value'))} to "
                          f"{_fmt_num(c.get('b_value'))}")})
    return {"verdict": verdict, "text": "\n".join(lines),
            "receipts": receipts, "caveat": None,
            "followups": [followup_pair("Grade every team's WR room"),
                          followup_pair("Who has the strongest roster?")],
            "act_link": _pack_act_link("teams", pack)
            or {"href": "/teams", "label": "View on Teams"},
            "actions": [], "mentions": []}


def _nfl_game_line(g: dict) -> str:
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


def _nfl_kickoff_receipt(g: dict) -> dict:
    return {"label": f"{g.get('away_name')} at {g.get('home_name')} kickoff",
            "value": f"{g.get('date_label')}, {g.get('kickoff_label')}"}


def _nfl_bucket(g: dict) -> str:
    m = re.search(r"(\d{1,2}):(\d{2})\s*(AM|PM)",
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


def _compose_nfl_schedule(pack: dict) -> dict:
    """Deterministic NFL-schedule answer: the games, days, kickoff times
    and networks are computed in the pack, so the text only renders
    them; no draft can move a kickoff, invent a network, or answer a
    bare schedule question with the user's fantasy matchup."""
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

    if scope == "team":
        team = pack.get("team") or {}
        name = team.get("name") or team.get("abbr") or "That team"
        followups = [followup_pair("Who plays tonight?"),
                     followup_pair("What games are on Sunday?")]
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
        return {"verdict": verdict, "text": text, "receipts": receipts,
                "caveat": None, "followups": followups,
                "act_link": _league_link("nfl_schedule", pack),
                "actions": [], "mentions": []}

    if scope == "week":
        verdict = f"Week {week} NFL schedule"
        by_day: dict[str, list] = {}
        for g in games:
            by_day.setdefault(g.get("date_label") or "", []).append(g)
        lines = [f"Week {week} NFL schedule:"]
        for day_label, rows in by_day.items():
            lines.append(f"{day_label}: "
                         + "; ".join(_nfl_game_line(g) for g in rows))
        receipts = [_nfl_kickoff_receipt(g) for g in games[:4]]
        followups = [followup_pair("Who plays tonight?"),
                     followup_pair("What do I need tonight to win?")]
        return {"verdict": verdict, "text": "\n".join(lines),
                "receipts": receipts, "caveat": None,
                "followups": followups,
                "act_link": _league_link("nfl_schedule", pack),
                "actions": [], "mentions": []}

    # Day scopes: tonight / today / tomorrow / a named weekday.
    day_name = pack.get("day_name") or ""
    rel = {"tonight": "tonight", "today": "today",
           "tomorrow": "tomorrow"}.get(scope)
    followups = [followup_pair("Who plays this week?"),
                 followup_pair("What do I need tonight to win?")]
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
        return {"verdict": verdict, "text": text, "receipts": receipts,
                "caveat": None, "followups": followups,
                "act_link": _league_link("nfl_schedule", pack),
                "actions": [], "mentions": []}
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
        receipts = _single_receipts(g)
        return {"verdict": verdict, "text": text, "receipts": receipts,
                "caveat": None, "followups": followups,
                "act_link": _league_link("nfl_schedule", pack),
                "actions": [], "mentions": []}
    verdict = f"{len(games)} games on {day_name}"
    lines = [f"{day_name}'s NFL games:"]
    for bucket in ("Morning", "Early", "Late", "Primetime"):
        rows = [g for g in games if _nfl_bucket(g) == bucket]
        if rows:
            lines.append(f"{bucket}: "
                         + "; ".join(_nfl_game_line(g) for g in rows))
    receipts = [_nfl_kickoff_receipt(g) for g in games[:4]]
    return {"verdict": verdict, "text": "\n".join(lines),
            "receipts": receipts, "caveat": None, "followups": followups,
            "act_link": _league_link("nfl_schedule", pack),
            "actions": [], "mentions": []}


def compose_answer(intent: str, question: str, pack: dict, *, preferences=None, assumed_context=None, history=None, risk_profile=None) -> dict:
    """Compose an answer dict from an evidence pack. Pure eval seam."""
    if intent == "open":
        # The open path composes from fetched tool evidence: the service
        # plans and fetches first; the eval seam passes frozen evidence
        # straight through in the pack. Its own gating, ambiguity, and
        # decline handling lives in open_layer (entry-level, not pack-level).
        from dashboard_services.assistant import open_layer
        return open_layer.compose_open_answer(
            question, pack, history=history, preferences=preferences,
            assumed_context=assumed_context)
    if isinstance(pack, dict) and pack.get("gated"):
        return _enforce_grounding(gated_answer(pack), intent, pack, question)
    if isinstance(pack, dict) and pack.get("available") is False \
            and pack.get("type") == "leaderboard" \
            and pack.get("board_kind") in ("buy_low", "sell_high") \
            and not pack.get("historical"):
        # A genuinely empty value board gets the plain no-fit treatment
        # (closest miss named), never the generic no-data wrapper. A
        # historical ask is different: value boards have no past-season
        # form at all, so it takes the plain no-data shape.
        return _enforce_grounding(_empty_value_board_answer(pack), intent, pack, question)
    if isinstance(pack, dict) and pack.get("available") is False:
        return _enforce_grounding(no_data_answer(pack), intent, pack, question)
    if intent == "trade_finding" and isinstance(pack, dict) \
            and pack.get("position_no_fit"):
        # The outgoing-position constraint emptied the findings while an
        # overall fit exists: the answer is code-composed so the fallback
        # deal can never be presented as if it answered the position ask.
        return _enforce_grounding(_position_no_fit_answer(pack), intent, pack, question)
    if intent == "trade_finding" and isinstance(pack, dict) and pack.get("no_fit") \
            and not pack.get("findings") and not pack.get("pick_equivalents"):
        # The scan found no clean fit: the answer is code-composed so the
        # closest miss can never be dressed up as a recommendation.
        return _enforce_grounding(_no_fit_answer(pack), intent, pack, question)
    if intent == "leaderboard":
        return _compose_leaderboard(question, pack, preferences=preferences,
                                    assumed_context=assumed_context, history=history)
    if intent == "teams" and isinstance(pack, dict) and pack.get("sweep"):
        # A category sweep is fully computed in the pack; render it
        # deterministically instead of letting a draft count categories.
        return _enforce_grounding(_compose_sweep(pack), intent, pack, question)
    if intent == "nfl_schedule":
        # The NFL fixture list is fully computed in the pack (games,
        # kickoffs, networks); render it deterministically so a bare
        # schedule question can never be answered with fantasy
        # projections or a moved kickoff.
        return _enforce_grounding(_compose_nfl_schedule(pack), intent, pack, question)
    # This-week actions are deterministic from the pack; build them first so
    # the citation check sees the same surfaces the scorer will.
    actions = []
    if intent == "trade_finding":
        # One action row per finding, staged in the trade calculator: the
        # row is the finding, one tap from a staged deal the user sends.
        for f in ((pack or {}).get("findings") or [])[:3]:
            if not isinstance(f, dict):
                continue
            link = _finding_link(pack, f)
            recv_names = ", ".join(str(p.get("name")) for p in f.get("receive") or []
                                    if isinstance(p, dict) and p.get("name"))
            give_names = ", ".join(str(p.get("name")) for p in f.get("give") or []
                                    if isinstance(p, dict) and p.get("name"))
            gap = f.get("fairness_gap")
            key_number = None
            if isinstance(gap, (int, float)):
                key_number = {"value": ("+" if gap > 0 else "") + _fmt_num(gap),
                              "label": "value gap"}
            actions.append({
                "tag": "trade",
                "headline": f"Get {recv_names} from {f.get('partner_team') or 'another roster'}",
                "detail": f"You send {give_names}" if give_names else "",
                "href": (link or {}).get("href") or "/trade?tab=suggestions",
                "cta_label": "Open in Trade Calculator",
                "key_number": key_number,
            })
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
        "risk_profile": dict(risk_profile) if risk_profile else {},
        "assumed_context": assumed_context or {},
        "history": (history or [])[-3:],
        "prompt_version": PROMPT_VERSION,
    }
    system = SYSTEM + _PROFILE_SUFFIX if risk_profile else SYSTEM
    fixed_verdict = None
    if intent == "trade_eval":
        fixed_verdict = _trade_verdict(pack)
    elif intent == "draft":
        fixed_verdict = _draft_prep_verdict(pack)
    if fixed_verdict:
        payload["fixed_verdict"] = fixed_verdict
    allowed = _ground_set(pack, question)
    data = _draft(system, payload, allowed)
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
                           "Do not use the banned terms. Follow-ups are label/ask objects; "
                           "each ask is a question in the user's voice.",
        }
        try:
            data = _draft(system + _QUALITY_SUFFIX, retry_payload, allowed)
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
    if intent == "draft" and fixed_verdict:
        data = dict(data)
        data["verdict"] = fixed_verdict
    if intent == "draft":
        # A prep answer states its availability assumption (the board
        # goes in order, chalk) or the target reads as a guarantee the
        # site never made. The pack carries the sentence; append it
        # verbatim when the draft skipped it. It carries no numerals,
        # so it can never break grounding.
        assumption = ((pack or {}).get("prep") or {}).get("assumption")
        if assumption:
            prose = (str(data.get("verdict") or "") + " "
                     + str(data.get("text") or "")).lower()
            if ("in order" not in prose and "chalk" not in prose
                    and "presumed taken" not in prose):
                data = dict(data)
                data["text"] = (str(data.get("text") or "").rstrip()
                                + " " + assumption).strip()
    data = _ensure_required_mentions(intent, pack, data, receipts)
    verdict = data.get("verdict")
    if intent == "player" and not (verdict and str(verdict).strip()):
        # A player answer never ships an empty verdict (live eval pl2/pl3:
        # the model intermittently returns none). The fallback names the
        # pack's subject, the same "{name} update" shape the frozen eval
        # fixtures use.
        subject = ((pack or {}).get("player") or {}).get("name")
        if subject:
            verdict = f"{subject} update"
    if intent == "watchlist" and not (verdict and str(verdict).strip()):
        # Same failure class as pl2/pl3, one intent over (live eval
        # round 3 wl1): the model returned no verdict for a watchlist
        # answer. The fallback is the pack's top entry by name: the
        # biggest mover, else the first watched player.
        top = ((pack or {}).get("top_mover") or {}).get("name")
        if not top:
            for p in (pack or {}).get("players") or []:
                if isinstance(p, dict) and p.get("name"):
                    top = p["name"]
                    break
        if top:
            verdict = str(top)
    if intent == "live" and not (verdict and str(verdict).strip()):
        # Same empty-verdict failure class: the fallback is the score
        # line itself, from the pack's target pairing.
        mu = (pack or {}).get("matchup") or {}
        a, b = mu.get("team_a") or {}, mu.get("team_b") or {}
        if a.get("team_name") and b.get("team_name"):
            mine = bool(mu.get("is_mine"))
            lead = (a.get("current_points") or 0) >= (b.get("current_points") or 0)
            if mine:
                verdict = (f"You {'lead' if lead else 'trail'} "
                           f"{_fmt_num(a.get('current_points'))} to "
                           f"{_fmt_num(b.get('current_points'))}")
            else:
                verdict = (f"{a['team_name']} {('leads' if lead else 'trails')} "
                           f"{_fmt_num(a.get('current_points'))} to "
                           f"{_fmt_num(b.get('current_points'))}")
    if intent == "draft" and not (verdict and str(verdict).strip()):
        # Same failure class: the fallback is the best player still
        # available, else the prep target at the named slot.
        best = ((pack or {}).get("best_available") or [None])[0]
        cand = ((pack or {}).get("prep") or {}).get("target") or {}
        if cand and cand.get("name"):
            verdict = str(cand["name"])
        elif best and best.get("name"):
            verdict = str(best["name"])
    caveat = data.get("caveat")
    margin_thin = bool(((pack or {}).get("margin") or {}).get("thin"))
    if caveat and not margin_thin and not (pack or {}).get("stale"):
        caveat = None
    return _enforce_grounding({
        "verdict": verdict,
        "text": data.get("text") or "",
        "receipts": receipts,
        "caveat": caveat,
        "followups": _clean_followups(intent, data.get("followups")),
        "act_link": _resolve_act_link(intent, pack),
        "actions": actions,
        "mentions": _mentions(intent, pack),
    }, intent, pack, question)


def numbers_shown_from(receipts) -> dict:
    return {r["label"]: r["value"] for r in (receipts or []) if isinstance(r, dict) and r.get("label")}
