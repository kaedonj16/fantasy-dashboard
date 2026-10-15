"""Intent router: cheap model classification with a keyword fallback."""
from __future__ import annotations

import json
import logging
import re

from dashboard_services.ai.client import AIUnavailableError, AIRateLimitError
from dashboard_services.assistant import ROUTER_PROMPT_VERSION
from dashboard_services.assistant.model import call_model_json

logger = logging.getLogger(__name__)

INTENTS = ["start_sit", "trade_eval", "trade_finding", "buy_sell_hold", "player", "metric", "leaderboard", "waivers", "this_week",
           "playoffs", "power_rankings", "standings", "matchup", "schedule",
           "breakout_board", "breakout_track_record", "breakout_history", "teams", "pick_ownership",
           "keeper", "rookie", "cross_league", "replacement", "handcuff", "activity", "watchlist", "front_office",
           "remember", "out_of_scope"]

SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "player_names": {"type": "array", "items": {"type": "string"}},
        "give_player_names": {"type": "array", "items": {"type": "string"}},
        "receive_player_names": {"type": "array", "items": {"type": "string"}},
        "picks": {"type": "array", "items": {"type": "string"}},
        "give_picks": {"type": "array", "items": {"type": "string"}},
        "receive_picks": {"type": "array", "items": {"type": "string"}},
        "other_team_hint": {"type": ["string", "null"]},
        "team_names": {"type": "array", "items": {"type": "string"}},
        "metric_hint": {"type": ["string", "null"]},
        "target_position": {"type": ["string", "null"]},
        "position_hint": {"type": ["string", "null"]},
        "teams_mode": {"type": ["string", "null"], "enum": ["dynasty", "this_season", None]},
        "week": {"type": ["integer", "null"]},
        "draft_year": {"type": ["integer", "null"]},
        "position": {"type": ["string", "null"]},
        "preference_text": {"type": ["string", "null"]},
        "board_kind": {"type": ["string", "null"], "enum": ["board", "intersection", "dual", "movers", "matchup", "buy_low", "sell_high", None]},
        "second_metric_hint": {"type": ["string", "null"]},
        "top_n": {"type": ["integer", "null"]},
        "position": {"type": ["string", "null"]},
        "team": {"type": ["string", "null"]},
        "window_weeks": {"type": ["integer", "null"]},
        "direction": {"type": ["string", "null"], "enum": ["risers", "fallers", None]},
        "replace_mode": {"type": ["string", "null"], "enum": ["younger", "older", "cheaper", None]},
        "waiver_mode": {"type": ["string", "null"], "enum": ["stash", "drop", "stream", None]},
        "value_in_picks": {"type": ["boolean", "null"]},
        "shop_pool": {"type": ["boolean", "null"]},
    },
    "required": ["intent", "player_names", "give_player_names", "receive_player_names", "picks", "give_picks", "receive_picks", "other_team_hint", "team_names", "metric_hint", "target_position", "position_hint", "teams_mode", "week", "draft_year", "preference_text", "board_kind", "second_metric_hint", "top_n", "position", "team", "window_weeks", "direction", "replace_mode", "waiver_mode", "value_in_picks", "shop_pool"],
    "additionalProperties": False,
}

SYSTEM = (
    "You route fantasy football questions for the BR Fantasy assistant. "
    "Classify into exactly one intent and extract entities verbatim from the message. "
    "start_sit: who to start. trade_eval: a proposed trade with give and receive sides, or whether to "
    "accept a specific deal (should I trade X for Y is trade_eval). "
    "trade_finding: the user wants you to FIND or propose a trade (find me a trade, who should I "
    "target, what can I get for X, I need a running back, who is gettable). For trade_finding, put a "
    "named player (theirs to move, or a specific target on another team) in player_names and a wanted "
    "position in target_position (QB, RB, WR, TE), else null. "
    "If the user asks what a named player is worth in picks, or what picks they could get for him, "
    "it is trade_finding with value_in_picks true and the player in player_names. If the user asks "
    "who they should trade away or shop with no player named, it is trade_finding with shop_pool true. "
    "Draft picks go in give_picks or receive_picks by side when the side is clear, else in picks. "
    "buy_sell_hold: buy/sell/hold on a NAMED player (is X a buy low, should I buy X, "
    "sell or hold X). A buy or sell question that names no player is a leaderboard "
    "board instead (see below). player: a general player question. "
    "metric: a question about one named player and a metric, or a metric definition "
    "(metric_hint is the metric name). "
    "leaderboard: a pool-wide ranking question with no single player as the subject: "
    "top N in a metric, who leads a metric, players ranked by a metric, players in the "
    "top N of two metrics (board_kind intersection, second_metric_hint is the second "
    "metric), one metric ranked with a second metric shown alongside (board_kind dual), "
    "biggest risers or fallers in a metric (board_kind movers, direction risers or "
    "fallers), pool-wide buy or sell questions that name no player (who should I buy "
    "low, buy-low candidates, who is undervalued: board_kind buy_low; who should I "
    "sell high, sell-high candidates, who is overvalued: board_kind sell_high), "
    "or which defenses give up the most to a position (board_kind matchup, "
    "position is the offensive position). For leaderboard also extract top_n when a "
    "count is stated, position (QB/RB/WR/TE) and team (NFL abbreviation) filters when "
    "stated, and window_weeks when a last-N-weeks window is stated. "
    "waivers: adds, drops, FAAB, stashes, or streaming a position for this week. waiver_mode is "
    "stash for stash questions (who should I stash, best stash), drop for who-to-drop questions "
    "(who should I drop, my most droppable player, who do I cut), stream for streaming questions "
    "(who should I stream at TE), else null; position is the position filter when one is stated. "
    "this_week: a plan or actions question, what to do with my "
    "team this week; NOT whether the user will win this week's game (that is matchup, even "
    "when the question says 'this week'). "
    "playoffs: playoff odds, clinching, elimination, or what a team needs to make the playoffs. "
    "power_rankings: the power rankings, who is strongest, or the team to beat; a ranking of team "
    "strength, NOT the win-loss standings table. "
    "standings: the standings table itself, records, points for, seeds, or division races; the "
    "win-loss table, NOT power rankings. "
    "matchup: a head-to-head matchup preview, a win probability between two teams this week, "
    "or ANY question about the outcome of the user's own game this week: will I win, am I "
    "going to win, do I beat X, what are my chances, am I projected to win, will I lose. "
    "Outcome questions are matchup even when they name no team and even when they say "
    "'this week'; team_names holds any team names mentioned. "
    "schedule: remaining schedule strength, the easiest or hardest remaining schedule, or how "
    "hard a specific stretch of weeks is (a playoff stretch question is schedule, not playoffs). "
    "breakout_board: this week's breakout calls or the breakout board. "
    "breakout_track_record: how good or accurate BR's breakout calls have been. "
    "breakout_history: whether a specific named player was a breakout call (the player goes in player_names). "
    "teams: roster strength or grades across the league's teams (position_hint is a position room like TE or WR when one is named; "
    "teams_mode is this_season when they ask about this season or right now, otherwise dynasty). "
    "pick_ownership: who owns a draft pick (the pick text goes in picks). "
    "keeper: deciding which players to keep into next season, keeper costs or rounds, "
    "or keeping a player versus a draft pick. If a trade offer with two sides is on the "
    "table, it is trade_eval even when the user says keep; keeper is for the keeper "
    "decision itself, outside a specific offer. "
    "rookie: questions about college prospects and upcoming NFL draft classes (best "
    "rookie RBs in a class, tell me about a prospect, top players in the 2027 class); "
    "draft_year is the class year when stated and position the position filter when stated. "
    "A named player who is already in the NFL, asked about as a player, is player, "
    "buy_sell_hold, or start_sit as usual; rookie needs the prospect or draft-class framing. "
    "replacement: a succession or comparable-player question about ONE named player (a younger "
    "replacement for X, who can replace X long term, an older win-now version of X, a cheaper "
    "version of X). The named player goes in player_names; replace_mode is younger, older, or "
    "cheaper. It is not a trade offer and not a general player question. "
    "handcuff: who is next up behind one named player (his handcuff, who benefits if he gets "
    "hurt, who is next up behind him, who replaces him if he is out). The named player goes in "
    "player_names. "
    "cross_league: questions that explicitly span the user's leagues (across my leagues, "
    "normal intent even if the user has other leagues. "
    "activity: what has already happened in the user's league: completed trades, "
    "waiver moves that were made, and injury headlines (what trades went down, "
    "league activity, recent transactions, who traded). Past events only: advice "
    "about what to do next is waivers, trade_finding, or this_week, never activity. "
    "watchlist: the user's own watchlist of starred players (show my watchlist, who is on "
    "my watchlist, what is happening with my watchlist, any watchlist alerts): which "
    "watched players are moving in value or carrying an injury designation. Read-only: "
    "adding or removing players happens in the Watchlist tool, never in chat. "
    "front_office: the Front Office (GM) report for the user's team (my front office "
    "report, GM memo, front office top moves): its computed direction, playoff odds, "
    "weakest positions, top moves, trade and waiver targets, and cut candidates. A "
    "general what-should-I-do question without the report named stays with this_week. "
    "remember: the user asks you to remember a preference (preference_text is the preference). "
    "out_of_scope: anything else (trivia, real-game talk, other leagues). "
    "Use thread history only to resolve short follow-ups. Return JSON only. No em dashes."
)

_BOARD_ENTITY_DEFAULTS = {"board_kind": None, "second_metric_hint": None, "top_n": None,
                          "position": None, "team": None, "window_weeks": None,
                          "direction": None}


_METRIC_PHRASES = ("yards per route run", "air yards share", "target share",
                   "snap share", "opportunity share", "usage trend", "wopr",
                   "route rate", "yards per target", "yards per reception")

_POSITION_WORDS = (("quarterback", "QB"), ("running back", "RB"),
                   ("wide receiver", "WR"), ("tight end", "TE"),
                   (" qbs", "QB"), (" rbs", "RB"), (" wrs", "WR"), (" tes", "TE"),
                   (" qb", "QB"), (" rb", "RB"), (" wr", "WR"), (" te", "TE"))


def _metric_hints_in(m: str) -> list[str]:
    # Text order: the first metric named is the primary board.
    return sorted((cand for cand in _METRIC_PHRASES if cand in m), key=m.find)


# Pool-wide buy/sell: a buy or sell question that names no player is a
# candidate board, not a single-player buy_sell_hold (which would decline
# with no resolved player). Every pattern is pool-shaped (who-questions,
# "candidates", the plural noun), so a named control ("Is X a buy low?",
# "Should I buy X?") never matches and keeps the buy_sell_hold anchor below.
_BUY_LOW_RE = re.compile(
    r"\bwho (should|can|could) i buy\b"
    r"|\bbuy[ -]?low candidates?\b|\bcandidates? to buy\b"
    r"|\bwho (is|are) (a |the )?(best )?buy[ -]?low\b"
    r"|\bwho (is|are) (the )?(most )?undervalued\b"
    r"|\bbuy[ -]?lows\b")
_SELL_HIGH_RE = re.compile(
    r"\bwho (should|can|could) i sell\b"
    r"|\bsell[ -]?high candidates?\b|\bcandidates? to sell\b"
    r"|\bwho (is|are) (a |the )?(best )?sell[ -]?high\b"
    r"|\bwho (is|are) (the )?(most )?overvalued\b")


def _keyword_board(message: str, m: str) -> dict | None:
    """Pool-wide ranking phrasings -> the leaderboard intent (or None)."""
    hints = _metric_hints_in(m)
    top = re.search(r"\btop\s+(\d{1,2})\b", m)
    window = re.search(r"\blast\s+(\d{1,2})\s+weeks?\b", m)
    position = None
    for word, pos in _POSITION_WORDS:
        if word in m:
            position = pos
            break
    value_kind = None
    if _BUY_LOW_RE.search(m):
        value_kind = "buy_low"
    elif _SELL_HIGH_RE.search(m):
        value_kind = "sell_high"
    if value_kind:
        return {
            "intent": "leaderboard",
            "metric_hint": None,
            "second_metric_hint": None,
            "board_kind": value_kind,
            "top_n": int(top.group(1)) if top else None,
            "position": position,
            "team": None,
            "window_weeks": int(window.group(1)) if window else None,
            "direction": None,
        }
    kind = None
    direction = None
    if re.search(r"gives? up the most|allow(s|ed)? the most|worst defenses?|easiest defenses?", m):
        kind = "matchup"
    elif re.search(r"\brisers?\b|\brising\b|heating up|trending up", m):
        kind, direction = "movers", "risers"
    elif re.search(r"\bfallers?\b|\bfalling\b|cooling off|trending down", m):
        kind, direction = "movers", "fallers"
    elif len(hints) >= 2 and top:
        kind = "intersection"
    elif hints and (top or re.search(r"\bwho (leads|has)\b|\bleaderboard\b|\branked\b|\bmost\b.{0,30}\bshare\b", m)):
        kind = "board"
    if kind is None:
        return None
    return {
        "intent": "leaderboard",
        "metric_hint": hints[0] if hints else None,
        "second_metric_hint": hints[1] if len(hints) > 1 else None,
        "board_kind": kind,
        "top_n": int(top.group(1)) if top else None,
        "position": position,
        "team": None,
        "window_weeks": int(window.group(1)) if window else None,
        "direction": direction,
    }


# Finding vs evaluating, for the keyword fallback: finding asks the
# assistant to go find a deal; evaluating judges a deal already on the table.
_FINDING_RE = re.compile(
    r"\bfind (me |us )?(a |the )?trade\b|\bfind me\b"
    r"|\bwhat (can|could|would) i get for\b"
    r"|\bwho (should|can|could) i target\b|\bwho to target\b"
    r"|\bwho (should|can|could) i trade for\b"
    r"|\btrade targets?\b|\bgettable\b"
    r"|\btrade for (a |an |the )?(qb|rb|wr|te|quarterback|running ?back|wide ?receiver|wideout|tight ?end)\b"
    r"|\bi need (a |an |the )?(qb|rb|wr|te|quarterback|running ?back|wide ?receiver|wideout|tight ?end)\b")

_POSITION_PATTERNS = (
    (r"\bquarterback\b|\bqb\b", "QB"),
    (r"\brunning ?back\b|\brb\b", "RB"),
    (r"\bwide ?receiver\b|\bwideout\b|\bwr\b", "WR"),
    (r"\btight ?end\b|\bte\b", "TE"),
)


def _target_position(m: str):
    for pattern, pos in _POSITION_PATTERNS:
        if re.search(pattern, m):
            return pos
    return None

# Outcome-of-the-game questions: whether the user wins or loses this
# week's matchup. They outrank the this_week catch-all below ("this week"
# appears in most of them), so the matchup rule checks this pattern too.
# Plan/action questions carry no win/lose outcome anchor and still land
# in this_week; playoff-chances questions are claimed by the playoff rule
# above matchup, and add/drop questions keep their waiver anchor because
# no pattern here matches a bare "to win".
_OUTCOME_RE = re.compile(
    r"\bwill i (win|lose|beat)\b"
    r"|\bam i (going to |gonna )?(win|lose)\b"
    r"|\bam i projected to (win|lose)\b|\bprojected to (win|lose)\b"
    r"|\bdo i (win|lose|beat)\b|\bcan i (win|beat)\b"
    r"|\bwhat are my chances\b|\bmy chances of winning\b"
    # "who wins the game" is real-NFL trivia (out of scope per the SYSTEM
    # prompt); only the user's own game/matchup is a matchup question.
    r"|\bwho wins (my|our) (game|matchup)\b"
    r"|\bam i favored\b|\bam i the favorite\b")

# Trade-deadline date questions: they mention a trade but ask for a date,
# which no evidence pack can answer. Narrow on purpose (a date-seeking
# shape) so deal questions that merely mention the deadline still route
# to their trade intent.
_DEADLINE_RE = re.compile(
    r"\bwhen is (the |our |my )?trade deadline\b"
    r"|\bwhat (date|day) is (the |our |my )?trade deadline\b"
    r"|\bwhat is (the |our |my )?trade deadline\b"
    r"|\btrade deadline (date|day)\b")

# Player fallthrough for the keyword fallback, the last resort before
# out_of_scope: bare player questions ("How is Puka Nacua doing?") carry
# no intent anchor, and declining them never reaches entity resolution.
# A proper-name span in the raw message is the signal: two or more
# capitalized tokens in a row, or one capitalized token in the name slot
# after did/is/was/about ("did Chase score"). Calendar words and
# sentence-initial filler never count.
# Internal capitals ride along (McCaffrey, McBride, DeVonta): a bare
# [A-Z][a-z]+ token stops at the second capital and loses the name.
_NAME_TOKEN = r"[A-Z][a-z]+(?:[A-Z][a-z]+)?(?:'[A-Za-z]+)?"
_PAIR_STARTERS = {"How", "What", "Who", "Whom", "Is", "Are", "Was", "Were",
                  "Did", "Does", "Do", "Tell", "Show", "Give", "The", "This",
                  "That", "When", "Why", "Can", "Could", "Should", "Would",
                  "Has", "Have", "Will", "Am", "If", "My"}
# A name pair may not START with a question word: in "Is Josh Allen
# healthy?" the pair is "Josh Allen", not "Is Josh Allen".
_PROPER_NAME_RE = re.compile(
    rf"\b(?!(?:{'|'.join(sorted(_PAIR_STARTERS))})\b)"
    rf"{_NAME_TOKEN}(?:\s+(?:{_NAME_TOKEN}|Jr\.|Sr\.))+\b")
_SINGLE_NAME_RE = re.compile(rf"\b(?:did|is|was|about)\s+({_NAME_TOKEN})\b")
_NAME_STOPWORDS = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday",
                   "Friday", "Saturday", "Week", "Season"}
_QUESTION_SHAPE_RE = re.compile(
    r"^\s*(how|what|who|whom|is|are|was|were|did|does|do|should|can|could|"
    r"would|tell|show|give|has|have)\b", re.IGNORECASE)


def _player_name_in(message: str) -> str | None:
    """The name-like span in a message, or None. Runs on the raw message:
    capitalization is the signal, so the lowered copy cannot be used."""
    text = message or ""
    pair = _PROPER_NAME_RE.search(text)
    if pair and pair.group(0).split()[0] not in _NAME_STOPWORDS:
        return pair.group(0).removesuffix("'s")
    single = _SINGLE_NAME_RE.search(text)
    if single and single.group(1) not in _NAME_STOPWORDS:
        return single.group(1).removesuffix("'s")
    return None


def _keyword_route(message: str) -> dict:
    m = (message or "").lower()
    base = {"player_names": [], "give_player_names": [], "receive_player_names": [], "picks": [],
            "give_picks": [], "receive_picks": [],
            "other_team_hint": None, "team_names": [], "metric_hint": None, "target_position": None,
            "position_hint": None, "teams_mode": None, "week": None,
            "draft_year": None, "position": None,
            "replace_mode": None, "waiver_mode": None,
            "value_in_picks": None, "shop_pool": None,
            "preference_text": None, **_BOARD_ENTITY_DEFAULTS}
    if re.search(r"\bremember\b|\bdon'?t forget\b|\bkeep in mind\b", m):
        return {**base, "intent": "remember", "preference_text": message.strip()}
    if re.search(r"across (all )?my leagues|all my leagues|my other leagues|which of my teams|in my leagues\b", m):
        return {**base, "intent": "cross_league"}
    if re.search(r"watch ?list|starred players|players? i'?m watching|players? i starred", m):
        return {**base, "intent": "watchlist"}
    if re.search(r"front office|\bgm (report|memo)\b|general manager", m):
        return {**base, "intent": "front_office"}
    # League-context intents run before the player/team-week rules below:
    # their phrasings overlap ("this week", "my team", "who leads"), so the
    # specific subject wins. Disambiguation: power rankings need the word
    # power (or team-to-beat / strongest), standings own the record table
    # (standings, division, points for, place), and a matchup mention beats
    # the generic this-week digest.
    if re.search(r"playoff stretch", m):
        # "Playoff stretch" names the schedule window (weeks 15-17), not the
        # odds question; it is the one phrase that outranks the playoff rule.
        return {**base, "intent": "schedule"}
    if re.search(r"playoff|postseason|clinch|eliminat", m):
        return {**base, "intent": "playoffs"}
    if re.search(r"power ranking|power rank|team to beat|strongest team", m):
        return {**base, "intent": "power_rankings"}
    if re.search(r"standings|division|most points|points leader|highest scoring|"
                 r"\bseed\b|first place|what place am i", m):
        return {**base, "intent": "standings"}
    if re.search(r"\bmatchup\b|who wins between|win probability|win prob\b", m) or _OUTCOME_RE.search(m):
        return {**base, "intent": "matchup"}
    if re.search(r"remaining schedule|schedule strength|easiest schedule|hardest schedule|"
                 r"toughest schedule|playoff stretch|my schedule|league schedule", m):
        return {**base, "intent": "schedule"}
    # Pick ownership before trades: "who owns my 2027 first" names no deal.
    if re.search(r"who owns|whose pick|who has my .*pick|own my .*pick", m):
        return {**base, "intent": "pick_ownership"}
    # Breakout family: track record (accuracy) vs one player's call history
    # (past-tense framing) vs the board itself.
    if "breakout" in m:
        if re.search(r"track record|how good|how accurate|hit rate|record on", m):
            return {**base, "intent": "breakout_track_record"}
        if re.search(r"\bwas\b|\bdid\b|has .* been|call for|call on", m):
            return {**base, "intent": "breakout_history"}
        return {**base, "intent": "breakout_board"}
    # Waiver family: adds, drops, stashes, streamers, and the free-agent
    # pool itself. The mode rides along so the pack can rank for the ask
    # (a stash is not an add, a drop question wants the drop ranking, a
    # streamer is ranked by this week's projection, not season value).
    wmode = None
    if re.search(r"\bstash", m):
        wmode = "stash"
    elif re.search(r"\bstream", m):
        wmode = "stream"
    elif re.search(r"\bmost droppable\b|\bdroppable\b", m):
        wmode = "drop"
    if (re.search(r"\bwaiver|\badd\b|\bdrop\b|\bfaab\b|\bpick up\b|\bfree agents?\b", m)
            or wmode is not None):
        if wmode is None and re.search(r"\bdrop\b|\bcut\b", m):
            wmode = "drop"
        return {**base, "intent": "waivers", "waiver_mode": wmode,
                "position": _target_position(m)}
    # League activity: what already happened (completed trades, made waiver
    # moves, injury headlines). Feed or past-tense framing is required, so
    # advice questions ("what waiver moves should I make") keep the intents
    # below.
    if re.search(r"league activity|activity feed|\btransactions?\b|what happened in my league|"
                 r"what'?s happening in my league|what trades\b|who traded\b|recent trades\b|"
                 r"latest trade\b|trades? (went down|gone down|go down|happened)|"
                 r"(moves|claims) (went down|happened|so far)\b", m):
        return {**base, "intent": "activity"}
    board = _keyword_board(message, m)
    if board is not None:
        return {**base, **board}
    # Player-anchored succession questions (checked before start/sit and
    # the trade anchors: "if X is out, who do I start" carries a start
    # anchor but asks who is next up, and "a cheaper version of X" is not
    # a deal). The name must be present; without one these fall through.
    subj_name = _player_name_in(message)
    if subj_name:
        if (re.search(r"\bhandcuff\b|\bnext (up|man up)\b|\bbenefits if\b"
                      r"|\bdepth chart\b|\bwho (steps up|takes over|replaces)\b", m)
                or re.search(r"\bif\b.+\b(gets? hurt|is hurt|gets? injured|"
                             r"is injured|is out|misses time)\b", m)):
            return {**base, "intent": "handcuff", "player_names": [subj_name]}
        rmode = None
        if re.search(r"\bcheaper\b|\bcheapest\b|\bbudget version\b", m):
            rmode = "cheaper"
        elif re.search(r"\bolder\b|\bwin[- ]now\b|\bveteran version\b", m):
            rmode = "older"
        elif re.search(r"\byounger\b|\blong[- ]term\b|\breplacement\b|"
                       r"\bversion of\b", m):
            rmode = "younger"
        if rmode is not None:
            return {**base, "intent": "replacement",
                    "player_names": [subj_name], "replace_mode": rmode}
    if re.search(r"\bstart\b|\bsit\b|\bflex\b|\bbench\b|who (should|do) i play", m):
        return {**base, "intent": "start_sit"}
    if _FINDING_RE.search(m):
        return {**base, "intent": "trade_finding", "target_position": _target_position(m)}
    if _DEADLINE_RE.search(m):
        # "When is the trade deadline?" asks for a date, not a deal: the
        # trade_eval pack needs two sides and cannot answer it. Placed
        # after the finding rule so "target before the trade deadline"
        # still finds a trade.
        return {**base, "intent": "out_of_scope"}
    # Worth-in-picks: a named player plus a pick-value phrasing is a
    # finding question (what his value buys in picks), not an evaluation
    # of a deal and not a metric. Placed before the trade anchor so
    # "what picks could I get for X" is not read as trade_eval, and the
    # metric tail below never sees "what is X worth in picks".
    pick_name = _player_name_in(message)
    if (pick_name and re.search(r"\bpicks?\b", m)
            and re.search(r"\bworth\b|\bget for\b|\bcould i get\b|\bcan i get\b", m)):
        return {**base, "intent": "trade_finding",
                "player_names": [pick_name], "value_in_picks": True}
    # Shop-the-pool: who of mine should move, with nobody named.
    if re.search(r"\bwho (should|can|could) i (trade away|shop|move|sell)\b"
                 r"|\btrade away\b", m):
        return {**base, "intent": "trade_finding", "shop_pool": True}
    if re.search(r"\btrad(?:e|es|ed|ing)\b|\bswap\b"
                 r"|\bshould i accept\b|\baccept (this|the) (trade|deal|offer)\b", m):
        return {**base, "intent": "trade_eval"}
    if re.search(r"\bbuy\b|\bsell\b|\bhold\b", m):
        return {**base, "intent": "buy_sell_hold"}
    # League-wide roster strength/grades, before the this_week catch below
    # ("grade my team's WR room" is a Teams question, not a digest ask).
    if re.search(r"strongest roster|strongest team|roster is strongest|team is strongest|"
                 r"weakest at|weakest team|grade every|grade all|"
                 r"grade my team|position room|\b(qb|rb|wr|te) room\b|roster grades|team grades", m):
        pos = None
        pm = re.search(r"\b(qb|rb|wr|te)\b", m)
        if pm:
            pos = pm.group(1).upper()
        mode = "this_season" if re.search(r"this season|right now|this year", m) else None
        return {**base, "intent": "teams", "position_hint": pos, "teams_mode": mode}
    if re.search(r"\bkeepers?\b|who should i keep|keep .* (next year|next season)|\bkeep\b.*\bor the\b", m):
        return {**base, "intent": "keeper"}
    if re.search(r"\brookies?\b|\bprospects?\b|\bdraft class\b|\b(202[4-9]) class\b|\bpre[- ]draft\b", m):
        year = re.search(r"\b(202[4-9])\b", m)
        pos = re.search(r"\b(qb|rb|wr|te)s?\b", m)
        return {**base, "intent": "rookie",
                "draft_year": int(year.group(1)) if year else None,
                "position": pos.group(1).upper() if pos else None}
    if re.search(r"what should i do|this week|my team", m):
        return {**base, "intent": "this_week"}
    if re.search(r"target share|snap share|route|metric|what is|who leads", m):
        hint = None
        for cand in ("target share", "snap share", "route rate", "opportunity share"):
            if cand in m:
                hint = cand
                break
        return {**base, "intent": "metric", "metric_hint": hint}
    # Player fallthrough: a question-shaped message carrying a player
    # name belongs to the player intent even with no other anchor. The
    # extracted name rides along so the resolver can confirm it (or ask
    # which player) instead of the question being declined unread.
    name = _player_name_in(message)
    if name and ("?" in (message or "") or _QUESTION_SHAPE_RE.search(message or "")
                 or len((message or "").split()) <= 6):
        return {**base, "intent": "player", "player_names": [name]}
    return {**base, "intent": "out_of_scope"}


def classify(question: str, context: dict | None = None, history: list | None = None) -> dict:
    """Return {intent, entities{...}, preference_text}."""
    payload = {
        "question": question,
        "context": {k: (context or {}).get(k) for k in ("league_name", "team_name", "week", "season")},
        "history": (history or [])[-4:],
    }
    try:
        data = call_model_json(SYSTEM, json.dumps(payload, default=str), SCHEMA, "assistant_router")
        intent = data.get("intent") if data.get("intent") in INTENTS else "out_of_scope"
        entities = {
            "player_names": data.get("player_names") or [],
            "give_player_names": data.get("give_player_names") or [],
            "receive_player_names": data.get("receive_player_names") or [],
            "picks": data.get("picks") or [],
            "give_picks": data.get("give_picks") or [],
            "receive_picks": data.get("receive_picks") or [],
            "other_team_hint": data.get("other_team_hint"),
            "team_names": data.get("team_names") or [],
            "metric_hint": data.get("metric_hint"),
            "target_position": data.get("target_position"),
            "position_hint": data.get("position_hint"),
            "teams_mode": data.get("teams_mode"),
            "week": data.get("week"),
            "draft_year": data.get("draft_year"),
            "board_kind": data.get("board_kind"),
            "second_metric_hint": data.get("second_metric_hint"),
            "top_n": data.get("top_n"),
            "position": data.get("position"),
            "team": data.get("team"),
            "window_weeks": data.get("window_weeks"),
            "direction": data.get("direction"),
            "replace_mode": data.get("replace_mode"),
            "waiver_mode": data.get("waiver_mode"),
            "value_in_picks": data.get("value_in_picks"),
            "shop_pool": data.get("shop_pool"),
        }
        return {"intent": intent, "entities": entities, "preference_text": data.get("preference_text"), "router_model": "model", "prompt_version": ROUTER_PROMPT_VERSION}
    except Exception as exc:  # model down (rate limit / unavailable): keyword fallback
        logger.debug("[assistant] router model failed, keyword fallback: %s", exc)
        routed = _keyword_route(question)
        intent = routed.pop("intent")
        pref = routed.pop("preference_text")
        return {"intent": intent, "entities": routed, "preference_text": pref, "router_model": "keyword-fallback", "prompt_version": ROUTER_PROMPT_VERSION}
