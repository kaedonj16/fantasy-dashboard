"""Outcome graders for the BR Assistant (workshop stage 2, section 3).

Grades are written onto the advice log rows themselves (grade, grader,
graded_at, outcome_detail), never by the bot. Each grader is versioned;
a grading change bumps the version string instead of rewriting history.

Grader definitions (workshop section 3):

- Start/sit, "start-sit-outcome-v1": graded after the answer's week has
  finalized (a later NFL week has begun, per the site's NFL state). The
  recommended starter's actual points for that week, from the Sleeper
  weekly stats the start/sit engine consumes, scored under the league's
  own settings with the site's week_stat_points, vs the best benched
  alternative stored in the answer's evidence pack.
  Hit: the pick scored at least as much as the best alternative.
  Partial: the best alternative outscored the pick by no more than the
  pack's own close-call margin (10 percent of the alternative's points,
  with a 1.0 point floor): the coin flip landed the other way.
  Miss: the alternative outscored the pick by more than that margin.

- Buy/sell/hold, "value-drift-v1": graded at the 4-week mark (the doc's
  3 to 4 week horizon, taken at 4). The player's value then vs now, both
  read from the site's value history (the data behind
  /api/player-value-history/<id>): the last snapshot on or before the
  answer date, and the latest snapshot. Bands mirror the thresholds the
  evidence pack itself uses to synthesize its buy/sell/hold signal
  (plus/minus 5 percent): a buy hits at +5 percent or better and misses
  at -5 percent or worse, a sell mirrored, a hold hits inside plus/minus
  5 percent, is partial out to plus/minus 15 percent, and misses beyond.
  The player's PPG change over the same span is recorded in
  outcome_detail alongside, as the doc's second signal.

Rows that cannot be judged stay ungraded, never a miss: an unfinalized
week, a pack with no pick or no stored alternative, a player with no
value snapshot on or before the answer date. Graders only ever write
through store.write_grade, whose grade IS NULL guard makes every run
idempotent.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

from dashboard_services.assistant import store

logger = logging.getLogger(__name__)

START_SIT_GRADER = "start-sit-outcome-v1"
VALUE_DRIFT_GRADER = "value-drift-v1"

# The pack's own close-call definition (build_start_sit: gap/base < 0.10
# is "thin"); a lost coin flip inside that margin grades partial.
PARTIAL_MARGIN_FRACTION = 0.10
PARTIAL_MARGIN_FLOOR = 1.0

# The pack's signal thresholds (build_buy_sell_hold: buy at >= +5 percent
# 30-day drift, sell at <= -5 percent); the drift bands reuse them.
DRIFT_BAND_PCT = 5.0
HOLD_PARTIAL_PCT = 15.0

VALUE_DRIFT_MIN_AGE_DAYS = 28  # the doc's 4-week mark


# -- lazy shims (monkeypatchable in tests, same pattern as evidence.py) -------

def load_sleeper_week_stats(*args, **kwargs):
    from utils.utils import load_sleeper_week_stats as _fn
    return _fn(*args, **kwargs)


def week_stat_points(*args, **kwargs):
    from utils.projections import week_stat_points as _fn
    return _fn(*args, **kwargs)


def get_league_ctx_from_cache(*args, **kwargs):
    from app import get_league_ctx_from_cache as _fn
    return _fn(*args, **kwargs)


def get_nfl_state(*args, **kwargs):
    from dashboard_services.api import get_nfl_state as _fn
    return _fn(*args, **kwargs)


def get_player_value_history(*args, **kwargs):
    from dashboard_services.player_value_history import get_player_value_history as _fn
    return _fn(*args, **kwargs)


def get_player_metrics(*args, **kwargs):
    from data_building.advanced_metrics import get_player_metrics as _fn
    return _fn(*args, **kwargs)


# -- pure banding ---------------------------------------------------------------

def classify_start_sit(pick_points: float, best_alt_points: float) -> str:
    """hit / partial / miss for a graded start/sit comparison."""
    delta = pick_points - best_alt_points
    if delta >= 0:
        return "hit"
    margin = max(PARTIAL_MARGIN_FLOOR, PARTIAL_MARGIN_FRACTION * best_alt_points)
    return "partial" if -delta <= margin else "miss"


def classify_value_drift(call: str, pct_change: float) -> str:
    """hit / partial / miss for a buy/sell/hold call vs its value drift."""
    if call == "buy":
        if pct_change >= DRIFT_BAND_PCT:
            return "hit"
        return "miss" if pct_change <= -DRIFT_BAND_PCT else "partial"
    if call == "sell":
        if pct_change <= -DRIFT_BAND_PCT:
            return "hit"
        return "miss" if pct_change >= DRIFT_BAND_PCT else "partial"
    # hold: the call was that nothing much changes
    mag = abs(pct_change)
    if mag <= DRIFT_BAND_PCT:
        return "hit"
    return "partial" if mag <= HOLD_PARTIAL_PCT else "miss"


# -- shared helpers ---------------------------------------------------------------

def _as_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if hasattr(value, "isoformat"):  # a date
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except Exception:
        return None


def _as_datetime(value):
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _pack(row) -> dict:
    pack = row.get("evidence_pack")
    return pack if isinstance(pack, dict) else {}


def _week_finalized(season, week, state) -> bool:
    """True once a later NFL week has begun (the doc: grade on Tuesday,
    after the week finalizes; the state check is the enforceable form)."""
    if not season or not week or not isinstance(state, dict):
        return False
    try:
        cur_season = int(state.get("season") or 0)
        cur_week = int(state.get("week") or state.get("display_week") or 0)
        season, week = int(season), int(week)
    except (TypeError, ValueError):
        return False
    if not cur_season or not cur_week:
        return False
    if season != cur_season:
        return season < cur_season
    return week < cur_week


def _league_scoring(row, ctx_cache: dict) -> dict:
    """The league's scoring settings from its cached league ctx (the same
    source build_start_sit reads via evidence._scoring). Empty dict when
    the ctx cannot load: week_stat_points then falls back to the stat
    line's own published totals, and the grade records that it did."""
    key = (row.get("platform") or "sleeper", str(row.get("league_id") or ""), row.get("season"))
    if key in ctx_cache:
        return ctx_cache[key]
    scoring: dict = {}
    if row.get("league_id"):
        try:
            ctx = get_league_ctx_from_cache(key[0], key[1], key[2]) or {}
            if isinstance(ctx, dict):
                scoring = ctx.get("raw_scoring_settings") or ctx.get("scoring_settings") or {}
        except Exception:
            logger.debug("[assistant] grading: league ctx lookup failed", exc_info=True)
            scoring = {}
    ctx_cache[key] = scoring if isinstance(scoring, dict) else {}
    return ctx_cache[key]


# -- start / sit ------------------------------------------------------------------

def grade_start_sit_row(row, *, state=None, ctx_cache=None) -> tuple[str, dict] | None:
    """Grade one start_sit log row, or None when it is not due or not
    gradeable (unfinalized week, empty stats for the week, no pick or no
    stored alternative in the pack)."""
    pack = _pack(row)
    season = row.get("season") or (pack.get("league") or {}).get("season")
    week = row.get("week") or pack.get("week")
    if state is None:
        try:
            state = get_nfl_state()
        except Exception:
            state = None
    if not _week_finalized(season, week, state):
        return None
    players = [p for p in (pack.get("players") or []) if isinstance(p, dict) and p.get("player_id")]
    pick_id = str(pack.get("pick_player_id") or "")
    pick = next((p for p in players if str(p.get("player_id")) == pick_id), None)
    alternatives = [p for p in players if str(p.get("player_id")) != pick_id]
    if pick is None or not alternatives:
        return None
    try:
        stats = load_sleeper_week_stats(int(season), int(week)) or {}
    except Exception:
        return None
    if not isinstance(stats, dict) or not stats:
        return None  # the week's stats are not loaded yet: not finalized here
    scoring = _league_scoring(row, ctx_cache if ctx_cache is not None else {})

    def _points(p):
        line = stats.get(str(p.get("player_id")))
        if not isinstance(line, dict):
            return 0.0  # no stat line in a loaded week: the player did not play
        try:
            return round(float(week_stat_points(line, scoring, p.get("position") or "")), 1)
        except Exception:
            return 0.0

    pick_points = _points(pick)
    alt_lines = [
        {"player_id": str(p.get("player_id")), "name": p.get("name") or str(p.get("player_id")),
         "points": _points(p)}
        for p in alternatives
    ]
    best_alt = max(alt_lines, key=lambda a: a["points"])
    grade = classify_start_sit(pick_points, best_alt["points"])
    detail = {
        "season": int(season), "week": int(week),
        "pick": {"player_id": pick_id, "name": pick.get("name") or pick_id, "points": pick_points},
        "best_alternative": best_alt,
        "alternatives": alt_lines,
        "delta": round(pick_points - best_alt["points"], 1),
        "league_scoring": bool(scoring),
    }
    return grade, detail


# -- buy / sell / hold --------------------------------------------------------------

_CALL_RE = re.compile(r"\b(buy|sell|hold)\b", re.IGNORECASE)


def _drift_call(row, pack) -> str | None:
    signal = str(pack.get("signal") or "").strip().lower()
    if signal in ("buy", "sell", "hold"):
        return signal
    verdict = row.get("verdict")
    call_text = ""
    if isinstance(verdict, dict):
        call_text = str(verdict.get("call") or "")
    elif isinstance(verdict, str):
        call_text = verdict
    m = _CALL_RE.search(call_text)
    return m.group(1).lower() if m else None


def grade_buy_sell_row(row, *, now=None) -> tuple[str, dict] | None:
    """Grade one buy_sell_hold log row at the 4-week mark, or None when it
    is not due or not gradeable (no determinable call, no value baseline
    on or before the answer date, no current value)."""
    now = now or datetime.now(timezone.utc)
    created = _as_datetime(row.get("created_at"))
    if created is None or now - created < timedelta(days=VALUE_DRIFT_MIN_AGE_DAYS):
        return None
    pack = _pack(row)
    player = pack.get("player") if isinstance(pack.get("player"), dict) else {}
    pid = player.get("player_id") or player.get("id")
    if not pid:
        return None
    call = _drift_call(row, pack)
    if call is None:
        return None
    answer_date = created.date()
    days_back = max((now.date() - answer_date).days + 10, VALUE_DRIFT_MIN_AGE_DAYS + 7)
    try:
        history = get_player_value_history(str(pid), days=days_back) or []
    except Exception:
        return None
    points = []
    for r in history:
        if not isinstance(r, dict):
            continue
        d = _as_date(r.get("as_of_date") or r.get("date"))
        v = r.get("value")
        if d is None or v is None:
            continue
        try:
            points.append((d, float(v)))
        except (TypeError, ValueError):
            continue
    if not points:
        return None  # no value history at all: ungraded, not a miss
    points.sort(key=lambda p: p[0])
    baseline = None
    for d, v in points:
        if d <= answer_date:
            baseline = (d, v)
        else:
            break
    if baseline is None or not baseline[1]:
        return None  # nothing recorded on or before the answer: no baseline
    current_date, current_value = points[-1]
    if current_date <= answer_date:
        return None  # history has not moved past the answer yet
    pct = round((current_value - baseline[1]) / abs(baseline[1]) * 100, 1)
    grade = classify_value_drift(call, pct)
    detail = {
        "player_id": str(pid), "name": player.get("name") or str(pid),
        "call": call,
        "value_at_answer": baseline[1], "value_at_answer_date": baseline[0].isoformat(),
        "value_now": current_value, "value_now_date": current_date.isoformat(),
        "value_delta_pct": pct,
    }
    # PPG change, the doc's second signal: recorded, best effort.
    ppg_then = (pack.get("role") or {}).get("ppg") if isinstance(pack.get("role"), dict) else None
    ppg_now = None
    try:
        metrics = get_player_metrics(str(pid)) or {}
        if isinstance(metrics, dict):
            ppg_now = metrics.get("ppg")
    except Exception:
        ppg_now = None
    if ppg_then is not None and ppg_now is not None:
        try:
            detail["ppg_at_answer"] = float(ppg_then)
            detail["ppg_now"] = float(ppg_now)
            detail["ppg_delta"] = round(float(ppg_now) - float(ppg_then), 2)
        except (TypeError, ValueError):
            pass
    return grade, detail


# -- runner -------------------------------------------------------------------------

_GRADERS = (
    ("start_sit", START_SIT_GRADER),
    ("buy_sell_hold", VALUE_DRIFT_GRADER),
)


def run_graders(*, dry_run: bool = False, now=None, limit: int = 500) -> dict:
    """Grade every due, ungraded start/sit and buy/sell/hold log row.

    Returns per-intent counts: considered, graded (hit/partial/miss), and
    skipped (not due yet or not gradeable from what the row stored). With
    dry_run=True the same grades are computed and counted, nothing is
    written.
    """
    now = now or datetime.now(timezone.utc)
    state = None
    try:
        state = get_nfl_state()
    except Exception:
        state = None
    ctx_cache: dict = {}
    summary: dict = {}
    for intent, grader_name in _GRADERS:
        counts = {"considered": 0, "graded": 0, "hit": 0, "partial": 0, "miss": 0, "skipped": 0}
        for row in store.fetch_ungraded_rows(intent, limit=limit):
            counts["considered"] += 1
            if intent == "start_sit":
                result = grade_start_sit_row(row, state=state, ctx_cache=ctx_cache)
            else:
                result = grade_buy_sell_row(row, now=now)
            if result is None:
                counts["skipped"] += 1
                continue
            grade, detail = result
            if not dry_run:
                if not store.write_grade(row["log_id"], grade, grader_name, detail):
                    counts["skipped"] += 1
                    continue
            counts["graded"] += 1
            counts[grade] += 1
        summary[intent] = counts
    return summary
