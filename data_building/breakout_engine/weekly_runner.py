"""
In-season execution for weekly breakout detection.

Two responsibilities:

1. ``resolve_scoring_context`` - decide, from the NFL schedule and state (NOT the
   calendar month), whether this is a weekly (in-season) or offseason run, and
   which completed regular-season week is the as-of cutoff. Reading the schedule
   means January regular-season games are handled correctly - a naive
   "month in 9..12" rule drops week 18 when it falls in January.

2. ``run_weekly_breakout`` - refresh the weekly usage data, score every skill
   candidate from that data, persist the snapshot, and log season / mode /
   cutoff / coverage / candidates / records. When the required data cannot be
   refreshed it PRESERVES the previous snapshot and records a skipped/stale run
   rather than overwriting good results with an empty board.

The context resolver is pure given a schedule-reading callable, so it is unit
tested without disk or network.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional

MODE_WEEKLY = "weekly"
MODE_OFFSEASON = "offseason"

# Regular-season game dates in a week's schedule file look like "20250904".
_SCHEDULE_DIR = os.path.join("cache", "schedule")


# =============================================================================
# schedule reading (thin, injectable)
# =============================================================================

def _default_week_games(season: int, week: int) -> List[Dict[str, Any]]:
    """Regular-season games for a season/week from the cached schedule file.
    Returns [] when the file is missing/unreadable."""
    path = os.path.join(_SCHEDULE_DIR, f"schedule_s{season}_w{week}.json")
    try:
        with open(path, encoding="utf-8") as f:
            games = json.load(f)
    except (OSError, ValueError):
        return []
    out = []
    for g in games or []:
        st = str(g.get("seasonType") or "").lower()
        if "regular" in st or st in ("reg", "2", ""):
            out.append(g)
    return out


def _game_date(g: Dict[str, Any]) -> Optional[date]:
    raw = str(g.get("gameDate") or "").strip()
    if len(raw) == 8 and raw.isdigit():
        try:
            return datetime.strptime(raw, "%Y%m%d").date()
        except ValueError:
            return None
    return None


# =============================================================================
# context resolution
# =============================================================================

@dataclass
class ScoringContext:
    season: int
    mode: str                       # MODE_WEEKLY | MODE_OFFSEASON
    as_of_date: date
    cutoff_week: Optional[int] = None       # last fully completed regular-season week
    completed_weeks: List[int] = field(default_factory=list)
    reason: str = ""

    def to_log(self) -> str:
        return (
            f"season={self.season} mode={self.mode} "
            f"cutoff_week={self.cutoff_week} "
            f"completed_weeks={self.completed_weeks} ({self.reason})"
        )


def resolve_scoring_context(
    nfl_state: Optional[Dict[str, Any]],
    *,
    as_of_date: Optional[date] = None,
    week_games: Callable[[int, int], List[Dict[str, Any]]] = _default_week_games,
    max_week: int = 18,
) -> ScoringContext:
    """Resolve season, mode and cutoff week from schedule + state.

    A regular-season week counts as a completed cutoff candidate when it has at
    least one regular-season game and ALL of that week's regular-season games
    were played strictly before ``as_of_date`` (so a slate still in progress does
    not become the cutoff). The cutoff is the latest such week; mode is weekly
    when any completed week exists, else offseason.
    """
    state = nfl_state or {}
    as_of_date = as_of_date or date.today()
    try:
        season = int(state.get("season") or as_of_date.year)
    except (TypeError, ValueError):
        season = as_of_date.year

    completed: List[int] = []
    for wk in range(1, max_week + 1):
        games = week_games(season, wk)
        if not games:
            continue
        dates = [d for d in (_game_date(g) for g in games) if d is not None]
        if not dates:
            continue
        # Fully completed only if every scheduled game already happened.
        if max(dates) < as_of_date:
            completed.append(wk)

    if completed:
        cutoff = max(completed)
        return ScoringContext(
            season=season, mode=MODE_WEEKLY, as_of_date=as_of_date,
            cutoff_week=cutoff, completed_weeks=completed,
            reason=f"{len(completed)} completed regular-season week(s) by schedule",
        )

    return ScoringContext(
        season=season, mode=MODE_OFFSEASON, as_of_date=as_of_date,
        cutoff_week=None, completed_weeks=[],
        reason="no completed regular-season games yet",
    )


# =============================================================================
# candidate assembly + injury context (DB / feed backed, best-effort)
# =============================================================================

def _prior_baseline_map(season: int) -> Dict[str, Dict[str, Any]]:
    """Prior-season per-game usage keyed by sleeper player id, for baseline
    fallback when a player has too little current-season data. Read from the
    usage_rows_{season-1}.json cache the historical builder already writes; empty
    when absent (rookies then run provisionally)."""
    path = os.path.join("cache", "player_history", f"usage_rows_{season - 1}.json")
    try:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
    except (OSError, ValueError):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows or []:
        # Historical exports normally contain both keys, but older/backfilled
        # caches may contain only one. Both values are Sleeper player IDs.
        pid = str(row.get("id") or row.get("player_id") or "")
        u = row.get("usage") or {}
        games = float(u.get("games") or 0)
        if not pid or games <= 0:
            continue

        def _pg(total_key, avg_key):
            avg = u.get(avg_key)
            if avg not in (None, ""):
                try:
                    return float(avg)
                except (TypeError, ValueError):
                    pass
            tot = u.get(total_key)
            if tot not in (None, "") and games > 0:
                try:
                    return float(tot) / games
                except (TypeError, ValueError):
                    pass
            return None

        def _observed_value(key, *, evidence_keys=()):
            """Reject provider-shaped zero defaults unless corroborated.

            nflverse history rows contain every usage key, frequently filled
            with zero even when that statistic was not collected. Positive
            values are observed; zero is observed only when another raw field
            proves the player participated in the relevant phase.
            """
            value = u.get(key)
            if value in (None, ""):
                return None
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            if value != 0:
                return value
            for evidence_key in evidence_keys:
                evidence = u.get(evidence_key)
                try:
                    if evidence not in (None, "") and float(evidence) > 0:
                        return 0.0
                except (TypeError, ValueError):
                    continue
            return None

        # A zero snap percentage alongside positive offensive snaps is
        # contradictory and is a known provider default, not an observed zero.
        snap = _observed_value("snap_share")
        if snap is None:
            snap = _observed_value("avg_off_snap_pct")
        try:
            snap = float(snap) * 100.0 if snap is not None and float(snap) <= 1.0 else (
                float(snap) if snap is not None else None)
        except (TypeError, ValueError):
            snap = None
        ts = _observed_value(
            "target_share", evidence_keys=("total_targets", "targets", "avg_targets"))
        try:
            ts = float(ts) * 100.0 if ts is not None and float(ts) <= 1.0 else (
                float(ts) if ts is not None else None)
        except (TypeError, ValueError):
            ts = None
        baseline = {
            "snap_pct": snap,
            "target_share": ts,
            "targets_pg": _observed_value(
                "avg_targets", evidence_keys=("total_targets", "targets", "avg_off_snaps")),
            "carries_pg": _observed_value(
                "avg_carries", evidence_keys=("carries", "avg_off_snaps")),
            "pass_att_pg": _observed_value(
                "avg_pass_att", evidence_keys=("pass_attempts", "avg_off_snaps")),
        }
        baseline["usable_signals"] = [k for k, v in baseline.items() if v is not None]
        baseline["history_status"] = (
            "usable" if len(baseline["usable_signals"]) >= 2 else
            "partial" if baseline["usable_signals"] else "provider_defaults_only"
        )
        baseline["games"] = int(games)
        out[pid] = baseline
    return out


def _injury_context_map(full_players: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Best-effort {player_id: {vacated, source}} from the live Sleeper players
    feed: a player is treated as having an opening when the depth-chart slot
    directly ahead of him on the same team/position is injured (IR/Out/Doubtful).
    Mirrors the waiver/breakout injury-vacancy signal. Empty when no feed.
    """
    if not full_players:
        return {}
    by_team_pos: Dict[tuple, List[Dict[str, Any]]] = {}
    for pid, p in full_players.items():
        if not isinstance(p, dict):
            continue
        pos = (p.get("position") or "").upper()
        team = p.get("team")
        order = p.get("depth_chart_order")
        if pos not in ("QB", "RB", "WR", "TE") or not team or order is None:
            continue
        try:
            order = int(order)
        except (TypeError, ValueError):
            continue
        by_team_pos.setdefault((team, pos), []).append({
            "pid": str(pid),
            "order": order,
            "name": p.get("full_name") or p.get("last_name") or str(pid),
            "injury": (p.get("injury_status") or "").strip(),
        })
    out: Dict[str, Dict[str, Any]] = {}
    _OUT = {"ir", "out", "doubtful", "pup", "sus"}
    for (team, pos), players in by_team_pos.items():
        players.sort(key=lambda x: x["order"])
        for i, pl in enumerate(players):
            # Is anyone ahead of this player injured?
            for ahead in players[:i]:
                if ahead["injury"].lower() in _OUT:
                    out[pl["pid"]] = {
                        "vacated": True,
                        "source": f"{ahead['name']} ({ahead['injury']})",
                    }
                    break
    return out


# =============================================================================
# orchestration
# =============================================================================

def derive_lifecycle(result: Dict[str, Any], previous: Optional[Dict[str, Any]],
                     cutoff_week: int, watchlist_min: float) -> Dict[str, Any]:
    """Deterministic lifecycle transition, separated for replay/tests."""
    previous_score = float(previous.get("breakout_score") or 0) if previous else None
    previous_evidence = (previous or {}).get("evidence") or {}
    if isinstance(previous_evidence, str):
        try:
            previous_evidence = json.loads(previous_evidence)
        except ValueError:
            previous_evidence = {}
    previous_lifecycle = previous_evidence.get("lifecycle") or {}
    flagged = float(result.get("breakout_score") or 0) >= watchlist_min
    prior_streak = int(previous_lifecycle.get("consecutive_flagged_weeks") or 0)
    score_change = (round(float(result["breakout_score"]) - previous_score, 1)
                    if previous_score is not None else None)
    if previous is None:
        state = "new"
    elif not flagged and previous_score >= watchlist_min:
        state = ("graduated" if prior_streak >= 4 and
                 float(result.get("current_role_score") or 0) >= 50 else "invalidated")
    elif score_change is not None and score_change <= -8:
        state = "cooling"
    elif prior_streak >= 2 and result.get("classification") == "emerging_breakout":
        state = "confirmed"
    elif flagged:
        state = "rising"
    else:
        state = "graduated"
    first_week = previous_lifecycle.get("first_detected_week")
    if first_week is None:
        first_week = (previous or {}).get("as_of_week", cutoff_week)
    return {"previous_score": previous_score, "score_change": score_change,
            "first_detected_week": first_week,
            "consecutive_flagged_weeks": prior_streak + 1 if flagged else 0,
            "lifecycle_state": state}

def run_weekly_breakout(
    context: ScoringContext,
    *,
    refresh: bool = True,
    min_score: float = 0.0,
) -> Dict[str, Any]:
    """Score and persist the weekly breakout board for ``context``.

    Preserves the previous snapshot (records a skipped/stale run, writes nothing)
    when the required weekly data can't be refreshed or is empty. Returns a
    summary dict for logging.
    """
    # Local imports keep this module importable in the pure test suite; only the
    # actual run touches the DB / feeds.
    from data_building.weekly_metrics import build_weekly_metrics, get_player_weekly_series
    from data_building.breakout_engine.weekly_breakout import (
        score_player, SCORING_VERSION, WATCHLIST_MIN_SCORE,
    )
    from data_building.breakout_engine import weekly_store
    from utils.utils import load_players_index

    season = context.season
    cutoff = context.cutoff_week
    summary: Dict[str, Any] = {
        "season": season, "mode": context.mode, "cutoff_week": cutoff,
        "scoring_version": SCORING_VERSION, "as_of_date": context.as_of_date.isoformat(),
    }

    if context.mode != MODE_WEEKLY or cutoff is None:
        summary["status"] = "skipped"
        summary["reason"] = "not an in-season context"
        return summary

    # ── refresh the weekly usage data up to the cutoff ───────────────────────
    weeks_covered = 0
    if refresh:
        try:
            # Incremental: fills any missing weeks and rebuilds the latest two to
            # pick up stat corrections. Cheaper than refetching every week, and it
            # ensures all weeks 1..cutoff are present for the windows.
            build_weekly_metrics(season)
            weeks_covered = cutoff
        except Exception as exc:  # noqa: BLE001 - refresh failure must not wipe results
            print(f"[weekly_breakout] data refresh failed: {exc}; preserving last snapshot")
            weekly_store.record_run(
                season, cutoff, mode=context.mode, status="skipped",
                weeks_covered=0, detail={**summary, "error": str(exc)},
                as_of_date=context.as_of_date,
            )
            summary["status"] = "skipped"
            summary["reason"] = f"refresh failed: {exc}"
            return summary

    # ── candidate universe: skill players on active rosters ──────────────────
    players_index = load_players_index() or {}
    prior = _prior_baseline_map(season)
    try:
        previous_scores = weekly_store.load_previous_week_scores(season, cutoff)
    except Exception:
        previous_scores = {}

    full_players: Dict[str, Any] = {}
    try:
        from dashboard_services.api import get_nfl_players
        full_players = get_nfl_players() or {}
    except Exception:
        full_players = {}
    injuries = _injury_context_map(full_players)

    results: List[Dict[str, Any]] = []
    scanned = 0
    baseline_counts = {"current_season": 0, "prior_season": 0, "none": 0}
    for pid, meta in players_index.items():
        pos = (meta.get("pos") or meta.get("position") or "").upper()
        team = meta.get("team")
        if pos not in ("QB", "RB", "WR", "TE") or not team:
            continue
        try:
            series = get_player_weekly_series(str(pid), season)
        except Exception:
            continue
        if not series:
            continue
        scanned += 1
        player = {
            "player_id": str(pid),
            "player_name": meta.get("name") or meta.get("full_name"),
            "team": team,
            "position": pos,
            "season": season,
            "years_exp": (full_players.get(str(pid)) or {}).get("years_exp", meta.get("years_exp")),
            "rookie_year": (full_players.get(str(pid)) or {}).get("rookie_year", meta.get("rookie_year")),
            "draft_year": ((full_players.get(str(pid)) or {}).get("draft_year") or
                           meta.get("draft_year") or meta.get("draft_yr")),
            "draft_round": ((full_players.get(str(pid)) or {}).get("draft_round") or
                            meta.get("draft_round")),
            "depth_chart_order": (full_players.get(str(pid)) or {}).get("depth_chart_order"),
        }
        res = score_player(
            player, series,
            prior_baseline=prior.get(str(pid)),
            injury_context=injuries.get(str(pid)),
            cutoff_week=cutoff,
        )
        source = res.get("baseline_source") or "none"
        baseline_counts[source] = baseline_counts.get(source, 0) + 1
        previous = previous_scores.get(str(pid))
        res["previous_breakout_status"] = ((previous or {}).get("classification")
                                           if previous else None)
        lifecycle = derive_lifecycle(res, previous, cutoff, WATCHLIST_MIN_SCORE)
        res["lifecycle"] = lifecycle
        res.update(lifecycle)
        # Only keep players who cleared the candidacy floor; watchlist below the
        # floor is noise on the board (still reproducible from the raw data).
        if (res.get("breakout_score") or 0) >= min_score:
            results.append(res)

    # ── coverage: how many scored candidates had each signal ─────────────────
    n = len(results) or 1
    with_snap = sum(1 for r in results if r["signals"].get("snap_share", {}).get("available"))
    coverage = round(with_snap / n, 3)
    scores = [float(r.get("breakout_score") or 0) for r in results]
    hundreds = sum(1 for score in scores if score == 100.0)
    provisional = [r for r in results if r.get("provisional")]
    provisional_max = max((float(r.get("breakout_score") or 0) for r in provisional), default=None)
    tied_at_provisional_max = sum(
        1 for r in provisional if float(r.get("breakout_score") or 0) == provisional_max)
    tied_pct = (100.0 * tied_at_provisional_max / len(provisional)) if provisional else 0.0
    cap_reduced = sum(1 for r in provisional if r.get("provisional_adjustment_applied"))
    distribution = {
        "min": round(min(scores), 1) if scores else None,
        "median": round(sorted(scores)[len(scores) // 2], 1) if scores else None,
        "max": round(max(scores), 1) if scores else None,
        "exactly_100": hundreds,
        "exactly_100_pct": round(100.0 * hundreds / len(scores), 2) if scores else 0.0,
        "tied_score_counts": {str(score): scores.count(score) for score in sorted(set(scores))
                              if scores.count(score) > 1},
        "provisional_max": provisional_max,
        "provisional_max_tied": tied_at_provisional_max,
        "provisional_max_tied_pct": round(tied_pct, 2),
        "provisional_adjusted_pct": round(100.0 * cap_reduced / len(provisional), 2)
                                    if provisional else 0.0,
    }
    if len(provisional) >= 10 and tied_pct > 20.0:
        print("[weekly_breakout] WARNING: provisional ranking separation guard failed: "
              f"{tied_at_provisional_max}/{len(provisional)} ({tied_pct:.1f}%) share "
              f"the maximum score {provisional_max}")
    def _percentiles(values):
        ordered = sorted(values)
        if not ordered:
            return {"p50": None, "p90": None, "p95": None}
        return {name: round(ordered[int(frac * (len(ordered) - 1))], 1)
                for name, frac in (("p50", .50), ("p90", .90), ("p95", .95))}

    distribution["by_position"] = {
        pos: _percentiles([float(r["breakout_score"]) for r in results if r.get("position") == pos])
        for pos in ("QB", "RB", "WR", "TE")
    }
    distribution["season_phase"] = "early" if cutoff <= 4 else "mid" if cutoff <= 11 else "late"
    prior_statuses = {status: sum(1 for value in prior.values()
                                  if value.get("history_status") == status)
                      for status in ("usable", "partial", "provider_defaults_only")}
    coverage_detail = {"players_scored": scanned, **baseline_counts,
                       "prior_cache_rows": len(prior), "prior_cache_status": prior_statuses}
    # Once games exist, a near-empty prior map means an ID/schema/unit pipeline
    # regression, not a rookie-heavy class. Make that operationally obvious.
    prior_eligible = baseline_counts["prior_season"] + baseline_counts["none"]
    if prior_eligible >= 20 and baseline_counts["prior_season"] / prior_eligible < 0.25:
        print("[weekly_breakout] WARNING: prior-season baseline coverage collapsed "
              f"({baseline_counts['prior_season']}/{prior_eligible}); check Sleeper IDs "
              "and cache/player_history usage units")

    # ── preserve on empty: never wipe a good board with nothing ──────────────
    if not results:
        weekly_store.record_run(
            season, cutoff, mode=context.mode, status="stale",
            candidates_scored=scanned, records_saved=0, weeks_covered=weeks_covered,
            detail={**summary, "baseline_coverage": coverage_detail,
                    "score_distribution": distribution,
                    "note": "no candidates scored; previous snapshot preserved"},
            as_of_date=context.as_of_date,
        )
        summary.update(status="stale", candidates_scanned=scanned, records_saved=0,
                       coverage=coverage)
        _log_summary(summary, context)
        return summary

    saved = weekly_store.save_weekly_scores(season, cutoff, results, as_of_date=context.as_of_date)
    weekly_store.record_run(
        season, cutoff, mode=context.mode, status="success",
        candidates_scored=scanned, records_saved=saved, weeks_covered=weeks_covered,
        detail={**summary, "coverage": coverage, "baseline_coverage": coverage_detail,
                "score_distribution": distribution,
                "classifications": _class_counts(results)},
        as_of_date=context.as_of_date,
    )
    summary.update(status="success", candidates_scanned=scanned, records_saved=saved,
                   coverage=coverage, baseline_coverage=coverage_detail,
                   score_distribution=distribution,
                   classifications=_class_counts(results))
    _log_summary(summary, context)
    return summary


def _class_counts(results: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in results:
        c = r.get("classification") or "unknown"
        out[c] = out.get(c, 0) + 1
    return out


def _log_summary(summary: Dict[str, Any], context: ScoringContext) -> None:
    print(
        "[weekly_breakout] "
        f"season={summary['season']} mode={summary['mode']} "
        f"cutoff_week={summary['cutoff_week']} "
        f"scoring={summary['scoring_version']} "
        f"data_cutoff={summary['as_of_date']} "
        f"coverage={summary.get('coverage')} "
        f"candidates_scanned={summary.get('candidates_scanned')} "
        f"records_saved={summary.get('records_saved')} "
        f"baselines={summary.get('baseline_coverage')} "
        f"scores={summary.get('score_distribution')} "
        f"status={summary.get('status')} "
        f"classes={summary.get('classifications')}"
    )
