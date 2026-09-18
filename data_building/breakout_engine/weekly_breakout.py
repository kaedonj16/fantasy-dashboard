"""
Weekly (in-season) breakout detection.

The offseason engine (build_historical_scores + core/components) scores role
*opportunity* from roster churn and prior-season production. That path is wrong
during the season: it reads last season's stats and labels everything
``phase="offseason"`` even in October, so a rookie who just took over a backfield
never surfaces until the following spring.

This module is the dedicated in-season path. Its primary evidence is *current
season weekly usage* - the per-game snap share, target share, targets, carries
and passing volume already persisted in ``player_weekly_metrics`` by
``data_building.weekly_metrics``. It compares a player's latest 1-2 completed
games against the immediately preceding 3-4 (non-overlapping windows, so momentum
is real and not diluted by including the same games on both sides), rewards
usage growth *before* it turns into fantasy points, and separates three things
the old board conflated:

    classification  - what kind of situation this is (emerging / temporary / watchlist)
    breakout_score  - how large the role change is (0-100, sample-independent)
    confidence      - how much to trust it (0-100, from sample size, coverage,
                      freshness, persistence, and signal agreement)

Everything in this module is pure and DB-free: ``score_player`` takes plain lists
of weekly-row dicts so it can be unit-tested without Postgres. The thin DB layer
(loading rows, refreshing data, persisting results) lives in
``weekly_store`` and ``weekly_runner``.

Share units: ``snap_pct`` and ``target_share`` are stored as percentages on a
0-100 scale (see weekly_metrics.build_weekly_metrics). A missing share is
``None`` and MUST stay ``None`` (unknown) - never coerced to 0, which would read
as "played but had no role".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# Bump when the scoring math changes so persisted rows are self-identifying and a
# stale row can be told apart from a current-version one.
SCORING_VERSION = "weekly-v5"

_POSITIONS = ("QB", "RB", "WR", "TE")

# Trend window shape. "recent" is the latest 1-2 completed games; "baseline" the
# 3-4 immediately before it. The two never overlap.
RECENT_MAX = 2
BASELINE_MAX = 4

# Per-position "full workload" anchors used to turn raw per-game counts into a
# 0-100 growth signal. These are typical every-down-starter per-game volumes, not
# ceilings - they only set the scale on which a delta is judged "large".
_FULL_TARGETS_PG = {"WR": 8.5, "TE": 6.5}
_FULL_OPPORTUNITY_PG = {"RB": 18.0}   # carries + targets
_FULL_PASS_ATT_PG = {"QB": 33.0}
_FULL_RUSH_PG = {"QB": 6.0, "RB": 15.0}

# Classification / candidacy thresholds. Kept here (not inline) so tuning is
# auditable and tests can import the exact cutoffs.
WATCHLIST_MIN_SCORE = 18.0     # below this a player is not a breakout candidate
EMERGING_MIN_SCORE = 42.0      # sustained role change large enough to headline
EMERGING_MIN_BASELINE_GAMES = 2

# Initial-role and quality gates. These are product semantics, not hidden magic
# numbers: one game can only be a watchlist; two held games can be provisional.
INITIAL_ONE_GAME_CAP = 35.0
INITIAL_PERSISTENT_CAP = 45.0
INITIAL_ROLE_DISCOVERY_MIN = 35.0
MIN_SUPPORTING_SIGNALS = 2
ROLE_QUALITY_FLOORS = {
    "WR": {"route_participation": 60.0, "target_share": 15.0, "targets_pg": 5.0},
    "TE": {"route_participation": 60.0, "target_share": 15.0, "targets_pg": 5.0},
    "RB": {"carry_opportunity_pg": 10.0, "targets_pg": 3.0},
    "QB": {"dropback_share": 75.0, "pass_att_pg": 24.0, "rush_pg": 4.0},
}

# Absolute-role anchors. The weighted averages are naturally bounded 0..100;
# unavailable inputs are omitted and reduce confidence rather than becoming 0.
CURRENT_ROLE_WEIGHTS = {
    "WR": {"route_participation": .40, "target_share": .35, "targets_pg": .25},
    "TE": {"route_participation": .45, "target_share": .35, "targets_pg": .20},
    "RB": {"carry_opportunity_pg": .60, "targets_pg": .25, "routes_pg": .15},
    "QB": {"dropback_share": .45, "pass_att_pg": .40, "rush_pg": .15},
}

# A single game's fantasy output this many times the baseline, with no matching
# usage growth, is flagged as efficiency/TD-driven rather than a role change.
FANTASY_SPIKE_RATIO = 1.8


# =============================================================================
# small numeric helpers - all preserve "unknown" (None) rather than inventing 0
# =============================================================================

def _num(v: Any) -> Optional[float]:
    """Coerce to float, but keep None/'' as None (unknown). Never turns a missing
    value into 0.0."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _mean_present(rows: Sequence[Dict], key: str) -> Tuple[Optional[float], int]:
    """Mean of the present (non-None) values of ``key`` across rows.

    Returns (mean_or_None, count_present). Missing values are excluded from both
    the sum and the divisor, so a share that is unknown in some weeks does not
    drag the average toward zero.
    """
    vals = [_num(r.get(key)) for r in rows]
    present = [v for v in vals if v is not None]
    if not present:
        return None, 0
    return sum(present) / len(present), len(present)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _round(v: Optional[float], n: int = 1) -> Optional[float]:
    return None if v is None else round(v, n)


# =============================================================================
# game-status classification
# =============================================================================
# player_weekly_metrics only stores weeks a player was active with usage, so an
# absent week is either a bye, an inactive game, or missing data. We disambiguate
# with the team's played-week set when it is available.

STATUS_ACTIVE = "active"
STATUS_ACTIVE_NO_USAGE = "active_no_usage"
STATUS_BYE = "bye"
STATUS_INACTIVE = "inactive"
STATUS_MISSING = "missing"


def classify_week_status(
    row: Optional[Dict],
    week: int,
    team_weeks_played: Optional[set] = None,
) -> str:
    """Label one week for one player.

    ``row`` is the player's ``player_weekly_metrics`` row for that week, or None.
    ``team_weeks_played`` is the set of weeks the player's team actually played
    (from the schedule); when provided it separates byes from inactives.
    """
    if row is not None:
        snaps = _num(row.get("snaps")) or 0.0
        targets = _num(row.get("targets")) or 0.0
        carries = _num(row.get("carries")) or 0.0
        pass_att = _num(row.get("pass_att")) or 0.0
        if snaps <= 0 and targets <= 0 and carries <= 0 and pass_att <= 0:
            return STATUS_ACTIVE_NO_USAGE
        return STATUS_ACTIVE
    if team_weeks_played is not None:
        if week not in team_weeks_played:
            return STATUS_BYE
        return STATUS_INACTIVE
    return STATUS_MISSING


# =============================================================================
# trend windows
# =============================================================================

def split_windows(
    active_rows: Sequence[Dict],
    recent_max: int = RECENT_MAX,
    baseline_max: int = BASELINE_MAX,
) -> Tuple[List[Dict], List[Dict]]:
    """Split a player's active weekly rows (oldest first) into non-overlapping
    recent and baseline windows.

    Sizing keeps momentum meaningful with small samples - the old
    recent-3-vs-season-average comparison put the same games on both sides and
    reported zero momentum through three games:

        1 game  -> recent=1, baseline=0   (provisional; caller supplies a prior baseline)
        2-3     -> recent=1, baseline=rest (1 vs 1, 1 vs 2)
        4+      -> recent=2, baseline=up to 4 preceding
    """
    rows = list(active_rows)
    n = len(rows)
    if n == 0:
        return [], []
    if n == 1:
        return rows[-1:], []
    if n <= 3:
        return rows[-1:], rows[:-1][-baseline_max:]
    r = min(recent_max, 2)
    recent = rows[n - r:]
    baseline = rows[max(0, n - r - baseline_max): n - r]
    return recent, baseline


# =============================================================================
# per-signal growth scoring
# =============================================================================

def _share_growth(baseline: Optional[float], recent: Optional[float], *, allow_initial: bool = True) -> Dict[str, Any]:
    """Score growth in a 0-100 team-share metric (snap share, target share).

    Rewards BOTH the change and the resulting workload level, so a 45%->65% move
    (bigger role, bigger jump) outranks 5%->15%. The resulting-level term only
    applies when there was actual growth, so an already-established starter who
    holds steady scores 0 - this board is about role *change*, not role size.
    """
    out = {
        "baseline": _round(baseline), "recent": _round(recent),
        "delta": None, "points": 0.0, "available": recent is not None,
    }
    if recent is None:
        return out
    if baseline is None:
        # This is an initial-role observation, not growth from zero.  Give a
        # useful but deliberately conservative workload-level signal; the final
        # provisional score is capped too.  Most importantly, baseline/delta
        # remain null in stored evidence and in the UI.
        out["points"] = (round(_clamp(recent / 75.0 * 45.0, 0.0, 45.0), 1)
                         if allow_initial else 0.0)
        out["initial_role"] = True
        return out
    delta = recent - baseline
    out["delta"] = _round(delta)
    if delta <= 0:
        return out
    delta_pts = _clamp(delta / 25.0 * 60.0, 0.0, 60.0)     # +25pp -> 60
    # The resulting-level bonus scales with how *meaningful* the growth is, so a
    # trivial +1pp wiggle on an established 55%-snap starter does not unlock the
    # full "grew into a big role" credit. Full level bonus needs ~15pp of growth.
    level_frac = _clamp(delta / 15.0, 0.0, 1.0)
    level_pts = _clamp(recent / 100.0 * 40.0, 0.0, 40.0) * level_frac
    out["points"] = round(delta_pts + level_pts, 1)
    return out


def _count_growth(baseline: Optional[float], recent: Optional[float], full: float,
                  *, allow_initial: bool = True) -> Dict[str, Any]:
    """Score growth in a per-game *count* metric (targets, carry+target
    opportunity, pass attempts) against a position "full workload" anchor.

    Counts are secondary to shares because a team that simply runs more plays
    inflates everyone's counts; the share signals carry the team-normalized role
    read, and this adds workload magnitude on top.
    """
    out = {
        "baseline": _round(baseline), "recent": _round(recent),
        "delta": None, "points": 0.0, "available": recent is not None,
    }
    if recent is None:
        return out
    if baseline is None:
        out["points"] = (round(_clamp(recent / full * 35.0, 0.0, 35.0), 1)
                         if allow_initial else 0.0)
        out["initial_role"] = True
        return out
    delta = recent - baseline
    out["delta"] = _round(delta)
    if delta <= 0:
        return out
    delta_pts = _clamp(delta / (full * 0.5) * 60.0, 0.0, 60.0)   # +half a full load -> 60
    # Level bonus scaled by growth significance (full at ~35% of a full workload
    # gained), so a negligible count bump on an already-busy player scores small.
    level_frac = _clamp(delta / (full * 0.35), 0.0, 1.0)
    level_pts = _clamp(recent / full * 40.0, 0.0, 40.0) * level_frac
    out["points"] = round(delta_pts + level_pts, 1)
    return out


def _weighted_score(signals: Dict[str, Dict[str, Any]], weights: Dict[str, float]) -> float:
    """Fixed, explicit 0-100 contribution budget.

    Weights sum to one for each position. Missing optional evidence contributes
    nothing rather than re-scaling the remaining signals to a larger maximum.
    Thus routes/red-zone detail can support a result but can never create a 100.
    """
    return sum(weights[key] * float((signals.get(key) or {}).get("points") or 0.0)
               for key in weights)


def _absolute_role_score(position: str, signals: Dict[str, Dict[str, Any]]) -> float:
    """Fantasy relevance of the resulting role, independent of role change."""
    floors = ROLE_QUALITY_FLOORS.get(position, {})
    weights = CURRENT_ROLE_WEIGHTS.get(position, {})
    parts = []
    for key, weight in weights.items():
        recent = (signals.get(key) or {}).get("recent")
        floor = floors.get(key)
        if recent is None or not floor:
            continue
        # Twice the emerging floor represents a full 100-level role.
        parts.append((weight, _clamp(float(recent) / (2.0 * floor) * 100.0, 0.0, 100.0)))
    if not parts:
        return 0.0
    # Fixed weights: missing optional inputs cannot inflate the inputs that remain.
    return round(sum(weight * value for weight, value in parts), 1)


def _signal_diagnostics(position: str, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Independent support/conflict accounting, not a count of every raw field."""
    groups = {
        "WR": (("route_participation",), ("target_share",), ("targets_pg",)),
        "TE": (("route_participation",), ("target_share",), ("targets_pg",)),
        "RB": (("carry_opportunity_pg",), ("targets_pg",), ("high_value_opportunities_pg",)),
        "QB": (("dropback_share",), ("pass_att_pg",), ("rush_pg",)),
    }.get(position, ())
    supporting = []
    conflicts = []
    for group in groups:
        positive = [key for key in group
                    if (((signals.get(key) or {}).get("delta") is not None and
                         float(signals[key]["delta"]) > 0) or
                        (signals.get(key) or {}).get("initial_role"))
                    and float(signals[key].get("points") or 0) >= 10.0]
        if positive:
            supporting.append(positive[0])
    snap_delta = (signals.get("snap_share") or {}).get("delta")
    route_delta = (signals.get("routes_pg") or {}).get("delta")
    target_delta = (signals.get("targets_pg") or {}).get("delta")
    if snap_delta is not None and snap_delta > 0 and route_delta is not None and route_delta < 0:
        conflicts.append("snaps_up_routes_down")
    if target_delta is not None and target_delta > 0 and route_delta is not None and route_delta <= 0:
        conflicts.append("targets_up_without_route_growth")
    available_groups = sum(1 for group in groups if any((signals.get(k) or {}).get("available") for k in group))
    agreement = 100.0 * len(supporting) / max(1, available_groups)
    agreement -= 20.0 * len(conflicts)
    return {"supporting_signal_count": len(supporting), "supporting_signals": supporting,
            "conflicting_signals": conflicts,
            "signal_agreement_score": round(_clamp(agreement, 0.0, 100.0), 1)}


def _trend_profile(recent: List[Dict], position: str) -> Dict[str, Any]:
    key = "target_share" if position in ("WR", "TE") else "snap_pct"
    vals = [v for v in (_num(row.get(key)) for row in recent) if v is not None]
    if len(vals) < 2:
        return {"state": "one_game", "persistent": False, "score": 35.0}
    delta = vals[-1] - vals[-2]
    spread = max(vals) - min(vals)
    if delta < -max(5.0, vals[-2] * .20):
        return {"state": "reversal", "persistent": False, "score": 20.0}
    if spread <= max(8.0, sum(vals) / len(vals) * .20):
        return {"state": "elevated_and_held", "persistent": True, "score": 80.0}
    if delta > 0:
        return {"state": "consecutive_growth", "persistent": True, "score": 70.0}
    return {"state": "volatile", "persistent": False, "score": 40.0}


# =============================================================================
# per-position role signal assembly
# =============================================================================

def _pg(total: Optional[float], games: int) -> Optional[float]:
    if total is None or games <= 0:
        return None
    return total / games


def _position_signals(
    position: str,
    recent: List[Dict],
    baseline: List[Dict],
    *,
    initial_role: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], float]:
    """Compute per-signal growth detail and the combined role-change score for a
    position. Returns (signals, role_score) where role_score is 0-100.
    """
    rn = max(1, len(recent))
    bn = max(1, len(baseline))

    # Shares are already per-game (each row is one game); average the present ones.
    snap_r, _ = _mean_present(recent, "snap_pct")
    snap_b, _ = _mean_present(baseline, "snap_pct")
    tgtshare_r, _ = _mean_present(recent, "target_share")
    tgtshare_b, _ = _mean_present(baseline, "target_share")

    # Per-game counts: sum present values then divide by games in the window.
    def _count_pg(rows: List[Dict], key: str) -> Optional[float]:
        total, present = _mean_present(rows, key)
        return total  # _mean_present already returns per-present-row mean == per-game

    tgt_r = _count_pg(recent, "targets")
    tgt_b = _count_pg(baseline, "targets")
    car_r = _count_pg(recent, "carries")
    car_b = _count_pg(baseline, "carries")
    pass_r = _count_pg(recent, "pass_att")
    pass_b = _count_pg(baseline, "pass_att")
    route_r = _count_pg(recent, "routes")
    route_b = _count_pg(baseline, "routes")
    dropbacks_r = _count_pg(recent, "team_dropbacks")
    dropbacks_b = _count_pg(baseline, "team_dropbacks")
    route_part_r = (route_r / dropbacks_r * 100.0
                    if route_r is not None and dropbacks_r else None)
    route_part_b = (route_b / dropbacks_b * 100.0
                    if route_b is not None and dropbacks_b else None)
    player_dropbacks_r = _count_pg(recent, "dropbacks")
    player_dropbacks_b = _count_pg(baseline, "dropbacks")
    dropback_share_r = (player_dropbacks_r / dropbacks_r * 100.0
                        if player_dropbacks_r is not None and dropbacks_r else None)
    dropback_share_b = (player_dropbacks_b / dropbacks_b * 100.0
                        if player_dropbacks_b is not None and dropbacks_b else None)
    rz_r = _count_pg(recent, "red_zone_opportunities")
    rz_b = _count_pg(baseline, "red_zone_opportunities")

    signals: Dict[str, Dict[str, Any]] = {}
    snap = _share_growth(snap_b, snap_r, allow_initial=initial_role)
    signals["snap_share"] = snap

    if position in ("WR", "TE"):
        tshare = _share_growth(tgtshare_b, tgtshare_r, allow_initial=initial_role)
        tcount = _count_growth(tgt_b, tgt_r, _FULL_TARGETS_PG.get(position, 8.0),
                               allow_initial=initial_role)
        signals["target_share"] = tshare
        signals["targets_pg"] = tcount
        routes = _count_growth(route_b, route_r, 30.0, allow_initial=initial_role)
        signals["routes_pg"] = routes
        signals["route_participation"] = _share_growth(
            route_part_b, route_part_r, allow_initial=initial_role)
        high_value = _count_growth(rz_b, rz_r, 3.0, allow_initial=initial_role)
        signals["high_value_opportunities_pg"] = high_value
        role_score = _weighted_score(signals, {
            "route_participation": .35, "target_share": .35, "targets_pg": .22,
            "high_value_opportunities_pg": .08,
        })

    elif position == "RB":
        # Opportunity = carries + targets per game (NOT touches: receptions depend
        # on completion outcomes, so touches understates a back's earned work).
        # Composite opportunity is only comparable when both constituents were
        # observed; a missing receiving/carry feed is not an observed zero.
        opp_r = None if (car_r is None or tgt_r is None) else car_r + tgt_r
        opp_b = None if (car_b is None or tgt_b is None) else car_b + tgt_b
        opp = _count_growth(opp_b, opp_r, _FULL_OPPORTUNITY_PG["RB"], allow_initial=initial_role)
        tcount = _count_growth(tgt_b, tgt_r, 4.0, allow_initial=initial_role)
        signals["carry_opportunity_pg"] = opp
        signals["targets_pg"] = tcount
        routes = _count_growth(route_b, route_r, 22.0, allow_initial=initial_role)
        signals["routes_pg"] = routes
        high_value = _count_growth(rz_b, rz_r, 3.0, allow_initial=initial_role)
        signals["high_value_opportunities_pg"] = high_value
        role_score = _weighted_score(signals, {
            "carry_opportunity_pg": .60, "targets_pg": .22,
            "routes_pg": .08, "high_value_opportunities_pg": .10,
        })

    elif position == "QB":
        signals["dropback_share"] = _share_growth(
            dropback_share_b, dropback_share_r, allow_initial=initial_role)
        pass_g = _count_growth(pass_b, pass_r, _FULL_PASS_ATT_PG["QB"], allow_initial=initial_role)
        rush_g = _count_growth(car_b, car_r, _FULL_RUSH_PG["QB"], allow_initial=initial_role)
        signals["pass_att_pg"] = pass_g
        signals["rush_pg"] = rush_g
        role_score = _weighted_score(signals, {
            "dropback_share": .35, "pass_att_pg": .45, "rush_pg": .20,
        })

    else:
        role_score = _clamp(snap["points"], 0.0, 100.0)

    return signals, round(role_score, 1)


# =============================================================================
# confidence
# =============================================================================

def _sample_factor(recent_games: int, baseline_games: int) -> float:
    """0-1 from total games observed. Provisional 1-game samples land low."""
    total = recent_games + baseline_games
    table = {0: 0.0, 1: 0.30, 2: 0.45, 3: 0.60, 4: 0.72, 5: 0.82}
    return table.get(total, 0.90)


def _persistence_factor(recent: List[Dict], baseline: List[Dict], key: str) -> Optional[float]:
    """0-1: did the recent window hold up rather than being one spike? Every
    recent game at or above the baseline mean scores high; a lone spike scores
    low. None when the signal is unavailable."""
    b_mean, _ = _mean_present(baseline, key)
    r_vals = [v for v in (_num(r.get(key)) for r in recent) if v is not None]
    if not r_vals:
        return None
    if b_mean is None:
        return 0.5  # provisional: no baseline to persist against
    above = sum(1 for v in r_vals if v >= b_mean - 1e-9)
    return above / len(r_vals)


def _compute_confidence(
    signals: Dict[str, Dict[str, Any]],
    recent: List[Dict],
    baseline: List[Dict],
    position: str,
    weeks_stale: int,
    provisional: bool,
) -> Tuple[float, Dict[str, Any]]:
    """Blend sample size, source coverage, freshness, persistence and signal
    agreement into a 0-100 confidence. This is deliberately NOT the breakout
    score and NOT presented as a calibrated probability - it is how much to trust
    the score given the evidence behind it."""
    recent_games = len(recent)
    baseline_games = len(baseline)

    sample = _sample_factor(recent_games, baseline_games)

    expected = [k for k in signals]
    available = [k for k, v in signals.items() if v.get("available")]
    coverage = (len(available) / len(expected)) if expected else 0.0

    freshness = _clamp(1.0 - 0.25 * max(0, weeks_stale), 0.0, 1.0)

    key = "snap_pct" if position == "QB" else (
        "target_share" if position in ("WR", "TE") else "snap_pct")
    persistence = _persistence_factor(recent, baseline, key)
    if persistence is None:
        persistence = 0.5

    # Agreement: how many scored signals point the same (positive) way.
    pos_signals = [v for v in signals.values() if v.get("available") and (v.get("points") or 0) > 0]
    scored = [v for v in signals.values() if v.get("available")]
    agreement = (len(pos_signals) / len(scored)) if scored else 0.0

    conf = 100.0 * (
        0.34 * sample +
        0.16 * coverage +
        0.16 * freshness +
        0.20 * persistence +
        0.14 * agreement
    )
    if provisional:
        conf = min(conf, 35.0)
    conf = round(_clamp(conf, 0.0, 100.0), 1)
    detail = {
        "sample": round(sample, 3),
        "coverage": round(coverage, 3),
        "freshness": round(freshness, 3),
        "persistence": round(persistence, 3),
        "agreement": round(agreement, 3),
        "recent_games": recent_games,
        "baseline_games": baseline_games,
        "weeks_stale": weeks_stale,
    }
    return conf, detail


# =============================================================================
# deterministic explanations
# =============================================================================

_SIGNAL_LABELS = {
    "snap_share": ("Snap share", "%"),
    "target_share": ("Target share", "%"),
    "targets_pg": ("Targets/game", ""),
    "carry_opportunity_pg": ("Carries+targets/game", ""),
    "pass_att_pg": ("Pass attempts/game", ""),
    "rush_pg": ("Rush attempts/game", ""),
    "routes_pg": ("Routes/game", ""),
    "route_participation": ("Route participation", "%"),
    "dropback_share": ("Dropback share", "%"),
    "high_value_opportunities_pg": ("Red-zone opportunities/game", ""),
}


def _fmt(v: Optional[float], unit: str) -> str:
    if v is None:
        return "n/a"
    if unit == "%":
        return f"{v:.0f}%"
    return f"{v:.1f}"


def _build_reasons(
    signals: Dict[str, Dict[str, Any]],
    recent_games: int,
    baseline_games: int,
) -> List[str]:
    """Deterministic, input-grounded reason strings. No adjectives that aren't
    derived from the numbers - e.g.
    'Snap share increased from 42% to 68% over the last 2 games.'"""
    window_phrase = (
        f"over the last {recent_games} game{'s' if recent_games != 1 else ''}"
    )
    # Rank scored signals by contribution.
    ranked = sorted(
        ((k, v) for k, v in signals.items() if k != "snap_share" and
         v.get("available") and (v.get("points") or 0) > 0),
        key=lambda kv: kv[1].get("points") or 0,
        reverse=True,
    )
    reasons: List[str] = []
    for key, sig in ranked[:3]:
        label, unit = _SIGNAL_LABELS.get(key, (key, ""))
        base = sig.get("baseline")
        rec = sig.get("recent")
        if base is None:
            reasons.append(
                f"{label} at {_fmt(rec, unit)} {window_phrase} "
                f"(no prior-window baseline yet)."
            )
        else:
            verb = "increased" if (rec or 0) >= (base or 0) else "changed"
            reasons.append(
                f"{label} {verb} from {_fmt(base, unit)} to {_fmt(rec, unit)} "
                f"{window_phrase}."
            )
    if not reasons:
        reasons.append("No meaningful usage growth versus the prior window.")
    return reasons


# =============================================================================
# top-level pure scorer
# =============================================================================

def score_player(
    player: Dict[str, Any],
    weekly_rows: Sequence[Dict[str, Any]],
    *,
    prior_baseline: Optional[Dict[str, Any]] = None,
    injury_context: Optional[Dict[str, Any]] = None,
    team_weeks_played: Optional[set] = None,
    cutoff_week: Optional[int] = None,
) -> Dict[str, Any]:
    """Score one player from current-season weekly usage. Pure and DB-free.

    Args:
        player: {player_id, player_name, team, position, ...}
        weekly_rows: that player's player_weekly_metrics rows for the season,
            any order; each is a dict with week/snap_pct/snaps/targets/carries/
            pass_att/target_share/ppr_pts. Only active weeks appear (the builder
            drops inactive ones), which is why absent weeks are classified via
            team_weeks_played.
        prior_baseline: optional prior-season per-game usage
            {snap_pct, target_share, targets_pg, carries_pg, pass_att_pg} used as
            the baseline window ONLY when the player has too few current-season
            games. Rookies pass None and are handled provisionally.
        injury_context: optional {vacated: bool, source: str} indicating a
            teammate ahead is out - drives the "temporary opportunity" class.
        team_weeks_played: weeks the team played (for bye vs inactive labeling).
        cutoff_week: as-of week; rows after it are ignored (strict cutoff so
            historical scoring can never read future weeks).

    Returns a structured result dict (see module docstring).
    """
    position = (player.get("position") or "").upper()

    # Strict as-of cutoff: never read a week beyond the cutoff.
    rows = [dict(r) for r in weekly_rows if _num(r.get("week")) is not None]
    if cutoff_week is not None:
        rows = [r for r in rows if int(r["week"]) <= int(cutoff_week)]
    rows.sort(key=lambda r: int(r["week"]))

    # Active weeks only carry usage; that is what the windows compare.
    active = [r for r in rows if classify_week_status(r, int(r["week"]), team_weeks_played) in
              (STATUS_ACTIVE, STATUS_ACTIVE_NO_USAGE)]

    evaluated_weeks = [int(r["week"]) for r in active]
    recent, baseline = split_windows(active)

    baseline_source = "current_season"
    provisional = False

    # Too little current-season baseline: fall back to a prior-season baseline row
    # if one was supplied. Rookies (no prior) run provisionally on current data.
    prior_usable = bool(prior_baseline and any(
        prior_baseline.get(key) is not None for key in
        ("snap_pct", "target_share", "targets_pg", "carries_pg", "pass_att_pg")
    ))
    if len(baseline) == 0:
        provisional = True
        if prior_usable:
            baseline_source = "prior_season"
        else:
            baseline_source = "none"
    if not prior_usable and len(active) <= 2:
        # Two initial games can prove persistence, but are not yet a normal
        # baseline-to-recent comparison. Week 3 creates the first real window.
        provisional = True
        baseline_source = "none"
        recent, baseline = list(active), []

    # Build the baseline window. When falling back to prior season, synthesize a
    # single pseudo-game row from the per-game prior values so the same signal
    # math applies.
    baseline_rows = list(baseline)
    if baseline_source == "prior_season" and prior_baseline:
        baseline_rows = [{
            "week": 0,
            "snap_pct": prior_baseline.get("snap_pct"),
            "target_share": prior_baseline.get("target_share"),
            "targets": prior_baseline.get("targets_pg"),
            "carries": prior_baseline.get("carries_pg"),
            "pass_att": prior_baseline.get("pass_att_pg"),
        }]

    score_basis = "initial_role" if baseline_source == "none" else "role_change"
    signals, raw_role_change = _position_signals(
        position, recent, baseline_rows, initial_role=(score_basis == "initial_role"))
    role_change_score = None if score_basis == "initial_role" else raw_role_change
    current_role_score = _absolute_role_score(position, signals)
    diagnostics = _signal_diagnostics(position, signals)
    trend = _trend_profile(active[-3:], position)

    # A cache miss alone never means rookie.
    years_exp = _num(player.get("years_exp"))
    rookie_year = _num(player.get("rookie_year"))
    draft_year = _num(player.get("draft_year") or player.get("draft_yr"))
    season = _num(player.get("season"))
    is_rookie = bool(years_exp == 0 or (season is not None and
                     (rookie_year == season or draft_year == season)))

    draft_round = _num(player.get("draft_round"))
    depth_order = _num(player.get("depth_chart_order"))
    expected_role = None
    expectation_inputs = []
    if depth_order is not None:
        expected_role = _clamp(85.0 - 20.0 * (depth_order - 1.0), 15.0, 85.0)
        expectation_inputs.append("preseason_depth_chart")
    if is_rookie and draft_round is not None:
        draft_expectation = _clamp(85.0 - 10.0 * (draft_round - 1.0), 20.0, 85.0)
        expected_role = (draft_expectation if expected_role is None else
                         (expected_role + draft_expectation) / 2.0)
        expectation_inputs.append("draft_capital")
    expectation_delta_score = (50.0 if expected_role is None else
                               _clamp(50.0 + current_role_score - expected_role, 0.0, 100.0))

    prior_level = None
    if prior_usable:
        prior_values = [prior_baseline.get(k) for k in ("snap_pct", "target_share")
                        if prior_baseline.get(k) is not None]
        prior_level = max(prior_values, default=None)
    established_evidence = False
    if prior_usable:
        if position in ("WR", "TE"):
            established_evidence = any((prior_baseline.get(key) or 0) >= floor for key, floor in
                                       (("snap_pct", 60), ("target_share", 15), ("targets_pg", 5)))
        elif position == "RB":
            established_evidence = ((prior_baseline.get("snap_pct") or 0) >= 55 or
                                    (prior_baseline.get("carries_pg") or 0) +
                                    (prior_baseline.get("targets_pg") or 0) >= 12)
        elif position == "QB":
            established_evidence = ((prior_baseline.get("snap_pct") or 0) >= 75 or
                                    (prior_baseline.get("pass_att_pg") or 0) >= 24)
    established_role_penalty = 35.0 if established_evidence else (
        min(35.0, (prior_level - 55.0) * 1.4)
        if prior_level is not None and prior_level >= 60.0 else 0.0)
    novelty = _clamp((role_change_score if role_change_score is not None
                      else expectation_delta_score) - established_role_penalty, 0.0, 100.0)
    established_role_score = _clamp(
        (70.0 if established_evidence else 0.0) + min(30.0, float(prior_baseline.get("games") or 0) * 2.0)
        if prior_baseline else 0.0, 0.0, 100.0)

    # ── fantasy context: opportunity BEFORE production ────────────────────────
    # ppr is never part of the score. We only use it to (a) report and (b) flag a
    # points spike unmatched by role growth as a risk, so a TD/long-play game
    # doesn't masquerade as a breakout.
    recent_ppg, _ = _mean_present(recent, "ppr_pts")
    baseline_ppg, _ = _mean_present(baseline_rows, "ppr_pts")
    spike_without_role = False
    if (recent_ppg is not None and baseline_ppg not in (None, 0)
            and recent_ppg >= baseline_ppg * FANTASY_SPIKE_RATIO
            and (role_change_score or 0) < WATCHLIST_MIN_SCORE):
        spike_without_role = True

    weeks_stale = 0
    if cutoff_week is not None and evaluated_weeks:
        weeks_stale = max(0, int(cutoff_week) - evaluated_weeks[-1])

    confidence, conf_detail = _compute_confidence(
        signals, recent, baseline_rows, position, weeks_stale, provisional
    )
    confidence = round(_clamp(
        confidence - 7.5 * len(diagnostics["conflicting_signals"]), 0.0, 100.0), 1)

    # Explainable component view. These are evidence summaries, not calibrated
    # probabilities. Missing optional route/RZ data remains unavailable rather
    # than being represented as a zero observation.
    available_points = [float(v.get("points") or 0) for v in signals.values() if v.get("available")]
    opportunity_jump = max(available_points, default=0.0)
    expected_recent = []
    for sig in signals.values():
        if sig.get("available") and sig.get("recent") is not None and sig.get("baseline") is not None:
            expected_recent.append(max(0.0, float(sig["recent"]) - float(sig["baseline"])))
    unexpected_usage = _clamp((sum(expected_recent) / max(1, len(expected_recent))) * 8.0, 0.0, 100.0)
    hv = signals.get("high_value_opportunities_pg", {})
    high_value_score = float(hv.get("points") or 0.0) if hv.get("available") else None

    # ── classification: separate from score & confidence ─────────────────────
    injury_vacated = bool(injury_context and injury_context.get("vacated"))
    starter_returning = bool(injury_context and injury_context.get("starter_returning"))
    garbage_time = bool(recent and any(bool(r.get("garbage_time")) for r in recent))
    sustainability = .55 * trend["score"] + .45 * (100.0 * conf_detail["persistence"])
    if injury_vacated:
        sustainability += 15.0 if (injury_context or {}).get("multi_week") else -10.0
    if (injury_context or {}).get("starter_returned") and trend["persistent"]:
        sustainability += 15.0
    if starter_returning:
        sustainability -= 30.0
    if garbage_time:
        sustainability -= 35.0
    sustainability = round(_clamp(sustainability, 0.0, 100.0), 1)

    if score_basis == "initial_role":
        # Initial-role formula: current relevance dominates; novelty can reward
        # an unexpected late-round/depth-chart rise, but cannot fabricate change.
        final_score = (.55 * current_role_score + .20 * sustainability +
                       .15 * expectation_delta_score +
                       .10 * diagnostics["signal_agreement_score"])
        cap = INITIAL_ONE_GAME_CAP if len(active) <= 1 else INITIAL_PERSISTENT_CAP
        if not is_rookie and not expectation_inputs:
            # A missing cache match is not rookie evidence. Unknown-history
            # veterans remain visible at the low watchlist edge without being
            # rewarded as surprising debuts.
            cap = min(cap, WATCHLIST_MIN_SCORE - 0.1)
        final_score = min(final_score, cap)
    else:
        final_score = (.50 * float(role_change_score or 0) + .20 * current_role_score +
                       .10 * sustainability + .10 * novelty +
                       .10 * expectation_delta_score)
        final_score = max(0.0, final_score - 5.0 * len(diagnostics["conflicting_signals"]))
        # Supporting context can rank a real change; it cannot manufacture one.
        final_score *= min(1.0, float(role_change_score or 0) / WATCHLIST_MIN_SCORE)
    final_score = round(final_score, 1)
    pre_provisional_adjustment_score = final_score
    if len(active) == 1:
        # Continuous shrinkage preserves ranking separation; the ceiling is a
        # last-resort guard rather than the normal destination for every debut.
        final_score *= .68
        final_score = min(final_score, 39.5 if not established_evidence else 17.9)
    if garbage_time:
        final_score = round(final_score * .25, 1)
    ranking_score = round(final_score * (.75 + .25 * confidence / 100.0), 1)

    floors = ROLE_QUALITY_FLOORS.get(position, {})
    quality_hits = [key for key, floor in floors.items()
                    if (signals.get(key) or {}).get("recent") is not None
                    and float(signals[key]["recent"]) >= floor]
    route_available = bool((signals.get("route_participation") or {}).get("available"))
    targets = float((signals.get("targets_pg") or {}).get("recent") or 0)
    target_share = float((signals.get("target_share") or {}).get("recent") or 0)
    if position in ("WR", "TE"):
        # Routes are required when collected.  Missing routes invoke a deliberately
        # conservative target fallback; snaps are never substituted for routes.
        receiving_floor = (("route_participation" in quality_hits and
                            ("target_share" in quality_hits or "targets_pg" in quality_hits))
                           if route_available else (targets >= 5 and target_share >= 15))
        meaningful_role = receiving_floor
    elif position == "QB":
        meaningful_role = ("dropback_share" in quality_hits and
                           ("pass_att_pg" in quality_hits or "rush_pg" in quality_hits))
    else:
        meaningful_role = "carry_opportunity_pg" in quality_hits and len(quality_hits) >= 2

    non_snap_support = diagnostics["supporting_signal_count"]
    adequate_coverage = ((signals.get("target_share") or {}).get("available") and
                         (route_available or (signals.get("targets_pg") or {}).get("available"))) \
        if position in ("WR", "TE") else conf_detail["coverage"] >= .5
    exceptional_one_game = bool(
        len(active) == 1 and meaningful_role and non_snap_support >= 2 and
        novelty >= 25 and expectation_delta_score >= 60 and not established_evidence and
        not garbage_time and adequate_coverage)
    rejection_reasons = []
    if not meaningful_role:
        rejection_reasons.append("position_specific_role_floor_not_met")
    if non_snap_support < MIN_SUPPORTING_SIGNALS:
        rejection_reasons.append("fewer_than_two_independent_non_snap_signals")
    if established_evidence:
        rejection_reasons.append("established_player_without_material_role_transformation")
    if garbage_time:
        rejection_reasons.append("garbage_time_usage")
    if not adequate_coverage:
        rejection_reasons.append("insufficient_opportunity_data_coverage")
    if len(active) == 1 and not exceptional_one_game:
        rejection_reasons.append("one_game_evidence_not_exceptional")
    main_board_eligible = not rejection_reasons and (
        exceptional_one_game or (len(active) >= 2 and trend["persistent"] and
                                 final_score >= EMERGING_MIN_SCORE))
    if garbage_time:
        classification = "watchlist"
    elif main_board_eligible:
        classification = "emerging_breakout"
    elif score_basis == "initial_role":
        if injury_vacated and not (injury_context or {}).get("multi_week"):
            classification = "temporary_opportunity"
        else:
            classification = ("early_watch" if final_score >= WATCHLIST_MIN_SCORE else "watchlist")
    elif injury_vacated and not (injury_context or {}).get("multi_week"):
        classification = "temporary_opportunity"
    elif (final_score >= EMERGING_MIN_SCORE and meaningful_role and
          diagnostics["supporting_signal_count"] >= MIN_SUPPORTING_SIGNALS):
        classification = "emerging_breakout"
    else:
        classification = ("early_watch" if final_score >= WATCHLIST_MIN_SCORE else "watchlist")

    if garbage_time:
        opportunity_source = "garbage_time"
        source_confidence = 90.0
        source_reason = "Usage was concentrated in garbage-time appearances."
    elif injury_vacated:
        source_text = str((injury_context or {}).get("source") or "").lower()
        opportunity_source = ("multi_week_injury_opening" if
                              (injury_context or {}).get("multi_week") else "injury_replacement")
        source_confidence = 90.0
        source_reason = f"Opening linked to {(injury_context or {}).get('source') or 'a teammate absence'}."
    elif depth_order is not None and depth_order > 1 and trend["persistent"]:
        opportunity_source = "depth_chart_promotion"
        source_confidence = 65.0
        source_reason = "Usage held above the preseason depth-chart expectation."
    elif trend["state"] == "volatile":
        opportunity_source = "committee_rotation"
        source_confidence = 55.0
        source_reason = "Recent usage is volatile and consistent with a rotating role."
    elif recent and any(bool(row.get("game_script")) for row in recent):
        opportunity_source = "game_script"
        source_confidence = 60.0
        source_reason = "Usage was tagged as game-script dependent."
    else:
        opportunity_source = "unknown"
        source_confidence = 25.0
        source_reason = "No verified structural opportunity event is available."

    reasons = _build_reasons(signals, len(recent), len(baseline_rows))
    risks = _build_risks(
        provisional=provisional,
        baseline_source=baseline_source,
        spike_without_role=spike_without_role,
        injury_vacated=injury_vacated,
        weeks_stale=weeks_stale,
        confidence=confidence,
        recent_games=len(recent),
    )
    if injury_vacated and classification == "temporary_opportunity":
        src = (injury_context or {}).get("source") or "a teammate's absence"
        reasons.insert(0, f"Opening created by {src}.")

    return {
        "player_id": str(player.get("player_id") or ""),
        "player_name": player.get("player_name") or player.get("name"),
        "team": player.get("team"),
        "position": position,
        "scoring_version": SCORING_VERSION,
        "classification": classification,
        "breakout_score": final_score,
        "final_breakout_score": final_score,
        "pre_provisional_adjustment_score": pre_provisional_adjustment_score,
        "provisional_adjustment_applied": final_score < pre_provisional_adjustment_score,
        "ranking_score": ranking_score,
        "confidence": confidence,
        "provisional": provisional,
        "baseline_source": baseline_source,
        "score_basis": score_basis,
        "role_change_score": _round(role_change_score),
        "current_role_score": current_role_score,
        "sustainability_score": sustainability,
        "breakout_novelty_score": round(novelty, 1),
        "expectation_delta_score": round(expectation_delta_score, 1),
        "established_role_penalty": round(established_role_penalty, 1),
        "established_player": established_evidence,
        "established_role_score": round(established_role_score, 1),
        "role_novelty_score": round(novelty, 1),
        "previous_breakout_status": None,
        "role_novelty_reason": ("No prior NFL usage baseline; novelty uses available expectations."
                                if score_basis == "initial_role" else
                                "Measured against observed prior/current usage."),
        "is_rookie": is_rookie,
        "expected_role": _round(expected_role),
        "expectation_inputs": expectation_inputs,
        "opportunity_source": opportunity_source,
        "opportunity_source_confidence": source_confidence,
        "opportunity_source_reason": source_reason,
        "early_watch": classification == "early_watch",
        "main_board_eligible": main_board_eligible,
        "main_board_rejection_reasons": rejection_reasons,
        "baseline_method": ("healthy_prior_role" if baseline_source == "prior_season" else
                            "current_season_non_overlapping_windows" if baseline_source == "current_season"
                            else "initial_role_no_baseline"),
        "baseline_games_used": len(baseline_rows),
        "partial_games_excluded": int((prior_baseline or {}).get("partial_games_excluded") or 0),
        "baseline_quality": ("good" if len(baseline_rows) >= 2 else
                             "limited" if baseline_rows else "unavailable"),
        **diagnostics,
        "trend": trend,
        "evaluated_weeks": evaluated_weeks,
        "recent_weeks": [int(r["week"]) for r in recent],
        "baseline_weeks": [int(r["week"]) for r in baseline],
        "signals": signals,
        "components": {
            "role_change": _round(role_change_score),
            "current_role": current_role_score,
            "novelty": round(novelty, 1),
            "expectation_delta": round(expectation_delta_score, 1),
            "opportunity_jump": round(opportunity_jump, 1),
            "unexpected_usage": round(unexpected_usage, 1),
            "high_value_touches": _round(high_value_score),
            "sustainability": sustainability,
            "production_confirmation": None if recent_ppg is None else round(_clamp(recent_ppg * 4, 0, 100), 1),
        },
        "sample": {
            "recent_games": len(recent),
            "baseline_games": len(baseline),
            "total_games": len(active),
            "quality_signals": quality_hits,
        },
        "coverage": {
            "available": [k for k, v in signals.items() if v.get("available")],
            "expected": list(signals.keys()),
            "fraction": conf_detail["coverage"],
        },
        "confidence_detail": conf_detail,
        "fantasy": {
            "recent_ppg": _round(recent_ppg),
            "baseline_ppg": _round(baseline_ppg),
            "spike_without_role": spike_without_role,
        },
        "reasons": reasons,
        "risks": risks,
    }


def _classify(
    *,
    breakout_score: float,
    baseline_games: int,
    persistence: float,
    provisional: bool,
    injury_vacated: bool,
) -> str:
    """Emerging vs temporary vs watchlist. Score gates candidacy; the *kind* of
    situation is decided by persistence, sample, and whether a short-term absence
    is the identifiable cause."""
    if breakout_score < WATCHLIST_MIN_SCORE:
        return "watchlist"
    # A verified teammate absence driving the work is a temporary opportunity even
    # when the number is large - it may not persist once the starter returns.
    if injury_vacated and (provisional or baseline_games < EMERGING_MIN_BASELINE_GAMES):
        return "temporary_opportunity"
    if (not provisional
            and breakout_score >= EMERGING_MIN_SCORE
            and baseline_games >= EMERGING_MIN_BASELINE_GAMES
            and persistence >= 0.6):
        return "emerging_breakout"
    return "watchlist"


def _build_risks(
    *,
    provisional: bool,
    baseline_source: str,
    spike_without_role: bool,
    injury_vacated: bool,
    weeks_stale: int,
    confidence: float,
    recent_games: int,
) -> List[str]:
    risks: List[str] = []
    if provisional:
        risks.append(
            f"Provisional: only {recent_games} qualifying game"
            f"{'s' if recent_games != 1 else ''} of current-season evidence."
        )
    if baseline_source == "prior_season":
        risks.append("Baseline is last season's per-game usage, not this season's.")
    if baseline_source == "none":
        risks.append("No baseline available; change is measured against zero.")
    if injury_vacated:
        risks.append("Role may contract when the injured player ahead returns.")
    if spike_without_role:
        risks.append(
            "Recent fantasy points spiked without matching usage growth "
            "(efficiency/TD-driven, not a role change)."
        )
    if weeks_stale >= 1:
        risks.append(
            f"Latest usable game is {weeks_stale} week"
            f"{'s' if weeks_stale != 1 else ''} behind the cutoff."
        )
    if confidence < 40 and not provisional:
        risks.append("Low confidence: limited sample, coverage, or agreement among signals.")
    return risks
