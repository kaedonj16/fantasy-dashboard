"""Shared "unexpected big game" detector and sustainability classifier.

A player who erupts for a big week is only a *waiver* story if two separate
things are true, and this module keeps them separate on purpose (they answer
different questions and must not be blurred into one number):

  1. **Performance surprise** — did the game genuinely exceed what was expected
     of the player *going in*? This is a league-scored comparison of realized
     fantasy points against a saved pregame projection (preferred) or, failing
     that, a clearly-labeled historical baseline. It rewards both *relative*
     surprise (beating expectation) and *absolute* production (so a 2-ppg player
     popping for 12 is not treated as an eight-alarm breakout).

  2. **Role sustainability** — will the usage that produced it continue? This is
     a raw role/usage read (snap share, targets, target share, routes, carries,
     touches, red-zone / goal-line work, and whether a teammate injury opened
     the door) plus cautions when the fantasy day leaned on touchdowns, freak
     efficiency, or one explosive play.

The two feed an explainable **classification**:

  * ``priority``    — strong surprise, credible sustainable role.
  * ``speculative`` — encouraging, but the role evidence is thin or uncertain.
  * ``watchlist``   — surprising box score with little sign the role sticks
                      (the classic fluky-touchdown line).

Everything here is pure and league-scoring-agnostic on the role side: fantasy
points must already be scored for the league by the caller (item 6), while the
usage features are raw counts/shares that mean the same in any league. The
module never invents data — a missing usage feature reads as *role unconfirmed*,
never as zero opportunity (item 8) — and never emits a probability: the scores
are bounded 0..1 confidences, labeled as such, not calibrated likelihoods.

League-specific availability and roster fit are deliberately *not* computed
here; the waiver route layers those on so the shared detector can be reused by
the digest, dashboard, and notifications without dragging a league in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.draft_grade import clamp01 as _clamp01


# ---------------------------------------------------------------------------
# Tunables (one surface, mirroring utils.waiver_score.WEIGHTS)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BigGameConfig:
    # Shrinkage added to the expectation when computing relative surprise, so a
    # tiny baseline can't manufacture an enormous ratio (a 2->18 game is a real
    # surprise, but not 8x a 12->28 game). Points.
    surprise_shrink_pts: float = 8.0
    # Relative-surprise points that map to a full 1.0 relative component.
    surprise_full_ratio: float = 1.6
    # Absolute production gate: a game below `abs_floor` league points isn't a
    # "big game" no matter how far it beat a microscopic projection; `abs_ceiling`
    # is a genuinely elite day.
    abs_floor: float = 10.0
    abs_ceiling: float = 30.0
    # Weight of the absolute vs relative component in the surprise blend.
    abs_weight: float = 0.45
    # Usage-sustainability: snap-share delta (fraction, 0..1) that reads as a full
    # role jump, and target/route/touch deltas that read as meaningful growth.
    snap_share_full_delta: float = 0.25
    target_share_full_delta: float = 0.12
    routes_full_delta: float = 12.0
    touches_full_delta: float = 8.0
    targets_full_delta: float = 5.0
    # Fraction of fantasy points from TDs above which the day is TD-dependent.
    td_dependence_frac: float = 0.5
    # Fraction of receiving/rushing yards from the single longest play above which
    # the day leaned on one explosive gain.
    explosive_play_frac: float = 0.45
    # Yards per touch / per route above which efficiency is unsustainable.
    yards_per_touch_hot: float = 9.0
    yards_per_route_hot: float = 3.2
    # Games of history at/under which a baseline is "uncertain" (rookies, returns).
    thin_history_games: int = 3
    # Classification thresholds on the (surprise, sustainability) plane.
    priority_surprise: float = 0.55
    priority_sustain: float = 0.5
    speculative_surprise: float = 0.35
    # A strongly sustainable role can surface a discovery even when the single-game
    # surprise was only moderate (the "fewer points, real usage" case), provided
    # the game cleared a minimum surprise and a minimum absolute production.
    role_surface_min: float = 0.2
    role_surface_abs: float = 0.2
    # Sustainability credited to a confirmed teammate-vacancy alone (no usage data
    # yet), bounded because the starter may return.
    vacancy_sustain_max: float = 0.6


CONFIG = BigGameConfig()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GameContext:
    """One player's single-game performance and the context needed to judge it.

    Fantasy-point fields (``actual_points``, ``pregame_projection``,
    ``baseline_ppg``, ``position_baseline_ppg``) must already be scored for the
    target league by the caller. Usage fields are raw and league-agnostic; every
    one is optional and ``None`` means *unknown / not yet reported* — never zero.
    """
    player_id: str
    position: str
    season: int
    week: int

    # League-scored fantasy points.
    actual_points: Optional[float] = None
    # Saved pregame projection snapshot — the only honest expectation. When
    # absent we fall back to the historical baseline and say so.
    pregame_projection: Optional[float] = None
    projection_source: Optional[str] = None      # e.g. "sleeper", "espn"
    projection_saved_at: Optional[str] = None     # ISO ts; None => no snapshot

    # Established historical baseline (labeled), used only when no snapshot.
    baseline_ppg: Optional[float] = None
    baseline_source: Optional[str] = None          # "season_avg" | "trailing4" | "career"
    baseline_games: Optional[int] = None            # sample size behind the baseline
    # Startable/replacement production at the position (position-relative read).
    position_baseline_ppg: Optional[float] = None

    # Raw usage (all optional; None => unconfirmed).
    snap_share: Optional[float] = None              # 0..1 this game
    snap_share_prev: Optional[float] = None         # 0..1 prior baseline
    routes: Optional[float] = None
    routes_prev: Optional[float] = None
    targets: Optional[float] = None
    targets_prev: Optional[float] = None
    target_share: Optional[float] = None            # 0..1
    target_share_prev: Optional[float] = None
    carries: Optional[float] = None
    carries_prev: Optional[float] = None
    touches: Optional[float] = None
    touches_prev: Optional[float] = None
    redzone_touches: Optional[float] = None
    goalline_touches: Optional[float] = None

    # Efficiency / explosiveness context.
    touchdowns: Optional[float] = None
    td_points: Optional[float] = None               # league points from TDs, if known
    total_yards: Optional[float] = None
    longest_play_yards: Optional[float] = None
    yards_per_touch: Optional[float] = None
    yards_per_route: Optional[float] = None

    # Opportunity context.
    teammate_out: bool = False                       # a starter ahead is confirmed out
    teammate_out_share: Optional[float] = None       # role fraction plausibly freed (committee-aware)

    # Uncertainty flags.
    is_rookie: bool = False
    returning_from_injury: bool = False
    limited_history: bool = False

    # Lifecycle (item 8): "final" | "in_progress".
    status: str = "final"


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BigGameAssessment:
    player_id: str
    season: int
    week: int
    category: str                       # priority | speculative | watchlist | none
    performance_surprise: float          # 0..1 confidence (NOT a probability)
    role_sustainability: float           # 0..1 confidence
    absolute_score: float                # 0..1 how big the raw box score was
    expectation: Optional[float]         # pts the surprise was measured against
    expectation_basis: str               # "pregame_projection" | "baseline:<src>" | "position" | "none"
    role_confirmed: bool                 # did we have real usage data?
    cautions: tuple = ()                 # e.g. ("td_dependent", "one_big_play")
    factors: tuple = ()                  # human-readable evidence, ranked
    status: str = "final"                # "final" | "in_progress" (provisional)

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "season": self.season,
            "week": self.week,
            "category": self.category,
            "performance_surprise": round(self.performance_surprise, 3),
            "role_sustainability": round(self.role_sustainability, 3),
            "absolute_score": round(self.absolute_score, 3),
            "expectation": (round(self.expectation, 1) if self.expectation is not None else None),
            "expectation_basis": self.expectation_basis,
            "role_confirmed": self.role_confirmed,
            "cautions": list(self.cautions),
            "factors": list(self.factors),
            "status": self.status,
        }


# ---------------------------------------------------------------------------
# Expectation resolution (item 6: snapshots first, labeled baseline otherwise)
# ---------------------------------------------------------------------------

def resolve_expectation(g: GameContext) -> "tuple[Optional[float], str, bool]":
    """Return (expectation_points, basis_label, is_uncertain).

    A *saved pregame projection* is authoritative — it's the only thing that was
    truly "expected" before the game. Absent one, we use a clearly-labeled
    historical baseline, and absent that, the position baseline. Rookies, players
    with thin history, and injury returns are flagged uncertain so callers never
    treat an invented baseline as established.
    """
    uncertain = bool(g.is_rookie or g.returning_from_injury or g.limited_history)
    if g.pregame_projection is not None and g.projection_saved_at:
        return (float(g.pregame_projection), "pregame_projection", uncertain)
    if g.baseline_ppg is not None:
        games = g.baseline_games
        if games is not None and games <= CONFIG.thin_history_games:
            uncertain = True
        src = g.baseline_source or "baseline"
        return (float(g.baseline_ppg), f"baseline:{src}", uncertain)
    if g.position_baseline_ppg is not None:
        return (float(g.position_baseline_ppg), "position", True)
    return (None, "none", True)


# ---------------------------------------------------------------------------
# Performance surprise (relative + absolute, so tiny baselines don't explode)
# ---------------------------------------------------------------------------

def absolute_component(actual: Optional[float], cfg: BigGameConfig = CONFIG) -> float:
    """0..1 for how big the raw box score was, independent of expectation."""
    if actual is None:
        return 0.0
    return _clamp01((float(actual) - cfg.abs_floor) / max(1e-6, cfg.abs_ceiling - cfg.abs_floor))


def relative_component(actual: Optional[float], expectation: Optional[float],
                       cfg: BigGameConfig = CONFIG) -> float:
    """0..1 for how far the game beat expectation, shrunk so a microscopic
    expectation can't produce an absurd ratio. Falls back to the absolute read
    when there's no expectation at all."""
    if actual is None:
        return 0.0
    if expectation is None:
        return absolute_component(actual, cfg)
    over = float(actual) - float(expectation)
    if over <= 0:
        return 0.0
    ratio = over / (max(0.0, float(expectation)) + cfg.surprise_shrink_pts)
    return _clamp01(ratio / cfg.surprise_full_ratio)


def performance_surprise(g: GameContext, cfg: BigGameConfig = CONFIG) -> "tuple[float, float]":
    """Blend relative and absolute surprise into a single 0..1 confidence.

    Both matter: a huge *relative* jump off a tiny base still needs real absolute
    production to count, and a merely-good absolute day that was fully expected is
    no surprise. Returns ``(surprise, absolute_score)``.
    """
    rel = relative_component(g.actual_points, _resolve_pts(g), cfg)
    ab = absolute_component(g.actual_points, cfg)
    # Blend relative surprise with absolute production: a huge *ratio* off a tiny
    # box score (2->14) lands below a big absolute day (18->31), while the
    # shrinkage in `relative_component` keeps a real low-baseline breakout in play.
    surprise = _clamp01((1.0 - cfg.abs_weight) * rel + cfg.abs_weight * ab)
    return surprise, ab


def _resolve_pts(g: GameContext) -> Optional[float]:
    exp, _basis, _unc = resolve_expectation(g)
    return exp


# ---------------------------------------------------------------------------
# Role sustainability (raw usage; missing => unconfirmed, never zero)
# ---------------------------------------------------------------------------

def _delta_component(cur, prev, full_delta) -> Optional[float]:
    """0..1 growth signal for one usage stat, or None when either side is
    unknown (role unconfirmed on that axis — never scored as zero growth)."""
    if cur is None or prev is None:
        return None
    try:
        d = float(cur) - float(prev)
    except (TypeError, ValueError):
        return None
    if d <= 0:
        return 0.0
    return _clamp01(d / full_delta)


def _level_component(cur, full_level) -> Optional[float]:
    """0..1 absolute-level signal (e.g. an 80% snap share is a real role even
    with no prior to compare), or None when unknown."""
    if cur is None:
        return None
    try:
        return _clamp01(float(cur) / full_level)
    except (TypeError, ValueError):
        return None


def role_sustainability(g: GameContext, cfg: BigGameConfig = CONFIG) -> "tuple[float, bool, tuple, tuple]":
    """Return ``(sustainability, role_confirmed, cautions, factors)``.

    Sustainability rises with real, *rising* opportunity (snap share, targets,
    target share, routes, touches) and dedicated high-value work (red zone /
    goal line). It's discounted — not inflated — by cautions: a TD-dependent day,
    one explosive play, or unsustainable efficiency. When no usage data is
    available at all the role is *unconfirmed* (item 8): sustainability stays low
    and the caller should not claim a lasting role.
    """
    cfg = cfg or CONFIG
    growth: list = []
    factors: list = []

    snap_growth = _delta_component(g.snap_share, g.snap_share_prev, cfg.snap_share_full_delta)
    snap_level = _level_component(g.snap_share, 0.85)
    tgt_share_growth = _delta_component(g.target_share, g.target_share_prev, cfg.target_share_full_delta)
    tgt_growth = _delta_component(g.targets, g.targets_prev, cfg.targets_full_delta)
    route_growth = _delta_component(g.routes, g.routes_prev, cfg.routes_full_delta)
    touch_growth = _delta_component(g.touches, g.touches_prev, cfg.touches_full_delta)
    carry_growth = _delta_component(g.carries, g.carries_prev, cfg.touches_full_delta)

    def _note(val, label):
        if val is not None and val > 0.15:
            factors.append((val, label))

    _note(snap_growth, _fmt_delta("snap share", g.snap_share_prev, g.snap_share, pct=True))
    _note(tgt_share_growth, _fmt_delta("target share", g.target_share_prev, g.target_share, pct=True))
    _note(tgt_growth, _fmt_delta("targets", g.targets_prev, g.targets))
    _note(route_growth, _fmt_delta("routes", g.routes_prev, g.routes))
    _note(touch_growth, _fmt_delta("touches", g.touches_prev, g.touches))
    _note(carry_growth, _fmt_delta("carries", g.carries_prev, g.carries))

    for c in (snap_growth, tgt_share_growth, tgt_growth, route_growth, touch_growth, carry_growth):
        if c is not None:
            growth.append(c)

    # High absolute role even without a prior comparison (e.g. an 85% snap share
    # the first week a starter is out) is itself a sustainability signal.
    level_signals = [x for x in (snap_level, _level_component(g.target_share, 0.28)) if x is not None]

    if g.redzone_touches and g.redzone_touches >= 2:
        factors.append((0.5, f"{int(g.redzone_touches)} red-zone touches"))
    if g.goalline_touches and g.goalline_touches >= 1:
        factors.append((0.55, f"{int(g.goalline_touches)} goal-line touches"))

    # Opportunity from a confirmed teammate absence, committee-adjusted: we do NOT
    # assume the whole workload transfers to one player (item 6). It's bounded
    # because the starter may return, and counts as opportunity even before any
    # usage data lands.
    opp = 0.0
    if g.teammate_out:
        share = g.teammate_out_share
        if share is None:
            share = 0.5  # conservative committee split when unknown
        opp = _clamp01(float(share))
        factors.append((0.4 + 0.3 * opp, "starter ahead ruled out"))
    opp_sustain = cfg.vacancy_sustain_max * opp

    # A real, rising or high-level usage read confirms the role; a confirmed
    # vacancy is opportunity even without usage data yet.
    role_confirmed = bool(growth) or bool(level_signals)

    base_signals = growth + level_signals
    base = max(base_signals) if base_signals else 0.0
    # Corroboration: a second independent rising signal lifts confidence with
    # diminishing returns (mirrors the opportunity-combine in waiver_score, so we
    # don't double-count correlated usage stats).
    corrob = sorted(base_signals, reverse=True)
    combined = base + (0.35 * corrob[1] if len(corrob) > 1 else 0.0)
    sustain = _clamp01(max(combined, opp_sustain))

    # Cautions reduce sustainability.
    cautions: list = []
    td_frac = _td_fraction(g)
    if td_frac is not None and td_frac >= cfg.td_dependence_frac:
        cautions.append("td_dependent")
        sustain *= 0.55
    exp_frac = _explosive_fraction(g)
    if exp_frac is not None and exp_frac >= cfg.explosive_play_frac:
        cautions.append("one_big_play")
        sustain *= 0.6
    if _hot_efficiency(g, cfg):
        cautions.append("hot_efficiency")
        sustain *= 0.7
    if not role_confirmed:
        cautions.append("role_unconfirmed")
        # No usage data: a confirmed vacancy still carries (bounded) opportunity,
        # but with nothing at all the role stays near-zero (never negative).
        if opp <= 0:
            sustain = min(sustain, 0.2)
    if g.is_rookie or g.returning_from_injury or g.limited_history:
        cautions.append("uncertain_baseline")

    factors.sort(key=lambda t: t[0], reverse=True)
    factor_labels = tuple(lbl for _s, lbl in factors)
    return _clamp01(sustain), role_confirmed, tuple(cautions), factor_labels


def _td_fraction(g: GameContext) -> Optional[float]:
    if g.actual_points is None or g.actual_points <= 0:
        return None
    if g.td_points is not None:
        return _clamp01(float(g.td_points) / float(g.actual_points))
    if g.touchdowns is not None:
        # Assume ~6 fantasy pts per TD (offensive skill) when explicit TD points
        # aren't provided; clamp so it stays a fraction.
        return _clamp01(float(g.touchdowns) * 6.0 / float(g.actual_points))
    return None


def _explosive_fraction(g: GameContext) -> Optional[float]:
    if not g.total_yards or g.longest_play_yards is None:
        return None
    try:
        return _clamp01(float(g.longest_play_yards) / float(g.total_yards))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _hot_efficiency(g: GameContext, cfg: BigGameConfig) -> bool:
    if g.yards_per_touch is not None and g.yards_per_touch >= cfg.yards_per_touch_hot:
        return True
    if g.yards_per_route is not None and g.yards_per_route >= cfg.yards_per_route_hot:
        return True
    return False


def _fmt_delta(label, prev, cur, pct=False) -> str:
    """Concise evidence string like 'snap share 41%->78%' or 'targets 3->9'."""
    def _f(x):
        if x is None:
            return "?"
        if pct:
            return f"{round(float(x) * 100)}%"
        return f"{round(float(x))}"
    return f"{label} {_f(prev)}->{_f(cur)}"


# ---------------------------------------------------------------------------
# Classification (explainable categories, item 7)
# ---------------------------------------------------------------------------

def classify(surprise: float, sustainability: float, absolute: float,
             cfg: BigGameConfig = CONFIG) -> str:
    """Map the (surprise, sustainability) plane onto an explainable category.

    priority    — the game was a real surprise AND the role looks like it sticks.
    speculative — encouraging surprise with moderate/uncertain role evidence.
    watchlist   — a surprising box score whose role evidence is thin (fluky).
    none        — not surprising enough to surface.
    """
    # A credible, sustainable role can surface a discovery even when the single
    # game surprise was only moderate ("fewer points, real usage"), as long as
    # the game cleared a floor of surprise and real production.
    role_surface = (surprise >= cfg.role_surface_min
                    and sustainability >= cfg.priority_sustain
                    and absolute >= cfg.role_surface_abs)
    if surprise < cfg.speculative_surprise and not role_surface:
        return "none"
    if sustainability >= cfg.priority_sustain and (surprise >= cfg.priority_surprise
                                                   or role_surface):
        return "priority"
    if surprise >= cfg.speculative_surprise:
        return "speculative" if sustainability >= 0.3 else "watchlist"
    return "watchlist"


def assess_big_game(g: GameContext, cfg: BigGameConfig = CONFIG) -> BigGameAssessment:
    """Full assessment for one player-game. Pure; safe on partial data."""
    cfg = cfg or CONFIG
    exp, basis, _uncertain = resolve_expectation(g)
    surprise, absolute = performance_surprise(g, cfg)
    sustain, role_confirmed, cautions, role_factors = role_sustainability(g, cfg)
    category = classify(surprise, sustain, absolute, cfg)

    factors: list = []
    if g.actual_points is not None:
        if exp is not None:
            factors.append(f"{round(g.actual_points, 1)} pts vs {round(exp, 1)} expected "
                           f"({_basis_phrase(basis)})")
        else:
            factors.append(f"{round(g.actual_points, 1)} pts (no pregame projection saved)")
    factors.extend(role_factors)
    if "td_dependent" in cautions:
        factors.append("leaned on touchdowns")
    if "one_big_play" in cautions:
        factors.append("one long play drove the yardage")
    if "hot_efficiency" in cautions:
        factors.append("efficiency unlikely to hold")
    if "role_unconfirmed" in cautions:
        factors.append("snap/usage not yet reported")

    return BigGameAssessment(
        player_id=g.player_id,
        season=g.season,
        week=g.week,
        category=category,
        performance_surprise=surprise,
        role_sustainability=sustain,
        absolute_score=absolute,
        expectation=exp,
        expectation_basis=basis,
        role_confirmed=role_confirmed,
        cautions=cautions,
        factors=tuple(factors),
        status=g.status or "final",
    )


def _basis_phrase(basis: str) -> str:
    if basis == "pregame_projection":
        return "pregame projection"
    if basis.startswith("baseline:"):
        return f"{basis.split(':', 1)[1]} baseline"
    if basis == "position":
        return "positional baseline"
    return "no baseline"


# ---------------------------------------------------------------------------
# Idempotent lifecycle (item 8: live -> final -> stat-correction, no dupes)
# ---------------------------------------------------------------------------

def discovery_key(player_id, season, week) -> str:
    """Stable identity for one player-game discovery, so re-running during a game,
    after final stats, and after stat corrections updates one row rather than
    creating duplicates."""
    return f"{season}:{int(week)}:{player_id}"


# Rank of lifecycle states — a later/more-final state supersedes an earlier one.
_STATUS_RANK = {"in_progress": 0, "final": 1, "corrected": 2}


def merge_assessment(existing: Optional[BigGameAssessment],
                     incoming: BigGameAssessment) -> BigGameAssessment:
    """Idempotently fold a new assessment into an existing discovery for the same
    player-game. The more-final status wins; an equal-or-newer status refreshes
    the numbers in place. Never produces a duplicate — callers key on
    ``discovery_key`` and store the merged result."""
    if existing is None:
        return incoming
    if discovery_key(existing.player_id, existing.season, existing.week) != \
       discovery_key(incoming.player_id, incoming.season, incoming.week):
        return incoming
    old_rank = _STATUS_RANK.get(existing.status, 1)
    new_rank = _STATUS_RANK.get(incoming.status, 1)
    if new_rank < old_rank:
        # A late live-scored update arriving after final stats: keep the final one.
        return existing
    return incoming
