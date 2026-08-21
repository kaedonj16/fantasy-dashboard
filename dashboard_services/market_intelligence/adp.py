from __future__ import annotations

import bisect

from .config import MIN_SIGNAL_CONFIDENCE

# The /api/league-players payload carries redraft ADP under ``redraft_avg_pick``
# (and ``sf_redraft_avg_pick`` for superflex) - the same fields the rankings,
# draft room, and cheat-sheet frontend read. Earlier this module looked only for
# ``redraft_adp``/``adp``, which the payload never sets, so every player produced
# an empty curve and a null ``market_vs_adp`` (rendered as "-"). Resolve the real
# fields first, keeping the legacy keys as fallbacks for callers/tests that pass
# a bare ``adp``.
_ADP_KEYS = ("redraft_avg_pick", "sf_redraft_avg_pick", "redraft_adp", "adp")


def _resolve_adp(player: dict) -> float | None:
    for key in _ADP_KEYS:
        value = player.get(key)
        try:
            adp = float(value)
        except (TypeError, ValueError):
            continue
        if adp > 0:
            return adp
    return None


def build_adp_curve(player_pool: list[dict]) -> tuple[list[float], list[float]]:
    """Sorted (per-game production, ADP) sample arrays for interpolation.

    Built once per pool so attach_market_vs_adp can map every player against the
    same curve in O(log n) instead of rebuilding and re-sorting it per player."""
    samples = []
    for player in player_pool:
        try:
            points = float(player.get("proj_ppg") or player.get("projected_ppg"))
        except (TypeError, ValueError):
            continue
        adp = _resolve_adp(player)
        if points > 0 and adp is not None:
            samples.append((points, adp))
    samples.sort()
    return [s[0] for s in samples], [s[1] for s in samples]


def interp_adp(curve: tuple[list[float], list[float]], market_points: float) -> float | None:
    """Piecewise-linear production-to-ADP mapping over a prebuilt curve."""
    xs, ys = curve
    if len(xs) < 2:
        return None
    target = float(market_points)
    if target <= xs[0]:
        return ys[0]
    if target >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_left(xs, target)  # xs[i-1] < target <= xs[i]
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    if x1 == x0:
        return (y0 + y1) / 2
    return y0 + (target - x0) / (x1 - x0) * (y1 - y0)


def expected_adp(market_points: float, player_pool: list[dict]) -> float | None:
    """Piecewise-linear production-to-ADP mapping, without ordinal market ranks."""
    return interp_adp(build_adp_curve(player_pool), market_points)


# A fantasy season is 17 games. The season market projection is a full-season
# point total, so divide by this to compare it against the per-game PPG curve.
_GAMES_PER_SEASON = 17


def attach_market_vs_adp(players: list[dict], projections: dict[str, dict]) -> None:
    curve = build_adp_curve(players)  # built once, mapped per player below
    for player in players:
        market = projections.get(str(player.get("id")))
        if not market:
            continue
        components = market.get("components") or {}
        basis = components.get("basis") or "season_props"
        confidence = float(market.get("confidence") or 0)
        player["market_vs_adp"] = None
        player["market_expected_adp"] = None
        player["market_confidence"] = round(confidence, 2)
        player["market_confidence_label"] = ("High" if confidence >= 0.7 else
                                             "Moderate" if confidence >= 0.5 else
                                             "Low" if confidence > 0 else "Unavailable")
        player["market_basis"] = basis
        # Baseline-only rows and weak context are useful diagnostics, not an
        # independent market edge. Do not put a number behind the Market label.
        if basis == "projection_only" or confidence < MIN_SIGNAL_CONFIDENCE:
            continue
        actual = _resolve_adp(player)
        if actual is None:
            continue
        try:
            season_points = float(market["fantasy_points"])
        except (TypeError, ValueError):
            continue
        # The curve is built from per-game PPG, so bring the market's SEASON-long
        # total onto the same per-game scale first. Passing the raw season total
        # (~250) against a per-game curve (~5-25) pinned every player to the top
        # pick's ADP.
        implied = interp_adp(curve, season_points / _GAMES_PER_SEASON)
        if implied is not None:
            player["market_expected_adp"] = round(implied, 1)
            player["market_vs_adp"] = round(actual - implied, 1)
            player["market_confidence"] = round(confidence, 2)
            player["market_signal"] = ("bullish" if actual - implied > 1 else
                                       "bearish" if actual - implied < -1 else "aligned")
