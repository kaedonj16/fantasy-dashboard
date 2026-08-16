from __future__ import annotations

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


def expected_adp(market_points: float, player_pool: list[dict]) -> float | None:
    """Piecewise-linear production-to-ADP mapping, without ordinal market ranks."""
    samples = []
    for player in player_pool:
        try:
            points = float(player.get("proj_ppg") or player.get("projected_ppg"))
        except (TypeError, ValueError):
            continue
        adp = _resolve_adp(player)
        if points > 0 and adp is not None:
            samples.append((points, adp))
    if len(samples) < 2:
        return None
    samples.sort()
    target = float(market_points)
    if target <= samples[0][0]:
        return samples[0][1]
    if target >= samples[-1][0]:
        return samples[-1][1]
    for (p0, a0), (p1, a1) in zip(samples, samples[1:]):
        if p0 <= target <= p1:
            if p1 == p0:
                return (a0 + a1) / 2
            ratio = (target - p0) / (p1 - p0)
            return a0 + ratio * (a1 - a0)
    return None


# A fantasy season is 17 games. The season market projection is a full-season
# point total, so divide by this to compare it against the per-game PPG curve.
_GAMES_PER_SEASON = 17


def attach_market_vs_adp(players: list[dict], projections: dict[str, dict]) -> None:
    for player in players:
        market = projections.get(str(player.get("id")))
        if not market:
            continue
        actual = _resolve_adp(player)
        if actual is None:
            continue
        try:
            season_points = float(market["fantasy_points"])
        except (TypeError, ValueError):
            continue
        # expected_adp's curve is built from per-game PPG, so bring the market's
        # SEASON-long total onto the same per-game scale first. Passing the raw
        # season total (~250) against a per-game curve (~5-25) pinned every player
        # to the top pick's ADP.
        implied = expected_adp(season_points / _GAMES_PER_SEASON, players)
        if implied is not None:
            player["market_expected_adp"] = round(implied, 1)
            player["market_vs_adp"] = round(actual - implied, 1)
            player["market_confidence"] = round(float(market.get("confidence") or 0), 2)
