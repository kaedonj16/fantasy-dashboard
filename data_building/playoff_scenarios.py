"""Exact playoff clinch / elimination scenarios for the regular-season stretch.

The Monte Carlo simulator (``simulate_playoff_odds``) answers *how likely* a team
is to make the playoffs. This module answers the complementary, deterministic
question managers actually ask in the final weeks:

    "Am I in yet? What do I need? Can a loss knock me out?"

It enumerates every possible combination of the remaining regular-season game
results and classifies each team against the playoff cutoff:

  * **clinched**   - in the top-N in *every* remaining scenario (a berth is
    mathematically guaranteed).
  * **eliminated** - in the top-N in *no* remaining scenario.
  * **alive**      - some scenarios in, some out; we then report the concrete
    levers: does the team control its own destiny (win out => in), the magic
    number of wins that guarantees a berth, and the punchy "win this week and
    you're in" / "lose and you're out" one-game swings.

Seeding matches the simulator: order by (wins, points-for), top N make it, the
top ``n_byes`` earn first-round byes. Points-for is the standard Sleeper
tiebreaker; since future points aren't known we hold current points-for as the
tiebreak order. Clinches almost always turn on the *wins* column, not that
proxy, so a guarantee is real in practice; the one caveat is a berth decided
purely by a points-for tie, which we treat as ordered by today's points-for.

Enumeration is 2**G in the number of remaining games G, so it is only exact for
the end-of-season window (``G <= MAX_ENUM_GAMES``). Earlier than that, callers
should fall back to the Monte Carlo odds for a probabilistic read; this module
signals that by returning ``exact=False`` with no per-team guarantees.

Division leagues seed division winners ahead of wild cards, which this simple
top-N model does not capture, so exact scenarios are skipped when divisions are
in play (``divisions=True``) - again deferring to the odds.

Pure Python, no third-party dependencies: no NumPy, no network, no DB - fully
unit-testable anywhere.
"""
from __future__ import annotations

# 2**14 = 16_384 scenarios. Above this the single-pass enumeration stops being
# cheap enough to run per request, and we're past the final-two-weeks window
# (a 12-team league plays 6 games/week, so two weeks is 12 games) where exact
# clinch math is the interesting answer.
MAX_ENUM_GAMES = 14


def compute_scenarios(
    teams: "list[dict]",
    matchups: "dict[int, list]",
    playoff_teams: int,
    *,
    n_byes: int = 0,
    divisions: bool = False,
) -> dict:
    """Classify every team's playoff standing across all remaining-game outcomes.

    Args:
        teams: dicts with ``roster_id``, ``wins``, ``pf`` (``ties`` optional).
        matchups: ``{week: [(rid_a, rid_b), ...]}`` of remaining regular-season
            games. Entries whose teams aren't both known are ignored.
        playoff_teams: number of playoff berths (top-N cutoff).
        n_byes: number of first-round byes (top seeds); enables "clinched bye".
        divisions: when True the league seeds by division and this top-N model
            doesn't apply, so we return ``exact=False``.

    Returns:
        ``{"exact": bool, "remaining_games": int, "teams": {roster_id: {...}}}``.
        Each per-team dict (only when ``exact``) carries: ``status``
        (clinched_bye / clinched / alive / eliminated), ``controls_destiny``,
        ``wins_to_clinch`` (magic number or None if help is needed),
        ``clinch_if_win_next``, ``out_if_lose_next``, ``next_game``,
        ``best_seed`` and ``worst_seed``.
    """
    idx_of = {int(t["roster_id"]): i for i, t in enumerate(teams)}
    m = len(teams)

    # Flatten remaining games, keeping week order so "next game" is well defined.
    games: list[tuple[int, int, int]] = []  # (week, a_idx, b_idx)
    for wk in sorted(matchups or {}):
        for pair in matchups[wk] or []:
            if not pair or len(pair) < 2:
                continue
            a, b = int(pair[0]), int(pair[1])
            if a in idx_of and b in idx_of and a != b:
                games.append((int(wk), idx_of[a], idx_of[b]))
    g = len(games)

    result: dict = {"exact": True, "remaining_games": g, "teams": {}}
    if divisions or g > MAX_ENUM_GAMES or m == 0 or playoff_teams <= 0:
        result["exact"] = False
        return result

    base = [float(t.get("wins", 0)) + 0.5 * float(t.get("ties", 0)) for t in teams]
    pf = [float(t.get("pf", 0.0)) for t in teams]
    # Points-for as a sub-win tiebreak: scaled below 1 so a win always outranks
    # any points-for edge, but points-for still orders teams level on wins.
    pf_denom = (max(pf) + 1.0) if pf and max(pf) > 0 else 1.0
    base_key = [base[i] + pf[i] / pf_denom for i in range(m)]

    # Each team's remaining games in week order; the first is its "next game".
    own_games: list[list[tuple[int, bool]]] = [[] for _ in range(m)]  # (game_idx, is_a)
    for gi, (_wk, a, b) in enumerate(games):
        own_games[a].append((gi, True))
        own_games[b].append((gi, False))
    next_game_idx = [og[0][0] if og else None for og in own_games]  # earliest is first added

    # Single pass over the 2**g scenarios, accumulating per-team facts.
    ever_in = [False] * m
    ever_out = [False] * m
    best_seed = [m + 1] * m
    worst_seed = [0] * m
    worst_fail_w = [-1] * m          # max own-wins seen in a scenario where team is OUT
    ever_in_bye = [False] * m
    ever_out_bye = [False] * m
    nxt_win_seen = [False] * m       # saw a scenario where team wins its next game
    nxt_win_out = [False] * m        # ...and was out in one of them
    nxt_lose_seen = [False] * m      # saw a scenario where team loses its next game
    nxt_lose_in = [False] * m        # ...and was in in one of them

    for s in range(1 << g):
        # Final win total per team for this scenario (bit gi == 1 -> team A wins).
        added = [0] * m
        for gi, (_wk, a, b) in enumerate(games):
            if (s >> gi) & 1:
                added[a] += 1
            else:
                added[b] += 1
        key = [base_key[i] + added[i] for i in range(m)]

        for i in range(m):
            ki = key[i]
            rank = 0
            for j in range(m):
                if key[j] > ki:
                    rank += 1
            seed = rank + 1
            is_in = rank < playoff_teams
            if seed < best_seed[i]:
                best_seed[i] = seed
            if seed > worst_seed[i]:
                worst_seed[i] = seed
            if is_in:
                ever_in[i] = True
            else:
                ever_out[i] = True
                if added[i] > worst_fail_w[i]:
                    worst_fail_w[i] = added[i]
            if n_byes > 0:
                if rank < n_byes:
                    ever_in_bye[i] = True
                else:
                    ever_out_bye[i] = True
            ngi = next_game_idx[i]
            if ngi is not None:
                is_a = games[ngi][1] == i
                won_next = (((s >> ngi) & 1) == 1) == is_a
                if won_next:
                    nxt_win_seen[i] = True
                    if not is_in:
                        nxt_win_out[i] = True
                else:
                    nxt_lose_seen[i] = True
                    if is_in:
                        nxt_lose_in[i] = True

    for i, t in enumerate(teams):
        rid = int(t["roster_id"])
        clinched = not ever_out[i]
        eliminated = not ever_in[i]
        g_i = len(own_games[i])

        entry: dict = {
            "best_seed": best_seed[i],
            "worst_seed": worst_seed[i],
            "controls_destiny": False,
            "wins_to_clinch": 0 if clinched else None,
            "clinch_if_win_next": False,
            "out_if_lose_next": False,
            "next_game": None,
        }

        if clinched:
            got_bye = n_byes > 0 and not ever_out_bye[i]
            entry["status"] = "clinched_bye" if got_bye else "clinched"
        elif eliminated:
            entry["status"] = "eliminated"
        else:
            entry["status"] = "alive"

        ngi = next_game_idx[i]
        if ngi is not None:
            gwk, ga, gb = games[ngi]
            opp = gb if ga == i else ga
            entry["next_game"] = {"week": gwk, "opp": int(teams[opp]["roster_id"])}
            if entry["status"] == "alive":
                entry["clinch_if_win_next"] = nxt_win_seen[i] and not nxt_win_out[i]
                entry["out_if_lose_next"] = nxt_lose_seen[i] and not nxt_lose_in[i]

        if entry["status"] == "alive":
            # Magic number: fewest own wins that guarantee a berth. If a team is
            # out even when winning all its games (worst_fail_w == g_i), no number
            # of its own wins is enough -> it needs help (None). Otherwise the
            # smallest safe threshold is one past the worst losing-scenario.
            if worst_fail_w[i] >= g_i:
                entry["wins_to_clinch"] = None
                entry["controls_destiny"] = False
            else:
                entry["wins_to_clinch"] = worst_fail_w[i] + 1
                entry["controls_destiny"] = True

        result["teams"][rid] = entry

    return result


def scenario_summary(entry: dict) -> str:
    """Short "what you need" label for a single team entry (no dashes)."""
    status = entry.get("status")
    if status == "clinched_bye":
        return "Clinched bye"
    if status == "clinched":
        return "Clinched"
    if status == "eliminated":
        return "Eliminated"
    if entry.get("clinch_if_win_next"):
        return "Win and you're in"
    if entry.get("controls_destiny"):
        n = entry.get("wins_to_clinch")
        if n:
            return f"{n} win{'s' if n != 1 else ''} to clinch"
        return "Control your destiny"
    if entry.get("out_if_lose_next"):
        return "Must win to survive"
    if entry.get("wins_to_clinch") is None:
        return "Alive, needs help"
    return "Alive"
