"""Championship odds: a SEPARATE Monte Carlo that sims only the playoffs.

This is deliberately not an extension of the regular-season sim in
``data_building/simulate_playoff_odds.py``. That sim answers "who makes
the playoffs"; this one answers "given the playoff race as it stands,
who wins the bracket". It never simulates a regular-season game.

How a run works (10,000 runs):

1. FIELD FLEX. The pool is every non-eliminated team. Clinched teams are
   in every run; eliminated teams are in no run (their title pct is
   exactly 0.0). The remaining open spots are filled per run, sampled
   without replacement from the bubble teams, weighted by their
   ``playoff_pct`` from the regular-season sim (renormalized among the
   contenders for the open spots). Clinch / elimination status comes
   from that sim's percentages, which are exact 100.0 / 0.0 only via the
   shared ``mathematical_clinch_elimination`` bounds, so there is one
   definition of clinched / eliminated across both sims.
2. SEED / BYE FLEX. Within the run's field, the 1 seed rotates among
   the top-seed race weighted by ``first_seed_pct`` and the remaining
   bye spots rotate among the bye race weighted by ``bye_pct``. A
   mathematically locked bye / 1 seed (passed in by the caller from the
   standings picture) never rotates. All other seeds follow the site's
   real seeding order (``utils.standings.playoff_seed_order``: division
   winners first when division data exists) applied to the sampled
   field, not a wins/PF-only shortcut.
3. BRACKET. Pairings follow the site's projected-bracket convention
   (``utils.standings.build_full_bracket`` canonical seed order: byes
   enter in round 2). ASSUMPTION, documented per spec: the bracket is
   fixed (no re-seeding after a round) unless league settings carry a
   detectable re-seed flag (Sleeper ``playoff_seed_type == 1``); rounds
   are one week each unless settings detectably say two weeks per round
   (Sleeper ``playoff_round_type == 2``, or ``== 1`` for a two-week
   final), in which case a round's score is the sum of two draws.
   Playoff weeks have no NFL byes after Week 14, and no Sleeper weekly
   projections exist yet for Weeks 15-17, so a round score is drawn
   from the team's own scoring distribution (the season average / std
   the regular-season sim's week profiles blend toward), never from an
   invented week-specific projection.

FAIL CLOSED on format: if the league's playoff format is not verified
(the Fleaflicker case: its API serves no playoff settings and the
adapter's 6 teams / Week 15 values are unverified defaults), this
module serves nothing for the league instead of assuming a format.

Outputs per team: ``semis_pct`` (share of runs where the team plays in
the semifinal round; a bye team starts there, a non-bye team must win
its way there, and for a two-team final the final itself is the semi),
``final_pct``, ``title_pct``, ``title_if_in_pct`` (title pct conditional
on being in that run's field) and ``in_field_pct``. Undecided values
are clamped to 0.1-99.9 like the regular sim; a title can never be a
mathematical lock before the final, so title pct never reads 100.
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import Optional

import numpy as np

from data_building.simulate_playoff_odds import _MIN_STD, _n_byes

logger = logging.getLogger(__name__)

_N_SIMS = 10_000


# ---------------------------------------------------------------------------
# Format gate (fail closed)
# ---------------------------------------------------------------------------

def bracket_format(ctx: dict, platform: str = "sleeper") -> Optional[dict]:
    """The league's verified playoff bracket format, or None when unknown.

    Returns ``{playoff_teams, weeks_per_round, final_weeks, reseed}``.
    A format counts as verified only when the league settings actually
    carry a playoff team count (and, for Fleaflicker, an explicit
    verification marker, because that adapter silently defaults the
    values its API never serves).
    """
    settings = (ctx or {}).get("league_settings") or {}
    if settings.get("playoff_format_verified") is False:
        return None
    if (platform or "").strip().lower() == "fleaflicker" and not (
        settings.get("playoff_format_verified")
    ):
        # Fleaflicker's API exposes no playoff settings at all; the
        # adapter defaults (6 teams, Week 15) are unverified, so serving
        # championship odds would mean assuming a bracket.
        return None
    try:
        playoff_teams = int(settings.get("playoff_teams") or 0)
    except (TypeError, ValueError):
        playoff_teams = 0
    if playoff_teams < 2:
        return None

    weeks_per_round = 1
    final_weeks = 1
    try:
        round_type = int(settings.get("playoff_round_type"))
    except (TypeError, ValueError):
        round_type = None
    if round_type == 2:
        weeks_per_round = 2
        final_weeks = 2
    elif round_type == 1:
        final_weeks = 2
    if int(settings.get("playoff_matchup_period_length") or 0) == 2:
        weeks_per_round = 2
        final_weeks = 2

    reseed = False
    try:
        reseed = int(settings.get("playoff_seed_type")) == 1
    except (TypeError, ValueError):
        reseed = bool(settings.get("playoff_reseed"))

    return {
        "playoff_teams": playoff_teams,
        "weeks_per_round": weeks_per_round,
        "final_weeks": final_weeks,
        "reseed": reseed,
    }


# ---------------------------------------------------------------------------
# Core sim (pure: teams + regular-sim odds in, championship rows out)
# ---------------------------------------------------------------------------

def _clamp_undecided(raw: float, *, eliminated: bool = False) -> float:
    """Playoff-odds display clamp: exact 0.0 only for eliminated teams,
    everything else inside 0.1-99.9 at one decimal."""
    if eliminated:
        return 0.0
    return round(min(99.9, max(0.1, float(raw))), 1)


def _weighted_take(rng, candidates, weights, k):
    """Sample ``k`` items from ``candidates`` without replacement,
    weighted by ``weights`` (aligned lists). Falls back to candidate
    order when every weight is zero / missing."""
    if k <= 0 or not candidates:
        return []
    if k >= len(candidates):
        return list(candidates)
    # Sequential weighted draws (k is tiny: open spots / one seed slot).
    # rng.choice(replace=False, p=...) costs far more than this loop at
    # 10k runs x several draws per run.
    pool = list(candidates)
    pw = [max(float(x or 0.0), 0.0) for x in weights]
    if sum(pw) <= 0:
        return pool[:k]
    out = []
    for _ in range(k):
        total = sum(pw)
        if total <= 0:
            out.extend(pool[:k - len(out)])
            break
        u = float(rng.random()) * total
        acc = 0.0
        pick = len(pool) - 1
        for j, x in enumerate(pw):
            acc += x
            if u < acc:
                pick = j
                break
        out.append(pool.pop(pick))
        pw.pop(pick)
    return out


def _run_one_bracket(rng, seed_rids, seed_no, mean, std, weeks_for_round,
                     total_rounds, reseed, slot_order):
    """Sim one bracket. ``seed_rids`` is the field in seed order and
    ``slot_order`` the precomputed canonical bracket slot order.

    Returns ``(champion_rid, semis_rids, final_rids)`` where semis / final
    are the sets of teams that PLAY in those rounds of this run.
    """
    # Slots in bracket order; seeds beyond the field are bye slots.
    entrants = [
        seed_rids[s - 1] if s <= len(seed_rids) else None for s in slot_order
    ]
    semis_rids: set = set()
    final_rids: set = set()
    semi_round = total_rounds - 1 if total_rounds >= 2 else 1

    rnd = 1
    current = entrants
    while len(current) > 1:
        advancing = []
        for i in range(0, len(current), 2):
            a, b = current[i], current[i + 1]
            if a is None:
                advancing.append(b)
                continue
            if b is None:
                advancing.append(a)
                continue
            if rnd == semi_round:
                semis_rids.update((a, b))
            if rnd == total_rounds:
                final_rids.update((a, b))
            weeks = weeks_for_round(rnd)
            if weeks == 1:
                sa = float(rng.normal(mean[a], std[a]))
                sb = float(rng.normal(mean[b], std[b]))
            else:
                sa = float(rng.normal(mean[a], std[a], size=weeks).sum())
                sb = float(rng.normal(mean[b], std[b], size=weeks).sum())
            if sa == sb:
                winner = a if seed_no[a] < seed_no[b] else b
            else:
                winner = a if sa > sb else b
            advancing.append(winner)
        if reseed and len(advancing) > 1:
            # Re-pair best vs worst next round: sort by seed, then
            # interleave so adjacent pairs are (best, worst),
            # (2nd best, 2nd worst), and so on.
            advancing.sort(key=lambda rid: seed_no[rid])
            ordered = []
            pool = list(advancing)
            while pool:
                ordered.append(pool.pop(0))
                if pool:
                    ordered.append(pool.pop(-1))
            advancing = ordered
        current = advancing
        rnd += 1
    champion = current[0] if current else None
    if total_rounds == 1:
        semis_rids = set(final_rids)
    return champion, semis_rids, final_rids


def simulate_playoff_bracket(
    teams: list,
    odds_by_rid: dict,
    *,
    playoff_teams: int,
    n_sims: int = _N_SIMS,
    seed: Optional[int] = None,
    weeks_per_round: int = 1,
    final_weeks: Optional[int] = None,
    reseed: bool = False,
    locked_bye_ids=(),
    locked_first_id=None,
) -> list:
    """Run the playoff-only Monte Carlo.

    ``teams`` are standings rows: roster_id, name, wins, losses, ties,
    pf, pa, avg (scoring mean), std, and optionally division /
    div_record. ``odds_by_rid`` maps roster_id to the regular-season
    sim's row (playoff_pct / bye_pct / first_seed_pct, optionally
    explicit clinched / eliminated flags).

    Deterministic for a given ``seed``.
    """
    from utils.standings import _canonical_seed_order, playoff_seed_order

    if not teams or playoff_teams < 2:
        return []
    if seed is None:
        seed = 20261103
    rng = np.random.default_rng(int(seed))

    n_byes = _n_byes(playoff_teams)
    bracket_size = 1 << int(math.ceil(math.log2(max(playoff_teams, 2))))
    total_rounds = int(round(math.log2(bracket_size)))
    slot_order = _canonical_seed_order(bracket_size)
    if final_weeks is None:
        final_weeks = weeks_per_round

    def _odds(t):
        return (odds_by_rid or {}).get(t["roster_id"]) or {}

    def _status(t):
        o = _odds(t)
        if "clinched" in o or "eliminated" in o:
            return bool(o.get("clinched")), bool(o.get("eliminated"))
        pct = float(o.get("playoff_pct") or 0.0)
        return pct >= 100.0, pct <= 0.0

    # Standings seed order over the whole pool (division winners first).
    seed_rows = [{
        "wins": t.get("wins", 0), "ties": t.get("ties", 0),
        "pf": t.get("pf", 0), "pa": t.get("pa", 0),
        "division": t.get("division") or 0,
        "div_record": t.get("div_record"),
    } for t in teams]
    pool_order = playoff_seed_order(seed_rows)  # indices into teams
    pool_rank = {idx: rank for rank, idx in enumerate(pool_order)}

    clinched_idx, bubble_idx, eliminated_idx = [], [], []
    for i, t in enumerate(teams):
        c, e = _status(t)
        if e:
            eliminated_idx.append(i)
        elif c:
            clinched_idx.append(i)
        else:
            bubble_idx.append(i)
    bubble_idx.sort(key=lambda i: pool_rank[i])
    clinched_idx.sort(key=lambda i: pool_rank[i])

    mean = {t["roster_id"]: float(t.get("avg") or t.get("mean") or 80.0)
            for t in teams}
    std = {}
    for t in teams:
        s = float(t.get("std") or 0.0)
        std[t["roster_id"]] = s if s > 0 else _MIN_STD
    locked_bye = {int(x) for x in (locked_bye_ids or ())}
    locked_first = (
        int(locked_first_id) if locked_first_id is not None else None
    )

    counts = {
        t["roster_id"]: {"in": 0, "semis": 0, "final": 0, "title": 0}
        for t in teams
    }

    open_spots = playoff_teams - len(clinched_idx)
    bubble_weights = [
        float(_odds(teams[i]).get("playoff_pct") or 0.0) for i in bubble_idx
    ]

    def weeks_for_round(r):
        return final_weeks if r == total_rounds else weeks_per_round

    for _ in range(int(n_sims)):
        # ── Field ──────────────────────────────────────────────────────
        if open_spots >= len(bubble_idx):
            field = list(clinched_idx) + list(bubble_idx)
        elif open_spots > 0:
            field = list(clinched_idx) + _weighted_take(
                rng, bubble_idx, bubble_weights, open_spots,
            )
        else:
            # More clinched than spots cannot happen by the bounds, but
            # never crash on inconsistent inputs: keep the top seeds.
            field = sorted(clinched_idx, key=lambda i: pool_rank[i])[
                :playoff_teams
            ]
        if len(field) < 2:
            continue
        field.sort(key=lambda i: pool_rank[i])
        field_rids = [teams[i]["roster_id"] for i in field]
        field_pos = {rid: pos for pos, rid in enumerate(field_rids)}

        # ── Seeds ──────────────────────────────────────────────────────
        seeds: list = [None] * len(field)
        remaining = list(field_rids)  # standings order within the field

        def _take(cands, weights):
            got = _weighted_take(rng, cands, weights, 1)
            return got[0] if got else (cands[0] if cands else None)

        if seeds:
            # 1 seed: locked, else weighted among the top-seed race.
            if locked_first is not None and locked_first in remaining:
                first = locked_first
            else:
                race = [
                    r for r in remaining
                    if float(_odds({"roster_id": r}).get("first_seed_pct")
                             or 0.0) > 0
                    or field_pos[r] < 2
                ] or list(remaining)
                first = _take(race, [
                    float(_odds({"roster_id": r}).get("first_seed_pct")
                          or 0.0) for r in race
                ])
            seeds[0] = first
            remaining.remove(first)
            # Remaining bye spots.
            for pos in range(1, min(n_byes, len(seeds))):
                locked_here = [r for r in remaining if r in locked_bye]
                if locked_here:
                    pick = locked_here[0]
                else:
                    race = [
                        r for r in remaining
                        if float(_odds({"roster_id": r}).get("bye_pct")
                                 or 0.0) > 0
                        or field_pos[r] <= n_byes + 1
                    ] or list(remaining)
                    pick = _take(race, [
                        float(_odds({"roster_id": r}).get("bye_pct") or 0.0)
                        for r in race
                    ])
                seeds[pos] = pick
                remaining.remove(pick)
            for pos in range(len(seeds)):
                if seeds[pos] is None:
                    seeds[pos] = remaining.pop(0)

        seed_no = {rid: i + 1 for i, rid in enumerate(seeds)}
        for rid in field_rids:
            counts[rid]["in"] += 1
        champion, semis, finals = _run_one_bracket(
            rng, seeds, seed_no, mean, std, weeks_for_round,
            total_rounds, reseed, slot_order,
        )
        for rid in semis:
            counts[rid]["semis"] += 1
        for rid in finals:
            counts[rid]["final"] += 1
        if champion is not None:
            counts[champion]["title"] += 1

    out = []
    elim_set = {teams[i]["roster_id"] for i in eliminated_idx}
    for t in teams:
        rid = t["roster_id"]
        c = counts[rid]
        elim = rid in elim_set
        in_runs = c["in"]
        out.append({
            "roster_id": rid,
            "team_name": t.get("name"),
            "semis_pct": _clamp_undecided(
                100.0 * c["semis"] / n_sims, eliminated=elim),
            "final_pct": _clamp_undecided(
                100.0 * c["final"] / n_sims, eliminated=elim),
            "title_pct": _clamp_undecided(
                100.0 * c["title"] / n_sims, eliminated=elim),
            "title_if_in_pct": _clamp_undecided(
                (100.0 * c["title"] / in_runs) if in_runs else 0.0,
                eliminated=elim),
            "in_field_pct": _clamp_undecided(
                100.0 * in_runs / n_sims, eliminated=elim),
            "n_sims": int(n_sims),
        })
    return out


# ---------------------------------------------------------------------------
# League-context wrapper
# ---------------------------------------------------------------------------

def simulate_championship_odds(
    ctx: dict,
    odds_rows: list,
    platform: str = "sleeper",
    n_sims: int = _N_SIMS,
    seed: Optional[int] = None,
) -> list:
    """Championship rows for a league ctx + its regular-season odds rows.

    Returns [] when the bracket format is unverified (fail closed), when
    the season has not started (title odds are gated off then anyway),
    or when there is nothing to sim.
    """
    fmt = bracket_format(ctx, platform)
    if fmt is None or not odds_rows:
        return []

    from data_building.simulate_playoff_odds import (
        _build_teams, team_stats_have_played_games,
    )

    team_stats = (ctx or {}).get("team_stats")
    is_complete = bool(odds_rows and odds_rows[0].get("is_complete"))
    if not is_complete and not team_stats_have_played_games(team_stats):
        return []
    teams = _build_teams(team_stats, (ctx or {}).get("roster_map") or {})
    if len(teams) < 2:
        return []

    settings = (ctx or {}).get("league_settings") or {}
    # Division context, built the same way the regular sim builds it.
    division_map: dict = {}
    n_divisions = int(settings.get("divisions") or 0)
    if n_divisions >= 2:
        for r in (ctx or {}).get("rosters") or []:
            rid = r.get("roster_id")
            if rid is not None:
                try:
                    division_map[int(rid)] = int(
                        (r.get("settings") or {}).get("division") or 1)
                except (TypeError, ValueError):
                    pass
    div_records = None
    if division_map:
        try:
            from utils.standings import division_records_for_ctx
            div_records = division_records_for_ctx(ctx)
        except Exception:
            logger.debug("championship: div records skipped", exc_info=True)
    for t in teams:
        if division_map:
            t["division"] = division_map.get(t["roster_id"], 0)
        if div_records and t["roster_id"] in div_records:
            t["div_record"] = div_records[t["roster_id"]]

    # Locked bye / 1 seed from the standings picture (its BYE status is
    # the safe locked-bye test); absent data simply means no locks.
    locked_bye_ids: list = []
    locked_first_id = None
    try:
        from utils.standings import compute_playoff_picture, BYE as _BYE
        total_weeks = max(
            0, int(settings.get("playoff_week_start") or 15) - 1)
        pic = compute_playoff_picture(
            [{
                "id": t["roster_id"], "wins": t["wins"],
                "losses": t["losses"], "ties": t["ties"], "pf": t["pf"],
                "division": t.get("division") or 0,
                "div_record": t.get("div_record"),
            } for t in teams],
            playoff_spots=fmt["playoff_teams"],
            total_regular_weeks=total_weeks,
        )
        by_rid = {row["id"]: row for row in (pic or [])}
        ranked = sorted(
            teams, key=lambda t: (-(t["wins"] + 0.5 * t["ties"]), -t["pf"]))
        if ranked and by_rid.get(ranked[0]["roster_id"], {}).get(
                "status") == _BYE:
            locked_first_id = ranked[0]["roster_id"]
        for t in teams:
            if by_rid.get(t["roster_id"], {}).get("status") == _BYE:
                locked_bye_ids.append(t["roster_id"])
    except Exception:
        logger.debug("championship: locked seeds skipped", exc_info=True)

    odds_by_rid = {
        row.get("roster_id"): row for row in odds_rows
        if row and row.get("roster_id") is not None
    }
    if seed is None:
        league_id = str((ctx or {}).get("league_id") or "")
        season = int((ctx or {}).get("season") or 0)
        if league_id:
            seed = int(hashlib.md5(
                f"champ:{league_id}:{season}".encode()).hexdigest(),
                16) % (2 ** 32)
    return simulate_playoff_bracket(
        teams, odds_by_rid,
        playoff_teams=fmt["playoff_teams"],
        n_sims=n_sims, seed=seed,
        weeks_per_round=fmt["weeks_per_round"],
        final_weeks=fmt["final_weeks"],
        reseed=fmt["reseed"],
        locked_bye_ids=locked_bye_ids,
        locked_first_id=locked_first_id,
    )
