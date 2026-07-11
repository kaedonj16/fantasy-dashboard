"""Data-driven backtest for the draft pick-score weights.

The pick-score weights in ``utils.pick_score.PS_WEIGHTS`` (and their JS mirror)
were hand-tuned. This harness makes them *falsifiable*: given past completed
drafts and how those teams actually finished, it measures how well a team's
average pick score predicts real success, and sweeps candidate weight tables to
find the one that predicts best.

Design goals:
  * The correlation/sweep core is pure Python (no pandas/DB/network), so it runs
    and is unit-tested anywhere — see tests/test_draft_grade_backtest.py.
  * ``compute_pick_score`` already accepts a ``weights`` override (defaulting to
    the shipped table), so sweeping never mutates the live weights and the JS
    parity test is unaffected.
  * The real-data path (``load_sleeper_samples``) is documented and defensive:
    it pulls past seasons' drafts + final standings from Sleeper and looks each
    picked player's valuation up through an injected ``value_fn`` (so the DB
    dependency is a caller concern, not baked in). With no network/DB it returns
    ``[]`` rather than raising, so importing/using the module offline is safe.

Typical use (with data + DB access):

    from data_building.draft_grade_backtest import (
        load_sleeper_samples, sweep, correlate_grades_to_finish, WEIGHT_CANDIDATES)

    samples = load_sleeper_samples(league_ids, season, value_fn=my_lookup)
    print("baseline r =", correlate_grades_to_finish(samples))   # shipped weights
    for label, w, r in sweep(samples, WEIGHT_CANDIDATES):
        print(f"{r:+.3f}  {label}")
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from utils.pick_score import PS_WEIGHTS, compute_pick_score


# --------------------------------------------------------------------------- #
# Sample model
# --------------------------------------------------------------------------- #
@dataclass
class TeamSample:
    """One team's draft in a *completed* league.

    ``picks`` is a list of kwargs dicts for ``compute_pick_score`` (everything
    except ``weights``). ``outcome`` is a success metric where HIGHER IS BETTER
    (season points-for, or an inverted final rank via ``outcome_from_rank``), so
    a good weight table yields a POSITIVE grade-vs-outcome correlation.
    """
    picks: List[dict]
    outcome: float
    label: str = ""
    weight: float = 1.0
    meta: Dict = field(default_factory=dict)


def outcome_from_rank(final_rank: int, num_teams: int) -> float:
    """Convert a final standing (1 = champion) into a higher-is-better outcome.

    ``num_teams`` down to 1: the champion scores ``num_teams``, last place 1.
    """
    n = max(int(num_teams), 1)
    r = min(max(int(final_rank), 1), n)
    return float(n - r + 1)


# --------------------------------------------------------------------------- #
# Pure statistics (no numpy/scipy dependency)
# --------------------------------------------------------------------------- #
def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Pearson correlation coefficient, or None if undefined (n<2 or zero variance)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _rank_avg_ties(vals: Sequence[float]) -> List[float]:
    """Fractional ranks (1-based), ties share the average rank (for Spearman)."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # positions i..j (0-based) -> 1-based avg
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation (Pearson on average-tie ranks)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    return pearson(_rank_avg_ties(xs), _rank_avg_ties(ys))


_CORR = {"pearson": pearson, "spearman": spearman}


# --------------------------------------------------------------------------- #
# Grading a team from its picks
# --------------------------------------------------------------------------- #
def team_avg_ps(sample: TeamSample, weights: Optional[dict] = None) -> Optional[float]:
    """Mean pick score across a team's picks under ``weights`` (None = shipped).

    A deliberately simple team grade: the average per-pick score. The full
    dr_team_grade_score composite (lineup strength, construction) layers league
    context on top; for a weight *sweep* the mean pick score isolates the effect
    of the weights themselves, which is what we want to tune.
    """
    scores = []
    for pk in sample.picks:
        try:
            scores.append(compute_pick_score(weights=weights, **pk))
        except Exception:
            continue
    if not scores:
        return None
    return sum(scores) / len(scores)


def grades_and_outcomes(
    samples: Sequence[TeamSample], weights: Optional[dict] = None
) -> Tuple[List[float], List[float]]:
    """Paired (grade, outcome) lists over samples that grade successfully."""
    gs, os_ = [], []
    for s in samples:
        g = team_avg_ps(s, weights)
        if g is None:
            continue
        gs.append(g)
        os_.append(float(s.outcome))
    return gs, os_


def correlate_grades_to_finish(
    samples: Sequence[TeamSample],
    weights: Optional[dict] = None,
    method: str = "spearman",
) -> Optional[float]:
    """Correlation between team grade and outcome (higher-is-better outcome, so a
    good weight table gives a positive coefficient). Returns None if undefined."""
    corr = _CORR.get(method)
    if corr is None:
        raise ValueError(f"unknown method: {method!r}")
    gs, os_ = grades_and_outcomes(samples, weights)
    return corr(gs, os_)


def sweep(
    samples: Sequence[TeamSample],
    candidates: Sequence[Tuple[str, dict]],
    method: str = "spearman",
) -> List[Tuple[str, dict, Optional[float]]]:
    """Rank candidate weight tables by predictive power.

    ``candidates`` is a sequence of (label, weights) pairs. Returns
    [(label, weights, corr), ...] sorted best-first (highest correlation, i.e.
    the weight table whose grades best track real success). Candidates whose
    correlation is undefined sort last.
    """
    out = []
    for label, w in candidates:
        r = correlate_grades_to_finish(samples, w, method=method)
        out.append((label, w, r))
    out.sort(key=lambda t: (t[2] is not None, t[2] if t[2] is not None else 0.0), reverse=True)
    return out


def _perturb(base: dict, key: str, delta: float) -> dict:
    """A copy of ``base`` with ``key`` nudged by ``delta`` then renormalized to
    the original weight sum, so sweeps compare re-weightings, not rescalings."""
    w = dict(base)
    w[key] = max(0.0, w.get(key, 0.0) + delta)
    tot = sum(w.values())
    base_tot = sum(base.values()) or 1.0
    if tot > 0:
        w = {k: v * base_tot / tot for k, v in w.items()}
    return w


def candidate_grid(base: dict, deltas: Sequence[float] = (-0.10, -0.05, 0.05, 0.10)) -> List[Tuple[str, dict]]:
    """A default candidate set: the base table plus single-component nudges.

    Useful as a starting sweep — it shows which lever (value, adp, ppg, ...)
    moving up or down improves the grade-vs-finish correlation the most.
    """
    cands: List[Tuple[str, dict]] = [("base", dict(base))]
    for key in base:
        for d in deltas:
            cands.append((f"{key}{d:+.2f}", _perturb(base, key, d)))
    return cands


# Convenience default: nudges around the shipped startup table.
WEIGHT_CANDIDATES: List[Tuple[str, dict]] = candidate_grid(PS_WEIGHTS["startup"])


# --------------------------------------------------------------------------- #
# Real-data loader (Sleeper). Documented + defensive; returns [] offline.
# --------------------------------------------------------------------------- #
def detect_sleeper_meta(
    league: dict, draft: dict, num_rosters: int, *,
    default_type: str = "startup", default_sf: bool = False, default_teams: int = 12,
) -> Tuple[bool, str, int]:
    """Infer (is_sf, draft_type, num_teams) from a Sleeper league + its draft.

    * SF: ``SUPER_FLEX`` in roster_positions, or >=2 QB-eligible starting slots.
    * draft_type: settings.type == 0 -> redraft; else a short draft (<=5 rounds)
      is a dynasty rookie draft, a long one is a startup. This matters because a
      dynasty league's *first* season is a startup but later seasons are rookie
      drafts, and each uses a different ADP/value basis.
    * num_teams: roster count (falls back to total_rosters, then default).

    Anything unparseable falls back to the passed defaults — never raises.
    """
    is_sf, draft_type, num_teams = default_sf, default_type, default_teams
    if not league:
        return is_sf, draft_type, num_teams
    try:
        rp = league.get("roster_positions") or []
        if rp:  # only override SF when we actually know the roster shape
            qb_slots = sum(1 for s in rp if s in ("QB", "SUPER_FLEX"))
            is_sf = ("SUPER_FLEX" in rp) or (qb_slots >= 2)
        num_teams = int(num_rosters) or int(league.get("total_rosters") or 0) or default_teams
        settings = league.get("settings") or {}
        rounds = int(((draft or {}).get("settings") or {}).get("rounds") or 0)
        if int(settings.get("type") or 0) == 0:
            draft_type = "redraft"
        elif 1 <= rounds <= 5:
            draft_type = "rookie"
        else:
            draft_type = "startup"
    except Exception:
        pass
    return is_sf, draft_type, num_teams


def load_sleeper_samples(
    league_ids: Sequence[str],
    season: int,
    *,
    value_fn: Optional[Callable[[dict], Optional[dict]]] = None,
    value_fn_factory: Optional[Callable[[bool, str], Callable[[dict], Optional[dict]]]] = None,
    draft_type: str = "startup",
    is_sf: bool = False,
    num_teams: int = 12,
    auto_detect: bool = False,
) -> List[TeamSample]:
    """Build ``TeamSample`` rows from completed Sleeper leagues.

    For each league it pulls the completed draft's picks and the final standings,
    then turns each roster's picks into ``compute_pick_score`` inputs via a
    ``value_fn`` (the caller's DB-backed valuation lookup: given a pick dict it
    returns {value, vor, tier, age, rank_change_7d, avg_pick, max_val, ppg_norm}
    or None to skip). The outcome is season points-for (settings.fpts), a
    lower-noise success signal than final rank.

    Two modes:
      * Fixed: pass ``value_fn`` and the global ``is_sf``/``draft_type`` — every
        league is graded on that one basis.
      * Auto (``auto_detect=True`` + ``value_fn_factory``): each league's SF-ness
        and draft type are detected from Sleeper (``detect_sleeper_meta``) and the
        matching ``value_fn = value_fn_factory(is_sf, draft_type)`` is used. This
        is what a MIXED 1QB/SF portfolio needs so each league is scored correctly.

    This only runs where Sleeper is reachable and the valuation lookup has DB
    access; offline it swallows the fetch error and returns ``[]`` so the module
    stays importable and testable everywhere.
    """
    try:
        from dashboard_services.api import (
            get_drafts, get_draft_picks, get_rosters, get_league,
        )
    except Exception:
        return []
    if value_fn is None and value_fn_factory is None:
        raise ValueError("provide value_fn or value_fn_factory")

    samples: List[TeamSample] = []
    for league_id in league_ids:
        try:
            drafts = get_drafts(str(league_id)) or []
            done = [d for d in drafts if (d.get("status") == "complete")]
            if not done:
                continue
            draft_obj = done[0]
            draft_id = str(draft_obj.get("draft_id"))
            picks = get_draft_picks(draft_id) or []
            rosters = get_rosters(str(league_id)) or []
        except Exception:
            # Network/DB unavailable (offline sandbox) — skip this league.
            continue

        lg_is_sf, lg_type, lg_teams = is_sf, draft_type, (len(rosters) or num_teams)
        if auto_detect and value_fn_factory is not None:
            try:
                league = get_league(str(league_id)) or {}
            except Exception:
                league = {}
            lg_is_sf, lg_type, lg_teams = detect_sleeper_meta(
                league, draft_obj, len(rosters),
                default_type=draft_type, default_sf=is_sf, default_teams=num_teams,
            )
            league_value_fn = value_fn_factory(lg_is_sf, lg_type)
        else:
            league_value_fn = value_fn

        total_picks = len(picks) or (lg_teams * 15)
        # Season points-for per roster (settings.fpts[.fpts_decimal]).
        pf_by_roster: Dict[str, float] = {}
        for r in rosters:
            rid = str(r.get("roster_id"))
            st = r.get("settings") or {}
            pf = float(st.get("fpts") or 0) + float(st.get("fpts_decimal") or 0) / 100.0
            pf_by_roster[rid] = pf

        by_roster: Dict[str, List[dict]] = {}
        for pk in picks:
            rid = str(pk.get("roster_id"))
            meta = pk.get("metadata") or {}
            pos = (meta.get("position") or "").upper()
            pick_no = int(pk.get("pick_no") or 0)
            row = {
                "pos": pos, "pick_no": pick_no, "draft_type": lg_type,
                "is_sf": lg_is_sf, "num_teams": lg_teams, "total_picks": total_picks,
            }
            vals = league_value_fn({**pk, "position": pos, "pick_no": pick_no})
            if not vals:
                continue
            row.update(vals)
            row.setdefault("qb_count", 0)
            row.setdefault("need_raw", 0.5)
            by_roster.setdefault(rid, []).append(row)

        for rid, team_picks in by_roster.items():
            if rid not in pf_by_roster:
                continue
            samples.append(TeamSample(
                picks=team_picks, outcome=pf_by_roster[rid],
                label=f"{league_id}:{rid}",
                meta={"league_id": str(league_id), "is_sf": lg_is_sf, "draft_type": lg_type},
            ))
    return samples


# --------------------------------------------------------------------------- #
# Synthetic samples — a self-contained signal for the offline test.
# --------------------------------------------------------------------------- #
def _synthetic_samples(seed: int = 7, n_teams: int = 60, picks_per_team: int = 10) -> List[TeamSample]:
    """Deterministic synthetic leagues with a KNOWN signal: a team's outcome is a
    noisy function of the true dynasty value it drafted. Because the picks encode
    that quality in ``value``/``vor``, a value-heavy weight table should track
    the outcome better than a value-light one, which the sweep test verifies.
    """
    import random

    rnd = random.Random(seed)
    positions = ["RB", "WR", "WR", "TE", "QB", "RB", "WR"]
    samples: List[TeamSample] = []
    max_val = 10000.0
    for t in range(n_teams):
        picks = []
        true_quality = 0.0
        for i in range(picks_per_team):
            pos = positions[i % len(positions)]
            pick_no = i * n_teams + (t % n_teams) + 1
            # Underlying quality: earlier picks are better, plus team-level skill.
            base = max_val * (1.0 - i / (picks_per_team + 2))
            skill = rnd.uniform(-0.15, 0.15) * max_val
            value = max(200.0, base + skill + rnd.uniform(-500, 500))
            true_quality += value
            picks.append(dict(
                pos=pos, value=value, vor=value * 0.5, tier=1 + i // 2, age=24,
                rank_change_7d=0, avg_pick=pick_no, pick_no=pick_no,
                max_val=max_val, draft_type="startup", is_sf=False,
                need_raw=0.5, qb_count=0, total_picks=picks_per_team * n_teams,
                num_teams=n_teams, ppg_norm=min(1.0, value / max_val),
            ))
        # Outcome tracks true quality with noise (higher = better).
        outcome = true_quality + rnd.uniform(-0.6, 0.6) * max_val
        samples.append(TeamSample(picks=picks, outcome=outcome, label=f"syn{t}"))
    return samples


if __name__ == "__main__":  # pragma: no cover - manual/reporting entry point
    samples = _synthetic_samples()
    base_r = correlate_grades_to_finish(samples)
    print(f"synthetic baseline spearman r = {base_r:+.3f} over {len(samples)} teams\n")
    print("candidate sweep (best predictor first):")
    for label, _w, r in sweep(samples, WEIGHT_CANDIDATES):
        print(f"  {r if r is None else f'{r:+.3f}'}  {label}")
