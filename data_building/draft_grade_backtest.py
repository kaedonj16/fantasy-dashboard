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


def rank_success(final_rank: int, num_teams: int) -> float:
    """Final standing -> [0, 1] success (champion 1.0, last 0.0). Unlike
    ``outcome_from_rank`` this is normalized across league sizes, so successes
    from different seasons/leagues are comparable and can be averaged."""
    n = max(int(num_teams), 2)
    r = min(max(int(final_rank), 1), n)
    return (n - r) / (n - 1)


def multiyear_outcome(successes: Sequence[float], decay: float = 0.75) -> Optional[float]:
    """Combine per-season successes into one outcome, weighting the draft season
    most and decaying forward (a startup draft's payoff is heaviest early but
    should still reward sustained contention). ``successes`` is ordered
    [draft_season, +1, +2, ...]. Returns None if empty."""
    if not successes:
        return None
    wsum = tot = 0.0
    for i, s in enumerate(successes):
        w = decay ** i
        wsum += w * float(s)
        tot += w
    return wsum / tot if tot > 0 else None


def final_ranks(rosters: Sequence[dict], winners_bracket: Sequence[dict]) -> Dict[str, int]:
    """Map roster_id -> final standing (1 = champion) for one completed season.

    Playoff teams take their bracket placement; everyone else is ranked beneath
    them by regular-season record (wins, then points-for). With no bracket it
    degrades to a pure regular-season ranking. Pure — no IO.
    """
    from utils.pick_slots import placements_from_bracket

    _playoff_rids, placements = placements_from_bracket(list(winners_bracket or []))
    ranked: Dict[str, int] = {str(rid): int(p) for rid, p in placements.items()}
    max_place = max(placements.values()) if placements else 0

    remaining = []
    for r in rosters or []:
        rid = str(r.get("roster_id"))
        if rid in ranked:
            continue
        st = r.get("settings") or {}
        wins = float(st.get("wins") or 0)
        pf = float(st.get("fpts") or 0) + float(st.get("fpts_decimal") or 0) / 100.0
        remaining.append((rid, wins, pf))
    remaining.sort(key=lambda x: (x[1], x[2]), reverse=True)  # best record first
    nxt = max_place + 1
    for rid, _w, _pf in remaining:
        ranked[rid] = nxt
        nxt += 1
    return ranked


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
def team_avg_ps(sample: TeamSample, weights: Optional[dict] = None,
                depth_slope: Optional[float] = None) -> Optional[float]:
    """Mean pick score across a team's picks under ``weights`` (None = shipped).

    A deliberately simple team grade: the average per-pick score. The full
    dr_team_grade_score composite (lineup strength, construction) layers league
    context on top; for a weight *sweep* the mean pick score isolates the effect
    of the weights themselves, which is what we want to tune. ``depth_slope``
    overrides the depth-normalization slope (None = shipped 0.44) so the same
    harness can tune the by-round flatness.
    """
    scores = []
    for pk in sample.picks:
        try:
            scores.append(compute_pick_score(weights=weights, depth_slope=depth_slope, **pk))
        except Exception:
            continue
    if not scores:
        return None
    return sum(scores) / len(scores)


def pick_score_by_round(
    samples: Sequence[TeamSample], weights: Optional[dict] = None,
    depth_slope: Optional[float] = None,
) -> List[dict]:
    """Mean pick score per draft ROUND across every pick in ``samples``.

    A calibration check on the depth-normalization (the ``_par`` re-anchoring in
    compute_pick_score): if that scale is right, average pick score should be
    roughly FLAT across rounds. A downward slope means late rounds are
    under-scored (par too aggressive); an upward slope means over-corrected.
    Round is ``(pick_no - 1) // num_teams + 1``. Returns [{round, n, score_mean}]
    ordered by round.
    """
    agg: Dict[int, List[float]] = {}
    for s in samples:
        for pk in s.picks:
            try:
                teams = int(pk.get("num_teams") or 12) or 12
                pick_no = int(pk.get("pick_no") or 0)
                if pick_no <= 0:
                    continue
                rnd = (pick_no - 1) // teams + 1
                score = compute_pick_score(weights=weights, depth_slope=depth_slope, **pk)
            except Exception:
                continue
            row = agg.setdefault(rnd, [0.0, 0.0])  # [count, sum]
            row[0] += 1
            row[1] += score
    return [
        {"round": r, "n": int(agg[r][0]), "score_mean": agg[r][1] / agg[r][0]}
        for r in sorted(agg)
    ]


def grades_and_outcomes(
    samples: Sequence[TeamSample], weights: Optional[dict] = None,
    depth_slope: Optional[float] = None,
) -> Tuple[List[float], List[float]]:
    """Paired (grade, outcome) lists over samples that grade successfully."""
    gs, os_ = [], []
    for s in samples:
        g = team_avg_ps(s, weights, depth_slope=depth_slope)
        if g is None:
            continue
        gs.append(g)
        os_.append(float(s.outcome))
    return gs, os_


def correlate_grades_to_finish(
    samples: Sequence[TeamSample],
    weights: Optional[dict] = None,
    method: str = "spearman",
    depth_slope: Optional[float] = None,
) -> Optional[float]:
    """Correlation between team grade and outcome (higher-is-better outcome, so a
    good weight table gives a positive coefficient). Returns None if undefined."""
    corr = _CORR.get(method)
    if corr is None:
        raise ValueError(f"unknown method: {method!r}")
    gs, os_ = grades_and_outcomes(samples, weights, depth_slope=depth_slope)
    return corr(gs, os_)


def sweep_depth(
    samples: Sequence[TeamSample],
    slopes: Sequence[float],
    weights: Optional[dict] = None,
    method: str = "spearman",
) -> List[Tuple[float, Optional[float]]]:
    """Rank depth-normalization slopes by predictive power (higher corr = better).

    The shipped slope is 0.44; a steeper slope boosts later picks more, flattening
    the by-round pick-score curve. Returns [(slope, corr), ...] best-first so we
    can pick the depth calibration that best tracks real success rather than
    eyeballing flatness. Undefined correlations sort last.
    """
    out = [(sl, correlate_grades_to_finish(samples, weights, method=method, depth_slope=sl))
           for sl in slopes]
    out.sort(key=lambda t: (t[1] is not None, t[1] if t[1] is not None else 0.0), reverse=True)
    return out


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


def calibration_bins(
    samples: Sequence[TeamSample], weights: Optional[dict] = None, n_bins: int = 5,
) -> List[dict]:
    """Bin teams into ``n_bins`` equal-count grade tiers and report the mean
    outcome of each — a reliability check on the grade SCALE (not the weights).

    A well-calibrated grade is monotonic: the top grade bin should have a clearly
    higher mean outcome than the bottom. If the top two bins have the same
    outcome, the scale saturates up top (an 'A' means nothing beyond a 'B'); if
    it's flat or inverted, the grade isn't measuring success. Returns one dict
    per bin: {bin, n, grade_lo, grade_hi, grade_mean, outcome_mean}, low->high.
    """
    gs, os_ = grades_and_outcomes(samples, weights)
    n = len(gs)
    if n < n_bins:
        return []
    order = sorted(range(n), key=lambda i: gs[i])
    out: List[dict] = []
    for b in range(n_bins):
        lo = (b * n) // n_bins
        hi = ((b + 1) * n) // n_bins
        idx = order[lo:hi]
        if not idx:
            continue
        gvals = [gs[i] for i in idx]
        ovals = [os_[i] for i in idx]
        out.append({
            "bin": b + 1, "n": len(idx),
            "grade_lo": min(gvals), "grade_hi": max(gvals),
            "grade_mean": sum(gvals) / len(gvals),
            "outcome_mean": sum(ovals) / len(ovals),
        })
    return out


_LETTER_ORDER = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"]

# Default lineup slots + construction targets per format, so the composite grade
# (dr_team_grade_score) can be reconstructed from a team's pick-score inputs. A
# calibration approximation, not a per-league roster read.
_SLOTS_1QB = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX"]
_SLOTS_SF = _SLOTS_1QB + ["SF"]
_TARGETS_1QB = {"QB": 2, "RB": 5, "WR": 6, "TE": 2}
_TARGETS_SF = {"QB": 3, "RB": 5, "WR": 5, "TE": 2}


def _team_composite(sample: TeamSample, league_val_list, league_ppg_list,
                    weights: Optional[dict] = None) -> Optional[float]:
    """Reconstruct the shipped Value/Starters/Construction composite
    (dr_team_grade_score, 0-100) for one team from its pick-score inputs — the
    same raw score the Draft Room / Teams page feed to the field curve. Uses the
    league's value/ppg lists for the 'vs a league-average team' component."""
    from utils.draft_grade import dr_team_grade_score

    is_sf = bool(sample.meta.get("is_sf"))
    picks = []
    for i, pk in enumerate(sample.picks):
        try:
            ps = compute_pick_score(weights=weights, **pk)
        except Exception:
            continue
        picks.append({
            "id": i, "pos": pk.get("pos"), "ps": ps, "pn": pk.get("pick_no"),
            "val": pk.get("value"), "ppg": pk.get("ppg_norm"),
        })
    if not picks:
        return None
    try:
        teams = int(sample.picks[0].get("num_teams") or 12) or 12
    except Exception:
        teams = 12
    slots = _SLOTS_SF if is_sf else _SLOTS_1QB
    targets = _TARGETS_SF if is_sf else _TARGETS_1QB
    dtype = "redraft" if (sample.meta.get("draft_type") == "redraft") else "startup"
    return dr_team_grade_score(
        picks, slots=slots, targets=targets, num_teams=teams, draft_type=dtype,
        league_ppg_list=list(league_ppg_list), league_val_list=list(league_val_list),
    )


def letter_calibration(
    samples: Sequence[TeamSample], raw_grade_fn=None,
    rounds_done: int = 99, weights: Optional[dict] = None,
    include_types=("startup", "redraft"),
) -> List[dict]:
    """Report mean outcome per LETTER grade a team would actually receive.

    Runs the SHIPPED composite (Value/Starters/Construction) + field curve + band
    mapping, curved within each team's own league exactly as the Draft Room /
    Teams page do — so it answers: does the A-F letter a user sees track success?
    If the A rows don't out-perform B/C, the anchor/bands are miscalibrated and
    this table shows which way to move them.

    Defaults to the real composite (``raw_grade_fn=None``); pass a
    ``raw_grade_fn(sample) -> 0-100`` to override (e.g. mean pick score). The
    field curve is the startup/redraft model (rookie drafts use a different
    letter system), so by default only those ``include_types`` are graded.
    Leagues with <3 teams are skipped (the curve needs a field). Returns
    [{letter, n, outcome_mean}] ordered best->worst; only letters that occur.
    """
    from utils.draft_grade import dr_apply_field_curve, dr_grade_letter

    use_composite = raw_grade_fn is None
    by_league: Dict[object, List[TeamSample]] = {}
    for s in samples:
        if use_composite and include_types and (s.meta.get("draft_type") or "startup") not in include_types:
            continue
        by_league.setdefault(s.meta.get("league_id"), []).append(s)

    pairs: List[Tuple[str, float]] = []
    for _lg, members in by_league.items():
        if use_composite:
            lvl = [pk.get("value") for s in members for pk in s.picks if pk.get("value") is not None]
            lpl = [pk.get("ppg_norm") for s in members for pk in s.picks if pk.get("ppg_norm") is not None]
            graded = [(s, _team_composite(s, lvl, lpl, weights)) for s in members]
        else:
            graded = [(s, raw_grade_fn(s)) for s in members]
        graded = [(s, g) for s, g in graded if g is not None]
        if len(graded) < 3:
            continue  # no field to curve against
        curved = dr_apply_field_curve([g for _s, g in graded], rounds_done)
        for (s, _g), c in zip(graded, curved):
            pairs.append((dr_grade_letter(c), float(s.outcome)))

    agg: Dict[str, List[float]] = {}
    for letter, outcome in pairs:
        row = agg.setdefault(letter, [0.0, 0.0])  # [count, sum]
        row[0] += 1
        row[1] += outcome
    return [
        {"letter": L, "n": int(agg[L][0]), "outcome_mean": agg[L][1] / agg[L][0]}
        for L in _LETTER_ORDER if L in agg
    ]



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


def load_multiyear_samples(
    current_league_id: str,
    season: int,
    *,
    value_fn_factory: Callable[[bool, str], Callable[[dict], Optional[dict]]],
    draft_types: Sequence[str] = ("startup",),
    num_teams: int = 12,
    min_seasons: int = 2,
    decay: float = 0.75,
) -> List[TeamSample]:
    """Grade drafts of the given ``draft_types`` against a MULTI-YEAR outcome
    instead of same-season points-for.

    Same-season points-for barely tracks a dynasty draft's grade because the
    payoff lands over the *following* seasons, not the draft year - true for a
    startup's long-term value and doubly so for a rookie class (a rookie barely
    plays year 1, breaks out year 2-3). This walks the league's full history
    (``build_league_history_map``), and for each qualifying draft season it scores
    the draft against how each manager's team finished across that season and
    every later one (``rank_success`` per season, combined by ``multiyear_outcome``
    with a forward ``decay`` - use a higher decay for rookies so later seasons,
    where the class matures, count more). Managers are matched across seasons by
    owner_id (roster_id can change; the owner does not).

    Real-data only (needs Sleeper + DB-backed valuations); returns ``[]`` offline.
    """
    try:
        from dashboard_services.api import (
            build_league_history_map, get_league, get_drafts, get_draft_picks,
            get_rosters, get_bracket,
        )
    except Exception:
        return []
    try:
        hist = build_league_history_map("sleeper", str(current_league_id), int(season)) or {}
    except Exception:
        return []
    if len(hist) < min_seasons:
        return []
    seasons_sorted = sorted(int(y) for y in hist.keys())

    # Per-season standings + owner map, fetched once and reused across draft years.
    season_data: Dict[int, Optional[dict]] = {}
    for yr in seasons_sorted:
        lid = hist[yr]
        try:
            rosters = get_rosters(str(lid)) or []
            try:
                bracket = get_bracket(str(lid), "winners") or []
            except Exception:
                bracket = []
            season_data[yr] = {
                "ranks": final_ranks(rosters, bracket),
                "owner": {str(r.get("roster_id")): (str(r.get("owner_id")) if r.get("owner_id") is not None else None)
                          for r in rosters},
                "teams": len(rosters) or num_teams,
            }
        except Exception:
            season_data[yr] = None

    samples: List[TeamSample] = []
    for yr in seasons_sorted:
        sd = season_data.get(yr)
        if not sd:
            continue
        lid = hist[yr]
        try:
            drafts = get_drafts(str(lid)) or []
            done = [d for d in drafts if d.get("status") == "complete"]
            if not done:
                continue
            draft_obj = done[0]
            league = get_league(str(lid)) or {}
        except Exception:
            continue
        is_sf, dtype, teams = detect_sleeper_meta(
            league, draft_obj, sd["teams"], default_teams=num_teams)
        if dtype not in draft_types:
            continue  # only the requested draft type(s)
        forward = [s for s in seasons_sorted if s >= yr and season_data.get(s)]
        if len(forward) < min_seasons:
            continue  # too little forward history to judge sustained success
        try:
            picks = get_draft_picks(str(draft_obj.get("draft_id"))) or []
        except Exception:
            continue

        value_fn = value_fn_factory(is_sf, dtype)
        total_picks = len(picks) or teams * 15
        owner_this = sd["owner"]
        by_owner: Dict[str, List[dict]] = {}
        for pk in picks:
            owner = owner_this.get(str(pk.get("roster_id")))
            if not owner:
                continue
            meta = pk.get("metadata") or {}
            pos = (meta.get("position") or "").upper()
            pick_no = int(pk.get("pick_no") or 0)
            row = {
                "pos": pos, "pick_no": pick_no, "draft_type": dtype,
                "is_sf": is_sf, "num_teams": teams, "total_picks": total_picks,
            }
            vals = value_fn({**pk, "position": pos, "pick_no": pick_no})
            if not vals:
                continue
            row.update(vals)
            row.setdefault("qb_count", 0)
            row.setdefault("need_raw", 0.5)
            by_owner.setdefault(owner, []).append(row)

        for owner, team_picks in by_owner.items():
            successes = []
            for s in forward:
                sds = season_data[s]
                rid_s = next((rid for rid, ow in sds["owner"].items() if ow == owner), None)
                if rid_s is not None and rid_s in sds["ranks"]:
                    successes.append(rank_success(sds["ranks"][rid_s], sds["teams"]))
            outcome = multiyear_outcome(successes, decay=decay)
            if outcome is None:
                continue
            samples.append(TeamSample(
                picks=team_picks, outcome=outcome, label=f"{lid}:{owner}",
                meta={"league_id": str(lid), "is_sf": is_sf, "draft_type": dtype,
                      "seasons": len(successes)},
            ))
    return samples


def load_startup_multiyear_samples(current_league_id, season, **kw):
    """Back-compat wrapper: multi-year outcome for startup drafts only."""
    return load_multiyear_samples(current_league_id, season, draft_types=("startup",), **kw)


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
