"""Age curves: distribution of hits vs conditional hit rate (pure).

Two statistics that must never be collapsed:

1. **Distribution** — among seasons that *did* hit (e.g. RB1), what share
   fell in an age window. "X% of RB1 seasons came from ages 23–27."
2. **Conditional hit rate** — among qualifying seasons *in* that window,
   what share hit. "X% of age-23–27 RB seasons finished RB1."

Prime windows are derived from the data: Bayes-smoothed P(top-12 | integer
age, position) vs the position baseline; ages at or above baseline with
n large enough to not be ``low`` confidence. The window is the longest
consecutive run (ties broken by successes). Do not hard-code 23–27 as prime.

Rows with missing age are omitted from age curves only.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

from dashboard_services.historical.definitions import (
    AGE_BUCKETS,
    CONFIDENCE_LOW,
    RELIABLE_SEASON_FLOOR,
    SKILL_POSITIONS,
    age_bucket,
    integer_age,
)
from dashboard_services.historical.finish_rates import (
    cohort_hit_rate,
    filter_era,
    filter_position,
    is_tier_hit,
    make_share,
    position_baseline,
)

# Prime ages require at least moderate sample size (not "low").
MIN_PRIME_N = 15

PRIME_WINDOW_METHOD = (
    "smoothed P(tier | integer age, position) >= position baseline "
    "among known-age seasons, and sample_size >= 15 (not low confidence); "
    "window = longest consecutive qualifying ages, ties broken by successes"
)


def rows_with_known_age(rows: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Skip missing age. Never treat missing as 0."""
    out = []
    for row in rows:
        if integer_age(row.get("age")) is None:
            continue
        out.append(dict(row))
    return out


def age_window_pair(
    rows: Sequence[Mapping[str, Any]],
    position: Any,
    age_lo: int,
    age_hi: int,
    *,
    tier: str = "top_12",
    scoring: str = "ppr",
    prior_rate: Optional[float] = None,
) -> dict:
    """Distribution share **and** conditional hit rate for one inclusive age window.

    The two numbers are different statistics. Callers must keep both.
    """
    pos_aged = rows_with_known_age(filter_position(rows, position))
    hits = [r for r in pos_aged if is_tier_hit(r, tier=tier, scoring=scoring)]
    in_window = [
        r for r in pos_aged
        if age_lo <= integer_age(r.get("age")) <= age_hi  # type: ignore[operator]
    ]
    hits_in_window = [r for r in in_window if is_tier_hit(r, tier=tier, scoring=scoring)]
    if prior_rate is None:
        prior_rate = position_baseline(
            pos_aged, position, tier=tier, scoring=scoring
        ).get("raw_rate")
    return {
        "position": str(position or "").upper(),
        "tier": tier,
        "scoring": scoring,
        "age_lo": age_lo,
        "age_hi": age_hi,
        "n_known_age": len(pos_aged),
        "n_hits": len(hits),
        "n_in_window": len(in_window),
        "distribution": make_share(len(hits_in_window), len(hits), seasons=hits),
        "conditional": cohort_hit_rate(
            in_window, tier=tier, scoring=scoring, prior_rate=prior_rate
        ),
    }


def derive_prime_window(
    by_integer_age: Mapping[int, Mapping[str, Any]],
    baseline_rate: Optional[float],
    *,
    min_n: int = MIN_PRIME_N,
) -> Optional[dict]:
    """Longest consecutive integer-age run at/above baseline with n >= min_n."""
    if baseline_rate is None:
        return None
    prime_ages = []
    for age, rec in by_integer_age.items():
        conditional = rec.get("conditional") or rec
        n = conditional.get("sample_size") or 0
        smoothed = conditional.get("smoothed_rate")
        if smoothed is None or n < min_n:
            continue
        if smoothed >= baseline_rate:
            prime_ages.append(int(age))
    prime_ages.sort()
    if not prime_ages:
        return None

    runs: list[tuple[int, int]] = []
    start = prev = prime_ages[0]
    for age in prime_ages[1:]:
        if age == prev + 1:
            prev = age
        else:
            runs.append((start, prev))
            start = prev = age
    runs.append((start, prev))

    def _score(lo: int, hi: int) -> tuple[int, int]:
        successes = 0
        for age in range(lo, hi + 1):
            rec = by_integer_age.get(age) or {}
            cond = rec.get("conditional") or rec
            successes += int(cond.get("successes") or 0)
        return (hi - lo + 1, successes)

    lo, hi = max(runs, key=lambda run: _score(*run))
    return {
        "age_start": lo,
        "age_end": hi,
        "ages": list(range(lo, hi + 1)),
        "method": PRIME_WINDOW_METHOD,
        "baseline_rate": baseline_rate,
        "min_n": min_n,
    }


def _by_integer_age(
    aged: Sequence[Mapping[str, Any]],
    hits: Sequence[Mapping[str, Any]],
    *,
    tier: str,
    scoring: str,
    prior_rate: Optional[float],
) -> dict[int, dict]:
    ages = sorted({integer_age(r.get("age")) for r in aged} - {None})  # type: ignore[misc]
    out: dict[int, dict] = {}
    for age in ages:
        at_age = [r for r in aged if integer_age(r.get("age")) == age]
        hits_at = [r for r in at_age if is_tier_hit(r, tier=tier, scoring=scoring)]
        out[int(age)] = {
            "integer_age": int(age),
            "distribution": make_share(len(hits_at), len(hits), seasons=hits),
            "conditional": cohort_hit_rate(
                at_age, tier=tier, scoring=scoring, prior_rate=prior_rate
            ),
        }
    return out


def _by_ui_bucket(
    aged: Sequence[Mapping[str, Any]],
    hits: Sequence[Mapping[str, Any]],
    position: str,
    *,
    tier: str,
    scoring: str,
    prior_rate: Optional[float],
) -> dict[str, dict]:
    labels = [label for _lo, _hi, label in AGE_BUCKETS.get(position, ())]
    out: dict[str, dict] = {}
    for label in labels:
        at_bucket = [r for r in aged if age_bucket(position, r.get("age")) == label]
        hits_at = [r for r in at_bucket if is_tier_hit(r, tier=tier, scoring=scoring)]
        out[label] = {
            "bucket": label,
            "distribution": make_share(len(hits_at), len(hits), seasons=hits),
            "conditional": cohort_hit_rate(
                at_bucket, tier=tier, scoring=scoring, prior_rate=prior_rate
            ),
        }
    return out


def build_position_age_curve(
    rows: Sequence[Mapping[str, Any]],
    position: Any,
    *,
    tier: str = "top_12",
    scoring: str = "ppr",
) -> dict:
    pos = str(position or "").upper()
    aged = rows_with_known_age(filter_position(rows, pos))
    hits = [r for r in aged if is_tier_hit(r, tier=tier, scoring=scoring)]
    baseline = position_baseline(aged, pos, tier=tier, scoring=scoring)
    prior_rate = baseline.get("raw_rate")
    by_age = _by_integer_age(
        aged, hits, tier=tier, scoring=scoring, prior_rate=prior_rate
    )
    prime = derive_prime_window(by_age, prior_rate)
    prime_pair = None
    if prime is not None:
        prime_pair = age_window_pair(
            aged,
            pos,
            prime["age_start"],
            prime["age_end"],
            tier=tier,
            scoring=scoring,
            prior_rate=prior_rate,
        )
    return {
        "position": pos,
        "tier": tier,
        "scoring": scoring,
        "n_known_age": len(aged),
        "n_hits": len(hits),
        "n_missing_age_excluded": len(filter_position(rows, pos)) - len(aged),
        "baseline": baseline,
        "prime_window": prime,
        "prime_window_pair": prime_pair,
        "by_integer_age": {str(age): rec for age, rec in sorted(by_age.items())},
        "by_bucket": _by_ui_bucket(
            aged, hits, pos, tier=tier, scoring=scoring, prior_rate=prior_rate
        ),
    }


def build_age_curves(
    rows: Sequence[Mapping[str, Any]],
    *,
    scoring: str = "ppr",
    tier: str = "top_12",
    season_from: int = RELIABLE_SEASON_FLOOR,
    season_to: Optional[int] = None,
) -> dict:
    era = filter_era(rows, season_from, season_to)
    return {
        pos: build_position_age_curve(era, pos, tier=tier, scoring=scoring)
        for pos in SKILL_POSITIONS
    }
