"""Career-path overlay for Hist: prior elite vs last-year-only comps.

A down year after a top-12 season is not the same cohort as never having
been elite. Request-path Hist uses these rates for the headline chance.
Display only; not a Pick Score input.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from dashboard_services.historical.comps import extract_comp_query
from dashboard_services.historical.definitions import (
    CAREER_STAGE_ORDER,
    COMP_BOARD_TIERS,
    DRAFT_CAPITAL_ORDER,
    MIN_COMP_CELL_N,
    career_stage,
    prior_finish_bucket,
    _optional_int,
)
from dashboard_services.historical.finish_rates import cohort_hit_rate


LAST_YEAR_ELITE = frozenset({"top_5", "top_12"})


def _prior_finish_of(row: Mapping[str, Any]) -> Optional[str]:
    prior = prior_finish_bucket(
        row.get("previous_season_finish"),
        years_experience=row.get("years_experience"),
    )
    if prior:
        return prior
    raw = row.get("prior_finish")
    return str(raw) if raw else None


def is_bounce_back_query(row: Mapping[str, Any]) -> bool:
    """Previously top-12, last year outside top-12 (or no last-year elite)."""
    count = _optional_int(row.get("prior_top12_count"))
    if count is None or count < 1:
        return False
    return _prior_finish_of(row) not in LAST_YEAR_ELITE


def is_bounce_back_row(row: Mapping[str, Any]) -> bool:
    """Warehouse season: already elite before this year, last year not top-12."""
    count = _optional_int(row.get("prior_top12_count"))
    if count is None or count < 1:
        return False
    last = _optional_int(row.get("previous_season_finish"))
    return last is None or last > 12


def _tier_rates(
    rows: Sequence[Mapping[str, Any]],
    *,
    scoring: str,
    prior_rate: Optional[float],
) -> dict[str, dict]:
    return {
        tier: cohort_hit_rate(rows, tier=tier, scoring=scoring, prior_rate=prior_rate)
        for tier in COMP_BOARD_TIERS
    }


def _rate_n(block: Any) -> int:
    if not isinstance(block, Mapping):
        return 0
    top12 = block.get("top_12") if isinstance(block.get("top_12"), Mapping) else {}
    return int(top12.get("sample_size") or block.get("sample_size") or 0)


def build_bounce_back_rates(
    pos_rows: Sequence[Mapping[str, Any]],
    *,
    scoring: str = "ppr",
    prior_rate: Optional[float] = None,
) -> dict[str, Any]:
    """P(hit | previously top-12 and last year outside top-12)."""
    cohort = [row for row in pos_rows if is_bounce_back_row(row)]
    out: dict[str, Any] = {
        "bounce_back": _tier_rates(cohort, scoring=scoring, prior_rate=prior_rate),
        "n_bounce_back": len(cohort),
        "bounce_back_by_stage": {},
        "bounce_back_by_capital": {},
    }
    by_stage = out["bounce_back_by_stage"]
    for stage in CAREER_STAGE_ORDER:
        at = [
            row for row in cohort
            if career_stage(row.get("years_experience")) == stage
        ]
        if at:
            by_stage[stage] = _tier_rates(at, scoring=scoring, prior_rate=prior_rate)
    by_cap = out["bounce_back_by_capital"]
    for bucket in DRAFT_CAPITAL_ORDER:
        at = [row for row in cohort if row.get("draft_capital_bucket") == bucket]
        if at:
            by_cap[bucket] = _tier_rates(at, scoring=scoring, prior_rate=prior_rate)
    return out


def apply_career_path_history(
    query: Mapping[str, Any],
    looked: Mapping[str, Any],
    aggregates: Mapping[str, Any],
) -> dict:
    """Replace last-year-only comps with bounce-back rates when career says so."""
    out = dict(looked or {})
    if not is_bounce_back_query(query):
        return out
    pos = str(query.get("position") or out.get("position") or "").upper()
    block = ((aggregates.get("repeat_and_breakout") or {}).get(pos) or {})
    if not isinstance(block, Mapping):
        return out
    feats = extract_comp_query(query)
    stage = feats.get("career_stage")
    capital = feats.get("draft_capital")
    staged = (block.get("bounce_back_by_stage") or {}).get(stage) if stage else None
    capped = (block.get("bounce_back_by_capital") or {}).get(capital) if capital else None
    overall = block.get("bounce_back") or {}
    used = "overall"
    chosen: Any = overall
    if _rate_n(staged) >= MIN_COMP_CELL_N:
        chosen = staged
        used = "stage"
    elif _rate_n(capped) >= MIN_COMP_CELL_N:
        chosen = capped
        used = "capital"
    if _rate_n(chosen) <= 0:
        return out
    rates = {}
    for tier in COMP_BOARD_TIERS:
        rec = chosen.get(tier) if isinstance(chosen, Mapping) else None
        if isinstance(rec, Mapping) and rec.get("display_pct") is not None:
            rates[str(tier)] = dict(rec)
    if not rates:
        return out
    profile_key = dict(feats)
    profile_key["prior_elite"] = "has_been"
    key_used = {"position": pos, "prior_elite": "has_been"}
    if feats.get("prior_finish"):
        key_used["prior_finish"] = feats["prior_finish"]
    if stage:
        key_used["career_stage"] = stage
    if capital:
        key_used["draft_capital"] = capital
    out["rates"] = rates
    out["n"] = _rate_n(chosen)
    out["key_used"] = key_used
    out["profile_key"] = profile_key
    out["career_path"] = "bounce_back"
    out["career_path_rate"] = used
    out["source"] = "career_path"
    out["dropped"] = []
    out["fallback"] = used != "stage"
    out["examples"] = []
    return out
