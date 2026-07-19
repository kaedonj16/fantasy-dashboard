"""Property-based validation of the archetype suggestion engine.

The fixed-fixture e2e tests pin specific scenarios; this runs the engine across
many *random* leagues (loaded / one-stud / thin / balanced viewers) and asserts
that each archetype's guardrails hold everywhere, not just on the hand-built
cases - the real "does the model do what it claims" check. Uses the offline
analytical path (no sim/network), driven by scripts.eval_suggestion_quality.

Needs the full stack (the engine imports app); skipped in the pure suite.
"""
import statistics

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

from scripts.eval_suggestion_quality import evaluate  # noqa: E402

_ARCHES = ("contending", "rebuilding", "consolidate", "distribute")


@pytest.fixture(scope="module")
def report():
    # 20 random 12-team leagues x 4 archetypes = 80 suggestion runs (~15s).
    return evaluate(n_leagues=20, seed=7)


def test_every_archetype_produces_suggestions(report):
    for a in _ARCHES:
        assert report[a]["leagues_with_sugg"] > 0, f"{a} surfaced nothing in any league"


def test_guardrails_hold_across_random_leagues(report):
    """Zero guardrail violations anywhere: consolidate never ships a lone stud,
    always trades up past its send headliner, and never bundles rocks; distribute
    always returns several usable pieces for one stud; every acquire has a send
    and a plausible acceptance."""
    for a in _ARCHES:
        r = report[a]
        assert r["violations"] == 0, f"{a} guardrail violations: {r['viol_samples']}"


def test_consolidate_is_more_win_now_than_distribute(report):
    """Directional sanity: consolidating depth into a stud should read as more
    win-now than breaking a stud into depth. Robust across seeds (consolidate is
    ceiling-raising; distribute trades current lineup value for flexibility)."""
    cons = report["consolidate"]["net_wpd"]
    dist = report["distribute"]["net_wpd"]
    assert cons and dist
    assert statistics.mean(cons) > statistics.mean(dist)
