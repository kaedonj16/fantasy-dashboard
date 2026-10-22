"""The Weekly Recap's projected win probability must be the matchup page's
model, not a private copy.

Regression: ``weekly_recap._proj_win_prob`` used a flat
``sigma = max(0.4 * proj, 4.0)`` per starter with no team CV floor — the
superseded pre-2026-09-29 model — and read a 120 vs 110 game at ~67%
where ``compute_win_prob`` says ~62%. It now delegates to
``compute_win_prob`` (faithful call: recap starters carry pid/pos,
``proj_by_pid`` is the projection map, and an upcoming week's starters
are all not-yet-started, which is compute_win_prob's default for pids
absent from the status map).
"""
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("openai")  # weekly_recap pulls it via dashboard_services.ai.client

from dashboard_services.ai.weekly_recap import _proj_win_prob  # noqa: E402
from dashboard_services.matchups import compute_win_prob  # noqa: E402

_SLOTS = [("QB", 22), ("RB", 16), ("RB", 13), ("WR", 15), ("WR", 12),
          ("TE", 10), ("WR", 11), ("K", 8), ("DEF", 8)]


def _lineup(prefix, total):
    """9 starters whose projections scale to ``total`` points."""
    base = sum(v for _, v in _SLOTS)
    starters, proj = [], {}
    for i, (pos, v) in enumerate(_SLOTS):
        pid = f"{prefix}{i}"
        starters.append({"pid": pid, "pos": pos, "name": pid})
        proj[pid] = v * total / base
    return starters, proj


def test_recap_win_prob_agrees_with_matchup_page_120_vs_110():
    a, pa = _lineup("a", 120.0)
    b, pb = _lineup("b", 110.0)
    proj = {**pa, **pb}
    page = compute_win_prob({"starters": a}, {"starters": b}, {}, proj)
    recap = _proj_win_prob(a, b, proj)
    assert recap == pytest.approx(page, abs=1e-9)
    # The audit's worked example: the CV floor binds, Phi(10/sqrt(1060)).
    assert recap == pytest.approx(0.6206, abs=0.001)


def test_recap_win_prob_coinflip_and_favorite_still_hold():
    a, pa = _lineup("a", 115.0)
    b, pb = _lineup("b", 115.0)
    proj = {**pa, **pb}
    assert _proj_win_prob(a, b, proj) == pytest.approx(0.5, abs=1e-6)
    _, pa2 = _lineup("a", 150.0)
    proj2 = {**pa2, **pb}
    assert _proj_win_prob(a, b, proj2) > 0.8
