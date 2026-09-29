"""Regression tests for random future-injury simulation in the season sim.

Covers: (a) injury events occur at roughly the configured per-week rate,
(b) an injured team's totals drop vs the no-injury counterfactual,
(c) the rate table is pinned to published research targets,
(d) rates are configurable/overridable through one table, and
(e) a roster with no eligible bench player gets a waiver-wire replacement
(45% of starter PPG), never a zero.
"""
import pytest

# simulate_playoff_odds imports numpy at module load. The fast "lint" CI shard
# has flask but not numpy, so guard on numpy first — otherwise importing the
# sim module here raises at COLLECTION and aborts the whole run. The full
# integration shard has numpy and runs these tests normally.
np = pytest.importorskip("numpy")

import data_building.injury_rates as ir
import data_building.simulate_playoff_odds as spo


@pytest.fixture
def restore_rates():
    """Snapshot the shared injury table; restore it after the test."""
    orig = {
        "hazard": dict(ir._INJURY_HAZARD),
        "default": ir._INJURY_HAZARD_DEFAULT,
        "choices": ir._INJURY_DURATION_CHOICES,
        "probs": ir._INJURY_DURATION_PROBS,
        "replacement": ir._INJURY_REPLACEMENT,
    }
    yield
    ir._INJURY_HAZARD.clear()
    ir._INJURY_HAZARD.update(orig["hazard"])
    ir.override_injury_rates(
        duration_choices=orig["choices"],
        duration_probs=orig["probs"],
        replacement=orig["replacement"],
        default=orig["default"],
    )


def test_hazard_matches_published_games_missed():
    # Research calibration pin: hazard * 17 games must equal the published
    # games-missed-per-season targets the table was built from
    # (zinkelburger snap-count study; see data_building/injury_rates.py).
    targets = {"QB": 2.06, "RB": 2.57, "WR": 2.28, "TE": 2.73}
    for pos, games in targets.items():
        assert ir._INJURY_HAZARD[pos] * 17 == pytest.approx(games, abs=0.05)


def test_injury_events_at_configured_rate(restore_rates):
    # Drive _apply_injuries over a full 17-week season with constant scores
    # (no scoring noise) so every point of score loss is an injury event.
    # The fraction of starter-weeks out must converge to the hazard.
    ir.override_injury_rates(hazard={"RB": 0.06})
    n_sims, kmax, weeks = 4000, 3, 17
    lost = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    haz = np.full(kmax, 0.06, dtype=np.float32)
    onset = haz / np.float32(ir._INJURY_MEAN_DURATION)
    rng = np.random.default_rng(1234)
    state = np.zeros((n_sims, kmax), dtype=np.int16)
    out_weeks = 0
    for _ in range(weeks):
        scores = np.full(n_sims, 200.0, dtype=np.float32)
        scores, state = spo._apply_injuries(rng, scores, lost, onset, state, n_sims)
        # scores = 200 - 10 * n_out exactly (lost=10 constant per slot here)
        n_out = np.round((200.0 - scores) / 10.0).astype(int)
        out_weeks += int(n_out.sum())
    frac = out_weeks / (n_sims * weeks * kmax)
    assert frac == pytest.approx(0.06, abs=0.012), f"injury rate {frac} != 0.06"


def test_injured_team_total_drops_vs_no_injury(restore_rates):
    # Same seed, same profiles: the only difference is injuries on/off.
    # Team totals must drop, by roughly the expected loss per week.
    weeks = list(range(5, 18))
    lost = np.array([10.0, 6.0], dtype=np.float32)
    haz = np.array([0.151, 0.134], dtype=np.float32)  # RB, WR
    profiles = {
        w: {1: {"mean": 110.0, "std": 12.0, "lost": lost, "haz": haz}}
        for w in weeks
    }
    kw = dict(roster_id=1, playing_weeks=weeks, fb_avg=110.0, fb_std=12.0,
              n_sims=4000, seed=99, kmax=2)
    with_inj = np.concatenate([
        spo._score_one_team(week_profiles=profiles, **kw)[w] for w in weeks
    ])
    ir.override_injury_rates(hazard={"RB": 0.0, "WR": 0.0, "QB": 0.0,
                                     "TE": 0.0, "K": 0.0, "DEF": 0.0})
    profiles_off = {
        w: {1: {"mean": 110.0, "std": 12.0, "lost": lost,
                "haz": np.zeros(2, dtype=np.float32)}}
        for w in weeks
    }
    no_inj = np.concatenate([
        spo._score_one_team(week_profiles=profiles_off, **kw)[w] for w in weeks
    ])
    diff = float(no_inj.mean() - with_inj.mean())
    # Expected weekly loss = sum(lost_i * haz_i) = 10*.151 + 6*.134 ≈ 2.31
    assert 0.5 < diff < 4.5, f"unexpected injury drag: {diff}"
    # And no single-week total went UP from injuries (paired streams).
    assert bool((with_inj <= no_inj + 1e-6).all())


def test_override_injury_rates_reconfigures_table(restore_rates):
    ir.override_injury_rates(
        hazard={"QB": 0.5},
        duration_choices=(1, 4),
        duration_probs=(0.5, 0.5),
        replacement=0.9,
    )
    assert ir._INJURY_HAZARD == {"QB": 0.5}
    assert ir._INJURY_MEAN_DURATION == pytest.approx(2.5)
    assert ir._INJURY_REPLACEMENT == pytest.approx(0.9)
    assert ir.injury_onset_rate("QB") == pytest.approx(0.5 / 2.5)
    # The playoff-odds engine reads through the shared module, so an
    # override takes effect without re-importing.
    assert spo._inj._INJURY_HAZARD == {"QB": 0.5}
    assert spo._inj._INJURY_MEAN_DURATION == pytest.approx(2.5)


def test_zero_hazard_disables_injuries(restore_rates):
    ir.override_injury_rates(hazard={})
    n_sims, kmax = 2000, 2
    lost = np.array([10.0, 10.0], dtype=np.float32)
    haz = np.array(
        [ir._INJURY_HAZARD.get("RB", ir._INJURY_HAZARD_DEFAULT)] * kmax,
        dtype=np.float32,
    )
    onset = haz / np.float32(ir._INJURY_MEAN_DURATION)
    rng = np.random.default_rng(7)
    state = np.zeros((n_sims, kmax), dtype=np.int16)
    scores = np.full(n_sims, 200.0, dtype=np.float32)
    out, _ = spo._apply_injuries(rng, scores, lost, onset, state, n_sims)
    assert bool((out == 200.0).all())


def test_no_bench_replacement_uses_waiver_wire_fallback():
    # A roster with zero bench players: the injured starter is replaced by a
    # waiver-wire pickup at 45% of the starter's PPG — never a zero.
    ppg_map = {"p1": {"pos": "QB", "ppg": 20.0}}
    total, starters, repls = spo._lineup_with_replacements(
        ["p1"], ppg_map, {"p1": "QB"},
        ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"],
    )
    assert starters[0][0] == "QB"
    assert repls[0] == pytest.approx(0.45 * 20.0)
    assert repls[0] > 0
