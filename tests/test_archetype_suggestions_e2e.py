"""End-to-end regression tests for the archetype suggestion pipeline.

These drive get_archetype_suggestions with a hand-built league context (no DB,
no live Sleeper, no Monte Carlo state) so the whole route -> scoring -> package
-> assemble path runs offline and degrades to the analytical model. They lock in
the response contract and the invariants the UI depends on (every suggestion is
value-matched, acceptance is bounded, results are ranked), so a future edit to
the engine can't silently regress the pipeline.
"""
import pytest

# The engine lazily imports app / data_building helpers inside try/except and
# degrades, but the value augmentation and sim state need the scientific stack
# to be meaningful; skip cleanly on the pure interpreter.
pytest.importorskip("pandas")

from dashboard_services import archetype_engine as ae


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Keep the pipeline fully offline. get_archetype_suggestions lazily imports
    app (whose init makes live Sleeper calls) and may build sim state, so stub
    the HTTP layer to canned NFL state before any of that runs - otherwise the
    test blocks on network retries."""
    import dashboard_services.api as api

    def _fake_fetch_json(path, timeout=25, retries=3):
        if path == "/state/nfl":
            return {"season": "2026", "week": 0, "leg": 0,
                    "season_type": "off", "display_week": 1,
                    "season_start_date": "2026-09-10"}
        return {}

    monkeypatch.setattr(api, "fetch_json", _fake_fetch_json)


def _seed_ctx():
    """A compact 4-team league: viewer (roster 1) holds mid-tier pieces; rivals
    hold a stud each, so consolidate/contending have real targets and
    distribute/rebuilding have plausible partners."""
    def P(pid, name, pos, val, age, team="FA"):
        return {"id": pid, "name": name, "position": pos, "team": team,
                "age": age, "value": val, "sf_value": val,
                "redraft_value_1qb": val * 0.8, "redraft_value_sf": val * 0.8,
                "pos_rank_label": f"{pos}1", "rank_change_7d": 0}

    table = [
        # Rival studs (roster 2, 3, 4)
        P("r2_rb", "Stud RB", "RB", 1000, 23, "DET"),
        P("r3_wr", "Stud WR", "WR", 950, 24, "CIN"),
        P("r4_qb", "Stud QB", "QB", 700, 25, "BUF"),
        # Rival depth
        P("r2_wr", "Rival WR2", "WR", 300, 27, "DET"),
        P("r3_rb", "Rival RB2", "RB", 280, 28, "CIN"),
        P("r4_te", "Rival TE2", "TE", 260, 26, "BUF"),
        P("r2_young", "Young WR", "WR", 450, 22, "DET"),
        P("r3_young", "Young RB", "RB", 420, 21, "CIN"),
        # Viewer (roster 1): mid-tier and one agable stud for distribute/rebuild
        P("v_wr1", "Viewer WR1", "WR", 560, 26, "PHI"),
        P("v_rb1", "Viewer RB1", "RB", 480, 27, "PHI"),
        P("v_wr2", "Viewer WR2", "WR", 300, 29, "PHI"),
        P("v_rb2", "Viewer RB2", "RB", 220, 30, "PHI"),
        P("v_te1", "Viewer TE1", "TE", 240, 28, "PHI"),
        P("v_stud", "Viewer Stud", "WR", 900, 29, "PHI"),
        P("v_qb1", "Viewer QB1", "QB", 260, 31, "PHI"),
    ]
    rosters = [
        {"roster_id": 1, "players": ["v_wr1", "v_rb1", "v_wr2", "v_rb2", "v_te1", "v_stud", "v_qb1"]},
        {"roster_id": 2, "players": ["r2_rb", "r2_wr", "r2_young"]},
        {"roster_id": 3, "players": ["r3_wr", "r3_rb", "r3_young"]},
        {"roster_id": 4, "players": ["r4_qb", "r4_te"]},
    ]
    return {
        "rosters": rosters,
        "roster_map": {1: "Viewer", 2: "Team Two", 3: "Team Three", 4: "Team Four"},
        "standings_map": {1: 6, 2: 1, 3: 2, 4: 9},  # viewer mid-pack
        "model_value_table": table,
        "picks_by_roster": {},
        "settings": {"playoff_week_start": 15},
    }


def _run(archetype):
    return ae.get_archetype_suggestions(
        archetype=archetype, platform="sleeper", league_id="testlg",
        season=2026, viewer_roster_id="1", league_type="1qb", league_size=10,
        ctx=_seed_ctx(),
    )


@pytest.mark.parametrize("archetype", ["consolidate", "contending", "rebuilding", "distribute"])
def test_pipeline_returns_valid_contract(archetype):
    out = _run(archetype)
    assert isinstance(out, dict)
    assert isinstance(out.get("suggestions"), list)
    assert "current_playoff_pct" in out
    for s in out["suggestions"]:
        # Core contract the UI renders.
        assert s.get("name")
        assert s.get("direction") in ("acquire", "distribute")
        assert 5 <= s.get("acceptance_pct", 0) <= 90
        assert "net_playoff_odds_delta" in s
        # A send package is always present (list; may be empty only as a fallback).
        assert isinstance(s.get("suggested_send"), list)


def test_consolidate_has_targets_and_is_ranked():
    out = _run("consolidate")
    sugg = out["suggestions"]
    assert sugg, "consolidate should surface at least one target for this league"
    # Ranked by the composite key, best first.
    keys = [ae._suggestion_rank(s) for s in sugg]
    assert keys == sorted(keys, reverse=True)
    # Every consolidate package sends 2+ assets (never a 1-for-1 trade-up).
    for s in sugg:
        if s["suggested_send"]:
            assert len(s["suggested_send"]) >= 2


def test_buy_suggestions_carry_fit_note_field():
    # fit_note is always present on acquire suggestions (may be empty string).
    out = _run("consolidate")
    for s in out["suggestions"]:
        assert "fit_note" in s


def test_analytical_impact_is_bounded_without_sim_state(monkeypatch):
    """The Impact table's per-target win % / playoff-odds come from the analytical
    model when a league has no sim state. That model used to mix the target's raw
    dynasty value into a lineup/PPG-scale total, saturating the win-prob curve so
    every strong target showed the same absurd ~+50%. It now scores the whole new
    lineup in consistent units and is clamped, so values stay modest and varied."""
    import time as _t
    ctx = _seed_ctx()
    # Force the no-sim path so the analytical fallback (not simulate_with_swap)
    # produces the per-target impact numbers.
    ae._SIM_CACHE["sleeper:testlg:2026"] = {"sim_state": None, "base_odds": {}, "ts": _t.time()}
    try:
        out = ae.get_archetype_suggestions(
            archetype="consolidate", platform="sleeper", league_id="testlg",
            season=2026, viewer_roster_id="1", league_type="1qb", league_size=10,
            ctx=ctx,
        )
    finally:
        ae._SIM_CACHE.pop("sleeper:testlg:2026", None)

    wpds = [s["win_prob_delta"] for s in out["suggestions"]]
    assert wpds, "expected suggestions on the no-sim analytical path"
    # Clamped to a plausible single-acquisition swing (the old bug was ~0.51).
    assert all(-0.20 <= w <= 0.20 for w in wpds)
    # Not the old degenerate case where every strong target showed one number.
    assert len(set(round(w, 3) for w in wpds)) > 1


def test_monte_carlo_swaps_are_deduped(monkeypatch):
    """simulate_with_swap is deterministic (common-random-numbers seed), so
    identical post-trade rosters must reuse a single 10k run rather than
    re-simulating. Inject a fake sim state and count the calls."""
    import time as _t
    import data_building.simulate_playoff_odds as sim

    calls = {"pids": []}

    def _fake_swap(sim_state, vid, pids_after, n_sims=10_000):
        calls["pids"].append(frozenset(str(p) for p in pids_after))
        return (50.0, 100.0)  # (playoff_pct, new_avg)

    monkeypatch.setattr(sim, "simulate_with_swap", _fake_swap)

    ctx = _seed_ctx()
    viewer_pids = [str(p) for p in ctx["rosters"][0]["players"]]
    fake_state = {
        "ppg_map": {}, "pos_map": {}, "roster_positions": [],
        "teams": [{"avg": 100.0}, {"avg": 98.0}, {"avg": 102.0}, {"avg": 99.0}],
        "roster_pid_map": {1: viewer_pids},
    }
    ae._SIM_CACHE[f"sleeper:testlg:2026"] = {
        "sim_state": fake_state, "base_odds": {1: 50.0}, "ts": _t.time(),
    }
    try:
        out = ae.get_archetype_suggestions(
            archetype="consolidate", platform="sleeper", league_id="testlg",
            season=2026, viewer_roster_id="1", league_type="1qb", league_size=10,
            ctx=ctx,
        )
    finally:
        ae._SIM_CACHE.pop("sleeper:testlg:2026", None)

    sugg = out["suggestions"]
    assert sugg, "expected consolidate suggestions with the injected sim state"

    # Memoization: every simulated roster is unique - no key appears twice.
    assert len(calls["pids"]) == len(set(calls["pids"])), "swap results were not memoized"

    # Each surfaced package's exact post-trade roster (drop the sent players, add
    # the target) must have been simulated - that is the accurate net_* number.
    simulated = set(calls["pids"])
    for s in sugg:
        sent = {str(a.get("player_id")) for a in s["suggested_send"]
                if a.get("player_id") and not a.get("is_pick")}
        roster = frozenset([p for p in viewer_pids if p not in sent] + [str(s["player_id"])])
        assert roster in simulated
