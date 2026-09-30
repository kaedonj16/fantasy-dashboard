"""Progressive-loading contracts for the archetype suggestion engine.

The Strategy view loads in phases:
  phase="slate": the analytical pipeline only. No Monte Carlo state is even
    built; rows whose numbers depend on the sim are marked sim_pending with
    the sim-derived fields set to None (never an estimate dressed as a sim).
  phase="sim" (per group_key): the pipeline re-run with sims gated to one
    player group. Its rows must be exactly the rows the one-shot full phase
    produces for that group (same walk, same numbers), and it reports the
    full slate's group list so the client can reconcile.
  phase="full": the original one-shot behavior, internal phase keys removed.

A deterministic fake sim (patched into data_building.simulate_playoff_odds,
which the engine imports lazily) makes full-vs-gated outputs comparable
offline and lets the tests count exactly which swaps each phase runs.
"""
import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")

from dashboard_services import archetype_engine as ae


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    import dashboard_services.api as api

    def _fake_fetch_json(path, timeout=25, retries=3):
        if path == "/state/nfl":
            return {"season": "2026", "week": 0, "leg": 0,
                    "season_type": "off", "display_week": 1,
                    "season_start_date": "2026-09-10"}
        return {}

    monkeypatch.setattr(api, "fetch_json", _fake_fetch_json)


@pytest.fixture(autouse=True)
def _clear_caches():
    ae._RESULT_CACHE.clear()
    ae._TARGET_SIM_CACHE.clear()
    ae._SIM_CACHE.clear()
    yield
    ae._RESULT_CACHE.clear()
    ae._TARGET_SIM_CACHE.clear()
    ae._SIM_CACHE.clear()


@pytest.fixture
def fake_sim(monkeypatch):
    """Deterministic stand-in for the Monte Carlo layer. Swap results are a
    pure function of the swapped roster, so a gated walk and a full walk
    produce identical numbers for the rows they share."""
    import data_building.simulate_playoff_odds as spo

    calls = {"swap": [], "build_state": 0, "base": 0}

    def _ppg(ctx):
        ppg, pos = {}, {}
        for p in ctx["model_value_table"]:
            ppg[p["id"]] = float(p["value"]) / 100.0
            pos[p["id"]] = p["position"]
        return ppg, pos

    def build_sim_state(ctx, platform=None):
        calls["build_state"] += 1
        ppg_map, pos_map = _ppg(ctx)
        teams = []
        for r in ctx["rosters"]:
            pids = [str(x) for x in r["players"]]
            avg = (sum(ppg_map.get(p, 0.0) for p in pids) / max(1, len(pids))) * 8.0
            teams.append({"roster_id": int(r["roster_id"]), "avg": avg})
        return {"ppg_map": ppg_map, "pos_map": pos_map, "teams": teams,
                "roster_positions": ctx.get("roster_positions") or []}

    def run_base_simulation(sim_state, n_sims=2000):
        calls["base"] += 1
        return {1: 42.0, 2: 55.0, 3: 50.0, 4: 30.0}

    def simulate_with_swap(sim_state, roster_id, new_pids, n_sims=2000):
        calls["swap"].append((roster_id, tuple(str(x) for x in new_pids)))
        ppg = sim_state["ppg_map"]
        total = sum(ppg.get(str(x), 0.0) for x in new_pids)
        return (min(99.0, max(1.0, total % 89 + 5)), total / 8.0)

    monkeypatch.setattr(spo, "build_sim_state", build_sim_state)
    monkeypatch.setattr(spo, "run_base_simulation", run_base_simulation)
    monkeypatch.setattr(spo, "build_ppg_map", lambda ctx: _ppg(ctx))
    monkeypatch.setattr(spo, "simulate_with_swap", simulate_with_swap)
    monkeypatch.setattr(spo, "playoff_schedule_sig", lambda ctx, platform: "sig")
    return calls


def _seed_ctx():
    """Same compact league as the e2e suite: viewer mid-tier, rivals hold a
    stud each, so every archetype has real work to do."""
    def P(pid, name, pos, val, age, team="FA"):
        return {"id": pid, "name": name, "position": pos, "team": team,
                "age": age, "value": val, "sf_value": val,
                "redraft_value_1qb": val * 0.8, "redraft_value_sf": val * 0.8,
                "pos_rank_label": f"{pos}1", "rank_change_7d": 0}

    table = [
        P("r2_rb", "Stud RB", "RB", 1000, 23, "DET"),
        P("r3_wr", "Stud WR", "WR", 950, 24, "CIN"),
        P("r4_qb", "Stud QB", "QB", 700, 25, "BUF"),
        P("r2_wr", "Rival WR2", "WR", 300, 27, "DET"),
        P("r2_rb2", "Rival RB3", "RB", 500, 24, "DET"),
        P("r3_rb", "Rival RB2", "RB", 280, 28, "CIN"),
        P("r4_te", "Rival TE2", "TE", 260, 26, "BUF"),
        P("r2_young", "Young WR", "WR", 450, 22, "DET"),
        P("r3_young", "Young RB", "RB", 420, 21, "CIN"),
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
        {"roster_id": 2, "players": ["r2_rb", "r2_wr", "r2_young", "r2_rb2"]},
        {"roster_id": 3, "players": ["r3_wr", "r3_rb", "r3_young"]},
        {"roster_id": 4, "players": ["r4_qb", "r4_te"]},
    ]
    return {
        "rosters": rosters,
        "roster_map": {1: "Viewer", 2: "Team Two", 3: "Team Three", 4: "Team Four"},
        "standings_map": {1: 6, 2: 1, 3: 2, 4: 9},
        "model_value_table": table,
        "picks_by_roster": {},
        "settings": {"playoff_week_start": 15},
        "teams": [
            {"team_id": 1, "roster_id": 1, "is_mine": True,
             "players": ["v_wr1", "v_rb1", "v_wr2", "v_rb2", "v_te1", "v_stud", "v_qb1"]},
            {"team_id": 2, "roster_id": 2,
             "players": ["r2_rb", "r2_wr", "r2_young", "r2_rb2"]},
            {"team_id": 3, "roster_id": 3,
             "players": ["r3_wr", "r3_rb", "r3_young"]},
            {"team_id": 4, "roster_id": 4,
             "players": ["r4_qb", "r4_te"]},
        ],
    }


VIEWER_PLAYERS = {"v_wr1", "v_rb1", "v_wr2", "v_rb2", "v_te1", "v_stud", "v_qb1"}
ARCHETYPES = ["consolidate", "contending", "rebuilding", "distribute"]


def _run(archetype, **kw):
    return ae.get_archetype_suggestions(
        archetype=archetype, platform="sleeper", league_id="testlg",
        season=2026, viewer_roster_id="1", league_type="1qb", league_size=10,
        ctx=_seed_ctx(), **kw,
    )


def _row_key(r):
    return (
        str(r.get("player_id")),
        tuple(str(a.get("player_id")) for a in (r.get("suggested_send") or [])),
        tuple(str(a.get("player_id")) for a in (r.get("suggested_receive") or [])),
    )


@pytest.mark.parametrize("archetype", ARCHETYPES)
def test_slate_phase_runs_no_sims_and_marks_pending(archetype, fake_sim):
    out = _run(archetype, phase="slate")
    assert out["phase"] == "slate"
    assert out["current_playoff_pct"] is None
    # The slate phase must not even build Monte Carlo state.
    assert fake_sim["build_state"] == 0
    assert fake_sim["swap"] == []
    rows = out["suggestions"]
    assert rows, f"{archetype} slate unexpectedly empty"
    # groups = the rows' group keys, in row order, deduplicated.
    expect = list(dict.fromkeys(str(r["group_key"]) for r in rows))
    assert out["groups"] == expect
    for r in rows:
        assert r["group_key"]
        if r["sim_pending"]:
            assert r["rank"] is None
            for f in ae._sim_delta_fields(archetype):
                assert r[f] is None, f"{archetype} pending row leaked a value in {f}"
        else:
            assert r["rank"] is not None


@pytest.mark.parametrize("archetype", ARCHETYPES)
def test_sim_phase_rows_equal_progressive_full(archetype, fake_sim):
    """The progressive pipeline is internally consistent: the slate phase
    announces every group the sim phase can produce, and the per-group sim
    results are exactly the rows of one ungated sim run on that same
    analytical slate (same walk, same numbers). The ungated reference run
    uses force_analytical_slate, the sim phase's own selection basis; the
    one-shot full phase keeps its historical sim-baseline selection and is
    deliberately not the reference here."""
    slate = _run(archetype, phase="slate")
    expected = ae._get_archetype_suggestions_impl(
        archetype=archetype, platform="sleeper", league_id="testlg", season=2026,
        viewer_roster_id="1", league_type="1qb", league_size=10,
        ctx=_seed_ctx(), force_analytical_slate=True,
    )
    exp_by_group = {}
    for r in expected["suggestions"]:
        exp_by_group.setdefault(ae._row_group_key(r, archetype), []).append(r)
    # No group materializes at sim time that the slate never announced.
    assert set(exp_by_group) <= set(slate["groups"])
    # The slate may carry rows the sim filters drop, never the reverse.
    slate_keys = {_row_key(r) for r in slate["suggestions"]}
    assert {_row_key(r) for r in expected["suggestions"]} <= slate_keys
    gathered = []
    for gk in slate["groups"]:
        out = _run(archetype, phase="sim", group_key=gk)
        assert out["phase"] == "sim"
        assert out["group_key"] == gk
        assert out["current_playoff_pct"] == 42.0
        # The sim walk reports only groups that produced sim rows, which
        # (consolidate's sim filter) can be a subset of the slate's groups.
        # The hard direction: it never invents a group the slate lacked.
        assert set(out["slate_groups"]) <= set(slate["groups"])
        expect = exp_by_group.get(gk, [])
        assert [_row_key(r) for r in out["suggestions"]] == [_row_key(r) for r in expect]
        for got, want in zip(out["suggestions"], expect):
            for f in ("win_prob_delta", "playoff_odds_delta", "net_win_prob_delta",
                      "net_playoff_odds_delta", "acceptance_pct"):
                assert got[f] == want[f], f"{archetype}/{gk} field {f} diverged"
            assert got["rank"] == ae._suggestion_rank(want)
        gathered.extend(out["suggestions"])
    assert sorted(_row_key(r) for r in gathered) == \
           sorted(_row_key(r) for r in expected["suggestions"])


@pytest.mark.parametrize("archetype", ARCHETYPES)
def test_sim_phase_sims_only_the_requested_group(archetype, fake_sim):
    """Gating works: a group call runs swaps only for that group, so the
    progressive total work stays close to one full run instead of one full
    run per player."""
    slate = _run(archetype, phase="slate")
    assert slate["groups"]
    gk = slate["groups"][0]
    fake_sim["swap"].clear()
    out = _run(archetype, phase="sim", group_key=gk)
    if archetype != "consolidate":
        assert out["suggestions"]
    # (Consolidate's sim filter can legitimately drop every package of the
    # first group; the swap gating below is the assertion that matters.)
    assert fake_sim["swap"], "sim phase ran no swaps at all"
    is_sell = archetype in ("distribute", "rebuilding")
    for _rid, roster in fake_sim["swap"]:
        roster_set = set(roster)
        if is_sell:
            # Sell swaps remove exactly the grouped vet/stud from the lineup.
            assert VIEWER_PLAYERS - roster_set == {gk}
        else:
            # Acquire swaps add exactly the grouped target.
            assert roster_set - VIEWER_PLAYERS == {gk}


def test_sim_phase_result_is_cached(fake_sim):
    slate = _run("contending", phase="slate")
    gk = slate["groups"][0]
    first = _run("contending", phase="sim", group_key=gk)
    swaps_after_first = len(fake_sim["swap"])
    assert swaps_after_first > 0
    second = _run("contending", phase="sim", group_key=gk)
    assert len(fake_sim["swap"]) == swaps_after_first, "repeat group sim must hit the cache"
    assert second == first


@pytest.mark.parametrize("archetype", ARCHETYPES)
def test_full_phase_has_no_internal_keys(archetype, fake_sim):
    out = _run(archetype, phase="full")
    assert "_sim_available" not in out
    assert "_all_group_keys" not in out
    assert "phase" not in out
    for r in out["suggestions"]:
        assert "sim_pending" not in r
        assert "group_key" not in r
