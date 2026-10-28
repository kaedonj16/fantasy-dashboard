"""Duplicate-call fixes for the Trade Hub / Calculator (audit 2026-10-27).

Four findings, one contract each — numbers identical, call pattern changed:

1. Client: one settled edit = one eval POST. recomputeTrade coalesces the
   same-tick burst (renderChips A + B + explicit calls), aborts the
   superseded in-flight eval, and analyzeTrade's POST is abortable too.
   Trade intel is ONE batch request per edit with a per-session cache
   (was: one GET per player per edit, no cache).
2. Archetype slate cache is now READ by the sim phase: a group fill off
   the cached slate must not re-run the suggestions pipeline, and a stale
   or missing slate falls back to the pipeline re-run.
3. Playoff Impact serves only from the frozen shared sim-state snapshot:
   a cold snapshot defers (background warm kicked, {pending: true} back)
   and never runs a swap on the request thread; the viewer + flipped
   partner calls share one week-profile build / overlay standings run.
4. /api/trade-targets: the calculator's two loaders share one promise.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")


def _js_function_body(src: str, name: str) -> str:
    """Extract a function body by brace counting (indent-agnostic)."""
    m = re.search(rf"function {name}\([^)]*\) \{{", src)
    assert m, f"{name}() not found"
    depth = 0
    i = m.end() - 1
    start = i + 1
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i]
        i += 1
    raise AssertionError(f"{name}() body not closed")


# ── Finding 1: client coalescing / abort / batch ─────────────────────────────

def test_recompute_coalesces_same_tick_burst_before_generation():
    body = _js_function_body(APP_JS, "recomputeTrade")
    burst = body.index("const _burst = ++_recomputeBurst;")
    guard = body.index("if (_burst !== _recomputeBurst) return;")
    gen = body.index("const gen = ++_tradeGeneration;")
    assert burst < guard < gen, (
        "the burst ticket + supersede guard must run before the generation "
        "increment, so sibling calls in one tick never start work"
    )
    assert "await Promise.resolve();" in body


def test_recompute_aborts_superseded_eval_and_stays_quiet():
    body = _js_function_body(APP_JS, "recomputeTrade")
    assert "if (_evalAbort) _evalAbort.abort();" in body
    assert "signal: _evalAbort.signal" in body
    catch = body.index('console.error("[trade] error in recomputeTrade:"')
    abort_guard = body.index('err.name === "AbortError"')
    assert abort_guard < catch, "an aborted eval must not paint or retry"


def test_analyze_trade_post_is_abortable():
    body = _js_function_body(APP_JS, "analyzeTrade")
    assert "if (_analyzeAbort) _analyzeAbort.abort();" in body
    assert "signal: _analyzeAbort.signal" in body
    assert 'err.name === "AbortError"' in body


def test_trade_intel_is_one_batch_call_with_session_cache():
    body = _js_function_body(APP_JS, "fetchTradeIntel")
    assert "/api/trade-intel/players?ids=" in body
    assert "/api/trade-intel/player/${" not in body, (
        "the per-player fan-out must be gone from the calculator"
    )
    assert "_tradeIntelCache" in body


def test_playoff_impact_polls_the_deferred_state():
    body = _js_function_body(APP_JS, "fetchPlayoffImpact")
    assert "data.pending" in body
    assert "_piPendingAttempts" in body
    # The pending branch must not fall into the terminal "Not available"
    # paint on the first deferred answer: it is checked before it.
    assert body.index("data.pending") < body.index('"Not available"')


def test_targets_loaders_share_one_fetch():
    assert "function _fetchTradeTargetsShared(url)" in APP_JS
    for loader in ("loadTradeTargets", "loadSuggTargets"):
        body = _js_function_body(APP_JS, loader)
        assert "_fetchTradeTargetsShared(" in body, loader
        assert "await fetch(" not in body, f"{loader} still fetches directly"


# ── Finding 2: slate cache read ──────────────────────────────────────────────
# (Heavy imports stay inside fixtures/tests so the client pins above still
# run on the lint shard, where pandas/numpy are absent.)

@pytest.fixture
def _offline(monkeypatch):
    pytest.importorskip("pandas")
    import dashboard_services.api as api

    def _fake_fetch_json(path, timeout=25, retries=3):
        if path == "/state/nfl":
            return {"season": "2026", "week": 0, "leg": 0,
                    "season_type": "off", "display_week": 1,
                    "season_start_date": "2026-09-10"}
        return {}

    monkeypatch.setattr(api, "fetch_json", _fake_fetch_json)


@pytest.fixture
def _clear_caches():
    pytest.importorskip("pandas")
    from dashboard_services import archetype_engine as ae

    ae._RESULT_CACHE.clear()
    ae._TARGET_SIM_CACHE.clear()
    ae._SIM_CACHE.clear()
    ae._SLATE_CACHE.clear()
    yield
    ae._RESULT_CACHE.clear()
    ae._TARGET_SIM_CACHE.clear()
    ae._SIM_CACHE.clear()
    ae._SLATE_CACHE.clear()


@pytest.fixture
def fake_sim(monkeypatch):
    """Deterministic stand-in for the Monte Carlo layer (mirrors
    tests/test_archetype_progressive.py)."""
    pytest.importorskip("pandas")
    pytest.importorskip("numpy")
    import data_building.simulate_playoff_odds as spo

    calls = {"swap": [], "build_state": 0, "base": 0}

    def _ppg(ctx):
        ppg, pos = {}, {}
        for p in ctx["model_value_table"]:
            ppg[p["id"]] = {"ppg": float(p["value"]) / 100.0, "pos": p["position"]}
            pos[p["id"]] = p["position"]
        return ppg, pos

    def build_sim_state(ctx, platform=None):
        calls["build_state"] += 1
        ppg_map, pos_map = _ppg(ctx)
        teams = []
        for r in ctx["rosters"]:
            pids = [str(x) for x in r["players"]]
            avg = (sum(ppg_map[p]["ppg"] for p in pids if p in ppg_map)
                   / max(1, len(pids))) * 8.0
            teams.append({"roster_id": int(r["roster_id"]), "avg": avg})
        return {"ppg_map": ppg_map, "pos_map": pos_map, "teams": teams,
                "roster_positions": ctx.get("roster_positions") or []}

    def run_base_simulation(sim_state, n_sims=2000):
        calls["base"] += 1
        return {1: 42.0, 2: 55.0, 3: 50.0, 4: 30.0}

    def simulate_with_swap(sim_state, roster_id, new_pids, n_sims=2000):
        calls["swap"].append((roster_id, tuple(str(x) for x in new_pids)))
        ppg = sim_state["ppg_map"]
        total = sum(ppg.get(str(x), {}).get("ppg", 0.0) for x in new_pids)
        return (min(99.0, max(1.0, total % 89 + 5)), total / 8.0)

    monkeypatch.setattr(spo, "build_sim_state", build_sim_state)
    monkeypatch.setattr(spo, "run_base_simulation", run_base_simulation)
    monkeypatch.setattr(spo, "build_ppg_map", lambda ctx: _ppg(ctx))
    monkeypatch.setattr(spo, "simulate_with_swap", simulate_with_swap)
    monkeypatch.setattr(spo, "playoff_schedule_sig", lambda ctx, platform: "sig")
    return calls


def _seed_ctx():
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
            {"team_id": 4, "roster_id": 4, "players": ["r4_qb", "r4_te"]},
        ],
    }


def _run(archetype, **kw):
    from dashboard_services import archetype_engine as ae
    return ae.get_archetype_suggestions(
        archetype=archetype, platform="sleeper", league_id="testlg",
        season=2026, viewer_roster_id="1", league_type="1qb", league_size=10,
        ctx=_seed_ctx(), **kw,
    )


@pytest.fixture
def impl_counter(monkeypatch):
    pytest.importorskip("pandas")
    from dashboard_services import archetype_engine as ae
    real = ae._get_archetype_suggestions_impl
    count = {"n": 0}

    def counting(*a, **k):
        count["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(ae, "_get_archetype_suggestions_impl", counting)
    return count


@pytest.mark.parametrize("archetype", ["contending", "consolidate", "distribute", "rebuilding"])
def test_sim_phase_fills_off_slate_without_rerunning_pipeline(
        archetype, fake_sim, impl_counter, _offline, _clear_caches):
    slate = _run(archetype, phase="slate")
    assert impl_counter["n"] == 1
    assert slate["groups"]
    for gk in slate["groups"]:
        out = _run(archetype, phase="sim", group_key=gk)
        assert out["phase"] == "sim"
    assert impl_counter["n"] == 1, (
        "every group sim must be served from the cached slate; the "
        "pipeline ran again"
    )
    assert fake_sim["swap"], "the fill must still run the group's swaps"


def test_stale_slate_falls_back_to_pipeline(fake_sim, impl_counter, _offline, _clear_caches):
    from dashboard_services import archetype_engine as ae
    slate = _run("contending", phase="slate")
    assert impl_counter["n"] == 1
    for entry in ae._SLATE_CACHE.values():
        entry["ts"] -= ae._RESULT_CACHE_TTL + 1
    _run("contending", phase="sim", group_key=slate["groups"][0])
    assert impl_counter["n"] == 2, "a slate past the TTL must not be served"


def test_sim_without_slate_still_uses_pipeline(fake_sim, impl_counter, _offline, _clear_caches):
    out = _run("contending", phase="sim", group_key="r2_rb")
    assert impl_counter["n"] == 1
    assert out["phase"] == "sim"


def test_slate_cache_keeps_analytical_values_after_pending_nulls(fake_sim, _offline, _clear_caches):
    """The client-facing slate nulls sim fields on pending rows; the cached
    copy must keep the analytical values the fill starts from."""
    from dashboard_services import archetype_engine as ae
    out = _run("contending", phase="slate")
    pending = [r for r in out["suggestions"] if r.get("sim_pending")]
    assert pending
    entry = next(iter(ae._SLATE_CACHE.values()))
    cached = {str(r["player_id"]): r for r in entry["rows"]}
    for r in pending:
        crow = cached[str(r["player_id"])]
        assert crow["win_prob_delta"] is not None
        assert crow is not r, "cached rows must be copies, not the live rows"


# ── Finding 3: shared week-profile build for both sides of a swap ────────────

def test_swap_impact_pair_shares_one_profile_build(monkeypatch):
    pytest.importorskip("numpy")
    import data_building.simulate_playoff_odds as spo

    counts = {"profiles": 0, "standings": 0}
    monkeypatch.setattr(spo, "_ensure_freeze", lambda state, n: {
        "result": [
            {"roster_id": 1, "playoff_pct": 50.0, "avg_final_wins": 7.0, "top3_pick_pct": 2.0},
            {"roster_id": 2, "playoff_pct": 40.0, "avg_final_wins": 6.0, "top3_pick_pct": 3.0},
        ]})

    def fake_profiles(sim_state, overrides):
        counts["profiles"] += 1
        return {}

    def fake_standings(sim_state, after_profiles, overrides, n_sims):
        counts["standings"] += 1
        bump = float(len(overrides))
        return [
            {"roster_id": 1, "playoff_pct": 50.0 + bump, "avg_final_wins": 7.0, "top3_pick_pct": 2.0},
            {"roster_id": 2, "playoff_pct": 40.0 - bump, "avg_final_wins": 6.0, "top3_pick_pct": 3.0},
        ]

    monkeypatch.setattr(spo, "_override_profiles", fake_profiles)
    monkeypatch.setattr(spo, "_standings_with_overrides", fake_standings)

    state = {
        "teams": [{"roster_id": 1}, {"roster_id": 2}],
        "week_profiles": {},
        "roster_pid_map": {1: ["a", "b"], 2: ["c", "d"]},
    }
    shared: dict = {}
    viewer = spo.simulate_swap_impact(state, 1, ["a"], ["c"], _shared=shared)
    partner = spo.simulate_swap_impact(state, 2, ["c"], ["a"], _shared=shared)
    assert counts == {"profiles": 1, "standings": 1}, (
        "the flipped partner call must reuse the viewer's profile build "
        "and overlay standings"
    )
    # And the shared results equal two independent computations.
    viewer2 = spo.simulate_swap_impact(state, 1, ["a"], ["c"])
    partner2 = spo.simulate_swap_impact(state, 2, ["c"], ["a"])
    assert viewer == viewer2
    assert partner == partner2
    assert viewer["available"] and partner["available"]


# ── Finding 3: the handler defers on a cold snapshot ─────────────────────────

def test_impact_handler_structure_defers_cold_snapshot():
    start = APP_PY.index("def api_trade_eval_playoff_impact")
    end = APP_PY.index("\ndef _trade_future_outlook", start)
    body = APP_PY[start:end]
    assert "peek_sim_state" in body
    assert "_kick_impact_sim_warm" in body
    assert '"pending": True' in body
    assert "build_sim_state" not in body, (
        "the handler must never build sim state on the request thread"
    )
    assert "_shared=_shared" in body


@pytest.mark.integration
def test_impact_handler_cold_snapshot_defers_then_serves(monkeypatch):
    flask = pytest.importorskip("flask")  # noqa: F841
    import app as appmod
    import data_building.simulate_playoff_odds as spo

    ctx = {
        "league_id": "lgpi", "season": 2026, "current_week": 5,
        "league_settings": {"playoff_week_start": 15, "playoff_teams": 6},
        "rosters": [
            {"roster_id": 1, "players": ["a", "b"]},
            {"roster_id": 2, "players": ["c", "d"]},
        ],
        "roster_positions": [],
    }
    fake_state = {
        "teams": [{"roster_id": 1, "avg": 100.0}, {"roster_id": 2, "avg": 95.0}],
        "ppg_map": {}, "pos_map": {}, "roster_positions": [],
        "roster_pid_map": {1: ["a", "b"], 2: ["c", "d"]},
        "week_profiles": {},
    }
    swap_calls = []

    def fake_swap(sim_state, rid, give, get, n_sims=2000, _shared=None):
        swap_calls.append({"rid": rid, "shared": _shared})
        return {
            "available": True,
            "before": {"playoff_pct": 50.0, "avg_final_wins": 7.0, "avg_ppg": 100.0, "top3_pick_pct": 1.0},
            "after": {"playoff_pct": 55.0, "avg_final_wins": 7.5, "avg_ppg": 102.0, "top3_pick_pct": 1.0},
            "delta": {"playoff_pct": 5.0, "avg_final_wins": 0.5, "avg_ppg": 2.0, "top3_pick_pct": 0.0},
            "league_avg_ppg": 97.5,
            "missing_give_ids": [],
        }

    monkeypatch.setattr(appmod, "has_premium_for_viewer", lambda *a, **k: True)
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", lambda *a, **k: ctx)
    monkeypatch.setattr(appmod, "_league_is_redraft", lambda c: True)
    monkeypatch.setattr(spo, "_build_sim_state_impl", lambda c, p: fake_state)
    monkeypatch.setattr(
        spo, "run_base_simulation",
        lambda state, n_sims=2000: state.__setitem__("_mc_freeze", {"n_sims": n_sims}) or {})
    monkeypatch.setattr(spo, "simulate_swap_impact", fake_swap)
    spo._SIM_STATE_CACHE.clear()
    appmod._IMPACT_SIM_BUILDING.clear()
    appmod._IMPACT_SIM_WARMED.clear()

    client = appmod.app.test_client()
    payload = {
        "league_id": "lgpi", "platform": "sleeper", "season": 2026,
        "roster_id": 1, "give_ids": ["a"], "get_ids": ["c"],
        "opponent_roster_id": 2,
    }
    env = {"REMOTE_ADDR": "10.44.0.7"}

    resp = client.post("/api/trade-eval/playoff-impact", json=payload, environ_overrides=env)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["available"] is False and data.get("pending") is True
    assert swap_calls == [], "no swap may run on the request thread while cold"

    served = None
    for _ in range(40):
        time.sleep(0.25)
        resp = client.post("/api/trade-eval/playoff-impact", json=payload, environ_overrides=env)
        if resp.get_json().get("available"):
            served = resp.get_json()
            break
    assert served is not None, "the warmed snapshot never served"
    assert served["before"]["playoff_pct"] == 50.0
    assert served["partner"]["after"]["playoff_pct"] == 55.0
    assert len(swap_calls) == 2
    assert swap_calls[0]["shared"] is swap_calls[1]["shared"], (
        "viewer and partner calls must share the profile-build cache"
    )


# ── Finding 1 (server): batch intel payloads equal single payloads ───────────

class _FakeResult:
    def __init__(self, row=None, rows=None):
        self._row = row
        self._rows = rows or []

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._rows


class _FakeConn:
    """Minimal conn stub: stat rows + package rows keyed by player id."""

    STATS = {
        "p1": {"trade_count_7d": 3, "trade_count_30d": 9, "trade_count": 21,
               "buy_sell_ratio": 0.7, "avg_package_value": 4200.0,
               "calibration_source": "model", "model_value": 5000.0,
               "market_value": 5300.0, "calibrated_value": 5100.0},
        "p2": {"trade_count_7d": 0, "trade_count_30d": 1, "trade_count": 4,
               "buy_sell_ratio": None, "avg_package_value": None,
               "calibration_source": None, "model_value": 0.0,
               "market_value": 0.0, "calibrated_value": 0.0},
    }

    def execute(self, sql, params=()):
        if "trade_intel_player_stats" in sql:
            pid = params[1]
            return _FakeResult(row=self.STATS.get(pid))
        if "trade_intel_packages" in sql:
            return _FakeResult(rows=[{"package_key": "p9|p8", "occurrence_count": 2}])
        raise AssertionError(f"unexpected SQL: {sql[:60]}")


def test_intel_payload_helper_parity(monkeypatch):
    import app as appmod

    extras = {"p1": {"velocity": {"recent_90d": 5, "prior_90d": 2, "trend": "up"},
                     "buyer_split": {"contender": {"count": 3}}}}
    monkeypatch.setattr(appmod, "_trade_intel_col_exists", lambda conn, col: False)
    monkeypatch.setattr(appmod, "trade_intel_card_extras",
                        lambda ids: {p: extras[p] for p in ids if p in extras})
    players_map = {"p8": {"name": "Eight", "pos": "WR"}, "p9": {"name": "Nine", "pos": "RB"}}
    conn = _FakeConn()

    for pid in ("p1", "p2", "ghost"):
        single = appmod._trade_intel_player_payload(
            conn, pid, 2026, "1qb", 10, players_map, 2027)
        batch = appmod._trade_intel_player_payload(
            conn, pid, 2026, "1qb", 10, players_map, 2027,
            extras=appmod.trade_intel_card_extras([pid]))
        assert single == batch, pid
    p1 = appmod._trade_intel_player_payload(conn, "p1", 2026, "1qb", 10, players_map, 2027)
    assert p1["trade_count_all"] == 21
    assert p1["value_delta"] == 300.0
    assert p1["velocity"]["recent_90d"] == 5
    ghost = appmod._trade_intel_player_payload(conn, "ghost", 2026, "1qb", 10, players_map, 2027)
    assert ghost["trade_count"] == 0 and ghost["common_packages"]
