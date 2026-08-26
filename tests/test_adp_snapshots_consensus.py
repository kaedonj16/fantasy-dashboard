"""Snapshot store, source isolation, and capability-aware consensus tests for
dashboard_services.adp_service. Network and DB are never touched: global sources
are read from disk snapshots written into a tmp DATA_DIR, and the DB mirror
degrades silently without a DSN."""

import pytest

from dashboard_services import adp_service as A
from dashboard_services.adp_formats import AdpFormat, EXACT, COMPATIBLE, GENERIC


@pytest.fixture(autouse=True)
def _tmp_data_dir(monkeypatch, tmp_path):
    import utils.paths
    monkeypatch.setattr(utils.paths, "DATA_DIR", tmp_path)
    return tmp_path


# ── Snapshot store ────────────────────────────────────────────────────────────

def test_snapshot_write_read_roundtrip():
    payload = {"adp": {"a": 3.4, "b": 10.0}, "extra": {"a": {"min_pick": 1}},
               "meta": {"scope": "global", "ppr": "unknown"}, "raw_count": 2, "mapped_count": 2}
    assert A.write_adp_snapshot("yahoo", "redraft", 2026, payload) is True
    assert A.snapshot_adp_map("yahoo", "redraft", 2026) == {"a": 3.4, "b": 10.0}
    assert A.snapshot_freshness("yahoo", "redraft", 2026) is not None
    snap = A.load_adp_snapshot("yahoo", "redraft", 2026)
    assert snap["meta"]["scope"] == "global" and snap["extra"]["a"]["min_pick"] == 1


def test_snapshot_empty_does_not_clobber_last_good():
    A.write_adp_snapshot("mfl", "redraft", 2026, {"adp": {"a": 5.0}})
    # An empty fetch (outage / empty response) must retain the last good data.
    assert A.write_adp_snapshot("mfl", "redraft", 2026, {"adp": {}}) is False
    assert A.snapshot_adp_map("mfl", "redraft", 2026) == {"a": 5.0}


def test_snapshot_map_filters_bad_and_nonpositive():
    A.write_adp_snapshot("espn", "redraft", 2026,
                         {"adp": {"a": 1.0, "b": 0, "c": -3, "d": "x", "e": "4.5"}})
    assert A.snapshot_adp_map("espn", "redraft", 2026) == {"a": 1.0, "e": 4.5}


def test_missing_snapshot_is_empty_not_error():
    assert A.snapshot_adp_map("espn", "redraft", 2030) == {}
    assert A.load_adp_snapshot("espn", "redraft", 2030) == {}
    assert A.snapshot_freshness("espn", "redraft", 2030) is None


def test_espn_ppr_rank_is_read_separately():
    A.write_adp_snapshot("espn", "redraft", 2026,
                         {"adp": {"a": 1.2}, "ppr_rank": {"a": 1, "b": 5}})
    # The ADP source map must expose only ADP, never the PPR rank.
    assert A.snapshot_adp_map("espn", "redraft", 2026) == {"a": 1.2}
    assert A.espn_ppr_rank(2026) == {"a": 1.0, "b": 5.0}


# ── Central refresh isolation ─────────────────────────────────────────────────

def test_refresh_isolates_provider_failures(monkeypatch):
    from dashboard_services.providers import global_adp as G
    monkeypatch.setattr(G, "fetch_yahoo_global_adp", lambda s: {"adp": {"y": 1.0}, "mapped_count": 1})
    monkeypatch.setattr(G, "fetch_espn_global_adp",
                        lambda s: (_ for _ in ()).throw(RuntimeError("espn down")))
    monkeypatch.setattr(G, "fetch_mfl_adp", lambda s: {"adp": {"m": 2.0}, "mapped_count": 1})
    summary = A.refresh_global_adp_sources(2026)
    # Yahoo and MFL succeed despite ESPN failing.
    assert summary["yahoo"]["ok"] and summary["mfl"]["ok"]
    assert summary["espn"]["ok"] is False
    assert A.snapshot_adp_map("yahoo", "redraft", 2026) == {"y": 1.0}
    assert A.snapshot_adp_map("mfl", "redraft", 2026) == {"m": 2.0}
    assert A.snapshot_adp_map("espn", "redraft", 2026) == {}


def test_refresh_empty_keeps_last_good(monkeypatch):
    from dashboard_services.providers import global_adp as G
    A.write_adp_snapshot("yahoo", "redraft", 2026, {"adp": {"y": 9.0}})
    monkeypatch.setattr(G, "fetch_yahoo_global_adp", lambda s: {"adp": {}})
    monkeypatch.setattr(G, "fetch_espn_global_adp", lambda s: {"adp": {}})
    monkeypatch.setattr(G, "fetch_mfl_adp", lambda s: {"adp": {}})
    A.refresh_global_adp_sources(2026)
    assert A.snapshot_adp_map("yahoo", "redraft", 2026) == {"y": 9.0}   # retained


# ── yahoo source: OAuth when token, else global snapshot ──────────────────────

def test_yahoo_source_uses_global_snapshot_without_token():
    A.write_adp_snapshot("yahoo", "redraft", 2026, {"adp": {"g": 4.0}})
    assert A._yahoo_adp_source(2026, False, "redraft", None, None) == {"g": 4.0}


def test_yahoo_source_prefers_league_token(monkeypatch):
    A.write_adp_snapshot("yahoo", "redraft", 2026, {"adp": {"g": 4.0}})
    monkeypatch.setattr(A, "fetch_yahoo_adp", lambda lg, tok, s, sf: {"league": 1.0})
    assert A._yahoo_adp_source(2026, False, "redraft", "L", "T") == {"league": 1.0}


def test_yahoo_source_off_redraft_axis_is_empty():
    A.write_adp_snapshot("yahoo", "dynasty", 2026, {"adp": {"g": 4.0}})
    assert A._yahoo_adp_source(2026, False, "dynasty", None, None) == {}


# ── Detailed capability-aware consensus ───────────────────────────────────────

def _mock_sources(monkeypatch, maps):
    monkeypatch.setattr(A, "_raw_source_map",
                        lambda name, season, is_sf, axis, lg=None, tok=None: dict(maps.get(name, {})))


def test_detailed_single_source_is_single_confidence(monkeypatch):
    _mock_sources(monkeypatch, {"sleeper": {"p": 5.0}})
    out = A.resolve_market_adp_detailed(2026, AdpFormat("redraft", "1qb", 1.0))
    rec = out["p"]
    assert rec["source_count"] == 1 and rec["confidence"] == "single-source"
    assert rec["consensus_adp"] == 5.0 and rec["spread"] == 0.0
    assert rec["sources"] == {"sleeper": 5.0}


def test_detailed_two_sources_low_confidence(monkeypatch):
    _mock_sources(monkeypatch, {"sleeper": {"p": 4.0}, "brfantasy": {"p": 6.0}})
    rec = A.resolve_market_adp_detailed(2026, AdpFormat("redraft", "1qb", 1.0))["p"]
    assert rec["source_count"] == 2 and rec["confidence"] == "low"
    assert rec["exact_source_count"] == 2
    assert rec["min_adp"] == 4.0 and rec["max_adp"] == 6.0 and rec["spread"] == 2.0


def test_detailed_three_plus_sources_normal_confidence(monkeypatch):
    _mock_sources(monkeypatch, {"sleeper": {"p": 4.0}, "brfantasy": {"p": 6.0}, "yahoo": {"p": 8.0}})
    rec = A.resolve_market_adp_detailed(2026, AdpFormat("redraft", "1qb", 1.0))["p"]
    assert rec["source_count"] == 3 and rec["confidence"] == "normal"


def test_detailed_five_sources_and_tier_weighting(monkeypatch):
    # exact (sleeper, brfantasy) weigh more than generic (yahoo, espn) and
    # compatible (mfl), so consensus leans toward the exact values.
    _mock_sources(monkeypatch, {
        "sleeper": {"p": 20.0}, "brfantasy": {"p": 22.0},     # exact  (w=3)
        "mfl": {"p": 25.0},                                    # compat (w=2)
        "yahoo": {"p": 30.0}, "espn": {"p": 31.0},             # generic(w=1)
    })
    rec = A.resolve_market_adp_detailed(2026, AdpFormat("redraft", "1qb", 1.0))["p"]
    assert rec["source_count"] == 5 and rec["exact_source_count"] == 2
    assert rec["match_quality"] == EXACT
    plain_mean = (20 + 22 + 25 + 30 + 31) / 5          # 25.6
    weighted = (20*3 + 22*3 + 25*2 + 30*1 + 31*1) / (3+3+2+1+1)  # ~23.9
    assert rec["consensus_adp"] == round(weighted, 2)
    assert rec["consensus_adp"] < plain_mean            # exact sources pull it down
    assert set(rec["sources"]) == {"sleeper", "brfantasy", "mfl", "yahoo", "espn"}


def test_detailed_dynasty_excludes_redraft_only_feeds(monkeypatch):
    _mock_sources(monkeypatch, {
        "sleeper": {"p": 3.0}, "brfantasy": {"p": 4.0},
        "espn": {"p": 99.0}, "yahoo": {"p": 88.0}, "mfl": {"p": 77.0},
    })
    rec = A.resolve_market_adp_detailed(2026, AdpFormat("startup", "superflex", 1.0))["p"]
    assert set(rec["sources"]) == {"sleeper", "brfantasy"}   # redraft feeds excluded


def test_detailed_min_quality_filters_generic(monkeypatch):
    _mock_sources(monkeypatch, {"sleeper": {"p": 4.0}, "yahoo": {"p": 9.0}, "espn": {"p": 10.0}})
    rec = A.resolve_market_adp_detailed(2026, AdpFormat("redraft", "1qb", 1.0),
                                        min_quality=COMPATIBLE)["p"]
    # generic yahoo/espn dropped; only the exact sleeper remains.
    assert set(rec["sources"]) == {"sleeper"}


def test_detailed_restrict_ids(monkeypatch):
    _mock_sources(monkeypatch, {"sleeper": {"a": 1.0, "b": 2.0}, "brfantasy": {"a": 1.5, "b": 2.5}})
    out = A.resolve_market_adp_detailed(2026, AdpFormat("redraft", "1qb", 1.0),
                                        restrict_ids={"a"})
    assert set(out) == {"a"}


def test_detailed_from_legacy_args(monkeypatch):
    _mock_sources(monkeypatch, {"sleeper": {"p": 5.0}, "brfantasy": {"p": 7.0}})
    out = A.resolve_market_adp_detailed(2026, is_sf=False, scoring_type="redraft", ppr=1.0)
    assert out["p"]["source_count"] == 2


# ── Backward compatibility of the simple API ──────────────────────────────────

def test_simple_resolver_still_returns_plain_map(monkeypatch):
    monkeypatch.setattr(A, "fetch_sleeper_adp", lambda season: {"1": {"adp_ppr": 3.0}})
    out = A.resolve_market_adp(2026, False, "redraft", "sleeper")
    assert out == {"1": 3.0}   # unchanged {id: adp} contract


def test_consensus_adp_simple_unchanged():
    assert A.consensus_adp([{"a": 3.0}]) == {"a": 3.0}
    c = A.consensus_adp([{"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 1.0}])
    assert c["a"] == c["b"] == 1.5


def test_source_options_hide_empty_globals_with_season(monkeypatch):
    # No snapshots -> globals hidden when a season is supplied. Stub the
    # retrieve-on-miss path so this assertion is about gating, not live HTTP.
    monkeypatch.setattr(A, "_hydrate_snapshot_from_db", lambda *a, **k: False)
    monkeypatch.setattr(A, "_live_fetch_global_sources", lambda *a, **k: {})
    vals = [v for v, _ in A.adp_source_options("redraft", 2026)]
    assert "espn" not in vals and "yahoo" not in vals and "mfl" not in vals
    assert "sleeper" in vals and "brfantasy" in vals and vals[0] == "consensus"
    A.write_adp_snapshot("espn", "redraft", 2026, {"adp": {"x": 1.0}})
    vals2 = [v for v, _ in A.adp_source_options("redraft", 2026)]
    assert "espn" in vals2                       # shown once it has data
    # Without a season, all configured sources are listed (legacy behavior).
    assert "yahoo" in [v for v, _ in A.adp_source_options("redraft")]


# ── Retrieve-on-miss (Sleeper-style) for Yahoo / ESPN / MFL ───────────────────

def test_espn_source_uses_disk_snapshot_without_refetch(monkeypatch):
    A.write_adp_snapshot("espn", "redraft", 2026, {"adp": {"p": 3.0}})
    monkeypatch.setattr(A, "_hydrate_snapshot_from_db",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("db")))
    monkeypatch.setattr(A, "_live_fetch_global_sources",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("live")))
    assert A._espn_adp_source(2026, False, "redraft") == {"p": 3.0}


def test_espn_source_hydrates_from_db_when_disk_empty(monkeypatch):
    def hydrate(source, axis, season):
        assert source == "espn" and axis == "redraft"
        A.write_adp_snapshot(source, axis, season, {"adp": {"p": 12.0}})
        return True
    monkeypatch.setattr(A, "_hydrate_snapshot_from_db", hydrate)
    live = []
    monkeypatch.setattr(A, "_live_fetch_global_sources",
                        lambda srcs, season: live.append(list(srcs)))
    assert A._espn_adp_source(2026, False, "redraft") == {"p": 12.0}
    assert live == []


def test_espn_source_fetches_live_like_sleeper_when_no_cache(monkeypatch):
    monkeypatch.setattr(A, "_hydrate_snapshot_from_db", lambda *a, **k: False)

    def live(sources, season):
        assert list(sources) == ["espn"]
        A.write_adp_snapshot("espn", "redraft", season, {"adp": {"p": 7.0}})
        return {"espn": {"ok": True, "written": True}}

    monkeypatch.setattr(A, "_live_fetch_global_sources", live)
    assert A._espn_adp_source(2026, False, "redraft") == {"p": 7.0}


def test_yahoo_and_mfl_sources_fetch_on_miss(monkeypatch):
    monkeypatch.setattr(A, "_hydrate_snapshot_from_db", lambda *a, **k: False)
    fetched = []

    def live(sources, season):
        fetched.extend(sources)
        for s in sources:
            A.write_adp_snapshot(s, "redraft", season, {"adp": {s[0]: 4.0}})
        return {s: {"ok": True} for s in sources}

    monkeypatch.setattr(A, "_live_fetch_global_sources", live)
    assert A._yahoo_adp_source(2026, False, "redraft", None, None) == {"y": 4.0}
    assert A._mfl_adp_source(2026, False, "redraft") == {"m": 4.0}
    assert fetched == ["yahoo", "mfl"]


def test_mfl_source_filters_sparse_snapshot_rows(monkeypatch):
    """Already-written MFL snapshots still drop selected-only ADP on read."""
    A.write_adp_snapshot("mfl", "redraft", 2026, {
        "adp": {"jam": 57.76, "star": 2.4},
        "extra": {"jam": {"draft_pct": 10.0}, "star": {"draft_pct": 99.0}},
    })
    monkeypatch.setattr(A, "ensure_global_adp_snapshot", lambda *a, **k: None)
    assert A._mfl_adp_source(2026, False, "redraft") == {"star": 2.4}


def test_source_options_show_globals_after_on_miss_fetch(monkeypatch):
    monkeypatch.setattr(A, "_hydrate_snapshot_from_db", lambda *a, **k: False)

    def live(sources, season):
        for s in sources:
            A.write_adp_snapshot(s, "redraft", season, {"adp": {"x": 1.0}})
        return {s: {"ok": True} for s in sources}

    monkeypatch.setattr(A, "_live_fetch_global_sources", live)
    vals = [v for v, _ in A.adp_source_options("redraft", 2026)]
    assert "espn" in vals and "yahoo" in vals and "mfl" in vals


def test_hydrate_from_db_writes_disk_snapshot(monkeypatch):
    class _Cur:
        def execute(self, sql, params):
            assert params == ("espn", 2026, "redraft")
        def fetchall(self):
            return [
                {"player_id": "a", "adp": 5.5, "draft_pct": 10.0,
                 "min_pick": 1, "max_pick": 9},
                {"player_id": "b", "adp": 10},
            ]
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    class _Conn:
        def cursor(self):
            return _Cur()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr("dashboard_services.db.get_conn", lambda: _Conn())
    assert A._hydrate_snapshot_from_db("espn", "redraft", 2026) is True
    assert A.snapshot_adp_map("espn", "redraft", 2026) == {"a": 5.5, "b": 10.0}
    snap = A.load_adp_snapshot("espn", "redraft", 2026)
    assert snap["extra"]["a"]["draft_pct"] == 10.0
    assert "b" not in (snap.get("extra") or {})


def test_hydrate_from_db_empty_or_error_is_false(monkeypatch):
    monkeypatch.setattr("dashboard_services.db.get_conn",
                        lambda: (_ for _ in ()).throw(RuntimeError("no db")))
    assert A._hydrate_snapshot_from_db("yahoo", "redraft", 2026) is False
    assert A.snapshot_adp_map("yahoo", "redraft", 2026) == {}
