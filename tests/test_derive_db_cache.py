"""
DB-backed derive state (derive v1.5).

Files written to the app tree from a Render shell do not persist:
shell containers recycle and take the overlay with them, while
/tmp and Postgres survive. The derive's file cache - and a
completed artifact file - vanished within minutes, three separate
times. So when DATABASE_URL is set, the per-class cache lives in
prospect_v4_derive_cache (per-class upserts, committed
immediately, mirroring the reference-population build that
survived the same chaos) and the finished artifact is upserted
into prospect_v4_reference_artifact for retrieval via
scripts/print_v4_reference.py. The file backend remains for
local runs/tests (no DATABASE_URL).

Pinned here:
- DB round-trip: a GP-bearing class written to Postgres reads
  back as a cache HIT and build_reference pools season legs from
  it (no live load, no cache file);
- miss rules are the file backend's, shared: stale cache_version
  and GP-less rows read as misses;
- sub-floor classes are never written to Postgres;
- the artifact upsert stores the exact artifact dict and
  print_v4_reference emits it (plain and --base64 round-trips),
  failing closed with no DATABASE_URL or no stored artifact;
- backend selection: file when DATABASE_URL is unset;
- --no-cache forces live loads under the DB backend;
- a failing cache DB degrades to live loads with a warning, and
  a failing artifact store warns without failing the run.

No network, no live DB: the backtest machinery and the model
scorer are faked via monkeypatch (same pattern as
test_derive_v4_reference.py, including the targets-stub
containment registration) and both scripts' _connect is an
in-memory fake psycopg connection.
"""

import base64
import copy
import json
import os
import sys

import pytest

pytest.importorskip("pandas")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tests"))
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import derive_v4_reference as drv  # noqa: E402
import print_v4_reference as pvr  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Fake machinery (prospects + scorer), mirroring
# test_derive_v4_reference.py so the loaded classes are shaped
# exactly as the live path extracts them.
# ─────────────────────────────────────────────────────────────────────

def _prospect(pid, pos="WR", with_seasons=True, dom=0.30, gp=12):
    seasons = [{
        "season": 2022, "games_played": gp, "team": "State",
        "conference": "SEC",
        "dominator_rating": dom, "market_share_yards": 0.30,
        "market_share_tds": 0.28, "receptions": 60, "targets": 90,
        "receiving_yards": 900, "receiving_tds": 8,
        "yds_per_reception": 15.0,
        "rush_attempts": 5, "rush_yards": 40, "rush_tds": 0,
        "pass_attempts": 0, "pass_yards": 0, "pass_tds": 0,
        "team_pass_yards": 3000, "team_total_yards": 5000,
    }] if with_seasons else []
    return {
        "player_id": pid, "name": pid, "position": pos,
        "school": "State", "height_inches": 73,
        "age": 21.5, "weight_lbs": 200,
        "seasons": seasons,
        "athleticism": {"forty_yard": 4.45},
        "recruiting": {"composite_rating": 0.95},
    }


def _score_row(p, base=70.0):
    return {"player_id": p["player_id"],
            "production_score": base,
            "efficiency_score": base - 5,
            "breakout_profile_score": base - 10,
            "athleticism_score": base - 15,
            "prospect_score": base + 20}


def _install_fake_machinery(monkeypatch, classes_map):
    import backtest_prospect_model as bt
    from data_building.rookie_pipeline import ingestion, prospect_model as pm

    # run_real's targets stub is a deliberate process-global
    # mutation; registering the original with monkeypatch keeps it
    # from leaking into other test modules in this process.
    monkeypatch.setattr(ingestion, "fetch_cfbd_targets_from_plays",
                        ingestion.fetch_cfbd_targets_from_plays)

    counts = {"cfbd_loads": 0}

    def fake_cfbd(year, draft_class, fetch_games_played=False,
                  gp_stateless=False):
        counts["cfbd_loads"] += 1
        return {}

    def fake_build(draft_class, athleticism, year, college_stats):
        return ([copy.deepcopy(p) for p in classes_map[year]], {})

    monkeypatch.setattr(bt, "_load_draft_class", lambda year: [{"x": 1}])
    monkeypatch.setattr(bt, "_load_combine_athleticism", lambda year: {})
    monkeypatch.setattr(bt, "_load_cfbd_college_stats", fake_cfbd)
    monkeypatch.setattr(bt, "_build_prospect_dicts", fake_build)
    monkeypatch.setattr(
        pm, "score_all_prospects",
        lambda prospects, consensus, skip_sagarin=True:
            [_score_row(p) for p in prospects])
    monkeypatch.setattr(drv, "_pull_recruiting_composites",
                        lambda years: {})
    monkeypatch.setenv("CFBD_API_KEY", "test-key")
    return counts


# ─────────────────────────────────────────────────────────────────────
# In-memory fake psycopg connection: just the two derive tables.
# Mirrors the real API the code uses (cursor() context manager,
# execute/fetchone, commit) like the cleanup tests' fake.
# ─────────────────────────────────────────────────────────────────────

class _FakeCursor:
    def __init__(self, state):
        self.state = state
        self._row = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        norm = " ".join(sql.split()).upper()
        self.state["executed"].append((norm, params))
        if norm.startswith("CREATE TABLE"):
            return
        if norm.startswith("INSERT INTO PROSPECT_V4_DERIVE_CACHE"):
            year, version, gp, payload_json = params
            # psycopg hands jsonb back parsed; store the parsed
            # dict so reads behave like the real driver.
            self.state["cache"][(year, version)] = {
                "games_played": gp, "payload": json.loads(payload_json)}
            return
        if norm.startswith("INSERT INTO PROSPECT_V4_REFERENCE_ARTIFACT"):
            script_version, artifact_json = params
            self.state["artifact"] = {
                "script_version": script_version,
                "artifact": json.loads(artifact_json)}
            return
        if norm.startswith("SELECT") and "PROSPECT_V4_DERIVE_CACHE" in norm:
            year, version = params
            entry = self.state["cache"].get((year, version))
            self._row = (entry["payload"],) if entry else None
            return
        if norm.startswith("SELECT") and "PROSPECT_V4_REFERENCE_ARTIFACT" in norm:
            stored = self.state["artifact"]
            self._row = (stored["artifact"],) if stored else None
            return
        raise AssertionError(f"unexpected SQL: {norm}")

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return _FakeCursor(self.state)

    def commit(self):
        self.state["commits"] += 1


def _db_state():
    return {"cache": {}, "artifact": None, "executed": [], "commits": 0}


@pytest.fixture()
def db_env(monkeypatch):
    """DATABASE_URL set + both scripts' _connect wired to one
    shared in-memory state."""
    state = _db_state()
    monkeypatch.setenv("DATABASE_URL", "postgres://fake")
    monkeypatch.setattr(drv, "_connect", lambda: _FakeConn(state))
    monkeypatch.setattr(pvr, "_connect", lambda: _FakeConn(state))
    return state


def _wr_class(n=12):
    """A GP-bearing WR class whose per-game legs vary by player."""
    out = []
    for i in range(n):
        p = _prospect(f"W{i}", gp=12)
        p["seasons"][0]["receiving_yards"] = 600 + 75 * i
        out.append(p)
    return out


# ─────────────────────────────────────────────────────────────────────
# Backend selection
# ─────────────────────────────────────────────────────────────────────

class TestBackendSelection:
    def test_file_backend_without_database_url(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert drv._cache_backend() == "file"

    def test_db_backend_with_database_url(self, db_env):
        assert drv._cache_backend() == "db"

    def test_file_backend_round_trip_still_works(
            self, monkeypatch, tmp_path):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        classes = {2016: [_prospect(f"P{i}") for i in range(4)]}
        counts = _install_fake_machinery(monkeypatch, classes)
        cache_dir = str(tmp_path / "cache")
        out = str(tmp_path / "v4_reference.json")
        drv.run_real([2016], out, cache_dir=cache_dir)
        assert os.path.exists(os.path.join(cache_dir, "2016.json"))
        counts["cfbd_loads"] = 0
        drv.run_real([2016], out, cache_dir=cache_dir)
        assert counts["cfbd_loads"] == 0


# ─────────────────────────────────────────────────────────────────────
# DB cache: round-trip, miss rules, gate, --no-cache, degradation
# ─────────────────────────────────────────────────────────────────────

class TestDbCache:
    def test_round_trip_hit_pools_legs_and_writes_no_file(
            self, monkeypatch, tmp_path, db_env, capsys):
        counts = _install_fake_machinery(monkeypatch, {2016: _wr_class()})
        cache_dir = str(tmp_path / "cache")
        out = str(tmp_path / "v4_reference.json")

        artifact = drv.run_real([2016], out, cache_dir=cache_dir)
        assert counts["cfbd_loads"] == 1
        # The class landed in Postgres; no cache file appeared.
        assert (2016, drv.CACHE_VERSION) in db_env["cache"]
        assert not os.path.exists(cache_dir)
        assert db_env["commits"] >= 1  # per-class commit, immediate

        counts["cfbd_loads"] = 0
        capsys.readouterr()
        artifact2 = drv.run_real([2016], out, cache_dir=cache_dir)
        assert counts["cfbd_loads"] == 0
        assert ("[2016] from derive cache: 12 prospects"
                in capsys.readouterr().out)
        for key in ("anchors", "percentiles", "advanced", "populations"):
            assert artifact2[key] == artifact[key]
        # The cached read-back feeds build_reference identically:
        # the per-game legs pooled from the Postgres rows.
        pools = artifact2["populations"]["season_legs"]["WR"]
        assert pools["rec_yds_pg"] == 12
        assert "rec_yds_pg" in artifact2["anchors"]["WR"]

    def test_no_cache_forces_live_under_db(
            self, monkeypatch, tmp_path, db_env):
        counts = _install_fake_machinery(monkeypatch, {2016: _wr_class()})
        out = str(tmp_path / "v4_reference.json")
        drv.run_real([2016], out, cache_dir=str(tmp_path / "cache"))
        assert (2016, drv.CACHE_VERSION) in db_env["cache"]
        counts["cfbd_loads"] = 0
        drv.run_real([2016], out, cache_dir=str(tmp_path / "cache"),
                     use_cache=False)
        assert counts["cfbd_loads"] == 1

    def test_subfloor_class_never_written_to_db(
            self, monkeypatch, tmp_path, db_env):
        healthy = [_prospect(f"H{i}") for i in range(4)]
        starved = ([_prospect("S0", with_seasons=True)]
                   + [_prospect(f"S{i}", with_seasons=False)
                      for i in range(1, 4)])  # 1/4 = 0.25 matched
        _install_fake_machinery(monkeypatch,
                                {2016: healthy, 2017: starved})
        out = str(tmp_path / "v4_reference.json")
        with pytest.raises(SystemExit) as exc:
            drv.run_real([2016, 2017], out,
                         cache_dir=str(tmp_path / "cache"))
        assert exc.value.code == 1
        assert not os.path.exists(out)
        assert (2016, drv.CACHE_VERSION) in db_env["cache"]
        assert (2017, drv.CACHE_VERSION) not in db_env["cache"]

    def test_version_mismatch_and_gpless_rows_are_misses(self, db_env):
        prospects = [drv._extract_prospect(_prospect(f"P{i}"))
                     for i in range(3)]
        rows = {p["player_id"]: drv._extract_component_row(_score_row(p))
                for p in prospects}
        assert drv._write_class_cache_db(2016, prospects, rows,
                                         games_played=True)
        assert drv._read_class_cache_db(2016) is not None

        # A stale-version row is ignored (the version is part of
        # the lookup key, exactly like the file name + payload
        # check retiring old entries unread).
        db_env["cache"][(2016, 1)] = db_env["cache"].pop(
            (2016, drv.CACHE_VERSION))
        assert drv._read_class_cache_db(2016) is None

        # A GP-less payload at the current version is a miss, same
        # marker rule as the file backend.
        assert drv._write_class_cache_db(2016, prospects, rows,
                                         games_played=False)
        assert drv._read_class_cache_db(2016) is None

    def test_failing_cache_db_degrades_to_live(
            self, monkeypatch, tmp_path, db_env, capsys):
        def dead_connect():
            raise RuntimeError("db down")

        monkeypatch.setattr(drv, "_connect", dead_connect)
        counts = _install_fake_machinery(monkeypatch, {2016: _wr_class()})
        out = str(tmp_path / "v4_reference.json")
        artifact = drv.run_real([2016], out,
                                cache_dir=str(tmp_path / "cache"))
        assert counts["cfbd_loads"] == 1
        assert os.path.exists(out)
        text = capsys.readouterr().out
        assert "read failed" in text and "write failed" in text
        assert artifact["populations"]["players"] == 12


# ─────────────────────────────────────────────────────────────────────
# Artifact store + print script
# ─────────────────────────────────────────────────────────────────────

class TestArtifactStoreAndPrint:
    def test_artifact_stored_exactly_and_printed(
            self, monkeypatch, tmp_path, db_env, capsys):
        _install_fake_machinery(monkeypatch, {2016: _wr_class()})
        artifact = drv.run_real([2016], str(tmp_path / "v4_reference.json"),
                                cache_dir=str(tmp_path / "cache"))
        stored = db_env["artifact"]
        assert stored is not None
        assert stored["script_version"] == drv.SCRIPT_VERSION
        assert stored["artifact"] == artifact

        capsys.readouterr()
        assert pvr.main([]) == 0
        printed = capsys.readouterr().out
        assert json.loads(printed) == artifact

        assert pvr.main(["--base64"]) == 0
        encoded = capsys.readouterr().out.strip()
        decoded = base64.b64decode(encoded).decode("utf-8")
        assert json.loads(decoded) == artifact

    def test_failing_artifact_store_warns_but_run_passes(
            self, monkeypatch, tmp_path, db_env, capsys):
        state = db_env

        class _ArtifactDeadCursor(_FakeCursor):
            def execute(self, sql, params=None):
                if "PROSPECT_V4_REFERENCE_ARTIFACT" in sql.upper():
                    raise RuntimeError("artifact store down")
                return super().execute(sql, params)

        class _ArtifactDeadConn(_FakeConn):
            def cursor(self):
                return _ArtifactDeadCursor(state)

        monkeypatch.setattr(drv, "_connect",
                            lambda: _ArtifactDeadConn(state))
        _install_fake_machinery(monkeypatch, {2016: _wr_class()})
        out = str(tmp_path / "v4_reference.json")
        artifact = drv.run_real([2016], out,
                                cache_dir=str(tmp_path / "cache"))
        assert os.path.exists(out)
        assert artifact["populations"]["players"] == 12
        assert "WARNING" in capsys.readouterr().out
        # The cache side still worked; only the artifact store died.
        assert (2016, drv.CACHE_VERSION) in state["cache"]

    def test_print_fails_closed_without_database_url(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setattr(pvr, "_connect",
                            lambda: (_ for _ in ()).throw(
                                AssertionError("must not connect")))
        assert pvr.main([]) == 1

    def test_print_fails_closed_with_no_artifact_stored(
            self, db_env, capsys):
        assert pvr.main([]) == 1
        assert capsys.readouterr().out == ""  # nothing partial printed
