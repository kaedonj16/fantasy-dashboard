"""
Tests for scripts/derive_v4_reference.py v1.4 behavior: the
play-by-play targets stub, the per-class derive cache, the
stat-coverage gate, and the games-played plumbing (live loads
fetch GP stateless; GP-less and pre-v1.4 cache entries read as
misses).

No network, no DB, no CFBD key: the backtest machinery and the
current model's scorer are replaced with fakes via monkeypatch,
and the recruiting pull is stubbed out. The cache contract under
test is the one build_reference() defines: per prospect it reads
player_id / position / age / weight_lbs / seasons / athleticism /
recruiting, and per component row only the four COMPONENT_FIELDS.
"""

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


def _prospect(pid, pos="WR", with_seasons=True, dom=0.30):
    seasons = [{
        "season": 2022, "games_played": 12, "team": "State",
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
            # Not consumed by build_reference; extraction must drop it.
            "prospect_score": base + 20}


def _install_fake_machinery(monkeypatch, classes_map):
    """Replace the backtest loaders + model scorer with fakes.
    classes_map: {year: [prospect dicts]}. Returns a counters dict."""
    import backtest_prospect_model as bt
    from data_building.rookie_pipeline import ingestion, prospect_model as pm

    # run_real's targets stub is a deliberate process-global mutation
    # (harmless in the one-shot Render script). In the shared test
    # process it must NOT leak: registering the original with
    # monkeypatch here makes teardown restore it even though
    # run_real overwrites the attribute mid-test. (Without this,
    # test_prospect_data_sources.py's targets tests all see {}.)
    monkeypatch.setattr(ingestion, "fetch_cfbd_targets_from_plays",
                        ingestion.fetch_cfbd_targets_from_plays)

    counts = {"cfbd_loads": 0, "gp_flags": [], "stateless_flags": []}

    def fake_cfbd(year, draft_class, fetch_games_played=False,
                  gp_stateless=False):
        counts["cfbd_loads"] += 1
        counts["gp_flags"].append(fetch_games_played)
        counts["stateless_flags"].append(gp_stateless)
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


class TestTargetsStub:
    def test_run_real_stubs_targets_crawl(self, monkeypatch, tmp_path):
        from data_building.rookie_pipeline import ingestion

        def sentinel(*a, **k):
            raise AssertionError("targets crawl must never run")

        monkeypatch.setattr(
            ingestion, "fetch_cfbd_targets_from_plays", sentinel)
        classes = {2016: [_prospect("A"), _prospect("B")]}
        _install_fake_machinery(monkeypatch, classes)
        drv.run_real([2016], str(tmp_path / "v4_reference.json"),
                     cache_dir=str(tmp_path / "cache"))
        # run_real replaced the crawl with the {} stub before loading.
        assert ingestion.fetch_cfbd_targets_from_plays is not sentinel
        assert ingestion.fetch_cfbd_targets_from_plays(2016, {}) == {}


class TestCacheRoundTrip:
    def test_cache_reload_feeds_identical_leg_inputs(self, tmp_path):
        prospects = [_prospect("A", dom=0.20), _prospect("B", dom=0.35),
                     _prospect("C", pos="TE", dom=0.28)]
        rows = {p["player_id"]: _score_row(p) for p in prospects}
        live_prospects = [drv._extract_prospect(p) for p in prospects]
        live_rows = {pid: drv._extract_component_row(r)
                     for pid, r in rows.items()}
        # Extraction keeps exactly the consumed fields.
        assert "school" not in live_prospects[0]
        assert "prospect_score" not in live_rows["A"]

        cache_dir = str(tmp_path / "cache")
        drv._write_class_cache(cache_dir, 2016, live_prospects, live_rows)
        loaded = drv._read_class_cache(cache_dir, 2016)
        assert loaded is not None
        c_prospects, c_rows = loaded
        assert c_prospects == live_prospects
        assert c_rows == live_rows

        prov = {"fixture_sample": True}
        art_live = drv.build_reference(
            live_prospects, live_rows, {}, prov)
        art_cache = drv.build_reference(c_prospects, c_rows, {}, prov)
        assert art_cache == art_live

    def test_subfloor_cache_reads_as_miss(self, tmp_path):
        prospects = [drv._extract_prospect(_prospect("A", with_seasons=False)),
                     drv._extract_prospect(_prospect("B", with_seasons=False)),
                     drv._extract_prospect(_prospect("C", with_seasons=True))]
        cache_dir = str(tmp_path / "cache")
        drv._write_class_cache(cache_dir, 2016, prospects, {})
        # 1/3 matched < the 0.50 floor: never served from cache, so
        # the coverage gate cannot be short-circuited by a stale
        # starved entry.
        assert drv._read_class_cache(cache_dir, 2016) is None


class TestCoverageGate:
    def test_starved_class_refuses_and_good_cache_stays(
            self, monkeypatch, tmp_path):
        healthy = [_prospect(f"H{i}") for i in range(4)]
        starved = ([_prospect("S0", with_seasons=True)]
                   + [_prospect(f"S{i}", with_seasons=False)
                      for i in range(1, 4)])  # 1/4 = 0.25 matched
        counts = _install_fake_machinery(
            monkeypatch, {2016: healthy, 2017: starved})
        cache_dir = str(tmp_path / "cache")
        out = str(tmp_path / "v4_reference.json")

        with pytest.raises(SystemExit) as exc:
            drv.run_real([2016, 2017], out, cache_dir=cache_dir)
        assert exc.value.code == 1
        assert not os.path.exists(out)
        # The healthy class cached; the starved one did not, so a
        # rerun re-fetches only the starved class.
        assert os.path.exists(os.path.join(cache_dir, "2016.json"))
        assert not os.path.exists(os.path.join(cache_dir, "2017.json"))

        counts["cfbd_loads"] = 0
        with pytest.raises(SystemExit):
            drv.run_real([2016, 2017], out, cache_dir=cache_dir)
        assert counts["cfbd_loads"] == 1

    def test_healthy_run_writes_then_serves_from_cache(
            self, monkeypatch, tmp_path, capsys):
        classes = {2016: [_prospect(f"P{i}", dom=0.20 + 0.05 * i)
                          for i in range(4)]}
        counts = _install_fake_machinery(monkeypatch, classes)
        cache_dir = str(tmp_path / "cache")
        out = str(tmp_path / "v4_reference.json")

        artifact = drv.run_real([2016], out, cache_dir=cache_dir)
        assert os.path.exists(out)
        assert counts["cfbd_loads"] == 1
        prov = artifact["provenance"]
        assert prov["script"].endswith("v1.4")
        assert prov["class_coverage"] == {"2016": 1.0}
        assert "derive_cache" in prov and "coverage_gate" in prov

        counts["cfbd_loads"] = 0
        capsys.readouterr()
        artifact2 = drv.run_real([2016], out, cache_dir=cache_dir)
        assert counts["cfbd_loads"] == 0
        assert ("[2016] from derive cache: 4 prospects"
                in capsys.readouterr().out)
        for key in ("anchors", "percentiles", "advanced", "populations"):
            assert artifact2[key] == artifact[key]

        # --no-cache forces a live reload.
        counts["cfbd_loads"] = 0
        drv.run_real([2016], out, cache_dir=cache_dir, use_cache=False)
        assert counts["cfbd_loads"] == 1


class TestGamesPlayed:
    def test_live_load_requests_games_played(
            self, monkeypatch, tmp_path):
        classes = {2016: [_prospect(f"P{i}") for i in range(4)]}
        counts = _install_fake_machinery(monkeypatch, classes)
        artifact = drv.run_real(
            [2016], str(tmp_path / "v4_reference.json"),
            cache_dir=str(tmp_path / "cache"))
        # The derive's live loads ask the backtest loader for real
        # games-played counts - the whole point of v1.3 - fetched
        # stateless since v1.4 (the tracked lookup's marks describe
        # totals the derive never persists).
        assert counts["gp_flags"] == [True]
        assert counts["stateless_flags"] == [True]
        assert "games_played" in artifact["provenance"]

    def test_gpless_cache_entry_reads_as_miss(self, tmp_path):
        prospects = [drv._extract_prospect(_prospect(f"P{i}"))
                     for i in range(3)]
        rows = {p["player_id"]: drv._extract_component_row(_score_row(p))
                for p in prospects}
        cache_dir = str(tmp_path / "cache")

        # An entry explicitly recorded as built without GP.
        drv._write_class_cache(cache_dir, 2016, prospects, rows,
                               games_played=False)
        assert drv._read_class_cache(cache_dir, 2016) is None

        # A legacy v1.2 entry: no games_played marker at all. This
        # is the shape actually sitting under derive_cache/ on
        # Render; it must not be served either.
        drv._write_class_cache(cache_dir, 2016, prospects, rows,
                               games_played=True)
        path = os.path.join(cache_dir, "2016.json")
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        del payload["games_played"]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        assert drv._read_class_cache(cache_dir, 2016) is None

        # And a GP entry still serves.
        drv._write_class_cache(cache_dir, 2016, prospects, rows,
                               games_played=True)
        assert drv._read_class_cache(cache_dir, 2016) is not None

    def test_pre_v14_cache_version_reads_as_miss(self, tmp_path):
        # CACHE_VERSION bumped to 2 in v1.4: an entry written by any
        # earlier script - including a v1.3 run whose GP came from
        # the poisoned tracker - is retired unread, however healthy
        # its payload looks.
        prospects = [drv._extract_prospect(_prospect(f"P{i}"))
                     for i in range(3)]
        rows = {p["player_id"]: drv._extract_component_row(_score_row(p))
                for p in prospects}
        cache_dir = str(tmp_path / "cache")
        drv._write_class_cache(cache_dir, 2016, prospects, rows,
                               games_played=True)
        path = os.path.join(cache_dir, "2016.json")
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        assert payload["cache_version"] == drv.CACHE_VERSION == 2
        payload["cache_version"] = 1
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        assert drv._read_class_cache(cache_dir, 2016) is None


def _wr_prospect(pid, rec_yds, gp):
    """One WR prospect shaped exactly as the live path extracts,
    with a single season whose games_played is `gp` (None = the
    GP-less load that produced the empty v1.2 artifact)."""
    p = _prospect(pid)
    p["seasons"] = [{
        "season": 2022, "games_played": gp, "team": "State",
        "conference": "SEC",
        "dominator_rating": 0.30, "market_share_yards": 0.25,
        "market_share_tds": 0.28, "receptions": 60, "targets": 90,
        "receiving_yards": rec_yds, "receiving_tds": 8,
        "yds_per_reception": rec_yds / 60.0,
        "rush_attempts": 5, "rush_yards": 40, "rush_tds": 0,
        "pass_attempts": 0, "pass_yards": 0, "pass_tds": 0,
        "team_pass_yards": 3000, "team_total_yards": 5000,
    }]
    return drv._extract_prospect(p)


class TestSeasonLegPools:
    """Regression for the empty v1.2 artifact: build_reference over
    a population whose seasons carry GP must pool the per-game legs
    and emit knot anchors for them; the identical population with
    GP missing pools nothing (missing stays missing - that half is
    by design, and is why the loaders now fetch GP)."""

    def _population(self, gp):
        prospects = [_wr_prospect(f"W{i}", 600 + 75 * i, gp)
                     for i in range(12)]
        rows = {p["player_id"]: drv._extract_component_row(
                    _score_row({"player_id": p["player_id"]}))
                for p in prospects}
        return prospects, rows

    def test_seasons_with_gp_pool_per_game_legs(self):
        prospects, rows = self._population(gp=12)
        artifact = drv.build_reference(prospects, rows, {},
                                       {"fixture_sample": True})
        pools = artifact["populations"]["season_legs"]["WR"]
        assert pools["rec_yds_pg"] == 12
        assert pools["rec_tds_pg"] == 12
        marks = artifact["anchors"]["WR"]["rec_yds_pg"]
        assert len(marks) == len(drv.KNOT_PERCENTILES)
        # 600/12 .. 1425/12 across the twelve fixtures.
        assert marks[0] == pytest.approx(50.0)
        assert marks[-1] == pytest.approx(118.75)

    def test_seasons_without_gp_pool_nothing(self):
        prospects, rows = self._population(gp=None)
        artifact = drv.build_reference(prospects, rows, {},
                                       {"fixture_sample": True})
        pools = artifact["populations"]["season_legs"]["WR"]
        assert pools["rec_yds_pg"] == 0
        assert "rec_yds_pg" not in artifact["anchors"]["WR"]
