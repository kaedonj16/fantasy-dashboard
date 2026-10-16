"""Regression tests for the rookie pipeline data-integrity fixes.

Covers the four defects from the 2026-10-16 board-divergence diagnostic:
1. Name joins keyed on raw lowercase strings (CJ Carr's missing seasons,
   Ryan Coleman-Williams split from "Ryan Williams").
2. Class-wide backfill gate (a season missed for one player never
   refetched because teammates had rows).
3. Age coverage (all 62 seed profiles carry age: null; the CFBD roster
   fill read height/weight only, so missing ages printed as a flat 50.0).
4. Stale rankings rows never deleted + the board serving stored
   overall_rank order (a 47.02 from model 2.0 presented as #1 overall).

Fakes mirror the real adapter shapes (dict rows, psycopg-style cursors).
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import os
import sys
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dashboard_services.db as _db_mod
import data_building.rookie_pipeline.ingestion as ing
import data_building.rookie_pipeline.pipeline as pl


# ─────────────────────────────────────────────────────────────────────────
# 1. Name normalization + canonical keys
# ─────────────────────────────────────────────────────────────────────────

class TestNormalizeName:
    @pytest.mark.parametrize("a,b", [
        ("C.J. Carr", "CJ Carr"),                    # periods in initials
        ("K C Concepcion", "K.C. Concepcion"),       # spaced initials
        ("Trey'Dez Green", "Treydez Green"),         # apostrophe
        ("Ryan Coleman-Williams", "Ryan Coleman Williams"),  # hyphen vs space
        ("Harold Fannin Jr.", "Harold Fannin"),      # suffix with period
        ("Harold Fannin Jr", "Harold Fannin Sr."),   # suffix variants
        ("  Nick   Marsh ", "Nick Marsh"),           # whitespace
    ])
    def test_spelling_variants_share_a_key(self, a, b):
        assert ing.normalize_name(a) == ing.normalize_name(b)

    def test_empty_name(self):
        assert ing.normalize_name("") == ""
        assert ing.normalize_name(None) == ""


class TestCanonicalKey:
    def test_coleman_williams_alias_both_directions(self):
        full = ing.canonical_prospect_key("Ryan Coleman-Williams")
        short = ing.canonical_prospect_key("Ryan Williams")
        assert full == short == "ryan coleman williams"

    def test_concepcion_alias_both_directions(self):
        assert (ing.canonical_prospect_key("Kevin Concepcion")
                == ing.canonical_prospect_key("K.C. Concepcion")
                == "kc concepcion")

    def test_unaliased_name_passes_through_normalized(self):
        assert ing.canonical_prospect_key("C.J. Carr") == "cj carr"


# ─────────────────────────────────────────────────────────────────────────
# 1b. Stats attach through the canonical key (upsert + fetch)
# ─────────────────────────────────────────────────────────────────────────

class _RecordingCursor:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchall(self):
        return []


class _RecordingConn:
    def __init__(self):
        self.cur = _RecordingCursor()

    def cursor(self):
        return self.cur


def _season(year, **kw):
    base = {"season": year, "games_played": 12, "receptions": 50,
            "receiving_yards": 800, "receiving_tds": 9}
    base.update(kw)
    return base


class TestUpsertAttach:
    def test_aliased_stats_land_on_seeded_player_id(self):
        conn = _RecordingConn()
        prospect = {"player_id": "ROOKIE_2027_ryan_coleman_williams",
                    "name": "Ryan Coleman-Williams"}
        # Stats dict as fetch produces it now: canonical key.  The same
        # attach must also work for a raw-spelling key (repair tool).
        stats = {ing.canonical_prospect_key("Ryan Williams"): [_season(2026)]}
        saved = pl.upsert_prospect_source_data([prospect], stats, 2027, conn)
        assert saved == 1
        inserts = [p for (sql, p) in conn.cur.calls
                   if "INSERT INTO rookie_prospect_source_data" in sql]
        assert len(inserts) == 1
        assert inserts[0]["player_id"] == "ROOKIE_2027_ryan_coleman_williams"
        assert inserts[0]["receiving_yards"] == 800

    def test_punctuation_variant_attaches(self):
        conn = _RecordingConn()
        prospect = {"player_id": "ROOKIE_2027_cj_carr", "name": "CJ Carr"}
        stats = {"C.J. Carr": [_season(2025, receptions=0,
                                        receiving_yards=0, receiving_tds=0,
                                        pass_yards=2400)]}
        saved = pl.upsert_prospect_source_data([prospect], stats, 2027, conn)
        assert saved == 1


class TestFetchKeying:
    @pytest.fixture()
    def cfbd_env(self, monkeypatch):
        team_rows = [
            {"team": "Alabama", "statName": n, "statValue": v}
            for n, v in [("netPassingYards", 3000), ("rushingYards", 1500),
                         ("passingTDs", 30), ("rushingTDs", 20),
                         ("passAttempts", 400), ("rushingAttempts", 500)]
        ]
        player_rows = [
            {"player": "Ryan Williams", "playerId": 999, "position": "WR",
             "team": "Alabama", "conference": "SEC",
             "category": "receiving", "statType": t, "stat": v}
            for t, v in [("REC", 50), ("YDS", 800), ("TD", 9)]
        ]

        def fake_get(path, params=None, retries=2):
            if path == "/stats/season":
                return team_rows
            if path == "/stats/player/season":
                return player_rows
            return []

        monkeypatch.setattr(ing, "CFBD_KEY", "test-key")
        monkeypatch.setattr(ing, "_cfbd_get", fake_get)
        monkeypatch.setattr(ing, "_filter_years_needing_fetch",
                            lambda years, dy, player_ids=None: sorted(years))
        monkeypatch.setattr(ing, "fetch_cfbd_targets_from_plays",
                            lambda *a, **k: {})

    def test_fetch_keys_stats_by_canonical_name(self, cfbd_env):
        result = ing.fetch_cfbd_college_stats(2027, skip_sagarin=True)
        key = ing.canonical_prospect_key("Ryan Coleman-Williams")
        assert key in result
        season_2026 = [s for s in result[key] if s["season"] == 2026]
        assert len(season_2026) == 1
        assert season_2026[0]["receiving_yards"] == 800
        assert season_2026[0]["receptions"] == 50


# ─────────────────────────────────────────────────────────────────────────
# 1c. Duplicate prospects merge onto the canonical (seeded) record
# ─────────────────────────────────────────────────────────────────────────

class TestDedupeCanonicalMerge:
    def test_discovered_alias_merges_into_seed_row(self):
        seed = {"player_id": "ROOKIE_2027_ryan_coleman_williams",
                "name": "Ryan Coleman-Williams", "position": "WR",
                "school": "Alabama", "source": "seed_2027",
                "projected_pick": 19, "age": None, "seasons": [],
                "athleticism": {}}
        discovered = {"player_id": "ROOKIE_2027_ryan_williams",
                      "name": "Ryan Williams", "position": "WR",
                      "school": "Alabama", "source": "cfbd_discovery",
                      "age": None, "seasons": [_season(2026)],
                      "athleticism": {}}
        out = pl._deduplicate_prospects([seed, discovered])
        assert len(out) == 1
        merged = out[0]
        assert merged["player_id"] == "ROOKIE_2027_ryan_coleman_williams"
        assert merged["name"] == "Ryan Coleman-Williams"
        assert merged["projected_pick"] == 19
        assert [s["season"] for s in merged["seasons"]] == [2026]

    def test_distinct_players_not_merged(self):
        a = {"player_id": "ROOKIE_2027_nick_marsh", "name": "Nick Marsh",
             "source": "seed_2027", "seasons": [], "athleticism": {}}
        b = {"player_id": "ROOKIE_2027_chris_marshall",
             "name": "Chris Marshall", "source": "seed_2027",
             "seasons": [], "athleticism": {}}
        assert len(pl._deduplicate_prospects([a, b])) == 2


# ─────────────────────────────────────────────────────────────────────────
# 2. Per-player backfill gate
# ─────────────────────────────────────────────────────────────────────────

class _GateCursor:
    def __init__(self, state):
        self.state = state
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        s = " ".join(sql.split()).upper()
        if "FROM ROOKIE_PROSPECTS" in s:
            self._rows = [{"player_id": pid} for pid in self.state["roster"]]
        elif "FROM ROOKIE_PROSPECT_SOURCE_DATA" in s:
            roster, seasons = params
            self._rows = [
                {"player_id": pid, "season": yr}
                for (pid, yr) in self.state["stored"]
                if pid in roster and yr in seasons
            ]
        else:  # pragma: no cover
            raise AssertionError(f"Unexpected SQL in gate fake: {s[:80]}")

    def fetchall(self):
        return list(self._rows)


class _GateConn:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return _GateCursor(self.state)

    def commit(self):
        pass


@pytest.fixture()
def gate_env(monkeypatch):
    state = {"roster": ["ROOKIE_2027_a", "ROOKIE_2027_b"], "stored": []}

    @contextmanager
    def fake_get_conn(*args, **kwargs):
        yield _GateConn(state)

    monkeypatch.setattr(_db_mod, "get_conn", fake_get_conn)
    return state


class TestBackfillGate:
    ALL_YEARS = [2026, 2025, 2024, 2023]

    def test_year_skipped_when_every_player_has_it(self, gate_env):
        gate_env["stored"] = [
            (pid, yr) for pid in gate_env["roster"] for yr in (2025, 2024, 2023)
        ]
        assert ing._filter_years_needing_fetch(self.ALL_YEARS, 2027) == [2026]

    def test_year_refetched_when_one_player_lacks_it(self, gate_env):
        # Player B is missing 2025 (the Carr shape): the class-wide gate
        # used to skip 2025 forever because player A had it.
        gate_env["stored"] = [
            (pid, yr) for pid in gate_env["roster"] for yr in (2024, 2023)
        ] + [("ROOKIE_2027_a", 2025)]
        needed = ing._filter_years_needing_fetch(self.ALL_YEARS, 2027)
        assert 2025 in needed and 2024 not in needed and 2026 in needed

    def test_explicit_player_ids_define_the_run(self, gate_env):
        gate_env["roster"] = ["ROOKIE_2027_someone_else"]
        gate_env["stored"] = [("ROOKIE_2027_a", 2025), ("ROOKIE_2027_a", 2024),
                              ("ROOKIE_2027_a", 2023)]
        needed = ing._filter_years_needing_fetch(
            self.ALL_YEARS, 2027,
            player_ids=["ROOKIE_2027_a", "ROOKIE_2027_b"],
        )
        # B (in the run, not in the stored roster) lacks everything.
        assert needed == sorted(self.ALL_YEARS)


# ─────────────────────────────────────────────────────────────────────────
# 3. Age coverage
# ─────────────────────────────────────────────────────────────────────────

class TestAgeEnrichment:
    @pytest.fixture()
    def roster_env(self, monkeypatch):
        rows = [
            {"firstName": "Charlie", "lastName": "Becker", "height": 74,
             "weight": 200, "year": 3, "team": "Indiana", "position": "WR"},
            {"firstName": "Exact", "lastName": "Age",
             "dateOfBirth": "2004-06-15", "height": 72, "weight": 190,
             "year": 4, "team": "Indiana", "position": "WR"},
        ]
        monkeypatch.setattr(ing, "CFBD_KEY", "test-key")
        monkeypatch.setattr(ing, "_cfbd_get", lambda *a, **k: rows)

    def test_empty_age_filled_from_roster_year(self, roster_env):
        p = {"name": "Charlie Becker", "age": None,
             "height_inches": None, "weight_lbs": None}
        out = ing._enrich_bio_from_cfbd_roster([p], 2027)
        assert out[0]["age"] == ing._estimate_age("JR", 2027)
        assert out[0]["_age_estimated"] is True
        assert out[0]["height_inches"] == 74

    def test_dob_row_fills_exact_age(self, roster_env):
        p = {"name": "Exact Age", "age": None,
             "height_inches": 72, "weight_lbs": 190}
        out = ing._enrich_bio_from_cfbd_roster([p], 2027)
        # Born 2004-06-15; draft reference 2027-04-25 -> 22 full years.
        assert out[0]["age"] is not None and 22.0 < out[0]["age"] < 23.0
        assert not out[0].get("_age_estimated")

    def test_present_age_never_overwritten(self, roster_env):
        p = {"name": "Charlie Becker", "age": 21.3,
             "height_inches": 74, "weight_lbs": 200}
        out = ing._enrich_bio_from_cfbd_roster([p], 2027)
        assert out[0]["age"] == 21.3


class TestAgeCoverageLine:
    def test_coverage_line_emitted(self, capsys):
        prospects = [{"name": "A", "age": 21.5}, {"name": "B", "age": None}]
        with_age, total = pl._print_age_coverage(prospects, 2027)
        out = capsys.readouterr().out
        assert (with_age, total) == (1, 2)
        assert "Age coverage: 1 of 2 prospects in the 2027 class" in out


# ─────────────────────────────────────────────────────────────────────────
# 4. Stale-row sweep + board ordering
# ─────────────────────────────────────────────────────────────────────────

class _SweepCursor:
    def __init__(self, state):
        self.state = state
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        s = " ".join(sql.split()).upper()
        assert s.startswith("DELETE FROM ROOKIE_RANKINGS")
        draft_year, keep = params
        before = len(self.state["rows"])
        self.state["rows"] = [
            r for r in self.state["rows"]
            if not (r[1] == draft_year and r[0] not in keep)
        ]
        self.rowcount = before - len(self.state["rows"])

    def fetchall(self):
        return []


class _SweepConn:
    def __init__(self, state):
        self.state = state

    def cursor(self):
        return _SweepCursor(self.state)


def _run_ids(n, start=0):
    return [f"ROOKIE_2027_p{i}" for i in range(start, start + n)]


class TestSweepStaleRankings:
    def test_stale_row_deleted_current_rows_kept(self):
        keep = _run_ids(25)
        state = {"rows": [(pid, 2027) for pid in keep]
                 + [("ROOKIE_2027_makai_lemon", 2027),      # the stale 2.0 row
                    ("ROOKIE_2026_someone", 2026)]}         # other class: untouched
        deleted = pl.sweep_stale_rankings(2027, keep, 25, _SweepConn(state))
        assert deleted == 1
        remaining = {pid for pid, _yr in state["rows"]}
        assert "ROOKIE_2027_makai_lemon" not in remaining
        assert "ROOKIE_2026_someone" in remaining
        assert all(pid in remaining for pid in keep)

    def test_degenerate_run_sweeps_nothing(self):
        state = {"rows": [("ROOKIE_2027_makai_lemon", 2027)]}
        deleted = pl.sweep_stale_rankings(2027, _run_ids(3), 3, _SweepConn(state))
        assert deleted == 0
        assert len(state["rows"]) == 1

    def test_low_share_run_sweeps_nothing(self):
        keep = _run_ids(30)  # >= minimum count, but only 30% of a 100 universe
        state = {"rows": [(pid, 2027) for pid in keep]
                 + [("ROOKIE_2027_makai_lemon", 2027)]}
        deleted = pl.sweep_stale_rankings(2027, keep, 100, _SweepConn(state))
        assert deleted == 0
        assert len(state["rows"]) == 31


class _BoardCursor:
    """Simulates Postgres honoring the query's ORDER BY clause."""

    def __init__(self, rows):
        self.rows = rows
        self._rows = []
        self.sql_seen = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.sql_seen = " ".join(sql.split()).upper()
        rows = list(self.rows)
        if "PROSPECT_SCORE DESC" in self.sql_seen:
            rows.sort(key=lambda r: (r["prospect_score"] is None,
                                     -(r["prospect_score"] or 0),
                                     r["overall_rank"]))
        elif "OVERALL_RANK" in self.sql_seen:
            rows.sort(key=lambda r: r["overall_rank"])
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


class _BoardConn:
    def __init__(self, rows):
        self.cur = _BoardCursor(rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self.cur


class TestBoardOrdering:
    def test_board_follows_score_not_stored_rank(self, monkeypatch):
        rows = [
            {"player_id": "ROOKIE_2027_makai_lemon", "overall_rank": 1,
             "prospect_score": 47.02, "model_version": "2.0"},
            {"player_id": "ROOKIE_2027_jeremiah_smith", "overall_rank": 65,
             "prospect_score": 91.58, "model_version": "3.3"},
        ]
        conn = _BoardConn(rows)

        @contextmanager
        def fake_get_conn(*args, **kwargs):
            yield conn

        monkeypatch.setattr(pl, "_db_available", lambda: True)
        monkeypatch.setattr(_db_mod, "get_conn", fake_get_conn)
        import utils.utils as _uu
        monkeypatch.setattr(_uu, "load_model_value_table", lambda: [])

        out = pl.get_rookie_rankings_from_db(2027)
        assert [r["player_id"] for r in out] == [
            "ROOKIE_2027_jeremiah_smith", "ROOKIE_2027_makai_lemon",
        ]
        assert "PROSPECT_SCORE DESC" in conn.cur.sql_seen
