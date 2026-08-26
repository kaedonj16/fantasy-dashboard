"""Unit tests for the tokenless global ADP fetchers (providers.global_adp).

All network is mocked via ``_get_json`` and the crosswalks are pre-seeded, so the
suite never touches a live API (Priority 5: CI must never depend on live APIs)."""

import pytest

from dashboard_services.providers import global_adp as G


@pytest.fixture(autouse=True)
def _isolate_crosswalks():
    G.clear_crosswalk_cache()
    yield
    G.clear_crosswalk_cache()


# ── Yahoo public global ADP ───────────────────────────────────────────────────

def _yahoo_page(entries, count):
    return {"fantasy_content": {"game": [
        {"game_key": "nfl"},
        {"players": {"count": count, **{str(i): {"player": p} for i, p in enumerate(entries)}}},
    ]}}


def test_yahoo_pagination_walks_until_short_page(monkeypatch):
    # Two full pages of 2, then a short page -> pagination stops.
    G._XWALK_CACHE["yahoo"] = {"1": "a", "2": "b", "3": "c"}
    pages = {
        0: _yahoo_page([
            {"player_id": "1", "name": {"full": "A"}, "draft_analysis": {"average_pick": "1.5"}},
            {"player_id": "2", "name": {"full": "B"}, "draft_analysis": {"average_pick": "2.5"}},
        ], 2),
        2: _yahoo_page([
            {"player_id": "3", "name": {"full": "C"}, "draft_analysis": {"average_pick": "3.5"}},
        ], 1),
    }
    calls = []

    def fake_get(url, **kw):
        start = int(url.split("start=")[1].split(";")[0])
        calls.append(start)
        return pages[start]

    monkeypatch.setattr(G, "_get_json", fake_get)
    out = G.fetch_yahoo_global_adp(2026, max_players=100, page=2)
    assert out["adp"] == {"a": 1.5, "b": 2.5, "c": 3.5}
    assert out["mapped_count"] == 3
    assert calls == [0, 2]                    # stopped after the short page
    assert out["meta"]["scoring"] == "mixed"  # never labelled a specific format


def test_yahoo_needs_no_token_and_skips_zero_and_unmapped(monkeypatch):
    G._XWALK_CACHE["yahoo"] = {"1": "a"}
    page = _yahoo_page([
        {"player_id": "1", "name": {"full": "A"}, "draft_analysis": {"average_pick": "4.0"}},
        {"player_id": "1", "name": {"full": "Z"}, "draft_analysis": {"average_pick": "0"}},   # 0 -> skip
        {"player_id": "99", "name": {"full": "U"}, "draft_analysis": {"average_pick": "5"}},  # unmapped
    ], 3)
    monkeypatch.setattr(G, "_get_json", lambda url, **kw: page)
    out = G.fetch_yahoo_global_adp(2026, max_players=3, page=25)
    assert out["adp"] == {"a": 4.0}
    assert "U" in out["unmapped"]


def _yahoo_page_list_shape(entries):
    """Current json_f shape: game is a single dict and players is a plain list of
    {"player": {...}} entries (no count / string indices)."""
    return {"fantasy_content": {"game": {
        "game_key": "470",
        "players": [{"player": p} for p in entries],
    }}}


def test_yahoo_parses_current_list_players_shape(monkeypatch):
    # Regression: Yahoo now returns players as a list, not a count-indexed dict.
    G._XWALK_CACHE["yahoo"] = {"40059": "gibbs"}
    page = _yahoo_page_list_shape([
        {"player_id": "40059", "name": {"full": "Jahmyr Gibbs"},
         "draft_analysis": {"average_pick": "1.44"}},
    ])
    monkeypatch.setattr(G, "_get_json", lambda url, **kw: page)
    out = G.fetch_yahoo_global_adp(2026, max_players=1, page=25)
    assert out["adp"] == {"gibbs": 1.44}


def test_yahoo_name_fallback_when_yahoo_id_not_in_crosswalk(monkeypatch):
    """Regression: Sleeper's yahoo_id lags for recent players, so the whole top of
    the board (Gibbs et al.) misses the id crosswalk. With no yahoo id in the
    players_index to merge, fall back to a name/position match."""
    G.clear_crosswalk_cache()
    # Crosswalk has an OLDER player but not the recent star.
    G._XWALK_CACHE["yahoo"] = {"30121": "cmc"}
    _stub_players_index(monkeypatch, {
        "gibbs": {"name": "Jahmyr Gibbs", "pos": "RB"},
        "cmc": {"name": "Christian McCaffrey", "pos": "RB"},
    })
    page = _yahoo_page_list_shape([
        {"player_id": "40059", "name": {"full": "Jahmyr Gibbs"},
         "display_position": "RB", "draft_analysis": {"average_pick": "1.4"}},
        {"player_id": "30121", "name": {"full": "Christian McCaffrey"},
         "display_position": "RB", "draft_analysis": {"average_pick": "5.6"}},
    ])
    monkeypatch.setattr(G, "_get_json", lambda url, **kw: page)
    out = G.fetch_yahoo_global_adp(2026, max_players=2, page=25)
    # Gibbs mapped by name (id crosswalk miss); CMC by id crosswalk.
    assert out["adp"] == {"gibbs": 1.4, "cmc": 5.6}


def test_yahoo_empty_on_network_failure(monkeypatch):
    G._XWALK_CACHE["yahoo"] = {"1": "a"}

    def boom(url, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(G, "_get_json", boom)
    out = G.fetch_yahoo_global_adp(2026)
    assert out["adp"] == {} and out["source"] == "yahoo"


# ── ESPN public global ADP + separate PPR rank ────────────────────────────────

def test_espn_separates_adp_from_ppr_rank(monkeypatch):
    G._XWALK_CACHE["espn"] = {"111": "cmc", "222": "jj"}
    payload = {"players": [
        {"player": {"id": 111, "fullName": "CMC",
                    "ownership": {"averageDraftPosition": 1.2},
                    "draftRanksByRankType": {"PPR": {"rank": 1}}}},
        {"player": {"id": 222, "fullName": "JJ",
                    "ownership": {"averageDraftPosition": 3.4},
                    "draftRanksByRankType": {"PPR": {"rank": 5}}}},
    ]}
    monkeypatch.setattr(G, "_get_json", lambda url, **kw: payload)
    out = G.fetch_espn_global_adp(2026)
    assert out["adp"] == {"cmc": 1.2, "jj": 3.4}
    assert out["ppr_rank"] == {"cmc": 1.0, "jj": 5.0}
    # ADP and PPR rank live in separate maps and must never be merged.
    assert out["adp"] is not out["ppr_rank"]


def test_espn_filter_header_requests_draft_rank_sort(monkeypatch):
    G._XWALK_CACHE["espn"] = {"111": "cmc"}
    captured = {}

    def fake_get(url, **kw):
        captured["headers"] = kw.get("headers") or {}
        return {"players": [{"player": {"id": 111, "ownership": {"averageDraftPosition": 2.0},
                                        "draftRanksByRankType": {"PPR": {"rank": 2}}}}]}

    monkeypatch.setattr(G, "_get_json", fake_get)
    G.fetch_espn_global_adp(2026, limit=400)
    import json
    flt = json.loads(captured["headers"]["X-Fantasy-Filter"])
    assert flt["players"]["limit"] == 400
    assert "sortDraftRanks" in flt["players"]
    assert flt["players"]["filterSlotIds"]["value"]  # offensive positions requested


def test_espn_missing_adp_still_captures_ppr_rank(monkeypatch):
    G._XWALK_CACHE["espn"] = {"111": "cmc"}
    payload = {"players": [
        {"player": {"id": 111, "fullName": "CMC",
                    "ownership": {},  # no averageDraftPosition
                    "draftRanksByRankType": {"PPR": {"rank": 7}}}},
    ]}
    monkeypatch.setattr(G, "_get_json", lambda url, **kw: payload)
    out = G.fetch_espn_global_adp(2026)
    assert out["adp"] == {} and out["ppr_rank"] == {"cmc": 7.0}


# ── MFL free ADP export ───────────────────────────────────────────────────────

def test_mfl_param_construction_only_verified_filters(monkeypatch):
    G._XWALK_CACHE["mfl:2026"] = {"1": "a"}
    captured = {}

    def fake_get(url, **kw):
        captured["params"] = kw.get("params")
        return {"adp": {"player": [{"id": "1", "averagePick": "2.5",
                                    "minPick": "1", "maxPick": "5", "draftSelPct": "88"}]}}

    monkeypatch.setattr(G, "_get_json", fake_get)
    out = G.fetch_mfl_adp(2026, is_ppr=1, fcount=12, is_mock=0, period="RECENT")
    p = captured["params"]
    assert p["TYPE"] == "adp" and p["JSON"] == 1
    assert p["IS_PPR"] == 1 and p["FCOUNT"] == 12 and p["IS_MOCK"] == 0 and p["PERIOD"] == "RECENT"
    # No unverified dynasty/rookie/SF/TEP filters are ever sent.
    assert not ({"IS_KEEPER", "IS_ROOKIE", "DYNASTY", "SF", "TEP"} & set(p))
    assert out["adp"] == {"a": 2.5}
    assert out["extra"]["a"] == {"min_pick": 1.0, "max_pick": 5.0, "draft_pct": 88.0}
    assert out["meta"]["ppr"] == 1.0 and out["meta"]["draft_type"] == "redraft"


def test_mfl_records_unknown_when_ppr_omitted(monkeypatch):
    G._XWALK_CACHE["mfl:2026"] = {"1": "a"}
    monkeypatch.setattr(G, "_get_json",
                        lambda url, **kw: {"adp": {"player": [{"id": "1", "averagePick": "9"}]}})
    out = G.fetch_mfl_adp(2026, is_ppr=None)
    assert out["meta"]["ppr"] == "unknown"


def test_mfl_unmapped_tracked_and_bad_pick_skipped(monkeypatch):
    G._XWALK_CACHE["mfl:2026"] = {"1": "a"}
    monkeypatch.setattr(G, "_get_json", lambda url, **kw: {"adp": {"player": [
        {"id": "1", "averagePick": "3"},
        {"id": "1", "averagePick": "bad"},   # unparseable -> skipped
        {"id": "77", "averagePick": "8"},    # unmapped id
    ]}})
    out = G.fetch_mfl_adp(2026)
    assert out["adp"] == {"a": 3.0}
    assert "77" in out["unmapped"]


def test_mfl_skips_low_draft_selection_pct(monkeypatch):
    """MFL averagePick is selected-only. A 10% dart-throw around pick 58 must
    not become ADP 57.8 (the Jam Miller / Tanner Koziol consensus leak)."""
    G._XWALK_CACHE["mfl:2026"] = {"17486": "jam", "17644": "tanner", "1": "star"}
    monkeypatch.setattr(G, "_get_json", lambda url, **kw: {"adp": {"player": [
        {"id": "17486", "averagePick": "57.76", "draftSelPct": "10"},
        {"id": "17644", "averagePick": "64.19", "draftSelPct": "7"},
        {"id": "1", "averagePick": "2.4", "draftSelPct": "99"},
    ]}})
    out = G.fetch_mfl_adp(2026)
    assert out["adp"] == {"star": 2.4}
    assert "jam" not in out["adp"] and "tanner" not in out["adp"]


def test_mfl_keeps_unknown_draft_pct(monkeypatch):
    # Legacy / test payloads with no draftSelPct stay usable.
    G._XWALK_CACHE["mfl:2026"] = {"1": "a"}
    monkeypatch.setattr(G, "_get_json", lambda url, **kw: {"adp": {"player": [
        {"id": "1", "averagePick": "12.0"},
    ]}})
    out = G.fetch_mfl_adp(2026)
    assert out["adp"] == {"a": 12.0}


def test_filter_mfl_snapshot_adp_drops_sparse_rows():
    snap = {
        "adp": {"jam": 57.76, "star": 2.4, "legacy": 40.0},
        "extra": {
            "jam": {"draft_pct": 10.0},
            "star": {"draft_pct": 88.0},
            # legacy has no extra row
        },
    }
    assert G.filter_mfl_snapshot_adp(snap) == {"star": 2.4, "legacy": 40.0}
    assert G.filter_mfl_snapshot_adp({}) == {}
    assert G.mfl_adp_is_usable(10) is False
    assert G.mfl_adp_is_usable(25) is True
    assert G.mfl_adp_is_usable(None) is True


# ── Crosswalk building (provider id -> canonical) ─────────────────────────────

def test_yahoo_crosswalk_from_sleeper_feed(monkeypatch):
    G.clear_crosswalk_cache()
    monkeypatch.setattr(G, "_sleeper_feed", lambda: {
        "sleeperA": {"yahoo_id": "500", "espn_id": "900"},
        "sleeperB": {"yahoo_id": "501"},
    })
    assert G.yahoo_id_to_canonical() == {"500": "sleeperA", "501": "sleeperB"}


def _stub_players_index(monkeypatch, index):
    import sys
    import types
    fake_utils = types.ModuleType("utils.utils")
    fake_utils.load_players_index = lambda: index
    fake_utils.normalize_name = lambda n: (n or "").strip().lower()
    monkeypatch.setitem(sys.modules, "utils.utils", fake_utils)


def test_espn_crosswalk_prefers_sleeper_feed(monkeypatch):
    G.clear_crosswalk_cache()
    monkeypatch.setattr(G, "_sleeper_feed", lambda: {"sleeperA": {"espn_id": "900"}})
    # Index also has an espnID for 900 but Sleeper is authoritative and wins.
    _stub_players_index(monkeypatch, {"other": {"espnID": "900"}})
    assert G.espn_id_to_canonical() == {"900": "sleeperA"}


def test_espn_crosswalk_merges_index_for_ids_sleeper_lacks(monkeypatch):
    """Regression: Sleeper's espn_id lags for recent players, so the players_index
    must fill ids Sleeper's feed doesn't carry (the observed 26%-mapped top-of-board
    misses like Gibbs/Chase)."""
    G.clear_crosswalk_cache()
    monkeypatch.setattr(G, "_sleeper_feed", lambda: {"cmc": {"espn_id": "3117251"}})
    # Index carries the newer stars' espnID that Sleeper's feed is missing.
    _stub_players_index(monkeypatch, {
        "gibbs": {"espnID": "4429795"},
        "chase": {"espnID": "4362628"},
        "cmc": {"espnID": "3117251"},  # already covered by Sleeper; not overwritten
    })
    out = G.espn_id_to_canonical()
    assert out["3117251"] == "cmc"          # Sleeper's mapping preserved
    assert out["4429795"] == "gibbs"        # filled from the index
    assert out["4362628"] == "chase"


def test_mfl_crosswalk_matches_by_name_pos(monkeypatch):
    G.clear_crosswalk_cache()
    import types
    # Fake load_players_index / normalize_name via the utils.utils module.
    fake_utils = types.ModuleType("utils.utils")
    fake_utils.load_players_index = lambda: {
        "sleeperA": {"full_name": "Josh Allen", "position": "QB"},
        "sleeperB": {"full_name": "Bijan Robinson", "position": "RB"},
    }
    fake_utils.normalize_name = lambda n: (n or "").strip().lower()
    import sys
    monkeypatch.setitem(sys.modules, "utils.utils", fake_utils)
    monkeypatch.setattr(G, "_mfl_player_rows", lambda season: [
        {"id": "0001", "name": "josh allen", "position": "QB"},
        {"id": "0002", "name": "bijan robinson", "position": "RB"},
        {"id": "0003", "name": "nobody here", "position": "WR"},
    ])
    out = G.mfl_id_to_canonical(2026)
    assert out == {"0001": "sleeperA", "0002": "sleeperB"}


def test_flip_comma_name():
    # MFL exports "Last, First" (optionally with a suffix); reorder to "First Last".
    assert G._flip_comma_name("Gibbs, Jahmyr") == "Jahmyr Gibbs"
    assert G._flip_comma_name("Cook III, James") == "James Cook III"
    assert G._flip_comma_name("St. Brown, Amon-Ra") == "Amon-Ra St. Brown"
    # No comma -> passthrough; empty/None -> "".
    assert G._flip_comma_name("Jaxon Smith-Njigba") == "Jaxon Smith-Njigba"
    assert G._flip_comma_name("") == ""
    assert G._flip_comma_name(None) == ""


def test_mfl_crosswalk_handles_last_comma_first(monkeypatch):
    """Regression: MFL's real "Last, First" names must map. normalize_name does
    not reorder the comma form, so without _flip_comma_name every star (whose
    index name is "First Last") fails to match — the observed 0% MFL mapping."""
    G.clear_crosswalk_cache()
    import sys
    import types
    fake_utils = types.ModuleType("utils.utils")
    fake_utils.load_players_index = lambda: {
        "sleeperA": {"full_name": "Jahmyr Gibbs", "position": "RB"},
        "sleeperB": {"full_name": "Bijan Robinson", "position": "RB"},
    }
    # A minimal stand-in that lowercases/strips but does NOT reorder commas, like
    # the real normalize_name — so the flip is what makes the names line up.
    fake_utils.normalize_name = lambda n: (n or "").strip().lower()
    monkeypatch.setitem(sys.modules, "utils.utils", fake_utils)
    monkeypatch.setattr(G, "_mfl_player_rows", lambda season: [
        {"id": "0001", "name": "Gibbs, Jahmyr", "position": "RB"},
        {"id": "0002", "name": "Robinson, Bijan", "position": "RB"},
    ])
    out = G.mfl_id_to_canonical(2026)
    assert out == {"0001": "sleeperA", "0002": "sleeperB"}
