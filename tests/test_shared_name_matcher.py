"""Shared player-name matcher (brNameMatch) tests.

The JS matcher is exercised behaviorally by tests/name_match_harness.mjs,
which extracts the [br-name-match] block verbatim from the shipped bundles
(app.js core + the rankings/draft_room/cheat_sheet ports, pinned identical)
and runs the 44-case battery from the 2026-10-29 name-matching audit plus
the ranking pins. The Python mirror in utils/players.py (used by the
/api/trades and compare-list ?q= endpoints) runs the same battery here
against the same stub players and stub values, so both implementations
have to land the same names in the top 5 and the same ranking pins on top.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from utils.players import name_match_query, name_match_score

ROOT = Path(__file__).resolve().parents[1]

# (name, stub value) — stars carry realistic values, decoys carry small
# ones, so the score -> value ranking pins are exercised honestly.
PLAYERS = [
    ("Ja'Marr Chase", 1050), ("Jamari Thrash", 40), ("Jamar Johnson", 5),
    ("Amon-Ra St. Brown", 980), ("Equanimeous St. Brown", 60), ("Earnest Brown IV", 20),
    ("Jaxon Smith-Njigba", 990),
    ("De'Von Achane", 920), ("DeVonta Smith", 700),
    ("Travis Etienne Jr.", 650), ("Caleb Etienne", 10), ("Trevor Etienne", 30),
    ("Kenneth Walker III", 700), ("Walker Little", 15), ("PJ Walker", 25), ("Tracy Walker III", 30),
    ("Marvin Harrison Jr.", 900), ("Marvin Jones Jr.", 300), ("Marvin Mims Jr.", 200),
    ("Harrison Bryant", 90), ("Harrison Butker", 110),
    ("Bijan Robinson", 1090), ("Brian Robinson Jr.", 420), ("Ty Robinson", 35), ("David Martin-Robinson", 15),
    ("Jahmyr Gibbs", 1010), ("Antonio Gibson", 260),
    ("Justin Jefferson", 1020), ("Jermar Jefferson", 45), ("Van Jefferson", 120), ("Jeffery Simmons", 55),
    ("Christian McCaffrey", 950),
    ("Brian Thomas Jr.", 880),
    ("J.J. McCarthy", 250),
    ("A.J. Brown", 960), ("Sam Brown Jr.", 20),
    ("DJ Moore", 860), ("David Moore", 70), ("A.J. Moore Jr.", 15),
    ("CeeDee Lamb", 1000),
    ("Isiah Pacheco", 380),
    ("Rachaad White", 340),
    ("Hollywood Brown", 410), ("Tank Dell", 300),
    ("Puka Nacua", 970), ("Emeka Egbuka", 620),
    ("Tetairoa McMillan", 560), ("Stetson Bennett IV", 18),
    ("Patrick Mahomes", 780), ("Jaylen Mahoney", 12),
    ("Kyren Williams", 620),
    ("Chris Olave", 610),
    ("D'Andre Swift", 590),
    ("Josh Allen", 940), ("Allen Lazard", 150),
    ("Chase Brown", 540),
]

# The audit battery; two expectations corrected against the real index
# ("DJ Moore" is the listed name, and "Hollywood Brown" is itself listed).
CASES = [
    ("jamarr", "Ja'Marr Chase"), ("ja'marr", "Ja'Marr Chase"), ("jamar chase", "Ja'Marr Chase"),
    ("amonra", "Amon-Ra St. Brown"), ("amon-ra", "Amon-Ra St. Brown"), ("st brown", "Amon-Ra St. Brown"), ("st. brown", "Amon-Ra St. Brown"),
    ("jsn", "Jaxon Smith-Njigba"), ("smith njigba", "Jaxon Smith-Njigba"), ("njigba", "Jaxon Smith-Njigba"),
    ("devon achane", "De'Von Achane"), ("de'von", "De'Von Achane"),
    ("etienne jr", "Travis Etienne Jr."), ("walker iii", "Kenneth Walker III"), ("harrison jr", "Marvin Harrison Jr."),
    ("bijon", "Bijan Robinson"), ("gibs", "Jahmyr Gibbs"), ("jeffeson", "Justin Jefferson"),
    ("cmc", "Christian McCaffrey"), ("btj", "Brian Thomas Jr."), ("mhj", "Marvin Harrison Jr."), ("jj", "Justin Jefferson"),
    ("a.j. brown", "A.J. Brown"), ("aj brown", "A.J. Brown"), ("d.j. moore", "DJ Moore"), ("dj moore", "DJ Moore"),
    ("ceedee", "CeeDee Lamb"), ("cee dee lamb", "CeeDee Lamb"), ("cd lamb", "CeeDee Lamb"),
    ("chase jamarr", "Ja'Marr Chase"), ("brown amon", "Amon-Ra St. Brown"),
    ("isaiah pacheco", "Isiah Pacheco"), ("rashad white", "Rachaad White"),
    ("hollywood brown", "Hollywood Brown"), ("tank dell", "Tank Dell"),
    ("puka", "Puka Nacua"), ("tet", "Tetairoa McMillan"), ("marv", "Marvin Harrison Jr."),
    ("mahome", "Patrick Mahomes"), ("kyran williams", "Kyren Williams"), ("olavee", "Chris Olave"),
    ("dandre swift", "D'Andre Swift"), ("jeffer", "Justin Jefferson"), ("robinson", "Bijan Robinson"),
]

RANK_PINS = [
    ("robinson", "Bijan Robinson"),
    ("st brown", "Amon-Ra St. Brown"),
    ("marv", "Marvin Harrison Jr."),
    ("walker iii", "Kenneth Walker III"),
]


def _ranked(query: str, limit: int = 5) -> list[str]:
    pq = name_match_query(query)
    scored = [
        (name_match_score(name, pq), value, name)
        for name, value in PLAYERS
    ]
    scored = [row for row in scored if row[0] > 0]
    scored.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return [name for _, _, name in scored[:limit]]


def _run_harness() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available for the JS behavioral harness")
    result = subprocess.run(
        [node, str(ROOT / "tests" / "name_match_harness.mjs")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CHECKS PASSED" in result.stdout, result.stdout + result.stderr


def test_js_matcher_battery_harness():
    """Battery + ranking pins + port parity against the shipped JS blocks."""
    _run_harness()


def test_python_mirror_battery():
    for query, expected in CASES:
        top = _ranked(query)
        assert expected in top, f"{query!r}: expected {expected} in top 5, got {top}"


def test_python_mirror_ranking_pins():
    for query, expected in RANK_PINS:
        top = _ranked(query)
        assert top and top[0] == expected, f"{query!r}: expected top {expected}, got {top}"


def test_python_mirror_no_match_sanity():
    assert name_match_score("Bijan Robinson", "zzzqq") == 0
    assert name_match_score("Bijan Robinson", "jr") == 0
    assert name_match_score("Bijan Robinson", "") == 0


# ── Surface contracts: every search surface routes through the matcher ────

def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_matcher_block_in_core_above_public_marker():
    src = _read("static/app.js")
    block_at = src.index("// [br-name-match:begin]")
    marker_at = src.index("// @public-js:core-end")
    assert block_at < marker_at, "matcher must live in the core bundle"
    assert "window.brNameMatch" in src[block_at:marker_at]


def test_surfaces_route_through_shared_matcher():
    app_js = _read("static/app.js")
    assert "brRankPlayersByName(_players, q" in app_js          # nav search
    assert "function fuzzyNameScore(name, query)" in app_js     # trade calc wrapper
    assert app_js.count("brScorePlayerName") >= 4               # wrappers + callers
    rankings = _read("static/rankings.js")
    assert "brScorePlayerName(name, query)" in rankings         # prFuzzyScore delegates
    cheat = _read("static/cheat_sheet.js")
    assert "brScorePlayerName(x, state.search)" in cheat
    draft = _read("static/draft_room.js")
    assert "brScorePlayerName(p, q)" in draft
    am = _read("dashboard_services/pages/advanced_metrics_page.py")
    assert "brScorePlayerName(r, q)" in am
    ti = _read("dashboard_services/pages/trade_intel_page.py")
    assert "brRankPlayersByName(_allPlayers, q" in ti
    dvt = _read("dashboard_services/pages/dynasty_pages.py")
    assert "brScorePlayerName(name, q)" in dvt
    rk = _read("dashboard_services/pages/rookies_page.py")
    assert "brRankPlayersByName(players, q" in rk               # rookies board
    assert "rkFuzzy" not in rk                                  # local copy retired
    app_py = _read("app.py")
    assert app_py.count("name_match_score as _nms") == 2        # both ?q= endpoints
    assert "brScorePlayerName(r.dataset.name,fn)" in app_py     # portfolio holdings


def test_nav_payload_carries_search_fields():
    board = _read("dashboard_services/league_players_board.py")
    assert '"search_name", "value"' in board
