"""Display-only Start/Sit context layer: absences, venue, weather specifics.

Covers the pure helpers (utils.start_sit_context), the weather-tag data
source (utils.game_conditions), and frontend contracts for the waivers
Start/Sit rows and the compare Start/Sit tab. No Flask import: everything
under test is importable without the app stack.
"""
import re

import pytest

from dashboard_services.pages.waivers_page import build_waivers_body
from utils.game_conditions import weather_tag
from utils.start_sit_context import absence_notes, build_absence_index


def _players():
    return {
        # Teammates (CIN skill players, various statuses).
        "1": {"full_name": "Ja'Marr Chase", "team": "CIN", "position": "WR",
              "injury_status": "OUT", "injury_body_part": "Shoulder",
              "depth_chart_order": 1},
        "2": {"full_name": "Tee Higgins", "team": "CIN", "position": "WR",
              "injury_status": "Questionable", "injury_body_part": "Hamstring",
              "depth_chart_order": 2},
        "3": {"full_name": "Joe Burrow", "team": "CIN", "position": "QB",
              "injury_status": "IR", "injury_body_part": "Wrist",
              "depth_chart_order": 1},
        "4": {"full_name": "Evan McPherson", "team": "CIN", "position": "K",
              "injury_status": "OUT", "injury_body_part": "Groin",
              "depth_chart_order": 1},
        "5": {"full_name": "Healthy Back", "team": "CIN", "position": "RB",
              "injury_status": "", "depth_chart_order": 3},
        # The player himself (must be excluded from his own notes).
        "6": {"full_name": "Chase Brown", "team": "CIN", "position": "RB",
              "injury_status": "OUT", "injury_body_part": "Ankle",
              "depth_chart_order": 1},
        # Opponent defense (PIT) with several serious injuries.
        "10": {"full_name": "T.J. Watt", "team": "PIT", "position": "LB",
               "injury_status": "OUT", "injury_body_part": "Knee",
               "depth_chart_order": 1},
        "11": {"full_name": "Minkah Fitzpatrick", "team": "PIT", "position": "S",
               "injury_status": "SUSP", "depth_chart_order": 1},
        "12": {"full_name": "Cam Heyward", "team": "PIT", "position": "DT",
               "injury_status": "Doubtful", "injury_body_part": "Elbow",
               "depth_chart_order": 1},
        "13": {"full_name": "Joey Porter Jr.", "team": "PIT", "position": "CB",
               "injury_status": "PUP", "depth_chart_order": 2},
        "14": {"full_name": "Patrick Queen", "team": "PIT", "position": "LB",
               "injury_status": "IR", "depth_chart_order": 2},
        # Opponent offense (must never appear in defensive notes).
        "15": {"full_name": "George Pickens", "team": "PIT", "position": "WR",
               "injury_status": "OUT", "depth_chart_order": 1},
    }


def test_serious_statuses_only():
    notes = absence_notes(build_absence_index(_players()), "CIN", "PIT",
                          exclude_pid="6")
    names = [e["name"] for e in notes["teammates"]]
    # OUT and IR surface; Questionable and healthy do not.
    assert "Ja'Marr Chase" in names
    assert "Joe Burrow" in names
    assert "Tee Higgins" not in names
    assert "Healthy Back" not in names


def test_teammate_scope_is_skill_positions_only():
    notes = absence_notes(build_absence_index(_players()), "CIN", "PIT",
                          exclude_pid="6")
    names = [e["name"] for e in notes["teammates"]]
    # Kickers are not skill-position teammates for absence purposes.
    assert "Evan McPherson" not in names


def test_player_excluded_from_own_notes():
    notes = absence_notes(build_absence_index(_players()), "CIN", "PIT",
                          exclude_pid="6")
    assert "Chase Brown" not in [e["name"] for e in notes["teammates"]]


def test_opponent_defense_capped_and_deterministic():
    notes = absence_notes(build_absence_index(_players()), "CIN", "PIT",
                          exclude_pid="6")
    opps = notes["opponents"]
    assert len(opps) == 3, "opponent defensive notes must never get noisy"
    names = [e["name"] for e in opps]
    # Opponent offensive players never leak into defensive notes.
    assert "George Pickens" not in names
    # Deterministic: depth order first, then name.
    assert names == ["Cam Heyward", "Minkah Fitzpatrick", "T.J. Watt"]


def test_opponent_defense_cap_respected():
    notes = absence_notes(build_absence_index(_players()), "CIN", "PIT",
                          exclude_pid="6", max_defense=2)
    assert len(notes["opponents"]) == 2


def test_silent_when_nothing_notable():
    notes = absence_notes(build_absence_index(_players()), "KC", "DEN")
    assert notes == {"teammates": [], "opponents": []}
    notes = absence_notes({}, "CIN", "PIT", exclude_pid="6")
    assert notes == {"teammates": [], "opponents": []}


def test_absence_text_format():
    notes = absence_notes(build_absence_index(_players()), "CIN", "PIT",
                          exclude_pid="6")
    by_name = {e["name"]: e for e in notes["teammates"]}
    assert by_name["Ja'Marr Chase"]["text"] == "Ja'Marr Chase (Out, Shoulder)"
    assert by_name["Joe Burrow"]["text"] == "Joe Burrow (IR, Wrist)"
    opp = {e["name"]: e for e in notes["opponents"]}
    # SUSP renders as a readable label even without a body part.
    assert opp["Minkah Fitzpatrick"]["text"] == "Minkah Fitzpatrick (Suspended)"


def test_weather_tag_carries_specifics():
    # The app's only weather source already resolves wind speed into the tag
    # label; the UI must surface the label, not a generic chip.
    tag = weather_tag(False, 70.0, 22.0, 10.0)
    assert tag["kind"] == "wind"
    assert tag["label"] == "22 mph wind"
    tag = weather_tag(False, 68.0, 8.0, 80.0)
    assert tag["kind"] == "precip"
    assert "rain/snow" in tag["label"]
    assert weather_tag(False, 70.0, 8.0, 10.0) is None
    assert weather_tag(True, 70.0, 30.0, 90.0) is None


def test_bundle_carries_weather_label():
    # Bundle rows must keep the specific weather label (not just the kind) so
    # bundle-path Start/Sit rows can show it without a live weather lookup.
    from pathlib import Path
    src = Path("data_building/start_score_bundle.py").read_text()
    assert '"weather_label"' in src
    assert "weather_label" in src


# ── Frontend contracts ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def waivers_script():
    page = build_waivers_body("sleeper", 2026, "12345", {})
    scripts = re.findall(r"<script>(.*?)</script>", page, re.S)
    matches = [s for s in scripts if "function wvLoad(" in s]
    assert matches, "waivers inline script not found"
    return matches[0]


@pytest.fixture(scope="module")
def compare_js():
    from pathlib import Path
    return Path("static/app.js").read_text(encoding="utf-8")


def test_waivers_matchup_line_keeps_home_away(waivers_script):
    # The matchup line renders the server's "vs DAL" / "@ DAL" label.
    assert "${p.opponent}" in waivers_script


def test_waivers_venue_row(waivers_script):
    assert "function wvSsVenueText(" in waivers_script
    assert ">VENUE<" in waivers_script
    # Home/away decoded from the matchup label, no new data fetch.
    assert "indexOf('vs ')" in waivers_script
    assert "indexOf('@ ')" in waivers_script


def test_waivers_absence_evidence_rows(waivers_script):
    assert ">TEAMMATES OUT<" in waivers_script
    assert ">OPP DEFENSE OUT<" in waivers_script
    assert "p.absences" in waivers_script


def test_waivers_weather_demotion_is_specific(waivers_script):
    # A weather demotion shows the specific condition, not "Bad weather".
    assert "p.demotion === 'weather' && p.weather && p.weather.label" in waivers_script


def test_compare_tab_why_row(compare_js):
    assert "function _ssWhyLine(" in compare_js
    assert "'WHY'" in compare_js
    assert "rWhy" in compare_js


def test_compare_tab_demotion_chips(compare_js):
    assert "function _ssDemoteChip(" in compare_js
    assert "_SS_DEMOTION_LABELS" in compare_js
    assert ".ss-demote" in compare_js
    # Weather demotions prefer the specific label on the tab too.
    assert "dem === 'weather' && ssx.weather && ssx.weather.label" in compare_js


def test_compare_tab_venue_expanded(compare_js):
    assert "x.is_home === true ? 'Home'" in compare_js
    assert "_ssVenueChip(x)" in compare_js


def test_compare_tab_opponent_home_away(compare_js):
    assert "ss(p).opponent_label || ss(p).opponent" in compare_js


def test_compare_tab_absence_rows(compare_js):
    assert "'Teammates out'" in compare_js
    assert "'Opp defense out'" in compare_js
    assert "_ssAbsText(p, 'teammates')" in compare_js
    assert "_ssAbsText(p, 'opponents')" in compare_js


def test_no_em_dashes_in_new_copy(waivers_script, compare_js):
    for snippet in ("wvSsVenueText", "TEAMMATES OUT", "OPP DEFENSE OUT",
                    "_ssWhyLine", "_ssDemoteChip", "Teammates out",
                    "Opp defense out"):
        for src, name in ((waivers_script, "waivers"), (compare_js, "app.js")):
            if snippet not in src:
                continue
            start = max(0, src.find(snippet) - 2000)
            window = src[start:src.find(snippet) + 2000]
            assert "—" not in window, f"em dash near {snippet} in {name}"
