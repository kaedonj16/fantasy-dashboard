"""Display-only Start/Sit context layer: absences, venue, weather specifics.

Covers the pure helpers (utils.start_sit_context), the weather-tag data
source (utils.game_conditions), and frontend contracts for the waivers
Start/Sit rows and the compare Start/Sit tab. No Flask import: everything
under test is importable without the app stack.
"""
import json
import re
import shutil
import subprocess

import pytest

from dashboard_services.pages.waivers_page import build_waivers_body
from utils.game_conditions import weather_tag
from utils.start_sit_context import (
    absence_notes,
    build_absence_index,
    productive_pids_from_weekly_points,
    starting_lineman_pids,
)


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


# ── Importance gate: only notable absences ─────────────────────────────────

def _single(pid="1", **kw):
    base = {"full_name": "Test Player", "team": "SF", "position": "WR",
            "injury_status": "IR", "injury_body_part": "Knee",
            "depth_chart_order": 7}
    base.update(kw)
    return {pid: base}


def _teammate_names(players, productive=None):
    notes = absence_notes(
        build_absence_index(players, productive_pids=productive), "SF", "DEN")
    return [e["name"] for e in notes["teammates"]]


def _opponent_names(players, productive=None):
    notes = absence_notes(
        build_absence_index(players, productive_pids=productive), "DEN", "SF")
    return [e["name"] for e in notes["opponents"]]


def test_deep_depth_skill_excluded_unless_productive():
    # A WR buried at depth order 7 (De'Zhaun Stribling territory) is noise.
    assert _teammate_names(_single()) == []
    # ...unless his pooled production says he actually matters (the Ricky
    # Pearsall case: real starter whose order slid because of the injury).
    assert _teammate_names(_single(), productive={"1"}) == ["Test Player"]


def test_wr_depth_two_included_without_production():
    assert _teammate_names(_single(depth_chart_order=2)) == ["Test Player"]


def test_qb_depth_gate_is_starter_only():
    assert _teammate_names(_single(position="QB", depth_chart_order=2)) == []
    assert _teammate_names(_single(position="QB", depth_chart_order=1)) == ["Test Player"]
    # A productive backup QB still counts (production flag applies to skill).
    assert _teammate_names(_single(position="QB", depth_chart_order=2),
                           productive={"1"}) == ["Test Player"]


def test_defense_depth_gate_ignores_production():
    # Sleeper defensive depth order is per-slot; None means deep/IR, and a
    # depth-3 defender is not notable either.
    assert _opponent_names(_single(position="LB", depth_chart_order=None)) == []
    assert _opponent_names(_single(position="LB", depth_chart_order=3)) == []
    assert _opponent_names(_single(position="LB", depth_chart_order=2)) == ["Test Player"]
    # The productive flag never rescues a defender.
    assert _opponent_names(_single(position="LB", depth_chart_order=None),
                           productive={"1"}) == []
    assert _opponent_names(_single(position="LB", depth_chart_order=3),
                           productive={"1"}) == []


def test_productive_pids_thresholds():
    assert productive_pids_from_weekly_points({"1": [10.0, 10.0, 10.0]}) == set()
    assert productive_pids_from_weekly_points({"1": [10.0, 10.0, 10.0, 10.0]}) == {"1"}
    assert productive_pids_from_weekly_points({"1": [5.0, 5.0, 5.0, 5.0]}) == set()


def test_productive_pids_pool_across_seasons():
    pooled = productive_pids_from_weekly_points({"1": [8.0, 8.0]},
                                                {"1": [8.0, 8.0]})
    assert pooled == {"1"}


def test_productive_pids_never_raises_on_junk():
    assert productive_pids_from_weekly_points(None, "junk", {"1": "nope"},
                                              {"2": [None, "x"]}, []) == set()


def test_teammates_capped_at_four():
    players = {}
    for i, order in enumerate((1, 1, 1, 2, 2)):
        players[str(i)] = {"full_name": f"Wideout {i}", "team": "SF",
                           "position": "WR", "injury_status": "IR",
                           "depth_chart_order": order}
    notes = absence_notes(build_absence_index(players), "SF", "DEN")
    assert len(notes["teammates"]) == 4


# ── Starting offensive linemen (snap-share identified) ────────────────────

def _lineman(pid="20", **kw):
    # Sleeper linemen carry no depth_chart_order at all.
    base = {"full_name": "Tyler Smith", "team": "SF", "position": "G",
            "injury_status": "IR", "injury_body_part": "Thumb"}
    base.update(kw)
    return {pid: base}


def _lineman_notes(players, starting=None):
    return absence_notes(
        build_absence_index(players, starting_linemen=starting), "SF", "DEN")


def test_lineman_counts_only_when_starting():
    # A snap-share-identified starting guard on IR is a notable absence.
    notes = _lineman_notes(_lineman(), starting={"20"})
    assert [e["name"] for e in notes["teammates"]] == ["Tyler Smith"]
    # Without the starting set, no lineman ever shows (no depth signal).
    assert _lineman_notes(_lineman())["teammates"] == []
    assert _lineman_notes(_lineman(), starting={"99"})["teammates"] == []


def test_lineman_still_needs_serious_status():
    notes = _lineman_notes(_lineman(injury_status="Questionable"),
                           starting={"20"})
    assert notes["teammates"] == []


def test_lineman_entry_text_carries_ol_tag():
    notes = _lineman_notes(_lineman(), starting={"20"})
    (entry,) = notes["teammates"]
    assert entry["text"].startswith("Tyler Smith")
    assert "(OL · " in entry["text"]
    assert entry["text"] == "Tyler Smith (OL · IR, Thumb)"
    assert entry["pos"] == "G"


def test_linemen_have_own_cap_and_follow_skill():
    players = {}
    for i, order in enumerate((1, 1, 1, 2, 2)):
        players[str(i)] = {"full_name": f"Wideout {i}", "team": "SF",
                           "position": "WR", "injury_status": "IR",
                           "depth_chart_order": order}
    for j, name in enumerate(("Zack Martin", "Aaron Banks", "Trent Brown")):
        players[str(20 + j)] = {"full_name": name, "team": "SF",
                                "position": "OL", "injury_status": "OUT"}
    notes = absence_notes(
        build_absence_index(players, starting_linemen={"20", "21", "22"}),
        "SF", "DEN")
    names = [e["name"] for e in notes["teammates"]]
    # 4 capped skill teammates, then linemen under their own cap of 2,
    # sorted by name; linemen appear even with the skill cap full.
    assert len(names) == 6
    assert names[:4] == [f"Wideout {i}" for i in range(4)]
    assert names[4:] == ["Aaron Banks", "Trent Brown"]


def test_starting_lineman_pids_share_math():
    pos = {"1": "G"}
    # 120/130 snaps over 2 games: a starter.
    assert starting_lineman_pids([{"1": (120, 130, 2)}], pos) == {"1"}
    # 40/130: a rotational backup, not a starter.
    assert starting_lineman_pids([{"1": (40, 130, 2)}], pos) == set()
    # 100% for a single game is not enough sample.
    assert starting_lineman_pids([{"1": (65, 65, 1)}], pos) == set()
    # Two partial seasons pool: 1 game each at starter share qualifies.
    pooled = starting_lineman_pids([{"1": (60, 65, 1)}, {"1": (60, 65, 1)}], pos)
    assert pooled == {"1"}


def test_starting_lineman_pids_position_gate_and_junk():
    # Starter-level snaps do not make a non-lineman a lineman.
    assert starting_lineman_pids([{"1": (130, 130, 3)}], {"1": "WR"}) == set()
    # Zero team snaps: no share to compute, no crash, no qualifier.
    assert starting_lineman_pids([{"1": (0, 0, 3)}], {"1": "T"}) == set()
    # Junk never raises.
    assert starting_lineman_pids(None, None) == set()
    assert starting_lineman_pids(["junk", {"1": "nope"},
                                  {"2": (None, None, None)}],
                                 {"1": "G", "2": "G"}) == set()


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


def test_waivers_compare_oline_contracts(waivers_script):
    # The waivers compare carries the O-line factor end to end: a verdict
    # reason, a numeric pair for the factor bars, and a full-table row.
    assert "'oline'" in waivers_script
    assert "a stronger offensive line" in waivers_script
    assert "'O-LINE'" in waivers_script
    assert "da.olineNum" in waivers_script
    assert "row('O-Line'" in waivers_script


def test_waivers_absence_lines_stack(waivers_script):
    # Absences render as stacked .wv-abs lines, not a '; ' joined wall.
    assert "wvAbsLine" in waivers_script
    assert 'class="wv-abs"' in waivers_script
    assert "wv-abs-st" in waivers_script


def test_compare_tab_why_line_top_four(compare_js):
    start = compare_js.index("function _ssWhyLine(")
    window = compare_js[start:start + 1500]
    assert "slice(0, 4)" in window


def test_compare_tab_absence_lines_stack(compare_js):
    start = compare_js.index("function _ssAbsText(")
    window = compare_js[start:start + 800]
    # Entries are escaped individually and stacked with <br>.
    assert "_ssEsc(txt)" in window
    assert "join('<br>')" in window


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_waivers_oline_reason_fires_in_node(waivers_script):
    # An oline-only score-factor edge must produce the O-line verdict reason.
    sig = "function wvVerdictReasons(a, b, wi) {"
    start = waivers_script.index(sig)
    i = start + len(sig) - 1
    depth, j = 0, i
    while True:
        if waivers_script[j] == "{":
            depth += 1
        elif waivers_script[j] == "}":
            depth -= 1
            if depth == 0:
                j += 1
                break
        j += 1
    fn_src = waivers_script[start:j]
    a = {"score_factors": {"proj": 12.0, "oline": 1.06}}
    b = {"score_factors": {"proj": 12.0, "oline": 0.94}}
    driver = (
        "const a = %s;\nconst b = %s;\n"
        "process.stdout.write(JSON.stringify(wvVerdictReasons(a, b, 0)));"
        % (json.dumps(a), json.dumps(b))
    )
    res = subprocess.run(["node", "-e", fn_src + "\n" + driver],
                         capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout) == ["a stronger offensive line"]
