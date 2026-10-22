"""Playoff Picture rework (Direction C) on the full standings table.

Status is its own column (dot + word) after Team; the team cell carries no
playoff chips, division leaders get a plain "Division leader" subline,
scenario sentences are not rendered, the legend dots are gone from the
context line, and the CSS kills the old left bars / row tints on this
table, renders luck as plain text, and boxes podium seeds like the rest.
"""
import re
import sys
import types
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

CSS = Path("static/dashboard.css")


@pytest.fixture
def _third_party_stubs(monkeypatch):
    """Scope third-party stubs to this module (see test_standings_mobile_ux)."""
    try:
        import openai  # noqa: F401
        return
    except ImportError:
        pass

    class _SmartStub(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            cls = type(name, (), {})
            setattr(self, name, cls)
            return cls

    def _install(dotted):
        parts = dotted.split(".")
        for i in range(1, len(parts) + 1):
            sub = ".".join(parts[:i])
            if sub not in sys.modules:
                mod = _SmartStub(sub)
                mod.__path__ = []
                monkeypatch.setitem(sys.modules, sub, mod)
                if i > 1:
                    setattr(sys.modules[".".join(parts[:i - 1])],
                            parts[i - 1], mod)

    for _pkg in ("openai", "espn_api", "espn_api.football",
                 "espn_api.football.box_score", "stripe"):
        _install(_pkg)


def _team_stats():
    # Week-6 snapshot, 2 playoff spots, 14 regular weeks. Hand-computed
    # picture: Alpha (6-0) is In, Bravo (4-2) is an inside Bubble (1 GB
    # clear of the first team out), Charlie (3-3) is 1 GB outside.
    return pd.DataFrame([
        {"owner": "Alpha", "Rank": 1, "Wins": 6, "Losses": 0, "Ties": 0,
         "PF": 900.0, "PA": 500.0, "Streak": "W6", "avatar": "",
         "Win%": 1.0, "AVG": 150.0, "STD": 10.0,
         "past_sos": 100.0, "ros_sos": 100.0},
        {"owner": "Bravo", "Rank": 2, "Wins": 4, "Losses": 2, "Ties": 0,
         "PF": 800.0, "PA": 600.0, "Streak": "W1", "avatar": "",
         "Win%": 0.667, "AVG": 133.3, "STD": 10.0,
         "past_sos": 100.0, "ros_sos": 100.0},
        {"owner": "Charlie", "Rank": 3, "Wins": 3, "Losses": 3, "Ties": 0,
         "PF": 700.0, "PA": 650.0, "Streak": "L1", "avatar": "",
         "Win%": 0.5, "AVG": 116.7, "STD": 10.0,
         "past_sos": 100.0, "ros_sos": 100.0},
        {"owner": "Delta", "Rank": 4, "Wins": 2, "Losses": 4, "Ties": 0,
         "PF": 600.0, "PA": 700.0, "Streak": "L2", "avatar": "",
         "Win%": 0.333, "AVG": 100.0, "STD": 10.0,
         "past_sos": 100.0, "ros_sos": 100.0},
        {"owner": "Echo", "Rank": 5, "Wins": 1, "Losses": 5, "Ties": 0,
         "PF": 500.0, "PA": 800.0, "Streak": "L4", "avatar": "",
         "Win%": 0.167, "AVG": 83.3, "STD": 10.0,
         "past_sos": 100.0, "ros_sos": 100.0},
        {"owner": "Foxtrot", "Rank": 6, "Wins": 0, "Losses": 6, "Ties": 0,
         "PF": 400.0, "PA": 900.0, "Streak": "L6", "avatar": "",
         "Win%": 0.0, "AVG": 66.7, "STD": 10.0,
         "past_sos": 100.0, "ros_sos": 100.0},
    ])


def _render(_third_party_stubs, df=None, **kw):
    pytest.importorskip("flask")
    import app as appmod
    o2r = {name: str(i + 1) for i, name in enumerate(
        ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"])}
    return appmod.render_standings(
        _team_stats() if df is None else df, length=6, owner_to_rid=o2r,
        playoff_spots=2, total_regular_weeks=14, **kw)


def test_status_column_header_and_cells(_third_party_stubs):
    html = _render(_third_party_stubs)
    thead = html.split("<thead>")[1].split("</thead>")[0]
    headers = [h.split(">")[-1] for h in thead.split("</th>")[:-1]]
    assert headers[2] == "Status"
    assert len(headers) == 19
    # In team: green dot + "In".
    assert "pp-status-cell pp-status-in" in html
    assert "<i class='pp-sdot'></i>In</td>" in html
    # Inside bubble team: amber dot + "Bubble".
    assert "pp-status-cell pp-status-bub" in html
    assert "<i class='pp-sdot'></i>Bubble</td>" in html
    # Outside bubble team: games back instead of the word.
    assert "<i class='pp-sdot'></i>1 GB</td>" in html


def test_outside_team_level_on_record_reads_bubble_not_0gb(_third_party_stubs):
    # Charlie level with the cut (Bravo, 4-2) on record, out only on PF:
    # games_back is 0, which must read "Bubble", never "0 GB".
    df = _team_stats()
    mask = df["owner"] == "Charlie"
    df.loc[mask, ["Wins", "Losses", "PF", "Win%"]] = [4, 2, 700.0, 0.667]
    html = _render(_third_party_stubs, df=df)
    assert "0 GB" not in html
    charlie_row = html.split('data-rk-key="Charlie"')[1].split("</tr>")[0]
    assert "<i class='pp-sdot'></i>Bubble</td>" in charlie_row
    # A team genuinely behind on record still reads its games back.
    assert "<i class='pp-sdot'></i>2 GB</td>" in html


def test_no_chips_scenarios_or_legend(_third_party_stubs):
    html = _render(_third_party_stubs)
    # No team-cell playoff chips of any status.
    assert "pp-tag" not in html
    # Scenario sentences are not rendered as rows (the engine still
    # computes them; this renderer must not print them).
    assert "pp-scnrow" not in html
    assert "Win out and you're in" not in html
    assert "back of the last playoff spot" not in html
    # Context line keeps the format text but drops the dot legend.
    assert "pp-ctx-main" in html
    assert "Top 2 advance" in html
    assert "pp-legend" not in html
    assert "pp-dot" not in html
    # No pp-* status row classes (the old left-bar / tint hooks).
    assert "pp-bubble" not in html
    assert "pp-clinched" not in html


def test_no_picture_renders_dash_status(_third_party_stubs):
    pytest.importorskip("flask")
    import app as appmod
    html = appmod.render_standings(
        _team_stats(), length=6,
        owner_to_rid={"Alpha": "1", "Bravo": "2", "Charlie": "3",
                      "Delta": "4", "Echo": "5", "Foxtrot": "6"})
    assert ">Status</th>" in html
    assert "pp-status-cell pp-status-none" in html


def test_division_leader_subline_and_colspan(_third_party_stubs):
    pytest.importorskip("flask")
    import app as appmod
    df = _team_stats().head(4)
    o2r = {"Alpha": "1", "Bravo": "2", "Charlie": "3", "Delta": "4"}
    divisions = {"by_rid": {1: 1, 2: 1, 3: 2, 4: 2},
                 "names": {1: "East", 2: "West"}, "ids": [1, 2], "count": 2}
    html = appmod.render_standings(
        df, length=4, owner_to_rid=o2r, divisions=divisions,
        playoff_spots=2, total_regular_weeks=14)
    assert "<div class='st-div-sub'>Division leader</div>" in html
    assert "<span class='st-div-lead'" not in html
    assert "colspan='19'" in html
    assert "colspan='18'" not in html


def _css():
    return CSS.read_text()


def test_css_status_cell_styles():
    css = _css()
    m = re.search(r"\.pp-status-cell \.pp-sdot\s*\{([^}]*)\}", css)
    assert m, ".pp-sdot rule missing"
    assert "border-radius: 50%" in m.group(1)
    assert "width: 8px" in m.group(1)
    assert re.search(r"td\.pp-status-in\s*\{[^}]*var\(--win\)", css)
    assert re.search(r"td\.pp-status-bub\s*\{[^}]*--warning", css)
    assert re.search(r"td\.pp-status-bye\s*\{[^}]*var\(--pp-bye\)", css)


def test_css_scoped_no_bar_no_tint_override():
    css = _css()
    # The scoped override on the standings page table zeroes the inset
    # bars (pp statuses + division leader) and the row tints.
    m = re.search(
        r'\.standings-table\[data-page="standings"\] tr\.pp-bubble '
        r'td:first-child[^{]*\{([^}]*)\}', css)
    assert m and "box-shadow: none" in m.group(1)
    assert re.search(
        r'\.standings-table\[data-page="standings"\] tr\.st-div-leader '
        r'td:first-child[^{]*\{[^}]*box-shadow: none', css)
    assert re.search(
        r'\.standings-table\[data-page="standings"\] tr\.pp-bubble td,'
        r'[^}]*background: transparent', css)


def test_css_luck_is_plain_text():
    css = _css()
    base = re.search(r"\.luck-chip\s*\{([^}]*)\}", css)
    assert base, ".luck-chip rule missing"
    assert "background" not in base.group(1)
    assert "border" not in base.group(1)
    assert "border-radius" not in base.group(1)
    assert "tabular-nums" in base.group(1)
    pos = re.search(r"\.luck-chip\.luck-pos\s*\{([^}]*)\}", css)
    assert pos and "#b45309" in pos.group(1)
    assert "font-weight: 800" in pos.group(1)
    assert "background" not in pos.group(1)
    neg = re.search(r"\.luck-chip\.luck-neg\s*\{([^}]*)\}", css)
    assert neg and "#2563eb" in neg.group(1)
    assert "font-weight: 800" in neg.group(1)
    neu = re.search(r"\.luck-chip\.luck-neu\s*\{([^}]*)\}", css)
    assert neu and "var(--text-subtle)" in neu.group(1)


def test_css_uniform_seed_boxes_scoped_and_desktop_gated():
    css = _css()
    m = re.search(
        r"@media \(min-width: 769px\)\s*\{\s*"
        r'\.standings-table\[data-page="standings"\] \.rank-first',
        css)
    assert m, "desktop-gated uniform seed box override missing"
    block = css[m.start():m.start() + 900]
    assert ".rank-second" in block and ".rank-third" in block
    assert "border: 1px solid var(--border)" in block
    assert "width: 22px" in block


def test_css_div_subline_style_and_mobile_hide():
    css = _css()
    bodies = re.findall(r"\.st-div-sub\s*\{([^}]*)\}", css)
    assert any("font-size: 11.5px" in b for b in bodies), bodies
    # Hidden in the <=768 scoreboard block (gated like its neighbours).
    assert re.search(
        r":not\(\.show-detail\):not\(\.standings-compact\) \.st-div-sub\s*"
        r"\{\s*display: none", css)
    # The old chip rule no longer lives in the scoreboard block.
    block_start = css.find("Mobile standings scoreboard")
    block = css[block_start:block_start + 20000]
    assert ".st-div-lead {" not in block


def test_js_default_sort_targets_record_then_pf_not_status():
    """initStandingsSort hard-codes column indexes for #stats. With Status
    inserted at index 2 (PR #2743), the default sort and numeric set must
    point at Record (3) / PF (4) — pointing at 2 sorts the table by the
    Status text ("2 GB" > "1 GB" > In/Bubble), which shipped as a bug."""
    js = Path("static/app.js").read_text()
    i = js.find("function initStandingsSort")
    assert i != -1
    block = js[i:i + 4000]
    assert "let sortSpec = [{ col: 3, dir: -1 }, { col: 4, dir: -1 }];" in block
    m = re.search(r"const NUMERIC_COLS = new Set\(\[([0-9, ]+)\]\)", block)
    assert m, "NUMERIC_COLS not found in initStandingsSort"
    cols = {int(x) for x in m.group(1).split(",")}
    assert cols == {3, 4, 5, 6, 7, 8, 9}
    assert 2 not in cols
