"""Mobile standings UX: Sleeper-style readability.

On small screens the standings table keeps its data columns inside the
horizontally-scrolling .st-tblscroll wrapper (Seed/Team frozen via the global
sticky rules), but reads like the Sleeper app: the team name wraps to two
lines instead of truncating, and a "3-0 (1-0) · ▲ 3W" sub-line under the name
carries the record + streak (so the Record and Streak columns hide on mobile
in ordinary mode). Data cells get bigger type and breathing room; headers go
small-caps. Detailed mode keeps every column and hides the sub-line.
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
    """Make third-party imports of ``app`` work locally
    (app -> dashboard_services.ai.client -> openai,
     app -> dashboard_services.providers.espn_api -> espn_api,
     app -> stripe), for the test duration only. CI installs the real
    packages, so the stubs only materialize locally. Scoped via monkeypatch
    so they never shadow the real packages for other test modules (see the
    2026-09-25 stub-poisoning lesson in AGENTS.md)."""
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
    return pd.DataFrame([
        {"owner": "Alpha", "Wins": 10, "Losses": 2, "Ties": 0,
         "PF": 1400.0, "PA": 1100.0, "Streak": "W3", "avatar": ""},
        {"owner": "Bravo", "Wins": 7, "Losses": 5, "Ties": 0,
         "PF": 1200.5, "PA": 1150.25, "Streak": "L1", "avatar": ""},
    ])


def test_sub_line_present_in_full_table(_third_party_stubs):
    import app as appmod
    html = appmod.render_standings(
        _team_stats(), length=2, owner_to_rid={"Alpha": "1", "Bravo": "2"})
    assert html.count("st-team-sub") == 2


def test_sub_line_win_streak_format(_third_party_stubs):
    import app as appmod
    html = appmod.render_standings(
        _team_stats(), length=2, owner_to_rid={"Alpha": "1", "Bravo": "2"})
    # "10-2 · ▲ 3W": record, up-triangle, Sleeper-style streak.
    assert "10-2" in html
    assert "st-streak-dir up" in html and "&#9650;" in html and "3W" in html


def test_sub_line_loss_streak_format(_third_party_stubs):
    import app as appmod
    html = appmod.render_standings(
        _team_stats(), length=2, owner_to_rid={"Alpha": "1", "Bravo": "2"})
    assert "st-streak-dir down" in html and "&#9660;" in html and "1L" in html


def test_sub_line_escapes_streak(_third_party_stubs):
    import app as appmod
    df = _team_stats()
    df.loc[0, "Streak"] = "<b>W9</b>"
    html = appmod.render_standings(
        df, length=2, owner_to_rid={"Alpha": "1", "Bravo": "2"})
    assert "<b>W9</b>" not in html


def test_no_sub_line_in_compact_table(_third_party_stubs):
    import app as appmod
    html = appmod.render_standings_compact(
        _team_stats(), owner_to_rid={"Alpha": "1", "Bravo": "2"})
    assert "st-team-sub" not in html


def _mobile_blocks():
    """All @media (max-width: 640px) blocks, brace-matched, as text."""
    css = CSS.read_text()
    blocks = []
    for m in re.finditer(r"@media\s*\(max-width:\s*640px\)\s*\{", css):
        depth = 0
        for i in range(m.end() - 1, len(css)):
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(css[m.end():i])
                    break
    return blocks


def test_css_mobile_hides_record_and_streak_columns():
    # Record (col 3) and Streak (col 7) live in the sub-line on mobile, so the
    # columns hide in ordinary mode (not in Detailed mode).
    hidden = set()
    for block in _mobile_blocks():
        for m in re.finditer(
                r"([^{}]+)\{\s*display:\s*none\s*;", block):
            sel = m.group(1)
            if "standings-table[data-page=\"standings\"]" not in sel:
                continue
            if "nth-child(" not in sel:
                continue  # not a column rule (e.g. the div-meta pill)
            assert ":not(.show-detail)" in sel, (
                f"standings column hidden in Detailed mode: {sel.strip()[:80]}")
            for col in re.finditer(r"nth-child\((\d+)\)", sel):
                hidden.add(int(col.group(1)))
    assert {3, 7} <= hidden, f"expected cols 3 and 7 hidden, got {hidden}"
    assert hidden <= {3, 7}, f"unexpected standings columns hidden: {hidden}"


def test_css_mobile_sub_line_shown_non_detail_only():
    css = CSS.read_text()
    # Desktop default: hidden.
    assert re.search(
        r'\.st-tblscroll \.standings-table\[data-page="standings"\] '
        r'\.st-team-sub\s*\{\s*display:\s*none\s*;',
        css), "desktop .st-team-sub hide rule missing"
    shown = False
    for block in _mobile_blocks():
        m = re.search(
            r':not\(\.show-detail\)\s+\.st-team-sub\s*\{([^}]*)\}', block)
        if m and "display: block" in m.group(1):
            shown = True
    assert shown, "mobile .st-team-sub show rule missing"


def test_css_mobile_team_name_wraps_two_lines():
    found = False
    for block in _mobile_blocks():
        m = re.search(
            r"td\.team\s+\.team-clickable\s*\{([^}]*)\}", block)
        if m:
            found = True
            body = m.group(1)
            assert "-webkit-line-clamp: 2" in body
            assert "white-space: normal" in body
    assert found, "mobile team-name wrap rule missing"


def test_css_mobile_team_column_has_blink_clamp():
    found = False
    for block in _mobile_blocks():
        m = re.search(
            r":not\(\.standings-compact\)\s+td\.team\s*\{([^}]*)\}", block)
        if m:
            found = True
            body = m.group(1)
            assert "max-width: 0;" in body
            assert "max-width: 50vw;" in body
    assert found, "mobile td.team clamp rule missing"


def test_css_no_fixed_layout_on_standings():
    # table-layout:fixed mis-distributes free space when the division/clinch
    # colspan rows are present (Team collapsed to ~13px). It must not come back
    # as a declaration (the words appear in an explanatory comment).
    for block in _mobile_blocks():
        stripped = re.sub(r"/\*.*?\*/", "", block, flags=re.S)
        assert not re.search(r"table-layout\s*:", stripped)


def test_css_sticky_seed_team_rules_intact():
    css = CSS.read_text()
    # The frozen-pane rules (global, not in the media query) keep Seed/Team
    # stuck while the table scrolls horizontally on mobile.
    assert re.search(
        r'\.standings-table\[data-page="standings"\] tbody '
        r'tr:not\(\.st-div-row\):not\(\.pp-scnrow\):not\(\.pp-cutrow\) '
        r'td:nth-child\(1\),\n'
        r'\.standings-table\[data-page="standings"\] thead '
        r'tr:not\(\.st-div-row\):not\(\.pp-scnrow\):not\(\.pp-cutrow\) '
        r'th:nth-child\(2\)',
        css), "sticky Seed/Team frozen-pane rule missing"


def test_css_scroll_wrapper_allows_horizontal_scroll():
    css = CSS.read_text()
    m = re.search(r"\.st-tblscroll\s*\{([^}]*)\}", css)
    assert m, ".st-tblscroll rule missing"
    assert re.search(r"overflow-x\s*:\s*auto", m.group(1)), \
        ".st-tblscroll must allow horizontal scrolling"


def test_css_mobile_character_cards():
    """Character cards (standings + awards) get intentional mobile rules:
    tighter headers, wrapping team names (never truncated), comfortable rows."""
    head_rules = team_rules = award_rules = None
    for block in _mobile_blocks():
        m = re.search(r"\.cc-st-head,\s*\.cc-aw-head\s*\{([^}]*)\}", block)
        if m:
            head_rules = m.group(1)
        m = re.search(r"\.cc-st-team\s*\{([^}]*)\}", block)
        if m:
            team_rules = m.group(1)
        m = re.search(r"\.cc-award-val\s*\{([^}]*)\}", block)
        if m:
            award_rules = m.group(1)
    assert head_rules, "mobile character-card header rule missing"
    assert "padding: 14px 16px" in head_rules
    assert team_rules, "mobile team-name wrap rule missing"
    assert "overflow-wrap: break-word" in team_rules
    assert award_rules, "mobile award-value rule missing"
    assert "font-size: 22px" in award_rules
