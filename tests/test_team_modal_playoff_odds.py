"""Team-modal playoff odds: 100 / 0 are reserved for settled teams.

Kaedon's rule (same as the Hub fix): playoff odds can never read 100 or 0
unless the team is mathematically in or out. The simulator already caps
undecided odds at 99.9 / 0.1 and reserves exact 100.0 / 0.0 for clinched /
eliminated teams, but the team modal rendered the tile with Math.round, so
99.9 displayed as a false 100% and 0.1 as a false 0%. Every team-detail
surface must use the floored one-decimal display, and the Teams page drawer
must show the odds at all (its payload carried none).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
TEAMS_JS = (ROOT / "static" / "teams.js").read_text(encoding="utf-8")
TEAMS_PAGE = (ROOT / "dashboard_services" / "pages" / "teams_page.py").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def _extract_fn(src: str, name: str) -> str:
    start = src.index(f"function {name}(")
    brace = src.index("{", start)
    depth = 0
    for i in range(brace, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
    raise AssertionError(f"could not extract {name}")


def _run_formatter(src: str, name: str, values) -> list[str]:
    node = shutil.which("node")
    assert node, "node is required to execute the JS formatters"
    fn = _extract_fn(src, name)
    vals = ",".join(repr(v) for v in values)
    script = f"{fn}\nconsole.log([{vals}].map(v => {name}(v)).join(','))"
    out = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, timeout=30,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().split(",")


CASES = [99.9, 99.99, 100, 0, 0.1, 42.25, 50]
EXPECTED = ["99.9", "99.9", "100", "0", "0.1", "42.2", "50.0"]


def test_team_modal_formatter_never_rounds_to_100_or_0():
    got = _run_formatter(APP_JS, "_fmtPlayoffOddsDisplay", CASES)
    assert got == EXPECTED


def test_teams_drawer_formatter_never_rounds_to_100_or_0():
    got = _run_formatter(TEAMS_JS, "_tdFmtPlayoffOdds", CASES)
    assert got == EXPECTED


def test_team_modal_tile_uses_formatter_not_math_round():
    modal = APP_JS[APP_JS.index("function renderTeamDetails("):]
    tile = modal[modal.index("data.playoff_odds") - 200: modal.index("data.playoff_odds") + 600]
    assert "_fmtPlayoffOddsDisplay(data.playoff_odds)" in tile
    assert "Math.round(parseFloat(data.playoff_odds))" not in APP_JS


def test_drawer_payload_carries_playoff_odds_for_every_mode():
    # Odds map is built outside the redraft-only branch...
    grades = TEAMS_PAGE[TEAMS_PAGE.index("team_grades = {rid:"):]
    assert "_po_by_rid = {" in grades
    assert grades.index("_po_by_rid = {") < grades.index("if is_redraft_m:")
    # ...and ships in the drawer payload.
    assert '"playoff_odds": _po_by_rid.get(str(rid))' in TEAMS_PAGE


def test_my_leagues_cards_use_the_same_formatter():
    """The odds VALUE must use the floored one-decimal formatter wherever
    odds still display: the team modal and the Front Office board group
    meta. The My Leagues league cards no longer display odds at all
    (2026-10-27: the stat never populated, a permanent dash), so neither
    the client hydrate nor the server card may reference the odds value.
    The trend delta arrows keep their whole-number shape by design."""
    assert "_fmtPlayoffOddsDisplay(data.playoff_odds)" in APP_JS  # team modal
    assert "_fmtPlayoffOddsDisplay(meta.odds_pct)" in APP_JS
    assert "Math.round(Number(data.playoff_odds_pct))" not in APP_JS
    assert "Math.round(Number(meta.odds_pct))" not in APP_JS
    rs = APP_JS[APP_JS.index("function renderSummary("):]
    rs = rs[:rs.index("function updateAggregateRecord()")]
    assert "playoff_odds_pct" not in rs
    app_py = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "{_fmt_playoff_pct_display(_odds_pct)}%" not in app_py
    assert "{float(_odds_pct):.0f}%" not in app_py


def test_changelog_announces_team_modal_odds_fix():
    from dashboard_services.changelog import CHANGELOG

    entry = CHANGELOG[16]  # prepended above it: recap/standings dedupe, duplicate Tuesday dot removal, phone tabs + card odds removal, phone notifications panel + More-dot fix, recap glance username fix, Weekly Hub revamp, graphs overhaul, holdings sort + insights speed, the rooting/bye cleanup, the Front Office scan fix
    assert entry["date"] == "2026-10-27"
    assert entry["tag"] == "fix"
    assert "playoff odds" in entry["text"]
    assert "My Leagues" in entry["text"]
    assert "100%" in entry["text"] and "clinched" in entry["text"]
    assert "—" not in entry["text"]  # no em dashes in UI copy


def test_drawer_head_renders_odds_pill():
    head = TEAMS_JS[TEAMS_JS.index("function _tdHeadHtml("):]
    head = head[: head.index("function ", 10) if "function " in head[10:] else len(head)]
    assert "td-head-pills" in head
    assert "td-odds" in head
    assert "Playoff odds" in head
    assert "_tdFmtPlayoffOdds(t.playoff_odds)" in head
    # Pills row + odds pill are styled.
    assert re.search(r"\.td-head-pills\s*\{", CSS)
    assert re.search(r"\.td-odds\s*\{", CSS)
