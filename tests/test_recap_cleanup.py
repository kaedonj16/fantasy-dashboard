"""Contract tests for the Weekly Recap cleanup.

Two layout fixes plus a clickability sweep, all on the recap page:

- The Injury Report lives in a 336px rail; the old row put the status pill
  in a left column and the fantasy team on the right capped at 40%, so the
  player name in the middle truncated ("Amon-Ra St. B..."). The status now
  sits to the right of the name and the team drops to the sub-line.
- The Standings & Power Rankings pair used align-items:start, so each card
  sized to its own content. The grid now stretches and the rows inside
  each card flex to fill, so both cards always share one height.
- Every structured team/player name on the page opens the team/player
  modal: injury team, award-card opponent, League Activity teams and its
  injury player, Lineup Review teams and viewer-banner players, and the
  Game of the Week teams and notable outs.

Layout itself is verified in a browser harness; these guard the contracts.
"""
import os
import re

_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read(*parts):
    with open(os.path.join(_ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _recap_py():
    return _read("dashboard_services", "pages", "recap_page.py")


def _injuries_css():
    src = _recap_py()
    start = src.index("_RECAP_INJURIES_CSS")
    return src[start:src.index("</style>", start)]


# ── Injury Report ────────────────────────────────────────────────────────

def test_injury_status_no_longer_a_left_column():
    css = _injuries_css()
    # The team is in the sub-line now, not a right column capped at 40%.
    assert "max-width:40%" not in css
    assert ".rc-inj-top" in css
    # The row's top line is name + status side by side.
    src = _recap_py()
    row = src[src.index('class="rc-inj-row"'):src.index('class="rc-inj-sub"')]
    assert 'class="rc-inj-top"' in row
    assert 'class="rc-inj-name"' in row
    assert 'class="rc-inj-status"' in row


def test_injury_team_in_sub_line_and_clickable():
    src = _recap_py()
    assert 'team_link(inj["team"], inj.get("rid"))' in src
    seg = src[src.index("sub_html +="):src.index("sub_html +=") + 200]
    assert "rc-inj-team" in seg


# ── Standings & Power Rankings equal height ─────────────────────────────

def test_rank_grid_stretches_and_rows_fill():
    css = _read("static", "dashboard.css")
    grid = re.search(r"\.weekly-recap \.recap-rank-grid \{[^}]+\}", css)
    assert grid and "align-items:stretch" in grid.group(0)
    card = re.search(r"\.weekly-recap \.recap-rank-card \{[^}]+\}", css)
    assert card and "flex-direction:column" in card.group(0)
    row = re.search(r"\.weekly-recap \.recap-rank-card \.st-row \{[^}]+\}", css)
    assert row and "flex:1" in row.group(0)
    # Both cards carry the class.
    assert _recap_py().count("recap-rank-card") == 2


# ── Clickability sweep ───────────────────────────────────────────────────

def test_award_opponent_and_activity_names_clickable():
    src = _recap_py()
    assert 'team_link(opp["owner"], opp["rid"])' in src
    assert 'team_link(str(tm.get("name") or ""), rid)' in src
    assert 'team_link(str(data.get("name") or ""), data.get("rid"))' in src
    assert 'player_span(pname, pl.get("pid"))' in src


def test_activity_injury_rows_carry_rid():
    service = _read("dashboard_services", "service.py")
    assert '"rid": rid,' in service[service.index("def _append_injury_rows"):]


def test_lineup_review_team_names_clickable():
    app = _read("app.py")
    seg = app[app.index("def _build_lineup_analysis_html"):]
    seg = seg[:seg.index("def _mock_lineup_analysis_html")]
    # Player rows, missed-opportunity rows, and the viewer banner.
    assert seg.count("_clickable_team_name(") >= 3
    assert "_viewer_player(own['bench_player'])" in seg
    assert "_viewer_player(own['starter']" in seg


def test_gotw_names_clickable():
    wr = _read("dashboard_services", "ai", "weekly_recap.py")
    render = wr[wr.index("def _render_next_week_html"):]
    assert "def _team_span" in render and "team-clickable" in render
    assert "def _player_span" in render and "player-clickable" in render
    assert 'g.get("roster_id_a")' in render and 'g.get("roster_id_b")' in render
    # The producer keeps the ids the renderer needs.
    assert '"roster_id_a": g.get("roster_id_a")' in wr
    flags = wr[wr.index("def _starter_flags"):wr.index("def _starter_flags") + 2500]
    assert '"pid": pid,' in flags


def test_changelog_and_sw_for_recap_cleanup():
    from dashboard_services.changelog import CHANGELOG

    top = CHANGELOG[13]  # prepended above it: duplicate Tuesday dot removal, phone tabs + card odds removal, phone notifications panel + More-dot fix, recap glance username fix, Weekly Hub revamp, graphs overhaul, holdings sort + insights speed, rooting/bye cleanup, Front Office scan fix, fair-band fix, assistant fix, dot fix
    assert top["date"] == "2026-10-27"
    assert top["link"] == "/recap"
    assert "Injury Report" in top["text"]
    assert "—" not in top["text"]  # no em dashes in UI copy
    assert "br-fantasy-v125" in _read("static", "sw.js")
