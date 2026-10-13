"""Team modal Roster tab A2: depth-chart rooms, production stats, rail cards.

The tab was rebuilt from the approved A2 mock: position rooms with
collapsible benches, per-player league-scored PPG + finish, headshots, and
strength / picks / roster-shape cards under the rooms for every format.
These source-contract tests lock the payload fields and the render hooks
the layout depends on.
"""
from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).parents[1]
_APP_PY = (_REPO / "app.py").read_text()
_APP_JS = (_REPO / "static" / "app.js").read_text()
_CSS = (_REPO / "static" / "dashboard.css").read_text()


def _handler_body() -> str:
    start = _APP_PY.find('@app.route("/api/team-details/<roster_id>")')
    end = _APP_PY.find('@app.route("/api/player-league-trades/<player_id>")', start)
    assert start > 0 and end > start
    return _APP_PY[start:end]


def test_payload_carries_production_and_grouping_fields():
    body = _handler_body()
    # League-scored weekly points are the PPG source (same helper the
    # player modal uses), and the finish ranking is computed server-side.
    assert "_load_season_weekly_points(" in body
    for key in ('"ppg"', '"games"', '"ppg_finish"', '"slot"',
                '"is_ir"', '"is_taxi"', '"bye_week"', '"espn_id"'):
        assert key in body, key
    # Value rank labels are rebuilt on the displayed value's basis.
    assert "_value_rank_label" in body


def test_positional_strength_no_longer_redraft_only():
    body = _handler_body()
    assert '"positional_strength": positional_strength' in body
    # The old gate computed the profile only for redraft leagues.
    assert "Positional strength -- redraft's answer" not in body
    start = body.find("positional_strength = []")
    chunk = body[start:start + 200]
    assert "if is_redraft" not in chunk


def test_render_uses_rooms_bench_lines_and_rail_cards():
    for marker in ("tm2-bench-line", "data-tm2-toggle", "data-tm2-group",
                   "data-tm2-mode", "Starter PPG combined",
                   "Value in starters", "Roster Shape",
                   "Positional Strength", "tm2-pick-chip",
                   "tm2ImgFallback"):
        assert marker in _APP_JS, marker
    # The flat one-list roster render is gone.
    assert "tm-roster-list" not in _APP_JS
    # Picks stay dynasty-only in the client as well.
    assert "if (!data.is_redraft)" in _APP_JS


def test_css_keys_layout_to_modal_width():
    assert ".team-modal { container-type: inline-size; }" in _CSS
    assert "@container (max-width: 700px)" in _CSS
    assert "tm2-rooms" in _CSS
    # Rooms can never be narrower than a player row needs.
    assert "minmax(min(430px, 100%), 1fr)" in _CSS
