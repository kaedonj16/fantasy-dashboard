"""Portfolio "Last Week" stat bar cell + refresh repaint guards.

Covers:
- last_finalized_week_result(): (week, W/L/T) of the most recently finalized
  week per league, (None, None) when nothing is finalized yet.
- build_portfolio_body(): renders the 4th "Last Week" pf-stat cell only after
  week 1 (current_week > 1) and only when at least one league has a result;
  per-card data-lw-result hooks for the client-side recompute.
- The repaint-guard contract: setHtmlIfChanged/updateAggregateRecord/
  updateLastWeekRecord exist in static/app.js so refresh re-renders skip
  unchanged DOM.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytest.importorskip("pandas")
pytest.importorskip("flask")


# ── Minimal DataFrame stand-in (no pandas needed) ────────────────────────────
class _FakeSeries:
    def __init__(self, values):
        self._v = list(values)

    def astype(self, _t):
        return _FakeSeries([str(v) for v in self._v])

    def __eq__(self, other):
        return _FakeMask([v == other for v in self._v])


class _FakeMask:
    def __init__(self, flags):
        self._f = list(flags)


class _FakeDF:
    def __init__(self, rows):
        self._rows = list(rows)
        self.columns = list(self._rows[0].keys()) if self._rows else []
        self.empty = not self._rows

    def __getitem__(self, key):
        if isinstance(key, str):
            return _FakeSeries([r.get(key) for r in self._rows])
        return _FakeDF([r for r, f in zip(self._rows, key._f) if f])

    def sort_values(self, col):
        return _FakeDF(sorted(self._rows, key=lambda r: r.get(col)))

    @property
    def iloc(self):
        rows = self._rows

        class _I:
            def __getitem__(self, _i, _rows=rows):
                return dict(_rows[_i])

        return _I()


def _weekly(rows):
    return _FakeDF(rows)


def _row(rid, week, pts, pa, finalized=True):
    return {"roster_id": rid, "week": week, "points": pts,
            "points_against": pa, "finalized": finalized}


# ── last_finalized_week_result ────────────────────────────────────────────────
def test_last_finalized_week_picks_latest_finalized():
    from dashboard_services.portfolio_summary import last_finalized_week_result
    df = _weekly([
        _row("7", 1, 100.0, 90.0),
        _row("7", 2, 80.0, 95.0),
        _row("7", 3, 110.0, 110.0, finalized=False),  # in progress, ignored
        _row("9", 2, 200.0, 50.0),  # other roster, ignored
    ])
    assert last_finalized_week_result(df, "7") == (2, "L")


def test_last_finalized_week_win_and_tie():
    from dashboard_services.portfolio_summary import last_finalized_week_result
    df = _weekly([_row("7", 1, 100.0, 90.0)])
    assert last_finalized_week_result(df, "7") == (1, "W")
    df = _weekly([_row("7", 1, 100.0, 100.0)])
    assert last_finalized_week_result(df, "7") == (1, "T")


def test_last_finalized_week_none_when_nothing_finalized():
    from dashboard_services.portfolio_summary import last_finalized_week_result
    assert last_finalized_week_result(None, "7") == (None, None)
    assert last_finalized_week_result(_weekly([]), "7") == (None, None)
    df = _weekly([_row("7", 1, 100.0, 90.0, finalized=False)])
    assert last_finalized_week_result(df, "7") == (None, None)


# ── build_portfolio_body rendering ────────────────────────────────────────────
def _league(league_id, name, wins, losses, lw_result=None, ties=0):
    lg = {
        "league_id": league_id,
        "name": name,
        "platform": "sleeper",
        "season": 2026,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "record": f"{wins}-{losses}" + (f"-{ties}" if ties else ""),
        "rank": 3,
        "total_teams": 10,
        "pf": 500.0,
        "total_value": 100.0,
        "all_players": {},
        "streak": [],
        "urgency": 0,
        "pos_user_vals": {},
        "pos_league_avgs": {},
        "pos_user_pctile": {},
        "pos_user_rank": {},
        "offseason": False,
        "team_name": name,
        "is_favorite": False,
    }
    if lw_result is not None:
        lg["last_week_result"] = lw_result
        lg["last_week"] = 2
    return lg


def _render(leagues, current_week):
    from app import app, build_portfolio_body
    tw = sum(l["wins"] for l in leagues)
    tl = sum(l["losses"] for l in leagues)
    with app.test_request_context("/"):
        return build_portfolio_body("kaedon", leagues, leagues, 2026,
                                    num_leagues=len(leagues),
                                    total_wins=tw, total_losses=tl,
                                    current_week=current_week)


def test_last_week_cell_shown_after_week_1():
    leagues = [_league("1", "Alpha", 3, 1, lw_result="W"),
               _league("2", "Beta", 2, 2, lw_result="L")]
    html = _render(leagues, current_week=3)
    assert "pf-stat-bar pf-stat-bar--4" in html
    m = re.search(r"data-portfolio-lw-record[^>]*>([^<]*)<", html)
    assert m and m.group(1) == "1-1"
    assert html.count("data-lw-result=") == 2
    assert "Last Week" in html


def test_last_week_cell_hidden_in_week_1_and_preseason():
    leagues = [_league("1", "Alpha", 0, 0, lw_result="W")]
    for week in (0, 1):
        html = _render(leagues, current_week=week)
        assert "data-portfolio-lw-record" not in html
        assert "class='pf-stat-bar pf-stat-bar--4'" not in html
        assert "Last Week" not in html


def test_last_week_cell_hidden_when_no_results():
    leagues = [_league("1", "Alpha", 3, 1), _league("2", "Beta", 2, 2)]
    html = _render(leagues, current_week=3)
    assert "data-portfolio-lw-record" not in html
    assert "class='pf-stat-bar pf-stat-bar--4'" not in html


def test_last_week_aggregate_with_tie():
    leagues = [_league("1", "Alpha", 3, 1, lw_result="W"),
               _league("2", "Beta", 2, 2, lw_result="L"),
               _league("3", "Gamma", 1, 3, lw_result="T")]
    html = _render(leagues, current_week=4)
    m = re.search(r"data-portfolio-lw-record[^>]*>([^<]*)<", html)
    assert m and m.group(1) == "1-1-1"


def test_last_week_cell_color_follows_record():
    leagues = [_league("1", "Alpha", 3, 1, lw_result="W"),
               _league("2", "Beta", 2, 2, lw_result="W")]
    html = _render(leagues, current_week=3)
    m = re.search(r"class='pf-stat-val ([^']*)' data-portfolio-lw-record", html)
    assert m and "color-win" in m.group(1)


# ── repaint-guard contract in static/app.js ───────────────────────────────────
def test_refresh_repaint_guards_present():
    src = (ROOT / "static" / "app.js").read_text()
    assert "function setHtmlIfChanged(el, html)" in src
    assert "setHtmlIfChanged(stats, statsHtml)" in src
    assert "card._pfStrengthHtml" in src
    assert "agg._pfRec" in src
    assert "function updateLastWeekRecord()" in src
    assert 'data-lw-result' in src or "dataset.lwResult" in src
