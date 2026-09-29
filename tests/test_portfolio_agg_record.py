"""Regression: portfolio aggregate record must stay hydratable.

The My Leagues summary bar is server-rendered from warm leagues only; cold
leagues hydrate later via /api/portfolio/card. app.js's updateAggregateRecord()
recomputes the combined W-L-T as cards hydrate, but only when the HTML carries
the data hooks (data-portfolio-agg-record on the summary bar, data-wins/losses/ties
per card). Commit 7a1a597a silently reverted those attributes (whole-file
overlay from a stale base), so the JS early-returned and the bar went stale.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytest.importorskip("flask")


def _league(league_id, name, wins, losses, ties=0):
    return {
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


def _render(tw, tl, tt=0):
    from app import app, build_portfolio_body

    leagues = [_league("1", "Alpha", 3, 1), _league("2", "Beta", 2, 2, ties=1)]
    with app.test_request_context("/"):
        return build_portfolio_body("kaedon", leagues, leagues, 2026, num_leagues=2,
                                    total_wins=tw, total_losses=tl, total_ties=tt)


def test_summary_bar_carries_aggregate_hook():
    html = _render(5, 3, 1)
    m = re.search(
        r"<div class='pf-stat-val [^']*' data-portfolio-agg-record"
        r" data-wins='(\d+)' data-losses='(\d+)' data-ties='(\d+)'>5-3-1</div>",
        html,
    )
    assert m, "summary record element must carry data-portfolio-agg-record + data-wins/losses/ties"


def test_league_cards_carry_per_card_record():
    html = _render(5, 3, 1)
    assert "data-lg-key='sleeper:1' data-favorite='false' data-platform='sleeper' data-league-id='1' data-season='2026' data-wins='3' data-losses='1' data-ties='0'" in html
    assert "data-lg-key='sleeper:2' data-favorite='false' data-platform='sleeper' data-league-id='2' data-season='2026' data-wins='2' data-losses='2' data-ties='1'" in html


def test_js_aggregate_recompute_wired_to_hook():
    js = (ROOT / "static" / "app.js").read_text()
    assert "querySelector('[data-portfolio-agg-record]')" in js
    assert "function updateAggregateRecord()" in js
