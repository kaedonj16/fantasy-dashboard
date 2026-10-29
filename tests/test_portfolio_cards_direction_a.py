"""Direction A league-card redesign (2026-10-29): H4 header (team name
headline, full-width meta of league / platform / posture only), tinted
record band, Fix / Best room columns, flat state-aware score strip,
no footer, one Updated line above the grid.

The mock was approved piece by piece in artifact
my-leagues-cards-3-directions; these tests pin the real build to it.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("flask")
pytest.importorskip("pandas")


def _league(**kw):
    lg = {
        "league_id": "1", "name": "Alpha League", "platform": "sleeper",
        "season": 2026, "wins": 6, "losses": 1, "ties": 0,
        "record": "6-1", "rank": 1, "total_teams": 10, "pf": 900.0,
        "total_value": 4200.0, "all_players": {}, "streak": ["L", "W", "W"],
        "urgency": 3, "pos_user_vals": {}, "pos_league_avgs": {},
        "pos_user_pctile": {},
        "pos_user_rank": {"QB": 5, "RB": 3, "WR": 1, "TE": 8},
        "offseason": False, "team_name": "Caleb's Casting Couch",
        "is_favorite": False, "posture": "Contender",
        "posture_cls": "win", "posture_color": "#22c55e",
        "posture_shape": "TOP-HEAVY",
    }
    lg.update(kw)
    return lg


def _render(leagues, account=False):
    from flask import session

    from app import app, build_portfolio_body
    tw = sum(l["wins"] for l in leagues)
    tl = sum(l["losses"] for l in leagues)
    with app.test_request_context("/"):
        if account:
            # The kebab menu holds Remove league, which only exists for
            # account-linked leagues (_can_unlink reads the session).
            session["account_id"] = 7
        return build_portfolio_body(
            "kaedon", leagues, leagues, 2026, num_leagues=len(leagues),
            holdings=[], nfl_exposure=[], cross_pos={}, total_wins=tw,
            total_losses=tl, total_ties=0, current_week=9,
        )


def _card(html):
    return html.split("data-summary-card", 1)[1][:4000]


def test_header_team_headline_and_meta_contents():
    card = _card(_render([_league()]))
    # Team name headlines; league name is demoted to the meta line.
    assert "class='pf-lg-name'>Caleb&#x27;s Casting Couch</a>" in card \
        or "Caleb's Casting Couch" in card
    meta = card.split("class='pf-lg-meta'", 1)[1].split("</div>", 1)[0]
    assert "Alpha League" in meta
    assert "Sleeper" in meta
    assert "data-posture" in meta and "Contender" in meta
    # Meta holds ONLY league / platform / posture: no team count, no
    # shape tag, no move count.
    assert "10 teams" not in meta
    assert "TOP-HEAVY" not in card
    assert "data-posture-shape" not in card


def test_header_star_and_kebab_with_remove_in_menu():
    card = _card(_render([_league()], account=True))
    assert "pf-lg-fav" in card
    assert "pf-lg-menu-btn" in card
    assert "pf-lg-menu" in card
    assert ">Remove league</button>" in card
    # The remove action is not a bare x on the header anymore.
    assert "pf-lg-crest" not in card


def test_record_band_and_plain_text_streak():
    html = _render([_league()])
    card = _card(html)
    assert "class='pf-lg-rec color-win'>6-1</span>" in card
    assert "pf-lg-standing" in card and "1<small>st</small>" in card
    assert "<span class='pf-lg-streak'>L-W-W</span>" in card
    assert "pf-s-pill" not in card
    # Band layout: record + streak in the left column (streak under the
    # record), standing alone in the right column, set bigger than the
    # first build had it.
    band = card.split("data-summary-stats", 1)[1].split("pf-lg-retry", 1)[0]
    main_col = band.split("pf-lg-band-main", 1)[1].split("pf-lg-band-side", 1)[0]
    assert "pf-lg-rec" in main_col and "pf-lg-streak" in main_col
    side_col = band.split("pf-lg-band-side", 1)[1]
    assert "pf-lg-standing" in side_col and "pf-lg-streak" not in side_col
    assert ".pf-lg-standing{font-size:26px" in html
    assert "justify-content:space-between" in html.split(".pf-lg-stats{", 1)[1].split("}", 1)[0]
    # Full-bleed tint band, light + dark tones from the approved mock.
    assert "background:#e8f0f7" in html
    assert ":root[data-theme='dark'] .pf-lg-stats{background:#0b1d31;}" in html
    assert ".pf-lg-rec{font-size:50px" in html
    assert ".pf-lg-rec{font-size:42px;}" in html  # phone step-down


def test_rooms_fix_left_best_right_from_same_ranks():
    card = _card(_render([_league()]))
    rooms = card.split("data-rooms", 1)[1].split("</div>", 1)[0]
    # pos_user_rank {QB 5, RB 3, WR 1, TE 8}: weakest TE, strongest WR.
    assert "Fix: " in rooms and "TE · 8th of 10" in rooms
    assert "Best: " in rooms and "WR · 1st of 10" in rooms
    assert rooms.index("Fix:") < rooms.index("Best:")


def test_rooms_single_rank_shows_fix_only():
    lg = _league(pos_user_rank={"QB": 7, "RB": None, "WR": None, "TE": None})
    card = _card(_render([lg]))
    assert "Fix: " in card and "QB · 7th of 10" in card
    assert "Best: " not in card.split("data-rooms", 1)[1].split("</div>", 1)[0]


def test_standing_colors_top3_green_bottom_third_red_middle_muted():
    html = _render([_league()])
    assert ".pf-lg-v--top{color:var(--win);}" in html
    assert ".pf-lg-v--weak{color:var(--loss);}" in html
    # The rank-1 fixture is a top-3 standing: green.
    assert "pf-lg-standing pf-lg-v--top" in _card(html)
    top3 = _card(_render([_league(rank=3)]))
    assert "pf-lg-standing pf-lg-v--top" in top3
    low = _card(_render([_league(rank=8, wins=1, losses=6, record="1-6")]))
    assert "pf-lg-standing pf-lg-v--weak" in low
    mid = _card(_render([_league(rank=5, wins=4, losses=3, record="4-3")]))
    assert "class='pf-lg-standing'>5" in mid


def test_fix_room_reads_red_best_reads_dark():
    html = _render([_league()])
    assert ".pf-lg-room.fix strong{color:var(--loss);}" in html
    # Best is bold dark like the mock, not its tier color (a strong room
    # read all green in the first build).
    assert ".pf-lg-room.best strong{color:var(--text);}" in html


def test_no_footer_and_one_updated_line_above_grid():
    html = _render([_league()])
    card = _card(html)
    assert "pf-lg-foot" not in html
    assert "class='pf-lg-fo'" not in card and "pf-lg-open" not in card
    assert "data-summary-updated" not in html
    assert html.count("data-portfolio-updated") == 1
    # The Updated line sits above the grid, not inside a card.
    assert html.index("data-portfolio-updated") < html.index("<div class='pf-lg-grid'>")


def test_score_strip_is_last_and_flat():
    html = _render([_league()])
    card = _card(html)
    # Score strip (live slot) comes after the rooms, at the card's end.
    assert card.index("data-rooms") < card.index("data-lg-live")
    # Flat: hairline on top, no inset background on the strip itself.
    assert ".pf-lg-live{border-top:1px solid var(--grid);" in html
    live_rule = html.split(".pf-lg-live{", 1)[1].split("}", 1)[0]
    assert "background" not in live_rule
    assert "border-radius" not in live_rule


def test_card_chrome_light_border_and_dark_shadow():
    html = _render([_league()])
    assert "border:1px solid var(--border);border-radius:12px" in html
    assert ":root[data-theme='dark'] .pf-lg-card{box-shadow:0 2px 10px rgba(0,0,0,.35);}" in html


def test_score_strip_matches_mock_head_inline_you_and_pink_bar():
    client = (ROOT / "static" / "app.js").read_text()
    fn = client.split("function matchupHtml(data)")[1].split("function renderMatchup")[0]
    # Head line (mock strip): "Wk n · vs <opponent>" on the left, your
    # win chance on the same line ("% to win", red when under 50 via
    # is-low). The opponent is named only in this head line.
    assert "pf-live-head" in fn and "' \u00b7 vs '" in fn
    assert "pf-live-head-wp" in fn and "% to win" in fn and "is-low" in fn
    # Your score line is inline: "You <number> proj" (You word first,
    # proj tag after the number); no caption element over either number.
    assert "pf-live-you" in fn and ">You</span>" in fn
    assert "pf-live-tag" in fn
    assert "pf-live-lbl" not in fn
    # No percentage labels under the bar at all: no label row, and the
    # opponent's complementary percentage is never rendered.
    assert "pf-live-wp-lbls" not in fn and "pf-live-wp-opp" not in fn
    html = _render([_league()])
    assert "pf-live-lbl" not in html
    assert "pf-live-wp-lbls" not in html and "pf-live-wp-opp" not in html
    assert ".pf-live-head{" in html and ".pf-live-you{" in html
    # The bar track is the mock's pale pink (#fecaca), not neutral.
    track = html.split(".pf-live-wp-track{", 1)[1].split("}", 1)[0]
    assert "#fecaca" in track


def test_pregame_strip_shows_projection_never_fake_zero():
    client = (ROOT / "static" / "app.js").read_text()
    fn = client.split("function matchupHtml(data)")[1].split("function renderMatchup")[0]
    # Pregame big number is the projection; the score is only read for
    # live / final states.
    assert "status === 'pre' ? number(team.proj, 1)" in fn
    assert "number(team.score" in fn


# ── strongest_room: the Best column's data source ─────────────────────
# It must be the exact mirror of weakest_room over the same rank dict
# the card already carries (pos_user_rank from rank_rosters_by_position).

def test_strongest_room_mirrors_weakest_over_same_dict():
    from dashboard_services.portfolio_board import strongest_room, weakest_room
    ranks = {"QB": 5, "RB": 3, "WR": 1, "TE": 8}
    assert strongest_room(ranks, 10) == {"pos": "WR", "rank": 1, "total": 10}
    assert weakest_room(ranks, 10) == {"pos": "TE", "rank": 8, "total": 10}


def test_strongest_room_skips_missing_and_bad_ranks():
    from dashboard_services.portfolio_board import strongest_room
    assert strongest_room({"QB": None, "RB": "?", "WR": 0, "TE": 4}, 12) == \
        {"pos": "TE", "rank": 4, "total": 12}
    assert strongest_room({}, 10) is None
    assert strongest_room(None, 10) is None


def test_strongest_room_ties_break_toward_earlier_position():
    from dashboard_services.portfolio_board import strongest_room
    assert strongest_room({"QB": 2, "RB": 2, "WR": 5, "TE": 9}, 10)["pos"] == "QB"
