"""Guards the deterministic 'game of the week' pick in the weekly recap.

The AI only narrates the pick; Python chooses which game leads by combining six
normalized signals (playoff stakes, projected closeness, all-play strength, a
rivalry angle, star-availability drama, momentum) and records *why* it won.
These tests pin that scoring and the reason it surfaces.

weekly_recap imports pandas at module load, so the whole module is skipped in
the base suite when pandas isn't installed.
"""
import pytest

pd = pytest.importorskip("pandas")

from dashboard_services.ai.weekly_recap import (  # noqa: E402
    _all_play_strength,
    _build_next_week_preview,
    _proj_win_prob,
    _prior_series,
    _regular_stakes,
    _render_next_week_html,
    _starter_flags,
)


def _fin_row(week, mid, rid, pts, opp):
    return {"week": week, "matchup_id": mid, "roster_id": rid,
            "points": pts, "points_against": opp, "finalized": True}


def _storyline(rid, team, rank, record="4-2", streak="W1"):
    return {"rid": str(rid), "team": team, "rank_after": rank,
            "record_after": record, "streak": streak}


def _starter(pid, name, pos="RB", nfl="KC"):
    return {"pid": pid, "name": name, "pos": pos, "nfl": nfl}


def _matchup(rid_a, rid_b, starters_a=None, starters_b=None):
    return {
        "left": {"roster_id": str(rid_a), "starters": starters_a or []},
        "right": {"roster_id": str(rid_b), "starters": starters_b or []},
    }


def _nctx(matchups, **kw):
    base = {"matchups": matchups, "proj_by_pid": {}, "player_index": {},
            "value_by_pid": {}, "playing_teams": set(), "is_playoff": False,
            "playoff_round_label": "Playoff game", "as_of": "Tue Nov 04"}
    base.update(kw)
    return base


# ── pure helpers ───────────────────────────────────────────────────────────────

def test_prior_series_counts_only_meetings_between_the_two():
    df = pd.DataFrame([
        _fin_row(1, 10, "1", 120, 100), _fin_row(1, 10, "2", 100, 120),  # 1 beat 2
        _fin_row(2, 11, "1", 90, 110), _fin_row(2, 11, "3", 110, 90),    # unrelated
        _fin_row(3, 12, "1", 105, 130), _fin_row(3, 12, "2", 130, 105),  # 2 beat 1
    ])
    assert _prior_series(df, "1", "2", upto_week=5) == (2, 1, 1)


def test_all_play_strength_rewards_high_scoring_regardless_of_record():
    # Team 3 scores highest every week -> highest all-play pct even if unlucky h2h.
    df = pd.DataFrame([
        _fin_row(1, 1, "1", 100, 0), _fin_row(1, 1, "2", 90, 0), _fin_row(1, 2, "3", 130, 0),
        _fin_row(2, 3, "1", 95, 0), _fin_row(2, 3, "2", 85, 0), _fin_row(2, 4, "3", 140, 0),
    ])
    strength = _all_play_strength(df, upto_week=2)
    assert strength["3"] > strength["1"] > strength["2"]
    assert strength["3"] == pytest.approx(1.0)


def test_proj_win_prob_is_a_coinflip_for_equal_projections():
    a = [_starter("a1", "A1"), _starter("a2", "A2")]
    b = [_starter("b1", "B1"), _starter("b2", "B2")]
    proj = {"a1": 15, "a2": 15, "b1": 15, "b2": 15}
    assert _proj_win_prob(a, b, proj) == pytest.approx(0.5, abs=1e-6)
    # Heavier projected team is favored.
    proj_lopsided = {"a1": 30, "a2": 30, "b1": 5, "b2": 5}
    assert _proj_win_prob(a, b, proj_lopsided) > 0.8


def test_starter_flags_split_out_questionable_and_bye_with_impact():
    starters = [
        _starter("p1", "Out Star", nfl="KC"),
        _starter("p2", "Q Guy", nfl="KC"),
        _starter("p3", "Bye Guy", nfl="DET"),   # DET not playing next week
        _starter("p4", "Healthy", nfl="KC"),
    ]
    pidx = {"p1": {"injury_status": "OUT"}, "p2": {"injury_status": "QUESTIONABLE"}}
    proj = {"p1": 20.0, "p2": 10.0, "p3": 12.0, "p4": 18.0}
    out, maybe, byes, risk = _starter_flags(
        {"starters": starters}, pidx, proj, {}, playing_teams={"KC"},
    )
    assert [p["name"] for p in out] == ["Out Star"]
    assert [p["name"] for p in maybe] == ["Q Guy"]
    assert [p["name"] for p in byes] == ["Bye Guy"]
    # risk = 20 (out) + 10*0.5 (q) + 12*0.5 (bye) = 31
    assert risk == pytest.approx(31.0)


def test_regular_stakes_flags_a_win_and_in_finale():
    # Both teams sit right on the cutline (6) in the final regular week.
    val, label = _regular_stakes(6, 7, playoff_teams=6, weeks_left_after=0, has_bye_seed=True)
    assert label == "Win-and-in"
    assert val >= 0.9


# ── selection ──────────────────────────────────────────────────────────────────

def test_playoff_bubble_decider_beats_a_higher_ranked_blowout():
    # Game A: #1 (proj blowout) vs #9. Game B: two bubble teams (#6, #7), a coin
    # flip, in the last regular week -> the bubble game must be game of the week.
    df = pd.DataFrame([
        _fin_row(1, 1, "1", 150, 80), _fin_row(1, 1, "9", 80, 150),
        _fin_row(1, 2, "6", 100, 99), _fin_row(1, 2, "7", 99, 100),
    ])
    storylines = {
        "1": _storyline(1, "Alpha", 1), "9": _storyline(9, "India", 9),
        "6": _storyline(6, "Foxtrot", 6), "7": _storyline(7, "Golf", 7),
    }
    a = [_starter("a1", "A1"), _starter("a2", "A2")]
    nine = [_starter("i1", "I1"), _starter("i2", "I2")]
    f = [_starter("f1", "F1"), _starter("f2", "F2")]
    g = [_starter("g1", "G1"), _starter("g2", "G2")]
    proj = {"a1": 40, "a2": 40, "i1": 5, "i2": 5,   # lopsided
            "f1": 15, "f2": 15, "g1": 15, "g2": 15}  # coin flip
    nctx = _nctx([_matchup("1", "9", a, nine), _matchup("6", "7", f, g)], proj_by_pid=proj)
    preview = _build_next_week_preview(
        df, storylines, selected_week=12, playoff_start=14, playoff_teams=6, num_teams=10, nctx=nctx,
    )
    got = preview["game_of_the_week"]
    assert {got["team_a"], got["team_b"]} == {"Foxtrot", "Golf"}
    assert got["why"] in ("Win-and-in", "Playoff bubble decider")


def test_star_watch_surfaces_the_missing_player_as_the_why():
    df = pd.DataFrame([_fin_row(1, 1, "1", 100, 90), _fin_row(1, 1, "2", 90, 100)])
    storylines = {"1": _storyline(1, "Alpha", 5), "2": _storyline(2, "Bravo", 6)}
    b = [_starter("b1", "Bijan", nfl="ATL"), _starter("b2", "B2", nfl="ATL")]
    pidx = {"b1": {"injury_status": "OUT"}}
    proj = {"b1": 21.0, "b2": 12.0}
    nctx = _nctx([_matchup("1", "2", starters_b=b)], proj_by_pid=proj, player_index=pidx,
                 playing_teams={"ATL"})
    preview = _build_next_week_preview(
        df, storylines, selected_week=3, playoff_start=14, playoff_teams=6, num_teams=10, nctx=nctx,
    )
    got = preview["game_of_the_week"]
    assert got["why"] == "Star on the shelf: Bijan (OUT)"
    assert got["out_b"] and got["out_b"][0]["proj"] == 21.0


def test_playoff_week_uses_the_round_label_as_the_why():
    df = pd.DataFrame([_fin_row(13, 1, "1", 120, 80), _fin_row(13, 1, "2", 80, 120)])
    storylines = {"1": _storyline(1, "Alpha", 1), "2": _storyline(2, "Bravo", 4)}
    nctx = _nctx([_matchup("1", "2")], is_playoff=True, playoff_round_label="Semifinal")
    preview = _build_next_week_preview(
        df, storylines, selected_week=14, playoff_start=14, playoff_teams=6, num_teams=10, nctx=nctx,
    )
    assert preview["is_playoff"] is True
    assert preview["game_of_the_week"]["why"] == "Semifinal"


def test_render_shows_matchup_availability_and_blurb():
    # The card is a banner header + matchup + AI blurb + availability chips.
    # (The standalone WHY badge was removed in the #695 polish pass; the reason
    # now lives in the blurb, so this asserts the surviving structure.)
    df = pd.DataFrame([_fin_row(1, 1, "1", 100, 90), _fin_row(1, 1, "2", 90, 100)])
    storylines = {"1": _storyline(1, "Alpha", 1), "2": _storyline(2, "Bravo", 2)}
    b = [_starter("b1", "Bijan", nfl="ATL")]
    nctx = _nctx([_matchup("1", "2", starters_b=b)], player_index={"b1": {"injury_status": "OUT"}},
                 proj_by_pid={"b1": 20.0}, playing_teams={"ATL"})
    preview = _build_next_week_preview(
        df, storylines, selected_week=6, playoff_start=14, playoff_teams=6, num_teams=10, nctx=nctx,
    )
    out = _render_next_week_html(preview, "Top two teams in the league, going at it.")
    assert "Game of the Week" in out                      # banner title
    assert "Alpha" in out and "Bravo" in out              # the matchup sides
    assert "Bijan (OUT)" in out                           # availability chip
    assert "availability as of Tue Nov 04" in out         # freshness stamp
    assert "Top two teams in the league, going at it." in out   # AI blurb


def test_no_matchups_yields_no_preview():
    df = pd.DataFrame([_fin_row(1, 1, "1", 100, 90)])
    assert _build_next_week_preview(df, {}, 6, 14, 6, 10, _nctx([])) is None
    assert _render_next_week_html(None, "") == ""
