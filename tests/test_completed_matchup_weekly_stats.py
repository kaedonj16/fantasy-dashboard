from dashboard_services import matchups as mmod


def test_position_stat_lines_include_requested_weekly_volume():
    stats = {
        "BUF": {
            "QB": {"josh allen": {"pass_cmp": 20, "pass_att": 30, "pass_yds": 250, "pass_td": 2, "int": 1, "rush_yds": 35, "rush_td": 1}},
            "RB": {"james cook": {"rush_att": 15, "rush_yds": 70, "rush_td": 1, "rec": 4, "tgt": 5, "rec_yds": 28, "rec_td": 0}},
            "WR": {"keon coleman": {"rec": 3, "tgt": 7, "rec_yds": 51, "rec_td": 1}},
        }
    }
    qb = mmod.format_player_stats(stats, "BUF", "QB", "Josh Allen")
    rb = mmod.format_player_stats(stats, "BUF", "RB", "James Cook")
    wr = mmod.format_player_stats(stats, "BUF", "WR", "Keon Coleman")
    # Grouped shorthand: the PASS/RUSH/REC label carries the stat type, so
    # yards and TDs inside a group need no qualifier. A QB rushing line with
    # no recorded attempts shows yards only.
    assert qb == "PASS 20/30 250 yds, 2 TD, 1 INT • RUSH 35 yds, 1 TD"
    assert rb == "RUSH 15 70 yds, 1 TD • REC 4/5 28 yds"
    assert wr == "REC 3/7 51 yds, 1 TD"
    # rec_td 0 is dropped, not shown as "0 TD".
    assert "0 TD" not in rb


def test_stat_lines_match_the_approved_examples():
    stats = {
        "BUF": {
            "QB": {"qb one": {"pass_cmp": 30, "pass_att": 55, "pass_yds": 390, "pass_td": 2, "int": 2, "rush_att": 2, "rush_yds": 13, "rush_td": 0}},
            "RB": {"rb one": {"rush_att": 15, "rush_yds": 75, "rush_td": 1, "rec": 4, "tgt": 5, "rec_yds": 41, "rec_td": 0}},
        }
    }
    assert (
        mmod.format_player_stats(stats, "BUF", "QB", "QB One")
        == "PASS 30/55 390 yds, 2 TD, 2 INT • RUSH 2 13 yds"
    )
    assert (
        mmod.format_player_stats(stats, "BUF", "RB", "RB One")
        == "RUSH 15 75 yds, 1 TD • REC 4/5 41 yds"
    )


def test_stat_lines_use_space_not_dash_before_yards():
    stats = {"BUF": {"WR": {"busy wr": {"rec": 9, "tgt": 12, "rec_yds": 98, "rec_td": 1}}}}
    assert (
        mmod.format_player_stats(stats, "BUF", "WR", "Busy WR")
        == "REC 9/12 98 yds, 1 TD"
    )
    qb_stats = {"BUF": {"QB": {"josh allen": {
        "pass_cmp": 30, "pass_att": 55, "pass_yds": 390, "pass_td": 2,
        "int": 2, "rush_att": 2, "rush_yds": 13}}}}
    assert (
        mmod.format_player_stats(qb_stats, "BUF", "QB", "Josh Allen")
        == "PASS 30/55 390 yds, 2 TD, 2 INT • RUSH 2 13 yds"
    )
    rb_stats = {"BUF": {"RB": {"james cook": {
        "rush_att": 15, "rush_yds": 75, "rush_td": 1,
        "rec": 4, "tgt": 5, "rec_yds": 41}}}}
    assert (
        mmod.format_player_stats(rb_stats, "BUF", "RB", "James Cook")
        == "RUSH 15 75 yds, 1 TD • REC 4/5 41 yds"
    )


def test_stat_line_zero_rules():
    stats = {
        "BUF": {
            # No TD anywhere: no TD segment at all.
            "RB": {"quiet rb": {"rush_att": 12, "rush_yds": 48, "rush_td": 0, "rec": 3, "tgt": 5, "rec_yds": 29, "rec_td": 0}},
            # Targeted but shut out: the 0-catch line stays visible.
            "WR": {"blanked wr": {"rec": 0, "tgt": 3, "rec_yds": 0, "rec_td": 0}},
        }
    }
    quiet = mmod.format_player_stats(stats, "BUF", "RB", "Quiet RB")
    assert quiet == "RUSH 12 48 yds • REC 3/5 29 yds"
    assert "TD" not in quiet
    assert mmod.format_player_stats(stats, "BUF", "WR", "Blanked WR") == "REC 0/3 0 yds"


def test_stat_line_rush_group_comes_first_even_for_wr():
    stats = {"BUF": {"WR": {"gadget wr": {"rush_att": 2, "rush_yds": 18, "rec": 5, "tgt": 6, "rec_yds": 60}}}}
    assert (
        mmod.format_player_stats(stats, "BUF", "WR", "Gadget WR")
        == "RUSH 2 18 yds • REC 5/6 60 yds"
    )


def test_completed_week_renders_starter_stats_but_not_bench(monkeypatch):
    weekly = {"BUF": {"QB": {"starter qb": {"pass_yds": 200}}, "WR": {"bench wr": {"rec": 2, "tgt": 4, "rec_yds": 20}}}}
    monkeypatch.setattr(mmod, "load_teams_index", lambda: {})
    monkeypatch.setattr(mmod, "build_offense_rankings", lambda *_: {})
    monkeypatch.setattr(mmod, "load_week_stats", lambda season, week: weekly)
    monkeypatch.setattr(mmod, "load_week_schedule", lambda *_: [])
    monkeypatch.setattr(mmod, "build_team_schedule_lookup", lambda *_: {})
    matchup = {
        "left": {"name": "Left", "roster_id": "1", "starters": [{"pid": "1", "name": "Starter QB", "pos": "QB", "nfl": "BUF", "pts": 12.0}], "bench": [{"pid": "2", "name": "Bench WR", "pos": "WR", "nfl": "BUF", "pts": 0.0}], "pts_total": 12.0},
        "right": {"name": "Right", "roster_id": "2", "starters": [], "bench": [], "pts_total": 0.0},
    }
    html = mmod.render_matchup_slide("2025", matchup, 2, 2, {}, {}, {}, {}, {})
    assert "Starter QB" in html and "PASS 200 yds" in html
    assert "Bench WR" not in html and "REC 2/4 20 yds" not in html
    assert "m-row--bench" not in html


def test_completed_week_keeps_canonical_stats_when_schedule_is_final(monkeypatch):
    weekly = {"BUF": {"QB": {"starter qb": {"pass_yds": 200}}}}
    monkeypatch.setattr(mmod, "load_teams_index", lambda: {})
    monkeypatch.setattr(mmod, "build_offense_rankings", lambda *_: {})
    monkeypatch.setattr(mmod, "load_week_stats", lambda *_: weekly)
    monkeypatch.setattr(mmod, "load_week_schedule", lambda *_: [{
        "teamAbv": "BUF", "opponent": "MIA", "gameStatusCode": "2",
        "gameDate": "20250907",
    }])
    monkeypatch.setattr(mmod, "build_team_schedule_lookup", lambda rows: {"BUF": rows[0]})
    matchup = {
        "left": {"name": "Left", "roster_id": "1", "starters": [{"pid": "1", "name": "Starter QB", "pos": "QB", "nfl": "BUF", "pts": 12.0}], "pts_total": 12.0},
        "right": {"name": "Right", "roster_id": "2", "starters": [], "pts_total": 0.0},
    }
    rendered = mmod.render_matchup_slide("2025", matchup, 1, 1, {}, {}, {}, {}, {})
    assert "PASS 200 yds" in rendered


def test_shared_stat_resolver_handles_suffix_nickname_and_historical_team():
    stats = {"SEA": {"RB": {"ken walker": {"rush_att": 18, "rush_yds": 91}}},
             "NE": {"WR": {"stefon diggs": {"rec": 6, "tgt": 8, "rec_yds": 74}}}}
    assert mmod.format_player_stats(stats, "NYG", "RB", "Kenneth Walker III") == "RUSH 18 91 yds"
    assert mmod.format_player_stats(stats, "BUF", "WR", "Stefon Diggs") == "REC 6/8 74 yds"
