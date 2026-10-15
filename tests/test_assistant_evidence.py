"""Evidence builder tests (fix 4): packs assemble the site's real internals.

Everything external is monkeypatched at the evidence module's lazy shims:
weekly projections + schedule for start/sit, the Sleeper week-stat cache for
game logs, ctx rosters for trade partner fit / balancing, and Sleeper
trending adds + the FAAB machinery for waivers.
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import evidence

INDEX = {
    "1": {"name": "Rome Odunze", "team": "CHI", "position": "WR"},
    "2": {"name": "Darnell Mooney", "team": "ATL", "position": "WR"},
    "9": {"name": "Woody Marks", "team": "HOU", "position": "RB"},
    "10": {"name": "Dameon Pierce", "team": "HOU", "position": "RB"},
}

VALUES = [
    {"id": "1", "value": 4200}, {"id": "2", "value": 2600},
    {"id": "9", "value": 900}, {"id": "10", "value": 300},
]


@pytest.fixture()
def base(monkeypatch):
    monkeypatch.setattr(evidence, "get_players_index_global", lambda *a, **k: dict(INDEX))
    monkeypatch.setattr(evidence, "get_model_value_table_cached", lambda *a, **k: list(VALUES))
    monkeypatch.setattr(evidence, "_metrics_for",
                        lambda pid, season=None: ({"ppg": 14.0, "expected_ppr_per_game": 13.5,
                                                   "snap_share": 0.88}, ["fake_metrics"]))
    return monkeypatch


def _bundle(**over):
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "Blackedraw", "team_name": "Couch", "week": 7,
              "ctx": {"raw_scoring_settings": {"rec": 1.0, "rec_yd": 0.1, "rec_td": 6.0},
                      "rosters": [], "roster_positions": []},
              "roster": {"players": [], "starters": []}, "roster_player_ids": set()}
    bundle.update(over)
    return bundle


# -- start / sit -----------------------------------------------------------------

def test_start_sit_uses_week_projections_schedule_and_def_rank(base, monkeypatch):
    week_stats = {
        6: {"1": {"rec": 6, "rec_yd": 90, "rec_td": 1}, "2": {"rec": 3, "rec_yd": 40, "rec_td": 0}},
        5: {"1": {"rec": 5, "rec_yd": 70, "rec_td": 0}, "2": {"rec": 4, "rec_yd": 55, "rec_td": 1}},
    }
    monkeypatch.setattr(evidence, "load_week_projection",
                        lambda season, week: {"1": {"proj": 1}, "2": {"proj": 1}})
    monkeypatch.setattr("utils.fantasy_scoring.weekly_projection_points",
                        lambda week_map, pid, scoring, pos: {"1": 18.4, "2": 10.2}.get(str(pid)))
    monkeypatch.setattr(evidence, "load_week_sched",
                        lambda season, week: [{"home": "CHI", "away": "ATL"}] if week == 7
                        else [{"home": "DET", "away": "CHI"}, {"home": "ATL", "away": "GB"}])
    monkeypatch.setattr(evidence, "matchup_rank_table", lambda season, pos: ({"ATL": 27, "DET": 9}, 32))
    monkeypatch.setattr(evidence, "load_sleeper_week_stats",
                        lambda season, week: week_stats.get(week, {}))

    pack, tools = evidence.build_start_sit(_bundle(), {}, ["1", "2"])
    odunze, mooney = pack["players"]
    # Projection comes from the week projection map, and the engine scored it.
    assert odunze["projection"] == 18.4 and mooney["projection"] == 10.2
    assert odunze["start_score"] is not None and mooney["start_score"] is not None
    assert pack["pick_player_id"] == "1"
    # Schedule evidence: CHI hosts ATL in week 7.
    assert odunze["opponent"] == "ATL" and odunze["matchup"] == "vs ATL"
    assert mooney["opponent"] == "CHI" and mooney["matchup"] == "@ CHI"
    assert odunze["def_rank_vs_position"] == 27 and odunze["def_rank_total"] == 32
    # Game-log form is attached and averaged (week 6: 6+9+6=21, week 5: 5+7=12).
    assert [f["points"] for f in odunze["recent_form"]] == [12.0, 21.0]
    assert odunze["recent_ppg"] == 16.5
    assert "utils.start_sit.compute_start_score" in tools


def test_player_pack_has_game_logs_and_upcoming_schedule(base, monkeypatch):
    week_stats = {6: {"1": {"rec": 6, "rec_yd": 90, "rec_td": 1}}}
    sched = {7: [{"home": "CHI", "away": "ATL"}], 8: [{"home": "GB", "away": "DET"}],
             9: [{"home": "MIN", "away": "CHI"}]}
    monkeypatch.setattr(evidence, "load_sleeper_week_stats",
                        lambda season, week: week_stats.get(week, {}))
    monkeypatch.setattr(evidence, "load_week_sched", lambda season, week: sched.get(week, []))
    monkeypatch.setattr("dashboard_services.news.get_player_news", lambda name, limit=3: {})

    pack, tools = evidence.build_player(_bundle(), {}, ["1"])
    assert pack["recent_form"] == [{"week": 6, "points": 21.0, "opponent": None, "matchup": None}]
    upcoming = pack["upcoming_schedule"]
    assert upcoming[0] == {"week": 7, "opponent": "ATL", "matchup": "vs ATL", "bye": False}
    assert upcoming[1]["bye"] is True  # CHI absent from the loaded week 8 schedule
    assert upcoming[2]["matchup"] == "@ MIN"


# -- trade evaluation --------------------------------------------------------------

TRADE_INDEX = {
    "q1": {"name": "My QB", "team": "BUF", "position": "QB"},
    "r1": {"name": "My RB", "team": "MIA", "position": "RB"},
    "w1": {"name": "My WR", "team": "CIN", "position": "WR"},
    "t1": {"name": "My TE", "team": "KC", "position": "TE"},
    "q2": {"name": "His QB", "team": "PHI", "position": "QB"},
    "r2": {"name": "His RB", "team": "SF", "position": "RB"},
    "r3": {"name": "His RB2", "team": "DAL", "position": "RB"},
    "w2": {"name": "His WR", "team": "NYJ", "position": "WR"},
    "t2": {"name": "His TE", "team": "BAL", "position": "TE"},
    "x1": {"name": "His Bench WR", "team": "SEA", "position": "WR"},
}
TRADE_VALUES = [
    {"id": "q1", "value": 3000}, {"id": "r1", "value": 3500}, {"id": "w1", "value": 4000},
    {"id": "t1", "value": 2500}, {"id": "q2", "value": 3200}, {"id": "r2", "value": 3300},
    {"id": "r3", "value": 2900}, {"id": "w2", "value": 3900}, {"id": "t2", "value": 2400},
    {"id": "x1", "value": 1050},
]


def test_trade_eval_partner_fit_and_balancing_asset(base, monkeypatch):
    monkeypatch.setattr(evidence, "get_players_index_global", lambda *a, **k: dict(TRADE_INDEX))
    monkeypatch.setattr(evidence, "get_model_value_table_cached", lambda *a, **k: list(TRADE_VALUES))
    ctx = {
        "rosters": [
            {"roster_id": 1, "owner_id": "me", "players": ["q1", "r1", "w1", "t1"],
             "metadata": {"team_name": "My Team"}},
            {"roster_id": 2, "owner_id": "him", "players": ["q2", "r2", "r3", "w2", "t2", "x1"],
             "metadata": {"team_name": "Other Guys"}},
        ],
        "users": [{"user_id": "me", "display_name": "Me"}, {"user_id": "him", "display_name": "Him"}],
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
    }
    bundle = _bundle(ctx=ctx, roster={"roster_id": 1, "players": ["q1", "r1", "w1", "t1"]})
    entities = {"give_player_ids": ["w1"], "receive_player_ids": ["r2"],
                "other_team_hint": "Other Guys"}
    pack, tools = evidence.build_trade_eval(bundle, entities, ["w1", "r2"])
    assert pack["fairness_gap"] == round(3300 - 4000, 1)
    assert pack["partner"]["team_name"] == "Other Guys"
    assert "needed_positions" in pack["partner"] and "surplus_positions" in pack["partner"]
    assert pack["roster_fit"]["surplus_positions"] is not None
    # I give 700 more than I get: the partner's bench WR (1050) is the real
    # piece nearest the gap on his roster.
    balancing = pack["balancing_asset"]
    assert balancing["side"] == "partner_adds"
    assert balancing["player"]["player_id"] == "x1"
    assert balancing["closes_gap_to"] == round(-700 + 1050, 1)


# -- waivers -----------------------------------------------------------------------

def test_waivers_trending_and_faab_wired(base, monkeypatch):
    monkeypatch.setattr(evidence, "sleeper_trending_adds",
                        lambda limit=25: [{"player_id": "9", "count": 54321}])
    monkeypatch.setattr(evidence, "waiver_faab_context",
                        lambda ctx: {"enabled": True, "total": 100,
                                     "waiver_type": "faab", "phase": "mid"})
    monkeypatch.setattr("routes.waiver_bp.load_do_not_drop", lambda *a, **k: set())
    ctx = {"rosters": [{"roster_id": 1, "players": ["10"]}], "roster_positions": [],
           "raw_scoring_settings": {}}
    bundle = _bundle(ctx=ctx, roster={"roster_id": 1, "players": ["10"]},
                     roster_player_ids={"10"})
    pack, tools = evidence.build_waivers(bundle, {})
    marks = next(c for c in pack["candidates"] if c["player_id"] == "9")
    assert marks["value"] == 900  # free agent 900 clears worst-RB bar 300 x 1.05
    assert marks["trending_adds_48h"] == 54321
    assert pack["trending_free_agents"][0]["player_id"] == "9"
    assert pack["faab"]["enabled"] is True and pack["faab"]["budget_total"] == 100
    assert "pct_target" in marks["faab"]
    assert "waiver_score.faab_recommendation" in tools


def test_trade_eval_pack_carries_analyzer_band_verdict(base, monkeypatch):
    # te2's shape: give 7770, receive 4680. The shared fair band is
    # max(7% of 7770, 25) = 543.9, so a 3090 loss is a code-set DECLINE.
    monkeypatch.setattr(evidence, "get_players_index_global", lambda *a, **k: {
        "g1": {"name": "Brian Thomas Jr.", "team": "JAX", "position": "WR"},
        "g2": {"name": "Woody Marks", "team": "HOU", "position": "RB"},
        "r1": {"name": "Zay Flowers", "team": "BAL", "position": "WR"},
    })
    monkeypatch.setattr(evidence, "get_model_value_table_cached", lambda *a, **k: [
        {"id": "g1", "value": 5120}, {"id": "g2", "value": 2650}, {"id": "r1", "value": 4680},
    ])
    entities = {"give_player_ids": ["g1", "g2"], "receive_player_ids": ["r1"]}
    pack, tools = evidence.build_trade_eval(_bundle(), entities, ["g1", "g2", "r1"])
    assert pack["fairness_gap"] == -3090.0
    assert pack["trade_verdict"] == "DECLINE"
    assert "utils.trade_value.fairness_label" in tools


# -- this week: the dashboard's Next Steps queue is the source ---------------------

def test_this_week_actions_come_from_dashboard_next_steps_queue(base, monkeypatch):
    # In the queue's own ranked order (the shim sorts by score, like the site).
    queue = [
        {"priority": "high", "tag": "Lineup", "kind": "lineup",
         "action": "Start Rome Odunze over Jordan Addison",
         "why": "Odunze projects 15.7 vs Addison's 13.8.",
         "impact": "+1.9 projected points", "cta_label": "Fix lineup",
         "cta_url": "/waivers?tab=startsit", "score": 101.9},
        {"priority": "med", "tag": "Waivers", "kind": "waiver",
         "action": "Claim Rashid Shaheed", "why": "Shaheed clears your gate.",
         "impact": None, "cta_label": "Claim now", "cta_url": "/waivers?player=9",
         "score": 55},
    ]
    monkeypatch.setattr(evidence, "next_steps_actions", lambda ctx, rid, rows: list(queue))
    bundle = _bundle(ctx={"rosters": []}, roster={"roster_id": 1, "players": ["1"]})
    pack, tools = evidence.build_this_week(bundle, {}, [])
    assert "app.next_steps_queue" in tools
    assert [a["headline"] for a in pack["actions"]] == [
        "Start Rome Odunze over Jordan Addison", "Claim Rashid Shaheed"]
    swap = pack["actions"][0]
    assert swap["tag"] == "Lineup"
    assert swap["detail"] == "Odunze projects 15.7 vs Addison's 13.8."
    assert swap["key_number"] == "+1.9 projected points"
    assert swap["act_link"] == "/waivers?tab=startsit"
    assert swap["cta_label"] == "Fix lineup"


def test_this_week_without_roster_id_gives_empty_actions(base, monkeypatch):
    monkeypatch.setattr(evidence, "next_steps_actions",
                        lambda ctx, rid, rows: (_ for _ in ()).throw(AssertionError("must not be called")))
    pack, tools = evidence.build_this_week(_bundle(roster={}), {}, [])
    assert pack["actions"] == []
    assert pack["available"] is True


def test_waivers_no_faab_when_league_does_not_use_it(base, monkeypatch):
    monkeypatch.setattr(evidence, "sleeper_trending_adds", lambda limit=25: [])
    monkeypatch.setattr(evidence, "waiver_faab_context",
                        lambda ctx: {"enabled": False, "total": None,
                                     "waiver_type": "priority", "phase": "mid"})
    monkeypatch.setattr("routes.waiver_bp.load_do_not_drop", lambda *a, **k: set())
    ctx = {"rosters": [{"roster_id": 1, "players": ["10"]}], "roster_positions": [],
           "raw_scoring_settings": {}}
    bundle = _bundle(ctx=ctx, roster={"roster_id": 1, "players": ["10"]},
                     roster_player_ids={"10"})
    pack, tools = evidence.build_waivers(bundle, {})
    assert "faab" not in pack
    assert all("faab" not in c for c in pack["candidates"])
