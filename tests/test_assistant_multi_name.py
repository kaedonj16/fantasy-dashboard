"""Multi-name coverage tests (the compare-collapse family audit).

Single-subject builders used to read player_ids[0] and drop every
other resolved name without a trace in the pack, so a two-name
question answered for the first name alone and read as complete.
These tests pin the fix: the extra names ride in the pack
(also_named / player_ranks / unanswered_names), the composers name
them, and single-name packs are unchanged.
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant import evidence
from assistant_eval.mock_composer import compose_mock

INDEX = {
    "w1": {"name": "Garrett Wilson", "team": "NYJ", "position": "WR", "age": 26},
    "w2": {"name": "Nico Collins", "team": "HOU", "position": "WR", "age": 27},
    "h1": {"name": "Derrick Henry", "team": "TEN", "position": "RB", "age": 32},
    "h2": {"name": "Tyjae Spears", "team": "TEN", "position": "RB", "age": 25},
}
VALUES = [{"id": "w1", "value": 316.5}, {"id": "w2", "value": 305.0},
          {"id": "h1", "value": 298.0}, {"id": "h2", "value": 89.0}]
# Raw division output for PPG / target share (the lite profile rounds
# those itself), and snap_share ALREADY on the percent scale: that is
# the real _metrics_for output contract since the 2026-10-16 share fix.
# The stub used to fake the pre-fix 0-1 fraction here, which kept a
# stale second rescale alive in _lite_named that printed a genuine
# 1.2% snap player as 120 (2026-10-22 audit).
RAW_METRICS = {"ppg": 17.279999999999998, "target_share": 30.24,
               "snap_share": 90, "games": 6}

CTX = {
    "rosters": [
        {"roster_id": 1, "owner_id": "me", "players": ["w1"],
         "metadata": {"team_name": "My Team"}},
        {"roster_id": 2, "owner_id": "him", "players": ["w2", "h1", "h2"],
         "metadata": {"team_name": "Other Guys"}},
    ],
    "users": [{"user_id": "me", "display_name": "Me"},
              {"user_id": "him", "display_name": "Him"}],
    "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
}


def _bundle(**over):
    b = {"platform": "sleeper", "league_id": "L1", "season": 2026,
         "league_name": "Blackedraw", "team_name": "My Team", "week": 7,
         "ctx": CTX, "roster": {"roster_id": 1, "players": ["w1"]},
         "viewer": {"viewer_username": "kaedon", "viewer_user_id": "u1"},
         "roster_player_ids": {"w1"}}
    b.update(over)
    return b


@pytest.fixture()
def base(monkeypatch):
    monkeypatch.setattr(evidence, "get_players_index_global",
                        lambda *a, **k: dict(INDEX))
    monkeypatch.setattr(evidence, "get_model_value_table_cached",
                        lambda *a, **k: list(VALUES))
    monkeypatch.setattr(evidence, "_metrics_for",
                        lambda pid, season=None: (dict(RAW_METRICS), ["stub"]))
    monkeypatch.setattr(evidence, "load_sleeper_week_stats", lambda *a, **k: {})
    monkeypatch.setattr(evidence, "load_week_sched", lambda *a, **k: [])
    import data_building.advanced_metrics as am
    monkeypatch.setattr(am, "get_player_metric_ranks",
                        lambda pid, season=None: {})
    import dashboard_services.news as news_mod
    monkeypatch.setattr(news_mod, "get_player_news", lambda name, limit=3: {})
    return monkeypatch


# -- builders: the second name is in the pack ---------------------------------

def test_player_pack_carries_also_named_normalized(base):
    pack, _tools = evidence.build_player(_bundle(), {}, ["w1", "w2"])
    assert pack["player"]["name"] == "Garrett Wilson"
    assert pack["also_named"] == [{
        "player_id": "w2", "name": "Nico Collins", "team": "HOU",
        "position": "WR", "value": 305.0, "ppg": 17.3,
        "target_share": 30.2, "snap_share": 90.0}]


def test_player_pack_single_name_has_no_also_named(base):
    pack, _tools = evidence.build_player(_bundle(), {}, ["w1"])
    assert "also_named" not in pack


def test_buy_sell_hold_pack_carries_also_named_with_trend(base, monkeypatch):
    import dashboard_services.player_value_history as pvh
    monkeypatch.setattr(
        pvh, "get_player_value_history",
        lambda pid, days=30: [{"value": 300.0}, {"value": 330.0}])
    pack, _tools = evidence.build_buy_sell_hold(_bundle(), {}, ["w1", "w2"])
    extra = pack["also_named"][0]
    assert extra["name"] == "Nico Collins"
    assert extra["delta_pct"] == 10.0
    assert extra["value"] == 305.0


def test_replacement_pack_records_unanswered_subject(base):
    pack, _tools = evidence.build_replacement(_bundle(), {}, ["w1", "w2"])
    assert pack["unanswered_names"] == ["Nico Collins"]


def test_handcuff_pack_records_unanswered_subject(base):
    pack, _tools = evidence.build_handcuff(_bundle(), {}, ["h1", "w2"])
    assert pack["subject"]["name"] == "Derrick Henry"
    assert pack["unanswered_names"] == ["Nico Collins"]


def test_breakout_history_pack_carries_also_named(base, monkeypatch):
    monkeypatch.setattr(evidence, "breakout_call_history_for",
                        lambda pid, season=None: None)
    monkeypatch.setattr(evidence, "breakout_board_candidates",
                        lambda season=None: {})
    pack, _tools = evidence.build_breakout_history(_bundle(), {}, ["w1", "w2"])
    assert pack["also_named"] == [{"player_id": "w2", "name": "Nico Collins",
                                   "was_call": False}]


def test_metric_pack_ranks_every_named_player(base, monkeypatch):
    import data_building.advanced_metrics as am

    def _row(pid, name, value):
        return {"player_id": pid, "name": name, "team": "T", "position": "WR",
                "value": value, "games": 6}

    rows = [_row("x1", "Top Guy", 0.32), _row("w1", "Garrett Wilson", 0.302),
            _row("x3", "Third", 0.29), _row("x4", "Fourth", 0.28),
            _row("x5", "Fifth", 0.27), _row("x6", "Sixth", 0.26),
            _row("x7", "Seventh", 0.25), _row("w2", "Nico Collins", 0.24)]
    monkeypatch.setattr(am, "get_metric_leaderboard",
                        lambda metric, **k: list(rows))
    monkeypatch.setattr(evidence, "metric_leaderboard",
                        lambda metric, position=None, limit=500, season=None,
                        min_vol=None, combine=False: list(rows))
    monkeypatch.setattr(evidence, "_metric_config", lambda: {
        "target_share": {"label": "Target Share", "desc": "Share.",
                         "positions": ["WR", "TE", "RB"],
                         "min_vol": {"col": "games", "opts": [4],
                                     "label": "Min Games"}}})

    class _Policy:
        games_min = 4
        provisional = False

        def minimum(self, col, full):
            return full

    monkeypatch.setattr(evidence, "qualification_policy_for",
                        lambda season, week_start=None, week_end=None: _Policy())
    monkeypatch.setattr(evidence, "pro_metric_keys", lambda: set())
    monkeypatch.setattr(evidence, "premium_metric_keys", lambda: set())
    monkeypatch.setattr(evidence, "premium_metrics_exposed", lambda: False)
    pack, _tools = evidence.build_metric(
        _bundle(), {"metric_hint": "target share"}, ["w1", "w2"])
    # Collins is rank 8, outside the carried top 5: his number used to
    # be absent from the pack entirely.
    assert [r["player_id"] for r in pack["player_ranks"]] == ["w1", "w2"]
    assert pack["player_ranks"][1]["rank"] == 8
    assert pack["player_ranks"][1]["value"] == 0.24
    assert pack["player_rank"]["player_id"] == "w1"


def test_trade_finding_two_named_players_records_second(monkeypatch):
    """Named mode runs for the first named player; the second is
    recorded, not dropped ('what can I get for Waddle and Olave?')."""
    idx = {
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
    vals = [{"id": k, "value": v} for k, v in
            [("q1", 3000), ("r1", 3500), ("w1", 4000), ("t1", 2500),
             ("q2", 3200), ("r2", 3300), ("r3", 2900), ("w2", 900),
             ("t2", 2400), ("x1", 1050)]]
    ctx = {
        "rosters": [
            {"roster_id": 1, "owner_id": "me",
             "players": ["q1", "r1", "w1", "t1"],
             "metadata": {"team_name": "My Team"}},
            {"roster_id": 2, "owner_id": "him",
             "players": ["q2", "r2", "r3", "w2", "t2", "x1"],
             "metadata": {"team_name": "Other Guys"}},
        ],
        "users": [{"user_id": "me", "display_name": "Me"},
                  {"user_id": "him", "display_name": "Him"}],
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
    }
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "Blackedraw", "team_name": "My Team", "week": 7,
              "ctx": ctx,
              "roster": {"roster_id": 1, "players": ["q1", "r1", "w1", "t1"]},
              "viewer": {"viewer_username": "kaedon", "viewer_user_id": "u1"},
              "roster_player_ids": {"q1", "r1", "w1", "t1"}}
    monkeypatch.setattr(evidence, "get_players_index_global",
                        lambda *a, **k: dict(idx))
    monkeypatch.setattr(evidence, "get_model_value_table_cached",
                        lambda *a, **k: list(vals))
    monkeypatch.setattr(evidence, "has_premium_for_viewer",
                        lambda *a, **k: True)

    def _asset(pid):
        row = next(v for v in vals if v["id"] == pid)
        return {"player_id": pid, "id": pid, "name": idx[pid]["name"],
                "position": idx[pid]["position"], "value": float(row["value"])}

    monkeypatch.setattr(
        evidence, "value_matched_acquire_packages",
        lambda focus, players, picks, **kw: [
            {"assets": [_asset("r3"), _asset("x1")], "send_value": 3950.0}])
    pack, _tools = evidence.build_trade_finding(bundle, {}, ["w1", "r1"])
    assert pack["mode"] == "sell"
    assert pack["sell_player"]["name"] == "My WR"
    assert pack["unanswered_names"] == ["My RB"]


def test_rookie_pack_records_unanswered_prospect(base, monkeypatch):
    rows = [
        {"player_id": "rk-a", "sleeper_id": "w1", "name": "Alpha Prospect",
         "position": "WR", "school": "State", "model_version": "v3.3"},
        {"player_id": "rk-b", "sleeper_id": "w2", "name": "Beta Prospect",
         "position": "WR", "school": "Tech", "model_version": "v3.3"},
    ]
    monkeypatch.setattr(evidence, "rookie_rankings", lambda year: list(rows))
    monkeypatch.setattr(evidence, "active_rookie_class", lambda: 2027)
    monkeypatch.setattr(evidence, "prospect_detail", lambda pid, row: {})
    pack, _tools = evidence.build_rookie(
        _bundle(), {"player_names": ["Alpha Prospect", "Beta Prospect"]},
        ["w1", "w2"])
    assert pack["prospect"]["name"] in ("Alpha Prospect", "Beta Prospect")
    assert len(pack["unanswered_names"]) == 1


# -- composers: the answer names the second player ----------------------------

def test_mock_player_answer_names_both(base):
    pack, _tools = evidence.build_player(_bundle(), {}, ["w1", "w2"])
    ans = compose_mock({"id": "t", "type": "player", "question": "how are they doing",
                        "evidence_pack": pack})
    assert "Nico Collins" in ans["text"]
    assert "17.3" in ans["text"]


def test_mock_breakout_history_answer_names_both(base, monkeypatch):
    monkeypatch.setattr(evidence, "breakout_call_history_for",
                        lambda pid, season=None: None)
    monkeypatch.setattr(evidence, "breakout_board_candidates",
                        lambda season=None: {})
    pack, _tools = evidence.build_breakout_history(_bundle(), {}, ["w1", "w2"])
    ans = compose_mock({"id": "t", "type": "breakout_history",
                        "question": "were they calls?", "evidence_pack": pack})
    assert "Garrett Wilson" in ans["text"]
    assert "Nico Collins was not a breakout call." in ans["text"]


def test_no_data_answer_names_unanswered_subject(base):
    pack, _tools = evidence.build_replacement(_bundle(), {}, ["w1", "w2"])
    pack["available"] = False
    pack["missing"] = "no younger replacement for Garrett Wilson in a comparable value band"
    ans = compose_mod.no_data_answer(pack)
    assert "Nico Collins" in ans["text"]


def test_compose_lines_skip_players_already_mentioned():
    pack = {"also_named": [{"player_id": "w2", "name": "Nico Collins",
                            "value": 305.0, "ppg": 15.1}]}
    assert compose_mod._multi_name_lines(
        "player", pack, "Wilson update", "Wilson and Collins are both fine.") == []
    lines = compose_mod._multi_name_lines(
        "player", pack, "Wilson update", "Wilson is scoring well.")
    assert lines and "Nico Collins" in lines[0] and "305" in lines[0]


def test_compose_lines_unanswered_pointer():
    pack = {"subject": {"name": "Garrett Wilson"},
            "unanswered_names": ["Nico Collins"]}
    lines = compose_mod._multi_name_lines("handcuff", pack, "Spears", "Spears is next up.")
    assert lines == ["That covers Garrett Wilson; ask about Nico Collins "
                     "for the same check on him."]


def test_lite_named_low_snap_share_is_not_rescaled(base, monkeypatch):
    # A genuine 1.2% snap share (percent scale, as _metrics_for ships)
    # must stay 1.2 in the lite row: the pre-fix _lite_named rescaled
    # anything <= 1.5 as if it were a 0-1 fraction and printed 120.
    monkeypatch.setattr(
        evidence, "_metrics_for",
        lambda pid, season=None: ({"ppg": 2.0, "target_share": 3.0,
                                   "snap_share": 1.2, "games": 6}, ["stub"]))
    rows = evidence._lite_named(["w1", "w2"], INDEX,
                                evidence._value_rows(), 2026)
    assert rows[0]["snap_share"] == 1.2
