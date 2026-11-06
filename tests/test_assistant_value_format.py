"""League-format value pricing for assistant packs (PR A: values integrity).

Live gap audit 2026-11-06: the packs' shared value read took the raw
model table's ``value`` column, which is the 1QB dynasty number, in
every league. In Kaedon's SF redraft league a trade question priced
Josh Allen at his 1QB dynasty value while the picks beside him came
from the SF-aware pick table, one fairness gap mixing two formats.
The builders now read the site's canonical league-format rewrite
(league_format_value_lookup via evidence._format_values): SF vs 1QB,
redraft vs dynasty, scoring format and TE premium follow the league,
exactly like the player modal and the Teams page. These tests pin the
three format shapes and the activity pick pricing that rides along:
completed trades in the activity pack price the picks that moved, and
a trade with an unpriced asset gets no fairness verdict.
"""
import pytest

pytest.importorskip("openai")
pytest.importorskip("pandas")
pytestmark = pytest.mark.integration

import pandas as pd

from dashboard_services.assistant import evidence

INDEX = {
    "q1": {"name": "Josh Allen", "team": "BUF", "position": "QB"},
    "w1": {"name": "Ja'Marr Chase", "team": "CIN", "position": "WR"},
    "x1": {"name": "Player X", "team": "NYJ", "position": "WR"},
    "y1": {"name": "Player Y", "team": "DAL", "position": "RB"},
}

# One row, four materially different format prices.
VALUES = [
    {"id": "q1", "value": 3000, "sf_value": 6500,
     "redraft_value_1qb": 5200, "redraft_value_sf": 8800},
    {"id": "w1", "value": 4000, "sf_value": 3900,
     "redraft_value_1qb": 3600, "redraft_value_sf": 3500},
    {"id": "x1", "value": 1000, "sf_value": 1000,
     "redraft_value_1qb": 1000, "redraft_value_sf": 1000},
    {"id": "y1", "value": 3000, "sf_value": 3000,
     "redraft_value_1qb": 3000, "redraft_value_sf": 3000},
]

SF_SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "SUPER_FLEX", "BN", "BN"]
QB_SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN", "BN"]


@pytest.fixture()
def env(monkeypatch):
    monkeypatch.setattr(evidence, "get_players_index_global",
                        lambda *a, **k: dict(INDEX))
    monkeypatch.setattr(evidence, "get_model_value_table_cached",
                        lambda *a, **k: [dict(r) for r in VALUES])
    monkeypatch.setattr(evidence, "_metrics_for",
                        lambda pid, season=None: ({}, []))
    monkeypatch.setattr("dashboard_services.news.get_player_news",
                        lambda name, limit=3: {})
    monkeypatch.setattr("data_building.advanced_metrics.get_player_metric_ranks",
                        lambda *a, **k: {})
    monkeypatch.setattr(evidence, "load_sleeper_week_stats",
                        lambda season, week: {})
    monkeypatch.setattr(evidence, "load_week_sched", lambda season, week: [])
    return monkeypatch


def _bundle(ctx, **over):
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "League", "team_name": "Couch", "week": 7,
              "ctx": ctx, "roster": {"roster_id": 1, "players": [],
                                     "starters": []},
              "roster_player_ids": set()}
    bundle.update(over)
    return bundle


def _eval(bundle):
    entities = {"give_player_ids": ["q1"], "receive_player_ids": ["w1"]}
    pack, _tools = evidence.build_trade_eval(bundle, entities, ["q1", "w1"])
    return pack


# -- (a) superflex dynasty: the QB is priced off the SF rewrite ---------------

def test_sf_league_trade_eval_prices_qb_at_sf_value(env):
    ctx = {"rosters": [], "roster_positions": SF_SLOTS}
    pack = _eval(_bundle(ctx))
    assert pack["give"]["players"][0]["value"] == 6500
    assert pack["give"]["total_value"] == 6500
    assert pack["receive"]["total_value"] == 3900
    assert pack["fairness_gap"] == round(3900 - 6500, 1)


def test_sf_league_player_pack_value_is_sf(env):
    ctx = {"rosters": [], "roster_positions": SF_SLOTS}
    pack, _tools = evidence.build_player(_bundle(ctx), {}, ["q1"])
    assert pack["player"]["value"] == 6500


# -- (b) redraft: values come from the redraft rewrite -------------------------

def test_redraft_league_trade_eval_uses_redraft_values(env):
    ctx = {"rosters": [], "roster_positions": QB_SLOTS,
           "league_settings": {"type": 1}}
    pack = _eval(_bundle(ctx))
    assert pack["give"]["total_value"] == 5200
    assert pack["receive"]["total_value"] == 3600


def test_redraft_league_player_pack_value_is_redraft(env):
    ctx = {"rosters": [], "roster_positions": QB_SLOTS,
           "league_settings": {"type": 1}}
    pack, _tools = evidence.build_player(_bundle(ctx), {}, ["q1"])
    assert pack["player"]["value"] == 5200


def test_sf_redraft_league_uses_redraft_sf_column(env):
    ctx = {"rosters": [], "roster_positions": SF_SLOTS,
           "league_settings": {"type": 1}}
    pack = _eval(_bundle(ctx))
    assert pack["give"]["total_value"] == 8800


# -- (c) 1QB dynasty: unchanged, the raw value column is the format ------------

def test_1qb_dynasty_values_unchanged(env):
    ctx = {"rosters": [], "roster_positions": QB_SLOTS}
    pack = _eval(_bundle(ctx))
    assert pack["give"]["total_value"] == 3000
    assert pack["receive"]["total_value"] == 4000
    pack, _tools = evidence.build_player(_bundle(ctx), {}, ["q1"])
    assert pack["player"]["value"] == 3000


# -- (d)/(e) activity: picks priced into totals, identity kept -----------------

def _pick_frame(pick):
    rows = [
        {"kind": "trade", "week": 7, "ts": pd.Timestamp("2026-10-12", tz="UTC"),
         "data": {"teams": [
             {"rid": "1", "roster_id": 1, "name": "Team A",
              "gets": [{"pid": "y1", "name": "Player Y", "pos": "RB"}],
              "sends": [{"pid": "x1", "name": "Player X", "pos": "WR"}]},
             {"rid": "2", "roster_id": 2, "name": "Team B",
              "gets": [{"pid": "x1", "name": "Player X", "pos": "WR"}],
              "sends": [{"pid": "y1", "name": "Player Y", "pos": "RB"}]}],
             "draft_picks": [pick]}},
    ]
    return pd.DataFrame(rows, columns=["kind", "week", "ts", "data"])


@pytest.fixture()
def pick_env(env, monkeypatch):
    monkeypatch.setattr(evidence, "ensure_activity_bits", lambda ctx: None)
    monkeypatch.setattr(evidence, "traded_picks_for", lambda *a, **k: [])
    monkeypatch.setattr(evidence, "pick_value_table",
                        lambda league_teams=10, is_sf=False: {"fake": True})

    def _price(table, season, rnd, slot, num_teams=10):
        return 2100.0 if (season, rnd) == (2027, 1) else None

    monkeypatch.setattr("utils.draft.pick_value_from_table", _price)
    return monkeypatch


def _activity_bundle(frame):
    ctx = {"activity_df": frame, "rosters": [], "roster_positions": QB_SLOTS}
    return _bundle(ctx)


def test_activity_trade_prices_pick_into_totals_and_fairness(pick_env):
    # Team A sends Player X (1000) plus its 2027 1st (2100) for Player Y
    # (3000). Player-only math reads 3000 vs 1000, a lopsided win for A;
    # with the pick priced, B receives 3100 and the deal is fair.
    pick = {"season": "2027", "round": 1, "roster_id": 1,
            "previous_owner_id": 1, "owner_id": 2}
    pack, _tools = evidence.build_activity(
        _activity_bundle(_pick_frame(pick)), {}, [])
    latest = pack["latest_trade"]
    team_a, team_b = latest["teams"]
    assert team_a["gets_value"] == 3000
    assert team_b["gets_value"] == 3100
    assert latest["value_gap"] == 100
    assert latest["fairness"] == "fair"
    assert latest["ahead_team"] == "Team B"
    assert latest["picks_moved"] == 1
    (entry,) = latest["picks"]
    assert entry["season"] == 2027 and entry["round"] == 1
    assert entry["value"] == 2100 and entry["priced"] is True
    assert "2027" in entry["label"]
    assert team_b["gets_picks"] == [entry]
    assert team_a["sends_picks"] == [entry]


def test_activity_unpriced_pick_suppresses_fairness(pick_env):
    # A 2028 2nd has no price in the table: it stays in the pack marked
    # unpriced, and the trade gets no fairness verdict from partial math.
    pick = {"season": "2028", "round": 2, "roster_id": 1,
            "previous_owner_id": 1, "owner_id": 2}
    pack, _tools = evidence.build_activity(
        _activity_bundle(_pick_frame(pick)), {}, [])
    latest = pack["latest_trade"]
    (entry,) = latest["picks"]
    assert entry["priced"] is False and entry["value"] is None
    assert latest["teams"][1]["gets_picks"] == [entry]
    assert latest["unpriced"] >= 1
    assert "fairness" not in latest
    assert "value_gap" not in latest
    assert "ahead_team" not in latest
