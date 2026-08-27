import json
from types import SimpleNamespace

import pytest

from dashboard_services.pages.draft_room_page import build_draft_room_body
from dashboard_services.providers import espn_api
from utils.fantasy_scoring import score_stats
from utils.projection_resolver import resolve_projected_ppg
from utils.league_scoring import normalize_league_scoring


def _items(rec):
    return [
        {"statId": 53, "points": rec},
        {"statId": 3, "points": .04},
        {"statId": 4, "points": 4},
        {"statId": 20, "points": -2},
        {"statId": 24, "points": .1},
        {"statId": 25, "points": 6},
        {"statId": 42, "points": .1},
        {"statId": 43, "points": 6},
        {"statId": 72, "points": -2},
    ]


@pytest.mark.parametrize(("reception_value", "expected"), [
    (0, 0.0), (.5, .5), (1, 1.0), (.75, .75),
])
def test_espn_scoring_item_53_preserves_standard_half_full_and_custom(reception_value, expected):
    normalized = espn_api.normalize_espn_scoring_items(_items(reception_value))
    assert normalized["rec"] == expected


def test_espn_explicit_zero_points_override_is_not_discarded():
    normalized = espn_api.normalize_espn_scoring_items([
        {"statId": 53, "points": 1, "pointsOverrides": {"16": 0}},
    ])
    assert normalized["rec"] == 0


def test_get_league_globals_reads_raw_msettings_not_coarse_scoring_type(monkeypatch):
    class Request:
        def league_get(self, params):
            assert params == {"view": "mSettings"}
            return {"settings": {"scoringSettings": {"scoringItems": _items(1)}}}

    league = SimpleNamespace(
        # This is the previous failure: missing/coarse metadata said standard.
        settings=SimpleNamespace(scoring_type="standard", roster_slots=["QB", "RB", "WR", "TE"],
                                 playoff_team_count=6),
        teams=[object()] * 12,
        espn_request=Request(),
    )
    monkeypatch.setattr(espn_api, "_league", lambda season, league_id: league)
    out = espn_api.get_league_globals(2026, "123")
    assert out["scoring_settings"]["rec"] == 1.0
    assert out["scoring_settings"]["pointsPerReception"] == 1.0


@pytest.mark.parametrize(("rec", "expected"), [(0, 16), (.5, 21), (1, 26), (.75, 23.5)])
def test_player_modal_stat_math_uses_normalized_espn_scoring(rec, expected):
    scoring = espn_api.normalize_espn_scoring_items(_items(rec))
    line = {"rec": 10, "rec_yd": 100, "rec_td": 1}
    assert score_stats(line, scoring, "WR") == expected


def test_draft_room_config_and_projection_resolver_share_espn_scoring():
    scoring = espn_api.normalize_espn_scoring_items(_items(1))
    body = build_draft_room_body("123", 2026, "espn", scoring={
        "ppr": scoring["rec"], "tep": 0, "passTd": scoring["pass_td"],
    })
    marker = "window.__draftCfg = "
    cfg = json.loads(body.split(marker, 1)[1].split(";", 1)[0])
    assert cfg["scoring"]["ppr"] == 1.0

    projection = resolve_projected_ppg(
        "wr", scoring, 2026, weekly_maps={1: {"wr": {"raw_stats": {
            "rec": 10, "rec_yd": 100, "rec_td": 1,
        }}}}, position="WR",
    )
    assert projection["ppg"] == 26


def test_draft_room_standard_remains_zero():
    scoring = espn_api.normalize_espn_scoring_items(_items(0))
    assert scoring["rec"] == 0


@pytest.mark.parametrize("value", [0, .5, 1, .75])
def test_provider_agnostic_contract_preserves_reception_values(value):
    assert normalize_league_scoring("espn", {"rec": value})["rec"] == value


@pytest.mark.parametrize("platform", ["sleeper", "espn", "yahoo", "mfl"])
def test_cross_provider_scoring_contract_uses_same_internal_shape(platform):
    raw = {"pointsPerReception": .5, "passTD": 6, "receivingYards": .1}
    normalized = normalize_league_scoring(platform, raw)
    assert normalized["rec"] == .5
    assert normalized["pass_td"] == 6
    assert normalized["rec_yd"] == .1


def test_connected_scoring_is_applied_before_first_player_load():
    source = open("static/draft_room.js", encoding="utf-8").read()
    apply_cfg = source.index("if (cfg.scoring)")
    first_start_load = source.index("loadPlayers();", source.index("function startDraft()"))
    assert apply_cfg < first_start_load


def test_roster_projection_card_is_directly_after_grade_before_roster():
    source = open("static/draft_room.js", encoding="utf-8").read()
    render = source[source.index("function renderNeeds()"):
                    source.index("// Monochrome inline icons", source.index("function renderNeeds()"))]
    assert render.count('class="dr-proj-card"') == 1
    assert render.index('class="dr-grade-card"') < render.index('class="dr-proj-card"')
    assert render.index('class="dr-proj-card"') < render.index('class="dr-roster"')
