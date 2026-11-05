"""Start/Sit role chips (Wave 2 PR 1, spec section 2.4 / section 5 PR 1).

A promoted QB's Start/Sit row carries the role chip with the state's
reason, and role states are display-only: compute_start_score outputs
are identical with and without role states present.
"""
import pytest

pytest.importorskip("pandas")

WEEK = 5
NAMES = {"qb_out": "Sam Starter", "qb_up": "Drew Backup",
         "rb1": "Runner One"}
TEAMS = {"qb_out": "BAL", "qb_up": "BAL", "rb1": "KC"}
POS = {"qb_out": "QB", "qb_up": "QB", "rb1": "RB"}
PROJ = {"qb_up": 20.67, "rb1": 15.0}
SCHED = [{"home": "BAL", "away": "KC"}, {"home": "KC", "away": "BAL"}]


def _ctx():
    idx = {}
    for pid, name in NAMES.items():
        idx[pid] = {
            "pos": POS[pid], "position": POS[pid], "team": TEAMS[pid],
            "full_name": name, "name": name,
            "injury_status": "IR" if pid == "qb_out" else "",
            "depth_chart_order": 1 if pid in ("qb_out", "rb1") else 2,
            "depth_chart_position": POS[pid],
            "status": "Inactive" if pid == "qb_out" else "Active",
        }
    return {
        "viewer": {"viewer_roster_id": 1},
        "current_week": WEEK,
        "rosters": [
            {"roster_id": 1, "owner_id": "me-user",
             "players": ["qb_out", "qb_up", "rb1"],
             "starters": ["qb_out", "rb1"],
             "reserve": [], "taxi": [], "metadata": {"team_name": "Mine"}},
        ],
        "users": [{"user_id": "me-user", "display_name": "Kaedon"}],
        "players_index": idx,
        "players": idx,
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX",
                             "BN", "BN"],
        "scoring_settings": {},
        "raw_scoring_settings": {},
        "proj_by_week": {WEEK: {"projections": dict(PROJ)}},
    }


@pytest.fixture
def ss_env(offline_client, monkeypatch):
    import app as appmod
    import data_building.start_score_bundle as bundle_mod
    import data_building.weekly_metrics as wm_mod
    import utils.game_conditions as gc_mod
    import utils.utils as uu_mod

    monkeypatch.setattr(appmod, "_session_signed_in", lambda: True)
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache",
                        lambda *a, **k: _ctx())
    rows = [{"id": p, "name": NAMES[p], "position": POS[p], "team": TEAMS[p],
             "value": 5000, "sf_value": 5000, "pos_rank_label": "QB1"}
            for p in NAMES]
    monkeypatch.setattr(appmod, "get_model_value_table_cached",
                        lambda *a, **k: rows)
    monkeypatch.setattr(appmod, "_matchup_rank_table",
                        lambda *a, **k: ({}, 32, None, None))
    monkeypatch.setattr(appmod, "_fpts_against_effective", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "_load_team_play_volume", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "_load_season_weekly_points",
                        lambda *a, **k: {})
    monkeypatch.setattr(appmod, "_load_season_snap_totals",
                        lambda *a, **k: {})
    monkeypatch.setattr(appmod, "build_projections_by_week",
                        lambda *a, **k: {})
    monkeypatch.setattr(appmod, "start_score_pos_anchors",
                        lambda *a, **k: {})
    monkeypatch.setattr(appmod, "_oline_for_player", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_ss_game_env", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_ss_qb_situation", lambda *a, **k: None)
    monkeypatch.setattr(bundle_mod, "variant_for_exact_scoring",
                        lambda *a, **k: None)
    monkeypatch.setattr(bundle_mod, "load_start_score_bundles",
                        lambda *a, **k: {})
    monkeypatch.setattr(wm_mod, "get_usage_trends", lambda *a, **k: {})
    monkeypatch.setattr(gc_mod, "build_week_conditions", lambda *a, **k: {})
    monkeypatch.setattr(uu_mod, "load_week_projection", lambda *a, **k: {})
    monkeypatch.setattr(uu_mod, "load_week_sched", lambda *a, **k: list(SCHED))
    return appmod


def _rows(payload):
    return {p["player_id"]: p
            for rows in payload["positions"].values() for p in rows}


def test_promoted_qb_row_carries_role_chip(ss_env):
    payload, status = ss_env._build_start_sit_options(
        "sleeper", "L1", 2026)
    assert status == 200
    rows = _rows(payload)
    up = rows["qb_up"]
    assert up["role_state"] == "promoted"
    chip = up["role_situation"]
    assert chip is not None
    assert chip["kind"] == "promoted"
    assert chip["label"] == "Starting role"
    # The chip carries the state's reason.
    assert "Sam Starter" in chip["note"]
    assert "BAL" in chip["note"]
    assert up["role_reason"] == chip["note"]
    # An ordinary starter carries no role chip.
    assert rows["rb1"]["role_state"] == "starter"
    assert rows["rb1"]["role_situation"] is None


def test_compute_start_score_identical_with_and_without_role_states(
        ss_env, monkeypatch):
    import utils.role_state as role_state_mod

    with_roles, status = ss_env._build_start_sit_options(
        "sleeper", "L1", 2026)
    assert status == 200
    monkeypatch.setattr(role_state_mod, "get_role_states",
                        lambda *a, **k: {})
    without_roles, status = ss_env._build_start_sit_options(
        "sleeper", "L1", 2026)
    assert status == 200

    rows_with = _rows(with_roles)
    rows_without = _rows(without_roles)
    assert rows_with["qb_up"]["role_situation"] is not None
    assert rows_without["qb_up"]["role_situation"] is None
    assert set(rows_with) == set(rows_without)
    for pid in rows_with:
        a, b = rows_with[pid], rows_without[pid]
        # The no-silent-influence test: scores and every factor behind
        # them are byte-identical whether role states exist or not.
        assert a["start_score"] == b["start_score"]
        assert a["start_score_pct"] == b["start_score_pct"]
        assert a["score_factors"] == b["score_factors"]
        assert a["proj_pts"] == b["proj_pts"]
        assert a["demotion"] == b["demotion"]
