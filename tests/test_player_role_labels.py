"""Tests for the bulk role labels path (Start/Sit + Rankings rows).

Covers dashboard_services.player_roles.build_role_labels /
get_role_labels and GET /api/player-roles/labels: response shape,
position filtering, Contingent labelling, agreement with the
single-player payload's job/security, and the bounded-cache behaviour
(one build per TTL window, shared by concurrent readers).
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

from dashboard_services import player_roles as pr  # noqa: E402

LEAD_RB, NEXT_RB, OTHER_RB = "1001", "1002", "1003"
ALPHA_WR, SECOND_WR = "2001", "2002"


def _row(week, **kw):
    base = {"week": week, "snap_pct": 50.0, "snaps": 30, "team_snaps": 60,
            "targets": 0, "receptions": 0, "carries": 0, "touches": 0,
            "target_share": None, "carry_share": None, "ppr_pts": 0.0,
            "rec_yards": 0.0, "rush_yards": 0.0, "pass_att": 0,
            "rec_tds": 0, "rush_tds": 0, "pass_tds": 0,
            "rz_targets": 0, "rz_carries": 0}
    base.update(kw)
    return base


def _pool():
    """Small pool with one Contingent back: NEXT_RB produces almost
    only in the weeks LEAD_RB is out (weeks 3-6)."""
    pool = {
        LEAD_RB: {"pos": "RB", "team": "DET", "name": "Lead Back",
                  "series": [_row(w, snap_pct=65.0, snaps=45, carries=18,
                                  targets=5, receptions=4, carry_share=60.0,
                                  target_share=12.0, rush_yards=85.0,
                                  rec_yards=30.0, ppr_pts=18.0, rz_carries=2)
                             for w in (1, 2)]},
        NEXT_RB: {"pos": "RB", "team": "DET", "name": "Next Man",
                  "series": (
                      [_row(w, snap_pct=25.0, snaps=15, carries=4,
                            targets=1, receptions=1, carry_share=14.0,
                            target_share=3.0, rush_yards=16.0,
                            rec_yards=6.0, ppr_pts=2.0)
                       for w in (1, 2)]
                      + [_row(w, snap_pct=60.0, snaps=40, carries=14,
                              targets=3, receptions=2, carry_share=50.0,
                              target_share=8.0, rush_yards=62.0,
                              rec_yards=18.0, ppr_pts=16.0, rz_carries=1)
                         for w in (3, 4, 5, 6)])},
        OTHER_RB: {"pos": "RB", "team": "KC", "name": "Other Back",
                   "series": [_row(w, snap_pct=55.0, snaps=36, carries=12,
                                   targets=3, receptions=2, carry_share=45.0,
                                   target_share=9.0, rush_yards=52.0,
                                   rec_yards=14.0, ppr_pts=11.0, rz_carries=1)
                              for w in range(1, 7)]},
        ALPHA_WR: {"pos": "WR", "team": "DAL", "name": "Alpha Wideout",
                   "series": [_row(w, snap_pct=95.0, snaps=58, targets=11,
                                   receptions=8, target_share=30.0,
                                   rec_yards=105.0, ppr_pts=19.0, rz_targets=2)
                              for w in range(1, 7)]},
        SECOND_WR: {"pos": "WR", "team": "DAL", "name": "Second Wideout",
                    "series": [_row(w, snap_pct=70.0, snaps=42, targets=5,
                                    receptions=3, target_share=14.0,
                                    rec_yards=48.0, ppr_pts=8.0, rz_targets=1)
                               for w in range(1, 7)]},
    }
    return pool


@pytest.fixture()
def labels(monkeypatch):
    monkeypatch.setattr(pr, "load_pool", lambda season: _pool())
    pr.clear_labels_cache()
    yield pr.get_role_labels(2026)
    pr.clear_labels_cache()


# ---------------------------------------------------------------------------
# Engine: shape + content
# ---------------------------------------------------------------------------

def test_labels_cover_pool_with_job_security_pos(labels):
    assert set(labels) == {LEAD_RB, NEXT_RB, OTHER_RB, ALPHA_WR, SECOND_WR}
    for entry in labels.values():
        assert set(entry) == {"job", "security", "pos"}
        assert entry["job"]
        assert entry["security"] in ("Locked", "Stable", "Volatile",
                                     "Contingent")


def test_contingent_back_is_labelled_contingent(labels):
    assert labels[NEXT_RB]["security"] == "Contingent"
    text = pr.role_label_text(labels[NEXT_RB])
    assert text == f"{labels[NEXT_RB]['job']} · Contingent"


def test_non_contingent_label_text_is_job_only(labels):
    assert labels[ALPHA_WR]["security"] != "Contingent"
    assert pr.role_label_text(labels[ALPHA_WR]) == labels[ALPHA_WR]["job"]
    assert pr.role_label_text(None) == ""
    assert pr.role_label_text({}) == ""


def test_labels_match_single_player_payload(monkeypatch):
    """The bulk labels must agree with /api/player-role's job+security
    for the season window (the payload path adds PBP/FTN data, which
    never feeds job or security)."""
    pool = _pool()
    monkeypatch.setattr(pr, "load_pool", lambda season: pool)
    monkeypatch.setattr(pr, "get_pbp_tables", lambda season, window: None)
    monkeypatch.setattr(pr, "load_pa_feats", lambda win_pool, season: {})
    bulk = pr.build_role_labels(2026, pool=pool)
    for pid in (NEXT_RB, ALPHA_WR, OTHER_RB):
        payload = pr.build_player_payload(pid, 2026, "season", pool=pool)
        assert payload is not None
        assert bulk[pid]["job"] == payload["role"]["job"]
        assert bulk[pid]["security"] == payload["role"]["security"]


def test_empty_pool_gives_no_labels(monkeypatch):
    monkeypatch.setattr(pr, "load_pool", lambda season: {})
    assert pr.build_role_labels(2026) == {}


# ---------------------------------------------------------------------------
# Engine: bounded cache behaviour
# ---------------------------------------------------------------------------

def test_labels_cache_builds_once_until_cleared(monkeypatch):
    calls = {"n": 0}

    def counting_load(season):
        calls["n"] += 1
        return _pool()

    monkeypatch.setattr(pr, "load_pool", counting_load)
    pr.clear_labels_cache()
    first = pr.get_role_labels(2026)
    second = pr.get_role_labels(2026)
    assert first is second
    assert calls["n"] == 1
    pr.clear_labels_cache()
    pr.get_role_labels(2026)
    assert calls["n"] == 2
    pr.clear_labels_cache()


def test_labels_cache_rebuilds_after_ttl(monkeypatch):
    calls = {"n": 0}

    def counting_load(season):
        calls["n"] += 1
        return _pool()

    monkeypatch.setattr(pr, "load_pool", counting_load)
    pr.clear_labels_cache()
    pr.get_role_labels(2026)
    # Age the cached entry past the TTL; the next read must rebuild.
    with pr._CACHE_LOCK:
        ts, cached = pr._LABELS_CACHE[2026]
        pr._LABELS_CACHE[2026] = (ts - pr._LABELS_TTL - 1, cached)
    pr.get_role_labels(2026)
    assert calls["n"] == 2
    pr.clear_labels_cache()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(monkeypatch):
    from flask import Flask

    import routes.player_roles_bp as bp_mod

    monkeypatch.setattr(pr, "load_pool", lambda season: _pool())
    pr.clear_labels_cache()
    bp_mod.clear_cache()
    app = Flask(__name__)
    app.register_blueprint(bp_mod.player_roles_bp)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
    pr.clear_labels_cache()


def test_labels_endpoint_shape(client):
    resp = client.get("/api/player-roles/labels?season=2026")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["season"] == 2026
    assert body["window"] == "season"
    labels = body["labels"]
    assert set(labels) == {LEAD_RB, NEXT_RB, OTHER_RB, ALPHA_WR, SECOND_WR}
    for entry in labels.values():
        # The wire shape is exactly {job, security}; pos stays internal.
        assert set(entry) == {"job", "security"}


def test_labels_endpoint_contingent_entry(client):
    labels = client.get("/api/player-roles/labels?season=2026").get_json()["labels"]
    assert labels[NEXT_RB]["security"] == "Contingent"
    assert labels[NEXT_RB]["job"]


def test_labels_endpoint_position_filter(client):
    rb = client.get("/api/player-roles/labels?season=2026&position=RB").get_json()["labels"]
    assert set(rb) == {LEAD_RB, NEXT_RB, OTHER_RB}
    wr = client.get("/api/player-roles/labels?season=2026&position=wr").get_json()["labels"]
    assert set(wr) == {ALPHA_WR, SECOND_WR}
    # A bogus position is ignored (full pool), never an error.
    all_ = client.get("/api/player-roles/labels?season=2026&position=K").get_json()["labels"]
    assert set(all_) == {LEAD_RB, NEXT_RB, OTHER_RB, ALPHA_WR, SECOND_WR}


# ---------------------------------------------------------------------------
# Pages: dynasty hub carries NO role labels; the ROS hub keeps them
# (dynasty value is a multi-year asset price, so a current-season job
# label would mislead there; labels stay on /players, ROS, Start/Sit)
# ---------------------------------------------------------------------------

def _value_table():
    return [
        {"id": LEAD_RB, "name": "Lead Back", "position": "RB", "team": "DET",
         "value": 5200.0, "sf_value": 6100.0, "age": 24},
        {"id": ALPHA_WR, "name": "Alpha Wideout", "position": "WR",
         "team": "DAL", "value": 6800.0, "sf_value": 5900.0, "age": 25},
    ]


def test_dynasty_hub_renders_no_role_labels():
    from dashboard_services.pages.dynasty_pages import (
        build_rankings_hub_body)

    for position in (None, "RB"):
        body = build_rankings_hub_body(_value_table(), position=position)
        assert "Lead Back" in body  # rows still render
        assert "rnk-role" not in body


def test_ros_hub_still_renders_role_labels(labels):
    from dashboard_services.pages.dynasty_pages import (
        build_ros_rankings_body)

    payload = {
        "players": {LEAD_RB: {"ros_ppg": 18.0, "ros_games": 7.0,
                              "ros_total": 126.0, "ros_score": 91.0}},
        "window": {"start_week": 12, "end_week": 17},
    }
    body = build_ros_rankings_body(payload, _value_table(), position="RB",
                                   role_labels=labels)
    expected = pr.role_label_text(labels[LEAD_RB])
    assert expected
    assert f'<span class="rnk-role">{expected}</span>' in body
    # Without a labels map the ROS page renders no span either.
    body = build_ros_rankings_body(payload, _value_table(), position="RB")
    assert "rnk-role" not in body
