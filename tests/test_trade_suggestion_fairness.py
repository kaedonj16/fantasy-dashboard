"""Guards the value-fairness layer in build_trade_suggestions_context.

The suggestion engine used to rank trade partners purely by positional fit and
never looked at whether the two sides were close in value, so it could surface
lopsided deals (give ~2x what you get) that no manager would ink. These tests
pin the new behavior: fleece-level mismatches are dropped, surfaced partners are
ranked by a composite score, and each carries the fairness/balance annotations.

Pure functions only (the heavy GM-context dependency is stubbed), so this runs
in the base suite without Flask/pandas.
"""
import dashboard_services.ai.context_builders as cb


def _mv(pid, name, pos, val):
    return {"player_id": pid, "name": name, "position": pos, "value": val}


def _build_ctx():
    """A 10-team league where the viewer (roster '1') is loaded at WR and thin at
    RB. Teams 2-4 are the natural partners: RB-rich AND WR-poor, so they hold an
    RB surplus the viewer needs and need the WR depth the viewer can spare. Teams
    5-10 are balanced filler so the rank cutoffs land realistically."""
    mvt = []
    rosters = []
    roster_map = {}
    # Viewer: elite WRs, weak RB.
    mvt += [
        _mv("v_wr1", "Viewer WR1", "WR", 3600),
        _mv("v_wr2", "Viewer WR2", "WR", 3100),
        _mv("v_wr3", "Viewer WR3", "WR", 2400),
        _mv("v_rb1", "Viewer RB1", "RB", 700),
        _mv("v_qb1", "Viewer QB1", "QB", 1500),
        _mv("v_te1", "Viewer TE1", "TE", 1000),
    ]
    rosters.append({"roster_id": "1", "players": ["v_wr1", "v_wr2", "v_wr3", "v_rb1", "v_qb1", "v_te1"]})
    roster_map["1"] = "Viewer"

    def _team(i, rb_hi, rb_lo, wr, qb=1100, te=1000):
        rid = str(i)
        mvt.extend([
            _mv(f"r{i}_rb1", f"T{i} RB1", "RB", rb_hi),
            _mv(f"r{i}_rb2", f"T{i} RB2", "RB", rb_lo),
            _mv(f"r{i}_wr1", f"T{i} WR1", "WR", wr),
            _mv(f"r{i}_qb1", f"T{i} QB1", "QB", qb),
            _mv(f"r{i}_te1", f"T{i} TE1", "TE", te),
        ])
        rosters.append({"roster_id": rid, "players": [f"r{i}_rb1", f"r{i}_rb2", f"r{i}_wr1", f"r{i}_qb1", f"r{i}_te1"]})
        roster_map[rid] = f"Team {i}"

    # Partners 2-4: RB-rich, WR-poor (mirror the viewer).
    _team(2, 3300, 2700, 520)
    _team(3, 3100, 2600, 560)
    _team(4, 2900, 2500, 600)
    # Filler 5-10: balanced, mid RB / mid-high WR so they aren't WR-needy.
    for i in range(5, 11):
        _team(i, 1500, 1200, 1500)

    return {
        "rosters": rosters,
        "model_value_table": mvt,
        "roster_map": roster_map,
        "picks_by_roster": {},
        "standings_map": {r["roster_id"]: {"wins": 5, "losses": 5} for r in rosters},
        "rookie_rankings": [],
        "league_type": "1qb",
    }


def _run(monkeypatch):
    monkeypatch.setattr(
        cb, "build_team_gm_context",
        lambda ctx, rid: {"team_name": "Viewer", "direction": "balanced"},
    )
    return cb.build_trade_suggestions_context(_build_ctx(), "1")


def test_surfaced_partners_are_not_fleece_level(monkeypatch):
    res = _run(monkeypatch)
    assert res is not None
    partners = res["top_partners"]
    assert partners, "expected at least one realistic trade partner"
    # Every surfaced deal clears the fairness floor and carries the ranking fields.
    for p in partners:
        assert p["fairness"] >= 0.62
        assert "suggestion_score" in p


def test_partners_ranked_by_composite_score(monkeypatch):
    res = _run(monkeypatch)
    scores = [p["suggestion_score"] for p in res["top_partners"]]
    assert scores == sorted(scores, reverse=True), "partners must be ranked best-first"


def test_viewer_need_is_detected(monkeypatch):
    res = _run(monkeypatch)
    # Viewer is stacked at WR and thin at RB, so RB should read as a need and WR
    # as surplus regardless of the exact rank cutoffs.
    assert "RB" in res["viewer_needs"]
    assert "WR" in res["viewer_surplus"]
