"""Guards the ranking layers in build_trade_suggestions_context.

The suggestion engine used to rank trade partners purely by positional fit. It
now also (1) drops fleece-level value mismatches and ranks the rest by a fairness
composite, and (2) nudges the ranking by how well the acquisition's age profile
fits the viewer's competitive window (contenders → proven, rebuilders → youth).
These tests pin both behaviors.

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


# ── Team-direction (age) weighting ─────────────────────────────────────────────

def _mva(pid, name, pos, val, age):
    return {"player_id": pid, "name": name, "position": pos, "value": val, "age": age}


def _build_age_ctx():
    """WR-rich / RB-poor viewer with two equal-value RB partners: Team 2 offers
    young backs (22), Team 3 offers older backs (30). Filler keeps cutoffs sane."""
    mvt = [
        _mva("v_wr1", "V WR1", "WR", 3600, 25), _mva("v_wr2", "V WR2", "WR", 3100, 26),
        _mva("v_wr3", "V WR3", "WR", 2400, 24), _mva("v_rb1", "V RB1", "RB", 700, 27),
        _mva("v_te1", "V TE1", "TE", 1000, 26),
    ]
    rosters = [{"roster_id": "1", "players": ["v_wr1", "v_wr2", "v_wr3", "v_rb1", "v_te1"]}]
    roster_map = {"1": "Viewer"}

    def team(i, age, rb_hi=3300, rb_lo=2700, wr=520):
        rid = str(i)
        mvt.extend([
            _mva(f"r{i}_rb1", f"T{i} RB1", "RB", rb_hi, age),
            _mva(f"r{i}_rb2", f"T{i} RB2", "RB", rb_lo, age),
            _mva(f"r{i}_wr1", f"T{i} WR1", "WR", wr, 26),
            _mva(f"r{i}_te1", f"T{i} TE1", "TE", 1000, 26),
        ])
        rosters.append({"roster_id": rid, "players": [f"r{i}_rb1", f"r{i}_rb2", f"r{i}_wr1", f"r{i}_te1"]})
        roster_map[rid] = f"Team {i}"

    team(2, 22)   # young backs
    team(3, 30)   # older backs
    for i in range(4, 11):
        team(i, 26, rb_hi=1500, rb_lo=1200, wr=1500)  # balanced filler

    return {
        "rosters": rosters, "model_value_table": mvt, "roster_map": roster_map,
        "picks_by_roster": {},
        "standings_map": {r["roster_id"]: {"wins": 5, "losses": 5} for r in rosters},
        "rookie_rankings": [], "league_type": "1qb",
    }


def _score_for(res, team_name):
    for p in res["top_partners"]:
        if p["team_name"] == team_name:
            return p["suggestion_score"]
    return None


def _run_dir(monkeypatch, direction):
    monkeypatch.setattr(
        cb, "build_team_gm_context",
        lambda ctx, rid: {"team_name": "Viewer", "direction": direction},
    )
    return cb.build_trade_suggestions_context(_build_age_ctx(), "1")


def test_rebuilder_prefers_younger_acquisition(monkeypatch):
    res = _run_dir(monkeypatch, "rebuilding")
    young, old = _score_for(res, "Team 2"), _score_for(res, "Team 3")
    assert young is not None and old is not None
    assert young > old, "a rebuilder should rank the younger RB package higher"


def test_contender_prefers_proven_acquisition(monkeypatch):
    res = _run_dir(monkeypatch, "contending")
    young, old = _score_for(res, "Team 2"), _score_for(res, "Team 3")
    assert young is not None and old is not None
    assert old > young, "a contender should rank the proven (older) RB package higher"


# ── Roster-aware consolidation ceiling (end-to-end through the real path) ──────

def _mva2(pid, name, pos, val):
    return {"player_id": pid, "name": name, "position": pos, "value": val, "age": 25}


def _build_ceiling_ctx():
    """WR-thin viewer (its only WR is flex-tier) that can still afford a rival's
    elite WR one-for-one with an RB, so absent the ceiling that elite would be a
    real, fair target. The ceiling should block it (a flex-only team shouldn't be
    pitched a top-of-position stud)."""
    mvt = [
        _mva2("v_wr1", "V WR1", "WR", 300),   # lone flex WR -> WR is a need, best WR is flex
        _mva2("v_rb1", "V RB1", "RB", 900), _mva2("v_rb2", "V RB2", "RB", 850),  # RB surplus to send
    ]
    rosters = [{"roster_id": "1", "players": ["v_wr1", "v_rb1", "v_rb2"]}]
    roster_map = {"1": "Viewer"}

    mvt += [_mva2("p_wr1", "Elite WR", "WR", 950), _mva2("p_rb1", "P RB1", "RB", 250)]
    rosters.append({"roster_id": "2", "players": ["p_wr1", "p_rb1"]})
    roster_map["2"] = "Partner"

    for i in range(3, 13):  # filler to seed the need/surplus cutoffs
        mvt += [_mva2(f"f{i}_wr", f"F{i} WR", "WR", 400), _mva2(f"f{i}_rb", f"F{i} RB", "RB", 450)]
        rosters.append({"roster_id": str(i), "players": [f"f{i}_wr", f"f{i}_rb"]})
        roster_map[str(i)] = f"Team {i}"

    return {
        "rosters": rosters, "model_value_table": mvt, "roster_map": roster_map,
        "picks_by_roster": {},
        "standings_map": {r["roster_id"]: {"wins": 5, "losses": 5} for r in rosters},
        "rookie_rankings": [], "league_type": "1qb",
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"],
    }


def _acquire_names(res):
    return [t["name"] for p in res["top_partners"] for t in (p.get("targets_they_have") or [])]


def test_ceiling_blocks_the_elite_for_a_flex_only_viewer_end_to_end(monkeypatch):
    """Differential: the ONLY thing that changes is whether the shared ceiling is
    active. With it on, the elite WR is never surfaced to a flex-only viewer; with
    it stubbed off, the exact same league DOES surface it - proving the ceiling
    (not the need/fairness machinery) is what removes it, through the real path."""
    monkeypatch.setattr(cb, "build_team_gm_context",
                        lambda ctx, rid: {"team_name": "Viewer", "direction": "balanced"})

    on = cb.build_trade_suggestions_context(_build_ceiling_ctx(), "1")
    assert "Elite WR" not in _acquire_names(on), "flex-only viewer must not be pitched the elite WR"

    # Disable just the ceiling (the function re-imports it per call, so patching
    # the source binds through) and re-run the identical league.
    import utils.player_tiers as pt
    monkeypatch.setattr(pt, "consolidate_target_allowed", lambda a, b: True)
    off = cb.build_trade_suggestions_context(_build_ceiling_ctx(), "1")
    assert "Elite WR" in _acquire_names(off), "without the ceiling the elite WR is a fair, real target"
