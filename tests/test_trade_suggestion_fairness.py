"""Guards the ranking layers in build_trade_suggestions_context.

The suggestion engine used to rank trade partners purely by positional fit. It
now also (1) drops value mismatches more than ~25% lopsided in either direction
and ranks the rest by a fairness composite, (2) builds consolidation packages by
keeping the fairest prefix instead of overshooting, and (3) nudges the ranking
by how well the acquisition's age profile fits the viewer's competitive window
(contenders → proven, rebuilders → youth). These tests pin all three behaviors.

Pure functions only (the heavy GM-context dependency is stubbed), so this runs
in the base suite without Flask/pandas.
"""
import dashboard_services.ai.context_builders as cb


def _mv(pid, name, pos, val):
    return {"player_id": pid, "name": name, "position": pos, "value": val}


_LINEUP = ["QB", "RB", "RB", "WR", "WR", "TE"]  # floors: RB 2, WR 2, QB 1, TE 1


def _build_ctx():
    """A 10-team league where the viewer (roster '1') can field a WR surplus (3
    startable WRs, one beyond its two WR slots) but has a starter-sized hole at RB
    (only one startable back). Teams 2-4 are the natural partners: they roster an
    RB surplus the viewer needs (3 startable backs) and a starter hole at WR the
    viewer's spare WR fills. Teams 5-10 are balanced filler (exactly their starter
    count at every spot, so neither needy nor in surplus) to keep cutoffs realistic.

    Values sit clearly above/below the league starter bar (350 * 12/10 = 420) so
    the starter-gap need model reads each roster the way a manager would."""
    mvt = []
    rosters = []
    roster_map = {}
    # Viewer: three startable WRs (surplus of one), a single startable RB (hole).
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

    def _team(i, rbs, wrs, qb=1100, te=1000):
        rid = str(i)
        pids = [f"r{i}_qb1", f"r{i}_te1"]
        mvt.extend([_mv(f"r{i}_qb1", f"T{i} QB1", "QB", qb), _mv(f"r{i}_te1", f"T{i} TE1", "TE", te)])
        for j, val in enumerate(rbs, start=1):
            mvt.append(_mv(f"r{i}_rb{j}", f"T{i} RB{j}", "RB", val))
            pids.append(f"r{i}_rb{j}")
        for j, val in enumerate(wrs, start=1):
            mvt.append(_mv(f"r{i}_wr{j}", f"T{i} WR{j}", "WR", val))
            pids.append(f"r{i}_wr{j}")
        rosters.append({"roster_id": rid, "players": pids})
        roster_map[rid] = f"Team {i}"

    # Partners 2-4: RB surplus (3 startable), starter hole at WR (1 startable).
    _team(2, [3300, 2700, 900], [520])
    _team(3, [3100, 2600, 900], [560])
    _team(4, [2900, 2500, 900], [600])
    # Filler 5-10: exactly two startable RBs and two startable WRs — balanced, so
    # they read as neither needy nor in surplus and don't crowd the partner ranks.
    for i in range(5, 11):
        _team(i, [1500, 1200], [1500, 1300])

    return {
        "rosters": rosters,
        "model_value_table": mvt,
        "roster_map": roster_map,
        "picks_by_roster": {},
        "standings_map": {r["roster_id"]: {"wins": 5, "losses": 5} for r in rosters},
        "rookie_rankings": [],
        "league_type": "1qb",
        "roster_positions": _LINEUP,
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
        assert p["fairness"] >= 0.80
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
    """WR-surplus / RB-hole viewer with two equal-value RB partners: Team 2 offers
    young backs (22), Team 3 offers older backs (30). Both hold a real RB surplus
    (3 startable) and a WR hole, so both surface; only their age profiles differ.
    Filler teams are balanced so they don't crowd the ranking."""
    mvt = [
        _mva("v_wr1", "V WR1", "WR", 3600, 25), _mva("v_wr2", "V WR2", "WR", 3100, 26),
        _mva("v_wr3", "V WR3", "WR", 2400, 24), _mva("v_rb1", "V RB1", "RB", 700, 27),
        _mva("v_te1", "V TE1", "TE", 1000, 26),
    ]
    rosters = [{"roster_id": "1", "players": ["v_wr1", "v_wr2", "v_wr3", "v_rb1", "v_te1"]}]
    roster_map = {"1": "Viewer"}

    def team(i, age, rbs=(3300, 2700, 900), wrs=(520,)):
        rid = str(i)
        pids = [f"r{i}_te1"]
        mvt.append(_mva(f"r{i}_te1", f"T{i} TE1", "TE", 1000, 26))
        for j, val in enumerate(rbs, start=1):
            mvt.append(_mva(f"r{i}_rb{j}", f"T{i} RB{j}", "RB", val, age))
            pids.append(f"r{i}_rb{j}")
        for j, val in enumerate(wrs, start=1):
            mvt.append(_mva(f"r{i}_wr{j}", f"T{i} WR{j}", "WR", val, 26))
            pids.append(f"r{i}_wr{j}")
        rosters.append({"roster_id": rid, "players": pids})
        roster_map[rid] = f"Team {i}"

    team(2, 22)   # young backs
    team(3, 30)   # older backs
    for i in range(4, 11):
        team(i, 26, rbs=(1500, 1200), wrs=(1500, 1300))  # balanced filler

    return {
        "rosters": rosters, "model_value_table": mvt, "roster_map": roster_map,
        "picks_by_roster": {},
        "standings_map": {r["roster_id"]: {"wins": 5, "losses": 5} for r in rosters},
        "rookie_rankings": [], "league_type": "1qb",
        "roster_positions": _LINEUP,
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


def test_redraft_omits_pick_trade_partners(monkeypatch):
    """Leftover Sleeper pick rows must not become pick-for-player ideas."""
    ctx = _build_ctx()
    ctx["league_settings"] = {"type": 0}
    ctx["picks_by_roster"] = {
        "1": [{"season": 2026, "round": 1, "original_owner": "1"}],
    }
    monkeypatch.setattr(
        cb, "build_team_gm_context",
        lambda _ctx, _rid: {"team_name": "Viewer", "direction": "balanced"},
    )
    res = cb.build_trade_suggestions_context(ctx, "1")
    assert res is not None
    assert res["scoring_type"] == "redraft"
    assert res["picks_tradable"] is False
    assert res["pick_trade_partners"] == []
    assert res["projected_picks"] == []


def test_dynasty_can_include_pick_trade_partners(monkeypatch):
    ctx = _build_ctx()
    ctx["league_settings"] = {"type": 2}
    ctx["picks_by_roster"] = {
        "1": [{"season": 2026, "round": 1, "original_owner": "1"}],
    }
    monkeypatch.setattr(
        cb, "build_team_gm_context",
        lambda _ctx, _rid: {"team_name": "Viewer", "direction": "balanced"},
    )
    res = cb.build_trade_suggestions_context(ctx, "1")
    assert res is not None
    assert res["scoring_type"] == "dynasty"
    assert res["picks_tradable"] is True
    assert res["pick_trade_partners"], "dynasty should still offer pick-for-player ideas"


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
    """WR-hole viewer whose best WR is flex-tier (below the 350 starter bar) and
    who holds a real RB surplus (3 startable backs) to trade. The partner rosters
    a WR surplus that includes an elite WR and has a starter hole at RB, so absent
    the ceiling that elite WR is a real, fair target for the viewer's spare RB. The
    ceiling should block it: a flex-only team shouldn't be pitched a top-of-position
    stud, only steered to a pure starter."""
    mvt = [
        _mva2("v_wr1", "V WR1", "WR", 300),   # lone flex WR -> WR is a hole, best WR is flex
        _mva2("v_rb1", "V RB1", "RB", 900), _mva2("v_rb2", "V RB2", "RB", 850),
        _mva2("v_rb3", "V RB3", "RB", 800),   # 3 startable RBs -> a startable RB to spare
    ]
    rosters = [{"roster_id": "1", "players": ["v_wr1", "v_rb1", "v_rb2", "v_rb3"]}]
    roster_map = {"1": "Viewer"}

    # Partner: WR surplus (3 startable incl. the elite), starter hole at RB.
    mvt += [
        _mva2("p_wr1", "Elite WR", "WR", 950), _mva2("p_wr2", "P WR2", "WR", 500),
        _mva2("p_wr3", "P WR3", "WR", 480), _mva2("p_rb1", "P RB1", "RB", 250),
    ]
    rosters.append({"roster_id": "2", "players": ["p_wr1", "p_wr2", "p_wr3", "p_rb1"]})
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
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE"],
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


# ── Overpay packages are never suggested ──────────────────────────────────────

def _build_overpay_ctx():
    """The Gibbs-for-two-WR1s shape: the viewer holds two elite WRs (~720 each,
    Lamb/London analogues) plus a third startable WR, and two startable RBs (so
    RB is no starter-gap need, only a ceiling want). The partner holds an elite
    RB (1055, the Gibbs analogue) plus RB depth, and a starter hole at WR. The
    only package the viewer can offer is both elite WRs (1435) for the 1055 RB:
    a 36% overpay at 0.735 fairness. It must not be suggested."""
    mvt = [
        _mv("v_wr1", "Elite WR1", "WR", 720), _mv("v_wr2", "Elite WR2", "WR", 715),
        _mv("v_wr3", "V WR3", "WR", 500),
        _mv("v_rb1", "V RB1", "RB", 770), _mv("v_rb2", "V RB2", "RB", 500),
        _mv("v_qb1", "V QB1", "QB", 1500), _mv("v_te1", "V TE1", "TE", 1000),
    ]
    rosters = [{"roster_id": "1",
                "players": ["v_wr1", "v_wr2", "v_wr3", "v_rb1", "v_rb2", "v_qb1", "v_te1"]}]
    roster_map = {"1": "Viewer"}

    # Partner: elite-RB surplus, starter hole at WR, sub-bar QB/TE so the RB is
    # the only package-trade candidate.
    mvt += [
        _mv("p_rb1", "Stud RB", "RB", 1055), _mv("p_rb2", "P RB2", "RB", 600),
        _mv("p_rb3", "P RB3", "RB", 500), _mv("p_wr1", "P WR1", "WR", 300),
        _mv("p_qb1", "P QB1", "QB", 300), _mv("p_te1", "P TE1", "TE", 300),
    ]
    rosters.append({"roster_id": "2",
                    "players": ["p_rb1", "p_rb2", "p_rb3", "p_wr1", "p_qb1", "p_te1"]})
    roster_map["2"] = "Team 2"

    for i in range(3, 11):  # balanced filler: no needs, no surpluses
        mvt += [_mv(f"f{i}_rb1", f"F{i} RB1", "RB", 1500), _mv(f"f{i}_rb2", f"F{i} RB2", "RB", 1200),
                _mv(f"f{i}_wr1", f"F{i} WR1", "WR", 1500), _mv(f"f{i}_wr2", f"F{i} WR2", "WR", 1300),
                _mv(f"f{i}_qb1", f"F{i} QB1", "QB", 1100), _mv(f"f{i}_te1", f"F{i} TE1", "TE", 1000)]
        rosters.append({"roster_id": str(i),
                        "players": [f"f{i}_rb1", f"f{i}_rb2", f"f{i}_wr1", f"f{i}_wr2",
                                    f"f{i}_qb1", f"f{i}_te1"]})
        roster_map[str(i)] = f"Team {i}"

    return {
        "rosters": rosters, "model_value_table": mvt, "roster_map": roster_map,
        "picks_by_roster": {},
        "standings_map": {r["roster_id"]: {"wins": 5, "losses": 5} for r in rosters},
        "rookie_rankings": [], "league_type": "1qb",
        "roster_positions": _LINEUP,
    }


def test_two_wr1s_for_one_rb1_overpay_is_not_suggested(monkeypatch):
    """Regression: the engine once suggested giving two ~720 WRs (1435) for a
    1055 RB (0.735 fairness, a 36% overpay). Anything past ~25% lopsided must
    never surface, so this partner is dropped, not suggested."""
    monkeypatch.setattr(
        cb, "build_team_gm_context",
        lambda ctx, rid: {"team_name": "Viewer", "direction": "balanced"},
    )
    # Neutralize the consolidation ceiling so the test isolates the fairness
    # gate (the function re-imports it per call, so patching the source binds).
    import utils.player_tiers as pt
    monkeypatch.setattr(pt, "consolidate_target_allowed", lambda a, b: True)

    res = cb.build_trade_suggestions_context(_build_overpay_ctx(), "1")
    assert res is not None
    names = [t["name"] for p in res["top_partners"] for t in (p.get("targets_they_have") or [])]
    assert "Stud RB" not in names, (
        "a 36%-overpay package (1435 of WR value for a 1055 RB) must not be suggested"
    )
    assert "Team 2" not in [p["team_name"] for p in res["top_partners"]]
# ---------------------------------------------------------------------------
# Cheapest-sufficient construction: the engine must not default to
# "their best player for your best players". It targets the cheapest partner
# player who actually solves the need and pays with the cheapest fair package
# from bench surplus, never gutting the starting lineup.
# ---------------------------------------------------------------------------

def _build_cheapest_ctx():
    """Viewer needs an RB (one 300-value body). Partner has RB surplus with a
    1500 RB1 and a 700 RB2; viewer has WR surplus (1000/900/800). Fillers push
    the 1500 RB out of elite range so both RBs are tier-allowed; the engine
    must still CHOOSE the cheaper one."""
    mvt = [
        _mv("v_wr1", "Viewer WR1", "WR", 1000),
        _mv("v_wr2", "Viewer WR2", "WR", 900),
        _mv("v_wr3", "Viewer WR3", "WR", 800),
        _mv("v_rb1", "Viewer RB1", "RB", 300),
        _mv("v_qb1", "Viewer QB1", "QB", 1500),
        _mv("v_te1", "Viewer TE1", "TE", 1000),
        _mv("t2_rb1", "Partner RB1", "RB", 1500),
        _mv("t2_rb2", "Partner RB2", "RB", 700),
        _mv("t2_wr1", "Partner WR1", "WR", 400),
        _mv("t2_qb1", "Partner QB1", "QB", 1100),
        _mv("t2_te1", "Partner TE1", "TE", 1000),
    ]
    for i, val in enumerate([2100, 2000, 1900, 1800, 1700, 1600]):
        mvt.append(_mv(f"frb{i}", f"Filler RB{i}", "RB", val))
    rosters = [
        {"roster_id": "1",
         "players": ["v_wr1", "v_wr2", "v_wr3", "v_rb1", "v_qb1", "v_te1"]},
        {"roster_id": "2",
         "players": ["t2_rb1", "t2_rb2", "t2_wr1", "t2_qb1", "t2_te1"]},
    ]
    return {
        "rosters": rosters,
        "model_value_table": mvt,
        "roster_map": {"1": "Viewer", "2": "Partner"},
        "picks_by_roster": {},
        "standings_map": {"1": {"wins": 5, "losses": 5},
                           "2": {"wins": 5, "losses": 5}},
        "rookie_rankings": [],
        "league_type": "1qb",
        "roster_positions": _LINEUP,
    }


def _run_cheapest(monkeypatch):
    monkeypatch.setattr(
        cb, "build_team_gm_context",
        lambda ctx, rid: {"team_name": "Viewer", "direction": "balanced"},
    )
    return cb.build_trade_suggestions_context(_build_cheapest_ctx(), "1")


def test_engine_targets_cheapest_need_filler_not_their_best(monkeypatch):
    res = _run_cheapest(monkeypatch)
    partners = res["top_partners"]
    assert len(partners) >= 1, "expected a partner suggestion"
    got = [t["name"] for t in partners[0]["targets_they_have"]]
    assert got == ["Partner RB2"], f"should target the 700 RB2, not the 1500 RB1: {got}"
    assert partners[0]["fairness"] >= 0.80


def test_engine_pays_from_bench_never_guts_starters(monkeypatch):
    res = _run_cheapest(monkeypatch)
    give_names = [t["name"] for t in res["top_partners"][0]["targets_viewer_sends"]]
    assert give_names == ["Viewer WR3"], f"should offer only the bench WR3: {give_names}"


def _build_package_cheapest_ctx():
    """No gap needs. Viewer has WR surplus; partner has two TE upgrades
    (900 and 650) over the viewer's 500 TE."""
    mvt = [
        _mv("v_wr1", "Viewer WR1", "WR", 1000),
        _mv("v_wr2", "Viewer WR2", "WR", 900),
        _mv("v_wr3", "Viewer WR3", "WR", 800),
        _mv("v_rb1", "Viewer RB1", "RB", 700),
        _mv("v_rb2", "Viewer RB2", "RB", 650),
        _mv("v_qb1", "Viewer QB1", "QB", 1500),
        _mv("v_te1", "Viewer TE1", "TE", 500),
        _mv("v_te2", "Viewer TE2", "TE", 200),
        _mv("t2_te1", "Partner TE1", "TE", 900),
        _mv("t2_te2", "Partner TE2", "TE", 650),
        _mv("t2_wr1", "Partner WR1", "WR", 400),
        _mv("t2_qb1", "Partner QB1", "QB", 1100),
        _mv("t2_rb1", "Partner RB1", "RB", 700),
        _mv("t2_rb2", "Partner RB2", "RB", 650),
    ]
    rosters = [
        {"roster_id": "1", "players": ["v_wr1", "v_wr2", "v_wr3", "v_rb1",
                                      "v_rb2", "v_qb1", "v_te1", "v_te2"]},
        {"roster_id": "2", "players": ["t2_te1", "t2_te2", "t2_wr1", "t2_qb1",
                                      "t2_rb1", "t2_rb2"]},
    ]
    return {
        "rosters": rosters,
        "model_value_table": mvt,
        "roster_map": {"1": "Viewer", "2": "Partner"},
        "picks_by_roster": {},
        "standings_map": {"1": {"wins": 5, "losses": 5},
                           "2": {"wins": 5, "losses": 5}},
        "rookie_rankings": [],
        "league_type": "1qb",
        "roster_positions": _LINEUP,
    }


def _run_package_cheapest(monkeypatch):
    monkeypatch.setattr(
        cb, "build_team_gm_context",
        lambda ctx, rid: {"team_name": "Viewer", "direction": "balanced"},
    )
    return cb.build_trade_suggestions_context(_build_package_cheapest_ctx(), "1")


def test_package_path_targets_cheapest_genuine_upgrade(monkeypatch):
    res = _run_package_cheapest(monkeypatch)
    partners = res["top_partners"]
    assert len(partners) >= 1, "expected a package suggestion"
    p = partners[0]
    got = [t["name"] for t in p["targets_they_have"]]
    assert got == ["Partner TE2"], f"should target the 650 TE2, not the 900 TE1: {got}"
    give_names = [t["name"] for t in p["targets_viewer_sends"]]
    assert give_names == ["Viewer WR3"], f"should fund from the bench WR3: {give_names}"
    assert p["fairness"] >= 0.80
    assert p["is_package_trade"] is True
