"""Guards the consolidate flex/spot-upgrade boost in archetype_engine.

Consolidate suggestions used to surface only top-top players: scoring was
dominated by raw value and loaded teams got an availability boost on rival
studs, so a mid-tier target that genuinely upgrades a weak starting slot
(a FLEX WR30 -> WR14 type move) never made the slate. Now a "starter"-category
target worth >= 1.15x the viewer's weakest starter at its position gets a 1.30
ranking lift (the same magnitude as the affordable-stud availability boost) and
a why-line note naming the slot. Elites are excluded: they already get the
stud boost and this must not double-count.

Conventions follow tests/test_archetype_suggestions_e2e.py: a hand-built
league context, offline stubs, and result-cache clearing. The engine path needs
pandas (lazy app import), so the whole module skips cleanly without it.
"""
import pytest

pytest.importorskip("pandas")

from dashboard_services import archetype_engine as ae


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Keep the pipeline fully offline: stub the HTTP layer the engine's lazy
    app import would otherwise use (which blocks on network retries)."""
    import dashboard_services.api as api

    def _fake_fetch_json(path, timeout=25, retries=3):
        if path == "/state/nfl":
            return {"season": "2026", "week": 0, "leg": 0,
                    "season_type": "off", "display_week": 1,
                    "season_start_date": "2026-09-10"}
        return {}

    monkeypatch.setattr(api, "fetch_json", _fake_fetch_json)


@pytest.fixture(autouse=True)
def _clear_result_cache():
    """The engine memoizes finished results per request key; clear it around
    each test so runs with different monkeypatches stay independent."""
    ae._RESULT_CACHE.clear()
    yield
    ae._RESULT_CACHE.clear()


# ── _weakest_starters ─────────────────────────────────────────────────────────

def _v(pairs):
    """values_by_id from (pid, position, value) triples."""
    return {pid: {"position": pos, "value": val, "name": pid}
            for pid, pos, val in pairs}


_LINEUP = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"]
_ROSTER = ["rb1", "rb2", "wr1", "wr2", "wr3", "te1", "qb1"]
_VALS = _v([
    ("rb1", "RB", 800), ("rb2", "RB", 250),
    ("wr1", "WR", 1200), ("wr2", "WR", 1000), ("wr3", "WR", 1050),
    ("te1", "TE", 400), ("qb1", "QB", 500),
])


def test_weakest_starters_assigns_positional_slots_then_flex():
    # WR slots take the two best WRs (1200, 1050); the 1000-WR falls to FLEX
    # and is the weakest WR starter. RB2 (250) is the weakest RB starter.
    got = ae._weakest_starters(_ROSTER, _VALS, _LINEUP)
    assert got["RB"] == ("rb2", 250.0, "RB2")
    assert got["WR"] == ("wr2", 1000.0, "FLEX")
    assert got["TE"] == ("te1", 400.0, "TE1")
    assert "QB" not in got  # only RB/WR/TE are tracked


def test_weakest_starters_respects_restricted_flex_eligibility():
    vals = _v([
        ("rb1", "RB", 800), ("rb2", "RB", 250),
        ("wr1", "WR", 1200), ("wr2", "WR", 1000),
        ("te_bench", "TE", 900),  # high value but TE can't fill an RB_WR slot
        ("rb_bench", "RB", 200),
    ])
    players = ["rb1", "rb2", "wr1", "wr2", "te_bench", "rb_bench"]
    lineup = ["QB", "RB", "RB", "WR", "WR", "TE", "RB_WR"]
    got = ae._weakest_starters(players, vals, lineup)
    # The RB_WR slot takes the 200-RB (best remaining RB/WR-eligible); the
    # 900-TE holds the TE positional slot and never the flex slot.
    assert got["RB"] == ("rb_bench", 200.0, "FLEX")
    assert got["WR"] == ("wr2", 1000.0, "WR2")
    assert got["TE"] == ("te_bench", 900.0, "TE1")


def test_weakest_starters_handles_bare_wrrb_alias():
    # "WRRB" (bare Sleeper-style name) is a RB/WR flex like "RB_WR".
    vals = _v([("rb1", "RB", 800), ("wr1", "WR", 1000), ("rb_bench", "RB", 300)])
    got = ae._weakest_starters(
        ["rb1", "wr1", "rb_bench"], vals, ["QB", "RB", "WR", "TE", "WRRB"])
    assert got["RB"] == ("rb_bench", 300.0, "FLEX")


def test_weakest_starters_ignores_superflex_and_empty_inputs():
    vals = _v([("qb1", "QB", 900), ("rb1", "RB", 800)])
    got = ae._weakest_starters(["qb1", "rb1"], vals,
                               ["QB", "RB", "SUPER_FLEX", "BN"])
    # SUPER_FLEX is QB-eligible so it never feeds the RB/WR/TE map.
    assert got == {"RB": ("rb1", 800.0, "RB1")}
    assert ae._weakest_starters([], {}, []) == {}
    assert ae._weakest_starters(["rb1"], _v([("rb1", "RB", 800)]), None) == {}


# ── _slot_upgrade_multiplier ──────────────────────────────────────────────────

_WS_RB2 = ("rb2", 250.0, "RB2")


def test_multiplier_applies_at_or_above_115x():
    assert ae._slot_upgrade_multiplier("starter", 600.0, _WS_RB2) == 1.30
    # Exactly 1.15x qualifies (boundary inclusive).
    assert ae._slot_upgrade_multiplier("starter", 287.5, _WS_RB2) == 1.30


def test_multiplier_rejects_below_115x():
    assert ae._slot_upgrade_multiplier("starter", 287.49, _WS_RB2) == 1.0
    assert ae._slot_upgrade_multiplier("starter", 200.0, _WS_RB2) == 1.0


def test_multiplier_rejects_elite_targets():
    # Elites already get the affordable-stud availability boost; the slot
    # bonus must not double-count them, however big the upgrade.
    assert ae._slot_upgrade_multiplier("elite", 2000.0, _WS_RB2) == 1.0


def test_multiplier_rejects_other_categories_and_missing_weakest():
    assert ae._slot_upgrade_multiplier("flex", 600.0, _WS_RB2) == 1.0
    assert ae._slot_upgrade_multiplier("depth", 600.0, _WS_RB2) == 1.0
    assert ae._slot_upgrade_multiplier("starter", 600.0, None) == 1.0
    assert ae._slot_upgrade_multiplier("starter", 600.0, ("rb2", 0.0, "RB2")) == 1.0


# ── _build_why slot-upgrade sentence ──────────────────────────────────────────

def _why_target(**kw):
    t = {"name": "Jaxon Smith-Njigba", "position": "WR", "age": 24,
         "value": 600.0, "redraft_value": 480.0, "partner_name": "Team Two"}
    t.update(kw)
    return t


def test_why_appends_slot_upgrade_sentence():
    t = _why_target(slot_upgrade={"slot": "FLEX", "from_pid": "wr2",
                                  "from_name": "Alec Pierce"})
    why = ae._build_why(t, "consolidate", 0.0, 0.01)
    assert why.endswith("Upgrades your FLEX: Alec Pierce -> Jaxon Smith-Njigba.")
    assert "Consolidating around" in why


def test_why_omits_slot_upgrade_sentence_without_tag():
    why = ae._build_why(_why_target(), "consolidate", 0.0, 0.01)
    assert "Upgrades your" not in why


# ── End to end: the boost in the real consolidate pipeline ────────────────────

def _P(pid, name, pos, val, age=26, team="FA"):
    return {"id": pid, "name": name, "position": pos, "team": team,
            "age": age, "value": val, "sf_value": val,
            "redraft_value_1qb": val * 0.8, "redraft_value_sf": val * 0.8,
            "pos_rank_label": f"{pos}1", "rank_change_7d": 0}


def _seed_ctx():
    """4-team league plus FA padding. The padding shapes positional ranks so
    Mid RB (600) and Big WR (720) land in the "starter" tier (rank > elite
    cutoff 6) instead of "elite"; FAs are never trade targets themselves.

    Viewer weakest starters: RB2 = 250, WR(FLEX) = 1000.
      - Mid RB (600): 600 >= 1.15 * 250 -> gets the slot-upgrade lift.
      - Big WR (720): 720 < 1.15 * 1000 -> higher raw value, no lift.
      - Stud RB (2000, rank 1): elite -> no lift.
    """
    table = [
        # Viewer (roster 1)
        _P("v_rb1", "Viewer RB1", "RB", 800, 25, "PHI"),
        _P("v_rb2", "Viewer RB2", "RB", 250, 29, "PHI"),
        _P("v_wr1", "Viewer WR1", "WR", 1200, 26, "PHI"),
        _P("v_wr2", "Viewer WR2", "WR", 1000, 27, "PHI"),
        _P("v_wr3", "Viewer WR3", "WR", 1050, 24, "PHI"),
        _P("v_te1", "Viewer TE1", "TE", 400, 28, "PHI"),
        _P("v_qb1", "Viewer QB1", "QB", 500, 30, "PHI"),
        # Rival 2: holds the mid-tier RB target (their only RB)
        _P("t2_rb", "Mid RB", "RB", 600, 24, "DET"),
        _P("t2_wr1", "Rival2 WR1", "WR", 950, 27, "DET"),
        _P("t2_te1", "Rival2 TE1", "TE", 300, 26, "DET"),
        # Rival 3: holds the higher-raw-value WR target (their only WR)
        _P("t3_wr", "Big WR", "WR", 720, 25, "CIN"),
        _P("t3_rb1", "Rival3 RB1", "RB", 700, 28, "CIN"),
        _P("t3_te1", "Rival3 TE1", "TE", 320, 26, "CIN"),
        # Rival 4: holds an elite RB (rank 1)
        _P("t4_rb", "Stud RB", "RB", 2000, 23, "BUF"),
        _P("t4_wr1", "Rival4 WR1", "WR", 900, 27, "BUF"),
        _P("t4_te1", "Rival4 TE1", "TE", 280, 29, "BUF"),
    ]
    for i, v in enumerate([1500, 1400, 1300, 1200, 1100, 1000, 950]):
        table.append(_P(f"fa_rb{i}", f"FA RB{i}", "RB", v, 30))
    for i, v in enumerate([1600, 1500, 1400, 1300, 1250, 1220, 1180]):
        table.append(_P(f"fa_wr{i}", f"FA WR{i}", "WR", v, 30))
    rosters = [
        {"roster_id": 1, "players": ["v_rb1", "v_rb2", "v_wr1", "v_wr2",
                                    "v_wr3", "v_te1", "v_qb1"]},
        {"roster_id": 2, "players": ["t2_rb", "t2_wr1", "t2_te1"]},
        {"roster_id": 3, "players": ["t3_wr", "t3_rb1", "t3_te1"]},
        {"roster_id": 4, "players": ["t4_rb", "t4_wr1", "t4_te1"]},
    ]
    return {
        "rosters": rosters,
        "roster_map": {1: "Viewer", 2: "Team Two", 3: "Team Three",
                       4: "Team Four"},
        "standings_map": {1: 5, 2: 2, 3: 8, 4: 9},
        "model_value_table": table,
        "picks_by_roster": {},
        "settings": {"playoff_week_start": 15},
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
    }


def _run_consolidate(monkeypatch, league_id):
    """Run consolidate while recording the pre-slate (final, target) ranking
    the 1.30 lift is applied to. Returns (suggestions, slate)."""
    slate = {}

    _orig = ae._select_varied_slate

    def _spy(scored, max_targets=8, max_per_pos=3):
        for final, t in scored:
            slate[t["name"]] = (final, bool(t.get("slot_upgrade")))
        return _orig(scored, max_targets, max_per_pos)

    monkeypatch.setattr(ae, "_select_varied_slate", _spy)
    out = ae.get_archetype_suggestions(
        archetype="consolidate", platform="sleeper", league_id=league_id,
        season=2026, viewer_roster_id="1", league_type="1qb", league_size=10,
        ctx=_seed_ctx())
    return out.get("suggestions") or [], slate


def test_slot_upgrade_lift_is_exactly_130x(monkeypatch):
    """The mid-tier slot upgrade's pre-slate score is exactly 1.30x what the
    same pipeline produces with the lift disabled (counterfactual run)."""
    _, slate = _run_consolidate(monkeypatch, "flexlift")
    boosted, tagged = slate["Mid RB"]
    assert tagged is True

    monkeypatch.setattr(ae, "_slot_upgrade_multiplier", lambda *a: 1.0)
    _, slate_off = _run_consolidate(monkeypatch, "flexlift-off")
    unboosted, tagged_off = slate_off["Mid RB"]
    assert tagged_off is False
    assert boosted == pytest.approx(unboosted * 1.30)


def test_slot_upgrade_outranks_higher_raw_value_in_slate(monkeypatch):
    """Mid RB (600, lifted) makes the slate above Big WR (720, no lift):
    consolidate can upgrade a flex spot, not just chase raw value."""
    _, slate = _run_consolidate(monkeypatch, "flexorder")
    assert "Mid RB" in slate and "Big WR" in slate
    assert slate["Mid RB"][0] > slate["Big WR"][0]
    assert slate["Big WR"][1] is False  # 720 < 1.15 * weakest WR (1000)


def test_slot_upgrade_sentence_only_on_lifted_targets(monkeypatch):
    """The why-line names the upgraded slot for lifted targets only: the
    elite (Stud RB) and the non-upgrade (Big WR) carry no such sentence."""
    suggestions, _ = _run_consolidate(monkeypatch, "flexwhy")
    by_name = {}
    for s in suggestions:
        by_name.setdefault(s["name"], s)

    mid = by_name["Mid RB"]
    assert "Upgrades your RB2: Viewer RB2 -> Mid RB." in mid["why"]

    big = by_name["Big WR"]
    assert "Upgrades your" not in big["why"]

    stud_rows = [s for s in suggestions if s["name"] == "Stud RB"]
    assert stud_rows, "expected the elite target to surface"
    for s in stud_rows:
        assert "Upgrades your" not in s["why"]
