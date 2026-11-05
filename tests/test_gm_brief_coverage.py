"""Wave 3 PR 2: QB coverage map + both-rosters bye panel.

Spec: Wave 3 sections 3.3, 3.4, 4, 5 and 8 PR 2.
"""
import pytest

pytest.importorskip("pandas")

from dashboard_services import gm_brief as gb

WEEKS = list(range(9, 14))  # projections exist for 9..13; 14 is missing


def _player(team, pos, name, bye=None, status=""):
    row = {"full_name": name, "team": team, "position": pos,
           "injury_status": status}
    if bye:
        row["byeWeek"] = bye
    return row


def _sf_ctx(**over):
    players = {
        "qb_a": _player("AAA", "QB", "QB Alpha", bye=10),
        "qb_b": _player("BBB", "QB", "QB Bravo", bye=13),
        "qb_c": _player("CCC", "QB", "QB Charlie", status="IR"),
        "rb1": _player("DDD", "RB", "RB One"),
        "rb2": _player("EEE", "RB", "RB Two"),
        "wr1": _player("FFF", "WR", "WR One"),
        "wr2": _player("GGG", "WR", "WR Two"),
        "te1": _player("HHH", "TE", "TE One"),
    }
    proj = {}
    for wk in WEEKS:
        proj[wk] = {"qb_a": 18.0, "qb_b": 17.0, "rb1": 12.0, "rb2": 11.0,
                    "wr1": 13.0, "wr2": 12.0, "te1": 9.0}
    ctx = {
        "current_week": 9,
        "season": 2026,
        "current_season": 2026,
        "players": players,
        "proj_by_week": {wk: {"projections": p} for wk, p in proj.items()},
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE",
                             "SUPER_FLEX"],
        "league_rules": {"source": "user_confirmed", "playoff_teams": 8,
                         "playoff_week_start": 15},
        "rosters": [{"roster_id": "1", "players": list(players),
                     "starters": list(players)}],
        "roster_map": {"1": "Viewer Team"},
        "role_states": {pid: {"state": "starter", "reason": "",
                              "week": 9} for pid in players},
    }
    ctx.update(over)
    return ctx


def test_sf_coverage_holes_weeks_10_and_13_with_causes():
    brief = gb.build_gm_brief(_sf_ctx(), "1", [], now=0)
    cov = brief["qb_coverage"]
    assert cov is not None
    assert cov["requirements"]["QB"] == 2
    holes = {(h["week"], h["position"]) for h in cov["holes"]}
    assert (10, "QB") in holes
    assert (13, "QB") in holes
    hole10 = next(h for h in cov["holes"] if h["week"] == 10)
    causes = " ".join(hole10["causes"])
    assert "bye: QB Alpha" in causes       # Milton-style bye cause
    assert "IR: QB Charlie" in causes      # and the IR cause: both named
    hole13 = next(h for h in cov["holes"] if h["week"] == 13)
    assert "bye: QB Bravo" in " ".join(hole13["causes"])
    # Week 9 is fully covered at QB.
    wk9 = next(w for w in cov["weeks"] if w["week"] == 9)
    assert wk9["qb"]["count"] == 2
    assert wk9["qb"]["hole"] is False
    assert brief["layers"]["roles"] is True
    assert cov["label"] is None


def test_coverage_missing_week_is_named_not_interpolated():
    brief = gb.build_gm_brief(_sf_ctx(), "1", [], now=0)
    cov = brief["qb_coverage"]
    assert cov["missing_weeks"] == [14]
    assert all(w["week"] != 14 for w in cov["weeks"])


def test_1qb_requirement_is_one():
    ctx = _sf_ctx(roster_positions=["QB", "RB", "RB", "WR", "WR", "TE",
                                    "FLEX"])
    # Only qb_a is available in Week 10 in this variant: qb_b joins
    # the IR'd qb_c on the shelf, and qb_a's own bye moves off Week 10.
    ctx["players"]["qb_b"]["byeWeek"] = 10
    del ctx["players"]["qb_a"]["byeWeek"]
    brief = gb.build_gm_brief(ctx, "1", [], now=0)
    cov = brief["qb_coverage"]
    assert cov["requirements"]["QB"] == 1
    assert not any(h["week"] == 10 and h["position"] == "QB"
                   for h in cov["holes"])


def test_role_absent_mode_is_labeled_role_unconfirmed():
    ctx = _sf_ctx()
    ctx["gm_role_states"] = None  # the Wave 2 layer is absent
    del ctx["role_states"]
    brief = gb.build_gm_brief(ctx, "1", [], now=0)
    cov = brief["qb_coverage"]
    assert cov is not None
    assert cov["label"] == "role-unconfirmed"
    assert brief["layers"]["roles"] is False
    # Same holes from projections + byes alone.
    assert any(h["week"] == 10 and h["position"] == "QB"
               for h in cov["holes"])


# ---------------------------------------------------------------------------
# bye panel (3.4)
# ---------------------------------------------------------------------------

def _bye_ctx(**over):
    players = {
        "v_qb": _player("AAA", "QB", "Viewer QB"),
        "v_wr1": _player("VVV", "WR", "Viewer WR One", bye=11),
        "v_wr2": _player("WWW", "WR", "Viewer WR Two", bye=11),
        "o_qb": _player("BBB", "QB", "Opp QB"),
        "o_rb1": _player("XXX", "RB", "Opp RB One", bye=11),
        "o_rb2": _player("YYY", "RB", "Opp RB Two", bye=11),
    }
    ctx = {
        "current_week": 9,
        "season": 2026,
        "current_season": 2026,
        "players": players,
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE"],
        "rosters": [
            {"roster_id": "1", "players": ["v_qb", "v_wr1", "v_wr2"],
             "starters": ["v_qb", "v_wr1", "v_wr2"]},
            {"roster_id": "2", "players": ["o_qb", "o_rb1", "o_rb2"],
             "starters": ["o_qb", "o_rb1", "o_rb2"]},
        ],
        "roster_map": {"1": "Viewer Team", "2": "Rival Team"},
        "gm_standings": [
            {"roster_id": "1", "team_name": "Viewer Team", "wins": 5,
             "losses": 3, "pf": 900.0},
            {"roster_id": "2", "team_name": "Rival Team", "wins": 4,
             "losses": 4, "pf": 850.0},
        ],
        "matchups_by_week": {
            wk: [{"left": {"roster_id": "1"}, "right": {"roster_id": "2"}}]
            for wk in (9, 10, 11)
        },
    }
    ctx.update(over)
    return ctx


def test_bye_panel_both_sides_losses_and_crunch():
    panel = gb.build_bye_panel(_bye_ctx(), "1")
    assert panel is not None
    wk11 = next(w for w in panel["weeks"] if w["week"] == 11)
    assert wk11["viewer"]["by_pos"] == {"WR": 2}
    assert wk11["viewer"]["players_by_pos"]["WR"] == [
        "Viewer WR One", "Viewer WR Two"]
    assert wk11["opponent"]["team_name"] == "Rival Team"
    assert wk11["opponent"]["by_pos"] == {"RB": 2}
    assert wk11["opponent"]["players_by_pos"]["RB"] == [
        "Opp RB One", "Opp RB Two"]
    assert wk11["crunch"] is True
    assert panel["has_crunch"] is True
    assert "you lose 2" in wk11["net"]


def test_bye_panel_no_crunch_flag_when_no_tight_week():
    ctx = _bye_ctx()
    # One WR on bye in Week 11 is a loss but not a crunch (two start).
    del ctx["players"]["v_wr2"]["byeWeek"]
    del ctx["players"]["o_rb2"]["byeWeek"]
    panel = gb.build_bye_panel(ctx, "1")
    assert panel is not None
    assert panel["has_crunch"] is False


# ---------------------------------------------------------------------------
# renderer strips (section 4): flag-gated, hidden when nothing to say
# ---------------------------------------------------------------------------

def _render_ctx(ctx, monkeypatch):
    import app as appmod

    action = {"kind": "lineup", "tag": "Lineup", "action": "Start someone",
              "why": "Projects more.", "cta_label": "Fix lineup",
              "cta_url": "#", "score": 90, "priority": "high"}
    monkeypatch.setattr(appmod, "_next_steps_lineup_actions",
                        lambda c, r: [action])
    monkeypatch.setattr(appmod, "_next_steps_waiver_actions",
                        lambda c, r, m: [])
    monkeypatch.setattr(appmod, "_next_steps_trade_actions",
                        lambda c, r: [])
    monkeypatch.setattr(appmod, "_next_steps_roster_actions",
                        lambda c, r: [])
    monkeypatch.setattr(appmod, "_next_steps_bench_check_actions",
                        lambda c, r, w: [])
    monkeypatch.setattr(appmod, "_next_steps_bench_check_note",
                        lambda c, r, w: "")
    return appmod._render_next_steps_queue(
        ctx, "1", [], season=2026, current_week=9)


def test_render_omits_bye_panel_without_crunch(monkeypatch):
    pytest.importorskip("flask")
    ctx = _bye_ctx()
    del ctx["players"]["v_wr2"]["byeWeek"]
    del ctx["players"]["o_rb2"]["byeWeek"]
    html_out = _render_ctx(ctx, monkeypatch)
    assert "Bye outlook" not in html_out


def test_render_shows_bye_panel_with_crunch_and_quiet_coverage_line(monkeypatch):
    pytest.importorskip("flask")
    ctx = _bye_ctx()
    # The crunch comes from the opponent's RB room; the viewer's own
    # byes move off so the viewer coverage map stays clean.
    del ctx["players"]["v_wr1"]["byeWeek"]
    del ctx["players"]["v_wr2"]["byeWeek"]
    # Give the viewer a clean QB coverage map through Week 14.
    ctx["players"].update({
        "v_qb2": _player("CCC", "QB", "Viewer QB Two"),
        "v_rb1": _player("DDD", "RB", "Viewer RB One"),
        "v_rb2": _player("EEE", "RB", "Viewer RB Two"),
        "v_te1": _player("FFF", "TE", "Viewer TE One"),
    })
    ctx["rosters"][0]["players"].extend(["v_qb2", "v_rb1", "v_rb2",
                                         "v_te1"])
    ctx["roster_positions"] = ["QB", "RB", "RB", "WR", "WR", "TE",
                               "SUPER_FLEX"]
    proj = {wk: {"v_qb": 18.0, "v_qb2": 17.0, "v_wr1": 12.0,
                 "v_wr2": 11.0, "v_rb1": 12.0, "v_rb2": 11.0,
                 "v_te1": 9.0} for wk in range(9, 15)}
    ctx["proj_by_week"] = {wk: {"projections": p} for wk, p in proj.items()}
    ctx["league_rules"] = {"source": "user_confirmed",
                           "playoff_week_start": 15, "playoff_teams": 8}
    html_out = _render_ctx(ctx, monkeypatch)
    assert "Bye outlook" in html_out
    assert "QB coverage set through Week 14" in html_out
