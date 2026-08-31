"""Approximate injury return planner (roadmap R07)."""
from utils.injury_plan import injury_plan, resolve_weeks_out, status_weeks_band


def test_status_weeks_band():
    assert status_weeks_band("IR") == 6.0
    assert status_weeks_band("OUT") == 1.0
    assert status_weeks_band("Q") == status_weeks_band("QUESTIONABLE")
    assert status_weeks_band("ACTIVE") is None


def test_resolve_weeks_prefers_espn():
    weeks, src = resolve_weeks_out(status="IR", espn_weeks=2.5)
    assert weeks == 2.5 and src == "espn"
    weeks, src = resolve_weeks_out(status="OUT", espn_weeks=None)
    assert weeks == 1.0 and src == "status"


def test_injury_plan_verdicts():
    assert injury_plan(status=None) is None
    mon = injury_plan(status="OUT", espn_weeks=0.5, player_value=100)
    assert mon["verdict"] == "Monitor"
    assert mon["approximate"] is True

    stash = injury_plan(status="OUT", espn_weeks=2.0, player_value=100)
    assert stash["verdict"] == "Stash"

    drop = injury_plan(status="OUT", espn_weeks=2.0, player_value=10)
    assert drop["verdict"] == "Drop candidate"

    ir = injury_plan(status="IR", espn_weeks=5.0, player_value=50, has_open_ir_slot=True)
    assert ir["verdict"] == "IR"
    assert "approx" in ir["reason"].lower() or ir["approximate"]


def test_player_modal_renders_return_plan_badge():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "static" / "player_modal.js").read_text()
    assert "return_plan" in src
    assert "Approximate" in src or "approx" in src
