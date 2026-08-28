"""Guards the trade-eval AI wrapper's call contract.

The "Analyze" button (POST /api/trade-eval) renders the AI take via app's
get_trade_ai_analysis, a thin wrapper over the renderer. The wrapper once fell
one argument behind the renderer: the endpoint passed opponent_roster_id but the
wrapper didn't accept it, so every call raised TypeError, got swallowed by the
endpoint's try/except, and the UI showed "No AI take yet". This pins the wrapper
to the renderer so that drift can't recur silently.

Needs the full Flask stack (app import); skipped in the pure suite.
"""
import inspect

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


def test_wrapper_signature_covers_renderer(offline_client):
    import app
    import dashboard_services.ai.renderer as rnd

    wrapper_params = set(inspect.signature(app.get_trade_ai_analysis).parameters)
    renderer_params = set(inspect.signature(rnd.get_trade_ai_analysis).parameters)
    # The wrapper must accept every argument the renderer does (it forwards them).
    missing = renderer_params - wrapper_params
    assert not missing, f"wrapper is missing renderer params: {missing}"
    assert "opponent_roster_id" in wrapper_params
    assert "scoring_type" in wrapper_params


def test_wrapper_forwards_opponent_roster_id(offline_client, monkeypatch):
    import app
    import dashboard_services.ai.renderer as rnd

    captured = {}

    def _fake(ctx, viewer_roster_id, viewer_side, side_a, side_b,
              opponent_roster_id="", scoring_type="dynasty"):
        captured["opponent_roster_id"] = opponent_roster_id
        captured["scoring_type"] = scoring_type
        return "<ai-take>"

    monkeypatch.setattr(rnd, "get_trade_ai_analysis", _fake)

    # The exact call shape the endpoint uses - must not raise and must forward.
    out = app.get_trade_ai_analysis(
        ctx={}, viewer_roster_id="1", viewer_side="a",
        side_a={}, side_b={}, opponent_roster_id="7",
        scoring_type="redraft",
    )
    assert out == "<ai-take>"
    assert captured["opponent_roster_id"] == "7"
    assert captured["scoring_type"] == "redraft"


def test_trade_eval_endpoint_forwards_scoring_type_to_analyst(offline_client):
    import app

    src = inspect.getsource(app.api_trade_eval)
    start = src.index("get_trade_ai_analysis(")
    chunk = src[start:start + 800]
    assert "scoring_type=scoring_type" in chunk


@pytest.mark.parametrize(
    ("viewer_side", "expected_gets", "expected_gives", "expected_delta"),
    [
        ("a", 2000.0, 1200.0, 800.0),
        ("b", 1200.0, 2000.0, -800.0),
    ],
)
def test_trade_ai_payload_uses_selected_side_as_assets_received(
    offline_client,
    monkeypatch,
    viewer_side,
    expected_gets,
    expected_gives,
    expected_delta,
):
    import dashboard_services.ai.renderer as rnd

    captured = {}
    monkeypatch.setattr(
        rnd,
        "build_team_gm_context",
        lambda _ctx, _rid: {"team_name": "Viewer", "direction": "balanced"},
    )
    monkeypatch.setattr(rnd, "ai_available", lambda: True)
    monkeypatch.setattr(rnd, "load_cached_ai_text", lambda _key: None)
    monkeypatch.setattr(rnd, "save_cached_ai_text", lambda _key, _html: None)

    def fake_generate(payload):
        captured.update(payload)
        return {
            "verdict": "ACCEPT",
            "summary": "Correct perspective.",
            "helps": [],
            "risks": [],
            "counter": "",
            "confidence": "high",
        }

    monkeypatch.setattr(rnd, "generate_trade_ai_result", fake_generate)

    side_a = {"assets": [], "pick_ids": [], "effective_total": 2000.0}
    side_b = {"assets": [], "pick_ids": [], "effective_total": 1200.0}
    rnd.get_trade_ai_analysis(
        ctx={"players": {}, "rosters": []},
        viewer_roster_id="1",
        viewer_side=viewer_side,
        side_a=side_a,
        side_b=side_b,
    )

    trade = captured["trade"]
    assert trade["viewer_gets"]["effective_total"] == expected_gets
    assert trade["viewer_gives"]["effective_total"] == expected_gives
    assert trade["market_delta"] == expected_delta
    assert captured["scoring_type"] == "dynasty"
    assert captured["league_format"]["picks_tradable"] is True


def _stub_trade_ai(monkeypatch, captured):
    import dashboard_services.ai.renderer as rnd

    monkeypatch.setattr(
        rnd,
        "build_team_gm_context",
        lambda _ctx, _rid: {"team_name": "Viewer", "direction": "balanced"},
    )
    monkeypatch.setattr(rnd, "ai_available", lambda: True)
    monkeypatch.setattr(rnd, "load_cached_ai_text", lambda _key: None)
    monkeypatch.setattr(rnd, "save_cached_ai_text", lambda _key, _html: None)

    def fake_generate(payload):
        captured.update(payload)
        return {
            "verdict": "COUNTER",
            "summary": "Need a sweetener.",
            "helps": [],
            "risks": [],
            "counter": "Ask for a bench WR.",
            "confidence": "medium",
        }

    monkeypatch.setattr(rnd, "generate_trade_ai_result", fake_generate)


def test_trade_ai_payload_forwards_redraft_and_hides_picks(offline_client, monkeypatch):
    import dashboard_services.ai.renderer as rnd

    captured = {}
    _stub_trade_ai(monkeypatch, captured)

    side_a = {"assets": [], "pick_ids": ["2026_1_01"], "effective_total": 800.0}
    side_b = {"assets": [], "pick_ids": [], "effective_total": 1200.0}
    rnd.get_trade_ai_analysis(
        ctx={"players": {}, "rosters": []},
        viewer_roster_id="1",
        viewer_side="a",
        side_a=side_a,
        side_b=side_b,
        scoring_type="redraft",
    )

    assert captured["scoring_type"] == "redraft"
    assert captured["league_format"]["scoring_type"] == "redraft"
    assert captured["league_format"]["picks_tradable"] is False
    assert "cannot be traded" in captured["league_format"]["note"].lower()
    assert captured["trade"]["pick_prospects"] == {}
