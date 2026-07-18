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


def test_wrapper_forwards_opponent_roster_id(offline_client, monkeypatch):
    import app
    import dashboard_services.ai.renderer as rnd

    captured = {}

    def _fake(ctx, viewer_roster_id, viewer_side, side_a, side_b, opponent_roster_id=""):
        captured["opponent_roster_id"] = opponent_roster_id
        return "<ai-take>"

    monkeypatch.setattr(rnd, "get_trade_ai_analysis", _fake)

    # The exact call shape the endpoint uses - must not raise and must forward.
    out = app.get_trade_ai_analysis(
        ctx={}, viewer_roster_id="1", viewer_side="a",
        side_a={}, side_b={}, opponent_roster_id="7",
    )
    assert out == "<ai-take>"
    assert captured["opponent_roster_id"] == "7"
