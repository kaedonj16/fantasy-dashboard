"""Season Wrapped ends on a shareable recap card.

The deck's final slide mirrors the Share card: a champion block over the same
highlight rows. Built from the same season summary as the rest of the deck, so
it agrees with what the Share button draws.

Skipped when Flask/pandas aren't installed; runs in CI with the full stack.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


def test_wrapped_deck_ends_with_recap_card(offline_client):
    import app
    from dashboard_services.pages import history_page as H

    ctx = app._build_tour_mock_history_ctx()
    ctx["summary"] = {
        "top_scorer_value": 1720.4, "top_scorer_team": "Rebuild from Hell", "top_scorer_avg": 122.9,
        "biggest_blowout_margin": 84.2, "biggest_blowout": "Butter Boys",
        "closest_margin": 0.6, "closest_matchup": "Jiggy vs Odell",
        "champion": "Rebuild from Hell", "champion_record": "11-3", "runner_up": "Butter Boys",
    }
    with app.app.test_request_context("/"):
        html = H.render_history_wrapped_overlay(ctx, 2025)

    # The recap slide exists and is the finale (after the champion slide).
    assert "data-kind='recap'" in html
    assert html.rfind("data-kind='recap'") > html.rfind("data-kind='champion'")
    # It carries the champion block and the same highlight rows the Share draws.
    assert "wrapped-recap-champ-name" in html and "Rebuild from Hell" in html
    assert "wrapped-record-badge" in html and "11-3" in html
    assert "TOP SCORER" in html
