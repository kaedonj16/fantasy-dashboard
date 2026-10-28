"""Guest landing demo payload: cached in the app.py wrapper.

_landing_demo_json used to rebuild the demo pools on every call (up to
3x per guest-home render). Non-empty builds are now cached on the
payload's input stamps; failures and empty pools are never cached.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


@pytest.fixture()
def landing(monkeypatch):
    import app as appmod
    import dashboard_services.landing_demos as ld

    appmod._LANDING_DEMO_CACHE["key"] = None
    appmod._LANDING_DEMO_CACHE["ts"] = 0.0
    appmod._LANDING_DEMO_CACHE["json"] = None
    monkeypatch.setattr(appmod, "get_nfl_state",
                        lambda *a, **k: {"season": "2026", "week": 7, "season_type": "reg"})
    counts = {"n": 0}
    payload = {"value_pairs": [{"a": 1}], "start_sit_pairs": [], "week_label": "Week 7"}

    def fake_build():
        counts["n"] += 1
        return dict(payload)

    monkeypatch.setattr(ld, "build_landing_demo_data", fake_build)
    return appmod, counts


def test_second_call_serves_cached_payload(landing):
    appmod, counts = landing
    first = appmod._landing_demo_json()
    second = appmod._landing_demo_json()
    assert counts["n"] == 1
    assert first == second
    assert '"value_pairs": [{"a": 1}]' in first


def test_empty_pools_and_failures_are_not_cached(landing, monkeypatch):
    import dashboard_services.landing_demos as ld

    appmod, counts = landing
    monkeypatch.setattr(ld, "build_landing_demo_data",
                        lambda: {"value_pairs": [], "start_sit_pairs": [], "week_label": "This week"})
    appmod._landing_demo_json()
    appmod._landing_demo_json()
    # Both calls rebuilt: an empty result means sources were unavailable.
    assert appmod._LANDING_DEMO_CACHE["json"] is None

    def boom():
        raise RuntimeError("nope")

    monkeypatch.setattr(ld, "build_landing_demo_data", boom)
    out = appmod._landing_demo_json()
    assert out == '{"value_pairs": [], "start_sit_pairs": [], "week_label": "This week"}'
    assert appmod._LANDING_DEMO_CACHE["json"] is None
