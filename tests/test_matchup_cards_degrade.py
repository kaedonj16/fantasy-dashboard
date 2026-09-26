"""Regression: matchup_cards_last_week must degrade (not 503) when the
provider's scoreboard fetch fails, e.g. a Fleaflicker edge/WAF HTML 403."""
import sys
import types

import pytest

pd = pytest.importorskip("pandas")

from dashboard_services.providers.base import ProviderUnavailableError


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


@pytest.fixture()
def svc(monkeypatch):
    """Import dashboard_services.service with its heavy web-stack imports stubbed.

    Stubs are scoped via monkeypatch (never left in sys.modules), and the real
    service module is restored afterwards so other test files are unaffected.
    """
    monkeypatch.setitem(sys.modules, "flask",
                        _stub("flask", g=None, has_app_context=lambda: False))
    monkeypatch.setitem(sys.modules, "utils.utils",
                        _stub("utils.utils", safe_owner_name=lambda *a, **k: ""))
    monkeypatch.setitem(sys.modules, "dashboard_services.api",
                        _stub("dashboard_services.api",
                              avatar_url=lambda *a, **k: "",
                              get_nfl_state=lambda *a, **k: {},
                              avatar_from_users=lambda *a, **k: "",
                              team_avatar=lambda *a, **k: ""))
    monkeypatch.setitem(sys.modules, "dashboard_services.display_names",
                        _stub("dashboard_services.display_names",
                              public_owner_label=lambda *a, **k: ""))
    monkeypatch.setitem(sys.modules, "dashboard_services.matchups",
                        _stub("dashboard_services.matchups",
                              build_matchup_preview=lambda *a, **k: ""))
    platform_api = _stub("dashboard_services.platform_api",
                         get_matchups=lambda *a, **k: [],
                         get_transactions=lambda *a, **k: [])
    monkeypatch.setitem(sys.modules, "dashboard_services.platform_api", platform_api)
    monkeypatch.setitem(sys.modules, "dashboard_services.players",
                        _stub("dashboard_services.players",
                              build_roster_display_maps=lambda *a, **k: {}))
    monkeypatch.setitem(sys.modules, "dashboard_services.team_crest",
                        _stub("dashboard_services.team_crest",
                              team_crest_data_uri=lambda *a, **k: ""))
    monkeypatch.delitem(sys.modules, "dashboard_services.service", raising=False)
    import dashboard_services.service as svc_mod
    monkeypatch.setitem(sys.modules, "dashboard_services.service", svc_mod)
    return svc_mod, platform_api


def _week_df():
    return pd.DataFrame([{"week": 3, "points": 100.0}])


def test_matchup_cards_last_week_degrades_on_provider_unavailable(svc, monkeypatch):
    svc_mod, _platform_api = svc

    def boom(*a, **k):
        raise ProviderUnavailableError("Fleaflicker is temporarily unavailable.")

    monkeypatch.setattr(svc_mod, "get_matchups", boom)
    week, html_out, top = svc_mod.matchup_cards_last_week(
        "92916", _week_df(), {}, {}, [], [], "fleaflicker", "2026",
    )
    assert week == 3
    assert html_out == ""
    assert top == {}


def test_matchup_cards_last_week_reraises_unexpected_errors(svc, monkeypatch):
    svc_mod, _platform_api = svc

    def boom(*a, **k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(svc_mod, "get_matchups", boom)
    with pytest.raises(RuntimeError):
        svc_mod.matchup_cards_last_week(
            "92916", _week_df(), {}, {}, [], [], "fleaflicker", "2026",
        )
