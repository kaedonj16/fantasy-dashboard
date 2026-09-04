"""Facade behavior for optional provider capabilities."""
import pytest

from dashboard_services.platform_api import get_bracket
from dashboard_services.providers.fleaflicker_api import FleaflickerProvider
from dashboard_services.providers.mfl_api import MFLProvider


@pytest.mark.parametrize("platform", ["fleaflicker", "mfl"])
def test_get_bracket_returns_empty_when_derivation_fails(platform, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("no bracket")
    if platform == "fleaflicker":
        monkeypatch.setattr(FleaflickerProvider, "get_bracket", boom)
    else:
        monkeypatch.setattr(MFLProvider, "get_bracket", boom)
    assert get_bracket(platform, "92916", "winners", 2026) == []
    assert get_bracket(platform, "92916", "losers", 2026) == []


def test_providers_derive_or_return_empty_not_raise(monkeypatch):
    flea = FleaflickerProvider()
    mfl = MFLProvider()
    monkeypatch.setattr(flea, "get_league", lambda *_a, **_k: {"settings": {}})
    monkeypatch.setattr(flea, "get_matchups", lambda *_a, **_k: [])
    monkeypatch.setattr(flea, "_playoff_seeds", lambda *_a, **_k: [])
    monkeypatch.setattr(mfl, "get_league", lambda *_a, **_k: {"settings": {}})
    monkeypatch.setattr(mfl, "get_matchups", lambda *_a, **_k: [])
    monkeypatch.setattr(mfl, "_playoff_seeds", lambda *_a, **_k: [])
    assert flea.get_bracket("1", 2026, "winners") == []
    assert mfl.get_bracket("1", 2026, "winners") == []
    assert flea.get_bracket("1", 2026, "losers") == []
    assert mfl.get_bracket("1", 2026, "losers") == []
