"""Facade behavior for optional provider capabilities."""
import pytest

from dashboard_services.platform_api import get_bracket
from dashboard_services.providers.base import UnsupportedCapabilityError
from dashboard_services.providers.fleaflicker_api import FleaflickerProvider
from dashboard_services.providers.mfl_api import MFLProvider


@pytest.mark.parametrize("platform", ["fleaflicker", "mfl"])
def test_get_bracket_returns_empty_when_unsupported(platform):
    assert get_bracket(platform, "92916", "winners", 2026) == []
    assert get_bracket(platform, "92916", "losers", 2026) == []


def test_providers_still_raise_when_called_directly():
    with pytest.raises(UnsupportedCapabilityError):
        FleaflickerProvider().get_bracket("1", 2026, "winners")
    with pytest.raises(UnsupportedCapabilityError):
        MFLProvider().get_bracket("1", 2026, "winners")
