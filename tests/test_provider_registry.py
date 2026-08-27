import pytest

from dashboard_services.providers.base import FUTURE_PICKS, BRACKET, ProviderNotFoundError
from dashboard_services.providers.registry import get_provider, get_provider_capabilities


@pytest.mark.parametrize("value,key", [
    ("sleeper", "sleeper"), (" ESPN ", "espn"), ("YaHoO", "yahoo"),
    ("MFL", "mfl"), ("FleaFlicker", "fleaflicker"),
])
def test_registry_resolves_and_normalizes(value, key):
    assert get_provider(value).metadata.key == key


def test_registry_rejects_explicit_unknown_provider():
    with pytest.raises(ProviderNotFoundError):
        get_provider("not-real")


def test_capabilities_are_explicit():
    assert FUTURE_PICKS in get_provider_capabilities("mfl")
    assert BRACKET not in get_provider_capabilities("mfl")
    assert FUTURE_PICKS in get_provider_capabilities("fleaflicker")
    assert BRACKET not in get_provider_capabilities("fleaflicker")
