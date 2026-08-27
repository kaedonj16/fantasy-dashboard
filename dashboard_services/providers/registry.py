"""Single source of truth for fantasy provider discovery and metadata."""
from __future__ import annotations

from functools import lru_cache

from .base import ProviderNotFoundError


def normalize_platform(platform: str | None, *, default: str | None = "sleeper") -> str:
    value = str(platform or "").strip().lower()
    if not value and default is not None:
        value = default
    return value


@lru_cache(maxsize=None)
def get_provider(platform: str | None):
    key = normalize_platform(platform)
    if key == "sleeper":
        from .adapters import SleeperProvider
        return SleeperProvider()
    if key == "espn":
        from .adapters import ESPNProvider
        return ESPNProvider()
    if key == "yahoo":
        from .adapters import YahooProvider
        return YahooProvider()
    if key == "mfl":
        from .mfl_api import MFLProvider
        return MFLProvider()
    if key == "fleaflicker":
        from .fleaflicker_api import FleaflickerProvider
        return FleaflickerProvider()
    raise ProviderNotFoundError(f"Unknown fantasy provider: {key or '<empty>'}")


def provider_keys() -> frozenset[str]:
    return frozenset({"sleeper", "espn", "yahoo", "mfl", "fleaflicker"})


def get_provider_capabilities(platform: str) -> frozenset[str]:
    return get_provider(platform).metadata.capabilities


def get_provider_metadata():
    return tuple(get_provider(key).metadata for key in sorted(provider_keys()))
