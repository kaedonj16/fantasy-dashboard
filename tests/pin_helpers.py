"""Canonical pins for the service-worker cache version and changelog lookups.

Why this module exists
----------------------
Historically ~16 test files each hard-pinned the exact SW cache string
(``br-fantasy-vNNN``) and ~9 test files pinned changelog entries by
positional index (``CHANGELOG[23]``). Every UI PR bumps the SW version and
prepends a changelog entry, so every such PR had to edit all of those test
files, and parallel PRs collided or failed CI on stale pins.

This module is now the ONLY place that carries either kind of pin:

* ``EXPECTED_SW_CACHE_NAME`` below is the single kept exact SW pin. A
  census of the repo (2026-10-28) confirmed production code carries the
  version in exactly one place, ``static/sw.js`` (``CACHE_NAME``); there
  is no template, JS bundle, or sidecar copy to stay in sync with. So the
  legitimately-synced location for a version bump is THIS line: a bump PR
  edits ``static/sw.js`` plus this one line, instead of ~16 test files.
  Every former pin site calls :func:`assert_sw_cache_current`, which
  derives the actual version from ``static/sw.js`` at test time and
  asserts it agrees with the canonical pin. Bumping ``sw.js`` alone
  (forgetting this line) still fails every one of those tests, exactly
  the protection the scattered pins provided.

* Changelog entries are looked up by content via
  :func:`changelog_entry` (unique text substrings, plus tag/link when a
  substring repeats), never by positional index. Prepending an entry
  shifts every index but no content, so converted tests need no edits.
  The one genuine positional behaviour, "CHANGELOG[0] is the newest
  entry" (``app._notify_changelog_on_startup`` broadcasts ``CHANGELOG[0]``
  and the API serves the list newest-first), is asserted once, in
  ``tests/test_sw_changelog_pins.py``.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The single kept exact pin. Bump this line together with CACHE_NAME in
# static/sw.js; no other test file may carry the literal version.
EXPECTED_SW_CACHE_NAME = "br-fantasy-v145"

_CACHE_NAME_RE = re.compile(r"const CACHE_NAME = '(br-fantasy-v(\d+))';")


def sw_cache_name(sw_text: str | None = None) -> str:
    """Derive the SW cache name from static/sw.js at test time.

    Asserts there is exactly one CACHE_NAME definition, so a stray second
    version string in sw.js cannot silently diverge from the one the
    install/activate handlers actually use.
    """
    if sw_text is None:
        sw_text = (ROOT / "static" / "sw.js").read_text(encoding="utf-8")
    matches = _CACHE_NAME_RE.findall(sw_text)
    assert len(matches) == 1, (
        f"expected exactly one CACHE_NAME definition in static/sw.js, found {len(matches)}"
    )
    return matches[0][0]


def sw_cache_version(sw_text: str | None = None) -> int:
    """The numeric SW cache version derived from static/sw.js."""
    name = sw_cache_name(sw_text)
    return int(name.rsplit("-v", 1)[1])


def assert_sw_cache_current(sw_text: str | None = None) -> str:
    """Consistency assertion replacing every former exact SW pin.

    Derives the version from static/sw.js and asserts:
    * it equals the canonical pin above (an unsynced bump fails here);
    * the install handler opens CACHE_NAME and the activate handler
      deletes every cache that is not CACHE_NAME, so the derived name is
      the one that actually purges old caches at runtime.
    Returns the derived cache name.
    """
    if sw_text is None:
        sw_text = (ROOT / "static" / "sw.js").read_text(encoding="utf-8")
    name = sw_cache_name(sw_text)
    assert name == EXPECTED_SW_CACHE_NAME, (
        f"static/sw.js CACHE_NAME is {name!r}, canonical pin is "
        f"{EXPECTED_SW_CACHE_NAME!r}: if the bump is intentional, update "
        "EXPECTED_SW_CACHE_NAME in tests/pin_helpers.py (that one line is "
        "the whole pin sweep now)"
    )
    assert "caches.open(CACHE_NAME)" in sw_text
    assert "k !== CACHE_NAME" in sw_text
    return name


def assert_sw_cache_bumped_past(floor_version: int, sw_text: str | None = None) -> int:
    """Assert the derived SW version is strictly past ``floor_version``.

    For tests whose stated invariant was always a floor ("bumped past
    v51"), not equality with the version current at authoring time.
    """
    version = sw_cache_version(sw_text)
    assert version > floor_version, (
        f"SW cache version v{version} is not past v{floor_version}"
    )
    return version


def changelog_entry(*substrings: str, tag: str | None = None, link: str | None = None):
    """Find a changelog entry by content, never by positional index.

    Returns the first (newest) entry whose text contains every substring
    and, when given, whose tag/link match. Callers must pass a substring
    combination that identifies exactly one entry; the tests' original
    assertions (date/tag/link/text fragments) are kept after the lookup.
    Fails with AssertionError when no entry matches, so a removed or
    reworded announcement still fails its test.
    """
    from dashboard_services.changelog import CHANGELOG

    for entry in CHANGELOG:
        text = entry.get("text", "")
        if any(s not in text for s in substrings):
            continue
        if tag is not None and entry.get("tag") != tag:
            continue
        if link is not None and entry.get("link") != link:
            continue
        return entry
    raise AssertionError(
        f"no changelog entry matches substrings={substrings!r} tag={tag!r} link={link!r}"
    )
