"""Canonical pins for the service-worker cache name and changelog lookups.

Why this module exists
----------------------
Historically ~16 test files each hard-pinned the exact SW cache string
(``br-fantasy-vNNN``) and ~9 test files pinned changelog entries by
positional index (``CHANGELOG[23]``). Every UI PR bumped the SW version
and prepended a changelog entry, so every such PR had to edit all of
those test files, and parallel PRs collided or failed CI on stale pins.

The changelog half was decoupled by PR #2862 (content lookups via
:func:`changelog_entry`, below). The SW half is now decoupled at the
source: since the auto-cache-name change, nobody bumps a version at
all. The served worker's ``CACHE_NAME`` is DERIVED at boot from a hash
of the cached asset set (``utils/sw_cache.py``; the ``/sw.js`` routes
serve the boot-built ``static/sw.built.js``). So:

* ``EXPECTED_SW_CACHE_NAME`` below is not a literal pin a human keeps
  in sync; it is the derived name for the current tree, computed by
  the same function production uses. A frontend change moves the
  expectation and the served name together, with no test edits.
  It is evaluated lazily, on every access (PEP 562), never frozen at
  import time: app boot regenerates ``static/critical-home.css`` -- a
  hashed input -- AFTER building the served worker, so in a fresh
  checkout the tree a collection-time import would snapshot is not
  the tree the rest of the run sees. Deriving the expectation at
  assert time keeps both sides of every comparison on the same tree.
* Every former pin site calls :func:`assert_sw_cache_current`, which
  verifies the mechanism end to end: the template carries exactly one
  ``CACHE_NAME`` definition, the built (served) worker carries the
  derived name, and the install/activate handlers key off that name
  (activate deletes every cache that is not the current one, so a
  rolled name really does purge old caches).
* The ``SW_CACHE_NAME`` env override, when set, is the expected name
  instead; the derivation defers to it exactly like production does.

Changelog entries are still looked up by content via
:func:`changelog_entry` (unique text substrings, plus tag/link when a
substring repeats), never by positional index. The one genuine
positional behaviour, "CHANGELOG[0] is the newest entry"
(``app._notify_changelog_on_startup`` broadcasts ``CHANGELOG[0]`` and
the API serves the list newest-first), is asserted once, in
``tests/test_sw_changelog_pins.py``.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_CACHE_NAME_RE = re.compile(r"const CACHE_NAME = '([^']+)';")


def _sw_cache():
    from utils.sw_cache import build_sw_text, effective_cache_name

    return build_sw_text, effective_cache_name


# The derived expectation: whatever name the current tree's asset set
# hashes to (or the SW_CACHE_NAME override). Computed, never edited --
# and computed at access time, not import time (see the module
# docstring: boot's critical-CSS regeneration can move the tree
# mid-run, so a frozen snapshot can straddle it).
_build_sw_text, _effective_cache_name = _sw_cache()


def _expected_sw_cache_name() -> str:
    return _effective_cache_name(ROOT / "static")


def __getattr__(name: str):
    if name == "EXPECTED_SW_CACHE_NAME":
        return _expected_sw_cache_name()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def sw_cache_name(sw_text: str | None = None) -> str:
    """The CACHE_NAME carried by ``sw_text`` (default: the served worker).

    "Served worker" means the boot-built text (template + derived
    name), which is what the /sw.js routes hand to browsers; the raw
    template in static/sw.js only carries the fallback literal.
    Asserts there is exactly one CACHE_NAME definition, so a stray
    second version string cannot silently diverge from the one the
    install/activate handlers actually use.
    """
    if sw_text is None:
        sw_text = _build_sw_text(ROOT / "static")
    matches = _CACHE_NAME_RE.findall(sw_text)
    assert len(matches) == 1, (
        f"expected exactly one CACHE_NAME definition in the service "
        f"worker, found {len(matches)}"
    )
    return matches[0]


def assert_sw_cache_current(sw_text: str | None = None) -> str:
    """Mechanism assertion replacing every former exact SW pin.

    Verifies, against the repo tree (plus any provided worker text):
    * the template static/sw.js carries exactly one CACHE_NAME
      definition (its literal is only the boot fallback);
    * the built worker a browser would be served carries the derived
      name, ``EXPECTED_SW_CACHE_NAME``;
    * the install handler opens CACHE_NAME and the activate handler
      deletes every cache that is not CACHE_NAME, so the derived name
      is the one that actually purges old caches at runtime.
    Returns the derived cache name.
    """
    template = (ROOT / "static" / "sw.js").read_text(encoding="utf-8")
    assert len(_CACHE_NAME_RE.findall(template)) == 1, (
        "static/sw.js must carry exactly one CACHE_NAME definition"
    )
    served = _build_sw_text(ROOT / "static")
    name = sw_cache_name(served)
    expected = _expected_sw_cache_name()
    assert name == expected, (
        f"served CACHE_NAME is {name!r}, derived expectation is "
        f"{expected!r}: the served worker must be built "
        "from the current asset set (utils/sw_cache.py)"
    )
    for text in (template, served, sw_text):
        if text is None:
            continue
        assert "caches.open(CACHE_NAME)" in text
        assert "k !== CACHE_NAME" in text
    return name


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
