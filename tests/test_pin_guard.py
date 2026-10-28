"""Guard: the SW-version / changelog pin collision pattern must not return.

Before PR #2862, ~15 test files hard-pinned the exact service-worker
cache version and ~9 files addressed changelog entries by absolute
index, while every PR bumps the version and prepends an entry - so
parallel PRs conflicted with the tests (and with each other) by
construction. #2862 decoupled both: the single canonical SW pin is
``EXPECTED_SW_CACHE_NAME`` in ``tests/pin_helpers.py`` (every other
test derives the version from ``static/sw.js`` via the helpers), and
changelog checks find entries by content via ``changelog_entry(...)``;
the one consolidated newest-first assertion lives in
``tests/test_sw_changelog_pins.py``.

This guard scans every other test file's AST and fails if the old
pattern reappears:

* a hardcoded service-worker version literal (the ``br-fantasy-v``
  prefix followed by digits) in any string constant, or
* a positional changelog subscript (``CHANGELOG`` subscripted by an
  integer literal).

AST scanning (not raw text) keeps comments and docstrings - which may
legitimately *discuss* the old pattern - from tripping the guard.
The guard's own patterns are assembled from fragments so its source
never contains the literal text it polices, and it is exempt by name
alongside the two designated pin homes.
"""

import ast
import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

# The designated homes for the surviving pins, plus this guard itself.
EXEMPT = {"pin_helpers.py", "test_sw_changelog_pins.py", Path(__file__).name}

# Assembled from fragments: the literal pattern never appears in this file.
_SW_LITERAL_RE = re.compile("br-fantasy-" + "v" + r"\d+")


def _scan():
    """Yield (filename, lineno, kind, detail) for every violation."""
    for path in sorted(TESTS_DIR.glob("*.py")):
        if path.name in EXEMPT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if _SW_LITERAL_RE.search(node.value):
                    yield (path.name, node.lineno, "sw-version literal",
                           "hardcoded service-worker version; derive it via "
                           "tests/pin_helpers.py instead")
            elif isinstance(node, ast.Subscript):
                value = node.value
                slc = node.slice
                if (isinstance(value, ast.Name) and value.id == "CHANGELOG"
                        and isinstance(slc, ast.Constant)
                        and isinstance(slc.value, int)):
                    yield (path.name, node.lineno, "positional changelog pin",
                           "CHANGELOG subscripted by position; find the entry "
                           "by content via pin_helpers.changelog_entry(...)")


def _report(kind):
    hits = [(f, ln, d) for f, ln, k, d in _scan() if k == kind]
    return "\n".join(f"  {f}:{ln}: {d}" for f, ln, d in hits)


def test_no_hardcoded_sw_version_literals():
    report = _report("sw-version literal")
    assert not report, (
        "Hardcoded service-worker version literal(s) found outside "
        "tests/pin_helpers.py:\n" + report)


def test_no_positional_changelog_subscripts():
    report = _report("positional changelog pin")
    assert not report, (
        "Positional CHANGELOG subscript(s) found outside "
        "tests/test_sw_changelog_pins.py:\n" + report)
