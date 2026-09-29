"""ScoreZone branding guard: the old "RedZone" product brand must not appear in
user-facing surfaces after the 2026-09-29 rename.

Allows the intentional leftovers:
  * ``redzone_plays`` Postgres table / index names (existing data, no migration)
  * ``redzone_scores`` notification catalog key and ``redzone_td:`` dedupe
    prefix (existing subscriptions / stored keys)
  * ``_isInRedZone`` (descriptive: inside the 20-yard line, not the brand)
  * the legacy ``/redzone`` page redirect + old API route aliases
  * descriptive red-zone stat plumbing (redzone_touches, fetch_season_redzone_stats)
"""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# Files whose rendered output (or notification copy) a user can see.
_USER_FACING = [
    _ROOT / "app.py",
    _ROOT / "static" / "app.js",
    _ROOT / "static" / "scorezone.js",
    _ROOT / "static" / "dashboard.css",
    _ROOT / "static" / "player_modal.js",
    _ROOT / "static" / "sw.js",
    _ROOT / "dashboard_services" / "pages" / "dashboard_page.py",
    _ROOT / "dashboard_services" / "pages" / "weekly_hub_page.py",
    _ROOT / "dashboard_services" / "pages" / "nfl_teams_page.py",
    _ROOT / "dashboard_services" / "pages" / "recap_page.py",
    _ROOT / "dashboard_services" / "changelog.py",
    _ROOT / "utils" / "push_notifications.py",
    _ROOT / "routes" / "user_pages_bp.py",
]

# Standalone brand words. Lowercase "redzone" is allowed below only for the
# documented compat identifiers; title/upper case must be gone everywhere.
_BRAND_RE = re.compile(r"\b(RedZone|Redzone|REDZONE)\b")

# Substrings that are intentionally kept (compat keys, table names, the
# descriptive helper, legacy aliases). A flagged line is excused only when it
# contains one of these.
_ALLOWED_SUBSTRINGS = (
    "_isInRedZone",  # descriptive: yard-line check, not the brand
    "page_redzone_legacy",  # legacy /redzone -> /scorezone redirect
    '"/<platform>/<int:season>/<league_id>/redzone"',
    '"/api/<platform>/<int:season>/<league_id>/redzone-data"',
    '"/api/<platform>/<int:season>/<league_id>/redzone-player"',
    '"/api/redzone/moments"',
    "renamed ScoreZone",  # the legacy-redirect comment itself
)


def _flagged_lines(path: Path) -> list[str]:
    bad = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if _BRAND_RE.search(line) and not any(s in line for s in _ALLOWED_SUBSTRINGS):
            bad.append(f"{path.name}:{i}: {line.strip()[:100]}")
    return bad


def test_no_redzone_brand_in_user_facing_sources():
    bad: list[str] = []
    for path in _USER_FACING:
        assert path.exists(), f"guard target moved: {path}"
        bad.extend(_flagged_lines(path))
    assert not bad, "old RedZone branding still user-visible:\n" + "\n".join(bad)


def test_scorezone_brand_present_in_key_surfaces():
    app_src = (_ROOT / "app.py").read_text(encoding="utf-8")
    assert 'render_page("BR ScoreZone"' in app_src
    assert '"/<platform>/<int:season>/<league_id>/scorezone"' in app_src
    js = (_ROOT / "static" / "scorezone.js").read_text(encoding="utf-8")
    assert "ScoreZone" in js
    pn = (_ROOT / "utils" / "push_notifications.py").read_text(encoding="utf-8")
    assert "ScoreZone" in pn
    # Stored keys stay on the old names so existing subscriptions keep working.
    assert '"redzone_scores"' in pn
    assert "redzone_td:" in pn


def test_legacy_redzone_urls_still_routed():
    app_src = (_ROOT / "app.py").read_text(encoding="utf-8")
    assert '"/<platform>/<int:season>/<league_id>/redzone"' in app_src
    assert '"/api/<platform>/<int:season>/<league_id>/redzone-data"' in app_src
    assert '"/api/<platform>/<int:season>/<league_id>/redzone-player"' in app_src
    moments = (_ROOT / "routes" / "user_pages_bp.py").read_text(encoding="utf-8")
    assert '"/api/redzone/moments"' in moments
