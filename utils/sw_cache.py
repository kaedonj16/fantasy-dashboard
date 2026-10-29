"""Content-derived service-worker cache name (no manual version bumps).

Why this exists
---------------
Until 2026-10-29 every PR that shipped JS or CSS had to hand-bump
``CACHE_NAME`` in ``static/sw.js`` (``br-fantasy-vNNN``) plus the synced
pin in ``tests/pin_helpers.py``. Two files edited by nearly every PR
meant nearly every pair of parallel PRs conflicted on them, and the
version numbers cascaded across the queue (v141-v148 claimed in a
single day).

The cache name is now DERIVED from the bytes it exists to invalidate:

* the ``static/sw.js`` template itself (its fetch/precache logic),
* the precache asset set, parsed out of the template's own
  ``PRECACHE_URLS`` (offline shell + brand PNGs), and
* the first-party JS/CSS sources the boot pipeline minifies and pages
  load (``ASSET_FILES`` below: app.js, player_modal.js, dashboard.css,
  critical-home.css, the per-page bundles, ...).

:func:`compute_cache_name` hashes those bytes into
``br-fantasy-h<12 hex>``. At boot, app.py calls :func:`ensure_sw_js`,
which writes the served worker to ``static/sw.built.js`` (gitignored,
the same boot-build pattern as ``app.min.js`` / ``app-features.min.js``)
with the derived name substituted into the template's ``CACHE_NAME``
line, gated by a ``.src`` sidecar so an unchanged asset set is a no-op.
The ``/sw.js`` routes serve the built file; the tracked template is
only the fallback for a boot where the build declined, and its literal
``CACHE_NAME`` is a sane last-manual-version name, never edited again.

Consequences for PR authors: NONE. Ship JS/CSS/backend changes as
usual; when a cached asset's bytes change, the served worker's bytes
change with them, browsers pick up the new worker, and its activate
handler (which deletes every cache that is not the current name)
purges the old cache. When no cached asset changes, the name is
byte-stable, so backend-only deploys do not roll users' caches.

Escape hatch: set the ``SW_CACHE_NAME`` env var to force a specific
cache name (any ``[A-Za-z0-9._-]`` string up to 80 chars). The served
worker then carries that name verbatim until the var is removed. This
is for the day a poisoned cache must be force-rolled by hand; it is
not part of any routine workflow.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

TEMPLATE_FILENAME = "sw.js"
BUILT_FILENAME = "sw.built.js"
SIDECAR_FILENAME = "sw.built.js.src"

CACHE_NAME_ENV = "SW_CACHE_NAME"
_NAME_SAFE_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
_CACHE_NAME_DEF_RE = re.compile(r"const CACHE_NAME = '([^']+)';")
_PRECACHE_BLOCK_RE = re.compile(r"PRECACHE_URLS\s*=\s*\[(.*?)\]", re.DOTALL)
_PRECACHE_ENTRY_RE = re.compile(r"'(/static/[^']+)'")

# First-party sources the boot minify pipeline turns into the bundles
# pages load (app.py's _ensure_* / utils.static_minify.served_name),
# plus the offline shell and web manifest. The precache set is NOT
# repeated here: it is parsed out of the template itself, so editing
# PRECACHE_URLS in sw.js re-derives the name from the right files.
ASSET_FILES = (
    "app.js",
    "player_modal.js",
    "dashboard.css",
    "critical-home.css",
    "offline.html",
    "manifest.json",
    "analytics_labels.js",
    "cheat_sheet.js",
    "custom_selects.js",
    "draft_board_core.js",
    "draft_grade_curve.js",
    "draft_grade_team.js",
    "draft_room.js",
    "draft_room_wizard.js",
    "font-awesome.css",
    "icons.css",
    "keeper.js",
    "landing_lite.css",
    "paywall.css",
    "paywall.js",
    "pick_score.js",
    "player_page.js",
    "rankings.js",
    "scorezone.js",
    "seo_lite.css",
    "teams.js",
    "week_range.js",
)


def cache_name_override() -> str | None:
    """The SW_CACHE_NAME env override, or None when unset/unsafe."""
    raw = (os.environ.get(CACHE_NAME_ENV) or "").strip()
    if raw and _NAME_SAFE_RE.match(raw):
        return raw
    if raw:
        logger.warning(
            "[sw.js] ignoring %s=%r: not a safe cache name", CACHE_NAME_ENV, raw
        )
    return None


def _template_text(static_dir: Path) -> str:
    return (static_dir / TEMPLATE_FILENAME).read_text(encoding="utf-8")


def _precache_files(template_text: str) -> list[str]:
    """Filenames from the template's PRECACHE_URLS (query strings stripped)."""
    block = _PRECACHE_BLOCK_RE.search(template_text)
    if not block:
        return []
    files = []
    for entry in _PRECACHE_ENTRY_RE.findall(block.group(1)):
        name = entry[len("/static/"):].split("?", 1)[0]
        if name and name not in files:
            files.append(name)
    return files


def _asset_set(template_text: str) -> list[str]:
    """Ordered, de-duplicated files whose bytes feed the cache name."""
    ordered: list[str] = []
    for name in _precache_files(template_text) + list(ASSET_FILES):
        if name not in ordered:
            ordered.append(name)
    return ordered


def compute_cache_name(static_dir: Path = STATIC_DIR) -> str:
    """Derive the cache name from the template + cached asset bytes.

    The template's own CACHE_NAME value is normalised out before
    hashing (the name must not feed itself). Missing files hash as a
    fixed marker, so adding/removing a listed asset also rolls the
    name. Pure function of the tree: no env, no clock, no randomness.
    """
    template = _template_text(static_dir)
    normalised = _CACHE_NAME_DEF_RE.sub("const CACHE_NAME = '';", template, count=1)
    digest = hashlib.md5()  # noqa: S324 (cache key, not a security hash)
    digest.update(b"br-sw-cache-v1\0")
    digest.update(normalised.encode("utf-8"))
    for name in _asset_set(template):
        digest.update(b"\0file:")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update((static_dir / name).read_bytes())
        except OSError:
            digest.update(b"<missing>")
    return f"br-fantasy-h{digest.hexdigest()[:12]}"


def effective_cache_name(static_dir: Path = STATIC_DIR) -> str:
    """The name the served worker must carry (env override wins)."""
    return cache_name_override() or compute_cache_name(static_dir)


def build_sw_text(static_dir: Path = STATIC_DIR, name: str | None = None) -> str:
    """The template text with CACHE_NAME replaced by the effective name.

    Raises ValueError unless the template carries exactly one
    CACHE_NAME definition, so a stray second version string can never
    silently diverge from the one the handlers actually use.
    """
    template = _template_text(static_dir)
    matches = _CACHE_NAME_DEF_RE.findall(template)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one CACHE_NAME definition in static/sw.js, "
            f"found {len(matches)}"
        )
    return _CACHE_NAME_DEF_RE.sub(
        f"const CACHE_NAME = '{name or effective_cache_name(static_dir)}';",
        template,
        count=1,
    )


def ensure_sw_js(static_dir: Path = STATIC_DIR) -> str:
    """Write the served worker (sw.built.js) when the derived name moved.

    Same boot-build contract as the bundle builders in app.py: a
    ``.src`` sidecar stores the name the built file was generated for,
    so repeat boots with an unchanged asset set write nothing. Returns
    the filename the /sw.js routes should serve; on ANY failure the
    tracked template is served instead, so the site never breaks.
    """
    try:
        name = effective_cache_name(static_dir)
        built = static_dir / BUILT_FILENAME
        sidecar = static_dir / SIDECAR_FILENAME
        if (
            built.exists()
            and sidecar.exists()
            and sidecar.read_text(encoding="utf-8").strip() == name
        ):
            return BUILT_FILENAME
        built.write_text(build_sw_text(static_dir, name), encoding="utf-8")
        sidecar.write_text(name, encoding="utf-8")
        logger.info("[sw.js] built served worker with cache name %s", name)
        return BUILT_FILENAME
    except Exception as exc:  # build must never break the boot
        logger.info("[sw.js] auto cache-name build declined, serving template: %s", exc)
        return TEMPLATE_FILENAME


def served_sw_path(static_dir: Path = STATIC_DIR) -> Path:
    """Path of the worker to serve right now (built file once boot ran)."""
    built = static_dir / BUILT_FILENAME
    if built.exists() and (static_dir / SIDECAR_FILENAME).exists():
        return built
    return static_dir / TEMPLATE_FILENAME
