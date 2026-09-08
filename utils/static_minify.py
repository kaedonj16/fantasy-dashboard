"""Boot-time minify of first-party static JS/CSS.

Same pattern as app.js / dashboard.css: write a sibling ``*.min.*`` when
rjsmin/rcssmin succeed, keyed by a ``.src`` hash sidecar so the next boot is a
no-op until the source changes. Callers serve whatever ``served_name`` returns.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

_CACHE: dict[str, str] = {}


def static_hash(filename: str) -> str:
    """8-char MD5 of a file under ``static/``, or ``"0"`` if it is missing."""
    path = STATIC_DIR / filename
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()[:8]
    except OSError:
        return "0"


def served_name(filename: str) -> str:
    """Filename under ``/static/`` to serve (minified when minify succeeded)."""
    cached = _CACHE.get(filename)
    if cached is not None:
        return cached
    out = _minify(filename)
    _CACHE[filename] = out
    return out


def _minify(filename: str) -> str:
    src = STATIC_DIR / filename
    if not src.exists():
        return filename
    suffix = src.suffix
    if suffix not in (".js", ".css"):
        return filename
    out_name = f"{src.stem}.min{suffix}"
    out = STATIC_DIR / out_name
    meta = STATIC_DIR / f"{out_name}.src"
    try:
        src_bytes = src.read_bytes()
        src_hash = hashlib.md5(src_bytes).hexdigest()
    except OSError:
        return filename
    try:
        if out.exists() and meta.exists() and meta.read_text().strip() == src_hash:
            return out_name
    except OSError:
        pass
    text = src_bytes.decode("utf-8")
    # Guard against a broken minifier emitting near-empty output. Typical
    # rjsmin/rcssmin shrink is 20–40%; anything below 40%/30% of source is
    # treated as a failure and we keep the original.
    min_ratio = 0.4 if suffix == ".js" else 0.3
    try:
        if suffix == ".js":
            import rjsmin
            minified = rjsmin.jsmin(text)
        else:
            import rcssmin
            minified = rcssmin.cssmin(text)
        if not minified or len(minified) < len(text) * min_ratio:
            logger.info("[%s] minify sanity check failed, serving original", filename)
            return filename
        out.write_text(minified, encoding="utf-8")
        meta.write_text(src_hash, encoding="utf-8")
        logger.info(
            "[%s] minified: %d KB -> %d KB",
            filename, len(text) // 1024, len(minified) // 1024,
        )
        return out_name
    except Exception as exc:
        logger.info("[%s] minify unavailable, serving unminified: %s", filename, exc)
        return filename
