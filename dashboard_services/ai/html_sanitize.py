"""Allowlist sanitizer for AI-rendered HTML before it hits innerHTML.

Stdlib-only (html.parser) -- strips script/iframe/object/embed, event-handler
attributes, and javascript: URLs. Keeps the small tag set used by
``dashboard_services.ai.renderer``.
"""
from __future__ import annotations

from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import urlparse

# Tags the AI renderer / fallbacks emit. Anything else is dropped (contents kept).
# ``details`` / ``summary`` are here because the Front Office board and
# Reference sections are built on them: without the tags, the Evidence
# toggles rendered as loose unstyled text (Kaedon, 2026-11-05).
_ALLOWED_TAGS = frozenset({
    "p", "div", "span", "strong", "em", "b", "i", "ul", "ol", "li", "br",
    "h3", "h4", "table", "thead", "tbody", "tr", "th", "td", "a", "button",
    "details", "summary",
})
_VOID_TAGS = frozenset({"br"})
# Global attrs safe on any allowed tag. ``href`` is validated separately on <a>.
_ALLOWED_ATTRS = frozenset({"class", "id"})
# Inline style is how several deterministic renderers carry data-driven
# presentation (verdict stamp colors, grade-bar widths, avatar colors).
# It survives only as whitelisted properties with inert values: no
# url(), no expression(), nothing that can load or execute.
_SAFE_STYLE_PROPS = frozenset({
    "color", "background", "background-color", "border", "border-color",
    "border-radius", "width", "min-width", "max-width", "height",
    "min-height", "max-height", "margin", "margin-top", "margin-bottom",
    "margin-left", "margin-right", "padding", "padding-top",
    "padding-bottom", "padding-left", "padding-right", "opacity",
    "display", "font-weight", "font-size", "font-style", "text-align",
    "text-transform", "text-decoration", "line-height", "letter-spacing",
    "white-space", "vertical-align", "flex", "flex-basis", "flex-wrap",
    "gap", "align-items", "justify-content", "cursor", "position",
    "top", "left", "right", "bottom", "z-index", "overflow",
    "box-shadow", "float", "clear",
})
_BANNED_STYLE_TOKENS = (
    "url", "expression", "javascript", "behavior", "-moz-binding",
    "<", ">", "\\", "@", "/*",
)


def _safe_style(value: str) -> str | None:
    decls = []
    for part in (value or "").split(";"):
        if ":" not in part:
            continue
        prop, val = part.split(":", 1)
        prop = prop.strip().lower()
        val = val.strip()
        if prop not in _SAFE_STYLE_PROPS or not val:
            continue
        if any(tok in val.lower() for tok in _BANNED_STYLE_TOKENS):
            continue
        decls.append(f"{prop}:{val}")
    return ";".join(decls) if decls else None


def _safe_href(value: str) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None
    # Block protocol-relative and javascript:/data: etc.
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    if scheme in ("http", "https"):
        return raw
    # Allow same-origin relative paths only (no scheme, no //evil).
    if not scheme and not raw.startswith("//") and not raw.lower().startswith("javascript:"):
        if raw.startswith("/") or raw.startswith("#") or raw.startswith("?"):
            return raw
    return None


class _AllowlistParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []
        self._skip_depth = 0  # inside a fully-stripped element (script etc.)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_l = tag.lower()
        if self._skip_depth:
            if tag_l not in _VOID_TAGS:
                self._skip_depth += 1
            return
        if tag_l in ("script", "iframe", "object", "embed", "link", "meta", "style", "base"):
            self._skip_depth = 1
            return
        if tag_l not in _ALLOWED_TAGS:
            return
        safe_attrs: list[str] = []
        for name, val in attrs:
            if name is None:
                continue
            n = name.lower()
            if n.startswith("on"):
                continue
            if tag_l == "a" and n == "href":
                href = _safe_href(val or "")
                if href is not None:
                    safe_attrs.append(f'href="{_escape_attr(href)}"')
                continue
            if tag_l == "details" and n == "open":
                safe_attrs.append("open")
                continue
            if n == "style":
                cleaned = _safe_style(val or "")
                if cleaned is not None:
                    safe_attrs.append(f'style="{_escape_attr(cleaned)}"')
                continue
            # data-* attributes are inert data hooks the site's own JS
            # delegation reads (player modal, report jump buttons);
            # stripping them silently disabled those controls.
            if n.startswith("data-") and val is not None:
                safe_attrs.append(f'{n}="{_escape_attr(val)}"')
                continue
            if n in _ALLOWED_ATTRS and val is not None:
                safe_attrs.append(f'{n}="{_escape_attr(val)}"')
        attr_s = (" " + " ".join(safe_attrs)) if safe_attrs else ""
        if tag_l in _VOID_TAGS:
            self._out.append(f"<{tag_l}{attr_s}>")
        else:
            self._out.append(f"<{tag_l}{attr_s}>")

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        if self._skip_depth:
            if tag_l not in _VOID_TAGS:
                self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag_l in _ALLOWED_TAGS and tag_l not in _VOID_TAGS:
            self._out.append(f"</{tag_l}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Treat as start; void tags don't need a close.
        self.handle_starttag(tag, attrs)

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._out.append(_escape_text(data))

    def handle_entityref(self, name: str) -> None:
        if self._skip_depth:
            return
        self._out.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._skip_depth:
            return
        self._out.append(f"&#{name};")

    def result(self) -> str:
        return "".join(self._out)


def _escape_text(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _escape_attr(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def sanitize_ai_html(raw: str | None) -> str:
    """Return allowlisted HTML safe for assignment to ``innerHTML``."""
    if not raw:
        return ""
    parser = _AllowlistParser()
    try:
        parser.feed(str(raw))
        parser.close()
    except Exception:
        # On parse failure, escape everything rather than pass raw through.
        return _escape_text(str(raw))
    return parser.result()


def sanitize_ai_html_fragments(parts: Iterable[str | None]) -> str:
    return "".join(sanitize_ai_html(p) for p in parts)
