"""Shareable public links for Season Wrapped / Weekly Wrapped decks.

A share stores the *rendered* overlay HTML (slides + branding) plus the share
payload, so the public link keeps working even if the league is deleted and no
auth is needed to view it. The stored payload is exactly what the image Share
card already exposes (league name, week, highlights, slide contents) -- no
rosters, no user identity.

Overlay HTML is untrusted client input rendered verbatim on a public page, so
it is sanitized through a strict whitelist before storage (and again at render
time, which also covers shares minted before sanitization existed).

Payloads live in Postgres, not in process memory.
"""
from __future__ import annotations

import logging
import secrets
from html.parser import HTMLParser

logger = logging.getLogger(__name__)

_TABLES_READY = False

#: Links expire a year after creation; the public route 404s past expiry.
SHARE_TTL_SQL = "INTERVAL '1 year'"


#: Tags the Wrapped deck renderer actually emits (presentation only).
_SANITIZE_ALLOWED_TAGS = frozenset({
    "section", "div", "span", "p",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "i", "em", "strong", "b", "small", "br",
    "ul", "ol", "li", "img",
})

#: Tags whose content must be dropped entirely (active content / embeds).
_SANITIZE_DROP_TAGS = frozenset({
    "script", "style", "iframe", "object", "embed", "base", "link", "meta",
    "form", "input", "button", "textarea", "select", "option",
    "video", "audio", "source", "track", "canvas", "svg", "math",
    "noscript", "template", "slot",
})

#: Void elements never have end tags; dropping them must not touch the
#: drop-depth counter or everything after them would be swallowed.
_SANITIZE_VOID_TAGS = frozenset({
    "br", "img", "input", "link", "meta", "base", "source", "track",
})

#: URL schemes allowed in src attributes. Anything else (javascript:, data:
#: text/html, ...) is stripped.
_SANITIZE_SAFE_SCHEMES = ("http://", "https://", "data:image/")


class _OverlaySanitizer(HTMLParser):
    """Whitelist HTML sanitizer for Wrapped share overlays.

    Keeps presentation tags/attributes the deck renderer emits; drops active
    content, event-handler attributes, and unsafe URLs. Unknown tags are
    unwrapped (inner text kept); dangerous tags are dropped with content.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._out: list[str] = []
        self._drop_depth = 0

    def _emit(self, s: str) -> None:
        if self._drop_depth == 0:
            self._out.append(s)

    @staticmethod
    def _esc_attr(v: str) -> str:
        return (
            v.replace("&", "&amp;")
            .replace('"', "&quot;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def _clean_attrs(self, tag: str, attrs: list[tuple[str, str | None]]) -> str:
        bits = []
        for name, value in attrs:
            name = (name or "").lower()
            value = value or ""
            # No event handlers, no matter the tag.
            if name.startswith("on"):
                continue
            if name in ("class", "id", "title", "alt") or name.startswith("data-"):
                bits.append(f'{name}="{self._esc_attr(value)}"')
            elif name == "style":
                # Inline styles are presentation-only in modern browsers
                # (no JS execution via CSS); keep but quote safely.
                bits.append(f'style="{self._esc_attr(value)}"')
            elif name == "src" and tag == "img":
                low = value.strip().lower()
                if low.startswith(_SANITIZE_SAFE_SCHEMES):
                    bits.append(f'src="{self._esc_attr(value)}"')
            # href and everything else: dropped (the overlay needs no links).
        return (" " + " ".join(bits)) if bits else ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SANITIZE_DROP_TAGS:
            # Void elements (input, link, ...) have no end tag: dropping them
            # must not engage the depth counter.
            if tag not in _SANITIZE_VOID_TAGS:
                self._drop_depth += 1
            return
        if self._drop_depth:
            return
        if tag in _SANITIZE_ALLOWED_TAGS:
            self._emit(f"<{tag}{self._clean_attrs(tag, attrs)}>")
        # Unknown tags: unwrapped (content kept, tag dropped).

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SANITIZE_DROP_TAGS or self._drop_depth:
            return
        if tag in _SANITIZE_ALLOWED_TAGS:
            self._emit(f"<{tag}{self._clean_attrs(tag, attrs)} />")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SANITIZE_DROP_TAGS:
            self._drop_depth = max(0, self._drop_depth - 1)
            return
        if self._drop_depth:
            return
        if tag in _SANITIZE_ALLOWED_TAGS:
            self._emit(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        self._emit(data)

    def handle_entityref(self, name: str) -> None:
        self._emit(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._emit(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        pass  # comments carry nothing the deck needs

    def result(self) -> str:
        return "".join(self._out)


def sanitize_overlay_html(html: str) -> str:
    """Strip active content from Wrapped share overlay HTML.

    Never raises: on parse failure returns "" (the share is rejected
    upstream when the sanitized result loses the required marker).
    """
    if not isinstance(html, str) or not html:
        return ""
    try:
        parser = _OverlaySanitizer()
        parser.feed(html)
        parser.close()
        return parser.result()
    except Exception:
        logger.warning("[wrapped-share] overlay sanitize failed", exc_info=True)
        return ""


def init_wrapped_shares_table() -> None:
    """Create the wrapped_shares table once per process."""
    global _TABLES_READY
    if _TABLES_READY:
        return
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS wrapped_shares (
                token        TEXT PRIMARY KEY,
                kind         TEXT NOT NULL,           -- 'season' | 'weekly'
                ns           TEXT NOT NULL DEFAULT 'wrapped',
                overlay_html TEXT NOT NULL,
                share_data   JSONB,
                label        TEXT,                    -- e.g. "Blackedraw — Week 2 Wrapped" (OG title)
                created_at   TIMESTAMPTZ DEFAULT now(),
                expires_at   TIMESTAMPTZ DEFAULT now() + INTERVAL '1 year'
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_wrapped_shares_expires "
            "ON wrapped_shares (expires_at)"
        )
        conn.commit()
    _TABLES_READY = True


def create_wrapped_share(*, kind: str, ns: str, overlay_html: str,
                         share_data: dict | None, label: str) -> str:
    """Store a share and return its token. Raises on DB errors."""
    from dashboard_services.db import get_conn
    from psycopg.types.json import Json
    init_wrapped_shares_table()
    token = secrets.token_urlsafe(16)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO wrapped_shares (token, kind, ns, overlay_html, share_data, label)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (token, kind, ns, overlay_html, Json(share_data or {}), label),
        )
        conn.commit()
    return token


def get_wrapped_share(token: str) -> dict | None:
    """Fetch a share by token; None when unknown or expired."""
    from dashboard_services.db import get_conn
    init_wrapped_shares_table()
    token = (token or "").strip()
    if not token or len(token) > 128:
        return None
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT token, kind, ns, overlay_html, share_data, label, created_at
            FROM wrapped_shares
            WHERE token = %s AND expires_at > now()
            """,
            (token,),
        ).fetchone()
    if not row:
        return None
    return dict(row)
