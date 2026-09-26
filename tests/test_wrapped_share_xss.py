"""XSS hardening for Wrapped share links: overlay_html sanitization.

The share overlay is untrusted client HTML rendered verbatim on a public
/wrapped/<token> page (whose CSP allows 'unsafe-inline'). These tests pin
the whitelist sanitizer: active content is stripped, presentation markup
survives, and the render path sanitizes even pre-existing stored shares.
"""
import pytest

pytest.importorskip("flask")

from dashboard_services.wrapped_shares import sanitize_overlay_html


def test_sanitize_strips_script_tags_and_content():
    html = (
        "<section class='wrapped-slide' data-kind='season'>"
        "<div>Hello</div><script>alert(document.cookie)</script>"
        "</section>"
    )
    out = sanitize_overlay_html(html)
    assert "<script" not in out
    assert "alert(document.cookie)" not in out
    assert "Hello" in out
    assert "wrapped-slide" in out


def test_sanitize_strips_event_handlers():
    html = (
        "<section class='wrapped-slide'>"
        "<div onclick=\"fetch('https://evil.example/x')\" onmouseover='x()'>Hi</div>"
        "<img src='https://example.com/a.png' onerror='alert(1)'>"
        "</section>"
    )
    out = sanitize_overlay_html(html)
    assert "onclick" not in out
    assert "onmouseover" not in out
    assert "onerror" not in out
    assert "evil.example" not in out
    # The benign image itself survives (src allowlisted).
    assert 'src="https://example.com/a.png"' in out


def test_sanitize_strips_javascript_urls():
    html = (
        "<section class='wrapped-slide'>"
        "<img src='javascript:alert(1)'>"
        "<img src='data:text/html,<script>alert(1)</script>'>"
        "</section>"
    )
    out = sanitize_overlay_html(html)
    assert "javascript:" not in out
    assert "data:text/html" not in out


def test_sanitize_drops_embeds_and_keeps_text():
    html = (
        "<section class='wrapped-slide'>"
        "<iframe src='https://evil.example'></iframe>"
        "<form action='https://evil.example'><input name='x'></form>"
        "<b>Bold</b> and <unknown-tag>plain</unknown-tag>"
        "</section>"
    )
    out = sanitize_overlay_html(html)
    assert "<iframe" not in out
    assert "<form" not in out
    assert "evil.example" not in out
    assert "<b>Bold</b>" in out
    assert "plain" in out  # unknown tags unwrapped, text kept


def test_sanitize_keeps_deck_presentation_markup():
    html = (
        "<section class='wrapped-slide' data-kind='season'>"
        "<div class='wrapped-num'>"
        "<span class='wrapped-big' data-w-count='1234' data-w-dp='1'>0</span>"
        "<span class='wrapped-unit'>pts</span></div>"
        "<div class='wrapped-winbar-row'><i style='width:72%'></i></div>"
        "</section>"
    )
    out = sanitize_overlay_html(html)
    assert "wrapped-slide" in out
    assert "data-w-count='1234'" in out or 'data-w-count="1234"' in out
    assert "width:72%" in out


def test_sanitize_never_raises_and_handles_empty():
    assert sanitize_overlay_html("") == ""
    assert sanitize_overlay_html(None) == ""
    assert sanitize_overlay_html("<div><span>unclosed") != ""


def test_render_path_sanitizes_legacy_stored_share():
    """render_wrapped_share_page must sanitize even shares stored before
    the create-time sanitizer existed."""
    try:
        from dashboard_services.pages import history_page as H
    except ImportError:
        pytest.skip("history_page imports require full deps (openai)")
        return

    evil = (
        "<section class='wrapped-slide'>"
        "<script>alert(1)</script><div>Deck</div>"
        "</section>"
    )
    page = H.render_wrapped_share_page(
        overlay_html=evil,
        share_data={},
        label="Wrapped",
        ns="wrapped",
        css_url="/static/dashboard.css",
        logo_url="https://example.com/logo.png",
    )
    # The page itself ships a legitimate bootstrap <script>; what must be gone
    # is the overlay's hostile markup and its payload.
    assert "<script>alert(1)</script>" not in page
    assert "alert(1)" not in page
    assert "wrapped-slide" in page
