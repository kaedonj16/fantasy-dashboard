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


def test_sanitize_keeps_inline_svg_crown():
    """The deck's crowns are inline SVG (the fa-crown font icon renders
    rust through the site's PNG icon layer). The sanitizer must let the
    glyph through with its inert attributes -- dropping it silently
    uncrowned every shared deck (the in-app deck kept it)."""
    html = (
        "<section class='wrapped-slide'>"
        "<svg class='wrapped-face-crown' viewBox='0 0 24 24' "
        "aria-hidden='true'><path d='M5 16 3 6l5.5 4L12 4l3.5 6L21 6l-2 "
        "10H5Z M5 18h14v2H5Z'/></svg>"
        "</section>"
    )
    out = sanitize_overlay_html(html)
    assert "<svg" in out and "wrapped-face-crown" in out
    assert "viewbox=" in out  # parser lowercases; browsers adjust to viewBox
    assert "<path" in out and "M5 16" in out


def test_sanitize_svg_stays_inert():
    """SVG is only allowed as the crown glyph: event handlers, scripts,
    links and style are stripped even inside it."""
    html = (
        "<section class='wrapped-slide'>"
        "<svg onload='alert(1)'><script>alert(2)</script>"
        "<a href='javascript:alert(3)'><path d='M0 0h1' fill='red'/></a>"
        "<path d='M1 1h1' style='fill:url(https://evil.example/x)'/>"
        "</svg></section>"
    )
    out = sanitize_overlay_html(html)
    assert "onload" not in out
    assert "alert" not in out
    assert "javascript:" not in out
    assert "evil.example" not in out
    assert "<a" not in out
    # The glyph paths themselves survive, with d/fill intact.
    assert out.count("<path") == 2
    assert "fill=" in out


def test_sanitize_keeps_root_relative_img_src():
    """The deck's own /static assets (footer wordmark, cover logo) ride
    root-relative srcs; stripping them broke the images on share pages.
    Protocol-relative (cross-origin) srcs stay dropped."""
    html = (
        "<section class='wrapped-slide'>"
        "<img src='/static/BR_Logo_dark.png?v=6c0c4828' alt=''>"
        "<img src='//evil.example/x.png' alt=''>"
        "</section>"
    )
    out = sanitize_overlay_html(html)
    assert "/static/BR_Logo_dark.png?v=6c0c4828" in out
    assert "evil.example" not in out


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


def test_share_page_keeps_crown_and_relative_logos():
    """The ground-truth failure (Kaedon's shared Week 9 deck, 2026-11):
    the stored share rendered with no crown on the top performer and a
    broken cover logo, because the sanitizer dropped <svg> and
    root-relative img srcs. The public page must keep both."""
    try:
        from dashboard_services.pages import history_page as H
    except ImportError:
        pytest.skip("history_page imports require full deps (openai)")
        return

    overlay = (
        "<div class='wrapped-overlay wrapped-overlay-wk'>"
        "<section class='wrapped-slide' data-kind='performers'>"
        "<div class='wrapped-face'>"
        "<svg class='wrapped-face-crown' viewBox='0 0 24 24' "
        "aria-hidden='true'><path d='M5 16 3 6Z'/></svg>"
        "</div></section>"
        "<section class='wrapped-slide' data-kind='cover'>"
        "<img src='/static/BR_Logo_dark.png?v=6c0c4828' alt='BR Fantasy' "
        "class='wrapped-cover-logo'>"
        "</section></div>"
    )
    page = H.render_wrapped_share_page(
        overlay_html=overlay,
        share_data={},
        label="Wrapped",
        ns="weekly-wrapped",
        css_url="/static/dashboard.css",
        logo_url="https://example.com/logo.png",
    )
    assert "wrapped-face-crown" in page
    assert "/static/BR_Logo_dark.png?v=6c0c4828" in page
