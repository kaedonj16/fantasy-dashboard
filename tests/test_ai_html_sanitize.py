"""Unit tests for AI HTML allowlist sanitizer."""
from dashboard_services.ai.html_sanitize import sanitize_ai_html


def test_strips_script_tags():
    dirty = '<div class="ai-copy"><p>ok</p><script>alert(1)</script></div>'
    clean = sanitize_ai_html(dirty)
    assert "<script" not in clean.lower()
    assert "alert(1)" not in clean
    assert "ok" in clean
    assert 'class="ai-copy"' in clean


def test_strips_img_onerror():
    dirty = '<p>hi</p><img src=x onerror=alert(1)>'
    clean = sanitize_ai_html(dirty)
    assert "<img" not in clean.lower()
    assert "onerror" not in clean.lower()
    assert "alert(1)" not in clean
    assert "hi" in clean


def test_strips_event_handlers_on_allowed_tags():
    dirty = '<div class="ai-copy" onclick="evil()">safe</div>'
    clean = sanitize_ai_html(dirty)
    assert "onclick" not in clean.lower()
    assert "evil" not in clean
    assert "safe" in clean
    assert 'class="ai-copy"' in clean


def test_blocks_javascript_urls():
    dirty = '<a href="javascript:alert(1)">click</a><a href="https://example.com/x">ok</a>'
    clean = sanitize_ai_html(dirty)
    assert "javascript:" not in clean.lower()
    assert 'href="https://example.com/x"' in clean
    assert "ok" in clean


def test_strips_iframe_object_embed():
    dirty = (
        '<p>x</p><iframe src="https://evil"></iframe>'
        '<object data="x"></object><embed src="y">'
    )
    clean = sanitize_ai_html(dirty)
    assert "<iframe" not in clean.lower()
    assert "<object" not in clean.lower()
    assert "<embed" not in clean.lower()
    assert "x" in clean


def test_preserves_common_ai_markup():
    raw = (
        '<div class="ai-copy trade-ai-wrap">'
        '<p><strong>Headline</strong></p>'
        '<ul class="trade-ai-list"><li>One</li></ul>'
        "<br>"
        "</div>"
    )
    clean = sanitize_ai_html(raw)
    assert "<strong>Headline</strong>" in clean
    assert "<li>One</li>" in clean
    assert 'class="trade-ai-list"' in clean


def test_empty_and_none():
    assert sanitize_ai_html("") == ""
    assert sanitize_ai_html(None) == ""
