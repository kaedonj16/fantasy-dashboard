"""AdSense review readiness: consent CMP is deployed and dev HTML comments are
stripped from served pages."""
from __future__ import annotations

import pytest

pytest.importorskip("flask")

from utils.html_sanitize import strip_html_comments


def test_strips_markup_comments():
    html = "<div><!-- internal note: fix the CLS --><p>hi</p></div>"
    assert strip_html_comments(html) == "<div><p>hi</p></div>"


def test_strips_multiline_comments():
    html = "<div>\n<!-- line one\nline two -->\n<p>hi</p></div>"
    assert "<!--" not in strip_html_comments(html)
    assert "<p>hi</p>" in strip_html_comments(html)


def test_leaves_script_content_alone():
    html = '<script>var s = "<!-- not a comment -->";</script><p><!-- real --></p>'
    out = strip_html_comments(html)
    assert '<!-- not a comment -->' in out
    assert "<!-- real -->" not in out


def test_leaves_style_pre_textarea_alone():
    html = "<style>/* <!-- */</style><pre><!-- x --></pre><textarea><!-- y --></textarea>"
    assert strip_html_comments(html) == html


def test_no_comments_no_change():
    html = "<p>plain</p>"
    assert strip_html_comments(html) == html


def test_homepage_serves_funding_choices_and_no_html_comments(offline_client):
    html = offline_client.get("/").get_data(as_text=True)
    assert "fundingchoicesmessages.google.com/i/pub-9164153092633845" in html
    assert "googlefcPresent" in html
    assert "<!--" not in html


def test_consent_placeholder_comment_is_gone(offline_client):
    html = offline_client.get("/").get_data(as_text=True)
    assert "Cookie consent handled by" not in html


def test_public_pages_have_no_html_comments(offline_client):
    for path in ("/privacy", "/terms", "/about", "/guides"):
        html = offline_client.get(path).get_data(as_text=True)
        assert "<!--" not in html, path
