"""Strip developer HTML comments from served pages.

AdSense reviewers (and crawlers) see raw page text, so internal notes left in
HTML comments read as sloppy and can leak implementation detail. Source files
keep their comments (useful while developing); this removes them from the
responses users and reviewers actually receive.
"""
from __future__ import annotations

import re

# Blocks whose raw text must be left alone: a `<!--` inside JS/CSS/pre/textarea
# is content, not markup (e.g. JS strings, JSON-LD).
_LITERAL_BLOCK_RE = re.compile(
    r"(?is)(<(?:script|style|pre|textarea)\b.*?</(?:script|style|pre|textarea)>)"
)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def strip_html_comments(html: str) -> str:
    """Remove ``<!-- ... -->`` comments outside literal (script/style/pre/textarea) blocks."""
    parts = _LITERAL_BLOCK_RE.split(html)
    for i in range(0, len(parts), 2):
        parts[i] = _COMMENT_RE.sub("", parts[i])
    return "".join(parts)
