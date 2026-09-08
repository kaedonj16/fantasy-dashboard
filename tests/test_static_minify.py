"""Boot minify helper: serve *.min.* when rjsmin/rcssmin work, else the source."""
from __future__ import annotations

from utils.static_minify import served_name, static_hash


def test_missing_file_falls_back_to_original_name():
    assert served_name("definitely-missing-xyz.js") == "definitely-missing-xyz.js"
    assert static_hash("definitely-missing-xyz.js") == "0"


def test_existing_css_serves_source_or_minified():
    name = served_name("font-awesome.css")
    assert name in ("font-awesome.css", "font-awesome.min.css")
    assert static_hash(name) != "0"
