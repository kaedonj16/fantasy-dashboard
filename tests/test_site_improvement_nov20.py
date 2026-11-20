"""Copy pins for the 2026-11-20 site-improvement fix: home hero honesty.

The home reimagine (PR #3188) dropped the "Sample matchup" label PR
#3114 put on the hardcoded hero board, and the gallery header claimed
"Real screens from a connected league" over the same static panels
(verified wrong against the Week 11 slate: BAL @ CAR not @ KC, CLE on
bye under a "vs CLE" row, ATL on bye, Lamar Jackson shown playing
while on IR). The guest header now frames the gallery as sample
screens with sample data, and the matchup panel carries the Sample
matchup strip again.

Skipped when Flask/pandas aren't installed; runs in CI with the full stack.
"""
import os

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _form_body() -> str:
    source = (_ROOT + "/app.py")
    with open(source, encoding="utf-8") as fh:
        text = fh.read()
    start = text.index('FORM_BODY = """')
    end = text.index('"""', start + len('FORM_BODY = """'))
    return " ".join(text[start:end].split())


def test_guest_gallery_header_no_longer_claims_real_screens():
    body = _form_body()
    assert "Real screens from a connected league" not in body
    assert "Sample screens with sample data. Tap one to look closer." in body


def test_matchup_panel_carries_sample_label():
    body = _form_body()
    panel = body.split('data-gal-panel="matchup"')[1].split("data-gal-panel=")[0]
    assert "hp-mb-strip" in panel
    assert "Sample matchup" in panel
