"""Regression tests for league-page blueprint dependencies."""

import pytest

pytest.importorskip("flask")

from dashboard_services.pages.waivers_page import (
    build_waivers_body as service_build_waivers_body,
)
from routes.league_pages_bp import build_waivers_body


def test_waivers_page_uses_extracted_service_builder():
    """The builder no longer exists in app.py after its service extraction."""
    assert build_waivers_body is service_build_waivers_body
