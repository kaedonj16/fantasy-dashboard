"""Locked PRO columns on Advanced Metrics render blocked with their real label.

Regression tests for the free-viewer treatment of PRO-gated metric columns:
headers, compare chips, CSV headers and tooltips must show the friendly label
(e.g. "FPOE/G"), never the raw metric key, and the column must render as a
blocked paywalled column (blurred placeholder bars + lock) instead of blank
cells under a raw-key header.
"""
import json
import re
import subprocess

import pytest

from dashboard_services.pages import advanced_metrics_page as _amp
from dashboard_services.pages.advanced_metrics_page import _AM_JS

_PAGE_SRC = open(_amp.__file__).read()


def _extract_fn(name):
    """Pull a top-level JS function definition out of the page bundle."""
    m = re.search(
        r"function %s\(.*?\) \{(?:[^{}]|\{[^{}]*\})*\}" % re.escape(name),
        _AM_JS,
        re.S,
    )
    assert m, f"function {name} not found in _AM_JS"
    return m.group(0)


_MLABEL_JS = _extract_fn("_mLabel")
_MLOCKED_JS = _extract_fn("_mLocked")


def _run_label_logic(cfg, expr):
    """Evaluate a JS expression against the label helpers with a stub cfg."""
    script = (
        "var cfg = %s;\n%s\n%s\nconsole.log(JSON.stringify(%s));"
        % (json.dumps(cfg), _MLABEL_JS, _MLOCKED_JS, expr)
    )
    out = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout.strip())


FREE_CFG = {
    "hasPremium": False,
    "metrics": {"target_share": {"label": "Target Share"}},
    "proMetricInfo": {
        "ppr_over_expected_per_game": {"label": "FPOE/G"},
        "opportunity_trend": {"label": "Usage Trend"},
        "xfp_trend": {"label": "xFP Trend"},
    },
}
PRO_CFG = dict(FREE_CFG, hasPremium=True)


def test_label_helper_prefers_metric_label_then_pro_info():
    assert _run_label_logic(FREE_CFG, "_mLabel('target_share')") == "Target Share"
    assert (
        _run_label_logic(FREE_CFG, "_mLabel('ppr_over_expected_per_game')")
        == "FPOE/G"
    )
    assert _run_label_logic(FREE_CFG, "_mLabel('opportunity_trend')") == "Usage Trend"


def test_label_helper_never_returns_raw_key_for_known_metrics():
    for key in (
        "ppr_over_expected_per_game",
        "opportunity_trend",
        "xfp_trend",
        "target_share",
    ):
        assert _run_label_logic(FREE_CFG, "_mLabel('%s')" % key) != key


def test_locked_helper_flags_pro_metrics_for_free_viewers_only():
    assert (
        _run_label_logic(FREE_CFG, "_mLocked('ppr_over_expected_per_game')") is True
    )
    assert _run_label_logic(FREE_CFG, "_mLocked('target_share')") is False
    assert (
        _run_label_logic(PRO_CFG, "_mLocked('ppr_over_expected_per_game')") is False
    )


def test_locked_header_carries_lock_and_label():
    # The schema marks locked columns; the header render appends the gold lock
    # SVG after the plain label (textContent would escape it inside the label).
    assert "locked: _locked" in _AM_JS
    assert "th.insertAdjacentHTML('beforeend', ' ' + _LOCK_SVG)" in _AM_JS
    assert "_LOCK_SVG" in _AM_JS
    assert "am-th-locked" in _AM_JS


def test_no_lock_emoji_anywhere():
    assert "🔒" not in _PAGE_SRC


def test_locked_column_renders_blocked_not_blank():
    # Blurred placeholder bars + gold lock icon in the markup, blur in the
    # CSS; never bare dashes under a raw-key header.
    assert "am-locked-blur" in _AM_JS
    assert "am-lock-ico" in _AM_JS
    assert "filter:blur(4px)" in _PAGE_SRC
    assert "th.am-th-locked" in _PAGE_SRC


def test_locked_preset_pills_use_gold_lock_icon():
    # Decision pills for PRO presets render the same gold SVG lock, no emoji.
    assert "_LOCK_SVG_HTML" in _PAGE_SRC
    assert 'lock = " " + _LOCK_SVG_HTML if locked else ""' in _PAGE_SRC


def test_guests_get_no_custom_set_controls():
    # build_advanced_metrics_body takes is_guest and strips the dropdown +
    # Save/Delete controls for guests via the __CUSTOM_SETS__ placeholder.
    import inspect

    from dashboard_services.pages.advanced_metrics_page import (
        build_advanced_metrics_body,
    )

    assert "is_guest" in inspect.signature(build_advanced_metrics_body).parameters
    assert "__CUSTOM_SETS__" in _PAGE_SRC
    # The guest substitution leaves none of the set controls in the markup.
    start = _PAGE_SRC.find('    html = """') + len('    html = """')
    end = _PAGE_SRC.find('""".replace(', start)
    tpl = _PAGE_SRC[start:end]
    guest_html = tpl.replace("__CUSTOM_SETS__", "")
    for el in ("amSavedSet", "amSaveSetBtn", "amDeleteSetBtn", "amActiveSet"):
        assert el not in guest_html, el


def test_route_passes_guest_flag():
    route_src = open(
        _amp.__file__.replace(
            "dashboard_services/pages/advanced_metrics_page.py",
            "routes/league_pages_bp.py",
        )
    ).read()
    assert 'is_guest=not session.get("viewer_username")' in route_src


def test_locked_header_opens_paywall_not_sort():
    assert "advanced-metrics-metric-' + c.metricKey" in _AM_JS


def test_no_raw_key_label_fallbacks_remain():
    # Every display label must resolve through _mLabel (which falls back to
    # the PRO info map). The only raw-key-tolerant fallback allowed is the
    # _mLabel helper itself.
    src_without_helper = _AM_JS.replace(_MLABEL_JS, "")
    leftovers = re.findall(
        r"\(cfg\.metrics\[[^\]]+\]\s*&&\s*cfg\.metrics\[[^\]]+\]\.label\)\s*\|\|",
        src_without_helper,
    )
    assert leftovers == [], leftovers
