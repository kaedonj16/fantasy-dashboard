"""Player modal mock fidelity: Rookie + Breakout tabs match the approved
mock layouts (frames 5 and 6 of player-modal-mocks.html), plus the Value
History chart resize when the Overview tab becomes visible.

Content assertions over the shipped JS/CSS, following
tests/test_player_modal_followup_layout.py. No app import needed.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def _segment(start: str, end: str) -> str:
    i = JS.find(start)
    assert i != -1, f"segment start not found: {start}"
    j = JS.find(end, i)
    assert j != -1, f"segment end not found: {end}"
    return JS[i:j]


# ---------------------------------------------------------------------------
# Rookie tab (prospect panel)
# ---------------------------------------------------------------------------

def test_rookie_hero_is_four_equal_cards():
    # Four-card hero row: score, class rank, and the two rookie values.
    assert 'class="pm-rk-hero"' in JS
    assert "_pdHeroCard('Prospect score'" in JS
    assert "_pdHeroCard('Class rank'" in JS
    assert "_pdHeroCard('Rookie value · 1QB'" in JS
    assert "_pdHeroCard('Rookie value · SF'" in JS
    # Class rank card subs: class year + overall rank.
    assert "`${pdClassYr} class`" in JS
    assert "`#${pd.overall_rank} overall`" in JS
    # Value card subs name the draft year.
    assert "`At the ${pdClassYr} draft`" in JS
    # CSS: 4 equal columns on desktop, 2 on phone.
    assert ".pm-rk-hero { display: grid; grid-template-columns: repeat(4, 1fr);" in CSS
    assert ".pm-rk-hero { grid-template-columns: 1fr 1fr; }" in CSS


def test_rookie_no_tier_card_or_standalone_rows():
    # The old Tier card and the standalone Draft / Measurables rows are gone.
    assert "Tier ${pdTier}" not in JS
    assert "pdTierClass" not in JS
    assert "pdDraftRow" not in JS
    assert "pdMeasRow" not in JS
    assert "Historical Comparables" not in JS


def test_rookie_draft_and_measurables_folded_into_profile():
    assert 'pm-rk-card-h">Profile<' in JS
    assert "_pdKv('School', pd.school)" in JS
    assert "_pdKv('Size', pdSizeStr)" in JS
    assert "_pdKv('40 dash', pdFortyStr)" in JS
    assert "_pdKv('RAS', pdRasStr)" in JS
    assert "_pdKv('NFL team', data.team)" in JS
    assert "_pdKv('Draft capital', pdDraftCap)" in JS
    assert "_pdKv('Mock ADP', pdAdpStr)" in JS
    # Card hides when it has no rows.
    assert "const pdProfileCard = pdProfileRows" in JS


def test_rookie_components_card_and_right_stack():
    assert 'class="pm-rk-cols"' in JS
    assert 'pm-rk-card-h">Score components<' in JS
    # Radar and bars stay inside the components card; confidence is a footer.
    assert "${pdRadarHtml}" in JS and "${pdCompsHtml}" in JS
    assert 'class="pm-rk-conf">Data confidence: ${pdConf.toFixed(0)}</div>' in JS
    # Right stack exists and the returns mount sits inside it.
    assert 'class="pm-rk-stack"' in JS
    assert JS.find('class="pm-rk-stack"') < JS.find('id="pmRookieReturns"')
    # No-prospect-data path still renders the bare mount on its own.
    assert "const prospectPanelHTML = hasProspectData ? pdColHTML" in JS
    # CSS grid: two equal columns, stacking on phone.
    assert ".pm-rk-cols { display: grid; grid-template-columns: 1fr 1fr;" in CSS
    assert ".pm-rk-cols { grid-template-columns: 1fr; }" in CSS
    assert ".pm-rk-card {" in CSS and ".pm-rk-kv {" in CSS


def test_rookie_comparables_card_keeps_loader_contract():
    # Card chrome with the mock's title; both loader ids preserved.
    assert 'class="pm-rk-card" id="pmProspectComparables"' in JS
    assert 'pm-rk-card-h">Comparables<' in JS
    assert 'id="pmComparablesBody"' in JS
    # Loader still resolves empty comps to the "none" message in the card.
    assert "No close historical comps found." in JS
    assert "/api/prospects/comparables/" in JS


def test_rookie_scouting_notes_card_below_columns():
    assert 'class="pm-rk-card pm-rk-scout"' in JS
    assert 'pm-rk-card-h">Scouting notes<' in JS
    assert ".pm-rk-scout { margin-top: 12px; }" in CSS


def test_rookie_returns_loader_renders_card_with_bare_snap_values():
    seg = _segment("function _pmLoadRookieReturns", "function pmLoadTradesTab")
    # The loader renders the full card chrome inside the bare mount.
    assert 'class="pm-rk-card"' in seg
    assert "Early NFL returns · ${year}" in seg
    assert 'class="pm-rk-kv"' in seg
    # Snap share: bare 1-decimal values, no week labels or % signs.
    assert "snaps.map(x => x.v.toFixed(1)).join(' · ')" in seg
    assert "`W${x.week}" not in seg
    # Old divider wrapper is gone.
    assert "rk-section-divider" not in seg


# ---------------------------------------------------------------------------
# Breakout tab (in-season weekly builder only)
# ---------------------------------------------------------------------------

def _weekly_bk_segment() -> str:
    return _segment("function _buildWeeklyBkTabHTML", "function _buildBkTabHTML")


def test_breakout_weekly_score_card_layout():
    seg = _weekly_bk_segment()
    # Row 1 grid: score card + why card.
    assert 'class="pm-bk-top' in seg
    assert 'class="pm-bk-card"' in seg
    # Kind line carries the classification (no standalone pill row).
    assert "In season · ${_pmEsc(clsLabel)}" in seg
    assert 'class="pm-bk-kind"' in seg
    # Big untinted score, tier line, definition, folded-in confidence line.
    assert 'class="pm-bk-score-big">${score}</div>' in seg
    assert "Confidence ${conf} · evidence strength" in seg
    # Both score bases keep their titles and tier lines.
    assert "'Initial Role Score' : 'Role Change Score'" in seg
    assert "'No prior NFL baseline' : tier + ' role change'" in seg
    # No tinted hero-stat cards remain in the weekly builder.
    assert "pm-hero-stat" not in seg
    assert "clsColor" not in seg


def test_breakout_weekly_why_card():
    seg = _weekly_bk_segment()
    assert 'pm-bk-card-h">Why the role is growing<' in seg
    assert 'class="pm-bk-why-list"' in seg
    # Reasons still capped at 3 and escaped.
    assert ".slice(0, 3)" in seg
    assert "${_pmEsc(r)}" in seg or "_pmEsc(r)" in seg


def test_breakout_weekly_charts_are_cards():
    assert ".pm-bk-chart { background: var(--bg-alt); border: 1px solid var(--border);" in CSS
    assert ".pm-bk-top { display: grid; grid-template-columns: 1fr 300px;" in CSS
    assert ".pm-bk-top { grid-template-columns: 1fr; }" in CSS
    # Chart builder + geometry untouched.
    assert "function _pmBkWeeklyChart" in JS
    assert "class=\"pm-bk-chart\"" in JS


def test_breakout_offseason_builder_untouched_shape():
    # The offseason renderer keeps its own signature and stays out of scope.
    assert "function _buildBkTabHTML(data, scoreColor)" in JS


# ---------------------------------------------------------------------------
# Value History chart resize on Overview activation
# ---------------------------------------------------------------------------

def test_overview_switch_resizes_value_history_chart():
    seg = _segment("function pmSwitchTab", "function _pmTradePathCtx")
    # The hook runs only for Overview, after the panel is activated.
    assert "if (tab === 'overview')" in seg
    assert seg.find("panel.classList.add('pm-panel-active')") < seg.find("Plotly.Plots.resize")
    assert "requestAnimationFrame" in seg
    assert "document.getElementById('playerValueChart')" in seg
    assert "cd._fullLayout" in seg
    assert "Plotly.Plots.resize(cd)" in seg
