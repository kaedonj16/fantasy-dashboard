"""Player modal Overview Recent Form: skeleton + per-bar opponent labels.

- While the game-logs fetch is in flight, the card shows a skeleton in the
  real graph's footprint (same divider/header/column structure, shared
  .skeleton shimmer); the real bars replace it, and the no-data / failure
  paths re-hide the card so the skeleton never sticks.
- Each bar gains an opponent label from the game-log payload ("@LV" away,
  "LV" home), rendered in the modal schedule convention: "@ LV" / "vs LV".
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def _loader_body() -> str:
    start = JS.index("function _pmLoadRecentForm")
    end = JS.index("function pmInjectContextActions")
    return JS[start:end]


def test_skeleton_builder_mirrors_graph_structure():
    assert "function _pmRecentFormSkeleton()" in JS
    body = JS[JS.index("function _pmRecentFormSkeleton()"):JS.index("function _pmLoadRecentForm")]
    # Same outer structure as the real render: divider, header, bars row.
    assert 'class="pm-section-divider"' in body
    assert 'class="pm-section-header"' in body
    assert 'class="pm-recent-bars"' in body
    # Columns keep the real column/track structure with shimmer bars.
    assert 'class="pm-recent-col"' in body
    assert 'class="pm-recent-track"' in body
    assert 'class="skeleton pm-recent-skel-bar"' in body
    assert 'class="skeleton pm-recent-skel-val"' in body
    assert 'class="skeleton pm-recent-skel-wk"' in body
    assert 'class="skeleton pm-recent-skel-opp"' in body


def test_skeleton_shown_up_front_and_rehidden_without_data():
    body = _loader_body()
    # Skeleton goes in before the fetch starts.
    assert "el.innerHTML = _pmRecentFormSkeleton();" in body
    assert body.index("_pmRecentFormSkeleton()") < body.index("_pmFetch(logsUrl)")
    # No logs / no real games / failure all re-hide the card.
    assert "if (!Array.isArray(logs) || !logs.length) { el.hidden = true; return; }" in body
    assert "if (!real.length) { el.hidden = true; return; }" in body
    assert "if (!_barsRendered) el.hidden = true;" in body


def test_opponent_label_rendered_per_bar():
    body = _loader_body()
    # Payload shape: "@LV" away, "LV" home, "BYE"/"" carry no label text.
    assert "const _oppRaw = String(g.opponent || '').trim();" in body
    assert "_oppRaw === 'BYE' ? ''" in body
    assert "'@ ' + _oppRaw.slice(1)" in body
    assert "'vs ' + _oppRaw" in body
    # Label sits under the week label in every column, escaped.
    assert '<span class="pm-recent-opp">${_pmEsc(_opp)}</span>' in body
    assert body.index('class="pm-recent-wk"') < body.index('class="pm-recent-opp"')


def test_recent_form_css_for_opponent_and_skeleton():
    assert ".pm-recent-opp { font-size: 10px;" in CSS
    assert "white-space: nowrap" in CSS
    # Skeleton blocks sized to the real line boxes (swap keeps height).
    assert ".pm-recent-skel-label { width: 80px; height: 15px; }" in CSS
    assert ".pm-recent-skel-val { width: 24px; height: 17px; }" in CSS
    assert ".pm-recent-skel-wk { width: 18px; height: 15px; }" in CSS
    assert ".pm-recent-skel-opp { width: 32px; height: 13px; }" in CSS
    assert ".pm-recent-skel-bar { width: 100%;" in CSS
