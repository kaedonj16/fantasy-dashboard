"""Source contracts for the ScoreZone live-defect fixes.

Behavioral coverage lives in tests/scorezone_stale_harness.mjs (wired into
test_scorezone_refresh_polling.py); these pin the cross-file contracts the
harness cannot see:

- the client stale threshold matches the server-side store_stale threshold
  (utils/scorezone_store.STALE_THRESHOLD_S), so "Stale" means the same age
  on both sides of the wire;
- the badge derives from the payload collect clock (updated_at), and the
  304 path re-syncs the badge DOM (the latch bug: the flag was cleared in
  memory while the element stayed on screen);
- the sticky header offset token (--rz-nav-h) tracks the top-nav's measured
  height per breakpoint instead of the old flat 62px that matched no
  breakpoint (slit under the nav on phones, header behind the nav on
  desktop);
- the brand block can never shrink/ellipsize to make room for the stale
  badge (flex 0 0 auto + nowrap; the right group wraps instead).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "static" / "scorezone.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
STORE = (ROOT / "utils" / "scorezone_store.py").read_text(encoding="utf-8")


def _block(src: str, start_marker: str, end_marker: str) -> str:
    a = src.index(start_marker)
    return src[a:src.index(end_marker, a)]


# ── Stale threshold alignment ────────────────────────────────────────────────
def test_client_stale_threshold_matches_server_store_threshold():
    client = re.search(r"var _STALE_BADGE_MS = (\d+);", JS)
    server = re.search(r"STALE_THRESHOLD_S = ([\d.]+)", STORE)
    assert client and server
    assert int(client.group(1)) == int(float(server.group(1)) * 1000) == 180_000


def test_badge_ages_the_payload_collect_clock_not_poll_time():
    note = _block(JS, "function _noteCollectAt(", "function _dataIsStale(")
    assert "data.updated_at" in note
    sync = _block(JS, "function _syncStaleBadge(", "// Patch a mounted panel")
    assert "_dataIsStale()" in sync
    assert "_lastCollectAt" in sync  # "updated" prints the collect time


def test_304_path_resyncs_badge_and_does_not_advance_collect_clock():
    block = _block(JS, "if (resp.status === 304) {", "_rzLog('refresh-304'")
    assert "_syncStaleBadge();" in block
    assert "_noteCollectAt" not in block  # 304 = same content = same collect


def test_render_badge_uses_data_age_not_poll_failure():
    assert "_dataIsStale() ? '<span class=\"rz-stale-badge rz-data-stale\">" in JS
    # The partial updater delegates to the shared sync, which selects the
    # data-stale badge specifically (never the "Scores delayed" chip that
    # shares the .rz-stale-badge styling).
    assert "querySelector('.rz-data-stale')" in JS


# ── Sticky offset = real nav height ──────────────────────────────────────────
def test_rz_nav_h_tracks_measured_nav_heights_per_breakpoint():
    # Desktop >=1025: 99px measured (87 + the nav's max(12px, inset) top pad)
    assert "--rz-nav-h: calc(87px + max(12px, env(safe-area-inset-top)));" in CSS
    # 769-1024: 67px measured (55 + same 12px pad formula)
    assert "--rz-nav-h: calc(55px + max(12px, env(safe-area-inset-top)));" in CSS
    # <=768: 59px measured (49 + the mobile nav's max(10px, inset) top pad)
    assert "--rz-nav-h: calc(49px + max(10px, env(safe-area-inset-top)));" in CSS
    assert "--rz-nav-h: 62px" not in CSS  # the magic number is gone


# ── Header overflow: brand never compresses ──────────────────────────────────
def test_brand_never_shrinks_or_wraps_for_the_stale_badge():
    brand = _block(CSS, ".rz-brand {", ".rz-brand-dot")
    assert "flex: 0 0 auto;" in brand
    name = _block(CSS, ".rz-brand-name {", ".rz-brand-week")
    assert "white-space: nowrap;" in name
    week = _block(CSS, ".rz-brand-week {", ".rz-header-right")
    assert "white-space: nowrap;" in week  # "Wk 7" never wraps to two lines
    right = _block(CSS, ".rz-header-right {", ".rz-live-chip")
    assert "flex-wrap: wrap;" in right  # chips wrap below the brand instead
