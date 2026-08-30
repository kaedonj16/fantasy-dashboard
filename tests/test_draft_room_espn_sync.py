"""Draft Room UI/source guards for ESPN live companion sync."""
from pathlib import Path

from dashboard_services.pages.draft_room_page import build_draft_room_body

REPO = Path(__file__).resolve().parents[1]
ROOM_JS = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
PAGE = (REPO / "dashboard_services" / "pages" / "draft_room_page.py").read_text(encoding="utf-8")


def test_espn_sync_indicator_and_fallback_markup():
    body = build_draft_room_body("123", 2026, "espn", viewer_user_id="{AAA}", viewer_roster_id="3")
    assert 'id="drEspnSync"' in body
    assert 'id="drEspnFallback"' in body
    assert 'id="drEspnTools"' in body
    assert "ESPN sync needs a hand" in ROOM_JS or "who + ' sync needs a hand'" in ROOM_JS
    assert "Track manually" in ROOM_JS
    assert "showEspnTools({ unavailable: true })" in ROOM_JS
    assert "is-unavailable" in ROOM_JS
    assert "Get Chrome extension" in ROOM_JS
    assert "ESPN Draft · LIVE" in ROOM_JS or "who + ' Draft · LIVE" in ROOM_JS
    assert "ESPN Draft · Sync Unavailable" in ROOM_JS or "who + ' Draft · Sync Unavailable'" in ROOM_JS
    assert "ESPN Draft · Not Started" in ROOM_JS or "who + ' Draft · Not Started'" in ROOM_JS
    assert ".dr-pill-espn" in body
    assert '"viewerRosterId": "3"' in body
    assert '"chromeExtensionZipUrl"' in body


def test_espn_unavailable_uses_single_tools_card():
    """Broken sync folds into one tools card — no stacked warning + helpers."""
    body = build_draft_room_body("123", 2026, "espn", viewer_user_id="{AAA}", viewer_roster_id="3")
    assert "ESPN live sync unavailable" not in ROOM_JS
    assert "Switch to Manual Tracking" not in ROOM_JS
    assert ".dr-espn-tools.is-unavailable" in body
    assert ".dr-espn-tools-actions.is-split" in body
    assert "_onEspnHelperClick" in ROOM_JS
    assert "_espnToolsEl" in ROOM_JS
    assert "Track manually" in ROOM_JS
    # Generic start banners still stack their CTA under copy on phones.
    mobile = body.split("@media (max-width: 640px)")[1].split("@media")[0]
    assert ".dr-start-banner" in mobile
    assert "flex-wrap: wrap" in mobile
    assert "flex: 1 1 100%" in mobile


def test_sim_flag_still_declared():
    """Regression: ESPN relay vars must not drop the mock-draft `sim` flag."""
    assert "var sim = false;" in ROOM_JS
    assert "var simTimer = null;" in ROOM_JS


def test_extension_relay_wired_and_skips_manual_fallback():
    assert "function applyEspnExtensionRelay(detail)" in ROOM_JS
    assert "/api/draft/espn-relay" in ROOM_JS
    assert "brfantasy:espn-draft-relay" in ROOM_JS
    assert "if (_espnRelayActive) return false;" in ROOM_JS
    assert "_espnRelayActive = true;" in ROOM_JS
    assert "function _relayPlatform(src)" in ROOM_JS
    assert "function _leagueIdsMatch(relayId, cfgId)" in ROOM_JS
    assert "function _mapExtensionPicksRaw(raw)" in ROOM_JS
    assert "function _pullExtensionRelaySnapshot()" in ROOM_JS
    assert "_pullExtensionRelaySnapshot()" in ROOM_JS
    assert "function reconnectExtensionSync(" in ROOM_JS
    assert 'id="drEspnReconnect"' in PAGE
    assert "drEspnReconnectTools" in ROOM_JS
    assert "brfantasy:request-extension-reconnect" in ROOM_JS
    assert "function openEspnMobileSync()" not in ROOM_JS
    assert "drEspnMobileSync" not in ROOM_JS
    assert "/api/draft/espn-relay/token" not in ROOM_JS
    assert "Copy bookmarklet" not in ROOM_JS
    assert "Copy iOS Shortcut JS" not in ROOM_JS
    assert "Mobile Sync" not in ROOM_JS
    assert "dr-msync-title" in PAGE
    assert "github.com/kaedonj16/fantasy-dashboard/tree/main/extension" not in ROOM_JS
    assert "Extension setup" not in ROOM_JS
    assert "Get Chrome extension" in ROOM_JS
    assert "function openEspnExtensionInstall()" in ROOM_JS
    assert "chromeExtensionZipUrl" in PAGE
    assert "Load unpacked" in ROOM_JS
    assert "dr-espn-tools-body" in PAGE
    assert "dr-espn-tools-kicker" in PAGE
    assert "drEspnToolsDismiss" in ROOM_JS
    assert "Sync ESPN picks automatically" in ROOM_JS or "Sync ' + who + ' picks automatically" in ROOM_JS
    assert "Auto-sync needs a computer" in ROOM_JS
    assert "drEspnManualFromTools" in ROOM_JS
    assert "_espnToolsEl" in ROOM_JS
    assert "credentials: 'same-origin'" in ROOM_JS
    assert "function applyYahooExtensionRelay(detail)" in ROOM_JS
    assert "/api/draft/yahoo-relay" in ROOM_JS
    assert "brfantasy:yahoo-draft-relay" in ROOM_JS
    assert "function isExtLiveSource()" in ROOM_JS


def test_live_detect_requests_espn_sync_flag():
    assert "detectUrl += '&sync=1'" in ROOM_JS
    assert "Live sync currently supports Sleeper, ESPN, and Yahoo leagues." in ROOM_JS
    assert "plat === 'espn' || plat === 'yahoo'" in ROOM_JS or "detectUrl += '&sync=1'" in ROOM_JS


def test_sequential_missing_picks_and_idempotent_apply():
    assert "function applyOneLivePick(p)" in ROOM_JS
    assert "function applyMissingLivePicks(picks)" in ROOM_JS
    assert "if (state.picks[p.pick_no]) return false;" in ROOM_JS
    assert "if (pid && !p.unresolved) drafted[pid] = true;" in ROOM_JS
    assert "applyMissingLivePicks(d.picks)" in ROOM_JS


def test_predraft_placeholder_picks_are_not_applied():
    assert "function livePickIsSelection(p)" in ROOM_JS
    assert "if (!livePickIsSelection(p)) return;" in ROOM_JS
    assert "var remote = (picks || []).slice().filter(livePickIsSelection);" in ROOM_JS
    assert "if (state.mode === 'live' && String(state.status) === 'pre_draft' && !state.isComplete) return false;" in ROOM_JS
    assert "var done = _draftComplete();" in ROOM_JS


def test_live_picks_normalize_kicker_and_dst_positions():
    assert "function _normLivePos(pos)" in ROOM_JS
    assert "if (pos === 'PK') return 'K';" in ROOM_JS
    assert "if (pos === 'PK') pos = 'K';" in ROOM_JS
    assert "_normLivePos((meta && meta.position) || p.position)" in ROOM_JS


def test_offline_bar_clears_mobile_dock():
    css = (REPO / "static" / "dashboard.css").read_text(encoding="utf-8")
    app_js = (REPO / "static" / "app.js").read_text(encoding="utf-8")
    assert ".offline-bar {" in css
    assert "z-index: var(--z-toast);" in css.split(".offline-bar {")[1].split("}")[0]
    assert "bottom: calc(var(--dock-safe-bottom) + 20px);" in css
    # Hidden while online: opacity/visibility (not translate-only peek) + [hidden].
    assert "visibility: hidden;" in css.split(".offline-bar {")[1].split(".offline-bar.offline-bar-show")[0]
    assert ".offline-bar[hidden]" in css
    assert "navigator.onLine === false" in app_js
    assert "bar.setAttribute('hidden', '');" in app_js
    assert "bar.classList.add('offline-bar-show');" in app_js


def test_auth_errors_do_not_retry_and_fallback_stops_polling():
    assert "d.error === 'auth_denied'" in ROOM_JS
    assert "_espnAuthFailed = true" in ROOM_JS
    assert "function switchEspnToManual()" in ROOM_JS
    assert "state.mode = undefined;" in ROOM_JS
    assert "if (_pollInFlight) return;" in ROOM_JS


def test_refresh_forces_full_reconcile():
    assert "lastLivePicks = null;  // force a full ESPN/Sleeper reconcile on the first poll" in ROOM_JS
    assert "lastLivePicks = null;  // refresh reconcile after the tab was away" in ROOM_JS


def test_sleeper_live_path_still_uses_apply_live_picks():
    assert "applyLivePicks(d.picks);" in ROOM_JS
    assert "ms = 2000;  // active Sleeper draft" in ROOM_JS


def test_poll_interval_uses_espn_payload():
    assert "state.pollIntervalMs" in ROOM_JS
    assert "espnToStart > _START_WINDOW_MS) ms = 60000" in ROOM_JS
