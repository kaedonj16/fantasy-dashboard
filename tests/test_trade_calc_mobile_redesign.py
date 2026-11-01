"""Trade Calculator phone redesign (approved pick v3 mock) + chip headshots.

What shipped, and what these pins hold:

- SLIM HEAD (phones): title + a quiet context line (league name, scoring
  type, scoring format, QB setting) + the gear. The head controls
  (Viewing-as radios, Dynasty/Redraft, scoring format) relocate INTO the
  settings dropdown at phone widths; every control keeps its id and
  bindings because the same elements move.
- FLAT SIDES (phones): team cards lose their chrome; each side header is
  one quiet line (title, "gets · N pieces", running total). The header
  total mirrors #sideATotal/#sideBTotal, the summary's own elements, so
  it can never drift from the eval the summary and sticky bar use.
- FULL CHIPS WITH HEADSHOTS (all widths): renderChips prepends a face
  (ESPN photo over an initials disc, the rankings board's fallback
  shape) to player chips and a round tile to pick chips. The trade
  payload already carried espnHeadshot; normalizePlayerRow used to drop
  it.
- BUILDER'S END (phones): Analyze (production's #clearTradeBtn), the
  summary's balance section (stats + balance bar + verdict shell), and
  Share relocate to the .otc-mobile-flow slots under the sides; Clear
  joins Share as a pair, wired to the page's own window._clearTrade
  reset path (the one the sample-trade clear link already used). The
  summary card then carries Trade Fit without duplicating the balance.
- FAIR BAND: still centered by CSS (left:50% + translateX(-50%)); the
  render paths set its width only, never an inline left.
- STICKY BAR: the hide-while-visible observer targets the balance block
  on phones (the card no longer holds the balance there) and the
  summary card on desktop; the tap scroll follows the same split.

The relocation is one inline script in the page template (the pattern
the viewer-toggle relocation already used), keyed to the page's
existing 640px phone breakpoint.
"""
import os
import re

_HERE = os.path.dirname(__file__)
_PAGE = os.path.join(_HERE, "..", "dashboard_services", "pages", "trade_calculator_page.py")
_JS = os.path.join(_HERE, "..", "static", "app.js")
_CSS = os.path.join(_HERE, "..", "static", "dashboard.css")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _body(js, start_marker, end_marker):
    return js.split(start_marker, 1)[1].split(end_marker, 1)[0]


# ── Template structure ───────────────────────────────────────────────

def test_template_ships_the_phone_structure():
    page = _read(_PAGE)
    for needle in (
        'id="otcContextLine"',
        'id="otcFormatSlot"',
        'id="otcScoringSlot"',
        'id="otcMobileFlow"',
        'id="otcMobileAnalyze"',
        'id="otcMobileBalance"',
        'id="otcMobilePair"',
        'id="otcMobileClearBtn"',
        'id="sideAHeadTotal"',
        'id="sideBHeadTotal"',
        'id="sideAHeadPieces"',
        'id="sideBHeadPieces"',
        'class="otc-share-label"',
        'class="otc-page-subtitle"',
    ):
        assert needle in page, needle


def test_rendered_page_carries_the_slots_once_each():
    from dashboard_services.pages.trade_calculator_page import (
        build_trade_calculator_body,
    )

    html = build_trade_calculator_body(
        "1312067280816832512", 2026, num_teams=10, scoring_format="ppr",
        viewer_roster_id="3", has_premium=True, is_superflex=False,
        platform="sleeper", scoring_type="dynasty",
    )
    for el_id in ("otcContextLine", "otcMobileBalance", "otcMobileClearBtn",
                  "sideAHeadTotal", "sideBHeadTotal"):
        assert html.count(f'id="{el_id}"') == 1, el_id
    # Guest variant keeps production's disabled Analyze label.
    guest = build_trade_calculator_body(None, 2026)
    assert "Connect League to Analyze" in guest
    assert 'id="otcMobileClearBtn"' in guest


def test_relocation_script_moves_every_phone_piece():
    page = _read(_PAGE)
    assert "matchMedia('(max-width:640px)')" in page
    for pair in (
        "['.otc-page-head-controls .otc-viewer-toggles', 'otcSideSlot']",
        "['#scoringTypeControl', 'otcFormatSlot']",
        "['#scoringFormatControl', 'otcScoringSlot']",
        "['#clearTradeBtn', 'otcMobileAnalyze']",
        "['#shareTradeBtn', 'otcMobilePair']",
        "['.otc-summary-stats', 'otcMobileBalance']",
        "['.otc-balance-wrap', 'otcMobileBalance']",
        "['.otc-verdict-shell', 'otcMobileBalance']",
    ):
        assert pair in page, pair
    # Restores run in reverse so recorded next-siblings anchor correctly.
    assert "moves.length - 1" in page
    # The moved controls keep their ids: nothing is cloned or re-created.
    assert "cloneNode" not in page.split("Phone layout: reparent", 1)[1].split("</script>", 1)[0]


# ── Chips: headshots at both widths ──────────────────────────────────

def test_pool_normalizer_keeps_the_headshot():
    js = _read(_JS)
    norm = _body(js, "function normalizePlayerRow(p)", "let _tierThresholds")
    assert "espnHeadshot: p.espnHeadshot || \"\"" in norm
    # Persisted selections keep it too, so a restored chip paints its
    # photo on first render instead of waiting for the live refresh.
    save = _body(js, "function saveState()", "function loadState()")
    assert save.count("espnHeadshot: p.espnHeadshot") == 2


def test_render_chips_builds_faces_and_pick_tiles():
    js = _read(_JS)
    chips = _body(js, "function renderChips(side)", "function formatPickId")
    assert "chip.appendChild(buildChipFace(p));" in chips
    assert "chip.appendChild(buildPickTile(pk));" in chips
    face = _body(js, "function buildChipFace(p)", "function pickRoundLabel")
    assert 'face.className = "otc-chip-face"' in face
    assert "chipInitials(p.name)" in face
    assert "p.espnHeadshot || p.headshot" in face
    assert "_hiResHeadshot(url, 110)" in face
    # Failed photo hides itself after the raw-URL retry; initials show.
    assert 'this.style.visibility = "hidden"' in face
    tile = _body(js, "function buildPickTile(pk)", "function syncSideHeadMeta")
    assert 'tile.className = "otc-chip-pick-tile"' in tile
    assert "pickRoundLabel(pk)" in tile


def test_chip_face_css_is_desktop_and_phone():
    css = _read(_CSS)
    base = css.split(".otc-chip-face {", 1)[1].split("}", 1)[0]
    assert "width: 36px" in base and "height: 40px" in base
    assert "background: #dbe4ee" in base  # initials disc under the photo
    phone = css[css.index("chip headshots (all widths)"):].split("@media (max-width: 640px)", 1)[1]
    assert ".otc-chip-face" in phone and "width: 38px" in phone and "height: 42px" in phone
    tile = css.split(".otc-chip-pick-tile {", 1)[1].split("}", 1)[0]
    assert "background: #122d4b" in tile


# ── Header totals + context line wiring ──────────────────────────────

def test_head_totals_mirror_the_summary_totals():
    js = _read(_JS)
    mirror = _body(js, "function syncSideHeadTotals()", "function syncContextLine()")
    assert '"#sideATotal", "#sideAHeadTotal"' in mirror
    assert '"#sideBTotal", "#sideBHeadTotal"' in mirror
    bind = _body(js, "function bindMobileChrome()", "Button handlers - lean")
    assert "MutationObserver(syncSideHeadTotals)" in bind
    # Piece counts sync from the same render path that rebuilds chips.
    chips = _body(js, "function renderChips(side)", "function formatPickId")
    assert "syncSideHeadMeta(side);" in chips


def test_context_line_reads_chrome_name_and_control_state():
    js = _read(_JS)
    sync = _body(js, "function syncContextLine()", "function bindMobileChrome()")
    assert "#brLeagueChrome .br-ctx-name" in sync
    assert "window.__brctx" in sync
    assert 'getScoringType() === "redraft" ? "Redraft" : "Dynasty"' in sync
    assert 'getLeagueType() === "sf" ? "SF" : "1QB"' in sync
    bind = _body(js, "function bindMobileChrome()", "Button handlers - lean")
    for sel in ("#scoringTypeSelect", "#scoringFormatSelect", "#leagueTypeToggle"):
        assert sel in bind
    assert '"change", syncContextLine' in bind


def test_mobile_clear_uses_the_production_reset_path():
    js = _read(_JS)
    bind = _body(js, "function bindMobileChrome()", "Button handlers - lean")
    assert '"#otcMobileClearBtn"' in bind
    assert "window._clearTrade" in bind
    # The reset path itself is production's: clears all four lists.
    clear = _body(js, "window._clearTrade = function()", "Suggestions tab")
    for lst in ("sideAPlayers", "sideBPlayers", "sideAPicks", "sideBPicks"):
        assert f"state.{lst}.length = 0;" in clear


# ── Fair band: width only, never an inline left ──────────────────────

def test_fair_band_render_paths_set_width_only():
    js = _read(_JS)
    assert "fairZoneEl.style.width = _geo.fairWidthPct" in js
    assert "fairZoneEl.style.left" not in js
    sticky = _body(js, "function renderStickyBar()", "function scrollStickyBarToSummary")
    assert "fair.style.width = geo.fairWidthPct" in sticky
    assert "fair.style.left" not in sticky
    # The band elements keep their CSS centering (left:50% + translate).
    css = _read(_CSS)
    band = css.split(".otc-balance-fair {", 1)[1].split("}", 1)[0]
    assert "left: 50%" in band and "translateX(-50%)" in band
    sticky_band = css.split(".otc-sticky-band-fair {", 1)[1].split("}", 1)[0]
    assert "left: 50%" in sticky_band and "translateX(-50%)" in sticky_band


# ── Sticky bar: observer + tap target follow the layout ──────────────

def test_sticky_observer_targets_balance_block_on_phones():
    js = _read(_JS)
    sync = _body(js, "function syncStickyObserver()", "Phone chrome:")
    assert 'window.matchMedia("(max-width: 640px)").matches' in sync
    assert '"#otcMobileBalance"' in sync
    assert '".otc-summary-card"' in sync
    assert "io.observe(target);" in sync
    bind = _body(js, "function bindStickyBar()", "function syncStickyObserver()")
    assert "syncStickyObserver();" in bind
    scroll = _body(js, "function scrollStickyBarToSummary()", "function renderUnvaluedStrip")
    assert '"#otcMobileBalance"' in scroll
    assert "mobileBalance.childElementCount" in scroll


# ── Phone CSS: structure, existing breakpoints only ──────────────────

def _redesign_css():
    css = _read(_CSS)
    start = css.index("chip headshots (all widths)")
    end = css.index("MERGED: icons.css")
    assert start < end  # new CSS lives before the icons merge marker
    return css[start:end]


def test_phone_css_flattens_and_reflows():
    block = _redesign_css()
    phone = block.split("@media (max-width: 640px)", 1)[1]
    card = phone.split(".otc-team-card {", 1)[1].split("}", 1)[0]
    assert "border: none" in card and "background: transparent" in card
    flow = phone.split(".otc-mobile-flow {", 1)[1].split("}", 1)[0]
    assert "display: flex" in flow
    analyze = phone.split(".otc-mobile-analyze .otc-clear-btn {", 1)[1].split("}", 1)[0]
    assert "width: 100%" in analyze and "height: 44px" in analyze
    title_after = phone.split(".otc-page-title::after {", 1)[1].split("}", 1)[0]
    assert "width: 100%" in title_after  # the 300px h1 bar sizes to the title
    chip = phone.split(".otc-chip {", 1)[1].split("}", 1)[0]
    assert "border-bottom: 1px solid var(--border)" in chip
    assert "min-height: 52px" in chip
    # Phone-only pieces stay hidden off-phone.
    base = block.split("@media (max-width: 640px)", 1)[0]
    for cls in (".otc-context-line", ".otc-mobile-flow", ".otc-side-head-total"):
        assert cls in base
    assert "display: none" in base


def test_redesign_css_uses_no_new_breakpoints_or_pill_radius():
    block = _redesign_css()
    widths = set(re.findall(r"max-width:\s*(\d+)px", block))
    assert widths <= {"640", "375"}, widths
    assert "999px" not in block


def test_changelog_announces_the_redesign():
    from pin_helpers import changelog_entry

    entry = changelog_entry("slimmer header", "headshots", tag="new", link="/trade")
    assert entry["date"] == "2026-10-31"
    assert "—" not in entry["text"]  # no em dashes in UI copy
