"""Contract guards for the BR Assistant popup UI (Stage 1 frontend).

The popup is built dynamically in static/app.js (initAssistant, next to
initPills) and styled by a delimited section in static/dashboard.css.
These tests lock the mock geometry, the API contract paths, and the
copy rules (no em dashes, no 999px radius) without needing a browser.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
COMPOSE = (ROOT / "dashboard_services" / "assistant" / "compose.py").read_text(encoding="utf-8")

_CSS_START = CSS.index("BR ASSISTANT (Stage 1 popup chat)")
_CSS_END = CSS.index("/* == BR ASSISTANT END == */")
ASSISTANT_CSS = CSS[_CSS_START:_CSS_END]

_JS_START = JS.index("function initAssistant()")
_JS_END = JS.index("function init()", _JS_START)
ASSISTANT_JS = JS[_JS_START:_JS_END]


def _rule(css: str, selector: str) -> str:
    match = re.search(re.escape(selector) + r"\s*\{([^}]+)\}", css)
    assert match, f"missing CSS rule for {selector}"
    return match.group(1)


def test_assistant_css_selectors_exist():
    for selector in (
        ".br-asst-launcher",
        ".br-asst-dot",
        ".br-asst-teaser",
        ".br-asst-popup",
        ".br-asst-scrim",
        ".br-asst-grab",
        ".br-asst-head",
        ".br-asst-body",
        ".br-asst-composer",
        ".br-asst-msg-user",
        ".br-asst-msg",
        ".br-asst-chip",
        ".br-asst-sys",
        ".br-asst-fb-btn",
        ".br-asst-memory",
        ".br-asst-pref-row",
        ".br-asst-starters",
        ".br-asst-starters-label",
        ".br-asst-meter",
        ".br-asst-meter-fill",
        ".br-asst-plans",
        ".br-asst-plan-row",
        ".br-asst-plan-current",
        ".br-asst-plan-badge",
        ".br-asst-mention",
        ".br-asst-list",
        ".br-asst-list-item",
        ".br-asst-receipt-toggle",
        ".br-asst-receipt-rows",
        ".br-asst-teaser-main",
        ".br-asst-teaser-line",
    ):
        assert selector in ASSISTANT_CSS, f"missing assistant CSS for {selector}"


def test_desktop_geometry_matches_mock():
    launcher = _rule(ASSISTANT_CSS, ".br-asst-launcher")
    assert "position: fixed" in launcher
    assert "width: 56px" in launcher
    assert "height: 56px" in launcher
    assert "bottom: 66px" in launcher  # stacked above #floating-pill-group
    assert "right: 16px" in launcher

    popup = _rule(ASSISTANT_CSS, ".br-asst-popup")
    assert "width: 392px" in popup
    assert "height: 600px" in popup
    assert "bottom: 66px" in popup
    assert "right: 16px" in popup

    teaser = _rule(ASSISTANT_CSS, ".br-asst-teaser")
    assert "width: 326px" in teaser
    assert "bottom: 136px" in teaser


def test_mobile_sheet_uses_dock_safe_token():
    # Everything mobile anchors to --dock-safe-bottom instead of raw pixels.
    assert ASSISTANT_CSS.count("var(--dock-safe-bottom)") >= 4
    media = ASSISTANT_CSS[ASSISTANT_CSS.index("@media (max-width: 768px)"):]
    # Launcher: dock-safe + 14px gap, which resolves to 86px at the
    # default 72px dock-safe value.
    launcher = _rule(media, ".br-asst-launcher")
    assert "bottom: calc(var(--dock-safe-bottom) + 14px)" in launcher
    # Sheet: dock-safe + 8px, which resolves to 80px at the default value.
    popup = _rule(media, ".br-asst-popup")
    assert "bottom: calc(var(--dock-safe-bottom) + 8px)" in popup
    # Tall sheet: viewport height minus the dock line minus 48px (the
    # 8px dock gap plus a 40px scrim sliver at the top), capped so
    # very tall screens still read as a sheet.
    assert "height: calc(100dvh - var(--dock-safe-bottom) - 48px)" in popup
    assert "max-height: 880px" in popup
    assert "452px" not in ASSISTANT_CSS
    assert "left: 8px" in popup
    assert "right: 8px" in popup
    # Scrim stops at the dock-safe line so the dock is never dimmed.
    scrim = _rule(media, ".br-asst-scrim")
    assert "bottom: var(--dock-safe-bottom)" in scrim


def test_keyboard_aware_mobile_sheet():
    # iOS does not shrink the layout viewport for fixed elements when
    # the keyboard opens, so the dock-parked sheet kept its geometry
    # and the composer sat behind the keyboard. The JS must track the
    # visual viewport: publish the keyboard inset as --br-asst-kb on
    # the root, flag the popup while the keyboard is up, listen only
    # while the sheet is open, and tear down on collapse (the path
    # both minimize and close take).
    for needle in (
        "window.visualViewport",
        "function syncKbInset()",
        "function kbListen(on)",
        "setProperty('--br-asst-kb'",
        "br-asst-kb-open",
        "vv.addEventListener('resize', syncKbInset)",
        "vv.addEventListener('scroll', syncKbInset)",
        "vv.removeEventListener('resize', syncKbInset)",
        "vv.removeEventListener('scroll', syncKbInset)",
        "kbListen(true)",
        "kbListen(false)",
    ):
        assert needle in ASSISTANT_JS, f"missing keyboard-sheet piece {needle}"
    # Listeners attach on open and detach on collapse, not at build.
    assert ASSISTANT_JS.index("kbListen(true)") > ASSISTANT_JS.index("function openPopup")
    assert ASSISTANT_JS.index("kbListen(false)") > ASSISTANT_JS.index("function collapsePopup")
    # The inset is consumed only inside the mobile media query: the
    # desktop popup rules never reference it, so desktop geometry is
    # pixel-unchanged.
    media_at = ASSISTANT_CSS.index("@media (max-width: 768px)")
    desktop_css = ASSISTANT_CSS[:media_at]
    assert "--br-asst-kb" not in desktop_css
    assert "br-asst-kb-open" not in desktop_css
    media = ASSISTANT_CSS[media_at:]
    kb_rule = _rule(media, ".br-asst-popup.br-asst-kb-open")
    # Bottom edge sits directly on the keyboard top; the sheet
    # stretches from near the screen top (top + bottom, auto height)
    # instead of keeping the fixed 452px dock-parked height.
    assert "bottom: var(--br-asst-kb, 0px)" in kb_rule
    assert "top: 8px" in kb_rule
    assert "height: auto" in kb_rule
    assert "max-height: none" in kb_rule
    # Opening the sheet must not summon the keyboard on sheet widths:
    # the composer focus on open is gated to non-sheet layouts. (The
    # starter fill chips still focus on an explicit tap; that is a
    # user gesture, not the open path.)
    assert "if (state.view === 'thread' && !sheetLayout()) inputEl.focus();" in ASSISTANT_JS
    assert "function sheetLayout()" in ASSISTANT_JS
    assert "'(max-width: 768px)'" in ASSISTANT_JS


def test_assistant_js_endpoints_and_wiring():
    for path in (
        "/api/assistant/context",
        "/api/assistant/chat",
        "/api/assistant/feedback",
        "/api/assistant/correction",
        "/api/assistant/preferences",
    ):
        assert path in ASSISTANT_JS, f"missing endpoint {path}"
    # initAssistant is defined before the public bundle marker, on the
    # same side as initPills, and wired from the same init().
    assert JS.index("function initAssistant()") < JS.index("// @public-js:core-end")
    init_body = JS[JS.index("function init()"):]
    assert "initAssistant();" in init_body[:400]
    assert "initPills();" in init_body[:400]


def test_assistant_js_renders_contract_pieces():
    for needle in (
        "trade-ai-verdict-accept",
        "trade-ai-verdict-decline",
        "bo-receipt",
        "bo-rc-row",
        "What BR remembers about you",
        "What was wrong?",
        "Wrong player data",
        "Bad reasoning",
        "Ignored my situation",
        "Tap again to confirm",
        "Forget everything BR remembers",
        "sessionStorage",
        "player_id",
        "BR_Mark_dark.png",
        # Starter chips (empty-thread state): the three core asks.
        "Try asking",
        "What should I do with my team this week?",
        "Who should I start at FLEX this week?",
        "Which player is a buy-low right now?",
        "hideStarters",
        # A wrong-data correction can return a revised answer to append.
        "data.revision",
        "data.revision.answer",
    ):
        assert needle in ASSISTANT_JS, f"missing assistant JS piece {needle}"
    # Cap (429) and unavailable (503) states are handled explicitly.
    assert "res.status === 429" in ASSISTANT_JS
    assert "res.status === 503" in ASSISTANT_JS


def test_answer_contract_fields_consumed():
    # The backend composer contract (tests/test_assistant_contract.py
    # pins the producing side): act_link is an OBJECT with href/label.
    # This-week actions still CARRY href/cta_label/key_number in the
    # payload, but the thread renders them as bullets, so the UI
    # consumes the who (headline) and the why (detail, key_number as
    # the fallback why) instead of building cards with CTA buttons.
    for needle in (
        "answer.act_link.href",
        "answer.act_link.label",
        "action.headline",
        "action.detail",
        "action.key_number",
        # Mentions: pack player ids ride the answer so a name in the text
        # opens the player modal instead of rendering as plain text.
        "answer.mentions",
        "br-asst-mention",
        "openPlayerModal",
    ):
        assert needle in ASSISTANT_JS, f"UI does not consume contract field {needle}"


def test_minimize_and_close_are_distinct():
    # Minimize collapses and preserves; Close collapses and ends the
    # conversation. They must be separate handlers wired to separate
    # header buttons, with the launcher toggle and the scrim on
    # minimize semantics (neither ends the conversation).
    assert "function minimizePopup()" in ASSISTANT_JS
    assert "function closeConversation()" in ASSISTANT_JS
    assert "querySelector('#brAsstMin').addEventListener('click', minimizePopup)" in ASSISTANT_JS
    assert "querySelector('#brAsstClose').addEventListener('click', closeConversation)" in ASSISTANT_JS
    assert "scrim.addEventListener('click', minimizePopup)" in ASSISTANT_JS
    # Minimize saves the thread scroll; reopening restores it.
    assert "state.savedScroll = threadEl.scrollTop" in ASSISTANT_JS
    assert "threadEl.scrollTop = state.savedScroll" in ASSISTANT_JS
    # Close drops the server thread and rebuilds the starter state.
    assert "state.threadId = null" in ASSISTANT_JS
    assert "threadEl.appendChild(startersEl)" in ASSISTANT_JS
    assert "startersEl.hidden = false" in ASSISTANT_JS
    # An answer in flight for the closed conversation never renders.
    assert "gen !== state.gen" in ASSISTANT_JS


def test_panel_motion_and_typing_indicator():
    # Enter/exit motion rides the Next Steps glide curve at .38s, with
    # a sheet variant on mobile and a reduced-motion guard.
    assert "@keyframes br-asst-in" in ASSISTANT_CSS
    assert "@keyframes br-asst-out" in ASSISTANT_CSS
    enter = _rule(ASSISTANT_CSS, ".br-asst-popup.br-asst-enter")
    assert ".38s" in enter and "cubic-bezier(0.22, 1, 0.36, 1)" in enter
    exit_rule = _rule(ASSISTANT_CSS, ".br-asst-popup.br-asst-exit")
    assert ".38s" in exit_rule and "cubic-bezier(0.22, 1, 0.36, 1)" in exit_rule
    media = ASSISTANT_CSS[ASSISTANT_CSS.index("@media (max-width: 768px)"):]
    assert "animation-name: br-asst-sheet-in" in media
    assert "animation-name: br-asst-sheet-out" in media
    assert "@media (prefers-reduced-motion: reduce)" in ASSISTANT_CSS
    # Typing indicator: bouncing ellipsis bubble rendered into the
    # thread by setBusy, removed when the answer lands.
    assert ".br-asst-typing" in ASSISTANT_CSS
    assert "@keyframes br-asst-typing-dot" in ASSISTANT_CSS
    assert "br-asst-dot-anim" in ASSISTANT_JS
    assert "threadEl.appendChild(typingEl)" in ASSISTANT_JS
    assert "typingEl.remove()" in ASSISTANT_JS
    # The old foot-of-panel "Thinking..." text is gone.
    assert "brAsstThinking" not in ASSISTANT_JS
    assert "br-asst-thinking" not in ASSISTANT_CSS


def test_usage_meter_and_plans_panel():
    # Meter: rendered from the caps payload for every user type, updated
    # after each send, and tappable into the plans panel.
    for needle in (
        "brAsstMeter",
        "brAsstMeterLabel",
        "brAsstMeterFill",
        "renderMeter",
        "chats used this month",
        # Owner/exempt state: a label, no depleting bar math.
        "caps.exempt",
        "Assistant chats: unlimited",
        # Plans panel: the tier ladder from the caps payload, the caller's
        # tier marked, upgrade into the existing pricing flow.
        "brAsstPlans",
        "showPlans",
        "renderPlans",
        "caps.tiers",
        "Assistant chats by plan",
        "Current plan",
        "Upgrade",
        "assistant chats a month",
        "/pricing",
    ):
        assert needle in ASSISTANT_JS, f"missing meter/plans piece {needle}"
    # The meter refreshes from the chat response caps (no extra polling).
    assert "if (data.caps) state.caps = data.caps;" in ASSISTANT_JS
    # At cap the chat lands on the plans panel instead of a dead-end note.
    cap_branch = ASSISTANT_JS[ASSISTANT_JS.index("res.status === 429"):]
    assert "showPlans();" in cap_branch[:200]
    # Near-cap and at-cap tones exist on the meter.
    assert ".br-asst-meter-warn" in ASSISTANT_CSS
    assert ".br-asst-meter-full" in ASSISTANT_CSS


def test_no_banned_values_in_new_sections():
    assert "999px" not in ASSISTANT_CSS
    assert "999px" not in ASSISTANT_JS
    assert "\u2014" not in ASSISTANT_CSS  # em dash
    assert "\u2014" not in ASSISTANT_JS  # em dash


def test_answers_render_light_no_cards():
    # Kaedon's presentation rule: no big card components inside an
    # answer. No Next Steps items, no per-action CTA buttons, no big
    # number blocks anywhere in the assistant renderer.
    for banned in ("ns-item", "ns-cta", "ns-priority", "br-asst-actions", "br-asst-num"):
        assert banned not in ASSISTANT_JS, f"card remnant {banned} in assistant JS"
        assert banned not in ASSISTANT_CSS, f"card remnant {banned} in assistant CSS"
    # Actions are a bulleted list: who (headline) plus why (detail).
    assert "el('ul', 'br-asst-list')" in ASSISTANT_JS
    assert "br-asst-list-item" in ASSISTANT_JS
    list_rule = _rule(ASSISTANT_CSS, ".br-asst-list")
    assert "list-style: disc" in list_rule
    # The light pieces stay: verdict pill, chips, receipts, mentions.
    assert "trade-ai-verdict" in ASSISTANT_JS
    assert "br-asst-chip" in ASSISTANT_JS


def test_new_answers_anchor_to_their_top():
    # A fresh answer must land scrolled to its TOP (verdict first),
    # not at its receipts. renderAnswer ends by anchoring the new
    # message; the minimize restore path keeps using savedScroll.
    assert "function scrollToAnswer(node)" in ASSISTANT_JS
    render = ASSISTANT_JS[ASSISTANT_JS.index("function renderAnswer"):]
    render = render[:render.index("function buildFeedback")]
    assert "threadEl.appendChild(wrap);" in render
    assert "scrollToAnswer(wrap);" in render
    assert "scrollThread();" not in render
    # User and system lines keep the bottom anchor.
    assert "function appendUser" in ASSISTANT_JS
    assert "state.savedScroll = threadEl.scrollTop" in ASSISTANT_JS


def test_receipts_start_collapsed_behind_toggle():
    # Every answer's receipts render collapsed: the toggle row says
    # Receipts, starts aria-expanded=false, and the rows are in the
    # DOM inside the animated wrapper, expandable per answer.
    assert "br-asst-receipt-toggle" in ASSISTANT_JS
    assert "'Receipts'" in ASSISTANT_JS
    assert "recToggle.setAttribute('aria-expanded', 'false')" in ASSISTANT_JS
    assert "br-asst-collapsed" in ASSISTANT_JS
    assert "rec.classList.toggle('br-asst-collapsed')" in ASSISTANT_JS
    assert "br-asst-receipt-rows" in ASSISTANT_JS
    assert "br-asst-receipt-inner" in ASSISTANT_JS
    collapsed = _rule(ASSISTANT_CSS, ".br-asst-receipts.br-asst-collapsed .br-asst-receipt-rows")
    assert "grid-template-rows: 0fr" in collapsed
    rows = _rule(ASSISTANT_CSS, ".br-asst-receipt-rows")
    assert "cubic-bezier(0.22, 1, 0.36, 1)" in rows
    toggle = _rule(ASSISTANT_CSS, ".br-asst-receipt-toggle")
    assert "background: transparent" in toggle  # a text row, not a card


def test_act_link_is_a_destination_labeled_pill():
    # One small link-button under the answer, labeled with where it
    # goes. Labels are fixed at the source in compose.py.
    assert '"label": "View dashboard"' in COMPOSE
    assert "Open dashboard" not in COMPOSE
    assert '"label": "View on Advanced Metrics"' in COMPOSE
    assert '"label": "View on Waivers"' in COMPOSE
    assert '"label": "View in Start/Sit"' in COMPOSE
    # Client treatment: bordered pill with an arrow, not a filled CTA.
    link = _rule(ASSISTANT_CSS, ".br-asst-actlink")
    assert "border: 1px solid var(--border)" in link
    assert "border-radius: var(--radius-pill)" in link
    assert "display: inline-flex" in link
    assert "br-asst-actlink-arrow" in ASSISTANT_JS
    assert "SVG_ARROW" in ASSISTANT_JS


def test_teaser_is_a_slim_single_line():
    # The closed-state teaser is one line (mark + top action + chevron
    # + dismiss), not a card with a CTA button.
    assert "See the plan" not in ASSISTANT_JS
    assert "br-asst-teaser-head" not in ASSISTANT_JS
    assert "br-asst-teaser-title" not in ASSISTANT_JS
    assert "br-asst-teaser-main" in ASSISTANT_JS
    assert "br-asst-teaser-line" in ASSISTANT_JS
    assert "SVG_CHEVRON_RIGHT" in ASSISTANT_JS
    line = _rule(ASSISTANT_CSS, ".br-asst-teaser-line")
    assert "text-overflow: ellipsis" in line
    assert "white-space: nowrap" in line


def test_grab_bar_drags_sheet_to_minimize():
    # The grab handle (and header strip) actually drags: pointer
    # capture, the sheet follows via translateY, past 120px or a fast
    # flick it minimizes with minimize semantics, otherwise it springs
    # back on the glide curve. Sheet widths only; buttons in the
    # header are never hijacked.
    for needle in (
        "addEventListener('pointerdown', dragStart)",
        "addEventListener('pointermove', dragMove)",
        "addEventListener('pointerup', dragEnd)",
        "setPointerCapture",
        "'translateY(' + dragState.dy + 'px)'",
        "dy > 120",
        "vel > 0.5",
        "e.target.closest('button')",
    ):
        assert needle in ASSISTANT_JS, f"missing grab-drag piece {needle}"
    drag = ASSISTANT_JS[ASSISTANT_JS.index("function dragStart"):]
    drag = drag[:drag.index("node.addEventListener('pointerdown'")]
    assert "minimizePopup();" in drag
    assert "cubic-bezier(0.22, 1, 0.36, 1)" in drag  # spring-back curve
    assert "sheetLayout()" in drag  # inert on desktop widths
    media = ASSISTANT_CSS[ASSISTANT_CSS.index("@media (max-width: 768px)"):]
    grab = _rule(media, ".br-asst-grab")
    assert "touch-action: none" in grab


def test_server_text_is_not_injected_as_html():
    # innerHTML is allowed only for static skeleton markup and icons.
    # Answer rendering must go through textContent (the el() helper).
    assert "answer.text" in ASSISTANT_JS
    assert ".innerHTML = SVG_" in ASSISTANT_JS or "innerHTML = SVG_" in ASSISTANT_JS
    # No dynamic server field is concatenated into an innerHTML string.
    for bad in ("innerHTML = answer", "innerHTML += answer", "innerHTML = data."):
        assert bad not in ASSISTANT_JS
