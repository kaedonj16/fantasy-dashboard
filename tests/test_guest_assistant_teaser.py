"""Contract guards for the BR Assistant guest teaser.

Logged-out visitors never saw the assistant: the signed-in chrome
(initAssistant/buildChrome) only mounts after /api/assistant/context
returns a league, and that endpoint 401s for guests. The guest teaser
is a static mount in the public bundle (initGuestAssistantTeaser in
static/app.js) that renders the shipped launcher + intro bubble with
no fetch and routes every tap to the sign-in modal. These tests pin
the mount's placement (public core), its gates, its exact copy, the
lite stylesheet rules guests actually load, and the SW bump.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
LANDING_LITE = (ROOT / "static" / "landing_lite.css").read_text(encoding="utf-8")
SW = (ROOT / "static" / "sw.js").read_text(encoding="utf-8")

MARKER = "// @public-js:core-end"
CORE = JS.split(MARKER, 1)[0]

_MOUNT_START = JS.index("function initGuestAssistantTeaser()")
_MOUNT_END = JS.index("})();", _MOUNT_START) + len("})();")
MOUNT = JS[_MOUNT_START:_MOUNT_END]


def test_guest_mount_lives_in_public_core():
    """Guests are served public.js (above the marker); the mount must
    be in that slice or it never reaches the guest homepage."""
    assert _MOUNT_START < JS.index(MARKER)
    assert "initGuestAssistantTeaser" in CORE


def test_guest_mount_makes_no_fetch():
    """The whole point: no /api/assistant/context call (it 401s for
    guests). The mount renders statically."""
    assert "fetch(" not in MOUNT
    assert "/api/assistant" not in MOUNT
    assert "apiJson" not in MOUNT


def test_guest_mount_gates():
    """Homepage only, logged-out only, never with a league context,
    and never alongside the real chrome."""
    assert "window._isSignedIn" in MOUNT
    assert "window.__brctx" in MOUNT
    assert "ctx.leagueId" in MOUNT
    assert "data-page" in MOUNT
    assert "'home'" in MOUNT
    # Double-render guard: buildChrome's launcher id is the sentinel.
    assert "document.getElementById('brAsstLauncher')" in MOUNT


def test_guest_mount_copy_is_exact():
    for needle in (
        "'Meet BR Assistant'",
        "Connect your league and ask anything: start/sit calls, "
        "trade values, waiver pickups, all from your league's data.",
        "'Who should I start at FLEX this week?'",
        "'Sign in free to try it'",
        "br-asst-bubble-signin",
        "brAsstGuestIntro",
    ):
        assert needle in MOUNT, f"guest teaser copy missing: {needle}"
    # Copy rules: no em dashes in the mount.
    assert "\u2014" not in MOUNT


def test_guest_mount_behavior_wiring():
    """Every tap routes to sign-in; retirement shares the signed-in
    intro bubble's localStorage key; entrance honors reduced motion."""
    assert "'brAsstIntroSeen'" in MOUNT
    assert "window.brOpenSignin" in MOUNT
    assert "br-asst-bubble-in" in MOUNT
    assert "prefers-reduced-motion: reduce" in MOUNT
    assert ", 900);" in MOUNT


def test_landing_lite_carries_teaser_rules():
    """Guests on `/` load landing_lite.css, NOT dashboard.css, so the
    shipped .br-asst-* rules the teaser uses must live there too."""
    for needle in (
        ".br-asst-launcher {",
        ".br-asst-launcher img {",
        ".br-asst-dot {",
        ".br-asst-bubble {",
        ".br-asst-bubble::after {",
        ".br-asst-bubble-main {",
        ".br-asst-bubble-lead {",
        ".br-asst-bubble-line {",
        ".br-asst-bubble-chiprow {",
        ".br-asst-bubble-dismiss {",
        ".br-asst-chip {",
        ".br-asst-bubble-signin {",
        "#brAsstGuestIntro {",
        "#brAsstGuestIntro.br-asst-bubble-in {",
        # Desktop parking spot: left of the Get Started card.
        "right: calc(50vw - 140px);",
        # Mobile dock-stack slot, same as the signed-in bubble.
        "bottom: calc(var(--dock-safe-bottom) + 14px + 56px + 14px);",
        # Tokens the launcher references that the lite sheet lacked.
        "--accent-strong: #0f2747;",
        "--gold: #ca8a04;",
    ):
        assert needle in LANDING_LITE, f"landing_lite.css missing: {needle}"


def test_landing_lite_carries_signin_modal_rules():
    """The modal build_nav ships on the guest home is styled by this
    sheet; without the overlay rule the markup would render inline."""
    for needle in (
        ".signin-modal-overlay {",
        ".signin-modal-box {",
        ".signin-modal-title {",
        ".signin-modal-sub {",
        ".signin-modal-actions {",
        ".signin-modal-cancel {",
        "--z-modal-top: 10001;",
    ):
        assert needle in LANDING_LITE, f"landing_lite.css missing: {needle}"


def test_guest_home_ships_signin_modal(offline_client):
    """The teaser's taps call window.brOpenSignin(), which opens
    #signinModal by id. The league-less nav return used to ship no
    modal at all, so on the real guest home that call was a no-op;
    the account variant now ships there (and only there)."""
    r = offline_client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "id='signinModal'" in html
    assert "Sign in to access your saved leagues and personalized features." in html


def test_guest_seo_page_stays_modal_free(offline_client):
    """Other league-less pages keep their direct Google links and do
    not gain the modal markup."""
    r = offline_client.get("/guides")
    assert r.status_code == 200
    assert "signinModal" not in r.get_data(as_text=True)


def test_sw_bumped_for_guest_teaser():
    assert "br-fantasy-v67" in SW
