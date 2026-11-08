"""Contracts for the 2026-11-08 banner / reset / notification fixes.

1. Banner queue: every fixed bottom banner (Discord, League PRO invite,
   recap, trial, Google link, payment failed) goes through the shared
   brPromoGate, one at a time, in priority order. Before this, only the
   Discord bar and the assistant bubble used the gate; the rest all
   rendered at the same fixed spot at once and stacked (screenshot:
   League PRO invite card with the Discord bar on top of it).
2. Discord phone bar: the close button takes the last slot and the
   copy line is dropped (it truncated to "Talk dyn...").
3. "Not me?" (/reset-user) returns to the same page at the same scroll
   spot instead of the top of the home page.
4. Notification picker / settings modals: the height cap is 100% of
   the fixed overlay, not calc(100vh - 96px). 100vh is the LARGE iOS
   viewport (the same root cause as the 2026-10-27 changelog panel
   fix), so the old cap let the modal run off the bottom of the
   screen and its last rows / Done button were unreachable.
5. Queue count: when a run holds more than one banner, the showing
   banner carries a small "1/x" label beside its close button, its
   position in the run by priority (Kaedon's pick from the mock).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
PAYWALL_JS = (ROOT / "static" / "paywall.js").read_text(encoding="utf-8")
AUTH = (ROOT / "routes" / "auth_bp.py").read_text(encoding="utf-8")

MARKER = "// @public-js:core-end"
CORE = JS.split(MARKER, 1)[0]

_GATE_START = JS.index("window.brPromoGate = (function () {")
_GATE_END = JS.index("})();", _GATE_START) + len("})();")
GATE = JS[_GATE_START:_GATE_END]


def _fn(name: str, next_name: str) -> str:
    return APP_PY[APP_PY.index(f"def {name}"):APP_PY.index(f"def {next_name}")]


RECAP = _fn("_recap_ready_banner", "_draft_imminent_banner")
GOOGLE = _fn("_google_link_pro_banner", "_pro_trial_banner")
TRIAL = _fn("_pro_trial_banner", "_connection_status_banner")
DUNNING = _fn("_dunning_banner", "_discord_banner")
DISCORD = APP_PY[APP_PY.index("def _discord_banner()"):APP_PY.index("DEFAULT_META_DESCRIPTION")]


# ---------------------------------------------------------------- gate

def test_gate_accepts_priorities_and_keeps_priority_queue():
    assert "function register(name, show, hide, priority)" in GATE
    assert "enqueue" in GATE
    assert "prio(name) > prio(current)" in GATE


def test_gate_shows_banners_one_at_a_time_in_priority_order(tmp_path):
    """Behavioral (node): banners request in DOM arm order (discord,
    google, trial, dunning, recap, leaguepro); the highest priority
    (dunning) holds the slot, and each dismissal surfaces the next
    most important banner. Never two visible at once."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    gate_file = tmp_path / "gate.js"
    gate_file.write_text(GATE, encoding="utf-8")
    harness = tmp_path / "harness.js"
    harness.write_text(
        """
const fs = require('fs');
const window = {};
global.window = window;
eval(fs.readFileSync(process.argv[2], 'utf8'));
const gate = window.brPromoGate;
const vis = {};
const prio = { dunning: 90, google: 40, trial: 50, recap: 60, leaguepro: 20, discord: 10 };
for (const name of Object.keys(prio)) {
  vis[name] = false;
  gate.register(name, () => { vis[name] = true; }, () => { vis[name] = false; }, prio[name]);
}
const out = [];
// DOM arm order on a page where every banner is eligible.
for (const name of ['discord', 'google', 'trial', 'dunning', 'recap', 'leaguepro']) {
  gate.request(name);
  out.push({ step: 'request-' + name, current: gate.current(),
             visible: Object.keys(vis).filter(k => vis[k]) });
}
for (const name of ['dunning', 'recap', 'trial', 'google', 'leaguepro', 'discord']) {
  gate.dismiss(name);
  out.push({ step: 'dismiss-' + name, current: gate.current(),
             visible: Object.keys(vis).filter(k => vis[k]) });
}
console.log(JSON.stringify(out));
""",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [node, str(harness), str(gate_file)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    steps = {row["step"]: row for row in json.loads(proc.stdout)}
    # After all requests, only the most urgent banner is visible.
    assert steps["request-leaguepro"]["current"] == "dunning"
    assert steps["request-leaguepro"]["visible"] == ["dunning"]
    # Dismissals surface the queue in priority order.
    order = ["dunning", "recap", "trial", "google", "leaguepro", "discord"]
    for i, name in enumerate(order):
        row = steps[f"dismiss-{name}"]
        nxt = order[i + 1] if i + 1 < len(order) else None
        assert row["current"] == nxt, f"after {name}: {row}"
        assert row["visible"] == ([nxt] if nxt else []), f"after {name}: {row}"
    for row in steps.values():
        assert len(row["visible"]) <= 1


def test_gate_count_label_tracks_position_in_run(tmp_path):
    """Behavioral (node): the showing banner's count label is its
    position in the run by priority, "1/6" through "6/6"; the total
    never shrinks as banners dismiss, and a lone banner gets none."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    gate_file = tmp_path / "gate.js"
    gate_file.write_text(GATE, encoding="utf-8")
    harness = tmp_path / "harness.js"
    harness.write_text(
        """
const fs = require('fs');
const window = {};
global.window = window;
eval(fs.readFileSync(process.argv[2], 'utf8'));
const gate = window.brPromoGate;
const prio = { dunning: 90, google: 40, trial: 50, recap: 60, leaguepro: 20, discord: 10 };
for (const name of Object.keys(prio)) {
  gate.register(name, () => {}, () => {}, prio[name]);
}
const out = [];
for (const name of ['discord', 'google', 'trial', 'dunning', 'recap', 'leaguepro']) {
  gate.request(name);
}
out.push({ step: 'all-requested', current: gate.current(), label: gate.countLabel() });
for (const name of ['dunning', 'recap', 'trial', 'google', 'leaguepro', 'discord']) {
  gate.dismiss(name);
  out.push({ step: 'dismiss-' + name, current: gate.current(), label: gate.countLabel() });
}
// A fresh run with a single banner carries no count.
gate.request('discord');
out.push({ step: 'lone-discord', current: gate.current(), label: gate.countLabel() });
console.log(JSON.stringify(out));
""",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [node, str(harness), str(gate_file)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    steps = {row["step"]: row for row in json.loads(proc.stdout)}
    assert steps["all-requested"]["current"] == "dunning"
    assert steps["all-requested"]["label"] == "1/6"
    order = ["dunning", "recap", "trial", "google", "leaguepro", "discord"]
    for i, name in enumerate(order):
        row = steps[f"dismiss-{name}"]
        nxt = order[i + 1] if i + 1 < len(order) else None
        assert row["current"] == nxt, f"after {name}: {row}"
        assert row["label"] == (f"{i + 2}/6" if nxt else ""), f"after {name}: {row}"
    assert steps["lone-discord"]["current"] == "discord"
    assert steps["lone-discord"]["label"] == ""


# ---------------------------------------------------------------- banners

@pytest.mark.parametrize("slice_,name,prio", [
    (RECAP, "recap", "60"),
    (TRIAL, "trial", "50"),
    (DUNNING, "dunning", "90"),
])
def test_server_banners_register_with_gate(slice_, name, prio):
    assert f"brPromoGate.register('{name}', show, hide, {prio})" in slice_
    assert f"brPromoGate.request('{name}')" in slice_
    assert f"brPromoGate.dismiss('{name}')" in slice_
    # Arming waits for DOMContentLoaded, when the deferred gate exists.
    assert "DOMContentLoaded', arm" in slice_ or "'DOMContentLoaded', arm" in slice_


def test_google_banner_registers_hard_and_soft_priorities():
    assert "brPromoGate.register('google', show, hide, hard ? 80 : 40)" in GOOGLE
    assert "brPromoGate.request('google')" in GOOGLE
    assert "brPromoGate.dismiss('google')" in GOOGLE
    # Both modes start hidden; the gate shows the banner in turn.
    assert 'display = "flex"' not in GOOGLE


def test_dunning_banner_starts_hidden():
    assert "display:none;flex-direction:column" in DUNNING


def test_discord_registers_with_lowest_priority():
    assert "brPromoGate.register('discord', show, hide, 10)" in DISCORD


def test_league_pro_invite_uses_gate():
    assert "brPromoGate.register('leaguepro', showLeaguePro, hideLeaguePro, 20)" in PAYWALL_JS
    assert "brPromoGate.request('leaguepro')" in PAYWALL_JS
    assert "brPromoGate.dismiss('leaguepro')" in PAYWALL_JS


def test_every_banner_carries_a_count_slot_by_its_close_button():
    # The gate paints the "1/x" label into a span parked right before
    # each banner's close button; no span, no count.
    assert 'data-qcount="recap"' in RECAP
    assert 'data-qcount="google"' in GOOGLE
    assert 'data-qcount="trial"' in TRIAL
    assert 'data-qcount="dunning"' in DUNNING
    assert 'data-qcount="discord"' in DISCORD
    assert PAYWALL_JS.count('data-qcount="leaguepro"') == 3
    assert "setAttribute('data-qcount', 'assistant')" in JS
    assert "querySelectorAll('[data-qcount=\"'" in GATE


def test_discord_phone_bar_count_sits_before_close():
    media = DISCORD[DISCORD.index("@media (max-width: 768px)"):]
    # Count takes the slot after the Join pill; the close stays last.
    assert "#discordBanner .br-qcount {{" in media
    assert "order:5" in media  # close button, after the count


def test_discord_phone_bar_close_last_and_no_copy():
    media = DISCORD[DISCORD.index("@media (max-width: 768px)"):]
    # (Braces are doubled in the app.py f-string source.)
    assert "#discordBanner .discord-banner-head {{ display:contents !important; }}" in media
    assert "#discordBanner .discord-banner-copy {{ display:none !important; }}" in media
    assert "order:4" in media  # close button in the last slot


# ---------------------------------------------------------------- not me

def test_reset_return_capture_and_restore_live_in_core():
    # Guests get the public core bundle; the stash/restore must be in it.
    assert "br_reset_return" in CORE
    assert "initResetReturn" in CORE
    assert "a[href=\"/reset-user\"]" in CORE
    # This app scrolls <body>: restore writes every scroll candidate.
    assert "document.body.scrollTop = y" in CORE


def test_reset_interstitial_returns_to_stashed_page():
    assert "br_reset_return" in AUTH
    # Same-origin path only, never back into the reset routes, and the
    # redirect actually uses the computed target.
    assert "_saved.u.charAt(1) !== '/'" in AUTH
    assert "_saved.u.indexOf('/reset-user') !== 0" in AUTH
    assert "window.location.replace(_target)" in AUTH
    # The identity cleanup the route exists for is untouched.
    assert "localStorage.removeItem('saved_viewer')" in AUTH
    assert "localStorage.removeItem('saved_account')" in AUTH


# ---------------------------------------------------------------- notif scroll

def test_notif_modals_capped_by_overlay_not_large_viewport():
    assert "max-height:calc(100vh - 96px)" not in JS
    # Picker + settings modals (and the subscribe prompt shares the cap).
    assert JS.count("max-height:100%;overflow-y:auto") == 3
    assert "-webkit-overflow-scrolling:touch" in JS
    assert "overscroll-behavior:contain" in JS


def test_picker_actions_stay_pinned_while_scrolling():
    picker = JS[JS.index("window.openPushPicker"):JS.index("window.openNotifPrefs")]
    assert "position:sticky;bottom:-20px" in picker
    assert 'id="ppDone"' in picker and 'id="ppSkip"' in picker
