"""Contract guards for GUI audit PR B (Findings 2 and 6).

Finding 2: the assistant intro/teaser promo and the Discord banner used
to render simultaneously on the guest home and physically overlap
(88px, bottom 28% of the phone viewport). The fix is a shared promo
queue in the public JS core (window.brPromoGate): one floating promo at
a time, the assistant holds priority, the Discord banner only takes the
slot when it is free, and on phones the banner renders as a slim
one-line bar above the dock instead of the 158px card.

Finding 6: the locked 6-step type scale (11/13/15/18/22/28) from the
October 6 audit is adopted as --fs-* variables in the public
stylesheets (landing_lite / seo_lite / paywall) and declarations that
map to a step (exact or within 0.5px) use the variables. dashboard.css
is explicitly out of scope for this pass.
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
LANDING = (ROOT / "static" / "landing_lite.css").read_text(encoding="utf-8")
SEO = (ROOT / "static" / "seo_lite.css").read_text(encoding="utf-8")
PAYWALL = (ROOT / "static" / "paywall.css").read_text(encoding="utf-8")
SW = (ROOT / "static" / "sw.js").read_text(encoding="utf-8")

MARKER = "// @public-js:core-end"
CORE = JS.split(MARKER, 1)[0]

_GATE_START = JS.index("window.brPromoGate = (function () {")
_GATE_END = JS.index("})();", _GATE_START) + len("})();")
GATE = JS[_GATE_START:_GATE_END]

_MOUNT_START = JS.index("function initGuestAssistantTeaser()")
_MOUNT_END = JS.index("})();", _MOUNT_START) + len("})();")
MOUNT = JS[_MOUNT_START:_MOUNT_END]

_BANNER_START = APP_PY.index("def _discord_banner()")
_BANNER_END = APP_PY.index('"""', APP_PY.index("<script>", _BANNER_START))
BANNER = APP_PY[_BANNER_START:_BANNER_END]


# ---------------------------------------------------------------- gate

def test_gate_lives_in_public_core():
    """Guests are served public.js (above the marker); the gate must be
    in that slice or the guest-home queue never runs."""
    assert _GATE_START < JS.index(MARKER)
    assert "window.brPromoGate" in CORE


def test_gate_api_and_assistant_priority():
    for needle in ("register", "request", "release", "dismiss", "current"):
        assert needle in GATE
    # The assistant preempts; other promos queue.
    assert "name === 'assistant'" in GATE


def test_gate_never_shows_two_promos(tmp_path):
    """Behavioral: drive the real gate source in node through the audit
    scenario. Both promos can never be visible at once, the Discord
    promo surfaces after the assistant retires, and a promo dismissed
    while queued is never resurrected by a later release."""
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
const vis = { discord: false, assistant: false };
gate.register('discord', () => { vis.discord = true; }, () => { vis.discord = false; });
gate.register('assistant', () => { vis.assistant = true; }, () => { vis.assistant = false; });
const out = [];
// Discord first (its banner script arms at DOMContentLoaded, the
// teaser mount runs in the same dispatch): it shows.
gate.request('discord');
out.push({ step: 'discord-first', ...vis, current: gate.current() });
// Assistant requests: preempts, Discord hides.
gate.request('assistant');
out.push({ step: 'assistant-preempts', ...vis, current: gate.current() });
// Assistant retires (its retire() removes the bubble itself, then
// releases the slot): queued Discord returns.
vis.assistant = false;
gate.release('assistant');
out.push({ step: 'assistant-retires', ...vis, current: gate.current() });
// Repeat, but Discord is dismissed while queued behind the assistant:
// releasing the assistant must NOT resurrect it.
gate.dismiss('discord');
vis.discord = false; vis.assistant = false;
gate.request('assistant');
gate.request('discord');
gate.dismiss('discord');
vis.assistant = false;
gate.release('assistant');
out.push({ step: 'dismissed-while-queued', ...vis, current: gate.current() });
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
    assert steps["discord-first"] == {
        "step": "discord-first", "discord": True,
        "assistant": False, "current": "discord",
    }
    assert steps["assistant-preempts"] == {
        "step": "assistant-preempts", "discord": False,
        "assistant": True, "current": "assistant",
    }
    assert steps["assistant-retires"] == {
        "step": "assistant-retires", "discord": True,
        "assistant": False, "current": "discord",
    }
    assert steps["dismissed-while-queued"]["discord"] is False
    assert steps["dismissed-while-queued"]["current"] is None
    for row in steps.values():
        assert not (row["discord"] and row["assistant"])


# ---------------------------------------------------------------- teaser

def test_guest_teaser_uses_gate():
    assert "brPromoGate.register('assistant'" in MOUNT
    assert "brPromoGate.request('assistant')" in MOUNT
    assert "brPromoGate.release('assistant')" in MOUNT
    # The bubble only enters the DOM through the gate's show callback.
    show_idx = MOUNT.index("function showBubble()")
    assert MOUNT.index("appendChild(bubble)", show_idx) > show_idx
    assert MOUNT.index("document.body.appendChild(bubble)") > show_idx


def test_signed_in_intro_uses_gate():
    # Both assistant intro surfaces (guest teaser + signed-in bubble)
    # register with the gate and release it on retire.
    assert JS.count("brPromoGate.register('assistant'") >= 2
    assert JS.count("brPromoGate.release('assistant')") >= 2


# ---------------------------------------------------------------- banner

def test_discord_banner_uses_gate_and_keeps_dismissal():
    assert "brPromoGate.register('discord'" in BANNER
    assert "brPromoGate.request('discord')" in BANNER
    assert "brPromoGate.dismiss('discord')" in BANNER
    # The weekly dismissal persistence is untouched: a prior dismissal
    # returns before the gate is ever armed.
    assert "localStorage.getItem(key)" in BANNER
    assert "localStorage.setItem(key, '1')" in BANNER
    assert BANNER.index("localStorage.getItem(key)") < BANNER.index(
        "brPromoGate.register('discord'"
    )


def test_discord_banner_phone_slim_bar():
    assert "discord-banner-head" in BANNER
    assert "discord-banner-copy" in BANNER
    assert "discord-banner-link" in BANNER
    # Inside the phone media query the card becomes a one-line row.
    media_idx = BANNER.index("@media (max-width: 768px)")
    assert "flex-direction:row !important" in BANNER[media_idx:]
    assert "text-overflow:ellipsis" in BANNER[media_idx:]


# ---------------------------------------------------------------- type scale

SCALE = {
    "--fs-micro": "11px",
    "--fs-sm": "13px",
    "--fs-body": "15px",
    "--fs-title": "18px",
    "--fs-h2": "22px",
    "--fs-hero": "28px",
}

PUBLIC_SHEETS = {"landing_lite": LANDING, "seo_lite": SEO, "paywall": PAYWALL}


@pytest.mark.parametrize("sheet", sorted(PUBLIC_SHEETS))
def test_type_scale_variables_defined(sheet):
    text = PUBLIC_SHEETS[sheet]
    for var, value in SCALE.items():
        assert f"{var}: {value};" in text, f"{sheet} missing {var}"


@pytest.mark.parametrize("sheet", sorted(PUBLIC_SHEETS))
def test_type_scale_variables_used(sheet):
    text = PUBLIC_SHEETS[sheet]
    uses = sum(text.count(f"var({var})") for var in SCALE)
    assert uses >= 25, f"{sheet} only uses the scale {uses} times"


def test_type_scale_fractional_sizes_snapped():
    """The one-off fractional sizes that sat within 0.5px of a step are
    gone from the public sheets (they now render through the step)."""
    for sheet, text in PUBLIC_SHEETS.items():
        for frac in ("10.5px", "11.5px", "12.5px", "13.5px"):
            assert f"font-size: {frac}" not in text, f"{sheet} keeps {frac}"
            assert f"font-size:{frac}" not in text, f"{sheet} keeps {frac}"


def test_dashboard_css_font_sizes_untouched_by_this_pass():
    """dashboard.css font sizes are opportunistic-later (audit wording);
    this PR must not carry a big-bang sweep of it."""
    dash = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    assert "--fs-micro" not in dash


def test_sw_bumped():
    # Authored at v52; the merge sweep carried the cache past it (v53 in
    # PR #2704, v54 with this PR's merge). Invariant: bumped past v51.
    assert "br-fantasy-v51" not in SW
    assert "br-fantasy-v52" in SW or "br-fantasy-v53" in SW or "br-fantasy-v54" in SW or "br-fantasy-v55" in SW or "br-fantasy-v56" in SW or "br-fantasy-v57" in SW or "br-fantasy-v58" in SW or "br-fantasy-v60" in SW or "br-fantasy-v61" in SW or "br-fantasy-v64" in SW or "br-fantasy-v66" in SW or "br-fantasy-v67" in SW or "br-fantasy-v68" in SW or "br-fantasy-v71" in SW or "br-fantasy-v72" in SW or "br-fantasy-v73" in SW or "br-fantasy-v74" in SW or "br-fantasy-v75" in SW or "br-fantasy-v76" in SW or "br-fantasy-v77" in SW or "br-fantasy-v78" in SW or "br-fantasy-v79" in SW or "br-fantasy-v80" in SW or "br-fantasy-v81" in SW or "br-fantasy-v82" in SW or "br-fantasy-v85" in SW or "br-fantasy-v83" in SW or "br-fantasy-v86" in SW or "br-fantasy-v87" in SW or "br-fantasy-v88" in SW or "br-fantasy-v89" in SW or "br-fantasy-v90" in SW or "br-fantasy-v91" in SW or "br-fantasy-v92" in SW
