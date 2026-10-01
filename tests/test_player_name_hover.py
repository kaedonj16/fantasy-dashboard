"""Player-name hover contract: names dim like team names, never blue/underline.

Kaedon's call (2026-10-01): hovering a player name turned it bright blue with
an underline, while team names just dim slightly (opacity 0.72, see the
span.team-clickable rule in dashboard.css). Player names now use the same dim
everywhere: no brand-blue / accent / #38bdf8 color, no underline, no lift.
These tests read the shipped CSS/JS sources and lock that in.
"""
import os
import re

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_CSS_PATH = os.path.join(_ROOT, "static", "dashboard.css")
_JS_PATH = os.path.join(_ROOT, "static", "app.js")
_APP_PATH = os.path.join(_ROOT, "app.py")
_ROOKIES_PATH = os.path.join(_ROOT, "dashboard_services", "pages", "rookies_page.py")

_FORBIDDEN = ("brand-blue", "--accent", "#38bdf8", "underline", "translatey")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _rule_body(css, selector):
    """Return the declaration body of the first rule containing `selector`."""
    idx = css.find(selector)
    assert idx != -1, f"selector not found: {selector}"
    open_brace = css.find("{", idx)
    close_brace = css.find("}", open_brace)
    assert open_brace != -1 and close_brace != -1, f"rule body not found: {selector}"
    return css[open_brace + 1:close_brace].lower()


def _assert_dim(body, selector):
    compact = re.sub(r"\s+", "", body)
    assert "opacity:0.72" in compact or "opacity:.72" in compact, (
        f"{selector} must dim to opacity 0.72 on hover, got: {body.strip()}"
    )
    for token in _FORBIDDEN:
        assert token not in body, f"{selector} hover still uses {token}: {body.strip()}"


def test_global_player_clickable_hover_dims():
    css = _read(_CSS_PATH)
    match = re.search(r"(?m)^\.player-clickable:hover\s*\{([^}]*)\}", css)
    assert match, "global .player-clickable:hover rule missing"
    _assert_dim(match.group(1).lower(), ".player-clickable:hover")


# Every player-name hover override that used to go blue / underlined.
_DIM_SELECTORS = [
    ".otc-breakout-row .otc-player-name:hover",
    ".breakout-player-name.player-clickable:hover",
    ".usage-mover.player-clickable:hover .um-name",
    ".tm-mu-hname.player-clickable:hover",
    ".tm-team-link:hover",
    ".opt-player-name:hover",
    ".dvt-player-link:hover",
    ".rnk-player-link:hover",
    ".rf-name:hover",
    "a.otc-value-name:hover",
    ".inj-pname:hover",
    ".whl-name:hover",
    ".sched-pname:hover",
    ".team-strength-card .pos-detail-inner .player-clickable:hover",
    ".compare-chip:hover .compare-chip-name",
]


def test_player_name_hover_overrides_dim():
    css = _read(_CSS_PATH)
    for selector in _DIM_SELECTORS:
        _assert_dim(_rule_body(css, selector), selector)


def test_usage_mover_row_itself_does_not_dim():
    """The usage-mover row is a block container: only its name dims."""
    css = _read(_CSS_PATH)
    body = _rule_body(css, ".usage-mover.player-clickable:hover {")
    assert "opacity: 1" in body, f"usage-mover row must neutralize the dim: {body.strip()}"


def test_focus_visible_styles_untouched():
    css = _read(_CSS_PATH)
    assert ".opt-player-name:focus-visible" in css
    assert "outline:2px solid var(--accent)" in css


def test_rookie_names_dim():
    src = _read(_ROOKIES_PATH)
    assert ".rk-name:hover { opacity: 0.72; }" in src
    assert ".rk-name:hover { text-decoration: underline; }" not in src


def test_compare_modal_names_dim():
    src = _read(_JS_PATH)
    assert ".cmp3-head:hover .cmp3-name{opacity:.72;}" in src
    assert ".cmp3-head:hover .cmp3-name{text-decoration:underline;}" not in src


def test_portfolio_movers_dim_name_not_row():
    """pfm-row is a block-level .player-clickable: the row keeps its own
    background hover, only the name dims."""
    src = _read(_APP_PATH)
    assert ".pfm-row.player-clickable:hover{opacity:1;}" in src
    assert ".pfm-row.player-clickable:hover .pfm-name{opacity:.72;}" in src
