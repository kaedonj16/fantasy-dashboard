"""Team-name hover contract: names dim, they never lift like cards.

Kaedon reported (2026-10-13, with a matchup-header screenshot) a weird
stretched-out plate on hover "in some spots on the site". Root cause: the
hover rule in dashboard.css was keyed on the element tag -- every
div/article.team-clickable got a card lift (translateY(-2px) + box-shadow).
Most div carriers are plain team names sitting in a much wider grid/flex
cell (.m-espn-tname in the matchup header, .team-name in activity and
score rows), so the lift painted an empty plate across the whole cell,
far past the text. Span carriers only dimmed, which is why it appeared
in some spots and not others.

The fix makes the dim tag-agnostic (the .player-clickable precedent in
test_player_name_hover.py): every .team-clickable dims to opacity 0.72
on hover except table rows (their cells highlight instead) and
.os-snap-row (its own border/background hover). These tests read the
shipped CSS source and lock that in.
"""
import os
import re

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_CSS_PATH = os.path.join(_ROOT, "static", "dashboard.css")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _rules(css):
    """Yield (selector, body) pairs for flat rules in the stylesheet."""
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        yield match.group(1).strip(), match.group(2)


def test_no_team_clickable_card_lift_anywhere():
    css = _read(_CSS_PATH)
    offenders = []
    for selector, body in _rules(css):
        if "team-clickable" not in selector or ":hover" not in selector:
            continue
        lowered = body.lower()
        if "translatey" in lowered or "box-shadow" in lowered:
            offenders.append((selector, body.strip()))
    assert not offenders, (
        "team-clickable hover must never lift/shadow (stretched plate bug): "
        f"{offenders}"
    )


def test_team_clickable_hover_dims_regardless_of_tag():
    css = _read(_CSS_PATH)
    match = re.search(
        r"(?m)^\.team-clickable:hover:not\(tr\):not\(\.os-snap-row\)\s*\{([^}]*)\}",
        css,
    )
    assert match, "unified .team-clickable hover dim rule missing"
    compact = re.sub(r"\s+", "", match.group(1).lower())
    assert "opacity:0.72" in compact, (
        f"team-clickable hover must dim to opacity 0.72, got: {match.group(1).strip()}"
    )


def test_old_tag_keyed_lift_rule_is_gone():
    css = _read(_CSS_PATH)
    assert "div.team-clickable:hover" not in css
    assert "article.team-clickable:hover" not in css


def test_table_row_highlight_kept():
    css = _read(_CSS_PATH)
    assert "tr.team-clickable:hover td" in css


def test_os_snap_row_keeps_its_own_hover():
    css = _read(_CSS_PATH)
    match = re.search(r"(?m)^\.os-snap-row:hover\s*\{([^}]*)\}", css)
    assert match, ".os-snap-row own hover rule missing"
    assert "border-color" in match.group(1)
