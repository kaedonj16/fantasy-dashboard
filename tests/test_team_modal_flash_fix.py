"""Regression coverage for the team-modal size flash and pick-via spacing.

The flash: a :has() rule turned .team-modal-body into a centered flex
container whenever ANY .team-modal-loading / .team-modal-error node
existed inside it (a tab's first load, or one schedule week's lineup
fetch). The active panel became a shrink-to-fit flex item, so the modal
flashed big -> small -> big on every load. The modal also had no fixed
height, so a spinner-only tab collapsed its height the same way.

The pick chips rendered "R2" + a via span separated by a single 3px
space in a smaller muted font, which read as one token ("R2via X").
They now carry a middot separator plus a CSS margin, matching the
player modal's "R2 · via X" treatment.
"""
from pathlib import Path

ROOT = Path(__file__).parents[1]
CSS = (ROOT / "static" / "dashboard.css").read_text()
JS = (ROOT / "static" / "app.js").read_text()


def _block(css, selector, start=0):
    """Return the declaration block body for the first `selector { ... }`."""
    i = css.index(selector, start)
    j = css.index("{", i)
    k = css.index("}", j)
    return css[j + 1:k]


def test_no_has_flex_override_on_modal_body():
    assert ".team-modal-body:has(" not in CSS
    # The loading/error blocks keep their own centering instead.
    loading = _block(CSS, ".team-modal-loading,")
    assert "width: 100%" in loading
    assert "align-items: center" in loading


def test_team_modal_has_fixed_height_desktop_and_mobile():
    base = _block(CSS, ".team-modal {")
    assert "height: 90vh" in base
    assert "height: 90dvh" in base
    # The <=768px sheet pins its height to the same expression as its
    # max-height so phones do not resize between loading and loaded.
    anchor = CSS.index("Fixed height so the sheet doesn't resize")
    mobile = _block(CSS, ".team-modal {", CSS.rindex("@media", 0, anchor))
    assert "height: auto" not in mobile
    assert "height: calc(100vh - 24px)" in mobile
    assert "100dvh" in mobile


def test_pick_via_has_middot_separator_and_margin():
    assert 'tm3-pick-via">· via' in JS
    assert 'tm2-pick-via">· via' in JS
    assert 'tm3-pick-via">via' not in JS
    assert 'tm2-pick-via">via' not in JS
    assert "margin-left: 3px" in _block(CSS, ".tm2-pick-via")
    assert "margin-left: 3px" in _block(CSS, ".tm3-pick-via")
