"""Power-rankings rank numbers are never streak-colored.

Kaedon's call (2026-10-01): on the power rankings list, teams on a win
streak of 2+ had their rank number painted with the loss red, which read
as an error, while losing-streak teams got an accent tint that looked
identical to the default. The coloring is removed: .pwr-pos keeps its
base muted color for every row, streak classes no longer recolor it.
These tests read the shipped CSS and lock that in.
"""
import os

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_CSS_PATH = os.path.join(_ROOT, "static", "dashboard.css")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def test_pwr_pos_base_rule_survives():
    css = _read(_CSS_PATH)
    assert ".pwr-pos {" in css
    assert "color: var(--text-muted)" in css


def test_streak_classes_do_not_recolor_rank_numbers():
    css = _read(_CSS_PATH)
    assert ".pwr-row.streak-hot .pwr-pos" not in css
    assert ".pwr-row.streak-cold .pwr-pos" not in css
