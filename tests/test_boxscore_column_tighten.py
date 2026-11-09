"""Box-score stat columns: tightened and aligned across position groups.

Both sheets that render .rz-bs-table (ScoreZone's own box-score sheet in
scorezone.js and the Matchups game sheet's Stats tab in app.js) share one
column system: fixed table layout, a colgroup with FP at 48px and every
stat column at 52px, the name column absorbing the leftover width, and an
inline min-width of 128 + 48 + 52*n per table so a wide group scrolls
instead of squeezing - at that min-width the name column is exactly 128px,
which is the FP column's sticky offset.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
SCOREZONE_JS = (ROOT / "static" / "scorezone.js").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def test_both_builders_emit_the_tight_colgroup():
    for js in (SCOREZONE_JS, APP_JS):
        assert 'class="rz-bs-cname"' in js
        assert 'class="rz-bs-cfp"' in js
        assert 'class="rz-bs-cstat"' in js
        # Inline min-width: 128 name floor + 48 FP + 52 per stat column,
        # on the sortable table itself.
        assert "(176 + 52 * cols.length)" in js
        assert 'data-sortable style="min-width:' in js


def test_css_column_widths_are_tight_and_uniform():
    assert "table-layout: fixed" in CSS
    assert "table.rz-bs-table col.rz-bs-cfp { width: 48px; }" in CSS
    assert "table.rz-bs-table col.rz-bs-cstat { width: 52px; }" in CSS
    # The old loose widths are gone from the tightened column block.
    block = CSS.split("Tight, uniform stat columns")[-1].split("Trade Hub")[0]
    assert "width: 68px" not in block
    assert "width: 56px" not in block
    assert "width: 52px" in block and "width: 48px" in block


def test_fp_sticky_offset_matches_name_floor():
    # FP sticks at the name column's floor width (128px), which the
    # builders' min-width formula guarantees at scroll widths.
    sticky_section = CSS.split("fantasy points column")[-1].split("Group title")[0]
    assert "left: 128px" in sticky_section
    assert "left: 108px" not in sticky_section
