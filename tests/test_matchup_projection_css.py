"""Regression coverage for matchup projection typography ownership."""
from pathlib import Path
import re


CSS = (Path(__file__).parents[1] / "static" / "dashboard.css").read_text(encoding="utf-8")


def test_row_projection_modifier_owns_smaller_type_after_primary_score_rules():
    matches = list(re.finditer(r"\.m-row\s+\.num\.proj\s*\{([^}]*)\}", CSS))
    assert matches
    rule = matches[-1]
    assert "font-size: 0.8em" in rule.group(1)
    # It must occur after the broad .num.mid rule which caused the regression,
    # and specificity must not depend on the l/r class order.
    assert rule.start() > CSS.index(".num.mid {")
    assert ".num.mid.l.proj" not in rule.group(0)
    assert ".num.mid.r.proj" not in rule.group(0)


def test_mobile_mid_rule_cannot_override_scoped_projection_size():
    assert ".num.mid {\n        min-width: 26px;\n        font-size: 11px;" in CSS
    assert CSS.count(".m-row .num.proj") == 1
