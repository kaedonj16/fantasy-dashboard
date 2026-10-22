"""Pin the landing page's phone stacking rules.

The A+ landing grids (hero, PRO, free tools, demos) are multi-column on
desktop and must collapse inside the max-width:1024px media blocks. A
connect-card CSS rework once deleted those lines from the shared media
block (they lived in the same block as the connect stack), which left
the hero board and PRO grid crushed into slivers on phones while every
desktop check stayed green. These assertions keep the block whole.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

REQUIRED = [
    ".home-hero.ld-hero { grid-template-columns: 1fr; }",
    ".ld-pro { grid-template-columns: 1fr; }",
    ".ld-pro-grid { grid-template-columns: 1fr; }",
    ".ld-free-grid { grid-template-columns: 1fr; }",
    ".ld-demo-opts { grid-template-columns: 1fr 1fr; }",
    ".ld-hero-card { padding: 24px 22px; }",
    ".ld-card { padding: 22px 20px; }",
    ".ld-connect-grid { grid-template-columns: 1fr; }",
]


def _media_1024_text(css: str) -> str:
    """Concatenate every @media (max-width: 1024px) block's body."""
    out = []
    for m in re.finditer(r"@media \(max-width: 1024px\) \{", css):
        depth = 1
        i = m.end()
        while i < len(css) and depth:
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
            i += 1
        out.append(css[m.end():i - 1])
    return "\n".join(out)


@pytest.mark.parametrize("sheet", ["static/landing_lite.css", "static/dashboard.css"])
def test_landing_stacks_on_phones(sheet):
    css = (ROOT / sheet).read_text()
    block = _media_1024_text(css)
    assert block, f"no max-width:1024px block in {sheet}"
    for rule in REQUIRED:
        assert rule in block, f"{sheet} lost its phone stacking rule: {rule}"
