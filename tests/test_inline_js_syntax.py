"""Guardrail: every inline <script> embedded in Python templates must be valid JS.

Large chunks of UI (most of the Draft Room, the analytics/draft accordion, the
trade pages, …) live as JavaScript inside Python string templates. Python never
parses that JS and no JS linter sees it, so a stray token pasted into one of
those strings is a silent, page-breaking SyntaxError that ships unnoticed (this
test exists because exactly that happened: `}scored 98 this` wedged into the
Draft Room script took the entire page down).

The test extracts each self-contained inline <script> block via AST (so it can
neutralize f-string interpolation precisely) and runs it through `node --check`.
It is skipped when Node.js isn't available so it never blocks a Node-less env.
"""
from __future__ import annotations

import ast
import glob
import os
import re
import shutil
import subprocess
import tempfile

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_RE = re.compile(r"<script\b[^>]*>(.*?)</script>", re.DOTALL)


def _source_files() -> list[str]:
    pats = ["app.py", "dashboard_services/**/*.py", "routes/*.py"]
    out: list[str] = []
    for pat in pats:
        out.extend(glob.glob(os.path.join(_REPO_ROOT, pat), recursive=True))
    return sorted(set(out))


def _format_template_values(tree: ast.AST) -> set[str]:
    """Collect string values that are used as ``str.format`` templates.

    A ``.format`` template carrying JS has single-brace ``{field}`` placeholders
    that aren't valid JS, so we can't node-check it — but literal JS also
    contains braces, so we can't tell them apart by content. Instead we detect
    the ``.format`` call directly: ``"...".format(...)`` and the common
    ``TEMPLATE = "..."`` / ``TEMPLATE.format(...)`` indirection.
    """
    name_to_value: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    name_to_value[tgt.id] = node.value.value
    fmt: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ("format", "format_map"):
            recv = node.func.value
            if isinstance(recv, ast.Constant) and isinstance(recv.value, str):
                fmt.add(recv.value)
            elif isinstance(recv, ast.Name) and recv.id in name_to_value:
                fmt.add(name_to_value[recv.id])
    return fmt


def _string_nodes(tree: ast.AST):
    """Yield (text, lineno) for *literal* JS templates only.

    We check plain/raw ``Constant`` strings (e.g. the Draft Room's
    ``_DRAFT_ROOM_HTML = r"..."``) and skip f-strings and ``str.format``
    templates, whose ``{field}`` placeholders aren't valid JS and can't be
    reconstructed reliably (that would make the test flaky). This still covers
    the big raw-string JS bodies where a stray token silently breaks a page.
    """
    fmt_values = _format_template_values(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in fmt_values:
                continue  # str.format template — placeholders aren't valid JS
            yield node.value, node.lineno


def _inline_scripts() -> list[tuple[str, int, str]]:
    blocks: list[tuple[str, int, str]] = []
    for path in _source_files():
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except Exception:
            continue
        for text, lineno in _string_nodes(tree):
            if "<script" not in text or "</script>" not in text:
                continue
            for m in _SCRIPT_RE.finditer(text):
                js = m.group(1)
                if len(js) < 80:  # tiny config injectors carry no real logic
                    continue
                blocks.append((os.path.relpath(path, _REPO_ROOT), lineno, js))
    return blocks


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_inline_scripts_parse():
    blocks = _inline_scripts()
    assert blocks, "expected to find inline <script> blocks to check"
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        for i, (path, lineno, js) in enumerate(blocks):
            fp = os.path.join(td, f"b{i}.js")
            with open(fp, "w", encoding="utf-8") as fh:
                fh.write(js)
            res = subprocess.run(
                ["node", "--check", fp], capture_output=True, text=True
            )
            if res.returncode != 0:
                err = res.stderr.strip().splitlines()
                msg = next((l for l in err if "SyntaxError" in l), err[-1] if err else "")
                failures.append(f"{path} (string starting ~L{lineno}): {msg}")
    assert not failures, "Inline <script> JavaScript failed to parse:\n" + "\n".join(failures)


# First-party static JS we hand-maintain (draft_room.js was extracted from an inline
# template, so keep it under the same guard). Excludes vendored / generated *.min.js.
_STATIC_JS = ["app.js", "draft_room.js", "redzone.js", "player_page.js", "paywall.js", "sw.js"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_static_js_parses():
    failures: list[str] = []
    checked = 0
    for name in _STATIC_JS:
        fp = os.path.join(_REPO_ROOT, "static", name)
        if not os.path.exists(fp):
            continue
        checked += 1
        res = subprocess.run(["node", "--check", fp], capture_output=True, text=True)
        if res.returncode != 0:
            err = res.stderr.strip().splitlines()
            msg = next((l for l in err if "SyntaxError" in l), err[-1] if err else "")
            failures.append(f"static/{name}: {msg}")
    assert checked, "expected to find first-party static JS to check"
    assert not failures, "Static JavaScript failed to parse:\n" + "\n".join(failures)
