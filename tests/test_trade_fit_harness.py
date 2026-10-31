"""Behavioral harness wrapper for the Trade Calculator Fit grade JS.

The contracts live in tests/trade_fit_harness.mjs, which extracts the
shipped functions from static/app.js by source and drives them with
stub state: the Fit composite averages only the components that exist,
an ungraded trade exposes no components at all, and the playoff
component is read only when the sim's signature matches the trade on
screen.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_trade_fit_harness():
    node = shutil.which("node")
    if not node:
        import pytest

        pytest.skip("node not available for the JS behavioral harness")
    result = subprocess.run(
        [node, str(ROOT / "tests" / "trade_fit_harness.mjs")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CHECKS PASSED" in result.stdout, result.stdout + result.stderr
