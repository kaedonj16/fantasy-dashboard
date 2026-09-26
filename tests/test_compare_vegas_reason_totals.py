"""The compare "a higher team total" reason must compare implied totals.

Regression test: the Start/Sit compare header once claimed the winner had "a
higher team total" when the *Vegas multiplier* (a position-adjusted haircut)
was higher, even though the loser's actual implied total was larger (e.g. an
RB at 10 implied with a 0.96 multiplier "beating" a WR at 15.12 implied with a
0.92 multiplier). The reason text names the team total, so the comparison must
use implied totals, not multipliers. This covers both copies of the logic: the
waivers-page inline script and the Start/Sit page's app.js port.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")

ROOT = Path(__file__).resolve().parents[1]

# Judkins (RB, winner on projection): implied 10 -> 0.96 multiplier.
# Wilson (WR): implied 15.12 -> 0.92 multiplier. The multiplier comparison
# says the winner "wins" the vegas factor; the totals say he does not.
A_WIN = {"implied_total": 10, "score_factors": {"proj": 12.3, "vegas": 0.96}}
B_LOSE = {"implied_total": 15.12, "score_factors": {"proj": 11.8, "vegas": 0.92}}


def _unbalanced(src: str, start: str, opener: str) -> str:
    sig = src.index(start)
    i = sig + len(start) - len(opener)
    d, j = 0, i
    while True:
        two = src[j : j + 2]
        if opener == "{{":
            if two == "{{":
                d += 1
                j += 2
                continue
            if two == "}}":
                d -= 1
                j += 2
                if d == 0:
                    break
                continue
            j += 1
        else:
            if src[j] == "{":
                d += 1
            elif src[j] == "}":
                d -= 1
                if d == 0:
                    j += 1
                    break
            j += 1
    return src[sig:j]


def _run(fn_src: str, driver: str):
    res = subprocess.run(
        ["node", "-e", fn_src + "\n" + driver],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def _wv_fn() -> str:
    from dashboard_services.pages.waivers_page import build_waivers_body

    body = build_waivers_body("sleeper", 2026, "league", {})
    return _unbalanced(body, "function wvVerdictReasons(a, b, wi) {", "{")


def _ss_fn() -> str:
    src = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    return _unbalanced(src, "function _ssVerdictReasons(win, lose) {", "{")


def _ss_players():
    # app.js reads the total off player.stats.start_sit.implied_total.
    return (
        {"stats": {"start_score_factors": A_WIN["score_factors"],
                   "start_sit": {"implied_total": A_WIN["implied_total"]}}},
        {"stats": {"start_score_factors": B_LOSE["score_factors"],
                   "start_sit": {"implied_total": B_LOSE["implied_total"]}}},
    )


def test_waivers_vegas_reason_uses_implied_totals():
    driver = (
        "const a = %s;\nconst b = %s;\n"
        "process.stdout.write(JSON.stringify(wvVerdictReasons(a, b, 0)));"
        % (json.dumps(A_WIN), json.dumps(B_LOSE))
    )
    reasons = _run(_wv_fn(), driver)
    assert "a higher team total" not in reasons
    assert any(r.startswith("higher projection") for r in reasons)


def test_waivers_vegas_reason_fires_when_total_genuinely_higher():
    win = {"implied_total": 26, "score_factors": {"proj": 12.3, "vegas": 1.02}}
    driver = (
        "const a = %s;\nconst b = %s;\n"
        "process.stdout.write(JSON.stringify(wvVerdictReasons(a, b, 0)));"
        % (json.dumps(win), json.dumps(B_LOSE))
    )
    reasons = _run(_wv_fn(), driver)
    assert "a higher team total" in reasons


def test_start_sit_vegas_reason_uses_implied_totals():
    w, l = _ss_players()
    driver = (
        "const w = %s;\nconst l = %s;\n"
        "process.stdout.write(JSON.stringify(_ssVerdictReasons(w, l)));"
        % (json.dumps(w), json.dumps(l))
    )
    reasons = _run(_ss_fn(), driver)
    assert "a higher team total" not in reasons
    assert any(r.startswith("higher projection") for r in reasons)


def test_start_sit_vegas_reason_fires_when_total_genuinely_higher():
    _, l = _ss_players()
    w = {"stats": {"start_score_factors": {"proj": 12.3, "vegas": 1.02},
                    "start_sit": {"implied_total": 26}}}
    driver = (
        "const w = %s;\nconst l = %s;\n"
        "process.stdout.write(JSON.stringify(_ssVerdictReasons(w, l)));"
        % (json.dumps(w), json.dumps(l))
    )
    reasons = _run(_ss_fn(), driver)
    assert "a higher team total" in reasons
