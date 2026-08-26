"""JS getPlayerValue and Python player_trade_value must return the same number.

Extracts SCORING_MULTS + getPlayerValue from static/app.js and runs them in Node
against the same case table as utils.trade_value.player_trade_value.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from utils.trade_value import player_trade_value

REPO = Path(__file__).resolve().parents[1]
APP_JS = REPO / "static" / "app.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available"
)


CASES = [
    {
        "league_type": "1qb",
        "league_size": 10,
        "scoring_format": "ppr",
        "scoring_type": "dynasty",
        "te_premium": 0.0,
        "player": {"position": "RB", "value": 8000, "sf_value": 7200},
    },
    {
        "league_type": "sf",
        "league_size": 10,
        "scoring_format": "ppr",
        "scoring_type": "dynasty",
        "te_premium": 0.0,
        "player": {"position": "QB", "value": 5000, "sf_value": 9100},
    },
    {
        "league_type": "1qb",
        "league_size": 12,
        "scoring_format": "half",
        "scoring_type": "dynasty",
        "te_premium": 0.0,
        "player": {"position": "WR", "value": 6000, "value_12": 6400, "sf_value": 5500},
    },
    {
        "league_type": "sf",
        "league_size": 12,
        "scoring_format": "std",
        "scoring_type": "dynasty",
        "te_premium": 0.0,
        "player": {
            "position": "RB",
            "value": 4000,
            "sf_value": 3800,
            "sf_value_12": 4100,
        },
    },
    {
        "league_type": "1qb",
        "league_size": 10,
        "scoring_format": "ppr",
        "scoring_type": "dynasty",
        "te_premium": 1.0,
        "player": {"position": "TE", "value": 5000, "sf_value": 4800},
    },
    {
        "league_type": "sf",
        "league_size": 10,
        "scoring_format": "ppr",
        "scoring_type": "redraft",
        "te_premium": 0.0,
        "player": {
            "position": "WR",
            "value": 3000,
            "redraft_value_1qb": 2200,
            "redraft_value_sf": 2500,
        },
    },
    {
        "league_type": "1qb",
        "league_size": 10,
        "scoring_format": "ppr",
        "scoring_type": "redraft",
        "te_premium": 0.0,
        "player": {
            "position": "RB",
            "value": 3000,
            "redraft_value_1qb": 1800,
            "redraft_value_sf": 1900,
        },
    },
]


def _extract_js_value_math() -> str:
    text = APP_JS.read_text(encoding="utf-8")
    start = text.index("const SCORING_MULTS = {")
    fn = text.index("function getPlayerValue(player) {", start)
    depth = 0
    end = fn
    for i, ch in enumerate(text[fn:], fn):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    block = text[start:end]
    assert "function getPlayerValue" in block
    return block


def _js_values(cases):
    driver = (
        "let CASE;\n"
        "function getLeagueType() { return CASE.league_type; }\n"
        "function getLeagueSize() { return CASE.league_size; }\n"
        "function getScoringFormat() { return CASE.scoring_format; }\n"
        "function getScoringType() { return CASE.scoring_type; }\n"
        "function getTePremium() { return CASE.te_premium; }\n"
        "%s\n"
        "const cases = %s;\n"
        "const out = cases.map(c => { CASE = c; return getPlayerValue(c.player); });\n"
        "process.stdout.write(JSON.stringify(out));\n"
        % (_extract_js_value_math(), json.dumps(cases))
    )
    res = subprocess.run(
        ["node", "-e", driver], capture_output=True, text=True, timeout=30
    )
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def test_player_trade_value_matches_js_get_player_value():
    js_vals = _js_values(CASES)
    assert len(js_vals) == len(CASES)
    for case, js_val in zip(CASES, js_vals):
        py_val = player_trade_value(
            case["player"],
            league_type=case["league_type"],
            league_size=case["league_size"],
            scoring_format=case["scoring_format"],
            scoring_type=case["scoring_type"],
            te_premium=case["te_premium"],
        )
        assert py_val == js_val, (
            f"drift for {case['league_type']}/{case['league_size']}/"
            f"{case['scoring_format']}/{case['scoring_type']}: py={py_val} js={js_val}"
        )


def test_te_premium_increases_tight_end_value():
    te = {"position": "TE", "value": 5000}
    base = player_trade_value(te, te_premium=0.0)
    bumped = player_trade_value(te, te_premium=1.0)
    assert bumped > base
    # 5000 * (1 + 1.0 * 0.20) = 6000, half-up to one decimal.
    assert bumped == 6000.0
