"""Focused tests for dashboard_services/portfolio_insights.py: Bye Watch
coverage verdicts and Rooting Guide grouping / need lines (v25 redesign)."""
import pytest

from dashboard_services import portfolio_insights as pi


def _et_epoch(year, month, day, hour, minute=0):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime(year, month, day, hour, minute,
                    tzinfo=ZoneInfo("America/New_York")).timestamp()


OCT25_1PM_ET = _et_epoch(2026, 10, 25, 13)      # Sunday 1 PM ET
OCT25_425PM_ET = _et_epoch(2026, 10, 25, 16, 25)  # Sunday 4:25 PM ET
OCT26_815PM_ET = _et_epoch(2026, 10, 26, 20, 15)  # Monday 8:15 PM ET


# ── Bye Watch ────────────────────────────────────────────────────────────────

def _league(**kw):
    base = {"name": "Alpha", "platform": "sleeper", "league_id": "1",
            "out": [], "bench": [], "top_fa": {}}
    base.update(kw)
    return base


def test_bye_full_strength():
    row = pi.bye_row(_league())
    assert row["verdict"] == "Full strength"
    assert row["out_count"] == 0


def test_bye_covered_by_startable_bench():
    row = pi.bye_row(_league(
        out=[{"pid": "r1", "name": "Out RB", "pos": "RB", "slot": "RB"}],
        bench=[{"pid": "r2", "name": "Bench RB", "pos": "RB", "value": 300}],
    ))
    assert row["verdict"] == "Covered"
    assert "Bench RB" in row["cover_note"]


def test_bye_stash_bench_is_not_coverage():
    row = pi.bye_row(_league(
        out=[{"pid": "r1", "name": "Out RB", "pos": "RB", "slot": "RB"}],
        bench=[{"pid": "r2", "name": "Deep Stash", "pos": "RB", "value": 5}],
        top_fa={"RB": {"pid": "fa1", "name": "Top FA RB", "value": 400}},
    ))
    assert row["verdict"] == "Hole · RB"
    assert row["hole_pos"] == "RB"
    assert "Top FA RB" in row["fix"]["text"]
    assert row["fix"]["move"]["pid"] == "fa1"


def test_bye_flex_slot_accepts_flex_positions():
    row = pi.bye_row(_league(
        out=[{"pid": "w1", "name": "Out WR", "pos": "WR", "slot": "FLEX"}],
        bench=[{"pid": "r9", "name": "Bench RB", "pos": "RB", "value": 200}],
    ))
    assert row["verdict"] == "Covered"


def test_bye_wrong_position_does_not_cover():
    row = pi.bye_row(_league(
        out=[{"pid": "t1", "name": "Out TE", "pos": "TE", "slot": "TE"}],
        bench=[{"pid": "w9", "name": "Bench WR", "pos": "WR", "value": 900}],
    ))
    assert row["verdict"] == "Hole · TE"
    assert row["fix"]["move"] is None  # no FA known: honest fix, no promotion


def test_bye_one_bench_player_covers_once():
    row = pi.bye_row(_league(
        out=[{"pid": "r1", "name": "RB A", "pos": "RB", "slot": "RB"},
             {"pid": "r2", "name": "RB B", "pos": "RB", "slot": "RB"}],
        bench=[{"pid": "r3", "name": "Only Cover", "pos": "RB", "value": 200}],
    ))
    assert row["verdict"] == "Hole · RB"


# ── Rooting Guide ────────────────────────────────────────────────────────────

def _game(away, home, epoch, status="pre", **kw):
    g = {"away": away, "home": home, "epoch": epoch, "status": status,
         "away_pts": kw.get("away_pts"), "home_pts": kw.get("home_pts"),
         "game_quarter": kw.get("q", ""), "game_clock": kw.get("clock", ""),
         "game_status": ""}
    return g


def _starter(pid, name, pos, team, league_key, **kw):
    s = {"pid": pid, "name": name, "pos": pos, "team": team,
         "league_key": league_key, "league_name": league_key,
         "points": kw.get("points"), "proj": kw.get("proj")}
    return s


def test_windows_bucket_and_sort():
    assert pi.window_for(OCT25_1PM_ET)["label"] == "Sun 1 PM"
    assert pi.window_for(OCT25_425PM_ET)["label"] == "Sun 4 PM"
    assert pi.window_for(OCT26_815PM_ET)["label"] == "Mon night"


def test_group_rooting_windows_stakes_and_sides():
    games = [
        _game("CLE", "BAL", OCT25_1PM_ET, "in", away_pts=17, home_pts=24,
              q="3", clock="2:14"),
        _game("DAL", "PHI", OCT26_815PM_ET),
        _game("NYJ", "BUF", OCT25_1PM_ET),  # no user starters: dropped
    ]
    starters = [
        _starter("1", "RB One", "RB", "CLE", "sleeper|1", points=14.2),
        _starter("2", "TE One", "TE", "BAL", "sleeper|1", points=9.6),
        _starter("3", "RB Two", "RB", "DAL", "sleeper|2"),
    ]
    windows = pi.group_rooting(games, starters, {"CLE": "2-5", "BAL": "4-3"})
    assert [w["label"] for w in windows] == ["Sun 1 PM", "Mon night"]
    card = windows[0]["games"][0]
    assert card["status"] == "in"
    assert card["status_line"] == "Q3 · 2:14"
    assert card["away_rec"] == "2-5"
    assert [s["name"] for s in card["sides"]["away"]] == ["RB One"]
    assert [s["name"] for s in card["sides"]["home"]] == ["TE One"]
    assert windows[0]["stakes"] == "2 starters across 1 of your leagues."
    assert windows[1]["games"][0]["status_line"].startswith("Mon ·")


def test_game_status_line_variants():
    assert pi.game_status_line({"status": "final"}) == "Final"
    assert pi.game_status_line({"status": "in", "game_quarter": "Q4",
                                "game_clock": "0:32"}) == "Q4 · 0:32"
    line = pi.game_status_line({"status": "pre", "epoch": OCT25_1PM_ET})
    assert line == "Sun · 1:00 PM"


def test_need_lines_flip_and_out_of_reach():
    margins = {
        "sleeper|2": {"league_name": "BLITZ", "my_score": 92.1,
                      "opp_score": 105.3, "final": False,
                      "remaining_names": ["RB Two", "TE Two"]},
        "sleeper|3": {"league_name": "Greatness", "my_score": 70.0,
                      "opp_score": 105.0, "final": False,
                      "remaining_names": []},
        "sleeper|4": {"league_name": "Done League", "my_score": 10.0,
                      "opp_score": 99.0, "final": True, "remaining_names": []},
        "sleeper|5": {"league_name": "Leading", "my_score": 120.0,
                      "opp_score": 100.0, "final": False,
                      "remaining_names": ["WR Left"]},
    }
    lines = pi.need_lines(margins)
    assert lines[0] == ("BLITZ needs 13.3 combined from RB Two and TE Two "
                        "to flip their game.")
    assert lines[1] == ("Greatness has nobody left in this window and is "
                        "out of reach.")
    assert len(lines) == 2


# ── Rooting / Bye cleanup (2026-10-27 screenshots) ──────────────────────────

def test_bye_full_strength_does_not_repeat_itself():
    # The screenshot row said the same fact three times: muted text, a
    # "Full strength" pill and a "Nobody on bye." second line. The builder
    # now leaves cover_note empty for full-strength rows and the renderer
    # (pinned below) draws them as one line with no pill.
    row = pi.bye_row(_league())
    assert row["verdict"] == "Full strength"
    assert row["cover_note"] == ""


def test_rooting_bye_cleanup_client_and_css():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    js = (root / "static" / "app.js").read_text(encoding="utf-8")
    # Pregame: no bold 0.0 points; the projection renders muted instead.
    assert "function rgSide(starters, status)" in js
    assert "if (status === 'pre')" in js
    assert "pf-rg-pts proj" in js
    assert "rgSide(g.sides.away, g.status)" in js
    assert "rgSide(g.sides.home, g.status)" in js
    # Away and home starters sit in two columns, each headed by its team,
    # and the header uses her custom brTeamBadge instead of the old
    # colored-circle pf-rg-badge (kept only as a fallback string).
    assert "pf-rg-cols" in js
    assert "pf-rg-colhead" in js
    assert "window.brTeamBadge(team, 'xs')" in js
    # Dedupe: one row per player per side, every league chip together,
    # and a grouped value is a single number or a low-high range built
    # by rgValue (never an average).
    assert "function rgGroup(starters)" in js
    assert "'pid:' + s.pid" in js
    assert "pf-rg-chips" in js
    assert "function rgValue(entries, field)" in js
    # Bye Watch: attention first, full-strength rows are a single line.
    bye = js[js.index("function renderBye(bye)"):js.index("function loadInsights")]
    assert "rank(a) - rank(b)" in bye
    assert "pf-bye-dot" in bye
    assert "pf-bye-full" in bye
    # Compact type scale lives in the app.py pf-ins CSS block, and the
    # position badge shrink is scoped to rooting rows only.
    app_src = (root / "app.py").read_text(encoding="utf-8")
    assert ".pf-rg-badge{width:24px;height:24px" in app_src
    assert ".pf-rg-winname{font-size:12px" in app_src
    assert ".pf-rg-team{font-size:12px" in app_src
    assert ".pf-rg-p .pos-badge{width:29px;height:19px;font-size:10px" in app_src
    assert ".pf-bye-row{display:flex;align-items:center;gap:8px;padding:5px 0" not in app_src
    # Bye rows are a consistent 3-column grid (name / content / status):
    # the old flex row let the out text float at a different x per row
    # and pinned full-strength text to the far edge with margin-left.
    assert ".pf-bye-row{display:grid;grid-template-columns:minmax(170px,220px) minmax(0,1fr) auto" in app_src
    assert ".pf-bye-full{grid-column:3;justify-self:end" in app_src
    assert ".pf-bye-fix{grid-column:2;" in app_src


def test_rooting_games_two_per_row_and_xs_badge_fits_text():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    js = (root / "static" / "app.js").read_text(encoding="utf-8")
    # Game cards render inside a grid wrapper per window; the day-group
    # header and the need lines stay full width outside it.
    assert "'<div class=\"pf-rg-games\">'" in js
    app_src = (root / "app.py").read_text(encoding="utf-8")
    # Two game cards per row once the CARD is wide enough (container
    # query on the card itself, not the viewport), one column below
    # that, which is also the phone layout.
    assert "#pfSecRooting{container-type:inline-size;}" in app_src
    assert ".pf-rg-games{display:grid;grid-template-columns:1fr;gap:6px;}" in app_src
    assert ".pf-rg-games .pf-rg-game{margin-bottom:0;}" in app_src
    assert "@container (min-width:720px){.pf-rg-games{grid-template-columns:repeat(2,1fr);}}" in app_src
    # The xs team badge sizes to its text: in the old fixed 24px box a
    # 3-letter abbreviation (~25px at 11px/800) slid under the 3px left
    # stripe and overflow:hidden clipped its first letter.
    css = (root / "static" / "dashboard.css").read_text(encoding="utf-8")
    badge = css[css.index(".team-badge-xs {"):css.index(".team-badge-xs::before")]
    assert "width: auto;" in badge
    assert "min-width: 24px;" in badge
    assert "padding: 0 5px 0 8px;" in badge


def _extract_js_fn(src, name):
    start = src.index(f"function {name}(")
    depth = 0
    for i in range(src.index("{", start), len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
    raise AssertionError(f"unterminated function {name}")


def test_rg_value_collapses_equal_displayed_ends():
    # The grouped-row value compares values AS DISPLAYED (fmt1's
    # 1-decimal rounding): raw values that differ but round to the same
    # displayed number render as that single number, never a stuttering
    # "9.5-9.5" range; only genuinely different displayed values range.
    import json
    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    root = Path(__file__).resolve().parents[1]
    js = (root / "static" / "app.js").read_text(encoding="utf-8")
    fns = "\n".join([
        _extract_js_fn(js, "fmt1"),
        _extract_js_fn(js, "rgValue"),
    ])
    driver = fns + """
const E = (...vals) => vals.map(v => ({ proj: v, points: v }));
console.log(JSON.stringify({
  equalRaw: rgValue(E(19.3, 19.3), 'proj'),
  displayEqual: rgValue(E(9.54, 9.46), 'proj'),
  roundingEdge: rgValue(E(10.21, 10.18), 'proj'),
  threeWayDisplayEqual: rgValue(E(14.82, 14.79, 14.81), 'proj'),
  realRange: rgValue(E(6.6, 10.5), 'proj'),
  realRangeReversed: rgValue(E(10.5, 6.6), 'proj'),
  single: rgValue(E(22.3), 'proj'),
  pointsField: rgValue([{points: 14.82}, {points: 14.79}], 'points'),
  pointsRange: rgValue([{points: 11.9}, {points: 16.0}], 'points'),
  nullsSkipped: rgValue([{proj: null}, {proj: 12.8}], 'proj'),
  allNull: rgValue([{proj: null}, {}], 'proj'),
}));
"""
    res = subprocess.run([node, "-e", driver], capture_output=True, text=True,
                         timeout=30)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])

    assert out["equalRaw"] == "19.3"
    assert out["displayEqual"] == "9.5"
    assert out["roundingEdge"] == "10.2"
    assert out["threeWayDisplayEqual"] == "14.8"
    assert out["realRange"] == "6.6-10.5"
    assert out["realRangeReversed"] == "6.6-10.5"
    assert out["single"] == "22.3"
    assert out["pointsField"] == "14.8"
    assert out["pointsRange"] == "11.9-16.0"
    assert out["nullsSkipped"] == "12.8"
    assert out["allNull"] == ""


def test_rooting_row_value_rail_structure():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    js = (root / "static" / "app.js").read_text(encoding="utf-8")
    # Name and league chips live in one flex block (.pf-rg-id) so the
    # chips wrap as a group inside it; the value span stays a direct,
    # fixed rail child of the row and can never be pushed off its line.
    assert "'<span class=\"pf-rg-id\"><span class=\"pf-rg-pname\">'" in js
    assert "'<span class=\"pf-rg-chips\">' + chips + '</span></span>'" in js
    app_src = (root / "app.py").read_text(encoding="utf-8")
    assert (".pf-rg-id{flex:1 1 auto;min-width:0;display:flex;"
            "align-items:center;flex-wrap:wrap;") in app_src
    # Names are never truncated: no ellipsis machinery on the name, it
    # takes its natural width and wraps inside the block if it must.
    assert (".pf-rg-pname{font-weight:600;flex:0 1 auto;min-width:0;"
            "overflow-wrap:anywhere;}") in app_src
    pname_rule = app_src[app_src.index(".pf-rg-pname{"):app_src.index(".pf-rg-p .pos-badge")]
    assert "text-overflow:ellipsis" not in pname_rule
    assert "white-space:nowrap" not in pname_rule
    # Value rail: flex-none, never wraps, right-aligned on the name line.
    assert (".pf-rg-pts{flex:0 0 auto;margin-left:auto;white-space:nowrap;"
            "text-align:right;") in app_src
    # Rows top-align so a wrapped chip group never drags the value off
    # the name line.
    assert ".pf-rg-p{display:flex;align-items:flex-start;" in app_src
