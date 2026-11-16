"""Behavioral harness for the ScoreZone sticky bar + section gutter fixes.

Kaedon's live screenshot (2026-11-15, SF 46 @ DAL 40): the Moments
section and the Box score box ran on a wider inset than the board card,
and the "sticky" bar rendered in flow under the Box score button --
visible while the hero field board was directly above it. Her direction:
the Moments section, the Box score box and the sticky bar all take the
board card's gutter; the sticky bar uses the Matchups drive bar
(mb-fld-*) for its field; and it only shows when the field board isn't
visible.

This harness boots the REAL static/scorezone.js + static/dashboard.css
(Playwright, iPhone viewport + touch) against the same scorezone-data
fixture shape as the Moments click harness, deep-linked to one live
game (?game=g1: BUF 17 @ KC 24, KC ball at the BUF 30, 2nd & 6), and
asserts:
  * the sticky bar's field is the Matchups drive bar markup with the
    right geometry (fill 56%, FD tick at 29.2%, ball at 34%);
  * the bar is hidden at rest (field board in view), appears pinned
    directly under the ScoreZone header once the field board has
    scrolled under it, and hides again scrolled back;
  * measured insets: Moments label/rail, Box score box (re-parented to
    the board's next sibling, the shared renderer's DOM shape) and the
    board card share left/right edges within 2px;
  * the sticky bar animates: a poll that moves the ball (BUF 30 ->
    BUF 18) glides the ball/fill through intermediate positions to the
    new spot (sampled every 30ms), a score change (KC 24 -> 31) puts
    the hero board's team-colour flash on the bar's home score only,
    and under prefers-reduced-motion the same update lands instantly
    with no intermediate samples.

Env: RZ_STATIC_DIR overrides where scorezone.js/dashboard.css are read
from (for before/after screenshots); RZ_SHOT_DIR is where the top and
scrolled screenshots are written.

Run: python3 tests/scorezone_sticky_spacing_harness.py  (exit 0 = pass)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(os.environ.get("RZ_STATIC_DIR", ROOT / "static"))
SCOREZONE_JS = (STATIC_DIR / "scorezone.js").read_text(encoding="utf-8")
DASHBOARD_CSS = (STATIC_DIR / "dashboard.css").read_text(encoding="utf-8")
SHOT_DIR = Path(os.environ.get("RZ_SHOT_DIR", "/tmp/rz-sticky-shots"))

KICKOFF = 1763322000
GID = "g1"
SCORING = {
    "pass_yd": 0.04, "pass_td": 4.0, "pass_int": -2.0,
    "rush_yd": 0.1, "rush_td": 6.0,
    "rec": 0.5, "rec_yd": 0.1, "rec_td": 6.0,
    "fgm": 3.0, "xpm": 1.0, "sack": 1.0, "int": 2.0, "fum_rec": 2.0,
    "def_td": 6.0,
}
ZERO = {"pass_yds": 0, "pass_td": 0, "int": 0, "carries": 0, "rush_yds": 0,
        "rush_td": 0, "rec": 0, "rec_yds": 0, "rec_td": 0, "targets": 0}

PLAYERS = {
    "1001": {"name": "Rush Man", "pos": "RB", "team": "KC"},
    "1002": {"name": "Catch Man", "pos": "WR", "team": "KC"},
    "1003": {"name": "Pass Man", "pos": "QB", "team": "KC"},
}

_seq = [0]


def play(pid, quarter, clock, line, text, is_td=False, play_id=None, clip=None):
    _seq[0] += 1
    p = {
        "play_id": play_id or f"p{_seq[0]}",
        "seq": _seq[0],
        "game_id": GID,
        "quarter": quarter,
        "clock": clock,
        "down": "2", "distance": "6", "yard_line": "BUF 30",
        "play_text": text,
        "pid": pid, "name": PLAYERS[pid]["name"], "team": "KC",
        "stat_line": line, "is_td": is_td,
    }
    if clip:
        p.update(clip)
    return p


def line(**kw):
    out = dict(ZERO)
    out.update(kw)
    return out


RAIL_CLIP = {
    "clip_tweet_id": "1790000000000000001",
    "clip_url": "https://x.com/NFL/status/1790000000000000001",
    "clip_video_url": "https://video.twimg.com/ext_tw_video/1790/pu/vid/avc1/rail.mp4",
    "clip_poster_url": "https://pbs.twimg.com/ext_tw_video_thumb/1790/pu/img/rail.jpg",
}
CTL_CLIP = {
    "clip_tweet_id": "1790000000000000002",
    "clip_url": "https://x.com/NFL/status/1790000000000000002",
    "clip_video_url": "https://video.twimg.com/ext_tw_video/1790/pu/vid/avc1/ctl.mp4",
    "clip_poster_url": "https://pbs.twimg.com/ext_tw_video_thumb/1790/pu/img/ctl.jpg",
}


def build_plays():
    plays = []
    plays.append(play("1001", "Q2", "5:00", line(carries=1, rush_yds=12, rush_td=1),
                      "Rush Man rushes for 12 yards for a touchdown",
                      is_td=True, play_id="rail-td", clip=RAIL_CLIP))
    for i, clk in enumerate(["4:30", "4:00", "3:30", "3:00", "2:30",
                             "2:00", "1:30", "1:00", "0:30", "0:05"]):
        pid = ["1002", "1003", "1001"][i % 3]
        if pid == "1002":
            plays.append(play(pid, "Q2", clk, line(targets=1), "Pass incomplete to Catch Man"))
        elif pid == "1003":
            plays.append(play(pid, "Q2", clk, line(pass_yds=7), "Pass Man pass for 7 yards"))
        else:
            plays.append(play(pid, "Q2", clk, line(carries=1, rush_yds=3), "Rush Man rushes for 3 yards"))
    plays.append(play("1002", "Q3", "2:00", line(rec=1, targets=1, rec_yds=18, rec_td=1),
                      "Pass Man pass to Catch Man for 18 yards for a touchdown",
                      is_td=True, play_id="ctl-td", clip=CTL_CLIP))
    for i in range(15):
        clk = f"1:{55 - 5 * i:02d}" if i < 12 else f"0:{55 - 5 * (i - 12):02d}"
        pid = ["1003", "1002", "1001"][i % 3]
        if pid == "1002":
            plays.append(play(pid, "Q3", clk, line(rec=1, targets=1, rec_yds=6), "Catch Man reception for 6 yards"))
        elif pid == "1003":
            plays.append(play(pid, "Q3", clk, line(pass_yds=9), "Pass Man pass for 9 yards"))
        else:
            plays.append(play(pid, "Q3", clk, line(carries=1, rush_yds=4), "Rush Man rushes for 4 yards"))
    return plays


def payload():
    player_info = {}
    for pid, p in PLAYERS.items():
        player_info[pid] = {
            "name": p["name"], "pos": p["pos"], "team": p["team"],
            "game_id": GID, "game_code": "1", "game_quarter": "Q3",
            "game_clock": "0:45", "game_status": "",
            "away": "BUF", "home": "KC", "away_pts": "17", "home_pts": "24",
            "game_time_epoch": KICKOFF,
            "stat_line": line(carries=8, rush_yds=40, rec=4, targets=6,
                              rec_yds=50, pass_yds=180),
        }
    return {
        "week": 11, "season": 2026, "platform": "sleeper", "league_id": "demo",
        "scope": "league",
        # A poll payload without viewer identity fields is discarded by
        # _refresh before _setState (refresh-no-identity), so refreshes
        # would rebuild from the boot snapshot forever. Production
        # payloads carry the field; the animation checks below need
        # their updated snapshots to actually apply.
        "viewer_roster_ids": [1],
        "player_info": player_info,
        "scoring": dict(SCORING),
        "pbp_by_game": {GID: build_plays()},
        "games": {GID: {
            "game_id": GID, "away": "BUF", "home": "KC",
            "away_pts": "17", "home_pts": "24",
            "game_status": "", "game_code": "1", "status": "live",
            "game_clock": "0:45", "game_quarter": "Q3",
            "game_time_epoch": KICKOFF,
            "possession": "KC", "down": "2", "distance": "6",
            "yard_line": "BUF 30", "field_position_reliable": True,
        }},
        "matchups": [
            {"roster_id": 1, "matchup_id": 1, "points": 24.0,
             "projected_pts": 30.0, "starters": ["1001", "1002", "1003"],
             "players": ["1001", "1002", "1003"],
             "players_points": {"1001": 7.2, "1002": 9.1, "1003": 3.4}},
            {"roster_id": 2, "matchup_id": 1, "points": 17.0,
             "projected_pts": 28.0, "starters": [], "players": [],
             "players_points": {}},
        ],
        "updated_at": 1763325000,
    }


def payload_with(**kw):
    """payload() with the game's live state moved: the games entry and
    the mirrored fields on every player_info row change together, the
    way a real poll snapshot would carry them. Situation fields (yard
    line / down / distance) are read from the newest PBP play rather
    than the scoreboard row, so the snapshot also appends the play
    that moved the ball, sequenced past every existing play."""
    data = payload()
    g = data["games"][GID]
    g.update(kw)
    for info in data["player_info"].values():
        info["game_clock"] = g["game_clock"]
        info["away_pts"] = g["away_pts"]
        info["home_pts"] = g["home_pts"]
    plays = data["pbp_by_game"][GID]
    seq = max(p["seq"] for p in plays) + 1
    plays.append({
        "play_id": f"p{seq}", "seq": seq, "game_id": GID,
        "quarter": g["game_quarter"], "clock": g["game_clock"],
        "down": str(g["down"]), "distance": str(g["distance"]),
        "yard_line": g["yard_line"],
        "play_text": "Pass Man pass for 12 yards",
        "pid": "1003", "name": "Pass Man", "team": "KC",
        "stat_line": line(pass_yds=12), "is_td": False,
    })
    return data


HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="/static/dashboard.css">
</head><body>
<nav class="top-nav" style="height:56px">BR</nav>
<div id="app-scale"><div class="page-shell" data-page="scorezone">
<div id="page-root"><div id="rz-root"></div></div>
</div></div>
<script>window.__rz__ = __PAYLOAD__;</script>
<script src="/static/scorezone.js"></script>
</body></html>"""

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000a49444154789c6300010000050001a5f6453b0000"
    "000049454e44ae426082")


def main():
    from playwright.sync_api import sync_playwright

    failures = []

    def check(name, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + name + (f" -- {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(name)

    fixture = payload()
    # The scorezone-data route serves whatever `current` holds, so the
    # test can move the drive/score between polls; an auto-poll that
    # lands between flips just re-serves the same snapshot (a no-op
    # re-render), never the next one early.
    current = {"body": json.dumps(fixture)}
    SHOT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        try:
            browser = pw.webkit.launch()
            device = dict(pw.devices["iPhone 14"])
            device.pop("default_browser_type", None)
        except Exception:
            browser = pw.chromium.launch()
            device = dict(pw.devices["iPhone 14"])
            device.pop("default_browser_type", None)
        ctx = browser.new_context(**device)
        page = ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))

        def route_handler(route):
            url = route.request.url
            if "scorezone-data" in url:
                route.fulfill(status=200, content_type="application/json",
                              body=current["body"])
            elif url.endswith("/static/scorezone.js"):
                route.fulfill(status=200, content_type="application/javascript",
                              body=SCOREZONE_JS)
            elif url.endswith("/static/dashboard.css"):
                route.fulfill(status=200, content_type="text/css", body=DASHBOARD_CSS)
            elif "sleepercdn.com" in url or "pbs.twimg.com" in url:
                route.fulfill(status=200, content_type="image/png", body=PNG)
            elif "video.twimg.com" in url:
                route.fulfill(status=200, content_type="video/mp4", body=b"")
            elif "rz.test" in url and "scorezone" in url:
                route.fulfill(status=200, content_type="text/html",
                              body=HTML.replace("__PAYLOAD__", json.dumps(fixture)))
            else:
                route.fulfill(status=200, content_type="text/plain", body="")

        page.route("**/*", route_handler)
        page.goto("http://rz.test/sleeper/2026/demo/scorezone?game=g1")
        page.wait_for_selector("#rz-nfl-board .rz-fb", timeout=15000)
        # Attached, not visible: hidden at rest is the correct state.
        page.wait_for_selector(".rz-sticky-bar", state="attached", timeout=15000)
        page.wait_for_timeout(400)

        def rect(sel):
            return page.evaluate(
                "(function(){"
                "var el = document.querySelector('" + sel + "');"
                "if (!el) return null;"
                "var r = el.getBoundingClientRect();"
                "return { left: r.left, right: r.right, top: r.top,"
                "  bottom: r.bottom, width: r.width, height: r.height };"
                "})()")

        def close(a, b, tol=2):
            return a is not None and b is not None and abs(a - b) <= tol

        board = rect("#rz-nfl-board")
        check("board + field board render for the filtered live game", board is not None)

        # --- C1: the sticky bar's field is the Matchups drive bar. -----
        drive = page.evaluate(
            "(function(){"
            "var bar = document.querySelector('.rz-sticky-bar .mb-fld-bar');"
            "if (!bar) return null;"
            "var fill = bar.querySelector('.mb-fld-track > .mb-fld-fill');"
            "var fd = bar.querySelector('.mb-fld-track > .mb-fld-fd');"
            "var ball = bar.querySelector('.mb-fld-track > .mb-fld-ball');"
            "return { tc: bar.getAttribute('style') || '',"
            "  fillCls: fill ? fill.className : '',"
            "  fillW: fill ? fill.style.width : '',"
            "  fdLeft: fd ? fd.style.left : null,"
            "  ballLeft: ball ? ball.style.left : '' };"
            "})()")
        check("sticky bar field uses Matchups mb-fld markup", drive is not None)
        if drive:
            # KC (home) ball at the BUF 30: spot 30 -> ball x = 34%;
            # fill runs from KC's goal line: (100 - 30) * 0.8 = 56%;
            # 2nd & 6 -> line to gain at 24 -> x = 29.2%.
            check("drive bar geometry (fill/FD/ball)",
                  drive["fillCls"].endswith("mb-home") and drive["fillW"] == "56%"
                  and drive["fdLeft"] == "29.2%" and drive["ballLeft"] == "34%"
                  and "--mb-tc:" in drive["tc"],
                  json.dumps(drive))
        check("old custom mini field is gone",
              page.query_selector(".rz-sticky-mini") is None)

        # --- C2: hidden while the field board is in view. --------------
        state = page.evaluate(
            "(function(){"
            "var b = document.querySelector('.rz-sticky-bar');"
            "var cs = getComputedStyle(b);"
            "return { on: b.classList.contains('is-on'), vis: cs.visibility,"
            "  pos: cs.position };"
            "})()")
        check("sticky bar is fixed, hidden at rest (board in view)",
              state["pos"] == "fixed" and not state["on"] and state["vis"] == "hidden",
              json.dumps(state))

        # --- A/B: one gutter for Moments, Box score and the board. -----
        # The harness boots scorezone.js alone, so the fallback renderer
        # nests the Box score button inside the board; the production
        # shared renderer emits it as the board's next sibling. Move it
        # out to reproduce the production DOM shape before measuring.
        page.evaluate(
            "var board = document.querySelector('#rz-nfl-board');"
            "var btn = board.querySelector('.rz-boxscore-btn');"
            "if (btn) board.after(btn);")
        page.wait_for_timeout(150)
        board = rect("#rz-nfl-board")
        wrap = rect(".rz-moments-rail-wrap")
        label = rect(".rz-moments-rail-label")
        btn = rect(".rz-boxscore-btn")
        check("Moments wrap edges match the board card",
              wrap and close(wrap["left"], board["left"]) and close(wrap["right"], board["right"]),
              f"wrap={wrap} board={board}")
        check("Moments label edges match the board card",
              label and close(label["left"], board["left"]) and close(label["right"], board["right"]),
              f"label={label} board={board}")
        check("Box score box edges match the board card",
              btn and close(btn["left"], board["left"]) and close(btn["right"], board["right"]),
              f"btn={btn} board={board}")

        page.screenshot(path=str(SHOT_DIR / "top.png"))

        # --- C2: scroll the field board out -> bar pins under header. --
        # The app shell scrolls on <body> (html/body are height:100% with
        # body overflow-y:auto), so scroll the body, not the window.
        page.evaluate(
            "var navH = parseFloat(getComputedStyle(document.documentElement)"
            "  .getPropertyValue('--rz-nav-h')) || 0;"
            "var headH = document.querySelector('.rz-header')"
            "  .getBoundingClientRect().height;"
            "var fr = document.querySelector('#rz-nfl-board .rz-fb')"
            "  .getBoundingClientRect();"
            "document.body.scrollTop = document.body.scrollTop"
            "  + fr.bottom - (navH + headH) + 80;")
        try:
            page.wait_for_function(
                "document.querySelector('.rz-sticky-bar')"
                ".classList.contains('is-on')", timeout=5000)
            shown = True
        except Exception:
            shown = False
        check("sticky bar shows once the field board is scrolled away", shown)
        page.wait_for_timeout(350)  # fade/slide settles
        bar = rect(".rz-sticky-bar")
        header = rect(".rz-header")
        # Pinned directly under the header: never overlapping it, with
        # the designed 6px gap (a couple px of slack for a header whose
        # wrapped height changed after the last observer bind).
        gap = (bar["top"] - header["bottom"]) if bar and header else None
        check("sticky bar pins directly under the ScoreZone header",
              gap is not None and -1 <= gap <= 9,
              f"gap={gap} bar={bar} header={header}")
        check("sticky bar edges match the board card",
              bar and close(bar["left"], board["left"]) and close(bar["right"], board["right"]),
              f"bar={bar} board={board}")
        vis = page.evaluate(
            "getComputedStyle(document.querySelector('.rz-sticky-bar')).visibility")
        check("sticky bar is visible while shown", vis == "visible", vis)
        page.screenshot(path=str(SHOT_DIR / "scrolled.png"))

        # --- C2: scroll back -> bar hides again. ------------------------
        page.evaluate("document.body.scrollTop = 0")
        try:
            page.wait_for_function(
                "!document.querySelector('.rz-sticky-bar')"
                ".classList.contains('is-on')", timeout=5000)
            hidden = True
        except Exception:
            hidden = False
        check("sticky bar hides again when the board returns", hidden)

        # --- Animation: drive updates glide, score changes flash. -----
        # Scroll the field board away again so the bar is on screen and
        # sample the ball/fill positions (as % of the track) every 30ms
        # across a poll that moves the ball BUF 30 -> BUF 18. A jump
        # would step 34% -> 24.4% (ball) / 56% -> 65.6% (fill) between
        # two samples; a glide passes through the values in between.
        scroll_away = (
            "var navH = parseFloat(getComputedStyle(document.documentElement)"
            "  .getPropertyValue('--rz-nav-h')) || 0;"
            "var headH = document.querySelector('.rz-header')"
            "  .getBoundingClientRect().height;"
            "var fr = document.querySelector('#rz-nfl-board .rz-fb')"
            "  .getBoundingClientRect();"
            "document.body.scrollTop = document.body.scrollTop"
            "  + fr.bottom - (navH + headH) + 80;")
        sampler = (
            "window.__glide = [];"
            "window.__glideTimer = setInterval(function(){"
            "var track = document.querySelector('.rz-sticky-bar .mb-fld-track');"
            "var ball = document.querySelector('.rz-sticky-bar .mb-fld-ball');"
            "var fill = document.querySelector('.rz-sticky-bar .mb-fld-fill');"
            "if (!track || !ball || !fill) return;"
            "var tr = track.getBoundingClientRect();"
            "if (!tr.width) return;"
            "var br = ball.getBoundingClientRect();"
            "var fr2 = fill.getBoundingClientRect();"
            "window.__glide.push({"
            "  ball: (br.left + br.width / 2 - tr.left) / tr.width * 100,"
            "  fill: fr2.width / tr.width * 100 });"
            "}, 30);")
        page.evaluate(scroll_away)
        page.wait_for_function(
            "document.querySelector('.rz-sticky-bar')"
            ".classList.contains('is-on')", timeout=5000)
        page.wait_for_timeout(450)  # entry slide settles
        page.evaluate(sampler)
        page.wait_for_timeout(250)  # baseline samples at the old spot
        current["body"] = json.dumps(payload_with(
            yard_line="BUF 18", game_clock="0:38"))
        page.tap("#rz-timer")
        page.wait_for_timeout(1150)
        samples = page.evaluate(
            "clearInterval(window.__glideTimer); window.__glide")
        ball_vals = [s["ball"] for s in samples]
        fill_vals = [s["fill"] for s in samples]
        check("glide: ball passes through intermediate positions",
              any(25.5 < v < 33.0 for v in ball_vals),
              f"ball samples={[round(v, 1) for v in ball_vals]}")
        check("glide: fill passes through intermediate widths",
              any(57.5 < v < 64.0 for v in fill_vals),
              f"fill samples={[round(v, 1) for v in fill_vals]}")
        check("glide: ball lands on the new spot (24.4%)",
              bool(ball_vals) and abs(ball_vals[-1] - 24.4) <= 1.5,
              f"last ball={ball_vals[-1] if ball_vals else None}")
        check("glide: fill lands on the new width (65.6%)",
              bool(fill_vals) and abs(fill_vals[-1] - 65.6) <= 1.5,
              f"last fill={fill_vals[-1] if fill_vals else None}")
        page.screenshot(path=str(SHOT_DIR / "updated.png"))

        # Score change while the bar is up: KC TD, 24 -> 31. The home
        # score in the bar takes the hero board's team-colour flash;
        # the unchanged away score stays plain.
        current["body"] = json.dumps(payload_with(
            yard_line="BUF 18", game_clock="0:31", home_pts="31"))
        page.tap("#rz-timer")
        page.wait_for_timeout(400)
        flash = page.evaluate(
            "(function(){"
            "var bs = document.querySelectorAll('.rz-sticky-bar .rz-sticky-team b');"
            "return { away: bs[0] ? bs[0].className : '',"
            "  home: bs[1] ? bs[1].className : '',"
            "  homeVar: bs[1] ? bs[1].style.getPropertyValue('--rz-flash') : '' };"
            "})()")
        check("score change flashes the scoring team's bar score",
              "rz-score-flash-team" in flash["home"] and bool(flash["homeVar"]),
              json.dumps(flash))
        check("unchanged score does not flash",
              "rz-score-flash-team" not in flash["away"],
              json.dumps(flash))

        check("no page errors", errors == [], "; ".join(errors[:3]))
        browser.close()

        # --- Reduced motion: the same update lands instantly. ---------
        # Chromium: it reliably emulates prefers-reduced-motion.
        current_rm = {"body": json.dumps(payload())}
        browser2 = pw.chromium.launch()
        ctx2 = browser2.new_context(reduced_motion="reduce", **device)
        page2 = ctx2.new_page()
        errors2 = []
        page2.on("pageerror", lambda e: errors2.append(str(e)))

        def route_handler_rm(route):
            url = route.request.url
            if "scorezone-data" in url:
                route.fulfill(status=200, content_type="application/json",
                              body=current_rm["body"])
            elif url.endswith("/static/scorezone.js"):
                route.fulfill(status=200, content_type="application/javascript",
                              body=SCOREZONE_JS)
            elif url.endswith("/static/dashboard.css"):
                route.fulfill(status=200, content_type="text/css", body=DASHBOARD_CSS)
            elif "sleepercdn.com" in url or "pbs.twimg.com" in url:
                route.fulfill(status=200, content_type="image/png", body=PNG)
            elif "video.twimg.com" in url:
                route.fulfill(status=200, content_type="video/mp4", body=b"")
            elif "rz.test" in url and "scorezone" in url:
                route.fulfill(status=200, content_type="text/html",
                              body=HTML.replace("__PAYLOAD__", json.dumps(fixture)))
            else:
                route.fulfill(status=200, content_type="text/plain", body="")

        page2.route("**/*", route_handler_rm)
        page2.goto("http://rz.test/sleeper/2026/demo/scorezone?game=g1")
        page2.wait_for_selector("#rz-nfl-board .rz-fb", timeout=15000)
        page2.wait_for_selector(".rz-sticky-bar", state="attached", timeout=15000)
        page2.wait_for_timeout(300)
        page2.evaluate(scroll_away)
        page2.wait_for_function(
            "document.querySelector('.rz-sticky-bar')"
            ".classList.contains('is-on')", timeout=5000)
        page2.wait_for_timeout(200)
        page2.evaluate(sampler)
        page2.wait_for_timeout(250)
        current_rm["body"] = json.dumps(payload_with(
            yard_line="BUF 18", game_clock="0:38"))
        page2.tap("#rz-timer")
        page2.wait_for_timeout(900)
        samples2 = page2.evaluate(
            "clearInterval(window.__glideTimer); window.__glide")
        ball2 = [s["ball"] for s in samples2]
        check("reduced motion: no intermediate ball positions",
              bool(ball2) and all(v >= 32.5 or v <= 25.9 for v in ball2),
              f"ball samples={[round(v, 1) for v in ball2]}")
        check("reduced motion: ball lands on the new spot instantly",
              bool(ball2) and abs(ball2[-1] - 24.4) <= 1.5,
              f"last ball={ball2[-1] if ball2 else None}")
        check("reduced motion: no page errors", errors2 == [],
              "; ".join(errors2[:3]))
        browser2.close()

    print("\n%d failure(s)" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
