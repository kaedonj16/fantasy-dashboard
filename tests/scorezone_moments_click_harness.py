"""Behavioral harness for the ScoreZone Moments rail Watch buttons.

Kaedon, live on her phone (2026-11-15): "I can't click anything on the
moments bar." The rail's Watch handler (scorezone.js) did not open a clip
itself -- it looked up the play's FEED card toggle
([data-clip-toggle="<playId>"]) and clicked that. But the feed is
paginated (20 plays per page, only the current page exists in the DOM)
while the rail is built from the whole feed, so any moment whose feed
card is not on the current page had no toggle to click: the tap landed,
the handler ran, the lookup returned null, and nothing happened.

This harness boots the REAL static/scorezone.js + static/dashboard.css
in Chromium/WebKit (Playwright, iPhone viewport + touch) against a
fixture payload shaped exactly like the scorezone-data API response:
  * a clipped rushing TD early in Q2 (the "rail TD"), buried under 26
    newer plays so its feed card is NOT on feed page 0;
  * a clipped receiving TD late in Q3 (the "control TD") that IS on
    feed page 0.
It taps the rail Watch buttons with real touch taps and asserts a clip
panel actually opens -- red before the fix, green after. It also proves
the taps themselves land on the buttons (elementFromPoint + a click
counter), so a failure here is the handler, never hit interception.

Run: python3 tests/scorezone_moments_click_harness.py  (exit 0 = pass)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOREZONE_JS = (ROOT / "static" / "scorezone.js").read_text(encoding="utf-8")
DASHBOARD_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")

KICKOFF = 1763322000  # arbitrary; only relative play order matters
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

RAIL_PLAY_ID = f"{GID}:rail-td"
CTL_PLAY_ID = f"{GID}:ctl-td"


def build_plays(extra_late=False):
    plays = []
    # The rail TD: Q2 5:00 rushing TD with a matched clip.
    plays.append(play("1001", "Q2", "5:00", line(carries=1, rush_yds=12, rush_td=1),
                      "Rush Man rushes for 12 yards for a touchdown",
                      is_td=True, play_id="rail-td", clip=RAIL_CLIP))
    # 10 ordinary Q2 plays after it (all newer than the rail TD).
    for i, clk in enumerate(["4:30", "4:00", "3:30", "3:00", "2:30",
                             "2:00", "1:30", "1:00", "0:30", "0:05"]):
        pid = ["1002", "1003", "1001"][i % 3]
        if pid == "1002":
            plays.append(play(pid, "Q2", clk, line(targets=1), "Pass incomplete to Catch Man"))
        elif pid == "1003":
            plays.append(play(pid, "Q2", clk, line(pass_yds=7), "Pass Man pass for 7 yards"))
        else:
            plays.append(play(pid, "Q2", clk, line(carries=1, rush_yds=3), "Rush Man rushes for 3 yards"))
    # The control TD: Q3 2:00 receiving TD with a matched clip.
    plays.append(play("1002", "Q3", "2:00", line(rec=1, targets=1, rec_yds=18, rec_td=1),
                      "Pass Man pass to Catch Man for 18 yards for a touchdown",
                      is_td=True, play_id="ctl-td", clip=CTL_CLIP))
    # 15 ordinary Q3 plays after it (newer than BOTH TDs). Feed page 0 is
    # the newest 20 events: control TD is 16th (on page), rail TD is 27th
    # (not on page -- the bug case).
    for i in range(15):
        clk = f"1:{55 - 5 * i:02d}" if i < 12 else f"0:{55 - 5 * (i - 12):02d}"
        pid = ["1003", "1002", "1001"][i % 3]
        if pid == "1002":
            plays.append(play(pid, "Q3", clk, line(rec=1, targets=1, rec_yds=6), "Catch Man reception for 6 yards"))
        elif pid == "1003":
            plays.append(play(pid, "Q3", clk, line(pass_yds=9), "Pass Man pass for 9 yards"))
        else:
            plays.append(play(pid, "Q3", clk, line(carries=1, rush_yds=4), "Rush Man rushes for 4 yards"))
    if extra_late:
        plays.append(play("1001", "Q3", "0:30", line(carries=1, rush_yds=2),
                          "Rush Man rushes for 2 yards"))
    return plays


def payload(extra_late=False):
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
        "player_info": player_info,
        "scoring": dict(SCORING),
        "pbp_by_game": {GID: build_plays(extra_late)},
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


HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="/static/dashboard.css">
</head><body>
<div id="page-root"><div id="rz-root"></div></div>
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

    fixture_a = payload(False)
    fixture_b = payload(True)
    state = {"fetches": 0}

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
                state["fetches"] += 1
                body = fixture_a if state["fetches"] <= 1 else fixture_b
                route.fulfill(status=200, content_type="application/json",
                              body=json.dumps(body))
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
                              body=HTML.replace("__PAYLOAD__", json.dumps(fixture_a)))
            else:
                route.fulfill(status=200, content_type="text/plain", body="")

        page.route("**/*", route_handler)
        # Count real clicks landing on rail Watch buttons (capture phase,
        # independent of the module's own handlers): proves the tap is
        # delivered and any failure is downstream of the button.
        page.add_init_script(
            "window.__railClicks = 0;"
            "document.addEventListener('click', function(e){"
            "  if (e.target && e.target.closest &&"
            "      e.target.closest('[data-rail-clip],[data-clip-toggle]'))"
            "    window.__railClicks++;"
            "}, true);")
        page.goto("http://rz.test/sleeper/2026/demo/scorezone")
        page.wait_for_selector(".rz-moments-rail", timeout=15000)

        rail_btns = page.query_selector_all("[data-rail-clip]")
        check("rail renders both clipped TD moments", len(rail_btns) == 2,
              f"found {len(rail_btns)}")
        feed_toggle_rail = page.query_selector(f'[data-clip-toggle="{RAIL_PLAY_ID}"]')
        feed_toggle_ctl = page.query_selector(f'[data-clip-toggle="{CTL_PLAY_ID}"]')
        check("fixture: rail TD feed card is NOT on feed page 0", feed_toggle_rail is None)
        check("fixture: control TD feed card IS on feed page 0", feed_toggle_ctl is not None)

        def open_panels():
            return page.evaluate(
                "Array.from(document.querySelectorAll('.rz-clip-panel.is-open'))"
                ".map(function(p){ return { key: p.getAttribute('data-clip-panel'),"
                "  video: !!p.querySelector('.rz-clip-video'),"
                "  html: p.innerHTML.length }; })")

        def tap_rail(play_id):
            page.tap(f'[data-rail-clip="{play_id}"]')
            page.wait_for_timeout(450)

        # Hit delivery: the topmost element at the rail Watch button's
        # centre is the button itself (no overlay eats the tap).
        hit = page.evaluate(
            "var b = document.querySelector('[data-rail-clip=\"" + RAIL_PLAY_ID + "\"]');"
            "var r = b.getBoundingClientRect();"
            "var el = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);"
            "el === b || b.contains(el)")
        check("tap point on rail Watch is the button itself (no interception)", hit is True)

        # Control: the page-0 moment's rail Watch works through the feed.
        tap_rail(CTL_PLAY_ID)
        panels = open_panels()
        check("control: rail Watch opens the page-0 TD clip",
              len(panels) == 1 and panels[0]["key"] == CTL_PLAY_ID and panels[0]["video"],
              f"panels={panels}")
        tap_rail(CTL_PLAY_ID)
        check("control: second tap closes it", open_panels() == [])

        # The bug: the off-page moment's rail Watch.
        clicks_before = page.evaluate("window.__railClicks")
        tap_rail(RAIL_PLAY_ID)
        clicks_after = page.evaluate("window.__railClicks")
        check("bug case: the tap registers as a click on the button",
              clicks_after > clicks_before,
              f"clicks {clicks_before} -> {clicks_after}")
        panels = open_panels()
        check("bug case: rail Watch opens the off-page TD clip",
              len(panels) == 1 and panels[0]["key"] == RAIL_PLAY_ID and panels[0]["video"],
              f"panels={panels}")

        # One clip at a time across rail + feed: open the control TD from
        # its FEED card, then open the rail TD from the rail.
        page.tap(f'[data-clip-toggle="{CTL_PLAY_ID}"]')
        page.wait_for_timeout(450)
        panels = open_panels()
        check("feed Watch opens the control TD clip",
              len(panels) == 1 and panels[0]["key"] == CTL_PLAY_ID, f"panels={panels}")
        tap_rail(RAIL_PLAY_ID)
        panels = open_panels()
        check("rail Watch supersedes the feed clip (one open at a time)",
              len(panels) == 1 and panels[0]["key"] == RAIL_PLAY_ID and panels[0]["video"],
              f"panels={panels}")
        tap_rail(RAIL_PLAY_ID)
        check("rail Watch toggles its own clip closed", open_panels() == [])

        # Re-render survival: a poll applies fixture B (one new play) and
        # the page re-renders; the rebuilt rail's buttons must still work.
        old_btn = page.query_selector(f'[data-rail-clip="{RAIL_PLAY_ID}"]')
        page.tap("#rz-timer")
        page.wait_for_timeout(1200)
        new_btn = page.query_selector(f'[data-rail-clip="{RAIL_PLAY_ID}"]')
        check("poll re-render rebuilt the rail", new_btn is not None and new_btn != old_btn)
        tap_rail(RAIL_PLAY_ID)
        panels = open_panels()
        check("rail Watch still opens the clip after a re-render",
              len(panels) == 1 and panels[0]["key"] == RAIL_PLAY_ID and panels[0]["video"],
              f"panels={panels}")

        check("no page errors", errors == [], "; ".join(errors[:3]))
        browser.close()

    print("\n%d failure(s)" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
