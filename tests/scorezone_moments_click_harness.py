"""Behavioral harness for the ScoreZone Moments rail cards.

History: the rail's Watch buttons originally forwarded taps to the
play's FEED card toggle, but the feed is paginated (20 plays per page)
while the rail is built from the whole feed, so most moments had no
toggle in the DOM and their taps silently did nothing (Kaedon,
2026-11-15: "I can't click anything on the moments bar"). The rail
became its own clip surface with a panel under the strip.

Current model (Kaedon's locked design, 2026-11-15): there is no Watch
button at all. The WHOLE card is the tap target. Tapping a card grows
it in place to the rail's full width over 450ms -- its media becomes
the 16:9 clip and starts playing, siblings slide away -- and an X
button on the grown card's name line (or another tap on the card)
shrinks it back. Compact cards carry only a poster, a name and the
points. This harness boots the REAL static/scorezone.js +
static/dashboard.css in Playwright (iPhone viewport + touch) against a
fixture shaped like the scorezone-data API response:
  * a clipped rushing TD early in Q2 (the "rail TD"), buried under 26
    newer plays so its feed card is NOT on feed page 0;
  * a clipped receiving TD late in Q3 (the "control TD") that IS on
    feed page 0.
It taps cards on their name and on their poster (never a button),
samples the grow animation mid-flight, switches cards, checks the
one-clip exclusivity against the feed, re-render survival, and the
reduced-motion path.

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
    # (not on page -- the original dead-tap case).
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

        def make_page(reduced=False):
            ctx_kwargs = dict(device)
            if reduced:
                ctx_kwargs["reduced_motion"] = "reduce"
            ctx = browser.new_context(**ctx_kwargs)
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
            page.add_init_script(
                "window.__cardClicks = 0;"
                "document.addEventListener('click', function(e){"
                "  if (e.target && e.target.closest &&"
                "      e.target.closest('[data-rail-card],[data-clip-toggle]'))"
                "    window.__cardClicks++;"
                "}, true);")
            page.goto("http://rz.test/sleeper/2026/demo/scorezone")
            page.wait_for_selector(".rz-moments-rail", timeout=15000)
            return ctx, page, errors

        def open_panels(page):
            return page.evaluate(
                "Array.from(document.querySelectorAll('.rz-clip-panel.is-open'))"
                ".map(function(p){ return { key: p.getAttribute('data-clip-panel'),"
                "  video: !!p.querySelector('.rz-clip-video'),"
                "  inCard: !!p.closest('.rz-moment-card'),"
                "  html: p.innerHTML.length }; })")

        def grown_key(page):
            return page.evaluate(
                "var c = document.querySelector('.rz-moment-card.is-grown');"
                "c ? c.getAttribute('data-rail-card') : null")

        # ── Main flow ────────────────────────────────────────────────
        ctx, page, errors = make_page()

        cards = page.query_selector_all("[data-rail-card]")
        check("rail renders both clipped TD moments as cards", len(cards) == 2,
              f"found {len(cards)}")
        check("no Watch pill anywhere in the rail",
              page.query_selector(".rz-moment-watch") is None
              and page.query_selector("[data-rail-clip]") is None)
        meta = page.query_selector(".rz-moment-meta")
        check("compact cards carry no kind label row", meta is None)
        feed_toggle_rail = page.query_selector(f'[data-clip-toggle="{RAIL_PLAY_ID}"]')
        feed_toggle_ctl = page.query_selector(f'[data-clip-toggle="{CTL_PLAY_ID}"]')
        check("fixture: rail TD feed card is NOT on feed page 0", feed_toggle_rail is None)
        check("fixture: control TD feed card IS on feed page 0", feed_toggle_ctl is not None)

        # Hit delivery: the topmost element at the rail card's name is
        # inside the card itself (no overlay eats the tap).
        hit = page.evaluate(
            "var c = document.querySelector('[data-rail-card=\"" + RAIL_PLAY_ID + "\"]');"
            "var n = c.querySelector('.rz-moment-name');"
            "var r = n.getBoundingClientRect();"
            "var el = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);"
            "!!el && c.contains(el)")
        check("tap point on the card name is inside the card (no interception)", hit is True)

        # Grow, sampled mid-flight: tap the NAME of the off-page card.
        widths = page.evaluate(
            "(id) => new Promise(function(res){"
            "  var card = document.querySelector('[data-rail-card=\"' + id + '\"]');"
            "  var name = card.querySelector('.rz-moment-name');"
            "  var out = [card.getBoundingClientRect().width];"
            "  var t0 = performance.now();"
            "  name.dispatchEvent(new MouseEvent('click', {bubbles: true}));"
            "  var iv = setInterval(function(){"
            "    out.push(card.getBoundingClientRect().width);"
            "    if (performance.now() - t0 > 700) { clearInterval(iv); res(out); }"
            "  }, 40);"
            "})", RAIL_PLAY_ID)
        compact_w, full_w = widths[0], widths[-1]
        rail_w = page.evaluate(
            "document.querySelector('.rz-moments-rail').getBoundingClientRect().width")
        monotonic = all(widths[i + 1] >= widths[i] - 0.6 for i in range(len(widths) - 1))
        intermediates = [w for w in widths if compact_w + 2 < w < full_w - 2]
        check("grow animates monotonically (no jump)", monotonic,
              f"widths={[round(w) for w in widths]}")
        check("grow passes through intermediate widths", len(intermediates) >= 2,
              f"widths={[round(w) for w in widths]}")
        check("grown card takes the rail's full width",
              abs(full_w - rail_w) <= 3, f"card={full_w} rail={rail_w}")
        check("off-page card grew in place", grown_key(page) == RAIL_PLAY_ID)
        panels = open_panels(page)
        check("clip opens INSIDE the grown card with a video",
              len(panels) == 1 and panels[0]["key"] == RAIL_PLAY_ID
              and panels[0]["video"] and panels[0]["inCard"],
              f"panels={panels}")
        aspect = page.evaluate(
            "var v = document.querySelector('.rz-moment-card.is-grown .rz-clip-video');"
            "var r = v.getBoundingClientRect(); r.width / r.height")
        check("grown video is 16:9", abs(aspect - 16 / 9) < 0.06, f"aspect={aspect:.3f}")
        away = page.evaluate(
            "var rail = document.querySelector('.rz-moments-rail');"
            "var c = document.querySelector('[data-rail-card=\"" + CTL_PLAY_ID + "\"]');"
            "var rr = rail.getBoundingClientRect();"
            "var cr = c.getBoundingClientRect();"
            "cr.right <= rr.left + 1 || cr.left >= rr.right - 1")
        check("sibling card slid out of view while one is grown", away is True)
        x_btn = page.query_selector(f'[data-rail-card="{RAIL_PLAY_ID}"] [data-rail-close]')
        check("grown card shows its X button", x_btn is not None and x_btn.is_visible())

        # X closes: card shrinks back, siblings return, panel empties.
        page.tap(f'[data-rail-card="{RAIL_PLAY_ID}"] [data-rail-close]')
        page.wait_for_timeout(700)
        check("X closes the grown card", grown_key(page) is None and open_panels(page) == [])
        back_w = page.evaluate(
            "document.querySelector('[data-rail-card=\"" + RAIL_PLAY_ID + "\"]')"
            ".getBoundingClientRect().width")
        check("card is compact again after close", abs(back_w - compact_w) <= 3,
              f"width={back_w} compact={compact_w}")

        # Tap the POSTER of the control card: grows that one instead.
        page.tap(f'[data-rail-card="{CTL_PLAY_ID}"] .rz-moment-media')
        page.wait_for_timeout(650)
        check("poster tap grows the control card", grown_key(page) == CTL_PLAY_ID)
        panels = open_panels(page)
        check("control clip opens inside its card",
              len(panels) == 1 and panels[0]["key"] == CTL_PLAY_ID and panels[0]["video"],
              f"panels={panels}")

        # Switch: tap the other card -- shrink-then-grow, one clip open.
        page.tap(f'[data-rail-card="{RAIL_PLAY_ID}"] .rz-moment-name')
        page.wait_for_timeout(1250)
        check("switch grows the newly tapped card", grown_key(page) == RAIL_PLAY_ID)
        panels = open_panels(page)
        check("switch leaves exactly one open clip (the new one)",
              len(panels) == 1 and panels[0]["key"] == RAIL_PLAY_ID and panels[0]["video"],
              f"panels={panels}")

        # Exclusivity vs the feed, both directions.
        page.tap(f'[data-clip-toggle="{CTL_PLAY_ID}"]')
        page.wait_for_timeout(700)
        panels = open_panels(page)
        check("feed Watch supersedes the grown rail card",
              len(panels) == 1 and panels[0]["key"] == CTL_PLAY_ID
              and not panels[0]["inCard"] and grown_key(page) is None,
              f"panels={panels}")
        page.tap(f'[data-rail-card="{RAIL_PLAY_ID}"] .rz-moment-name')
        page.wait_for_timeout(650)
        panels = open_panels(page)
        check("rail card supersedes the feed clip (one open at a time)",
              len(panels) == 1 and panels[0]["key"] == RAIL_PLAY_ID and panels[0]["inCard"],
              f"panels={panels}")

        # Poll survival: polls patch in place and never rebuild the
        # rail, so the grown card and its live clip ride through.
        page.tap("#rz-timer")
        page.wait_for_timeout(1500)
        check("grown card survives a poll with its clip open",
              grown_key(page) == RAIL_PLAY_ID and len(open_panels(page)) == 1,
              f"grown={grown_key(page)} panels={open_panels(page)}")

        # Full re-render survival: a filter toggle rebuilds the page;
        # the rebuilt rail restores the grown card from the open key.
        page.evaluate(
            "document.querySelector('.rz-moment-card.is-grown')"
            ".setAttribute('data-testmark', 'old')")
        page.tap("#rz-bigplays-btn")
        page.wait_for_timeout(900)
        restored = page.evaluate(
            "var c = document.querySelector('.rz-moment-card.is-grown');"
            "c ? { key: c.getAttribute('data-rail-card'),"
            "      rebuilt: !c.hasAttribute('data-testmark') } : null")
        check("full re-render restores the grown card by key",
              restored is not None and restored["key"] == RAIL_PLAY_ID and restored["rebuilt"],
              f"restored={restored}")
        panels = open_panels(page)
        check("restored grown card's clip is open again",
              len(panels) == 1 and panels[0]["key"] == RAIL_PLAY_ID and panels[0]["inCard"],
              f"panels={panels}")
        page.tap(f'[data-rail-card="{RAIL_PLAY_ID}"] [data-rail-close]')
        page.wait_for_timeout(700)
        check("rebuilt card's X still closes it",
              grown_key(page) is None and open_panels(page) == [])

        check("no page errors", errors == [], "; ".join(errors[:3]))
        ctx.close()

        # ── Reduced motion: everything lands instantly ───────────────
        ctx2, page2, errors2 = make_page(reduced=True)
        page2.tap(f'[data-rail-card="{RAIL_PLAY_ID}"] .rz-moment-name')
        page2.wait_for_timeout(200)
        check("reduced motion: card grows instantly",
              grown_key(page2) == RAIL_PLAY_ID)
        w2 = page2.evaluate(
            "var rail = document.querySelector('.rz-moments-rail');"
            "var card = document.querySelector('.rz-moment-card.is-grown');"
            "Math.abs(card.getBoundingClientRect().width"
            "  - rail.getBoundingClientRect().width) <= 3")
        check("reduced motion: grown width is already final", w2 is True)
        panels = open_panels(page2)
        check("reduced motion: clip still opens and plays in the card",
              len(panels) == 1 and panels[0]["video"] and panels[0]["inCard"],
              f"panels={panels}")
        page2.tap(f'[data-rail-card="{RAIL_PLAY_ID}"] [data-rail-close]')
        page2.wait_for_timeout(200)
        check("reduced motion: close is instant too",
              grown_key(page2) is None and open_panels(page2) == [])
        check("no page errors (reduced motion)", errors2 == [], "; ".join(errors2[:3]))
        ctx2.close()

        browser.close()

    print("\n%d failure(s)" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
