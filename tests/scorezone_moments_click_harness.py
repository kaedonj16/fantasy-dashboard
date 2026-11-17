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
points.

From-the-thumbnail refinement (Kaedon, 2026-11-16): the clip must
grow FROM the little thumbnail, not from below it. The first build
collapsed the thumbnail away and popped a separate panel in under
the name line, so the expansion appeared to start underneath the
thumbnail. Now the clip panel lives inside the card's media box and
the thumbnail's own box is the expanding surface: its top edge stays
anchored, its width and height grow monotonically from the compact
thumbnail rect to the full-width 16:9 player rect, the grown card
holds exactly one media surface, and closing returns the box to the
thumbnail rect. The grow geometry is sampled by seeking the running
CSS transitions (headless WebKit starves rAF/setInterval sampling
down to a handful of frames, which made tick-based sampling report
a jump that the transition timeline disproves; seeking the
transition's own currentTime reads the same per-frame geometry
deterministically, and a build with no transition has nothing to
seek and fails).

This harness boots the REAL static/scorezone.js +
static/dashboard.css in Playwright (iPhone viewport + touch) against a
fixture shaped like the scorezone-data API response:
  * a clipped rushing TD early in Q2 (the "rail TD"), buried under 26
    newer plays so its feed card is NOT on feed page 0;
  * a clipped receiving TD late in Q3 (the "control TD") that IS on
    feed page 0.
It taps cards on their name and on their poster (never a button),
samples the grow animation mid-flight, switches cards, checks the
one-clip exclusivity against the feed, re-render survival, and the
reduced-motion path. A third fixture loads six clipped TDs with long
player names (Pat Freiermuth ... Equanimeous St. Brown) and asserts
names render in full in compact cards and on the grown header --
the 2026-11-15 "don't truncate the names" fix.

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


LONG_PLAYERS = {
    "2001": {"name": "Pat Freiermuth", "pos": "TE", "team": "KC"},
    "2002": {"name": "Samaje Perine", "pos": "RB", "team": "KC"},
    "2003": {"name": "Jaxon Smith-Njigba", "pos": "WR", "team": "KC"},
    "2004": {"name": "Amon-Ra St. Brown", "pos": "WR", "team": "KC"},
    "2005": {"name": "Equanimeous St. Brown", "pos": "WR", "team": "KC"},
    "2006": {"name": "DK Metcalf", "pos": "WR", "team": "KC"},
}
LONG_NAMES = [p["name"] for p in LONG_PLAYERS.values()]


def payload_long_names():
    """Six clipped TD moments whose names stress the compact card's
    name line (the 2026-11-15 truncation bug: 'Pat Freierm…',
    'Samaje Per…'). DK Metcalf is the short-name control."""
    specs = [
        ("2001", "ln-1", line(rec=1, targets=1, rec_yds=11, rec_td=1)),
        ("2002", "ln-2", line(carries=1, rush_yds=8, rush_td=1)),
        ("2003", "ln-3", line(rec=1, targets=1, rec_yds=25, rec_td=1)),
        ("2004", "ln-4", line(rec=1, targets=1, rec_yds=14, rec_td=1)),
        ("2005", "ln-5", line(rec=1, targets=1, rec_yds=40, rec_td=1)),
        ("2006", "ln-6", line(rec=1, targets=1, rec_yds=9, rec_td=1)),
    ]
    plays = []
    for i, (pid, play_id, stat_line) in enumerate(specs):
        clip = {
            "clip_tweet_id": f"179000000000000001{i}",
            "clip_url": f"https://x.com/NFL/status/179000000000000001{i}",
            "clip_video_url": f"https://video.twimg.com/ext_tw_video/1790/pu/vid/avc1/ln{i}.mp4",
            "clip_poster_url": f"https://pbs.twimg.com/ext_tw_video_thumb/1790/pu/img/ln{i}.jpg",
        }
        _seq[0] += 1
        plays.append({
            "play_id": play_id, "seq": _seq[0], "game_id": GID,
            "quarter": "Q2", "clock": f"{10 - i}:00",
            "down": "2", "distance": "6", "yard_line": "BUF 30",
            "play_text": f"{LONG_PLAYERS[pid]['name']} scores a touchdown",
            "pid": pid, "name": LONG_PLAYERS[pid]["name"], "team": "KC",
            "stat_line": stat_line, "is_td": True, **clip,
        })
    player_info = {}
    for pid, p in LONG_PLAYERS.items():
        player_info[pid] = {
            "name": p["name"], "pos": p["pos"], "team": p["team"],
            "game_id": GID, "game_code": "1", "game_quarter": "Q3",
            "game_clock": "0:45", "game_status": "",
            "away": "BUF", "home": "KC", "away_pts": "17", "home_pts": "24",
            "game_time_epoch": KICKOFF,
            "stat_line": line(carries=8, rush_yds=40, rec=4, targets=6,
                              rec_yds=50, pass_yds=180),
        }
    base = payload(False)
    base["player_info"] = player_info
    base["pbp_by_game"] = {GID: plays}
    base["matchups"] = [
        {"roster_id": 1, "matchup_id": 1, "points": 24.0,
         "projected_pts": 30.0, "starters": list(LONG_PLAYERS),
         "players": list(LONG_PLAYERS),
         "players_points": {pid: 7.0 for pid in LONG_PLAYERS}},
        {"roster_id": 2, "matchup_id": 1, "points": 17.0,
         "projected_pts": 28.0, "starters": [], "players": [],
         "players_points": {}},
    ]
    return base


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

        def make_page(reduced=False, fixture=None):
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
                    if fixture is not None:
                        body = fixture
                    else:
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
                                  body=HTML.replace("__PAYLOAD__",
                                                    json.dumps(fixture if fixture is not None
                                                               else fixture_a)))
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
        # Sampling seeks the running transitions (see module docstring):
        # each seek lands on a real frame of the grow, so the recorded
        # card widths and media rects are the animation's own geometry.
        # Do NOT rewrite this as rAF/setInterval tick sampling: headless
        # WebKit starves those callbacks to a few frames per animation,
        # so tick sampling sees only the endpoints and reports a phantom
        # jump. Seeking transition.currentTime is what makes the
        # intermediate frames observable at all. The two transitions
        # sought below are the card's flex-basis and the media box's
        # height; both must exist or the grow is not actually animating.
        # The first snap() in the series is the pre-tap compact rect:
        # every later frame is asserted against it (anchored media top,
        # monotonic growth), never against fixed pixel expectations.
        frames = page.evaluate(
            "(id) => {"
            "  var card = document.querySelector('[data-rail-card=\"' + id + '\"]');"
            "  var media = card.querySelector('.rz-moment-media');"
            "  var name = card.querySelector('.rz-moment-name');"
            "  function snap(){"
            "    var cr = card.getBoundingClientRect();"
            "    var mr = media.getBoundingClientRect();"
            "    return { cw: cr.width, mw: mr.width, mh: mr.height, mt: mr.top };"
            "  }"
            "  var out = [snap()];"
            "  name.dispatchEvent(new MouseEvent('click', {bubbles: true}));"
            "  var cardAnim = null, mediaAnim = null;"
            "  card.getAnimations().forEach(function(a){"
            "    if (a.transitionProperty === 'flex-basis') cardAnim = a; });"
            "  media.getAnimations().forEach(function(a){"
            "    if (a.transitionProperty === 'height') mediaAnim = a; });"
            "  if (cardAnim && mediaAnim) {"
            "    [90, 180, 270, 360, 450].forEach(function(t){"
            "      cardAnim.currentTime = t; mediaAnim.currentTime = t;"
            "      out.push(snap());"
            "    });"
            "    cardAnim.finish(); mediaAnim.finish();"
            "  }"
            "  return out;"
            "}", RAIL_PLAY_ID)
        widths = [f["cw"] for f in frames]
        compact_w, full_w = widths[0], widths[-1]
        rail_w = page.evaluate(
            "document.querySelector('.rz-moments-rail').getBoundingClientRect().width")
        monotonic = all(widths[i + 1] >= widths[i] - 0.6 for i in range(len(widths) - 1))
        intermediates = [w for w in widths if compact_w + 2 < w < full_w - 2]
        check("grow animates monotonically (no jump)", monotonic,
              f"widths={[round(w) for w in widths]}")
        check("grow passes through intermediate widths", len(intermediates) >= 2,
              f"widths={[round(w) for w in widths]}")
        # The thumbnail is the expanding surface: its top edge never
        # moves, and its own width/height grow monotonically from the
        # compact thumbnail rect to the full-width 16:9 player rect.
        m0 = frames[0]
        m_top_ok = all(abs(f["mt"] - m0["mt"]) <= 2 for f in frames)
        check("grow: media top edge stays anchored at the thumbnail's top",
              m_top_ok, f"tops={[round(f['mt']) for f in frames]}")
        m_mono = all(frames[i + 1]["mw"] >= frames[i]["mw"] - 0.6
                     and frames[i + 1]["mh"] >= frames[i]["mh"] - 0.6
                     for i in range(len(frames) - 1))
        check("grow: media width and height grow monotonically from the thumbnail",
              m_mono, f"frames={[(round(f['mw']), round(f['mh'])) for f in frames]}")
        m_inter = [f for f in frames if m0["mh"] + 2 < f["mh"] < frames[-1]["mh"] - 2]
        check("grow: media height passes through intermediate sizes",
              len(m_inter) >= 2,
              f"heights={[round(f['mh']) for f in frames]}")
        chrome = m0["cw"] - m0["mw"]
        check("grow: media ends at the rail's full width at 16:9",
              abs(frames[-1]["mw"] - (rail_w - chrome)) <= 3
              and abs(frames[-1]["mh"] - frames[-1]["mw"] * 9 / 16) <= 3,
              f"final media={frames[-1]['mw']}x{frames[-1]['mh']} rail={rail_w}")
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
        # One media surface in the grown card: the clip panel lives
        # INSIDE the thumbnail's media box (never a second block below
        # it), the box itself is the full 16:9 player (not collapsed
        # away, the original from-below build had it at height 0), and
        # the video fills that box exactly.
        surface = page.evaluate(
            "var c = document.querySelector('.rz-moment-card.is-grown');"
            "var media = c.querySelector('.rz-moment-media');"
            "var open = c.querySelector('.rz-clip-panel.is-open');"
            "var v = c.querySelector('.rz-clip-video');"
            "var mr = media.getBoundingClientRect();"
            "var vr = v ? v.getBoundingClientRect() : null;"
            "({ mediaCount: c.querySelectorAll('.rz-moment-media').length,"
            "   panelCount: c.querySelectorAll('.rz-clip-panel').length,"
            "   panelInMedia: !!(open && media.contains(open)),"
            "   mediaW: mr.width, mediaH: mr.height,"
            "   videoFillsBox: !!vr && Math.abs(vr.width - mr.width) <= 2"
            "     && Math.abs(vr.height - mr.height) <= 2"
            "     && Math.abs(vr.top - mr.top) <= 2 })")
        check("grown card holds exactly one media surface (panel inside the thumbnail box)",
              surface["mediaCount"] == 1 and surface["panelCount"] == 1
              and surface["panelInMedia"],
              f"surface={surface}")
        check("grown media box is the player itself (full width, 16:9, not collapsed)",
              surface["mediaH"] > 100
              and abs(surface["mediaH"] - surface["mediaW"] * 9 / 16) <= 3,
              f"media={surface['mediaW']}x{surface['mediaH']}")
        check("video fills the media box exactly (one continuous surface)",
              surface["videoFillsBox"], f"surface={surface}")
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
        closed_m = page.evaluate(
            "var m = document.querySelector('[data-rail-card=\"" + RAIL_PLAY_ID + "\"]')"
            "  .querySelector('.rz-moment-media').getBoundingClientRect();"
            "({ w: m.width, h: m.height, top: m.top })")
        check("media box returns to the thumbnail rect on close",
              abs(closed_m["w"] - m0["mw"]) <= 3
              and abs(closed_m["h"] - m0["mh"]) <= 3
              and abs(closed_m["top"] - m0["mt"]) <= 3,
              f"closed={closed_m} compact={m0}")

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
        rm_media = page2.evaluate(
            "var m = document.querySelector('.rz-moment-card.is-grown .rz-moment-media')"
            ".getBoundingClientRect(); ({ w: m.width, h: m.height })")
        check("reduced motion: media box is instantly at the final 16:9 player size",
              rm_media["h"] > 100
              and abs(rm_media["h"] - rm_media["w"] * 9 / 16) <= 3,
              f"media={rm_media}")
        page2.tap(f'[data-rail-card="{RAIL_PLAY_ID}"] [data-rail-close]')
        page2.wait_for_timeout(200)
        check("reduced motion: close is instant too",
              grown_key(page2) is None and open_panels(page2) == [])
        check("no page errors (reduced motion)", errors2 == [], "; ".join(errors2[:3]))
        ctx2.close()

        # ── Long names: never truncated (Kaedon, 2026-11-15) ────────
        # Compact cards must show every name IN FULL beside the points:
        # no ellipsis, no clipping, at phone width. Long names wrap to
        # a second line; single-line names render exactly as before.
        ctx3, page3, errors3 = make_page(fixture=payload_long_names())
        cards3 = page3.query_selector_all("[data-rail-card]")
        check("long-name fixture renders all six moment cards", len(cards3) == 6,
              f"found {len(cards3)}")

        def measure_names(page):
            return page.evaluate(
                "Array.from(document.querySelectorAll('[data-rail-card]')).map(function(card){"
                "  var name = card.querySelector('.rz-moment-name');"
                "  var delta = card.querySelector('.rz-moment-delta');"
                "  var cs = getComputedStyle(name);"
                "  var probe = document.createElement('div');"
                "  probe.style.cssText = 'position:absolute;left:-9999px;top:0;visibility:hidden;'"
                "    + 'font-size:' + cs.fontSize + ';font-weight:' + cs.fontWeight"
                "    + ';font-family:' + cs.fontFamily + ';line-height:' + cs.lineHeight"
                "    + ';letter-spacing:' + cs.letterSpacing"
                "    + ';white-space:normal;overflow-wrap:break-word;'"
                "    + 'width:' + name.clientWidth + 'px;';"
                "  probe.textContent = name.textContent;"
                "  document.body.appendChild(probe);"
                "  var needH = probe.clientHeight;"
                "  probe.remove();"
                "  var nr = name.getBoundingClientRect();"
                "  var dr = delta.getBoundingClientRect();"
                "  var cr = card.getBoundingClientRect();"
                "  var tnode = name.firstChild;"
                "  var broken = [];"
                "  if (tnode) {"
                "    var segs = name.textContent.split(/[\\s-]+/).filter(Boolean);"
                "    var sidx = 0;"
                "    for (var si = 0; si < segs.length; si++) {"
                "      var at = name.textContent.indexOf(segs[si], sidx);"
                "      var rg = document.createRange();"
                "      rg.setStart(tnode, at); rg.setEnd(tnode, at + segs[si].length);"
                "      if (rg.getClientRects().length !== 1) broken.push(segs[si]);"
                "      sidx = at + segs[si].length;"
                "    }"
                "  }"
                "  return { text: name.textContent, brokenWords: broken,"
                "    whiteSpace: cs.whiteSpace, textOverflow: cs.textOverflow,"
                "    scrollW: name.scrollWidth, clientW: name.clientWidth,"
                "    clientH: name.clientHeight, needH: needH,"
                "    deltaText: delta.textContent,"
                "    deltaScrollW: delta.scrollWidth, deltaClientW: delta.clientWidth,"
                "    deltaInside: dr.left >= cr.left - 1 && dr.right <= cr.right + 1,"
                "    nameCenterY: nr.top + nr.height / 2,"
                "    deltaCenterY: dr.top + dr.height / 2,"
                "    cardH: cr.height,"
                "    mediaTop: card.querySelector('.rz-moment-media')"
                "      .getBoundingClientRect().top };"
                "})")

        measured = {m["text"]: m for m in measure_names(page3)}
        check("every long-name card carries its full name text",
              all(n in measured for n in LONG_NAMES),
              f"cards={sorted(measured)}")
        for long_name in LONG_NAMES[:-1]:
            m = measured.get(long_name)
            if not m:
                continue
            check(f"compact name not on the ellipsis path: {long_name}",
                  m["whiteSpace"] != "nowrap" and m["textOverflow"] != "ellipsis",
                  f"whiteSpace={m['whiteSpace']} textOverflow={m['textOverflow']}")
            check(f"compact name fully rendered (no clip): {long_name}",
                  m["scrollW"] <= m["clientW"] + 1 and m["clientH"] >= m["needH"] - 1,
                  f"scrollW={m['scrollW']} clientW={m['clientW']} "
                  f"clientH={m['clientH']} needH={m['needH']}")
            check(f"points fully visible with the name: {long_name}",
                  m["deltaScrollW"] <= m["deltaClientW"] + 1 and m["deltaInside"]
                  and m["deltaText"].startswith("+"),
                  f"delta={m['deltaText']} scrollW={m['deltaScrollW']} "
                  f"clientW={m['deltaClientW']} inside={m['deltaInside']}")
            check(f"name never breaks mid-word: {long_name}",
                  m["brokenWords"] == [], f"broken={m['brokenWords']}")
        if len(measured) == len(LONG_NAMES):
            heights = [measured[n]["cardH"] for n in LONG_NAMES]
            tops = [measured[n]["mediaTop"] for n in LONG_NAMES]
            check("rail row stays even: all cards share one height",
                  max(heights) - min(heights) <= 2, f"heights={[round(h) for h in heights]}")
            check("rail row stays even: posters align at the top",
                  max(tops) - min(tops) <= 2, f"tops={[round(t) for t in tops]}")
            one_line_h = measured["DK Metcalf"]["clientH"]
            check("long names actually wrap to a second line",
                  max(measured[n]["clientH"] for n in LONG_NAMES[:-1]) > one_line_h + 4,
                  f"one_line_h={one_line_h}")
            dk = measured["DK Metcalf"]
            check("short name stays on one line beside its points (unchanged)",
                  dk["clientH"] <= dk["needH"] + 1
                  and abs(dk["nameCenterY"] - dk["deltaCenterY"]) <= 2,
                  f"clientH={dk['clientH']} needH={dk['needH']}")
            check("short name not broken mid-word either",
                  dk["brokenWords"] == [], f"broken={dk['brokenWords']}")

        # The grown card's header line must not truncate either.
        eq_key = page3.evaluate(
            "var c = Array.from(document.querySelectorAll('[data-rail-card]'))"
            ".find(function(x){ return x.querySelector('.rz-moment-name')"
            "  .textContent === 'Equanimeous St. Brown'; });"
            "c ? c.getAttribute('data-rail-card') : null")
        check("found the Equanimeous St. Brown card", eq_key is not None)
        if eq_key:
            page3.tap(f'[data-rail-card="{eq_key}"] .rz-moment-name')
            page3.wait_for_timeout(700)
            check("long-name card grows in place", grown_key(page3) == eq_key)
            # NB: no top-level `var name` here -- at global scope that
            # assigns window.name (a string), not the element.
            grown_m = page3.evaluate(
                "var gCard = document.querySelector('.rz-moment-card.is-grown');"
                "var nm = gCard.querySelector('.rz-moment-name');"
                "var xBtn = gCard.querySelector('[data-rail-close]');"
                "var cs = getComputedStyle(nm);"
                "var probe = document.createElement('div');"
                "probe.style.cssText = 'position:absolute;left:-9999px;top:0;visibility:hidden;'"
                "  + 'font-size:' + cs.fontSize + ';font-weight:' + cs.fontWeight"
                "  + ';font-family:' + cs.fontFamily + ';line-height:' + cs.lineHeight"
                "  + ';letter-spacing:' + cs.letterSpacing"
                "  + ';white-space:normal;overflow-wrap:break-word;'"
                "  + 'width:' + nm.clientWidth + 'px;';"
                "probe.textContent = nm.textContent;"
                "document.body.appendChild(probe);"
                "var needH = probe.clientHeight;"
                "probe.remove();"
                "var tnode = nm.firstChild;"
                "var broken = [];"
                "if (tnode) {"
                "  var segs = nm.textContent.split(/[\\s-]+/).filter(Boolean);"
                "  var sidx = 0;"
                "  for (var si = 0; si < segs.length; si++) {"
                "    var at = nm.textContent.indexOf(segs[si], sidx);"
                "    var rg = document.createRange();"
                "    rg.setStart(tnode, at); rg.setEnd(tnode, at + segs[si].length);"
                "    if (rg.getClientRects().length !== 1) broken.push(segs[si]);"
                "    sidx = at + segs[si].length;"
                "  }"
                "}"
                "({ text: nm.textContent, whiteSpace: cs.whiteSpace, brokenWords: broken,"
                "   scrollW: nm.scrollWidth, clientW: nm.clientWidth,"
                "   clientH: nm.clientHeight, needH: needH,"
                "   nameRight: nm.getBoundingClientRect().right,"
                "   xLeft: xBtn.getBoundingClientRect().left,"
                "   xVisible: !!(xBtn.offsetWidth || xBtn.offsetHeight) })")
            check("grown header shows the full name (no truncation)",
                  grown_m["text"] == "Equanimeous St. Brown"
                  and grown_m["whiteSpace"] != "nowrap"
                  and grown_m["scrollW"] <= grown_m["clientW"] + 1
                  and grown_m["clientH"] >= grown_m["needH"] - 1
                  and grown_m["brokenWords"] == [],
                  f"grown={grown_m}")
            check("grown name does not run under the X button",
                  grown_m["xVisible"] and grown_m["nameRight"] <= grown_m["xLeft"] + 2,
                  f"nameRight={grown_m['nameRight']} xLeft={grown_m['xLeft']}")
            page3.tap(f'[data-rail-card="{eq_key}"] [data-rail-close]')
            page3.wait_for_timeout(700)
            check("long-name card closes again", grown_key(page3) is None)
        check("no page errors (long names)", errors3 == [], "; ".join(errors3[:3]))
        ctx3.close()

        browser.close()

    print("\n%d failure(s)" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
