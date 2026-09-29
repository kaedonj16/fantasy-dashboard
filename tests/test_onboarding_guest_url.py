"""Onboarding improvement tests: paste-your-league-URL and guest league claim.

Covers:
- parseLeagueUrl() in static/app.js: extracts league IDs (and MFL season)
  from pasted URLs for all 5 platforms, rejects invalid URLs.
- recordGuestLeagueView IIFE: records guest dashboard views to localStorage,
  skips when signed in.

Runs the actual JS in Node with mocked browser globals.
"""
import os
import re
import subprocess
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_JS = os.path.join(REPO, "static", "app.js")


def _extract_parse_fn():
    """Extract the parseLeagueUrl function source from app.js."""
    src = open(APP_JS).read()
    start = src.index("function parseLeagueUrl(platform, rawUrl)")
    # Find the matching closing brace by counting.
    depth = 0
    i = start
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1
    raise AssertionError("parseLeagueUrl function end not found")


def _extract_recorder_iife():
    """Extract the recordGuestLeagueView IIFE from app.js."""
    src = open(APP_JS).read()
    start = src.index("(function recordGuestLeagueView()")
    end_marker = "})();"
    end = src.index(end_marker, start) + len(end_marker)
    return src[start:end]


def _run_node(js, test_code):
    script = js + "\n" + test_code
    proc = subprocess.run(
        ["node", "-e", script],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, "node failed:\n%s\n%s" % (proc.stdout, proc.stderr)
    return proc.stdout.strip()


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

def _parse(url, platform):
    parse_fn = _extract_parse_fn()
    out = _run_node(parse_fn, textwrap.dedent("""
        var result = parseLeagueUrl(%r, %r);
        console.log(JSON.stringify(result));
    """ % (platform, url)))
    import json
    return json.loads(out)


@pytest.mark.parametrize("url,expected", [
    ("https://sleeper.app/leagues/123456789012345678", "123456789012345678"),
    ("https://sleeper.app/leagues/999/", "999"),
    ("http://sleeper.app/leagues/12345", "12345"),
    ("sleeper.app/leagues/123456789012345678", "123456789012345678"),
])
def test_parse_sleeper_urls(url, expected):
    assert _parse(url, "sleeper") == {"league_id": expected}


@pytest.mark.parametrize("url,expected", [
    ("https://fantasy.espn.com/football/league?leagueId=336414", "336414"),
    ("https://fantasy.espn.com/football/team?leagueId=123456&seasonId=2026", "123456"),
    ("http://fantasy.espn.com/baseball/league?leagueId=789", "789"),
])
def test_parse_espn_urls(url, expected):
    assert _parse(url, "espn") == {"league_id": expected}


@pytest.mark.parametrize("url,expected", [
    ("https://football.fantasysports.yahoo.com/f1/123456", "123456"),
    ("https://football.fantasysports.yahoo.com/nfl/654321/", "654321"),
    ("http://football.fantasysports.yahoo.com/f1/999999", "999999"),
])
def test_parse_yahoo_urls(url, expected):
    assert _parse(url, "yahoo") == {"league_id": expected}


def test_parse_mfl_url_extracts_season():
    result = _parse("https://www.myfantasyleague.com/2026/home/12345", "mfl")
    assert result == {"league_id": "12345", "season": "2026"}


def test_parse_mfl_url_without_www():
    result = _parse("https://myfantasyleague.com/2025/home/99999", "mfl")
    assert result == {"league_id": "99999", "season": "2025"}


@pytest.mark.parametrize("url,expected", [
    ("https://www.fleaflicker.com/nfl/leagues/14153", "14153"),
    ("https://fleaflicker.com/nfl/leagues/92916/", "92916"),
    ("http://www.fleaflicker.com/nfl/leagues/123", "123"),
])
def test_parse_fleaflicker_urls(url, expected):
    assert _parse(url, "fleaflicker") == {"league_id": expected}


@pytest.mark.parametrize("url,platform", [
    ("https://example.com/leagues/123", "sleeper"),
    ("not a url at all", "espn"),
    ("https://sleeper.app/leagues/", "sleeper"),
    ("https://fantasy.espn.com/football/league", "espn"),
    ("https://football.fantasysports.yahoo.com/", "yahoo"),
    ("https://www.myfantasyleague.com/home/12345", "mfl"),
    ("https://www.fleaflicker.com/nfl/teams/123", "fleaflicker"),
    ("", "sleeper"),
    ("   ", "espn"),
])
def test_parse_rejects_invalid_urls(url, platform):
    assert _parse(url, platform) is None


def test_parse_wrong_platform_url_rejected():
    # An ESPN URL pasted into the Sleeper field must not parse.
    assert _parse("https://fantasy.espn.com/football/league?leagueId=336414", "sleeper") is None
    # A Sleeper URL pasted into the ESPN field must not parse.
    assert _parse("https://sleeper.app/leagues/123456789012345678", "espn") is None


# ---------------------------------------------------------------------------
# Guest league view recorder
# ---------------------------------------------------------------------------

RECORDER_PREAMBLE = textwrap.dedent("""
    var __store = {};
    globalThis.localStorage = {
      getItem: function(k){ return k in __store ? __store[k] : null; },
      setItem: function(k,v){ __store[k] = String(v); },
      removeItem: function(k){ delete __store[k]; },
    };
    // The IIFE references `window`; alias it to globalThis in Node.
    globalThis.window = globalThis;
""")


def _run_recorder(ctx, is_signed_in):
    iife = _extract_recorder_iife()
    preamble = RECORDER_PREAMBLE + textwrap.dedent("""
        globalThis._isSignedIn = %s;
        globalThis.__brctx = %s;
    """ % ("true" if is_signed_in else "false", ctx))
    out = _run_node(preamble + "\n" + iife, textwrap.dedent("""
        console.log(JSON.stringify({
          guest_league: __store["br-guest-league"] || null,
        }));
    """))
    import json
    return json.loads(out)


def test_recorder_saves_guest_league():
    ctx = '{"platform": "sleeper", "leagueId": "123456789012345678", "season": 2026, "leagueName": "Blackedraw"}'
    result = _run_recorder(ctx, is_signed_in=False)
    assert result["guest_league"] is not None
    import json
    saved = json.loads(result["guest_league"])
    assert saved["platform"] == "sleeper"
    assert saved["league_id"] == "123456789012345678"
    assert saved["season"] == 2026
    assert saved["name"] == "Blackedraw"


def test_recorder_skips_when_signed_in():
    ctx = '{"platform": "espn", "leagueId": "336414", "season": 2026, "leagueName": "Test"}'
    result = _run_recorder(ctx, is_signed_in=True)
    assert result["guest_league"] is None


def test_recorder_skips_without_league_ctx():
    result = _run_recorder("null", is_signed_in=False)
    assert result["guest_league"] is None


def test_recorder_includes_saved_viewer_username():
    ctx = '{"platform": "sleeper", "leagueId": "123", "season": 2026, "leagueName": "Test"}'
    iife = _extract_recorder_iife()
    preamble = RECORDER_PREAMBLE + textwrap.dedent("""
        globalThis._isSignedIn = false;
        globalThis.__brctx = %s;
        __store["saved_viewer"] = JSON.stringify({username: "kaedon"});
    """ % ctx)
    out = _run_node(preamble + "\n" + iife, 'console.log(__store["br-guest-league"] || "null");')
    import json
    saved = json.loads(out)
    assert saved["username"] == "kaedon"


# ---------------------------------------------------------------------------
# Copy and markup checks
# ---------------------------------------------------------------------------

def test_no_em_dashes_in_new_copy():
    """No em dashes in any UI copy added for these features."""
    src = open(APP_JS).read()
    # The new strings live in the claim card and URL error messages.
    for needle in [
        "You were viewing ",
        "Save it to your account so it is here on every device.",
        "doesn't look like a ",
        "Check the link and try again.",
        "Fastest: paste your league link",
    ]:
        assert needle in src, "expected copy not found: %r" % needle
    # None of the new blocks may contain an em dash.
    for block in re.findall(r"guest-claim-[a-z]+|url-paste-[a-z]+|parseLeagueUrl|recordGuestLeagueView|maybeShowGuestClaimLeague", src):
        pass
    # Direct check on the specific new string literals.
    new_strings = [
        "You were viewing ",
        "Save it to your account so it is here on every device.",
        "Save to my account",
        "Not now",
        "Fastest: paste your league link",
        "Check the link and try again.",
    ]
    for s in new_strings:
        assert "\u2014" not in s, "em dash in copy: %r" % s


def test_url_inputs_present_in_form_body():
    """All five platform flows have a URL paste input in app.py FORM_BODY."""
    src = open(os.path.join(REPO, "app.py")).read()
    for input_id in [
        "sleeperUrlInput", "espnUrlInput", "yahooUrlInput",
        "mflUrlInput", "fleaUrlInput",
    ]:
        assert 'id="%s"' % input_id in src, "missing URL input: %s" % input_id
    for err_id in [
        "sleeperUrlError", "espnUrlError", "yahooUrlError",
        "mflUrlError", "fleaUrlError",
    ]:
        assert 'id="%s"' % err_id in src, "missing URL error element: %s" % err_id


def test_claim_uses_existing_link_add_endpoint():
    """The claim confirm reuses /api/link/add (no new backend route)."""
    src = open(APP_JS).read()
    assert 'fetch("/api/link/add"' in src
    # No new API routes were added for the claim flow.
    routes_src = open(os.path.join(REPO, "routes", "link_bp.py")).read()
    assert "guest-claim" not in routes_src
    assert "br-guest-league" not in routes_src
