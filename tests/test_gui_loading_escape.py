"""GUI Fix 1: one loading rule for every card and tab.

Kaedon approved the top-3 GUI mock (Fix 01: "loading states that never
strand") after the portfolio fix in #2696. Audit of origin/main found the
portfolio pattern had NOT generalized: several loaders used a raw fetch()
with no timeout, so a hung request never settled, the .catch retry card
never fired, and the skeleton/spinner stayed up forever.

Proven stranding paths fixed here (each loader's catch/error state
already existed and already offered Retry; it was simply unreachable on
a hang):

  * Player modal (static/player_modal.js): Team tab, Stats tab, Breakout
    tab, Recent Form card, box score, and the shared _pmSmallFetch (news)
    all used raw fetch(). They now route through _pmFetch, which uses
    brFetchWithTimeout. (The modal's Trades tab already bounded its fetch
    at 15s and the details loader at 12s; those are untouched.)
  * Weekly Hub League Scores tab (static/app.js _lsLoad): raw fetch plus
    an unbounded 3s pending re-poll. The fetch is now bounded (25s) and
    the pending stretch follows the portfolio pattern: pendingSince
    stamp, slow note past ~30s, unavailable + Retry past ~90s.
  * Waivers page (waivers_page.py): wvLoadCandidates created an
    AbortController but nothing ever timed it out; wvLoadLab used a raw
    fetch with no failsafe. Candidates now aborts at 20s (timeout abort
    is an error, a superseding load's abort stays quiet) and the Lab got
    the same 25s DOM failsafe wvLoadStartSit already had.
  * Weekly Hub week switch (weekly_hub_page.py): the AbortController was
    only aborted by the NEXT week change, so a hang left the loading
    overlay up and the selector disabled forever. A 25s timeout now
    aborts and rolls the selector back via failWeekChange.

Surfaces audited and deliberately NOT changed (see the PR description):
portfolio cards (#2696), modal details/ADP/Advanced/Trades loaders,
Advanced Metrics page (15s timeout + Retry), waivers Start/Sit and
Unexpected Performances (20s abort + 25s failsafe), standings and recap
pages (server-rendered, no client loader).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text()


# ── Player modal ─────────────────────────────────────────────────────────────

def _modal():
    return _read("static/player_modal.js")


def test_pm_fetch_helper_bounds_requests_via_shared_timeout():
    src = _modal()
    helper = src.split("function _pmFetch(url, ms)")[1].split("function _pmSmallFetch")[0]
    assert "brFetchWithTimeout" in helper
    assert "20000" in helper


def test_pm_small_fetch_is_bounded():
    src = _modal()
    small = src.split("function _pmSmallFetch(key, url)")[1][:2500]
    assert "_pmFetch(url, 15000)" in small


def test_pm_team_tab_fetch_is_bounded():
    src = _modal()
    team = src.split("function _pmLoadTeamPanel")[1].split("window.pmPickTeamSeason")[0]
    assert "_pmFetch(_teamUrl)" in team
    assert "fetch(_teamUrl)" not in team.replace("_pmFetch(_teamUrl)", "")


def test_pm_stats_and_breakout_tab_fetches_are_bounded():
    src = _modal()
    assert "_bkRequest = _pmFetch(_boUrl)" in src
    stats = src.split("// ── Lazy-load Stats tab")[1].split("// ── Lazy-load Rookie tab")[0]
    assert "_pmFetch(logsUrl)" in stats


def test_pm_recent_form_and_boxscore_fetches_are_bounded():
    src = _modal()
    assert "_pmFetch('/api/player-team-boxscore?' + qs.toString())" in src
    # Recent Form is the first _pmFetch(logsUrl) call site in the file,
    # ahead of the Stats tab's.
    assert src.index("_pmFetch(logsUrl)") < src.index("// ── Lazy-load Stats tab")


# ── League Scores tab (app.js) ───────────────────────────────────────────────

def _league_scores():
    src = _read("static/app.js")
    return src.split("/* Weekly hub: League Scores tab")[1].split("/* Weekly hub: ScoreZone Moments")[0]


def test_ls_fetch_is_bounded():
    ls = _league_scores()
    assert "brFetchWithTimeout(url, { credentials: 'same-origin' }, 25000)" in ls


def test_ls_pending_stretch_is_bounded_like_portfolio():
    ls = _league_scores()
    pending = ls.split("if (d && d.pending)")[1].split("setTimeout(function () { delete tabs._lsLoading;")[0]
    assert "_lsPendingSince" in pending
    assert "30000" in pending  # slow note
    assert "90000" in pending  # unavailable escape
    assert "data-ls-retry" in pending
    # A manual Retry starts a fresh pending budget.
    assert "delete tabsEl._lsPendingSince" in ls


# ── Waivers page ─────────────────────────────────────────────────────────────

def _waivers():
    return _read("dashboard_services/pages/waivers_page.py")


def test_waiver_candidates_fetch_times_out_into_error_state():
    src = _waivers()
    cand = src.split("function wvLoadCandidates()")[1].split("// ── Start/Sit guests")[0]
    assert "candTimedOut" in cand
    assert "20000" in cand
    # Timeout abort reaches the error state; supersede abort stays quiet.
    assert "err.name === 'AbortError' && !candTimedOut" in cand


def test_waiver_lab_has_dom_failsafe_and_stale_guard():
    src = _waivers()
    lab = src.split("function wvLoadLab()")[1].split("function wvLabEsc")[0]
    assert "labFailsafe" in lab
    assert "25000" in lab
    assert ".wv-lab-loading" in lab
    assert "labSeq !== wvLabLoadSeq" in lab


def test_waiver_start_sit_failsafes_still_present_not_redone():
    src = _waivers()
    # The 2026-09-23 escapes the audit verified: do not regress them.
    assert "ssFailsafe" in src and "bgFailsafe" in src


# ── Weekly Hub week switch ───────────────────────────────────────────────────

def test_weekly_hub_week_change_times_out_and_rolls_back():
    src = _read("dashboard_services/pages/weekly_hub_page.py")
    assert "weekTimedOut" in src
    assert "25000" in src
    catch = src.split(".catch(function(err)")[1].split(".finally")[0]
    assert "err.name === 'AbortError' && !weekTimedOut" in catch
    assert "failWeekChange(w, mySeq)" in catch


# ── Service worker ───────────────────────────────────────────────────────────

def test_service_worker_cache_bumped():
    assert "br-fantasy-v76" in _read("static/sw.js")
