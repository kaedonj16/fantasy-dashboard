"""Playoff Path tab gating for Yahoo leagues.

Regression (2026-11-10): Kaedon got the Playoff Path tab for her Sleeper
league but not for her Yahoo league (1307110). The whole downstream chain
(client loader, /api/playoff-odds, /api/playoff-scenarios, the sim) is
platform-neutral, but the standings-page gate in
``_standings_playoff_params`` read ``league_settings.playoff_week_start``
with NO fallback, while the sim, the odds route and the scenarios layer
all resolve ``playoff_week_start or 15``. A Yahoo ctx whose settings
reached the standings page without that key (the Yahoo globals shape)
silently lost the tab (and the standings cut line) while Playoff Odds
simulated the race on the default.

These tests pin the parity resolution: rules record -> settings (either
spelling) -> the shared default of 15; and the final-five-weeks window
semantics are unchanged when a real start week IS present.
"""
import datetime as _dt
import random
from pathlib import Path

import pytest

pytest.importorskip("flask")
pd = pytest.importorskip("pandas")
pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]

OWNERS = [f"Team {chr(65 + i)}" for i in range(10)]
RIDS = [str(i + 1) for i in range(10)]


def _build_df_weekly(weeks_played: int) -> "pd.DataFrame":
    random.seed(7)
    rows = []
    for w in range(1, weeks_played + 1):
        order = list(range(10))
        random.shuffle(order)
        for mid, i in enumerate(range(0, 10, 2), start=1):
            a, b = order[i], order[i + 1]
            sa = round(random.uniform(85, 165), 2)
            sb = round(random.uniform(85, 165), 2)
            for me, opp, mine, theirs in ((a, b, sa, sb), (b, a, sb, sa)):
                rows.append({
                    "owner": OWNERS[me], "roster_id": RIDS[me], "week": w,
                    "points": mine, "points_against": theirs,
                    "finalized": True, "avatar": "", "matchup_id": mid,
                    "opponent": OWNERS[opp],
                })
    return pd.DataFrame(rows)


def _ctx(monkeypatch, weeks_played: int, league_settings: dict,
         rules_record: dict | None = None):
    import dashboard_services.api as api

    def _fake_fetch_json(path, timeout=25, retries=3):
        if path == "/state/nfl":
            return {"season": "2026", "week": weeks_played + 1,
                    "leg": weeks_played + 1, "season_type": "regular",
                    "display_week": weeks_played + 1,
                    "season_start_date": "2026-09-03"}
        return {}

    monkeypatch.setattr(api, "fetch_json", _fake_fetch_json)

    import app as appmod
    monkeypatch.setattr(appmod, "daily_completed", _dt.date.today(),
                        raising=False)

    from dashboard_services.service import finalize_team_stats

    df_weekly = _build_df_weekly(weeks_played)
    team_stats = finalize_team_stats(
        df_weekly[df_weekly["finalized"]],
        {o: "" for o in OWNERS}, {}, [], weeks_played,
    )
    ctx = {
        "platform": "yahoo", "season": 2026,
        "league_id": "1307110", "resolved_league_id": "1307110",
        "df_weekly": df_weekly, "team_stats": team_stats,
        "roster_map": dict(zip(RIDS, OWNERS)),
        "rosters": [{"roster_id": r, "owner_id": f"u{r}", "players": []}
                    for r in RIDS],
        "users": [], "matchups_by_week": {},
        "league_settings": league_settings,
        # Yahoo's get_league is bare: no settings live on the league dict.
        "league": {"name": "Yahoo League", "settings": {}},
        "offseason_mode": False,
        "current_week": weeks_played + 1,
    }
    if rules_record is not None:
        ctx["_league_rules"] = rules_record
    return appmod, ctx


# The Yahoo globals shape (yahoo_api.get_league_globals) when the feed's
# playoff_start_week did not land: playoff_teams present, no start week.
YAHOO_GLOBALS_NO_PWS = {"playoff_teams": 4, "num_teams": 10, "type": 0,
                        "league_type": "redraft"}


def test_yahoo_path_tab_renders_when_settings_lack_playoff_start(monkeypatch):
    # 9 played; on the shared default (playoffs Week 15) 5 weeks remain,
    # inside the window -- the same answer the sim/scenarios give.
    appmod, ctx = _ctx(monkeypatch, 9, dict(YAHOO_GLOBALS_NO_PWS))
    spots, weeks = appmod._standings_playoff_params(ctx, ctx["team_stats"])
    assert (spots, weeks) == (4, 14)
    body = appmod.build_standings_body(ctx)
    assert 'data-tab="playoff-path"' in body
    assert 'id="playoffPathPanel"' in body
    assert 'data-platform="yahoo"' in body
    assert 'data-league-id="1307110"' in body


def test_yahoo_path_tab_renders_with_raw_start_week_alias(monkeypatch):
    settings = {"playoff_teams": 4, "playoff_start_week": 15}
    appmod, ctx = _ctx(monkeypatch, 9, settings)
    body = appmod.build_standings_body(ctx)
    assert 'data-tab="playoff-path"' in body


def test_yahoo_path_tab_absent_with_six_weeks_left(monkeypatch):
    # Real Week-16 playoff start (Yahoo's classic default): 6 weeks
    # remain after Week 9, outside the final-five window by design.
    settings = {**YAHOO_GLOBALS_NO_PWS, "playoff_week_start": 16}
    appmod, ctx = _ctx(monkeypatch, 9, settings)
    body = appmod.build_standings_body(ctx)
    assert 'data-tab="playoff-path"' not in body
    # ...and it opens once only five weeks remain.
    appmod, ctx = _ctx(monkeypatch, 10, settings)
    body = appmod.build_standings_body(ctx)
    assert 'data-tab="playoff-path"' in body


def test_yahoo_rules_record_wins_over_settings(monkeypatch):
    # The serve-time rules record is the source of truth: a Week-16
    # record keeps the tab closed even if settings carry Week 15.
    settings = {**YAHOO_GLOBALS_NO_PWS, "playoff_week_start": 15}
    appmod, ctx = _ctx(monkeypatch, 9, settings,
                       rules_record={"playoff_week_start": 16,
                                     "playoff_teams": 4})
    body = appmod.build_standings_body(ctx)
    assert 'data-tab="playoff-path"' not in body
