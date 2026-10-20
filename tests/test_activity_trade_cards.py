"""League Activity trade cards: mock rebuild + honest pricing contracts.

Covers the rebuild of html_trade in build_activity_body:
- players price BY PID first, with the canonical normalize_name() fallback
  on both sides (hyphen / apostrophe / suffix names must not price as 0);
- the collapsed summary shows BOTH teams' gets, draft picks included;
- the winner pill shows the single net (never the doubled abs(netA-netB));
- an unvalued asset makes the trade Not graded (no verdict on bad math);
- the consolidation adjustment is printed, and picks count toward the
  asset-count delta that drives it.
"""
from __future__ import annotations

import pytest

pytest.importorskip("flask")
pd = pytest.importorskip("pandas")


PICK_VALUES = {
    "2027_2_early": 137.7,
    "2027_2_mid": 120.0,
    "2027_2_late": 100.0,
    "2027_2": 110.0,
}


def _row(pid, name, search_name, pos, team, value, label):
    return {
        "id": pid, "name": name, "search_name": search_name,
        "position": pos, "team": team, "value": value,
        "pos_rank_label": label,
    }


BASE_TABLE = [
    _row("8146", "Garrett Wilson", "garrett wilson", "WR", "NYJ", 517.4, "WR10"),
    _row("11631", "Brian Thomas Jr.", "brian thomas", "WR", "JAX", 167.6, "WR43"),
    _row("6806", "J.K. Dobbins", "jk dobbins", "RB", "DEN", 120.9, "RB42"),
    # Canonical (DB) search_name form: hyphens -> spaces.
    _row("7547", "Amon-Ra St. Brown", "amon ra st brown", "WR", "DET", 828.4, "WR4"),
    _row("7564", "Ja'Marr Chase", "jamarr chase", "WR", "CIN", 980.4, "WR1"),
]


def _player(pid, name, pos, team):
    p = {"name": name, "pos": pos, "team": team}
    if pid is not None:
        p["pid"] = pid
    return p


def _team(rid, name, gets, sends):
    return {
        "roster_id": rid, "name": name, "avatar": "",
        "gets": gets, "sends": sends,
    }


@pytest.fixture
def activity_env(monkeypatch):
    import app as appmod

    monkeypatch.setattr(appmod, "ensure_activity_bits", lambda ctx: None)
    monkeypatch.setattr(appmod, "get_users", lambda *a, **k: [])
    monkeypatch.setattr(
        appmod, "load_pick_value_table", lambda: dict(PICK_VALUES)
    )
    monkeypatch.setattr(
        appmod, "resolve_exact_pick_slot", lambda *a, **k: None
    )
    return appmod


def _render(activity_env, trades, table):
    from dashboard_services.pages.activity_page import build_activity_body

    rows = [
        {
            "kind": "trade", "week": 1,
            "ts": pd.Timestamp("2026-08-12", tz="UTC"),
            "data": {"teams": teams, "draft_picks": picks},
        }
        for teams, picks in trades
    ]
    ctx = {
        "league_id": "L1",
        "resolved_league_id": "L1",
        "platform": "sleeper",
        "season": 2026,
        "activity_df": pd.DataFrame(rows),
        "injury_df": pd.DataFrame(),
        "standings_map": {1: 1, 2: 2},
        "model_value_table": table,
        "scoring_settings": {},
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
        "rosters": [{"roster_id": i} for i in range(1, 11)],
        "users": [],
    }
    return build_activity_body(ctx)


def _wilson_trade():
    wilson = _player("8146", "Garrett Wilson", "WR", "NYJ")
    thomas = _player("11631", "Brian Thomas Jr.", "WR", "JAX")
    dobbins = _player("6806", "J.K. Dobbins", "RB", "DEN")
    teams = [
        _team(1, "I Chase Brown Kids", [wilson], [thomas, dobbins]),
        _team(2, "the foolies", [thomas, dobbins], [wilson]),
    ]
    return teams, []


def test_hyphenated_player_priced_by_pid(activity_env):
    """Amon-Ra's table search_name is the canonical 'amon ra st brown';
    the pid lookup must price him anyway."""
    amon = _player("7547", "Amon-Ra St. Brown", "WR", "DET")
    wilson = _player("8146", "Garrett Wilson", "WR", "NYJ")
    teams = [
        _team(1, "Rebuild from Hell", [wilson], [amon]),
        _team(2, "jmoneybags91", [amon], [wilson]),
    ]
    html = _render(activity_env, [(teams, [])], BASE_TABLE)
    assert "828.4" in html
    assert "Not graded" not in html
    assert "<span class='act-pill-win'>" in html


def test_special_names_priced_by_normalized_fallback(activity_env):
    """Without a pid, normalize_name on both sides still prices a
    hyphenated name and a suffixed name (old one-off normalization
    missed both)."""
    amon = _player(None, "Amon-Ra St. Brown", "WR", "DET")
    thomas = _player(None, "Brian Thomas Jr.", "WR", "JAX")
    teams = [
        _team(1, "Team One", [amon], [thomas]),
        _team(2, "Team Two", [thomas], [amon]),
    ]
    html = _render(activity_env, [(teams, [])], BASE_TABLE)
    assert "828.4" in html
    assert "167.6" in html
    assert "Not graded" not in html


def test_winner_pill_shows_single_net_not_doubled(activity_env):
    html = _render(activity_env, [_wilson_trade()], BASE_TABLE)
    # Net is +289 for the winner; the old summary printed the doubled
    # abs(netA - netB) as a delta.
    assert "I Chase Brown Kids +289" in html
    assert "<span class='act-swing'>" not in html
    assert "Δ 579" not in html and "Δ 578" not in html


def test_adjustment_footer_printed(activity_env):
    html = _render(activity_env, [_wilson_trade()], BASE_TABLE)
    assert "Received total" in html
    assert "Consolidation adjustment" in html
    assert "Effective total" in html
    assert "Net +289 to I Chase Brown Kids" in html
    assert "Strong win" in html
    # The old duplicated layout / mislabeled total is gone.
    assert "Total Value" not in html


def test_pick_chip_in_summary_and_pick_counts_for_adjustment(activity_env):
    wilson = _player("8146", "Garrett Wilson", "WR", "NYJ")
    thomas = _player("11631", "Brian Thomas Jr.", "WR", "JAX")
    pick = {
        "season": 2027, "round": 2,
        "owner_id": 1, "previous_owner_id": 2, "roster_id": 2,
    }
    teams = [
        _team(1, "Team One", [wilson], [thomas]),
        _team(2, "Team Two", [thomas], [wilson]),
    ]
    html = _render(activity_env, [(teams, [pick])], BASE_TABLE)
    # The pick shows as a chip in the collapsed summary...
    assert "act-chip-pick" in html
    assert "2027 2nd" in html
    # ...and in the expanded received column at its price.
    assert "137.7" in html
    # Players alone are 1-for-1 (no adjustment); the pick makes it
    # 2-for-1, so the consolidation adjustment must appear.
    assert "Consolidation adjustment" in html


def test_unvalued_asset_makes_trade_not_graded(activity_env):
    odunze = _player("9991", "Rome Odunze", "WR", "CHI")
    kraft = _player("9992", "Tucker Kraft", "TE", "GB")
    amon = _player("7547", "Amon-Ra St. Brown", "WR", "DET")
    table = [
        _row("9991", "Rome Odunze", "rome odunze", "WR", "CHI", 285.7, "WR27"),
        _row("9992", "Tucker Kraft", "tucker kraft", "TE", "GB", 267.7, "TE6"),
        # Amon-Ra deliberately absent from the value table. The valued side
        # receives 2-for-1, so an adjustment WOULD print if it were graded.
    ]
    teams = [
        _team(1, "Rebuild from Hell", [odunze, kraft], [amon]),
        _team(2, "jmoneybags91", [amon], [odunze, kraft]),
    ]
    html = _render(activity_env, [(teams, [])], table)
    assert "Not graded" in html
    assert "Can’t grade — Amon-Ra St. Brown has no current value" in html
    assert "No current value" in html
    assert "Strong win" not in html
    assert "<span class='act-pill-win'>" not in html
    # Mock fidelity: an ungraded trade prints no Effective total / adjustment
    # on either side — the valued side shows only its Received total.
    assert "Effective total" not in html
    assert "Consolidation adjustment" not in html
