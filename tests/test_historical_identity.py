import pandas as pd

from dashboard_services.historical_identity import canonicalize_weekly_owners, roster_id_for_owner
from dashboard_services.pages.graphs_page import build_career_graphs_ctx


def _ctx(league, season, teams):
    rosters = [{"roster_id": rid, "owner_id": uid} for rid, uid, _ in teams]
    users = [{"user_id": uid, "display_name": name} for _, uid, name in teams]
    rows = []
    for rid, _uid, name in teams:
        rows.append({"roster_id": rid, "owner": name, "week": 1, "matchup_id": 1,
                     "points": 100 + rid, "points_against": 90, "finalized": True})
    return {"league_id": league, "resolved_league_id": league, "season": season,
            "league": {"settings": {"playoff_week_start": 15}}, "rosters": rosters,
            "users": users, "roster_map": {str(r): n for r, _, n in teams},
            "df_weekly": pd.DataFrame(rows)}


def test_canonical_owner_survives_renames_and_does_not_merge_equal_names(monkeypatch):
    old = _ctx("old", 2024, [(1, "owner-1", "Team A"), (2, "owner-2", "Same")])
    new = _ctx("new", 2025, [(8, "owner-1", "Team B"), (9, "owner-3", "Same")])
    contexts = {("old", 2024): old, ("new", 2025): new}
    monkeypatch.setattr("dashboard_services.api.resolve_league_id_for_season",
                        lambda *args, **kwargs: "old" if (kwargs.get("target_season") or args[-1]) == 2024 else "new")

    result = build_career_graphs_ctx(
        "sleeper", "new", 2025, [2024, 2025],
        lambda _p, lid, season: contexts[(lid, season)],
    )

    rows = result["team_stats"].set_index("owner_key")
    assert set(rows.index) == {"owner-1", "owner-2", "owner-3"}
    assert rows.loc["owner-1", "owner"] == "Team B"
    assert rows.loc["owner-1", "PF"] == 202
    assert len(result["season_pf_df"].query("owner_key == 'owner-1'")) == 2


def test_season_resolution_uses_owner_id_not_display_name():
    ctx = _ctx("league", 2025, [(4, "alpha", "Same"), (7, "beta", "Same")])
    df = canonicalize_weekly_owners(ctx["df_weekly"], ctx)
    assert set(df["owner_key"]) == {"alpha", "beta"}
    assert roster_id_for_owner(ctx, "beta") == "7"
