"""Playoff Picture seed pills must match the Standings seeds.

The bracket's seed pills used to come from seed_top6_from_team_stats
(Wins -> PF -> PA, top 6 only), so on the 92916 screenshot Fearless
wore #1 in the Picture while the standings table seeded him #3
(division winners first, division-record tiebreak). When the
standings panels supply their seed map, render_power_and_playoffs
must use it for the pills; without one, the old Wins/PF pills stay.
"""
import re

import pytest

pytest.importorskip("flask")
pd = pytest.importorskip("pandas")

import app as appmod
from utils.standings import build_full_bracket

# rid -> (owner, wins, PF) in the 92916 shape.
TEAMS = {
    21: ("Triple B II", 7, 1053.7),
    31: ("dolphins fan", 7, 1043.6),
    11: ("Fearless And Ready To Rumble", 7, 1183.4),
    22: ("Spartyhater", 5, 997.8),
    12: ("Password is Taco", 5, 1094.0),
    23: ("FLASHOVERS", 5, 1078.7),
    13: ("American Dream", 5, 944.95),
    32: ("Filler Five", 4, 850.0),
}
# Standings seed order: division winners first, div-record tiebreak.
STANDINGS_SEEDS = [21, 31, 11, 22, 12, 23, 13, 32]
ROSTER_MAP = {str(rid): name for rid, (name, _w, _pf) in TEAMS.items()}


def _team_stats():
    return pd.DataFrame([
        {"owner": name, "avatar": "", "Wins": w, "Losses": 9 - w,
         "G": 9, "Ties": 0, "PF": pf, "PA": 900.0, "PowerScore": 0.5}
        for _rid, (name, w, pf) in TEAMS.items()
    ])


def _pills(html):
    """{team_name: seed} from the bracket's named team rows."""
    out = {}
    for chunk in re.split(r"<div class='pb-trow'>", html)[1:]:
        seed = re.search(r"pb-seed[^']*'>(\d+)<", chunk)
        name = re.search(r"data-team-name='([^']+)'", chunk)
        if seed and name:
            out[name.group(1)] = int(seed.group(1))
    return out


def _render(**kw):
    return appmod.render_power_and_playoffs(
        _team_stats(), ROSTER_MAP, "92916", "fleaflicker", 2026,
        bracket_override=build_full_bracket(STANDINGS_SEEDS),
        playoff_week_start=15, **kw,
    )


def test_pills_match_standings_seeds_when_map_supplied():
    standings_map = {str(rid): i + 1 for i, rid in enumerate(STANDINGS_SEEDS)}
    pills = _pills(_render(standings_seed_map=standings_map))
    assert pills == {
        name: standings_map[str(rid)]
        for rid, (name, _w, _pf) in TEAMS.items()
    }
    # The two seeds the screenshot got wrong:
    assert pills["Triple B II"] == 1
    assert pills["Fearless And Ready To Rumble"] == 3
    assert pills["Spartyhater"] == 4


def test_pills_keep_wins_pf_order_without_standings_map():
    # No standings map (every pre-existing caller): the Wins/PF pill
    # order is unchanged, capped at the top 6.
    pills = _pills(_render())
    assert pills["Fearless And Ready To Rumble"] == 1
    assert pills["Triple B II"] == 2
    assert "American Dream" not in pills
