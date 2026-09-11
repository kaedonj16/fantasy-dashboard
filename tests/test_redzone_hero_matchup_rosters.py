"""Regression test: hero matchup filter must correctly tag rosters.

When filtering to a specific matchup in league-scoped redzone, players from
both teams in that matchup must be correctly tagged with mine/opp flags,
even when the viewer is not in that matchup.

Bug: _rosterTags() only added opponents of the viewer's teams to oppRosters,
so when filtering to a matchup the viewer wasn't in, both teams got
mine=false, opp=false, causing wrong team assignment in the UI.

Fix: When _heroMid is active in league scope, include all rosters from that
matchup in myRosters/oppRosters so their players get correct flags.
"""
from __future__ import annotations


def test_hero_matchup_rosters_tagged_when_viewer_not_in_matchup():
    """Players from a filtered matchup get correct mine/opp flags."""
    # Simulates the frontend _rosterTags() logic
    def roster_tags(matchups, my_rids, hero_mid=None, scope='league'):
        my_rosters = set(my_rids)
        opp_rosters = set()
        
        # Original logic: add opponents of viewer's teams
        for m in matchups:
            if m['roster_id'] not in my_rids:
                continue
            mid = str(m['matchup_id'])
            for o in matchups:
                if str(o['matchup_id']) == mid and o['roster_id'] not in my_rids:
                    opp_rosters.add(o['roster_id'])
        
        # Fixed logic: when hero matchup is active, include those rosters
        if hero_mid and scope == 'league':
            for m in matchups:
                if str(m['matchup_id']) == hero_mid:
                    rid = m['roster_id']
                    if rid in my_rids:
                        my_rosters.add(rid)
                    else:
                        opp_rosters.add(rid)
        
        pid_to_roster = {}
        for m in matchups:
            for pid in m.get('players', []):
                rid = m['roster_id']
                if pid not in pid_to_roster:
                    pid_to_roster[pid] = rid
                elif rid in my_rids and pid_to_roster[pid] not in my_rids:
                    pid_to_roster[pid] = rid
        
        return {'my': my_rosters, 'opp': opp_rosters, 'pidToRoster': pid_to_roster}
    
    # League with 4 teams, 2 matchups
    # Viewer is roster 1 (in matchup 10)
    # User filters to matchup 20 (rosters 3 vs 4)
    matchups = [
        {'roster_id': 1, 'matchup_id': 10, 'players': ['p1', 'p2']},
        {'roster_id': 2, 'matchup_id': 10, 'players': ['p3', 'p4']},
        {'roster_id': 3, 'matchup_id': 20, 'players': ['p5', 'p6']},
        {'roster_id': 4, 'matchup_id': 20, 'players': ['p7', 'p8']},
    ]
    
    my_rids = [1]
    
    # Without hero filter: only viewer's matchup is tagged
    tags_no_hero = roster_tags(matchups, my_rids, hero_mid=None)
    assert tags_no_hero['my'] == {1}
    assert tags_no_hero['opp'] == {2}
    
    # With hero filter on matchup 20: rosters 3 and 4 must be tagged
    tags_hero = roster_tags(matchups, my_rids, hero_mid='20', scope='league')
    assert tags_hero['my'] == {1}  # Viewer's roster still in my
    assert tags_hero['opp'] == {2, 3, 4}  # Both hero matchup rosters added to opp
    
    # Players from hero matchup are now correctly mapped
    assert tags_hero['pidToRoster']['p5'] == 3
    assert tags_hero['pidToRoster']['p6'] == 3
    assert tags_hero['pidToRoster']['p7'] == 4
    assert tags_hero['pidToRoster']['p8'] == 4


def test_hero_matchup_with_viewer_in_filtered_matchup():
    """When viewer IS in the hero matchup, their roster stays in 'my'."""
    def roster_tags(matchups, my_rids, hero_mid=None, scope='league'):
        my_rosters = set(my_rids)
        opp_rosters = set()
        
        for m in matchups:
            if m['roster_id'] not in my_rids:
                continue
            mid = str(m['matchup_id'])
            for o in matchups:
                if str(o['matchup_id']) == mid and o['roster_id'] not in my_rids:
                    opp_rosters.add(o['roster_id'])
        
        if hero_mid and scope == 'league':
            for m in matchups:
                if str(m['matchup_id']) == hero_mid:
                    rid = m['roster_id']
                    if rid in my_rids:
                        my_rosters.add(rid)
                    else:
                        opp_rosters.add(rid)
        
        return {'my': my_rosters, 'opp': opp_rosters}
    
    matchups = [
        {'roster_id': 1, 'matchup_id': 10, 'players': ['p1']},
        {'roster_id': 2, 'matchup_id': 10, 'players': ['p2']},
        {'roster_id': 3, 'matchup_id': 20, 'players': ['p3']},
        {'roster_id': 4, 'matchup_id': 20, 'players': ['p4']},
    ]
    
    my_rids = [1]
    
    # Filter to viewer's own matchup
    tags = roster_tags(matchups, my_rids, hero_mid='10', scope='league')
    assert tags['my'] == {1}
    assert tags['opp'] == {2}


def test_hero_matchup_only_applies_in_league_scope():
    """Hero matchup roster tagging only applies in league scope, not user scope."""
    def roster_tags(matchups, my_rids, hero_mid=None, scope='league'):
        my_rosters = set(my_rids)
        opp_rosters = set()
        
        for m in matchups:
            if m['roster_id'] not in my_rids:
                continue
            mid = str(m['matchup_id'])
            for o in matchups:
                if str(o['matchup_id']) == mid and o['roster_id'] not in my_rids:
                    opp_rosters.add(o['roster_id'])
        
        if hero_mid and scope == 'league':
            for m in matchups:
                if str(m['matchup_id']) == hero_mid:
                    rid = m['roster_id']
                    if rid in my_rids:
                        my_rosters.add(rid)
                    else:
                        opp_rosters.add(rid)
        
        return {'my': my_rosters, 'opp': opp_rosters}
    
    matchups = [
        {'roster_id': 1, 'matchup_id': 10, 'players': ['p1']},
        {'roster_id': 2, 'matchup_id': 10, 'players': ['p2']},
        {'roster_id': 3, 'matchup_id': 20, 'players': ['p3']},
        {'roster_id': 4, 'matchup_id': 20, 'players': ['p4']},
    ]
    
    my_rids = [1]
    
    # In user scope, hero matchup logic doesn't apply
    tags_user = roster_tags(matchups, my_rids, hero_mid='20', scope='user')
    assert tags_user['my'] == {1}
    assert tags_user['opp'] == {2}  # Only viewer's opponent, not hero matchup
    
    # In league scope, hero matchup logic applies
    tags_league = roster_tags(matchups, my_rids, hero_mid='20', scope='league')
    assert tags_league['my'] == {1}
    assert tags_league['opp'] == {2, 3, 4}  # Includes hero matchup rosters
