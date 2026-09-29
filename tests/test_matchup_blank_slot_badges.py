"""Regression test: blank lineup slots must not misalign the slot badges.

ESPN returns None (or empty) for an unfilled starter slot, e.g. a blank RB.
The old code filtered falsy entries out of the starters list, shifting every
later player up a row. The centre-rail badge reads the slot from
roster_positions[row_idx], so every badge below the blank slot read one slot
too high (WR showed RB, TE showed WR, K showed D/ST, ...).
"""
import dashboard_services.matchups as matchups


def _p(pid, name, pos, pts):
    return {"pid": pid, "name": name, "pos": pos, "nfl": "FA", "pts": pts}


def _team(starters):
    return {
        "name": "Test Team",
        "roster_id": "1",
        "starters": starters,
        "bench": [],
        "pts_total": 100.0,
    }


def test_normalize_starter_slots_preserves_blank():
    # ESPN shape: None for the blank RB slot at index 2.
    raw = ["qb1", "rb1", None, "wr1", "wr2", "te1", "k1", "dst1"]
    normed = matchups._normalize_starter_slots(raw)
    assert normed == ["qb1", "rb1", None, "wr1", "wr2", "te1", "k1", "dst1"]
    assert len(normed) == len(raw)


def test_normalize_starter_slots_fleaflicker_zero():
    # Fleaflicker uses "0" for empty/unresolved slots.
    assert matchups._normalize_starter_slots(["a", "0", "b"]) == ["a", None, "b"]
    assert matchups._normalize_starter_slots(["a", "", "b"]) == ["a", None, "b"]
    assert matchups._normalize_starter_slots(None) == []


def test_blank_slot_keeps_badge_alignment():
    # Left team has a blank RB slot at index 2 (ESPN shape).
    left = _team([
        _p("qb1", "QB One", "QB", 18.6),
        _p("rb1", "RB One", "RB", 20.4),
        None,  # blank RB slot
        _p("wr1", "WR One", "WR", 15.9),
        _p("wr2", "WR Two", "WR", 8.5),
        _p("te1", "TE One", "TE", 9.5),
        _p("k1", "K One", "K", 8.9),
        _p("dst1", "D/ST One", "DEF", 7.6),
    ])
    right = _team([
        _p("qb2", "QB Two", "QB", 10.0),
        _p("rb2", "RB Two", "RB", 10.0),
        _p("rb3", "RB Three", "RB", 10.0),
        _p("wr3", "WR Three", "WR", 10.0),
        _p("wr4", "WR Four", "WR", 10.0),
        _p("te2", "TE Two", "TE", 10.0),
        _p("k2", "K Two", "K", 10.0),
        _p("dst2", "D/ST Two", "DEF", 10.0),
    ])
    m = {"left": left, "right": right}
    html = matchups.render_matchup_slide(
        season="2026",
        m=m,
        w=4,
        proj_week=4,
        status_by_pid={},
        projections={},
        players={},
        teams={},
        team_game_lookup={},
        roster_positions=["QB", "RB", "RB", "WR", "WR", "TE", "K", "D/ST", "BN", "BN"],
    )
    # The blank RB row keeps the RB badge; everyone below stays aligned.
    assert html.count("pos-badge RB") == 2  # Henry's row + the blank RB row
    assert "pos-badge WR" in html
    assert "pos-badge TE" in html
    assert "pos-badge K" in html
    assert "pos-badge DEF" in html
    # WR One must sit in a WR-badged row, not an RB-badged one. The badge
    # renders after the player cell within each row.
    wr1_idx = html.index("WR One")
    rb_badges = [i for i in range(len(html)) if html.startswith("pos-badge RB", i)]
    wr_badges = [i for i in range(len(html)) if html.startswith("pos-badge WR", i)]
    next_badges = [i for i in rb_badges + wr_badges if i > wr1_idx]
    assert next_badges, "no badge found after WR One"
    assert min(next_badges) in wr_badges
