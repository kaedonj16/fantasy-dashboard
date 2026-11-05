"""Redraft tier thresholds: computed on redraft values, not dynasty ones.

Regression for Kaedon's 2026-11-05 Trade Calculator screenshot: the Player
Values panel in Redraft + SF mode showed a T1 of EIGHT players (JSN 1081.0
through Kenneth Walker III 836.9, span 244.1) because the frontend judged
redraft values against the payload's dynasty tier thresholds (the dynasty
SF/8 first threshold, 830.7, landed between Walker and McCaffrey 813.5).

The (position, redraft_value_sf, redraft_value_1qb) rows below are the real
production values captured from /api/league-players on 2026-11-05 (every
skill player with a redraft value >= 5), so the assertions pin the tiers
her screenshot should have shown.

Pure logic - no app / DB import - so these run anywhere pytest does.
"""
from pathlib import Path

from utils.tier_thresholds import (
    FALLBACK_THRESHOLDS,
    compute_redraft_tier_thresholds,
)

# (position, redraft_value_sf, redraft_value_1qb), sorted by SF value desc.
_PROD_ROWS = [
    ("WR",1080.95,1022.43), ("RB",1071.46,1182.98), ("QB",988.55,646.5), ("RB",976.36,1031.21),
    ("WR",879.99,802.62), ("WR",879.53,874.6), ("WR",853.62,828.27), ("RB",836.85,888.28),
    ("RB",813.49,832.68), ("WR",780.87,824.63), ("TE",692.16,633.36), ("RB",685.89,789.32),
    ("RB",674.87,707.41), ("TE",644.22,588.33), ("QB",601.63,314.98), ("WR",560.06,647.23),
    ("WR",558.66,598.29), ("RB",536.77,528.75), ("WR",536.59,558.14), ("WR",527.6,527.78),
    ("WR",500.98,500.19), ("RB",497.7,573.39), ("RB",437.0,477.83), ("RB",436.31,526.01),
    ("RB",433.63,474.34), ("QB",430.85,212.84), ("RB",430.81,440.33), ("RB",430.75,471.15),
    ("QB",430.69,250.43), ("WR",424.51,413.02), ("QB",412.96,252.75), ("RB",412.4,429.15),
    ("RB",412.18,468.36), ("QB",406.37,268.91), ("WR",405.39,450.01), ("WR",399.92,475.37),
    ("WR",369.39,388.75), ("WR",362.13,405.31), ("RB",360.08,418.16), ("RB",354.89,352.94),
    ("RB",340.08,359.08), ("WR",329.26,288.86), ("QB",326.02,197.89), ("WR",321.49,333.58),
    ("RB",319.84,322.04), ("TE",318.14,361.47), ("QB",295.9,178.6), ("QB",295.06,165.39),
    ("WR",294.96,391.88), ("QB",293.23,174.23), ("WR",290.65,345.59), ("QB",286.62,164.69),
    ("WR",281.6,274.97), ("WR",268.64,314.58), ("WR",266.6,337.89), ("TE",241.93,220.17),
    ("RB",236.81,197.38), ("WR",234.87,235.91), ("RB",232.41,215.12), ("TE",231.43,237.28),
    ("TE",230.02,225.65), ("TE",225.06,195.7), ("WR",214.74,226.94), ("RB",213.6,215.8),
    ("RB",209.6,211.94), ("WR",206.15,242.7), ("RB",196.43,218.37), ("WR",193.55,223.5),
    ("RB",191.28,203.99), ("RB",185.8,217.88), ("WR",179.86,179.49), ("TE",176.89,215.0),
    ("RB",166.95,169.52), ("WR",166.91,186.75), ("QB",162.87,91.99), ("QB",157.66,71.67),
    ("TE",155.01,159.52), ("QB",153.63,69.83), ("WR",146.98,145.86), ("WR",144.44,173.88),
    ("WR",140.15,165.04), ("QB",136.51,90.16), ("RB",135.16,188.58), ("WR",130.65,135.29),
    ("TE",129.21,115.08), ("RB",128.83,124.07), ("RB",121.91,106.01), ("QB",121.79,69.49),
    ("WR",119.92,130.01), ("RB",118.8,144.36), ("WR",116.64,106.78), ("RB",113.0,137.3),
    ("RB",106.18,92.33), ("RB",104.01,93.95), ("WR",101.34,100.86), ("RB",101.28,111.75),
    ("RB",99.67,111.64), ("RB",98.96,120.25), ("RB",93.07,104.97), ("RB",92.06,80.05),
    ("WR",88.06,100.03), ("TE",86.11,74.88), ("RB",82.4,95.56), ("WR",82.1,81.31),
    ("TE",80.3,113.59), ("WR",79.22,94.63), ("WR",78.07,78.78), ("WR",75.93,66.03),
    ("RB",74.82,78.1), ("WR",74.23,64.55), ("WR",73.76,81.54), ("WR",73.7,64.09),
    ("RB",70.4,61.22), ("WR",68.02,82.69), ("TE",67.25,68.1), ("WR",64.45,56.05),
    ("TE",64.33,55.94), ("WR",64.19,55.81), ("TE",62.57,72.36), ("WR",54.55,66.26),
    ("WR",53.27,53.52), ("WR",49.63,50.42), ("RB",49.23,47.43), ("RB",46.37,40.32),
    ("WR",46.32,49.05), ("WR",44.77,38.93), ("TE",43.99,38.25), ("RB",43.2,51.46),
    ("QB",41.98,33.19), ("WR",40.85,45.36), ("QB",40.45,23.89), ("RB",39.49,34.34),
    ("WR",38.87,41.92), ("QB",37.39,17.0), ("TE",37.38,32.51), ("QB",36.93,19.06),
    ("WR",36.39,41.69), ("WR",36.12,41.58), ("QB",35.66,22.63), ("QB",34.54,19.41),
    ("RB",33.78,35.61), ("RB",33.77,34.57), ("QB",33.51,19.64), ("QB",33.0,15.51),
    ("WR",29.45,33.8), ("WR",29.35,31.12), ("QB",28.56,12.98), ("RB",27.45,45.36),
    ("QB",26.71,21.7), ("QB",25.33,18.72), ("QB",24.39,11.61), ("WR",24.24,28.71),
    ("WR",24.02,40.32), ("RB",23.25,20.22), ("WR",22.65,21.36), ("WR",22.39,23.89),
    ("RB",21.14,18.38), ("TE",21.03,23.54), ("QB",20.81,14.93), ("QB",20.03,11.37),
    ("RB",19.23,19.99), ("WR",18.71,22.74), ("WR",18.49,16.08), ("TE",17.91,18.49),
    ("RB",17.91,18.72), ("TE",14.89,19.87), ("WR",14.52,13.89), ("QB",13.96,7.35),
    ("TE",13.44,16.31), ("TE",12.83,19.29), ("WR",8.04,9.31), ("RB",7.48,7.92),
    ("QB",6.81,3.91), ("QB",6.72,4.32), ("TE",5.81,5.05), ("TE",4.96,5.05),
    ("TE",4.47,5.17), ("RB",1.89,7.7), ("RB",1.78,24.12),
]


def _table():
    return [
        {"position": pos, "redraft_value_sf": sf, "redraft_value_1qb": qb}
        for pos, sf, qb in _PROD_ROWS
    ]


def _tier1(values, thresholds):
    return [v for v in values if v >= thresholds[0]]


def test_shape_one_set_per_league_type_no_size_dimension():
    out = compute_redraft_tier_thresholds(_table())
    assert set(out) == {"1qb", "sf"}
    for thresholds in out.values():
        assert len(thresholds) >= 2
        assert thresholds == sorted(thresholds, reverse=True)
        assert thresholds != FALLBACK_THRESHOLDS


def test_redraft_sf_t1_is_four_not_eight():
    # The screenshot case: with thresholds derived from the redraft SF
    # values themselves, the first boundary lands in the 96.4-point gap
    # between Bijan Robinson (976.36) and CeeDee Lamb (879.99).
    table = _table()
    out = compute_redraft_tier_thresholds(table)["sf"]
    assert 879.99 < out[0] < 976.36
    t1 = _tier1([r["redraft_value_sf"] for r in table], out)
    assert len(t1) == 4
    # The tier function's own MAX_SPAN rule holds for the new T1.
    assert max(t1) - min(t1) <= 220.0


def test_redraft_1qb_t1_is_three():
    table = _table()
    out = compute_redraft_tier_thresholds(table)["1qb"]
    # T1 = Gibbs 1182.98, Bijan 1031.21, JSN 1022.43; next is Walker 888.28.
    assert 888.28 < out[0] < 1022.43
    t1 = _tier1([r["redraft_value_1qb"] for r in table], out)
    assert len(t1) == 3


def test_dynasty_columns_do_not_move_redraft_thresholds():
    base = compute_redraft_tier_thresholds(_table())
    poisoned = [
        dict(row, value=5000.0, sf_value=5000.0, value_8=1.0, sf_value_12=1.0)
        for row in _table()
    ]
    assert compute_redraft_tier_thresholds(poisoned) == base


def test_missing_redraft_values_fall_back():
    table = [{"position": "RB", "value": 900.0, "sf_value": 950.0}
             for _ in range(60)]
    out = compute_redraft_tier_thresholds(table)
    assert out == {"1qb": FALLBACK_THRESHOLDS, "sf": FALLBACK_THRESHOLDS}


def test_string_values_from_payload_are_tolerated():
    table = [
        {"position": pos, "redraft_value_sf": str(sf),
         "redraft_value_1qb": str(qb)}
        for pos, sf, qb in _PROD_ROWS
    ]
    assert compute_redraft_tier_thresholds(table) == \
        compute_redraft_tier_thresholds(_table())


def test_frontends_use_redraft_thresholds_in_redraft_mode():
    root = Path(__file__).parents[1]
    rankings = (root / "static" / "rankings.js").read_text(encoding="utf-8")
    assert "resp.tier_thresholds_redraft" in rankings
    assert "prTierThresholdsRedraft[lt]" in rankings
    app_js = (root / "static" / "app.js").read_text(encoding="utf-8")
    assert "data.tier_thresholds_redraft" in app_js
    assert "_tierThresholdsRedraft[lt]" in app_js
