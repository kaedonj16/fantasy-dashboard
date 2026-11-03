"""KeepTradeCut as the dynasty blend's second market source.

Covers the three layers of the swap:
  1. Ingestion guardrails in external_values_scraper: a KTC page parse is
     validated before it may overwrite the last good CSV (player-count
     floor, blend coverage, plausible top value), and the blend-side
     loader treats a missing or >48h-old CSV as absent.
  2. Canonical name+team matching: suffix spellings ("Kenneth Walker
     III") and DP's "JAC" now match the site's spellings for BOTH
     sources (the old exact match silently missed 13 of DP's top 100).
  3. Blend math: KTC present -> FC+KTC at the old DP weights; KTC
     missing for a player -> DP fills that player; no fresh KTC at all
     -> byte-for-byte the old FC+DP blend.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")
pytest.importorskip("requests")  # external_values_scraper imports requests at load
pytest.importorskip("bs4")  # utils.utils (pulled in for DATA_DIR) imports BeautifulSoup
pytest.importorskip("flask")  # utils.utils also imports dashboard_services.api

from data_building.external_data import external_values_scraper as evs
from data_building import value_model_training as vmt


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

def _ktc_entry(name, pos, team, blend_1qb, blend_sf, i=0):
    return {
        "playerName": name,
        "position": pos,
        "team": team,
        "age": 25.0,
        "oneQBValues": {
            "value": blend_1qb, "vftValue": blend_1qb - 50,
            "blendValue": blend_1qb,
        },
        "superflexValues": {
            "value": blend_sf, "vftValue": blend_sf - 50,
            "blendValue": blend_sf,
        },
    }


def _ktc_html(n_players=450):
    entries = []
    for i in range(n_players):
        blend = 9900 - i * 10  # top stays inside the plausible band
        entries.append(_ktc_entry(
            f"Player {i:03d}", ["QB", "RB", "WR", "TE"][i % 4], "KC",
            blend, blend - 100, i))
    entries.append({"playerName": "2027 Early 1st", "position": "PICK",
                    "team": "", "oneQBValues": {}, "superflexValues": {}})
    return ('<html><body><script type="application/json" id="ktc-players">'
            + json.dumps(entries) + "</script></body></html>")


def _good_rows(n=450):
    return evs.parse_keeptradecut_players(_ktc_html(n))


# ---------------------------------------------------------------
# 1. Ingestion: parse, validation, last-good preservation, staleness
# ---------------------------------------------------------------

def test_parse_extracts_blend_values_and_drops_picks():
    rows = _good_rows()
    assert len(rows) == 450
    top = rows[0]
    assert top["name"] == "Player 000"
    assert top["value_1qb_blend"] == 9900.0
    assert top["value_sf_blend"] == 9800.0
    assert top["value_1qb_crowd"] == 9900.0
    assert all(r["position"] in ("QB", "RB", "WR", "TE") for r in rows)


def test_parse_rejects_page_without_dataset():
    with pytest.raises(ValueError):
        evs.parse_keeptradecut_players("<html><body>no data</body></html>")
    with pytest.raises(ValueError):
        evs.parse_keeptradecut_players("")


def test_write_good_rows_then_bad_parse_preserves_last_good(tmp_path):
    out = tmp_path / "keeptradecut_values.csv"
    assert evs.write_keeptradecut_values_csv(_good_rows(), out_csv=out)
    before = out.read_text()

    # Too few players: rejected, file untouched.
    few = _good_rows(25)
    assert not evs.write_keeptradecut_values_csv(few, out_csv=out)
    assert out.read_text() == before


def test_write_rejects_implausible_top_value(tmp_path):
    rows = _good_rows()
    for r in rows:
        r["value_1qb_blend"] = 4000.0  # top value far below the 0-9999 band
        r["value_sf_blend"] = 3900.0
    out = tmp_path / "keeptradecut_values.csv"
    assert not evs.write_keeptradecut_values_csv(rows, out_csv=out)
    assert not out.exists()


def test_write_rejects_thin_blend_coverage(tmp_path):
    rows = _good_rows()
    for r in rows[100:]:
        r["value_sf_blend"] = None  # SF blend populated for <25% of players
    out = tmp_path / "keeptradecut_values.csv"
    assert not evs.write_keeptradecut_values_csv(rows, out_csv=out)
    assert not out.exists()


def test_load_missing_and_stale_csv_count_as_absent(tmp_path):
    missing = tmp_path / "nope.csv"
    assert evs.load_keeptradecut_values(missing) is None

    out = tmp_path / "keeptradecut_values.csv"
    stale_stamp = datetime.now(timezone.utc) - timedelta(hours=49)
    assert evs.write_keeptradecut_values_csv(
        _good_rows(), out_csv=out, fetched_at=stale_stamp)
    assert evs.load_keeptradecut_values(out) is None

    fresh_stamp = datetime.now(timezone.utc) - timedelta(hours=2)
    assert evs.write_keeptradecut_values_csv(
        _good_rows(), out_csv=out, fetched_at=fresh_stamp)
    rows = evs.load_keeptradecut_values(out)
    assert rows is not None and len(rows) == 450
    assert rows[0]["fetched_at"]


def test_refresh_failure_keeps_last_good(tmp_path, monkeypatch):
    out = tmp_path / "keeptradecut_values.csv"
    assert evs.write_keeptradecut_values_csv(_good_rows(), out_csv=out)
    before = out.read_text()

    def boom():
        raise RuntimeError("site down")
    monkeypatch.setattr(evs, "fetch_keeptradecut_values", boom)
    assert not evs.refresh_keeptradecut_values(out_csv=out)
    assert out.read_text() == before


# ---------------------------------------------------------------
# 2. Canonical matching
# ---------------------------------------------------------------

@pytest.mark.parametrize("suffixed, plain", [
    ("Kenneth Walker III", "Kenneth Walker"),
    ("Patrick Mahomes II", "Patrick Mahomes"),
    ("Marvin Harrison Jr.", "Marvin Harrison"),
    ("James Cook III", "James Cook"),
    ("Brian Thomas Jr.", "Brian Thomas"),
])
def test_match_key_strips_name_suffixes(suffixed, plain):
    assert vmt.market_match_key(suffixed, "SEA") == vmt.market_match_key(plain, "SEA")


def test_match_key_canonicalizes_jac_to_jax():
    assert vmt.market_match_key("Trevor Lawrence", "JAC") == \
        vmt.market_match_key("Trevor Lawrence", "JAX")
    assert vmt.market_match_key("Parker Washington", "JAC") == \
        vmt.market_match_key("Parker Washington", "JAX")


@pytest.mark.parametrize("ktc_team, site_team", [
    ("KCC", "KC"), ("GBP", "GB"), ("NEP", "NE"), ("NOS", "NO"),
    ("SFO", "SF"), ("TBB", "TB"), ("LVR", "LV"),
])
def test_match_key_canonicalizes_ktc_team_spellings(ktc_team, site_team):
    # KTC's own team codes (verified in its live dataset) must land on
    # the site's codes, or whole teams silently miss the KTC source.
    assert vmt.market_match_key("Some Player", ktc_team) == \
        vmt.market_match_key("Some Player", site_team)


def test_map_lookup_falls_back_to_unique_name_only():
    m = {vmt.market_match_key("Kenneth Walker III", "KCC"): 8000.0}
    # Team registered differently on the site side: unique name saves it.
    assert vmt.market_map_lookup(m, vmt.market_match_key("Kenneth Walker", "SEA")) == 8000.0
    # Exact key still wins.
    assert vmt.market_map_lookup(m, vmt.market_match_key("Kenneth Walker III", "KC")) == 8000.0


def test_map_lookup_never_guesses_on_shared_names():
    m = {
        vmt.market_match_key("Josh Allen", "BUF"): 9000.0,
        vmt.market_match_key("Josh Allen", "JAX"): 500.0,
    }
    # Wrong team on a shared name: no guess.
    assert vmt.market_map_lookup(m, vmt.market_match_key("Josh Allen", "NYJ")) == 0.0
    assert vmt.market_map_lookup(m, vmt.market_match_key("Josh Allen", "BUF")) == 9000.0


def test_match_key_distinguishes_players_and_teams():
    assert vmt.market_match_key("James Cook", "BUF") != \
        vmt.market_match_key("James Cook", "DAL")
    assert vmt.market_match_key("James Cook", "BUF") != \
        vmt.market_match_key("James Conner", "BUF")


# ---------------------------------------------------------------
# 3. Blend math
# ---------------------------------------------------------------

def _legacy_blend(fc, dp, dp_weight=0.20):
    """Today's FC+DP blend, as the pre-swap code computed it."""
    s = t = 0.0
    if fc > 0:
        s += 0.40 * fc; t += 0.40
    if dp > 0:
        s += dp_weight * dp; t += dp_weight
    return s / t if t > 0 else 0.0


def test_blend_prefers_ktc_at_the_old_dp_weights():
    blended, sec = vmt.market_blend_value(800.0, 900.0, 300.0, dp_weight=0.20)
    assert sec == 900.0
    assert blended == pytest.approx((0.40 * 800.0 + 0.20 * 900.0) / 0.60)
    # And it is NOT the DP result: DP's lower read would drag it down.
    assert blended > _legacy_blend(800.0, 300.0)


def test_blend_falls_back_to_dp_per_player():
    blended, sec = vmt.market_blend_value(800.0, 0.0, 300.0, dp_weight=0.20)
    assert sec == 300.0
    assert blended == pytest.approx(_legacy_blend(800.0, 300.0))


def test_blend_without_ktc_is_identical_to_legacy():
    for fc, dp in [(800.0, 300.0), (999.9, 999.9), (0.0, 500.0), (500.0, 0.0)]:
        blended, _ = vmt.market_blend_value(fc, 0.0, dp, dp_weight=0.20)
        assert blended == pytest.approx(_legacy_blend(fc, dp))


def test_blend_dp_fill_keeps_staleness_weight():
    # KTC absent + stale DP: the fill carries the halved stale weight,
    # while a KTC read would have carried the full 0.20.
    blended, _ = vmt.market_blend_value(800.0, 0.0, 300.0, dp_weight=0.10)
    assert blended == pytest.approx(_legacy_blend(800.0, 300.0, dp_weight=0.10))
    ktc_blended, _ = vmt.market_blend_value(800.0, 300.0, 300.0, dp_weight=0.10)
    assert ktc_blended == pytest.approx(_legacy_blend(800.0, 300.0, dp_weight=0.20))


def test_blend_no_sources_is_zero():
    assert vmt.market_blend_value(0.0, 0.0, 0.0, dp_weight=0.20) == (0.0, 0.0)


def test_sf_blend_prefers_ktc_then_dp():
    ktc = vmt.sf_market_blend_value(900.0, 700.0, 200.0)
    assert ktc == pytest.approx((0.50 * 900.0 + 0.30 * 700.0) / 0.80)
    dp = vmt.sf_market_blend_value(900.0, 0.0, 200.0)
    assert dp == pytest.approx((0.50 * 900.0 + 0.30 * 200.0) / 0.80)
    assert vmt.sf_market_blend_value(0.0, 0.0, 0.0) == 0.0


def test_second_source_pick_order():
    assert vmt.second_source_pick(500.0, 300.0) == (500.0, True)
    assert vmt.second_source_pick(0.0, 300.0) == (300.0, False)
    assert vmt.second_source_pick(0.0, 0.0) == (0.0, False)


# ---------------------------------------------------------------
# 4. Second-source normalization: quantile map, not linear min-max
# ---------------------------------------------------------------
# Captured live 2026-11-03 (FC API + KTC dataset): the top-15 TE
# ladders of both sources. FC's scale decays steeply (top TE 635
# normalized, TE15 126); KTC's is flat (top TE blend 7986, TE14
# Ferguson 3652 — still 46% of its top). The old linear min-max of
# KTC to 0-999.9 read Ferguson at 350.8 vs FC's 126.1 for the same
# player, and the blend served him at TE8. Quantile-mapping KTC onto
# the FC distribution reads him at FC's rank-14 value instead.

_FC_TE_NORM_LIVE = [635.2, 590.7, 351.3, 314.3, 261.5, 254.8, 252.0,
                    235.2, 230.0, 185.9, 185.0, 169.6, 146.3, 129.0, 126.1]
_KTC_TE_BLEND_LIVE = [7986.0, 7420.0, 5724.0, 5613.0, 5217.0, 5216.0,
                      5068.0, 4836.0, 4721.0, 4358.0, 4254.0, 4053.0,
                      3723.0, 3652.0, 3455.0]  # index 13 = Terrance Ferguson


def test_quantile_mapped_ktc_read_for_ferguson_matches_fc_scale():
    from data_building.value_guardrails import quantile_map_to_reference
    mapped = quantile_map_to_reference(_KTC_TE_BLEND_LIVE, _FC_TE_NORM_LIVE)
    ferguson_read = mapped[13]
    assert ferguson_read == 129.0  # FC's value at KTC rank 14
    # The old linear stretch of the same raw value, for contrast: the
    # blend normalized over the FULL KTC pool (live min 252, max 9943),
    # which is what priced him at 350.8 on the served board.
    linear = (3652.0 - 252.0) / (9943.0 - 252.0) * 999.9
    assert linear == pytest.approx(350.8, abs=0.5)
    assert ferguson_read < linear / 2


def test_blend_with_quantile_mapped_ktc_lands_near_fc_for_ferguson():
    # FC prices Ferguson at 126.1 normalized; the quantile-mapped KTC
    # read is 129.0, so the blend stays ~127 — not the ~201 the linear
    # stretch produced (which the anchor then lifted to the served 225).
    blended, sec = vmt.market_blend_value(126.1, 129.0, 0.0, dp_weight=0.20)
    assert sec == 129.0
    assert blended == pytest.approx((0.40 * 126.1 + 0.20 * 129.0) / 0.60)
    assert blended < 140.0
    old_blend, _ = vmt.market_blend_value(126.1, 350.8, 0.0, dp_weight=0.20)
    assert old_blend == pytest.approx(201.0, abs=0.5)
