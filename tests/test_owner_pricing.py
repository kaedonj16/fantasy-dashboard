"""Frontier 8: owner pricing memory (utils/owner_pricing.py).

Free-text assistant preferences become pricing signals only on an
exact full-name match plus a recognized pricing phrase; anything
fuzzier is ignored so a misread preference never reprices a player.
"""
from utils.owner_pricing import owner_factor, parse_owner_pricing

NAMES = {"jahmyr gibbs": "g1", "justin jefferson": "j1",
         "bijan robinson": "b1", "puka nacua": "p1"}


def _pref(text, league_id=None):
    return {"text": text, "league_id": league_id}


def test_never_offer_and_pricing_phrases():
    prefs = [
        _pref("Remember: never trade Jahmyr Gibbs, he is my guy"),
        _pref("I value Justin Jefferson more than your model does"),
        _pref("Bijan Robinson is overrated, worth less to me"),
        _pref("I like watching football on Sundays"),
    ]
    out = parse_owner_pricing(prefs, NAMES)
    assert set(out["never_offer"]) == {"g1"}
    assert set(out["boost"]) == {"j1"}
    assert set(out["discount"]) == {"b1"}
    assert owner_factor(out, "j1")[0] == 1.10
    assert owner_factor(out, "b1")[0] == 0.90
    assert owner_factor(out, "p1") == (1.0, None)
    # The causing preference text rides along for the stamp.
    assert "never trade" in out["never_offer"]["g1"].lower()


def test_league_scoped_preferences_apply_only_to_their_league():
    prefs = [_pref("Puka Nacua is untouchable", league_id="L2")]
    assert parse_owner_pricing(prefs, NAMES, league_id="L1")[
        "never_offer"] == {}
    assert set(parse_owner_pricing(prefs, NAMES, league_id="L2")[
        "never_offer"]) == {"p1"}
    # Unscoped preferences apply everywhere.
    assert set(parse_owner_pricing(
        [_pref("Puka Nacua is untouchable")], NAMES, league_id="L1")[
        "never_offer"]) == {"p1"}


def test_no_phrase_no_signal():
    prefs = [_pref("Jahmyr Gibbs is my favorite player"),
             _pref("Thinking about trading Justin Jefferson")]
    out = parse_owner_pricing(prefs, NAMES)
    assert out == {"never_offer": {}, "boost": {}, "discount": {}}
