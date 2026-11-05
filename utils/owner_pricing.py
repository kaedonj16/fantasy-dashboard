"""Owner pricing memory (Frontier 8).

The assistant remembers what the owner TELLS it ("remember: never
trade Gibbs", "I value Chase higher than your model"). Those
statements are free text in the assistant preference store; this
module parses them conservatively into pricing signals the trade
suggestions engine can honor:

* ``never_offer`` — a hard rule: no suggestion may spend the
  player. Phrasings: "never trade/offer/send", "don't trade",
  "untouchable", "not trading".
* ``boost`` / ``discount`` — the owner prices the player above or
  below the model ("worth more", "higher on", "undervalued" /
  "worth less", "lower on", "overvalued", "overrated"). Applied as
  a bounded ±10% on the target's price, always stamped with the
  preference text that caused it.

Conservative by construction: a signal fires only when a player's
FULL name appears in the same preference as a recognized pricing
phrase. Anything fuzzier is ignored — a misread preference must
never silently reprice a player. League-scoped preferences apply
only to their league; unscoped ones apply everywhere.
"""
from __future__ import annotations

import re
from typing import Optional

BOOST_FACTOR = 1.10
DISCOUNT_FACTOR = 0.90

_NEVER_PHRASES = (
    "never trade", "never offer", "never send", "don't trade",
    "do not trade", "not trading", "won't trade", "will not trade",
    "untouchable",
)
_BOOST_PHRASES = (
    "worth more", "value more", "valued more", "higher on",
    "undervalued", "underrated",
)
_DISCOUNT_PHRASES = (
    "worth less", "value less", "valued less", "lower on",
    "overvalued", "overrated",
)
# "I value <name> more/less than ..." spans the name, so the value
# constructions match on a window rather than a fixed phrase.
_BOOST_WINDOW = re.compile(r"\bvalue[sd]?\b[^.!?]{0,48}\bmore\b")
_DISCOUNT_WINDOW = re.compile(r"\bvalue[sd]?\b[^.!?]{0,48}\bless\b")


def parse_owner_pricing(preferences, name_to_pid: dict,
                        league_id: Optional[str] = None) -> dict:
    """Parse preference rows into {never_offer, boost, discount},
    each {player_id: preference_text}."""
    out = {"never_offer": {}, "boost": {}, "discount": {}}
    names = sorted((name_to_pid or {}).items(),
                   key=lambda kv: -len(kv[0]))
    for pref in preferences or []:
        if not isinstance(pref, dict):
            continue
        pref_league = pref.get("league_id")
        if pref_league and league_id and str(pref_league) != str(league_id):
            continue
        text = str(pref.get("text") or "")
        low = text.lower()
        if not low:
            continue
        for name, pid in names:
            if not name or name not in low:
                continue
            pid = str(pid)
            if any(p in low for p in _NEVER_PHRASES):
                out["never_offer"][pid] = text
            elif any(p in low for p in _BOOST_PHRASES) \
                    or _BOOST_WINDOW.search(low):
                out["boost"][pid] = text
            elif any(p in low for p in _DISCOUNT_PHRASES) \
                    or _DISCOUNT_WINDOW.search(low):
                out["discount"][pid] = text
    return out


def owner_factor(signals: dict, player_id) -> tuple:
    """(factor, preference_text) for one player; (1.0, None) when
    the owner has said nothing priced about him."""
    pid = str(player_id or "")
    boost = (signals or {}).get("boost") or {}
    discount = (signals or {}).get("discount") or {}
    if pid in boost:
        return BOOST_FACTOR, boost[pid]
    if pid in discount:
        return DISCOUNT_FACTOR, discount[pid]
    return 1.0, None


__all__ = ["parse_owner_pricing", "owner_factor",
           "BOOST_FACTOR", "DISCOUNT_FACTOR"]
