"""School -> competition-level classification (model v3.14).

The competition leg prices a profile from its seasons'
conference strings. A profile whose stored seasons carry NO
conference at all (and no Notre Dame team row, and no SP+ SOS
on file) has no level evidence anywhere in its own rows, so
the leg returns None and - before v3.14 - the level tax simply
never fired: Derek Robertson (Monmouth) graded 93.11, #2 in
the 2026 class, on undiscounted FCS counting production while
the system's own confidence score for the file read 30.

This module is the classification of record the model consults
for exactly those profiles: a school whose programme plays at
the FCS level, priced at the model's FCS value (CONF_QUALITY
"FCS" 0.48 -> competition 5.45) by filling the missing
conference with the level label at scoring time. It is a
school classification, never a player list: it can only fire
on a profile with no priced level of its own, and a school
absent from the table classifies as nothing (the profile
stays unpriced and the production cap backstop in
score_prospect is the remaining guard).

Provenance: the FCS programme list is derived from
team-conference data (the Sportradar NCAA teams hierarchy
snapshot carried in this repo: school -> conference, filtered
to the FCS conferences - Big Sky, Big South-OVC, Coastal
Athletic, Ivy, MEAC, Missouri Valley, Northeast, Patriot,
Pioneer, Southern, Southland, SWAC, United Athletic and FCS
independents), normalized with the same function the lookup
uses. North Dakota State is carried as FCS on an era note:
its Mountain West move takes effect with the 2026 season,
after every college career this model grades (it played MVFC
football through 2025). A handful of alias keys (California
Davis, Texas Rio Grande Valley, Long Island) cover the other
name forms the pipeline's sources use for the same schools.
Refresh the list from the same team-conference source when
programmes change level.
"""
from __future__ import annotations

import re
from typing import Optional

# Normalized school keys (see _normalize_school) whose
# programme plays at the FCS level.
FCS_SCHOOLS = frozenset({
    "abilene christian",
    "alabama a and m",
    "alabama state",
    "albany",
    "alcorn state",
    "arkansas pine bluff",
    "austin peay",
    "bethune cookman",
    "brown",
    "bryant",
    "bucknell",
    "butler",
    "cal poly",
    "california davis",
    "campbell",
    "central arkansas",
    "central connecticut state",
    "charleston southern",
    "chattanooga",
    "chicago state",
    "citadel",
    "colgate",
    "columbia",
    "cornell",
    "dartmouth",
    "davidson",
    "dayton",
    "delaware state",
    "drake",
    "duquesne",
    "east tennessee state",
    "east texas a and m",
    "eastern illinois",
    "eastern kentucky",
    "eastern washington",
    "elon",
    "florida a and m",
    "fordham",
    "furman",
    "gardner webb",
    "georgetown",
    "grambling state",
    "hampton",
    "harvard",
    "holy cross",
    "houston christian",
    "howard",
    "idaho",
    "idaho state",
    "illinois state",
    "incarnate word",
    "indiana state",
    "jackson state",
    "lafayette",
    "lamar",
    "lehigh",
    "lindenwood",
    "liu",
    "long island",
    "maine",
    "marist",
    "mcneese",
    "mercer",
    "mercyhurst",
    "merrimack",
    "mississippi valley state",
    "monmouth",
    "montana",
    "montana state",
    "morehead state",
    "morgan state",
    "murray state",
    "new hampshire",
    "new haven",
    "nicholls state",
    "norfolk state",
    "north alabama",
    "north carolina a and t",
    "north carolina central",
    "north dakota",
    "north dakota state",
    "northern arizona",
    "northern colorado",
    "northern iowa",
    "northwestern state",
    "penn",
    "portland state",
    "prairie view a and m",
    "presbyterian",
    "princeton",
    "rhode island",
    "richmond",
    "robert morris",
    "sacred heart",
    "samford",
    "san diego",
    "south carolina state",
    "south dakota",
    "south dakota state",
    "southeast missouri state",
    "southeastern louisiana",
    "southern",
    "southern illinois",
    "southern utah",
    "st thomas",
    "stephen f austin",
    "stetson",
    "stonehill",
    "stony brook",
    "tarleton state",
    "tennessee martin",
    "tennessee state",
    "tennessee tech",
    "texas rio grande valley",
    "texas southern",
    "towson",
    "uc davis",
    "ut rio grande valley",
    "utah tech",
    "valparaiso",
    "villanova",
    "vmi",
    "wagner",
    "weber state",
    "west florida",
    "west georgia",
    "western carolina",
    "western illinois",
    "william and mary",
    "wofford",
    "yale",
    "youngstown state",
})


def _normalize_school(name) -> str:
    """The one normalization both the table and the lookup
    use: case, punctuation and the common university
    prefixes/suffixes are not identity."""
    s = str(name or "").strip().lower().replace("&", " and ")
    s = re.sub(r"[.'’,-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for _pre in ("university at ", "university of ", "the "):
        if s.startswith(_pre):
            s = s[len(_pre):]
    if s.endswith(" university"):
        s = s[:-len(" university")]
    return s.strip()


def school_competition_level(school) -> Optional[str]:
    """The stored-conference label for a school whose level
    is classified here ("FCS"), else None. The label is a
    CONF_QUALITY key, so every downstream price (the leg, the
    per-season discount) reads it exactly like a stored
    conference string."""
    if not school:
        return None
    return "FCS" if _normalize_school(school) in FCS_SCHOOLS else None
