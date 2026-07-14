"""Playoff picture: clinch / elimination / seeding math for a fantasy league.

Pure and dependency-free so it can be unit-tested in isolation. The caller
gathers the inputs (records, playoff size, regular-season length) from league
context and feeds them in; this module decides each team's playoff status.

Model
-----
Wins-based with points-for (PF) as the seeding tiebreaker, which matches the
standings sort. Clinch and elimination use *safe* sufficient conditions built
on each team's win floor (loses out) and ceiling (wins out):

- ``eliminated``  when at least ``playoff_spots`` teams already have more wins
  than this team can still reach. Those teams finish above it in every outcome.
- ``clinched``    when fewer than ``playoff_spots`` other teams can even reach
  this team's win floor. It is then top-N no matter what.

Both conditions never fire incorrectly (a team is never wrongly told it is in
or out); at worst a genuinely-decided team is left as "bubble" a week longer
than a full schedule-enumeration would. That trade — correctness over
aggressiveness — is deliberate, since a wrong "ELIMINATED" tag is the one
mistake this feature cannot make.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional

# Status values, ordered best → worst.
BYE = "bye"
CLINCHED = "clinched"
IN = "in"
BUBBLE = "bubble"
ELIMINATED = "eliminated"


def bye_count(playoff_spots: int) -> int:
    """First-round byes for a standard single-elimination bracket: the seeds
    that skip round one. 6→2, 4→0, 8→0, 7→1, and so on (next power of two minus
    the field)."""
    if playoff_spots < 2:
        return 0
    nxt = 1 << (playoff_spots - 1).bit_length()   # smallest power of 2 ≥ spots
    return nxt - playoff_spots


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _games_back(w_ref: int, l_ref: int, w: int, l: int) -> float:
    """Standard games-back: average of the win gap and the loss gap."""
    return ((w_ref - w) + (l - l_ref)) / 2.0


def compute_playoff_picture(
    teams: List[Dict[str, Any]],
    playoff_spots: int,
    total_regular_weeks: int,
    bye_spots: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return the teams sorted by seed, each annotated with playoff status.

    ``teams``: dicts with ``id``, ``name``, ``wins``, ``losses`` and optionally
    ``ties`` and ``pf``. ``total_regular_weeks`` is the number of regular-season
    games each team plays (``playoff_week_start - 1``). ``bye_spots`` defaults to
    the standard bracket byes for ``playoff_spots``.

    Each returned dict adds: ``seed``, ``status`` (one of the module constants),
    ``games_left``, ``max_wins``, ``games_back`` (from the playoff line, ≥ 0),
    ``controls_own_fate`` (winning out clinches a berth) and ``scenario`` (a
    short, factual line, or ``None``).
    """
    if bye_spots is None:
        bye_spots = bye_count(playoff_spots)

    ts = []
    for t in teams:
        w = int(t.get("wins", 0) or 0)
        l = int(t.get("losses", 0) or 0)
        ti = int(t.get("ties", 0) or 0)
        played = w + l + ti
        gl = max(0, int(total_regular_weeks) - played)
        ts.append({
            "id": t.get("id"),
            "name": t.get("name", ""),
            "wins": w, "losses": l, "ties": ti,
            "pf": float(t.get("pf", 0.0) or 0.0),
            "games_left": gl,
            "max_wins": w + gl,
        })

    # Seed by wins, then PF (mirrors the standings sort).
    ts.sort(key=lambda x: (-x["wins"], -x["pf"]))
    for i, t in enumerate(ts):
        t["seed"] = i + 1

    n = len(ts)
    spots = min(playoff_spots, n)

    def _threats(floor: int, self_id) -> int:
        """Teams (other than self) that can reach `floor` wins — i.e. could tie
        or pass a team sitting on `floor`. Used for the safe clinch test."""
        return sum(1 for o in ts if o["id"] != self_id and o["max_wins"] >= floor)

    def _locked_above(ceiling: int, self_id) -> int:
        """Teams that already have more wins than `ceiling` — locked above a team
        whose best case is `ceiling`. Used for the safe elimination test."""
        return sum(1 for o in ts if o["id"] != self_id and o["wins"] > ceiling)

    # Wins at the playoff line, for games-back and comfort.
    cut_in_wins = ts[spots - 1]["wins"] if spots >= 1 else 0
    cut_in_losses = ts[spots - 1]["losses"] if spots >= 1 else 0
    first_out = ts[spots] if n > spots else None

    for t in ts:
        clinched_playoff = _threats(t["wins"], t["id"]) < spots
        clinched_bye = bye_spots > 0 and _threats(t["wins"], t["id"]) < bye_spots
        eliminated = _locked_above(t["max_wins"], t["id"]) >= spots
        # Would winning out guarantee a berth?
        controls = _threats(t["max_wins"], t["id"]) < spots and not clinched_playoff

        inside = t["seed"] <= spots
        if inside:
            ref = first_out
            gb = _games_back(t["wins"], t["losses"], ref["wins"], ref["losses"]) if ref else float(t["games_left"])
        else:
            gb = _games_back(cut_in_wins, cut_in_losses, t["wins"], t["losses"])
        t["games_back"] = round(max(0.0, gb), 1)
        t["controls_own_fate"] = bool(controls)

        if clinched_bye:
            t["status"] = BYE
        elif clinched_playoff:
            t["status"] = CLINCHED
        elif eliminated:
            t["status"] = ELIMINATED
        elif inside and t["games_back"] > 1:
            t["status"] = IN
        else:
            t["status"] = BUBBLE

        t["scenario"] = _scenario(t, spots, bye_spots, inside)

    return ts


def _scenario(t: Dict[str, Any], spots: int, bye_spots: int, inside: bool) -> Optional[str]:
    st = t["status"]
    if st in (BYE, CLINCHED, ELIMINATED):
        return None
    if t["games_left"] <= 0:
        return None
    if t["controls_own_fate"]:
        return "Win out and you're in."
    if inside:
        return f"Holds the {_ordinal(t['seed'])} seed, but it isn't safe yet."
    gb = t["games_back"]
    gb_txt = "level with" if gb <= 0 else f"{gb:g} back of"
    return f"{gb_txt} the last playoff spot with {t['games_left']} to play."
