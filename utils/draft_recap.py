"""Pure helpers for the Draft Recap feature.

The recap grades every pick of a completed draft by how the player actually
performed (league scoring) versus (1) his draft position in that league and
(2) his market ADP. All of the ranking / verdict / leaderboard math that does
not need the app, a league, or the network lives here so it can be unit
tested in isolation. Rows are plain dicts in the shape the payload builder in
app.py produces (see ``_draft_recap_payload``):

    pick_no, round, draft_slot, roster_id, team, player_id, name, pos,
    total, delta_pick (primary delta, None when ungraded), delta_adp.

Verdict bands (on the primary delta, positional ranks):
    Steal >= +8, Hit >= +3, As drafted -2..+2, Reach -7..-3, Bust <= -8.
"""
from __future__ import annotations

import statistics
from bisect import bisect_right
from typing import Dict, Iterable, List, Optional

CORE_POSITIONS = ("QB", "RB", "WR", "TE")

# Site position-chip colors.
POS_COLORS = {"QB": "#3b82f6", "RB": "#22c55e", "WR": "#f97316", "TE": "#a855f7"}

# Verdict label -> color (matches the approved recap design).
VERDICT_COLORS = {
    "Steal": "#16a34a",
    "Hit": "#22c55e",
    "As drafted": "#64748b",
    "Reach": "#f59e0b",
    "Bust": "#dc2626",
    "No stats": "#94a3b8",
    "Injured": "#64748b",
}

# Current Sleeper designations that exempt a zero-appearance pick: the
# player never had a chance to play, so the miss is unfortunate, not a bust.
INJURY_EXEMPT_STATUSES = ("IR", "PUP")


def _span_cover(out_weeks, res_weeks, completed_weeks):
    """(label, covered-weeks) for the first history arm whose covered weeks
    reach ``completed_weeks - 1``, else None. Shared by both evidence
    formats. Needs at least 2 completed weeks before a near-full absence
    means anything; None week-sets mean that source was unavailable and the
    arm cannot fire -- missing history is never a clean bill of health."""
    try:
        cw = int(completed_weeks or 0)
    except (TypeError, ValueError):
        cw = 0
    if cw < 2:
        return None
    need = cw - 1
    for weeks, label in ((out_weeks, "Out"), (res_weeks, "Reserve")):
        if weeks is None:
            continue
        covered = sorted({int(w) for w in weeks if str(w).strip()})
        if len(covered) >= need:
            return label, covered
    return None


def injury_exemption(current_status=None, body_part=None, out_weeks=None,
                     res_weeks=None, completed_weeks=0) -> Optional[str]:
    """Evidence string when a zero-appearance pick is injury-exempt.

    Arm (a): the player's CURRENT Sleeper injury_status is IR or PUP
    (evidence: "IR · Hamstring"). Arm (b): the official record had him out
    for essentially the whole completed stretch -- report_status "Out" on
    the nflverse weekly injury reports, or a reserve (RES) status on the
    nflverse weekly rosters -- in at least ``completed_weeks - 1`` of the
    completed weeks (evidence: "Out W1-W6" / "Reserve W1-W6").
    ``out_weeks`` / ``res_weeks`` are the covered week numbers (already
    intersected with the completed weeks by the caller). Returns None when
    neither arm exempts."""
    status = str(current_status or "").strip().upper()
    if status in INJURY_EXEMPT_STATUSES:
        part = str(body_part or "").strip()
        return f"{status} · {part}" if part else status
    hit = _span_cover(out_weeks, res_weeks, completed_weeks)
    if hit:
        label, covered = hit
        cw = int(completed_weeks)
        if len(covered) == 1:
            return f"{label} W{covered[0]}"
        if covered == list(range(covered[0], covered[-1] + 1)):
            return f"{label} W{covered[0]}-W{covered[-1]}"
        return f"{label} in {len(covered)} of {cw} weeks"
    return None


def injury_span_exemption(out_weeks=None, res_weeks=None, completed_weeks=0,
                          body_part=None) -> Optional[str]:
    """Evidence string when a pick who DID appear but never scored (0 total
    points) lost essentially the whole completed stretch to injury.

    Duration is the only gate here: the same Out/Reserve span bar as arm
    (b) above (covered weeks >= ``completed_weeks - 1``). A current IR/PUP
    tag alone does NOT exempt an appeared pick -- a player who dressed,
    produced nothing, and then missed all but one week on the official
    record is unfortunate, not a bust; anyone who scored grades on what he
    did. Evidence names the span first ("Reserve · missed 5 of 6 ·
    Ankle") because how long he was out is the whole point."""
    hit = _span_cover(out_weeks, res_weeks, completed_weeks)
    if not hit:
        return None
    label, covered = hit
    cw = int(completed_weeks)
    ev = f"{label} · missed {len(covered)} of {cw}"
    part = str(body_part or "").strip()
    return f"{ev} · {part}" if part else ev


def verdict_for(delta: Optional[int]):
    """(label, color) for a primary delta. None means the player has no
    graded performance yet (0 games), which is never a bust."""
    if delta is None:
        return ("No stats", VERDICT_COLORS["No stats"])
    if delta >= 8:
        return ("Steal", VERDICT_COLORS["Steal"])
    if delta >= 3:
        return ("Hit", VERDICT_COLORS["Hit"])
    if delta >= -2:
        return ("As drafted", VERDICT_COLORS["As drafted"])
    if delta >= -7:
        return ("Reach", VERDICT_COLORS["Reach"])
    return ("Bust", VERDICT_COLORS["Bust"])


def rank_values(values_by_key: Dict[str, float]) -> Dict[str, int]:
    """Competition ranks (1 = best/highest value; ties share the better rank).

    rank = 1 + #{values strictly greater}, computed on the raw values."""
    vals = sorted(values_by_key.values())
    n = len(vals)
    return {k: 1 + (n - bisect_right(vals, v)) for k, v in values_by_key.items()}


def median_or_none(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return float(statistics.median(vals))


def graded_rows(rows: List[dict]) -> List[dict]:
    """Rows with a primary delta (core position, at least one game)."""
    return [r for r in rows if r.get("delta_pick") is not None]


def first_half_bust_pool(rows: List[dict], total_picks: int) -> List[dict]:
    """Graded rows eligible for 'biggest bust': picks from the first half of
    the draft only, so a last-round flier who never played cannot out-bust a
    genuine early-round miss. Falls back to every graded row when the gate
    would leave nobody (degenerate drafts)."""
    graded = graded_rows(rows)
    gate = (total_picks or 0) / 2
    pool = [r for r in graded if (r.get("pick_no") or 0) <= gate]
    return pool or graded


def team_leaderboard(rows: List[dict]) -> List[dict]:
    """Per-team draft summary, best first.

    Grouped by drafting roster (credit the drafter, not the current owner).
    Ranked by the MEDIAN primary delta over the team's graded picks (one
    catastrophic outlier should not sink an otherwise strong draft the way a
    mean would); ties break on total draft points. Draft points are the sum
    of the team's picks' actual fantasy points."""
    by_roster: Dict[str, dict] = {}
    for r in rows:
        rid = str(r.get("roster_id"))
        entry = by_roster.setdefault(rid, {
            "roster_id": r.get("roster_id"),
            "team": (r.get("team") or "").strip(),
            "graded_deltas": [],
            "draft_points_raw": 0.0,
            "best": None,
            "worst": None,
        })
        try:
            entry["draft_points_raw"] += float(r.get("total") or 0.0)
        except (TypeError, ValueError):
            pass
        d = r.get("delta_pick")
        if d is None:
            continue
        entry["graded_deltas"].append(d)
        if entry["best"] is None or d > entry["best"]["delta_pick"]:
            entry["best"] = r
        if entry["worst"] is None or d < entry["worst"]["delta_pick"]:
            entry["worst"] = r
    out = []
    for entry in by_roster.values():
        deltas = entry.pop("graded_deltas")
        if not deltas:
            continue
        entry["median"] = median_or_none(deltas)
        entry["draft_points"] = round(entry.pop("draft_points_raw"), 1)
        entry["hits"] = sum(1 for d in deltas if d >= 3)
        entry["graded"] = len(deltas)
        out.append(entry)
    out.sort(key=lambda e: (-(e["median"] if e["median"] is not None else 0),
                            -e["draft_points"],
                            e["team"] or ""))
    return out


def pick_superlatives(rows: List[dict], total_picks: int,
                      teams: Optional[List[dict]] = None) -> dict:
    """Headline cards: biggest steal, biggest bust (first-half gated), best
    vs ADP, and the best draft (top team on the leaderboard)."""
    graded = graded_rows(rows)
    steal = max(graded, key=lambda r: (r["delta_pick"], -(r.get("pick_no") or 0)),
                default=None)
    bust_pool = first_half_bust_pool(rows, total_picks)
    bust = min(bust_pool, key=lambda r: (r["delta_pick"], r.get("pick_no") or 0),
               default=None)
    with_adp = [r for r in graded if r.get("delta_adp") is not None]
    best_vs_adp = max(with_adp,
                      key=lambda r: (r["delta_adp"], -(r.get("pick_no") or 0)),
                      default=None)
    if teams is None:
        teams = team_leaderboard(rows)
    return {
        "steal": steal,
        "bust": bust,
        "best_vs_adp": best_vs_adp,
        "best_draft": teams[0] if teams else None,
    }


def story_lists(rows: List[dict], mode: str, total_picks: int) -> dict:
    """Curated lists for the page: top-5 steals, top-5 busts (first-half
    gated), and the best pick by round (rookie drafts: one per round;
    redraft/startup: rounds 1-4, 5-10, 11+)."""
    graded = graded_rows(rows)
    steals = sorted(graded, key=lambda r: (-r["delta_pick"], r.get("pick_no") or 0))[:5]
    busts = sorted(first_half_bust_pool(rows, total_picks),
                   key=lambda r: (r["delta_pick"], r.get("pick_no") or 0))[:5]
    by_round: List[dict] = []
    if mode == "rookie":
        groups = [(f"Round {rd}", [r for r in graded if r.get("round") == rd])
                  for rd in sorted({r.get("round") for r in rows if r.get("round")})]
    else:
        groups = [
            ("Rounds 1-4", [r for r in graded if (r.get("round") or 0) <= 4]),
            ("Rounds 5-10", [r for r in graded if 5 <= (r.get("round") or 0) <= 10]),
            ("Rounds 11+", [r for r in graded if (r.get("round") or 0) >= 11]),
        ]
    for label, pool in groups:
        if pool:
            best = max(pool, key=lambda r: (r["delta_pick"], -(r.get("pick_no") or 0)))
            by_round.append({"label": label, "row": best})
    return {"steals": steals, "busts": busts, "by_round": by_round}
