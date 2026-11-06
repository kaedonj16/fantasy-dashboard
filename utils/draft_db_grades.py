"""Persisted grade summaries for the Draft Database corpus.

Two grade families exist on the site and they stay distinct here, exactly
as the live endpoints define them:

(a) At-draft family (GET /api/draft-grades): each pick is graded against
    market ADP. ``adp_diff = pick_no - avg_pick`` (positive = the player
    fell past his ADP, i.e. value). The per-pick letter bands and the
    team-letter averaging below are the endpoint's own ``pick_grade`` /
    ``team_grade`` math (app.py, api_draft_grades). The corpus has no
    pre-draft rosters, so the endpoint's *need* modifier cannot be
    computed at crawl time and is passed as False; the BPA modifier IS
    computed, from the draft's own pick sequence plus the segment ADP
    map (a player drafted later with a better ADP was still available).
    The 0-100 team composite (utils/draft.py ``dr_team_grade_score``)
    needs player values / starter PPG inputs the corpus does not carry,
    so the persisted team grade is the endpoint's letter, not that score.

(b) Performance family (GET /api/draft-recap): each pick is graded by
    how the player actually performed. ``delta_pick`` is the positional
    draft rank minus the positional finish rank, and the verdict bands
    are utils/draft_recap.py's own ``verdict_for`` (Steal >= +8,
    Hit >= +3, As drafted -2..+2, Reach -7..-3, Bust <= -8). The finish
    rank here is the rank among the players drafted at the position in
    this draft (the recap's rookie-mode "class finish" basis); the
    recap's startup/redraft basis ranks against the whole league pool,
    which the corpus cannot reproduce per league. Only computed when a
    PPG map for the draft's season is supplied; rookie drafts without
    performance data carry the note that performance grades land after
    Week 18, mirroring the recap's pending state.

Summary shape (JSON, stored on draft_adp_drafts.grade_summary):

    {
      "version": 1,
      "at_draft": {
        "family": "at_draft",
        "adp_source": "draft_adp segment aggregate",
        "best_draft":  {"roster_id", "team", "grade", "avg_adp_diff",
                        "graded_picks"} | None,
        "biggest_steal": {pick fields + "grade", "adp_diff"} | None,
        "biggest_bust":  {pick fields + "grade", "adp_diff"} | None,
      },
      "performance": {
        "family": "performance",
        "graded_picks": int,
        "biggest_steal": {pick fields + "verdict", "delta_pick"} | None,
        "biggest_bust":  {pick fields + "verdict", "delta_pick"} | None,
      } | None,
      "performance_note": str | None,
      "per_pick": {player_id: {"grade": letter | None,
                               "adp_diff": float | None,
                               "verdict": str | None,
                               "delta_pick": int | None}},
    }

Pick fields: player_id, pick_no, round, roster_id, team.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from utils.draft_recap import rank_values, verdict_for

CORE_POSITIONS = ("QB", "RB", "WR", "TE")

_GRADE_VAL = {"A+": 5, "A": 4, "B": 3, "C": 2, "D": 1, "F": 0, "N/A": 2}

ROOKIE_PERF_NOTE = "Performance grades land after Week 18."


def pick_grade(adp_diff: Optional[float], bpa_gap: Optional[int],
               is_bpa: bool, pos: str, is_sf: bool, qb_count: int,
               num_teams: int) -> str:
    """The /api/draft-grades per-pick letter, minus the need modifier
    (the corpus has no pre-draft rosters; need is always False here)."""
    if adp_diff is None:
        return "N/A"
    big_reach = -(num_teams * 1.1)
    if adp_diff >= 4:
        score = 4
    elif adp_diff >= 2:
        score = 3
    elif adp_diff >= -3:
        score = 2
    elif adp_diff >= big_reach:
        score = 1
    else:
        score = 0
    if is_bpa:
        score += 1 if adp_diff < -3 else 2
    elif bpa_gap is not None and bpa_gap >= 5:
        score = max(score - 1, 0)
    # No need modifier (see module docstring); the redundant-QB penalty
    # from the endpoint still applies because qb_count is known from the
    # draft's own earlier picks.
    if pos == "QB" and not is_sf:
        if qb_count >= 2:
            score = max(score - 2, 0)
        elif qb_count >= 1:
            score = max(score - 1, 0)
    if adp_diff >= -3:
        score = max(score, 1)
    return {5: "A+", 4: "A", 3: "B", 2: "C", 1: "D", 0: "F"}.get(
        min(score, 5), "F")


def team_grade(pick_letters: List[str]) -> str:
    """The /api/draft-grades team letter: mean of per-pick letter values."""
    letters = [g for g in pick_letters if g and g != "N/A"]
    if not letters:
        return "N/A"
    avg = sum(_GRADE_VAL.get(g, 2) for g in letters) / len(letters)
    if avg >= 4.5:
        return "A+"
    if avg >= 3.5:
        return "A"
    if avg >= 2.5:
        return "B"
    if avg >= 1.5:
        return "C"
    if avg >= 0.5:
        return "D"
    return "F"


def _pick_out(p: dict, extra: dict) -> dict:
    out = {
        "player_id": str(p.get("player_id") or ""),
        "pick_no": p.get("pick_no"),
        "round": p.get("round"),
        "roster_id": str(p.get("roster_id")) if p.get("roster_id") is not None else None,
        "team": p.get("team") or "",
    }
    out.update(extra)
    return out


def build_grade_summary(picks: List[dict], adp_map: Dict[str, float],
                        positions: Dict[str, str], *,
                        is_superflex: bool = False, num_teams: int = 12,
                        draft_type: str = "",
                        ppg_by_player: Optional[Dict[str, float]] = None) -> dict:
    """Build the persisted summary for one draft.

    picks: [{player_id, pick_no, round, roster_id, team}] in any order.
    adp_map: {player_id: avg_pick} for the draft's segment (draft_adp).
    positions: {player_id: position}.
    ppg_by_player: {player_id: season PPG} when performance data exists
    for the draft's season, else None (performance family stays empty).
    """
    ordered = sorted(
        (p for p in picks if p.get("player_id") and p.get("pick_no")),
        key=lambda p: int(p["pick_no"]),
    )
    per_pick: Dict[str, dict] = {}
    graded: List[dict] = []  # rows with an at-draft grade

    # BPA context: for each pick, the best (lowest) avg_pick among players
    # drafted AFTER it. Suffix minimum over the ADP-ordered sequence.
    suffix_best: List[Optional[float]] = [None] * (len(ordered) + 1)
    for i in range(len(ordered) - 1, -1, -1):
        ap = adp_map.get(str(ordered[i]["player_id"]))
        nxt = suffix_best[i + 1]
        suffix_best[i] = ap if ap is not None and (nxt is None or ap < nxt) else nxt

    qb_counts: Dict[str, int] = {}
    for i, p in enumerate(ordered):
        pid = str(p["player_id"])
        pos = str(positions.get(pid) or "").upper()
        avg_pick = adp_map.get(pid)
        adp_diff = (int(p["pick_no"]) - float(avg_pick)) if avg_pick is not None else None
        rid = str(p.get("roster_id") or "")
        if adp_diff is None:
            letter = "N/A"
            entry = {"grade": None, "adp_diff": None,
                     "verdict": None, "delta_pick": None}
        else:
            later_best = suffix_best[i + 1]
            is_bpa = later_best is None or (avg_pick is not None and later_best >= avg_pick)
            bpa_gap = None
            if not is_bpa and later_best is not None and avg_pick is not None:
                bpa_gap = int(round(float(avg_pick) - float(later_best)))
            letter = pick_grade(adp_diff, bpa_gap, is_bpa, pos,
                                bool(is_superflex), qb_counts.get(rid, 0),
                                int(num_teams or 12))
            entry = {"grade": letter, "adp_diff": round(float(adp_diff), 1),
                     "verdict": None, "delta_pick": None}
            graded.append({**p, "grade": letter, "adp_diff": float(adp_diff)})
        per_pick[pid] = entry
        if pos == "QB":
            qb_counts[rid] = qb_counts.get(rid, 0) + 1

    # ── At-draft superlatives ─────────────────────────────────────────────
    best_draft = None
    if graded:
        by_roster: Dict[str, dict] = {}
        for r in graded:
            rid = str(r.get("roster_id") or "")
            e = by_roster.setdefault(rid, {
                "roster_id": r.get("roster_id"),
                "team": r.get("team") or "",
                "letters": [], "diffs": [],
            })
            e["letters"].append(r["grade"])
            e["diffs"].append(r["adp_diff"])
        teams = [{
            "roster_id": e["roster_id"],
            "team": e["team"],
            "grade": team_grade(e["letters"]),
            "avg_adp_diff": round(sum(e["diffs"]) / len(e["diffs"]), 1),
            "graded_picks": len(e["letters"]),
            "_gv": sum(_GRADE_VAL.get(g, 2) for g in e["letters"]) / len(e["letters"]),
        } for e in by_roster.values()]
        teams.sort(key=lambda t: (-t["_gv"], -t["graded_picks"], t["team"] or ""))
        top = dict(teams[0])
        top.pop("_gv", None)
        best_draft = top
    steal_ad = max(graded, key=lambda r: (r["adp_diff"], -(r.get("pick_no") or 0)),
                   default=None)
    bust_ad = min(graded, key=lambda r: (r["adp_diff"], r.get("pick_no") or 0),
                  default=None)
    at_draft = {
        "family": "at_draft",
        "adp_source": "draft_adp segment aggregate",
        "best_draft": best_draft,
        "biggest_steal": (_pick_out(steal_ad, {"grade": steal_ad["grade"],
                                               "adp_diff": round(steal_ad["adp_diff"], 1)})
                          if steal_ad else None),
        "biggest_bust": (_pick_out(bust_ad, {"grade": bust_ad["grade"],
                                             "adp_diff": round(bust_ad["adp_diff"], 1)})
                         if bust_ad else None),
    }

    # ── Performance family ────────────────────────────────────────────────
    performance = None
    performance_note = None
    if ppg_by_player:
        # Positional draft rank + finish rank among the drafted pool at
        # the position (recap rookie-mode class-finish basis).
        draft_counter: Dict[str, int] = {}
        rows: List[dict] = []
        for p in ordered:
            pid = str(p["player_id"])
            pos = str(positions.get(pid) or "").upper()
            if pos not in CORE_POSITIONS:
                continue
            draft_counter[pos] = draft_counter.get(pos, 0) + 1
            rows.append({**p, "pos": pos,
                         "draft_pos_rank": draft_counter[pos],
                         "ppg": ppg_by_player.get(pid)})
        finish = {
            pos: rank_values({str(r["player_id"]): r["ppg"] for r in rows
                              if r["pos"] == pos and r["ppg"] is not None})
            for pos in CORE_POSITIONS
        }
        perf_rows = []
        for r in rows:
            fin = finish.get(r["pos"], {}).get(str(r["player_id"]))
            if fin is None:
                continue
            delta = r["draft_pos_rank"] - fin
            verdict = verdict_for(delta)[0]
            perf_rows.append({**r, "delta_pick": delta, "verdict": verdict})
            entry = per_pick.get(str(r["player_id"]))
            if entry is not None:
                entry["verdict"] = verdict
                entry["delta_pick"] = delta
        if perf_rows:
            steal = max(perf_rows,
                        key=lambda r: (r["delta_pick"], -(r.get("pick_no") or 0)))
            total = len(ordered)
            first_half = [r for r in perf_rows
                          if (r.get("pick_no") or 0) <= (total / 2)] or perf_rows
            bust = min(first_half,
                       key=lambda r: (r["delta_pick"], r.get("pick_no") or 0))
            performance = {
                "family": "performance",
                "graded_picks": len(perf_rows),
                "biggest_steal": _pick_out(steal, {
                    "verdict": steal["verdict"],
                    "delta_pick": steal["delta_pick"]}),
                "biggest_bust": _pick_out(bust, {
                    "verdict": bust["verdict"],
                    "delta_pick": bust["delta_pick"]}),
            }
    if performance is None and draft_type == "rookie":
        performance_note = ROOKIE_PERF_NOTE

    return {
        "version": 1,
        "at_draft": at_draft,
        "performance": performance,
        "performance_note": performance_note,
        "per_pick": per_pick,
    }


def headline_superlatives(summary: Optional[dict]) -> dict:
    """Card-level steal/bust pick: the performance family when it exists,
    else the at-draft family. The returned dict always names its family so
    the API/UI never present one family's verdict as the other's."""
    if not summary:
        return {"family": None, "biggest_steal": None, "biggest_bust": None,
                "best_draft": None, "best_draft_family": None}
    perf = summary.get("performance") or {}
    ad = summary.get("at_draft") or {}
    if perf.get("biggest_steal") or perf.get("biggest_bust"):
        return {
            "family": "performance",
            "biggest_steal": perf.get("biggest_steal"),
            "biggest_bust": perf.get("biggest_bust"),
            "best_draft": ad.get("best_draft"),
            "best_draft_family": "at_draft" if ad.get("best_draft") else None,
        }
    return {
        "family": "at_draft" if (ad.get("biggest_steal") or ad.get("biggest_bust")) else None,
        "biggest_steal": ad.get("biggest_steal"),
        "biggest_bust": ad.get("biggest_bust"),
        "best_draft": ad.get("best_draft"),
        "best_draft_family": "at_draft" if ad.get("best_draft") else None,
    }
