"""Cross-league action digest helpers (roadmap R04).

Ranks per-league to-dos for the My Leagues hub. Pure functions so ranking is
unit-testable without Flask / provider I/O.
"""
from __future__ import annotations

import re
from typing import Any, Optional

# Higher = more urgent. Keep the scale small and documented.
_PRIORITY = {
    "lineup": 100,
    "injury": 70,
    "roster": 60,
    "waiver": 50,
    "calendar": 40,
    "trade": 30,
}


def action_priority(kind: str, *, severity: float = 0.0) -> float:
    base = float(_PRIORITY.get(str(kind or "").lower(), 10))
    try:
        sev = max(0.0, min(1.0, float(severity)))
    except (TypeError, ValueError):
        sev = 0.0
    return base + sev * 9.0


def make_action(
    *,
    kind: str,
    platform: str,
    season: int,
    league_id: str,
    league_name: str = "",
    title: str,
    detail: str = "",
    href: str = "",
    severity: float = 0.0,
) -> dict[str, Any]:
    plat = (platform or "sleeper").strip().lower()
    lid = str(league_id or "").strip()
    path = href or (
        f"/{plat}/{int(season)}/{lid}/waivers?tab=startsit"
        if kind == "lineup"
        else f"/{plat}/{int(season)}/{lid}/waivers"
        if kind == "waiver"
        else f"/{plat}/{int(season)}/{lid}/dashboard"
    )
    return {
        "kind": kind,
        "platform": plat,
        "season": int(season),
        "league_id": lid,
        "league_name": league_name or lid,
        "title": title,
        "detail": detail,
        "href": path,
        "priority": action_priority(kind, severity=severity),
    }


def rank_cross_league_actions(actions: list[dict], *, limit: int = 8) -> list[dict]:
    """Sort by priority desc, then league name; cap to ``limit``."""
    rows = [a for a in (actions or []) if isinstance(a, dict) and a.get("title")]
    rows.sort(key=lambda a: (-float(a.get("priority") or 0), str(a.get("league_name") or "")))
    return rows[: max(0, int(limit or 0))]


def lineup_actions_from_issues(
    issues: list[dict],
    *,
    platform: str,
    season: int,
    league_id: str,
    league_name: str = "",
) -> list[dict]:
    """Turn ``find_lineup_issues`` rows into digest actions."""
    if not issues:
        return []
    kinds = {str(i.get("kind") or "") for i in issues}
    if "empty" in kinds:
        title = "Empty starting slot"
        sev = 1.0
    elif "injury" in kinds:
        title = "Injured starter needs a swap"
        sev = 0.85
    elif "bye" in kinds:
        title = "Starter on bye"
        sev = 0.7
    else:
        title = "Lineup needs attention"
        sev = 0.5
    detail = "; ".join(
        str(i.get("detail") or i.get("name") or "").strip()
        for i in issues[:3]
        if (i.get("detail") or i.get("name"))
    )
    return [make_action(
        kind="lineup",
        platform=platform,
        season=season,
        league_id=league_id,
        league_name=league_name,
        title=title,
        detail=detail,
        severity=sev,
    )]


def injury_stash_action(
    *,
    platform: str,
    season: int,
    league_id: str,
    league_name: str,
    player_name: str,
    verdict: str,
    weeks_label: str = "",
    already_on_ir: bool = False,
) -> Optional[dict]:
    v = str(verdict or "").strip()
    if v not in ("IR", "Drop candidate", "Stash"):
        return None
    # Already occupying an IR slot — stash/move-to-IR tips are not actionable.
    # Drop candidate can still matter (free the IR slot).
    if already_on_ir and v in ("IR", "Stash"):
        return None
    return make_action(
        kind="injury",
        platform=platform,
        season=season,
        league_id=league_id,
        league_name=league_name,
        title=f"{v}: {player_name}",
        detail=(f"Approx return {weeks_label}" if weeks_label else "Approximate injury guidance"),
        href=f"/{(platform or 'sleeper').strip().lower()}/{int(season)}/{league_id}/waivers?tab=startsit",
        severity=0.8 if v == "Drop candidate" else 0.55,
    )


# Waiver pickups are only worth a cross-league nudge when the add is actually
# startable (or fills a hole). Highest leftover value is not enough — RB35 /
# QB21 / WR44 are the top *remaining* names in most leagues, not great adds.
#
# Value scale is the shared model (WEIGHTS.min_value == 25). Redraft churns the
# wire for rest-of-season production (lower bar); dynasty only pings for a
# genuine long-term asset (higher bar). Rank ceilings are 12-team startable
# depth, scaled by league size.
WAIVER_MIN_VALUE_MULT = {"redraft": 1.6, "dynasty": 2.4}

# Positional ranks at/inside this are weekly-startable in a typical 12-team
# league. Deeper than this is replacement-level leftover, not a "this week's
# move" unless the roster has a real starter hole (short stretch below).
WAIVER_RANK_CEILING_12 = {
    "redraft": {"QB": 14, "RB": 28, "WR": 32, "TE": 12},
    "dynasty": {"QB": 18, "RB": 30, "WR": 40, "TE": 14},
}
WAIVER_SF_QB_CEILING_12 = {"redraft": 24, "dynasty": 28}

# Aging dynasty players must still be clearly startable — WR35 Tyreek is not a
# stash just because he is the highest leftover dynasty value.
WAIVER_VETERAN_RANK_CEILING_12 = {"QB": 12, "RB": 20, "WR": 24, "TE": 10}

_SKILL_POS = ("QB", "RB", "WR", "TE")
_OUT_STATUS = {"IR", "PUP", "NFI", "OUT", "SUSP", "DOUBTFUL"}
_FA_TEAMS = {"", "FA", "FREE AGENT", "N/A"}
_POS_RANK_RE = re.compile(r"(\d+)\s*$")


def waiver_value_threshold(base_min_value: float, *, is_redraft: bool) -> float:
    """Format-aware minimum value for a waiver pickup to be worth surfacing."""
    try:
        base = float(base_min_value)
    except (TypeError, ValueError):
        base = 25.0
    return base * WAIVER_MIN_VALUE_MULT["redraft" if is_redraft else "dynasty"]


def _clamp_league_size(n_teams) -> int:
    try:
        n = int(n_teams or 12)
    except (TypeError, ValueError):
        n = 12
    return max(8, min(16, n))


def waiver_rank_ceiling(
    pos: str,
    *,
    is_redraft: bool,
    is_sf: bool = False,
    n_teams: int = 12,
) -> int:
    """Deepest positional rank that still counts as a worthwhile add."""
    fmt = "redraft" if is_redraft else "dynasty"
    scale = _clamp_league_size(n_teams) / 12.0
    p = str(pos or "").upper()
    if p == "QB" and is_sf:
        base = WAIVER_SF_QB_CEILING_12[fmt]
    else:
        base = WAIVER_RANK_CEILING_12[fmt].get(p, 24)
    return max(8, int(round(base * scale)))


def parse_pos_rank(rank=None, label: str = "") -> Optional[int]:
    """Numeric positional rank from a value-table field or 'RB35' label."""
    try:
        if rank not in (None, ""):
            n = int(rank)
            if n > 0:
                return n
    except (TypeError, ValueError):
        pass
    m = _POS_RANK_RE.search(str(label or ""))
    return int(m.group(1)) if m else None


def waiver_add_clears_quality_bar(
    *,
    pos: str,
    pos_rank: Optional[int],
    value: float,
    is_redraft: bool,
    is_sf: bool = False,
    n_teams: int = 12,
    age: float = 0.0,
    starter_gap: float = 0.0,
    min_value: float = 25.0,
) -> bool:
    """True when this free agent is worth a cross-league 'Add' nudge.

    Highest remaining value is not enough. The player must clear the format
    value floor *and* be startable-ish (or fill a real starter hole). 1QB
    streamers and past-prime dynasty leftovers are dropped even when they are
    the top name on a barren wire.
    """
    try:
        val = float(value or 0)
    except (TypeError, ValueError):
        val = 0.0
    threshold = waiver_value_threshold(min_value, is_redraft=is_redraft)
    if val < threshold:
        return False

    p = str(pos or "").upper()
    if p not in _SKILL_POS:
        return False

    try:
        gap = float(starter_gap or 0)
    except (TypeError, ValueError):
        gap = 0.0

    ceiling = waiver_rank_ceiling(
        p, is_redraft=is_redraft, is_sf=is_sf, n_teams=n_teams,
    )
    # A lineup hole can stretch slightly past the startable line; replacement
    # RBs (RB35+) still do not qualify.
    stretch = int(round(ceiling * 1.15)) if gap > 0 else ceiling

    if pos_rank is None:
        # Unknown rank: only ping when the value is unambiguously real.
        return val >= threshold * 2.0

    try:
        rk = int(pos_rank)
    except (TypeError, ValueError):
        return False
    if rk <= 0 or rk > stretch:
        return False

    # 1QB: QB15-24 are streamers, not weekly adds, unless the roster cannot
    # start a quarterback.
    if p == "QB" and not is_sf:
        qb_bar = max(8, int(round(12 * (_clamp_league_size(n_teams) / 12.0))))
        if gap <= 0 and rk > qb_bar:
            return False

    # Dynasty: aging players must still be clearly startable.
    if not is_redraft:
        try:
            age_f = float(age or 0)
        except (TypeError, ValueError):
            age_f = 0.0
        if age_f:
            from utils.waiver_score import WAIVER_PRIME_MAX
            prime = float(WAIVER_PRIME_MAX.get(p, 28))
            if age_f > prime + 1:
                vet = max(8, int(round(
                    WAIVER_VETERAN_RANK_CEILING_12.get(p, 16)
                    * (_clamp_league_size(n_teams) / 12.0)
                )))
                if rk > vet:
                    return False
    return True


def waiver_add_detail(
    *,
    pos_rank_label: str = "",
    position: str = "",
    is_redraft: bool = False,
    starter_gap: float = 0.0,
    pos_rank: Optional[int] = None,
    is_sf: bool = False,
    n_teams: int = 12,
) -> str:
    """One-line reason for a cross-league waiver card (not 'top leftover')."""
    bits = []
    if pos_rank_label:
        bits.append(str(pos_rank_label).strip())
    pos = str(position or "").upper()
    if starter_gap and starter_gap > 0 and pos:
        bits.append(f"Fills a {pos} need")
    else:
        startable = waiver_rank_ceiling(
            pos or "WR", is_redraft=is_redraft, is_sf=is_sf, n_teams=n_teams,
        )
        # Tighter than the surfacing ceiling: "startable" copy is for names that
        # would actually be in a lineup, not the last name that cleared the bar.
        startable = max(8, int(round(startable * 0.85)))
        if pos_rank and pos_rank <= startable:
            bits.append("Startable on the wire")
        else:
            fmt = "redraft" if is_redraft else "dynasty"
            bits.append(f"Worth adding by {fmt} value")
    return " · ".join(bits)


def _waiver_need_context(
    roster_players: Optional[list],
    roster_positions: Optional[list],
    pidx: Optional[dict],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (need_mult by pos, starter_gap by pos) for the viewer's roster."""
    need_mults = {p: 1.0 for p in _SKILL_POS}
    gaps = {p: 0.0 for p in _SKILL_POS}
    try:
        from utils.lineup_slots import count_lineup_slots, start_sit_pos, starter_need_counts
        from utils.waiver_score import need_multiplier, positional_need_scores
        counts: dict[str, int] = {}
        for pid in roster_players or []:
            meta = (pidx or {}).get(str(pid)) or {}
            pos = start_sit_pos(meta.get("position") or meta.get("pos") or "")
            if pos in _SKILL_POS:
                counts[pos] = counts.get(pos, 0) + 1
        targets = starter_need_counts(roster_positions or [], extra_depth=1)
        slots = count_lineup_slots(roster_positions or [])
        # A "hole" is fewer bodies than dedicated starters — FLEX does not
        # create a WR/RB need by itself, or every 2-WR league looks empty.
        dedicated = {
            "QB": max(1, int(slots.get("QB") or 0) + int(slots.get("SUPER_FLEX") or 0)),
            "RB": max(1, int(slots.get("RB") or 0)),
            "WR": max(1, int(slots.get("WR") or 0)),
            "TE": max(1, int(slots.get("TE") or 0)),
        }
        need_scores = positional_need_scores(counts, targets)
        need_mults = {pos: need_multiplier(pos, need_scores) for pos in _SKILL_POS}
        for pos in _SKILL_POS:
            gaps[pos] = max(0.0, float(dedicated.get(pos, 0)) - float(counts.get(pos, 0)))
    except Exception:
        pass
    return need_mults, gaps


def select_waiver_add(
    rows: list,
    rostered_ids: set[str],
    *,
    value_key: str,
    fallback_key: str = "value",
    rank_key: str = "pos_rank",
    rank_label_key: str = "pos_rank_label",
    is_redraft: bool = False,
    is_sf: bool = False,
    n_teams: int = 12,
    roster_players: Optional[list] = None,
    roster_positions: Optional[list] = None,
    pidx: Optional[dict] = None,
    min_value: float = 25.0,
    injured_status_by_pid: Optional[dict] = None,
) -> Optional[dict]:
    """Pick one actually-worth-adding free agent, or None.

    Ranks eligible names with ``waiver_pickup_score`` (value, age, roster need,
    trend) after the quality bar drops leftover RB35s / streamer QBs / aging
    dynasty depth. Empty when the wire has nobody worth a cross-league nudge.
    """
    from utils.waiver_score import waiver_pickup_score

    owned = {str(p) for p in (rostered_ids or set())}
    pidx = pidx or {}
    inj_map = injured_status_by_pid or {}
    need_mults, gaps = _waiver_need_context(roster_players, roster_positions, pidx)

    best: Optional[dict] = None
    best_score = -1.0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("id") or row.get("player_id") or "").strip()
        if not pid or pid in owned:
            continue
        pos = str(row.get("position") or row.get("pos") or "").upper()
        if pos not in _SKILL_POS:
            continue
        meta = pidx.get(pid) or {}
        team = str(row.get("team") or meta.get("team") or "").upper()
        if team in _FA_TEAMS:
            continue
        inj = str(inj_map.get(pid) or meta.get("injury_status") or "").upper()
        if inj in _OUT_STATUS:
            continue
        try:
            val = float(row.get(value_key) or row.get(fallback_key) or row.get("value") or 0.0)
        except (TypeError, ValueError):
            val = 0.0
        label = str(row.get(rank_label_key) or row.get("pos_rank_label") or "")
        rk = parse_pos_rank(row.get(rank_key) or row.get("pos_rank"), label)
        try:
            age = float(row.get("age") or meta.get("age") or 0)
        except (TypeError, ValueError):
            age = 0.0
        gap = gaps.get(pos, 0.0)
        if not waiver_add_clears_quality_bar(
            pos=pos, pos_rank=rk, value=val, is_redraft=is_redraft,
            is_sf=is_sf, n_teams=n_teams, age=age, starter_gap=gap,
            min_value=min_value,
        ):
            continue
        name = (
            row.get("name") or row.get("full_name") or row.get("player")
            or meta.get("name") or meta.get("full_name") or ""
        )
        name = str(name).strip()
        if not name or name == pid:
            continue
        try:
            chg = float(row.get("rank_change_7d") or 0)
        except (TypeError, ValueError):
            chg = 0.0
        cand = {
            "player_id": pid,
            "value": val,
            "age": age,
            "position": pos,
            "rank_change_7d": chg,
            "need_mult": need_mults.get(pos, 1.0),
            "pos_rank": rk,
            "self_status": inj,
        }
        try:
            score = float(waiver_pickup_score(cand, {}))
        except Exception:
            score = val
        if score <= best_score:
            continue
        best_score = score
        rank_t = 0.0 if not rk else max(0.0, 1.0 - (rk / 40.0))
        need_t = min(1.0, max(0.0, gap) / 2.0)
        severity = 0.35 + 0.35 * rank_t + 0.30 * need_t
        best = {
            "player_id": pid,
            "name": name,
            "position": pos,
            "pos_rank_label": label,
            "pos_rank": rk,
            "value": val,
            "score": score,
            "starter_gap": gap,
            "severity": severity,
            "reason": waiver_add_detail(
                pos_rank_label=label, position=pos, is_redraft=is_redraft,
                starter_gap=gap, pos_rank=rk, is_sf=is_sf, n_teams=n_teams,
            ),
        }
    return best


def waiver_pickup_action(
    *,
    platform: str,
    season: int,
    league_id: str,
    league_name: str,
    player_name: str,
    position: str = "",
    is_redraft: bool = False,
    pos_rank_label: str = "",
    value: float = 0.0,
    reason: str = "",
    severity: float = 0.0,
    starter_gap: float = 0.0,
    pos_rank: Optional[int] = None,
    is_sf: bool = False,
    n_teams: int = 12,
) -> dict:
    """A waiver add that already cleared ``select_waiver_add``'s quality bar."""
    pos = str(position or "").upper()
    name = str(player_name or "").strip() or "a free agent"
    title = f"Add {name}" + (f" ({pos})" if pos else "")
    detail = str(reason or "").strip() or waiver_add_detail(
        pos_rank_label=pos_rank_label, position=pos, is_redraft=is_redraft,
        starter_gap=starter_gap, pos_rank=pos_rank, is_sf=is_sf, n_teams=n_teams,
    )
    try:
        sev = float(severity or 0)
    except (TypeError, ValueError):
        sev = 0.0
    if sev <= 0:
        sev = 0.5
    return make_action(
        kind="waiver",
        platform=platform,
        season=season,
        league_id=league_id,
        league_name=league_name,
        title=title,
        detail=detail,
        severity=sev,
    )


# Higher = surfaced first when a roster has more than one wasted-capacity issue.
_ROSTER_ISSUE_RANK = {"ir_activate": 0.9, "ir_stash": 0.8, "taxi_stash": 0.6}
_ROSTER_ISSUE_TITLE = {
    "ir_activate": "Activate or drop a recovered IR player",
    "ir_stash": "Move an injured player to your IR slot",
    "taxi_stash": "Stash a rookie on your open taxi slot",
}


def roster_slot_action(
    issues: list[dict],
    *,
    platform: str,
    season: int,
    league_id: str,
    league_name: str,
) -> Optional[dict]:
    """Turn ``roster_compliance_issues`` rows into one digest action per league.

    Surfaces the most actionable wasted-capacity issue (a recovered player stuck
    on IR, an injured player who could free a spot, or an open taxi slot).
    """
    rows = [i for i in (issues or []) if isinstance(i, dict) and i.get("kind")]
    if not rows:
        return None
    top = max(rows, key=lambda i: _ROSTER_ISSUE_RANK.get(str(i.get("kind")), 0.0))
    kind = str(top.get("kind"))
    plat = (platform or "sleeper").strip().lower()
    return make_action(
        kind="roster",
        platform=plat,
        season=season,
        league_id=league_id,
        league_name=league_name,
        title=_ROSTER_ISSUE_TITLE.get(kind, "Free up a roster spot"),
        detail=str(top.get("detail") or ""),
        href=f"/{plat}/{int(season)}/{league_id}/teams",
        severity=_ROSTER_ISSUE_RANK.get(kind, 0.5),
    )


def calendar_action(
    *,
    platform: str,
    season: int,
    league_id: str,
    league_name: str,
    week: int,
    trade_deadline: int = 0,
    playoff_week_start: int = 0,
) -> Optional[dict]:
    """A time-sensitive league-calendar nudge (trade deadline, then playoffs).

    Only fires in-season and within two weeks of an event. The trade deadline
    takes precedence over the playoff countdown when both are near.
    """
    try:
        wk = int(week or 0)
    except (TypeError, ValueError):
        wk = 0
    if wk <= 0:
        return None
    plat = (platform or "sleeper").strip().lower()

    def _weeks(n: int) -> str:
        return f"{n} week" + ("s" if n != 1 else "")

    # Trade deadline: only trust a sane in-season week number (Sleeper stores 0
    # or a large sentinel when there is no deadline).
    try:
        dl = int(trade_deadline or 0)
    except (TypeError, ValueError):
        dl = 0
    if 1 <= dl <= 18:
        gap = dl - wk
        if gap == 0:
            return make_action(
                kind="calendar", platform=plat, season=season, league_id=league_id,
                league_name=league_name,
                title="Trade deadline is this week",
                detail=f"Week {dl} is the last week to make a deal.",
                href=f"/{plat}/{int(season)}/{league_id}/trade",
                severity=0.95,
            )
        if 0 < gap <= 2:
            return make_action(
                kind="calendar", platform=plat, season=season, league_id=league_id,
                league_name=league_name,
                title=f"Trade deadline in {_weeks(gap)}",
                detail=f"Deadline is Week {dl}. Line up any deals before the wire closes.",
                href=f"/{plat}/{int(season)}/{league_id}/trade",
                severity=0.75 if gap == 1 else 0.6,
            )

    # Playoffs approaching: seed-and-lock-your-roster nudge.
    try:
        pw = int(playoff_week_start or 0)
    except (TypeError, ValueError):
        pw = 0
    if 1 <= pw <= 18:
        gap = pw - wk
        if 0 < gap <= 2:
            return make_action(
                kind="calendar", platform=plat, season=season, league_id=league_id,
                league_name=league_name,
                title=f"Playoffs start in {_weeks(gap)}",
                detail=f"Week {pw} begins the playoffs. Lock your roster and seeding now.",
                href=f"/{plat}/{int(season)}/{league_id}/matchups",
                severity=0.55 if gap == 1 else 0.45,
            )
    return None
