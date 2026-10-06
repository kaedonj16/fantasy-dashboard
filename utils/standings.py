"""Consolidated utils module: standings.

standings, playoff bracket/picture, history, all-play

Merged from: utils/standings_divisions.py, utils/standings_viz.py, utils/playoff_bracket.py, utils/playoff_picture.py, utils/all_play.py, utils/seed_series.py, utils/season_review.py, utils/history_seasons.py.
Old import paths keep working via compatibility shims.
"""
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations


# ======================================================================
# From utils/standings_divisions.py
# ======================================================================

"""Division-aware standings helpers.

When a league has 2+ divisions with per-team assignments, standings should
group by division and seed playoff spots as division winners first, then wild
cards by record — matching Sleeper / playoff-scenario behavior.
"""

import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Last known-good ``roster_id -> division_id`` maps, keyed by
# ``"<platform>:<league_id>"``. Guards the Div badge (and division-aware
# standings) against transient provider responses that omit per-roster
# division assignments: with the short live-game cache TTL the league ctx
# rebuilds every couple of minutes, and one bad roster payload would
# otherwise wipe the badges until the next good rebuild.
_LAST_GOOD_DIV_MAP: Dict[str, Dict[int, int]] = {}


def _divisions_configured(settings: Optional[Mapping[str, Any]]) -> Optional[bool]:
    """True/False when league settings explicitly say divisions on/off.

    Returns None when the settings carry no division count, in which case the
    roster assignments alone decide.
    """
    try:
        raw = (settings or {}).get("divisions")
    except Exception:
        return None
    if raw in (None, ""):
        return None
    try:
        return int(raw) >= 2
    except (TypeError, ValueError):
        return None


def roster_division_map(
    rosters: Optional[Iterable[Mapping[str, Any]]],
    *,
    league_key: Optional[str] = None,
    settings: Optional[Mapping[str, Any]] = None,
) -> Dict[int, int]:
    """``roster_id -> division_id`` for teams with a positive division setting.

    When ``league_key`` is given, a non-empty result is remembered per league.
    If a later call builds an empty map for a league whose settings still show
    divisions configured, the last good map is returned instead, so a
    transient provider hiccup does not silently drop Div badges. Leagues
    without divisions are unaffected: their map stays empty because no good
    map was ever cached, and an explicit ``divisions < 2`` setting clears any
    stale entry.
    """
    out: Dict[int, int] = {}
    for r in rosters or []:
        rid = r.get("roster_id")
        if rid is None:
            continue
        try:
            div = int((r.get("settings") or {}).get("division") or 0)
        except (TypeError, ValueError):
            div = 0
        if div:
            try:
                out[int(rid)] = div
            except (TypeError, ValueError):
                continue
    if league_key:
        key = str(league_key)
        if out:
            _LAST_GOOD_DIV_MAP[key] = dict(out)
        else:
            configured = _divisions_configured(settings)
            if configured is False:
                # League explicitly runs without divisions: drop any stale map
                # rather than resurrecting one.
                _LAST_GOOD_DIV_MAP.pop(key, None)
            else:
                cached = _LAST_GOOD_DIV_MAP.get(key)
                if cached:
                    logger.warning(
                        "[divisions] roster data missing divisions for league %s, "
                        "using cached map",
                        key,
                    )
                    return dict(cached)
    return out


def div_map_for_ctx(ctx: Mapping[str, Any]) -> Dict[int, int]:
    """``roster_division_map`` with league-keyed fallback, derived from a ctx."""
    settings = ctx.get("league_settings") or (ctx.get("league") or {}).get("settings") or {}
    platform = ctx.get("platform") or "sleeper"
    league_id = ctx.get("league_id") or ctx.get("resolved_league_id")
    league_key = f"{platform}:{league_id}" if league_id else None
    return roster_division_map(ctx.get("rosters"), league_key=league_key, settings=settings)


def division_name_map(
    league: Optional[Mapping[str, Any]],
    division_ids: Iterable[int],
    rosters: Optional[Iterable[Mapping[str, Any]]] = None,
) -> Dict[int, str]:
    """Human labels for division ids (Sleeper ``metadata.division_N``, else fallback)."""
    meta = (league or {}).get("metadata") if isinstance(league, Mapping) else None
    if not isinstance(meta, dict):
        meta = {}
    # ESPN / some hosts store the label on each roster instead of league metadata.
    from_roster: Dict[int, str] = {}
    for r in rosters or []:
        try:
            div = int((r.get("settings") or {}).get("division") or 0)
        except (TypeError, ValueError):
            continue
        if not div or div in from_roster:
            continue
        label = (r.get("metadata") or {}).get("division_name")
        if isinstance(label, str) and label.strip():
            from_roster[div] = label.strip()

    names: Dict[int, str] = {}
    for div in sorted({int(d) for d in division_ids if d}):
        label = (
            meta.get(f"division_{div}")
            or meta.get(f"Division {div}")
            or meta.get(str(div))
            or from_roster.get(div)
        )
        if isinstance(label, str) and label.strip():
            names[div] = label.strip()
        else:
            names[div] = f"Division {div}"
    return names


def active_divisions(
    settings: Optional[Mapping[str, Any]],
    rosters: Optional[Iterable[Mapping[str, Any]]],
    *,
    league_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return division info when the league should split standings, else None.

    Active when at least two distinct per-team division ids are present. If
    ``settings.divisions`` is explicitly 0/1 we stay flat (host says no
    divisions); a missing count still splits when roster assignments exist.
    ``league_key`` enables the last-good-map fallback inside
    :func:`roster_division_map` for transient provider hiccups.
    """
    try:
        raw = (settings or {}).get("divisions")
        n_settings = int(raw) if raw not in (None, "") else None
    except (TypeError, ValueError):
        n_settings = None
    by_rid = roster_division_map(rosters, league_key=league_key, settings=settings)
    unique = sorted({d for d in by_rid.values() if d})
    if len(unique) < 2:
        return None
    if n_settings is not None and n_settings < 2:
        return None
    return {
        "by_rid": by_rid,
        "ids": unique,
        "names": division_name_map(None, unique, rosters),
        "count": len(unique),
    }


def resolve_divisions(ctx: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull active division info from a league ctx (settings + rosters + metadata).

    Derives the league key from the ctx so the last-good-map fallback applies
    here too (division records, standings grouping).
    """
    settings = ctx.get("league_settings") or (ctx.get("league") or {}).get("settings") or {}
    rosters = ctx.get("rosters")
    platform = ctx.get("platform") or "sleeper"
    league_id = ctx.get("league_id") or ctx.get("resolved_league_id")
    league_key = f"{platform}:{league_id}" if league_id else None
    info = active_divisions(settings, rosters, league_key=league_key)
    if not info:
        return None
    league = ctx.get("league") if isinstance(ctx.get("league"), Mapping) else {}
    info["names"] = division_name_map(league, info["ids"], rosters)
    return info


def division_win_pct(div_record: Optional[Sequence[float]]) -> float:
    """Winning percentage for a ``(wins, losses, ties)`` division record.

    Ties count half a win. A team with no division games yet (or no record)
    gets 0.0, so it cannot outrank a division winner on this tiebreak alone.
    Display standings break overall-record ties by this value before PF:
    a 1-0 division team ranks above a 0-1 team with the same overall record
    even when the 0-1 team has scored more points.
    """
    if not div_record:
        return 0.0
    try:
        w, l, t = (float(x or 0) for x in div_record)
    except (TypeError, ValueError):
        return 0.0
    games = w + l + t
    if games <= 0:
        return 0.0
    return (w + 0.5 * t) / games


def playoff_seed_order(
    teams: Sequence[Mapping[str, Any]],
    *,
    division_key: str = "division",
) -> List[int]:
    """Return indices into ``teams`` in playoff-seed order (1st seed first).

    Each team mapping needs ``wins``, ``pf``, and optionally ``pa`` / ``ties``.
    With 2+ distinct divisions, division winners are seeded ahead of wild cards.

    The tiebreak chain matches the standings tables: overall wins, then
    division win% (from an optional ``div_record`` ``(w, l, t)`` per team),
    then PF, then PA. A team with no ``div_record`` counts as 0.0, so callers
    that cannot supply division records keep the old wins/PF/PA behavior.
    """
    m = len(teams)
    if m == 0:
        return []

    def _wins(t: Mapping[str, Any]) -> float:
        try:
            w = float(t.get("wins", 0) or 0)
            ti = float(t.get("ties", 0) or 0)
            return w + 0.5 * ti
        except (TypeError, ValueError):
            return 0.0

    def _pf(t: Mapping[str, Any]) -> float:
        try:
            return float(t.get("pf", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    def _pa(t: Mapping[str, Any]) -> float:
        try:
            return float(t.get("pa", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    def _div(t: Mapping[str, Any]) -> int:
        try:
            return int(t.get(division_key) or 0)
        except (TypeError, ValueError):
            return 0

    def _seed_key(t: Mapping[str, Any]) -> Tuple[float, float, float, float]:
        # Higher is better on every element (PA negated).
        return (
            _wins(t),
            division_win_pct(t.get("div_record")),
            _pf(t),
            -_pa(t),
        )

    idxs = list(range(m))
    idxs.sort(key=lambda i: _seed_key(teams[i]), reverse=True)

    divs = [_div(t) for t in teams]
    unique = {d for d in divs if d}
    if len(unique) < 2:
        return idxs

    by_div: Dict[int, List[int]] = {}
    for i, d in enumerate(divs):
        by_div.setdefault(d or 0, []).append(i)

    winners: List[int] = []
    rest: List[int] = []
    for div_idxs in by_div.values():
        # Best record in the division; stable tie-break by original index.
        w = max(div_idxs, key=lambda i: (_seed_key(teams[i]), -i))
        winners.append(w)
        rest.extend(i for i in div_idxs if i != w)

    winners.sort(key=lambda i: _seed_key(teams[i]), reverse=True)
    rest.sort(key=lambda i: _seed_key(teams[i]), reverse=True)
    return winners + rest


def assign_playoff_seeds(
    teams: Sequence[Mapping[str, Any]],
    *,
    division_key: str = "division",
) -> List[int]:
    """1-based playoff seed for each team index."""
    order = playoff_seed_order(teams, division_key=division_key)
    seeds = [0] * len(teams)
    for rank, i in enumerate(order):
        seeds[i] = rank + 1
    return seeds


def _norm_rid(rid) -> Optional[int]:
    try:
        return int(rid)
    except (TypeError, ValueError):
        return None


def division_records(df_weekly, by_rid: Mapping[int, int]) -> Dict[int, Tuple[int, int, int]]:
    """Per-team ``(wins, losses, ties)`` in games vs same-division opponents.

    Opponents are paired via ``(week, matchup_id)`` on finalized rows. A game
    counts toward the division record only when both teams carry the same
    nonzero division id. Groups that aren't exactly two teams, or rows missing
    ``matchup_id``, are skipped. Keys are int roster ids.
    """
    out: Dict[int, List[int]] = {}
    if df_weekly is None or getattr(df_weekly, "empty", True):
        return {}
    try:
        cols = set(df_weekly.columns)
    except Exception:
        return {}
    if not {"week", "matchup_id", "roster_id", "points", "points_against"} <= cols:
        return {}
    try:
        frame = df_weekly[df_weekly["finalized"] == True]  # noqa: E712
    except Exception:
        return {}
    div_of = {_norm_rid(k): v for k, v in (by_rid or {}).items()}
    for (_wk, _mid), grp in frame.groupby(["week", "matchup_id"]):
        if len(grp) != 2:
            continue
        rows = list(grp.itertuples())
        try:
            ra, rb = _norm_rid(rows[0].roster_id), _norm_rid(rows[1].roster_id)
        except AttributeError:
            continue
        if ra is None or rb is None:
            continue
        da, db = div_of.get(ra) or 0, div_of.get(rb) or 0
        if not da or da != db:
            continue
        pa = float(rows[0].points or 0)
        pb = float(rows[1].points or 0)
        for rid, won, tied in ((ra, pa > pb, pa == pb), (rb, pb > pa, pa == pb)):
            rec = out.setdefault(rid, [0, 0, 0])
            if tied:
                rec[2] += 1
            elif won:
                rec[0] += 1
            else:
                rec[1] += 1
    return {rid: (w, l, t) for rid, (w, l, t) in out.items()}


def format_record(wins: int, losses: int, ties: int = 0,
                  div_record: Optional[Tuple[int, int, int]] = None) -> str:
    """``'2-1'`` / ``'2-1-1'``, with the division record appended as
    ``'2-1 (2-0)'`` when ``div_record`` is given (ties shown only when > 0)."""
    rec = f"{int(wins)}-{int(losses)}"
    if int(ties or 0):
        rec += f"-{int(ties)}"
    if div_record is not None:
        dw, dl, dt = (int(x or 0) for x in div_record)
        div = f"{dw}-{dl}"
        if dt:
            div += f"-{dt}"
        rec += f" ({div})"
    return rec


def format_record_html(wins: int, losses: int, ties: int = 0,
                       div_record: Optional[Tuple[int, int, int]] = None) -> str:
    """HTML version of :func:`format_record` for table cells and meta lines.

    The whole record sits in ``<span class="st-record">`` (``white-space:
    nowrap``) so ``'2-0 (1-0)'`` never wraps mid-record on narrow phones, and
    the parenthesized division part gets ``<span class="st-div-rec">`` (muted,
    slightly smaller) to help it fit. Plain-text callers keep using
    ``format_record``.
    """
    import html as _html
    rec = f"{int(wins)}-{int(losses)}"
    if int(ties or 0):
        rec += f"-{int(ties)}"
    rec_html = _html.escape(rec)
    if div_record is not None:
        dw, dl, dt = (int(x or 0) for x in div_record)
        div = f"{dw}-{dl}"
        if dt:
            div += f"-{dt}"
        rec_html += f' <span class="st-div-rec">({_html.escape(div)})</span>'
    return f'<span class="st-record">{rec_html}</span>'


def division_records_for_ctx(ctx: Mapping[str, Any]) -> Optional[Dict[int, Tuple[int, int, int]]]:
    """``{roster_id: (w, l, t)}`` vs division opponents from the ctx's weekly
    frame, or ``None`` when the league doesn't use divisions. Callers pass the
    result straight into renderers so records show as ``'2-1 (2-0)'``."""
    info = resolve_divisions(ctx) or {}
    by_rid = info.get("by_rid") or {}
    if not by_rid:
        return None
    return division_records(ctx.get("df_weekly"), by_rid)


def is_division_game(rid_a: Any, rid_b: Any,
                     div_by_rid: Optional[Mapping[Any, Any]] = None) -> bool:
    """True when both roster ids are in the same nonzero division.

    Returns False when the league has no divisions (empty/None map), when
    either team is unassigned, or when the ids are missing/unparseable.
    """
    if not div_by_rid:
        return False
    try:
        da = int((div_by_rid.get(int(rid_a)) if rid_a is not None else None) or 0)
        db = int((div_by_rid.get(int(rid_b)) if rid_b is not None else None) or 0)
    except (TypeError, ValueError):
        return False
    return bool(da) and da == db


# ======================================================================
# From utils/standings_viz.py
# ======================================================================

"""Server-rendered SVG scatter charts for league-wide team views.

Two league scatters shared by the Graphs page and the team modal's Graphs tab
(both call the same functions so the two surfaces can't drift):

  - luck_quadrant_svg: all-play win rate (true strength, x) vs actual win rate
    (y), with a dashed y=x "deserved" diagonal. Above the line = lucky, below =
    unlucky. Position is the signal; an amber/blue point tint (CVD-safe) only
    reinforces it.
  - value_age_svg: average roster age (x) vs total dynasty value (y), split into
    quadrants by the league medians (young+loaded / win-now / rebuilding / aging).

Pure string builders (no app/pandas imports) so they unit-test cleanly and run
identically server-side on either surface. Colors use CSS custom properties with
literal fallbacks so they adapt to light/dark themes.
"""

import html

# CVD-safe reinforcement colors (position is the primary encoding).
_LUCK = "#f59e0b"    # amber: luckier than scoring earned
_UNLUCK = "#3b82f6"  # blue: unluckier than scoring earned
_NEU = "#94a3b8"     # gray: within a game of deserved
_ACCENT = "#6366f1"  # indigo: neutral team marker for value/age

# Point labels get a background-colored halo (paint-order stroke) so they stay
# legible where teams cluster and labels overlap points/gridlines.
_LBL = ('font-size="9.5" fill="var(--text,#334155)" paint-order="stroke" '
        'stroke="var(--card,#ffffff)" stroke-width="2.5" stroke-linejoin="round"')


def _esc(s) -> str:
    return html.escape(str(s))


def luck_quadrant_svg(analysis: dict, viewer_owner: str = "", owner_colors: dict = None) -> str:
    """SVG scatter of actual win% (y) vs all-play win% (x) with a 'deserved'
    diagonal. Points use the shared per-owner color map (identity); the luck
    signal is the position above/below the diagonal. Returns '' when fewer than
    3 teams have played, or before any team has actually won a game (at season
    start every point sits on the zero line and the chart is meaningless)."""
    rows: List[Tuple[str, dict]] = [
        (o, a) for o, a in (analysis or {}).items()
        if a.get("games") and a["games"] > 0
    ]
    if len(rows) < 3:
        return ""
    if sum(float(a.get("actual_wins") or 0) for _, a in rows) <= 0:
        return ""

    colors = owner_colors or {}
    W, H, pad = 460, 340, 44
    x0, y0, x1, y1 = pad, 16, W - 16, H - pad

    def sx(pct):  # all-play % -> x
        return x0 + pct * (x1 - x0)

    def sy(pct):  # actual % -> y (inverted)
        return y1 - pct * (y1 - y0)

    parts = [
        f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'style="width:100%;max-width:520px;height:auto;display:block;margin:0 auto;" '
        f'class="luck-quadrant" role="img" '
        f'aria-label="Performance vs luck: each team plotted by all-play win rate against actual win rate">'
    ]
    parts.append(
        f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" '
        f'stroke="var(--border,#e2e8f0)" stroke-width="1"/>'
    )
    for g in (0.25, 0.5, 0.75):
        parts.append(f'<line x1="{sx(g):.1f}" y1="{y0}" x2="{sx(g):.1f}" y2="{y1}" stroke="var(--border,#e2e8f0)" stroke-width="0.5" opacity="0.5"/>')
        parts.append(f'<line x1="{x0}" y1="{sy(g):.1f}" x2="{x1}" y2="{sy(g):.1f}" stroke="var(--border,#e2e8f0)" stroke-width="0.5" opacity="0.5"/>')
    # "Deserved" diagonal (y = x).
    parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}" stroke="var(--text-muted,#94a3b8)" stroke-width="1.5" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{x0+8}" y="{y0+16}" font-size="11" font-weight="700" fill="{_LUCK}">Lucky</text>')
    parts.append(f'<text x="{x1-8}" y="{y1-8}" font-size="11" font-weight="700" fill="{_UNLUCK}" text-anchor="end">Unlucky</text>')
    parts.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-8}" font-size="11" fill="var(--text-muted,#94a3b8)" text-anchor="middle">All-play win rate (true strength) &rarr;</text>')
    parts.append(f'<text x="14" y="{(y0+y1)/2:.0f}" font-size="11" fill="var(--text-muted,#94a3b8)" text-anchor="middle" transform="rotate(-90 14 {(y0+y1)/2:.0f})">Actual win rate &rarr;</text>')

    for owner, a in sorted(rows, key=lambda r: r[1]["all_play_pct"]):
        ax = sx(a["all_play_pct"])
        ay = sy(a["actual_wins"] / a["games"])
        delta = a.get("luck_delta", 0)
        col = colors.get(str(owner)) or _NEU
        is_me = viewer_owner and str(owner) == str(viewer_owner)
        r = 7 if is_me else 5
        ring = ' stroke="var(--text,#0f172a)" stroke-width="2"' if is_me else ' stroke="#fff" stroke-width="1"'
        short = _esc(str(owner)[:12])
        sign = "+" if delta > 0 else ""
        parts.append(
            f'<g><title>{_esc(owner)}: {a["actual_wins"]:.0f} actual wins vs '
            f'{a["expected_wins"]:.1f} expected ({sign}{delta:.1f})</title>'
            f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="{r}" fill="{col}"{ring}/>'
        )
        if ax > x1 - 70:
            parts.append(f'<text x="{ax-r-3:.1f}" y="{ay+3:.1f}" {_LBL} text-anchor="end">{short}</text></g>')
        else:
            parts.append(f'<text x="{ax+r+3:.1f}" y="{ay+3:.1f}" {_LBL}>{short}</text></g>')

    parts.append("</svg>")
    return "".join(parts)


def _median(vals: List[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def value_age_svg(rows: List[dict], viewer_owner: str = "", owner_colors: dict = None) -> str:
    """SVG scatter of total dynasty value (y) vs average roster age (x), split
    into quadrants by the league medians. Younger + more valuable (top-left) is
    the ascending-dynasty corner. Points use the shared per-owner color map.
    Returns '' with fewer than 3 valued teams."""
    pts = [
        r for r in (rows or [])
        if (r.get("total_value") or 0) > 0 and (r.get("avg_age") or 0) > 0
    ]
    if len(pts) < 3:
        return ""
    colors = owner_colors or {}

    ages = [float(r["avg_age"]) for r in pts]
    vals = [float(r["total_value"]) for r in pts]
    a_min, a_max = min(ages), max(ages)
    v_min, v_max = min(vals), max(vals)
    # Pad the ranges a touch so edge points aren't glued to the frame.
    a_pad = max((a_max - a_min) * 0.12, 0.5)
    v_pad = max((v_max - v_min) * 0.12, 1.0)
    a_lo, a_hi = a_min - a_pad, a_max + a_pad
    v_lo, v_hi = v_min - v_pad, v_max + v_pad
    a_med, v_med = _median(ages), _median(vals)

    W, H, pad = 460, 340, 46
    x0, y0, x1, y1 = pad, 16, W - 16, H - pad

    def sx(age):
        return x0 + (age - a_lo) / max(a_hi - a_lo, 1e-9) * (x1 - x0)

    def sy(val):
        return y1 - (val - v_lo) / max(v_hi - v_lo, 1e-9) * (y1 - y0)

    parts = [
        f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'style="width:100%;max-width:520px;height:auto;display:block;margin:0 auto;" '
        f'class="luck-quadrant" role="img" '
        f'aria-label="Dynasty value versus average roster age for each team">'
    ]
    parts.append(
        f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" '
        f'stroke="var(--border,#e2e8f0)" stroke-width="1"/>'
    )
    # Median split lines make the four quadrants.
    mx, my = sx(a_med), sy(v_med)
    parts.append(f'<line x1="{mx:.1f}" y1="{y0}" x2="{mx:.1f}" y2="{y1}" stroke="var(--text-muted,#94a3b8)" stroke-width="1" stroke-dasharray="4 4" opacity="0.7"/>')
    parts.append(f'<line x1="{x0}" y1="{my:.1f}" x2="{x1}" y2="{my:.1f}" stroke="var(--text-muted,#94a3b8)" stroke-width="1" stroke-dasharray="4 4" opacity="0.7"/>')
    # Corner labels (younger is left, more valuable is up).
    parts.append(f'<text x="{x0+8}" y="{y0+15}" font-size="10" font-weight="700" fill="var(--text-muted,#94a3b8)">Young &amp; loaded</text>')
    parts.append(f'<text x="{x1-8}" y="{y0+15}" font-size="10" font-weight="700" fill="var(--text-muted,#94a3b8)" text-anchor="end">Win-now</text>')
    parts.append(f'<text x="{x0+8}" y="{y1-8}" font-size="10" font-weight="700" fill="var(--text-muted,#94a3b8)">Rebuilding</text>')
    parts.append(f'<text x="{x1-8}" y="{y1-8}" font-size="10" font-weight="700" fill="var(--text-muted,#94a3b8)" text-anchor="end">Aging out</text>')
    # Axis titles.
    parts.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-8}" font-size="11" fill="var(--text-muted,#94a3b8)" text-anchor="middle">Average roster age &rarr;</text>')
    parts.append(f'<text x="14" y="{(y0+y1)/2:.0f}" font-size="11" fill="var(--text-muted,#94a3b8)" text-anchor="middle" transform="rotate(-90 14 {(y0+y1)/2:.0f})">Total dynasty value &rarr;</text>')

    for r in sorted(pts, key=lambda r: r["total_value"]):
        owner = r["owner"]
        px, py = sx(float(r["avg_age"])), sy(float(r["total_value"]))
        is_me = viewer_owner and str(owner) == str(viewer_owner)
        rad = 7 if is_me else 5
        col = colors.get(str(owner)) or _ACCENT
        ring = ' stroke="var(--text,#0f172a)" stroke-width="2"' if is_me else ' stroke="#fff" stroke-width="1"'
        short = _esc(str(owner)[:12])
        parts.append(
            f'<g><title>{_esc(owner)}: {r["total_value"]:.0f} total value, '
            f'{r["avg_age"]:.1f} avg age</title>'
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{rad}" fill="{col}"{ring}/>'
        )
        if px > x1 - 70:
            parts.append(f'<text x="{px-rad-3:.1f}" y="{py+3:.1f}" {_LBL} text-anchor="end">{short}</text></g>')
        else:
            parts.append(f'<text x="{px+rad+3:.1f}" y="{py+3:.1f}" {_LBL}>{short}</text></g>')

    parts.append("</svg>")
    return "".join(parts)


# ======================================================================
# From utils/playoff_bracket.py
# ======================================================================

"""Derive a Sleeper-shaped winners bracket from weekly matchup rows.

Fleaflicker and MFL do not publish a bracket object. Playoff weeks still
have paired matchups. This helper turns those rows into the
``{r, m, t1, t2, t1_from, t2_from, w, l}`` shape ``playoff_bracket``
already renders.

When playoff weeks have not been played yet, ``project_bracket_from_seeds``
builds a first-round pairing from standings order so the Playoff Picture
tab is not empty all season.
"""

from collections import defaultdict


def _rid(row: dict) -> Optional[int]:
    raw = row.get("roster_id")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _mid(row: dict, fallback: int) -> int:
    try:
        return int(row.get("matchup_id") or fallback)
    except (TypeError, ValueError):
        return fallback


def _pts(row: dict) -> Optional[float]:
    raw = row.get("points")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def pair_matchup_sides(rows: Iterable[dict]) -> List[tuple[int, dict, dict]]:
    """Group roster rows that share a ``matchup_id`` into two-team games."""
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for i, row in enumerate(rows or [], 1):
        if not isinstance(row, dict):
            continue
        if _rid(row) is None:
            continue
        grouped[_mid(row, i)].append(row)
    out: List[tuple[int, dict, dict]] = []
    for mid, sides in grouped.items():
        if len(sides) < 2:
            continue
        out.append((mid, sides[0], sides[1]))
    return sorted(out, key=lambda g: g[0])


def _winner_loser(left: dict, right: dict) -> tuple[Optional[int], Optional[int]]:
    p1, p2 = _pts(left), _pts(right)
    r1, r2 = _rid(left), _rid(right)
    if p1 is None or p2 is None or r1 is None or r2 is None:
        return None, None
    # Unplayed / 0-0 games stay undecided.
    if p1 == 0 and p2 == 0:
        return None, None
    if p1 > p2:
        return r1, r2
    if p2 > p1:
        return r2, r1
    return None, None


def derive_bracket_from_matchups(
    matchups_by_week: Dict[Any, Sequence[dict]],
    playoff_week_start: int,
    *,
    kind: str = "winners",
    max_rounds: int = 4,
) -> List[Dict[str, Any]]:
    """Build bracket rounds from playoff-week matchup rows.

    ``kind`` other than ``winners`` returns [] (consolation is not derived).
    """
    if str(kind or "winners").lower() != "winners":
        return []
    try:
        start = int(playoff_week_start)
    except (TypeError, ValueError):
        return []
    if start <= 0:
        return []

    weeks = []
    for raw in (matchups_by_week or {}):
        try:
            week = int(raw)
        except (TypeError, ValueError):
            continue
        if start <= week < start + max_rounds:
            weeks.append(week)
    weeks = sorted(set(weeks))

    out: List[Dict[str, Any]] = []
    for i, week in enumerate(weeks):
        rows = matchups_by_week.get(week) or matchups_by_week.get(str(week)) or []
        for mid, left, right in pair_matchup_sides(rows):
            winner, loser = _winner_loser(left, right)
            out.append({
                "r": i + 1,
                "m": mid,
                "t1": _rid(left),
                "t2": _rid(right),
                "t1_from": None,
                "t2_from": None,
                "w": winner,
                "l": loser,
                "derived": True,
            })
    return out


def project_bracket_from_seeds(
    seed_roster_ids: Sequence[Any],
    *,
    playoff_teams: int = 6,
) -> List[Dict[str, Any]]:
    """First-round pairings from standings order when playoff weeks are empty.

    Standard field: 4 teams (1v4, 2v3), 6 teams (3v6 and 4v5; 1 and 2 bye),
    8 teams (1v8, 4v5, 2v7, 3v6). Other sizes pair the bottom of the field
    and leave the top as byes.
    """
    seeds: List[int] = []
    for raw in seed_roster_ids or []:
        try:
            seeds.append(int(raw))
        except (TypeError, ValueError):
            continue
    try:
        n = int(playoff_teams or 0)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        n = len(seeds)
    seeds = seeds[:n]
    if len(seeds) < 2:
        return []

    # Number of first-round games = field size minus next power-of-two byes.
    import math
    bracket = 1 << int(math.ceil(math.log2(max(len(seeds), 2))))
    byes = bracket - len(seeds)
    playing = seeds[byes:]
    if len(playing) < 2:
        return []
    games = []
    lo, hi = 0, len(playing) - 1
    mid = 1
    while lo < hi:
        games.append({
            "r": 1,
            "m": mid,
            "t1": playing[lo],
            "t2": playing[hi],
            "t1_from": None,
            "t2_from": None,
            "w": None,
            "l": None,
            "derived": True,
            "projected": True,
        })
        mid += 1
        lo += 1
        hi -= 1
    return games


def derive_or_project_bracket(
    *,
    matchups_by_week: Optional[Dict[Any, Sequence[dict]]] = None,
    playoff_week_start: int = 15,
    seed_roster_ids: Optional[Sequence[Any]] = None,
    playoff_teams: int = 6,
    kind: str = "winners",
) -> List[Dict[str, Any]]:
    """Prefer real playoff-week games; otherwise project the first round."""
    actual = derive_bracket_from_matchups(
        matchups_by_week or {}, playoff_week_start, kind=kind,
    )
    if actual:
        return actual
    if str(kind or "winners").lower() != "winners":
        return []
    return project_bracket_from_seeds(seed_roster_ids or [], playoff_teams=playoff_teams)


# ======================================================================
# From utils/playoff_picture.py
# ======================================================================

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
    ``ties``, ``pf``, ``division``, and ``div_record`` (a ``(w, l, t)`` record
    vs division opponents; seeding breaks overall-record ties by division
    win% before PF, matching the standings tables). ``total_regular_weeks`` is the number of
    regular-season games each team plays (``playoff_week_start - 1``).
    ``bye_spots`` defaults to the standard bracket byes for ``playoff_spots``.

    When 2+ distinct ``division`` ids are present, seeding follows division
    winners then wild cards (same rule as playoff scenarios / standings).

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
        try:
            div = int(t.get("division") or 0)
        except (TypeError, ValueError):
            div = 0
        ts.append({
            "id": t.get("id"),
            "name": t.get("name", ""),
            "wins": w, "losses": l, "ties": ti,
            "pf": float(t.get("pf", 0.0) or 0.0),
            "division": div,
            "div_record": t.get("div_record"),
            "games_left": gl,
            "max_wins": w + gl,
        })


    use_div = len({t["division"] for t in ts if t["division"]}) >= 2
    # Seed by wins/PF, or division winners + wild cards when divisions exist.
    order = playoff_seed_order(ts)
    ts = [ts[i] for i in order]
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

    def _in_under(win_map: Dict[Any, int], self_id) -> bool:
        """Whether ``self_id`` is inside the playoff field under ``win_map``
        wins, using the same seeding rules as the live standings."""
        hypo = []
        for o in ts:
            hypo.append({
                "wins": win_map.get(o["id"], o["wins"]),
                "ties": o["ties"],
                "pf": o["pf"],
                "pa": 0.0,
                "division": o["division"],
                "div_record": o.get("div_record"),
                "id": o["id"],
            })
        seeds = assign_playoff_seeds(hypo)
        for i, o in enumerate(hypo):
            if o["id"] == self_id:
                return seeds[i] <= spots
        return False

    # Wins at the playoff line, for games-back and comfort.
    cut_in_wins = ts[spots - 1]["wins"] if spots >= 1 else 0
    cut_in_losses = ts[spots - 1]["losses"] if spots >= 1 else 0
    first_out = ts[spots] if n > spots else None

    for t in ts:
        if use_div:
            # Safe clinch / elim under division seeding: win-out / lose-out
            # snapshots re-seeded the same way the standings page does.
            best_wins = {o["id"]: (o["max_wins"] if o["id"] == t["id"] else o["wins"])
                         for o in ts}
            worst_wins = {o["id"]: (o["wins"] if o["id"] == t["id"] else o["max_wins"])
                         for o in ts}
            even_wins = {o["id"]: o["max_wins"] for o in ts}
            clinched_playoff = _in_under(worst_wins, t["id"])
            eliminated = not _in_under(best_wins, t["id"])
            controls = _in_under(even_wins, t["id"]) and not clinched_playoff
            # Bye clinch: win floor still locks a top-``bye_spots`` seed under
            # division rules when every rival wins out against this team's floor.
            if bye_spots > 0:
                bye_worst = {o["id"]: (o["wins"] if o["id"] == t["id"] else o["max_wins"])
                             for o in ts}
                hypo = [{"wins": bye_worst[o["id"]], "ties": o["ties"], "pf": o["pf"],
                         "division": o["division"], "div_record": o.get("div_record"),
                         "id": o["id"]} for o in ts]
                seeds = assign_playoff_seeds(hypo)
                self_seed = next(seeds[i] for i, o in enumerate(hypo) if o["id"] == t["id"])
                clinched_bye = self_seed <= bye_spots
            else:
                clinched_bye = False
        else:
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


# ======================================================================
# From utils/all_play.py
# ======================================================================

"""All-play / luck analysis for standings.

Pure computation so it can be unit-tested without the app. Given each finalized
week's scores per team, computes:

  - all-play record: your W-L if you had played every other team every week
    (immune to who your actual opponent was), and the derived all-play win %.
  - expected wins: all_play_pct * games played -- how many wins your scoring
    "deserved" against an average schedule.
  - luck delta: actual wins minus expected wins. Positive = luckier than your
    scoring warranted; negative = unlucky.
  - expected seed: standings rank by all-play (1 = best), vs the real seed.

Ties within a week are split (0.5 win / 0.5 loss vs each equal-scoring team),
so all-play wins can be fractional.
"""
import math


def all_play_analysis(
    weekly_scores: Dict[int, Dict[str, float]],
    actual_wins: Dict[str, float],
) -> Dict[str, dict]:
    """
    Args:
        weekly_scores: {week: {team: score}} for finalized weeks only.
        actual_wins: {team: actual head-to-head wins so far} (ties count 0.5).

    Returns {team: {
        all_play_wins, all_play_losses, all_play_pct, games,
        expected_wins, actual_wins, luck_delta, expected_seed, actual_rank
    }} for every team seen. Empty dict when there are no weeks.
    """
    # Normalize defensively.  Callers normally do this while constructing the
    # weekly map, but this public helper should never compare NaN/inf (whose
    # ordering semantics would silently turn bad input into losses or ties).
    valid_weeks: Dict[int, Dict[str, float]] = {}
    for week, scores in (weekly_scores or {}).items():
        clean = {}
        for team, score in (scores or {}).items():
            try:
                value = float(score)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(value):
                clean[team] = value
        # All-play has no meaning without an opponent comparison.
        if len(clean) >= 2:
            valid_weeks[week] = clean

    # Collect the full team set across valid weeks (a team missing from one
    # week, e.g. a bye in odd leagues, is simply not scored that week).
    teams = set()
    for wk in valid_weeks.values():
        teams.update(wk.keys())
    if not teams:
        return {}

    ap_wins = {t: 0.0 for t in teams}
    ap_losses = {t: 0.0 for t in teams}
    weeks_played = {t: 0 for t in teams}

    for wk in valid_weeks.values():
        rows = list(wk.items())
        for t, s in rows:
            weeks_played[t] += 1
            for u, s2 in rows:
                if u == t:
                    continue
                if s > s2:
                    ap_wins[t] += 1.0
                elif s < s2:
                    ap_losses[t] += 1.0
                else:  # tie: split
                    ap_wins[t] += 0.5
                    ap_losses[t] += 0.5

    out: Dict[str, dict] = {}
    for t in teams:
        w, l = ap_wins[t], ap_losses[t]
        total = w + l
        pct = (w / total) if total > 0 else 0.0
        games = weeks_played[t]
        exp_w = pct * games
        try:
            act_w = float(actual_wins[t])
            if not math.isfinite(act_w):
                raise ValueError("non-finite actual wins")
        except (KeyError, TypeError, ValueError, OverflowError):
            act_w = None
        out[t] = {
            "all_play_wins": round(w, 1),
            "all_play_losses": round(l, 1),
            "all_play_pct": round(pct, 4),
            "games": games,
            "expected_wins": round(exp_w, 1),
            "actual_wins": act_w,
            "luck_delta": round(act_w - exp_w, 1) if act_w is not None else None,
        }

    # Expected seed: rank by all-play pct (desc), tie-broken by all-play wins.
    order = sorted(out.keys(), key=lambda t: (-out[t]["all_play_pct"], -out[t]["all_play_wins"]))
    for i, t in enumerate(order):
        out[t]["expected_seed"] = i + 1

    # Actual seed: rank by actual wins (desc). Only meaningful as a comparison
    # point; the caller usually already has the real standings order.
    order_actual = sorted(
        (t for t in out if out[t]["actual_wins"] is not None),
        key=lambda t: -out[t]["actual_wins"],
    )
    for i, t in enumerate(order_actual):
        out[t]["actual_rank"] = i + 1

    return out


def luck_label(luck_delta: float, threshold: float = 1.0) -> str:
    """'Lucky' / 'Unlucky' / '' from a luck delta, with a neutral dead zone."""
    if luck_delta >= threshold:
        return "Lucky"
    if luck_delta <= -threshold:
        return "Unlucky"
    return ""


# ======================================================================
# From utils/seed_series.py
# ======================================================================

"""Weekly standings-seed series for the team modal's Seed Movement chart.

Pure computation (pandas only) so it can be unit-tested without the app.
Seed is the standings rank through each finalized week by cumulative
(wins, points-for), the same ordering used for week-over-week movement.
"""


def seed_series_for(df_weekly, owner) -> list:
    """Return [(week, seed)] for one owner across finalized weeks.

    Best-effort: any failure (or no usable data) returns [].
    """
    try:
        cols = getattr(df_weekly, "columns", [])
        fin = df_weekly[df_weekly["finalized"] == True] if "finalized" in cols else df_weekly
        if fin is None or fin.empty:
            return []
        if "win" not in cols or "points" not in cols:
            return []
        weeks = sorted(int(w) for w in fin["week"].unique())
        out = []
        for wk in weeks:
            sub = fin[fin["week"] <= wk]
            agg = (
                sub.groupby("owner")
                .agg(W=("win", "sum"), PF=("points", "sum"))
                .reset_index()
                .sort_values(by=["W", "PF"], ascending=[False, False])
                .reset_index(drop=True)
            )
            seeds = {str(r["owner"]): i + 1 for i, r in agg.iterrows()}
            s = seeds.get(str(owner))
            if s is not None:
                out.append((wk, s))
        return out
    except Exception:
        return []


# ======================================================================
# From utils/season_review.py
# ======================================================================

"""Per-team season-in-review summary.

Pure computation (no pandas/app imports) so it unit-tests cleanly. Given one
team's finalized weekly rows plus a few league-context numbers, it derives the
headline facts for a "Season in Review" card: record, scoring, best/worst week,
longest win streak, and the luck read (all-play record, luck delta, expected vs
actual seed) that the all-play analysis already produced.
"""



def season_review(
    weekly: List[dict],
    all_play_entry: Optional[dict] = None,
    finish_rank: Optional[int] = None,
    num_teams: Optional[int] = None,
    pf_rank: Optional[int] = None,
) -> Dict:
    """
    Args:
        weekly: finalized weekly rows for ONE team, each {week, points, win},
            where win is 1 (win), 0 (loss) or 0.5 (tie).
        all_play_entry: this team's entry from all_play_analysis (optional).
        finish_rank: the team's actual standings rank (1 = first).
        num_teams: league size.
        pf_rank: the team's rank by points-for (1 = most points).

    Returns {} when there are no finalized weeks, else a dict of review facts.
    """
    rows = [w for w in (weekly or []) if w.get("points") is not None]
    games = len(rows)
    if games == 0:
        return {}

    pts = [float(w.get("points") or 0) for w in rows]
    wins = sum(1 for w in rows if float(w.get("win") or 0) >= 1)
    ties = sum(1 for w in rows if 0 < float(w.get("win") or 0) < 1)
    losses = games - wins - ties

    # Best / worst scoring week.
    best = max(rows, key=lambda w: float(w.get("points") or 0))
    worst = min(rows, key=lambda w: float(w.get("points") or 0))

    # Longest win streak across the season (in week order).
    ordered = sorted(rows, key=lambda w: int(w.get("week") or 0))
    longest = cur = 0
    for w in ordered:
        if float(w.get("win") or 0) >= 1:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0

    out: Dict = {
        "games": games,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "record": f"{wins}-{losses}" + (f"-{ties}" if ties else ""),
        "points_for": round(sum(pts), 1),
        "avg_points": round(sum(pts) / games, 1),
        "best_week": {"week": int(best.get("week") or 0), "points": round(float(best.get("points") or 0), 1)},
        "worst_week": {"week": int(worst.get("week") or 0), "points": round(float(worst.get("points") or 0), 1)},
        "longest_win_streak": longest,
        "finish_rank": finish_rank,
        "num_teams": num_teams,
        "pf_rank": pf_rank,
    }

    if all_play_entry:
        out["all_play_record"] = (
            f"{all_play_entry.get('all_play_wins', 0):.0f}-{all_play_entry.get('all_play_losses', 0):.0f}"
        )
        out["luck_delta"] = all_play_entry.get("luck_delta")
        out["expected_seed"] = all_play_entry.get("expected_seed")

    return out


# ======================================================================
# From utils/history_seasons.py
# ======================================================================

"""Pure history-season selection helper.

Extracted from app.py so the default-season logic can be unit-tested without
the pandas/DB stack. (The cached, league-chain-traversing
``get_available_history_seasons`` stays in app.py; only this pure selector moves.)
"""



def get_default_history_season(available_seasons: List[int], current_season: int) -> int:
    """
    Default to the most recent completed season, not the current season.
    If there is no prior season, fall back to the newest available season.
    """
    available = sorted({int(s) for s in available_seasons if s}, reverse=True)
    if not available:
        return int(current_season)

    past = [s for s in available if s < int(current_season)]
    if past:
        return past[0]

    return available[0]
