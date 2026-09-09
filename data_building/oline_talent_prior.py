"""
Open-data offensive-line *talent prior* — a projection, not a measurement.

This module estimates how much an O-line's *offseason roster change* should
nudge last season's realized (opponent-adjusted) prior before that prior is
blended with current-season play-by-play. It does not replace the results
pipeline in oline_ratings.py; it only adjusts the prior term that feeds
`_regress_to_prior`. Once current-season sample grows, the existing n_cur/K
blend washes the adjustment out. That is intentional: by mid-season the grade
is again a measurement of this year's plays.

Open-data provenance (nothing licensed)
---------------------------------------
  * Prior-season snap counts — nflverse PFR snap_counts release
    (github.com/nflverse/nflverse-data/releases/tag/snap_counts).
  * Current vs prior roster — nflverse weekly_rosters (week 1, the offseason
    picture) with seasonal rosters as fallback
    (releases/tag/weekly_rosters, releases/tag/rosters).
  * Draft capital — nflverse draft_picks release, valued with Chase Stuart's
    public expected-AV chart redistributed as the `stuart` column of
    github.com/nflverse/nfldata/blob/master/data/draft_values.csv.
    That CSV also carries a `pff` column; it is never read.
  * Player id join — nflverse players file (PFR id ↔ GSIS id). Snap counts
    identify linemen by PFR id; rosters identify them by GSIS. The join is
    identity-only; the players file's `pff_id` column is ignored.
  * Veteran additions/losses — inferred from roster team changes, weighted by
    prior-season OL snaps. That captures free agency, trades, and cuts without
    a licensed transaction feed.

Coaching / scheme change is omitted: there is no clean, redistributable
encoding of OC / OL-coach turnover that we can stand behind, and inventing
one would pretend to a precision we don't have.

This prior is optional and labeled. `build_oline_ratings(use_talent_prior=False)`
(the default) never calls it; the results-based composite is unchanged.

A 2022-2025 weeks 1-4 -> rest-of-season backtest (see
oline_backtest.sweep_talent_prior) found the residual is a wash vs last
season's prior: equal-mix W=0.45 scored 0.3476 against 0.3474. That is
not a real win, so the flag stays off.
"""
from __future__ import annotations

import math
from collections import defaultdict

# Same codes oline_ratings uses, plus PFR abbreviations that show up on
# snap-count / draft-pick dumps (GNB, KAN, NWE, ...).
_TEAM_ALIAS = {
    "JAC": "JAX", "LA": "LAR", "STL": "LAR", "OAK": "LV", "SD": "LAC",
    "WSH": "WAS", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
    "GNB": "GB", "GBP": "GB", "KAN": "KC", "KCC": "KC", "NWE": "NE", "NEP": "NE",
    "NOR": "NO", "SFO": "SF", "TAM": "TB", "TBB": "TB", "LVR": "LV",
    "SDG": "LAC", "RAM": "LAR", "OTI": "TEN", "RAI": "LV",
}


def _norm_team(t) -> str:
    t = (str(t) or "").upper().strip()
    if t in ("NONE", "NAN", ""):
        return ""
    return _TEAM_ALIAS.get(t, t)


def _f(v):
    try:
        if v is None:
            return None
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None

# PFR / nflverse offensive-line position tokens. Long-snappers and tight ends
# are excluded: LS is not an OL grade, and TE snaps are mostly receiving.
_OL_POS = frozenset({
    "C", "G", "T", "OT", "OG", "OL", "IOL",
    "LT", "RT", "LG", "RG", "C-G", "G-C", "T-G", "G-T",
})
_OL_CATEGORY = frozenset({"OL", "OLINE", "OFFENSIVE LINE"})

# Direct nflverse release assets — same fallback pattern as play-by-play.
_SNAP_URLS = (
    "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{year}.parquet",
)
_ROSTER_WEEKLY_URLS = (
    "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{year}.parquet",
)
_ROSTER_SEASONAL_URLS = (
    "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{year}.parquet",
)
_DRAFT_PICKS_URLS = (
    "https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.parquet",
)
# Public Chase Stuart expected-AV chart. Do not use the `pff` column.
_DRAFT_VALUES_URLS = (
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/draft_values.csv",
)
# nflverse players file is the public GSIS ↔ PFR id crosswalk. Snap counts
# key on PFR ids; weekly/seasonal rosters key on GSIS. Without this join,
# returning-starter continuity is always zero. PFF ids in that file are ignored.
_PLAYERS_URLS = (
    "https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet",
)

# Exponential fallback calibrated to Stuart pick-1 = 34.6 (Football Perspective).
# Used only when the CSV cannot be loaded. Not a PFF chart.
_STUART_PICK1 = 34.6
_STUART_DECAY = 0.035


def _is_ol_pos(pos, category=None) -> bool:
    cat = str(category or "").upper().strip()
    if cat in _OL_CATEGORY:
        return True
    p = str(pos or "").upper().strip()
    if not p or p in ("LS", "TE", "FB", "QB", "RB", "WR"):
        return False
    if p in _OL_POS:
        return True
    tokens = {t for t in p.replace("-", "/").split("/") if t}
    return bool(tokens & _OL_POS)


def stuart_pick_value(pick, chart=None) -> float:
    """Public expected-AV value of an overall draft pick.

    `chart` is {pick: stuart_value} from nfldata draft_values.csv. The PFF
    column of that file is never accepted here. Without a chart, fall back to
    an exponential calibrated to Stuart's pick-1 value of 34.6.
    """
    try:
        p = int(pick)
    except (TypeError, ValueError):
        return 0.0
    if p < 1:
        return 0.0
    if chart:
        v = chart.get(p)
        if v is not None:
            try:
                return max(0.0, float(v))
            except (TypeError, ValueError):
                pass
    return max(0.0, _STUART_PICK1 * math.exp(-_STUART_DECAY * (p - 1)))


def _zscore(values):
    """Population z-score. Zero-variance or tiny n -> empty (caller skips)."""
    items = [(t, v) for t, v in values.items() if v is not None]
    if len(items) < 4:
        return {}
    mean = sum(v for _, v in items) / len(items)
    var = sum((v - mean) ** 2 for _, v in items) / len(items)
    sd = var ** 0.5
    if sd <= 1e-12:
        return {}
    return {t: (v - mean) / sd for t, v in items}


def _sd(values):
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return var ** 0.5


def talent_scores_from_components(
    continuity_by_team,
    draft_value_by_team,
    veteran_net_by_team,
    w_continuity=1.0 / 3.0,
    w_draft=1.0 / 3.0,
    w_veteran=1.0 / 3.0,
):
    """Blend z-scored components into a mean-0, sd-1 talent residual.

    Missing / zero-variance components are dropped and the remaining weights
    renormalized. Returns ({team: score}, {component: weight actually used}).
    """
    parts = []
    used = {}
    for name, raw, w in (
        ("continuity", continuity_by_team, w_continuity),
        ("draft", draft_value_by_team, w_draft),
        ("veteran_net", veteran_net_by_team, w_veteran),
    ):
        if w is None or w <= 0:
            continue
        z = _zscore(raw or {})
        if not z:
            continue
        parts.append((name, z, float(w)))
        used[name] = float(w)
    if not parts:
        return {}, {}
    wsum = sum(w for _, _, w in parts)
    if wsum <= 0:
        return {}, {}
    used = {k: v / wsum for k, v in used.items()}
    teams = set()
    for _, z, _ in parts:
        teams |= set(z)
    raw_score = {}
    for t in teams:
        s = 0.0
        for name, z, w in parts:
            s += (w / wsum) * z.get(t, 0.0)
        raw_score[t] = s
    # Re-standardize the blend so TALENT_PRIOR_W is in "talent SDs".
    zblend = _zscore(raw_score)
    return (zblend or raw_score), used


def shift_prior(prior_adj, talent_score, weight, higher_is_better=True, league=None):
    """Shift last-season prior by a standardized talent residual.

    prior' = prior + sign * weight * score * sd(prior)

    `weight` is in units of last-year cross-sectional SDs per 1 SD of talent.
    Positive talent_score means a better-looking offseason, so we add for
    higher-is-better metrics (line yards, success) and subtract for
    lower-is-better ones (pressure, sacks). Teams with no score are left
    unchanged. Empty inputs return a copy of the unshifted prior — the
    results pipeline then behaves exactly as today.
    """
    prior_adj = dict(prior_adj or {})
    talent_score = talent_score or {}
    if not talent_score or not weight:
        return prior_adj
    if league is not None:
        for t in talent_score:
            prior_adj.setdefault(t, league)
    if not prior_adj:
        return prior_adj
    sd = _sd(list(prior_adj.values()))
    if sd <= 1e-12:
        return prior_adj
    sign = 1.0 if higher_is_better else -1.0
    out = {}
    for t, p in prior_adj.items():
        s = talent_score.get(t, 0.0) or 0.0
        out[t] = p + sign * float(weight) * s * sd
    return out


def _pid(*candidates):
    for c in candidates:
        if c is None:
            continue
        s = str(c).strip()
        if s and s.lower() not in ("none", "nan", ""):
            return s
    return None


def ol_continuity_and_veteran_net(prior_snaps, current_team_by_player):
    """Snap-weighted returning continuity and veteran addition/loss.

    `prior_snaps` is an iterable of dicts
        {player_id, team, snaps}  (player_id is pfr_id / gsis_id / name)
    `current_team_by_player` maps the same player_id space -> current team.

    continuity[t] = returning_snaps / prior_ol_snaps on team t
    veteran_net[t] = incoming veterans' prior snaps - departing players' snaps

    A returning starter is any prior-season OL snap on a player whose current
    team is the same team; backups are automatically down-weighted by snaps.
    """
    current_team_by_player = current_team_by_player or {}
    prior_snaps_by_team = defaultdict(float)
    returning = defaultdict(float)
    lost = defaultdict(float)
    gained = defaultdict(float)
    # Track each player's prior (team, snaps) so a mover is a loss for A and
    # a gain for B, weighted by the snaps they actually played last year.
    by_player = defaultdict(lambda: defaultdict(float))  # pid -> team -> snaps

    for row in prior_snaps or []:
        pid = _pid(row.get("player_id"), row.get("pfr_id"), row.get("gsis_id"),
                   row.get("name"))
        team = _norm_team(row.get("team"))
        snaps = _f(row.get("snaps"))
        if not pid or not team or not snaps or snaps <= 0:
            continue
        by_player[pid][team] += snaps
        prior_snaps_by_team[team] += snaps

    for pid, team_snaps in by_player.items():
        now = _norm_team(current_team_by_player.get(pid)) if pid in current_team_by_player else ""
        # A player can (rarely) have snaps for two teams last year; credit
        # each prior team separately.
        for prior_team, snaps in team_snaps.items():
            if now and now == prior_team:
                returning[prior_team] += snaps
            else:
                lost[prior_team] += snaps
                if now:
                    gained[now] += snaps

    continuity = {}
    veteran_net = {}
    detail = {}
    teams = set(prior_snaps_by_team) | set(gained) | set(current_team_by_player.values())
    teams = {_norm_team(t) for t in teams if t}
    for t in teams:
        tot = prior_snaps_by_team.get(t, 0.0)
        ret = returning.get(t, 0.0)
        continuity[t] = (ret / tot) if tot > 0 else None
        veteran_net[t] = gained.get(t, 0.0) - lost.get(t, 0.0)
        detail[t] = {
            "continuity": None if continuity[t] is None else round(continuity[t], 4),
            "returning_snaps": round(ret, 1),
            "prior_snaps": round(tot, 1),
            "veteran_gained_snaps": round(gained.get(t, 0.0), 1),
            "veteran_lost_snaps": round(lost.get(t, 0.0), 1),
            "veteran_net": round(veteran_net[t], 1),
        }
    return continuity, veteran_net, detail


def ol_draft_value(draft_picks, pick_chart=None):
    """Sum of Stuart pick-value for OL drafted by each team."""
    out = defaultdict(float)
    n_picks = defaultdict(int)
    for row in draft_picks or []:
        if not _is_ol_pos(row.get("position"), row.get("category")):
            continue
        team = _norm_team(row.get("team"))
        if not team:
            continue
        val = stuart_pick_value(row.get("pick"), pick_chart)
        out[team] += val
        n_picks[team] += 1
    detail = {t: {"draft_value": round(v, 3), "n_ol_picks": n_picks[t]}
              for t, v in out.items()}
    return dict(out), detail


def compute_talent_scores(
    prior_snaps,
    current_team_by_player,
    draft_picks,
    pick_chart=None,
    w_continuity=1.0 / 3.0,
    w_draft=1.0 / 3.0,
    w_veteran=1.0 / 3.0,
):
    """Pure computation of the labeled talent residual from already-parsed lists.

    Returns (score_by_team, detail_by_team, used_weights). score is z-scored.
    """
    continuity, veteran_net, cdetail = ol_continuity_and_veteran_net(
        prior_snaps, current_team_by_player)
    draft_val, ddetail = ol_draft_value(draft_picks, pick_chart)
    # Teams with no OL pick get draft_value 0 so z-scoring has a real floor.
    for t in set(continuity) | set(veteran_net):
        draft_val.setdefault(t, 0.0)
        ddetail.setdefault(t, {"draft_value": 0.0, "n_ol_picks": 0})
    scores, used = talent_scores_from_components(
        continuity, draft_val, veteran_net,
        w_continuity=w_continuity, w_draft=w_draft, w_veteran=w_veteran,
    )
    detail = {}
    teams = set(cdetail) | set(ddetail) | set(scores)
    for t in teams:
        row = dict(cdetail.get(t, {}))
        row.update(ddetail.get(t, {}))
        row["score"] = None if t not in scores else round(scores[t], 4)
        detail[t] = row
    return scores, detail, used


# ---------------------------------------------------------------------------
# Frame loaders. Each returns a DataFrame or None; never raises to the caller.
# ---------------------------------------------------------------------------

def _read_first_url(urls, pd, year=None):
    last_err = None
    for url in urls:
        u = url.format(year=year) if year is not None and "{year}" in url else url
        try:
            if u.endswith(".csv"):
                d = pd.read_csv(u)
            else:
                d = pd.read_parquet(u)
            if d is not None and not d.empty:
                return d
        except Exception as e:
            last_err = e
            print(f"[oline_talent_prior] {u.rsplit('/', 1)[-1]} -> {e}")
    if last_err is not None:
        print(f"[oline_talent_prior] all URLs failed ({last_err})")
    return None


def _load_snaps(year, pd, nfl=None):
    if nfl is not None:
        try:
            d = nfl.import_snap_counts([year])
            if d is not None and not d.empty:
                return d
        except Exception as e:
            print(f"[oline_talent_prior] nfl_data_py snaps {year} failed ({e})")
    return _read_first_url(_SNAP_URLS, pd, year)


def _load_weekly_roster(year, pd, nfl=None):
    if nfl is not None:
        try:
            d = nfl.import_weekly_rosters([year])
            if d is not None and not d.empty:
                return d
        except Exception as e:
            print(f"[oline_talent_prior] nfl_data_py weekly roster {year} failed ({e})")
    return _read_first_url(_ROSTER_WEEKLY_URLS, pd, year)


def _load_seasonal_roster(year, pd, nfl=None):
    if nfl is not None:
        try:
            d = nfl.import_seasonal_rosters([year])
            if d is not None and not d.empty:
                return d
        except Exception as e:
            print(f"[oline_talent_prior] nfl_data_py seasonal roster {year} failed ({e})")
    return _read_first_url(_ROSTER_SEASONAL_URLS, pd, year)


def _load_draft_picks(year, pd, nfl=None):
    d = None
    if nfl is not None:
        try:
            d = nfl.import_draft_picks([year])
        except Exception as e:
            print(f"[oline_talent_prior] nfl_data_py draft_picks failed ({e})")
            d = None
    if d is None or getattr(d, "empty", True):
        d = _read_first_url(_DRAFT_PICKS_URLS, pd)
        if d is not None and "season" in d.columns:
            sn = pd.to_numeric(d["season"], errors="coerce")
            d = d[sn == year]
    return d


def _load_id_xwalk(pd, nfl=None):
    """PFR id / display name -> GSIS id from the public nflverse players file.

    Snap counts identify players by PFR id; rosters identify them by GSIS.
    This crosswalk is how we join the two. The players file also carries a
    `pff_id` column — it is never read.
    """
    d = None
    if nfl is not None:
        try:
            d = nfl.import_players()
        except Exception as e:
            print(f"[oline_talent_prior] nfl_data_py players failed ({e})")
            d = None
    if d is None or getattr(d, "empty", True):
        d = _read_first_url(_PLAYERS_URLS, pd)
    if d is None or getattr(d, "empty", True):
        print("[oline_talent_prior] no players crosswalk; snap/roster match will be weak")
        return {}
    pfr_col = next((c for c in ("pfr_id", "pfr_player_id") if c in d.columns), None)
    gsis_col = next((c for c in ("gsis_id", "player_id") if c in d.columns), None)
    name_col = next((c for c in ("display_name", "full_name", "player_name")
                     if c in d.columns), None)
    if not gsis_col:
        return {}
    xwalk = {}
    n = len(d)
    gsiss = d[gsis_col].tolist()
    pfrs = d[pfr_col].tolist() if pfr_col else [None] * n
    names = d[name_col].tolist() if name_col else [None] * n
    for i in range(n):
        gsis = _pid(gsiss[i])
        if not gsis:
            continue
        pfr = _pid(pfrs[i])
        if pfr and pfr not in xwalk:
            xwalk[pfr] = gsis
        name = _pid(names[i])
        if name:
            key = name.lower()
            xwalk.setdefault(key, gsis)
    return xwalk


def _canonical_pid(pfr, gsis, name, xwalk):
    """Prefer GSIS so snap rows and roster rows share a key."""
    gsis = _pid(gsis)
    if gsis:
        return gsis
    pfr = _pid(pfr)
    name = _pid(name)
    if xwalk:
        if pfr and pfr in xwalk:
            return xwalk[pfr]
        if name and name.lower() in xwalk:
            return xwalk[name.lower()]
    return pfr or name


def _load_stuart_chart(pd, nfl=None):
    """Return {pick: stuart_value}. Never reads the PFF column."""
    d = None
    if nfl is not None:
        try:
            d = nfl.import_draft_values()
        except Exception as e:
            print(f"[oline_talent_prior] nfl_data_py draft_values failed ({e})")
            d = None
    if d is None or getattr(d, "empty", True):
        d = _read_first_url(_DRAFT_VALUES_URLS, pd)
    if d is None or getattr(d, "empty", True):
        return None
    if "pick" not in d.columns or "stuart" not in d.columns:
        print("[oline_talent_prior] draft_values missing pick/stuart columns; using exponential fallback")
        return None
    chart = {}
    picks = d["pick"].tolist()
    vals = d["stuart"].tolist()
    for p, v in zip(picks, vals):
        try:
            chart[int(p)] = float(v)
        except (TypeError, ValueError):
            continue
    return chart or None


def _parse_prior_ol_snaps(snap_df, pd, xwalk=None):
    """-> list[{player_id, team, snaps}] for regular-season OL snaps.

    `player_id` is GSIS when the public players crosswalk can resolve the
    PFR id (the key snap counts actually carry).
    """
    if snap_df is None or getattr(snap_df, "empty", True):
        return []
    d = snap_df
    if "game_type" in d.columns:
        d = d[d["game_type"].astype(str).str.upper() == "REG"]
    pos_col = next((c for c in ("position", "pos") if c in d.columns), None)
    if pos_col:
        d = d[d[pos_col].map(_is_ol_pos)]
    pid_col = next((c for c in ("pfr_player_id", "pfr_id", "gsis_id", "player_id")
                    if c in d.columns), None)
    name_col = next((c for c in ("player", "pfr_player_name", "player_name")
                     if c in d.columns), None)
    team_col = next((c for c in ("team", "recent_team") if c in d.columns), None)
    snap_col = next((c for c in ("offense_snaps", "offense") if c in d.columns), None)
    if not team_col or not snap_col:
        return []
    out = []
    teams = d[team_col].tolist()
    snaps = d[snap_col].tolist()
    pids = d[pid_col].tolist() if pid_col else [None] * len(d)
    names = d[name_col].tolist() if name_col else [None] * len(d)
    gsis_col = next((c for c in ("gsis_id",) if c in d.columns), None)
    gsiss = d[gsis_col].tolist() if gsis_col else [None] * len(d)
    for i in range(len(d)):
        pid = _canonical_pid(pids[i], gsiss[i], names[i], xwalk)
        team = _norm_team(teams[i])
        n = _f(snaps[i])
        if not pid or not team or not n or n <= 0:
            continue
        out.append({"player_id": pid, "team": team, "snaps": n})
    # Aggregate to player-team in case the frame is weekly.
    agg = defaultdict(float)
    for row in out:
        k = (row["player_id"], row["team"])
        agg[k] += row["snaps"]
    return [{"player_id": pid, "team": team, "snaps": s}
            for (pid, team), s in agg.items()]


def _parse_current_ol_teams(roster_df, pd, prefer_week=1):
    """-> ({player_id: team}, source_note). Keys include pfr, gsis, and name."""
    if roster_df is None or getattr(roster_df, "empty", True):
        return {}, "missing"
    d = roster_df
    if "game_type" in d.columns:
        gt = d["game_type"].astype(str).str.upper()
        # Some weekly roster dumps leave game_type blank; don't drop those.
        d = d[gt.isin(["REG", "NAN", "NONE", ""]) | gt.isna()]
        if d.empty:
            d = roster_df
    source = "seasonal"
    if "week" in d.columns:
        wk = pd.to_numeric(d["week"], errors="coerce")
        d1 = d[wk == prefer_week]
        if d1.empty:
            d1 = d[wk >= 1]
            if not d1.empty:
                min_wk = pd.to_numeric(d1["week"], errors="coerce").min()
                d1 = d1[pd.to_numeric(d1["week"], errors="coerce") == min_wk]
                source = f"week_{int(min_wk)}"
        else:
            source = f"week_{prefer_week}"
        if not d1.empty:
            d = d1
    pos_col = next((c for c in ("position", "pos", "depth_chart_position")
                    if c in d.columns), None)
    if pos_col:
        ol = d[d[pos_col].map(_is_ol_pos)]
        if not ol.empty:
            d = ol
    team_col = next((c for c in ("team", "recent_team", "club_code") if c in d.columns), None)
    if not team_col:
        return {}, "missing"
    pfr_col = next((c for c in ("pfr_id", "pfr_player_id") if c in d.columns), None)
    gsis_col = next((c for c in ("player_id", "gsis_id") if c in d.columns), None)
    name_col = next((c for c in ("player_name", "full_name", "football_name")
                     if c in d.columns), None)
    mapping = {}
    n = len(d)
    teams = d[team_col].tolist()
    pfrs = d[pfr_col].tolist() if pfr_col else [None] * n
    gsiss = d[gsis_col].tolist() if gsis_col else [None] * n
    names = d[name_col].tolist() if name_col else [None] * n
    for i in range(n):
        team = _norm_team(teams[i])
        if not team:
            continue
        for key in (
            _canonical_pid(pfrs[i], gsiss[i], names[i], None),
            _pid(gsiss[i]),
            _pid(names[i]),
            (_pid(names[i]) or "").lower() or None,
        ):
            if key and key not in mapping:
                mapping[key] = team
    return mapping, source


def _parse_draft_picks(draft_df, year, pd):
    if draft_df is None or getattr(draft_df, "empty", True):
        return []
    d = draft_df
    if "season" in d.columns:
        sn = pd.to_numeric(d["season"], errors="coerce")
        d = d[sn == year]
    team_col = next((c for c in ("team", "club_code") if c in d.columns), None)
    pick_col = next((c for c in ("pick", "overall", "overall_pick") if c in d.columns), None)
    pos_col = next((c for c in ("position", "pos") if c in d.columns), None)
    cat_col = "category" if "category" in d.columns else None
    if not team_col or not pick_col:
        return []
    out = []
    n = len(d)
    teams = d[team_col].tolist()
    picks = d[pick_col].tolist()
    poss = d[pos_col].tolist() if pos_col else [None] * n
    cats = d[cat_col].tolist() if cat_col else [None] * n
    for i in range(n):
        out.append({
            "team": _norm_team(teams[i]),
            "pick": picks[i],
            "position": poss[i],
            "category": cats[i],
        })
    return out


def load_talent_inputs(season, pd, nfl=None):
    """Load the open-data inputs for `season`'s offseason prior.

    Returns a dict with parsed lists, or None if the *required* pieces
    (prior-season snaps AND current-season roster) are missing. Draft is
    optional: without it the draft component is simply dropped.
    """
    xwalk = _load_id_xwalk(pd, nfl)
    snaps = _load_snaps(season - 1, pd, nfl)
    prior_snaps = _parse_prior_ol_snaps(snaps, pd, xwalk=xwalk)
    if not prior_snaps:
        print(f"[oline_talent_prior] no OL snaps for {season - 1}; skipping talent prior")
        return None

    roster, roster_source = _parse_current_ol_teams(
        _load_weekly_roster(season, pd, nfl), pd)
    if not roster:
        roster, roster_source = _parse_current_ol_teams(
            _load_seasonal_roster(season, pd, nfl), pd)
        if roster_source == "seasonal" or roster:
            roster_source = "seasonal"
    if not roster:
        print(f"[oline_talent_prior] no current OL roster for {season}; skipping talent prior")
        return None

    draft_picks = _parse_draft_picks(_load_draft_picks(season, pd, nfl), season, pd)
    chart = _load_stuart_chart(pd, nfl)
    return {
        "prior_snaps": prior_snaps,
        "current_team_by_player": roster,
        "draft_picks": draft_picks,
        "pick_chart": chart,
        "roster_source": roster_source,
        "n_prior_ol_rows": len(prior_snaps),
        "n_current_ol": len(roster),
        "n_draft_rows": len(draft_picks),
        "n_id_xwalk": len(xwalk),
    }


def compute_oline_talent_prior(
    season,
    pd=None,
    nfl=None,
    w_continuity=1.0 / 3.0,
    w_draft=1.0 / 3.0,
    w_veteran=1.0 / 3.0,
    inputs=None,
):
    """Load (unless `inputs` is given) and score the talent residual.

    Returns None when data is missing so callers fall back to last-season
    prior unchanged. Otherwise:
        {"scores": {team: float}, "detail": {team: dict},
         "weights_used": {...}, "meta": {...}}
    """
    if inputs is None:
        if pd is None:
            return None
        inputs = load_talent_inputs(season, pd, nfl)
    if not inputs:
        return None
    scores, detail, used = compute_talent_scores(
        inputs["prior_snaps"],
        inputs["current_team_by_player"],
        inputs.get("draft_picks") or [],
        pick_chart=inputs.get("pick_chart"),
        w_continuity=w_continuity,
        w_draft=w_draft,
        w_veteran=w_veteran,
    )
    if not scores:
        return None
    return {
        "scores": scores,
        "detail": detail,
        "weights_used": used,
        "meta": {
            "roster_source": inputs.get("roster_source"),
            "n_prior_ol_rows": inputs.get("n_prior_ol_rows"),
            "n_current_ol": inputs.get("n_current_ol"),
            "n_draft_rows": inputs.get("n_draft_rows"),
            "n_id_xwalk": inputs.get("n_id_xwalk"),
            "pick_chart": "stuart" if inputs.get("pick_chart") else "exponential_fallback",
            "sources": [
                "nflverse PFR snap_counts (prior season)",
                "nflverse weekly/seasonal rosters (week-1 current vs prior snaps)",
                "nflverse players file (PFR↔GSIS id join only; pff_id ignored)",
                "nflverse draft_picks + nfldata draft_values.stuart "
                "(Chase Stuart public expected-AV; PFF column ignored)",
            ],
            "omitted": (
                "Coaching/scheme change: no clean redistributable encoding of "
                "OC / OL-coach turnover."
            ),
        },
    }
