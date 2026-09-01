"""Shared Trends matching predicates and preseason feature extraction (pure).

Scout (current board players) and historical cohort matching MUST use the
same feature keys and the same AND/OR grouping:

* same logical group → OR
* different groups → AND

A missing feature does not match a required filter (unknown stays unknown).
This module does not scan parquet and does not enter ranking / Pick Score.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

from dashboard_services.historical.comps import extract_comp_query
from dashboard_services.historical.usage import USAGE_RATE_SPECS, VOLUME_USAGE_IDS
from dashboard_services.historical.definitions import (
    ADOT_BUCKETS,
    RYOE_BUCKETS,
    SKILL_POSITIONS,
    SNAP_RELIABLE_FLOOR,
    TRAJECTORY_SNAP_DOWN,
    TRAJECTORY_SNAP_UP,
    TRAJECTORY_TARGET_SHARE_DOWN,
    TRAJECTORY_TARGET_SHARE_UP,
    TRAJECTORY_WORKLOAD_DOWN,
    TRAJECTORY_WORKLOAD_UP,
    SNAP_PCT_UP_PTS,
    TARGET_SHARE_UP_PTS,
    WORKLOAD_CHANGE_CLIFFS,
    age_as_of_season_start,
    draft_capital_bucket,
    integer_age,
    normalize_team_abbr,
    offense_rank_bucket,
    value_bucket,
    _optional_float,
    _optional_int,
)


def matches_trend_filter(feats: Mapping[str, Any], spec: Any) -> bool:
    """AND/OR-free predicate for one trend row's match spec. Display only."""
    if not isinstance(spec, Mapping) or not spec:
        return True
    if spec.get("all"):
        return all(matches_trend_filter(feats, part) for part in spec.get("all") or [])
    field = spec.get("field")
    val = feats.get(field) if field else None
    if spec.get("null_as") is not None and val is None:
        val = spec.get("null_as")
    if "eq" in spec:
        want = spec.get("eq")
        return val == want or (val is not None and str(val) == str(want))
    if "in" in spec:
        options = list(spec.get("in") or [])
        return val in options or (val is not None and str(val) in {str(x) for x in options})
    if "gte" in spec:
        try:
            return val is not None and float(val) >= float(spec.get("gte"))
        except (TypeError, ValueError):
            return False
    if "lte" in spec:
        try:
            return val is not None and float(val) <= float(spec.get("lte"))
        except (TypeError, ValueError):
            return False
    if "between" in spec:
        bounds = spec.get("between") or []
        if len(bounds) < 2:
            return False
        try:
            number = float(val)
            return float(bounds[0]) <= number <= float(bounds[1])
        except (TypeError, ValueError):
            return False
    return False


def filter_group_id(spec: Mapping[str, Any], fallback: Any = None) -> str:
    """Logical group for OR-within / AND-across semantics."""
    group = spec.get("group") or spec.get("field") or fallback
    return str(group or "")


def group_filters(filters: Sequence[Mapping[str, Any]]) -> dict[str, list[dict]]:
    """Bucket filter specs by logical group. Empty / non-mapping entries drop."""
    groups: dict[str, list[dict]] = {}
    for i, spec in enumerate(filters or []):
        if not isinstance(spec, Mapping) or not spec:
            continue
        rec = dict(spec)
        gid = filter_group_id(rec, fallback=rec.get("id") or i)
        if not gid:
            continue
        groups.setdefault(gid, []).append(rec)
    return groups


def matches_filter_groups(feats: Mapping[str, Any], filters: Sequence[Mapping[str, Any]]) -> bool:
    """True when ``feats`` satisfies OR-within-group, AND-across-groups.

    No filters → False (a profile is not "all players"; it is a selected set).
    """
    groups = group_filters(filters)
    if not groups:
        return False
    return all(
        any(matches_trend_filter(feats, spec) for spec in specs)
        for specs in groups.values()
    )


def _volume_of(row: Mapping[str, Any], field: str) -> Optional[float]:
    """Outcome volume on this player-season row (not previous_season_*)."""
    if field == "touches":
        carries = _optional_float(row.get("carries"))
        recs = _optional_float(row.get("receptions"))
        if carries is None and recs is None:
            return _optional_float(row.get("touches"))
        return float((carries or 0.0) + (recs or 0.0))
    if field == "targets":
        return _optional_float(row.get("targets"))
    if field == "passing_attempts":
        return _optional_float(row.get("passing_attempts"))
    return _optional_float(row.get(field))


def trajectory_buckets(
    prev: Optional[Mapping[str, Any]],
    last: Optional[Mapping[str, Any]],
    *,
    position: Any = None,
    require_consecutive: bool = True,
    current_season: Any = None,
) -> dict[str, str]:
    """Bucket YoY pre-outcome change from two prior seasons.

    ``prev`` is S-2 actuals, ``last`` is S-1 actuals. Neither may be the
    season whose finish is the outcome. Non-consecutive seasons, a gap
    before ``current_season``, or a missing value → that key is omitted.
    """
    if not isinstance(prev, Mapping) or not isinstance(last, Mapping):
        return {}
    prev_s = _optional_int(prev.get("season"))
    last_s = _optional_int(last.get("season"))
    if prev_s is None or last_s is None:
        return {}
    if require_consecutive and last_s != prev_s + 1:
        return {}
    cur = _optional_int(current_season)
    if cur is not None and last_s != cur - 1:
        return {}
    pos = str(position or last.get("position") or prev.get("position") or "").upper()
    out: dict[str, str] = {}

    ts_last = _optional_float(last.get("target_share"))
    ts_prev = _optional_float(prev.get("target_share"))
    if ts_last is not None and ts_prev is not None:
        delta = ts_last - ts_prev
        if delta >= TARGET_SHARE_UP_PTS:
            out["target_share_change"] = TRAJECTORY_TARGET_SHARE_UP
        elif delta <= -TARGET_SHARE_UP_PTS:
            out["target_share_change"] = TRAJECTORY_TARGET_SHARE_DOWN

    snap_last = _optional_float(last.get("snap_pct"))
    snap_prev = _optional_float(prev.get("snap_pct"))
    if (
        snap_last is not None
        and snap_prev is not None
        and last_s >= SNAP_RELIABLE_FLOOR
        and prev_s >= SNAP_RELIABLE_FLOOR
    ):
        delta = snap_last - snap_prev
        if delta >= SNAP_PCT_UP_PTS:
            out["snap_pct_change"] = TRAJECTORY_SNAP_UP
        elif delta <= -SNAP_PCT_UP_PTS:
            out["snap_pct_change"] = TRAJECTORY_SNAP_DOWN

    cliff = WORKLOAD_CHANGE_CLIFFS.get(pos)
    if cliff:
        field, cutoff = cliff
        vol_last = _volume_of(last, field)
        vol_prev = _volume_of(prev, field)
        if vol_last is not None and vol_prev is not None:
            delta = vol_last - vol_prev
            if delta >= cutoff:
                out["workload_change"] = TRAJECTORY_WORKLOAD_UP
            elif delta <= -cutoff:
                out["workload_change"] = TRAJECTORY_WORKLOAD_DOWN
    return out


def extract_trend_features(row: Mapping[str, Any]) -> dict[str, Any]:
    """Compact preseason buckets used by Scout and historical cohort matching.

    Same keys as ``build_player_feature_index``. Missing dims are omitted.
    Same-season actuals / ADP / projections are not features.
    """
    if not isinstance(row, Mapping):
        return {}
    feats = extract_comp_query(row)
    if not feats:
        return {}
    pick = _optional_int(row.get("nfl_draft_pick"))
    if pick is None:
        pick = _optional_int(row.get("draft_pick"))
    if pick is not None and pick > 0:
        feats["nfl_draft_pick"] = pick
    age = integer_age(row.get("age"))
    if age is not None:
        feats["age"] = age
    count = _optional_int(row.get("prior_top12_count"))
    if count is not None:
        feats["prior_top12_count"] = count
    adot = value_bucket(row.get("previous_season_adot"), ADOT_BUCKETS)
    if adot:
        feats["adot"] = adot
    ryoe = value_bucket(
        row.get("previous_season_ngs_rush_yards_over_expected_per_att"),
        RYOE_BUCKETS,
    )
    if ryoe:
        feats["ryoe"] = ryoe
    for spec in USAGE_RATE_SPECS:
        spec_id = spec["id"]
        if spec_id not in VOLUME_USAGE_IDS:
            continue
        bucket = value_bucket(row.get(spec["field"]), spec["buckets"])
        if bucket:
            feats[spec_id] = bucket
    for key in (
        "target_share_change",
        "snap_pct_change",
        "workload_change",
    ):
        val = row.get(key)
        if isinstance(val, str) and val:
            feats[key] = val
    team = normalize_team_abbr(row.get("team") or row.get("nfl_team"))
    if team:
        feats["team"] = team
    rank = _optional_int(row.get("prior_offense_rank"))
    if rank is not None and rank > 0:
        feats["prior_offense_rank"] = rank
        bucket = offense_rank_bucket(rank)
        if bucket:
            feats["prior_offense_rank_bucket"] = bucket
    return feats


def _norm_player_name(name: Any) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def live_board_trend_features(player: Mapping[str, Any]) -> dict[str, Any]:
    """Scout buckets from a live board / identity row. No warehouse actuals.

    Draft capital comes from an explicit bucket or from NFL round/pick.
    Rookies (``years_experience`` / ``years_exp`` == 0) get prior_top12_count=0.
    Missing dims stay omitted. Does not invent usage or trajectory.
    """
    if not isinstance(player, Mapping):
        return {}
    pos = str(player.get("position") or player.get("pos") or "").upper()
    if pos not in SKILL_POSITIONS:
        return {}
    row: dict[str, Any] = {"position": pos}
    ye = _optional_int(
        player.get("years_experience")
        if player.get("years_experience") is not None
        else player.get("years_exp")
    )
    if ye is not None:
        row["years_experience"] = ye
        if ye == 0:
            row["prior_top12_count"] = 0
    cap = player.get("draft_capital_bucket") or player.get("draft_capital")
    if not cap:
        cap = draft_capital_bucket(
            player.get("draft_round") or player.get("nfl_draft_round"),
            player.get("draft_pick") or player.get("nfl_draft_pick"),
            undrafted=bool(player.get("undrafted")),
        )
    if cap:
        row["draft_capital_bucket"] = cap
    pick = _optional_int(
        player.get("nfl_draft_pick")
        if player.get("nfl_draft_pick") is not None
        else player.get("draft_pick")
    )
    if pick is not None and pick > 0:
        row["nfl_draft_pick"] = pick
    age = integer_age(player.get("age"))
    if age is None:
        age_f = _optional_float(player.get("age"))
        age = integer_age(age_f) if age_f is not None else None
    if age is not None:
        row["age"] = age
    count = _optional_int(player.get("prior_top12_count"))
    if count is not None:
        row["prior_top12_count"] = count
    team = normalize_team_abbr(
        player.get("team") or player.get("nfl_team") or player.get("actual_nfl_team")
    )
    if team:
        row["team"] = team
    rank = _optional_int(player.get("prior_offense_rank"))
    if rank is not None and rank > 0:
        row["prior_offense_rank"] = rank
    return extract_trend_features(row)


def live_class_preseason_profile(
    pick: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    upcoming_season: Any = None,
) -> dict[str, Any]:
    """Stub preseason row for a current-class draftee with no warehouse season."""
    pos = str(
        pick.get("position") or identity.get("position") or identity.get("pos") or ""
    ).upper()
    if pos not in SKILL_POSITIONS:
        return {}
    cap = draft_capital_bucket(pick.get("round") or pick.get("draft_round"), pick.get("pick") or pick.get("draft_pick"))
    rec: dict[str, Any] = {
        "position": pos,
        "years_experience": 0,
        "prior_top12_count": 0,
    }
    if cap:
        rec["draft_capital_bucket"] = cap
    overall = _optional_int(pick.get("pick") if pick.get("pick") is not None else pick.get("draft_pick"))
    if overall is not None and overall > 0:
        rec["nfl_draft_pick"] = overall
    age = age_as_of_season_start(
        identity.get("bDay") or identity.get("bday") or identity.get("birth_date"),
        upcoming_season,
    )
    if age is None:
        age_f = _optional_float(identity.get("age") or pick.get("age"))
        age = age_f
    if age is not None:
        rec["age"] = age
    team = normalize_team_abbr(
        pick.get("nfl_team") or pick.get("team") or identity.get("team")
    )
    if team:
        rec["team"] = team
    return rec


def stamp_live_draft_class_profiles(
    by_player: dict[str, Any],
    picks: Sequence[Mapping[str, Any]],
    players_index: Mapping[str, Any],
    *,
    upcoming_season: Any = None,
) -> int:
    """Add current-class draftees who have no warehouse preseason profile.

    Name + position must uniquely match (team breaks ties). Existing warehouse
    rows are left untouched. Returns how many stubs were inserted.
    """
    if not isinstance(by_player, dict) or not picks or not isinstance(players_index, Mapping):
        return 0
    by_key: dict[tuple[str, str], list[tuple[str, Mapping[str, Any]]]] = {}
    for pid, meta in players_index.items():
        sid = str(pid or "").strip()
        if not sid or sid in by_player or not isinstance(meta, Mapping):
            continue
        pos = str(meta.get("pos") or meta.get("position") or "").upper()
        if pos not in SKILL_POSITIONS:
            continue
        key = (_norm_player_name(meta.get("name") or meta.get("full_name")), pos)
        if not key[0]:
            continue
        by_key.setdefault(key, []).append((sid, meta))
    added = 0
    for pick in picks:
        if not isinstance(pick, Mapping):
            continue
        pos = str(pick.get("position") or "").upper()
        key = (_norm_player_name(pick.get("player_name") or pick.get("name")), pos)
        if not key[0] or pos not in SKILL_POSITIONS:
            continue
        cands = list(by_key.get(key) or [])
        if len(cands) != 1:
            team = str(pick.get("nfl_team") or pick.get("team") or "").upper()
            if team:
                cands = [
                    item for item in cands
                    if str((item[1] or {}).get("team") or "").upper() == team
                ]
        if len(cands) != 1:
            continue
        pid, ident = cands[0]
        if pid in by_player:
            continue
        rec = live_class_preseason_profile(pick, ident, upcoming_season=upcoming_season)
        if not rec.get("draft_capital_bucket"):
            continue
        by_player[pid] = rec
        added += 1
    return added


def apply_nfl_draft_pick_overlay(data: dict, pick_map: Mapping[str, Any]) -> int:
    """Stamp overall NFL pick onto cohort feats and preseason profiles.

    Pick is player-constant. Existing values win. Missing pick stays omitted
    (unknown does not match a pick-range filter).
    """
    if not isinstance(data, dict) or not isinstance(pick_map, Mapping) or not pick_map:
        return 0
    resolved: dict[str, int] = {}
    raw = pick_map.get("picks") if isinstance(pick_map.get("picks"), Mapping) else pick_map
    if not isinstance(raw, Mapping):
        return 0
    for pid, value in raw.items():
        pick = _optional_int(value)
        if pick is None or pick <= 0:
            continue
        resolved[str(pid)] = pick
    if not resolved:
        return 0
    stamped = 0
    index = data.get("cohort_index") if isinstance(data.get("cohort_index"), dict) else {}
    for obs in index.get("observations") or []:
        if not isinstance(obs, dict):
            continue
        pid = str(obs.get("pid") or "")
        pick = resolved.get(pid)
        if pick is None:
            continue
        feats = obs.get("feats")
        if not isinstance(feats, dict):
            feats = {}
            obs["feats"] = feats
        if feats.get("nfl_draft_pick") is None:
            feats["nfl_draft_pick"] = pick
            stamped += 1
    pre = data.get("preseason_profiles") if isinstance(data.get("preseason_profiles"), dict) else {}
    by_player = pre.get("by_player") if isinstance(pre.get("by_player"), dict) else {}
    for pid, rec in by_player.items():
        if not isinstance(rec, dict):
            continue
        pick = resolved.get(str(pid))
        if pick is None or rec.get("nfl_draft_pick") is not None:
            continue
        rec["nfl_draft_pick"] = pick
        stamped += 1
    return stamped


def matched_filter_labels(
    feats: Mapping[str, Any],
    filters: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Human labels of the selected filters this feature row actually hit."""
    labels: list[str] = []
    seen: set[str] = set()
    for spec in filters or []:
        if not isinstance(spec, Mapping):
            continue
        if not matches_trend_filter(feats, spec):
            continue
        label = str(spec.get("label") or spec.get("eq") or spec.get("field") or "").strip()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def scout_matching_players(
    board_features: Mapping[str, Any],
    filters: Sequence[Mapping[str, Any]],
) -> list[dict]:
    """Board players whose stamped trend feats match the selected profile.

    Authoritative Scout path — same predicates as historical cohort matching.
    ``board_features`` maps player id → feature dict. Display only.
    """
    if not isinstance(board_features, Mapping) or not filters:
        return []
    out: list[dict] = []
    for pid, feats in board_features.items():
        sid = str(pid or "").strip()
        if not sid or not isinstance(feats, Mapping):
            continue
        if not matches_filter_groups(feats, filters):
            continue
        out.append({
            "id": sid,
            "why": matched_filter_labels(feats, filters),
        })
    return out


def _cache_value(value: Any) -> tuple:
    """Orderable stand-in so mixed int/str filter specs can share a cache key."""
    if isinstance(value, tuple):
        return ("tuple",) + tuple(_cache_value(v) for v in value)
    if isinstance(value, list):
        return ("list",) + tuple(_cache_value(v) for v in value)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    if value is None:
        return ("none",)
    return ("str", str(value))


def canonical_filter_key(filters: Iterable[Mapping[str, Any]]) -> tuple:
    """Stable cache key. Labels are display-only and ignored."""
    parts = []
    for spec in filters or []:
        if not isinstance(spec, Mapping):
            continue
        rec = {
            str(k): spec[k]
            for k in ("group", "field", "eq", "in", "gte", "lte", "between", "null_as")
            if k in spec and spec[k] is not None
        }
        if rec.get("in") is not None:
            rec["in"] = tuple(rec["in"])
        if rec.get("between") is not None:
            rec["between"] = tuple(rec["between"])
        parts.append(tuple(sorted((k, _cache_value(rec[k])) for k in rec)))
    return tuple(sorted(parts))
