"""
Trade Pattern ML Model — K-Means clustering on sent-package feature vectors.

Training groups historical trades by target-player class (pos × value tier),
then clusters the sent packages within each class. Representative examples
from each cluster are stored so the serving layer can match them to any
viewer's roster — including throw-ins that commonly accompany big deals.

Model JSON layout
-----------------
{
  "version": 1,
  "trained_at": "...",
  "n_trades": N,
  "classes": {
    "RB-T2": {
      "clusters": [
        {
          "centroid": [f0..f9],
          "size": 42,
          "examples": [
            {"trade_id": "...", "target_value": 900.0, "sent_value": 840.0,
             "sent_assets": [{asset_type, sent_player_id, pick_round, ...}, ...]}
          ]
        }, ...
      ]
    }, ...
  }
}

Feature vector (10 dims)
------------------------
  [0] value_ratio       sent_value / target_value, clipped [0.3, 2.0]
  [1] n_players_norm    # players sent / 4
  [2] n_picks_norm      # picks sent / 4
  [3] top_tier_inv      1 / tier_of_best_sent_player (T1→1.0, T4→0.25); 0 if none
  [4] has_second_player 1.0 if 2+ players in package
  [5] r1_count_norm     # 1st-round picks / 3
  [6] r2_count_norm     # 2nd-round picks / 3
  [7] rb_frac           fraction of sent players who are RB
  [8] wr_frac           fraction of sent players who are WR
  [9] young_frac        fraction of sent players age < 25
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "trade_pattern_model.json"
)

# Tier thresholds (mirrors _FALLBACK_THRESHOLDS in app.py)
_TIER_BOUNDS = [850.0, 700.0, 550.0, 420.0, 300.0, 200.0, 120.0, 60.0]

FEATURE_DIM = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _value_to_tier(value: float) -> int:
    for i, bound in enumerate(_TIER_BOUNDS, 1):
        if value >= bound:
            return i
    return 9


def _pick_rough_value(rnd: int) -> float:
    return 450.0 if rnd == 1 else 175.0 if rnd == 2 else 70.0


def _target_class(position: str, value: float) -> str:
    pos = str(position or "WR").upper()
    if pos not in {"QB", "RB", "WR", "TE"}:
        pos = "WR"
    tier = _value_to_tier(value)
    return f"{pos}-T{tier}"


def _size_bucket(num_teams: int) -> str:
    """Map team count to canonical size bucket."""
    if num_teams <= 9:
        return "8"
    if num_teams <= 11:
        return "10"
    if num_teams == 12:
        return "12"
    return "14"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def featurize(
    sent_assets: list[dict],
    target_value: float,
    values_by_id: dict,
) -> Optional[list[float]]:
    """
    Build a FEATURE_DIM-dimensional vector for a sent package.
    Returns None when the package has no meaningful value.

    sent_assets items:
      {asset_type, sent_player_id, pick_round, pick_season, pick_order}
    values_by_id:
      player_id → {position, value, age, ...}
    """
    players = [a for a in sent_assets if a.get("asset_type") == "player"]
    picks   = [a for a in sent_assets if a.get("asset_type") == "pick"]

    if not players and not picks:
        return None

    # Player details
    player_vals: list[tuple[float, str, float]] = []  # (value, pos, age)
    rb_count = wr_count = young_count = 0

    for p in players:
        pid  = str(p.get("sent_player_id") or p.get("player_id") or "")
        info = values_by_id.get(pid) or {}
        val  = float(info.get("value") or 0)
        pos  = str(info.get("position") or "WR").upper()
        age  = float(info.get("age") or 0)
        if val < 10:
            continue
        player_vals.append((val, pos, age))
        if pos == "RB":
            rb_count += 1
        elif pos == "WR":
            wr_count += 1
        if age and age < 25:
            young_count += 1

    # Pick details
    pick_rounds = [int(pk.get("pick_round") or 3) for pk in picks]
    r1_count    = sum(1 for r in pick_rounds if r == 1)
    r2_count    = sum(1 for r in pick_rounds if r == 2)
    pick_vals   = [_pick_rough_value(r) for r in pick_rounds]

    all_vals   = [v for v, _, _ in player_vals] + pick_vals
    total_sent = sum(all_vals)

    if total_sent < 50:
        return None

    target_value = max(target_value, 100.0)
    value_ratio  = min(2.0, max(0.3, total_sent / target_value))

    n_pl = len(player_vals)
    n_pk = len(picks)

    if player_vals:
        player_vals.sort(key=lambda x: -x[0])
        top_tier     = _value_to_tier(player_vals[0][0])
        top_tier_inv = 1.0 / top_tier
    else:
        top_tier_inv = 0.0

    has_second = 1.0 if n_pl >= 2 else 0.0
    rb_frac    = rb_count    / n_pl if n_pl > 0 else 0.0
    wr_frac    = wr_count    / n_pl if n_pl > 0 else 0.0
    young_frac = young_count / n_pl if n_pl > 0 else 0.0

    return [
        value_ratio,
        min(1.0, n_pl / 4.0),
        min(1.0, n_pk / 4.0),
        top_tier_inv,
        has_second,
        min(1.0, r1_count / 3.0),
        min(1.0, r2_count / 3.0),
        rb_frac,
        wr_frac,
        young_frac,
    ]


# ---------------------------------------------------------------------------
# K-Means (numpy-only implementation)
# ---------------------------------------------------------------------------

def _kmeans(
    X: np.ndarray,
    k: int,
    n_iter: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's K-Means with K-Means++ init. Returns (centroids, labels)."""
    rng = np.random.default_rng(seed)
    n   = len(X)
    k   = min(k, n)

    # K-Means++ initialisation
    first    = int(rng.integers(n))
    centers  = [X[first]]
    for _ in range(k - 1):
        dists = np.array(
            [min(float(np.sum((x - c) ** 2)) for c in centers) for x in X]
        )
        total = dists.sum()
        if total == 0:
            break
        probs = dists / total
        idx   = int(rng.choice(n, p=probs))
        centers.append(X[idx])

    centroids = np.array(centers, dtype=float)
    labels    = np.zeros(n, dtype=int)

    for _ in range(n_iter):
        # Assignment step
        diffs      = X[:, None, :] - centroids[None, :, :]  # (n, k, d)
        sq_dists   = (diffs ** 2).sum(axis=2)               # (n, k)
        new_labels = np.argmin(sq_dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # Update step
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = X[mask].mean(axis=0)

    return centroids, labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    trade_rows: list[dict],
    values_by_id: dict,
    k_max: int = 4,
    min_trades_player: int = 5,
    min_per_cluster: int = 5,
) -> dict:
    """
    Build the trade pattern model from pre-processed trade rows.

    Each trade_row must have:
      target_player_id  str
      target_value      float
      sent_assets       list[{asset_type, sent_player_id, pick_round, pick_season, pick_order}]

    Clusters per individual player first; falls back to pos×tier class clusters
    for players with fewer than min_trades_player trades.

    Returns a model dict ready to pass to save_model().
    """
    from collections import defaultdict

    # Bucket trades by player_id AND by class (for fallback)
    player_trades: dict[str, list[dict]] = defaultdict(list)
    class_trades:  dict[str, list[dict]] = defaultdict(list)

    for trade in trade_rows:
        target_id   = str(trade.get("target_player_id") or "")
        target_info = values_by_id.get(target_id)
        if not target_info:
            continue

        target_val = float(target_info.get("value") or 0)
        if target_val < 100:
            continue

        target_pos = str(target_info.get("position") or "WR").upper()
        cls        = _target_class(target_pos, target_val)
        sent       = trade.get("sent_assets") or []
        vec        = featurize(sent, target_val, values_by_id)
        if vec is None:
            continue

        sent_val = sum(
            float((values_by_id.get(str(a.get("sent_player_id") or a.get("player_id") or "")) or {}).get("value") or 0)
            for a in sent if a.get("asset_type") == "player"
        ) + sum(
            _pick_rough_value(int(a.get("pick_round") or 3))
            for a in sent if a.get("asset_type") == "pick"
        )

        row = {
            "trade_id":     trade.get("trade_id"),
            "target_value": target_val,
            "sent_value":   sent_val,
            "sent_assets":  sent,
            "feature_vec":  vec,
        }
        player_trades[target_id].append(row)
        class_trades[cls].append(row)

    logger.info(
        "[trade_model] %d players, %d classes",
        len(player_trades), len(class_trades),
    )

    def _fit_clusters(trades: list[dict], label: str) -> list[dict]:
        n = len(trades)
        X = np.array([t["feature_vec"] for t in trades], dtype=float)
        k = max(1, min(k_max, n // min_per_cluster))
        if k <= 1:
            centroid = X.mean(axis=0)
            return [{"centroid": centroid.tolist(), "size": n,
                     "examples": _closest_examples(trades, centroid, X)}]
        centroids, labels = _kmeans(X, k)
        clusters = []
        for j in range(k):
            mask = labels == j
            ct   = [trades[i] for i in range(n) if mask[i]]
            if not ct:
                continue
            clusters.append({
                "centroid": centroids[j].tolist(),
                "size":     len(ct),
                "examples": _closest_examples(ct, centroids[j], X[mask]),
            })
        clusters.sort(key=lambda c: -c["size"])
        return clusters

    # ── Per-player clusters ───────────────────────────────────────────────
    model_players: dict[str, dict] = {}
    for pid, trades in player_trades.items():
        if len(trades) < min_trades_player:
            continue
        info = values_by_id.get(pid) or {}
        model_players[pid] = {
            "name":     info.get("name", pid),
            "class":    _target_class(str(info.get("position") or "WR").upper(),
                                      float(info.get("value") or 0)),
            "clusters": _fit_clusters(trades, pid),
        }

    logger.info("[trade_model] %d players with clusters", len(model_players))

    # ── Class fallback clusters ───────────────────────────────────────────
    model_classes: dict[str, dict] = {}
    for cls, trades in class_trades.items():
        model_classes[cls] = {"clusters": _fit_clusters(trades, cls)}

    total = sum(len(v) for v in player_trades.values())
    return {
        "version":    2,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_trades":   total,
        "players":    model_players,
        "classes":    model_classes,
    }


def train_bucketed(
    trade_rows: list[dict],
    values_by_id: dict,
    k_max: int = 4,
    min_trades_player: int = 5,
    min_per_cluster: int = 5,
) -> dict:
    """
    Train one model per league-size bucket and return a combined model dict.

    trade_rows must include a 'num_teams' field (added by analytics.py).
    Produces {"version": 3, "trained_at": ..., "n_trades": N, "buckets": {
        "8":  {players, classes},
        "10": {players, classes},
        "12": {players, classes},
        "14": {players, classes},
    }}
    """
    from collections import defaultdict

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for row in trade_rows:
        bucket = _size_bucket(int(row.get("num_teams") or 12))
        by_bucket[bucket].append(row)

    buckets_out: dict[str, dict] = {}
    for bucket in ["8", "10", "12", "14"]:
        rows = by_bucket.get(bucket) or []
        logger.info("[trade_model] bucket=%s  %d trades", bucket, len(rows))
        if len(rows) < 10:
            logger.warning("[trade_model] bucket=%s too few trades — skipping", bucket)
            continue
        m = train(rows, values_by_id, k_max=k_max,
                  min_trades_player=min_trades_player,
                  min_per_cluster=min_per_cluster)
        buckets_out[bucket] = {"players": m["players"], "classes": m["classes"]}

    return {
        "version":    3,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_trades":   len(trade_rows),
        "buckets":    buckets_out,
    }


def _closest_examples(
    trades: list[dict],
    centroid: np.ndarray,
    X: np.ndarray,
    max_ex: int = 20,
) -> list[dict]:
    """Return up to max_ex examples nearest to the centroid."""
    dists = ((X - centroid) ** 2).sum(axis=1)
    idx   = np.argsort(dists)[:max_ex]
    out   = []
    for i in idx:
        t = trades[int(i)]
        out.append({
            "trade_id":    t["trade_id"],
            "target_value": t["target_value"],
            "sent_value":  t["sent_value"],
            "sent_assets": t["sent_assets"],
        })
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: dict, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(model, f, separators=(",", ":"))
    logger.info(
        "[trade_model] Saved %s  (%d bytes, %d classes, %d trades)",
        path,
        os.path.getsize(path),
        len(model.get("classes") or {}),
        model.get("n_trades", 0),
    )


def load_model(path: str = MODEL_PATH) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            model = json.load(f)
        trained_at = model.get("trained_at")
        if trained_at:
            try:
                age = datetime.now(timezone.utc) - datetime.fromisoformat(trained_at)
                model["model_stale_days"] = age.days
                if age.days > 14:
                    logger.warning("[trade_model] Model is %d days old", age.days)
            except Exception:
                pass
        return model
    except Exception as exc:
        logger.warning("[trade_model] Failed to load model from %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------

def suggest_packages(
    model: dict,
    target_player_id: str,
    target_pos: str,
    target_value: float,
    viewer_players: list[dict],
    viewer_picks: list[dict],
    values_by_id: dict,
    n: int = 5,
    value_floor_ratio: float = 0.80,
    num_teams: int = 12,
) -> list[dict]:
    """
    Given a loaded model and viewer's roster, return suggested packages.

    Looks up clusters for the specific player first; falls back to the
    pos×tier class clusters when the player has insufficient trade history.

    Returns list of package dicts compatible with _real_trade_packages_for_target:
      {send: [...], trades_like_this: N, sig: [...], pattern_source: 'ml'}
    """
    # Resolve the right bucket sub-model for this league size (v3 format).
    # Falls back to adjacent buckets, then the top-level model (v2 format).
    if "buckets" in model:
        bucket = _size_bucket(num_teams)
        buckets = model["buckets"]
        # Try exact bucket, then adjacent sizes, then any available bucket
        for b in [bucket, "12", "10", "14", "8"]:
            if b in buckets:
                model = buckets[b]
                break
        else:
            model = next(iter(buckets.values())) if buckets else model

    # 1. Try player-specific clusters
    player_data = (model.get("players") or {}).get(str(target_player_id))
    if player_data:
        clusters = player_data.get("clusters") or []
    else:
        # 2. Fall back to pos×tier class
        cls        = _target_class(target_pos, target_value)
        class_data = (model.get("classes") or {}).get(cls)
        if not class_data:
            tier = _value_to_tier(target_value)
            for adj in [tier - 1, tier + 1, tier - 2]:
                if 1 <= adj <= 9:
                    class_data = (model.get("classes") or {}).get(
                        f"{target_pos.upper()}-T{adj}"
                    )
                    if class_data:
                        break
        if not class_data:
            return []
        clusters = class_data.get("clusters") or []
    packages: list[dict] = []
    seen_shapes: set[str] = set()

    for cluster in clusters:
        if len(packages) >= n:
            break
        # Pick-only clusters get a looser floor since a single 1st-rounder
        # won't hit 80% of a top player's value on its own.
        centroid_n_players = max(0, round(float(cluster["centroid"][1]) * 4))
        centroid_n_picks   = max(0, round(float(cluster["centroid"][2]) * 4))
        floor_ratio = 0.60 if centroid_n_players == 0 else value_floor_ratio
        floor = target_value * floor_ratio
        pkg = _match_viewer_to_cluster(
            centroid       = cluster["centroid"],
            target_value   = target_value,
            viewer_players = viewer_players,
            viewer_picks   = viewer_picks,
            values_by_id   = values_by_id,
            value_floor    = floor,
        )
        if pkg is None:
            continue
        shape = _shape_key(pkg["send"])
        if shape in seen_shapes:
            continue
        seen_shapes.add(shape)
        pkg["trades_like_this"] = cluster.get("size", 1)
        pkg["pattern_source"]   = "ml"
        packages.append(pkg)

    return packages


def _shape_key(assets: list[dict]) -> str:
    """Stable canonical string for deduplicating packages by shape."""
    parts = []
    for a in sorted(assets, key=lambda x: -(float(x.get("value") or x.get("send_value") or 0))):
        if a.get("is_pick"):
            parts.append(f"pk{a.get('pick_round', 3)}")
        else:
            tier = _value_to_tier(float(a.get("value") or a.get("send_value") or 0))
            pos  = str(a.get("position") or "?")[:2].upper()
            parts.append(f"{pos}T{tier}")
    return "+".join(parts)


def _match_viewer_to_cluster(
    centroid: list[float],
    target_value: float,
    viewer_players: list[dict],
    viewer_picks: list[dict],
    values_by_id: dict,
    value_floor: float = 0.0,
    exclude_pids: Optional[set] = None,
) -> Optional[dict]:
    """
    Assemble the best matching package from the viewer's roster given a
    cluster centroid (feature vector).

    centroid layout: see module docstring.
    """
    target_value = max(target_value, 100.0)
    exclude_pids = exclude_pids or set()

    # Denormalise centroid targets; cap value_ratio so stale training data
    # (when the target player was cheaper) doesn't inflate the send value.
    value_ratio = min(float(centroid[0]), 1.15)
    n_players   = max(0, round(float(centroid[1]) * 4))
    n_picks     = max(0, round(float(centroid[2]) * 4))
    n_r1        = max(0, round(float(centroid[5]) * 3))
    n_r2        = max(0, round(float(centroid[6]) * 3))
    rb_frac     = float(centroid[7])
    wr_frac     = float(centroid[8])

    target_sent = target_value * value_ratio

    # Estimate pick contribution using actual viewer pick values (sorted best-first)
    _sorted_picks = sorted(viewer_picks, key=lambda p: float(p.get("value") or 0), reverse=True)
    _r1_picks = [p for p in _sorted_picks if int(p.get("pick_round") or 3) == 1]
    _r2_picks = [p for p in _sorted_picks if int(p.get("pick_round") or 3) == 2]
    _r3_picks = [p for p in _sorted_picks if int(p.get("pick_round") or 3) >= 3]
    _r1_val = float(_r1_picks[0].get("value") or 450.0) if _r1_picks else 450.0
    _r2_val = float(_r2_picks[0].get("value") or 175.0) if _r2_picks else 175.0
    _r3_val = float(_r3_picks[0].get("value") or 70.0) if _r3_picks else 70.0
    pick_contribution = n_r1 * _r1_val + n_r2 * _r2_val + max(0, n_picks - n_r1 - n_r2) * _r3_val
    player_target_total = max(target_sent - pick_contribution, target_value * 0.5)

    # Per-slot value target for each position
    def _slot_target(pos_frac: float, n_slots: int) -> float:
        if n_slots <= 0 or n_players <= 0:
            return player_target_total
        return player_target_total * (pos_frac / max(rb_frac + wr_frac, 0.01)) / n_slots

    n_rb = round(n_players * rb_frac) if rb_frac > 0.15 else 0
    n_wr = round(n_players * wr_frac) if wr_frac > 0.15 else 0
    n_flex = max(0, n_players - n_rb - n_wr)

    rb_slot_target = _slot_target(rb_frac, n_rb)
    wr_slot_target = _slot_target(wr_frac, n_wr)
    flex_slot_target = player_target_total / max(n_players, 1)

    # ── Select players by tier-match, position-aware ──────────────────────
    used_pids: set[str] = set(exclude_pids)
    sent_assets: list[dict] = []

    def _pick_players(positions: list[str], slot_target: float, n: int) -> None:
        """Pick n players matching any of positions, prioritising exact tier then ±1."""
        req_tier = _value_to_tier(slot_target)
        for _ in range(n):
            candidates = [
                p for p in viewer_players
                if (not positions or p.get("position") in positions)
                and str(p.get("player_id") or "") not in used_pids
                and float(p.get("value") or 0) >= 30
                and _value_to_tier(float(p.get("value") or 0)) == req_tier
            ]
            if not candidates:
                break
            chosen = min(candidates, key=lambda p: abs(float(p.get("value") or 0) - slot_target))
            pid  = str(chosen.get("player_id") or "")
            val  = float(chosen.get("value") or 0)
            info = values_by_id.get(pid) or {}
            used_pids.add(pid)
            sent_assets.append({
                "player_id":  pid,
                "name":       chosen.get("name") or info.get("name") or "",
                "position":   chosen.get("position") or info.get("position") or "",
                "value":      val,
                "send_value": val,
                "is_pick":    False,
            })

    _pick_players(["RB"], rb_slot_target, n_rb)
    _pick_players(["WR"], wr_slot_target, n_wr)
    _pick_players(["RB", "WR", "TE", "QB"], flex_slot_target, n_flex)

    # ── Select picks ──────────────────────────────────────────────────────
    pick_pool = sorted(
        viewer_picks,
        key=lambda p: (int(p.get("pick_round") or 3), -float(p.get("value") or 0)),
    )
    used_names: set[str] = set()
    r1_added = r2_added = extra_added = 0

    for pk in pick_pool:
        total_picks_added = r1_added + r2_added + extra_added
        if total_picks_added >= n_picks:
            break
        pname = str(pk.get("name") or "")
        if pname in used_names:
            continue
        rnd = int(pk.get("pick_round") or 3)

        if rnd == 1 and r1_added < n_r1:
            r1_added += 1
        elif rnd == 2 and r2_added < n_r2:
            r2_added += 1
        elif total_picks_added < n_picks:
            extra_added += 1
        else:
            continue

        used_names.add(pname)
        sent_assets.append({
            "name":        pname,
            "value":       float(pk.get("value") or 0),
            "send_value":  float(pk.get("value") or 0),
            "is_pick":     True,
            "pick_round":  rnd,
            "pick_season": pk.get("pick_season"),
            "pick_order":  pk.get("pick_order"),
        })

    if not sent_assets:
        return None

    # Value gate
    total_val = sum(float(a.get("value") or a.get("send_value") or 0) for a in sent_assets)
    if value_floor and total_val < value_floor:
        return None

    # Build sig tokens for display
    sig: list[str] = []
    for a in sent_assets:
        if a.get("is_pick"):
            order = a.get("pick_order")
            try:
                o = int(order)
                bucket = "Early" if o <= 4 else ("Mid" if o <= 8 else "Late")
            except (TypeError, ValueError):
                bucket = ""
            rnd = a.get("pick_round", 3)
            sig.append(f"K:{rnd}:{bucket}" if bucket else f"K:{rnd}")
        else:
            tier = _value_to_tier(float(a.get("value") or 0))
            pos  = str(a.get("position") or "WR").upper()
            sig.append(f"P:{pos}:T{tier}")

    return {"send": sent_assets, "sig": sig}
