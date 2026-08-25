"""Pure helpers for Build Around acquire packages and pattern ranking.

Kept free of Flask/DB so the "never a blank page for a searched player"
invariants can be unit-tested in the base suite.
"""
from __future__ import annotations

import math
from typing import Optional

# Rough current-market value of a T{n} chip, used to rank historical pattern
# signatures against the searched player's *current* value. A T2 pattern at 2%
# frequency should not outrank a T7 pattern for a T7 dart-throw just because
# both cleared the "seen twice" bar.
_TIER_VAL_EST = {
    1: 1300.0, 2: 900.0, 3: 640.0, 4: 440.0, 5: 310.0,
    6: 200.0, 7: 110.0, 8: 55.0, 9: 20.0,
}


def estimate_display_sig_value(sig: str) -> float:
    """Estimate the current-market value of a display signature like
    ``WR-T4 + PICK:R2`` or ``PICK:R1:Early``."""
    if not sig:
        return 0.0
    total = 0.0
    for part in sig.split(" + "):
        part = (part or "").strip()
        if not part:
            continue
        if part.startswith("PICK"):
            segs = part.split(":")
            rnd = 3
            if len(segs) > 1:
                token = segs[1]
                if token.startswith("R"):
                    token = token[1:]
                try:
                    rnd = int(token)
                except ValueError:
                    rnd = 3
            total += 450.0 if rnd == 1 else 175.0 if rnd == 2 else 70.0
            continue
        tier = 5
        for bit in part.split("-"):
            if bit.startswith("T") and bit[1:].isdigit():
                tier = int(bit[1:])
        total += _TIER_VAL_EST.get(tier, 200.0)
    return total


def _value_fit(est: float, focus_value: float) -> float:
    """0 is a perfect value match. 2x and 0.5x are equally far."""
    if focus_value <= 0 or est <= 0:
        return 0.0
    return abs(math.log(max(est / focus_value, 0.05)))


def rank_archetype_patterns(
    merged: dict,
    focus_value: float,
    pkg_sigs: set,
    total_trade_count: int,
    limit: int = 8,
) -> list:
    """Rank historical pattern signatures so value-appropriate shapes surface
    first. ``merged`` maps ``"WR-T4|throw-in"`` → trade count.

    A low-tier target with hundreds of one-off trades used to show four
    expensive chips (the only shapes that hit count≥2) and nothing the
    viewer's roster could actually make. Ranking by value-fit, then
    frequency, and keeping count=1 shapes that sit near the player's
    current value, keeps the grid useful.
    """
    rows = []
    for canon, cnt in (merged or {}).items():
        parts = str(canon).split("|", 1)
        core = parts[0]
        throw = parts[1] if len(parts) > 1 else ""
        if not core:
            continue
        est = estimate_display_sig_value(core)
        rows.append({
            "pattern_sig": core,
            "throw_in_sig": throw,
            "count": int(cnt or 0),
            "pct": round(int(cnt or 0) / max(int(total_trade_count) or 1, 1) * 100),
            "fits_your_team": core in (pkg_sigs or set()),
            "_fit": _value_fit(est, float(focus_value or 0)),
        })
    if not rows:
        return []

    rows.sort(key=lambda r: (r["_fit"], -r["count"]))
    # ~0.58x–1.73x of the target's current value.
    close = [r for r in rows if r["_fit"] <= 0.55]
    frequent = [r for r in rows if r["count"] >= 2]

    picked: list = []
    seen: set = set()

    def _add(r: dict) -> None:
        sig = r["pattern_sig"]
        if sig in seen:
            return
        seen.add(sig)
        picked.append(r)

    for r in close:
        if len(picked) >= limit:
            break
        _add(r)
    for r in frequent:
        if len(picked) >= limit:
            break
        _add(r)
    # Guarantee at least a handful of chips even when nothing is value-close.
    for r in rows:
        if len(picked) >= min(4, limit):
            break
        _add(r)

    for r in picked:
        r.pop("_fit", None)
    return picked[:limit]


def vm_pkg_to_real(vm: dict) -> dict:
    """Convert a value-matched (assets/value_label) package into the
    real_packages shape the player-packages API enriches and the UI renders."""
    send = []
    for a in vm.get("assets") or []:
        if a.get("is_pick") or a.get("position") == "PICK":
            slot = a.get("pick_slot")
            order = a.get("pick_order") or "mid"
            year = a.get("pick_season")
            rnd = a.get("pick_round")
            pick_id = (
                f"{year}_{rnd}_{int(slot):02d}" if slot else
                f"{year}_{rnd}_{order}" if (year and rnd) else None
            )
            send.append({
                "name": a.get("name", ""),
                "value": float(a.get("value") or 0),
                "send_value": float(a.get("value") or 0),
                "is_pick": True,
                "pick_round": rnd,
                "pick_season": year,
                "pick_slot": slot,
                "pick_order": order,
                "pick_id": pick_id,
            })
        else:
            send.append({
                "player_id": str(a.get("player_id") or a.get("id") or ""),
                "name": a.get("name", ""),
                "position": a.get("position", ""),
                "value": float(a.get("value") or 0),
                "send_value": float(a.get("value") or 0),
                "is_pick": False,
            })
    return {
        "send": send,
        "send_value": round(sum(float(x.get("value") or 0) for x in send), 1),
        "trades_like_this": 0,
        "pattern_source": "value",
        "sig": [],
    }


def package_asset_key(pkg: dict) -> frozenset:
    assets = pkg.get("send") or pkg.get("assets") or []
    return frozenset(
        str(a.get("player_id") or a.get("name") or "")
        for a in assets
    )


def filter_acquire_packages(
    packages: list,
    focus_player_id: str,
    focus_value: float,
    max_ratio: float = 1.40,
) -> list:
    """Drop packages that echo the focus player on the give side, or that send
    more than ``max_ratio`` of the target's current value (extreme overpays
    from historical data, not useful suggestions)."""
    focus = str(focus_player_id or "")
    cap = (focus_value or 1) * max_ratio
    out = []
    for pkg in packages or []:
        assets = pkg.get("send") or pkg.get("assets") or []
        if focus and any(
            (not a.get("is_pick")) and str(a.get("player_id") or "") == focus
            for a in assets
        ):
            continue
        send_val = float(pkg.get("send_value") or 0)
        if send_val <= 0:
            send_val = sum(float(a.get("value") or a.get("send_value") or 0) for a in assets)
        if send_val > cap:
            continue
        out.append(pkg)
    return out


def value_matched_acquire_packages(
    focus_value: float,
    players: list,
    picks: list,
    max_options: int = 12,
    sorted_vals: Optional[list] = None,
    league_size: int = 10,
    min_results: int = 0,
) -> list:
    """Build value-matched ASSET packages (players + picks from the viewer's
    roster) whose combined value is close to ``focus_value``.

    Labels and the acceptable band are computed on *effective* (depth-adjusted)
    value so a multi-asset offer is judged the way the trade card scores it.

    When ``min_results`` is set and the fair-value band can't fill that many
    slots, the closest remaining combos (honestly labeled overpay/steal) are
    appended so a searched player is never a blank page when the roster has
    anything to offer.
    """
    if focus_value <= 0:
        return []

    from dashboard_services.archetype_engine import _depth_penalty

    def _eff(assets: list) -> float:
        raw = sum(a["value"] for a in assets)
        return raw - _depth_penalty(max(0, len(assets) - 1), sorted_vals, league_size)

    lo, hi = focus_value * 0.80, focus_value * 1.25

    def _passet(p: dict) -> dict:
        return {
            "name": p.get("name", ""),
            "position": str(p.get("position") or "").upper(),
            "value": float(p.get("value") or 0),
            "is_pick": False,
            "player_id": str(p.get("player_id") or p.get("id") or ""),
        }

    def _pkasset(pk: dict) -> dict:
        return {
            "name": pk.get("name", "Pick"),
            "position": "PICK",
            "value": float(pk.get("value") or 0),
            "is_pick": True,
            "pick_season": pk.get("pick_season"),
            "pick_round": pk.get("pick_round"),
            "pick_slot": pk.get("pick_slot"),
            "pick_order": pk.get("pick_order") or "mid",
        }

    players_a = sorted(
        (_passet(p) for p in players if float(p.get("value") or 0) >= 30),
        key=lambda x: -x["value"],
    )
    picks_a = sorted((_pkasset(p) for p in picks), key=lambda x: -x["value"])
    if not players_a and not picks_a:
        return []

    def _label(eff: float) -> tuple:
        r = eff / focus_value if focus_value else 0
        if r <= 0.94:
            return "Great deal", "great"
        if r <= 1.08:
            return "Fair value", "fair"
        return "Overpay", "overpay"

    def _record(assets: list) -> tuple | None:
        total = round(sum(a["value"] for a in assets), 1)
        if total <= 0:
            return None
        key = frozenset(a.get("player_id") or a["name"] for a in assets)
        eff = _eff(assets)
        in_band = (lo <= total <= hi) and eff >= focus_value * 0.90
        return (abs(eff - focus_value), key, assets, total, eff, in_band)

    combos: list = []
    seen_keys: set = set()

    def _consider(assets: list) -> None:
        rec = _record(assets)
        if rec is None or rec[1] in seen_keys:
            return
        seen_keys.add(rec[1])
        combos.append(rec)

    for a in players_a:
        _consider([a])
    for a in picks_a:
        _consider([a])
    for p in players_a:
        for k in picks_a:
            _consider([p, k])
    for i, p1 in enumerate(players_a):
        if p1["value"] >= hi:
            continue
        for p2 in players_a[i + 1:]:
            _consider([p1, p2])
    for i, k1 in enumerate(picks_a):
        for k2 in picks_a[i + 1:]:
            _consider([k1, k2])

    def _to_pkg(rec: tuple) -> dict:
        _fit, _key, assets, total, eff, _in_band = rec
        label, cls = _label(eff)
        return {
            "assets": assets,
            "send_value": total,
            "value_label": label,
            "value_class": cls,
            "is_profile_match": True,
            "frequency": 0,
            "_fit": _fit,
        }

    in_band = sorted((c for c in combos if c[5]), key=lambda c: c[0])
    out = [_to_pkg(c) for c in in_band[:max_options]]
    used = {frozenset(a.get("player_id") or a["name"] for a in p["assets"]) for p in out}

    if min_results and len(out) < min_results:
        for rec in sorted(combos, key=lambda c: c[0]):
            if len(out) >= min_results:
                break
            if rec[1] in used:
                continue
            used.add(rec[1])
            out.append(_to_pkg(rec))

    out.sort(key=lambda x: x["_fit"])
    for o in out:
        o.pop("_fit", None)
    return out[:max(max_options, min_results or 0)]
