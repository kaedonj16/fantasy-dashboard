"""Pure TE-premium and format-aware value helpers.

Extracted from app.py so this logic can be unit-tested without importing the
full application (pandas / DB) stack. A league that awards bonus points per TE
reception ("TE premium") makes tight ends more valuable; these helpers snap the
league's Sleeper ``bonus_rec_te`` to the supported tiers (0 / 0.5 / 1.0) and
scale TE values by +20% per full point — matching the trade calculator, activity
feed and player modal so values stay consistent on every page that shows them.

Also owns the redraft/dynasty column picker and the unpriced-redraft depth fill
so waiver, start/sit, and the draft board all read the same numbers.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

_SKILL_POS = {"QB", "RB", "WR", "TE"}


def te_premium_from_settings(scoring_settings) -> float:
    """Snap a league's Sleeper ``bonus_rec_te`` to a supported premium tier.

    Returns 1.0 (full), 0.5 (half), or 0.0 (none). Non-dict / non-numeric input
    yields 0.0 rather than raising.
    """
    try:
        b = float((scoring_settings or {}).get("bonus_rec_te") or 0)
    except (TypeError, ValueError, AttributeError):
        return 0.0
    return 1.0 if b >= 0.75 else 0.5 if b >= 0.25 else 0.0


def apply_te_premium(value, position, te_premium) -> float:
    """Scale a TE's value up for TE-premium leagues; pass-through otherwise.

    +20% per full premium point. ``value`` is coerced to float (DB values arrive
    as Decimal, which would otherwise raise on ``Decimal * float``), returning
    0.0 on non-numeric input rather than raising.
    """
    try:
        v = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    if te_premium and str(position or "").upper() == "TE":
        return v * (1.0 + te_premium * 0.20)
    return v


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def format_value_keys(*, is_redraft: bool, is_sf: bool) -> tuple[str, str]:
    """(primary, fallback) value columns for a league's scoring type + QB format.

    Redraft/keeper uses ``redraft_value_*``. Dynasty Superflex uses ``sf_value``.
    Fallback is the dynasty column so a missing redraft price still ranks the
    player instead of dropping them below the value floor.
    """
    if is_redraft:
        return (
            ("redraft_value_sf" if is_sf else "redraft_value_1qb"),
            ("sf_value" if is_sf else "value"),
        )
    return (("sf_value" if is_sf else "value"), "value")


def format_rank_label_key(*, is_redraft: bool, is_sf: bool) -> str:
    """Position-rank label field matching ``format_value_keys``."""
    if is_redraft:
        return "redraft_sf_pos_rank_label" if is_sf else "redraft_pos_rank_label"
    return "sf_pos_rank_label" if is_sf else "pos_rank_label"


def row_format_value(row: dict, primary: str, fallback: str) -> float:
    """Numeric value from ``primary``, then ``fallback``, then 0."""
    if not isinstance(row, dict):
        return 0.0
    return _num(row.get(primary) or row.get(fallback) or 0.0)


def row_format_rank_label(row: dict, label_key: str) -> str:
    """Rank label for the active format, falling back to the dynasty 1QB label."""
    if not isinstance(row, dict):
        return ""
    return str(row.get(label_key) or row.get("pos_rank_label") or "")


def rerank_pos_labels(
    table: list,
    val_key: str,
    rank_key: str,
    label_key: str,
    positions: Iterable[str] | None = None,
) -> list:
    """Stamp ``rank_key`` / ``label_key`` from descending ``val_key`` within each position."""
    if not table:
        return table
    allowed = {str(p).upper() for p in positions} if positions is not None else None
    grp: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(table):
        if not isinstance(row, dict):
            continue
        pos = str(row.get("position") or row.get("pos") or "").upper()
        if not pos or pos == "PICK":
            continue
        if allowed is not None and pos not in allowed:
            continue
        grp[pos].append(i)
    for pos, idxs in grp.items():
        idxs.sort(key=lambda i: _num(table[i].get(val_key)), reverse=True)
        for rank, i in enumerate(idxs, 1):
            table[i][rank_key] = rank
            table[i][label_key] = f"{pos}{rank}"
    return table


def fill_unpriced_redraft_values(table: list) -> list:
    """Give skill players with no redraft price a value scaled below the priced floor.

    FantasyCalc only prices roughly the top ~64 RB / ~150 WR. Without this fill,
    waiver/start-sit ``or``-fallback to dynasty ``value`` and a redraft league
    shows dynasty numbers for most free agents. Derived values stay strictly
    below every priced player so real redraft prices always rank first.
    Mutates rows in place and returns the same list. Idempotent when a
    positive redraft value is already present.
    """
    if not table:
        return table
    for rd_field, dyn_field in (
        ("redraft_value_1qb", "value"),
        ("redraft_value_sf", "sf_value"),
    ):
        priced = [
            _num(row.get(rd_field))
            for row in table
            if isinstance(row, dict)
            and str(row.get("position") or "").upper() in _SKILL_POS
            and _num(row.get(rd_field)) > 0
        ]
        floor = min(priced) if priced else 1.0
        unpriced_dyn = [
            _num(row.get(dyn_field))
            for row in table
            if isinstance(row, dict)
            and str(row.get("position") or "").upper() in _SKILL_POS
            and _num(row.get(rd_field)) <= 0
        ]
        dyn_max = max(unpriced_dyn) if unpriced_dyn else 0.0
        if dyn_max <= 0:
            continue
        cap = floor * 0.9
        for row in table:
            if not isinstance(row, dict):
                continue
            if str(row.get("position") or "").upper() not in _SKILL_POS:
                continue
            if _num(row.get(rd_field)) > 0:
                continue
            dyn = _num(row.get(dyn_field))
            if dyn > 0:
                row[rd_field] = round(cap * (dyn / dyn_max), 2)
    return table


def apply_redraft_display_fields(table: list) -> list:
    """Fill missing redraft values and stamp redraft position-rank labels."""
    fill_unpriced_redraft_values(table)
    if table:
        rerank_pos_labels(table, "redraft_value_1qb", "redraft_pos_rank", "redraft_pos_rank_label")
        rerank_pos_labels(table, "redraft_value_sf", "redraft_sf_pos_rank", "redraft_sf_pos_rank_label")
    return table
