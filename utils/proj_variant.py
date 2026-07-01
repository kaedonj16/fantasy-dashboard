"""Projection-variant selection.

Extracted from utils/utils.py so this pure logic can be unit-tested without
importing that module's heavier dependencies (requests / bs4 / dashboard_services).
Given a league's raw Sleeper scoring settings, returns the key of the projection
set that matches its scoring — reception points, TE premium, and passing-TD value.
"""
from __future__ import annotations


def pick_proj_variant(raw_sleeper_settings: dict) -> str:
    """
    Return the projection variant key that matches a league's scoring settings.
    Keys: ppr | half_ppr | std | tep | 6pt_ppr | 6pt_half | 6pt_tep
    """
    s = raw_sleeper_settings or {}
    rec      = float(s.get("rec", 1.0))
    te_bonus = float(s.get("bonus_rec_te", 0.0))
    pass_td  = float(s.get("pass_td", 4.0))

    tep   = te_bonus >= 0.25
    six   = pass_td >= 5.5

    if rec >= 1.0:
        base = "ppr"
    elif rec >= 0.4:
        base = "half_ppr"
    else:
        base = "std"

    if six and tep and base == "ppr":
        return "6pt_tep"
    if six and base == "ppr":
        return "6pt_ppr"
    if six and base == "half_ppr":
        return "6pt_half"
    if tep and base == "ppr":
        return "tep"
    return base
