"""
Prospect archetypes v2: scored NFL-outlook fits (model v3.2).

ONE isolated, data-driven rule module: every band, weight, and gate lives
in the tables below, evaluated by classify_archetype(). To revise the
taxonomy, edit this file only; nothing else in the pipeline hardcodes
archetype thresholds. The full spec, with validation against the elite,
named, and 2027 panels, is ~/workspace/prospect-archetypes-v2.md; this
module implements it (bands and weights verbatim, gates included).

Framing rule (non-negotiable): the archetype is an outlook to the player's
NFL archetype, in role and style. It is NOT a description of his college
role. College data is evidence, not the verdict; scheme artifacts are
discounted, and where usage and translatable traits disagree, traits win.

Design rules:
- Fit scores, not first match: every archetype at a position gets a fit
  score 0-100, a weighted average of input subscores scored on published
  bands (piecewise linear, clamped at the ends). Missing inputs are
  excluded and the remaining weights renormalize; a missing input never
  scores as neutral or zero, it lowers presence (confidence) instead.
- Basis season: the demonstrated peak profile. Among seasons with 8+
  games (all seasons if none qualify), the best per-game production rate
  discounted by min(1, games/12); seasons within 3% of the best are
  tie-broken by share (market share WR/TE, dominator RB, TD:INT QB).
- Draft capital is a supporting input only (the NFL's own role
  projection): capped weights, never the sole driver, and Developmental
  uses it as a veto, never as a positive case for a role.
- Confidence High/Medium/Low comes from presence (share of the position's
  full input set that is populated) plus the fit margin, with projection
  caps at Medium (single starter-volume season; a capital-led case with
  size and speed unmeasured; a small-sample TD rate driving the win).
- A second fit within 6 points and at least 55 becomes one secondary
  trait, encoded in the stored label as "Primary · Secondary".
- Omission (no label at all) when presence < 0.50, when no basis season
  has 5+ games, or when the winning fit is under 40.
- Label only: the archetype is derived FROM the grading inputs and must
  never feed back into the grade.

Returns {"archetype": label or None, "archetype_receipt": one line naming
the driving inputs, "archetype_confidence": "High"|"Medium"|"Low"|None,
"basis_season": year or None, "fits": {label: fit}}.
"""

from typing import Any, Dict, List, Optional, Tuple

Band = List[Tuple[float, float]]


def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _band(x: Optional[float], pts: Band) -> Optional[float]:
    """Piecewise-linear band score, clamped at the ends."""
    if x is None:
        return None
    if x <= pts[0][0]:
        return float(pts[0][1])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return float(pts[-1][1])


def _inv(sub: Optional[float]) -> Optional[float]:
    return (100.0 - sub) if sub is not None else None


def _mean_subs(subs: List[Optional[float]]) -> Optional[float]:
    present = [s for s in subs if s is not None]
    return (sum(present) / len(present)) if present else None


# ─────────────────────────────────────────────────────────────────────────────
# BANDS. "a to x, b to y" reads: at value a the subscore is x, linear
# between listed points, clamped outside. Verbatim from the v2 spec.
# ─────────────────────────────────────────────────────────────────────────────

BANDS: Dict[str, Band] = {
    # Shared
    "capital": [(1, 100), (5, 97), (10, 90), (16, 82), (24, 72), (40, 58),
                (64, 44), (100, 30), (160, 16), (260, 6)],
    "forty": [(4.30, 100), (4.40, 96), (4.45, 88), (4.55, 74), (4.65, 58),
              (4.75, 42), (4.85, 24), (4.97, 6)],
    "age": [(20.2, 95), (21.0, 78), (21.5, 66), (22.5, 45), (23.5, 22),
            (24.5, 8)],
    "trajectory": [(0.8, 20), (1.1, 45), (1.4, 70), (1.8, 90), (2.4, 100)],
    # QB
    "qb_rush_pg": [(0, 0), (10, 25), (20, 50), (35, 75), (50, 92), (65, 100)],
    "qb_rush_share": [(0, 0), (5, 25), (10, 48), (15, 68), (22, 88), (30, 100)],
    "qb_rush_td_pg": [(0, 5), (0.25, 35), (0.5, 62), (0.75, 85), (1.0, 100)],
    "qb_pass_pg": [(180, 15), (230, 45), (260, 62), (290, 78), (320, 92),
                   (360, 100)],
    "qb_ypa": [(7.0, 10), (7.8, 35), (8.4, 58), (9.0, 78), (9.6, 92),
               (10.4, 100)],
    "qb_cmp": [(56, 10), (60, 30), (64, 55), (68, 78), (72, 92), (76, 100)],
    "qb_td_int": [(1.2, 8), (1.8, 28), (2.4, 48), (3.0, 65), (3.8, 82),
                  (5.0, 94), (7.0, 100)],
    "qb_td_rate": [(3.5, 15), (5, 40), (6.5, 62), (8, 80), (9.5, 93),
                   (11, 100)],
    "qb_yds_cmp": [(11, 10), (12, 32), (13, 55), (14, 76), (15, 90),
                   (16, 100)],
    "qb_att_pg": [(24, 20), (28, 40), (32, 58), (36, 74), (40, 88), (45, 100)],
    "qb_wt": [(195, 25), (205, 45), (215, 62), (225, 78), (235, 90),
              (245, 100)],
    "qb_cmp_risk": [(55, 45), (60, 75), (64, 95), (67, 100), (70, 70),
                    (73, 40), (76, 20)],
    "qb_ypa_short": [(7.2, 35), (7.9, 75), (8.5, 100), (9.0, 92), (9.6, 68),
                     (10.4, 45)],
    "qb_rush_low": [(0, 100), (8, 88), (15, 62), (25, 35), (40, 12), (60, 0)],
    "qb_pass_low": [(180, 95), (230, 70), (280, 40), (330, 15)],
    # RB
    "rb_rush_pg": [(40, 10), (60, 30), (80, 52), (100, 72), (120, 88),
                   (145, 100)],
    "rb_ypc": [(4.4, 8), (4.9, 30), (5.4, 55), (5.9, 76), (6.4, 90),
               (7.0, 100)],
    "rb_rec_pg": [(0, 0), (8, 22), (15, 45), (22, 62), (30, 78), (40, 92),
                  (50, 100)],
    "rb_rec_share": [(2, 4), (6, 18), (10, 32), (15, 48), (20, 62), (26, 78),
                     (32, 92), (38, 100)],
    "rb_dom": [(0.10, 15), (0.18, 38), (0.24, 55), (0.30, 72), (0.35, 86),
               (0.42, 100)],
    "rb_total_pg": [(70, 15), (90, 38), (110, 60), (130, 78), (150, 92),
                    (175, 100)],
    "rb_td_rate": [(2, 15), (3.5, 35), (5, 55), (6.5, 72), (8, 88),
                   (9.5, 100)],
    "rb_wt": [(185, 20), (200, 42), (215, 60), (228, 76), (240, 90),
              (252, 100)],
    "rb_dom_inv": [(0.15, 100), (0.20, 82), (0.25, 65), (0.30, 48),
                   (0.35, 30), (0.42, 10)],
    "rb_total_mid": [(60, 30), (85, 80), (100, 100), (110, 92), (120, 70),
                     (140, 40), (165, 18)],
    "rb_rec_mid": [(4, 25), (10, 45), (14, 75), (19, 95), (24, 80), (30, 50),
                   (36, 25)],
    "rb_dom_mid": [(0.14, 30), (0.20, 62), (0.24, 88), (0.28, 95),
                   (0.32, 62), (0.38, 30)],
    "rb_cap_mid": [(1, 12), (20, 32), (40, 58), (70, 85), (110, 100),
                   (180, 78), (260, 50)],
    "rb_ypc_mid": [(4.2, 25), (4.7, 60), (5.1, 88), (5.5, 100), (5.9, 78),
                   (6.4, 50), (7.0, 28)],
    # WR
    "wr_dom": [(0.08, 8), (0.15, 30), (0.22, 52), (0.28, 70), (0.34, 86),
               (0.42, 100)],
    "wr_ms": [(0.10, 12), (0.18, 35), (0.24, 55), (0.30, 75), (0.36, 92),
              (0.42, 100)],
    "wr_ypr": [(10, 8), (11.5, 25), (13, 45), (14.5, 65), (16, 80),
               (17.5, 92), (19, 100)],
    "wr_ypr_alpha": [(11, 25), (13, 55), (14.5, 80), (16, 95), (18, 90),
                     (20, 75), (22, 60)],
    "wr_yds_pg": [(40, 12), (55, 32), (70, 52), (85, 70), (100, 85),
                  (120, 96), (140, 100)],
    "wr_rec_pg": [(3, 15), (4, 32), (5, 50), (6, 66), (7, 80), (8, 92),
                  (9.5, 100)],
    "wr_td_rate": [(4, 12), (7, 32), (10, 52), (13, 70), (16, 85), (20, 100)],
    "wr_wt": [(168, 5), (178, 22), (188, 42), (196, 60), (205, 76), (215, 90),
              (228, 100)],
    "wr_wt_inv": [(168, 100), (178, 92), (188, 78), (196, 62), (205, 45),
                  (215, 25), (228, 5)],
    "wr_ypr_low": [(10, 100), (11, 88), (12, 72), (13, 52), (14, 32),
                   (15.5, 15), (17, 5)],
    "wr_field_role": [(0.15, 95), (0.20, 75), (0.25, 55), (0.30, 35),
                      (0.36, 15)],
    # TE
    "te_rec_pg": [(15, 5), (25, 25), (35, 45), (45, 62), (55, 76), (65, 87),
                  (80, 95), (95, 100)],
    "te_ms": [(0.05, 8), (0.10, 25), (0.15, 42), (0.20, 58), (0.25, 74),
              (0.30, 88), (0.36, 100)],
    "te_ypr": [(9.5, 8), (10.5, 25), (11.5, 42), (12.5, 58), (13.5, 72),
               (14.5, 84), (16, 93), (18, 100)],
    "te_td_rate": [(3, 10), (6, 30), (9, 50), (12, 68), (16, 85), (20, 100)],
    "te_td_pg": [(0.1, 10), (0.3, 35), (0.5, 58), (0.7, 78), (0.9, 92),
                 (1.1, 100)],
    "te_wt": [(225, 15), (238, 35), (246, 55), (252, 72), (260, 88),
              (270, 100)],
    "te_wt_lean": [(218, 100), (228, 92), (238, 76), (246, 58), (252, 40),
                   (260, 22), (270, 5)],
    "te_rec_seam": [(15, 20), (30, 50), (45, 80), (60, 95), (75, 80),
                    (90, 55), (105, 35)],
    "te_rec_mid": [(15, 15), (25, 40), (35, 65), (45, 85), (55, 98),
                   (65, 85), (80, 60), (95, 40)],
    "te_ypr_mid": [(9.5, 30), (10.5, 55), (11.5, 78), (12.5, 95), (13.5, 88),
                   (14.5, 65), (16, 40), (18, 20)],
    "te_sustained": [(0, 15), (1, 55), (2, 85), (3, 100)],
    "te_cap_low": [(1, 4), (20, 14), (40, 30), (70, 52), (110, 74),
                   (180, 92), (260, 100)],
}


# ─────────────────────────────────────────────────────────────────────────────
# THE RULE TABLE. Each archetype: fit inputs as (feature, weight, band).
# A band of None means the feature value already IS a subscore (derived
# inverses and composites, built in _features). Gates cap a fit after all
# fits are computed; they encode definitions, not tuning.
# ─────────────────────────────────────────────────────────────────────────────

ARCHETYPE_RULES: Dict[str, List[Dict[str, Any]]] = {
    "QB": [
        {"label": "Dual-Threat", "gate": None, "inputs": [
            ("rush_pg", 0.30, "qb_rush_pg"), ("rush_share", 0.25, "qb_rush_share"),
            ("rush_td_pg", 0.15, "qb_rush_td_pg"), ("forty", 0.15, "forty"),
            ("wt", 0.15, "qb_wt")]},
        {"label": "Gunslinger", "gate": "gunslinger_precision", "inputs": [
            ("ypa", 0.25, "qb_ypa"), ("yds_cmp", 0.25, "qb_yds_cmp"),
            ("td_rate", 0.20, "qb_td_rate"), ("att_pg", 0.15, "qb_att_pg"),
            ("cmp", 0.15, "qb_cmp_risk")]},
        {"label": "Distributor", "gate": "distributor_accuracy", "inputs": [
            ("cmp", 0.30, "qb_cmp"), ("td_int", 0.25, "qb_td_int"),
            ("ypa", 0.20, "qb_ypa_short"), ("pass_pg", 0.15, "qb_pass_pg"),
            ("rush_pg", 0.10, "qb_rush_low")]},
        {"label": "Pocket Passer", "gate": None, "inputs": [
            ("pass_pg", 0.22, "qb_pass_pg"), ("cmp", 0.18, "qb_cmp"),
            ("ypa", 0.18, "qb_ypa"), ("td_int", 0.14, "qb_td_int"),
            ("rush_pg", 0.12, "qb_rush_low"), ("wt", 0.08, "qb_wt"),
            ("pick", 0.08, "capital")]},
        {"label": "Developmental", "gate": "qb_dev_dual_threat", "inputs": [
            ("polish_inv", 0.30, None), ("pass_pg", 0.20, "qb_pass_low"),
            ("forty", 0.20, "forty"), ("wt", 0.15, "qb_wt"),
            ("age", 0.15, "age")]},
    ],
    "RB": [
        {"label": "Workhorse", "gate": None, "inputs": [
            ("dom", 0.22, "rb_dom"), ("rush_pg", 0.22, "rb_rush_pg"),
            ("total_pg", 0.20, "rb_total_pg"), ("rec_pg", 0.16, "rb_rec_pg"),
            ("ypc", 0.12, "rb_ypc"), ("pick", 0.08, "capital")]},
        {"label": "Pass-Catching Back", "gate": "pcb_runner_first", "inputs": [
            ("rec_share", 0.30, "rb_rec_share"), ("rec_pg", 0.28, "rb_rec_pg"),
            ("ypc", 0.14, "rb_ypc"), ("forty", 0.14, "forty"),
            ("dom", 0.14, "rb_dom")]},
        {"label": "Power Back", "gate": None, "inputs": [
            ("ypc", 0.18, "rb_ypc"), ("wt", 0.20, "rb_wt"),
            ("rec_share_inv", 0.18, None), ("td_rate", 0.16, "rb_td_rate"),
            ("rush_pg", 0.16, "rb_rush_pg"), ("dom", 0.12, "rb_dom")]},
        {"label": "Home-Run Threat", "gate": None, "inputs": [
            ("ypc", 0.28, "rb_ypc"), ("forty", 0.26, "forty"),
            ("total_pg", 0.16, "rb_total_pg"), ("dom", 0.18, "rb_dom_inv"),
            ("rec_pg", 0.12, "rb_rec_pg")]},
        {"label": "Committee Back", "gate": None, "inputs": [
            ("total_pg", 0.30, "rb_total_mid"), ("rec_share", 0.20, "rb_rec_mid"),
            ("dom", 0.20, "rb_dom_mid"), ("pick", 0.15, "rb_cap_mid"),
            ("ypc", 0.15, "rb_ypc_mid")]},
    ],
    "WR": [
        {"label": "Alpha X", "gate": "alpha_size", "inputs": [
            ("dom", 0.24, "wr_dom"), ("ms", 0.20, "wr_ms"),
            ("yds_pg", 0.16, "wr_yds_pg"), ("pick", 0.16, "capital"),
            ("wt", 0.14, "wr_wt"), ("ypr", 0.10, "wr_ypr_alpha")]},
        {"label": "Playmaker", "gate": "alpha_usage_85", "inputs": [
            ("yds_pg", 0.26, "wr_yds_pg"), ("td_rate", 0.22, "wr_td_rate"),
            ("ypr", 0.18, "wr_ypr"), ("dom", 0.18, "wr_dom"),
            ("forty", 0.16, "forty")]},
        {"label": "Deep Threat", "gate": "alpha_usage_84", "inputs": [
            ("ypr", 0.30, "wr_ypr"), ("forty", 0.20, "forty"),
            ("yds_pg", 0.20, "wr_yds_pg"), ("td_rate", 0.18, "wr_td_rate"),
            ("ms", 0.12, "wr_field_role")]},
        {"label": "Possession", "gate": None, "inputs": [
            ("ypr", 0.26, "wr_ypr_low"), ("rec_pg", 0.24, "wr_rec_pg"),
            ("dom", 0.20, "wr_dom"), ("ms", 0.18, "wr_ms"),
            ("td_rate_inv", 0.12, None)]},
        {"label": "Slot", "gate": None, "inputs": [
            ("rec_pg", 0.26, "wr_rec_pg"), ("ypr", 0.26, "wr_ypr_low"),
            ("forty", 0.18, "forty"), ("ms", 0.16, "wr_ms"),
            ("wt", 0.14, "wr_wt_inv")]},
        {"label": "Developmental", "gate": "dev_wr", "inputs": [
            ("raw_prod_inv", 0.30, None), ("tools", 0.22, None),
            ("pick", 0.18, "capital"), ("age", 0.18, "age"),
            ("growth", 0.12, "trajectory")]},
    ],
    "TE": [
        {"label": "Move TE", "gate": None, "inputs": [
            ("rec_pg", 0.28, "te_rec_pg"), ("ms", 0.20, "te_ms"),
            ("sustained", 0.18, "te_sustained"), ("ypr", 0.14, "te_ypr"),
            ("forty", 0.12, "forty"), ("wt", 0.08, "te_wt_lean")]},
        {"label": "Seam Stretcher", "gate": None, "inputs": [
            ("ypr", 0.30, "te_ypr"), ("rec_pg", 0.24, "te_rec_seam"),
            ("td_rate", 0.18, "te_td_rate"), ("forty", 0.14, "forty"),
            ("ms", 0.14, "te_ms")]},
        {"label": "Two-Way TE", "gate": None, "inputs": [
            ("wt", 0.20, "te_wt"), ("rec_pg", 0.22, "te_rec_mid"),
            ("ypr", 0.18, "te_ypr_mid"), ("ms", 0.16, "te_ms"),
            ("pick", 0.12, "capital"), ("forty", 0.12, "forty")]},
        {"label": "Red-Zone Target", "gate": None, "inputs": [
            ("td_rate", 0.30, "te_td_rate"), ("td_pg", 0.20, "te_td_pg"),
            ("rec_pg_inv", 0.18, None), ("wt", 0.16, "te_wt"),
            ("ms", 0.16, "te_ms")]},
        {"label": "Blocking TE", "gate": None, "inputs": [
            ("rec_pg_inv", 0.34, None), ("ms_inv", 0.22, None),
            ("wt", 0.22, "te_wt"), ("pick", 0.22, "te_cap_low")]},
        {"label": "Developmental", "gate": "dev_te", "inputs": [
            ("raw_prod_inv", 0.30, None), ("tools", 0.22, None),
            ("pick", 0.18, "capital"), ("age", 0.18, "age"),
            ("growth", 0.12, "trajectory")]},
    ],
}

# The position's full input set: presence is the share of these features
# actually populated for the player.
PRESENCE_FEATURES: Dict[str, List[str]] = {
    "QB": ["pass_pg", "rush_pg", "rush_share", "rush_td_pg", "ypa", "cmp",
           "td_int", "td_rate", "yds_cmp", "att_pg", "wt", "forty", "age",
           "pick"],
    "RB": ["dom", "rush_pg", "total_pg", "rec_pg", "rec_share", "ypc",
           "td_rate", "wt", "forty", "pick"],
    "WR": ["dom", "ms", "ypr", "yds_pg", "rec_pg", "td_rate", "wt", "forty",
           "pick", "age", "growth"],
    "TE": ["rec_pg", "ms", "ypr", "td_rate", "td_pg", "sustained", "wt",
           "forty", "pick", "age", "growth"],
}


def _f1(x: Optional[float]) -> str:
    return f"{x:.1f}" if x is not None else "n/a"


def _f2(x: Optional[float]) -> str:
    return f"{x:.2f}" if x is not None else "n/a"


def _pct1(x: Optional[float]) -> str:
    return f"{x * 100:.0f}%" if x is not None else "n/a"


def _pctv(x: Optional[float]) -> str:
    return f"{x:.0f}%" if x is not None else "n/a"


# Receipt text per (position, feature): how a driving input reads on the card.
RECEIPT_LABELS: Dict[str, Dict[str, Any]] = {
    "QB": {
        "pass_pg": lambda f: f"{_f1(f['pass_pg'])} pass yds/g",
        "rush_pg": lambda f: f"{_f1(f['rush_pg'])} rush yds/g",
        "rush_share": lambda f: f"{_pctv(f['rush_share'])} of total yards on the ground",
        "rush_td_pg": lambda f: f"{_f2(f['rush_td_pg'])} rush TD/g",
        "ypa": lambda f: f"{_f1(f['ypa'])} yds/att",
        "cmp": lambda f: f"{_f1(f['cmp'])}% completions",
        "td_int": lambda f: f"{_f1(f['td_int'])} TD:INT",
        "td_rate": lambda f: f"{_f1(f['td_rate'])}% TD rate",
        "yds_cmp": lambda f: f"{_f1(f['yds_cmp'])} yds/completion",
        "att_pg": lambda f: f"{_f1(f['att_pg'])} attempts/g",
        "wt": lambda f: f"{_f1(f['wt'])} lb",
        "forty": lambda f: f"{_f2(f['forty'])} forty",
        "age": lambda f: f"{_f1(f['age'])} draft age",
        "pick": lambda f: f"pick {_f1(f['pick'])} capital",
        "polish_inv": lambda f: "unpolished passing profile",
    },
    "RB": {
        "dom": lambda f: f"{_pct1(f['dom'])} dominator",
        "rush_pg": lambda f: f"{_f1(f['rush_pg'])} rush yds/g",
        "total_pg": lambda f: f"{_f1(f['total_pg'])} scrimmage yds/g",
        "rec_pg": lambda f: f"{_f1(f['rec_pg'])} rec yds/g",
        "rec_share": lambda f: f"{_pctv(f['rec_share'])} receiving share",
        "ypc": lambda f: f"{_f1(f['ypc'])} yds/carry",
        "td_rate": lambda f: f"{_f1(f['td_rate'])}% rush TD rate",
        "wt": lambda f: f"{_f1(f['wt'])} lb",
        "forty": lambda f: f"{_f2(f['forty'])} forty",
        "pick": lambda f: f"pick {_f1(f['pick'])} capital",
        "rec_share_inv": lambda f: f"{_pctv(f['rec_share'])} receiving share",
    },
    "WR": {
        "dom": lambda f: f"{_pct1(f['dom'])} dominator",
        "ms": lambda f: f"{_pct1(f['ms'])} of team pass yds",
        "ypr": lambda f: f"{_f1(f['ypr'])} yds/rec",
        "yds_pg": lambda f: f"{_f1(f['yds_pg'])} rec yds/g",
        "rec_pg": lambda f: f"{_f1(f['rec_pg'])} rec/g",
        "td_rate": lambda f: f"{_f1(f['td_rate'])}% TD rate",
        "wt": lambda f: f"{_f1(f['wt'])} lb",
        "forty": lambda f: f"{_f2(f['forty'])} forty",
        "pick": lambda f: f"pick {_f1(f['pick'])} capital",
        "age": lambda f: f"{_f1(f['age'])} draft age",
        "growth": lambda f: f"production {_f2(f['growth'])}x year over year",
        "raw_prod_inv": lambda f: "thin college production",
        "tools": lambda f: "traits over production",
        "td_rate_inv": lambda f: f"{_f1(f['td_rate'])}% TD rate",
    },
    "TE": {
        "rec_pg": lambda f: f"{_f1(f['rec_pg'])} rec yds/g",
        "ms": lambda f: f"{_pct1(f['ms'])} of team pass yds",
        "ypr": lambda f: f"{_f1(f['ypr'])} yds/rec",
        "td_rate": lambda f: f"{_f1(f['td_rate'])}% TD rate",
        "td_pg": lambda f: f"{_f2(f['td_pg'])} TD/g",
        "sustained": lambda f: f"{_f1(f['sustained'])} seasons at 45+ rec yds/g",
        "wt": lambda f: f"{_f1(f['wt'])} lb",
        "forty": lambda f: f"{_f2(f['forty'])} forty",
        "pick": lambda f: f"pick {_f1(f['pick'])} capital",
        "age": lambda f: f"{_f1(f['age'])} draft age",
        "growth": lambda f: f"production {_f2(f['growth'])}x year over year",
        "raw_prod_inv": lambda f: "thin college production",
        "tools": lambda f: "traits over production",
        "rec_pg_inv": lambda f: f"{_f1(f['rec_pg'])} rec yds/g",
        "ms_inv": lambda f: f"{_pct1(f['ms'])} of team pass yds",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

def _season_rate(pos: str, s: Dict) -> Optional[float]:
    gp = _num(s.get("games_played"))
    if not gp:
        return None
    if pos == "QB":
        return ((_num(s.get("pass_yards")) or 0.0)
                + (_num(s.get("rush_yards")) or 0.0)) / gp
    if pos == "RB":
        return ((_num(s.get("rush_yards")) or 0.0)
                + (_num(s.get("receiving_yards")) or 0.0)) / gp
    return (_num(s.get("receiving_yards")) or 0.0) / gp


def _pick_basis(pos: str, seasons: List[Dict]):
    """v2 basis season: discounted peak rate, share tie-break (spec 2.2)."""
    ordered = sorted(seasons, key=lambda s: _num(s.get("season")) or 0)
    full = [s for s in ordered if (_num(s.get("games_played")) or 0) >= 8]
    pool = full if full else ordered

    def discounted(s):
        r = _season_rate(pos, s)
        if r is None:
            return None
        gp = _num(s.get("games_played")) or 0
        return r * min(1.0, gp / 12.0)

    def tiebreak(s):
        if pos == "QB":
            v = _num(s.get("td_int_ratio"))
        elif pos == "RB":
            v = _num(s.get("dominator_rating"))
        else:
            v = _num(s.get("market_share_yards"))
        return v if v is not None else -1.0

    scored = [(discounted(s), s) for s in pool]
    scored = [(d, s) for d, s in scored if d is not None]
    if not scored:
        return None, None
    best = max(d for d, _ in scored)
    cands = [s for d, s in scored if d >= 0.97 * best]
    basis = max(cands, key=tiebreak)
    idx = ordered.index(basis)
    growth = None
    if idx > 0:
        prior = _season_rate(pos, ordered[idx - 1])
        cur = _season_rate(pos, basis)
        if prior and cur:
            growth = cur / prior
    return basis, growth


def _features(pos: str, basis: Dict, seasons: List[Dict],
              profile: Dict, growth: Optional[float]) -> Dict[str, Optional[float]]:
    gp = _num(basis.get("games_played")) or 12.0
    f: Dict[str, Optional[float]] = {
        "wt": _num(profile.get("weight_lbs", profile.get("weight"))),
        "forty": _num(profile.get("forty_yard", profile.get("forty"))),
        "pick": _num(profile.get("projected_pick", profile.get("pick"))),
        "age": _num(profile.get("age")),
        "growth": growth,
    }
    rec_yds = _num(basis.get("receiving_yards")) or 0.0
    rush_yds = _num(basis.get("rush_yards")) or 0.0
    rec = _num(basis.get("receptions"))
    ms = _num(basis.get("market_share_yards"))
    if ms is None:
        team_pass = _num(basis.get("team_pass_yards"))
        if team_pass:
            ms = rec_yds / team_pass
    f["ms"] = ms
    f["dom"] = _num(basis.get("dominator_rating"))
    f["ypr"] = _num(basis.get("yds_per_reception"))
    if pos == "QB":
        att = _num(basis.get("pass_attempts"))
        pass_yds = _num(basis.get("pass_yards")) or 0.0
        pass_tds = _num(basis.get("pass_tds"))
        total = pass_yds + rush_yds
        f["pass_pg"] = pass_yds / gp
        f["rush_pg"] = rush_yds / gp
        f["rush_share"] = (100.0 * rush_yds / total) if total > 0 else 0.0
        rush_tds = _num(basis.get("rush_tds"))
        f["rush_td_pg"] = (rush_tds / gp) if rush_tds is not None else None
        f["ypa"] = _num(basis.get("yds_per_attempt"))
        f["cmp"] = _num(basis.get("completion_pct"))
        f["td_int"] = _num(basis.get("td_int_ratio"))
        f["td_rate"] = (100.0 * pass_tds / att) if (att and pass_tds is not None) else None
        f["yds_cmp"] = (f["ypa"] / (f["cmp"] / 100.0)) if (f["ypa"] and f["cmp"]) else None
        f["att_pg"] = (att / gp) if att else None
        cmp_sub = _band(f["cmp"], BANDS["qb_cmp"])
        ti_sub = _band(f["td_int"], BANDS["qb_td_int"])
        f["polish_inv"] = (100.0 - (cmp_sub + ti_sub) / 2.0) \
            if (cmp_sub is not None and ti_sub is not None) else None
    elif pos == "RB":
        scr = rush_yds + rec_yds
        rush_att = _num(basis.get("rush_attempts"))
        rush_tds = _num(basis.get("rush_tds"))
        f["rush_pg"] = rush_yds / gp
        f["total_pg"] = scr / gp
        f["rec_pg"] = rec_yds / gp
        f["rec_share"] = (100.0 * rec_yds / scr) if scr > 0 else 0.0
        f["ypc"] = _num(basis.get("yds_per_carry"))
        f["td_rate"] = (100.0 * rush_tds / rush_att) \
            if (rush_att and rush_tds is not None) else None
        f["rec_share_inv"] = _inv(_band(f["rec_share"], BANDS["rb_rec_share"]))
    elif pos == "WR":
        f["yds_pg"] = rec_yds / gp
        f["rec_pg"] = (rec / gp) if rec is not None else None
        rec_tds = _num(basis.get("receiving_tds"))
        f["td_rate"] = (100.0 * rec_tds / rec) if (rec and rec_tds is not None) else None
        f["td_rate_inv"] = _inv(_band(f["td_rate"], BANDS["wr_td_rate"]))
        yp_sub = _band(f["yds_pg"], BANDS["wr_yds_pg"])
        dom_sub = _band(f["dom"], BANDS["wr_dom"])
        f["raw_prod_inv"] = (100.0 - (0.55 * yp_sub + 0.45 * dom_sub)) \
            if (yp_sub is not None and dom_sub is not None) else None
        f["tools"] = _mean_subs([_band(f["wt"], BANDS["wr_wt"]),
                                 _band(f["forty"], BANDS["forty"])])
    elif pos == "TE":
        f["rec_pg"] = rec_yds / gp
        rec_tds = _num(basis.get("receiving_tds"))
        f["td_rate"] = (100.0 * rec_tds / rec) if (rec and rec_tds is not None) else None
        f["td_pg"] = (rec_tds / gp) if rec_tds is not None else None
        sustained = 0
        for s in seasons:
            sgp = _num(s.get("games_played"))
            syds = _num(s.get("receiving_yards"))
            if sgp and syds is not None and syds / sgp >= 45.0:
                sustained += 1
        f["sustained"] = float(sustained)
        f["rec_pg_inv"] = _inv(_band(f["rec_pg"], BANDS["te_rec_pg"]))
        f["ms_inv"] = _inv(_band(f["ms"], BANDS["te_ms"]))
        rp_sub = _band(f["rec_pg"], BANDS["te_rec_pg"])
        ms_sub = _band(f["ms"], BANDS["te_ms"])
        f["raw_prod_inv"] = (100.0 - (0.6 * rp_sub + 0.4 * ms_sub)) \
            if (rp_sub is not None and ms_sub is not None) else None
        f["tools"] = _mean_subs([_band(f["wt"], BANDS["te_wt"]),
                                 _band(f["forty"], BANDS["forty"])])
    return f


def _fit(spec: Dict, feats: Dict) -> Tuple[Optional[float], List]:
    """Weighted fit over present inputs; returns (fit, contributions)."""
    contribs = []
    num = 0.0
    den = 0.0
    for feature, weight, band_key in spec["inputs"]:
        raw = feats.get(feature)
        sub = raw if band_key is None else _band(raw, BANDS[band_key])
        if sub is None:
            continue
        contribs.append((feature, weight, sub))
        num += weight * sub
        den += weight
    if den == 0:
        return None, contribs
    return num / den, contribs


def _apply_gates(fits: Dict[str, Optional[float]], feats: Dict) -> None:
    def cap(label: str, ceiling: float) -> None:
        if fits.get(label) is not None:
            fits[label] = min(fits[label], ceiling)

    cmp_ = feats.get("cmp")
    if cmp_ is not None and cmp_ >= 73.0:
        cap("Gunslinger", 62.0)  # precision is a different archetype family
    if cmp_ is not None and cmp_ < 68.0:
        cap("Distributor", 68.0)  # distributor fits above 68 need 68%+ accuracy
    dt = fits.get("Dual-Threat")
    if dt is not None and dt >= 60.0:
        cap("Developmental", 55.0)  # a real rushing role is a defined archetype
    rush_pg = feats.get("rush_pg")
    if rush_pg is not None and rush_pg >= 95.0:
        cap("Pass-Catching Back", 72.0)  # runner-first: receiving is a trait
    wt = feats.get("wt")
    if wt is not None and wt < 185.0:
        cap("Alpha X", 55.0)  # sub-185 frames do not project as NFL X receivers
    dom = feats.get("dom")
    if dom is not None and dom >= 0.34:
        cap("Playmaker", 85.0)  # at that share he IS the offense's primary
        cap("Deep Threat", 84.0)
    pick = feats.get("pick")
    ms = feats.get("ms")
    yds_pg = feats.get("yds_pg")
    rec_pg = feats.get("rec_pg")
    if pick is not None and pick <= 10:
        cap("Developmental", 60.0)  # the NFL has already issued a role verdict
    if dom is not None and dom >= 0.24 and yds_pg is not None and yds_pg >= 55.0:
        cap("Developmental", 60.0)  # demonstrated WR producer, not a project
    if rec_pg is not None and rec_pg >= 35.0 and ms is not None and ms >= 0.10:
        cap("Developmental", 60.0)  # TE with a defined receiving role


def _starter_volume_seasons(pos: str, seasons: List[Dict]) -> int:
    n = 0
    for s in seasons:
        if pos == "QB":
            ok = (_num(s.get("pass_attempts")) or 0) >= 150
        elif pos == "RB":
            ok = ((_num(s.get("rush_attempts")) or 0)
                  + (_num(s.get("receptions")) or 0)) >= 120
        else:
            ok = (_num(s.get("receptions")) or 0) >= 40
        n += 1 if ok else 0
    return n


def classify_archetype(
    position: str,
    seasons: List[Dict],
    production_score: Optional[float] = None,
    profile: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Score every archetype fit for a prospect and assign the label.

    profile carries the non-season inputs: weight_lbs, forty_yard,
    projected_pick, age. production_score is accepted for call-site
    compatibility and unused: v2 fits read stats and traits only.
    Returns label None (omitted) when the data cannot support one.
    """
    empty = {"archetype": None, "archetype_receipt": None,
             "archetype_confidence": None, "basis_season": None, "fits": {}}
    pos = (position or "").upper()
    rules = ARCHETYPE_RULES.get(pos)
    if not rules or not seasons:
        return dict(empty)
    basis, growth = _pick_basis(pos, seasons)
    if basis is None:
        return dict(empty)
    basis_gp = _num(basis.get("games_played"))
    feats = _features(pos, basis, seasons, profile or {}, growth)

    fits: Dict[str, Optional[float]] = {}
    contribs: Dict[str, List] = {}
    for spec in rules:
        fit, c = _fit(spec, feats)
        fits[spec["label"]] = fit
        contribs[spec["label"]] = c
    _apply_gates(fits, feats)

    ranked = sorted(((k, v) for k, v in fits.items() if v is not None),
                    key=lambda kv: -kv[1])
    out_fits = {k: (round(v, 1) if v is not None else None)
                for k, v in fits.items()}
    if not ranked:
        return {**empty, "basis_season": basis.get("season"), "fits": out_fits}
    top_label, top_fit = ranked[0]
    second_label, second_fit = ranked[1] if len(ranked) > 1 else (None, None)

    presence_keys = PRESENCE_FEATURES[pos]
    presence = (sum(1 for k in presence_keys if feats.get(k) is not None)
                / len(presence_keys))
    if presence < 0.50 or (basis_gp is not None and basis_gp < 5) or top_fit < 40.0:
        return {"archetype": None, "archetype_receipt": None,
                "archetype_confidence": None,
                "basis_season": basis.get("season"), "fits": out_fits}

    margin = (top_fit - second_fit) if second_fit is not None else top_fit
    has_pair = (second_fit is not None and second_fit >= 55.0
                and (top_fit - second_fit) <= 6.0)

    # Projection caps at Medium (spec 2.4).
    win_contribs = contribs.get(top_label, [])
    max_contrib = max((w * s for _, w, s in win_contribs), default=0.0)
    capital_led = any(
        feat == "pick" and (w * s) >= max_contrib and max_contrib > 0
        for feat, w, s in win_contribs
    ) and feats.get("wt") is None and feats.get("forty") is None
    basis_rec = _num(basis.get("receptions"))
    td_led = (
        pos in ("WR", "TE") and basis_rec is not None and basis_rec < 40
        and any(feat == "td_rate" and (w * s) >= max_contrib and max_contrib > 0
                for feat, w, s in win_contribs)
    )
    capped = (
        _starter_volume_seasons(pos, seasons) < 2
        or capital_led
        or td_led
    )

    if has_pair:
        confidence = "Medium"  # a pair is the honest answer, never High or Low
    elif presence >= 0.80 and margin >= 6.5 and not capped:
        confidence = "High"
    elif presence < 0.65 or top_fit < 55.0:
        confidence = "Low"
    else:
        confidence = "Medium"

    label = f"{top_label} · {second_label}" if has_pair else top_label

    # Receipt: the winning fit's top contributing inputs, plus the
    # projection clause the winning conditions require (spec 2.6 and 12).
    labels = RECEIPT_LABELS.get(pos, {})
    parts = []
    for feature, _w, _s in sorted(win_contribs, key=lambda c: -(c[1] * c[2]))[:3]:
        fn = labels.get(feature)
        if fn is not None:
            parts.append(fn(feats))
    receipt = " · ".join(parts)
    if capital_led:
        receipt += "; projection led by draft capital; size and speed unmeasured"
    elif td_led:
        receipt += f"; TD rate rests on {int(basis_rec)} catches"
    elif (feats.get("wt") is None and feats.get("forty") is None
          and presence < 0.85):
        receipt += "; size and speed unmeasured"

    return {
        "archetype": label,
        "archetype_receipt": receipt or None,
        "archetype_confidence": confidence,
        "basis_season": basis.get("season"),
        "fits": out_fits,
    }
