"""Semantic stat and metric phrase resolution for the assistant.

The failure this closes (live, 2026-10-15): "who is top 5 in the most
receiving stats" declined with "that metric is not tracked. The nearest
tracked options are vorp, war, ppr_pts" — a string-distance guess that
offered value metrics for a receiving question. Users name stats the way
they talk about them ("receiving stats", "catches", "carries"), not by
LEADERBOARD_METRICS key. This module maps those phrases semantically:

- a raw counting stat (season totals from the Sleeper week-stat cache:
  receiving yards, receptions, targets, carries, passing yards, ...),
- a tracked metric key (per-game and rate stats live in the AM catalog),
- or a stat FAMILY default: bare "receiving stats" means receiving yards
  (the family's headline stat), with the family's other stats offered as
  the sensible neighbors a decline may name — never unrelated keys.

Pure text logic: no data access, no model. Used by the evidence builders
(hint resolution for metric/leaderboard packs), by the open layer's tool
planner fallback, and by the router's keyword board detection.
"""
from __future__ import annotations

import re

# Raw counting stats: season totals summed from the weekly stat cache.
# sleeper_keys are the stat-line keys in utils.utils.load_sleeper_week_stats
# rows (the same keys utils.projections.week_stat_points scores).
STAT_DEFS = {
    "rec_yd": {"label": "Receiving yards", "sleeper_keys": ("rec_yd",),
               "family": "receiving", "positions": ("WR", "TE", "RB")},
    "rec": {"label": "Receptions", "sleeper_keys": ("rec",),
            "family": "receiving", "positions": ("WR", "TE", "RB")},
    "rec_tgt": {"label": "Targets", "sleeper_keys": ("rec_tgt",),
                "family": "receiving", "positions": ("WR", "TE", "RB")},
    "rec_td": {"label": "Receiving touchdowns", "sleeper_keys": ("rec_td",),
               "family": "receiving", "positions": ("WR", "TE", "RB")},
    "rush_yd": {"label": "Rushing yards", "sleeper_keys": ("rush_yd",),
                "family": "rushing", "positions": ("RB", "QB", "WR")},
    "rush_att": {"label": "Carries", "sleeper_keys": ("rush_att",),
                 "family": "rushing", "positions": ("RB", "QB", "WR")},
    "rush_td": {"label": "Rushing touchdowns", "sleeper_keys": ("rush_td",),
                "family": "rushing", "positions": ("RB", "QB", "WR")},
    "pass_yd": {"label": "Passing yards", "sleeper_keys": ("pass_yd",),
                "family": "passing", "positions": ("QB",)},
    "pass_td": {"label": "Passing touchdowns", "sleeper_keys": ("pass_td",),
                "family": "passing", "positions": ("QB",)},
    "pass_att": {"label": "Pass attempts", "sleeper_keys": ("pass_att",),
                 "family": "passing", "positions": ("QB",)},
    "pass_cmp": {"label": "Completions", "sleeper_keys": ("pass_cmp",),
                 "family": "passing", "positions": ("QB",)},
}

# Family defaults: the headline stat a bare family phrase means, and the
# family's stats in the order a decline should offer them as neighbors.
FAMILIES = {
    "receiving": {"default": "rec_yd",
                  "stats": ("rec_yd", "rec", "rec_tgt", "rec_td")},
    "rushing": {"default": "rush_yd",
                "stats": ("rush_yd", "rush_att", "rush_td")},
    "passing": {"default": "pass_yd",
                "stats": ("pass_yd", "pass_td", "pass_att", "pass_cmp")},
}

# Phrase -> stat key. Matched longest-first on word boundaries against the
# normalized question, so "receiving touchdowns" wins over "receiving".
_STAT_PHRASES = {
    "receiving yards": "rec_yd", "receiving yardage": "rec_yd",
    "rec yards": "rec_yd", "rec yd": "rec_yd",
    "receiving touchdowns": "rec_td", "receiving tds": "rec_td",
    "receiving scores": "rec_td", "rec tds": "rec_td", "rec td": "rec_td",
    "receptions": "rec", "catches": "rec", "rec": "rec",
    "targets": "rec_tgt", "receiving targets": "rec_tgt",
    "rushing yards": "rush_yd", "rushing yardage": "rush_yd",
    "rush yards": "rush_yd", "rush yd": "rush_yd",
    "rushing touchdowns": "rush_td", "rushing tds": "rush_td",
    "rushing scores": "rush_td", "rush tds": "rush_td", "rush td": "rush_td",
    "carries": "rush_att", "rushing attempts": "rush_att",
    "rush attempts": "rush_att",
    "passing yards": "pass_yd", "passing yardage": "pass_yd",
    "pass yards": "pass_yd", "pass yd": "pass_yd",
    "passing touchdowns": "pass_td", "passing tds": "pass_td",
    "pass tds": "pass_td", "pass td": "pass_td",
    "passing attempts": "pass_att", "pass attempts": "pass_att",
    "completions": "pass_cmp", "pass completions": "pass_cmp",
}

# Bare family phrases: "receiving stats" names no single stat, so it
# resolves to the family default with the family recorded for the basis
# line and the neighbor list.
_FAMILY_PHRASES = {
    "receiving stats": "receiving", "receiving numbers": "receiving",
    "receiving production": "receiving", "receiving": "receiving",
    "rushing stats": "rushing", "rushing numbers": "rushing",
    "rushing production": "rushing", "rushing": "rushing",
    "passing stats": "passing", "passing numbers": "passing",
    "passing production": "passing", "passing": "passing",
}

# Natural phrasings of tracked metric keys (the AM catalog). Per-game
# phrasings resolve here instead of to season totals; fantasy-points
# phrasings resolve to the site's own PPR board so "best receiver" and
# "most fantasy points" share one honest basis (stated in the answer).
METRIC_PHRASES = {
    "fantasy points": "ppr_pts", "ppr points": "ppr_pts",
    "points scored": "ppr_pts",
    "fantasy points per game": "ppr_pts_per_game",
    "points per game": "ppr_pts_per_game", "ppg": "ppr_pts_per_game",
    "receiving yards per game": "rec_yards_per_game",
    "rushing yards per game": "rush_yards_per_game",
    "targets per game": "targets_per_game",
    "receptions per game": "receptions_per_game",
    "carries per game": "carries_per_game",
    "total yards": "total_yards",
    "target share": "target_share", "snap share": "snap_share",
    "air yards share": "air_yards_share",
    # Broken tackles ARE missed tackles: the tackles a ball carrier
    # forces defenders to miss (PFR charting). Users ask for them as
    # "missed tackles" or "elusiveness" far more often than by the
    # catalog's "Broken Tackles" labels, and the concept names a family
    # (see METRIC_FAMILIES): the per-carry rate is the headline key.
    "broken tackles per carry": "avoided_tackles_per_carry",
    "broken tackles per catch": "rec_broken_tackles_per_reception",
    "broken tackles per reception": "rec_broken_tackles_per_reception",
    "missed tackles": "avoided_tackles_per_carry",
    "missed tackle": "avoided_tackles_per_carry",
    "missed tackles forced": "avoided_tackles_per_carry",
    "broken tackles": "avoided_tackles_per_carry",
    "broken tackle": "avoided_tackles_per_carry",
    "elusiveness": "avoided_tackles_per_carry",
    "most elusive": "avoided_tackles_per_carry",
}

# Metric concepts that name a FAMILY of tracked metrics, not one board:
# a "what missed tackle metrics do I have" question is only answered
# honestly by naming every member. The headline key (first in "keys")
# carries the board; builders attach the whole family to the pack so
# the answer can enumerate it. Phrases map to family ids.
METRIC_FAMILIES = {
    "broken_tackles": {
        "keys": ("avoided_tackles_per_carry",
                 "rec_broken_tackles_per_reception"),
        "blurb": ("Broken tackles are missed tackles forced by the ball "
                  "carrier (Pro Football Reference charting)."),
    },
}

_CONCEPT_FAMILY_PHRASES = {
    "missed tackles": "broken_tackles",
    "missed tackle": "broken_tackles",
    "missed tackles forced": "broken_tackles",
    "broken tackles": "broken_tackles",
    "broken tackle": "broken_tackles",
    "elusiveness": "broken_tackles",
}


def metric_family_for(text: str) -> dict | None:
    """The tracked-metric family a concept phrase names, else None.

    Returns {"id", "keys", "blurb"}: the family's catalog keys in
    headline-first order and the one-line concept explanation.
    """
    norm = _norm(str(text or "").replace("_", " "))
    if not norm:
        return None
    hit = _find_phrase(norm, _CONCEPT_FAMILY_PHRASES)
    if not hit:
        return None
    family = METRIC_FAMILIES[hit[1]]
    return {"id": hit[1], "keys": family["keys"], "blurb": family["blurb"]}

# "Best receiver" and friends: a superlative over a position means the
# fantasy-points board for that position, basis stated, never a guess at
# which counting stat the user meant.
_BEST_POSITION = (
    (re.compile(r"\b(?:best|top)\s+(?:wide\s+)?receivers?\b|\bbest\s+wr\b"), "WR"),
    (re.compile(r"\b(?:best|top)\s+running\s+backs?\b|\bbest\s+rb\b"), "RB"),
    (re.compile(r"\b(?:best|top)\s+tight\s+ends?\b|\bbest\s+te\b"), "TE"),
    (re.compile(r"\b(?:best|top)\s+quarterbacks?\b|\bbest\s+qb\b"), "QB"),
)


def _norm(text: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", str(text or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def _find_phrase(norm: str, phrases: dict) -> tuple[str, str] | None:
    """(phrase, value) for the longest phrase in the map that appears in
    the normalized text on word boundaries, else None."""
    best = None
    for phrase, value in phrases.items():
        if re.search(rf"\b{re.escape(phrase)}\b", norm):
            if best is None or len(phrase) > len(best[0]):
                best = (phrase, value)
    return best


def resolve_phrase(text: str) -> dict | None:
    """Resolve a natural stat/metric phrase to a board target, or None.

    Returns {"kind": "stat"|"metric", "key", "label", "family",
    "phrase", "position"}: kind stat keys are STAT_DEFS keys (season
    totals), kind metric keys are LEADERBOARD_METRICS keys. "position"
    is set only by the best-<position> rule (fantasy points board).
    """
    norm = _norm(text)
    if not norm:
        return None
    for pattern, pos in _BEST_POSITION:
        if pattern.search(norm):
            return {"kind": "metric", "key": "ppr_pts", "label": "PPR points",
                    "family": None, "phrase": "fantasy points", "position": pos}
    per_game = _find_phrase(norm, METRIC_PHRASES)
    hit = _find_phrase(norm, _STAT_PHRASES)
    # A longer metric phrase wins over a shorter stat phrase it contains
    # ("targets per game" is the metric, not the season-total "targets").
    if per_game and (hit is None or len(per_game[0]) > len(hit[0])):
        phrase, key = per_game
        return {"kind": "metric", "key": key, "label": phrase.title(),
                "family": None, "phrase": phrase, "position": None}
    if hit:
        phrase, key = hit
        spec = STAT_DEFS[key]
        return {"kind": "stat", "key": key, "label": spec["label"],
                "family": spec["family"], "phrase": phrase, "position": None}
    fam = _find_phrase(norm, _FAMILY_PHRASES)
    if fam:
        phrase, family = fam
        key = FAMILIES[family]["default"]
        spec = STAT_DEFS[key]
        return {"kind": "stat", "key": key, "label": spec["label"],
                "family": family, "phrase": phrase, "position": None}
    return None


def resolve_hint(hint: str) -> dict | None:
    """resolve_phrase for a router metric_hint (already extracted text)."""
    return resolve_phrase(hint)


def neighbors_for(text: str) -> list[str]:
    """Sensible semantic neighbors for a decline: the stats of the family
    the text belongs to (labels, headline first), else the families of
    any stat word present, else []. A receiving question's neighbors are
    receiving stats, never unrelated catalog keys."""
    norm = _norm(text)
    if not norm:
        return []
    resolved = resolve_phrase(norm)
    family = resolved.get("family") if resolved else None
    if family is None:
        for word, fam in (("receiv", "receiving"), ("rush", "rushing"),
                          ("pass", "passing")):
            if word in norm:
                family = fam
                break
    if family is None:
        return []
    return [STAT_DEFS[k]["label"] for k in FAMILIES[family]["stats"]]
