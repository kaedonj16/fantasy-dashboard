"""Normalized ADP format model, source capability metadata, and match quality.

This is the pure-logic half of the ADP source-adapter architecture: it never
performs I/O, so it imports cleanly in the lightweight CI suite and every rule
here is unit-testable in isolation. ``adp_service`` (the I/O half) re-exports the
public names, so callers may import either module.

The model exists so every normalized ADP datapoint records *exactly* what it
represents — including the dimensions a source leaves unspecified. A source that
publishes one global ADP with no scoring split is recorded as ``ppr="unknown"``
rather than being mislabelled PPR; a superflex request served from a 2QB feed is
recorded as a ``compatible`` proxy rather than an ``exact`` superflex match. We
never fabricate specificity we did not observe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

# ── Vocabulary ────────────────────────────────────────────────────────────────
# Sentinels for a dimension a source does not resolve. ``mixed`` means the feed
# aggregates several values of that dimension together (e.g. Yahoo's global ADP
# blends every league's scoring); ``unknown`` means the value is simply not
# reported. They are treated the same way by matching (never an exact match) but
# kept distinct so a snapshot records which is true.
MIXED = "mixed"
UNKNOWN = "unknown"

DRAFT_TYPES = ("redraft", "startup", "rookie")
QB_FORMATS = ("1qb", "2qb", "superflex", MIXED)

# Match-quality classes, best to worst. ``excluded`` means the source must not be
# used for the request at all (e.g. a redraft-only feed for a dynasty request).
EXACT = "exact"
COMPATIBLE = "compatible"
GENERIC = "generic"
EXCLUDED = "excluded"
MATCH_QUALITY_ORDER = (EXACT, COMPATIBLE, GENERIC, EXCLUDED)

# The resolver's historical scoring *axis* names map onto AdpFormat.draft_type.
# "dynasty" is the axis a caller asks for; "startup" is the draft_type a dynasty
# startup draft actually is. Rookie and redraft share their name across both.
AXIS_TO_DRAFT_TYPE = {"dynasty": "startup", "redraft": "redraft", "rookie": "rookie"}
DRAFT_TYPE_TO_AXIS = {"startup": "dynasty", "redraft": "redraft", "rookie": "rookie"}


def axis_to_draft_type(axis: str) -> str:
    return AXIS_TO_DRAFT_TYPE.get(str(axis or "").lower(), "redraft")


def draft_type_to_axis(draft_type: str) -> str:
    return DRAFT_TYPE_TO_AXIS.get(str(draft_type or "").lower(), "redraft")


# ── TEP buckets ───────────────────────────────────────────────────────────────
# Exact TE-premium samples are sparse outside of BR Fantasy's observed drafts, so
# a requested premium is snapped to a documented bucket. Thresholds are on the
# additional per-reception TE premium (points added to a TE catch on top of the
# league's base reception scoring).
TEP_NONE = "none"        # no TE premium
TEP_MODERATE = "moderate"  # a half-point-ish premium
TEP_STRONG = "strong"      # a full-point-ish premium

# (inclusive_low, exclusive_high, bucket) — a premium of exactly the boundary
# falls in the higher bucket's low edge. 0.25 and 0.75 are the midpoints between
# the canonical 0 / 0.5 / 1.0 premiums.
TEP_BUCKET_BOUNDS: Tuple[Tuple[float, float, str], ...] = (
    (0.0, 0.25, TEP_NONE),
    (0.25, 0.75, TEP_MODERATE),
    (0.75, float("inf"), TEP_STRONG),
)

# Canonical exact premiums we expose as first-class options.
TEP_EXACT_VALUES = (0.0, 0.5, 1.0)


def tep_bucket(te_premium: Union[float, int, None]) -> str:
    """Snap a numeric TE premium to its documented bucket."""
    try:
        v = float(te_premium or 0.0)
    except (TypeError, ValueError):
        v = 0.0
    if v < 0:
        v = 0.0
    for low, high, name in TEP_BUCKET_BOUNDS:
        if low <= v < high:
            return name
    return TEP_STRONG


# ── Normalized format model ───────────────────────────────────────────────────
@dataclass(frozen=True)
class AdpFormat:
    """What a single normalized ADP dataset (or a request for one) represents.

    Every dimension is explicit. ``ppr`` is a float (0 / 0.5 / 1.0) or the
    string ``"unknown"``; ``qb_format`` may be ``"mixed"`` for a feed that blends
    QB formats; ``te_premium`` is the additional per-reception TE premium (0.0
    means none). ``num_teams`` is optional and ``None`` when the feed does not
    resolve league size.
    """
    draft_type: str = "redraft"
    qb_format: str = "1qb"
    ppr: Union[float, str] = 1.0
    te_premium: float = 0.0
    num_teams: Optional[int] = None

    def __post_init__(self):
        # Normalize without mutating a frozen dataclass by going through object.
        dt = str(self.draft_type or "redraft").lower()
        object.__setattr__(self, "draft_type", dt if dt in DRAFT_TYPES else "redraft")
        qb = str(self.qb_format or "1qb").lower()
        object.__setattr__(self, "qb_format", qb if qb in QB_FORMATS else "1qb")
        if isinstance(self.ppr, str):
            object.__setattr__(self, "ppr", UNKNOWN if self.ppr in (UNKNOWN, MIXED) else _coerce_ppr(self.ppr))
        else:
            object.__setattr__(self, "ppr", _coerce_ppr(self.ppr))
        try:
            object.__setattr__(self, "te_premium", max(0.0, float(self.te_premium or 0.0)))
        except (TypeError, ValueError):
            object.__setattr__(self, "te_premium", 0.0)
        if self.num_teams is not None:
            try:
                object.__setattr__(self, "num_teams", int(self.num_teams))
            except (TypeError, ValueError):
                object.__setattr__(self, "num_teams", None)

    # Convenience views ────────────────────────────────────────────────────────
    @property
    def axis(self) -> str:
        return draft_type_to_axis(self.draft_type)

    @property
    def is_superflex(self) -> bool:
        """Whether this format wants two-QB-slot pricing (SF or 2QB)."""
        return self.qb_format in ("superflex", "2qb")

    @property
    def tep_bucket(self) -> str:
        return tep_bucket(self.te_premium)

    @property
    def wants_tep(self) -> bool:
        return self.te_premium > 0.0

    def to_dict(self) -> Dict:
        return {
            "draft_type": self.draft_type,
            "qb_format": self.qb_format,
            "ppr": self.ppr,
            "te_premium": self.te_premium,
            "num_teams": self.num_teams,
        }

    @classmethod
    def from_league(cls, *, is_sf: bool, scoring_type: str = "redraft",
                    ppr: Union[float, str] = 1.0, te_premium: float = 0.0,
                    num_teams: Optional[int] = None) -> "AdpFormat":
        """Build a request format from the legacy (is_sf, scoring_type) callers."""
        return cls(
            draft_type=axis_to_draft_type(scoring_type),
            qb_format="superflex" if is_sf else "1qb",
            ppr=ppr,
            te_premium=te_premium,
            num_teams=num_teams,
        )


def _coerce_ppr(value) -> Union[float, str]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return UNKNOWN
    # Snap to the canonical scoring points so 0.99 doesn't miss an == on 1.0.
    for canon in (0.0, 0.5, 1.0):
        if abs(v - canon) < 1e-6:
            return canon
    return v


# ── Source capability metadata ────────────────────────────────────────────────
@dataclass(frozen=True)
class FormatCapability:
    """One native format a source can serve, plus how specific that data is.

    ``draft_types`` / ``qb_formats`` list the values this capability covers.
    ``ppr`` and ``te_premium`` may be a concrete value or ``UNKNOWN``/``MIXED``
    when the feed does not resolve them. ``specificity`` caps the best match a
    capability can yield: a global aggregate feed is capped at ``GENERIC`` no
    matter how well its dimensions line up, because it is not format-specific.
    """
    draft_types: FrozenSet[str]
    qb_formats: FrozenSet[str] = field(default_factory=lambda: frozenset({"1qb"}))
    ppr: Union[float, str] = UNKNOWN
    te_premium: Union[float, str] = 0.0
    specificity: str = EXACT  # ceiling on match quality: EXACT | COMPATIBLE | GENERIC


@dataclass(frozen=True)
class SourceCapability:
    """Everything the resolver and UI need to know about one ADP source."""
    key: str
    display_name: str
    scope: str                       # "global" | "observed" | "platform"
    formats: Tuple[FormatCapability, ...]
    provides_tep: bool = False       # native, per-reception TE-premium ADP
    real_vs_mock_known: bool = False
    league_size_known: bool = False
    requires_auth: bool = False
    notes: str = ""

    def axes(self) -> FrozenSet[str]:
        out = set()
        for f in self.formats:
            out |= {draft_type_to_axis(dt) for dt in f.draft_types}
        return frozenset(out)

    def serves_axis(self, axis: str) -> bool:
        return axis in self.axes()


# The canonical capability declarations. Each records only what the feed has been
# verified to expose (Sleeper's explicit scoring fields, BR Fantasy's observed
# draft settings) or what its public contract documents (the global aggregates).
# Global feeds are capped at GENERIC because they publish one blended ADP.
SOURCE_CAPABILITIES: Dict[str, SourceCapability] = {
    "sleeper": SourceCapability(
        key="sleeper", display_name="Sleeper", scope="global",
        formats=(
            # Redraft: explicit std / half / full PPR, plus a 2QB field.
            FormatCapability(frozenset({"redraft"}), frozenset({"1qb"}), ppr=0.0),
            FormatCapability(frozenset({"redraft"}), frozenset({"1qb"}), ppr=0.5),
            FormatCapability(frozenset({"redraft"}), frozenset({"1qb"}), ppr=1.0),
            FormatCapability(frozenset({"redraft"}), frozenset({"2qb", "superflex"}), ppr=1.0,
                             specificity=COMPATIBLE),  # adp_2qb used as SF proxy
            # Dynasty startup: explicit std / half / full PPR + 2QB.
            FormatCapability(frozenset({"startup"}), frozenset({"1qb"}), ppr=0.0),
            FormatCapability(frozenset({"startup"}), frozenset({"1qb"}), ppr=0.5),
            FormatCapability(frozenset({"startup"}), frozenset({"1qb"}), ppr=1.0),
            FormatCapability(frozenset({"startup"}), frozenset({"2qb", "superflex"}), ppr=1.0,
                             specificity=COMPATIBLE),
            # Rookie: explicit rookie field where present.
            FormatCapability(frozenset({"rookie"}), frozenset({"1qb", "2qb", "superflex"}),
                             ppr=UNKNOWN),
        ),
        notes="Explicit scoring-format ADP fields; treats 999 as undrafted. "
              "No native TE-premium field. Superflex served from the 2QB field.",
    ),
    "brfantasy": SourceCapability(
        key="brfantasy", display_name="BR Fantasy", scope="observed",
        formats=(
            # Observed drafts: every axis, both QB formats, and real TE-premium
            # drafts kept separate from normal PPR.
            FormatCapability(frozenset({"redraft", "startup", "rookie"}),
                             frozenset({"1qb", "2qb", "superflex"}), ppr=UNKNOWN),
            FormatCapability(frozenset({"startup", "redraft"}),
                             frozenset({"1qb", "2qb", "superflex"}),
                             ppr=UNKNOWN, te_premium=0.5),
            FormatCapability(frozenset({"startup", "redraft"}),
                             frozenset({"1qb", "2qb", "superflex"}),
                             ppr=UNKNOWN, te_premium=1.0),
        ),
        provides_tep=True, real_vs_mock_known=True, league_size_known=True,
        notes="Observed real/mock drafts crawled from league settings; the only "
              "native TE-premium source.",
    ),
    # Same observed-draft capability as BR Fantasy, but limited to drafts from
    # the past N days (Live ADP). Selector-only — never blended into Consensus
    # (that would double-count recent drafts already in season-long BR Fantasy).
    "brfantasy_live": SourceCapability(
        key="brfantasy_live", display_name="BR Fantasy Live (7d)", scope="observed",
        formats=(
            FormatCapability(frozenset({"redraft", "startup", "rookie"}),
                             frozenset({"1qb", "2qb", "superflex"}), ppr=UNKNOWN),
            FormatCapability(frozenset({"startup", "redraft"}),
                             frozenset({"1qb", "2qb", "superflex"}),
                             ppr=UNKNOWN, te_premium=0.5),
            FormatCapability(frozenset({"startup", "redraft"}),
                             frozenset({"1qb", "2qb", "superflex"}),
                             ppr=UNKNOWN, te_premium=1.0),
        ),
        provides_tep=True, real_vs_mock_known=True, league_size_known=True,
        notes="Rolling window over recently started BR Fantasy observed drafts "
              "(default past 7 days). Not part of Consensus.",
    ),
    "yahoo": SourceCapability(
        key="yahoo", display_name="Yahoo", scope="global",
        formats=(
            FormatCapability(frozenset({"redraft"}), frozenset({MIXED}),
                             ppr=UNKNOWN, specificity=GENERIC),
        ),
        notes="Public global redraft ADP (no login). Scoring and QB format are "
              "mixed/global, not a specific format.",
    ),
    "espn": SourceCapability(
        key="espn", display_name="ESPN", scope="global",
        formats=(
            FormatCapability(frozenset({"redraft"}), frozenset({MIXED}),
                             ppr=UNKNOWN, specificity=GENERIC),
        ),
        notes="Public global redraft ADP (no login) from averageDraftPosition. "
              "A separate PPR draft-room rank is stored but never mixed into ADP "
              "consensus.",
    ),
    "mfl": SourceCapability(
        key="mfl", display_name="MFL", scope="global",
        formats=(
            # Only redraft PPR / standard are verified via IS_PPR. Keeper, rookie,
            # dynasty, SF and TEP are NOT inferable from MFL's ADP filters, so no
            # capability is declared for them until verified from returned data.
            FormatCapability(frozenset({"redraft"}), frozenset({"1qb"}), ppr=1.0,
                             specificity=COMPATIBLE),
            FormatCapability(frozenset({"redraft"}), frozenset({"1qb"}), ppr=0.0,
                             specificity=COMPATIBLE),
        ),
        real_vs_mock_known=True, league_size_known=True,
        notes="Free ADP export. Verified filters: IS_PPR, FCOUNT (league size), "
              "IS_MOCK, PERIOD. Dynasty/rookie/SF/TEP are NOT exposed by MFL's "
              "ADP filters and are recorded as unknown.",
    ),
}


def source_capability(source: str) -> Optional[SourceCapability]:
    return SOURCE_CAPABILITIES.get(str(source or "").lower())


# ── Match classification ──────────────────────────────────────────────────────
def _qb_match(requested: str, offered: FrozenSet[str]) -> str:
    """Match quality for the QB dimension alone."""
    if requested == MIXED or MIXED in offered:
        return GENERIC
    if requested in offered:
        return EXACT
    # Superflex <-> 2QB are close proxies for each other (two QB-capable slots).
    sf_like = {"superflex", "2qb"}
    if requested in sf_like and (offered & sf_like):
        return COMPATIBLE
    # A 1QB request served from an SF/2QB feed (or vice-versa) is a poor proxy.
    return EXCLUDED


def _ppr_match(requested: Union[float, str], offered: Union[float, str]) -> str:
    # Two separate ideas are kept apart here. UNKNOWN means the source simply
    # does not split on scoring (an observed-draft feed captures real scoring
    # per draft even if the aggregate blends them), so it is non-constraining and
    # does not downgrade a match. MIXED means the feed publishes one deliberately
    # blended number; that is a weaker signal and pins the dimension at GENERIC.
    # (Truly non-specific feeds are additionally capped by FormatCapability
    # .specificity, so ppr never has to carry that job alone.)
    if requested == UNKNOWN or offered == UNKNOWN:
        return EXACT
    if offered == MIXED:
        return GENERIC
    try:
        diff = abs(float(requested) - float(offered))
    except (TypeError, ValueError):
        return GENERIC
    if diff < 1e-6:
        return EXACT
    # A neighbouring scoring (half vs full, std vs half) is a usable proxy.
    return COMPATIBLE


def _tep_match(requested: float, offered: Union[float, str]) -> str:
    if offered in (UNKNOWN, MIXED):
        # A feed that doesn't resolve TEP serves a no-TEP request fine, but for a
        # TEP request it is only a non-TEP fallback — never an exact TEP match.
        # We never manufacture TEP by moving tight ends up a non-TEP board.
        return COMPATIBLE if requested > 0 else EXACT
    try:
        offered_f = float(offered)
    except (TypeError, ValueError):
        return GENERIC
    if abs(requested - offered_f) < 1e-6:
        return EXACT
    if tep_bucket(requested) == tep_bucket(offered_f):
        return COMPATIBLE
    # Different premium magnitude. A no-TEP feed for a TEP request is a
    # compatible fallback; a TEP feed for a no-TEP request is wrong (its tight
    # ends are inflated) and must not be used.
    if requested == 0 and offered_f > 0:
        return EXCLUDED
    return COMPATIBLE


def _worst(*qualities: str) -> str:
    """The lowest (worst) match quality among the dimensions."""
    worst = EXACT
    for q in qualities:
        if MATCH_QUALITY_ORDER.index(q) > MATCH_QUALITY_ORDER.index(worst):
            worst = q
    return worst


def classify_capability(requested: AdpFormat, cap: FormatCapability) -> str:
    """Best match quality one capability can offer the request."""
    if requested.draft_type not in cap.draft_types:
        return EXCLUDED
    qb = _qb_match(requested.qb_format, cap.qb_formats)
    if qb == EXCLUDED:
        return EXCLUDED
    ppr = _ppr_match(requested.ppr, cap.ppr)
    tep = _tep_match(requested.te_premium, cap.te_premium)
    if tep == EXCLUDED:
        return EXCLUDED
    quality = _worst(qb, ppr, tep)
    # A capability can be capped below EXACT (e.g. global aggregate feeds).
    if MATCH_QUALITY_ORDER.index(cap.specificity) > MATCH_QUALITY_ORDER.index(quality):
        quality = cap.specificity
    return quality


def classify_match(requested: AdpFormat, source: str) -> str:
    """Best match quality a whole source can offer the request.

    Returns one of exact | compatible | generic | excluded. This is the single
    gate the consensus resolver uses to decide whether — and how strongly — to
    trust a source for a given request.
    """
    cap = source_capability(source)
    if cap is None:
        return EXCLUDED
    best = EXCLUDED
    for fmt in cap.formats:
        q = classify_capability(requested, fmt)
        if MATCH_QUALITY_ORDER.index(q) < MATCH_QUALITY_ORDER.index(best):
            best = q
        if best == EXACT:
            break
    return best


def rank_sources_by_match(requested: AdpFormat, sources) -> List[Tuple[str, str]]:
    """[(source, quality)] for the given sources, best match first, excluded dropped.

    Ties keep the caller's input order (a stable sort), so a preferred-source
    ordering passed in is respected within a quality tier.
    """
    scored = []
    for i, s in enumerate(sources):
        q = classify_match(requested, s)
        if q != EXCLUDED:
            scored.append((MATCH_QUALITY_ORDER.index(q), i, s, q))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [(s, q) for _rank, _i, s, q in scored]
