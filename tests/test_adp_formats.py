"""Unit tests for the pure ADP format model, capability metadata, and match
quality classification (dashboard_services.adp_formats). No I/O; runs in the
lightweight CI suite."""

from dashboard_services import adp_formats as F
from dashboard_services.adp_formats import AdpFormat


# ── AdpFormat model ───────────────────────────────────────────────────────────

def test_format_normalizes_and_snaps_ppr():
    f = AdpFormat(draft_type="STARTUP", qb_format="SuperFlex", ppr=1.0 - 1e-9, te_premium=-1, num_teams="12")
    assert f.draft_type == "startup"
    assert f.qb_format == "superflex"
    assert f.ppr == 1.0            # float artifact snapped to canonical
    assert f.te_premium == 0.0     # clamped to >= 0
    assert f.num_teams == 12
    assert f.is_superflex and f.axis == "dynasty"


def test_format_keeps_genuinely_distinct_ppr():
    # A real 0.99 is not a float artifact of 1.0 -> kept as-is.
    assert AdpFormat("redraft", "1qb", 0.99).ppr == 0.99


def test_format_unknown_ppr_and_bad_values():
    f = AdpFormat(draft_type="bogus", qb_format="bogus", ppr="unknown", num_teams="x")
    assert f.draft_type == "redraft"   # invalid -> default
    assert f.qb_format == "1qb"
    assert f.ppr == "unknown"
    assert f.num_teams is None


def test_from_league_maps_axis_and_qb():
    f = AdpFormat.from_league(is_sf=True, scoring_type="dynasty", te_premium=0.5)
    assert f.draft_type == "startup" and f.qb_format == "superflex"
    assert f.te_premium == 0.5 and f.tep_bucket == F.TEP_MODERATE


def test_axis_roundtrip():
    assert F.axis_to_draft_type("dynasty") == "startup"
    assert F.draft_type_to_axis("startup") == "dynasty"
    assert F.axis_to_draft_type("redraft") == "redraft"
    assert F.axis_to_draft_type("rookie") == "rookie"


# ── TEP buckets ───────────────────────────────────────────────────────────────

def test_tep_buckets_thresholds():
    assert F.tep_bucket(0.0) == F.TEP_NONE
    assert F.tep_bucket(0.2) == F.TEP_NONE
    assert F.tep_bucket(0.25) == F.TEP_MODERATE
    assert F.tep_bucket(0.5) == F.TEP_MODERATE
    assert F.tep_bucket(0.74) == F.TEP_MODERATE
    assert F.tep_bucket(0.75) == F.TEP_STRONG
    assert F.tep_bucket(1.0) == F.TEP_STRONG
    assert F.tep_bucket(None) == F.TEP_NONE


# ── Capability metadata ───────────────────────────────────────────────────────

def test_source_capabilities_axes():
    caps = F.SOURCE_CAPABILITIES
    assert caps["sleeper"].serves_axis("dynasty")
    assert caps["brfantasy"].serves_axis("rookie")
    # Global redraft feeds must not claim dynasty/rookie.
    for s in ("yahoo", "espn", "mfl"):
        assert caps[s].serves_axis("redraft")
        assert not caps[s].serves_axis("dynasty")
        assert not caps[s].serves_axis("rookie")


def test_only_brfantasy_declares_native_tep():
    assert F.SOURCE_CAPABILITIES["brfantasy"].provides_tep is True
    for s in ("sleeper", "yahoo", "espn", "mfl"):
        assert F.SOURCE_CAPABILITIES[s].provides_tep is False


# ── Match classification (the plan's priority lists) ──────────────────────────

def test_redraft_ppr_1qb_ranking():
    req = AdpFormat("redraft", "1qb", 1.0)
    assert F.classify_match(req, "sleeper") == F.EXACT
    assert F.classify_match(req, "brfantasy") == F.EXACT
    assert F.classify_match(req, "mfl") == F.COMPATIBLE
    assert F.classify_match(req, "yahoo") == F.GENERIC
    assert F.classify_match(req, "espn") == F.GENERIC


def test_dynasty_sf_tep_prioritizes_brfantasy_excludes_redraft_feeds():
    req = AdpFormat("startup", "superflex", 1.0, te_premium=0.5)
    assert F.classify_match(req, "brfantasy") == F.EXACT
    # SF dynasty without native TEP is a compatible proxy, never exact TEP.
    assert F.classify_match(req, "sleeper") == F.COMPATIBLE
    # Redraft-only global feeds are excluded from a dynasty request.
    for s in ("yahoo", "espn", "mfl"):
        assert F.classify_match(req, s) == F.EXCLUDED


def test_sf_uses_2qb_as_compatible_proxy_not_exact():
    # Sleeper serves SF from its 2QB field: compatible, not exact.
    req = AdpFormat("redraft", "superflex", 1.0)
    assert F.classify_match(req, "sleeper") == F.COMPATIBLE


def test_1qb_request_excludes_sf_only_data():
    # A 1QB request must not be served from SF/2QB-only data.
    from dashboard_services.adp_formats import FormatCapability, classify_capability
    sf_only = FormatCapability(frozenset({"redraft"}), frozenset({"superflex", "2qb"}), ppr=1.0)
    assert classify_capability(AdpFormat("redraft", "1qb", 1.0), sf_only) == F.EXCLUDED


def test_no_tep_request_excludes_tep_only_feed():
    from dashboard_services.adp_formats import FormatCapability, classify_capability
    tep_feed = FormatCapability(frozenset({"startup"}), frozenset({"1qb"}), ppr=1.0, te_premium=1.0)
    # Asking for no-TEP must never pull a TEP feed (its TEs are inflated).
    assert classify_capability(AdpFormat("startup", "1qb", 1.0, te_premium=0.0), tep_feed) == F.EXCLUDED
    # Asking for TEP, a non-TEP feed is a compatible fallback, never exact.
    non_tep = FormatCapability(frozenset({"startup"}), frozenset({"1qb"}), ppr=1.0, te_premium=0.0)
    assert classify_capability(AdpFormat("startup", "1qb", 1.0, te_premium=1.0), non_tep) == F.COMPATIBLE


def test_tep_exact_and_bucket_match():
    from dashboard_services.adp_formats import FormatCapability, classify_capability
    half = FormatCapability(frozenset({"startup"}), frozenset({"1qb"}), ppr=1.0, te_premium=0.5)
    assert classify_capability(AdpFormat("startup", "1qb", 1.0, te_premium=0.5), half) == F.EXACT


def test_scoring_neighbor_is_compatible():
    # Sleeper offers explicit std/half/full; a std-only capability for a PPR
    # request is a compatible (neighbouring) proxy.
    from dashboard_services.adp_formats import FormatCapability, classify_capability
    std = FormatCapability(frozenset({"redraft"}), frozenset({"1qb"}), ppr=0.0)
    assert classify_capability(AdpFormat("redraft", "1qb", 1.0), std) == F.COMPATIBLE


def test_rank_sources_orders_by_quality_and_drops_excluded():
    req = AdpFormat("redraft", "1qb", 1.0)
    ranked = F.rank_sources_by_match(req, ["yahoo", "sleeper", "mfl", "espn", "brfantasy"])
    qualities = [q for _s, q in ranked]
    # exact sources first, then compatible, then generic; monotonic non-improving.
    idx = [F.MATCH_QUALITY_ORDER.index(q) for q in qualities]
    assert idx == sorted(idx)
    assert ("sleeper", F.EXACT) in ranked
    assert dict(ranked)["mfl"] == F.COMPATIBLE


def test_rank_sources_dynasty_drops_redraft_feeds():
    req = AdpFormat("startup", "superflex", 1.0, te_premium=0.5)
    ranked = dict(F.rank_sources_by_match(req, ["sleeper", "brfantasy", "yahoo", "espn", "mfl"]))
    assert set(ranked) == {"sleeper", "brfantasy"}   # redraft feeds excluded
    assert ranked["brfantasy"] == F.EXACT
