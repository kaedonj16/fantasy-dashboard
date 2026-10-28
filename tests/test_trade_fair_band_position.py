"""Trade Analyzer fair-range band: the navy bar must survive the tooltip upgrader.

Bug (2026-10-27, Kaedon's screenshot): the Trade Summary track under a
155.7 vs 177.7 trade (difference -22.0, verdict "This trade looks very
fair (about 12.4% apart)") showed only the small sky-blue marker tick.
The navy fair-range band in the middle of the track was gone.

Mechanism, proven in a Playwright mirror of the live /trade page (real
markup, real dashboard.min.css, real app.min.js, her exact eval payload
stubbed): markup, navy background and JS width were all correct (band
computed background rgba(14, 40, 71, 0.92), inline width 14.0687%), but
the band's computed position was "relative" and its height 0px, so the
empty div painted nothing. Removing all scripts restored position
absolute / height 20px. The chain:

1. The band markup carries title="Fair range"
   (dashboard_services/pages/trade_calculator_page.py).
2. PR #2592 (e8f0ee62, 2026-10-14, global tooltip positioner) added the
   app.js title upgrader: on load it converts every title= in the app
   into [data-tooltip] (title text copied, title removed).
3. dashboard.css carries a generic [data-tooltip] { position: relative }
   rule (added in the 2026-07-27 dark-mode audit, a45e86c9) at line
   ~18592, far below the balance-bar rules (~5897). Both selectors are
   specificity (0,1,0), so the later tooltip rule overrode the band's
   grouped position: absolute and dropped it back into flow.

Before #2592 the title attribute was inert and the band painted; the
band itself (markup, navy rule, JS width from fair_threshold) never
changed. Fix: root the balance positioning rule at the bar
(.otc-balance-bar .otc-balance-fair, specificity (0,2,0)) so no generic
attribute rule can reposition the band again, whatever attributes the
upgrader or future tooling adds to it.

The share/OG trade card is a separate server-rendered variant
(.bar-fair-zone in app.py): it carries no title, so the upgrader never
touches it, and its positioning lives in its own page CSS. Not affected.
"""
import os
import re

_HERE = os.path.dirname(__file__)
_CSS = os.path.join(_HERE, "..", "static", "dashboard.css")
_JS = os.path.join(_HERE, "..", "static", "app.js")
_PAGE = os.path.join(
    _HERE, "..", "dashboard_services", "pages", "trade_calculator_page.py"
)


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# ── Minimal CSS cascade for the band element ─────────────────────────────
#
# The band element as the app leaves it after the title upgrader runs:
#   <div class="otc-balance-fair" data-tooltip="Fair range">
# inside <div class="otc-balance-bar">. We collect every top-level rule in
# dashboard.css that declares `position` and whose selector matches that
# element, then apply the real cascade (specificity, then source order)
# and look at which position value wins.

_ELEMENT = {
    "tag": "div",
    "id": None,
    "classes": {"otc-balance-fair"},
    "attrs": {"data-tooltip", "class", "style"},
    "ancestors": [
        {"tag": "div", "id": None, "classes": {"otc-balance-bar"}, "attrs": {"class"}}
    ],
}


def _iter_rules(css):
    """Yield (selector_list_text, body_text, source_index) for each rule,
    recursing into @media/@supports blocks (conditions ignored: the band
    and tooltip rules live at top level and apply at every width)."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    rules = []

    def _scan(text):
        i = 0
        while True:
            j = text.find("{", i)
            if j == -1:
                return
            selector = text[i:j].strip()
            depth = 1
            k = j + 1
            while k < len(text) and depth:
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                k += 1
            body = text[j + 1 : k - 1]
            if selector.startswith("@"):
                _scan(body)
            elif selector:
                rules.append((selector, body, len(rules)))
            i = k

    _scan(css)
    return rules


def _specificity(selector):
    ids = len(re.findall(r"#[\w-]+", selector))
    classes = len(re.findall(r"\.[\w-]+", selector))
    attrs = len(re.findall(r"\[[\w-]+", selector))
    pseudo_classes = len(re.findall(r"(?<!:):(?!:)[\w-]+", selector))
    pseudo_elements = len(re.findall(r"::[\w-]+", selector))
    stripped = re.sub(r"#[\w-]+|\.[\w-]+|\[.*?\]|::?[\w-]+(\([^)]*\))?", " ", selector)
    elements = len([t for t in re.split(r"[\s>+~]+", stripped) if t and t != "*"])
    return (ids, classes + attrs + pseudo_classes, elements + pseudo_elements)


def _compound_matches(compound, node):
    if "::" in compound:  # pseudo-element rules style the pseudo, not the element
        return False
    for idsel in re.findall(r"#([\w-]+)", compound):
        if node.get("id") != idsel:
            return False
    for cls in re.findall(r"\.([\w-]+)", compound):
        if cls not in node["classes"]:
            return False
    for attr in re.findall(r"\[([\w-]+)", compound):
        if attr not in node["attrs"]:
            return False
    if ":not(" in compound:  # only :not(.class) forms appear on these rules
        for neg in re.findall(r":not\(\.([\w-]+)\)", compound):
            if neg in node["classes"]:
                return False
    tag = re.match(r"^([a-zA-Z][\w-]*)", compound)
    if tag and tag.group(1) != node["tag"]:
        return False
    return True


def _selector_matches(selector, element):
    if ":hover" in selector or ":focus" in selector:
        return False  # resting state only
    parts = [p for p in re.split(r"\s*>\s*|\s+", selector.strip()) if p]
    if not parts or not _compound_matches(parts[-1], element):
        return False
    ancestors = element["ancestors"]
    ai = 0
    for part in parts[:-1]:
        found = False
        while ai < len(ancestors):
            if _compound_matches(part, ancestors[ai]):
                found = True
                ai += 1
                break
            ai += 1
        if not found:
            return False
    return True


def _declarations(body):
    out = {}
    for chunk in body.split(";"):
        if ":" in chunk:
            prop, _, val = chunk.partition(":")
            out[prop.strip()] = val.strip()
    return out


def _position_candidates():
    css = _read(_CSS)
    candidates = []
    for selector_list, body, order in _iter_rules(css):
        decls = _declarations(body)
        if "position" not in decls:
            continue
        for selector in selector_list.split(","):
            selector = selector.strip()
            if _selector_matches(selector, _ELEMENT):
                candidates.append(
                    {
                        "selector": selector,
                        "specificity": _specificity(selector),
                        "order": order,
                        "position": decls["position"],
                        "decls": decls,
                    }
                )
    return candidates


def _winning_position():
    candidates = _position_candidates()
    assert candidates, "no position rule matches the fair band at all"
    return max(candidates, key=lambda c: (c["specificity"], c["order"]))


# ── The regression ────────────────────────────────────────────────────────

def test_fair_band_position_survives_data_tooltip():
    """With [data-tooltip] on it (what the title upgrader leaves behind),
    the band's winning position must be absolute, not the tooltip rule's
    relative. On the regressed CSS both rules were (0,1,0) and the later
    [data-tooltip] rule won, collapsing the band to 0px tall."""
    winner = _winning_position()
    assert winner["position"] == "absolute", (
        f"band position resolves to {winner['position']!r} via "
        f"{winner['selector']!r}; the generic [data-tooltip] rule must "
        "never reposition the fair band"
    )
    # Full-height anchoring rides with the winning rule.
    assert winner["decls"].get("top") == "0"
    assert winner["decls"].get("bottom") == "0"


def test_fair_band_rule_outranks_generic_attribute_rules():
    """The band's positioning selector must beat a lone class/attribute
    selector on specificity alone, so source order can never flip it."""
    winner = _winning_position()
    assert winner["specificity"] > (0, 1, 0), winner


def test_fair_band_keeps_navy_background():
    css = _read(_CSS)
    m = re.search(r"\.otc-balance-fair\s*\{([^}]*)\}", css)
    assert m, "base .otc-balance-fair rule missing"
    assert "rgba(14, 40, 71, 0.92)" in m.group(1)  # the navy bar Kaedon remembers


def test_band_markup_still_ships_with_its_title():
    page = _read(_PAGE)
    assert '<div class="otc-balance-fair" title="Fair range"></div>' in page
    assert '<div class="otc-balance-bar">' in page


def test_js_still_sizes_band_from_fair_threshold():
    js = _read(_JS)
    assert 'root.querySelector(".otc-balance-fair")' in js
    assert "(ft / maxSideTotal) * 100" in js
    assert "fairZoneEl.style.width = fairWidthPct" in js


# ── Intended geometry (the component's own math) ─────────────────────────
#
# Server: fair threshold = max(7% of the larger side, 25 points)
# (utils/trade.py fair_value_band). JS: the bar maps normalizedDiff
# [-1, +1] across its full width; marker left = ((1 - diff/maxSide) / 2)
# * 100; band width = (fair_threshold / maxSide) * 100, centered. A trade
# the verdict calls "very fair" (|diff| <= threshold) must show its
# marker INSIDE the band, and the band must be visibly wide.

def _geometry(a_total, b_total):
    diff = a_total - b_total
    max_side = max(abs(a_total), abs(b_total), 1)
    threshold = max(max_side * 0.07, 25.0)
    band_width = (threshold / max_side) * 100
    band = (50 - band_width / 2, 50 + band_width / 2)
    marker = ((1 - max(-1, min(1, diff / max_side))) / 2) * 100
    return {"diff": diff, "threshold": threshold, "band": band,
            "band_width": band_width, "marker": marker}


def test_geometry_her_reported_trade():
    g = _geometry(155.7, 177.7)
    assert g["diff"] == -22.0
    assert g["threshold"] == 25.0  # 7% of 177.7 is 12.44; the 25 floor wins
    assert abs(g["band_width"] - 14.07) < 0.01
    assert abs(g["band"][0] - 42.97) < 0.01 and abs(g["band"][1] - 57.03) < 0.01
    assert abs(g["marker"] - 56.19) < 0.01
    # "Very fair" (22.0 <= 25.0): the marker sits inside the band.
    assert g["band"][0] < g["marker"] < g["band"][1]


def test_geometry_spread():
    cases = [
        # (a, b, expect marker inside band, note)
        (150.0, 150.0, True, "even trade: marker dead center"),
        (180.0, 160.0, True, "slight Team 1 edge, still within the fair band"),
        (100.0, 200.0, False, "strong Team 2 edge: marker right of the band"),
        (200.0, 100.0, False, "strong Team 1 edge: marker left of the band"),
        (1.0, 1000.0, False, "extreme blowout: marker clamps at the end"),
    ]
    for a, b, inside, note in cases:
        g = _geometry(a, b)
        assert 0.0 <= g["marker"] <= 100.0, note
        assert g["band_width"] >= 2.5, f"band visibly wide: {note}"
        is_inside = g["band"][0] < g["marker"] < g["band"][1]
        assert is_inside == inside, f"{note}: marker {g['marker']:.2f}, band {g['band']}"
        # Inside exactly when the verdict would call it fair.
        assert is_inside == (abs(g["diff"]) <= g["threshold"]), note


def test_changelog_and_sw_for_fair_band_fix():
    from dashboard_services.changelog import CHANGELOG

    top = CHANGELOG[23]  # prepended above it: breakout grading production-first + hit floor, rooting guide two-per-row + badge fit, Stripe period-end grant fix, awards page rework, graphs fill and audit, portfolio light resolver, trade calculator dedupe, recap/standings dedupe, sitewide speed pass, Weekly Hub dedupe refresh, waivers/start-sit call consolidation, AM batch + players cache, ScoreZone user-cache speedup, My Leagues portfolio consolidation, duplicate Tuesday dot removal, phone tabs + card odds removal, phone notifications panel + More-dot fix, recap glance username fix, Weekly Hub revamp, graphs overhaul, holdings sort + insights speed, the rooting/bye cleanup, the Front Office scan fix
    assert top["date"] == "2026-10-27"
    assert top["tag"] == "fix"
    assert "fair-range" in top["text"]
    assert top["link"] == "/trade"
    assert "—" not in top["text"]  # no em dashes in UI copy
    sw = _read(os.path.join(_HERE, "..", "static", "sw.js"))
    assert "br-fantasy-v132" in sw
