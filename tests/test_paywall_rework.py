"""PRO paywall rework (page-reworks round 2 mock): presentation only.

Pins the approved redesign of /pricing:
- Plans headrow (heading + billing toggle on one row) with the trial strip
  inside the plans section, directly above the plan grid.
- A PRO metrics section grouping only metrics in the real PRO_METRICS set
  (xfp_stddev is not gated and must never be labeled PRO).
- "What remains free" in its styled section, wording verbatim.
- FAQ as one card, first item open by default, all 9 Q/A verbatim.
- Every icon class used by the pricing body has a mapping in the site's
  icon system (static/icons.css PNG mappings + glyph fallbacks).
- paywall.css carries the rework rules (featured lift, FAQ card, chips).
No copy, price, or coverage changes.
"""
import re
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]
PAYWALL_CSS = (ROOT / "static" / "paywall.css").read_text(encoding="utf-8")
ICONS_CSS = (ROOT / "static" / "icons.css").read_text(encoding="utf-8")


def _pricing_main(html: str) -> str:
    start = html.index('<main class="pricing-page">')
    end = html.index("</main>", start)
    return html[start:end]


def test_trial_strip_inside_plans_section_above_grid(offline_client):
    main = _pricing_main(offline_client.get("/pricing").get_data(as_text=True))
    plans_at = main.index('class="pricing-section pricing-plans"')
    trial_at = main.index("pricing-trial-card")
    grid_at = main.index("pricing-plan-grid")
    assert plans_at < trial_at < grid_at
    assert "pricing-plans-headrow" in main
    headrow = main[main.index("pricing-plans-headrow"):trial_at]
    assert "Choose your coverage" in headrow
    assert "billing-toggle" in headrow


def test_pro_metrics_section_labels_only_gated_metrics(offline_client):
    from data_building.advanced_metrics import PRO_METRICS

    main = _pricing_main(offline_client.get("/pricing").get_data(as_text=True))
    assert 'id="pro-metrics-title">PRO metrics<' in main
    for group in ("Efficiency", "Opportunity", "Consistency", "Value"):
        assert f"<h3>{group}</h3>" in main
    for label in (
        "Points over expected, per game",
        "Points over expected",
        "Target Quality",
        "WOPR",
        "Opportunity trend",
        "xFP trend",
        "Role Score",
        "Consistency (fantasy point CV)",
        "Value over replacement (VORP)",
        "WAR",
    ):
        assert label in main
    # The labeled entries map onto real gated keys only.
    assert "xfp_stddev" not in PRO_METRICS
    assert "xfp_stddev" not in main
    assert "fp_cv" in PRO_METRICS and "vorp" in PRO_METRICS and "war" in PRO_METRICS


def test_free_section_wording_verbatim(offline_client):
    main = _pricing_main(offline_client.get("/pricing").get_data(as_text=True))
    assert 'class="pricing-section pricing-free-section"' in main
    for label in (
        "Trade calculator &amp; Sleeper comps",
        "Advanced Metrics",
        "Auction Values",
        "Live cheat-sheet overlay &amp; CSV",
    ):
        assert label in main
    assert "ADP rankings, basic player data, and the non-storyline sections of Weekly Recap also remain free." in main


def test_faq_card_first_open_all_nine_verbatim(offline_client):
    import routes.billing_bp as billing

    main = _pricing_main(offline_client.get("/pricing").get_data(as_text=True))
    faq = main[main.index("pricing-faq-card"):main.index("</section>", main.index("pricing-faq-card"))]
    items = re.findall(r"<details( open)?><summary>(.*?)</summary><p>(.*?)</p></details>", faq, re.S)
    assert len(items) == 9
    assert items[0][0] == " open"
    assert all(not flag for flag, _q, _a in items[1:])
    # Rendered FAQ matches the canonical _PRICING_FAQ list verbatim, modulo
    # apostrophe style (the page copy uses a typographic apostrophe in one
    # answer where the canonical list uses a straight one).
    straight = lambda s: s.replace("’", "'")
    rendered = [(straight(q), straight(a)) for _flag, q, a in items]
    expected = [(straight(q), straight(a)) for q, a in billing._PRICING_FAQ]
    assert rendered == expected


def test_pricing_body_icons_all_mapped(offline_client):
    main = _pricing_main(offline_client.get("/pricing").get_data(as_text=True))
    used = set(re.findall(r"fa-solid (fa-[a-z0-9-]+)", main))
    assert used, "expected icons on the pricing page"
    unmapped = {name for name in used if f".{name}" not in ICONS_CSS}
    assert not unmapped, f"icons with no site mapping: {sorted(unmapped)}"
    for gone in (
        "fa-handshake", "fa-calendar-week", "fa-gauge-high", "fa-wave-square",
        "fa-calculator", "fa-table", "fa-gavel", "fa-file-csv",
    ):
        assert gone not in used


def test_paywall_css_rework_rules():
    assert ".pricing-page .pricing-faq .pricing-faq-card" in PAYWALL_CSS
    assert "translateY(-6px)" in PAYWALL_CSS
    assert ".pricing-option.featured { transform:none; }" in PAYWALL_CSS
    assert ".pricing-benefit-groups.pricing-metrics-groups" in PAYWALL_CSS
    assert "grid-template-columns:repeat(2, minmax(0, 1fr))" in PAYWALL_CSS
    assert ".pricing-free-section" in PAYWALL_CSS
    assert ".pricing-ichip" in PAYWALL_CSS
    assert "pricing-plans-headrow" in PAYWALL_CSS
    assert "inset 3px 0 0 var(--accent)" in PAYWALL_CSS
