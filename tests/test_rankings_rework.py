"""Player Rankings rework (approved mock rounds 1-5, 2026-10-30).

Presentation only: merged header rows, a sliding-pill sort, a headshot
cell with the modal treatment, age in the subline, and one centered
numeric grid. No values, sorting logic, filter logic or data change.
The rankings payload already carries bDay + espnHeadshot (stamped from
the players index server side); rankings.js used to drop espnHeadshot in
its whitelist map, so the row renderer never saw it.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = (ROOT / "dashboard_services" / "pages" / "players_page.py").read_text(encoding="utf-8")
JS = (ROOT / "static" / "rankings.js").read_text(encoding="utf-8")
APP = (ROOT / "app.py").read_text(encoding="utf-8")


def test_header_row1_title_format_toggles_and_search():
    assert '<h2 class="page-title">Player Rankings</h2>' in PAGE
    assert 'id="prSubtitle"' in PAGE
    # Format toggle: the redraft side is labeled "Redraft" on this page;
    # the internal mode value stays redraft.
    assert "onclick=\"prSetScoringType('dynasty')\">Dynasty</button>" in PAGE
    assert "onclick=\"prSetScoringType('redraft')\">Redraft</button>" in PAGE
    assert "onclick=\"prSetLeagueType('1qb')\">1QB</button>" in PAGE
    assert "onclick=\"prSetLeagueType('sf')\">SF</button>" in PAGE
    assert 'id="prSearch"' in PAGE


def test_header_row2_sliding_sort_pill_and_live_position_pills():
    # Sliding pill: Teams page toggle classes + thumb, five sort options.
    assert 'class="teams-mode-toggle pr-sort-toggle"' in PAGE
    assert '<span class="teams-mode-thumb" aria-hidden="true"></span>' in PAGE
    for key, label in [("value", "Value"), ("ppg", "PPG"), ("age", "Age"), ("pos_rank", "Pos Rank")]:
        assert f'data-sortkey="{key}"' in PAGE, key
        assert f">{label}</button>" in PAGE, label
    assert 'data-sortkey="total_pts"' in PAGE
    assert '<span class="pr-sort-full">Total Points</span>' in PAGE
    # The thumb measures the active button, like _modeThumb on Teams.
    assert "thumb.style.width = br.width + 'px';" in JS
    assert "thumb.style.transform = 'translateX(' + (br.left - wr.left - borderL) + 'px)';" in JS
    assert "function prSyncSortToggle()" in JS
    # Position filter pills exactly as live: canonical classes, untouched.
    assert 'class="otc-day-filter pos-pill active" data-pos="ALL"' in PAGE
    assert 'class="otc-day-filter pos-pill" data-pos="PICK"' in PAGE
    # Legend / Settings pinned right on desktop.
    assert 'id="prLegendBtn"' in PAGE
    assert 'id="prSettingsBtn"' in PAGE
    assert 'class="pr-pinright"' in PAGE


def test_desktop_grid_one_track_set_and_centered_numerics():
    # Header inline tracks and the row rule are the same five tracks.
    assert "grid-template-columns:50px minmax(0,1fr) 64px 96px 108px;" in PAGE
    assert "grid-template-columns: 50px minmax(0, 1fr) 64px 96px 108px;" in PAGE
    assert ".pr-ppg { text-align: center;" in PAGE
    assert ".pr-total { text-align: center;" in PAGE
    assert ".pr-value {\n        text-align: center;\n        font-size: 19px;" in PAGE
    # Header labels for the three numeric columns.
    assert 'id="prPpgHeader"' in PAGE
    assert 'id="prTotalHeader"' in PAGE
    assert 'id="prSortHeader"' in PAGE


def test_headshot_cell_geometry_and_fallback_chain():
    # Modal treatment: 62px desktop / 50px phone, full row height, square.
    assert ".pr-hs {\n        width: 62px;\n        height: 100%;" in PAGE
    assert "object-fit: cover;" in PAGE
    assert "object-position: top;" in PAGE
    assert ".pr-hs { width: 50px; }" in PAGE
    assert ".pr-player-row {\n        padding: 0 12px;\n        min-height: 56px;" in PAGE
    # JS: ESPN combiner via the shared helper, Sleeper fallback, initials
    # disc hidden unless every source fails.
    assert "function prHeadshotHTML(p, isPick)" in JS
    assert "prHiResHeadshot(espn, 128)" in JS
    assert "function prHiResHeadshot(url, width)" in JS
    assert "https://sleepercdn.com/content/nfl/players/thumb/" in JS
    assert "function prImgFallback(img)" in JS
    assert "mono.style.display = 'flex'" in JS or "mono) mono.style.display = 'flex'" in JS
    # The whitelist map now carries the fields the payload already sends.
    assert "espnHeadshot:" in JS and "p.espnHeadshot || ''" in JS
    assert "bDay:" in JS and "p.bDay || null" in JS
    # SSR rows mirror the same cell.
    assert 'class="pr-hs"' in APP
    assert "_hires_headshot_url(_espn, 128)" in APP


def test_subline_age_from_payload_age_with_separator():
    # Age segment: muted middle-dot separator element, then "Age X.X"
    # from the precise age (calculateAgeFromBirthday over payload bDay).
    assert '<span class="pr-age-sep" aria-hidden="true">·</span>' in JS
    assert "Age ${Number(p.age).toFixed(1)}" in JS
    assert "p.age != null" in JS  # omitted entirely when the payload has none
    assert "calculateAgeFromBirthday(p.bDay)" in JS
    # SSR mirrors it.
    assert "pr-age-inline" in APP
    # Position stays plain text via the live join, never a badge.
    assert "[posRank, (p.team && p.team !== '–') ? p.team : ''].filter(Boolean).join(' · ')" in JS


def test_tier_dividers_unchanged():
    assert ".pr-tier-divider {" in PAGE
    assert "prGetTier(p)" in JS
    assert "'pr-tier-divider'" in JS or '"pr-tier-divider"' in JS or "pr-tier-divider" in JS


def test_changelog_entry_present_by_content():
    from pin_helpers import changelog_entry

    entry = changelog_entry("Player Rankings, reworked", link="/players")
    assert entry is not None


def test_headshots_are_right_sized_and_lazy():
    # Page-speed acceptance: combiner width 128 for a 62px / 50px cell
    # (~2x, retina-sharp) -- never the 600px original as the primary src.
    assert "prHiResHeadshot(espn, 128)" in JS
    assert "_hires_headshot_url(_espn, 128)" in APP
    # The site's own lazy pattern (loading="lazy" decoding="async") plus
    # explicit dimensions; the cell's fixed CSS width + row min-height
    # reserve the box, so a late image causes zero layout shift.
    assert 'loading="lazy" decoding="async" width="62" height="56"' in JS
    assert 'loading="lazy" decoding="async" width="62" height="56"' in APP
    assert ".pr-photo {" in PAGE and "width: 62px;" in PAGE
