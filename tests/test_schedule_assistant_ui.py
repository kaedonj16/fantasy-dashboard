from pathlib import Path


SOURCE = Path("dashboard_services/pages/schedule_page.py").read_text()


def test_range_presets_and_manual_sync_are_derived_from_ranges():
    assert 'class="sched-presets" role="group" aria-label="Schedule range presets"' in SOURCE
    assert 'id="schedRosPreset"' in SOURCE
    assert "applyPreset(CFG.startWeek, CFG.playoffEnd)" in SOURCE
    assert "applyPreset(1, CFG.maxWeek)" in SOURCE
    assert "applyPreset(CFG.playoffStart, CFG.playoffEnd)" in SOURCE
    assert "wkStart === CFG.startWeek && wkEnd === CFG.playoffEnd" in SOURCE
    assert "wkStart === 1 && wkEnd === CFG.maxWeek" in SOURCE
    assert "syncPresetBtns();" in SOURCE
    assert "for (var w = 1; w <= CFG.maxWeek; w++)" in SOURCE


def test_mobile_range_presets_have_a_dedicated_wrapping_row():
    css = Path("static/dashboard.css").read_text()
    assert ".sched-presets" in css
    assert "grid-template-columns: repeat(3, minmax(0, 1fr));" in css
    assert "flex: 0 0 100%;" in css
    assert ".sched-week-range .sched-select" in css


def test_tooltips_explain_metric_seasons_weights_and_not_projection():
    assert "Ease ranks the selected schedule" in SOURCE
    assert "This is not ' + r.name + '’s projected score" in SOURCE
    assert "Early-season ratings blend" in SOURCE
    assert "currentWeight" in SOURCE and "previousWeight" in SOURCE
    assert "Ease is the 0 to 100 score for the selected range" in SOURCE


def test_unavailable_values_are_not_rendered_as_zero():
    assert "c.adjusted_percent != null" in SOURCE


def test_scope_toggle_and_position_pills():
    assert 'data-view="league"' in SOURCE
    assert 'data-view="myteam"' in SOURCE
    assert 'data-pos="all"' in SOURCE
    assert 'data-pos="QB"' in SOURCE
    # The custom player list is gone: no search/add input, no bye banner.
    assert "schedAddInput" not in SOURCE
    assert "Bye Outlook" not in SOURCE


def test_week_by_week_columns_without_summary_columns_or_phone_cards():
    # Rank | Player | Ease, then the week columns: the Adj Avg / Next 3 /
    # Playoffs summary columns and the separate phone cards are gone.
    assert '<th class="sched-th srk-th-ease">Ease</th>' in SOURCE
    assert '<th class="sched-th">Adj Avg</th>' not in SOURCE
    assert '<th class="sched-th">Next 3</th>' not in SOURCE
    assert '<th class="sched-th">Playoffs</th>' not in SOURCE
    assert "srk-mcard" not in SOURCE
    assert "schedRankingsCards" not in SOURCE
    assert "n3Rank" not in SOURCE and "poRank" not in SOURCE


def test_band_spacer_splits_so_ease_hiding_cannot_shift_the_band():
    # Rank+Player spacer is its own pinned cell; the Ease spacer hides with
    # the Ease column on phones instead of shifting the Playoffs band.
    assert '<th colspan="2" class="srk-band-pin"></th><th class="srk-band-ease"></th>' in SOURCE
    css = Path("static/dashboard.css").read_text()
    assert ".srk-band .srk-band-pin" in css
    assert ".schedule-card .srk-ease-cell" in css
