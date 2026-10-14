"""Position-dependent usage/role sections + team strip margin (2026-10-14).

Kaedon, looking at Jahmyr Gibbs' Team tab: "for all usage/role sections,
it should be pos dependent, like why is a rb showing target info in the
usage." The Usage by Week card charted targets for every position. Now
the metric follows the position (RB touches, QB pass attempts, WR/TE
targets) with a fallback through the other series when the position's
own metric is missing, and the Overview Role card gains a per-game usage
line for RBs (touches) and QBs (pass attempts). The weekly series
therefore carries carries/touches/pass_att alongside targets.

She also flagged the pm-team-strip: it is the first child of the
full-bleed .pm-team-wrap, so with only a bottom margin its border sat
flush on the modal edges. It now carries the same 18px side inset as
the Team tab's other sections.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def test_usage_card_metric_follows_position():
    assert "pos === 'RB' ? 'touches' : pos === 'QB' ? 'pass_att' : 'targets'" in JS
    assert "{ key: 'touches', noun: 'Touches', low: 'touches', sing: 'touch' }" in JS
    assert "{ key: 'pass_att', noun: 'Pass attempts', low: 'pass attempts', sing: 'pass attempt' }" in JS
    # Header + rows render the picked metric's noun, never a fixed "targets".
    assert "Usage by week · ${_metric.low}" in JS
    assert "${_metric.noun}, last ${_last3.length}" in JS
    assert "${_metric.sing}${_hi.v === 1 ? '' : 's'}" in JS


def test_usage_card_falls_back_when_position_metric_missing():
    # An RB series with no touches still renders (on targets/carries)
    # instead of an empty card.
    assert "if (!_mvals.some(v => v != null))" in JS
    assert "for (const m of _metricDefs)" in JS


def test_focus_weekly_mapping_carries_all_metrics():
    assert ("touches: p.touches, pass_att: p.pass_att" in JS
            or "touches: p.touches" in JS)
    assert "carries: p.carries" in JS


def test_role_card_usage_line_by_position():
    assert "<span>Touches / game</span>" in JS
    assert "<span>Pass att / game</span>" in JS
    # RB touches = carries + targets from the game logs themselves.
    assert "(_nv(s.rush_att) || 0) + (_nv(s.rec_tgt) || 0)" in JS


def test_team_strip_has_margin_around_it():
    assert ".pm-team-strip { margin: 14px 18px; }" in CSS


def test_right_column_is_a_stack_not_one_stretched_card():
    # Usage + situational (RB) + week-by-week cards stack on the right;
    # the pair no longer stretches them to the room card's height, which
    # left a large blank area under Usage by Week.
    assert '<div class="pm-team-stack">${usageCard}${_sitCard}${detailCard}</div>' in JS
    pair_rule = CSS[CSS.find(".pm-team-pair { display: grid;"):]
    pair_rule = pair_rule[:pair_rule.find("}")]
    assert "align-items: start" in pair_rule
    assert ".pm-team-stack { display: flex; flex-direction: column; gap: 12px;" in CSS


def test_situational_card_lives_in_stack_and_stays_hidden_without_data():
    assert 'pm-team-pair-card pm-pair-situational" hidden' in JS
    assert 'class="pm-usage-bars-wrap"' in JS
    # The room card no longer carries the situational wrap.
    room_card = JS[JS.find('pm-team-pair-card pm-pair-room'):]
    room_card = room_card[:room_card.find('pm-team-note')]
    assert "pm-usage-bars-wrap" not in room_card
    # The loader unhides the card only when bars actually render.
    assert "container.closest('.pm-pair-situational')" in JS
    assert "card.hidden = false" in JS


def test_week_by_week_detail_card_is_position_shaped():
    assert '<span class="pm-section-label">Week by week</span>' in JS
    assert '`${p.carries} car`' in JS and '`${p.touches} touch`' in JS  # RB line
    assert '`${p.pass_att} att`' in JS and '`${p.carries} rush`' in JS  # QB line
    assert "pm-pair-weekly" in JS


def test_weekly_series_payload_includes_pos_metrics(monkeypatch):
    from dashboard_services import breakout_api

    rows = [
        {"week": 1, "snap_pct": 74.0, "snaps": 55, "team_snaps": 74,
         "targets": 5, "carries": 22, "touches": 27, "pass_att": 0},
        {"week": 2, "snap_pct": 83.1, "snaps": 61, "team_snaps": 73,
         "targets": 8, "carries": 18, "touches": 26, "pass_att": 0},
    ]
    monkeypatch.setattr(
        "data_building.weekly_metrics.get_player_weekly_series",
        lambda pid, season: rows,
    )
    out = breakout_api._weekly_series_for_payload("9221", 2026)
    assert out[0]["touches"] == 27 and out[0]["carries"] == 22
    assert out[1]["touches"] == 26 and out[1]["pass_att"] == 0
    assert out[0]["targets"] == 5  # targets still present for WR/TE cards
