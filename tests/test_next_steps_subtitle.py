"""Next steps card subtitle: bye fork only (Kaedon 2026-11-11).

The card header used to stack the GM brief's race line (seed, record,
PF cushion on the first team out, still-to-play - all already in the
hero / matchup card) over the full Playoff Path line (magic number,
root-for, three bye percentages), with "Most important first." as the
filler fallback. The subtitle is now ONE line: bye odds with a win
this week vs a loss, from the brief's structured ``playoff_path.bye``
fork. No live fork means no subtitle at all. The rules confirmation
prompt (an action, not filler) still takes the slot alone when the
playoff format is unverified.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RACE_TEXT = ("2nd seed · 7-2 · +288.28 PF on Veiny Oilers "
             "(tiebreak yours) · Pittsburgh Pilots still to play")
PATH_TEXT = ("Magic number 1 · Root for Veiny: your odds go "
             "70.2% vs 41.0% · Bye 59.8%: 68.4% with a win, "
             "50.8% with a loss")


ACTION = {"kind": "lineup", "tag": "Lineup",
          "action": "Left 9.3 on the bench", "why": "Projects more.",
          "cta_label": "Lineup efficiency", "cta_url": "#",
          "score": 90, "priority": "high"}


def _render(monkeypatch, brief):
    pytest.importorskip("pandas")
    pytest.importorskip("flask")
    import app as appmod
    import dashboard_services.gm_brief as gb

    action = dict(ACTION)
    monkeypatch.setattr(appmod, "_next_steps_lineup_actions",
                        lambda c, r: [action])
    monkeypatch.setattr(appmod, "_next_steps_waiver_actions",
                        lambda c, r, m: [])
    monkeypatch.setattr(appmod, "_next_steps_trade_actions",
                        lambda c, r: [])
    monkeypatch.setattr(appmod, "_next_steps_roster_actions",
                        lambda c, r: [])
    monkeypatch.setattr(appmod, "_next_steps_bench_check_actions",
                        lambda c, r, w: [])
    monkeypatch.setattr(appmod, "_next_steps_bench_check_note",
                        lambda c, r, w: "")
    monkeypatch.setattr(gb, "build_gm_brief", lambda *a, **k: brief)
    with appmod.app.test_request_context("/"):
        return appmod._render_next_steps_queue(
            {}, "1", [], season=2026, current_week=10)


def _brief(**over):
    brief = {
        # The render takes its action list from the brief when one
        # exists, so the canned brief carries the same action.
        "actions": [dict(ACTION)],
        "race_line": {"text": RACE_TEXT},
        "race_prompt": None,
        "playoff_path": {
            "text": PATH_TEXT,
            "bye": {"pct": 59.8, "if_win": 68.4, "if_lose": 50.8},
        },
    }
    brief.update(over)
    return brief


def test_subtitle_is_the_bye_fork_only(monkeypatch):
    html_out = _render(monkeypatch, _brief())
    assert ("Win this week and your bye odds jump to 68.4%. "
            "Lose and they drop to 50.8%.") in html_out
    # The race line and the rest of the path line are gone, and the
    # base bye rate no longer stacks a third percentage on the line.
    assert RACE_TEXT not in html_out
    assert "Magic number" not in html_out
    assert "Root for Veiny" not in html_out
    assert "59.8" not in html_out
    assert "Most important first." not in html_out
    # Exactly one subtitle line renders.
    assert html_out.count("os-section-subtitle") == 1


def test_no_bye_fork_no_subtitle(monkeypatch):
    html_out = _render(monkeypatch, _brief(
        playoff_path={"text": PATH_TEXT, "bye": None}))
    assert "os-section-subtitle" not in html_out
    assert RACE_TEXT not in html_out


def test_level_bye_fork_no_subtitle(monkeypatch):
    html_out = _render(monkeypatch, _brief(playoff_path={
        "text": PATH_TEXT,
        "bye": {"pct": 100.0, "if_win": 100.0, "if_lose": 100.0}}))
    assert "os-section-subtitle" not in html_out


def test_no_brief_no_subtitle(monkeypatch):
    html_out = _render(monkeypatch, None)
    assert "os-section-subtitle" not in html_out
    assert "Most important first." not in html_out


def test_rules_prompt_still_takes_the_slot(monkeypatch):
    prompt = {"source": "default_unverified",
              "text": "Playoff format not confirmed for this league. "
                      "Confirm it on the Standings page."}
    html_out = _render(monkeypatch, _brief(race_prompt=prompt))
    assert prompt["text"] in html_out
    assert "bye odds jump" not in html_out
    assert html_out.count("os-section-subtitle") == 1


@pytest.mark.parametrize("source", ["feed", "derived"])
def test_rules_prompt_does_not_mask_fork_for_real_fields(monkeypatch,
                                                         source):
    """Regression (2026-11-11): the brief carries the confirmation
    prompt for EVERY non-user_confirmed source, feed and derived
    included. Those leagues have a real playoff field; the old
    render masked their prompt behind the race line, so it was
    never seen. The cleanup must not surface it either - only a
    default_unverified league gets the prompt instead of the fork.
    """
    prompt = {"source": source,
              "text": "Playoff format not confirmed for this league. "
                      "Confirm it on the Standings page."}
    html_out = _render(monkeypatch, _brief(race_prompt=prompt))
    assert prompt["text"] not in html_out
    assert "bye odds jump to 68.4%" in html_out
    assert html_out.count("os-section-subtitle") == 1


def test_old_subtitle_copy_is_gone_from_app_source():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "Most important first." not in src
