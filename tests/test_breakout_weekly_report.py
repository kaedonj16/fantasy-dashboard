"""Wednesday breakout report: weekly grading + calibration proposal email.

Design rules pinned here:
- "This week" is the grades recorded in the 7 days before as_of (grade
  rows carry graded_at); a row without a readable graded_at is never
  new, but still counts toward the cumulative record.
- The cumulative tables are calibration.summarize_bands over the
  current version's rows, byte for byte, so the email can never
  disagree with the sidebar or the calibrate CLI.
- A conversion is suggested only when the sample guards clear and the
  fitted curve's 50% crossing sits 5+ points from EMERGING_MIN_SCORE;
  inversions and invalid confidence are Investigate lines, never
  conversions. Below the guards the plan says so, with counts.
- Zero rows renders a short honest email, and no fixture's subject or
  body ever contains an em dash.
"""
from __future__ import annotations

import importlib.util
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from data_building.breakout_engine import calibration as cal
from data_building.breakout_engine import weekly_report as wr

AS_OF = datetime(2026, 10, 7, 13, 0)  # a Wednesday, the cron's slot
RECENT = AS_OF - timedelta(days=2)
OLD = AS_OF - timedelta(days=20)
EM_DASH = "—"


def _row(grade="hit", score=50.0, conf=50.0, classification="watchlist",
         version="weekly-v6", graded_at=RECENT, **extra):
    row = {
        "player_id": "p1", "player_name": "Player One", "season": 2026,
        "as_of_week": 2, "grade": grade, "breakout_score": score,
        "confidence": conf, "classification": classification,
        "scoring_version": version, "graded_at": graded_at,
        "ppg_delta": None, "baseline_ppg": None, "outcome_ppg": None,
        "opp_delta": None, "snap_delta": None,
    }
    row.update(extra)
    return row


def _rows(grade, n, **kw):
    return [_row(grade, **kw) for _ in range(n)]


def _render(rows, **kw):
    report = wr.build_weekly_report(rows, season=2026, as_of=AS_OF, **kw)
    return report, wr.render_email(report)


# ---------------------------------------------------------------------------
# newly-graded window
# ---------------------------------------------------------------------------

def test_newly_graded_window_counts_only_the_last_7_days():
    rows = (
        _rows("hit", 3) + _rows("miss", 2) + _rows("partial", 1)
        + _rows("hit", 4, graded_at=OLD)
        + [_row("hit", graded_at=None)]
        + [_row("partial", graded_at=AS_OF - timedelta(days=7))]
        + [_row("miss", graded_at=AS_OF - timedelta(days=7, hours=1))]
    )
    report, _ = _render(rows)
    bucket = report["newly_graded"]["bucket"]
    # 3 hits + 2 misses + 1 partial from two days ago, plus the partial
    # at exactly 7 days (window edge is inclusive). The miss one hour
    # past the window, the old rows, and the row with no graded_at are
    # not new.
    assert bucket["graded"] == 7
    assert (bucket["hit"], bucket["partial"], bucket["miss"]) == (3, 2, 2)
    assert report["nothing_new_line"] is None
    # The not-new rows still count toward the season record.
    assert report["cumulative"]["overall"]["graded"] == 13


def test_graded_at_accepts_strings_and_dates():
    rows = [
        _row("hit", graded_at="2026-10-06T09:30:00"),
        _row("miss", graded_at=date(2026, 10, 5)),
        _row("hit", graded_at="2026-09-01T09:30:00"),
    ]
    report, _ = _render(rows)
    assert report["newly_graded"]["bucket"]["graded"] == 2


def test_newly_graded_hit_rate_renders():
    rows = (_rows("hit", 6, ppg_delta=3.0) + _rows("partial", 3)
            + _rows("miss", 3))
    _, (_, body) = _render(rows)
    assert "12 calls graded: 6 hits, 3 partials, 3 misses" in body
    assert "hit rate 50.0%" in body


def test_nothing_new_this_week_is_one_honest_line():
    rows = _rows("hit", 12, graded_at=OLD)
    report, (subject, body) = _render(rows, latest_week=6)
    assert subject == "Breakout report: 0 calls graded this week"
    assert "Nothing new graded in the last 7 days." in body
    assert "most recent grades were recorded on 2026-09-17" in body
    assert "Stored weekly data currently reaches week 6." in body


# ---------------------------------------------------------------------------
# notable hits / misses
# ---------------------------------------------------------------------------

def test_notable_hits_and_misses_capped_at_five_and_sorted():
    hits = [_row("hit", player_name=f"Hit {i}", ppg_delta=float(i),
                 baseline_ppg=5.0, outcome_ppg=5.0 + i)
            for i in range(1, 8)]
    misses = [_row("miss", player_name=f"Miss {i}", ppg_delta=-float(i))
              for i in range(1, 8)]
    report, (subject, body) = _render(hits + misses)
    assert subject == "Breakout report: 14 calls graded this week"
    top_hits = report["newly_graded"]["notable_hits"]
    top_misses = report["newly_graded"]["notable_misses"]
    assert [e["ppg_delta"] for e in top_hits] == [7.0, 6.0, 5.0, 4.0, 3.0]
    assert [e["ppg_delta"] for e in top_misses] == [-7.0, -6.0, -5.0,
                                                     -4.0, -3.0]
    assert "Hit 7" in body and "Hit 2" not in body
    assert "PPG delta +7.0 (5 to 12)" in body


# ---------------------------------------------------------------------------
# cumulative record matches calibration exactly
# ---------------------------------------------------------------------------

def test_cumulative_tables_match_calibration_summarize_bands():
    current = (_rows("hit", 12, score=55.0, conf=80.0)
               + _rows("miss", 11, score=25.0, conf=30.0))
    old_version = _rows("hit", 7, version="weekly-v5")
    report, _ = _render(current + old_version)
    expected = cal.summarize_bands(current)
    assert report["cumulative"]["overall"] == expected["overall"]
    assert report["cumulative"]["score_bands"] == expected["score_bands"]
    assert (report["cumulative"]["confidence_bands"]
            == expected["confidence_bands"])
    assert (report["cumulative"]["by_classification"]
            == expected["by_classification"])
    assert report["total_graded_all_versions"] == 30


# ---------------------------------------------------------------------------
# suggested changes
# ---------------------------------------------------------------------------

def _conversion_rows():
    # Curve: 0 at scores 30 and 45, 1.0 at 55, so the fitted 50%
    # crossing lands on 55, thirteen points above EMERGING_MIN_SCORE.
    return (_rows("miss", 100, score=30.0, conf=30.0)
            + _rows("miss", 50, score=45.0, conf=30.0)
            + _rows("hit", 100, score=55.0, conf=80.0))


def test_suggested_conversion_names_constants_and_sample():
    report, (_, body) = _render(_conversion_rows())
    plan = report["suggested_changes"]
    assert len(plan["conversions"]) == 1
    assert plan["investigations"] == []
    line = plan["conversions"][0]
    assert line.startswith("Convert: EMERGING_MIN_SCORE 42 -> 55")
    assert "SCORING_VERSION weekly-v6 -> weekly-v7" in line
    assert "based on 250 graded weekly-v6 calls" in line
    assert line in body


def _inversion_rows():
    # The watchlist band is perfect and the emerging band all misses:
    # bands invert. Pooled, the fitted curve tops out at 0.4, so there
    # is no 50% crossing and nothing to convert.
    return (_rows("hit", 100, score=35.0)
            + _rows("miss", 150, score=50.0))


def test_inversion_produces_investigate_not_convert():
    report, (_, body) = _render(_inversion_rows())
    plan = report["suggested_changes"]
    assert plan["conversions"] == []
    assert plan["investigations"]
    assert any("not monotonic" in line for line in plan["investigations"])
    assert all(line.startswith("Investigate: ")
               for line in plan["investigations"])
    assert "Convert:" not in body
    assert "Investigate: " in body


def test_below_guard_produces_no_changes_with_counts():
    report, (_, body) = _render(_rows("hit", 50, score=55.0))
    plan = report["suggested_changes"]
    assert plan["conversions"] == [] and plan["investigations"] == []
    assert "No changes suggested this week." in body
    assert "Insufficient data: 50 graded calls under weekly-v6" in body
    assert "(need 200)" in body


def _clean_rows():
    # Monotone bands, valid confidence, and the fitted crossing at 38,
    # within 5 points of the current threshold: nothing to change.
    return (_rows("miss", 70, score=25.0, conf=30.0)
            + _rows("partial", 70, score=38.0, conf=50.0)
            + _rows("hit", 70, score=55.0, conf=80.0)
            + _rows("hit", 40, score=65.0, conf=80.0))


def test_clean_monotone_produces_no_changes_with_reason():
    report, (_, body) = _render(_clean_rows())
    plan = report["suggested_changes"]
    assert plan["conversions"] == [] and plan["investigations"] == []
    assert "No changes suggested this week." in body
    assert "crossing sits at 38" in body
    assert "within 5 points of EMERGING_MIN_SCORE 42" in body


# ---------------------------------------------------------------------------
# zero data and copy hygiene
# ---------------------------------------------------------------------------

def test_zero_rows_render_a_short_honest_email():
    report, (subject, body) = _render([], latest_week=3)
    assert subject == "Breakout report: no calls graded yet"
    assert "Nothing has been graded yet this season" in body
    assert "Stored weekly data currently reaches week 3." in body
    assert "No graded calls yet under weekly-v6." in body
    assert "No changes suggested this week." in body
    assert "Hand this email to Muse" in body
    assert len(body.splitlines()) <= 20


@pytest.mark.parametrize("rows", [
    _conversion_rows(), _inversion_rows(), _clean_rows(),
    _rows("hit", 50, score=55.0), [],
    _rows("hit", 6, ppg_delta=3.0) + _rows("miss", 6),
])
def test_no_em_dash_anywhere_in_the_email(rows):
    _, (subject, body) = _render(rows)
    assert EM_DASH not in subject
    assert EM_DASH not in body


# ---------------------------------------------------------------------------
# send script (module functions + monkeypatched loader, no live DB)
# ---------------------------------------------------------------------------

_SEND_PATH = (Path(__file__).resolve().parent.parent / "scripts"
              / "send_weekly_breakout_report.py")


def _load_send_module():
    spec = importlib.util.spec_from_file_location(
        "send_weekly_breakout_report", _SEND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_send_script_dry_run_prints_and_never_sends(monkeypatch, capsys):
    module = _load_send_module()
    # The script reports against the live clock, so its fixture rows are
    # stamped relative to now, not the report tests' fixed Wednesday.
    fresh = datetime.now() - timedelta(days=1)
    rows = (_rows("hit", 6, ppg_delta=4.0, graded_at=fresh)
            + _rows("miss", 6, graded_at=fresh))
    monkeypatch.setattr(module, "load_graded_calls", lambda season: rows)
    monkeypatch.setattr(module, "_latest_week", lambda season: 5)

    def _boom(*a, **k):
        raise AssertionError("deliver must not run on a dry run")
    monkeypatch.setattr(module, "deliver", _boom)
    monkeypatch.setenv("BREAKOUT_REPORT_EMAIL", "owner@example.com")
    assert module.main(["--season", "2026", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Graded this week" in out
    assert "12 calls graded" in out
    assert "[dry-run] report printed only" in out


def test_send_script_without_email_config_prints_only(monkeypatch, capsys):
    import utils.email_delivery as email_delivery

    module = _load_send_module()
    monkeypatch.setattr(module, "load_graded_calls", lambda season: [])
    monkeypatch.setattr(module, "_latest_week", lambda season: None)
    monkeypatch.setenv("BREAKOUT_REPORT_EMAIL", "owner@example.com")
    monkeypatch.setattr(email_delivery, "is_configured", lambda: False)
    assert module.main(["--season", "2026"]) == 0
    out = capsys.readouterr().out
    assert "no calls graded yet" in out
    assert "email not configured, report printed only" in out


def test_send_script_without_recipient_prints_only(monkeypatch, capsys):
    module = _load_send_module()
    monkeypatch.setattr(module, "load_graded_calls", lambda season: [])
    monkeypatch.setattr(module, "_latest_week", lambda season: None)
    for key in ("BREAKOUT_REPORT_EMAIL", "EMAIL_USER",
                "BREVO_SENDER_EMAIL"):
        monkeypatch.delenv(key, raising=False)
    assert module.main(["--season", "2026"]) == 0
    out = capsys.readouterr().out
    assert "no recipient" in out


# ---------------------------------------------------------------------------
# send script loader fail-soft: the Wednesday cron must never crash when
# the database is unreachable; it degrades to the honest zero-data report.
# ---------------------------------------------------------------------------


def test_load_graded_calls_returns_empty_when_db_unreachable(monkeypatch):
    module = _load_send_module()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert module.load_graded_calls(2026) == []


def test_dry_run_without_database_prints_zero_data_report(monkeypatch, capsys):
    module = _load_send_module()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(module, "_latest_week", lambda season: None)

    def _boom(*a, **k):
        raise AssertionError("deliver must not run on a dry run")
    monkeypatch.setattr(module, "deliver", _boom)
    assert module.main(["--season", "2026", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "no calls graded yet" in out
    assert "[dry-run] report printed only" in out
