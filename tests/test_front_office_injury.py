"""Tests for the Front Office Report injury cluster.

Covers the deterministic injury section (card + modal), the injury
fingerprint in the cache key inputs, _urgent_needs extensions
(QUESTIONABLE starters, benched serious injuries), injury pills on
trade/waiver targets, return estimates in the roster table, and the
opponent injury line.

Heavy deps are stubbed like tests/test_front_office_report_render.py;
injury_return and utils.utils are stubbed per-test with monkeypatch so the
stubs never leak into other test modules.
"""
import json
import sys
import types

import pytest


_INSTALLED_STUBS = []


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    if name not in sys.modules:
        sys.modules[name] = m
        _INSTALLED_STUBS.append(name)


class _E(Exception):
    pass


_stub("dashboard_services.ai.client", AIRateLimitError=_E, AIUnavailableError=_E)
_stub(
    "dashboard_services.ai.context_builders",
    _ctx_is_sf=lambda *a: False,
    _record_pf_pa_for_roster=lambda *a: ("", 0.0, 0.0),
    build_model_value_lookup=lambda *a: {},
    build_team_gm_context=lambda *a: {},
    build_trade_suggestions_context=lambda *a: {},
    ctx_scoring_type=lambda *a: "ppr",
)
_stub(
    "dashboard_services.ai.prompts",
    build_front_office_prompt_payload=lambda *a: {},
    generate_front_office_report_result=lambda *a: {},
    normalize_trade_scoring_type=lambda x: x,
)
_stub(
    "dashboard_services.ai.renderer",
    _ai_error_notice=lambda r="": "",
    _ctx_with_playoff_odds=lambda x: x,
    _emit_ai_html=lambda x: x,
    ai_available=lambda: False,
)
_stub("dashboard_services.providers.espn_api", safe_float=lambda x, d=0.0: d)
_stub("utils.roster_strength", STARTER_THRESHOLD=0)

from dashboard_services.ai.front_office_report import (  # noqa: E402
    _build_injury_rows,
    _drop_add_pairs,
    _exclude_hurt_waiver_targets,
    _fmt_weeks_out,
    _inj_canonical,
    _inj_is_reportable,
    _inj_severity,
    _injury_card_html,
    _injury_df_map,
    _injury_section_html,
    _opponent_injuries,
    _roster_table_html,
    _trade_targets_html,
    _urgent_needs,
    _waivers_cuts_html,
    render_front_office_card_html,
    render_front_office_report_html,
)

for _stub_name in _INSTALLED_STUBS:
    sys.modules.pop(_stub_name, None)


def _stub_injury_return(monkeypatch, weeks_out_map=None, verdict_map=None):
    """Hermetic injury_return: no ESPN fetch, no disk."""
    weeks_out_map = weeks_out_map or {}
    verdict_map = verdict_map or {}
    mod = types.ModuleType("dashboard_services.injury_return")
    mod.weeks_out_for_player = lambda pid: weeks_out_map.get(str(pid))
    mod.injury_roster_verdict = lambda **kw: verdict_map.get(
        str(kw.get("status") or ""), {"label": "Hold"}
    )
    monkeypatch.setitem(sys.modules, "dashboard_services.injury_return", mod)


def _stub_utils_utils(monkeypatch):
    mod = types.ModuleType("utils.utils")
    mod.path_teams_index = lambda: "/nonexistent"
    mod.read_json_cached = lambda p: {}
    monkeypatch.setitem(sys.modules, "utils.utils", mod)


def test_injury_severity_ordering():
    assert _inj_severity("IR") < _inj_severity("OUT")
    assert _inj_severity("OUT") < _inj_severity("DOUBTFUL")
    assert _inj_severity("DOUBTFUL") < _inj_severity("QUESTIONABLE")
    assert _inj_severity("PUP") == _inj_severity("IR")
    assert _inj_canonical("Q") == "QUESTIONABLE"
    assert _inj_canonical("O") == "OUT"
    assert _inj_canonical("D") == "DOUBTFUL"
    assert _inj_canonical("IR") == "IR"


def test_fmt_weeks_out():
    assert _fmt_weeks_out(None) == ""
    assert _fmt_weeks_out(0.4) == "this week"
    assert _fmt_weeks_out(1.0) == "~1 wk"
    assert _fmt_weeks_out(2.0) == "~2 wks"
    assert _fmt_weeks_out(2.5) == "~2.5 wks"


def test_build_injury_rows_sorts_and_enriches(monkeypatch):
    # Fallback path: no injury_df in ctx, so the players index is scanned.
    _stub_injury_return(
        monkeypatch,
        weeks_out_map={"1": 3.0},
        verdict_map={"IR": {"label": "Move to IR"}, "QUESTIONABLE": {"label": "Stash"}},
    )
    ctx = {"players": {
        "1": {"injury_status": "IR", "injury_body_part": "Knee"},
        "2": {"injury_status": "Q"},
    }}
    rows = [
        {"id": "2", "name": "Jayden Daniels", "position": "QB", "team": "WAS",
         "role": "Starter", "value": 500.0, "injury": "Q"},
        {"id": "1", "name": "Christian McCaffrey", "position": "RB", "team": "SF",
         "role": "Starter", "value": 900.0, "injury": "IR"},
        {"id": "3", "name": "Healthy Guy", "position": "WR", "team": "DAL",
         "role": "Depth", "value": 100.0, "injury": ""},
    ]
    out = _build_injury_rows(ctx, rows)
    assert [r["name"] for r in out] == ["Christian McCaffrey", "Jayden Daniels"]
    ir_row = out[0]
    assert ir_row["injury"] == "IR"
    assert ir_row["body"] == "Knee"
    assert ir_row["return_label"] == "out ~3 wks"
    assert ir_row["action"] == "Move to IR"
    # Single-letter codes are canonicalized on the row.
    assert out[1]["injury"] == "QUESTIONABLE"
    assert out[1]["return_label"] == ""
    assert out[1]["action"] == "Stash"


class _FakeInjuryDF:
    """Duck-typed stand-in for the pandas injury_df (no pandas needed)."""

    def __init__(self, records, empty=False):
        self._records = records
        self.empty = empty

    def to_dict(self, orient):
        assert orient == "records"
        return self._records


def test_injury_df_map_reads_canonical_source():
    df = _FakeInjuryDF([
        {"PlayerID": "1", "Status": "Out", "Injury": "Out", "Body": "Knee"},
        {"PlayerID": "2", "Status": "Active", "Injury": "Questionable", "Body": ""},
    ])
    out = _injury_df_map({"injury_df": df})
    assert out["1"] == {"designation": "Out", "body": "Knee"}
    assert out["2"] == {"designation": "Questionable", "body": ""}
    assert _injury_df_map({}) is None
    assert _injury_df_map({"injury_df": _FakeInjuryDF([], empty=True)}) is None


def test_build_injury_rows_prefers_injury_df(monkeypatch):
    # The df wins over the players index when both are present.
    _stub_injury_return(monkeypatch, weeks_out_map={}, verdict_map={})
    df = _FakeInjuryDF([
        {"PlayerID": "1", "Status": "Out", "Injury": "Out", "Body": "Ankle"},
    ])
    ctx = {
        "injury_df": df,
        "players": {"1": {"injury_status": "Q", "injury_body_part": "Knee"}},
    }
    rows = [
        {"id": "1", "name": "Puka Nacua", "position": "WR", "team": "LAR",
         "role": "Starter", "value": 800.0, "injury": "Q"},
    ]
    out = _build_injury_rows(ctx, rows)
    assert len(out) == 1
    assert out[0]["injury"] == "OUT"
    assert out[0]["body"] == "Ankle"


def test_inj_is_reportable_filters_non_injury_status():
    for desig in ("OUT", "o", "Q", "IR", "DOUBTFUL", "d", "PUP", "SUSP"):
        assert _inj_is_reportable(desig), desig
    # Healthy scratches and other non-injury statuses are not injuries.
    for desig in ("INACTIVE", "ACTIVE", "", "ACT"):
        assert not _inj_is_reportable(desig), desig
    # A body-part note alone is still worth reporting.
    assert _inj_is_reportable("", "Knee")
    assert _inj_is_reportable("Active", "Hamstring")
    assert not _inj_is_reportable("", "")


def test_build_injury_rows_skips_healthy_scratch(monkeypatch):
    _stub_injury_return(monkeypatch)
    ctx = {"players": {"9": {"status": "Inactive"}}}
    rows = [
        {"id": "9", "name": "Healthy Scratch", "position": "WR", "team": "DAL",
         "role": "Depth", "value": 100.0, "injury": "INACTIVE"},
    ]
    assert _build_injury_rows(ctx, rows) == []


def test_urgent_needs_includes_questionable_and_bench(monkeypatch):
    _stub_utils_utils(monkeypatch)
    roster = {"starters": ["1", "2"], "players": ["1", "2", "3"]}
    rows = [
        {"id": "1", "name": "Jayden Daniels", "position": "QB", "team": "WAS", "injury": "Q"},
        {"id": "2", "name": "CeeDee Lamb", "position": "WR", "team": "DAL", "injury": "O"},
        {"id": "3", "name": "Bench Back", "position": "RB", "team": "KC", "injury": "OUT"},
    ]
    urgent = _urgent_needs({}, roster, rows, week=4)
    details = [u["detail"] for u in urgent]
    assert any("game-time call" in d for d in details), details
    assert any("bench" in d for d in details), details
    # Single-letter O still counts as a serious starter injury.
    assert any(d == "CeeDee Lamb is O" for d in details), details


def test_urgent_needs_bench_questionable_excluded(monkeypatch):
    _stub_utils_utils(monkeypatch)
    roster = {"starters": ["1"], "players": ["1", "2"]}
    rows = [
        {"id": "1", "name": "Starter", "position": "QB", "team": "WAS", "injury": ""},
        {"id": "2", "name": "Bench Q", "position": "WR", "team": "DAL", "injury": "Q"},
    ]
    urgent = _urgent_needs({}, roster, rows, week=4)
    assert urgent == []


def test_injury_section_renders_rows():
    rows = [
        {"name": "Christian McCaffrey", "position": "RB", "team": "SF",
         "injury": "IR", "body": "Knee", "return_label": "out ~3 wks",
         "action": "Move to IR"},
    ]
    html_out = _injury_section_html(rows)
    assert "Injury report" in html_out
    assert "Christian McCaffrey" in html_out
    assert "Knee" in html_out
    assert "out ~3 wks" in html_out
    assert "Move to IR" in html_out
    assert "for-inj" in html_out


def test_injury_section_empty_state():
    html_out = _injury_section_html([])
    assert "Injury report" in html_out
    assert "No rostered players" in html_out


def test_card_injury_compact_caps_at_three():
    rows = [
        {"name": f"P{i}", "injury": "Q", "body": "", "return_label": "", "action": ""}
        for i in range(5)
    ]
    html_out = _injury_card_html(rows)
    assert html_out.count("<li>") == 3
    assert "+2 more in the full report" in html_out
    assert _injury_card_html([]) == ""


def test_trade_target_injury_pill():
    targets = [
        {
            "gets": [{"id": "1", "name": "Puka Nacua", "position": "WR",
                      "age": 24, "value": 900.1, "injury": "Q"}],
            "gives": [{"name": "Jaylen Warren", "position": "RB"}],
            "partner": "Veiny Oilers",
            "analyzer_url": "/trade?x=1",
        }
    ]
    html_out = _trade_targets_html(targets, {})
    assert "Puka Nacua</strong> <span class='for-inj'>Q</span>" in html_out


def test_waiver_target_injury_pill():
    data = {
        "waiver_targets": [
            {"id": "9", "name": "WanDale Robinson", "position": "WR",
             "team": "NYG", "pos_rank_label": "", "injury": "O"},
        ],
        "cut_candidates": [],
        "drop_add_pairs": [],
    }
    html_out = _waivers_cuts_html(data, {})
    assert "WanDale Robinson</strong> <span class='for-inj'>O</span>" in html_out


def test_roster_table_return_estimate():
    rows = [
        {"name": "Jayden Daniels", "position": "QB", "team": "WAS", "age": 24,
         "value": 500.0, "pos_rank_label": "QB8", "role": "Starter",
         "trend_7d": None, "injury": "Q", "weeks_out": 2.0},
    ]
    html_out = _roster_table_html(rows)
    assert "Q · ~2 wks" in html_out


def test_opponent_injuries_top3_sorted():
    ctx = {
        "rosters": [{"roster_id": 7, "starters": ["a", "b", "c", "d"]}],
        "players": {
            "a": {"injury_status": "OUT"},
            "b": {"injury_status": "Q"},
            "c": {"injury_status": "DOUBTFUL"},
            "d": {"injury_status": "IR"},
        },
        "players_index": {
            "a": {"full_name": "A Player"},
            "b": {"full_name": "B Player"},
            "c": {"full_name": "C Player"},
            "d": {"full_name": "D Player"},
        },
    }
    out = _opponent_injuries(ctx, {"opponent_roster_id": "7"})
    assert [e["injury"] for e in out["entries"]] == ["IR", "OUT", "DOUBTFUL"]
    assert out["line"] == "D Player (IR), A Player (OUT), C Player (DOUBTFUL)"
    assert _opponent_injuries(ctx, None) == {"entries": [], "line": ""}


def test_opponent_injuries_from_injury_df():
    df = _FakeInjuryDF([
        {"PlayerID": "a", "Status": "Out", "Injury": "Out", "Body": ""},
        {"PlayerID": "b", "Status": "Active", "Injury": "", "Body": ""},
    ])
    ctx = {
        "injury_df": df,
        "rosters": [{"roster_id": 7, "starters": ["a", "b"]}],
        "players_index": {"a": {"full_name": "A Player"}},
    }
    out = _opponent_injuries(ctx, {"opponent_roster_id": "7"})
    assert [e["name"] for e in out["entries"]] == ["A Player"]
    assert out["line"] == "A Player (OUT)"


def test_urgent_needs_ignores_non_injury_status(monkeypatch):
    _stub_utils_utils(monkeypatch)
    roster = {"starters": ["1"], "players": ["1"]}
    rows = [
        {"id": "1", "name": "Healthy Scratch", "position": "WR",
         "team": "DAL", "injury": "INACTIVE"},
    ]
    assert _urgent_needs({}, roster, rows, week=4) == []


def test_report_renders_injury_section_and_opp_line():
    data = {
        "team_name": "Caleb's Casting Couch",
        "week": 4,
        "record": "2-1",
        "playoff_pct": 80.0,
        "grades": [],
        "roster_rows": [],
        "trade_targets": [],
        "waiver_targets": [],
        "cut_candidates": [],
        "drop_add_pairs": [],
        "injury_rows": [
            {"name": "Jayden Daniels", "position": "QB", "team": "WAS",
             "injury": "Q", "body": "Knee", "return_label": "out this week",
             "action": "Hold"},
        ],
        "opponent_injuries": {
            "entries": [{"name": "A Player", "injury": "OUT"}],
            "line": "A Player (OUT)",
        },
    }
    ai = {"verdict": "HOLD", "headline": "Test headline.", "posture": "",
          "gm_alert": "", "trade_notes": {}, "waiver_notes": {}}
    modal = render_front_office_report_html(data, ai)
    assert "Injury report" in modal
    assert "Jayden Daniels" in modal
    assert "Opponent missing" in modal
    assert "A Player (OUT)" in modal
    # Injury Report sits right after the hero, before the headline.
    assert modal.index("Injury report") < modal.index("for-headline")
    card = render_front_office_card_html(data, ai)
    assert "Injuries" in card
    assert "Jayden Daniels" in card


def test_no_em_dashes_in_injury_copy():
    rows = [
        {"name": "Jayden Daniels", "position": "QB", "team": "WAS",
         "injury": "Q", "body": "Knee", "return_label": "out this week",
         "action": "Hold"},
    ]
    for html_out in (
        _injury_section_html(rows),
        _injury_section_html([]),
        _injury_card_html(rows),
    ):
        assert "\u2014" not in html_out


# ---------------------------------------------------------------------------
# Dart hardening: seriously-hurt players can never be waiver "add"
# candidates in redraft. Dynasty keeps them, labeled stash-only.
# ---------------------------------------------------------------------------

def _dart_like_targets():
    return [
        {"id": "dart", "name": "Jaxson Dart", "position": "QB", "team": "NYG",
         "value": 900.0, "injury": "IR"},
        {"id": "healthy", "name": "Healthy Vet", "position": "QB", "team": "DAL",
         "value": 800.0, "injury": ""},
        {"id": "gametime", "name": "GameTime QB", "position": "QB", "team": "BUF",
         "value": 700.0, "injury": "Q"},
    ]


def test_redraft_excludes_ir_waiver_target():
    out = _exclude_hurt_waiver_targets(_dart_like_targets(), "redraft")
    names = [w["name"] for w in out]
    assert "Jaxson Dart" not in names
    # QUESTIONABLE stays: a game-time call is still a playable add.
    assert names == ["Healthy Vet", "GameTime QB"]


def test_redraft_excludes_every_serious_designation():
    from utils.waiver_score import SERIOUS_INJURY_STATUSES
    for status in sorted(SERIOUS_INJURY_STATUSES):
        targets = [{"id": "x", "name": "Hurt Guy", "position": "RB",
                    "team": "KC", "value": 500.0, "injury": status}]
        assert _exclude_hurt_waiver_targets(targets, "redraft") == [], status
    # Single-letter codes canonicalize too.
    for code in ("O", "D"):
        targets = [{"id": "x", "name": "Hurt Guy", "position": "RB",
                    "team": "KC", "value": 500.0, "injury": code}]
        assert _exclude_hurt_waiver_targets(targets, "redraft") == [], code


def test_dynasty_keeps_hurt_labeled_stash_only():
    targets = _dart_like_targets()
    out = _exclude_hurt_waiver_targets(targets, "dynasty")
    assert [w["name"] for w in out] == ["Jaxson Dart", "Healthy Vet", "GameTime QB"]
    dart = next(w for w in out if w["name"] == "Jaxson Dart")
    assert dart["stash_only"] is True
    assert all(not w.get("stash_only") for w in out if w["name"] != "Jaxson Dart")


def test_drop_add_pairs_cannot_add_hurt_player():
    filtered = _exclude_hurt_waiver_targets(_dart_like_targets(), "redraft")
    cuts = [{"id": "watson", "name": "Deshaun Watson", "position": "QB",
             "team": "CLE", "value": 100.0}]
    pairs = _drop_add_pairs(cuts, filtered)
    assert all(p["add"]["name"] != "Jaxson Dart" for p in pairs)
    # The stale top_move pairing ("Drop Deshaun Watson, add Jaxson Dart")
    # is no longer constructible from redraft candidacy.
    assert not any(
        p["drop"]["name"] == "Deshaun Watson" and p["add"]["name"] == "Jaxson Dart"
        for p in pairs
    )


def _load_prompts(monkeypatch):
    """Import the real prompts module with hermetic client/prose stubs."""
    client = types.ModuleType("dashboard_services.ai.client")
    client.clean_ai_text = lambda s: s
    client.get_ai_client = lambda: None
    prose = types.ModuleType("dashboard_services.ai.prose")
    prose.scrub_ai_prose_field_names = lambda d: d
    prose.scrub_ai_result_strings = lambda d: d
    monkeypatch.setitem(sys.modules, "dashboard_services.ai.client", client)
    monkeypatch.setitem(sys.modules, "dashboard_services.ai.prose", prose)
    sys.modules.pop("dashboard_services.ai.prompts", None)
    import dashboard_services.ai.prompts as prompts
    return prompts


def test_prompt_bars_injured_top_move_add(monkeypatch):
    prompts = _load_prompts(monkeypatch)
    system = prompts.build_front_office_report_prompt({}, "redraft")
    assert "It must NEVER recommend" in system
    assert "serious injury designation" in system
    assert "If every candidate at the needed position is" in system


def test_prompt_labels_dynasty_stash_only(monkeypatch):
    prompts = _load_prompts(monkeypatch)
    system = prompts.build_front_office_report_prompt({}, "dynasty")
    assert 'label it "IR stash' in system


def test_dart_scenario_absent_from_redraft_payload(monkeypatch):
    prompts = _load_prompts(monkeypatch)
    filtered = _exclude_hurt_waiver_targets(_dart_like_targets(), "redraft")
    cuts = [{"id": "watson", "name": "Deshaun Watson", "position": "QB",
             "team": "CLE", "value": 100.0}]
    payload = prompts.build_front_office_prompt_payload({
        "scoring_type": "redraft",
        "waiver_targets": filtered,
        "drop_add_pairs": _drop_add_pairs(cuts, filtered),
    })
    blob = json.dumps(payload)
    assert "Jaxson Dart" not in blob
