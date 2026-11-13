"""Focused tests for the 2026-11-13 site-improvement fixes.

Covers the /faq answers that contradicted the live product (platforms
served and how you connect; personal-only PRO plans) and the dynasty
rankings hub subtitle (full-date freshness stamp and a plain statement
of the sort format the page actually uses). The ROS overall forward
floor lives in tests/test_ros_rankings.py with the other ROS page tests.
"""
from datetime import datetime

from routes.public_bp import FAQ_SECTIONS


def _answers():
    return {q: a for _section, items in FAQ_SECTIONS for q, a in items}


def _sample_table():
    return [
        {"name": "Bijan Robinson", "position": "RB", "team": "ATL",
         "age": 24, "value": 1091, "sf_value": 1150},
        {"name": "Jaxon Smith-Njigba", "position": "WR", "team": "SEA",
         "age": 24, "value": 1054, "sf_value": 1180},
        {"name": "Jahmyr Gibbs", "position": "RB", "team": "DET",
         "age": 24, "value": 1050, "sf_value": 1120},
    ]


# ── FAQ accuracy ────────────────────────────────────────────────────────────

def test_faq_connect_answer_names_every_platform():
    answer = _answers()["What do I need to use it?"]
    for platform in ("Sleeper", "ESPN", "Yahoo", "MFL", "Fleaflicker"):
        assert platform in answer
    assert "Sleeper or ESPN league ID" not in answer


def test_faq_intro_is_not_sleeper_only():
    answer = _answers()["What is the BR Fantasy Dashboard?"]
    assert "Sleeper league data" not in answer


def test_faq_premium_answer_is_personal_only():
    answer = _answers()["Is there a premium or ad-free mode?"]
    assert "Every plan is personal" in answer
    assert "not your league mates" in answer
    assert "league and personal subscriptions" not in answer


def test_faq_rewritten_answers_have_no_em_dash():
    answers = _answers()
    for q in ("What is the BR Fantasy Dashboard?",
              "What do I need to use it?",
              "Is there a premium or ad-free mode?"):
        assert "\u2014" not in answers[q]


# ── Dynasty hub subtitle: full date + stated sort format ────────────────────

def test_hub_subtitle_defaults_to_full_date_and_superflex_order():
    from dashboard_services.pages.dynasty_pages import build_rankings_hub_body

    body = build_rankings_hub_body(_sample_table())
    today = datetime.now().strftime("%B %d, %Y")
    assert f"updated {today}" in body
    assert "Ranked in Superflex order." in body
    # SF sort: the higher SF value leads even with the lower 1QB value.
    assert body.index("Jaxon Smith-Njigba") < body.index("Bijan Robinson")


def test_hub_subtitle_reflects_1qb_toggle():
    from dashboard_services.pages.dynasty_pages import build_rankings_hub_body

    body = build_rankings_hub_body(
        _sample_table(), as_of_date="November 13, 2026", format="1qb")
    assert "updated November 13, 2026" in body
    assert "Ranked in 1QB order." in body
    assert "Ranked in Superflex order." not in body
    # 1QB sort: the higher 1QB value leads.
    assert body.index("Bijan Robinson") < body.index("Jaxon Smith-Njigba")


def test_hub_position_subtitle_states_format_and_full_date():
    from dashboard_services.pages.dynasty_pages import build_rankings_hub_body

    body = build_rankings_hub_body(
        _sample_table(), position="RB",
        as_of_date="November 13, 2026", format="sf")
    assert "updated November 13, 2026" in body
    assert "Ranked in Superflex order." in body
