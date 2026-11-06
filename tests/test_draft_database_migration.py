"""Migration sanity for 056_draft_database.sql (same read-the-file
pattern as test_monthly_plans / test_account_lifecycle_contract)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SQL = (ROOT / "migrations" / "056_draft_database.sql").read_text(encoding="utf-8")


def test_migration_is_idempotent():
    assert SQL.count("ADD COLUMN IF NOT EXISTS") >= 8
    assert "CREATE INDEX IF NOT EXISTS" in SQL


def test_migration_columns_present():
    for col in ("league_name", "scoring_type", "league_format",
                "slot_to_roster_id", "draft_order_type", "roster_names",
                "roster_owners", "grade_summary", "grades_computed_at"):
        assert col in SQL


def test_migration_targets_draft_adp_drafts():
    assert "ALTER TABLE draft_adp_drafts" in SQL
