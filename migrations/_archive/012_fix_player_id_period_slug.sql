-- Migration 012: Remove duplicate prospect records caused by period-in-name slugging.
--
-- Bug: _slug("K.C. Concepcion") → "K_C_CONCEPCION" but _slug("KC Concepcion") → "KC_CONCEPCION",
-- producing two rows for the same player.  The code fix (removing periods before slugifying)
-- makes KC_CONCEPCION the canonical form going forward.
--
-- Tables with ON DELETE CASCADE (rookie_prospect_source_data, rookie_prospect_athleticism,
-- rookie_mock_draft_entries, rookie_mock_draft_consensus) are cleaned up automatically.
-- rookie_rankings and rookie_value_history have no FK constraint so are deleted first.

DELETE FROM rookie_rankings      WHERE player_id = 'ROOKIE_2026_K_C_CONCEPCION';
DELETE FROM rookie_value_history WHERE player_id = 'ROOKIE_2026_K_C_CONCEPCION';
DELETE FROM rookie_prospects     WHERE player_id = 'ROOKIE_2026_K_C_CONCEPCION';
