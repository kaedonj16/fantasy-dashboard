"""
Rookie prospect evaluation pipeline for dynasty fantasy football.

Modules:
    ingestion            - fetch/normalize college stats and bio data
    mock_draft_consensus - aggregate mock drafts into projected pick ranges
    prospect_model       - multi-factor scoring model (position-aware)
    value_translation    - map prospect scores to dynasty dollar values
    pipeline             - orchestration: ingest → score → save
"""
