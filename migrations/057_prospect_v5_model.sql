-- Prospect model v5: the trained model artifact store.
--
-- v5 (data_building/rookie_pipeline/prospect_model_v5.py) is a
-- learned model, not a table of hand-set weights: the trained
-- per-position gradient-boosted models, the recorded outcome
-- ECDFs they grade against, and the training fingerprint live in
-- one pickled artifact written by scripts/train_prospect_model_v5.py
-- and read by the pipeline when PROSPECT_MODEL=v5 is selected.
--
-- Single-row table (id pinned to 1), mirroring
-- prospect_v4_reference_artifact: artifact is the pickled bytes,
-- fingerprint is the training provenance (population row count and
-- hash, class list, ordered feature lists, sklearn version, random
-- seed) duplicated as JSONB so the stored model can be identified
-- without unpickling it.
--
-- Grading fails closed when no row exists: the board stays on the
-- incumbent model until a trained artifact is stored and v5 is
-- explicitly selected and shipped.

CREATE TABLE IF NOT EXISTS prospect_v5_model_artifact (
    id int PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    artifact bytea NOT NULL,
    fingerprint jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);
