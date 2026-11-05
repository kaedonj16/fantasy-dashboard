-- GM brief live data (waiver derivation): mark league_rules waiver
-- fields that were derived from the league's own processed-claim
-- history (utils/league_rules.derive_waiver_from_activity).
--
-- The marker is what lets consumers tell "the waiver process time was
-- derived from consistent claim timestamps" apart from a record that
-- merely carries derived playoff values (whose waiver fields stay
-- untrusted). utils/league_rules.py also adds the column idempotently
-- at table-ensure time, so this file and the code path are both safe
-- to re-run.
ALTER TABLE league_rules
    ADD COLUMN IF NOT EXISTS waiver_process_derived BOOLEAN NOT NULL DEFAULT FALSE;
