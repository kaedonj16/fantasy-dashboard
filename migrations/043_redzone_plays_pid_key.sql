-- 043_redzone_plays_pid_key.sql
--
-- Re-key redzone_plays per player: PRIMARY KEY (season, game_id, play_id, pid).
--
-- The extractors emit one row per involved player under the same play_id
-- (a TD pass = a QB row + a receiver row; an interception = a QB row + a
-- defender row), but the 041 key (season, game_id, play_id) kept only one
-- player's row per play. Readers asking for the other player's plays --
-- ScoreZone Moments asking for the QB's plays, above all -- silently
-- missed them.
--
-- Idempotent: the rebuild runs only when the live primary key is exactly
-- the pre-pid key. Rows are preserved; each surviving row's pid comes from
-- its payload (the old key kept one row per play, so extracted pids cannot
-- collide). utils/scorezone_store._migrate_plays_pid_key applies the same
-- rebuild as a safety net for environments where this file hasn't run.

DO $$
DECLARE
    pk_cols text;
BEGIN
    SELECT string_agg(a.attname, ',' ORDER BY array_position(c.conkey, a.attnum))
      INTO pk_cols
      FROM pg_constraint c
      JOIN pg_class t ON t.oid = c.conrelid
      JOIN pg_namespace n ON n.oid = t.relnamespace
      JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY (c.conkey)
     WHERE t.relname = 'redzone_plays' AND n.nspname = 'public' AND c.contype = 'p'
     GROUP BY c.oid;

    IF pk_cols = 'season,game_id,play_id' THEN
        CREATE TABLE redzone_plays_pidmig (
            season      INTEGER NOT NULL,
            game_id     TEXT NOT NULL,
            play_id     TEXT NOT NULL,
            pid         TEXT NOT NULL DEFAULT '',
            seq         INTEGER NOT NULL DEFAULT 0,
            is_td       BOOLEAN NOT NULL DEFAULT FALSE,
            week        INTEGER,
            payload     JSONB NOT NULL,
            observed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (season, game_id, play_id, pid)
        );
        INSERT INTO redzone_plays_pidmig
            (season, game_id, play_id, pid, seq, is_td, week, payload, observed_at)
        SELECT season, game_id, play_id, COALESCE(payload->>'pid', ''),
               seq, is_td, week, payload, observed_at
          FROM redzone_plays;
        DROP TABLE redzone_plays;
        ALTER TABLE redzone_plays_pidmig RENAME TO redzone_plays;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_redzone_plays_game_seq
    ON redzone_plays (season, game_id, seq);
CREATE INDEX IF NOT EXISTS idx_redzone_plays_td_seen
    ON redzone_plays (season, observed_at) WHERE is_td;
CREATE INDEX IF NOT EXISTS idx_redzone_plays_observed
    ON redzone_plays (observed_at);
CREATE INDEX IF NOT EXISTS idx_redzone_plays_week
    ON redzone_plays (season, week);
