-- Autopilot Phase 1 (ESPN lineups): per-league settings, account kill
-- switch, queued plans, and the append-only audit log.
--
-- Identity is the user_leagues identity (account_id, platform, league_id,
-- season), the same key the rest of the account system uses. The runtime
-- also creates these tables idempotently from
-- dashboard_services/autopilot_store.py (init_autopilot_tables), following
-- the accounts.py / start_score_bundle pattern; this file is the canonical
-- DDL record, mirroring migration 032's shape.

CREATE TABLE IF NOT EXISTS autopilot_settings (
    id                  SERIAL PRIMARY KEY,
    account_id          INTEGER NOT NULL,
    platform            TEXT NOT NULL,
    league_id           TEXT NOT NULL,
    season              INTEGER NOT NULL,
    mode                TEXT NOT NULL DEFAULT 'off',
    lineups_enabled     BOOLEAN NOT NULL DEFAULT TRUE,
    problem_slots_only  BOOLEAN NOT NULL DEFAULT FALSE,
    notify_push         BOOLEAN NOT NULL DEFAULT TRUE,
    notify_email        BOOLEAN NOT NULL DEFAULT TRUE,
    protected_player_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    paused              BOOLEAN NOT NULL DEFAULT FALSE,
    -- Phase 1.5 columns, stored now so claims need no second migration.
    -- The Phase 1 API refuses to set them (see autopilot_store).
    waivers_enabled     BOOLEAN NOT NULL DEFAULT FALSE,
    faab_cap_per_claim  INTEGER,
    faab_cap_per_week   INTEGER,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT valid_autopilot_mode CHECK (mode IN ('off', 'approve', 'auto')),
    UNIQUE (account_id, platform, league_id, season)
);

CREATE INDEX IF NOT EXISTS idx_autopilot_settings_account
    ON autopilot_settings(account_id);

CREATE TABLE IF NOT EXISTS autopilot_account_state (
    account_id  INTEGER PRIMARY KEY,
    kill_switch BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS autopilot_plans (
    id              SERIAL PRIMARY KEY,
    account_id      INTEGER NOT NULL,
    platform        TEXT NOT NULL,
    league_id       TEXT NOT NULL,
    season          INTEGER NOT NULL,
    team_id         INTEGER NOT NULL,
    scoring_period  INTEGER NOT NULL,
    mode            TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'open',
    snapshot        JSONB NOT NULL DEFAULT '{}'::jsonb,
    snapshot_hash   TEXT NOT NULL,
    swaps           JSONB NOT NULL DEFAULT '[]'::jsonb,
    lineup_items    JSONB NOT NULL DEFAULT '[]'::jsonb,
    issues          JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT valid_autopilot_plan_status CHECK (
        status IN ('open', 'executed', 'partial', 'failed', 'superseded', 'already_applied')
    )
);

CREATE INDEX IF NOT EXISTS idx_autopilot_plans_league
    ON autopilot_plans(account_id, platform, league_id, season, status);

-- Append-only (guardrail 10): no update or delete path exists in code.
CREATE TABLE IF NOT EXISTS autopilot_audit (
    id              SERIAL PRIMARY KEY,
    account_id      INTEGER NOT NULL,
    platform        TEXT NOT NULL,
    league_id       TEXT NOT NULL,
    season          INTEGER NOT NULL,
    plan_id         INTEGER,
    scoring_period  INTEGER,
    action          TEXT NOT NULL,
    mode            TEXT,
    player_id       INTEGER,
    player_name     TEXT,
    from_slot       INTEGER,
    to_slot         INTEGER,
    score_delta     DOUBLE PRECISION,
    detail          TEXT,
    verification    TEXT,
    espn_status     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT valid_autopilot_audit_action CHECK (
        action IN ('planned', 'approved', 'executed', 'skipped', 'failed')
    )
);

CREATE INDEX IF NOT EXISTS idx_autopilot_audit_league
    ON autopilot_audit(account_id, platform, league_id, season, created_at DESC);

COMMENT ON TABLE autopilot_settings IS
    'Per-league Autopilot controls (mode, guardrails, protected players)';
COMMENT ON TABLE autopilot_audit IS
    'Append-only Autopilot event log: planned/approved/executed/skipped/failed';
