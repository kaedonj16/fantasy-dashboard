-- Migration 004: Create roster_alerts table for lineup issue notifications
-- This table stores active roster alerts for each team (injuries, byes, value mismatches, etc.)

CREATE TABLE IF NOT EXISTS roster_alerts (
    league_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    roster_id INTEGER NOT NULL,
    alert_type TEXT NOT NULL,
    player_id TEXT,
    player_name TEXT,
    severity TEXT NOT NULL, -- 'critical', 'warning', 'info'
    message TEXT NOT NULL,
    dismissed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    alert_id SERIAL PRIMARY KEY,

    UNIQUE (league_id, season, week, roster_id, alert_type, player_id)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_roster_alerts_roster ON roster_alerts(roster_id, week);
CREATE INDEX IF NOT EXISTS idx_roster_alerts_severity ON roster_alerts(severity, dismissed);
CREATE INDEX IF NOT EXISTS idx_roster_alerts_type ON roster_alerts(alert_type);

-- Comments
COMMENT ON TABLE roster_alerts IS 'Active roster alerts for lineup issues (injuries, byes, value mismatches)';
COMMENT ON COLUMN roster_alerts.alert_type IS 'Type of alert: empty_slot, injured_starter, bye_week, value_mismatch, questionable_starter';
COMMENT ON COLUMN roster_alerts.severity IS 'Alert severity: critical (must fix), warning (should consider), info (FYI)';
COMMENT ON COLUMN roster_alerts.dismissed IS 'Whether user has dismissed this alert';
