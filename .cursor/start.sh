#!/usr/bin/env bash
#
# Cloud Agent start: per-boot service startup + idempotent schema reconciliation.
#
# Brings up the local PostgreSQL cluster (its process does not survive a reboot),
# ensures the dev role/database exist, and applies the SQL migrations. Every step
# is idempotent and the script returns once the database is ready; the Flask dev
# server itself runs in the "flask" terminal.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:$PATH"
export DATABASE_URL="postgresql://brfantasy:brfantasy@127.0.0.1:5432/brfantasy"

# --- Start PostgreSQL (idempotent: start is a no-op if already online) ---
PG_VERSION="$(pg_lsclusters -h | awk 'NR==1{print $1}')"
sudo pg_ctlcluster "$PG_VERSION" main start 2>/dev/null || true

echo "start.sh: waiting for PostgreSQL to accept connections..."
for _ in $(seq 1 30); do
  if sudo -u postgres pg_isready -q; then break; fi
  sleep 1
done
sudo -u postgres pg_isready

# --- Ensure dev role + database exist ---
if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='brfantasy'" | grep -q 1; then
  sudo -u postgres psql -c "CREATE ROLE brfantasy LOGIN PASSWORD 'brfantasy' CREATEDB SUPERUSER;"
fi
if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='brfantasy'" | grep -q 1; then
  sudo -u postgres createdb -O brfantasy brfantasy
fi

# --- Reconcile schema (idempotent) ---
# The accounts/* tables are created at runtime by the app and are required by
# migrations 021+, so initialize them before running the migration files.
.venv/bin/python -c "from dashboard_services.accounts import init_accounts_tables; init_accounts_tables()"
.venv/bin/python scripts/run_migrations.py

echo "start.sh: PostgreSQL ready and schema up to date"
