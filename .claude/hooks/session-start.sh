#!/bin/bash
# SessionStart hook: install dependencies so tests + linters work in
# Claude Code on the web sessions. Idempotent and non-interactive.
# Note: NOT using `set -e` — a single failed transitive wheel must not abort
# session startup; the pure-logic test suite only needs pytest/pyflakes.
set -uo pipefail

# Web sessions only — local machines already have their own setup.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

# Test + lint tooling first (fast, essential, always needed).
pip install -q pytest pyflakes || true

# Full app dependencies (cached after first run). Non-fatal: if a transitive
# package fails to build in this environment, the session still starts and the
# dependency-light tests still run.
pip install -q -r requirements.txt || \
  echo "[session-start] warning: some app deps failed to install; pure-logic tests still run" >&2

# Let tests import the repo packages (data_building, utils, ...) from the root.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo 'export PYTHONPATH="."' >> "$CLAUDE_ENV_FILE"
fi

exit 0
