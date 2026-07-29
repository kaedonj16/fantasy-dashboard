#!/usr/bin/env bash
#
# Trigger a full model-value rebuild on the deployed app. This runs cron_daily's
# value step (build_daily_model_values -> trade_value_model), which is what
# applies the SF top-5 re-anchor. Returns immediately; the rebuild runs in a
# background thread on the server (watch the server logs).
#
# Requires the server's CRON_SECRET. The base URL comes from $APP_URL or arg 1.
#
# Usage:
#   CRON_SECRET=xxx ./scripts/rebuild_values.sh https://your-domain.com
#   # or, if APP_URL is exported:
#   CRON_SECRET=xxx APP_URL=https://your-domain.com ./scripts/rebuild_values.sh
#
set -euo pipefail

URL="${1:-${APP_URL:-}}"

if [[ -z "${CRON_SECRET:-}" ]]; then
  echo "error: set CRON_SECRET (the server's CRON_SECRET env value)" >&2
  exit 1
fi
if [[ -z "$URL" ]]; then
  echo "usage: CRON_SECRET=... $0 <base-url>   (or export APP_URL)" >&2
  exit 1
fi
URL="${URL%/}"

echo "Triggering full value rebuild at ${URL}/api/run-daily-cron (force=true)..."
curl -fsS -X POST "${URL}/api/run-daily-cron" \
  -H 'Content-Type: application/json' \
  -d "{\"secret\":\"${CRON_SECRET}\",\"force\":true}"
echo
echo
echo "Started. It runs in the background on the server."
echo "Watch the server logs for:  [calibration] Normalizing ...  and  [run-daily-cron] Completed"
echo "When it finishes, the SF board's top-5 average should read ~999.9."
