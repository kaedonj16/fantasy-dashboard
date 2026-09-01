"""Brevo weekly digest setup, dry-run, and delivery notes.

The app still owns recipients, fantasy analysis, HTML, unsubscribe, and
weekly dedupe. Brevo delivers the generated HTML and reports bounce/open/click
events. Do not recreate digest logic as Brevo template variables.

## 1. Create and verify a sender in Brevo

1. Create a Brevo account and add a sender (or verify your sending domain).
2. Recommended sender: `noreply@<your domain>` with display name `BR Fantasy`.
3. Complete Brevo's sender/domain verification before production sends.

## 2. SPF / DKIM

Follow Brevo's current domain-authentication instructions (SPF + DKIM, and DMARC
if you publish a policy). Until the domain is authenticated, messages are more
likely to land in spam.

## 3. API key

In Brevo: SMTP & API → API keys → generate a key with transactional email
permission. Store it only in Render (or your secrets manager). Never commit it.

## 4. Render environment variables

Set these on the **web service** (`brfantasy`). The weekly Render cron only POSTs
`/api/cron/notifications`; the web process performs the send.

| Variable | Required | Notes |
|---|---|---|
| `BREVO_API_KEY` | yes (production) | Transactional API key |
| `BREVO_SENDER_EMAIL` | recommended | Verified sender. Falls back to `EMAIL_USER` then `noreply@PRIMARY_DOMAIN` |
| `BREVO_SENDER_NAME` | optional | Default `BR Fantasy` |
| `BREVO_REPLY_TO_EMAIL` | optional | Falls back to `CONTACT_EMAIL` / `EMAIL_USER` |
| `BREVO_WEBHOOK_SECRET` | yes if using webhooks | Shared secret; see below |
| `SITE_BASE_URL` | recommended | Used for dashboard and unsubscribe links |
| `FLASK_SECRET_KEY` | yes | Signs unsubscribe tokens |
| `SMTP_SERVER` / `EMAIL_USER` / `EMAIL_PASSWORD` | fallback only | Used only when `BREVO_API_KEY` is unset |

SMTP remains a temporary fallback. With `BREVO_API_KEY` set, SMTP is not used
for the weekly digest.

## 5. Webhook (optional)

Weekly send does **not** require webhooks. To record delivered/opened/clicked
and suppress hard bounces:

1. Set `BREVO_WEBHOOK_SECRET` to a long random token.
2. In Brevo, create a **transactional** webhook pointing at:

   `https://<your-domain>/webhooks/brevo/email?secret=<BREVO_WEBHOOK_SECRET>`

   Or send the same token as `Authorization: Bearer <token>` / header
   `X-Brevo-Webhook-Secret` (configure that auth on the Brevo webhook).

3. Subscribe to at least: `delivered`, `opened`, `click`, `hardBounce`,
   `softBounce`, `blocked`, `spam`, `unsubscribed`.

Brevo does not sign webhook bodies with HMAC. If `BREVO_WEBHOOK_SECRET` is
unset, the endpoint returns 503 and ignores events — it is not a public
unauthenticated collector.

Hard bounce, blocked, spam, invalid, and unsubscribed events suppress that
address for future weekly sends. Soft bounces are logged and are **not**
permanent suppressions.

Postgres `account_notification_preferences` remains the source of truth for
subscription. Brevo contacts are not.

## 6. Dry-run / preview

Content generation only; never calls Brevo:

```bash
python -m utils.weekly_email --dry-run --limit 5
python -m utils.weekly_email --dry-run --account-id 123 --out /tmp/digest.html
```

Preview one league without an account row:

```bash
python -m utils.weekly_email \
  --preview-platform sleeper --preview-league LEAGUE_ID \
  --preview-season 2026 --preview-roster 1 \
  --preview-name Sam --out /tmp/digest.html
```

`scripts/preview_weekly_digest.py` is the same CLI.

## 7. Send a test digest to one account

Does **not** fan out to everyone. Respects weekly dedupe unless `--force`:

```bash
python -m utils.weekly_email --account-id 123
python -m utils.weekly_email --account-id 123 --force
```

Production weekly cron already de-dupes per ISO week, so a one-account test
with `--force` can re-send that user for the current week without mailing the
rest of the list.

## 8. Weekly deduplication

After Brevo (or SMTP fallback) **accepts** a message, the app writes
`app_state["weekly_email_sent:<account_id>"] = "<ISO-week>"` (e.g. `2026-W36`).
A retry the same ISO week skips that account (`skipped_already_sent`). Failed
or rate-limited sends are **not** marked complete, so the next run retries them.

Opt-outs live in `account_notification_preferences` (`weekly_digest`, channel
`email`). If no row exists, `accounts.email_opt_out` is the legacy fallback.
The signed `/email/unsubscribe?token=` link disables **weekly_digest only**.

## Cron

Render cron `weekly-email` runs `python scripts/trigger_notifications.py weekly`,
which POSTs the web app. Render evaluates cron expressions in **UTC**. The
schedule is `0 13 * * 2` (Tuesday 13:00 UTC ≈ 9am Eastern during EDT / 8am
during EST). The `TZ=America/New_York` env var does not change Render's cron
clock.

## League format

The digest classifies each league (dynasty / redraft / keeper, 1QB / superflex,
TEP) with `utils.league_format.classify_league_roster_format` and reorders
shared sections. Redraft leads with matchup and start/sit; dynasty leads with
meaningful roster value movement. Empty sections are omitted.
