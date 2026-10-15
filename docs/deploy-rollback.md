# Deploy rollback runbook

Use this when a deploy to production is bad enough that fixing forward
will take longer than returning to the last known good deploy. A rollback
restores code only. It does not undo database migrations or data writes
(see the warning list at the bottom).

## Before you roll back

1. Note the bad deploy: its commit SHA, deploy time, and what broke.
2. Note the rollback target: the last deploy that was live and healthy
   before the bad one. Keep its full commit SHA. Verification below
   compares against this SHA.
3. Check whether any background job or backfill is running (for example
   in the Render shell or the daily job service). A redeploy or rollback
   restarts processes and can kill a long running job mid-run. If one is
   running, decide explicitly: wait for it, or stop it and rerun it from
   the start after the rollback.

## Roll back on Render

1. Open the Render dashboard: https://dashboard.render.com
2. Open the BR Fantasy web service (the service that serves
   www.brfantasyfootball.com).
3. Open the Events tab (deploy history for the service).
4. Find the rollback target deploy from the step above: the last live,
   successful deploy before the bad deploy.
5. Open that deploy's actions menu (the three dot menu on the deploy
   entry) and choose "Rollback to this deploy".
6. Confirm the rollback. Render redeploys that exact prior build. No
   rebuild from source and no new commit are involved.
7. Watch the Events tab and the Logs tab until the rollback deploy is
   live and the service has finished starting.
8. If the daily job or cron services shipped the same bad change, repeat
   the same steps for each affected service so web and jobs run the same
   commit.

After the rollback is verified, also fix the repo: open a revert PR (or a
fix forward PR) for the bad change on main. Otherwise the next deploy
from main ships the bad code again.

## Post-rollback verification

1. Check the version endpoint:

   ```bash
   curl -s https://www.brfantasyfootball.com/healthz/version
   ```

   The response is JSON with bundle hashes and a `git_sha` field. The
   `git_sha` value must match the rollback target commit SHA you noted
   before rolling back. If it still shows the bad deploy's SHA, the
   rollback has not taken over yet (or the wrong service was rolled
   back). Wait for the rollback deploy to go live and check again.

2. Check the homepage loads:

   ```bash
   curl -s -o /dev/null -w "%{http_code}\n" https://www.brfantasyfootball.com/
   ```

   Expect 200.

3. Smoke test in a browser, signed in: dashboard loads, one league page
   loads, and the feature touched by the bad deploy behaves as it did on
   the rollback target.

4. Check the service Logs tab for new startup errors, and watch for a
   few minutes after the rollback before calling it done.

## Service worker cache version

Any deploy or rollback that changes shipped JS or CSS must ship a new
service worker cache version, or phones keep painting the old cached
shell:

1. In `static/sw.js`, bump `CACHE_NAME` by one version (for example
   `br-fantasy-v35` to `br-fantasy-v36`) in the same PR as the JS or CSS
   change.
2. Update the tests that pin the current version:
   `tests/test_pwa_blank_screen.py` and `tests/test_site_audit_2026.py`.
3. A rollback is no exception. If the rollback target predates a cache
   version bump, the rolled back code carries its own older `CACHE_NAME`.
   Confirm which cache version the live service worker reports after the
   rollback, and if users could be stuck on a newer poisoned cache, fix
   forward with a fresh bump instead of relying on the rollback alone.

## Warning: what a code rollback does not undo

Migrations run forward only. Rolling back the code does not reverse
schema changes, dropped objects, or data written by the newer code. If
the bad deploy included any of the following, the database change stays
in place after the rollback, and the rolled back code must be able to
live with it (or a separate, manually written data fix is needed):

- `migrations/043_redzone_plays_pid_key.sql`: rebuilds `redzone_plays`
  with a new primary key. Inside the rebuild it runs
  `DROP TABLE redzone_plays;` and renames the replacement table into
  place. Rows are carried over, but the old table object is gone and the
  new key shape stays after a code rollback.
- No migration in `migrations/` currently runs a `DROP COLUMN`. This was
  checked by grepping `migrations/` for `DROP TABLE` and `DROP COLUMN`
  when this runbook was written. Re-run that grep before relying on it:

  ```bash
  grep -rn "DROP TABLE\|DROP COLUMN" migrations/
  ```

- Other forward-only drops exist as constraint, index, and trigger
  drops (for example in migrations 001, 002, 005, 023, 025, 031, 032,
  036, and 046). A dropped constraint or index is not restored by a code
  rollback.
- Data backfills and one-off scripts that ran on the newer code are not
  undone. If one wrote bad rows, the fix is a targeted data correction,
  not the rollback.
