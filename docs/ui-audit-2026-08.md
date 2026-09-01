# UI/UX Audit — August 2026

Local walkthrough using the **UI audit fixture** (`UI_AUDIT=1`, league id `ui-audit`).

## How to run

```bash
bash .cursor/install.sh && bash .cursor/start.sh
export UI_AUDIT=1 FLASK_SECRET_KEY=dev DATABASE_URL=postgresql://brfantasy:brfantasy@127.0.0.1:5432/brfantasy
source .venv/bin/activate && python app.py
```

Open **http://localhost:5000/ui-audit** → **Bootstrap signed-in session** → follow the link catalog.

Automated smoke coverage: `pytest tests/test_ui_audit_fixture.py` (48 routes, all 200).

## Mock data

| Field | Value |
|-------|-------|
| League | UI Audit Dynasty (`ui-audit`) |
| Platform / season | Sleeper / 2026 |
| State | Week 11, in-season, 10 teams |
| Draft | Completed ~120 days ago |
| HTTP | Sleeper `fetch_json` patched; no live API |

## Findings

### Critical (fixed in this branch)

| Issue | Page | Fix |
|-------|------|-----|
| Weekly Hub showed “draft has not completed” | `/weekly` | Mock ctx now includes `latest_draft` + past `draft_day` |

### Medium (open / partially addressed)

| Issue | Page | Notes |
|-------|------|-------|
| Dashboard tab bar clips last tab on narrow widths | Dashboard | Consider scroll hint or shorter labels |
| Player names truncate in trending strip | Waivers | Ellipsis + tooltip or wider chip |
| Graph scatter labels overlap | Graphs (`?tour=1`) | Tour mock only; smart label offset for dense clusters |
| Tour modal “Remind me later” touch targets small | Dashboard tour | **Fixed:** 44px min-height on skip/later buttons |
| Player card detail truncated (“QB • NO • Q…”) | Dashboard | Review card min-width / line-clamp |
| Empty-state cards leave large vertical gap | Weekly, Activity | Tighten min-height on `.card.central` mobile |

### Minor (polish backlog)

- BUBBLE badge contrast (standings)
- 13px base font at ≤480px — consider 14px
- Inconsistent waiver status label weights
- “This league” chip alignment in header
- Trade calculator header spacing
- Footer link touch targets on mobile
- Pricing feature grid density on small screens

### Expected / not bugs

- **CSP console noise** in local dev (Sentry, GA, AdSense blocked) — production headers differ
- **`/` and `/trade` redirect** when session is bootstrapped — intentional signed-in routing
- **`/scout` and `/optimal`** redirect to Weekly tabs — hub links point to `weekly?tab=…`
- **Graphs `?tour=1`** uses 6-team tour mock, not the 10-team audit league — by design for offline graphs preview

## Positive patterns

- Mobile bottom dock consistent across league pages
- No horizontal overflow at 390px
- Pro gates clearly marked
- Card layout and hierarchy read well in dark mode
- Empty states use clear copy (when data preconditions are met)
