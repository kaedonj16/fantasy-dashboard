# BR Fantasy (fantasy-dashboard)

Dynasty and redraft fantasy football analytics: live values, trade tools, draft room,
breakout detection, playoff odds, and multi-platform league connect
(Sleeper, ESPN, Yahoo, MFL, Fleaflicker).

## Product features

See **[`FEATURES.md`](./FEATURES.md)** for the full, navigation-organized feature list
(what is free vs PRO, platform coverage, and current UI placement). Keep that file in
sync when product behavior changes — `tests/test_product_honesty.py` guards key claims.

## Quick start

### Prerequisites

- Python 3.9+
- PostgreSQL
- API keys as needed (OpenAI for AI features, Stripe for billing, etc.)

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create DB and set DATABASE_URL, then:
python3 -m data_building.breakout_engine.setup_database
```

Copy `.env.example` to `.env` when present and fill secrets. See `docs/weekly-email.md`
for Brevo / digest env vars (`BREVO_API_KEY`, etc.).

### Run

```bash
python app.py                 # development
gunicorn app:app              # production-style
```

App defaults to `http://localhost:5000`.

## Project layout

```
app.py                 # Flask app + remaining monolith routes
routes/                # Blueprints (auth, billing, draft, trade, SEO, …)
dashboard_services/    # Page builders, APIs, AI, providers
data_building/         # Breakout engine, values, simulations, crawlers
static/                # CSS/JS (app.js splits into public + app-features at startup)
extension/             # Browser Draft Assistant (ESPN/Yahoo relay + overlays)
docs/                  # Deeper design and ops notes
tests/                 # pytest (includes product-honesty contracts)
FEATURES.md            # User-facing feature inventory
```

## Breakout engine (ops)

```bash
python3 -m data_building.breakout_engine.calculate_breakouts_with_real_data --season 2026
python3 -m data_building.breakout_engine.display_results --summary --min-score 40
```

Daily scheduling and API details live under `data_building/breakout_engine/`.

## Deployment

Render is the primary host (`render.yaml`). Configure secrets in the dashboard;
do not commit keys.

## Contributing

1. Branch from `main`
2. Add or update tests for behavior you change (especially product-honesty / PRO gating)
3. Update `FEATURES.md` when user-visible behavior or placement changes
4. Open a PR

## License

MIT — see repository license details if present.
