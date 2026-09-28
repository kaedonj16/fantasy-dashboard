# Coolify / Hetzner deployment for brfantasyfootball.com
#
# Matches the Render web service as closely as possible:
#   - Python 3.11.9 (same as render.yaml PYTHON_VERSION)
#   - pip install -r requirements.txt (same build command, minus --no-cache-dir
#     which was a Render-specific workaround for a corrupt cached wheel)
#   - python startup.py (same start command)
#   - Playwright Chromium installed (Render currently skips the browser
#     download while diagnosing perf issues; Coolify gets the full install
#     so og_render.py can render real OG images again)
#
# Coolify notes:
#   - Set the app's PORT env var to 5000 (or leave unset; the app defaults
#     to 5000 when PORT is missing).
#   - Paste the Cloudflare Origin Certificate + key in the app's SSL settings
#     and keep the domain proxied (orange cloud) with SSL mode Full (strict).
#   - Copy every env var from Render, replacing DATABASE_URL / REDIS_URL with
#     the Coolify-managed services. APP_URL must be exactly
#     https://www.brfantasyfootball.com (no trailing whitespace).
#   - IMPORTANT: restore the Postgres dump into the Coolify database BEFORE
#     the web container first boots, otherwise the app runs a slow first-time
#     init against an empty database.
#   - render.yaml is intentionally left untouched; CI asserts against it.

FROM python:3.11.9-slim

# Playwright needs these OS libs for Chromium. Installed first so the layer
# is cached unless the base image changes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps before copying the full repo so this layer is cached
# unless requirements.txt changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Playwright browser binary (the pip package is in requirements.txt; this
# downloads Chromium itself).
RUN python -m playwright install chromium

# Copy the app. .dockerignore keeps caches, local DBs, and venvs out.
COPY . ./

EXPOSE 5000

# Same entrypoint as Render's startCommand.
CMD ["python", "startup.py"]
