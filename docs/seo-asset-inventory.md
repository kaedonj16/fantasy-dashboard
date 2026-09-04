# SEO Asset Inventory (R14 / site audit #14)

Logged-out public shells vs signed-in full bundles. Measured on
`cursor/audit-backlog-all-af1e` with the Flask test client (files on disk;
minified when the boot-time minify step has produced them).

## Bundle sizes (on disk)

| Asset | Raw | Served (typical) |
|-------|-----|------------------|
| `seo_lite.css` | ~44 KB | ~44 KB (not minified separately) |
| `landing_lite.css` | ~83 KB | ~83 KB (seo_lite + home extract) |
| `dashboard.css` | ~789 KB | `dashboard.min.css` ~515 KB |
| `public.js` | ~570 KB | `public.min.js` ~382 KB |
| `app.js` | ~865 KB | `app.min.js` ~594 KB |
| `app-features.js` | (built at boot) | lazy on lite pages |

## `render_page` logic (`app.py`)

```text
_use_lite = lite_js and not signed_in and features bundle built
CSS       = landing_lite.css  if _use_lite and active == "home"
          = seo_lite.css      if _use_lite and active != "home"
          = dashboard(.min).css otherwise
JS        = public(.min).js if _use_lite else app(.min).js
```

Guest landing (`active == "home"`) opts into `lite_js` **and** `landing_lite.css`
(R14 Option A) — hero / onboarding / feature grid / ticker extracted from
`dashboard.css` onto the seo_lite shell. Naive `seo_lite.css` alone is still
forbidden for home (guarded by `tests/test_seo_lite_js.py` +
`tests/test_signed_in_home.py`). Signed-in home keeps full `dashboard.css`.

Static URLs use content hashes (`?v=<hash>`). Versioned `/static/*` responses
get `Cache-Control: public, max-age=31536000, immutable`.

## Route matrix (logged-out guest)

| Route | `lite_js` | CSS pack | Primary JS | Notes |
|-------|-----------|----------|------------|-------|
| `/` (landing) | Yes | **landing_lite.css** (~83 KB) | public.min.js | Was dashboard.min (~515 KB) |
| `/rankings/dynasty` (+ pos) | Yes | seo_lite.css (~44 KB) | public.min.js | + `rankings.js` defer |
| `/compare` | Yes | seo_lite.css | public.min.js | |
| `/players` | Yes | seo_lite.css | public.min.js | |
| `/player/<slug>` (+ trade-value) | Yes | seo_lite.css | public.min.js | |
| `/breakouts` | Yes | seo_lite.css | public.min.js | |
| `/prospects` | Yes | seo_lite.css | public.min.js | |
| `/dynasty-trade-value-chart` | Yes | seo_lite.css | public.min.js | via `page_players` |
| `/top-movers` | Yes | seo_lite.css | public.min.js | |
| `/guides`, `/glossary`, legal | Yes | seo_lite.css | public.min.js | `public_bp._render` |
| `/pricing` | **Yes (R14)** | seo_lite.css | public.min.js | Was full dashboard + app |
| `/…/pricing` (league) | **Yes (R14)** | seo_lite when guest | public when guest | Signed-in keeps full |

Signed-in on any of the above: full `app.min.js` + `dashboard.min.css`
regardless of `lite_js=True`.

## Measured linked CSS/JS (guest HTML)

| Route | CSS linked | JS linked |
|-------|------------|-----------|
| `/compare` | seo_lite.css **44,405 B** | public.min.js **381,991 B** |
| `/rankings/dynasty` | seo_lite.css **44,405 B** | public.min.js **381,991 B** |
| `/` landing **before** | dashboard.min.css **515,335 B** | public.min.js **381,991 B** |
| `/` landing **after** | landing_lite.css **~82,820 B** | public.min.js **381,991 B** |
| `/pricing` **before** | dashboard.min.css **515,335 B** | app.min.js **593,926 B** |
| `/pricing` **after** | seo_lite.css **44,405 B** | public.min.js **381,991 B** |

### Weight savings shipped (`/` guest)

| | Before | After | Delta |
|--|--------|-------|-------|
| CSS | 515,335 B | ~82,820 B | **≈ −432 KB** |
| JS | (already public.min.js) | same | — |
| **CSS first-paint** | ~515 KB | ~83 KB | **≈ −84%** |

### Weight savings shipped (`/pricing` guest)

| | Before | After | Delta |
|--|--------|-------|-------|
| CSS | 515,335 B | 44,405 B | **−470,930 B (~459 KB)** |
| JS | 593,926 B | 381,991 B | **−211,935 B (~207 KB)** |
| **Total first-paint assets** | ~1,109 KB | ~426 KB | **≈ −683 KB** |

Pricing markup is `.card` / checkout chrome already covered by `seo_lite.css`;
no feature removal. League-scoped `/…/pricing` gets the same guest path;
signed-in sessions still ignore `lite_js`.

## Option A (landing → lite CSS) — shipped

`static/landing_lite.css` = full `seo_lite.css` + selective extract from
`dashboard.css` (`home-*`, ticker, signed-home, espn/mfl/flea-home methods,
google-continue, platform-btn/selector, fullscreen-loading / flo-*, provider
choice, referenced `@keyframes`).

- Guest `/` primary stylesheet is `landing_lite.css` (not `seo_lite.css` alone,
  not full `dashboard.css`).
- Signed-in `/` still uses `dashboard(.min).css`.
- Other SEO shells still use `seo_lite.css`.

## Cache / hashing (option C) — verified OK

- `render_page` links CSS/JS with `?v=_static_hash(...)` (`_LANDING_LITE_CSS_V`
  for the landing pack).
- No SEO HTML hardcodes bare `/static/app.js` or `/static/dashboard.css`
  without a version query for the main shell assets.
- Service worker no longer precaches unversioned heavy shells (site audit #10).

## Tests

- `tests/test_seo_lite_js.py` — guest SEO paths → `public.js`, no full `app.js`;
  `seo_lite.css` on `/compare`; guest `/` → `landing_lite.css`; signed-in `/`
  → dashboard CSS; `/pricing` lite.
- `tests/test_signed_in_home.py` — landing_lite wiring + seo_lite has no home-*.
- `tests/test_product_honesty.py` — source-level `lite_js` opts-in.
