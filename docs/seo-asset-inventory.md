# SEO Asset Inventory (R14 / site audit #14)

Logged-out public shells vs signed-in full bundles. Measured on
`cursor/site-improvements-af1e` with the Flask test client (files on disk;
minified when the boot-time minify step has produced them).

## Bundle sizes (on disk)

| Asset | Raw | Served (typical) |
|-------|-----|------------------|
| `seo_lite.css` | ~44 KB | ~44 KB (not minified separately) |
| `dashboard.css` | ~789 KB | `dashboard.min.css` ~515 KB |
| `public.js` | ~570 KB | `public.min.js` ~382 KB |
| `app.js` | ~865 KB | `app.min.js` ~594 KB |
| `app-features.js` | (built at boot) | lazy on lite pages |

## `render_page` logic (`app.py`)

```text
_use_lite     = lite_js and not signed_in and features bundle built
_use_lite_css = _use_lite and active != "home"
CSS           = seo_lite.css if _use_lite_css else dashboard(.min).css
JS            = public(.min).js if _use_lite else app(.min).js
```

Landing (`active == "home"`) opts into `lite_js` for a slim JS paint but **keeps
`dashboard.css`** — hero / onboarding / feature grid / ticker are not in
`seo_lite.css` (guarded by `tests/test_seo_lite_js.py` +
`tests/test_signed_in_home.py`).

Static URLs use content hashes (`?v=<hash>`). Versioned `/static/*` responses
get `Cache-Control: public, max-age=31536000, immutable`.

## Route matrix (logged-out guest)

| Route | `lite_js` | CSS pack | Primary JS | Notes |
|-------|-----------|----------|------------|-------|
| `/` (landing) | Yes | **dashboard.min.css** (~515 KB) | public.min.js | Intentional CSS exception |
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
| `/` landing | dashboard.min.css **515,335 B** | public.min.js **381,991 B** |
| `/pricing` **before** | dashboard.min.css **515,335 B** | app.min.js **593,926 B** |
| `/pricing` **after** | seo_lite.css **44,405 B** | public.min.js **381,991 B** |

### Weight savings shipped (`/pricing` guest)

| | Before | After | Delta |
|--|--------|-------|-------|
| CSS | 515,335 B | 44,405 B | **−470,930 B (~459 KB)** |
| JS | 593,926 B | 381,991 B | **−211,935 B (~207 KB)** |
| **Total first-paint assets** | ~1,109 KB | ~426 KB | **≈ −683 KB** |

Pricing markup is `.card` / checkout chrome already covered by `seo_lite.css`;
no feature removal. League-scoped `/…/pricing` gets the same guest path;
signed-in sessions still ignore `lite_js`.

## Option A (landing → lite CSS) — deferred this pass

Landing still loads full `dashboard.min.css` (~515 KB). A focused extract is
not low-risk in one pass:

- `FORM_BODY` uses **100+** classes (`home-*`, platform connect flows, Google
  buttons, ticker, feature grid, loading overlay, etc.).
- Almost none of those selectors exist in `seo_lite.css` (only shared tokens /
  shell / a generic `.hint`).
- Home-specific slices in `dashboard.css` alone are ~35–40 KB, but they depend
  on many shared form/button/platform rules still outside that slice.
- R14.3 already tried applying `seo_lite.css` to `active=="home"` and unstyled
  the page; regression tests explicitly forbid that naive swap.

Safer follow-up: generate `landing_lite.css` = `seo_lite.css` + extracted home
rules, prove visual parity on logged-out `/`, then flip
`_use_lite_css` for home. Prefer that over extending `seo_lite.css` itself so
SEO shells stay lean.

## Cache / hashing (option C) — verified OK

- `render_page` links CSS/JS with `?v=_static_hash(...)`.
- No SEO HTML hardcodes bare `/static/app.js` or `/static/dashboard.css`
  without a version query for the main shell assets.
- Service worker no longer precaches unversioned heavy shells (site audit #10).

## Tests

- `tests/test_seo_lite_js.py` — guest SEO paths → `public.js`, no full `app.js`;
  `seo_lite.css` on `/compare`; landing keeps dashboard CSS; `/pricing` lite.
- `tests/test_signed_in_home.py` — `_use_lite_css` excludes homepage.
- `tests/test_product_honesty.py` — source-level `lite_js` opts-in.
