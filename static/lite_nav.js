/* Guest lite-shell soft navigation (PJAX-style), phase 1.
 *
 * Served ONLY on logged-out lite_js shells (render_page stamps a
 * <meta name="br-shell" content="lite"> marker and this script tag when
 * the lite bundle is in use; signed-in pages never see either). It makes
 * moving between guest pages swap only #page-root instead of reloading
 * the whole document: no re-download of CSS/JS, no chrome repaint, no
 * blank flash. The rankings position nav (All/QB/RB/WR/TE across the
 * dynasty and rest-of-season families) is the primary flow.
 *
 * Deliberately conservative. A navigation is soft ONLY when every check
 * passes, and any failure falls back to a normal full load:
 *   - same-origin internal link, plain left click, no target/download;
 *   - the fetch returns 200 HTML;
 *   - the fetched document carries the lite-shell marker (so the target
 *     is also a lite guest page, not a signed-in or full-bundle page);
 *   - the fetched #page-root exists and contains no <script> beyond
 *     JSON-LD blocks (a page whose body carries scripts that must
 *     execute is always full-loaded, never half-initialised);
 *   - ad eligibility matches the current page (ad containers live
 *     outside #page-root; a mismatch needs the server's fresh slots).
 * Scripts in swapped content are NEVER executed here.
 *
 * SEO/meta: document.title, meta description and the canonical link
 * are swapped from the fetched document when present. Open Graph tags
 * are not swapped (they describe the URL a crawler fetches directly,
 * which is unaffected by client-side navigation).
 */
(function initLiteNav() {
  if (!window.history || !window.fetch || !window.DOMParser) return;
  var ROOT_ID = 'page-root';
  var token = 0;

  function sameOrigin(u) {
    try { return new URL(u, location.href).origin === location.origin; } catch (e) { return false; }
  }
  function isLiteDoc(doc) {
    return !!doc.querySelector('meta[name="br-shell"][content="lite"]');
  }
  function setScrollTop() {
    try { window.scrollTo(0, 0); } catch (_) {}
    document.documentElement.scrollTop = 0;
    if (document.body) document.body.scrollTop = 0;
  }
  function swapMeta(doc, selector, attr) {
    var next = doc.querySelector(selector);
    var cur = document.querySelector(selector);
    if (next && cur) {
      cur.setAttribute(attr, next.getAttribute(attr) || '');
    } else if (next && !cur) {
      var el = document.createElement(next.tagName.toLowerCase());
      for (var i = 0; i < next.attributes.length; i++) {
        el.setAttribute(next.attributes[i].name, next.attributes[i].value);
      }
      document.head.appendChild(el);
    } else if (!next && cur && selector.indexOf('canonical') !== -1) {
      cur.remove();
    }
  }

  function softNav(url, isPop) {
    var mine = ++token;
    var curRoot = document.getElementById(ROOT_ID);
    if (!curRoot) { location.href = url; return; }
    fetch(url, { headers: { 'X-Lite-Nav': '1' }, credentials: 'same-origin' })
      .then(function (resp) {
        var ct = resp.headers.get('content-type') || '';
        if (!resp.ok || ct.indexOf('text/html') === -1) throw new Error('bad response');
        if (resp.redirected) {
          try {
            if (new URL(resp.url).pathname !== new URL(url, location.href).pathname) {
              location.href = resp.url;
              throw new Error('redirected');
            }
          } catch (e) { if (e.message !== 'redirected') location.href = url; throw e; }
        }
        return resp.text();
      })
      .then(function (html) {
        if (mine !== token) return;   // a newer navigation superseded this one
        var doc = new DOMParser().parseFromString(html, 'text/html');
        if (!isLiteDoc(doc)) throw new Error('target is not a lite shell');
        var newRoot = doc.getElementById(ROOT_ID);
        if (!newRoot) throw new Error('no page-root');
        // Conservative script gate: JSON-LD is data, anything else must
        // execute to be correct, so those pages keep full navigation.
        var scripts = newRoot.querySelectorAll('script');
        for (var i = 0; i < scripts.length; i++) {
          var type = (scripts[i].getAttribute('type') || '').toLowerCase();
          if (type !== 'application/ld+json') throw new Error('page-root script');
        }
        var incomingAds = newRoot.dataset.adEligible === 'true';
        var currentAds = curRoot.dataset.adEligible === 'true';
        if (incomingAds !== currentAds) throw new Error('ad eligibility changed');

        curRoot.innerHTML = newRoot.innerHTML;
        if (newRoot.dataset.adEligible != null) curRoot.dataset.adEligible = newRoot.dataset.adEligible;
        if (newRoot.dataset.premium != null) curRoot.dataset.premium = newRoot.dataset.premium;
        if (doc.title) document.title = doc.title;
        swapMeta(doc, 'meta[name="description"]', 'content');
        swapMeta(doc, 'link[rel="canonical"]', 'href');
        if (!isPop) history.pushState({ brLiteNav: 1 }, '', url);
        setScrollTop();
        try { curRoot.focus({ preventScroll: true }); } catch (e) {}
      })
      .catch(function () {
        if (mine === token) location.href = url;   // safe fallback: full navigation
      });
  }

  document.addEventListener('click', function (e) {
    if (e.defaultPrevented || e.button || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    var a = e.target && e.target.closest ? e.target.closest('a[href]') : null;
    if (!a) return;
    var href = a.getAttribute('href');
    if (!href || href.charAt(0) === '#') return;
    if (a.target && a.target !== '_self') return;
    if (a.hasAttribute('download') || a.hasAttribute('data-no-lite-nav')) return;
    if (!sameOrigin(href)) return;
    var url = new URL(href, location.href);
    if (url.href === location.href) return;
    // Non-page destinations keep native behaviour.
    if (url.pathname.indexOf('/api/') === 0 || url.pathname.indexOf('/auth/') === 0) return;
    if (/\.[a-z0-9]{2,5}$/i.test(url.pathname)) return;
    e.preventDefault();
    softNav(url.href, false);
  }, true);

  window.addEventListener('popstate', function () {
    softNav(location.href, true);
  });
})();
