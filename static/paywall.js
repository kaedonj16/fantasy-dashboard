/**
 * Paywall UI for premium features
 *
 * Usage:
 *   checkPremiumAccess(userId, leagueId).then(hasPremium => {
 *     if (!hasPremium) {
 *       showPaywall('breakout-candidates');
 *     }
 *   });
 */

/**
 * Check if user has premium access
 */
async function checkPremiumAccess(userId, leagueId) {
  try {
    const params = new URLSearchParams();
    if (userId) params.append('user_id', userId);
    if (leagueId) params.append('league_id', leagueId);
    const platform = (window.__brctx || {}).platform;
    if (platform) params.append('platform', platform);
    const season = (window.__brctx || {}).season;
    if (season) params.append('season', season);

    const response = await fetch(`/api/subscription-status?${params}`);
    const data = await response.json();
    return data.has_premium || false;
  } catch (error) {
    console.error('[paywall] Error checking premium access:', error);
    return false;
  }
}

/**
 * Get subscription info
 */
async function getSubscriptionInfo(userId, leagueId) {
  try {
    const params = new URLSearchParams();
    if (userId) params.append('user_id', userId);
    if (leagueId) params.append('league_id', leagueId);
    const platform = (window.__brctx || {}).platform;
    if (platform) params.append('platform', platform);
    const season = (window.__brctx || {}).season;
    if (season) params.append('season', season);

    const response = await fetch(`/api/subscription-status?${params}`);
    return await response.json();
  } catch (error) {
    console.error('[paywall] Error getting subscription info:', error);
    return { has_premium: false, subscription_type: null };
  }
}

/**
 * Show a value-forward PRO preview in a container (does not expose gated content).
 */
window.brProPreview = function brProPreview(container, opts) {
  opts = opts || {};
  var el = (typeof container === 'string') ? document.getElementById(container) : container;
  if (!el) return;
  var count = opts.count;
  var countHtml = (count != null && count !== '')
    ? '<div class="br-pro-preview-count">' + count + '</div>'
    : '';
  var msg = opts.message || 'Unlock the full analysis for your league.';
  var ctaLabel = opts.ctaLabel || 'Unlock';
  var feature = opts.feature || 'trade-suggestions';
  el.innerHTML =
    '<div class="br-pro-preview">' +
      countHtml +
      '<div class="br-pro-preview-msg">' + msg + '</div>' +
      '<button type="button" class="br-pro-preview-cta" data-pro-feature="' + feature + '">' + ctaLabel + ' &rarr;</button>' +
    '</div>';
  var btn = el.querySelector('.br-pro-preview-cta');
  if (btn) {
    btn.addEventListener('click', function () {
      if (typeof showPaywall === 'function') showPaywall(feature, opts);
    });
  }
};

/**
 * Shared dismissible upsell-nudge infra. Every prompt built on this:
 * - never blocks a free feature (it is a banner, not a gate),
 * - is dismissible, and the dismissal is remembered in localStorage so the
 *   user is never re-nagged,
 * - respects the existing promo frequency cap (window._brPromoEligible: only
 *   on a later visit, >= 1 day after first seen) for passive prompts.
 * User-initiated prompts (tapping a locked metric) skip the eligibility check:
 * the user asked what they're missing, so answering is not nagging.
 */
(function () {
  var STORE_KEY = 'br-upsell-dismissed.v1';

  function _readStore() {
    try {
      var raw = localStorage.getItem(STORE_KEY);
      var obj = raw ? JSON.parse(raw) : {};
      return (obj && typeof obj === 'object') ? obj : {};
    } catch (_) { return {}; }
  }
  function _writeStore(obj) {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(obj || {})); }
    catch (_) { /* private mode: dismissal just won't persist */ }
  }
  function _esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  window.brUpsell = {
    /** True when the user already dismissed this nudge key. */
    dismissed: function (key) {
      if (!key) return false;
      return !!_readStore()[key];
    },
    /** Remember a dismissal so the nudge never shows again. */
    dismiss: function (key) {
      if (!key) return;
      var store = _readStore();
      store[key] = new Date().toISOString();
      _writeStore(store);
    },
    /**
     * Passive-prompt frequency cap. Reuses the existing promo eligibility
     * (first-seen + 1 day) when the host page defines it; pages without it
     * (share pages, modals) fall back to always eligible.
     */
    eligible: function () {
      try {
        if (typeof window._brPromoEligible === 'function') return !!window._brPromoEligible();
      } catch (_) {}
      return true;
    },
    /**
     * Render a slim dismissible inline nudge into `container`.
     * opts: { key (required, stable), message, ctaLabel, feature }.
     * Returns true when the nudge rendered, false when skipped (dismissed,
     * ineligible, or missing container/key). The CTA opens the paywall for
     * `feature`; the x persists the dismissal.
     */
    nudge: function (container, opts) {
      opts = opts || {};
      var key = opts.key;
      var el = (typeof container === 'string') ? document.getElementById(container) : container;
      if (!el || !key) return false;
      if (window.brUpsell.dismissed(key)) return false;
      if (!window.brUpsell.eligible()) return false;
      var feature = opts.feature || 'pro';
      var ctaLabel = opts.ctaLabel || 'Unlock PRO';
      el.innerHTML =
        '<div class="br-upsell-nudge" role="note">' +
          '<span class="br-upsell-nudge-msg">' + _esc(opts.message || 'Unlock more with PRO.') + '</span>' +
          '<button type="button" class="br-upsell-nudge-cta">' + _esc(ctaLabel) + '</button>' +
          '<button type="button" class="br-upsell-nudge-x" aria-label="Dismiss">' +
            '<i class="fa-solid fa-xmark" aria-hidden="true"></i>' +
          '</button>' +
        '</div>';
      var root = el.querySelector('.br-upsell-nudge');
      if (!root) return false;
      root.querySelector('.br-upsell-nudge-cta').addEventListener('click', function () {
        if (typeof showPaywall === 'function') showPaywall(feature, { source: 'nudge:' + key });
      });
      root.querySelector('.br-upsell-nudge-x').addEventListener('click', function () {
        window.brUpsell.dismiss(key);
        if (root.parentNode) root.parentNode.removeChild(root);
      });
      return true;
    }
  };
})();

/**
 * Resolve a paywall `feature` key to the { name, benefit } shown in the modal
 * headline. PRO-gated metric and preset keys (advanced-metrics-metric-<key>,
 * advanced-metrics-<preset>) resolve against the info maps the Advanced
 * Metrics page publishes on window, so a free user tapping a lock sees WHAT
 * they are missing (metric name + one-line why it matters) instead of a
 * generic "Premium Feature" dead end.
 */
window.brResolveProFeature = function brResolveProFeature(feature) {
  if (typeof feature !== 'string') return null;
  var mm = feature.match(/^advanced-metrics-metric-([A-Za-z0-9_]+)$/);
  if (mm) {
    var info = (window.__brProMetricInfo || {})[mm[1]];
    if (info && info.label) {
      return { name: info.label, benefit: info.why || 'A PRO intelligence metric.' };
    }
    return { name: 'PRO metric', benefit: 'A PRO intelligence metric.' };
  }
  var pm = feature.match(/^advanced-metrics-([a-z_]+)$/);
  if (pm) {
    var pi = (window.__brProPresetInfo || {})[pm[1]];
    if (pi && pi.label) {
      return { name: pi.label, benefit: pi.tagline || 'A PRO decision view.' };
    }
  }
  var extra = {
    'advanced-metrics-movers': {
      name: 'Movers: heating up and cooling off',
      benefit: 'PRO tracks usage and xFP trends across every metric, so you see who is heating up and cooling off before your league does.'
    },
    'wrapped-pro': {
      name: 'The full story',
      benefit: 'PRO unlocks the Front Office Report, Breakout Engine, AI trade analysis, and the premium Wrapped storyline for your league.'
    }
  };
  return extra[feature] || null;
};

const BR_PRO_PLANS = [
  { key: 'starter', name: 'Starter', leagues: '1 league', annual: '$10/year', monthly: '$1.49/mo', coverage: 'PRO for you in 1 league of your choice. Pick it after checkout and change it anytime.', cta: 'Choose Starter' },
  { key: 'all_pro', name: 'All-Pro', leagues: '5 leagues', annual: '$30/year', monthly: '$4.49/mo', coverage: 'PRO for you in up to 5 leagues. Pick them after checkout and change them anytime.', cta: 'Choose All-Pro', recommended: true },
  { key: 'hall_of_fame', name: 'Hall of Fame', leagues: 'Unlimited leagues', annual: '$50/year', monthly: '$7.49/mo', coverage: 'PRO for you in every league you play. Your league mates are not upgraded.', cta: 'Choose Hall of Fame' }
];

/** Plans that need a league chosen before checkout (seeds the first PRO slot).
 * Empty: no current plan requires a league at checkout. Starter/All-Pro buyers
 * pick their slot leagues after purchase from the Your PRO card on /pricing. */
const BR_LEAGUE_PLANS = {};

function proPlanCards(options) {
  options = options || {};
  return BR_PRO_PLANS.map(function (plan) {
    const action = options.dataPlan
      ? `data-plan="${plan.key}"`
      : `onclick="initiatePurchase('${plan.key}', this)"`;
    return `<article class="pricing-option${plan.recommended ? ' featured' : ''}" data-plan-card="${plan.key}">
      <div class="pricing-header"><h4>${plan.name}</h4>${plan.recommended ? '<div class="pricing-badge">Recommended</div>' : ''}</div>
      <p class="pricing-leagues">${plan.leagues}</p>
      <div class="pricing-price">${plan.annual.replace('/year', '<span>/year</span>')}<span class="paywall-price-alt"> or ${plan.monthly}</span></div>
      <p class="pricing-desc">${plan.coverage}</p>
      <button type="button" class="btn ${plan.recommended ? 'btn-primary' : 'btn-secondary'} paywall-cta" ${action}>${plan.cta}</button>
    </article>`;
  }).join('');
}

/**
 * Show paywall modal for a specific feature
 *
 * @param {string} feature - Feature name ('breakout-candidates', 'playoff-impact', 'gm-memo')
 * @param {object} [opts] - Optional preview context (count, message) for the modal headline
 */
window.showPaywall = function showPaywall(feature, opts) {
  opts = opts || {};
  const featureNames = {
    'breakout-candidates': 'Breakout Engine',
    'breakout-analysis': 'Breakout Engine',
    'ai-insights': 'AI Insights',
    'trade-history': 'Trade Intelligence',
    'trade-suggestions': 'Roster-Based Trade Suggestions',
    'trade-ai': 'AI Trade Analysis',
    'playoff-impact': 'Playoff Impact',
    'gm-memo': 'Front Office Report',
    'weekly-recap': 'Weekly Recap',
    'draft-cheat-sheet': 'Custom Draft Board',
    'draft-trends-scout': 'Trend Scout',
    'draft-analyzer': 'Draft Deep Dive Analyzer'
  };

  // Dynamic PRO-gated metric/preset keys resolve to the metric name + its
  // one-line why-it-matters, so a lock tap never lands on a generic headline.
  const _resolved = (typeof window.brResolveProFeature === 'function')
    ? window.brResolveProFeature(feature) : null;
  const featureName = (_resolved && _resolved.name) || featureNames[feature] || 'Premium Feature';
  const featureBenefits = {
    'breakout-candidates': 'Find emerging players by opportunity, peer history, and confidence.',
    'breakout-analysis': 'See the opportunity signals and confidence behind a breakout case.',
    'trade-history': 'Use real market activity to understand how players are being moved.',
    'trade-suggestions': 'Find roster-fit targets and packages built around your needs and surplus.',
    'trade-ai': 'Get a roster-aware trade read and practical counter ideas.',
    'playoff-impact': 'See how a proposed trade could shift your playoff outlook.',
    'gm-memo': 'Turn your roster, standings, and needs into a focused action plan.',
    'weekly-recap': 'Unlock the premium AI storyline; scores and other recap sections remain free.',
    'draft-cheat-sheet': 'Save a custom draft board that follows you into the Draft Room.',
    'draft-trends-scout': 'Spot historical ranking and ADP movement before your draft.',
    'draft-analyzer': 'Review draft decisions against the players who were still available.'
  };
  const featureBenefit = (_resolved && _resolved.benefit) || featureBenefits[feature] || 'Unlock more decision support for your fantasy teams.';
  const previewLine = opts.count != null
    ? `<p class="paywall-preview-line"><strong>${opts.count}</strong> ${opts.message || 'available with PRO'}</p>`
    : (opts.message ? `<p class="paywall-preview-line">${opts.message}</p>` : '');

  // CRITICAL: Properly close existing paywalls before removing them to restore inert state
  var existingPaywalls = document.querySelectorAll('.paywall-modal');
  if (existingPaywalls.length > 0) {
    var inertRoot = document.getElementById('app-scale') || document.getElementById('page-root');
    if (inertRoot && inertRoot.hasAttribute('inert')) {
      inertRoot.removeAttribute('inert');
    }
    existingPaywalls.forEach(function (el) { el.remove(); });
  }

  const modal = document.createElement('div');
  modal.className = 'paywall-modal';
  modal.setAttribute('role', 'dialog');
  modal.setAttribute('aria-modal', 'true');
  modal.setAttribute('aria-labelledby', 'paywallTitle');
  modal.innerHTML = `
    <div class="paywall-overlay"></div>
    <div class="paywall-content">
      <div class="paywall-header">
        <h2 id="paywallTitle"><i class="fa-solid fa-lock" aria-hidden="true"></i> Premium Feature</h2>
        <button type="button" class="paywall-close" aria-label="Close">&times;</button>
      </div>
      <div class="paywall-body">
        <div class="paywall-icon"><i class="fa-solid fa-star" aria-hidden="true"></i></div>
        <h3>${featureName}</h3>
        ${previewLine}
        <p class="paywall-benefit">${featureBenefit}</p>
        <div class="paywall-pricing">${proPlanCards()}</div>
        <p class="paywall-auth-note"><i class="fa-brands fa-google" aria-hidden="true"></i> Google sign-in is required at checkout.</p>
        <details class="paywall-more"><summary>See other PRO tools</summary>
          <p>Trade Intel, Breakout Engine, Front Office Report, premium weekly recap storyline, playoff-impact simulations, Custom Draft Board, Trend Scout, and Draft Deep Dive.</p>
        </details>
        <a class="paywall-full-pricing" href="/pricing">Compare plans and see sample previews</a>        </div>
      </div>
    </div>
  `;

  document.body.appendChild(modal);

  var inertRoot = document.getElementById('app-scale') || document.getElementById('page-root');
  if (inertRoot) inertRoot.setAttribute('inert', '');

  const prevFocus = document.activeElement;
  function closePaywall() {
    modal.remove();
    if (inertRoot) inertRoot.removeAttribute('inert');
    document.removeEventListener('keydown', onKey);
    if (prevFocus && typeof prevFocus.focus === 'function') {
      try { prevFocus.focus(); } catch (_) {}
    }
  }
  function focusables() {
    return modal.querySelectorAll('a[href], button:not([disabled]), input:not([disabled]), select, textarea, [tabindex]:not([tabindex="-1"])');
  }
  function onKey(e) {
    if (modal.dataset.nestedOpen) return;
    if (e.key === 'Escape') { e.preventDefault(); closePaywall(); return; }
    if (e.key !== 'Tab') return;
    const nodes = focusables();
    if (!nodes.length) return;
    const first = nodes[0], last = nodes[nodes.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }
  document.addEventListener('keydown', onKey);
  modal.querySelector('.paywall-overlay').addEventListener('click', closePaywall);
  modal.querySelector('.paywall-close').addEventListener('click', closePaywall);
  const first = focusables()[0];
  if (first) try { first.focus(); } catch (_) {}
  _maybeAddTrialCta(modal);
}

/**
 * Prepend a one-click free-trial CTA to the paywall modal when the visitor
 * could still claim one. trial_available comes from /api/subscription-status
 * (signed in, never used the trial, not already PRO). Guests always see it:
 * the start endpoint routes them through Google sign-in first, then starts
 * the trial automatically.
 */
function _maybeAddTrialCta(modal) {
  var body = modal && modal.querySelector('.paywall-body');
  if (!body || body.querySelector('.paywall-trial-strip')) return;
  var showForGuest = !window._hasAccount;
  getSubscriptionInfo().then(function (d) {
    if (!d || d.has_premium) return;
    if (!d.trial_available && !showForGuest) return;
    var strip = document.createElement('div');
    strip.className = 'paywall-trial-strip';
    strip.innerHTML =
      '<div class="paywall-trial-copy"><strong>New to PRO?</strong>' +
      '<span>Start a 7-day free trial. No card required.</span></div>' +
      '<button type="button" class="btn btn-primary paywall-trial-btn">Start free trial</button>';
    strip.querySelector('.paywall-trial-btn').addEventListener('click', function () {
      var next = window.location.pathname + window.location.search;
      window.location.href = '/pro-trial/start?next=' + encodeURIComponent(next);
    });
    body.insertBefore(strip, body.firstChild);
  }).catch(function () {});
}

function _hasGoogleAccount() {
  return !!window._hasAccount;
}

function _checkoutLeagueId() {
  const ctx = window.__brctx || {};
  return (
    new URLSearchParams(window.location.search).get('league_id') ||
    window.location.pathname.split('/').filter(Boolean)[2] ||
    ctx.leagueId ||
    ''
  );
}

function _startGoogleSubscribe(planType, triggerBtn, extra) {
  extra = extra || {};
  const ctx = window.__brctx || {};
  // A league open on the page seeds the first PRO league slot; it is never
  // required at checkout under the new catalog.
  const leagueId = extra.leagueId || _checkoutLeagueId();
  const needsLeague = !!BR_LEAGUE_PLANS[planType];
  if (needsLeague && !leagueId) {
    _showIdentifyModal(planType, triggerBtn);
    return;
  }
  const payload = {
    plan: planType,
    league_id: leagueId || '',
    platform: extra.platform || ctx.platform || 'sleeper',
    season: extra.season || ctx.season || new Date().getFullYear(),
    username: extra.username || '',
  };
  if (triggerBtn) {
    triggerBtn.disabled = true;
    if (!triggerBtn.dataset.origText) triggerBtn.dataset.origText = triggerBtn.innerHTML;
    triggerBtn.innerHTML = 'Continue with Google…';
  }
  fetch('/api/pro-signup/pending', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify(payload),
  })
    .then(function (r) {
      return r.json().then(function (d) { return { ok: r.ok, data: d }; });
    })
    .then(function (res) {
      if (!res.ok) {
        if (triggerBtn) {
          triggerBtn.disabled = false;
          triggerBtn.innerHTML = triggerBtn.dataset.origText;
        }
        if (window.showToast) {
          showToast((res.data && res.data.error) || 'Sign in with Google to subscribe.', 'error');
        }
        _showIdentifyModal(planType, triggerBtn);
        return;
      }
      window.location.href = (res.data && res.data.auth_url)
        || '/auth/google?intent=onboarding&next=/pro/resume-checkout';
    })
    .catch(function () {
      window.location.href = '/auth/google?intent=onboarding&next=/pro/resume-checkout';
    });
}

async function initiatePurchase(type, btn) {
  // Checkout requires a Google account site-wide. Guests get the Google
  // sign-in prompt first. New-catalog plans never require a league at
  // checkout; a league open on the page seeds the first PRO slot.
  const ctx = window.__brctx || {};
  const billingInterval = window.__brBillingInterval === 'month' ? 'month' : 'year';
  const leagueId = new URLSearchParams(window.location.search).get('league_id') ||
    window.location.pathname.split('/').filter(Boolean)[2] ||
    (ctx.leagueId || '');

  if (!_hasGoogleAccount()) {
    if (leagueId) {
      _startGoogleSubscribe(type, btn);
      return;
    }
    _showIdentifyModal(type, btn);
    return;
  }

  // Build a destination that lands in the league dashboard after payment
  const _platform = ctx.platform || 'sleeper';
  const _season   = ctx.season   || new Date().getFullYear();
  const _welcome = 'personal';  // All current plans are buyer-only coverage.
  // No league context at checkout: land on /pricing so the Your PRO card
  // can nudge the buyer to assign their league slots.
  const returnUrl = leagueId
    ? `/${_platform}/${_season}/${leagueId}/dashboard?new_subscriber=1&welcome=${_welcome}`
    : `/pricing?new_subscriber=1&welcome=${_welcome}`;

  if (btn) {
    btn.disabled = true;
    btn.dataset.origText = btn.innerHTML;
    btn.innerHTML = '<span style="display:inline-flex;align-items:center;gap:8px;justify-content:center;">' +
      '<span style="width:16px;height:16px;border:2px solid currentColor;border-top-color:transparent;' +
      'border-radius:50%;display:inline-block;animation:paywall-spin .7s linear infinite;flex-shrink:0;"></span>' +
      'Redirecting…</span>';
  }

  try {
    const res = await fetch('/api/create-checkout-session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan: type, league_id: leagueId, return_url: returnUrl,
        platform: _platform, season: _season, interval: billingInterval }),
    });
    const data = await res.json();
    if (data.url) {
      window.location.href = data.url;
    } else {
      if (btn) { btn.disabled = false; btn.innerHTML = btn.dataset.origText; }
      if (_handleAlreadySubscribed(data, leagueId)) return;
      if (window.showToast) showToast(data.error || 'Could not start checkout. Sign in with Google to subscribe.', 'error', 5000);
      else alert(data.error || 'Could not start checkout. Sign in with Google to subscribe.');
    }
  } catch (e) {
    if (btn) { btn.disabled = false; btn.innerHTML = btn.dataset.origText; }
    if (window.showToast) showToast('Checkout unavailable. Please try again.', 'error', 5000);
    else alert('Checkout unavailable. Please try again.');
  }
}

/**
 * Show premium badge on locked features
 */
function addPremiumBadge(element) {
  const badge = document.createElement('span');
  badge.className = 'premium-badge';
  badge.innerHTML = '<i class="fa-solid fa-star" aria-hidden="true"></i> Premium';
  badge.style.cssText = `
    display: inline-block;
    padding: 2px 8px;
    background: linear-gradient(135deg, #122d4b 0%, #2563eb 100%);
    color: white;
    font-size: 11px;
    font-weight: 600;
    border-radius: 12px;
    margin-left: 8px;
    vertical-align: middle;
  `;
  element.appendChild(badge);
}

/**
 * Lock a feature behind paywall
 */
async function protectFeature(featureName, userId, leagueId, callbackIfPremium) {
  const hasPremium = await checkPremiumAccess(userId, leagueId);

  if (hasPremium) {
    // User has premium - execute callback
    if (callbackIfPremium) {
      callbackIfPremium();
    }
  } else {
    // Show paywall
    showPaywall(featureName);
  }

  return hasPremium;
}

/**
 * Self-contained Google subscribe gate. Sleeper username is not enough to pay.
 */
function _handleAlreadySubscribed(data, leagueId) {
  if (!data.error || !data.error.toLowerCase().includes('already have')) return false;

  // They're already subscribed - mark premium and offer invite copy for league plans.
  if (window.__brctx) window.__brctx.isPremium = true;

  const ctx = window.__brctx || {};
  const platform = ctx.platform || 'sleeper';
  const season   = ctx.season   || new Date().getFullYear();
  const lid      = leagueId || ctx.leagueId || '';

  if (lid && typeof window.copyLeagueProInvite === 'function') {
    window.copyLeagueProInvite(platform, season, lid).then(function (ok) {
      if (ok && window.showToast) {
        showToast('PRO is already on -- invite link copied for your league mates.', 'success', 5000);
      }
    });
  }

  const dest = lid
    ? `/${platform}/${season}/${lid}/dashboard`
    : window.location.pathname;

  window.location.href = dest;
  return true;
}

/** Build + copy the shareable league-PRO invite URL. */
window.copyLeagueProInvite = async function copyLeagueProInvite(platform, season, leagueId) {
  const plat = platform || (window.__brctx || {}).platform || 'sleeper';
  const sea = season || (window.__brctx || {}).season || new Date().getFullYear();
  const lid = leagueId || (window.__brctx || {}).leagueId || '';
  if (!lid) return false;
  const url = `${window.location.origin}/invite/${encodeURIComponent(plat)}/${encodeURIComponent(sea)}/${encodeURIComponent(lid)}`;
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(url);
    } else {
      const ta = document.createElement('textarea');
      ta.value = url; document.body.appendChild(ta); ta.select();
      document.execCommand('copy'); ta.remove();
    }
    return true;
  } catch (e) {
    return false;
  }
};

/** Show Invite league control when the viewer bought a league/combo plan. */
window.refreshLeagueProInviteCta = async function refreshLeagueProInviteCta() {
  const ctx = window.__brctx || {};
  const lid = ctx.leagueId || '';
  if (!lid || !ctx.is_logged_in) return;
  try {
    const params = new URLSearchParams({
      league_id: lid,
      platform: ctx.platform || 'sleeper',
      season: String(ctx.season || ''),
    });
    const res = await fetch(`/api/subscription-status?${params}`, { cache: 'no-store' });
    if (!res.ok) return;
    const data = await res.json();
    if (!data.has_league_subscription) return;

    const mount = document.getElementById('leagueProInviteMount')
      || document.querySelector('[data-league-pro-invite]');
    // Floating dismissible banner when buyer, teammate with PRO, or nudge for
    // league-mates who haven't claimed shared access yet.
    const key = data.is_league_buyer
      ? `league-pro-invite-${lid}`
      : data.has_premium
        ? `league-pro-teammate-${lid}`
        : `league-pro-nudge-${lid}`;
    try {
      if (sessionStorage.getItem('br_skip_league_pro_banner') === '1') return;
    } catch (e) {}
    try { if (localStorage.getItem(key) === '1') return; } catch (e) {}

    let el = document.getElementById('leagueProShareBanner');
    if (el) el.remove();
    el = document.createElement('div');
    el.id = 'leagueProShareBanner';
    el.setAttribute('role', 'status');
    // Lift above the mobile dock nav so the CTA button isn't clipped behind it;
    // --dock-safe-bottom is 0 on desktop and the dock height + safe area on mobile.
    el.style.cssText = 'position:fixed;bottom:calc(24px + var(--dock-safe-bottom, 0px));right:24px;z-index:9998;max-width:min(320px, calc(100vw - 48px));background:var(--card);border:1px solid var(--border);border-top:3px solid #2563eb;border-radius:14px;box-shadow:0 12px 40px rgba(0,0,0,.22);padding:16px 18px;display:flex;flex-direction:column;gap:10px;';
    if (data.is_league_buyer) {
      if (!data.invite_path) return;
      el.innerHTML = `
        <div style="display:flex;justify-content:space-between;gap:8px;align-items:flex-start;">
          <strong style="font-size:14px;color:var(--text);">Invite your league</strong>
          <button type="button" aria-label="Dismiss" data-dismiss
            style="background:none;border:none;color:var(--text-muted);font-size:18px;cursor:pointer;line-height:1;">&times;</button>
        </div>
        <p style="margin:0;font-size:13px;color:var(--text-muted);line-height:1.45;">
          PRO is on for every manager. Copy a link they can open to sign in.
        </p>
        <button type="button" data-copy
          style="padding:10px 12px;border:none;border-radius:9px;background:#2563eb;color:#fff;font-weight:700;font-size:13px;cursor:pointer;">
          Copy invite link
        </button>`;
    } else if (data.has_premium) {
      el.innerHTML = `
        <div style="display:flex;justify-content:space-between;gap:8px;align-items:flex-start;">
          <strong style="font-size:14px;color:var(--text);">League PRO is unlocked</strong>
          <button type="button" aria-label="Dismiss" data-dismiss
            style="background:none;border:none;color:var(--text-muted);font-size:18px;cursor:pointer;line-height:1;">&times;</button>
        </div>
        <p style="margin:0;font-size:13px;color:var(--text-muted);line-height:1.45;">
          A league mate unlocked shared premium. Try Trade Intel or the Breakout Engine.
        </p>
        <a href="/${encodeURIComponent(ctx.platform || 'sleeper')}/${encodeURIComponent(ctx.season || '')}/${encodeURIComponent(lid)}/trade-intel"
           style="display:inline-block;text-align:center;padding:10px 12px;border-radius:9px;background:#2563eb;color:#fff;font-weight:700;font-size:13px;text-decoration:none;">
          Open Trade Intel
        </a>`;
    } else {
      const claimHref = data.invite_path
        ? `${window.location.origin}${data.invite_path}`
        : '/pricing';
      el.innerHTML = `
        <div style="display:flex;justify-content:space-between;gap:8px;align-items:flex-start;">
          <strong style="font-size:14px;color:var(--text);">Your league has PRO</strong>
          <button type="button" aria-label="Dismiss" data-dismiss
            style="background:none;border:none;color:var(--text-muted);font-size:18px;cursor:pointer;line-height:1;">&times;</button>
        </div>
        <p style="margin:0;font-size:13px;color:var(--text-muted);line-height:1.45;">
          A league mate unlocked shared premium. Claim access to try Trade Intel and the Breakout Engine.
        </p>
        <a href="${claimHref}"
           style="display:inline-block;text-align:center;padding:10px 12px;border-radius:9px;background:#2563eb;color:#fff;font-weight:700;font-size:13px;text-decoration:none;">
          Claim league PRO
        </a>`;
    }
    document.body.appendChild(el);
    const dismiss = el.querySelector('[data-dismiss]');
    if (dismiss) dismiss.addEventListener('click', function () {
      el.remove();
      try { localStorage.setItem(key, '1'); } catch (e) {}
    });
    const copy = el.querySelector('[data-copy]');
    if (copy) copy.addEventListener('click', function () {
      window.copyLeagueProInvite(ctx.platform, ctx.season, lid).then(function (ok) {
        if (ok) {
          copy.textContent = 'Copied';
          if (window.showToast) showToast('Invite link copied', 'success');
        }
      });
    });
  } catch (e) {
    console.debug('[league-pro-invite]', e);
  }
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', function () {
    if (typeof window.refreshLeagueProInviteCta === 'function') window.refreshLeagueProInviteCta();
  });
} else if (typeof window.refreshLeagueProInviteCta === 'function') {
  window.refreshLeagueProInviteCta();
}

function _activePaywall() {
  return document.querySelector('.paywall-modal');
}

function _pausePaywallForNested() {
  const paywall = _activePaywall();
  if (!paywall) return;
  paywall.dataset.nestedOpen = '1';
  paywall.setAttribute('aria-hidden', 'true');
  paywall.setAttribute('inert', '');
}

function _resumePaywallAfterNested() {
  const paywall = _activePaywall();
  if (!paywall) return;
  delete paywall.dataset.nestedOpen;
  paywall.removeAttribute('aria-hidden');
  paywall.removeAttribute('inert');
}

function _stackAbovePaywall(modal) {
  if (!modal) return;
  modal.classList.add('over-paywall');
  if (modal.parentElement !== document.body) document.body.appendChild(modal);
  _pausePaywallForNested();
}

const _CHECKOUT_PLANS = { starter: 1, all_pro: 1, hall_of_fame: 1 };

function _hookCheckoutLinkModal() {
  if (window.__brCheckoutLinkHooked) return;
  window.__brCheckoutLinkHooked = true;
  const origClose = window.closeLinkModal;
  window.closeLinkModal = function () {
    const link = document.getElementById('linkModal');
    if (link && link.classList.contains('over-paywall')) {
      link.classList.remove('over-paywall');
      const title = link.querySelector('.link-head span');
      if (title && link.dataset.prevTitle) title.textContent = link.dataset.prevTitle;
      delete link.dataset.prevTitle;
      const hint = document.getElementById('linkCheckoutHint');
      if (hint) hint.remove();
      _resumePaywallAfterNested();
      window.__brCheckoutPlan = null;
      window.__brCheckoutBtn = null;
    }
    if (typeof origClose === 'function') origClose();
  };
}

/** Open the Link-a-league modal so checkout can pick any platform, then Google. */
function _openCheckoutLeaguePicker(planType, triggerBtn) {
  const link = document.getElementById('linkModal');
  if (!link || typeof window.openLinkModal !== 'function') {
    _showIdentifyModal(planType, triggerBtn);
    return;
  }
  _hookCheckoutLinkModal();
  window.__brCheckoutPlan = planType;
  window.__brCheckoutBtn = triggerBtn;
  const title = link.querySelector('.link-head span');
  if (title && !link.dataset.prevTitle) {
    link.dataset.prevTitle = title.textContent || 'Link a league';
    title.textContent = 'Choose a league';
  }
  let hint = document.getElementById('linkCheckoutHint');
  if (!hint) {
    hint = document.createElement('p');
    hint.id = 'linkCheckoutHint';
    hint.className = 'link-help';
    hint.style.margin = '0 0 12px';
    const head = link.querySelector('.link-head');
    if (head && head.parentNode) head.insertAdjacentElement('afterend', hint);
  }
  hint.textContent = (function () {
    const names = Array.from(link.querySelectorAll('.link-tab')).map(function (t) {
      return (t.textContent || '').trim();
    }).filter(Boolean);
    const list = names.length ? names.join(', ') : 'Sleeper, ESPN, MFL, Fleaflicker, or Yahoo';
    return 'Pick ' + list + ', then continue. Google sign-in happens after you choose a league.';
  })();
  _stackAbovePaywall(link);
  window.openLinkModal();
  const ctx = window.__brctx || {};
  const lid = ctx.leagueId && ctx.leagueId !== 'None' ? String(ctx.leagueId) : '';
  if (lid && ctx.platform && typeof window.linkMyTeam === 'function') {
    window.linkMyTeam(ctx.platform, lid, ctx.season);
  } else if (typeof window.linkTab === 'function') {
    window.linkTab('sleeper');
  }
}

/** Signed-in league picker when a plan needs a league but URL has none. */
function _showLeaguePickerModal(planType, triggerBtn) {
  const existing = document.getElementById('_leaguePickerModal');
  if (existing) existing.remove();

  const modal = document.createElement('div');
  modal.id = '_leaguePickerModal';
  modal.className = 'signin-modal-overlay';
  modal.setAttribute('role', 'dialog');
  modal.setAttribute('aria-modal', 'true');
  modal.setAttribute('aria-labelledby', '_leaguePickerTitle');
  modal.style.display = 'flex';
  modal.innerHTML = `
    <div class="signin-modal-box">
      <h3 class="signin-modal-title" id="_leaguePickerTitle">Choose a league</h3>
      <p class="signin-modal-sub">Select which league this subscription applies to.</p>
      <div id="_leaguePickerWrap" style="margin-bottom:16px;">
        <label style="display:block;font-size:11px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px;">Your leagues</label>
        <select class="signin-modal-input" id="_leaguePickerSelect" style="margin-bottom:0;cursor:pointer;">
          <option value="">Loading…</option>
        </select>
      </div>
      <div id="_leaguePickerError" style="display:none;font-size:12px;color:#ef4444;margin:-8px 0 12px;"></div>
      <div class="signin-modal-actions">
        <button type="button" class="signin-modal-submit" id="_leaguePickerSubmit" disabled>Continue to Checkout</button>
        <button type="button" class="signin-modal-cancel" id="_leaguePickerCancel">Cancel</button>
      </div>
      <button type="button" class="signin-modal-cancel" id="_leaguePickerOther" style="width:100%;margin-top:10px;">Connect a league on another platform</button>
    </div>`;
  document.body.appendChild(modal);
  _stackAbovePaywall(modal);

  const select = modal.querySelector('#_leaguePickerSelect');
  const submitBtn = modal.querySelector('#_leaguePickerSubmit');
  const errorEl = modal.querySelector('#_leaguePickerError');
  const prevFocus = document.activeElement;

  function closePicker() {
    document.removeEventListener('keydown', onKey);
    _resumePaywallAfterNested();
    modal.remove();
    if (prevFocus && typeof prevFocus.focus === 'function') {
      try { prevFocus.focus(); } catch (_) {}
    }
  }
  function onKey(e) {
    if (e.key === 'Escape') { e.preventDefault(); closePicker(); }
  }
  document.addEventListener('keydown', onKey);
  modal.addEventListener('click', e => { if (e.target === modal) closePicker(); });
  modal.querySelector('#_leaguePickerCancel').addEventListener('click', closePicker);
  modal.querySelector('#_leaguePickerOther').addEventListener('click', function () {
    closePicker();
    _openCheckoutLeaguePicker(planType, triggerBtn);
  });

  window.brGetMyLeagues({ force: false })
    .then(data => {
      const leagues = (data && data.leagues) || [];
      if (!leagues.length) {
        select.innerHTML = '<option value="">No leagues found</option>';
        errorEl.textContent = 'Connect a league first, then subscribe.';
        errorEl.style.display = 'block';
        return;
      }
      select.innerHTML = leagues.map(lg => {
        const id = lg.league_id || lg.id || '';
        const name = lg.name || lg.league_name || id;
        const plat = lg.platform || 'sleeper';
        const season = lg.season || '';
        const label = season ? `${name} (${plat} · ${season})` : `${name} (${plat})`;
        return `<option value="${id}" data-platform="${plat}" data-season="${season}">${label}</option>`;
      }).join('');
      submitBtn.disabled = false;
      select.focus();
    })
    .catch(() => {
      select.innerHTML = '<option value="">Unable to load leagues</option>';
      errorEl.textContent = 'Could not load your leagues. Try again.';
      errorEl.style.display = 'block';
    });

  submitBtn.addEventListener('click', () => {
    const opt = select.options[select.selectedIndex];
    const leagueId = (select.value || '').trim();
    if (!leagueId) {
      errorEl.textContent = 'Pick a league to continue.';
      errorEl.style.display = 'block';
      return;
    }
    if (window.__brctx) {
      window.__brctx.leagueId = leagueId;
      if (opt && opt.dataset.platform) window.__brctx.platform = opt.dataset.platform;
      if (opt && opt.dataset.season) window.__brctx.season = Number(opt.dataset.season) || window.__brctx.season;
    }
    closePicker();
    _initiatePurchaseWithLeague(planType, triggerBtn, leagueId);
  });
}

function _showIdentifyModal(planType, triggerBtn) {
  const existing = document.getElementById('_identifyModal');
  if (existing) existing.remove();

  const needsLeague = !!BR_LEAGUE_PLANS[planType];
  const next = encodeURIComponent(window.location.pathname + window.location.search);
  const yahooOn = !!document.querySelector('#linkModal .link-tab[data-lp="yahoo"]');
  const googleCtl =
    `<a class="google-continue-btn" id="_identifyGoogle" href="/auth/google?intent=login&amp;next=${next}"><span class="google-button-title">Continue with Google</span></a>`;
  const platformTabs = needsLeague ? `
      <div class="link-tabs" id="_identifyPlatTabs" role="tablist" style="margin-bottom:14px;">
        <button type="button" class="link-tab active" data-lp="sleeper">Sleeper</button>
        <button type="button" class="link-tab" data-lp="espn">ESPN</button>
        <button type="button" class="link-tab" data-lp="mfl">MFL</button>
        <button type="button" class="link-tab" data-lp="fleaflicker">Fleaflicker</button>
        ${yahooOn ? '<button type="button" class="link-tab" data-lp="yahoo">Yahoo</button>' : ''}
      </div>` : '';

  const modal = document.createElement('div');
  modal.id = '_identifyModal';
  modal.className = 'signin-modal-overlay';
  modal.setAttribute('role', 'dialog');
  modal.setAttribute('aria-modal', 'true');
  modal.setAttribute('aria-labelledby', '_identifyTitle');
  modal.style.display = 'flex';
  modal.innerHTML = `
    <div class="signin-modal-box">
      <h3 class="signin-modal-title" id="_identifyTitle">${needsLeague ? 'Choose a league' : 'Sign in with Google to subscribe'}</h3>
      <p class="signin-modal-sub" id="_identifySub">${needsLeague
        ? 'Pick a league on any platform, then continue with Google. A Google account is required to subscribe.'
        : 'A Google account is required to subscribe. Continue with Google, or enter a Sleeper username to find your leagues.'}</p>
      ${platformTabs}
      ${googleCtl}
      <div class="signin-modal-or">or</div>
      <div id="_identifySleeperPane">
        <input class="signin-modal-input" id="_identifyInput" type="text" placeholder="Sleeper username" aria-label="Sleeper username" autocomplete="username">
      </div>
      <div id="_identifyExtWrap" style="display:none;margin-bottom:16px;">
        <label style="display:block;font-size:11px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px;">League ID</label>
        <input class="signin-modal-input" id="_identifyExtId" type="text" placeholder="e.g. 123456" autocomplete="off">
        <label style="display:block;font-size:11px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px;">Season</label>
        <input class="signin-modal-input" id="_identifyExtSeason" type="text" inputmode="numeric" placeholder="current season" autocomplete="off">
      </div>
      <div id="_identifyLeagueWrap" style="display:none;margin-bottom:16px;">
        <label style="display:block;font-size:11px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px;">Select League</label>
        <select class="signin-modal-input" id="_identifyLeague" style="margin-bottom:0;cursor:pointer;"></select>
      </div>
      <div id="_identifyError" style="display:none;font-size:12px;color:#ef4444;margin:-8px 0 12px;"></div>
      <div class="signin-modal-actions">
        <button type="button" class="signin-modal-submit" id="_identifySubmit">Continue</button>
        <button type="button" class="signin-modal-cancel" id="_identifyCancel">Cancel</button>
      </div>
    </div>`;
  document.body.appendChild(modal);
  _stackAbovePaywall(modal);

  const input = modal.querySelector('#_identifyInput');
  const submitBtn = modal.querySelector('#_identifySubmit');
  const errorEl = modal.querySelector('#_identifyError');
  const leagueWrap = modal.querySelector('#_identifyLeagueWrap');
  const leagueSel = modal.querySelector('#_identifyLeague');
  const subText = modal.querySelector('#_identifySub');
  const sleeperPane = modal.querySelector('#_identifySleeperPane');
  const extWrap = modal.querySelector('#_identifyExtWrap');
  const extId = modal.querySelector('#_identifyExtId');
  const extSeason = modal.querySelector('#_identifyExtSeason');
  const googleBtn = modal.querySelector('#_identifyGoogle');
  const prevFocus = document.activeElement;
  let identPlat = 'sleeper';

  function focusables() {
    return modal.querySelectorAll('a[href], button:not([disabled]), input:not([disabled]), select, textarea, [tabindex]:not([tabindex="-1"])');
  }
  function closeIdentify() {
    document.removeEventListener('keydown', onKey);
    _resumePaywallAfterNested();
    modal.remove();
    if (prevFocus && typeof prevFocus.focus === 'function') {
      try { prevFocus.focus(); } catch (_) {}
    }
  }
  function onKey(e) {
    if (e.key === 'Escape') { e.preventDefault(); closeIdentify(); return; }
    if (e.key !== 'Tab') return;
    const nodes = focusables();
    if (!nodes.length) return;
    const first = nodes[0], last = nodes[nodes.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }
  document.addEventListener('keydown', onKey);
  modal.addEventListener('click', e => { if (e.target === modal) closeIdentify(); });
  modal.querySelector('#_identifyCancel').addEventListener('click', closeIdentify);
  const first = focusables()[0];
  if (first) try { first.focus(); } catch (_) {}

  let identified = false;

  if (needsLeague) {
    modal.querySelectorAll('#_identifyPlatTabs .link-tab').forEach(function (tab) {
      tab.addEventListener('click', function () {
        identPlat = tab.dataset.lp || 'sleeper';
        modal.querySelectorAll('#_identifyPlatTabs .link-tab').forEach(function (b) {
          b.classList.toggle('active', b === tab);
        });
        const sleeper = identPlat === 'sleeper';
        if (sleeperPane) sleeperPane.style.display = sleeper ? '' : 'none';
        if (extWrap) extWrap.style.display = sleeper ? 'none' : 'block';
        if (sleeper) leagueWrap.style.display = identified ? 'block' : 'none';
        else leagueWrap.style.display = 'none';
        errorEl.style.display = 'none';
      });
    });
  }

  async function goGoogleWithLeague() {
    let leagueId = '';
    let season = '';
    let name = '';
    let username = '';
    if (identPlat === 'sleeper') {
      leagueId = (leagueSel && leagueSel.value) || '';
      username = (input && input.value || '').trim();
      if (leagueSel && leagueSel.selectedIndex >= 0) {
        name = leagueSel.options[leagueSel.selectedIndex].textContent || '';
      }
    } else {
      leagueId = (extId && extId.value || '').trim();
      season = (extSeason && extSeason.value || '').trim();
    }
    if (!leagueId) {
      errorEl.textContent = 'Pick a league before continuing with Google.';
      errorEl.style.display = 'block';
      return;
    }
    errorEl.style.display = 'none';
    try {
      const payload = {
        platform: identPlat, league_id: leagueId, name, username,
        checkout_plan: planType,
      };
      if (season) payload.season = Number(season) || season;
      const res = await fetch('/api/link/pending', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        errorEl.textContent = data.error || 'Could not save that league.';
        errorEl.style.display = 'block';
        return;
      }
      window.location.href = data.auth_url || '/auth/google';
    } catch (e) {
      errorEl.textContent = 'Network error. Please try again.';
      errorEl.style.display = 'block';
    }
  }

  if (needsLeague && googleBtn) {
    googleBtn.addEventListener('click', function (e) {
      e.preventDefault();
      goGoogleWithLeague();
    });
  } else if (googleBtn && googleBtn.tagName === 'A') {
    googleBtn.addEventListener('click', function (e) {
      e.preventDefault();
      closeIdentify();
      _startGoogleSubscribe(planType, triggerBtn);
    });
  }

  async function doStep() {
    if (needsLeague && identPlat !== 'sleeper') {
      await goGoogleWithLeague();
      return;
    }
    if (!identified) {
      await doIdentify();
    } else {
      doCheckout();
    }
  }

  async function doIdentify() {
    const username = (input.value || '').trim();
    if (!username) { input.focus(); return; }

    submitBtn.disabled = true;
    submitBtn.textContent = 'Checking…';
    errorEl.style.display = 'none';

    try {
      const res = await fetch('/api/identify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        errorEl.textContent = data.error || 'Could not verify username.';
        errorEl.style.display = 'block';
        submitBtn.disabled = false;
        submitBtn.textContent = 'Continue';
        return;
      }

      identified = true;

      if (needsLeague && data.leagues && data.leagues.length > 0) {
        input.disabled = true;
        subText.textContent = 'Choose which league to subscribe for, then continue with Google.';
        leagueSel.innerHTML = data.leagues
          .map(lg => `<option value="${lg.league_id}" data-platform="sleeper" data-season="${lg.season || ''}">${lg.name}</option>`)
          .join('');
        leagueWrap.style.display = 'block';
        submitBtn.disabled = false;
        submitBtn.textContent = 'Continue with Google';
        leagueSel.focus();
      } else {
        closeIdentify();
        _startGoogleSubscribe(planType, triggerBtn, { username });
      }
    } catch (e) {
      errorEl.textContent = 'Network error. Please try again.';
      errorEl.style.display = 'block';
      submitBtn.disabled = false;
      submitBtn.textContent = 'Continue';
    }
  }

  function doCheckout() {
    const opt = leagueSel.options[leagueSel.selectedIndex];
    const leagueId = leagueSel.value || '';
    if (window.__brctx) {
      window.__brctx.leagueId = leagueId;
      if (opt && opt.dataset.platform) window.__brctx.platform = opt.dataset.platform;
      if (opt && opt.dataset.season) window.__brctx.season = Number(opt.dataset.season) || window.__brctx.season;
    }
    closeIdentify();
    if (!_hasGoogleAccount()) {
      _startGoogleSubscribe(planType, triggerBtn, { leagueId: leagueId });
      return;
    }
    _initiatePurchaseWithLeague(planType, triggerBtn, leagueId);
  }

  submitBtn.addEventListener('click', doStep);
  if (input) input.addEventListener('keydown', e => { if (e.key === 'Enter') doStep(); });
  if (extId) extId.addEventListener('keydown', e => { if (e.key === 'Enter') doStep(); });
}

async function _initiatePurchaseWithLeague(type, btn, leagueId) {
  if (!_hasGoogleAccount()) {
    _startGoogleSubscribe(type, btn, { leagueId });
    return;
  }
  const billingInterval = window.__brBillingInterval === 'month' ? 'month' : 'year';
  // Build a post-checkout destination: league dashboard if we have a league,
  // otherwise the current page. Append ?new_subscriber=1 to trigger the welcome tour.
  const ctx = window.__brctx || {};
  const platform = ctx.platform || 'sleeper';
  const season   = ctx.season   || new Date().getFullYear();
  const _welcome = 'personal';  // All current plans are buyer-only coverage.
  // No league context at checkout: land on /pricing so the Your PRO card
  // can nudge the buyer to assign their league slots.
  const returnUrl = leagueId
    ? `/${platform}/${season}/${leagueId}/dashboard?new_subscriber=1&welcome=${_welcome}`
    : `/pricing?new_subscriber=1&welcome=${_welcome}`;

  if (btn) {
    btn.disabled = true;
    btn.dataset.origText = btn.innerHTML;
    btn.innerHTML = '<span style="display:inline-flex;align-items:center;gap:8px;justify-content:center;">' +
      '<span style="width:16px;height:16px;border:2px solid currentColor;border-top-color:transparent;' +
      'border-radius:50%;display:inline-block;animation:paywall-spin .7s linear infinite;flex-shrink:0;"></span>' +
      'Redirecting…</span>';
  }
  try {
    const res = await fetch('/api/create-checkout-session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan: type, league_id: leagueId, return_url: returnUrl,
        platform, season, interval: billingInterval }),
    });
    const data = await res.json();
    if (data.url) {
      window.location.href = data.url;
    } else {
      if (btn) { btn.disabled = false; btn.innerHTML = btn.dataset.origText; }
      if (_handleAlreadySubscribed(data, leagueId)) return;
      if (window.showToast) showToast(data.error || 'Could not start checkout.', 'error', 5000);
      else alert(data.error || 'Could not start checkout.');
    }
  } catch (e) {
    if (btn) { btn.disabled = false; btn.innerHTML = btn.dataset.origText; }
    if (window.showToast) showToast('Checkout unavailable. Please try again.', 'error', 5000);
    else alert('Checkout unavailable. Please try again.');
  }
}

window._initiatePurchaseWithLeague = _initiatePurchaseWithLeague;

(function resumeCheckoutFromGoogle() {
  try {
    const params = new URLSearchParams(window.location.search);
    if (params.get('checkout') !== '1') return;
    const plan = params.get('plan') || '';
    if (!_CHECKOUT_PLANS[plan]) return;
    if (!_hasGoogleAccount()) return;
    history.replaceState(null, '', window.location.pathname);
    const start = function () {
      const btn = document.querySelector('[onclick*="initiatePurchase"]');
      initiatePurchase(plan, btn);
    };
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', start);
    } else {
      start();
    }
  } catch (e) {}
})();

function openHomeProModal() {
  // CRITICAL: Properly close existing paywalls before removing them to restore inert state
  const existingPaywalls = document.querySelectorAll('.paywall-modal');
  if (existingPaywalls.length > 0) {
    const inertRoot = document.getElementById('app-scale') || document.getElementById('page-root');
    if (inertRoot && inertRoot.hasAttribute('inert')) {
      inertRoot.removeAttribute('inert');
    }
    existingPaywalls.forEach(function (el) { el.remove(); });
  }

  const modal = document.createElement('div');
  modal.className = 'paywall-modal';
  modal.id = 'homeProModal';
  modal.setAttribute('role', 'dialog');
  modal.setAttribute('aria-modal', 'true');
  modal.setAttribute('aria-labelledby', 'homeProModalTitle');
  modal.innerHTML = `
    <div class="paywall-overlay"></div>
    <div class="paywall-content">
      <div class="paywall-header">
        <h2 id="homeProModalTitle"><i class="fa-solid fa-unlock" aria-hidden="true"></i> Unlock PRO</h2>
        <button type="button" class="paywall-close" aria-label="Close">&times;</button>
      </div>
      <div class="paywall-body">
        <div id="homeProStepPlan" class="home-pro-step">
          <h3>Choose a plan</h3>
          <p>A Google account is required to subscribe. After checkout, assign your PRO leagues from the Your PRO card on the pricing page.</p>
          <div class="paywall-pricing">${proPlanCards({ dataPlan: true })}</div>
          <p class="paywall-auth-note"><i class="fa-brands fa-google" aria-hidden="true"></i> Google sign-in is required at checkout.</p>
          <a class="paywall-full-pricing" href="/pricing">Compare features and see sample previews</a>
        </div>
      </div>
    </div>`;
  document.body.appendChild(modal);

  const inertRoot = document.getElementById('app-scale') || document.getElementById('page-root');
  if (inertRoot) inertRoot.setAttribute('inert', '');
  const prevFocus = document.activeElement;

  function closeModal() {
    modal.remove();
    if (inertRoot) inertRoot.removeAttribute('inert');
    document.removeEventListener('keydown', onKey);
    if (prevFocus && typeof prevFocus.focus === 'function') {
      try { prevFocus.focus(); } catch (_) {}
    }
  }
  function focusables() {
    return modal.querySelectorAll('a[href], button:not([disabled]), input:not([disabled]), select, textarea, [tabindex]:not([tabindex="-1"])');
  }
  function onKey(e) {
    if (e.key === 'Escape') { e.preventDefault(); closeModal(); return; }
    if (e.key !== 'Tab') return;
    const nodes = focusables();
    if (!nodes.length) return;
    const first = nodes[0], last = nodes[nodes.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }
  document.addEventListener('keydown', onKey);
  modal.querySelector('.paywall-overlay').addEventListener('click', closeModal);
  modal.querySelector('.paywall-close').addEventListener('click', closeModal);

  // No plan needs a league at checkout: picking a plan goes straight to the
  // Google sign-in / checkout flow. Starter/All-Pro buyers assign their league
  // slots from the Your PRO card on /pricing after purchase.
  modal.querySelectorAll('[data-plan]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      const selectedPlan = btn.getAttribute('data-plan') || '';
      if (window._hasAccount) {
        initiatePurchase(selectedPlan, btn);
      } else {
        _startGoogleSubscribe(selectedPlan, btn);
      }
    });
  });

  const first = focusables()[0];
  if (first) try { first.focus(); } catch (_) {}
}
window.openHomeProModal = openHomeProModal;

function initHomeProSignup() {
  document.querySelectorAll('[data-home-pro-open]').forEach(function (btn) {
    btn.addEventListener('click', function (e) {
      e.preventDefault();
      openHomeProModal();
    });
  });
  if (window.location.hash === '#homeProSignup') {
    openHomeProModal();
  }
}

// Defensive cleanup: Remove stuck inert from app root on page load
function cleanupStuckInert() {
  var inertRoot = document.getElementById('app-scale') || document.getElementById('page-root');
  if (inertRoot && inertRoot.hasAttribute('inert')) {
    // Only remove if no active paywall modal exists
    if (!document.querySelector('.paywall-modal')) {
      inertRoot.removeAttribute('inert');
      console.warn('[paywall] Removed stuck inert attribute from app root');
    }
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', function() {
    cleanupStuckInert();
    initHomeProSignup();
  });
} else {
  cleanupStuckInert();
  initHomeProSignup();
}
