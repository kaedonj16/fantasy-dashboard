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

  const featureName = featureNames[feature] || 'Premium Feature';
  const previewLine = opts.count != null
    ? `<p class="paywall-preview-line"><strong>${opts.count}</strong> ${opts.message || 'available with PRO'}</p>`
    : (opts.message ? `<p class="paywall-preview-line">${opts.message}</p>` : '');

  document.querySelectorAll('.paywall-modal').forEach(function (el) { el.remove(); });

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
        <p>This is a premium feature. Upgrade to access:</p>
        <ul class="paywall-features">
          <li>✓ Roster-Based Trade Suggestions</li>
          <li>✓ Full Trade Intelligence feed &amp; history</li>
          <li>✓ Breakout Engine candidate predictions</li>
          <li>✓ Playoff Impact simulations</li>
          <li>✓ Front Office Report</li>
          <li>✓ Weekly Recap</li>
          <li>✓ Custom Draft Board</li>
          <li>✓ Draft Deep Dive Analyzer</li>
        </ul>
        <div class="paywall-pricing">
          <div class="pricing-option">
            <div class="pricing-header">
              <h4>League Plan</h4>
            </div>
            <div class="pricing-price">$15<span>/year</span></div>
            <p class="pricing-desc">Premium for all managers in your league</p>
            <button class="btn btn-secondary paywall-cta" onclick="initiatePurchase('league', this)">
              Subscribe for League
            </button>
          </div>
          <div class="pricing-option featured">
            <div class="pricing-header">
              <h4>League + Personal</h4>
              <div class="pricing-badge">Best value</div>
            </div>
            <div class="pricing-price">$20<span>/year</span></div>
            <p class="pricing-desc">League premium + all your personal leagues</p>
            <button class="btn btn-primary paywall-cta" onclick="initiatePurchase('combo', this)">
              Subscribe Both
            </button>
          </div>
          <div class="pricing-option">
            <div class="pricing-header">
              <h4>Personal Plan</h4>
            </div>
            <div class="pricing-price">$10<span>/year</span></div>
            <p class="pricing-desc">Premium for all your leagues</p>
            <button class="btn btn-secondary paywall-cta" onclick="initiatePurchase('user', this)">
              Subscribe Personally
            </button>
          </div>
        </div>
      </div>
    </div>
  `;

  document.body.appendChild(modal);

  const prevFocus = document.activeElement;
  function closePaywall() {
    modal.remove();
    document.removeEventListener('keydown', onKey);
    if (prevFocus && typeof prevFocus.focus === 'function') {
      try { prevFocus.focus(); } catch (_) {}
    }
  }
  function focusables() {
    return modal.querySelectorAll('a[href], button:not([disabled]), input:not([disabled]), select, textarea, [tabindex]:not([tabindex="-1"])');
  }
  function onKey(e) {
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
}

async function initiatePurchase(type, btn) {
  // Prompt login before hitting the API
  const ctx = window.__brctx || {};
  // `_isSignedIn` is retained as a fallback for an older cached page shell.
  // New shells provide __brctx, including the provider needed by checkout.
  if (!(ctx.is_logged_in || window._isSignedIn)) {
    const navModal = document.getElementById('signinModal');
    if (navModal) {
      if (window.brOpenSignin) window.brOpenSignin();
      else navModal.style.display = 'flex';
    } else {
      _showIdentifyModal(type, btn);
    }
    return;
  }

  const leagueId = new URLSearchParams(window.location.search).get('league_id') ||
    window.location.pathname.split('/').filter(Boolean)[2] || '';

  // Build a destination that lands in the league dashboard after payment
  const _platform = ctx.platform || 'sleeper';
  const _season   = ctx.season   || new Date().getFullYear();
  const returnUrl = leagueId
    ? `/${_platform}/${_season}/${leagueId}/dashboard?new_subscriber=1`
    : window.location.href;

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
        platform: _platform, season: _season }),
    });
    const data = await res.json();
    if (data.url) {
      window.location.href = data.url;
    } else {
      if (btn) { btn.disabled = false; btn.innerHTML = btn.dataset.origText; }
      if (_handleAlreadySubscribed(data, leagueId)) return;
      if (window.showToast) showToast(data.error || 'Could not start checkout. Make sure you are logged in.', 'error', 5000);
      else alert(data.error || 'Could not start checkout. Make sure you are logged in.');
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
 * Self-contained "enter username → subscribe" modal for guest pages
 * that don't have the nav signin modal in the DOM.
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
        showToast('PRO is already on — invite link copied for your league mates.', 'success', 5000);
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
    try { if (localStorage.getItem(key) === '1') return; } catch (e) {}

    let el = document.getElementById('leagueProShareBanner');
    if (el) el.remove();
    el = document.createElement('div');
    el.id = 'leagueProShareBanner';
    el.setAttribute('role', 'status');
    el.style.cssText = 'position:fixed;bottom:24px;right:24px;z-index:9998;max-width:320px;background:var(--card);border:1px solid var(--border);border-top:3px solid #2563eb;border-radius:14px;box-shadow:0 12px 40px rgba(0,0,0,.22);padding:16px 18px;display:flex;flex-direction:column;gap:10px;';
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

function _showIdentifyModal(planType, triggerBtn) {
  const existing = document.getElementById('_identifyModal');
  if (existing) existing.remove();

  const needsLeague = planType === 'league' || planType === 'combo';
  const next = encodeURIComponent(window.location.pathname + window.location.search);

  const modal = document.createElement('div');
  modal.id = '_identifyModal';
  modal.className = 'signin-modal-overlay';
  modal.setAttribute('role', 'dialog');
  modal.setAttribute('aria-modal', 'true');
  modal.setAttribute('aria-labelledby', '_identifyTitle');
  modal.style.display = 'flex';
  modal.innerHTML = `
    <div class="signin-modal-box">
      <h3 class="signin-modal-title" id="_identifyTitle">Sign in to subscribe</h3>
      <p class="signin-modal-sub" id="_identifySub">Continue with Google to use your account, or enter a Sleeper username.</p>
      <a class="google-continue-btn" id="_identifyGoogle" href="/auth/google?intent=login&amp;next=${next}">
        <span class="google-button-title">Continue with Google</span>
      </a>
      <div class="signin-modal-or">or</div>
      <input class="signin-modal-input" id="_identifyInput" type="text" placeholder="Sleeper username" aria-label="Sleeper username" autocomplete="username">
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

  const input = modal.querySelector('#_identifyInput');
  const submitBtn = modal.querySelector('#_identifySubmit');
  const errorEl = modal.querySelector('#_identifyError');
  const leagueWrap = modal.querySelector('#_identifyLeagueWrap');
  const leagueSel = modal.querySelector('#_identifyLeague');
  const subText = modal.querySelector('#_identifySub');
  const prevFocus = document.activeElement;

  function focusables() {
    return modal.querySelectorAll('a[href], button:not([disabled]), input:not([disabled]), select, textarea, [tabindex]:not([tabindex="-1"])');
  }
  function closeIdentify() {
    document.removeEventListener('keydown', onKey);
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

  async function doStep() {
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

      if (window.__brctx) window.__brctx.is_logged_in = true;
      identified = true;

      if (needsLeague && data.leagues && data.leagues.length > 0) {
        input.disabled = true;
        subText.textContent = 'Choose which league to subscribe for.';
        leagueSel.innerHTML = data.leagues
          .map(lg => `<option value="${lg.league_id}">${lg.name}</option>`)
          .join('');
        leagueWrap.style.display = 'block';
        submitBtn.disabled = false;
        submitBtn.textContent = 'Continue to Checkout';
        leagueSel.focus();
      } else {
        closeIdentify();
        initiatePurchase(planType, triggerBtn);
      }
    } catch (e) {
      errorEl.textContent = 'Network error. Please try again.';
      errorEl.style.display = 'block';
      submitBtn.disabled = false;
      submitBtn.textContent = 'Continue';
    }
  }

  function doCheckout() {
    const leagueId = leagueSel.value || '';
    if (window.__brctx) window.__brctx.leagueId = leagueId;
    closeIdentify();
    _initiatePurchaseWithLeague(planType, triggerBtn, leagueId);
  }

  submitBtn.addEventListener('click', doStep);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') doStep(); });
}

async function _initiatePurchaseWithLeague(type, btn, leagueId) {
  // Build a post-checkout destination: league dashboard if we have a league,
  // otherwise the current page. Append ?new_subscriber=1 to trigger the welcome tour.
  const ctx = window.__brctx || {};
  const platform = ctx.platform || 'sleeper';
  const season   = ctx.season   || new Date().getFullYear();
  const returnUrl = leagueId
    ? `/${platform}/${season}/${leagueId}/dashboard?new_subscriber=1`
    : (window.location.pathname + '?new_subscriber=1');

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
        platform, season }),
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
