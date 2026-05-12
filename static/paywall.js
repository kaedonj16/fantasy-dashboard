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

    const response = await fetch(`/api/subscription-status?${params}`);
    return await response.json();
  } catch (error) {
    console.error('[paywall] Error getting subscription info:', error);
    return { has_premium: false, subscription_type: null };
  }
}

/**
 * Show paywall modal for a specific feature
 *
 * @param {string} feature - Feature name ('breakout-candidates', 'advanced-metrics', 'ai-insights')
 */
window.showPaywall = function showPaywall(feature) {
  const featureNames = {
    'breakout-candidates': 'Breakout Engine',
    'advanced-metrics': 'Advanced Metrics',
    'ai-insights': 'AI Insights',
    'trade-history': 'Trade History',
    'trade-suggestions': 'Trade Suggestions',
    'auction-values': 'Auction Values'
  };

  const featureName = featureNames[feature] || 'Premium Feature';

  const modal = document.createElement('div');
  modal.className = 'paywall-modal';
  modal.innerHTML = `
    <div class="paywall-overlay"></div>
    <div class="paywall-content">
      <div class="paywall-header">
        <h2><i class="fa-solid fa-lock" aria-hidden="true"></i> Premium Feature</h2>
        <button class="paywall-close" onclick="this.closest('.paywall-modal').remove()">×</button>
      </div>
      <div class="paywall-body">
        <div class="paywall-icon"><i class="fa-solid fa-star" aria-hidden="true"></i></div>
        <h3>${featureName}</h3>
        <p>This is a premium feature. Upgrade to access:</p>
        <ul class="paywall-features">
          <li>✓ AI-powered trade suggestions</li>
          <li>✓ Full Trade Intelligence feed &amp; history</li>
          <li>✓ Breakout Engine candidate predictions</li>
          <li>✓ Advanced player metrics and analytics</li>
          <li>✓ All future premium features</li>
        </ul>
        <div class="paywall-pricing">
          <div class="pricing-option">
            <div class="pricing-header">
              <h4>League Plan</h4>
            </div>
            <div class="pricing-price">$10<span>/year</span></div>
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
            <div class="pricing-price">$12<span>/year</span></div>
            <p class="pricing-desc">League premium + all your personal leagues</p>
            <button class="btn btn-primary paywall-cta" onclick="initiatePurchase('combo', this)">
              Subscribe Both
            </button>
          </div>
          <div class="pricing-option">
            <div class="pricing-header">
              <h4>Personal Plan</h4>
            </div>
            <div class="pricing-price">$5<span>/year</span></div>
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

  // Close on overlay click
  modal.querySelector('.paywall-overlay').addEventListener('click', () => {
    modal.remove();
  });
}

async function initiatePurchase(type, btn) {
  // Prompt login before hitting the API
  const ctx = window.__brctx || {};
  if (!ctx.is_logged_in) {
    const navModal = document.getElementById('signinModal');
    if (navModal) {
      navModal.style.display = 'flex';
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
      body: JSON.stringify({ plan: type, league_id: leagueId, return_url: returnUrl }),
    });
    const data = await res.json();
    if (data.url) {
      window.location.href = data.url;
    } else {
      if (btn) { btn.disabled = false; btn.innerHTML = btn.dataset.origText; }
      if (_handleAlreadySubscribed(data, leagueId)) return;
      alert(data.error || 'Could not start checkout. Make sure you are logged in.');
    }
  } catch (e) {
    if (btn) { btn.disabled = false; btn.innerHTML = btn.dataset.origText; }
    alert('Checkout unavailable. Please try again.');
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
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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

  // They're already subscribed — mark premium and redirect to their league or refresh
  if (window.__brctx) window.__brctx.isPremium = true;

  const ctx = window.__brctx || {};
  const platform = ctx.platform || 'sleeper';
  const season   = ctx.season   || new Date().getFullYear();
  const lid      = leagueId || ctx.leagueId || '';

  const dest = lid
    ? `/${platform}/${season}/${lid}/dashboard`
    : window.location.pathname;

  window.location.href = dest;
  return true;
}

function _showIdentifyModal(planType, triggerBtn) {
  const existing = document.getElementById('_identifyModal');
  if (existing) existing.remove();

  const needsLeague = planType === 'league' || planType === 'combo';

  const modal = document.createElement('div');
  modal.id = '_identifyModal';
  modal.className = 'signin-modal-overlay';
  modal.style.display = 'flex';
  modal.innerHTML = `
    <div class="signin-modal-box">
      <h3 class="signin-modal-title">Sign in to subscribe</h3>
      <p class="signin-modal-sub" id="_identifySub">Enter your Sleeper username to continue.</p>
      <input class="signin-modal-input" id="_identifyInput" type="text" placeholder="Sleeper username" autocomplete="username" autofocus>
      <div id="_identifyLeagueWrap" style="display:none;margin-bottom:16px;">
        <label style="display:block;font-size:11px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px;">Select League</label>
        <select class="signin-modal-input" id="_identifyLeague" style="margin-bottom:0;cursor:pointer;"></select>
      </div>
      <div id="_identifyError" style="display:none;font-size:12px;color:#ef4444;margin:-8px 0 12px;"></div>
      <div class="signin-modal-actions">
        <button class="signin-modal-submit" id="_identifySubmit">Continue</button>
        <button class="signin-modal-cancel" onclick="document.getElementById('_identifyModal').remove()">Cancel</button>
      </div>
    </div>`;
  document.body.appendChild(modal);

  const input = modal.querySelector('#_identifyInput');
  const submitBtn = modal.querySelector('#_identifySubmit');
  const errorEl = modal.querySelector('#_identifyError');
  const leagueWrap = modal.querySelector('#_identifyLeagueWrap');
  const leagueSel = modal.querySelector('#_identifyLeague');
  const subText = modal.querySelector('#_identifySub');

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
        // Show league picker
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
        // No league needed (personal plan) or no leagues found — go straight to checkout
        modal.remove();
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
    modal.remove();
    // Pass the selected leagueId directly into the checkout payload
    _initiatePurchaseWithLeague(planType, triggerBtn, leagueId);
  }

  submitBtn.addEventListener('click', doStep);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') doStep(); });
  modal.addEventListener('click', e => { if (e.target === modal) modal.remove(); });
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
      body: JSON.stringify({ plan: type, league_id: leagueId, return_url: returnUrl }),
    });
    const data = await res.json();
    if (data.url) {
      window.location.href = data.url;
    } else {
      if (btn) { btn.disabled = false; btn.innerHTML = btn.dataset.origText; }
      if (_handleAlreadySubscribed(data, leagueId)) return;
      alert(data.error || 'Could not start checkout.');
    }
  } catch (e) {
    if (btn) { btn.disabled = false; btn.innerHTML = btn.dataset.origText; }
    alert('Checkout unavailable. Please try again.');
  }
}
