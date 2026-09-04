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
        <p>This is a premium feature. A Google account is required to subscribe.</p>
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
              <h4>One League</h4>
            </div>
            <div class="pricing-price">$5<span>/year</span></div>
            <p class="pricing-desc">PRO for you in one league you choose</p>
            <button class="btn btn-secondary paywall-cta" onclick="initiatePurchase('single_league', this)">
              Choose a League
            </button>
          </div>
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
  const leagueId = extra.leagueId || _checkoutLeagueId();
  const needsLeague = planType === 'league' || planType === 'combo' || planType === 'single_league';
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
  // Checkout requires a Google account site-wide. Sleeper-only sign-in is
  // enough to view a league, not to subscribe. Guests who pick a league plan
  // without a league open the platform picker above the paywall, then Google.
  const ctx = window.__brctx || {};
  const leagueId = new URLSearchParams(window.location.search).get('league_id') ||
    window.location.pathname.split('/').filter(Boolean)[2] ||
    (ctx.leagueId || '');
  const needsLeague = type === 'league' || type === 'combo' || type === 'single_league';

  if (!_hasGoogleAccount()) {
    if (needsLeague && !leagueId) {
      _openCheckoutLeaguePicker(type, btn);
      return;
    }
    if (needsLeague && leagueId) {
      _startGoogleSubscribe(type, btn);
      return;
    }
    _showIdentifyModal(type, btn);
    return;
  }

  if (needsLeague && !leagueId) {
    _showLeaguePickerModal(type, btn);
    return;
  }

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

const _CHECKOUT_PLANS = { single_league: 1, league: 1, combo: 1, user: 1 };

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

  fetch('/api/my-leagues', { cache: 'no-store' })
    .then(r => r.json())
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

  const needsLeague = planType === 'league' || planType === 'combo' || planType === 'single_league';
  const next = encodeURIComponent(window.location.pathname + window.location.search);
  const yahooOn = !!document.querySelector('#linkModal .link-tab[data-lp="yahoo"]');
  const googleCtl = needsLeague
    ? `<button type="button" class="google-continue-btn" id="_identifyGoogle"><span class="google-button-title">Continue with Google</span></button>`
    : `<a class="google-continue-btn" id="_identifyGoogle" href="/auth/google?intent=login&amp;next=${next}"><span class="google-button-title">Continue with Google</span></a>`;
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
  const PLAN_LABELS = {
    single_league: 'One League · $5/year',
    league: 'League · $15/year',
    combo: 'League + Personal · $20/year',
    user: 'Personal · $10/year',
  };
  const NEEDS_SEASON = { mfl: true, fleaflicker: true };
  const year = (window.__brctx && window.__brctx.season) || new Date().getFullYear();

  document.querySelectorAll('.paywall-modal').forEach(function (el) { el.remove(); });

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
        <ol class="home-pro-progress" aria-label="PRO signup steps">
          <li class="is-active" data-home-pro-step="plan">1. Plan</li>
          <li data-home-pro-step="league">2. League</li>
        </ol>
        <div id="homeProStepPlan" class="home-pro-step">
          <h3>Choose a plan</h3>
          <p>Then enter your league. A Google account is required to subscribe.</p>
          <div class="paywall-pricing">
            <div class="pricing-option">
              <div class="pricing-header"><h4>One League</h4></div>
              <div class="pricing-price">$5<span>/year</span></div>
              <p class="pricing-desc">PRO for you in one league you choose</p>
              <button type="button" class="btn btn-secondary paywall-cta" data-plan="single_league">Choose this plan</button>
            </div>
            <div class="pricing-option">
              <div class="pricing-header"><h4>League Plan</h4></div>
              <div class="pricing-price">$15<span>/year</span></div>
              <p class="pricing-desc">Premium for every manager in your league</p>
              <button type="button" class="btn btn-secondary paywall-cta" data-plan="league">Choose this plan</button>
            </div>
            <div class="pricing-option featured">
              <div class="pricing-header">
                <h4>League + Personal</h4>
                <div class="pricing-badge">Best value</div>
              </div>
              <div class="pricing-price">$20<span>/year</span></div>
              <p class="pricing-desc">Your league plus all your personal leagues</p>
              <button type="button" class="btn btn-primary paywall-cta" data-plan="combo">Choose this plan</button>
            </div>
            <div class="pricing-option">
              <div class="pricing-header"><h4>Personal Plan</h4></div>
              <div class="pricing-price">$10<span>/year</span></div>
              <p class="pricing-desc">Premium for all your leagues</p>
              <button type="button" class="btn btn-secondary paywall-cta" data-plan="user">Choose this plan</button>
            </div>
          </div>
        </div>
        <div id="homeProStepLeague" class="home-pro-step" hidden>
          <button type="button" id="homeProBack" class="home-pro-back">Change plan</button>
          <p class="home-pro-picked">Selected: <strong id="homeProPickedLabel"></strong></p>
          <div id="homeProSavedWrap" hidden>
            <label for="homeProSavedSelect">Your saved leagues</label>
            <select id="homeProSavedSelect"><option value="">Choose a saved league</option></select>
            <p class="home-pro-or">or connect a different league</p>
          </div>
          <div class="home-pro-platforms" role="radiogroup" aria-label="League platform">
            <button type="button" class="home-pro-platform is-active" data-platform="sleeper" aria-pressed="true">Sleeper</button>
            <button type="button" class="home-pro-platform" data-platform="espn" aria-pressed="false">ESPN</button>
            <button type="button" class="home-pro-platform" data-platform="yahoo" aria-pressed="false">Yahoo</button>
            <button type="button" class="home-pro-platform" data-platform="mfl" aria-pressed="false">MFL</button>
            <button type="button" class="home-pro-platform" data-platform="fleaflicker" aria-pressed="false">Fleaflicker</button>
          </div>
          <div id="homeProSleeperFields" class="home-pro-fields">
            <label for="homeProSleeperUser">Sleeper username</label>
            <div class="home-pro-inline">
              <input type="text" id="homeProSleeperUser" autocomplete="username" placeholder="Your Sleeper username">
              <button type="button" id="homeProFindLeagues">Find leagues</button>
            </div>
            <div id="homeProSleeperLeagueWrap" hidden>
              <label for="homeProSleeperLeague">Choose league</label>
              <select id="homeProSleeperLeague"><option value="">Select a league</option></select>
            </div>
          </div>
          <div id="homeProIdFields" class="home-pro-fields" hidden>
            <label for="homeProLeagueId">League ID</label>
            <input type="text" id="homeProLeagueId" autocomplete="off" placeholder="From your league URL">
            <div id="homeProSeasonWrap" hidden>
              <label for="homeProSeason">Season</label>
              <input type="text" id="homeProSeason" inputmode="numeric" value="${year}">
            </div>
          </div>
          <p id="homeProError" class="home-pro-error" hidden></p>
          <div class="home-pro-actions">
            <button type="button" id="homeProGoogle" class="google-continue-btn">
              <span class="google-button-title">Continue with Google</span>
              <span>Creates your account and opens secure checkout</span>
            </button>
            <button type="button" id="homeProCheckout" class="home-pro-checkout-btn" hidden>Continue to checkout</button>
          </div>
        </div>
      </div>
    </div>`;
  document.body.appendChild(modal);

  const inertRoot = document.getElementById('app-scale') || document.getElementById('page-root');
  if (inertRoot) inertRoot.setAttribute('inert', '');
  const prevFocus = document.activeElement;
  const stepPlan = modal.querySelector('#homeProStepPlan');
  const stepLeague = modal.querySelector('#homeProStepLeague');
  const pickedLabel = modal.querySelector('#homeProPickedLabel');
  const errorEl = modal.querySelector('#homeProError');
  const savedWrap = modal.querySelector('#homeProSavedWrap');
  const savedSelect = modal.querySelector('#homeProSavedSelect');
  const sleeperFields = modal.querySelector('#homeProSleeperFields');
  const idFields = modal.querySelector('#homeProIdFields');
  const seasonWrap = modal.querySelector('#homeProSeasonWrap');
  const sleeperUser = modal.querySelector('#homeProSleeperUser');
  const sleeperLeagueWrap = modal.querySelector('#homeProSleeperLeagueWrap');
  const sleeperLeague = modal.querySelector('#homeProSleeperLeague');
  const leagueIdInput = modal.querySelector('#homeProLeagueId');
  const seasonInput = modal.querySelector('#homeProSeason');
  const findBtn = modal.querySelector('#homeProFindLeagues');
  const googleBtn = modal.querySelector('#homeProGoogle');
  const checkoutBtn = modal.querySelector('#homeProCheckout');
  const progressItems = modal.querySelectorAll('[data-home-pro-step]');
  let selectedPlan = '';
  let selectedPlatform = 'sleeper';

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

  function currentSeason() {
    const raw = (seasonInput && seasonInput.value) || year;
    return parseInt(raw, 10) || new Date().getFullYear();
  }
  function showError(msg) {
    if (!errorEl) return;
    errorEl.hidden = !msg;
    errorEl.textContent = msg || '';
  }
  function setStep(step) {
    const league = step === 'league';
    if (stepPlan) stepPlan.hidden = league;
    if (stepLeague) stepLeague.hidden = !league;
    progressItems.forEach(function (item) {
      item.classList.toggle('is-active', item.getAttribute('data-home-pro-step') === step);
    });
    showError('');
  }
  function setPlatform(platform) {
    selectedPlatform = platform;
    modal.querySelectorAll('.home-pro-platform').forEach(function (btn) {
      const on = btn.getAttribute('data-platform') === platform;
      btn.classList.toggle('is-active', on);
      btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
    const sleeper = platform === 'sleeper';
    if (sleeperFields) sleeperFields.hidden = !sleeper;
    if (idFields) idFields.hidden = sleeper;
    if (seasonWrap) seasonWrap.hidden = !NEEDS_SEASON[platform];
  }
  function collectPayload() {
    const savedVal = savedSelect && savedSelect.value ? savedSelect.value : '';
    if (savedVal) {
      const opt = savedSelect.options[savedSelect.selectedIndex];
      return {
        plan: selectedPlan,
        platform: opt.getAttribute('data-platform') || 'sleeper',
        league_id: savedVal,
        season: parseInt(opt.getAttribute('data-season') || '', 10) || currentSeason(),
        name: (opt.textContent || '').trim() || null,
        username: (sleeperUser && sleeperUser.value || '').trim() || null,
      };
    }
    if (selectedPlatform === 'sleeper') {
      const opt = sleeperLeague && sleeperLeague.options[sleeperLeague.selectedIndex];
      return {
        plan: selectedPlan,
        platform: 'sleeper',
        league_id: (sleeperLeague && sleeperLeague.value || '').trim(),
        season: currentSeason(),
        name: opt && opt.textContent ? opt.textContent.trim() : null,
        username: (sleeperUser && sleeperUser.value || '').trim() || null,
      };
    }
    return {
      plan: selectedPlan,
      platform: selectedPlatform,
      league_id: (leagueIdInput && leagueIdInput.value || '').trim(),
      season: currentSeason(),
      username: null,
      name: null,
    };
  }
  function validatePayload(payload) {
    if (!payload.plan) return 'Pick a plan to continue.';
    if (!payload.league_id) return 'Enter your league info to continue.';
    if (payload.platform === 'sleeper' && !payload.username && !(savedSelect && savedSelect.value)) {
      return 'Enter your Sleeper username and choose a league.';
    }
    return '';
  }
  function loadSavedLeagues() {
    if (!window._hasAccount || !savedWrap || !savedSelect) return;
    fetch('/api/my-leagues', { cache: 'no-store' })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        const leagues = (data && data.leagues) || [];
        if (!leagues.length) return;
        savedSelect.innerHTML = '<option value="">Choose a saved league</option>' + leagues.map(function (lg) {
          const id = lg.league_id || lg.id || '';
          const name = lg.name || lg.league_name || id;
          const plat = lg.platform || 'sleeper';
          const season = lg.season || '';
          const label = season ? (name + ' (' + plat + ' · ' + season + ')') : (name + ' (' + plat + ')');
          return '<option value="' + String(id).replace(/"/g, '') + '" data-platform="' + plat + '" data-season="' + season + '">' + label + '</option>';
        }).join('');
        savedWrap.hidden = false;
      })
      .catch(function () {});
  }

  modal.querySelectorAll('[data-plan]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      selectedPlan = btn.getAttribute('data-plan') || '';
      if (pickedLabel) pickedLabel.textContent = PLAN_LABELS[selectedPlan] || selectedPlan;
      if (checkoutBtn) checkoutBtn.hidden = !window._hasAccount;
      setStep('league');
      loadSavedLeagues();
    });
  });
  modal.querySelector('#homeProBack').addEventListener('click', function () { setStep('plan'); });
  modal.querySelectorAll('.home-pro-platform').forEach(function (btn) {
    btn.addEventListener('click', function () {
      setPlatform(btn.getAttribute('data-platform') || 'sleeper');
    });
  });
  findBtn.addEventListener('click', async function () {
    const username = (sleeperUser && sleeperUser.value || '').trim();
    if (!username) { showError('Enter a Sleeper username.'); return; }
    showError('');
    findBtn.disabled = true;
    findBtn.textContent = 'Loading...';
    try {
      const res = await fetch('/api/sleeper-user-leagues?username=' + encodeURIComponent(username));
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || 'Unable to load leagues.');
      const leagues = data.leagues || [];
      if (!leagues.length) throw new Error('No leagues found for that username.');
      sleeperLeague.innerHTML = '<option value="">Select a league</option>' + leagues.map(function (lg) {
        const id = lg.league_id || lg.id || '';
        return '<option value="' + String(id).replace(/"/g, '') + '">' + (lg.name || 'League') + '</option>';
      }).join('');
      sleeperLeagueWrap.hidden = false;
    } catch (err) {
      showError(err.message || 'Unable to load leagues.');
      sleeperLeagueWrap.hidden = true;
    } finally {
      findBtn.disabled = false;
      findBtn.textContent = 'Find leagues';
    }
  });
  googleBtn.addEventListener('click', async function () {
    const payload = collectPayload();
    const invalid = validatePayload(payload);
    if (invalid) { showError(invalid); return; }
    showError('');
    googleBtn.disabled = true;
    try {
      const res = await fetch('/api/pro-signup/pending', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || 'Could not save this signup.');
      window.location.href = data.auth_url || '/auth/google?intent=onboarding&next=/pro/resume-checkout';
    } catch (err) {
      showError(err.message || 'Unable to continue with Google.');
      googleBtn.disabled = false;
    }
  });
  checkoutBtn.addEventListener('click', async function () {
    const payload = collectPayload();
    const invalid = validatePayload(payload);
    if (invalid) { showError(invalid); return; }
    showError('');
    if (window.__brctx) {
      window.__brctx.platform = payload.platform;
      window.__brctx.season = payload.season;
      window.__brctx.leagueId = payload.league_id;
    }
    if (!_hasGoogleAccount()) {
      _startGoogleSubscribe(payload.plan, checkoutBtn, {
        leagueId: payload.league_id,
        platform: payload.platform,
        season: payload.season,
        username: payload.username,
      });
      return;
    }
    if (typeof _initiatePurchaseWithLeague === 'function') {
      _initiatePurchaseWithLeague(payload.plan, checkoutBtn, payload.league_id);
    }
  });

  setPlatform('sleeper');
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

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initHomeProSignup);
} else {
  initHomeProSignup();
}
