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
function showPaywall(feature) {
  const featureNames = {
    'breakout-candidates': 'Breakout Engine',
    'advanced-metrics': 'Advanced Metrics',
    'ai-insights': 'AI Insights',
    'trade-history': 'Trade History',
    'trade-suggestions': 'Trade Suggestions'
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
          <li>✓ AI-powered insights and recommendations</li>
          <li>✓ Breakout candidate predictions</li>
          <li>✓ Advanced player metrics and analytics</li>
          <li>✓ All premium features for your league</li>
        </ul>
        <div class="paywall-pricing">
          <div class="pricing-option">
            <div class="pricing-header">
              <h4>League Plan</h4>
              <div class="pricing-badge">Recommended</div>
            </div>
            <div class="pricing-price">$10<span>/month</span></div>
            <p class="pricing-desc">Premium for all managers in your league</p>
            <button class="btn btn-primary paywall-cta" onclick="initiatePurchase('league')">
              Subscribe for League
            </button>
          </div>
          <div class="pricing-option">
            <div class="pricing-header">
              <h4>Personal Plan</h4>
            </div>
            <div class="pricing-price">$5<span>/month</span></div>
            <p class="pricing-desc">Premium for all your leagues</p>
            <button class="btn btn-secondary paywall-cta" onclick="initiatePurchase('user')">
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

/**
 * Initiate purchase flow (placeholder - implement with Stripe)
 */
function initiatePurchase(type) {
  window.location.href = '/pricing?plan=' + type;
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
