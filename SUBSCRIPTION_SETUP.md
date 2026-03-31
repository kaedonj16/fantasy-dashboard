# Subscription System Setup

This guide explains how to set up and use the league-based subscription system.

## Features

**Free Tier:**
- Basic stats and standings
- Matchup previews
- Trade calculator
- Player search
- League activity
- Most dashboard features

**Premium Tier ($19.99/year personal, $29.99/year league):**
- 🔒 AI-powered insights and recommendations
- 🔒 Breakout candidate predictions
- 🔒 Advanced player metrics (snap share, opportunity share, efficiency trends)

## Subscription Types

### 1. League Subscription ($29.99/year)
- **Best value** - entire league gets premium access
- Any user who paid can subscribe for their league
- All league members automatically get premium features
- Perfect for commissioners or active league members

### 2. Personal Subscription ($19.99/year)
- User gets premium access across ALL their leagues
- Good for users in multiple leagues
- Individual access only

## Setup Instructions

### 1. Run Database Migrations

```bash
python scripts/run_migrations.py
```

This creates the subscription tables:
- `league_subscriptions`
- `user_subscriptions`

### 2. Add Paywall UI to Frontend

Add to your HTML `<head>`:

```html
<link rel="stylesheet" href="/static/paywall.css">
<script src="/static/paywall.js"></script>
```

### 3. Protect Premium Features

The following endpoints are already protected:
- `/api/player-advanced-metrics/<player_id>` - Advanced player metrics
- `/api/offseason-breakout-candidates` - Breakout predictions
- (AI insights endpoint - to be added)

### 4. Frontend Usage

#### Check Premium Access

```javascript
const userId = 'john_doe';  // Sleeper username
const leagueId = '123456789';

const hasPremium = await checkPremiumAccess(userId, leagueId);
if (hasPremium) {
  // Load premium feature
} else {
  showPaywall('advanced-metrics');
}
```

#### Protect a Feature

```javascript
// Show paywall if user doesn't have premium
protectFeature('breakout-candidates', userId, leagueId, () => {
  // This callback only runs if user has premium
  loadBreakoutCandidates();
});
```

#### Add Premium Badge

```javascript
const heading = document.querySelector('h2');
addPremiumBadge(heading);  // Adds ⭐ Premium badge
```

## API Endpoints

### Check Subscription Status
```
GET /api/subscription-status?user_id=<user>&league_id=<league>

Response:
{
  "has_premium": true,
  "subscription_type": "league",  // or "user" or null
  "expires_at": "2026-03-31T00:00:00Z",
  "subscriber_user_id": "john_doe"  // Who paid (league subs only)
}
```

### Protected Endpoints

All return 403 error if no premium access:

```json
{
  "premium_required": true,
  "error": "Premium subscription required to view <feature>"
}
```

## Database Management

### Create Subscription (Python)

```python
from dashboard_services.subscriptions import create_league_subscription, create_user_subscription
from datetime import datetime, timezone, timedelta

# League subscription
expires_at = datetime.now(timezone.utc) + timedelta(days=365)
create_league_subscription(
    league_id="123456789",
    subscriber_user_id="john_doe",
    expires_at=expires_at,
    stripe_subscription_id="sub_abc123"
)

# User subscription
create_user_subscription(
    user_id="jane_smith",
    expires_at=expires_at,
    stripe_subscription_id="sub_def456"
)
```

### Check Access (Python)

```python
from dashboard_services.subscriptions import has_premium_access, get_subscription_info

# Check access
has_access = has_premium_access("john_doe", "123456789")

# Get details
info = get_subscription_info("john_doe", "123456789")
print(info)
# {
#   "has_premium": True,
#   "subscription_type": "league",
#   "expires_at": "2026-03-31T...",
#   "subscriber_user_id": "john_doe"
# }
```

## Payment Integration (TODO)

The system is ready for Stripe integration:

1. Set up Stripe account and get API keys
2. Create Stripe Products for:
   - League Plan: $29.99/year
   - Personal Plan: $19.99/year
3. Implement checkout flow in `initiatePurchase()` function (paywall.js)
4. Create webhook handler for Stripe events:
   - `checkout.session.completed` → create subscription
   - `customer.subscription.updated` → update subscription
   - `customer.subscription.deleted` → cancel subscription

### Stripe Webhook Example

```python
@app.route("/api/stripe-webhook", methods=["POST"])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )

        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']

            # Extract metadata
            subscription_type = session['metadata']['type']  # 'league' or 'user'

            if subscription_type == 'league':
                create_league_subscription(
                    league_id=session['metadata']['league_id'],
                    subscriber_user_id=session['metadata']['user_id'],
                    expires_at=...,
                    stripe_subscription_id=session['subscription']
                )
            else:
                create_user_subscription(
                    user_id=session['metadata']['user_id'],
                    expires_at=...,
                    stripe_subscription_id=session['subscription']
                )

        return jsonify(success=True)

    except Exception as e:
        return jsonify(error=str(e)), 400
```

## Testing

### Manually Grant Access

```sql
-- Grant league premium access
INSERT INTO league_subscriptions (
    league_id, platform, subscriber_user_id,
    subscription_status, expires_at
) VALUES (
    '123456789', 'sleeper', 'test_user',
    'active', NOW() + INTERVAL '1 year'
);

-- Grant user premium access
INSERT INTO user_subscriptions (
    user_id, platform, subscription_status, expires_at
) VALUES (
    'test_user', 'sleeper', 'active', NOW() + INTERVAL '1 year'
);
```

### Remove Test Access

```sql
DELETE FROM league_subscriptions WHERE league_id = '123456789';
DELETE FROM user_subscriptions WHERE user_id = 'test_user';
```

## UI Customization

Edit `static/paywall.css` to match your brand colors:

```css
/* Change gradient */
.btn-primary {
  background: linear-gradient(135deg, #your-color-1 0%, #your-color-2 100%);
}

/* Change prices in paywall.js */
const modal = ...
  <div class="pricing-price">$XX.XX<span>/year</span></div>
```

## Troubleshooting

### "Premium subscription required" but user has subscription

1. Check subscription exists in database
2. Verify `expires_at` is in the future
3. Verify `subscription_status = 'active'`
4. Check `user_id` matches exactly (case-sensitive)
5. Check `league_id` matches exactly

### Paywall not showing

1. Ensure `paywall.css` and `paywall.js` are loaded
2. Check browser console for errors
3. Verify `showPaywall()` is being called

### Database connection issues

1. Verify `DATABASE_URL` environment variable is set
2. Check PostgreSQL is running
3. Verify credentials in connection string
