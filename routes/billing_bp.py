"""
Billing / subscription routes.

Routes: /pricing, /api/create-checkout-session, /api/stripe-webhook,
        /api/subscription-status, /api/pro-signup/pending, /pro/resume-checkout
Also handles: /<platform>/<season>/<league_id>/pricing
"""
from __future__ import annotations

import html
import json
import logging
import os
import urllib.parse
from datetime import datetime, timedelta, timezone

from flask import Blueprint, jsonify, request, session

from dashboard_services.subscriptions import (
    cancel_subscription,
    create_league_subscription,
    create_user_league_subscription,
    create_user_subscription,
    has_premium_access,
    has_premium_for_viewer,
    has_user_league_subscription,
)

billing_bp = Blueprint("billing", __name__)
logger = logging.getLogger(__name__)

_STRIPE_LEAGUE_PRODUCT = "prod_USjDJYPhNGnmvM"
_STRIPE_USER_PRODUCT   = "prod_USjDRuVDcwH1xb"
_STRIPE_COMBO_PRODUCT  = "prod_UT5DaCA4u6hWgb"
_STRIPE_SINGLE_LEAGUE_PRODUCT = (
    os.environ.get("STRIPE_SINGLE_LEAGUE_PRODUCT", "").strip()
    or "prod_VB4fRiQaKCTcu9"
)


def _stripe():
    """Lazy-import stripe so missing package doesn't break the whole blueprint."""
    import stripe as _s
    _s.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
    return _s


def _stripe_field(obj, key, default=None):
    """Read a field from a Stripe object or a plain dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, "get"):
        try:
            return obj.get(key, default)
        except Exception:
            pass
    return getattr(obj, key, default)


def _stripe_id(value) -> str:
    """Coerce a Stripe id that may arrive as a string or expanded object."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(_stripe_field(value, "id", "") or "")


def _metadata_dict(obj) -> dict:
    """Normalize Stripe metadata to a plain str→str dict.

    ``dict(stripe_object.metadata)`` is unreliable across stripe-python
    versions; Checkout success already uses ``to_dict()``. The webhook must
    do the same or ``plan`` / ``league_id`` come through empty and One League
    never writes ``user_league_subscriptions``.
    """
    raw = _stripe_field(obj, "metadata", None)
    if not raw:
        return {}
    if hasattr(raw, "to_dict"):
        try:
            raw = raw.to_dict()
        except Exception:
            pass
    if not isinstance(raw, dict):
        try:
            raw = dict(raw)
        except Exception:
            return {}
    out = {}
    for key, value in raw.items():
        if key is None:
            continue
        out[str(key)] = "" if value is None else str(value)
    return out


def _product_plan_map() -> dict:
    return {
        _STRIPE_LEAGUE_PRODUCT: "league",
        _STRIPE_USER_PRODUCT: "user",
        _STRIPE_COMBO_PRODUCT: "combo",
        _STRIPE_SINGLE_LEAGUE_PRODUCT: "single_league",
    }


def _plan_from_subscription(sub) -> str:
    """Infer plan from the Stripe product on a subscription's items."""
    if not sub:
        return ""
    items = _stripe_field(sub, "items")
    data = _stripe_field(items, "data") or []
    product_map = _product_plan_map()
    for item in data:
        price = _stripe_field(item, "price") or {}
        product = _stripe_field(price, "product")
        product_id = _stripe_id(product)
        plan = product_map.get(product_id)
        if plan:
            return plan
    return ""


def _subscriber_user_id(meta: dict) -> str:
    """Prefer checkout ``user_id``; fall back to ``acct:<account_id>``."""
    user_id = str(meta.get("user_id") or "").strip()
    if user_id:
        return user_id
    account_id = str(meta.get("account_id") or "").strip()
    if not account_id:
        return ""
    return account_id if account_id.startswith("acct:") else f"acct:{account_id}"


def _checkout_metadata(plan: str, user_id: str, league_id: str, platform: str, season: int) -> dict:
    return {
        "plan": plan,
        "user_id": user_id,
        "league_id": league_id,
        "platform": platform,
        "season": str(season),
        "account_id": str(session.get("account_id") or ""),
    }


def _apply_plan_grant(
    plan: str,
    user_id: str,
    league_id: str,
    platform: str,
    expires_at,
    sub_id: str,
    cust_id: str,
    *,
    source: str,
) -> None:
    """Write the entitlement row(s) for a paid plan. Idempotent upserts."""
    plan = (plan or "").strip()
    user_id = (user_id or "").strip()
    league_id = (league_id or "").strip()
    platform = (platform or "sleeper").strip() or "sleeper"
    sub_id = _stripe_id(sub_id) or None
    cust_id = _stripe_id(cust_id) or None
    if not plan:
        logger.info(
            "[stripe] %s missing plan metadata user=%s league=%s",
            source, user_id, league_id,
        )
        return

    if plan in ("league", "combo") and league_id:
        ok = create_league_subscription(
            league_id, user_id or "", expires_at,
            stripe_subscription_id=sub_id,
            stripe_customer_id=cust_id,
            platform=platform,
        )
        logger.info(
            "[stripe] %s league subscription %s for league=%s user=%s expires=%s",
            source, "created" if ok else "FAILED", league_id, user_id, expires_at,
        )
    if plan in ("user", "combo") and user_id:
        ok = create_user_subscription(
            user_id, expires_at,
            stripe_subscription_id=sub_id,
            stripe_customer_id=cust_id,
            platform=platform,
        )
        logger.info(
            "[stripe] %s user subscription %s for user=%s expires=%s",
            source, "created" if ok else "FAILED", user_id, expires_at,
        )
    if plan == "single_league":
        if user_id and league_id:
            ok = create_user_league_subscription(
                user_id, league_id, expires_at,
                stripe_subscription_id=sub_id,
                stripe_customer_id=cust_id,
                platform=platform,
            )
            logger.info(
                "[stripe] %s single-league subscription %s for user=%s league=%s expires=%s",
                source, "created" if ok else "FAILED", user_id, league_id, expires_at,
            )
        else:
            logger.warning(
                "[stripe] %s single_league grant skipped: user_id=%r league_id=%r",
                source, user_id, league_id,
            )
    elif plan not in ("league", "user", "combo"):
        logger.warning(
            "[stripe] %s unhandled plan=%s league=%s user=%s",
            source, plan, league_id, user_id,
        )


_STRIPE_PRICES = {
    "league": {"unit_amount": 1500, "product": _STRIPE_LEAGUE_PRODUCT},
    "user":   {"unit_amount": 1000, "product": _STRIPE_USER_PRODUCT},
    "combo":  {"unit_amount": 2000, "product": _STRIPE_COMBO_PRODUCT},
    "single_league": {
        "unit_amount": 500,
        "product": _STRIPE_SINGLE_LEAGUE_PRODUCT,
        "product_name": "BR Fantasy Single League PRO",
    },
}

_LEAGUE_REQUIRED_PLANS = frozenset({"league", "combo", "single_league"})
_MEMBERSHIP_REQUIRED_PLANS = frozenset({"league", "combo", "single_league"})

_SUPPORTED_PLATFORMS = {"sleeper", "espn", "yahoo", "mfl", "fleaflicker"}


def _request_platform(payload=None) -> str:
    """Resolve the provider without silently turning an ESPN flow into Sleeper."""
    payload = payload if isinstance(payload, dict) else {}
    return str(
        payload.get("platform") or request.values.get("platform")
        or session.get("viewer_platform") or session.get("last_platform")
        or "sleeper"
    ).strip().lower()


def _safe_local_url(value: str, fallback: str) -> str:
    """Allow same-site absolute/local redirects, rejecting protocol-relative URLs."""
    from utils.safe_url import safe_local_url
    return safe_local_url(value, fallback, host_url=request.host_url)


def _checkout_user_id() -> str | None:
    """Stable checkout identity: provider viewer, else Google account."""
    return (
        session.get("viewer_user_id")
        or session.get("viewer_username")
        or (("acct:" + str(session.get("account_id")).strip()) if session.get("account_id") else None)
    )


def pending_checkout_resume_path() -> str | None:
    """Return the resume URL when a home-page PRO signup is staged."""
    pending = session.get("pending_checkout")
    if isinstance(pending, dict) and pending.get("plan"):
        return "/pro/resume-checkout"
    return None


def _normalize_pending_checkout(data: dict) -> tuple[dict | None, str | None]:
    """Validate a home PRO signup payload. Returns (pending, error)."""
    plan = str((data or {}).get("plan") or "").strip()
    if plan not in _STRIPE_PRICES:
        return None, "Pick a plan to continue."
    platform = _request_platform(data)
    if platform not in _SUPPORTED_PLATFORMS:
        return None, "Choose a supported platform."
    league_id = str((data or {}).get("league_id") or "").strip()
    if plan in _LEAGUE_REQUIRED_PLANS and not league_id:
        return None, "Enter your league info to continue."
    try:
        season = int((data or {}).get("season") or datetime.now().year)
    except (TypeError, ValueError):
        return None, "Invalid season."
    username = str((data or {}).get("username") or "").strip() or None
    team_id = str((data or {}).get("team_id") or "").strip() or None
    name = str((data or {}).get("name") or "").strip() or None
    return {
        "plan": plan,
        "platform": platform,
        "league_id": league_id,
        "season": season,
        "username": username,
        "team_id": team_id,
        "name": name,
    }, None


def _require_google_to_subscribe():
    """New subscriptions require a Google account. Sleeper-only identity is not enough."""
    if session.get("account_id"):
        return None
    return jsonify({"error": "Sign in with Google to subscribe."}), 401


# ── Helpers ───────────────────────────────────────────────────────────────────

def _subscription_period_end(sub) -> "datetime":
    """Return the subscription's current period end as a UTC datetime.

    Stripe API 2025-03-31+ (stripe-python v15) moved ``current_period_end`` off
    the Subscription object onto each subscription *item*. This reads whichever
    location is present and falls back to a 32-day grant if neither is found.
    """
    ts = getattr(sub, "current_period_end", None)
    if not ts:
        try:
            items = (sub.get("items") if hasattr(sub, "get") else sub["items"])
            data = items["data"] if items else []
            ts = max((it.get("current_period_end") or 0) for it in data) or None
        except Exception:
            ts = None
    if ts:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    return datetime.now(timezone.utc) + timedelta(days=32)


def _try_grant_from_stripe_success() -> None:
    """
    When a user returns from Stripe checkout, verify the session server-side
    and grant the subscription immediately. This is a reliable fallback for
    when the webhook is delayed or misconfigured.
    """
    if request.args.get("success") != "1":
        return
    checkout_session_id = request.args.get("session_id", "").strip()
    if not checkout_session_id:
        return
    try:
        cs = _stripe().checkout.Session.retrieve(checkout_session_id)
        if cs.status != "complete":
            return

        meta      = _metadata_dict(cs)
        plan      = (meta.get("plan") or "").strip()
        user_id   = _subscriber_user_id(meta)
        platform  = meta.get("platform") or "sleeper"
        league_id = meta.get("league_id") or ""
        sub_id    = _stripe_id(_stripe_field(cs, "subscription"))
        cust_id   = _stripe_id(_stripe_field(cs, "customer"))

        try:
            sub        = _stripe().Subscription.retrieve(sub_id) if sub_id else None
            expires_at = (
                _subscription_period_end(sub)
                if sub else datetime.now(timezone.utc) + timedelta(days=366)
            )
            if plan not in ("league", "user", "combo", "single_league"):
                plan = _plan_from_subscription(sub)
        except Exception:
            expires_at = datetime.now(timezone.utc) + timedelta(days=366)
            sub = None

        if plan not in ("league", "user", "combo", "single_league"):
            return
        if plan == "user" and not user_id:
            return
        if plan in ("league", "single_league") and not league_id:
            return
        if plan == "combo" and not league_id and not user_id:
            return

        _apply_plan_grant(
            plan, user_id, league_id, platform, expires_at, sub_id, cust_id,
            source="success-page",
        )
    except Exception:
        logger.exception("[stripe] success-page session verification failed")


def _pricing_feature_item(icon: str, label: str, *, free: bool = False) -> str:
    tier = "pricing-feature-free" if free else "pricing-feature-pro"
    return (
        f'<div class="pricing-feature-item {tier}">'
        f'<span class="pricing-feature-icon" aria-hidden="true"><i class="fa-solid {html.escape(icon)}"></i></span>'
        f'<span class="pricing-feature-label">{label}</span>'
        f"</div>"
    )


def _pricing_features_grid(items: list[tuple[str, str]], *, free: bool = False) -> str:
    cells = "".join(_pricing_feature_item(icon, label, free=free) for icon, label in items)
    return f'<div class="pricing-features-grid">{cells}</div>'


_PRO_FEATURES = [
    ("fa-handshake", "Roster-Based Trade Suggestions"),
    ("fa-chart-line", "Full Trade Intelligence feed &amp; history"),
    ("fa-fire", "Breakout Engine candidate predictions"),
    ("fa-trophy", "Playoff Impact simulations"),
    ("fa-briefcase", "Front Office Report"),
    ("fa-newspaper", "Weekly Recap"),
    ("fa-clipboard-list", "Custom Draft Board"),
    ("fa-magnifying-glass-chart", "Draft Deep Dive Analyzer"),
]

_FREE_FEATURES = [
    ("fa-calculator", "Trade calculator &amp; Sleeper comps"),
    ("fa-table", "Advanced Metrics"),
    ("fa-gavel", "Auction Values"),
    ("fa-file-csv", "Live cheat-sheet overlay &amp; CSV"),
]


def _pricing_body() -> str:
    from flask import session as _session
    plan      = request.args.get("plan", "")
    success   = request.args.get("success") == "1"
    canceled  = request.args.get("canceled") == "1"
    return_to = request.args.get("return_to", "").strip()

    if success:
        # Build a proper destination from Stripe session metadata if return_to is missing
        session_id = request.args.get("session_id", "").strip()
        if not return_to and session_id:
            try:
                from datetime import datetime as _dt
                cs   = _stripe().checkout.Session.retrieve(session_id)
                meta = cs.metadata.to_dict() if cs.metadata else {}
                league_id_meta = meta.get("league_id", "")
                if league_id_meta:
                    season = int(meta.get("season") or _dt.now().year)
                    platform = meta.get("platform") or "sleeper"
                    return_to = f"/{platform}/{season}/{league_id_meta}/dashboard?new_subscriber=1&welcome=personal"
            except Exception:
                logger.debug("suppressed exception", exc_info=True)

        # Sanitize before embedding in HTML/JS — checkout already filters return_url,
        # but a crafted /pricing?success=1&return_to=https://evil link must not redirect.
        return_to = _safe_local_url(return_to, "/pricing")
        safe_return = html.escape(return_to) if return_to else ""
        viewer_user_id = (
            _session.get("viewer_user_id")
            or _session.get("viewer_username")
            or (("acct:" + str(_session.get("account_id")).strip()) if _session.get("account_id") else "")
        )
        return f"""
    <div class="card central" style="max-width:560px;text-align:center;">
      <div class="card-body" style="padding:48px 32px;">
        <div id="sub-icon" style="font-size:56px;margin-bottom:20px;">
          <i class="fa-solid fa-circle-check" style="color:#22c55e;"></i>
        </div>
        <h2 id="sub-heading" style="margin:0 0 10px;font-size:24px;">Payment confirmed!</h2>
        <p id="sub-msg" style="color:var(--text-muted);margin:0 0 28px;">
          Activating your premium access&hellip;
        </p>
        <div id="sub-spinner" style="margin:0 auto 16px;width:32px;height:32px;border:3px solid #e5e7eb;border-top-color:#2563eb;border-radius:50%;animation:paywall-spin .8s linear infinite;"></div>
        <div id="sub-invite" style="display:none;text-align:left;margin:0 0 20px;padding:16px;border:1px solid var(--border);border-radius:12px;background:var(--bg-alt, #f8fafc);">
          <div style="font-size:14px;font-weight:700;margin-bottom:6px;">PRO is on for your league</div>
          <p style="margin:0 0 12px;font-size:13px;color:var(--text-muted);line-height:1.5;">
            Share this link so every manager can sign in and unlock the same tools.
          </p>
          <div style="display:flex;gap:8px;align-items:stretch;">
            <input id="sub-invite-url" type="text" readonly
              style="flex:1;min-width:0;padding:10px 12px;border-radius:8px;border:1px solid var(--border);background:var(--card);font-size:12px;color:var(--text);"/>
            <button type="button" id="sub-invite-copy"
              style="flex-shrink:0;padding:10px 14px;border-radius:8px;border:none;background:#2563eb;color:#fff;font-weight:700;font-size:13px;cursor:pointer;">
              Copy invite
            </button>
          </div>
          <p id="sub-invite-copied" style="display:none;margin:8px 0 0;font-size:12px;color:#16a34a;">Invite link copied.</p>
        </div>
        <a id="sub-return" href="{safe_return or '/pricing'}" style="display:none;margin-top:8px;padding:12px 28px;border-radius:9px;background:linear-gradient(135deg,#122d4b,#2563eb);color:white;font-weight:700;text-decoration:none;font-size:15px;">Continue to dashboard</a>
      </div>
    </div>
    <script>
    (function() {{
      var returnTo = {json.dumps(return_to)};
      var userId   = {json.dumps(viewer_user_id)};
      var platform = {json.dumps(request.args.get("platform") or "")};
      var attempts = 0, maxAttempts = 8;

      var leagueId = '';
      var season = '';
      try {{
        if (returnTo) {{
          var parts = new URL(returnTo, window.location.origin).pathname.split('/').filter(Boolean);
          if (parts.length >= 3) {{
            platform = parts[0];
            season = parts[1];
            leagueId = parts[2];
          }} else if (parts.length >= 1) {{
            platform = parts[0];
          }}
        }}
      }} catch(e) {{}}

      var params = [];
      if (userId)   params.push('user_id='   + encodeURIComponent(userId));
      if (leagueId) params.push('league_id=' + encodeURIComponent(leagueId));
      if (platform) params.push('platform=' + encodeURIComponent(platform));
      if (season)   params.push('season=' + encodeURIComponent(season));
      var statusUrl = '/api/subscription-status' + (params.length ? '?' + params.join('&') : '');

      function inviteUrl() {{
        if (!leagueId || !season || !platform) return '';
        return window.location.origin + '/invite/' + encodeURIComponent(platform) + '/'
          + encodeURIComponent(season) + '/' + encodeURIComponent(leagueId);
      }}

      function showInvitePanel() {{
        var panel = document.getElementById('sub-invite');
        var input = document.getElementById('sub-invite-url');
        var copyBtn = document.getElementById('sub-invite-copy');
        var url = inviteUrl();
        if (!panel || !input || !url) return false;
        input.value = url;
        panel.style.display = 'block';
        if (copyBtn && !copyBtn.dataset.bound) {{
          copyBtn.dataset.bound = '1';
          copyBtn.addEventListener('click', function() {{
            var done = function() {{
              var note = document.getElementById('sub-invite-copied');
              if (note) note.style.display = 'block';
              copyBtn.textContent = 'Copied';
            }};
            if (navigator.clipboard && navigator.clipboard.writeText) {{
              navigator.clipboard.writeText(url).then(done).catch(function() {{
                input.select(); document.execCommand('copy'); done();
              }});
            }} else {{
              input.select(); document.execCommand('copy'); done();
            }}
          }});
        }}
        return true;
      }}

      function redirect() {{
        window.location.href = returnTo || '/pricing';
      }}

      function finishActive(msg, leaguePlan) {{
        document.getElementById('sub-spinner').style.display = 'none';
        document.getElementById('sub-msg').textContent = msg;
        var btn = document.getElementById('sub-return');
        var showedInvite = showInvitePanel();
        if (btn) btn.style.display = 'inline-block';
        // League buyers already got the invite moment here — tag the dashboard
        // welcome as the feature CTA (no second invite), and suppress the
        // floating invite banner on arrival.
        if (leaguePlan) {{
          try {{ sessionStorage.setItem('br_skip_league_pro_banner', '1'); }} catch (e) {{}}
          try {{
            if (returnTo) {{
              var u = new URL(returnTo, window.location.origin);
              u.searchParams.set('new_subscriber', '1');
              u.searchParams.set('welcome', 'league');
              returnTo = u.pathname + u.search;
              if (btn) btn.setAttribute('href', returnTo);
            }}
          }} catch (e) {{}}
        }}
        if (!showedInvite) setTimeout(redirect, 800);
      }}

      function activate() {{
        attempts++;
        fetch(statusUrl)
          .then(function(r) {{ return r.json(); }})
          .then(function(d) {{
            if (d.has_premium) {{
              var leaguePlan = !!(d.has_league_subscription);
              finishActive(leaguePlan
                ? 'PRO is active for your league.'
                : 'Premium is active - taking you there now!', leaguePlan);
              if (!leaguePlan) setTimeout(redirect, 800);
            }} else if (attempts < maxAttempts) {{
              setTimeout(activate, 1000);
            }} else {{
              // Grant may be on its way via webhook - show continue anyway
              finishActive('Access granted! If features take a moment to appear, try refreshing.', !!leagueId);
              if (!leagueId) setTimeout(redirect, 2000);
            }}
          }})
          .catch(function() {{
            if (attempts < maxAttempts) setTimeout(activate, 1000);
            else setTimeout(redirect, 1000);
          }});
      }}

      // Start quickly - grant was applied server-side before page rendered
      setTimeout(activate, 400);
    }})();
    </script>
    """

    league_highlight = "border-color:#2563eb;box-shadow:0 8px 24px rgba(37,99,235,.2);" if plan == "league" else ""
    user_highlight   = "border-color:#2563eb;box-shadow:0 8px 24px rgba(37,99,235,.2);" if plan == "user"   else ""
    single_highlight = "border-color:#2563eb;box-shadow:0 8px 24px rgba(37,99,235,.2);" if plan == "single_league" else ""
    canceled_banner = """
    <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:10px;padding:14px 18px;margin-bottom:20px;color:#dc2626;font-size:14px;">
      <i class="fa-solid fa-circle-xmark" style="margin-right:6px;"></i>
      Checkout was canceled. You have not been charged.
    </div>""" if canceled else ""
    return f"""
    {canceled_banner}
    <div class="card central" style="max-width:920px;">
      <div class="card-header" style="border-bottom:1px solid var(--border);padding-bottom:16px;margin-bottom:0;text-align:center;">
        <h2 style="margin:0 0 6px;font-size:22px;">Premium</h2>
        <div style="font-size:14px;color:var(--text-muted);">
          Unlock the shipped PRO tools. A Google account is required to subscribe.
          Calculator, Advanced Metrics, and Auction Values stay free.
        </div>
      </div>
      <div class="card-body" style="padding-top:28px;">

        <!-- Feature list — must match static/paywall.js .paywall-features -->
        <div class="pricing-features-block">
          <div class="pricing-features-heading">What PRO includes</div>
          {_pricing_features_grid(_PRO_FEATURES)}
          <div class="pricing-features-heading pricing-features-heading-secondary">Free includes</div>
          {_pricing_features_grid(_FREE_FEATURES, free=True)}
        </div>

        <!-- Pricing cards -->
        <div class="pricing-plan-grid" style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:14px;margin-bottom:28px;">

          <!-- Single-league personal plan -->
          <div style="border:2px solid #e5e7eb;border-radius:14px;padding:22px;transition:all .2s;background:var(--card);{single_highlight}">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;min-height:28px;">
              <div style="font-size:16px;font-weight:700;">One League</div>
            </div>
            <div style="font-size:36px;font-weight:800;line-height:1;margin-bottom:4px;">
              $5<span style="font-size:15px;font-weight:500;color:var(--text-muted);">/year</span>
            </div>
            <div style="font-size:13px;color:var(--text-muted);margin-bottom:20px;">PRO for you in one league you choose</div>
            <button onclick="initiatePurchase('single_league', this)" style="width:100%;padding:11px;border-radius:9px;border:2px solid #2563eb;background:var(--card);color:#2563eb;font-size:14px;font-weight:700;cursor:pointer;">
              Choose a League
            </button>
          </div>

          <!-- League plan -->
          <div style="border:2px solid #e5e7eb;border-radius:14px;padding:22px;transition:all .2s;background:var(--card);{league_highlight}">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;min-height:28px;">
              <div style="font-size:16px;font-weight:700;">League Plan</div>
            </div>
            <div style="font-size:36px;font-weight:800;line-height:1;margin-bottom:4px;">
              $15<span style="font-size:15px;font-weight:500;color:var(--text-muted);">/year</span>
            </div>
            <div style="font-size:13px;color:var(--text-muted);margin-bottom:20px;">Premium for every manager in your league</div>
            <button onclick="initiatePurchase('league', this)" style="width:100%;padding:11px;border-radius:9px;border:2px solid #2563eb;background:var(--card);color:#2563eb;font-size:14px;font-weight:700;cursor:pointer;">
              Subscribe for League
            </button>
          </div>

          <!-- Combo plan -->
          <div style="border:2px solid #2563eb;border-radius:14px;padding:22px;transition:all .2s;background:var(--card);">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
              <div style="font-size:16px;font-weight:700;">League + Personal</div>
              <div style="background:linear-gradient(135deg,#122d4b,#2563eb);color:white;font-size:10px;font-weight:700;padding:3px 9px;border-radius:10px;text-transform:uppercase;letter-spacing:.4px;">Best Value</div>
            </div>
            <div style="font-size:36px;font-weight:800;line-height:1;margin-bottom:4px;">
              $20<span style="font-size:15px;font-weight:500;color:var(--text-muted);">/year</span>
            </div>
            <div style="font-size:13px;color:var(--text-muted);margin-bottom:20px;">Premium for your league and all your personal leagues</div>
            <button onclick="initiatePurchase('combo', this)" style="width:100%;padding:11px;border-radius:9px;border:none;background:linear-gradient(135deg,#122d4b,#2563eb);color:white;font-size:14px;font-weight:700;cursor:pointer;">
              Subscribe Both
            </button>
          </div>

          <!-- Personal plan -->
          <div style="border:2px solid #e5e7eb;border-radius:14px;padding:22px;transition:all .2s;background:var(--card);{user_highlight}">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;min-height:28px;">
              <div style="font-size:16px;font-weight:700;">Personal Plan</div>
            </div>
            <div style="font-size:36px;font-weight:800;line-height:1;margin-bottom:4px;">
              $10<span style="font-size:15px;font-weight:500;color:var(--text-muted);">/year</span>
            </div>
            <div style="font-size:13px;color:var(--text-muted);margin-bottom:20px;">Premium for all your leagues, one account</div>
            <button onclick="initiatePurchase('user', this)" style="width:100%;padding:11px;border-radius:9px;border:2px solid #2563eb;background:var(--card);color:#2563eb;font-size:14px;font-weight:700;cursor:pointer;">
              Subscribe Personally
            </button>
          </div>

        </div>

        <!-- Free tier note -->
        <div style="text-align:center;font-size:13px;color:var(--text-muted);padding-top:12px;border-top:1px solid var(--border);">
          <i class="fa-solid fa-circle-info" style="margin-right:4px;"></i>
          ADP rankings and basic player data are always free. One League unlocks PRO for you only — not your league mates.
        </div>

      </div>
    </div>

    <style>
      @media (max-width: 920px) {{
        .pricing-plan-grid {{ grid-template-columns: 1fr 1fr !important; }}
      }}
      @media (max-width: 560px) {{
        .pricing-plan-grid {{ grid-template-columns: 1fr !important; }}
      }}
    </style>
    """


# ── Pricing pages ─────────────────────────────────────────────────────────────

# ── League PRO invite landing ─────────────────────────────────────────────────

@billing_bp.route("/invite/<platform>/<int:season>/<league_id>")
def page_league_pro_invite(platform: str, season: int, league_id: str):
    """Shareable invite for a league-plan unlock.

    Signed-in visitors go straight to the league dashboard. Guests get a short
    landing that stores the target league and points them at Identify / Connect.
    """
    from flask import redirect
    from utils.league_invite import (
        dashboard_after_invite,
        league_invite_path,
        normalize_invite_platform,
    )

    platform = normalize_invite_platform(platform)
    league_id = str(league_id or "").strip()
    if not league_id or platform not in _SUPPORTED_PLATFORMS:
        return redirect("/pricing")

    dest = dashboard_after_invite(platform, season, league_id)
    # Remember the target so Identify / Connect can land here after sign-in.
    session["invite_platform"] = platform
    session["invite_season"] = int(season)
    session["invite_league_id"] = league_id
    session["last_platform"] = platform
    session["last_season"] = int(season)
    session["last_league_id"] = league_id

    signed_in = bool(
        session.get("account_id")
        or session.get("viewer_username")
        or session.get("viewer_user_id")
    )
    if signed_in:
        return redirect(dest)

    from app import render_page

    invite_path = league_invite_path(platform, season, league_id)
    # Platform-specific connect CTAs; Sleeper can identify by username on home.
    if platform == "sleeper":
        primary_href = "/?invite=1"
        primary_label = "Sign in with Sleeper"
        secondary = (
            '<a href="/auth/google?next='
            + html.escape(invite_path, quote=True)
            + '" style="display:inline-block;margin-top:10px;font-size:13px;color:var(--accent);">'
            "Or continue with Google</a>"
        )
    elif platform == "espn":
        primary_href = f"/espn/{season}/{league_id}/dashboard"
        primary_label = "Connect ESPN league"
        secondary = ""
    else:
        primary_href = f"/{platform}/{season}/{league_id}/dashboard"
        primary_label = f"Open {platform.upper()} league"
        secondary = ""

    body = f"""
    <div class="card central" style="max-width:520px;text-align:center;">
      <div class="card-body" style="padding:40px 28px;">
        <div style="font-size:40px;margin-bottom:14px;"><i class="fa-solid fa-unlock" style="color:#2563eb;"></i></div>
        <h1 style="margin:0 0 10px;font-size:22px;">Your league unlocked PRO</h1>
        <p style="margin:0 0 22px;color:var(--text-muted);font-size:14px;line-height:1.55;">
          A league mate already paid for shared premium. Sign in as a manager in this
          league to use Trade Intel, Breakouts, Front Office, and the rest of PRO.
        </p>
        <a href="{html.escape(primary_href, quote=True)}"
           style="display:inline-block;padding:12px 22px;border-radius:9px;background:linear-gradient(135deg,#122d4b,#2563eb);color:#fff;font-weight:700;text-decoration:none;font-size:14px;">
          {html.escape(primary_label)}
        </a>
        {secondary}
      </div>
    </div>
    """
    return render_page(
        "League PRO Invite | BR Fantasy",
        None, "pricing", body, platform, season,
        description="Join your league's shared BR Fantasy PRO access.",
        noindex=True,
        lite_js=True,
    )


@billing_bp.route("/<platform>/<int:season>/<league_id>/pricing")
def page_pricing(platform: str, season: int, league_id: str):
    from app import render_page
    _try_grant_from_stripe_success()
    body_html = _pricing_body()
    # active="pricing" keeps AdSense off this checkout/utility page.
    # lite_js: guests get public.js + seo_lite.css (pricing is card chrome only);
    # signed-in visitors still receive full app.js / dashboard.css via render_page.
    return render_page(
        "Pricing", league_id, "pricing", body_html, platform, season,
        lite_js=True,
    )


@billing_bp.route("/pricing")
def page_pricing_guest():
    from app import get_nfl_state, render_page
    _try_grant_from_stripe_success()
    nfl_state = get_nfl_state() or {}
    current_season = int(nfl_state.get("season") or datetime.now().year)
    body_html = _pricing_body()
    platform = _request_platform()
    if platform not in _SUPPORTED_PLATFORMS:
        platform = "sleeper"
    return render_page(
        "Pricing", None, "pricing", body_html, platform, current_season,
        lite_js=True,
    )


@billing_bp.route("/api/pro-signup/pending", methods=["POST"])
def pro_signup_pending():
    """Stage plan + league from the home-page PRO wizard, then send guests to Google."""
    data = request.get_json(force=True) or {}
    pending, error = _normalize_pending_checkout(data)
    if error or not pending:
        return jsonify({"ok": False, "error": error or "Could not save this signup."}), 400
    session["pending_checkout"] = {
        "plan": pending["plan"],
        "platform": pending["platform"],
        "league_id": pending["league_id"],
        "season": pending["season"],
    }
    session["pending_link"] = {
        "platform": pending["platform"],
        "league_id": pending["league_id"],
        "season": pending["season"],
        "team_id": pending.get("team_id"),
        "name": pending.get("name"),
        "username": pending.get("username"),
    }
    return jsonify({
        "ok": True,
        "auth_url": "/auth/google?intent=onboarding&next=/pro/resume-checkout",
    })


@billing_bp.route("/pro/resume-checkout")
def resume_pro_checkout():
    """After Google (or Yahoo) returns, open Stripe for the staged home PRO signup."""
    from flask import redirect

    pending = session.get("pending_checkout")
    if not isinstance(pending, dict) or not pending.get("plan"):
        return redirect("/pricing")

    if not session.get("account_id"):
        return redirect("/auth/google?intent=onboarding&next=/pro/resume-checkout")
    user_id = _checkout_user_id()
    if not user_id:
        return redirect("/auth/google?intent=onboarding&next=/pro/resume-checkout")

    plan = str(pending.get("plan") or "").strip()
    league_id = str(pending.get("league_id") or "").strip()
    platform = str(pending.get("platform") or "sleeper").strip().lower()
    try:
        season = int(pending.get("season") or datetime.now().year)
    except (TypeError, ValueError):
        season = datetime.now().year
    if plan not in _STRIPE_PRICES or platform not in _SUPPORTED_PLATFORMS:
        session.pop("pending_checkout", None)
        return redirect("/pricing")
    if plan in _LEAGUE_REQUIRED_PLANS and not league_id:
        return redirect("/pricing")

    if plan in _MEMBERSHIP_REQUIRED_PLANS and league_id:
        from dashboard_services.subscriptions import viewer_is_league_member
        member_id = session.get("viewer_user_id") or session.get("viewer_username")
        if not viewer_is_league_member(member_id, league_id, platform, season):
            return redirect("/pricing?canceled=1")

    return_url = (
        f"/{platform}/{season}/{urllib.parse.quote(league_id, safe='')}/dashboard?new_subscriber=1&welcome=personal"
        if league_id else "/pricing?success=1"
    )
    payload = {
        "plan": plan,
        "league_id": league_id,
        "platform": platform,
        "season": season,
        "return_url": return_url,
    }
    url, error = _stripe_checkout_url(user_id, payload)
    if url:
        session.pop("pending_checkout", None)
        return redirect(url)
    logger.warning("[checkout] resume failed: %s", error)
    return redirect("/pricing?canceled=1")


def _stripe_checkout_url(user_id: str, payload: dict) -> tuple[str | None, str | None]:
    """Create a Stripe Checkout session. Returns (url, error)."""
    plan = str(payload.get("plan") or "").strip()
    league_id = str(payload.get("league_id") or "").strip()
    platform = _request_platform(payload)
    try:
        season = int(payload.get("season") or datetime.now().year)
    except (TypeError, ValueError):
        return None, "Invalid season"
    return_url = str(payload.get("return_url") or "").strip()
    if plan not in _STRIPE_PRICES:
        return None, "Invalid plan"
    if platform not in _SUPPORTED_PLATFORMS:
        return None, "Invalid platform"

    price_spec = _STRIPE_PRICES[plan]
    base_url = request.host_url.rstrip("/")
    return_url = _safe_local_url(return_url, "")
    success_url = base_url + "/pricing?success=1&session_id={CHECKOUT_SESSION_ID}"
    if return_url:
        success_url += "&return_to=" + urllib.parse.quote(return_url, safe="")
    success_url += "&platform=" + urllib.parse.quote(platform, safe="")
    if league_id:
        cancel_url = f"{base_url}/{platform}/{season}/{urllib.parse.quote(league_id, safe='')}/pricing?canceled=1"
    else:
        cancel_url = base_url + "/pricing?canceled=1&platform=" + urllib.parse.quote(platform, safe="")
    price_data = {
        "currency": "usd",
        "unit_amount": price_spec["unit_amount"],
        "recurring": {"interval": "year"},
    }
    if price_spec.get("product"):
        price_data["product"] = price_spec["product"]
    else:
        price_data["product_data"] = {
            "name": price_spec.get("product_name") or "BR Fantasy PRO",
        }
    try:
        checkout = _stripe().checkout.Session.create(
            mode="subscription",
            line_items=[{
                "price_data": price_data,
                "quantity": 1,
            }],
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={"plan": plan, "user_id": user_id, "league_id": league_id,
                      "platform": platform, "season": str(season),
                      "account_id": str(session.get("account_id") or "")},
        )
        return checkout.url, None
    except Exception:
        logger.exception("[stripe] checkout session error")
        return None, "Internal error"


# ── Stripe API endpoints ──────────────────────────────────────────────────────

@billing_bp.route("/api/create-checkout-session", methods=["POST"])
def create_checkout_session():
    # New subscriptions require a Google account. Existing rows keyed by a
    # username remain readable through the entitlement resolver. Google-only
    # managers have account_id without a Sleeper viewer id.
    blocked = _require_google_to_subscribe()
    if blocked:
        return blocked
    user_id = (
        session.get("viewer_user_id")
        or session.get("viewer_username")
        or (("acct:" + str(session.get("account_id")).strip()) if session.get("account_id") else None)
    )
    logger.info("[checkout] Request from user: %s", user_id)
    if not user_id:
        return jsonify({"error": "Sign in with Google to subscribe."}), 401

    payload    = request.get_json(force=True)
    plan       = str(payload.get("plan") or "").strip()
    league_id  = str(payload.get("league_id") or "").strip()
    return_url = str(payload.get("return_url") or "").strip()
    platform   = _request_platform(payload)
    try:
        season = int(payload.get("season") or datetime.now().year)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid season"}), 400
    
    logger.info("[checkout] Request payload: plan=%s, league_id=%s, return_url=%s", plan, league_id, return_url)

    if plan not in _STRIPE_PRICES:
        logger.info("[checkout] Invalid plan: %s, available plans: %s", plan, list(_STRIPE_PRICES.keys()))
        return jsonify({"error": "Invalid plan"}), 400
    if platform not in _SUPPORTED_PLATFORMS:
        return jsonify({"error": "Invalid platform"}), 400
    if plan in _LEAGUE_REQUIRED_PLANS and not league_id:
        return jsonify({"error": "Choose a league before purchasing this plan."}), 400

    # League/combo/single_league require membership. Shared plans unlock
    # co-managers; single_league is buyer-only but still must be a real league
    # the buyer belongs to.
    if plan in _MEMBERSHIP_REQUIRED_PLANS and league_id:
        from dashboard_services.subscriptions import viewer_is_league_member
        member_id = session.get("viewer_user_id") or session.get("viewer_username")
        if not viewer_is_league_member(member_id, league_id, platform, season):
            return jsonify({
                "error": "You must be a member of this league to purchase a league plan."
            }), 403

    username = session.get("viewer_username")
    stable_id = session.get("viewer_user_id")
    account_id = session.get("account_id")
    has_league = bool(league_id and has_premium_access(None, league_id, platform))
    has_user = bool(
        (stable_id and has_premium_access(stable_id, None, platform))
        or (username and has_premium_access(username, None, platform))
        or (account_id and has_premium_access(None, None, platform, account_id=account_id))
    )
    has_single = bool(
        league_id and (
            (stable_id and has_user_league_subscription(stable_id, league_id, platform))
            or (username and has_user_league_subscription(username, league_id, platform))
            or (account_id and has_user_league_subscription(
                None, league_id, platform, account_id=account_id,
            ))
            or (user_id and has_user_league_subscription(user_id, league_id, platform))
        )
    )
    # A combo is its own Stripe subscription, not an in-place upgrade. Starting
    # one while either component is active would double-bill the customer.
    # Single-league is redundant when a full personal or shared league plan
    # already covers this room.
    duplicate = ((plan == "league" and has_league)
                 or (plan == "user" and has_user)
                 or (plan == "combo" and (has_league or has_user))
                 or (plan == "single_league" and (has_single or has_user or has_league)))
    logger.info("[checkout] Existing components league=%s user=%s single=%s",
                has_league, has_user, has_single)
    if duplicate:
        return jsonify({"error": "You already have this premium subscription."}), 400

    price_spec = _STRIPE_PRICES[plan]
    base_url   = request.host_url.rstrip("/")

    return_url = _safe_local_url(return_url, "")

    success_url = base_url + "/pricing?success=1&session_id={CHECKOUT_SESSION_ID}"
    if return_url:
        success_url += "&return_to=" + urllib.parse.quote(return_url, safe="")
    success_url += "&platform=" + urllib.parse.quote(platform, safe="")

    if league_id:
        cancel_url = f"{base_url}/{platform}/{season}/{urllib.parse.quote(league_id, safe='')}/pricing?canceled=1"
    else:
        cancel_url = base_url + "/pricing?canceled=1&platform=" + urllib.parse.quote(platform, safe="")

    price_data = {
        "currency": "usd",
        "unit_amount": price_spec["unit_amount"],
        "recurring": {"interval": "year"},
    }
    if price_spec.get("product"):
        price_data["product"] = price_spec["product"]
    else:
        price_data["product_data"] = {
            "name": price_spec.get("product_name") or "BR Fantasy PRO",
        }

    try:
        checkout = _stripe().checkout.Session.create(
            mode="subscription",
            line_items=[{
                "price_data": price_data,
                "quantity": 1,
            }],
            success_url=success_url,
            cancel_url=cancel_url,
            metadata=_checkout_metadata(plan, user_id, league_id, platform, season),
            # Copy onto the Subscription so invoice / subscription.created
            # events can grant One League even if session metadata is dropped.
            subscription_data={
                "metadata": _checkout_metadata(plan, user_id, league_id, platform, season),
            },
        )
        return jsonify({"url": checkout.url})
    except Exception:
        logger.exception("[stripe] checkout session error")
        return jsonify({"error": "Internal error"}), 500


@billing_bp.route("/api/stripe-webhook", methods=["POST"])
def stripe_webhook():
    payload = request.get_data()
    sig     = request.headers.get("Stripe-Signature", "")
    secret  = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

    if not secret:
        logger.error("[stripe] STRIPE_WEBHOOK_SECRET not set - webhook will always fail signature check")
        return "", 400

    try:
        event = _stripe().Webhook.construct_event(payload, sig, secret)
    except ValueError as e:
        logger.error("[stripe] webhook bad payload: %s", e)
        return "", 400
    except _stripe().SignatureVerificationError as e:
        logger.error("[stripe] webhook signature mismatch: %s", e)
        return "", 400

    etype = event["type"]

    if etype in ("checkout.session.completed", "customer.subscription.created"):
        s         = event["data"]["object"]
        meta      = _metadata_dict(s)
        plan      = (meta.get("plan") or "").strip()
        # Prefer checkout user_id; Google-only sessions store acct identity here.
        user_id   = meta.get("user_id") or meta.get("account_id")
        user_id   = _subscriber_user_id({
            "user_id": user_id or "",
            "account_id": meta.get("account_id") or "",
        })
        platform  = meta.get("platform") or "sleeper"
        league_id = meta.get("league_id") or ""
        if etype == "customer.subscription.created":
            sub_id = _stripe_id(_stripe_field(s, "id"))
            cust_id = _stripe_id(_stripe_field(s, "customer"))
            sub = s
            try:
                expires_at = _subscription_period_end(sub)
            except Exception:
                expires_at = datetime.now(timezone.utc) + timedelta(days=32)
            if plan not in ("league", "user", "combo", "single_league"):
                plan = _plan_from_subscription(sub)
        else:
            sub_id    = _stripe_id(_stripe_field(s, "subscription"))
            cust_id   = _stripe_id(_stripe_field(s, "customer"))
            try:
                sub = _stripe().Subscription.retrieve(sub_id) if sub_id else None
                expires_at = _subscription_period_end(sub) if sub else (
                    datetime.now(timezone.utc) + timedelta(days=32)
                )
                if plan not in ("league", "user", "combo", "single_league"):
                    plan = _plan_from_subscription(sub)
            except Exception:
                expires_at = datetime.now(timezone.utc) + timedelta(days=32)

        _apply_plan_grant(
            plan, user_id, league_id, platform, expires_at, sub_id, cust_id,
            source=f"webhook:{etype}",
        )

    elif etype == "invoice.paid":
        s      = event["data"]["object"]
        sub_id = s.subscription
        if sub_id:
            try:
                sub        = _stripe().Subscription.retrieve(sub_id)
                expires_at = _subscription_period_end(sub)
                from dashboard_services.db import get_conn
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE league_subscriptions SET expires_at=%s, updated_at=NOW() WHERE stripe_subscription_id=%s",
                            (expires_at, sub_id),
                        )
                        cur.execute(
                            "UPDATE user_subscriptions SET expires_at=%s, updated_at=NOW() WHERE stripe_subscription_id=%s",
                            (expires_at, sub_id),
                        )
                        cur.execute(
                            "UPDATE user_league_subscriptions SET expires_at=%s, updated_at=NOW() WHERE stripe_subscription_id=%s",
                            (expires_at, sub_id),
                        )
            except Exception as e:
                logger.exception("[stripe] invoice.paid renewal error: %s", e)

    elif etype in ("customer.subscription.deleted", "customer.subscription.updated"):
        s = event["data"]["object"]
        if s.status in ("canceled", "unpaid", "past_due"):
            sub_id = s.id
            cancel_subscription(sub_id, "league")
            cancel_subscription(sub_id, "user")
            cancel_subscription(sub_id, "single_league")

    return "", 200


# ── Subscription status API ───────────────────────────────────────────────────

@billing_bp.route("/api/subscription-status")
def api_subscription_status():
    """Check if user has premium access for a league."""
    from dashboard_services.subscriptions import get_subscription_info

    # Identity is taken from the session, never from a client-supplied user_id,
    # so a caller cannot enumerate other users' subscription details.
    username = session.get("viewer_username")
    stable_id = session.get("viewer_user_id")
    league_id = request.args.get("league_id")
    platform = _request_platform()

    if platform not in _SUPPORTED_PLATFORMS:
        return jsonify({"has_premium": False, "subscription_type": None,
                        "error": "Invalid platform"}), 400

    try:
        # Current subscriptions use the immutable provider id. Check the legacy
        # handle only when needed so older rows remain manageable and visible.
        sub_info = get_subscription_info(stable_id or username, league_id, platform)
        if username and stable_id and not sub_info.get("has_user_subscription"):
            legacy = get_subscription_info(username, None, platform)
            if legacy.get("has_user_subscription"):
                sub_info["has_user_subscription"] = True
                sub_info["has_premium"] = True
                sub_info["subscription_type"] = (
                    "combo" if sub_info.get("has_league_subscription") else "user"
                )
                sub_info["expires_at"] = sub_info.get("expires_at") or legacy.get("expires_at")
                sub_info["stripe_customer_id"] = (
                    sub_info.get("stripe_customer_id") or legacy.get("stripe_customer_id")
                )
        # Detailed league rows alone are not an entitlement for an unrelated
        # Sleeper user; use the same membership/account-aware gate as pages.
        sub_info["has_premium"] = has_premium_for_viewer(
            username, stable_id, league_id, platform, request.args.get("season"),
        )
        from dashboard_services.subscriptions import needs_google_link_for_pro, pro_require_google
        from utils.league_invite import is_league_plan_buyer, league_invite_path
        sub_info["needs_google_link"] = needs_google_link_for_pro(username, stable_id, platform)
        sub_info["pro_require_google"] = pro_require_google()
        buyer_id = sub_info.get("subscriber_user_id")
        viewer_ids = {
            str(username or "").strip(),
            str(stable_id or "").strip(),
            (("acct:" + str(session.get("account_id")).strip()) if session.get("account_id") else ""),
            str(session.get("account_id") or "").strip(),
        }
        sub_info["is_league_buyer"] = is_league_plan_buyer(viewer_ids, buyer_id)
        try:
            season_i = int(request.args.get("season") or session.get("last_season") or 0)
        except (TypeError, ValueError):
            season_i = 0
        if sub_info.get("has_league_subscription") and league_id and season_i:
            sub_info["invite_path"] = league_invite_path(platform, season_i, league_id)
        else:
            sub_info["invite_path"] = None
        # Strip internal/PII fields - the client only needs entitlement flags.
        for _k in ("stripe_customer_id", "subscriber_user_id"):
            sub_info.pop(_k, None)
        return jsonify(sub_info)
    except Exception:
        logger.exception("[api_subscription_status] Error")
        return jsonify({"has_premium": False, "subscription_type": None, "error": "Internal error"}), 500


# ── Stripe Customer Portal ────────────────────────────────────────────────────

@billing_bp.route("/api/create-portal-session", methods=["POST"])
def api_create_portal_session():
    """Create a Stripe billing portal session so users can manage subscriptions."""
    from dashboard_services.subscriptions import get_subscription_info

    # Mirror checkout identity: Google-only managers have account_id without a
    # Sleeper viewer id, and their Stripe rows are keyed as acct:<id>.
    user_id = (
        session.get("viewer_user_id")
        or session.get("viewer_username")
        or (("acct:" + str(session.get("account_id")).strip()) if session.get("account_id") else None)
    )
    league_id = request.json.get("league_id") if request.is_json else request.form.get("league_id")
    payload = request.get_json(silent=True) if request.is_json else request.form
    payload = payload or {}
    platform = _request_platform(payload)

    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    if platform not in _SUPPORTED_PLATFORMS:
        return jsonify({"error": "Invalid platform"}), 400

    try:
        sub_info    = get_subscription_info(user_id, league_id, platform)
        customer_id = sub_info.get("stripe_customer_id")
        # Fall back to user-only lookup (personal plan) if league lookup has no customer
        if not customer_id and league_id:
            user_sub    = get_subscription_info(user_id, None, platform)
            customer_id = user_sub.get("stripe_customer_id")
        if not customer_id and session.get("viewer_username") and session.get("viewer_user_id"):
            legacy_sub = get_subscription_info(session.get("viewer_username"), None, platform)
            customer_id = legacy_sub.get("stripe_customer_id")
        if not customer_id and session.get("account_id"):
            acct_key = "acct:" + str(session.get("account_id")).strip()
            if user_id != acct_key:
                acct_sub = get_subscription_info(acct_key, None, platform)
                customer_id = acct_sub.get("stripe_customer_id")
            if not customer_id:
                bare_sub = get_subscription_info(str(session.get("account_id")).strip(), None, platform)
                customer_id = bare_sub.get("stripe_customer_id")
        if not customer_id:
            return jsonify({"error": "No Stripe customer found for your account. Contact support if you believe this is an error."}), 404

        return_url = request.json.get("return_url") if request.is_json else request.form.get("return_url")
        return_url = _safe_local_url(
            return_url,
            request.host_url.rstrip("/") + f"/pricing?platform={urllib.parse.quote(platform, safe='')}",
        )

        portal_session = _stripe().billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )
        return jsonify({"url": portal_session.url})
    except Exception:
        logger.exception("[api_create_portal_session] Error")
        return jsonify({"error": "Internal error"}), 500
