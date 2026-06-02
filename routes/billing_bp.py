"""
Billing / subscription routes.

Routes: /pricing, /api/create-checkout-session, /api/stripe-webhook,
        /api/subscription-status
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
    create_user_subscription,
    has_premium_access,
)

billing_bp = Blueprint("billing", __name__)
logger = logging.getLogger(__name__)

_STRIPE_LEAGUE_PRODUCT = "prod_USjDJYPhNGnmvM"
_STRIPE_USER_PRODUCT   = "prod_USjDRuVDcwH1xb"
_STRIPE_COMBO_PRODUCT  = "prod_UT5DaCA4u6hWgb"


def _stripe():
    """Lazy-import stripe so missing package doesn't break the whole blueprint."""
    import stripe as _s
    _s.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
    return _s


_STRIPE_PRICES = {
    "league": {"unit_amount": 1000, "product": _STRIPE_LEAGUE_PRODUCT},
    "user":   {"unit_amount":  500, "product": _STRIPE_USER_PRODUCT},
    "combo":  {"unit_amount": 1200, "product": _STRIPE_COMBO_PRODUCT},
}


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

        meta      = cs.metadata.to_dict() if cs.metadata else {}
        plan      = meta.get("plan")
        user_id   = meta.get("user_id")
        league_id = meta.get("league_id") or ""
        sub_id    = cs.subscription
        cust_id   = cs.customer

        if plan not in ("league", "user", "combo"):
            return
        if plan == "user" and not user_id:
            return
        if plan == "league" and not league_id:
            return
        if plan == "combo" and not league_id and not user_id:
            return

        if has_premium_access(user_id or None, league_id or None, "sleeper"):
            return

        try:
            sub        = _stripe().Subscription.retrieve(sub_id) if sub_id else None
            expires_at = (
                _subscription_period_end(sub)
                if sub else datetime.now(timezone.utc) + timedelta(days=366)
            )
        except Exception:
            expires_at = datetime.now(timezone.utc) + timedelta(days=366)

        if plan in ("league", "combo") and league_id:
            create_league_subscription(
                league_id, user_id or "", expires_at,
                stripe_subscription_id=sub_id,
                stripe_customer_id=cust_id,
            )
        if plan in ("user", "combo") and user_id:
            create_user_subscription(
                user_id, expires_at,
                stripe_subscription_id=sub_id,
                stripe_customer_id=cust_id,
            )
    except Exception:
        logger.exception("[stripe] success-page session verification failed")


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
                    season = _dt.now().year
                    return_to = f"/sleeper/{season}/{league_id_meta}/dashboard?new_subscriber=1"
            except Exception:
                pass

        safe_return = html.escape(return_to) if return_to else ""
        viewer_user_id = _session.get("viewer_user_id") or _session.get("viewer_username") or ""
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
        <div id="sub-spinner" style="margin:0 auto 16px;width:32px;height:32px;border:3px solid #e5e7eb;border-top-color:#667eea;border-radius:50%;animation:paywall-spin .8s linear infinite;"></div>
        <a id="sub-return" href="{safe_return or '/pricing'}" style="display:none;margin-top:8px;padding:12px 28px;border-radius:9px;background:linear-gradient(135deg,#667eea,#764ba2);color:white;font-weight:700;text-decoration:none;font-size:15px;">Continue</a>
      </div>
    </div>
    <script>
    (function() {{
      var returnTo = {json.dumps(return_to)};
      var userId   = {json.dumps(viewer_user_id)};
      var attempts = 0, maxAttempts = 8;

      var leagueId = '';
      try {{
        if (returnTo) {{
          var parts = new URL(returnTo, window.location.origin).pathname.split('/').filter(Boolean);
          if (parts.length >= 3) leagueId = parts[2];
        }}
      }} catch(e) {{}}

      var params = [];
      if (userId)   params.push('user_id='   + encodeURIComponent(userId));
      if (leagueId) params.push('league_id=' + encodeURIComponent(leagueId));
      var statusUrl = '/api/subscription-status' + (params.length ? '?' + params.join('&') : '');

      function redirect() {{
        window.location.href = returnTo || '/pricing';
      }}

      function activate() {{
        attempts++;
        fetch(statusUrl)
          .then(function(r) {{ return r.json(); }})
          .then(function(d) {{
            if (d.has_premium) {{
              document.getElementById('sub-spinner').style.display = 'none';
              document.getElementById('sub-msg').textContent = 'Premium is active - taking you there now!';
              setTimeout(redirect, 800);
            }} else if (attempts < maxAttempts) {{
              setTimeout(activate, 1000);
            }} else {{
              // Grant may be on its way via webhook - redirect anyway
              document.getElementById('sub-spinner').style.display = 'none';
              document.getElementById('sub-msg').textContent = 'Access granted! If features take a moment to appear, try refreshing.';
              var btn = document.getElementById('sub-return');
              if (btn) btn.style.display = 'inline-block';
              setTimeout(redirect, 2000);
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

    league_highlight = "border-color:#667eea;box-shadow:0 8px 24px rgba(102,126,234,.2);" if plan == "league" else ""
    user_highlight   = "border-color:#667eea;box-shadow:0 8px 24px rgba(102,126,234,.2);" if plan == "user"   else ""
    canceled_banner = """
    <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:10px;padding:14px 18px;margin-bottom:20px;color:#dc2626;font-size:14px;">
      <i class="fa-solid fa-circle-xmark" style="margin-right:6px;"></i>
      Checkout was canceled. You have not been charged.
    </div>""" if canceled else ""
    return f"""
    {canceled_banner}
    <div class="card central" style="max-width:760px;">
      <div class="card-header" style="border-bottom:1px solid var(--border);padding-bottom:16px;margin-bottom:0;text-align:center;">
        <h2 style="margin:0 0 6px;font-size:22px;">BR Fantasy Premium</h2>
        <div style="font-size:14px;color:var(--text-muted);">
          Unlock advanced analytics and insights for your dynasty league
        </div>
      </div>
      <div class="card-body" style="padding-top:28px;">

        <!-- Feature list -->
        <div style="margin-bottom:28px;">
          <div style="font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:var(--text-muted);margin-bottom:12px;">What you get</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
            <div style="display:flex;align-items:center;gap:8px;font-size:14px;">
              <i class="fa-solid fa-chart-line" style="color:#667eea;width:16px;text-align:center;"></i>
              Full Trade Intelligence feed
            </div>
            <div style="display:flex;align-items:center;gap:8px;font-size:14px;">
              <i class="fa-solid fa-fire" style="color:#667eea;width:16px;text-align:center;"></i>
              All Breakout Engine candidates
            </div>
            <div style="display:flex;align-items:center;gap:8px;font-size:14px;">
              <i class="fa-solid fa-clock-rotate-left" style="color:#667eea;width:16px;text-align:center;"></i>
              Player trade history
            </div>
            <div style="display:flex;align-items:center;gap:8px;font-size:14px;">
              <i class="fa-solid fa-star" style="color:#667eea;width:16px;text-align:center;"></i>
              All future premium features
            </div>
          </div>
        </div>

        <!-- Pricing cards -->
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:28px;">

          <!-- League plan -->
          <div style="border:2px solid #e5e7eb;border-radius:14px;padding:24px;transition:all .2s;background:var(--card);">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;min-height:28px;">
              <div style="font-size:17px;font-weight:700;">League Plan</div>
            </div>
            <div style="font-size:38px;font-weight:800;line-height:1;margin-bottom:4px;">
              $10<span style="font-size:16px;font-weight:500;color:var(--text-muted);">/year</span>
            </div>
            <div style="font-size:13px;color:var(--text-muted);margin-bottom:20px;">Premium for every manager in your league</div>
            <button onclick="initiatePurchase('league', this)" style="width:100%;padding:11px;border-radius:9px;border:2px solid #667eea;background:var(--card);color:#667eea;font-size:14px;font-weight:700;cursor:pointer;">
              Subscribe for League
            </button>
          </div>

          <!-- Combo plan -->
          <div style="border:2px solid #667eea;border-radius:14px;padding:24px;transition:all .2s;background:var(--card);{league_highlight}">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
              <div style="font-size:17px;font-weight:700;">League + Personal</div>
              <div style="background:linear-gradient(135deg,#667eea,#764ba2);color:white;font-size:10px;font-weight:700;padding:3px 9px;border-radius:10px;text-transform:uppercase;letter-spacing:.4px;">Best Value</div>
            </div>
            <div style="font-size:38px;font-weight:800;line-height:1;margin-bottom:4px;">
              $12<span style="font-size:16px;font-weight:500;color:var(--text-muted);">/year</span>
            </div>
            <div style="font-size:13px;color:var(--text-muted);margin-bottom:20px;">Premium for your league and all your personal leagues</div>
            <button onclick="initiatePurchase('combo', this)" style="width:100%;padding:11px;border-radius:9px;border:none;background:linear-gradient(135deg,#667eea,#764ba2);color:white;font-size:14px;font-weight:700;cursor:pointer;">
              Subscribe Both
            </button>
          </div>

          <!-- Personal plan -->
          <div style="border:2px solid #e5e7eb;border-radius:14px;padding:24px;transition:all .2s;background:var(--card);{user_highlight}">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;min-height:28px;">
              <div style="font-size:17px;font-weight:700;">Personal Plan</div>
            </div>
            <div style="font-size:38px;font-weight:800;line-height:1;margin-bottom:4px;">
              $5<span style="font-size:16px;font-weight:500;color:var(--text-muted);">/year</span>
            </div>
            <div style="font-size:13px;color:var(--text-muted);margin-bottom:20px;">Premium for all your leagues, one account</div>
            <button onclick="initiatePurchase('user', this)" style="width:100%;padding:11px;border-radius:9px;border:2px solid #667eea;background:var(--card);color:#667eea;font-size:14px;font-weight:700;cursor:pointer;">
              Subscribe Personally
            </button>
          </div>

        </div>

        <!-- Free tier note -->
        <div style="text-align:center;font-size:13px;color:var(--text-muted);padding-top:12px;border-top:1px solid var(--border);">
          <i class="fa-solid fa-circle-info" style="margin-right:4px;"></i>
          ADP rankings and basic player data are always free.
        </div>

      </div>
    </div>

    <style>
      @media (max-width: 760px) {{
        .card-body > div:nth-child(2) {{ grid-template-columns: 1fr !important; }}
        .card-body > div:nth-child(3) {{ grid-template-columns: 1fr !important; }}
      }}
    </style>
    """


# ── Pricing pages ─────────────────────────────────────────────────────────────

@billing_bp.route("/<platform>/<int:season>/<league_id>/pricing")
def page_pricing(platform: str, season: int, league_id: str):
    from app import render_page
    _try_grant_from_stripe_success()
    body_html = _pricing_body()
    return render_page("Pricing", league_id, None, body_html, platform, season)


@billing_bp.route("/pricing")
def page_pricing_guest():
    from app import get_nfl_state, render_page
    _try_grant_from_stripe_success()
    nfl_state = get_nfl_state() or {}
    current_season = int(nfl_state.get("season") or datetime.now().year)
    body_html = _pricing_body()
    return render_page("Pricing", None, None, body_html, "sleeper", current_season)


# ── Stripe API endpoints ──────────────────────────────────────────────────────

@billing_bp.route("/api/create-checkout-session", methods=["POST"])
def create_checkout_session():
    user_id = session.get("viewer_username")
    logger.info("[checkout] Request from user: %s", user_id)
    if not user_id:
        return jsonify({"error": "Must be logged in to subscribe"}), 401

    payload    = request.get_json(force=True)
    plan       = str(payload.get("plan") or "").strip()
    league_id  = str(payload.get("league_id") or "").strip()
    return_url = str(payload.get("return_url") or "").strip()
    
    logger.info("[checkout] Request payload: plan=%s, league_id=%s, return_url=%s", plan, league_id, return_url)

    if plan not in _STRIPE_PRICES:
        logger.info("[checkout] Invalid plan: %s, available plans: %s", plan, list(_STRIPE_PRICES.keys()))
        return jsonify({"error": "Invalid plan"}), 400

    check_league = league_id if league_id else None
    has_premium = has_premium_access(user_id, check_league, "sleeper")
    logger.info("[checkout] User premium status for league %s: %s", check_league, has_premium)
    if has_premium:
        return jsonify({"error": "You already have an active premium subscription."}), 400

    price_spec = _STRIPE_PRICES[plan]
    base_url   = request.host_url.rstrip("/")

    if return_url and not (return_url.startswith(base_url) or return_url.startswith("/")):
        return_url = ""

    success_url = base_url + "/pricing?success=1&session_id={CHECKOUT_SESSION_ID}"
    if return_url:
        success_url += "&return_to=" + urllib.parse.quote(return_url, safe="")

    try:
        checkout = _stripe().checkout.Session.create(
            mode="subscription",
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product": price_spec["product"],
                    "unit_amount": price_spec["unit_amount"],
                    "recurring": {"interval": "year"},
                },
                "quantity": 1,
            }],
            success_url=success_url,
            cancel_url=base_url + "/pricing?canceled=1",
            metadata={"plan": plan, "user_id": user_id, "league_id": league_id},
        )
        return jsonify({"url": checkout.url})
    except Exception as e:
        logger.exception("[stripe] checkout session error: %s", e)
        return jsonify({"error": str(e)}), 500


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

    if etype == "checkout.session.completed":
        s         = event["data"]["object"]
        meta      = dict(s.metadata) if s.metadata else {}
        plan      = meta.get("plan")
        user_id   = meta.get("user_id")
        league_id = meta.get("league_id") or ""
        sub_id    = s.subscription
        cust_id   = s.customer

        try:
            sub = _stripe().Subscription.retrieve(sub_id)
            expires_at = _subscription_period_end(sub)
        except Exception:
            expires_at = datetime.now(timezone.utc) + timedelta(days=32)

        if plan in ("league", "combo") and league_id:
            ok = create_league_subscription(
                league_id, user_id or "", expires_at,
                stripe_subscription_id=sub_id,
                stripe_customer_id=cust_id,
            )
            logger.info("[stripe] webhook league subscription %s for league=%s user=%s expires=%s",
                        "created" if ok else "FAILED", league_id, user_id, expires_at)
        if plan in ("user", "combo") and user_id:
            ok = create_user_subscription(
                user_id, expires_at,
                stripe_subscription_id=sub_id,
                stripe_customer_id=cust_id,
            )
            logger.info("[stripe] webhook user subscription %s for user=%s expires=%s",
                        "created" if ok else "FAILED", user_id, expires_at)
        if plan not in ("league", "user", "combo"):
            logger.warning("[stripe] webhook checkout.session.completed unhandled: plan=%s league=%s user=%s",
                           plan, league_id, user_id)

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
            except Exception as e:
                logger.exception("[stripe] invoice.paid renewal error: %s", e)

    elif etype in ("customer.subscription.deleted", "customer.subscription.updated"):
        s = event["data"]["object"]
        if s.status in ("canceled", "unpaid", "past_due"):
            sub_id = s.id
            cancel_subscription(sub_id, "league")
            cancel_subscription(sub_id, "user")

    return "", 200


# ── Subscription status API ───────────────────────────────────────────────────

@billing_bp.route("/api/subscription-status")
def api_subscription_status():
    """Check if user has premium access for a league."""
    from dashboard_services.subscriptions import get_subscription_info

    # Identity is taken from the session, never from a client-supplied user_id,
    # so a caller cannot enumerate other users' subscription details.
    user_id = session.get("viewer_username")
    league_id = request.args.get("league_id")
    platform = request.args.get("platform", "sleeper")

    try:
        sub_info = get_subscription_info(user_id, league_id, platform)
        # Strip internal/PII fields — the client only needs entitlement flags.
        for _k in ("stripe_customer_id", "subscriber_user_id"):
            sub_info.pop(_k, None)
        return jsonify(sub_info)
    except Exception as e:
        logger.error("[api_subscription_status] Error: %s", e)
        return jsonify({"has_premium": False, "subscription_type": None, "error": str(e)}), 500


# ── Stripe Customer Portal ────────────────────────────────────────────────────

@billing_bp.route("/api/create-portal-session", methods=["POST"])
def api_create_portal_session():
    """Create a Stripe billing portal session so users can manage subscriptions."""
    from dashboard_services.subscriptions import get_subscription_info

    user_id   = session.get("viewer_user_id") or session.get("viewer_username")
    league_id = request.json.get("league_id") if request.is_json else request.form.get("league_id")
    platform  = "sleeper"

    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    try:
        sub_info    = get_subscription_info(user_id, league_id, platform)
        customer_id = sub_info.get("stripe_customer_id")
        # Fall back to user-only lookup (personal plan) if league lookup has no customer
        if not customer_id and league_id:
            user_sub    = get_subscription_info(user_id, None, platform)
            customer_id = user_sub.get("stripe_customer_id")
        if not customer_id:
            return jsonify({"error": "No Stripe customer found for your account. Contact support if you believe this is an error."}), 404

        return_url = request.json.get("return_url") if request.is_json else request.form.get("return_url")
        if not return_url:
            return_url = request.host_url.rstrip("/") + "/pricing"

        portal_session = _stripe().billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )
        return jsonify({"url": portal_session.url})
    except Exception as e:
        logger.exception("[api_create_portal_session] Error: %s", e)
        return jsonify({"error": str(e)}), 500
