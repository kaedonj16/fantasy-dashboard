from __future__ import annotations

from .config import MIN_SIGNAL_CONFIDENCE, START_SIT_BASE_THRESHOLD


def market_vs_projection(market_points, site_points, confidence) -> dict | None:
    if market_points is None or site_points is None or float(confidence or 0) < MIN_SIGNAL_CONFIDENCE:
        return None
    delta = round(float(market_points) - float(site_points), 1)
    threshold = START_SIT_BASE_THRESHOLD * (1.0 + max(0.0, 0.7 - float(confidence or 0)))
    label = "Market Bullish" if delta >= threshold else "Market Caution" if delta <= -threshold else "Market Aligned"
    return {"delta": delta, "label": label, "confidence": round(float(confidence), 2)}


def market_opportunity(market_points, site_points, confidence, rostered_pct=0) -> dict | None:
    signal = market_vs_projection(market_points, site_points, confidence)
    if not signal:
        return None
    availability = max(0.0, 1.0 - float(rostered_pct or 0) / 100.0)
    adjusted = signal["delta"] * float(confidence) * availability
    label = "High" if adjusted >= 2.5 else "Moderate" if adjusted >= 1.0 else "Low" if adjusted <= -1 else "Neutral"
    return {**signal, "label": label}
