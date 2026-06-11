"""
Per-player trade-value pages (SEO landing pages).

These are crawlable, server-rendered pages at /player/<slug>/trade-value that
mirror the look of the in-app player modal (same pm-* / player-modal-* CSS
classes) while rendering the core facts into the initial HTML so search engines
can index them. The interactive value-history chart hydrates client-side from
embedded JSON, and recent trades hydrate from the existing trade-intel endpoint.
"""
from __future__ import annotations

import html
import json
import re
from typing import Optional


# ── Slug helpers ──────────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    """Turn a player name into a URL slug: 'Justin Jefferson' -> 'justin-jefferson'."""
    s = (name or "").strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[‘’']", "", s)          # drop apostrophes (Ja'Marr -> jamarr)
    s = re.sub(r"[^a-z0-9]+", "-", s)              # non-alnum -> hyphen
    return s.strip("-")


def build_slug_index(value_table: list) -> dict:
    """Map slug -> player_id from the value table.

    Only real, tradeable players with a positive value are included (no picks,
    kickers or defenses). On a slug collision the higher-valued player wins, so
    the canonical page for a name points at the more relevant player.
    """
    index: dict = {}
    best_val: dict = {}
    for row in value_table or []:
        pid = str(row.get("id") or "").strip()
        name = str(row.get("name") or "").strip()
        pos = str(row.get("position") or "").upper()
        if not pid or not name or pos in ("PICK", "K", "DEF"):
            continue
        try:
            val = float(row.get("value") or 0)
        except (TypeError, ValueError):
            val = 0.0
        if val <= 0:
            continue
        slug = slugify(name)
        if not slug:
            continue
        if slug not in index or val > best_val.get(slug, 0):
            index[slug] = pid
            best_val[slug] = val
    return index


# ── Page rendering ────────────────────────────────────────────────────────────

def _fmt(v, dash: str = "-") -> str:
    if v is None:
        return dash
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else f"{f:.1f}"
    except (TypeError, ValueError):
        return str(v)


def _trend_text(value_history: list, name: str) -> tuple:
    """Return (trend_phrase, trend_class) summarizing 30-day movement."""
    pts = [h for h in (value_history or []) if h.get("value_1qb") is not None]
    if len(pts) < 2:
        return ("", "")
    latest = float(pts[-1].get("value_1qb") or 0)
    # ~30 days back (history is daily-ish); fall back to the oldest point.
    ref = pts[-31] if len(pts) > 31 else pts[0]
    start = float(ref.get("value_1qb") or 0)
    if start <= 0:
        return ("", "")
    delta = latest - start
    pct = (delta / start) * 100
    if abs(pct) < 2:
        return (f"{html.escape(name)}'s trade value has held steady over the past month.", "flat")
    direction = "risen" if delta > 0 else "fallen"
    cls = "up" if delta > 0 else "down"
    return (
        f"{html.escape(name)}'s trade value has {direction} "
        f"{abs(delta):.0f} points ({abs(pct):.0f}%) over the past month.",
        cls,
    )


def build_player_page_body(
        *,
        player_id: str,
        name: str,
        position: Optional[str],
        team: Optional[str],
        age,
        headshot: Optional[str],
        value_1qb,
        sf_value,
        pos_rank_label: Optional[str],
        ovr_rank,
        sf_pos_rank_label: Optional[str],
        sf_ovr_rank,
        ppg,
        value_history: list,
        season: int,
) -> str:
    """Server-rendered body for a player's trade-value page."""
    esc_name = html.escape(name or "Player")
    pos = (position or "").upper()
    team_u = (team or "").upper()

    meta_bits = []
    if pos:
        meta_bits.append(pos)
    if team_u:
        meta_bits.append(team_u)
    if age:
        meta_bits.append(f"Age {age}")
    if pos_rank_label:
        meta_bits.append(html.escape(str(pos_rank_label)))
    meta_line = "  &middot;  ".join(meta_bits)

    headshot_html = (
        f'<img class="player-modal-headshot" src="{html.escape(headshot)}" '
        f'alt="{esc_name}" loading="lazy" '
        f'onerror="this.style.display=\'none\'">'
        if headshot else ""
    )

    # Hero value cards (same classes as the modal overview)
    ppg_card = ""
    if ppg:
        ppg_card = f"""
          <div class="pm-hero-stat">
            <div class="pm-hero-label">PPG</div>
            <div class="pm-hero-val">{_fmt(ppg)}</div>
            <div class="pm-hero-sub">{season} season</div>
          </div>"""
    hero_count = 2 + (1 if ppg_card else 0)
    val1qb_sub = (
        f"POS : {html.escape(str(pos_rank_label))} &middot; OVR : {_fmt(ovr_rank, '-')}"
        if pos_rank_label else "-"
    )
    sf_sub = (
        f"POS : {html.escape(str(sf_pos_rank_label))} &middot; OVR : {_fmt(sf_ovr_rank, '-')}"
        if sf_pos_rank_label else "-"
    )

    # Prose summary + trend (crawlable SEO content)
    trend_phrase, trend_cls = _trend_text(value_history, name)
    summary_parts = []
    subj = esc_name
    if age and pos and team_u:
        summary_parts.append(f"{subj} is a {html.escape(str(age))}-year-old {pos} for the {team_u}.")
    elif pos:
        summary_parts.append(f"{subj} is a {pos} in the NFL.")
    if value_1qb:
        rank_phrase = f", ranked {html.escape(str(pos_rank_label))}" if pos_rank_label else ""
        summary_parts.append(
            f"In {season} dynasty leagues, {subj}'s trade value is "
            f"<strong>{_fmt(value_1qb)}</strong>{rank_phrase}."
        )
    if sf_value:
        summary_parts.append(
            f"In superflex formats {subj} is worth <strong>{_fmt(sf_value)}</strong>."
        )
    if trend_phrase:
        summary_parts.append(trend_phrase)
    summary_parts.append(
        f"Use the trade calculator below to see exactly what {subj} is worth in "
        f"your league's scoring and roster settings."
    )
    summary_html = " ".join(summary_parts)

    # Chart (hydrated client-side from embedded JSON)
    history_json = json.dumps([
        {
            "as_of_date": h.get("as_of_date"),
            "value_1qb": h.get("value_1qb"),
            "value_sf": h.get("value_sf"),
        }
        for h in (value_history or [])
        if h.get("value_1qb") is not None
    ])
    has_chart = bool(value_history and len(value_history) > 1)
    chart_html = ""
    if has_chart:
        chart_html = f"""
          <hr class="pm-section-divider">
          <div class="pm-section-header"><span class="pm-section-label">Value History</span></div>
          <div class="player-modal-chart-container" id="ppValueChart" style="min-height:240px;"></div>
        """

    trade_value_label = "Dynasty Trade Value"
    return f"""
    <div class="page-shell-narrow" style="max-width:760px;margin:0 auto;">
      <nav class="pp-breadcrumb" aria-label="Breadcrumb" style="font-size:12px;color:var(--text-muted);margin-bottom:12px;">
        <a href="/players" style="color:var(--text-muted);">Player Rankings</a>
        <span style="margin:0 6px;">/</span>
        <span>{esc_name} Trade Value</span>
      </nav>

      <div class="card central" style="max-width:760px;">
        <div class="player-modal-header" style="border-bottom:1px solid var(--border);">
          <div class="player-modal-headshot-container">{headshot_html}</div>
          <div class="player-modal-title-section">
            <div class="player-modal-title-text">
              <h1 class="player-modal-name">{esc_name}</h1>
              <div class="player-modal-meta">{meta_line}</div>
            </div>
          </div>
          <div style="flex-shrink:0;">
            <a href="/trade" class="otc-btn otc-btn-primary" style="text-decoration:none;white-space:nowrap;">Open Trade Calculator</a>
          </div>
        </div>

        <div class="player-modal-body">
          <h2 style="font-size:16px;margin:0 0 14px;">{esc_name} {trade_value_label} ({season})</h2>

          <div class="pm-hero-row" style="grid-template-columns:repeat({hero_count},1fr);">
            <div class="pm-hero-stat pm-hero-primary">
              <div class="pm-hero-label">1QB Value</div>
              <div class="pm-hero-val" style="color:#3b82f6;">{_fmt(value_1qb)}</div>
              <div class="pm-hero-sub">{val1qb_sub}</div>
            </div>
            <div class="pm-hero-stat">
              <div class="pm-hero-label">SF Value</div>
              <div class="pm-hero-val">{_fmt(sf_value)}</div>
              <div class="pm-hero-sub">{sf_sub}</div>
            </div>
            {ppg_card}
          </div>

          <p class="pp-summary" style="margin:18px 0 0;line-height:1.6;color:var(--text-muted);font-size:14px;">
            {summary_html}
          </p>

          {chart_html}

          <hr class="pm-section-divider">
          <div class="pm-section-header"><span class="pm-section-label">Recent Trades</span></div>
          <div id="ppRecentTrades" data-player-id="{html.escape(str(player_id))}" data-season="{season}"
               style="font-size:13px;color:var(--text-muted);padding:6px 0;">
            <div style="display:flex;align-items:center;gap:8px;">
              <div class="loading-spinner" style="width:14px;height:14px;flex-shrink:0;"></div>Loading recent trades&hellip;
            </div>
          </div>

          <hr class="pm-section-divider">
          <div class="pp-cta" style="text-align:center;padding:8px 0 4px;">
            <a href="/trade" class="otc-btn otc-btn-primary" style="text-decoration:none;display:inline-block;">
              Value {esc_name} in your league &rarr;
            </a>
          </div>
        </div>
      </div>
    </div>

    <script>
      window.__ppHistory = {history_json};
      window.__ppName = {json.dumps(name or "Player")};
    </script>
    <script src="/static/player_page.js?v={season}"></script>
    """
