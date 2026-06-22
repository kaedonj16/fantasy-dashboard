"""
Dynasty SEO hub pages: trade value chart, risers/fallers, rankings by position.
All pages are public (no league context required).
"""
from __future__ import annotations
import html
import re
from datetime import datetime

# Regex to detect raw pick IDs: "2027_5_early", "2027_1", "2028_3_mid", etc.
_PICK_ID_RE = re.compile(r'^\d{4}_\d+')
# Regex to detect formatted pick names: "2027 5th (Early)", "2027 1st", etc.
_PICK_NAME_RE = re.compile(r'^\d{4}\s+\d+(st|nd|rd|th)', re.IGNORECASE)
# Max pick round to show (rounds 4+ are filtered out even if position isn't "PICK")
_MAX_PICK_ROUND = 3


def _pick_round(p: dict) -> int | None:
    """Return pick round from player_id or formatted name, or None if not a pick."""
    pid  = str(p.get("player_id") or "")
    name = p.get("name") or ""
    m = _PICK_ID_RE.match(pid)
    if m:
        try:
            return int(pid.split("_")[1])
        except (IndexError, ValueError):
            pass
    m2 = _PICK_NAME_RE.match(name)
    if m2:
        try:
            return int(re.split(r'\s+', name)[1])
        except (IndexError, ValueError):
            pass
    return None


def _should_skip_mover(p: dict) -> bool:
    """Return True if this mover entry should be hidden from the risers/fallers page."""
    pos  = (p.get("position") or "").upper()
    name = p.get("name") or ""
    # Explicit skip positions
    if pos in ("PICK", "K", "DEF"):
        return True
    # Fallback "Player {raw_id}" names — data never resolved a real name
    if name.startswith("Player ") and "_" in name:
        return True
    # Raw pick IDs stored as names (e.g. "2027_5_early" or "2027 5th (Early)")
    if _PICK_ID_RE.match(name) or _PICK_NAME_RE.match(name):
        rd = _pick_round(p)
        return rd is None or rd > _MAX_PICK_ROUND
    return False

_POS_COLOR = {
    "QB": "#f59e0b", "RB": "#22c55e", "WR": "#3b82f6",
    "TE": "#8b5cf6", "K": "#94a3b8", "DEF": "#64748b",
}

def _pc(pos: str) -> str:
    return _POS_COLOR.get((pos or "").upper(), "#94a3b8")


def _rank_arrow(change: int | None) -> str:
    if not change:
        return '<span class="dvt-change dvt-change-flat">-</span>'
    if change > 0:
        return f'<span class="dvt-change dvt-change-up">&#9650; {change}</span>'
    return f'<span class="dvt-change dvt-change-down">&#9660; {abs(change)}</span>'


def _tier_label(value: float) -> str:
    if value >= 850: return "ELITE"
    if value >= 650: return "GREAT"
    if value >= 400: return "GOOD"
    if value >= 150: return "SOLID"
    return "DEPTH"


def _val_color(value: float) -> str:
    if value >= 850: return "#38bdf8"
    if value >= 650: return "#a78bfa"
    if value >= 400: return "#e2e8f0"
    return "#94a3b8"


# ── Dynasty Trade Value Chart ─────────────────────────────────────────────────

def build_dynasty_value_chart_body(value_table: list[dict], as_of_date: str | None = None) -> str:
    """Full-page dynasty trade value chart table."""
    from dashboard_services.pages.player_page import slugify

    date_str = as_of_date or datetime.now().strftime("%B %Y")

    rows = [
        r for r in value_table
        if (r.get("position") or "").upper() not in ("PICK", "K", "DEF")
        and float(r.get("value") or 0) > 0
    ]
    rows.sort(key=lambda r: float(r.get("value") or 0), reverse=True)

    positions = ["All", "QB", "RB", "WR", "TE"]
    pos_btns = "".join(
        f'<button type="button" class="dvt-pos-btn{"  dvt-pos-btn-active" if p == "All" else ""}" '
        f'data-pos="{p}">{p}</button>'
        for p in positions
    )

    # Tier divider tracking
    TIERS = [
        (850, "ELITE"),
        (650, "GREAT"),
        (400, "GOOD"),
        (150, "SOLID"),
        (0,   "DEPTH"),
    ]
    last_tier = None

    table_rows = ""
    for rank, row in enumerate(rows, 1):
        name   = row.get("name") or "Unknown"
        pos    = (row.get("position") or "").upper()
        team   = row.get("team") or ""
        age    = row.get("age")
        val    = float(row.get("value") or 0)
        sf_val = float(row.get("sf_value") or 0)
        plabel = row.get("pos_rank_label") or ""
        change = row.get("rank_change_7d")
        slug   = slugify(name)

        tier = _tier_label(val)
        if tier != last_tier:
            table_rows += (
                f'<tr class="dvt-tier-divider" data-pos="ALL">'
                f'<td colspan="7"><span class="dvt-tier-label dvt-tier-{tier.lower()}">{tier}</span></td>'
                f'</tr>'
            )
            last_tier = tier

        val_color  = _val_color(val)
        sf_color   = _val_color(sf_val)
        pos_border = _pc(pos)

        table_rows += (
            f'<tr data-pos="{html.escape(pos)}" class="dvt-row">'
            f'<td class="dvt-rank">{rank}</td>'
            f'<td class="dvt-name-cell" style="--pos-color:{pos_border};">'
            f'<div class="dvt-name-inner">'
            f'<span class="dvt-pos-badge" style="background:{pos_border};">{html.escape(pos)}</span>'
            f'<a class="dvt-player-link" href="/player/{slug}/trade-value">{html.escape(name)}</a>'
            f'<span class="dvt-team">{html.escape(team)}</span>'
            f'</div>'
            f'</td>'
            f'<td class="dvt-val" style="color:{val_color};">{val:.0f}</td>'
            f'<td class="dvt-val dvt-sf" style="color:{sf_color};">{sf_val:.0f}</td>'
            f'<td class="dvt-pos-rank">{html.escape(plabel)}</td>'
            f'<td class="dvt-age">{age or "-"}</td>'
            f'<td>{_rank_arrow(change)}</td>'
            f'</tr>'
        )

    player_count = len(rows)

    return f"""
<div class="dvt-page">
  <div class="dvt-hero">
    <h1 class="dvt-title">Dynasty Fantasy Football Trade Value Chart</h1>
    <p class="dvt-subtitle">
      Updated {html.escape(date_str)}: real dynasty trade values for 1QB and Superflex leagues.
      Use these values in the <a href="/trade">Trade Calculator</a> to evaluate any deal.
    </p>
    <div class="dvt-stat-pills">
      <span class="dvt-stat-pill"><strong>{player_count}</strong> players</span>
      <span class="dvt-stat-pill">Updated weekly</span>
      <span class="dvt-stat-pill">1QB &amp; Superflex</span>
    </div>
  </div>

  <div class="dvt-controls">
    <div class="dvt-pos-filter">{pos_btns}</div>
    <input type="search" class="dvt-search" id="dvtSearch" placeholder="Search player...">
  </div>

  <div class="dvt-table-wrap">
    <table class="dvt-table" id="dvtTable">
      <thead>
        <tr>
          <th class="dvt-rank">#</th>
          <th>Player</th>
          <th class="dvt-val" title="1QB Dynasty Value">1QB</th>
          <th class="dvt-val" title="Superflex Dynasty Value">SF</th>
          <th class="dvt-pos-rank">Pos Rank</th>
          <th class="dvt-age">Age</th>
          <th title="7-Day Rank Change">7d</th>
        </tr>
      </thead>
      <tbody id="dvtBody">
        {table_rows}
      </tbody>
    </table>
  </div>

  <div class="dvt-seo-content">
    <h2>How to Use Dynasty Trade Values</h2>
    <p>Dynasty fantasy football trade values represent what a player is worth in a trade
    based on real transaction data and statistical modeling. Unlike redraft, dynasty values
    factor in age, positional scarcity, and long-term production curves.</p>

    <h2>1QB vs Superflex Dynasty Values</h2>
    <p>In Superflex leagues, QBs are significantly more valuable because a second quarterback
    can start. This inflates QB values by 20&ndash;40% compared to standard 1QB formats.
    Use the SF column when evaluating trades in your Superflex league.</p>

    <h2>How Often Do Values Update?</h2>
    <p>BR Fantasy updates dynasty trade values regularly using a hybrid model that combines
    real trade data from thousands of dynasty leagues, advanced analytics, and market
    consensus. Values shift based on performance, injuries, and trade volume.</p>
  </div>
</div>

<script>
(function() {{
  var search = document.getElementById("dvtSearch");
  var body   = document.getElementById("dvtBody");
  var rows   = Array.from(body ? body.querySelectorAll("tr.dvt-row") : []);
  var dividers = Array.from(body ? body.querySelectorAll("tr.dvt-tier-divider") : []);
  var activePos = "All";

  function filter() {{
    var q = (search ? search.value : "").toLowerCase().trim();
    var visibleByTier = {{}};

    rows.forEach(function(r) {{
      var pos  = r.dataset.pos || "";
      var link = r.querySelector(".dvt-player-link");
      var name = link ? link.textContent.toLowerCase() : "";
      var posOk  = activePos === "All" || pos === activePos;
      var nameOk = !q || name.indexOf(q) !== -1;
      var show   = posOk && nameOk;
      r.style.display = show ? "" : "none";

      // Track which tiers have visible rows (for divider visibility)
      if (show) {{
        var prev = r;
        while ((prev = prev.previousElementSibling)) {{
          if (prev.classList.contains("dvt-tier-divider")) {{
            visibleByTier[prev.dataset.tier || prev.querySelector(".dvt-tier-label")?.textContent] = true;
            break;
          }}
        }}
      }}
    }});

    // Show tier dividers only when filtering by position (not text search)
    dividers.forEach(function(d) {{ d.style.display = q ? "none" : ""; }});
  }}

  if (search) search.addEventListener("input", filter);

  document.querySelectorAll(".dvt-pos-btn").forEach(function(btn) {{
    btn.addEventListener("click", function() {{
      document.querySelectorAll(".dvt-pos-btn").forEach(function(b) {{
        b.classList.remove("dvt-pos-btn-active");
      }});
      btn.classList.add("dvt-pos-btn-active");
      activePos = btn.dataset.pos;
      filter();
    }});
  }});
}})();
</script>
"""


# ── Risers & Fallers ──────────────────────────────────────────────────────────

def build_risers_fallers_body(movers: dict, as_of_date: str | None = None) -> str:
    """Weekly risers and fallers page."""
    from dashboard_services.pages.player_page import slugify

    date_str    = as_of_date or datetime.now().strftime("%B %d, %Y")
    latest_date = movers.get("latest_date", "")
    comp_date   = movers.get("comparison_date", "")
    risers      = [p for p in (movers.get("risers")  or []) if not _should_skip_mover(p)]
    fallers     = [p for p in (movers.get("fallers") or []) if not _should_skip_mover(p)]

    def _player_row(p: dict, direction: str) -> str:
        name   = p.get("name") or "Unknown"
        pos    = (p.get("position") or "").upper()
        team   = p.get("team") or ""
        val    = float(p.get("new_value") or p.get("value") or 0)
        change = float(p.get("change") or p.get("delta") or 0)
        slug   = slugify(name)
        sign   = "+" if change >= 0 else ""
        accent = _pc(pos)
        delta_color = "#22c55e" if direction == "riser" else "#ef4444"
        pct = abs(change / val * 100) if val else 0
        pct_str = f"{pct:.0f}%" if pct >= 1 else ""
        # Real players (not draft picks) carry data-player-id so the in-app player
        # modal opens for signed-in users; logged-out visitors follow the href to
        # the public player page (handled by the global click handler in app.js).
        pid = str(p.get("player_id") or "")
        data_attrs = ""
        if pid and pos != "PICK" and "_" not in pid:
            data_attrs = (
                f' data-player-id="{html.escape(pid, quote=True)}"'
                f' data-player-name="{html.escape(name, quote=True)}"'
            )
        return (
            f'<div class="rf-row" style="--pos-accent:{accent};">'
            f'<div class="rf-left-bar"></div>'
            f'<div class="rf-info">'
            f'<span class="rf-pos-badge" style="background:{accent};">{html.escape(pos)}</span>'
            f'<a class="rf-name" href="/player/{slug}/trade-value"{data_attrs}>{html.escape(name)}</a>'
            f'<span class="rf-team">{html.escape(team)}</span>'
            f'</div>'
            f'<div class="rf-val-wrap">'
            f'<span class="rf-val">{val:.0f}</span>'
            f'<span class="rf-delta" style="color:{delta_color};">{sign}{change:.0f}'
            f'{f" <small>({pct_str})</small>" if pct_str else ""}</span>'
            f'</div>'
            f'</div>'
        )

    risers_html  = "".join(_player_row(p, "riser")  for p in risers)
    fallers_html = "".join(_player_row(p, "faller") for p in fallers)

    if not risers_html:
        risers_html  = '<p class="rf-empty">No data available yet.</p>'
    if not fallers_html:
        fallers_html = '<p class="rf-empty">No data available yet.</p>'

    range_note = (
        f"Comparing {html.escape(str(comp_date))} to {html.escape(str(latest_date))}"
        if comp_date and latest_date else html.escape(date_str)
    )

    return f"""
<div class="rf-page">
  <div class="rf-hero">
    <h1 class="rf-title">Dynasty Fantasy Football Top Movers</h1>
    <p class="rf-subtitle">
      Biggest dynasty trade value movers this week, {range_note}.
      Use the <a href="/trade">Trade Calculator</a> to act on these moves.
    </p>
  </div>

  <div class="rf-grid">
    <div class="rf-col">
      <h2 class="rf-col-title rf-col-title-up">
        <i class="fa-solid fa-arrow-trend-up"></i> Top Risers
      </h2>
      <div class="rf-list">{risers_html}</div>
    </div>
    <div class="rf-col">
      <h2 class="rf-col-title rf-col-title-down">
        <i class="fa-solid fa-arrow-trend-down"></i> Top Fallers
      </h2>
      <div class="rf-list">{fallers_html}</div>
    </div>
  </div>

  <div class="rf-seo-content">
    <h2>Why Do Dynasty Values Change?</h2>
    <p>Dynasty trade values shift based on real transaction data from thousands of leagues,
    injury reports, depth chart changes, and model updates. Big risers typically follow
    breakout performances or positive depth chart news; big fallers often reflect injuries
    or increased competition.</p>

    <h2>How to Use Top Movers</h2>
    <p>Risers are buy candidates, their real market value is rising but roster holders
    may not have adjusted their asking price yet. Fallers are sell candidates for the same
    reason. Use the <a href="/trade">BR Fantasy Trade Calculator</a> to turn this intel
    into winning trades in your dynasty league.</p>
  </div>
</div>
"""


# ── Rankings Hub ──────────────────────────────────────────────────────────────

def build_rankings_hub_body(
    value_table: list[dict],
    position: str | None = None,
    as_of_date: str | None = None,
) -> str:
    """Dynasty rankings hub page, optionally filtered to a single position."""
    from dashboard_services.pages.player_page import slugify

    date_str   = as_of_date or datetime.now().strftime("%B %Y")
    pos_filter = (position or "").upper() if position else None

    rows = [
        r for r in value_table
        if (r.get("position") or "").upper() not in ("PICK", "K", "DEF")
        and float(r.get("value") or 0) > 0
        and (not pos_filter or (r.get("position") or "").upper() == pos_filter)
    ]
    rows.sort(key=lambda r: float(r.get("value") or 0), reverse=True)

    if pos_filter:
        title    = f"Dynasty {pos_filter} Rankings {datetime.now().year}"
        subtitle = (
            f"Top dynasty {pos_filter} trade values updated {date_str}. "
            f"Rankings based on real dynasty trade data and BR Fantasy model."
        )
    else:
        title    = f"Dynasty Fantasy Football Rankings {datetime.now().year}"
        subtitle = (
            f"Overall dynasty player rankings updated {date_str}. "
            f"Values from real dynasty leagues and BR Fantasy model analysis."
        )

    pos_nav_items = [
        ("All",  "/rankings/dynasty"),
        ("QB",   "/rankings/dynasty-qb"),
        ("RB",   "/rankings/dynasty-rb"),
        ("WR",   "/rankings/dynasty-wr"),
        ("TE",   "/rankings/dynasty-te"),
    ]
    active_path = f"/rankings/dynasty-{pos_filter.lower()}" if pos_filter else "/rankings/dynasty"
    nav_html = "".join(
        f'<a class="rnk-nav-btn{" rnk-nav-btn-active" if path == active_path else ""}" href="{path}">{label}</a>'
        for label, path in pos_nav_items
    )

    table_rows = ""
    for rank, row in enumerate(rows, 1):
        name   = row.get("name") or "Unknown"
        pos    = (row.get("position") or "").upper()
        team   = row.get("team") or ""
        age    = row.get("age")
        val    = float(row.get("value") or 0)
        sf_val = float(row.get("sf_value") or 0)
        change = row.get("rank_change_7d")
        slug   = slugify(name)

        pos_badge = (
            "" if pos_filter
            else f'<span class="rnk-pos-badge" style="background:{_pc(pos)};">{html.escape(pos)}</span>'
        )

        table_rows += (
            f'<tr>'
            f'<td class="rnk-rank">{rank}</td>'
            f'<td class="rnk-name-cell">'
            f'{pos_badge}'
            f'<a class="rnk-player-link" href="/player/{slug}/trade-value">{html.escape(name)}</a>'
            f'<span class="rnk-team">{html.escape(team)}</span>'
            f'</td>'
            f'<td class="rnk-val" style="color:{_val_color(val)};">{val:.0f}</td>'
            f'<td class="rnk-val" style="color:{_val_color(sf_val)};">{sf_val:.0f}</td>'
            f'<td class="rnk-age">{age or "-"}</td>'
            f'<td>{_rank_arrow(change)}</td>'
            f'</tr>'
        )

    return f"""
<div class="rnk-page">
  <div class="rnk-hero">
    <h1 class="rnk-title">{html.escape(title)}</h1>
    <p class="rnk-subtitle">{html.escape(subtitle)}</p>
  </div>

  <div class="rnk-pos-nav">{nav_html}</div>

  <div class="rnk-table-wrap">
    <table class="rnk-table">
      <thead>
        <tr>
          <th class="rnk-rank">#</th>
          <th>Player</th>
          <th class="rnk-val">1QB</th>
          <th class="rnk-val">SF</th>
          <th class="rnk-age">Age</th>
          <th>7d</th>
        </tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
  </div>
</div>
"""
