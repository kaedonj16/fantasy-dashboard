"""
Dynasty SEO hub pages: trade value chart, risers/fallers, rankings by position.
All pages are public (no league context required).
"""
from __future__ import annotations
import logging
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
            logging.getLogger(__name__).debug("suppressed exception", exc_info=True)
    m2 = _PICK_NAME_RE.match(name)
    if m2:
        try:
            return int(re.split(r'\s+', name)[1])
        except (IndexError, ValueError):
            logging.getLogger(__name__).debug("suppressed exception", exc_info=True)
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

# Matches the canonical .pos-badge palette (dashboard.css) so a position is the
# same color on the top-movers page as everywhere else.
_POS_COLOR = {
    "QB": "#3b82f6", "RB": "#22c55e", "WR": "#f59e0b",
    "TE": "#8b5cf6", "K": "#c92c68", "DEF": "#475569",
}

def _pc(pos: str) -> str:
    return _POS_COLOR.get((pos or "").upper(), "#94a3b8")


def _rank_arrow(change: int | None) -> str:
    if not change:
        return '<span class="dvt-change dvt-change-flat">&#8212;</span>'
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
            f'<td class="dvt-age">{age or "&#8212;"}</td>'
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

def build_risers_fallers_body(movers: dict, as_of_date: str | None = None,
                              signed_in: bool = False, days: int = 7) -> str:
    """Weekly risers and fallers page.

    signed_in: when True, player names render as clickable spans that open the
    in-app player modal (via the global [data-player-id] handler) instead of
    anchors that navigate to the public player page. Guests keep the crawlable
    <a> link for SEO.

    days: active timeframe window (7/30/90). Drives the timeframe toggle so the
    selected range is highlighted; each option is a crawlable ?days= link.
    """
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
        delta_color = "var(--win)" if direction == "riser" else "var(--loss)"
        # Percent change is measured against the pre-move (baseline) value, not the
        # post-move value - dividing by the new value understates a rise and
        # overstates a drop. Use old_value when present, else back it out (new - delta).
        _old = p.get("old_value")
        baseline = float(_old) if _old not in (None, "") else (val - change)
        pct = abs(change / baseline * 100) if baseline else 0
        pct_str = f"{pct:.0f}%" if pct >= 1 else ""
        # Real players (not draft picks) carry data-player-id so the in-app player
        # modal opens for signed-in users; logged-out visitors follow the href to
        # the public player page (handled by the global click handler in app.js).
        pid = str(p.get("player_id") or "")
        is_real = bool(pid) and pos != "PICK" and "_" not in pid
        data_attrs = ""
        if is_real:
            data_attrs = (
                f' data-player-id="{html.escape(pid, quote=True)}"'
                f' data-player-name="{html.escape(name, quote=True)}"'
            )
        # Signed-in users get a click-to-open-modal span (no href to race with
        # the delegated handler); guests/picks keep the crawlable public link.
        if is_real and signed_in:
            name_el = (
                f'<span class="rf-name player-clickable" role="button" tabindex="0"'
                f'{data_attrs}>{html.escape(name)}</span>'
            )
        else:
            name_el = (
                f'<a class="rf-name" href="/player/{slug}/trade-value"{data_attrs}>'
                f'{html.escape(name)}</a>'
            )
        return (
            f'<div class="rf-row" style="--pos-accent:{accent};">'
            f'<div class="rf-left-bar"></div>'
            f'<div class="rf-info">'
            f'<span class="rf-pos-badge" style="background:{accent};">{html.escape(pos)}</span>'
            f'{name_el}'
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

    _rf_empty = (
        '<div class="empty-state is-compact">'
        '<span class="empty-state-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19V5"/>'
        '<path d="M4 19h16"/><path d="m7 14 3-3 3 2 4-5"/></svg></span>'
        '<p class="empty-state-msg">No movement to show yet — check back once values shift.</p>'
        '</div>'
    )
    if not risers_html:
        risers_html = _rf_empty
    if not fallers_html:
        fallers_html = _rf_empty

    range_note = (
        f"Comparing {html.escape(str(comp_date))} to {html.escape(str(latest_date))}"
        if comp_date and latest_date else html.escape(date_str)
    )

    _win_word = {7: "this week", 30: "over the last 30 days", 90: "over the last 90 days"}.get(days, "this week")

    def _tf_opt(d: int, lbl: str) -> str:
        active = d == days
        cls = "rf-tf-opt is-active" if active else "rf-tf-opt"
        aria = ' aria-current="true"' if active else ""
        return f'<a class="{cls}" href="/top-movers?days={d}"{aria}>{lbl}</a>'

    _toggle = "".join(_tf_opt(d, lbl) for d, lbl in ((7, "7d"), (30, "30d"), (90, "90d")))
    toggle_html = f'<div class="rf-timeframe" role="group" aria-label="Timeframe">{_toggle}</div>'

    return f"""
<div class="rf-page">
  <div class="rf-hero">
    <h1 class="rf-title">Dynasty Fantasy Football Top Movers</h1>
    <p class="rf-updated"><span class="rf-updated-dot"></span>Updated {html.escape(date_str)} · refreshed daily</p>
    <p class="rf-subtitle">
      Biggest dynasty trade value movers {_win_word}, {range_note}.
      Use the <a href="/trade">Trade Calculator</a> to act on these moves.
    </p>
    {toggle_html}
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


# ── Position analysis ─────────────────────────────────────────────────────────
# Unique editorial copy per position so /rankings/dynasty-{qb,rb,wr,te} are not
# near-duplicate templates (a thin-content signal). Rendered above the table.
# Evergreen framing — no player names or numbers that would age out.

_POSITION_ANALYSIS = {
    "QB": (
        "How to read dynasty QB value",
        "<p>Quarterback is the position where your league format matters most. In "
        "<strong>Superflex and 2QB leagues</strong>, where you can start two passers, "
        "quality quarterbacks are the most valuable assets in dynasty &mdash; the "
        "position is scarce and the points are enormous. In standard "
        "<strong>1QB leagues</strong> the calculus flips: you only need one, streaming is "
        "viable, and paying a premium for an elite passer is often a luxury rather than a "
        "necessity.</p>"
        "<p>Age curves are also gentler here than anywhere else on the field. Quarterbacks "
        "routinely produce into their mid-30s, so a proven starter holds dynasty value for "
        "far longer than a running back of the same age. That durability is why young, "
        "ascending quarterbacks with rushing upside command the highest prices: rushing "
        "production raises their weekly floor and stacks on top of passing points. When you "
        "read the values below, weight the format toggle first, then reward mobility and a "
        "secure starting job.</p>"
    ),
    "RB": (
        "How to read dynasty RB value",
        "<p>Running back is the most volatile asset in dynasty, because the position ages "
        "out first. Most backs peak in their early-to-mid 20s and can fall off sharply by "
        "their late 20s, so you're renting production rather than banking it. That's why the "
        "market pays enormous premiums for young, three-down workhorses who project to hold "
        "a bell-cow role &mdash; volume is king, and secure volume is rare.</p>"
        "<p>Receiving work is the tell that separates a durable dynasty RB from a "
        "replaceable one. A back who catches passes keeps scoring even when his team trails "
        "and the game turns pass-heavy &mdash; exactly the script in which a pure early-down "
        "runner disappears. The classic trap is the aging veteran coming off a monster year: "
        "his redraft rank looks great, but his dynasty window is short and the committee "
        "behind him is one draft pick away. If you're rebuilding, that veteran is your sell.</p>"
    ),
    "WR": (
        "How to read dynasty WR value",
        "<p>Wide receiver is the backbone of most dynasty rosters. Receivers take a year or "
        "two to develop but then hold value longer than any skill position except "
        "quarterback, often producing into their early 30s. That combination &mdash; long "
        "shelf life plus every-week target volume &mdash; makes ascending young receivers the "
        "safest premium assets in the format.</p>"
        "<p>The signal to trust is opportunity: target share and air yards tell you how "
        "central a receiver is to his offense, and they stabilize faster than touchdowns, "
        "which bounce around year to year. A young wideout locked into a heavy target role is "
        "worth more than an older one with a gaudy but touchdown-driven stat line. When you "
        "scan the values below, favor secure volume and a clear path to the ball over last "
        "season's scoring, which often regresses.</p>"
    ),
    "TE": (
        "How to read dynasty TE value",
        "<p>Tight end is the position of scarcity. A small handful of every-week difference-"
        "makers sit far above a long, replaceable middle, so the value curve is steep at the "
        "top and flat thereafter. In <strong>TE Premium</strong> scoring, which awards extra "
        "points per reception, that top tier is worth even more and the gap widens further.</p>"
        "<p>Tight ends are also the slowest to develop &mdash; many don't break out until "
        "their third season &mdash; but the elite ones then hold value for years. That makes "
        "the position a patience game: if you don't roster one of the difference-makers, "
        "chasing the muddled middle rarely pays, and streaming is often the smarter play. When "
        "you read the values below, decide first whether you're buying into the scarce top "
        "tier or punting the position, because the middle offers little edge either way.</p>"
    ),
    None: (
        "How dynasty rankings work",
        "<p>These rankings estimate <strong>trade value</strong>: what the rest of your "
        "league would give up to acquire a player, not how many points he'll score this week. "
        "That's why they never match a redraft list &mdash; dynasty prices in age, long-term "
        "outlook, and the runway a player has left. A younger ascending player will often "
        "out-rank an older one who scores more today.</p>"
        "<p>Values blend consensus market data from real dynasty trades with recent usage, "
        "the positional aging curve, and each player's situation, and they refresh daily. Use "
        'the position tabs above for a deeper read on each spot, keep the '
        '<a href="/guides/dynasty-trade-value">trade-value guide</a> handy, and take any deal '
        'to the free <a href="/trade">trade calculator</a> to see the net value.</p>'
    ),
}


def _position_analysis_html(pos_filter: str | None) -> str:
    entry = _POSITION_ANALYSIS.get(pos_filter) or _POSITION_ANALYSIS.get(None)
    heading, prose = entry
    return (
        '<section class="rnk-analysis">'
        f'<h2 class="rnk-analysis-title">{html.escape(heading)}</h2>'
        f'{prose}'
        "</section>"
    )


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
            f'<td class="rnk-age">{age or "&#8212;"}</td>'
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

  {_position_analysis_html(pos_filter)}

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
