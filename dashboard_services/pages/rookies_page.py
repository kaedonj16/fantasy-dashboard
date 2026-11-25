"""
Rookies page HTML builder.

Returns the full body HTML for the /<platform>/<season>/<league_id>/rookies route.
Data is loaded client-side from /api/prospects/rankings so the page is fast and
the filters/sorts are instant without round-trips. The top 50 rows are also
server-rendered into the initial HTML (see the SSR section below) so search
engines and link previews see real names and grades; the client replaces
them with the full board on load.

Display model (2026-10-17): the board and modal show NO numeric grade.
They show rank (ordered by the internal grade), a band tag, and component
percentiles computed within position against the graded reference
population.

Layout: the board uses the same grammar as the site's other ranking
pages (card shell, two-row filter bar, pr-grid rows, tier dividers,
quiet header row, tinted tag chips). The modal is a single-scroll
report card (Kaedon-approved mock, 2026-11-24): a navy grade header,
an archetype band, then a rail (comparable players + class standing)
beside the main column (production-by-year chart, season lines,
grouped metric breakdown, numbered scouting notes). Band and
percentile colors use the BR Fantasy palette only: navy #122d4b and
light blue (#38bdf8 / #60a5fa) with slate for muted text; no amber,
green or purple band/bar colors. (The heavier Direction B board
treatment shipped 2026-11-22 was walked back the same day at Kaedon's
direction: it did not match the rest of the site.)
"""
from __future__ import annotations

import time as _time

# ── Server-rendered board rows (SEO shell) ──────────────────────────────────
# The board below is client-rendered from /api/prospects/rankings, which
# left the served HTML an empty shell: no prospect names for search
# engines or link previews. The top rows are now also rendered on the
# server, in the exact markup the client's rkBoardRow/rkBandDivider
# produce, from the same enriched rows the API serves (rookie_api's
# get_board_rows). On load the client fetches the full board and
# replaces #rkList wholesale, so the static rows are the initial paint
# only: no duplicate table, and sorting/filters/modal are untouched.
# Fail-soft: any data error renders the shell exactly as before (the
# loading skeleton), and a short TTL cache keeps page renders cheap.

_SSR_LIMIT = 50
_SSR_TTL_SECONDS = 600
_ssr_cache: dict = {}

_RK_BAND_COLORS = {"elite": "var(--accent)", "great": "#0ea5e9",
                   "average": "#64748b", "day3": "#64748b"}
_RK_BAND_SHORT = {"elite": "Elite", "great": "Great",
                  "average": "Average", "day3": "Day 3"}
_RK_PCT_CELLS = (("production", ""), ("efficiency", "pr-eff"),
                 ("breakout", "pr-brk"))


def _rk_esc(s) -> str:
    """rkEsc parity: escape & < > and double quotes."""
    return (str("" if s is None else s)
            .replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _rk_pct(row: dict, key: str):
    p = (row.get("percentiles") or {}).get(key)
    if p is None or p == "":
        return None
    try:
        return float(p)
    except (TypeError, ValueError):
        return None


def _rk_stat_cell(row: dict, key: str, extra_cls: str) -> str:
    p = _rk_pct(row, key)
    if p is None:
        return f'<span class="rk-pct rk-pct-dash {extra_cls}">&ndash;</span>'
    txt = str(int(p)) if p.is_integer() else str(p)
    return f'<span class="rk-pct {extra_cls}">{txt}</span>'


def _rk_grade_cell(row: dict) -> str:
    s = row.get("prospect_score")
    try:
        f = float(s) if s not in (None, "") else float("nan")
    except (TypeError, ValueError):
        f = float("nan")
    if f != f:  # NaN: muted dash, never a placeholder number
        return '<span class="pr-grade none">-</span>'
    return f'<span class="pr-grade">{f:.1f}</span>'


def _rk_board_row_html(r: dict) -> str:
    """One board row, mirroring the client's rkBoardRow markup.

    The watchlist star is the only client-only piece (it reads
    localStorage); the client render adds it back on load.
    """
    pid = r.get("player_id") or ""
    pos = str(r.get("position") or "").upper()
    rank_txt = "-"
    rank = r.get("overall_rank")
    if rank not in (None, ""):
        try:
            rank_txt = str(int(float(rank)))
        except (TypeError, ValueError):
            pass
    delta = ""
    d = r.get("rank_delta")
    if d not in (None, ""):
        try:
            di = int(float(d))
        except (TypeError, ValueError):
            di = 0
        if di != 0:
            delta = (
                f'<span class="pr-rank-delta {"up" if di > 0 else "down"}" '
                f'title="{abs(di)} spot{"s" if abs(di) != 1 else ""} overall">'
                f'{"&#9650;" if di > 0 else "&#9660;"}{abs(di)}</span>'
            )
    pos_rank = f"{pos}{r.get('position_rank')}" if pos and r.get("position_rank") else ""
    school = r.get("school") or ""
    if pos_rank and school:
        meta = f"{_rk_esc(pos_rank)} &middot; {_rk_esc(school)}"
    elif pos_rank:
        meta = _rk_esc(pos_rank)
    elif school:
        meta = _rk_esc(school)
    else:
        meta = ""
    badge = (f'<span class="pos-badge {_rk_esc(pos)}">{_rk_esc(pos)}</span>'
             if pos else "")
    meta_html = f'<span class="rk-meta">{meta}</span>' if meta else ""
    cells = "".join(_rk_stat_cell(r, key, cls) for key, cls in _RK_PCT_CELLS)
    return (
        f'<div class="pr-player-row pr-grid-row" data-pid="{_rk_esc(pid)}" '
        f'role="button" tabindex="0" aria-label="{_rk_esc(r.get("name") or "")}">'
        f'<span class="pr-rank">#{rank_txt}{delta}</span>'
        f'<span class="rk-player"><span class="pr-name">'
        f'<span class="pr-nm">{_rk_esc(r.get("name") or "Unknown")}</span>'
        f'{badge}</span>{meta_html}</span>'
        f'{cells}{_rk_grade_cell(r)}</div>'
    )


def _rk_band_divider_html(band: str) -> str:
    color = _RK_BAND_COLORS.get(band, "#94a3b8")
    label = _RK_BAND_SHORT.get(band, "")
    return (
        f'<div class="pr-tier-divider">'
        f'<div class="pr-tier-divider-line" style="background:{color}"></div>'
        f'<span class="pr-tier-divider-label" style="color:{color}">'
        f'{_rk_esc(label)}</span>'
        f'<div class="pr-tier-divider-line" style="background:{color}"></div>'
        f'</div>'
    )


def _prospects_ssr():
    """(rows_html, shown, total) for the initial server render.

    Default board order, mirroring rkVisiblePlayers: overall rank
    ascending, unranked last, band dividers where the band changes.
    ("", 0, 0) when no board can be built (no data yet, DB down).
    """
    now = _time.time()
    cached = _ssr_cache.get("v")
    if cached and now - cached[3] < _SSR_TTL_SECONDS:
        return cached[0], cached[1], cached[2]
    rows_html, shown, total = "", 0, 0
    try:
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class
        from dashboard_services.rookie_api import get_board_rows

        rows = get_board_rows(get_active_rookie_class())
        total = len(rows)

        def _rank_key(r):
            v = r.get("overall_rank")
            if v in (None, ""):
                return (1, 0.0)
            try:
                return (0, float(v))
            except (TypeError, ValueError):
                return (1, 0.0)

        parts, last_band = [], None
        for r in sorted(rows, key=_rank_key)[:_SSR_LIMIT]:
            band = (r.get("tag") or {}).get("band")
            if band and band != last_band:
                parts.append(_rk_band_divider_html(band))
                last_band = band
            parts.append(_rk_board_row_html(r))
        rows_html = "".join(parts)
        shown = min(total, _SSR_LIMIT)
    except Exception:
        rows_html, shown, total = "", 0, 0
    if rows_html:
        _ssr_cache["v"] = (rows_html, shown, total, now)
    return rows_html, shown, total


def _inject_ssr_rows(body: str) -> str:
    """Swap the shell's empty board for the server-rendered top rows.

    With rows present, the column header and count line render visible
    (the states the client sets after load) and the loading skeleton is
    hidden; the client still replaces #rkList with the full board when
    its fetch lands. Without rows the body is returned untouched.
    """
    rows_html, shown, total = _prospects_ssr()
    if not rows_html:
        return body
    body = body.replace(
        '<div id="rkTableHeader" class="pr-grid-row" style="display:none">',
        '<div id="rkTableHeader" class="pr-grid-row">', 1)
    body = body.replace(
        '<div id="rkCount" style="display:none;"></div>',
        f'<div id="rkCount">Showing {shown} of {total} prospects</div>', 1)
    body = body.replace(
        '<div id="rkLoading" class="sk-list" style="margin-top:4px;">',
        '<div id="rkLoading" class="sk-list" style="display:none;margin-top:4px;">', 1)
    body = body.replace(
        '<div id="rkList"></div>',
        f'<div id="rkList">{rows_html}</div>', 1)
    return body


def _rk_class_options() -> str:
    """Class selector options: All, then every class from the active
    class down to 2016 (the archive's first class). The active class
    is the default selection; the board's first paint is always the
    active class."""
    try:
        from data_building.rookie_pipeline.pipeline import (
            get_active_rookie_class)
        active = int(get_active_rookie_class())
    except Exception:
        import datetime as _dt
        active = _dt.datetime.now().year
    opts = ['<option value="all">All</option>']
    for y in range(active, 2015, -1):
        sel = ' selected' if y == active else ''
        opts.append(f'<option value="{y}"{sel}>{y}</option>')
    return "".join(opts)


def build_prospects_body(is_admin: bool = False) -> str:
    admin_flag = "true" if is_admin else "false"
    body = f"""<script>window.RK_IS_ADMIN = {admin_flag};</script>""" + """
<div class="card central" id="prospectsCard">
  <div class="card-header" style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
    <div>
      <h1 id="rookiesTitle">Rookie Prospects</h1>
      <div style="font-size: 15px; color: var(--text-muted); margin-top: 4px;">
        <span id="rkPoolLine">Dynasty prospect rankings - production, athleticism, and draft capital combined</span>
      </div>
      <div id="rkPausedBanner" style="display:none;margin-top:10px;padding:8px 12px;border-radius:8px;
           border:1px solid color-mix(in srgb,#b45309 35%,var(--border));background:color-mix(in srgb,#b45309 12%,transparent);
           font-size:13px;color:var(--text);">
        Prospect rankings are paused until the next draft cycle. Showing the last published board
        <span id="rkPausedAsOf"></span>.
      </div>
    </div>
    <button type="button" class="btn btn-ghost btn-sm" onclick="rkExportCSV()" title="Export CSV" aria-label="Export CSV"><img src="/static/images/download-solid.png" alt="" style="width:14px;height:14px;vertical-align:-2px;opacity:.75;"> Export</button>
  </div>

  <!-- Rankings panel -->
  <div id="rk-panel-rankings" class="card-body" style="padding-top:0;">

    <!-- Controls -->
    <div class="filter-controls-container">
      <!-- Row 1: Search + pills -->
      <div class="filter-row filter-row-primary">
        <div class="filter-search">
          <input id="rookieSearch" type="text" placeholder="Search prospects…" autocomplete="off">
          <span class="mag"><i class="fa-solid fa-magnifying-glass" aria-hidden="true"></i></span>
          <button id="rookieSearchClear" onclick="rkClearSearch()" aria-label="Clear search"
            style="display:none;position:absolute;right:8px;top:50%;transform:translateY(-50%);
                   background:none;border:none;cursor:pointer;color:var(--text-muted);
                   font-size:15px;padding:2px;">&#x2715;</button>
        </div>
        <div class="otc-day-filters filter-positions">
          <button class="otc-day-filter pos-pill active" data-pos="ALL" onclick="rkTogglePos('ALL')">All</button>
          <button class="otc-day-filter pos-pill" data-pos="QB"  onclick="rkTogglePos('QB')">QB</button>
          <button class="otc-day-filter pos-pill" data-pos="RB"  onclick="rkTogglePos('RB')">RB</button>
          <button class="otc-day-filter pos-pill" data-pos="WR"  onclick="rkTogglePos('WR')">WR</button>
          <button class="otc-day-filter pos-pill" data-pos="TE"  onclick="rkTogglePos('TE')">TE</button>
        </div>
      </div>

      <!-- Row 2: Active setting tags + sort -->
      <div class="filter-row filter-row-secondary">
        <div class="active-settings-indicator" id="rkActiveSettings" style="display:none">
          <span class="active-setting-tag" id="rkClassTag"></span>
          <span class="active-setting-tag" id="rkSeasonTag" style="display:none">In-Season</span>
        </div>
        <div class="filter-sort-group">
          <div class="filter-sort">
            <label class="filter-label" for="rkClass">Class</label>
            <select id="rkClass" onchange="rkSetClass(this.value)"><!--RK_CLASS_OPTIONS--></select>
          </div>
          <div class="filter-sort">
            <label class="filter-label" for="rkSort">Sort</label>
            <select id="rkSort" onchange="rkSetSortKey(this.value)">
              <option value="rank">Rank</option>
              <option value="prod">Production %</option>
              <option value="eff">Efficiency %</option>
              <option value="brk">Breakout %</option>
              <option value="name">Name</option>
            </select>
          </div>
        </div>
      </div>
    </div>

    <!-- Past-class note: archive views only (shown by rkApplyBoard) -->
    <div id="rkArchiveNote" style="display:none;font-size:12px;color:var(--text-muted);margin:0 0 10px;">
      Past classes graded with the current model. Not the board as it stood on draft day.
    </div>

    <!-- Count -->
    <div id="rkCount" style="display:none;"></div>

    <!-- Column header (native board grammar; shown by rkRender) -->
    <div id="rkTableHeader" class="pr-grid-row" style="display:none">
      <span>#</span>
      <span data-pr-sort-col="player">Player</span>
      <span class="th-r" data-pr-sort-col="prod">Prod</span>
      <span class="th-r" id="rkEffHeader" data-pr-sort-col="eff">Eff</span>
      <span class="th-r" id="rkBrkHeader" data-pr-sort-col="brk">Brk</span>
      <span class="th-r">Grade</span>
    </div>

    <!-- Loading -->
    <div id="rkLoading" class="sk-list" style="margin-top:4px;">
      <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line" style="width:46%"></div><div class="skeleton skeleton-line" style="width:30%;height:9px"></div></div><div class="skeleton sk-chip"></div></div>
      <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line" style="width:38%"></div><div class="skeleton skeleton-line" style="width:26%;height:9px"></div></div><div class="skeleton sk-chip"></div></div>
      <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line" style="width:52%"></div><div class="skeleton skeleton-line" style="width:34%;height:9px"></div></div><div class="skeleton sk-chip"></div></div>
      <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line" style="width:42%"></div><div class="skeleton skeleton-line" style="width:24%;height:9px"></div></div><div class="skeleton sk-chip"></div></div>
      <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line" style="width:48%"></div><div class="skeleton skeleton-line" style="width:30%;height:9px"></div></div><div class="skeleton sk-chip"></div></div>
      <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line" style="width:40%"></div><div class="skeleton skeleton-line" style="width:28%;height:9px"></div></div><div class="skeleton sk-chip"></div></div>
    </div>

    <!-- Rows -->
    <div id="rkList"></div>

    <!-- Empty -->
    <div id="rkEmpty" style="display:none;">
      <div class="empty-state">
        <span class="empty-state-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m20 20-4.7-4.7"/></svg></span>
        <p class="empty-state-title">No prospects match</p>
        <p class="empty-state-msg">Try clearing a filter or widening your position and tier selections.</p>
      </div>
    </div>

  </div><!-- end rk-panel-rankings -->

</div>

<!-- Prospect detail modal -->
<div id="rkModal" style="display:none;position:fixed;inset:0;z-index:10500;
     display:none;align-items:center;justify-content:center;padding:20px;
     background:rgba(15,23,42,0.7);backdrop-filter:blur(4px);">
  <div id="rkModalContent"
    style="background:var(--card);border-radius:16px;max-width:1080px;width:100%;
           max-height:90vh;overflow-y:auto;overflow-x:hidden;position:relative;
           box-shadow:0 24px 48px rgba(15,23,42,0.25);">
    <!-- filled by JS -->
  </div>
</div>
""" + """
<style>
  /* ===== Card: native board geometry (players-board grammar) ===== */
  #prospectsCard { max-width: 1180px; overflow: visible; }
  /* Real h1 (was h2) with the exact specified values the h2 rendered with,
     so the swap changes semantics only, not the layout. */
  #rookiesTitle { font-size: 18px; font-weight: 700; margin: 0; text-align: left; }
  /* The global h1::after is a 300px centered sky bar meant for standalone
     hero titles; inside this card header it renders as a stray floating
     bar between the title and the subtitle. Suppress it (same opt-out as
     .wl-page-title / .compare-page-title in dashboard.css). */
  #rookiesTitle::after { display: none; }
  /* Export is the only .btn on this page and dashboard.css defines no
     .btn/.btn-ghost/.btn-sm at all, so it fell back to the UA button
     padding (1px 6px, 22px tall) in the global accent fill. Give it real
     ghost-button geometry, scoped to this card. */
  #prospectsCard .btn-ghost { display: inline-flex; align-items: center; gap: 6px; padding: 7px 12px; min-height: 32px; background: transparent; border: 1px solid var(--border); border-radius: 8px; color: var(--text-muted); font-size: 13px; font-weight: 600; line-height: 1; white-space: nowrap; }
  #prospectsCard .btn-ghost:hover { color: var(--text); border-color: var(--accent); }

  /* ===== Filter controls: players-board two-row bar ===== */
  .filter-controls-container { display: flex; flex-direction: column; gap: 10px; padding: 14px 0 12px; border-bottom: 1px solid var(--border); margin-bottom: 10px; }
  .filter-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; justify-content: space-between; }
  .filter-row-primary { gap: 12px; }
  .filter-row-secondary { padding-top: 2px; flex-wrap: nowrap; align-items: center; }
  .filter-search { position: relative; flex: 1; min-width: 200px; }
  .filter-search input { width: 100%; height: 36px; padding: 8px 32px 8px 34px; border-radius: 8px; border: 1px solid var(--border); background: var(--card-bg); color: var(--text); font-size: 13px; outline: none; box-sizing: border-box; font-family: inherit; }
  .filter-search .mag { position: absolute; left: 10px; top: 50%; transform: translateY(-50%); color: var(--text-muted); font-size: 13px; pointer-events: none; }
  .filter-positions { display: flex; gap: 6px; flex-wrap: wrap; }
  .active-settings-indicator { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; flex: 1; min-width: 0; }
  .active-setting-tag { display: inline-flex; align-items: center; height: 24px; padding: 0 10px; line-height: 1; border-radius: var(--radius-pill, 8px); background: var(--accent-soft); color: var(--accent); font-size: 11px; font-weight: 600; white-space: nowrap; }
  .filter-sort-group { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
  .filter-sort { display: flex; align-items: center; gap: 8px; }
  .filter-label { font-size: 11px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: .04em; }
  .filter-sort select { padding: 7px 30px 7px 11px; border-radius: 9px; border: 1px solid var(--border); background: var(--card-bg); color: var(--text); font-size: 13px; font-weight: 600; cursor: pointer; outline: none; min-height: 36px; min-width: 128px; font-family: inherit; -webkit-appearance: none; appearance: none; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 11px center; }
  .filter-sort select:hover { border-color: var(--accent); }

  /* ===== Board: pr-grid rows ===== */
  #rkCount { font-size: 13px; color: var(--text-muted); margin: 0 0 10px; }
  .pr-grid-row { display: grid; grid-template-columns: 76px minmax(0,1fr) 64px 64px 64px 72px; align-items: center; column-gap: 0; }
  .pr-grade { text-align: right; font-size: 14px; font-weight: 700; line-height: 1.2; color: var(--accent); font-variant-numeric: tabular-nums; white-space: nowrap; }
  .pr-grade.none { color: var(--text-subtle); font-weight: 600; }
  #rkTableHeader { display: grid; padding: 6px 12px; border-radius: 6px; background: var(--accent-soft); font-size: 11px; font-weight: 700; color: var(--accent); letter-spacing: .04em; text-transform: uppercase; position: sticky; top: 60px; z-index: 20; box-shadow: 0 -8px 0 0 var(--card), 0 10px 14px -12px rgba(0,0,0,.4); margin-bottom: 2px; }
  #rkTableHeader [data-pr-sort-col] { cursor: pointer; user-select: none; }
  #rkTableHeader [data-pr-sort-col]:hover { color: var(--text); }
  #rkTableHeader .th-r { text-align: right; }
  .pr-player-row { padding: 8px 12px; cursor: pointer; transition: background .12s ease; }
  .pr-player-row:hover { background: var(--accent-soft); }
  .pr-player-row + .pr-player-row { border-top: 1px solid var(--border); }
  .pr-player-row:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
  .pr-rank { font-size: 13px; font-weight: 700; color: var(--text-muted); display: flex; align-items: center; gap: 6px; justify-content: flex-start; white-space: nowrap; font-variant-numeric: tabular-nums; }
  .pr-rank-delta { font-size: 11px; font-weight: 800; line-height: 1; white-space: nowrap; }
  .pr-rank-delta.up { color: var(--win); }
  .pr-rank-delta.down { color: var(--loss); }
  .pr-name { font-size: 13px; font-weight: 600; color: var(--text); display: flex; align-items: center; gap: 5px; flex-wrap: nowrap; min-width: 0; }
  .pr-nm { min-width: 0; line-height: 1.3; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  /* Player-name hover contract (site-wide): names dim, never underline. */
  .pr-nm:hover { opacity: 0.72; }
  .pr-wl-star { color: var(--gold); font-size: 12px; line-height: 1; margin-left: 5px; }
  .rk-player { display: block; min-width: 0; }
  .rk-meta { display: block; font-size: 11px; color: var(--text-muted); margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  /* Percentile cells: plain numbers only (clean-numbers board), right
     aligned and vertically centered in the row. */
  .rk-pct { display: flex; align-items: center; justify-content: flex-end; font-size: 13px; font-weight: 600; font-variant-numeric: tabular-nums; color: var(--text); }
  .rk-pct-dash { color: var(--text-muted); font-weight: 400; }
  .pr-tier-divider { display: flex; align-items: center; gap: 8px; padding: 5px 0 3px; pointer-events: none; }
  .pr-tier-divider-line { flex: 1; height: 1px; opacity: .5; }
  .pr-tier-divider-label { font-size: 11px; font-weight: 700; letter-spacing: .06em; opacity: .75; white-space: nowrap; text-transform: uppercase; }

  /* Position chip: the site-wide .pos-badge from dashboard.css (the
     matchups page chip), at the matchup board's compact render size. */
  .pr-player-row .pos-badge { width: 34px; height: 20px; font-size: 11px; padding: 1px 4px; display: inline-flex; align-items: center; justify-content: center; flex: none; }
  /* Tag chips: one geometry, equal tint/border alpha across the bands */
  .tag-chip { display: inline-flex; align-items: center; white-space: nowrap; font-size: 11px; font-weight: 700; letter-spacing: .02em; padding: 4px 9px; border-radius: var(--radius-pill); border: 1px solid var(--border); color: var(--text); }
  .tag-chip.tag-mini { font-size: 10px; padding: 2px 7px; }
  .tag-elite { background: color-mix(in srgb, #122d4b 12%, transparent); color: var(--accent); border: 1px solid color-mix(in srgb, #122d4b 30%, transparent); }
  .tag-great { background: rgba(56,189,248,.13); color: color-mix(in srgb, #38bdf8 65%, var(--text)); border: 1px solid rgba(56,189,248,.30); }
  .tag-average { background: rgba(148,163,184,.14); color: #94a3b8; border: 1px solid rgba(148,163,184,.30); }
  .tag-day3 { background: rgba(100,116,139,.14); color: #94a3b8; border: 1px solid rgba(100,116,139,.30); }
  /* In-season chip: quieter wash than the position chip */
  .m-early { display: inline-block; padding: 2px 6px; border-radius: 5px; font-size: 10px; font-weight: 600; background: color-mix(in srgb, var(--text-muted) 10%, transparent); color: var(--text-muted); white-space: nowrap; }
  .state-chip { display: inline-block; padding: 3px 7px; border-radius: 5px; font-size: 10px; font-weight: 600; background: color-mix(in srgb, var(--text-muted) 10%, transparent); color: var(--text-muted); white-space: nowrap; }

  @media (max-width: 1100px) {
    #prospectsCard .pr-grid-row { grid-template-columns: 72px minmax(0,1fr) 60px 60px 60px 68px; column-gap: 0; }
  }
  @media (max-width: 768px) {
    /* GUI audit Fix 3: 11px floor for board chip labels on mobile
       (the In-Season chip was 10px at this width; the position badge
       is 11px at every width via the board row rule). */
    .m-early { font-size: 11px; }
    .filter-row-primary { flex-direction: column; align-items: stretch; }
    .filter-search { max-width: 100%; }
    .filter-positions { justify-content: flex-start; gap: 6px; }
    .filter-row-secondary { flex-wrap: wrap; gap: 8px; }
    .active-settings-indicator { justify-content: center; order: -1; width: 100%; }
    .filter-sort-group { display: grid; grid-auto-flow: column; grid-auto-columns: minmax(0,1fr); gap: 10px; width: 100%; }
    .filter-sort { min-width: 0; flex-direction: column; align-items: stretch; gap: 5px; }
    .filter-sort select { width: 100%; min-width: 0; }
    /* board: the breakout column yields first, like Age on the players board
       (scoped to #prospectsCard: dashboard.css carries global players-board
       .pr-grid-row phone rules that must not leak into this board) */
    #prospectsCard .pr-grid-row { grid-template-columns: 72px minmax(0,1fr) 54px 54px 46px !important; column-gap: 0; }
    .pr-brk, #rkBrkHeader { display: none !important; }
  }
  @media (max-width: 600px) {
    .filter-row-primary { display: grid; grid-template-columns: 1fr auto; grid-template-rows: auto auto; gap: 8px; }
    .filter-search { grid-column: 1 / -1; min-width: 0; }
    .filter-positions { flex-wrap: nowrap; overflow-x: auto; scrollbar-width: none; min-width: 0; align-items: center; }
    .filter-positions::-webkit-scrollbar { display: none; }
    #prospectsCard .pr-grid-row { grid-template-columns: 64px minmax(0,1fr) 50px 50px 56px !important; column-gap: 0; }
    .m-early { display: none; }
  }
  @media (max-width: 480px) {
    /* phone: rank | player | production | grade, like rank | name | ppg | value on players */
    #prospectsCard .pr-grid-row { grid-template-columns: 66px minmax(0,1fr) 54px 60px !important; column-gap: 0; }
    .pr-eff, #rkEffHeader { display: none !important; }
    .pr-wl-star { font-size: 11px; margin-left: 4px; }
  }

  /* ===== Modal: prospect report card (single scroll, no tabs) ===== */
  /* Navy header band is constant in both themes; every other surface
     rides the site variables so dark mode stays native. */
  #rkModalContent .tag-chip, #rkModalContent .state-chip { font-size: 11px; }
  .rc-actions { position: absolute; top: 10px; right: 12px; z-index: 5; display: flex; gap: 8px; }
  .rc-icon-btn { width: 34px; height: 34px; border-radius: 9px; border: 1px solid rgba(255,255,255,.28); background: rgba(255,255,255,.10); color: #fff; font-size: 17px; line-height: 1; cursor: pointer; display: flex; align-items: center; justify-content: center; padding: 0; }
  .rc-icon-btn:hover { background: rgba(255,255,255,.20); }
  .rc-icon-btn .wl-star-glyph { font-size: 17px; }
  .rc-hero { background: #122d4b; color: #fff; display: flex; gap: 22px; align-items: stretch; border-radius: 16px 16px 0 0; }
  .rc-hero-left { display: flex; gap: 18px; align-items: center; padding: 22px 0 22px 22px; flex: 1; min-width: 0; }
  .rc-initials { width: 86px; height: 96px; flex: 0 0 86px; background: linear-gradient(180deg, #224a73, #162f4d); border: 1px solid rgba(255,255,255,.18); border-bottom: 4px solid #38bdf8; display: flex; align-items: center; justify-content: center; flex-direction: column; text-align: center; padding: 4px; }
  .rc-initials b { font-size: 32px; font-weight: 900; letter-spacing: .02em; line-height: 1; color: #fff; }
  .rc-initials span { font-size: 9px; letter-spacing: .18em; font-weight: 700; color: #a9c6e2; margin-top: 6px; text-transform: uppercase; line-height: 1.25; }
  .rc-headshot { width: 86px; height: 96px; flex: 0 0 86px; display: block; object-fit: cover; object-position: 50% 0; background: linear-gradient(180deg, #224a73, #162f4d); border: 1px solid rgba(255,255,255,.18); border-bottom: 4px solid #38bdf8; }
  .rc-mini-headshot { width: 26px; height: 26px; flex: 0 0 26px; display: block; object-fit: cover; object-position: 50% 0; background: #1b3f66; border: 1px solid rgba(255,255,255,.25); border-bottom: 2px solid #38bdf8; }
  .rc-nameblock { min-width: 0; }
  .rc-first { font-size: 15px; font-weight: 700; letter-spacing: .38em; text-transform: uppercase; color: #cfe6f7; line-height: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .rc-last { font-size: 56px; font-weight: 900; letter-spacing: -.02em; line-height: .95; text-transform: uppercase; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .rc-meta1 { margin-top: 12px; font-size: 13.5px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; color: #fff; }
  .rc-meta1 .rc-sep { color: #6f8fb3; font-weight: 400; }
  .rc-meta1 .rc-pos { color: #38bdf8; }
  .rc-vitals { margin-top: 6px; font-size: 12.5px; color: #b9cfe4; font-weight: 500; letter-spacing: .02em; }
  .rc-vitals b { color: #fff; font-weight: 700; }
  .rc-hero-right { text-align: right; padding: 40px 24px 22px 18px; border-left: 1px solid rgba(255,255,255,.16); display: flex; flex-direction: column; justify-content: center; align-items: flex-end; flex: 0 0 300px; }
  .rc-grade-num { font-size: 80px; font-weight: 900; line-height: .9; letter-spacing: -.03em; color: #fff; font-variant-numeric: tabular-nums; }
  .rc-grade-label { margin-top: 8px; font-size: 11px; font-weight: 800; letter-spacing: .24em; text-transform: uppercase; color: #7dd3fc; }
  .rc-grade-context { margin-top: 10px; font-size: 13.5px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; color: #fff; }
  .rc-grade-context .rc-dot { color: #6f8fb3; font-weight: 400; margin: 0 5px; }
  .rc-grade-context .rc-band-name { color: #38bdf8; }
  .rc-hero-note { margin: 10px 0 0; font-size: 12.5px; line-height: 1.45; color: #c2d6ea; max-width: 250px; font-weight: 500; }
  .rc-hero-note strong { color: #fff; }
  .rc-grade-stack { min-width: 0; }
  .rc-inseason-pill { display: inline-block; margin-top: 9px; background: transparent; border: 1px solid #38bdf8; color: #7dd3fc; font-size: 9.5px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; padding: 5px 8px; border-radius: 999px; line-height: 1; white-space: nowrap; }

  /* Sticky mini-header: sticks to the top of the modal scroll box
     once the hero header has scrolled out of it, sliding in over
     ~240ms; hidden and inert while the hero is visible. Its flow
     height is cancelled by the negative bottom margin, so toggling
     it never shifts the content under it. */
  .rc-mini { position: sticky; top: 0; z-index: 45; height: 44px; margin-bottom: -44px; background: #122d4b; color: #fff; display: flex; align-items: center; gap: 10px; padding: 0 20px; border-bottom: 1px solid #38bdf8; border-radius: 16px 16px 0 0; transform: translateY(-100%); opacity: 0; visibility: hidden; pointer-events: none; transition: transform 240ms ease-out, opacity 240ms ease-out, visibility 0s linear 240ms; }
  .rc-mini.is-visible { transform: translateY(0); opacity: 1; visibility: visible; pointer-events: auto; transition: transform 240ms ease-out, opacity 240ms ease-out, visibility 0s; }
  @media (prefers-reduced-motion: reduce) {
    .rc-mini, .rc-mini.is-visible { transition: none; }
  }
  .rc-mini-initials { width: 26px; height: 26px; flex: 0 0 26px; background: #1b3f66; border: 1px solid rgba(255,255,255,.25); border-bottom: 2px solid #38bdf8; display: flex; align-items: center; justify-content: center; font-size: 10.5px; font-weight: 800; letter-spacing: .02em; color: #fff; line-height: 1; }
  .rc-mini-name { font-size: 14px; font-weight: 800; letter-spacing: .01em; color: #fff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .rc-mini-grade { margin-left: auto; display: flex; align-items: baseline; gap: 8px; white-space: nowrap; }
  .rc-mini-num { font-size: 17px; font-weight: 800; letter-spacing: -.02em; color: #fff; line-height: 1; font-variant-numeric: tabular-nums; }
  .rc-mini-band { font-size: 10px; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; color: #38bdf8; }

  /* Archetype band */
  .rc-arch { background: var(--card); border-bottom: 1px solid var(--border); border-left: 4px solid var(--accent); padding: 20px 22px; display: flex; gap: 24px; justify-content: space-between; align-items: flex-start; }
  .rc-arch-main { flex: 1; min-width: 0; }
  .rc-arch-label { font-size: 11px; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; color: var(--text); }
  .rc-arch-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-top: 8px; }
  .rc-arch-title { margin: 0; font-size: 38px; font-weight: 900; letter-spacing: -.02em; line-height: 1; color: var(--text); }
  .rc-arch-tag { background: rgba(56,189,248,.14); border: 1px solid rgba(56,189,248,.35); color: color-mix(in srgb, #38bdf8 55%, var(--text)); font-size: 11px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; padding: 6px 10px; border-radius: 999px; line-height: 1; white-space: nowrap; }
  .rc-arch-ev { margin: 11px 0 0; font-size: 12px; line-height: 1.55; font-weight: 400; color: var(--text-muted); }
  .rc-arch-ev b { color: var(--text); font-weight: 600; }
  .rc-ev-sep { color: var(--text-subtle); margin: 0 4px; }
  .rc-arch-conf { flex: 0 0 172px; width: 172px; border: 1px solid var(--border); background: var(--bg-alt); padding: 13px; }
  .rc-arch-conf-label { font-size: 10.5px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; color: var(--text); line-height: 1.35; }
  .rc-meter { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 4px; margin-top: 10px; }
  .rc-meter .rc-seg { height: 8px; border-radius: 999px; background: color-mix(in srgb, var(--text-muted) 22%, transparent); display: block; }
  .rc-meter .rc-seg.on { background: #38bdf8; }

  /* Body grid: rail + main */
  .rc-body { display: grid; grid-template-columns: 300px 1fr; }
  .rc-rail { border-right: 1px solid var(--border); padding: 20px 18px 22px 20px; min-width: 0; }
  /* The Class Standing panel is a sibling of the comps section (it
     reorders separately on phones); its own top margin carries the
     gap the section margin used to add. */
  .rc-rail .rc-section { margin-bottom: 0; }
  .rc-main { padding: 20px 22px 22px; min-width: 0; }
  .rc-section { margin-bottom: 26px; }
  .rc-section:last-child { margin-bottom: 4px; }
  .rc-sec-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; border-bottom: 2px solid var(--accent); padding-bottom: 7px; margin-bottom: 12px; }
  .rc-sec-head h3 { margin: 0; font-size: 13px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; color: var(--text); white-space: nowrap; }
  .rc-sec-head .rc-side { font-size: 10px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--text-subtle); }
  .rc-subline { font-size: 12px; color: var(--text-muted); margin: -4px 0 12px; line-height: 1.45; font-weight: 500; }
  .rc-empty { font-size: 13px; color: var(--text-muted); padding: 6px 0; }

  /* Comparables + class standing (rail) */
  .rc-comp { display: flex; gap: 11px; align-items: center; padding: 11px 0; border-bottom: 1px solid var(--border); }
  .rc-comp:last-of-type { border-bottom: none; }
  .rc-disc { width: 42px; height: 42px; border-radius: 50%; flex: 0 0 42px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 800; letter-spacing: .02em; background: color-mix(in srgb, var(--pc, var(--accent)) 15%, transparent); color: var(--pc, var(--accent)); border: 1px solid color-mix(in srgb, var(--pc, var(--accent)) 35%, transparent); }
  .rc-comp-info { flex: 1; min-width: 0; }
  .rc-comp-name { font-size: 13.5px; font-weight: 800; line-height: 1.2; color: var(--text); }
  .rc-comp-meta { font-size: 11.5px; color: var(--text-muted); margin-top: 3px; font-weight: 500; line-height: 1.35; }
  .rc-panel { border: 1px solid var(--border); background: var(--bg-alt); padding: 14px 14px 12px; margin-top: 18px; }
  .rc-panel-h { margin: 0 0 10px; font-size: 11px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; color: var(--text); }
  .rc-kv { display: flex; justify-content: space-between; gap: 12px; padding: 7px 0; border-top: 1px solid var(--border); font-size: 12.5px; }
  .rc-kv:first-of-type { border-top: none; padding-top: 2px; }
  .rc-kv .rc-k { color: var(--text-muted); font-weight: 500; white-space: nowrap; }
  .rc-kv .rc-v { font-weight: 800; color: var(--text); text-align: right; }
  .rc-kv .rc-v.sky { color: #0284c7; }
  [data-theme="dark"] .rc-kv .rc-v.sky { color: #38bdf8; }
  .rc-panel-note { font-size: 12.5px; line-height: 1.5; color: var(--text-muted); margin: 10px 0 0; }
  .rc-panel-note strong { color: var(--text); }

  /* Production chart */
  .rc-trend { display: inline-flex; align-items: center; gap: 6px; background: rgba(56,189,248,.14); border: 1px solid rgba(56,189,248,.35); color: color-mix(in srgb, #38bdf8 55%, var(--text)); font-size: 11px; font-weight: 800; letter-spacing: .04em; padding: 6px 9px; border-radius: 999px; white-space: nowrap; }
  .rc-trend b { color: #0284c7; }
  [data-theme="dark"] .rc-trend b { color: #38bdf8; }
  .rc-trend.down { background: var(--accent-soft); border-color: var(--border); color: var(--text-muted); }
  .rc-trend.down b { color: var(--text-muted); }
  .rc-chart { display: grid; gap: 18px; align-items: end; padding: 18px 6px 0; border-bottom: 2px solid var(--accent); }
  .rc-barcol { display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%; min-width: 0; }
  .rc-barval { font-size: 22px; font-weight: 900; color: var(--text); line-height: 1; margin-bottom: 8px; letter-spacing: -.02em; font-variant-numeric: tabular-nums; }
  .rc-barval.live { color: #0284c7; }
  [data-theme="dark"] .rc-barval.live { color: #38bdf8; }
  .rc-bar { width: 100%; max-width: 118px; border-radius: 3px 3px 0 0; background: color-mix(in srgb, #122d4b 45%, #38bdf8); }
  .rc-bar.live { background: #38bdf8; }
  .rc-barlabels { display: grid; gap: 18px; padding: 10px 6px 0; text-align: center; }
  .rc-barlab { min-width: 0; }
  .rc-yr { font-size: 15px; font-weight: 800; color: var(--text); }
  .rc-yr.liveyr { color: #0284c7; }
  [data-theme="dark"] .rc-yr.liveyr { color: #38bdf8; }
  .rc-detail { font-size: 11.5px; color: var(--text-muted); margin-top: 3px; line-height: 1.4; font-weight: 500; }
  .rc-inprog { font-size: 10px; font-weight: 800; letter-spacing: .14em; color: #0284c7; text-transform: uppercase; margin-top: 3px; }
  [data-theme="dark"] .rc-inprog { color: #38bdf8; }

  /* Season lines table (report-card dialect) */
  .rc-tablewrap { border: 1px solid var(--border); }
  .rc-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .rc-table th { text-align: right; font-size: 10.5px; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; color: var(--text-muted); padding: 9px 12px; background: var(--bg-alt); border-bottom: 1px solid var(--border); white-space: nowrap; }
  .rc-table th:first-child { text-align: left; }
  .rc-table td { padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: right; font-weight: 600; color: var(--text); white-space: nowrap; font-variant-numeric: tabular-nums; }
  .rc-table td:first-child { text-align: left; font-weight: 800; }
  .rc-table tr:last-child td { border-bottom: none; }
  .rc-table tr.live td { background: color-mix(in srgb, #38bdf8 7%, transparent); }
  .rc-livepill { display: inline-block; font-size: 9px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; background: #38bdf8; color: #082f49; padding: 3px 6px; border-radius: 999px; margin-left: 7px; vertical-align: 2px; }

  /* Grade Breakdown: the four component percentiles as two-line
     rows (name + number, then the bar alone). */
  .rc-gb-row { padding: 10px 0 11px; border-bottom: 1px solid var(--border); }
  .rc-gb-row:last-of-type { border-bottom: none; }
  .rc-gb-line1 { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 7px; }
  .rc-gb-name { font-size: 13.5px; font-weight: 600; color: var(--text); }
  .rc-gb-val { font-size: 14px; font-weight: 800; color: var(--text); font-variant-numeric: tabular-nums; }
  .rc-read-line { margin: 12px 0 0; font-size: 12.5px; line-height: 1.55; color: var(--text-muted); font-weight: 500; }

  /* Season Lines phone grid: strict fixed columns so every value
     right-aligns to a shared column edge; open hairline rows, no
     box, no filled header band. The table above serves wider
     layouts; the grid replaces it at the phone breakpoint. */
  .rc-season-grid { display: none; }
  .rc-sg-row { display: grid; column-gap: 10px; align-items: start; }
  .rc-sg-head { padding: 0 0 10px; border-bottom: 1px solid var(--border); }
  .rc-sg-h { font-size: 10.5px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--text-muted); text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; line-height: 1.25; }
  .rc-sg-h.rc-sg-season { text-align: left; }
  .rc-sg-row:not(.rc-sg-head) { padding: 12px 0; border-bottom: 1px solid var(--border); }
  .rc-sg-row.rc-sg-last { border-bottom: none; }
  .rc-sg-row.live { background: color-mix(in srgb, #38bdf8 7%, transparent); }
  .rc-sg-c { font-size: 14px; line-height: 1.25; font-weight: 600; color: var(--text); text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
  .rc-sg-c.rc-sg-season { text-align: left; font-weight: 700; }
  .rc-sg-row.live .rc-sg-c.rc-sg-season { color: #0284c7; }
  [data-theme="dark"] .rc-sg-row.live .rc-sg-c.rc-sg-season { color: #38bdf8; }
  .rc-sg-dash { color: var(--text-subtle); }
  /* Year-over-year delta: a quiet second line under the Yds/G value,
     in the phone grid and in the desktop table's rate column. */
  .rc-sg-c .rc-yds-val, .rc-table .rc-yds-val { display: block; }
  .rc-sg-c .rc-yds-delta, .rc-table .rc-yds-delta { display: block; margin-top: 2px; font-size: 10.5px; font-weight: 600; line-height: 1.3; color: #0284c7; white-space: nowrap; }
  [data-theme="dark"] .rc-sg-c .rc-yds-delta, [data-theme="dark"] .rc-table .rc-yds-delta { color: #38bdf8; }
  .rc-sg-c .rc-yds-delta.neg, .rc-table .rc-yds-delta.neg { color: var(--text-muted); }

  /* Metric breakdown */
  .rc-metric-head, .rc-metric-row { display: grid; grid-template-columns: 1fr 64px 130px 118px; gap: 12px; align-items: center; }
  .rc-metric-head { font-size: 10.5px; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; color: var(--text-subtle); padding: 0 0 7px; }
  .rc-metric-head .r { text-align: right; }
  .rc-group { border: 1px solid var(--border); margin-bottom: 10px; display: grid; grid-template-columns: 108px 1fr; background: var(--card); }
  .rc-group-label { background: var(--accent-soft); color: var(--text); font-size: 11px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; display: flex; align-items: center; justify-content: center; text-align: center; padding: 12px 8px; line-height: 1.3; border-right: 1px solid var(--border); }
  .rc-group-rows { padding: 2px 12px; min-width: 0; }
  .rc-metric-row { padding: 9px 0; border-bottom: 1px solid var(--border); }
  .rc-metric-row:last-child { border-bottom: none; }
  .rc-m-name { font-size: 13px; font-weight: 600; color: var(--text); }
  .rc-m-val { font-size: 13.5px; font-weight: 800; color: var(--text); text-align: right; font-variant-numeric: tabular-nums; }
  .rc-mbar { height: 8px; background: color-mix(in srgb, var(--text-muted) 14%, transparent); border-radius: 999px; display: block; position: relative; }
  .rc-mbar i { display: block; height: 100%; background: #38bdf8; border-radius: 999px; }
  /* Midpoint tick at the 50% mark on every percentile bar. */
  .rc-mbar::after { content: ""; position: absolute; left: 50%; top: -2px; margin-left: -1px; width: 2px; height: 12px; border-radius: 1px; background: var(--text-muted); opacity: .55; pointer-events: none; }
  .rc-mbar.empty { background: transparent; }
  .rc-mpct { font-size: 12.5px; font-weight: 700; color: var(--text); text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
  .rc-mpct.high { color: #60a5fa; font-weight: 800; }
  .rc-mpct.none { color: var(--text-subtle); font-weight: 500; }

  /* Scouting notes */
  .rc-notes { list-style: none; margin: 0; padding: 0; }
  .rc-notes li { display: flex; gap: 11px; padding: 10px 0; border-bottom: 1px solid var(--border); font-size: 13.5px; line-height: 1.5; color: var(--text); font-weight: 500; }
  .rc-notes li:last-child { border-bottom: none; }
  .rc-notes .rc-n { flex: 0 0 22px; width: 22px; height: 22px; border-radius: 50%; background: var(--accent); color: var(--card); font-size: 11px; font-weight: 800; display: flex; align-items: center; justify-content: center; margin-top: 1px; }

  /* Footer strip */
  .rc-foot { border-top: 1px solid var(--border); padding: 12px 22px; display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap; background: var(--bg-alt); border-radius: 0 0 16px 16px; align-items: center; }
  .rc-foot p { margin: 0; font-size: 11px; line-height: 1.5; color: var(--text-muted); font-weight: 500; }
  .rc-foot .rc-brand { font-weight: 800; letter-spacing: .12em; text-transform: uppercase; color: var(--text); font-size: 11px; white-space: nowrap; }

  @media (max-width: 860px) {
    /* Phone report card (approved mobile mock): one column, roomy
       20px sides, ~34px section gaps, only the navy header and the
       archetype band full-bleed. The rail/main wrappers dissolve
       (display: contents) so the sections order as: archetype,
       class standing, grade breakdown, production, season lines,
       metric breakdown, comparables, scouting notes. */
    .rc-hero { flex-direction: column; }
    .rc-hero-left { padding: 20px 76px 4px 20px; }
    .rc-last { font-size: 34px; }
    .rc-first { letter-spacing: .3em; font-size: 11px; }
    .rc-initials { width: 56px; height: 64px; flex-basis: 56px; }
    .rc-initials b { font-size: 22px; }
    .rc-headshot { width: 56px; height: 64px; flex-basis: 56px; }
    .rc-hero-right { flex: none; text-align: left; align-items: stretch; justify-content: flex-start; border-left: none; border-top: 1px solid rgba(255,255,255,.16); padding: 14px 20px 20px; flex-direction: column; gap: 0; }
    .rc-grade-block { display: flex; align-items: center; gap: 18px; flex-wrap: wrap; text-align: left; }
    .rc-grade-num { font-size: 56px; }
    .rc-hero-note { flex: none; max-width: 100%; margin-top: 12px; }
    .rc-arch { flex-direction: column; gap: 16px; padding: 20px; }
    .rc-arch-title { font-size: 32px; }
    /* Confidence: a slim borderless row, label left, meter right. */
    .rc-arch-conf { flex: none; width: 100%; border: none; border-top: 1px solid var(--border); background: transparent; padding: 11px 0 0; display: flex; align-items: center; justify-content: space-between; gap: 14px; }
    .rc-meter { display: flex; gap: 4px; margin-top: 0; }
    .rc-meter .rc-seg { width: 30px; height: 8px; flex: 0 0 30px; }
    .rc-body { display: flex; flex-direction: column; padding: 2px 20px 10px; }
    .rc-rail, .rc-main, .rc-comps-section { display: contents; }
    .rc-section { margin-top: 34px; margin-bottom: 0; }
    #rkRcStanding { order: 1; margin-top: 26px; }
    .rc-sec-grade { order: 2; }
    .rc-sec-prod { order: 3; }
    .rc-sec-season { order: 4; }
    .rc-sec-metrics { order: 5; }
    .rc-comps-head { order: 6; margin-top: 34px; }
    .rc-comps-sub { order: 6; }
    #rkRcComps { order: 6; }
    .rc-sec-notes { order: 7; }
    .rc-chart, .rc-barlabels { gap: 10px; }
    .rc-barval { font-size: 18px; }
    /* The live season reads from the sky year + games text alone. */
    .rc-inprog { display: none; }
    .rc-sec-head { flex-wrap: wrap; }
    .rc-sec-head h3 { white-space: normal; }
    .rc-trend { white-space: normal; line-height: 1.35; }
    /* Season Lines: the strict phone grid replaces the table. */
    .rc-season-desktop { display: none; }
    .rc-season-grid { display: block; }
    /* Metric Breakdown: plain group headers over two-line rows
       (name + value, then the bar with a compact percentile label). */
    .rc-metric-head { display: none; }
    .rc-group { grid-template-columns: 1fr; border: none; background: transparent; margin-bottom: 14px; }
    .rc-group-label { background: transparent; justify-content: flex-start; text-align: left; padding: 8px 0 6px; border-right: none; border-bottom: 1px solid var(--border); }
    .rc-group-rows { padding: 0; }
    .rc-metric-row { grid-template-columns: minmax(0,1fr) auto; grid-template-areas: "name val" "bar pct"; column-gap: 10px; row-gap: 7px; align-items: center; padding: 11px 0 12px; }
    .rc-m-name { grid-area: name; font-size: 13.5px; }
    .rc-m-val { grid-area: val; }
    .rc-mbar { grid-area: bar; }
    .rc-mbar.empty { background: color-mix(in srgb, var(--text-muted) 14%, transparent); }
    .rc-mpct { grid-area: pct; font-size: 11px; min-width: 30px; }
    .rc-mpct .rc-mpct-word { display: none; }
    .rc-table th, .rc-table td { padding: 8px 6px; }
    .rc-table th { font-size: 9.5px; letter-spacing: .06em; }
    .rc-table { font-size: 12.5px; }
    .rc-foot { padding: 12px 20px; }
  }

  /* Player-modal-style prospect modal: initials tile for the headshot slot
     when a prospect has no photo (mirrors the player modal fallback). */
  .rk-hs-initials { width: 160px; height: 135px; background: var(--accent-soft); color: var(--accent);
    display: flex; align-items: center; justify-content: center; font-size: 40px; font-weight: 800; letter-spacing: .02em; }

  .m-sec-hint { font-size: 11px; color: var(--text-subtle); margin-left: auto; }

  /* Class context strip */
  .ctx-strip { display: flex; flex-wrap: wrap; align-items: center; gap: 5px 7px; background: var(--bg-alt); border: 1px solid var(--border); border-radius: var(--radius); padding: 9px 12px; font-size: 12px; color: var(--text-muted); margin-bottom: 2px; }
  .ctx-strip .stand { font-weight: 700; color: var(--text); }
  .ctx-strip .who { font-weight: 700; color: var(--text); }
  .ctx-sep { opacity: .35; }

  /* Percentile panel: number-first read, railed bar, raw-stat sub-line */
  .pct-item { display: grid; grid-template-columns: 112px 1fr 38px; column-gap: 12px; align-items: center; margin-top: 13px; }
  .pct-label { font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; color: var(--text-muted); }
  .pct-track { display: block; height: 7px; background: color-mix(in srgb, var(--accent) 18%, transparent); border-radius: 999px; overflow: hidden; }
  .pct-fill { display: block; height: 100%; border-radius: 999px; opacity: .92; }
  .pct-val { font-size: 14px; font-weight: 700; text-align: right; color: var(--text); font-variant-numeric: tabular-nums; line-height: 1; }
  .pct-sub { grid-column: 2; font-size: 11px; color: var(--text-subtle); margin-top: 4px; line-height: 1.35; }

  /* Season table (modal dialect). The scroll wrapper is the site's
     table treatment: the full per-season stat set scrolls inside its
     own container; the modal/page never overflows horizontally. */
  .rk-table-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; max-width: 100%; }
  .rk-table-scroll .m-season-table { min-width: 100%; }
  .m-season-table { width: 100%; border-collapse: collapse; font-size: 11.5px; font-family: ui-monospace, Menlo, monospace; color: var(--text); }
  .m-season-table th { text-align: right; font-weight: 700; color: var(--text-muted); padding: 7px 8px; border-bottom: 1px solid var(--border); white-space: nowrap; }
  .m-season-table th:first-child, .m-season-table td:first-child { text-align: left; }
  .m-season-table td { text-align: right; padding: 7px 8px; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
  .m-season-table tr:last-child td { border-bottom: none; }
  .m-season-table td.yr { font-weight: 700; color: var(--text-muted); }
  .m-season-table td.pg { font-weight: 700; }
  .m-season-table td.team { text-align: left; font-family: var(--font-sans); font-size: 12px; }
  .m-season-table th.team { text-align: left; }
  .m-season-table td.tst { text-align: left; font-family: var(--font-sans); font-size: 12px; font-weight: 600; }
  .m-season-table th.tst { text-align: left; }
  .inprog { display: inline-block; margin-left: 8px; padding: 2px 6px; border-radius: 5px; font-size: 9.5px; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; font-family: var(--font-sans); background: color-mix(in srgb, var(--accent) 14%, transparent); color: var(--accent); vertical-align: 1px; white-space: nowrap; }
  /* Testing table percentile cell: mini railed bar + value */
  .tpct { display: flex; align-items: center; justify-content: flex-end; gap: 8px; }
  .tpct .pct-track { width: 64px; flex: none; }
  .tpct .n { font-weight: 700; color: var(--text); font-variant-numeric: tabular-nums; }
  .tpct .none { color: var(--text-subtle); }
  /* Season-line percentile cell */
  .pct-mini { display: inline-flex; align-items: center; gap: 7px; justify-content: flex-end; }
  .pct-mini .pct-track { width: 52px; height: 5px; flex: none; }
  .pct-mini b { font-weight: 700; color: var(--text); }
  .pct-mini-none { color: var(--text-subtle); }

  /* Production trajectory bars (live treatment) */
  .rk-traj { display: flex; gap: 14px; align-items: flex-end; margin: 8px 0; width: 100%; }
  .rk-tbar { position: relative; display: flex; flex-direction: column; align-items: center; flex: 1 1 0; min-width: 0; }
  .rk-ttrack { display: flex; align-items: flex-end; justify-content: center; height: 110px; width: 100%; }
  .rk-tfill { width: 100%; border-radius: 6px 6px 0 0; background: color-mix(in srgb, #122d4b 45%, #38bdf8); opacity: 1; }
  .rk-tbar.live .rk-tfill { background: #38bdf8; }
  .rk-tpg { font-size: 15px; font-weight: 800; font-variant-numeric: tabular-nums; margin-top: 6px; }
  .rk-tyr { font-size: 10.5px; color: var(--text-muted); font-weight: 700; margin-top: 1px; text-align: center; }
  .rk-live-tag { position: absolute; top: -8px; left: 50%; transform: translateX(-50%); font-size: 8.5px; font-weight: 800; letter-spacing: .08em; color: #122d4b; background: #38bdf8; border-radius: 4px; padding: 2px 6px; white-space: nowrap; z-index: 2; }
  .rk-trend-note { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--text-muted); margin-top: 10px; background: var(--bg-alt); border: 1px solid var(--border); border-radius: 8px; padding: 9px 12px; }
  .rk-trend-note b { color: var(--text); }

  /* Advanced metrics: percentile-only table */
  .rk-adv-tbl { width: 100%; border-collapse: collapse; font-size: 12px; }
  .rk-adv-tbl th { text-align: left; font-size: 9.5px; font-weight: 800; letter-spacing: .08em; color: var(--text-subtle); text-transform: uppercase; padding: 5px 8px; border-bottom: 1px solid var(--border); }
  .rk-adv-tbl th.r, .rk-adv-tbl td.r { text-align: right; }
  .rk-adv-tbl td { padding: 7px 8px; border-bottom: 1px solid var(--border); font-family: ui-monospace, Menlo, monospace; }
  .rk-adv-tbl tr:last-child td { border-bottom: none; }
  .rk-adv-tbl td.rk-adv-name { font-family: var(--font-sans); font-weight: 600; color: var(--text); }
  .rk-adv-raw { font-weight: 800; font-size: 13px; }
  .rk-pbar { display: inline-block; width: 72px; height: 5px; background: color-mix(in srgb, var(--text-muted) 14%, transparent); border-radius: 3px; vertical-align: middle; margin-right: 8px; overflow: hidden; }
  .rk-pbar i { display: block; height: 100%; border-radius: 3px; }
  .rk-pctl { font-weight: 700; color: var(--text-muted); font-size: 11.5px; }
  .rk-pctl b { color: var(--text); }

  /* Historical comparables */
  .m-sim-row { display: flex; align-items: center; gap: 10px; padding: 8px 0; border-bottom: 1px solid var(--border); }
  .m-sim-row:last-child { border-bottom: none; }
  .m-sim-disc {
    width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 800;
    background: color-mix(in srgb, var(--pc, var(--accent)) 15%, transparent);
    color: var(--pc, var(--accent));
    border: 1px solid color-mix(in srgb, var(--pc, var(--accent)) 35%, transparent);
    position: relative; overflow: hidden;
  }
  .m-sim-disc img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
  .m-sim-name { font-size: 13px; font-weight: 600; color: var(--text); }
  .m-sim-meta { font-size: 11px; color: var(--text-muted); margin-top: 1px; }
  .m-sim-right { margin-left: auto; display: flex; align-items: center; gap: 8px; flex-shrink: 0; }

  /* Detail loading shimmer */
  .rk-detail-loading { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-muted); padding: 12px 0; }

  @media (max-width: 768px) {
    .rk-hs-initials { width: 64px; height: 64px; min-height: 0; font-size: 22px; align-self: flex-start; flex: none; }

    /* Modal: same centered-card treatment as the standard player modal on mobile */
    #rkModal { padding: 16px 4px !important; align-items: center !important; }
    #rkModalContent {
      /* Fixed height (not max-height) so the card stays the same size whether the
         detail content is loading or loaded -- mirrors .player-modal. */
      height: 88dvh;
      height: 88vh;
      max-height: 88dvh;
      max-height: 88vh;
      max-width: calc(100% - 8px) !important;
      border-radius: var(--radius, 16px) !important;
      margin: auto;
      /* The sheet itself never scrolls sideways; wide inner tables keep
         their own scroll wrappers. */
      overflow-x: hidden;
    }

    /* Header meta on phones: the dot separators are separate flex items and
       strand at line ends when the line wraps. Drop them and let the flex
       gap separate the parts instead. */
    #rkModalContent .rk-meta-line { gap: 2px 10px !important; }
    #rkModalContent .rk-meta-sep { display: none; }

    /* Season table: compact padding */
    .m-season-table th, .m-season-table td { padding: 6px 5px; }
    .pct-item { grid-template-columns: 96px 1fr 34px; column-gap: 10px; }
    .pct-mini .pct-track { width: 38px; }
    .rk-pbar { width: 52px; margin-right: 6px; }
    .rk-adv-tbl td { padding: 7px 5px; }
    .rk-adv-tbl th { padding: 5px 5px; }
    .rk-traj { gap: 8px; }
    .rk-tfill { max-width: 64px; }

    /* Comparables: tighter */
    .m-sim-row { gap: 8px; padding: 7px 0; }
    .m-sim-disc { width: 28px; height: 28px; font-size: 10px; }
  }

  /* Small-phone fit for the report-card header: the split name and
     grade numeral step down so neither crowds the initials tile. */
  @media (max-width: 480px) {
    .rc-last { font-size: 34px; }
    .rc-grade-num { font-size: 46px; }
    .rc-hero-left { gap: 12px; }
  }
</style>
""" + """
<script>
  var rkAllPlayers = [];   // full unfiltered list - never replaced after load
  var rkPercentilePool = '';  // percentile basis line from the rankings payload
  var rkPosFilters = new Set();
  var rkSearch    = '';
  var rkLoaded    = false;
  var rkPipelinePaused = false;
  var rkDraftYear = null;
  var rkDraftComplete = false;
  var rkViewSel = '';      // class on show: 'all' or a year string (set on load)
  var rkIsArchiveView = false;  // payload came from the past-class archive
  var rkShowYearMeta = false;   // append the class year to row meta lines
  var rkSortKey = 'rank';  // active sort key (mirrors the #rkSort dropdown)
  var rkSortDir = 'asc';   // 'asc' | 'desc' -- flipped by clicking a column header
  // Default direction when a sort key is first chosen.
  var RK_SORT_DIRS = { rank: 'asc', prod: 'desc', eff: 'desc', brk: 'desc', name: 'asc' };

  // Watchlist (localStorage-backed) with the shared undo toast.
  var rkWatch = {};
  try {
    JSON.parse(localStorage.getItem('rkWatchlistV1') || '[]').forEach(function(id){ rkWatch[id] = true; });
  } catch (e) {}
  function rkSaveWatch() {
    try { localStorage.setItem('rkWatchlistV1', JSON.stringify(Object.keys(rkWatch))); } catch (e) {}
  }

  var RK_DISC_COLORS = ['#38bdf8','#60a5fa','#7dd3fc','#0ea5e9','#64748b','#94a3b8'];
  function rkEsc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function rkInitials(name) {
    var parts = String(name || '').split(' ');
    return (((parts[0] || '').charAt(0) + (parts[1] ? parts[1].charAt(0) : '')) || '?').toUpperCase();
  }
  function rkDiscColor(pid) {
    var s = String(pid || ''), hsh = 0;
    for (var i = 0; i < s.length; i++) hsh = (hsh * 31 + s.charCodeAt(i)) >>> 0;
    return RK_DISC_COLORS[hsh % RK_DISC_COLORS.length];
  }
  // ── Percentile display ─────────────────────────────────────────────────
  // Percentile values come from the API (computed within position against
  // the graded reference population); a null percentile renders nothing,
  // never a placeholder. The numeric prospect grade renders beside them
  // (board Grade column, modal Prospect Grade card).
  var RK_PCT_BOARD = [
    { key: 'production', label: 'Production', color: '#38bdf8' },
    { key: 'efficiency', label: 'Efficiency', color: '#38bdf8' },
    { key: 'breakout', label: 'Breakout', color: '#38bdf8' }
  ];
  var RK_TAG_CLASS = { elite: 'tag-elite', great: 'tag-great', average: 'tag-average', day3: 'tag-day3' };
  var RK_BAND_COLORS = { elite: 'var(--accent)', great: '#0ea5e9', average: '#64748b', day3: '#64748b' };
  var RK_BAND_SHORT = { elite: 'Elite', great: 'Great', average: 'Average', day3: 'Day 3' };
  function rkTagChip(tag, mini) {
    if (!tag || !tag.label) return '';
    return '<span class="tag-chip' + (mini ? ' tag-mini' : '') + ' ' +
      (RK_TAG_CLASS[tag.band] || '') + '">' + rkEsc(tag.label) + '</span>';
  }
  function rkPosRankLabel(r) {
    return (r.position && r.position_rank) ? String(r.position).toUpperCase() + r.position_rank : '';
  }
  function rkHeightStr(r) {
    var ht = parseInt(r.height_inches, 10);
    if (isNaN(ht)) return '';
    return Math.floor(ht / 12) + "'" + (ht % 12) + '"';
  }
  function rkWeightStr(r) {
    return r.weight_lbs ? r.weight_lbs + ' lbs' : '';
  }
  function rkPctOf(r, key) {
    var p = r && r.percentiles ? r.percentiles[key] : null;
    if (p == null || p === '' || isNaN(+p)) return null;
    return +p;
  }

  // ── Board ──────────────────────────────────────────────────────────────
  // One percentile stat cell: the plain number only, right-aligned and
  // vertically centered in the row (clean-numbers board, per the
  // approved mock). A missing percentile renders a muted dash, never a
  // blank hole.
  function rkStatCell(r, def, extraCls) {
    var p = rkPctOf(r, def.key);
    if (p === null) return '<span class="rk-pct rk-pct-dash ' + extraCls + '">&ndash;</span>';
    return '<span class="rk-pct ' + extraCls + '">' + p + '</span>';
  }

  // The numeric prospect grade cell (one decimal, right edge of the row).
  // A missing score renders the muted dash, never a placeholder number.
  function rkGradeCell(r) {
    var s = r.prospect_score == null || r.prospect_score === ''
      ? NaN : parseFloat(r.prospect_score);
    if (isNaN(s)) return '<span class="pr-grade none">-</span>';
    return '<span class="pr-grade">' + s.toFixed(1) + '</span>';
  }

  // One board row (native pr-grid grammar): rank + movement triangles,
  // name with the site-wide position badge (the matchups page chip)
  // over a "WR1 · Ohio State" meta line, percentile stats as plain
  // numbers, numeric grade. No band chip and no In-Season chip on the
  // row: the tier divider lines carry the band, and the season state
  // shows once in the filter tags.
  function rkBoardRow(r) {
    var pid = r.player_id || '';
    var pos = (r.position || '').toUpperCase();
    var rank = r.overall_rank != null && r.overall_rank !== '' ? +r.overall_rank : null;
    var delta = '';
    var d = r.rank_delta == null || r.rank_delta === '' ? null : parseInt(r.rank_delta, 10);
    if (d !== null && !isNaN(d) && d !== 0) {
      delta = '<span class="pr-rank-delta ' + (d > 0 ? 'up' : 'down') + '" title="' +
        Math.abs(d) + ' spot' + (Math.abs(d) === 1 ? '' : 's') + ' overall">' +
        (d > 0 ? '&#9650;' : '&#9660;') + Math.abs(d) + '</span>';
    }
    var star = rkWatch[pid]
      ? '<span class="pr-wl-star" title="On your watchlist">&#9733;</span>' : '';
    var posRank = rkPosRankLabel(r);
    var metaTxt = '';
    if (posRank && r.school) metaTxt = rkEsc(posRank) + ' &middot; ' + rkEsc(r.school);
    else if (posRank) metaTxt = rkEsc(posRank);
    else if (r.school) metaTxt = rkEsc(r.school);
    // Archive and combined views tag each row with its class year:
    // the same player name can appear in more than one class era.
    if (rkShowYearMeta && r.draft_class_year) {
      metaTxt = metaTxt ? metaTxt + ' &middot; ' + rkEsc(r.draft_class_year)
                        : rkEsc(r.draft_class_year);
    }
    return '<div class="pr-player-row pr-grid-row" data-pid="' + rkEsc(pid) + '" role="button" tabindex="0" aria-label="' + rkEsc(r.name || '') + '">' +
      '<span class="pr-rank">#' + (rank !== null && !isNaN(rank) ? rank : '-') + delta + '</span>' +
      '<span class="rk-player"><span class="pr-name"><span class="pr-nm">' + rkEsc(r.name || 'Unknown') + star + '</span>' +
        (pos ? '<span class="pos-badge ' + rkEsc(pos) + '">' + rkEsc(pos) + '</span>' : '') +
        '</span>' +
        (metaTxt ? '<span class="rk-meta">' + metaTxt + '</span>' : '') +
      '</span>' +
      rkStatCell(r, RK_PCT_BOARD[0], '') +
      rkStatCell(r, RK_PCT_BOARD[1], 'pr-eff') +
      rkStatCell(r, RK_PCT_BOARD[2], 'pr-brk') +
      rkGradeCell(r) +
    '</div>';
  }

  // Tier line at the head of each band group on the default board order
  // (short band name in the band hue). Printed above the first group
  // too: with no tag chips on the rows, the tier lines are what name
  // every band.
  function rkBandDivider(band) {
    var color = RK_BAND_COLORS[band] || '#94a3b8';
    var label = RK_BAND_SHORT[band] || '';
    return '<div class="pr-tier-divider"><div class="pr-tier-divider-line" style="background:' +
      color + '"></div><span class="pr-tier-divider-label" style="color:' + color + '">' +
      rkEsc(label) + '</span><div class="pr-tier-divider-line" style="background:' +
      color + '"></div></div>';
  }

  // Filtered + sorted view of the board (shared by render and CSV export).
  function rkVisiblePlayers() {
    var players = rkAllPlayers.slice();
    if (rkPosFilters.size > 0) {
      players = players.filter(function(r) {
        return rkPosFilters.has((r.position || '').toUpperCase());
      });
    }
    if (rkSearch.length > 0) {
      var q = rkSearch;
      // Shared matcher (brNameMatch, core bundle): normalized / compact /
      // initials / typo tolerant, ranked score -> prospect_score.
      if (typeof brRankPlayersByName === 'function') {
        return brRankPlayersByName(players, q, function(r) {
          return parseFloat(r.prospect_score || 0);
        });
      }
      var qL = q.toLowerCase();
      return players
        .map(function(r) {
          var n = (r.name || '').toLowerCase();
          return { r: r, s: n.includes(qL) ? 100 + (100 - n.indexOf(qL)) : 0 };
        })
        .filter(function(x) { return x.s > 0; })
        .sort(function(a, b) {
          return b.s - a.s ||
            (parseFloat(b.r.prospect_score || 0) - parseFloat(a.r.prospect_score || 0));
        })
        .map(function(x) { return x.r; });
    }
    var sortBy = rkSortKey;
    var rkDirMult = rkSortDir === 'asc' ? 1 : -1;
    players.sort(function(a, b) {
      if (sortBy === 'prod' || sortBy === 'eff' || sortBy === 'brk') {
        var ck = sortBy === 'prod' ? 'production' : (sortBy === 'eff' ? 'efficiency' : 'breakout');
        var av = rkPctOf(a, ck), bv = rkPctOf(b, ck);
        if (av === null && bv === null) return 0;
        if (av === null) return 1;   // missing percentiles always last
        if (bv === null) return -1;
        return (av - bv) * rkDirMult;
      }
      if (sortBy === 'name') {
        return String(a.name || '').localeCompare(String(b.name || '')) * rkDirMult;
      }
      var ar = a.overall_rank > 0 ? +a.overall_rank : null;
      var br = b.overall_rank > 0 ? +b.overall_rank : null;
      if (ar === null && br === null) return 0;
      if (ar === null) return 1;
      if (br === null) return -1;
      return (ar - br) * rkDirMult;
    });
    return players;
  }

  function rkRender() {
    if (!rkLoaded) return;
    var listEl = document.getElementById('rkList');
    var emptyEl = document.getElementById('rkEmpty');
    var countEl = document.getElementById('rkCount');
    var headEl = document.getElementById('rkTableHeader');
    if (!listEl) return;
    var list = rkVisiblePlayers();

    if (!list.length) {
      listEl.innerHTML = '';
      if (emptyEl) {
        emptyEl.style.display = 'block';
        var emptyTitle = emptyEl.querySelector('.empty-state-title');
        var emptyMsg = emptyEl.querySelector('.empty-state-msg');
        var pausedEmpty = rkPipelinePaused && rkAllPlayers.length === 0 &&
          rkPosFilters.size === 0 && !rkSearch;
        if (emptyTitle) {
          emptyTitle.textContent = pausedEmpty ? 'Prospect rankings paused' : 'No prospects match';
        }
        if (emptyMsg) {
          emptyMsg.textContent = pausedEmpty
            ? 'The rookie pipeline is paused until the next draft cycle. Rankings will return when the class is rebuilt.'
            : 'Try clearing a filter or widening your position and tier selections.';
        }
      }
      if (headEl) headEl.style.display = 'none';
      if (countEl) countEl.style.display = 'none';
      return;
    }

    if (emptyEl) emptyEl.style.display = 'none';
    if (headEl) headEl.style.display = '';
    if (countEl) {
      countEl.textContent = 'Showing ' + list.length + ' of ' + rkAllPlayers.length + ' prospects';
      countEl.style.display = '';
    }

    // Band dividers only make sense on the default board order: all
    // positions, no search, sorted by overall rank ascending.
    var showDividers = rkPosFilters.size === 0 && !rkSearch &&
                       rkSortKey === 'rank' && rkSortDir === 'asc';
    var byPid = {};
    list.forEach(function(r) { byPid[r.player_id || ''] = r; });
    var html = '', lastBand = null;
    list.forEach(function(r) {
      var band = r.tag ? r.tag.band : null;
      if (showDividers && band && band !== lastBand) {
        html += rkBandDivider(band);
        lastBand = band;
      }
      html += rkBoardRow(r);
    });
    listEl.innerHTML = html;
    listEl.querySelectorAll('.pr-player-row').forEach(function(rowEl) {
      var r = byPid[rowEl.getAttribute('data-pid')];
      if (!r) return;
      rowEl.addEventListener('click', function() { rkOpenModal(r); });
      rowEl.addEventListener('keydown', function(ev) {
        if (ev.key === 'Enter' || ev.key === ' ') {
          ev.preventDefault();
          rkOpenModal(r);
        }
      });
    });
  }

  // Export the current (filtered, sorted) board as CSV.
  function rkExportCSV() {
    var rows = [['Rank', 'Player', 'Position', 'School', 'Pos Rank',
                 'Production Pct', 'Efficiency Pct', 'Breakout Pct',
                 'Grade']];
    rkVisiblePlayers().forEach(function(r) {
      var gs = r.prospect_score == null || r.prospect_score === ''
        ? NaN : parseFloat(r.prospect_score);
      rows.push([
        r.overall_rank != null ? r.overall_rank : '',
        r.name || '', (r.position || '').toUpperCase(), r.school || '',
        rkPosRankLabel(r),
        rkPctOf(r, 'production') !== null ? rkPctOf(r, 'production') : '',
        rkPctOf(r, 'efficiency') !== null ? rkPctOf(r, 'efficiency') : '',
        rkPctOf(r, 'breakout') !== null ? rkPctOf(r, 'breakout') : '',
        isNaN(gs) ? '' : gs.toFixed(1)
      ]);
    });
    var csv = rows.map(function(row) {
      return row.map(function(c) {
        var s = String(c == null ? '' : c);
        return /[",\\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
      }).join(',');
    }).join('\\n');
    var blob = new Blob([csv], { type: 'text/csv' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'rookie-prospects-' + (rkViewSel || rkDraftYear || 'class') + '.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function() { URL.revokeObjectURL(a.href); }, 1000);
  }

  function rkSetSortKey(key) {
    rkSortKey = key;
    rkSortDir = RK_SORT_DIRS[key] || 'desc';
    rkRender();
  }
  // Clickable column headers: clicking the active key flips direction,
  // clicking a new key takes its default direction. The dropdown is kept
  // in sync so both controls always agree.
  function rkHeaderSort(key) {
    if (rkSortKey === key) rkSortDir = (rkSortDir === 'desc' ? 'asc' : 'desc');
    else { rkSortKey = key; rkSortDir = RK_SORT_DIRS[key] || 'desc'; }
    var sel = document.getElementById('rkSort');
    if (sel) sel.value = rkSortKey;
    rkRender();
  }

  // Sync the open modal's star + nudge row after a watchlist change.
  function rkSyncModalWatch(card, pid, name) {
    var on = !!rkWatch[pid];
    card.querySelectorAll('.rk-star').forEach(function(s) {
      s.classList.toggle('watching', on);
      if (s.classList.contains('player-modal-watchlist-btn')) {
        s.classList.toggle('player-modal-watchlist-btn--active', on);
        s.setAttribute('aria-pressed', on ? 'true' : 'false');
        s.title = on ? 'Remove from watchlist' : 'Add to watchlist';
      }
    });
    var n = card.querySelector('.m-nudge');
    if (n) {
      n.classList.toggle('on', on);
      var sp = n.querySelector('span');
      if (sp) sp.textContent = on ? 'You will be nudged if ' + name + ' changes tiers.'
                                  : 'Get nudged if ' + name + ' changes tiers.';
    }
  }
  function rkToggleWatchModal(ev, pid, name) {
    ev.stopPropagation();
    if (rkWatch[pid]) {
      delete rkWatch[pid];
      rkSaveWatch();
    } else {
      rkWatch[pid] = true;
      rkSaveWatch();
      if (window.brUndoToast) {
        window.brUndoToast('Watching ' + name + '.', function() {
          delete rkWatch[pid];
          rkSaveWatch();
          rkRender();
          var card = document.getElementById('rkModalContent');
          if (card) rkSyncModalWatch(card, pid, name);
        });
      }
    }
    rkRender();
    var card = document.getElementById('rkModalContent');
    if (card) rkSyncModalWatch(card, pid, name);
  }

  function rkTogglePos(pos) {
    if (pos === 'ALL') {
      rkPosFilters.clear();
    } else {
      if (rkPosFilters.has(pos)) rkPosFilters.delete(pos);
      else rkPosFilters.add(pos);
    }
    document.querySelectorAll('.pos-pill').forEach(function(b) {
      var p = b.getAttribute('data-pos');
      b.classList.toggle('active', p === 'ALL' ? rkPosFilters.size === 0 : rkPosFilters.has(p));
    });
    rkRender();
  }

  function rkClearSearch() {
    document.getElementById('rookieSearch').value = '';
    rkSearch = '';
    document.getElementById('rookieSearchClear').style.display = 'none';
    rkRender();
  }

</script>
""" + """
<script>

  function rkLoadingHtml() {
    return '<div class="rk-detail-loading"><div class="loading-spinner" style="width:12px;height:12px;flex-shrink:0;"></div>Loading\\u2026</div>';
  }

  // ── Prospect report card modal ─────────────────────────────────────
  // Single-scroll report card (approved mock): navy grade header,
  // archetype band, rail (comparables + class standing) beside the
  // main column (production chart, season lines, grouped metric
  // breakdown, numbered scouting notes). Every value renders only
  // when the API returns it; a section with no data is omitted, never
  // filled with a placeholder.
  function rkFmtNum(v) {
    if (v == null || v === '') return '-';
    var n = Number(v);
    if (isNaN(n)) return '-';
    return n >= 1000 ? n.toLocaleString() : String(Math.round(n));
  }

  function rkOrdinal(n) {
    var s = ['th', 'st', 'nd', 'rd'], v = n % 100;
    return n + (s[(v - 20) % 10] || s[v] || s[0]);
  }

  function rkDisplayPercentiles(r, d) {
    if (d && d.percentiles) return d.percentiles;
    return (r && r.percentiles) || {};
  }

  // Name split for the report-card header: everything but the last
  // token is the first name (a generational suffix stays with the
  // last name).
  function rkRcNameParts(name) {
    var parts = String(name || '').trim().split(/\\s+/).filter(Boolean);
    if (!parts.length) return { first: '', last: '' };
    if (parts.length === 1) return { first: '', last: parts[0] };
    var suffixes = { jr: 1, sr: 1, ii: 1, iii: 1, iv: 1, v: 1 };
    var lastIdx = parts.length - 1;
    if (lastIdx > 1 && suffixes[parts[lastIdx].toLowerCase().replace(/\\./g, '')]) lastIdx--;
    return { first: parts.slice(0, lastIdx).join(' '), last: parts.slice(lastIdx).join(' ') };
  }

  function rkRcInitials(name) {
    var p = rkRcNameParts(name);
    var a = (p.first || p.last || '?').charAt(0);
    var b = p.first && p.last ? p.last.charAt(0) : '';
    return (a + b).toUpperCase();
  }

  // Headshot error fallback: a dead CDN URL swaps back to the initials
  // tile the photo replaced, so the hero/mini never shows a broken hole.
  function rkRcImgFallback(img) {
    var mini = img.hasAttribute('data-fb-mini');
    var tile = document.createElement(mini ? 'span' : 'div');
    tile.className = mini ? 'rc-mini-initials' : 'rc-initials';
    tile.innerHTML = mini
      ? rkEsc(rkRcInitials(img.getAttribute('data-fb-name') || ''))
      : '<b>' + rkEsc(rkRcInitials(img.getAttribute('data-fb-name') || '')) + '</b>' +
        (img.getAttribute('data-fb-school')
          ? '<span>' + rkEsc(img.getAttribute('data-fb-school')) + '</span>' : '');
    img.replaceWith(tile);
  }

  function rkRcGradeOf(r, d) {
    var v = (d && d.prospect_score != null && d.prospect_score !== '')
      ? d.prospect_score : r.prospect_score;
    var n = v == null || v === '' ? NaN : parseFloat(v);
    return isNaN(n) ? null : n;
  }

  // Grade number + context line ("WR1 · Elite Prospect") in the navy
  // header; the detail bundle's grade and tag win over the board row's.
  function rkRcFillContext(r, d) {
    var g = document.getElementById('rkRcGrade');
    if (g) {
      var gv = rkRcGradeOf(r, d);
      g.textContent = gv == null ? '-' : gv.toFixed(1);
    }
    var el = document.getElementById('rkRcContext');
    if (!el) return;
    var tag = (d && d.tag) || r.tag || {};
    var lead = rkPosRankLabel(r) ||
      (r.overall_rank != null && r.overall_rank !== '' ? '#' + r.overall_rank : '');
    var parts = [];
    if (lead) parts.push(rkEsc(lead));
    if (tag.label) parts.push('<span class="rc-band-name">' + rkEsc(tag.label) + '</span>');
    el.innerHTML = parts.join('<span class="rc-dot">&middot;</span>');
    // In-Season pill beside the grade context: shown when the board
    // row flags the player in-season or the detail bundle carries
    // an in-progress season line.
    var pill = document.getElementById('rkRcInSeason');
    if (pill) {
      var inSeason = !!(r && r.in_season);
      if (!inSeason && d && d.season_lines) {
        inSeason = d.season_lines.some(function(l) { return l.in_progress; });
      }
      pill.style.display = inSeason ? '' : 'none';
    }
    // Sticky mini-header mirrors the grade + band.
    var mg = document.getElementById('rkRcMiniGrade');
    if (mg) {
      var gv2 = rkRcGradeOf(r, d);
      mg.textContent = gv2 == null ? '-' : gv2.toFixed(1);
    }
    var mb = document.getElementById('rkRcMiniBand');
    if (mb) mb.textContent = tag.label || '';
  }

  // Header note: the breakout percentile against the drafted pool,
  // from the same percentiles payload the board carries.
  function rkRcFillHeroNote(r, d) {
    var el = document.getElementById('rkRcHeroNote');
    if (!el) return;
    var b = rkDisplayPercentiles(r, d).breakout;
    if (b == null || b === '' || isNaN(+b)) { el.style.display = 'none'; el.innerHTML = ''; return; }
    var pos = (r.position || '').toUpperCase();
    var years = '';
    var m = /(\\d{4})\\s*-\\s*(\\d{4})/.exec(rkPercentilePool || '');
    if (m) years = ', ' + m[1] + '-' + m[2];
    el.innerHTML = '<strong>' + rkOrdinal(Math.round(+b)) +
      ' percentile breakout</strong> vs drafted ' + rkEsc(pos) + 's' + years;
    el.style.display = '';
  }

  // Archetype band: the stored label is "<role> · <style>" (a single
  // role when unpaired); the evidence line splits the stored receipt.
  function rkRcRenderArch(r, d) {
    var host = document.getElementById('rkRcArch');
    if (!host) return;
    var arch = (d && d.archetype) || r.archetype || '';
    if (!arch) { host.style.display = 'none'; host.innerHTML = ''; return; }
    var parts = String(arch).split(' · ');
    var title = parts[0];
    var chip = parts.slice(1).join(' · ');
    var receipt = (d && d.archetype_receipt) || r.archetype_receipt || '';
    var chips = receipt
      ? receipt.split(/[·;]/).map(function(s) { return s.trim(); }).filter(Boolean)
      : [];
    var conf = (d && d.archetype_confidence) || r.archetype_confidence || '';
    var lit = conf === 'High' ? 3 : conf === 'Medium' ? 2 : conf === 'Low' ? 1 : 0;
    var html = '<div class="rc-arch-main">' +
      '<div class="rc-arch-label">Projected NFL Role / Archetype</div>' +
      '<div class="rc-arch-row"><h3 class="rc-arch-title">' + rkEsc(title) + '</h3>' +
      (chip ? '<span class="rc-arch-tag">' + rkEsc(chip) + '</span>' : '') + '</div>' +
      (chips.length
        ? '<p class="rc-arch-ev" aria-label="Archetype evidence">' + chips.map(function(c) {
            return rkEsc(c).replace(/(\\d[\\d.,]*%?)/, '<b>$1</b>');
          }).join('<span class="rc-ev-sep" aria-hidden="true">·</span>') + '</p>'
        : '') +
      '</div>';
    if (conf) {
      var segs = '';
      for (var i = 0; i < 3; i++) {
        segs += '<span class="rc-seg' + (i < lit ? ' on' : '') + '"></span>';
      }
      html += '<div class="rc-arch-conf"><div class="rc-arch-conf-label">Confidence: ' +
        rkEsc(conf) + '</div><div class="rc-meter">' + segs + '</div></div>';
    }
    host.innerHTML = html;
    host.style.display = '';
  }

  // Positional peers in board order, shared by the class-standing
  // panel. Null when the board cannot place the player.
  function rkRcPeers(r) {
    if (!r || !r.position || !rkAllPlayers.length) return null;
    var pos = (r.position || '').toUpperCase();
    var peers = rkAllPlayers.filter(function(x) {
      return (x.position || '').toUpperCase() === pos;
    });
    if (peers.length < 2) return null;
    peers.sort(function(a, b) {
      var ar = a.position_rank > 0 ? +a.position_rank : (a.overall_rank > 0 ? +a.overall_rank : 9999);
      var br = b.position_rank > 0 ? +b.position_rank : (b.overall_rank > 0 ? +b.overall_rank : 9999);
      return ar - br;
    });
    var idx = -1;
    for (var i = 0; i < peers.length; i++) {
      if (peers[i].player_id === r.player_id) { idx = i; break; }
    }
    if (idx < 0) return null;
    return { peers: peers, idx: idx, pos: pos };
  }

  // Class Standing panel (rail): position line, overall line, the
  // consensus mock-pick line, and the next player at the position.
  // Each row renders only from data the payloads actually carry.
  function rkRcRenderStanding(r) {
    var host = document.getElementById('rkRcStanding');
    if (!host) return;
    var ctx = rkRcPeers(r);
    var rows = '';
    if (ctx) {
      var myRank = r.position_rank > 0 ? +r.position_rank : ctx.idx + 1;
      var cls = r.draft_class_year || rkDraftYear || '';
      rows += '<div class="rc-kv"><span class="rc-k">Position</span><span class="rc-v">' +
        rkEsc(ctx.pos + myRank) + ' of ' + ctx.peers.length +
        (cls ? ' in the ' + cls + ' class' : '') + '</span></div>';
    }
    if (r.overall_rank != null && r.overall_rank !== '') {
      rows += '<div class="rc-kv"><span class="rc-k">Overall</span>' +
        '<span class="rc-v sky">#' + r.overall_rank + ' overall</span></div>';
    }
    var mockPick = r.projected_pick != null && r.projected_pick !== ''
      ? Math.round(parseFloat(r.projected_pick)) : null;
    if (mockPick != null) {
      var round = r.projected_round != null && r.projected_round !== ''
        ? parseInt(r.projected_round, 10) : Math.ceil(mockPick / 32);
      rows += '<div class="rc-kv"><span class="rc-k">Mock draft</span><span class="rc-v">' +
        'Mock pick ' + mockPick + ' &middot; Round ' + round + '</span></div>';
    }
    var note = '';
    if (ctx && ctx.idx < ctx.peers.length - 1) {
      var nxt = ctx.peers[ctx.idx + 1];
      var nRank = nxt.position_rank > 0 ? +nxt.position_rank : ctx.idx + 2;
      note = '<p class="rc-panel-note">Chased by <strong>' + rkEsc(nxt.name || '') +
        '</strong> (' + rkEsc(ctx.pos + nRank) +
        (nxt.school ? ', ' + rkEsc(nxt.school) : '') + ')</p>';
    }
    if (!rows && !note) { host.style.display = 'none'; host.innerHTML = ''; return; }
    host.innerHTML = '<h3 class="rc-panel-h">Class Standing</h3>' + rows + note;
    host.style.display = '';
  }

  // Comparables (rail): initials disc, name, class/pick/school meta,
  // band chip. Same payload the old modal rendered.
  function rkRcRenderComps(comps) {
    var host = document.getElementById('rkRcComps');
    if (!host) return;
    if (!comps.length) {
      host.innerHTML = '<div class="rc-empty">No close historical comps found.</div>';
      return;
    }
    host.innerHTML = comps.map(function(c) {
      var meta = [c.draft_class_year,
                  c.actual_pick ? 'Pick ' + c.actual_pick : null,
                  c.school].filter(Boolean).join(' · ');
      return '<div class="rc-comp"><div class="rc-disc" style="--pc:' +
        rkDiscColor(c.player_id) + '">' + rkEsc(rkInitials(c.name)) + '</div>' +
        '<div class="rc-comp-info"><div class="rc-comp-name">' + rkEsc(c.name) + '</div>' +
        '<div class="rc-comp-meta">' + rkEsc(meta) + '</div></div>' +
        rkTagChip(c.tag) + '</div>';
    }).join('');
  }

  // Grade Breakdown: the four component percentiles (the same
  // display percentiles the hero note and the Competition row read)
  // as two-line rows, plus one generated line naming the lowest
  // component. A component with no percentile hides its row; the
  // section hides when none survive.
  var RK_RC_COMPONENTS = [
    { key: 'production', label: 'Production' },
    { key: 'efficiency', label: 'Efficiency' },
    { key: 'breakout', label: 'Breakout' },
    { key: 'competition', label: 'Competition' }
  ];

  function rkRcRenderGrade(r, d) {
    var host = document.getElementById('rkRcGradeBreakdown');
    var section = document.getElementById('rkRcGradeSection');
    if (!host) return;
    var pcts = rkDisplayPercentiles(r, d);
    var rows = [];
    RK_RC_COMPONENTS.forEach(function(c) {
      var v = pcts ? pcts[c.key] : null;
      if (v == null || v === '' || isNaN(+v)) return;
      rows.push({ label: c.label, pct: Math.round(+v) });
    });
    if (!rows.length) {
      host.innerHTML = '';
      if (section) section.style.display = 'none';
      return;
    }
    var html = rows.map(function(x) {
      return '<div class="rc-gb-row"><div class="rc-gb-line1"><span class="rc-gb-name">' +
        rkEsc(x.label) + '</span><span class="rc-gb-val">' + x.pct + '</span></div>' +
        '<span class="rc-mbar"><i style="width:' + x.pct + '%"></i></span></div>';
    }).join('');
    if (rows.length >= 2) {
      var low = rows[0];
      rows.forEach(function(x) { if (x.pct < low.pct) low = x; });
      var othersMin = null;
      rows.forEach(function(x) {
        if (x !== low && (othersMin === null || x.pct < othersMin)) othersMin = x.pct;
      });
      html += '<p class="rc-read-line">' + rkEsc(low.label) +
        ' is the soft spot: ' + rkOrdinal(low.pct) +
        ' percentile, every other component ' + rkOrdinal(othersMin) +
        ' or higher.</p>';
    }
    host.innerHTML = html;
    var side = document.getElementById('rkRcGradeSide');
    if (side) {
      var pos = (r.position || '').toUpperCase();
      side.textContent = pos ? 'Percentile vs drafted ' + pos + 's' : '';
    }
    if (section) section.style.display = '';
  }

  // Production by Year: the primary per-game rate for every stored
  // season, scaled to the peak; the in-progress season's bar is sky
  // and labeled. Trend chip compares the first and latest seasons.
  function rkRcRenderChart(r, d) {
    var host = document.getElementById('rkRcChart');
    if (!host) return;
    var lines = ((d && d.season_lines) || []).filter(function(l) { return l.rate != null; });
    var trend = document.getElementById('rkRcTrend');
    if (!lines.length) {
      host.innerHTML = '<div class="rc-empty">No season rates yet.</div>';
      if (trend) { trend.style.display = 'none'; trend.innerHTML = ''; }
      return;
    }
    var peak = 0;
    lines.forEach(function(l) { if (+l.rate > peak) peak = +l.rate; });
    var cols = 'repeat(' + lines.length + ', 1fr)';
    // Bars scale to the peak rate: a 120px ceiling on phones (the
    // value label rides above the bar inside the column, so the
    // tallest season never crowds the section head), 168px on
    // wider layouts.
    var maxH = window.innerWidth <= 860 ? 120 : 168;
    var bars = '', labels = '';
    lines.forEach(function(l) {
      var h = peak > 0 ? Math.max(14, Math.round((+l.rate) / peak * maxH)) : 14;
      bars += '<div class="rc-barcol"><div class="rc-barval' +
        (l.in_progress ? ' live' : '') + '">' + (+l.rate).toFixed(1) + '</div>' +
        '<div class="rc-bar' + (l.in_progress ? ' live' : '') +
        '" style="height:' + h + 'px"></div></div>';
      var detail = [l.team, l.gp != null ? l.gp + ' games' : '']
        .filter(Boolean).join(' · ');
      labels += '<div class="rc-barlab"><div class="rc-yr' +
        (l.in_progress ? ' liveyr' : '') + '">' + l.season + '</div>' +
        (detail ? '<div class="rc-detail">' + rkEsc(detail) + '</div>' : '') +
        (l.in_progress ? '<div class="rc-inprog">In Progress</div>' : '') + '</div>';
    });
    host.innerHTML =
      '<p class="rc-subline">' + rkEsc(lines[0].rate_label || '') + ', by season</p>' +
      '<div class="rc-chart" style="grid-template-columns:' + cols + '">' + bars + '</div>' +
      '<div class="rc-barlabels" style="grid-template-columns:' + cols + '">' + labels + '</div>';
    if (trend) {
      var t = '';
      if (lines.length >= 2) {
        var f = lines[0], last = lines[lines.length - 1];
        if (+f.rate > 0) {
          var chg = Math.round(((+last.rate) - (+f.rate)) / (+f.rate) * 100);
          if (Math.abs(chg) >= 1) {
            t = 'Trend <b>' + (chg > 0 ? '+' : '') + chg + '% per-game vs ' + f.season + '</b>';
            trend.className = 'rc-trend' + (chg > 0 ? '' : ' down');
          }
        }
      }
      trend.innerHTML = t;
      trend.style.display = t ? '' : 'none';
    }
  }

  // Per-season dominator (fraction) from the detail bundle's
  // advanced_seasons, keyed by season year.
  function rkRcDomBySeason(d) {
    var by = {};
    ((d && d.advanced_seasons) || []).forEach(function(a) {
      if (a && a.season != null && a.dominator != null && a.dominator !== '') {
        by[a.season] = +a.dominator;
      }
    });
    return by;
  }

  // Year-over-year change of the primary per-game rate, as a signed
  // rounded percent; null when it cannot be computed or is flat.
  function rkRcRateDelta(lines, idx) {
    if (idx <= 0) return null;
    var prev = lines[idx - 1], cur = lines[idx];
    if (!prev || prev.rate == null || cur.rate == null || +prev.rate <= 0) return null;
    var dlt = Math.round(((+cur.rate) - (+prev.rate)) / (+prev.rate) * 100);
    return dlt !== 0 ? dlt : null;
  }

  function rkRcDeltaHtml(dlt) {
    if (dlt === null) return '';
    return '<span class="rc-yds-delta' + (dlt < 0 ? ' neg' : '') + '">' +
      (dlt > 0 ? '+' : '') + dlt + '%</span>';
  }

  // Season Lines, phone rendering: a strict grid of Season | GP |
  // Tgt/G | Yds/G | Yds/Rec | Dom on shared column edges. Columns
  // with no data anywhere stay out (the Dom column hides when no
  // season carries a dominator); the live season reads from its
  // tinted row and sky year alone, no pill.
  function rkRcSeasonGrid(lines, d, domBy) {
    var cols = (d && d.season_line_columns) || [];
    function colByLabel(re) {
      for (var i = 0; i < cols.length; i++) {
        if (re.test(cols[i].label || '')) return cols[i];
      }
      return null;
    }
    function cellVal(l, c) {
      var v = (l.cells || {})[c.key];
      return v != null ? (+v).toFixed(c.dp || 1) : '-';
    }
    var plan = [
      { h: 'Season', w: 'minmax(76px,1fr)', cls: 'rc-sg-season',
        val: function(l) { return String(l.season); } },
      { h: 'GP', w: '22px',
        val: function(l) { return l.gp != null ? String(l.gp) : '-'; } }
    ];
    var tgtCol = colByLabel(new RegExp('^Tgt/G$'));
    if (tgtCol) {
      plan.push({ h: 'Tgt/G', w: '40px',
        val: function(l) { return cellVal(l, tgtCol); } });
    }
    plan.push({ h: 'Yds/G', w: '42px', yds: true,
      val: function(l) { return l.rate != null ? (+l.rate).toFixed(1) : '-'; } });
    var yprCol = colByLabel(new RegExp('^Yds/(Rec|Carry)$'));
    if (yprCol) {
      plan.push({ h: yprCol.label, w: '56px',
        val: function(l) { return cellVal(l, yprCol); } });
    }
    var showDom = lines.some(function(l) { return domBy[l.season] != null; });
    if (showDom) {
      plan.push({ h: 'Dom', w: '32px',
        val: function(l) {
          var v = domBy[l.season];
          return v != null ? Math.round(v * 100) + '%' : '-';
        } });
    }
    var tpl = plan.map(function(c) { return c.w; }).join(' ');
    var html = '<div class="rc-sg-row rc-sg-head" style="grid-template-columns:' + tpl + '">' +
      plan.map(function(c) {
        return '<span class="rc-sg-h' + (c.cls ? ' ' + c.cls : '') +
          '" data-col="' + rkEsc(c.h) + '">' + rkEsc(c.h) + '</span>';
      }).join('') + '</div>';
    lines.forEach(function(l, idx) {
      html += '<div class="rc-sg-row' + (l.in_progress ? ' live' : '') +
        (idx === lines.length - 1 ? ' rc-sg-last' : '') +
        '" style="grid-template-columns:' + tpl + '">' +
        plan.map(function(c) {
          var inner = rkEsc(c.val(l));
          if (c.yds) {
            var dlt = rkRcRateDelta(lines, idx);
            if (dlt !== null) {
              inner = '<span class="rc-yds-val">' + inner + '</span>' +
                rkRcDeltaHtml(dlt);
            }
          }
          return '<span class="rc-sg-c' + (c.cls ? ' ' + c.cls : '') +
            '" data-col="' + rkEsc(c.h) + '">' + inner + '</span>';
        }).join('') + '</div>';
    });
    return html;
  }

  // Season Lines: the desktop table (the overview column set from
  // the API's season_line_columns, per-game rates only, IN PROGRESS
  // pill on the live season, a Dom column and a quiet YoY delta in
  // the rate column) plus the phone grid. Each layout shows one.
  function rkRcRenderSeasonTable(r, d) {
    var host = document.getElementById('rkRcSeasonTable');
    if (!host) return;
    var lines = (d && d.season_lines) || [];
    if (!lines.length) {
      host.innerHTML = '<div class="rc-empty">No season lines yet.</div>';
      return;
    }
    var cols = (d && d.season_line_columns) || [];
    var useCols = cols.filter(function(c) { return c.ov; });
    if (!useCols.length) {
      useCols = [{ key: '__rate', label: lines[0].rate_label || '', dp: 1 }];
    }
    var domBy = rkRcDomBySeason(d);
    var showDom = lines.some(function(l) { return domBy[l.season] != null; });
    // The rate column is the overview column whose label matches
    // the primary rate's label (WR/TE/QB); where no column carries
    // it (RB scrimmage rate), the desktop table skips the delta.
    var rateColIdx = -1;
    useCols.forEach(function(c, ci) {
      if (rateColIdx < 0 && c.label && c.label === lines[0].rate_label) rateColIdx = ci;
    });
    var html = '<div class="rk-table-scroll rc-season-desktop"><table class="rc-table"><thead><tr>' +
      '<th>Season</th><th>GP</th>' +
      useCols.map(function(c) { return '<th>' + rkEsc(c.label) + '</th>'; }).join('') +
      (showDom ? '<th>Dom</th>' : '') +
      '</tr></thead><tbody>';
    lines.forEach(function(l, li) {
      html += '<tr' + (l.in_progress ? ' class="live"' : '') + '><td>' + l.season +
        (l.in_progress ? '<span class="rc-livepill">In Progress</span>' : '') + '</td>' +
        '<td>' + (l.gp != null ? l.gp : '-') + '</td>' +
        useCols.map(function(c, ci) {
          var v = c.key === '__rate' ? l.rate : (l.cells || {})[c.key];
          var txt = v != null ? (+v).toFixed(c.dp || 1) : '-';
          if (ci === rateColIdx) {
            var dlt = rkRcRateDelta(lines, li);
            if (dlt !== null) {
              return '<td><span class="rc-yds-val">' + txt + '</span>' +
                rkRcDeltaHtml(dlt) + '</td>';
            }
          }
          return '<td>' + txt + '</td>';
        }).join('') +
        (showDom
          ? '<td>' + (domBy[l.season] != null
              ? Math.round(domBy[l.season] * 100) + '%' : '-') + '</td>'
          : '') +
        '</tr>';
    });
    html += '</tbody></table></div>' +
      '<div class="rc-season-grid">' + rkRcSeasonGrid(lines, d, domBy) + '</div>';
    host.innerHTML = html;
  }

  // Metric Breakdown groups: every advanced row the API builds for
  // the position lands in exactly one group by its nature; a row the
  // map does not name falls to Context. Component-score rows never
  // render here (they are grades, not stored-stat metrics).
  var RK_RC_GROUPS = [
    { label: 'Production', keys: ['Dominator', 'Best Dominator', 'Mkt Share', 'Market Share',
        'TD Share', 'Rec Yds/Gm', 'Rec Yds/Game', 'Scrim Yds/Gm', 'Rush Yds/Gm', 'Rush TDs'] },
    { label: 'Efficiency', keys: ['Yds/Rec', 'Career Yds/Rec', 'Catch%', 'Catch Rate',
        'Yds/Carry', 'Cmp%', 'TD:INT', 'Yds/Att', 'AY/A', 'WEPA'] },
    { label: 'Context', keys: ['Breakout Age', 'Tgt/Gm', 'Targets/Game', 'Speed Score'] }
  ];

  function rkRcMetricRow(a) {
    var pct = a.percentile != null && a.percentile !== '' && !isNaN(+a.percentile)
      ? +a.percentile : null;
    return '<div class="rc-metric-row"><span class="rc-m-name">' + rkEsc(a.label) + '</span>' +
      '<span class="rc-m-val">' + rkEsc(a.raw) + '</span>' +
      (pct != null
        ? '<span class="rc-mbar"><i style="width:' + pct + '%"></i></span>' +
          '<span class="rc-mpct' + (pct >= 90 ? ' high' : '') + '">' +
            rkOrdinal(pct) + '<span class="rc-mpct-word"> percentile</span></span>'
        : '<span class="rc-mbar empty"></span><span class="rc-mpct none">-</span>') +
      '</div>';
  }

  // Conference for the Context group's Competition row: the latest
  // completed season's stored conference (the same season the old
  // percentile panel anchored its Competition sub-line to).
  function rkRcConference(d) {
    var lines = (d && d.season_lines) || [];
    var line = null;
    for (var i = lines.length - 1; i >= 0; i--) {
      if (!lines[i].in_progress) { line = lines[i]; break; }
    }
    if (!line && lines.length) line = lines[lines.length - 1];
    if (!line) return '';
    var srow = null;
    ((d && d.seasons) || []).forEach(function(s) {
      if (s.season == line.season) srow = s;
    });
    return srow && srow.conference ? String(srow.conference) : '';
  }

  function rkRcRenderMetrics(r, d) {
    var host = document.getElementById('rkRcMetrics');
    var section = document.getElementById('rkRcMetricsSection');
    if (!host) return;
    var rows = ((d && d.advanced) || []).filter(function(a) {
      return a && a.raw != null && !a.component;
    });
    var byGroup = RK_RC_GROUPS.map(function() { return []; });
    rows.forEach(function(a) {
      var gi = 2;  // unmapped rows fall to Context
      RK_RC_GROUPS.forEach(function(g, i) {
        if (g.keys.indexOf(a.label) >= 0) gi = i;
      });
      byGroup[gi].push(a);
    });
    // Competition row: conference as the value, the component
    // percentile as its bar. Both already ride the detail payload.
    var conf = rkRcConference(d);
    if (conf) {
      var cp = rkDisplayPercentiles(r, d).competition;
      byGroup[2].push({ label: 'Competition', raw: conf,
                        percentile: cp != null ? cp : null });
    }
    var html = '';
    RK_RC_GROUPS.forEach(function(g, i) {
      if (!byGroup[i].length) return;
      html += '<div class="rc-group"><div class="rc-group-label">' + g.label +
        '</div><div class="rc-group-rows">' +
        byGroup[i].map(rkRcMetricRow).join('') + '</div></div>';
    });
    host.innerHTML = html;
    if (section) section.style.display = html ? '' : 'none';
  }

  // Scouting Notes: the model's stored key_reasons, numbered. The
  // projected-pick bullet is excluded (the pick lives in the header
  // meta and Class Standing now); capped at five.
  function rkRcRenderNotes(r, d) {
    var host = document.getElementById('rkRcNotes');
    var section = document.getElementById('rkRcNotesSection');
    if (!host) return;
    var src = (d && d.key_reasons) || r.key_reasons || '';
    var lines = String(src).split('\\n')
      .map(function(l) { return l.trim().replace(/^[·\\-•*]\\s*/, ''); })
      .filter(function(l) { return l && !/projected pick/i.test(l); })
      .slice(0, 5);
    if (!lines.length) {
      host.innerHTML = '';
      if (section) section.style.display = 'none';
      return;
    }
    host.innerHTML = lines.map(function(l, i) {
      return '<li><span class="rc-n">' + (i + 1) + '</span><span>' +
        rkEsc(l) + '</span></li>';
    }).join('');
    if (section) section.style.display = '';
  }
</script>
""" + """
<script>
  function rkOpenModal(r) {
    var modal   = document.getElementById('rkModal');
    var content = document.getElementById('rkModalContent');
    var pid  = r.player_id || '';
    var name = r.name || 'Unknown';
    var pos  = (r.position || '').toUpperCase();
    var on = !!rkWatch[pid];
    var np = rkRcNameParts(name);

    // Header meta line: position (sky) | school | class year.
    var metaParts = [];
    if (pos) metaParts.push('<span class="rc-pos">' + rkEsc(pos) + '</span>');
    if (r.school) metaParts.push('<span>' + rkEsc(r.school) + '</span>');
    if (r.draft_class_year) metaParts.push('<span>' + r.draft_class_year + ' Class</span>');
    var metaHtml = metaParts.join('<span class="rc-sep">|</span>');

    // Vitals line: age, height, weight (present values only).
    var vitals = [];
    var ageNum = parseFloat(r.age);
    if (r.age != null && !isNaN(ageNum)) vitals.push('<b>' + ageNum.toFixed(1) + ' yrs</b>');
    var ht = rkHeightStr(r);
    if (ht) vitals.push(rkEsc(ht));
    var wt = rkWeightStr(r);
    if (wt) vitals.push(rkEsc(wt));

    var grade = rkRcGradeOf(r, null);

    // Hero/mini photo: the rankings payload already carries the ESPN
    // college headshot (espnHeadshot/headshot_url); initials are only
    // the fallback for prospects with no photo or a dead URL.
    var hsUrl = r.espnHeadshot || r.headshot_url || '';
    var heroTile = hsUrl
      ? '<img class="rc-headshot" src="' + rkEsc(hsUrl) + '" alt="" loading="lazy" data-fb-name="' + rkEsc(name) + '" data-fb-school="' + rkEsc(r.school || '') + '" onerror="rkRcImgFallback(this)">'
      : '<div class="rc-initials"><b>' + rkEsc(rkRcInitials(name)) + '</b>' +
          (r.school ? '<span>' + rkEsc(r.school) + '</span>' : '') + '</div>';
    var miniTile = hsUrl
      ? '<img class="rc-mini-headshot" src="' + rkEsc(hsUrl) + '" alt="" loading="lazy" data-fb-mini="1" data-fb-name="' + rkEsc(name) + '" onerror="rkRcImgFallback(this)">'
      : '<span class="rc-mini-initials">' + rkEsc(rkRcInitials(name)) + '</span>';

    content.innerHTML =
      '<div class="rc-actions">' +
        '<button class="player-modal-watchlist-btn rk-star rc-icon-btn' +
          (on ? ' watching player-modal-watchlist-btn--active' : '') +
          '" data-rk-star="1" title="' + (on ? 'Remove from watchlist' : 'Add to watchlist') +
          '" aria-pressed="' + (on ? 'true' : 'false') + '" aria-label="Watch ' +
          rkEsc(name) + '"><span class="wl-star-glyph" aria-hidden="true">☆</span></button>' +
        '<button class="rc-icon-btn" onclick="rkCloseModal()" aria-label="Close">×</button>' +
      '</div>' +
      '<div class="rc-hero">' +
        '<div class="rc-hero-left">' +
          heroTile +
          '<div class="rc-nameblock">' +
            (np.first ? '<div class="rc-first">' + rkEsc(np.first) + '</div>' : '') +
            '<div class="rc-last">' + rkEsc(np.last || name) + '</div>' +
            '<div class="rc-meta1">' + metaHtml + '</div>' +
            (vitals.length
              ? '<div class="rc-vitals">' + vitals.join(' &nbsp;|&nbsp; ') + '</div>'
              : '') +
          '</div>' +
        '</div>' +
        '<div class="rc-hero-right">' +
          '<div class="rc-grade-block">' +
            '<div class="rc-grade-num" id="rkRcGrade">' +
              (grade == null ? '-' : grade.toFixed(1)) + '</div>' +
            '<div class="rc-grade-stack">' +
              '<div class="rc-grade-label">Prospect Grade</div>' +
              '<div class="rc-grade-context" id="rkRcContext"></div>' +
              '<span class="rc-inseason-pill" id="rkRcInSeason" style="display:none">In-Season</span>' +
            '</div>' +
          '</div>' +
          '<p class="rc-hero-note" id="rkRcHeroNote" style="display:none"></p>' +
        '</div>' +
      '</div>' +
      '<div class="rc-mini" id="rkRcMini" aria-hidden="true">' +
        miniTile +
        '<span class="rc-mini-name">' + rkEsc(name) + '</span>' +
        '<span class="rc-mini-grade"><span class="rc-mini-num" id="rkRcMiniGrade">-</span>' +
        '<span class="rc-mini-band" id="rkRcMiniBand"></span></span>' +
      '</div>' +
      '<section class="rc-arch" id="rkRcArch" style="display:none"></section>' +
      '<div class="rc-body">' +
        '<aside class="rc-rail">' +
          '<section class="rc-section rc-comps-section">' +
            '<div class="rc-sec-head rc-comps-head"><h3>Comparable Players</h3></div>' +
            '<p class="rc-subline rc-comps-sub">Historical comps from the BR Fantasy database</p>' +
            '<div id="rkRcComps">' + rkLoadingHtml() + '</div>' +
          '</section>' +
          '<div class="rc-panel" id="rkRcStanding" style="display:none"></div>' +
        '</aside>' +
        '<div class="rc-main">' +
          '<section class="rc-section rc-sec-grade" id="rkRcGradeSection" style="display:none">' +
            '<div class="rc-sec-head"><h3>Grade Breakdown</h3>' +
              '<span class="rc-side" id="rkRcGradeSide"></span></div>' +
            '<div id="rkRcGradeBreakdown"></div>' +
          '</section>' +
          '<section class="rc-section rc-sec-prod">' +
            '<div class="rc-sec-head"><h3>Production by Year</h3>' +
              '<span class="rc-trend" id="rkRcTrend" style="display:none"></span></div>' +
            '<div id="rkRcChart">' + rkLoadingHtml() + '</div>' +
          '</section>' +
          '<section class="rc-section rc-sec-season">' +
            '<div class="rc-sec-head"><h3>Season Lines</h3></div>' +
            '<div id="rkRcSeasonTable">' + rkLoadingHtml() + '</div>' +
          '</section>' +
          '<section class="rc-section rc-sec-metrics" id="rkRcMetricsSection">' +
            '<div class="rc-sec-head"><h3>Metric Breakdown</h3>' +
              '<span class="rc-side">Percentile</span></div>' +
            '<div class="rc-metric-head"><span>Metric</span><span class="r">Value</span>' +
              '<span></span><span class="r">Percentile</span></div>' +
            '<div id="rkRcMetrics">' + rkLoadingHtml() + '</div>' +
          '</section>' +
          '<section class="rc-section rc-sec-notes" id="rkRcNotesSection">' +
            '<div class="rc-sec-head"><h3>Scouting Notes</h3></div>' +
            '<ul class="rc-notes" id="rkRcNotes"></ul>' +
          '</section>' +
        '</div>' +
      '</div>' +
      '<footer class="rc-foot"><span class="rc-brand">BR Fantasy</span>' +
        '<p>BR Fantasy prospect report' +
          (rkPercentilePool ? ' &middot; ' + rkEsc(rkPercentilePool) : '') +
          ' &middot; Data: BR Fantasy / CFBD</p></footer>';

    // Watchlist star (direct binding, no inline handler).
    var starBtn = content.querySelector('[data-rk-star]');
    if (starBtn) starBtn.addEventListener('click', function(ev) {
      rkToggleWatchModal(ev, pid, name);
    });
    rkRcFillContext(r, {});
    rkRcFillHeroNote(r, {});
    rkRcRenderArch(r, {});
    // In the combined (All) view the board row's ranks are combined
    // ranks, so the class-standing panel waits for the detail payload
    // (which carries class ranks) instead of mislabeling them.
    if (rkViewSel !== 'all') rkRcRenderStanding(r);
    rkRcRenderGrade(r, {});
    rkRcRenderNotes(r, {});
    // Archive rows carry no seasons, advanced metrics, or comps: the
    // sections that would render them are removed outright, never
    // shown as empty placeholders.
    if (r.archive) {
      ['.rc-sec-prod', '.rc-sec-season', '.rc-comps-section',
       '#rkRcMetricsSection'].forEach(function(sel) {
        var sec = content.querySelector(sel);
        if (sec) sec.style.display = 'none';
      });
    }
    content.scrollTop = 0;
    rkRcSetupMini();

    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';

    // Auto-link to Sleeper ID silently in background (stored
    // prospects only; archive ids have no Sleeper link to make).
    if (pid && !r.archive) {
      fetch('/api/prospects/auto-link/' + encodeURIComponent(pid)).catch(function(){});
    }

    // Detail bundle: season lines, advanced metrics, archetype,
    // percentiles, key reasons.
    var yearQ = r.draft_class_year || rkDraftYear || '';
    fetch('/api/prospects/player/' + encodeURIComponent(pid) + '?year=' + encodeURIComponent(yearQ))
      .then(function(res){ return res.json(); })
      .then(function(d) { rkFillDetail(r, d || {}); })
      .catch(function() {
        ['rkRcChart', 'rkRcSeasonTable', 'rkRcMetrics'].forEach(function(id) {
          var el = document.getElementById(id);
          if (el) el.innerHTML = '<span style="font-size:13px;color:var(--text-muted);">Details unavailable.</span>';
        });
      });

    // Historical comparables from real past draft classes (stored
    // prospects only; the archive has no comps surface).
    if (!r.archive) fetch('/api/prospects/comparables/' + encodeURIComponent(pid))
      .then(function(res){ return res.json(); })
      .then(function(cd) { rkRcRenderComps((cd && cd.comparables) || []); })
      .catch(function() {
        var cb = document.getElementById('rkRcComps');
        if (cb) cb.innerHTML = '<span style="font-size:13px;color:var(--text-muted);">Could not load comparables.</span>';
      });
  }

  function rkFillDetail(r, d) {
    rkRcFillContext(r, d);
    rkRcFillHeroNote(r, d);
    rkRcRenderArch(r, d);
    rkRcRenderGrade(r, d);
    rkRcRenderNotes(r, d);
    rkRcRenderChart(r, d);
    rkRcRenderSeasonTable(r, d);
    rkRcRenderMetrics(r, d);
    if (rkViewSel === 'all') rkRcRenderClassStanding(r, d);
  }

  // Class Standing for the combined (All) view, rendered from the
  // detail payload because the board row's ranks there are combined
  // ranks. Only lines the payload can actually support are rendered:
  // archive details carry class sizes, stored details carry class
  // ranks; a line with no data behind it is omitted, never faked.
  function rkRcRenderClassStanding(r, d) {
    var host = document.getElementById('rkRcStanding');
    if (!host || !d) return;
    var pos = (r.position || '').toUpperCase();
    var cls = d.draft_class_year || r.draft_class_year || '';
    var rows = '';
    if (d.position_rank && d.position_class_size) {
      rows += '<div class="rc-kv"><span class="rc-k">Position</span><span class="rc-v">' +
        rkEsc(pos + d.position_rank) + ' of ' + d.position_class_size +
        (cls ? ' in the ' + cls + ' class' : '') + '</span></div>';
    }
    if (d.overall_rank != null && d.overall_rank !== '') {
      rows += '<div class="rc-kv"><span class="rc-k">Overall</span>' +
        '<span class="rc-v sky">#' + d.overall_rank + ' overall' +
        (d.class_size ? ' of ' + d.class_size : '') +
        (cls ? ' · ' + cls + ' class' : '') + '</span></div>';
    }
    var pick = d.actual_pick != null && d.actual_pick !== ''
      ? Math.round(parseFloat(d.actual_pick)) : null;
    if (pick != null && !isNaN(pick)) {
      rows += '<div class="rc-kv"><span class="rc-k">Draft pick</span><span class="rc-v">' +
        'Pick ' + pick + ' &middot; Round ' + Math.ceil(pick / 32) + '</span></div>';
    }
    var note = '';
    if (d.archive) {
      note = '<p class="rc-panel-note">Graded with the current model' +
        (d.model_version ? ' (' + rkEsc(d.model_version) + ')' : '') +
        '. Not the board as it stood on draft day.</p>';
    }
    if (!rows && !note) { host.style.display = 'none'; host.innerHTML = ''; return; }
    host.innerHTML = '<h3 class="rc-panel-h">Class Standing</h3>' + rows + note;
    host.style.display = '';
  }

  // Sticky mini-header: observe the hero inside the modal's own
  // scroll container; the bar slides in once the hero leaves the
  // scroll viewport and back out when it returns.
  var rkMiniObserver = null;

  function rkRcSetupMini() {
    var content = document.getElementById('rkModalContent');
    var mini = document.getElementById('rkRcMini');
    var hero = content ? content.querySelector('.rc-hero') : null;
    if (rkMiniObserver) { rkMiniObserver.disconnect(); rkMiniObserver = null; }
    if (!content || !mini || !hero || !('IntersectionObserver' in window)) return;
    mini.classList.remove('is-visible');
    mini.setAttribute('aria-hidden', 'true');
    rkMiniObserver = new IntersectionObserver(function(entries) {
      var vis = !(entries.length && entries[0].isIntersecting);
      mini.classList.toggle('is-visible', vis);
      mini.setAttribute('aria-hidden', vis ? 'false' : 'true');
    }, { root: content, threshold: 0 });
    rkMiniObserver.observe(hero);
  }

  function rkCloseModal() {
    if (rkMiniObserver) { rkMiniObserver.disconnect(); rkMiniObserver = null; }
    document.getElementById('rkModal').style.display = 'none';
    document.body.style.overflow = '';
  }

  // Close on backdrop click
  document.getElementById('rkModal').addEventListener('click', function(e) {
    if (e.target === this) rkCloseModal();
  });

  // ── Load data ──────────────────────────────────────────────────────────────
  // The Class select (like Sort) joins the site's custom dropdown
  // (CSD). The app shell enhances selects on full loads and on
  // soft-nav roots; this explicit call covers the prospects root
  // itself, following the house pattern for page-owned selects.
  if (window.initCustomSelects) window.initCustomSelects(document.getElementById('prospectsCard'));

  (function() {
    var inp = document.getElementById('rookieSearch');
    var clr = document.getElementById('rookieSearchClear');
    inp.addEventListener('input', function() {
      rkSearch = inp.value.trim();
      clr.style.display = rkSearch.length > 0 ? 'block' : 'none';
      rkRender();
    });
  })();

  // Clickable column headers (native board pattern): click a column to
  // sort by it, click again to flip direction. Bound once; targets resolve
  // at event time so soft-nav re-renders keep working.
  if (!window.__rkSortBound) {
    window.__rkSortBound = true;
    (function() {
      function rkHeaderCell(e) {
        var cell = e.target && e.target.closest ? e.target.closest('#rkTableHeader [data-pr-sort-col]') : null;
        return cell || null;
      }
      function rkHeaderKey(cell) {
        var col = cell.getAttribute('data-pr-sort-col');
        if (col === 'player') return 'name';
        if (col === 'prod') return 'prod';
        if (col === 'eff') return 'eff';
        if (col === 'brk') return 'brk';
        return rkSortKey;
      }
      document.addEventListener('click', function(e) {
        var cell = rkHeaderCell(e);
        if (cell) rkHeaderSort(rkHeaderKey(cell));
      });
      document.addEventListener('keydown', function(e) {
        if (e.key !== 'Enter' && e.key !== ' ') return;
        var cell = rkHeaderCell(e);
        if (!cell) return;
        e.preventDefault();
        rkHeaderSort(rkHeaderKey(cell));
      });
    })();
  }

  fetch('/api/prospects/active-class')
    .then(function(r){ return r.json(); })
    .then(function(d) {
      rkDraftYear = d.draft_class_year || new Date().getFullYear();
      // Fetch draft status
      return fetch('/api/prospects/draft-status?year=' + rkDraftYear);
    })
    .then(function(r){ return r.json(); })
    .then(function(status) {
      rkDraftComplete = status.draft_complete || false;
      rkViewSel = String(rkDraftYear);
      return fetch('/api/prospects/rankings?year=' + rkDraftYear);
    })
    .then(function(r){ return r.json(); })
    .then(function(data) { rkApplyBoard(data); })
    .catch(function(err) {
      console.error('[rookies] Load error:', err);
      document.getElementById('rkLoading').innerHTML =
        '<div style="color:var(--text-muted);">Failed to load rookie data. Please refresh.</div>';
    });

  // Apply one rankings payload to the board: the initial active-class
  // load and every Class-selector switch share this path, so filters,
  // sort, count, and tags behave identically per class.
  function rkApplyBoard(data) {
    document.getElementById('rkLoading').style.display = 'none';
    rkAllPlayers = data.rankings || [];
    rkPercentilePool = data.percentile_pool || '';
    rkLoaded = true;
    rkPipelinePaused = !!data.paused;
    rkIsArchiveView = !!data.archive;
    rkShowYearMeta = rkIsArchiveView;
    // Subtitle slot: the class on show + the pool line generated
    // server-side from the reference population actually used for
    // the percentiles.
    var poolEl = document.getElementById('rkPoolLine');
    if (poolEl && data.percentile_pool) {
      var clsLabel = rkViewSel === 'all' ? 'All draft classes' :
        (rkViewSel || rkDraftYear) + ' draft class';
      poolEl.textContent = clsLabel + '. ' + data.percentile_pool + '.';
    }
    // Active-setting tags: the class on show, plus the In-Season tag
    // while any player on the board is mid-season.
    var settingsEl = document.getElementById('rkActiveSettings');
    if (settingsEl && rkViewSel) {
      document.getElementById('rkClassTag').textContent =
        rkViewSel === 'all' ? 'All Classes' : rkViewSel + ' Class';
      var anyInSeason = rkAllPlayers.some(function(p) { return p.in_season; });
      document.getElementById('rkSeasonTag').style.display = anyInSeason ? '' : 'none';
      settingsEl.style.display = '';
    }
    // Past-class honesty line: archive boards are graded with the
    // current model, not the board as it stood on draft day.
    var noteEl = document.getElementById('rkArchiveNote');
    if (noteEl) noteEl.style.display = rkIsArchiveView ? '' : 'none';
    var pausedBanner = document.getElementById('rkPausedBanner');
    if (pausedBanner) {
      if (rkPipelinePaused && rkAllPlayers.length) {
        pausedBanner.style.display = '';
        var asOf = document.getElementById('rkPausedAsOf');
        if (asOf && data.last_updated) asOf.textContent = '(as of ' + data.last_updated + ')';
      } else {
        pausedBanner.style.display = 'none';
      }
    }
    rkRender();
  }

  // Class selector: fetch the chosen class (or the combined board)
  // and re-render through the same path as the initial load.
  function rkSetClass(sel) {
    rkViewSel = String(sel);
    rkLoaded = false;
    var loadEl = document.getElementById('rkLoading');
    if (loadEl) loadEl.style.display = '';
    fetch('/api/prospects/rankings?year=' + encodeURIComponent(rkViewSel))
      .then(function(r){ return r.json(); })
      .then(function(data) { rkApplyBoard(data); })
      .catch(function(err) {
        console.error('[rookies] Class load error:', err);
        document.getElementById('rkLoading').innerHTML =
          '<div style="color:var(--text-muted);">Failed to load rookie data. Please refresh.</div>';
      });
  }
</script>
"""
    body = body.replace("<!--RK_CLASS_OPTIONS-->", _rk_class_options())
    return _inject_ssr_rows(body)
