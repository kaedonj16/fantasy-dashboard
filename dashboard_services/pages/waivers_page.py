"""
Waiver Wire / Start-Sit Advisor page builder.

Renders the HTML shell for /<platform>/<season>/<league_id>/waivers.
Actual data is loaded client-side via /api/waiver-candidates and
/api/start-sit-options.
"""


def build_waivers_body(platform: str, season: int, league_id: str, ctx: dict) -> str:
    """Return the full HTML body for the Waivers page."""

    style = """
<style>
.wv-page { padding: 16px; max-width: 1100px; margin: 0 auto; }
.wv-filters { margin-bottom: 16px; }
.wv-filter-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.wv-pos-pills { display: flex; gap: 6px; flex-wrap: wrap; }

/* Mobile tab bar */
.wv-tab-bar {
  display: none; border-radius: 10px; overflow: hidden;
  border: 1px solid var(--border); margin-bottom: 16px; background: var(--card);
}
.wv-tab-btn {
  flex: 1; padding: 10px 0; min-height: 44px; font-size: 13px; font-weight: 700;
  border: none; background: none; color: var(--text-muted); cursor: pointer;
  transition: background .15s, color .15s;
}
.wv-tab-btn.active { background: var(--accent); color: #fff; }
.wv-tab-btn:first-child { border-right: 1px solid var(--border); }

@media(max-width: 768px) {
  .wv-tab-bar { display: flex; }
  .wv-layout { grid-template-columns: 1fr !important; }
  .wv-section { display: none; }
  .wv-section.wv-tab-active { display: block; }
}

.wv-layout { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
/* Grid items default to min-width:auto, so a wide flex child (the horizontally
   scrolling trending strip) would force its column -- and the whole 2-column
   layout -- to expand past the viewport instead of scrolling. min-width:0 lets
   the column stay at its 1fr track and the strip scroll inside it. */
.wv-section { min-width: 0; }
@media(min-width: 769px) { .wv-section { display: block !important; } }
.wv-section-title { font-size: 14px; font-weight: 700; margin-bottom: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: .05em; }
.wv-loading { display: flex; justify-content: center; padding: 40px; }
/* Content-shaped loading placeholder -- mirrors the real rows so the layout
   doesn't jump when data arrives (replaces a bare centered spinner). */
.wv-skel { display: flex; flex-direction: column; gap: 8px; }
.wv-skel-row {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  padding: 12px; border-radius: 8px; background: var(--card); border: 1px solid var(--border);
}
.wv-skel-left { display: flex; flex-direction: column; gap: 7px; flex: 1; min-width: 0; }
.wv-skel-row .skeleton-line { margin: 0; }

/* Waiver wire rows -- one bordered container per list, rows separated by
   hairlines instead of each row owning a full border (no stack-of-boxes). */
.wv-list-card { background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; margin-bottom: 8px; }
.wv-player-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 13px 15px; background: transparent;
  border: 0; margin: 0; cursor: pointer; transition: background .12s;
}
.wv-player-row + .wv-player-row { border-top: 1px solid var(--border); }
.wv-player-row:hover { background: var(--row); }
.wv-player-name { font-weight: 600; font-size: 14px; color: var(--text); }
.wv-player-sub { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
/* Right cluster -- still used by the big-game strip (single metric + live tag). */
.wv-right { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; justify-content: flex-end; }
.wv-advice-metric { display: flex; flex-direction: column; align-items: center; gap: 3px; min-width: 0; }
.wv-advice-label {
  font-size: 9px; line-height: 1; font-weight: 800; letter-spacing: .05em;
  text-transform: uppercase; color: var(--text-subtle); white-space: nowrap;
}
.wv-value { font-size: 13px; font-weight: 700; color: var(--text); font-variant-numeric: tabular-nums; }

/* Best-moves grid: Player | Recommendation | Proj | Value. */
.wv-bm-head, .wv-bm-row { display: grid; grid-template-columns: minmax(180px,.8fr) minmax(220px,1.2fr) auto; gap: 16px; }
.wv-bm-head { padding: 9px 16px; border-bottom: 1px solid var(--border); background: var(--row); }
.wv-bm-head span { font-size: 9px; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; color: var(--text-subtle); }
.wv-bm-row { align-items: start; padding: 13px 16px; cursor: pointer; transition: background .12s; }
.wv-bm-row + .wv-bm-row { border-top: 1px solid var(--border); }
.wv-bm-row:hover { background: var(--row); }
.wv-bm-action { display: flex; flex-direction: column; gap: 6px; align-items: flex-start; min-width: 0; }
.wv-bm-chips { display: flex; flex-wrap: wrap; gap: 5px; min-width: 0; }
.wv-bm-claim { font-size: 10.5px; font-weight: 700; color: var(--text-subtle); line-height: 1.3; max-width: 100%; overflow-wrap: anywhere; }
.wv-bm-claim.hi { color: var(--warning); }
.wv-bm-main { min-width: 0; }
.wv-bm-name { font-weight: 700; font-size: 14px; color: var(--text); line-height: 1.2; overflow-wrap: anywhere; }
.wv-bm-sub { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
/* Proj / Value grouped into one right-hand cell so the columns line up and can
   collapse together on small screens. */
.wv-bm-nums, .wv-bm-h-nums { display: flex; gap: 18px; }
.wv-bm-num, .wv-bm-h-nums span { min-width: 46px; text-align: right; }
.wv-bm-num { line-height: 1.05; }
.wv-bm-num b { font-weight: 800; font-size: 16px; color: var(--text); font-variant-numeric: tabular-nums; letter-spacing: -.01em; }
.wv-bm-num.empty b { color: var(--text-subtle); font-weight: 700; }
.wv-bm-num i { display: block; font-style: normal; font-size: 9px; font-weight: 800; letter-spacing: .05em; text-transform: uppercase; color: var(--text-subtle); margin-top: 3px; }
.wv-ctx-links { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; min-width: 0; }
.wv-bm-links { grid-column: 1; margin-top: -8px; }
.wv-ctx-link { font-size: 11px; font-weight: 600; color: var(--accent); text-decoration: none; padding: 3px 8px; border-radius: 6px; border: 1px solid var(--border); background: var(--surface); }
.wv-ctx-link:hover { background: var(--accent-soft); }
.wv-section-heading { display: flex; align-items: center; justify-content: space-between; gap: 8px 12px; flex-wrap: wrap; }
.wv-faab-toggle { display: inline-flex; align-items: center; gap: 6px; font-size: 11px; color: var(--text-muted); cursor: pointer; }
/* The inline-flex above beats the UA stylesheet's [hidden] -> display:none, so
   without this the FAAB checkbox stays visible in non-FAAB leagues even when
   JS correctly sets hidden. Tapping it then re-renders nothing (no FAAB data). */
.wv-faab-toggle[hidden] { display: none; }
.wv-faab-toggle input { accent-color: var(--accent); width: 14px; height: 14px; margin: 0; }
@media (max-width: 700px) {
  /* Big-game strip: drop its single metric onto a full-width row under the name. */
  .wv-player-row { flex-direction: column; align-items: stretch; gap: 10px; }
  .wv-right {
    width: 100%; gap: 10px 14px; flex-wrap: wrap;
    justify-content: flex-start; padding-top: 10px; border-top: 1px solid var(--border);
  }
}
@media (max-width: 720px) {
  /* Compact card: identity first, fixed metric cluster upper-right, then
     recommendation, claim guidance, and contextual actions. */
  .wv-bm-head { display: none; }
  .wv-list-card { overflow: clip; }
  .wv-bm-row { grid-template-columns: minmax(0,1fr) max-content; gap: 9px 12px; padding: 13px 12px; }
  .wv-bm-main { grid-column: 1; grid-row: 1; }
  .wv-bm-nums { grid-column: 2; grid-row: 1; align-self: start; }
  .wv-bm-action { grid-column: 1 / -1; grid-row: 2; gap: 7px; }
  .wv-bm-links { grid-column: 1 / -1; grid-row: 3; margin-top: 0; }
  .wv-bm-chips { width: 100%; }
  .wv-bm-nums { gap: 14px; }
  .wv-bm-num, .wv-bm-h-nums span { min-width: 40px; }
  .wv-ctx-links { margin-top: 2px; }
  .wv-horizon { max-width: 100%; flex-wrap: wrap; }
}
/* Waiver signal chips use the site's canonical `.chip .chip--sm` + a .signal-*
   colour alias (all defined once in dashboard.css). Nothing chip-related is
   overridden here. */
.wv-usage-chip {
  display: inline-block; font-size: 10px; font-weight: 700; color: var(--win);
  margin-left: 6px; white-space: nowrap;
}
/* Add/drop pairing: the suggested cut to make room for this target. Muted so it
   reads as a secondary hint under the player, with a small position tag. */
.wv-drop-hint {
  font-size: 11px; color: var(--text-muted); margin-top: 3px;
  display: flex; align-items: center; gap: 5px;
}
.wv-drop-lbl {
  font-size: 9px; font-weight: 800; letter-spacing: .04em; text-transform: uppercase;
  color: var(--loss); background: color-mix(in srgb, var(--loss) 13%, transparent);
  padding: 1px 5px; border-radius: 4px;
}
.wv-drop-pos { font-weight: 700; color: var(--text); }

/* Recommendation-horizon segmented control (#2): this week / next 4 wks / stash. */
.wv-horizon { display: inline-flex; flex-wrap: wrap; gap: 4px; margin: 0 0 12px; }
.wv-horizon-btn {
  font-size: 11px; font-weight: 600; padding: 5px 10px; border-radius: 999px;
  border: 1px solid var(--border); background: transparent; color: var(--text-muted);
  cursor: pointer; white-space: nowrap;
}
.wv-horizon-btn.active {
  background: color-mix(in srgb, var(--accent) 15%, transparent);
  border-color: var(--accent); color: var(--text);
}
/* Unexpected-performances list shares the waiver-row look. */
.wv-biggames { margin-bottom: 8px; }
/* Muted chip fallback for watchlist / provisional tags. */
.chip--muted {
  background: color-mix(in srgb, var(--text-muted) 14%, transparent);
  color: var(--text-muted);
}

/* Trending-across-leagues strip: a horizontal, scrollable row of the most-added
   players Sleeper-wide that this league can still claim. */
.wv-trending-title { display: flex; align-items: center; gap: 6px; }
.wv-trending-title i { color: #f97316; }
.wv-trending-strip {
  display: flex; gap: 8px; overflow-x: auto; padding-bottom: 8px; margin-bottom: 18px;
  -webkit-overflow-scrolling: touch; scrollbar-width: thin;
}
.wv-trend-chip {
  flex: 0 0 auto; display: flex; flex-direction: column; gap: 2px; text-align: left;
  padding: 8px 12px; border-radius: 10px; background: var(--card);
  border: 1px solid var(--border); cursor: pointer; min-width: 120px; max-width: 148px;
  transition: border-color .12s;
}
.wv-trend-chip:hover { border-color: var(--accent); }
.wv-trend-adds {
  font-size: 11px; font-weight: 800; color: #f97316;
  display: flex; align-items: center; gap: 4px; font-variant-numeric: tabular-nums;
}
.wv-trend-name {
  font-size: 13px; font-weight: 700; color: var(--text); white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; max-width: 100%;
}
.wv-trend-sub { font-size: 11px; color: var(--text-muted); white-space: nowrap; }

/* Streaming this week: matchup-ranked D/ST and K, shown under the waiver list
   only in season. Implied-total chip is green for a good spot, red for a dud
   (meaning is inverted for defenses, where a low opponent total is good). */
.wv-stream-head { margin-top: 22px; }
.wv-stream-group { margin-bottom: 10px; }
.wv-stream-sub {
  font-size: 10px; font-weight: 700; color: var(--text-subtle, var(--text-muted));
  text-transform: uppercase; letter-spacing: .05em; margin: 4px 2px 6px;
}
.wv-stream-row {
  display: flex; align-items: center; gap: 8px; padding: 8px 12px; margin-bottom: 6px;
  border-radius: 8px; background: var(--card); border: 1px solid var(--border); cursor: pointer;
}
.wv-stream-row:hover { border-color: var(--accent); }
.wv-stream-name { font-weight: 700; font-size: 13px; color: var(--text); }
.wv-stream-matchup { font-size: 12px; color: var(--text-muted); flex: 1; }
.wv-stream-imp {
  font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: var(--radius-pill, 8px);
  font-variant-numeric: tabular-nums; white-space: nowrap;
}
.wv-stream-imp-good { background: color-mix(in srgb, var(--win) 16%, transparent); color: var(--win); }
.wv-stream-imp-bad  { background: color-mix(in srgb, var(--loss) 14%, transparent); color: var(--loss); }
.wv-stream-imp-mid  { background: rgba(148,163,184,.16); color: var(--text-muted); }

/* Lineup advice banner -- points left on the bench + suggested swaps */
.wv-ss-advice { border-radius: 10px; padding: 12px 14px; margin-bottom: 16px; border: 1px solid var(--border); }
.wv-ss-advice-ok { background: color-mix(in srgb, var(--win) 8%, transparent); border-color: color-mix(in srgb, var(--win) 30%, var(--border)); color: var(--text); font-size: 13px; font-weight: 600; }
.wv-ss-advice-ok i { color: var(--win); margin-right: 6px; }
.wv-ss-advice-warn { background: color-mix(in srgb, var(--accent) 7%, transparent); border-color: color-mix(in srgb, var(--accent) 30%, var(--border)); }
.wv-ss-advice-head { font-size: 14px; font-weight: 700; color: var(--text); }
.wv-ss-advice-head i { color: var(--accent); margin-right: 6px; }
.wv-ss-advice-head strong { color: var(--accent); }
.wv-ss-advice-sub { font-size: 11px; color: var(--text-muted); margin: 3px 0 10px; }
.wv-ss-swap { display: flex; align-items: center; gap: 8px; font-size: 13px; padding: 5px 0; border-top: 1px solid var(--border); flex-wrap: wrap; }
.wv-ss-swap-in { font-weight: 700; color: var(--win); }
.wv-ss-swap-arrow { font-size: 11px; color: var(--text-muted); }
.wv-ss-swap-out { font-weight: 600; color: var(--text-muted); text-decoration: line-through; }
.wv-ss-swap-pos { font-size: 10px; font-weight: 700; color: var(--text-muted); letter-spacing: .04em; }
.wv-ss-swap-slot { font-size: 9px; font-weight: 700; letter-spacing: .04em; color: var(--text-muted); border: 1px solid var(--border); border-radius: 4px; padding: 1px 5px; flex-shrink: 0; }
.wv-ss-swap-gain { margin-left: auto; font-weight: 800; font-size: 12px; color: var(--win); font-variant-numeric: tabular-nums; }
.wv-ss-swap-gain.wv-ss-swap-gain-neg { color: var(--loss); }
.wv-ss-winprob { font-size: 11px; color: var(--text-muted); margin-top: 4px; }
.wv-ss-winprob strong { color: var(--win); font-variant-numeric: tabular-nums; }
.wv-ss-demote { font-size: 10px; font-weight: 700; padding: 1px 6px; border-radius: 5px; background: color-mix(in srgb, var(--loss) 12%, transparent); color: var(--loss); margin-left: 6px; }

/* Start/Sit player cards -- one bordered container per position, players as
   hairline-separated rows (no card-in-card borders); each row leads with its
   verdict badge and pulls the projection out as a hero number. */
.wv-ss-pos-group { margin-bottom: 16px; background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
.wv-ss-pos-label { font-size: 12px; font-weight: 800; color: var(--text); text-transform: uppercase; letter-spacing: .05em; margin: 0; padding: 12px 15px 11px; border-bottom: 1px solid var(--border); background: var(--row); }
.wv-ss-player {
  padding: 13px 15px; border-radius: 0; background: transparent;
  border: 0; margin: 0;
}
.wv-ss-player + .wv-ss-player { border-top: 1px solid var(--border); }
.wv-ss-player.wv-ss-start  { background: color-mix(in srgb, var(--win) 6%, transparent); }
.wv-ss-player.wv-ss-bye    { opacity: .55; }
.wv-ss-player.wv-ss-selected { background: var(--accent-soft); box-shadow: inset 0 0 0 2px var(--accent); }

/* Body row under the name: demoted stat strip on the left, hero projection
   pinned to the right. The dashed rule separates the read (who + verdict) from
   the supporting detail. */
.wv-ss-body { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; margin-top: 9px; padding-top: 9px; border-top: 1px dashed var(--border); }
.wv-ss-body .wv-ss-stats { margin-top: 0; flex: 1 1 auto; min-width: 0; }
.wv-ss-proj { flex: 0 0 auto; text-align: right; line-height: 1; }
.wv-ss-proj-num { font-size: 23px; font-weight: 800; color: var(--text); font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
.wv-ss-proj-lbl { font-size: 9px; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; color: var(--text-subtle); margin-top: 3px; }

/* Top row: name + badge */
.wv-ss-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.wv-ss-name-block { display: flex; align-items: center; gap: 6px; flex: 1; min-width: 0; cursor: pointer; }
.wv-ss-actions { display: flex; align-items: center; gap: 6px; flex-shrink: 0; }
@media (max-width: 700px) {
  /* Name/badge and the action buttons cannot share one narrow row -- START/FLEX
     pills overlapped "Open player" and long names wrapped into the buttons. */
  .wv-ss-top { flex-direction: column; align-items: stretch; gap: 8px; }
  .wv-ss-name-block { flex-wrap: wrap; row-gap: 4px; }
  .wv-ss-actions { flex-wrap: wrap; flex-shrink: 1; width: 100%; }
  .wv-ss-player .wv-player-name {
    min-width: 0; max-width: 100%;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
}

/* Stats row under the name */
.wv-ss-stats { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 6px; align-items: stretch; }
.wv-ss-div { width: 1px; align-self: stretch; background: var(--border); margin: 1px 0; flex: 0 0 auto; }
.wv-ss-stat { display: flex; flex-direction: column; gap: 1px; }
.wv-ss-stat-lbl { font-size: 10px; color: var(--text-subtle); text-transform: uppercase; letter-spacing: .04em; font-weight: 600; }
.wv-ss-stat-val { font-size: 13px; font-weight: 700; color: var(--text); }
.wv-ss-stat-val.muted { color: var(--text-muted); }
.wv-ss-env, .wv-ss-total, .wv-ss-cons { font-size: 11px; font-weight: 700; padding: 1px 7px; border-radius: 6px; align-self: flex-start; }
.wv-ss-cons-steady   { background: color-mix(in srgb, var(--win) 16%, transparent); color: var(--win); }
.wv-ss-cons-balanced { background: rgba(148,163,184,.16); color: var(--text-muted); }
.wv-ss-cons-volatile { background: color-mix(in srgb, var(--warning) 18%, transparent); color: var(--warning); }
.wv-ss-cons-boombust { background: rgba(168,85,247,.16); color: #7e22ce; }
[data-theme="dark"] .wv-ss-cons-boombust { background: rgba(168,85,247,.24); color: #c084fc; }
/* Subtle "from last season" superscript on blended floor-ceiling */
.wv-yr { font-size: .68em; font-weight: 600; color: var(--text-muted); vertical-align: super; margin-left: 2px; opacity: .85; }
.wv-ss-env-dome { background: rgba(59,130,246,.14); color: #1d4ed8; }
.wv-ss-env-cold { background: rgba(56,189,248,.16); color: #0369a1; }
.wv-ss-env-wind { background: rgba(148,163,184,.20); color: #475569; }
.wv-ss-env-precip { background: rgba(59,130,246,.14); color: #1d4ed8; }
[data-theme="dark"] .wv-ss-env-dome { background: rgba(59,130,246,.20); color: #60a5fa; }
[data-theme="dark"] .wv-ss-env-cold { background: rgba(56,189,248,.20); color: #7dd3fc; }
[data-theme="dark"] .wv-ss-env-wind { background: rgba(148,163,184,.24); color: #cbd5e1; }
[data-theme="dark"] .wv-ss-env-precip { background: rgba(59,130,246,.22); color: #93c5fd; }
/* Vegas implied team total: green when high, muted when low */
.wv-ss-total-high { background: color-mix(in srgb, var(--win) 16%, transparent); color: var(--win); }
.wv-ss-total-mid  { background: rgba(148,163,184,.16); color: var(--text-muted); }
.wv-ss-total-low  { background: color-mix(in srgb, var(--loss) 15%, transparent); color: var(--loss); }

/* Matchup chip: a 4-step easy→hard scale, so the middle "ok" keeps its own
   lime step; the endpoints use the win/warning/loss tokens. */
.wv-mu { font-size: 11px; font-weight: 700; padding: 2px 7px; border-radius: 6px; }
.wv-mu-easy { background: color-mix(in srgb, var(--win) 16%, transparent); color: var(--win); }
.wv-mu-ok   { background: color-mix(in srgb, #84cc16 13%, transparent); color: #65a30d; }
.wv-mu-avg  { background: color-mix(in srgb, var(--warning) 16%, transparent); color: var(--warning); }
.wv-mu-hard { background: color-mix(in srgb, var(--loss) 15%, transparent); color: var(--loss); }
[data-theme="dark"] .wv-mu-ok { color: #a3e635; }

/* Opp plays faced: a tappable pace/possession stat. The value toggles a
   full-width detail panel (season / last-4 splits, sample, opp possession)
   kept collapsed so the card stays lean. Neutral colour throughout -- more
   plays is context, not a verdict. */
.wv-ss-plays-btn { cursor: pointer; border-bottom: 1px dashed var(--border); }
.wv-ss-plays-btn:hover { border-bottom-color: var(--accent); }
.wv-ss-plays-caret { font-size: .7em; margin-left: 3px; color: var(--text-muted); }
.wv-ss-plays-sub { font-size: 10px; color: var(--text-muted); margin-top: 1px; }
.wv-ss-plays-detail {
  flex-basis: 100%; display: none; margin-top: 6px; padding: 8px 10px;
  border-radius: 6px; background: var(--surface); border: 1px solid var(--border);
}
.wv-ss-plays-detail.open { display: block; }
.wv-ss-plays-detail-row {
  display: flex; align-items: baseline; justify-content: space-between;
  gap: 12px; font-size: 11px; padding: 2px 0;
}
.wv-ss-plays-detail-lbl { color: var(--text-muted); }
.wv-ss-plays-detail-val { font-weight: 700; color: var(--text); font-variant-numeric: tabular-nums; }

/* Badges */
.wv-ss-start-badge      { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: color-mix(in srgb, var(--win) 15%, transparent); color: var(--win); flex-shrink: 0; }
.wv-ss-flex-start-badge { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: color-mix(in srgb, var(--win) 15%, transparent); color: var(--win); border: 1px solid color-mix(in srgb, var(--win) 30%, transparent); flex-shrink: 0; }
.wv-ss-flex-badge       { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: color-mix(in srgb, var(--accent) 14%, transparent); color: var(--accent); flex-shrink: 0; }
.wv-ss-sit-badge        { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: var(--row); color: var(--text-muted); flex-shrink: 0; }
.wv-ss-bye-badge        { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: color-mix(in srgb, var(--warning) 15%, transparent); color: var(--warning); flex-shrink: 0; }
.wv-inj-out { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: color-mix(in srgb, var(--loss) 16%, transparent); color: var(--loss); }
.wv-inj-q   { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: color-mix(in srgb, var(--inj-q) 18%, transparent); color: var(--inj-q); }
.wv-inj-d   { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 6px; background: color-mix(in srgb, var(--orange) 16%, transparent); color: var(--orange); }

/* Compare button */
.wv-cmp-btn {
  font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 6px;
  border: 1px solid var(--border); background: transparent; color: var(--text-muted);
  cursor: pointer; transition: background .12s, color .12s, border-color .12s; flex-shrink: 0;
}
.wv-cmp-btn:hover { border-color: var(--accent); color: var(--accent); }
.wv-cmp-btn.selected { background: var(--accent); color: #fff; border-color: var(--accent); }

/* Compare panel */
.wv-compare-panel {
  background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
  margin-bottom: 16px; overflow: hidden;
}
.wv-compare-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 14px; border-bottom: 1px solid var(--border);
  font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em;
  color: var(--text-muted);
}
.wv-compare-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 0;
}
.wv-compare-col { padding: 14px; }
.wv-compare-col:first-child { border-right: 1px solid var(--border); }
.wv-compare-player-name { font-size: 15px; font-weight: 700; margin-bottom: 2px; }
.wv-compare-player-sub  { font-size: 12px; color: var(--text-muted); margin-bottom: 12px; }
.wv-compare-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 0; border-bottom: 1px solid var(--border); font-size: 13px;
}
.wv-compare-row:last-child { border-bottom: none; }
.wv-compare-lbl { color: var(--text-muted); font-size: 12px; }
.wv-compare-val { font-weight: 700; }
.wv-compare-win { color: var(--win); }
.wv-compare-lose { color: var(--loss); }

/* Shared-label compare: value(a) · LABEL · value(b), one label per metric */
.wv-cmp { padding: 6px 16px 14px; }
.wv-cmp-row, .wv-cmp-head {
  display: grid; grid-template-columns: 1fr minmax(86px, auto) 1fr;
  align-items: center; gap: 10px; padding: 9px 0;
  border-bottom: 1px solid color-mix(in srgb, var(--border) 55%, transparent);
}
.wv-cmp-row:last-child { border-bottom: none; }
.wv-cmp-head { border-bottom: 2px solid var(--border); padding: 4px 0 10px; align-items: end; }
/* Values are muted by default so a highlighted winner actually stands out; the
   winner keeps full-strength colour + a soft pill (rules below). */
.wv-cmp-a { text-align: right; font-weight: 700; font-size: 13px; color: var(--text-muted); }
.wv-cmp-b { text-align: left;  font-weight: 700; font-size: 13px; color: var(--text-muted); }
.wv-cmp-lbl {
  text-align: center; color: var(--text-subtle); font-size: 10.5px;
  text-transform: uppercase; letter-spacing: .04em; font-weight: 700;
}
/* Winner gets a soft green pill; the loser just stays neutral (no red -- a
   lower projection isn't a failure). The value is wrapped in .wv-cmp-v so the
   pill hugs the number instead of filling the whole column. */
.wv-cmp-v { display: inline-block; }
.wv-cmp-a.wv-compare-win, .wv-cmp-b.wv-compare-win { color: var(--win); font-weight: 800; }
.wv-compare-win .wv-cmp-v {
  background: color-mix(in srgb, var(--win) 15%, transparent);
  padding: 3px 10px; border-radius: 999px;
}
.wv-compare-lose { color: var(--text-muted); font-weight: 700; }
.wv-cmp-hcol:first-child { text-align: right; }
.wv-cmp-name { font-size: 15px; font-weight: 800; color: var(--text); line-height: 1.15; }
.wv-cmp-sub  { font-size: 11px; color: var(--text-muted); }
.wv-cmp-headmeta { display: flex; align-items: center; gap: 6px; margin-top: 5px; flex-wrap: wrap; }
.wv-cmp-hcol:first-child .wv-cmp-headmeta { justify-content: flex-end; }
.wv-cmp-poschip {
  font-size: 10px; font-weight: 800; letter-spacing: .02em;
  padding: 1px 7px; border-radius: var(--radius-pill, 8px); white-space: nowrap;
}
.wv-cmp-vs {
  align-self: center; justify-self: center; font-size: 10px; font-weight: 800;
  color: var(--text-subtle); letter-spacing: .06em;
  border: 1px solid var(--border); border-radius: var(--radius-pill, 8px); padding: 3px 7px;
}
/* Verdict banner: the advisor's actual call, up top where it's read first. */
.wv-cmp-verdict {
  display: flex; align-items: center; gap: 11px; flex-wrap: wrap;
  padding: 13px 16px; border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, color-mix(in srgb, var(--win) 12%, transparent), transparent);
}
.wv-cmp-verdict.toss { background: rgba(148,163,184,.10); }
.wv-cmp-verdict-pill {
  font-size: 10px; font-weight: 800; letter-spacing: .07em;
  padding: 3px 9px; border-radius: 6px; background: var(--win); color: #fff; flex-shrink: 0;
}
.wv-cmp-verdict.toss .wv-cmp-verdict-pill { background: var(--text-muted); }
.wv-cmp-verdict-name { font-size: 14px; font-weight: 800; color: var(--text); }
.wv-cmp-verdict-why  { font-size: 12px; color: var(--text-muted); }
</style>
"""

    _skel_widths = [46, 38, 52, 34, 44, 40]
    wv_skel = '<div class="wv-skel">' + ''.join(
        '<div class="wv-skel-row"><div class="wv-skel-left">'
        f'<div class="skeleton skeleton-line" style="width:{w}%;height:11px"></div>'
        f'<div class="skeleton skeleton-line" style="width:{max(22, w - 16)}%;height:9px"></div>'
        '</div><div class="skeleton" style="width:44px;height:22px;border-radius:8px"></div></div>'
        for w in _skel_widths
    ) + '</div>'

    is_bb = False
    try:
        from utils.league_format import is_best_ball
        is_bb = is_best_ball(
            ctx.get("league") or {},
            settings=(ctx.get("league_settings")
                      or (ctx.get("league") or {}).get("settings")
                      or ctx.get("settings") or {}),
        )
    except Exception:
        is_bb = False
    startsit_tab_html = (
        "" if is_bb else
        '<button class="wv-tab-btn" id="wvTabStartSit" onclick="wvSetTab(\'startsit\')">Start/Sit</button>'
    )
    startsit_section_hidden = " hidden" if is_bb else ""
    bb_note = (
        '<div class="muted" style="font-size:13px;margin:0 0 12px;">'
        'Best Ball league: weekly Start/Sit is hidden.</div>'
        if is_bb else ""
    )

    html_body = f"""
<div class="wv-page">
  <!-- Position filter pills -->
  <div class="wv-filters">
    <div class="wv-filter-row">
      <div class="otc-day-filters wv-pos-pills">
        <button class="otc-day-filter wv-pos-btn active" data-pos="ALL" onclick="wvSetPos('ALL')">ALL</button>
        <button class="otc-day-filter wv-pos-btn" data-pos="QB" onclick="wvSetPos('QB')">QB</button>
        <button class="otc-day-filter wv-pos-btn" data-pos="RB" onclick="wvSetPos('RB')">RB</button>
        <button class="otc-day-filter wv-pos-btn" data-pos="WR" onclick="wvSetPos('WR')">WR</button>
        <button class="otc-day-filter wv-pos-btn" data-pos="TE" onclick="wvSetPos('TE')">TE</button>
        <button class="otc-day-filter wv-pos-btn" data-pos="K" hidden onclick="wvSetPos('K')">K</button>
        <button class="otc-day-filter wv-pos-btn" data-pos="DEF" hidden onclick="wvSetPos('DEF')">D/ST</button>
      </div>
    </div>
  </div>

  {bb_note}
  <!-- Mobile tab bar -->
  <div class="wv-tab-bar">
    <button class="wv-tab-btn active" id="wvTabWaivers" onclick="wvSetTab('waivers')">Waiver Wire</button>
    {startsit_tab_html}
  </div>

  <!-- Two-column layout -->
  <div class="wv-layout">
    <!-- Left: Waiver Wire -->
    <div class="wv-section wv-tab-active" id="wvSectionWaivers">
      <div id="wvTrendingWrap" hidden>
        <div class="wv-section-title wv-trending-title">
          <i class="fa-solid fa-fire" aria-hidden="true"></i> Trending across leagues
          <span class="wv-trending-window" style="font-weight:400;font-size:12px;color:var(--muted);">last 48h</span>
        </div>
        <div id="wvTrendingStrip" class="wv-trending-strip"></div>
      </div>
      <!-- Unexpected performances available in your league (big-game detector) -->
      <div id="wvBigGamesWrap" hidden>
        <div class="wv-section-title">Unexpected performances available in your league</div>
        <div id="wvBigGamesList" class="wv-biggames"></div>
      </div>
      <div class="wv-section-heading">
        <div class="wv-section-title" id="wvBestMovesTitle">Best moves for your team</div>
        <label class="wv-faab-toggle" id="wvFaabToggle" hidden>
          <input type="checkbox" id="wvShowFaab" onchange="wvToggleFaab(this.checked)">
          <span>FAAB</span>
        </label>
      </div>
      <div class="wv-horizon" id="wvHorizonCtl" role="tablist" aria-label="Recommendation horizon">
        <button type="button" class="wv-horizon-btn active" data-h="this_week" onclick="wvSetHorizon('this_week')">This week</button>
        <button type="button" class="wv-horizon-btn" data-h="four_week" onclick="wvSetHorizon('four_week')">Next 4 weeks</button>
        <button type="button" class="wv-horizon-btn" data-h="stash" onclick="wvSetHorizon('stash')">Long-term stash</button>
      </div>
      <div id="wvWaiverList">
        {wv_skel}
      </div>
      <div id="wvStreamWrap" hidden>
        <div class="wv-section-title wv-stream-head">Streaming this week</div>
        <div id="wvStreamDef" class="wv-stream-group"></div>
        <div id="wvStreamK" class="wv-stream-group"></div>
      </div>
    </div>

    <!-- Right: Start/Sit -->
    <div class="wv-section" id="wvSectionStartSit"{startsit_section_hidden}>
      <div class="wv-section-title">Start/Sit Advisor</div>
      <!-- Compare panel (hidden until 2 players selected) -->
      <div id="wvComparePanel" style="display:none;scroll-margin-top:16px;"></div>
      <div id="wvStartSit">
        {wv_skel}
      </div>
    </div>
  </div>
</div>
"""

    script = f"""
<script>
const WV_PLATFORM = '{platform}';
const WV_SEASON = {season};
const WV_LEAGUE_ID = '{league_id}';
let wvCurrentPos = 'ALL';
let wvWaiverData = [];
let wvHorizon = 'this_week';  // #2: default in-season to personalized immediate help
let wvShowFaab = false;  // FAAB hidden by default; the toggle opts it in
let wvFaabPreferenceSet = false;
let wvCandidateRequestSeq = 0;
let wvCandidateController = null;
let wvTrendingData = [];
let wvBigGamesData = [];
let wvStartSitData = {{}};
let wvCompare = [null, null]; // [playerA, playerB]

if (!window.__brctx) window.__brctx = {{}};
if (!window.__brctx.leagueId) window.__brctx.leagueId = WV_LEAGUE_ID;

function wvLeaguePath(suffix) {{
  return '/' + WV_PLATFORM + '/' + WV_SEASON + '/' + WV_LEAGUE_ID + suffix;
}}

function wvSetTab(tab) {{
  const isWaivers = tab === 'waivers';
  const ss = document.getElementById('wvSectionStartSit');
  const ssTab = document.getElementById('wvTabStartSit');
  document.getElementById('wvSectionWaivers').classList.toggle('wv-tab-active', isWaivers);
  if (ss) ss.classList.toggle('wv-tab-active', !isWaivers);
  document.getElementById('wvTabWaivers').classList.toggle('active', isWaivers);
  if (ssTab) ssTab.classList.toggle('active', !isWaivers);
}}

function wvSetPos(pos) {{
  wvCurrentPos = pos;
  document.querySelectorAll('.wv-pos-btn').forEach(b => {{
    const key = b.getAttribute('data-pos') || b.textContent;
    b.classList.toggle('active', key === pos);
  }});
  wvRenderWaivers();
  wvRenderStartSit();
  wvRenderTrending(wvTrendingData);
  wvRenderBigGames(wvBigGamesData);
}}

function wvToggleFaab(show) {{
  wvShowFaab = !!show;
  wvFaabPreferenceSet = true;
  wvRenderWaivers();
}}

// #2: switch recommendation horizon and re-fetch (the server re-weights the
// ranking so the list meaningfully reflects the choice).
function wvSetHorizon(h) {{
  if (h === wvHorizon) return;
  wvHorizon = h;
  document.querySelectorAll('.wv-horizon-btn').forEach(b => {{
    b.classList.toggle('active', b.getAttribute('data-h') === h);
  }});
  wvLoadCandidates();
}}

// ── Matchup chip helper ───────────────────────────────────────────────────────
function wvMuChip(rank, total) {{
  if (!rank || !total) return '';
  const pct = rank / total;
  if (pct <= 0.25) return `<span class="wv-mu wv-mu-easy">#${{rank}} easiest</span>`;
  if (pct <= 0.50) return `<span class="wv-mu wv-mu-ok">#${{rank}} favorable</span>`;
  if (pct <= 0.75) return `<span class="wv-mu wv-mu-avg">#${{rank}} tough</span>`;
  return `<span class="wv-mu wv-mu-hard">#${{rank}} hardest</span>`;
}}

function wvMuClass(rank, total) {{
  if (!rank || !total) return '';
  const pct = rank / total;
  if (pct <= 0.25) return 'wv-compare-win';
  if (pct <= 0.75) return '';
  return 'wv-compare-lose';
}}

// ── Injury badge ──────────────────────────────────────────────────────────────
function wvInjBadge(inj) {{
  if (!inj) return '';
  const u = inj.toUpperCase();
  if (['IR','OUT','PUP','SUSP'].includes(u)) return `<span class="wv-inj-out">${{u}}</span>`;
  if (['DOUBTFUL','D'].includes(u))          return `<span class="wv-inj-d">D</span>`;
  if (['QUESTIONABLE','Q','GTD'].includes(u)) return `<span class="wv-inj-q">Q</span>`;
  return `<span class="wv-inj-q">${{inj}}</span>`;
}}

// Venue / weather chip: live weather when available, else the static dome/cold
// venue tag. Rendered separately so it can sit after the matchup chip.
function wvVenueChip(p) {{
  const env = p.weather || p.game_env;
  if (!env) return '';
  const lbl = p.weather ? 'Weather' : 'Venue';
  const note = (env.note || '').replace(/"/g, '&quot;');
  return '<div class="wv-ss-stat"><span class="wv-ss-stat-lbl">' + lbl + '</span>' +
         '<span class="wv-ss-env wv-ss-env-' + env.kind + '" title="' + note + '">' +
         env.label + '</span></div>';
}}

// Vegas implied team total chip.
function wvVegasChip(p) {{
  if (p.implied_total == null) return '';
  const it = p.implied_total;
  const k = it >= 26 ? 'high' : (it <= 18 ? 'low' : 'mid');
  return '<div class="wv-ss-stat"><span class="wv-ss-stat-lbl">Vegas</span>' +
         '<span class="wv-ss-total wv-ss-total-' + k + '" title="Implied team total (Vegas)">' +
         it + ' implied</span></div>';
}}

// Weekly consistency chips: floor-ceiling range + boom/bust profile label.
function wvConsistencyChips(p) {{
  const c = p.consistency;
  if (!c || c.small_sample) return '';
  const k = c.label === 'Steady' ? 'steady'
          : c.label === 'Volatile' ? 'volatile'
          : c.label === 'Boom or bust' ? 'boombust' : 'balanced';
  const yr = c.season ? '<sup class="wv-yr" title="from the ' + c.season + ' season">’' + String(c.season).slice(-2) + '</sup>' : '';
  const tip = 'Consistency ' + c.consistency + '/100 · boom ' +
              Math.round(c.boom_rate * 100) + '% · bust ' +
              Math.round(c.bust_rate * 100) + '% (' + c.games + ' g' +
              (c.season ? ', ' + c.season : '') + ')';
  return '<div class="wv-ss-stat"><span class="wv-ss-stat-lbl">Floor–Ceil</span>' +
         '<span class="wv-ss-stat-val muted">' + c.floor + '–' + c.ceiling + yr + '</span></div>' +
         '<div class="wv-ss-stat"><span class="wv-ss-stat-lbl">Profile</span>' +
         '<span class="wv-ss-cons wv-ss-cons-' + k + '" title="' + tip + '">' + c.label + '</span></div>';
}}

// ── Opp plays faced (pace / possession context) ──────────────────────────────
// One tappable stat: the offensive plays this player's OPPONENT defense faces
// per game. The headline is the season average with a "x above/below NFL
// average" sub; tapping reveals the season/last-4 splits, sample size, and the
// opponent's own offensive possession, kept collapsed so the card stays lean.
// Deliberately neutral (no start/sit implication) -- pace is context only.
function wvPlaysRow(lbl, val) {{
  return '<div class="wv-ss-plays-detail-row">' +
         '<span class="wv-ss-plays-detail-lbl">' + lbl + '</span>' +
         '<span class="wv-ss-plays-detail-val">' + val + '</span></div>';
}}
function wvPlaysFacedChip(pv) {{
  if (!pv || pv.plays_faced_pg == null) return '';
  const faced = pv.plays_faced_pg;
  let sub = '';
  if (pv.vs_avg != null) {{
    const a = Math.abs(pv.vs_avg).toFixed(1);
    sub = pv.vs_avg > 0 ? (a + ' above NFL average')
        : (pv.vs_avg < 0 ? (a + ' below NFL average') : 'at NFL average');
  }}
  const rows = [wvPlaysRow('Season average', faced + '/gm')];
  if (pv.plays_faced_l4_pg != null) rows.push(wvPlaysRow('Last 4 games', pv.plays_faced_l4_pg + '/gm'));
  if (pv.games != null) rows.push(wvPlaysRow('Sample', pv.games + (pv.games === 1 ? ' game' : ' games')));
  if (pv.off_plays_pg != null) rows.push(wvPlaysRow('Opp off. possession', pv.off_plays_pg + ' plays/gm'));
  const detail = '<div class="wv-ss-plays-detail">' + rows.join('') + '</div>';
  const tip = 'Offensive plays this opponent&#39;s defense faces per game (pace/possession). Context only, not part of the start/sit score.';
  return '<div class="wv-ss-stat"><span class="wv-ss-stat-lbl">Opp plays faced</span>' +
         '<span class="wv-ss-stat-val wv-ss-plays-btn" role="button" tabindex="0" ' +
         'aria-expanded="false" title="' + tip + '" ' +
         'onclick="wvTogglePlays(this)" onkeydown="wvPlaysKey(event,this)">' +
         faced + '/gm<span class="wv-ss-plays-caret" aria-hidden="true">&#9662;</span></span>' +
         (sub ? '<span class="wv-ss-plays-sub">' + sub + '</span>' : '') +
         '</div>' + detail;
}}
function wvTogglePlays(el) {{
  const wrap = el.closest('.wv-ss-stats');
  if (!wrap) return;
  const d = wrap.querySelector('.wv-ss-plays-detail');
  if (!d) return;
  const open = d.classList.toggle('open');
  el.setAttribute('aria-expanded', open ? 'true' : 'false');
}}
function wvPlaysKey(e, el) {{
  if (e.key === 'Enter' || e.key === ' ') {{ e.preventDefault(); wvTogglePlays(el); }}
}}

// Compose the start/sit stats row in three groups, most-decisive first, with a
// thin divider between groups:
//   1. Output      - what to expect  (Proj PPG, L4 PPG)
//   2. Reliability - can I trust it   (Floor-Ceil, Profile, Usage)
//   3. The spot    - this week's game (Opp, Matchup, Vegas, Venue)
// 'Def vs pos' is folded into the Matchup chip's tooltip to keep the row lean.
function wvStatsRow(p) {{
  const chip = (lbl, val, cls) =>
    '<div class="wv-ss-stat"><span class="wv-ss-stat-lbl">' + lbl + '</span>' +
    '<span class="wv-ss-stat-val ' + (cls || '') + '">' + val + '</span></div>';

  // Proj PPG is pulled out of the strip and rendered as the card's hero number
  // (see the card template), so it is intentionally not pushed here.
  const g1 = [];
  if (p.recent_ppg > 0) g1.push(chip('L4 PPG', p.recent_ppg));
  if (p.value > 0)      g1.push(chip('Value', Math.round(p.value)));
  if (p.market_signal) {{
    const d = p.market_signal.delta;
    const cls = d > 0 ? 'signal-positive' : (d < 0 ? 'signal-negative' : 'muted');
    g1.push(chip('Market vs Projection', (d > 0 ? '+' : '') + d.toFixed(1), cls));
    g1.push('<span class="chip chip--sm chip--neutral" title="Market Confidence ' +
      Math.round(p.market_signal.confidence * 100) + '%">' + p.market_signal.label + '</span>');
  }}

  const g2 = [wvConsistencyChips(p)];
  if (p.usage_delta != null && Math.abs(p.usage_delta) >= 1) {{
    const upU = p.usage_delta > 0;
    const lblU = p.usage_stat === 'snap_pct' ? 'snap%' : (p.usage_stat === 'touches' ? 'touches' : 'targets');
    g2.push('<div class="wv-ss-stat"><span class="wv-ss-stat-lbl">Usage</span><span class="wv-ss-stat-val" style="color:' +
            (upU ? 'var(--win)' : 'var(--loss)') + '" title="Last-3-week ' + lblU + ' vs season avg">' +
            (upU ? '&#9650;' : '&#9660;') + ' ' + (upU ? '+' : '') + p.usage_delta + '</span></div>');
  }}

  const g3 = [];
  if (p.opponent) g3.push(chip('Opp', p.opponent, 'muted'));
  const mu = !p.on_bye ? wvMuChip(p.def_rank, p.def_total) : '';
  if (mu) {{
    const dtip = p.fpts_against > 0
      ? ' · ' + p.fpts_against + ' pts/gm allowed to ' + (p.position || 'the position')
      : '';
    g3.push('<div class="wv-ss-stat"><span class="wv-ss-stat-lbl">Matchup</span>' +
            '<span class="wv-ss-stat-val" title="Matchup rank (1 = easiest)' + dtip + '">' + mu + '</span></div>');
  }}
  if (!p.on_bye) g3.push(wvPlaysFacedChip(p.play_volume));
  g3.push(wvVegasChip(p));
  g3.push(wvVenueChip(p));

  const groups = [g1, g2, g3].map(g => g.filter(Boolean).join('')).filter(Boolean);
  return '<div class="wv-ss-stats">' +
         groups.join('<span class="wv-ss-div" aria-hidden="true"></span>') +
         '</div>';
}}

// ── Load ──────────────────────────────────────────────────────────────────────
function wvLoadCandidates() {{
  const requestSeq = ++wvCandidateRequestSeq;
  const requestedHorizon = wvHorizon;
  const requestedContext = `${{WV_PLATFORM}}:${{WV_SEASON}}:${{WV_LEAGUE_ID}}`;
  if (wvCandidateController) wvCandidateController.abort();
  wvCandidateController = typeof AbortController !== 'undefined' ? new AbortController() : null;
  const _wvRid = window._viewerRid ? ('&rid=' + encodeURIComponent(window._viewerRid)) : '';
  const _h = '&horizon=' + encodeURIComponent(requestedHorizon);
  // Without an identified roster we can't claim personalized improvement (#1).
  const bm = document.getElementById('wvBestMovesTitle');
  if (bm) bm.textContent = window._viewerRid ? 'Best moves for your team' : 'Top available players';
  window.brLoadingState('wvWaiverList', {{ rows: 4, compact: true, message: 'Loading ' + requestedHorizon.replace('_', ' ') + ' recommendations' }});
  fetch(`/api/waiver-candidates?platform=${{WV_PLATFORM}}&league_id=${{WV_LEAGUE_ID}}&season=${{WV_SEASON}}${{_wvRid}}${{_h}}`,
        wvCandidateController ? {{ signal: wvCandidateController.signal }} : {{}})
    .then(r => r.json().then(d => ({{ ok: r.ok, d }})))
    .then(({{ ok, d }}) => {{
      if (requestSeq !== wvCandidateRequestSeq || requestedHorizon !== wvHorizon || requestedContext !== `${{WV_PLATFORM}}:${{WV_SEASON}}:${{WV_LEAGUE_ID}}`) return;
      if (!ok || d.error) {{
        window.brErrorState('wvWaiverList', (d && d.error) || 'Unable to load waiver data.', wvLoadCandidates);
        return;
      }}
      wvWaiverData = d.candidates || [];
      window.wvFaabEnabled = d.faab_enabled === true;
      const toggle = document.getElementById('wvFaabToggle');
      if (toggle) toggle.hidden = !window.wvFaabEnabled;
      // FAAB leagues: show bid bands by default so managers don't have to hunt
      // for the toggle. Non-FAAB leagues keep the control hidden.
      if (window.wvFaabEnabled && !wvFaabPreferenceSet) {{
        wvShowFaab = true;
        const cb = document.getElementById('wvShowFaab');
        if (cb) cb.checked = true;
      }}
      wvRenderWaivers();
    }})
    .catch(err => {{
      if (requestSeq !== wvCandidateRequestSeq || (err && err.name === 'AbortError')) return;
      window.brErrorState('wvWaiverList', 'Unable to load waiver data.', wvLoadCandidates);
    }});
}}

function wvLoad() {{
  wvLoadCandidates();

  // Unexpected performances available in your league (#6/#7). Best-effort strip;
  // an ownership failure is visible rather than masquerading as no discoveries.
  wvLoadBigGames();

  fetch(`/api/trending-adds?platform=${{WV_PLATFORM}}&league_id=${{WV_LEAGUE_ID}}&season=${{WV_SEASON}}`)
    .then(r => r.json())
    .then(d => {{ wvTrendingData = d.trending || []; wvRenderTrending(wvTrendingData); }})
    .catch(() => {{}});

  fetch(`/api/streaming-options?platform=${{WV_PLATFORM}}&league_id=${{WV_LEAGUE_ID}}&season=${{WV_SEASON}}`)
    .then(r => r.json())
    .then(d => wvRenderStreaming(d))
    .catch(() => {{}});

  wvLoadStartSit();
}}

function wvLoadBigGames() {{
  // Abort if the backend hangs so the section falls through to the error
  // state with a retry instead of skeletons forever (matches wvLoadStartSit).
  var bgController = (typeof AbortController !== 'undefined') ? new AbortController() : null;
  var bgTimer = null;
  var bgFailsafe = null;
  var bgSettled = false;
  function bgShowError() {{
    if (bgSettled) return;
    bgSettled = true;
    if (bgTimer) clearTimeout(bgTimer);
    if (bgFailsafe) clearTimeout(bgFailsafe);
    const wrap = document.getElementById('wvBigGamesWrap'); if (wrap) wrap.hidden = false;
    window.brErrorState('wvBigGamesList', 'Availability could not be verified.', wvLoadBigGames);
  }}
  if (bgController) {{
    bgTimer = setTimeout(function() {{ try {{ bgController.abort(); }} catch (_) {{}} }}, 20000);
  }}
  // Failsafe: force the error state if skeletons persist past 25s.
  bgFailsafe = setTimeout(function() {{
    var list = document.getElementById('wvBigGamesList');
    if (list && list.querySelector('.skeleton')) bgShowError();
  }}, 25000);
  fetch(`/api/waiver-big-games?platform=${{WV_PLATFORM}}&league_id=${{WV_LEAGUE_ID}}&season=${{WV_SEASON}}`,
        bgController ? {{ signal: bgController.signal }} : undefined)
    .then(r => r.json().then(d => ({{ ok: r.ok, d }})))
    .then(({{ok, d}}) => {{
      if (bgSettled) return;
      bgSettled = true;
      if (bgTimer) clearTimeout(bgTimer);
      if (bgFailsafe) clearTimeout(bgFailsafe);
      if (!ok || d.availability === 'unavailable' || d.availability === 'stale') {{
        const wrap = document.getElementById('wvBigGamesWrap');
        if (wrap) wrap.hidden = false;
        window.brErrorState('wvBigGamesList', d.message || 'Availability could not be verified.', wvLoadBigGames, {{ title: 'Availability unavailable' }});
        return;
      }}
      wvBigGamesData = d.discoveries || []; wvRenderBigGames(wvBigGamesData);
    }})
    .catch(() => {{
      bgShowError();
    }});
}}

function wvLoadStartSit() {{
  window.brLoadingState('wvStartSit', {{ rows: 3, compact: true, message: 'Loading lineup' }});
  // Abort if the backend hangs (e.g. slow league-context fetch) so the panel
  // falls through to the error state with a retry instead of skeletons forever.
  var ssController = (typeof AbortController !== 'undefined') ? new AbortController() : null;
  var ssTimer = null;
  var ssFailsafe = null;
  var ssSettled = false;
  function ssShowError() {{
    if (ssSettled) return;
    ssSettled = true;
    if (ssTimer) clearTimeout(ssTimer);
    if (ssFailsafe) clearTimeout(ssFailsafe);
    window.brErrorState('wvStartSit', 'Unable to load lineup data.', wvLoadStartSit);
  }}
  if (ssController) {{
    ssTimer = setTimeout(function() {{ try {{ ssController.abort(); }} catch (_) {{}} }}, 20000);
  }}
  // Failsafe: if skeletons are still showing after 25s (abort didn't reject the
  // fetch, r.json() hung, etc.), force the error state with a retry.
  ssFailsafe = setTimeout(function() {{
    var el = document.getElementById('wvStartSit');
    if (el && el.querySelector('.skeleton')) ssShowError();
  }}, 25000);
  fetch(`/api/start-sit-options?platform=${{WV_PLATFORM}}&league_id=${{WV_LEAGUE_ID}}&season=${{WV_SEASON}}`,
        ssController ? {{ signal: ssController.signal }} : undefined)
    .then(r => r.json().then(d => ({{ ok: r.ok, d }})))
    .then(({{ok, d}}) => {{
      if (ssSettled) return;
      ssSettled = true;
      if (ssTimer) clearTimeout(ssTimer);
      if (ssFailsafe) clearTimeout(ssFailsafe);
      const state = d.state || (ok ? 'loaded' : 'temporarily_unavailable');
      if (state === 'sign_in_required') {{
        showLoginGate('wvStartSit', {{
          title: 'Sign in to see your lineup',
          description: 'Sign in to get personalized start/sit recommendations for your roster.'
        }});
        return;
      }}
      if (state === 'team_not_linked') {{
        window.brEmptyState('wvStartSit', {{ icon: 'users', title: 'Select your team', message: d.message,
          cta: '<a class="empty-state-cta" href="' + wvLeaguePath('/teams') + '">Select or link team</a>' }});
        return;
      }}
      if (!ok || state === 'temporarily_unavailable') {{
        window.brErrorState('wvStartSit', d.message || 'Unable to load lineup data.', wvLoadStartSit);
        return;
      }}
      if (state === 'empty_roster') {{
        window.brEmptyState('wvStartSit', {{ icon: 'users', title: 'No eligible players', message: 'This roster has no active players in supported lineup positions.' }});
        return;
      }}
      wvStartSitData = d;
      wvStartSitData._lineup_requirements = d.lineup_requirements || {{}};
      wvSyncPosPills();
      wvRenderStartSit();
    }})
    .catch(() => {{
      ssShowError();
    }});
}}

// ── Unexpected performances available (big-game detector) ──────────────────────
const WV_BG_CATEGORY = {{
  priority: {{ label: 'Priority pickup', cls: 'chip--accent' }},
  speculative: {{ label: 'Speculative add', cls: 'chip--neutral' }},
  watchlist: {{ label: 'Watchlist', cls: 'chip--muted' }},
}};
const WV_BG_CAUTION = {{
  td_dependent: 'leaned on TDs',
  one_big_play: 'one long play',
  hot_efficiency: 'unsustainable efficiency',
  role_unconfirmed: 'usage not confirmed',
  uncertain_baseline: 'thin history',
}};

// ── Unexpected performances (big-game detector) ───────────────────────────────
// Compact rows: category badge + player + what-changed + week points. Factors,
// caution, and ROS projection expand on tap.
function wvRenderBigGames(items) {{
  const wrap = document.getElementById('wvBigGamesWrap');
  const list = document.getElementById('wvBigGamesList');
  if (!wrap || !list) return;
  let rows = items || [];
  if (wvCurrentPos !== 'ALL') rows = rows.filter(d => d.position === wvCurrentPos);
  if (!rows.length) {{ wrap.hidden = true; list.innerHTML = ''; return; }}
  wrap.hidden = false;
  const CAT = {{
    priority: ['PRIORITY', 'wv-cx-priority'],
    speculative: ['SPECULATIVE', 'wv-cx-spec'],
    watchlist: ['WATCHLIST', 'wv-cx-watch'],
  }};
  list.innerHTML = '<div class="wv-cx-card">' + rows.slice(0, 8).map(d => {{
    const meta = CAT[d.category] || [(d.category || '').toUpperCase() || 'WATCHLIST', 'wv-cx-watch'];
    const badge = `<span class="wv-cx-badge ${{meta[1]}}">${{meta[0]}}</span>`;
    const live = d.status === 'in_progress'
      ? '<span class="wv-cx-chip warn">Game in progress</span>' : '';
    // Evidence-based copy generated from the actual factors (no generic filler).
    const facts = (d.factors || []).slice(0, 2).join(' · ');
    const cautions = (d.cautions || []).map(c => WV_BG_CAUTION[c]).filter(Boolean);
    const sub = [d.team, d.position].filter(Boolean).join(' · ');
    const wkPts = d.actual_points != null ? d.actual_points
      : (d.week_points != null ? d.week_points : null);
    const projBlock = wkPts != null
      ? `<span class="wv-cx-proj"><span class="wv-cx-proj-num">${{wkPts}}</span><span class="wv-cx-proj-lbl">WK PTS</span></span>`
      : '';
    const escName = (d.name || '').replace(/'/g, "\\\\'");
    const pid = d.player_id;

    const ev = [];
    if (facts) ev.push(`<div class="wv-cx-evl"><span class="k">FACTORS</span>${{facts}}</div>`);
    if (cautions.length) ev.push(`<div class="wv-cx-evl caution"><span class="k">CAUTION</span>${{cautions.join(' · ')}}</div>`);
    if (d.ros_ppg != null) ev.push(`<div class="wv-cx-evl"><span class="k">ROS PROJ</span>${{d.ros_ppg}} pts/gm</div>`);

    return `
    <div class="wv-cx-row-wrap">
      <button type="button" class="wv-cx-row" aria-expanded="false"
          onclick="wvToggleSsRow(this)">
        ${{badge}}
        <span class="wv-cx-main">
          <span class="wv-cx-name" data-wl-star-pid="${{pid}}">${{d.name || ('Player ' + pid)}}</span>
          ${{sub ? `<span class="wv-cx-sub">${{sub}}</span>` : ''}}
          ${{facts ? `<span class="wv-cx-why"><span class="wv-cx-k">WHAT CHANGED</span>${{facts}}</span>` : ''}}
          ${{live}}
        </span>
        ${{projBlock}}
        <span class="wv-cx-chev" aria-hidden="true">›</span>
      </button>
      <div class="wv-cx-detail">
        ${{ev.length ? `<div class="wv-cx-evidence-list">${{ev.join('')}}</div>` : ''}}
        <div class="wv-cx-actions" onclick="event.stopPropagation()">
          <button type="button" onclick="event.stopPropagation();openPlayerModal('${{pid}}', '${{escName}}')">Open player</button>
          <button type="button" onclick="event.stopPropagation();wvToggleCompare(${{JSON.stringify({{player_id: pid, name: d.name || ('Player ' + pid), position: d.position, team: d.team}}).replace(/"/g, '&quot;')}})">+ Compare</button>
        </div>
      </div>
    </div>`;
  }}).join('') + '</div>';
  if (window._wlStarDecorate) window._wlStarDecorate(list);
}}


// ── Waiver list ───────────────────────────────────────────────────────────────
// ── Best moves (waiver list) ──────────────────────────────────────────────────
// Compact rows: outcome badge + player + lineup gain + proj/value. Drop,
// schedule, usage, and FAAB evidence expand on tap. The group verdict names
// the single best move before any row is read.
function wvBmGroupVerdict(players) {{
  if (!window._viewerRid || !players.length) return '';
  const top = players[0];
  const gain = Number(top.lineup_gain) || 0;
  if (!(gain > 0)) return '';
  // Signed gain, never "+-5.0".
  const gainTxt = '+' + gain.toFixed(1);
  let txt = `Top move: add <b>${{top.name}}</b> (${{gainTxt}} pts).`;
  if (top.drop && top.drop.name) txt += ` Drop <b>${{top.drop.name}}</b>.`;
  return `<div class="wv-cx-verdict">${{txt}}</div>`;
}}

function wvBmClaimLine(p) {{
  let claimBid = '', claimTip = '', claimHi = false;
  if (p.faab_mode === 'waiver_priority' && p.faab_claim_guidance) {{
    claimBid = p.faab_claim_guidance;
    claimTip = p.faab_rationale || '';
    claimHi = /high/i.test(p.faab_claim_guidance);
  }} else if (window.wvFaabEnabled && wvShowFaab && (p.faab_high != null || p.faab_target != null || p.faab_dollars_target != null)) {{
    const hasDollars = p.faab_dollars_target != null;
    if (hasDollars) {{
      claimBid = 'FAAB bid $' + p.faab_dollars_low + ' · $' + p.faab_dollars_target + ' · $' + p.faab_dollars_high;
      claimTip = 'Suggested FAAB bid (low · target · stretch), capped at your remaining budget. ';
    }} else {{
      const parts = [p.faab_low, p.faab_target, p.faab_high].filter(v => v != null);
      claimBid = 'FAAB bid ' + parts.join(' · ') + '%';
      const denom = p.faab_pct_denominator === 'remaining_budget' ? 'remaining budget' : 'season budget';
      claimTip = 'Suggested FAAB as % of your ' + denom + ' (low · target · stretch). ';
    }}
    claimTip += (p.faab_rationale || '') + (p.faab_heuristic ? ' (heuristic estimate)' : '');
  }}
  if (!claimBid) return '';
  return `<div class="wv-cx-evl${{claimHi ? ' caution' : ''}}"><span class="k">CLAIM</span><span title="${{String(claimTip).replace(/"/g, '&quot;')}}">${{claimBid}}</span></div>`;
}}

function wvRenderWaivers() {{
  const list = document.getElementById('wvWaiverList');
  let players = wvWaiverData;
  if (wvCurrentPos !== 'ALL') players = players.filter(p => p.position === wvCurrentPos);
  if (!players.length) {{ window.brEmptyState('wvWaiverList', {{ icon: 'search', title: 'No waiver targets', message: 'Nothing to add at this position right now.', compact: true }}); return; }}
  const OUTCOME = {{
    add: ['ADD', 'wv-cx-add'],
    add_drop: ['ADD & DROP', 'wv-cx-adddrop'],
    stash: ['STASH', 'wv-cx-stash'],
  }};
  const verdict = wvBmGroupVerdict(players);
  list.innerHTML = (verdict ? `<div class="wv-cx-group"><div class="wv-cx-group-head">${{verdict}}</div></div>` : '') +
    '<div class="wv-cx-card">' + players.slice(0, 20).map(p => {{
    const oc = OUTCOME[p.outcome] || ['ADD', 'wv-cx-add'];
    const badge = `<span class="wv-cx-badge ${{oc[1]}}">${{oc[0]}}</span>`;

    // One-line summary: the lineup gain is the answer to "why add?".
    let gainTxt = '';
    let gainLbl = 'LINEUP';
    const gain = Number(p.lineup_gain) || 0;
    if (gain > 0) {{
      gainTxt = '+' + gain.toFixed(1) + ' pts this week';
      if (p.replaces && p.replaces.name) gainTxt += ' · Starts over ' + p.replaces.name;
    }} else if (p.signal) {{
      gainTxt = p.signal;
      gainLbl = 'Why add';
    }}

    const sub = [p.position, p.team, p.pos_rank_label,
      p.rostered_pct != null ? Math.round(p.rostered_pct) + '% rostered' : '',
      p.adds_48h ? ('+' + wvFmtAdds(p.adds_48h) + ' adds') : ''].filter(Boolean).join(' · ');

    let usageChip = '';
    if (p.usage_delta != null && p.usage_delta >= 1) {{
      const statLbl = p.usage_stat === 'snap_pct' ? 'snap%' : (p.usage_stat === 'touches' ? 'touches' : 'targets');
      usageChip = `<span class="wv-cx-chip">&#9650; +${{p.usage_delta}} ${{statLbl}}</span>`;
    }}
    // Big-game flag from last week rides along as a chip, not a row.
    let bgChip = '';
    if (p.big_game && p.big_game.category) {{
      const BG = {{ priority: 'Priority big game', speculative: 'Speculative big game', watchlist: 'Watchlist big game' }};
      const facts = (p.big_game.factors || []).join(' · ');
      bgChip = `<span class="wv-cx-chip" title="Unexpected performance last week${{facts ? ': ' + facts : ''}}">${{BG[p.big_game.category] || 'Big game'}}</span>`;
    }}

    const projBlock = `<span class="wv-cx-proj">` +
      `<span class="wv-cx-proj-num">${{p.ros_ppg != null ? p.ros_ppg : '–'}}</span>` +
      `<span class="wv-cx-proj-lbl">PROJ</span>` +
      (p.value > 0 ? `<span class="wv-cx-proj-sub">value ${{Math.round(p.value)}}</span>` : '') +
      `</span>`;

    // Evidence behind the tap: drop, schedule, usage, claim.
    const ev = [];
    if (p.drop && p.drop.name) {{
      ev.push(`<div class="wv-cx-evl"><span class="k">DROP</span>${{p.drop.name}}${{p.drop.position ? ' (' + p.drop.position + ')' : ''}} · Weakest spare below this target's value</div>`);
    }}
    if (p.schedule_urgency) {{
      ev.push(`<div class="wv-cx-evl"><span class="k">SCHEDULE</span>${{p.schedule_urgency}}</div>`);
    }}
    if (p.usage_delta != null && Math.abs(p.usage_delta) >= 1) {{
      const statLbl = p.usage_stat === 'snap_pct' ? 'snap%' : (p.usage_stat === 'touches' ? 'touches' : 'targets');
      const up = p.usage_delta > 0;
      ev.push(`<div class="wv-cx-evl"><span class="k">USAGE</span><span class="wv-cx-chip${{up ? '' : ' bad'}}">${{up ? '▲' : '▼'}} ${{up ? '+' : ''}}${{p.usage_delta}} ${{statLbl}}</span> last-3-week avg vs season avg</div>`);
    }}
    const vac = (p.vacated || []).filter(v => v && (v.weeks_out != null || v.return_source));
    if (vac.length) {{
      const espn = vac.some(v => v.return_source === 'espn');
      const wks = Math.max.apply(null, vac.map(v => Number(v.weeks_out) || 0));
      ev.push(`<div class="wv-cx-evl"><span class="k">RETURN</span>${{espn ? 'ESPN return' : 'Status estimate'}}${{wks > 0 ? ': ~' + wks + ' wk' + (wks === 1 ? '' : 's') + ' out' : ''}}</div>`);
    }}
    const claimLine = wvBmClaimLine(p);
    if (claimLine) ev.push(claimLine);
    if (p.market_opportunity) {{
      ev.push(`<div class="wv-cx-evl"><span class="k">MARKET</span>${{p.market_opportunity.label}} · Market projection ${{p.market_projection}}</div>`);
    }}

    const escName = (p.name || '').replace(/'/g, "\\\\'");
    const cmpObj = JSON.stringify({{player_id: p.player_id, name: p.name, position: p.position, team: p.team}}).replace(/"/g, '&quot;');

    return `
    <div class="wv-cx-row-wrap">
      <button type="button" class="wv-cx-row" aria-expanded="false"
          onclick="wvToggleSsRow(this)">
        ${{badge}}
        <span class="wv-cx-main">
          <span class="wv-cx-name" data-wl-star-pid="${{p.player_id}}">${{p.name}}</span>
          ${{sub ? `<span class="wv-cx-sub">${{sub}}${{usageChip}}</span>` : ''}}
          ${{gainTxt ? `<span class="wv-cx-why"><span class="wv-cx-k">${{gainLbl}}</span>${{gainTxt}}</span>` : ''}}
          ${{bgChip}}
        </span>
        ${{projBlock}}
        <span class="wv-cx-chev" aria-hidden="true">›</span>
      </button>
      <div class="wv-cx-detail">
        ${{ev.length ? `<div class="wv-cx-evidence-list">${{ev.join('')}}</div>` : ''}}
        <div class="wv-cx-actions" onclick="event.stopPropagation()">
          <button type="button" onclick="event.stopPropagation();openPlayerModal('${{p.player_id}}', '${{escName}}')">Open player</button>
          <button type="button" onclick="event.stopPropagation();wvToggleCompare(${{cmpObj}})">+ Compare</button>
          <a class="wv-cx-link" href="${{wvLeaguePath('/compare')}}?p1=${{encodeURIComponent(p.player_id)}}">Compare to roster</a>
        </div>
      </div>
    </div>`;
  }}).join('') + '</div>';
  if (window._wlStarDecorate) window._wlStarDecorate(list);
}}


// ── Trending across leagues (Sleeper league-wide add counts) ───────────────────
function wvFmtAdds(n) {{
  n = +n || 0;
  return n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1).replace(/\\.0$/, '') + 'k' : String(n);
}}
function wvRenderTrending(items) {{
  const wrap = document.getElementById('wvTrendingWrap');
  const strip = document.getElementById('wvTrendingStrip');
  items = items || [];
  // Mirror the active position filter so the strip only shows players the rest
  // of the page is scoped to (ALL shows everything).
  const shown = (wvCurrentPos && wvCurrentPos !== 'ALL')
    ? items.filter(p => (p.position || '').toUpperCase() === wvCurrentPos)
    : items;
  if (!wrap || !strip || !shown.length) {{ if (wrap) wrap.hidden = true; return; }}
  // Cap the strip so a long league-wide add list can't dominate the page; the
  // strip scrolls horizontally, so a small set keeps it a glanceable accent.
  strip.innerHTML = shown.slice(0, 8).map(p => {{
    const pos = p.position || '';
    const sub = [pos, p.team].filter(Boolean).join(' · ');
    const nm = (p.name || '').replace(/'/g, "\\\\'");
    const tipName = (p.name || '').replace(/"/g, '&quot;');
    const tipAdds = wvFmtAdds(p.adds) + ' adds across Sleeper leagues in the last 48h';
    return `
      <button type="button" class="wv-trend-chip" onclick="openPlayerModal('${{p.player_id}}', '${{nm}}')"
              title="${{tipName}} · ${{tipAdds}}">
        <span class="wv-trend-adds"><i class="fa-solid fa-fire" aria-hidden="true"></i> ${{wvFmtAdds(p.adds)}}</span>
        <span class="wv-trend-name">${{p.name || ''}}</span>
        <span class="wv-trend-sub">${{sub}}</span>
      </button>`;
  }}).join('');
  wrap.hidden = false;
}}

// ── Streaming this week (matchup-based D/ST + K) ───────────────────────────────
function wvStreamRow(p, implied, isDef) {{
  const nm = (p.name || '').replace(/'/g, "\\\\'");
  const impNum = implied != null ? +implied : null;
  let impCls = 'mid';
  if (impNum != null) impCls = impNum >= 26 ? 'high' : (impNum <= 18 ? 'low' : 'mid');
  // For a defense a LOW opponent total is good, so invert the color meaning.
  const good = isDef ? (impNum != null && impNum <= 18) : (impNum != null && impNum >= 26);
  const bad  = isDef ? (impNum != null && impNum >= 26) : (impNum != null && impNum <= 18);
  const impCls2 = good ? 'good' : (bad ? 'bad' : 'mid');
  const impLabel = impNum != null
    ? (isDef ? (impNum.toFixed(0) + ' opp') : (impNum.toFixed(0) + ' impl'))
    : '';
  const impChip = impLabel
    ? `<span class="wv-stream-imp wv-stream-imp-${{impCls2}}" title="Vegas implied team total">${{impLabel}}</span>`
    : '';
  return `
    <div class="wv-stream-row" onclick="openPlayerModal('${{p.player_id}}', '${{nm}}')">
      <span class="wv-stream-name">${{p.name || ''}}</span>
      <span class="wv-stream-matchup">${{p.matchup || ''}}</span>
      ${{impChip}}
      ${{p.adds_48h ? '<span class="wv-stream-imp wv-stream-imp-mid" title="Sleeper adds last 48h">+' + wvFmtAdds(p.adds_48h) + ' adds</span>' : ''}}
    </div>`;
}}
function wvRenderStreaming(d) {{
  const wrap = document.getElementById('wvStreamWrap');
  const defEl = document.getElementById('wvStreamDef');
  const kEl = document.getElementById('wvStreamK');
  if (!wrap || !d || !d.in_season) {{ if (wrap) wrap.hidden = true; return; }}
  const defs = d.defense || [], ks = d.kicker || [];
  if (!defs.length && !ks.length) {{ wrap.hidden = true; return; }}
  defEl.innerHTML = defs.length
    ? '<div class="wv-stream-sub">Defense</div>' + defs.slice(0, 5).map(p => wvStreamRow(p, p.opp_implied, true)).join('')
    : '';
  kEl.innerHTML = ks.length
    ? '<div class="wv-stream-sub">Kicker</div>' + ks.slice(0, 5).map(p => wvStreamRow(p, p.own_implied, false)).join('')
    : '';
  wrap.hidden = false;
}}

// ── Compare slot management ───────────────────────────────────────────────────
function wvToggleCompare(p) {{
  const id = p.player_id;
  const inSlot0 = wvCompare[0] && wvCompare[0].player_id === id;
  const inSlot1 = wvCompare[1] && wvCompare[1].player_id === id;

  if (inSlot0) {{ wvCompare[0] = null; }}
  else if (inSlot1) {{ wvCompare[1] = null; }}
  else if (!wvCompare[0]) {{ wvCompare[0] = p; }}
  else if (!wvCompare[1]) {{ wvCompare[1] = p; }}
  else {{ wvCompare[0] = wvCompare[1]; wvCompare[1] = p; }}

  wvRenderCompare();
  wvRenderStartSit();

  // Once both players are picked the panel appears (often above the fold if you
  // scrolled down to pick) - bring it into view.
  if (wvCompare[0] && wvCompare[1]) {{
    const panel = document.getElementById('wvComparePanel');
    if (panel) panel.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
  }}
}}

function wvIsSelected(id) {{
  return (wvCompare[0] && wvCompare[0].player_id === id) ||
         (wvCompare[1] && wvCompare[1].player_id === id);
}}

// Derive every comparable field for one player (values + a sortable number
// where a winner makes sense).
function wvCmpDerive(p) {{
  const c = p.consistency, cOk = c && !c.small_sample;
  const yr = cOk && c.season ? '<sup class="wv-yr" title="from the ' + c.season + ' season">’' + String(c.season).slice(-2) + '</sup>' : '';
  const pk = !cOk ? '' : (c.label === 'Steady' ? 'steady'
    : c.label === 'Volatile' ? 'volatile'
    : c.label === 'Boom or bust' ? 'boombust' : 'balanced');
  const env = p.weather || p.game_env;
  return {{
    proj:     p.proj_pts > 0 ? p.proj_pts : null,
    l4:       p.recent_ppg > 0 ? p.recent_ppg : (p.season_ppg > 0 ? p.season_ppg : null),
    floorNum: cOk ? c.floor : null,
    flCeil:   cOk ? (c.floor + '–' + c.ceiling + yr) : '–',
    profile:  cOk ? `<span class="wv-ss-cons wv-ss-cons-${{pk}}">${{c.label}}</span>` : '–',
    boomBust: cOk ? (Math.round(c.boom_rate * 100) + '% / ' + Math.round(c.bust_rate * 100) + '%') : '–',
    opp:      p.opponent || (p.on_bye ? 'BYE' : '–'),
    def:      p.fpts_against > 0 ? `${{p.fpts_against}} pts` : (p.on_bye ? 'BYE' : '–'),
    defCls:   wvMuClass(p.def_rank, p.def_total),
    mu:       (!p.on_bye ? wvMuChip(p.def_rank, p.def_total) : '') || (p.on_bye ? '–' : 'No data'),
    playsFaced: (p.play_volume && p.play_volume.plays_faced_pg != null) ? p.play_volume.plays_faced_pg : null,
    playsVsAvg: (p.play_volume && p.play_volume.vs_avg != null) ? p.play_volume.vs_avg : null,
    playsL4:    (p.play_volume && p.play_volume.plays_faced_l4_pg != null) ? p.play_volume.plays_faced_l4_pg : null,
    vegasNum: p.implied_total != null ? p.implied_total : null,
    vegas:    p.implied_total != null ? (p.implied_total + ' implied') : '–',
    venue:    env ? `<span class="wv-ss-env wv-ss-env-${{env.kind}}">${{env.label}}</span>` : (p.on_bye ? 'BYE' : '–'),
    value:    p.value > 0 ? Math.round(p.value) : null,
    rank:     p.pos_rank_label || '–',
  }};
}}

// Signed "vs NFL average" for the neutral play-volume compare row (e.g. -6.1 /
// +1.7). Null renders as a dash so the row drops when neither side has data.
function wvFmtVsAvg(v) {{
  if (v == null) return '–';
  return (v > 0 ? '+' : '') + v.toFixed(1);
}}

// Winner classes for a pair of numeric values (higher is better by default).
function wvWinPair(av, bv, higher) {{
  if (av == null || bv == null || av === bv) return ['', ''];
  const aBetter = (higher === false) ? av < bv : av > bv;
  return aBetter ? ['wv-compare-win', 'wv-compare-lose'] : ['wv-compare-lose', 'wv-compare-win'];
}}

// Position-tinted rank chip (QB38, WR12…) for the compare header.
const WV_POS_COL = {{ QB: '#3b82f6', RB: '#22c55e', WR: '#f59e0b', TE: '#8b5cf6', K: '#14b8a6', DEF: '#64748b' }};
function wvPosChip(p) {{
  const lbl = p.pos_rank_label || p.position || '';
  if (!lbl) return '';
  const c = WV_POS_COL[p.position] || 'var(--text-muted)';
  return '<span class="wv-cmp-poschip" style="color:' + c + ';background:color-mix(in srgb,' + c + ' 15%,transparent);">' + lbl + '</span>';
}}

// Human-readable reasons the winner (side wi) beat the loser, drawn from the
// same factor breakdown the score is built from, so the "why" always agrees
// with the pick. Sorted by each factor's actual impact on the score.
// Matchup is intentionally omitted: weekly projections already include the
// opponent, so it is not part of the score product.
function wvVerdictReasons(a, b, wi) {{
  const fw = (wi === 0 ? a : b).score_factors || {{}};
  const fl = (wi === 0 ? b : a).score_factors || {{}};
  const cand = [];
  const projW = fw.proj || 0, projL = fl.proj || 0;
  if (projW > projL) {{
    cand.push({{ imp: projL > 0 ? projW / projL : 2, txt: 'higher projection (+' + (projW - projL).toFixed(1) + ')' }});
  }}
  const mult = [
    ['floor',   'a safer floor'],
    ['form',    'better recent form'],
    ['usage',   'a rising role'],
    ['vegas',   'a higher team total'],
    ['weather', 'a cleaner forecast'],
    ['avail',   'fewer injury concerns'],
  ];
  mult.forEach(f => {{
    const mw = fw[f[0]] != null ? fw[f[0]] : 1, ml = fl[f[0]] != null ? fl[f[0]] : 1;
    if (mw > ml + 1e-9) cand.push({{ imp: ml > 0 ? mw / ml : 2, txt: f[1] }});
  }});
  cand.sort((x, y) => y.imp - x.imp);
  return cand.slice(0, 2).map(c => c.txt);
}}

// The advisor's call: the single unified start_score decides it -- the exact same
// score behind the START badges and the optimal-lineup banner, so the three can
// never contradict each other. Returns the winning side plus the one or two
// signals that most decided it, or a toss-up when the scores are level.
function wvVerdict(a, b) {{
  const sa = a.start_score, sb = b.start_score;
  if (sa == null || sb == null) return null;
  if (Math.abs(sa - sb) < 0.1) return {{ idx: null }};
  const wi = sa > sb ? 0 : 1;
  return {{ idx: wi, reasons: wvVerdictReasons(a, b, wi) }};
}}

// ── Verdict-first compare ─────────────────────────────────────────────────────
// The verdict leads, then the deciding factors as visual bars, then the full
// comparison table behind one tap. The winner is the unified start_score; the
// reasons come from its score_factors breakdown.
function wvCmpBar(label, aVal, bVal) {{
  if (aVal == null || bVal == null) return '';
  const total = Number(aVal) + Number(bVal);
  const aPct = total > 0 ? Math.max(4, Math.min(96, Number(aVal) / total * 100)) : 50;
  const bPct = 100 - aPct;
  const fmt = v => (Math.round(v * 10) / 10);
  return `<div class="wv-cmp2-factor"><div class="fl">${{label}}</div>` +
    `<div class="wv-cmp2-bar-row"><span class="vn">${{fmt(aVal)}}</span>` +
    `<span class="wv-cmp2-track"><i class="a" style="width:${{aPct.toFixed(1)}}%"></i><i class="b" style="width:${{bPct.toFixed(1)}}%"></i></span>` +
    `<span class="vn r">${{fmt(bVal)}}</span></div></div>`;
}}

// Numeric pair behind a verdict reason, for the factor bars. Reasons without
// a clean numeric pair (weather, availability) return null and are skipped.
function wvCmpReasonBar(reasonTxt, da, db) {{
  const t = reasonTxt || '';
  if (t.indexOf('higher projection') === 0) return ['PROJECTION', da.proj, db.proj];
  if (t === 'a safer floor') return ['FLOOR', da.floorNum, db.floorNum];
  if (t === 'better recent form') return ['RECENT FORM (L4 PPG)', da.l4, db.l4];
  if (t === 'a higher team total') return ['VEGAS TOTAL', da.vegasNum, db.vegasNum];
  return null;
}}

function wvToggleCmpFull(btn) {{
  const full = document.getElementById('wvCmpFull');
  if (!full) return;
  const open = full.style.maxHeight && full.style.maxHeight !== '0px';
  full.style.maxHeight = open ? '0px' : full.scrollHeight + 'px';
  btn.setAttribute('aria-expanded', open ? 'false' : 'true');
  btn.textContent = open ? btn.dataset.label : 'Hide full comparison';
}}

function wvRenderCompare() {{
  const panel = document.getElementById('wvComparePanel');
  const a = wvCompare[0], b = wvCompare[1];
  if (!a || !b) {{ panel.style.display = 'none'; return; }}
  panel.style.display = 'block';

  const da = wvCmpDerive(a), db = wvCmpDerive(b);
  // True when a value carries no data, so a row where both sides are empty can
  // be dropped instead of printing a lonely pair of dashes.
  const isEmpty = (v) => {{
    if (v == null) return true;
    const t = String(v).replace(/<[^>]*>/g, '').trim();
    return t === '' || t === '–' || t === '-';
  }};
  // Shared single label per metric: value(a) · LABEL · value(b). Winning side's
  // value is pilled via .wv-cmp-v; rows empty on both sides are skipped.
  function row(label, aHtml, bHtml, aCls, bCls) {{
    if (isEmpty(aHtml) && isEmpty(bHtml)) return '';
    return '<div class="wv-cmp-row">' +
      '<span class="wv-cmp-a ' + (aCls || '') + '"><span class="wv-cmp-v">' + aHtml + '</span></span>' +
      '<span class="wv-cmp-lbl">' + label + '</span>' +
      '<span class="wv-cmp-b ' + (bCls || '') + '"><span class="wv-cmp-v">' + bHtml + '</span></span>' +
      '</div>';
  }}
  const dash = (v) => (v == null ? '–' : v);
  const wProj = wvWinPair(da.proj, db.proj, true);
  const wL4   = wvWinPair(da.l4, db.l4, true);
  const wVal  = wvWinPair(da.value, db.value, true);
  const wFl   = wvWinPair(da.floorNum, db.floorNum, true);
  const wVeg  = wvWinPair(da.vegasNum, db.vegasNum, true);
  // Position-relative 0-100 Start/Sit index (same one the player modal / Compare
  // page show), so a QB and a WR are comparable here too. Null when the week's
  // position pool was unavailable to build the anchor.
  const idxA = (a.start_score_pct == null ? null : Math.round(a.start_score_pct));
  const idxB = (b.start_score_pct == null ? null : Math.round(b.start_score_pct));
  const wIdx = wvWinPair(idxA, idxB, true);

  const sub = (p) => [p.team, p.position, p.opponent || (p.on_bye ? 'BYE' : '')].filter(Boolean).join(' · ');

  // Recommendation banner up top -- the reason people opened the advisor.
  // The single unified start_score decides it.
  const v = wvVerdict(a, b);
  let verdictHtml = '';
  let factorBars = '';
  if (v && v.idx != null) {{
    const w = v.idx === 0 ? a : b;
    const wName = wvLastName(w.name).toUpperCase();
    let why = v.reasons.join(' and ');
    why = why.charAt(0).toUpperCase() + why.slice(1);
    verdictHtml =
      '<div class="wv-cmp2-verdict">' +
        '<span class="wv-cmp2-pill">START</span>' +
        '<span><span class="wv-cmp2-name">' + w.name + '</span>' +
        '<span class="wv-cmp2-why">' + why + '.</span></span>' +
      '</div>';
    // Deciding factors as bars: the top reasons with numeric pairs, led by
    // the unified Start/Sit index itself.
    const bars = [wvCmpBar('START/SIT INDEX', idxA, idxB)];
    (v.reasons || []).slice(0, 2).forEach(r => {{
      const rb = wvCmpReasonBar(r, da, db);
      if (rb) bars.push(wvCmpBar('WHY ' + wName + ' WINS · ' + rb[0], rb[1], rb[2]));
    }});
    const barsHtml = bars.filter(Boolean).join('');
    if (barsHtml) factorBars = `<div class="wv-cmp2-factors">${{barsHtml}}</div>`;
  }} else if (v) {{
    verdictHtml =
      '<div class="wv-cmp2-verdict toss">' +
        '<span class="wv-cmp2-pill">TOSS-UP</span>' +
        '<span><span class="wv-cmp2-why">Nearly identical outlook - go with your gut.</span></span>' +
      '</div>';
    const idxBar = wvCmpBar('START/SIT INDEX', idxA, idxB);
    if (idxBar) factorBars = `<div class="wv-cmp2-factors">${{idxBar}}</div>`;
  }}

  const headCard = (p, dp, isWinner) => `
    <div class="wv-cmp2-head${{isWinner ? ' winner' : ''}}">
      <div class="nm">${{p.name}}</div>
      <div class="mt">${{sub(p)}}</div>
      <div class="pj">${{dp.proj != null ? dp.proj : '–'}}</div>
      <div class="pl">PROJ</div>
    </div>`;
  const winnerIdx = (v && v.idx != null) ? v.idx : -1;

  const fullRows =
    row('Start/Sit index', dash(idxA), dash(idxB), wIdx[0], wIdx[1]) +
    row('Proj PPG', dash(da.proj), dash(db.proj), wProj[0], wProj[1]) +
    row('L4 PPG', dash(da.l4), dash(db.l4), wL4[0], wL4[1]) +
    row('Value', dash(da.value), dash(db.value), wVal[0], wVal[1]) +
    row('Floor–Ceil', da.flCeil, db.flCeil, wFl[0], wFl[1]) +
    row('Profile', da.profile, db.profile) +
    row('Boom / Bust', da.boomBust, db.boomBust) +
    row('Opponent', da.opp, db.opp) +
    row('Def vs pos', da.def, db.def, da.defCls, db.defCls) +
    row('Matchup', da.mu, db.mu) +
    row('Opp plays faced/game', dash(da.playsFaced), dash(db.playsFaced)) +
    row('vs. NFL average', wvFmtVsAvg(da.playsVsAvg), wvFmtVsAvg(db.playsVsAvg)) +
    row('Last 4 games', dash(da.playsL4), dash(db.playsL4)) +
    row('Vegas total', da.vegas, db.vegas, wVeg[0], wVeg[1]) +
    row('Venue', da.venue, db.venue);
  const fullCount = (fullRows.match(/wv-cmp-row/g) || []).length;

  panel.innerHTML = `
    <div class="wv-compare-panel">
      <div class="wv-compare-header">
        <span>Compare</span>
        <button type="button" onclick="wvClearCompare()" style="font-size:11px;font-weight:700;padding:2px 8px;border-radius:6px;border:1px solid var(--border);background:transparent;color:var(--text-muted);cursor:pointer;">Clear</button>
      </div>
      ${{verdictHtml}}
      <div class="wv-cmp2-heads">
        ${{headCard(a, da, winnerIdx === 0)}}
        <div class="wv-cmp2-vs">VS</div>
        ${{headCard(b, db, winnerIdx === 1)}}
      </div>
      ${{factorBars}}
      <button type="button" class="wv-cmp2-more" aria-expanded="false" data-label="Full comparison · ${{fullCount}} stats" onclick="wvToggleCmpFull(this)">Full comparison · ${{fullCount}} stats</button>
      <div class="wv-cmp2-full" id="wvCmpFull">
        <div class="wv-cmp">${{fullRows}}</div>
      </div>
    </div>`;
}}


function wvClearCompare() {{
  wvCompare = [null, null];
  wvRenderCompare();
  wvRenderStartSit();
}}

function wvSyncPosPills() {{
  const req = wvStartSitData._lineup_requirements || {{}};
  const pos = wvStartSitData.positions || {{}};
  const showK = (req.K > 0) || ((pos.K || []).length > 0);
  const showDef = (req.DEF > 0) || ((pos.DEF || []).length > 0);
  document.querySelectorAll('.wv-pos-btn[data-pos="K"]').forEach(b => {{ b.hidden = !showK; }});
  document.querySelectorAll('.wv-pos-btn[data-pos="DEF"]').forEach(b => {{ b.hidden = !showDef; }});
}}

function wvStartSitPositions() {{
  if (wvCurrentPos !== 'ALL') return [wvCurrentPos];
  const keys = ['QB','RB','WR','TE'];
  const pos = wvStartSitData.positions || {{}};
  if ((pos.K || []).length) keys.push('K');
  if ((pos.DEF || []).length) keys.push('DEF');
  return keys;
}}

// ── Compact Start/Sit list ────────────────────────────────────────────────────
// Verdict + projection + one-line matchup is the whole row. Evidence
// (floor/ceiling, L4 PPG, Vegas, opp plays faced, profile, injury) expands on
// tap. The group verdict answers the decision before any row is read.
function wvLastName(full) {{
  const parts = String(full || '').trim().split(/\\s+/);
  const suffix = /^(II|III|IV|V|JR|SR)\\.?$/i;
  while (parts.length > 1 && suffix.test(parts[parts.length - 1])) parts.pop();
  return parts.length ? parts[parts.length - 1] : (full || '');
}}

// "Start Walker and Warren. Flex Love over Judkins." Derived from the same
// server flags as the badges, so it can never contradict them.
function wvSsGroupVerdict(players) {{
  const starts = players.filter(p => p.start === true);
  const flexStarts = players.filter(p => p.flex_start === true);
  const benchFlex = players.find(p => p.flex_eligible === true && p.start !== true && p.flex_start !== true);
  const bits = [];
  if (starts.length) {{
    const names = starts.map(p => '<b>' + wvLastName(p.name) + '</b>');
    bits.push('Start ' + (names.length > 1
      ? names.slice(0, -1).join(', ') + ' and ' + names[names.length - 1]
      : names[0]) + '.');
  }}
  if (flexStarts.length) {{
    const names = flexStarts.map(p => '<b>' + wvLastName(p.name) + '</b>').join(' and ');
    bits.push('Flex ' + names + (benchFlex ? ' over ' + wvLastName(benchFlex.name) : '') + '.');
  }}
  return bits.length ? `<div class="wv-cx-verdict">${{bits.join(' ')}}</div>` : '';
}}

function wvSsMatchupChip(rank, total) {{
  if (!rank || !total) return '';
  const pct = rank / total;
  const lbl = pct <= 0.25 ? 'easiest' : pct <= 0.5 ? 'favorable' : pct <= 0.75 ? 'tough' : 'hardest';
  const cls = pct <= 0.5 ? '' : ' bad';
  return `<span class="wv-cx-chip${{cls}}">#${{rank}} ${{lbl}}</span>`;
}}

// Evidence grid for one player: floor/ceiling, L4 PPG, Vegas, opp plays
// faced, profile, injury. Only rows with data render.
function wvSsEvidence(p) {{
  const ev = [];
  const c = p.consistency;
  if (c && !c.small_sample && c.floor != null && c.ceiling != null) {{
    ev.push(`<div class="wv-cx-ev"><span class="k">FLOOR - CEIL</span><span class="v">${{c.floor}} - ${{c.ceiling}}</span></div>`);
  }}
  if (p.recent_ppg > 0) {{
    ev.push(`<div class="wv-cx-ev"><span class="k">L4 PPG</span><span class="v">${{p.recent_ppg}}</span></div>`);
  }}
  if (p.implied_total != null) {{
    ev.push(`<div class="wv-cx-ev"><span class="k">VEGAS</span><span class="v">${{p.implied_total}} implied</span></div>`);
  }}
  const pv = p.play_volume;
  if (pv && pv.plays_faced_pg != null) {{
    let sub = '';
    if (pv.vs_avg != null) {{
      const a = Math.abs(pv.vs_avg).toFixed(1);
      sub = `<span class="s">${{pv.vs_avg > 0 ? a + ' above' : (pv.vs_avg < 0 ? a + ' below' : 'at')}} NFL avg</span>`;
    }}
    ev.push(`<div class="wv-cx-ev"><span class="k">OPP PLAYS FACED</span><span class="v">${{pv.plays_faced_pg}}/gm</span>${{sub}}</div>`);
  }}
  if (c && !c.small_sample && c.label) {{
    ev.push(`<div class="wv-cx-ev"><span class="k">PROFILE</span><span class="v">${{c.label}}</span></div>`);
  }}
  if (p.injury_status) {{
    const plan = p.return_plan;
    const sub = (plan && plan.verdict) ? `<span class="s">${{plan.verdict}}${{plan.weeks_label ? ' · ' + plan.weeks_label : ''}} (approx)</span>` : '';
    ev.push(`<div class="wv-cx-ev"><span class="k">INJURY</span><span class="v">${{p.injury_status}}</span>${{sub}}</div>`);
  }}
  if (!ev.length) return '';
  return `<div class="wv-cx-evidence">${{ev.join('')}}</div>`;
}}

function wvToggleSsRow(btn) {{
  const wrap = btn.closest('.wv-cx-row-wrap');
  if (!wrap) return;
  const detail = wrap.querySelector('.wv-cx-detail');
  if (!detail) return;
  const open = wrap.classList.toggle('wv-cx-expanded');
  btn.setAttribute('aria-expanded', open ? 'true' : 'false');
  detail.style.maxHeight = open ? detail.scrollHeight + 'px' : '0px';
}}
function wvRenderStartSit() {{
  const el = document.getElementById('wvStartSit');
  const positions = wvStartSitPositions();
  const reqs = wvStartSitData._lineup_requirements || {{}};

  const sections = positions.map(pos => {{
    const players = (wvStartSitData.positions || {{}})[pos] || [];
    if (!players.length) return '';

    let benchLineDone = false;
    const rows = players.slice(0, 8).map(p => {{
      const isStart     = p.start === true;
      const isFlexStart = p.flex_start === true;
      const isFlex      = p.flex_eligible === true && !isStart;
      const isBye       = p.on_bye === true;
      const isStarter   = isStart || isFlexStart;

      // BENCH LINE goes between the last starter and the first benched player.
      let sep = '';
      if (!isStarter && !isBye && !benchLineDone) {{
        // Only draw it when there was at least one starter above.
        const hadStarter = players.slice(0, 8).some(q => q.start === true || q.flex_start === true);
        if (hadStarter) sep = '<div class="wv-cx-benchline">BENCH LINE</div>';
        benchLineDone = true;
      }}

      const badge = isBye
        ? '<span class="wv-cx-badge wv-cx-bye">BYE</span>'
        : isFlexStart
          ? '<span class="wv-cx-badge wv-cx-flexstart">FLEX</span>'
          : isStart
            ? '<span class="wv-cx-badge wv-cx-start">START</span>'
            : isFlex
              ? '<span class="wv-cx-badge wv-cx-flexq">FLEX?</span>'
              : '<span class="wv-cx-badge wv-cx-sit">SIT</span>';

      const injBadge = wvInjBadge(p.injury_status);
      const escName = (p.name || '').replace(/'/g, "\\'");
      const matchup = p.opponent
        ? `${{p.opponent}} ${{wvSsMatchupChip(p.def_rank, p.def_total)}}` : '';
      const demoteChip = (p.demotion === 'low_total')
        ? '<span class="wv-cx-chip bad">Low team total</span>' : '';
      // Head-to-head win probability on the marginal call (server-flagged).
      const h2h = (isStart && p.close_call && p.close_call.win_prob != null)
        ? `<span class="wv-cx-h2h"><span class="bar"><i style="width:${{Math.round(p.close_call.win_prob * 100)}}%"></i></span><b>${{Math.round(p.close_call.win_prob * 100)}}%</b> to outscore ${{p.close_call.vs_name}}</span>`
        : '';
      const projBlock = (p.proj_pts > 0)
        ? `<span class="wv-cx-proj"><span class="wv-cx-proj-num">${{p.proj_pts}}</span><span class="wv-cx-proj-lbl">PROJ</span></span>`
        : '';
      const evidence = wvSsEvidence(p);
      const schedUrl = `${{wvLeaguePath('/schedule')}}?add=${{encodeURIComponent(p.player_id)}}`;

      return sep + `
        <div class="wv-cx-row-wrap">
          <button type="button" class="wv-cx-row" aria-expanded="false"
              onclick="wvToggleSsRow(this)">
            ${{badge}}
            <span class="wv-cx-main">
              <span class="wv-cx-name">${{p.name}}${{injBadge}}</span>
              ${{matchup || demoteChip ? `<span class="wv-cx-why">${{matchup}}${{demoteChip}}</span>` : ''}}
              ${{h2h}}
            </span>
            ${{projBlock}}
            <span class="wv-cx-chev" aria-hidden="true">›</span>
          </button>
          <div class="wv-cx-detail">
            ${{evidence}}
            <div class="wv-cx-actions" onclick="event.stopPropagation()">
              <button type="button" onclick="event.stopPropagation();openPlayerModal('${{p.player_id}}', '${{escName}}')">Open player</button>
              <button type="button" onclick="event.stopPropagation();window.location.href='${{schedUrl}}'">View schedule</button>
              <button type="button" class="${{wvIsSelected(p.player_id) ? 'selected' : ''}}"
                onclick="event.stopPropagation();wvToggleCompare(${{JSON.stringify(p).replace(/"/g, '&quot;')}})">${{wvIsSelected(p.player_id) ? '✓' : '+'}} Compare</button>
            </div>
          </div>
        </div>`;
    }}).join('');

    const slotCount = reqs[pos] || 1;
    const verdict = wvSsGroupVerdict(players.slice(0, 8));
    return `<div class="wv-cx-group"><div class="wv-cx-group-head">` +
      `<div class="wv-cx-group-title">${{pos}} <span>(${{slotCount}} starter${{slotCount > 1 ? 's' : ''}})</span></div>` +
      verdict + `</div><div class="wv-cx-card">${{rows}}</div></div>`;
  }}).join('');

  // Lineup advice banner only makes sense across the whole lineup, so show it
  // when no single position is filtered.
  const advice = (wvCurrentPos === 'ALL') ? wvLineupAdvice() : '';
  if (sections) {{ el.innerHTML = advice + sections; }}
  else if (advice) {{ el.innerHTML = advice; }}
  else {{ window.brEmptyState(el, {{ icon: 'search', title: 'No roster data', message: 'We couldn’t find a lineup to analyze for this position.' }}); }}
}}


// Optimal-lineup verdict: the points left on the bench vs the viewer's current
// starters, plus the specific swaps to fix it. Empty when the lineup is already
// optimal or the viewer has no starters set.
function wvLineupAdvice() {{
  const a = wvStartSitData.lineup_advice;
  if (!a || !a.has_current) return '';
  const esc = s => (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;');
  if (!a.swaps.length || a.delta < 1) {{
    return `<div class="wv-ss-advice wv-ss-advice-ok"><i class="fa-solid fa-circle-check" aria-hidden="true"></i> Your lineup is optimal. Start score ${{a.optimal_pts}}.</div>`;
  }}
  const swaps = a.swaps.slice(0, 4).map(s => {{
    const g = Number(s.gain) || 0;
    // Never prefix '+' onto a negative (that rendered as "+-5.0").
    const gainTxt = (g > 0 ? '+' : '') + g.toFixed(1);
    const gainCls = g < 0 ? ' wv-ss-swap-gain-neg' : '';
    const inPos = s.start && s.start.position ? ` <span class="wv-ss-swap-pos">${{esc(s.start.position)}}</span>` : '';
    const slot = s.slot || '';
    const slotNote = (slot && slot !== 'empty' && s.start && slot !== s.start.position)
      ? `<span class="wv-ss-swap-slot">${{esc(slot === 'SUPER_FLEX' ? 'SUPERFLEX' : slot)}}</span>` : '';
    if (!s.sit || !s.sit.name) {{
      return `<div class="wv-ss-swap">
        <span class="wv-ss-swap-in">${{esc(s.start.name)}}</span>${{inPos}}
        <span class="wv-ss-swap-arrow">into empty slot</span>
        ${{slotNote}}
        <span class="wv-ss-swap-gain${{gainCls}}">${{gainTxt}}</span>
      </div>`;
    }}
    const outPos = s.sit.position ? ` <span class="wv-ss-swap-pos">${{esc(s.sit.position)}}</span>` : '';
    return `<div class="wv-ss-swap">
      <span class="wv-ss-swap-in">${{esc(s.start.name)}}</span>${{inPos}}
      <span class="wv-ss-swap-arrow">over</span>
      <span class="wv-ss-swap-out">${{esc(s.sit.name)}}</span>${{outPos}}
      ${{slotNote}}
      <span class="wv-ss-swap-gain${{gainCls}}">${{gainTxt}}</span>
    </div>`;
  }}).join('');
  return `<div class="wv-ss-advice wv-ss-advice-warn">
    <div class="wv-ss-advice-head"><i class="fa-solid fa-arrow-trend-up" aria-hidden="true"></i> You're leaving <strong>${{a.delta.toFixed(1)}}</strong> start-score points on the bench</div>
    <div class="wv-ss-advice-sub">${{a.optimal_pts}} optimal vs ${{a.current_pts}} current lineup</div>
    ${{swaps}}
  </div>`;
}}

document.addEventListener('DOMContentLoaded', wvLoad);

// Deep link: ?tab=startsit opens the Start/Sit Advisor (switches the mobile
// tab and scrolls the section into view on desktop).
document.addEventListener('DOMContentLoaded', function() {{
  try {{
    const params = new URLSearchParams(window.location.search);
    if ((params.get('tab') || '').toLowerCase() === 'startsit') {{
      if (!document.getElementById('wvTabStartSit')) return;
      wvSetTab('startsit');
      const sec = document.getElementById('wvSectionStartSit');
      if (sec) sec.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }}
  }} catch (e) {{}}
}});
</script>
"""

    return style + html_body + script
