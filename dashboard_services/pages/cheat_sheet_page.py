"""
Draft Cheat Sheet page.

A printable, pre-draft board that is the static sibling of the Draft Room. When
opened from an active Draft Room it adds that room's live Recommendation ranks
as context without replacing the stable cheat-sheet order; when opened on its own it
ranks the shared /api/league-players pool by value-over-replacement using the
same roster-derived replacement index (BRPickScore.starterCounts) the draft room
and the server pick-score use. The draft room adds situational timing and roster
fit, then passes its resulting ranks into the in-draft sheet as a snapshot.

Self-contained like the draft room page: CSS is inlined here (scoped under
.cs-wrap and bridged to the site theme tokens), the render logic lives in
static/cheat_sheet.js, and server values arrive via a small window.__cheatCfg
blob the script reads on start. pick_score.js is loaded first for
BRPickScore.starterCounts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional


def _static_v(name: str) -> str:
    try:
        _p = Path(__file__).resolve().parents[2] / "static" / name
        return hashlib.md5(_p.read_bytes()).hexdigest()[:8]
    except OSError:
        return "0"


def build_cheat_sheet_body(
    league_id: Optional[str],
    season: Optional[int],
    platform: Optional[str] = None,
    *,
    num_teams: Optional[int] = None,
    is_superflex: bool = False,
    roster_positions: Optional[list] = None,
    scoring: Optional[dict] = None,
    mode: str = "redraft",
    viewer_user_id: Optional[str] = None,
    has_premium: bool = False,
) -> str:
    _has_league = bool(league_id and platform and season)
    cfg = {
        "leagueId": league_id or "",
        "season": int(season) if season else None,
        "platform": platform or "sleeper",
        "numTeams": int(num_teams) if num_teams else None,
        "isSuperflex": bool(is_superflex),
        "rosterPositions": list(roster_positions) if roster_positions else None,
        # Same {ppr, tep, passTd} contract as Draft Room setup.
        "scoring": scoring or None,
        "mode": "dynasty" if mode == "dynasty" else "redraft",
        "viewerUserId": str(viewer_user_id) if viewer_user_id else "",
        # Pro gate for live Sleeper sync and custom board edits
        # (the static board and CSV export are free).
        "hasPremium": bool(has_premium),
        "draftUrl": (
            f"/{platform}/{int(season)}/{league_id}/draft"
            if _has_league else "/draft"
        ),
    }
    # This JSON is embedded directly in a script element. Escape HTML-significant
    # characters so a malformed/tampered route value cannot terminate the script.
    cfg_json = (
        json.dumps(cfg)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    # Kick the player-pool fetch before the deferred board scripts download so
    # the redraft sheet's first paint overlaps JS parse instead of waiting on it.
    # Projection query params match Draft Room / DraftBoardCore scoring so the
    # prefetched pool is already scoring-aware (half PPR, TE premium, 6-pt TD).
    prefetch = (
        "<script>(function(){var c=window.__cheatCfg||{};"
        "var s=c.scoring||{};"
        "var ppr=s.ppr!=null?s.ppr:1,tep=s.tep!=null?s.tep:0,passTd=s.passTd>=6?6:4;"
        "var p=['view=board','league_type='+(c.isSuperflex?'sf':'1qb'),"
        "'proj_rec='+encodeURIComponent(String(ppr)),"
        "'proj_te_bonus='+encodeURIComponent(String(tep)),"
        "'proj_pass_td='+encodeURIComponent(String(passTd))];"
        "if(c.leagueId)p.push('league_id='+encodeURIComponent(c.leagueId));"
        "if(c.platform)p.push('platform='+encodeURIComponent(c.platform));"
        "var url='/api/league-players?'+p.join('&');"
        "var req=fetch(url,{cache:'no-store'}).then(function(r){"
        "if(!r.ok)throw new Error('Players request failed ('+r.status+')');"
        "return r.json();});req.url=url;window.__cheatPlayersP=req;})();</script>\n"
    )
    return (
        f"<script>window.__cheatCfg = {cfg_json};</script>\n"
        + prefetch
        + _CHEAT_HTML
        + f'\n<script src="/static/pick_score.js?v={_static_v("pick_score.js")}" defer></script>\n'
        + f'\n<script src="/static/draft_board_core.js?v={_static_v("draft_board_core.js")}" defer></script>\n'
        + f'\n<script src="/static/cheat_sheet.js?v={_static_v("cheat_sheet.js")}" defer></script>\n'
        + f'\n<script src="/static/custom_selects.js?v={_static_v("custom_selects.js")}" defer></script>\n'
    )


def build_cheat_sheet_embed_document(*args, **kwargs) -> str:
    """A full, chrome-less HTML document that renders ONLY the cheat sheet, for
    embedding in an iframe (the Draft Room's in-draft overlay). It links the site
    stylesheet for the theme tokens and mirrors the parent's light/dark choice via
    the shared same-origin localStorage, so it looks native inside the modal."""
    body = build_cheat_sheet_body(*args, **kwargs)
    css_v = _static_v("dashboard.css")
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Draft Cheat Sheet</title>"
        f"<link rel='stylesheet' href='/static/dashboard.css?v={css_v}'>"
        # Match the parent tab's theme (same-origin iframe shares localStorage).
        "<script>(function(){try{if(localStorage.getItem('theme')==='dark')"
        "document.documentElement.setAttribute('data-theme','dark');}catch(e){}})();</script>"
        "<style>html,body{margin:0;width:100%;min-width:0;min-height:100%;"
        "background:var(--bg,#eef1f7);overflow-x:hidden;}"
        "body{-webkit-overflow-scrolling:touch;}"
        ".cs-wrap{padding-top:14px;}</style></head><body>"
        + body +
        "</body></html>"
    )


# Plain (non-f) string — safe to contain { } freely.
_CHEAT_HTML = r"""
<style>
  /* Tokens live on both the sheet and the Hist overlay. The overlay is a
     sibling of .cs-wrap (so position:fixed is not clipped) and otherwise
     would miss --cs-surface / --cs-ink, leaving a transparent unstyled card. */
  .cs-wrap, .cs-hist-modal {
    --cs-surface: var(--card);
    --cs-surface-2: color-mix(in srgb, var(--text) 5%, var(--card));
    --cs-line: var(--border);
    --cs-line-strong: color-mix(in srgb, var(--text) 16%, var(--card));
    --cs-ink: var(--text);
    --cs-ink-soft: var(--text-muted);
    --cs-ink-faint: var(--text-subtle, var(--text-muted));
    --cs-accent: var(--accent, #38bdf8);
    --cs-accent-soft: color-mix(in srgb, var(--accent, #38bdf8) 16%, transparent);
    --cs-good: var(--win, #15803d);
    --cs-good-soft: color-mix(in srgb, var(--win, #15803d) 15%, transparent);
    --cs-bad: var(--loss, #b91c1c);
    --cs-bad-soft: color-mix(in srgb, var(--loss, #b91c1c) 15%, transparent);
    --cs-amber: var(--warning, #b5730b);
    --cs-amber-soft: color-mix(in srgb, var(--warning, #b5730b) 16%, transparent);
    --cs-qb: #3b82f6; --cs-qb-bg: rgba(59,130,246,.14);
    --cs-rb: #22c55e; --cs-rb-bg: rgba(34,197,94,.14);
    --cs-wr: #f59e0b; --cs-wr-bg: rgba(245,158,11,.16);
    --cs-te: #8b5cf6; --cs-te-bg: rgba(139,92,246,.14);
    --cs-pos: var(--cs-accent); --cs-pos-bg: var(--cs-accent-soft);
    --cs-bar: color-mix(in srgb, var(--accent, #38bdf8) 55%, transparent);
    --cs-bar-track: color-mix(in srgb, var(--text) 9%, transparent);
    --cs-mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
  }
  .cs-wrap {
    max-width: 1120px; margin: 0 auto; padding: 6px 4px 60px; color: var(--cs-ink);
  }
  .cs-wrap *, .cs-hist-modal, .cs-hist-modal * { box-sizing: border-box; }

  .cs-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; flex-wrap: nowrap; }
  .cs-top > :first-child { flex: 1 1 390px; min-width: 300px; }
  .cs-eyebrow { font-family: var(--cs-mono); font-size: 11px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--cs-accent); display: inline-flex; align-items: center; gap: 8px; }
  .cs-wrap h1 { font-size: clamp(23px, 4vw, 32px); line-height: 1.06; margin: 6px 0 4px; letter-spacing: -.02em; font-weight: 800; }
  .cs-sub { color: var(--cs-ink-soft); font-size: 14px; max-width: 64ch; margin: 0; line-height: 1.5; }
  .cs-backlink { font-size: 13px; font-weight: 700; color: var(--cs-accent); text-decoration: none; }
  .cs-backlink:hover { text-decoration: underline; }

  /* Controls sit top-right, beside the title, using the empty space next to the
     header text (space-between on .cs-top pushes them there). They stack mode/QB
     over the action buttons and right-align. On mobile they drop full-width. */
  .cs-controls { display: flex; flex: 0 0 auto; flex-direction: column; gap: 8px; align-items: flex-end; }
  .cs-ctrl-row { display: flex; align-items: center; gap: 9px; flex-wrap: nowrap; justify-content: flex-end; }
  .cs-scoring-row { flex-wrap: wrap; }
  .cs-cgroup { display: inline-flex; align-items: center; gap: 7px; }
  .cs-cgroup.cs-score { gap: 5px; }
  .cs-cgroup.cs-score .cs-src, .cs-cgroup.cs-score .csd-wrap { min-width: 0; }
  .cs-clabel { font-family: var(--cs-mono); font-size: 9.5px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--cs-ink-faint); }
  .cs-seg { display: inline-flex; padding: 3px; gap: 2px; background: var(--cs-surface-2); border: 1px solid var(--cs-line); border-radius: 10px; }
  .cs-seg button { font: inherit; font-size: 12px; font-weight: 700; cursor: pointer; border: 0; background: transparent; color: var(--cs-ink-soft); padding: 5px 11px; border-radius: 7px; }
  .cs-seg button[aria-pressed="true"] { background: var(--cs-accent); color: #fff; }
  .cs-seg.mode button[aria-pressed="true"] { background: var(--cs-ink); color: var(--cs-surface); }
  .cs-btn { font: inherit; font-size: 12px; font-weight: 700; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; background: var(--cs-surface); color: var(--cs-ink-soft); border: 1px solid var(--cs-line); border-radius: 9px; padding: 7px 11px; }
  .cs-btn:hover { border-color: var(--cs-accent); color: var(--cs-accent); }
  .cs-btn[aria-pressed="true"] { border-color: var(--cs-good); color: var(--cs-good); background: var(--cs-good-soft); }
  .cs-src { font: inherit; font-size: 12px; font-weight: 700; cursor: pointer; background: var(--cs-surface); color: var(--cs-ink-soft); border: 1px solid var(--cs-line); border-radius: 9px; padding: 7px 28px 7px 9px; }
  .cs-src:hover { border-color: var(--cs-accent); }
  /* Custom-select wrapper (CSD) replaces the native <select> chrome. */
  .cs-wrap .csd-wrap { vertical-align: middle; }
  .cs-wrap .csd-trigger { font-size: 12px; font-weight: 700; background: var(--cs-surface); color: var(--cs-ink-soft);
    border: 1px solid var(--cs-line); border-radius: 9px; padding: 7px 10px 7px 9px; }
  .cs-wrap .csd-trigger:hover { border-color: var(--cs-accent); }
  .cs-wrap .csd-list { z-index: 30; }
  /* Reset actions (Clear marks / Reset board) are secondary: they flow inline at
     the end of the controls, de-emphasized so a busy row still reads cleanly. */
  .cs-btn-reset { color: var(--cs-ink-faint); }
  .cs-btn-reset:hover { color: var(--cs-accent); border-color: var(--cs-accent); }

  /* ── Custom draft board (pro): overrides on top of the model board ────────── */
  /* The edit column is hidden until Edit board is on. */
  .cs-edit-th, .cs-edit-cell { display: none; }
  .cs-board.editing .cs-edit-th, .cs-board.editing .cs-edit-cell { display: table-cell; }
  .cs-edit-cell { text-align: right !important; white-space: nowrap; }
  .cs-ovbtns { display: inline-flex; gap: 3px; }
  .cs-ovbtn { font: inherit; font-size: 12px; line-height: 1; cursor: pointer; width: 24px; height: 24px;
    display: inline-flex; align-items: center; justify-content: center; padding: 0; border-radius: 6px;
    border: 1px solid var(--cs-line); background: var(--cs-surface); color: var(--cs-ink-soft); }
  .cs-ovbtn:hover { border-color: var(--cs-accent); color: var(--cs-accent); }
  .cs-ovbtn.on { background: var(--cs-accent); border-color: var(--cs-accent); color: #fff; }
  /* The "muted" button lights up red-ish since it sinks the player. */
  .cs-ovbtn[data-act="mute"].on { background: var(--cs-bad); border-color: var(--cs-bad); }
  /* Drag handle: grab cursor, and touch-action:none so a touch-drag reorders the
     row instead of scrolling the list. */
  .cs-drag { cursor: grab; touch-action: none; color: var(--cs-ink-faint); }
  .cs-drag:active { cursor: grabbing; }
  .cs-revert { color: var(--cs-ink-faint); }
  .cs-wrap tbody tr.cs-dragging { opacity: .45; }
  /* Insertion indicator that tracks the drop point during a drag. */
  .cs-drop-line { position: absolute; left: 0; right: 0; height: 2px; margin-top: -1px; background: var(--cs-accent); pointer-events: none; z-index: 5; }
  .cs-drop-line::before { content: ""; position: absolute; left: 0; top: -3px; width: 8px; height: 8px; border-radius: 50%; background: var(--cs-accent); }
  /* Override state chip next to the name. */
  .cs-ovchip { font-family: var(--cs-mono); font-size: 10px; font-weight: 800; padding: 1px 6px; border-radius: var(--radius-pill, 8px); margin-left: 8px; white-space: nowrap; }
  .cs-ovchip.bump { color: var(--cs-accent); background: var(--cs-accent-soft); }
  .cs-ovchip.pin { color: var(--cs-good); background: var(--cs-good-soft); }
  .cs-ovchip.mute { color: var(--cs-ink-faint); background: var(--cs-surface-2); }
  .cs-wrap tbody tr.cs-muted td { opacity: .5; }
  .cs-wrap tbody tr.cs-muted .cs-pname { color: var(--cs-ink-faint); }
  /* A subtle accent rail on any row you have personally moved. */
  .cs-wrap tbody tr.cs-ov td:first-child { box-shadow: inset 3px 0 0 var(--cs-accent); }
  /* One-shot highlight that confirms where a moved row landed. */
  .cs-wrap tbody tr.cs-flash td { animation: csFlash .65s ease-out; }
  @keyframes csFlash { 0% { background: var(--cs-accent-soft); } 100% { background: transparent; } }

  .cs-tabs { display: flex; gap: 4px; margin: 20px 0 0; border-bottom: 1px solid var(--cs-line); flex-wrap: wrap; }
  .cs-tabs button { font: inherit; font-size: 13.5px; font-weight: 700; cursor: pointer; border: 0; background: none; color: var(--cs-ink-faint); padding: 11px 14px; position: relative; }
  .cs-tabs button[aria-selected="true"] { color: var(--cs-ink); }
  .cs-tabs button[aria-selected="true"]::after { content: ""; position: absolute; left: 12px; right: 12px; bottom: -1px; height: 2.5px; background: var(--cs-accent); border-radius: 3px; }

  .cs-legend { display: flex; flex-wrap: wrap; gap: 8px 20px; align-items: center; background: var(--cs-surface); border: 1px solid var(--cs-line); border-radius: 12px; padding: 11px 16px; margin: 16px 0 18px; font-size: 12.5px; color: var(--cs-ink-soft); }
  .cs-legend b { color: var(--cs-ink); }
  .cs-lg { display: inline-flex; align-items: center; gap: 7px; white-space: nowrap; }
  .cs-val { font-family: var(--cs-mono); font-weight: 800; font-size: 11px; line-height: 1; padding: 2px 6px; border-radius: 5px; }
  .cs-val.g { color: var(--cs-good); background: var(--cs-good-soft); }
  .cs-val.b { color: var(--cs-bad); background: var(--cs-bad-soft); }
  .cs-val.n { color: var(--cs-ink-faint); }
  #csFmtNote { margin-left: auto; font-family: var(--cs-mono); color: var(--cs-ink-faint); }

  .cs-tbl-scroll { position: relative; overflow: auto; max-width: 100%; max-height: calc(100vh - 250px); background: var(--cs-surface); border: 1px solid var(--cs-line); border-radius: 14px; -webkit-overflow-scrolling: touch; touch-action: pan-x pan-y; }
  .cs-wrap table { border-collapse: collapse; width: 100%; min-width: 640px; }
  .cs-wrap thead th { position: sticky; top: 0; z-index: 3; font-family: var(--cs-mono); font-size: 10.5px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--cs-ink-faint); text-align: right; padding: 12px 14px 9px; border-bottom: 1px solid var(--cs-line); background: var(--cs-surface); }
  .cs-wrap thead th.l { text-align: left; }
  .cs-wrap thead th.cs-sort { cursor: pointer; user-select: none; white-space: nowrap; }
  .cs-wrap thead th.cs-sort:hover, .cs-wrap thead th.cs-sort:hover .cs-sortbtn { color: var(--cs-ink); }
  .cs-wrap thead th.cs-sort-asc, .cs-wrap thead th.cs-sort-desc,
  .cs-wrap thead th.cs-sort-asc .cs-sortbtn, .cs-wrap thead th.cs-sort-desc .cs-sortbtn { color: var(--cs-accent); }
  .cs-wrap thead th.cs-sort-asc .cs-sortbtn::after { content: " ▲"; font-size: 8px; margin-left: 3px; }
  .cs-wrap thead th.cs-sort-desc .cs-sortbtn::after { content: " ▼"; font-size: 8px; margin-left: 3px; }
  .cs-sortbtn { font: inherit; font-size: inherit; font-weight: inherit; letter-spacing: inherit; text-transform: inherit; color: inherit; background: none; border: 0; padding: 0; cursor: pointer; }
  .cs-wrap thead th.l .cs-sortbtn { text-align: left; }
  .cs-wrap tbody td { padding: 8px 14px; border-bottom: 1px solid var(--cs-line); font-size: 13.5px; text-align: right; vertical-align: middle; }
  .cs-wrap tbody tr:last-child td { border-bottom: 0; }
  .cs-wrap tbody tr.cs-p { cursor: pointer; }
  .cs-wrap tbody tr.cs-p:hover td { background: var(--cs-surface-2); }
  .cs-wrap tbody tr.done td { opacity: .4; }
  .cs-wrap tbody tr.done .cs-pname { text-decoration: line-through; }
  .cs-empty { text-align: center !important; color: var(--cs-ink-faint); padding: 26px 14px !important; }
  .cs-rk { font-family: var(--cs-mono); font-weight: 800; color: var(--cs-ink); }
  .cs-pcell { text-align: left; display: flex; align-items: center; gap: 9px; }
  .cs-pname { font-weight: 600; white-space: nowrap; color: var(--cs-ink); }
  .cs-num { font-family: var(--cs-mono); color: var(--cs-ink-soft); font-variant-numeric: tabular-nums; }
  .cs-ppg-last { color: var(--cs-ink-faint, var(--cs-ink-soft)); }
  .cs-posrk { font-family: var(--cs-mono); font-size: 11px; font-weight: 700; padding: 2px 6px; border-radius: 6px; }
  .cs-vorwrap { display: inline-flex; align-items: center; gap: 8px; justify-content: flex-end; }
  .cs-vorbar { width: 60px; height: 6px; border-radius: 12px; background: var(--cs-bar-track); overflow: hidden; flex-shrink: 0; }
  .cs-vorbar > i { display: block; height: 100%; background: var(--cs-bar); border-radius: 12px; }
  /* Market vs ADP remains part of the same primary table as VOR and Value. */
  .cs-hist-col { white-space: nowrap; }
  .cs-hist-cell { display: inline-flex; align-items: center; gap: 6px; justify-content: flex-end; }
  .cs-hist-btn { font: inherit; font-family: var(--cs-mono); font-size: 10px; font-weight: 800; cursor: pointer; width: 18px; height: 18px; padding: 0; border-radius: 5px; border: 1px solid var(--cs-line); background: var(--cs-surface); color: var(--cs-ink-faint); line-height: 1; }
  .cs-hist-btn:hover { border-color: var(--cs-pos); color: var(--cs-pos); }
  .cs-hist-modal { display: none; position: fixed; inset: 0; z-index: var(--z-modal, 10000); background: rgba(15, 23, 42, 0.7); backdrop-filter: blur(4px); -webkit-backdrop-filter: blur(4px); align-items: center; justify-content: center; padding: 16px; padding-bottom: max(16px, env(safe-area-inset-bottom)); }
  .cs-hist-modal.open { display: flex; }
  .cs-hist-card { background: var(--cs-surface, var(--card)); color: var(--cs-ink, var(--text)); border: 1px solid var(--cs-line, var(--border)); border-radius: 14px; max-width: 560px; width: 100%; max-height: min(80vh, 640px); overflow: auto; padding: 18px 20px 20px; box-shadow: 0 16px 48px color-mix(in srgb, #000 28%, transparent); }
  .cs-hist-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin: 0 0 14px; }
  .cs-hist-head > div { min-width: 0; flex: 1; }
  .cs-hist-card h2 { display: flex; align-items: center; gap: 8px; font-size: 16px; font-weight: 800; line-height: 1.2; margin: 0 0 6px; color: var(--cs-ink, var(--text)); }
  .cs-hist-sub { color: var(--cs-ink-soft, var(--text-muted)); font-size: 12.5px; margin: 0 0 12px; line-height: 1.45; }
  .cs-hist-head .cs-hist-sub { margin-bottom: 0; }
  .cs-hist-close { flex-shrink: 0; font: inherit; font-size: 12px; font-weight: 700; cursor: pointer; border: 1px solid var(--cs-line, var(--border)); background: var(--cs-surface, var(--card)); color: var(--cs-ink-soft, var(--text-muted)); border-radius: 8px; padding: 5px 9px; }
  .cs-hist-close:hover { border-color: var(--cs-pos); color: var(--cs-pos); }
  .cs-hist-dl { display: grid; grid-template-columns: max-content minmax(0, 1fr); column-gap: 16px; row-gap: 10px; align-items: baseline; font-size: 13px; margin: 0 0 16px; }
  .cs-hist-dl dt, .cs-hist-dl dd { margin: 0; }
  .cs-hist-dl dt { grid-column: 1; color: var(--cs-ink-soft, var(--text-muted)); font-family: var(--cs-mono); font-size: 10.5px; letter-spacing: .06em; text-transform: uppercase; white-space: nowrap; }
  .cs-hist-dl dd { grid-column: 2; color: var(--cs-ink, var(--text)); min-width: 0; overflow-wrap: anywhere; }
  .cs-hist-sec { margin: 0 0 16px; }
  .cs-hist-sec:last-child { margin-bottom: 0; }
  .cs-hist-sec h3 { font-size: 10.5px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; color: var(--cs-ink-soft, var(--text-muted)); margin: 0 0 8px; }
  .cs-hist-hits { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; border: 1px solid var(--cs-line, var(--border)); border-radius: 14px; overflow: hidden; background: var(--cs-line, var(--border)); }
  .cs-hist-hit { display: flex; flex-direction: column; gap: 0; padding: 10px 12px 11px; border: 0; border-radius: 0; background: var(--cs-surface, var(--card)); min-width: 0; }
  .cs-hist-hit:last-child:nth-child(odd) { grid-column: 1 / -1; }
  .cs-hist-hit-top { display: flex; align-items: baseline; gap: 8px; }
  .cs-hist-hit-top > div { flex: 1; min-width: 0; }
  .cs-hist-hit-label { font-size: 13px; font-weight: 700; color: var(--cs-ink, var(--text)); line-height: 1.3; overflow-wrap: anywhere; }
  .cs-hist-hit-meta { font-family: var(--cs-mono); font-size: 11px; color: var(--cs-ink-soft, var(--text-muted)); margin-top: 2px; }
  .cs-hist-hit-pct { font-size: 15px; font-weight: 800; font-variant-numeric: tabular-nums; color: var(--cs-ink, var(--text)); white-space: nowrap; }
  .cs-hist-hit-pct span { font-size: 11px; font-weight: 700; color: var(--cs-pos); }
  .cs-hist-hit.is-miss .cs-hist-hit-pct span { color: var(--cs-ink-faint, var(--text-muted)); }
  .cs-hist-hit .cs-trends-rail { margin-top: 6px; }
  .cs-hist-hit-top .cs-trends-conf { align-self: center; }
  .cs-hist-hit.is-this { box-shadow: inset 3px 0 0 var(--cs-pos, #22c55e); }
  .cs-hist-hit-role { display: block; margin: 3px 0 0; font-size: 10px; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; color: var(--cs-pos); }
  .cs-hist-tile-ex { margin-top: 6px; }
  .cs-hist-tile-ex > summary { cursor: pointer; list-style: none; font-size: 11px; font-weight: 700; color: var(--cs-ink-soft, var(--text-muted)); line-height: 1.35; overflow-wrap: anywhere; }
  .cs-hist-tile-ex > summary::-webkit-details-marker { display: none; }
  .cs-hist-tile-ex > summary::after { content: " +"; color: var(--cs-pos); }
  .cs-hist-tile-ex[open] > summary::after { content: "\2013"; }
  .cs-hist-tile-ex ul { list-style: none; margin: 6px 0 0; padding: 0; }
  .cs-hist-tile-ex li { display: flex; justify-content: space-between; gap: 8px; font-size: 12px; padding: 3px 0; color: var(--cs-ink, var(--text)); }
  .cs-hist-tile-ex li .cs-hist-ex-hit { font-size: 10px; }
  .cs-hist-profile { display: flex; flex-wrap: wrap; gap: 6px; }
  .cs-hist-chip { display: inline-flex; flex-direction: column; gap: 2px; border: 1px solid var(--cs-line, var(--border)); border-radius: 8px; padding: 6px 8px; min-width: 0; background: var(--cs-surface, var(--card)); }
  .cs-hist-chip-k { font-size: 10px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; color: var(--cs-ink-faint, var(--text-muted)); }
  .cs-hist-chip-v { font-size: 13px; font-weight: 700; color: var(--cs-ink, var(--text)); }
  .cs-hist-note { font-size: 12.5px; color: var(--cs-ink-soft, var(--text-muted)); line-height: 1.45; margin: 0 0 8px; }
  .cs-hist-note:last-child { margin-bottom: 0; }
  .cs-hist-ex { list-style: none; margin: 0; padding: 0; }
  .cs-hist-ex li { display: flex; justify-content: space-between; gap: 10px; padding: 8px 0; border-bottom: 1px solid var(--cs-line, var(--border)); font-size: 13px; align-items: flex-start; }
  .cs-hist-ex li:last-child { border-bottom: 0; }
  .cs-hist-ex li > span:first-child { min-width: 0; }
  .cs-hist-ex-right { display: flex; flex-direction: column; align-items: flex-end; gap: 2px; flex-shrink: 0; color: var(--cs-ink-soft, var(--text-muted)); font-family: var(--cs-mono); font-size: 12px; text-align: right; }
  .cs-hist-ex-meta { white-space: nowrap; }
  .cs-hist-ex-hit { font-size: 10.5px; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; color: var(--cs-ink-faint, var(--text-muted)); }
  .cs-hist-ex li.is-top_5 .cs-hist-ex-hit, .cs-hist-ex li.is-top_12 .cs-hist-ex-hit { color: var(--cs-pos); }
  .cs-hist-ex li.is-top_24 .cs-hist-ex-hit { color: var(--cs-ink, var(--text)); }
  .cs-hist-ex small { display: block; font-size: 11.5px; font-weight: 500; color: var(--cs-ink-soft, var(--text-muted)); margin-top: 2px; line-height: 1.4; overflow-wrap: anywhere; }
  .cs-hist-ex-sum { font-size: 12.5px; color: var(--cs-ink-soft, var(--text-muted)); margin: 0 0 8px; }
  .cs-hist-closest { margin-top: 4px; }
  .cs-hist-closest > summary { cursor: pointer; list-style: none; display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 12px 14px; border: 1px solid var(--cs-line, var(--border)); border-radius: 12px; background: var(--cs-surface-2, var(--card)); }
  .cs-hist-closest[open] > summary { border-bottom-left-radius: 0; border-bottom-right-radius: 0; }
  .cs-hist-closest > summary:hover { border-color: var(--cs-pos); }
  .cs-hist-closest > summary::-webkit-details-marker { display: none; }
  .cs-hist-closest > summary::after { content: "+"; flex-shrink: 0; font-size: 15px; line-height: 1; color: var(--cs-pos); }
  .cs-hist-closest[open] > summary::after { content: "\2013"; }
  .cs-hist-closest > summary:focus-visible { outline: 2px solid var(--cs-pos); outline-offset: 3px; border-radius: 4px; }
  .cs-hist-closest > summary h3 { margin: 0; }
  .cs-hist-ex-peek { font-family: var(--cs-mono); font-size: 11px; font-weight: 700; color: var(--cs-ink-soft, var(--text-muted)); text-align: right; min-width: 0; overflow-wrap: anywhere; }
  .cs-hist-closest-body { padding: 12px 14px 4px; border: 1px solid var(--cs-line, var(--border)); border-top: 0; border-radius: 0 0 12px 12px; }
  .cs-hist-vp { margin: 14px 0 0; }
  .cs-hist-vp-h { font-size: 10.5px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; color: var(--cs-ink-soft, var(--text-muted)); margin: 0 0 10px; }
  .cs-hist-vp-row { margin: 0 0 10px; }
  .cs-hist-vp-row:last-child { margin-bottom: 0; }
  .cs-hist-vp-top { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin: 0 0 5px; }
  .cs-hist-vp-k { font-size: 12.5px; font-weight: 700; color: var(--cs-ink, var(--text)); min-width: 0; overflow-wrap: anywhere; }
  .cs-hist-vp-v { font-family: var(--cs-mono); font-size: 12.5px; font-weight: 800; font-variant-numeric: tabular-nums; white-space: nowrap; }
  .cs-hist-vp-track { height: 9px; border-radius: 999px; background: var(--cs-surface-2, var(--card)); overflow: hidden; }
  .cs-hist-vp-fill { height: 100%; border-radius: 999px; }
  .cs-hist-vp-fill.is-hist { background: var(--cs-pos); }
  .cs-hist-vp-fill.is-mkt { background: var(--cs-line-strong, var(--border)); }
  .cs-hist-edge { display: flex; align-items: center; gap: 7px; margin: 11px 0 0; font-size: 12.5px; line-height: 1.4; color: var(--cs-ink-soft, var(--text-muted)); }
  .cs-hist-edge svg { flex-shrink: 0; }
  .cs-hist-edge.is-up { color: var(--cs-good, var(--win)); }
  .cs-hist-edge.is-down { color: var(--cs-bad, var(--loss)); }
  .cs-hist-edge.is-even { color: var(--cs-ink-soft, var(--text-muted)); }
  .cs-hist-gap { font-size: 12.5px; color: var(--cs-ink-soft, var(--text-muted)); line-height: 1.45; margin: 11px 0 0; }
  .cs-hist-tp { border: 1px solid var(--cs-line, var(--border)); border-radius: 12px; overflow: hidden; }
  .cs-hist-tp-row { display: flex; align-items: center; gap: 11px; padding: 9px 13px; border-bottom: 1px solid var(--cs-line, var(--border)); }
  .cs-hist-tp-row:last-child { border-bottom: 0; }
  .cs-hist-tp-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; background: var(--cs-line-strong, var(--border)); }
  .cs-hist-tp-dot.is-up { background: var(--cs-pos); }
  .cs-hist-tp-dot.is-miss { background: var(--cs-ink-faint, var(--text-muted)); }
  .cs-hist-tp-main { flex: 1; min-width: 0; }
  .cs-hist-tp-label { font-size: 12.5px; font-weight: 700; color: var(--cs-ink, var(--text)); line-height: 1.3; overflow-wrap: anywhere; }
  .cs-hist-tp-meta { font-family: var(--cs-mono); font-size: 10px; color: var(--cs-ink-faint, var(--text-muted)); margin-top: 1px; }
  .cs-hist-tp-bar { width: 84px; height: 6px; border-radius: 999px; background: var(--cs-surface-2, var(--card)); overflow: hidden; flex-shrink: 0; }
  .cs-hist-tp-fill { height: 100%; border-radius: 999px; }
  .cs-hist-tp-fill.is-up { background: var(--cs-pos); }
  .cs-hist-tp-fill.is-neutral { background: var(--cs-line-strong, var(--border)); }
  .cs-hist-tp-pct { font-family: var(--cs-mono); font-size: 12.5px; font-weight: 800; font-variant-numeric: tabular-nums; width: 58px; text-align: right; white-space: nowrap; color: var(--cs-ink, var(--text)); }
  .cs-hist-tp-vs { font-size: 10px; font-weight: 700; }
  .cs-hist-tp-vs.is-up { color: var(--cs-good, var(--win)); }
  .cs-hist-tp-vs.is-down { color: var(--cs-bad, var(--loss)); }
  .cs-hist-tmore { margin-top: 8px; }
  .cs-hist-tmore > summary { cursor: pointer; list-style: none; font-size: 11.5px; font-weight: 700; color: var(--cs-pos); padding: 2px 0; }
  .cs-hist-tmore > summary::-webkit-details-marker { display: none; }
  .cs-hist-tmore[open] > summary { margin-bottom: 8px; }
  .cs-trends-qb, .cs-hist-modal.cs-hist-qb { --cs-pos: var(--cs-qb); --cs-pos-bg: var(--cs-qb-bg); }
  .cs-trends-rb, .cs-hist-modal.cs-hist-rb { --cs-pos: var(--cs-rb); --cs-pos-bg: var(--cs-rb-bg); }
  .cs-trends-wr, .cs-hist-modal.cs-hist-wr { --cs-pos: var(--cs-wr); --cs-pos-bg: var(--cs-wr-bg); }
  .cs-trends-te, .cs-hist-modal.cs-hist-te { --cs-pos: var(--cs-te); --cs-pos-bg: var(--cs-te-bg); }
  .cs-trends { padding: 4px 0 24px; }
  .cs-trends-lede { color: var(--cs-ink-soft, var(--text-muted)); font-size: 13px; line-height: 1.45; margin: 0 0 14px; }
  .cs-trends-pos, .cs-trends-lanes, .cs-trends-tiers { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; margin: 0 0 16px; }
  .cs-trends-pos button, .cs-trends-lanes button, .cs-trends-tiers button { font: inherit; font-size: 12.5px; font-weight: 700; cursor: pointer; border: 1px solid var(--cs-line, var(--border)); background: var(--cs-surface, var(--card)); color: var(--cs-ink-soft, var(--text-muted)); border-radius: 8px; padding: 6px 10px; }
  .cs-trends-pos button[data-trends-pos="QB"][aria-pressed="true"] { border-color: var(--cs-qb); color: var(--cs-qb); background: var(--cs-qb-bg); }
  .cs-trends-pos button[data-trends-pos="RB"][aria-pressed="true"] { border-color: var(--cs-rb); color: var(--cs-rb); background: var(--cs-rb-bg); }
  .cs-trends-pos button[data-trends-pos="WR"][aria-pressed="true"] { border-color: var(--cs-wr); color: var(--cs-wr); background: var(--cs-wr-bg); }
  .cs-trends-pos button[data-trends-pos="TE"][aria-pressed="true"] { border-color: var(--cs-te); color: var(--cs-te); background: var(--cs-te-bg); }
  .cs-trends-lanes button[aria-pressed="true"], .cs-trends-tiers button[aria-pressed="true"] { border-color: var(--cs-pos); color: var(--cs-pos); background: var(--cs-pos-bg); }
  .cs-trends-lane-n, .cs-trends-tier-n { margin-left: auto; font-family: var(--cs-mono); font-size: 11px; color: var(--cs-ink-faint, var(--text-muted)); }
  .cs-trends-sticky { position: sticky; top: var(--cs-nav-offset, 0px); z-index: 5; display: grid; gap: 8px; margin: 0 0 12px; padding: 8px 10px 10px; border: 1px solid color-mix(in srgb, var(--cs-pos) 22%, var(--cs-line, var(--border))); border-radius: 14px; background: color-mix(in srgb, var(--cs-surface, var(--card)) 86%, var(--cs-pos-bg)); box-shadow: 0 8px 18px color-mix(in srgb, #000 12%, transparent); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); max-height: min(42vh, 340px); overflow: auto; }
  .cs-trends-sticky.is-picked { grid-template-columns: minmax(240px, .95fr) minmax(0, 1.15fr); align-items: stretch; }
  .cs-trends-sticky.is-picked .cs-trends-lanes { grid-column: 1 / -1; }
  .cs-trends-sticky .cs-trends-lanes { margin: 0; }
  .cs-trends-sticky-body { display: contents; }
  .cs-trends-sticky.is-collapsed { max-height: none; overflow: visible; grid-template-columns: 1fr; padding: 6px 10px; }
  .cs-trends-sticky.is-collapsed .cs-trends-sticky-body { display: none; }
  .cs-trends-lanes button.cs-trends-sticky-toggle { margin-left: auto; font-size: 11px; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; color: var(--cs-pos); border-color: color-mix(in srgb, var(--cs-pos) 35%, var(--cs-line, var(--border))); background: var(--cs-pos-bg); }
  .cs-trends-sticky .cs-hist-note { font-size: 10.5px; margin: 6px 0 0; line-height: 1.3; }
  .cs-trends-summary { display: flex; align-items: flex-end; gap: 16px; margin: 0 0 18px; padding: 12px 14px; border-radius: 14px; border: 1px solid color-mix(in srgb, var(--cs-pos) 28%, var(--cs-line, var(--border))); background: var(--cs-pos-bg); }
  .cs-trends-base-pct { font-weight: 800; font-size: 46px; line-height: .82; letter-spacing: -.03em; color: var(--cs-pos); font-variant-numeric: tabular-nums; }
  .cs-trends-base-pct sup { font-size: 18px; font-weight: 800; color: var(--cs-ink-soft, var(--text-muted)); vertical-align: super; }
  .cs-trends-base-copy { min-width: 0; padding-bottom: 2px; }
  .cs-trends-base-k { font-size: 13.5px; font-weight: 700; color: var(--cs-ink, var(--text)); }
  .cs-trends-base-v { font-size: 12.5px; color: var(--cs-ink-soft, var(--text-muted)); margin-top: 3px; line-height: 1.45; }
  .cs-trends-sec-head { margin: 0 0 6px; }
  .cs-trends-sec-head h3 { font-size: 10px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; color: var(--cs-ink-soft, var(--text-muted)); margin: 0 0 2px; }
  .cs-trends-sec-head p { font-size: 12px; color: var(--cs-ink-soft, var(--text-muted)); line-height: 1.35; margin: 0; }
  .cs-trends-board { margin: 0 0 18px; }
  .cs-trends-callouts { display: grid; grid-template-columns: 1fr 1fr; margin: 0; border: 1px solid var(--cs-line, var(--border)); border-radius: 14px; overflow: hidden; background: var(--cs-surface, var(--card)); }
  .cs-trends-callout-col { display: flex; flex-direction: column; min-width: 0; }
  .cs-trends-callout-col + .cs-trends-callout-col { border-left: 1px solid var(--cs-line, var(--border)); }
  .cs-trends-callout { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(72px, .9fr) auto; gap: 4px 10px; align-items: center; padding: 9px 12px; border: 0; border-bottom: 1px solid var(--cs-line, var(--border)); border-radius: 0; background: transparent; }
  .cs-trends-callout:last-child { border-bottom: 0; }
  .cs-trends-callout-copy { display: flex; align-items: baseline; gap: 8px; min-width: 0; }
  .cs-trends-rk { flex-shrink: 0; font-family: var(--cs-mono); font-size: 10px; font-weight: 800; color: var(--cs-pos); width: 14px; }
  .cs-trends-callout-v { font-size: 13px; font-weight: 700; color: var(--cs-ink, var(--text)); min-width: 0; }
  .cs-trends-callout-pct { font-size: 15px; font-weight: 800; font-variant-numeric: tabular-nums; white-space: nowrap; text-align: right; }
  .cs-trends-callout-pct span { font-size: 11px; font-weight: 700; color: var(--cs-pos); }
  .cs-trends-rail { position: relative; height: 14px; min-width: 64px; }
  .cs-trends-rail-track, .cs-trends-rail-fill { position: absolute; left: 0; right: 0; top: 50%; height: 4px; transform: translateY(-50%); border-radius: 99px; }
  .cs-trends-rail-track { background: color-mix(in srgb, var(--cs-ink, #0f172a) 10%, transparent); }
  .cs-trends-rail-fill { right: auto; background: var(--cs-pos); }
  .cs-trends-rail.is-down .cs-trends-rail-fill, .cs-trends-rail.is-miss .cs-trends-rail-fill { background: var(--cs-ink-faint, var(--text-muted)); }
  .cs-trends-rail-base { position: absolute; top: 1px; bottom: 1px; width: 2px; margin-left: -1px; border-radius: 1px; background: var(--cs-ink-soft, var(--text-muted)); opacity: .7; }
  .cs-trends-rail-mark { position: absolute; top: 50%; width: 10px; height: 10px; margin-left: -5px; transform: translateY(-50%); border-radius: 50%; background: var(--cs-pos); box-shadow: 0 0 0 2px var(--cs-surface, var(--card)); }
  .cs-trends-rail.is-down .cs-trends-rail-mark, .cs-trends-rail.is-miss .cs-trends-rail-mark { background: var(--cs-ink-faint, var(--text-muted)); }
  .cs-trends-agewrap { margin: 0 0 18px; }
  .cs-trends-ages { margin: 0; padding: 36px 10px 8px; border: 1px solid var(--cs-line, var(--border)); border-radius: 14px; background: var(--cs-surface, var(--card)); overflow-x: auto; }
  .cs-trends-ages-plot { position: relative; display: flex; align-items: flex-end; gap: 4px; height: 92px; }
  .cs-trends-ages-base { position: absolute; left: 0; right: 0; height: 0; border-top: 1px dashed color-mix(in srgb, var(--cs-ink, #0f172a) 32%, transparent); pointer-events: none; z-index: 1; }
  .cs-trends-age { flex: 1 0 18px; min-width: 18px; height: 100%; display: flex; align-items: flex-end; justify-content: center; position: relative; z-index: 2; margin: 0; padding: 0; border: 0; background: transparent; cursor: pointer; font: inherit; }
  .cs-trends-age-bar { width: 100%; max-width: 16px; border-radius: 4px 4px 0 0; background: color-mix(in srgb, var(--cs-pos) 45%, var(--cs-ink-faint, #94a3b8)); pointer-events: none; }
  .cs-trends-age.is-prime .cs-trends-age-bar { background: var(--cs-pos); }
  .cs-trends-age-tip { display: none; position: absolute; left: 50%; bottom: calc(100% + 6px); transform: translateX(-50%); z-index: 5; white-space: nowrap; font-family: var(--cs-mono); font-size: 10.5px; font-weight: 700; color: var(--cs-surface, var(--card)); background: var(--cs-ink, var(--text)); border-radius: 7px; padding: 4px 7px; box-shadow: 0 6px 16px color-mix(in srgb, #000 22%, transparent); pointer-events: none; }
  .cs-trends-age:hover .cs-trends-age-tip, .cs-trends-age:focus-visible .cs-trends-age-tip, .cs-trends-age.is-open .cs-trends-age-tip { display: block; }
  .cs-trends-ages-axis { display: flex; gap: 4px; margin-top: 4px; }
  .cs-trends-ages-axis span { flex: 1 0 18px; min-width: 18px; text-align: center; font-family: var(--cs-mono); font-size: 9px; color: var(--cs-ink-faint, var(--text-muted)); }
  .cs-trends-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 268px), 1fr)); gap: 14px; align-items: stretch; }
  .cs-trends-card { display: flex; flex-direction: column; height: 100%; background: var(--cs-surface, var(--card)); border: 1px solid var(--cs-line, var(--border)); border-radius: 14px; padding: 14px 14px 12px; box-shadow: inset 3px 0 0 var(--cs-pos); }
  .cs-trends-card > summary { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; cursor: pointer; list-style: none; }
  .cs-trends-card > summary::-webkit-details-marker { display: none; }
  .cs-trends-card h3 { font-size: 10.5px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; color: var(--cs-ink-soft, var(--text-muted)); margin: 0; }
  .cs-trends-card-peek { display: none; font-family: var(--cs-mono); font-size: 11px; font-weight: 700; color: var(--cs-ink-soft, var(--text-muted)); white-space: nowrap; }
  .cs-trends-card .cs-hist-note { margin: 8px 0 10px; }
  .cs-trends-card-rows { display: flex; flex-direction: column; gap: 10px; margin-top: 10px; }
  .cs-trends-srow-top { display: flex; align-items: baseline; gap: 8px; }
  .cs-trends-srow-label { flex: 1; min-width: 0; font-size: 13px; font-weight: 700; color: var(--cs-ink, var(--text)); }
  .cs-trends-srow-pct { font-size: 15px; font-weight: 800; font-variant-numeric: tabular-nums; color: var(--cs-ink, var(--text)); white-space: nowrap; }
  .cs-trends-srow-meta { font-family: var(--cs-mono); font-size: 11px; color: var(--cs-ink-soft, var(--text-muted)); margin-top: 3px; }
  .cs-trends-srow .cs-trends-rail { margin-top: 5px; }
  .cs-trends-srow.is-pick { display: block; width: 100%; text-align: left; font: inherit; color: inherit; background: transparent; border: 1px solid transparent; border-radius: 10px; padding: 6px 8px; margin: 0 -8px; cursor: pointer; }
  .cs-trends-srow.is-pick:hover { background: color-mix(in srgb, var(--cs-pos) 8%, transparent); }
  .cs-trends-srow.is-on { border-color: var(--cs-pos); background: var(--cs-pos-bg); }
  .cs-trends-minipcts { display: flex; gap: 8px; margin-top: 4px; font-family: var(--cs-mono); font-size: 10px; color: var(--cs-ink-faint, var(--text-muted)); }
  .cs-trends-minipcts .is-on { color: var(--cs-pos); font-weight: 800; }
  .cs-trends-scout { margin: 0; padding: 10px 12px; border: 1px solid var(--cs-line, var(--border)); border-radius: 12px; background: var(--cs-surface, var(--card)); min-height: 0; overflow: auto; }
  .cs-trends-scout.is-idle { max-height: none; overflow: visible; padding: 8px 10px; }
  .cs-trends-scout-chips { display: flex; flex-wrap: wrap; gap: 4px; margin: 0 0 6px; }
  .cs-trends-chip { font: inherit; font-size: 11px; font-weight: 700; cursor: pointer; border: 1px solid var(--cs-pos); color: var(--cs-pos); background: var(--cs-pos-bg); border-radius: 999px; padding: 2px 8px; }
  .cs-trends-chip.is-clear { border-color: var(--cs-line, var(--border)); color: var(--cs-ink-soft, var(--text-muted)); background: transparent; }
  .cs-trends-scout-list { display: grid; grid-template-columns: 1fr 1fr; gap: 2px 12px; }
  .cs-trends-player { display: flex; align-items: center; gap: 8px; width: 100%; text-align: left; font: inherit; cursor: pointer; border: 0; background: transparent; border-radius: 8px; padding: 4px 2px; color: var(--cs-ink, var(--text)); }
  .cs-trends-player:hover { background: color-mix(in srgb, var(--cs-pos) 8%, transparent); }
  .cs-trends-player.is-drafted { opacity: .45; }
  .cs-trends-player { align-items: flex-start; }
  .cs-trends-player-copy { display: flex; flex-direction: column; min-width: 0; flex: 1; gap: 0; }
  .cs-trends-player-n { font-weight: 700; font-size: 12.5px; }
  .cs-trends-player-adp { font-family: var(--cs-mono); font-size: 10.5px; color: var(--cs-ink-soft, var(--text-muted)); }
  .cs-trends-player-why { display: none; }
  .cs-trends-player-edge { font-size: 10.5px; color: var(--cs-ink, var(--text)); font-weight: 650; line-height: 1.3; }
  .cs-trends-callout.is-down .cs-trends-callout-pct span { color: var(--cs-ink-faint, var(--text-muted)); }
  .cs-trends-profile { margin: 0; padding: 10px 12px 11px; border: 1px solid color-mix(in srgb, var(--cs-pos) 26%, var(--cs-line, var(--border))); border-radius: 12px; background: var(--cs-surface, var(--card)); box-shadow: inset 3px 0 0 var(--cs-pos); min-height: 0; }
  .cs-trends-profile.is-idle { padding: 8px 10px; box-shadow: none; }
  .cs-trends-profile > .cs-trends-sec-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; margin: 0 0 8px; }
  .cs-trends-profile > .cs-trends-sec-head > div { min-width: 0; }
  .cs-trends-profile > .cs-trends-sec-head h3 { margin: 0 0 2px; }
  .cs-trends-profile > .cs-trends-sec-head p { font-size: 13.5px; font-weight: 800; color: var(--cs-ink, var(--text)); line-height: 1.25; }
  .cs-trends-profile-n { flex-shrink: 0; font-family: var(--cs-mono); font-size: 10.5px; font-weight: 700; color: var(--cs-ink-soft, var(--text-muted)); margin: 0; padding: 3px 8px; border-radius: 999px; background: color-mix(in srgb, var(--cs-ink, #0f172a) 6%, transparent); }
  .cs-trends-profile-stats { display: grid; gap: 8px; }
  .cs-trends-profile-tiers { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; margin: 0; }
  .cs-trends-profile-tier { min-width: 0; border: 1px solid var(--cs-line, var(--border)); border-radius: 10px; padding: 7px 8px 8px; background: color-mix(in srgb, var(--cs-ink, #0f172a) 3%, var(--cs-surface, var(--card))); }
  .cs-trends-profile-tier.is-on { border-color: var(--cs-pos); background: var(--cs-pos-bg); }
  .cs-trends-profile-k { font-family: var(--cs-mono); font-size: 9.5px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; color: var(--cs-ink-faint, var(--text-muted)); }
  .cs-trends-profile-v { font-size: 20px; font-weight: 800; font-variant-numeric: tabular-nums; line-height: 1.05; margin-top: 3px; color: var(--cs-ink, var(--text)); }
  .cs-trends-profile-tier.is-on .cs-trends-profile-v { color: var(--cs-pos); }
  .cs-trends-profile-ci { font-family: var(--cs-mono); font-size: 10px; color: var(--cs-ink-faint, var(--text-muted)); margin-top: 2px; }
  .cs-trends-profile-dl { display: grid; grid-template-columns: repeat(auto-fit, minmax(108px, 1fr)); gap: 6px; margin: 0; }
  .cs-trends-profile-dl > div { display: flex; flex-direction: column; gap: 2px; min-width: 0; border: 1px solid var(--cs-line, var(--border)); border-radius: 8px; padding: 6px 8px; background: color-mix(in srgb, var(--cs-ink, #0f172a) 2.5%, var(--cs-surface, var(--card))); }
  .cs-trends-profile-dl dt { color: var(--cs-ink-faint, var(--text-muted)); font-family: var(--cs-mono); font-size: 9.5px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
  .cs-trends-profile-dl dd { margin: 0; font-size: 13px; font-weight: 800; color: var(--cs-ink, var(--text)); font-variant-numeric: tabular-nums; }
  .cs-trends-profile-dl dd.is-up { color: var(--cs-good); }
  .cs-trends-profile-dl dd.is-down { color: var(--cs-ink-soft, var(--text-muted)); }
  .cs-trends-profile-dl > div.is-conf { border-color: color-mix(in srgb, var(--cs-pos) 35%, var(--cs-line, var(--border))); background: var(--cs-pos-bg); }
  .cs-trends-profile-dl > div.is-conf dd { color: var(--cs-pos); }
  .cs-trends-scout-more { font-size: 12.5px; color: var(--cs-ink-soft, var(--text-muted)); margin: 8px 0 0; }
  .cs-trends-scout-more button { font: inherit; font-size: 12.5px; font-weight: 800; cursor: pointer; border: 0; background: none; color: var(--cs-pos); padding: 0; }
  .cs-trends-conf { display: inline-flex; align-items: center; flex-shrink: 0; color: var(--cs-ink-faint, var(--text-muted)); }
  .cs-trends-conf i { width: 7px; height: 7px; border-radius: 50%; background: currentColor; display: inline-block; }
  .cs-trends-conf-low { color: var(--cs-ink-faint, var(--text-muted)); }
  .cs-trends-conf-moderate { color: var(--cs-amber); }
  .cs-trends-conf-good { color: var(--cs-accent); }
  .cs-trends-conf-strong { color: var(--cs-good); }
  .cs-trends-bar { height: 6px; background: color-mix(in srgb, var(--cs-ink, #0f172a) 10%, transparent); border-radius: 99px; margin: 0; overflow: hidden; }
  .cs-trends-bar > span { display: block; height: 100%; background: color-mix(in srgb, var(--cs-pos) 78%, var(--cs-ink, #0f172a)); border-radius: 99px; }
  .cs-trends-bar-miss > span { background: var(--cs-ink-faint, var(--text-muted)); }
  @media (max-width: 720px) {
    .cs-trends-callouts { grid-template-columns: 1fr 1fr; }
    .cs-trends-callout { grid-template-columns: 1fr auto; padding: 8px 10px; }
    .cs-trends-callout .cs-trends-rail { display: none; }
    .cs-trends-callout-v { font-size: 12px; }
    .cs-trends-summary { align-items: flex-start; }
    .cs-trends-lanes { margin: 0 0 8px; }
    .cs-trends-sticky { padding: 8px 10px 10px; gap: 8px; }
    .cs-trends-sticky.is-picked { grid-template-columns: 1fr; }
    .cs-trends-profile-v { font-size: 18px; }
    .cs-trends-lane-n { flex: 1 0 100%; margin: 2px 0 0; order: 2; }
    .cs-trends-sticky-toggle { order: 1; margin-left: auto; }
    .cs-trends-grid { grid-template-columns: 1fr; gap: 8px; }
    .cs-trends-scout-list { grid-template-columns: 1fr; }
    .cs-trends-card { height: auto; padding: 0; }
    .cs-trends-card > summary {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      align-items: start;
      gap: 4px;
      padding: 12px 14px;
    }
    .cs-trends-card h3 { min-width: 0; overflow-wrap: anywhere; }
    .cs-trends-card-peek {
      display: block;
      min-width: 0;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .cs-trends-card .cs-hist-note, .cs-trends-card-rows { padding: 0 14px 12px; margin-top: 0; }
    .cs-trends-card .cs-hist-note { margin: 0 0 8px; }
  }
  @media (min-width: 721px) {
    .cs-trends-card > summary { cursor: default; }
    .cs-trends-card-peek { display: none; }
  }
  .cs-tabs button.cs-hidden { display: none; }

  /* Verdict-first layout: lead with the top-12 rate, then per-bucket trends.
     The hero number is the one figure a drafter opened the card for; the
     tier stats, cohort line, and bucket trends supply the supporting story,
     and comps/profile tuck into a disclosure so the card reads in a glance. */
  .cs-hist-verdict { margin: 0 0 16px; }
  .cs-hist-banner { display: flex; align-items: center; gap: 16px; margin: 0 0 14px; padding: 15px 16px; border-radius: 14px; border: 1px solid color-mix(in srgb, var(--cs-pos) 32%, var(--cs-line, var(--border))); background: linear-gradient(180deg, var(--cs-pos-bg), color-mix(in srgb, var(--cs-pos) 4%, transparent)); }
  .cs-hist-banner-num { font-weight: 800; font-size: 52px; line-height: .82; letter-spacing: -.03em; color: var(--cs-pos); font-variant-numeric: tabular-nums; flex-shrink: 0; }
  .cs-hist-banner-num sup { font-size: 20px; font-weight: 800; color: var(--cs-ink-soft, var(--text-muted)); vertical-align: super; }
  .cs-hist-banner-cap { min-width: 0; }
  .cs-hist-banner-lead { font-size: 15px; font-weight: 800; line-height: 1.25; color: var(--cs-ink, var(--text)); }
  .cs-hist-banner-sub { font-size: 12.5px; color: var(--cs-ink-soft, var(--text-muted)); line-height: 1.4; margin-top: 3px; }
  .cs-hist-conf { display: inline-flex; align-items: center; gap: 6px; margin-top: 8px; font-family: var(--cs-mono); font-size: 10.5px; font-weight: 700; letter-spacing: .03em; color: var(--cs-pos); background: var(--cs-pos-bg); border-radius: 999px; padding: 3px 9px; }
  .cs-hist-conf i { width: 6px; height: 6px; border-radius: 50%; background: currentColor; display: inline-block; flex-shrink: 0; }
  .cs-hist-tiers { display: flex; gap: 8px; }
  .cs-hist-tier { flex: 1; border: 1px solid var(--cs-line, var(--border)); border-radius: 10px; padding: 9px 10px; background: color-mix(in srgb, var(--cs-ink, #0f172a) 3%, var(--cs-surface, var(--card))); }
  .cs-hist-tier-k { font-family: var(--cs-mono); font-size: 10px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; color: var(--cs-ink-faint, var(--text-muted)); }
  .cs-hist-tier-v { font-weight: 800; font-size: 20px; color: var(--cs-ink, var(--text)); margin-top: 3px; font-variant-numeric: tabular-nums; }
  .cs-hist-tier.lead { border-color: var(--cs-pos); background: var(--cs-pos-bg); }
  .cs-hist-tier.lead .cs-hist-tier-v { color: var(--cs-pos); }
  .cs-hist-cohort { font-family: var(--cs-mono); font-size: 11px; color: var(--cs-ink-soft, var(--text-muted)); margin: 13px 0 0; line-height: 1.5; }

  .cs-hist-tlist { display: flex; flex-direction: column; }
  .cs-hist-trow { padding: 10px 0; border-bottom: 1px dashed var(--cs-line, var(--border)); }
  .cs-hist-trow:first-child { padding-top: 2px; }
  .cs-hist-trow:last-child { border-bottom: 0; padding-bottom: 0; }
  .cs-hist-ttop { display: flex; align-items: baseline; gap: 10px; }
  .cs-hist-tcat { font-family: var(--cs-mono); font-size: 9px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--cs-ink-faint, var(--text-muted)); border: 1px solid var(--cs-line, var(--border)); border-radius: 5px; padding: 2px 5px; white-space: nowrap; flex-shrink: 0; min-width: 62px; text-align: center; }
  .cs-hist-tsent { flex: 1; min-width: 0; font-size: 12.5px; font-weight: 600; color: var(--cs-ink, var(--text)); line-height: 1.3; }
  .cs-hist-tpct { font-weight: 800; font-size: 15px; color: var(--cs-ink, var(--text)); font-variant-numeric: tabular-nums; white-space: nowrap; }
  .cs-hist-tbarline { display: flex; align-items: center; gap: 10px; margin-top: 7px; padding-left: 72px; }
  .cs-hist-tbar { flex: 1; height: 5px; border-radius: 3px; background: color-mix(in srgb, var(--cs-ink, #0f172a) 8%, transparent); overflow: hidden; }
  .cs-hist-tbar > i { display: block; height: 100%; background: var(--cs-pos); border-radius: 3px; }
  .cs-hist-tmeta { font-family: var(--cs-mono); font-size: 9.5px; color: var(--cs-ink-faint, var(--text-muted)); white-space: nowrap; }

  .cs-hist-more { margin-top: 16px; border-top: 1px solid var(--cs-line, var(--border)); padding-top: 12px; }
  .cs-hist-more > summary { cursor: pointer; list-style: none; font-family: var(--cs-mono); font-size: 11px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--cs-pos); display: flex; align-items: center; gap: 8px; }
  .cs-hist-more > summary::-webkit-details-marker { display: none; }
  .cs-hist-more > summary::after { content: "+"; margin-left: auto; font-size: 15px; line-height: 1; }
  .cs-hist-more[open] > summary::after { content: "\2013"; }
  .cs-hist-more:focus-within > summary { color: var(--cs-pos); }
  .cs-hist-more > summary:focus-visible { outline: 2px solid var(--cs-pos); outline-offset: 3px; border-radius: 4px; }
  .cs-hist-more-inner { padding-top: 14px; display: flex; flex-direction: column; gap: 16px; }
  .cs-hist-more-inner .cs-hist-sec { margin: 0; }

  .cs-pos-badge { font-family: var(--cs-mono); font-weight: 800; font-size: 11px; padding: 3px 7px; border-radius: 6px; flex-shrink: 0; }
  .cs-pos-QB { color: var(--cs-qb); background: var(--cs-qb-bg); } .cs-pos-RB { color: var(--cs-rb); background: var(--cs-rb-bg); }
  .cs-pos-WR { color: var(--cs-wr); background: var(--cs-wr-bg); } .cs-pos-TE { color: var(--cs-te); background: var(--cs-te-bg); }

  .cs-winpill { font-family: var(--cs-mono); font-size: 10px; font-weight: 800; padding: 2px 7px; border-radius: var(--radius-pill, 8px); }
  .win-asc { color: var(--cs-good); background: var(--cs-good-soft); }
  .win-prime { color: var(--cs-accent); background: var(--cs-accent-soft); }
  .win-now { color: var(--cs-amber); background: var(--cs-amber-soft); }
  .win-fade { color: var(--cs-ink-faint); background: var(--cs-surface-2); }

  .cs-wrap tr.cs-cliff td { padding: 0; border: 0; }
  .cs-cliffline { display: flex; align-items: center; gap: 10px; font-family: var(--cs-mono); font-size: 10px; letter-spacing: .08em; text-transform: uppercase; color: var(--cs-ink-faint); padding: 8px 14px 7px; background: var(--cs-surface-2); }
  .cs-cliffline::before, .cs-cliffline::after { content: ""; height: 1px; background: var(--cs-line); flex: 1; }
  /* Projected-pick windows for a selected snake slot. Distinct from tier
     cliffs so "where you pick" is obvious on screen and on paper. */
  .cs-wrap tr.cs-proj td { padding: 0; border: 0; }
  .cs-projline { display: flex; align-items: center; gap: 10px; font-family: var(--cs-mono); font-size: 11px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; color: var(--cs-accent); padding: 7px 14px 6px; background: var(--cs-accent-soft); }
  .cs-projline::before, .cs-projline::after { content: ""; height: 2px; background: var(--cs-accent); flex: 1; opacity: .55; }
  .cs-proj-ov { font-weight: 700; color: var(--cs-ink-soft); letter-spacing: .04em; }
  .cs-wrap tbody tr.cs-proj-row td { background: var(--cs-accent-soft); }
  .cs-wrap tbody tr.cs-proj-row:hover td { background: color-mix(in srgb, var(--cs-accent) 22%, var(--cs-surface)); }
  .cs-proj-mark { font-family: var(--cs-mono); font-size: 9.5px; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; color: var(--cs-accent); background: var(--cs-accent-soft); padding: 1px 6px; border-radius: 12px; margin-left: 6px; white-space: nowrap; }
  .cs-pgtier.cs-proj-bar { color: var(--cs-accent); background: var(--cs-accent-soft); border-top-color: color-mix(in srgb, var(--cs-accent) 40%, var(--cs-line)); border-bottom-color: color-mix(in srgb, var(--cs-accent) 40%, var(--cs-line)); }

  .cs-board.filteron tbody tr.cs-p[data-good="0"] { opacity: .32; }
  /* Live-draft: players already taken read as struck-through and dimmed; "Hide
     drafted" removes them entirely. */
  .cs-wrap tbody tr.cs-p.drafted td { opacity: .34; }
  .cs-wrap tbody tr.cs-p.drafted .cs-pname { text-decoration: line-through; }
  .cs-board.hidedrafted tbody tr.cs-p.drafted { display: none; }
  .cs-board.hidedrafted .cs-proj-taken { display: none; }
  .cs-pgc.drafted { opacity: .34; }
  .cs-pgc.drafted .cs-pgn { text-decoration: line-through; }
  .cs-board.hidedrafted .cs-pgc.drafted { display: none; }
  .cs-taken-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--cs-bad); display: inline-block; opacity: .6; }
  /* Roster-need shading: "Needs only" dims positions you have already filled. */
  .cs-board.needson tbody tr.cs-p[data-posfull="1"] { opacity: .3; }
  .cs-board.needson .cs-pgc[data-posfull="1"] { opacity: .3; }
  .cs-needs { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 14px 0 0; padding: 9px 14px; background: var(--cs-surface); border: 1px solid var(--cs-line); border-radius: 12px; font-size: 12.5px; }
  .cs-need-lbl { font-family: var(--cs-mono); font-size: 10px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--cs-ink-faint); }
  .cs-need { font-family: var(--cs-mono); font-size: 11px; font-weight: 800; padding: 3px 8px; border-radius: 7px; }
  .cs-need-open { color: var(--cs-good); background: var(--cs-good-soft); }
  .cs-need-full { color: var(--cs-ink-faint); background: var(--cs-surface-2); }
  .cs-need-hint { margin-left: auto; font-family: var(--cs-mono); font-size: 10px; color: var(--cs-ink-faint); }

  .cs-pgrid-scroll { overflow: auto; max-width: 100%; max-height: calc(100vh - 250px); background: var(--cs-surface); border: 1px solid var(--cs-line); border-radius: 14px; -webkit-overflow-scrolling: touch; touch-action: pan-x pan-y; }
  .cs-pgrid { min-width: 460px; }
  .cs-pgrid-head { display: grid; grid-template-columns: repeat(4, 1fr); background: var(--cs-ink); position: sticky; top: 0; z-index: 2; }
  .cs-pgrid-head > div { text-align: center; padding: 10px 6px; font-family: var(--cs-mono); font-size: 12px; font-weight: 800; letter-spacing: .06em; color: var(--cs-surface); }
  .cs-pgrow { display: grid; grid-template-columns: repeat(4, 1fr); }
  .cs-pgrow.alt { background: var(--cs-surface-2); }
  .cs-pgcell { padding: 7px 10px; border-right: 1px solid var(--cs-line); min-height: 40px; display: flex; flex-direction: column; justify-content: center; gap: 7px; }
  .cs-pgcell:last-child { border-right: 0; }
  .cs-pgc { display: block; cursor: pointer; border-radius: 6px; padding: 2px 4px; }
  .cs-pgc:hover { background: var(--cs-accent-soft); }
  .cs-pgn { font-size: 12.5px; font-weight: 600; line-height: 1.18; display: inline-flex; align-items: baseline; gap: 6px; }
  .cs-pgc .cs-pgv { font-family: var(--cs-mono); font-size: 10px; font-weight: 800; }
  .cs-c-QB .cs-pgn, .cs-pgrid-head > .cs-c-QB { color: var(--cs-qb); }
  .cs-c-RB .cs-pgn, .cs-pgrid-head > .cs-c-RB { color: var(--cs-rb); }
  .cs-c-WR .cs-pgn, .cs-pgrid-head > .cs-c-WR { color: var(--cs-wr); }
  .cs-c-TE .cs-pgn, .cs-pgrid-head > .cs-c-TE { color: var(--cs-te); }
  .cs-pgc.done .cs-pgn { text-decoration: line-through; opacity: .4; }
  .cs-board.filteron .cs-pgc[data-good="0"] { opacity: .3; }
  .cs-pgtier { display: flex; align-items: center; gap: 10px; font-family: var(--cs-mono); font-size: 10px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--cs-ink-faint); background: var(--cs-surface-2); padding: 7px 12px; border-top: 1px solid var(--cs-line-strong); border-bottom: 1px solid var(--cs-line); }
  .cs-pgtier .cs-sc { font-weight: 600; letter-spacing: 0; text-transform: none; color: var(--cs-ink-soft); }

  /* Filter bar: instant name search + position filter over the whole board. */
  .cs-filterbar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 14px 0 12px; }
  .cs-filterbar .cs-src, .cs-filterbar .csd-wrap { flex: 0 0 auto; min-width: 168px; }
  .cs-search { flex: 1 1 200px; min-width: 140px; padding: 8px 12px; border-radius: 10px; border: 1px solid var(--cs-line); background: var(--cs-surface); color: var(--cs-ink); font: inherit; font-size: 13px; outline: none; }
  .cs-search:focus { border-color: var(--cs-accent); }
  .cs-posf { display: inline-flex; gap: 6px; flex-wrap: wrap; }
  .cs-wrap .cs-posf button[data-pos="QB"][aria-pressed="true"] { border-color: var(--cs-qb); color: var(--cs-qb); background: var(--cs-qb-bg); }
  .cs-wrap .cs-posf button[data-pos="RB"][aria-pressed="true"] { border-color: var(--cs-rb); color: var(--cs-rb); background: var(--cs-rb-bg); }
  .cs-wrap .cs-posf button[data-pos="WR"][aria-pressed="true"] { border-color: var(--cs-wr); color: var(--cs-wr); background: var(--cs-wr-bg); }
  .cs-wrap .cs-posf button[data-pos="TE"][aria-pressed="true"] { border-color: var(--cs-te); color: var(--cs-te); background: var(--cs-te-bg); }
  .cs-pick { font-family: var(--cs-mono); font-weight: 800; font-size: 13px; text-align: center; }
  .cs-pick small { display: block; font-size: 9px; font-weight: 600; color: var(--cs-ink-faint); letter-spacing: .06em; }
  .cs-dtiers { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .cs-tchip { font-family: var(--cs-mono); font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 7px; display: inline-flex; align-items: center; gap: 6px; }
  .cs-tchip .cs-ex { font-family: inherit; color: var(--cs-ink-soft); font-weight: 600; }

  .cs-prose { background: var(--cs-surface); border: 1px solid var(--cs-line); border-radius: 14px; }
  .cs-rule { display: grid; grid-template-columns: 112px 1fr; gap: 16px; padding: 16px 20px; border-bottom: 1px solid var(--cs-line); }
  .cs-rule:last-child { border-bottom: 0; }
  .cs-k { font-family: var(--cs-mono); font-weight: 800; color: var(--cs-accent); font-size: 12px; letter-spacing: .04em; text-transform: uppercase; padding-top: 2px; }
  .cs-prose h3 { margin: 0 0 4px; font-size: 15px; color: var(--cs-ink); }
  .cs-prose p { margin: 0; color: var(--cs-ink-soft); font-size: 13.5px; line-height: 1.5; }

  .cs-hidden { display: none; }
  .cs-foot { margin-top: 22px; color: var(--cs-ink-faint); font-size: 12px; }
  @media (max-width: 640px) {
    .cs-wrap { width: 100%; max-width: 100%; padding: 0 8px calc(24px + env(safe-area-inset-bottom)); overflow-x: hidden; }
    .cs-top { flex-direction: column; gap: 12px; }
    /* Reset the desktop flex-basis. In a column it becomes height, which was
       creating a several-hundred-pixel blank gap above the controls. */
    .cs-top > :first-child { flex: 0 0 auto; min-width: 0; width: 100%; }
    .cs-controls { align-items: stretch; width: 100%; margin-top: 0; gap: 10px; }
    .cs-ctrl-row { justify-content: flex-start; }
    /* Mode and QB toggles share a clean two-column row, each filling its half. */
    .cs-ctrl-row:first-child { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .cs-ctrl-row:first-child .cs-cgroup { display: flex; }
    .cs-ctrl-row:first-child .cs-seg { flex: 1; }
    .cs-ctrl-row:first-child .cs-seg button { flex: 1; }
    .cs-scoring-row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; width: 100%; }
    .cs-scoring-row .cs-cgroup { display: flex; flex-direction: column; align-items: stretch; gap: 4px; min-width: 0; }
    .cs-scoring-row .cs-src, .cs-scoring-row .csd-wrap { width: 100%; min-width: 0; }
    /* Actions wrap inside the viewport. The ADP selector gets a full row while
       visible actions form balanced, thumb-friendly cells underneath. */
    .cs-ctrl-row.cs-actions-row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; width: 100%; }
    .cs-ctrl-row.cs-actions-row .cs-src, .cs-ctrl-row.cs-actions-row .csd-wrap { grid-column: 1 / -1; width: 100%; min-width: 0; }
    .cs-ctrl-row.cs-actions-row .cs-btn { min-width: 0; width: 100%; justify-content: center; white-space: normal; padding: 8px 6px; }
    /* Keep every primary signal on mobile. The table scrolls horizontally, as
       it did before Market vs ADP was added, rather than hiding VOR or Value. */
    .cs-wrap table { min-width: 980px; border-collapse: separate; border-spacing: 0; }
    .cs-wrap thead th.cs-rk, .cs-wrap tbody td.cs-rk {
      position: sticky; left: 0; z-index: 4; width: 42px; min-width: 42px; max-width: 42px;
      box-sizing: border-box; background: var(--cs-surface); padding-left: 8px; padding-right: 6px;
    }
    .cs-wrap thead th.cs-player, .cs-wrap tbody td.cs-player {
      position: sticky; left: 42px; z-index: 4; min-width: 132px; max-width: 148px;
      box-sizing: border-box; background: var(--cs-surface); padding-right: 8px;
      border-right: 1px solid var(--cs-line); box-shadow: 6px 0 7px -5px color-mix(in srgb, #000 18%, transparent);
    }
    .cs-wrap thead th.cs-rk, .cs-wrap thead th.cs-player { z-index: 6; top: 0; }
    .cs-wrap tbody tr.cs-p:hover td.cs-rk, .cs-wrap tbody tr.cs-p:hover td.cs-player { background: var(--cs-surface-2); }
    .cs-wrap tbody tr.done td.cs-rk, .cs-wrap tbody tr.done td.cs-player,
    .cs-wrap tbody tr.cs-muted td.cs-rk, .cs-wrap tbody tr.cs-muted td.cs-player { opacity: 1; }
    .cs-wrap tbody tr.done td.cs-rk, .cs-wrap tbody tr.done td.cs-player .cs-pname { opacity: .4; }
    .cs-tbl-scroll, .cs-pgrid-scroll { max-height: none; height: auto; }
    .cs-wrap thead th, .cs-wrap tbody td { padding-left: 6px; padding-right: 6px; }
    .cs-pcell { gap: 5px; min-width: 0; }
    /* Fill the sticky cell. Do not clip at an inner 108px while empty space
       sits unused (or the column bloats from table width distribution). */
    .cs-pname { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; max-width: none; }
    /* Keep the Proj Pick banner label in the visible viewport while the board
       scrolls horizontally. A full-table-width centered label would slide under
       the sticky Rk/Player columns and look like overlapping text. */
    .cs-wrap tr.cs-proj td { background: var(--cs-accent-soft); }
    .cs-projline {
      position: sticky; left: 0; z-index: 1;
      width: calc(100vw - 32px); max-width: 100%;
      box-sizing: border-box; justify-content: flex-start;
      padding-left: 12px; padding-right: 12px;
    }
    .cs-projline::before, .cs-projline::after { display: none; }
    .cs-wrap tbody tr.cs-proj-row td.cs-rk,
    .cs-wrap tbody tr.cs-proj-row td.cs-player {
      background: color-mix(in srgb, var(--cs-accent) 14%, var(--cs-surface));
    }
    /* Tabs scroll sideways rather than wrapping onto a second line. */
    .cs-tabs { flex-wrap: nowrap; overflow-x: auto; -webkit-overflow-scrolling: touch; scrollbar-width: none; }
    .cs-tabs::-webkit-scrollbar { display: none; }
    .cs-tabs button { flex: 0 0 auto; white-space: nowrap; padding: 11px 12px; }
    /* Format note drops to its own line, left aligned, not floated off to the side. */
    #csFmtNote { margin-left: 0; width: 100%; }
    /* Bigger edit-board tap targets so the grip/arrows/pin are thumb-friendly. */
    .cs-ovbtns { gap: 5px; }
    .cs-ovbtn { width: 34px; height: 34px; font-size: 14px; border-radius: 8px; }
    .cs-filterbar .cs-src, .cs-filterbar .csd-wrap { width: 100%; min-width: 0; }
    .cs-hist-modal { align-items: center; justify-content: center; padding: max(12px, env(safe-area-inset-top)) 10px max(12px, env(safe-area-inset-bottom)); }
    .cs-hist-card { width: 100%; max-width: 100%; max-height: min(88dvh, calc(100vh - env(safe-area-inset-top, 0px) - env(safe-area-inset-bottom, 0px) - 12px)); border-radius: 14px; margin: 0; padding: 14px 14px 16px; }
    .cs-hist-banner-num { font-size: 46px; }
    .cs-hist-hits { grid-template-columns: 1fr; }
    .cs-hist-hit:last-child:nth-child(odd) { grid-column: auto; }
    .cs-hist-hit { padding: 12px 14px 13px; }
    .cs-hist-hit-top { align-items: flex-start; }
    .cs-hist-hit-top .cs-trends-conf { display: none; }
    .cs-hist-hit-label { font-size: 13px; }
    .cs-hist-hit-pct { font-size: 16px; }
    .cs-hist-hit .cs-trends-rail { display: none; }
    .cs-hist-tile-ex { margin-top: 8px; }
    .cs-hist-closest > summary { align-items: flex-start; }
    .cs-hist-ex-peek { flex: 1 1 100%; text-align: left; order: 3; }
    .cs-hist-tbarline { padding-left: 0; }
  }
  @media print {
    .cs-hist-modal, .cs-hist-btn { display: none !important; }
    .cs-controls, .cs-tabs, .cs-backlink, .cs-needs, .cs-filterbar, #csPrintBtn, #csValBtn, #csClearBtn, #csEditBtn, #csResetBoardBtn { display: none !important; }
    /* Never print the edit column even if edit mode is left on. */
    .cs-edit-th, .cs-board.editing .cs-edit-th, .cs-edit-cell, .cs-board.editing .cs-edit-cell { display: none !important; }
    /* Only the active tab prints; the JS leaves the other panels .hidden. */
    .cs-wrap { max-width: none; padding: 0; }
    /* Undo the on-screen height cap so the whole board flows onto pages. */
    .cs-tbl-scroll, .cs-pgrid-scroll { overflow: visible; border: 0; max-height: none; }
    .cs-wrap thead th { position: static; }
    .cs-wrap tbody td.cs-rk, .cs-wrap tbody td.cs-player { position: static; box-shadow: none; }
    .cs-trends-sticky { position: static; max-height: none; }
    .cs-trends-sticky.is-collapsed .cs-trends-sticky-body { display: contents; }
    .cs-trends-sticky-toggle { display: none !important; }
    /* Keep a tier heading with the rows under it, and don't split a row. */
    .cs-wrap tr.cs-cliff, .cs-wrap tr.cs-proj { break-before: auto; break-after: avoid; }
    .cs-wrap tbody tr { break-inside: avoid; }
    .cs-pgtier { break-after: avoid; }
    .cs-pgrow { break-inside: avoid; }
    /* Score bars and the accent fills don't render well on paper. */
    .cs-vorbar { display: none !important; }
    .cs-vorwrap { gap: 0; }
    .cs-legend { break-inside: avoid; }
  }
</style>

<div class="cs-wrap">
  <header class="cs-top">
    <div>
      <h1 id="csTitle">Redraft Cheat Sheet</h1>
      <p class="cs-sub" id="csSub">Ranked by value over replacement for your league scoring and roster.</p>
      <a class="cs-backlink" id="csBack" href="/draft">&larr; Open in Draft Room</a>
    </div>
    <div class="cs-controls">
      <div class="cs-ctrl-row">
        <div class="cs-cgroup"><span class="cs-clabel">Mode</span>
          <div class="cs-seg mode" id="csMode" role="group" aria-label="Mode">
            <button data-mode="redraft" aria-pressed="true">Redraft</button>
            <button data-mode="dynasty" aria-pressed="false">Dynasty</button>
          </div>
        </div>
        <div class="cs-cgroup"><span class="cs-clabel">QB</span>
          <div class="cs-seg" id="csQb" role="group" aria-label="QB format">
            <button data-qb="1QB" aria-pressed="true">1QB</button>
            <button data-qb="SF" aria-pressed="false">Superflex</button>
          </div>
        </div>
      </div>
      <div class="cs-ctrl-row cs-scoring-row">
        <div class="cs-cgroup cs-score"><span class="cs-clabel">PPR</span>
          <select class="cs-src" id="csPpr" aria-label="Reception scoring" title="Projected PPG uses this reception scoring (full, half, or standard).">
            <option value="1" selected>Full PPR</option>
            <option value="0.5">Half PPR</option>
            <option value="0">Standard</option>
          </select>
        </div>
        <div class="cs-cgroup cs-score"><span class="cs-clabel">TE Premium</span>
          <select class="cs-src" id="csTep" aria-label="Tight end premium" title="Projected PPG for tight ends includes this TE premium.">
            <option value="0" selected>None</option>
            <option value="0.5">+0.5 PPR</option>
            <option value="1">+1.0 PPR</option>
          </select>
        </div>
        <div class="cs-cgroup cs-score"><span class="cs-clabel">Passing TDs</span>
          <select class="cs-src" id="csPassTd" aria-label="Points per passing touchdown" title="Adjusts quarterback projected PPG">
            <option value="4" selected>4 points</option>
            <option value="6">6 points</option>
          </select>
        </div>
      </div>
      <div class="cs-ctrl-row cs-actions-row">
        <select class="cs-src" id="csAdpSrc" aria-label="ADP source" style="display:none;"></select>
        <button class="cs-btn" id="csNeedsBtn" aria-pressed="false" style="display:none;">Needs only</button>
        <button class="cs-btn" id="csHideDrafted" aria-pressed="false" style="display:none;">Hide drafted</button>
        <button class="cs-btn" id="csConnectLive" style="display:none;">Connect live draft</button>
        <button class="cs-btn" id="csEditBtn" aria-pressed="false" style="display:none;" title="Make this your board: reorder, pin or mute players (Pro)">Edit board</button>
        <button class="cs-btn" id="csValBtn" aria-pressed="false">Values only</button>
        <button class="cs-btn" id="csCsvBtn">CSV</button>
        <button class="cs-btn" id="csPrintBtn">Print</button>
        <button class="cs-btn cs-btn-reset" id="csClearBtn" style="display:none;">Clear marks</button>
        <button class="cs-btn cs-btn-reset" id="csResetBoardBtn" style="display:none;">Reset board</button>
      </div>
    </div>
  </header>

  <nav class="cs-tabs" role="tablist">
    <button role="tab" aria-selected="true" data-tab="board">Big Board</button>
    <button role="tab" aria-selected="false" data-tab="pos">By Position</button>
    <button role="tab" aria-selected="false" data-tab="trends" class="cs-hidden">Trends</button>
    <button role="tab" aria-selected="false" data-tab="logic">The Logic</button>
  </nav>

  <div class="cs-needs" id="csNeeds" style="display:none;"></div>
  <div class="cs-legend" id="csLegend"></div>

  <div class="cs-filterbar" id="csFilterbar">
    <input type="search" class="cs-search" id="csSearch" placeholder="Search players&hellip;" autocomplete="off" aria-label="Search players">
    <div class="otc-day-filters cs-posf" id="csPosF" role="group" aria-label="Filter by position">
      <button type="button" class="otc-day-filter" data-pos="ALL" aria-pressed="true">All</button>
      <button type="button" class="otc-day-filter" data-pos="QB" aria-pressed="false">QB</button>
      <button type="button" class="otc-day-filter" data-pos="RB" aria-pressed="false">RB</button>
      <button type="button" class="otc-day-filter" data-pos="WR" aria-pressed="false">WR</button>
      <button type="button" class="otc-day-filter" data-pos="TE" aria-pressed="false">TE</button>
    </div>
    <select class="cs-src" id="csPickSlot" aria-label="Projected pick slot"></select>
  </div>

  <section class="cs-board" id="cs-panel-board">
    <div class="cs-tbl-scroll"><table><thead id="csBoardHead"></thead><tbody id="csBoardBody"></tbody></table></div>
    <p class="cs-foot" id="csBoardFoot"></p>
  </section>

  <section class="cs-board cs-hidden" id="cs-panel-pos">
    <div class="cs-pgrid-scroll"><div class="cs-pgrid" id="csPosGrid"></div></div>
    <p class="cs-foot" id="csPosFoot"></p>
  </section>

  <section class="cs-hidden" id="cs-panel-trends">
    <div class="cs-trends" id="csTrends">
      <p class="cs-trends-lede">Historical finish rates by bucket.</p>
    </div>
  </section>

  <section class="cs-hidden" id="cs-panel-logic">
    <div class="cs-prose">
      <div class="cs-rule"><span class="cs-k">VOR</span><div><h3>Ranked by value over replacement</h3><p>The board is ordered by VOR: a player's value minus the value of the last startable player at his position in your league. Each position is measured against its own replacement, so QB, RB, WR and TE compare fairly on one board instead of by raw points. It is the honest cross-position value, which is what a draft board should sort on. Click a column header (ADP, Value, Proj PPG, Sched Rk, Off Rk, and the rest) to reorder the Big Board without changing that model ranking; Rk stays the VOR rank, and By Position stays on VOR order.</p></div></div>
      <div class="cs-rule"><span class="cs-k">Recommendation</span><div><h3>Live context without reordering the sheet</h3><p>Open the cheat sheet from an active Draft Room and each available player can show the room's current REC rank, including roster fit, remaining slots and expected availability. VOR still controls the cheat-sheet order, so the printable board stays stable. Pick Score remains the separate player-and-price grade in Draft Room.</p></div></div>
      <div class="cs-rule"><span class="cs-k">Roster</span><div><h3>Your league sets the replacement line</h3><p>Replacement level comes from your roster slots and league size, the same starter counts the Draft Room uses. Superflex moves that line: up to twice as many QBs start, so the replacement QB is far weaker and every startable QB climbs. Nothing is added by hand, the baseline simply moves.</p></div></div>
      <div class="cs-rule"><span class="cs-k">Tiers</span><div><h3>Tiers are value cliffs</h3><p>Players group where the drop-off is small inside the group and large to the next. Inside a tier, order barely matters, so take need or the falling price. Do not reach across a cliff.</p></div></div>
      <div class="cs-rule"><span class="cs-k">Value</span><div><h3>Where "above ADP" comes from</h3><p>Our rank is this VOR board. ADP is the consensus average draft position from real drafts. Value is ADP minus our rank. A green plus means the room lets him fall later than he is worth, so wait a beat and take him. A red minus means he goes early.</p></div></div>
      <div class="cs-rule"><span class="cs-k">Proj Pick</span><div><h3>Your snake slot on this board</h3><p>Choose a draft slot to draw labeled lines at each of that seat's snake-draft picks — Proj Pick 1.05, 2.08, and so on. The player under each line is who this ranking would take there. Lines follow the displayed order, including any custom-board moves, and they print with the sheet.</p></div></div>
      <div class="cs-rule"><span class="cs-k">Scoring</span><div><h3>Same settings as Draft Room setup</h3><p>PPR (full / half / standard), TE premium, and passing-TD points match the Draft Room Format step. They rescale projected PPG and TE roster targets the same way the room does. Opened from a live or mock draft, the sheet inherits that room's scoring.</p></div></div>
      <div class="cs-rule"><span class="cs-k">Proj PPG</span><div><h3>Expected weekly scoring</h3><p>Projected PPG is the player's upcoming-season fantasy points per game from Sleeper, adjusted for your PPR, TE premium, and passing-TD settings — the same projection pool the Draft Room uses. Players Sleeper does not project show a dash rather than last-season actuals.</p></div></div>
      <div class="cs-rule"><span class="cs-k">Schedule</span><div><h3>Full-season matchup context</h3><p>Schedule Rank compares each player's position-specific matchups across fantasy Weeks 1-17. Rank 1 is the easiest schedule. It is useful context for close calls inside a tier, but it does not change the stable VOR order.</p></div></div>
      <div class="cs-rule"><span class="cs-k">Offense</span><div><h3>Projected team offense</h3><p>Off Rk is the player's current NFL team's season-long projected offense, ranked 1-32 from implied totals (spread and total on regular-season games). Rank 1 is the strongest projected offense. It is the same number Hist uses for projected-offense buckets. Context for close calls, not a ranking input.</p></div></div>
      <div class="cs-rule"><span class="cs-k">Hist</span><div><h3>Historical top-12 chance</h3><p>On the redraft Big Board, Hist is this player's historical chance of a top-12 season given career and situation. Green marks a strong cell, or when history beats that ADP round. Early ADP is a high bar (round-1 hit rates are often 60-90%), so stars like Chase or Bijan are not painted as misses just because the market bucket is hotter. The info panel compares players-like-this vs that ADP round. Trends stays available in dynasty with a 1QB redraft caveat. Hist does not change VOR or Pick Score.</p></div></div>
      <div class="cs-rule"><span class="cs-k">Live</span><div><h3>It knows your live draft</h3><p>Open the sheet from your league during a draft and players already taken are struck through automatically. REC badges show the current Draft Room view without changing the VOR board. Reopen the sheet after more picks to refresh those ranks, or use Connect live draft to keep drafted-player status synchronized.</p></div></div>
      <div class="cs-rule"><span class="cs-k">Dynasty</span><div><h3>Dynasty values the window, not just this year</h3><p>Dynasty mode ranks on dynasty value, which already weights youth and multi-year outlook, and swaps in Age and a career-window tag in place of ADP, because you are drafting the next several seasons.</p></div></div>
    </div>
  </section>

  <p class="cs-foot">Computed for your league's scoring, roster and format from the same projections and values the Draft Room uses. Tap a player to cross him off; use the Hist info button for historical trends. Print for a paper copy.</p>
</div>
<div class="cs-hist-modal" id="csHistModal" role="dialog" aria-modal="true" aria-labelledby="csHistTitle">
  <div class="cs-hist-card">
    <div class="cs-hist-head">
      <div>
        <h2><span id="csHistPos" class="cs-pos-badge" hidden></span><span id="csHistTitle">History</span></h2>
        <p class="cs-hist-sub" id="csHistSub">Historical chance for this career and situation.</p>
      </div>
      <button type="button" class="cs-hist-close" id="csHistClose">Close</button>
    </div>
    <div id="csHistBody"></div>
  </div>
</div>
"""
