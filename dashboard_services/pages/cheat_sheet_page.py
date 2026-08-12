"""
Draft Cheat Sheet page.

A printable, pre-draft board that is the static sibling of the Draft Room: it
ranks the shared /api/league-players pool by value-over-replacement using the
same roster-derived replacement index (BRPickScore.starterCounts) the draft room
and the server pick-score use, so a player's cheat-sheet rank cannot contradict
their live Pick Score. The draft room adds the situational timing terms on top.

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
        "mode": "dynasty" if mode == "dynasty" else "redraft",
        "viewerUserId": str(viewer_user_id) if viewer_user_id else "",
        # Pro gate for live Sleeper sync and CSV export (the static board is free).
        "hasPremium": bool(has_premium),
        "draftUrl": (
            f"/{platform}/{int(season)}/{league_id}/draft"
            if _has_league else "/draft"
        ),
    }
    cfg_json = json.dumps(cfg)
    return (
        f"<script>window.__cheatCfg = {cfg_json};</script>\n"
        + _CHEAT_HTML
        + f'\n<script src="/static/pick_score.js?v={_static_v("pick_score.js")}" defer></script>\n'
        + f'\n<script src="/static/draft_board_core.js?v={_static_v("draft_board_core.js")}" defer></script>\n'
        + f'\n<script src="/static/cheat_sheet.js?v={_static_v("cheat_sheet.js")}" defer></script>\n'
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
        "<style>html,body{margin:0;background:var(--bg,#eef1f7);}"
        ".cs-wrap{padding-top:14px;}</style></head><body>"
        + body +
        "</body></html>"
    )


# Plain (non-f) string — safe to contain { } freely.
_CHEAT_HTML = r"""
<style>
  .cs-wrap {
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
    --cs-qb: #e0483f; --cs-qb-bg: rgba(224,72,63,.14);
    --cs-rb: #199a4d; --cs-rb-bg: rgba(25,154,77,.14);
    --cs-wr: #2f6df0; --cs-wr-bg: rgba(47,109,240,.14);
    --cs-te: #b5730b; --cs-te-bg: rgba(181,115,11,.16);
    --cs-bar: color-mix(in srgb, var(--accent, #38bdf8) 55%, transparent);
    --cs-bar-track: color-mix(in srgb, var(--text) 9%, transparent);
    --cs-mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
    max-width: 1120px; margin: 0 auto; padding: 6px 4px 60px; color: var(--cs-ink);
  }
  .cs-wrap * { box-sizing: border-box; }

  .cs-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; flex-wrap: wrap; }
  .cs-eyebrow { font-family: var(--cs-mono); font-size: 11px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--cs-accent); display: inline-flex; align-items: center; gap: 8px; }
  .cs-wrap h1 { font-size: clamp(23px, 4vw, 32px); line-height: 1.06; margin: 6px 0 4px; letter-spacing: -.02em; font-weight: 800; }
  .cs-sub { color: var(--cs-ink-soft); font-size: 14px; max-width: 64ch; margin: 0; line-height: 1.5; }
  .cs-backlink { font-size: 13px; font-weight: 700; color: var(--cs-accent); text-decoration: none; }
  .cs-backlink:hover { text-decoration: underline; }

  .cs-controls { display: flex; flex-direction: column; gap: 8px; align-items: flex-end; }
  .cs-ctrl-row { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; justify-content: flex-end; }
  .cs-cgroup { display: inline-flex; align-items: center; gap: 7px; }
  .cs-clabel { font-family: var(--cs-mono); font-size: 9.5px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--cs-ink-faint); }
  .cs-seg { display: inline-flex; padding: 3px; gap: 2px; background: var(--cs-surface-2); border: 1px solid var(--cs-line); border-radius: 10px; }
  .cs-seg button { font: inherit; font-size: 12px; font-weight: 700; cursor: pointer; border: 0; background: transparent; color: var(--cs-ink-soft); padding: 5px 11px; border-radius: 7px; }
  .cs-seg button[aria-pressed="true"] { background: var(--cs-accent); color: #fff; }
  .cs-seg.mode button[aria-pressed="true"] { background: var(--cs-ink); color: var(--cs-surface); }
  .cs-btn { font: inherit; font-size: 12px; font-weight: 700; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; background: var(--cs-surface); color: var(--cs-ink-soft); border: 1px solid var(--cs-line); border-radius: 9px; padding: 7px 11px; }
  .cs-btn:hover { border-color: var(--cs-accent); color: var(--cs-accent); }
  .cs-btn[aria-pressed="true"] { border-color: var(--cs-good); color: var(--cs-good); background: var(--cs-good-soft); }
  .cs-src { font: inherit; font-size: 12px; font-weight: 700; cursor: pointer; background: var(--cs-surface); color: var(--cs-ink-soft); border: 1px solid var(--cs-line); border-radius: 9px; padding: 7px 9px; }
  .cs-src:hover { border-color: var(--cs-accent); }
  /* Clear marks sits on its own line below the controls so it never shoves the
     row when it appears. The zero-height break forces the wrap. */
  .cs-break { flex-basis: 100%; height: 0; margin: 0; padding: 0; }
  .cs-btn-clear { margin-top: 4px; }

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
  /* Override state chip next to the name. */
  .cs-ovchip { font-family: var(--cs-mono); font-size: 10px; font-weight: 800; padding: 1px 6px; border-radius: 999px; margin-left: 8px; white-space: nowrap; }
  .cs-ovchip.bump { color: var(--cs-accent); background: var(--cs-accent-soft); }
  .cs-ovchip.pin { color: var(--cs-good); background: var(--cs-good-soft); }
  .cs-ovchip.mute { color: var(--cs-ink-faint); background: var(--cs-surface-2); }
  .cs-wrap tbody tr.cs-muted td { opacity: .5; }
  .cs-wrap tbody tr.cs-muted .cs-pname { color: var(--cs-ink-faint); }
  /* A subtle accent rail on any row you have personally moved. */
  .cs-wrap tbody tr.cs-ov td:first-child { box-shadow: inset 3px 0 0 var(--cs-accent); }

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

  .cs-tbl-scroll { overflow: auto; max-height: calc(100vh - 250px); background: var(--cs-surface); border: 1px solid var(--cs-line); border-radius: 14px; }
  .cs-wrap table { border-collapse: collapse; width: 100%; min-width: 640px; }
  .cs-wrap thead th { position: sticky; top: 0; z-index: 3; font-family: var(--cs-mono); font-size: 10.5px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--cs-ink-faint); text-align: right; padding: 12px 14px 9px; border-bottom: 1px solid var(--cs-line); background: var(--cs-surface); }
  .cs-wrap thead th.l { text-align: left; }
  .cs-wrap tbody td { padding: 8px 14px; border-bottom: 1px solid var(--cs-line); font-size: 13.5px; text-align: right; vertical-align: middle; }
  .cs-wrap tbody tr:last-child td { border-bottom: 0; }
  .cs-wrap tbody tr.cs-p { cursor: pointer; }
  .cs-wrap tbody tr.cs-p:hover td { background: var(--cs-surface-2); }
  .cs-wrap tbody tr.done td { opacity: .4; }
  .cs-wrap tbody tr.done .cs-pname { text-decoration: line-through; }
  .cs-empty { text-align: center !important; color: var(--cs-ink-faint); padding: 26px 14px !important; }
  .cs-rk { font-family: var(--cs-mono); font-weight: 800; color: var(--cs-ink); }
  .cs-pcell { text-align: left; display: flex; align-items: center; gap: 9px; }
  .cs-pname { font-weight: 600; white-space: nowrap; }
  .cs-num { font-family: var(--cs-mono); color: var(--cs-ink-soft); font-variant-numeric: tabular-nums; }
  .cs-posrk { font-family: var(--cs-mono); font-size: 11px; font-weight: 700; padding: 2px 6px; border-radius: 6px; }
  .cs-vorwrap { display: inline-flex; align-items: center; gap: 8px; justify-content: flex-end; }
  .cs-vorbar { width: 60px; height: 6px; border-radius: 999px; background: var(--cs-bar-track); overflow: hidden; flex-shrink: 0; }
  .cs-vorbar > i { display: block; height: 100%; background: var(--cs-bar); border-radius: 999px; }

  .cs-pos-badge { font-family: var(--cs-mono); font-weight: 800; font-size: 11px; padding: 3px 7px; border-radius: 6px; flex-shrink: 0; }
  .cs-pos-QB { color: var(--cs-qb); background: var(--cs-qb-bg); } .cs-pos-RB { color: var(--cs-rb); background: var(--cs-rb-bg); }
  .cs-pos-WR { color: var(--cs-wr); background: var(--cs-wr-bg); } .cs-pos-TE { color: var(--cs-te); background: var(--cs-te-bg); }

  .cs-winpill { font-family: var(--cs-mono); font-size: 10px; font-weight: 800; padding: 2px 7px; border-radius: 999px; }
  .win-asc { color: var(--cs-good); background: var(--cs-good-soft); }
  .win-prime { color: var(--cs-accent); background: var(--cs-accent-soft); }
  .win-now { color: var(--cs-amber); background: var(--cs-amber-soft); }
  .win-fade { color: var(--cs-ink-faint); background: var(--cs-surface-2); }

  .cs-wrap tr.cs-cliff td { padding: 0; border: 0; }
  .cs-cliffline { display: flex; align-items: center; gap: 10px; font-family: var(--cs-mono); font-size: 10px; letter-spacing: .08em; text-transform: uppercase; color: var(--cs-ink-faint); padding: 8px 14px 7px; background: var(--cs-surface-2); }
  .cs-cliffline::before, .cs-cliffline::after { content: ""; height: 1px; background: var(--cs-line); flex: 1; }

  .cs-board.filteron tbody tr.cs-p[data-good="0"] { opacity: .32; }
  /* Live-draft: players already taken read as struck-through and dimmed; "Hide
     drafted" removes them entirely. */
  .cs-wrap tbody tr.cs-p.drafted td { opacity: .34; }
  .cs-wrap tbody tr.cs-p.drafted .cs-pname { text-decoration: line-through; }
  .cs-board.hidedrafted tbody tr.cs-p.drafted { display: none; }
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

  .cs-pgrid-scroll { overflow: auto; max-height: calc(100vh - 250px); background: var(--cs-surface); border: 1px solid var(--cs-line); border-radius: 14px; }
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
  .cs-c-QB .cs-pgn { color: var(--cs-qb); } .cs-c-RB .cs-pgn { color: var(--cs-rb); }
  .cs-c-WR .cs-pgn { color: var(--cs-wr); } .cs-c-TE .cs-pgn { color: var(--cs-te); }
  .cs-pgc.done .cs-pgn { text-decoration: line-through; opacity: .4; }
  .cs-board.filteron .cs-pgc[data-good="0"] { opacity: .3; }
  .cs-pgtier { display: flex; align-items: center; gap: 10px; font-family: var(--cs-mono); font-size: 10px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--cs-ink-faint); background: var(--cs-surface-2); padding: 7px 12px; border-top: 1px solid var(--cs-line-strong); border-bottom: 1px solid var(--cs-line); }
  .cs-pgtier .cs-sc { font-weight: 600; letter-spacing: 0; text-transform: none; color: var(--cs-ink-soft); }

  /* Filter bar: instant name search + position filter over the whole board. */
  .cs-filterbar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 14px 0 12px; }
  .cs-search { flex: 1 1 200px; min-width: 140px; padding: 8px 12px; border-radius: 10px; border: 1px solid var(--cs-line); background: var(--cs-surface); color: var(--cs-ink); font: inherit; font-size: 13px; outline: none; }
  .cs-search:focus { border-color: var(--cs-accent); }
  .cs-posf { display: inline-flex; gap: 4px; flex-wrap: wrap; }
  .cs-posf button { font: inherit; font-size: 12px; font-weight: 700; cursor: pointer; border: 1px solid var(--cs-line); background: var(--cs-surface); color: var(--cs-ink-soft); padding: 8px 12px; border-radius: 9px; }
  .cs-posf button:hover { border-color: var(--cs-accent); color: var(--cs-accent); }
  .cs-posf button[aria-pressed="true"] { background: var(--cs-accent); color: #fff; border-color: var(--cs-accent); }
  .cs-pick { font-family: var(--cs-mono); font-weight: 800; font-size: 13px; text-align: center; }
  .cs-pick small { display: block; font-size: 9px; font-weight: 600; color: var(--cs-ink-faint); letter-spacing: .06em; }
  .cs-dtiers { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .cs-tchip { font-family: var(--cs-mono); font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 7px; display: inline-flex; align-items: center; gap: 6px; }
  .cs-tchip .cs-ex { font-family: inherit; color: var(--cs-ink-soft); font-weight: 600; }
  .cs-runflag { font-family: var(--cs-mono); font-size: 10px; color: var(--cs-accent); font-weight: 800; margin-left: 8px; white-space: nowrap; }

  .cs-prose { background: var(--cs-surface); border: 1px solid var(--cs-line); border-radius: 14px; }
  .cs-rule { display: grid; grid-template-columns: 112px 1fr; gap: 16px; padding: 16px 20px; border-bottom: 1px solid var(--cs-line); }
  .cs-rule:last-child { border-bottom: 0; }
  .cs-k { font-family: var(--cs-mono); font-weight: 800; color: var(--cs-accent); font-size: 12px; letter-spacing: .04em; text-transform: uppercase; padding-top: 2px; }
  .cs-prose h3 { margin: 0 0 4px; font-size: 15px; color: var(--cs-ink); }
  .cs-prose p { margin: 0; color: var(--cs-ink-soft); font-size: 13.5px; line-height: 1.5; }

  .cs-hidden { display: none; }
  .cs-foot { margin-top: 22px; color: var(--cs-ink-faint); font-size: 12px; }
  @media (max-width: 640px) {
    .cs-controls { align-items: stretch; width: 100%; margin-top: 14px; }
    .cs-ctrl-row { justify-content: flex-start; }
    /* Mode and QB toggles share a clean two-column row, each filling its half. */
    .cs-ctrl-row:first-child { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .cs-ctrl-row:first-child .cs-cgroup { display: flex; }
    .cs-ctrl-row:first-child .cs-seg { flex: 1; }
    .cs-ctrl-row:first-child .cs-seg button { flex: 1; }
    /* Action buttons stretch to fill the row evenly instead of straggling. */
    .cs-ctrl-row:last-child { display: flex; }
    .cs-ctrl-row:last-child .cs-btn, .cs-ctrl-row:last-child .cs-src { flex: 1 1 auto; justify-content: center; }
    /* Tabs scroll sideways rather than wrapping onto a second line. */
    .cs-tabs { flex-wrap: nowrap; overflow-x: auto; -webkit-overflow-scrolling: touch; scrollbar-width: none; }
    .cs-tabs::-webkit-scrollbar { display: none; }
    .cs-tabs button { flex: 0 0 auto; white-space: nowrap; padding: 11px 12px; }
    /* Format note drops to its own line, left aligned, not floated off to the side. */
    #csFmtNote { margin-left: 0; width: 100%; }
  }
  @media print {
    .cs-controls, .cs-tabs, .cs-backlink, .cs-needs, .cs-filterbar, #csPrintBtn, #csValBtn, #csClearBtn, #csEditBtn, #csResetBoardBtn { display: none !important; }
    /* Never print the edit column even if edit mode is left on. */
    .cs-edit-th, .cs-board.editing .cs-edit-th, .cs-edit-cell, .cs-board.editing .cs-edit-cell { display: none !important; }
    /* Only the active tab prints; the JS leaves the other panels .hidden. */
    .cs-wrap { max-width: none; padding: 0; }
    /* Undo the on-screen height cap so the whole board flows onto pages. */
    .cs-tbl-scroll, .cs-pgrid-scroll { overflow: visible; border: 0; max-height: none; }
    .cs-wrap thead th { position: static; }
    /* Keep a tier heading with the rows under it, and don't split a row. */
    .cs-wrap tr.cs-cliff { break-before: auto; break-after: avoid; }
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
      <div class="cs-ctrl-row">
        <select class="cs-src" id="csAdpSrc" aria-label="ADP source" style="display:none;"></select>
        <button class="cs-btn" id="csNeedsBtn" aria-pressed="false" style="display:none;">Needs only</button>
        <button class="cs-btn" id="csHideDrafted" aria-pressed="false" style="display:none;">Hide drafted</button>
        <button class="cs-btn" id="csEditBtn" aria-pressed="false" style="display:none;" title="Make this your board: bump, pin or mute players (Pro)">Edit board</button>
        <button class="cs-btn" id="csValBtn" aria-pressed="false">Values only</button>
        <button class="cs-btn" id="csCsvBtn">CSV</button>
        <button class="cs-btn" id="csPrintBtn">Print</button>
        <span class="cs-break" aria-hidden="true"></span>
        <button class="cs-btn cs-btn-clear" id="csClearBtn" style="display:none;">Clear marks</button>
        <button class="cs-btn cs-btn-clear" id="csResetBoardBtn" style="display:none;">Reset board</button>
      </div>
    </div>
  </header>

  <nav class="cs-tabs" role="tablist">
    <button role="tab" aria-selected="true" data-tab="board">Big Board</button>
    <button role="tab" aria-selected="false" data-tab="pos">By Position</button>
    <button role="tab" aria-selected="false" data-tab="logic">The Logic</button>
  </nav>

  <div class="cs-needs" id="csNeeds" style="display:none;"></div>
  <div class="cs-legend" id="csLegend"></div>

  <div class="cs-filterbar" id="csFilterbar">
    <input type="search" class="cs-search" id="csSearch" placeholder="Search players&hellip;" autocomplete="off" aria-label="Search players">
    <div class="cs-posf" id="csPosF" role="group" aria-label="Filter by position">
      <button type="button" data-pos="ALL" aria-pressed="true">All</button>
      <button type="button" data-pos="QB" aria-pressed="false">QB</button>
      <button type="button" data-pos="RB" aria-pressed="false">RB</button>
      <button type="button" data-pos="WR" aria-pressed="false">WR</button>
      <button type="button" data-pos="TE" aria-pressed="false">TE</button>
    </div>
  </div>

  <section class="cs-board" id="cs-panel-board">
    <div class="cs-tbl-scroll"><table><thead id="csBoardHead"></thead><tbody id="csBoardBody"></tbody></table></div>
    <p class="cs-foot" id="csBoardFoot"></p>
  </section>

  <section class="cs-board cs-hidden" id="cs-panel-pos">
    <div class="cs-pgrid-scroll"><div class="cs-pgrid" id="csPosGrid"></div></div>
    <p class="cs-foot" id="csPosFoot"></p>
  </section>

  <section class="cs-hidden" id="cs-panel-logic">
    <div class="cs-prose">
      <div class="cs-rule"><span class="cs-k">VOR</span><div><h3>Ranked by value over replacement</h3><p>The board is ordered by VOR: a player's value minus the value of the last startable player at his position in your league. Each position is measured against its own replacement, so QB, RB, WR and TE compare fairly on one board instead of by raw points. It is the honest cross-position value, which is what a draft board should sort on.</p></div></div>
      <div class="cs-rule"><span class="cs-k">Pick Score</span><div><h3>The board vs the live recommendation</h3><p>The Draft Room's Pick Score answers a different question: given your roster and your exact pick, who should you take right now. It layers roster need, ADP timing and survival onto this same value. The cheat sheet is the value board; the Draft Room is the on-the-clock recommender. Both read the same underlying value, so they never disagree on who is more valuable, only on fit for your next pick.</p></div></div>
      <div class="cs-rule"><span class="cs-k">Roster</span><div><h3>Your league sets the replacement line</h3><p>Replacement level comes from your roster slots and league size, the same starter counts the Draft Room uses. Superflex moves that line: up to twice as many QBs start, so the replacement QB is far weaker and every startable QB climbs. Nothing is added by hand, the baseline simply moves.</p></div></div>
      <div class="cs-rule"><span class="cs-k">Tiers</span><div><h3>Tiers are value cliffs</h3><p>Players group where the drop-off is small inside the group and large to the next. Inside a tier, order barely matters, so take need or the falling price. Do not reach across a cliff.</p></div></div>
      <div class="cs-rule"><span class="cs-k">Value</span><div><h3>Where "above ADP" comes from</h3><p>Our rank is this VOR board. ADP is the consensus average draft position from real drafts. Value is ADP minus our rank. A green plus means the room lets him fall later than he is worth, so wait a beat and take him. A red minus means he goes early.</p></div></div>
      <div class="cs-rule"><span class="cs-k">Live</span><div><h3>It knows your live draft</h3><p>Open the sheet from your league during a draft and players already taken are struck through automatically, or hidden entirely with Hide drafted, so the board always shows who is actually still available. Everything else is the same board you would see in the Draft Room, minus your pick slot and roster need, which only apply once you are on the clock.</p></div></div>
      <div class="cs-rule"><span class="cs-k">Dynasty</span><div><h3>Dynasty values the window, not just this year</h3><p>Dynasty mode ranks on dynasty value, which already weights youth and multi-year outlook, and swaps in Age and a career-window tag in place of ADP, because you are drafting the next several seasons.</p></div></div>
    </div>
  </section>

  <p class="cs-foot">Computed for your league's scoring, roster and format from the same projections and values the Draft Room uses. Tap a player to cross him off; use Print for a paper copy.</p>
</div>
"""
