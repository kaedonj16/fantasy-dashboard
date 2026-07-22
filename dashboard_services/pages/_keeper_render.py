"""HTML template for the Keeper Assistant page.

Kept separate from keeper_page.py so the data-assembly logic stays readable. All
colors come from the app's CSS custom properties (var(--card), var(--accent),
etc.) so the page inherits light/dark theming for free; only keeper-specific
layout lives in the scoped .kpr- block below.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_STATIC = Path(__file__).resolve().parents[2] / "static"


def _asset_v(name: str) -> str:
    try:
        return hashlib.md5((_STATIC / name).read_bytes()).hexdigest()[:10]
    except OSError:
        return "0"


def render_keeper_html(seed: dict) -> str:
    seed_json = json.dumps(seed, separators=(",", ":"))
    kjs_v = _asset_v("keeper.js")
    draft_btn = (
        '<button type="button" id="kpr-to-draft" class="kpr-draft-btn">'
        '<i class="fa-solid fa-arrow-right-to-bracket" aria-hidden="true"></i> Open in Draft Room</button>'
        if seed.get("draftUrl") else ""
    )
    auto = seed.get("autoDraft")
    auto_badge = (
        '<span class="kpr-auto"><i class="fa-solid fa-circle-check" aria-hidden="true"></i> '
        'Draft rounds auto-detected</span>'
        if auto else
        '<span class="kpr-auto kpr-auto-off"><i class="fa-solid fa-circle-info" aria-hidden="true"></i> '
        'Set each player’s keeper round below</span>'
    )
    return f"""
<style>
  .kpr-wrap{{max-width:960px;margin:0 auto;}}
  .kpr-draft-btn{{display:inline-flex;align-items:center;gap:7px;font:inherit;font-size:13px;font-weight:700;
    color:#fff;background:var(--accent);border:0;border-radius:10px;padding:9px 15px;cursor:pointer;white-space:nowrap;}}
  .kpr-draft-btn:hover{{filter:brightness(1.06);}}
  .kpr-cfg{{display:flex;align-items:center;gap:10px 18px;flex-wrap:wrap;padding:14px 16px;
    border-bottom:1px solid var(--border);}}
  .kpr-auto{{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:700;
    letter-spacing:.03em;color:var(--win,#15803d);background:color-mix(in srgb,var(--win,#15803d) 14%,transparent);
    padding:4px 10px;border-radius:999px;}}
  .kpr-auto-off{{color:var(--text-muted);background:color-mix(in srgb,var(--text-muted) 14%,transparent);}}
  .kpr-rule{{display:flex;flex-direction:column;gap:2px;}}
  .kpr-rule label{{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--text-muted);}}
  .kpr-rule select,.kpr-rule input{{font:inherit;font-size:13px;font-weight:600;color:var(--text);
    background:var(--card-bg,var(--card));border:1px solid var(--border);border-radius:8px;padding:5px 8px;}}
  .kpr-views{{margin-left:auto;display:inline-flex;background:var(--bg-alt,var(--card-soft));border:1px solid var(--border);
    border-radius:10px;padding:3px;gap:2px;}}
  .kpr-views button{{font:inherit;font-size:12.5px;font-weight:600;color:var(--text-muted);background:transparent;
    border:0;border-radius:7px;padding:6px 12px;cursor:pointer;}}
  .kpr-views button[aria-selected="true"]{{background:var(--card);color:var(--text);box-shadow:0 1px 2px rgba(0,0,0,.08);}}

  .kpr-opt{{padding:16px;}}
  .kpr-opt-head{{display:flex;align-items:center;gap:16px 22px;flex-wrap:wrap;margin-bottom:14px;}}
  .kpr-limit{{display:flex;align-items:center;gap:12px;}}
  .kpr-limit label{{font-size:12px;color:var(--text-muted);font-weight:600;}}
  .kpr-limit .kpr-pill{{display:inline-flex;align-items:center;justify-content:center;background:var(--accent);
    color:#fff;font-weight:800;font-size:14px;border-radius:999px;padding:5px 13px;min-width:52px;}}
  .kpr-limit input[type=range]{{-webkit-appearance:none;appearance:none;width:150px;height:6px;border-radius:5px;
    background:var(--border);outline:none;}}
  .kpr-limit input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;width:18px;height:18px;border-radius:50%;
    background:var(--card);border:2px solid var(--accent);cursor:pointer;}}
  .kpr-limit input[type=range]::-moz-range-thumb{{width:18px;height:18px;border-radius:50%;background:var(--card);
    border:2px solid var(--accent);cursor:pointer;}}
  .kpr-total{{margin-left:auto;text-align:right;}}
  .kpr-total .l{{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--text-muted);}}
  .kpr-total .n{{font-size:20px;font-weight:800;color:var(--win,#15803d);font-variant-numeric:tabular-nums;}}

  .kpr-list{{display:flex;flex-direction:column;gap:8px;}}
  .kpr-row{{display:grid;grid-template-columns:24px 1fr auto auto;gap:12px;align-items:center;
    padding:11px 13px;border:1px solid var(--border);border-radius:11px;background:var(--card-soft,var(--card));}}
  .kpr-row.keep{{background:color-mix(in srgb,var(--win,#15803d) 12%,transparent);
    border-color:color-mix(in srgb,var(--win,#15803d) 40%,var(--border));}}
  .kpr-row.cut{{opacity:.6;}}
  .kpr-chk{{width:20px;height:20px;border-radius:6px;border:2px solid var(--border);display:grid;place-items:center;
    color:transparent;font-size:12px;}}
  .kpr-row.keep .kpr-chk{{background:var(--win,#15803d);border-color:var(--win,#15803d);color:#fff;}}
  .kpr-nm{{font-weight:650;color:var(--text);}}
  .kpr-sub{{font-size:11.5px;color:var(--text-muted);}}
  .kpr-mid{{font-size:12px;color:var(--text-muted);white-space:nowrap;}}
  .kpr-val{{font-weight:800;text-align:right;min-width:52px;font-variant-numeric:tabular-nums;}}
  .kpr-val.keep{{color:var(--win,#15803d);}} .kpr-val.toss{{color:var(--inj-q,#ca8a04);}} .kpr-val.pass{{color:var(--text-muted);}}

  .kpr-tbl-scroll{{overflow-x:auto;padding:4px 2px 8px;}}
  table.kpr-tbl{{width:100%;border-collapse:collapse;min-width:560px;}}
  .kpr-tbl thead th{{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--text-muted);
    text-align:left;font-weight:700;padding:10px 14px;border-bottom:1px solid var(--border);}}
  .kpr-tbl th.r,.kpr-tbl td.r{{text-align:right;}}
  .kpr-tbl tbody td{{padding:11px 14px;border-bottom:1px solid color-mix(in srgb,var(--border) 55%,transparent);
    vertical-align:middle;font-variant-numeric:tabular-nums;}}
  .kpr-tbl tbody tr:hover{{background:var(--card-soft,var(--bg-alt));}}
  .kpr-pos{{font-size:10px;font-weight:800;color:#fff;border-radius:5px;padding:2px 5px;margin-right:8px;}}
  .kpr-pos.QB{{background:#c026d3;}} .kpr-pos.RB{{background:#0d9488;}} .kpr-pos.WR{{background:#2563eb;}}
  .kpr-pos.TE{{background:#ea580c;}} .kpr-pos.K,.kpr-pos.DEF{{background:#64748b;}}
  .kpr-verdict{{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:750;
    padding:4px 10px;border-radius:999px;white-space:nowrap;}}
  .kpr-verdict .d{{width:6px;height:6px;border-radius:50%;background:currentColor;}}
  .kpr-verdict.keep{{color:var(--win,#15803d);background:color-mix(in srgb,var(--win,#15803d) 14%,transparent);}}
  .kpr-verdict.toss{{color:var(--inj-q,#ca8a04);background:color-mix(in srgb,var(--inj-q,#ca8a04) 15%,transparent);}}
  .kpr-verdict.pass{{color:var(--text-muted);background:color-mix(in srgb,var(--text-muted) 14%,transparent);}}
  .kpr-bar{{display:inline-block;width:70px;height:7px;border-radius:5px;background:var(--border);
    overflow:hidden;vertical-align:middle;margin-right:8px;}}
  .kpr-bar i{{display:block;height:100%;border-radius:5px;}}
  .kpr-empty{{padding:34px 16px;text-align:center;color:var(--text-muted);}}
  .kpr-note{{padding:12px 16px;border-top:1px solid var(--border);font-size:12px;color:var(--text-muted);}}
  .kpr-drnd{{width:58px;font:inherit;font-size:12px;font-weight:600;color:var(--text);
    background:var(--card);border:1px solid var(--border);border-radius:7px;padding:3px 6px;text-align:center;}}
  .kpr-yrs{{width:42px;font:inherit;font-size:11px;font-weight:600;color:var(--text);
    background:var(--card);border:1px solid var(--border);border-radius:6px;padding:1px 4px;text-align:center;}}
  .kpr-warn{{font-size:12.5px;color:var(--inj-q,#b45309);background:color-mix(in srgb,var(--inj-q,#b45309) 12%,transparent);
    border:1px solid color-mix(in srgb,var(--inj-q,#b45309) 32%,transparent);border-radius:9px;
    padding:8px 12px;margin-bottom:8px;}}
  /* Mobile: the optimizer is the primary view. Drop the redundant middle label,
     tighten rows, and compact the table so columns aren't chopped. */
  @media (max-width:560px){{
    .kpr-cfg{{gap:8px 12px;padding:12px;}}
    .kpr-opt{{padding:12px;}}
    .kpr-row{{grid-template-columns:20px 1fr auto;gap:9px;padding:10px 11px;}}
    .kpr-mid{{display:none;}}
    .kpr-nm{{font-size:14px;}}
    .kpr-draft-btn{{width:100%;justify-content:center;}}
    table.kpr-tbl{{min-width:470px;}}
    .kpr-tbl thead th,.kpr-tbl tbody td{{padding:9px 8px;}}
    .kpr-tbl .kpr-sub{{white-space:nowrap;}}
    .kpr-drnd{{width:44px;}}
    .kpr-yrs{{width:36px;}}
  }}
  @media (prefers-reduced-motion:reduce){{.kpr-row{{transition:none;}}}}
</style>

<div class="card central kpr-wrap">
  <div class="card-header" style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
    <div>
      <h2>Keeper Assistant</h2>
      <div style="font-size:14px;color:var(--text-muted);margin-top:4px;">
        Who’s worth keeping? Surplus = where a player drafts today minus what he costs to keep.
      </div>
    </div>
    {draft_btn}
  </div>

  <div class="card-body" style="padding-top:0;">
    <div class="kpr-cfg">
      {auto_badge}
      <div class="kpr-rule">
        <label for="kpr-cost">Keeper cost</label>
        <select id="kpr-cost">
          <option value="0" selected>Round drafted</option>
          <option value="-1">Round drafted − 1 (earlier)</option>
          <option value="1">Round drafted + 1 (later)</option>
        </select>
      </div>
      <div class="kpr-rule">
        <label for="kpr-esc">Escalation</label>
        <select id="kpr-esc">
          <option value="0">None</option>
          <option value="1" selected>+1 round / year kept</option>
          <option value="2">+2 rounds / year kept</option>
        </select>
      </div>
      <div class="kpr-rule">
        <label for="kpr-undr">Undrafted cost</label>
        <input id="kpr-undr" type="number" min="1" inputmode="numeric" placeholder="last">
      </div>
      <div class="kpr-views" role="tablist" aria-label="View">
        <button id="kpr-view-opt" role="tab" aria-selected="true">Optimizer</button>
        <button id="kpr-view-tbl" role="tab" aria-selected="false">Full table</button>
      </div>
    </div>

    <div class="kpr-opt" id="kpr-optimizer">
      <div class="kpr-opt-head">
        <div class="kpr-limit">
          <label for="kpr-lim">Keep up to</label>
          <input id="kpr-lim" type="range" min="0" max="6" value="2" step="1" aria-label="Keeper limit">
          <span class="kpr-pill" id="kpr-limn">2</span>
        </div>
        <div class="kpr-total">
          <div class="l">Total surplus</div>
          <div class="n" id="kpr-tot">+0 rd</div>
        </div>
      </div>
      <div class="kpr-list" id="kpr-list"></div>
    </div>

    <div class="kpr-tbl-scroll" id="kpr-table" hidden>
      <table class="kpr-tbl">
        <thead><tr>
          <th>Player</th><th class="r">Keeper cost</th><th class="r">Market (ADP)</th>
          <th class="r">Surplus</th><th class="r">Verdict</th>
        </tr></thead>
        <tbody id="kpr-tbody"></tbody>
      </table>
    </div>

    <div class="kpr-note">
      Surplus is in draft rounds, from BR’s redraft value model + market ADP. Keeper cost auto-fills from last
      season’s draft where available; edit any player’s round in the table. Two keepers that would cost the same
      round is a league-specific rule the tool flags rather than resolves.
    </div>
  </div>
</div>

<script type="application/json" id="kpr-seed">{seed_json}</script>
<script src="/static/keeper.js?v={kjs_v}" defer></script>
"""
