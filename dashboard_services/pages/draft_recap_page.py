"""Draft Recap page.

Server-renders the performance-graded recap of a league's most recent
completed draft: every pick graded by how the player actually played (the
league's own scoring) versus his draft position in this league and his
market ADP. The computation lives in app._draft_recap_payload (reached
through the lazy shim below, the pattern routes/tool_pages_bp.py uses);
this module only renders the approved design: superlative cards, a team
leaderboard, curated Steals/Busts lists, best pick by round, and an
expandable full board (draft-board grid + headshot scatter chart).

No initials anywhere: a player without a headshot gets a plain
position-tinted disc with no letters.
"""
from __future__ import annotations

import html
import json
import logging

from utils.draft_recap import CORE_POSITIONS, POS_COLORS, rank_values, verdict_for

logger = logging.getLogger(__name__)


def _payload(platform, league_id, season):
    from app import _draft_recap_payload as _fn
    return _fn(platform, league_id, season)


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""), quote=True)


_POS_FALLBACK = "#64748b"


def _avatar(row, size=30) -> str:
    c = POS_COLORS.get(row.get("pos") or "", _POS_FALLBACK)
    hs = row.get("headshot") or ""
    img = (f'<img src="{_esc(hs)}" alt="" loading="lazy" onerror="this.remove()">'
           if hs else "")
    return (f'<span class="drc-av" style="--pc:{c};width:{size}px;height:{size}px">'
            f'{img}</span>')


def _pos_chip(pos) -> str:
    c = POS_COLORS.get(pos or "", _POS_FALLBACK)
    return f'<span class="drc-pos" style="--pc:{c}">{_esc(pos)}</span>'


def _delta_txt(d) -> str:
    if d is None:
        return '<span class="drc-na">-</span>'
    cls = "drc-up" if d > 0 else ("drc-dn" if d < 0 else "drc-flat")
    sign = "+" if d > 0 else ""
    return f'<span class="{cls}">{sign}{d}</span>'


def _finish_of(row, mode):
    return row.get("class_finish") if mode == "rookie" else row.get("finish_ppg")


_INJURED_COLOR = "#64748b"  # slate, matches utils.draft_recap VERDICT_COLORS


def _verdict_of(row):
    """(label, color) for a row, honoring the injury exemption: an exempt
    pick's delta is None (verdict_for would call it "No stats"), but its
    verdict is "Injured" and it must read injured everywhere it appears."""
    if row.get("injury_exempt") or row.get("verdict") == "Injured":
        return ("Injured", _INJURED_COLOR)
    return verdict_for(row.get("delta_pick"))


_CSS = """
.drc-title::after{display:none!important}
.drc-wrap{width:100%;max-width:1180px;margin:0 auto;padding:18px 14px 56px;box-sizing:border-box}
.drc-hero{background:#0b1f3a;border-radius:16px;padding:22px;margin-bottom:16px;position:relative;overflow:hidden}
.drc-hero::before{content:"";position:absolute;inset:0;background:radial-gradient(560px 220px at 85% -30%,rgba(56,189,248,.28),transparent 65%),radial-gradient(420px 200px at 8% 120%,rgba(56,189,248,.14),transparent 60%)}
.drc-hero-in{position:relative;display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:14px}
.drc-title{font-size:30px;font-weight:800;letter-spacing:-0.02em;color:#fff;margin:0;text-align:left}
.drc-sub{font-size:14px;color:#a9c3e2;margin:3px 0 0;text-align:left}
.drc-chips{display:flex;gap:6px;margin-top:12px;flex-wrap:wrap}
.drc-chip{font-size:11.5px;font-weight:700;color:#dbeafe;background:rgba(148,197,255,.14);border:1px solid rgba(148,197,255,.28);padding:3px 9px;border-radius:999px}
.drc-sec{display:flex;align-items:center;gap:12px;font-size:14px;font-weight:800;text-transform:uppercase;letter-spacing:.06em;color:var(--text);margin:26px 0 12px}
.drc-sec::after{content:"";flex:1;height:1px;background:var(--border)}
.drc-sups{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.drc-sup{background:var(--card);border:1px solid var(--border);border-top:3px solid var(--vc,#38bdf8);border-radius:12px;padding:12px 14px 14px;box-shadow:var(--shadow-sm,0 2px 8px rgba(15,23,42,.05))}
.drc-sup-k{font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted);margin-bottom:10px;display:flex;align-items:center;gap:7px}
.drc-sup-k::before{content:"";width:7px;height:7px;border-radius:50%;background:var(--vc,#38bdf8);flex-shrink:0}
.drc-sup-v{font-size:32px;font-weight:800;letter-spacing:-0.02em;color:var(--text);margin-top:10px}
.drc-sup-s{font-size:12.5px;color:var(--text-muted);margin-top:2px}
.drc-sup-team{font-size:17px;font-weight:800;color:var(--text);line-height:1.25;min-height:42px;display:flex;align-items:center}
.drc-card{background:var(--card);border:1px solid var(--border);border-radius:12px;box-shadow:var(--shadow-sm,0 2px 8px rgba(15,23,42,.05));overflow:hidden}
.drc-scroll{overflow-x:auto}
.drc-table{width:100%;border-collapse:collapse;font-size:13.5px;min-width:880px}
.drc-table th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--text-muted);font-weight:800;padding:10px;border-bottom:1px solid var(--border);white-space:nowrap}
.drc-table td{padding:8px 10px;border-bottom:1px solid var(--border);color:var(--text);vertical-align:middle;white-space:nowrap}
.drc-table tr:last-child td{border-bottom:0}
.drc-pick{font-weight:800;color:var(--text-muted)}
.drc-pcell{display:flex;align-items:center;gap:9px}
.drc-pmeta{display:flex;flex-direction:column;line-height:1.15;min-width:0}
.drc-pname{font-weight:700}
.drc-psub{font-size:11.5px;color:var(--text-muted)}
.drc-av{position:relative;border-radius:50%;background:color-mix(in srgb,var(--pc) 16%,transparent);display:inline-flex;align-items:center;justify-content:center;flex-shrink:0;overflow:hidden}
.drc-av img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;border-radius:50%}
.drc-pos{font-size:10.5px;font-weight:800;color:var(--pc);background:color-mix(in srgb,var(--pc) 13%,transparent);border:1px solid color-mix(in srgb,var(--pc) 35%,transparent);padding:1px 6px;border-radius:999px}
.drc-num{font-variant-numeric:tabular-nums}
.drc-up{color:#16a34a;font-weight:800}
.drc-dn{color:#dc2626;font-weight:800}
.drc-flat{color:var(--text-muted);font-weight:700}
.drc-na{color:var(--text-muted)}
.drc-verdict{font-size:11px;font-weight:800;color:var(--vc);background:color-mix(in srgb,var(--vc) 12%,transparent);border:1px solid color-mix(in srgb,var(--vc) 35%,transparent);padding:2px 8px;border-radius:999px;white-space:nowrap}
.drc-teamname{font-weight:700}
.drc-row1{background:color-mix(in srgb,#f59e0b 6%,transparent)}
.drc-rank1{font-weight:800;color:#b45309}
.drc-bar{position:relative;width:118px;height:8px;background:color-mix(in srgb,var(--text-muted) 14%,transparent);border-radius:99px;flex-shrink:0;display:inline-block}
.drc-bar::before{content:"";position:absolute;left:50%;top:-2px;width:1px;height:12px;background:color-mix(in srgb,var(--text-muted) 45%,transparent)}
.drc-bar i{position:absolute;top:0;height:8px;border-radius:99px;display:block}
.drc-bar .pos{left:50%;background:#22c55e}.drc-bar .neg{right:50%;background:#ef4444}
.drc-medcell{display:flex;align-items:center;gap:9px}
.drc-lists{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.drc-listcard{background:var(--card);border:1px solid var(--border);border-radius:12px;box-shadow:var(--shadow-sm,0 2px 8px rgba(15,23,42,.05));overflow:hidden}
.drc-listhead{font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted);padding:12px 14px 4px}
.drc-row{display:flex;align-items:center;gap:11px;padding:10px 14px;border-bottom:1px solid var(--border)}
.drc-row:last-child{border-bottom:0}
.drc-rk{width:22px;height:22px;border-radius:50%;background:color-mix(in srgb,var(--text-muted) 12%,transparent);display:inline-flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;color:var(--text-muted);flex-shrink:0}
.drc-row .drc-pcell{flex:1;min-width:0}
.drc-rstats{display:flex;flex-direction:column;text-align:right;line-height:1.2;font-weight:700;color:var(--text);font-size:13px}
.drc-rsub{font-size:11.5px;color:var(--text-muted);font-weight:600}
.drc-rdelta{display:flex;flex-direction:column;align-items:flex-end;gap:3px;min-width:74px}
.drc-big{font-size:19px;font-weight:800}
.drc-roundstrip{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.drc-roundcard{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px;box-shadow:var(--shadow-sm,0 2px 8px rgba(15,23,42,.05))}
.drc-roundcard .drc-sup-v{font-size:26px;margin-top:8px}
.drc-boardwrap{text-align:center;margin-top:22px}
.drc-boardbtn{font:inherit;font-size:13.5px;font-weight:800;color:var(--text);background:var(--card);border:1px solid var(--border);border-radius:999px;padding:10px 22px;cursor:pointer;box-shadow:var(--shadow-sm,0 2px 8px rgba(15,23,42,.05))}
.drc-boardbtn:hover{border-color:#38bdf8;color:#0284c7}
.drc-board{margin-top:14px;text-align:left}
.drc-seg{display:inline-flex;background:var(--card);border:1px solid var(--border);border-radius:999px;padding:3px;gap:2px}
.drc-seg button{border:0;background:transparent;padding:7px 14px;border-radius:999px;font:inherit;font-size:13px;font-weight:700;color:var(--text-muted);cursor:pointer}
.drc-bseg{margin-bottom:12px}
.drc-bseg button.on{background:var(--text);color:var(--card)}
.drc-gridwrap{overflow-x:auto;padding-bottom:4px}
.drc-grid{display:grid;gap:6px;min-width:640px}
.drc-ghead{font-size:11px;font-weight:800;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em;padding:4px 2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.drc-grow{display:flex;align-items:center;justify-content:center;font-weight:800;color:var(--text-muted);font-size:12px}
.drc-cell{border:1px solid var(--border);border-radius:10px;padding:7px 8px;min-height:66px;display:flex;flex-direction:column;gap:2px;text-align:left}
.drc-cell-empty{background:transparent;border-style:dashed;opacity:.35}
.drc-cell-top{display:flex;align-items:center;justify-content:space-between}
.drc-cell-pos{font-size:10px;font-weight:800;color:var(--text-muted)}
.drc-cell-name{font-size:12.5px;font-weight:700;color:var(--text);line-height:1.15;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.drc-cell-sub{font-size:11px;color:var(--text-muted)}
.drc-cell-delta{font-size:14px;font-weight:800;margin-top:auto}
.drc-chart{width:100%;height:auto;display:block}
.drc-chartcard{padding:10px 8px 4px;position:relative}
.drc-chart-lab{font-size:11.5px;font-weight:700;fill:var(--text)}
.drc-chart-tick{font-size:10.5px;font-weight:600;fill:var(--text-muted)}
.drc-chart-axis{font-size:11px;font-weight:700;fill:var(--text-muted)}
.drc-chart-note{font-size:12px;color:var(--text-muted);padding:2px 6px 10px}
.drc-dot{cursor:pointer}
.drc-dot:hover .drc-ring{stroke-width:5}
.drc-tip{position:absolute;width:238px;background:var(--card);border:1px solid var(--border);border-radius:12px;box-shadow:0 12px 32px rgba(15,23,42,.18);padding:10px 12px;z-index:6;pointer-events:none;text-align:left}
.drc-tip-head{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.drc-tip-head img{width:32px;height:32px;border-radius:50%;object-fit:cover;background:#f1f5f9}
.drc-tip-disc{width:32px;height:32px;border-radius:50%;background:#e2e8f0;display:inline-block}
.drc-tip-name{font-weight:800;font-size:13.5px;color:var(--text);line-height:1.1}
.drc-tip-sub{font-size:11px;color:var(--text-muted);font-weight:700}
.drc-tip-head .drc-verdict{margin-left:auto}
.drc-tip-row{font-size:12px;color:var(--text);padding:1.5px 0}
.drc-tip-dim{color:var(--text-muted)}
.drc-tip-deltas{display:flex;gap:14px;font-size:12px;color:var(--text-muted);border-top:1px solid var(--border);margin-top:6px;padding-top:6px}
.drc-empty{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:36px 18px;text-align:center;color:var(--text-muted);font-size:14px;line-height:1.55}
.drc-empty a{color:#0284c7;font-weight:700}
@media (max-width:760px){
  .drc-lists{grid-template-columns:1fr}
  .drc-rstats{display:none}
  .drc-sups{grid-template-columns:repeat(2,1fr)}
  .drc-roundstrip{grid-template-columns:1fr}
  .drc-title{font-size:24px}
  .drc-bar{display:none}
}
"""


def _hero(payload, league_id, season, platform):
    if payload and payload.get("status") == "ok":
        sub = f'{_esc(payload.get("league_name") or "")} · {_esc(payload.get("draft_label") or "")}'
        chips = ""
        if payload.get("through_week"):
            chips += f'<span class="drc-chip">Through Week {int(payload["through_week"])}</span>'
        chips += f'<span class="drc-chip">{_esc(payload.get("scoring_label") or "")}</span>'
        chips += f'<span class="drc-chip">{int(payload.get("picks_graded") or 0)} picks graded</span>'
    else:
        sub = "How every pick actually performed"
        chips = ""
    return (f'<div class="drc-hero"><div class="drc-hero-in"><div>'
            f'<h1 class="drc-title">Draft Recap</h1>'
            f'<p class="drc-sub">{sub}</p>'
            f'<div class="drc-chips">{chips}</div></div></div></div>')


def _empty(payload, league_id, season, platform, message=None):
    hist = (f"/{_esc(platform)}/{int(season)}/{_esc(league_id)}/draft/history"
            if league_id and season else "/draft/history")
    if message is None:
        if payload is not None and payload.get("status") == "unsupported":
            message = ("This draft was an auction, and auction recaps are not "
                       "supported yet.")
        elif payload is not None and payload.get("status") == "no_draft":
            message = ("No completed draft was found for this league and "
                       "season, so there is nothing to grade yet.")
        elif payload is not None and payload.get("status") == "ok":
            if not payload.get("through_week"):
                message = ("The season has no completed weeks yet, so no "
                           "pick can be graded. Check back after Week 1.")
            else:
                message = ("No drafted player has recorded a game yet, so "
                           "there is nothing to grade.")
        else:
            message = "Open one of your leagues to see its draft recap."
    body = (f'<div class="drc-wrap">{_hero(payload, league_id, season, platform)}'
            f'<div class="drc-empty">{_esc(message)} '
            f'<a href="{hist}">View Draft History</a></div></div>'
            f'<style>{_CSS}</style>')
    return body


def _sup_cards(payload):
    mode = payload.get("mode")
    sup = payload.get("superlatives") or {}
    steal, bust, vsadp = sup.get("steal"), sup.get("bust"), sup.get("best_vs_adp")
    best_team = sup.get("best_draft") or {}

    def _card(label, r, val, sub, vc):
        return (f'<div class="drc-sup" style="--vc:{vc}">'
                f'<div class="drc-sup-k">{_esc(label)}</div>'
                f'<div class="drc-pcell">{_avatar(r, 38)}<div class="drc-pmeta">'
                f'<span class="drc-pname">{_esc(r.get("name"))}</span>'
                f'<span class="drc-psub">{_esc(r.get("label"))} · {_esc(r.get("team"))}</span>'
                f'</div>{_pos_chip(r.get("pos"))}</div>'
                f'<div class="drc-sup-v">{_esc(val)}</div>'
                f'<div class="drc-sup-s">{_esc(sub)}</div></div>')

    out = []
    if steal:
        out.append(_card(
            "Biggest steal", steal, f'+{steal["delta_pick"]}',
            f'Drafted {steal["pos"]}{steal["draft_pos_rank"]}, finished '
            f'{steal["pos"]}{_finish_of(steal, mode)}',
            verdict_for(steal["delta_pick"])[1]))
    if bust:
        out.append(_card(
            "Biggest bust", bust, f'{bust["delta_pick"]}',
            f'Drafted {bust["pos"]}{bust["draft_pos_rank"]}, finished '
            f'{bust["pos"]}{_finish_of(bust, mode)}',
            "#dc2626"))
    if vsadp:
        out.append(_card(
            "Best vs ADP", vsadp, f'+{vsadp["delta_adp"]}',
            f'ADP {vsadp["pos"]}{vsadp["adp_pos_rank"]}, finished '
            f'{vsadp["pos"]}{_finish_of(vsadp, mode)}',
            "#16a34a"))
    out.append(
        f'<div class="drc-sup" style="--vc:#f59e0b"><div class="drc-sup-k">Best draft</div>'
        f'<div class="drc-sup-team">{_esc(best_team.get("team") or "")}</div>'
        f'<div class="drc-sup-s">Highest median finish vs draft position in the league</div></div>')
    return f'<div class="drc-sups">{"".join(out)}</div>'


def _team_table(payload):
    teams = payload.get("teams") or []
    maxabs = max((abs(t["median"]) for t in teams if t.get("median") is not None),
                 default=0) or 1
    rows = []
    for i, t in enumerate(teams, 1):
        avg = t["median"]
        cls = "drc-up" if avg > 0 else ("drc-dn" if avg < 0 else "drc-flat")
        w = round(abs(avg) / maxabs * 50, 1)
        if avg > 0:
            bar = f'<span class="drc-bar"><i class="pos" style="width:{w}%"></i></span>'
        elif avg < 0:
            bar = f'<span class="drc-bar"><i class="neg" style="width:{w}%"></i></span>'
        else:
            bar = '<span class="drc-bar"></span>'
        sign = "+" if avg > 0 else ""
        b, wst = t.get("best") or {}, t.get("worst") or {}
        _tr_cls = ' class="drc-row1"' if i == 1 else ""
        _rk_cls = " drc-rank1" if i == 1 else ""
        rows.append(
            f'<tr{_tr_cls}>'
            f'<td class="drc-pick{_rk_cls}">{i}</td>'
            f'<td class="drc-teamname">{_esc(t.get("team"))}</td>'
            f'<td class="drc-num"><span class="drc-medcell"><span class="{cls}">{sign}{avg:.1f}</span>{bar}</span></td>'
            f'<td class="drc-num">{t["draft_points"]:.1f}</td>'
            f'<td class="drc-num">{t["hits"]} of {t["graded"]}</td>'
            f'<td><div class="drc-pcell">{_avatar(b, 26)}<span>{_esc(b.get("name"))}</span>{_delta_txt(b.get("delta_pick"))}</div></td>'
            f'<td><div class="drc-pcell">{_avatar(wst, 26)}<span>{_esc(wst.get("name"))}</span>{_delta_txt(wst.get("delta_pick"))}</div></td>'
            f'</tr>')
    return ('<div class="drc-sec">Team leaderboard</div>'
            '<div class="drc-card"><div class="drc-scroll"><table class="drc-table">'
            '<thead><tr><th>#</th><th>Team</th><th>Median vs pick</th>'
            '<th>Draft pts</th><th>Hits</th><th>Best pick</th><th>Worst pick</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div></div>')


def _pick_row(i, r, mode):
    d = r["delta_pick"]
    v, vc = verdict_for(d)
    fin = _finish_of(r, mode)
    sign = "+" if d and d > 0 else ""
    vs_adp = ""
    if r.get("delta_adp") is not None:
        da = r["delta_adp"]
        vs_adp = f'<span class="drc-rsub">vs ADP {"+" if da > 0 else ""}{da}</span>'
    return (f'<div class="drc-row">'
            f'<span class="drc-rk">{i}</span>'
            f'<div class="drc-pcell">{_avatar(r, 34)}<div class="drc-pmeta">'
            f'<span class="drc-pname">{_esc(r.get("name"))} {_pos_chip(r.get("pos"))}</span>'
            f'<span class="drc-psub">{_esc(r.get("label"))} · {_esc(r.get("team"))}</span>'
            f'</div></div>'
            f'<div class="drc-rstats"><span>{r["ppg"]:.1f} PPG</span>'
            f'<span class="drc-rsub">Finished {r["pos"]}{fin} · drafted {r["pos"]}{r["draft_pos_rank"]}</span>'
            f'{vs_adp}</div>'
            f'<div class="drc-rdelta"><span class="drc-big" style="color:{vc}">{sign}{d}</span>'
            f'<span class="drc-verdict" style="--vc:{vc}">{v}</span></div></div>')


def _story(payload):
    mode = payload.get("mode")
    lists = payload.get("lists") or {}
    steals = lists.get("steals") or []
    busts = lists.get("busts") or []

    def _lst(title, items):
        body = "".join(_pick_row(i + 1, r, mode) for i, r in enumerate(items))
        return (f'<div class="drc-listcard"><div class="drc-listhead">{_esc(title)}</div>'
                f'{body}</div>')

    round_cards = []
    for g in lists.get("by_round") or []:
        r = g["row"]
        d = r["delta_pick"]
        fin = _finish_of(r, mode)
        round_cards.append(
            f'<div class="drc-roundcard"><div class="drc-sup-k" style="--vc:#38bdf8">{_esc(g["label"])}</div>'
            f'<div class="drc-pcell" style="margin-top:10px">{_avatar(r, 38)}<div class="drc-pmeta">'
            f'<span class="drc-pname">{_esc(r.get("name"))}</span>'
            f'<span class="drc-psub">{_esc(r.get("label"))} · {_esc(r.get("team"))}</span>'
            f'</div>{_pos_chip(r.get("pos"))}</div>'
            f'<div class="drc-sup-v" style="color:{verdict_for(d)[1]}">{"+" if d > 0 else ""}{d}</div>'
            f'<div class="drc-sup-s">{r["ppg"]:.1f} PPG · finished {r["pos"]}{fin}, '
            f'drafted {r["pos"]}{r["draft_pos_rank"]}</div></div>')
    rounds_html = ""
    if round_cards:
        rounds_html = ('<div class="drc-sec" style="margin-top:16px">Best pick by round</div>'
                       f'<div class="drc-roundstrip">{"".join(round_cards)}</div>')
    return ('<div class="drc-sec">The story</div>'
            f'<div class="drc-lists"><div>{_lst("Steals", steals)}</div>'
            f'<div>{_lst("Busts", busts)}</div></div>' + rounds_html)


def _board_grid(payload):
    rows = payload.get("picks") or []
    slot_teams = payload.get("slot_teams") or {}
    rounds = sorted({r["round"] for r in rows if r.get("round")})
    slots = sorted({r["draft_slot"] for r in rows if r.get("draft_slot")})
    if not rounds or not slots:
        return ""
    by_key = {(r["round"], r["draft_slot"]): r for r in rows}
    parts = ['<div class="drc-gridwrap"><div class="drc-grid" '
             f'style="grid-template-columns:40px repeat({len(slots)},minmax(96px,1fr))">']
    parts.append('<div class="drc-ghead"></div>')
    for sl in slots:
        parts.append(f'<div class="drc-ghead">{_esc(slot_teams.get(str(sl), ""))}</div>')
    for rd in rounds:
        parts.append(f'<div class="drc-grow">R{rd}</div>')
        for sl in slots:
            r = by_key.get((rd, sl))
            if not r or r.get("pos") not in CORE_POSITIONS:
                parts.append('<div class="drc-cell drc-cell-empty"></div>')
                continue
            d = r.get("delta_pick")
            _v, vc = _verdict_of(r)
            if r.get("injury_exempt"):
                # No delta numeral and no red: slate tint, the missed time
                # explained, and an Injured tag where the delta would go.
                # The lead-in stays truthful for a pick who dressed before
                # losing the season: "1 game", not "No games".
                _g = r.get("g") or 0
                _lead = ("No games" if not _g
                         else f"{_g} game" if _g == 1 else f"{_g} games")
                fin_txt = f'{_lead} · {_esc(r.get("injury_evidence") or "Injured")}'
                delta = (f'<span class="drc-verdict" style="--vc:{vc}">Injured</span>')
                alpha = 8
                sub = fin_txt
            else:
                alpha = 0 if d is None else min(26, 5 + min(abs(d), 30) * 0.7)
                fin = _finish_of(r, payload.get("mode"))
                fin_txt = f'{r["pos"]}{fin} finish' if fin is not None else "no games yet"
                delta = f'{"+" if d and d > 0 else ""}{d}' if d is not None else "-"
                sub = f'{r["ppg"]:.1f} PPG · {fin_txt}'
            parts.append(
                f'<div class="drc-cell" style="background:color-mix(in srgb,{vc} {alpha:.0f}%,var(--card))">'
                f'<div class="drc-cell-top">{_avatar(r, 24)}<span class="drc-cell-pos">{_esc(r["pos"])}</span></div>'
                f'<div class="drc-cell-name">{_esc(r.get("name"))}</div>'
                f'<div class="drc-cell-sub">{sub}</div>'
                f'<div class="drc-cell-delta" style="color:{vc}">{delta}</div></div>')
    parts.append('</div></div>')
    return "".join(parts)


def _chart(payload):
    """(svg_html, tips) scatter: x = pick_no, y = PPG finish among drafted."""
    mode = payload.get("mode")
    rows = [r for r in (payload.get("picks") or []) if r.get("pos") in CORE_POSITIONS]
    if not rows:
        return "", {}
    fin = rank_values({r["player_id"]: (r["ppg_raw"] if r.get("ppg_raw") is not None else 0.0)
                       for r in rows})
    W, H = 700, 470
    L, R, T, B = 48, 18, 18, 36
    xmax = max(r["pick_no"] for r in rows)
    ymax = max(fin.values())

    def X(v):
        return L + (v - 1) / max(1, xmax - 1) * (W - L - R)

    def Y(v):
        return T + (v - 1) / max(1, ymax - 1) * (H - T - B)

    def step_for(mx):
        for st in (1, 2, 5, 10, 15, 20, 25, 50):
            if mx / st <= 7:
                return st
        return 100

    parts = [f'<svg class="drc-chart" viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Draft position versus finish">']
    v = 1
    sx = step_for(xmax)
    while v <= xmax:
        parts.append(f'<line x1="{X(v):.0f}" y1="{T}" x2="{X(v):.0f}" y2="{H - B}" stroke="#e8eef5" stroke-width="1"/>')
        parts.append(f'<text x="{X(v):.0f}" y="{H - B + 18}" text-anchor="middle" class="drc-chart-tick">{v}</text>')
        v += sx
    v = 1
    sy = step_for(ymax)
    while v <= ymax:
        parts.append(f'<line x1="{L}" y1="{Y(v):.0f}" x2="{W - R}" y2="{Y(v):.0f}" stroke="#e8eef5" stroke-width="1"/>')
        parts.append(f'<text x="{L - 8}" y="{Y(v) + 4:.0f}" text-anchor="end" class="drc-chart-tick">{v}</text>')
        v += sy
    parts.append(f'<line x1="{X(1):.0f}" y1="{Y(1):.0f}" x2="{X(xmax):.0f}" y2="{Y(ymax):.0f}" stroke="#94a3b8" stroke-width="1.4" stroke-dasharray="5 5"/>')
    parts.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H - B}" stroke="#cbd5e1" stroke-width="1.4"/>')
    parts.append(f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="#cbd5e1" stroke-width="1.4"/>')

    rad = 13 if len(rows) <= 40 else 10
    defs, dots = [], []
    for r in rows:
        _v, vc = _verdict_of(r)
        cx, cy = X(r["pick_no"]), Y(fin[r["player_id"]])
        hs = r.get("headshot") or ""
        if hs:
            cid = f'cp{r["player_id"]}'
            defs.append(f'<clipPath id="{cid}"><circle cx="{cx:.0f}" cy="{cy:.0f}" r="{rad}"/></clipPath>')
            dots.append(
                f'<g class="drc-dot" data-pid="{_esc(r["player_id"])}">'
                f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{rad + 2.5}" fill="#fff"/>'
                f'<image href="{_esc(hs)}" x="{cx - rad:.0f}" y="{cy - rad:.0f}" width="{rad * 2}" height="{rad * 2}" '
                f'clip-path="url(#{cid})" preserveAspectRatio="xMidYMid slice"/>'
                f'<circle class="drc-ring" cx="{cx:.0f}" cy="{cy:.0f}" r="{rad + 2.5}" fill="none" stroke="{vc}" stroke-width="3"/></g>')
        else:
            dots.append(
                f'<g class="drc-dot" data-pid="{_esc(r["player_id"])}">'
                f'<circle class="drc-ring" cx="{cx:.0f}" cy="{cy:.0f}" r="{rad - 2}" fill="{vc}" '
                f'fill-opacity="0.85" stroke="#fff" stroke-width="1.6"/></g>')
    parts.append("<defs>" + "".join(defs) + "</defs>")
    parts.extend(dots)

    lists = payload.get("lists") or {}
    lab = (lists.get("steals") or [])[:2] + (lists.get("busts") or [])[:2]
    for i, r in enumerate(lab):
        cx, cy = X(r["pick_no"]), Y(fin[r["player_id"]])
        last = _esc((r.get("name") or "").split()[-1] if r.get("name") else "")
        if i == 2:
            parts.append(f'<text x="{cx:.0f}" y="{cy - rad - 6:.0f}" text-anchor="middle" class="drc-chart-lab">{last}</text>')
        elif i == 3:
            parts.append(f'<text x="{cx:.0f}" y="{cy + rad + 15:.0f}" text-anchor="middle" class="drc-chart-lab">{last}</text>')
        elif cx > W - 170:
            parts.append(f'<text x="{cx - rad - 6:.0f}" y="{cy + 4:.0f}" text-anchor="end" class="drc-chart-lab">{last}</text>')
        else:
            parts.append(f'<text x="{cx + rad + 6:.0f}" y="{cy + 4:.0f}" class="drc-chart-lab">{last}</text>')
    parts.append(f'<text x="{L}" y="{H - 6}" class="drc-chart-axis">Draft pick</text>')
    parts.append(f'<text x="12" y="{T + 4}" class="drc-chart-axis">Finish</text>')
    parts.append('</svg>')
    parts.append('<div class="drc-chart-note">Finish among drafted players, by PPG. '
                 'Above the line: outplayed their draft spot. Below it: busts.</div>')

    tips = {}
    for r in rows:
        v, vc = _verdict_of(r)
        if r.get("injury_exempt"):
            _g = r.get("g") or 0
            finl = ("No games" if not _g
                    else f"{_g} game, no points" if _g == 1
                    else f"{_g} games, no points")
            fins = r.get("injury_evidence") or "Injured"
        elif mode == "rookie":
            finl = f'{r["pos"]}{r.get("class_finish")} in class'
            fins = f'{r["pos"]}{r.get("finish_ppg")} overall'
        else:
            finl = f'{r["pos"]}{r.get("finish_ppg")}'
            fins = f'#{fin[r["player_id"]]} of {len(rows)} drafted'
        tips[r["player_id"]] = {
            "name": r.get("name") or "", "pos": r.get("pos") or "",
            "nfl": r.get("nfl") or "", "pick": r.get("label") or "",
            "team": (r.get("team") or "").strip(),
            "g": r.get("g") or 0, "ppg": r.get("ppg") or 0.0,
            "tot": r.get("total") or 0.0, "fin": finl, "fins": fins,
            "dp": r.get("delta_pick"), "da": r.get("delta_adp"),
            "verdict": v, "vc": vc, "hs": r.get("headshot") or "",
        }
    return "".join(parts), tips


_JS = """
(function(){
  var TIPS = __TIPS__;
  var boardBtn = document.querySelector("[data-board]");
  var board = document.getElementById("drcBoard");
  if (boardBtn && board) boardBtn.addEventListener("click", function(){
    var open = board.hasAttribute("hidden");
    if (open) { board.removeAttribute("hidden"); boardBtn.textContent = "Hide full board"; }
    else { board.setAttribute("hidden", ""); boardBtn.textContent = "Show full board"; }
  });
  document.querySelectorAll("[data-bview]").forEach(function(b){ b.addEventListener("click", function(){
    var kind = b.getAttribute("data-bview");
    document.getElementById("drcGrid").toggleAttribute("hidden", kind !== "grid");
    document.getElementById("drcChart").toggleAttribute("hidden", kind !== "chart");
    b.parentElement.querySelectorAll("button").forEach(function(x){ x.classList.toggle("on", x === b); });
  });});
  function fmtD(d){ return (d === null || d === undefined) ? "-" : (d > 0 ? "+" + d : "" + d); }
  function esc(s){ return String(s == null ? "" : s).replace(/[&<>"]/g, function(c){
    return ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"})[c]; }); }
  function tipHtml(t){
    var av = t.hs ? '<img src="' + esc(t.hs) + '" alt="">' : '<span class="drc-tip-disc"></span>';
    return '<div class="drc-tip-head">' + av + '<div><div class="drc-tip-name">' + esc(t.name) + '</div>'
      + '<div class="drc-tip-sub">' + esc(t.pos) + (t.nfl ? ' \\u00b7 ' + esc(t.nfl) : '') + '</div></div>'
      + '<span class="drc-verdict" style="--vc:' + t.vc + '">' + esc(t.verdict) + '</span></div>'
      + '<div class="drc-tip-row">Pick ' + esc(t.pick) + ' \\u00b7 ' + esc(t.team) + '</div>'
      + '<div class="drc-tip-row">' + t.g + ' G \\u00b7 ' + Number(t.ppg).toFixed(1) + ' PPG \\u00b7 ' + Number(t.tot).toFixed(1) + ' total</div>'
      + '<div class="drc-tip-row">Finished ' + esc(t.fin) + ' <span class="drc-tip-dim">(' + esc(t.fins) + ')</span></div>'
      + '<div class="drc-tip-deltas"><span>vs Pick <b style="color:' + (t.dp > 0 ? '#16a34a' : t.dp < 0 ? '#dc2626' : 'inherit') + '">' + fmtD(t.dp) + '</b></span>'
      + '<span>vs ADP <b style="color:' + (t.da > 0 ? '#16a34a' : t.da < 0 ? '#dc2626' : 'inherit') + '">' + fmtD(t.da) + '</b></span></div>';
  }
  var tip = document.getElementById("drcTip");
  document.querySelectorAll(".drc-dot").forEach(function(g){
    var card = g.closest(".drc-chartcard");
    function show(e){
      var t = TIPS[g.getAttribute("data-pid")]; if (!t || !tip) return;
      tip.innerHTML = tipHtml(t); tip.removeAttribute("hidden");
      var cr = card.getBoundingClientRect(), x = e.clientX - cr.left + 14, y = e.clientY - cr.top + 14;
      if (x + 240 > cr.width) x = e.clientX - cr.left - 246;
      if (y + 168 > cr.height) y = e.clientY - cr.top - 174;
      tip.style.left = Math.max(4, x) + "px"; tip.style.top = Math.max(4, y) + "px";
    }
    g.addEventListener("mouseenter", show);
    g.addEventListener("mousemove", show);
    g.addEventListener("mouseleave", function(){ tip.setAttribute("hidden", ""); });
    g.addEventListener("click", function(e){ if (tip.hasAttribute("hidden")) { show(e); } else { tip.setAttribute("hidden", ""); } });
  });
})();
"""


def build_draft_recap_body(league_id, season, platform) -> str:
    """Draft Recap page body for one league/season (or the honest empty
    state when there is no league, no completed draft, an auction draft, or
    no completed weeks yet)."""
    platform = (platform or "sleeper").strip().lower()
    if not league_id:
        return _empty(None, league_id, season, platform)
    try:
        payload = _payload(platform, league_id, season)
    except Exception:
        logger.exception("[draft-recap] payload build failed")
        return _empty(None, league_id, season, platform,
                      message="The draft recap could not be built for this league right now.")
    if not payload or payload.get("status") != "ok" or not payload.get("picks_graded"):
        return _empty(payload, league_id, season, platform)

    chart_html, tips = _chart(payload)
    grid_html = _board_grid(payload)
    board_html = ""
    if grid_html:
        board_html = (
            '<div class="drc-boardwrap">'
            '<button type="button" class="drc-boardbtn" data-board="1">Show full board</button>'
            '<div class="drc-board" id="drcBoard" hidden>'
            '<div class="drc-seg drc-bseg" role="group" aria-label="Board view">'
            '<button type="button" class="on" data-bview="grid">Board</button>'
            '<button type="button" data-bview="chart">Chart</button></div>'
            f'<div id="drcGrid">{grid_html}</div>'
            f'<div id="drcChart" hidden><div class="drc-card drc-chartcard">{chart_html}'
            '<div class="drc-tip" id="drcTip" hidden></div></div></div>'
            '</div></div>')
    tips_json = json.dumps(tips).replace("</", "<\\/")
    js = _JS.replace("__TIPS__", tips_json)
    return (f'<div class="drc-wrap">{_hero(payload, league_id, season, platform)}'
            f'{_sup_cards(payload)}{_team_table(payload)}{_story(payload)}'
            f'{board_html}</div>'
            f'<style>{_CSS}</style><script>{js}</script>')
