"""Reusable email-safe HTML sections for the weekly digest.

Keep markup table-light, inline-styled, and ~600px wide so it survives mobile
clients. Sections return "" when they have nothing useful to show.
"""
from __future__ import annotations

from html import escape
from typing import Any, Optional

from utils.digest_actions import action_section_html, player_deep_link

MAX_WIDTH_PX = 600


def email_shell(
    inner_html: str,
    *,
    subtitle: str,
    dash_url: str = "",
    cta_label: str = "Open your dashboard →",
    unsub_href: str = "{UNSUB}",
) -> str:
    sub = escape(subtitle or "Your weekly fantasy digest", quote=False)
    cta = ""
    if dash_url:
        cta = (
            f'<a href="{escape(dash_url, quote=True)}" style="display:inline-block;'
            f'margin-top:22px;background:#2563eb;color:#ffffff;text-decoration:none;'
            f'font-weight:700;font-size:14px;padding:11px 20px;border-radius:9px;">'
            f"{escape(cta_label, quote=False)}</a>"
        )
    return f"""\
<div style="background:#f1f5f9;padding:24px 0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
  <div style="max-width:{MAX_WIDTH_PX}px;margin:0 auto;background:#ffffff;border-radius:14px;overflow:hidden;border:1px solid #e2e8f0;">
    <div style="background:#0f172a;padding:20px 24px;">
      <div style="color:#ffffff;font-size:18px;font-weight:800;">BR Fantasy</div>
      <div style="color:#94a3b8;font-size:13px;margin-top:2px;">{sub}</div>
    </div>
    <div style="padding:24px;">
      {inner_html}
      {cta}
    </div>
    <div style="padding:16px 24px;border-top:1px solid #e2e8f0;background:#f8fafc;">
      <p style="margin:0;font-size:11px;color:#94a3b8;line-height:1.6;">
        You're getting this because you signed in to BR Fantasy.
        <a href="{escape(unsub_href, quote=True)}" style="color:#64748b;">Unsubscribe</a>
        from weekly digest emails.
      </p>
    </div>
  </div>
</div>"""


def greeting_html(first_name: Optional[str]) -> str:
    hi = escape(first_name.strip(), quote=False) if first_name and first_name.strip() else "there"
    return f'<p style="margin:0 0 14px;font-size:15px;color:#0f172a;">Hey {hi},</p>'


def heading(title: str) -> str:
    t = escape(str(title or "").strip(), quote=False)
    if not t:
        return ""
    return (
        f'<h3 style="margin:20px 0 6px;font-size:13px;text-transform:uppercase;'
        f'letter-spacing:.04em;color:#64748b;">{t}</h3>'
    )


def league_summary_html(
    *,
    league_name: str,
    rank: Optional[int],
    wins: int = 0,
    losses: int = 0,
    format_label: str = "",
) -> str:
    lg = escape(league_name or "Your league", quote=False)
    bits = []
    if rank:
        rec = f"{int(wins or 0)}-{int(losses or 0)}"
        bits.append(
            f'You\'re <strong>#{int(rank)}</strong> in {lg} at <strong>{escape(rec, quote=False)}</strong>.'
        )
    else:
        bits.append(f'Here\'s your weekly report for {lg}.')
    if format_label:
        bits.append(
            f'<span style="color:#64748b;font-size:13px;">{escape(format_label, quote=False)}</span>'
        )
    return (
        f'<p style="margin:0 0 4px;font-size:15px;color:#0f172a;">'
        + "<br>".join(bits)
        + "</p>"
    )


def matchup_html(matchup: Optional[dict], *, href: str = "") -> str:
    if not matchup:
        return ""
    opp = str(matchup.get("opponent_name") or "").strip()
    if not opp:
        return ""
    you = matchup.get("user_proj")
    them = matchup.get("opp_proj")
    margin = matchup.get("margin")
    wp = matchup.get("win_prob")
    lines = [f"vs {escape(opp)}"]
    if you is not None and them is not None:
        try:
            yu, ot = float(you), float(them)
            lines.append(f"Projected {yu:.1f} – {ot:.1f}")
            if margin is not None:
                m = float(margin)
                if abs(m) >= 0.05:
                    verb = "favored by" if m > 0 else "projected behind by"
                    lines.append(f"{verb} {abs(m):.1f}")
        except (TypeError, ValueError):
            pass
    if wp is not None:
        try:
            pct = int(round(float(wp) * 100))
            pct = max(1, min(99, pct))
            lines.append(f"Win probability {pct}%")
        except (TypeError, ValueError):
            pass
    body = ". ".join(lines) + "."
    return action_section_html("This week's matchup", body, href=href, cta="Open matchup →")


def start_sit_html(note: Optional[dict], *, href: str = "") -> str:
    if not note:
        return ""
    title = str(note.get("title") or "Start/Sit")
    body = str(note.get("body") or "")
    if not body:
        return ""
    return action_section_html(title, body, href=href, cta="Fix lineup →")


def waiver_html(targets: list, *, href: str = "") -> str:
    if not targets:
        return ""
    lines = []
    for t in targets[:3]:
        name = str(t.get("name") or "").strip()
        if not name:
            continue
        pos = str(t.get("pos") or "").upper()
        reason = str(t.get("reason") or "").strip()
        label = f"{pos} {name}".strip() if pos else name
        if reason:
            lines.append(f"{label} — {reason}")
        else:
            lines.append(label)
    if not lines:
        return ""
    if len(lines) == 1:
        body = f"Top waiver target: {lines[0]}"
    else:
        body = "Top waiver targets: " + "; ".join(lines)
    return action_section_html("Waiver wire", body, href=href, cta="View waivers →")


def injury_html(note: Optional[dict], *, href: str = "") -> str:
    if not note:
        return ""
    body = str(note.get("body") or "").strip()
    if not body:
        return ""
    title = str(note.get("title") or "Injury")
    return action_section_html(title, body, href=href, cta="Review roster →")


def _mover_rows(
    pairs: list,
    *,
    up: bool,
    base: str,
    platform: str,
    season: int,
    league_id: str,
    pidx: dict,
    notes: Optional[dict] = None,
) -> str:
    color = "#16a34a" if up else "#dc2626"
    arrow = "▲" if up else "▼"
    cells = ""
    for item in pairs:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            pid, d = str(item[0]), item[1]
            extra = item[2] if len(item) > 2 else ""
        elif isinstance(item, dict):
            pid = str(item.get("player_id") or "")
            d = item.get("delta")
            extra = item.get("note") or ""
        else:
            continue
        if not pid or d is None:
            continue
        raw_name = _player_name(pid, pidx)
        if not raw_name:
            continue
        try:
            delta = float(d)
        except (TypeError, ValueError):
            continue
        note = extra or (notes or {}).get(pid) or ""
        nm = escape(raw_name, quote=False)
        href = escape(player_deep_link(base, platform, season, league_id, pid, raw_name), quote=True)
        note_html = (
            f'<div style="font-size:12px;color:#64748b;font-weight:400;">{escape(str(note), quote=False)}</div>'
            if note else ""
        )
        cells += (
            f'<tr><td style="padding:6px 0;font-size:14px;">'
            f'<a href="{href}" style="color:#0f172a;text-decoration:none;font-weight:600;">'
            f"{nm}</a>{note_html}</td>"
            f'<td style="padding:6px 0;font-size:14px;font-weight:700;color:{color};'
            f'text-align:right;white-space:nowrap;">{arrow} {abs(delta):.0f}</td></tr>'
        )
    if not cells:
        return ""
    return f'<table style="width:100%;border-collapse:collapse;">{cells}</table>'


def player_movement_html(
    *,
    my_risers: list,
    my_fallers: list,
    lg_risers: list,
    base: str,
    platform: str,
    season: int,
    league_id: str,
    pidx: dict,
    notes: Optional[dict] = None,
    show_leaguewide: bool = True,
    dynasty: bool = True,
) -> str:
    parts: list[str] = []
    if my_risers:
        title = "Your risers this week" if dynasty else "Roster trends this week"
        table = _mover_rows(
            my_risers, up=True, base=base, platform=platform, season=season,
            league_id=league_id, pidx=pidx, notes=notes,
        )
        if table:
            parts.append(heading(title) + table)
    if my_fallers:
        table = _mover_rows(
            my_fallers, up=False, base=base, platform=platform, season=season,
            league_id=league_id, pidx=pidx, notes=notes,
        )
        if table:
            parts.append(heading("Your fallers this week") + table)
    if show_leaguewide and lg_risers:
        table = _mover_rows(
            lg_risers, up=True, base=base, platform=platform, season=season,
            league_id=league_id, pidx=pidx, notes=notes,
        )
        if table:
            parts.append(heading("Biggest risers leaguewide") + table)
    return "".join(parts)


def breakout_html(watch: Optional[dict], *, href: str = "") -> str:
    if not watch:
        return ""
    name = str(watch.get("name") or "").strip()
    if not name:
        return ""
    score = watch.get("score")
    hit = watch.get("hit_probability")
    bits = [escape(name)]
    try:
        if score is not None:
            bits.append(f"Breakout Score {int(round(float(score)))}")
    except (TypeError, ValueError):
        pass
    try:
        if hit is not None:
            pct = float(hit)
            if pct <= 1:
                pct *= 100
            bits.append(f"{int(round(pct))}% hit rate")
    except (TypeError, ValueError):
        pass
    body = " · ".join(bits)
    return action_section_html("Breakout Watch", body, href=href, cta="Open player →")


def trade_insight_html(insight: Optional[dict], *, href: str = "") -> str:
    if not insight:
        return ""
    body = str(insight.get("body") or "").strip()
    if not body:
        return ""
    title = str(insight.get("title") or "Roster construction")
    return action_section_html(title, body, href=href, cta="Open trades →")


def format_chip(fmt: dict) -> str:
    """Short human label like ``12tm SF TEP dynasty`` without internal ids."""
    if not fmt:
        return ""
    kind = str(fmt.get("type") or "fantasy")
    qb = "SF" if fmt.get("is_superflex") else "1QB"
    tep = " TEP" if fmt.get("is_tep") else ""
    return f"{qb}{tep} {kind}".strip()


def _player_name(pid: str, pidx: dict) -> str:
    meta = (pidx or {}).get(str(pid)) or {}
    name = (
        meta.get("full_name")
        or meta.get("name")
        or ((meta.get("first_name") or "") + " " + (meta.get("last_name") or "")).strip()
    )
    name = str(name or "").strip()
    if not name or name == str(pid) or name.lower().startswith("player "):
        return ""
    return name
