"""Reusable email-safe HTML sections for the weekly digest.

Keep markup table-light, inline-styled, and ~600px wide so it survives mobile
clients. Sections return "" when they have nothing useful to show.
"""
from __future__ import annotations

from html import escape
from typing import Any, Optional

from utils.digest_actions import action_section_html, player_deep_link, section_card

MAX_WIDTH_PX = 600


def email_shell(
    inner_html: str,
    *,
    subtitle: str,
    dash_url: str = "",
    cta_label: str = "Open your dashboard →",
    unsub_href: str = "{UNSUB}",
    logo_url: str = "",
    brand_mark_url: str = "",
    footer_kind: str = "weekly_digest",
    header_theme: str = "dark",
) -> str:
    """Wrap email body in the BR Fantasy chrome.

    ``footer_kind`` controls unsubscribe copy:
      - ``weekly_digest`` (default)
      - ``onboarding``: signup / PRO welcome emails

    ``header_theme`` controls the masthead:
      - ``dark`` (default): navy header with the light distressed wordmark.
      - ``light``: white header with the full-color navy wordmark and no
        redundant kicker line (pair with light-mode logo assets).
    """
    sub = escape(subtitle or "Your weekly fantasy digest", quote=False)
    base_logo = (logo_url or "").strip()
    mark = (brand_mark_url or "").strip()
    logo_block = ""
    if base_logo or mark:
        imgs = []
        if mark:
            imgs.append(
                f'<img src="{escape(mark, quote=True)}" alt="" width="36" height="36" '
                f'style="display:block;border:0;outline:none;width:36px;height:36px;'
                f'border-radius:8px;" />'
            )
        if base_logo:
            imgs.append(
                f'<img src="{escape(base_logo, quote=True)}" alt="BR Fantasy" width="160" '
                f'style="display:block;border:0;outline:none;width:160px;height:auto;'
                f'max-width:70%;" />'
            )
        logo_block = (
            '<div style="margin:0 0 14px;">'
            + (
                f'<table role="presentation" cellpadding="0" cellspacing="0"><tr>'
                f'<td style="vertical-align:middle;padding-right:12px;">{imgs[0]}</td>'
                f'<td style="vertical-align:middle;">{imgs[1] if len(imgs) > 1 else ""}</td>'
                f"</tr></table>"
                if len(imgs) == 2
                else imgs[0]
            )
            + "</div>"
        )
    cta = ""
    if dash_url:
        cta = f"""\
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin-top:24px;">
  <tr>
    <td>
      <a href="{escape(dash_url, quote=True)}" style="display:block;background:#2563eb;color:#ffffff;text-decoration:none;font-weight:700;font-size:15px;padding:14px 20px;border-radius:10px;text-align:center;">{escape(cta_label, quote=False)}</a>
    </td>
  </tr>
</table>"""
    if footer_kind == "onboarding":
        footer = (
            "You're getting this because you created a BR Fantasy account "
            "(or upgraded to PRO). "
            f'<a href="{escape(unsub_href, quote=True)}" style="color:#64748b;">Unsubscribe</a> '
            "from welcome and onboarding emails. Weekly digests are separate."
        )
    else:
        footer = (
            "You're getting this because you signed in to BR Fantasy. "
            f'<a href="{escape(unsub_href, quote=True)}" style="color:#64748b;">Unsubscribe</a> '
            "from weekly digest emails."
        )
    if header_theme == "light":
        header_bg = "#ffffff"
        subtitle_color = "#0f172a"
        kicker_html = ""
    else:
        header_bg = "#0b1220"
        subtitle_color = "#ffffff"
        kicker_html = (
            '<div style="color:#93c5fd;font-size:11px;font-weight:800;'
            'letter-spacing:.14em;text-transform:uppercase;">BR Fantasy</div>'
        )
    return f"""\
<div style="background:#e8eef5;padding:28px 12px;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
  <div style="max-width:{MAX_WIDTH_PX}px;margin:0 auto;background:#ffffff;border-radius:16px;overflow:hidden;border:1px solid #dbe3ee;">
    <div style="background:{header_bg};padding:22px 24px 18px;">
      {logo_block}
      {kicker_html}
      <div style="color:{subtitle_color};font-size:20px;font-weight:800;margin-top:6px;line-height:1.25;">{sub}</div>
    </div>
    <div style="height:4px;background:#2563eb;line-height:4px;font-size:0;">&nbsp;</div>
    <div style="padding:22px 20px 24px;background:#f7f9fc;">
      {inner_html}
      {cta}
    </div>
    <div style="padding:16px 22px;background:#ffffff;border-top:1px solid #e6ebf2;">
      <p style="margin:0;font-size:11px;color:#94a3b8;line-height:1.6;">
        {footer}
      </p>
    </div>
  </div>
</div>"""


def greeting_html(first_name: Optional[str]) -> str:
    hi = escape(first_name.strip(), quote=False) if first_name and first_name.strip() else "there"
    return (
        f'<p style="margin:0 0 12px;font-size:16px;color:#0f172a;font-weight:600;">'
        f"Hey {hi},</p>"
    )


def heading(title: str) -> str:
    t = escape(str(title or "").strip(), quote=False)
    if not t:
        return ""
    return (
        f'<h3 style="margin:20px 0 0;font-size:11px;font-weight:800;text-transform:uppercase;'
        f'letter-spacing:.06em;color:#334155;">{t}</h3>'
    )


def format_chip_html(label: str) -> str:
    text = str(label or "").strip()
    if not text:
        return ""
    return (
        f'<span style="display:inline-block;margin-top:8px;padding:5px 12px;border-radius:999px;'
        f'background:#dbeafe;color:#1d4ed8;font-size:11px;font-weight:700;letter-spacing:.02em;">'
        f"{escape(text, quote=False)}</span>"
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
    games = int(wins or 0) + int(losses or 0)
    if rank and games > 0:
        rec = f"{int(wins or 0)}-{int(losses or 0)}"
        headline = (
            f'You\'re <strong>#{int(rank)}</strong> in {lg} at '
            f'<strong>{escape(rec, quote=False)}</strong>.'
        )
        size = "16px"
    else:
        headline = (
            f'<strong style="font-size:20px;letter-spacing:-0.02em;">{lg}</strong>'
        )
        size = "16px"
    chip = format_chip_html(format_label)
    return (
        f'<div style="margin:0 0 8px;font-size:{size};color:#0f172a;line-height:1.4;">'
        f"{headline}{chip}</div>"
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
            lines.append(f"Projected {yu:.1f} to {ot:.1f}")
            if margin is not None:
                m = float(margin)
                if abs(m) >= 0.05:
                    verb = "Favored by" if m > 0 else "Projected behind by"
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


def waiver_html(
    targets: list,
    *,
    href: str = "",
    base: str = "",
    platform: str = "",
    season: int = 0,
    league_id: str = "",
) -> str:
    if not targets:
        return ""
    rows = ""
    shown = 0
    for t in targets[:3]:
        name = str(t.get("name") or "").strip()
        if not name:
            continue
        pos = str(t.get("pos") or "").upper()
        reason = str(t.get("reason") or "").strip()
        pid = str(t.get("player_id") or "")
        label = escape(name, quote=False)
        if base and platform and season and league_id and pid:
            link = player_deep_link(base, platform, season, league_id, pid, name)
            label = (
                f'<a href="{escape(link, quote=True)}" style="color:#0f172a;'
                f'text-decoration:none;font-weight:700;">{label}</a>'
            )
        meta = " · ".join(p for p in (pos, reason) if p)
        border = "border-top:1px solid #e2e8f0;" if shown else ""
        rows += (
            f'<tr><td style="padding:8px 0;{border}font-size:15px;color:#0f172a;">{label}'
            f'<div style="font-size:12px;color:#64748b;margin-top:2px;">{escape(meta, quote=False)}</div>'
            f"</td></tr>"
        )
        shown += 1
    if not shown:
        return ""
    title = "Top waiver target" if shown == 1 else "Waiver wire"
    return section_card(
        title,
        f'<table style="width:100%;border-collapse:collapse;">{rows}</table>',
        href=href,
        cta="View waivers →" if href else "",
        accent=True,
    )


def roster_core_html(
    players: list,
    *,
    base: str = "",
    platform: str = "",
    season: int = 0,
    league_id: str = "",
) -> str:
    if not players or len(players) < 2:
        return ""
    rows = ""
    for p in players[:3]:
        name = str(p.get("name") or "").strip()
        if not name:
            continue
        pos = str(p.get("pos") or "").upper()
        try:
            val = float(p.get("value") or 0)
        except (TypeError, ValueError):
            val = 0.0
        pid = str(p.get("player_id") or "")
        label = escape(name, quote=False)
        if base and platform and season and league_id and pid:
            link = player_deep_link(base, platform, season, league_id, pid, name)
            label = (
                f'<a href="{escape(link, quote=True)}" style="color:#0f172a;'
                f'text-decoration:none;font-weight:600;">{label}</a>'
            )
        pos_s = escape(pos, quote=False)
        rows += (
            f'<tr><td style="padding:6px 0;font-size:14px;">{label}'
            f'<div style="font-size:12px;color:#64748b;">{pos_s}</div></td>'
            f'<td style="padding:6px 0;font-size:14px;font-weight:700;color:#0f172a;'
            f'text-align:right;white-space:nowrap;">{val:.0f}</td></tr>'
        )
    if not rows:
        return ""
    return section_card(
        "Your top assets",
        f'<table style="width:100%;border-collapse:collapse;">{rows}</table>',
        accent=False,
    )


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
            parts.append(section_card(title, table, accent=False))
    if my_fallers:
        table = _mover_rows(
            my_fallers, up=False, base=base, platform=platform, season=season,
            league_id=league_id, pidx=pidx, notes=notes,
        )
        if table:
            parts.append(section_card("Your fallers this week", table, accent=False))
    if show_leaguewide and lg_risers:
        table = _mover_rows(
            lg_risers, up=True, base=base, platform=platform, season=season,
            league_id=league_id, pidx=pidx, notes=notes,
        )
        if table:
            parts.append(section_card("Biggest risers leaguewide", table, accent=False))
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
    """Short human label like ``SF · TEP · Dynasty`` without internal ids."""
    if not fmt:
        return ""
    kind = str(fmt.get("type") or "").strip()
    if kind:
        kind = kind[0].upper() + kind[1:]
    parts = ["SF" if fmt.get("is_superflex") else "1QB"]
    if fmt.get("is_tep"):
        parts.append("TEP")
    if kind:
        parts.append(kind)
    return " · ".join(parts)


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
