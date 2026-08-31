"""Local UI/UX audit hub — only registered when ``UI_AUDIT=1``.

Visit ``/ui-audit`` for a link catalog of every public and league page backed
by the deterministic ``ui-audit`` mock league (mid-season dynasty, no live HTTP).
"""
from __future__ import annotations

import html

from flask import Blueprint, redirect, request, session, url_for

from utils.ui_audit_fixture import (
    UI_AUDIT_LEAGUE_ID,
    _DEFAULT_PLATFORM,
    _DEFAULT_SEASON,
    all_audit_hrefs,
    bootstrap_viewer_session,
    league_page_href,
    ui_audit_enabled,
)

ui_audit_bp = Blueprint("ui_audit", __name__)


def _hub_body() -> str:
    public_links: list[tuple[str, str]] = []
    league_links: list[tuple[str, str]] = []
    for href, label in all_audit_hrefs():
        if href.startswith(f"/{_DEFAULT_PLATFORM}/"):
            league_links.append((href, label))
        else:
            public_links.append((href, label))

    def _section(title: str, links: list[tuple[str, str]]) -> str:
        rows = "".join(
            f'<li><a href="{html.escape(h)}">{html.escape(lbl)}</a>'
            f' <span class="ui-audit-path">{html.escape(h)}</span></li>'
            for h, lbl in links
        )
        return (
            f'<section class="card central ui-audit-section">'
            f'<div class="card-header"><h2>{html.escape(title)}</h2></div>'
            f'<div class="card-body"><ul class="ui-audit-links">{rows}</ul></div>'
            f"</section>"
        )

    dash = league_page_href("dashboard")
    return (
        '<style>'
        ".ui-audit-hero{margin:0 0 1rem;color:var(--text-muted);font-size:15px;line-height:1.5}"
        ".ui-audit-links{list-style:none;margin:0;padding:0;display:grid;gap:10px}"
        ".ui-audit-links a{font-weight:600}"
        ".ui-audit-path{display:block;font-size:12px;color:var(--text-muted);font-family:monospace}"
        ".ui-audit-actions{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 1.25rem}"
        ".ui-audit-actions .btn{min-height:44px}"
        "</style>"
        '<div class="ui-audit-hero">'
        "<p>Deterministic mock league <strong>UI Audit Dynasty</strong> "
        f"(<code>{UI_AUDIT_LEAGUE_ID}</code>) — week 11, 10 teams, in-season. "
        "No live Sleeper calls.</p>"
        '<div class="ui-audit-actions">'
        f'<a class="btn btn-primary" href="{url_for("ui_audit.bootstrap")}">'
        "Bootstrap signed-in session</a>"
        f'<a class="btn btn-secondary" href="{html.escape(dash)}">Open dashboard</a>'
        "</div></div>"
        + _section("Public & account pages", public_links)
        + _section("League pages (mock data)", league_links)
    )


@ui_audit_bp.route("/ui-audit")
def hub():
    if not ui_audit_enabled():
        return ("UI audit mode is off. Set UI_AUDIT=1 and restart.", 404)
    from app import render_page

    body = _hub_body()
    return render_page(
        "UI Audit Hub",
        None,
        "",
        body,
        description="Local UI/UX walkthrough catalog for BR Fantasy.",
    )


@ui_audit_bp.route("/ui-audit/bootstrap")
def bootstrap():
    if not ui_audit_enabled():
        return ("UI audit mode is off.", 404)
    bootstrap_viewer_session(session)
    session.modified = True
    nxt = request.args.get("next") or league_page_href("dashboard")
    if not nxt.startswith("/"):
        nxt = league_page_href("dashboard")
    return redirect(nxt)
