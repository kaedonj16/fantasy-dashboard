"""SEO endpoints.

Routes:
    /robots.txt
    /sitemap.xml

Extracted from app.py to reduce monolith size.
Dependencies: flask only - no app.py internals.
"""
from __future__ import annotations

import logging

from flask import Blueprint, Response, request

logger = logging.getLogger(__name__)

seo_bp = Blueprint("seo", __name__)


@seo_bp.route("/robots.txt")
def robots_txt():
    # Build sitemap URL from the actual request host so staging/dev never
    # advertises the production sitemap URL.
    host = request.host_url.rstrip("/")
    txt = (
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /api/\n"
        "Disallow: /admin/\n"
        "\n"
        f"Sitemap: {host}/sitemap.xml\n"
    )
    return Response(txt, mimetype="text/plain")


@seo_bp.route("/sitemap.xml")
def sitemap_xml():
    host = request.host_url.rstrip("/")
    urls = [
        ("/",                           "weekly",  "1.0"),
        ("/players",                    "weekly",  "0.9"),
        ("/metrics",                    "weekly",  "0.9"),
        ("/draft",                      "weekly",  "0.9"),
        ("/draft/history",              "monthly", "0.7"),
        ("/prospects",                  "weekly",  "0.8"),
        ("/trade",                      "weekly",  "0.9"),
        ("/trade-database",             "weekly",  "0.7"),
        ("/top-movers",                 "daily",   "0.8"),
        ("/breakouts",                  "weekly",  "0.8"),
        ("/dynasty-trade-value-chart",  "weekly",  "0.9"),
        ("/rankings/dynasty",           "weekly",  "0.8"),
        ("/rankings/dynasty-qb",        "weekly",  "0.7"),
        ("/rankings/dynasty-rb",        "weekly",  "0.7"),
        ("/rankings/dynasty-wr",        "weekly",  "0.7"),
        ("/rankings/dynasty-te",        "weekly",  "0.7"),
    ]
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for path, freq, priority in urls:
        lines.append(
            f"  <url><loc>{host}{path}</loc>"
            f"<changefreq>{freq}</changefreq>"
            f"<priority>{priority}</priority></url>"
        )
    lines.append("</urlset>")
    return Response("\n".join(lines), mimetype="application/xml")
