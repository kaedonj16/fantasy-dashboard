"""
Public SEO / landing pages.

Routes:
    /dynasty-trade-value-chart
    /top-movers
    /compare
    /rankings/dynasty[-qb|-rb|-wr|-te]
    /player/<slug>[/trade-value]
    /breakouts   (guest)
    /prospects   (guest)

Extracted from app.py to reduce monolith size. These are all public, GET-only
marketing/landing pages that render via app.render_page.

Dependencies on app.py internals (render_page, page_players,
get_model_value_table_cached, _displayed_value_map, get_player_slug_index,
get_nfl_state, get_player_value_history, age_from_bday, page_breakouts,
page_prospects) are resolved through the lazy shims below rather than a top-level
``from app import ...`` so importing this module during app start-up does not
trigger a circular import — the real functions are only fetched when a request
is actually served.
"""
from __future__ import annotations

import html
import logging
from datetime import datetime

from flask import Blueprint, redirect, request, session

logger = logging.getLogger(__name__)

seo_pages_bp = Blueprint("seo_pages", __name__)


# ── Lazy shims to app.py internals (resolved at request time) ─────────────────

def render_page(*args, **kwargs):
    from app import render_page as _fn
    return _fn(*args, **kwargs)


def page_players(*args, **kwargs):
    from app import page_players as _fn
    return _fn(*args, **kwargs)


def get_model_value_table_cached(*args, **kwargs):
    from app import get_model_value_table_cached as _fn
    return _fn(*args, **kwargs)


def _displayed_value_map(*args, **kwargs):
    from app import _displayed_value_map as _fn
    return _fn(*args, **kwargs)


def get_player_slug_index(*args, **kwargs):
    from app import get_player_slug_index as _fn
    return _fn(*args, **kwargs)


def get_nfl_state(*args, **kwargs):
    from app import get_nfl_state as _fn
    return _fn(*args, **kwargs)


def get_player_value_history(*args, **kwargs):
    from app import get_player_value_history as _fn
    return _fn(*args, **kwargs)


def age_from_bday(*args, **kwargs):
    from app import age_from_bday as _fn
    return _fn(*args, **kwargs)


def page_breakouts(*args, **kwargs):
    from app import page_breakouts as _fn
    return _fn(*args, **kwargs)


def page_prospects(*args, **kwargs):
    from app import page_prospects as _fn
    return _fn(*args, **kwargs)


# ── Dynasty Trade Value Chart ─────────────────────────────────────────────────

@seo_pages_bp.route("/dynasty-trade-value-chart")
def dynasty_trade_value_chart():
    """Public dynasty trade value chart — same UI as player rankings, with SEO-optimised metadata."""
    from datetime import datetime as _dt
    as_of = _dt.now().strftime("%B %Y")
    year  = _dt.now().year
    return page_players(
        _title=f"Dynasty Fantasy Football Trade Value Chart {year} | BR Fantasy",
        _desc=(
            f"Updated {as_of}: real dynasty trade values for 1QB and Superflex leagues. "
            f"Sortable by position, age, and value. Use with the free Trade Calculator."
        ),
        _canonical="/dynasty-trade-value-chart",
    )


# ── Top Movers (was Risers & Fallers) ─────────────────────────────────────────

# /risers-fallers extracted to routes/misc_api_bp.py

@seo_pages_bp.route("/top-movers")
def top_movers_page():
    """Weekly dynasty risers and fallers — freshness content for SEO."""
    from dashboard_services.pages.dynasty_pages import build_risers_fallers_body
    from data_building.player_value_history import get_top_movers
    # Timeframe toggle: 7 / 30 / 90 days. Clamp to the supported set so a hand-
    # typed ?days= can't push an unbounded window into the query.
    try:
        days = int(request.args.get("days", 7))
    except (ValueError, TypeError):
        days = 7
    if days not in (7, 30, 90):
        days = 7
    try:
        movers = get_top_movers(days=days, limit=20, min_baseline_value=5,
                                min_current_value=20.0,
                                current_values=_displayed_value_map("1qb"))
    except Exception:
        movers = {"risers": [], "fallers": []}

    from datetime import datetime as _dt
    date_label = _dt.now().strftime("%B %d, %Y")
    body = build_risers_fallers_body(movers, as_of_date=date_label,
                                     signed_in=bool(session.get("viewer_username")),
                                     days=days)

    _win_label = {7: "week", 30: "30 days", 90: "90 days"}[days]
    return render_page(
        f"Top Movers: {date_label} | BR Fantasy",
        None, "top-movers", body,
        description=(
            f"Dynasty fantasy football risers and fallers over the last {_win_label} "
            f"({date_label}). Biggest trade value movers, act fast with the BR Fantasy "
            f"Trade Calculator."
        ),
    )


# ── Player Comparison ─────────────────────────────────────────────────────────

def _compare_name_for_id(pid: str | None) -> str | None:
    """Resolve a player display name from an id via the cached value table."""
    if not pid:
        return None
    pid = str(pid).strip()
    try:
        for r in (get_model_value_table_cached() or []):
            if str(r.get("id")) == pid:
                return r.get("name")
    except Exception:
        pass
    return None


def _compare_popular_matchups(n_pairs: int = 5) -> str:
    """A few marquee matchups for the empty state.

    Pairs same-position, adjacent-in-value players (RB1 vs RB2, WR1 vs WR2, ...)
    rather than pairing the top players sequentially by overall value. Adjacent
    same-position players are the real "who's better" debates - same role,
    similar value - whereas a positional-blind 1st-vs-2nd-overall pairing can put
    a QB next to a WR, which is not a meaningful comparison.
    """
    try:
        table = get_model_value_table_cached() or []
    except Exception:
        table = []

    from collections import defaultdict
    by_pos: dict = defaultdict(list)
    for r in sorted(
        (x for x in table if x.get("id") and x.get("name") and (x.get("value") or 0) > 0),
        key=lambda r: float(r.get("value") or 0), reverse=True,
    ):
        pos = str(r.get("position") or "").upper()
        if pos in ("QB", "RB", "WR", "TE"):
            by_pos[pos].append(r)

    # Adjacent pairs within each position: (1,2), (3,4), (5,6) - the top few.
    pos_pairs: dict = {}
    for pos, players in by_pos.items():
        pos_pairs[pos] = [(players[i], players[i + 1])
                          for i in range(0, min(len(players) - 1, 6), 2)]

    # Interleave across positions so the row is varied (a RB debate, a WR debate,
    # a QB debate, a TE debate, then the next tier) instead of all one position.
    order = ["RB", "WR", "QB", "TE"]
    chips: list = []
    round_i = 0
    while len(chips) < n_pairs:
        added = False
        for pos in order:
            plist = pos_pairs.get(pos) or []
            if round_i < len(plist):
                a, b = plist[round_i]
                href = f"/compare?p1={a['id']}&p2={b['id']}"
                _cpos = html.escape(pos)
                chips.append(
                    f"<a class='compare-chip' href='{href}'>"
                    f"<span class='compare-chip-pos pos-{_cpos}'>{_cpos}</span>"
                    f"<span class='compare-chip-name'>{html.escape(str(a['name']))}</span>"
                    f"<span class='compare-chip-vs'>vs</span>"
                    f"<span class='compare-chip-name'>{html.escape(str(b['name']))}</span></a>"
                )
                added = True
                if len(chips) >= n_pairs:
                    break
        if not added:
            break
        round_i += 1
    return "".join(chips)


def build_compare_page_body(popular_html: str = "") -> str:
    """Shell for the standalone compare page. Client-driven: static/app.js's
    initComparePage wires the two pickers, reads any ?p1=&p2= deep link, and
    renders the comparison inline via renderCompareInline."""
    return f"""
    <div class="page-layout" data-page="compare">
      <main class="page-main">
        <div class="compare-page">
          <header class="compare-page-head">
            <span class="compare-page-eyebrow"><i class="fa-solid fa-scale-balanced" aria-hidden="true"></i> Head to head</span>
            <h1 class="compare-page-title">Compare Players</h1>
            <p class="compare-page-sub">Put two players side by side and see who comes out ahead, or add a third for a shortlist. Type a tier like <strong>WR1</strong> or <strong>RB2</strong> to compare against the average of those top players.</p>
          </header>
          <div class="compare-pickers">
            <div class="compare-picker">
              <label class="compare-pick-label">Player 1</label>
              <div class="compare-pick-field">
                <input type="text" class="compare-pick-input" id="cmpPick1" placeholder="Search a player or type WR1…" autocomplete="off" role="combobox" aria-expanded="false" aria-controls="cmpResults1" aria-autocomplete="list" aria-label="Search player 1">
                <button type="button" class="compare-pick-clear" id="cmpClear1" aria-label="Clear player 1" hidden>&times;</button>
                <div class="compare-pick-results" id="cmpResults1" role="listbox"></div>
              </div>
              <div class="compare-tier-suggest" id="cmpSuggest1" hidden></div>
            </div>
            <div class="compare-vs" aria-hidden="true">VS</div>
            <div class="compare-picker">
              <label class="compare-pick-label">Player 2</label>
              <div class="compare-pick-field">
                <input type="text" class="compare-pick-input" id="cmpPick2" placeholder="Search a player or type WR1…" autocomplete="off" role="combobox" aria-expanded="false" aria-controls="cmpResults2" aria-autocomplete="list" aria-label="Search player 2">
                <button type="button" class="compare-pick-clear" id="cmpClear2" aria-label="Clear player 2" hidden>&times;</button>
                <div class="compare-pick-results" id="cmpResults2" role="listbox"></div>
              </div>
              <div class="compare-tier-suggest" id="cmpSuggest2" hidden></div>
            </div>
            <div class="compare-vs compare-vs-opt" aria-hidden="true">VS</div>
            <div class="compare-picker compare-picker-opt">
              <label class="compare-pick-label">Player 3 <span class="compare-pick-opt">optional</span></label>
              <div class="compare-pick-field">
                <input type="text" class="compare-pick-input" id="cmpPick3" placeholder="Add a third…" autocomplete="off" role="combobox" aria-expanded="false" aria-controls="cmpResults3" aria-autocomplete="list" aria-label="Search player 3">
                <button type="button" class="compare-pick-clear" id="cmpClear3" aria-label="Clear player 3" hidden>&times;</button>
                <div class="compare-pick-results" id="cmpResults3" role="listbox"></div>
              </div>
              <div class="compare-tier-suggest" id="cmpSuggest3" hidden></div>
            </div>
          </div>
          <div class="compare-actions" id="cmpActions" hidden>
            <button type="button" class="compare-action-btn" data-cmp-action="swap" title="Swap the two players">&#8646; Swap sides</button>
            <button type="button" class="compare-action-btn" data-cmp-action="copy" title="Copy a shareable link">Copy link</button>
            <button type="button" class="compare-action-btn" data-cmp-action="watch" title="Add both players to your watchlist">&#9734; Watch both</button>
            <a class="compare-action-btn" id="cmpTradeLink" href="/trade" title="Load these two into the trade calculator">Trade calculator &#8599;</a>
          </div>
          <div class="compare-empty" id="cmpEmptyState">
            <div class="compare-features">
              <div class="compare-feature"><span class="cf-ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="3 17 9 11 13 15 21 6"></polyline><polyline points="15 6 21 6 21 12"></polyline></svg></span><span>Dynasty value</span></div>
              <div class="compare-feature"><span class="cf-ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="6" y1="20" x2="6" y2="11"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="18" y1="20" x2="18" y2="14"></line></svg></span><span>Advanced metrics</span></div>
              <div class="compare-feature"><span class="cf-ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="17" rx="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="16" y1="2" x2="16" y2="6"></line></svg></span><span>Weekly usage</span></div>
              <div class="compare-feature"><span class="cf-ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="9" y1="6" x2="21" y2="6"></line><line x1="9" y1="12" x2="21" y2="12"></line><line x1="9" y1="18" x2="21" y2="18"></line><line x1="4" y1="6" x2="4.01" y2="6"></line><line x1="4" y1="12" x2="4.01" y2="12"></line><line x1="4" y1="18" x2="4.01" y2="18"></line></svg></span><span>Game logs</span></div>
            </div>
            <div class="compare-empty-block" id="cmpRecent" hidden>
              <div class="compare-empty-title">Recently compared</div>
              <div class="compare-chip-row" id="cmpRecentChips"></div>
            </div>
            <div class="compare-empty-block">
              <div class="compare-empty-title">Popular matchups</div>
              <div class="compare-chip-row" id="cmpPopularChips">{popular_html}</div>
            </div>
          </div>
          <div id="comparePageResult" class="compare-page-result"></div>
        </div>
      </main>
    </div>
    """


@seo_pages_bp.route("/compare")
def page_compare():
    p1 = request.args.get("p1")
    p2 = request.args.get("p2")
    n1 = _compare_name_for_id(p1)
    n2 = _compare_name_for_id(p2)
    if n1 and n2:
        title = f"{n1} vs {n2} Dynasty Comparison | BR Fantasy"
        desc = (f"Compare {n1} and {n2}: dynasty fantasy football trade value, advanced "
                f"metrics, weekly usage, and game logs side by side.")
    else:
        title = "Compare Players | BR Fantasy"
        desc = ("Put any two dynasty fantasy football players side by side: trade value, "
                "advanced metrics, weekly usage, and game logs.")
    body = build_compare_page_body(_compare_popular_matchups())
    nav_lid = session.get("last_league_id")
    nav_platform = session.get("last_platform")
    try:
        nav_season = int(session.get("last_season")) if session.get("last_season") else None
    except (TypeError, ValueError):
        nav_season = None
    return render_page(title, nav_lid, "compare", body, nav_platform, nav_season, description=desc)


# ── Rankings Hub ──────────────────────────────────────────────────────────────

def _rankings_page(position: str | None = None):
    from dashboard_services.pages.dynasty_pages import build_rankings_hub_body
    # Use the shared 15-min cache (it already loads from the DB and applies the
    # same FC-zeroing / rookie processing other pages use). Calling
    # load_current_values_from_db() directly here bypassed the cache on all five
    # ranking routes and could show raw, inconsistent values.
    try:
        value_table = get_model_value_table_cached() or []
    except Exception:
        value_table = []

    from datetime import datetime as _dt
    as_of  = _dt.now().strftime("%B %Y")
    year   = _dt.now().year
    body   = build_rankings_hub_body(value_table, position=position, as_of_date=as_of)
    pos_lbl = f" {position}" if position else ""
    return render_page(
        f"Dynasty{pos_lbl} Rankings {year} | BR Fantasy",
        None, "players", body,
        description=(
            f"Dynasty fantasy football{pos_lbl.lower()} rankings updated {as_of}. "
            f"Real trade values for 1QB and Superflex leagues."
        ),
    )


@seo_pages_bp.route("/rankings/dynasty")
def rankings_dynasty():
    return _rankings_page(None)

@seo_pages_bp.route("/rankings/dynasty-qb")
def rankings_dynasty_qb():
    return _rankings_page("QB")

@seo_pages_bp.route("/rankings/dynasty-rb")
def rankings_dynasty_rb():
    return _rankings_page("RB")

@seo_pages_bp.route("/rankings/dynasty-wr")
def rankings_dynasty_wr():
    return _rankings_page("WR")

@seo_pages_bp.route("/rankings/dynasty-te")
def rankings_dynasty_te():
    return _rankings_page("TE")


# ── Player value pages + guest breakouts/prospects ───────────────────────────

@seo_pages_bp.route("/player/<slug>")
@seo_pages_bp.route("/player/<slug>/trade-value")
def page_player_trade_value(slug: str):
    from dashboard_services.pages.player_page import build_player_page_body, slugify
    from utils.utils import load_relevant_index, load_players_index

    slug_norm = slugify(slug)
    idx = get_player_slug_index()
    pid = idx.get(slug_norm)
    if not pid:
        # Stale/unknown slug (player renamed, retired, or fell out of the value
        # table): consolidate to the rankings hub rather than 404 - these were
        # showing up as Not Found errors in Search Console after slug churn.
        return redirect("/players", code=301)

    # Canonical-slug redirect: if the requested slug isn't the normalized one,
    # or it's the bare /player/<slug> form, send 301 to /player/<slug>/trade-value.
    canonical_path = f"/player/{slug_norm}/trade-value"
    if request.path != canonical_path:
        return redirect(canonical_path, code=301)

    try:
        nfl_state = get_nfl_state() or {}
        season = int(nfl_state.get("season") or datetime.now().year)

        players_index = load_relevant_index() or {}
        meta = players_index.get(pid)
        if not meta:
            meta = (load_players_index() or {}).get(pid) or {}

        table = get_model_value_table_cached() or []
        pv = next((p for p in table if str(p.get("id")) == str(pid)), {})

        # Single sorted ranking by 1QB value, reused for overall rank + neighbors.
        ranked_1qb = sorted(
            [x for x in table
             if x.get("position") not in ("K", "DEF", "PICK") and float(x.get("value") or 0) > 0],
            key=lambda x: float(x.get("value") or 0), reverse=True,
        )
        my_idx = next((i for i, p in enumerate(ranked_1qb) if str(p.get("id")) == str(pid)), None)
        ovr_rank = (my_idx + 1) if my_idx is not None else None

        ranked_sf = sorted(
            [x for x in table
             if x.get("position") not in ("K", "DEF", "PICK") and float(x.get("sf_value") or 0) > 0],
            key=lambda x: float(x.get("sf_value") or 0), reverse=True,
        )
        sf_ovr_rank = next((i + 1 for i, p in enumerate(ranked_sf) if str(p.get("id")) == str(pid)), None)

        # Players with the nearest 1QB value (internal links / discovery)
        from dashboard_services.pages.player_page import slugify as _slugify
        similar_players = []
        if my_idx is not None:
            lo = max(0, my_idx - 3)
            neighbors = ranked_1qb[lo:my_idx] + ranked_1qb[my_idx + 1:my_idx + 4]
            for sp in neighbors:
                sp_name = sp.get("name") or ""
                sp_slug = _slugify(sp_name)
                if sp_slug:
                    similar_players.append({
                        "name": sp_name,
                        "slug": sp_slug,
                        "value": sp.get("value"),
                        "pos_rank_label": sp.get("pos_rank_label"),
                    })

        try:
            history = get_player_value_history(pid, days=365, league_type="1qb", league_size=10)
        except Exception:
            history = []

        name = meta.get("name") or pv.get("name") or "Player"
        pos = str(meta.get("pos") or pv.get("position") or "").upper()
        team = meta.get("team")
        age = age_from_bday(meta.get("bDay")) or pv.get("age") or meta.get("age")

        body = build_player_page_body(
            player_id=pid, name=name, position=pos, team=team, age=age,
            headshot=meta.get("espnHeadshot"),
            value_1qb=pv.get("value"), sf_value=pv.get("sf_value"),
            pos_rank_label=pv.get("pos_rank_label"), ovr_rank=ovr_rank,
            sf_pos_rank_label=pv.get("sf_pos_rank_label"), sf_ovr_rank=sf_ovr_rank,
            ppg=None, value_history=history, season=season,
            similar_players=similar_players,
        )

        _val = pv.get("value")
        _pos_phrase = f"{pos} " if pos else ""
        title = f"{name} Dynasty Trade Value {season} - {_pos_phrase}Rankings | BR Fantasy"
        desc_val = f" Current value: {int(_val)}." if _val else ""
        description = (
            f"{name} {season} fantasy football trade value for dynasty and redraft leagues."
            f"{desc_val} See {name}'s value history chart, positional rank, and recent real "
            f"trades, then run the deal through the trade calculator."
        )
        _img = meta.get("espnHeadshot") or ""
        og_tags = (
            f"<meta property='og:title' content='{html.escape(title)}'>"
            f"<meta property='og:description' content='{html.escape(description)}'>"
            f"<meta property='og:type' content='profile'>"
            + (f"<meta property='og:image' content='{html.escape(_img)}'>" if _img else "")
            + "<meta name='twitter:card' content='summary'>"
        )

        return render_page(title, None, "players", body,
                           description=description, og_tags=og_tags)
    except Exception:
        logger.exception("[player-page] render failed for slug=%s pid=%s", slug, pid)
        # Never 5xx a public page for Googlebot/users: return a valid minimal
        # page (keeps the URL alive for when data returns) instead of a 500.
        _fb_name = html.escape((slug or 'Player').replace('-', ' ').title())
        return render_page(
            f"{_fb_name} Trade Value | BR Fantasy", None, "players",
            f"<div class='static-page'><div class='static-card-page'>"
            f"<h1>{_fb_name}</h1><p>Trade value details are temporarily unavailable. "
            f"<a href='/players'>Browse all player values</a> or "
            f"<a href='/trade'>open the trade calculator</a>.</p></div></div>",
            description=f"{_fb_name} dynasty and redraft trade value.",
        )



@seo_pages_bp.route("/breakouts")
def page_breakouts_guest():
    nfl_state = get_nfl_state() or {}
    current_season = int(nfl_state.get("season") or datetime.now().year)
    return page_breakouts(platform="sleeper", season=current_season, league_id=None)


@seo_pages_bp.route("/prospects")
def page_prospects_guest():
    nfl_state = get_nfl_state() or {}
    current_season = int(nfl_state.get("season") or datetime.now().year)
    return page_prospects(platform="sleeper", season=current_season, league_id=None)
