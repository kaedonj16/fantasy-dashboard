"""
Opt-in multi-platform trade contribution for the Trade Intelligence Engine.

Backend only: no UI, no cron, nothing runs unless a user explicitly opts in
(via ``set_opt_in`` / ``scripts/contribute_trades.py --opt-in``).

Flow
----
``contribute_user_leagues(user_id)`` gates on the ``contrib_opt_in`` flag on
``trade_intel_users``, enumerates the user's leagues, pulls completed trades
for each league, and stores them in ``trade_intel_trades`` /
``trade_intel_assets``. Idempotent: ``transaction_id`` carries a
``{provider}:`` prefix and inserts use ``ON CONFLICT DO NOTHING``, exactly
like the Sleeper crawler in ``trade_crawler.py``.

Platform support (as of this writing)
-------------------------------------
* ``sleeper``: fully supported. ``trade_intel_users.user_id`` values are
  Sleeper user ids, Sleeper's user-leagues and transactions endpoints are
  public (no credentials), and Sleeper player ids are already the canonical
  ids used throughout the intel tables. Normalization reuses
  ``trade_crawler._extract_assets`` so contributed rows match crawler rows.
* ``espn`` / ``yahoo`` / ``mfl`` / ``fleaflicker``: NOT supported yet.
  ``dashboard_services.platform_api`` exposes per-league fetches
  (``get_transactions``) but has no user-scoped "list this user's leagues"
  operation, and the provider registry contract has no such capability either.
  Enumerating a user's non-Sleeper leagues would require their platform
  credentials (ESPN cookies, Yahoo OAuth guid token resolved through a Flask
  session, MFL/Fleaflicker keys), which are keyed by internal ``account_id``
  and are not resolvable from a Sleeper ``user_id`` in a headless context.
  Each platform also returns a different transaction shape needing its own
  normalizer and player-id crosswalk. The ``_CONTRIBUTORS`` registry below is
  the extension point: add ``"<platform>": <contributor_fn>`` when a
  credential path and normalizer exist. Until then, requesting one logs a
  warning and is skipped.

Notes
-----
* Contributed league rows are inserted into ``trade_intel_leagues`` first
  (``trade_intel_trades.league_id`` is a foreign key) with
  ``crawl_enabled = FALSE`` so the public discovery crawler does not
  re-crawl the same trades with unprefixed ids and double-count them in
  market aggregates.
* Player-id mapping is best-effort via ``utils.player_identity`` against the
  local cached players index. Unmapped ids are logged at warning level but the
  rows are still stored tagged with their provider -- data is never dropped.
* ``trade_time_values`` snapshots are left to the existing analytics jobs;
  this module only writes the raw trade/asset rows.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

from dashboard_services.db import get_conn

logger = logging.getLogger(__name__)

# transaction_id prefix and trade_intel_assets.provider value per platform.
PROVIDER_SLEEPER = "sleeper"


# ---------------------------------------------------------------------------
# Opt-in flag
# ---------------------------------------------------------------------------

def set_opt_in(user_id: str, opted_in: bool) -> None:
    """Set the contribution consent flag for a trade_intel_users row.

    UPDATE-first: if the user has never been seeded into trade_intel_users
    (no login/discovery row yet), the row is created with source='contrib'
    so the opt-in is not silently lost.
    """
    user_id = str(user_id)
    with get_conn() as conn:
        updated = conn.execute(
            "UPDATE trade_intel_users SET contrib_opt_in = %s WHERE user_id = %s",
            (bool(opted_in), user_id),
        ).rowcount
        if not updated:
            conn.execute(
                """
                INSERT INTO trade_intel_users (user_id, contrib_opt_in, source)
                VALUES (%s, %s, 'contrib')
                ON CONFLICT (user_id) DO UPDATE
                    SET contrib_opt_in = EXCLUDED.contrib_opt_in
                """,
                (user_id, bool(opted_in)),
            )
    logger.info("[contrib] user_id=%s opt_in=%s", user_id, bool(opted_in))


def is_opted_in(user_id: str) -> bool:
    """True when the user has explicitly opted in to trade contribution."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT contrib_opt_in FROM trade_intel_users WHERE user_id = %s",
            (str(user_id),),
        ).fetchone()
    return bool(row and row.get("contrib_opt_in"))


# ---------------------------------------------------------------------------
# Player identity (best-effort)
# ---------------------------------------------------------------------------

_resolver = None  # lazy PlayerIdentityResolver | False when unavailable


def _identity_resolver():
    """PlayerIdentityResolver over the local cached players index, or None."""
    global _resolver
    if _resolver is None:
        try:
            from utils.player_identity import PlayerIdentityResolver
            from utils.utils import load_players_index
            players = load_players_index()
            _resolver = PlayerIdentityResolver(players) if players else False
        except Exception as exc:
            logger.debug("[contrib] player identity resolver unavailable: %s", exc)
            _resolver = False
    return _resolver or None


def _check_player_identity(provider: str, player_id: str) -> None:
    """Best-effort identity check; logs unmapped ids, never drops the row."""
    resolver = _identity_resolver()
    if resolver is None:
        return
    try:
        result = resolver.resolve(provider=provider, provider_player_id=player_id)
    except Exception as exc:
        logger.debug("[contrib] identity resolve failed for %s: %s", player_id, exc)
        return
    if not result.get("canonical_player_id"):
        logger.warning(
            "[contrib] unmapped %s player id %s (method=%s) - storing raw",
            provider, player_id, result.get("resolution_method"),
        )


# ---------------------------------------------------------------------------
# Sleeper contributor
# ---------------------------------------------------------------------------

def _sleeper_user_leagues(user_id: str, season: int) -> List[Dict[str, Any]]:
    """Public Sleeper endpoint: the user's leagues for a season. No creds."""
    from dashboard_services.api import get_sleeper_user_leagues
    return get_sleeper_user_leagues(str(user_id), int(season)) or []


def _ensure_league_row(league_id: str, season: int, league: Dict[str, Any]) -> None:
    """Insert the trade_intel_leagues row the trades FK requires.

    crawl_enabled=FALSE keeps the public crawler from re-ingesting the same
    trades under unprefixed transaction ids.
    """
    settings = league.get("settings") or {}
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO trade_intel_leagues
                (league_id, season, num_teams, league_type, crawl_enabled)
            VALUES (%s, %s, %s, %s, FALSE)
            ON CONFLICT (league_id) DO NOTHING
            """,
            (
                league_id,
                season,
                league.get("total_rosters"),
                settings.get("type"),
            ),
        )


def _insert_trade(
    conn,
    league_id: str,
    season: int,
    week: int,
    txn: Dict[str, Any],
    assets: List[Dict[str, Any]],
) -> bool:
    """Insert one trade + assets. Returns True when the trade was new."""
    raw_id = str(txn.get("transaction_id", ""))
    if not raw_id:
        return False
    transaction_id = f"{PROVIDER_SLEEPER}:{raw_id}"

    created_ms = txn.get("created")
    created_at = (
        datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc)
        if created_ms else None
    )

    row = conn.execute(
        """
        INSERT INTO trade_intel_trades
            (league_id, transaction_id, season, week, status, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (transaction_id) DO NOTHING
        RETURNING id
        """,
        (league_id, transaction_id, season, week,
         txn.get("status", "complete"), created_at),
    ).fetchone()
    if not row:
        return False

    trade_db_id = row["id"]
    if assets:
        for a in assets:
            if a.get("asset_type") == "player" and a.get("player_id"):
                _check_player_identity(PROVIDER_SLEEPER, str(a["player_id"]))
        conn.execute(
            """
            INSERT INTO trade_intel_assets
                (trade_id, side, asset_type, player_id,
                 pick_season, pick_round, pick_order,
                 pick_roster_id, pick_slot, provider)
            VALUES """
            + ",".join(["(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"] * len(assets)),
            [v for a in assets for v in (
                trade_db_id, a["side"], a["asset_type"], a["player_id"],
                a["pick_season"], a["pick_round"],
                a["pick_order"], a.get("pick_roster_id"),
                a.get("pick_slot"), PROVIDER_SLEEPER,
            )],
        )
    return True


def _contribute_sleeper_league(
    league: Dict[str, Any], season: int, current_week: int
) -> int:
    """Contribute one Sleeper league's completed trades. Returns new trades."""
    from dashboard_services import platform_api
    from data_building.trade_intel.trade_crawler import (
        _extract_assets,
        _fetch_draft_slot_map,
    )

    league_id = str(league.get("league_id", ""))
    if not league_id:
        return 0
    lg_season = int(league.get("season") or season)

    _ensure_league_row(league_id, lg_season, league)
    slot_map = _fetch_draft_slot_map(league_id)

    new_trades = 0
    with get_conn() as conn:
        for week in range(1, current_week + 1):
            try:
                txns = platform_api.get_transactions(
                    PROVIDER_SLEEPER, league_id, week, lg_season
                ) or []
            except Exception as exc:
                logger.debug(
                    "[contrib] sleeper league %s week %d fetch failed: %s",
                    league_id, week, exc,
                )
                continue
            for txn in txns:
                if not isinstance(txn, dict):
                    continue
                if txn.get("type") != "trade" or txn.get("status") != "complete":
                    continue
                try:
                    assets = _extract_assets(txn, slot_map=slot_map)
                    if _insert_trade(conn, league_id, lg_season, week, txn, assets):
                        new_trades += 1
                except Exception as exc:
                    logger.debug(
                        "[contrib] sleeper league %s txn insert failed: %s",
                        league_id, exc,
                    )
                    continue
    return new_trades


def _contribute_sleeper(user_id: str, season: int) -> Dict[str, int]:
    """Contribute all of a user's Sleeper leagues. Returns leagues/trades."""
    from data_building.trade_intel.trade_crawler import _current_nfl_week

    leagues = _sleeper_user_leagues(user_id, season)
    if not leagues:
        logger.info("[contrib] no Sleeper leagues found for user_id=%s", user_id)
        return {"leagues": 0, "trades": 0}

    current_week = _current_nfl_week()
    ok_leagues = 0
    new_trades = 0
    for league in leagues:
        lid = str(league.get("league_id", "?"))
        try:
            n = _contribute_sleeper_league(league, season, current_week)
            ok_leagues += 1
            new_trades += n
            logger.info("[contrib] sleeper league %s: %d new trade(s)", lid, n)
        except Exception as exc:
            logger.warning("[contrib] sleeper league %s failed: %s", lid, exc)
            continue
    return {"leagues": ok_leagues, "trades": new_trades}


# Provider name -> contributor(user_id, season) -> {"leagues": int, "trades": int}.
# See module docstring for why only Sleeper is registered.
_CONTRIBUTORS: Dict[str, Callable[[str, int], Dict[str, int]]] = {
    PROVIDER_SLEEPER: _contribute_sleeper,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def contribute_user_leagues(
    user_id: str,
    season: Optional[int] = None,
    providers: Iterable[str] = (PROVIDER_SLEEPER,),
) -> Dict[str, Any]:
    """Contribute an opted-in user's leagues' trades to the intel pool.

    Returns {"opted_in": bool, "leagues": N, "trades": M,
             "skipped_providers": [...]}. Per-league and per-week failures
    are logged and skipped; an unsupported provider is logged and skipped.
    """
    user_id = str(user_id)
    if not is_opted_in(user_id):
        logger.info(
            "[contrib] user_id=%s has not opted in - skipping", user_id
        )
        return {"opted_in": False, "trades": 0, "skipped_providers": []}

    if season is None:
        from data_building.trade_intel.league_discovery import _current_season
        season = _current_season()

    leagues = 0
    trades = 0
    skipped: List[str] = []
    for provider in providers:
        contributor = _CONTRIBUTORS.get(str(provider).lower())
        if contributor is None:
            logger.warning(
                "[contrib] provider '%s' is not supported for contribution yet - "
                "skipping (see contrib.py docstring)",
                provider,
            )
            skipped.append(str(provider))
            continue
        try:
            result = contributor(user_id, season)
        except Exception as exc:
            logger.warning(
                "[contrib] provider '%s' failed for user_id=%s: %s",
                provider, user_id, exc,
            )
            continue
        leagues += int(result.get("leagues", 0))
        trades += int(result.get("trades", 0))

    logger.info(
        "[contrib] user_id=%s done: %d league(s), %d new trade(s)",
        user_id, leagues, trades,
    )
    return {
        "opted_in": True,
        "leagues": leagues,
        "trades": trades,
        "skipped_providers": skipped,
    }
