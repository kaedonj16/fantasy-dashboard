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
* ``sleeper``: fully supported via ``contribute_user_leagues`` (the original
  path). ``trade_intel_users.user_id`` values are Sleeper user ids, Sleeper's
  user-leagues and transactions endpoints are public (no credentials), and
  Sleeper player ids are already the canonical ids used throughout the intel
  tables. Normalization reuses ``trade_crawler._extract_assets`` so
  contributed rows match crawler rows.
* ``espn`` / ``yahoo`` / ``mfl`` / ``fleaflicker``: supported via
  ``contribute_account_leagues`` (the account path). Linked leagues are
  enumerated from ``dashboard_services.accounts.list_user_leagues`` keyed by
  internal ``account_id`` -- NOT from a Sleeper ``user_id``, which cannot
  resolve other platforms' credentials headless. All four adapters already
  return the unified Sleeper-like transaction shape with canonical (Sleeper)
  player ids through ``platform_api.get_transactions``, and credentials
  resolve inside the adapters (Yahoo stored league token, ESPN stored league
  cookies, MFL/Fleaflicker credential resolution; public leagues need no
  auth). Headless credential failures fail open per league. ESPN and
  Fleaflicker are fetched per week; Yahoo and MFL return the whole season in
  one call, so their trade weeks are derived from the trade timestamp.
  Non-Sleeper league rows are stored under ``"{provider}:{league_id}"`` so
  numeric ids from different platforms cannot collide with each other or
  with Sleeper snowflakes.

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
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

from dashboard_services.db import get_conn

logger = logging.getLogger(__name__)

# transaction_id prefix and trade_intel_assets.provider value per platform.
PROVIDER_SLEEPER = "sleeper"

# Platforms reachable through linked (connected) leagues on an account.
ACCOUNT_PROVIDERS = ("espn", "yahoo", "mfl", "fleaflicker")

# Adapters that return the WHOLE SEASON on every get_transactions call
# (the week param is ignored / stamped, not a filter). Fetch once and
# derive each trade's week from its timestamp instead of looping weeks --
# looping would re-insert the same trades under wrong weeks (deduped only
# by transaction_id, keeping the first, wrong week).
_SEASON_WIDE_PROVIDERS = frozenset({"yahoo", "mfl"})

_ET = ZoneInfo("America/New_York")


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


def _stored_league_id(provider: str, league_id: str) -> str:
    """trade_intel_leagues id for a contributed league.

    Sleeper keeps its raw snowflake id (back-compat with the crawler).
    Every other provider is namespaced ``"{provider}:{raw_id}"`` because
    numeric ESPN/Yahoo/MFL/Fleaflicker ids can collide with each other.
    """
    league_id = str(league_id)
    if provider == PROVIDER_SLEEPER or league_id.startswith(f"{provider}:"):
        return league_id
    return f"{provider}:{league_id}"


def _ensure_league_row(
    league_id: str,
    season: int,
    league: Optional[Dict[str, Any]] = None,
    *,
    provider: str = PROVIDER_SLEEPER,
    num_teams: Optional[int] = None,
    league_type: Optional[int] = None,
) -> str:
    """Insert the trade_intel_leagues row the trades FK requires.

    crawl_enabled=FALSE keeps the public crawler from re-ingesting the same
    trades under unprefixed transaction ids. Returns the stored league id
    (namespaced for non-Sleeper providers -- see ``_stored_league_id``).
    Explicit ``num_teams`` / ``league_type`` win over the league dict.
    """
    league = league or {}
    settings = league.get("settings") or {}
    stored_id = _stored_league_id(provider, league_id)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO trade_intel_leagues
                (league_id, season, num_teams, league_type, crawl_enabled)
            VALUES (%s, %s, %s, %s, FALSE)
            ON CONFLICT (league_id) DO NOTHING
            """,
            (
                stored_id,
                season,
                num_teams if num_teams is not None else league.get("total_rosters"),
                league_type if league_type is not None else settings.get("type"),
            ),
        )
    return stored_id


def _created_ms(txn: Dict[str, Any]) -> Optional[int]:
    """The trade's created timestamp in epoch milliseconds, or None.

    Adapters emit ms, but normalize defensively: a positive value below
    1e11 can only be epoch SECONDS (1e11 ms = 1973; 1e11 s = year 5138),
    so scale it up instead of stamping a 1970 created_at.
    """
    created = txn.get("created")
    if not created:
        return None
    try:
        created = int(created)
    except (TypeError, ValueError):
        return None
    if 0 < created < 100_000_000_000:
        created *= 1000
    return created


def _insert_trade(
    conn,
    league_id: str,
    season: int,
    week: int,
    txn: Dict[str, Any],
    assets: List[Dict[str, Any]],
    provider: str = PROVIDER_SLEEPER,
) -> bool:
    """Insert one trade + assets. Returns True when the trade was new.

    ``league_id`` must be the STORED id (see ``_stored_league_id``);
    ``provider`` drives the transaction_id prefix and the assets'
    provider column.
    """
    raw_id = str(txn.get("transaction_id", ""))
    if not raw_id:
        return False
    transaction_id = f"{provider}:{raw_id}"

    created_ms = _created_ms(txn)
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
                _check_player_identity(provider, str(a["player_id"]))
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
                a.get("pick_slot"), provider,
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


# ---------------------------------------------------------------------------
# Connected-league (non-Sleeper) contributor
# ---------------------------------------------------------------------------

def _week_from_ms(ts_ms: Optional[int], season: int) -> Optional[int]:
    """Map an epoch-milli timestamp onto NFL regular-season week 1-18.

    Same Labor-Day anchor as ``fleaflicker_api._fantasy_week_from_ms``:
    fantasy weeks start the Tuesday after Labor Day (first Monday in
    September). Activity from the two months before week 1 lands in week 1;
    anything older, or more than 22 weeks out, returns None.
    """
    if not ts_ms:
        return None
    try:
        when = datetime.fromtimestamp(ts_ms / 1000.0, tz=_ET).date()
        year = int(season)
    except (OSError, OverflowError, TypeError, ValueError):
        return None
    sep1 = date(year, 9, 1)
    labor = sep1 + timedelta(days=(0 - sep1.weekday()) % 7)
    week1_tue = labor + timedelta(days=1)
    delta = (when - week1_tue).days
    if delta < -90:
        return None
    if delta < 0:
        return 1
    week = delta // 7 + 1
    if week > 22:
        return None
    return min(18, week)


def _contribute_platform_league(
    provider: str,
    raw_league_id: str,
    season: int,
    num_teams: Optional[int] = None,
    league_type: Optional[int] = None,
) -> int:
    """Contribute one connected ESPN/Yahoo/MFL/Fleaflicker league's trades.

    Returns the number of NEW trades stored. Adapters already return the
    unified Sleeper-like shape with canonical player ids, so
    ``trade_crawler._extract_assets`` consumes them directly (no slot map:
    non-Sleeper pick slots are unknown). Per-week and per-trade failures are
    logged and skipped, never raised -- a league whose stored credentials
    have gone stale must not sink the rest of the account's leagues.
    """
    from dashboard_services import platform_api
    from data_building.trade_intel.trade_crawler import (
        _current_nfl_week,
        _extract_assets,
    )

    provider = str(provider).lower()
    raw_league_id = str(raw_league_id)
    season = int(season)

    # League metadata for the FK row: fail-open to the caller's values.
    if num_teams is None or league_type is None:
        try:
            info = platform_api.get_league(provider, raw_league_id, season) or {}
        except Exception as exc:
            logger.debug(
                "[contrib] %s league %s info fetch failed: %s",
                provider, raw_league_id, exc,
            )
            info = {}
        if isinstance(info, dict):
            if num_teams is None:
                num_teams = info.get("total_rosters")
            if league_type is None:
                league_type = (info.get("settings") or {}).get("type")

    stored_id = _ensure_league_row(
        raw_league_id, season, provider=provider,
        num_teams=num_teams, league_type=league_type,
    )

    # Collect (week, txn) pairs. Season-wide adapters get ONE fetch with
    # timestamp-derived weeks; per-week adapters loop the season's weeks.
    week_txns: List[Any] = []
    if provider in _SEASON_WIDE_PROVIDERS:
        try:
            txns = platform_api.get_transactions(
                provider, raw_league_id, 1, season
            ) or []
        except Exception as exc:
            logger.warning(
                "[contrib] %s league %s transactions fetch failed: %s",
                provider, raw_league_id, exc,
            )
            txns = []
        for txn in txns:
            if not isinstance(txn, dict):
                continue
            week = _week_from_ms(_created_ms(txn), season)
            if week is None:
                continue
            week_txns.append((week, txn))
    else:
        current_week = _current_nfl_week()
        for week in range(1, current_week + 1):
            try:
                txns = platform_api.get_transactions(
                    provider, raw_league_id, week, season
                ) or []
            except Exception as exc:
                logger.debug(
                    "[contrib] %s league %s week %d fetch failed: %s",
                    provider, raw_league_id, week, exc,
                )
                continue
            for txn in txns:
                if isinstance(txn, dict):
                    week_txns.append((week, txn))

    new_trades = 0
    with get_conn() as conn:
        for week, txn in week_txns:
            if txn.get("type") != "trade" or txn.get("status") != "complete":
                continue
            try:
                assets = _extract_assets(txn)
                if not assets:
                    continue
                if _insert_trade(
                    conn, stored_id, season, week, txn, assets,
                    provider=provider,
                ):
                    new_trades += 1
            except Exception as exc:
                logger.debug(
                    "[contrib] %s league %s txn insert failed: %s",
                    provider, raw_league_id, exc,
                )
                continue
    return new_trades


# ---------------------------------------------------------------------------
# Account opt-in + connected-league contribution
# ---------------------------------------------------------------------------

_ACCOUNT_TABLE_READY = False


def _ensure_account_table() -> None:
    """Create the account opt-in table once per process (accounts.py pattern)."""
    global _ACCOUNT_TABLE_READY
    if _ACCOUNT_TABLE_READY:
        return
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_intel_contrib_accounts (
                account_id BIGINT PRIMARY KEY,
                opted_in   BOOLEAN NOT NULL DEFAULT FALSE,
                updated_at TIMESTAMP DEFAULT now()
            )
            """
        )
    _ACCOUNT_TABLE_READY = True


def set_account_opt_in(account_id: int, opted_in: bool) -> None:
    """Set the trade-contribution consent flag for an internal account."""
    _ensure_account_table()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO trade_intel_contrib_accounts (account_id, opted_in, updated_at)
            VALUES (%s, %s, now())
            ON CONFLICT (account_id) DO UPDATE
                SET opted_in = EXCLUDED.opted_in, updated_at = now()
            """,
            (int(account_id), bool(opted_in)),
        )
    logger.info("[contrib] account_id=%s opt_in=%s", account_id, bool(opted_in))


def is_account_opted_in(account_id: int) -> bool:
    """True when the account has explicitly opted in to contribution."""
    _ensure_account_table()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT opted_in FROM trade_intel_contrib_accounts WHERE account_id = %s",
            (int(account_id),),
        ).fetchone()
    return bool(row and row.get("opted_in"))


def contribute_account_leagues(
    account_id: int,
    season: Optional[int] = None,
    providers: Iterable[str] = ACCOUNT_PROVIDERS,
) -> Dict[str, Any]:
    """Contribute an opted-in account's connected leagues' trades.

    Enumerates the account's linked leagues via
    ``accounts.list_user_leagues`` and contributes each league whose
    platform is in ``providers``. Returns {"opted_in": bool, "leagues": N,
    "trades": M, "skipped_providers": [...]}. Not opted in -> does nothing.
    """
    account_id = int(account_id)
    if not is_account_opted_in(account_id):
        logger.info(
            "[contrib] account_id=%s has not opted in - skipping", account_id
        )
        return {"opted_in": False, "leagues": 0, "trades": 0,
                "skipped_providers": []}

    if season is None:
        from data_building.trade_intel.league_discovery import _current_season
        season = _current_season()

    from dashboard_services import accounts

    wanted = {str(p).lower() for p in providers}
    skipped = sorted(wanted - set(ACCOUNT_PROVIDERS) - {PROVIDER_SLEEPER})
    for provider in skipped:
        logger.warning(
            "[contrib] provider '%s' is not supported for account contribution "
            "- skipping", provider,
        )

    leagues = 0
    trades = 0
    for row in accounts.list_user_leagues(account_id) or []:
        platform = str(row.get("platform") or "").lower()
        if platform not in wanted or platform not in ACCOUNT_PROVIDERS:
            continue
        raw_league_id = str(row.get("league_id") or "")
        if not raw_league_id:
            continue
        row_season = row.get("season") or season
        try:
            row_season = int(row_season)
        except (TypeError, ValueError):
            row_season = int(season)
        try:
            n = _contribute_platform_league(platform, raw_league_id, row_season)
            leagues += 1
            trades += n
            logger.info(
                "[contrib] %s league %s: %d new trade(s)",
                platform, raw_league_id, n,
            )
        except Exception as exc:
            logger.warning(
                "[contrib] %s league %s failed: %s",
                platform, raw_league_id, exc,
            )
            continue
    logger.info(
        "[contrib] account_id=%s done: %d league(s), %d new trade(s)",
        account_id, leagues, trades,
    )
    return {
        "opted_in": True,
        "leagues": leagues,
        "trades": trades,
        "skipped_providers": skipped,
    }


# Provider name -> contributor(user_id, season) -> {"leagues": int, "trades": int}.
# Sleeper only: it is the one platform resolvable from a bare user id.
# ESPN/Yahoo/MFL/Fleaflicker go through contribute_account_leagues instead.
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
