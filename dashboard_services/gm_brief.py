"""GM brief builder (Site Knowledge Wave 3, PR 1).

A pure, render-free builder that extends the dashboard Next Steps queue
into a weekly per-league brief: the same ranked actions, now carrying
deadline stamps (spec section 3.1), plus a playoff-race line computed
from standings arithmetic on the true playoff field (section 3.2).

The brief joins the layers the earlier waves built:

* Wave 1 rules record (``ctx["_league_rules"]`` summary plus the
  serve-time overlay fields on ``ctx["league_settings"]``, or a full
  record dict on ``ctx["league_rules"]`` when a caller supplies one):
  playoff field, waiver process time, trade deadline, and the source
  label that decides what may be stated as fact.
* The NFL schedule (``load_week_schedule``) for kickoff locks. Kickoff
  stamps work with no rules record at all.
* Wave 2 role state and the partner model land in later PRs; the
  ``layers`` contract reports which inputs were real so renderers can
  degrade honestly (spec section 5).

Fail soft throughout: a missing layer yields fewer stamps, a ``None``
race line, or a confirmation prompt, never an exception and never a
fabricated value. If an element has no data it is absent, not
approximated.
"""
from __future__ import annotations

import time
from datetime import date, datetime, time as _dtime, timedelta

try:
    from zoneinfo import ZoneInfo

    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - zoneinfo is stdlib on 3.9+
    _ET = None

STARTABLE_STATES = frozenset({"starter", "promoted", "returning"})

# Sources allowed to stamp a waiver-process or trade deadline. The
# rules record must actually publish or have been confirmed for these
# values; a derived or defaulted process time is never guessed (3.1).
_STAMP_SOURCES = frozenset({"feed", "user_confirmed"})
# Sources allowed to render the race line on their field (Kaedon's
# call): feed, derived and user_confirmed. Only default_unverified
# suppresses it.
_RACE_SOURCES = frozenset({"feed", "derived", "user_confirmed"})

_RULE_FIELDS = (
    "playoff_teams",
    "playoff_week_start",
    "waiver_type",
    "waiver_process_time",
    "waiver_process_days",
    "waiver_process_tz",
    "trade_deadline",
    "max_keepers",
)

_WEEKDAY_NAMES = {
    "mon": 0, "monday": 0, "tue": 1, "tuesday": 1, "wed": 2,
    "wednesday": 2, "thu": 3, "thursday": 3, "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5, "sun": 6, "sunday": 6,
}


# ---------------------------------------------------------------------------
# small coercion helpers
# ---------------------------------------------------------------------------

def _f(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _tz(name):
    if name:
        try:
            from zoneinfo import ZoneInfo

            return ZoneInfo(str(name))
        except Exception:
            pass
    return _ET


# ---------------------------------------------------------------------------
# ctx extraction: players, rules, standings
# ---------------------------------------------------------------------------

def player_index(ctx) -> dict:
    """Merged ``{pid: info}`` from the ctx player payloads.

    ``players_map`` / ``players_index`` carry name/team/pos shapes; the
    canonical ``players`` payload wins where both speak.
    """
    out: dict = {}
    if not isinstance(ctx, dict):
        return out
    for key in ("players_index", "players_map", "players"):
        src = ctx.get(key)
        if isinstance(src, dict):
            for pid, info in src.items():
                if isinstance(info, dict):
                    merged = dict(out.get(str(pid)) or {})
                    merged.update(info)
                    out[str(pid)] = merged
    return out


def player_info(ctx, pid, _index=None) -> dict:
    index = _index if _index is not None else player_index(ctx)
    return index.get(str(pid)) or {}


def player_name(ctx, pid, _index=None) -> str:
    info = player_info(ctx, pid, _index)
    name = str(info.get("full_name") or info.get("name") or "").strip()
    return name or f"Player {pid}"


def player_team(info: dict) -> str:
    return str((info or {}).get("team") or "").strip().upper()


def player_pos(info: dict) -> str:
    pos = str((info or {}).get("position") or (info or {}).get("pos") or "")
    pos = pos.strip().upper()
    return "DEF" if pos in ("DST", "D/ST") else pos


def player_bye_week(info: dict):
    for key in ("byeWeek", "bye_week", "bye"):
        val = _i((info or {}).get(key))
        if val:
            return val
    return None


def get_rules(ctx) -> dict:
    """The league's rules record as the brief reads it.

    A full record dict supplied on the ctx wins; otherwise the Wave 1
    serve-time overlay is reconstructed from ``ctx["_league_rules"]``
    (source, field, week start) plus ``ctx["league_settings"]`` (the
    overlay merges every record field there). Absent everything, the
    source is ``default_unverified`` and nothing is treated as fact.
    """
    ctx = ctx if isinstance(ctx, dict) else {}
    record: dict = {}
    for key in ("league_rules", "_gm_rules", "rules_record"):
        candidate = ctx.get(key)
        if isinstance(candidate, dict) and candidate:
            record = candidate
            break
    overlay = ctx.get("_league_rules") or {}
    settings = ctx.get("league_settings")
    if not isinstance(settings, dict):
        settings = (ctx.get("league") or {}).get("settings") or {}
    out: dict = {}
    for field in _RULE_FIELDS:
        value = None
        for src in (record, overlay, settings):
            if isinstance(src, dict) and src.get(field) is not None:
                value = src.get(field)
                break
        out[field] = value
    out["playoff_teams"] = _i(out.get("playoff_teams"))
    out["playoff_week_start"] = _i(out.get("playoff_week_start"))
    source = None
    for src in (record, overlay):
        if isinstance(src, dict) and src.get("source"):
            source = str(src["source"])
            break
    out["source"] = source or "default_unverified"
    out["keepers_known"] = bool(
        record.get("keepers_known") or overlay.get("keepers_known"))
    return out


def _roster_team_name(ctx, roster, rid) -> str:
    roster_map = ctx.get("roster_map") or {}
    raw = None
    if isinstance(roster_map, dict):
        raw = roster_map.get(rid) or roster_map.get(str(rid))
        if raw is None:
            try:
                raw = roster_map.get(int(rid))
            except (TypeError, ValueError):
                raw = None
    if isinstance(raw, dict):
        name = raw.get("team_name") or raw.get("name") or raw.get("owner")
        if name:
            return str(name)
    elif raw:
        return str(raw)
    if isinstance(roster, dict):
        for key in ("team_name", "name", "owner"):
            if roster.get(key):
                return str(roster[key])
        owner_id = roster.get("owner_id")
        for user in ctx.get("users") or []:
            if isinstance(user, dict) and str(user.get("user_id")) == str(owner_id):
                name = user.get("display_name") or user.get("username")
                if name:
                    return str(name)
    return f"Roster {rid}"


def _team_stats_records(ctx) -> list:
    ts = ctx.get("team_stats")
    if ts is None:
        return []
    if hasattr(ts, "to_dict"):
        try:
            return [r for r in ts.to_dict("records") if isinstance(r, dict)]
        except Exception:
            return []
    if isinstance(ts, list):
        return [r for r in ts if isinstance(r, dict)]
    return []


def _record_from_settings(settings: dict):
    wins = _f(settings.get("wins"))
    losses = _f(settings.get("losses"))
    ties = _f(settings.get("ties"))
    pf = _f(settings.get("f_points"))
    dec = settings.get("f_points_decimal")
    if dec is not None:
        pf += _f(dec) / 100.0
    return wins, losses, ties, pf


def standings_rows(ctx) -> list:
    """Seeded standings: ``[{roster_id, team_name, wins, losses, ties,
    pf, seed}]`` sorted by win value (ties half) then PF, best first.

    Sources, in order: an explicit standings list on the ctx, a
    dict-shaped ``standings_map``, the ``team_stats`` frame/records
    matched by owner name, then each roster's own settings. A roster
    with no record source contributes a 0-0 row, never an invented one.
    """
    ctx = ctx if isinstance(ctx, dict) else {}
    explicit = ctx.get("gm_standings") or ctx.get("standings")
    rows: list = []
    if isinstance(explicit, list) and explicit:
        for row in explicit:
            if not isinstance(row, dict):
                continue
            rid = row.get("roster_id")
            if rid is None:
                continue
            rows.append({
                "roster_id": str(rid),
                "team_name": str(row.get("team_name") or row.get("name")
                                 or f"Roster {rid}"),
                "wins": _f(row.get("wins")),
                "losses": _f(row.get("losses")),
                "ties": _f(row.get("ties")),
                "pf": _f(row.get("pf", row.get("points_for"))),
            })
    else:
        ts_by_owner = {}
        for rec in _team_stats_records(ctx):
            owner = rec.get("owner") or rec.get("Owner")
            if owner is not None:
                ts_by_owner[str(owner)] = rec
        standings_map = ctx.get("standings_map") or {}
        for roster in ctx.get("rosters") or []:
            if not isinstance(roster, dict):
                continue
            rid = roster.get("roster_id")
            if rid is None:
                continue
            name = _roster_team_name(ctx, roster, rid)
            wins = losses = ties = pf = 0.0
            standing = None
            if isinstance(standings_map, dict):
                standing = (standings_map.get(str(rid))
                            or standings_map.get(rid))
            if isinstance(standing, dict) and (
                    standing.get("wins") is not None
                    or standing.get("pf") is not None
                    or standing.get("PF") is not None):
                wins = _f(standing.get("wins"))
                losses = _f(standing.get("losses"))
                ties = _f(standing.get("ties"))
                pf = _f(standing.get("PF", standing.get("pf")))
            elif name in ts_by_owner:
                rec = ts_by_owner[name]
                wins = _f(rec.get("Wins", rec.get("wins")))
                losses = _f(rec.get("Losses", rec.get("losses")))
                ties = _f(rec.get("Ties", rec.get("ties")))
                pf = _f(rec.get("PF", rec.get("pf")))
            else:
                wins, losses, ties, pf = _record_from_settings(
                    roster.get("settings") or {})
            rows.append({
                "roster_id": str(rid), "team_name": name,
                "wins": wins, "losses": losses, "ties": ties, "pf": pf,
            })
    def _win_value(row):
        return row["wins"] + 0.5 * row["ties"]

    rows.sort(key=lambda r: (-_win_value(r), -r["pf"], r["team_name"]))
    for idx, row in enumerate(rows):
        row["seed"] = idx + 1
    return rows


def _games_back(a: dict, b: dict) -> float:
    """Games between two record rows (leader minus trailer)."""
    return abs(((a["wins"] - b["wins"]) + (b["losses"] - a["losses"])) / 2.0)


def _format_record(row: dict) -> str:
    def _num(v):
        return str(int(v)) if float(v) == int(v) else str(v)

    text = f"{_num(row['wins'])}-{_num(row['losses'])}"
    if row.get("ties"):
        text += f"-{_num(row['ties'])}"
    return text


# ---------------------------------------------------------------------------
# schedule / deadline stamps (spec 3.1)
# ---------------------------------------------------------------------------

def schedule_games(ctx, week=None) -> list:
    """This week's NFL games: an explicit ctx slate wins, else the
    shared schedule loader. Fail-soft to an empty list."""
    ctx = ctx if isinstance(ctx, dict) else {}
    for key in ("gm_schedule_games", "week_schedule", "schedule_games"):
        games = ctx.get(key)
        if isinstance(games, list) and games:
            return games
    try:
        season = _i(ctx.get("current_season") or ctx.get("season"))
        wk = _i(week if week is not None
                else (ctx.get("current_week") or ctx.get("week")))
        if season and wk:
            from utils.data_cache import load_week_schedule

            return load_week_schedule(season, wk) or []
    except Exception:
        pass
    return []


def _game_epoch(game: dict):
    for key in ("gameTime_epoch", "kickoff_epoch", "epoch", "start_epoch"):
        raw = game.get(key)
        if raw is None:
            continue
        val = _f(raw)
        if val:
            return val / 1000.0 if val > 1e12 else val
    return None


def _team_kickoffs(games) -> dict:
    out: dict = {}
    for game in games or []:
        if not isinstance(game, dict):
            continue
        epoch = _game_epoch(game)
        if not epoch:
            continue
        for side in ("home", "away"):
            team = str(game.get(side) or "").strip().upper()
            if team and (team not in out or epoch < out[team]):
                out[team] = epoch
    return out


def _kickoff_label(epoch: float, tz) -> str:
    dt = datetime.fromtimestamp(epoch, tz=tz)
    hour = dt.strftime("%I").lstrip("0") or "12"
    return (f"Locks {dt.strftime('%a')} {hour}:{dt.strftime('%M')} "
            f"{dt.strftime('%p')} ET")


def _parse_process_time(value):
    text = str(value or "").strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        hh = int(parts[0])
        mm = int(parts[1]) if len(parts) > 1 else 0
    except (TypeError, ValueError):
        return None
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None
    return hh, mm


def _process_weekdays(days_value):
    """Weekday ints the waiver process runs on, or None for daily."""
    if days_value is None:
        return None
    if isinstance(days_value, (list, tuple)):
        items = list(days_value)
    else:
        text = str(days_value).strip().lower()
        if text in ("", "daily", "every day", "everyday"):
            return None
        items = [p.strip() for p in text.replace(",", " ").split() if p.strip()]
    weekdays = set()
    for item in items:
        if isinstance(item, int):
            weekdays.add(item % 7)
            continue
        text = str(item).strip().lower()
        if text in _WEEKDAY_NAMES:
            weekdays.add(_WEEKDAY_NAMES[text])
        elif text.isdigit():
            weekdays.add(int(text) % 7)
    return weekdays or None


def next_waiver_process(rules: dict, now: float):
    """Next waiver-process instant from the rules record, or None.

    Only feed / user_confirmed records stamp (spec 3.1): the process
    time is never guessed from a derived or defaulted record.
    """
    if (rules or {}).get("source") not in _STAMP_SOURCES:
        return None
    parsed = _parse_process_time(rules.get("waiver_process_time"))
    if not parsed:
        return None
    hh, mm = parsed
    tz = _tz(rules.get("waiver_process_tz"))
    weekdays = _process_weekdays(rules.get("waiver_process_days"))
    today = datetime.fromtimestamp(now, tz=tz).date()
    for offset in range(0, 9):
        day = today + timedelta(days=offset)
        if weekdays is not None and day.weekday() not in weekdays:
            continue
        candidate = datetime.combine(day, _dtime(hh, mm), tzinfo=tz)
        if candidate.timestamp() > now:
            return candidate.timestamp()
    return None


def waiver_process_label(rules: dict) -> str:
    parsed = _parse_process_time((rules or {}).get("waiver_process_time"))
    if not parsed:
        return ""
    hh, mm = parsed
    suffix = "AM" if hh < 12 else "PM"
    hour12 = hh % 12 or 12
    return f"Waiver process {hour12}:{mm:02d} {suffix} ET"


def _trade_deadline_stamp(rules: dict, now: float):
    """Trade-deadline stamp in the final two weeks before the date."""
    if (rules or {}).get("source") not in _STAMP_SOURCES:
        return None
    raw = rules.get("trade_deadline")
    if not raw:
        return None
    try:
        if isinstance(raw, datetime):
            deadline_date = raw.date()
        elif isinstance(raw, date):
            deadline_date = raw
        else:
            deadline_date = date.fromisoformat(str(raw)[:10])
    except (TypeError, ValueError):
        return None
    tz = _tz(rules.get("waiver_process_tz"))
    today = datetime.fromtimestamp(now, tz=tz).date()
    if not (deadline_date - timedelta(days=14) <= today <= deadline_date):
        return None
    epoch = datetime.combine(deadline_date, _dtime(23, 59), tzinfo=tz)
    label = (f"Before the trade deadline, "
             f"{deadline_date.strftime('%b')} {deadline_date.day}")
    return {"kind": "trade_deadline", "at": epoch.timestamp(), "label": label}


def action_player_ids(action: dict) -> list:
    """Every player id an action touches, in field order.

    Collectors attach ids explicitly (``player_ids`` for lineup swaps
    including the bench hedge replacement, ``player_id`` for claims,
    ``send_ids`` / ``receive_ids`` for trades); the brief never parses
    action prose to find them.
    """
    out: list = []
    seen: set = set()

    def _add(value):
        if value is None:
            return
        if isinstance(value, dict):
            value = value.get("id") or value.get("player_id")
            if value is None:
                return
        pid = str(value)
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)

    for key in ("player_ids", "touched_player_ids", "hedge_player_ids",
                "send_ids", "send_player_ids", "receive_ids",
                "receive_player_ids"):
        value = action.get(key)
        if isinstance(value, (list, tuple)):
            for item in value:
                _add(item)
        elif value is not None:
            _add(value)
    for key in ("player_id", "plugin_player_id", "hedge_player_id",
                "in_player_id", "out_player_id"):
        if action.get(key) is not None:
            _add(action.get(key))
    for key in ("gives", "sends", "gets", "receives", "package"):
        value = action.get(key)
        if isinstance(value, (list, tuple)):
            for item in value:
                _add(item)
    return out


def stamp_deadlines(ctx, actions, rules, now=None, games=None) -> list:
    """Attach ``deadline`` stamps to copies of the queue actions.

    Kickoff: the earliest game among the players an action touches,
    stamped only when that game kicks off before Sunday (a Thursday
    hedge replacement stamps the action even when the starter plays
    Sunday). Waiver actions take the process stamp when the record
    allows it; trade actions take the deadline stamp in its final two
    weeks. Sunday-only actions carry no stamp.
    """
    now = time.time() if now is None else now
    tz = _tz((rules or {}).get("waiver_process_tz"))
    kickoffs = _team_kickoffs(
        games if games is not None else schedule_games(ctx))
    index = player_index(ctx)
    waiver_at = next_waiver_process(rules, now)
    trade_stamp = _trade_deadline_stamp(rules, now)
    out = []
    for action in actions or []:
        if not isinstance(action, dict):
            continue
        item = dict(action)
        if not item.get("deadline"):
            kind = str(item.get("kind") or "")
            stamp = None
            if kind == "waiver" and waiver_at:
                stamp = {
                    "kind": "waiver_process",
                    "at": waiver_at,
                    "label": waiver_process_label(rules),
                }
            elif kind == "trade" and trade_stamp:
                stamp = dict(trade_stamp)
            else:
                epochs = []
                for pid in action_player_ids(item):
                    team = player_team(player_info(ctx, pid, index))
                    if team and team in kickoffs:
                        epochs.append(kickoffs[team])
                if epochs:
                    earliest = min(epochs)
                    weekday = datetime.fromtimestamp(earliest, tz=tz).weekday()
                    if weekday in (3, 4, 5):  # Thu / Fri / Sat locks only
                        stamp = {
                            "kind": "kickoff",
                            "at": earliest,
                            "label": _kickoff_label(earliest, tz),
                        }
            if stamp:
                item["deadline"] = stamp
        out.append(item)
    return out


def sort_actions(actions, now=None) -> list:
    """Score stays primary; an action whose deadline falls within 24
    hours floats above equal-score actions without one (spec 3.1)."""
    now = time.time() if now is None else now

    def _key(action):
        score = _f(action.get("score"))
        deadline = action.get("deadline") or {}
        at = _f(deadline.get("at")) if isinstance(deadline, dict) else 0.0
        soon = 0 if (at and at - now <= 86400) else 1
        return (-score, soon)

    return sorted(actions or [], key=_key)


# ---------------------------------------------------------------------------
# playoff-race line (spec 3.2)
# ---------------------------------------------------------------------------

def build_race_line(ctx, viewer_roster_id, rules=None) -> dict | None:
    """Standings arithmetic on the true field, or None.

    Suppressed entirely when the rules source is default_unverified
    (the brief carries the confirmation prompt instead); the field
    size never comes from a fabricated default.
    """
    rules = rules if rules is not None else get_rules(ctx)
    if rules.get("source") not in _RACE_SOURCES:
        return None
    playoff_teams = rules.get("playoff_teams")
    if not playoff_teams:
        return None
    rows = standings_rows(ctx)
    if len(rows) < 2:
        return None
    viewer = next((r for r in rows
                   if r["roster_id"] == str(viewer_roster_id)), None)
    if viewer is None:
        return None
    seed = viewer["seed"]
    record = _format_record(viewer)
    if seed <= playoff_teams:
        status = "holds"
        games_back = 0.0
        rival = next((r for r in rows if r["seed"] == playoff_teams + 1),
                     None)
    else:
        status = "chasing"
        rival = next((r for r in rows if r["seed"] == playoff_teams), None)
        games_back = _games_back(rival, viewer) if rival else 0.0
    pf_delta = None
    rival_name = None
    if rival is not None:
        pf_delta = round(viewer["pf"] - rival["pf"], 2)
        rival_name = rival["team_name"]
    # Head-to-head remaining vs the teams within one game, from the
    # published matchup slate when the ctx carries it.
    h2h: list = []
    try:
        week = _i(ctx.get("current_week") or ctx.get("week")) or 0
        matchups = ctx.get("matchups_by_week") or {}
        by_rid = {r["roster_id"]: r for r in rows}
        seen = set()
        for wk, slate in (matchups.items() if isinstance(matchups, dict)
                          else []):
            if _i(wk) is None or _i(wk) < week:
                continue
            for matchup in slate or []:
                if not isinstance(matchup, dict):
                    continue
                sides = [matchup.get("left") or {}, matchup.get("right") or {}]
                rids = [str(s.get("roster_id")) for s in sides
                        if isinstance(s, dict) and s.get("roster_id") is not None]
                if str(viewer_roster_id) not in rids or len(rids) != 2:
                    continue
                opp_id = next(r for r in rids if r != str(viewer_roster_id))
                opp = by_rid.get(opp_id)
                if opp is None or opp_id in seen:
                    continue
                if _games_back(viewer, opp) <= 1.0:
                    seen.add(opp_id)
                    h2h.append(opp["team_name"])
    except Exception:
        h2h = []
    # Copy is arithmetic, never sim prose.
    bits = [f"{_ordinal(seed)} seed", record]
    if status == "holds":
        bits.append(f"Holds the {_ordinal(playoff_teams)} seed")
    else:
        gb_text = (str(int(games_back)) if games_back == int(games_back)
                   else f"{games_back:.1f}")
        bits.append(f"{gb_text} GB of the {_ordinal(playoff_teams)} seed")
    if rival_name and pf_delta is not None:
        sign = "+" if pf_delta > 0 else ""
        tiebreak = "tiebreak yours" if pf_delta > 0 else (
            "tiebreak theirs" if pf_delta < 0 else "PF even")
        bits.append(f"{sign}{_fmt_pf(pf_delta)} PF on {rival_name} ({tiebreak})")
    if h2h:
        bits.append(f"{' and '.join(h2h[:2])} still to play")
    return {
        "seed": seed,
        "record": record,
        "wins": viewer["wins"],
        "losses": viewer["losses"],
        "ties": viewer["ties"],
        "pf": viewer["pf"],
        "playoff_teams": playoff_teams,
        "status": status,
        "games_back": games_back,
        "pf_delta": pf_delta,
        "pf_rival": rival_name,
        "h2h_remaining": h2h,
        "source": rules.get("source"),
        "text": " · ".join(bits),
    }


def _ordinal(n) -> str:
    n = int(n)
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _fmt_pf(value) -> str:
    val = float(value)
    if val == int(val):
        return str(int(val))
    return f"{val:.2f}".rstrip("0").rstrip(".")


def rules_prompt(rules: dict) -> dict | None:
    """The Wave 1 confirmation prompt that rides along until the
    record is user_confirmed (it replaces the race line outright when
    the source is default_unverified)."""
    if (rules or {}).get("source") == "user_confirmed":
        return None
    return {
        "kind": "rules_confirmation",
        "source": (rules or {}).get("source"),
        "text": ("Playoff format not confirmed for this league. "
                 "Confirm it on the Standings page."),
    }


# ---------------------------------------------------------------------------
# the brief (PR 1 shape; coverage / bye / claims land in PRs 2-3)
# ---------------------------------------------------------------------------

def build_gm_brief(ctx, viewer_roster_id, actions=None, now=None) -> dict:
    """Build the render-free GM brief dict for one viewer.

    ``actions`` are the Next Steps collectors' items; when omitted,
    ``ctx["gm_actions"]`` is used (fixture and tool callers). Actions
    are copied, deadline-stamped, and ranked per spec 3.1.
    """
    ctx = ctx if isinstance(ctx, dict) else {}
    now = time.time() if now is None else now
    rules = get_rules(ctx)
    if actions is None:
        actions = ctx.get("gm_actions") or []
    stamped = stamp_deadlines(ctx, list(actions or []), rules, now=now)
    ranked = sort_actions(stamped, now=now)
    race_line = None
    try:
        race_line = build_race_line(ctx, viewer_roster_id, rules)
    except Exception:
        race_line = None
    week = _i(ctx.get("current_week") or ctx.get("week")) or 0
    return {
        "week": week,
        "actions": ranked,
        "race_line": race_line,
        "race_prompt": rules_prompt(rules),
        "qb_coverage": None,
        "bye_panel": None,
        "claims": [],
        "layers": {
            "rules": rules.get("source"),
            "roles": False,
            "partner_model": False,
        },
    }
