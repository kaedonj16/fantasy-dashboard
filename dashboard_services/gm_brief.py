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
    claims = []
    try:
        claims = build_claims(ctx, viewer_roster_id, ranked, rules)
    except Exception:
        claims = []
    try:
        ranked = apply_sequencing(ctx, ranked, claims, now=now)
    except Exception:
        pass
    coverage = None
    roles_present = False
    try:
        coverage, roles_present = build_coverage(
            ctx, viewer_roster_id, rules, ranked)
    except Exception:
        coverage, roles_present = None, False
    bye_panel = None
    try:
        bye_panel = build_bye_panel(ctx, viewer_roster_id, rules)
    except Exception:
        bye_panel = None
    partner_model = any(
        isinstance(a, dict) and a.get("kind") == "trade"
        and (a.get("partner") or a.get("acceptance_rate") is not None)
        for a in ranked
    )
    return {
        "week": week,
        "actions": ranked,
        "race_line": race_line,
        "race_prompt": rules_prompt(rules),
        "qb_coverage": coverage,
        "bye_panel": bye_panel,
        "claims": claims,
        "layers": {
            "rules": rules.get("source"),
            "roles": roles_present,
            "partner_model": partner_model,
        },
    }


# ---------------------------------------------------------------------------
# claims block (spec 3.5, PR 3)
# ---------------------------------------------------------------------------

def _values_by_pid(ctx) -> dict:
    out: dict = {}
    rows = ctx.get("model_value_table") or ctx.get("values") or []
    if isinstance(rows, dict):
        rows = [{"id": k, "value": v} for k, v in rows.items()]
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        pid = row.get("id") or row.get("player_id")
        if pid is None:
            continue
        value = None
        for key in ("value", "sf_value", "value_sf", "model_value",
                    "trade_value"):
            if row.get(key) is not None:
                value = _f(row.get(key))
                break
        if value is not None:
            out[str(pid)] = value
    return out


def waiver_priority(ctx, viewer_roster_id, rows=None):
    """The viewer's waiver priority: roster settings first (Sleeper
    carries ``waiver_position``), then an explicit waiver order, then
    reverse standings seed. None when no source names it."""
    roster = _viewer_roster(ctx, viewer_roster_id)
    if roster is not None:
        pos = _i((roster.get("settings") or {}).get("waiver_position"))
        if pos:
            return pos
    order = ctx.get("waiver_order") or ctx.get("gm_waiver_order")
    if isinstance(order, (list, tuple)):
        ids = [str(x) for x in order]
        if str(viewer_roster_id) in ids:
            return ids.index(str(viewer_roster_id)) + 1
    rows = rows if rows is not None else standings_rows(ctx)
    viewer = next((r for r in rows
                   if r["roster_id"] == str(viewer_roster_id)), None)
    if viewer and rows:
        return len(rows) - viewer["seed"] + 1
    return None


def _keeper_locked_ids(ctx) -> set:
    locked: set = set()
    explicit = ctx.get("gm_keeper_locked") or ctx.get("keeper_locked")
    if explicit:
        locked.update(str(p) for p in explicit)
    try:
        from utils.league_rules import keeper_locked_by_roster

        for ids in keeper_locked_by_roster(ctx).values():
            locked.update(str(p) for p in ids)
    except Exception:
        pass
    return locked


def _funding_sources(ctx, viewer_roster_id, index):
    """(ir_moves, drop_candidates) for funding claims, in order.

    IR moves come from the shared roster-compliance machinery (the
    Harvey-to-IR move); drops are the viewer's lowest-value droppable
    bench players at any position, using the same value table the
    waiver gate's drop-bar machinery reads. Starters, IR/taxi
    residents, seriously-out players and designated keepers are never
    named as drops.
    """
    roster = _viewer_roster(ctx, viewer_roster_id)
    if roster is None:
        return [], []
    pids = [str(p) for p in (roster.get("players") or [])]
    starters = {str(p) for p in (roster.get("starters") or [])}
    reserve = {str(p) for p in (roster.get("reserve") or [])}
    taxi = {str(p) for p in (roster.get("taxi") or [])}
    ir_moves = []
    try:
        from utils.lineups import roster_compliance_issues

        league = ctx.get("league") or {}
        settings = league.get("settings") or {}
        reserve_slots = _i(settings.get("reserve_slots"))
        if reserve_slots is None:
            reserve_slots = _i((ctx.get("league_settings") or {}).get(
                "reserve_slots")) or 0
        taxi_slots = _i(settings.get("taxi_slots")) or 0
        if reserve_slots or taxi_slots:
            info_by_pid = {}
            for pid in pids:
                info = player_info(ctx, pid, index)
                info_by_pid[pid] = {
                    "name": player_name(ctx, pid, index),
                    "injury_status": info.get("injury_status") or "",
                    "years_exp": info.get("years_exp"),
                }
            issues = roster_compliance_issues(
                players=pids, starters=sorted(starters),
                reserve=sorted(reserve), taxi=sorted(taxi),
                player_info=info_by_pid, reserve_slots=reserve_slots,
                taxi_slots=taxi_slots,
                current_week=_i(ctx.get("current_week")) or 0,
            )
            for issue in issues or []:
                if issue.get("kind") == "ir_stash":
                    ir_moves.append({
                        "kind": "ir_move",
                        "player_id": str(issue.get("pid") or ""),
                        "name": issue.get("name") or "",
                        "detail": issue.get("detail") or "",
                    })
    except Exception:
        ir_moves = []
    values = _values_by_pid(ctx)
    locked = _keeper_locked_ids(ctx)
    candidates = []
    for pid in pids:
        if pid in starters or pid in reserve or pid in taxi:
            continue
        if pid in locked:
            continue
        info = player_info(ctx, pid, index)
        status = str(info.get("injury_status")
                     or info.get("status") or "").strip().upper()
        if status in _OUT_STATUSES:
            continue
        if pid not in values:
            continue
        candidates.append({
            "kind": "drop",
            "player_id": pid,
            "name": player_name(ctx, pid, index),
            "value": values[pid],
            "detail": f"Drop {player_name(ctx, pid, index)}",
        })
    candidates.sort(key=lambda c: (c["value"], c["name"]))
    return ir_moves, candidates


def build_claims(ctx, viewer_roster_id, actions, rules=None) -> list:
    """The claims subset of the queue, grouped in waiver-selector
    order (spec 3.5): each claim stamped with tonight's process time
    and the viewer's waiver priority, and carrying its funding move
    (IR move first, then the named drop). A claim with no legal
    funding move carries its blocker named instead of a Claim CTA.
    Funding state is consumed in claim order, so two claims never
    spend the same roster spot twice.
    """
    ctx = ctx if isinstance(ctx, dict) else {}
    rules = rules if rules is not None else get_rules(ctx)
    index = player_index(ctx)
    priority = waiver_priority(ctx, viewer_roster_id)
    ir_moves, drops = _funding_sources(ctx, viewer_roster_id, index)
    process_label = waiver_process_label(rules) \
        if (rules or {}).get("source") in _STAMP_SOURCES else None
    is_faab = str((rules or {}).get("waiver_type") or "").lower() == "faab"
    claims = []
    ir_i = 0
    drop_i = 0
    for action in actions or []:
        if not isinstance(action, dict) or action.get("kind") != "waiver":
            continue
        pid = str(action.get("player_id") or "")
        if not pid:
            continue
        info = player_info(ctx, pid, index)
        funding = None
        blocker = None
        if ir_i < len(ir_moves):
            funding = ir_moves[ir_i]
            ir_i += 1
        elif drop_i < len(drops):
            funding = drops[drop_i]
            drop_i += 1
        else:
            blocker = ("No legal funding move: no open IR slot and no "
                       "droppable bench player remains (the bench is "
                       "only starters, designated keepers, or players "
                       "who are out)")
        faab = None
        if is_faab:
            faab = action.get("faab_recommendation")
            if faab is None and action.get("pickup_score") is not None:
                try:
                    from utils.waivers import faab_recommendation

                    faab = faab_recommendation(
                        _f(action.get("pickup_score")),
                        waiver_type="faab")
                except Exception:
                    faab = None
        claim = {
            "player_id": pid,
            "name": player_name(ctx, pid, index),
            "position": player_pos(info),
            "action": action.get("action"),
            "why": action.get("why"),
            "role_reason": action.get("role_reason"),
            "dropped_today": bool(action.get("dropped_today")),
            "deadline": action.get("deadline"),
            "process_label": process_label,
            "waiver_priority": priority,
            "faab": faab,
            "funding": funding,
            "blocker": blocker,
        }
        action["funding"] = funding
        if blocker:
            action["blocker"] = blocker
        claims.append(claim)
    return claims


# ---------------------------------------------------------------------------
# asset-conflict sequencing (spec 3.6, PR 3)
# ---------------------------------------------------------------------------

def _asset_key(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if value.get("season") and value.get("round"):
            return (f"pick:{value.get('season')}:{value.get('round')}:"
                    f"{value.get('original_owner') or ''}")
        value = value.get("id") or value.get("player_id")
        if value is None:
            return None
    text = str(value)
    if not text:
        return None
    return text if text.startswith("pick:") else f"player:{text}"


def _send_asset_keys(action: dict) -> list:
    keys = []
    for field in ("send_ids", "send_player_ids", "send_assets"):
        value = action.get(field)
        if isinstance(value, (list, tuple)):
            for item in value:
                key = _asset_key(item)
                if key and key not in keys:
                    keys.append(key)
    for field in ("gives", "sends", "package"):
        value = action.get(field)
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, dict) and item.get("season") \
                        and item.get("round"):
                    key = _asset_key(item)
                    if key and key not in keys:
                        keys.append(key)
                elif isinstance(item, dict) and (
                        item.get("id") or item.get("player_id")):
                    key = _asset_key(item)
                    if key and key not in keys:
                        keys.append(key)
    for field in ("send_picks", "picks"):
        value = action.get(field)
        if isinstance(value, (list, tuple)):
            for item in value:
                key = _asset_key(item)
                if key and key not in keys:
                    keys.append(key)
    return keys


def apply_sequencing(ctx, actions, claims, now=None) -> list:
    """One asset, one plan (spec 3.6).

    If two trade actions send the same viewer asset, the higher-scored
    is step 1 ("Send first") and the other is step 2 contingent: its
    why-line is prefixed "If the {partner} deal dies:" and its score
    is demoted below every step-1 action. Picks count as assets. A
    player named as a claim's funding drop who also appears in a
    package is the same conflict (the claim is the step-1 holder when
    it outranks the deal). Contingent deals are resolved fresh at
    brief build time: the pass runs over the actions as built, every
    build.
    """
    actions = [a for a in actions or [] if isinstance(a, dict)]
    if not actions:
        return actions
    index = player_index(ctx)
    groups: dict = {}
    for action in actions:
        if action.get("kind") == "trade":
            for key in _send_asset_keys(action):
                groups.setdefault(key, []).append(action)
    action_by_pid = {}
    for action in actions:
        if action.get("kind") == "waiver" and action.get("player_id"):
            action_by_pid[str(action["player_id"])] = action
    for claim in claims or []:
        funding = claim.get("funding") or {}
        if funding.get("kind") == "drop" and funding.get("player_id"):
            key = f"player:{funding['player_id']}"
            claim_action = action_by_pid.get(str(claim.get("player_id")))
            if claim_action is not None and groups.get(key):
                groups[key].append(claim_action)
    step2: list = []
    for key, members in groups.items():
        unique = []
        for member in members:
            if member not in unique:
                unique.append(member)
        if len(unique) < 2:
            continue
        ordered = sorted(unique, key=lambda a: -_f(a.get("score")))
        winner = next(
            (a for a in ordered
             if (a.get("sequence") or {}).get("step") != 2),
            ordered[0])
        if winner.get("kind") == "trade":
            winner_label = str(winner.get("partner") or "the first")
            winner_note = "Send first"
            prefix = f"If the {winner_label} deal dies:"
        else:
            winner_label = player_name(
                ctx, winner.get("player_id") or "", index)
            winner_note = "Claim first"
            prefix = f"If the {winner_label} claim dies:"
        if not winner.get("sequence"):
            winner["sequence"] = {
                "group": key, "step": 1, "note": winner_note}
        for other in ordered:
            if other is winner or (other.get("sequence") or {}).get(
                    "step") == 2:
                continue
            why = str(other.get("why") or "")
            if not why.startswith("If the "):
                other["why"] = f"{prefix} {why}".strip()
            other["sequence"] = {
                "group": key, "step": 2,
                "note": f"Contingent on the {winner_label} move",
                "contingent_on": winner.get("action"),
            }
            step2.append(other)
    if step2:
        step1_scores = [
            _f(a.get("score")) for a in actions
            if (a.get("sequence") or {}).get("step") == 1
        ]
        if step1_scores:
            floor = min(step1_scores)
            for action in step2:
                action["pre_sequence_score"] = action.get("score")
                action["score"] = min(_f(action.get("score")), floor - 1)
    return sort_actions(actions, now=now)



# ---------------------------------------------------------------------------
# coverage map (spec 3.3, PR 2)
# ---------------------------------------------------------------------------

_COVERAGE_POSITIONS = ("QB", "RB", "WR", "TE")
_OUT_STATUSES = frozenset({
    "IR", "PUP", "NFI", "OUT", "O", "SUS", "SUSPENDED", "NA",
})


def week_projections(ctx, week):
    """``{pid: pts}`` for one week, or None when the week cannot be
    computed (no projection map at all). Sources: an explicit
    ``gm_projections`` map, the ctx ``proj_by_week`` bundles, then a
    flat ``projections`` map for the current week."""
    ctx = ctx if isinstance(ctx, dict) else {}
    week = _i(week)
    if week is None:
        return None
    explicit = ctx.get("gm_projections")
    if isinstance(explicit, dict):
        for key in (week, str(week)):
            if isinstance(explicit.get(key), dict):
                return {str(k): _f(v) for k, v in explicit[key].items()}
    bundles = ctx.get("proj_by_week")
    if isinstance(bundles, dict):
        for key in (week, str(week)):
            bundle = bundles.get(key)
            if isinstance(bundle, dict) and isinstance(
                    bundle.get("projections"), dict):
                return {str(k): _f(v)
                        for k, v in bundle["projections"].items()}
            if isinstance(bundle, dict) and bundle and all(
                    isinstance(v, (int, float)) for v in bundle.values()):
                return {str(k): _f(v) for k, v in bundle.items()}
    current = _i(ctx.get("current_week") or ctx.get("week"))
    if week == current and isinstance(ctx.get("projections"), dict):
        return {str(k): _f(v) for k, v in ctx["projections"].items()}
    return None


def role_states_for(ctx, week):
    """Role states for one week, or None when the layer is absent.

    Explicit ctx overrides win (``role_states_by_week`` /
    ``role_states`` / a ``gm_role_states`` map, with
    ``gm_role_states=None`` marking the layer absent); otherwise the
    Wave 2 ``get_role_states`` derivation runs fail-soft.
    """
    ctx = ctx if isinstance(ctx, dict) else {}
    if "gm_role_states" in ctx:
        explicit = ctx.get("gm_role_states")
        return explicit if isinstance(explicit, dict) else None
    by_week = ctx.get("role_states_by_week")
    if isinstance(by_week, dict):
        for key in (week, str(week), _i(week)):
            if isinstance(by_week.get(key), dict):
                return by_week[key]
        return None
    flat = ctx.get("role_states")
    if isinstance(flat, dict) and flat:
        return flat
    try:
        from utils.role_state import get_role_states

        states = get_role_states(ctx, int(week))
        return states if isinstance(states, dict) else None
    except Exception:
        return None


def _is_superflex(ctx) -> bool:
    rp = ctx.get("roster_positions")
    if rp is not None and hasattr(rp, "tolist"):
        rp = rp.tolist()
    slots = [str(s).upper() for s in (rp or [])]
    if any("SUPER" in s for s in slots):
        return True
    league_type = str(ctx.get("league_type") or ctx.get("format") or "").lower()
    if league_type in ("sf", "superflex", "super_flex"):
        return True
    return bool(ctx.get("is_sf"))


def _position_requirements(ctx) -> dict:
    rp = ctx.get("roster_positions")
    if rp is not None and hasattr(rp, "tolist"):
        rp = rp.tolist()
    slots = [str(s).upper().replace(" ", "_") for s in (rp or [])]
    reqs = {
        "QB": 2 if _is_superflex(ctx) else 1,
        "RB": slots.count("RB") or 2,
        "WR": slots.count("WR") or 2,
        "TE": slots.count("TE") or 1,
    }
    return reqs


def _regular_season_end(ctx, rules) -> int:
    pws = (rules or {}).get("playoff_week_start")
    if pws:
        return int(pws) - 1
    weeks = []
    bundles = ctx.get("proj_by_week") or {}
    if isinstance(bundles, dict):
        weeks.extend(_i(k) for k in bundles.keys())
    explicit = ctx.get("gm_projections") or {}
    if isinstance(explicit, dict):
        weeks.extend(_i(k) for k in explicit.keys())
    weeks = [w for w in weeks if w]
    current = _i(ctx.get("current_week") or ctx.get("week")) or 0
    return max(weeks) if weeks else current


def _viewer_roster(ctx, viewer_roster_id):
    for roster in ctx.get("rosters") or []:
        if isinstance(roster, dict) and str(roster.get("roster_id")) == str(
                viewer_roster_id):
            return roster
    return None


def build_coverage(ctx, viewer_roster_id, rules=None, actions=None):
    """Position coverage through the end of the regular season.

    Returns ``(coverage, roles_present)``. A week is computable only
    when a projection map exists for it; uncomputable weeks are named
    in ``missing_weeks`` and never interpolated. Without role state
    the map is built from projections + byes only and labeled
    ``role-unconfirmed`` (spec 5).
    """
    ctx = ctx if isinstance(ctx, dict) else {}
    rules = rules if rules is not None else get_rules(ctx)
    roster = _viewer_roster(ctx, viewer_roster_id)
    if roster is None:
        return None, False
    index = player_index(ctx)
    pids = [str(p) for p in (roster.get("players") or [])]
    if not pids:
        return None, False
    current = _i(ctx.get("current_week") or ctx.get("week")) or 0
    end = _regular_season_end(ctx, rules)
    if end < current:
        return None, False
    reqs = _position_requirements(ctx)
    # Role presence is judged once, on the current week: a layer that
    # names at least one non-unknown state is real.
    current_states = role_states_for(ctx, current)
    roles_present = bool(current_states) and any(
        str((entry or {}).get("state") or "") not in ("", "unknown")
        for entry in current_states.values()
        if isinstance(entry, dict)
    )
    weeks_out = []
    holes = []
    missing = []
    for week in range(current, end + 1):
        projections = week_projections(ctx, week)
        if projections is None:
            missing.append(week)
            continue
        states = role_states_for(ctx, week) if roles_present else None
        positions_out = {}
        week_hole = False
        for pos in _COVERAGE_POSITIONS:
            startable = []
            causes = []
            for pid in pids:
                info = player_info(ctx, pid, index)
                if player_pos(info) != pos:
                    continue
                name = player_name(ctx, pid, index)
                status = str(info.get("injury_status")
                             or info.get("status") or "").strip().upper()
                if player_bye_week(info) == week:
                    causes.append(f"bye: {name}")
                    continue
                if status in _OUT_STATUSES:
                    causes.append(f"{status}: {name}")
                    continue
                proj = projections.get(pid)
                if proj is None or proj <= 0:
                    continue
                if states is not None:
                    entry = states.get(pid) or {}
                    state = str(entry.get("state") or "") \
                        if isinstance(entry, dict) else str(entry or "")
                    if state not in STARTABLE_STATES:
                        continue
                startable.append({"id": pid, "name": name,
                                  "projection": proj})
            required = reqs[pos]
            hole = len(startable) < required
            entry_out = {
                "startable": startable,
                "count": len(startable),
                "required": required,
                "hole": hole,
                "causes": causes if hole else [],
            }
            positions_out[pos] = entry_out
            if hole:
                week_hole = True
                hole_rec = {
                    "week": week, "position": pos,
                    "count": len(startable), "required": required,
                    "causes": list(causes),
                    "fills_action": None,
                }
                # Cross-reference the queue action that fills the hole:
                # a claim at the position, or a trade receiving it.
                for action in actions or []:
                    if not isinstance(action, dict):
                        continue
                    fills = False
                    if action.get("kind") == "waiver":
                        claim_pos = player_pos(player_info(
                            ctx, action.get("player_id") or "", index))
                        fills = claim_pos == pos
                    elif action.get("kind") == "trade":
                        for rid in action.get("receive_ids") or []:
                            if player_pos(player_info(
                                    ctx, rid, index)) == pos:
                                fills = True
                    if fills:
                        hole_rec["fills_action"] = action.get("action")
                        break
                holes.append(hole_rec)
        weeks_out.append({
            "week": week,
            "positions": positions_out,
            "qb": positions_out.get("QB"),
            "hole": week_hole,
        })
    if not weeks_out:
        return None, roles_present
    coverage = {
        "weeks": weeks_out,
        "holes": holes,
        "missing_weeks": missing,
        "through_week": weeks_out[-1]["week"],
        "label": None if roles_present else "role-unconfirmed",
        "roles_confirmed": roles_present,
        "requirements": reqs,
    }
    return coverage, roles_present


# ---------------------------------------------------------------------------
# bye-cliff panel (spec 3.4, PR 2)
# ---------------------------------------------------------------------------

def _bye_by_team(ctx, index) -> dict:
    explicit = ctx.get("bye_by_team") or ctx.get("gm_bye_by_team")
    if isinstance(explicit, dict) and explicit:
        return {str(k).upper(): _i(v) for k, v in explicit.items() if _i(v)}
    out: dict = {}
    for pid, info in (index or {}).items():
        team = player_team(info)
        bye = player_bye_week(info)
        if team and bye:
            out[team] = bye
    return out


def _roster_player_rows(ctx, roster, index) -> list:
    rows = []
    for pid in (roster or {}).get("players") or []:
        pid = str(pid)
        info = player_info(ctx, pid, index)
        rows.append({
            "pid": pid,
            "name": player_name(ctx, pid, index),
            "pos": player_pos(info),
            "position": player_pos(info),
            "team": player_team(info),
        })
    return rows


def _opponent_by_week(ctx, viewer_roster_id) -> dict:
    out: dict = {}
    matchups = ctx.get("matchups_by_week") or {}
    if not isinstance(matchups, dict):
        return out
    for wk, slate in matchups.items():
        week = _i(wk)
        if week is None:
            continue
        for matchup in slate or []:
            if not isinstance(matchup, dict):
                continue
            sides = [matchup.get("left") or {}, matchup.get("right") or {}]
            rids = [str(s.get("roster_id")) for s in sides
                    if isinstance(s, dict)
                    and s.get("roster_id") is not None]
            if str(viewer_roster_id) in rids and len(rids) == 2:
                out[week] = next(r for r in rids
                                 if r != str(viewer_roster_id))
    return out


def build_bye_panel(ctx, viewer_roster_id, rules=None) -> dict | None:
    """Both-rosters bye outlook for the next 3 weeks (spec 3.4).

    Runs ``build_bye_outlook`` on the viewer's roster and on each
    week's scheduled opponent's roster, side by side, with the lost
    players named by position and crunch weeks marked.
    """
    ctx = ctx if isinstance(ctx, dict) else {}
    try:
        from utils.lineups import build_bye_outlook
    except Exception:
        return None
    index = player_index(ctx)
    bye_by_team = _bye_by_team(ctx, index)
    if not bye_by_team:
        return None
    viewer = _viewer_roster(ctx, viewer_roster_id)
    if viewer is None:
        return None
    current = _i(ctx.get("current_week") or ctx.get("week")) or 0
    horizon = [current, current + 1, current + 2]
    rp = ctx.get("roster_positions")
    if rp is not None and hasattr(rp, "tolist"):
        rp = rp.tolist()
    slots = [str(s).upper() for s in (rp or [])]
    reqs: dict = {}
    for pos in ("QB", "RB", "WR", "TE", "K", "DEF"):
        count = slots.count(pos)
        if count:
            reqs[pos] = count
    if _is_superflex(ctx):
        reqs["QB"] = max(reqs.get("QB", 1), 2)
    rosters_by_id = {
        str(r.get("roster_id")): r for r in ctx.get("rosters") or []
        if isinstance(r, dict) and r.get("roster_id") is not None
    }
    rows_by_id = {
        rid: _roster_player_rows(ctx, roster, index)
        for rid, roster in rosters_by_id.items()
    }
    opponents = _opponent_by_week(ctx, viewer_roster_id)
    names_by_rid = {r["roster_id"]: r["team_name"]
                    for r in standings_rows(ctx)}

    def _side(rid, week):
        rows = rows_by_id.get(str(rid)) or []
        outlook = build_bye_outlook(
            bye_by_team, rows, lineup_reqs=reqs, from_week=week)
        entry = next((o for o in outlook if o.get("week") == week), None)
        players_by_pos: dict = {}
        for row in rows:
            if bye_by_team.get(row["team"]) == week and row["pos"]:
                players_by_pos.setdefault(row["pos"], []).append(row["name"])
        return {
            "roster_id": str(rid),
            "team_name": names_by_rid.get(str(rid)) or f"Roster {rid}",
            "total": (entry or {}).get("total", 0),
            "by_pos": (entry or {}).get("by_pos", {}),
            "tight": (entry or {}).get("tight", []),
            "crunch": bool((entry or {}).get("crunch")),
            "players_by_pos": players_by_pos,
            "roster_counts": {
                pos: sum(1 for r in rows if r["pos"] == pos)
                for pos in ("QB", "RB", "WR", "TE", "K", "DEF")
            },
        }

    weeks_out = []
    has_crunch = False
    for week in horizon:
        viewer_side = _side(viewer_roster_id, week)
        opp_id = opponents.get(week)
        opp_side = _side(opp_id, week) if opp_id else None
        if not viewer_side["total"] and not (opp_side or {}).get("total"):
            continue
        crunch = viewer_side["crunch"] or bool(
            (opp_side or {}).get("crunch"))
        has_crunch = has_crunch or crunch
        notes = []
        for side, label in ((viewer_side, "You have"),
                            (opp_side, None)):
            if side is None:
                continue
            for pos, names in (side["players_by_pos"] or {}).items():
                remaining = side["roster_counts"].get(pos, 0) - len(names)
                if names and remaining == 0:
                    who = label or f"{side['team_name']} has"
                    notes.append(f"{who} no {pos} left in Week {week}")
        net = (f"Week {week}: you lose {viewer_side['total']}, "
               f"{(opp_side or {}).get('team_name', 'opponent')} loses "
               f"{(opp_side or {}).get('total', 0)}")
        weeks_out.append({
            "week": week,
            "viewer": viewer_side,
            "opponent": opp_side,
            "crunch": crunch,
            "notes": notes,
            "net": net,
        })
    if not weeks_out:
        return None
    return {"weeks": weeks_out, "has_crunch": has_crunch,
            "horizon": horizon}

