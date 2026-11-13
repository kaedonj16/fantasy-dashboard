"""Shared trade-partner model (Wave 2 PR 3, spec sections 3.2 / 3.3).

One partner-check module for both trade engines: Trade Suggestions
(``build_trade_suggestions_context``) and Trade Targets (the
player-packages path behind ``/api/trade-targets``). The October guard
only fired when a target was the partner's *sole* startable player at
his position, and the Trade Targets engine had no guard at all — it
pitched Joe Burrow from a Superflex team holding exactly two startable
QBs, and Caleb Williams packages that orphaned Goodfellas' QB room.

Pure functions over plain data:

* rosters as lists of player-info dicts
  ``{"id"/"player_id", "position", "value", "name", ...}``,
* a package as ``send`` (viewer -> partner) / ``receive`` (partner ->
  viewer) asset lists in the same shape (picks carry ``is_pick`` /
  ``position == "PICK"`` and ``pick_round``),
* ``requirements``: starting slots per position for viability, where the
  QB requirement in Superflex is QB slots + 1 (see
  :func:`slot_requirements`),
* ``role_states`` / ``projections``: the Wave 2 situation layer
  (``utils/role_state.py``). Both are optional and fail-soft: with no
  role layer, "startable" falls back to projection-vs-bar, and with no
  projections layer at all, to the caller's value bar (the pre-Wave-2
  definition). One fail-closed exception: at QB, a player missing
  from a supplied projections layer earns no startable credit — no
  projection, no QB2 (see :func:`player_startable`).

Checks implemented (spec 3.2 numbering):

1. Surplus-by-format — partner's startable count at the target's
   position AFTER the deal must stay >= the format requirement, or the
   package returns a startable there. The requirement binds capped at
   the partner's pre-trade count (a team already below the line cannot
   be pushed further below it, and its last startable can never be
   taken — the October sole-starter guard is a strict subset), and a
   shortfall at a flex-eligible position (RB/WR/TE) is covered when
   the deal itself lands startable pieces above the requirement at
   another flex-eligible position. QB has no flex substitute: the SF
   requirement (QB slots + 1) binds in full for any team at the line.
2. Partner-viability + QB-back repair — the same simulation at every
   touched position; a shortfall at the target's own position is
   auto-repaired with the viewer's cheapest startable there beyond the
   viewer's own requirement (the caller re-checks the fairness band;
   a repair that breaks the band drops the deal, never pitches it
   broken).
3. Headline-piece caliber — the package's highest-value asset must
   crack the partner's post-trade starting lineup at its position (by
   projection, value as fallback), or be a top-2-round pick in
   dynasty/keeper formats. Only partner-starting pieces earn the
   acceptance-fit need bonus (see :func:`acceptance_fit_restricted`).
5. Contention window — contender / bubble / seller from record + PF
   against the TRUE playoff line from the Wave 1 league_rules record;
   ``default_unverified`` falls back to record + PF percentile only.
6. Bye collision — the target's bye week vs the viewer's crunch weeks
   (``utils/lineups.build_bye_outlook``): a flag, never a block.

Acceptance (spec 3.3) lives here too: :func:`conditioned_acceptance`
wraps ``archetype_engine._estimate_acceptance`` (logistic on the
effective ratio + restricted fit) with the surplus-match and contention
terms, and demotes the similar-league frequency rate to a capped
(+12) prior. Viability is a GATE: callers never compute acceptance for
a package :func:`evaluate_package` rejected.
"""
from __future__ import annotations

from typing import Optional

from utils.role_state import is_startable as _role_is_startable
from utils.role_state import starter_bar as _starter_bar

POSITIONS = ("QB", "RB", "WR", "TE")

# Fallback value bars (a player at/above the bar counts as startable
# when neither role states nor projections are available). Callers pass
# their league-derived thresholds; these mirror the 12-team defaults in
# utils.trade.derive_league_thresholds.
DEFAULT_VALUE_BARS = {"QB": 500.0, "RB": 350.0, "WR": 350.0, "TE": 200.0}

# Fairness band for a repaired package (spec 3.2 check 2: repair, then
# re-check the band; a repair that breaks it drops the deal).
FAIR_BAND_MIN = 0.80
FAIR_BAND_MAX = 1.40


def _f(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pid(asset: dict) -> str:
    return str(asset.get("player_id") or asset.get("id") or "")


def _pos(asset: dict) -> str:
    return str(asset.get("position") or "").upper()


def _is_pick(asset: dict) -> bool:
    return bool(asset.get("is_pick")) or _pos(asset) == "PICK"


# ---------------------------------------------------------------------------
# Slot requirements
# ---------------------------------------------------------------------------

def slot_requirements(roster_positions=None, is_sf: bool = False,
                      num_teams: int = 12) -> dict:
    """Required starters per position for viability checks.

    Delegates to ``utils.trade.derive_league_thresholds`` (whose depth
    floor already counts a SUPER_FLEX slot toward QB, per spec: the QB
    requirement in Superflex = QB slots + 1). Fail-soft defaults when
    the slot list is unusable.
    """
    try:
        from utils.trade import derive_league_thresholds
        _thr, floors = derive_league_thresholds(
            list(roster_positions or []), num_teams, is_sf=is_sf)
        req = {pos: int(floors.get(pos) or 0) for pos in POSITIONS}
        if any(req.values()):
            return req
    except Exception:
        pass
    req = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
    if is_sf:
        req["QB"] = 2
    return req


# ---------------------------------------------------------------------------
# Startable (the one shared definition, spec 2.2)
# ---------------------------------------------------------------------------

def player_startable(info: dict, role_states: Optional[dict] = None,
                     projections: Optional[dict] = None,
                     value_bars: Optional[dict] = None,
                     ctx: Optional[dict] = None) -> bool:
    """Is this player startable this week?

    With a role entry: ``utils.role_state.is_startable`` (startable
    state AND projection over the position bar). The projection comes
    from the projections map, then the info dict; when no projection
    layer exists at all, a value at/above the caller's value bar stands
    in for the projection test so value-only callers keep the old
    behavior. Without a role entry (role layer absent or the player
    missing from it): projection over the bar, else the value bar.

    QB exception (fail closed): when a projections layer WAS supplied,
    a QB missing from it (and carrying no projection of his own) is
    never startable — no projection means we do not know he starts,
    so he must not satisfy the Superflex starter minimum on value
    alone. The exception is QB-scoped on purpose: QB has no flex
    substitute and a QB shortfall is never offset, while at the flex
    positions this same shared definition also feeds viewer-surplus
    and headline logic, where projections maps are legitimately
    partial (flat-map callers cover a subset) and failing closed
    would silently zero every uncovered position.
    """
    if not isinstance(info, dict) or _is_pick(info):
        return False
    pos = _pos(info)
    pid = _pid(info)
    bars = value_bars or DEFAULT_VALUE_BARS
    bar = _starter_bar(pos, ctx)
    proj = None
    if projections:
        proj = projections.get(pid)
    if proj is None:
        proj = info.get("projection")
    entry = (role_states or {}).get(pid) if role_states else None
    if proj is None:
        if projections and pos == "QB":
            # A projections layer exists for this week and this QB is
            # not in it: fail closed (see docstring). The live 92916
            # case: Goodfellas' QB3 Cam Ward had no Week 9 projection
            # but cleared the QB value bar, so a Watson-alone offer
            # passed viability on a QB2 we had no evidence would play.
            return False
        # No projection FOR THIS PLAYER (layer absent, or a non-QB
        # missing from it): the value bar is the fallback definition of
        # startable (spec 3.2 shared inputs) — unless a role state
        # already rules him out (starter_out / benched).
        if isinstance(entry, dict) and entry.get("state") not in (
                None, "starter", "promoted", "returning", "unknown"):
            return False
        if isinstance(entry, str) and entry not in (
                "starter", "promoted", "returning", "unknown"):
            return False
        vbar = bars.get(pos)
        if vbar is not None:
            return _f(info.get("value")) >= _f(vbar, 0.0)
        return False
    proj = _f(proj, 0.0)
    if entry is not None:
        return _role_is_startable(entry, proj, pos, ctx)
    return proj >= bar


def startable_players(players: list, pos: str, role_states=None,
                      projections=None, value_bars=None, ctx=None) -> list:
    """The startable subset of ``players`` at ``pos``, best first.

    Ordering is by projection when a projection layer exists, else by
    value — the same order the headline-caliber check uses for "cracks
    the starting lineup".
    """
    pos = str(pos or "").upper()
    out = [
        p for p in (players or [])
        if isinstance(p, dict) and _pos(p) == pos
        and player_startable(p, role_states, projections, value_bars, ctx)
    ]

    def _key(p):
        proj = None
        if projections:
            proj = projections.get(_pid(p))
        if proj is None:
            proj = p.get("projection")
        return _f(proj if proj is not None else p.get("value"))

    return sorted(out, key=_key, reverse=True)


def _rank_key(info: dict, projections: Optional[dict]) -> float:
    proj = None
    if projections:
        proj = projections.get(_pid(info))
    if proj is None:
        proj = info.get("projection")
    return _f(proj if proj is not None else info.get("value"))


# ---------------------------------------------------------------------------
# Checks 1 + 2 + 3: package evaluation (surplus, viability, headline)
# ---------------------------------------------------------------------------

def evaluate_package(
    *,
    partner_players: list,
    viewer_players: list,
    send: list,
    receive: list,
    requirements: dict,
    role_states: Optional[dict] = None,
    projections: Optional[dict] = None,
    value_bars: Optional[dict] = None,
    ctx: Optional[dict] = None,
    is_dynasty_or_keeper: bool = False,
    target_starter_caliber: bool = True,
) -> dict:
    """Run checks 1-3 for one package. Pure; never raises on bad input.

    ``send`` = assets the viewer gives (partner receives), ``receive`` =
    assets the partner gives (the target first). Returns a dict with:

    * ``viable`` — checks 1+2 pass (partner can still field the
      required starters at every touched position after the deal),
    * ``shortfalls`` — ``{pos: {"have", "need"}}`` post-trade,
    * ``reasons`` — internal drop reasons (never user copy verbatim),
    * ``notes`` — structured viability copy for renderers, generated
      from the shortfall data (e.g. "Leaves them without a second QB"),
    * ``headline`` / ``headline_starts`` / ``headline_ok`` — check 3,
    * ``starting_pieces`` — ids of send players who crack the partner's
      post-trade lineup (the only pieces that earn a fit bonus),
    * ``target_surplus`` — partner still holds surplus at the target's
      position after the deal (acceptance term, spec 3.3),
    * ``post_counts`` — startable counts per touched position after.
    """
    send = [a for a in (send or []) if isinstance(a, dict)]
    receive = [a for a in (receive or []) if isinstance(a, dict)]
    send_players = [a for a in send if not _is_pick(a)]
    recv_players = [a for a in receive if not _is_pick(a)]
    send_ids = {_pid(a) for a in send_players}
    recv_ids = {_pid(a) for a in recv_players}

    post_partner = [
        p for p in (partner_players or [])
        if isinstance(p, dict) and _pid(p) not in recv_ids
    ] + send_players

    touched = []
    for a in recv_players + send_players:
        p = _pos(a)
        if p in POSITIONS and p not in touched:
            touched.append(p)

    req = {pos: int((requirements or {}).get(pos) or 0) for pos in POSITIONS}
    # The binding requirement at a position is the format requirement
    # capped at the partner's PRE-trade startable count: a team already
    # below the line cannot be pushed further below it, and its last
    # startable can never be taken (the October sole-starter guard is a
    # strict subset of this rule). A team AT the line — the Burrow /
    # Caleb case — must stay at it.
    post_counts: dict = {}
    shortfalls: dict = {}
    for pos in touched:
        have = len(startable_players(
            post_partner, pos, role_states, projections, value_bars, ctx))
        post_counts[pos] = have
        pre = len(startable_players(
            list(partner_players or []), pos,
            role_states, projections, value_bars, ctx))
        need = min(req.get(pos, 0), pre)
        if need > 0 and have < need:
            shortfalls[pos] = {"have": have, "need": need}
    # Flex offset: RB/WR/TE are fungible through the FLEX slot, so a
    # shortfall at one flex-eligible position is covered when the deal
    # itself lands startable pieces above the requirement at another
    # flex-eligible position (the partner re-slots his lineup). QB has
    # no flex substitute — a QB shortfall is never offset.
    flex_covered: list = []
    _FLEX_POS = ("RB", "WR", "TE")
    if any(pos in _FLEX_POS for pos in shortfalls):
        pool = 0
        for pos in touched:
            if pos in _FLEX_POS and pos not in shortfalls:
                pre = len(startable_players(
                    list(partner_players or []), pos,
                    role_states, projections, value_bars, ctx))
                pool += max(0, post_counts.get(pos, 0)
                            - min(req.get(pos, 0), pre))
        for pos in sorted(p for p in shortfalls if p in _FLEX_POS):
            deficit = shortfalls[pos]["need"] - shortfalls[pos]["have"]
            if pool >= deficit:
                pool -= deficit
                flex_covered.append(pos)
        for pos in flex_covered:
            shortfalls.pop(pos, None)

    reasons: list = []
    notes: list = []
    for pos, sf in shortfalls.items():
        reasons.append(f"partner_shortfall:{pos}")
        if pos == "QB" and req.get("QB", 0) >= 2 and sf["have"] < 2:
            notes.append("Leaves them without a second QB")
        elif sf["have"] <= 0:
            notes.append(f"Leaves them without a startable {pos}")
        else:
            notes.append(
                f"Leaves them short at {pos} "
                f"({sf['have']} of {sf['need']} starters)")

    # -- Check 3: headline caliber -------------------------------------
    headline = None
    headline_starts = False
    headline_is_elite_pick = False
    starting_pieces: list = []
    for a in send_players:
        pos = _pos(a)
        if pos not in POSITIONS:
            continue
        lineup = startable_players(
            post_partner, pos, role_states, projections, value_bars, ctx)
        top = lineup[: max(req.get(pos, 1), 1)]
        if _pid(a) in {_pid(x) for x in top}:
            starting_pieces.append(_pid(a))
    if send:
        headline = max(send, key=lambda a: _f(a.get("value")))
        if _is_pick(headline):
            try:
                rnd = int(headline.get("pick_round") or 99)
            except (TypeError, ValueError):
                rnd = 99
            headline_is_elite_pick = bool(is_dynasty_or_keeper and rnd <= 2)
        else:
            headline_starts = _pid(headline) in set(starting_pieces)
    headline_ok = True
    if target_starter_caliber and send:
        headline_ok = bool(headline_starts or headline_is_elite_pick)
        if not headline_ok:
            reasons.append("headline_not_starting_caliber")

    target_pos = _pos(recv_players[0]) if recv_players else ""
    target_surplus = bool(
        target_pos in post_counts
        and post_counts.get(target_pos, 0) > req.get(target_pos, 0))

    viable = not shortfalls
    result = {
        "viable": viable,
        "shortfalls": shortfalls,
        "flex_covered": flex_covered,
        "reasons": reasons,
        "notes": notes,
        "headline": headline,
        "headline_starts": headline_starts,
        "headline_ok": headline_ok,
        "starting_pieces": starting_pieces,
        "target_surplus": target_surplus,
        "post_counts": post_counts,
        "touched_positions": touched,
    }
    # Frontier 3 (keeper costs): in a keeper format a package's real
    # price includes the keeper surplus of any projected keepers it
    # moves (a Round 6 keeper cost is an asset beyond the sticker
    # value). Annotation only — viability math is unchanged — and
    # absent entirely when no priced keeper is in the deal.
    if ctx is not None and is_dynasty_or_keeper:
        try:
            from utils.keeper_cost import keeper_block_for_package

            block = keeper_block_for_package(ctx, send, receive)
            if block:
                result["keeper"] = block
        except Exception:
            pass
    return result


def surplus_ok(*, partner_players: list, target: dict, requirements: dict,
               role_states=None, projections=None, value_bars=None,
               ctx=None) -> bool:
    """Check 1 alone (target selection, before any package exists).

    Pitchable iff the partner's startable count at the target's
    position stays >= the requirement after removing the target. (A
    package returning a startable there is check 2's repair path, at
    package time.)
    """
    pos = _pos(target)
    if pos not in POSITIONS:
        return True
    need = int((requirements or {}).get(pos) or 0)
    if need <= 0:
        return True
    # Same cap as evaluate_package: the requirement binds at the
    # partner's pre-trade count (a team at the line must stay at it;
    # a team already below it cannot lose its last startable).
    pre = len(startable_players(
        list(partner_players or []), pos,
        role_states, projections, value_bars, ctx))
    need = min(need, pre)
    if need <= 0:
        return True
    remaining = [
        p for p in (partner_players or [])
        if isinstance(p, dict) and _pid(p) != _pid(target)
    ]
    have = len(startable_players(
        remaining, pos, role_states, projections, value_bars, ctx))
    return have >= need


def find_repair_player(
    *,
    viewer_players: list,
    send: list,
    position: str,
    viewer_requirements: dict,
    role_states=None, projections=None, value_bars=None, ctx=None,
) -> Optional[dict]:
    """Check 2 repair: the viewer's cheapest startable at ``position``
    that the viewer can spare — i.e. dealing him still leaves the
    viewer at/above his own requirement at that position after the
    current ``send`` side is removed. None when the viewer has no
    spare (the deal then drops, never pitches broken).
    """
    pos = str(position or "").upper()
    if pos not in POSITIONS:
        return None
    send_ids = {_pid(a) for a in (send or []) if isinstance(a, dict)
                and not _is_pick(a)}
    remaining = [
        p for p in (viewer_players or [])
        if isinstance(p, dict) and _pid(p) not in send_ids
    ]
    need = int((viewer_requirements or {}).get(pos) or 0)
    spares = startable_players(
        remaining, pos, role_states, projections, value_bars, ctx)
    # spares is best-first; the viewer keeps his best ``need`` and can
    # spare the rest. Cheapest spare = the last of the spare tail.
    if len(spares) <= need:
        return None
    return spares[-1]


def gate_package(
    *,
    partner_players: list,
    viewer_players: list,
    receive: list,
    package: list,
    requirements: dict,
    role_states: Optional[dict] = None,
    projections: Optional[dict] = None,
    value_bars: Optional[dict] = None,
    ctx: Optional[dict] = None,
    is_dynasty_or_keeper: bool = False,
    locked_ids=None,
    blocked_repair_ids=None,
) -> tuple:
    """Checks 1-3 as a gate with check-2 repair, for suggestion surfaces.

    Returns ``(ok, package, evaluation)``. ``receive`` is the asset
    list the partner gives (the target first); ``package`` is what the
    viewer sends. This is the one gate every suggestion surface runs
    before a package is surfaced or scored (spec 3.2/3.3: viability is
    a GATE, and acceptance is computed from the checked evaluation,
    never a pre-check estimate):

    * A send side carrying a keeper-locked player fails outright.
      Locked players are untradeable in keeper formats; the ban is a
      legality rule, checked before any evaluation runs, and is never
      fail-open. Picks cannot be locked.
    * A viable + headline-ok package passes as-is.
    * A package whose ONLY shortfall is at the first receive player's
      position is repaired with the viewer's cheapest spare startable
      there (never a locked player, never an id in
      ``blocked_repair_ids`` — callers pass redraft serious injuries
      there). The repaired package must re-pass checks 1-3 AND stay
      inside the fairness band, or the deal drops: it is never pitched
      broken.
    * An evaluation error fails OPEN (``ok`` True, evaluation None),
      matching the engines' shipped behavior: a checker bug must not
      silently empty a suggestion surface. The keeper-locked ban above
      still applies in that case.
    """
    locked = {str(x) for x in (locked_ids or ())}
    package = list(package or [])
    receive = [a for a in (receive or []) if isinstance(a, dict)]
    if locked:
        for a in package:
            if isinstance(a, dict) and not _is_pick(a) \
                    and _pid(a) in locked:
                return False, package, None
    try:
        ev = evaluate_package(
            partner_players=partner_players,
            viewer_players=viewer_players,
            send=package,
            receive=receive,
            requirements=requirements,
            role_states=role_states,
            projections=projections,
            value_bars=value_bars,
            ctx=ctx,
            is_dynasty_or_keeper=is_dynasty_or_keeper,
        )
    except Exception:
        return True, package, None
    if ev["viable"] and ev["headline_ok"]:
        return True, package, ev
    target = receive[0] if receive else {}
    tpos = _pos(target)
    if set(ev["shortfalls"].keys()) != {tpos}:
        return False, package, ev
    blocked = locked | {str(x) for x in (blocked_repair_ids or ())}
    try:
        repair = find_repair_player(
            viewer_players=[
                p for p in (viewer_players or [])
                if not (isinstance(p, dict) and _pid(p) in blocked)
            ],
            send=package, position=tpos,
            viewer_requirements=requirements,
            role_states=role_states, projections=projections,
            value_bars=value_bars, ctx=ctx,
        )
    except Exception:
        repair = None
    if repair is None or _pid(repair) in blocked:
        return False, package, ev
    if any(_pid(a) == _pid(repair) for a in package if isinstance(a, dict)):
        return False, package, ev
    repaired = package + [{
        "id": _pid(repair), "player_id": _pid(repair),
        "name": repair.get("name", ""),
        "value": _f(repair.get("value")),
        "position": _pos(repair),
        "age": repair.get("age"),
    }]
    try:
        ev2 = evaluate_package(
            partner_players=partner_players,
            viewer_players=viewer_players,
            send=repaired,
            receive=receive,
            requirements=requirements,
            role_states=role_states,
            projections=projections,
            value_bars=value_bars,
            ctx=ctx,
            is_dynasty_or_keeper=is_dynasty_or_keeper,
        )
    except Exception:
        return False, package, ev
    if not (ev2["viable"] and ev2["headline_ok"]):
        return False, package, ev
    give = sum(_f(a.get("value")) for a in repaired if isinstance(a, dict))
    get = sum(_f(a.get("value")) for a in receive)
    if give <= 0 or get <= 0:
        return False, package, ev
    if not (FAIR_BAND_MIN <= fairness_ratio(give, get)
            and give / get <= FAIR_BAND_MAX):
        return False, package, ev
    return True, repaired, ev2


def fairness_ratio(send_value: float, receive_value: float) -> float:
    hi = max(_f(send_value), _f(receive_value))
    lo = min(_f(send_value), _f(receive_value))
    return (lo / hi) if hi > 0 else 0.0


# ---------------------------------------------------------------------------
# Check 5: contention window
# ---------------------------------------------------------------------------

def parse_record(record) -> tuple:
    """(wins, losses, ties) from a "W-L" / "W-L-T" record string."""
    parts = str(record or "").replace("–", "-").split("-")
    nums = []
    for part in parts[:3]:
        try:
            nums.append(float(part.strip()))
        except (TypeError, ValueError):
            nums.append(0.0)
    while len(nums) < 3:
        nums.append(0.0)
    return nums[0], nums[1], nums[2]


def classify_contention(
    *,
    record=None,
    pf: float = 0.0,
    pf_percentile: Optional[float] = None,
    seed: Optional[int] = None,
    games_behind_cut: Optional[float] = None,
    playoff_teams: Optional[int] = None,
    rules_source: Optional[str] = None,
) -> dict:
    """Classify a partner contender / bubble / seller (check 5).

    With a verified rules record (source feed / derived /
    user_confirmed and a real ``playoff_teams``), the class is read
    against the true playoff line: in the field (or within a game with
    top-half PF) = contender; >= 2 games out with bottom-half PF =
    seller. When the record is ``default_unverified`` the fabricated
    field size is NEVER used: record + PF percentile only.
    """
    wins, losses, ties = parse_record(record)
    games = wins + losses + ties
    win_pct = ((wins + 0.5 * ties) / games) if games > 0 else 0.5
    pf_pct = pf_percentile
    if pf_pct is None:
        pf_pct = 0.5
    verified = bool(
        playoff_teams and str(rules_source or "") != "default_unverified"
        and str(rules_source or "") != "")
    cls = "bubble"
    basis = "record_pf"
    if verified:
        basis = "playoff_line"
        in_field = seed is not None and int(seed) <= int(playoff_teams)
        gb = games_behind_cut
        if in_field or (gb is not None and gb <= 1.0 and pf_pct >= 0.5):
            cls = "contender"
        elif gb is not None and gb >= 2.0 and pf_pct < 0.5:
            cls = "seller"
        elif gb is not None and gb >= 2.0 and win_pct <= 0.35:
            cls = "seller"
        else:
            cls = "bubble"
    else:
        if win_pct >= 0.55 and pf_pct >= 0.5:
            cls = "contender"
        elif win_pct <= 0.40 and pf_pct < 0.5:
            cls = "seller"
        else:
            cls = "bubble"
    return {
        "class": cls,
        "win_pct": round(win_pct, 3),
        "pf_percentile": round(_f(pf_pct, 0.5), 3),
        "basis": basis,
    }


def contention_acceptance_adj(contention) -> int:
    """Acceptance points for the partner's window (spec 3.3): sellers
    take fair value more readily; contenders require the win-now
    premium the raw ratio does not price."""
    cls = ""
    if isinstance(contention, dict):
        cls = str(contention.get("class") or "")
    elif isinstance(contention, str):
        cls = contention
    return {"seller": 6, "contender": -5, "bubble": 0}.get(cls, 0)


def contention_gettability_adj(contention) -> float:
    """Gettability nudge (0-1 scale): sellers move starters, contenders
    hold them."""
    cls = ""
    if isinstance(contention, dict):
        cls = str(contention.get("class") or "")
    elif isinstance(contention, str):
        cls = contention
    return {"seller": 0.2, "contender": -0.2, "bubble": 0.0}.get(cls, 0.0)


# ---------------------------------------------------------------------------
# Check 6: bye collision (flag, never a block)
# ---------------------------------------------------------------------------

def viewer_crunch_weeks(viewer_players: list, bye_by_team: dict,
                        requirements: dict, from_week: int = 1) -> list:
    """The viewer's bye-crunch weeks from ``build_bye_outlook``.

    ``viewer_players`` entries need position + NFL team (``team`` /
    ``nfl_team`` / ``nfl`` keys all accepted). Returns the raw outlook
    rows (each ``{"week", "by_pos", "tight", "crunch", ...}``).
    """
    if not bye_by_team:
        return []
    try:
        from utils.lineups import build_bye_outlook
    except Exception:
        return []
    roster = []
    for p in viewer_players or []:
        if not isinstance(p, dict):
            continue
        roster.append({
            "position": _pos(p),
            "team": str(p.get("team") or p.get("nfl_team")
                        or p.get("nfl") or "").upper(),
        })
    reqs = {pos: int((requirements or {}).get(pos) or 0)
            for pos in POSITIONS}
    try:
        return build_bye_outlook(
            bye_by_team, roster, lineup_reqs=reqs, from_week=from_week)
    except Exception:
        return []


def bye_collision_flag(target: dict, viewer_players: list,
                       bye_by_team: dict, requirements: dict,
                       from_week: int = 1) -> Optional[dict]:
    """Check 6: does the target's bye land on a viewer crunch week at
    his position? Returns chip data (``{"week", "position", "label"}``)
    or None. A collision flags and prices; it never blocks."""
    bye = target.get("bye_week") or target.get("byeWeek") \
        or target.get("bye")
    try:
        bye = int(bye)
    except (TypeError, ValueError):
        return None
    pos = _pos(target)
    for row in viewer_crunch_weeks(
            viewer_players, bye_by_team, requirements, from_week):
        if int(row.get("week") or 0) != bye:
            continue
        if row.get("crunch") and pos in (row.get("tight") or []):
            return {
                "week": bye,
                "position": pos,
                "label": f"Bye hits your Week {bye} crunch",
            }
    return None


# ---------------------------------------------------------------------------
# Spec 3.3: partner-conditioned acceptance
# ---------------------------------------------------------------------------

def acceptance_fit_restricted(pkg: list, need, stacked,
                              starting_ids) -> int:
    """``archetype_engine._acceptance_fit`` with the check-3 rule:
    the need bonus accrues ONLY from pieces that start for the partner
    post-trade (``starting_ids`` from :func:`evaluate_package`). The
    stacked penalty still applies to any piece — piling onto depth is
    unappealing whether or not the piece starts. Picks earn nothing.
    """
    if not pkg or not (need or stacked):
        return 0
    starting = {str(x) for x in (starting_ids or ())}
    bonus = 0
    for a in pkg:
        if not isinstance(a, dict) or _is_pick(a):
            continue
        pos = a.get("position")
        if need and pos in need and _pid(a) in starting:
            bonus += 6
        elif stacked and pos in stacked:
            bonus -= 8
    return max(-12, min(12, bonus))


def surplus_match_term(evaluation: Optional[dict]) -> int:
    """Surplus-match term (spec 3.3): a target drawn from partner
    surplus accepts easier; one that leaves the partner exactly at the
    requirement is a harder sell."""
    if not isinstance(evaluation, dict):
        return 0
    return 4 if evaluation.get("target_surplus") else -3


def frequency_prior(trades_like_this, total_trades) -> tuple:
    """The similar-league real-trade rate, demoted to a prior.

    Returns ``(points, has_prior)``: capped at +12 as before; when no
    similar trades exist the prior is empty (0 points, ``has_prior``
    False) and acceptance is computed from ratio + fit alone.
    """
    try:
        count = int(trades_like_this or 0)
        total = int(total_trades or 0)
    except (TypeError, ValueError):
        return 0, False
    if count <= 0 or total <= 0:
        return 0, False
    return min(int(count / max(total, 1) * 60), 12), True


def conditioned_acceptance(
    send_value: float,
    receive_value: float,
    *,
    fit: int = 0,
    evaluation: Optional[dict] = None,
    contention=None,
    trades_like_this=0,
    total_trades=0,
    is_preferred: bool = False,
    extra_adj: int = 0,
) -> tuple:
    """The shared conditioned acceptance (spec 3.3). Returns
    ``(probability:int, basis:str)``.

    Core = ``archetype_engine._estimate_acceptance`` (logistic on the
    effective ratio + restricted fit). Terms added: surplus match,
    contention window, bye penalty (via ``extra_adj``), and the
    frequency prior (capped +12). ``basis`` records whether the
    frequency prior contributed, so the UI never presents a
    false-precision number without its basis: ``"ratio_fit"`` when the
    prior is empty, ``"ratio_fit+prior"`` when it is not.
    """
    from dashboard_services.archetype_engine import _estimate_acceptance
    base = _estimate_acceptance(
        _f(send_value), _f(receive_value),
        is_preferred=is_preferred, fit=int(fit or 0))
    prior_pts, has_prior = frequency_prior(trades_like_this, total_trades)
    total = (
        base
        + surplus_match_term(evaluation)
        + contention_acceptance_adj(contention)
        + prior_pts
        + int(extra_adj or 0)
    )
    prob = int(min(93, max(5, round(total))))
    return prob, ("ratio_fit+prior" if has_prior else "ratio_fit")


__all__ = [
    "POSITIONS",
    "DEFAULT_VALUE_BARS",
    "FAIR_BAND_MIN",
    "FAIR_BAND_MAX",
    "slot_requirements",
    "player_startable",
    "startable_players",
    "evaluate_package",
    "gate_package",
    "surplus_ok",
    "find_repair_player",
    "fairness_ratio",
    "parse_record",
    "classify_contention",
    "contention_acceptance_adj",
    "contention_gettability_adj",
    "viewer_crunch_weeks",
    "bye_collision_flag",
    "acceptance_fit_restricted",
    "surplus_match_term",
    "frequency_prior",
    "conditioned_acceptance",
]
