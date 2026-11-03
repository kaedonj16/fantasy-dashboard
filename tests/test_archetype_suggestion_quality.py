"""Suggestion-quality guards: quantity-for-quality offers must not read fair.

Kaedon's 2026-11-03 case (all three cards targeting the same RB9 from a
balanced-build partner):

- Terrance Ferguson + a 2029 1st at "59% likely" - a pick three drafts out
  co-headlining a deal for a producing starter, with a TE who does not move
  the partner's lineup.
- Ferguson + Xavier Worthy at "40%" - two bench pieces whose raw values sum
  to the target, so the old math called it fair value.
- A 2027 1st + 2028 2nd at "49%" - the one good suggestion: near-term,
  liquid capital. It must survive every guard below.

Three mechanisms, one per failure: the far-pick guard and best-player
premium in _select_packages, and the partner-lineup acceptance cap in
_partner_acceptance_cap.
"""
import sys
import types

import pytest

from dashboard_services import archetype_engine as ae


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    fake = types.ModuleType("dashboard_services.api")
    fake.fetch_json = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "dashboard_services.api", fake)
    ae._RESULT_CACHE.clear()
    yield
    ae._RESULT_CACHE.clear()


TARGET = 406.0  # the RB9's 1QB value on the live board, 2026-11-03
SEASON = 2026

# A league-wide value ladder so _depth_penalty prices the extra roster spot
# the way production does (tail value at the rank-380 bench slot).
SORTED_VALS = [900.0 - i * 2.2 for i in range(400)]


def _player(pid, pos, val):
    return {"player_id": pid, "name": pid, "position": pos, "value": val}


def _pick(season, rnd, val):
    return {"player_id": f"pick_{season}_{rnd}", "name": f"{season} R{rnd}",
            "position": "PICK", "value": val, "is_pick": True,
            "pick_season": str(season), "pick_round": rnd, "pick_bucket": "mid"}


def _select(sends, **kw):
    kw.setdefault("sorted_vals", SORTED_VALS)
    kw.setdefault("league_size", 10)
    kw.setdefault("current_season", SEASON)
    return ae._select_packages(sends, TARGET, "contending", max_pkgs=3, **kw)


def _pkg_ids(pkgs):
    return [sorted(a["player_id"] for a in p) for p in pkgs]


# ── Far-pick guard ──────────────────────────────────────────────────────────

def test_far_pick_cannot_co_headline_for_a_starter():
    sends = [_player("ferguson", "TE", 225.0), _pick(2029, 1, 209.0)]
    # Raw/effective value lands in the fair band (434 raw), so only the
    # far-pick guard can stop this package.
    assert _select(sends) == []
    # With the guard disabled (season unknown) the same package surfaces,
    # proving the guard - not the band - is what blocks it.
    pkgs = _select(sends, current_season=0)
    assert ["ferguson", "pick_2029_1"] in _pkg_ids(pkgs)


def test_far_pick_ok_as_sweetener_behind_a_real_headliner():
    sends = [_player("headliner", "RB", 300.0), _pick(2029, 1, 120.0)]
    pkgs = _select(sends)
    assert ["headliner", "pick_2029_1"] in _pkg_ids(pkgs)


def test_near_picks_package_survives():
    # The good suggestion: 2027 1st + 2028 2nd for the same target.
    sends = [_pick(2027, 1, 295.0), _pick(2028, 2, 115.0)]
    pkgs = _select(sends)
    assert ["pick_2027_1", "pick_2028_2"] in _pkg_ids(pkgs)


# ── Best-player premium ─────────────────────────────────────────────────────

def test_two_bench_pieces_at_parity_are_not_fair():
    sends = [_player("ferguson", "TE", 225.0), _player("worthy", "WR", 183.0)]
    # 408 raw / ~398 effective sits inside the old 0.96-1.08 fair band, but
    # neither piece reaches 75% of the target, so the premium applies and
    # the package must clear 1.06x - and the fallback floor keeps it out.
    assert _select(sends) == []


def test_two_pieces_that_pay_the_premium_surface():
    sends = [_player("te_premium", "TE", 250.0), _player("wr_premium", "WR", 195.0)]
    pkgs = _select(sends)
    assert ["te_premium", "wr_premium"] in _pkg_ids(pkgs)


def test_headliner_package_pays_no_premium():
    # One piece near the target plus a throw-in: no premium, plain band.
    sends = [_player("near_equal", "RB", 330.0), _player("throw_in", "WR", 85.0)]
    pkgs = _select(sends)
    assert ["near_equal", "throw_in"] in _pkg_ids(pkgs)


# ── Partner-lineup acceptance cap ───────────────────────────────────────────

def _partner():
    vals = {
        "qb1": ("QB", 500.0), "target_rb": ("RB", TARGET), "rb2": ("RB", 300.0),
        "rb3": ("RB", 240.0), "rb4": ("RB", 90.0),
        "wr1": ("WR", 350.0), "wr2": ("WR", 320.0), "wr3": ("WR", 210.0),
        "wr4": ("WR", 150.0), "wr5": ("WR", 80.0), "te1": ("TE", 180.0),
        "ferguson": ("TE", 225.0), "worthy": ("WR", 183.0),
        "big_wr": ("WR", 340.0),
    }
    values = {pid: {"position": pos, "value": val}
              for pid, (pos, val) in vals.items()}
    roster = [pid for pid in vals if pid not in ("ferguson", "worthy", "big_wr")]
    return roster, values


def _cap(pkg, roster, values, arch="balanced", target_pid="target_rb",
         target_val=TARGET, eff_ratio=None):
    return ae._partner_acceptance_cap(
        pkg, target_pid, target_val, roster, values, "1qb",
        partner_arch=arch, season=SEASON, eff_ratio=eff_ratio)


def test_cap_when_bench_pieces_gut_the_partner_lineup():
    roster, values = _partner()
    pkg = [_player("ferguson", "TE", 225.0), _player("worthy", "WR", 183.0)]
    # Partner's optimal lineup drops ~7%: a long shot, not a ~50% accept.
    assert _cap(pkg, roster, values) == 20


def test_cap_scales_with_the_lineup_hit():
    roster, values = _partner()
    pkg = [_player("ferguson", "TE", 225.0)]
    # Losing their WR2 for a TE upgrade: ~4% lineup drop -> milder cap.
    assert _cap(pkg, roster, values, target_pid="wr2", target_val=320.0) == 35


def test_no_cap_for_the_near_picks_package():
    roster, values = _partner()
    pkg = [_pick(2027, 1, 295.0), _pick(2028, 2, 115.0)]
    # The partner's lineup still gets worse - but a 1st next draft is real
    # compensation, so the good suggestion keeps its honest acceptance.
    assert _cap(pkg, roster, values) is None


# ── Logical overpays stay logical (championship push) ───────────────────────

def test_push_overpay_structure_surfaces_next_to_the_par_deal():
    # A par 1-for-1 exists, and so does a deliberate ~1.10x overpay built
    # player+pick. Both are logical offers; the overpay must not be
    # band-rejected just for exceeding par.
    sends = [_player("par_rb", "RB", 400.0),
             _player("push_wr", "WR", 250.0), _pick(2027, 1, 205.0)]
    pkgs = _pkg_ids(_select(sends))
    assert ["par_rb"] in pkgs
    assert ["pick_2027_1", "push_wr"] in pkgs


def test_overpay_softens_the_partner_cap():
    roster, values = _partner()
    pkg = [_player("ferguson", "TE", 225.0), _player("worthy", "WR", 183.0)]
    # Same gutting package: at par it caps at 20, a 1.10x overpay softens
    # to 35, and a 1.15x overpay makes the partner whole - no cap.
    assert _cap(pkg, roster, values, eff_ratio=1.11) == 35
    assert _cap(pkg, roster, values, eff_ratio=1.16) is None
    mid = [_player("ferguson", "TE", 225.0)]
    assert _cap(mid, roster, values, target_pid="wr2", target_val=320.0,
                eff_ratio=1.11) == 50


def test_no_cap_for_rebuilding_partner_or_near_equal_headliner():
    roster, values = _partner()
    pkg = [_player("ferguson", "TE", 225.0), _player("worthy", "WR", 183.0)]
    assert _cap(pkg, roster, values, arch="rebuilding") is None
    star_pkg = [_player("big_wr", "WR", 340.0)]
    assert _cap(star_pkg, roster, values) is None
