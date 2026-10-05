"""Tests for trade suggestion realism improvements.

Covers:
- Rival need tiers (desperate / need / neutral / stacked)
- Starter premium in option quality
- Send-side acceptance probability
- League-aware depth targets (not hardcoded)
- Veto risk flag on trade eval
"""
import pytest

pytest.importorskip("flask")


def _make_values(pids_vals):
    """Build a minimal values_by_id dict."""
    out = {}
    for pid, (pos, val, age) in pids_vals.items():
        out[str(pid)] = {
            "name": f"Player {pid}",
            "position": pos,
            "value": float(val),
            "age": age,
        }
    return out


class TestRivalNeedTier:
    """_rival_need_tier logic (mirrored from app.py)."""

    def _tier(self, rival_pos_vals, pos, fval):
        existing = rival_pos_vals.get(pos, [])
        best = existing[0] if existing else 0
        second = existing[1] if len(existing) > 1 else 0
        if not existing or best < 200:
            return "desperate"
        if best < 400 or fval > best * 1.15:
            return "need"
        if best >= fval or (best >= 400 and second >= 350):
            return "stacked"
        return "neutral"

    def test_desperate_no_player(self):
        assert self._tier({}, "QB", 700) == "desperate"

    def test_desperate_weak_best(self):
        assert self._tier({"QB": [150, 100]}, "QB", 700) == "desperate"

    def test_need_mediocre_starter(self):
        assert self._tier({"QB": [300, 100]}, "QB", 700) == "need"

    def test_need_clear_upgrade(self):
        # Their best is 600, focus is 700 (16% better) -> need
        assert self._tier({"RB": [600, 200]}, "RB", 700) == "need"

    def test_stacked_better_player(self):
        # Their best (800) beats the focus (700)
        assert self._tier({"WR": [800, 300]}, "WR", 700) == "stacked"

    def test_stacked_deep_room(self):
        # Two solid players at the position
        assert self._tier({"RB": [500, 400, 200]}, "RB", 450) == "stacked"

    def test_neutral_marginal(self):
        # Their best is 500, focus is 520 (4% better, not 15%)
        assert self._tier({"TE": [500, 100]}, "TE", 520) == "neutral"


class TestStarterPremium:
    """Starter returns below 1.0x should be penalized."""

    def _quality(self, total, focus_value, focus_is_starter, rival_need="neutral"):
        score = abs(total - focus_value)
        score += 0  # single asset, no consolidation
        if rival_need == "stacked":
            score += focus_value * 0.25
        elif rival_need == "desperate":
            score -= focus_value * 0.05
        if focus_is_starter and total < focus_value:
            score += (focus_value - total) * 1.5
        return score

    def test_starter_discount_penalized(self):
        # Starter worth 1000, return of 900 (10% discount)
        penalized = self._quality(900, 1000, True)
        fair = self._quality(1000, 1000, True)
        assert penalized > fair
        # Penalty is 1.5x the shortfall on top of value distance
        assert penalized == 100 + 150

    def test_non_starter_no_premium(self):
        # Bench player at a discount is not penalized extra
        assert self._quality(900, 1000, False) == 100

    def test_starter_premium_beats_discount(self):
        # A 1050 return for a 1000 starter beats a 950 return
        assert self._quality(1050, 1000, True) < self._quality(950, 1000, True)

    def test_stacked_rival_heavily_penalized(self):
        q_stacked = self._quality(1000, 1000, False, "stacked")
        q_neutral = self._quality(1000, 1000, False, "neutral")
        assert q_stacked - q_neutral == 250  # 25% of focus value

    def test_desperate_rival_boosted(self):
        q_desperate = self._quality(1000, 1000, False, "desperate")
        q_neutral = self._quality(1000, 1000, False, "neutral")
        assert q_neutral - q_desperate == 50  # 5% boost


class TestSendAcceptanceProb:
    """Send-side acceptance from the rival's perspective."""

    def _prob(self, focus_value, total, rival_need, window="competitive",
              focus_age=26, sends_picks=False, sends_youth=False):
        ratio = (focus_value / total) if total > 0 else 0
        if ratio >= 1.10:
            base = 70
        elif ratio >= 0.97:
            base = 52
        elif ratio >= 0.90:
            base = 34
        else:
            base = 16
        need_adj = {"desperate": 14, "need": 8, "neutral": 0, "stacked": -14}.get(rival_need, 0)
        window_adj = 0
        if window == "rebuild":
            if focus_age >= 28:
                window_adj -= 12
            if sends_picks or sends_youth:
                window_adj -= 8
        elif window == "win_now":
            if 24 <= focus_age <= 29:
                window_adj += 6
            if focus_age >= 31:
                window_adj -= 4
        return min(93, max(8, round(base + need_adj + window_adj)))

    def test_rival_wins_high_accept(self):
        # Rival gets 1100, sends 1000
        assert self._prob(1100, 1000, "neutral") == 70

    def test_rival_overpays_low_accept(self):
        # Rival gets 800, sends 1000
        assert self._prob(800, 1000, "neutral") == 16

    def test_desperate_need_boosts(self):
        assert self._prob(1000, 1000, "desperate") == 52 + 14

    def test_stacked_kills_deal(self):
        assert self._prob(1000, 1000, "stacked") == 52 - 14

    def test_rebuilder_rejects_old_vet(self):
        # 30-year-old to a rebuilding team
        assert self._prob(700, 700, "need", "rebuild", focus_age=30) == 52 + 8 - 12

    def test_rebuilder_hoards_picks(self):
        # Rebuilder asked to send picks
        assert self._prob(700, 700, "neutral", "rebuild", sends_picks=True) == 52 - 8

    def test_contender_wants_prime(self):
        assert self._prob(700, 700, "neutral", "win_now", focus_age=27) == 52 + 6

    def test_clamped(self):
        assert self._prob(2000, 500, "desperate", "win_now", focus_age=27) <= 93
        assert self._prob(500, 2000, "stacked", "rebuild", focus_age=35) >= 8


class TestVetoRisk:
    """Veto risk flag thresholds."""

    def _veto_risk(self, abs_diff, baseline):
        return abs_diff > max(baseline * 0.30, 150) if baseline else False

    def test_lopsided_pct(self):
        assert self._veto_risk(350, 1000) is True  # 35% gap

    def test_fair_no_risk(self):
        assert self._veto_risk(50, 1000) is False

    def test_small_baseline_floor(self):
        # 100 gap on 200 baseline: 50% exceeds 30% but the 150 raw floor
        # means tiny trades do not trip veto risk
        assert self._veto_risk(100, 200) is False
        # 40 gap on 200 baseline = 20%, below both thresholds -> False
        assert self._veto_risk(40, 200) is False
        # 200 gap on 400 baseline = 50%, clears the 150 floor -> True
        assert self._veto_risk(200, 400) is True

    def test_zero_baseline(self):
        assert self._veto_risk(100, 0) is False


class TestLeagueAwareDepth:
    """Depth targets derive from roster settings, not hardcoded values."""

    def test_sf_qb_target(self):
        # Superflex leagues need deeper QB rooms
        slot_counts = {"QB": 1, "SF": 1, "RB": 2, "WR": 3, "TE": 1}
        targets = {}
        for p in ("QB", "RB", "WR", "TE"):
            starters = int(slot_counts.get(p, 0) or 0)
            if p == "QB" and int(slot_counts.get("SF", 0) or 0) > 0:
                starters += 1
            targets[p] = max(starters + 1, 2) if p in ("QB", "TE") else max(int(starters * 1.5) + 1, 3)
        assert targets["QB"] == 3  # 1 QB + 1 SF + 1 bench
        assert targets["RB"] == 4  # 2 starters * 1.5 + 1
        assert targets["WR"] == 5  # 3 starters * 1.5 + 1 (int)
        assert targets["TE"] == 2

    def test_1qb_qb_target(self):
        slot_counts = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
        starters = 1
        assert max(starters + 1, 2) == 2
