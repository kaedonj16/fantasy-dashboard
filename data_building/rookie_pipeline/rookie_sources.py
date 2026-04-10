from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from data_building.rookie_pipeline.rookie_metric_derivations import (
    derive_adjusted_comp_pct_proxy,
    derive_explosive_run_rate,
    derive_mtf_per_att_proxy,
    derive_performance_vs_top_defenses,
    derive_player_level_sos,
    derive_routes_run_proxy,
    derive_tprr_proxy,
    derive_true_early_declare,
    derive_twp_rate_proxy,
    derive_yac_per_att_proxy,
    derive_yprr_proxy,
)
from data_building.rookie_pipeline.rookie_storage import utc_now_iso


@dataclass
class RookieMetricSpec:
    name: str
    best_source_candidate: Optional[str] = None


def base_metric_payload(
    value: Any,
    season: int,
    source_name: str,
    source_type: str,
    source_url: Optional[str],
    confidence: float,
) -> Dict[str, Any]:
    return {
        "value": value,
        "season": season,
        "source_name": source_name,
        "source_type": source_type,
        "source_url": source_url,
        "confidence": confidence,
        "updated_at": utc_now_iso(),
    }


class RookieSource:
    source_name = "base"
    source_type = "manual"
    source_url: Optional[str] = None

    def fetch_player_season_metrics(
        self,
        player: Dict[str, Any],
        season_record: Dict[str, Any],
        requested_metrics: Iterable[RookieMetricSpec],
    ) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError


class ProspectSeasonStatsSource(RookieSource):
    """
    Direct source from normalized college season tables (rookie_prospect_source_data).

    DIRECT_MAP entries are tuples of (source_field, confidence) or plain strings
    (source_field only, default confidence 0.70). Entries with a callable value
    receive the season_record and return (computed_value, confidence).
    """

    source_name = "rookie_prospect_source_data"
    source_type = "api"

    # metric_name → source field name (or callable for inline computation)
    # Each entry is (field_or_callable, confidence)
    DIRECT_MAP: Dict[str, Any] = {
        "snap_counts": ("games_played", 0.70),
        # adjusted_comp_pct: raw completion_pct is a solid direct proxy (QB only)
        "adjusted_comp_pct": ("completion_pct", 0.65),
    }

    # Inline calculations where we need more than one field
    # key → callable(season_record) → Optional[float]
    _INLINE: Dict[str, Any] = {
        # twp_rate proxy: INT / pass_attempts * 100 (QB only)
        "twp_rate": lambda sr: (
            round((float(sr["interceptions"]) / float(sr["pass_attempts"])) * 100.0, 3)
            if sr.get("interceptions") is not None
            and sr.get("pass_attempts") is not None
            and float(sr.get("pass_attempts", 0)) >= 50
            else None
        ),
    }
    _INLINE_CONFIDENCE: Dict[str, float] = {
        "twp_rate": 0.55,
    }

    def fetch_player_season_metrics(
        self,
        player: Dict[str, Any],
        season_record: Dict[str, Any],
        requested_metrics: Iterable[RookieMetricSpec],
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        season = int(season_record.get("season") or player.get("draft_class_year") or 0)
        player_key = player.get("player_id") or player.get("name") or "unknown"

        for metric in requested_metrics:
            # --- direct field map ---
            if metric.name in self.DIRECT_MAP:
                entry = self.DIRECT_MAP[metric.name]
                field, confidence = entry if isinstance(entry, tuple) else (entry, 0.70)
                raw_value = season_record.get(field)
                if raw_value is not None:
                    print(f"[direct_hit]  player={player_key} season={season} metric={metric.name} field={field} value={raw_value!r}")
                    out[metric.name] = base_metric_payload(
                        value=raw_value,
                        season=season,
                        source_name=self.source_name,
                        source_type=self.source_type,
                        source_url=self.source_url,
                        confidence=confidence,
                    )
                else:
                    print(f"[direct_miss] player={player_key} season={season} metric={metric.name} field={field} → None in season_record")
                continue

            # --- inline computation map ---
            if metric.name in self._INLINE:
                try:
                    computed = self._INLINE[metric.name](season_record)
                except Exception:
                    computed = None
                if computed is not None:
                    print(f"[inline_hit]  player={player_key} season={season} metric={metric.name} value={computed!r}")
                    out[metric.name] = base_metric_payload(
                        value=computed,
                        season=season,
                        source_name=self.source_name,
                        source_type=self.source_type,
                        source_url=self.source_url,
                        confidence=self._INLINE_CONFIDENCE.get(metric.name, 0.55),
                    )

        return out


class DerivedRookieMetricsSource(RookieSource):
    """
    Derives rookie evaluation metrics from available college stats using
    deterministic formulas. See rookie_metric_derivations.py for each formula,
    its assumptions, and its stated confidence level.
    """

    source_name = "rookie_metric_derivations"
    source_type = "derived"

    def fetch_player_season_metrics(
        self,
        player: Dict[str, Any],
        season_record: Dict[str, Any],
        requested_metrics: Iterable[RookieMetricSpec],
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        season = int(season_record.get("season") or player.get("draft_class_year") or 0)
        position = (player.get("position") or "").upper()

        handlers = {
            # --- existing derivations ---
            "explosive_run_rate": lambda: derive_explosive_run_rate(season_record),
            "player_level_sos": lambda: derive_player_level_sos(season_record),
            "performance_vs_top_defenses": lambda: derive_performance_vs_top_defenses(season_record),
            "true_early_declare": lambda: derive_true_early_declare(player),
            # --- new proxy derivations ---
            "routes_run": lambda: derive_routes_run_proxy(season_record, position),
            "yprr": lambda: derive_yprr_proxy(season_record, position),
            "tprr": lambda: derive_tprr_proxy(season_record, position),
            "yac_per_att": lambda: derive_yac_per_att_proxy(season_record),
            "mtf_per_att": lambda: derive_mtf_per_att_proxy(season_record),
            "adjusted_comp_pct": lambda: derive_adjusted_comp_pct_proxy(season_record),
            "twp_rate": lambda: derive_twp_rate_proxy(season_record),
        }
        confidences = {
            "explosive_run_rate": 0.45,
            "player_level_sos": 0.55,
            "performance_vs_top_defenses": 0.40,
            "true_early_declare": 0.80,
            # routes_run/yprr/tprr use receptions fallback when targets absent (CFBD)
            # → lower confidence than the targets-path to reflect extra indirection
            "routes_run": 0.30,
            "yprr": 0.28,
            "tprr": 0.28,
            "yac_per_att": 0.40,
            "mtf_per_att": 0.30,
            "adjusted_comp_pct": 0.60,
            "twp_rate": 0.65,
        }

        player_key = player.get("player_id") or player.get("name") or "unknown"

        for metric in requested_metrics:
            fn = handlers.get(metric.name)
            if not fn:
                continue
            try:
                value = fn()
            except Exception as exc:
                print(f"[derive_error] player={player_key} season={season} metric={metric.name}: {type(exc).__name__}: {exc}")
                value = None
            if value is None:
                # Log which inputs were present/missing for this derivation
                _DERIVE_INPUTS = {
                    "routes_run":  ["targets", "games_played"],
                    "yprr":        ["receiving_yards", "targets", "games_played"],
                    "tprr":        ["targets", "games_played"],
                    "yac_per_att": ["yds_per_carry", "rush_attempts"],
                    "mtf_per_att": ["yds_per_carry", "rush_attempts"],
                    "adjusted_comp_pct": ["completion_pct", "pass_attempts", "td_int_ratio"],
                    "twp_rate":    ["interceptions", "pass_attempts"],
                    "explosive_run_rate": ["yds_per_carry", "rush_attempts"],
                    "player_level_sos":   ["conference"],
                    "performance_vs_top_defenses": ["dominator_rating", "market_share_yards", "conference"],
                    "true_early_declare": [],
                }
                needed = _DERIVE_INPUTS.get(metric.name, [])
                missing_inputs = [f for f in needed if season_record.get(f) is None]
                present_inputs = {f: season_record.get(f) for f in needed if season_record.get(f) is not None}
                print(
                    f"[derive_null] player={player_key} season={season} metric={metric.name} "
                    f"present={present_inputs} missing_inputs={missing_inputs}"
                )
                continue
            print(f"[derive_ok]   player={player_key} season={season} metric={metric.name} value={value!r}")
            out[metric.name] = base_metric_payload(
                value=value,
                season=season,
                source_name=self.source_name,
                source_type=self.source_type,
                source_url=self.source_url,
                confidence=confidences.get(metric.name, 0.5),
            )

        return out


class SportradarNCAAFBSource(RookieSource):
    """
    Real college stats from the Sportradar NCAAFB API.

    Provides routes_run, yprr, tprr, and snap_counts computed from real
    target counts — significantly more accurate than the CFBD-based proxies
    that must estimate routes from receptions.

    Requires a SportradarNCAAIndex built by build_sportradar_ncaa_index().
    When no index is provided (API key absent), this source returns {} for
    every player and the derivation sources handle coverage as before.
    """

    source_name = "sportradar_ncaafb"
    source_type = "api"

    # Metric → (confidence_with_targets, confidence_fallback)
    # routes_run: target-path uses real numerator; tprr baseline is still estimated
    # yprr: real yards ÷ estimated routes = good directional signal
    # tprr: targets ÷ estimated routes = close to baseline (less useful), low conf
    _METRIC_CONF: Dict[str, float] = {
        "routes_run": 0.60,
        "yprr":       0.58,
        "tprr":       0.40,
        "snap_counts": 0.85,
    }

    def __init__(self, index=None):
        # index is a SportradarNCAAIndex or None
        self._index = index

    def fetch_player_season_metrics(
        self,
        player: Dict[str, Any],
        season_record: Dict[str, Any],
        requested_metrics: Iterable[RookieMetricSpec],
    ) -> Dict[str, Dict[str, Any]]:
        if self._index is None:
            return {}

        name = player.get("name", "")
        season = int(season_record.get("season") or player.get("draft_class_year") or 0)
        pos = (player.get("position") or "").upper()

        sr_stats = self._index.get_season_stats(name, season)
        if not sr_stats:
            return {}

        from data_building.rookie_pipeline.rookie_metric_derivations import (
            _TPRR_BASELINE,
        )

        out: Dict[str, Dict[str, Any]] = {}
        targets = sr_stats.get("targets")
        rec_yards = sr_stats.get("receiving_yards", 0) or 0
        games = sr_stats.get("games_played")

        baseline = _TPRR_BASELINE.get(pos)
        routes_estimated: Optional[float] = None
        if baseline and targets is not None and targets > 0:
            routes_estimated = round(targets / baseline, 1)

        player_key = player.get("player_id") or name

        for metric in requested_metrics:
            payload = None

            if metric.name == "routes_run":
                if routes_estimated is not None:
                    print(f"[sr_ncaa_src] ok player={player_key} season={season} metric=routes_run "
                          f"targets={targets} baseline={baseline} value={routes_estimated}")
                    payload = base_metric_payload(
                        value=routes_estimated,
                        season=season,
                        source_name=self.source_name,
                        source_type=self.source_type,
                        source_url=None,
                        confidence=self._METRIC_CONF["routes_run"],
                    )

            elif metric.name == "yprr":
                if routes_estimated and routes_estimated > 0:
                    yprr = round(rec_yards / routes_estimated, 3)
                    print(f"[sr_ncaa_src] ok player={player_key} season={season} metric=yprr "
                          f"rec_yards={rec_yards} routes={routes_estimated} value={yprr}")
                    payload = base_metric_payload(
                        value=yprr,
                        season=season,
                        source_name=self.source_name,
                        source_type=self.source_type,
                        source_url=None,
                        confidence=self._METRIC_CONF["yprr"],
                    )

            elif metric.name == "tprr":
                # tprr = targets / routes; routes derived from targets → result ≈ baseline
                # Only emit this when targets are real and routes are estimated from them,
                # as a sanity-check metric with low confidence.
                if routes_estimated and routes_estimated > 0 and targets is not None:
                    tprr = round(targets / routes_estimated, 4)
                    if tprr > 0:
                        payload = base_metric_payload(
                            value=tprr,
                            season=season,
                            source_name=self.source_name,
                            source_type=self.source_type,
                            source_url=None,
                            confidence=self._METRIC_CONF["tprr"],
                        )

            elif metric.name == "snap_counts":
                if games is not None:
                    payload = base_metric_payload(
                        value=games,
                        season=season,
                        source_name=self.source_name,
                        source_type=self.source_type,
                        source_url=None,
                        confidence=self._METRIC_CONF["snap_counts"],
                    )

            if payload:
                out[metric.name] = payload

        return out


def rookie_metric_specs() -> List[RookieMetricSpec]:
    specs = [
        RookieMetricSpec("routes_run", "PFF College"),
        RookieMetricSpec("yprr", "PFF College"),
        RookieMetricSpec("tprr", "PFF College"),
        RookieMetricSpec("alignment_slot_pct", "Sports Info Solutions"),
        RookieMetricSpec("alignment_wide_pct", "Sports Info Solutions"),
        RookieMetricSpec("alignment_inline_pct", "Sports Info Solutions"),
        RookieMetricSpec("contested_catch_rate", "PFF College"),
        RookieMetricSpec("yac_per_att", "PFF College"),
        RookieMetricSpec("mtf_per_att", "PFF College"),
        RookieMetricSpec("explosive_run_rate", "TruMedia"),
        RookieMetricSpec("pass_block_snaps", "PFF College"),
        RookieMetricSpec("pressures_allowed", "PFF College"),
        RookieMetricSpec("pressure_to_sack_rate", "PFF College"),
        RookieMetricSpec("adjusted_comp_pct", "PFF College"),
        RookieMetricSpec("btt_rate", "PFF College"),
        RookieMetricSpec("twp_rate", "PFF College"),
        RookieMetricSpec("epa_clean", "CFBData play-by-play"),
        RookieMetricSpec("epa_pressured", "CFBData play-by-play"),
        RookieMetricSpec("time_to_throw", "PFF College"),
        RookieMetricSpec("player_level_sos", "CFBData advanced team stats"),
        RookieMetricSpec("performance_vs_top_defenses", "CFBData game-level stats"),
        RookieMetricSpec("snap_counts", "PFF College"),
        RookieMetricSpec("true_early_declare", "NFL Draft declaration tracker"),
        RookieMetricSpec("injury_flags", "Draft Sharks college injuries"),
    ]
    return specs
