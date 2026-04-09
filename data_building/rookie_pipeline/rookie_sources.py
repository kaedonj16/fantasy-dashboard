from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from data_building.rookie_pipeline.rookie_metric_derivations import (
    derive_explosive_run_rate,
    derive_performance_vs_top_defenses,
    derive_player_level_sos,
    derive_true_early_declare,
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
    """Direct source from normalized college season tables."""

    source_name = "rookie_prospect_source_data"
    source_type = "api"

    DIRECT_MAP = {
        "snap_counts": "games_played",
    }

    def fetch_player_season_metrics(
        self,
        player: Dict[str, Any],
        season_record: Dict[str, Any],
        requested_metrics: Iterable[RookieMetricSpec],
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        season = int(season_record.get("season") or player.get("draft_class_year") or 0)

        for metric in requested_metrics:
            if metric.name in self.DIRECT_MAP:
                raw_value = season_record.get(self.DIRECT_MAP[metric.name])
                if raw_value is not None:
                    out[metric.name] = base_metric_payload(
                        value=raw_value,
                        season=season,
                        source_name=self.source_name,
                        source_type=self.source_type,
                        source_url=self.source_url,
                        confidence=0.7,
                    )

        return out


class DerivedRookieMetricsSource(RookieSource):
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

        handlers = {
            "explosive_run_rate": lambda: derive_explosive_run_rate(season_record),
            "player_level_sos": lambda: derive_player_level_sos(season_record),
            "performance_vs_top_defenses": lambda: derive_performance_vs_top_defenses(season_record),
            "true_early_declare": lambda: derive_true_early_declare(player),
        }
        confidences = {
            "explosive_run_rate": 0.45,
            "player_level_sos": 0.55,
            "performance_vs_top_defenses": 0.4,
            "true_early_declare": 0.8,
        }

        for metric in requested_metrics:
            fn = handlers.get(metric.name)
            if not fn:
                continue
            value = fn()
            if value is None:
                continue
            out[metric.name] = base_metric_payload(
                value=value,
                season=season,
                source_name=self.source_name,
                source_type=self.source_type,
                source_url=self.source_url,
                confidence=confidences.get(metric.name, 0.5),
            )

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
