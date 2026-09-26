"""
Advanced Metrics leaderboard page.

Pick an advanced metric (e.g. yards/carry) and see every player ranked at that
metric in a sortable, searchable table with a relative bar. The position filter
auto-narrows to the positions where the metric is meaningful (with manual
override). Data comes from /api/advanced-metrics/leaderboard. The page is free.
"""
import logging
import json
from html import escape as _esc
from typing import Optional


# Presets are data, rather than UI behaviour.  Keeping this metadata in Python
# makes it possible to validate every key against LEADERBOARD_METRICS and gives
# the client one authoritative definition for ordering, position and sorting.
# ``samples`` uses canonical identities (not labels), which is also what the
# table/export column builder uses for de-duplication.
ADVANCED_METRIC_PRESETS = {
    "rushing": {"label": "Rushing", "description": "Rushing workload and per-attempt efficiency.", "position": None, "primary": "rushing_epa_per_att", "metrics": ["rushing_epa_per_att", "carries_per_game", "yards_per_carry", "ngs_rush_yards_over_expected_per_att", "rushing_success_rate", "breakaway_percentage", "red_zone_usage"], "sort": "desc", "samples": ["games", "carries"]},
    "receiving": {"label": "Receiving", "description": "Receiving opportunity and target efficiency.", "position": None, "primary": "target_share", "metrics": ["target_share", "receptions_per_game", "yards_per_target", "receiving_epa_per_target", "avg_depth_of_target", "ngs_avg_yac_above_expectation", "rz_targets_pg"], "sort": "desc", "samples": ["games", "targets", "receptions"]},
    "passing": {"label": "Passing", "description": "Passing efficiency per dropback and attempt.", "position": "QB", "primary": "epa_per_play", "metrics": ["epa_per_play", "cpoe", "success_rate", "explosive_pass_rate", "sack_rate", "td_rate", "int_rate"], "sort": "desc", "samples": ["games", "attempts", "dropbacks"]},
    "rb": {"label": "RB", "description": "Running back opportunity and production.", "position": "RB", "primary": "opportunity_share", "metrics": ["opportunity_share", "carries_per_game", "target_share", "red_zone_usage", "ngs_rush_yards_over_expected_per_att", "yards_per_touch", "total_tds_per_game"], "sort": "desc", "samples": ["games", "carries", "targets", "receptions"]},
    "wr": {"label": "WR", "description": "Wide receiver opportunity and target efficiency.", "position": "WR", "primary": "target_share", "metrics": ["target_share", "air_yards_share", "avg_depth_of_target", "yards_per_target", "receiving_epa_per_target", "rz_targets_pg", "fpts_per_target"], "sort": "desc", "samples": ["games", "targets", "receptions"]},
    "te": {"label": "TE", "description": "Tight end opportunity and target efficiency.", "position": "TE", "primary": "target_share", "metrics": ["target_share", "receptions_per_game", "rz_targets_pg", "yards_per_target", "receiving_epa_per_target", "ngs_avg_yac_above_expectation", "fpts_per_target"], "sort": "desc", "samples": ["games", "targets", "receptions"]},
    "qb": {"label": "QB", "description": "Quarterback passing efficiency and dual-threat production.", "position": "QB", "primary": "epa_per_play", "metrics": ["epa_per_play", "cpoe", "pass_tds_per_game", "rush_yards_per_game", "scramble_rate", "sack_rate", "int_rate"], "sort": "desc", "samples": ["games", "attempts", "dropbacks", "carries"]},
    "general": {"label": "General / Opportunity", "description": "Cross-position opportunity and red-zone workload.", "position": None, "primary": "opportunity_share", "metrics": ["opportunity_share", "snap_share", "carries_per_game", "target_share", "air_yards_share", "red_zone_usage", "rz_targets_pg"], "sort": "desc", "samples": ["games", "carries", "targets"]},
    "expected": {"label": "Expected Points", "description": "Opportunity-based expected PPR production and efficiency vs. expectation; FPOE is not a promise of future points.", "position": None, "primary": "expected_ppr_per_game", "metrics": ["expected_ppr_per_game", "ppr_over_expected_per_game", "xfp_trend", "opportunity_share", "target_share", "red_zone_usage", "total_tds_per_game"], "sort": "desc", "samples": ["games"]},
    "receiving_profile": {"label": "Receiving Profile", "description": "Target shape, alignment context, and after-catch performance.", "position": None, "primary": "target_share", "metrics": ["target_share", "avg_depth_of_target", "air_yards_share", "ngs_avg_cushion", "ngs_avg_separation", "yards_after_catch_per_reception", "ngs_avg_yac_above_expectation"], "sort": "desc", "samples": ["games", "targets", "receptions"]},
    # --- Decision presets: organized by the question being answered, not the
    # metric taxonomy. "kind": "decision" puts them in their own picker group
    # and drives the preset-pill row above the table. ---
    "key_metrics": {"label": "Key Metrics", "kind": "decision", "tagline": "The metrics that actually predict fantasy points.", "description": "The predictive core: expected points, efficiency vs. expectation, and the opportunity signals that stick week to week. This is the default landing view.", "position": None, "primary": "expected_ppr_per_game", "metrics": ["expected_ppr_per_game", "ppr_over_expected_per_game", "opportunity_share", "target_share", "air_yards_share", "snap_share", "red_zone_usage", "opportunity_trend", "xfp_trend"], "sort": "desc", "samples": ["games"]},
    "start_sit": {"label": "Start / Sit", "kind": "decision", "tagline": "Set your lineup with confidence.", "description": "Role first (expected PPR), then TD equity (red-zone usage), then week-to-week reliability (bust rate, consistency) and matchup ease.", "position": None, "primary": "expected_ppr_per_game", "metrics": ["expected_ppr_per_game", "ppr_over_expected_per_game", "opportunity_share", "snap_share", "red_zone_usage", "rz_targets_pg", "schedule_ease", "bust_rate", "fp_cv"], "sort": "desc", "samples": ["games"]},
    "buy_low_sell_high": {"label": "Buy Low / Sell High", "kind": "decision", "tagline": "Find mispriced players before your league does.", "description": "Sorts by PPR over expected, ascending: the most negative names are the buy-low list (role is real, production hasn't caught up); flip the sort for the sell-high list.", "position": None, "primary": "ppr_over_expected_per_game", "metrics": ["ppr_over_expected_per_game", "expected_ppr_per_game", "target_share", "air_yards_share", "opportunity_trend", "xfp_trend"], "sort": "asc", "samples": ["games"]},
    "waiver_wire": {"label": "Waiver Wire", "kind": "decision", "tagline": "Who's earning a role worth adding?", "description": "Sorted by usage trend: players whose opportunity is growing fastest, with the snap/target/carry volume to back it up.", "position": None, "primary": "opportunity_trend", "metrics": ["opportunity_trend", "snap_share", "target_share", "carries_per_game", "expected_ppr_per_game", "red_zone_usage"], "sort": "desc", "samples": ["games"]},
    "breakout_check": {"label": "Is This Breakout Real?", "kind": "decision", "tagline": "Separate role growth from hot streaks.", "description": "Role growing + efficient = real. Role flat + way over expected = regression candidate. Compares xFP/usage trend against efficiency over expectation.", "position": None, "primary": "xfp_trend", "metrics": ["xfp_trend", "opportunity_trend", "snap_share", "ppr_over_expected_per_game", "target_share", "air_yards_share"], "sort": "desc", "samples": ["games"]},
    "ceiling_dfs": {"label": "Ceiling / DFS", "kind": "decision", "tagline": "Who can win you a week?", "description": "Sorted by boom rate: players with the per-target, per-touch, and breakaway efficiency to post a slate-breaking score.", "position": None, "primary": "boom_rate", "metrics": ["boom_rate", "expected_ppr_per_game", "fp_cv", "fpts_per_target", "yards_per_touch", "breakaway_percentage"], "sort": "desc", "samples": ["games"]},
}


# Gold lock icon for PRO-locked UI (decision pills, etc.). Theme-colored like
# the PRO badge; no emoji. Inherits gold via .am-lock-ico.
_LOCK_SVG_HTML = (
    '<span class="am-lock-ico" aria-hidden="true">'
    '<svg width="12" height="12" viewBox="0 0 12 12" fill="none">'
    '<rect x="2.5" y="5.5" width="7" height="5" rx="1.2" fill="currentColor"/>'
    '<path d="M4 5.5V4a2 2 0 0 1 4 0v1.5" stroke="currentColor" '
    'stroke-width="1.6" stroke-linecap="round"/>'
    "</svg></span>"
)


def build_advanced_metrics_body(
        has_premium: bool,
        metrics_spec: dict,
        league_id: Optional[str] = None,
        season: Optional[int] = None,
        platform: Optional[str] = None,
        is_guest: bool = False,
) -> str:
    from data_building.advanced_metrics import (
        get_available_seasons, get_available_weeks_by_season, _WEEKLY_METRICS,
        ADV_WEEKLY_METRIC_KEYS, adv_weekly_vol_spec,
        PREMIUM_METRICS, premium_metrics_exposed, PRO_METRICS,
        LEADERBOARD_METRICS,
    )
    _hide_premium = not premium_metrics_exposed()
    # Decision presets whose primary metric is PRO-gated (the four advanced
    # decision presets). Non-PRO users see them locked; tapping opens the
    # paywall instead of loading the view. Key Metrics and Start / Sit stay
    # free: their primary (expected_ppr_per_game) is free.
    _pro_presets = [
        key for key, p in ADVANCED_METRIC_PRESETS.items()
        if p.get("kind") == "decision" and p.get("primary") in PRO_METRICS
    ]
    # Public-safe "what am I missing" copy for the PRO-gated metrics and the
    # locked decision presets: names plus a one-line why-it-matters. Shipped
    # to every client (names/descriptions only, never values) so a free user
    # tapping a lock sees the metric name and why it matters instead of a
    # generic paywall headline. No em dashes in UI copy (house style).
    _PRO_METRIC_WHY = {
        "ppr_over_expected_per_game": "Shows whether a player's points came from efficiency or just volume. Negative FPOE flags buy-low targets whose role is better than their box score.",
        "half_ppr_over_expected_per_game": "The efficiency-vs-expectation read, tuned for half-PPR scoring.",
        "standard_over_expected_per_game": "The efficiency-vs-expectation read, tuned for non-PPR scoring.",
        "wopr": "Target share plus air-yards share in one number. One of the stickiest predictors of WR fantasy value.",
        "opportunity_trend": "Whether the player's role is growing or shrinking over the last 3 weeks. Catches breakouts and benchings before the box score does.",
        "xfp_trend": "Whether the quality of the player's opportunity is trending up. Separates real role growth from one hot week.",
        "fp_cv": "How steady the weekly scoring is. Low CV means set-and-forget starters; high CV means boom-or-bust.",
        "xfp_stddev": "How much the workload itself swings week to week. Steadier roles mean more predictable lineups.",
        "role_score": "One composite of target, carry, and red-zone share. The backbone of breakout detection.",
        "target_quality_score": "Rates the quality of a receiver's targets, not just the count. Separates target hogs from efficient ones.",
        "vorp": "Season points above a replacement-level starter, sized to your league. The single best 'how valuable is this player' number.",
        "war": "Translates VORP into wins. Tells you how many standings spots a player is actually worth.",
    }
    _pro_metric_info = {
        key: {
            "label": (LEADERBOARD_METRICS.get(key) or {}).get("label") or key,
            "why": _PRO_METRIC_WHY.get(key) or "A PRO intelligence metric.",
        }
        for key in PRO_METRICS
    }
    _pro_preset_info = {
        key: {
            "label": ADVANCED_METRIC_PRESETS[key]["label"],
            "tagline": ADVANCED_METRIC_PRESETS[key].get("tagline") or "",
        }
        for key in _pro_presets
    }
    # Public metrics also live in this table; season discovery must not depend
    # on PFF entitlement (otherwise a newly ingested season such as 2026 is
    # hidden from the selector for ordinary users).
    available_seasons: list = get_available_seasons()
    # Week-filterable metrics: usage-derived (_WEEKLY_METRICS) plus the
    # NGS/FTN/EPA metrics that have a per-week store (ADV_WEEKLY_METRIC_KEYS).
    weekly_metric_keys: list = sorted(
        set(_WEEKLY_METRICS.keys()) | set(ADV_WEEKLY_METRIC_KEYS))
    # weeklyVol: per-metric min filter spec for when week-range mode is active
    _weekly_vol_map: dict = {
        k: {"label": v["min_label"], "opts": v["min_opts"]}
        for k, v in _WEEKLY_METRICS.items()
        if v.get("min_opts")
    }
    for _k in ADV_WEEKLY_METRIC_KEYS:
        _vs = adv_weekly_vol_spec(_k)
        if _vs:
            _weekly_vol_map[_k] = _vs
    # Group metrics into <optgroup>s by category (General / Passing / Rushing / Receiving).
    _CAT_ORDER = ["Value", "General", "Passing", "Rushing", "Receiving"]
    groups: dict = {}
    for key, spec in metrics_spec.items():
        if spec.get("hidden"):
            continue
        if _hide_premium and key in PREMIUM_METRICS:
            continue  # don't offer premium (PFF) metrics on the public site
        if not has_premium and key in PRO_METRICS:
            continue  # PRO intelligence metrics stay out of the free picker
        cat = spec.get("category", "Other")
        groups.setdefault(cat, []).append((key, spec["label"]))

    def _group_key(cat):
        try:
            return _CAT_ORDER.index(cat)
        except ValueError:
            return len(_CAT_ORDER)

    preset_optgroup = '<optgroup label="Decisions">' + "".join(
        f'<option value="__preset__{key}">{_esc(p["label"])}</option>'
        for key, p in ADVANCED_METRIC_PRESETS.items() if p.get("kind") == "decision"
    ) + '</optgroup><optgroup label="Metric Sets">' + "".join(
        f'<option value="__preset__{key}">{_esc(p["label"])} Set</option>'
        for key, p in ADVANCED_METRIC_PRESETS.items() if p.get("kind") != "decision"
    ) + '</optgroup>'
    metric_options = preset_optgroup + "\n" + "\n".join(
        '<optgroup label="{label}">{cat_preset}{opts}</optgroup>'.format(
            label=cat,
            cat_preset='',
            opts="".join(
                f'<option value="{k}"{" selected" if k == "opportunity_share" else ""}>{lbl}</option>'
                for k, lbl in groups[cat]
            ),
        )
        for cat in sorted(groups, key=_group_key)
    )

    def _min_vol_cfg(spec: dict) -> Optional[dict]:
        mv = spec.get("min_vol")
        if not mv:
            return None
        # One is always selectable for explicit Week 1/small-sample analysis.
        return {"label": mv["label"], "opts": sorted(set([1, *mv["opts"]]))}

    # Glossary: every metric grouped by category (same order as the dropdown).
    # PRO-gated metrics carry the PRO badge.
    _legend_sections = []
    for cat in sorted(groups, key=_group_key):
        _rows = "".join(
            '<div class="am-legend-row">'
            '<div class="am-legend-name">{label}{pro}</div>'
            '<div class="am-legend-desc">{desc}</div>'
            '</div>'.format(
                label=_esc(metrics_spec[k]["label"]),
                pro=' <span class="nav-pro-badge">PRO</span>' if k in PRO_METRICS else "",
                desc=_esc(metrics_spec[k].get("desc") or "No description available."),
            )
            for k, _lbl in groups[cat]
        )
        _legend_sections.append(
            '<div class="am-legend-group">'
            '<div class="am-legend-grouphead">{head}</div>{rows}</div>'.format(
                head=_esc(cat), rows=_rows,
            )
        )
    legend_html = "".join(_legend_sections)

    # Week availability is the union of both weekly stores, per season.
    _weeks_by_season = get_available_weeks_by_season()

    cfg = json.dumps({
        "hasPremium": bool(has_premium),
        "proMetricInfo": _pro_metric_info,
        "proPresetInfo": _pro_preset_info,
        "leagueId": league_id or "",
        "platform": platform or "sleeper",
        "seasons": available_seasons,
        "availableWeeksBySeason": _weeks_by_season,
        "presets": ADVANCED_METRIC_PRESETS,
        "proPresets": _pro_presets,
        "weeklyMetrics": weekly_metric_keys,
        "metrics": {
            key: {
                "label": spec["label"],
                "positions": spec["positions"],
                "category": spec.get("category", "Other"),
                "lowerBetter": bool(spec.get("lower_better")),
                "efficiency": bool(spec.get("efficiency")),
                "pct": bool(spec.get("pct")),
                "pctFrac": bool(spec.get("pct_frac")),
                "desc": spec.get("desc", ""),
                "pro": key in PRO_METRICS,
                "minVol": _min_vol_cfg(spec),
                "weeklyCapable": key in weekly_metric_keys,
                "weeklyVol": _weekly_vol_map.get(key) or None,
                "subcategory": spec.get("subcategory", ""),
                "sample": (
                    "dropbacks" if key in {"epa_per_play", "cpoe", "success_rate", "sack_rate", "scramble_rate", "qb_hit_rate"}
                    else "attempts" if spec.get("category") == "Passing"
                    else "carries" if spec.get("category") == "Rushing" and key != "total_carries"
                    else "receptions" if key in {"yards_after_catch_per_reception", "ngs_avg_yac", "ngs_avg_expected_yac", "ngs_avg_yac_above_expectation"}
                    else "targets" if spec.get("category") == "Receiving" and key != "total_targets"
                    else "games"
                ),
            }
            for key, spec in metrics_spec.items()
            if not spec.get("hidden")
               and not (_hide_premium and key in PREMIUM_METRICS)
               and not (not has_premium and key in PRO_METRICS)
        },
    })

    # available_seasons only contains seasons with real data, so [0] is always right.
    if season and season in available_seasons:
        _default_season = season
    else:
        _default_season = available_seasons[0] if available_seasons else None

    season_options = "".join(
        f'<option value="{s}"{"selected" if s == _default_season else ""}>{s}</option>'
        for s in available_seasons
    ) or '<option value="">No data</option>'

    # Decision pills: one per decision preset, in dict order (Key Metrics first).
    # PRO presets render locked for non-PRO users; tapping opens the paywall.
    def _pill_html(key, p):
        locked = (not has_premium) and key in _pro_presets
        cls = "am-pill am-pill-locked" if locked else "am-pill"
        lock = " " + _LOCK_SVG_HTML if locked else ""
        return (
            '<button type="button" class="{cls}" role="tab" data-preset="{key}"'
            ' data-locked="{locked}" title="{tag}">{label}{lock}</button>'
        ).format(
            cls=cls, key=key, locked="1" if locked else "0",
            tag=_esc(p.get("tagline") or p.get("description") or p["label"]),
            label=_esc(p["label"]), lock=lock,
        )
    decision_pills = "".join(
        _pill_html(key, p)
        for key, p in ADVANCED_METRIC_PRESETS.items() if p.get("kind") == "decision"
    )
    # "All metrics" restores the classic browse-everything view.
    decision_pills += '<button type="button" class="am-pill am-pill-all" data-preset="" title="Browse all metrics by category">All metrics</button>'

    # Custom metric sets (saved to localStorage) are a signed-in perk: guests
    # get no dropdown, no Save/Delete buttons.
    _custom_sets_html = (
        '<select id="amSavedSet" class="am-select am-mobile-filter" title="Saved metric sets" style="max-width:150px"><option value="">Custom sets…</option></select>\n'
        '          <span id="amActiveSet" class="am-ctrl-label am-mobile-filter" title="Active metric set">Custom</span>\n'
        '          <button id="amSaveSetBtn" type="button" class="am-add-stat-btn am-mobile-filter" title="Save or update a named custom set">Save set</button>\n'
        '          <button id="amDeleteSetBtn" type="button" class="am-add-stat-btn am-clear-btn am-mobile-filter" title="Delete the selected custom set">Delete</button>'
        if not is_guest else ""
    )

    html = """
    <div class="card central">
      <div class="card-header am-head">
        <div class="am-head-title">
          <h2>Advanced Metrics</h2>
          <div class="am-head-desc" style="font-size:14px;color:var(--text-muted);margin-top:4px;">
            Rank every player by a single advanced metric. Bars are relative to the leader.
          </div>
        </div>
        <div class="am-head-actions" style="display:flex;gap:8px;flex-shrink:0;flex-wrap:wrap;">
          <button id="amGraphBtn" type="button" class="am-legend-btn" title="Graph metrics" onclick="amOpenGraph()">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style="flex-shrink:0"><path d="M2 2v10h10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><circle cx="5" cy="9" r="1.3" fill="currentColor"/><circle cx="8" cy="5.5" r="1.3" fill="currentColor"/><circle cx="11" cy="7.5" r="1.3" fill="currentColor"/></svg>
            <span class="am-legend-btn-label">Graph Metrics</span>
          </button>
          <button id="amLegendBtn" type="button" class="am-legend-btn" title="Metric glossary"
            onclick="document.getElementById('amLegendModal').style.display='flex'">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style="flex-shrink:0"><circle cx="7" cy="7" r="6" stroke="currentColor" stroke-width="1.5"/><path d="M7 6.5v3M7 4.5h.01" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
            <span class="am-legend-btn-label">Metric Glossary</span>
          </button>
          <button id="amExportBtn" type="button" class="am-legend-btn" title="Download the current filtered view as a CSV">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style="flex-shrink:0"><path d="M7 1.5v7m0 0 2.3-2.3M7 8.5 4.7 6.2M2.5 10v1.5a1 1 0 0 0 1 1h7a1 1 0 0 0 1-1V10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
            <span class="am-legend-btn-label">CSV</span>
          </button>
        </div>
      </div>

      <div id="amLegendModal" class="am-legend-modal" style="display:none;"
        onclick="if(event.target===this)this.style.display='none'">
        <div class="am-legend-card" role="dialog" aria-label="Metric glossary">
          <div class="am-legend-head">
            <span>Metric Glossary <span style="font-size:11px;font-weight:500;opacity:.55;margin-left:6px;">{count} metrics</span></span>
            <button type="button" class="am-legend-close" aria-label="Close"
              onclick="document.getElementById('amLegendModal').style.display='none'">&times;</button>
          </div>
          <div class="am-legend-body">__LEGEND__</div>
        </div>
      </div>
      <div class="card-body" style="padding-top:0;">

        <div class="am-toolbar" id="amToolbar">
        <div class="am-controls" id="amControls">
          <div class="am-ctrl">
            <div class="am-metric-picker" id="amMetricPickerWrap">
              <button type="button" class="am-select am-metric-btn" id="amMetricBtn" aria-haspopup="listbox" aria-expanded="false" aria-label="Primary metric">
                <span id="amMetricBtnLabel"></span>
                <i class="fa-solid fa-chevron-down am-metric-chevron"></i>
              </button>
              <div class="am-stat-picker am-metric-dropdown" id="amMetricDropdown" role="listbox" style="display:none;right:auto;left:0;"></div>
            </div>
            <select id="amMetric" style="display:none">__METRIC_OPTIONS__</select>
          </div>
          <div class="am-ctrl am-ctrl-season" id="amSeasonCtrl">
            <div class="am-season-row">
              <div class="am-season-multi" id="amSeasonMulti">
                <button type="button" class="am-select am-season-select am-season-btn" id="amSeasonBtn"
                  aria-haspopup="listbox" aria-expanded="false" aria-label="Select seasons">
                  <span id="amSeasonBtnLabel"></span>
                  <i class="fa-solid fa-chevron-down am-metric-chevron"></i>
                </button>
                <div class="am-season-menu" id="amSeasonMenu" role="listbox" aria-multiselectable="true" style="display:none;"></div>
              </div>
              <div id="amCombineToggle" class="otc-day-filters am-combine-toggle" style="display:none;" title="Each year keeps one row per player-season. Combine merges selected years into one row per player.">
                <button type="button" class="otc-day-filter am-combine-btn active" data-combine="0">Each year</button>
                <button type="button" class="otc-day-filter am-combine-btn" data-combine="1">Combine</button>
              </div>
            </div>
            <select id="amSeason" class="am-select am-season-select" style="display:none" aria-hidden="true">__SEASON_OPTIONS__</select>
          </div>
          <div class="am-ctrl am-ctrl-search">
            <input id="amSearch" type="text" autocomplete="off" placeholder="Search players…" class="am-search" aria-label="Search players">
          </div>
        </div>

        <div class="am-subcontrols">
          <div id="amPositions" class="otc-day-filters am-positions">
            <button class="otc-day-filter am-pos active" data-pos="ALL">All</button>
            <button class="otc-day-filter am-pos" data-pos="QB">QB</button>
            <button class="otc-day-filter am-pos" data-pos="RB">RB</button>
            <button class="otc-day-filter am-pos" data-pos="WR">WR</button>
            <button class="otc-day-filter am-pos" data-pos="TE">TE</button>
          </div>
          <!-- Add Metric picker lives here so it's always accessible even when
               the compare bar is collapsed (no extra metrics selected). -->
          <div id="amAddStatWrap" style="position:relative;flex-shrink:0;">
            <button id="amAddStatBtn" type="button" class="am-add-stat-btn">&#43; Metric</button>
            <div id="amStatPicker" class="am-stat-picker" style="display:none;"></div>
          </div>
          <button id="amAddFilterBtn" type="button" class="am-add-stat-btn">&#43; Filter</button>
          __CUSTOM_SETS__
          <button id="amFiltersBtn" type="button" class="am-sort-btn am-filters-btn">Filters &#9662;</button>
          <span class="am-ctl-divider" aria-hidden="true"></span>
          <label class="am-roster-toggle am-toggle-chip am-mobile-filter" id="amTrendToggleWrap" title="Show each player's recent usage trend (last 6 weeks) next to the metric">
            <input type="checkbox" class="am-toggle-input" id="amTrendToggle">
            <span class="am-toggle-box" aria-hidden="true"><svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M1.5 5.2 4 7.7 8.5 2.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg></span>
            <span class="am-toggle-text">Usage trends</span>
          </label>
          <label class="am-roster-toggle am-toggle-chip am-mobile-filter" id="amRosterToggleWrap" style="display:none;">
            <input type="checkbox" class="am-toggle-input" id="amRosterToggle">
            <span class="am-toggle-box" aria-hidden="true"><svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M1.5 5.2 4 7.7 8.5 2.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg></span>
            <span class="am-toggle-text">My roster only</span>
          </label>
        </div>

        <!-- Decision presets: one-tap views organized by the question being answered.
             Pills are server-rendered; JS wires clicks and the active state. -->
        <div class="am-decisions" id="amDecisionPills" role="tablist" aria-label="Decision views">
          __DECISION_PILLS__
        </div>
        <div class="am-preset-tagline" id="amPresetTagline" aria-live="polite"></div>

        <!-- What changed: usage/xFP movers and efficiency outliers, filled by JS. -->
        <div class="am-movers" id="amMovers" style="display:none;">
          <div class="am-movers-head" id="amMoversHead" role="button" tabindex="0" aria-expanded="true" title="Collapse">
            <span class="am-ctrl-label">What changed</span>
            <span class="am-movers-sub" id="amMoversSub"></span>
            <span class="am-movers-chev" id="amMoversChev"><svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M4 6l4 4 4-4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg></span>
          </div>
          <div class="am-movers-groups" id="amMoversGroups"></div>
        </div>

        <div class="am-ctrl am-mobile-filter am-ctrl-weekbar" id="amWeekCtrl">
          <div class="am-weekbar-head">
            <label class="am-ctrl-label">Week Range</label>
            <div class="otc-day-filters am-quick-ranges" id="amQuickRanges">
              <button type="button" class="otc-day-filter am-qr active" data-range="">Season</button>
              <button type="button" class="otc-day-filter am-qr" data-range="last2">Last 2</button>
              <button type="button" class="otc-day-filter am-qr" data-range="last4">Last 4</button>
            </div>
          </div>
          <div id="amWkBarHost"></div>
        </div>

        <!-- Compare bar: only visible when extra metrics or pinned-compare is active. -->
        <div id="amCompareBar" class="am-compare-bar" style="display:none;">
          <div id="amCompareChips" class="am-compare-chips"></div>
          <button id="amComparePinnedBtn" type="button" class="am-add-stat-btn" style="display:none;">&#8645; Compare Pinned</button>
          <button id="amClearExtrasBtn" type="button" class="am-add-stat-btn am-clear-btn" style="display:none;" onclick="amClearExtras()">&#10005; Clear</button>
          <button id="amClearExtrasLink" type="button" class="am-clear-link" onclick="amClearExtras()">Clear all</button>
        </div>

        <!-- Filter bar: team/sort always; age, vol, and combo chips when active. -->
        <div id="amFilterBar" class="am-filter-bar" style="display:none;">
          <div class="am-filter-ctrl am-mobile-filter" id="amTeamCtrl">
            <span class="am-filter-label">Team</span>
            <select id="amTeamFilter" class="am-select am-season-select">
              <option value="">All Teams</option>
            </select>
          </div>
          <div class="am-filter-ctrl am-mobile-filter" id="amSortCtrl">
            <span class="am-filter-label">Sort</span>
            <button id="amSortBtn" type="button" class="am-sort-btn">High &rarr; Low</button>
          </div>
          <button id="amAddFilterBtnM" type="button" class="am-add-stat-btn am-add-filter-m">&#43; Filter</button>
          <div class="am-filter-chips" id="amFilterChips"></div>
          <div class="am-age-wrap am-mobile-filter" id="amAgeWrap" style="display:none;">
            <span class="am-filter-label">Age</span>
            <input type="number" id="amAgeMin" class="am-age-input" placeholder="Min" min="18" max="45">
            <span class="am-filter-sep">&#8211;</span>
            <input type="number" id="amAgeMax" class="am-age-input" placeholder="Max" min="18" max="45">
          </div>
          <div class="am-vol-ctrl am-mobile-filter" id="amGamesCtrl" style="display:none;">
            <span class="am-filter-label" id="amVolLabel">Min</span>
            <select id="amMinGames" class="am-select am-season-select" style="font-size:12px;padding:4px 8px;"></select>
          </div>
          <div id="amFilterForm" class="am-filter-form" style="display:none;">
            <select id="amFilterKey" class="am-select am-season-select" style="min-width:110px;font-size:12px;padding:5px 8px;"></select>
            <select id="amFilterOp" class="am-select am-season-select" style="min-width:52px;font-size:12px;padding:5px 8px;">
              <option value="gte">&ge;</option>
              <option value="lte">&le;</option>
            </select>
            <input type="number" id="amFilterVal" class="am-age-input" placeholder="Value" style="width:70px;">
            <button id="amFilterApply" type="button" class="am-filter-apply-btn">Add</button>
            <button id="amFilterCancel" type="button" class="am-filter-cancel-btn">Cancel</button>
          </div>
        </div>
        </div>

        <div id="amCompareModal" class="am-legend-modal" style="display:none;"
          onclick="if(event.target===this){this.style.display='none';var b=document.getElementById('amCompareBody');if(b)b.dataset.cmpReady='';}">
          <div class="am-legend-card am-cmp-card" role="dialog" aria-label="Compare pinned players">
            <div class="am-legend-head">
              <span>Compare Pinned Players</span>
              <button type="button" class="am-legend-close" aria-label="Close"
                onclick="document.getElementById('amCompareModal').style.display='none';var b=document.getElementById('amCompareBody');if(b)b.dataset.cmpReady='';">&times;</button>
            </div>
            <div class="am-legend-body" id="amCompareBody"></div>
          </div>
        </div>

        <!-- Graph metrics modal: scatter of X vs Y (+ optional bubble metric) -->
        <div id="amGraphModal" class="am-legend-modal" style="display:none;"
          onclick="if(event.target===this)this.style.display='none'">
          <div class="am-legend-card am-graph-card" role="dialog" aria-label="Graph metrics">
            <div class="am-legend-head">
              <span>Graph Metrics <span id="amGraphCtxNote" style="font-size:11px;font-weight:500;opacity:.55;margin-left:6px;"></span></span>
              <button type="button" class="am-legend-close" aria-label="Close"
                onclick="document.getElementById('amGraphModal').style.display='none'">&times;</button>
            </div>
            <button type="button" class="am-graph-ctrl-toggle" id="amGraphCtrlToggle" onclick="amToggleGraphControls()">
              <span id="amGraphCtrlLabel">Settings</span>
              <span id="amGraphCtrlChev">&#9660;</span>
            </button>
            <div class="am-graph-controls">
              <div class="am-gctrl"><label for="amGraphX">X axis</label><select id="amGraphX" onchange="amRenderGraph()"></select></div>
              <div class="am-gctrl"><label for="amGraphY">Y axis</label><select id="amGraphY" onchange="amRenderGraph()"></select></div>
              <div class="am-gctrl"><label for="amGraphZ">Bubble size</label><select id="amGraphZ" onchange="amRenderGraph()"></select></div>
              <div class="am-gctrl"><label for="amGraphTopN">Show</label><select id="amGraphTopN" onchange="amRenderGraph()">
                <option value="25">Top 25 by X</option>
                <option value="50">Top 50 by X</option>
                <option value="75">Top 75 by X</option>
              </select></div>
              <div class="am-gctrl" id="amGraphVolCtrl" style="display:none;">
                <label id="amGraphVolLabel">Min</label>
                <select id="amGraphMinVolSel"></select>
              </div>
              <div class="am-gctrl am-graph-actions">
                <label>&nbsp;</label>
                <div style="display:flex;gap:6px;">
                  <button type="button" id="amGraphThemeBtn" class="am-add-stat-btn" onclick="amToggleGraphTheme()" title="Toggle light / dark"></button>
                  <button type="button" id="amGraphDownloadBtn" class="am-add-stat-btn" onclick="amDownloadGraph()" title="Download image">
                    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" style="vertical-align:-1px"><path d="M8 1.5v8M8 9.5 5.5 7M8 9.5 10.5 7" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M3 11v3.5h10V11" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
                    Download
                  </button>
                  <button type="button" id="amGraphCopyBtn" class="am-add-stat-btn am-graph-copy-btn" onclick="amCopyGraphLink()" title="Copy a link that reopens this graph">
                    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" style="vertical-align:-1px"><path d="M6.5 9.5a2.5 2.5 0 0 0 3.6.1l2-2a2.5 2.5 0 0 0-3.5-3.6l-1 1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M9.5 6.5a2.5 2.5 0 0 0-3.6-.1l-2 2a2.5 2.5 0 0 0 3.5 3.6l1-1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
                    Copy link
                  </button>
                </div>
              </div>
            </div>
            <div class="otc-day-filters am-graph-pos-bar" id="amGraphPosBar">
              <button class="otc-day-filter am-pos active" data-gpos="">All</button>
              <button class="otc-day-filter am-pos" data-gpos="QB">QB</button>
              <button class="otc-day-filter am-pos" data-gpos="RB">RB</button>
              <button class="otc-day-filter am-pos" data-gpos="WR">WR</button>
              <button class="otc-day-filter am-pos" data-gpos="TE">TE</button>
            </div>
            <div class="am-graph-plot-wrap">
              <div id="amGraphPlot"><div class="am-graph-empty">Pick two metrics to plot.</div></div>
              <div id="amGraphHover" class="am-graph-hover"></div>
              <div id="amGraphTip" class="am-graph-tip"></div>
            </div>
          </div>
        </div>

        <div id="amEmpty" style="display:none;">
          <div class="empty-state">
            <span class="empty-state-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19V5"/><path d="M4 19h16"/><rect x="7" y="11" width="3" height="5" rx="1"/><rect x="12.5" y="8" width="3" height="8" rx="1"/><rect x="18" y="13" width="3" height="3" rx="1" opacity=".5"/></svg></span>
            <p class="empty-state-title">No data yet</p>
            <p class="empty-state-msg">This metric doesn’t have enough sample to chart for the current filters.</p>
          </div>
        </div>

        <div id="amAvgNote" class="am-avg-note" style="display:none;">
          <button type="button" id="amAvgToggle" class="am-avg-toggle" aria-expanded="false" aria-controls="amAvgGrid">
            <span id="amAvgToggleText"></span>
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M4 6l4 4 4-4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
          </button>
          <div class="am-avg-full" id="amAvgFull">
            <span id="amAvgNoteText"></span>
            <span id="amTrendLegend" class="am-trend-legend" style="display:none;">
              <span class="am-trend-up">&#8593;</span><span class="am-trend-down">&#8595;</span>
              vs last season
            </span>
          </div>
          <div class="am-avg-grid" id="amAvgGrid" hidden></div>
        </div>

        <div id="amWeekNote" class="am-week-note" style="display:none;">
          <svg width="13" height="13" viewBox="0 0 14 14" fill="none" style="flex-shrink:0;opacity:.7"><circle cx="7" cy="7" r="6" stroke="currentColor" stroke-width="1.4"/><path d="M7 6.5v3M7 4.5h.01" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>
          <span id="amWeekNoteText">Full Season Only: this metric doesn't support week filtering.</span>
        </div>

        <div class="am-table-wrap">
        <table id="amTable" class="am-table">
          <thead>
            <tr>
              <th class="am-rank">#</th>
              <th class="am-player">Player</th>
              <th class="am-season-col" id="amSeasonColHdr" style="display:none" title="NFL season">Year</th>
              <th class="am-games" title="Games played">G</th>
              <th class="am-weeks" style="display:none" title="Weeks in range">Wks</th>
              <th class="am-barcell" id="amMetricHeader">–</th>
            </tr>
          </thead>
          <tbody id="amTableBody"></tbody>
        </table>
        </div>

        <div id="amPagination" class="am-pagination" style="display:none;"></div>

      </div>
    </div>
    """.replace("__METRIC_OPTIONS__", metric_options).replace("__SEASON_OPTIONS__", season_options).replace(
        "__LEGEND__", legend_html).replace("__DECISION_PILLS__", decision_pills).replace("__CUSTOM_SETS__", _custom_sets_html).replace("{count}", str(sum(1 for s in metrics_spec.values() if not s.get("hidden"))))

    style = """
    <style>
      /* Widen the card on desktop so 7 metric columns fit comfortably */
      .card.central:has(#amTable) { max-width:1400px; }
      .am-legend-btn {
        flex-shrink:0; display:flex; align-items:center; gap:6px;
        padding:5px 10px; border:1px solid var(--border); border-radius:8px;
        background:var(--card); color:var(--text-muted); font-size:12px;
        cursor:pointer; white-space:nowrap; text-align:left; transition:background .14s;
      }
      .am-legend-btn:hover { background:var(--row); }
      .am-legend-modal {
        position:fixed; inset:0; z-index:1000; display:flex; align-items:center;
        justify-content:center; padding:20px; background:rgba(0,0,0,.5);
      }
      .am-legend-card {
        background:var(--card); border:1px solid var(--border); border-radius:14px;
        width:100%; max-width:560px; max-height:82vh; display:flex; flex-direction:column;
        box-shadow:0 16px 48px rgba(0,0,0,.3); overflow:hidden;
      }
      .am-legend-head {
        display:flex; align-items:center; justify-content:space-between;
        padding:14px 18px; border-bottom:1px solid var(--border);
        font-size:15px; font-weight:800; color:var(--text);
      }
      .am-legend-close {
        border:none; background:none; color:var(--text-muted); font-size:24px;
        line-height:1; cursor:pointer; padding:0 4px;
      }
      .am-legend-close:hover { color:var(--text); }
      .am-legend-body { padding:8px 18px 18px; overflow-y:auto; }
      .am-legend-group { margin-top:14px; }
      .am-legend-grouphead {
        font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.05em;
        color:var(--accent); margin-bottom:6px; padding-bottom:4px;
        border-bottom:1px solid var(--border);
      }
      .am-legend-row { padding:7px 0; border-bottom:1px solid var(--border); }
      .am-legend-row:last-child { border-bottom:none; }
      .am-legend-name { font-size:13px; font-weight:700; color:var(--text); }
      .am-legend-desc { font-size:12px; color:var(--text-muted); margin-top:2px; line-height:1.4; }
      @media (max-width:600px) {
        .am-legend-btn { padding:6px 10px; font-size:11px; }
      }
      .am-toolbar { margin:12px 0 4px; }
      /* Decision preset pills */
      .am-decisions { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:10px 0 2px; }
      .am-decisions-label { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.05em; color:var(--text-muted); }
      .am-pill {
        padding:7px 14px; border:1px solid var(--border); border-radius:999px;
        background:var(--card); color:var(--text); font-size:13px; font-weight:600;
        cursor:pointer; white-space:nowrap; transition:background .14s, border-color .14s;
      }
      .am-pill:hover { background:var(--row); }
      .am-pill.active { background:var(--accent); border-color:var(--accent); color:#fff; }
      .am-pill-all { border-style:dashed; color:var(--text-muted); }
      .am-pill-locked { opacity:.65; border-style:dashed; }
      .am-preset-tagline { font-size:12.5px; color:var(--text-muted); font-style:italic; margin:2px 0 6px; min-height:0; }
      .am-preset-tagline:empty { display:none; }
      /* What-changed movers strip */
      .am-movers { margin:8px 0 4px; padding:10px 12px; border:1px solid var(--border); border-radius:12px; background:var(--card); }
      .am-movers-head { display:flex; align-items:center; gap:8px; margin-bottom:8px; cursor:pointer; user-select:none; }
      .am-movers-head .am-ctrl-label { flex-shrink:0; }
      .am-movers-chev { margin-left:auto; display:flex; align-items:center; color:var(--text-muted); transition:transform .15s ease; }
      .am-movers.collapsed .am-movers-chev { transform:rotate(-90deg); }
      .am-movers.collapsed .am-movers-groups { display:none; }
      .am-movers.collapsed .am-movers-head { margin-bottom:0; }
      .am-movers-sub { font-size:11.5px; color:var(--text-muted); }
      .am-movers-groups { display:flex; gap:14px; overflow-x:auto; padding-bottom:2px; }
      .am-movers-group { min-width:200px; flex:1; }
      .am-movers-ghead { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.05em; margin-bottom:6px; display:flex; align-items:center; gap:6px; }
      .am-movers-ghead.up { color:#2e9e5b; }
      .am-movers-ghead.down { color:#d64545; }
      .am-movers-ghead.out { color:var(--accent); }
      /* When only one group has data, show chips in a wrapped grid so it doesn't look sparse. */
      .am-movers-groups.single { display:block; }
      .am-movers-groups.single .am-movers-group { min-width:0; display:flex; flex-wrap:wrap; gap:8px; }
      .am-movers-groups.single .am-movers-ghead { width:100%; margin-bottom:2px; }
      .am-movers-groups.single .am-mover-chip { width:auto; flex:1 1 220px; max-width:320px; }
      .am-mover-chip {
        display:flex; align-items:center; gap:8px; width:100%; text-align:left;
        padding:5px 8px; border:none; border-radius:8px; background:none;
        cursor:pointer; font-size:12.5px; color:var(--text);
      }
      .am-mover-chip:hover { background:var(--row); }
      .am-mover-name { font-weight:700; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
      .am-mover-meta { color:var(--text-muted); font-size:11px; white-space:nowrap; }
      .am-mover-val { margin-left:auto; font-weight:700; font-variant-numeric:tabular-nums; white-space:nowrap; }
      .am-mover-val.up { color:#2e9e5b; }
      .am-mover-val.down { color:#d64545; }
      @media (max-width:600px) {
        .am-movers-groups { gap:10px; }
        .am-movers-group { min-width:170px; }
        .am-pill { padding:6px 11px; font-size:12px; }
      }
      .am-controls { display:flex; gap:10px; flex-wrap:wrap; align-items:flex-end; margin:0 0 8px; }
      .am-ctrl { display:flex; flex-direction:column; gap:4px; }
      .am-ctrl-search { flex:1; min-width:160px; }
      .am-ctrl-season { min-width:min(280px,100%); }
      .am-season-row { display:flex; align-items:center; gap:8px; min-width:0; }
      .am-season-row .am-season-multi { flex:1 1 140px; min-width:0; }
      .am-ctrl-weekbar { width:100%; margin:0 0 8px; }
      .am-ctrl-label { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.05em; color:var(--text-muted); }
      .am-select, .am-search, .am-sort-btn {
        padding:8px 12px; border:1px solid var(--border); border-radius:8px;
        background-color:var(--card); color:var(--text); font-size:13px; outline:none;
      }
      .am-select { min-width:180px; cursor:pointer; }
      /* Custom metric picker */
      .am-metric-picker { position:relative; }
      .am-metric-btn { min-width:180px; display:flex; align-items:center; justify-content:space-between; gap:8px; text-align:left; }
      .am-metric-chevron { font-size:10px; opacity:.55; flex-shrink:0; transition:transform .15s; }
      .am-metric-picker.open .am-metric-chevron { transform:rotate(180deg); }
      .am-stat-picker.am-metric-dropdown {
        display:none; overflow:hidden; padding:0;
        min-width:min(440px, calc(100vw - 16px));
        max-width:min(560px, calc(100vw - 16px));
        width:min(520px, calc(100vw - 16px));
        max-height:none;
      }
      .am-metric-picker.open .am-metric-dropdown {
        display:flex !important; flex-direction:column;
      }
      .am-md-search-wrap { padding:8px 10px 8px; border-bottom:1px solid var(--border); flex-shrink:0; }
      .am-md-body { display:flex; min-height:240px; flex:1 1 auto; }
      .am-md-cats {
        width:148px; flex:0 0 148px; overflow-y:auto; overflow-x:hidden;
        border-right:1px solid var(--border); overscroll-behavior:contain;
      }
      .am-md-cat {
        display:flex; align-items:center; justify-content:space-between; gap:8px;
        padding:8px 10px 8px 12px; font-size:13px; font-weight:600; color:var(--text);
        cursor:pointer; user-select:none;
      }
      .am-md-cat:hover { background:var(--row,rgba(0,0,0,.04)); }
      .am-md-cat.am-md-cat-active {
        background:color-mix(in srgb, var(--accent,#2563eb) 10%, transparent);
        color:var(--accent,#2563eb);
      }
      .am-md-cat.am-md-cat-dim { opacity:.38; }
      .am-md-cat-chevron { font-size:12px; opacity:.45; flex-shrink:0; font-weight:400; }
      .am-md-metrics { flex:1 1 auto; min-width:0; overflow-y:auto; overscroll-behavior:contain; padding:4px 0; }
      .am-md-item-label { flex:1; min-width:0; }
      .am-md-item-cat {
        margin-left:auto; flex-shrink:0;
        font-size:10px; font-weight:600; color:var(--text-muted); letter-spacing:.02em;
      }
      .am-md-empty { padding:18px 14px; font-size:12px; color:var(--text-muted); }
      @media (max-width:600px) {
        .am-md-cats { width:120px; flex-basis:120px; }
        .am-md-cat { padding:9px 8px 9px 10px; font-size:12px; }
      }
      .am-season-select { min-width:90px; }
      .am-season-multi { position:relative; }
      .am-season-btn { display:flex; align-items:center; justify-content:space-between; gap:8px; width:100%; text-align:left; cursor:pointer; }
      .am-season-btn #amSeasonBtnLabel { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; min-width:0; }
      .am-season-menu {
        position:absolute; z-index:40; top:calc(100% + 4px); left:0; min-width:100%;
        background:var(--card); border:1px solid var(--border); border-radius:10px;
        box-shadow:0 8px 24px rgba(0,0,0,.18); padding:6px 0; max-height:280px; overflow-y:auto;
      }
      .am-season-opt {
        display:flex; align-items:center; gap:8px; width:100%; padding:7px 12px;
        background:transparent; border:0; color:var(--text); font-size:13px; font-weight:600;
        cursor:pointer; text-align:left;
      }
      .am-season-opt:hover { background:var(--row,rgba(0,0,0,.04)); }
      .am-season-opt input { margin:0; accent-color:var(--accent,#2563eb); cursor:pointer; }
      .am-season-opt-hint { padding:6px 12px 4px; font-size:11px; color:var(--text-muted); }
      .am-season-col { width:48px; text-align:center; color:var(--text-muted); font-size:12px; white-space:nowrap; }
      .am-table th.am-season-col, td.am-season-col { border-left:1px solid var(--border); }
      .am-search { width:100%; box-sizing:border-box; }
      .am-sort-btn { cursor:pointer; font-weight:600; white-space:nowrap; }
      /* Week Range header: label + quick-range chips share one row so the
         presets don't add height or push the bar out of line. */
      .am-weekbar-head { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
      /* Quick week-range chips (Season / Last 2 / Last 4) */
      .am-quick-ranges { display:flex; gap:6px; }
      /* Compare position-average baseline column */
      .am-cmp-baseline-head .am-cmp-head-name { color:var(--text-muted); }
      .am-cmp-baseline { opacity:.75; border-left:1px dashed var(--border); }
      /* Subcontrols row: positions + action buttons + toggles */
      .am-subcontrols { display:flex; align-items:center; gap:6px; margin-bottom:8px; flex-wrap:wrap; }
      .am-positions { display:flex; gap:6px; flex:1 1 auto; min-width:0; overflow-x:auto; padding-bottom:1px; }
      .am-combine-toggle { display:flex; gap:6px; flex-shrink:0; }
      .am-roster-toggle { flex-shrink:0; }
      .am-filters-btn { display:none; }
      /* Mobile-only add-filter button living inside the Filters panel; on
         desktop the standalone + Filter chip covers this. Needs the extra class
         specificity so it beats .am-add-stat-btn's display (defined later), which
         was leaking a second "+ Filter" onto desktop. */
      .am-add-stat-btn.am-add-filter-m { display:none; }
      /* Mobile: metric full width, seasons + search below. Week/team/sort/age
         sit behind the Filters button on the position row. */
      @media (max-width:600px) {
        .am-controls { gap:8px; }
        .am-ctrl { flex:1 1 calc(50% - 4px); min-width:0; }
        .am-controls .am-ctrl:first-child { flex:1 1 100%; }
        #amSeasonCtrl { flex:1 1 100%; min-width:0; }
        .am-season-row { flex-wrap:wrap; }
        .am-ctrl-search { flex:1 1 100%; }
        .am-ctrl .am-select, .am-filter-ctrl .am-select, .am-filter-ctrl .am-sort-btn { width:100%; min-width:0; box-sizing:border-box; }
        #amToolbar:not(.am-open) .am-mobile-filter { display:none !important; }
        #amToolbar:not(.am-open) #amWeekCtrl { display:none !important; }
        .am-filters-btn { display:inline-block; flex-shrink:0; padding:6px 12px; font-size:12px; border-radius:8px; }
        /* One filter entry point on mobile: the Filters dropdown. The standalone
           + Filter chip hides; its action moves inside the opened panel. */
        #amAddFilterBtn { display:none; }
        #amFilterBar.am-mobile-open .am-add-filter-m { display:inline-block; }
        .am-subcontrols { row-gap:8px; }
        .am-positions { flex:1 1 auto; flex-wrap:wrap; overflow-x:visible; min-width:0; }
        /* Decide pills: one swipeable row instead of three wrapped rows. */
        .am-decisions { flex-wrap:nowrap; overflow-x:auto; -webkit-overflow-scrolling:touch; padding-bottom:4px; }
        .am-decisions .am-decisions-label, .am-decisions .am-pill { flex-shrink:0; }
      }
      .am-pos {
        padding:6px 14px; border-radius:8px; border:1px solid var(--border);
        background:var(--card); color:var(--text-muted); cursor:pointer;
        font-size:12px; font-weight:600; transition:all .15s; white-space:nowrap;
      }
      .am-pos.active { background:var(--text); color:var(--card); border-color:var(--text); }
      /* Metric description tooltip -- chrome matches .adv-def-tip (shared --tooltip-* tokens) */
      .am-info { position:relative; display:inline-flex; margin-left:5px; color:var(--text-muted); cursor:help; vertical-align:middle; }
      .am-info i { font-size:11px; }
      /* My roster toggle */
      .am-roster-toggle {
        display:inline-flex; align-items:center; gap:6px; padding:6px 12px; border-radius:8px;
        border:1px solid var(--border); background:var(--card); color:var(--text); cursor:pointer;
        font-size:12px; font-weight:600; white-space:nowrap;
      }
      .am-roster-toggle input { width:14px; height:14px; margin:0; accent-color:var(--accent,#2563eb); cursor:pointer; }
      .am-table { width:100%; border-collapse:collapse; }
      .am-table th { text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.04em;
        color:var(--text-muted); padding:8px 10px; border-bottom:1px solid var(--border); }
      .am-table th.am-sortable { cursor:pointer; user-select:none; }
      .am-table th.am-sortable:hover { color:var(--text); }
      /* Column header with a metric definition shows a dotted underline on hover */
      .am-table th[data-def] { cursor:help; }
      .am-table th[data-def]:hover { color:var(--text); text-decoration:underline dotted; text-underline-offset:2px; }
      .am-table th.am-sort-desc::after { content:' ↓'; }
      .am-table th.am-sort-asc::after { content:' ↑'; }
      .am-table td { padding:9px 10px; border-bottom:1px solid var(--border); font-size:14px; }
      /* Column dividers */
      .am-games, .am-weeks, .am-barcell,
      .am-table th.am-games, .am-table th.am-weeks, .am-table th.am-barcell {
        border-left:1px solid var(--border);
      }
      .am-row:hover { background:var(--bg-alt, rgba(0,0,0,.03)); }
      .am-row.am-owned { background:color-mix(in srgb, var(--accent) 8%, transparent); }
      .am-row.am-owned:hover { background:color-mix(in srgb, var(--accent) 14%, transparent); }
      .am-owned-badge {
        font-size:9px; font-weight:800; letter-spacing:.04em; flex-shrink:0;
        color:var(--accent,#2563eb); border:1px solid var(--accent,#2563eb); border-radius:4px;
        padding:1px 4px; white-space:nowrap;
      }
      .am-rank { width:52px; color:var(--text-muted); font-size:12px; }
      .am-games { width:40px; text-align:center; color:var(--text-muted); font-size:12px; white-space:nowrap; }
      .am-weeks { width:36px; text-align:center; color:var(--text-muted); font-size:12px; white-space:nowrap; }
      .am-player { width:220px; max-width:220px; }
      .am-barcell { width:auto; min-width:120px; }
      .am-table-wrap { overflow-x:auto; -webkit-overflow-scrolling:touch; }
      /* Sticky Player column: stays pinned during horizontal scroll on all viewports. */
      .am-table thead th.am-player, .am-table tbody td.am-player {
        position:sticky; left:0; z-index:2; background:var(--card);
        box-shadow:8px 0 8px -8px rgba(0,0,0,.18);
      }
      .am-table thead th.am-player { z-index:3; }
      /* Keep the sticky cells opaque through the row washes (opaque equivalents
         of the translucent row backgrounds, blended over var(--card)). */
      .am-table tbody tr.am-row:hover td.am-player { background:var(--bg-alt, rgba(0,0,0,.03)); }
      .am-table tbody tr.am-row.am-owned td.am-player { background:color-mix(in srgb, var(--accent) 8%, var(--card)); }
      .am-table tbody tr.am-row.am-owned:hover td.am-player { background:color-mix(in srgb, var(--accent) 14%, var(--card)); }
      .am-table tbody tr.am-row.am-pinned td.am-player { background:color-mix(in srgb, var(--accent) 5%, var(--card)); }
      .am-table tbody tr.am-row.am-pinned:hover td.am-player,
      .am-table tbody tr.am-row.am-pinned.am-owned td.am-player { background:color-mix(in srgb, var(--accent) 10%, var(--card)); }
      /* Player cell: name on left, pos+team on right */
      .am-player td, td.am-player { display:table-cell; }
      .am-player-inner { display:flex; align-items:center; justify-content:space-between; gap:8px; overflow:hidden; }
      .am-name { font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; min-width:0; }
      .am-player-right { display:flex; align-items:center; gap:4px; flex-shrink:0; }
      .am-meta { font-size:11px; color:var(--text-muted); }
      /* Merged metric cell: bar on left, value on right */
      .am-metric-cell { display:flex; align-items:center; gap:10px; }
      .am-metric-bar { flex:1; min-width:0; }
      .am-val { font-weight:700; white-space:nowrap; font-size:13px; flex-shrink:0; min-width:46px; text-align:right; }
      .am-bar-track { position:relative; background:rgba(128,128,128,.18); border-radius:6px; height:10px; width:100%; }
      .am-bar-fill { height:100%; border-radius:6px; }
      /* Positional-average marker on each bar */
      .am-bar-avg { position:absolute; top:-3px; bottom:-3px; width:2px; background:var(--text-muted); opacity:.55; border-radius:1px; }
      .am-bar-avg-lbl {
        position:absolute; bottom:calc(100% + 1px); left:50%; transform:translateX(-50%);
        font-size:8px; font-weight:800; letter-spacing:.06em; color:var(--text-muted);
        opacity:1; white-space:nowrap;
      }
      .am-avg-note { font-size:11px; color:var(--text-muted); margin:0 0 10px; }
      .am-avg-toggle { display:none; }
      .am-avg-full { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
      .am-avg-grid { display:none; }
      .am-trend-legend { display:inline-flex; align-items:center; gap:3px; padding-left:8px; margin-left:2px; border-left:1px solid var(--border); }
      .am-trend-legend .am-trend-up, .am-trend-legend .am-trend-down { font-size:11px; }
      @media (max-width:600px){
        .am-games, .am-table th.am-games,
        .am-weeks, .am-table th.am-weeks { display:none !important; }
        .am-metric-bar { display:none; }
        .am-metric-cell { gap:0; }
        .am-val { min-width:38px; font-size:12px; text-align:center; }
        .am-controls { gap:10px; }
        .am-ctrl:first-child { flex:1 0 100%; }
        .am-ctrl-search { flex:1 0 100%; }
        .am-ctrl-season { flex:1 0 100%; }
        .am-select { min-width:0; width:100%; }
        .am-roster-toggle { padding:5px 10px; font-size:11px; }
        .am-barcell { min-width:52px; }
      }
      .am-pagination { display:flex; align-items:center; justify-content:center; gap:16px; padding:16px 0 4px; }
      .am-page-btn {
        padding:6px 14px; border-radius:8px; border:1px solid var(--border);
        background:var(--card); color:var(--text); font-size:13px; font-weight:600; cursor:pointer;
      }
      .am-page-btn:disabled { opacity:.35; cursor:not-allowed; }
      .am-page-info { font-size:13px; color:var(--text-muted); white-space:nowrap; }

      /* ── Multi-stat compare bar ─────────────────────────────────────── */
      .am-compare-bar {
        display:flex; align-items:center; gap:6px; flex-wrap:wrap;
        margin-bottom:8px;
      }
      .am-compare-chips { display:flex; flex-wrap:wrap; gap:6px; flex:1; min-width:0; }
      @media (max-width:600px) {
        .am-compare-chips { flex-wrap:nowrap; overflow-x:auto; -webkit-overflow-scrolling:touch; padding-bottom:2px; }
      }
      .am-chip {
        display:inline-flex; align-items:center; gap:5px;
        padding:3px 10px; border-radius:14px;
        border:1px solid var(--border); background:var(--bg-alt);
        font-size:12px; font-weight:600; color:var(--text-muted); white-space:nowrap;
      }
      .am-chip.am-chip-primary {
        background:var(--text); color:var(--card); border-color:var(--text);
      }
      .am-chip-x {
        border:none; background:none; padding:0; margin:0; line-height:1;
        font-size:15px; cursor:pointer; color:inherit; opacity:.6;
      }
      .am-chip-x:hover { opacity:1; }
      .am-add-stat-btn {
        display:inline-flex; align-items:center; gap:4px;
        padding:3px 11px; border-radius:14px;
        border:1px dashed var(--border); background:transparent;
        color:var(--text-muted); font-size:12px; font-weight:600; cursor:pointer;
        transition:border-color .15s, color .15s; white-space:nowrap;
      }
      .am-add-stat-btn:hover { border-color:var(--accent,#2563eb); color:var(--accent,#2563eb); }
      .am-add-stat-btn:disabled { opacity:.35; cursor:not-allowed; }
      .am-clear-btn { border-style:solid; }
      .am-clear-btn:hover { border-color:var(--loss); color:var(--loss); }
      /* Stat picker dropdown */
      .am-stat-picker {
        position:absolute; top:calc(100% + 6px); right:0; z-index:200;
        background:var(--card); border:1px solid var(--border); border-radius:12px;
        box-shadow:0 8px 32px rgba(0,0,0,.18);
        min-width:220px; width:max-content; max-width:min(320px,90vw); max-height:300px; overflow-y:auto;
        overscroll-behavior:contain;
        padding:6px 0;
      }
      .am-sp-group { border-bottom:1px solid var(--border); padding:4px 0; }
      .am-sp-group:last-child { border-bottom:none; }
      .am-sp-head {
        font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.05em;
        color:var(--text-muted); padding:4px 14px 2px;
      }
      .am-sp-cat-head {
        font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.05em;
        color:var(--accent,#2563eb); padding:8px 14px 4px;
        border-bottom:1px solid var(--border);
      }
      .am-sp-cat-note {
        font-size:11px; color:var(--text-muted); padding:8px 14px;
        border-top:1px solid var(--border); line-height:1.4; font-style:italic;
      }
      .am-sp-item {
        display:flex; align-items:center; gap:8px;
        padding:7px 14px; font-size:13px; color:var(--text); cursor:pointer;
      }
      .am-sp-item:hover { background:var(--row,rgba(0,0,0,.04)); }
      .am-sp-item.am-sp-active { color:var(--accent,#2563eb); font-weight:600; }
      .am-sp-weekly-badge {
        margin-left:auto; flex-shrink:0;
        font-size:9px; font-weight:800; letter-spacing:.04em;
        padding:1px 4px; border-radius:4px;
        background:color-mix(in srgb, var(--win) 12%, transparent); color:var(--win);
        border:1px solid color-mix(in srgb, var(--win) 30%, transparent);
      }
      .am-sp-check { width:14px; text-align:center; flex-shrink:0; font-size:12px; }
      /* Multi-stat stacked bar rows in table cell */
      .am-multi { vertical-align:middle; }
      .am-mrow {
        display:flex; align-items:center; gap:8px;
        padding:2px 0;
      }
      .am-mlabel {
        font-size:10px; font-weight:600; color:var(--text-muted);
        width:72px; flex-shrink:0;
        overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
      }
      .am-mbar-wrap { flex:1; min-width:0; }
      .am-mbar-track { background:rgba(128,128,128,.18); border-radius:4px; height:7px; }
      .am-mbar-fill { height:100%; border-radius:4px; transition:width .3s; }
      .am-mval {
        font-size:11px; font-weight:700; min-width:42px; text-align:right;
        white-space:nowrap; flex-shrink:0;
      }
      /* Skeleton shimmer for loading extra metric columns */
      @keyframes am-shimmer {
        0%   { background-position: 200% 0; }
        100% { background-position: -200% 0; }
      }
      .am-skel-bar {
        background: linear-gradient(90deg,
          var(--bg-alt,rgba(0,0,0,.06)) 25%,
          rgba(0,0,0,.1) 50%,
          var(--bg-alt,rgba(0,0,0,.06)) 75%);
        background-size: 200% 100%;
        animation: am-shimmer 1.2s ease-in-out infinite;
        border-radius: 4px; height: 10px; width: 100%;
      }
      @media (max-width:600px) { .am-skel-bar { display:none; } }
      /* Percentile badge */
      .am-val-wrap { display:flex; flex-direction:column; align-items:flex-end; flex-shrink:0; width:78px; }
      .am-pct-badge { font-size:9px; font-weight:700; letter-spacing:.02em; line-height:1.3; white-space:nowrap; opacity:.85; }
      @media (max-width:600px) { .am-pct-badge { display:none; } }
      /* Weekly usage trend column */
      .am-trendcell { white-space:nowrap; min-width:110px; }
      th.am-trendcell { font-size:11px; }
      .am-trend-inner { display:flex; align-items:center; gap:7px; }
      .am-spark { display:block; opacity:.85; }
      .am-trend-delta { font-size:10px; font-weight:700; white-space:nowrap; }
      .am-trend-delta-up   { color:var(--win); }
      .am-trend-delta-down { color:var(--loss); }
      .am-trend-delta-flat { color:var(--text-muted); opacity:.6; }
      /* PRO-locked column: blocked like a paywalled column. Blurred
         placeholder bars read as hidden data; the lock badge and the
         tinted column are one tap target that opens the paywall. */
      th.am-th-locked { color:var(--gold, #ca8a04); }
      th.am-th-locked .am-lock-ico { color:var(--gold, #ca8a04); }
      .am-cell-locked { cursor:pointer; background:rgba(202,138,4,.05); }
      .am-cell-locked:hover { background:rgba(202,138,4,.10); }
      .am-cell-locked .am-locked-blur { background:var(--text-muted); filter:blur(4px); opacity:.55; }
      .am-cell-locked .am-val-wrap { position:relative; }
      .am-lock-ico { display:inline-block; vertical-align:-2px; color:var(--gold, #ca8a04); }
      @media (max-width:600px) { .am-trendcell { min-width:80px; } .am-spark { width:44px; height:14px; } }
      /* Pinned-player comparison modal -- width grows with player count */
      .am-cmp-card { max-width:min(95vw,1100px); }
      .am-legend-body { overflow-x:auto; }
      .am-cmp-table { min-width:520px; }
      .am-cmp-table { width:100%; border-collapse:separate; border-spacing:0; font-size:13px; margin-top:4px; }
      .am-cmp-table th, .am-cmp-table td { padding:10px 12px; border-bottom:1px solid var(--border); text-align:left; vertical-align:middle; }
      .am-cmp-table thead th { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; color:var(--text-muted); border-bottom:2px solid var(--border); }
      .am-cmp-table tbody tr:hover td { background:rgba(128,128,128,.04); }
      .am-cmp-table th:first-child, .am-cmp-table td:first-child { padding-left:2px; }
      /* Player header cell */
      .am-cmp-player-head { vertical-align:top; min-width:150px; text-transform:none !important; letter-spacing:0 !important; color:var(--text) !important; }
      .am-cmp-head-name { display:flex; align-items:center; gap:6px; font-size:14px; font-weight:800; color:var(--text); }
      .am-cmp-head-pos { font-size:10px; font-weight:800; padding:1px 6px; border-radius:6px; color:#fff; letter-spacing:.02em; flex-shrink:0; }
      .am-cmp-player-meta { font-size:11px; font-weight:600; color:var(--text-muted); margin-left:5px; }
      /* Sizing feeds the custom dropdown (CSD copies font-weight/radius/padding/
         min-width from the original select onto its .csd-trigger). */
      .am-cmp-season-sel { font-weight:600; border-radius:8px; padding:4px 10px; min-width:92px; font-size:12px; margin-top:8px; }
      .am-cmp-player-head .csd-wrap { margin-top:8px; }
      /* Per-player week-range bar (reuses the page's draggable wk-bar). Ticks are
         hidden in this compact column; the selected weeks show in the note below. */
      .am-cmp-wkbar-wrap { margin-top:11px; }
      .am-cmp-wkbar-wrap .wk-bar { flex:unset; display:block; width:100%; }
      /* Compact column can't fit every week label; show every other week
         starting at W1 (W1, W3, W5 …) as a scale. Exact selected weeks show
         in the note below. */
      .am-cmp-wkbar-wrap .wk-bar-ticks { display:flex; margin-top:3px; }
      .am-cmp-wkbar-wrap .wk-tick { font-size:0; }
      .am-cmp-wkbar-wrap .wk-tick:nth-child(odd) { font-size:9px; }
      .am-cmp-wknote { margin-top:6px; font-size:10px; font-weight:700; color:var(--text-muted); }
      .am-cmp-wknote-warn { color:var(--warning); }
      .am-cmp-wknote-muted { font-weight:600; opacity:.55; }
      /* Metric rows */
      .am-cmp-metric { font-weight:700; color:var(--text-muted); white-space:nowrap; font-size:12px; }
      .am-cmp-val { font-weight:800; font-variant-numeric:tabular-nums; font-size:14px; }
      .am-cmp-best { color:var(--win); }
      .am-cmp-rank { font-size:10px; color:var(--text-muted); margin-left:6px; font-weight:700; }
      .am-cmp-bar { height:6px; border-radius:4px; background:rgba(128,128,128,.16); margin-top:6px; overflow:hidden; max-width:160px; }
      .am-cmp-bar > div { height:100%; border-radius:4px; transition:width .2s ease; }
      .am-cmp-cat-row td { font-size:10px !important; font-weight:800; text-transform:uppercase; letter-spacing:.05em;
        color:var(--text-muted); background:rgba(128,128,128,.05); padding:5px 12px !important; }
      /* YoY trend arrows - inline beside the value */
      .am-val-row { display:flex; align-items:center; gap:3px; }
      .am-trend-up   { font-size:11px; font-weight:700; color:var(--win); line-height:1; flex-shrink:0; }
      .am-trend-down { font-size:11px; font-weight:700; color:var(--loss); line-height:1; flex-shrink:0; }
      /* Pin button */
      .am-rank-cell { display:flex; align-items:center; gap:3px; }
      .am-pin-btn {
        border:none; background:none; padding:1px 2px; margin:0; line-height:1;
        cursor:pointer; color:var(--text-muted); opacity:0; transition:opacity .12s, color .12s;
        flex-shrink:0;
      }
      .am-row:hover .am-pin-btn { opacity:.55; }
      .am-pin-btn.am-pin-active { opacity:1 !important; color:var(--accent,#2563eb); }
      .am-row.am-pinned { background:color-mix(in srgb, var(--accent) 5%, transparent); }
      .am-row.am-pinned:hover { background:color-mix(in srgb, var(--accent) 10%, transparent); }
      .am-row.am-pinned.am-owned { background:color-mix(in srgb, var(--accent) 10%, transparent); }
      .am-pin-divider td { border-bottom:2px dashed var(--accent,#2563eb) !important; padding:0 !important; height:2px !important; }
      /* Mobile: pack the compare table so two players fit on screen without the
         right column getting clipped. Drop the 520px min-width, shrink padding /
         fonts, let the metric-label column wrap so player columns get the room. */
      @media (max-width:600px) {
        .am-cmp-table { min-width:0; font-size:12px; }
        .am-cmp-table th, .am-cmp-table td { padding:7px 5px; }
        .am-cmp-table th:first-child, .am-cmp-table td:first-child { padding-left:2px; }
        .am-cmp-player-head { min-width:0; }
        .am-cmp-head-name { font-size:12px; gap:4px; }
        .am-cmp-head-pos { font-size:9px; padding:1px 4px; }
        .am-cmp-metric { font-size:10px; white-space:normal; line-height:1.2; }
        .am-cmp-val { font-size:12px; }
        .am-cmp-rank { font-size:9px; margin-left:3px; }
        .am-cmp-bar { max-width:none; }
        .am-cmp-season-sel { min-width:0; width:100%; padding:3px 6px; font-size:11px; margin-top:6px; }
        .am-cmp-wkbar-wrap { margin-top:8px; }
        .am-cmp-wknote { font-size:9px; }
      }
      /* ── Filter bar ──────────────────────────────────────────────────────── */
      .am-filter-bar {
        display:flex; align-items:center; gap:8px; flex-wrap:wrap;
        margin-bottom:8px; padding-right:4px;
      }
      .am-filter-ctrl { display:flex; align-items:center; gap:6px; flex-shrink:0; }
      .am-filter-ctrl .am-select, .am-filter-ctrl .am-sort-btn {
        padding:5px 10px; font-size:12px;
      }
      .am-filter-chips { display:flex; flex-wrap:wrap; gap:5px; flex:1; min-width:0; }
      @media (max-width:600px) {
        .am-filter-chips { flex-wrap:nowrap; overflow-x:auto; -webkit-overflow-scrolling:touch; }
      }
      .am-filter-chip {
        display:inline-flex; align-items:center; gap:4px;
        padding:3px 9px; border-radius:8px;
        border:1px solid var(--accent,#2563eb); background:color-mix(in srgb, var(--accent) 8%, transparent);
        font-size:12px; font-weight:600; color:var(--accent,#2563eb); white-space:nowrap;
      }
      .am-filter-label { font-size:11px; font-weight:700; color:var(--text-muted); white-space:nowrap; }
      .am-filter-sep { font-size:12px; color:var(--text-muted); }
      .am-age-wrap { display:flex; align-items:center; gap:4px; flex-shrink:0; }
      .am-vol-ctrl { display:flex; align-items:center; gap:6px; flex-shrink:0; }
      .am-age-input {
        padding:5px 8px; border:1px solid var(--border); border-radius:8px;
        background:var(--card); color:var(--text); font-size:12px; width:58px;
        outline:none; box-sizing:border-box;
      }
      .am-age-input:focus { border-color:var(--accent,#2563eb); }
      .am-filter-form {
        display:flex; align-items:center; gap:5px; flex-wrap:wrap;
        padding:5px 10px; border:1px solid var(--border); border-radius:10px;
        background:var(--card); box-shadow:0 2px 8px rgba(0,0,0,.08);
      }
      .am-filter-apply-btn {
        padding:5px 12px; border-radius:8px; border:none;
        background:var(--accent,#2563eb); color:#fff;
        font-size:12px; font-weight:700; cursor:pointer;
      }
      .am-filter-apply-btn:hover { opacity:.88; }
      .am-filter-cancel-btn {
        padding:5px 10px; border-radius:8px; border:1px solid var(--border);
        background:var(--card); color:var(--text-muted);
        font-size:12px; font-weight:600; cursor:pointer;
      }
      .am-filter-cancel-btn:hover { background:var(--row,rgba(0,0,0,.04)); }
      /* Preset load button in stat picker */
      .am-sp-search-wrap { padding:8px 10px 6px; border-bottom:1px solid var(--border); }
      .am-sp-search { width:100%; padding:5px 8px; font-size:12px; border:1px solid var(--border); border-radius:6px; background:var(--bg-alt,#f1f5f9); color:var(--text); outline:none; box-sizing:border-box; }
      .am-sp-search:focus { border-color:var(--accent,#2563eb); }
      .am-sp-preset-wrap { padding:6px 10px; border-bottom:1px solid var(--border); }
      .am-sp-preset-btn {
        width:100%; padding:7px 10px; border:1px solid var(--accent,#2563eb); border-radius:8px;
        background:color-mix(in srgb, var(--accent) 7%, transparent); color:var(--accent,#2563eb);
        font-size:12px; font-weight:700; cursor:pointer; text-align:center;
      }
      .am-sp-preset-btn:hover { background:color-mix(in srgb, var(--accent) 14%, transparent); }
      /* Filter column headers inserted before primary metric column */
      th.am-filter-col-hdr { border-left:1px solid var(--border); }
      /* Mobile: show filter bar content (age inputs etc.) when Filters is open */
      @media (max-width:600px) {
        #amFilterBar.am-mobile-open { flex-wrap:wrap; }
      }
      /* Week range note */
      .am-week-note {
        display:flex; align-items:center; gap:6px;
        font-size:12px; font-weight:600; color:var(--text-muted);
        background:color-mix(in srgb, var(--warning) 8%, transparent); border:1px solid color-mix(in srgb, var(--warning) 25%, transparent);
        border-radius:8px; padding:6px 12px; margin-bottom:10px;
      }
      /* Graph (scatter) modal */
      .am-graph-card { max-width:720px; max-height:97vh; }
      .am-graph-controls {
        display:flex; flex-wrap:wrap; gap:12px;
        padding:12px 18px; border-bottom:1px solid var(--border);
      }
      .am-graph-controls .am-gctrl { display:flex; flex-direction:column; gap:3px; flex:1 1 130px; min-width:120px; }
      .am-graph-controls label {
        font-size:10px; text-transform:uppercase; letter-spacing:.04em;
        color:var(--text-muted); font-weight:700;
      }
      .am-graph-controls select {
        font-size:12px; padding:6px 8px; border:1px solid var(--border);
        border-radius:7px; background:var(--card); color:var(--text); cursor:pointer; width:100%;
      }
      .am-graph-actions { flex:0 0 auto !important; min-width:0 !important; justify-content:flex-end; margin-left:auto; }
      .am-graph-actions .am-add-stat-btn { display:inline-flex; align-items:center; gap:5px; white-space:nowrap; }
      /* Position filter bar inside the graph modal */
      .am-graph-pos-bar {
        display:flex; gap:6px; padding:8px 18px;
        border-bottom:1px solid var(--border); flex-wrap:wrap;
      }
      /* Plot area flexes and scrolls inside the 82vh card so controls never clip. */
      .am-graph-plot-wrap { position:relative; padding:14px 16px 16px; overflow:auto; flex:1 1 auto; min-height:0; -webkit-overflow-scrolling:touch; }
      /* Hover/tap card: player headshot + the selected stats. Uses site-theme
         vars (it's screen chrome, not part of the exported image). */
      .am-graph-hover {
        position:absolute; z-index:6; pointer-events:none; display:none;
        background:var(--tooltip-bg,var(--card)); color:var(--tooltip-fg,var(--text)); border:1px solid var(--tooltip-border,var(--border)); border-radius:var(--tooltip-radius,10px);
        box-shadow:var(--tooltip-shadow,0 10px 30px rgba(0,0,0,.30)); padding:9px 11px;
        min-width:150px; max-width:240px;
      }
      .am-graph-hover.show { display:block; }
      .am-graph-hover.pinned { pointer-events:auto; border-color:var(--accent,#f59e0b); box-shadow:0 10px 30px rgba(0,0,0,.30),0 0 0 2px color-mix(in srgb,var(--accent,#f59e0b) 25%,transparent); }
      .am-graph-hover-close { position:absolute; top:5px; right:7px; background:none; border:none; padding:2px 4px; cursor:pointer; font-size:13px; line-height:1; color:var(--text-muted); }
      .am-graph-hover-close:hover { color:var(--text); }
      .am-graph-hover-top { display:flex; align-items:center; gap:9px; }
      .am-graph-hover-hs {
        width:42px; height:42px; border-radius:8px; object-fit:cover;
        background:var(--row,rgba(0,0,0,.06)); flex-shrink:0;
      }
      .am-graph-hover-nm { font-size:13px; font-weight:800; color:var(--text); line-height:1.15; }
      .am-graph-hover-pos { font-size:11px; color:var(--text-muted); font-weight:600; margin-top:1px; }
      .am-graph-hover-stats { margin-top:7px; display:flex; flex-direction:column; gap:3px; font-size:12px; color:var(--text); }
      .am-graph-hover-stats .k { color:var(--text-muted); }
      /* Tap-to-inspect line (touch has no hover for the <title> tooltips). */
      .am-graph-tip {
        margin-top:8px; min-height:18px; text-align:center;
        font-size:12px; color:var(--text-muted); line-height:1.4;
      }
      .am-graph-tip b { color:var(--text); }
      /* Mobile controls toggle button (hidden on desktop) */
      .am-graph-ctrl-toggle {
        display:none; width:100%; padding:7px 14px;
        border:none; border-bottom:1px solid var(--border);
        background:var(--bg-alt,rgba(0,0,0,.03)); cursor:pointer;
        font-size:11px; font-weight:700; color:var(--text-muted);
        text-transform:uppercase; letter-spacing:.04em;
        align-items:center; justify-content:space-between;
      }
      .am-graph-ctrl-toggle:active { background:var(--row,rgba(0,0,0,.06)); }
      /* Controls hidden state (mobile) */
      .am-graph-controls.am-ctrl-hidden,
      .am-graph-pos-bar.am-ctrl-hidden { display:none !important; }
      @media (max-width:600px) {
        .am-graph-ctrl-toggle { display:flex; }
        .am-legend-modal { padding:8px; }
        .am-graph-card { max-height:92vh; }
        .am-graph-controls { gap:8px; padding:10px 12px; }
        .am-graph-controls .am-gctrl { flex:1 1 46%; min-width:0; }
        .am-graph-actions { flex:1 1 100% !important; }
        .am-graph-actions > div { width:100%; }
        .am-graph-actions .am-add-stat-btn { flex:1 1 0; justify-content:center; padding:8px 10px; }
        .am-graph-plot-wrap { padding:10px 10px 12px; }
        .am-graph-dot { stroke-width:1.25; }
      }
      .am-graph-svg { width:100%; height:auto; display:block; touch-action:none; }
      .am-graph-empty { text-align:center; color:var(--text-muted); font-size:13px; padding:48px 0; }
      .am-graph-axis { stroke:var(--border); }
      .am-graph-tick { fill:var(--text-muted); font-size:10px; }
      .am-graph-axislbl { fill:var(--text); font-size:11px; font-weight:700; }
      .am-graph-ptlbl { fill:var(--text-muted); font-size:9px; pointer-events:none; }
      .am-graph-dot { cursor:pointer; transition:fill-opacity .12s; }
      .am-graph-dot:hover { fill-opacity:1 !important; }

      /* og=1 social-preview render mode: turn the graph modal into a clean,
         full-bleed 1200x630 dark canvas with the scatter centered. Rather than
         hiding the modal's own ancestors (which would also hide the modal), we
         lay an opaque, top-of-stack overlay over the whole page. Used by
         /metrics/og.png (headless screenshot). */
      html.og-render, html.og-render body { background:#eef2f7 !important; margin:0 !important; padding:0 !important; overflow:hidden !important; }
      html.og-render #appSplash { display:none !important; }
      /* Any transform on an ancestor would make position:fixed relative to it,
         not the viewport -- neutralize so the overlay truly fills the frame. */
      html.og-render #app-scale { transform:none !important; }
      html.og-render #amGraphModal {
        position:fixed !important; inset:0 !important; z-index:2147483600 !important; display:flex !important;
        align-items:center !important; justify-content:center !important;
        background:radial-gradient(circle at 50% 0%, #ffffff 0%, #e2e8f0 75%) !important; padding:0 !important;
      }
      html.og-render .am-graph-card {
        width:1200px !important; height:630px !important; max-width:1200px !important; max-height:630px !important;
        border:none !important; border-radius:0 !important; box-shadow:none !important;
        background:transparent !important; display:flex !important; flex-direction:column !important;
      }
      html.og-render .am-legend-head, html.og-render .am-graph-controls,
      html.og-render #amGraphHover, html.og-render #amGraphTip { display:none !important; }
      html.og-render .am-graph-plot-wrap {
        flex:1 1 auto !important; display:flex !important; align-items:center !important; justify-content:center !important;
        padding:18px !important; overflow:hidden !important;
      }
      html.og-render #amGraphPlot { width:100%; display:flex; align-items:center; justify-content:center; }
      html.og-render .am-graph-svg { width:auto !important; height:594px !important; max-width:1164px !important; }

      /* ── Advanced Metrics mobile visual rework ──────────────────────────
         Icon-button header, search+season on one row, solid + Metric button,
         fading preset rail, wrapping chips, collapsible field averages,
         sticky player column. (The old all-caps section labels were removed
         from the markup entirely by the desktop rework; desktop rules above
         untouched.) */
      .am-clear-link { display:none; }
      @media (max-width:600px) {
        /* 1. Header: icon-only actions, description hidden. */
        .am-head .am-head-desc { display:none; }
        .am-head .am-legend-btn-label { display:none; }
        .am-head .am-legend-btn { padding:8px; }
        .am-head .am-legend-btn svg { width:16px; height:16px; }
        .am-head-actions { flex-wrap:nowrap !important; }
        /* 2. Search and season share one row: search flex-grows, season compact.
           Mockup order is search-then-season, so re-order the DOM pair. */
        #amSeasonCtrl { flex:0 1 128px; min-width:0; order:2; }
        .am-ctrl-search { flex:1 1 0; min-width:0; order:1; }
        .am-ctrl-search .am-search { font-size:14px; }
        /* 3. + Metric: solid outlined button matching the position pills,
           not the near-invisible dashed ghost. 8px radius per --radius-pill. */
        #amAddStatBtn {
          border:1px solid var(--border); border-radius:8px; background:var(--card);
          color:var(--text); padding:6px 14px; font-size:12px; font-weight:600;
        }
        #amAddStatBtn:hover { border-color:var(--accent,#2563eb); color:var(--accent,#2563eb); }
        /* 4. Decide presets: edge fade so the rail reads as swipeable.
           (The "Decide:" label was removed from the markup by the desktop
           rework; the group keeps aria-label="Decision views".) */
        .am-decisions {
          -webkit-mask-image:linear-gradient(to right,#000 88%,transparent 100%);
          mask-image:linear-gradient(to right,#000 88%,transparent 100%);
        }
        /* 5. Chips wrap to multiple lines; Clear becomes a text link. */
        .am-compare-chips { flex-wrap:wrap; overflow-x:visible; padding-bottom:0; }
        #amClearExtrasBtn { display:none !important; }
        .am-clear-link {
          display:none; border:0; background:none; padding:6px 4px; cursor:pointer;
          color:var(--text-muted); font-size:13px; font-weight:600;
          text-decoration:underline; text-underline-offset:3px; white-space:nowrap;
        }
        .am-clear-link.am-clear-link-show { display:inline-block; }
        /* 6. Field averages: collapsed one-liner, tap to expand the grid. */
        .am-avg-note { margin:2px 0 8px; }
        .am-avg-toggle {
          display:flex; align-items:center; justify-content:space-between; gap:8px;
          width:100%; padding:10px 2px; background:none; border:0; cursor:pointer;
          font-size:13px; color:var(--text-muted); text-align:left;
        }
        .am-avg-toggle #amAvgToggleText b, .am-avg-toggle #amAvgToggleText strong { color:var(--text); }
        .am-avg-toggle svg { flex-shrink:0; color:var(--text-muted); transition:transform .15s ease; }
        .am-avg-note.am-avg-open .am-avg-toggle svg { transform:rotate(180deg); }
        .am-avg-full { display:none; }
        .am-avg-note.am-avg-open .am-avg-grid {
          display:grid; grid-template-columns:1fr 1fr; gap:6px 14px;
          padding:2px 2px 10px; font-size:12.5px; color:var(--text-muted);
        }
        .am-avg-cell b, .am-avg-cell strong { color:var(--text); font-weight:600; }
        /* 7. Results table: edge fade on horizontal scroll. */
        .am-table-wrap {
          -webkit-mask-image:linear-gradient(to right,#000 92%,transparent 100%);
          mask-image:linear-gradient(to right,#000 92%,transparent 100%);
        }
      }

      /* ── Advanced Metrics desktop visual rework ───────────────────────────
         Kaedon-approved mockup: icon-only header actions, no all-caps control
         labels, compact control rows, solid + Metric / + Filter buttons,
         Usage-trends / My-roster toggle chips, unlabeled preset pills,
         single-row week range, Clear-all text link, collapsible field
         averages. The data table and all data logic are untouched.
         Rules that would change the approved mobile look stay inside
         min-width:601px; the rest are safe to share. */
      /* 1. Header: icon-only actions (labels live in title attributes). */
      .am-head .am-legend-btn-label { display:none; }
      .am-head .am-legend-btn { padding:8px; }
      .am-head .am-legend-btn svg { width:16px; height:16px; }
      /* 2. Chips row: the Clear-all text link replaces the ghost button
         everywhere now (JS already toggles .am-clear-link-show). */
      #amClearExtrasBtn { display:none !important; }
      .am-clear-link {
        display:none; border:0; background:none; padding:6px 4px; cursor:pointer;
        color:var(--text-muted); font-size:13px; font-weight:600;
        text-decoration:underline; text-underline-offset:3px; white-space:nowrap;
      }
      .am-clear-link:hover { color:var(--loss); }
      .am-clear-link.am-clear-link-show { display:inline-block; }
      /* 3. Toggle chips replace the native checkboxes on desktop. The real
         checkbox stays inside the label (visually hidden) so every existing
         change-listener keeps working with zero JS changes. */
      .am-toggle-chip { position:relative; color:var(--text-muted); }
      .am-toggle-chip .am-toggle-input {
        position:absolute; opacity:0; width:1px; height:1px; margin:0; pointer-events:none;
      }
      .am-toggle-chip .am-toggle-box {
        width:14px; height:14px; flex-shrink:0;
        border:1.5px solid color-mix(in srgb, var(--text-muted) 55%, transparent);
        border-radius:4px; display:inline-flex; align-items:center; justify-content:center;
        font-size:10px; line-height:1; color:transparent; background:transparent;
        transition:background .12s, border-color .12s, color .12s;
      }
      .am-toggle-chip .am-toggle-input:checked + .am-toggle-box {
        background:var(--accent,#2563eb); border-color:var(--accent,#2563eb); color:#fff;
      }
      .am-toggle-chip .am-toggle-input:checked ~ .am-toggle-text { color:var(--text); font-weight:700; }
      .am-toggle-chip:focus-within { outline:2px solid var(--accent,#2563eb); outline-offset:1px; }
      @media (max-width:600px) {
        /* Mobile keeps the native checkbox look from the approved mobile rework. */
        .am-toggle-chip { color:var(--text); }
        .am-toggle-chip .am-toggle-input {
          position:static; opacity:1; width:14px; height:14px; pointer-events:auto;
        }
        .am-toggle-chip .am-toggle-box { display:none; }
        .am-toggle-chip .am-toggle-text { font-weight:600; }
      }
      .am-ctl-divider { display:none; }
      @media (min-width:601px) {
        /* 4. Row 1: metric + season + search share one clean row, no labels. */
        #amControls { align-items:center; }
        #amControls > .am-ctrl:first-child { flex:0 1 250px; min-width:0; }
        #amSeasonCtrl { flex:0 1 auto; min-width:0; }
        .am-ctrl-search { flex:1 1 200px; min-width:160px; }
        #amMetricBtn { width:100%; min-width:0; }
        /* 5. Row 2: positions left, tools right; solid buttons, no "Custom" echo. */
        .am-subcontrols { gap:8px; }
        #amActiveSet { display:none; }
        #amAddStatBtn, #amAddFilterBtn, #amSaveSetBtn, #amDeleteSetBtn {
          border:1px solid var(--border); border-radius:var(--radius-pill,8px);
          background:var(--card); color:var(--text);
          padding:6px 14px; font-size:12px; font-weight:600;
        }
        #amAddStatBtn:hover, #amAddFilterBtn:hover, #amSaveSetBtn:hover {
          border-color:var(--accent,#2563eb); color:var(--accent,#2563eb);
        }
        #amDeleteSetBtn:hover { border-color:var(--loss); color:var(--loss); }
        .am-ctl-divider { display:block; width:1px; height:24px; background:var(--border); flex-shrink:0; }
        /* 6. Presets: no label in the markup; keep the rail tidy. */
        .am-decisions { margin:8px 0 2px; }
        /* 7. What-changed strip, quieter. */
        .am-movers { padding:8px 14px; }
        .am-movers-head { margin-bottom:6px; }
        /* 8. Week range: quick pills + slim slider in one compact row. */
        #amWeekCtrl { display:flex; flex-direction:row; align-items:center; gap:14px; }
        #amWeekCtrl .am-weekbar-head { flex:0 0 auto; }
        #amWeekCtrl .am-weekbar-head .am-ctrl-label { display:none; }
        #amWkBarHost { flex:1 1 auto; min-width:0; }
        /* Slimmer slider, scoped to this host so the shared wk-bar keeps its
           size in the player and compare modals. */
        #amWkBarHost .wk-bar-track { height:24px; }
        #amWkBarHost .wk-bar-bg { top:8px; bottom:8px; }
        #amWkBarHost .wk-bar-sel { top:8px; bottom:8px; }
        #amWkBarHost .wk-bar-grip { height:22px; width:20px; }
        #amWkBarHost .wk-bar-grip-l { margin-left:-10px; }
        #amWkBarHost .wk-bar-grip-r { margin-right:-10px; }
        /* 9. Field averages: collapsed one-liner, chevron expands the grid. */
        .am-avg-note { margin:2px 0 8px; }
        .am-avg-toggle {
          display:flex; align-items:center; justify-content:space-between; gap:8px;
          width:100%; padding:8px 2px; background:none; border:0; cursor:pointer;
          font-size:13px; color:var(--text-muted); text-align:left;
        }
        .am-avg-toggle #amAvgToggleText b { color:var(--text); }
        .am-avg-toggle svg { flex-shrink:0; color:var(--text-muted); transition:transform .15s ease; }
        .am-avg-note.am-avg-open .am-avg-toggle svg { transform:rotate(180deg); }
        .am-avg-full { display:none; }
        .am-avg-note.am-avg-open .am-avg-grid {
          display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
          gap:6px 18px; padding:2px 2px 10px; font-size:12.5px; color:var(--text-muted);
        }
        .am-avg-cell b { color:var(--text); font-weight:600; }
      }
    </style>
    """

    script = (
            "<script>\n(function(){\n"
            "const AM_CFG = " + cfg + ";\n"
            + _AM_JS +
            "\n})();\n</script>"
    )

    return html + style + script


# Plain JS (template literals use ${...}; kept out of any f-string).
_AM_JS = r"""
  const cfg = AM_CFG;
  // PRO-gated metric/preset names + why-it-matters, for the paywall headline
  // when a free user taps a lock (resolved in paywall.js).
  window.__brProMetricInfo = cfg.proMetricInfo || {};
  window.__brProPresetInfo = cfg.proPresetInfo || {};
  // Metric display name. PRO-gated metrics are absent from cfg.metrics for
  // free viewers, so fall back to the PRO info map: a locked column header
  // must read "FPOE/G", never the raw key "ppr_over_expected_per_game".
  function _mLabel(key) {
    return (cfg.metrics[key] && cfg.metrics[key].label)
        || (cfg.proMetricInfo && cfg.proMetricInfo[key] && cfg.proMetricInfo[key].label)
        || key;
  }
  function _mLocked(key) {
    return !cfg.hasPremium && !!(
      (cfg.metrics[key] && cfg.metrics[key].pro) ||
      (cfg.proMetricInfo && cfg.proMetricInfo[key]));
  }
  // Gold lock icon for locked PRO columns, theme-colored like the PRO badge.
  // No emoji: the icon inherits the gold via .am-lock-ico.
  const _LOCK_SVG = '<span class="am-lock-ico" aria-hidden="true">'
    + '<svg width="12" height="12" viewBox="0 0 12 12" fill="none">'
    + '<rect x="2.5" y="5.5" width="7" height="5" rx="1.2" fill="currentColor"/>'
    + '<path d="M4 5.5V4a2 2 0 0 1 4 0v1.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>'
    + '</svg></span>';
  // paywall.js loads deferred, so the brUpsell nudge infra may not exist yet
  // while this inline script runs during the initial parse. Run fn once it
  // does (deferred scripts always run before DOMContentLoaded).
  function _whenUpsellReady(fn) {
    if (window.brUpsell || document.readyState !== 'loading') { fn(); return; }
    document.addEventListener('DOMContentLoaded', fn, { once: true });
  }
  const metricSel = document.getElementById('amMetric');
  const posWrap   = document.getElementById('amPositions');
  const searchEl  = document.getElementById('amSearch');
  const sortBtn   = document.getElementById('amSortBtn');
  const seasonSel = document.getElementById('amSeason');
  const seasonCtrl= document.getElementById('amSeasonCtrl');
  const teamSel   = document.getElementById('amTeamFilter');
  const minGamesSel = document.getElementById('amMinGames');
  const gamesCtrl = document.getElementById('amGamesCtrl');
  const rosterWrap= document.getElementById('amRosterToggleWrap');
  const rosterChk = document.getElementById('amRosterToggle');
  const avgNote   = document.getElementById('amAvgNote');
  const avgNoteTxt= document.getElementById('amAvgNoteText');
  const tbody     = document.getElementById('amTableBody');
  const loading   = document.getElementById('amLoading');
  const empty     = document.getElementById('amEmpty');
  const tableWrap = document.querySelector('.am-table-wrap');
  if (!metricSel || !tbody) return;

  function amParseSeasons(raw) {
    return String(raw || '').split(',').map(function(s) { return s.trim(); })
      .filter(function(s) { return /^\d{4}$/.test(s); })
      .map(Number);
  }
  function amSelectedSeasons() { return amParseSeasons(state && state.season); }
  function amIsMultiSeason() { return amSelectedSeasons().length > 1; }
  function amIsCombine() { return !!(state && state.combine) && amIsMultiSeason(); }
  function amIsEachYear() { return amIsMultiSeason() && !amIsCombine(); }
  function amRowKey(r) {
    if (r && r.season != null && amIsEachYear()) return String(r.player_id) + ':' + r.season;
    return String(r && r.player_id != null ? r.player_id : '');
  }
  function amSeasonLabel(years) {
    years = (years || []).slice();
    if (!years.length) return 'Season';
    if (years.length === 1) return String(years[0]);
    if (years.length === 2) return years[0] + ' + ' + years[1];
    return years[0] + ' + ' + (years.length - 1) + ' more';
  }

  // ── Read URL params and pre-set controls so state inherits them ───────────
  (function() {
    const p = new URLSearchParams(window.location.search);
    const m = p.get('metric');
    if (m && Array.from(metricSel.options).some(o => o.value === m)) metricSel.value = m;
    const s = p.get('season');
    const allowed = new Set((cfg.seasons || []).map(Number));
    const years = amParseSeasons(s).filter(function(y) { return allowed.has(y); });
    if (years.length && seasonSel) {
      seasonSel.value = String(years[0]);
      seasonSel.dataset.multi = years.join(',');
    }
  })();

  // Hide season selector when only one season (or none) is available.
  if (seasonCtrl && (!cfg.seasons || cfg.seasons.length <= 1)) seasonCtrl.style.display = 'none';

  const PAGE_SIZE = 25;
  function _loadPins() { try { return new Set(JSON.parse(localStorage.getItem('am_pins') || '[]')); } catch { return new Set(); } }
  function _savePins() { try { localStorage.setItem('am_pins', JSON.stringify([...state.pinnedIds])); } catch {} }
  const _CUSTOM_SET_KEY = 'advanced_metrics_custom_sets_v1';
  function _loadCustomSets() {
    try { const v = JSON.parse(localStorage.getItem(_CUSTOM_SET_KEY) || '{}'); return v && typeof v === 'object' ? v : {}; }
    catch (_) { return {}; }
  }
  function _validCustomSet(raw) {
    if (!raw || !cfg.metrics[raw.primary]) return null;
    const seen = new Set([raw.primary]);
    const extras = (Array.isArray(raw.metrics) ? raw.metrics : []).filter(function(k) {
      if (!cfg.metrics[k] || seen.has(k)) return false; seen.add(k); return true;
    }).slice(0, MAX_COMPARE);
    return {primary:raw.primary, metrics:extras, position:['ALL','QB','RB','WR','TE'].includes(raw.position) ? raw.position : 'ALL', sort:raw.sort === 'asc' ? 'asc' : 'desc', minVol:raw.minVol == null ? '' : String(raw.minVol)};
  }

  const _initParams = new URLSearchParams(window.location.search);
  function syncURL() {
    const p = new URLSearchParams();
    if (state.metric) p.set('metric', state.metric);
    if (_activePresetId) p.set('preset', _activePresetId);
    if (state.position && state.position !== 'ALL') p.set('pos', state.position);
    if (state.season) p.set('season', state.season);
    if (state.combine && amIsMultiSeason()) p.set('combine', '1');
    if (state.minVol) p.set('minvol', String(state.minVol));
    if (state.team) p.set('team', state.team);
    const urlRange = resolveWeekRange();
    if (isMetricWeeklyCapable() && urlRange.ws != null && urlRange.we != null) {
      p.set('week_start', String(urlRange.ws)); p.set('week_end', String(urlRange.we));
    }
    const qs = p.toString();
    history.replaceState(null, '', qs ? '?' + qs : window.location.pathname);
  }

  const state = { metric: metricSel.value,
                  position: _initParams.get('pos') || 'ALL',
                  sortDir: 'desc', sortBy: metricSel.value, rows: [], search: '',
                  season: (seasonSel && (seasonSel.dataset.multi || seasonSel.value)) || '',
                  combine: (_initParams.get('combine') || '') === '1' || (_initParams.get('combine') || '').toLowerCase() === 'true',
                  minVol: _initParams.get('minvol') || '',
                  rosterOnly: false, page: 0,
                  team: _initParams.get('team') || '',
                  fetching: false, volCol: 'games', team: '',
                  weekRange: '',       // '', 'last4', 'last8', 'last12', 'custom'
                  weekStart: null,     // resolved numeric week start
                  weekEnd: null,       // resolved numeric week end
                  extraMetrics: [],     // up to 4 extra metric keys
                  extraData: {},        // key -> { byId:{player_id->value}, maxAbs }
                  playerPos: {},        // player_id -> position (for positional ranks/bounds)
                  extraPrevData: {},    // key -> { player_id -> prev-season value }
                  prevData: {},         // player_id -> previous-season value (for YoY trend)
                  showTrends: false,    // weekly usage trend column toggle
                  trendsBySeason: {},   // seasonKey -> { player_id -> trend obj }
                  ageMin: '', ageMax: '',
                  comboFilters: [],     // [{key, op, val}]
                  filterColKeys: new Set(), // keys auto-shown as compact cols when used in filters
                  cmpSeasons: {},       // player_id -> season override in the Compare modal
                  cmpRanges: {},        // player_id -> week-range key ('' | 'first' | 'second' | 'last4' | 'custom')
                  cmpWk: {},            // player_id -> { start, end } for custom ranges
                  requestToken: 0,       // invalidates responses from an older schema/selection
                  pinnedIds: _loadPins() };
  // Seven displayed metrics means six comparisons plus the primary.
  const MAX_COMPARE = Math.max(6, ...Object.values(cfg.presets || {}).map(p => (p.metrics || []).length - 1));
  const _amCmpWeekly = {};  // `${pid}_${season}` -> weekly series (Compare modal week ranges)
  const _amCmpSeason = {};   // `${pid}_${season}` -> season-level metrics (non-page seasons)
  let _amCmpToken = 0;       // guards against overlapping/stale re-renders while dragging

  // Fetch JSON with a hard timeout so a slow/overloaded endpoint can't leave the
  // Compare modal stuck on "Loading…" forever; on timeout/failure resolve null.
  function _amCmpFetch(url, ms, mode) {
    const ctl = (typeof AbortController !== 'undefined') ? new AbortController() : null;
    const t = ctl ? setTimeout(function() { ctl.abort(); }, ms || 8000) : null;
    return fetch(url, ctl ? { signal: ctl.signal } : undefined)
      .then(function(r) {
        // In 'pro' mode a 403 means the metric is PRO-gated: surface a
        // distinct signal so the column can render locked, not empty.
        if (mode === 'pro' && r.status === 403) return { proLocked: true };
        return r.ok ? r.json() : null;
      })
      .catch(function() { return null; })
      .finally(function() { if (t) clearTimeout(t); });
  }

  // Resolve a week-range key into [lo, hi] against the season's last played week.
  function _amCmpBounds(range, wk, maxWk) {
    maxWk = maxWk || 18;
    if (range === 'custom') {
      const a = Math.min(maxWk, Math.max(1, Math.round(Number(wk && wk.start) || 1)));
      const b = Math.min(maxWk, Math.max(1, Math.round(Number(wk && wk.end) || maxWk)));
      return [Math.min(a, b), Math.max(a, b)];
    }
    if (range === 'first')  return [1, Math.ceil(maxWk / 2)];
    if (range === 'second') return [Math.ceil(maxWk / 2) + 1, maxWk];
    if (range === 'last4')  return [Math.max(1, maxWk - 3), maxWk];
    return [1, maxWk];
  }

  // Aggregate weekly rows over [lo,hi] into the week-sliceable metric keys, using
  // the SAME formulas as the server's _WEEKLY_METRICS (advanced_metrics.py) so a
  // range here matches the week-range leaderboard exactly. Provider-only metrics
  // such as PFF grades and role score remain unavailable in range mode.
  function _amCmpAgg(weeks, lo, hi) {
    const sel = (weeks || []).filter(w => { const wk = Number(w.week); return wk >= lo && wk <= hi; });
    const out = { _games: sel.length };
    if (!sel.length) return out;
    let snap = 0, snapN = 0, ts = 0, tsN = 0;
    let tgt = 0, rec = 0, car = 0, tch = 0, recYds = 0, rushYds = 0;
    let rzTgt = 0, rzCar = 0;
    sel.forEach(w => {
      const sp = parseFloat(w.snap_pct); if (!isNaN(sp)) { snap += sp; snapN++; }
      const tsv = parseFloat(w.target_share); if (!isNaN(tsv)) { ts += tsv; tsN++; }
      tgt += Number(w.targets || 0); rec += Number(w.receptions || 0);
      car += Number(w.carries || 0); tch += Number(w.touches || 0);
      recYds += Number(w.rec_yards || 0); rushYds += Number(w.rush_yards || 0);
      rzTgt += Number(w.rz_targets || 0); rzCar += Number(w.rz_carries || 0);
    });
    const n = sel.length;
    const div = (a, b) => b > 0 ? a / b : null;
    let pprPts = 0;
    sel.forEach(w => { pprPts += Number(w.ppr_pts || 0); });
    out.snap_share = snapN ? (snap / snapN) / 100 : null;
    out.target_share = tsN ? (ts / tsN) : null;
    out.yards_per_target = div(recYds, tgt);
    out.yards_per_reception = div(recYds, rec);
    out.catch_rate = div(rec, tgt);
    out.yards_per_carry = div(rushYds, car);
    out.yards_per_touch = div(recYds + rushYds, tch);
    out.total_targets = tgt;       out.targets_per_game = div(tgt, n);
    out.total_receptions = rec;    out.receptions_per_game = div(rec, n);
    out.total_rec_yards = recYds;  out.rec_yards_per_game = div(recYds, n);
    out.total_carries = car;       out.carries_per_game = div(car, n);
    out.total_rush_yards = rushYds; out.rush_yards_per_game = div(rushYds, n);
    out.total_touches = tch;       out.touches_per_game = div(tch, n);
    out.ppr_pts = pprPts;          out.ppr_pts_per_game = div(pprPts, n);
    out.fpts_per_target = tgt > 0 ? (rec + recYds * 0.1 + (sel.reduce((s,w) => s + Number(w.rec_tds||0),0)) * 6) / tgt : null;
    out.fpts_per_carry = car > 0 ? (rushYds * 0.1 + (sel.reduce((s,w) => s + Number(w.rush_tds||0),0)) * 6) / car : null;
    out.rz_targets_pg = div(rzTgt, n);
    out.rz_carries_pg = div(rzCar, n);
    out.red_zone_usage = div(rzTgt + rzCar, n);

    // NGS/FTN/EPA metrics: totals summed, rates volume-weighted -- parity with
    // the server's get_adv_weekly_range_leaderboard so ranges match the board.
    const ADV_TOTALS = ['passing_epa', 'rushing_epa', 'receiving_epa',
      'yards_after_catch', 'explosive_runs_10_plus', 'ngs_rush_yards_over_expected',
      'expected_ppr', 'expected_half_ppr', 'expected_standard',
      'ppr_over_expected', 'half_ppr_over_expected', 'standard_over_expected'];
    const ADV_WEIGHTED = {
      epa_per_play: 'w_dropbacks', cpoe: 'w_dropbacks', success_rate: 'w_dropbacks',
      sack_rate: 'w_dropbacks', scramble_rate: 'w_dropbacks', nfl_passer_rating: 'w_dropbacks',
      adjusted_completion_rate: 'w_pass_att',
      qb_hit_rate: 'w_dropbacks', explosive_pass_rate: 'w_pass_att',
      play_action_rate: 'w_dropbacks', play_action_epa: 'w_dropbacks',
      out_of_pocket_rate: 'w_dropbacks', blitz_rate_faced: 'w_dropbacks',
      epa_vs_blitz: 'w_dropbacks',
      ngs_avg_time_to_throw: 'w_pass_att', ngs_aggressiveness: 'w_pass_att',
      ngs_avg_completed_air_yards: 'w_pass_att',
      ngs_avg_air_yards_differential: 'w_pass_att',
      ngs_avg_air_yards_to_sticks: 'w_pass_att', ngs_cpoe: 'w_pass_att',
      ngs_max_completed_air_distance: 'w_pass_att',
      pacr: 'w_pass_air_yards',
      ngs_rush_yards_over_expected_per_att: 'w_carries', ngs_rush_efficiency: 'w_carries',
      breakaway_percentage: 'w_carries',
      rushing_success_rate: 'w_carries', rushing_epa_per_att: 'w_carries',
      ngs_avg_time_to_los: 'w_carries',
      ngs_percent_attempts_gte_eight_defenders: 'w_carries',
      epa_vs_stacked_box: 'w_carries',
      ngs_avg_separation: 'w_targets', ngs_avg_cushion: 'w_targets',
      ngs_avg_intended_air_yards: 'w_targets', avg_depth_of_target: 'w_targets',
      ngs_catch_pct: 'w_targets', drop_rate: 'w_targets', contested_catch_rate: 'w_targets',
      ngs_created_separation: 'w_targets',
      receiving_success_rate: 'w_targets', receiving_epa_per_target: 'w_targets',
      racr: 'w_rec_air_yards',
      yards_after_catch_per_reception: 'w_receptions', ngs_avg_yac: 'w_receptions',
      ngs_avg_expected_yac: 'w_receptions', ngs_avg_yac_above_expectation: 'w_receptions',
    };
    ADV_TOTALS.forEach(k => {
      let s = 0, any = false;
      sel.forEach(w => { const v = w[k]; if (v != null && !isNaN(v)) { s += Number(v); any = true; } });
      if (any) out[k] = s;
    });
    Object.keys(ADV_WEIGHTED).forEach(k => {
      const wc = ADV_WEIGHTED[k];
      let num = 0, den = 0;
      sel.forEach(w => {
        const v = w[k], wt = w[wc];
        if (v != null && !isNaN(v) && wt != null && !isNaN(wt) && wt > 0) {
          num += Number(v) * Number(wt); den += Number(wt);
        }
      });
      if (den > 0) out[k] = num / den;
    });
    return out;
  }

  // Preset sets: clicking "Load X Set" clears current extras and loads these metrics.
  // All free/redistributable (NGS + EPA + computed); no gated PFF metrics.
  const _PRESETS = cfg.presets || {};
  let _activeSet = 'Custom';
  let _activePresetId = null;
  function _showActiveSet(name) { _activeSet = name || 'Custom'; const el = document.getElementById('amActiveSet'); if (el) el.textContent = _activeSet; }
  window.amLoadPreset = function(cat) {
    const preset = _PRESETS[cat] || _PRESETS[String(cat).toLowerCase()];
    const keys = preset && preset.metrics;
    if (!keys || !keys.length) return;
    // PRO presets are locked for non-PRO users: open the paywall instead.
    if (!cfg.hasPremium && (cfg.proPresets || []).indexOf(cat) !== -1) {
      if (typeof window.showPaywall === 'function') window.showPaywall('advanced-metrics-' + cat);
      return;
    }
    const primary = keys[0];
    const extras = keys.slice(1);
    if (extras.length > MAX_COMPARE) throw new Error('Preset exceeds comparison-column limit');
    state.metric = primary;
    if (metricSel) metricSel.value = primary;
    state.page = 0;
    state.extraMetrics = extras.slice();
    // A threshold on the old primary has different units and must not leak.
    state.comboFilters = state.comboFilters.filter(f => f.key === 'age' || f.key === 'exp');
    state.filterColKeys = new Set();
    state.prevData = {};
    const rel = new Set(relevantPositions(state.metric));
    if (state.position !== 'ALL' && !rel.has(state.position)) state.position = 'ALL';
    state.sortDir = preset.sort || 'desc';
    state.sortBy = state.metric;
    state.minVol = defaultVol(state.metric);
    // Position presets auto-filter to their position
    if (preset.position) state.position = preset.position;
    _activePresetId = Object.keys(_PRESETS).find(k => _PRESETS[k] === preset) || String(cat).toLowerCase();
    _showActiveSet(preset.label);
    try { localStorage.setItem('amLastPreset', _activePresetId); } catch (e) {}
    _updateDecisionUI(preset);
    const picker = document.getElementById('amStatPicker');
    if (picker) picker.style.display = 'none';
    updateSortBtn(); updatePosButtons(); updateMetricTip(); updateVolCtrl(); updateVolHeader();
    _amRefreshWeekControls();
    updateSortHeaders(); updateCompareBar(); syncExtraCols(); updateFilterBar();
    fetchData();
  };

  // Decision pills + tagline stay in sync with the active preset.
  function _updateDecisionUI(preset) {
    const tag = document.getElementById('amPresetTagline');
    if (tag) tag.textContent = (preset && preset.tagline) || '';
    const pills = document.getElementById('amDecisionPills');
    if (pills) pills.querySelectorAll('.am-pill').forEach(function(b) {
      const on = !!preset && b.dataset.preset === _activePresetId;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
  }
  // "All metrics": leave preset mode and return to the classic single-metric browse.
  window.amClearDecision = function() {
    _activePresetId = null;
    _showActiveSet('Custom');
    try { localStorage.removeItem('amLastPreset'); } catch (e) {}
    _updateDecisionUI(null);
    state.extraMetrics = []; state.extraData = {}; state.extraPrevData = {};
    state.page = 0;
    updateCompareBar(); syncExtraCols(); updateFilterBar(); syncURL(); fetchData();
  };
  (function _wireDecisionPills() {
    const pills = document.getElementById('amDecisionPills');
    if (!pills || pills.dataset.wired) return;
    pills.dataset.wired = '1';
    pills.addEventListener('click', function(e) {
      const b = e.target.closest('[data-preset]');
      if (!b) return;
      // Locked PRO pills open the paywall; the amLoadPreset guard covers
      // any other path (e.g. ?preset= deep links).
      if (b.dataset.locked === '1' && !cfg.hasPremium) {
        if (typeof window.showPaywall === 'function') window.showPaywall('advanced-metrics-' + b.dataset.preset);
        return;
      }
      if (b.dataset.preset) window.amLoadPreset(b.dataset.preset);
      else window.amClearDecision();
    });
  })();

  // PRO-locked extra-column cells: tap the lock to open the paywall.
  // Delegated on the table body so re-renders stay wired.
  (function _wireProCells() {
    const host = document.getElementById('amTableBody');
    if (!host || host.dataset.proWired) return;
    host.dataset.proWired = '1';
    host.addEventListener('click', function(e) {
      const cell = e.target.closest('[data-pro-metric]');
      if (!cell || cfg.hasPremium) return;
      if (typeof window.showPaywall === 'function') window.showPaywall('advanced-metrics-metric-' + cell.dataset.proMetric);
    });
  })();

  // What-changed strip: heating up / cooling off / efficiency outliers.
  function _moverSeason() {
    const sel = (typeof amSelectedSeasons === 'function') ? amSelectedSeasons() : null;
    if (sel && sel.length === 1) return sel[0];
    return (cfg.seasons && cfg.seasons[0]) || '';
  }
  function _moverChip(m, cls) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'am-mover-chip'; b.title = 'Filter table to ' + m.name;
    const nm = document.createElement('span'); nm.className = 'am-mover-name'; nm.textContent = m.name;
    const meta = document.createElement('span'); meta.className = 'am-mover-meta';
    meta.textContent = [m.team, m.position].filter(Boolean).join(' · ');
    const val = document.createElement('span'); val.className = 'am-mover-val ' + cls;
    const v = Number(m.value);
    val.textContent = (isFinite(v) ? ((v >= 0 ? '+' : '') + v.toFixed(1)) : '—');
    b.appendChild(nm); b.appendChild(meta); b.appendChild(val);
    b.addEventListener('click', function() {
      if (!searchEl) return;
      searchEl.value = m.name; state.search = m.name; state.page = 0; render();
      const t = document.getElementById('amTable');
      if (t && t.scrollIntoView) t.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    return b;
  }
  function _moverGroup(titleHtml, cls, items, valCls) {
    const g = document.createElement('div'); g.className = 'am-movers-group';
    const h = document.createElement('div'); h.className = 'am-movers-ghead ' + cls; h.innerHTML = titleHtml;
    g.appendChild(h);
    items.forEach(function(m) { g.appendChild(_moverChip(m, valCls)); });
    return g;
  }
  // Collapsible "What changed" card. Persists collapsed state in localStorage.
  function _initMoversCollapse() {
    const host = document.getElementById('amMovers');
    const head = document.getElementById('amMoversHead');
    if (!host || !head || head.dataset.collapseInit) return;
    head.dataset.collapseInit = '1';
    function _apply(collapsed) {
      host.classList.toggle('collapsed', collapsed);
      head.setAttribute('aria-expanded', String(!collapsed));
      head.title = collapsed ? 'Expand' : 'Collapse';
      try { localStorage.setItem('amMoversCollapsed', collapsed ? '1' : '0'); } catch (e) {}
    }
    head.addEventListener('click', function() { _apply(!host.classList.contains('collapsed')); });
    head.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); _apply(!host.classList.contains('collapsed')); }
    });
    try {
      if (localStorage.getItem('amMoversCollapsed') === '1') _apply(true);
    } catch (e) {}
  }
  function _loadMovers() {
    _initMoversCollapse();
    const host = document.getElementById('amMovers');
    if (!host) return;
    // The movers strip is PRO intelligence. Free users get one slim
    // dismissible nudge instead of the strip: a value moment for it, never a
    // gate, and the dismissal is remembered (brUpsell infra in paywall.js).
    if (!cfg.hasPremium) {
      _whenUpsellReady(function () {
        var _shown = window.brUpsell && window.brUpsell.nudge(host, {
          key: 'am-movers',
          feature: 'advanced-metrics-movers',
          message: 'PRO tracks who is heating up and cooling off across every metric.',
          ctaLabel: 'Unlock'
        });
        if (_shown) {
          host.style.display = '';
          host.addEventListener('click', function _amHideOnDismiss(e) {
            if (e.target && e.target.closest && e.target.closest('.br-upsell-nudge-x')) {
              host.style.display = 'none';
              host.removeEventListener('click', _amHideOnDismiss);
            }
          });
        } else {
          host.style.display = 'none';
        }
      });
      return;
    }
    const groups = document.getElementById('amMoversGroups');
    if (!groups) return;
    const params = new URLSearchParams({ platform: cfg.platform });
    const season = _moverSeason();
    if (season) params.set('season', String(season));
    if (cfg.leagueId) params.set('league_id', cfg.leagueId);
    fetch('/api/advanced-metrics/movers?' + params.toString())
      .then(function(r) { return r.ok ? r.json() : null; })
      .then(function(d) {
        groups.innerHTML = '';
        if (!d) { host.style.display = 'none'; return; }
        // Filter out zero-value items — they mean "not enough data", not "no change".
        // This keeps the section from showing a wall of +0.0s early in the season.
        function _nonzero(items) {
          return (items || []).filter(function(m) {
            const v = Number(m.value);
            return isFinite(v) && v !== 0;
          });
        }
        const heating = _nonzero(d.heating);
        const cooling = _nonzero(d.cooling);
        const outliers = _nonzero(d.outliers);
        if (!heating.length && !cooling.length && !outliers.length) {
          host.style.display = 'none'; return;
        }
        const _UP_SVG = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" style="flex-shrink:0"><path d="M2 13l4-4 3 3 5-6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M11 6h3v3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
        const _DOWN_SVG = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" style="flex-shrink:0"><path d="M2 3l4 4 3-3 5 6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M11 10h3V7" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
        const _BOLT_SVG = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" style="flex-shrink:0"><path d="M9 1L3 9h4l-1 6 6-8H8l1-6z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/></svg>';
        const _groupCount = (heating.length ? 1 : 0) + (cooling.length ? 1 : 0) + (outliers.length ? 1 : 0);
        // Single group gets a wrapped grid layout so it doesn't look sparse in the wide container.
        groups.classList.toggle('single', _groupCount === 1);
        if (heating.length) groups.appendChild(_moverGroup(_UP_SVG + '<span>Heating up</span>', 'up', heating, 'up'));
        if (cooling.length) groups.appendChild(_moverGroup(_DOWN_SVG + '<span>Cooling off</span>', 'down', cooling, 'down'));
        if (outliers.length) groups.appendChild(_moverGroup(_BOLT_SVG + '<span>Efficiency outliers</span>', 'out', outliers, ''));
        const sub = document.getElementById('amMoversSub');
        if (sub) sub.textContent = d.note || '';
        host.style.display = '';
      })
      .catch(function() { host.style.display = 'none'; });
  }

  window.amSpFilter = function(q) {
    const picker = document.getElementById('amStatPicker');
    if (!picker) return;
    const term = q.trim().toLowerCase();
    picker.querySelectorAll('.am-sp-item').forEach(function(el) {
      el.style.display = (!term || el.textContent.toLowerCase().includes(term)) ? '' : 'none';
    });
    // Hide category headers when all their items are filtered out
    picker.querySelectorAll('.am-sp-cat-head').forEach(function(hdr) {
      let sib = hdr.nextElementSibling;
      let visible = false;
      while (sib && !sib.classList.contains('am-sp-cat-head') && !sib.classList.contains('am-sp-cat-note')) {
        if (sib.classList.contains('am-sp-item') && sib.style.display !== 'none') { visible = true; break; }
        sib = sib.nextElementSibling;
      }
      hdr.style.display = visible ? '' : 'none';
    });
  };

  const _PIN_SVG = '<svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor"><path d="M9.828.722a.5.5 0 0 1 .354.146l4.95 4.95a.5.5 0 0 1 0 .707c-.48.48-1.072.588-1.503.588-.177 0-.335-.018-.46-.039l-3.134 3.134a5.927 5.927 0 0 1 .16 1.013c.046.702-.032 1.687-.72 2.375a.5.5 0 0 1-.707 0l-2.829-2.828-3.182 3.182c-.195.195-1.219.902-1.414.707-.195-.195.512-1.22.707-1.414l3.182-3.182-2.828-2.829a.5.5 0 0 1 0-.707c.688-.688 1.673-.767 2.375-.72a5.922 5.922 0 0 1 1.013.16l3.134-3.133a2.772 2.772 0 0 1-.04-.461c0-.43.108-1.022.589-1.503a.5.5 0 0 1 .353-.146z"/></svg>';

  function percentileBadge(rank, total, tied) {
    // Rank is meaningless when every displayed value ties (e.g. all-0.0
    // trends): every row would get an arbitrary rank and the first few
    // would wear a bogus "Top 5%". Skip the badge in that case.
    if (!total || !rank || tied) return '';
    const pct = rank / total;
    let label, color;
    if      (pct <= 0.05) { label = 'Top 5%';  color = '#10b981'; }
    else if (pct <= 0.10) { label = 'Top 10%'; color = '#22c55e'; }
    else if (pct <= 0.25) { label = 'Top 25%'; color = '#3b82f6'; }
    else if (pct <= 0.50) { label = 'Top 50%'; color = '#94a3b8'; }
    else return '';
    return '<span class="am-pct-badge" style="color:' + color + '">' + label + '</span>';
  }

  const _AM_MAX_PINS = 5;
  window.amTogglePin = function(id) {
    const sid = String(id);
    if (state.pinnedIds.has(sid)) {
      state.pinnedIds.delete(sid);
    } else {
      if (state.pinnedIds.size >= _AM_MAX_PINS) return; // silently ignore beyond 5
      state.pinnedIds.add(sid);
    }
    _savePins();
    render();
  };
  let ownedIds = new Set();
  const paginationEl = document.getElementById('amPagination');
  const volLabel = document.getElementById('amVolLabel');

  // ── Multi-stat compare ────────────────────────────────────────────────────
  function updateCompareBar() {
    const bar     = document.getElementById('amCompareBar');
    const chipsEl = document.getElementById('amCompareChips');
    const addBtn  = document.getElementById('amAddStatBtn');
    const pinnedBtn = document.getElementById('amComparePinnedBtn');
    if (!chipsEl) return;
    // Only show extra-metric chips (not the primary) -- the primary is already
    // visible in the dropdown, so showing it here when alone is redundant clutter.
    chipsEl.innerHTML = state.extraMetrics.map(function(key) {
      const lbl = _mLabel(key);
      return '<span class="am-chip">' + lbl
        + '<button class="am-chip-x" onclick="event.stopPropagation();amRemoveExtra(\'' + key + '\')" aria-label="Remove">\xd7</button></span>';
    }).join('');
    if (addBtn) addBtn.disabled = state.extraMetrics.length >= MAX_COMPARE;
    const clearBtn = document.getElementById('amClearExtrasBtn');
    if (clearBtn) clearBtn.style.display = state.extraMetrics.length ? '' : 'none';
    // Mobile shows a "Clear all" text link instead of the ghost button.
    const clearLink = document.getElementById('amClearExtrasLink');
    if (clearLink) clearLink.classList.toggle('am-clear-link-show', state.extraMetrics.length > 0);
    // Show the compare bar only when there's something to show.
    const hasPinned = pinnedBtn && pinnedBtn.style.display !== 'none';
    if (bar) bar.style.display = (state.extraMetrics.length > 0 || hasPinned) ? 'flex' : 'none';
  }

  function buildStatPicker() {
    const picker = document.getElementById('amStatPicker');
    if (!picker) return;
    const primaryCat = (cfg.metrics[state.metric] && cfg.metrics[state.metric].category) || 'Other';
    const active = new Set([state.metric, ...state.extraMetrics]);
    const primaryPositions = new Set((cfg.metrics[state.metric] && cfg.metrics[state.metric].positions) || []);

    function _allowed([, spec]) {
      const cat = spec.category || 'Other';
      // Non-General same-category: always include
      if (cat === primaryCat && cat !== 'General') return true;
      // General metrics: position overlap + subcategory affinity (subcategory only
      // matters when primary is a specific category, not when it is itself General)
      if (cat === 'General') {
        if (spec.positions && !spec.positions.some(p => primaryPositions.has(p))) return false;
        if (spec.subcategory && primaryCat !== 'General' && spec.subcategory !== primaryCat) return false;
        return true;
      }
      // When primary is General, also include specific-category metrics whose
      // positions overlap so users can compare across categories.
      if (primaryCat === 'General') {
        if (!spec.positions || spec.positions.some(p => primaryPositions.has(p))) return true;
      }
      return false;
    }

    // Group allowed metrics by their category
    const groups = {};
    for (const entry of Object.entries(cfg.metrics).filter(_allowed)) {
      const grp = entry[1].category || 'Other';
      (groups[grp] = groups[grp] || []).push(entry);
    }

    // Category display order: primary category first, then General, then others
    const _CAT_ORDER = ['Value', 'Passing', 'Rushing', 'Receiving', 'General'];
    const catOrder = [];
    if (primaryCat !== 'General' && groups[primaryCat]) catOrder.push(primaryCat);
    if (groups['General']) catOrder.push('General');
    // When primary is General, append the specific categories that have items
    if (primaryCat === 'General') {
      for (const c of _CAT_ORDER) {
        if (c !== 'General' && groups[c]) catOrder.push(c);
      }
    }

    const _presetKey = String(primaryCat).toLowerCase();
    const _preset = _PRESETS[_presetKey];
    let html = '<div class="am-sp-search-wrap"><input type="text" id="amSpSearch" class="am-sp-search" placeholder="Search metrics…" oninput="amSpFilter(this.value)" autocomplete="off"></div>'
      + (_preset ? '<div class="am-sp-preset-wrap"><button type="button" class="am-sp-preset-btn" onclick="amLoadPreset(\'' + _presetKey + '\')">&#9889; Load ' + _preset.label + ' Set</button></div>' : '');

    for (const grp of catOrder) {
      const grpItems = groups[grp] || [];
      if (!grpItems.length) continue;
      html += '<div class="am-sp-cat-head">' + grp + '</div>';
      for (const [key, spec] of grpItems) {
        const on = active.has(key);
        const isPrimary = key === state.metric;
        const wBadge = spec.weeklyCapable
          ? ' <span class="am-sp-weekly-badge" title="Supports week-range filtering">W</span>'
          : '';
        html += '<div class="am-sp-item' + (on ? ' am-sp-active' : '') + '" onclick="amPickerClick(\'' + key + '\')">'
          + '<span class="am-sp-check">' + (on ? '&#10003;' : '') + '</span>'
          + spec.label + wBadge
          + (isPrimary ? ' <span style="font-size:10px;opacity:.6">(primary)</span>' : '')
          + '</div>';
      }
    }

    if (primaryCat !== 'General') {
      const otherCats = ['Passing', 'Rushing', 'Receiving'].filter(c => c !== primaryCat);
      if (otherCats.length) {
        html += '<div class="am-sp-cat-note">To compare ' + otherCats.join(' or ') + ' stats, switch the Primary Metric</div>';
      }
    }
    picker.innerHTML = html;
  }

  window.amPickerClick = function(key) {
    _showActiveSet('Custom');
    _activePresetId = null;
    if (key === state.metric) return;
    if (state.extraMetrics.includes(key)) {
      amRemoveExtra(key);
    } else {
      if (state.extraMetrics.length < MAX_COMPARE) {
        // If this key was a filter col, promote it to extra metric (richer view).
        if (state.filterColKeys && state.filterColKeys.has(key)) {
          state.filterColKeys.delete(key);
          syncFilterCols();
        }
        state.extraMetrics.push(key);
        updateCompareBar();
        syncExtraCols();
        updateFilterBar();
        render();        // show skeleton column immediately
        fetchExtraData(key);
      }
    }
    buildStatPicker();
  };

  window.amRemoveExtra = function(key) {
    _showActiveSet('Custom');
    _activePresetId = null;
    state.extraMetrics = state.extraMetrics.filter(k => k !== key);
    delete state.extraData[key];
    delete state.extraPrevData[key];
    if (state.filterColKeys) state.filterColKeys.delete(key);
    state.comboFilters = state.comboFilters.filter(f => f.key !== key);
    if (state.sortBy === key) {
      state.sortBy = state.metric;
      state.sortDir = (cfg.metrics[state.metric] && cfg.metrics[state.metric].lowerBetter) ? 'asc' : 'desc';
    }
    updateCompareBar();
    buildStatPicker();
    syncExtraCols();
    updateFilterBar();
    render();
  };

  window.amClearExtras = function() {
    _showActiveSet('Custom');
    _activePresetId = null;
    state.extraMetrics.forEach(key => {
      delete state.extraData[key];
      delete state.extraPrevData[key];
      if (state.filterColKeys) state.filterColKeys.delete(key);
    });
    state.extraMetrics = [];
    state.comboFilters = state.comboFilters.filter(f => f.key === 'primary' || f.key === 'age');
    if (state.sortBy !== state.metric) {
      state.sortBy = state.metric;
      state.sortDir = (cfg.metrics[state.metric] && cfg.metrics[state.metric].lowerBetter) ? 'asc' : 'desc';
    }
    updateCompareBar();
    buildStatPicker();
    syncExtraCols();
    updateFilterBar();
    render();
  };

  function _orderByCategory(keys) {
    const CAT_ORDER = ['General', 'Passing', 'Rushing', 'Receiving'];
    const groups = {};
    for (const key of keys) {
      const spec = cfg.metrics[key];
      const cat = (spec && spec.category) || 'Other';
      (groups[cat] = groups[cat] || []).push(key);
    }
    const ordered = [];
    for (const cat of CAT_ORDER) { if (groups[cat]) ordered.push(...groups[cat]); }
    for (const [cat, ks] of Object.entries(groups)) {
      if (!CAT_ORDER.includes(cat)) ordered.push(...ks);
    }
    return ordered;
  }

  function fetchExtraData(key, _attempt) {
    _attempt = _attempt || 0;
    const requestToken = state.requestToken;
    function _buildExtraParams(s) {
      const p = new URLSearchParams({ metric: key, platform: cfg.platform });
      if (cfg.leagueId) p.set('league_id', cfg.leagueId);
      if (s) p.set('season', String(s));
      if (state.combine && amIsMultiSeason()) p.set('combine', '1');
      const vol = defaultVol(key);
      if (vol) p.set('min_vol', vol);
      // Apply the same week range as the primary metric so columns stay in sync.
      const { ws, we } = resolveWeekRange();
      if (ws) p.set('week_start', String(ws));
      if (we) p.set('week_end',   String(we));
      return p;
    }
    const curSeason = state.season || (cfg.seasons && cfg.seasons[0] ? String(cfg.seasons[0]) : '');
    const seasons = cfg.seasons || [];
    const primaryYear = amSelectedSeasons()[0] || Number(curSeason) || null;
    const curIdx = primaryYear != null ? seasons.indexOf(Number(primaryYear)) : -1;
    const prevSeason = (!amIsMultiSeason() && curIdx >= 0 && curIdx + 1 < seasons.length) ? seasons[curIdx + 1] : null;
    const curUrl = '/api/advanced-metrics/leaderboard?' + _buildExtraParams(curSeason);
    const prevUrl = prevSeason ? '/api/advanced-metrics/leaderboard?' + _buildExtraParams(String(prevSeason)) : null;
    // Primary (current-season) column, with a hard timeout so a hung request
    // can't leave the column's skeleton spinning forever. A 403 means the
    // metric is PRO-gated for this viewer: lock the column instead of
    // retrying or clearing it, so the paywall has a real entry point.
    return _amCmpFetch(curUrl, 12000, 'pro').then(function(curr) {
      if (requestToken !== state.requestToken || !state.extraMetrics.includes(key) && !(state.filterColKeys && state.filterColKeys.has(key))) return;
      if (curr && curr.proLocked) {
        state.extraData[key] = { byId: {}, maxAbs: 1, proLocked: true };
        render();
        return;
      }
      if (!curr) {
        // Transient failure/timeout (often server contention from a preset's
        // burst of requests): retry once, then give up and clear the skeleton.
        if (_attempt < 1) {
          return new Promise(function(res) { setTimeout(res, 1200); })
            .then(function() { return fetchExtraData(key, _attempt + 1); });
        }
        state.extraData[key] = { byId: {}, maxAbs: 1, failed: true };
        render();
        return;
      }
      const rows = curr.players || [];
      const maxAbs = rows.reduce((m, r) => Math.max(m, Math.abs(Number(r.value) || 0)), 0) || 1;
      // Capture positions so the Compare modal can rank/scale within position.
      rows.forEach(r => {
        if (r.player_id != null && r.position) {
          state.playerPos[String(r.player_id)] = String(r.position).toUpperCase();
        }
      });
      state.extraData[key] = {
        byId: Object.fromEntries(rows.map(r => [amRowKey(r), Number(r.value)])),
        context: Object.fromEntries(rows.map(r => [amRowKey(r), {tgt:r.tgt, car:r.car, rec:r.rec, att:r.att, db:r.db, games:r.games}])),
        maxAbs: maxAbs
      };
      render();
      // Previous-season values (YoY trend arrows) are non-essential -- fetch them
      // off the critical path so the column shows immediately and the initial
      // request burst is halved.
      if (prevUrl) {
        _amCmpFetch(prevUrl, 12000).then(function(prev) {
          if (prev) {
            state.extraPrevData[key] = Object.fromEntries((prev.players || []).map(r => [amRowKey(r), Number(r.value)]));
            render();
          }
        });
      }
    });
  }
  // Load several extra-metric columns with bounded concurrency so a preset
  // doesn't fire a dozen leaderboard requests at once and starve the server
  // (which made columns load slowly or never finish).
  function _loadExtras(keys) {
    const list = (keys || []).filter(Boolean);
    const MAX = 4;
    let i = 0, active = 0;
    function pump() {
      while (active < MAX && i < list.length) {
        active++;
        fetchExtraData(list[i++]).catch(function() {}).finally(function() { active--; pump(); });
      }
    }
    pump();
  }

  // ── Weekly usage trends ───────────────────────────────────────────────────
  function trendSeasonKey() { return String(state.season || 'cur'); }
  function fetchTrends() {
    const q = state.season ? ('?season=' + encodeURIComponent(state.season)) : '';
    fetch('/api/weekly-trends' + q)
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        state.trendsBySeason[trendSeasonKey()] = (d && d.players) || {};
        syncTrendHeader();  // header label reflects how many weeks exist
        render();
      })
      .catch(() => { state.trendsBySeason[trendSeasonKey()] = {}; syncTrendHeader(); render(); });
  }
  function sparkline(series, color) {
    if (!series || series.length < 2) return '';
    const w = 60, h = 18;
    const max = Math.max.apply(null, series.concat([1]));
    const step = w / (series.length - 1);
    const pts = series.map(function(v, i) {
      return (i * step).toFixed(1) + ',' + (h - 1 - (v / max) * (h - 3)).toFixed(1);
    }).join(' ');
    return '<svg class="am-spark" width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h + '">'
      + '<polyline fill="none" stroke="' + color + '" stroke-width="1.5" stroke-linejoin="round" points="' + pts + '"/></svg>';
  }
  function trendCellHtml(playerId, color) {
    const tm = state.trendsBySeason[trendSeasonKey()];
    if (!tm) {
      return '<td class="am-trendcell" data-column-id="trend"><div class="am-skel-bar" style="width:60px;height:10px;border-radius:4px;"></div></td>';
    }
    const t = tm[String(playerId)];
    if (!t || !t.series || t.series.length < 2) {
      return '<td class="am-trendcell" data-column-id="trend"><span style="opacity:.35">–</span></td>';
    }
    const statLbl = { snap_pct: 'snap%', touches: 'touches', targets: 'targets' }[t.stat] || t.stat;
    const d = t.delta;
    // A null delta means the recent window is degenerate (<=3 weeks: the
    // "last-3" window IS the season sample, so any number would be a lie).
    // The sparkline still shows the real weekly series.
    let deltaHtml = '<span class="am-trend-delta am-trend-delta-flat" title="Not enough weeks yet for a trend signal">&ndash;</span>';
    if (d != null && d >= 0.5) deltaHtml = '<span class="am-trend-delta am-trend-delta-up">&#9650; +' + d.toFixed(1) + '</span>';
    else if (d != null && d <= -0.5) deltaHtml = '<span class="am-trend-delta am-trend-delta-down">&#9660; ' + d.toFixed(1) + '</span>';
    const recentN = Math.min(3, t.weeks_played || 3);
    const tip = (d == null || t.recent_avg == null)
      ? 'Recent usage: only ' + (t.weeks_played || 0) + ' week(s) so far — trend signal starts with more data'
      : 'Last-' + recentN + '-week avg ' + statLbl + ' (' + t.recent_avg
        + ') vs season avg (' + t.season_avg + ')';
    return '<td class="am-trendcell" data-column-id="trend" title="' + tip + '">'
      + '<div class="am-trend-inner">' + sparkline(t.series, color) + deltaHtml + '</div></td>';
  }
  function trendWindowWeeks() {
    // Longest series among loaded players - the label adapts to how many
    // weeks of data actually exist (early season shows L1W/L2W, etc).
    const tm = state.trendsBySeason[trendSeasonKey()];
    if (!tm) return 0;
    let n = 0;
    Object.values(tm).forEach(function(t) {
      if (t.series && t.series.length > n) n = t.series.length;
    });
    return n;
  }
  function syncTrendHeader() {
    syncTableHeader();
    const th = document.querySelector('#amTable thead th[data-column-id="trend"]');
    if (!th) return;
    th.id = 'amTrendHeader'; th.className = 'am-trendcell';
    const n = trendWindowWeeks();
    th.textContent = n > 0 ? ('Usage L' + n + 'W') : 'Usage Trend';
    th.title = 'Recent usage: ' + (n > 0 ? 'last-' + n + '-week' : 'recent') + ' trend of the key volume stat for the position (QB snap %, RB touches, WR/TE targets)';
  }

  // ── Pinned-player comparison ──────────────────────────────────────────────
  function pinnedRows() {
    return state.rows.filter(r => state.pinnedIds.has(String(r.player_id)));
  }
  function updateComparePinnedBtn() {
    const btn = document.getElementById('amComparePinnedBtn');
    if (btn) btn.style.display = pinnedRows().length >= 2 ? '' : 'none';
  }
  // Missing metric values must sort after every real number, in both
  // directions. Extra columns used to coerce gaps to 0 (`?? 0`), which put
  // uncharted players at the top of signed stats (EPA, created sep, CPOE).
  function _amMissing(v) {
    if (v == null || v === '') return true;
    return Number.isNaN(Number(v));
  }
  function _amCmpVal(av, bv, desc) {
    const aMiss = _amMissing(av);
    const bMiss = _amMissing(bv);
    if (aMiss !== bMiss) return aMiss ? 1 : -1;
    if (aMiss) return 0;
    const diff = Number(av) - Number(bv);
    return desc ? -diff : diff;
  }
  // Positional rank + value bounds for a metric, computed from the loaded
  // leaderboard field (same data & filters the page is showing). Ranks are
  // WITHIN each position -- matching the player modal / a position-filtered
  // leaderboard -- so a multi-position efficiency metric like yards/touch doesn't
  // bury RBs beneath WRs. bounds[pos] = [min, max] drive position-aware bars.
  function _amPosStats(key) {
    const lower = !!(cfg.metrics[key] && cfg.metrics[key].lowerBetter);
    let entries;  // [id, value, position]
    if (key === state.metric) {
      entries = state.rows.map(r => [String(r.player_id), r.value,
        String(r.position || state.playerPos[String(r.player_id)] || '').toUpperCase()]);
    } else {
      const ed = state.extraData[key];
      if (!ed) return null;  // still loading
      entries = Object.entries(ed.byId).map(([id, v]) =>
        [id, v, String(state.playerPos[id] || '').toUpperCase()]);
    }
    entries = entries.filter(e => !_amMissing(e[1])).map(e => [e[0], Number(e[1]), e[2]]);
    const byPos = {};
    entries.forEach(e => { (byPos[e[2]] = byPos[e[2]] || []).push(e); });
    const ranks = {}, bounds = {}, counts = {}, avgs = {};
    Object.keys(byPos).forEach(pos => {
      const arr = byPos[pos];
      arr.sort((a, b) => lower ? a[1] - b[1] : b[1] - a[1]);
      arr.forEach((e, i) => { ranks[e[0]] = i + 1; });
      const vals = arr.map(e => e[1]);
      bounds[pos] = [Math.min(...vals), Math.max(...vals)];
      counts[pos] = arr.length;
      avgs[pos] = vals.reduce((a, b) => a + b, 0) / vals.length;
    });
    return { ranks: ranks, bounds: bounds, counts: counts, avgs: avgs };
  }
  // Bar fill (8–100%) by RANK within position -- so a mid-ranked player in a
  // bunched-top metric (role score, snap share, yards/touch…) doesn't show a
  // near-full bar. #1 → 100, last → 8.
  function _amRankFill(rank, count) {
    if (!rank || !count || count < 2) return null;
    const t = (count - rank) / (count - 1);
    return 8 + Math.max(0, Math.min(1, t)) * 92;
  }
  // Bar fill (8–100%) by where `val` sits in its position's [min,max] range --
  // the same magnitude-preserving, position-aware scaling the player modal uses.
  function _amBoundsFill(key, val, bnds) {
    if (val == null || !bnds) return null;
    const lo = bnds[0], hi = bnds[1];
    if (!(hi > lo)) return null;  // degenerate (single player / all equal)
    let t = (val - lo) / (hi - lo);
    if (cfg.metrics[key] && cfg.metrics[key].lowerBetter) t = 1 - t;
    t = Math.max(0, Math.min(1, t));
    return 8 + t * 92;  // 8% floor so the worst still shows a sliver
  }
  // Unified bar fill (%) for the Compare modal -- same four-shape model as the
  // player modal:
  //  • SCORE  (grades, ratings, VORP/WAR) → value ÷ ceiling.
  //  • MINMAX (EPA totals) → position [min,max], so big leads & negatives show.
  //  • RANK   (efficiency rates) → percentile within position (page columns).
  //  • LEADER (volume) → value ÷ position leader.
  const _AM_SCORE_CEIL = { role_score: 100, grades_offense: 100, pff_passing_grade: 100,
    pff_rushing_grade: 100, nfl_passer_rating: 158.3, vorp: 150, war: 6 };
  const _AM_MINMAX = ['passing_epa', 'rushing_epa', 'receiving_epa'];
  const _AM_RATE = ['avoided_tackles_pg', 'explosive_runs_pg'];
  function _amBarFill(key, v, pos, mode, rk, stats, barMax) {
    const m = cfg.metrics[key] || {};
    const valFb = Math.min(100, Math.max(3, Math.round(Math.abs(v) / barMax * 100)));
    const ceil = _AM_SCORE_CEIL[key];
    if (ceil) return Math.max(4, Math.min(100, (v / ceil) * 100));
    const b = (stats && stats.bounds[pos]) || null;
    const rankF = (mode === 'page' && stats && rk) ? _amRankFill(rk, stats.counts[pos]) : null;
    const mmF = b ? _amBoundsFill(key, v, b) : null;
    if (_AM_MINMAX.indexOf(key) >= 0) return mmF != null ? mmF : (rankF != null ? rankF : valFb);
    const isEff = m.efficiency || m.pct || m.pctFrac || _AM_RATE.indexOf(key) >= 0;
    if (isEff) return rankF != null ? rankF : (mmF != null ? mmF : valFb);
    if (b && b[1] > 0) return Math.max(4, Math.min(100, (v / b[1]) * 100));  // volume → leader
    return rankF != null ? rankF : valFb;
  }
  // Change a single player's season in the Compare modal and re-render.
  window.amSetCmpSeason = function(pid, season) {
    state.cmpSeasons[String(pid)] = season;
    state.cmpRanges[String(pid)] = '';  // week ranges are season-specific
    window.amShowCompare();
  };
  // Per-player draggable week-range bar + games label for the modal header.
  // Reuses the page-level wk-bar component (_wkBarBuild/_wkBarInit) so the
  // compare splits use the same control as the main leaderboard.
  function cmpRangeControls(p, meta) {
    if (typeof _wkBarBuild !== 'function') return '';
    const pid = String(p.player_id);
    const r = state.cmpRanges[pid] || '';
    const s = String(state.cmpSeasons[pid] || state.season || (cfg.seasons && cfg.seasons[0]) || '');
    const latest = String((cfg.seasons && cfg.seasons[0]) || '');
    // Before weekly data loads we don't know the player's true last week, so
    // default to the current NFL week for the latest season, else a full 18.
    const defMax = (s === latest) ? (cfg.currentWeek || 18) : 18;
    const maxWk = (meta && meta.maxWk) ? meta.maxWk : defMax;
    let ws = 1, we = maxWk;
    if (r === 'custom') {
      const wk = state.cmpWk[pid] || { start: 1, end: maxWk };
      ws = Math.min(maxWk, Math.max(1, wk.start || 1));
      we = Math.min(maxWk, Math.max(ws, wk.end || maxWk));
    }
    let note;
    if (r && meta) {
      note = meta.games > 0
        ? 'Wks ' + meta.lo + '&ndash;' + meta.hi + ' &middot; ' + meta.games + 'g'
        : '<span class="am-cmp-wknote-warn">No games in range</span>';
    } else {
      note = '<span class="am-cmp-wknote-muted">Full season &middot; drag to filter</span>';
    }
    return '<div class="am-cmp-wkbar-wrap">' + _wkBarBuild('amCmpWk_' + pid, 1, maxWk, ws, we)
      + '<div class="am-cmp-wknote">' + note + '</div></div>';
  }

  // Wire drag interaction on every per-player week bar after a modal re-render.
  function _amCmpInitWkBars(players) {
    if (typeof _wkBarInit !== 'function') return;
    players.forEach(function(p) {
      const pid = String(p.player_id);
      const bar = document.getElementById('amCmpWk_' + pid);
      if (!bar) return;
      const mx = Number(bar.dataset.max) || (cfg.currentWeek || 18);
      _wkBarInit('amCmpWk_' + pid, function(ws, we) {
        if (ws <= 1 && we >= mx) {
          state.cmpRanges[pid] = '';   // full season
          state.cmpWk[pid] = null;
        } else {
          state.cmpRanges[pid] = 'custom';
          state.cmpWk[pid] = { start: ws, end: we };
        }
        window.amShowCompare();
      });
    });
  }

  window.amShowCompare = async function() {
    const modal = document.getElementById('amCompareModal');
    const body = document.getElementById('amCompareBody');
    if (!modal || !body) return;
    const players = pinnedRows();
    if (players.length < 2) return;
    // Size card to content: each player column needs ~180px, metric label ~140px.
    const card = modal.querySelector('.am-cmp-card');
    if (card) {
      const ideal = 140 + players.length * 180;
      card.style.maxWidth = Math.min(Math.max(560, ideal), window.innerWidth * 0.95) + 'px';
    }
    let metricsList = [state.metric, ...state.extraMetrics];
    const pageSeason = String(state.season || (cfg.seasons && cfg.seasons[0]) || '');
    const seasonsList = (cfg.seasons || []).slice();
    const seasonFor = (p) => String(state.cmpSeasons[String(p.player_id)] || pageSeason);
    const rangeFor = (p) => state.cmpRanges[String(p.player_id)] || '';

    modal.style.display = 'flex';
    const token = ++_amCmpToken;

    // Only the players whose split needs uncached data require a network call.
    // Page-season / full-season columns reuse already-loaded values, so adjusting
    // one player's week range never re-fetches the others -- it just re-aggregates
    // that one player's (cached) weekly series locally.
    const needsFetch = players.some((p) => {
      const pid = String(p.player_id);
      const s = seasonFor(p);
      if (rangeFor(p)) return !_amCmpWeekly[pid + '_' + s];
      if (s === pageSeason) return false;
      return !_amCmpSeason[pid + '_' + s];
    });
    // Show the spinner only when we actually have to wait on the network and the
    // table isn't already on screen -- so tweaking a cached range never flashes
    // (or sticks on) "Loading…".
    if (needsFetch && body.dataset.cmpReady !== '1') {
      body.innerHTML = '<div style="padding:18px;color:var(--text-muted);font-size:13px;">Loading…</div>';
    }

    // Resolve each pinned player's data source:
    //   • 'range' – client-side weekly aggregate for a selected week range
    //   • 'page'  – reuse the loaded leaderboard's in-memory values (keeps rank)
    //   • 'fetch' – pull that season's season-level metrics on demand
    const rangeMeta = {};  // pid -> { lo, hi, games, maxWk }
    const perPlayer = await Promise.all(players.map(async (p) => {
      const pid = String(p.player_id);
      const s = seasonFor(p);
      const range = rangeFor(p);
      if (range) {
        const ck = pid + '_' + s;
        let weeks = _amCmpWeekly[ck];
        if (!weeks) {
          const d = await _amCmpFetch('/api/player-weekly-metrics/' + encodeURIComponent(pid) + '?season=' + encodeURIComponent(s));
          weeks = (d && d.weeks) || [];
          // Only cache a real result; don't poison the cache on a transient
          // failure/timeout so a later interaction can retry.
          if (weeks.length) _amCmpWeekly[ck] = weeks;
        }
        const maxWk = weeks.length ? Math.max(...weeks.map(w => Number(w.week) || 0)) : 18;
        const [lo, hi] = _amCmpBounds(range, state.cmpWk[pid], maxWk);
        const agg = _amCmpAgg(weeks, lo, hi);
        rangeMeta[pid] = { lo: lo, hi: hi, games: agg._games || 0, maxWk: maxWk };
        return { mode: 'range', agg: agg };
      }
      if (s === pageSeason) return { mode: 'page' };
      const ck = pid + '_' + s;
      let metrics = _amCmpSeason[ck];
      if (!metrics) {
        const d = await _amCmpFetch('/api/player-advanced-metrics/' + encodeURIComponent(pid) + '?season=' + encodeURIComponent(s));
        metrics = (d && d.metrics) || {};
        if (Object.keys(metrics).length) _amCmpSeason[ck] = metrics;
      }
      return { mode: 'fetch', metrics: metrics };
    }));

    // A newer drag/season change started while we were awaiting -- drop this
    // stale render so the latest interaction wins (prevents flicker / clobber).
    if (token !== _amCmpToken) return;

    const valueFor = (p, i, key) => {
      const pp = perPlayer[i];
      if (pp.mode === 'range') {
        const v = pp.agg[key];
        return (v !== undefined && v !== null) ? Number(v) : null;
      }
      if (pp.mode === 'fetch') {
        const v = pp.metrics[key];
        return (v !== undefined && v !== null) ? Number(v) : null;
      }
      if (key === state.metric) return Number(p.value);
      const ed = state.extraData[key];
      const v = ed ? ed.byId[String(p.player_id)] : undefined;
      return (v !== undefined && v !== null) ? Number(v) : null;
    };

    const anyRange = players.some(p => rangeFor(p));

    // Always show exactly the metrics the user selected on the page (primary +
    // any added compare metrics) -- even when a week range is active. Week ranges
    // simply re-aggregate those same metrics over the chosen weeks.
    metricsList = [state.metric, ...state.extraMetrics];

    // Baseline column: when every pinned player shares a position, show that
    // position's average (from the loaded page-season leaderboard) as a
    // reference point next to the player columns.
    const _cmpBasePos = (players.length && players.every(p =>
        String(p.position || '').toUpperCase() === String(players[0].position || '').toUpperCase()
      )) ? String(players[0].position || '').toUpperCase() : null;

    let html = '<table class="am-cmp-table"><thead><tr><th>Metric</th>';
    players.forEach((p) => {
      const pid = String(p.player_id);
      const s = seasonFor(p);
      const opts = seasonsList.map(yr =>
        '<option value="' + yr + '"' + (String(yr) === s ? ' selected' : '') + '>' + yr + '</option>'
      ).join('');
      html += '<th class="am-cmp-player-head">'
        + '<div class="am-cmp-head-name">' + (p.name || '')
        + (p.position ? '<span class="am-cmp-head-pos" style="background:' + posColor(p.position) + '">' + p.position + '</span>' : '')
        + '</div>'
        + (seasonsList.length
            ? '<select class="am-cmp-season-sel" onchange="amSetCmpSeason(\'' + pid + '\', this.value)">' + opts + '</select>' + cmpRangeControls(p, rangeMeta[pid])
            : '<span class="am-cmp-player-meta">' + (p.team || '') + '</span>')
        + '</th>';
    });
    if (_cmpBasePos) {
      html += '<th class="am-cmp-player-head am-cmp-baseline-head">'
        + '<div class="am-cmp-head-name">' + _cmpBasePos + ' avg</div>'
        + '<span class="am-cmp-player-meta">page season</span>'
        + '</th>';
    }
    html += '</tr></thead><tbody>';

    metricsList.forEach(key => {
      const lbl = _mLabel(key);
      const lower = !!(cfg.metrics[key] && cfg.metrics[key].lowerBetter);
      const stats = _amPosStats(key);  // positional ranks + per-position bounds
      const vals = players.map((p, i) => valueFor(p, i, key));
      const present = vals.filter(v => v != null);
      const best = present.length
        ? (lower ? Math.min(...present) : Math.max(...present))
        : null;
      // Fallback scale (only used when a player's position has no bounds, e.g.
      // a different-season column): relative to the largest pinned value.
      const barMax = present.length ? Math.max(...present.map(v => Math.abs(v))) || 1 : 1;

      html += '<tr><td class="am-cmp-metric">' + lbl + '</td>';
      players.forEach((p, i) => {
        const v = vals[i];
        if (v == null) { html += '<td><span style="opacity:.4">–</span></td>'; return; }
        const pos = String(p.position || '').toUpperCase();
        const isBest = best != null && v === best && present.length > 1;
        // Rank only for page-season columns (positional, matching the leaderboard).
        const rk = (perPlayer[i].mode === 'page' && stats) ? stats.ranks[String(p.player_id)] : null;
        // Bar: role score uses 0–100; efficiency uses rank; volume scales to the
        // position leader (so half the leader's volume ≈ half a bar).
        const w = Math.round(_amBarFill(key, v, pos, perPlayer[i].mode, rk, stats, barMax));
        html += '<td><span class="am-cmp-val' + (isBest ? ' am-cmp-best' : '') + '">' + fmtVal(v, key) + '</span>'
          + (rk ? '<span class="am-cmp-rank">#' + rk + '</span>' : '')
          + '<div class="am-cmp-bar"><div style="width:' + w + '%;background:' + posColor(p.position) + '"></div></div></td>';
      });
      if (_cmpBasePos) {
        const av = (stats && stats.avgs) ? stats.avgs[_cmpBasePos] : null;
        html += '<td class="am-cmp-baseline">'
          + (av != null ? '<span class="am-cmp-val">' + fmtVal(av, key) + '</span>' : '<span style="opacity:.4">–</span>')
          + '</td>';
      }
      html += '</tr>';
    });
    html += '</tbody></table>';
    html += '<div style="font-size:11px;color:var(--text-muted);margin-top:10px;">'
      + (anyRange
          ? 'Week ranges are aggregated from weekly usage data (matching the week-range leaderboard); season-level metrics like PFF grades and role score show “–” for a range, and ranks apply only to full page-season columns.'
          : 'Pick a season or week range per player to compare across splits. Showing the primary metric plus any added metrics. Ranks and bars are within position, using the current page filters, matching the leaderboard.')
      + '</div>';
    body.innerHTML = html;
    body.dataset.cmpReady = '1';
    if (window.initCustomSelects) window.initCustomSelects(body);
    _amCmpInitWkBars(players);
    modal.style.display = 'flex';
  };

  function toggleStatPicker() {
    const picker = document.getElementById('amStatPicker');
    const btn = document.getElementById('amAddStatBtn');
    if (!picker) return;
    const open = picker.style.display !== 'none' && picker.style.display !== '';
    if (open) {
      picker.style.display = 'none';
      const srch = document.getElementById('amSpSearch');
      if (srch) { srch.value = ''; amSpFilter(''); }
      return;
    }
    buildStatPicker();
    picker.style.position = 'fixed';
    picker.style.display = '';
    requestAnimationFrame(function() {
      if (!btn) return;
      const rect = btn.getBoundingClientRect();
      const ph = picker.offsetHeight;
      picker.style.right = (window.innerWidth - rect.right) + 'px';
      picker.style.left = 'auto';
      if (rect.bottom + 6 + ph > window.innerHeight - 20) {
        picker.style.bottom = (window.innerHeight - rect.top + 6) + 'px';
        picker.style.top = 'auto';
      } else {
        picker.style.top = (rect.bottom + 6) + 'px';
        picker.style.bottom = 'auto';
      }
    });
  }

  // Close picker when clicking outside.
  document.addEventListener('click', function(e) {
    const wrap = document.getElementById('amAddStatWrap');
    if (wrap && !wrap.contains(e.target)) {
      const p = document.getElementById('amStatPicker');
      if (p) p.style.display = 'none';
    }
  });

  document.getElementById('amAddStatBtn').addEventListener('click', function(e) {
    e.stopPropagation();
    toggleStatPicker();
  });

  const _cmpBtn = document.getElementById('amComparePinnedBtn');
  if (_cmpBtn) _cmpBtn.addEventListener('click', function() { amShowCompare(); });

  function relevantPositions(m) {
    return (cfg.metrics[m] && cfg.metrics[m].positions) || ['QB','RB','WR','TE'];
  }
  function isEfficiency(m) { return !!(cfg.metrics[m] && cfg.metrics[m].efficiency); }
  function posColor(p) {
    return ({ QB:'#3b82f6', RB:'#22c55e', WR:'#f59e0b', TE:'#8b5cf6' })[p] || '#888';
  }
  function fmtVal(v, m) {
    if (v == null) return '-';
    const n = Number(v);
    const spec = m ? cfg.metrics[m] : null;
    if (spec && spec.pct) {
      const pct = spec.pctFrac ? n * 100 : n;
      return (Math.abs(pct) >= 10 ? pct.toFixed(1) : pct.toFixed(2)) + '%';
    }
    if ((spec && spec.integer) || Number.isInteger(n)) return Math.round(n).toFixed(0);
    return Math.abs(n) >= 100 ? n.toFixed(0) : n.toFixed(2);
  }
  function updateSortBtn() {
    sortBtn.innerHTML = state.sortDir === 'desc' ? 'High &rarr; Low' : 'Low &rarr; High';
  }
  function updateSortHeaders() {
    const ths = [
      { el: document.querySelector('#amTable thead th.am-player'), col: 'name' },
      { el: document.querySelector('#amTable thead th.am-games'), col: 'games' },
      { el: document.getElementById('amMetricHeader'), col: state.metric },
    ];
    state.extraMetrics.forEach(function(key) {
      const el = document.getElementById('amExtraHeader_' + key);
      if (el) ths.push({ el: el, col: key });
    });
    if (state.filterColKeys) {
      [...state.filterColKeys].forEach(function(key) {
        const el = document.getElementById('amFilterColHdr_' + key);
        if (el) ths.push({ el: el, col: key });
      });
    }
    ths.forEach(function({ el, col }) {
      if (!el) return;
      el.classList.add('am-sortable');
      el.classList.toggle('am-sort-asc', state.sortBy === col && state.sortDir === 'asc');
      el.classList.toggle('am-sort-desc', state.sortBy === col && state.sortDir === 'desc');
    });
  }
  function sortByCol(col) {
    if (state.sortBy === col) {
      state.sortDir = state.sortDir === 'desc' ? 'asc' : 'desc';
    } else {
      state.sortBy = col;
      if (col === 'name' || col === 'games') {
        state.sortDir = 'asc';
      } else {
        state.sortDir = (cfg.metrics[col] && cfg.metrics[col].lowerBetter) ? 'asc' : 'desc';
      }
    }
    state.page = 0;
    updateSortHeaders();
    render();
  }
  // Filter columns: compact columns inserted right before the primary metric column
  // when a metric key is used as a combo filter condition. Share extraData with
  // extra metrics but render differently (number only, no bar, after Games column).
  function syncFilterCols() {
    syncTableHeader();
  }
  // Shared semantic sample-column builder for table rows, presets, manual
  // comparisons and CSV. Counts are grouped beside Games and de-duplicated by
  // identity, not by their display text.
  function contextColsFor() {
    const defs = {
      targets:{key:'tgt',label:'Tgt',title:'Targets in the selected season/week range'},
      carries:{key:'car',label:'Car',title:'Carries in the selected season/week range'},
      attempts:{key:'att',label:'Att',title:'Pass attempts in the selected season/week range'},
      dropbacks:{key:'db',label:'DB',title:'Qualifying dropbacks covered by the metric'},
      receptions:{key:'rec',label:'Rec',title:'Receptions in the selected season/week range'}
    };
    const selected = [state.metric, ...state.extraMetrics];
    const explicit = new Set();
    if (selected.includes('total_targets')) explicit.add('targets');
    if (selected.includes('total_carries')) explicit.add('carries');
    if (selected.includes('total_receptions')) explicit.add('receptions');
    if (selected.includes('total_pass_att')) explicit.add('attempts');
    const wanted = [];
    // Presets declare their complete context (for example Receiving needs both
    // targets and receptions even though Target Share itself only uses targets).
    const preset = _activePresetId && _PRESETS[_activePresetId];
    (preset && preset.samples || []).forEach(function(identity) {
      if (identity !== 'games' && !explicit.has(identity) && !wanted.includes(identity)) wanted.push(identity);
    });
    selected.forEach(function(k) {
      const identity = cfg.metrics[k] && cfg.metrics[k].sample;
      if (identity && identity !== 'games' && !explicit.has(identity) && !wanted.includes(identity)) wanted.push(identity);
    });
    const cols = wanted.map(k => Object.assign({id:'sample:' + k, semantic:k, kind:'sample'}, defs[k])).filter(Boolean);
    const rows = state.rows || [];
    return cols.filter(function(c) {
      if (rows.some(function(r) { return r[c.key] != null; })) return true;
      return state.extraMetrics.some(function(k) {
        const ed = state.extraData[k];
        return ed && ed.context && Object.values(ed.context).some(v => v && v[c.key] != null);
      });
    });
  }
  function contextValue(row, key) {
    if (row[key] != null) return row[key];
    const rk = amRowKey(row);
    for (const metric of state.extraMetrics) {
      const ed = state.extraData[metric];
      if (ed && ed.context && ed.context[rk] && ed.context[rk][key] != null) return ed.context[rk][key];
    }
    return null;
  }
  function schemaValue(column, row) {
    if (column.kind === 'games') return row.games;
    if (column.kind === 'weeks') return row.weeks;
    if (column.kind === 'sample') return contextValue(row, column.key);
    if (column.kind === 'metric') {
      if (column.metricKey === state.metric) return row.value;
      const ed = state.extraData[column.metricKey];
      return ed && ed.byId[amRowKey(row)] !== undefined ? ed.byId[amRowKey(row)] : null;
    }
    return null;
  }
  // The single ordered schema is the contract between headers, rows and CSV.
  // No renderer is allowed to infer an additional volume column from vol_col.
  function tableColumnSchema() {
    const cols = [
      {id:'rank', kind:'rank', label:'#'},
      {id:'player', kind:'player', label:'Player', sortKey:'name'},
      {id:'season', kind:'season', label:'Year'},
      {id:'games', kind:'games', label:'G', title:'Games played in the selected period', sortKey:'games'},
      {id:'weeks', kind:'weeks', label:'Wks', title:'Weeks in range'}
    ];
    contextColsFor().forEach(c => cols.push(c));
    if (state.filterColKeys) [...state.filterColKeys].forEach(function(key) {
      if (key !== state.metric && !state.extraMetrics.includes(key)) cols.push({id:'filter:' + key,kind:'filter',metricKey:key,label:_mLabel(key),sortKey:key});
    });
    cols.push({id:'metric:' + state.metric,kind:'metric',metricKey:state.metric,label:_mLabel(state.metric),sortKey:state.metric,primary:true});
    state.extraMetrics.forEach(function(key) {
      const _locked = _mLocked(key);
      cols.push({id:'metric:' + key, kind:'metric', metricKey:key,
        label: _mLabel(key), sortKey:key, locked: _locked});
    });
    if (state.showTrends) cols.push({id:'trend',kind:'trend',label:'Recent'});
    return cols;
  }
  function syncTableHeader() {
    const tr = document.querySelector('#amTable thead tr');
    if (!tr) return;
    tr.innerHTML = '';
    tableColumnSchema().forEach(function(c) {
      const th = document.createElement('th'); th.dataset.columnId = c.id;
      th.textContent = c.label; if (c.title) th.title = c.title;
      if (c.locked) th.insertAdjacentHTML('beforeend', ' ' + _LOCK_SVG);
      if (c.kind === 'rank') th.className = 'am-rank';
      else if (c.kind === 'player') th.className = 'am-player';
      else if (c.kind === 'season') { th.className = 'am-season-col'; th.id = 'amSeasonColHdr'; th.style.display = amIsEachYear() ? '' : 'none'; }
      else if (c.kind === 'weeks') { th.className = 'am-weeks'; th.style.display = state.responseWeekFiltered ? '' : 'none'; }
      else if (c.kind === 'metric') { th.className = 'am-barcell' + (c.primary ? '' : ' am-extra-header') + (c.locked ? ' am-th-locked' : ''); th.id = c.primary ? 'amMetricHeader' : 'amExtraHeader_' + c.metricKey; _bindColTip(th,c.metricKey); if (c.locked) th.title = _mLabel(c.metricKey) + ' is PRO only: tap to unlock'; }
      else { th.className = 'am-games' + (c.kind === 'sample' ? ' am-context-col-hdr' : ' am-filter-col-hdr'); if (c.kind === 'filter') { th.id='amFilterColHdr_'+c.metricKey; _bindColTip(th,c.metricKey); } }
      if (c.locked) {
        // Locked PRO column header: tapping opens the paywall, not a sort.
        th.classList.add('am-sortable');
        th.addEventListener('click', function() {
          if (typeof window.showPaywall === 'function') window.showPaywall('advanced-metrics-metric-' + c.metricKey);
        });
      } else if (c.sortKey) { th.classList.add('am-sortable'); th.addEventListener('click',function(){sortByCol(c.sortKey);}); }
      tr.appendChild(th);
    });
    updateSortHeaders();
  }
  function schemaSkeletonRows() {
    const ids = tableColumnSchema().map(c => c.id);
    return [0,1,2].map(function() {
      return '<tr class="am-row am-schema-skeleton">' + ids.map(function(id) {
        return '<td data-column-id="' + id + '"><span class="skeleton" style="display:block;height:12px;min-width:28px"></span></td>';
      }).join('') + '</tr>';
    }).join('');
  }
  function verifyRenderedSchema() {
    const expected = tableColumnSchema().map(c => c.id).join('|');
    document.querySelectorAll('#amTable tbody tr.am-row:not(.am-pin-divider)').forEach(function(tr) {
      const actual = [...tr.children].map(td => td.dataset.columnId || '').join('|');
      if (actual !== expected) console.error('Advanced Metrics column schema mismatch', {expected:expected, actual:actual});
    });
  }
  function syncContextCols() {
    syncTableHeader();
  }

  // Attach the shared metric-definition tooltip to a column header <th>.
  // Uses mouseenter/mouseleave (desktop hover) bound via addEventListener so
  // the sort click-handler already on the element is not affected.  Stores
  // the bound functions on the element itself so they can be removed cleanly
  // when the primary metric header is refreshed.
  function _bindColTip(th, key) {
    const desc = (cfg.metrics[key] && cfg.metrics[key].desc) || '';
    // Remove any previous tooltip listeners to avoid duplicates on re-bind.
    if (th._amTipEnter) { th.removeEventListener('mouseenter', th._amTipEnter); delete th._amTipEnter; }
    if (th._amTipLeave) { th.removeEventListener('mouseleave', th._amTipLeave); delete th._amTipLeave; }
    if (!desc) { delete th.dataset.def; th.removeAttribute('title'); th.style.cursor = ''; return; }
    th.dataset.def = desc;
    th.removeAttribute('title');
    th.style.cursor = 'help';
    th._amTipEnter = function(e) { if (typeof advEnterMetricDef === 'function') advEnterMetricDef(e); };
    th._amTipLeave = function(e) { if (typeof advLeaveMetricDef === 'function') advLeaveMetricDef(e); };
    th.addEventListener('mouseenter', th._amTipEnter);
    th.addEventListener('mouseleave', th._amTipLeave);
  }

  function syncExtraCols() {
    syncTableHeader();
    syncTrendHeader();
  }
  function updateMetricTip() {
    const info = document.getElementById('amMetricInfo');
    if (!info) return;
    const desc = (cfg.metrics[state.metric] && cfg.metrics[state.metric].desc) || '';
    if (desc) info.dataset.def = desc;
    else delete info.dataset.def;
  }
  // Lowest threshold for a metric - the sensible default so the leaderboard
  // isn't dominated by tiny-sample players (e.g. 1-carry QBs at 198 yds/carry).
  function defaultVol(m) {
    // Empty means "server-selected shared qualification", not unrestricted.
    // Explicit user choices remain in state.minVol and are sent unchanged.
    return '';
  }
  // Volume filter: switches between season options and weekly-appropriate options
  // depending on whether a week range is active.
  function updateVolCtrl() {
    if (!gamesCtrl || !minGamesSel) return;
    const mspec = cfg.metrics[state.metric];
    const isWeekly = state.weekRange && state.weekRange !== '';

    if (isWeekly) {
      const wv = mspec && mspec.weeklyVol;
      if (!wv || !wv.opts || !wv.opts.length) {
        gamesCtrl.style.display = 'none';
        return;
      }
      if (volLabel) volLabel.textContent = wv.label;
      const prev = state.minVol;
      minGamesSel.innerHTML = '<option value="">Any</option>'
        + wv.opts.map(v => '<option value="' + v + '"' + (String(v) === prev ? ' selected' : '') + '>' + v + '+</option>').join('');
      minGamesSel.value = prev || '';
      gamesCtrl.style.display = '';
    } else {
      const spec = mspec && mspec.minVol;
      if (!spec) { gamesCtrl.style.display = 'none'; return; }
      if (volLabel) volLabel.textContent = spec.label;
      const prev = state.minVol;
      minGamesSel.innerHTML = '<option value="">Any</option>'
        + (spec.opts || []).map(v => '<option value="' + v + '"' + (String(v) === prev ? ' selected' : '') + '>' + v + '+</option>').join('');
      minGamesSel.value = prev || '';
      gamesCtrl.style.display = '';
    }
    // Sync filter-bar visibility after the vol control state changes.
    updateFilterBar();
  }
  function updatePosButtons() {
    const rel = new Set(relevantPositions(state.metric));
    posWrap.querySelectorAll('[data-pos]').forEach(b => {
      const p = b.dataset.pos;
      const ok = p === 'ALL' || rel.has(p);
      b.disabled = !ok;
      b.style.opacity = ok ? '' : '0.35';
      b.style.cursor = ok ? 'pointer' : 'not-allowed';
      b.classList.toggle('active', p === state.position);
    });
  }
  // Compact week list -> "1-9" / "1-3, 7-10" for the traded-player tooltip.
  function amWeekRange(wks) {
    if (!wks || !wks.length) return '';
    wks = wks.slice().sort((a, b) => a - b);
    const parts = []; let s = wks[0], p = wks[0];
    for (let i = 1; i < wks.length; i++) {
      if (wks[i] === p + 1) { p = wks[i]; continue; }
      parts.push(s === p ? String(s) : s + '-' + p); s = p = wks[i];
    }
    parts.push(s === p ? String(s) : s + '-' + p);
    return parts.join(', ');
  }
  // Team cell label. For a player who was on one team all season this is just
  // the team code. For a mid-season trade it shows the team relevant to the
  // current view (the filtered team, else the primary) with a '*' whose tooltip
  // lists the weeks he was on it - so filtering "TB" for 2025 flags that a
  // traded player was only there part of the year.
  function amTeamLabel(r) {
    const teams = (r.teams && r.teams.length) ? r.teams : [r.team || ''];
    const sel = state.team ? state.team.toUpperCase() : '';
    let shown = r.team || '';
    if (sel && teams.some(t => (t || '').toUpperCase() === sel)) {
      shown = teams.find(t => (t || '').toUpperCase() === sel) || shown;
    }
    if (!r.multi_team) return shown;
    const wks = (r.team_weeks || {})[shown];
    const tip = (wks && wks.length)
      ? (shown + ': weeks ' + amWeekRange(wks) + ' · traded mid-season')
      : (shown + ' · multiple teams this season');
    return shown + '<span class="am-team-star" style="color:var(--accent);cursor:help;margin-left:1px;font-weight:700;" title="' + tip.replace(/"/g, '&quot;') + '">*</span>';
  }
  function populateTeamFilter() {
    if (!teamSel) return;
    // Include every team a player was on this season (traded players have a
    // `teams` array), not just their displayed team, so the filter offers both.
    const teams = [...new Set(state.rows.flatMap(r => (r.teams && r.teams.length ? r.teams : [r.team || ''])).filter(Boolean))].sort();
    const prev = state.team;
    teamSel.innerHTML = '<option value="">All Teams</option>'
      + teams.map(t => '<option value="' + t + '"' + (t === prev ? ' selected' : '') + '>' + t + '</option>').join('');
    if (!teams.includes(state.team)) state.team = '';
  }

  function trendArrow(curr, prev, metricKey) {
    if (prev == null || prev === undefined) return '';
    const mkey = metricKey || state.metric;
    const isLower = !!(cfg.metrics[mkey] && cfg.metrics[mkey].lowerBetter);
    const delta = Number(curr) - Number(prev);
    if (Math.abs(Number(prev)) < 0.001) return '';
    const pct = Math.abs(delta / Number(prev));
    if (pct < 0.03) return '';
    const improved = isLower ? delta < 0 : delta > 0;
    const word = improved ? 'Up' : 'Down';
    const tip = word + ' vs ' + (prevSeasonLabel() || 'last season') + ': '
      + fmtVal(prev, mkey) + ' → ' + fmtVal(curr, mkey);
    return improved
      ? '<span class="am-trend-up" title="' + tip + '">&#8593;</span>'
      : '<span class="am-trend-down" title="' + tip + '">&#8595;</span>';
  }
  function prevSeasonLabel() {
    const cur = state.season ? parseInt(state.season) : (cfg.seasons && cfg.seasons[0]);
    return cur ? String(cur - 1) : '';
  }

  function render() {
    if (state.fetching) return; // keep the spinner up while a fetch is in flight
    const rel = new Set(relevantPositions(state.metric));
    const up = v => String(v || '').toUpperCase();

    // Sort the full positional set first to establish canonical ranks.
    const posRows = (state.position === 'ALL'
      ? state.rows.filter(r => rel.has(up(r.position)))
      : state.rows.filter(r => up(r.position) === state.position));
    posRows.sort((a, b) => {
      const desc = state.sortDir === 'desc';
      if (state.sortBy === 'name') {
        const diff = (a.name || '').localeCompare(b.name || '');
        return desc ? -diff : diff;
      }
      if (state.sortBy === 'games') {
        const av = a.vol != null ? a.vol : a.games;
        const bv = b.vol != null ? b.vol : b.games;
        return _amCmpVal(av, bv, desc);
      }
      if (state.sortBy === state.metric || !state.extraData[state.sortBy]) {
        return _amCmpVal(a.value, b.value, desc);
      }
      const ed = state.extraData[state.sortBy];
      return _amCmpVal(
        ed.byId[amRowKey(a)],
        ed.byId[amRowKey(b)],
        desc);
    });

    // Rank map so roster/search filters preserve original rank numbers.
    const rankMap = new Map(posRows.map((r, i) => [amRowKey(r), i + 1]));

    // All displayed primary values identical? Then quality ranks are
    // arbitrary and the percentile badge would be a lie (see percentileBadge).
    const _primNonMissing = posRows.map(r => r.value).filter(v => !_amMissing(v));
    const _primAllTie = _primNonMissing.length > 1
      && _primNonMissing.every(v => Number(v) === Number(_primNonMissing[0]));

    // Quality rank: standing on the PRIMARY metric in its "good" direction
    // (ascending for lower-is-better, descending otherwise), independent of the
    // current display sort. Drives the percentile badge so flipping the sort to
    // show the worst values first doesn't mislabel them as "Top 5%".
    const _primLower = !!(cfg.metrics[state.metric] && cfg.metrics[state.metric].lowerBetter);
    const qualityRankMap = new Map(
      posRows.slice()
        .sort((a, b) => _amCmpVal(a.value, b.value, !_primLower))
        .map((r, i) => [amRowKey(r), i + 1])
    );

    // Scale to the true max, but if the leader is a big outlier (>30% above the
    // 95th-percentile value) cap the scale so one player doesn't squish the rest.
    // Bars above the cap clamp at 100% width.
    const _vals = posRows.filter(r => !_amMissing(r.value))
      .map(r => Math.abs(Number(r.value))).sort((a, b) => a - b);
    const _p95 = _vals[Math.min(Math.floor(_vals.length * 0.95), _vals.length - 1)] || 1;
    const _trueMax = _vals[_vals.length - 1] || 1;
    const maxAbs = Math.min(_trueMax, _p95 * 1.3) || 1;
    // Signed field detection: when any value is negative (EPA, CPOE, RYOE…),
    // |value|/max would draw a long bar for a below-average player. Use a
    // min→max mapping instead. Lower-better metrics are inverted so good = long.
    const _signedVals = posRows.map(r => r.value).filter(v => !_amMissing(v)).map(v => Number(v));
    const _fieldMin = _signedVals.length ? Math.min.apply(null, _signedVals) : 0;
    const _fieldMax = _signedVals.length ? Math.max.apply(null, _signedVals) : 1;
    // Bar fill %: relative-to-leader for normal non-negative higher-better
    // metrics; min→max (direction-aware) for signed or lower-better metrics.
    function _barPct(val, opts) {
      if (val == null || isNaN(Number(val))) return 2;
      const v = Number(val);
      if (opts.signed || opts.lower) {
        const span = (opts.fieldMax - opts.fieldMin) || 1;
        let t = (v - opts.fieldMin) / span;
        if (opts.lower) t = 1 - t;
        return Math.round(8 + Math.max(0, Math.min(1, t)) * 92);
      }
      return Math.min(100, Math.max(2, Math.round(Math.abs(v) / opts.capMax * 100)));
    }
    const _primBar = { signed: _fieldMin < 0, lower: _primLower,
      fieldMin: _fieldMin, fieldMax: _fieldMax, capMax: maxAbs };

    // Extra columns: same logic - scale to the max among the displayed rows so the leader fills the bar.
    const extraMaxMap = {};
    const extraAvgMap = {};
    const extraRankMap = {};
    const extraRankTotal = {};
    const extraTieMap = {};  // key -> true when every displayed value ties
    const extraBarMap = {};  // key -> {signed, lower, fieldMin, fieldMax, capMax}
    state.extraMetrics.forEach(function(key) {
      const ed = state.extraData[key];
      if (!ed) return;
      let mx = 0, sum = 0, n = 0, fMin = Infinity, fMax = -Infinity;
      posRows.forEach(function(r) {
        const v = ed.byId[amRowKey(r)];
        if (v != null) {
          const nv = Number(v) || 0;
          mx = Math.max(mx, Math.abs(nv));
          fMin = Math.min(fMin, nv); fMax = Math.max(fMax, nv);
          sum += nv;
          n++;
        }
      });
      extraMaxMap[key] = mx || 1;
      if (n) {
        const _eLow = !!(cfg.metrics[key] && cfg.metrics[key].lowerBetter);
        extraBarMap[key] = { signed: fMin < 0, lower: _eLow,
          fieldMin: fMin, fieldMax: fMax, capMax: mx || 1 };
      }
      if (n) extraAvgMap[key] = sum / n;
      const _extraLower = !!(cfg.metrics[key] && cfg.metrics[key].lowerBetter);
      const _ePairs = posRows
        .map(r => [amRowKey(r), ed.byId[amRowKey(r)]])
        .filter(([, v]) => v != null)
        .sort((a, b) => { const d = Number(a[1]) - Number(b[1]); return _extraLower ? d : -d; });
      const _eRankMap = {};
      _ePairs.forEach(([id], i) => { _eRankMap[id] = i + 1; });
      extraRankMap[key] = _eRankMap;
      extraRankTotal[key] = _ePairs.length;
      // All displayed values identical -> percentile badges would be arbitrary.
      const _eDistinct = new Set(_ePairs.map(([, v]) => Number(v)));
      extraTieMap[key] = _ePairs.length > 1 && _eDistinct.size <= 1;
    });



    // Apply roster/search filters for display only (order already set by posRows sort).
    // Pinned players float to the top of the display; rank numbers reflect sort position.
    let displayRows = state.pinnedIds.size > 0
      ? [...posRows.filter(r => state.pinnedIds.has(String(r.player_id))),
         ...posRows.filter(r => !state.pinnedIds.has(String(r.player_id)))]
      : posRows.slice();
    if (state.rosterOnly) displayRows = displayRows.filter(r => ownedIds.has(String(r.player_id)));
    if (state.search) {
      const q = state.search.toLowerCase();
      displayRows = displayRows.filter(r => (r.name || '').toLowerCase().includes(q));
    }
    if (state.team) {
      const _st = state.team.toUpperCase();
      // Match any team the player was on this season (handles mid-season trades).
      displayRows = displayRows.filter(r => {
        const ts = (r.teams && r.teams.length) ? r.teams : [r.team || ''];
        return ts.some(t => (t || '').toUpperCase() === _st);
      });
    }
    if (state.ageMin !== '' || state.ageMax !== '') {
      displayRows = displayRows.filter(function(r) {
        const age = r.age;
        if (age == null) return false;
        if (state.ageMin !== '' && age < Number(state.ageMin)) return false;
        if (state.ageMax !== '' && age > Number(state.ageMax)) return false;
        return true;
      });
    }
    state.comboFilters.forEach(function(f) {
      displayRows = displayRows.filter(function(r) {
        let v;
        if (f.key === 'primary') {
          v = Number(r.value);
        } else if (f.key === 'age') {
          v = r.age != null ? Number(r.age) : null;
        } else if (f.key === 'exp') {
          v = r.years_exp != null ? Number(r.years_exp) : null;
        } else {
          const ed2 = state.extraData[f.key];
          v = ed2 ? ed2.byId[amRowKey(r)] : undefined;
          v = (v !== undefined && v !== null) ? Number(v) : null;
        }
        if (v == null) return false;
        return f.op === 'gte' ? v >= Number(f.val) : v <= Number(f.val);
      });
    });
    // Hide rows that lack data in most of the loaded extra metric columns (e.g. a
    // fullback shown with all dashes when rushing efficiency metrics are added).
    if (state.extraMetrics.length > 0) {
      const loadedExtras = state.extraMetrics.filter(k => state.extraData[k]);
      if (loadedExtras.length > 0) {
        const minHits = Math.max(1, Math.ceil(loadedExtras.length / 2));
        displayRows = displayRows.filter(function(r) {
          let hits = 0;
          for (let i = 0; i < loadedExtras.length; i++) {
            const ed = state.extraData[loadedExtras[i]];
            if (ed && ed.byId[amRowKey(r)] != null) { hits++; if (hits >= minHits) return true; }
          }
          return false;
        });
      }
    }

    if (loading) loading.style.display = 'none';
    if (!displayRows.length) {
      // No data: hide the table (headers included) and show the empty state
      // in its place instead of stacking "No data yet" above the headers.
      empty.style.display = ''; tbody.innerHTML = '';
      if (tableWrap) tableWrap.style.display = 'none';
      if (avgNote) avgNote.style.display = 'none';
      if (paginationEl) paginationEl.style.display = 'none';
      window.brEmptyState(empty, state.rosterOnly
        ? { icon: 'search', title: 'No ranked players', message: 'None of your rostered players rank for this metric yet.' }
        : { icon: 'search', title: 'No data yet', message: 'This metric doesn’t have enough sample to rank for the current filters.' });
      return;
    }
    empty.style.display = 'none';
    if (tableWrap) tableWrap.style.display = '';

    // Average marker across all displayed rows (position-filtered but not roster/search filtered).
    let avgPct = null;
    if (posRows.length) {
      const avg = posRows.reduce((s, r) => s + (Number(r.value) || 0), 0) / posRows.length;
      avgPct = Math.max(0, Math.min(100, _barPct(avg, _primBar)));
      if (avgNote) {
        avgNote.style.display = '';
        const lbl = state.position !== 'ALL' ? state.position : 'Field';
        // One entry per visible metric column: primary first, then extras.
        const parts = [(_mLabel(state.metric))
          + ' ' + fmtVal(avg, state.metric)];
        state.extraMetrics.forEach(function(key) {
          if (extraAvgMap[key] != null) {
            parts.push((_mLabel(key))
              + ' ' + fmtVal(extraAvgMap[key], key));
          }
        });
        avgNoteTxt.textContent = lbl + ' averages: ' + parts.join(' · ');
        if (parts.length === 1) avgNoteTxt.textContent = lbl + ' average: ' + fmtVal(avg, state.metric);
        // Mobile rework: collapsed one-liner ("Field avg · <Metric> <val>") plus
        // an expandable grid. Desktop keeps the paragraph above untouched.
        const avgToggleTxt = document.getElementById('amAvgToggleText');
        if (avgToggleTxt) {
          avgToggleTxt.textContent = '';
          const primaryLabel = _mLabel(state.metric);
          avgToggleTxt.append(document.createTextNode(lbl + ' avg · ' + primaryLabel + ' '));
          const avgVal = document.createElement('b');
          avgVal.textContent = fmtVal(avg, state.metric);
          avgToggleTxt.append(avgVal);
        }
        const avgGrid = document.getElementById('amAvgGrid');
        if (avgGrid) {
          avgGrid.textContent = '';
          parts.forEach(function(p) {
            const cell = document.createElement('div');
            cell.className = 'am-avg-cell';
            const idx = p.lastIndexOf(' ');
            cell.append(document.createTextNode(idx > 0 ? p.slice(0, idx) + ' ' : p));
            if (idx > 0) {
              const vb = document.createElement('b');
              vb.textContent = p.slice(idx + 1);
              cell.append(vb);
            }
            avgGrid.append(cell);
          });
        }
        if (avgNote) avgNote.classList.toggle('am-avg-open', !!window.__amAvgOpen);
        // Legend only when there are trend arrows to explain (single-metric view
        // with prior-season data loaded).
        const trendLegend = document.getElementById('amTrendLegend');
        if (trendLegend) {
          const hasTrend = state.extraMetrics.length === 0
            && state.prevData && Object.keys(state.prevData).length > 0;
          trendLegend.style.display = hasTrend ? '' : 'none';
        }
      }
    } else if (avgNote) {
      avgNote.style.display = 'none';
    }

    // Snapshot the fully filtered/sorted view (pre-pagination) for CSV export.
    state._exportRows = displayRows;

    // Pagination: clamp current page then slice.
    const total = displayRows.length;
    const maxPage = Math.max(0, Math.ceil(total / PAGE_SIZE) - 1);
    if (state.page > maxPage) state.page = maxPage;
    const start = state.page * PAGE_SIZE;
    const pageRows = displayRows.slice(start, start + PAGE_SIZE);

    // Rank by sort position in posRows (not display position) so pinned players
    // show their true rank rather than 1, 2, 3... just because they float to top.
    const _visibleIds = new Set(displayRows.map(r => amRowKey(r)));
    const filteredRankMap = new Map(
      posRows
        .filter(r => _visibleIds.has(amRowKey(r)))
        .map((r, i) => [amRowKey(r), i + 1])
    );

    const multiMode = state.extraMetrics.length > 0;
    const totalRanked = posRows.length;
    tbody.innerHTML = pageRows.map((r, i) => {
      const safe = (r.name || '').replace(/'/g, "\\'");
      const col = posColor(r.position);
      const owned = ownedIds.has(String(r.player_id));
      const pinned = state.pinnedIds.has(String(r.player_id));
      const rank = filteredRankMap.get(amRowKey(r)) || '';
      const gamesCell = '<td class="am-games" data-column-id="games">' + (r.games != null ? r.games : '–') + '</td>';
      const weeksCell = '<td class="am-weeks" data-column-id="weeks" style="display:' + (!!(state.weekRange && state.weekRange !== '') ? '' : 'none') + '">' + (r.weeks != null ? r.weeks : '–') + '</td>';
      // Context volume cells (plain numbers, like Games) for the metric's category.
      let contextCells = '';
      contextColsFor().forEach(function(c) {
        const cv = schemaValue(c, r);
        contextCells += '<td class="am-games" data-column-id="' + c.id + '">' + (cv != null ? cv : '–') + '</td>';
      });
      const ownedBadge = owned ? '<span class="am-owned-badge">YOURS</span>' : '';
      const pinBtn = '<button class="am-pin-btn' + (pinned ? ' am-pin-active' : '') + '" '
        + 'onclick="event.stopPropagation();amTogglePin(\'' + r.player_id + '\')" '
        + 'title="' + (pinned ? 'Unpin' : (state.pinnedIds.size >= _AM_MAX_PINS ? 'Unpin another player first (max 5)' : 'Pin to compare')) + '">' + _PIN_SVG + '</button>';
      const rankCell = '<td class="am-rank" data-column-id="rank"><div class="am-rank-cell">' + pinBtn + '<span>' + rank + '</span></div></td>';
      const playerCell = '<td class="am-player" data-column-id="player"><div class="am-player-inner">'
        + '<span class="am-name">' + (r.name || '') + '</span>'
        + ownedBadge
        + '<span class="am-player-right">'
        + '<span class="am-meta">' + amTeamLabel(r) + '</span>'
        + '<span class="am-meta" style="color:' + col + ';font-weight:600">' + r.position + '</span>'
        + '</span></div></td>';
      const showYear = amIsEachYear();
      const seasonCell = '<td class="am-season-col" data-column-id="season"' + (showYear ? '' : ' style="display:none"') + '>'
        + (showYear && r.season != null ? r.season : '') + '</td>';

      // Compact filter-column cells (appear right before the primary metric cell).
      let filterColCells = '';
      if (state.filterColKeys && state.filterColKeys.size) {
        [...state.filterColKeys].forEach(function(fkey) {
          if (fkey === state.metric || state.extraMetrics.includes(fkey)) return;
          const fed = state.extraData[fkey];
          if (!fed) {
            filterColCells += '<td class="am-games" data-column-id="filter:' + fkey + '" style="opacity:.35">–</td>';
            return;
          }
          const fval = fed.byId[amRowKey(r)];
          filterColCells += '<td class="am-games" data-column-id="filter:' + fkey + '">' + (fval != null ? fmtVal(fval, fkey) : '–') + '</td>';
        });
      }

      const badge = percentileBadge(qualityRankMap.get(amRowKey(r)) || rank, totalRanked, _primAllTie);
      const primaryValue = schemaValue({kind:'metric', metricKey:state.metric}, r);
      const prevVal = amIsMultiSeason() ? null : state.prevData[String(r.player_id)];
      const trend = trendArrow(r.value, prevVal);
      let metricCell;
      if (!multiMode) {
        const pct = _barPct(Number(primaryValue), _primBar);
        const avgLbl = (avgPct != null && i === 0) ? '<span class="am-bar-avg-lbl">AVG</span>' : '';
        const avgMark = (avgPct != null)
          ? '<div class="am-bar-avg" style="left:' + avgPct + '%" title="' + (state.position !== 'ALL' ? state.position : 'Field') + ' average">' + avgLbl + '</div>'
          : '';
        metricCell = '<td class="am-barcell" data-column-id="metric:' + state.metric + '"><div class="am-metric-cell">'
          + '<div class="am-metric-bar"><div class="am-bar-track"><div class="am-bar-fill" style="width:' + pct + '%;background:' + col + '"></div>' + avgMark + '</div></div>'
          + '<div class="am-val-wrap"><div class="am-val-row">' + trend + '<span class="am-val">' + fmtVal(primaryValue, state.metric) + '</span></div>' + badge + '</div>'
          + '</div></td>';
      } else {
        const pct = _barPct(Number(primaryValue), _primBar);
        const avgLbl2 = (avgPct != null && i === 0) ? '<span class="am-bar-avg-lbl">AVG</span>' : '';
        const avgMark2 = (avgPct != null)
          ? '<div class="am-bar-avg" style="left:' + avgPct + '%" title="' + (state.position !== 'ALL' ? state.position : 'Field') + ' average">' + avgLbl2 + '</div>'
          : '';
        metricCell = '<td class="am-barcell" data-column-id="metric:' + state.metric + '"><div class="am-metric-cell">'
          + '<div class="am-metric-bar"><div class="am-bar-track"><div class="am-bar-fill" style="width:' + pct + '%;background:' + col + '"></div>' + avgMark2 + '</div></div>'
          + '<div class="am-val-wrap"><div class="am-val-row">' + trend + '<span class="am-val">' + fmtVal(primaryValue, state.metric) + '</span></div>' + badge + '</div>'
          + '</div></td>';
        state.extraMetrics.forEach(function(key) {
          const ed = state.extraData[key];
          if (!ed) {
            metricCell += '<td class="am-barcell" data-column-id="metric:' + key + '"><div class="am-metric-cell">'
              + '<div class="am-metric-bar"><div class="am-skel-bar"></div></div>'
              + '<div class="am-val-wrap"><span class="am-val" style="opacity:.25">–</span></div>'
              + '</div></td>';
            return;
          }
          // PRO-locked extra column: the column renders blocked, like a
          // paywalled column. Blurred placeholder bars (deterministic width
          // per player so it reads as hidden data) plus a lock badge; the
          // whole column is one click target that opens the paywall.
          if (ed.proLocked) {
            const _ml = _mLabel(key);
            const _rs = String(amRowKey(r)); let _hh = 0;
            for (let _i = 0; _i < _rs.length; _i++) _hh = (_hh * 31 + _rs.charCodeAt(_i)) % 97;
            const _bw = 30 + _hh;
            metricCell += '<td class="am-barcell am-cell-locked" data-column-id="metric:' + key + '"'
              + ' data-pro-metric="' + key + '" title="' + _esc(_ml) + ' is PRO only: tap to unlock">'
              + '<div class="am-metric-cell"><div class="am-metric-bar"><div class="am-bar-track"><div class="am-bar-fill am-locked-blur" style="width:' + _bw + '%"></div></div></div>'
              + '<div class="am-val-wrap">' + _LOCK_SVG + '</div>'
              + '</div></td>';
            return;
          }
          const val = schemaValue({kind:'metric', metricKey:key}, r);
          const _eBar = extraBarMap[key] || { signed: false, lower: false, fieldMin: 0, fieldMax: 1, capMax: extraMaxMap[key] };
          const pctBar = val != null ? _barPct(Number(val), _eBar) : 2;
          const disp = val != null ? fmtVal(val, key) : '–';
          const avgVE = extraAvgMap[key];
          const avgMarkE = (avgVE != null)
            ? '<div class="am-bar-avg" style="left:' + Math.max(0, Math.min(100, _barPct(avgVE, _eBar))) + '%" '
              + 'title="Average: ' + fmtVal(avgVE, key) + '"></div>'
            : '';
          const rkE = extraRankMap[key] ? extraRankMap[key][amRowKey(r)] : null;
          const badgeE = (rkE && extraRankTotal[key]) ? percentileBadge(rkE, extraRankTotal[key], extraTieMap[key]) : '';
          const prevE = (!amIsMultiSeason() && state.extraPrevData[key]) ? state.extraPrevData[key][String(r.player_id)] : undefined;
          const trendE = (val != null && prevE !== undefined) ? trendArrow(val, prevE, key) : '';
          metricCell += '<td class="am-barcell" data-column-id="metric:' + key + '"><div class="am-metric-cell">'
            + '<div class="am-metric-bar"><div class="am-bar-track"><div class="am-bar-fill" style="width:' + pctBar + '%;background:' + col + '"></div>' + avgMarkE + '</div></div>'
            + '<div class="am-val-wrap"><div class="am-val-row">' + trendE + '<span class="am-val">' + disp + '</span></div>' + badgeE + '</div>'
            + '</div></td>';
        });
      }

      // Divider row after the last pinned player (when pinned and unpinned are both present).
      const nextRow = pageRows[i + 1];
      const isPinBoundary = pinned && nextRow && !state.pinnedIds.has(String(nextRow.player_id));
      const divider = isPinBoundary
        ? '<tr class="am-pin-divider"><td colspan="99"></td></tr>'
        : '';

      return '<tr class="am-row' + (owned ? ' am-owned' : '') + (pinned ? ' am-pinned' : '') + '" style="cursor:pointer;" '
        + 'onclick="window.openPlayerModal&&openPlayerModal(\'' + r.player_id + '\',\'' + safe + '\',{tab:\'metrics\'})">'
        + rankCell
        + playerCell
        + seasonCell
        + gamesCell
        + weeksCell
        + contextCells
        + filterColCells
        + metricCell
        + (state.showTrends ? trendCellHtml(r.player_id, col) : '')
        + '</tr>' + divider;
    }).join('');
    verifyRenderedSchema();

    updateComparePinnedBtn();

    // Pagination controls.
    if (paginationEl) {
      if (total > PAGE_SIZE) {
        paginationEl.style.display = '';
        const end = Math.min(start + PAGE_SIZE, total);
        paginationEl.innerHTML = '<button class="am-page-btn" id="amPagePrev"' + (state.page === 0 ? ' disabled' : '') + '>&larr; Prev</button>'
          + '<span class="am-page-info">' + (start + 1) + '–' + end + ' of ' + total + '</span>'
          + '<button class="am-page-btn" id="amPageNext"' + (state.page >= maxPage ? ' disabled' : '') + '>Next &rarr;</button>';
        document.getElementById('amPagePrev').onclick = () => { state.page--; render(); };
        document.getElementById('amPageNext').onclick = () => { state.page++; render(); };
      } else {
        paginationEl.style.display = 'none';
      }
    }
  }
  function updateVolHeader() {
    // Legacy code relabelled the Games header from metric-specific vol_col while
    // the body still rendered games. That produced duplicate CAR headers and
    // shifted-looking Passing values. Rebuild both from the canonical schema.
    syncTableHeader();
  }
  // Resolve ranges against the selected season's actual populated weeks.
  function availableWeeks() {
    if (amIsMultiSeason()) return [];
    const year = amSelectedSeasons()[0];
    return ((cfg.availableWeeksBySeason || {})[String(year)] || []).map(Number).sort(function(a,b){return a-b;});
  }
  function isMetricWeeklyCapable() {
    return !!(cfg.metrics[state.metric] && cfg.metrics[state.metric].weeklyCapable) && !amIsMultiSeason();
  }
  function resolveWeekRange() {
    const weeks = availableWeeks();
    if (!state.weekRange || !weeks.length || !isMetricWeeklyCapable()) return { ws: null, we: null };
    const lo = weeks[0], hi = weeks[weeks.length - 1];
    if (state.weekRange === 'last2' || state.weekRange === 'last4') {
      const count = state.weekRange === 'last2' ? 2 : 4;
      const picked = weeks.slice(-count);
      return { ws: picked[0], we: picked[picked.length - 1] };
    }
    if (state.weekRange === 'custom') {
      const ws = Math.max(lo, Math.min(hi, Number(state.weekStart)));
      const we = Math.max(lo, Math.min(hi, Number(state.weekEnd)));
      return { ws: Math.min(ws,we), we: Math.max(ws,we) };
    }
    return { ws: null, we: null };
  }

  function updateWeekNote(isWeekFiltered, weekCapable) {
    const el = document.getElementById('amWeekNote');
    if (!el) return;
    // The response is authoritative about whether filtering was applied. For a
    // non-weekly metric, always explain why the disabled control is season-only.
    if (!weekCapable) el.style.display = 'flex';
    else el.style.display = 'none';
  }

  // ── Graph Metrics (scatter X vs Y, optional bubble = 3rd metric) ──────────
  function _amEsc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  // Short label = surname, keeping a generational suffix attached
  // (e.g. "Harold Fannin Jr." -> "Fannin Jr.", "Marvin Harrison Jr." -> "Harrison Jr.").
  function _amLastName(name) {
    const parts = String(name || '').trim().split(/\s+/).filter(Boolean);
    if (!parts.length) return '';
    const SUF = /^(jr|sr|ii|iii|iv|v)\.?$/i;
    const last = parts.length - 1;
    if (parts.length >= 2 && SUF.test(parts[last])) return parts[last - 1] + ' ' + parts[last];
    return parts[last];
  }
  // Graph's own theme (independent of the site, so shared images can be either).
  // Defaults to the current site theme when the modal opens.
  let _amGraphTheme = (document.documentElement.getAttribute('data-theme') === 'dark') ? 'dark' : 'light';
  function _amGraphPalette() {
    return (_amGraphTheme === 'dark')
      ? { bg: '#0f172a', text: '#e2e8f0', muted: '#94a3b8',
          grid: 'rgba(148,163,184,.22)', axis: 'rgba(148,163,184,.45)', border: 'rgba(148,163,184,.25)' }
      : { bg: '#ffffff', text: '#1e293b', muted: '#64748b',
          grid: 'rgba(100,116,139,.16)', axis: 'rgba(100,116,139,.40)', border: 'rgba(100,116,139,.20)' };
  }
  // Preload the BR logo (theme-appropriate) as a data URI so it embeds in the SVG
  // and survives canvas rasterization for the shareable PNG.
  const _amLogoCache = {};
  function _amLoadLogo(theme) {
    if (_amLogoCache[theme] !== undefined) return Promise.resolve(_amLogoCache[theme]);
    const url = (theme === 'dark') ? '/static/BR_Logo_dark.png?v=6c0c4828' : '/static/BR_Logo.png?v=6c0c4828';
    return fetch(url).then(function(r) { return r.ok ? r.blob() : null; }).then(function(b) {
      if (!b) { _amLogoCache[theme] = ''; return ''; }
      return new Promise(function(res) {
        const fr = new FileReader();
        fr.onload = function() { _amLogoCache[theme] = fr.result; res(fr.result); };
        fr.onerror = function() { _amLogoCache[theme] = ''; res(''); };
        fr.readAsDataURL(b);
      });
    }).catch(function() { _amLogoCache[theme] = ''; return ''; });
  }
  function _amSyncGraphThemeBtn() {
    const b = document.getElementById('amGraphThemeBtn');
    if (b) b.innerHTML = (_amGraphTheme === 'dark')
      ? '<span style="font-size:13px;">☾</span> Dark'
      : '<span style="font-size:13px;">☀</span> Light';
  }
  window.amToggleGraphTheme = function() {
    _amGraphTheme = (_amGraphTheme === 'dark') ? 'light' : 'dark';
    _amSyncGraphThemeBtn();
    window.amRenderGraph();
  };
  // Position filter local to the graph modal (null = all). Initialized from
  // state.position each time the modal opens; can be changed without leaving the modal.
  let _amGraphPos = null;
  // Build <optgroup> options from cfg.metrics, filtered to the graph-local position.
  function _amGraphMetricOptions(selectedKey) {
    const pos = _amGraphPos;
    const cats = {};
    Object.keys(cfg.metrics).forEach(function(k) {
      const m = cfg.metrics[k];
      if (pos && m.positions && m.positions.indexOf(pos) === -1) return;
      const c = m.category || 'Other';
      (cats[c] = cats[c] || []).push([k, m.label]);
    });
    const order = ['Value', 'General', 'Passing', 'Rushing', 'Receiving'];
    const catKeys = Object.keys(cats).sort(function(a, b) {
      const ia = order.indexOf(a), ib = order.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
    });
    let html = '';
    catKeys.forEach(function(c) {
      html += '<optgroup label="' + _amEsc(c) + '">';
      cats[c].sort(function(a, b) { return a[1].localeCompare(b[1]); }).forEach(function(pair) {
        html += '<option value="' + _amEsc(pair[0]) + '"'
          + (pair[0] === selectedKey ? ' selected' : '') + '>' + _amEsc(pair[1]) + '</option>';
      });
      html += '</optgroup>';
    });
    return html;
  }
  // Min vol state for the graph-modal X metric. Initialized to defaultVol(xk)
  // each time X changes; user can override via the #amGraphMinVolSel control.
  let _amGraphMinVol = '';
  let _amGraphLastXk = '';
  // Populate/show the min-vol control for a given X metric key, or hide it.
  function _amUpdateGraphVolCtrl(xk) {
    const ctrl = document.getElementById('amGraphVolCtrl');
    const sel = document.getElementById('amGraphMinVolSel');
    const lbl = document.getElementById('amGraphVolLabel');
    if (!ctrl || !sel) return;
    const spec = cfg.metrics[xk] && cfg.metrics[xk].minVol;
    if (!spec || !spec.opts || !spec.opts.length) { ctrl.style.display = 'none'; return; }
    if (lbl) lbl.textContent = spec.label;
    sel.innerHTML = '<option value="">Any</option>'
      + spec.opts.map(function(v) {
          return '<option value="' + v + '"' + (String(v) === _amGraphMinVol ? ' selected' : '') + '>' + v + '+</option>';
        }).join('');
    sel.value = _amGraphMinVol;
    ctrl.style.display = '';
  }
  // Leaderboard params for one metric. Uses the graph-modal's own position filter
  // (_amGraphPos) if set, otherwise falls back to the page's position filter.
  // For the X metric _amGraphMinVol overrides the default min-vol so the user
  // can adjust it from the graph modal's own control.
  function _amGraphParams(metricKey, isX) {
    const p = new URLSearchParams({ metric: metricKey, platform: cfg.platform });
    if (cfg.leagueId) p.set('league_id', cfg.leagueId);
    if (state.season) p.set('season', String(state.season));
    if (state.combine && amIsMultiSeason()) p.set('combine', '1');
    const graphPos = _amGraphPos || (state.position !== 'ALL' ? state.position : null);
    if (graphPos) p.set('position', graphPos);
    const vol = isX ? _amGraphMinVol : defaultVol(metricKey);
    if (vol) p.set('min_vol', vol); else p.delete('min_vol');
    const wr = resolveWeekRange();
    if (wr.ws) p.set('week_start', String(wr.ws));
    if (wr.we) p.set('week_end', String(wr.we));
    return p;
  }
  let _amGraphToken = 0;
  let _amPinnedDot = null;
  const _amHsPreload = {};
  // Client-side cache of per-metric leaderboard responses, keyed by request URL
  // (the params are deterministic for a given metric + filter set). Reopening a
  // graph or toggling back to a prior axis is then instant -- no refetch. Entries
  // are invalidated whenever the underlying season/week filters change.
  const _amLbCache = new Map();
  let _amLbCacheSig = '';
  function _amFetchLeaderboard(url) {
    const hit = _amLbCache.get(url);
    if (hit) return Promise.resolve(hit);
    return fetch(url).then(function(r) { return r.ok ? r.json() : null; })
      .then(function(d) {
        if (d) {
          _amLbCache.set(url, d);
          if (_amLbCache.size > 60) _amLbCache.delete(_amLbCache.keys().next().value);
        }
        return d;
      }).catch(function() { return null; });
  }
  // Drop cached leaderboards when the season/week/league context changes so we
  // never show stale data after the user switches the page-level filters.
  function _amSyncLbCache() {
    const sig = [state.season || '', state.combine ? '1' : '0', cfg.leagueId || '', cfg.platform || '',
                 (resolveWeekRange().ws || ''), (resolveWeekRange().we || '')].join('|');
    if (sig !== _amLbCacheSig) { _amLbCache.clear(); _amLbCacheSig = sig; }
  }
  // Sync the graph-modal's position button active states to _amGraphPos.
  function _amSyncGraphPosBtns() {
    const bar = document.getElementById('amGraphPosBar');
    if (!bar) return;
    bar.querySelectorAll('[data-gpos]').forEach(function(btn) {
      const v = btn.getAttribute('data-gpos');
      btn.classList.toggle('active', v === (_amGraphPos || ''));
    });
  }
  window.amOpenGraph = function() {
    const modal = document.getElementById('amGraphModal');
    const xSel = document.getElementById('amGraphX');
    const ySel = document.getElementById('amGraphY');
    const zSel = document.getElementById('amGraphZ');
    const note = document.getElementById('amGraphCtxNote');
    if (!modal || !xSel || !ySel || !zSel) return;
    // Initialize the graph-local position filter from the page's current position.
    _amGraphPos = (state.position && state.position !== 'ALL') ? state.position : null;
    _amSyncGraphPosBtns();
    const applic = Object.keys(cfg.metrics).filter(function(k) {
      const m = cfg.metrics[k];
      return !_amGraphPos || !m.positions || m.positions.indexOf(_amGraphPos) >= 0;
    });
    // X = current primary metric; Y = next metric in same category.
    // If primary is the last in its category, use the one above it instead
    // (so Y is always adjacent and never from a different category).
    let curX = (applic.indexOf(state.metric) >= 0) ? state.metric : (applic[0] || '');
    let curY = '', curZ = '';
    if (curX) {
      const xCat = cfg.metrics[curX] && cfg.metrics[curX].category;
      // Sort within category by label (same order as the dropdown) for a stable "next".
      const catMetrics = applic
        .filter(function(k) { return cfg.metrics[k].category === xCat; })
        .sort(function(a, b) { return cfg.metrics[a].label.localeCompare(cfg.metrics[b].label); });
      const idx = catMetrics.indexOf(curX);
      if (catMetrics.length > 1) {
        curY = idx < catMetrics.length - 1 ? catMetrics[idx + 1] : catMetrics[idx - 1];
      } else {
        curY = applic.find(function(k) { return k !== curX; }) || '';
      }
    }
    xSel.innerHTML = _amGraphMetricOptions(curX);
    ySel.innerHTML = _amGraphMetricOptions(curY);
    zSel.innerHTML = '<option value="">None</option>' + _amGraphMetricOptions('');
    xSel.value = curX; ySel.value = curY; zSel.value = curZ;
    // Initialise the min-vol control for the chosen X metric.
    _amGraphLastXk = curX;
    _amGraphMinVol = defaultVol(curX);
    _amUpdateGraphVolCtrl(curX);
    // Default graph theme to the site theme each open; preload both logos.
    _amGraphTheme = (document.documentElement.getAttribute('data-theme') === 'dark') ? 'dark' : 'light';
    _amSyncGraphThemeBtn();
    _amLoadLogo('light'); _amLoadLogo('dark');
    if (note) {
      const bits = [];
      bits.push(state.season || (cfg.seasons && cfg.seasons[0]) || '');
      bits.push(_amGraphPos ? _amGraphPos : 'All positions');
      const wr = resolveWeekRange();
      if (wr.ws && wr.we) bits.push('W' + wr.ws + '–W' + wr.we);
      note.textContent = bits.filter(Boolean).join(' · ');
    }
    modal.style.display = 'flex';
    // On mobile, start with controls collapsed so the chart is immediately visible.
    const isMobileView = window.matchMedia('(max-width:600px)').matches;
    const ctrl = modal.querySelector('.am-graph-controls');
    const pos = document.getElementById('amGraphPosBar');
    const chev = document.getElementById('amGraphCtrlChev');
    if (isMobileView && ctrl) {
      ctrl.classList.add('am-ctrl-hidden');
      if (pos) pos.classList.add('am-ctrl-hidden');
      if (chev) chev.textContent = '▼';
    } else {
      if (ctrl) ctrl.classList.remove('am-ctrl-hidden');
      if (pos) pos.classList.remove('am-ctrl-hidden');
      if (chev) chev.textContent = '▲';
    }
    window.amRenderGraph();
  };
  window.amToggleGraphControls = function() {
    const modal = document.getElementById('amGraphModal');
    if (!modal) return;
    const ctrl = modal.querySelector('.am-graph-controls');
    const pos = document.getElementById('amGraphPosBar');
    const chev = document.getElementById('amGraphCtrlChev');
    if (!ctrl) return;
    const nowHidden = ctrl.classList.toggle('am-ctrl-hidden');
    if (pos) pos.classList.toggle('am-ctrl-hidden', nowHidden);
    if (chev) chev.textContent = nowHidden ? '▼' : '▲';
  };
  window.amRenderGraph = function() {
    const plot = document.getElementById('amGraphPlot');
    const xSel = document.getElementById('amGraphX');
    const ySel = document.getElementById('amGraphY');
    const zSel = document.getElementById('amGraphZ');
    const nSel = document.getElementById('amGraphTopN');
    if (!plot || !xSel || !ySel) return;
    const xk = xSel.value, yk = ySel.value, zk = zSel ? zSel.value : '';
    const topN = nSel ? (parseInt(nSel.value, 10) || 25) : 25;
    if (!xk || !yk) { plot.innerHTML = '<div class="am-graph-empty">Pick an X and Y metric.</div>'; return; }
    // When X changes reset min-vol to the new metric's default and refresh the control.
    if (xk !== _amGraphLastXk) {
      _amGraphLastXk = xk;
      _amGraphMinVol = defaultVol(xk);
      _amUpdateGraphVolCtrl(xk);
    }
    _amSyncLbCache();
    const token = ++_amGraphToken;
    const uniq = [xk, yk]; if (zk && uniq.indexOf(zk) === -1) uniq.push(zk);
    const urls = uniq.map(function(k) { return '/api/advanced-metrics/leaderboard?' + _amGraphParams(k, k === xk); });
    // If every needed series is already cached, render synchronously (no spinner
    // flash) so reopening / toggling axes is effectively instant.
    const allCached = urls.every(function(u) { return _amLbCache.has(u); });
    if (!allCached) {
      plot.innerHTML = '<div class="skeleton" style="width:100%;height:340px;border-radius:14px;"></div>';
    }
    Promise.all(urls.map(_amFetchLeaderboard)).then(function(results) {
      if (token !== _amGraphToken) return;
      const byKey = {};
      uniq.forEach(function(k, i) { byKey[k] = (results[i] && results[i].players) || []; });
      const xRows = byKey[xk];
      const yMap = new Map(byKey[yk].map(function(r) { return [amRowKey(r), Number(r.value)]; }));
      const zMap = zk ? new Map((byKey[zk] || []).map(function(r) { return [amRowKey(r), Number(r.value)]; })) : null;
      let ptsAll = [];
      xRows.forEach(function(rx) {
        const pid = amRowKey(rx);
        if (!yMap.has(pid)) return;
        const xv = Number(rx.value), yv = yMap.get(pid);
        if (!isFinite(xv) || !isFinite(yv)) return;
        const nm = (amIsEachYear() && rx.season != null)
          ? ((rx.name || '') + ' · ' + rx.season)
          : (rx.name || '');
        ptsAll.push({ pid: pid, name: nm, position: rx.position, headshot: rx.headshot || '',
                      x: xv, y: yv, z: (zMap && zMap.has(pid)) ? zMap.get(pid) : null });
      });
      const xLower = cfg.metrics[xk] && cfg.metrics[xk].lowerBetter;
      ptsAll.sort(function(a, b) { return xLower ? a.x - b.x : b.x - a.x; });
      // Least-squares trend fit over the FULL eligible population, so the line
      // is stable regardless of the TopN display setting.
      let trend = null;
      if (ptsAll.length >= 3) {
        const n = ptsAll.length;
        let sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
        ptsAll.forEach(function(p) { sx += p.x; sy += p.y; sxx += p.x * p.x; sxy += p.x * p.y; syy += p.y * p.y; });
        const den = n * sxx - sx * sx;
        if (Math.abs(den) > 1e-9) {
          const m = (n * sxy - sx * sy) / den;
          const b = (sy - m * sx) / n;
          const denR = (n * sxx - sx * sx) * (n * syy - sy * sy);
          const r = denR > 0 ? (n * sxy - sx * sy) / Math.sqrt(denR) : 0;
          trend = { m: m, b: b, r2: r * r };
        }
      }
      const pts = ptsAll.slice(0, topN);
      if (!pts.length) {
        window.brEmptyState(plot, { icon: 'search', title: 'Nothing to plot', message: 'No players have data for both metrics with the current filters.' });
        return;
      }
      // Preload headshots so they're browser-cached by first hover.
      pts.forEach(function(p) {
        if (p.headshot && !_amHsPreload[p.headshot]) {
          const _pi = new Image(); _pi.src = p.headshot;
          _amHsPreload[p.headshot] = _pi;
        }
      });
      _amPinnedDot = null; _amHideHoverForce();
      _amLoadLogo(_amGraphTheme).then(function(logo) {
        if (token !== _amGraphToken) return;
        plot.innerHTML = _amBuildScatter(pts, xk, yk, zk, logo, trend);
        const tipEl = document.getElementById('amGraphTip');
        if (tipEl) tipEl.innerHTML = '<span style="opacity:.6">Hover or tap a point for player details</span>';
        // Signal the social-preview renderer that the graph is fully drawn.
        if (_initParams.get('og') === '1') {
          document.documentElement.setAttribute('data-og-ready', '1');
        }
      });
    }).catch(function() {
      if (token === _amGraphToken) window.brErrorState(plot, 'Could not load graph data.', window.amRenderGraph);
    });
  };
  // Builds a fully self-contained SVG (explicit theme colors, embedded logo
  // watermark, in-SVG title/legend) so it renders identically on screen and when
  // rasterized to a shareable PNG. `logo` is a data: URI (may be empty).
  // `trend` is a least-squares fit {m, b, r2} over the full population, so the
  // line reflects the whole position group, not just the visible TopN.
  function _amBuildScatter(pts, xk, yk, zk, logo, trend) {
    const TH = _amGraphPalette();
    const FONT = "'Archivo',system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif";
    // Responsive layout: a portrait viewBox with larger relative fonts on phones
    // (smaller viewBox width => each unit scales up on screen, so text stays legible).
    const isNarrow = (window.innerWidth || 800) <= 600;
    const L = isNarrow
      ? { W: 392, H: 540, padL: 46, padR: 16, padT: 92, padB: 96, fTitle: 16, fSub: 10.5, fTick: 11, fAxis: 10, fLeg: 10.5, ntX: 4, ntY: 5, dotMin: 5.5, dotMax: 13, fStar: 11.5, fLbl: 10 }
      : { W: 720, H: 600, padL: 60, padR: 24, padT: 100, padB: 96, fTitle: 19, fSub: 11.5, fTick: 10.5, fAxis: 10.5, fLeg: 10.5, ntX: 5, ntY: 6, dotMin: 4, dotMax: 15, fStar: 11, fLbl: 9 };
    const W = L.W, H = L.H, padL = L.padL, padR = L.padR, padT = L.padT, padB = L.padB;
    const xs = pts.map(function(p) { return p.x; });
    const ys = pts.map(function(p) { return p.y; });
    // Round tick domains: snap the axis to a nice step so ticks read "9, 12, 15"
    // instead of the raw data min/max.
    function niceStep(span, target) {
      const raw = span / Math.max(1, target);
      const mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
      const cands = [1, 2, 2.5, 5, 10];
      for (let i = 0; i < cands.length; i++) { if (mag * cands[i] >= raw - 1e-9) return mag * cands[i]; }
      return mag * 10;
    }
    function niceDomain(vmin, vmax, target) {
      if (vmin === vmax) { vmin -= 1; vmax += 1; }
      const step = niceStep(vmax - vmin, target);
      const lo = Math.floor(vmin / step) * step;
      const hi = Math.ceil(vmax / step) * step;
      return { lo: lo, hi: hi, step: step };
    }
    const dx = niceDomain(Math.min.apply(null, xs), Math.max.apply(null, xs), L.ntX);
    const dy = niceDomain(Math.min.apply(null, ys), Math.max.apply(null, ys), L.ntY);
    // Zoom the view out a touch beyond the nice tick domain so the data cloud
    // never hugs the frame -- the crammed upper-right cluster (and its stacked
    // labels) was the readability problem. Pad the *positioning* domain by a
    // fraction of the data span; ticks stay on the same nice values, they just
    // sit inside a roomier frame. min()/max() guarantee we only ever zoom out
    // relative to the old snapped domain, never in.
    const _gpad = 0.15;
    const _xlo = Math.min.apply(null, xs), _xhi = Math.max.apply(null, xs);
    const _ylo = Math.min.apply(null, ys), _yhi = Math.max.apply(null, ys);
    const _xp = ((_xhi - _xlo) || dx.step) * _gpad;
    const _yp = ((_yhi - _ylo) || dy.step) * _gpad;
    const xmin = Math.min(dx.lo, _xlo - _xp), xmax = Math.max(dx.hi, _xhi + _xp);
    const ymin = Math.min(dy.lo, _ylo - _yp), ymax = Math.max(dy.hi, _yhi + _yp);
    const px = function(v) { return padL + ((v - xmin) / (xmax - xmin)) * (W - padL - padR); };
    const py = function(v) { return H - padB - ((v - ymin) / (ymax - ymin)) * (H - padT - padB); };
    let rOf = function() { return isNarrow ? 6 : 5; };
    let zmin = null, zmax = null;
    if (zk) {
      const zs = pts.map(function(p) { return p.z; }).filter(function(v) { return v != null && isFinite(v); });
      if (zs.length) {
        zmin = Math.min.apply(null, zs); zmax = Math.max.apply(null, zs);
        const span = (zmax - zmin) || 1;
        rOf = function(p) { return (p.z == null || !isFinite(p.z)) ? (L.dotMin * 0.7) : (L.dotMin + ((p.z - zmin) / span) * (L.dotMax - L.dotMin)); };
      }
    }
    const fmtX = function(v) { return fmtVal(v, xk); };
    const fmtY = function(v) { return fmtVal(v, yk); };
    const fmtTick = function(v, step) { return (step % 1 === 0) ? String(Math.round(v)) : v.toFixed(step * 10 % 1 === 0 ? 1 : 2); };
    const txt = function(x, y, str, size, fill, weight, anchor, transform, ls) {
      return '<text x="' + x + '" y="' + y + '"'
        + (anchor ? ' text-anchor="' + anchor + '"' : '')
        + (transform ? ' transform="' + transform + '"' : '')
        + (ls ? ' letter-spacing="' + ls + '"' : '')
        + ' font-size="' + size + '" font-weight="' + (weight || 400)
        + '" fill="' + fill + '">' + str + '</text>';
    };
    const accent = posColor((pts[0] && pts[0].position) || 'WR');
    const chipBg = TH.dark ? '#1b2740' : '#f2f5f9';
    const chipBorder = TH.dark ? 'rgba(148,163,184,.32)' : '#dbe2ea';
    let s = '<svg class="am-graph-svg" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"'
      + ' viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet" style="font-family:' + FONT + ';">';
    // Themed panel background.
    s += '<rect x="0.5" y="0.5" width="' + (W - 1) + '" height="' + (H - 1) + '" rx="14" fill="' + TH.bg + '" stroke="' + TH.border + '"/>';
    // Editorial header: accent kicker rule + label, left-aligned title + context.
    s += '<rect x="' + padL + '" y="24" width="20" height="3.5" rx="1.75" fill="' + accent + '"/>';
    s += txt(padL + 28, 30, 'ADVANCED METRICS', L.fSub - 1, accent, 800, 'start', null, 2.2);
    s += txt(padL, 56, _amEsc(cfg.metrics[yk].label) + ' vs ' + _amEsc(cfg.metrics[xk].label), L.fTitle, TH.text, 800, 'start');
    const noteEl = document.getElementById('amGraphCtxNote');
    const ctx = noteEl ? noteEl.textContent : '';
    const sub = [ctx, (zk ? 'bubble = ' + cfg.metrics[zk].label : '')].filter(Boolean).join(' · ');
    if (sub) s += txt(padL, 76, _amEsc(sub), L.fSub, TH.muted, 500, 'start');
    // Grid: hairlines at round ticks; a single baseline instead of a hard frame.
    for (let xv = Math.ceil(xmin / dx.step - 1e-9) * dx.step; xv <= xmax + 1e-9; xv += dx.step) {
      const xx = px(xv);
      s += '<line x1="' + xx.toFixed(1) + '" y1="' + padT + '" x2="' + xx.toFixed(1) + '" y2="' + (H - padB) + '" stroke="' + TH.grid + '"/>';
      s += txt(xx.toFixed(1), (H - padB + 16), fmtTick(xv, dx.step), L.fTick, TH.muted, 600, 'middle');
    }
    for (let yv = Math.ceil(ymin / dy.step - 1e-9) * dy.step; yv <= ymax + 1e-9; yv += dy.step) {
      const yy = py(yv);
      s += '<line x1="' + padL + '" y1="' + yy.toFixed(1) + '" x2="' + (W - padR) + '" y2="' + yy.toFixed(1) + '" stroke="' + TH.grid + '"/>';
      s += txt((padL - 8), (yy + 3.5).toFixed(1), fmtTick(yv, dy.step), L.fTick, TH.muted, 600, 'end');
    }
    s += '<line x1="' + padL + '" y1="' + (H - padB) + '" x2="' + (W - padR) + '" y2="' + (H - padB) + '" stroke="' + TH.axis + '" stroke-width="1.2"/>';
    // Axis titles: small caps, letter-spaced.
    s += txt(((padL + W - padR) / 2).toFixed(1), (H - padB + 36), _amEsc(cfg.metrics[xk].label).toUpperCase(), L.fAxis, TH.text, 800, 'middle', null, 1.8);
    s += txt(0, 0, _amEsc(cfg.metrics[yk].label).toUpperCase(), L.fAxis, TH.text, 800, 'middle',
      'translate(13,' + ((padT + H - padB) / 2).toFixed(1) + ') rotate(-90)', 1.8);
    const avgCol = TH.dark ? 'rgba(148,163,184,.55)' : '#aab4c2';
    const chip = function(x, y, t, anchor) {
      const w = t.length * (L.fLeg - 3.2) * 0.62 + 16;
      const bx = anchor === 'end' ? x - w : x;
      return '<rect x="' + bx.toFixed(1) + '" y="' + (y - 11) + '" width="' + w.toFixed(1) + '" height="16" rx="8" fill="' + chipBg + '" stroke="' + chipBorder + '"/>'
        + txt((bx + w / 2).toFixed(1), (y + 1).toFixed(1), t, L.fLeg - 2, TH.muted, 700, 'middle');
    };
    // Obstacle boxes labels must dodge (the trend chip is registered here so
    // names never sit on top of it); Pass 2 appends every placed label too.
    const placed = [];
    const chipW = function(t) { return t.length * (L.fLeg - 3.2) * 0.62 + 16; };
    // Trend line: the population's least-squares fit, clipped to the plot area,
    // tagged with a chip carrying R^2 so the strength of the fit is visible.
    if (trend && isFinite(trend.m) && isFinite(trend.b)) {
      const yAt = function(x) { return trend.m * x + trend.b; };
      let x1 = xmin, x2 = xmax;
      if (Math.abs(trend.m) > 1e-12) {
        // Intersect the line with the y-range so a steep fit doesn't overshoot.
        const cand = [(ymin - trend.b) / trend.m, (ymax - trend.b) / trend.m].sort(function(a, b) { return a - b; });
        x1 = Math.max(x1, cand[0]); x2 = Math.min(x2, cand[1]);
      } else if (trend.b < ymin || trend.b > ymax) {
        x2 = x1 - 1;   // flat line outside the view: skip
      }
      if (x2 > x1) {
        const tx1 = px(x1), ty1 = py(yAt(x1)), tx2 = px(x2), ty2 = py(yAt(x2));
        s += '<line x1="' + tx1.toFixed(1) + '" y1="' + ty1.toFixed(1) + '" x2="' + tx2.toFixed(1) + '" y2="' + ty2.toFixed(1)
          + '" stroke="' + accent + '" stroke-width="1.6" stroke-dasharray="7 5" opacity="0.55" stroke-linecap="round"/>';
        // Chip near the line's right end, nudged clear of the plot edge.
        const chipTxt = 'TREND · R² ' + (Math.round(trend.r2 * 100) / 100).toFixed(2);
        const cy2 = Math.min(Math.max(ty2 + (ty2 < (padT + H - padB) / 2 ? 22 : -16), padT + 14), H - padB - 10);
        const cx2 = Math.min(tx2, W - padR - 4);
        s += chip(cx2, cy2, chipTxt, 'end');
        const cw = chipW(chipTxt);
        placed.push({ x1: cx2 - cw - 2, y1: cy2 - 12, x2: cx2 + 2, y2: cy2 + 6 });
      }
    }
    // Emphasis: the top-ranked players carry the story -- full-strength dots and
    // bold "Name 23.6" labels; the rest of the pool becomes a muted field with a
    // few quiet labels. pts is pre-sorted best-first, so index = rank.
    // Phones can't fit as many labels as the wide desktop chart, so cap them
    // tighter there -- a crammed stack of overlapping names reads as noise. Every
    // dot stays tappable for its full detail card, so fewer printed labels loses
    // no information, it just keeps the ones that print legible.
    const starCut = Math.min(isNarrow ? 6 : 8, Math.max(4, Math.round(pts.length * 0.3)));
    const showCut = Math.min(pts.length, starCut + (isNarrow ? 3 : (pts.length > 50 ? 14 : 10)));
    const lblSize = L.fLbl;
    const ptData = pts.map(function(p, idx) {
      const cx = px(p.x), cy = py(p.y), r = rOf(p), col = posColor(p.position);
      const stats = [[cfg.metrics[xk].label, fmtX(p.x)], [cfg.metrics[yk].label, fmtY(p.y)]];
      if (zk) stats.push([cfg.metrics[zk].label, p.z != null ? fmtVal(p.z, zk) : '–']);
      const info = { nm: p.name, pos: p.position || '', hs: p.headshot || '', stats: stats };
      return { p: p, idx: idx, cx: cx, cy: cy, r: r, col: col, info: info, star: idx < starCut };
    });
    // Pass 1 -- field dots first (muted), then stars on top, largest-first within
    // each group so small dots stay clickable. The field is kept quieter than the
    // stars but visible enough to read the shape of the distribution -- at the old
    // 0.24 the non-highlighted dots all but vanished on a bright phone screen.
    const fieldOp = TH.dark ? 0.42 : 0.36;
    ptData.filter(function(d) { return !d.star; }).sort(function(a, b) { return b.r - a.r; }).forEach(function(d) {
      s += '<circle class="am-graph-dot" cx="' + d.cx.toFixed(1) + '" cy="' + d.cy.toFixed(1) + '" r="' + d.r.toFixed(1)
        + '" fill="' + d.col + '" fill-opacity="' + fieldOp + '" stroke="' + TH.bg + '" stroke-width="1.4"'
        + ' data-info="' + _amEsc(JSON.stringify(d.info)) + '"></circle>';
    });
    ptData.filter(function(d) { return d.star; }).sort(function(a, b) { return b.r - a.r; }).forEach(function(d) {
      s += '<circle class="am-graph-dot" cx="' + d.cx.toFixed(1) + '" cy="' + d.cy.toFixed(1) + '" r="' + d.r.toFixed(1)
        + '" fill="' + d.col + '" stroke="' + TH.bg + '" stroke-width="2"'
        + ' data-info="' + _amEsc(JSON.stringify(d.info)) + '"></circle>';
    });
    // Pass 2 -- labels with collision avoidance; higher-ranked names claim space
    // first. Star labels are bold ink with the ranked value in the accent; the
    // rest are quiet and dropped when they'd collide.
    const lblH = lblSize + 2;
    // Intersection area of two boxes (0 when clear) -- lets a crowded star pick
    // the least-bad anchor instead of stacking on a neighbour.
    const ovArea = function(a, b) {
      const ix = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
      const iy = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
      return (ix > 0 && iy > 0) ? ix * iy : 0;
    };
    ptData.forEach(function(d) {
      if (d.idx >= showCut) return;
      const isStar = d.star;
      const nm = _amLastName(d.p.name);
      const valTxt = isStar ? ' ' + fmtX(d.p.x) : '';
      const fs = isStar ? L.fStar : lblSize;
      const tw = (nm.length + valTxt.length) * fs * 0.56;
      const gap = d.r + 4, vy = d.cy + 3.5;
      const up1 = d.cy - d.r - 4, dn1 = d.cy + d.r + lblH;
      // Eight anchors around the dot, near ones first, farther stacked ones last
      // so a tight cluster spreads vertically instead of piling up.
      const cands = [
        { x: d.cx + gap, y: vy, anchor: 'start' },
        { x: d.cx - gap, y: vy, anchor: 'end' },
        { x: d.cx + gap, y: up1, anchor: 'start' },
        { x: d.cx - gap, y: up1, anchor: 'end' },
        { x: d.cx + gap, y: dn1, anchor: 'start' },
        { x: d.cx - gap, y: dn1, anchor: 'end' },
        { x: d.cx + gap, y: up1 - lblH, anchor: 'start' },
        { x: d.cx - gap, y: up1 - lblH, anchor: 'end' },
        { x: d.cx + gap, y: dn1 + lblH, anchor: 'start' },
        { x: d.cx - gap, y: dn1 + lblH, anchor: 'end' }
      ];
      let chosen = null, chosenBox = null, best = null, bestBox = null, bestScore = Infinity;
      for (let ci = 0; ci < cands.length; ci++) {
        const c = cands[ci];
        const x1 = c.anchor === 'end' ? c.x - tw : c.x;
        const x2 = x1 + tw;
        if (x1 < padL - 2 || x2 > W - padR + 2) continue;
        if (c.y - lblH < padT || c.y + 3 > H - padB) continue;
        const box = { x1: x1 - 1, y1: c.y - lblH, x2: x2 + 1, y2: c.y + 3 };
        let score = 0;
        for (let pi = 0; pi < placed.length; pi++) score += ovArea(box, placed[pi]);
        if (score === 0) { chosen = c; chosenBox = box; break; }
        if (score < bestScore) { bestScore = score; best = c; bestBox = box; }
      }
      if (chosen) {
        placed.push(chosenBox);
      } else {
        // No clear slot: stars keep their name at the least-crowded anchor;
        // quiet field labels bow out rather than smear the plot.
        if (!isStar || !best) return;
        chosen = best; placed.push(bestBox);
      }
      const lblFill = isStar ? TH.text : TH.muted;
      // Leader line: when a crowded cluster pushed this name to a stacked anchor
      // away from its dot, draw a thin connector so you can tell which label
      // belongs to which point. Near labels (touching their dot) skip it -- the
      // line would just be noise. Runs dot-edge → just short of the label, tinted
      // to the point colour for stars so the pairing reads at a glance.
      const _lx = chosen.x, _ly = chosen.y - fs * 0.32;
      const _ldx = _lx - d.cx, _ldy = _ly - d.cy;
      const _ldist = Math.sqrt(_ldx * _ldx + _ldy * _ldy);
      if (_ldist > d.r + 7) {
        const _ux = _ldx / _ldist, _uy = _ldy / _ldist;
        s += '<line x1="' + (d.cx + _ux * d.r).toFixed(1) + '" y1="' + (d.cy + _uy * d.r).toFixed(1)
          + '" x2="' + (_lx - _ux * 2).toFixed(1) + '" y2="' + (_ly - _uy * 2).toFixed(1)
          + '" stroke="' + (isStar ? d.col : TH.muted) + '" stroke-width="1"'
          + ' opacity="' + (isStar ? 0.5 : 0.32) + '" stroke-linecap="round"/>';
      }
      s += '<text x="' + chosen.x.toFixed(1) + '" y="' + chosen.y.toFixed(1) + '"'
        + (chosen.anchor === 'end' ? ' text-anchor="end"' : '')
        + ' font-size="' + fs + '" font-weight="' + (isStar ? 800 : 500) + '"'
        + ' fill="' + lblFill + '"'
        + ' stroke="' + TH.bg + '" stroke-width="3" stroke-linejoin="round" paint-order="stroke fill"'
        + '>' + _amEsc(nm)
        + (isStar ? '<tspan font-weight="700" fill="' + d.col + '">' + _amEsc(valTxt) + '</tspan>' : '')
        + '</text>';
    });
    // Footer row: position chips (when mixed) + real bubble-size reference
    // circles + the brand bug, replacing the giant center watermark.
    const posPresent = [];
    pts.forEach(function(p) { if (p.position && posPresent.indexOf(p.position) === -1) posPresent.push(p.position); });
    const footY = H - 22;
    let fx = padL;
    if (posPresent.length > 1) {
      posPresent.forEach(function(pp) {
        s += '<circle cx="' + (fx + 4.5) + '" cy="' + (footY - 3.5) + '" r="4.5" fill="' + posColor(pp) + '"/>';
        s += txt(fx + 13, footY, _amEsc(pp), L.fLeg, TH.muted, 700, 'start');
        fx += 13 + pp.length * L.fLeg * 0.62 + 14;
      });
      fx += 6;
    }
    if (zk && zmin != null && zmax != null && zmax > zmin) {
      s += txt(fx, footY, _amEsc(cfg.metrics[zk].label).toUpperCase(), L.fLeg - 2, TH.muted, 800, 'start', null, 1.6);
      fx += (cfg.metrics[zk].label.length) * (L.fLeg - 2) * 0.78 + 26;
      const refs = isNarrow ? [zmin, zmax] : [zmin, (zmin + zmax) / 2, zmax];
      refs.forEach(function(v) {
        const rr = rOf({ z: v });
        s += '<circle cx="' + (fx + rr).toFixed(1) + '" cy="' + (footY - 4) + '" r="' + rr.toFixed(1) + '" fill="none" stroke="' + avgCol + '" stroke-width="1.4"/>';
        s += txt((fx + rr).toFixed(1), footY + 0.5, fmtTick(Math.round(v * 10) / 10, (zmax - zmin) >= 3 ? 1 : 0.1), L.fLeg - 2.5, TH.muted, 700, 'middle');
        fx += rr * 2 + 16;
      });
    }
    if (logo) {
      const blw = isNarrow ? 30 : 34, blh = blw * 0.887;
      const bx = W - padR - blw;
      if (bx - 14 > fx) {
        s += '<line x1="' + (fx + 8) + '" y1="' + (footY - 4) + '" x2="' + (bx - 12) + '" y2="' + (footY - 4) + '" stroke="' + TH.grid + '"/>';
      }
      s += '<image href="' + logo + '" xlink:href="' + logo + '" x="' + bx + '" y="' + (footY - 4 - blh / 2).toFixed(1)
        + '" width="' + blw + '" height="' + blh.toFixed(1) + '" opacity="0.9" preserveAspectRatio="xMidYMid meet"/>';
    }
    s += '</svg>';
    return s;
  }
  // Build the hover/tap card (headshot + selected stats) from a dot's data-info.
  function _amDotInfo(dot) {
    try { return JSON.parse(dot.getAttribute('data-info') || '{}'); } catch (e) { return null; }
  }
  function _amShowHover(dot, clientX, clientY, pinned) {
    const wrap = document.querySelector('.am-graph-plot-wrap');
    const hover = document.getElementById('amGraphHover');
    if (!wrap || !hover) return;
    const info = _amDotInfo(dot);
    if (!info) return;
    let html = '';
    if (pinned) html += '<button class="am-graph-hover-close" onclick="_amUnpin()" title="Close">✕</button>';
    html += '<div class="am-graph-hover-top">';
    if (info.hs) html += '<img class="am-graph-hover-hs" src="' + _amEsc(info.hs) + '" alt="" onerror="this.style.display=\'none\'"/>';
    html += '<div><div class="am-graph-hover-nm">' + _amEsc(info.nm) + '</div>'
      + '<div class="am-graph-hover-pos">' + _amEsc(info.pos) + '</div></div></div>';
    html += '<div class="am-graph-hover-stats">';
    (info.stats || []).forEach(function(st) {
      html += '<div><span class="k">' + _amEsc(st[0]) + ':</span> <b>' + _amEsc(st[1]) + '</b></div>';
    });
    html += '</div>';
    hover.innerHTML = html;
    hover.classList.toggle('pinned', !!pinned);
    hover.classList.add('show');
    const wr = wrap.getBoundingClientRect();
    let x = clientX - wr.left + 14, y = clientY - wr.top + 14;
    const hw = hover.offsetWidth, hh = hover.offsetHeight;
    if (x + hw > wrap.clientWidth - 4) x = clientX - wr.left - hw - 14;
    if (x < 4) x = 4;
    if (y + hh > wrap.clientHeight - 4) y = Math.max(4, clientY - wr.top - hh - 14);
    hover.style.left = x + 'px';
    hover.style.top = y + 'px';
  }
  function _amHideHover() {
    if (_amPinnedDot) return;
    const hover = document.getElementById('amGraphHover');
    if (hover) { hover.classList.remove('show'); hover.classList.remove('pinned'); }
  }
  function _amHideHoverForce() {
    const hover = document.getElementById('amGraphHover');
    if (hover) { hover.classList.remove('show'); hover.classList.remove('pinned'); }
  }
  window._amUnpin = function() {
    _amPinnedDot = null;
    _amHideHoverForce();
  };
  // One-time wiring: hover (desktop) / tap (touch) a point to show a card with
  // the headshot + selected stats; re-render on resize so layout stays responsive.
  // Also wires the graph-modal position filter buttons.
  (function _amInitGraphInteractions() {
    const graphVolSel = document.getElementById('amGraphMinVolSel');
    if (graphVolSel) {
      graphVolSel.addEventListener('change', function() {
        _amGraphMinVol = this.value;
        window.amRenderGraph();
      });
    }
    const posBar = document.getElementById('amGraphPosBar');
    if (posBar) {
      posBar.addEventListener('click', function(e) {
        const btn = e.target.closest('[data-gpos]');
        if (!btn) return;
        _amGraphPos = btn.getAttribute('data-gpos') || null;
        _amSyncGraphPosBtns();
        // Rebuild metric selectors filtered to the new position, preserving
        // current X/Y selections where valid.
        const xSel = document.getElementById('amGraphX');
        const ySel = document.getElementById('amGraphY');
        const zSel = document.getElementById('amGraphZ');
        const curX = xSel ? xSel.value : '';
        const curY = ySel ? ySel.value : '';
        const curZ = zSel ? zSel.value : '';
        if (xSel) { xSel.innerHTML = _amGraphMetricOptions(curX); xSel.value = curX; }
        if (ySel) { ySel.innerHTML = _amGraphMetricOptions(curY); ySel.value = curY; }
        if (zSel) { zSel.innerHTML = '<option value="">None</option>' + _amGraphMetricOptions(curZ); zSel.value = curZ; }
        // Update context note
        const note = document.getElementById('amGraphCtxNote');
        if (note) {
          const bits = [state.season || '', _amGraphPos || 'All positions'];
          const wr = resolveWeekRange();
          if (wr.ws && wr.we) bits.push('W' + wr.ws + '–W' + wr.we);
          note.textContent = bits.filter(Boolean).join(' · ');
        }
        window.amRenderGraph();
      });
    }
    const plot = document.getElementById('amGraphPlot');
    if (plot) {
      plot.addEventListener('mousemove', function(e) {
        if (_amPinnedDot) return;
        const dot = e.target.closest && e.target.closest('circle[data-info]');
        if (dot) _amShowHover(dot, e.clientX, e.clientY, false); else _amHideHover();
      });
      plot.addEventListener('mouseleave', function() {
        if (_amPinnedDot) return;
        _amHideHover();
      });
      plot.addEventListener('click', function(e) {
        const dot = e.target.closest && e.target.closest('circle[data-info]');
        if (dot) {
          if (_amPinnedDot === dot) {
            _amPinnedDot = null; _amHideHoverForce();
          } else {
            _amPinnedDot = dot; _amShowHover(dot, e.clientX, e.clientY, true);
          }
        } else {
          _amPinnedDot = null; _amHideHoverForce();
        }
      });
    }
    let _rt;
    window.addEventListener('resize', function() {
      const modal = document.getElementById('amGraphModal');
      if (!modal || modal.style.display !== 'flex') return;
      _amPinnedDot = null; _amHideHoverForce();
      clearTimeout(_rt);
      _rt = setTimeout(function() {
        if (modal.style.display === 'flex') window.amRenderGraph();
      }, 250);
    });
  })();
  // Download / share the current graph as a branded PNG.
  window.amDownloadGraph = function() {
    const svg = document.querySelector('#amGraphPlot svg.am-graph-svg');
    if (!svg) return;
    const btn = document.getElementById('amGraphDownloadBtn');
    const flash = function(msg) {
      if (!btn) return;
      const prev = btn.innerHTML;
      btn.innerHTML = msg;
      setTimeout(function() { btn.innerHTML = prev; }, 2500);
    };
    const TH = _amGraphPalette();
    const vb = (svg.getAttribute('viewBox') || '0 0 720 580').split(/\s+/).map(Number);
    const W = vb[2] || 720, H = vb[3] || 580, scale = 2;
    const clone = svg.cloneNode(true);
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    clone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
    clone.setAttribute('width', W); clone.setAttribute('height', H);
    const xml = new XMLSerializer().serializeToString(clone);
    // Encode as base64 data URI (not raw URI-encoded) -- more robust across browsers
    // for loading an <svg> into an <img>, and avoids issues with special chars.
    let src;
    try {
      src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(xml)));
    } catch (_) {
      src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(xml);
    }
    flash('Saving…');
    const xk = document.getElementById('amGraphX').value;
    const yk = document.getElementById('amGraphY').value;
    const fname = 'br-metrics-' + xk + '-vs-' + yk + '.png';
    const img = new Image();
    img.onload = function() {
      let canvas, blobMaker;
      try {
        canvas = document.createElement('canvas');
        canvas.width = W * scale; canvas.height = H * scale;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = TH.bg; ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      } catch (e) { flash('Failed'); return; }
      // Wrap toBlob in case the canvas is tainted (throws SecurityError).
      blobMaker = function(cb) {
        try { canvas.toBlob(cb, 'image/png'); }
        catch (e) { cb(null); }
      };
      blobMaker(function(blob) {
        if (!blob) {
          // Last-resort fallback: data-URL anchor download.
          try {
            const a = document.createElement('a');
            a.href = canvas.toDataURL('image/png'); a.download = fname;
            document.body.appendChild(a); a.click();
            setTimeout(function() { document.body.removeChild(a); }, 100);
            flash('Saved ✓');
          } catch (e2) { flash('Failed'); }
          return;
        }
        // Prefer the native share sheet on mobile (Save to Photos, Messages, …).
        let file = null;
        try { file = new File([blob], fname, { type: 'image/png' }); } catch (_) {}
        if (file && navigator.canShare && navigator.canShare({ files: [file] })) {
          navigator.share({ files: [file], title: 'BR Fantasy Metrics' })
            .then(function() { flash('Shared ✓'); })
            .catch(function(err) {
              if (err && err.name === 'AbortError') { flash('Download'); return; }
              _amAnchorDownload(blob, fname, flash);
            });
          return;
        }
        _amAnchorDownload(blob, fname, flash);
      });
    };
    img.onerror = function() { flash('Failed'); };
    img.src = src;
  };
  // Trigger a real file download from a Blob via an <a download> click.
  function _amAnchorDownload(blob, fname, flash) {
    try {
      const objUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = objUrl; a.download = fname;
      a.rel = 'noopener';
      a.style.position = 'fixed'; a.style.left = '-9999px';
      document.body.appendChild(a); a.click();
      setTimeout(function() {
        document.body.removeChild(a); URL.revokeObjectURL(objUrl);
      }, 1000);
      flash('Saved ✓');
    } catch (e) { flash('Failed'); }
  }
  // Copy a shareable link that will reopen this exact graph configuration.
  window.amCopyGraphLink = function() {
    const btn = document.getElementById('amGraphCopyBtn');
    const xk = (document.getElementById('amGraphX') || {}).value || '';
    const yk = (document.getElementById('amGraphY') || {}).value || '';
    const zEl = document.getElementById('amGraphZ');
    const zk = zEl ? zEl.value : '';
    const nEl = document.getElementById('amGraphTopN');
    const topN = nEl ? nEl.value : '25';
    const p = new URLSearchParams(window.location.search);
    p.set('graph', '1');
    if (xk) p.set('gx', xk); else p.delete('gx');
    if (yk) p.set('gy', yk); else p.delete('gy');
    if (zk) p.set('gz', zk); else p.delete('gz');
    p.set('gn', topN);
    const url = window.location.origin + window.location.pathname + '?' + p.toString();
    const flash = function(msg) {
      if (!btn) return;
      const prev = btn.innerHTML;
      btn.innerHTML = msg;
      setTimeout(function() { btn.innerHTML = prev; }, 2000);
    };
    navigator.clipboard.writeText(url).then(function() {
      flash('✓ Copied!');
    }).catch(function() {
      try {
        const ta = document.createElement('textarea');
        ta.value = url; ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.select(); document.execCommand('copy');
        document.body.removeChild(ta);
        flash('✓ Copied!');
      } catch(_) { flash('Failed'); }
    });
  };

  function fetchData() {
    // paywall removed -- advanced metrics is available to all users
    state.fetching = true;
    const requestToken = ++state.requestToken;
    if (loading) loading.style.display = ''; empty.style.display = 'none'; syncTableHeader(); tbody.innerHTML = schemaSkeletonRows();
    if (avgNote) avgNote.style.display = 'none';
    const params = new URLSearchParams({ metric: state.metric, platform: cfg.platform });
    if (cfg.leagueId) params.set('league_id', cfg.leagueId);
    if (state.season) params.set('season', state.season);
    if (state.combine && amIsMultiSeason()) params.set('combine', '1');
    if (state.minVol) params.set('min_vol', state.minVol);

    // Add week range params when a filter is active.
    const weekCapable = cfg.metrics[state.metric] && cfg.metrics[state.metric].weeklyCapable;
    const { ws, we } = resolveWeekRange();
    if (ws) params.set('week_start', String(ws));
    if (we) params.set('week_end',   String(we));

    // Determine the previous season to fetch for YoY trend arrows.
    // Skip YoY when a week range is active (not meaningful cross-season).
    const curSeason = amSelectedSeasons()[0] || (cfg.seasons && cfg.seasons[0]);
    const prevSeason = (!amIsMultiSeason() && curSeason) ? curSeason - 1 : null;
    const isWeekFiltered = !!(ws || we);
    const hasPrevInData = !amIsMultiSeason() && !isWeekFiltered && prevSeason && cfg.seasons && cfg.seasons.includes(prevSeason);
    const prevParams = new URLSearchParams({ metric: state.metric, platform: cfg.platform });
    if (cfg.leagueId) prevParams.set('league_id', cfg.leagueId);
    if (prevSeason) prevParams.set('season', String(prevSeason));
    if (state.minVol) prevParams.set('min_vol', state.minVol);

    // Hard timeout so a hung primary request shows the Retry (see .catch) rather
    // than spinning the whole table forever.
    const _mainCtl = (typeof AbortController !== 'undefined') ? new AbortController() : null;
    const _mainTo = _mainCtl ? setTimeout(function() { _mainCtl.abort(); }, 15000) : null;
    const mainFetch = fetch('/api/advanced-metrics/leaderboard?' + params, _mainCtl ? { signal: _mainCtl.signal } : undefined)
      .then(r => {
        if (_mainTo) clearTimeout(_mainTo);
        if (r.status === 403) return r.json().then(
          j => ({ _proOnly: !j || j.error === 'pro_only' }),
          () => ({ _proOnly: true }));
        return r.json();
      });
    const prevFetch = hasPrevInData
      ? _amCmpFetch('/api/advanced-metrics/leaderboard?' + prevParams, 15000)
      : Promise.resolve(null);

    Promise.all([mainFetch, prevFetch])
      .then(([d, pd]) => {
        if (requestToken !== state.requestToken) return;
        if (d && d._proOnly) {
          // PRO metric requested without PRO: open the paywall and fall back
          // to a free metric so the table is never stuck on an empty state.
          state.fetching = false; if (loading) loading.style.display = 'none';
          if (typeof window.showPaywall === 'function') window.showPaywall('advanced-metrics-' + state.metric);
          state.metric = 'opportunity_share';
          if (metricSel) metricSel.value = state.metric;
          state.page = 0; syncURL(); fetchData();
          return;
        }
        if (!d) { state.fetching = false; empty.style.display = ''; if (loading) loading.style.display = 'none'; tbody.innerHTML = ''; if (tableWrap) tableWrap.style.display = 'none'; return; }
        state.fetching = false;
        state.rows = d.players || [];
        state.volCol = d.vol_col || 'games';
        // Position lookup for positional ranks/bounds in the Compare modal.
        state.rows.forEach(r => {
          if (r.player_id != null && r.position) {
            state.playerPos[String(r.player_id)] = String(r.position).toUpperCase();
          }
        });
        // Build previous-season lookup for trend arrows.
        if (pd && pd.players) {
          state.prevData = Object.fromEntries(pd.players.map(r => [String(r.player_id), Number(r.value)]));
        } else {
          state.prevData = {};
        }
        state.responseWeekFiltered = !!d.is_week_filtered;
        updateWeekNote(d.is_week_filtered, weekCapable);
        updateVolHeader();
        populateTeamFilter();
        showAgeCtrl();
        // Re-fetch all extra metrics and filter col metrics (season / filter may have changed).
        state.extraData = {};
        state.extraPrevData = {};
        _loadExtras([...state.extraMetrics, ...(state.filterColKeys ? [...state.filterColKeys] : [])]);
        render();
      })
      .catch(() => {
        if (requestToken !== state.requestToken) return;
        state.fetching = false; if (loading) loading.style.display = 'none';
        // Network error (e.g. ERR_NETWORK_CHANGED): show a recoverable retry
        // rather than the misleading "No data for this metric yet." message.
        empty.style.display = ''; tbody.innerHTML = ''; if (tableWrap) tableWrap.style.display = 'none';
        empty.innerHTML = 'Couldn’t load this metric, network hiccup. '
          + '<button type="button" id="amRetryBtn" style="margin-left:6px;padding:5px 12px;'
          + 'border:1px solid var(--border);border-radius:8px;background:var(--card);'
          + 'color:var(--accent,#2563eb);cursor:pointer;font-weight:700;">Retry</button>';
        const _rb = document.getElementById('amRetryBtn');
        if (_rb) _rb.addEventListener('click', function() { empty.style.display = 'none'; fetchData(); });
      });
  }

  // Load the viewer's roster so owned players can be highlighted / filtered.
  function loadOwnedRoster() {
    if (!cfg.leagueId) return;
    fetch('/api/league-rosters?league_id=' + encodeURIComponent(cfg.leagueId) + '&platform=' + encodeURIComponent(cfg.platform), { cache: 'no-store' })
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (!d || !d.teams) return;
        const vid = String(d.viewer_roster_id || '');
        const mine = (d.teams || []).find(t => String(t.roster_id) === vid);
        if (mine && mine.player_ids && mine.player_ids.length) {
          ownedIds = new Set(mine.player_ids.map(String));
          if (rosterWrap) rosterWrap.style.display = '';
          render();
        }
      })
      .catch(() => {});
  }

  function updateFilterBar() {
    const bar = document.getElementById('amFilterBar');
    const chips = document.getElementById('amFilterChips');
    const filterKey = document.getElementById('amFilterKey');
    if (!bar || !chips) return;
    if (filterKey) {
      const primaryPositions = new Set(relevantPositions(state.metric));
      const added = new Set(['age', 'exp', 'primary']);

      // Position-aware ordered filter keys: volume counts → per-game rates → usage rates.
      const _FILTER_ORDER = [
        // Passing
        'total_pass_yards',
        // Rushing
        'total_carries', 'carries_per_game', 'total_rush_yards', 'rush_yards_per_game',
        // Receiving
        'total_targets', 'targets_per_game',
        'total_receptions', 'receptions_per_game',
        'total_rec_yards', 'rec_yards_per_game',
        // Combined
        'total_touches', 'touches_per_game',
        // Routes / usage rates
        'total_routes', 'routes_per_game', 'route_participation',
        'snap_share', 'target_share', 'air_yards_share',
        // Red zone
        'rz_targets_pg', 'rz_carries_pg', 'red_zone_usage',
      ];
      const volOpts = [];
      _FILTER_ORDER.forEach(function(key) {
        const spec = cfg.metrics[key];
        if (!spec) return;
        const mpos = new Set(spec.positions || []);
        const relevant = [...primaryPositions].some(function(p) { return mpos.has(p); });
        if (!relevant) return;
        volOpts.push({ value: key, label: spec.label });
        added.add(key);
      });
      // Order: Age/Exp → volume/rates → primary metric → extra metrics
      const opts = [{ value: 'age', label: 'Age' }, { value: 'exp', label: 'Years Exp' }];
      volOpts.forEach(function(o) { opts.push(o); });
      opts.push({ value: 'primary', label: _mLabel(state.metric) });
      state.extraMetrics.forEach(function(key) {
        if (added.has(key)) return;
        opts.push({ value: key, label: _mLabel(key) });
      });

      filterKey.innerHTML = opts.map(function(o) {
        return '<option value="' + o.value + '">' + o.label + '</option>';
      }).join('');
    }
    chips.innerHTML = state.comboFilters.map(function(f, idx) {
      const lbl = f.key === 'primary'
        ? (_mLabel(state.metric))
        : f.key === 'age' ? 'Age'
        : f.key === 'exp' ? 'Years Exp'
        : (_mLabel(f.key));
      const opSym = f.op === 'gte' ? '≥' : '≤';
      return '<span class="am-filter-chip">' + lbl + ' ' + opSym + ' ' + f.val
        + ' <button class="am-chip-x" onclick="amRemoveFilter(' + idx + ')" aria-label="Remove">\xd7</button></span>';
    }).join('');
    // Show the filter bar only when there's something to show: active chips,
    // the age-input controls, the vol (min games) control, or the filter form.
    const ageVis  = (document.getElementById('amAgeWrap')  || {}).style.display !== 'none';
    const volVis  = (document.getElementById('amGamesCtrl') || {}).style.display !== 'none';
    const formVis = (document.getElementById('amFilterForm') || {}).style.display !== 'none';
    // On mobile the opened Filters panel must show even when empty, so its
    // in-panel + Filter button has somewhere to live.
    const isMobile = window.innerWidth <= 600;
    const mOpen = bar && bar.classList.contains('am-mobile-open') && isMobile;
    if (bar) {
      // Desktop: team + sort live here, so the bar stays open. Mobile: only
      // when Filters is open (or a chip/form is active).
      if (isMobile) {
        bar.style.display = (state.comboFilters.length > 0 || formVis || mOpen) ? 'flex' : 'none';
      } else {
        bar.style.display = 'flex';
      }
    }
  }
  window.amRemoveFilter = function(idx) {
    const removed = state.comboFilters[idx];
    state.comboFilters.splice(idx, 1);
    // Clean up auto-added filter col if no other filter still references it
    if (removed && removed.key !== 'primary' && removed.key !== 'age' && state.filterColKeys && state.filterColKeys.has(removed.key)) {
      const stillUsed = state.comboFilters.some(function(f) { return f.key === removed.key; });
      if (!stillUsed && !state.extraMetrics.includes(removed.key)) {
        state.filterColKeys.delete(removed.key);
        delete state.extraData[removed.key];
        delete state.extraPrevData[removed.key];
        syncFilterCols();
      }
    }
    updateFilterBar();
    state.page = 0;
    render();
  };
  function showAgeCtrl() {
    const wrap = document.getElementById('amAgeWrap');
    if (!wrap) return;
    const hasAge = state.rows.some(r => r.age != null);
    if (!hasAge) { wrap.style.display = 'none'; updateFilterBar(); return; }
    // On mobile, only show the age inputs when the Filters dropdown is open.
    const isMobile = window.innerWidth <= 600;
    if (isMobile) {
      const fb = document.getElementById('amFilterBar');
      wrap.style.display = (fb && fb.classList.contains('am-mobile-open')) ? '' : 'none';
    } else {
      wrap.style.display = '';
    }
    updateFilterBar();
  }

  metricSel.addEventListener('change', () => {
    const _v = metricSel.value;
    if (_v && _v.startsWith('__preset__')) { amLoadPreset(_v.replace('__preset__', '')); return; }
    _showActiveSet('Custom');
    _activePresetId = null;
    _updateDecisionUI(null);
    state.metric = metricSel.value; state.page = 0;
    state.extraMetrics = []; state.extraData = {}; state.extraPrevData = {}; state.prevData = {};
    state.comboFilters = []; state.filterColKeys = new Set();
    const rel = new Set(relevantPositions(state.metric));
    if (state.position !== 'ALL' && !rel.has(state.position)) state.position = 'ALL';
    state.sortDir = (cfg.metrics[state.metric] && cfg.metrics[state.metric].lowerBetter) ? 'asc' : 'desc';
    state.sortBy = state.metric;
    state.minVol = defaultVol(state.metric);
    updateSortBtn(); updatePosButtons(); updateMetricTip(); updateVolCtrl(); updateVolHeader();
    _amRefreshWeekControls();
    updateSortHeaders(); updateCompareBar(); updateFilterBar();
    syncURL(); fetchData();
  });
  posWrap.addEventListener('click', e => {
    const b = e.target.closest('[data-pos]');
    if (!b || b.disabled) return;
    state.position = b.dataset.pos; state.page = 0;
    updatePosButtons(); syncURL(); render();
  });
  searchEl.addEventListener('input', () => { state.search = searchEl.value.trim(); state.page = 0; render(); });
  sortBtn.addEventListener('click', () => {
    state.sortBy = state.metric;
    state.sortDir = state.sortDir === 'desc' ? 'asc' : 'desc'; state.page = 0;
    updateSortBtn(); updateSortHeaders(); render();
  });
  function syncSeasonBtn() {
    const label = document.getElementById('amSeasonBtnLabel');
    if (label) label.textContent = amSeasonLabel(amSelectedSeasons());
  }
  function syncCombineToggle() {
    const wrap = document.getElementById('amCombineToggle');
    if (!wrap) return;
    const multi = amIsMultiSeason();
    wrap.style.display = multi ? '' : 'none';
    wrap.querySelectorAll('[data-combine]').forEach(function(b) {
      const on = (b.getAttribute('data-combine') === '1') === !!state.combine;
      b.classList.toggle('active', on);
    });
  }
  function syncMultiSeasonUI() {
    const multi = amIsMultiSeason();
    const weekCtrl = document.getElementById('amWeekCtrl');
    if (weekCtrl) weekCtrl.style.display = multi ? 'none' : '';
    const seasonHdr = document.getElementById('amSeasonColHdr');
    if (seasonHdr) seasonHdr.style.display = amIsEachYear() ? '' : 'none';
    if (multi) {
      state.weekRange = '';
      state.weekStart = null;
      state.weekEnd = null;
      _amSyncQuickChips('');
    }
    syncCombineToggle();
    syncSeasonBtn();
  }
  function applySeasonSelection(years) {
    const allowed = new Set((cfg.seasons || []).map(Number));
    const next = (years || []).map(Number).filter(function(y) { return allowed.has(y); });
    next.sort(function(a, b) { return b - a; });
    if (!next.length && cfg.seasons && cfg.seasons.length) next.push(Number(cfg.seasons[0]));
    state.season = next.join(',');
    if (seasonSel && next.length) seasonSel.value = String(next[0]);
    syncMultiSeasonUI();
    _amRefreshWeekControls(true);
    state.page = 0;
    syncURL();
    fetchData();
    _loadMovers();
  }
  function fillSeasonMenu() {
    const menu = document.getElementById('amSeasonMenu');
    if (!menu) return;
    const selected = new Set(amSelectedSeasons());
    const years = cfg.seasons || [];
    let html = '<div class="am-season-opt-hint">Pick any years that have data. Then choose Each year (one row per season) or Combine (one row per player).</div>';
    years.forEach(function(yr) {
      const on = selected.has(Number(yr));
      html += '<label class="am-season-opt">'
        + '<input type="checkbox" value="' + yr + '"' + (on ? ' checked' : '') + '>'
        + yr + '</label>';
    });
    menu.innerHTML = html;
    menu.querySelectorAll('input[type="checkbox"]').forEach(function(box) {
      box.addEventListener('change', function() {
        const picked = [];
        menu.querySelectorAll('input[type="checkbox"]:checked').forEach(function(el) {
          picked.push(Number(el.value));
        });
        if (!picked.length) {
          box.checked = true;
          return;
        }
        applySeasonSelection(picked);
      });
    });
  }
  (function initSeasonPicker() {
    const btn = document.getElementById('amSeasonBtn');
    const menu = document.getElementById('amSeasonMenu');
    if (!btn || !menu) return;
    fillSeasonMenu();
    syncSeasonBtn();
    syncMultiSeasonUI();
    btn.addEventListener('click', function(e) {
      e.stopPropagation();
      const open = menu.style.display !== 'none';
      menu.style.display = open ? 'none' : '';
      btn.setAttribute('aria-expanded', open ? 'false' : 'true');
      if (!open) fillSeasonMenu();
    });
    document.addEventListener('click', function(e) {
      if (menu.style.display === 'none') return;
      if (e.target.closest && e.target.closest('#amSeasonMulti')) return;
      menu.style.display = 'none';
      btn.setAttribute('aria-expanded', 'false');
    });
  })();
  if (seasonSel) {
    seasonSel.addEventListener('change', () => { applySeasonSelection([seasonSel.value]); });
  }
  const combineWrap = document.getElementById('amCombineToggle');
  if (combineWrap) {
    combineWrap.addEventListener('click', function(e) {
      const btn = e.target.closest('[data-combine]');
      if (!btn) return;
      const next = btn.getAttribute('data-combine') === '1';
      if (!!state.combine === next) return;
      state.combine = next;
      state.page = 0;
      syncMultiSeasonUI();
      syncURL();
      fetchData();
    });
  }
  if (teamSel) {
    teamSel.addEventListener('change', () => { state.team = teamSel.value || ''; state.page = 0; syncURL(); render(); });
  }
  if (minGamesSel) {
    minGamesSel.addEventListener('change', () => { state.minVol = minGamesSel.value || ''; state.page = 0; syncURL(); fetchData(); });
  }
  // Week-range helpers are provided by static/week_range.js, loaded before this page script.
  const amWkBarHost = document.getElementById('amWkBarHost');
  function _amSyncQuickChips(key) {
    document.querySelectorAll('#amQuickRanges .am-qr').forEach(function(b) {
      b.classList.toggle('active', (b.getAttribute('data-range') || '') === key);
    });
  }
  function _amBuildWkBar(selWs, selWe) {
    if (!amWkBarHost) return;
    const weeks = availableWeeks();
    if (!weeks.length) { amWkBarHost.innerHTML = ''; return; }
    const minW = weeks[0], maxW = weeks[weeks.length - 1];
    amWkBarHost.innerHTML = _wkBarBuild('amWkBar', minW, maxW, selWs, selWe, weeks);
    _wkBarInit('amWkBar', function(ws, we) {
      // A direct interaction is always an explicit week filter, even when the
      // season currently contains only one populated week (minW === maxW).
      // Only the Season quick button clears week_start/week_end.
      state.weekRange = 'custom';
      state.weekStart = ws; state.weekEnd = we;
      state.minVol = ''; if (minGamesSel) minGamesSel.value = '';
      updateVolCtrl(); _amSyncQuickChips('custom');
      state.page = 0; syncURL(); fetchData();
    });
  }
  function _amRefreshWeekControls(resetInvalid) {
    const capable = isMetricWeeklyCapable(), weeks = availableWeeks();
    const controls = document.getElementById('amWeekCtrl');
    if (controls) controls.classList.toggle('am-week-disabled', !capable);
    document.querySelectorAll('#amQuickRanges .am-qr').forEach(function(b){ b.disabled = !capable; });
    if (!capable) { if (amWkBarHost) amWkBarHost.innerHTML = ''; updateWeekNote(false, false); return; }
    const lo=weeks[0], hi=weeks[weeks.length-1];
    if (!weeks.length) return;
    if (resetInvalid && state.weekRange === 'custom' && (state.weekStart < lo || state.weekEnd > hi)) {
      state.weekRange=''; state.weekStart=null; state.weekEnd=null;
    }
    const r=resolveWeekRange(); _amSyncQuickChips(state.weekRange || '');
    _amBuildWkBar(r.ws == null ? lo : r.ws, r.we == null ? hi : r.we);
    updateWeekNote(false, true);
  }
  const initialWs = Number(_initParams.get('week_start'));
  const initialWe = Number(_initParams.get('week_end'));
  if (_initParams.has('week_start') && _initParams.has('week_end') && Number.isFinite(initialWs) && Number.isFinite(initialWe)) {
    state.weekRange='custom'; state.weekStart=Math.min(initialWs,initialWe); state.weekEnd=Math.max(initialWs,initialWe);
  }
  _amRefreshWeekControls(true);
  const quickWrap = document.getElementById('amQuickRanges');
  if (quickWrap) quickWrap.addEventListener('click', function(e) {
    const btn=e.target.closest('.am-qr'); if(!btn || btn.disabled)return;
    const key=btn.getAttribute('data-range')||'';
    state.weekRange=key; state.weekStart=null; state.weekEnd=null; state.minVol='';
    if(minGamesSel)minGamesSel.value=''; updateVolCtrl(); _amSyncQuickChips(key);
    const weeks=availableWeeks(),r=resolveWeekRange();
    _amBuildWkBar(r.ws == null ? weeks[0] : r.ws, r.we == null ? weeks[weeks.length-1] : r.we);
    state.page=0; syncURL(); fetchData();
  });
  // CSV export of the current filtered/sorted view (all pages, not just the
  // visible one). Columns: identity + primary metric + any added metrics.
  const exportBtn = document.getElementById('amExportBtn');
  if (exportBtn) {
    exportBtn.addEventListener('click', function() {
      const rows = state._exportRows || [];
      if (!rows.length) return;
      const metricLbl = _mLabel(state.metric);
      const extraKeys = state.extraMetrics.filter(k => state.extraData[k]);
      const sampleCols = contextColsFor();
      const head = ['Player', 'Team', 'Pos', 'Age', 'Exp', 'Games']
        .concat(sampleCols.map(c => c.title))
        .concat([metricLbl])
        .concat(extraKeys.map(k => _mLabel(k)));
      if (amIsEachYear()) head.splice(1, 0, 'Year');
      const esc = function(v) {
        if (v == null) return '';
        v = String(v);
        return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
      };
      const lines = [head.map(esc).join(',')];
      rows.forEach(function(r) {
        const line = [
          r.name || '', r.team || '', r.position || '',
          r.age != null ? r.age : '', r.years_exp != null ? r.years_exp : '',
          r.games != null ? r.games : '',
        ].concat(sampleCols.map(c => schemaValue(c, r) ?? '')).concat([
          r.value != null ? r.value : ''
        ]).concat(extraKeys.map(function(k) {
          const v = state.extraData[k].byId[amRowKey(r)];
          return v != null ? v : '';
        }));
        if (amIsEachYear()) line.splice(1, 0, r.season != null ? r.season : '');
        lines.push(line.map(esc).join(','));
      });
      const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
      const rangeTag = state.weekRange ? '_' + state.weekRange : '';
      const fname = 'metrics_' + state.metric + '_' + (state.season || 'season') + rangeTag + '.csv';
      if (window.navigator && window.navigator.msSaveOrOpenBlob) {
        window.navigator.msSaveOrOpenBlob(blob, fname);
        return;
      }
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fname;
      a.rel = 'noopener';
      a.target = '_self';           // download, never a new tab
      a.style.display = 'none';
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(function() { URL.revokeObjectURL(url); }, 4000);
    });
  }
  if (rosterChk) {
    rosterChk.addEventListener('change', () => { state.rosterOnly = rosterChk.checked; state.page = 0; render(); });
  }
  const filtersBtn = document.getElementById('amFiltersBtn');
  const toolbar = document.getElementById('amToolbar') || document.getElementById('amControls');
  if (filtersBtn && toolbar) {
    filtersBtn.addEventListener('click', () => {
      const open = toolbar.classList.toggle('am-open');
      const controlsRow = document.getElementById('amControls');
      if (controlsRow) controlsRow.classList.toggle('am-open', open);
      filtersBtn.innerHTML = open ? 'Filters &#9652;' : 'Filters &#9662;';
      // On mobile, also toggle the filter bar so age + filter controls become accessible.
      const filterBar = document.getElementById('amFilterBar');
      if (filterBar) filterBar.classList.toggle('am-mobile-open', open);
      showAgeCtrl();
      updateFilterBar();
      // Close the filter form if the dropdown is closing.
      if (!open && window.innerWidth <= 600) {
        const ff = document.getElementById('amFilterForm');
        if (ff) ff.style.display = 'none';
      }
    });
  }
  const trendChk = document.getElementById('amTrendToggle');
  if (trendChk) {
    trendChk.addEventListener('change', () => {
      state.showTrends = trendChk.checked;
      syncTrendHeader();
      if (state.showTrends && !state.trendsBySeason[trendSeasonKey()]) fetchTrends();
      render();
    });
  }

  // Age filter inputs.
  const ageMinEl = document.getElementById('amAgeMin');
  const ageMaxEl = document.getElementById('amAgeMax');
  if (ageMinEl) ageMinEl.addEventListener('input', function() { state.ageMin = ageMinEl.value || ''; state.page = 0; render(); });
  if (ageMaxEl) ageMaxEl.addEventListener('input', function() { state.ageMax = ageMaxEl.value || ''; state.page = 0; render(); });

  // Combo filter form.
  const addFilterBtn = document.getElementById('amAddFilterBtn');
  const filterForm   = document.getElementById('amFilterForm');
  const filterApply  = document.getElementById('amFilterApply');
  const filterCancel = document.getElementById('amFilterCancel');
  if (addFilterBtn && filterForm) {
    const toggleFilterForm = function() {
      const opening = filterForm.style.display === 'none';
      filterForm.style.display = opening ? '' : 'none';
      // Ensure the filter bar is visible when the form is open.
      const fb = document.getElementById('amFilterBar');
      if (opening && fb) { fb.style.display = 'flex'; updateFilterBar(); }
    };
    addFilterBtn.addEventListener('click', toggleFilterForm);
    // Mobile twin inside the Filters panel (the standalone chip hides <=600px).
    const addFilterBtnM = document.getElementById('amAddFilterBtnM');
    if (addFilterBtnM) addFilterBtnM.addEventListener('click', toggleFilterForm);
  }
  if (filterApply) {
    filterApply.addEventListener('click', function() {
      const keyEl = document.getElementById('amFilterKey');
      const opEl  = document.getElementById('amFilterOp');
      const valEl = document.getElementById('amFilterVal');
      if (!keyEl || !opEl || !valEl || !valEl.value.trim()) return;
      const fkey = keyEl.value;
      state.comboFilters.push({ key: fkey, op: opEl.value, val: valEl.value.trim() });
      valEl.value = '';
      if (filterForm) filterForm.style.display = 'none';
      // Auto-add a compact filter column for metric keys that aren't already shown
      if (fkey !== 'primary' && fkey !== 'age' && cfg.metrics[fkey]) {
        if (!state.extraMetrics.includes(fkey) && (!state.filterColKeys || !state.filterColKeys.has(fkey))) {
          if (!state.filterColKeys) state.filterColKeys = new Set();
          state.filterColKeys.add(fkey);
          fetchExtraData(fkey);
          syncFilterCols();
        }
      }
      updateFilterBar();
      state.page = 0;
      render();
    });
  }
  if (filterCancel) {
    filterCancel.addEventListener('click', function() {
      if (filterForm) filterForm.style.display = 'none';
    });
  }

  // Wire column-header sort clicks.
  const _thPlayer = document.querySelector('#amTable thead th.am-player');
  const _thGames  = document.querySelector('#amTable thead th.am-games');
  const _thMetric = document.getElementById('amMetricHeader');
  if (_thPlayer) _thPlayer.addEventListener('click', () => sortByCol('name'));
  if (_thGames)  _thGames.addEventListener('click', () => sortByCol('games'));
  if (_thMetric) _thMetric.addEventListener('click', () => sortByCol(state.metric));

  // Named custom sets use the same local preference convention as pins. Removed
  // metrics are discarded on load; a missing primary makes the saved set
  // unavailable rather than silently substituting another statistic.
  const savedSetEl = document.getElementById('amSavedSet');
  function refreshSavedSets(selected) {
    if (!savedSetEl) return;
    const sets = _loadCustomSets();
    savedSetEl.innerHTML = '<option value="">Custom sets…</option>';
    Object.keys(sets).sort().forEach(function(n) { const o = document.createElement('option'); o.value = n; o.textContent = n; savedSetEl.appendChild(o); });
    if (selected && sets[selected]) savedSetEl.value = selected;
  }
  if (savedSetEl) savedSetEl.addEventListener('change', function() {
    const name = this.value, saved = _validCustomSet(_loadCustomSets()[name]);
    if (!saved) return;
    state.metric = saved.primary; metricSel.value = saved.primary;
    state.extraMetrics = saved.metrics; state.position = saved.position;
    state.sortBy = saved.primary; state.sortDir = saved.sort; state.minVol = saved.minVol;
    state.comboFilters = []; state.filterColKeys = new Set(); state.page = 0; _activePresetId = null; _showActiveSet(name);
    _updateDecisionUI(null);
    updateSortBtn(); updatePosButtons(); updateMetricTip(); updateVolCtrl(); updateVolHeader();
    updateCompareBar(); syncExtraCols(); updateFilterBar(); fetchData();
  });
  const saveSetBtn = document.getElementById('amSaveSetBtn');
  if (saveSetBtn) saveSetBtn.addEventListener('click', function() {
    const current = savedSetEl && savedSetEl.value;
    const name = (window.prompt('Custom set name', current || (_activeSet === 'Custom' ? '' : _activeSet)) || '').trim();
    if (!name) return;
    const sets = _loadCustomSets();
    sets[name] = {primary:state.metric, metrics:state.extraMetrics.slice(), position:state.position, sort:state.sortDir, minVol:state.minVol};
    try { localStorage.setItem(_CUSTOM_SET_KEY, JSON.stringify(sets)); } catch (_) { return; }
    _showActiveSet(name); refreshSavedSets(name);
  });
  const deleteSetBtn = document.getElementById('amDeleteSetBtn');
  if (deleteSetBtn) deleteSetBtn.addEventListener('click', function() {
    const name = savedSetEl && savedSetEl.value; if (!name) return;
    const sets = _loadCustomSets(); delete sets[name];
    try { localStorage.setItem(_CUSTOM_SET_KEY, JSON.stringify(sets)); } catch (_) { return; }
    _showActiveSet('Custom'); refreshSavedSets();
  });
  refreshSavedSets();

  state.minVol = _initParams.get('minvol') || defaultVol(state.metric);
  const _searchInit = _initParams.get('search') || '';
  if (_searchInit && searchEl) { searchEl.value = _searchInit; state.search = _searchInit; }
  const _presetInit = _initParams.get('preset') || '';
  const _isProPreset = function(id) { return (cfg.proPresets || []).indexOf(id) !== -1; };
  let _presetLoaded = false;
  // PRO presets never auto-load for non-PRO users (?preset= deep links included:
  // amLoadPreset itself opens the paywall for those).
  if (_presetInit && _PRESETS[_presetInit] && (cfg.hasPremium || !_isProPreset(_presetInit))) {
    amLoadPreset(_presetInit); _presetLoaded = true;
  }
  else {
    // Default landing: the last-used decision view, else Key Metrics
    // (free). Free users with a stale PRO last-preset fall back to 'general'.
    let _lastPreset = null;
    try { _lastPreset = localStorage.getItem('amLastPreset'); } catch (e) {}
    let _landing = (_lastPreset && _PRESETS[_lastPreset]) ? _lastPreset : 'key_metrics';
    if (!cfg.hasPremium && _isProPreset(_landing)) _landing = 'general';
    if (_PRESETS[_landing]) { amLoadPreset(_landing); _presetLoaded = true; }
  }
  updateSortBtn(); updatePosButtons(); updateMetricTip(); updateVolCtrl(); updateVolHeader();
  updateSortHeaders(); updateCompareBar(); updateFilterBar(); syncURL();
  if (!_presetLoaded) fetchData();
  loadOwnedRoster(); _loadMovers();
  // Auto-open graph modal when ?graph=1 is in the URL (from a copied graph
  // link), or when ?og=1 (the headless social-preview render mode).
  const _isOgRender = _initParams.get('og') === '1';
  if (_initParams.get('graph') === '1' || _isOgRender) {
    const _gxInit = _initParams.get('gx') || '';
    const _gyInit = _initParams.get('gy') || '';
    const _gzInit = _initParams.get('gz') || '';
    const _gnInit = _initParams.get('gn') || '';
    if (_isOgRender) {
      // Strip page chrome so only the graph fills the 1200x630 capture frame,
      // and force the light theme for a consistent preview.
      document.documentElement.classList.add('og-render');
      document.documentElement.setAttribute('data-theme', 'light');
      _amGraphTheme = 'light';
    }
    window.addEventListener('load', function() {
      window.amOpenGraph();
      setTimeout(function() {
        const xSel = document.getElementById('amGraphX');
        const ySel = document.getElementById('amGraphY');
        const zSel = document.getElementById('amGraphZ');
        const nSel = document.getElementById('amGraphTopN');
        if (_gxInit && xSel) xSel.value = _gxInit;
        if (_gyInit && ySel) ySel.value = _gyInit;
        if (_gzInit && zSel) zSel.value = _gzInit;
        if (_gnInit && nSel) nSel.value = _gnInit;
        if (_isOgRender) _amGraphTheme = 'light';
        window.amRenderGraph();
      }, 50);
    });
  }

  // Info tooltip: reuse the shared .adv-def-tip (advEnterMetricDef) so the
  // Primary Metric ⓘ bubble matches column-header and player-modal tips.
  const _infoEl = document.getElementById('amMetricInfo');
  if (_infoEl) {
    _infoEl.addEventListener('mouseenter', function(e) {
      if (typeof advEnterMetricDef === 'function') advEnterMetricDef(e);
    });
    _infoEl.addEventListener('mouseleave', function(e) {
      if (typeof advLeaveMetricDef === 'function') advLeaveMetricDef(e);
    });
    _infoEl.addEventListener('click', function(e) {
      if (typeof advShowMetricDef === 'function') advShowMetricDef(e);
    });
    _infoEl.addEventListener('focus', function(e) {
      if (typeof advEnterMetricDef === 'function') advEnterMetricDef(e);
    });
    _infoEl.addEventListener('blur', function(e) {
      if (typeof advLeaveMetricDef === 'function') advLeaveMetricDef(e);
    });
  }

  // Field-averages one-liner toggle (mobile rework): tap to expand the grid.
  // The open state persists across re-renders via window.__amAvgOpen.
  const _avgToggle = document.getElementById('amAvgToggle');
  if (_avgToggle) {
    _avgToggle.addEventListener('click', function() {
      window.__amAvgOpen = !window.__amAvgOpen;
      const note = document.getElementById('amAvgNote');
      if (note) note.classList.toggle('am-avg-open', !!window.__amAvgOpen);
      _avgToggle.setAttribute('aria-expanded', window.__amAvgOpen ? 'true' : 'false');
    });
  }

  // ── Custom metric picker (search + category flyout) ──────────────────────
  (function() {
    const wrap   = document.getElementById('amMetricPickerWrap');
    const btn    = document.getElementById('amMetricBtn');
    const label  = document.getElementById('amMetricBtnLabel');
    const panel  = document.getElementById('amMetricDropdown');
    if (!wrap || !btn || !panel) return;

    let _mdActiveCat = '';
    let _mdGroups = [];

    function syncLabel() {
      const opt = metricSel.options[metricSel.selectedIndex];
      const txt = opt ? opt.textContent : '';
      const isWeekly = opt && new Set(cfg.weeklyMetrics || []).has(opt.value);
      label.innerHTML = txt + (isWeekly
        ? ' <span class="am-sp-weekly-badge" title="Supports week-range filtering">W</span>'
        : '');
    }
    syncLabel();

    function readGroups() {
      const weeklySet = new Set(cfg.weeklyMetrics || []);
      const groups = [];
      for (const og of metricSel.querySelectorAll('optgroup')) {
        const options = [];
        for (const o of og.querySelectorAll('option')) {
          options.push({
            value: o.value,
            text: o.textContent,
            weekly: weeklySet.has(o.value),
          });
        }
        if (options.length) groups.push({ label: og.label, options: options });
      }
      return groups;
    }

    function catForValue(val) {
      for (const g of _mdGroups) {
        if (g.options.some(function(o) { return o.value === val; })) return g.label;
      }
      return _mdGroups[0] ? _mdGroups[0].label : '';
    }

    function searchTerm() {
      const inp = panel.querySelector('#amMdSearch');
      return (inp && inp.value ? inp.value : '').trim().toLowerCase();
    }

    function optMatches(o, term) {
      if (!term) return true;
      return (o.text || '').toLowerCase().includes(term)
        || (o.value || '').toLowerCase().includes(term);
    }

    function metricItemHtml(o) {
      const isSel = o.value === metricSel.value;
      const badge = o.weekly
        ? '<span class="am-sp-weekly-badge" title="Supports week-range filtering">W</span>'
        : '';
      return '<div class="am-sp-item' + (isSel ? ' am-sp-active' : '') + '" data-val="'
        + o.value + '" role="option">'
        + '<span class="am-sp-check">' + (isSel ? '&#10003;' : '') + '</span>'
        + '<span class="am-md-item-label">' + o.text + '</span>'
        + badge + '</div>';
    }

    function fillMetrics() {
      const pane = panel.querySelector('#amMdMetrics');
      if (!pane) return;
      const term = searchTerm();
      const g = _mdGroups.find(function(x) { return x.label === _mdActiveCat; })
        || _mdGroups[0];
      const items = (g ? g.options : []).filter(function(o) { return optMatches(o, term); });
      if (!items.length) {
        pane.innerHTML = '';
        const empty = document.createElement('div');
        empty.className = 'am-md-empty';
        empty.textContent = term
          ? 'No metrics match "' + term + '" in ' + ((g && g.label) || 'this category') + '.'
          : 'No metrics in this category.';
        pane.appendChild(empty);
        return;
      }
      pane.innerHTML = items.map(metricItemHtml).join('');
      pane.querySelectorAll('.am-sp-item').forEach(function(el) {
        el.addEventListener('click', function() {
          metricSel.value = el.dataset.val;
          syncLabel();
          closePanel();
          metricSel.dispatchEvent(new Event('change'));
        });
      });
      const sel = pane.querySelector('.am-sp-item.am-sp-active');
      if (sel) sel.scrollIntoView({ block: 'nearest' });
    }

    function syncCatClasses() {
      const term = searchTerm();
      panel.querySelectorAll('.am-md-cat').forEach(function(el) {
        const g = _mdGroups.find(function(x) { return x.label === el.dataset.cat; });
        const hit = !term || (g && g.options.some(function(o) { return optMatches(o, term); }));
        el.classList.toggle('am-md-cat-active', el.dataset.cat === _mdActiveCat);
        el.classList.toggle('am-md-cat-dim', !hit);
      });
    }

    function setActiveCat(label) {
      if (!label) return;
      if (label === _mdActiveCat) { syncCatClasses(); return; }
      _mdActiveCat = label;
      syncCatClasses();
      fillMetrics();
    }

    function fillCats() {
      const pane = panel.querySelector('#amMdCats');
      if (!pane) return;
      pane.innerHTML = _mdGroups.map(function(g) {
        return '<div class="am-md-cat" data-cat="' + g.label + '" role="option">'
          + '<span>' + g.label + '</span>'
          + '<span class="am-md-cat-chevron">&#8250;</span></div>';
      }).join('');
      pane.querySelectorAll('.am-md-cat').forEach(function(el) {
        el.addEventListener('mouseenter', function() {
          if (window.matchMedia('(hover: hover)').matches) setActiveCat(el.dataset.cat);
        });
        el.addEventListener('click', function(e) {
          e.preventDefault();
          setActiveCat(el.dataset.cat);
        });
      });
      syncCatClasses();
    }

    function buildPanel() {
      _mdGroups = readGroups();
      _mdActiveCat = catForValue(metricSel.value);
      panel.innerHTML =
        '<div class="am-md-search-wrap">'
        + '<input type="text" id="amMdSearch" class="am-sp-search" placeholder="Search metrics…" autocomplete="off" aria-label="Search metrics">'
        + '</div>'
        + '<div class="am-md-body">'
        + '<div class="am-md-cats" id="amMdCats" role="list"></div>'
        + '<div class="am-md-metrics" id="amMdMetrics" role="listbox"></div>'
        + '</div>';
      fillCats();
      fillMetrics();
      const inp = panel.querySelector('#amMdSearch');
      if (inp) {
        inp.addEventListener('input', function() {
          const term = searchTerm();
          const active = _mdGroups.find(function(g) { return g.label === _mdActiveCat; });
          const activeHit = active && active.options.some(function(o) { return optMatches(o, term); });
          if (term && !activeHit) {
            const firstHit = _mdGroups.find(function(g) {
              return g.options.some(function(o) { return optMatches(o, term); });
            });
            if (firstHit) _mdActiveCat = firstHit.label;
          }
          syncCatClasses();
          fillMetrics();
        });
        inp.addEventListener('keydown', function(e) {
          if (e.key === 'Escape') { e.stopPropagation(); closePanel(); btn.focus(); }
        });
      }
    }

    // The panel is absolutely positioned inside the card, whose global
    // overflow:hidden clips a tall dropdown. Pin it with position:fixed relative
    // to the button (like the info tooltip) so it escapes the card, and cap its
    // height to the space below the button so it scrolls internally.
    function positionPanel() {
      const r = btn.getBoundingClientRect();
      const bottomGap = window.matchMedia('(max-width:768px)').matches ? 78 : 16;
      const avail = window.innerHeight - r.bottom - bottomGap;
      const maxH = Math.max(220, Math.min(520, avail));
      panel.style.position = 'fixed';
      panel.style.top = (r.bottom + 6) + 'px';
      panel.style.zIndex = '900';
      panel.style.maxHeight = maxH + 'px';
      const width = panel.getBoundingClientRect().width || 520;
      let left = r.left;
      if (left + width > window.innerWidth - 8) {
        left = Math.max(8, window.innerWidth - width - 8);
      }
      panel.style.left = left + 'px';
      panel.style.right = 'auto';
    }
    function clearPosition() {
      panel.style.position = '';
      panel.style.top = '';
      panel.style.left = '';
      panel.style.right = '';
      panel.style.zIndex = '';
      panel.style.maxHeight = '';
    }

    function openPanel() {
      buildPanel();
      wrap.classList.add('open');
      btn.setAttribute('aria-expanded', 'true');
      positionPanel();
      const inp = panel.querySelector('#amMdSearch');
      if (inp && window.matchMedia('(hover: hover)').matches) {
        setTimeout(function() { inp.focus(); }, 0);
      }
    }
    function closePanel() {
      wrap.classList.remove('open');
      btn.setAttribute('aria-expanded', 'false');
      clearPosition();
    }

    btn.addEventListener('click', function(e) {
      e.stopPropagation();
      wrap.classList.contains('open') ? closePanel() : openPanel();
    });
    panel.addEventListener('click', function(e) { e.stopPropagation(); });
    document.addEventListener('click', closePanel);
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape' && wrap.classList.contains('open')) closePanel();
    });
    window.addEventListener('resize', function() { if (wrap.classList.contains('open')) closePanel(); });
    window.addEventListener('scroll', function(e) {
      if (!wrap.classList.contains('open')) return;
      if (e.target && e.target.nodeType === 1 && (e.target === panel || panel.contains(e.target))) return;
      closePanel();
    }, true);

    metricSel.addEventListener('change', syncLabel);
  })();
"""