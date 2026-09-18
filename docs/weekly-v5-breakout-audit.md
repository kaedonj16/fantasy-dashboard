# Weekly Breakout v5: data and scoring audit

## Locally available weekly inputs

| Metric | Source | Coverage / ID mapping | Scoring safety |
|---|---|---|---|
| Snaps, team snaps, snap share | cached weekly player stats (`off_snp`, `tm_off_snp`) | weekly; Sleeper player ID; all seasons retained by the weekly cache | eligibility and negative gating only |
| Targets, carries, pass attempts | cached weekly player stats | weekly; Sleeper player ID; expected skill-player coverage | required position-specific evidence |
| Target share | player targets divided by the mapped historical team's targets | weekly when team target totals exist; Sleeper ID plus team-history mapping | required receiving evidence |
| Red-zone targets/carries | cached `rec_rz_tgt` / `rush_rz_att` fields | weekly but provider-dependent; Sleeper ID | optional high-value enrichment |
| Routes, team dropbacks, route participation, targets/route | no production ingestion exists | unavailable; fixture/replay rows may supply routes and dropbacks | route gates activate when present; otherwise conservative target-volume **and** target-share fallback |
| Air-yards share, first-read targets, red-zone routes | no production ingestion exists | unavailable | optional future enrichment; never imputed from snaps |
| Backfield opportunity share | no team-backfield denominator is persisted | unavailable; carries-plus-targets is available | optional future enrichment |
| Two-minute snaps, goal-line carries | no production ingestion exists (red-zone carries are broader) | unavailable | optional future enrichment |

Missing optional fields are represented as unavailable and reduce confidence. In
particular, WR/TE route participation is never inferred from snap share.

## v5 qualification contract

Snap share has zero positive scoring weight. WR/TE scores weight route
participation 35%, target share 35%, targets 22%, and red-zone opportunity 8%.
When routes are absent, the receiving floor requires both five targets and 15%
target share. RB scoring uses carries-plus-targets (60%), targets (22%), routes
(8%), and red-zone work (10%). QB scoring uses dropback share (35%), pass
attempts (45%), and rushing attempts (20%).

Main-board entry requires the position role floor, two independent non-snap
signals, sufficient novelty, adequate coverage, no garbage-time flag, and no
established-role exclusion. A one-game profile additionally has to pass every
exceptional-evidence check. Other promising profiles are retained as
`early_watch`; verified short injury openings are separated as
`temporary_opportunities`; declining prior detections are exposed as `cooling`.

One-game scores receive continuous 0.68 shrinkage before ranking and a 39.5
safety ceiling (17.9 for established players). Run metadata records exact-score
ties, the share tied at the provisional maximum, and provisional reductions; a
warning is emitted when more than 20% of a provisional cohort of at least ten
players shares its maximum.

## Recalculation

Production refresh uses:

```bash
python -m data_building.breakout_engine.calculate_breakouts_with_real_data
```

The command refreshes weekly metrics, scores through the last completed schedule
week, persists all tracked cohorts, and logs coverage and score distribution.
