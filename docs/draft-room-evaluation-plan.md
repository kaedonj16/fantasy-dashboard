# Draft Room evaluation improvement plan

## Scope and guardrails

This plan covers only redraft evaluation changes for late-pick influence, roster
facets, opportunity cost, meaningful reach/steal cards, score terminology,
bench utility, bye severity, and late-round upside. It does not propose a Draft
Room rewrite, a new top-level dashboard, or changes to unrelated draft UX.

The implementation should remain additive around the existing pure kernels:

- keep `BRPickScore.computePickScore` / `utils.pick_score.compute_pick_score` as
  the parity-tested **absolute Pick Score** kernel;
- keep live timing and roster context in `DraftBoardCore` and the Draft Room
  decision layer;
- extend `BRTeamGrade.teamGradeComposite` and
  `utils.draft_grade.dr_team_grade_score` together, with parity tests changed in
  the same commit;
- use the existing lineup eligibility, roster-role, roster-utility,
  streamability, availability, and ADP-uncertainty helpers rather than copying
  them into Deep Dive rendering code;
- preserve rookie grading unless a shared API must accept the mode, and preserve
  startup behavior unless a format-aware helper is intentionally common;
- keep K/DEF out of Pick Score and grade influence.

## Current-state data-flow trace

### 1. Live Recommendation ranking

1. `refreshPsPool()` obtains the remaining player pool and current roster
   counts, computes an absolute score into `p._ps`, prepares expected next-pick
   positional values, and computes `p._ds` with `liveDecisionScore()`.
2. `pickScore()` only gathers inputs (value, VOR, ADP, tier, need, age,
   momentum, PPG and scoring settings) for `BRPickScore.computePickScore()`.
3. `liveDecisionScore()` derives candidate roster role and positional utility
   through `DraftBoardCore.candidateRosterRole`, `rosterRole`,
   `positionNeedUtility`, and `rosterSlotUtility`. It also calculates shelf
   wait-loss, survival/wait penalty, required-slot pressure, recent redundant
   QB/TE investment, and redraft handcuff insurance.
4. `DraftBoardCore.decisionScore()` combines those contextual terms with the
   absolute Pick Score. `rankedRecommendationPool()` sorts by `_ds`, breaking
   ties with `_ps`. When advice is for a future owned pick, survival further
   scales the contextual score.
5. The sidebar intentionally presents Recommendation as an ordinal rank, not a
   numeric grade.

**Finding:** Recommendation and Pick Score are already separated in the main
kernel, but terse internal names (`_ds`, `_ps`) and several surface descriptions
still make the distinction harder to audit. The historical analyzer cannot yet
reconstruct the same contextual comparison at each old pick.

### 2. Absolute Pick Score

1. `static/pick_score.js` and `utils/pick_score.py` contain mirrored formulas,
   guarded by `tests/test_pick_score_parity.py`.
2. Redraft Pick Score weights VOR, model value, market/ADP, tier, need,
   momentum, and projected PPG. It already reduces ADP weight with draft round
   and redistributes that weight to football-quality inputs.
3. It applies format/scoring adjustments and redundancy penalties, then a
   draft-depth normalization and monotonic display relabel.
4. Survival and handcuff timing are deliberately excluded; those belong to the
   live decision layer.

**Finding:** Pick Score answers player-at-slot quality and should not absorb the
new late-round recommendation policy. Late-round upside belongs in contextual
Decision Score unless validation demonstrates that it predicts historical pick
quality and is added to both parity copies.

### 3. Board PS (historical/pool-relative Pick Score)

1. Live surfaces scale absolute Pick Score against the best current absolute
   score using `psDisplay()` / `psRelLive()`.
2. Made mock/manual picks can retain `psRel` at commit time.
3. Synced picks are reconstructed by `_ensureRelScores()`: replay picks in
   order, remove previously selected players, recompute absolute Pick Score at
   the historical slot, and scale the selection against the best candidate.
4. `relPS()` selects committed, reconstructed, or absolute fallback data. Deep
   Dive and the report ledger call this **Board PS**.

**Finding:** Board PS is a relative display/evaluation of the chosen player
against the historical pool, not the absolute kernel score and not the
contextual Recommendation score. The reconstruction currently uses a bounded
top-value candidate set and empty roster counts, so it is not sufficient for
opportunity-cost analysis that promises roster-aware alternatives.

### 4. Completed-team Draft Grade

1. Draft Room gathers every team's picks and recomputes absolute Pick Score in
   grading context.
2. `BRTeamGrade.teamGradeComposite()` and the Python mirror identify an optimal
   lineup by PPG, falling back to model value.
3. Redraft currently allocates 20 points to **Value**, 50 to **Starters**, and
   30 to **Construction**.
4. Value is a `1 / round^0.60` weighted average of starter Pick Scores only.
   Starters compare optimal-lineup PPG (or value fallback) with a roster-valid
   league-wide baseline. Construction combines lineup coverage, position-count
   balance against targets, and a count-based efficiency cap.
5. Raw team scores are curved against the league field separately.

**Finding:** final fringe picks do not enter Value directly if they remain on
the bench, but they can still affect which player is classified as a starter,
and every QB/RB/WR/TE affects count-based Construction without distinguishing a
useful RB3 from redundant QB2/TE2. There is no explicit primary-depth or upside
facet, and `avgPs` still averages all scored selections for display/reporting.

### 5. Deep Dive edges and risk flags

1. `ddMyPicks()` replays the taken set, computes selected/consensus ADP deltas,
   the best remaining ADP, BPA status, next-pick survival, ADP uncertainty,
   Board PS, and tier.
2. `ddVerdict()` delegates to `DraftBoardCore.adpDeltaVerdict()`. A reach is
   suppressed for a remaining ADP co-BPA, a player unlikely to survive, or a
   delta within the round/source uncertainty tolerance.
3. `ddEdgesHtml()` chooses the largest qualifying ADP fall as steal, the lowest
   ADP-delta row already labeled reach as Biggest Reach, and the highest Board
   PS as Best Pick.
4. Thin-position and bye flags are assembled directly in the renderer.

**Finding:** reach handling has useful uncertainty and survival safeguards, but
it is still market-first and has no record of realistic alternatives or their
decision-quality gap. The edge renderer forces an extreme once any row crosses
the ledger's reach boundary. Steal significance uses a fixed three-pick delta,
which is especially noisy late.

### 6. Roster construction scoring

The completed-grade kernel counts players by position, rewards progress toward
position targets, and treats picks through `target + 1` as useful. In contrast,
live Recommendation already has format-aware role/utility logic with diminishing
utility for backup-only QB/TE and support for SF/TEP. The two paths therefore
answer different construction questions.

**Root cause:** completed grading does not consume the same role and utility
concepts as live advice, so positional symmetry can score as well as functional
bench depth.

### 7. Bye-week warnings

The live player-row badge uses the raw count of already-owned players on the
same bye. Deep Dive groups all drafted players by week and emits a critical flag
when the largest group contains at least three players.

**Root cause:** neither path considers optimal-starter status, player strength,
FLEX competition, replacement cover, or format-aware QB/TE streamability. A
three-player fringe-bench cluster can therefore look worse than a concentrated
group of elite starters.

## Proposed shared evaluation model

### A. Shared roster evaluation snapshot

Add a pure, reusable snapshot helper to `static/draft_board_core.js` and a
Python mirror only where completed server grading needs identical output. Its
inputs should be picks, roster slots, format/scoring settings, team count, and
replacement/quality data; its output should contain:

- optimal starter IDs and assigned slots;
- each player's role: starter, primary bench, or fringe bench;
- position ordinal (RB3, WR4, QB2, and so on);
- `functionalUtility` based on the existing `rosterSlotUtility`,
  `positionNeedUtility`, and streamable-single-slot rules;
- normalized PPG/model value above replacement;
- `starterStrength`, `functionalDepth`, and `benchUpside` facets;
- bye-week severity summaries.

Do not create a second lineup solver. Either expose/adapt the current optimal
lineup helper or move the common eligibility/assignment primitive to the core
module and call it from team grading and Deep Dive.

#### Role classification

- **Starter:** selected by the optimal lineup solver.
- **Primary bench:** the best remaining cover for each dedicated/FLEX/SF path,
  plus additional RB/WR reserves whose above-replacement quality and lineup
  path clear a calibrated minimum. A player can cover multiple eligible slots
  but is counted once.
- **Fringe bench:** all remaining non-special-team reserves.
- K/DEF: retained for roster completeness but evaluation weight is zero.

This makes an RB3 or WR4 valuable because it is the first injury/bye replacement,
not merely because of its position label.

### B. Starter strength, functional depth/floor, and upside

Keep the current starter-strength ratio as the starting point. Split the old
Construction internals into separately inspectable values:

```text
starterStrength = current roster-valid optimal-lineup strength ratio mapped to 0..1

coverQuality(player, slot) =
    eligibility * functionalUtility * clamp01(aboveReplacementQuality)

functionalDepth = weighted mean of the best unused cover for every starter path
                  (dedicated slots first, then FLEX/SF), with diminishing credit
                  for a second cover at the same path

benchEfficiency = sum(functionalUtility for bench players)
                  / max(1, number of non-K/DEF bench spots)
```

For redraft, QB2 and TE2 receive lower functional utility when QB/TE are
streamable; SF and multi-QB restore QB utility, while TEP/multi-TE restore TE
utility. One QB/TE in a standard format receives no blanket penalty.

Upside must be evidence-based. Define a conservative `upsideEvidence` from
fields actually present in the payload, in this priority order:

1. explicit breakout/role/contingent-value metadata, if coverage and semantics
   are verified during implementation;
2. handcuff/teammate workload path already recognized by Draft Room;
3. tier proximity and model value/VOR above replacement;
4. projected role or normalized PPG that creates a plausible FLEX/starter path;
5. age only as a small interaction for RB/WR with a demonstrated role/path,
   never as a standalone bonus.

A proposed bounded proxy, subject to fixture/backtest calibration, is:

```text
pathScore = max(roleMetadata, contingentWorkload, flexPath)
qualityScore = 0.55 * aboveReplacement + 0.25 * tierQuality + 0.20 * ppgQuality
ageInteraction = eligibleYoungRBorWR * pathScore * 0.10
upsideEvidence = clamp01(0.55 * pathScore + 0.45 * qualityScore + ageInteraction)
benchUpside = weighted mean of the best 3-5 non-starter upsideEvidence values
```

If the payload lacks reliable `pathScore` inputs, omit that component and
renormalize over verified quality inputs; mark the resulting commentary
confidence low. Do not present a low-confidence proxy as a breakout prediction.

### C. Smooth marginal pick influence in Draft Grade

Replace starter-only Value aggregation with a role-aware weighted mean of all
meaningful non-K/DEF picks. Preserve the 20-point redraft Value cap initially so
the change is isolated from the field curve:

```text
round = ceil(pickNumber / teams)
roundDecay = (1 + (round - 1) / 5) ^ -0.85

roleWeight = 1.00  starter
             0.55  primary bench
             0.18  fringe bench

utilityWeight = 0.55 + 0.45 * functionalUtility
lineupCompletionWeight = 1.0 before the core lineup is filled,
                         then smoothly approaches 0.75 for bench additions

pickInfluence = roundDecay * roleWeight * utilityWeight * lineupCompletionWeight
```

The exact exponents/anchors are calibration parameters, not magic constants:
lock them only after sensitivity fixtures show (a) a final fringe swap changes
the raw total by at most roughly 0.5 grade point, (b) an early-starter downgrade
is materially larger, and (c) RB3/WR4 changes exceed QB2/TE2 changes in 1QB,
while SF/TEP reverse the relevant utility discount. K/DEF always have weight 0.

To avoid a lucky round-15 fall inflating a weak roster, use the weighted mean,
not a sum, and cap fringe players' combined Value weight (proposed cap: 10% of
total influence). Keep `avgPs` as a diagnostic only and relabel it clearly.

Construction should then blend observable roster results rather than counts:

```text
constructionRaw =
    0.45 * coverage
  + 0.35 * functionalDepth
  + 0.20 * benchEfficiency
```

`benchUpside` should initially drive Deep Dive commentary and late-round advice,
not silently add grade points. After outcome validation, at most move a small
documented portion of Construction (for example 0.05) from depth/efficiency to
upside; do not increase the 30-point Construction cap.

### D. Historical opportunity cost

Add a pure `historicalDecisionContext()`/`rankHistoricalAlternatives()` helper
around existing core primitives. For every user pick:

1. replay all prior picks and keeper removals;
2. reconstruct that user's roster immediately before the selection;
3. use the actual remaining pool at that slot;
4. compute absolute Pick Score for the selected player and plausible
   alternatives with historical roster counts/quality, not today's final roster;
5. pass each through the same roster role/utility, obligations, wait-loss,
   survival, and decision-score concepts that are reproducible without future
   information;
6. exclude K/DEF except when a required final-slot choice is being evaluated;
7. retain the top five and show three by default.

Do not use observed future selections to calculate recommendation economics.
The historical pool is valid; future room behavior is not. Use the calibrated
ADP survival model at that slot and only picks already made to infer runs.

```text
selectedDecisionScore = reproducible historical Decision Score
bestAlternativeScore = max(realistic alternative Decision Scores)
opportunityCost = bestAlternativeScore - selectedDecisionScore

gapSeverity = none        when gap < 4
              modest      when 4 <= gap < 9
              material    when 9 <= gap < 15
              severe      when gap >= 15
```

The initial 4-point tolerance matches the scale on which contextual terms can
break close absolute-score ties and must be calibrated against hand-labeled
fixtures. Store `bestAlternative`, `topAlternatives`, absolute scores,
historical Decision Scores, and opportunity-cost confidence on each Deep Dive
row. Board PS remains separate and unchanged in meaning.

### E. Significant reach and steal classification

Keep the current ADP uncertainty, remaining-BPA, and survival gates, then add
opportunity cost:

```text
significantReach =
    outsideExpectedMarketRange
    AND surviveToNextPick >= 20%
    AND opportunityCost >= 9
    AND not remaining/co-BPA
```

If ADP sources or historical candidate inputs are missing, reduce confidence
and require a larger observed gap; never infer a severe reach from ADP alone.
`ddVerdict()` may still label a 4-8 point gap “Aggressive,” but only a qualifying
row can populate Biggest Reach. Otherwise render **No major reaches** with
cautious copy.

For steals, require both an uncertainty-adjusted market fall and quality:

```text
significantSteal = marketFall >= max(8 picks, 0.5 * adpUncertainty)
                   AND Board PS >= 80
```

Otherwise show Best Pick rather than exaggerating the largest mathematical
fall. Tune these gates with late-round noise fixtures rather than globally
lowering uncertainty.

### F. Severity-aware bye evaluation

Compute bye concentration from the same optimal-lineup snapshot:

```text
roleImpact = 1.00 starter, 0.30 primary bench, 0.08 fringe bench
qualityImpact = 0.60 + 0.40 * normalized lineup quality
streamability = 0.55 for QB/TE in standard 1QB/non-TEP,
                1.00 for scarce/multi-slot formats and RB/WR
coverRelief = 0.35 * best eligible non-bye bench cover quality

playerByeImpact = max(0, roleImpact * qualityImpact * streamability - coverRelief)
weekSeverity = sum(playerByeImpact) + 0.25 * max(0, impactedStarters - 2)
```

Proposed output bands, to be fixture-calibrated:

- `< 1.0`: no warning;
- `1.0..<1.8`: mild concentration;
- `1.8..<2.8`: meaningful starter overlap;
- `>= 2.8` or four materially impacted starters: severe bye-week crunch.

FLEX-eligible players should be assigned once and cover relief should not reuse
one bench player for multiple starters. Copy must say this is a manageable
scheduling risk, not proof of a bad draft. The live candidate badge should use
the prospective severity delta rather than raw same-week count.

### G. Smooth late-round upside in Recommendation

Add pure helpers in `DraftBoardCore`:

```text
draftPhase(round, totalRounds) = clamp01((round - 4) / max(1, totalRounds - 4))
latePhase = smoothstep(draftPhase) = x*x*(3 - 2*x)

lateRoundUtility = upsideEvidence
                   * functionalUtility
                   * (0.65 + 0.35 * rosterNeedPath)

upsideDecisionBonus = latePhase * 10 * (lateRoundUtility - 0.35)
```

Clamp the bonus to `[-3, +7]` Decision Score points and apply it once in
`decisionScore()`, after absolute Pick Score and before final clamping. This
gives rounds 1-4 approximately no effect, a small transition in rounds 5-8,
meaningful separation in rounds 9-11, and the strongest (still bounded) effect
in rounds 12+. Use the league's configured total rounds rather than fixed round
numbers. K/DEF timing remains in its existing separate path.

The neutral anchor prevents every late player from receiving free points. A
young player with no role/path has low `upsideEvidence`; a veteran with clear
touches can outrank that player. This changes Recommendation ordering without
changing absolute Pick Score, which is an explicit semantic regression test.

## Semantic contract and UI copy

Use these definitions everywhere:

| Term | User question | Inputs | Must not mean |
| --- | --- | --- | --- |
| Recommendation Rank | Who should I draft right now? | Pick Score plus live roster fit, survival, scarcity, obligations, recent investment, handcuff and phase/upside | a historical grade |
| Pick Score | How good is this player at this pick? | parity-tested absolute player/pick-quality kernel | Recommendation or pool-relative score |
| Board PS | How good was this selection relative to what was available then? | historical absolute Pick Score scaled to the slot's remaining pool | live Recommendation |
| Draft Grade | How good is the resulting roster? | starters, functional depth/construction and meaningfully weighted pick efficiency | average Recommendation rank |

Update the glossary, tooltips, sort labels, pick ledger introduction, meter copy,
and Deep Dive explanations in `static/draft_room.js`. Where a local rename is
low-risk, use `absolutePickScore`, `decisionScore`, and `boardPickScore`; avoid a
repository-wide rename of persisted `psRel` or payload fields merely for style.

Planned Deep Dive examples:

- **Biggest Reach:** “18 picks outside the expected market range; [RB] carried
  a 14-point historical Decision Score advantage.”
- **No major reaches:** “Your picks stayed within reasonable market ranges or
  close decision-quality bands; late swings were judged with wider ADP
  uncertainty.”
- **Strong functional depth:** “Your first RB/WR reserves provide credible
  injury and FLEX cover.”
- **Strong upside bench:** only when verified upside evidence is present.
- **Safe but limited ceiling:** only when floor/depth is adequate and upside
  evidence is consistently low.
- **Redundant single-slot depth:** identify QB2/TE2-heavy usage only in formats
  where those positions are streamable.

## Implementation sequence and file-level changes

### Phase 0 — Freeze baselines and fixtures

1. Capture current focused tests and backtest output before changing formulas.
2. Add deterministic redraft fixtures for balanced, zero-RB, hero-RB, early-QB,
   late-QB, QB2/TE2-heavy, RB/WR-upside-heavy, starter-bye concentration, and
   aggressive-but-defensible late picks.
3. Record each fixture's component scores, edge cards, alternatives, and bye
   severity in a machine-readable artifact; snapshots must assert semantics,
   not entire HTML blobs.

Expected files: a new fixture module under `tests/`, additions to
`tests/test_draft_grade_backtest.py`, and an optional checked-in baseline summary
under `artifacts/` if repository policy permits generated evaluation reports.

### Phase 1 — Shared roles, facets, and grade influence

1. Extend `static/draft_board_core.js` with pure role/bench utility and facet
   helpers by composing existing roster APIs.
2. Change `static/draft_grade_team.js` and `utils/draft_grade.py` in lockstep to
   accept enough roster/format context, compute role-aware pick influence, and
   expose `starterStrength`, `functionalDepth`, `benchEfficiency`, and
   `benchUpside` diagnostics.
3. Keep existing return keys during migration (`starter`, `balance`) so callers
   do not break; add clearer keys and deprecate aliases only after all callers
   move.
4. Update the backtest facet implementation to call production helpers or an
   exact shared Python primitive instead of maintaining another count-based
   “depth” formula.

Expected files: `static/draft_board_core.js`, `static/draft_grade_team.js`,
`utils/draft_grade.py`, `data_building/draft_grade_backtest.py`,
`tests/test_draft_board_core.py`, `tests/test_draft_grade.py`, and
`tests/test_team_grade_parity.py`.

### Phase 2 — Historical alternatives and edge severity

1. Add historical context/ranking helpers to `DraftBoardCore` where they can be
   Node-tested.
2. Refactor `ddMyPicks()` to reconstruct the pre-pick roster and attach selected
   score, top alternatives, opportunity cost, confidence, and significance.
3. Make verdict/card selection use significance gates. Render No major reaches
   when appropriate and apply the steal quality floor.
4. Keep charts based on uncertainty-adjusted market delta; add opportunity cost
   to tooltip/ledger explanation rather than silently redefining the axis.

Expected files: `static/draft_board_core.js`, `static/draft_room.js`,
`tests/test_draft_board_core.py`, and `tests/test_draft_room_scoring_settings.py`.

### Phase 3 — Bye severity and bench commentary

1. Put bye severity in a pure shared helper consuming the roster snapshot.
2. Replace Deep Dive's raw three-player critical flag and the live raw-count
   badge with severity bands/prospective deltas.
3. Generate floor/depth/upside commentary from the facet values and confidence,
   without adding three new top-level meters.

Expected files: `static/draft_board_core.js`, `static/draft_room.js`, plus focused
core/Deep Dive tests.

### Phase 4 — Draft phase and upside-oriented recommendations

1. Add and export `draftPhase`, `lateRoundUtility`, and the bounded Decision
   Score adjustment in `static/draft_board_core.js`.
2. Gather only verified player inputs in `static/draft_room.js` and pass the
   result to both live recommendations and reproducible historical analysis.
3. Keep CPU/mock selection on the same helper where it models the same decision;
   do not leak user-only roster state or change K/DEF fill rules.
4. If Python recommendation tooling exists in the touched backtest path, mirror
   the helper and add parity. Do not add a Python copy solely for symmetry when
   no Python caller evaluates Recommendation.

Expected files: `static/draft_board_core.js`, `static/draft_room.js`, relevant
mock/benchmark call sites, and core/realism tests. `static/pick_score.js` and
`utils/pick_score.py` should remain unchanged unless validation establishes a
true absolute-quality signal, in which case they change together.

### Phase 5 — Terminology and calibration

1. Complete the narrow glossary/tooltips/internal-local renames.
2. Run all parity, Draft Room, mock, keeper, custom-roster, SF, TEP, rookie, and
   startup regression suites.
3. Compare baseline and candidate backtests. Ship only changes that improve
   hand-labeled realism without materially regressing outcome correlation or
   recommendation benchmark quality.

Expected files: primarily `static/draft_room.js` and tests; no unrelated layout
or responsive CSS work is planned.

## Regression-test matrix

### Late-round grade influence

- Two similarly valued 14.08 fringe darts move raw/final grade by no more than
  the calibrated small bound.
- A clear first-round starter downgrade moves the grade materially more.
- RB3/WR4 quality changes outweigh QB2/TE2 changes in standard redraft.
- SF QB depth and TEP TE depth regain appropriate weight.
- K/DEF swaps, additions, and ordering leave grade/component values invariant.

### Floor, depth, upside, and construction

- Four useful RB/WR reserves beat a redundant QB2/TE2/QB3 bench in 1QB/1TE.
- One QB and one TE incur no blanket construction penalty in standard redraft.
- SF values QB cover and TEP values TE cover.
- Equal position counts with different above-replacement quality produce
  different functional-depth scores.
- Upside commentary is withheld or low-confidence when explicit/path metadata
  is absent.

### Opportunity cost and reaches

- A one-point historical Decision Score gap is within tolerance.
- A 20-point gap is material/severe and records the best/top alternatives.
- Superior alternative ADP without superior decision quality is insufficient.
- No qualifying selection renders No major reaches.
- One market-outlier, likely-to-survive, high-opportunity-cost pick renders
  Biggest Reach.
- Late ADP noise alone does not qualify.
- A tiny late fall cannot become Biggest Steal; a large, high-Board-PS fall can.

### Semantic separation

- Survival/roster/phase context can reorder Recommendation while identical
  inputs preserve absolute Pick Score.
- Board PS differs from absolute Pick Score when pool quality changes.
- Draft Grade responds to roster output and weighted decision quality, never to
  the displayed Recommendation ordinal.
- Glossary and tooltips state all four questions consistently.

### Bye severity

- Three fringe bench players on one bye do not trigger severe warning.
- Three elite starters create meaningful or severe overlap based on available
  cover.
- Four material starters trigger severe crunch.
- Standard QB1+TE1 overlap is mild at most when streamable; SF/TEP can raise it.
- One FLEX reserve cannot relieve multiple simultaneous starter absences.

### Late-round upside

- A role-backed upside RB/WR can outrank a similar-ADP low-ceiling reserve late.
- The same modifier is approximately zero in round two.
- Youth without role/path does not beat a veteran with a clear workload solely
  because of age.
- K/DEF ordering/timing remains unchanged.

### Compatibility

Cover live, manual, synced, mock, keeper, rookie, startup, redraft 1QB/SF,
PPR/TEP, unusual lineup slots, partial drafts, missing projections/ADP/upside
metadata, and mobile Deep Dive content structure.

## Backtest and validation protocol

### Before and after

Run the same frozen league/sample set and random seeds for baseline and
candidate code. Persist configuration, sample counts, skipped-data counts, and
the git SHA with each report.

```bash
PYTHONPATH=. pytest -q \
  tests/test_pick_score.py tests/test_pick_score_parity.py \
  tests/test_team_grade_parity.py tests/test_draft_board_core.py \
  tests/test_draft_room_scoring_settings.py tests/test_draft_grade.py \
  tests/test_draft_grade_backtest.py

PYTHONPATH=. python -m data_building.run_draft_backtest \
  --league <frozen-league-ids> --season <completed-season> \
  --history --auto-type --method spearman
```

Also run the repository's broader Draft Room/CPU realism tests and full CI test
command after focused tests pass.

### Metrics to compare

- Spearman and Pearson correlation of raw composite and component/facet scores
  with existing outcome targets, with sample sizes and confidence caveats.
- Grade-quintile and letter-grade monotonicity.
- Per-round absolute Pick Score (expected unchanged unless the core changes).
- Sensitivity deltas for early starter, primary depth, redundant QB/TE, fringe
  dart, and K/DEF perturbations.
- Recommendation benchmark quality: average selected Decision Score,
  opportunity cost, roster coverage, redundant single-slot picks, and late
  upside utility.
- Fixture pass rate and reviewer labels for the nine named roster archetypes.
- Reach/steal false-positive counts, especially rounds 12+.
- Bye warning precision across hand-labeled no/mild/meaningful/severe cases.

Do not select constants from aggregate correlation alone. A candidate ships only
if parity holds, obvious scenarios improve, no key format regresses, and any
correlation change is reported. Prefer simpler coefficients within statistical
noise.

## Planned before/after examples

| Scenario | Current behavior/root cause | Intended behavior |
| --- | --- | --- |
| Round-15 fringe WR ADP fall | Can affect all-pick average displays and count construction like any other bench body | Tiny capped grade influence; may still be a good Board PS without transforming the roster grade |
| Round-11 RB3 vs round-8 QB2 | Construction mostly sees target/count progress | RB3 receives primary-cover utility; standard 1QB QB2 receives diminished bench utility |
| WR scored 84 vs RB scored 85 | Market delta can dominate reach language | Opportunity cost is inside tolerance; verdict stays fair/aggressive, not a major mistake |
| WR scored 71 vs RB scored 91 | No explicit passed-alternative record | Material opportunity cost names the RB and can support Biggest Reach if market/survival gates also pass |
| Three bench players share Week 7 | Raw count can emit a critical warning | No major warning unless their projected roles create real coverage loss |
| RB1/RB2/WR1 share Week 7 | Same raw counting mechanism | Meaningful/severe starter-overlap warning with cautious mitigation copy |
| Late young WR with no role | Age can look like generic upside if used naively | No standalone youth bonus; role/path evidence is required |
| Late contingent RB vs floor veteran | Small low-end projection/ADP differences dominate | Bounded late-phase utility can reorder Recommendation while Pick Score remains unchanged |

## Root causes, deliverables, and calibration risks

### Root causes found

1. Completed construction is count/target based while live roster utility is
   role- and format-aware.
2. The grade has starter strength but no explicit functional-depth/floor or
   evidence-based upside model.
3. Deep Dive reconstructs market context and Board PS, not the roster-aware
   alternatives passed at each pick.
4. Reach/steal cards select extrema after permissive classification instead of
   requiring card-level significance.
5. Internal names and mixed surface scales obscure the otherwise intentional
   distinction among Recommendation, absolute Pick Score, Board PS, and Draft
   Grade.
6. Bye warnings count bodies instead of impact.
7. Absolute Pick Score already fades ADP late, but Recommendation has no explicit
   smooth phase policy for asymmetric bench value.

### Files expected to change during implementation

- `static/draft_board_core.js`
- `static/draft_room.js`
- `static/draft_grade_team.js`
- `utils/draft_grade.py`
- `data_building/draft_grade_backtest.py`
- focused Draft Room/core/grade/parity/backtest tests and redraft fixtures
- `static/pick_score.js` and `utils/pick_score.py` only if evidence justifies an
  absolute Pick Score change; otherwise they remain deliberately untouched

### Results status

This document is an implementation plan, so it does not claim post-change
backtest results. The current focused baseline is recorded by the commands in
the change that introduced this plan. Implementation work must report the exact
formula values shipped, before/after fixture deltas, sample sizes, correlations,
tests added, and any format-specific tradeoffs.

### Areas requiring real-world calibration

- role/path metadata availability and consistency across player sources;
- thresholds for primary bench and upside-confidence classification;
- round-decay exponent, fringe cap, and the sensitivity bounds users perceive
  as “very little” versus “material”;
- historical Decision Score tolerance and reach/steal significance gates;
- QB/TE streamability and cover relief in unusual league sizes/settings;
- bye severity bands;
- maximum late-phase upside bonus and its effect on mock realism;
- whether bench upside predicts outcomes well enough to enter Draft Grade rather
  than remain advisory commentary.

These constants should live together near their pure helpers, be named, be
covered by fixtures, and be adjusted through the documented validation loop—not
scattered through rendering code.
