# Backtest Improvement Plan

Based on your 2016–2025 backtest output, your model has strong global signal (Pearson ~0.51–0.55 across tiers), but improvements should focus on **tail errors** (big misses) and **position-specific calibration**.

## 1) What is already working

- Cross-year correlation is solid, especially when at least 2–3 complete seasons are available.
- QB and RB currently show strongest positional correlation.
- CFBD data is adding measurable lift relative to no-college-data rows.

## 2) Highest-impact problems to fix

### A) WR false positives are costly
Examples from your report (Andy Isabella, J.J. Arcega-Whiteside) suggest your WR model is overweighting production/draft-capital combinations that miss on NFL translation.

**Fixes to implement:**
- Add WR penalty terms for low early-career separation proxies (where available) and contested-catch-only profiles.
- Add interaction features, not just thresholds:
  - early declare × draft capital
  - yards/route run proxy × age
  - target share × team pass volume context
- Train WR with a robust loss (Huber or quantile) to reduce sensitivity to outlier misses.

### B) Underrating Day-2/Day-3 WR breakouts
Large underrates (Kupp, Nacua, Renfrow archetypes) imply underweighting skill indicators when draft capital is modest.

**Fixes to implement:**
- Build a second-stage "outperform draft slot" model for WR/RB that predicts probability of beating expected value from draft pick.
- Blend final score as:
  - base talent score
  - draft-capital prior
  - outperform-prior residual score

### C) TE instability across classes
TE accuracy variance is high (strong some years, weak in 2021/2024).

**Fixes to implement:**
- Increase TE-specific shrinkage toward draft-capital + age priors when sample is sparse.
- Prefer route/receiving usage features over raw season totals.
- Use Bayesian smoothing for low-N TE thresholds to avoid noisy benchmarks.

## 3) Modeling upgrades

### A) Convert single regression into multitask outputs
Predict:
1. probability of top-tier hit (top-6 QB/TE, top-12 WR/RB)
2. expected 3-year cumulative PPR
3. expected peak season

Then combine these with position-specific weights.

### B) Time-aware training
- Use rolling-origin validation by draft class (train <=Y, validate Y+1).
- Refit annually to avoid stale weights.
- Report variance of metrics across folds, not just pooled metrics.

### C) Position-specific models
Keep separate models for QB/RB/WR/TE end-to-end (features + hyperparameters + calibration), then normalize outputs to a unified draft board score.

## 4) Data quality and leakage controls

Your logs show repeated 404s for future nflverse season files in backtests. This can introduce inconsistent label windows.

**Fixes to implement now:**
- Hard cap evaluation labels to `min(current_season - 1, draft_year + horizon)`.
- Treat missing future-season files as expected, not as data failures.
- Save per-player "seasons observed" and train/evaluate with censoring-aware weights.

## 5) Evaluation changes (optimize for decisions, not only correlation)

Add these metrics by position and year:
- Precision@10, Recall@10
- NDCG@k (k = 10, 25)
- Hit-rate lift in top decile vs baseline
- Calibration error (Brier/ECE) for hit-probability outputs

This better aligns to draft decisions than Pearson alone.

## 6) Concrete 2-week execution plan

### Week 1
1. Add censoring-safe label builder and remove future-file 404 dependence.
2. Split pipeline into position-specific training/eval.
3. Add rolling-origin CV reports.

### Week 2
1. Implement multitask heads (hit prob + cum PPR + peak).
2. Add residual "beat draft capital" model for WR/RB.
3. Tune blending weights using only out-of-fold predictions.
4. Re-run 2016–2024 backtest and compare to current baseline.

## 7) Targets for next iteration

Reasonable first-step goals:
- Lift overall Pearson from ~0.514 to >=0.54 on all-years-including-partial tier.
- Raise WR positional rank accuracy from +0.51 to >=+0.56.
- Raise TE positional rank accuracy from +0.47 to >=+0.52.
- Improve Top10 hit average by +1 correct player per class.

If you want, I can turn this into concrete code changes next: feature list, schema updates, training loop changes, and a reproducible experiment matrix.
