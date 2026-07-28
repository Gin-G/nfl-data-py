# Model experiment ledger

Every model change is measured on the same yardstick and kept only if it helps.
Revert anything that regresses without a compensating gain.

**Benchmark:** leakage-free backtest of **season 2025, weeks 1–18** (train on
2020–2024), epochs 60, single seed. Primary metric = **MAE** of the
fantasy-points point projection (lower is better); secondary = correlation.
Single-seed deltas under ~0.05 MAE are noise — confirm across seeds before
trusting small position-level moves.

Rebuild the dataset first (`build-data`) so `passing_interceptions` scoring is
correct. Dataset used: 2018–2025, 187,534 rows.

| # | Change | MAE | corr | vs prev | Decision |
|---|--------|----:|-----:|--------:|----------|
| 0 | Baseline (single last-game features, MSE) | 4.40 | 0.580 | — | reference |
| 1 | Fix `passing_interceptions` col + use `team` as a feature | *(folded into #2)* | | | **kept** — correctness |
| 2 | Rolling / usage features (last-3, last-5 + trend) | **4.17** | **0.611** | −0.23 | **KEPT — new default** |
| 3 | Huber loss (vs MSE) | 4.15 | 0.610 | ~0 (worse booms) | **reverted** — wash; keep MSE |
| 4 | Coarse opponent (pts allowed by position) on rolling | 4.32 | 0.614 | +0.15 | **opt-in only** — hurts default |
| 5 | pbp scheme splits + usage tendencies | 4.33 | 0.595 | +0.16 | **opt-in only** — no gain |
| 6 | Quantile model (floor/median/ceiling) | median 4.15 | — | ≈ #2 | **KEPT** — additive; calibration TODO |
| 7 | Volatility features (FP std, boom/bust rate) | mean 4.29 | 0.614 | +0.12 | **REVERTED** — regressed mean, no variance gain |
| 8 | Quantile conformal calib — additive (internal split) | median 4.10 | — | band 70.7→75.6% | ~no-op (noise) |
| 8b | Quantile conformal calib — multiplicative scaling | median 4.16 | — | band →36.8% | **REVERTED** — cal set too optimistic, shrank bands |
| 8c | Quantile conformal calib — additive (season holdout) | median 4.12 | — | band 71.8% | **KEPT** — principled, harmless (default) |
| 9 | Projector wiring: floor/median/ceiling in `predict` output | — | — | — | **KEPT** — `--quantiles` flag |
| 10 | Component-derived FP (score predicted stat components) | 4.83 | 0.555 | +0.63 | **REJECTED** — QB catastrophic (6.9→13.1); skill marginally better |
| 11 | Wider quantile band q05/q95 | median 4.14 | — | band 72→**81%** | **KEPT (option)** — ceiling well-calibrated; floor still high |
| 12 | FPA as an explicit post-projection multiplier | 4.21–4.41 | — | +0.004 → +0.21 | **REJECTED** — monotonically worse; best setting is "off" |
| 13 | Lineup backtest: ceiling vs mean objective | — | — | ceiling −7 pts/wk | **mean wins** — ceiling scored lower, same variance |

**Notes**
- #4/#5: matchup info doesn't help the single-game *mean* projection — the
  player's own recent usage/form already carries the predictable signal. Code
  stays in the repo, opt-in (`--opponent`), for future component-level work.
- #6: median projection is as accurate as the mean model (4.15 ≈ 4.17); it adds
  a floor (q10) and ceiling (q90). Open items on it, tracked below.

## Findings
- **Band width is ~90% a function of the median — and that's mostly correct.**
  In fantasy, variance scales with scoring level (a 20-pt player has more room to
  boom/bust than a 5-pt player). The quantile model already captures the residual
  player-specific variance (partial corr 0.44) about as well as this data
  supports; volatility features (#7) added noise, not signal. The remaining
  problem is *absolute* calibration, not per-player differentiation.

## Calibration finding (#8)
The [q10,q90] band empirically covers **~72%** of outcomes, not the nominal 80%
(floor a bit high). None of the conformal variants fixed it: additive offsets are
a near no-op (the calibration set is drawn from training, where the model is
already well-calibrated, so offsets ≈ 0), and multiplicative scaling *shrank* the
band because that in-era calibration set is too optimistic. Root cause is
**cross-season generalization decay** — 2025 is genuinely harder than any
in-training holdout, and an internal calibration set can't see that gap. Kept the
principled season-holdout additive version (harmless, median MAE unaffected).
**Label the band as an empirical ~72% interval, not 80%.** Options to widen if
wanted: a safety-factor inflation, or train wider nominal quantiles (q05/q95).

## Component-level finding (#10)
Recomputing FanDuel points from the model's *predicted* stat components (vs its
direct FP output) is **worse overall** (+0.63 MAE), driven by a QB blowup
(6.9 → 13.1): component errors compound through the QB scoring weights
(pass TD ×4, 300-yd bonus). Skill positions were marginally *better* (RB −0.01,
TE −0.06, WR −0.09, all within seed noise). Verdict: the direct FP target wins;
a full separate-heads architecture is **not justified** by this evidence — the
shared model already predicts FP better than scoring its own components, and QB
specifically needs the blended target. Kept direct FP; `derived` column stays in
backtest output as a diagnostic only.

## Wider band finding (#11)
q05/q95 gives a practically wider, more useful range: **~81% coverage** (vs ~72%
for q10/q90) with the ceiling well-calibrated (q95 empirical 0.909). The floor
stays too high (q5 empirical 0.21 vs 0.05) — the model still won't foresee 0-pt
busts, worst for QB (72% band coverage). Reasonable default for a floor/ceiling
range; pass `quantiles=(0.05,0.5,0.95)`.

## FPA / matchup — tested three ways, all negative (#4, #5, #12)
"Fantasy points allowed to a position" is the opponent feature set. It has now
failed to help the single-game point projection through three independent
mechanisms: as a model input feature (coarse #4 and pbp-scheme #5), and as an
explicit post-projection multiplier (#12, the way analysts apply it — worse at
every strength, best at "off"). Consistent conclusion: at single-game
granularity the matchup signal is dominated by week-to-week variance, and the
player's own recent usage already carries the predictable part. FPA is noisy and
not strength-of-schedule adjusted. Not a mean-accuracy lever here.

## Lineup backtest (#13)
Built mean- vs ceiling-optimized skill lineups for 2025 wk4-18 (salaries priced
off each player's trailing avg_fppg, leakage-free) and scored them by actual
results. Mean objective won on every metric: top-1 lineup 81.5 vs 74.5, best-of-5
97.8 vs 92.2, and identical score volatility (std 20.6 vs 20.7). Ceiling beat mean
in only ~6/15 weeks. So the ceiling objective is just a noisier mean here — no
upside payoff. Root cause ties back to the quantile model differentiating player
variance only weakly (ceiling ~0.9 correlated with the median), so "ceiling" ≈
"noisy mean". **Default the optimizer to `mean`.** Caveats: proxy salaries
(priced off form) can't show the real-market mispricings a ceiling strategy might
exploit, and it's a single season/seed — real FanDuel salaries could change the
GPP picture. Keep floor/ceiling available as options.

## Roadmap item 1 — naive baselines, GBDT, multi-seed (2025 wk1-18)

The most important measurement so far — it recalibrates every prior claim.

**Naive baselines** (no model, just look up the player's number):
| baseline | MAE | corr |
|---|---:|---:|
| predict last game | 4.976 | 0.478 |
| predict last-3 avg | 4.338 | 0.573 |
| predict season avg | 4.336 | 0.598 |
| **predict last-5 avg** | **4.255** | 0.595 |

**Model vs that floor:**
| model | MAE | corr |
|---|---:|---:|
| GBDT (LightGBM, same features) | 4.240 | 0.605 |
| NN, mean of 5 seeds | 4.237 ± 0.052 | — |
| NN, 5-seed **ensemble** | **4.201** | 0.618 |
| 0.5·NN-ensemble + 0.5·GBDT | 4.197 | 0.618 |

**Findings that change how we work:**
- **The whole model beats "average the last 5 games" by only ~0.05 MAE** (typical
  seed 4.24 vs naive 4.255; the ensemble stretches it to ~0.05). Single-game
  fantasy scoring is near-irreducibly noisy — this is the real denominator, and
  it means new features must add genuinely NEW info (Vegas, xFP, role changes)
  to move the needle, not just re-encode recent form.
- **The "4.17" we kept quoting was a lucky seed.** Seeds range 4.17–4.30
  (4.237 ± 0.052). So single-seed deltas of ±0.05 in earlier experiments were
  noise — the ledger's caveat was right; treat past small wins/losses as ties.
- **Adopt seed-ensembling as the default** — a reliable ~0.04 MAE gain over a
  typical single seed and it kills the seed lottery. The production model should
  be an ensemble.
- **GBDT is a strong production candidate**: 4.240, trains in ~3s, **no
  TensorFlow** (would let the NFL-API projections job drop the runtime TF
  install), and gives feature importances. Blending it with the NN ensemble is
  marginally best (4.197).
- **Top GBDT importances** confirm the backbone: `fppg_trend`,
  `fanduel_fantasy_points_roll5`, `avg_fppg`, `fanduel_fantasy_points_roll3`,
  `performance_vs_average`, `snap_trend`, `offensive_snap_pct_roll5`,
  `fantasy_per_snap`, `opportunity_score`, `target_share_roll5` — recent scoring
  + snap/usage trend. Validates ROADMAP's "move expected volume/usage" thesis.

## Where things stand
Naive last-5-avg is 4.255; our best (NN-ensemble + GBDT blend) is 4.197 — a real
but small edge. Point projection (rolling features, #2) looks near the
practical ceiling for this data/architecture: opponent/scheme (#4/#5),
volatility (#7), and component-scoring (#10) all failed to beat it. Quantile
floor/median/ceiling works and is wired in; the median matches the mean model,
the ceiling is trustworthy, the floor is the weak spot. Remaining honest levers:
per-position models, richer data (snaps/routes/betting lines), or accept current
accuracy and focus on usage (DFS optimizer, weekly workflow).
