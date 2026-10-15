# Model experiment ledger

Every model change is measured on the same yardstick and kept only if it helps.
Revert anything that regresses without a compensating gain.

**Benchmark:** leakage-free backtest of **season 2025, weeks 1–18** (train on
2020–2024), epochs 60, single seed. Primary metric = **MAE** of the
fantasy-points point projection (lower is better); secondary = correlation.
Single-seed deltas under ~0.05 MAE are noise — confirm across seeds before
trusting small position-level moves. For anything about ORDERING, MAE is blind
by construction — use `evaluate.rank_metrics` (#14).

> **Benchmark correction, 2026-08-11 (#21).** Until this date `utils.regular_games`
> did not filter `season_type` despite its name, so the training set, the backtest
> population and board ground truth all included **postseason weeks 19–22** — 279
> of 6,126 rows for 2025. Every row below numbered #0–#20 was measured on that
> contaminated population. It was not leakage and every A/B shared the flaw across
> both arms, so the RELATIVE deltas stand; the ABSOLUTE numbers are ~0.03 MAE off.
> Corrected baseline for the current shipped model: **MAE 4.187 over 5,844 rows,
> weeks 1–18.** Compare new work against that, not against 4.154.

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
| 14 | Rank-aware metrics (Spearman, top-12/24 overlap, #1/#5 & #1/#24 spread) | — | — | — | **KEPT** — measurement only; MAE structurally cannot see ordering |
| 15 | Board team-budget cap: league mean → p90 team-season | 4.138 | — | +0.044 vs mean cap | **KEPT** — buys the ordering back (see below); MAE cost accepted |
| 16 | Is seed averaging compressing the elite tail? | 4.154 | — | members 4.130–4.181 | **REJECTED** — ensemble spread sits inside its members' range |
| 17 | Blend each component with its own trailing-5 (not the points ratio) | unchanged | — | component MAE −30–45% | **KEPT** — display-only; points untouched |
| 18 | Gate the form blend on trailing-window depth | 4.138 / 3.625 | — | 2025 −0.008, 2024 −0.0001 | **KEPT as a guard rail** — NOT a measured win; never worse |
| 19 | Snap-share role multiplier (vs the depth-rank table) | 3.748 | — | MAE −0.066, WR spearman .946→.804 | **REJECTED** — buys MAE with bias, damages ordering |
| 20 | Preseason-board harness (`backtest_season_board`) | — | — | QB spearman 0.27, WR top12 0.42 | **KEPT** — the missing benchmark for every volume mechanism |
| 21 | `regular_games` now excludes the postseason | **4.187** | — | was 4.154 over a contaminated population | **KEPT** — correctness; new baseline |
| 22 | LIVE: 2026 week 1 published board vs actuals (first prospective week) | 4.502 pub / 4.291 cond | ρ 0.690 | naive 4.739; pub ties it, cond −0.25 | **measurement** — availability discount leaked into a weekly board |
| 23 | In-season path, first real run: season auto-detect + depth snapshots + injury/rookie fixes | — | — | week 2 dry run: 5 bugs found, 603-row board | **KEPT** — correctness; the weekly loop now sees the new season |
| 24 | LIVE: weeks 1-2 scored prospectively, first weeks on the fixed path | wk2 3.870 | ρ 0.62 | wk1 4.485 → wk2 3.870; skill +0.07 → +0.65 | **measurement** — the in-season fixes hold up out of sample |
| 25 | LIVE: four weeks scored; and does correcting the level help? | wk2-4 3.942 | ρ 0.680 | skill +0.312 over naive; level correction +0.09/+0.13 MAE | **measurement + REJECTED** — the negative bias is the skew, not an error |
| 26 | Train on the CURRENT SEASON ONLY (the four weeks in hand) | 4.217 (5 seeds) | ρ ~unchanged | +0.097 vs history-only; +0.068 retrained weekly, 12/14 weeks lost | **REJECTED** — the season's signal is in the FEATURES, not the training rows |
| 27 | How deep should history go? min_season 1 vs 2 vs 3 vs 6 prior seasons | 4.110-4.126 | ρ ~unchanged | whole spread 0.016 — a tie | **measurement** — depth past one prior season buys nothing, and costs nothing |
| 28 | History depth RETRAINED WEEKLY (what #27 could not ask) | 4.0355 vs 4.0338 | — | pooled delta +0.002; weekly deltas swing ±0.07 | **measurement** — a tie, and the week-to-week swings are the real finding |

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
- **SHIPPED (2026-08-05):** seed-ensembling is now the default everywhere —
  `model.train_ensemble(df, n_seeds=5)` featurizes once and fits 5 networks
  sharing one preprocessor; `predict_batch` averages their raw outputs before
  capping. `ProjectionService(n_seeds=...)`, `evaluate.backtest(n_seeds=...)`
  and CLI `--seeds` all default to `config.DEFAULT_N_SEEDS` (5); `train_model`
  stays single-network and gained a `seed` arg for reproducibility. Saved
  ensembles carry `n_members` in metadata.json (dirs written before this load
  as single models). Costs 5x training time. NFL-API's projections cronjob
  picks it up via the main-tarball install; `--seeds 1` for a fast run.
- **GBDT is a strong production candidate**: 4.240, trains in ~3s, **no
  TensorFlow** (would let the NFL-API projections job drop the runtime TF
  install), and gives feature importances. Blending it with the NN ensemble is
  marginally best (4.197).
- **Top GBDT importances** confirm the backbone: `fppg_trend`,
  `fanduel_fantasy_points_roll5`, `avg_fppg`, `fanduel_fantasy_points_roll3`,
  `performance_vs_average`, `snap_trend`, `offensive_snap_pct_roll5`,
  `fantasy_per_snap`, `opportunity_score`, `target_share_roll5` — recent scoring
  + snap/usage trend. Validates ROADMAP's "move expected volume/usage" thesis.

## Roadmap follow-up — component-first models (the fair test)

The stronger version of experiment #10: train models on the component stats
ONLY (no fanduel_fantasy_points target competing for capacity), then compute FP
from the predicted stats via the FanDuel formula. 5-seed ensembles vs the direct
5-seed ensemble, 2025 wk1-18.

| position | direct-FP | component-first | delta |
|---|---:|---:|---:|
| **ALL** | **4.201** | 4.329 | +0.128 |
| QB | 7.067 | 7.287 | +0.219 |
| RB | 4.333 | 4.416 | +0.083 |
| WR | 3.929 | 4.062 | +0.133 |
| TE | 3.097 | 3.220 | +0.124 |

**Verdict: direct-FP wins at every position — decisively.** And the component
model is ~5x more seed-sensitive (single-seed 4.529 ± 0.236 vs direct
4.237 ± 0.052), because component errors compound through the scoring weights
(TDs ×4–6, yardage bonuses) instead of cancelling. This *strengthens* #10: the
earlier "skill positions slightly better" was single-seed noise — with proper
5-seed ensembles the component approach is worse everywhere, position-routing
(direct QB + component skill) included (4.305). Corr identical (0.618).

**BUT the component predictions still have value we should use**: the model is
already multi-output (it predicts the stats too), so EXPOSE the per-stat
projections for props (over/under yards, receptions, anytime-TD) and to let
matchup/coaching features target the right component — while keeping direct-FP
as the headline fantasy projection. Predicting-stats-then-scoring is not the way
to compute the FP headline; it's the way to serve everything *around* it.

## Coaching / new-HC regime change (probe, 2025) — bias not MAE

Question: does the model mis-project players on teams with a NEW head coach (old-system
history no longer applies)? Identified 7 new-HC teams in 2025 (CHI Ben Johnson, DAL
Schottenheimer, JAX Coen, LV Carroll, NE Vrabel, NO Moore, NYJ Glenn) vs 25 continuity,
scored the 5-seed NN ensemble split by new-HC × early(wk1-4)/late.

| slice | n | MAE | bias (pred−act) |
|---|--:|--:|--:|
| new-HC wk1-4 | 283 | **4.071** | **−0.451** |
| continuity wk1-4 | 1037 | 4.175 | −0.007 |
| new-HC wk5-18 | 993 | 4.153 | −0.201 |
| continuity wk5-18 | 3536 | 4.232 | −0.267 |
| new-HC wk1-4 **QB** | 32 | 7.378 | **−1.729** |
| new-HC wk1-4 TE | 61 | 2.823 | −0.413 |

**Verdict: no MAE degradation** — new-HC teams are if anything slightly *more* accurate
early (excess-early-gap attributable to the new coach = **−0.025 ≈ 0**). The rolling
last-3/last-5 features already adapt to the new system within a few games. **BUT a real
directional BIAS**: the model *under-projects* new-HC offenses early (−0.45 overall,
−1.73 for QB) — 2025's new coaches ran more productive systems than old-system history
implied, and the history-anchored model shoots low. Same pattern as opponent/scheme/FPA
(#4/#5/#12): a genuine signal that can't beat variance-dominated single-game MAE, so it's
**not an accuracy lever**. Its real value is a *targeted early-season prior* for the ONE
use case where rolling features are useless — **week 1 of a season for a new-coach team**
(2026 wk1: features still carry 2025's system). Thin sample (7 teams, 32 QB obs); treat as
a small bias correction / informative prior, not a model-wide feature. Would apply only to
new-HC teams, only wk1-~4, decaying as real usage accrues.

## Coaching — do prior-job tendencies carry over? (foundation test)

Before building a wk1 coach prior, tested its core assumption: do a new coach's
PRIOR-job team tendencies predict his NEW team's 2025 behavior better than the
outgoing coach's tendencies (the old system the model is anchored to)? Curated
play-calling history for the 6 new-2025 HCs with offensive history (Glenn/NYJ is
defensive -> no offense prior), computed neutral-situation tendencies from PBP.

| tendency | |prior−actual| | |old−actual| | prior wins? |
|---|--:|--:|:--|
| pass_rate (run/pass balance) | 0.022 | 0.027 | YES |
| te_tgt_share | 0.042 | 0.059 | YES |
| top_rb_share (bellcow vs committee) | 0.148 | 0.204 | YES |
| plays_pg (pace) | 2.48 | 3.39 | YES |
| qb_rush_share | 0.045 | 0.033 | **no** |

Pooled normalized error: prior 1.087 vs old-system 1.300 (~16% better); prior beats
old on 57% of coach-metric cells. **Verdict: the premise holds for SCHEME traits —
run/pass balance, TE usage, RB committee split, pace carry over from a coach's prior
jobs. QB rush share does NOT** (it's personnel-driven — the coach adapts to his QB;
never apply a coach prior to QB rushing). Big caveat: tiny sample (6 coaches) and much
2025 deviation is PERSONNEL the coach acquired (LV top_rb_share 0.87 = rookie bellcow
Jeanty), which a scheme prior can't model. So a wk1 prior is justified for pass
rate / TE share / RB split / pace, applied only to new-coach teams, only wk1-~2,
decaying as rolling usage accrues; fall back to league avg (+ Vegas total for scoring
level) when a coach has no play-calling history. Modest, niche (wk1-only) — a bias
corrector for the one case rolling features can't cover, not an MAE lever.
Scripts: scratchpad/coaching_probe.py, coach_tendencies.py.

## Coaching — wk1 scheme prior applied (nfl_projections/coaching.py) — near-neutral

Built the prior the foundation test justified: per-team volume multipliers from a new
coach's prior-job tendencies (pass/rush balance, TE share, RB-committee, pace; QB rush
excluded), applied to new-coach teams' wk1-2 component projections, adding only the FP
*delta* to the direct-FP headline (decay wk1=1.0, wk2=0.5). Tested on 2025 wk1-2,
3-seed ensemble, teams with a play-calling prior (6, ex-NYJ).

| | n | MAE | bias |
|---|--:|--:|--:|
| baseline | 116 | 4.136 | −0.488 |
| coach-adjusted | 116 | 4.128 | −0.424 |

**Verdict: WASH — not shipped as default.** MAE −0.008 (noise); bias improved the right
direction but only ~13% of it (−0.49 → −0.42). Same wall as #4/#5/#7/#12: the ~0.45
systematic bias is trivial against ~4.1 single-game MAE, and correctly-scaled scheme
multipliers (league baseline near 1.0) move volume too little to matter. **Debugging
note:** a first run showed MAE +0.09 / bias −1.6 — an ARTIFACT of loading 2024 PBP twice
(coach history already contains 2024 seasons), which doubled the league-baseline
`plays_pg` (64 vs 32) and drove every multiplier to the 0.6 clip. Dedup the PBP load
(one row per game_id/play_id) — real baseline plays_pg ≈ 32. Lesson: always sanity-check
the reference-frame scale before trusting a multiplier feature.
`coaching.py` stays in the repo, **opt-in / diagnostic only** (like opponent) — its
tendency engine is genuinely useful as *analytics* (how a new coach's system differs),
just not as a point-projection lever. The rookie draft-capital angle (below) targets the
PERSONNEL effects this scheme prior explicitly can't (LV top_rb_share 0.87 = Jeanty) and
is the more promising untested lever for the new-regime wk1 problem.

## Rookie draft-capital prior — foundation is STRONG (unlike coaching)

Rookies are excluded from the standard backtest (no prior game) and the Projector falls
back to a crude ×0.15/0.4 hack. Tested whether draft capital predicts rookie early-season
(wk1-4) FanDuel PPG, 2011-2024 rookies:

| pos | R1 | R2 | R3 | R4-7 | corr(pick#, early ppg) |
|---|--:|--:|--:|--:|--:|
| QB | 14.4 | 9.4 | 8.2 | 7.7 | −0.41 |
| RB | 11.9 | 7.8 | 7.0 | 3.6 | −0.48 |
| WR | 9.4 | 5.6 | 4.0 | 2.4 | −0.50 |
| TE | 6.4 | 3.6 | 3.3 | 1.9 | −0.48 |

**Clean monotonic gradient every position; corr −0.41…−0.50 (strong for single-game FP).**
A 1st-round RB scores 3.3x a Day-3 back early. This is the opposite of the coaching wash:
coaching tried to shave a 0.45 bias off players the model already handles (drowned by
variance); rookies have NO history, so this replaces a guess with a calibrated
expectation — a large signal for a group the model currently punts on. 1st-round RB analog
class (Saquon 20.0, Zeke 18.1, Fournette 18.5, Bijan 16.7 early) is the reference for
projecting a 2026 first-rounder (e.g. Jeanty). draft_picks (nflreadpy) has round/pick/
gsis_id and already includes the 2026 class. NEXT: build a rookie prior (expected PPG +
component split by position × draft capital, refine by depth-chart role) and test it vs
the current rookie hack on 2023/2024 rookies (calibrate on earlier years).
Script: scratchpad/rookie_foundation.py.

## Rookie draft-capital prior — BUILT & WINS (nfl_projections/rookies.py)

RookiePrior.fit fits per-position ppg ≈ a + b·log(pick) on historical rookies (wk1-4),
with a q10/q90 residual band and a typical rookie component mix. Out-of-sample test
(fit on years < test, evaluate on rookies who played), vs a no-pick position mean and
the old tier-multiplier heuristic:

| year | draft-capital prior | position mean | tier heuristic |
|---|--:|--:|--:|
| 2023 (n=58) | **3.518** | 4.232 | 3.607 |
| 2024 (n=48) | **3.026** | 3.966 | 3.399 |

**KEPT — a real ~0.7-0.9 MAE win over ignoring draft capital, and beats the current
Projector hack.** (Update: added `SHRINK=0.6` regression toward the positional rookie mean
— the raw log-pick curve is survivorship-inflated at the very top, so a pick-1 QB (17.4)
and pick-3 RB (17.1) out-projected elite veterans in 2026 wk1. Shrunk: Mendoza 14.8 < Allen
16.1, Love 12.5 < Saquon 13.2. Costs ~0.2 rookie MAE (2024 3.03→3.23, still >> no-pick 3.97)
for realistic cross-population ranking — the right trade for uncertain rookies.)
Wins on RB/WR/TE every year; QB is noisy (n=4-6, starting job is
binary — weakest signal). Band (q10/q90 residuals) covers 60%/69% of actual (nominal
~60%, well-calibrated). This is the opposite of the coaching wash BECAUSE rookies have
no history — the prior replaces a guess, not a small bias on an already-good estimate.
Honest limits: can't foresee a 7th-round breakout or a 1st-round bust; it's the expected
value given draft capital, a starting picture to be superseded by real usage. Next: wire
into Projector.predict_rookie (replace the ×draft/depth/position multiplier hack).

## Why opponent/matchup keeps failing — defense-vs-position is weak & unstable

Foundation test for a proposed Monte-Carlo game simulator: is "defense vs position" a
stable, predictable team trait (Dolphins reliably softer than Texans) or week-to-week
noise? FP allowed by each defense to each position, 2022-2025:

| pos | split-half r (early→late DvP) | defense spread CV | player game-FP R² from defense |
|---|--:|--:|--:|
| QB | 0.224 | 0.131 | 0.045 |
| RB | 0.254 | 0.159 | 0.019 |
| WR | 0.057 | 0.125 | 0.009 |
| TE | 0.073 | 0.191 | 0.019 |

**Root-cause for #4/#5/#12:** (1) defense-vs-position barely persists within a season
(split-half r 0.06-0.25; WR/TE ≈ noise) — injuries, scheme, and offenses-faced wash the
ranking out; (2) defenses differ only ~13-19%; (3) the defense faced explains just
1-4.5% of a player's single-game FP. So a from-scratch simulator estimating defense
strength from PBP history would manufacture confident-looking noise on the MEAN — the
matchup signal is real over a full season but tiny and unstable per game. A simulator's
surviving value is **distributions + player correlation (stacks)** for DFS, not mean
accuracy. The reliable way to get matchup-dependent team scoring is **Vegas implied
totals/spreads** (market prices defense/injuries/weather/pace, far more stable than our
DvP) — anchor team scoring to the total, distribute by usage share, optional MC layer for
tails/correlation. Script: scratchpad/defense_stability.py.

**Correction — opponent-ADJUSTED TEAM grades (the fair test):** the DvP test above was
unadjusted for schedule. Opponent-adjusted SRS-style TEAM ratings (2021-2025) split-half
stability: offense **0.489**, defense **0.264** (raw defense 0.141 — adjustment ~doubles
it). **Offense is a stable, predictable team trait; defense is noisier but adjustment
helps.** A team's own offense is already in a player's rolling form; the opponent-*defense*
side is the weak matchup link (r 0.26, ~1-4.5% of player FP). Verdict: a 0-100
opponent-adjusted team grade (50=avg), updated weekly with regression to a preseason
prior, is a LEGIT interpretable rating — valuable for UI, game context, and as a
team-scoring anchor for matchup-varying SEASON projections, but it won't add much per-game
player accuracy on the defense side. 2025 grades sane (off 21-79, def 15-78; top off
LA/SEA/IND/BUF/JAX, top def SEA/HOU/PHI/MIN/DEN). Script: scratchpad/team_ratings.py.

## Player grades — opportunity-weighted, validated (nfl_projections/player_grades.py)

Foundation (2022-2025, split-half stability + predicts-rest-of-season-FP by position):
OPPORTUNITY metrics are both the most stable AND most predictive — WR target_share/wopr
~0.85 stable / ~0.76 predict; RB carries 0.83/0.75; TE wopr 0.85/0.72; QB carries
0.79/0.44 & epa 0.41/0.40. EFFICIENCY (EPA/play, RACR, YAC) is noise for skill positions
(stability ~0). Production (fp) predicts 0.71-0.79 but is less stable than opportunity.
→ grade = opportunity 0.45 / production 0.45 / efficiency 0.10, z-scored WITHIN position
→ 0-100 (50=avg), preseason-prior-blended (last yr regressed 60% to 50, worth ~5 games).

Validation (grade thru wk9 → wk10-18 FP, WITHIN position — the fair test; the pooled "ALL"
number is deflated because grades are position-relative by design, removing cross-position
scale): WR grade 0.78 vs raw-FP 0.73; TE 0.81 vs 0.74 (grade WINS both); RB 0.86 vs 0.84;
QB comparable. So the grade predicts as well as/better than raw production AND is more
stable + position-comparable. Face validity excellent (2025: RB CMC 87/Bijan 84/Gibbs 80;
WR Puka 87/JSN 86/Chase 82; TE McBride 91/Bowers 82; QB Allen 72/Hurts 67). Scripts:
scratchpad/player_grade_foundation.py, player_grades_test.py. Snap-share (offense_pct) is a
natural add but needs a gsis↔pfr crosswalk; wopr/target_share already proxy participation.

## Vegas implied team total — the definitive matchup test — WASH (not kept)

The most-cited untested lever: does the MARKET's implied team total (total_line/spread_line
from schedules, 100% populated 2015-2025, leakage-free) improve the player point
projection? Added implied_team_total/game_total_line/team_spread as features, trained with
them, INJECTED the current week's line at eval (not the prior game's), A/B vs baseline,
3 seeds, 2025 wk1-18:

| | MAE |
|---|--:|
| baseline | 3.642 ± 0.033 |
| + Vegas implied total | 3.655 ± 0.032 |

**+0.013 — wash/slightly worse. Not kept** (reverted the features.py scaffolding). Even the
best possible matchup signal — the market prices in offense/defense/injuries/weather/pace —
can't beat the player's own rolling form at the single-game player level. Structural reason:
implied total predicts TEAM points, but player FP = team pts × share × efficiency × TD luck,
and the share/efficiency/TD variance is player-specific noise no game-level signal resolves.
**This closes the matchup avenue: DvP (#4/#5), FPA (#12), coaching, team-grade multiplier,
and now Vegas ALL fail on the mean.** The rolling-form model is at the practical ceiling for
MEAN single-game point projection. Remaining value is NOT in the mean: distributions +
player correlation (a usage-based simulator, for DFS), or per-position models (untested).
(Note: this custom harness's absolute MAE ~3.64 is lower than the standard 4.24 backtest
because it includes more low-usage players; the A/B is valid — same harness both arms.)
Script: scratchpad/vegas_test.py.

## Per-position models — shared model wins (last mean lever, closed)

Trained separate QB/RB/WR/TE models (each on only its position's history) vs the shared
multi-output model, 2025, 3 seeds:

| pos | shared | per-position | delta |
|---|--:|--:|--:|
| QB | 7.165 | 7.167 | +0.002 (tie) |
| RB | 4.312 | 4.359 | +0.048 |
| WR | 3.845 | 3.957 | +0.112 |
| TE | 3.045 | 3.229 | +0.184 |

**Shared model wins or ties everywhere — keep it.** Per-position models are worse (less data
+ no shared cross-position representation; TE, the lowest-data position, suffers most). This
closes the last untested mean lever. Combined with the Vegas wash, the shared rolling-form
model is confirmed at the practical ceiling for single-game MEAN projection. Remaining value
is distributions + player correlation (usage-based simulator, for DFS), not the mean.
Script: scratchpad/per_position_test.py.

## Usage-based Monte-Carlo simulator (nfl_projections/simulate.py) — distributions + stacks

Since the mean is at ceiling, built the simulator for what the mean can't give:
DISTRIBUTIONS and player CORRELATION. Each sim draws a shared team scoring environment
split into team/pass/rush factors (the PASS factor, shared by QB + pass-catchers, is what
stacks them), samples opportunity (Poisson×env×per-player idio), converts to yards
(Normal) and TDs (Poisson), scores FanDuel. Calibrated on 2025 (leakage-free, trailing-5
expectations, 1000 sims):

| pos | [p10,p90] coverage (target ~80) | mean bias |
|---|--:|--:|
| ALL | 75% | +0.37 |
| QB | 69% | +1.39 |
| RB | 74% | +0.19 |
| WR | 77% | +0.40 |
| TE | 79% | +0.03 |

QB<->WR1 correlation: **modeled +0.317 vs empirical +0.359** — the stacking signal is
captured. Skill-position bands are near-calibrated; QB bands are tightest (highest-variance
position) and QB shows a +1.4 mean bias = regression-to-mean from raw trailing expectations.
**Fix + production architecture: `simulate(..., mean_anchor="proj")` rescales each player's
sims so the MEAN matches the point model's (accurate, at-ceiling) projection, while the
simulator supplies the shape + correlation** — best of both. stack_distribution() sums the
same sims for QB+WR stack queries (correlation preserved). Refinements left: explicit
passing-TD→receiving-TD coupling (would push corr toward 0.36), QB downside (benchings).
Scripts: scratchpad/sim_calibration.py.
## #20-21 The preseason board finally has a benchmark — and it exposed a bad one (2026-08-11)

**#20 `evaluate.backtest_season_board(dataset, season)`.** Builds the whole season board for
`season` from data strictly before it, then scores it against that season's real finishes with
the same rank metrics as the weekly backtest (`rank_metrics(totals=...)`). It runs the real
`season.project_season` assembly — depth-role multiplier, availability model, team budget,
share model — not a reimplementation.

This is the benchmark those mechanisms always needed. Every one of them exists for a board
built before a snap is played, where the model's inputs are a season stale and a depth-chart
change is genuinely new information. Scoring them in-season, with current usage already in the
features, measures the wrong thing — which is why #19 could only reject snap share for the
in-season case.

Leakage handled: model trains on < season; team grades come from
`ratings.preseason_prior(season - 1)`, NOT `grades(season)`, which blends in that season's own
results; league average and the games model already used season - 1. Rosters and depth charts
for `season` ARE used — a real August board has them — which is mildly optimistic about final
cuts and is the one place this knows more than August would.

**2025 board from pre-2025 data, 244 players with >= 8 games:**

| pos | n | spearman | top12 | top24 | in-season spearman |
|---|--:|--:|--:|--:|--:|
| QB | 23 | **0.270** | 0.667 | — | 0.919 |
| RB | 65 | 0.868 | 0.750 | 0.833 | 0.977 |
| WR | 95 | 0.748 | 0.417 | 0.625 | 0.975 |
| TE | 61 | 0.680 | 0.583 | 0.792 | 0.970 |

A preseason board is a categorically harder problem than the weekly numbers imply: WR top-12
overlap of 0.417 means five of the top twelve receivers. Per-game rates (injury luck removed)
barely move it — QB 0.286, WR 0.765 — so this is ranking difficulty, not health variance.
Judge board-stage changes against THESE numbers, not the in-season ones.

**#21 The benchmark itself was wrong.** Building the harness surfaced that
`utils.regular_games` never filtered `season_type` despite its name, so postseason weeks 19-22
were in the training set, the backtest population and the board's ground truth. For 2025 that
is 279 of 6,126 rows; for 2024, 489. Not leakage — but a biased sample containing only playoff
teams, which over-weights good offenses and inflates any season total for a team that went
deep. It is the same defect fixed in `games.py` earlier the same day, in a different place.

Weekly backtest, same model, before -> after the filter:

| | with postseason | regular season only |
|---|--:|--:|
| rows | 6,126 | 5,844 |
| MAE | 4.154 | **4.187** |
| QB / RB / WR / TE spearman | .949 / .984 / .978 / .975 | .919 / .977 / .975 / .970 |
| QB / RB / WR / TE top12 | .917 / .833 / .750 / .750 | 1.000 / 1.000 / .583 / .833 |

The top-12 sets move most, because removing playoff points changes who the top twelve actually
were. Board metrics barely shifted (QB 0.267 -> 0.270, WR 0.748 both ways), so #20's conclusions
hold either way. Every A/B in rows #0-#20 shared the flaw across both arms, so relative deltas
stand; absolute numbers are ~0.03 MAE optimistic. **New reference: 4.187.**

## #19 Snap share as the role multiplier — rejected, and the multiplier itself is suspect (2026-08-11)

Idea: `roles.per_game_role_multiplier` is a hardcoded lookup by depth rank that the code itself
calls a "snap-share proxy" (WR 1.00/0.92/0.72/0.50/0.30 by rank). We have MEASURED snap share
at 100% coverage every season, so use the real number instead of the proxy.

**First, the diagnostic that reframes the whole thing.** The multiplier assumes the model
projects everyone as if they were a starter. It does not — it projects each player from HIS OWN
usage history, and `offensive_snap_pct_roll5` / `snap_trend` are among its highest-importance
inputs. 2024 backtest (n=11,198), actual divided by RAW model prediction, by depth rank:

| pos | rank | actual/predicted | multiplier applied |
|---|--:|--:|--:|
| RB | 2 | 1.12 | 0.80 |
| RB | 3 | 1.00 | 0.55 |
| WR | 2 | 1.09 | 0.92 |
| WR | 3 | 1.03 | 0.72 |
| TE | 2 | 1.01 | 0.55 |
| QB | 2 | 0.82 | 0.35 |

The model already under-projects backups slightly at nearly every rank. The multiplier then
cuts them a further 20-65%. **It is double-counting a correction the model already makes.**

**Three arms on the same 2024 predictions:**

| arm | MAE | bias | spearman QB / RB / WR / TE |
|---|--:|--:|---|
| no multiplier | 3.814 | −0.658 | 0.941 / 0.934 / **0.946** / **0.944** |
| rank multiplier (shipped) | 3.762 | −1.206 | 0.899 / 0.939 / 0.950 / 0.863 |
| snap-share multiplier | **3.748** | −1.532 | 0.926 / 0.873 / 0.804 / 0.883 |

Both multipliers improve MAE and both damage ordering, by pushing bias from −0.66 to −1.21 and
−1.53 — the same trade the league-mean budget cap made (#15). Snap share is the WORST for
ordering (WR 0.946 -> 0.804) precisely because the model has already consumed snap features.
**Rejected: do not replace the rank table with measured snap share.**

**What is NOT settled.** This is in-season evidence, where the model has current-season usage.
The multiplier's real justification is the PRESEASON board, where the model's inputs are a
season stale and a depth-chart change is genuinely new information the model cannot see.
Evaluating that needs a reconstructed 2025 preseason board scored against 2025 finishes; until
that exists, do not change the production multiplier on this evidence. The indicated fix, if it
holds up, is to apply the multiplier only where the depth rank DISAGREES with the player's own
usage history — which is exactly the pattern `shares.py` already uses (rank prior for movers,
own history for stable returners).

**Two data facts worth not re-deriving:**
- nflreadpy depth charts changed schema. 2018-2024 use `depth_team`; 2025+ use `pos_abb` /
  `pos_rank`. `DepthChartAnalyzer` expects the NEW format, so current seasons parse fine and
  the OLD ones raise KeyError — the opposite of what it looks like from the dataset, where
  `depth_team` is 94-97% covered through 2024 and 0% for 2025.
- That empty `depth_team` column is NOT a model feature, so nothing is silently degraded by it.

## #18 Thin-history gating — a negative result worth writing down (2026-08-11)

Concern raised: a history-weighted projection should badly misjudge a player whose history
is unrepresentative — Jonathon Brooks, two NFL seasons, two ACL tears, no snaps in two years,
healthy now and plausibly a top-36 back.

**First finding: he is already handled, and not by anything measured here.** The Projector
filters history to `season >= season - 1` and routes anyone with fewer than TWO games in that
window to the draft-capital rookie prior, returning before any blending. Brooks projects as
`rookie_prior` at 3.38/week (RB66) off his 2024 second-round capital — not off his stale
3-game, 11%-snap 2024 line. 79 of 187 RBs on the 2026 board take that path. The worry that the
form blend would drag such players toward a meaningless trailing average was WRONG for exactly
the class of player it was raised about.

**Second finding: the population that DOES get blended on thin history is small and the
gating does not reliably help it.** 2025 backtest by trailing-window depth: 1-2 games is 3.5%
of player-weeks, 3-4 games another 3.4%, and both are low-scoring players (mean actual 3.6 and
5.0 vs 6.9 for a full window).

| shape | 2025 overall | 2025 thin (1-2) | 2024 overall | 2024 thin |
|---|--:|--:|--:|--:|
| fixed weight (before) | 4.1463 | 3.065 | 3.6250 | 2.971 |
| step: <3 games -> model only | 4.1380 | **2.825** | 3.6249 | 2.967 |
| linear w/5 | 4.1407 | 2.869 | 3.6256 | 2.953 |
| sqrt(w/5) | 4.1423 | 2.932 | 3.6252 | 2.952 |

On 2025 the fixed blend looked actively harmful on thin windows (raw 2.825 -> blended 3.065)
and the step recovered all of it. **On 2024 that effect is absent** (2.971 vs 2.967) and the
overall delta is 0.0001. One season's apparent 0.008 gain did not replicate; by this ledger's
own noise rule it is nothing.

**Kept anyway, labelled as a guard rail, not a win.** `min_periods=1` means a "5-game trailing
average" can be a single game, and weighting that like five games is indefensible regardless of
what a benchmark dominated by established players shows. It is never worse on either season.
Claiming it as an improvement would be overselling it.

Still open (untested): using MEASURED snap share instead of the hardcoded depth-rank table in
roles.py, which the code itself calls a "snap-share proxy". `offensive_snap_pct` has 100%
coverage. Complication to settle first: the rate model already takes `offensive_snap_pct_roll5`
and `snap_trend` as inputs — both high in the GBDT importances — so multiplying by a rank proxy
afterwards may double-discount players whose role did not change.

## #17 Component stat lines shrink to the positional mean too (2026-08-10)

Reported: the 2026 board had Matthew Stafford at 159 rushing yards. He gained ONE rushing yard
in 2025, at age 38. Jared Goff similar. Not a one-off — a floor.

**Diagnosis.** 2025 backtest, QB rushing yards per game: predicted range 7.8-33.5 (std 7.4)
against an actual range of 0.1-42.6 (std 11.2). **Correlation 0.922** — the ORDER is nearly
right, the SCALE is not. The model will not project a QB below ~8 rushing yards a game when
the true floor is zero: Stafford predicted 8.0 vs actual 0.1, Cousins 8.9 vs 0.7, Goff 7.8 vs
2.6. Over 17 games that reads as ~160 yards for a quarterback who gained one. Same shrinkage
as everywhere else in this ledger, but conspicuous here because the true value is ~0 and the
prediction cannot get there.

Note the components were being scaled by the POINTS blend ratio, a single scalar per player.
That keeps a stat line proportional to its total but does nothing about each component
shrinking toward its own positional mean.

**Fix: blend every component with that player's own trailing-5 for THAT stat.** Season-level
per-game MAE on the 2025 backtest:

| | raw | 50/50 blend | affine recal | recal + 50/50 |
|---|--:|--:|--:|--:|
| QB passing_yards | 25.37 | 17.10 | 19.19 | **14.94** |
| QB rushing_yards | 4.49 | 2.81 | 4.09 | **2.65** |
| RB rushing_yards | 7.23 | 4.79 | 5.41 | **3.70** |
| RB receiving_yards | 3.43 | 2.00 | 2.82 | **1.82** |
| WR receiving_yards | 6.45 | **4.13** | 6.27 | 4.25 |
| TE receiving_yards | 4.73 | 2.86 | 4.47 | **2.86** |

The plain 50/50 blend cuts component MAE 30-45% at every position. A fitted affine
recalibration per (position, component), slopes 1.3-1.9, wins a further 5-15% on 9 of 11
components — **measured and deliberately not shipped**: it needs a calibration artifact fit on
a holdout and carried with the model, for a second-order gain on a display-only number.

End to end through the Projector (2025 wk10): Stafford 7.5 -> 4.1/game, Cousins 8.8 -> 4.2,
Goff 6.6 -> 5.7, while Allen holds at 40.5 (actual 36.2) and Hurts at 21.6 — the real rushers
are not flattened. Floor 1.8 -> 0.7, correlation 0.836 -> 0.864, MAE 5.30 -> 4.46.

**Scope: display only.** Fantasy points are predicted directly, never assembled from these
components, so the headline projection and every rank metric are untouched. TD components have
no rolling feature to blend against and keep the old points-ratio scaling.

**Residual, stated plainly:** Stafford still projects ~4 yards a game (~70 a season) against a
true ~1. Half the weight still sits on a model that says 8. Pure trailing-5 would be closer
(season MAE 2.25 vs 2.65 for QB rushing) but throws the model away entirely, which fails
exactly where it is needed — players changing role or team, whom this ≥8-week sample cannot
see. Scripts: scratchpad/comp_check.py.

## #14-16 Board ordering — the cap was normalizing teams, and MAE liked it (2026-08-10)

Reported symptom: the 2026 board's elite tail was flat (QB1/QB5 = 1.006) with specific
inversions — Michael Wilson WR2 over St. Brown and Chase, Cade Otton TE1 over McBride.

**Diagnosis (evidence, not inference).** On the failing board, **23 of 32 teams' WR groups
summed to exactly 31.34** — the league-mean budget. The cap was not capping, it was
NORMALIZING: most teams were forced to an identical position-group total, so a player's
projection became his SHARE OF A FIXED PIE instead of his absolute expectation. ARI summed
to 30.85, just under, and escaped scaling entirely — so Wilson kept 9.22 while St. Brown was
compressed to 8.94 and Chase to 8.73. A WR1 on a weak corps beats a WR1 on a strong one by
construction. Same at TE (14/32 pinned at 10.25). It also explains why the compression was
confined to the elite tail: the cap only binds on high-output teams, which is where the
elite players are, so #1/#24 looked fine while #1/#5 collapsed.

**#14 — rank metrics, because MAE cannot see any of this.** MAE over 6k player-weeks is
dominated by low-usage players and is *improved* by shrinking everyone toward the positional
mean, which is precisely the failure. Added `evaluate.rank_metrics`: Spearman on season
totals within position, top-12/24 set overlap vs the actual finish, and #1/#5 & #1/#24
spread ratios projected vs actual. Projected and actual are summed over the SAME
player-weeks so availability cancels. Baseline, current shipped model (3-seed, scaled
targets, form blend), MAE 4.154:

| pos | n | spearman | top12 | top24 | r_1_5 proj (act) | r_1_24 proj (act) |
|---|--:|--:|--:|--:|--:|--:|
| QB | 36 | 0.949 | 0.917 | 0.958 | 1.257 (1.203) | 1.999 (2.313) |
| RB | 98 | 0.984 | 0.833 | 0.917 | 1.228 (1.246) | 1.922 (2.325) |
| WR | 162 | 0.978 | 0.750 | 0.750 | 1.392 (1.598) | 1.794 (2.374) |
| TE | 89 | 0.975 | 0.750 | 0.833 | 1.438 (1.585) | 2.208 (2.728) |

The MODEL was never the problem: QB projected #1/#5 is 1.257 against a realized 1.203, i.e.
slightly WIDER than outcomes. Flatness at #1/#24 is legitimate shrinkage.

**#15 — the cap trade, measured.** Applying each cap to the 2025 backtest by
(team, position, week):

| cap | MAE | QB r_1_5 | TE r_1_5 | TE top12 | TE spearman | rows scaled |
|---|--:|--:|--:|--:|--:|--:|
| none | 4.154 | 1.257 | 1.438 | 0.750 | 0.975 | 0% |
| league mean (old) | **4.094** | 1.183 | **1.149** | **0.667** | **0.964** | 24.7% |
| p90 (shipped) | 4.138 | 1.241 | 1.311 | 0.750 | 0.974 | 6.0% |
| p95 | 4.143 | 1.250 | 1.346 | 0.750 | 0.975 | 4.4% |

**The old cap had the BEST MAE of any setting.** That is why it survived: every check the
ledger ran until now would have approved it. It bought that MAE by flattening TE's top five
from 1.438 to 1.149 (actual 1.585) and dropping TE top-12 overlap and Spearman. p90 restores
almost all of it for +0.044 MAE. p95 is inside noise of p90 on every rank metric — not
adopted.

**#16 — seed averaging rejected as a suspect.** Trained one 3-seed ensemble and scored the
SAME networks four ways (each member alone, and averaged), so seed noise cannot confound it.
The ensemble's #1/#5 sits inside its members' range at every position (QB 1.257 vs members
1.254/1.251/1.257; TE 1.438 vs 1.445/1.463/1.417), and Spearman/top-k are identical.
Averaging does not compress the tail.

**Persistence defect, reported not fixed.** Availability IS baked into `projected_points`
(NFL-API `_apply_roles` multiplies each weekly row by expected_games/17), and `exp_games` IS
lost at persistence — NFL-API never calls `season.assemble_season` and `PlayerProjection`
has no column for it. `/projections/season/{season}` therefore reports `games = count(week)`
= 17 for everyone (18 weeks minus the bye; verified on Wilson, whose stored weeks skip 14),
and `ppg = total/17` is points per SCHEDULED game, already availability-discounted, not the
on-field rate. Season totals are arithmetically correct and ordering is unaffected, so this
does not cause the reported symptom — it is a reporting bug.
Scripts: scratchpad/seed_tail_test.py, rank_baseline.py.

## Quantile bands — the "floor problem" was never fixable (2026-08-09)

Re-measured the band after the form blend started scaling it (predict.py), expecting the
blend to have broken calibration. **It hadn't** — coverage 0.706 -> 0.705 (1 seed) and
0.711 -> 0.712 (3 seeds). Scaling the band with the projection preserves calibration.

**Seed-ensembling the quantile model is a wash**, unlike the mean model: band coverage
0.706 -> 0.711, median MAE 4.093 -> 4.126 (slightly worse), both inside noise. Kept the
capability (QuantileModel.extra_models, offsets calibrated on the ENSEMBLE's predictions,
not one member's) for run-to-run stability, but it is not an accuracy win — don't claim it.

**The real finding: the floor was never achievable.** The ledger has carried "q10 empirical
0.24, the floor is the weak spot" as an open bug for a long time. Training wider NOMINAL
quantiles to chase it (2025 backtest, 1 seed):

| nominal | emp low | emp high | band | width | median MAE |
|---|---:|---:|---:|---:|---:|
| 0.10 / 0.90 | 0.249 | 0.865 | 0.706 | 11.0 | 4.093 |
| **0.05 / 0.95** | **0.205** | **0.903** | **0.803** | 13.9 | 4.110 |
| 0.02 / 0.98 | 0.196 | 0.948 | 0.899 | 17.9 | 4.081 |

The floor barely moves — 0.249 -> 0.205 -> 0.196 — while the ceiling tracks its nominal
almost exactly. Cause: **19.1% of player-weeks score ZERO or less** (WR 23%, TE 20%, RB 16%,
QB 8%) and predictions are clipped at 0, so no non-negative floor can cover better than
~0.19. The measured 0.196 is essentially AT that bound. This was never a model defect and
no amount of recalibration was going to fix it — which also explains why the earlier
conformal attempts came out as no-ops.

**Shipped: DEFAULT_QUANTILES = (0.05, 0.5, 0.95)**, the first honest 80% band (0.803) with a
calibrated ceiling (0.903 vs 0.865 before), median MAE unchanged. The floor should be read
as roughly a 20th percentile, and that is the structural limit, not a to-do.

Note on where bands are used: the NFL-API board path REPLACES floor/median/ceiling with the
Monte-Carlo simulator for players with enough history (538 of 888 rows in the 2026 run), so
this matters most for the in-season weekly path, which uses quantile bands for every row.
Scripts: scratchpad/band_check.py, wider_bands.py.

## QB separation — the board was flat for two independent reasons (2026-08-07)

Symptom: on the 2026 board Josh Allen (16.95) and Jared Goff (16.94) project within
0.01 points of each other, when the real 2025 gap was 23.4 vs 18.9.

**Cause 1 — the model shrinks, and it shrinks QBs twice as hard.** Player-season means
on the 2025 backtest, predicted spread as a fraction of actual spread:

| pos | spread ratio | calibration slope (actual~pred) | season-mean corr | naive-5 corr |
|---|---:|---:|---:|---:|
| QB | **0.33** | **2.18** | 0.711 | **0.782** |
| RB | 0.69 | 1.38 | 0.950 | **0.980** |
| WR | 0.67 | 1.38 | 0.926 | **0.961** |
| TE | 0.70 | 1.35 | 0.943 | **0.961** |

The model beats "average his last five games" on per-game MAE at every position — and
LOSES to it at *ranking players*, everywhere. MSE on a target this noisy is minimized by
shrinking toward the positional mean; that is right for per-game error and wrong for every
comparison between players. QB is worst because its single-game variance is the largest.

**Cause 2 — the team-position budget cap was set at the league MEAN.** 40-46% of real
team-games exceed it at every position, so any above-average offense was truncated to
average. Applied to 2025 projections it scaled down 28.6% of rows and cut the Allen-Goff
gap from 4.18 to **0.68**. At QB one starter carries the whole group, so the cap acted as a
flat ~17-point ceiling on every good QB — exactly the reported symptom.

**Fix 1 (nfl_projections/blend.py): blend the network with the player's own form.**
Weights fitted on the 2024 backtest, applied unchanged to 2025 (never tuned there):

| pos | MAE model -> blend | corr model -> blend | spread model -> blend |
|---|---:|---:|---:|
| QB | 6.960 -> **6.798** | 0.711 -> **0.779** | 0.33 -> **0.65** |
| RB | 4.343 -> **4.315** | 0.950 -> **0.980** | 0.69 -> **0.92** |
| WR | 3.885 -> 3.908 | 0.926 -> **0.961** | 0.67 -> **0.89** |
| TE | 3.143 -> **3.122** | 0.943 -> **0.963** | 0.70 -> **0.78** |

Overall 2025 week MAE **4.178 -> 4.124** — a bigger gain than seed-ensembling — while
ranking and spread improve at the same time. Shipped weights (QB/RB/WR .5, TE .6) sit
between each season's own optimum and land within 0.02 MAE of both. The blend partner,
`fanduel_fantasy_points_roll5`, is already one of the network's input features: the model
has this information and under-weights it.

**Fix 2: preseason mode.** Before a season there is no current form, and the prior season's
FULL average beats its last five games at predicting the next season (corr 0.919 vs 0.886,
2022-25). Preseason-board test (2025 wk1 projections, which see only 2024, scored against
2025 season averages, n=253): blending against the season average cut MAE **QB 4.18 ->
3.29, WR 2.54 -> 2.10, RB 2.66 -> 2.39, TE 1.98 -> 1.82**, and QB correlation 0.157 ->
0.321. Projector switches automatically when the projected season has no games in history.

**Fix 3: budget cap at the p90 team-season average, not the league mean** (QB 17.3 -> 22.4,
RB 23.0 -> 31.0, WR 31.3 -> 41.9, TE 10.3 -> 14.4). Rows scaled down 28.6% -> 5.2%; the
Allen-Goff gap goes 0.68 -> **3.70** (actual 2.80). The cap still binds on its real target
(two RB1s on one team) without truncating good offenses.

**Honest limits.** The blend only *ties* the naive baseline on QB ranking (0.779 vs 0.782)
— it buys the naive predictor's discrimination while keeping the model's better per-game
MAE, rather than beating both. Spread is 0.65, still short of 1.0, so the board remains
somewhat compressed. And the p90 cap costs ~0.09 QB MAE vs the old one (6.657 -> 6.747):
tighter shrinkage always flatters MAE. Every stage of this pipeline had been trading
discrimination for MAE; these changes trade some back deliberately.
Scripts: scratchpad/blend_validate.py, board_check.py.

## Standardized training targets — fixes the broken component stat lines (2026-08-06)

Found in production: the stored component projections were degenerate. Every 2026 QB was
projected for ~1.8 rushing TDs/game (Goff and Stafford, who scored 0 rushing TDs in 2025,
got the same 1.78 as Josh Allen); the 2025 run gave every QB exactly 0.0. Season totals made
it plain — Josh Allen 31.2 rushing TDs and 11.6 passing TDs; Derrick Henry 7.4 receiving TDs
and 3.0 rushing TDs. A constant that changes between training runs is an untrained head.

**Cause:** the multi-output net trains all 9 targets under one MSE, but their scales differ by
orders of magnitude (passing_yards ~250 vs rushing_tds ~0.05). The yardage targets own the
gradient; the TD/reception heads barely train and settle near a per-run constant. Same reason
component-first models were 5x more seed-sensitive (#10 follow-up).

**Fix:** `model.TargetScaler` standardizes each target to zero mean / unit variance (fit on
train only), predictions invert through it in `predict_batch`. `scale_targets=True` default on
`train_model` / `train_ensemble` / `evaluate.backtest`; persisted in metadata.json (models
saved before this load with `target_scaler=None` and behave as before).

**A/B, standard 2025 wk1-18 backtest, 3-seed ensembles, 60 epochs, identical both arms:**

| | unscaled | scaled |
|---|---:|---:|
| **fantasy-point MAE (headline)** | **4.158** | **4.186** |
| corr | 0.615 | 0.610 |
| QB / RB / WR / TE MAE | 7.10 / 4.28 / 3.87 / 3.06 | 7.11 / 4.27 / 3.92 / 3.11 |
| FP recomputed from components | 4.279 | 4.189 |
| receiving_tds corr | **−0.016** | **0.253** |
| receiving_tds MAE | 0.244 | 0.202 |
| passing_interceptions MAE | 0.098 | 0.073 |
| receptions MAE / corr | 1.152 / 0.658 | 1.106 / 0.675 |
| QB rushing_tds mean (actual 0.164) | **0.030** | **0.141** |
| QB receiving_tds (nonsense stat) | 0.42–0.59 | 0.0 |
| passing_yards / rushing_yards MAE | 8.02 / 7.08 | 8.30 / 7.20 |

**Verdict: keep.** The headline moves +0.028 MAE — inside the noise band for 3-seed arms, so
read it as unchanged, not as a win or a loss. What clearly improves is everything the fix
targeted: no more hallucinated QB receiving TDs, QB rushing TDs go from collapsed-to-zero
(0.030) to about right (0.141 vs 0.164 actual), receiving_tds goes from *uncorrelated with
reality* to 0.25, and points recomputed from the components (4.189) now match the direct head
(4.186) instead of trailing it. Yardage MAEs are a hair worse.

**What it does NOT fix:** component *differentiation* is still compressed in both arms — the
predicted runner-vs-pocket QB rushing gap is +16.6 yds/g against an actual +27.3, rushing-TD
gap +0.14 vs +0.36, RB catcher-vs-ground receptions +1.89 vs +3.38. And it does nothing for
the headline QB problem (QB MAE 7.10 → 7.11): rushing production still isn't separating QBs in
the fantasy-point projection. Untried follow-up: standardize as now but weight the
fanduel_fantasy_points target higher in the loss, to buy back the headline delta.
Scripts: scratchpad/target_scaling_ab.py, ab_report.py.

Naive last-5-avg is 4.255; our best (NN-ensemble + GBDT blend) is 4.197 — a real
but small edge. Direct-FP beats component-first extrapolation at every position.
Point projection (rolling features, #2) looks near the
practical ceiling for this data/architecture: opponent/scheme (#4/#5),
volatility (#7), and component-scoring (#10) all failed to beat it. Quantile
floor/median/ceiling works and is wired in; the median matches the mean model,
the ceiling is trustworthy, the floor is the weak spot. Remaining honest levers:
per-position models, richer data (snaps/routes/betting lines), or accept current
accuracy and focus on usage (DFS optimizer, weekly workflow).

---

## Opportunity share vs trailing rate — week 1 2025 (2026-09-09)

**Question.** The projections read "what have you done". When a role changes in
the offseason the carries that left the building are invisible: a back promoted
to RB1 keeps his backup rate, and the man he replaced keeps a starter's. Does
projecting *opportunity* — depth rank to share of the team's carries, times the
player's own efficiency — beat it?

**Protocol** (the owner's, and the right one). Nothing after 2024 is used. The
share curve is fit on 2020-24, efficiency is each player's 2024 ypc regressed
toward the league mean by carry count (k=100), team volume is the 2024 rate, and
the depth chart is the earliest one published in 2025. Score against week 1
2025. Harness: `experiments/opportunity_share_wk1.py`.

**Result** (n=73 RBs with a prior season):

| model | MAE | bias | corr |
|---|---|---|---|
| trailing (what we ship) | 23.03 | **+11.48** | 0.588 |
| team_share | **19.29** | **+1.99** | 0.568 |
| blend of the two | 19.96 | +6.74 | **0.606** |

Restricted to RB1s — the promoted and lead backs, where a role change actually
bites — trailing is 32.24 MAE against team_share's 27.56.

**The bias is the finding, not the MAE.** Trailing over-projects by 11.5 yards a
game, because it hands every back last season's role. Share-based opportunity is
close to unbiased. For a yardage prop that difference is the whole game: a model
running eleven yards high takes overs it should not, and a board that suggested
24 overs to 14 unders is what that looks like from the outside.

**Be honest about what it does not do.** Correlation is flat to slightly worse,
and among RB1s alone both models correlate poorly (0.27 and 0.18). Week-1
rushing is noisy — Etienne went for 143 off a 37-yard trailing rate, Saquon for
60 off 125. This is "better calibrated", not "better informed", and the blend
column is a reminder that the two carry different information.

**Fixed alongside it.** The depth-rank multiplier was applied to
`fanduel_fantasy_points` and nothing else, so the component stats a prop settles
on kept a starter's rate while the points beside them were correctly discounted.
It now scales `roles.ROLE_SCALED_COMPONENTS` too. That is a straight bug and is
independent of the share model above.

**Still open.** The multiplier only ever scales *down* (`if role_mult < 1.0`),
so promotion remains unmodelled — the share curve is the natural reference for
fixing that, since it says what an RB1 should get rather than only what an RB3
should lose. And none of this touches matchup: personnel groupings are available
in nflverse participation data (Jacksonville ran 68.4% 11 personnel in 2025
against a league 61.4%, and only 7.9% two-RB), as are coverage fields, so
"who is covering this receiver and are they any good" is reachable from data
already on hand.

---

## Coverage matchup — a null result, and one narrow win (2026-09-09)

**Question.** The share model fixed bias but not correlation. Does knowing the
defence a receiver faces carry the information that would?

**Protocol.** Same as before, repeated over four week 1s (2022-25) so a single
noisy week could not carry the conclusion. Harness:
`experiments/coverage_wk1.py`.

| position | n | MAE | corr |
|---|---|---|---|
| TE | 183 | 18.67 -> **17.92** | 0.371 -> **0.424** |
| WR | 392 | 26.73 -> 26.94 | 0.472 -> 0.457 |
| RB | 188 | 13.59 -> 13.67 | 0.312 -> 0.321 |

**Only tight end works**, and it moves both error and correlation, which is
what separates a matchup signal from a recalibration. It holds in three of four
seasons (+0.025, +0.134, -0.011, +0.056).

**Wide receiver is actively worse, and that is the interesting part.** "Yards
allowed to WR" is spread across a defence's whole secondary and a team's whole
receiving corps, so it says nearly nothing about the matchup one receiver
faces. The thing that would — which corner travels with him and whether he is
any good — is a shadow-coverage assignment, and nflverse has no such field. A
tight end draws a much more specific assignment, usually a linebacker or
safety, which is why the team-level number is closer to a real matchup for him.

**Coverage scheme is a null result.** Man/zone rates are published (49% of
snaps classified) and a receiver's own man-vs-zone yards per target can be
computed, but regressed for sample size the adjustment spans 0.977 to 1.015 at
the 10th and 90th percentiles — a two percent nudge — and it changed MAE by
0.03 yards across 763 player-weeks. It is not wired in anywhere. Reviving it
needs per-route matchup data rather than a team rate.

**Shipped** in sharp-edge as `nfl/matchup.py`, TE receiving only, clamped to
0.75-1.30. The two positions that failed are named in that module so nobody
re-adds them on intuition.

---

## Opponent-adjusted defence beats raw allowed-per-game (2026-09-09)

**The objection, and it is right.** A raw allowed-per-game number measures who
a defence happened to draw as much as how it played. Face three elite tight
ends and it looks terrible; face three backups and it looks elite. Neither is a
statement about the defence.

**The fix.** Compare every player against himself. For each player-game against
a defence, the baseline is that player's production over all his *other* games
that season, and the game scores as a ratio; average those and opponent quality
cancels. Weighted by each player's baseline volume, because a fringe receiver's
ratio is mostly noise and an unweighted mean lets it count as much as a
starter's. Leave-one-out is load-bearing — a season average that includes the
game being scored leaks the answer into its own baseline.

**Result**, week 1 of 2022-25, prior season only. Bar: MAE improves in at least
three of four seasons, since correlation alone can shuffle ranks without
getting closer.

| market | MAE wins | corr wins | shipped |
|---|---|---|---|
| TE receiving yards | **4/4** | 3/4 | yes |
| RB receiving yards | **4/4** | 3/4 | yes |
| RB receptions | **3/4** | 3/4 | yes |
| TE receptions | 2/4 | 3/4 | no |
| RB rushing yards | 2/4 | 2/4 | no |
| WR (any market) | 0-2/4 | 0-1/4 | no |

Two things changed. The adjusted metric beats raw for tight ends (pooled MAE
18.02 -> 17.70), and more importantly it **unlocks running-back receiving**,
which raw did not justify — raw won 2 of 4 seasons, adjusted wins 4 of 4.

Unweighted leave-one-out is slightly *worse* than raw, which is the argument
for the volume weighting rather than a free parameter.

**Rushing still fails**, for a plain reason: the factor is built from receiving
yards allowed, a statement about pass defence. Tested against rushing anyway
and came back a coin flip, as it should have. **Wide receiver still fails
everywhere** — the dilution problem is not something a better denominator
fixes; it needs shadow-coverage data that nflverse does not publish.

Harness: `experiments/fpa_adjusted.py`.

---

## Granular defensive matchup, and the trait that is stable but inert (2026-09-11)

Follow-up to "Why opponent/matchup keeps failing", which tested defence-vs-
position at the team level and found split-half r of 0.06-0.25. That is one
crude aggregate and does not rule out the granular version — that a defence is
reliably good against 11 personnel and soft against 12. Same split-half method
so the numbers compare, 186,137 plays with participation data, 2022-25.
Harness: `experiments/granular_stability.py`.

| trait | split-half r | plays per cell |
|---|---|---|
| team: **success rate allowed** | **0.618** | 721 |
| team: EPA allowed per play | 0.397 | — |
| team: EPA per rush / per pass | 0.234 / 0.233 | — |
| team x **personnel**: EPA | **-0.062** | 21 |
| team x personnel: success rate | 0.026 | 21 |
| team x formation: success rate | 0.033 | 95 |
| team x man/zone: EPA | 0.139 | — |

**Slicing by personnel or formation destroys the signal rather than refining
it** — one cut comes back negative. The last column is why: a team-half-
personnel cell holds about 21 plays. A defence's true rate against 12 personnel
is not measurable from 21 snaps, and no modelling fixes a sample size.

**Success rate allowed is the real find.** r 0.618 is more than double anything
in the earlier work and well above EPA's 0.397. The difference is the metric:
the original test used fantasy points allowed, which is volume-contaminated and
noisy, while success rate is bounded per play so explosive plays cannot drag it.

**And it is nearly inert as a player-level multiplier**, which is the useful
half of the result. Four week-1 samples, prior season only
(`experiments/success_rate_predict.py`):

| market | MAE change | seasons won |
|---|---|---|
| TE receiving | -0.24 | 4/4 |
| WR receiving | -0.03 | 3/4 |
| RB rushing | +0.03 | 2/4 |
| RB receiving | +0.02 | 1/4 |
| QB passing | -0.98 | 2/4 |

The whole factor spans 0.94 to 1.06; a six per cent nudge cannot move a
projection far however well it is measured. This is the earlier warning arriving
from the other side — there the matchup signal was real but unstable, here it is
real *and* stable and still tiny.

Shipped in sharp-edge for tight-end receiving only, where it clears the bar and
is genuinely incremental: stacked on the existing matchup factor it takes TE
receiving from 16.71 to 16.52 MAE, 4/4 seasons, with the two factors correlating
**0.110**.

---

## Usage decomposition beats trailing rate — for tight ends and backs (2026-09-11)

The rushing share model already worked (week-1 MAE 23.03 -> 19.29, bias +11.48
-> +1.99). This asks whether the same shape holds when the opportunity being
divided is targets, and whether the share should come from the depth chart or
from the player's own prior share. Harness: `experiments/usage_share.py`.

**Week 1 of 2022-25**, prior season only:

| pos | model | MAE | bias | corr | seasons won |
|---|---|---|---|---|---|
| WR | trailing | 24.55 | +6.07 | 0.540 | |
| | depth-chart share | 26.98 | -2.96 | **0.249** | 1/4 |
| | **own prior share** | **23.21** | -0.54 | 0.535 | **4/4** |
| TE | trailing | 16.41 | +6.01 | 0.472 | |
| | **own prior share** | 14.99 | +1.02 | 0.474 | **4/4** |
| RB | trailing | 11.27 | +0.90 | 0.439 | |
| | **own prior share** | **10.49** | -1.51 | 0.464 | **4/4** |

**The depth chart loses to the player's own share for receivers**, which is the
opposite of the rushing result. Chart rank is coarse — two WR2s can see 25% and
15% of a team's targets, and collapsing them to one number destroys correlation
(0.540 -> 0.249). A running back's depth rank is much closer to binary, which is
why it survives there and not here.

**In-season (2025 weeks 2-6)** — the regime that matters from week 2 on, with
current-season data once two weeks exist:

| pos | n | MAE trail | MAE own | corr trail | corr own | weeks won |
|---|---|---|---|---|---|---|
| WR | 637 | 21.30 | 21.38 | 0.566 | 0.552 | 3/5 |
| TE | 291 | 15.28 | **14.66** | 0.521 | **0.548** | 4/5 |
| RB | 385 | 11.05 | **10.60** | 0.504 | **0.527** | 4/5 |

**WR does not survive.** Its week-1 win was the year-stale trailing average
being bad rather than the decomposition being good — once trailing is current
the two are indistinguishable and correlation is slightly worse. Tight ends and
backs win in both regimes, on MAE and correlation together, which is the
combination that has survived every other test here.

**Not shipped yet, and the reason is the honest one.** All of this beats a
*trailing average*. The production model is not a trailing average — it has an
ML layer on top — so "beats trailing" is not "beats what we ship". Testing that
needs the production projections replayed historically, which
`evaluate.backtest_season_board` can do and this harness does not.

---

## Next Gen Stats add nothing — and the near-miss is the lesson (2026-09-11)

**First, what the model already has.** The feature list carries EPA by position,
racr, pacr, dakota, target_share, air_yards_share, wopr, snap counts and snap
share, plus derived yards-per-target, epa-per-target and average target depth.
"We do not use advanced metrics" is not true; most of the public ones are in.

**DVOA specifically is not obtainable.** It is FTN/Football Outsiders'
proprietary metric and is published in no free feed. The closest public
equivalents are the Next Gen Stats efficiency numbers, which nflverse does
publish (2016-2026) and which the model does not touch: separation, cushion,
YAC above expectation, catch rate, rush yards over expected, time to line of
scrimmage, eight-defender rate.

**Step 1 looked promising.** Correlation of each NGS metric with the *residual*
of a trailing projection — the part the player's own rate gets wrong:

| metric | r vs actual | r vs residual |
|---|---|---|
| RB rush_yards_over_expected_per_att | +0.143 | **-0.130** |
| RB rush_pct_over_expected | +0.018 | **-0.137** |
| RB NGS efficiency | -0.091 | **+0.118** |
| TE catch_percentage | +0.023 | **-0.150** |
| WR (everything) | — | all < 0.10 |

Negative, which is the interesting direction: a back who ran above expectation
last year tends to *under*-perform his own trailing rate next year. Efficiency
regresses, exactly as a DVOA-minded reading would predict.

**Step 2 looked like a win.** Leave-one-season-out regression of actual on
trailing plus the standardised metric improved MAE in 4 of 4 seasons every time
— RB rushing 27.51 -> 25.46 with RYOE, TE 21.35 -> 17.80 with catch rate.

**Step 3 killed it.** Correlation fell in every one of those cases, which is the
signature of shrinkage rather than signal. The control — the same regression on
trailing *alone* — captures essentially the whole gain:

| model | MAE raw | MAE fit (trailing only) | MAE + NGS | NGS adds |
|---|---|---|---|---|
| RB rush, RYOE/att | 27.51 | 24.93 | 25.46 | **+0.53** |
| RB rush, NGS efficiency | 27.51 | 24.93 | 24.91 | -0.02 |
| TE rec, catch % | 21.35 | 17.83 | 17.80 | -0.03 |
| WR rec, air yards share | 28.78 | 27.09 | 27.14 | +0.05 |

**The entire improvement was the fitted slope pulling predictions toward the
mean.** NGS moves MAE by less than 0.05 either way once that is controlled for,
and one of the two "best" metrics makes it worse. The residual correlations at
r≈0.13 are real but too weak to survive contact with a baseline that is itself
shrunk.

Without the control this reads as a clean 4-of-4 win across three positions. It
is worth keeping as the example of why "improved MAE in every season" is not
sufficient evidence on its own — a model can buy MAE with variance it should
have kept.

Still untested: FTN charting (play action, screen, RPO, motion, backfield count,
box count), which is play-level context rather than a player efficiency rating
and might behave differently. Harness: `experiments/ngs_incremental.py`.

---

## FTN charting: the most stable traits measured here, and they add nothing (2026-09-11)

NGS failed because efficiency regresses. FTN charting is play-level *context*
rather than a player rating — was the snap play-action, a screen, an RPO, was
there motion, how many in the backfield, how many in the box — so the
hypothesis was that it describes a player's ROLE, and role is what persists.

**Stability is emphatic.** Season-to-season correlation of a player's own role
profile, 540 player-seasons:

| feature | r (season N → N+1) |
|---|---|
| n_offense_backfield | **0.748** |
| is_play_action | **0.719** |
| n_defense_box | 0.674 |
| is_screen_pass | 0.602 |
| is_motion | 0.582 |
| is_no_huddle | 0.371 |
| is_rpo | 0.344 |

These are the most stable traits measured anywhere in this file — above
success-rate-allowed (0.618), far above defence-vs-position (0.06-0.25). A
screen-game back is a screen-game back next year.

**And they predict nothing.** Leave-one-season-out, predicting a player's share
of his team's touches next season from his current share plus the profile:

| position | n | current share (fitted) | + FTN profile | FTN adds |
|---|---|---|---|---|
| RB | 253 | 7.109 | 7.345 | **+0.235** |
| WR | 338 | 2.950 | 2.919 | -0.031 |
| TE | 110 | 2.346 | 2.587 | **+0.241** |

(MAE in share percentage points.) Worse for two of three, nothing for the
third, and correlation falls for RB and TE.

**The reason is the finding.** A role profile is already *encoded in* the share
it produces. Knowing a back is a screen back adds nothing once you know he
takes 14% of his team's touches — the share is the role, expressed as a number.
Context describes usage; it does not predict it.

That is now four independent tests landing the same way — defence-vs-position,
success-rate-allowed, Next Gen Stats, and FTN charting. Stability keeps being
easy to find and incremental signal keeps being absent, and the two are not the
same question. The only things that have moved projections here are opportunity
quantities: share of carries, share of targets, and who the role belongs to
after it changes hands.

Harness: `experiments/ftn_usage.py`.

---

## The gate: usage decomposition vs the PRODUCTION model (2026-09-11)

Every usage result so far was scored against a trailing average, which the
shipped model is not — it has the ML layer, the form blend, the depth-role
multiplier and a team budget on top. "Beats trailing" was always a proxy, and a
weak baseline flatters any method. This runs `evaluate.backtest` for real:
trained on seasons before 2025, predicting weeks 1-8 with `include_components`,
compared on identical player-weeks. Single seed, so anything inside +/-0.05 MAE
is not readable; every delta below is far outside that.

| market | n | prod | trail | usage | blend | usage-prod | r prod | r use | wks won |
|---|---|---|---|---|---|---|---|---|---|
| RB rushing | 524 | **22.21** | 25.17 | 24.01 | 22.37 | **+1.80** | 0.537 | 0.495 | 4/8 |
| WR receiving | 888 | **22.59** | 25.20 | 24.09 | 22.62 | **+1.50** | 0.512 | 0.464 | 2/8 |
| TE receiving | 457 | 15.93 | 16.15 | **15.34** | 15.39 | **-0.59** | 0.522 | 0.525 | 5/8 |
| RB receiving | 545 | 11.33 | 12.12 | 11.50 | **11.18** | +0.16 | 0.439 | 0.473 | 5/8 |

**The answer is mostly no, and the earlier results were beating a weak
baseline.** Production beats trailing everywhere by 2.5-3 MAE, which is the ML
layer doing real work; usage also beats trailing everywhere, by rather less; and
on the two big markets production beats usage by 1.5-1.8 with the week count
going the same way (4/8 and 2/8). Whatever role-staleness the decomposition was
fixing, the shipped pipeline already handles most of it.

**Two narrow exceptions, and they are the same two markets as everything else.**
TE receiving is an outright win for usage — lower MAE, marginally better
correlation, 5 of 8 weeks. RB receiving is a wash on MAE but usage correlates
better (0.473 vs 0.439) and the blend beats production outright (11.18 vs
11.33). Those are exactly the markets where the opponent-adjusted matchup factor
and success-rate-allowed also earned their place, which is now three independent
methods pointing at the same two low-volume, role-dependent markets as the place
the production model is thinnest.

**Not shipped.** The blend is a tie or worse on the two big markets and the
RB-receiving margin is -0.15 on a single seed — above the noise floor but not by
much. A full five-seed run on those two markets is what would settle it, and
that is the next concrete step rather than a judgement call.

**What the share work is actually for**, then: the role-conflict diagnostic that
already ships in sharp-edge — flagging when our projection ranks a team's
players differently from the market — rather than a replacement projection. It
found the Jacksonville backfield inversion, which the production model does
make, even though the decomposition does not beat it on average.

Harness: `experiments/vs_production.py` (runs inside the nfl-api image; local
sklearn is binary-incompatible with the local numpy).

## Live week 1, 2026: the first prospective scorecard (2026-09-14)

The first week the model had to call in advance. Scored Monday afternoon with the
Denver–Kansas City game still to play, against nflverse actuals, because NFL-API's
own scorer waits on the Tuesday loader. Same rules as that scorer: the board as
`/projections/` served it, players with no stat line unscored, projections dated
after their team's gameday excluded. **n = 277** player-weeks from the 13 Sunday
games — one week, so every number here carries a lot of noise.

| | n | MAE | bias | naive MAE (same rows) | Spearman model / naive |
|---|---|---|---|---|---|
| Board as published | 277 | 4.50 | **−2.64** | 4.74 vs **4.75** (tie) | 0.69 / 0.67 |
| Availability weight removed | 277 | **4.29** | −1.17 | 4.74 vs **4.49** | — |
| QB (weight removed) | 29 | 7.10 | −4.20 | 5.92 vs 7.01 | 0.50 / 0.50 |
| RB (weight removed) | 68 | 4.71 | −1.51 | 5.80 vs 5.05 | 0.73 / 0.60 |
| WR (weight removed) | 118 | 3.97 | −0.43 | 4.13 vs 4.06 | 0.67 / 0.66 |
| TE (weight removed) | 62 | 3.14 | −0.76 | 4.06 vs 3.34 | 0.62 / 0.49 |

Top-N overlap with the actual finish (model / naive): QB-12 8/7, RB-24 17/14,
WR-36 21/19, TE-12 3/4. The floor–ceiling band caught 62.8% (26.7% above the
ceiling, 10.5% below the floor); with the availability weight divided out of both
ends it caught 72.6% (16.6% / 10.8%).

**The published board ran 2.6 points low on every player who played, and most of
that is the pipeline, not the model.** Week 1 went down the preseason path because
nflreadpy had no 2026 rows when it ran (`use_espn`). That path multiplies each week
by expected games / 17, around 0.75 for the scored players, which is a SEASON rate:
the chance of missing some games over 17 weeks. Applied to one week whose inactives
are already known, and scored only on players who suited up, it guarantees the
number comes in low. Dividing it back out drops MAE 4.50 → 4.29, turns a tie with
naive into a 0.25 lead, and puts band coverage back on the ~72% it has always had.
The in-season path writes `exp_games = 1.0`, so from week 2 this should disappear
without a change, provided the dataset build picks up 2026.

**What is left after that is mostly the week, not the model.** Week 1 games averaged
50.0 points against a 45.2 Vegas total; the two previous week 1s averaged 45.8 and
41.3. Naive usually over-projects week 1 by 1.1–1.3 points (2024, 2025) and this year
under-projected by 0.4, so the league ran about 1.5 points per player hotter than an
ordinary opening week. That shows up where scoring concentrates: five of the
fifteen biggest misses came from Chicago 59, Carolina 37.

**Still true after both corrections: QBs are too low.** They trail naive by 1.1 MAE
with the weight removed, 1.6 points more negative than naive's own bias. Their
ordering is no worse than naive (0.50 each, 8 vs 7 of the top 12), so the problem
is level, not ordering, which is the QB compression the Aug-07 QB separation work
found. One week and 29 QBs cannot say more than that.

**Data facts found along the way.**
- The only week-1 projection for NE, SEA, SF and LA was computed 2026-09-12 04:00 UTC,
  after the Wednesday and Thursday games, so it can never be honestly scored. The
  earlier Wednesday run was overwritten before the vintage archive existed.
- Twelve skill players with stats had no projection, all late signings: Diggs (WAS,
  15.5), Keenan Allen (IND, 9.2), Jonnu Smith (GB, 7.0), Waller (CAR, 4.8). Nine more
  were projected for teams they had left. The ESPN roster sync failed today with a 403.
- The preseason path runs with `use_injuries=False`, so players ruled out in advance
  stayed on the board: Josh Jacobs 9.6, Tua Tagovailoa 8.1, Brock Bowers 6.5. The
  scorer drops them, so they never count as misses, but a user of the board still
  sees them.

**Not settled:** whether the weekly board should carry an availability weight at
all, and if so which one (the questionable/out tag, not a season durability rate).
The rows NFL-API freezes for week 1 will use the published, discounted numbers, so
its record will show the tie with naive unless it is rescored.

Harness: `experiments/live_week_accuracy.py 2026 1` (read-only: public API +
nflreadpy).

## The in-season path, run on 2026 for the first time (2026-09-14)

Not a model change, a pipeline audit. The loop has always been right — every
Wednesday rebuilds from nflverse, retrains, projects the next week — but
`config.SEASONS` was a hand-bumped constant still ending at 2025. The dataset never
included 2026, and NFL-API reads "season newer than the dataset" as preseason, so
week 1 went out on the ESPN/rookie-prior/availability path and every later week
would have too: last season's form, re-scaled by environment, stamped `as_of_week=N`.
Now `config.current_season()` (nflreadpy's date rule), and `build_dataset` drops only
an unpublished NEWEST season on a 404. Any other load failure still raises, because
silently falling back to last season is exactly the failure being fixed. Dataset:
148,267 stat rows through 2026.

Flipping that exposed the in-season path to 2026 data for the first time. Three
single-seed dry runs of week 2, in the nfl-api image against a throwaway SQLite DB,
each turned up something:

| run | published | found |
|---|---|---|
| 1 | 544 | 4 healthy players zeroed as injured; 63 week-1 inactives dropped |
| 2 | 603 | rookie-prior backup QBs at 6-14 pts; one cross-team depth rank |
| 3 | 603 | clean |

**Depth charts.** nflverse's 2025+ format is dated league-wide snapshots (178 for 2026,
newest first). `DepthChartAnalyzer` let the last row win, i.e. the OLDEST: 252 of 574
offensive players got a stale rank or team. For 2025 the oldest snapshot is early
August, and 212 of 546 players differ from the chart going into week 1. **So #20's
board harness, and any 2025 Projector run, read the August 2025 depth chart.** A/Bs
inside it shared the flaw; absolute numbers there may move. Fixed with
`current_depth_chart`: the latest snapshot on or before the week's first gameday.

**Injury name collisions.** The first-initial + last-name fallback crossed teams.
Four healthy players were zeroed: David Montgomery (IND's D.J. Montgomery is on the
reserve list; Montgomery scored 28 in week 1), Travis Etienne (CAR's Trevor), Justice
Hill (NE's Julian), Barion Brown (CHI's Brittain). Now matched by gsis ID first; a
name match only counts on the same team with no conflicting ID. After the fix all 84
zero-outs are real reserve-list players.

**Stale gameday inactives.** The Projector skips non-`ACT` rows, and nflverse serves
week 1 rosters until week 2's are posted. 63 week-1 `INA` skill players (36 with 2025
stat lines) vanished, including Penix, Tua, Bowers, Kamara and TreVeyon Henderson.
`INA` from an earlier week no longer blocks a player; the injury report decides.

**Rookie prior without a depth discount.** It is fit on rookies who played, so it's a
starter's rate. Veterans get the depth-rank multiplier and rookie rows returned before
it. The preseason availability weight had been hiding this. In season: Fernando
Mendoza 14.4 as LV's QB2, Ty Simpson 11.2 behind Stafford. Rookie rows now take the
same multiplier (5.0 and 1.7), and 104 rookie-prior rows moved. The depth lookup is
team-scoped too, which ends Travis Etienne reading Trevor's CAR RB4 slot (4.0 → 7.2).
This also changes preseason-path rookie numbers; #20 was not re-measured.

NFL-API side (its own commit): the in-season run keeps `rookie_fallback`, the published
week gets its game environment and simulator like the archived weeks, and the loader
refreshes the season in progress. Its skip-if-any-rows would have loaded 2026 week 1
and never week 2.

**The final dry-run board:** 603 rows (406 veteran_ml, 113 rookie_prior, 84
injured_out), `exp_games` 1.0 throughout. 9,648 archived rows for weeks 2-18 at
`as_of_week=1`. 14 minutes locally, single seed. Mean projection for the players
scored in week 1:

| | week 1 board | week 2 dry run | week 1 actual |
|---|---|---|---|
| QB | 8.7 | 13.4 | 15.7 |
| RB | 5.4 | 6.5 | 8.0 |
| WR | 4.4 | 4.9 | 5.7 |
| TE | 2.8 | 3.9 | 4.3 |

Different weeks, so this shows the discount is gone. It is not an accuracy claim.

**Not settled.**
- A veteran with fewer than two recent games still gets the rookie prior, looked up
  against the CURRENT draft class, so as undrafted. Deshaun Watson, CLE's QB1, is 6.0
  (0.18 on the week 1 board).
- The team-position budget is still not applied in season.
- Production runs five seeds; these were one.
- The dry run predates DEN@KC.
- A Wednesday run sees only a partial injury report.
- Two tests fail in the nfl-api image on pandas 2.1 (`include_groups` in
  `evaluate.summarize` / `summarize_quantiles`). They predate this work and are off the
  production path.

## Weeks 1 and 2, scored prospectively (2026-09-25)

The first read on whether #23's fixes hold up on football nobody had seen. Same
scorer, same population rules, frozen rows — week 1 was projected by the old
preseason path, week 2 by the fixed in-season one.

| | n | MAE | bias | naive MAE | skill | band |
|---|---|---|---|---|---|---|
| Week 1 (preseason path) | 302 | 4.485 | −2.44 | 4.804 | +0.07 | 63% |
| Week 2 (in-season path) | 354 | **3.870** | **−0.71** | 4.627 | **+0.65** | 70% |
| Both | 656 | 4.153 | −1.50 | 4.704 | +0.40 | 67% |

The −2.6 bias is gone, as predicted when the availability weight was divided out
by hand (#22 put the corrected week-1 figure at 4.29 / −1.17; week 2 came in
better than that). Skill over naive went from inside the noise to +0.65 on 354
player-weeks, against a backtest average margin of ~0.05 — one week, so read it
as "not broken", not as the new number. Band coverage 70% is the usual ~72%.

**QB is still the position that loses to naive**: 7.599 MAE against 6.915 over
70 player-weeks, bias −4.18, band coverage 0.529. Third independent look at the
same thing (#22, the week 2 dry run, now this): the level is too low and the
ordering is fine. Everything else beats naive — RB +0.71, TE +0.64, WR +0.55.

Nothing was tuned on this. It is the record, and the QB gap is the next thing
worth a real experiment rather than a fix.

## Four weeks in, and the bias that should not be corrected (2026-10-05)

Weeks 1-3 are frozen in `projection_accuracy`; week 4 is scored here with
`experiments/live_week_accuracy.py` (Sunday games only, before the Monday
nighter), same rules as the scorer.

| week | n | MAE | bias | naive | skill | band |
|---|---|---|---|---|---|---|
| 1 (preseason path) | 302 | 4.485 | −2.44 | 4.804 | +0.07 | 63% |
| 2 | 354 | 3.870 | −0.71 | 4.627 | +0.65 | 70% |
| 3 | 351 | 4.037 | −1.55 | 4.181 | +0.15 | 70% |
| 4 (partial) | 328 | 3.917 | −1.68 | 4.061 | +0.14 | 73% |
| **weeks 2-4 pooled** | **1033** | **3.942** | −1.30 | 4.293 | **+0.312** | 71% |

Pooled Spearman 0.680. Every position now beats naive on the fixed path: QB
+0.57, RB +0.44, WR +0.27, TE +0.13. **That reverses #24's read on QBs** —
which was dominated by week 1, where the availability discount hit QBs hardest.
Ordering is a tie with naive at the top of each position (top-12/24/36 overlap
19/19, 46/45, 51/51, 16/14 across three weeks), so the win is level, not sort.

**The board runs about 1.3 points low, and that is correct.** First question was
whether it is the population: the scorer only scores players who played, and if
the ones who sit are the low projections, conditioning on "played" selects
upward. It is not that. The miss holds at every projected level — −1.33 under 2
points, −1.83 in the 6-8 band, −1.02 above 12 where 93% play — and holds at
−1.51 among the 247 players who appeared in all three weeks.

So I fit the correction and tested it the only way worth testing: per-position
offset and scale fit on the weeks already played, applied to the NEXT week only.

| target week | MAE | + additive | + scaled | bias → |
|---|---|---|---|---|
| 3 (fit on wk 2) | 4.038 | 4.128 (**+0.090**) | 4.097 (+0.059) | −1.55 → −0.83 |
| 4 (fit on wk 2-3) | 3.917 | 4.051 (**+0.134**) | 4.024 (+0.108) | −1.68 → −0.55 |

**Halves the bias, costs MAE both times — rejected.** The reason is structural
rather than a tuning failure: MAE is minimised by the conditional MEDIAN, and
fantasy scoring is right-skewed, so the mean of outcomes sits above the median
every week. A projection that minimises absolute error is *supposed* to come in
under the mean. Shifting everything up pays for a handful of big games with a
small error on the many quiet ones. The naive baseline carries the same sign for
the same reason (−0.38, −0.62) and is smaller only because it averages actuals.

Per-position bias is also not stable enough to lean on: QB −1.28, −3.28, −2.00
across weeks 2-4; RB +0.52, −0.44, −1.64. The correction fit on week 2 would
have pushed RBs DOWN into week 3 (−0.52) and been roughly a no-op into week 4.

**Consequence for anything that needs expected points** (DFS, season totals):
take it from the simulator, whose mean is anchored but whose shape carries the
skew, not from shifting the point projection. The point projection is a median
and should stay one. Nothing to change in the model.

Harnesses: `experiments/live_week_accuracy.py`, plus the two throwaway scripts
this measured with (bias by projected level, and the next-week-only correction
fit) — the numbers above are the record.

## Training on the current season alone (2026-10-05)

The proposal, at week 5 of 2026: train only on this season. Four weeks is the
most relevant football there is, and everything the old seasons taught is
already baked into it. Testable without waiting — 2025 is finished, so train on
its first four weeks ALONE and project weeks 5-18.

Three windows, identical evaluation rows, model never sees the week it predicts:

| training window | rows supplied | rows TRAINED on | MAE (5 seeds) | MAE (1 seed) | bias |
|---|---:|---:|---:|---:|---:|
| history only, 2020-2024 | 152,185 | 91,810 | **4.120** | 4.094 | −0.24 |
| history + current (production today) | 156,594 | 94,628 | 4.125 | 4.148 | −0.27 |
| current season only, 2025 wk1-4 | 4,409 | 2,619 | 4.217 | 4.159 | +0.11 |

*(Corrected 2026-10-05: this table first reported only the supplied counts as
though they were the training set. `train_ensemble` then drops seasons before
`min_season`, the AVG rows and low-activity players, so the figures the networks
actually saw are the second column — a ~35x gap between the widest and narrowest
window, not the ~34x the first column implies. No measurement changes.)*

**Rejected: +0.097 against history-only at production's five seeds** — twice the
measured gain of the seed ensemble itself (#0, 0.04), and it would be paid every
week of the season. Note the single-seed column UNDERSTATED the gap (+0.066),
which is the noise floor doing exactly what it is documented to do; the 2024
replication (single seed) read +0.023. One seed cannot settle a 0.1 question.

**Retrained weekly, as the idea would actually run**, the answer does not
change. `experiments/train_window_expanding.py` retrains for each target week on
that season's prior weeks only, so week 12 gets eleven weeks rather than four:

| | current-only | fully trained | delta |
|---|---|---|---|
| weeks 5-10 | 4.283 | 4.198 | +0.086 |
| weeks 11-18 | 4.077 | 4.022 | +0.055 |
| pooled, 4,554 player-weeks | 4.161 | 4.094 | **+0.068** |

It won 2 of 14 weeks. The gap narrows as rows accumulate — +0.16 in week 5,
+0.04 by week 18 with 17,556 rows — and never closes, because 17k rows is still
a tenth of 150k and the mapping being learned is not season-specific.

**Why the intuition does not pay, and what it got right.** The current season
reaches a projection TWICE: through training rows, and through the player's own
rolling last-3/last-5 form, which in week 5 is entirely current-season football
whatever the training window. The features carry the recency; the training set
only estimates the stable mapping from form to next week. Which is why the third
arm matters most here: adding the four weeks to 150k historical rows moved
nothing (+0.005, a wash). The current season's rows are neither the problem nor
the lever.

The part the proposal gets right is operational: 4k rows trains in seconds
against most of an hour, so a current-season-only weekly run would be ~5 minutes
rather than ~42. That is a real saving and it buys a 0.1 MAE regression — and
the model cache (NFL-API) already makes the second and later runs of a week
cheap, so the saving is mostly gone anyway.

**Untested middle ground**, and the natural next probe: keep the history but
shorten or down-weight it — `config.TRAINING_MIN_SEASON` is already the knob
(2020 today), and sample weighting by season is a small change. "Recent football
matters more" is a different claim from "only recent football", and this
experiment does not speak to it.

Harnesses: `experiments/train_window.py` (one training window per arm),
`experiments/train_window_expanding.py` (weekly retrain). CAVEAT found while
scoring: the evaluation frame holds ~1.25 rows per player-week — the dataset's
depth-chart merge duplicates some rows — so n is inflated. Every arm shares the
same rows, so the deltas stand; absolute MAE here is comparable with other
entries, which share the flaw.

## How far back should training reach (2026-10-05)

#26's untested middle ground: "recent football matters more" rather than "only
recent football". `config.TRAINING_MIN_SEASON` is the dial, 2020 today. Same
harness and discipline — identical evaluation rows, five seeds, each arm trained
on its prior seasons PLUS the test season's first four weeks, exactly the shape
production uses. Tested on 2025, so the windows read one year back; the
min_season a 2026 run would set is in the second column.

| history kept | 2026 equivalent | rows supplied | MAE | bias | vs best |
|---|---|---:|---:|---:|---:|
| 1 prior season | min_season **2025** | 26,826 | **4.110** | −0.25 | — |
| 3 prior seasons | min_season 2023 | 71,230 | 4.119 | −0.22 | +0.009 |
| 6 prior seasons | min_season 2020 (today) | 135,617 | 4.123 | −0.17 | +0.013 |
| 2 prior seasons | min_season 2024 | 48,845 | 4.126 | −0.22 | +0.016 |

**The entire spread is 0.016 MAE — a tie four ways.** The ordering is
non-monotonic (1, then 3, then 6, then 2), which is what noise looks like; a
real trend in either direction would not jump about. Spearman is flat too (QB
0.937-0.943, RB 0.971-0.973, WR 0.966-0.968, TE 0.960-0.961), so the ordering
does not care either.

Read together with #26, the picture is specific rather than vague: 2,619 training
rows (four weeks) is genuinely too thin, at +0.097. Somewhere under 20,000 that
cost disappears, and from one prior season onward more history is neither help
nor harm. The mapping from form to next week is apparently stable enough that
2019 football teaches it as well as 2024 football, and shallow enough that one
season's worth of examples pins it down.

**Where the win actually is: training cost.** One prior season supplies 26,826
rows against 135,617, so a five-seed run trains in roughly a fifth of the time —
the weekly job's ~35 minutes of training becomes ~7. That is the same saving
#26's proposal chased, at none of its accuracy cost, which makes
`TRAINING_MIN_SEASON = CURRENT_SEASON - 1` the version of that idea worth
shipping.

**Not shipped yet, deliberately.** One test season cannot carry a production
change on a 0.016 spread; the 2024 replication is running. Two things this does
not measure and would want checking before it ships: whether a thinner window is
more seed-sensitive (variance across the five members, not just their mean), and
what it does to the preseason board, where #20's harness trains on prior seasons
only and a one-season window would be 18k rows rather than 90k.

Harness: `experiments/history_depth.py` (DEPTHS, TEST_SEASON, SEEDS).

## History depth, retrained every week (2026-10-07)

#27 compared training windows by training ONCE on prior seasons plus weeks 1-4
and projecting weeks 5-18. Fine for "which window is better for week 5", useless
for week 12: no arm had weeks 5-11 in training either, while production retrains
every Tuesday. The owner caught it. This retrains each arm at each target week on
everything played before it, so the only difference left is how far back history
reaches.

2025, one prior season (the 2026 equivalent of min_season 2025) against six
(min_season 2020), three seeds, identical player-weeks per checkpoint:

| week | current-season rows | 1 prior season | 6 prior seasons | delta |
|---|---:|---:|---:|---:|
| 5 | 4,409 | 4.133 | 4.071 | **+0.062** |
| 9 | 8,324 | 4.102 | 4.087 | +0.014 |
| 13 | 12,256 | 3.807 | 3.874 | **−0.067** |
| 17 | 16,488 | 4.118 | 4.112 | +0.006 |
| **pooled** (1,317 player-weeks) | | **4.0355** | **4.0338** | **+0.002** |

**A tie, and the swing is the point.** Pooled, the two windows are 0.002 apart —
nothing. Week to week the delta moves across a 0.13 range and changes sign three
times. Any single week would "prove" whichever answer that week happened to
favour, which is exactly why a weekly MAE readout is not evidence about a
training window.

There is a shape under the noise, weak but consistent with #26: the shallow
window trails early (+0.038 over weeks 5 and 9) and leads late (−0.030 over 13
and 17), as current-season rows accumulate from 4k to 16k. Same curve the
current-season-only arm traced, just shallower, because one prior season already
supplies enough examples that the current season's volume barely matters.

**What this does NOT say.** It compares one prior season + current against six +
current. It does not test the current season ALONE — that is #26 (+0.097 single
train; +0.068 pooled when retrained weekly, winning 2 of 14 weeks).

### Decision, 2026-10-07: the live model trains on the current season alone

The owner's call, taken with these numbers in hand rather than against them:
from week 6 of 2026 the weekly board trains on 2026 only, on the reasoning that
a board for this season should describe this season, that weekly ebbs and flows
make any single week's comparison unreliable, and that older football is a
stand-in for current-season data rather than a substitute for it.

Recorded here with its measured price — **~0.07-0.10 MAE**, by the two
experiments that tested that window — because a ledger that only keeps the
decisions the numbers made is not much of a ledger. The preseason board keeps
the historical window: with no current-season football there is nothing to train
on, which is the same reasoning read the other way.

**The trade is now measured rather than argued.** NFL-API's weekly job trains the
historical window as a SHADOW model, projects the same week with it, and
`GET /projections/shadow/accuracy` grades both against the frozen actuals on
identical player-weeks. Prospective, one population, no hindsight — the standard
the board is already held to (#22, #24, #25). Expect a verdict around week 11-12
of 2026: the margin at stake is the size of a week's noise, so it takes several.

Harness: `experiments/history_depth_weekly.py` (DEPTHS, WEEKS, SEEDS,
TEST_SEASON — runs on 2026 as weeks accrue).

---

## The trench report: line play, rebuilt from public data (2026-10-14)

Hand-charted line play — offensive lines against defensive fronts, receivers
against coverages — is sold as a subscription product and not obtainable. The
question here is whether the same three axes can be rebuilt from nflverse and
whether they survive the bar the rest of this file is held to.

One of the three was already settled. "Coverage matchup" found tight end works
(MAE 18.67 -> 17.92) and wide receiver is actively worse, because yards allowed
to WR is diluted across a whole secondary while the thing that would matter —
which corner travels with him — is not published. **The trenches had never been
tested here at all**, and they had the better prior: a line is five players
under contract together rather than a weekly assignment.

### Is a line a trait? Yes, and it is the offence's trait, not the defence's

Split-half within season, 52,884 non-QB carries and 80,753 dropbacks, 2022-25.
Adjusted line yards is FO's banding — losses 1.2x, 0-4 yards fully, 5-10 half,
11+ not at all. Harness: `experiments/trench_stability.py`.

| metric | offence r | defence r |
|---|---|---|
| pressure rate (sack or hit per dropback) | **0.557** | 0.338 |
| sack rate per dropback | **0.477** | 0.068 |
| adjusted line yards | 0.393 | 0.428 |
| stuff rate | 0.313 | 0.441 |
| second-level yards (5-10) | 0.425 | 0.314 |
| open-field yards (11+) | 0.296 | 0.271 |

Year over year the asymmetry widens: offence holds 0.25-0.42, defence collapses
to 0.11-0.28. **A pass-protection unit is the single most stable offensive trait
measured in this file**, and a defensive pass rush measured by sacks allowed per
dropback is not a trait at all (r = 0.068 — a coin flip inside one season).

Every one of these clears defence-vs-position (0.06-0.25) and none reaches
success-rate-allowed (0.618). And **cutting by gap destroys it again** —
team x run-location/gap falls to 0.091 for line yards and 0.138 for stuff rate,
the same collapse personnel and formation produced in September. "They cannot
block the left side" is not a measurable thing at public resolution; roughly 20
carries per gap-half is why.

### Does the front move a weekly rushing projection? Yes, barely

The adjusted-defence work closed TE and RB receiving and left rushing open in
writing, because its factor was built from receiving yards allowed. This is the
rushing-specific replacement: 2,865 back-games from week 6 on, baseline is the
back's own trailing five games, treatment multiplies by a shrunk opponent
factor from weeks strictly before the game. Harness: `experiments/trench_value.py`.

| treatment | MAE | corr | MAE wins |
|---|---|---|---|
| baseline (trailing 5) | 24.672 | 0.508 | — |
| **def adjusted line yards** (k=150) | **24.548** | 0.514 | **4/4** |
| def yards per carry (k=150) | 24.577 | 0.514 | 3/4 |
| def stuff rate, inverted (k=150) | 25.118 | 0.503 | 0/4 |

It passes the stated bar — four seasons of four, both error and correlation —
and it is worth **0.124 yards of MAE on a 24.7 baseline, half a percent**. The
banding is also nearly redundant: plain yards per carry allowed captures 0.095
of the 0.124, so the FO weights are buying 0.03 yards. Shippable, in the same
sense RB receptions was shippable, and not a lever.

### Yards per carry is not projectable, and the trap that looks like it is

Season N to N+1, 111 back-season pairs with 80+ carries in both, leave-one-
season-out. Every carry splits four ways with no overlap — negative, the first
four yards, 5-10, 11+ — so their means sum to YPC exactly and the baseline and
treatment see the same numbers arranged differently.
Harness: `experiments/trench_decomposition.py`.

| predictor | MAE | corr |
|---|---|---|
| **league mean YPC (no skill at all)** | **0.488** | 0.000 |
| own YPC | 0.507 | 0.007 |
| own YPC + carries | 0.517 | 0.013 |
| decomposed, 4 bands | 0.526 | 0.027 |
| decomposed + own team's line | 0.526 | 0.055 |
| decomposed + next team's PRIOR-year line | 0.550 | -0.080 |
| next team's prior-year line alone | 0.521 | -0.313 |

**Nothing beats knowing nothing.** The league mean wins outright; a back's own
YPC is worse than ignoring him, and taking it apart makes it worse still. Piece
stability is real but low (open field 0.347, first four yards 0.295, YPC 0.194)
and none of it converts.

The entry worth keeping is the one that got caught. Regressed on the line he
*actually* ran behind next season, the model looked transformative — MAE 0.416
against 0.488, correlation **0.514**, the largest jump anywhere in this file.
It was circular: a team's line yards are computed from its backs' carries, so
for a lead back the feature is mostly his own target. Recomputing the identical
feature with his own carries removed collapses it to **corr 0.022, MAE 0.531**,
worse than the mean. The prior-year version — the only one a board could
actually use — predicts *negatively*.

### What this adds up to

That is now five independent tests landing the same way: defence-vs-position,
success-rate-allowed, Next Gen Stats, FTN charting, and now the trenches.
Stability is easy to find and incremental signal is absent, and the trenches
were the strongest version of the hypothesis — the most stable offensive trait
in the file, and it still does not move a projection.

The reason is the same each time and now has a clean statement. **These metrics
describe how yards were produced; projections are driven by how many
opportunities a player gets.** Efficiency reverts to the positional mean
(#17, #8), so a model that predicts opportunity and applies a mean rate is
already extracting what is there. The one place the trenches earn a line of
code is the weekly opponent factor above, which is half a percent.

**What is not closed, and is a different experiment**: pass protection is the
most stable trait here (0.557) and nothing in this entry tested it against
anything — it belongs to quarterback projection and to sack/pressure props, not
to rushing yards. The defence side of it being a coin flip (0.068) is itself a
finding worth remembering before anyone builds a pass-rush matchup.

---

## Line ratings shipped, and which half of a matchup is real (2026-10-14)

Follow-up to the entry above, which measured the trenches and found them
stable but nearly inert as a projection input. The owner's call was to build
them anyway as a **rating** — four 0-100 grades per team (50 = league average,
15 points to a standard deviation, the same scale as the team power ratings)
plus a matchup number per game, with an endpoint and a dashboard.

`nfl_projections/trenches.py`: `run_block` and `pass_block` for the offence,
`run_stop` and `pass_rush` for the defence, each opponent-adjusted by the same
alternating-means construction the power ratings use, shrunk toward the league
mean by play count, and blended with last season's regressed grade until enough
football has accrued. Per-unit carry-over because the two sides do not regress
alike: 0.40 for protection, 0.20 for a run-stopping front, from the
year-over-year numbers above. Pass rush is built from pressures and not sacks,
because a defence's **sack rate allowed per dropback has a split-half r of
0.068** — the grade would have been noise.

### The backtest, and the half that carries information

1,540 team-games, weeks 6 on, 2022-25, every rating rebuilt from the weeks
before the game it is graded against. Harness: `trenches.backtest()`, so the
claim ships with the code rather than living here alone.

| band | team-games | yds/carry | vs own avg | rush yards | stuffed |
|---|---|---|---|---|---|
| fade (22) | 134 | 3.93 | -0.14 | 94.8 | 23.3% |
| lean down (36) | 318 | 4.12 | -0.03 | 98.5 | 20.5% |
| neutral (50) | 642 | 4.09 | -0.15 | 101.8 | 18.5% |
| lean up (64) | 312 | 4.35 | -0.07 | 115.8 | 16.7% |
| smash (75) | 134 | 4.67 | +0.09 | 125.5 | 15.8% |

The ladder is monotone in every raw column — 95 rushing yards at the bottom and
126 at the top. **And almost all of that is which offence is running, not which
matchup it drew.** Subtract each team's own season average and the spread
collapses: correlation with `ypc_vs_own` is 0.035, against 0.137 with raw yards
per carry.

Splitting the rating into its two halves says it plainly:

| half of the matchup | corr with game yds/carry | corr with yds/carry vs own avg |
|---|---|---|
| own run-block grade | +0.091 | **-0.051** |
| opponent run-stop grade | -0.102 | **-0.100** |

**The offence's own line adds nothing once you know how the offence has been
running** — which is exactly right, because its own line is what produced that
form. The opponent's front is the half that is new information, and it holds up
on its own:

| opponent front | team-games | yds/carry | vs own avg |
|---|---|---|---|
| very soft | 249 | 4.42 | **+0.14** |
| soft | 355 | 4.28 | -0.01 |
| average | 341 | 4.17 | -0.09 |
| stout | 382 | 3.99 | **-0.27** |
| very stout | 213 | 4.13 | -0.15 |

Monotone across four of five bands, **0.43 yards per carry from very soft to
stout** — about nine rushing yards on a typical workload. That is a bigger
number than the 0.124 MAE the weekly multiplier bought, and the two are
consistent: the multiplier was measured on top of a projection that already
knew the back's trailing production, so it could only ever be paid for the
opponent half. The top band reverting is the thing to watch rather than
explain; it is 213 games and the two stout bands are a tenth of a yard apart.

**Consequence for the product.** The dashboard leads with the opponent column
and ships both ladders, because the combined rating is the flattering one and
showing it alone would overstate what a number in the eighties means. The
`/trenches/backtest` endpoint returns the own-vs-opponent split next to the
bands so no caller can display one without the other.

### A speed note worth keeping

The readable opponent adjustment — groupby, then `Series.map` per team per
iteration — took 3.8 seconds a fit, and the weekly backtest pays for 156 of
them. Rewritten against integer team codes and `np.bincount` it is 0.08
seconds, identical arithmetic, a 47x difference. That is what moved line
ratings from a batch job to something an endpoint can compute inside a request.

---

## Positional scarcity: replacement level, and a flat bar that cannot work (2026-10-14)

The board ranks by projected points, which answers who scores more and never
who is harder to replace. `nfl_projections/scarcity.py` adds the missing half,
from 11 seasons of weekly stats scored as FanDuel half-PPR (2015-25), 12-team
slots of 1 QB / 2 RB / 3 WR / 1 TE.

| position | last starter | replacement | usable-week rate (>=15) |
|---|---|---|---|
| QB | 19.30 | 18.09 | **50.0%** |
| RB | 11.60 | 10.90 | 15.7% |
| WR | 10.20 | 9.82 | 12.4% |
| TE | 8.41 | 7.70 | **5.4%** |

**The flat 15-point startable bar cannot be used across positions.** Pooled, it
marks 26 quarterbacks a season as usable and **0.7 tight ends** — a bar that
nearly every starting quarterback clears and essentially no tight end does is
not measuring startability, it is measuring that quarterbacks score more. The
replacement levels are the fix: value over replacement puts a 12-point tight end
ahead of a 20-point quarterback, which is the ordering a lineup actually wants.

The simulator now reports both, since it already produces a distribution per
player-week: `p_usable` at the flat bar, and `p_start` against the positional
replacement level. On a worked example a tight end projected at 12.1 points has
`p_usable` 0.31 and `p_start` 0.63 — the same player, and only the second
number answers "should he be in the lineup".

### Top-100 composition, and a check against a published version

The diagnostic behind scarcity is how a position's scorers are spread. Under
FanDuel scoring, 2015-25, the top 100 by season total holds 27-33
quarterbacks, 23-31 backs, 34-42 receivers and 4-9 tight ends.

Recomputed under **full PPR** for 2024 it comes out 24 QB / 28 RB / 40 WR / 8
TE, against a publicly stated 26 / 27 / 39 / 8 for the same cut — within one or
two at every position, and an exact match at tight end. Worth recording because
it independently confirms the construction: whoever published those numbers was
working in full PPR, and this implementation reproduces them.

Harnesses are the module itself (`scarcity.baselines()`), plus
`tests/test_scarcity.py` for the invariants that matter — replacement below the
last starter, positional ordering, composition summing to the cut.

---

## Blocking scheme: the most stable trait in this file, and it predicts nothing (2026-10-14)

The coaching breakdowns that prompted the trench work classify all 32 teams by
blocking scheme — outside zone, power gap, split zone — attach the label to the
offensive line coach, and update it when a team switches. nflverse has no scheme
field, but it charts where every carry went (`run_location` left/middle/right,
`run_gap` guard/tackle/end), and zone and gap schemes distribute carries
differently by construction: zone attacks the edge, gap concentrates between the
tackles behind pulling blockers. So the direction profile is a proxy for the
label. 102,036 charted non-QB carries, 2018-25. Harness:
`experiments/blocking_scheme.py`.

League-average mix: 27.0% middle, 26.0% guard, 23.8% tackle, 23.2% end.

### The profile is the most persistent trait measured anywhere here

| metric | split-half r | year-over-year r |
|---|---|---|
| interior share (guard + middle) | **0.702** | 0.622 |
| middle share | 0.638 | **0.716** |
| outside share (either end) | 0.649 | 0.506 |
| tackle share | 0.565 | 0.632 |
| left/right balance | 0.194 | 0.143 |

**Year over year this beats everything else in the file.** For comparison: team
success rate allowed 0.618 split-half, offensive line grades 0.25-0.42
year-over-year, team offensive SRS 0.489. A team's run-direction identity
survives roster turnover and coaching changes better than its ability to block
does, which is exactly what you would expect if it is a scheme and not a
performance.

**Left/right balance is not a trait** (0.194 / 0.143). Teams do not
persistently favour one side. That is the second independent death of the
"they cannot block the left side" idea — the first was gap-level rating
stability collapsing to 0.09 in the entry above.

### And it tells you nothing about the backfield

The question worth asking, given that every result here says projections move on
opportunity and not efficiency: do zone teams ride one back while gap teams
rotate? Leave-one-season-out, 224 team-season pairs, predicting next season's
lead-back carry share and the Herfindahl concentration of the backfield.

| predictor | next lead share (MAE) | next backfield HHI (MAE) |
|---|---|---|
| **league mean (no skill)** | **0.1211** | 0.1156 |
| own prior value | 0.1209 | **0.1136** |
| own + scheme profile | 0.1228 | 0.1152 |
| scheme profile alone | 0.1240 | 0.1184 |

Adding the profile makes both targets **worse**, and the profile alone is worst
of all with negative correlation. Same season the association exists and is
tiny: interior share against backfield concentration is r = 0.142, outside share
-0.134. Interior-heavy teams concentrate carries very slightly more, and it does
not survive a year.

**A second finding fell out of the baseline.** A team's backfield concentration
barely predicts its own next season either — own prior value beats the league
mean by 0.0002 on lead share (corr 0.104). *Team* backfield structure resets
year to year even though the scheme does not. That matters for how the share
model is built: it works at the player level off depth rank and the player's own
history, and this says the team-level structure was never the thing carrying it.

### Six for six

Defence-vs-position, success-rate-allowed, Next Gen Stats, FTN charting, the
trenches, and now blocking scheme. **Stability keeps being easy to find and
incremental signal keeps being absent**, and this is the cleanest instance:
the single most persistent trait measured in this file predicts nothing it was
pointed at.

The pattern has a mechanism, stated once more because it has now survived six
tests. These metrics all describe **how** production happened — which direction,
which personnel, which coverage, which scheme. A projection needs **how much**:
carries, targets, and who owns the role when it changes hands. Context is
descriptive and stable precisely because it is not the causal quantity, and
stability is not evidence of predictive value. It is often the opposite: the
most stable things are the ones that have already expressed themselves in the
numbers a model is fitted on.

**What this does NOT close.** Scheme change attributed to a coaching hire is
untested and untestable from nflverse, which has no offensive-line-coach field —
a change in the profile is detectable, its cause is not. And nothing here tested
scheme against a *back's fit*, only against team-level structure.

---

## Vacated share: the assumption held, and the gap was somewhere else (2026-10-14)

`shares.py` carried one untested assumption, written in its own design notes:
vacated share is *"handled implicitly: a new depth chart re-ranks the
returners/newcomers and shares renormalize within the team, so a departed
starter's touches flow to whoever now sits at that rank."* This tests it, because
share is the quantity that actually moves projections and an untested assumption
sitting inside the model that works is worth more attention than another matchup
feature. Harness: `experiments/vacated_share.py`, 2018-25.

Teams vacate a lot: mean 34.5% of the carry pool and 31.2% of the target pool
every offseason, up to 90%.

### For returners, the assumption is right

1,813 returning player-seasons, leave-one-season-out, predicting this season's
share.

| predictor | carry share MAE | target share MAE |
|---|---|---|
| own prior share | 0.0357 | 0.0331 |
| own prior + rank | 0.0357 | 0.0332 |
| **+ team's vacated share** | 0.0358 | 0.0331 |
| + vacated x prior interaction | 0.0352 | 0.0331 |

**Knowing how much share a team vacated tells you nothing about its returners.**
The coefficient is +0.018 carry share per unit vacated: a team losing 30% of its
carries lifts a returner by half a percentage point. Splitting on teams that
vacated 15%+ of a pool changes nothing (0.0348 to 0.0349).

### Because the touches do not go to the returners

| vacated | team-seasons | mean vacated | share now held by players who were not here |
|---|---|---|---|
| 5-20% | 47 | 0.125 | 0.232 |
| 20-40% | 59 | 0.284 | 0.295 |
| 40-60% | 42 | 0.509 | 0.494 |
| 60%+ | 45 | 0.726 | 0.633 |

**Vacated share is replaced about one for one by players who were not on the
roster the year before** — median absorption 1.04 for carries, 1.03 for targets,
correlation 0.688 and 0.706. The next man up does not inherit the role; a new
body does, and at the low end newcomers take *more* than was vacated, displacing
returners at the margin.

So the model's renormalization is fine, and the prose describing it was
misleading in an instructive way: the player who now sits at rank 1 is usually
someone who was not there last year. Which relocates the question entirely —
**the accuracy of a big-vacancy team depends on getting the newcomer right.**

### The gap, and what shipped

Rookie arrivals already have a prior that wins (`rookies.py`). Veterans who
change teams had nothing: the model gave an arriving veteran his new depth rank
and discarded the share he held elsewhere. 661 such player-seasons.

Swept through `backtest_shares`, the production harness:

| mover_blend | overall MAE | changed roles | movers only |
|---|---|---|---|
| 0.0 (old model) | 0.0658 | 0.0700 | 0.0654 |
| 0.2 | 0.0652 | 0.0688 | 0.0627 |
| **0.3 (shipped)** | **0.0651** | **0.0686** | **0.0622** |
| 0.5 | 0.0652 | 0.0689 | 0.0629 |
| 0.6 | 0.0654 | 0.0693 | 0.0638 |
| 0.8 | 0.0660 | 0.0705 | 0.0666 |

**Movers improve 4.9%** (0.0654 to 0.0622) and overall share error 1.1%. Shipped
at `MOVER_BLEND = 0.3` in `project_shares`, with `backtest_shares` now reporting
`rankonly` beside `model` so the gain stays visible.

**A correction worth recording.** A standalone regression on the same data put
the weight near 0.64 and the improvement near 47%, because it compared against an
ordinal depth rank rather than the fitted rank-share prior the model actually
uses. Against the real baseline the gain is a fifth of that. The lesson is the
one this file keeps relearning from the other direction: a new feature's value is
whatever it adds to the thing already shipped, and any weaker baseline will
flatter it.

### Where this sits

This is the first feature in a while that is about **how much** rather than how,
and it is also the first to survive since the usage-decomposition work. That is
not a coincidence — it is the same lesson as the six negative results, read
forwards: a mover's previous share is an opportunity quantity, and a depth rank
tells you where he sits while saying nothing about how big the role is.

**Still untested**: whether a mover's share should be scaled by the specific
vacancy he is walking into. Adding team vacancy on top of old-team share was
flat here (0.0366 to 0.0371 for carries), but the test was crude — a signing's
share should plausibly depend on what opened up, and 661 rows split by vacancy
size gets thin fast.

---

## The starter is out, and the backup was being charged a backup's rate (2026-10-14)

The season-scale vacated-share work found that a vacated role transfers to
whoever holds it next, and that `shares.py` handles it by re-ranking. **The
weekly path does not re-rank.** `predict.py` zeroes a player who is OUT and then
hands every remaining teammate a multiplier from his static depth-chart rank —
so a back listed RB2 kept the RB2 discount (x0.80) in the one week where he was
the only back on the roster. The discount was also one-sided: `role_mult` was
applied only when below 1.0, so nothing in the weekly path could ever revise a
projection upward.

This is the most actionable call in weekly fantasy, and it was wrong in the
direction that matters. Harness: `experiments/starter_out.py`, 2018-25, 63,311
player-weeks at entering rank 1-4. Who the starter is comes from trailing usage
only; who actually played comes from snap counts, which state it rather than
inferring it from an injury report. **The entering rank-1 player missed 16.3% of
team-position-weeks** — this is not an edge case.

### What a promoted backup actually does

Usage per player-week (carries + targets; attempts for QB), starter in vs out:

| position | rank | starter in | starter out | factor | n |
|---|---|---|---|---|---|
| RB | 2 | 8.28 | 16.11 | **1.95** | 486 |
| RB | 3 | 4.33 | 9.04 | 2.09 | 419 |
| WR | 2 | 5.73 | 6.65 | 1.16 | 517 |
| TE | 2 | 1.78 | 3.28 | 1.84 | 592 |
| QB | 2 | 5.96 | 30.17 | **5.06** | 501 |

Within player — same man, both situations, 952 player-seasons — the median
factors are RB 1.96, TE 1.78, WR 1.29, QB 2.72.

### What survives out of sample, which is less

Factors fitted on 2018-23 and tested on 2024-25, predicting a promoted backup's
usage. Three predictors: his trailing usage times the production discount, times
nothing, and times the fitted factor.

| position | rank | n | actual | production discount | no discount | fitted factor | fit |
|---|---|---|---|---|---|---|---|
| RB | 2 | 108 | 15.75 | 8.543 | 7.387 | **6.628** | 1.42 |
| RB | 3 | 100 | 8.96 | 6.575 | 5.510 | **5.409** | 1.43 |
| WR | 2 | 178 | 6.78 | 2.665 | **2.594** | 2.594 | 1.00 |
| WR | 3 | 176 | 4.69 | 2.558 | **2.337** | 2.342 | 0.97 |
| TE | 2 | 140 | 2.99 | 2.087 | **1.837** | 1.852 | 1.10 |
| QB | 2 | 139 | 29.85 | 23.325 | 13.331 | **11.901** | 1.29 |

Two things, in order of size:

**Dropping the discount helps at every position and every rank.** RB2 8.54 to
7.39, QB2 23.33 to 13.33 (-43%), TE2 2.09 to 1.84, WR2 2.67 to 2.59. There is no
case where charging a promoted player a backup's rate is right.

**An extra factor beyond that helps only the backfield and quarterback** — RB
x1.42, QB x1.29. For receivers the fitted factor came back at 1.00 and 0.97, and
for tight ends 1.10 made 2024-25 slightly worse. Left at 1.0 for both.

The within-player medians (RB 1.96, QB 2.72) are **not** what shipped, because
they are ratios of a player's own means and the out-of-sample fit against
trailing usage is the number that holds up. That is the second time in two
entries that a within-group statistic overstated what survived validation — it
is worth treating as a standing bias of this kind of measurement rather than two
separate mistakes.

### Shipped

`roles.effective_rank(depth_rank, out_ranks)` promotes a player past absences
**strictly above** him, so an RB1 out moves RB2 to 1 and RB3 to 2, and an RB2 out
promotes nobody. `roles.promotion_multiplier` returns the total adjustment
relative to the static multiplier, so RB2 with his starter out is
(1/0.80) x 1.42 = **x1.78**, and QB2 is (1/0.35) x 1.29 = **x3.69**.

`predict.py` builds the absence map once per week (`_out_ranks`, cached — it
needs a roster pass and is then asked for on every player) and applies the
promotion after the static discount, scaling the components as well as the
points so a yardage prop cannot disagree with the fantasy number.

**Not yet validated end to end.** This is measured against actual usage, not
against the board's own FanDuel-point error through the trained model. Weeks
where a starter sits are about 16% of player-weeks, so the effect on aggregate
weekly MAE will be small even if the affected projections move a lot; the
prospective scorecard is where that shows up.

*Settled the next day — see "The promotion fix, scored the way the board is
scored". The re-ranking pays in points (+4.8% on the player-weeks it touches,
16.0% of them, almost exactly the share predicted above). The fitted factors
did not survive the change of currency: RB x1.42 is now 1.0, because it was
fitted to hit a promoted back's mean usage and the mean is not what MAE or the
ordering wants. QB x1.29 stands.*

## The mid-season mover, and two instruments that lie about him (2026-10-14)

The season-scale finding was that the SIZE of a role travels with the player: an
arriving veteran's share on his old team carries information his new depth rank
does not, and `shares.py` ships that as `MOVER_BLEND = 0.3`. That model runs once
a year over free agency. The weekly path has the same case and no term for it —
`predict.py` reads a mover's last game row, whose rolling usage and scoring form
are entirely his old team's, and then applies `per_game_role_multiplier` for his
new rank. So the obvious next move was the weekly analogue of MOVER_BLEND.

It is not there. The mechanism is real and the fix does not pay.

Harness: `experiments/midseason_mover.py`, 2013-25 (snap counts start in 2013),
**1,533 mid-season team changes** at skill positions — RB 422, WR 615, TE 325,
QB 171. Identification is from WEEKLY ROSTERS, not stats rows: a traded player
who never touches the ball still appears on the new roster, and conditioning on
having a stats row keeps only the moves that worked out. Those moves yield **369
arrival player-weeks** in the four weeks after one — he played, he had in-season
form for the path to carry, and the new chart listed him — against **62,853
control player-weeks** for everyone who did not just move.

### Two instruments that lie, found the hard way

**Ranking an arrival among his new teammates by trailing usage is circular.** The
first pass did exactly that — a busy player is mechanically slotted high — so
"moved up the chart" and "was productive" were the same variable, and every
rank-based predictor silently double-counted the trailing number. A rank-ratio
predictor came back at MAE 6.93 against a baseline of 2.69, which read as a
catastrophic result for the idea when it was a catastrophic result for the
instrument. Same trap as the team-line-yards leak in "The trench report", and it
is worth stating as a rule: **if a candidate rank is built from the feature
being tested, it is not a rank.**

**nflverse's two depth-chart formats are different instruments**, and pooling them
would have shipped a baseline table understating every mover's new role by nearly
half:

Mean targets per game for a receiver the chart calls WR1 / WR2, and how many
players each format lists at those ranks:

| | WR1 | WR2 | players listed at rank 1 / 2 |
|---|---|---|---|
| 2013-24 weekly charts | 6.47 | **2.66** | 2.09 / 1.71 |
| 2025 ESPN snapshots | 7.55 | **5.42** | 1.00 / 1.00 |

The old weekly charts stop at rank 3 and pile ties on each number, so their "WR2"
is the real #2 blended with the #3 and #4. The snapshots list one player per rank
and run deeper. Production reads the snapshot format. The fix is the **ordinal
slot** — the nth player the room lists — which reconciles them, because both
formats list the same NUMBER of players per room (WR 4.34 vs 4.56, RB 2.54 vs
2.54, TE 2.69 vs 2.84). Any future experiment keyed on a historical depth-chart
rank needs this.

### What a mover actually does

With a rank on both sides of the move, and trailing usage frozen before it:

| on paper | n | old chart rank | new chart rank | trailing | actual |
|---|---|---|---|---|---|
| moved down | 117 | 1.55 | 2.94 | 5.70 | **5.38** |
| level | 113 | 2.23 | 2.23 | 6.22 | 6.46 |
| moved up | 32 | 3.50 | 2.06 | 7.06 | **9.28** |

**A player who drops a full rank does essentially his old usage.** The new, worse
rank carries almost nothing about what he will actually do; his old role does.
Moving up lifts him 31%, on 32 rows.

### The blend looked good and then dissolved

A multiplier `w + (1-w) * baseline(slot) / trailing`, replacing the discount,
fitted leave-one-season-out:

| window | n | production | blend | gain | w | cap |
|---|---|---|---|---|---|---|
| weeks 1-2 | 135 | 3.810 | 3.563 | +6.5% | 0.65 | 2.0 |
| weeks 1-3 | 256 | 3.360 | 3.055 | **+9.1%** | 0.65 | 3.0 |
| weeks 1-4 | 369 | 3.469 | 3.291 | +5.1% | 0.65 | 3.0 |
| weeks 6-12 | 434 | 2.947 | 2.994 | **−1.6%** | 0.65 | 2.0 |

The last row is the falsification and it passes: once the trailing window has
refilled with new-team games there is nothing stale left and the term is worth
nothing. So the staleness is real and the window is right.

Then the base. Everything above compares predictors built on a raw trailing
average, and the discount's defence is that it is not applied to one — it is
applied to the network's output blended 50/50 with form, and the network shrinks
hard toward the positional mean (blend.py: predicted spread 0.33 of actual at QB,
~0.70 at RB/WR/TE). Shrinkage lifts a low-usage player, which is part of what the
blend was being paid for. Rebuilding the base with that shrinkage in it, and
**refitting the weight at each level** (a weight fitted against a raw average and
then applied to a shrunk base is not a fair test — it makes the fit's failure look
like the term's):

| shrinkage λ | arrivals gain | control gain | fitted w |
|---|---|---|---|
| 1.0 (raw average) | +7.3% | +3.8% | 0.55 / 0.65 |
| 0.8 | +3.7% | +1.3% | 0.65 / 0.85 |
| 0.6 | +1.7% | −1.4% | 0.75 / 0.85 |
| 0.4 | −1.4% | −4.5% | 0.85 / 0.85 |

Monotone decay, and at the λ production actually carries it is ~3% with the
fitted weight against the grid boundary. The control is the useful half: the
blend's apparent universal win at λ=1.0 **reverses** by λ=0.6, which means the
discount is doing real work for the general population and the "the discount
hurts everyone" reading I had at λ=1.0 was an artifact of the base.

### Dropping the discount instead, which needs no parameters

At every shrinkage level, simply not discounting an arrival beat the fitted
blend. That is a far better change — one boolean, no constants, no cap, the same
shape as the promotion fix. Cluster bootstrap over seasons (clustered because the
per-season sign flipped 7 times in 13), λ = 0.70:

| group | n | gain | 90% interval |
|---|---|---|---|
| all arrivals | 369 | +4.0% | **[−8.6%, +16.0%]** |
| QB | 18 | +19.9% | [−54.6%, +55.0%] |
| RB | 110 | +3.9% | [−6.2%, +11.9%] |
| TE | 61 | −1.7% | [−24.5%, +14.1%] |
| WR | 180 | −6.5% | **[−11.9%, −1.4%]** |
| control | 20,000 | +0.4% | [−0.8%, +1.5%] |

**Nothing survives.** Every interval straddles zero except WR, where dropping the
discount is reliably worse. **Not shipped.**

### The bias is real, and it still should not be corrected

| group | n | actual | production bias | as a share | without the discount |
|---|---|---|---|---|---|
| all arrivals | 369 | 5.85 | **−1.879** | −32.1% | −0.165 |
| control | 62,853 | 8.22 | −0.729 | −8.9% | +0.020 |

A mid-season arrival is priced about **3.6x harder, proportionally,** than a
settled player — he is listed deeper than his usage warrants because he just got
there. The bias is not in doubt; correcting it just does not improve MAE, because
the downward bias is buying variance reduction on a right-skewed target. This is
the same conclusion as "Four weeks in, and the bias that should not be corrected"
and for the same structural reason: MAE is minimised by the conditional median
and fantasy scoring has its mean above its median. Anything needing a mover's
expected points should take it from the simulator, not from un-discounting him.

One piece of the current behaviour is positively validated: a mover the new chart
does not list at all gets `roles._UNKNOWN_RANK = 3`, which is a guess rather than
a reading — and on 203 such weeks it is the right guess (MAE 2.673 discounted vs
2.907 undiscounted; trailing 4.39, actual 3.62).

### The other kind of arrival pulls the other way

Everything above is a TRADE: a player with in-season form on another team. The
other population is the mid-season SIGNING — free agent or practice-squad
promotion, no game yet this season, so the last game row the path reads is from
last season and the staleness is months rather than weeks. `signings()` finds 68
such weeks where he played.

| | last season's trailing | new slot's baseline | actual |
|---|---|---|---|
| mid-season signing, n=68 | 5.58 | 4.99 | **4.68** |

| predictor | MAE | bias |
|---|---|---|
| **production (base x discount)** | **2.739** | −0.549 |
| no discount | 3.566 | +1.117 |
| slot baseline alone | 2.802 | |

**Here the discount is correct and necessary**, and dropping it costs 30% —
bootstrap −30.2% [−65.9%, +0.1%], the one interval in this entry that does not
straddle zero on the harmful side. His stale form OVER-states him, which is the
opposite sign from the traded player, and it should be: a skill player who is out
of the league until October is usually there because he declined, and the deep
slot the new chart gives him knows that. Production's bias on a signing is −0.55,
not the −1.88 a trade gets.

So the two kinds of mid-season arrival want opposite corrections, the current
code applies one rule to both, and on this evidence that rule is the right one
for each. Nothing to change.

## The promotion fix, scored the way the board is scored (2026-10-15)

The previous entry shipped `roles.promotion_multiplier` and left a debt in
writing: *"Not yet validated end to end. This is measured against actual usage,
not against the board's own FanDuel-point error through the trained model."*
Three entries in a row had ended on some version of that sentence, so it had
stopped being a caveat and become the binding constraint.

Closing it says the mechanism pays and **one of its two parameters does not**.
`PROMOTION_FACTOR["RB"]` is now 1.0.

There were two gaps in that sentence, not one.

**The currency.** `starter_out.py` scored a multiplier against a trailing usage
average. The number the multiplier actually lands on in production has been
through the network, which shrinks hard toward the positional mean, then the
form blend, then the static depth-rank discount. The mover entry measured
exactly what that does to a multiplicative term fitted on a raw average: the
gain decayed monotonically as the base took on the network's shrinkage, +7.3%
to ~3%. A term fitted against trailing usage has not been tested until it is
tested on the base it ships against.

**The trigger.** This one was not written down anywhere. The factors were fitted
with an **oracle** absence signal — snap counts, which state who played.
`predict.py` fires on the injury report plus roster status, before the game. A
correct multiplier on the wrong player is a loss, not a wash.

### Why the standing harness could not see any of this

`evaluate.backtest` contains no reference to `roles`, to a depth chart, or to
the promotion — it runs network → form blend → MAE and stops. Every volume
mechanism in `roles.py` was therefore outside the only automated scorer in the
repo, which is how a term could ship on usage evidence and stay there.

So the harness is new: `experiments/promotion_endtoend.py` reproduces the weekly
points path exactly, batching the model call because `_predict_roster_row` costs
855ms a row — over two hours for one pass of this grid on a single-seed model,
and worse with the five-seed ensemble. `verify()` then runs production's
own function on a sample of each season and asserts the batched path reproduces
it — **54 and 57 players, worst deviation 0.000**. It is not decoration; see the
rounding trap below.

It reaches 2025 and 2026 only. `DepthChartAnalyzer` raises `KeyError: 'pos_abb'`
on a 2024 depth chart, so the shipped role code cannot run on the older format
at all — a consequence of the format break in the entry above, and the reason
this is 23 weeks rather than eight seasons.

**2025 weeks 1-18 and 2026 weeks 1-5, 7,334 scored player-weeks**, 5-seed
ensemble, model trained on seasons before the target. Arms are the same chain
with one link changed, paired on identical model output:

| arm | what it is |
|---|---|
| `net_only` | network + form blend, no role layer — what `evaluate.backtest` scores |
| `discount` | + the static depth-rank multiplier (the pre-fix board) |
| `lift` | + promotion to the rank he is actually filling, **no fitted factor** |
| `as_measured` | + the shipped factors, RB x1.42 / QB x1.29 |

### The injury report is not the weak link

The promotion fires on **1,172 player-weeks, 16.0%** of those scored. The usage
entry put the entering rank-1 player's absence rate at 16.3%, over
team-position-weeks rather than player-weeks — a different denominator, so the
agreement is a sanity check and not a replication.

| trigger | fires | precision vs oracle | recall |
|---|---|---|---|
| production (injury report + roster status) | 1,172 | **1.000** | 0.690 |
| practice report alone, no Friday designation | 1,276 | 0.925 | 0.695 |

Precision 1.000 is not a rounding artifact: in 23 weeks **not one** promoted
player's superior went on to take an offensive snap. Separately, of the **2,605**
player-weeks production zeroed outright, **zero** recorded a stat line and
**zero** took an offensive snap — two independent signals, every week, no
exceptions. What the report misses is the other 31%: **526 player-weeks** a
perfect signal would have promoted and it did not, the late scratches. It never
invents one.

I expected to have to discount all of this, because production runs Tuesday and
Wednesday while `report_status` is the Friday designation, and nflverse ships one
row per player-week with no timestamp, so the mid-week view cannot be
reconstructed. The nearest strictly-weaker signal that *is* mid-week information
is "did not practise", ignoring `report_status` entirely. It fires on **more**
player-weeks than the Friday report, at 0.925 precision, and on the whole board
gives 4.0095 MAE against the Friday trigger's 4.0084. **Running early costs
nothing measurable.**

### The mechanism pays

| arm | whole board (n=7,334) | the promoted (n=1,172) | bias on the promoted |
|---|---|---|---|
| `net_only` | 4.2223 | 3.3578 | −0.76 |
| `discount` | 4.0422 | 3.4648 | −2.46 |
| `lift` | **4.0084** | **3.2535** | −1.45 |
| `as_measured` | 4.0158 | 3.2994 | −0.96 |

Cluster bootstrap over weeks, because a week shares an injury report, a slate
and a model:

| | gain | 90% interval |
|---|---|---|
| shipped vs pre-fix, whole board | +0.65% | [+0.31%, +1.01%] |
| shipped vs pre-fix, the promoted | +4.77% | [+2.34%, +7.08%] |
| parameter-free vs pre-fix, the promoted | **+6.10%** | [+4.49%, +7.74%] |
| **shipped vs parameter-free** | **−1.41%** | **[−2.76%, −0.26%]** |

The debt is discharged on the first line and the last line is the finding: the
re-ranking is worth more without the fitted factors than with them, and that
interval excludes zero.

**The placebo holds.** Count the same absences as if they sat BELOW the player —
the direction `effective_rank` ignores by design, since an RB2 being out does
not promote the RB1. Same report, same teams, same arithmetic, 3,477
player-weeks, and on that one population:

| | MAE | gain vs pre-fix |
|---|---|---|
| promote on absences above (the real term) | 2.9885 | **+1.31%** |
| promote on the same absences, mirrored below | 3.2477 | **−7.25% [−8.71%, −5.77%]** |

The gain is the re-ranking, not "teams with injuries are different".

### Which half of the fix is real, by position

| position | n | pre-fix | parameter-free | shipped | parameter-free vs pre-fix | shipped vs parameter-free |
|---|---|---|---|---|---|---|
| QB | 61 | 9.709 | 6.418 | **6.263** | +33.9% [+24.5%, +42.1%] | +2.4% [−6.0%, +10.4%] |
| RB | 309 | 3.404 | **3.381** | 3.585 | +0.7% [−2.0%, +3.2%] | **−6.1% [−10.6%, −2.1%]** |
| WR | 660 | 3.218 | **3.152** | 3.152 | +2.1% [+0.9%, +3.3%] | factor is 1.0 |
| TE | 142 | 2.060 | 2.092 | 2.092 | −1.5% [−8.4%, +4.3%] | factor is 1.0 |

The promotion is **overwhelmingly a quarterback effect** — a third of the error
on a QB2 whose starter sits — plus a small, reliable receiver gain. At RB and TE
the re-ranking alone straddles zero. And the RB factor is the one term here that
is reliably *harmful*.

### The bias that should not be corrected, a third time

| promoted RBs, n=309 | MAE | bias | mean projected | mean actual |
|---|---|---|---|---|
| pre-fix discount | 3.404 | −2.156 | 2.27 | 4.42 |
| parameter-free | **3.381** | −1.274 | 3.15 | 4.42 |
| shipped x1.42 | 3.585 | **+0.050** | 4.47 | 4.42 |

x1.42 was fitted to hit a promoted back's mean usage and it hits it almost
exactly — bias +0.05 points. That is the trap in "Four weeks in, and the bias
that should not be corrected", arriving from the opposite direction: there the
lesson was not to inflate projections to remove a downward bias, and here is a
term that was inflated to remove one, costing 5.7% MAE for the privilege.

What closes the argument is that it costs the **ordering** too, which is usually
the defence against an MAE-only verdict. Per position per week over every scored
player at the position:

| RB arm | spearman | mean weekly top-12 overlap | mean actual points of the top 12 |
|---|---|---|---|
| pre-fix discount | 0.7318 | 0.4348 | 16.155 |
| parameter-free | **0.7319** | **0.4384** | **16.245** |
| shipped x1.42 | 0.7261 | 0.4275 | 15.997 |

Every metric that matters agrees, so there is no trade to adjudicate. At QB the
factor is also no better on ordering (top-6 overlap 0.3116 against 0.3188), but
its MAE point estimate is positive and 61 player-weeks cannot resolve its size.

### Shipped

`PROMOTION_FACTOR` RB **1.42 → 1.0**. QB stays at 1.29: confirmed in sign, not
in size. WR and TE were already 1.0, so the promotion is now parameter-free
everywhere but quarterback.

| | whole board | the promoted | promoted RBs |
|---|---|---|---|
| MAE, before → after | 4.0158 → 4.0071 | 3.2994 → 3.2454 | 3.5854 → 3.3806 |
| gain | +0.21% [+0.07%, +0.35%] | +1.64% [+0.54%, +2.78%] | **+5.71% [+2.04%, +9.60%]** |
| weeks favouring it | 15 of 23 | 15 of 23 | 15 of 23 |

The three weekly counts are identical because they have to be: the change moves
only promoted RBs, so a week where it helps them helps every aggregate they sit
in.

**Replicated on the production training window.** The live model trains on the
current season alone, so the whole grid was re-run that way — 15 weeks, 4,796
scored player-weeks, a model retrained per week. Every number moves by less than
its interval: trigger precision 1.000 / recall 0.709, shipped vs pre-fix on the
promoted +5.53% [+2.97%, +8.16%], parameter-free vs pre-fix +6.49% [+5.01%,
+7.98%], the change +4.67% [+1.13%, +7.98%] on promoted RBs, placebo −14.7%.
The verdict does not depend on the window. The one line that reads weaker is
shipped vs parameter-free pooled across positions, −1.04% [−2.60%, +0.50%]
against the historical window's [−2.76%, −0.26%] — same sign, straddling zero on
849 player-weeks instead of 1,172. The RB-specific interval, which is what the
change rests on, excludes zero on both.

### The static discount, validated on the way past

The arm that exists only as a control turns out to answer an older question. The
depth-rank multiplier is a much larger term than the promotion and had never been
scored in points either:

| | n | `net_only` | `discount` | gain |
|---|---|---|---|---|
| whole board | 7,334 | 4.2223 | 4.0422 | +4.26% [+3.65%, +4.87%] |
| nobody promoted, the term acting alone | 6,162 | 4.3867 | 4.1520 | +5.35% [+4.69%, +6.01%] |
| QB | 811 | 6.7352 | 6.4332 | +4.48% [+2.49%, +6.39%] |
| RB | 1,903 | 4.3761 | 4.2197 | +3.57% [+2.08%, +5.11%] |
| WR | 3,048 | 3.9631 | 3.7629 | +5.05% [+4.16%, +5.94%] |
| TE | 1,572 | 3.2420 | 3.1353 | +3.29% [+1.81%, +4.76%] |

Positive and clear of zero everywhere. The one blemish is that at TE the
ordering is flat to slightly worse (spearman 0.5891 undiscounted against 0.5829
discounted) while the MAE gain is clear, which is the only place in this entry
the two metrics disagree.

### The instrument that lied this time: `Series.round`

`verify` failed on exactly one player — Davante Adams, production 13.90 against
the harness's 14.00 — with every multiplier in agreement. `round(13.95, 1)` is
**13.9** and `pd.Series([13.95]).round(1)` is **14.0**: numpy scales by ten and
rints to even, Python rounds the decimal correctly. In 2,000,000 uniform draws
the two never disagree, which is why this looks safe and is not: every chain here
is a 2-decimal network output times a short multiplier, so it lands on an exact
half constantly. One player in 111 was enough to find it, and had the check been
`abs(diff) < 0.11` instead of exact, it would have passed silently and every
margin in this entry would have carried a few hundredths of unexplained noise.

A second one of the same shape: re-running the whole grid after the change was
meant to prove the entry's numbers reproduce from committed code, and every row
did except the placebo, which moved from −14.4% to −7.3%. Its multiplier was
still reading `roles.PROMOTION_FACTOR`, so lowering the constant quietly made the
placebo weaker. Both arms under comparison had been pinned against exactly that;
the control had not. It is now parameter-free like the arm it mirrors, and the
corrected number is the one in the table above.

Three entries running, the harness has been the thing that was wrong first — the
circular rank, the chart-format break, now the rounding and the constant leak.
Each was caught by checking the instrument against something independent rather
than by the result looking implausible. The reproducibility re-run is cheap once
the trained models are cached and has now earned its place in the routine.

## Charted line play: yards before contact, and the pressure field that exists (2026-10-15)

`trenches.py` grades four units off play-by-play and explains two of its choices
as workarounds: adjusted line yards uses Football Outsiders' banding "because
the first four yards of a carry belong to the line and the eleventh does not",
and pressure is sack-or-hit per dropback "because nflverse publishes no pressure
field". The first is a proxy for a measurement. The second is simply **not
true** — it is published, through Pro Football Reference's charting, weekly back
to 2018:

    rushing_yards_before_contact / _after_contact / rushing_broken_tackles
    times_pressured, times_hurried, times_blitzed, times_hit, times_sacked
    def_pressures, def_sacks, def_missed_tackles

`nflreadpy` 0.1.3 has no `load_pfr_advstats` (the R package does), so
`experiments/trench_charted.py` reads the nflverse-data release assets directly,
one parquet per season per stat type. 4,190 team-weeks, 2018-25.

Split-half within season over 2022-25, the same protocol and window as "The
trench report" above — but the shipped metrics are **recomputed through this
harness** rather than quoted from that entry, because its split-half ran over
raw plays and this one runs over team-week rates, and comparing a new metric
measured one way against an old one measured another way is not a comparison.

| metric | offence r | defence r | meant to measure |
|---|---|---|---|
| charted yards before contact / carry | **0.367** | 0.356 | the line |
| charted yards after contact / carry | 0.065 | 0.049 | the back |
| charted broken tackles / carry | 0.210 | 0.088 | the back |
| *shipped:* adjusted line yards | 0.339 | **0.418** | the line |
| *shipped:* stuff rate | 0.304 | 0.431 | the line |
| charted pressure rate | 0.377 | 0.160 | protection / rush |
| charted hurry rate | 0.125 | **−0.054** | protection / rush |
| charted blitz rate faced | 0.286 | **0.659** | scheme, not the line |
| *shipped:* pressure proxy (sack or hit) | **0.566** | **0.336** | protection / rush |
| *shipped:* sack rate | 0.510 | 0.040 | protection / rush |

Populations are 94% shared on the rushing side — PFR is per-rusher so QBs are
dropped entirely, while `trenches.plays` drops scrambles and keeps designed QB
runs — and 100.5% on dropbacks.

### Yards before contact is the right idea and a lateral move

As an offence trait it beats the banding, 0.367 against 0.339. As a defence trait
it loses, 0.356 against 0.418, and the rating blends the banding with stuff rate
precisely because stuff rate is the half that survives opponent adjustment on
defence. Year over year the same: 0.287/0.242 against 0.308/0.367. A new data
dependency for ±0.03 of split-half correlation in opposite directions by side is
not worth taking, so **nothing changes.**

What it does buy is the explanation the banding never had. **Yards after contact
is not a team trait at all** — 0.065 on offence, 0.049 on defence, which is zero
either way — and broken tackles barely more. So the carry really does divide into
a stable part and a noise part at roughly the halfway mark (before-contact is
0.50-0.60 of yardage, league-wide, by season), and FO's decision to count the
first four yards fully and nothing past ten was a good guess at where that line
falls. The proxy works because the thing it discards is unpredictable.

It also closes off a plausible-sounding defensive metric: **yards after contact
allowed cannot grade a front** (0.049). Tackling is not a measurable unit trait
at team-week scale, so "our line gets pushed around after contact" is not a
thing the data can say.

### The pressure field exists and is worse than the proxy

This is the one I expected to go the other way. Charted pressure is 1.6x the
volume of sack-or-hit and correlates with it at only 0.717, so it is genuinely
different information — and it is **materially less stable**: 0.377 against 0.566
on offence, and on defence 0.160 against 0.336, less than half. Charted hurry
rate on defence is **−0.054**, anti-correlated with itself across halves of the
same season, which is what a field reads like when the charting varies more than
the thing being charted.

A sack and a QB hit are scorebook events. A "pressure" is a judgment, and the
judgment moves between games. So the free proxy beats the hand-charted
measurement here, which is the reverse of the assumption the trench work started
from — that the subscription product has something nflverse cannot rebuild.

`trenches.py`'s docstring now says this instead of claiming the field does not
exist. The reason for the choice is stronger than the one it replaces: measured,
not unavailable.

### The one thing here with no counterpart in the ratings

**Blitz rate faced is a defensive trait at r = 0.659**, the most stable defensive
number anywhere in the trench work and second only to FTN role (0.748) in this
file. Nothing in `trenches.py` uses it, and defensive pass rush is the unit with
the weakest claim to being a trait at all (0.040 by sacks, 0.160 by charted
pressure, 0.336 by the shipped proxy).

That is a candidate and not a finding. "Blocking scheme is the most stable trait
in this file and predicts nothing" is two entries back, and blitz rate is the
same shape of thing: a scheme identity, stable because coordinators are
consistent, with no demonstrated link to anything a projection needs. It should
be tested against quarterback output and sack props before anyone believes it.
