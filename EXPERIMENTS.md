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
