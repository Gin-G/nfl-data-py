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
Projector hack.** Wins on RB/WR/TE every year; QB is noisy (n=4-6, starting job is
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
Naive last-5-avg is 4.255; our best (NN-ensemble + GBDT blend) is 4.197 — a real
but small edge. Direct-FP beats component-first extrapolation at every position.
Point projection (rolling features, #2) looks near the
practical ceiling for this data/architecture: opponent/scheme (#4/#5),
volatility (#7), and component-scoring (#10) all failed to beat it. Quantile
floor/median/ceiling works and is wired in; the median matches the mean model,
the ceiling is trustworthy, the floor is the weak spot. Remaining honest levers:
per-position models, richer data (snaps/routes/betting lines), or accept current
accuracy and focus on usage (DFS optimizer, weekly workflow).
