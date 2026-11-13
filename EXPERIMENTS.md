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

## Index

Auto-generated from the `##` headings; regenerate with
`python scripts/index_experiments.py` after adding an entry. **closed**
marks an avenue measured and abandoned — read it before proposing the
idea again.

- [Findings](#findings)
- [Calibration finding (#8)](#calibration-finding-8)
- [Component-level finding (#10)](#component-level-finding-10)
- [Wider band finding (#11)](#wider-band-finding-11)
- [FPA / matchup — tested three ways, all negative (#4, #5, #12)](#fpa-matchup-tested-three-ways-all-negative-4-5-12) — **closed**
- [Lineup backtest (#13)](#lineup-backtest-13)
- [Roadmap item 1 — naive baselines, GBDT, multi-seed (2025 wk1-18)](#roadmap-item-1-naive-baselines-gbdt-multi-seed-2025-wk1-18) — **shipped**
- [Roadmap follow-up — component-first models (the fair test)](#roadmap-follow-up-component-first-models-the-fair-test)
- [Coaching / new-HC regime change (probe, 2025) — bias not MAE](#coaching-new-hc-regime-change-probe-2025-bias-not-mae)
- [Coaching — do prior-job tendencies carry over? (foundation test)](#coaching-do-prior-job-tendencies-carry-over-foundation-test)
- [Coaching — wk1 scheme prior applied (nfl_projections/coaching.py) — near-neutral](#coaching-wk1-scheme-prior-applied-nfl_projectionscoachingpy-near-neutral) — **shipped**
- [Rookie draft-capital prior — foundation is STRONG (unlike coaching)](#rookie-draft-capital-prior-foundation-is-strong-unlike-coaching)
- [Rookie draft-capital prior — BUILT & WINS (nfl_projections/rookies.py)](#rookie-draft-capital-prior-built-wins-nfl_projectionsrookiespy)
- [Why opponent/matchup keeps failing — defense-vs-position is weak & unstable](#why-opponentmatchup-keeps-failing-defense-vs-position-is-weak-unstable)
- [Player grades — opportunity-weighted, validated (nfl_projections/player_grades.py)](#player-grades-opportunity-weighted-validated-nfl_projectionsplayer_gradespy)
- [Vegas implied team total — the definitive matchup test — WASH (not kept)](#vegas-implied-team-total-the-definitive-matchup-test-wash-not-kept) — **closed**
- [Per-position models — shared model wins (last mean lever, closed)](#per-position-models-shared-model-wins-last-mean-lever-closed)
- [Usage-based Monte-Carlo simulator (nfl_projections/simulate.py) — distributions + stacks](#usage-based-monte-carlo-simulator-nfl_projectionssimulatepy-distributions-stacks)
- [#20-21 The preseason board finally has a benchmark — and it exposed a bad one (2026-08-11)](#20-21-the-preseason-board-finally-has-a-benchmark-and-it-exposed-a-bad-one-2026-08-11)
- [#19 Snap share as the role multiplier — rejected, and the multiplier itself is suspect (2026-08-11)](#19-snap-share-as-the-role-multiplier-rejected-and-the-multiplier-itself-is-suspect-2026-08-11) — **shipped**
- [#18 Thin-history gating — a negative result worth writing down (2026-08-11)](#18-thin-history-gating-a-negative-result-worth-writing-down-2026-08-11) — **closed**
- [#17 Component stat lines shrink to the positional mean too (2026-08-10)](#17-component-stat-lines-shrink-to-the-positional-mean-too-2026-08-10) — **shipped**
- [#14-16 Board ordering — the cap was normalizing teams, and MAE liked it (2026-08-10)](#14-16-board-ordering-the-cap-was-normalizing-teams-and-mae-liked-it-2026-08-10) — **shipped**
- [Quantile bands — the "floor problem" was never fixable (2026-08-09)](#quantile-bands-the-floor-problem-was-never-fixable-2026-08-09) — **closed**
- [QB separation — the board was flat for two independent reasons (2026-08-07)](#qb-separation-the-board-was-flat-for-two-independent-reasons-2026-08-07) — **shipped**
- [Standardized training targets — fixes the broken component stat lines (2026-08-06)](#standardized-training-targets-fixes-the-broken-component-stat-lines-2026-08-06)
- [Opportunity share vs trailing rate — week 1 2025 (2026-09-09)](#opportunity-share-vs-trailing-rate-week-1-2025-2026-09-09)
- [Coverage matchup — a null result, and one narrow win (2026-09-09)](#coverage-matchup-a-null-result-and-one-narrow-win-2026-09-09) — **closed**
- [Opponent-adjusted defence beats raw allowed-per-game (2026-09-09)](#opponent-adjusted-defence-beats-raw-allowed-per-game-2026-09-09) — **shipped**
- [Granular defensive matchup, and the trait that is stable but inert (2026-09-11)](#granular-defensive-matchup-and-the-trait-that-is-stable-but-inert-2026-09-11) — **closed**
- [Usage decomposition beats trailing rate — for tight ends and backs (2026-09-11)](#usage-decomposition-beats-trailing-rate-for-tight-ends-and-backs-2026-09-11)
- [Next Gen Stats add nothing — and the near-miss is the lesson (2026-09-11)](#next-gen-stats-add-nothing-and-the-near-miss-is-the-lesson-2026-09-11) — **closed**
- [FTN charting: the most stable traits measured here, and they add nothing (2026-09-11)](#ftn-charting-the-most-stable-traits-measured-here-and-they-add-nothing-2026-09-11)
- [The gate: usage decomposition vs the PRODUCTION model (2026-09-11)](#the-gate-usage-decomposition-vs-the-production-model-2026-09-11) — **closed**
- [Live week 1, 2026: the first prospective scorecard (2026-09-14)](#live-week-1-2026-the-first-prospective-scorecard-2026-09-14)
- [The in-season path, run on 2026 for the first time (2026-09-14)](#the-in-season-path-run-on-2026-for-the-first-time-2026-09-14)
- [Weeks 1 and 2, scored prospectively (2026-09-25)](#weeks-1-and-2-scored-prospectively-2026-09-25) — **shipped**
- [Four weeks in, and the bias that should not be corrected (2026-10-05)](#four-weeks-in-and-the-bias-that-should-not-be-corrected-2026-10-05) — **shipped**
- [Training on the current season alone (2026-10-05)](#training-on-the-current-season-alone-2026-10-05)
- [How far back should training reach (2026-10-05)](#how-far-back-should-training-reach-2026-10-05) — **shipped**
- [History depth, retrained every week (2026-10-07)](#history-depth-retrained-every-week-2026-10-07)
- [The trench report: line play, rebuilt from public data (2026-10-14)](#the-trench-report-line-play-rebuilt-from-public-data-2026-10-14)
- [Line ratings shipped, and which half of a matchup is real (2026-10-14)](#line-ratings-shipped-and-which-half-of-a-matchup-is-real-2026-10-14) — **shipped**
- [Positional scarcity: replacement level, and a flat bar that cannot work (2026-10-14)](#positional-scarcity-replacement-level-and-a-flat-bar-that-cannot-work-2026-10-14)
- [Blocking scheme: the most stable trait in this file, and it predicts nothing (2026-10-14)](#blocking-scheme-the-most-stable-trait-in-this-file-and-it-predicts-nothing-2026-10-14)
- [Vacated share: the assumption held, and the gap was somewhere else (2026-10-14)](#vacated-share-the-assumption-held-and-the-gap-was-somewhere-else-2026-10-14)
- [The starter is out, and the backup was being charged a backup's rate (2026-10-14)](#the-starter-is-out-and-the-backup-was-being-charged-a-backups-rate-2026-10-14)
- [The mid-season mover, and two instruments that lie about him (2026-10-14)](#the-mid-season-mover-and-two-instruments-that-lie-about-him-2026-10-14) — **shipped**
- [The promotion fix, scored the way the board is scored (2026-10-15)](#the-promotion-fix-scored-the-way-the-board-is-scored-2026-10-15) — **shipped**
- [Charted line play: yards before contact, and the pressure field that exists (2026-10-15)](#charted-line-play-yards-before-contact-and-the-pressure-field-that-exists-2026-10-15) — **shipped**
- [Blitz rate against quarterback output — and the two positions it actually moves (2026-10-15)](#blitz-rate-against-quarterback-output-and-the-two-positions-it-actually-moves-2026-10-15)
- [Per-corner coverage grades: the yards are not a trait (2026-10-16)](#per-corner-coverage-grades-the-yards-are-not-a-trait-2026-10-16) — **closed**
- [Opportunity share: the premise is right, the board already has it (2026-10-16)](#opportunity-share-the-premise-is-right-the-board-already-has-it-2026-10-16)
- [Scoping the defense slot: it is matchup plus noise (2026-10-16)](#scoping-the-defense-slot-it-is-matchup-plus-noise-2026-10-16)
- [The first scored lineups: the edge is real and it is in the top five (2026-10-20)](#the-first-scored-lineups-the-edge-is-real-and-it-is-in-the-top-five-2026-10-20)
- [Week 7 lineups: the hits and the misses differ by one player, not by one shape (2026-10-26)](#week-7-lineups-the-hits-and-the-misses-differ-by-one-player-not-by-one-shape-2026-10-26)
- [Why week 7's later builds were worse: it was a different board, not a fresher one (2026-10-26)](#why-week-7s-later-builds-were-worse-it-was-a-different-board-not-a-fresher-one-2026-10-26)
- [Defenses: FanDuel's FPPG is worse than a flat average, and the pieces beat both (2026-10-26)](#defenses-fanduels-fppg-is-worse-than-a-flat-average-and-the-pieces-beat-both-2026-10-26) — **closed**
- [The optimizer learns that nine players can be one bet (2026-10-26)](#the-optimizer-learns-that-nine-players-can-be-one-bet-2026-10-26)
- [The field, and the players it has already written off (2026-10-28)](#the-field-and-the-players-it-has-already-written-off-2026-10-28)
- [A lineup's shape is not a free parameter (2026-10-28)](#a-lineups-shape-is-not-a-free-parameter-2026-10-28)
- [The starter comes back, and the fill-in keeps the job he lost (2026-10-28)](#the-starter-comes-back-and-the-fill-in-keeps-the-job-he-lost-2026-10-28) — **closed**
- [Week 8 scored: the field was right, and so was the list we doubted (2026-11-02)](#week-8-scored-the-field-was-right-and-so-was-the-list-we-doubted-2026-11-02)
- [The 147 the board agreed with and the optimizer would not buy (2026-11-02)](#the-147-the-board-agreed-with-and-the-optimizer-would-not-buy-2026-11-02)
- [Correction: the ownership page is a vintage, and I scored the flag against the wrong one (2026-11-02)](#correction-the-ownership-page-is-a-vintage-and-i-scored-the-flag-against-the-wrong-one-2026-11-02)
- [Cash and tournament builds, and the dial that cannot be set from two weeks (2026-11-02)](#cash-and-tournament-builds-and-the-dial-that-cannot-be-set-from-two-weeks-2026-11-02) — **closed**
- [Coverage is answered; the trench effect gets a prospective record (2026-11-02)](#coverage-is-answered-the-trench-effect-gets-a-prospective-record-2026-11-02)
- [A good matchup is a tiebreaker, not a reason to climb the board (2026-11-02)](#a-good-matchup-is-a-tiebreaker-not-a-reason-to-climb-the-board-2026-11-02)
- [Asking the matchup question where the builder decides, and the split that eats it (2026-11-02)](#asking-the-matchup-question-where-the-builder-decides-and-the-split-that-eats-it-2026-11-02)
- [Correction: it was not the split, and dollars per opportunity does not beat the board (2026-11-02)](#correction-it-was-not-the-split-and-dollars-per-opportunity-does-not-beat-the-board-2026-11-02) — **closed**
- [The second-half fade for older backs is real, mostly priced, and Henry is the case (2026-11-02)](#the-second-half-fade-for-older-backs-is-real-mostly-priced-and-henry-is-the-case-2026-11-02)
- [Boom/bust is real, it is volume, and it was mis-simulated](#boombust-is-real-it-is-volume-and-it-was-mis-simulated)
- [Wind is the one unpriced situational factor worth buying data for](#wind-is-the-one-unpriced-situational-factor-worth-buying-data-for)
- [A vetting layer, explicitly outside the objective](#a-vetting-layer-explicitly-outside-the-objective) — **closed**
- [Backtest: one cash lineup a week against the 125 target](#backtest-one-cash-lineup-a-week-against-the-125-target) — **shipped**
- [Fantasy points allowed, adjusted for who the defence has faced](#fantasy-points-allowed-adjusted-for-who-the-defence-has-faced) — **closed**
- [Three reports of the same defence: do trenches, coverage and FPA agree?](#three-reports-of-the-same-defence-do-trenches-coverage-and-fpa-agree)
- [Which lineup objective actually won: the record said floor, the record was wrong](#which-lineup-objective-actually-won-the-record-said-floor-the-record-was-wrong) — **closed**
- [Did a day-stale roster table corrupt the week 9 board? No](#did-a-day-stale-roster-table-corrupt-the-week-9-board-no)
- [The Tuesday projection run is structurally blind to injuries](#the-tuesday-projection-run-is-structurally-blind-to-injuries)
- [FantasyGuru snapshots: storage, a shell mix, and a gate (2026-11-04)](#fantasyguru-snapshots-storage-a-shell-mix-and-a-gate-2026-11-04) — **closed**
- [Reading a published DFS column against our own board (2026-11-06)](#reading-a-published-dfs-column-against-our-own-board-2026-11-06)
- [Week 9 scored: why 45 lineups all landed in the same place (2026-11-09)](#week-9-scored-why-45-lineups-all-landed-in-the-same-place-2026-11-09)
- [Negative result: a player's own high and low make a worse band than the model's (2026-11-09)](#negative-result-a-players-own-high-and-low-make-a-worse-band-than-the-models-2026-11-09) — **closed**
- [Consistency: scoring steadiness is not a property of a player (2026-11-09)](#consistency-scoring-steadiness-is-not-a-property-of-a-player-2026-11-09) — **closed**
- [A read of all three repos: three production readings, and the record learns about depth (2026-11-10)](#a-read-of-all-three-repos-three-production-readings-and-the-record-learns-about-depth-2026-11-10)
- [A usage trend is half real: what the game after an "up" or "down" looks like (2026-11-11)](#a-usage-trend-is-half-real-what-the-game-after-an-up-or-down-looks-like-2026-11-11)
- [Lineup depth: the top of a cash build is worth about ten points, and shrinking toward form is not (2026-11-12)](#lineup-depth-the-top-of-a-cash-build-is-worth-about-ten-points-and-shrinking-toward-form-is-not-2026-11-12)
- [FanDuel scoring corrected: lost fumbles and two-point conversions (2026-11-12)](#fanduel-scoring-corrected-lost-fumbles-and-two-point-conversions-2026-11-12)
- [Player identity: names were doing an id's job (2026-11-13)](#player-identity-names-were-doing-an-ids-job-2026-11-13) — **shipped**
- [Reruns on clean data: what the identity bugs did to earlier results (2026-11-13)](#reruns-on-clean-data-what-the-identity-bugs-did-to-earlier-results-2026-11-13)

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

*Tested the same day against quarterback output — see "Blitz rate against
quarterback output" below. It is inert for quarterbacks even with the rate read
out of the game itself, real for backs and receivers, and worth zero on the
board. Sack props were not used and will not be: a line cannot be a regressor
in a test for edges against lines.*

## Blitz rate against quarterback output — and the two positions it actually moves (2026-10-15)

The entry above left blitz rate faced as the one charted number with no counterpart
in the ratings: a defensive trait at r = 0.659 split-half, the most stable defensive
measurement anywhere in the trench work, and unused. It also left the reason not to
believe it yet — "blocking scheme is the most stable trait in this file and predicts
nothing" is three entries back, and a blitz rate is the same shape of claim. So:
does a defense's blitz rate move the quarterback it is pointed at?

**No betting lines anywhere in this test**, on instruction and on the merits. A sack
prop or an implied total already contains whatever the market knows about the
matchup, so conditioning on one measures what survives after the edge has been taken
out, which is the opposite of the question. A line is a benchmark to score against
later, never a regressor here.

`experiments/blitz_qb.py`. 3,889 quarterback starts (10+ attempts) 2018-25, 89
passers, matched to 98.6% of charted defense-weeks. 2026 is absent for a structural
reason: every outcome is scored against the passer's own season mean, and that needs
more games than the season has played.

### Three instruments, because one would not be believable

| instrument | what it is | r against the rate in the game |
|---|---|--:|
| in-season | the defense's rate over its own prior weeks, read at 100+ dropbacks | 0.450 |
| last-season | the same defense over the whole previous season | 0.283 |
| oracle | the rate in the game itself — not a forecast, a ceiling | 1.000 |

A blitz rate spans a lot: the bottom decile of defense-weeks blitzes 15.4% of
dropbacks, the top decile 39.4%, and one standard deviation is 7.3 points of rate.
Year over year the trait holds at r = 0.499 on 222 team-seasons. The instrument was
never the problem.

### The quarterback: nothing, by every instrument and at every control

Slope in FanDuel points per +1 sd of blitz rate, each week scored against the same
passer's own season mean with that week left out, so quarterback quality cancels
exactly. Intervals are a cluster bootstrap over (season, week); the p-value is a
permutation that reshuffles which defense each passer faced *inside its own week*,
holding the slate, the weather and the injury report fixed rather than modelling them.

| fit | n | slope | 95% CI | perm p |
|---|--:|--:|---|--:|
| raw, in-season | 3,130 | −0.083 | [−0.369, +0.211] | 0.604 |
| own-mean, in-season | 3,130 | −0.097 | [−0.366, +0.177] | 0.522 |
| own-mean, last-season | 3,431 | +0.191 | [−0.067, +0.438] | 0.172 |
| **own-mean, oracle** | 3,889 | **+0.043** | [−0.194, +0.295] | 0.780 |
| own-mean, in-season + points allowed | 3,130 | −0.117 | [−0.387, +0.151] | — |
| own-mean, in-season + 3 model columns | 3,130 | −0.114 | [−0.408, +0.180] | — |
| own-mean, oracle + 3 model columns | 3,130 | −0.037 | [−0.312, +0.257] | — |

The oracle row is the one that closes it. Knowing *exactly* how often a defense will
blitz — which no Thursday knows — is worth 0.04 ± 0.25 FanDuel points. There is no
forecast of a quantity that can beat perfect knowledge of it, so the two forecastable
instruments were never going to find anything, and a feature built on either cannot.

Nor is it hiding in the shape of the distribution. By tercile of the in-season rate
the residual means are −0.145 / −0.164 / +0.004, the residual spreads 8.11 / 8.12 /
8.29, the share of 20-point weeks 0.353 / 0.354 / 0.367 and the bust rate below ten
0.221 / 0.206 / 0.209. By decile the top-minus-bottom residual gap is −0.65 points
across a 15.4% → 39.4% span in blitz rate, inside the week-to-week noise. "A blitz
creates variance even if it does not move the mean" is a reasonable guess and it is
also false here.

### What *does* move, and why it cancels

The stat line is not inert — it moves in offsetting directions, per +1 sd of the
in-season rate:

| outcome | mean | slope | 95% CI |
|---|--:|--:|---|
| attempts | 32.7 | **+0.497** | [+0.175, +0.803] |
| sacks suffered | 2.28 | **+0.095** | [+0.036, +0.145] |
| sack rate | 0.066 | **+0.002** | [+0.000, +0.003] |
| passing yards | 236.2 | +2.448 | [−0.252, +5.096] |
| EPA per play | 0.058 | **−0.015** | [−0.027, −0.002] |
| carries | 3.73 | **−0.097** | [−0.167, −0.026] |
| air yards per attempt | 7.90 | −0.021 | [−0.099, +0.056] |

A blitz-heavy defense gives a quarterback half an extra throw and takes a little off
every one of them. **FanDuel does not charge for a sack** — sack yardage is not
subtracted from passing yards — so the volume gain and the efficiency loss meet in
the middle and the fantasy total does not know anything happened. The effect is real
in football and absent in the scoring system, which is a distinction this ledger has
not had to make before.

The folk version fails too. Mobile quarterbacks (top tercile of own carries per game,
5.0) are supposed to punish a blitz; their slope is **−0.543 [−1.085, +0.004]**,
pointing the wrong way, against +0.050 for pocket passers. Three bands, one that
grazes zero: not a finding, but nobody should repeat the opposite as though it were.

### The oracle is not clean either, which is why it is only a ceiling

Reading the rate out of the game being predicted buys a correlation with the game.
A defense that is behind blitzes more (r = −0.124 against its own mean scoreboard),
so the oracle's components are contaminated by score: oracle → interceptions is
**−0.087 per sd [−0.113, −0.063]**, and nobody believes a blitz *prevents* picks.
That is the feedback loop being read backwards as an effect. Holding the scoreboard
moves the quarterback's FanDuel slope from +0.043 to −0.144 [−0.384, +0.094] —
still nothing, from the other side of zero.

### The two positions that were not the question

The frame was already built, so the same test ran for the skill positions, on weeks
where the player took 40% of his offense's snaps. Slope in FanDuel points per +1 sd,
own-mean scored, same cluster bootstrap and same within-week permutation:

| pos | n | in-season | 95% CI | perm p | + points allowed | + 3 model cols |
|---|--:|--:|---|--:|--:|--:|
| **RB** | 3,573 | **−0.541** | [−0.807, −0.271] | **0.000** | −0.444 | −0.414 |
| **WR** | 10,246 | **+0.219** | [+0.064, +0.392] | **0.003** | +0.209 | +0.173 |
| TE | 3,713 | −0.057 | [−0.221, +0.121] | 0.515 | −0.050 | −0.043 |
| QB | 3,130 | −0.097 | [−0.366, +0.177] | 0.522 | −0.117 | −0.114 |

Backs lose and receivers gain, and both survive being fit beside the points that
defense has already conceded to that very position — the production model's own
coarse opponent column, rebuilt here on a longer memory than the real one, so if
anything a harder control than the model would apply. The components say the same
thing: RB carries −0.320 [−0.494, −0.136] and rushing yards −1.960 [−3.157, −0.817];
WR targets +0.193 [+0.120, +0.265] and receiving yards +1.057 [+0.296, +1.842].
Blitz-heavy fronts put bodies near the line and leave fewer behind it. **The
check-down story is backwards** — the ball goes outside, not to the back, and the
back's running room is what pays for it.

The weaker instruments do not contradict this; they are attenuated by a known
amount. A regression on a noisy regressor shrinks by the ratio of the instruments'
correlations with the thing itself, 0.283 / 0.450 = 0.630:

| pos | in-season | last-season predicted | last-season observed | 95% CI | consistent |
|---|--:|--:|--:|---|---|
| RB | −0.541 | −0.341 | −0.198 | [−0.516, +0.129] | yes |
| WR | +0.219 | +0.138 | +0.072 | [−0.086, +0.231] | yes |
| TE | −0.057 | −0.036 | +0.054 | [−0.107, +0.227] | yes |

And the oracle agrees once the scoreboard is held: WR +0.193 [+0.035, +0.330] on its
own strength (perm p 0.000), RB from +0.078 to **−0.176** [−0.440, +0.087]. Three
instruments, one direction per position. "Two instruments that lie" is two entries
back; this time checking whether the weak one's interval contains the *attenuated*
prediction, rather than whether it clears zero, is what kept them from looking like
a contradiction.

### Real, and worth almost nothing

Across the whole league span — bottom decile to top, 15.4% to 39.4%, about 3.3 sd —
the effect is **−1.8 FanDuel points for a back** on a 13.6 mean and **+0.7 for a
receiver** on a 9.0 mean. That sounds usable right up to the moment it is held out
of sample. Leave one season out, baseline carrying the points-allowed column:

| pos | n | base MAE | with blitz | gain | seasons won |
|---|--:|--:|--:|--:|--:|
| RB | 3,573 | 6.2791 | 6.2708 | **+0.0083** | 6/8 |
| WR | 10,246 | 5.1191 | 5.1186 | +0.0005 | 5/8 |
| TE | 3,713 | 3.9838 | 3.9832 | +0.0006 | 5/8 |
| QB | 3,130 | 6.4713 | 6.4746 | −0.0033 | 4/8 |

### Against the board itself, which is the only test that settles it

A slope against a player's own season mean is not a slope against what production
serves. That gap is what the promotion entry had to come back for a day ago, so this
one does not leave it open. `evaluate.backtest` over 2021-25, one seed, `opponent=False`
exactly as the board is served — the model carries **no opponent column at all**, the
three matchup entries above being why — so a blitz rate here is not competing against a
weaker version of itself. 31,574 scored player-weeks, board MAE 4.1791.

**The board's error does know about blitz rate.** Regressing the residual on the rate
the model never saw:

| pos | n | board bias | slope | 95% CI | perm p |
|---|--:|--:|--:|---|--:|
| QB | 2,372 | +0.483 | +0.103 | [−0.217, +0.434] | 0.550 |
| **RB** | 6,776 | +0.160 | **−0.284** | [−0.458, −0.128] | **0.000** |
| **WR** | 12,184 | +0.078 | **+0.136** | [+0.022, +0.256] | **0.013** |
| TE | 4,435 | −0.068 | −0.057 | [−0.206, +0.085] | 0.370 |

So the information is real, it is forecastable, and it is genuinely missing from
production. Everything up to here says build the feature. Then it gets scored.

One additive term per position, fitted leave-one-season-out, applied two ways —
the whole fitted line, and the slope with the intercept dropped, because a
per-position bias shift is **#31, the bias that should not be corrected**, and would
otherwise be charged to the blitz term:

| pos | board MAE | bias + slope | slope only | slope-only gain | 95% paired CI | weeks won |
|---|--:|--:|--:|--:|---|--:|
| QB | 6.4280 | 6.4606 | 6.4310 | −0.0031 | [−0.0086, +0.0025] | 37/79 |
| RB | 4.2513 | 4.3096 | 4.2529 | −0.0016 | [−0.0100, +0.0065] | 39/79 |
| WR | 3.9774 | 3.9998 | 3.9777 | −0.0003 | [−0.0036, +0.0027] | 44/79 |
| TE | 3.1671 | 3.1854 | 3.1685 | −0.0014 | [−0.0033, +0.0006] | 35/79 |

The slope is worth **nothing**: every interval straddles zero and the point estimates
are a thousandth of a FanDuel point. Splitting it from the intercept mattered —
`bias + slope` lost 1.37% of RB MAE, and essentially all of that was the bias term,
which is #31 reproducing itself in a second, independent place. That result now has two
sightings and should be treated as a property of the target rather than a quirk of the
week it was found in.

Ordering is worse, not merely flat. A per-position shift cannot reorder a
within-position board, so this arm is the slope alone:

| pos | top-N overlap | board | corrected | delta | 95% paired CI | weeks improved |
|---|--:|--:|--:|--:|---|--:|
| QB | 12 | 0.5631 | 0.5586 | **−0.0045** | [−0.0090, −0.0011] | 0/74 |
| RB | 12 | 0.4278 | 0.4211 | −0.0067 | [−0.0178, +0.0033] | 9/75 |
| WR | 24 | 0.4222 | 0.4178 | **−0.0044** | [−0.0083, −0.0006] | 3/75 |
| TE | 12 | 0.4833 | 0.4856 | +0.0022 | [−0.0022, +0.0067] | 3/75 |

The correction improved **nought of 74** quarterback weeks and three of 75 receiver
weeks, and the QB and WR intervals are clear of zero on the wrong side. A term fitted
to the mean residual, applied to a board that is read as an order, moves players the
wrong way more often than the right way.

### Verdict

**Nothing ships.** The question as asked is answered flatly: a defense's blitz rate
does not move quarterback FanDuel output, and the oracle version of the test means no
better forecast of it could. The two real effects it did turn up — backs down,
receivers up, roughly −1.8 and +0.7 points across the league's whole blitz-rate range
— are present in the board's own error and still worth zero MAE and slightly negative
ordering when they are put in.

This is the fourth stable trait in the ledger that predicts nothing a board can use
— success rate allowed (0.618), FTN role (0.748), blocking scheme, and now blitz
rate (0.659) — and the pattern is sharp enough to state as a rule: **stability is a property of the
measurement and says nothing about whether the quantity is in the scoring system.**
The blitz effect is the cleanest illustration yet — it moves attempts and sacks in
opposite directions and FanDuel charges for one of them and not the other, so a real
football result arrives at the board as exactly zero.

What would change the answer: a scoring system that charges for sacks, a target that
is passing yardage rather than fantasy points, or a market price for a back's rushing
yardage against a specific front. The first two are not this board and the third is a
line, which this test does not get to use.

## Per-corner coverage grades: the yards are not a trait (2026-10-16)

"Coverage matchup — a null result, and one narrow win" tested this at the team
level and found wide receiver **actively worse** (MAE 26.73 → 26.94, correlation
0.472 → 0.457). The diagnosis written there was specific, and half of it turns
out to be wrong:

> "yards allowed to WR" is spread across a defence's whole secondary and a
> team's whole receiving corps... The thing that would — which corner travels
> with him and whether he is any good — is a shadow-coverage assignment, and
> nflverse has no such field.

**Whether he is any good is published.** Pro Football Reference charts coverage
*per defender*, weekly back to 2018, through the same nflverse release assets
the trench work reads: targets, completions, yards, touchdowns and passer rating
allowed, plus how deep he was tested. 16,832 corner-weeks, 646 corners. Only the
assignment is genuinely missing. Harness: `experiments/coverage_grades.py`.

### A corner sees 26 targets a season, and it decides everything

Median targets per corner-season is **26** (25th percentile 8, 75th 60). That
number governs the whole file. Within-season split-half, 2022-25, is hopeless
for every outcome metric — yards allowed per target comes in at **0.037**. The
question is whether that is noise or absence, so the threshold was raised and
then the sample was maximised by correlating each corner against himself across
**odd versus even weeks of his entire career** (125 corners with 100+ targets a
side, which no single game can straddle):

| metric | within-season | career odd/even | what it is |
|---|--:|--:|---|
| targets per game | 0.293 | **0.728** | role: is he thrown at |
| average depth of target | 0.403 | **0.733** | role: how deep is he tested |
| completion % allowed | 0.220 | **0.519** | outcome, and it survives |
| yards after catch per completion | 0.232 | 0.341 | outcome |
| passer rating allowed | 0.119 | 0.204 | outcome |
| **yards allowed per target** | **0.037** | **0.188** | outcome |
| TDs allowed per target | 0.082 | 0.086 | outcome |

Raising the within-season minimum does not rescue the yardage either — 0.037 at
20 targets, 0.028 at 30, 0.054 at 40 — and at the tighter career threshold (200+
targets a side) yards per target goes **negative, −0.208**, as do YAC and TDs.
That is the signature of nothing being there, not of something being hard to see.

**So the number a corner ranking most wants to use is the one that does not
persist.** The mechanism is arithmetic: yards per target is completion rate times
yards per completion, and the second factor rides on a handful of long plays. One
70-yard touchdown on a 60-target season moves a corner's yards per target by 1.2.
Completion rate is a per-event binary on the identical sample, which is why it
survives at 0.519 where the yardage built on top of it does not.

The two strongest numbers in the table are **role, not quality** — how often and
how deep a corner is thrown at. That is the same shape as FTN role (0.748) and
blocking scheme, and it carries the same warning: a corner who is avoided and a
corner who is good are indistinguishable in an outcome rate. Only the volume
column tells them apart, which is why the shipped page puts them side by side.

### Against the board's own error

Grades were assembled into a team coverage number — target-weighted over the
corners who actually played, prior weeks only, shrunk toward the league rate by
60 targets, built from completion rate alone because putting yardage in would
import the noise the table above just rejected. Plus `cover_spread`: the
completion-rate gap between a secondary's best and worst regular corner, which
is the part of a "pick on him" read that does **not** need the assignment.

Scored against the production board's residual, reusing the backtest rows cached
by `blitz_qb.py` — 2021-25, `opponent=False` as served, so the board carries no
coverage information of any kind. 34,640 player-weeks. `+quality` is the same
slope fit beside points the defence has already conceded to that position:

| pos | grade | n | slope | 95% CI | +quality |
|---|---|--:|--:|---|--:|
| **WR** | completion % allowed | 16,380 | −0.005 | [−0.116, +0.101] | −0.017 |
| **WR** | corner spread | 16,380 | +0.068 | [−0.027, +0.166] | +0.061 |
| **WR** | corner aDOT | 15,554 | +0.113 | [−0.015, +0.228] | +0.099 |
| TE | corner spread | 5,938 | +0.126 | [+0.023, +0.226] | **+0.107** |
| RB | completion % allowed | 9,122 | +0.355 | [+0.219, +0.501] | **+0.304** |
| RB | corner aDOT | 8,678 | −0.249 | [−0.406, −0.105] | **−0.222** |
| QB | corner spread | 3,200 | +0.460 | [+0.197, +0.756] | **+0.327** |
| QB | corner aDOT | 3,034 | +0.334 | [+0.055, +0.626] | **+0.292** |

**Wide receiver is null on all three grades**, which is the third independent
failure for the WR coverage matchup — team-level yards allowed, per-corner
grades, and now the board's residual. The positions that light up are the ones
nobody asked about: a secondary whose corners allow completions and are tested
shallow concedes underneath, which is where a back's receiving work lives, and
an uneven secondary shows up for tight ends and quarterbacks. Twelve tests were
run, so read the survivors as candidates.

### And nothing pays

Each position's best grade, fitted leave-one-season-out, applied as one additive
term with the intercept dropped (#31 again):

| pos | grade | board MAE | corrected | gain | 95% paired CI | top-N delta |
|---|---|--:|--:|--:|---|--:|
| WR | aDOT | 3.9783 | 3.9815 | **−0.0032** | [−0.0059, −0.0007] | −0.0000 |
| TE | spread | 3.1469 | 3.1466 | +0.0003 | [−0.0026, +0.0033] | +0.0019 |
| RB | cmp % | 4.2899 | 4.2896 | +0.0003 | [−0.0073, +0.0080] | −0.0065 |
| QB | spread | 6.5116 | 6.4971 | +0.0145 | [−0.0031, +0.0320] | −0.0019 |

Wide receiver is the only interval clear of zero and it is on the harmful side.
The rest are zero, and the ordering column is flat or slightly negative
everywhere. This is the blitz-rate result again, arrived at from a completely
different direction: a clear slope against the board's residual, worth nothing
once it is actually applied.

### Shipped as a report, not as an input

`nfl_projections/coverage.py` — per-corner grades, secondary grades with
`cover_spread`, and a weekly matchup table — plus `/coverage/{corners,
secondaries,matchups}` on NFL-API and a Coverage Report tab beside the Trench
Report. 13 tests, including one that asserts `predict.py`, `blend.py`, `roles.py`
and `model.py` do not import the module, so the measured decision cannot be
undone by accident.

The page shows yards per target greyed out with the 0.19 next to it, shows
target volume beside every grade, and labels every wide receiver `context only`.
The one coverage adjustment that remains shipped anywhere is still tight-end
receiving in sharp-edge `nfl/matchup.py`, from the team-level work.

Sanity check against an outside source, which is a check and not evidence: a
podcast segment this week named Washington, Chicago and Arizona as the softest
coverage, and Mike Sainristil as the corner to attack in Washington. This module
independently puts Washington 7th and Chicago 8th of 32 and names Sainristil as
Washington's weakest regular corner. Arizona comes out 12th, softer than average
rather than bottom three. Both readings may well be derived from the same PFR
charting, so agreement here means the pipeline is wired up correctly and nothing
more.

## Opportunity share: the premise is right, the board already has it (2026-10-16)

The ask was for the optimizer to prefer players who "have their hands on the
ball a lot ... underpriced wide receivers that get volume, but maybe haven't
scored a lot of touchdowns", with targets and rush attempts both counting for a
running back. Harness: `experiments/opportunity_share.py`, 58,940 player-weeks,
2018-26, 1,561 players.

**One part of the ask cannot be measured at all.** There are no historical
FanDuel salaries in either repo and no free source for them (DFS pricing
sources were swept 2026-10-07; only the CSV upload and paid feeds survive). So
nothing here compares a player to his price. Everything below is measured
against the **board**, and "underpriced" follows only if FanDuel's salary tracks
recent fantasy points — an assumption about FanDuel, not a result in this file.

Opportunity is `targets + carries` for a skill position. For a quarterback it is
`attempts + carries`: fitting points on targets for a quarterback returns 5.1
points per target off a column that is almost entirely zeroes, which is how that
definition announced itself as wrong.

### The premise is not just right, it is the strongest signal in this file

Odd weeks against even weeks, per player:

| | in season | career |
|---|--:|--:|
| opportunities / game | **0.974** | **0.987** |
| opportunity share | **0.977** | **0.986** |
| targets or attempts / game | 0.979 | 0.989 |
| snap share | 0.963 | 0.979 |
| FanDuel points / game | 0.847 | 0.942 |
| touchdowns / game | 0.810 | 0.928 |
| **touchdowns / opportunity** | **0.178** | **0.288** |

Volume is very nearly deterministic; scoring is not; and the part of scoring
volume cannot explain — touchdowns per opportunity — is close to a coin flip.
Two players with the same volume are very nearly the same bet however
differently they finished. That is the ask's reasoning, and it holds.

(These levels are inflated by between-player spread, since a star and a scrub
are both in the pool. The ordering is the finding, not the absolute values.)

### But the board carries volume already, so only the residual is news

The board's features include `targets`, `carries`, `target_share`,
`air_yards_share`, `wopr` and an `opportunity_score`. Against its own error on
the cached 2021-25 production backtests (27,297 scored player-weeks, board MAE
4.1935), volume by itself is **nothing**, and that is the correct result:

| position | opp share | xFP | volume-TD gap | gap, +predicted held |
|---|--:|--:|--:|--:|
| QB | −0.025 | −0.015 | +0.072 (p=0.53) | −0.163 |
| RB | +0.102 | −0.012 | **+0.611** [+0.385, +0.827] | +0.515 |
| WR | +0.061 | −0.040 | **+0.633** [+0.452, +0.806] | +0.415 |
| TE | +0.654 | +0.202 | **+0.722** [+0.488, +0.977] | +0.685 |

Twelve slopes tested. `opp share` and `xFP` alone are null everywhere the sample
is large — the board has them. What it does **not** have is the *gap* between
volume-implied points and points actually scored: p=0.003 at RB, WR and TE, same
sign, and it survives holding the board's own projection. The board is partly
chasing recent touchdown luck, exactly as the ask supposed.

### What it is worth: MAE yes, order barely

Leave one season out, slope only (#31: a per-position bias shift is a separate
result and would be charged to this term):

| | MAE gain | 95% paired CI | weeks won | top-N delta | 95% paired CI |
|---|--:|--:|--:|--:|--:|
| WR gap | +0.33% | [+0.0008, +0.0279] | 43/83 | +0.0035 | [−0.0060, +0.0131] |
| WR opp share | +0.14% | [+0.0020, +0.0091] | 51/83 | **+0.0020** | **[+0.0005, +0.0040]** |
| TE gap | +0.46% | [+0.0007, +0.0300] | 45/77 | −0.0044 | [−0.0200, +0.0111] |
| TE opp share | **+1.95%** | [+0.0535, +0.0763] | **72/77** | −0.0044 | [−0.0123, +0.0022] |
| RB gap | +0.20% | [−0.0066, +0.0249] | 47/83 | +0.0042 | [−0.0105, +0.0190] |
| QB (all) | ≤0 | — | — | ≤0 | — |

`WR opp share` is the **first ordering gain of this whole run of entries whose
interval clears zero** — and it is 0.2 points of top-24 overlap, improving 4 of
83 weeks. The mean is carried by a handful of weeks and the rest are ties.

`TE opp share` is the instructive one: it wins **72 of 77 weeks on MAE, 1.95%**,
and makes the order *worse*. A term can compress errors without reordering
anything, and the optimizer reads nothing but the order.

### The promoted backup is backwards

"Backups starting due to injury" was tested as last week's opportunity share
against the season's, which is the form the board's season-rolling features
would miss:

| position | slope on board error | 95% CI | +predicted |
|---|--:|--:|--:|
| QB | +0.133 | [−0.007, +0.287] | +0.132 |
| RB | −0.141 | [−0.290, +0.006] | −0.067 |
| WR | **−0.964** | [−1.204, −0.715] | −0.769 |
| TE | **−0.640** | [−0.993, −0.292] | −0.566 |

Negative. On the clearest promotions — last week's share at least double the
season's — receivers actually scored **3.688 against a 4.470 projection**, 997
of them. A one-week volume spike is mostly a blowout or a garbage-time binge,
the board chases it, and the player disappoints. Chasing last week's volume is a
trap at receiver and tight end. The one place the instinct may hold is
quarterback (+2.134 residual on 56 cases) where a replacement starter inherits
the entire workload, but 56 cases and an interval touching zero is not a finding.
The *real* version of this is a known depth-chart or injury change, which
`roles.py` already handles and which the promotion entry already scored.

### Matchups move volume and still do not pay

This was the one matchup term with a measured reason to exist: the blitz entry
found a defence's blitz rate moves running-back carries (−0.320) and receiver
targets (+0.193) while leaving points alone, so volume is where an opponent
effect was last seen alive. Opponent opportunity allowed, from that defence's
prior weeks only, does move a player's own opportunities (QB +1.082, WR +0.151
p=0.003) and does predict the board's error — **QB +1.554 [+0.498, +2.582],
p=0.003, surviving the projection control at +1.524, the largest single slope in
the file.** The negative control is clean: the same number joined to the
player's *own* defence straddles zero at every position, so the merge is right.

Held out, it is **harmful**: QB −1.89% of MAE [−0.2150, −0.0281], TE −3.96%,
RB −1.26%, and QB ordering −0.0103. The biggest slope in the file costs about
two percent of quarterback MAE when it is not allowed to see its own season.

### Decision

**Nothing changes `lineup_points` and nothing changes the board.** The ordering
bar is the one the optimizer actually reads, and the single term that clears it
is worth 0.2 points of top-24 overlap in 4 weeks out of 83.

What ships is **visibility, not scoring**: the optimizer frame carries
`opp_per_game`, `opp_share`, `xfp_per_game` and `volume_td_gap` beside `value`,
so the player who gets volume and has not scored is *legible* in the pool the
lineups are built from. The same shape as the Coverage Report — a measured
number shown next to the board, explicitly not inside it. Tests assert the
optimizer's objective columns are untouched.

Three things are now measured and should stop being re-asked: volume is a trait
and touchdown rate is not (so a volume-light, touchdown-heavy player is the one
to fade, which the board partly knows); last week's volume spike is a fade at
WR/TE, not a buy; and an opponent's volume allowed is the fourth matchup
instrument to produce a real slope and fail out of sample.

## Scoping the defense slot: it is matchup plus noise (2026-10-16)

Not an experiment yet — the reliability measurement that has to come first,
prompted by "our defensive projections could use some work". `POSITIONS =
["QB", "RB", "WR", "TE"]`: **the model does not project defenses at all.** The
optimizer sets `lineup_points = FPPG` for them, so one slot in nine is picked
from FanDuel's own season average with no view of the opponent. 4,222 team-weeks
2018-25, FanDuel DST scoring reconstructed from `load_team_stats` plus schedule
scores for the points-allowed tier.

**Corrected 2026-10-16**, while writing the lineup scoring job: the first pass
counted return touchdowns twice, adding `pt_return_tds` on top of `def_tds +
special_teams_tds`. A team's `special_teams_tds` reconciles exactly with the sum
of its own players' (544 of 544 team-weeks in 2025); `pt_return_tds` does not
(it disagrees on 39), so it is a different measure and does not belong. The
numbers below are the corrected ones. Nothing about the conclusion moves.

DST fantasy points: mean **6.28**, sd **5.62**. Odd weeks against even weeks:

| component | mean | sd | in season | career |
|---|--:|--:|--:|--:|
| QB hits | 5.65 | 3.03 | **0.330** | **0.641** |
| points allowed | 22.89 | 10.00 | 0.286 | 0.625 |
| sacks | 2.40 | 1.75 | 0.282 | 0.525 |
| points-allowed tier points | 0.48 | 2.65 | 0.254 | 0.584 |
| passes defended | 4.26 | 2.43 | 0.185 | 0.396 |
| interceptions | 0.77 | 0.91 | 0.155 | 0.326 |
| takeaways (INT + fumble) | 1.29 | 1.18 | 0.109 | 0.327 |
| **defensive / return TDs** | 0.10 | 0.32 | **−0.013** | 0.138 |
| **safeties + blocked kicks** | 0.07 | 0.27 | **−0.026** | 0.093 |
| **DST fantasy points** | 6.28 | 5.62 | **0.192** | 0.561 |

For comparison, a receiver's own points per game split 0.847 in season. A
defense's split **0.192**. There is a persistent team quality (career 0.561),
but a single season barely sees it.

### The variance is in the parts nobody can predict

Mean contribution to a week's DST score, with its standard deviation:

| source | mean | sd |
|---|--:|--:|
| takeaways | 2.59 | 2.36 |
| sacks | 2.40 | 1.75 |
| **points-allowed tier** | 0.48 | **2.65** |
| **defensive / return TDs** | 0.62 | **1.93** |
| safeties + blocks | 0.20 | 0.63 |

The two largest sources of week-to-week swing are the tier (sd 2.65) and
touchdowns (sd 1.93) — and touchdowns are **noise by measurement**, −0.013
in season and 0.138 over a career. Sacks carry a comparable mean with less than
half the spread and four times the persistence. So the scoring splits three
ways: a predictable low-variance floor (pressure), a semi-predictable
high-variance swing (the tier), and a high-variance component that should be
shrunk to the league mean for every team and never modelled.

### The tier is a step function, so its mean is the wrong thing to project

Tier frequencies: −4 at 11.8%, −1 at 18.6%, 0 at 26.5%, +1 at 25.1%, +4 at
13.4%, +7 at 3.6%, +10 at 1.0%. Projecting mean points allowed and reading off
the tier is not the same as the expected tier value, and the gap is largest for
defenses facing volatile offenses — which is precisely the case a projection
needs to get right. This wants `simulate.py`, not a point estimate.

### What this does and does not share with the earlier defense nulls

"Why opponent/matchup keeps failing" found defense-vs-position barely persists
and explains 1-4.5% of a player's single game. That was defense as an
ADJUSTMENT to a skill player. Here the defense is the whole bet, so the null
does not transfer — but the same instability is visibly the reason the slot is
hard, and it caps how much any model can win.

Also noted: this is the position where the no-Vegas rule bites hardest, since
the industry estimator for points allowed is the opponent's implied team total.
The substitutes already in the repo are `ratings.py` (SRS, team grades) and its
`game_environments`.

No decision yet. Nothing has been built or shipped.

## The first scored lineups: the edge is real and it is in the top five (2026-10-20)

The first week with built lineups scored against what happened — 2026 week 6,
FanDuel main slate (Sunday 13:00–17:59 ET), 14 games, cap 60,000. Three builds
off one slate: ceiling × 12 at 50% max usage, mean × 12 at 50%, median × 15 at
40%. Baselines rebuilt from the stored slate CSV: the lineup FanDuel's own FPPG
would have built from the same pool under the same cap scored **116.34**, and
the best lineup the slate held, in hindsight, **212.66**.

**As submitted, every build lost to FPPG. Its top five beat it by 11%.**

| take top | mean actual | vs FPPG |
|---|--:|--:|
| 1 | 126.1 | +8.4% |
| 2 | 126.5 | +8.7% |
| 3 | 127.9 | +10.0% |
| **5** | **129.1** | **+11.0%** |
| 8 | 123.5 | +6.2% |
| 10 | 115.7 | −0.5% |
| 12 | 108.0 | −7.2% |
| 15 | 100.1 | −14.0% |

Per build, mean actual against a 116.34 baseline: ceiling 107.65 (6 of 12 lineups
beat it), mean 107.67 (7 of 12), median 100.11 (7 of 15). Pooled, 20 of 39 beat
it — a coin flip — and the mean is −10.0%. The whole edge lives in the ordering,
and past about eight lineups it is spent.

The ordering is what an optimizer is for, and it holds: projected against actual
**within** a build correlates 0.92 (ceiling), 0.69 (mean), 0.93 (median). The
pooled figure is +0.29 and is an artefact — the three objectives run on different
scales (mean projection 213, 112, 91), so pooling them mixes units. Ignore it.

A separate and better-than-expected signal: the `mean` objective's point total
was **112.25 projected against 107.67 actual**, 4% high. That is the absolute
scale, not the ordering, and nothing in this file had measured it before.

### What this is not

**One slate.** Thirty-nine lineups, but they share a pool, a projection vintage
and a usage cap, so they are nothing like 39 observations; the effective n is
about one. The sd of a single lineup is 28.5 points.

**The peak at five is fitted on the data that suggested it.** 3, 5 and 8 are
127.9 / 129.1 / 123.5, which on one slate is indistinguishable. What the week
does support is the *shape* — a good top, a long bad tail, and a crossover under
FPPG somewhere near ten. The location of the peak needs more weeks.

**All three objectives produced the identical top lineup** — Cousins, Swift,
Walker, Warren, Olave, Flowers, Boston, Likely, Raiders. So "top 1 = 126.1" is
one lineup counted three times. Ceiling, mean and median are far less
independent than three builds makes them look, which is worth knowing before
reading anything into agreement between them.

**69% of what was there.** Model best 147.6 of 212.66; FPPG 116.34, or 55%.
Nobody reaches the hindsight best, so it marks the ceiling rather than a target.

### The one thing already worth a second look

Build 4 is an accidental experiment: 40% max usage and 15 lineups gave the
**best** top five (132.3) and the **worst** overall (100.1). A tighter usage cap
buys a better head and a longer tail. If that repeats, the cap is a dial on where
the crossover sits, and it is free to measure — the builds are being generated
anyway.

## Week 7 lineups: the hits and the misses differ by one player, not by one shape (2026-10-26)

Second scored week, and the first with enough builds to ask the question that
matters for actually entering a contest: given a pile of lineups, can anything
visible beforehand tell the ones that hit from the ones that did not?

2026 week 7, FanDuel main slate, 12 builds, 195 lineups, 185 scored; 170 of them
complete classic main-slate lineups, which is the population used below. FPPG
baseline **73.52**, hindsight best **235.58**, pool mean **92.8** (sd 27.6),
best **190.98**, worst **31.3**, and 76% beat FPPG — a far softer baseline week
than week 6, where only 51% did.

### Thirteen lineup traits, two weeks, nine builds

For every lineup: the four projection totals rebuilt from the published board
(mean/floor/median/ceiling), salary spent, count of sub-$5k punts, max and min
salary, QB-stack count, bring-back count, distinct games, mean pool frequency of
its nine players ("chalk"), and its most-used player's frequency. Spearman
against actual **inside each build** (across builds is meaningless — the
objectives are on different scales), over the nine builds of ≥10 lineups in
weeks 6 and 7.

| trait | median rho | neg | pos |
|---|--:|--:|--:|
| salary spent | 0.58 | 1 | 8 |
| ceiling total | 0.58 | 0 | 9 |
| mean total | 0.53 | 1 | 8 |
| median total | 0.52 | 1 | 8 |
| punts (≤$5k) | −0.50 | 6 | 1 |
| chalk | 0.48 | 1 | 8 |
| max salary | 0.36 | 2 | 7 |
| bring-back | −0.23 | 6 | 2 |
| floor total | 0.22 | 2 | 7 |
| max exposure | −0.13 | 5 | 2 |
| min salary | 0.10 | 3 | 6 |
| distinct games | −0.04 | 5 | 4 |
| QB stack | 0.01 | 4 | 5 |

The top of that table is one thing wearing four hats: the optimizer spends
salary to buy projection and reuses the players it likes, so salary, chalk and
the projection totals are the same variable. **Strip the projected rank out and
nothing is left.** Correlating each trait with the residual after regressing
actual rank on projected rank, inside each build: punts −0.26 (5 of 7 negative),
bring-back −0.18 (5 of 8), QB stack +0.13 (6 of 9 positive, median effect a
third of a rank), everything else inside ±0.1. Two weeks cannot separate any of
those from zero. The one oddity worth a note is the ceiling total, positive in
**9 of 9** builds against a mean-rank residual — tiny (+0.09) and contaminated
by its collinearity with the mean, but it never once had the wrong sign.

### What did separate them

| owned | in | with | without | swing | he scored |
|---|--:|--:|--:|--:|--:|
| Trey McBride | 8% | 131.1 | 89.4 | +41.7 | 25.1 |
| Tony Pollard | 33% | 114.1 | 82.4 | +31.7 | 38.0 |
| Jonathan Taylor | 15% | 115.0 | 88.8 | +26.2 | 18.0 |
| Bo Nix | 28% | 111.0 | 85.7 | +25.3 | 24.2 |
| … | | | | | |
| Brian Robinson Jr. | 13% | 69.1 | 96.4 | −27.3 | 0.0 |
| Bryce Young | 13% | 66.3 | 96.8 | −30.5 | 7.4 |

Against a pool sd of 27.6, a single name moved a lineup a full standard
deviation. A week's hits and misses are a referendum on six or eight players,
which is why no construction rule shows up: there is nothing left for one to
explain.

### The bigger lever was which build, not which lineup

Two vintages exist for week 7. Builds 5–8 (Tuesday 10-20, one 25-lineup build
per objective) averaged **97.6**; builds 9–16 (Saturday night, 70 lineups)
averaged **86.1**, and six of them returned the *identical* top lineup, which
scored 77.2. The Saturday builds converged on Hubbard / Likely / Coker / Golden
/ LaPorta / B-Robinson / Bryce Young at ~31% each, and that core busted.

Like for like — same Tuesday vintage, 25 lineups, one build per objective:

| objective | top 1 | top 3 | **top 5** | top 10 | all 25 |
|---|--:|--:|--:|--:|--:|
| ceiling | 132.6 | 155.2 | **149.7** | 130.8 | 95.4 |
| floor | 125.3 | 117.8 | **114.9** | 110.3 | 104.7 |
| mean | 128.6 | 124.1 | **111.8** | 107.7 | 93.5 |
| median | 99.6 | 100.2 | **101.2** | 93.4 | 96.7 |

**Ceiling has the best head and the worst body.** That is the week-6 shape
sharpened: good top, long bad tail, crossover near ten. It also means
`GET /optimizer/performance`'s objective verdict for the week — floor best at
102.7, ceiling worst at 84.7 — is an artefact of the mix: seven of the twelve
builds were ceiling and most of those were the Saturday ones. Averaged over
every lineup, ceiling loses; at the top five, it wins by 35 points.

Rules for picking five, scored on both weeks (random five from the week's pool
as the control):

| rule | wk6 mean | wk7 mean |
|---|--:|--:|
| top 5 of the ceiling build | **131.9** | **149.7** |
| top 5 of the median build | 132.3 | 101.2 |
| top 5 of the mean build | 123.3 | 111.8 |
| pooled across builds, top 5 by mean total | 127.0 | 85.2 |
| pooled, top 5 by mean, 3-of-5 exposure cap | 110.2 | 85.0 |
| random five | 104.7 | 92.8 |

Pooling builds and re-ranking them on a common ruler is *worse than one build*,
because the common ruler is the latest board and the latest board is what built
the Saturday core.

### What this is not

Two slates. 170 lineups out of one pool is not 170 observations — they share a
projection vintage, a cap and most of their players, and six builds are
near-duplicates of each other (10 and 16 are identical). The vintage result is
one week and the opposite of what a staleness story would predict, so it is a
thing to watch, not a finding. And "top 5 of the ceiling build" quietly means
*the earlier* 25-lineup ceiling build in week 7; the Saturday one's top five
scored 84.4. Picking the build is still being done by hand, and on this evidence
it matters more than everything measured above.

## Why week 7's later builds were worse: it was a different board, not a fresher one (2026-10-26)

Follow-up to the entry above, which found that week 7's Tuesday builds (5–8)
averaged 97.6 and the Saturday-night builds (9–16) 86.1, and left the reason
open. The reason is not staleness, and the slate is innocent.

**The pricing never moved.** One week-7 CSV exists — slate_id 3, uploaded
2026-10-20T21:30:36Z, content hash `90e662c4a394`, 723 players — and all twelve
builds used it. Same pool, same salaries, same FPPG.

**The Tuesday builds read the August preseason board.** Matching each build's
stored per-player projection against every archived vintage of week 7, by the
build's own objective:

| build | when | best-matching vintage | mean abs diff |
|---|---|---|--:|
| 5 ceiling | Tue | **as_of 0** (2026-08-11) | **0.00** |
| 6 floor | Tue | **as_of 0** | **0.00** |
| 7 median | Tue | **as_of 0** | **0.00** |
| 8 mean | Tue | **as_of 0** | **0.00** |
| 9–16 | Sat | as_of 5 / 6 | 0.4–1.7 |

Exact, to the cent, on every player. `player_projections` still held week 7's
preseason row because the in-season recompute fires at 21:00 Denver on Tuesday
and the build ran at 15:30 Denver on Tuesday — about five hours early. Build 9,
the next afternoon, already matches the in-season board. So "earlier in the
week" meant the ESPN-roster preseason pipeline (v1.5.0, availability-weighted),
not a younger copy of the in-season one (v1.6.0).

**The late board was sharper, exactly where it should be.** Among players the
Tuesday ceiling build used: Tyler Warren 14.1 → 0.0 (scored 0.0), Troy Franklin
12.4 → 3.9 (0.0), Chimere Dike 8.6 → 1.4 (0.0), Jauan Jennings 13.4 → 5.7 (4.4).
The injury and inactive updates did their job on every one.

**It lost on one team.** Carolina took 16% of all Saturday player-slots — Bryce
Young, Hubbard, Coker, Tremble — and lost to Tampa Bay 43-13.

| Panthers in the lineup | lineups | mean actual |
|---|--:|--:|
| 0 | 102 | 101.6 |
| 1 | 30 | 87.5 |
| 2 | 35 | 75.1 |
| 3 | 3 | 54.1 |

Carolina was not a wild call. Young was the board's third QB at 19.68 and
cheap at $7,800, and the in-season QB tier is flatter than a quarterback's own
week-to-week noise — top-1 to top-12 spans **3.36 points**, and the four best by
value are Cousins 2.71, Stafford 2.67, Goff 2.60, Young 2.52 points per $1k.
Three of those four hit (20.4, 18.5, 19.0) and one scored 7.4. **At the top of
the board the optimizer is choosing between ties it cannot read, on price.**

**The preseason board's handicap is what protected it.** Season-long
availability weighting (mean exp_games 0.27 at QB, 0.79–0.83 at RB/WR) scales
every number down and flattens the field — top-24 QB mean 11.25 against the
in-season 16.20. Unable to tell anyone apart, the optimizer spread across 51
players, never more than 2.13 from one team on average, and left $8,300 of the
cap unspent. It bought an index fund by accident.

**What is *not* the explanation:** the sharp board did not collapse into
sameness. Saturday's lineups were *more* varied — mean pairwise overlap 0.21
against Tuesday's 0.32, 65 distinct players against 51. The concentration was by
team, not by player, which is the one axis the optimizer does not look at.

### Two things this says to build

1. **A build does not record the board it read.** Nothing in `lineup_builds`
   says `computed_at`, so a two-month-old projection produced lineups with no
   warning anywhere. Stamp the vintage on the build, and refuse — or at least
   shout — when the board for that week predates the last completed week.
2. **The board projects players independently and the optimizer stacks them
   freely.** No correlation term exists, so nothing stopped three Panthers in a
   lineup. A per-team or per-game cap is the cheap structural fix, though it
   would not have saved this week: only 3 of 170 lineups held three Panthers.

One week, and the Tuesday board's win is mostly Tony Pollard scoring 38.0
against a 17.6 ceiling. The finding here is the vintage mix-up and the flat QB
tier, not "build on Tuesday."

## Defenses: FanDuel's FPPG is worse than a flat average, and the pieces beat both (2026-10-26)

The DEF slot has always been filled with FanDuel's own season-average FPPG,
because the board projects no defenses. That turns out to be worse than
filling it with a constant.

Walk-forward, train on every prior season, predict the next, 3,906 team-weeks
over 2019-2026. Actual DST scores average 6.21 with an sd of 5.57.

| method | MAE | RMSE | rho | top-3 picked |
|---|--:|--:|--:|--:|
| FanDuel FPPG | 4.63 | 6.00 | 0.113 | 7.16 |
| trailing 4 games | 4.81 | 6.20 | 0.101 | 6.89 |
| **flat league average** | **4.31** | **5.57** | — | 6.51 |
| ridge on the FanDuel total | 4.21 | 5.42 | 0.237 | 8.23 |
| **components, tier integrated** | **4.18** | **5.42** | **0.244** | **8.50** |

"top-3 picked" is the mean actual score of the three defenses each method
liked best that week — the number that pays, since the slot takes one name.

**A defense's own season average is nearly worthless as a ranking.** FPPG
correlates 0.113 with what a defense goes on to score, and it loses to a
constant on error, because sacks and takeaways repeat weakly and most of the
fantasy points are the points-allowed tier, which is mostly the opponent.

### What the model does

Four component ridges (sacks, interceptions, fumble recoveries, defensive and
return touchdowns) on the defense's own blended rates and the offence it is
facing, scored at FanDuel's rates, plus the points-allowed tier. The tier is
the part worth writing down: it is a **step function**, so `E[tier(PA)]` is
taken over a distribution — the predicted points allowed plus the empirical
spread of the model's own residuals — rather than evaluating `tier(E[PA])` at
the mean. A defense projected to allow 21 sits exactly on a boundary where the
step says 0 and the distribution says about 1.2.

Every rate is a three-way blend of this season to date, last season and the
league, weighted by games, so week 2 is not decided by one fluke sack total.

Per season, MAE and the weekly top three, FPPG → model:

| season | MAE | top 3 |
|---|--:|--:|
| 2019 | 5.04 → 4.57 | 8.51 → 9.02 |
| 2020 | 4.45 → 3.97 | 6.45 → 8.37 |
| 2021 | 4.74 → 4.37 | 7.39 → 7.81 |
| 2022 | 4.31 → 3.90 | 7.93 → **7.54** |
| 2023 | 4.87 → 4.40 | 6.87 → 9.24 |
| 2024 | 4.50 → 4.11 | 7.04 → 9.00 |
| 2025 | 4.59 → 4.11 | 6.07 → 9.20 |
| 2026 | 4.47 → 3.82 | 6.71 → **6.10** |

Better on error in all eight; better on the top three in six.

### The market is still ahead

Ranking defenses by the **opponent's implied total** picks a top three worth
about 9.0 a week, and the implied total correlates 0.396 with points allowed
where this model manages 0.298. Lines are kept out of the features on purpose
— they are the benchmark, not an input — so the honest statement is that this
closes most of the gap between FPPG and the market without closing it.

### Negative result

Adding defensive pressure (QB hits, tackles for loss) and the opponent's pass
rate made it *worse*: rho 0.236 → 0.229, top-3 8.45 → 8.36. The sack rate
already carries the pressure, and the extra columns only cost degrees of
freedom.

### One bug worth remembering

The running season-to-date totals were built with `cumsum`, which leaves a
hole from the first row with no outcome onward. Invisible in a backtest, where
every row has been played — and it zeroed the whole season's form on exactly
the rows a live projection asks about, so the first live week came back with
every defense projected at 9.5-10.5 against a league mean of 6.3. There is a
test for it now. A second, smaller one went with it: the league prior was an
expanding mean over *rows*, so a row partway through a week was handed the
average of that same week's earlier games. Fixing it moved the model from rho
0.236 to 0.244.

Shipped as `nfl_projections.defense`, with `optimize_lineups` reading a
`defense_points` column when one is joined on and falling back to FPPG per row
when it is not.

## The optimizer learns that nine players can be one bet (2026-10-26)

Out of the week 7 post-mortem above: the board is made one player at a time
and the builder had no way to express that nine of them can be one bet on one
offence. Three rules now exist, all solved exactly rather than filtered after
the fact:

  * `max_per_team` — at most N of the nine from one team, defense included.
    **Defaults to 4, which is FanDuel's own limit**: the builder could
    previously return a lineup the site would refuse to enter.
  * `stack_qb` — at least N WR/TE from the quarterback's team.
  * `rb_with_qb=False` — no running back from his own quarterback.

The solver is a cost-profile convolution over slot pools, which sees price and
nothing else, so the rules needed a place to live. The teams a rule names get
split out of every slot pool and the running state carries how many of each
have been taken; teams nobody is asking about stay in one undifferentiated
pool and cost nothing. The team cap is applied lazily — solve, and only count
a team once it has actually broken the cap — because counting is what costs,
and on a real slate the cap binds on nobody.

A first attempt enforced the cap by branch and bound outside the solve,
excluding one offending player per branch. It is exact and it passed 140
brute-force cases, but on a slate where one team is the best value at every
slot it needs to exclude five or six players and the branching is
one-at-a-time: 16 seconds and then no lineup at all when the node budget ran
out. Recorded because the failure is not obvious from the outside — it only
shows up on the slates the rule exists for.

Verified against brute force on 224 small slates across every combination of
the three rules: no mismatches. On a real 2026 main slate (527 players), one
lineup takes 0.15-0.36s against 0.17s unconstrained, and a 25-lineup build
9.0s against 6.4s. Dominance pruning — dropping any player with k others
cheaper-and-better, where k is the number that slot needs — made the plain
path slightly faster than it was before the rules existed.

Builds now also record `max_per_team`, `stack_qb`, `rb_with_qb` and
`projection_computed_at`, which is the gap the week 7 entry above named: the
board vintage a build read was nowhere on file.

## The field, and the players it has already written off (2026-10-28)

The board has never had any read on what everyone else is doing. Footballguys
publishes projected DFS roster percentages per site and week, free, as plain
HTML tables, and `nfl_projections/ownership.py` now reads them.

The numbers are a real projection and they prove it themselves. Across a
FanDuel classic roster the positions have to sum to one QB, one defence and
2.4/3.2/1.2 of the flexible spots, and they do:

| week | QB | RB | WR | TE | DEF |
|---|--:|--:|--:|--:|--:|
| 6 | 98.9% | 255.5% | 315.9% | 120.5% | 100.0% |
| 7 | 99.1% | 236.5% | 323.8% | 121.7% | 99.3% |
| expected | 100 | ~240 | ~320 | ~120 | 100 |

That accounting is checked on every parse, because it is the cheap way to
notice the page has changed shape.

### The first thing it found was our own slate confusion

Attached to the week 6 FanDuel CSV, 33 players the board projected above 8
points had no ownership at all — Purdy, McCaffrey, Stroud, Jordan Love,
Lawrence, Daniels. All of them played. All of them scored.

They were not contrarian. The source projects **one** slate, the Sunday
13:00–16:25 windows, which is 20 teams. The CSV FanDuel hands out is the whole
week, 28 teams, and the eight extra were Thursday night, the London game,
Sunday night and Monday. Filtered to the slate the optimizer already calls
`main`, 145 of 146 published names matched.

The guard for this is not the obvious one. Match rate measured on the
published side was 145 of 146 — a *wider* pool is a superset and matches
everything. `ownership.coverage` measures the pool side instead: of the
players the board rates above 6 points, what share has a published number. On
the real files that reads 63% for the full CSV and 89% for the main slate.

### Unlisted is not zero, it is a scratch

Inside the right slate, an omission turned out to carry information the board
does not have.

| | listed | played | unlisted | played |
|---|--:|--:|--:|--:|
| wk 6, projected > 10 | 40 | **100%** | 5 | **0%** |
| wk 7, projected > 10 | 47 | **100%** | 4 | 50% |

The week 6 five were Lamar Jackson (projected 16.0), Hurts (13.3), Mayfield
(12.7), Breece Hall (11.6) and Monangai (11.5). Not one of them suited up, and
our own `exp_games` had all five at **1.00** — in-season availability
weighting catches none of this.

So an unlisted player's ownership is NaN, never a floor, and is never charged
for. Treated as 0% it would be the most attractive name on the board, and the
optimizer would roster the inactive list. It is reported separately
(`flag_unlisted`, and `unlisted_but_projected` on the API) as something to
check rather than something to exclude: in week 7 Kirk Cousins was unlisted at
a projected 18.4 because the field expected Penix to start, and he went 20.4.

### Charging for the field costs points, as it should

`optimize_lineups(ownership_weight=λ)` subtracts λ points per percent owned.
It is linear, so the solve stays exact. 2026 week 6 main slate, best five:

| build | top-5 actual | total ownership |
|---|--:|--:|
| no charge | 125.1 | 103.5% |
| λ = 0.10 | 109.3 | 80.7% |
| ceiling, λ = 0.10 | 100.1 | 88.2% |
| ceiling, λ = 0.25 | 93.0 | 64.6% |

It buys the tilt and pays for it in points, which is what a leverage dial is
supposed to do. Nothing here says whether that trade is worth making, because
raw score is the wrong yardstick for it — the right one is tournament finish,
and that needs a model of the field's lineups rather than of its ownership.
Logged as untested, not as working.

## A lineup's shape is not a free parameter (2026-10-28)

Built `simulate.pool_sims`, which turns a player pool into a correlated
per-simulation matrix, and `optimize_lineups(target=…)`, which solves a
candidate set and re-ranks it by the chance of clearing a score. The target
was the user's: 125 points, the number that cashes.

### The summed ceiling is as wrong as it looked

Ranking by `objective="ceiling"` adds up nine players' individual 90th
percentiles. A lineup's mean grows with nine players while its spread only
grows with three, so that sum is not a ceiling, it is an impossibility. On a
week 6 eight-player core: summed marginal p90 **238.7**, true lineup p90
**169.9**, and the summed number sits at the **99.38th percentile** of what
the lineup can actually do.

The simulator's correlations come through as designed — Stafford↔Nacua 0.487,
C. Williams↔Odunze 0.347, Allen↔Shakir 0.324, cross-team 0.005.

### And replacing it changes nothing, for a reason worth knowing

Over 343 distinct week 6 lineups, every ranking metric is the same metric:

| ranked by | corr. with actual | top-5 actual | top-10 |
|---|--:|--:|--:|
| P(≥125) | 0.764 | 126.5 | 125.7 |
| projected mean | 0.829 | 127.7 | 123.4 |
| summed ceiling | 0.813 | 124.3 | 125.6 |
| summed floor | 0.811 | 128.8 | 127.6 |

Sweeping the target from 100 to 180 gives **overlap 1.00 with the mean-ranked
five at every single target**. The reason is measurable: among the top
quintile by projected total (mean 119.5 ± 3.8), lineup standard deviation is
31.05 ± 0.90, and after removing the mean the residual spread in that standard
deviation is **0.750 points — 2.4%**. Spearman between P(≥125) and projected
mean is **0.9986**.

Nine players average out. Individual players differ enormously in how volatile
they are; a nine-player sum does not, because the differences wash out and one
extra correlated pair out of thirty-six barely moves the total. **You cannot
build a safe lineup and a volatile lineup at the same projected total.** There
is only a better and a worse projected total — and then there is the field.

That is a statement about this simulator, not yet about football. What would
falsify it is real outcome variance differing by player type more than the sim
allows, and that is the thing to measure next.

P(≥target) is kept anyway, because it answers a question the projection does
not: the best lineup on week 6 cleared 125 **48.6%** of the time, and on week
8 it is 51%. 125 is roughly a coin flip, which is worth knowing before paying
an entry fee. It is not wired into the API request path, since it costs a
history download to pick the identical lineups.

Calibration is monotone but flat — predicted 0.014 / 0.094 / 0.224 / 0.372 /
0.466 against realised 0.000 / 0.016 / 0.164 / 0.543 / 0.556. One slate, where
all 343 lineups share one set of player outcomes, so the effective n is about
one; the ordering is the only part worth reading.

### The thing that actually moved the number

| build | top-5 actual | hit 125 |
|---|--:|--:|
| 5 lineups, 50% usage, $600 step | 104.8 | 1/5 |
| 25 lineups, 50% usage, $600 step → take 5 | 123.4 | 3/5 |
| 5 lineups, no cap, $100 step | 125.1 | 3/5 |
| 25 lineups, ceiling, 50% usage → take 5 | **139.9** | **5/5** |

A usage cap is a spreading rule and spreading needs lineups to spread over.
Asked for five under a 50% cap, the builder lets each player appear twice and
forces lineups 3–5 down the price ladder — twenty points per lineup, for
nothing. Asking for fewer lineups should not make each one worse, so that case
now warns.

But the headline is the row that was already in production: **build 25 and
play the top five** gets 123.4, and the ceiling objective doing it got 139.9
with all five clearing 125. The existing workflow was not the problem. One
slate, and a 16-point gap is inside a single lineup's noise (sd ≈ 25–31), so
this is a reason to keep measuring and not a reason to change the objective.

## The starter comes back, and the fill-in keeps the job he lost (2026-10-28)

`roles.promotion_multiplier` handles one direction of a role change. Nothing
handled the other: a fill-in's network input row and trailing-5 form are built
from games he played as the starter, so in the weeks after the starter returns
both still describe a job he no longer has. Found on the betting board in 2026
week 8: Josh Jacobs back from suspension, and Pierre Strong, RB3, still carrying
24 rushing yards and 0.22 TDs off two weeks as the lead back. In week 7 he took
8 snaps and no carries.

The term is the promotion's exact mirror and has no parameters. For each game
in the player's form window (his last five, same team only), if a teammate
ranked above him who is playing THIS week did not take an offensive snap, that
game was played at the promoted rank and carries `_lift(position, rank,
absent)`; the projection is divided by the window's mean lift. Window absences
come from snap counts, which is legitimate here — those games are in the past —
and "playing this week" is production's own trigger. Harness:
`experiments/starter_returns.py`, on `promotion_endtoend`'s production-identical
chain, current-season window, 5 seeds, 2025 weeks 5-18 and 2026 weeks 5-7.

### The instrument that lied this time: players with no stat line

The first pass scored only player-weeks with a stats row, and came back null
(-0.15% where it fires, RB and TE reliably worse). But a player who takes snaps
and touches nothing has no stats row — and he is precisely the fill-in this term
is about. Counting the **565** player-weeks with offensive snaps and no stat
line at zero changes the answer:

| population | n | gain | 90% interval |
|---|---|---|---|
| whole board | 5,958 | +0.34% | [+0.21%, +0.47%] |
| where it fires | 1,870 | **+1.73%** | [+1.06%, +2.43%] |
| QB | 51 | **+31.5%** | [+18.9%, +39.7%] |
| WR | 1,045 | +2.2% | [+1.3%, +3.1%] |
| TE | 409 | +1.7% | [+0.2%, +3.5%] |
| RB | 365 | **-2.8%** | [-5.1%, -0.8%] |
| placebo (absences below) | 1,764 | +0.4% | [-0.4%, +1.3%] |

15 of 17 weeks favour it. Restricting to returning starters (rank 1 only) fires
on 1,066 and gives the same picture (+1.65%, QB +31.9%, RB -3.2%). Ordering is
flat either way. Strong's week 7 goes 3.8 → 2.6 against an actual 0.

Every scoring harness in this file inner-joins to the stats table, so every
one of them has been blind to zero-touch weeks. The promotion entry's numbers
are not wrong for that reason — a promoted player is rarely invisible — but
any term that moves projections *down* has been tested against a population
with its best cases removed.

### The same shape as the promotion

A large, clean QB effect, small real gains at WR and TE, and RB the one place it
is reliably harmful — the promotion entry found exactly this, and set
`PROMOTION_FACTOR["RB"]` to 1.0 for it. The rank multipliers that define the
lift (RB 1.0/0.80/0.55) are the suspect quantity at RB in both directions.

**To ship:** the demotion for QB, WR and TE, not RB, in `predict.py` after the
promotion, with the harness re-run on the shipped code so `verify()` covers it.

## Week 8 scored: the field was right, and so was the list we doubted (2026-11-02)

2026 week 8, FanDuel main slate, 11 games. Nine builds, 85 lineups, 80 of them
on the main slate and fully scorable. First week with ownership available
before lock, so the first real test of both things it was supposed to buy.

| | |
|---|--:|
| FanDuel's own FPPG would have built | 69.8 |
| our 80 lineups, mean | **105.2** (sd 24.4) |
| our best | 160.5 |
| the best lineup the slate held | 266.7 |
| beat FPPG | 74 of 80 (92%) |
| cleared the 125 target | 17 of 80 |

A 69.8 FPPG baseline is the lowest of the three weeks measured (116.3, 73.5,
69.8), so 92% is flattering; against the pool it is an ordinary week. We took
60% of what was there, in line with the 69% and 55% before it.

### The leverage premise failed, cleanly

Across the 65 main-slate players the board rated above 8 points and the field
had a number for:

**corr(ownership, actual − projected) = +0.130.** Below-median owned beat their
projection by **−0.78**; above-median by **+2.03**. The chalk was chalk because
it was good, and fading it was fading the better half of the board.

Two builds 26 minutes apart make a near-controlled pair — same objective, same
pool, same vintage, fade the only difference:

| build | fade | mean actual | top five | best |
|---|--:|--:|--:|--:|
| 21 | 0.00 | 110.2 | 109.5 | 141.1 |
| 22 | 0.05 | 105.4 | 114.9 | 142.1 |

Which is noise at a lineup sd of 24. Nothing here says the fade works, and the
one measurement with any power says the opposite. It stays a dial at 0.

The confusing part, and the reason this is not simply "ownership is useless":
the week's two best plays were both lowish-owned. Kirk Cousins at **0.7%**
owned scored 28.2 against an 18.3 projection, in 30 of our lineups. Tony
Pollard at 9.9% scored 30.2 against 15.4. So leverage pays in the tail while
costing in the mean, which is exactly what a tournament trade looks like and
exactly what one slate cannot separate from luck.

### The scratch detector was wrong, both times, and expensively

Logged on 2026-10-28 as the strongest new signal found: a well-projected player
the field has not rostered is usually one the field knows is inactive. Week 8
says otherwise.

| | projected | listed | actual |
|---|--:|--:|--:|
| Cam Ward | 10.7 | no | **29.3** |
| Braelon Allen | 10.3 | no | **25.3** |

Both played, both were among the best scores on the slate, and both were in
**zero** of our 80 lineups — not because anything excluded them, but because
10.5 projected points does not make an optimal lineup. The flag was advisory
and the advice ("check the news, usually he is out") was wrong twice.

Three weeks of it:

| week | unlisted & projected > 10 | played |
|---|--:|--:|
| 6 | 5 | 0 |
| 7 | 4 | 2 |
| 8 | 2 | **2** |
| total | 11 | 4 |

and the four who played went 20.4, 4.6, 29.3, 25.3 — mean 19.9. So week 6's 0
of 5 was small-sample luck, the real rate is nearer 4 of 11, and the misses are
the expensive direction: an unlisted player who plays is by construction
uncontested. The flag is kept, the claim is not. It reads "the field has not
priced this player, find out why" and nothing stronger.

### The one rule that keeps holding

| | week 6 | week 7 | week 8 |
|---|--:|--:|--:|
| ceiling build, top five | 139.9 | 149.7 | 116.8 / 116.0 |
| mean build, top five | 125.1 | — | 95.9 |

Third week running that the ceiling build's best five beats the mean build's,
and week 8 is the widest margin yet (+21). Still the only lineup-level finding
in this file with more than one week behind it, and still unexplained — the
summed ceiling it ranks on is a quantity that cannot happen (see 2026-10-28).
Worth trusting before it is worth understanding.

Biggest misses, by points per lineup given up: Omar Cooper Jr. −3.0 on an 11.0
projection in 30 lineups, Brissett 4.5 on 18.4 in 28, Javonte Williams 6.2 on
17.7 in 30, Zay Flowers 0.0 on 15.4 in 22, Tre Tucker 0.0 on 11.0 in 30. Four
of those five were on the leverage shortlist.

## The 147 the board agreed with and the optimizer would not buy (2026-11-02)

The week 8 lineup actually entered, in a 50/50, scored **147.0** at $59,600:
Prescott 19.8, Tuten 31.9, Jeanty 8.8, Ja'Kobi Lane 1.8, Jefferson 11.8, Chase
28.6, Hockenson 13.8, Stevenson 10.5, Jacksonville 20.0.

It beat **76 of our 80** main-slate lineups. Four of ours beat it. And **eight
of its nine players appeared in none of our eighty lineups** — only Hockenson
overlapped, in 7.

That is not the board disagreeing. On our own board those players ranked:

| | our projection | our rank | actual |
|---|--:|---|--:|
| Prescott | 18.3 | #8 of 75 QB | 19.8 |
| Tuten | 14.3 | #10 of 119 RB | 31.9 |
| Chase | 12.7 | #6 of 205 WR | 28.6 |
| Hockenson | 8.0 | #7 of 131 TE | 13.8 |
| **Jacksonville DEF** | 7.7 | **#3 of 22** | **20.0** |
| Jefferson | 9.6 | #17 WR | 11.8 |
| Stevenson | 10.3 | #18 RB | 10.5 |
| Jeanty | 11.2 | #16 RB | 8.8 |
| Ja'Kobi Lane | 4.2 | #47 WR | 1.8 |

Eight of nine are top-decile on our board. What excluded them was **points per
dollar** — Chase 1.46, Jefferson 1.21, Jacksonville 1.64 — against cheaper
players at 1.8–2.7. The board ranked the week correctly and the value filter
overrode it. Worth a dedicated test: an objective that pays up for the board's
top-ranked players instead of its most efficient ones.

### Pairing a running back with his own defence: right, and for one reason

The stated theory was "if the defence plays well the RB should see more work."
3,982 team-games, 2019-2026, lead scorer per team per game:

| pair | correlation |
|---|--:|
| **RB carries vs own DST** | **+0.177** |
| lead RB points vs own DST | +0.087 |
| QB vs own DST | −0.074 |
| lead WR vs own DST | −0.056 |
| **lead RB vs the OPPOSING DST** | **−0.339** |

Running back is the only position that pairs positively with its own defence,
exactly as claimed, and carries more than points — the mechanism named was the
right one.

| own DST | n | RB points | RB carries | margin |
|---|--:|--:|--:|--:|
| worst 25% | 1037 | 13.68 | 12.36 | −12.28 |
| best 25% | 933 | 15.73 | 15.65 | +13.40 |

**But partial out the scoring margin and it is gone**: RB points vs own DST
falls to −0.240 and carries to −0.036. The whole link is "my team is winning."
That does not make it useless — nobody knows the margin before kickoff, so
RB + own DEF is a cheap way to express a bet on the game going one way, which
is precisely how the Jacksonville side of this lineup was reasoned. It is a
correlated bet, not an independent edge, and should be sized like one.

The bigger number is the mirror image, and it is not a stack rule anyone states:

| opposing DST | n | RB points | RB carries |
|---|--:|--:|--:|
| worst 25% | 1037 | **18.79** | 15.84 |
| best 25% | 933 | **11.22** | 11.96 |

A 7.6-point swing on the defence a back is playing against, against 2.05 for
the defence he shares a sideline with. The `max_per_team` cap cannot see it,
because an RB and the defence facing him are on different teams.

### What this says about our simulator

`pool_sims` draws defences independently, so it models

- RB ↔ own DEF at **−0.006** where reality is +0.087
- RB ↔ opposing DEF at **0** where reality is **−0.339**

So P(≥target) understates the spread of exactly this lineup shape: the pair's
SD should be 13.11 rather than the 12.63 we simulate. Small for one pair, and
the opposing-DEF case is not small at all. Wiring the defence into the team
environment draw — and into the opponent's, with the sign flipped — is the fix,
and it is the first thing found that would make the target objective rank
differently from the mean rather than identically (see 2026-10-28).

## Correction: the ownership page is a vintage, and I scored the flag against the wrong one (2026-11-02)

The entry above retracted the unlisted-player signal on the strength of Cam
Ward (29.3) and Braelon Allen (25.3) — two big scores I had flagged on the
Friday as players the field had not rostered. The retraction was wrong, and the
reason is the same class of mistake as 2026-10-26's projection vintages.

**The roster-percentage page is republished.** Week 8's read "Last updated Fri"
when pulled on 2026-10-30 with 169 players, and "Sun, Nov 1 at 4:24 PM GMT" —
11:24 ET, minutes before lock — with 158. Between those two reads:

| | Friday | at lock | actual |
|---|---|--:|--:|
| Braelon Allen | unlisted | **30.1%** | 25.3 |
| Cam Ward | unlisted | 1.5% | 29.3 |
| Breece Hall | listed | **removed** | did not play |

Breece Hall was ruled out, and the field moved onto his backup. By kickoff Allen
was the third-most-rostered back on the slate. He was never an unlisted player
at the moment that matters; he was an unlisted player three days early.

So the flag did not mispredict. It was handed a snapshot taken before the
weekend's news and judged against outcomes from after it. Scored on the list as
it stood **at lock**, week 8's unlisted-and-projected-above-10 were:

| | projected | actual |
|---|--:|--:|
| Breece Hall | 12.9 | did not play |
| Daniel Jones | 11.2 | 8.5 |
| Tre Tucker | 10.9 | 0.0 |

Three weeks, all judged at lock:

| week | unlisted & projected > 10 | played |
|---|--:|--:|
| 6 | 5 | 0 |
| 7 | 4 | 2 |
| 8 | 3 | 2 |
| total | 12 | 4 |

and the four who played went 20.4, 4.6, 8.5, 0.0 — **mean 8.4** against
projections of 11 to 18, not the 19.9 the previous entry reported. That figure
was inflated by Ward and Allen, who do not belong in the sample. Eight of
twelve did not play at all. The original reading stands: a well-projected
player the field has not rostered *at lock* is usually one it knows something
about, and the exception so far is Cousins.

The leverage result is unaffected — it was computed from the at-lock list
throughout, and corr(ownership, actual − projected) = +0.130 either way, with
below-median-owned at −0.78 and above-median at +2.03. Fading the field still
lost.

### What to actually change

**Pull ownership close to lock, and record when it was pulled.** We stamp
`projection_computed_at` on every build precisely so a stale board cannot hide,
and then read the field from a page with no vintage at all. A Friday pull is
the field's guess before the inactive reports; it is not wrong, it is early,
and nothing in the build says which it was. `_attach_ownership` should store
the page's own "Last updated" and surface it beside the board's.

Two of the week's best scores, Tyler Allgeier (36.6, the top back on the slate)
and Emmett Johnson (25.8), were unlisted even on the final page — so the tail
of genuinely unrostered players who go off is real. It is just not the same set
as the Friday tail.

## Cash and tournament builds, and the dial that cannot be set from two weeks (2026-11-02)

Rebuilt the optimizer around what the last three weeks measured. The useful
part of this entry is which of the intended changes survived contact.

### What the best lineups actually look like

| | week 6 | week 8 |
|---|--:|--:|
| best lineup that existed | 206.1 | 265.7 |
| our median POINT rank at position | 8 | 6 |
| its worst-ranked player | #42 | #46 |
| its projected pts/$1k | 1.54 | 1.75 |
| what our mean build spent per point | 2.08 | 2.13 |

So the best lineup is *less* efficient than ours and its tail is deep-board
flukes. It is not a target, which the earlier entries already said.

### "Build the best, not the most valuable" cannot be done with a dial

Tried it as a continuous tilt, `objective = projection + alpha * salary/1000`,
swept alpha 0 to 3. Projected pts/$1k moved 2.10 to **2.09** and median rank
4.9 to 4.5. Nothing. The reason is structural: **the optimizer already pays
up.** Every lineup it builds spends $59,000-$60,000, so there is no withheld
salary to reward, and it skips an expensive top-ranked player only because the
board says nine cheaper points beat his eight.

The real obstacle is the board, and it is worse than compression. Regressing
actual on projected over 669 player-weeks the slope is **0.909** — the scale is
right, not compressed. But among the 136 players a lineup would actually
consider, projection SD is **3.67** against an outcome SD of **9.40**, and the
rank correlation is **+0.316**. Inside the usable set the board barely
discriminates. No objective over those numbers can find "the best nine", so
the optimizer's job is to avoid self-defeating shapes, not to rank harder.

### What shipped, and the evidence for each

**`rb_vs_opposing_def`** — no running back facing the lineup's own defence.
The largest pairing measured (-0.339; 18.79 points in the worst quartile of
opposing defence against 11.22 in the best). Solved exactly by enumerating the
defence, verified against brute force on 80 small slates. On week 6 **six of
ten** lineups broke it unprompted and enforcing it cost **0.1** projected
points. It is not free in time: 1.5 s a lineup against 0.11 s, because it
solves once per defence, and the bound only prunes 27 solves to 17. Week 8 it
never bound at all and cost nothing.

**`ownership_exempt_rank`** — the fade skips each position's top few, so
leverage is taken in the slots that can carry it instead of by downgrading the
best player on the slate.

**Defence correlation in `pool_sims`** — defences were drawn independently.
Now -0.45 against the opposing offence's total, +0.09/-0.07/-0.06 with their
own RB/QB/WR and -0.34 with the opposing back, all matched to measurement.
**It did not do what I expected.** I had logged this as the first candidate
that would make the target objective rank lineups differently from the mean.
It does not: lineup SD residual is still 0.9% and spearman(P, mean) is still
0.9995, top five overlap 5/5. Nine players average out even with the defence
wired in, so P(target) is redundant with the mean full stop — not for want of
a correlation. The fix is kept because the simulator is now right about
something it was wrong about, not because it unlocked anything.

### And the dial that two weeks cannot set

Cash aggression was going to mean "don't spread, play the best lineup", on week
6's evidence that five lineups under a 50% cap scored 104.8 against 125.1
uncapped. Week 8 says the exact opposite, and says it just as cleanly:

| cash aggression | week 6 | week 8 |
|---|--:|--:|
| 0 (hedge) | 105.6 | **106.9** |
| 50 | 116.2 | 96.5 |
| 100 (commit) | **123.5** | 88.9 |

Monotone in opposite directions. Spreading is not exposure management here, it
is a hedge against **our own error**, and whether the hedge pays depends on
whether the board was right that week — week 6 it was, week 8 its top picks
(Cooper Jr., Brissett, Javonte Williams, Tre Tucker) all failed together.

With a rank correlation of +0.316 inside the usable set, there is no basis for
picking a side, so the dial spans it and says so in the tooltip. That is the
honest answer and it is not the answer I set out to build.

GPP presets at the same aggressions: week 6 gave 127.3 / 121.9 / 98.3 mean
with a best top-five of **143.1** at aggression 50 and 8 of 12 lineups
clearing 125; week 8 gave 110.6 / 106.7 / 91.2. Aggression 100 was worst in
both weeks, which is consistent with the fade losing on the mean.

### The trench report against the board's error

Weeks 3-8, 1,938 player-weeks at their pre-week vintage, against
`/trenches/matchups`:

| position | run_matchup vs miss | pass_matchup vs miss |
|---|--:|--:|
| QB | +0.047 | +0.011 |
| RB | **+0.166** | +0.036 |
| WR | +0.102 | -0.143 |
| TE | -0.144 | -0.077 |

Backs projected above 8 in a "lean up" run matchup beat their projection by
**+5.49** (n=23). That is the most promising of sixteen tests run here, which
is exactly when to distrust it: at n=134 a correlation of +0.166 is around
p=0.05 before any correction for having looked sixteen times. Not wired into
anything. It is mechanically the same story as the RB/defence pairing — both
are rushing volume in a game going one way — so it is worth a proper
prospective test rather than a coefficient.

## Coverage is answered; the trench effect gets a prospective record (2026-11-02)

Two questions, two different outcomes.

### Coverage, tested a fourth way, is still nothing

Per-player it had already failed three ways — team yards allowed, per-corner
grades, and the board's own residual, where wide receiver came back actively
harmful. But the hypothesis actually being proposed was different: not "does
this receiver beat his projection" but **"does a weak secondary flag a game
worth stacking"**, which is a game-level question and had never been run.

87 games, 2026 weeks 3-8. Outcome is the game's aggregate miss — the fantasy
points its rosterable skill players scored minus what the board projected of
them. Significance at this n needs about |r| = 0.21:

| signal | r with game miss |
|---|--:|
| weakest secondary in the game | **−0.125** |
| both secondaries summed | +0.076 |
| best run matchup in the game | +0.072 |
| both run matchups summed | +0.078 |
| best pass matchup | +0.035 |
| both pass matchups summed | +0.000 |

Nothing, and the coverage term has the wrong sign. Checked once more at the
player level through the recording harness: opp_cover_grade against the board's
error is **+0.008** for receivers and **−0.007** for tight ends. That is four
independent tests now. It is context for a human reading a slate and nothing
more.

Cincinnati is the instructive case, because it looked like a hit. CIN was on
the "lean up" list in week 8 — but at **66.1 run** matchup with pass at 47.6,
dead neutral. What happened was Chase 28.6 and Higgins 35.5: a passing
explosion in a game flagged for running. Right game, wrong mechanism, which is
what a 0.07 correlation looks like from the inside.

### The trench effect held to a proper standard

Taken seriously, the running-back finding does not clear the bar:

| check | result |
|---|---|
| full-sample r (RB, projected ≥ 8) | +0.166, n = 134 |
| bootstrap 95% CI | **[−0.002, +0.324]** |
| weeks 3-5 / weeks 6-8 | +0.263 / +0.099 |
| "lean up or better" vs the rest | +5.10 against +0.13 (n = 25) |
| **weeks pointing the right way** | **5 of 5 testable** |

The interval touches zero, the effect halves out of sample, and it was the best
of sixteen tests on one dataset. Against that, 5 of 5 on a sign test is about
p = 0.03, the 58 threshold is the trench report's own band rather than one
fitted here, and it is mechanically the same story as the running-back/defence
pairing — rushing volume in a game going one way. Per week the gaps are +1.65,
+14.04, +1.30, +4.03, +5.34, so one slate is most of the pooled figure and the
robust part is the sign, not the size.

By position, the same test, which is the context that was missing:

| position | gap | weeks right |
|---|--:|--:|
| **RB** | **+4.98** | **5/5** |
| WR | +2.38 | 2/5 |
| QB | −0.27 | 2/6 |
| TE | −0.93 | 2/5 |

Only running back survives. Receiver's pooled gap is one week in disguise.

So nothing adjusts a projection. `nfl_projections.signals` freezes what the
signals said before a slate, `scripts/record_signals.py` runs it Thursday and
Saturday and scores earlier weeks as their stats land, and
**GET /trenches/signals** reads the verdict back with `weeks_right` out of
`weeks` and the per-week detail, so a single big week cannot hide inside a
pooled average. Backfilled over weeks 3-8, which reproduces the table above
exactly.

This is the first thing in this file built to be judged rather than fitted. If
it is real the record will say so by about week 14; if it is the best of
sixteen looks, it will say that instead, and that is the point.

## A good matchup is a tiebreaker, not a reason to climb the board (2026-11-02)

Asked directly: does a well-projected back in a good run matchup become
someone to dial in? Three tests, and they do not all say the same thing.

**The conjunction is real.** Flagged backs beat their projection in both
halves of the projection range:

| | n | projected | actual | miss |
|---|--:|--:|--:|--:|
| projected 8-11, neutral or worse | 58 | 10.22 | 10.60 | +0.38 |
| projected 8-11, **lean up** | 9 | 9.48 | 16.90 | **+7.42** |
| projected 11+, neutral or worse | 51 | 16.27 | 16.10 | −0.16 |
| projected 11+, **lean up** | 16 | 15.77 | 19.57 | **+3.80** |

**And chasing it up the board loses anyway.** Take the best flagged back each
week instead of the best back:

| | |
|---|--:|
| projection given up | **4.14** |
| mean actual, best back by projection | 25.68 |
| mean actual, best flagged back | 21.62 |
| difference | **−4.06** |
| weeks the flagged pick was better | 2 of 5 |

The matchup is worth about 4-7 points of outperformance, and the projection
gap you pay to reach it is 4.14 points on top of the top back ALSO beating his
number. The two cancel, and then some.

**But as a tiebreak it is strong.** Head-to-head pairs of backs inside a
projection window, one flagged and one not:

| window | pairs | flagged wins | mean margin | 95% CI |
|---|--:|--:|--:|---|
| ±1.0 | 59 | 64% | +6.73 | [+3.65, +9.89] |
| ±1.5 | 90 | **70%** | +6.97 | [+4.76, +9.45] |
| ±2.5 | 153 | 67% | +6.67 | [+4.67, +8.77] |

Clear of zero at every window and stable across them. The pairs are not
independent — 25 flagged player-weeks generate all 90 — so the interval is
optimistic, but the win rate is a different statistic from the correlation and
agrees with it.

So: `matchup_boost`, capped by documentation at about 1.5 points, which can
reorder near-ties and cannot promote a worse player past a better one.

### And it does not reach the lineup

| boost | wk 6 actual | wk 8 actual | flagged backs per lineup |
|---|--:|--:|--:|
| 0.0 | 123.7 | 93.7 | 0.9 / 0.0 |
| 0.5 | 121.8 | 93.7 | 1.0 / 0.0 |
| 1.0 | 121.3 | 93.7 | 1.3 / 0.0 |
| 1.5 | 123.7 | 94.5 | 1.7 / 0.1 |
| 3.0 | 114.9 | 107.5 | 2.6 / 1.6 |

Nothing at a defensible size. The reason is worth keeping: the head-to-head
evidence is about backs within 1.5 **projected** points, and the builder picks
on points per dollar under a cap, so its near-ties sit in value space. A
projection-space nudge mostly does not reach them — on week 8 it moved the
flagged-back count from 0.0 to 0.1 a lineup.

The 3.0 row is the only one that moved week 8, and it moved it a long way
(+13.8). That is the chase regime the second test says loses, on one week, so
it is noted and not believed.

Default 0. The effect is real where it was measured, the implementation is
honest about the size, and it has not been shown to improve a lineup. The
prospective record is what settles it.

## Asking the matchup question where the builder decides, and the split that eats it (2026-11-02)

`matchup_boost` added points before the solve and did nothing, because the
builder decides in value space: its near-ties are players with similar points
per dollar, not similar points. So the question was re-asked properly --
`matchup_swaps` looks, after a lineup is solved, at the players who could have
filled each slot within a tolerance in **points per $1,000**, and takes the one
in the best matchup.

**The player-level evidence survives the change of units.** Weeks 6 and 8,
the only two with a stored slate to recover salaries from:

| matched on | window | pairs | flagged wins | margin |
|---|---|--:|--:|--:|
| projection | ±1.5 pts | 27 | 74% | +6.85 |
| **value** | ±0.05 pts/$1k | 11 | 73% | +5.08 |
| **value** | ±0.10 | 16 | 69% | +6.68 |
| **value** | ±0.15 | 23 | 70% | +4.50 |
| **value** | ±0.25 | 44 | 68% | +4.65 |

**And the swap still loses.** Week 6, ten lineups, tolerance 0.15: four swaps,
actual 477.8 down to 425.4.

Which looks damning until the four swaps are read:

```
L7  Chuba Hubbard  -> Kyren Williams   value gap 0.036  projection +2.44
L8  Chuba Hubbard  -> Kyren Williams   value gap 0.036  projection +2.44
L9  D'Andre Swift  -> Kyren Williams   value gap 0.026  projection +2.93
L10 D'Andre Swift  -> Kyren Williams   value gap 0.026  projection +2.93
```

They are **one decision**, taken four times. Every swap was into Kyren
Williams, whose team graded 67.5 on run blocking and who returned 12.4 against
a 16.0 projection. The mechanism is working — the swaps are genuinely
value-tied, and they even raise projected points — and the sample is a single
running back.

### The thing the signal does not know

Los Angeles was flagged because the Rams block well. Whether a Rams back banks
that depends on how the carries divide, and they divided: Kyren 12.4, Corum
13.3, neither paying. The trench report rates a **team**; a lineup slot holds a
**player**. Nothing in the signal carries workload share, so it will keep
pointing at split backfields with the same confidence it points at bellcows.

Worth recording that this was called in advance, from the outside, by someone
reading the same report: *"I really liked the LAR RBs too because of the trench
report, but didn't like the split."* That is the missing term stated plainly,
and it is the next thing to test — the matchup signal conditioned on carry
share, rather than the matchup signal alone.

So `matchup_swaps` ships off by default. `signals` now records salary and
value alongside the projection, which is what makes the value-space question
answerable on a real sample instead of on one week and one running back.

## Correction: it was not the split, and dollars per opportunity does not beat the board (2026-11-02)

The entry above explained the failed matchup swap as a move "from a bellcow to
a split backfield" — the trench report rating a team while a lineup slot holds
a player. It read well and it is **wrong**. The volume says so plainly:

| week 6 | team | salary | expected opp/game | projected | actual | $/opp |
|---|---|--:|--:|--:|--:|--:|
| Chuba Hubbard | CAR | 6,800 | 18.0 | 14.3 | 21.6 | 378 |
| D'Andre Swift | CHI | 6,600 | 18.8 | 13.8 | 21.0 | 352 |
| **Kyren Williams** | LAR | 8,100 | **22.2** | 16.7 | **8.2** | 364 |

The swap moved **up** in opportunity, not down. Kyren was the Rams' lead back
by trailing volume — 22.2 a game against Corum's 9.8, a 69/31 split — and
carried a similar price per opportunity to the men he replaced. He simply had a
bad game. The swap lost to efficiency variance, which is the one thing nothing
in this repo has ever been able to forecast, and I attributed it to a structural
story the data does not support.

### And the proposal it prompted tests as a null

"Load up on players with the most opportunities; look at dollars per
opportunity." The premise is right and already on file: opportunity repeats at
0.974 across odd and even weeks while touchdowns per opportunity manage 0.178.
But the October entry could not test the price half, because no historical
FanDuel salaries existed anywhere. Weeks 6 and 8 now have stored slates, so for
the first time there is a price to compare against. 91 rosterable skill
player-weeks:

| selection metric | r with actual | spearman |
|---|--:|--:|
| **projected points** | **0.315** | **0.297** |
| salary | 0.294 | 0.330 |
| expected opportunities | 0.243 | 0.192 |
| projected points / $1k | 0.234 | 0.117 |
| opportunities / $1k | 0.164 | 0.113 |

The board's projection is the better selector, raw and per dollar. And against
the board's own error, volume at a price is nothing: expected opportunities
+0.015, opportunities per $1k +0.015, dollars per opportunity +0.029. Which is
the October result, now confirmed on the half of the question that could not be
asked then.

### Why volume cannot do the job it is being asked to do

The twelve highest-volume backs across the two weeks averaged 23.6 opportunities
a game, were projected 15.6, and scored 15.9 — with a standard deviation of
**8.9** and a range of **4.5 to 29.9**:

| | expected opp/game | projected | actual |
|---|--:|--:|--:|
| Gibbs (wk 8) | 29.0 | 22.4 | 26.8 |
| Walker (wk 6) | 26.5 | 20.6 | 29.9 |
| Henry (wk 6) | 24.2 | 17.6 | **4.5** |
| Bijan (wk 6) | 24.0 | 18.3 | **6.3** |
| Kyren (wk 6) | 22.2 | 16.7 | **8.2** |

Walker, Bijan and Kyren had near-identical volume and scored 29.9, 6.3 and 8.2.
Volume buys more chances at a touchdown, exactly as argued — and a touchdown
per opportunity is a coin flip, so more chances does not narrow the outcome
enough to beat a projection that already counts the chances. The board's mean
is the right summary of "volume times whatever efficiency we can forecast", and
re-ranking on one of its two inputs discards the other.

Both samples here are small — 91 rosterable player-weeks, 2 slates — and the
only reason the price question is answerable at all is the two stored slates.
`signals` now records salary and value every week, so this gets re-asked on a
real sample rather than re-argued.

## The second-half fade for older backs is real, mostly priced, and Henry is the case (2026-11-02)

Prompted by a read from outside the data: Henry "started the season hot, but
has cooled off ... typically doesn't deliver as many points in the second half
once he's gotten a little tired." Distinct from the age curve already on file,
which was about games PLAYED and did not survive out of sample. 427
running-back seasons with five or more games in each half, 2016-2026:

| age | n | 1st half pts/g | 2nd half | Δ points | Δ opportunities | Δ pts/opp |
|---|--:|--:|--:|--:|--:|--:|
| ≤24 | 185 | 11.76 | 12.29 | **+0.53** | +0.39 | +0.02 |
| 25-26 | 122 | 12.18 | 11.31 | −0.87 | −0.51 | −0.03 |
| 27-28 | 65 | 12.31 | 11.27 | −1.04 | −0.94 | −0.02 |
| 29-30 | 32 | 11.50 | 10.35 | −1.15 | −1.39 | −0.02 |
| 31+ | 23 | 9.57 | 8.23 | **−1.34** | −1.09 | −0.05 |

Monotone, and **it is opportunity leaving rather than efficiency falling**:
age correlates −0.151 with the change in opportunities and only −0.073 with
the change in points per opportunity. Workload sorts it harder than age does —
by first-half total opportunities, light +0.61, medium −0.07, **heavy −1.57**,
again almost all of it volume (−1.56).

Henry's own seasons flip exactly where claimed. Ages 22-26: +3.90, +0.89,
+12.05, +13.02, +4.85. Ages 28-30: **−5.09, −1.07, −5.82**. Age 31 bucked it at
+6.56. And 2026 to date is a textbook run:

| week | opportunities | points | trailing projection |
|---|--:|--:|--:|
| 3-5 | 27, 25, 26 | 21.4, 14.9, 21.5 | 27.0, 25.1, 22.6 |
| 6 | 18 | **4.5** | 18.5 |
| 7 | 19 | **5.4** | 15.6 |
| 8 | **13** | 13.0 | 11.6 |

### But the trend is already in the projection

The obvious exploit — fade a back whose opportunities are trending down — is
nothing. 8,804 player-weeks, trend being the last two games against the three
before them, against a trailing projection's error:

| position | n | r(trend, error) |
|---|--:|--:|
| RB | 5,500 | **−0.010** |
| WR | 2,872 | −0.077 |
| TE | 432 | −0.095 |

And by bucket for backs the error is −0.56 / −0.25 / −0.59 / −0.13 / −0.89
from falling-hard to rising-hard: no pattern at all, with flat worse than
falling. The reason is mechanical and is visible in Henry's own line above: a
falling trend drags the trailing average down with it, so by the time anyone
can see the decline the number has already moved. Falling-hard backs carry a
trailing projection of 9.96 against rising-hard's 14.42. The projection chases
the fade, and week 8 Henry duly beat his reduced number, 13.0 against 11.6.

### What is left over

The conjunction, which does not fully price:

| second half | n | trailing | actual | error |
|---|--:|--:|--:|--:|
| age 28+ **and** heavy first-half workload | 198 | 9.24 | 8.59 | **−1.45** |
| everyone else | 3,546 | — | — | −0.49 |

About a point of differential. Real, mechanically sensible, large-sample — and
a point against a back's weekly standard deviation of nine, which is why it is
recorded rather than applied. `signals` now carries age and season-to-date
opportunities so the rest of this season tests it.

The broader observation that prompted this — those backs "would have been
decent plays, some just didn't hit ... game script, tough matchup, weather" —
is the correct summary of everything in this file. Walker, Bijan and Kyren took
near-identical volume in week 6 to 29.9, 6.3 and 8.2 points. Nothing measured
here separates those three beforehand, and the honest position is that the
separation mostly does not exist rather than that it is waiting in a feature.

---

## Boom/bust is real, it is volume, and it was mis-simulated

Prompted by: "we don't really take into account boom or bust players either.
Tre Tucker is a good example, Jameson Williams ... their projections do get
skewed because they can have those big weeks which average out with the bust
weeks."

Correct on every count except the mechanism. 32,916 player-weeks, 2019–2026.

### Dispersion is a real, forecastable trait

A player's coefficient of variation persists odd-week to even-week inside a
season, and year over year:

| position | n | r (odd vs even, same season) | n | r (year over year) |
|---|--:|--:|--:|--:|
| QB | 217 | **+0.456** | 132 | +0.199 |
| RB | 474 | +0.348 | 272 | **+0.468** |
| WR | 668 | +0.178 | 381 | +0.242 |
| TE | 259 | +0.067 | 146 | +0.125 |

(Pooled "all positions" numbers are inflated by the QB/skill CV gap — the
within-position rows are the honest ones. TE is noise.)

### The driver is volume, not the deep-threat archetype

Holding the projection fixed, partial correlation of a player's SD with:

| position | n | opportunities/game | aDOT |
|---|--:|--:|--:|
| RB | 477 | **−0.342** | −0.048 |
| WR | 674 | **−0.270** | +0.141 |
| TE | 265 | −0.128 | +0.072 |

aDOT is close to nothing. **More opportunities is not just more points, it is
steadier points** — the same thesis as the opportunity work above, in a second
dimension. And the skew claim is exactly right, graded by volume:

| | median/mean | CV | p90/mean |
|---|--:|--:|--:|
| RB low volume | **0.752** | 0.872 | 2.02 |
| RB high volume | 0.923 | 0.551 | 1.66 |
| WR low volume | **0.787** | 0.820 | 1.95 |
| WR high volume | 0.878 | 0.620 | 1.76 |

A low-volume back's *typical* week is 75% of his projection. A workhorse's is
92%. A gamma with the matching CV reproduces both the median/mean and the
p90/mean to within ~0.02, so the existing machinery was right and only the
width was wrong.

The named players split, and the sample is 7 games each, so this is
illustrative only: Tucker CV 0.73 but **median/mean 0.49** — the archetype
exactly. Jameson Williams CV **0.55**, median/mean 1.16 — steadier in 2026 than
Chase (0.62) or Higgins (0.63). He has the reputation without the profile.

### The simulator was over-dispersed, worst where it hurts

`project_distributions` already gave players different shapes — the flat
`_SKILL_CV` was only the no-history fallback — and it tracked the volume
direction. But it was systematically too wide, and the error grew with volume:

| | sim CV | real CV | ratio |
|---|--:|--:|--:|
| RB low volume | 0.955 | 0.916 | 1.04 |
| RB high volume | 0.774 | 0.550 | **1.41** |
| WR high volume | 0.763 | 0.629 | 1.21 |
| QB | 0.647 | 0.492 | 1.32 |

So the system understated how safe a workhorse is — precisely the player a cash
lineup is built on. Fitted `log(real_cv/sim_cv) = a + b·log1p(opportunities)`
on 1,318 player-seasons (2019–2023), validated on 567 (2024–2026): mean |CV
error| falls for all four positions (RB 0.199→0.130, QB 0.197→0.111) and the
ratio lands at 0.96–1.01 in every position and volume tier. **Kept**, as
`simulate.spread_factor` + `calibrate_spread=True`. It rescales deviations
around each row's own mean, so no row's mean moves (verified to 3.6e-15) and
the board is untouched.

`_SKILL_CV`'s one global 0.75 also became per-position (QB 0.46, RB 0.72,
WR 0.72, TE 0.71) for the no-history fallback.

### Negative result: it does not change which lineup to build

The hypothesis was that a flat CV is *why* `p_target` ranked identically to
`mean` (#earlier, Spearman 0.9986). **Wrong.** With calibrated per-player
spread, Spearman(mean, P(125)) is 0.9959 over 4,000 lineups, and among the top
decile by projection the max-mean lineup *is* the max-P(125) lineup, giving up
0.00 projected points. Nine players still average out. That is now the fifth
independent confirmation, and the strongest, because the obvious confound is
gone.

### What it does change: the target curve

Lineup SD **26.81 → 20.29**, a 24% narrowing. Probabilities pivot around the
target instead of regressing to a coin flip:

| projected | P(125) raw | P(125) calibrated |
|---|--:|--:|
| 110 | 0.275 | 0.224 |
| 125 | 0.458 | 0.457 |
| 140 | 0.639 | **0.705** |

| clear 125 at | raw needs | calibrated needs |
|---|--:|--:|
| 50% | 127.6 | 126.7 |
| 60% | 135.7 | 132.7 |
| 70% | 147.2 | **138.4** |
| 80% | — | 147.2 |

The 125 goal is less of a coin flip than the earlier backtest implied; we were
24% too pessimistic about lineup variance. These figures are from a synthetic
board (season mean as projection) over random lineups — the *spread* is the
validated part; re-derive the curve on real slates before quoting it.

### Negative result: matchup does not rescue the volatile player

The companion hypothesis — a plus matchup is where you find the edge on a
boom/bust guy — fails. 18,359 player-weeks, leakage-safe, opponent's trailing
points allowed to the position, league-relative:

| position | volume tier | n | r(soft opp, residual) | r(soft opp, boom) |
|---|---|--:|--:|--:|
| RB | low | 2,050 | +0.069 | +0.047 |
| RB | high | 2,037 | **+0.107** | **+0.092** |
| WR | low | 3,133 | +0.020 | +0.015 |
| WR | high | 2,899 | +0.036 | +0.033 |

WR is null for a fifth time. RB carries real signal but it is **strongest on
the high-volume backs, weakest on the volatile ones** — the opposite of the
hypothesis. For low-volume players across all positions, going from the
toughest opponent quintile to the softest moves boom rate 24.6% → 29.1% and the
mean +0.66 points, while bust rate does not move at all (30.2% → 30.5%). A
soft matchup buys a little more upside and buys nothing on the downside, which
makes it a tiebreak, exactly like the trench finding.

---

## Wind is the one unpriced situational factor worth buying data for

Prompted by: "is the read on this weeks situation correct, is the opposing
defense really good vs the run ... or weather with Bijan Robinson and the game
with TB."

13,304 outdoor player-weeks, 2019–2026, against a trailing baseline. Wind is a
**threshold, not a slope** — which is exactly why it reads as nothing to a
linear fit and as a lot to a band:

| mph | 0–5 | 5–10 | 10–15 | **15+** | r(wind, error) |
|---|--:|--:|--:|--:|--:|
| QB | +0.23 | +0.31 | −0.37 | **−2.14** | −0.087 |
| WR | −0.10 | −0.16 | −0.43 | **−1.37** | −0.051 |
| TE | +0.33 | +0.50 | +0.15 | −0.36 | −0.065 |
| RB | +0.28 | +0.25 | 0.00 | **+0.01** | −0.006 |

The backs are completely untouched, which is the sensible shape — a team in the
wind runs more, so the volume moves toward them. A QB plus three receivers in a
15+ mph game is about six points of overstatement, which is larger than
anything else measured in this file.

Cold is weaker and probably confounded with wind and late season: QB −1.06 at
32–50°F (n=635), freezing −0.37 (n=114). Not separated out.

Roof is worth a third of a point (dome/closed vs outdoors: QB +0.10/+1.21 vs
−0.23, WR +0.18/+0.30 vs −0.35) and is complete and known in advance.

### But we cannot get the data

nflverse populates `temp`/`wind` only for games it has backfilled — **45 of 272
in 2026, none past week 4** — and never ahead of kickoff. So the measured
effect is real and unpriced and we have no feed for it. `explain.wind_adjustment`
carries the measured penalties and returns 0.0 without a reading, so it is
inert until someone supplies a forecast. **This is the one place where buying a
weather source has a measured payoff.** `roof` is usable today and is reported.

## A vetting layer, explicitly outside the objective

`explain.py`: per-player provenance for a built lineup — where the projection
sits against trailing form, how many opportunities it assumes and what share of
the team's, the predicted volatility, the **typical week** (the median behind
the mean, which is 75–79% of the projection for a low-volume player and 88–92%
for a workhorse), the upside, the opponent front's run band, roof/wind, and a
short `why` string.

Context only, never an input to the objective — the same boundary `opportunity`
observes, and for the same measured reason: the opponent run grade is a
tiebreak between comparable backs (r +0.166), coverage is null four ways, and a
soft opponent moves a low-volume player's boom rate 24.6% → 29.1% while moving
his bust rate not at all. Letting any of that move the board would be fitting
noise. Wind is the exception on the evidence and is still inert for want of a
feed.

---

## Backtest: one cash lineup a week against the 125 target

Leakage-safe, reproducing production's path: the model is trained on 2026 games
strictly before the target week (5 seeds, 60 epochs), projections come off each
player's latest row BEFORE the week, defences from `defense.project`, and the
build is the shipped cash preset at `num_lineups=1`.

Only two weeks are testable. Week 5's saved file is a showdown slate (TB@DAL,
51 rows, `MVP 1.5x Salary`), week 7's was never saved, and week 8's lived only
in the production database.

| week | projected | actual | cleared 125 |
|---|--:|--:|---|
| 4 | 137.3 | **143.3** | yes |
| 6 | 124.7 | 103.4 | no |

**1 of 2**, mean actual 123.4. Small, but the pattern is the one the calibrated
target curve predicts rather than a contradiction of it: the week that
projected 137 cleared and the week that projected 125 did not, and the curve
says 125 projected is a coin flip while ~138 is the 70% mark.

### Scoring bug found and fixed

The first run scored week 4 at 8.0. nflverse `player_name` is the abbreviated
form (`A.Rodgers`) and only the defence matched; the join needs
`player_display_name`. The lineups themselves were unaffected.

### What the vetting layer says about the week that missed

Running `explain.vet` over the same two lineups after the fact, with the
opportunity context, the trench grades and the schedule attached:

| week 6 pick | opp/game | opponent front | actual |
|---|--:|---|--:|
| Kenneth Walker III | 26.5 | soft (35.6) | 29.9 |
| Jaylen Warren | 19.2 | **stout (60.6)** | 8.8 |
| Chuba Hubbard | 18.0 | **stout (56.7)** | 21.6 |
| Denzel Boston (WR) | **5.8** | — | 2.0 |
| Juwan Johnson (TE) | **6.4** | — | 0.0 |

Two of the three backs were run into stout fronts, and both of the
sub-7-opportunity pass catchers returned nothing. In week 4, which cleared,
Bijan Robinson had 27.3 opportunities against a very soft front (20.4) indoors
and went for 30.2.

That is one week each way and proves nothing on its own. It is recorded as the
first prospective test of whether the vetting columns would have changed a
submission, not as evidence that they do. The run band is reported for backs
only — the receiving equivalent is coverage, which is measured null four ways.

---

## Fantasy points allowed, adjusted for who the defence has faced

Prompted by: "If a team lets up a lot of points to the TE and they have faced
lower ranking TEs then that is a team to target, where a team that allows a lot
of points to TE because they've faced 4 of the best TEs than that's not
necessarily an advantage."

Correct, and the FPA table that shipped (NFL-API `app/api/fpa.py`, ported from
NickKnows) cannot tell those two cases apart: it sums opposing PPR points per
(defence, week, position) and averages the weeks, with no schedule adjustment
and in PPR rather than FanDuel scoring.

`fpa.py` measures every opposing player against **his own entering baseline**
instead, so a hard schedule nets out. 21,998 player-weeks with a baseline,
2019–2026, leakage-safe throughout.

### The two rankings disagree, and the adjusted one is better

| | Spearman(raw, adjusted) | r(raw, next delta) | r(adjusted, next delta) |
|---|--:|--:|--:|
| QB | 0.758 | 0.068 | **0.094** |
| RB | 0.755 | **0.089** | 0.082 |
| WR | 0.602 | 0.026 | **0.051** |
| TE | 0.755 | 0.037 | **0.049** |

Roughly double the signal for QB, WR and TE; RB is a wash, which fits — the
backs already have the trench report. Restricted to ≥4 prior weeks the adjusted
numbers rise to QB 0.102, RB 0.115, WR 0.075, TE 0.049.

**This table is superseded — see "Three reports of the same defence" below.**
It is in-sample over the whole pool, and the raw arm was scored on a different
transform from the adjusted arm. Matched and held out, the two are a wash.

**These are still small.** 0.05–0.10 is a tiebreak between comparable players
and a sanity check on a build, not a signal to chase, and it earns the same
standing as the trench and coverage work: context, never an input to the
objective.

### What it changes in practice (2026, entering week 9)

| | allowed | faced | adjusted | grade | |
|---|--:|--:|--:|--:|---|
| PHI vs WR | 29.7 | **32.4** | −1.25 | 28.8 | raw calls them soft; they are very tough |
| IND vs WR | 16.3 | 24.3 | −1.37 | 26.6 | very tough |
| LV vs WR | **36.2** | 20.3 | +1.92 | 84.0 | genuinely very soft |
| CIN vs TE | 10.8 | **7.3** | +1.47 | 76.0 | very soft |
| MIA vs TE | **16.2** | 12.4 | +1.19 | 71.2 | soft, but less than raw implies |

### Outliers are clipped, and the median is the wrong fix

Prompted by: "Atlanta will have skewed numbers there ... TB didn't complete a
pass and still won. It was really rainy and they ran."

Exactly right, and the data is stark — 2026 week 8, Tampa Bay at Atlanta:

| position | players | points | pass yds | rush yds |
|---|--:|--:|--:|--:|
| QB | 1 | 1.4 | **0** | 14 |
| RB | 2 | 41.0 | 0 | **310** |

WR 0.0 and TE 0.0. That one game moves Atlanta's season WR number from 28.9 to
25.3 and their ranking with it, on weather and game script rather than coverage.

**That last sentence is wrong, and the correction is below** ("three corrections
the live run forced"). nflverse recorded no Tampa Bay receiver or tight end at
all for that game, so there is no 0.0 to average in and Atlanta's WR number is
untouched by it. The 28.9 → 25.3 figure came from a prototype that scored an
absent position as zero.

But taking the median over-corrects. Four estimators against next week's delta
(≥4 prior weeks):

| | mean | median | trimmed | winsorised |
|---|--:|--:|--:|--:|
| QB | **0.102** | 0.093 | 0.100 | 0.098 |
| RB | 0.110 | 0.105 | 0.111 | **0.115** |
| WR | 0.072 | 0.055 | 0.068 | **0.075** |
| TE | 0.049 | **0.060** | 0.044 | 0.043 |

The median is the **worst** of the four for receivers — a defence that genuinely
gets gashed carries real information, and dropping the tail throws it away.
Winsorising at ±12 points keeps the direction and caps the magnitude, and it is
best or near-best for RB and WR. Shipped as `WINSOR = 12.0`.

`ratings` also reports `worst_week` / `worst_week_swing`: drop each week in turn
and keep the biggest move, flagged past ±1.0. On the shipped module Atlanta's WR
row flags week 5 (+1.95) rather than week 8, for the reason above — and their RB
row grades **very soft** (76.2; 30.4 allowed against 20.6 faced), which is the
real Atlanta story the Tampa Bay game was hiding.

### Honest note on the week 6 lineup

With both matchup sides attached, the vetting layer flags Juwan Johnson cleanly
(tough vs TEs, grade 41, and 6.4 opportunities a game — he scored 0.0) and both
backs run into stout fronts. But **all three receivers in that lineup graded
"very soft" and all three busted** (2.0, 9.7, 7.9). That is what an r of 0.07
looks like from the inside, and it is the reason this is a tiebreak and not a
lean.

### Shipped to the endpoint, and three corrections the live run forced

`/fpa/` on NFL-API served the raw PPR table; it now serves both halves. The
legacy `qb`/`rb`/`wr`/`te` keys are untouched, because they are the historical
series and the NickKnows chart is built on them, and each row gains
`{pos}_allowed`, `{pos}_faced`, `{pos}_adjusted`, `{pos}_grade`, `{pos}_band`,
`{pos}_weeks` and `{pos}_outlier_week`. `through_week` gives the ratings as they
stood entering a week. Early in a season the adjusted half is legitimately empty
— no player has three prior games yet — so the response carries `adjusted: false`
and a sentence saying why, rather than failing and taking the chart down with it.

Shipping it turned up a **bug in `fpa._points`**: `weekly.get("rushing_fumbles_lost", 0)`
hands back a bare int when the column is absent, and an int has no `.fillna`, so
any frame omitting it raised `AttributeError` instead of scoring. The research
frames all carried a precomputed `fd`, which is why no test had reached it.
Fixed with a `_num` helper and four regression tests.

Then the endpoint was run over the real 2026 season (1,968 player-weeks, weeks
1-8, through a throwaway SQLite copy of `player_stats` and `schedules`). Three
numbers reported earlier came from the research prototype `fpa_adj.py`, which
has its own grouping, and the shipped module disagrees slightly:

| | prototype | shipped module |
|---|---|---|
| PHI vs WR | raw rank 1 → adj 24, grade 28.8 | raw rank 2 → adj 31, grade 30.3 |
| LV vs WR | grade 84.0 | grade 73.7 (still very soft, adj rank 4) |
| ATL vs RB | grade 74.6 | grade 76.2 |

The direction and the conclusions are the same; the module's numbers are the
ones that ship.

**The Atlanta worry is smaller than it looked, for a reason worth recording.**
nflverse has exactly three Tampa Bay rows for that week — Mayfield 1.4, Gainwell
17.0, Irving 24.0. No receiver or tight end recorded a line at all, because
there were no completions to record. So the WR 0.0 and TE 0.0 never enter the
rating: there is no player-week to average, and the feared distortion is
self-limiting. What week 8 does do is show up on the RB side, where Gainwell's
+12.29 gap is the one row the ±12 winsor actually bites on.

And Atlanta's "very tough vs QBs" (grade 23.8, 11.1 allowed against 21.1 faced)
is **not** the Tampa Bay game. Their three rated quarterback weeks are −11.19,
−8.31 and −10.54; drop week 8 and the other two still average −9.75. The
leave-one-out check says as much on its own by flagging week 7, the mildest of
the three, rather than week 8. The same check now puts Atlanta's biggest WR
swing at week 5 (+1.95), not week 8.

The thing to keep in view is sample size, not outliers: entering week 9 Atlanta
has three rated QB weeks and four WR weeks, because weeks 1-3 cannot produce a
baseline. A grade resting on three weeks is a tiebreak, which is all it was ever
claimed to be.

## Three reports of the same defence: do trenches, coverage and FPA agree?

The question, asked directly: the trench report, the coverage grades and the
adjusted FPA table all claim to describe how hard a defence is to score on, from
three unrelated sources — play-by-play line yards and stuff rate, PFR per-corner
charting, and what opposing players did against their own baselines. They ought
to correlate. Do they, and does the third add anything to the first two?

### Season level: yes, and it is unit-specific where it should be

256 team-seasons, 2019-2026. Spearman **within** each season so a league-wide
drift cannot manufacture anything, everything flipped to the "tough" direction
so positive means agreement. Partial columns hold the other two reports fixed.

| | raw ρ | partial ρ | 95% CI |
|---|--:|--:|---|
| `run_stop` ↔ FPA-RB | +0.377 | **+0.341** | [+0.216, +0.447] |
| `cover_grade` ↔ FPA-QB | +0.162 | **+0.249** | [+0.131, +0.369] |
| `cover_grade` ↔ FPA-WR | +0.156 | +0.146 | [+0.016, +0.268] |
| `cover_grade` ↔ FPA-TE | +0.137 | +0.117 | [−0.007, +0.250] |
| `pass_rush` ↔ FPA-QB | +0.162 | +0.085 | [−0.035, +0.212] |
| `pass_rush` ↔ FPA-WR | +0.075 | +0.018 | [−0.110, +0.140] |

Three things fall out. The **run side is the real agreement** — +0.34 after the
pass side is held fixed, and consistent in all eight seasons (+0.26 to +0.59,
never near zero). **Coverage shows up in the quarterback rather than the
receiver** (+0.249 against +0.146), which fits: a quarterback's line aggregates
the whole passing game, which is what a team coverage grade measures, while a
receiver's week depends on which corner he drew. And **pass rush explains
nothing once coverage is held fixed** — the raw +0.162 against quarterbacks was
coverage leaking through, the two correlating +0.247.

The controls earned their place. `pass_rush` ↔ FPA-RB came back +0.299 raw and
`cover_grade` ↔ FPA-RB +0.181, as strong as pairs that were predicted, because
RB is the position most explained by a general "this defence is good" factor:
17.4% of its rank variance, against 5.5% QB, 2.1% WR, 1.3% TE. Without the
controls this would have read as three reports each confirming its own unit.

They are nowhere near redundant, though. All three reports together explain
20.7% of FPA-RB, 8.5% of FPA-QB, 2.7% of FPA-WR and 2.4% of FPA-TE.

### Negative result: week to week, FPA adds nothing over the two reports

20,701 player-weeks, every rating as it stood entering the week, outcome = the
player's FanDuel points minus his own entering baseline. Fitted on 2019-2023,
scored on 2024-2026, 95% bootstrap intervals.

| | report alone | FPA alone | FPA given the reports | report given FPA |
|---|--:|--:|--:|--:|
| QB (coverage) | +0.075 [+0.013, +0.138] | +0.058 [−0.006, +0.122] | +0.024 [−0.043, +0.086] | +0.066 [+0.002, +0.129] |
| RB (run front) | +0.052 [+0.006, +0.098] | +0.054 [+0.009, +0.099] | +0.022 [−0.027, +0.070] | +0.036 [−0.012, +0.085] |
| WR (coverage) | +0.015 [−0.023, +0.053] | +0.009 [−0.028, +0.048] | +0.007 [−0.031, +0.045] | +0.014 [−0.023, +0.052] |
| TE (coverage) | +0.042 [−0.018, +0.102] | +0.018 [−0.043, +0.079] | −0.008 [−0.069, +0.050] | +0.044 [−0.018, +0.105] |

**No FPA increment survives its own interval at any position.** The reports hold
up marginally better in the other direction. Nothing here is big enough to lean
on; this is the sixth independent confirmation that a matchup read is a tiebreak.

### Correction: the adjustment is a wash against raw FPA, not double

An earlier section of this file claims the adjustment roughly doubles the
signal. **It does not.** That comparison was in-sample over the whole pool, and
worse, it scored a percentile rank against a clipped z-score grade. Matched —
every arm a percentile rank within (season, week, position) — against the same
outcome:

| | train raw | train adj | **test raw** | **test adj** |
|---|--:|--:|--:|--:|
| QB | +0.073 | +0.064 | **+0.049** | **+0.061** |
| RB | +0.084 | +0.085 | **+0.056** | **+0.043** |
| WR | +0.021 | +0.030 | **−0.025** | **+0.010** |
| TE | +0.045 | +0.052 | **+0.052** | **+0.020** |

Adjusted is ahead for receivers in both eras and raw is ahead for tight ends out
of sample; quarterbacks and backs swap places between the halves and every
interval overlaps. A wash. The grading step is neutral — "the gap, ungraded"
scores within 0.001 of the shipped grade everywhere.

A first, unmatched pass suggested raw *beat* adjusted everywhere, and that was
the scaling artifact, caught by rerunning on matched transforms before anything
was changed in response to it.

The hypothesis that raw carries game-environment information the adjustment
throws away is **not supported**: the schedule term on its own predicts +0.036
for backs in training, −0.050 for receivers out of sample and +0.070 for tight
ends out of sample — signs that do not agree across positions or eras, which is
what noise looks like.

### What this changes

Nothing in the code, and both numbers stay served. The adjustment is kept not
because it is more accurate — it is not — but because it is the only one of the
two that can be *read*: `allowed` next to `expected` answers "soft defence or
hard schedule", and raw FPA cannot answer that at all. The claims in
`nfl_projections/fpa.py` and `NFL-API/app/api/fpa.py` were rewritten to say so.

## Which lineup objective actually won: the record said floor, the record was wrong

The optimizer's performance endpoint ranks objectives by the mean of every
lineup they produced, and on 2026-11-03 it reported a clean answer across four
scored weeks: **best `floor`, worst `ceiling`, spread 12.0 points**, with
`one_week_only: false`. Two things are wrong with that, and the second one
survives fixing the first.

### Floor appeared in one week

`floor` was built in week 7 and nowhere else. Week 7 is also the week whose
FPPG baseline came in at 73.52 — low enough that *every* objective beat it, at
rates of 0.80 to 0.93. The flag that exists to stop exactly this reading,
`one_week_only`, was computed as `len(scored) <= 1` over the **season's** scored
weeks, not over the weeks belonging to the objectives being ranked. Four scored
weeks made it `false` while the winner stood on one.

Fixed in `NFL-API/app/api/optimizer.py`: `_perf_bucket` now carries the set of
weeks in which that bucket actually scored a lineup, `_perf_add` adds a week
only when the build contributed a *complete* lineup (an unscored build buys no
week), `_perf_summary` exposes `weeks_scored` per row, and the verdict reports
`best_weeks` / `worst_weeks` and sets `one_week_only` when either ranked end
rests on a single week. Two tests cover it: a winner seen in one week of two is
still a one-week verdict, and both ends on two weeks is not. The live record
now flags itself.

### The deeper confound: a build's mean measures its depth

A build of 25 lineups reaches further down the pool than a build of 5 — lineup
25 is by construction worse than lineup 1 — so pooling builds of unequal size
measures how many lineups were asked for as much as which objective asked.
Week 7 pooled 110 ceiling lineups against 30 mean lineups.

Matching builds on (week, lineups requested), classic only:

| week | n | ranking at that depth |
|---|--:|---|
| 6 | 12 | mean 107.67, ceiling 107.65 |
| 7 | 5 | mean 110.72, ceiling 107.71 (x3), floor 92.86 |
| 7 | 25 | floor 104.70, median 96.67, mean 93.48, ceiling 78.54 (x3) |
| 8 | 5 | mean 114.78 (x2), ceiling 98.54 |
| 8 | 15 | mean 102.09, ceiling 100.71 |

Head-to-head at matched depth, positive meaning the first is ahead:

| pair | cells | gaps | mean |
|---|--:|---|--:|
| ceiling vs mean | 5 | −0.02, −3.01, −14.94, −16.24, −1.38 | **−7.12** |
| ceiling vs floor | 2 | +14.85, −26.15 | −5.65 |
| floor vs mean | 2 | −17.86, +11.22 | −3.32 |
| ceiling vs median | 1 | −18.13 | −18.13 |
| floor vs median | 1 | +8.02 | +8.02 |
| mean vs median | 1 | −3.19 | −3.19 |

**`ceiling` is behind `mean` in 5 of 5 matched cells across three weeks**, by
7.12 points on average. That is the only comparison here with a consistent
sign, and it is the opposite of how the board is mostly being built: 14 of 25
builds and 182 of 344 scored lineups are ceiling.

`floor`'s two cells disagree (+14.85 and −26.15); it won week 7 at n=25 and was
last at n=5 in the same week. The endpoint's "floor is best" has no support
once depth is matched, and neither does any verdict about median on one cell.

### What stops this being established

The spread between same-objective, same-depth builds is larger than the gap
being measured. Week 7's four ceiling builds at n=25 scored 95.44, 71.98 and
68.21 — a 27-point range at identical settings-as-exposed, which means
something not in the payload (pool composition at build time, since
`ownership_weight` and `stack_qb` are null for every week-7 build) moves a
build further than the objective does. Five cells with one sign is suggestive;
it is not a result.

Also found while matching: **builds 10 and 16 are the same numbers** (n=5, mean
115.94, best 146.86). The optimizer is deterministic, so building twice on the
same settings reproduces the lineups exactly — and the record counts both,
weighting whichever objective happened to be rebuilt. Here it flatters ceiling.

### What this changes

The `one_week_only` fix ships. The depth confound is **not** fixed — the
endpoint still pools builds of unequal size, and the table above had to be
computed outside it. Until that is normalized (compare each build's top-k
lineups, k the minimum across the builds being ranked, is the obvious
candidate), `objective_verdict` should be read as "which objective was built
most, and how deep", not as advice. No objective default was changed on the
strength of this.

## Did a day-stale roster table corrupt the week 9 board? No

The `nfl-api-roster-sync` cron died in the 2026-11-03 cluster outage, so
`current_roster` was holding its 2026-11-02T15:00 state when the week 9
projections were computed that evening. Checked rather than assumed.

Against nflverse's **season** roster, of the 326 projected players worth a
roster slot (≥ 1.0 pt): **4 differ, all on status alone**, nobody changed
teams, nobody is missing. The four are Kyle Monangai (RB CHI, 11.3), Rashid
Shaheed (WR SEA, 4.4), Drew Lock (QB SEA, 3.1) and Case Keenum (QB CHI, 2.1),
each `ACT -> INA` with description `A01`, which is a game-day designation and
not roster standing. Monangai is corroborated: the week 8 injury report has him
Out, DNP, thumb.

Depth charts had moved more — three snapshots published after the sync
(2026-11-02 17:01, 2026-11-03 07:02, 2026-11-03 14:59 UTC), 7,008 rows across
all 32 teams, 1,778 of them at skill positions. Comparing the newest chart's
rank-1 against the chart as it stood at sync time, **three starters changed**:

| team | pos | was | now | the board's projection for the new starter |
|---|---|---|---|--:|
| BAL | QB | Lamar Jackson | Tyler Huntley | 8.3 |
| HOU | RB | David Montgomery | Woody Marks | 8.8 |
| ARI | QB | Jacoby Brissett | Gardner Minshew II | 1.1 |

**The board already had all three right.** BAL QB reads Huntley 8.3, Fagnano
2.1, Jackson 1.2; HOU RB reads Marks 8.8 ahead of Montgomery 5.5. The depth
rank in `current_roster` is one input among several, and the usage history the
role and share models read had already moved — which is the answer to whether
a stale chart can quietly mis-rank a backfield: here it could not, because
snaps and touches had already said the same thing. Consistent with the loader
note that depth charts are not a top feature; snaps are.

No action taken. The 08:00 sync and the Wednesday 21:00 projection run restore
the table on their own.

### The measurement trap, again

The first pass at this used `load_rosters_weekly` and produced garbage: 20
"inactive" players and 47 "missing" ones. That table stops at the last
**completed** week (8), its `status` is game-day status for *that* week rather
than a forward-looking one, and it carries no rows at all for a team on bye —
so HOU, NO, NYG and SF appeared to have lost their entire rosters and anyone
inactive in week 8 looked out for week 9. This is the same `week.max()` hazard
already written up under the bye-week roster gap. Season rosters
(`load_rosters`) are the right table for "who is on this team now".

## The Tuesday projection run is structurally blind to injuries

Noticed when the user said Tyler Huntley was hurt and the week 9 board had him
as Baltimore's QB1 at 8.3 points.

`injuries.parse_nflverse_injuries` filters the report to the exact week being
projected (`df[df["week"] == week]`). nflverse's 2026 injury feed currently
holds **weeks 5-8 and no week 9 rows at all** — NFL teams file their first
report on Wednesday. So for a board built on Tuesday evening that filter
returns an empty frame and the weekly report contributes *nothing*.

What survives is `fetch_ir_players`, which reads IR/PUP/Reserve off the roster
`status` — 24 players on the week 9 board carry `prediction_type:
injured_out` from that path alone. Neither Baltimore quarterback is on IR, so
neither was zeroed. Huntley's 8.3 is pure usage history: he has been taking the
snaps, so the role and share models rank him first. Lamar Jackson, who the
newest depth chart no longer lists as QB1, lands at 1.2 the same way.

The cron is `0 21 * * 2,3` America/Denver. **The Tuesday run can never see an
injury report; the Wednesday run always can.** That is not a bug in either
piece — it is what the calendar allows — but it means the two runs of a week
are not the same kind of object, and only the second one should be built from.
The model cache does not get in the way: it is keyed on completed weeks, and
the reuse path exists precisely so a second run in the same week can "project
fresher injuries and depth charts against" the model already trained.

Worth considering, not done: the projections response says nothing about which
of the two it is. A flag reporting whether an injury report existed for the
week being projected would let the board say so out loud instead of leaving it
to be inferred. Carrying the prior week's OUT designations forward as a prior
is the other option and is worse — it would have zeroed nobody useful here and
would wrongly bury players who returned.

### A correction to the roster-staleness note above

That entry said the depth rank in `current_roster` is "one input among
several" to the projections. It is not an input at all. All 24 `injured_out`
players are absent from `current_roster`, which only happens because the
in-season path reads rosters straight from nflverse — as the loader note
records ("In-season projections were unaffected: they read nflverse directly,
not this table"). The stale sync affects the `/rosters` endpoints and the site,
not the board. The conclusion there stands; the reason given for it was wrong.

### Shipped: the board now records what it could see

Built 2026-11-03, after the entry above sat as "worth considering". A flag on
the response was the smaller half of it; the problem with a flag is that it is
only true while the response is fresh, and the question this week was asked
about a board that had already been built.

So the run is recorded instead, one row per projection run in a new
`projection_runs` table, written by `compute_projections` at publish time:

- **the injury sources**, separated — which weeks the nflverse feed held
  (`report_weeks`, `report_latest_week`), how many rows it had filed for the
  week being projected (`report_rows`, and the flag `report_available`), how
  many designations parsed out of it, and how many players came off roster
  `status` as IR/PUP/Reserve instead. That separation is the whole point: week
  9 zeroed 24 players and **not one of them because anybody filed a report**,
  which the board cannot say about itself.
- **the depth chart snapshot in force** (`depth_chart_as_of`), which nflverse
  overwrites roughly daily, so which chart a board read is otherwise gone.
- the roster source and row count, the model version, whether the model was
  reused from the cache, the training window, and what came out.

Served as `GET /projections/runs` and as a `provenance` block on
`GET /projections`, which also spells the conditions out in words — a run with
no report filed says so, naming the IR count as the only injury signal it had.
Vintages resolve on `as_of_week` rather than week, since an archived week 12
row came from whichever run had that many weeks of football behind it.

What this does not do: it changes no projection. A Tuesday board is still a
Tuesday board; it now says that it is one. Measuring how much that costs needs
Tuesday and Wednesday runs of the same week scored against each other, which
these rows make possible and did not exist before.

**First production reading (2026-11-05).** The Tuesday-night cron recorded run
1 for week 9 and it confirms the finding on live data:

| field | value |
|---|--:|
| `report_available` | true |
| `report_rows` (week 9) | 28 |
| `report_players` parsed | 18 |
| `questionable` | 18 |
| `ir_players` | 106 |
| `players_zeroed` | **106** |

`players_zeroed` equals `ir_players` exactly, so **every zeroing came from
roster status and not one from the injury report**, even though the report was
no longer empty — 28 rows had appeared for week 9 by Tuesday night. All 18
players it named parsed to QUESTIONABLE, which benches nobody. This is the
early-week `practice_status`-only shape, measured in production rather than
argued from the feed. Note also that 28 rows is against week 8's 320: the
Tuesday report is not a thin version of the Friday one, it is a different
document. The Wednesday run will overwrite this with its own row rather than
amend it, so the pair is now observable.

## FantasyGuru snapshots: storage, a shell mix, and a gate (2026-11-04)

The owner subscribes to FantasyGuru and has started exporting its weekly
"smash" matchup tables by hand, to read against `/coverage` and `/trenches`.
**No scraping and no automated pulls** — their terms are personal use only, so
the only route in is a human exporting a CSV. `nfl_projections/external.py`
has no network import at all and a test that fails if one is ever added; the
API accepts an upload and nothing else.

Built: storage plus upload (`/external/upload`, `/external/snapshots`,
`/external/team-lines`, `/external/wr-matchups`), two comparison endpoints
(`/external/compare/trenches`, `/external/compare/coverage`), a coverage-shell
mix in `coverage.py` exposed on `/coverage/shells` and beside each secondary,
and the grading harness `scripts/grade_external.py`. Nothing feeds
`/projections`: the harness is the gate and it refuses to open yet, for
reasons below that are worse than "not enough weeks".

### Their line ratings are a run-game measure, reproduced

Reproduced the sharp-edge session's comparison from the week 9 export against
`/trenches` ratings, re-run 2026-11-05 through the endpoint itself rather than
a standalone script (n=32 teams with both):

| their rating | against ours | Spearman | their session |
|---|---|--:|--:|
| O-line | run_block | **0.624** | 0.66 |
| O-line | pass_block | 0.413 | 0.42 |
| D-front | run_stop | **0.731** | 0.73 |
| D-front | pass_rush | 0.286 | 0.22 |
| O-line | their own D-front | 0.598 | 0.6 |

Confirms the read: taken as a run-game measure they largely agree with us,
taken as pass protection they do not, and their own two sides correlating 0.60
is what overall team strength leaking into both would look like. Biggest
disagreements on O-line rank: TEN (their 29 / our 12), CIN (18/4), DET (3/16),
NYJ (8/21), CAR (17/5).

Run against production on 2026-11-06 with the week 9 export uploaded for
real, the figures above are the deployed path's. One moved from the first
pass and one did not, and the reason is which `play_by_play` each read:
production uses the loaded table, while a local harness without that table
falls back to pulling nflverse directly and gives pass_block 0.419 against
production's 0.413. Same method, slightly different source rows; production
is the number that counts. The self-correlation really did move, 0.573 →
0.598, as did n (31 → 32), because the first pass was a standalone script and
the endpoint had never actually run. `trenches._grades` returns one DataFrame
while `coverage._grades` returns two, the comparison unpacked the first as a
pair, and the `ValueError` fell into a broad `except Exception` that reported
`no_ratings` — indistinguishable from a week with no ratings yet. The suite
was green because the only test for that route exercised the no-upload guard,
which returns before `_grades` is reached. Both routes now carry a `reason`
separating "nothing rated yet" from "the lookup broke", and the comparison
path has tests that drive it with real-shaped grades. A soft failure that
cannot say which failure it was is a silent one.

Their per-game line table is **entirely derived** — each ADV column is one
side's O-line minus the other's D-front, verified to 0.02 across all 30 sides
of week 9 — so the upload endpoint checks it and stores nothing, which is one
fewer thing to keep in sync. Two of the three exports are canonical.

### The coverage-shell mix: nflverse stopped charting it after 2025

`defense_coverage_type` and `defense_man_zone_type` live in nflverse's
**participation** release. nflreadpy refuses a later season outright ("Season
must be between 2016 and 2025"), and nothing else carries the fields: not
`load_pbp`, not `load_ftn_charting` (pressure and box counts, no coverage), not
`load_nextgen_stats`. Checked 2026-11-04.

So the mix is built and shipped — 32 defences, ~38 charted plays a team-week,
windowed and prior-weighted exactly like `corner_grades` — but **for 2026 it is
a scheme tendency last measured in 2025**, not this week's. Every row carries
`shell_seasons`, `plays`, `plays_prior` and `current_season_plays` so that is
visible in the data rather than in a footnote. It is still worth showing:
scheme travels with a coordinator. A defence with a new one is described by
its predecessor.

### FG's shell percentage does not measure what nflverse measures

This is the one part of their coverage table that can be checked against
charting, and the check fails. For the 21 of 25 pairings whose shell maps onto
nflverse's vocabulary (Match has no equivalent and shadow carries no shell):

| | FG's stated share | nflverse 2025, same defence |
|---|--:|--:|
| mean | **49.0%** | **23.3%** |
| Spearman | | **0.282** |

Roughly a factor of two, uniformly, across 21 defences. FG routinely states
50–65% for a single shell where no defence in nflverse's charting exceeds
40.3%. A systematic 2x gap is far too clean for year-over-year scheme drift,
so their percentage is on a **different denominator** — per-corner snaps, or
zone plays only, or some narrower play set. Whatever it is, it is not a
team-level shell share and the two numbers cannot be compared.

That matters for how much of this table is new. The advantage score is another
estimate of a matchup our own residual tests closed three times over. The
shell percentage turns out to be uncheckable against the only charting that
exists. **What is left is the assignment** — which corner a receiver draws, and
whether he travels — which nflverse genuinely does not publish.

### And the assignment accrues at one row a week

The export is 25 pairings a week, of which **one** is a shadow assignment.

| weeks | pairings | shadow rows | slope CI vs the n=16,380 WR test |
|--:|--:|--:|--:|
| 4 | 100 | 4 | 12.8x wider |
| 6 | 150 | 6 | 10.4x wider |
| 20 | 500 | 20 | 5.7x wider |
| 40 | 1,000 | 40 | 4.0x wider |

"Per-corner coverage grades" called WR null at n=16,380 with a slope interval
of about ±0.11. At n=100 the same precision is 12.8x worse, so a four-week
harness **cannot separate "no effect" from an effect three times larger than
anything we have ever measured.** The harness therefore reports
`inconclusive` rather than a null, carries its own power arithmetic on every
target, and keeps the gate shut even when a fit does pay — a signal that pays
only at 12x-too-wide intervals is the shape of a signal that does not pay.

Shadow coverage is reported separately and never folded into that aggregate,
because four observations averaged into a hundred would read as adequately
powered. At ~1 a week, 100 shadow rows is **96 more weeks**. The honest
statement is that the only genuinely new column here needs seasons, and the
4–6 week plan will not test it.

One method substitution, stated because it weakens the result: the original
fits leave-one-season-out and there is one season of snapshots, so a week
plays the part of a season. Weeks of one season share a model vintage and a
roster in a way seasons do not, so a pass here would be weaker evidence than a
pass there.

### Operational notes

Both teams and players map cleanly. FG's full names are city-style ("LA
Chargers", "NY Jets") and two of its abbreviations disagree with nflverse (JAC,
LVR); 32 of 32 teams and all 30 matchup sides resolve, and "LA" is deliberately
refused as ambiguous rather than guessed. Player ids resolve 25/25 both sides
against the season roster — which is the right scope, because normalising a
name strips suffixes and "Marvin Harrison Jr" otherwise collides with his
father while "Mike Evans" collides with a 1992 defensive tackle. Two active
players named Justin Jefferson (WR and LB) need the team to separate them, and
one name ("Jacob Parish" for nflverse's "Jacob Parrish") resolves through a
team-scoped fuzzy match that is recorded as `team_fuzzy` rather than passed off
as exact.

**The duplicate-header trap.** The WR export has two columns both headed
`Team` — the corner's and the receiver's. `csv.DictReader` keeps the last and
silently drops the corner's team; `pandas.read_csv` renames it `Team.1`. Both
produce a plausible-looking frame, so no name-keyed mapping is built over that
header, and a test pins the failure mode.

**Identifying a file by its contents (2026-11-06).** The first version read
all three exports by absolute column position against row 1, which was robust
to the duplicate header and to nothing else. These files arrive named
`table.csv`, `table (1).csv` and whatever the browser calls the next one, so
the filename was already ignored — but a title row, a blank leading line, a
reordered export or one added column would each have been refused outright,
and an added column before the receiver's `Team` would have been worse than
refused: absolute positions would have read the corner's team as the
receiver's and stored it, silently.

Now the header is *found* rather than assumed — the first row within twelve
that carries a known export's columns, in any order — and columns are resolved
by name. The two `Team` columns are the one case a name cannot settle, and
they are anchored to the column each follows: the corner's team is the first
`Team` after `Defensive Back`, the receiver's the first after `Primary
Receiver`. That is the file's actual grammar, each name followed by its team,
rather than a column number that an upstream change can shift. If the two
resolve to the same column, parsing raises instead of guessing.

A nameless upload now works too. A multipart part with no filename is, per the
spec, an ordinary form field, so Starlette delivered it as a `str` and the
`UploadFile` annotation rejected it with a 422 about types — a file whose only
defect was having no name. Verified against the real week 9 exports renamed to
`Weekly Smash Report (3).csv` and to `download` with no extension: byte-identical
content hashes to the original run, 25/25 and 32/32 mapped.

**Week 9 is banked (2026-11-06).** Uploaded to production: 25 pairings and 32
team lines, nothing unmapped, one `team_fuzzy` (Parrish) and one shadow
assignment (Benford on Justin Jefferson). The derived per-game table was sent
too and came back `checked_not_stored` with 30 self-consistent sides and no
drift against the stored team lines. Live `/external/compare/coverage`: 22 of
25 matched to our corner grades, Spearman −0.33, five of those grades resting
on under 20 targets. One week of four, and the shadow count is the number
that matters: 1.

## Reading a published DFS column against our own board (2026-11-06)

A subscriber write-up for week 9 2026 (FantasyGuru, "DFS Marlin's Catch of the
Week", finalised Thursday evening), checked claim by claim against our tables.
Three results, one of them a problem with us rather than with them.

### Their defensive matchup numbers are our raw numbers

31 checkable rank claims ("Detroit has allowed the most fantasy points to
quarterbacks", "New Orleans the most to running backs", and so on) against
`/fpa`:

| compared with | Spearman | median gap | within 5 ranks |
|---|--:|--:|--:|
| our **raw** points allowed | **0.96** | 1 rank | 100% |
| our **schedule-adjusted** grade | **0.61** | — | — |

So there is no disagreement about the arithmetic: their matchup table is the
same raw points-allowed table we compute, and at that level we agree almost
perfectly. The entire difference is the adjustment — who each defence has
actually faced — which they do not appear to apply.

That is not a cosmetic gap. The largest divergences invert the recommendation:

| | theirs (raw) | ours (adjusted) | their call |
|---|--:|--:|---|
| PHI vs WR | 3rd softest | **30th** (3rd toughest) | "the best wide receiver matchup among the premium group" |
| NYJ vs QB | 16th | **1st softest** | "a fine mid-tier option, just below Purdy and Goff" |
| LV vs TE | 4th softest | 20th | "my favorite tight end in this range" |
| GB vs RB | 9th softest | 27th | "I like him as a script play" |
| DEN vs RB | 10th softest | 23rd | "the matchup and the volume point the same way" |

A defence that has faced the league's best receivers looks soft on raw points
and is not. This is the one place our table is straightforwardly better, and
it is worth saying that the adjustment was previously measured as *not* buying
accuracy — it buys interpretation, which is what a matchup read is for.

### Their man/zone rates do not reconcile with our charting

Fourteen stated rates ("Cleveland plays man on 49% of its snaps, the highest
on the board", "Arizona plays zone on 85%") against `coverage.shell_mix`:

| | value |
|---|--:|
| Pearson | 0.476 |
| Spearman | 0.417 |
| mean absolute gap | 9.5pp |
| mean, theirs / ours | 27.1% / 33.3% |

Directionally they agree at the top (Cleveland highest in both), but PHI
(+18.5pp), DET (+16.6), SEA (+16.1), ARI (+14.1) and WAS (+14.0) are too far
apart to be noise. **This comparison cannot settle anything**, because our
mix is a 2025 tendency — nflverse stopped publishing participation after 2025
— and theirs is presumably 2026, so scheme change and a different denominator
are perfectly confounded. Consistent with the earlier finding that their
shell *percentage* sits on a denominator nflverse does not share.

### The finding that matters: we are blind to practice participation

Our week 9 board was computed Wednesday 21:32 (run 1, the last scheduled run —
the cron is Tuesday and Wednesday only). The article, written Thursday
evening, names 22 players as out or doubtful. **Every one of them is still
projected at full strength on our board**, including Sam Darnold (12.8, listed
OUT), Deshaun Watson (13.6, concussion protocol), Mike Evans (11.7, DNP both
days), Malik Nabers (14.6) and A.J. Brown (7.3). The four quarterbacks
expected to inherit starting jobs are projected at 3.1 (Lock), 1.4 (Sanders),
1.1 (Minshew) and absent entirely (Milton).

Re-running would not fix it. Measured on the live feed the same afternoon:

| nflverse week 9 injury feed, Friday | rows |
|---|--:|
| total | 340 |
| carrying a game designation (`report_status`) | **7** |
| of those, Out | 3 (all Baltimore, Thursday's game) |
| `practice_status` = Did Not Participate | **99** |

The feed knows perfectly well that Darnold, Evans, Brown and Nabers did not
practise. We throw it away: `parse_nflverse_injuries` maps any DNP without a
game designation to QUESTIONABLE, which zeroes nobody, and collapses "did not
participate" and "limited" into that same bucket. The designation only lands
on the final report, which arrives Friday and which nflverse had not ingested
at the time of checking.

So the gap against a human columnist is not matchup modelling or projection
quality. It is that they read Wednesday and Thursday participation and we read
only the game designation. That is a concrete, testable candidate — a discount
graded by participation, not a zero, with DNP-both-days separated from
limited — and it needs the usual residual treatment before it goes anywhere
near the board. Noted, not shipped.

One practical corollary: FanDuel's own slate CSV carries an `Injury Indicator`
(IR / O / Q), and the optimizer already refuses IR/O/D. That is the safety net
that keeps an out player out of a lineup even when the board still likes him —
but it does nothing for the inheritor, who stays underprojected.

### Shipped: `/dfs/targets`

The write-up's *shape* is worth copying even where its inputs are not: price
tiers per position, ordered inside a tier, a sentence per player. Built as
`nfl_projections.dfs_targets` plus `GET /dfs/targets`, ranking on our
projection divided by salary and nothing else. Opponent grades ride along per
player in a `context` block that the response states did not move anybody; no
Vegas, by standing rule; no named overrides, enforced by a test that renames
every player and asserts the output is identical. The provenance block reports
the board's vintage first, because a targets list is read days after the board
was computed and that interval is where the injuries live.

## Week 9 scored: why 45 lineups all landed in the same place (2026-11-09)

The week was a loss — 45 entries, nothing much over 100. Scored locally against
nflverse with the same `fanduel_points` the production scorer uses, because
`scripts.score_lineups` had run Sunday 19:00, before nflverse published, and
reports `scored_lineups: 0`. Production will say the same once it runs again.

| | week 9 actual |
|---|---|
| the 45 entered lineups | mean **87.24**, best 127.96, worst 66.36 |
| of those, 100+ | **5 of 45** |
| FanDuel's own FPPG, one lineup off the same pool | 109.44 |
| our board, mean objective | 79.74 |
| our board, ceiling objective | 79.74 — *the same lineup* |
| best available, hindsight (lower bound) | 224.26 |

Baselines are the 8 skill slots at $56,400 with the NO defense ($3,600, 5.0)
bolted on each, because the board does not project defenses at all and the
pool is the 248 board-matched slate players `/dfs/targets` returns, truncated
at 50 per position-tier — so "best available" is a floor on the real number.

### It was not the board

Every board player whose game was played, projection against actual:

| | n | proj | actual | bias | MAE | inside [floor, ceiling] |
|---|---|---|---|---|---|---|
| all | 469 | 4.37 | 4.22 | **+0.15** | 3.16 | 65% |
| QB | 76 | 6.74 | 5.87 | +0.87 | 3.98 | 30% |
| RB | 104 | 5.58 | 4.76 | +0.82 | 3.65 | 69% |
| WR | 182 | 3.72 | 4.18 | −0.46 | 3.26 | 68% |
| TE | 107 | 2.61 | 2.59 | +0.01 | 1.92 | 83% |

Essentially unbiased, and on the players that matter (projection ≥ 10) the
band came in at 13% below floor / 79% inside / 8% above ceiling against a
design of 10/80/10. The board did its job.

The bias lives entirely in the players the optimizer *chose*: **+2.45** on the
30 it used against **−0.01** on the 439 it did not. That is selection bias, not
calibration error — maximising a noisy estimate picks the names whose noise ran
positive, and it is the structural tax on any optimizer. Worth measuring every
week; not a bug.

Nor was it the injury blindness from last week's entry. The Saturday slate
re-download did its job: no rostered player was inactive, and the lowest actual
in any lineup was 1.40, not a zero. That fix held.

### Negative result: `ceiling` and `mean` are the same objective

Across the 245 slate players with a projection:

| pair | pearson | spearman |
|---|---|---|
| mean vs ceiling | 0.9961 | **0.9949** |
| mean vs floor | 0.9673 | 0.9345 |
| mean vs FPPG | 0.9273 | 0.8647 |

and on points per $1k, which is what the optimizer actually trades on, mean vs
ceiling ranks at Spearman **0.9927**; the top 30 by each share 28 players. The
ceiling/mean ratio has median 2.098 and an IQR of just 1.982–2.287.

So the ceiling is the mean times about 2.1, and selecting on it changes almost
nothing — which is why the mean and ceiling solves above returned the identical
nine men. All six week-9 builds ran the ceiling objective, believing they were
chasing upside; they were running the mean optimizer.

This sharpens the known quantile to-do. The bands are *marginally* well
calibrated (13/79/8) and simultaneously carry almost no cross-player
information: the model is not saying anybody is unusually skewed, only that
everyone's ceiling is twice their mean. For DFS the whole point of a ceiling is
to find the player whose distribution is right-skewed *relative to his mean*,
and the quantile head is not currently expressing that. Calibrated but
uninformative. `floor` is the only objective that meaningfully reorders
(Spearman 0.9147), which fits its by-objective record being the best of the
four.

### What actually went wrong: 45 entries, 31 bets

| | |
|---|---|
| lineups entered | 45 |
| distinct lineups | **31** |
| exact duplicates | **14** |
| builds 32 and 33 | identical, all five lineups |
| players at 55.6% (25/45) | Gibbs, Hubbard, Cousins, Allgeier, Tucker, Cooper, NO |
| tightest per-build exposure cap | 46% |
| mean overlap between any two lineups | 3.53 of 9 slots |

Six builds, each honouring its own exposure cap, pooled into seven players in
55.6% of the week — and the optimizer is deterministic, so re-running it with
the same parameters returns the same lineups. 14 of the 45 entries were paid
for twice. Those seven shared men returned 66.54 points across seven of nine
slots, so for 25 of the 45 lineups only two slots could differentiate, and the
whole portfolio landed in a 66–105 band.

`/optimizer/exposure` already reports exactly this, including the `over_cap`
list and a note that the per-build cap "binds inside a build only". The
diagnosis was sitting in the API before the games were played. What is missing
is the step that acts on it: nothing dedupes across builds for a week, and
nothing warns at build time that the lineups being added duplicate ones already
entered.

FPPG beating the board this week (109.44 vs 79.74) is one player — Kenneth
Walker III, whom FPPG priced at 23.0 and we had at 17.1, scored 37.30. The
season record is `beat_fppg_rate` 0.702 over 319 scored lineups, so week 9 is a
miss, not a regime change. Not evidence against the model.

### Reporting defect found while doing this

`/optimizer/performance` takes its `projected` from `LineupEntry.projected_points`,
which is the total of whatever column the objective used. So `error = projected
− actual` compares a sum of ceilings against a realised total: week 8's
headline "projected 187.09, actual 102.42, error 84.67" is not model error, it
is the word "ceiling". The same column reads −69.68 for floor builds. The
by-objective table is the only part of that response that is internally
comparable. Either the objective belongs on every row, or `projected` should
always be the mean projection with the objective total reported beside it.

Separately: `/dfs/targets` lists `DEF` in its default `positions` and silently
returns no defense group, because the board has no defense rows — the optimizer
fills that slot from FanDuel's FPPG (`optimizer.py` says so at `/slate-coverage`).
It should say that rather than omit a tier without comment.

## Negative result: a player's own high and low make a worse band than the model's (2026-11-09)

Proposal: drop the quantile head and report the lowest score a player has posted
as his floor, the highest as his ceiling, with the projection as the expectation.
Tested on weeks 5-9 2026, n = **617 player-weeks**, band built only from games
played before the test week, restricted to projections >= 5 and to players who
actually played. Week 7's board was computed after that week's Thursday game, so
CHI and NE are excluded from it. Three bands, same centre:

| band | coverage | mean width | below floor | above ceiling |
|---|---|---|---|---|
| model (quantile regression) | **81.7%** | 18.52 | 7.3% | 11.0% |
| his own min / max | 71.0% | 16.90 | **16.0%** | 13.0% |
| his own p10 / p90 | 57.4% | 12.92 | 24.6% | 18.0% |

Rescaled so all three have the same mean width — which asks only whether the
band puts its width in the right *places* — the model still wins: **73.7%**
against 67.7% for min/max and 68.2% for p10/p90.

### Why min/max loses, which is the useful part

A min and max over *n* games is not a range, it is an unstable estimate of the
1/(n+1) and n/(n+1) quantiles. With five prior games "the lowest he has ever
scored" is his 17th percentile, so it should be undercut about 17% of the time —
and it is, 16.0%. The band's meaning therefore drifts all season as *n* grows:

| prior games | n | width | below min | above max | covered |
|---|---|---|---|---|---|
| 3 | 32 | 13.74 | 25.0% | 15.6% | 59.4% |
| 4 | 161 | 14.82 | 23.6% | 16.8% | 59.6% |
| 5 | 149 | 16.86 | 13.4% | 14.1% | 72.5% |
| 6 | 123 | 17.82 | 12.2% | 8.9% | 78.9% |
| 7 | 106 | 18.66 | 13.2% | 11.3% | 75.5% |
| 8 | 46 | 20.07 | 8.7% | 8.7% | 82.6% |

Coverage climbs from 59% to 83% and width from 13.7 to 20.1 points with nothing
about the players changing — it is the sample growing. In week 4 such a "floor"
is broken a quarter of the time. The model band over the same split holds 79-88%
at a roughly flat width, because it is a quantile, not an order statistic.

### Neither band's width predicts anything

`|actual - projection|` regressed on band width, controlling for the projection
(standardised, 95% CI):

| band | width slope |
|---|---|
| model | +1.526 [−0.143, +3.194] |
| min/max | −0.136 [−0.556, +0.284] |
| p10/p90 | +0.056 [−0.373, +0.486] |

All three include zero. And no ceiling picks out who goes off: against
`actual − projection`, Spearman is −0.079 (model), −0.030 (min/max), −0.038
(p10/p90); against a big game (actual > projection + 8, 94 of 617) the
point-biserial correlations are +0.066, +0.054, +0.083. The p10/p90 one is
nominally p = 0.039, which is one marginal hit out of six tests on a tiny
effect — not a finding.

So historical range carries no information about this week's deviation beyond
the projection itself. This is consistent with the earlier reverted experiment
where rolling FP standard deviation and boom/bust rate were added as model
features, failed to widen player-specific bands, and regressed the mean model.
Player-specific dispersion in this data is largely not predictable.

### What follows

Keep the quantile band as the probabilistic band; it is the best of the three
and it is marginally calibrated. The proposal's real target — that the band
should mean something a person can read — is better served by reporting the
historical range as explicit *context* beside the projection ("4.2 to 28.1 over
8 games, median 11.4") rather than as the band, which is honest about it being
descriptive and is the kind of line that made the subscriber column's player
summaries readable.

The statistically correct version of the instinct is a per-player band built
from his own *residual* history — how far he has deviated from his own
projection — not from his raw score range. That cannot be tested yet: only
weeks 5-9 of boards are archived, so a player has at most four residuals.
`weeks_archived` grows every run, so it becomes testable later in the season.

### Shipped: a player pool, and lineups that stop repeating (2026-11-09)

Three fixes for the week-9 post-mortem above, none of which touch the board.

**The pool.** `optimize_lineups` takes `player_rules`, a per-player
`{min_exposure, max_exposure}` in percent of the lineups. It is one dial: 0 is
an exclusion, 100 a lock, 40 a target. A floor is enforced by forcing the
player into the lineups he still needs to reach it, which is the thing the
global `max_usage_percentage` could never do — it caps everyone equally and can
neither require a player nor name one. A bare number is a target rather than a
cap deliberately: capping a man at 60% and watching the solver use him twice in
ten is the failure this exists to fix.

Locks are implemented as a bonus larger than any real lineup can score, so the
exact solver is untouched: the optimum of the inflated objective is the best
lineup containing him whenever one exists, and the best without him when none
does — which is also how a contradictory pool is detected rather than silently
honoured. Two locked quarterbacks, locks over the cap, more locks than roster
slots, a floor above its own ceiling, a misspelled name and a lock on a player
FanDuel marks O are each refused by name. A mistyped lock that quietly does
nothing is the worst available outcome, so nothing is ignored.

Rules are stored in `slate_player_rules`, keyed on (slate_id, nickname) rather
than the week: a re-upload replaces the slate, and a rule set against
Wednesday's file must not attach itself to Saturday's different contest.
`GET /optimizer/pool` is the table — price, board, value, what the player has
actually scored, his share of the week so far, and his rule — with
`POST`/`DELETE` to set and clear, and a column of controls in the dashboard.

**Dedupe.** `seen_lineups` holds back anything already recorded for the week, on
by default. One trap found while building it: such a build must NOT supersede
the earlier one. It returned the next-best set precisely because the first set
exists, so the old "pressing Build twice is not two samples" rule would delete
the better lineups it had just stepped around and keep only the ones it settled
for — two clicks leaving the week worse than one. Superseding now happens only
when nothing was deduped against. `pool_exposure` additionally counts the
entered lineups toward the percentages, which is the only thing that holds a
player down across builds.

Also: when dedupe empties a build the refusal says so, rather than blaming the
cap. That came out of the test suite, where a session-scoped engine let builds
accumulate until a small slate had no distinct lineup left and every later
build came back empty — a true answer to a question no test meant to ask.

**The range in the summaries.** `/dfs/targets` now carries each player's own low,
high and median, and `summarize` adds "Has scored 4.2 to 31.2 in 7 games, median
19.0", plus a sentence when the projection sits outside it, which is the board
claiming something new about him. It is description and labelled as such — the
negative result above is what keeps it out of `value`, the tiers and the ranks.

And `/dfs/targets` no longer drops the DEF tier in silence: the board holds no
defenses at all, so a `positions_absent` list and a note say which positions
came back empty and that DEF is not a data gap but a slot that runs on
FanDuel's own FPPG.

709 → 754 tests in nfl_projections, 714 → 755 on NFL-API.

## Consistency: scoring steadiness is not a property of a player (2026-11-09)

Asked for "the most consistent players — lowest standard deviation from their
mean". Measured, that request splits in two, and only one half of it exists.

Method: split-half on odd weeks against even, per player, 2025 (210 players
with 6+ games and a 5-point mean) and 2026 through week 9 (174). The MEAN is
carried through every test as the yardstick — it is the thing known to be
measurable, so it says what a good reliability looks like on this data.

### Raw SD ranks the players who do not score

SD correlates **+0.647** with the mean (spearman +0.683); `log(sd) = 0.78 +
0.49 log(mean)`, r² 0.45. So the ten "most consistent" players of 2025 by
lowest SD are:

| player | mean | sd |
|---|---|---|
| Malik Washington | 5.2 | 3.14 |
| Darius Slayton | 5.5 | 3.34 |
| Isiah Pacheco | 6.0 | 3.38 |
| Greg Dulcich | 5.5 | 3.38 |
| T.J. Hockenson | 5.8 | 3.43 |

Nobody would roster any of them. Their scores do not move because they do not
score — the metric is measuring the absence of a role.

### And the level-adjusted version is nil

| split-half, 2025 | r | 95% CI |
|---|---|---|
| fantasy mean *(yardstick)* | **+0.643** | [+0.556, +0.716] |
| fantasy sd | +0.111 | [-0.025, +0.243] |
| fantasy cv | +0.235 | [+0.103, +0.359] |
| fantasy sd, level controlled | **-0.047** | [-0.181, +0.089] |
| fantasy cv, level controlled | +0.021 | [-0.114, +0.156] |

2026 agrees: level-controlled CV **-0.096**, CI [-0.241, +0.054]. CV's bare
+0.235 is inherited from the mean (the two correlate -0.635), and vanishes the
moment the level is partialled out of both halves. A player who scored steadily
in one half is no more likely than anyone else to score steadily in the other.

### Volume steadiness is real, and transmits — weakly

| split-half, 2025 | r | 95% CI |
|---|---|---|
| opportunity mean | +0.943 | [+0.926, +0.957] |
| opportunity sd | +0.734 | [+0.665, +0.791] |
| opportunity sd, volume controlled | **+0.412** | [+0.293, +0.518] |
| opportunity cv, volume controlled | +0.328 | [+0.201, +0.444] |

and across the halves, **opportunity CV predicts next-half fantasy CV at
+0.231**, CI [+0.099, +0.355] (2026: +0.285, CI [+0.142, +0.416]). Steady usage
buys some steadiness of scoring. It is the only route to it that measures.

The in-season caveat is power, not contradiction: over 2026's nine weeks the
volume-dispersion reliability falls to +0.070, CI [-0.080, +0.217]. Four or
five games a half cannot measure a spread.

### The best predictor of future consistency is the scoring level

Against next-half fantasy CV: **fantasy mean -0.338** (2026: -0.317), past
fantasy CV +0.235 (2026: +0.072), opportunity mean -0.237 (-0.298). The
consistent player is mostly just the good player — which is the same shape as
the quantile finding above, where the band turned out to be 2.1x the mean.

### Shipped

`nfl_projections.consistency` and `GET /dfs/consistency`, ranked on volume
steadiness (`100 x (1 - the CV of opportunities per game)`) with a floor of 5
opportunities a game, because a two-target role has a tiny spread and is not
what anyone means. Scoring range, hit rates and the usage trend ride along as
description; a test shakes the scoring hard without touching a carry and
asserts the ranking does not move. The response states the reliabilities,
including the one that says to read `games` before the rank.

The chart is a scatter — opportunities per game against steadiness, bubble size
points per game, one position at a time. One position because a quarterback
sees ~38 opportunities a game and a receiver ~8, and because a scatter is an
all-pairs form where four categorical hues cannot be separated safely. On first
render the three direct labels overlapped into a smear (the top three by
steadiness are clustered by construction); it labels the two extremes instead,
which are far apart and are the points worth naming.

## A read of all three repos: three production readings, and the record learns about depth (2026-11-10)

A review pass, not an experiment. Three things read off production through the
public API that change what the notes above assume, and one instrument fixed.
Nothing here touches the board.

### The weekend refresh had never run

`NFL-API/k8s/projections-refresh-cronjob.yaml` (commit c860255, 2026-10-28)
schedules Fri/Sat/Sun runs for exactly the gap "The Tuesday projection run is
structurally blind to injuries" describes. On 2026-11-09 `GET
/projections/refresh` gave `last_run_at` 2026-11-05T04:32Z — the Wednesday run
— and `/projections/runs` held one row for week 9, `players_zeroed` 106 =
`ir_players` 106. A full Friday, Saturday and Sunday had passed. `k8s/` is
applied by hand and ArgoCD syncs only the chart, so the manifest was a file,
not a job. Whether it was never applied or applied and failing cannot be told
over HTTPS; either way no late-week board existed for week 9, and the entries
of 2026-11-03 and 2026-11-06 that call the cron "Tuesday and Wednesday only"
were describing production correctly and the repo incorrectly.

Moved into `helm/nfl-api/templates/` under the same object name, so it deploys
on sync and adopts the hand-applied copy if there is one. **Not yet verified
live** — that needs a push and then a Friday.

What it should be worth is already measured, in "The promotion fix, scored the
way the board is scored": re-ranking on a known absence is +6.10% [+4.49%,
+7.74%] on the promoted and takes a QB2's error from 9.709 to 6.418. That same
entry found a practice-only trigger scores 4.0095 against the designation's
4.0084, so the participation-graded discount floated on 2026-11-06 buys
nothing once a run sees the designation. Dropped as a candidate.

The test, weeks 10-13: the Wednesday board against the Sunday 06:00 board, on
players whose projection moved 2+ points, a scratched player scored as 0 (the
standing scorer drops him, which hides half the effect). Expect ~80 rows;
that detects about 1.1 points of paired MAE. Under 0.5 with an interval
covering zero, cut it back to one Saturday run.

### The training window is a tie, prospectively

`GET /projections/shadow/accuracy`, weeks 5-9, identical player-weeks:

| | n | live (2026 only) | shadow (2020+) | delta |
|---|--:|--:|--:|--:|
| pooled | 1,529 | 3.979 | 3.975 | −0.004 |
| by week | | | | +0.032, +0.001, −0.022, +0.021, −0.056 |

Three weeks one way and two the other, pooled inside a hundredth. The backtest
cost of +0.068 to +0.097 (#26) has not shown up in five live weeks. Five weeks
is not many, but the sign flips are the #28 shape again. Nothing to change.

### The DEF slot is not on FPPG

`_attach_defense` joins `defense.project` onto the slate and week 9's exposure
shows New Orleans `from_model: true`. The board still holds no DEF rows, which
is what `/dfs/targets` reports; the optimizer's slot is a separate path and has
been on the component model since it shipped. CLAUDE.md said otherwise and is
corrected.

### A selection tax, derived rather than measured

Mean-objective builds, season to date: 67 lineups projected 100.45 and scored
102.41. The board runs 1.38 low per scored player (it is a median), so nine
unselected players would beat their total by about 12; these beat it by 2.
That puts the cost of choosing on a noisy estimate near **10 points a
lineup**, consistent in sign with week 9's +2.45 on the chosen thirty. It is
arithmetic on two aggregates with different populations (the −1.38 includes
players no lineup would hold, and is −1.02 above 12 projected), so it is an
order of magnitude and not a number to fit anything to.

The test that would make it one: the residual test on players projected 8+,
candidate = trailing-5 minus board, both already frozen per row in
`projection_accuracy` (~2,800 rows). A slope interval covering zero ends it.
`blend.py` found the form-weight curve flat board-wide, so the prior is under
even; the claim is only about the selected tail.

### Shipped: `/optimizer/performance` stops confusing depth with objective

The defects named on 2026-11-03 and 2026-11-09, both still live when read:
`projected 217.32, actual 87.24, error 130.08` for week 9, and a verdict of
"floor best, ceiling worst" resting on pooled builds of unequal size.

- `projected_scale` names the objective a total is a sum of ("mixed" when a
  bucket holds several), and `error` is only drawn for mean builds.
- `matched_depth`: objectives compared inside one (week, contest, lineups
  requested) cell, every build cut to the shallowest build's scored count, an
  objective built twice counted once. This is the table that had to be
  computed by hand on 2026-11-03.
- `depth`: each build's top five against its lineups ranked eleventh and
  below, averaged within a week so a week is one observation.
- the pooled verdict stays and says `basis: "pooled"`.

`depth` is the instrument for the open question the last four weeks keep
circling. Week 7 at one vintage read 111.8 against 84.0 (mean), 149.7 against
71.8 (ceiling), 114.9 against 101.0 (floor) — top five against ranks 11-25,
from the table in that entry — while week 8's commit build lost to its hedge
(88.9 against 106.9) and week 9's single best lineup scored 79.74 under a
portfolio mean of 87.24. If the gap is real, then in a cash contest a
duplicate of lineup 2 is a better entry than lineup 30, and deduping by
default (shipped 2026-11-09) is right for a tournament and wrong for cash.
**No default was changed on this.** The offline version is the test: the cash
preset at 25 lineups over 2025 weeks 4-18 on proxy salaries (#13's harness)
plus the saved 2026 slates, weekly top-5 minus ranks 11-25, clustered by week.
Under 5 points, or ahead in fewer than 13 of ~19 weeks, and the spread stays.

766 → 774 tests on NFL-API.

## A usage trend is half real: what the game after an "up" or "down" looks like (2026-11-11)

Built for a chart, not for the board: NFL-API `GET /opportunities/weekly/` and
the NickKnows page `/NFL/Opportunities/Trends` draw each player's
opportunities, targets and rush attempts by week and call a direction. The
call is his last three games against the games before them, in looks a game,
and it is `up` or `down` only when the change clears both a minimum (2.0
opportunities, 1.5 targets or carries) and the standard error of that
difference from his own games. Under five games there is no call, and if the
last three games reach back more than five weeks — across an absence — the
call is withheld as `interrupted`.

On 2026 weeks 1-9, 295 players at 3+ opportunities a game: 38 up, 34 down,
121 steady, 77 too few games, 25 interrupted. The old percent-change trend in
the same file calls a one-target player going to three "+200%"; this one
mostly says steady, which is the point.

### Does a called trend carry into the next game?

Walk-forward inside the season: call the direction on a player's games before
game *t* (t from his sixth), then read game *t*. Run before the `interrupted`
rule was added, so absences are in it.

| metric | call | n | called change | next game vs his earlier average | next game vs his season average |
|---|---|--:|--:|--:|--:|
| opportunities | up | 119 | +4.42 | **+2.31** | +0.04 ± 0.39 |
| opportunities | down | 65 | −4.25 | **−2.53** | −0.39 ± 0.58 |
| opportunities | steady | 363 | +0.25 | +0.28 | +0.14 ± 0.20 |
| targets | up | 95 | +3.14 | +1.59 | −0.05 ± 0.27 |
| targets | down | 61 | −3.00 | −1.23 | +0.30 ± 0.36 |
| carries | up | 54 | +5.13 | +3.95 | +1.44 ± 0.78 |
| carries | down | 37 | −4.74 | −3.60 | −1.24 ± 0.86 |

**About half of a called change in opportunities or targets shows up in the
next game, and about three quarters of a change in carries.** Against the
player's whole-season average the next game is indistinguishable from zero for
opportunities and targets — the recent three and the earlier games are worth
roughly equal weight, which is what "half" means. Carries are the exception:
the next game sits about 1.3 beyond even the season average in the direction
of the call, at under two standard errors on 54 and 37 calls.

That is consistent with everything above it in this file. The board already
reads trailing-3 and trailing-5 form, so a trend that is half regression is
not new information to it, and "the mid-season mover" found the same decay
through the network. **Nothing here feeds a projection**; the page says the
call describes what has happened and states the one-half figure. If any of it
is worth the residual test it is rush attempts, and n is too small to say:
one season, nine weeks, overlapping windows per player, so the standard
errors above are optimistic.

789 tests on NFL-API, 33 on NickKnows.

## Lineup depth: the top of a cash build is worth about ten points, and shrinking toward form is not (2026-11-12)

Two tests from the review's plan, kill rules written into each harness before
it ran. Harnesses: `experiments/lineup_depth.py`, `experiments/selection_tax.py`.

**Setup shared by both.** Five backtest seasons, 2021-2025 weeks 4-18, on the
cached leakage-free boards (`evaluate.backtest`, 1 seed, 60 epochs, trained on
prior seasons), plus the four real 2026 slates (weeks 6-9) from production's
`/optimizer/pool`, scored on nflverse weekly stats. Backtest salaries are a
proxy: each position's real 2026 salary regressed on FPPG (r 0.81-0.86),
applied to the player's season-to-date average. The cash preset builds 25
lineups of the eight skill slots under $56,000, since the backtest board holds
no defenses.

Two caveats on the backtest arm, both in the direction of flattering it: its
board only holds players who recorded a stat line, so nobody in the pool is
inactive; and a proxy price carries none of FanDuel's own information, so the
board's edge over the price is larger than on a real slate (in-build rank
correlation 0.29-0.35 here, 0.09-0.18 on the real slates).

### Trap found first: the 2021-2024 boards carry duplicate rows

About 1,150 player-weeks a season appear twice, as exact copies (the
dataset's depth-chart merge, noted under "Training on the current season
alone"). 2025's board is clean. Left in, the builder can put one player in two
flex slots, and season-to-date averages count his games twice. Both harnesses
drop them; every number below is after the fix. Anything else built on those
cached boards should do the same.

### Depth: passes on the backtest, unconfirmed on real slates

Weekly mean of a build's top five against its lineups ranked 11-25, ranked by
the builder's own projected total:

| cash aggression (usage cap) | weeks | gap | 95% CI | weeks top ahead |
|---|--:|--:|---|--:|
| 0 (60%) | 75 | +10.96 | [+7.25, +14.66] | 57 |
| 50 (80%) | 75 | +10.39 | [+8.22, +12.55] | 66 |
| 100 (100%) | 75 | +9.42 | [+7.40, +11.45] | 66 |
| 2026 real slates, aggression 50 | 4 | +1.52 | [-6.98, +10.01] | 3 |

By band at aggression 50, backtest: top 1 118.4, top 5 116.4, ranks 6-10
112.4, ranks 11-25 106.0. The drop is past rank ten. Every season is positive
at aggression 50 (2022 the weakest, +5.7). The kill rule (gap under 5, or
ahead in fewer than two thirds of weeks) is cleared at every aggression.

The real slates do not confirm it: four weeks, ranks 6-10 actually above the
top five (99.6 against 96.5). Four weeks cannot separate +10 from 0, and the
rank correlation there is a third of the backtest's, which is the proxy-salary
caveat showing. The production record of the lineups actually entered points
the other way: `/optimizer/performance` `depth`, the week's builds deep enough
to have a rank 11, reads +60.4 (week 6), +23.0 (week 7), +14.1 (week 8). Those
builds mixed objectives and settings, so they are corroboration, not a test.

**Reading.** For cash, the lineups past the first ten have cost about ten
points each in five seasons of backtest. So entering a duplicate of an early
lineup beats entering lineup 20 on expected score. The default dedupe
(2026-11-09) is right for a tournament and works against this in cash.
**No default changed**; the real-slate arm needs more weeks, and the `depth`
block now accrues it every week.

### Selection tax: real, and shrinking toward trailing form does not pay

The tax itself, on the backtest cash builds: the board's total for the chosen
eight beats what they scored by **+18.4 points a lineup**. Maximising a noisy
estimate picks the players whose noise ran positive; this is the size of it.

Slope stage, (actual - board) on (trailing-5 - board), players projected 8+,
CI clustered by week:

| | n | slope | 95% CI |
|---|--:|--:|---|
| 2026 live record | 758 | +0.285 | [-0.213, +0.784] |
| backtest 2021-2025 | 8,423 | +0.087 | [+0.005, +0.168] |
| by season | | +0.12, +0.03, +0.02, +0.22, -0.04 | |

The backtest scrapes past zero; three of five seasons straddle it. (Before the
duplicates were dropped it read +0.208, a measure of what they were doing.)

Payment stage: w fitted leaving one season out (0.05-0.10), applied to the 8+
players only.

| | board | adjusted | paired | 95% CI |
|---|--:|--:|--:|---|
| per-player MAE | 6.416 | 6.418 | +0.002 | [-0.003, +0.007] |
| top-5 lineup actual | 116.38 | 114.72 | **-1.66** | [-2.83, -0.49] |
| all 25 lineups | 109.35 | 109.33 | -0.02 | [-0.83, +0.79] |

**Killed.** The kill rule asked for +3 a lineup; the adjustment costs 1.7, and
the interval excludes zero on the wrong side. The board's form blend
(`blend.py`) already holds the weight trailing form deserves, and more of it
only reshuffles the near-ties toward worse ones. The tax is real and the
cure is not this; the cheap defence against it is not reading a lineup's
projected total as its expected score.

### Found on the way: production scores two FanDuel rules differently

`NFL-API/app/api/utils.py` `fanduel_points` charges -2 for every fumble and
pays nothing for a two-point conversion. FanDuel charges only fumbles lost and
pays +2 for a two-point conversion. On 2026 weeks 1-9, 151 of 2,872 scored
player-weeks differ by +2 and 15 by more, against nflverse scored through
`nfl_projections.opportunity._points`, which uses FanDuel's rules. The
training target (`dataset.py`) uses production's rule, so the system is
internally consistent and differs from the site; `opportunity._points` is the
odd one out inside the package. Changing it moves the training target and
every frozen accuracy and lineup record, so it is noted, not changed.

## FanDuel scoring corrected: lost fumbles and two-point conversions (2026-11-12)

Fixed, on the user's instruction, after the depth entry above found it.
FanDuel charges -2 for a fumble **lost** and pays +2 for a two-point
conversion. The training target (`dataset.py`) and production's scorer
(`NFL-API api/utils.fanduel_points`) charged every fumble and paid nothing for
a conversion; inside the package `opportunity` and `scarcity` had it right and
`fpa` charged lost fumbles but skipped sack fumbles and conversions. Four
versions of one rule.

Now there is one: `scoring.fanduel_points_from_stats`, used by the training
target, `opportunity`, `scarcity` and `fpa`. NFL-API mirrors it, and a test
there scores one row both ways and requires them to agree. Neither side falls
back to the all-fumbles columns when the lost ones are missing, since that
would restore the old rule without saying so.

**What moved in the training target**, 68,486 skill-position rows of the
local dataset:

| | rows changed | mean shift | share of rows |
|---|--:|--:|--:|
| all | 5,230 | +0.102 | 7.6% |
| QB | | **+0.607** | **31.3%** |
| RB | | +0.075 | 6.9% |
| WR | | +0.039 | 4.5% |
| TE | | +0.042 | 4.3% |

Where a row moves it moves by 1.5 points on average, almost all +2 (a
recovered fumble no longer charged, or a conversion paid) and +4. The handful
of ±0.1-0.2 differences are nflverse stat revisions since the CSV was built,
not the rule. Quarterbacks carry most of it: sack fumbles their team recovers
and two-point throws. Not a model change to A/B: a correction of what the
model is asked to predict, like #1's interception column. The production
model picks it up the next time it trains rather than reusing the cache,
which is the first run after week 10's games.

**The frozen records.** The rule is stamped on every row now
(`scoring_rule`, `api.utils.SCORING_RULE`). The daily accuracy and lineup jobs
re-score anything stamped with an older rule or none, and they re-score the
**actual** side only: actual points, error, band, the naive baseline and the
lineup baselines. Frozen projections are never touched, which is the promise
the frozen tables exist to keep. On 2026 weeks 1-9, 166 of 2,872 scored
player-weeks change, 151 of them by +2. The shadow-model grade reads those
same rows, so it follows. The projection scorer's old `--rescore` deletes and
rebuilds rows from current projections and is still not the tool for this.

Not changed in that fix: return touchdowns. FanDuel pays a skill player's
kick, punt or fumble-recovery return TD, and no scorer here counted
`special_teams_tds`. Added the next day; see below.

**Return and fumble-recovery touchdowns (2026-11-13).** Added on the user's
instruction. FanDuel pays +6 for each. nflverse holds them in two columns that
never overlap on 2018-2026 skill rows: `special_teams_tds` (kick, punt and
blocked-kick returns; 96 player-weeks) and `fumble_recovery_tds` (falling on
a teammate's loose ball in the end zone; 19). nflverse's own `fantasy_points`
counts the first and not the second, which is why some of those rows show
0.0 for a touchdown. Both are now in `fanduel_points_from_stats` and in
NFL-API's mirror, and the cross-repo test row carries both.

Size: 9-21 skill player-weeks a season out of ~5,800 (0.2%), 86 of 115 of
them WRs. On the 2026 scored record, 2 of 2,872 rows move, +6 each (weeks 3
and 8, both return men). Too small to move an MAE; it is here so an actual
is what FanDuel paid.

Plumbing: NFL-API's `player_stats` table had `special_teams_tds` but no
`fumble_recovery_tds`; the column is added by the light migration and filled
by the loader's forced current-season reload. Rows from earlier seasons stay
null and count as zero, which only matters for those seasons' FPPG history.
`SCORING_RULE` is now `fanduel-2026-11-13`, so the daily jobs re-score every
record's actual side once more. `record_signals.py` had a fifth private copy of
the rule (no return TDs, its own column choices) and now calls the package's.

## Player identity: names were doing an id's job (2026-11-13)

Found by reading the week 10 FantasyGuru position articles against the board:
the disagreements that were too large to be opinion (James Cook projected 9.3
after 15-25 every week, Kyle Pitts 2.5 after five straight games of 5+
targets) were our joins, not their reads. Fixed on the user's instruction:
"use player id and ensure we are mapping stats to the appropriate players".

**Depth-chart roles (shipped).** `DepthChartAnalyzer` keyed roles by display
name; nflverse's chart carries suffixes the roster lacks ("James Cook III",
"Kyle Pitts Sr.", "Aaron Jones Sr."). 14 rostered skill players in week 10 had
a chart row by gsis id and missed it by name, fell to `_UNKNOWN_RANK` = 3, and
were discounted as backups: RB x0.55, TE x0.30. The fuzzy fallback (last name +
first initial) also handed Bijan Robinson's RB1 rank to his teammate Brian
Robinson (RB3). On the 2026 record those 15 players were under-projected by
3.25 points a game against 1.39 for the board (n=64); Cook by 11.7 a game over
seven weeks, Minshew 11.3, Pitts 5.4. Roles are now looked up by gsis id, then
by suffix-free name on the same team, and a name never bridges two different
known ids. The draft-slot lookup (rookie prior) was an unanchored regex
`contains`; now id, then exact suffix-free name.

**One row per player (shipped).** Rosters and projections de-duplicated by
NAME: one player under two spellings came out as two rows with one id, and the
simulator's `float(sm.loc[pid])` killed the 2026-11-13 Friday refresh (NFL-API
guards it too). Two players sharing a name ("Justin Jefferson", MIN WR and CLE
LB) were one. Both are keyed by id now.

**FanDuel slates (shipped).** No shared id exists, so `names.Matcher` is now
the one matcher for everything that reads a slate (optimizer, /dfs/targets,
ownership, lineup scoring, /dfs salaries): name + team, then a name unique on
the board, then a unique last name at the position on the team with agreeing
first names. FanDuel spells "Tre' Harris", "Joshua Palmer", "Matt Hibner" and
writes JAC/LAR; all missed before. In lineup scoring a miss became "his team
played, so he scored 0" -- a false zero in the FPPG and best-possible
baselines. Unmatched stays visible as no projection; it is never guessed.

**Training data (held for the backtest).** `build_dataset` joined snap counts
on display name: 78 of 1,255 2026 player-weeks with 5+ looks had no snaps
(Burden III, Metcalf, Gordon II, Fannin Jr., DJ Moore), and the CLE linebacker
took the MIN receiver's 63 week-5 snaps. The pre-2025 depth-chart merge
multiplied rows (one per formation slot): ~6,000 duplicated player-weeks a
season in 2018-2024 -- the source of the cached-board duplicates logged above.
The fix (snaps by pfr_id -> gsis_id through `load_players`, 99.7% coverage;
one depth row per player-week) changes what the model trains on, so it goes
through the standard backtest before it ships.

**Backtest (2025 wk 1-18, train <2025, 60 epochs, both arms on the corrected
FanDuel target, paired on 5,880 player-weeks):**

| arm | rows | duplicated player-weeks | MAE 1 seed | MAE 3 seeds | corr 3 seeds |
|---|--:|--:|--:|--:|--:|
| name joins (as shipped) | 198,879 | 24,500 | 4.194 | 4.165 | 0.635 |
| id joins | 174,379 | 0 | 4.212 | **4.165** | **0.640** |

Paired 3-seed difference **+0.000, 95% CI by week [-0.017, +0.015]**, the id
arm better in 9 of 18 weeks; by position QB +0.028, RB +0.006, WR -0.021,
TE +0.021, none outside noise. The single-seed gap (+0.018) was seed noise:
each arm moved ~0.03-0.05 between one seed and three. A tie on accuracy, a
slightly better ranking, and correct data. **Kept.** 4.165 (3 seeds, corrected
target, id joins) is the new reference for the reruns that follow.

Note for the ledger as a whole: every model experiment before this one loaded
`data/nfl_dataset.csv` (built 2026-10-28) or a board cached from it, so it ran
on the name joins, the duplicated rows and the old target. A/B deltas mostly
share that flaw across both arms; the ones at risk are tests OF snaps, roles
and depth, thin-margin closures, and lineup results on real slates. Those are
being rerun on the clean data, each logged against its original.

## Reruns on clean data: what the identity bugs did to earlier results (2026-11-13)

Every model experiment before 2026-11-13 ran on `data/nfl_dataset.csv` as built
2026-10-28 (snap counts joined by name, 24,500 duplicated pre-2025
player-weeks, the old FanDuel target) or on boards cached from it, and the
role experiments ran production's name-keyed depth lookup. The dataset and the
2021-25 boards were rebuilt clean (174,379 rows, zero duplicated player-weeks;
the stale ones kept in `experiments/_cache/pre_idfix/`), and the experiments at
risk are rerun here against their originals. Order: role and promotion, usage,
training window, lineup record, quarterback findings, then spot checks of the
closed avenues. "Held" is a result too.

**1. The starter is out (`starter_out.py`) -- HELD, sample was inflated.** The
harness decided who played by matching snap counts to stats on a normalised
name, so a starter the two feeds spell differently read as absent in weeks he
played. Availability is now by gsis id (pfr id through `load_players`).

| | original | id-matched |
|---|--:|--:|
| entering starter missed | 16.3% | **14.4%** |
| RB2 usage, starter in / out | 8.28 / 16.11 (x1.95) | 8.39 / 15.91 (x1.90) |
| QB2 usage factor | x5.06 | x4.51 |
| out-of-sample factor RB / QB / WR / TE | 1.42 / 1.29 / 1.00 / 1.10 | **1.38 / 1.33 / 1.00 / 1.00** |

About one "starter out" week in eight was a name mismatch. The conclusion
stands: an extra factor helps at RB and QB, not WR or TE. The refit moves the
factors ~3%, inside what this harness can resolve; constants unchanged. (The RB
factor was later set to 1.0 end to end; that entry is rerun next.)

**3. Snap share as the role multiplier, #19 (`experiments/role_multiplier_rerun.py`,
new; the original ran inline) -- HELD, one diagnostic weaker.** Clean 2024
board, 5,714 rows, 5,345 with a depth rank. The snap-share arm is a
reconstruction (trailing five-game offensive snap share; the original did not
record its form).

| actual / raw prediction | RB2 | RB3 | WR2 | WR3 | TE2 | TE3 | QB2 |
|---|--:|--:|--:|--:|--:|--:|--:|
| original | 1.12 | 1.00 | 1.09 | 1.03 | 1.01 | -- | 0.82 |
| clean | 1.03 | 0.96 | 0.96 | 1.09 | 0.93 | 0.88 | 0.84 |
| multiplier applied | 0.80 | 0.55 | 0.92 | 0.72 | 0.55 | 0.30 | 0.35 |

| arm | MAE | bias | spearman QB / RB / TE / WR |
|---|--:|--:|---|
| none | 4.281 | -0.07 | 0.478 / 0.703 / 0.591 / 0.636 |
| rank multiplier (shipped) | **4.215** | -0.74 | 0.496 / 0.696 / 0.565 / 0.638 |
| snap share | 4.277 | -2.12 | 0.471 / 0.685 / 0.560 / 0.633 |

Rank multiplier vs none -0.066 MAE [-0.123, -0.015] (original -0.052); snap
share vs none -0.003 [-0.074, +0.059]. Snap share stays rejected, more firmly:
no MAE gain now and no better ordering. The double-counting diagnostic is
weaker: on clean data the raw model is roughly calibrated for backups (0.84-1.09)
rather than under-projecting them, though the multiplier still cuts 20-70%,
far more than those ratios ask for. Spearman here is per position over
player-weeks; the original's ~0.94 was a different population and is not
comparable in level. No change.

**2. The promotion fix, end to end (`promotion_endtoend.py`) -- HELD on every
line.** The harness's oracle (who played), its practice-report trigger and
production's depth lookup now all key on gsis id; 5-seed history models
retrained on the clean dataset; 2025 wk 1-18 plus 2026 wk 1-9 (28 weeks,
8,569 player-weeks; the original had 23 weeks, 7,334).

A first pass was itself contaminated and is discarded: the harness's cached
2026 snap, injury and roster files dated from 2026-10-28 and stopped at week
7-8, so weeks 8-9 were scored with no oracle and no week-9 report (it showed
51 "promotions whose superior played", all in those two weeks). Stale caches
are a contamination class of their own; the 2026 files were refreshed.

| | original | clean |
|---|--:|--:|
| production trigger precision / recall | 1.000 / 0.690 | **1.000 / 0.751** |
| parameter-free lift vs pre-fix, the promoted | +6.10% | **+7.10%** |
| shipped RB x1.42 vs parameter-free, promoted RBs | -6.1% [-10.6, -2.1] | **-6.1% [-9.6, -3.0]** |
| QB x1.29 vs parameter-free | +2.4% [-6.0, +10.4] | +5.7% [-0.6, +12.7] |
| WR re-ranking vs pre-fix | +2.1% [+0.9, +3.3] | +3.2% [+2.2, +4.1] |
| TE re-ranking vs pre-fix | -1.5% [-8.4, +4.3] | -0.8% [-7.1, +5.2] |
| placebo (absences below) vs pre-fix | -7.25% | -7.19% |

`PROMOTION_FACTOR` RB 1.0 stands; QB 1.29 stands (positive, still unresolved
at n=69). No change.

**4. Usage decomposition, #31 (`usage_share.py`) -- UNAFFECTED, not rerun.**
Raw target and carry counts with depth ranks joined by gsis id; it reads
neither the dataset nor fantasy points.

**5. Opportunity share, #53 (`opportunity_share.py`) -- DECISION HELD, two
findings moved.** Clean boards 2021-25: 21,014 scored player-weeks, board MAE
4.4026 (original 27,297 and 4.1935: the duplicated rows were disproportionately
low-scoring depth players, which flattered MAE).

| | original | clean |
|---|--:|--:|
| volume-TD gap slope RB / WR / TE | +0.61 / +0.63 / +0.72 | +0.40 / +0.38 / +0.69 |
| gap correction, MAE gain WR / TE / RB | +0.33% / +0.46% / +0.20% | **+0.07% / +0.31% / -0.06%**, all CIs span zero |
| opp share slope RB / WR | +0.10 / +0.06 (null) | **+0.34 [+0.03, +0.65] / +0.63 [+0.06, +1.23]** |
| opp share correction, MAE gain WR | +0.14% | **+1.43%, 74/75 weeks** |
| opp share correction, MAE gain TE | +1.95% | **+2.56%, 71/75 weeks** |
| xFP correction, MAE gain WR / TE | null | +0.50% / +1.22% |
| WR opp share, top-24 overlap | +0.0020 [+0.0005, +0.0040] | +0.0022 [-0.0011, +0.0061] |

The decision (context beside `value`, never in the objective) HOLDS and is if
anything stronger: no term moves the order anywhere, and the one ordering gain
the original reported no longer clears zero. The touchdown-luck finding
weakens: the board still leans on recent touchdowns (slopes positive) but
correcting it no longer pays. What is new is that **volume itself is now under-
used by the board for MAE** -- opportunity share pays 1.4% at WR and 2.6% at TE,
nearly every week. That is a projection-accuracy lead for /projections (not for
lineups), and it goes through the standard backtest before anything ships.

**6. Blitz rate against quarterback output, #51 (`blitz_qb.py`, `BLITZ_BOARD=1`)
-- HELD.** No name joins in the harness; rerun on the clean 2021-25 boards
(25,791 scored player-weeks, board MAE 4.3322; original 31,574 and 4.1791).
Own-mean slopes match closely (RB fp -0.527 vs -0.541, WR +0.211 vs +0.219).

| | original | clean |
|---|---|---|
| board-error slope RB | -0.284 [-0.458, -0.128] | -0.243 [-0.425, -0.079] |
| board-error slope WR | +0.136 [+0.022, +0.256] | +0.129 [+0.014, +0.240] |
| slope-only MAE gain, every position | ~0, all CIs span 0 | ~0 (QB -0.0040 [-0.0084, -0.0003]) |
| ordering, QB / WR | -0.0045 / -0.0044, CIs below 0 | +0.0000 / +0.0006, CIs span 0 |

The information is still in the board's error and still worth nothing applied.
The one softening is that it no longer actively damages the order; the
verdict (not built) rested on "worth nothing" and stands.

**7. Closed avenues, spot checks -- NONE NEEDED A RERUN, for stated reasons.**
- Coverage matchup #28 (`coverage_wk1.py`) and opponent-adjusted allowed #12
  (`fpa_adjusted.py`): raw nflverse yards and receptions (prop markets) with
  gsis-keyed participation; no dataset, board or FanDuel points. Unaffected.
- FPA adjusted for schedule #76: built from raw stats by player id through
  `fpa.py`; only the old scoring touched it, its verdict was already a wash
  and it stays display-only. Not rerun.
- Vegas implied total #16: tested the line as a model *feature*, which the
  standing rule now forbids; no rerun could produce a usable result.
- Per-position models #17: not a thin margin (shared model ahead by 0.11 WR,
  0.18 TE, tie QB) and both arms shared the data; not rerun.
The bugs were in volume, roles and the target; the matchup harnesses never
touched them.

**8. Training on the current season alone, #39 weekly retrain
(`train_window_expanding.py`, SEEDS=3) -- FLIPPED.** Stopped by
the host for low memory after week 14; weeks 15-18 still to run. 2025, each
week retrained on that season's prior weeks only, against one model trained on
all prior seasons, identical player-weeks:

| week | current rows | current only | full history | delta |
|--:|--:|--:|--:|--:|
| 5 | 4,390 | 4.202 | 4.085 | +0.117 |
| 6 | 5,351 | 4.243 | 4.225 | +0.018 |
| 7 | 6,365 | 4.387 | 4.399 | -0.013 |
| 8 | 7,400 | 4.380 | 4.395 | -0.015 |
| 9 | 8,292 | 4.052 | 4.069 | -0.017 |
| 10 | 9,240 | 4.158 | 4.198 | -0.040 |
| 11 | 10,203 | 4.055 | 4.104 | -0.048 |
| 12 | 11,235 | 3.943 | 4.071 | -0.127 |
| 13 | 12,202 | 3.786 | 3.939 | -0.153 |
| 14 | 13,283 | 4.022 | 4.016 | +0.006 |
| pooled 5-14 | | **4.118** | **4.147** | **-0.029** (n=3,158) |

The original (one seed, name-joined data, old target) had current-only worse
in 12 of 14 weeks, +0.068 pooled, "never closes". Clean, it trails at week 5,
catches up by week 7 and leads through week 13. That supports the live
model's current-season-only window (the user's call, 2026-10-07) rather than
contradicting it.

Weeks 15-18, run separately after the host stopped the first pass:

| week | current rows | current only | full history | delta |
|--:|--:|--:|--:|--:|
| 15 | 14,257 | 4.384 | 4.373 | +0.011 |
| 16 | 15,312 | 4.273 | 4.315 | -0.042 |
| 17 | 16,410 | 4.103 | 4.136 | -0.034 |
| 18 | 17,473 | 3.888 | 3.908 | -0.020 |

**Final, weeks 5-18, 4,554 player-weeks (the original's exact population
size): current-only 4.132 vs full history 4.158, -0.026; current-only wins 10
of 14 weeks; mean weekly delta -0.026, 95% CI by week [-0.060, +0.009].**
The original read +0.068 and 2 of 14. **FLIPPED in direction**; the interval
still touches zero, so the honest reading is "current-season-only is at least
as good after the first few weeks", which is what the live model already does.
Two differences between the runs, not one: clean data and corrected target,
and 3 seeds against 1. The original's "never closes" claim is withdrawn.
