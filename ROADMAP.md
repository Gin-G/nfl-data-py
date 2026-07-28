# Model improvement roadmap

Companion to [`EXPERIMENTS.md`](EXPERIMENTS.md) (the ledger of what was tried and
measured). This is the forward-looking list: validation gaps, new feature
avenues, and the honest priors for each, written after the first 13 experiments.

**Guiding finding from the ledger:** context features (opponent FPA — tested 3
ways, pbp scheme splits, volatility) all failed because single-game fantasy
variance dwarfs their signal, and a player's own recent usage already carries
the predictable part. New features can only win through two channels:
**(a) moving expected volume/opportunity** (the stable component), or
**(b) informing players the rolling features can't see** (rookies, role
changes, low-sample). Rank everything against that lens.

## Validation gaps (run these regardless)

1. **Skill-over-naive margin** — we've never measured MAE for "predict the
   rolling average" or "predict season average." That number is the honest
   denominator for every accuracy claim. Cheapest, most important test.
2. **External benchmark** — `evaluate.compare_to_external` supports
   FantasyPros-style CSVs and has never been run against real external
   projections. Are we competitive with industry at 4.17?
3. **GBDT baseline** — LightGBM/XGBoost on the same features. Tabular data this
   size usually favors GBDTs; if one hits ≤4.17 it's a better production model
   (seconds to train, no TF in the job image) and gives feature importances.
4. **Multi-seed + seed-ensembling** — all results are single-seed; 5 seeds give
   error bars, and averaging them is routinely worth 1–3% MAE.
5. **Per-position models** — QB (MAE ~7, corr ~0.4) is the weak spot with
   distinct dynamics; a dedicated QB model is the plausible first split.
6. **Prospective accuracy tracking** — NFL-API now stores projections with
   `computed_at`; score them against actuals when each week completes
   (`/projections/accuracy`). True out-of-sample eval accruing forever.
7. **Quantile floor calibration** — ceiling is calibrated, floor isn't (q10
   empirical 0.24). Untried: per-position conformal offsets; asymmetric
   trained quantiles (q02/q50/q90).

## New features, ranked

### Tier 1 — strong mechanism, free data, untested

- **Vegas lines** (`spread_line`, `total_line` are already in
  `nflreadpy.load_schedules()`). Implied team total is the market's aggregate
  of everything — injuries, matchup, weather, pace — and directly moves the
  expected scoring environment. Highest-expected-value untested feature.
- **xFP (expected fantasy points from opportunity)** via pbp: value each
  target by depth/field position, each carry by down-distance/goal-line.
  Rolling xFP is more stable than rolling actual FP. Companion: **TD luck**
  (actual − expected TDs, trailing) as a regression-candidate signal.
  `nfl_projections/pbp.py` already parses the raw inputs.
- **Role-change volume transfer (the Rico Dowdle fix)** — the worst backtest
  misses were all role changes. When injuries/depth charts elevate a player,
  build his feature row from team-position volume, not just his own thin
  history. Targets tail errors specifically.

### Tier 2 — coaching & player grades (real mechanisms, honest caveats)

- **Coaching features**: pace (plays/game), pass rate over expected
  (`pass_oe`, already in pbp.py), red-zone pass rate, and **regime-change
  flags** (new HC/play-caller, midseason firing). Legitimate channel — the
  coach sets *own-team volume* — but in stable weeks rolling usage already
  reflects it. Hypothesis to test: helps mainly at regime boundaries; measure
  MAE on regime-change weeks separately and gate the feature there if so.
- **Player grades** (NFL-API `EnhancedNFLPlayerGrader` /
  `player_season_analytics`): if derived from the same box scores, mostly
  redundant for veterans — but valuable as a **prior for low-information
  players** (rookies, elevated backups with 1–2 games), blended by games
  played. Genuinely new info in this space (PFF, routes run, YPRR) is paid;
  per-route receiving data would be the one paid source worth considering.

### Tier 3 — cheap adds, modest expectations

- Schedule spots: days rest, Thursday flag, post-bye, travel (from schedules).
- Weather/venue: `roof`, `temp`, `wind` (in schedules); wind >15mph suppresses
  passing. Vegas totals already price most of this in.
- Home/away alone, promoted out of the failed matchup bundle.

## Recommended order

1. Naive baseline + GBDT + multi-seed (one session; establishes the floor and
   maybe a better production model).
2. Vegas lines + schedule spots (data already loaded).
3. xFP / TD-luck rolling features.
4. Prospective accuracy tracking in NFL-API.
5. Role-change volume transfer.
6. Coaching regime features, then player-grade priors — tested against the
   regime/low-sample hypotheses so we learn *where* they help.

Same discipline as always: standard 2025 backtest, log in EXPERIMENTS.md,
revert what loses.
