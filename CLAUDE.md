# nfl-data-py — working notes

The `nfl_projections` package: the projection engine behind the NFL-API service
and the FanDuel optimizer. This file is the stuff that is **not** inferable from
reading the code. The code explains itself; these are the traps.

## Running anything

**One interpreter works.** The system Python's sklearn/numpy pairing is broken:

```bash
NP=/home/gin-g/miniforge3/pkgs/numpy-1.26.4-py310hb13e2d6_0/lib/python3.10/site-packages
PYTHONPATH=.:$NP /home/gin-g/miniforge3/bin/python -m pytest tests/ -q
```

There is **no network for pip**, so do not try to install the fix. `$NP` is a
numpy 1.26 shim on the path ahead of the broken one; drop it and imports fail.

A full run is ~20 s and ~766 tests. Run the whole suite, not a subset — several
of the interesting failures are cross-module.

## CI resolves different versions than this machine

Local pandas is **2.3.3**; CI installs latest, currently **pandas 3.x**. That
difference has already shipped one bug: a list of strings-and-`None`s assigned
to a DataFrame column infers the new `str` dtype on pandas 3 and turns the
`None` into `NaN`, so a test asserting `None` passed locally and failed in CI
for two days. When a column's null is part of its contract, construct it with
`pd.Series(..., dtype=object)` and **assert the dtype**, not just the values.

CI installs only `pandas numpy requests joblib pytest pytest-cov` plus the
package with `--no-deps`. Anything a test imports beyond that has to be added to
`.github/workflows/tests.yaml` or it goes red on push only.

## The rules that are not negotiable

These come from the user and from measured results. Do not relitigate them.

- **No Vegas lines as regressors.** A line may be a benchmark, never an input to
  an edge-hunting test.
- **No manual overrides on named players.** The board stands on the model; a
  named player is a symptom, never a target. Several modules carry a test that
  renames every player and asserts the output is identical — keep that pattern.
- **Matchup/coverage context never enters the optimizer's objective.** It is
  measured as null against the board's own error. Show it, never score it.
- **Nothing feeds `/projections` until the grading gate passes**
  (`scripts/grade_external.py`, which lives in the **NFL-API** repo under
  `app/scripts/`, not here).
- **No scraping from FantasyGuru.** Their terms are personal use only. The only
  route in is a human uploading a CSV. `external.py` imports no network library
  and has a test that fails if one is added.

## EXPERIMENTS.md is the ledger — and it is ~6,000 lines

Every test goes in it **unprompted**, negative results especially. It is the
single most valuable artifact here and it is too long to read whole: search it
before proposing anything, because a surprising number of good ideas are already
in there as measured failures. Known-closed avenues include matchup features,
Vegas, per-position models, volatility features for widening the quantile bands,
and ownership tilt as an improvement rather than a trade.

`ROADMAP.md` is the forward-looking companion.

## The standard yardstick

Model changes get the leakage-free backtest (2025 wk1–18, train <2025, epochs
60) against the **4.187 MAE** / 0.611 corr baseline (4.17 was measured with
postseason weeks in the population; see the correction at the top of the
ledger). **Single-seed deltas under
~0.05 MAE are noise.** Log it; revert what loses.

For a feature claimed to beat the board, the residual test is the method: a
slope stage (regress board residual on the candidate, 95% CI) then a payment
stage (leave-one-out fit, intercept dropped, paired MAE CI). State the power —
a null at n=100 is "inconclusive", not "no".

## Facts that cost time to rediscover

- `nflreadpy.load_participation` covers **2016–2025 and stopped**. It is the
  only home of coverage shell, man/zone, route and pressure data. No
  current-season substitute exists; `coverage.shell_mix` falls back and stamps
  the staleness onto every row.
- The board **does not project defenses**: `/projections` and `/dfs/targets`
  hold no DEF rows. The optimizer's DEF slot is a separate thing and is *not*
  on FPPG any more — NFL-API attaches `defense.project` to the slate and falls
  back to FanDuel's FPPG per row only where that is missing.
- `parse_nflverse_injuries` zeroes a player only on `report_status`, which clubs
  file on the **final Friday** report. The Tue+Wed projection runs therefore
  never see a game designation. The Fri/Sat/Sun refresh cron that fixes this
  was written 2026-10-28 and **had never run as of 2026-11-09** — it sat in
  NFL-API's hand-applied `k8s/` directory. It is in the helm chart from
  2026-11-10; before trusting a weekend board, check `computed_at` on
  `GET /projections/runs` rather than assuming the refresh fired.
- The quantile band is ~2.1× the mean (spearman 0.9949), so the optimizer's
  `ceiling` objective **is** its `mean` objective. Only `floor` reorders.
- `evaluate.backtest` skips `roles.py` entirely; `promotion_endtoend.py` is the
  harness that does not, 2025+ only.

## Shipping

Commit and push **only when asked**. The NFL-API image installs this package
from the GitHub main tarball, so **push here first** or that build tests against
the old code.
