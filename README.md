# nfl-data-py

NFL fantasy projections pipeline. Builds a historical dataset from
[nflreadpy](https://github.com/nflverse/nflreadpy), trains a neural network to
project next-week player stats and FanDuel fantasy points, layers in depth
chart roles and free nflverse injury reports, and turns projections into DFS
player pools and lineups. Can also produce a **floor / median / ceiling** range
per player via a quantile model.

All the code lives in one package, `nfl_projections/`. (Earlier iterations —
`billb/`, `nfl_ai_work/`, `nickknowsnfl/` — were consolidated into it; see git
history if you need them.) Model experiments and their measured impact are
tracked in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Setup

```bash
pip install -r requirements.txt
```

Or install the package (e.g. to import `nfl_projections` from another service):

```bash
pip install "nfl-projections @ git+https://github.com/Gin-G/nfl-data-py.git"
```

Injury data comes free from nflverse (no API key). A Sportradar key is optional
(`export SPORTRADAR_API_KEY=...`, then `--injury-source sportradar`).

## Use it as a library

Import the projections into your own code. A `ProjectionService` loads the
dataset and trains (or loads) the model once; project as many weeks/players as
you like afterwards:

```python
from nfl_projections import ProjectionService

svc = ProjectionService(quantiles=True)            # or model_dir=/quantile_model_dir=
frame = svc.project(2025, 3, as_frame=True)        # DataFrame, best first
#   -> player_name, team, fanduel_fantasy_points, floor, projection_median, ceiling, ...
rbs   = svc.project(2025, 3, positions=["RB"])     # {position: DataFrame}
bijan = svc.project_player("Bijan Robinson", 2025, 3)   # dict, or None

# One-shot (builds a service, projects one week):
from nfl_projections import project_week
df = project_week(2025, 3, quantiles=True)
```

Importing the package does **not** load TensorFlow — that happens only when you
train or predict.

## Weekly workflow

```bash
# 1. Build/refresh the historical dataset (2018-present) -> data/nfl_dataset.csv
python -m nfl_projections build-data

# 2. Project a week (trains the model, then predicts every rostered QB/RB/WR/TE)
python -m nfl_projections predict --season 2025 --week 13

# ...with a floor/median/ceiling range per player
python -m nfl_projections predict --season 2025 --week 13 --quantiles

# 3. Build DFS player pools from those predictions
python -m nfl_projections pools --season 2025 --week 13
```

Predictions land in `predictions/`, pools in `player_pools/`.

Useful variations:

```bash
# Only some positions or specific players
python -m nfl_projections predict --season 2025 --week 13 --positions RB WR
python -m nfl_projections predict --season 2025 --week 13 --players "Bijan Robinson" "CeeDee Lamb"

# Train once, reuse the saved model for multiple weeks
python -m nfl_projections train --save-dir models
python -m nfl_projections predict --season 2025 --week 13 --model-dir models

# Skip injury adjustments, or use Sportradar instead of nflverse
python -m nfl_projections predict --season 2025 --week 13 --no-injuries
python -m nfl_projections predict --season 2025 --week 13 --injury-source sportradar
```

## Backtesting and comparing to other projections

Backtest trains on seasons *before* the target season (no data leakage), then
projects each week from what was known at the time and scores it against what
actually happened:

```bash
python -m nfl_projections backtest --season 2024 --weeks 1-17 --output backtest_2024.csv
```

This prints MAE / RMSE / correlation overall, by position, and by week.

To see how we stack up against someone else's free projections (e.g. a
FantasyPros CSV export), match on player name and compare both sources:

```bash
python -m nfl_projections compare --results backtest_2024.csv --external fantasypros_week10.csv
```

When the results file has actuals (backtest output does), it reports each
source's error against reality and who was closer. Column names in the
external CSV are auto-detected (`Player`/`FPTS` etc.); override with
`--name-col` / `--points-col` if needed.

## Lineup optimization

End-to-end from a FanDuel salary export (projects the week, merges, and builds
lineups in one step):

```bash
python -m nfl_projections optimize --fanduel FanDuel-players-list.csv \
    --season 2025 --week 3 --objective ceiling --lineups 10
```

`--objective` picks what each lineup maximizes: `mean` (default, expected
points), `ceiling` (tournament upside), `floor` (cash-game safety), or `median`.
The non-mean options use the quantile model, which is trained automatically.
Lineups print with their floor–ceiling range.

From Python:

```python
from nfl_projections import optimize_week

lineups, merged = optimize_week(2025, 3, "FanDuel-players-list.csv",
                                objective="ceiling", num_lineups=10)
```

Or keep the two steps separate with a pre-merged CSV:

```python
from nfl_projections import optimizer
import pandas as pd

merged = optimizer.merge_fanduel_salaries(
    pd.read_csv("FanDuel-players-list.csv"),
    pd.read_csv("predictions/all_predictions.csv"),   # or ProjectionService output
)
merged.to_csv("merged.csv", index=False)
```

```bash
python -m nfl_projections optimize --csv merged.csv --objective floor --lineups 10
```

## Package layout

| Module | Purpose |
|---|---|
| `nfl_projections/config.py` | Seasons, positions, paths, FanDuel scoring rules |
| `nfl_projections/dataset.py` | Build the historical dataset from nflreadpy |
| `nfl_projections/scoring.py` | FanDuel fantasy point calculation |
| `nfl_projections/features.py` | Feature engineering, target creation, rookie detection |
| `nfl_projections/model.py` | Network definition, training, save/load, batch predict |
| `nfl_projections/quantiles.py` | Floor/median/ceiling quantile model |
| `nfl_projections/injuries.py` | nflverse (default) / Sportradar injuries + backup elevation |
| `nfl_projections/opponent.py` | Opponent-defense matchup features (opt-in) |
| `nfl_projections/pbp.py` | Play-by-play scheme splits + usage tendencies (opt-in) |
| `nfl_projections/predict.py` | Weekly projection engine (Projector) |
| `nfl_projections/service.py` | Importable API: `ProjectionService`, `project_week` |
| `nfl_projections/pools.py` | DFS player pool CSVs |
| `nfl_projections/optimizer.py` | FanDuel lineup builder |
| `nfl_projections/evaluate.py` | Season backtests, quantile calibration, external comparison |
| `nfl_projections/cli.py` | `python -m nfl_projections ...` commands |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The test suite covers the pure logic (scoring, feature engineering, rookie
detection, injury/depth-chart analysis, comparisons) and runs without
tensorflow or network access.
