# nfl-data-py

NFL fantasy projections pipeline. Builds a historical dataset from
[nflreadpy](https://github.com/nflverse/nflreadpy), trains a neural network to
project next-week player stats and FanDuel fantasy points, layers in depth
chart roles and live Sportradar injury data, and turns projections into DFS
player pools and lineups.

All the code lives in one package, `nfl_projections/`. (Earlier iterations —
`billb/`, `nfl_ai_work/`, `nickknowsnfl/` — were consolidated into it; see git
history if you need them.)

## Setup

```bash
pip install -r requirements.txt

# Optional: live injury adjustments
export SPORTRADAR_API_KEY='your_key_here'
```

## Weekly workflow

```bash
# 1. Build/refresh the historical dataset (2018-present) -> data/nfl_dataset.csv
python -m nfl_projections build-data

# 2. Project a week (trains the model, then predicts every rostered QB/RB/WR/TE)
python -m nfl_projections predict --season 2025 --week 13

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

# Skip Sportradar injury calls (e.g. no API key)
python -m nfl_projections predict --season 2025 --week 13 --no-injuries
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

Merge a FanDuel salary export with projections and build lineups:

```python
from nfl_projections import optimizer
import pandas as pd

merged = optimizer.merge_fanduel_salaries(
    pd.read_csv("FanDuel-players-list.csv"),
    pd.read_csv("predictions/all_predictions.csv"),
)
merged.to_csv("merged.csv", index=False)
```

```bash
python -m nfl_projections optimize --csv merged.csv --lineups 10 --salary-cap 60000
```

## Package layout

| Module | Purpose |
|---|---|
| `nfl_projections/config.py` | Seasons, positions, paths, FanDuel scoring rules |
| `nfl_projections/dataset.py` | Build the historical dataset from nflreadpy |
| `nfl_projections/scoring.py` | FanDuel fantasy point calculation |
| `nfl_projections/features.py` | Feature engineering, target creation, rookie detection |
| `nfl_projections/model.py` | Network definition, training, save/load, batch predict |
| `nfl_projections/injuries.py` | Sportradar injury fetch + backup elevation |
| `nfl_projections/predict.py` | Weekly projection engine (Projector) |
| `nfl_projections/pools.py` | DFS player pool CSVs |
| `nfl_projections/optimizer.py` | FanDuel lineup builder |
| `nfl_projections/evaluate.py` | Season backtests + external comparison |
| `nfl_projections/cli.py` | `python -m nfl_projections ...` commands |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The test suite covers the pure logic (scoring, feature engineering, rookie
detection, injury/depth-chart analysis, comparisons) and runs without
tensorflow or network access.
