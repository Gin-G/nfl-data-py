"""Central configuration for the projections pipeline."""

from datetime import date, timedelta


def current_season(today=None):
    """The NFL season in progress, by date: this calendar year from the Thursday
    after Labor Day, the previous year before it.

    Same rule as ``nflreadpy.get_current_season``, restated so importing config
    doesn't pull in nflreadpy. This used to be a hand-edited constant, bumped
    each August. 2026 week 1 was played with it still reading 2025, which kept
    every weekly retrain off the new season's games.
    """
    today = today or date.today()
    labor_day = date(today.year, 9, 1)
    labor_day += timedelta(days=(0 - labor_day.weekday()) % 7)   # first Monday
    return today.year if today >= labor_day + timedelta(days=3) else today.year - 1


# Seasons pulled into the historical dataset: through the season in progress.
# The current season's stats may not be published yet in its first days;
# dataset.build_dataset drops it then rather than failing.
CURRENT_SEASON = current_season()
SEASONS = list(range(2018, CURRENT_SEASON + 1))

# Only train on seasons from this year forward (older football is less relevant)
TRAINING_MIN_SEASON = 2020

# Networks averaged in the default seed ensemble. Measured (EXPERIMENTS.md,
# roadmap item 1): a single seed scores 4.237 +/- 0.052 MAE on the 2025
# backtest, the 5-seed ensemble 4.201 — a real gain, and it removes the seed
# lottery that made single-seed A/B deltas of +/-0.05 meaningless.
DEFAULT_N_SEEDS = 5

# Offensive skill positions we project
POSITIONS = ["QB", "RB", "WR", "TE"]

# Default file locations (all relative to the working directory)
DATA_DIR = "data"
DATASET_PATH = "data/nfl_dataset.csv"
PREDICTIONS_DIR = "predictions"
POOLS_DIR = "player_pools"
MODELS_DIR = "models"
LEARNING_CURVES_PATH = "predictions/learning_curves/enhanced_model_curves.png"

# FanDuel scoring rules
FANDUEL_SCORING = {
    "passing_yards": 0.04,
    "passing_tds": 4,
    "interceptions": -1,
    "passing_bonus_300": 3,  # Bonus for 300+ passing yards
    "rushing_yards": 0.1,
    "rushing_tds": 6,
    "rushing_bonus_100": 3,  # Bonus for 100+ rushing yards
    "receptions": 0.5,
    "receiving_yards": 0.1,
    "receiving_tds": 6,
    "receiving_bonus_100": 3,  # Bonus for 100+ receiving yards
    "fumbles": -2,
    "return_tds": 6,
    "two_point_conversions": 2,
    "field_goals_0_39": 3,
    "field_goals_40_49": 4,
    "field_goals_50_plus": 5,
    "extra_points": 1,
}

# Stats the model predicts for next week
TARGET_COLS = [
    "passing_yards", "passing_tds", "passing_interceptions",
    "rushing_yards", "rushing_tds",
    "receiving_yards", "receptions", "receiving_tds",
    "fanduel_fantasy_points",  # primary target
]
