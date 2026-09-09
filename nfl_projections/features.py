"""Feature engineering and training-data preparation (pure pandas/numpy)."""

import logging

import numpy as np
import pandas as pd

from . import config
from .utils import normalize_player_name, regular_games

logger = logging.getLogger(__name__)

# Feature groups pulled straight from the dataset when present
CORE_OFFENSIVE = [
    "passing_yards", "passing_tds", "passing_interceptions", "completions", "attempts",
    "rushing_yards", "rushing_tds", "carries", "rushing_fumbles", "rushing_fumbles_lost",
    "receiving_yards", "receiving_tds", "receptions", "targets", "receiving_fumbles",
    "receiving_fumbles_lost", "sacks", "sack_yards", "sack_fumbles", "sack_fumbles_lost",
]
ADVANCED_PASSING = [
    "passing_air_yards", "passing_yards_after_catch", "passing_first_downs",
    "passing_epa", "passing_2pt_conversions", "pacr", "dakota",
]
ADVANCED_RUSHING = ["rushing_first_downs", "rushing_epa", "rushing_2pt_conversions"]
ADVANCED_RECEIVING = [
    "receiving_air_yards", "receiving_yards_after_catch", "receiving_first_downs",
    "receiving_epa", "receiving_2pt_conversions", "racr", "target_share",
    "air_yards_share", "wopr",
]
SNAP_COUNT_FEATURES = [
    "offensive_snaps", "offensive_snap_pct",
    "defensive_snaps", "defensive_snap_pct",
    "special_teams_snaps", "special_teams_snap_pct",
    "total_snaps",
]
SPECIAL_MISC = ["special_teams_tds", "fantasy_points", "fantasy_points_ppr"]
PERFORMANCE_TRACKING = ["avg_fppg"]

# Features computed in add_derived_features
DERIVED_FEATURES = [
    "yards_per_attempt", "yards_per_carry", "yards_per_target",
    "completion_rate", "catch_rate",
    "fantasy_per_snap", "high_snap_count", "snap_role", "is_primary_player",
    "special_teams_player", "opportunity_score", "reduced_snaps",
    "epa_per_attempt", "epa_per_target", "avg_target_depth",
    "qb_passing_volume", "rb_total_touches", "wr_te_targets", "wr_te_snap_rate",
    "performance_vs_average", "is_consistent_performer", "above_season_average",
    "pass_heavy_script",
]

VALID_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"]

# Trailing-window (rolling) features: a player's recent level, usage, and trend.
# A single last game is noisy and makes the model regress to the mean; averaging
# the last few games gives a stronger signal of the player's true level/role.
ROLLING_BASE = [
    "fanduel_fantasy_points",
    "offensive_snap_pct",
    "targets", "carries", "receptions",
    "passing_yards", "rushing_yards", "receiving_yards",
    "attempts", "target_share",
    # pbp usage tendencies (present only when pbp features are merged in)
    "ten_outside_run_share", "ten_adot", "ten_deep_rate",
]
ROLLING_WINDOWS = (3, 5)
ROLLING_TREND = ["fppg_trend", "snap_trend"]


def rolling_feature_names():
    """Names of the columns add_rolling_features produces."""
    names = [f"{c}_roll{w}" for c in ROLLING_BASE for w in ROLLING_WINDOWS]
    return names + ROLLING_TREND


def add_rolling_features(df, windows=ROLLING_WINDOWS):
    """Add trailing-window means, usage, and trend per player.

    Each row gets the mean of the last N games (including the row's own game,
    which is known at projection time) for a curated set of level/volume/usage
    stats, plus short-vs-long trend features. Leakage-free: the target is the
    *next* game, and these only summarize the current game and earlier ones.
    """
    if "player_id" not in df.columns or "week" not in df.columns:
        return df.copy()

    out = df.copy()
    out["_week_num"] = pd.to_numeric(out["week"], errors="coerce")
    out = out.sort_values(["player_id", "season", "_week_num"])
    grouped = out.groupby("player_id", sort=False)

    for col in ROLLING_BASE:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        for w in windows:
            out[f"{col}_roll{w}"] = grouped[col].transform(
                lambda s, w=w: s.rolling(w, min_periods=1).mean()
            )

    fp, sp = "fanduel_fantasy_points", "offensive_snap_pct"
    if f"{fp}_roll3" in out.columns and f"{fp}_roll5" in out.columns:
        out["fppg_trend"] = out[f"{fp}_roll3"] - out[f"{fp}_roll5"]
    if f"{sp}_roll3" in out.columns and f"{sp}_roll5" in out.columns:
        out["snap_trend"] = out[f"{sp}_roll3"] - out[f"{sp}_roll5"]

    out = out.drop(columns=["_week_num"])
    fill_cols = [c for c in rolling_feature_names() if c in out.columns]
    out[fill_cols] = out[fill_cols].fillna(0)
    return out


def clean_training_data(df, min_season=config.TRAINING_MIN_SEASON, min_games=3,
                       drop_outliers=True):
    """Drop AVG rows, old seasons, and — for training — low-activity players
    and fantasy-point outliers.

    ``drop_outliers`` and ``min_games`` exist because the two callers want
    different things from this function. Training wants a trimmed frame: a
    freak 50-point game and a player with two career appearances both distort
    the fit. **Prediction does not**, and applying the trims there is a bug
    with a specific shape — see ``prepare_prediction_base``.
    """
    games = regular_games(df) if "week" in df.columns else df.copy()

    if min_games and min_games > 1:
        player_game_counts = games.groupby("player_id").size()
        active_players = player_game_counts[player_game_counts >= min_games].index
        games = games[games["player_id"].isin(active_players)]

    if drop_outliers:
        q99 = games["fanduel_fantasy_points"].quantile(0.99)
        q1 = games["fanduel_fantasy_points"].quantile(0.01)
        games = games[
            (games["fanduel_fantasy_points"] >= q1) & (games["fanduel_fantasy_points"] <= q99)
        ]

    if "season" in games.columns and min_season:
        games = games[games["season"] >= min_season]

    return games


def add_next_week_targets(df, target_cols=None):
    """Shift each target stat back one game so a row predicts its next week."""
    target_cols = target_cols or [c for c in config.TARGET_COLS if c in df.columns]
    out = df.sort_values(["player_id", "season", "week"]).copy()

    for col in target_cols:
        out[f"next_week_{col}"] = out.groupby(["player_id", "season"])[col].shift(-1)

    shifted = [f"next_week_{col}" for col in target_cols]
    out = out[~out[shifted].isna().all(axis=1)]
    return out


def get_base_features(df):
    """Base numerical features present in this dataset."""
    available = set(df.columns)
    all_potential = (
        CORE_OFFENSIVE + ADVANCED_PASSING + ADVANCED_RUSHING
        + ADVANCED_RECEIVING + SNAP_COUNT_FEATURES + SPECIAL_MISC + PERFORMANCE_TRACKING
    )
    return [col for col in all_potential if col in available]


def _position_column(df):
    if "position_x" in df.columns:
        return "position_x"
    if "position" in df.columns:
        return "position"
    return None


def add_derived_features(df):
    """Add efficiency, snap-share, and usage features derived from base stats."""
    out = df.copy()

    def ratio(num, den):
        return np.where(out[den] > 0, out[num] / out[den], 0)

    if {"attempts", "passing_yards"} <= set(out.columns):
        out["yards_per_attempt"] = ratio("passing_yards", "attempts")
    if {"carries", "rushing_yards"} <= set(out.columns):
        out["yards_per_carry"] = ratio("rushing_yards", "carries")
    if {"targets", "receiving_yards"} <= set(out.columns):
        out["yards_per_target"] = ratio("receiving_yards", "targets")

    if "offensive_snaps" in out.columns:
        out["high_snap_count"] = (out["offensive_snaps"] >= 50).astype(int)
        if "fanduel_fantasy_points" in out.columns:
            out["fantasy_per_snap"] = np.where(
                out["offensive_snaps"] > 0,
                out["fanduel_fantasy_points"] / out["offensive_snaps"], 0,
            )

    if "offensive_snap_pct" in out.columns:
        out["snap_role"] = pd.cut(
            out["offensive_snap_pct"], bins=[0, 25, 60, 100],
            labels=[0, 1, 2], include_lowest=True,
        ).astype(float)
        out["is_primary_player"] = (out["offensive_snap_pct"] >= 60).astype(int)
        out["reduced_snaps"] = (out["offensive_snap_pct"] < 50).astype(int)

    if "special_teams_snaps" in out.columns:
        out["special_teams_player"] = (out["special_teams_snaps"] > 0).astype(int)

    if {"carries", "targets", "offensive_snaps"} <= set(out.columns):
        touches = out["carries"].fillna(0) + out["targets"].fillna(0)
        out["opportunity_score"] = touches * 2 + out["offensive_snaps"].fillna(0) * 0.1

    if {"passing_epa", "attempts"} <= set(out.columns):
        out["epa_per_attempt"] = ratio("passing_epa", "attempts")
    if {"receiving_epa", "targets"} <= set(out.columns):
        out["epa_per_target"] = ratio("receiving_epa", "targets")
    if {"receiving_air_yards", "targets"} <= set(out.columns):
        out["avg_target_depth"] = ratio("receiving_air_yards", "targets")
    if {"completions", "attempts"} <= set(out.columns):
        out["completion_rate"] = ratio("completions", "attempts")
    if {"receptions", "targets"} <= set(out.columns):
        out["catch_rate"] = ratio("receptions", "targets")

    position_col = _position_column(out)
    if position_col:
        if "attempts" in out.columns:
            out["qb_passing_volume"] = np.where(
                out[position_col] == "QB", out["attempts"], 0
            )
        if {"carries", "targets"} <= set(out.columns):
            out["rb_total_touches"] = np.where(
                out[position_col] == "RB",
                out["carries"].fillna(0) + out["targets"].fillna(0), 0,
            )
        if "targets" in out.columns:
            out["wr_te_targets"] = np.where(
                out[position_col].isin(["WR", "TE"]), out["targets"], 0
            )
        if "offensive_snap_pct" in out.columns:
            out["wr_te_snap_rate"] = np.where(
                out[position_col].isin(["WR", "TE"]), out["offensive_snap_pct"], 0
            )

    if {"avg_fppg", "fanduel_fantasy_points"} <= set(out.columns):
        out["performance_vs_average"] = out["fanduel_fantasy_points"] - out["avg_fppg"]
        out["is_consistent_performer"] = (
            out["performance_vs_average"].abs() < 3
        ).astype(int)
        out["above_season_average"] = (
            out["fanduel_fantasy_points"] > out["avg_fppg"]
        ).astype(int)

    if {"attempts", "carries"} <= set(out.columns):
        total_plays = out["attempts"].fillna(0) + out["carries"].fillna(0)
        out["pass_heavy_script"] = np.where(
            total_plays > 0, out["attempts"].fillna(0) / total_plays > 0.6, 0
        ).astype(int)

    return out.fillna(0)


def select_feature_columns(df):
    """Return (numerical_features, categorical_features) available in df."""
    from .opponent import OPPONENT_FEATURES

    numerical = [c for c in get_base_features(df) if c in df.columns]
    numerical += [c for c in DERIVED_FEATURES if c in df.columns]
    numerical += [c for c in rolling_feature_names() if c in df.columns]
    numerical += [c for c in OPPONENT_FEATURES if c in df.columns]

    categorical = []
    position_col = _position_column(df)
    if position_col:
        categorical.append(position_col)
    # The dataset carries `team`; older frames used `recent_team`
    if "recent_team" in df.columns:
        categorical.append("recent_team")
    elif "team" in df.columns:
        categorical.append("team")

    return numerical, categorical


def clean_categorical_features(df, cat_features):
    """Map invalid positions to 'Unknown' and stringify categorical columns."""
    out = df.copy()
    for col in cat_features:
        if col not in out.columns:
            continue
        if "position" in col:
            out[col] = out[col].fillna("Unknown").astype(str).apply(
                lambda x: x if x in VALID_POSITIONS else "Unknown"
            )
        else:
            out[col] = (
                out[col].fillna("Unknown").astype(str)
                .replace(["nan", "None"], "Unknown")
            )
    return out


def is_rookie(player_name, player_id, historical_df, current_season):
    """True if a player entered the NFL this season and has fewer than 2 games.

    Players with 2+ games in the current season use the ML model even in
    their rookie year.
    """
    # Match on id first, then on the normalised name. A raw substring match
    # here mislabelled every veteran whose roster name carries a suffix —
    # nflverse stores "Kyle Pitts", the roster says "Kyle Pitts Sr.", and
    # containment in that direction fails. See PlayerPredictor._player_history.
    norm = normalize_player_name(player_name)
    history = historical_df[
        (historical_df["player_id"] == player_id)
        | (historical_df["player_display_name"].apply(normalize_player_name) == norm)
    ]
    if history.empty:
        return True

    actual_games = history[history["week"] != "AVG"]
    seasons_played = actual_games["season"].unique()

    if any(season < current_season for season in seasons_played):
        return False

    current_season_games = actual_games[actual_games["season"] == current_season]
    if len(current_season_games) >= 2:
        return False

    return True


def prepare_prediction_base(df, min_season=config.TRAINING_MIN_SEASON):
    """History table used to look up a player's latest game at prediction time.

    Keeps every real game, which is the whole point: this frame answers "what
    has this player actually done", and the answer must not be filtered.

    It used to inherit training's trims, and the docstring claiming it kept
    every game was simply wrong. The consequence was narrow and bad. The
    outlier bound sits at q1 = 0.0 fantasy points, so "outlier" in the lower
    tail means *any net-negative game* — which for a quarterback in limited
    relief is the normal result, not a freak one. A veteran whose recent
    season was a handful of replacement-level appearances therefore lost his
    entire recent history and fell through to the rookie prior. Gardner
    Minshew, four 2025 appearances scoring -0.30, -0.30, -0.12 and 1.40, kept
    one of them and was projected as a rookie. Across 2025 it hit 18 skill
    players, 14 of whom lost every game of the season.

    ``min_games`` goes for the same reason and one of its own: at 3 it
    contradicts ``is_rookie``, which says two games in the current season is
    enough to use the ML model — a player with exactly two would be dropped
    here and then read as having none.
    """
    cleaned = clean_training_data(df, min_season=min_season, min_games=1,
                                  drop_outliers=False)
    with_rolling = add_rolling_features(cleaned)
    with_features = add_derived_features(with_rolling)
    _, categorical = select_feature_columns(with_features)
    return clean_categorical_features(with_features, categorical)
