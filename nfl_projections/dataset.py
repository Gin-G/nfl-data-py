"""Build the historical dataset from nflreadpy.

Produces one CSV with:
- weekly player stats + snap counts for the configured seasons
- sportradar_id merged in from roster data
- season-average rows (week == 'AVG') per player-season
- rolling avg_fppg (fantasy points per game entering each week)
- FanDuel fantasy points per game
"""

import logging
import os

import numpy as np
import pandas as pd

from . import config
from .scoring import fanduel_points
from .utils import to_pandas

logger = logging.getLogger(__name__)

SNAP_COLUMN_MAPPING = {
    "offense_snaps": "offensive_snaps",
    "offense_pct": "offensive_snap_pct",
    "defense_snaps": "defensive_snaps",
    "defense_pct": "defensive_snap_pct",
    "st_snaps": "special_teams_snaps",
    "st_pct": "special_teams_snap_pct",
}

SNAP_COLUMNS = [
    "offensive_snaps", "defensive_snaps", "special_teams_snaps", "total_snaps",
    "offensive_snap_pct", "defensive_snap_pct", "special_teams_snap_pct",
]

PCT_COLUMNS = ["offensive_snap_pct", "defensive_snap_pct", "special_teams_snap_pct"]

SEASON_AVG_COLUMNS = [
    "passing_yards", "passing_tds", "interceptions", "attempts", "completions",
    "rushing_yards", "rushing_tds", "carries", "rushing_fumbles",
    "receiving_yards", "receiving_tds", "receptions", "targets", "receiving_fumbles",
    "offensive_snaps", "defensive_snaps", "special_teams_snaps", "total_snaps",
    "offensive_snap_pct", "defensive_snap_pct", "special_teams_snap_pct",
    "fanduel_fantasy_points",
]


def process_snap_counts(snap_data):
    """Rename snap columns to our schema, add total_snaps, clean percentages."""
    processed = snap_data.copy()

    player_id_col = None
    for col in ["player", "player_id", "player_display_name", "gsis_id"]:
        if col in processed.columns:
            player_id_col = col
            break
    if player_id_col is None:
        logger.warning("Could not find player ID column in snap data")
        return processed

    mapping = {player_id_col: "player_id", **SNAP_COLUMN_MAPPING}
    processed = processed.rename(
        columns={old: new for old, new in mapping.items() if old in processed.columns}
    )

    available_snaps = [
        c for c in ["offensive_snaps", "defensive_snaps", "special_teams_snaps"]
        if c in processed.columns
    ]
    if available_snaps:
        processed["total_snaps"] = processed[available_snaps].fillna(0).sum(axis=1)

    # Snap percentages arrive as 0-1 decimals in some sources; store as 0-100
    for col in PCT_COLUMNS:
        if col in processed.columns:
            if processed[col].max() <= 1.0:
                processed[col] = processed[col] * 100
            processed[col] = processed[col].fillna(0).clip(0, 100)

    for col in ["offensive_snaps", "defensive_snaps", "special_teams_snaps", "total_snaps"]:
        if col in processed.columns:
            processed[col] = processed[col].fillna(0)

    return processed


def add_season_averages(df):
    """Append one week='AVG' row per player-season with mean stats."""
    regular_weeks = df[df["week"] != "AVG"].copy()

    season_averages = []
    for (player_id, season), group in regular_weeks.groupby(["player_id", "season"]):
        if len(group) == 0:
            continue

        avg_row = {"player_id": player_id, "season": season, "week": "AVG"}

        first_game = group.iloc[0]
        for col in ["player_name", "player_display_name", "position", "recent_team"]:
            if col in first_game:
                avg_row[col] = first_game[col]

        for col in SEASON_AVG_COLUMNS:
            if col in group.columns:
                avg_row[col] = group[col].mean()

        avg_row["games_played"] = len(group)
        if "fanduel_fantasy_points" in group.columns:
            avg_row["fanduel_fantasy_points_total"] = group["fanduel_fantasy_points"].sum()

        season_averages.append(avg_row)

    if not season_averages:
        return df.copy()

    season_avg_df = pd.DataFrame(season_averages)
    season_avg_df = season_avg_df.reindex(columns=df.columns, fill_value=np.nan)
    return pd.concat([df, season_avg_df], ignore_index=True)


def add_rolling_averages(df):
    """Add avg_fppg: fantasy points per game entering each week.

    Week 1 uses the prior season's average (0 if none); later weeks use the
    mean of the current season's earlier games. AVG rows carry their own
    season average.
    """
    df["avg_fppg"] = np.nan
    df_sorted = df.sort_values(["player_id", "season", "week"]).copy()

    for player_id, player_data in df_sorted.groupby("player_id"):
        regular_weeks = player_data[player_data["week"] != "AVG"]
        avg_rows = player_data[player_data["week"] == "AVG"]

        for idx in avg_rows.index:
            if "fanduel_fantasy_points" in df.columns:
                df.at[idx, "avg_fppg"] = df.at[idx, "fanduel_fantasy_points"]

        for season, season_data in regular_weeks.groupby("season"):
            season_weeks = season_data.sort_values("week")

            for i, (idx, week_data) in enumerate(season_weeks.iterrows()):
                if i == 0:
                    prev_season_avg = avg_rows[avg_rows["season"] == (season - 1)]
                    if not prev_season_avg.empty and "fanduel_fantasy_points" in prev_season_avg.columns:
                        df.at[idx, "avg_fppg"] = prev_season_avg["fanduel_fantasy_points"].iloc[0]
                    else:
                        df.at[idx, "avg_fppg"] = 0
                else:
                    prev_weeks = season_weeks.iloc[:i]
                    df.at[idx, "avg_fppg"] = prev_weeks["fanduel_fantasy_points"].mean()

    return df


def build_dataset(seasons=None, output_path=config.DATASET_PATH):
    """Build the full historical dataset and (optionally) save it to CSV.

    Args:
        seasons: list of season years (default: config.SEASONS)
        output_path: where to save the CSV; pass None to skip saving
    """
    import nflreadpy as nfl

    seasons = seasons or config.SEASONS
    current_season = seasons[-1]

    print(f"Loading player stats for {seasons[0]}-{seasons[-1]} from nflreadpy...")
    player_stats = to_pandas(nfl.load_player_stats(seasons=seasons))
    print(f"Total player stat records: {len(player_stats)}")

    print("Loading rosters for sportradar_id mapping...")
    all_rosters = to_pandas(nfl.load_rosters_weekly(seasons=seasons))
    sportradar_mapping = all_rosters[["gsis_id", "sportradar_id"]].drop_duplicates("gsis_id")

    # player_stats 'player_id' is a gsis_id
    player_stats = pd.merge(
        player_stats, sportradar_mapping,
        left_on="player_id", right_on="gsis_id", how="left",
    )
    if "gsis_id" in player_stats.columns:
        player_stats = player_stats.drop(columns=["gsis_id"])

    has_sr = player_stats["sportradar_id"].notna().sum()
    print(f"Sportradar ID coverage: {has_sr}/{len(player_stats)} "
          f"({has_sr / len(player_stats) * 100:.1f}%)")

    print("Loading depth charts and snap counts...")
    depth_charts = to_pandas(nfl.load_depth_charts(seasons=seasons))
    snap_data = process_snap_counts(to_pandas(nfl.load_snap_counts(seasons=seasons)))

    print("Merging snap count data...")
    name_col = "player_display_name" if "player_display_name" in player_stats.columns else "player_name"
    snap_for_merge = snap_data.rename(columns={"player_id": name_col}).drop(
        columns=["position", "team", "opponent", "game_id", "pfr_game_id", "game_type"],
        errors="ignore",
    )
    df = pd.merge(player_stats, snap_for_merge, on=[name_col, "season", "week"], how="left")

    if "offensive_snaps" in df.columns:
        coverage = df["offensive_snaps"].notna().sum()
        print(f"Snap count coverage: {coverage:,}/{len(df):,} ({coverage / len(df) * 100:.1f}%)")

    print("Merging depth chart data...")
    depth_slim = depth_charts[["gsis_id", "season", "week", "position", "depth_team"]].copy()
    df = pd.merge(
        df, depth_slim,
        left_on=["player_id", "season", "week"],
        right_on=["gsis_id", "season", "week"],
        how="left", suffixes=("", "_depth_chart"),
    )
    if "gsis_id_depth_chart" in df.columns:
        df = df.drop(columns=["gsis_id_depth_chart"])
    if "gsis_id" in df.columns:
        df = df.drop(columns=["gsis_id"])

    for col in SNAP_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    if df["total_snaps"].isna().all() or (df["total_snaps"] == 0).all():
        df["total_snaps"] = (
            df["offensive_snaps"].fillna(0)
            + df["defensive_snaps"].fillna(0)
            + df["special_teams_snaps"].fillna(0)
        )
    for col in SNAP_COLUMNS:
        df[col] = df[col].fillna(0)
    for col in PCT_COLUMNS:
        df[col] = df[col].clip(0, 100)

    print("Calculating FanDuel fantasy points...")
    fumbles = sum(
        df[col].fillna(0) if col in df.columns else 0
        for col in ["rushing_fumbles", "receiving_fumbles", "sack_fumbles"]
    )
    df["fanduel_fantasy_points"] = fanduel_points(
        passing_yards=df.get("passing_yards", 0),
        passing_tds=df.get("passing_tds", 0),
        interceptions=df.get("interceptions", 0),
        rushing_yards=df.get("rushing_yards", 0),
        rushing_tds=df.get("rushing_tds", 0),
        receptions=df.get("receptions", 0),
        receiving_yards=df.get("receiving_yards", 0),
        receiving_tds=df.get("receiving_tds", 0),
        fumbles=fumbles,
    )

    print("Adding season averages...")
    df = add_season_averages(df)
    print("Adding rolling averages...")
    df = add_rolling_averages(df)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path} (shape {df.shape})")

    return df


def load_dataset(path=config.DATASET_PATH):
    """Load a previously built dataset CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run 'python -m nfl_projections build-data' first."
        )
    return pd.read_csv(path, low_memory=False)
