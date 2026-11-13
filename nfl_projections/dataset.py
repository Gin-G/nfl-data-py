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
from .scoring import fanduel_points_from_stats
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
    "passing_yards", "passing_tds", "passing_interceptions", "attempts", "completions",
    "rushing_yards", "rushing_tds", "carries", "rushing_fumbles",
    "receiving_yards", "receiving_tds", "receptions", "targets", "receiving_fumbles",
    "offensive_snaps", "defensive_snaps", "special_teams_snaps", "total_snaps",
    "offensive_snap_pct", "defensive_snap_pct", "special_teams_snap_pct",
    "fanduel_fantasy_points",
]


def process_snap_counts(snap_data):
    """Rename snap columns to our schema, add total_snaps, clean percentages."""
    processed = snap_data.copy()

    # The player is identified by `pfr_player_id` (and `player`, a display
    # name); merge_snaps turns that into a gsis id. This used to rename the
    # NAME column to `player_id` and join on it.
    mapping = dict(SNAP_COLUMN_MAPPING)
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


def _norm(series):
    from .utils import normalize_player_name
    return series.map(normalize_player_name)


def merge_snaps(player_stats, snap_data, crosswalk=None):
    """Attach snap counts to stats rows by PLAYER ID, not by name.

    Snap counts come from PFR and carry PFR's id; stats carry the gsis id. The
    crosswalk (nflreadpy.load_players: gsis_id <-> pfr_id) bridges them, and
    covered 99.7% of 2026 skill snap rows. The rest fall back to the suffix-free
    name on the same team and week.

    Until 2026-11-13 this joined on display name. The two feeds spell names
    differently ("Luther Burden III", "DK Metcalf", "Ollie Gordon II"), so 78 of
    1,255 2026 player-weeks with 5+ looks had no snaps, and names collide: the
    Browns' linebacker Justin Jefferson took the Vikings receiver's 63 week-5
    snaps, and the receiver got a second week-5 row with none.
    """
    snap = snap_data.copy()
    snap["gsis_id"] = pd.NA
    if (crosswalk is not None and not crosswalk.empty
            and "pfr_player_id" in snap.columns):
        xw = crosswalk.dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id")
        snap["gsis_id"] = snap["pfr_player_id"].map(dict(zip(xw.pfr_id, xw.gsis_id)))
    name_col = ("player_display_name" if "player_display_name" in player_stats.columns
                else "player_name")
    team_col = "team" if "team" in player_stats.columns else "recent_team"
    missing = snap["gsis_id"].isna()
    if missing.any() and "player" in snap.columns and team_col in player_stats.columns:
        keys = player_stats[["player_id", name_col, team_col, "season", "week"]].copy()
        keys["_n"] = _norm(keys[name_col])
        keys = keys.drop_duplicates(["_n", team_col, "season", "week"], keep=False)
        lookup = {(n, t, s_, w): pid for pid, n, t, s_, w in zip(
            keys.player_id, keys._n, keys[team_col], keys.season, keys.week)}
        sub = snap[missing]
        snap.loc[missing, "gsis_id"] = [
            lookup.get((n, t, s_, w)) for n, t, s_, w in zip(
                _norm(sub["player"]), sub.get("team"), sub["season"], sub["week"])]
    snap = snap.dropna(subset=["gsis_id"])
    keep = ["gsis_id", "season", "week"] + [c for c in SNAP_COLUMNS if c in snap.columns]
    snap = snap[keep].drop_duplicates(["gsis_id", "season", "week"], keep="first")
    out = player_stats.merge(snap, left_on=["player_id", "season", "week"],
                             right_on=["gsis_id", "season", "week"], how="left")
    return out.drop(columns=["gsis_id"])


def merge_depth(df, depth_charts):
    """Attach each player-week's best depth-chart line, once.

    Before 2025 nflverse's weekly charts list a player once per formation slot,
    and joining them as they were multiplied the rows: ~6,000 duplicated
    player-weeks a season in 2018-2024, which the cached backtest boards then
    carried (EXPERIMENTS.md, duplicate rows). One row per player-week now, the
    highest line he appears on.
    """
    cols = [c for c in ("gsis_id", "season", "week", "position", "depth_team")
            if c in depth_charts.columns]
    if not {"gsis_id", "season", "week"} <= set(cols):
        return df
    d = depth_charts[cols].dropna(subset=["gsis_id"]).copy()
    if "depth_team" in d.columns:
        d["_line"] = pd.to_numeric(d["depth_team"], errors="coerce")
        d = d.sort_values("_line", na_position="last").drop(columns="_line")
    d = d.drop_duplicates(["gsis_id", "season", "week"], keep="first")
    out = df.merge(d, left_on=["player_id", "season", "week"],
                   right_on=["gsis_id", "season", "week"],
                   how="left", suffixes=("", "_depth_chart"))
    return out.drop(columns=[c for c in ("gsis_id", "gsis_id_depth_chart")
                             if c in out.columns])


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


def _unpublished(exc):
    """True when nflreadpy failed because a season has no data file yet — a 404,
    or its own season-range check — as opposed to a network or parsing fault."""
    msg = str(exc)
    return "404" in msg or "Season must be between" in msg


def load_through_latest(load, seasons, what):
    """Call an nflreadpy loader over ``seasons``, tolerating only the NEWEST season
    being unpublished (the first days of a season, before nflverse posts it).

    Returns ``(frame, seasons_loaded)``. Anything else still raises: a transient
    failure that silently dropped the current season would put every projection
    back on last season's data with nothing downstream able to tell.
    """
    seasons = list(seasons)
    try:
        return to_pandas(load(seasons=seasons)), seasons
    except Exception as exc:
        if len(seasons) < 2 or not _unpublished(exc):
            raise
        logger.warning("%s for %d not published yet; loading through %d",
                       what, seasons[-1], seasons[-2])
        return to_pandas(load(seasons=seasons[:-1])), seasons[:-1]


def build_dataset(seasons=None, output_path=config.DATASET_PATH):
    """Build the full historical dataset and (optionally) save it to CSV.

    Args:
        seasons: list of season years (default: config.SEASONS, which runs
            through the season in progress)
        output_path: where to save the CSV; pass None to skip saving
    """
    import nflreadpy as nfl

    seasons = list(seasons or config.SEASONS)

    print(f"Loading player stats for {seasons[0]}-{seasons[-1]} from nflreadpy...")
    player_stats, seasons = load_through_latest(nfl.load_player_stats, seasons, "player stats")
    print(f"Total player stat records: {len(player_stats)} (through {seasons[-1]})")

    print("Loading rosters for sportradar_id mapping...")
    all_rosters, _ = load_through_latest(nfl.load_rosters_weekly, seasons, "weekly rosters")
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
    depth_charts, _ = load_through_latest(nfl.load_depth_charts, seasons, "depth charts")
    snap_data = process_snap_counts(
        load_through_latest(nfl.load_snap_counts, seasons, "snap counts")[0])

    print("Merging snap count data...")
    try:
        crosswalk = to_pandas(nfl.load_players())[["gsis_id", "pfr_id"]]
    except Exception as exc:  # pragma: no cover - network
        logger.warning("No player id crosswalk (%s); snaps join on name + team", exc)
        crosswalk = None
    df = merge_snaps(player_stats, snap_data, crosswalk)

    if "offensive_snaps" in df.columns:
        coverage = df["offensive_snaps"].notna().sum()
        print(f"Snap count coverage: {coverage:,}/{len(df):,} ({coverage / len(df) * 100:.1f}%)")

    print("Merging depth chart data...")
    df = merge_depth(df, depth_charts)

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
    # FanDuel's rule: fumbles LOST, plus two-point conversions. The target
    # used to charge every fumble and pay nothing for a conversion.
    df["fanduel_fantasy_points"] = fanduel_points_from_stats(df)

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
