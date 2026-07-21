"""Season backtesting and comparison against external projections.

Backtest: train on seasons *before* the evaluation season (no leakage), then
for each week predict every player from their most recent prior game and
compare to what actually happened.

Compare: line our projections up against someone else's (e.g. a FantasyPros
CSV export) by normalized player name, and score both against actuals.
"""

import logging

import numpy as np
import pandas as pd

from . import config, features
from .utils import normalize_player_name, regular_games

logger = logging.getLogger(__name__)


def backtest(dataset, season, weeks=None, positions=None, trained=None,
             epochs=100, min_season=config.TRAINING_MIN_SEASON):
    """Backtest the model over a season. Returns a results DataFrame with one
    row per player-week: predicted vs. actual FanDuel points.

    Args:
        dataset: full historical dataset
        season: season to evaluate
        weeks: list of week numbers (default: every week with games)
        positions: restrict to positions (default QB/RB/WR/TE)
        trained: reuse an existing TrainedModel; when None, a model is
            trained on seasons strictly before `season`
        epochs: training epochs when training here
    """
    from . import model as model_mod

    positions = positions or config.POSITIONS

    if trained is None:
        train_df = dataset[dataset["season"] < season]
        if train_df.empty:
            raise ValueError(f"No data before season {season} to train on")
        print(f"Training backtest model on seasons < {season}...")
        trained, _ = model_mod.train_model(
            train_df, epochs=epochs, min_season=min_season, plot_path=None
        )

    # Feature history: keep every game (targets not required), same cleaning
    history = features.prepare_prediction_base(dataset, min_season=min_season)
    position_col = "position_x" if "position_x" in history.columns else "position"

    season_games = regular_games(dataset[dataset["season"] == season])
    season_games = season_games[season_games["position"].isin(positions)]
    if weeks is None:
        weeks = sorted(season_games["week"].unique())

    history_games = regular_games(history)
    # Only use games a bettor could have seen: this season or the prior one
    history_games = history_games[history_games["season"] >= season - 1]

    results = []
    for week in weeks:
        week_games = season_games[season_games["week"] == week]
        if week_games.empty:
            continue

        # Latest game strictly before this week for each player
        prior = history_games[
            (history_games["season"] < season)
            | ((history_games["season"] == season) & (history_games["week"] < week))
        ]
        prior_latest = (
            prior.sort_values(["season", "week"])
            .groupby("player_id")
            .tail(1)
            .set_index("player_id")
        )

        week_games = week_games[week_games["player_id"].isin(prior_latest.index)]
        if week_games.empty:
            continue

        stat_rows = prior_latest.loc[week_games["player_id"]].reset_index()
        input_df = model_mod.build_input_rows(
            trained, stat_rows,
            positions=stat_rows[position_col].values,
            teams=stat_rows.get("recent_team", pd.Series(["Unknown"] * len(stat_rows))).values,
        )
        predicted = model_mod.predict_batch(trained, input_df)

        week_result = pd.DataFrame({
            "player_id": week_games["player_id"].values,
            "player_name": week_games["player_display_name"].values
            if "player_display_name" in week_games.columns
            else week_games["player_name"].values,
            "position": week_games["position"].values,
            "season": season,
            "week": week,
            "predicted": predicted["fanduel_fantasy_points"].values,
            "actual": week_games["fanduel_fantasy_points"].values,
        })
        results.append(week_result)
        print(f"Week {int(week)}: predicted {len(week_result)} players")

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def summarize(results):
    """Summarize backtest results: MAE/RMSE/correlation overall & by position."""
    if results.empty:
        print("No backtest results to summarize")
        return pd.DataFrame()

    def stats(group):
        err = group["predicted"] - group["actual"]
        return pd.Series({
            "n": len(group),
            "mae": err.abs().mean(),
            "rmse": np.sqrt((err ** 2).mean()),
            "corr": group["predicted"].corr(group["actual"]),
            "mean_predicted": group["predicted"].mean(),
            "mean_actual": group["actual"].mean(),
        })

    overall = stats(results).to_frame("ALL").T
    by_position = results.groupby("position").apply(stats, include_groups=False)
    summary = pd.concat([overall, by_position]).round(2)

    print("\n=== BACKTEST SUMMARY (FanDuel points) ===")
    print(summary.to_string())

    weekly = results.groupby("week").apply(
        lambda g: (g["predicted"] - g["actual"]).abs().mean(), include_groups=False
    ).round(2)
    print("\nMAE by week:")
    print(weekly.to_string())

    return summary


# -- external projections -------------------------------------------------

# Column names used by common free projection sources (FantasyPros etc.)
NAME_COLUMN_CANDIDATES = ["player", "player_name", "name", "Player", "PLAYER", "Nickname"]
POINTS_COLUMN_CANDIDATES = ["fpts", "FPTS", "points", "proj", "projection",
                            "projected_points", "fanduel_fantasy_points", "MISC FPTS"]
WEEK_COLUMN_CANDIDATES = ["week", "Week", "WK"]


def _find_column(df, candidates, kind):
    for col in candidates:
        if col in df.columns:
            return col
    # case-insensitive fallback
    lower_map = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in lower_map:
            return lower_map[col.lower()]
    raise ValueError(
        f"Could not find a {kind} column in {list(df.columns)}. "
        f"Expected one of {candidates}; rename the column or pass it explicitly."
    )


def load_external_projections(path, name_col=None, points_col=None, week_col=None):
    """Load someone else's projections CSV into (name, week?, points) form.

    Column names are sniffed from common formats; override with the
    *_col arguments for anything unusual.
    """
    df = pd.read_csv(path)
    name_col = name_col or _find_column(df, NAME_COLUMN_CANDIDATES, "player name")
    points_col = points_col or _find_column(df, POINTS_COLUMN_CANDIDATES, "points")

    out = pd.DataFrame({
        "player_name": df[name_col].astype(str),
        "external_points": pd.to_numeric(df[points_col], errors="coerce"),
    })
    if week_col is None:
        try:
            week_col = _find_column(df, WEEK_COLUMN_CANDIDATES, "week")
        except ValueError:
            week_col = None
    if week_col:
        out["week"] = pd.to_numeric(df[week_col], errors="coerce")

    out["normalized_name"] = out["player_name"].apply(normalize_player_name)
    return out.dropna(subset=["external_points"])


def compare_to_external(ours, external, by_week=False):
    """Compare our projections to an external source (and actuals if present).

    Args:
        ours: DataFrame with player_name + a points column — either backtest
            results (predicted/actual) or a predictions CSV
            (fanduel_fantasy_points)
        external: DataFrame from load_external_projections
        by_week: also join on week (both frames need a week column)

    Returns the merged frame; prints agreement stats.
    """
    ours = ours.copy()
    if "predicted" not in ours.columns:
        points_col = _find_column(ours, POINTS_COLUMN_CANDIDATES, "points")
        ours = ours.rename(columns={points_col: "predicted"})
    ours["normalized_name"] = ours["player_name"].apply(normalize_player_name)

    join_cols = ["normalized_name"] + (["week"] if by_week else [])
    external_cols = join_cols + ["external_points"]
    merged = pd.merge(
        ours, external[external_cols].drop_duplicates(join_cols),
        on=join_cols, how="inner",
    )

    if merged.empty:
        print("No overlapping players between the two sources — check name formats.")
        return merged

    print(f"\n=== COMPARISON vs EXTERNAL ({len(merged)} matched player-rows, "
          f"{len(ours) - len(merged)} of ours unmatched) ===")

    diff = merged["predicted"] - merged["external_points"]
    print(f"Ours vs theirs:   MAE {diff.abs().mean():.2f}, "
          f"correlation {merged['predicted'].corr(merged['external_points']):.3f}, "
          f"mean diff {diff.mean():+.2f}")

    if "actual" in merged.columns:
        our_err = (merged["predicted"] - merged["actual"]).abs().mean()
        their_err = (merged["external_points"] - merged["actual"]).abs().mean()
        our_corr = merged["predicted"].corr(merged["actual"])
        their_corr = merged["external_points"].corr(merged["actual"])
        print(f"Ours vs actual:   MAE {our_err:.2f}, correlation {our_corr:.3f}")
        print(f"Theirs vs actual: MAE {their_err:.2f}, correlation {their_corr:.3f}")
        verdict = "ahead of" if our_err < their_err else "behind"
        print(f"-> We are {verdict} the external source by "
              f"{abs(our_err - their_err):.2f} MAE points")

    biggest = merged.assign(diff=diff.abs()).nlargest(10, "diff")
    print("\nBiggest disagreements:")
    for _, row in biggest.iterrows():
        print(f"  {row['player_name']:24s} ours {row['predicted']:5.1f} "
              f"vs theirs {row['external_points']:5.1f}")

    return merged
