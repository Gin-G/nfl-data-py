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


def component_fantasy_points(pred_df):
    """FanDuel points recomputed from predicted stat components (vs the model's
    direct fantasy-points output). Fumbles aren't predicted, so treated as 0.

    Missing components broadcast as zero-arrays so the result always has one
    row per prediction (a single-target model predicts no components at all).
    """
    from .scoring import fanduel_points

    def g(col):
        if col in pred_df.columns:
            return pred_df[col].values
        return np.zeros(len(pred_df))

    return pd.Series(fanduel_points(
        passing_yards=g("passing_yards"), passing_tds=g("passing_tds"),
        interceptions=g("passing_interceptions"),
        rushing_yards=g("rushing_yards"), rushing_tds=g("rushing_tds"),
        receptions=g("receptions"), receiving_yards=g("receiving_yards"),
        receiving_tds=g("receiving_tds"), fumbles=0,
    ))


def backtest(dataset, season, weeks=None, positions=None, trained=None,
             epochs=100, min_season=config.TRAINING_MIN_SEASON, use_opponent=False,
             schedule=None, loss="mse", scheme_form=None, include_coarse=True,
             target_cols=None, include_components=False, n_seeds=None,
             scale_targets=True, blend_form=True):
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
        n_seeds: seed-ensemble size when training here (default
            config.DEFAULT_N_SEEDS, matching production). Use 1 only for quick
            probes — single-seed results carry a +/-0.05 MAE seed lottery, so
            A/B deltas smaller than that are not readable.
        blend_form: blend the network output with trailing-5 form, as the
            Projector does (blend.py). Pass False to score the raw network.
    """
    from . import model as model_mod
    from .opponent import OPPONENT_FEATURES

    positions = positions or config.POSITIONS

    if trained is None:
        train_df = dataset[dataset["season"] < season]
        if train_df.empty:
            raise ValueError(f"No data before season {season} to train on")

        train_matchup = None
        if use_opponent:
            from . import opponent

            print("Building opponent matchup table for training...")
            train_seasons = sorted(train_df["season"].unique())
            train_map = opponent.build_schedule_map(train_seasons, schedule=schedule)
            train_matchup = opponent.build_matchup_table(
                train_df, train_map, scheme_form=scheme_form,
                include_coarse=include_coarse,
            )

        n_seeds = config.DEFAULT_N_SEEDS if n_seeds is None else n_seeds
        print(f"Training backtest model on seasons < {season} ({n_seeds} seed(s))...")
        trained, _ = model_mod.train_ensemble(
            train_df, n_seeds=n_seeds, epochs=epochs, min_season=min_season,
            plot_path=None, matchup_table=train_matchup, loss=loss,
            target_cols=target_cols, scale_targets=scale_targets,
        )

    uses_opponent = any(c in OPPONENT_FEATURES for c in trained.numerical_features)
    eval_matchup = None
    if uses_opponent:
        from . import opponent

        eval_map = opponent.build_schedule_map([season], schedule=schedule)
        eval_matchup = opponent.build_matchup_table(
            dataset, eval_map, scheme_form=scheme_form, include_coarse=include_coarse,
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

        opp_features = None
        if eval_matchup is not None:
            from .opponent import OPPONENT_FEATURES, _fill_neutral

            opp_cols = [c for c in OPPONENT_FEATURES if c in eval_matchup.columns]
            wk = eval_matchup[eval_matchup["week"] == int(week)]
            merged = week_games[["team", "position"]].merge(
                wk[["team", "position", *opp_cols]],
                on=["team", "position"], how="left",
            )
            opp_features = _fill_neutral(merged)[opp_cols].reset_index(drop=True)

        team_series = stat_rows["team"] if "team" in stat_rows.columns else stat_rows.get(
            "recent_team", pd.Series(["Unknown"] * len(stat_rows))
        )
        input_df = model_mod.build_input_rows(
            trained, stat_rows,
            positions=stat_rows[position_col].values,
            teams=team_series.values,
            opponent_features=opp_features,
        )
        predicted = model_mod.predict_batch(trained, input_df)

        # FanDuel points recomputed from the predicted stat components. For a
        # component-only model (no direct fanduel_fantasy_points target) this
        # IS the projection; for the standard model it's a comparison column.
        derived = component_fantasy_points(predicted).values
        if "fanduel_fantasy_points" in predicted.columns:
            headline = predicted["fanduel_fantasy_points"].values
        else:
            headline = derived

        # Same recent-form blend the Projector applies, so the backtest scores
        # what production actually serves (see blend.py)
        if blend_form:
            from . import blend as blend_mod

            form = blend_mod.recent_form_from_rows(stat_rows).values
            positions = week_games["position"].values
            headline = np.array([
                blend_mod.blend_value(h, f, p)
                for h, f, p in zip(headline, form, positions)
            ])

        week_result = pd.DataFrame({
            "player_id": week_games["player_id"].values,
            "player_name": week_games["player_display_name"].values
            if "player_display_name" in week_games.columns
            else week_games["player_name"].values,
            "position": week_games["position"].values,
            "season": season,
            "week": week,
            "predicted": headline,
            "derived": derived,
            "actual": week_games["fanduel_fantasy_points"].values,
        })
        if include_components:
            team_vals = (week_games["team"].values if "team" in week_games.columns
                         else stat_rows.get("team", stat_rows.get("recent_team")).values)
            week_result["team"] = team_vals
            for c in config.TARGET_COLS:
                if c != "fanduel_fantasy_points" and c in predicted.columns:
                    week_result[c] = predicted[c].values
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


# -- quantile (floor / median / ceiling) backtest --------------------------

def _prior_latest(history_games, season, week):
    """Each player's latest game strictly before (season, week)."""
    prior = history_games[
        (history_games["season"] < season)
        | ((history_games["season"] == season) & (history_games["week"] < week))
    ]
    return (
        prior.sort_values(["season", "week"])
        .groupby("player_id").tail(1).set_index("player_id")
    )


def backtest_quantiles(dataset, season, quantiles=None, weeks=None, positions=None,
                       qmodel=None, epochs=100, min_season=config.TRAINING_MIN_SEASON):
    """Backtest floor/median/ceiling projections over a season (leakage-free).

    Trains a quantile model on seasons < ``season`` (unless one is supplied),
    then for each week predicts every player's quantiles from their latest prior
    game. Returns one row per player-week with the quantile columns + actual.
    """
    from . import model as model_mod
    from . import quantiles as q_mod

    quantiles = list(quantiles or q_mod.DEFAULT_QUANTILES)
    qcols = [f"q{int(round(q * 100))}" for q in quantiles]
    positions = positions or config.POSITIONS

    if qmodel is None:
        train_df = dataset[dataset["season"] < season]
        if train_df.empty:
            raise ValueError(f"No data before season {season} to train on")
        print(f"Training quantile model {quantiles} on seasons < {season}...")
        qmodel, _ = q_mod.train_quantile_model(
            train_df, quantiles=quantiles, epochs=epochs, min_season=min_season
        )

    history = features.prepare_prediction_base(dataset, min_season=min_season)
    position_col = "position_x" if "position_x" in history.columns else "position"

    season_games = regular_games(dataset[dataset["season"] == season])
    season_games = season_games[season_games["position"].isin(positions)]
    if weeks is None:
        weeks = sorted(season_games["week"].unique())

    history_games = regular_games(history)
    history_games = history_games[history_games["season"] >= season - 1]

    results = []
    for week in weeks:
        week_games = season_games[season_games["week"] == week]
        if week_games.empty:
            continue
        prior_latest = _prior_latest(history_games, season, week)
        week_games = week_games[week_games["player_id"].isin(prior_latest.index)]
        if week_games.empty:
            continue

        stat_rows = prior_latest.loc[week_games["player_id"]].reset_index()
        team_series = stat_rows["team"] if "team" in stat_rows.columns else pd.Series(
            ["Unknown"] * len(stat_rows)
        )
        input_df = model_mod.build_input_rows(
            qmodel, stat_rows, positions=stat_rows[position_col].values,
            teams=team_series.values,
        )
        qpreds = q_mod.predict_quantiles(qmodel, input_df)

        name_col = "player_display_name" if "player_display_name" in week_games.columns else "player_name"
        week_result = pd.DataFrame({
            "player_id": week_games["player_id"].values,
            "player_name": week_games[name_col].values,
            "position": week_games["position"].values,
            "season": season, "week": week,
            "actual": week_games["fanduel_fantasy_points"].values,
        })
        for c in qcols:
            week_result[c] = qpreds[c].values
        results.append(week_result)
        print(f"Week {int(week)}: quantiles for {len(week_result)} players")

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def summarize_quantiles(results, quantiles=None):
    """Calibration report: are the quantiles honest, and how wide is the band?"""
    from . import quantiles as q_mod

    if results.empty:
        print("No quantile results to summarize")
        return pd.DataFrame()

    quantiles = list(quantiles or q_mod.DEFAULT_QUANTILES)
    qcols = [f"q{int(round(q * 100))}" for q in quantiles]
    floor_c, ceil_c = qcols[0], qcols[-1]
    median_c = "q50" if "q50" in qcols else qcols[len(qcols) // 2]

    print("\n=== QUANTILE CALIBRATION (season backtest) ===")
    print(f"{'pred':>6} {'target':>7} {'empirical P(actual<=pred)':>28}")
    for q, c in zip(quantiles, qcols):
        emp = (results["actual"] <= results[c]).mean()
        flag = "" if abs(emp - q) <= 0.03 else "  <-- off"
        print(f"{c:>6} {q:>7.2f} {emp:>28.3f}{flag}")

    within = ((results["actual"] >= results[floor_c]) & (results["actual"] <= results[ceil_c])).mean()
    width = (results[ceil_c] - results[floor_c]).mean()
    merr = results[median_c] - results["actual"]
    target_cov = quantiles[-1] - quantiles[0]
    print(f"\nInterval [{floor_c},{ceil_c}] covers {within:.1%} of actuals "
          f"(target {target_cov:.0%}); mean width {width:.1f} pts")
    print(f"Median ({median_c}) vs actual: MAE {merr.abs().mean():.2f}, bias {merr.mean():+.2f}")

    def pos_stats(g):
        return pd.Series({
            "n": len(g),
            "interval_cov": ((g["actual"] >= g[floor_c]) & (g["actual"] <= g[ceil_c])).mean(),
            "median_mae": (g[median_c] - g["actual"]).abs().mean(),
            "mean_floor": g[floor_c].mean(),
            "mean_ceiling": g[ceil_c].mean(),
            "mean_actual": g["actual"].mean(),
        })

    print("\nBy position:")
    print(results.groupby("position").apply(pos_stats, include_groups=False).round(2).to_string())
    return results
