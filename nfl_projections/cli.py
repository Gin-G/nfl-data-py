"""Command-line interface.

    python -m nfl_projections build-data
    python -m nfl_projections train --save-dir models
    python -m nfl_projections predict --season 2025 --week 13
    python -m nfl_projections predict --season 2025 --week 13 --players "Bijan Robinson" "CeeDee Lamb"
    python -m nfl_projections pools --season 2025 --week 13
    python -m nfl_projections optimize --csv predictions/fanduel_value_week13.csv
    python -m nfl_projections backtest --season 2025 --weeks 1-17
    python -m nfl_projections compare --results backtest_2025.csv --external fantasypros.csv
"""

import argparse
import logging

from . import config
from .utils import parse_weeks


def _add_common_week_args(parser):
    parser.add_argument("--season", type=int, default=config.CURRENT_SEASON,
                        help=f"Season year (default: {config.CURRENT_SEASON})")
    parser.add_argument("--week", type=int, required=True, help="Week number")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="nfl_projections",
        description="NFL fantasy projections pipeline",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-data", help="Build the historical dataset CSV")
    p.add_argument("--seasons", type=str, default=None,
                   help="Season range like 2018-2025 (default: config.SEASONS)")
    p.add_argument("--output", type=str, default=config.DATASET_PATH)

    p = sub.add_parser("train", help="Train the projection model")
    p.add_argument("--data", type=str, default=config.DATASET_PATH)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--save-dir", type=str, default=config.MODELS_DIR,
                   help="Directory to save the trained model")
    p.add_argument("--opponent", action="store_true",
                   help="Include opponent-defense matchup features")
    p.add_argument("--loss", choices=["mse", "huber", "mae"], default="mse",
                   help="Training loss (default: mse)")

    p = sub.add_parser("predict", help="Generate weekly projections")
    _add_common_week_args(p)
    p.add_argument("--data", type=str, default=config.DATASET_PATH)
    p.add_argument("--positions", nargs="+", default=None,
                   choices=config.POSITIONS, help="Positions (default: all)")
    p.add_argument("--players", nargs="+", default=None,
                   help="Only these players (substring match on name)")
    p.add_argument("--model-dir", type=str, default=None,
                   help="Load a saved model instead of training fresh")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--opponent", action="store_true",
                   help="Include opponent-defense matchup features (fresh training)")
    p.add_argument("--no-injuries", action="store_true",
                   help="Skip injury adjustments")
    p.add_argument("--injury-source", choices=["nflverse", "sportradar", "auto"],
                   default="nflverse",
                   help="Injury data source (default: nflverse, free/no key)")
    p.add_argument("--rookie-fallback", action="store_true",
                   help="Give baseline projections to rookies with no games")
    p.add_argument("--quantiles", action="store_true",
                   help="Also produce floor/median/ceiling via a quantile model")
    p.add_argument("--quantile-model-dir", type=str, default=None,
                   help="Load a saved quantile model instead of training one")
    p.add_argument("--output-dir", type=str, default=config.PREDICTIONS_DIR)

    p = sub.add_parser("pools", help="Build DFS player pools from predictions")
    _add_common_week_args(p)
    p.add_argument("--data", type=str, default=config.DATASET_PATH)
    p.add_argument("--predictions-dir", type=str, default=config.PREDICTIONS_DIR)
    p.add_argument("--output-dir", type=str, default=config.POOLS_DIR)
    p.add_argument("--no-combined", action="store_true")

    p = sub.add_parser("optimize", help="Build FanDuel lineups from projections")
    p.add_argument("--csv", type=str, default=None,
                   help="Pre-merged FanDuel salary + projections CSV")
    p.add_argument("--fanduel", type=str, default=None,
                   help="FanDuel salary export; project --season/--week and merge automatically")
    p.add_argument("--season", type=int, default=config.CURRENT_SEASON)
    p.add_argument("--week", type=int, default=None,
                   help="Week to project (required with --fanduel)")
    p.add_argument("--objective", choices=["mean", "floor", "median", "ceiling"],
                   default="mean",
                   help="What to optimize: mean (default), ceiling (GPP), floor (cash)")
    p.add_argument("--data", type=str, default=config.DATASET_PATH)
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lineups", type=int, default=5)
    p.add_argument("--salary-cap", type=int, default=60000)
    p.add_argument("--exclude", nargs="+", default=None, help="Players to exclude")
    p.add_argument("--max-usage", type=float, default=50,
                   help="Max %% of lineups any one player can appear in")

    p = sub.add_parser("backtest", help="Backtest the model over a season")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--weeks", type=str, default=None,
                   help="Weeks like 1-17 or 1,2,5 (default: all played)")
    p.add_argument("--positions", nargs="+", default=None, choices=config.POSITIONS)
    p.add_argument("--data", type=str, default=config.DATASET_PATH)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--opponent", action="store_true",
                   help="Include opponent-defense matchup features")
    p.add_argument("--loss", choices=["mse", "huber", "mae"], default="mse",
                   help="Training loss (default: mse)")
    p.add_argument("--output", type=str, default=None,
                   help="Save per-player results CSV here")

    p = sub.add_parser("compare", help="Compare projections to an external source")
    p.add_argument("--results", type=str, required=True,
                   help="Our CSV: backtest results or a predictions file")
    p.add_argument("--external", type=str, required=True,
                   help="External projections CSV (e.g. FantasyPros export)")
    p.add_argument("--name-col", type=str, default=None)
    p.add_argument("--points-col", type=str, default=None)
    p.add_argument("--by-week", action="store_true",
                   help="Match on week as well as player name")
    p.add_argument("--output", type=str, default=None, help="Save merged CSV here")

    return parser


def _build_matchup_table(dataset_df):
    from . import opponent

    seasons = sorted(dataset_df["season"].unique())
    schedule_map = opponent.build_schedule_map(seasons)
    return opponent.build_matchup_table(dataset_df, schedule_map)


def _get_trained_model(args, dataset_df):
    from . import model as model_mod

    if args.model_dir:
        print(f"Loading model from {args.model_dir}/")
        return model_mod.TrainedModel.load(args.model_dir)
    matchup = _build_matchup_table(dataset_df) if getattr(args, "opponent", False) else None
    trained, _ = model_mod.train_model(dataset_df, epochs=args.epochs, matchup_table=matchup)
    return trained


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    if args.command == "build-data":
        from . import dataset

        seasons = None
        if args.seasons:
            start, end = args.seasons.split("-")
            seasons = list(range(int(start), int(end) + 1))
        dataset.build_dataset(seasons=seasons, output_path=args.output)

    elif args.command == "train":
        from . import dataset
        from . import model as model_mod

        df = dataset.load_dataset(args.data)
        matchup = _build_matchup_table(df) if args.opponent else None
        model_mod.train_model(df, epochs=args.epochs, save_dir=args.save_dir,
                              matchup_table=matchup, loss=args.loss)

    elif args.command == "predict":
        from . import dataset
        from .predict import Projector

        df = dataset.load_dataset(args.data)
        trained = _get_trained_model(args, df)

        quantile_model = None
        if args.quantiles or args.quantile_model_dir:
            from . import quantiles as q_mod

            if args.quantile_model_dir:
                print(f"Loading quantile model from {args.quantile_model_dir}/")
                quantile_model = q_mod.QuantileModel.load(args.quantile_model_dir)
            else:
                print("Training quantile model for floor/median/ceiling...")
                quantile_model, _ = q_mod.train_quantile_model(df, epochs=args.epochs)

        projector = Projector(
            df, trained, season=args.season, week=args.week,
            use_injuries=not args.no_injuries,
            rookie_fallback=args.rookie_fallback,
            injury_source=args.injury_source,
            quantile_model=quantile_model,
        )
        projector.run(positions=args.positions, players=args.players,
                      output_dir=args.output_dir)

    elif args.command == "pools":
        from . import pools

        files, total = pools.create_player_pool(
            season=args.season, week=args.week,
            main_dataset_path=args.data,
            predictions_directory=args.predictions_dir,
            output_directory=args.output_dir,
        )
        if files and not args.no_combined:
            pools.create_combined_player_pool(args.season, args.week, args.output_dir)
        print(f"\nDone: {total} players across {len(files)} position files")

    elif args.command == "optimize":
        from . import optimizer

        if args.fanduel:
            if args.week is None:
                raise SystemExit("--fanduel requires --week (and --season)")
            from .service import optimize_week

            lineups, _ = optimize_week(
                args.season, args.week, args.fanduel,
                objective=args.objective, num_lineups=args.lineups,
                salary_cap=args.salary_cap, exclude_players=args.exclude,
                max_usage_percentage=args.max_usage,
                model_dir=args.model_dir, data_path=args.data, epochs=args.epochs,
            )
        elif args.csv:
            lineups = optimizer.optimize_from_csv(
                args.csv, num_lineups=args.lineups, salary_cap=args.salary_cap,
                exclude_players=args.exclude, max_usage_percentage=args.max_usage,
                objective=args.objective,
            )
        else:
            raise SystemExit("optimize needs --csv (pre-merged) or --fanduel + --week")
        optimizer.display_lineups(lineups)

    elif args.command == "backtest":
        from . import dataset, evaluate

        df = dataset.load_dataset(args.data)
        weeks = parse_weeks(args.weeks) if args.weeks else None
        results = evaluate.backtest(
            df, season=args.season, weeks=weeks,
            positions=args.positions, epochs=args.epochs,
            use_opponent=args.opponent, loss=args.loss,
        )
        evaluate.summarize(results)
        if args.output and not results.empty:
            results.to_csv(args.output, index=False)
            print(f"\nPer-player results saved to {args.output}")

    elif args.command == "compare":
        import pandas as pd

        from . import evaluate

        ours = pd.read_csv(args.results)
        external = evaluate.load_external_projections(
            args.external, name_col=args.name_col, points_col=args.points_col,
        )
        merged = evaluate.compare_to_external(ours, external, by_week=args.by_week)
        if args.output and not merged.empty:
            merged.to_csv(args.output, index=False)
            print(f"Merged comparison saved to {args.output}")


if __name__ == "__main__":
    main()
