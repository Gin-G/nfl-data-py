"""Importable projection API — use the model from other code, not just the CLI.

    from nfl_projections import ProjectionService

    svc = ProjectionService(quantiles=True)        # trains once (or model_dir=...)
    results = svc.project(2025, 3)                  # {position: DataFrame}
    frame   = svc.project(2025, 3, as_frame=True)   # one flat DataFrame
    bijan   = svc.project_player("Bijan Robinson", 2025, 3)

One-shot convenience (builds a service, projects one week):

    from nfl_projections import project_week
    df = project_week(2025, 3, quantiles=True)

A ProjectionService loads the dataset and trains (or loads) the model ONCE, so
projecting many weeks/players afterwards is cheap. Pass ``model_dir`` /
``quantile_model_dir`` to load saved models instead of training.
"""

import pandas as pd

from . import config
from .predict import Projector


def results_to_frame(results):
    """Flatten a {position: DataFrame} result into one DataFrame, best first."""
    if not results:
        return pd.DataFrame()
    frame = pd.concat(results.values(), ignore_index=True)
    if "fanduel_fantasy_points" in frame.columns:
        frame = frame.sort_values(
            "fanduel_fantasy_points", ascending=False
        ).reset_index(drop=True)
    return frame


class ProjectionService:
    """Holds a loaded dataset + trained model(s) for repeated projection.

    Args:
        dataset: a pre-loaded DataFrame (skips loading from disk)
        data_path: dataset CSV to load when ``dataset`` is None
        model_dir: load a saved mean model instead of training one
        quantile_model_dir: load a saved quantile model
        quantiles: train a quantile model (floor/median/ceiling) if none loaded
        opponent: train with opponent-defense features
        epochs: training epochs when training here
        n_seeds: networks in the mean-model seed ensemble when training here
            (default 5 — measured ~0.04 MAE better than a typical single seed
            and immune to the seed lottery). Costs n_seeds x the training time;
            pass 1 for a quick run.
        blend_form: blend projections with trailing-5 scoring form (blend.py).
            On by default: it improves per-game MAE, cross-player ranking and
            spread at once. Pass False for raw network output.
    """

    def __init__(self, dataset=None, data_path=config.DATASET_PATH, model_dir=None,
                 quantile_model_dir=None, quantiles=False, opponent=False, epochs=100,
                 n_seeds=None, blend_form=True):
        from . import dataset as dataset_mod
        from . import model as model_mod

        self.dataset = dataset if dataset is not None else dataset_mod.load_dataset(data_path)
        self.blend_form = blend_form

        matchup = None
        if opponent:
            from . import opponent as opp_mod

            seasons = sorted(self.dataset["season"].unique())
            matchup = opp_mod.build_matchup_table(
                self.dataset, opp_mod.build_schedule_map(seasons)
            )

        if model_dir:
            self.model = model_mod.TrainedModel.load(model_dir)
        else:
            # No plot side-effect when used as a library
            n_seeds = model_mod.DEFAULT_N_SEEDS if n_seeds is None else n_seeds
            self.model, _ = model_mod.train_ensemble(
                self.dataset, n_seeds=n_seeds, epochs=epochs,
                matchup_table=matchup, plot_path=None,
            )

        self.quantile_model = None
        if quantile_model_dir or quantiles:
            from . import quantiles as q_mod

            if quantile_model_dir:
                self.quantile_model = q_mod.QuantileModel.load(quantile_model_dir)
            else:
                self.quantile_model, _ = q_mod.train_quantile_model(
                    self.dataset, epochs=epochs, matchup_table=matchup,
                    n_seeds=n_seeds,
                )

    def _projector(self, season, week, use_injuries=True, injury_source="nflverse",
                   rosters=None, depth_charts=None, schedule=None, rookie_fallback=False):
        return Projector(
            self.dataset, self.model, season=season, week=week,
            use_injuries=use_injuries, injury_source=injury_source,
            quantile_model=self.quantile_model, rookie_fallback=rookie_fallback,
            rosters=rosters, depth_charts=depth_charts, schedule=schedule,
            blend_form=self.blend_form,
        )

    def project(self, season, week, positions=None, players=None, use_injuries=True,
                injury_source="nflverse", save=False, output_dir=config.PREDICTIONS_DIR,
                as_frame=False, rosters=None, depth_charts=None, schedule=None,
                rookie_fallback=False):
        """Project a week. Returns {position: DataFrame}, or one DataFrame if
        ``as_frame``. Each row has fanduel_fantasy_points and, when a quantile
        model is loaded, floor / projection_median / ceiling.

        `rosters` / `depth_charts`: pass externally (e.g. from ESPN for a season
        nflreadpy hasn't published yet). `rookie_fallback`: also project rookies with no
        NFL history via the draft-capital prior."""
        projector = self._projector(
            season, week, use_injuries, injury_source, rosters, depth_charts, schedule,
            rookie_fallback=rookie_fallback,
        )
        results = projector.run(
            positions=positions, players=players, save=save, output_dir=output_dir
        )
        return results_to_frame(results) if as_frame else results

    def project_player(self, player_name, season, week, use_injuries=True,
                       injury_source="nflverse", rosters=None, depth_charts=None,
                       schedule=None):
        """Project a single player. Returns a dict of stats, or None."""
        projector = self._projector(
            season, week, use_injuries, injury_source, rosters, depth_charts, schedule
        )
        return projector.predict_player(player_name)

    def optimize(self, season, week, fanduel_csv, objective="mean", num_lineups=5,
                 salary_cap=60000, exclude_players=None, max_usage_percentage=50,
                 positions=None, use_injuries=True):
        """Project the week, merge with a FanDuel salary export, build lineups.

        Args:
            fanduel_csv: path to (or DataFrame of) a FanDuel main-slate export
            objective: "mean" (default), "ceiling", "floor", or "median" —
                needs a quantile model loaded for the non-mean options.
        Returns (lineups, merged_frame).
        """
        from . import optimizer

        projections = self.project(
            season, week, positions=positions, use_injuries=use_injuries, as_frame=True
        )
        fanduel_df = fanduel_csv if isinstance(fanduel_csv, pd.DataFrame) else pd.read_csv(fanduel_csv)
        merged = optimizer.merge_fanduel_salaries(fanduel_df, projections)
        lineups = optimizer.optimize_lineups(
            merged, num_lineups=num_lineups, salary_cap=salary_cap,
            exclude_players=exclude_players, max_usage_percentage=max_usage_percentage,
            objective=objective,
        )
        return lineups, merged


def project_week(season, week, *, quantiles=False, opponent=False, model_dir=None,
                 quantile_model_dir=None, data_path=config.DATASET_PATH, dataset=None,
                 epochs=100, n_seeds=None, positions=None, players=None, use_injuries=True,
                 as_frame=True, save=False, output_dir=config.PREDICTIONS_DIR):
    """One-shot weekly projection (builds a service, projects one week).

    For repeated use, construct a ProjectionService once and call .project().
    Returns one DataFrame by default (as_frame=True).
    """
    svc = ProjectionService(
        dataset=dataset, data_path=data_path, model_dir=model_dir,
        quantile_model_dir=quantile_model_dir, quantiles=quantiles,
        opponent=opponent, epochs=epochs, n_seeds=n_seeds,
    )
    return svc.project(
        season, week, positions=positions, players=players,
        use_injuries=use_injuries, save=save, output_dir=output_dir, as_frame=as_frame,
    )


def optimize_week(season, week, fanduel_csv, *, objective="mean", num_lineups=5,
                  salary_cap=60000, exclude_players=None, max_usage_percentage=50,
                  model_dir=None, quantile_model_dir=None, data_path=config.DATASET_PATH,
                  dataset=None, epochs=100, n_seeds=None, use_injuries=True):
    """One-shot: project a week, merge FanDuel salaries, build lineups.

    A quantile model is trained automatically when ``objective`` is floor/median/
    ceiling (needed for those columns). Returns (lineups, merged_frame).
    """
    svc = ProjectionService(
        dataset=dataset, data_path=data_path, model_dir=model_dir,
        quantile_model_dir=quantile_model_dir,
        quantiles=objective in ("floor", "median", "ceiling"),
        epochs=epochs, n_seeds=n_seeds,
    )
    return svc.optimize(
        season, week, fanduel_csv, objective=objective, num_lineups=num_lineups,
        salary_cap=salary_cap, exclude_players=exclude_players,
        max_usage_percentage=max_usage_percentage, use_injuries=use_injuries,
    )
