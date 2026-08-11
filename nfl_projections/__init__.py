"""NFL fantasy projections pipeline.

Consolidated from the nickknowsnfl / billb / nfl_ai_work iterations into a
single package.

Import it as a service (trains/loads once, project many weeks):

    from nfl_projections import ProjectionService
    svc = ProjectionService(quantiles=True)
    frame = svc.project(2025, 3, as_frame=True)     # floor/median/ceiling too
    bijan = svc.project_player("Bijan Robinson", 2025, 3)

Or one-shot:

    from nfl_projections import project_week
    df = project_week(2025, 3, quantiles=True)

Lower-level building blocks:

    from nfl_projections import dataset, model, predict
    df = dataset.build_dataset()
    trained, _ = model.train_ensemble(df)      # 5 seeds averaged; train_model = 1
    results = predict.Projector(df, trained, season=2025, week=13).run()

Command line:

    python -m nfl_projections build-data
    python -m nfl_projections predict --season 2025 --week 13 --quantiles
    python -m nfl_projections backtest --season 2025 --weeks 1-17
"""

from .service import (
    ProjectionService,
    optimize_week,
    project_week,
    results_to_frame,
)

__version__ = "1.5.0"

__all__ = [
    "ProjectionService",
    "project_week",
    "optimize_week",
    "results_to_frame",
]
