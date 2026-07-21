"""NFL fantasy projections pipeline.

Consolidated from the nickknowsnfl / billb / nfl_ai_work iterations into a
single package. Typical flow:

    from nfl_projections import dataset, model, predict

    df = dataset.build_dataset()                    # data/nfl_dataset.csv
    trained, _ = model.train_model(df)              # train the network
    projector = predict.Projector(df, trained, season=2025, week=13)
    results = projector.run()                       # all positions, all players

Or from the command line:

    python -m nfl_projections build-data
    python -m nfl_projections predict --season 2025 --week 13
    python -m nfl_projections backtest --season 2025 --weeks 1-17
"""

__version__ = "1.0.0"
