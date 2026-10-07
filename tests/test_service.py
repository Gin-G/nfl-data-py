import pandas as pd

import nfl_projections
from nfl_projections import service


def test_public_api_exposed():
    assert hasattr(nfl_projections, "ProjectionService")
    assert hasattr(nfl_projections, "project_week")


def test_results_to_frame_flattens_and_sorts():
    results = {
        "RB": pd.DataFrame({
            "player_name": ["A", "B"], "position": ["RB", "RB"],
            "fanduel_fantasy_points": [12.0, 20.0],
        }),
        "WR": pd.DataFrame({
            "player_name": ["C"], "position": ["WR"],
            "fanduel_fantasy_points": [15.0],
        }),
    }
    frame = service.results_to_frame(results)
    assert len(frame) == 3
    # best projection first
    assert frame.iloc[0]["player_name"] == "B"
    assert frame["fanduel_fantasy_points"].tolist() == [20.0, 15.0, 12.0]


def test_results_to_frame_empty():
    assert service.results_to_frame({}).empty


class TestTrainingWindow:
    """`min_season` moves how far back the model is TRAINED, leaving the dataset
    (and so the features a projection is built from) alone — what makes a
    shallow-history shadow model comparable with the deep production one."""

    def test_min_season_reaches_the_trainer(self, monkeypatch):
        import pandas as pd

        from nfl_projections import model as model_mod
        from nfl_projections import service as service_mod

        seen = {}

        def fake_train(df, **kw):
            seen.update(kw)
            return object(), []

        monkeypatch.setattr(model_mod, "train_ensemble", fake_train)
        data = pd.DataFrame({"season": [2024, 2025], "week": [1, 1]})
        service_mod.ProjectionService(dataset=data, min_season=2025, n_seeds=1, epochs=1)
        assert seen["min_season"] == 2025

    def test_default_is_the_configured_window(self, monkeypatch):
        import pandas as pd

        from nfl_projections import config
        from nfl_projections import model as model_mod
        from nfl_projections import service as service_mod

        seen = {}
        monkeypatch.setattr(model_mod, "train_ensemble",
                            lambda df, **kw: (seen.update(kw), (object(), []))[1])
        data = pd.DataFrame({"season": [2024], "week": [1]})
        service_mod.ProjectionService(dataset=data, n_seeds=1, epochs=1)
        assert seen["min_season"] == config.TRAINING_MIN_SEASON

    def test_the_dataset_is_not_filtered(self, monkeypatch):
        import pandas as pd

        from nfl_projections import model as model_mod
        from nfl_projections import service as service_mod

        monkeypatch.setattr(model_mod, "train_ensemble", lambda df, **kw: (object(), []))
        data = pd.DataFrame({"season": [2019, 2024, 2025], "week": [1, 1, 1]})
        svc = service_mod.ProjectionService(dataset=data, min_season=2025, n_seeds=1, epochs=1)
        assert sorted(svc.dataset["season"]) == [2019, 2024, 2025]
