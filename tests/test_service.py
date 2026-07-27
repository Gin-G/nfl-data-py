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
