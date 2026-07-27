import numpy as np
import pandas as pd

from nfl_projections.scoring import fanduel_points


def test_passing_only():
    # 250 pass yds, 2 TD, 1 INT = 10 + 8 - 1 = 17
    assert fanduel_points(passing_yards=250, passing_tds=2, interceptions=1) == 17.0


def test_passing_300_yard_bonus():
    # 300 yds crosses the bonus threshold: 12 + 3 = 15
    assert fanduel_points(passing_yards=300) == 15.0
    assert fanduel_points(passing_yards=299) == 299 * 0.04


def test_rushing_and_receiving_bonuses():
    # 100 rush yds: 10 + 3 bonus
    assert fanduel_points(rushing_yards=100) == 13.0
    # 100 rec yds + 5 catches: 10 + 3 + 2.5
    assert fanduel_points(receiving_yards=100, receptions=5) == 15.5


def test_fumbles_negative():
    assert fanduel_points(fumbles=2) == -4.0


def test_nan_counts_as_zero():
    assert fanduel_points(passing_yards=float("nan"), rushing_tds=1) == 6.0


def test_array_input():
    result = fanduel_points(
        passing_yards=pd.Series([300, 0]),
        rushing_yards=pd.Series([0, 100]),
    )
    assert isinstance(result, np.ndarray)
    assert list(result) == [15.0, 13.0]


def test_full_stat_line():
    # QB: 320 pass yds, 3 TD, 1 INT, 25 rush yds
    expected = 320 * 0.04 + 3 + 3 * 4 - 1 + 25 * 0.1
    assert fanduel_points(
        passing_yards=320, passing_tds=3, interceptions=1, rushing_yards=25
    ) == expected
