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


# ── fanduel_points_from_stats: the one rule every module scores nflverse by ──

def _frame(**cols):
    import pandas as pd
    return pd.DataFrame({k: [v] for k, v in cols.items()})


def test_from_stats_charges_lost_fumbles_only():
    from nfl_projections.scoring import fanduel_points_from_stats
    kept = _frame(rushing_yards=50, rushing_fumbles=1, rushing_fumbles_lost=0)
    lost = _frame(rushing_yards=50, rushing_fumbles=1, rushing_fumbles_lost=1)
    sack = _frame(sack_fumbles=1, sack_fumbles_lost=1)
    assert fanduel_points_from_stats(kept).iloc[0] == 5.0
    assert fanduel_points_from_stats(lost).iloc[0] == 3.0
    assert fanduel_points_from_stats(sack).iloc[0] == -2.0


def test_from_stats_never_falls_back_to_all_fumbles():
    """Without the lost columns a fumble costs nothing: falling back to the
    all-fumbles columns would quietly bring the old rule back."""
    from nfl_projections.scoring import fanduel_points_from_stats
    assert fanduel_points_from_stats(_frame(rushing_yards=50, rushing_fumbles=2)).iloc[0] == 5.0


def test_from_stats_pays_two_point_conversions():
    from nfl_projections.scoring import fanduel_points_from_stats
    for col in ("passing_2pt_conversions", "rushing_2pt_conversions",
                "receiving_2pt_conversions"):
        assert fanduel_points_from_stats(_frame(**{col: 1})).iloc[0] == 2.0


def test_from_stats_reads_either_interception_name_and_survives_nan():
    from nfl_projections.scoring import fanduel_points_from_stats
    assert fanduel_points_from_stats(_frame(passing_interceptions=2)).iloc[0] == -2.0
    assert fanduel_points_from_stats(_frame(interceptions=2)).iloc[0] == -2.0
    assert fanduel_points_from_stats(_frame(passing_yards=float("nan"), rushing_tds=1)).iloc[0] == 6.0


def test_every_module_scores_the_same_way():
    """The training target, opportunity, FPA and scarcity used to disagree."""
    import pandas as pd
    from nfl_projections import fpa, opportunity
    from nfl_projections.scoring import fanduel_points_from_stats
    row = _frame(passing_yards=312, passing_tds=2, passing_interceptions=1,
                 rushing_yards=104, rushing_tds=1, receptions=3, receiving_yards=22,
                 rushing_fumbles=2, rushing_fumbles_lost=1, sack_fumbles_lost=1,
                 passing_2pt_conversions=1)
    want = fanduel_points_from_stats(row).iloc[0]
    assert opportunity._points(row).iloc[0] == want
    assert fpa._points(row).iloc[0] == want
    assert isinstance(fanduel_points_from_stats(row), pd.Series)
