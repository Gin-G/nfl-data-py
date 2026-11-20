import numpy as np
import pandas as pd
import pytest

from nfl_projections import volume


def games(n, targets=8, carries=0):
    return pd.DataFrame({"targets": [targets] * n, "carries": [carries] * n})


def test_xfp_prior_values_this_seasons_volume():
    t, c = volume.XFP_RATES["WR"]
    assert volume.xfp_prior("WR", games(4, 8, 1)) == pytest.approx(8 * t + 1 * c)


def test_xfp_prior_needs_enough_games_and_a_corrected_position():
    assert np.isnan(volume.xfp_prior("WR", games(volume.MIN_PRIOR_GAMES - 1)))
    assert np.isnan(volume.xfp_prior("QB", games(6)))
    assert np.isnan(volume.xfp_prior("TE", games(6)))
    assert np.isnan(volume.xfp_prior("WR", None))


def test_adjustment_only_inside_the_tier():
    thr = volume.XFP_TIER_THRESHOLDS["RB"]
    c = volume.XFP_CENTRES["RB"]
    assert volume.adjustment("RB", thr - 0.01, c + 5) == 0.0
    assert volume.adjustment("RB", thr, c + 5) == pytest.approx(volume.XFP_SLOPES["RB"] * 5)
    assert volume.adjustment("RB", thr, c - 5) == pytest.approx(-volume.XFP_SLOPES["RB"] * 5)


def test_adjustment_ignores_missing_inputs_and_other_positions():
    assert volume.adjustment("WR", float("nan"), 12.0) == 0.0
    assert volume.adjustment("WR", 20.0, float("nan")) == 0.0
    assert volume.adjustment("QB", 30.0, 40.0) == 0.0
    assert volume.adjustment("TE", 30.0, 40.0) == 0.0


def test_constants_cover_the_same_positions():
    keys = set(volume.XFP_TIER_THRESHOLDS)
    assert keys == set(volume.XFP_RATES) == set(volume.XFP_SLOPES) == set(volume.XFP_CENTRES) == {"RB", "WR"}


def test_backtest_volume_adjust_reads_prior_games_only():
    from nfl_projections.evaluate import _volume_adjust

    prior = pd.DataFrame({
        "player_id": ["a"] * 6,
        "season": [2024, 2025, 2025, 2025, 2025, 2025],
        "week": [17, 1, 2, 3, 4, 5],
        "fanduel_fantasy_points": [30.0] * 6,
        "targets": [99, 10, 10, 10, 10, 10],   # the 2024 game must not count toward xFP
        "carries": [0] * 6,
    })
    got = _volume_adjust(prior, np.array(["a", "b"]), np.array(["WR", "WR"]), 2025)
    xfp = 10 * volume.XFP_RATES["WR"][0]
    assert got[0] == pytest.approx(volume.XFP_SLOPES["WR"] * (xfp - volume.XFP_CENTRES["WR"]))
    assert got[1] == 0.0
