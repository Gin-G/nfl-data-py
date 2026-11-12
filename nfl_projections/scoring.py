"""FanDuel fantasy point calculation. Works on scalars or pandas/numpy arrays."""

import numpy as np

from .config import FANDUEL_SCORING


def _clean(val):
    """Replace NaN with 0 and return a float array."""
    return np.nan_to_num(np.asarray(val, dtype=float), nan=0.0)


def fanduel_points(
    passing_yards=0, passing_tds=0, interceptions=0,
    rushing_yards=0, rushing_tds=0,
    receptions=0, receiving_yards=0, receiving_tds=0,
    fumbles=0, return_tds=0, two_point_conversions=0,
    field_goals_0_39=0, field_goals_40_49=0, field_goals_50_plus=0,
    extra_points=0, scoring=None,
):
    """Calculate FanDuel fantasy points, including yardage bonuses.

    All arguments accept scalars or array-likes of the same length; NaN values
    count as 0. Returns a float for scalar input, an ndarray otherwise.
    """
    s = scoring or FANDUEL_SCORING

    passing_yards = _clean(passing_yards)
    rushing_yards = _clean(rushing_yards)
    receiving_yards = _clean(receiving_yards)

    points = (
        passing_yards * s["passing_yards"]
        + _clean(passing_tds) * s["passing_tds"]
        + _clean(interceptions) * s["interceptions"]
        + np.where(passing_yards >= 300, s["passing_bonus_300"], 0)
        + rushing_yards * s["rushing_yards"]
        + _clean(rushing_tds) * s["rushing_tds"]
        + np.where(rushing_yards >= 100, s["rushing_bonus_100"], 0)
        + _clean(receptions) * s["receptions"]
        + receiving_yards * s["receiving_yards"]
        + _clean(receiving_tds) * s["receiving_tds"]
        + np.where(receiving_yards >= 100, s["receiving_bonus_100"], 0)
        + _clean(fumbles) * s["fumbles"]
        + _clean(return_tds) * s["return_tds"]
        + _clean(two_point_conversions) * s["two_point_conversions"]
        + _clean(field_goals_0_39) * s["field_goals_0_39"]
        + _clean(field_goals_40_49) * s["field_goals_40_49"]
        + _clean(field_goals_50_plus) * s["field_goals_50_plus"]
        + _clean(extra_points) * s["extra_points"]
    )

    if points.ndim == 0:
        return float(points)
    return points


def fanduel_points_from_stats(stats, scoring=None):
    """FanDuel points for every row of an nflverse weekly stats frame.

    The one place that decides which nflverse columns a FanDuel rule reads, so
    the training target, the accuracy record and every report score the same
    way. FanDuel charges -2 for a fumble LOST (rushing, receiving and sack) and
    pays +2 for a two-point conversion (passing, rushing or receiving). Until
    2026-11-12 the training target charged every fumble and paid nothing for a
    conversion, while three other modules each had their own version.

    Only the lost-fumble columns are read: falling back to the all-fumbles
    ones when they are absent would quietly restore the old rule. A missing
    column counts as zero. Returns a float Series on the frame's index.
    """
    import pandas as pd

    def col(*names):
        for name in names:
            if name in stats.columns:
                return pd.to_numeric(stats[name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=stats.index)

    return pd.Series(fanduel_points(
        passing_yards=col("passing_yards"), passing_tds=col("passing_tds"),
        interceptions=col("passing_interceptions", "interceptions"),
        rushing_yards=col("rushing_yards"), rushing_tds=col("rushing_tds"),
        receptions=col("receptions"), receiving_yards=col("receiving_yards"),
        receiving_tds=col("receiving_tds"),
        fumbles=(col("rushing_fumbles_lost") + col("receiving_fumbles_lost")
                 + col("sack_fumbles_lost")),
        two_point_conversions=(col("passing_2pt_conversions")
                               + col("rushing_2pt_conversions")
                               + col("receiving_2pt_conversions")),
        scoring=scoring,
    ), index=stats.index, dtype=float)
