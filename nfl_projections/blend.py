"""Blend the network's projection with the player's recent scoring form.

The network is trained to minimize per-game error on a target with enormous
single-game variance, so the error-minimizing prediction is a heavily shrunk
one. Measured on the 2025 backtest, predicted spread across players is only
0.33 (QB) to 0.70 (RB/WR/TE) of the actual spread, and the regression of actual
on predicted has slope 1.38-2.18 instead of 1.0. The model ranks players *worse*
than simply averaging their last five games (QB season-mean correlation 0.711
vs the naive 0.782) even though it beats that average on per-game MAE.

That shrinkage is correct for MAE and wrong for every use that compares players:
season boards, tiers, "is Allen better than Goff". The two predictors have
different biases, so blending them beats both. Weights fitted on the 2024
backtest and applied unchanged to 2025 improved per-game MAE, ranking
correlation and spread simultaneously: overall MAE 4.178 -> 4.124, QB
correlation 0.711 -> 0.779, QB spread 0.33 -> 0.65 of actual, and the
regression of actual on predicted from slope 2.18 to 1.20.

The naive side is `fanduel_fantasy_points_roll5`, which is already one of the
network's own input features — the network has this information and
under-weights it. The blend just restores some of that weight.
"""

import numpy as np
import pandas as pd

# Weight on the model; the remainder goes to trailing-5 form. Each season's own
# MAE-optimal weights differ (2024 wanted 0.5/0.3/0.3/0.7, 2025 wanted
# 0.6/0.6/0.7/0.7), so these sit between them — and land within 0.02 MAE of
# each season's optimum on BOTH seasons. The curve is flat from ~0.3 to ~0.7,
# so nothing here is knife-edge.
DEFAULT_MODEL_WEIGHTS = {"QB": 0.5, "RB": 0.5, "WR": 0.5, "TE": 0.6}
DEFAULT_WEIGHT = 0.5

# Projecting a season that hasn't started is a different problem: there is no
# current form, and the prior season's FULL average beats its last five games
# at predicting the next season (correlation 0.919 vs 0.886 over 2022-25).
# Measured on a preseason-board test — 2025 week-1 projections, which see only
# 2024, scored against 2025 season averages — blending against the season
# average cut MAE hard: QB 4.18 -> 3.29, WR 2.54 -> 2.10, RB 2.66 -> 2.39,
# TE 1.98 -> 1.82. The optimum also sits lower (more weight on history).
PRESEASON_MODEL_WEIGHTS = {"QB": 0.4, "RB": 0.5, "WR": 0.4, "TE": 0.4}
PRESEASON_WEIGHT = 0.4

RECENT_FORM_COL = "fanduel_fantasy_points_roll5"   # in-season: current form
SEASON_FORM_COL = "avg_fppg"                       # preseason: points per game
                                                   # entering the week (expanding,
                                                   # carries over from last season)


def weights_for_mode(preseason=False):
    """Per-position model weights for the in-season or preseason blend."""
    return PRESEASON_MODEL_WEIGHTS if preseason else DEFAULT_MODEL_WEIGHTS


def form_col_for_mode(preseason=False):
    """Which form column to blend against in each mode."""
    return SEASON_FORM_COL if preseason else RECENT_FORM_COL


def model_weight(position, weights=None):
    """Model weight for a position; the rest goes to recent form."""
    weights = DEFAULT_MODEL_WEIGHTS if weights is None else weights
    default = PRESEASON_WEIGHT if weights is PRESEASON_MODEL_WEIGHTS else DEFAULT_WEIGHT
    return float(weights.get(position, default))


def blend_value(projection, recent_form, position, weights=None):
    """Blend one projection with one recent-form number.

    ``recent_form`` of None/NaN (a player with no scoring history — rookies,
    debutants) leaves the projection untouched: there is no form to blend.
    """
    if recent_form is None or (isinstance(recent_form, float) and np.isnan(recent_form)):
        return float(projection)
    w = model_weight(position, weights)
    return w * float(projection) + (1.0 - w) * float(recent_form)


def recent_form_from_rows(stat_rows, col=None, preseason=False):
    """Pull each player's scoring form out of their latest stat row.

    In-season that is the trailing-5 mean — a rolling mean over the player's
    last five games *including* the row's own game and spanning season
    boundaries, so on a player's most recent row it is exactly the form
    available for projecting his next game. For a preseason board it is the
    season average instead (see PRESEASON_MODEL_WEIGHTS).
    """
    col = col or form_col_for_mode(preseason)
    if col not in stat_rows.columns:
        return pd.Series(np.nan, index=stat_rows.index)
    return pd.to_numeric(stat_rows[col], errors="coerce")


def blend_frame(frame, recent_form, positions=None, weights=None,
                points_col="fanduel_fantasy_points", scale_cols=None):
    """Blend a whole projection frame in place-safe fashion; returns a copy.

    ``scale_cols`` (floor/ceiling/components) are scaled by the same ratio the
    blend applied to ``points_col``, so a band or a stat line stays consistent
    with its projection instead of drifting away from it.
    """
    out = frame.copy()
    if points_col not in out.columns:
        return out

    form = pd.to_numeric(pd.Series(recent_form, index=out.index), errors="coerce")
    if positions is None:
        positions = out["position"] if "position" in out.columns else pd.Series("", index=out.index)
    positions = pd.Series(positions, index=out.index).astype(str)

    w = positions.map(lambda p: model_weight(p, weights)).astype(float)
    base = pd.to_numeric(out[points_col], errors="coerce")
    blended = w * base + (1.0 - w) * form
    blended = blended.where(form.notna(), base)  # no history -> leave alone

    ratio = (blended / base.replace(0, np.nan)).fillna(1.0)
    out[points_col] = blended
    for col in (scale_cols or []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * ratio
    return out
