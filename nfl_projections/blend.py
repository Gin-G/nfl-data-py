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

# The component stat lines shrink toward the positional mean the same way the
# points did, and it shows most where the true value is near zero: the model
# would not project a QB below ~8 rushing yards a game when Matthew Stafford
# actually gained 0.1 (Cousins 0.7, Goff 2.6). Over a season that reads as ~160
# rushing yards for a quarterback who gained one. Ordering is fine —
# correlation 0.92 — so this is a scale problem, and blending each component
# against that player's own trailing average for THAT stat fixes it.
#
# Season-level per-game MAE on the 2025 backtest, raw -> 50/50 blend:
#   QB passing_yards 25.37 -> 17.10   QB rushing_yards 4.49 -> 2.81
#   RB rushing_yards  7.23 ->  4.79   RB receiving_yards 3.43 -> 2.00
#   WR receiving_yards 6.45 -> 4.13   TE receiving_yards 4.73 -> 2.86
# A 30-45% cut at every component. (A fitted affine recalibration on top wins a
# further ~5-15% on 9 of 11 components but needs a per-position calibration
# artifact carried with the model — measured, deliberately not shipped.)
COMPONENT_FORM_COLS = {
    "passing_yards": "passing_yards_roll5",
    "rushing_yards": "rushing_yards_roll5",
    "receiving_yards": "receiving_yards_roll5",
    "receptions": "receptions_roll5",
}
COMPONENT_MODEL_WEIGHT = 0.5


# A trailing "5-game average" is built with min_periods=1, so it can be a single
# game. Below this many games the window is not form, it is one number, and the
# blend falls back to the model.
#
# GUARD RAIL, NOT A MEASURED WIN — say so rather than dressing it up. On 2025 it
# looked like a real gain (thin-window MAE 3.065 -> 2.825, overall -0.008) but
# that did not replicate on 2024 (2.971 -> 2.967, overall -0.0001). It is kept
# because it is never worse on either season and because trusting a one-game
# average as much as a five-game one is indefensible on its face; the benchmark
# population is mostly established players and barely exercises the case.
#
# Note this is NOT what protects a player like Jonathon Brooks (two seasons, two
# ACL tears, no recent snaps). The Projector routes anyone with fewer than two
# games in the season-or-prior window to the draft-capital rookie prior before
# any blending happens, so his projection never touches a stale stat line.
MIN_FORM_GAMES = 3


def form_adequacy(n_games):
    """How far to trust a player's trailing form, in [0, 1]."""
    if n_games is None:
        return 1.0
    return 1.0 if n_games >= MIN_FORM_GAMES else 0.0


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


def blend_value(projection, recent_form, position, weights=None, n_games=None):
    """Blend one projection with one recent-form number.

    ``recent_form`` of None/NaN (a player with no scoring history — rookies,
    debutants) leaves the projection untouched: there is no form to blend.
    ``n_games`` is how many games back the form window, and shrinks the weight
    on it when that is too few to mean anything (see MIN_FORM_GAMES).
    """
    if recent_form is None or (isinstance(recent_form, float) and np.isnan(recent_form)):
        return float(projection)
    form_w = (1.0 - model_weight(position, weights)) * form_adequacy(n_games)
    return (1.0 - form_w) * float(projection) + form_w * float(recent_form)


def blend_component_value(projection, form, weight=COMPONENT_MODEL_WEIGHT,
                          n_games=None):
    """Blend one component stat with the player's trailing average for it.

    Clipped at 0 — a blend toward a near-zero trailing average can otherwise go
    slightly negative, and no one rushes for -3 yards a game in expectation.
    Missing form (no history) leaves the projection alone, and a window too thin
    to be form falls back to the model (see MIN_FORM_GAMES).
    """
    if form is None or (isinstance(form, float) and np.isnan(form)):
        return float(projection)
    form_w = (1.0 - weight) * form_adequacy(n_games)
    return max(0.0, (1.0 - form_w) * float(projection) + form_w * float(form))


def blend_components(result, stat_rows, weight=COMPONENT_MODEL_WEIGHT,
                     fallback_ratio=1.0, n_games=None):
    """Blend every component in ``result`` that has a trailing column.

    ``result`` is a per-player prediction dict. Components with a rolling
    feature (yards, receptions) are blended against their own trailing average.
    Components without one — the TD stats, which have no rolling column — fall
    back to ``fallback_ratio``, the scaling the points blend applied, so they
    stay consistent with the projection they belong to.
    """
    out = dict(result)
    for stat, value in result.items():
        if stat == "fanduel_fantasy_points":
            continue
        col = COMPONENT_FORM_COLS.get(stat)
        if col and col in stat_rows.columns:
            form = pd.to_numeric(stat_rows[col], errors="coerce").iloc[0]
            out[stat] = blend_component_value(value, form, weight, n_games=n_games)
        else:
            out[stat] = value * fallback_ratio
    return out


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
