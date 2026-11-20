"""Expected fantasy points from volume: a correction for the hot RBs and WRs.

The board under-uses volume for the players whose recent scoring is high. For
an RB or WR whose trailing-5 form clears a fixed line, this season's targets and
carries, valued at league points-per-opportunity (xFP), predict the board's
error, and correcting for it pays out of sample in both seasons tested
(experiments/opportunity_gate.py, `tier5` arm, on boards that carry blend.py's
top-tier weights; EXPERIMENTS.md 2026-11-20):

    full pool MAE, fit on earlier seasons only:   2025 +0.019 [+0.007, +0.030]
                                                  2024 +0.013 [-0.001, +0.026]
    RB +0.026 / +0.018, WR +0.027 / +0.019 (2025 / 2024); TE and QB nothing.

It is RB and WR only, and the tier only. Fitted on every player, the same
slope hurts the starters (the gate's `all` arm), and the fixed trailing-16 tier
the blend uses does not carry it (`tier` arm, 2024 +0.001). The players it
corrects are the ones whose recent form is high, which is where volume has just
moved.

    adjustment = slope * (xfp_prior - centre)

xfp_prior is the mean of this season's games before the week, so it needs
MIN_PRIOR_GAMES of them. The slope and centre were fitted on the 2021-24
backtest boards. The rates are league points per target / per carry by least
squares with no intercept, averaged over 2018-25. That is what the gate's
leave-one-season-out rule gives a 2026 week.
"""

import numpy as np
import pandas as pd

# Trailing-5 points per game a player must reach (the gate's mean weekly 36th RB
# and 50th WR; the two seasons' fits agree to within 0.1).
XFP_TIER_THRESHOLDS = {"RB": 7.4, "WR": 7.3}
# FanDuel points per target and per carry, league-wide, 2018-25 mean.
XFP_RATES = {"RB": (0.9671, 0.7207), "WR": (1.4815, 0.8556)}
# Residual slope on xFP and the xFP it is centred on (2021-24 tier boards).
XFP_SLOPES = {"RB": 0.2025, "WR": 0.3498}
XFP_CENTRES = {"RB": 12.367, "WR": 10.767}
MIN_PRIOR_GAMES = 3
# Off until the same-network ship check (experiments/volume_verify.py) passes.
# The backtest and the Projector both read it.
ENABLED = False


def xfp_prior(position, games_this_season):
    """Mean expected points per game from this season's games so far.

    ``games_this_season`` holds the player's regular-season games before the
    week being projected, with ``targets`` and ``carries``. NaN when the
    position is not corrected or there are too few games.
    """
    rates = XFP_RATES.get(position)
    if rates is None or games_this_season is None or len(games_this_season) < MIN_PRIOR_GAMES:
        return float("nan")
    tgt = pd.to_numeric(games_this_season.get("targets"), errors="coerce").fillna(0.0)
    car = pd.to_numeric(games_this_season.get("carries"), errors="coerce").fillna(0.0)
    return float(tgt.mean() * rates[0] + car.mean() * rates[1])


def adjustment(position, form5, xfp):
    """Points to add to a blended projection; 0.0 outside the tier."""
    thr = XFP_TIER_THRESHOLDS.get(position)
    slope = XFP_SLOPES.get(position)
    if thr is None or slope is None:
        return 0.0
    try:
        f, x = float(form5), float(xfp)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(f) or np.isnan(x) or f < thr:
        return 0.0
    return float(slope * (x - XFP_CENTRES[position]))
