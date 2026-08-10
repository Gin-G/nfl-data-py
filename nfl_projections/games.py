"""Availability / games-played model.

Season fantasy total = per-game rate x GAMES PLAYED. The rate is the rolling-form model's job
and is at its ceiling; games played is a separate, un-modeled factor: the pipeline assumed a flat
~16.5 games for every starter (roles.expected_games gives games by depth ROLE only).

Measured out-of-sample (EXPERIMENTS.md, role players g_prev>=8, 2018-2025), predicting a season's
games, games-MAE:
  - flat 16.5 (old assumption)                 7.51   <- too high; role players average ~14 games
  - position mean (~14)                        7.29
  - REGRESS prior-year games toward the mean   6.48   <- ~14% better, the shipped model
  - raw "repeat last year"                     8.47   <- WORSE; the shrinkage is the whole trick
  - adding age / age^2 / workload              ~6.5   <- NO out-of-sample gain (age curve was
                                                          in-sample leakage; a clean negative result)

So durability comes from *shrunk prior availability*, not an aging curve. Two effects combine: a
LEVEL correction (16.5 -> ~14, most starters miss a few games) and a PLAYER term (a back who played
11 last year isn't a 16.5 bet). Games is intrinsically noisy (much is irreducible injury luck), so
the win is modest but real and additive — it dodges the per-game variance wall.

Model: per position, ridge fit  games ~ intercept + slope * prior_year_games  (slope << 1 =
shrinkage). Use expected_games_from_history() for established players (with prior-year games); fall
back to roles.expected_games (depth role) for rookies / no-history players.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import roles

SKILL = ("QB", "RB", "WR", "TE")
_SCHED = 17.0
_MIN_ROLE_GAMES = 8  # "had a role" threshold for fitting


def games_history(dataset: pd.DataFrame) -> pd.DataFrame:
    """Per (season, player_id, position) REGULAR-SEASON games played, with the
    prior season's games.

    Postseason games are excluded on purpose. Counting them made "games played"
    partly a measure of the team's playoff run: a Super Bowl quarterback showed
    21 games and drew a 0.99 availability weight, while a quarterback who
    started all 17 regular-season games for an eliminated team showed 17 and
    drew 0.83 — a 20% penalty for his team losing, dressed up as durability.
    """
    df = dataset.copy()
    if "week" in df.columns:
        df = df[df["week"] != "AVG"]
    if "season_type" in df.columns:
        df = df[(df["season_type"] == "REG") | df["season_type"].isna()]
    df = df[df["position"].isin(SKILL)]
    df["fp"] = pd.to_numeric(df["fanduel_fantasy_points"], errors="coerce")
    h = (df.groupby(["season", "player_id", "position"]).agg(games=("fp", "size")).reset_index()
           .sort_values(["player_id", "season"]))
    h["prev_games"] = h.groupby("player_id")["games"].shift(1)
    return h


def fit_games_model(dataset: pd.DataFrame, *, max_season: int | None = None,
                    ridge_lambda: float = 10.0) -> dict:
    """Per-position ridge of games on prior-year games (intercept + shrunk slope). Fit on role
    players (prev_games >= 8) in seasons <= max_season. Returns {position: (intercept, slope)},
    plus a per-position residual std for an availability spread, under key (pos, 'std')."""
    h = games_history(dataset).dropna(subset=["prev_games"])
    h = h[h["prev_games"] >= _MIN_ROLE_GAMES]
    if max_season is not None:
        h = h[h["season"] <= max_season]
    model: dict = {}
    for pos, g in h.groupby("position"):
        if len(g) < 20:
            model[pos] = (float(g["games"].mean()) if len(g) else 13.0, 0.0)
            model[(pos, "std")] = float(g["games"].std() or 4.0)
            continue
        X = np.c_[np.ones(len(g)), g["prev_games"].to_numpy(float)]
        y = g["games"].to_numpy(float)
        # don't regularize the intercept
        R = ridge_lambda * np.diag([0.0, 1.0])
        a, b = np.linalg.solve(X.T @ X + R, X.T @ y)
        model[pos] = (float(a), float(b))
        model[(pos, "std")] = float(np.std(y - X @ [a, b]) or 4.0)
    return model


def expected_games_from_history(position, prev_games, model: dict, *,
                                scheduled_games: float = _SCHED) -> float:
    """Predicted games for an established player from the fitted model, clipped to
    [1, scheduled_games]. ``prev_games`` is the player's games in the prior season."""
    coefs = model.get(position)
    if coefs is None or prev_games is None or (isinstance(prev_games, float) and np.isnan(prev_games)):
        return float("nan")
    a, b = coefs
    return float(np.clip(a + b * float(prev_games), 1.0, scheduled_games))


def expected_games(position, depth_rank, prev_games, model: dict, *,
                   scheduled_games: float = _SCHED) -> float:
    """Best available games estimate: the history model for an established role player
    (depth rank 1-2 with prior-year games), else the depth-role baseline (roles.expected_games)
    for rookies / backups / no-history — avoiding the role-vs-durability conflation of prior
    games for players whose role is set by the depth chart, not their (missing) history."""
    role_games = roles.expected_games(position, depth_rank, scheduled_games=scheduled_games)
    rank = roles.norm_rank(depth_rank)
    hist = expected_games_from_history(position, prev_games, model, scheduled_games=scheduled_games)
    if rank <= 2 and hist == hist:  # established starter/committee with history
        return hist
    return role_games


def prev_games_map(dataset: pd.DataFrame, season: int) -> dict:
    """{player_id: games played in ``season`` - 1} for feeding the model at projection time."""
    h = games_history(dataset)
    prev = h[h["season"] == season - 1]
    return dict(zip(prev["player_id"], prev["games"]))


# --------------------------------------------------------------------------- backtest

def backtest_games(dataset: pd.DataFrame, *, test_seasons=None) -> dict:
    """Out-of-sample games-MAE for the shipped model (shrunk prior games) vs baselines
    (flat 16.5, position mean, raw repeat-last-year), on role players (prev_games >= 8)."""
    h = games_history(dataset).dropna(subset=["prev_games"])
    h = h[h["prev_games"] >= _MIN_ROLE_GAMES]
    seasons = sorted(h["season"].unique())
    test_seasons = test_seasons or seasons[2:]
    rows = []
    for Y in test_seasons:
        if Y not in seasons:
            continue
        model = fit_games_model(dataset, max_season=Y - 1)
        pos_mean = h[h["season"] < Y].groupby("position")["games"].mean().to_dict()
        for _, p in h[h["season"] == Y].iterrows():
            actual = p["games"]
            pred = expected_games_from_history(p["position"], p["prev_games"], model)
            rows.append({
                "position": p["position"],
                "err_model": abs((pred if pred == pred else 16.5) - actual),
                "err_flat": abs(16.5 - actual),
                "err_posmean": abs(pos_mean.get(p["position"], 13.0) - actual),
                "err_repeat": abs(p["prev_games"] - actual),
            })
    res = pd.DataFrame(rows)
    if res.empty:
        return {"n": 0}
    keys = ("model", "flat", "posmean", "repeat")
    return {"n": len(res),
            "overall": {k: round(res[f"err_{k}"].mean(), 3) for k in keys},
            "by_position": {pos: {k: round(sub[f"err_{k}"].mean(), 3) for k in keys}
                            for pos, sub in res.groupby("position")}}
