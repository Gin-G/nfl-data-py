"""Depth-chart role → volume / availability logic for projections.

The weekly model produces a matchup-neutral per-game number from a player's own
rolling form. That number has NO notion of role: an injury fill-in who started a
few games carries a starter-level per-game rate, and the naive season assembly
then multiplies it by ~17 games for every rostered player. The result is backup
QBs who only played because the starter was hurt (Tyrod/Fields/Huntley) landing
as top-12 season QBs, and two RBs on the same team (Javonte + a fill-in) both
projecting as if each gets a bell-cow's workload.

This module supplies three conservative, role-based corrections:

  * ``per_game_role_multiplier`` — a snap-share proxy applied to the weekly number
    so a clear non-starter isn't read at a starter's per-game rate. This replaces
    the old predict.py penalty, which only fired for ``role == "backup"``
    (pos_rank 2 — deep backups escaped entirely) and, for skill players, was
    gated on ``avg_fppg < 8.0`` (inverted: the productive fill-ins we most want
    to discount were the ones that escaped).

  * ``expected_games`` — availability. How many games a player at a given depth
    rank actually plays a role in. This is the stark lever for QBs: a backup
    behind a healthy starter plays ~1-2 meaningful games, so multiplying his
    per-game form by 17 is the core bug behind "top-12 QB who isn't the starter".

  * ``position_budgets`` + the season-assembly cap — a finite team-position pool.
    Summing every rostered RB's "expected points if he plays" wildly exceeds a
    real team's single-game RB output; the budget forces same-team players to
    SHARE, so two backs can't both be bell cows.

Depth charts are preseason guesses, so every constant here is deliberately coarse
and tunable. The role multiplier sets the *split* between teammates; the budget
sets the *ceiling* on their combined total — they compose without double-counting.
"""
from __future__ import annotations

import math

import pandas as pd

# Rank we assume when a player has no usable depth-chart rank (not on the chart,
# or the name-match failed). Treated as a mid-roster backup: neither buried nor
# handed a starter's role on faith.
_UNKNOWN_RANK = 3

# Per-game production multiplier by position and depth rank (snap-share proxy).
# QBs are near-binary (you start or you don't); skill rooms rotate, so the taper
# is gentler and the team budget does most of the volume-sharing work.
_PER_GAME_MULT = {
    "QB": {1: 1.0, 2: 0.35, 3: 0.15},
    "RB": {1: 1.0, 2: 0.80, 3: 0.55, 4: 0.30},
    "WR": {1: 1.0, 2: 0.92, 3: 0.72, 4: 0.50, 5: 0.30},
    "TE": {1: 1.0, 2: 0.55, 3: 0.30},
}
_PER_GAME_DEEP = {"QB": 0.10, "RB": 0.18, "WR": 0.20, "TE": 0.20}

# Expected games a player at a depth rank actually fills a role in. QBs are
# all-or-nothing; skill backups are usually active (their volume is trimmed by
# the multiplier / budget, not by sitting), with a taper only for deep roster.
_EXPECTED_GAMES = {
    "QB": {1: 17.0, 2: 1.5, 3: 0.5},
    "RB": {1: 16.5, 2: 16.0, 3: 15.0, 4: 11.0, 5: 6.0},
    "WR": {1: 16.5, 2: 16.0, 3: 15.0, 4: 12.0, 5: 8.0},
    "TE": {1: 16.5, 2: 15.0, 3: 12.0, 4: 7.0},
}
_EXPECTED_GAMES_DEEP = {"QB": 0.3, "RB": 4.0, "WR": 5.0, "TE": 4.0}

# Fallback per-game team totals (fantasy points) for the finite-pool cap, used
# when no historical dataset is supplied. Position group ~= a realistic single
# game's combined output for that position on one team.
# Fallback pools when no dataset is available: the p90 team-season average
# per-game output at each position (2024-25), matching what position_budgets
# computes. A budget at the league MEAN truncates every good offense.
DEFAULT_BUDGETS = {"QB": 22.4, "RB": 31.0, "WR": 41.9, "TE": 14.4}

_SKILL = ("RB", "WR", "TE")


def norm_rank(depth_rank) -> int:
    """Coerce a depth-chart rank (float, str, ``'N/A'``, NaN, None) to an int,
    falling back to ``_UNKNOWN_RANK`` when it can't be read."""
    if depth_rank is None:
        return _UNKNOWN_RANK
    try:
        if isinstance(depth_rank, float) and math.isnan(depth_rank):
            return _UNKNOWN_RANK
    except (TypeError, ValueError):
        pass
    try:
        rank = int(float(depth_rank))
    except (TypeError, ValueError):
        return _UNKNOWN_RANK
    return rank if rank >= 1 else _UNKNOWN_RANK


def _lookup(table: dict, deep: dict, position, depth_rank, default: float) -> float:
    """Rank-keyed lookup that clamps ranks past the last explicit tier to the
    position's 'deep' value (and unknown positions to ``default``)."""
    pos = str(position).upper() if position is not None else ""
    tiers = table.get(pos)
    if tiers is None:
        return default
    rank = norm_rank(depth_rank)
    if rank in tiers:
        return tiers[rank]
    return deep.get(pos, min(tiers.values()))


def per_game_role_multiplier(position, depth_rank) -> float:
    """Snap-share proxy in [0, 1]: how much of a starter's per-game production a
    player at this depth rank is expected to put up. Unknown position -> 1.0."""
    return _lookup(_PER_GAME_MULT, _PER_GAME_DEEP, position, depth_rank, 1.0)


def expected_games(position, depth_rank, *, scheduled_games: float = 17.0) -> float:
    """Games a player at this depth rank is expected to fill a role in, capped at
    the number actually scheduled. Unknown position -> all scheduled games."""
    g = _lookup(_EXPECTED_GAMES, _EXPECTED_GAMES_DEEP, position, depth_rank, scheduled_games)
    return min(g, scheduled_games)


def play_weight(position, depth_rank, *, scheduled_games: float = 17.0) -> float:
    """Fraction of scheduled games this depth role is expected to play (availability),
    in [0, 1]. A starter ~1.0; a backup QB behind a healthy starter ~0.1. Multiply a
    per-game projection by this to spread it across the games actually played."""
    if not scheduled_games:
        return 1.0
    return expected_games(position, depth_rank, scheduled_games=scheduled_games) / scheduled_games


def apply_team_budget(frame: pd.DataFrame, budgets: dict, *, points_col: str,
                      group_cols=("team", "position"), scale_cols=None) -> pd.DataFrame:
    """Enforce a finite per-group scoring pool: within each group (e.g. team+position),
    if the summed ``points_col`` exceeds the position's budget, scale every player in the
    group by the same factor so the pool is finite and teammates SHARE. Only scales DOWN,
    preserving the role-based split. ``scale_cols`` (default: just ``points_col``) are all
    columns scaled by that factor — pass the projection + its components + floor/ceiling to
    keep them consistent. Returns the same frame (mutated in place)."""
    if frame.empty or not budgets:
        return frame
    group_cols = [c for c in group_cols if c in frame.columns]
    if not group_cols or points_col not in frame.columns or "position" not in frame.columns:
        return frame
    scale_cols = [c for c in (scale_cols or [points_col]) if c in frame.columns]
    grp_sum = frame.groupby(group_cols)[points_col].transform("sum")
    budget = frame["position"].map(budgets)
    factor = (budget / grp_sum).where(budget.notna() & (grp_sum > budget), 1.0)
    factor = factor.clip(upper=1.0).fillna(1.0)
    for c in scale_cols:
        frame[c] = frame[c].astype(float) * factor
    return frame


def position_budgets(dataset: pd.DataFrame | None = None, *, recent_seasons: int = 2,
                     fp_col: str = "fanduel_fantasy_points",
                     quantile: float | None = 0.90) -> dict:
    """Per-game team fantasy-point pool by position, for the finite-pool cap.

    The reference is each TEAM-SEASON's average per-game pool, and the budget is
    the ``quantile`` of that across team-seasons — i.e. "what a top offense
    averages at this position". The cap is applied to projections, which are
    means, so the ceiling has to allow for a genuinely good offense.

    This used to be the league mean over (season, week, team), which was far too
    tight: 40-46% of real team-games exceeded it at every position, so every
    above-average offense was truncated to average. That silently flattened the
    board — most visibly at QB, where one starter carries the whole group and
    the elite QBs all landed pinned just under a 17.3-point ceiling.

    Pass ``quantile=None`` for the old mean behaviour. Falls back to
    :data:`DEFAULT_BUDGETS` when the dataset lacks the columns or a position.
    """
    budgets = dict(DEFAULT_BUDGETS)
    if dataset is None:
        return budgets
    needed = {"season", "week", "position", fp_col}
    tcol = next((c for c in ("team", "recent_team", "team_abbr", "club_code")
                 if c in dataset.columns), None)
    if tcol is None or not needed <= set(dataset.columns):
        return budgets

    df = dataset[dataset["week"] != "AVG"].copy() if "week" in dataset.columns else dataset.copy()
    df = df[df["position"].isin(_SKILL + ("QB",))]
    if df.empty:
        return budgets
    seasons = sorted(pd.to_numeric(df["season"], errors="coerce").dropna().unique())
    if seasons:
        keep = set(seasons[-recent_seasons:])
        df = df[pd.to_numeric(df["season"], errors="coerce").isin(keep)]

    per_game = (df.groupby(["season", "week", tcol, "position"])[fp_col]
                  .sum().reset_index())
    if quantile is None:
        levels = per_game.groupby("position")[fp_col].mean()
    else:
        team_season = (per_game.groupby(["season", tcol, "position"])[fp_col]
                               .mean().reset_index())
        levels = team_season.groupby("position")[fp_col].quantile(quantile)
    for pos, val in levels.items():
        if pd.notna(val) and val > 0:
            budgets[pos] = float(val)
    return budgets
