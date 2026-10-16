"""Opportunity share: how much of the football a player has actually had.

Context for the lineup pool, deliberately **not** a correction to it.

`experiments/opportunity_share.py` measured this end to end. The premise the
numbers support, strongly: volume is very nearly a deterministic trait (a
player's opportunities per game correlate 0.974 across odd and even weeks of a
season, 0.987 across his career) while touchdowns per opportunity are close to a
coin flip (0.178 and 0.288). Two players with the same volume are very nearly
the same bet however differently they finished.

The premise does **not** survive into a board correction, and the reason is that
the board already has it. `targets`, `carries`, `target_share`,
`air_yards_share`, `wopr` and an `opportunity_score` are all input features, and
against the board's own error on five seasons of production backtests, both
opportunity share and volume-implied points come in null. The one thing left
over is `volume_td_gap` — volume-implied points minus points actually scored,
i.e. the receiver who gets the ball and has not found the end zone. That is real
(RB +0.611, WR +0.633, TE +0.722, all p=0.003, all surviving a control on the
board's own projection) and it is worth, held out of its own season, about a
third of a percent of receiver MAE and **0.2 points of top-24 overlap in 4 weeks
out of 83**. The optimizer reads nothing but the order, so that is not enough to
change a lineup with.

So these columns ride alongside `value` and the objective columns are untouched;
`tests/test_opportunity.py` asserts that. Two further results are worth knowing
while reading them:

  * A one-week spike in opportunity share is a **fade**, not a buy. On the
    clearest cases — last week's share at least double the season's — receivers
    scored 3.688 against a 4.470 projection across 997 player-weeks. The board
    chases the spike. `share_jump` is reported so the spike is visible, with its
    sign documented here rather than hidden in a lean.
  * A quarterback's opportunity is his pass attempts, not his targets. Fitting
    points on targets for a quarterback returns 5.1 points per target off a
    column that is almost entirely zeroes.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

POSITIONS = ("QB", "RB", "WR", "TE")
# The two columns that ARE the opportunity, per position. A quarterback throws;
# everyone else is thrown to.
_PASS_VOLUME = {"QB": "attempts"}
_DEFAULT_PASS_VOLUME = "targets"
# Below this many prior games a per-game rate is not a rate yet, and an early
# share is mostly which blowout he happened to play in.
MIN_PRIOR_GAMES = 3
# League points per opportunity, from the harness (least squares, no intercept,
# nine seasons). Carried as constants so a single week's pool does not have to
# refit them, which with one week of data it cannot do.
POINTS_PER_OPPORTUNITY = {
    "QB": (0.417, 0.880),      # per pass attempt, per carry
    "RB": (0.954, 0.716),      # per target, per carry
    "WR": (1.472, 0.754),
    "TE": (1.417, 1.230),
}
CONTEXT_COLUMNS = ("opp_per_game", "opp_share", "xfp_per_game",
                   "volume_td_gap", "share_jump", "prior_games")


def load_weekly(seasons=None) -> pd.DataFrame:
    """Regular-season player-weeks with the volume columns."""
    import nflreadpy as nfl

    from .dataset import load_through_latest

    seasons = list(seasons or config.SEASONS)
    stats, _ = load_through_latest(nfl.load_player_stats, seasons, "player stats")
    d = stats[stats.get("season_type", "REG") == "REG"].copy()
    return d[d.position.isin(POSITIONS) & d.player_id.notna()]


def _volume(d: pd.DataFrame) -> pd.DataFrame:
    """Add `opp` (position-appropriate) and the team share of it."""
    out = d.copy()
    for col in ("targets", "carries", "attempts", "receiving_tds", "rushing_tds",
                "passing_tds"):
        out[col] = (pd.to_numeric(out[col], errors="coerce").fillna(0.0)
                    if col in out.columns else 0.0)
    out["week"] = pd.to_numeric(out.week, errors="coerce")
    out = out[out.week.notna()]
    out["week"] = out.week.astype(int)
    is_qb = out.position == "QB"
    out["vol_pass"] = np.where(is_qb, out.attempts, out.targets)
    out["opp"] = out.vol_pass + out.carries
    out["tds"] = np.where(is_qb, out.passing_tds + out.rushing_tds,
                          out.receiving_tds + out.rushing_tds)
    # The team denominator counts a quarterback's attempts NOT at all: those are
    # the same plays as the targets they produced, and adding both would
    # double-count every throw and halve every receiver's share.
    team_col = "team" if "team" in out.columns else "recent_team"
    out["_team"] = out[team_col]
    skill = out[~is_qb]
    team = (skill.groupby(["season", "week", "_team"], as_index=False)
                 .agg(team_opp=("opp", "sum")))
    out = out.merge(team, on=["season", "week", "_team"], how="left")
    out["opp_share_week"] = out.opp / out.team_opp.replace(0, np.nan)
    return out


def _points(d: pd.DataFrame) -> pd.Series:
    """FanDuel points per player-week, from the same scoring the board uses."""
    from .scoring import fanduel_points

    def col(name):
        # `d.get(name)` returns None for a column that is not there, and
        # `pd.to_numeric(None)` is a scalar nan rather than a Series -- so the
        # convenient form breaks on any frame missing one of these, which real
        # nflverse frames are (the 2pt and fumble columns come and go).
        if name not in d.columns:
            return pd.Series(0.0, index=d.index)
        return pd.to_numeric(d[name], errors="coerce").fillna(0.0)

    return fanduel_points(
        passing_yards=col("passing_yards"), passing_tds=col("passing_tds"),
        interceptions=col("passing_interceptions"),
        rushing_yards=col("rushing_yards"), rushing_tds=col("rushing_tds"),
        receptions=col("receptions"), receiving_yards=col("receiving_yards"),
        receiving_tds=col("receiving_tds"),
        fumbles=(col("rushing_fumbles_lost") + col("receiving_fumbles_lost")
                 + col("sack_fumbles_lost")),
        two_point_conversions=(col("passing_2pt_conversions")
                               + col("rushing_2pt_conversions")
                               + col("receiving_2pt_conversions")),
    )


def opportunity_context(season: int, week: int, weekly: pd.DataFrame | None = None,
                        *, min_prior_games: int = MIN_PRIOR_GAMES) -> pd.DataFrame:
    """Per-player opportunity from weeks strictly BEFORE `week` of `season`.

    Leakage-safe by construction: the week being built for never contributes to
    its own context, which is what makes these columns readable next to a
    projection for that week rather than a description of it.
    """
    if weekly is None:
        weekly = load_weekly([season])
    if weekly is None or weekly.empty:
        return pd.DataFrame(columns=["player_id", *CONTEXT_COLUMNS])
    d = _volume(weekly)
    d = d[(d.season == int(season)) & (d.week < int(week))]
    if d.empty:
        return pd.DataFrame(columns=["player_id", *CONTEXT_COLUMNS])
    d["points"] = _points(d)

    agg = d.groupby("player_id").agg(
        games=("week", "nunique"), opp=("opp", "sum"), vol_pass=("vol_pass", "sum"),
        carries=("carries", "sum"), points=("points", "sum"),
        opp_share=("opp_share_week", "mean"), tds=("tds", "sum"),
        position=("position", "last"), team=("_team", "last"),
        player_name=("player_display_name", "last"))
    agg = agg[agg.games > 0]
    out = pd.DataFrame({
        "player_id": agg.index,
        "prior_games": agg.games.values,
        "opp_per_game": (agg.opp / agg.games).values,
        "opp_share": agg.opp_share.values,
        "tds_per_game": (agg.tds / agg.games).values,
        "fp_per_game": (agg.points / agg.games).values,
        "position": agg.position.values,
        "team": agg.team.values,
        "player_name": agg.player_name.values,
    })
    rates = out.position.map(POINTS_PER_OPPORTUNITY)
    pass_rate = rates.map(lambda r: r[0] if isinstance(r, tuple) else np.nan)
    carry_rate = rates.map(lambda r: r[1] if isinstance(r, tuple) else np.nan)
    out["xfp_per_game"] = ((agg.vol_pass.values / agg.games.values) * pass_rate
                           + (agg.carries.values / agg.games.values) * carry_rate)
    # The receiver who gets the ball and has not scored: positive means volume
    # says more than the scoreboard did.
    out["volume_td_gap"] = out.xfp_per_game - out.fp_per_game

    # Last week on its own, for the spike. Negative information: see the module
    # docstring — the board already chases this and the player disappoints.
    # Each player's OWN last appearance, not the slate's highest week number.
    # Those differ whenever the context is built mid-week: asked for week 7 on a
    # Thursday, `d.week.max()` is a week 6 that only one game has played, so the
    # jump would exist for twenty players and be blank for the rest of the
    # league. A player's last game is well defined for everyone.
    last = (d.sort_values("week").groupby("player_id").opp_share_week.last())
    out["share_jump"] = out.player_id.map(last) - out.opp_share
    # A rate needs games behind it; below the floor the columns are present but
    # empty rather than confidently wrong.
    thin = out.prior_games < int(min_prior_games)
    for col in ("opp_share", "xfp_per_game", "volume_td_gap", "share_jump"):
        out.loc[thin, col] = np.nan
    # A quarterback has no share to report. The denominator is deliberately the
    # skill positions' targets and carries, so dividing his attempts by it
    # returns about 0.7 for every starter in the league -- a number that looks
    # like a share, sorts like a share, and means nothing. Blanked rather than
    # shown, while his opportunities per game and his gap stay valid.
    is_qb = out.position == "QB"
    out.loc[is_qb, ("opp_share", "share_jump")] = np.nan
    return out.sort_values("opp_per_game", ascending=False).reset_index(drop=True)


def attach(frame: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Join the context onto a lineup pool. Scoring columns are not touched."""
    if context is None or context.empty or "player_id" not in frame.columns:
        out = frame.copy()
        for col in CONTEXT_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan
        return out
    cols = ["player_id", *CONTEXT_COLUMNS]
    return frame.merge(context[[c for c in cols if c in context.columns]],
                       on="player_id", how="left")
