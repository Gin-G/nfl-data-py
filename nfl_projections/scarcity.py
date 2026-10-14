"""Positional scarcity: what each position actually supplies, and what a start costs.

The board ranks players by projected points, which answers "who scores more?"
and not "who is harder to replace?" — and those are different questions. Eleven
quarterbacks finishing inside the top 50 overall does not make quarterback a
valuable position; it makes the twelfth quarterback cheap, because the
replacement for the one you lose is nearly as good. A tight end at the same
projection is scarce, because the next tight end is not.

Two things come out of here:

``replacement_level``  the points per game of the last startable player at each
                       position, from which value over replacement follows. This
                       is the number the board was missing.
``usable_rate``        how often a player clears a flat startable bar — the
                       usability idea, measured instead of asserted. A 14-point
                       average can be nine usable weeks or four, and those are
                       not the same player to own.

``top_n_composition`` is the diagnostic behind both: the positional make-up of
the top 25, 50 and 100 scorers in a season. It is what tells you whether a
position's good players are clustered or spread, which is what scarcity is.

Everything is computed from weekly player stats rather than from projections,
so these are facts about football and not about the model. Defaults are a
12-team league with 1 QB, 2 RB, 3 WR, 1 TE and a flex.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .scoring import fanduel_points
from .utils import to_pandas

logger = logging.getLogger(__name__)

# A startable week. 15 points is the bar the fantasy industry argues about and
# it holds up in both scorings: it is roughly the 12th-best QB, 24th-best RB and
# 12th-best TE week in a season, which is the definition of a starter.
USABLE_BAR = 15.0
# Where the positional starter pool ends in a 12-team league, counting the flex
# as receivers because that is where it usually goes.
STARTERS = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}
# Replacement is the shelf just past the starters, not the starter himself —
# averaged over a few ranks so one outlier season cannot set it.
REPLACEMENT_BAND = 6
TOP_N = (25, 50, 100)
POSITIONS = ("QB", "RB", "WR", "TE")
MIN_GAMES = 4           # below this a per-game average is noise


def weekly_points(seasons=None, *, scoring=None) -> pd.DataFrame:
    """One row per player-week in the regular season, with FanDuel points.

    Half-PPR with the yardage bonuses, i.e. the same scoring the board and the
    optimizer use — scarcity under full PPR is a different shape and mixing the
    two would make the replacement levels wrong for everything downstream.
    """
    import nflreadpy as nfl

    from .dataset import load_through_latest

    seasons = list(seasons or config.SEASONS)
    stats, loaded = load_through_latest(nfl.load_player_stats, seasons, "player stats")
    d = stats[stats.get("season_type", "REG") == "REG"].copy()
    d = d[d.position.isin(POSITIONS) & d.player_id.notna()]
    d["points"] = fanduel_points(
        passing_yards=d.get("passing_yards", 0), passing_tds=d.get("passing_tds", 0),
        interceptions=d.get("passing_interceptions", 0),
        rushing_yards=d.get("rushing_yards", 0), rushing_tds=d.get("rushing_tds", 0),
        receptions=d.get("receptions", 0), receiving_yards=d.get("receiving_yards", 0),
        receiving_tds=d.get("receiving_tds", 0),
        fumbles=(d.get("rushing_fumbles_lost", 0).fillna(0)
                 + d.get("receiving_fumbles_lost", 0).fillna(0)
                 + d.get("sack_fumbles_lost", 0).fillna(0)),
        two_point_conversions=(d.get("passing_2pt_conversions", 0).fillna(0)
                               + d.get("rushing_2pt_conversions", 0).fillna(0)
                               + d.get("receiving_2pt_conversions", 0).fillna(0)),
        scoring=scoring)
    name = d.player_display_name if "player_display_name" in d else d.player_name
    out = pd.DataFrame({"season": d.season, "week": d.week, "player_id": d.player_id,
                        "player_name": name, "position": d.position,
                        "team": d.get("team"), "points": d.points})
    logger.info("scarcity: %d player-weeks, %s-%s", len(out), loaded[0], loaded[-1])
    return out.reset_index(drop=True)


def per_season(weekly: pd.DataFrame, *, min_games=MIN_GAMES) -> pd.DataFrame:
    """Season totals and per-game averages per player, with positional ranks."""
    g = (weekly.groupby(["season", "player_id", "player_name", "position"])
         .agg(games=("points", "size"), total=("points", "sum"),
              ppg=("points", "mean"),
              usable=("points", lambda s: int((s >= USABLE_BAR).sum())))
         .reset_index())
    g = g[g.games >= min_games].copy()
    g["usable_rate"] = (g.usable / g.games).round(3)
    g["pos_rank_total"] = (g.groupby(["season", "position"]).total
                           .rank(ascending=False, method="min").astype(int))
    g["pos_rank_ppg"] = (g.groupby(["season", "position"]).ppg
                         .rank(ascending=False, method="min").astype(int))
    g["overall_rank_total"] = (g.groupby("season").total
                               .rank(ascending=False, method="min").astype(int))
    return g.sort_values(["season", "overall_rank_total"]).reset_index(drop=True)


def top_n_composition(seasons_frame: pd.DataFrame, ns=TOP_N) -> pd.DataFrame:
    """Positional make-up of the top N overall scorers, per season.

    The diagnostic behind scarcity: a position whose players crowd the top 25
    and vanish by 100 is top-heavy, and one spread evenly across all three is
    deep. Ranked on season totals, which is how the question is usually asked —
    availability is part of being valuable.
    """
    rows = []
    for season, g in seasons_frame.groupby("season"):
        for n in ns:
            top = g.nsmallest(n, "overall_rank_total")
            counts = top.position.value_counts()
            for pos in POSITIONS:
                rows.append({"season": season, "top_n": n, "position": pos,
                             "players": int(counts.get(pos, 0))})
    return pd.DataFrame(rows)


def replacement_level(seasons_frame: pd.DataFrame, *, starters=None,
                      band=REPLACEMENT_BAND) -> pd.DataFrame:
    """Starter and replacement points per game, per position per season.

    ``starter_ppg`` is the last startable player's average; ``replacement_ppg``
    is the mean of the next ``band`` players, which is what you would actually
    get off waivers. Value over replacement is the gap between a projection and
    that second number, and it is why a scarce position's points count more.
    """
    starters = starters or STARTERS
    rows = []
    for (season, pos), g in seasons_frame.groupby(["season", "position"]):
        depth = starters.get(pos)
        if not depth:
            continue
        ranked = g.sort_values("ppg", ascending=False).reset_index(drop=True)
        if len(ranked) <= depth:
            continue
        starter = float(ranked.ppg.iloc[depth - 1])
        shelf = ranked.ppg.iloc[depth:depth + band]
        rows.append({"season": season, "position": pos, "qualified": len(ranked),
                     "starter_ppg": round(starter, 2),
                     "replacement_ppg": round(float(shelf.mean()), 2),
                     "usable_players": int((ranked.ppg >= USABLE_BAR).sum())})
    return pd.DataFrame(rows).sort_values(["season", "position"]).reset_index(drop=True)


def usable_rate(weekly: pd.DataFrame, *, bar=USABLE_BAR) -> pd.DataFrame:
    """How often each position supplies a startable week, per season.

    ``player_weeks`` counts every appearance, so this is the chance that a
    randomly chosen starting-position player gives you a usable game — the
    denominator matters, and it is why quarterback looks so different from
    tight end.
    """
    d = weekly.assign(hit=(weekly.points >= bar).astype(int))
    g = (d.groupby(["season", "position"])
         .agg(player_weeks=("hit", "size"), usable_weeks=("hit", "sum"),
              mean_points=("points", "mean")).reset_index())
    g["usable_rate"] = (g.usable_weeks / g.player_weeks).round(3)
    g["mean_points"] = g.mean_points.round(2)
    return g.sort_values(["season", "position"]).reset_index(drop=True)


def baselines(weekly: pd.DataFrame = None, *, seasons=None, exclude_current=True) -> dict:
    """Everything above in one call, plus a pooled multi-season baseline.

    ``exclude_current`` drops the season in progress from the pooled numbers: a
    replacement level computed from four weeks of football is mostly noise, and
    a mid-season board should be valued against finished seasons.
    """
    weekly = weekly_points(seasons) if weekly is None else weekly
    frame = per_season(weekly)
    if exclude_current and frame.season.max() == config.CURRENT_SEASON:
        settled = frame[frame.season < config.CURRENT_SEASON]
        settled_weekly = weekly[weekly.season < config.CURRENT_SEASON]
    else:
        settled, settled_weekly = frame, weekly
    if settled.empty:              # first season of a fresh dataset
        settled, settled_weekly = frame, weekly

    repl = replacement_level(settled)
    pooled = (repl.groupby("position")
              .agg(starter_ppg=("starter_ppg", "mean"),
                   replacement_ppg=("replacement_ppg", "mean"),
                   usable_players=("usable_players", "mean"))
              .round(2).reset_index())
    return {
        "seasons": sorted(weekly.season.unique().tolist()),
        "baseline_seasons": sorted(settled.season.unique().tolist()),
        "players": frame,
        "composition": top_n_composition(frame),
        "replacement": repl,
        "replacement_pooled": pooled,
        "usable": usable_rate(settled_weekly),
        "bars": dict(zip(pooled.position, pooled.replacement_ppg)),
    }


def value_over_replacement(projections: pd.DataFrame, bars: dict, *,
                           column="fanduel_fantasy_points", out="vor") -> pd.DataFrame:
    """Add value over replacement to a projection frame.

    A copy, not in place: callers hold onto the raw projection for scoring and
    accuracy, and a column that silently changed its meaning between the two
    would be the kind of bug that takes a season to notice.
    """
    d = projections.copy()
    base = d.position.map(bars) if "position" in d.columns else np.nan
    d[out] = (d[column] - base).round(2)
    d[f"{out}_baseline"] = base
    return d


def usability(sims: np.ndarray, *, bar=USABLE_BAR, bust=None) -> dict:
    """P(startable) and P(bust) from a simulation matrix, shape (players, sims).

    The projection says what a player averages; this says how often he is worth
    starting, and those come apart exactly where it matters. Two backs at 13
    points can be eleven usable weeks or five, and the one with five is a trade
    candidate however the ranking looks.
    """
    sims = np.asarray(sims, dtype=float)
    if sims.ndim != 2:
        raise ValueError(f"sims must be (players, sims), got shape {sims.shape}")
    out = {"p_usable": (sims >= bar).mean(axis=1).round(3)}
    if bust is not None:
        out["p_bust"] = (sims < bust).mean(axis=1).round(3)
    return out
