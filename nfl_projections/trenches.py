"""Line-of-scrimmage ratings: 0-100 per unit, and a matchup number per game.

Four units get a grade, 50 = league average, 15 points to a standard deviation
(the same scale as `ratings.py` so the numbers read the same way):

    run_block     the offence's rushing front
    run_stop      the defence's rushing front
    pass_block    the offence's protection
    pass_rush     the defence's rush

A matchup rating puts one against the other, so an elite run-blocking front
against a soft front reads in the eighties and the reverse reads in the
twenties. 50 is an even matchup, not a mediocre one — a number in the eighties
means the edge is large, not that the offence is good.

**What these are worth, measured.** EXPERIMENTS.md, "The trench report: line
play, rebuilt from public data": pass protection is the most stable offensive
trait in the whole ledger (split-half r 0.557) and a defensive pass rush
measured by sacks is not a trait at all (0.068, which is why `pass_rush` is
built from pressures and not sacks). The rushing matchup, used as a weekly
multiplier, passes its bar in four seasons of four and is worth **0.124 yards
of MAE on a 24.7 baseline** — real, small, and nothing like the swing a number
in the eighties suggests. `backtest()` is in this module so that claim travels
with the rating rather than living in a file nobody opens, and the dashboard
shows what each band actually delivered.

Metric choices follow the same work: adjusted line yards uses FO's banding
because the first four yards of a carry belong to the line and the eleventh
does not, and pressure is sack-or-hit per dropback because nflverse publishes
no pressure field. Gap-level ratings are deliberately absent — cut by run gap
the signal collapses to r = 0.09, about twenty carries a side.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .pbp import load_pbp
from .ratings import to_grade

logger = logging.getLogger(__name__)

UNITS = ("run_block", "run_stop", "pass_block", "pass_rush")
# Offence/defence pairs that face each other, and which way the edge runs.
MATCHUPS = {"run": ("run_block", "run_stop"), "pass": ("pass_block", "pass_rush")}
# Plays a unit needs before its own number is trusted over the league mean.
# ~150 carries is half a season of rushing; the shrinkage is what keeps week 2
# from printing a 90.
SHRINK_RUSH, SHRINK_PASS = 150.0, 200.0
# Carry-over of a unit's grade into the next season, before any new football.
# Measured year-over-year at 0.25-0.42 for the offence and 0.11-0.28 for the
# defence, so they do not regress at the same rate.
YOY_CARRYOVER = {"run_block": 0.35, "pass_block": 0.40,
                 "run_stop": 0.20, "pass_rush": 0.28}
SRS_ITERS = 25


def band(yards: pd.Series) -> pd.Series:
    """Football-Outsiders adjusted-line-yards weighting of rushing yards.

    Losses count 1.2x (a stuff is the line's fault), the first four yards fully,
    5-10 at half, and nothing past ten — the bands are the whole idea: they keep
    a 60-yard run from making a bad line look good.
    """
    y = yards.astype(float)
    return np.where(y < 0, 1.2 * y,
                    np.where(y <= 4, y,
                             np.where(y <= 10, 4 + 0.5 * (y - 4), 7.0)))


def plays(seasons=None, pbp=None) -> pd.DataFrame:
    """Rushing and dropback plays with the per-play trench metrics attached.

    QB runs are excluded from the rushing side: a scramble is a protection
    failure, not a run-blocking rep, and counting it would score the same play
    twice in opposite directions.
    """
    seasons = list(seasons or config.SEASONS)
    if pbp is None:
        pbp = load_pbp(seasons)
    d = pbp[(pbp.get("season_type", "REG") == "REG")
            & pbp.posteam.notna() & pbp.defteam.notna()].copy()

    def flag(col):
        return d[col].fillna(0) == 1 if col in d.columns else pd.Series(False, index=d.index)

    is_rush = (flag("rush_attempt") & ~flag("qb_scramble") & ~flag("qb_kneel")
               & ~flag("two_point_attempt") & d.yards_gained.notna())
    is_drop = flag("qb_dropback") & ~flag("two_point_attempt")

    rush = d[is_rush].copy()
    rush["aly"] = band(rush.yards_gained)
    rush["stuffed"] = (rush.yards_gained <= 0).astype(float)
    drop = d[is_drop].copy()
    drop["pressured"] = (flag("sack")[is_drop] | flag("qb_hit")[is_drop]).astype(float)
    cols = ["season", "week", "posteam", "defteam"]
    return (rush[cols + ["aly", "stuffed", "yards_gained"]].assign(kind="rush"),
            drop[cols + ["pressured"]].assign(kind="pass"))


def _srs(frame: pd.DataFrame, metric: str, iters: int = SRS_ITERS):
    """Opponent-adjusted offence and defence effects on a per-play metric.

    Alternating means, the same construction as the team power ratings: a
    front's number is what it did minus what its opponents did to everyone
    else. Opponent adjustment roughly doubled defensive stability in the
    ratings work, and a line that drew three elite fronts needs it more.
    Returns (offence effect, defence effect, league mean) centred at zero.

    Written against integer team codes and bincount rather than groupby-and-map.
    The readable version is the same arithmetic and took 3.8 seconds per fit,
    which the weekly backtest pays 156 times over.
    """
    y = frame[metric].to_numpy(dtype=float)
    lg = float(y.mean())
    y = y - lg
    off_teams, off_codes = np.unique(frame.posteam.to_numpy(), return_inverse=True)
    def_teams, def_codes = np.unique(frame.defteam.to_numpy(), return_inverse=True)
    off_n = np.bincount(off_codes, minlength=len(off_teams)).astype(float)
    def_n = np.bincount(def_codes, minlength=len(def_teams)).astype(float)
    off = np.zeros(len(off_teams))
    dfn = np.zeros(len(def_teams))
    for _ in range(iters):
        off = np.bincount(off_codes, weights=y - dfn[def_codes],
                          minlength=len(off_teams)) / off_n
        dfn = np.bincount(def_codes, weights=y - off[off_codes],
                          minlength=len(def_teams)) / def_n
    return pd.Series(off, index=off_teams), pd.Series(dfn, index=def_teams), lg


def _shrunk(effect: pd.Series, counts: pd.Series, k: float) -> pd.Series:
    """Pull a unit's effect toward zero by how little football it has played."""
    n = counts.reindex(effect.index).fillna(0.0)
    return effect * (n / (n + k))


def unit_ratings(rush: pd.DataFrame, drop: pd.DataFrame) -> pd.DataFrame:
    """A 0-100 grade per team per unit from one window of plays.

    Signs are set so that higher is always better for the unit being graded:
    a run-blocking front wants adjusted line yards up, a run-stopping front
    wants them down, protection wants pressure down, a rush wants it up.
    """
    rows = {}
    aly_off, aly_def, _ = _srs(rush, "aly")
    stuff_off, stuff_def, _ = _srs(rush, "stuffed")
    prs_off, prs_def, _ = _srs(drop, "pressured")
    n_rush_off = rush.groupby("posteam").size()
    n_rush_def = rush.groupby("defteam").size()
    n_pass_off = drop.groupby("posteam").size()
    n_pass_def = drop.groupby("defteam").size()

    # Line yards and stuffs say the same thing from two directions; averaging the
    # two z-scores is steadier than either, and stuff rate is the half that
    # survives opponent adjustment better on defence.
    def z_of(effect, counts, k):
        e = _shrunk(effect, counts, k)
        return (e - e.mean()) / (e.std() or 1.0)

    def blend(a, b, counts, k, invert=False):
        # Centred as well as scaled: shrinking by play count moves the mean, and
        # a 50 that is not the league average is a rating nobody can read.
        z = (z_of(a, counts, k) - z_of(b, counts, k)) / 2.0
        return to_grade(z - z.mean(), invert=invert)

    rows["run_block"] = blend(aly_off, stuff_off, n_rush_off, SHRINK_RUSH)
    rows["run_stop"] = blend(aly_def, stuff_def, n_rush_def, SHRINK_RUSH, invert=True)
    rows["pass_block"] = to_grade(z_of(prs_off, n_pass_off, SHRINK_PASS), invert=True)
    rows["pass_rush"] = to_grade(z_of(prs_def, n_pass_def, SHRINK_PASS))
    out = pd.DataFrame(rows)
    out.index.name = "team"
    return out.round(1)


def ratings_through(rush: pd.DataFrame, drop: pd.DataFrame, season: int,
                    week: int | None = None, *, prior: pd.DataFrame | None = None,
                    prior_games: float = 5.0) -> pd.DataFrame:
    """Grades for `season` using only plays from weeks strictly before `week`.

    Leakage-safe by construction: the week being rated never contributes to the
    rating. Early in a season there is not enough football for a grade, so last
    season's regressed grade carries the load and fades out as games accrue —
    the same blend the team power ratings use, with per-unit carry-over because
    an offensive line holds its form twice as well as a defensive front.
    """
    cut = (lambda d: d[(d.season == season) & ((week is None) | (d.week < (week or 99)))])
    r, p = cut(rush), cut(drop)
    games = float(r.week.nunique()) if len(r) else 0.0
    if len(r) < 200 or len(p) < 200:
        if prior is None:
            return pd.DataFrame(columns=list(UNITS))
        return regressed(prior)
    live = unit_ratings(r, p)
    if prior is None or games >= 17:
        return live.assign(basis="in-season", games=games)
    base = regressed(prior).reindex(live.index)
    w = games / (games + prior_games)
    blended = (live[list(UNITS)] * w).add(base[list(UNITS)].fillna(50.0) * (1 - w))
    return blended.round(1).assign(basis=f"blend w={w:.2f}", games=games)


def regressed(prior: pd.DataFrame) -> pd.DataFrame:
    """Last season's grades pulled toward 50 by each unit's measured carry-over."""
    out = prior.copy()
    for unit in UNITS:
        if unit in out.columns:
            out[unit] = (50 + (out[unit] - 50) * YOY_CARRYOVER[unit]).round(1)
    keep = [c for c in out.columns if c in UNITS]
    return out[keep].assign(basis="prior-regressed", games=0.0)


def matchup(grades: pd.DataFrame, offense: str, defense: str,
            phase: str = "run") -> float:
    """One 0-100 number for a unit against the unit it faces.

    The difference of two grades has sqrt(2) times the spread of either, so it
    is rescaled back: 50 stays an even matchup and the tails stay as rare as
    they should be. 87 means the offence's edge is about two and a half standard
    deviations, which happens a handful of times a season — not every week.
    """
    off_unit, def_unit = MATCHUPS[phase]
    try:
        off = float(grades.loc[offense, off_unit])
        dfn = float(grades.loc[defense, def_unit])
    except KeyError:
        return float("nan")
    edge = (off - 50) - (dfn - 50)
    return round(float(np.clip(50 + edge / np.sqrt(2.0), 0, 100)), 1)


def matchups_for_week(grades: pd.DataFrame, schedule: pd.DataFrame, season: int,
                      week: int) -> pd.DataFrame:
    """Both sides of every game in a week, run and pass, as rating rows."""
    games = schedule[(schedule.season == season) & (schedule.week == week)]
    rows = []
    for _, g in games.iterrows():
        for off, dfn, home in ((g.home_team, g.away_team, True),
                               (g.away_team, g.home_team, False)):
            row = {"season": season, "week": week, "team": off, "opponent": dfn,
                   "home": home}
            for phase in MATCHUPS:
                row[f"{phase}_matchup"] = matchup(grades, off, dfn, phase)
            for unit in UNITS:
                side = off if unit.endswith("block") else dfn
                row[unit] = (float(grades.loc[side, unit])
                             if side in grades.index else float("nan"))
            rows.append(row)
    return pd.DataFrame(rows)


BANDS = [(0, 30, "fade"), (30, 42, "lean down"), (42, 58, "neutral"),
         (58, 70, "lean up"), (70, 101, "smash")]
# The opponent front graded on its own. Measured, this is the half of a matchup
# that carries information: a team's own line is already inside its players'
# recent form, so it cannot tell you anything new about this week.
OPPONENT_BANDS = [(0, 35, "very soft"), (35, 45, "soft"), (45, 55, "average"),
                  (55, 65, "stout"), (65, 101, "very stout")]


def _band(rating: float, bands=None) -> str:
    for lo, hi, label in (bands or BANDS):
        if lo <= rating < hi:
            return label
    return "n/a"


def backtest(rush: pd.DataFrame, drop: pd.DataFrame, *, seasons=None,
             first_week: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """What each rating band actually delivered, on games already played.

    For every team-game from `first_week` on, the rating is built only from
    earlier weeks of that season and then compared with what happened. The
    outcome is each team's yards per carry in the game *minus its own average
    for the season*, which is the only way to read a matchup: a good offence
    rushing well says nothing, a good offence rushing better than usual against
    a soft front is the claim being tested.

    Returns (per-game rows, band summary). The summary is what belongs on a
    dashboard next to the number — a rating of 87 is only worth something if
    the eighties historically produced more than the forties did.
    """
    seasons = sorted(seasons or rush.season.unique())
    team_ypc = (rush[rush.season.isin(seasons)].groupby(["season", "posteam"])
                .yards_gained.mean().rename("season_ypc"))
    game = (rush[rush.season.isin(seasons)]
            .groupby(["season", "week", "posteam", "defteam"])
            .agg(carries=("yards_gained", "size"), rush_yards=("yards_gained", "sum"),
                 game_ypc=("yards_gained", "mean"), game_aly=("aly", "mean"),
                 game_stuff=("stuffed", "mean")).reset_index())
    game = game[game.carries >= 10]

    rows = []
    for season in seasons:
        for week in sorted(game.loc[game.season == season, "week"].unique()):
            if week < first_week:
                continue
            grades = ratings_through(rush, drop, season, week)
            if grades.empty:
                continue
            now = game[(game.season == season) & (game.week == week)]
            for _, g in now.iterrows():
                rating = matchup(grades, g.posteam, g.defteam, "run")
                if not np.isfinite(rating):
                    continue
                rows.append({"season": season, "week": week, "team": g.posteam,
                             "opponent": g.defteam, "run_matchup": rating,
                             "band": _band(rating), "carries": g.carries,
                             "rush_yards": g.rush_yards, "game_ypc": g.game_ypc,
                             "game_aly": g.game_aly, "game_stuff": g.game_stuff,
                             "own_run_block": float(grades.loc[g.posteam, "run_block"]),
                             "opp_run_stop": float(grades.loc[g.defteam, "run_stop"])})
    per_game = pd.DataFrame(rows)
    if per_game.empty:
        return per_game, pd.DataFrame()
    per_game = per_game.merge(
        team_ypc.reset_index().rename(columns={"posteam": "team"}),
        on=["season", "team"], how="left")
    per_game["ypc_vs_own"] = (per_game.game_ypc - per_game.season_ypc).round(3)

    per_game["opp_band"] = [_band(v, OPPONENT_BANDS) for v in per_game.opp_run_stop]
    return per_game, bands(per_game)


def bands(per_game: pd.DataFrame, *, opponent_only: bool = False) -> pd.DataFrame:
    """Outcome averages per rating band, in the order the bands run.

    `opponent_only` groups by the opponent's front grade instead of the combined
    matchup. That table is the one to trust: the combined rating is half a
    statement about the offence, and the offence's own line is already priced
    into its players' form.
    """
    key, rating = ("opp_band", "opp_run_stop") if opponent_only else ("band", "run_matchup")
    order = [label for _, _, label in (OPPONENT_BANDS if opponent_only else BANDS)]
    summary = (per_game.groupby(key)
               .agg(team_games=(rating, "size"),
                    mean_rating=(rating, "mean"),
                    ypc=("game_ypc", "mean"),
                    ypc_vs_own=("ypc_vs_own", "mean"),
                    rush_yards=("rush_yards", "mean"),
                    stuff_rate=("game_stuff", "mean")).round(3).reset_index()
               .rename(columns={key: "band"}))
    summary["band"] = pd.Categorical(summary.band, categories=order, ordered=True)
    return summary.sort_values("band").reset_index(drop=True)


def correlations(per_game: pd.DataFrame) -> dict:
    """How tightly the rating tracked each outcome, pooled and per season.

    ``own_vs_opp`` is the part that matters for reading the number. A matchup
    rating is half a statement about the offence and half about the opponent,
    and those two halves are not worth the same: the offence's own line is
    already inside its players' form, so only the opponent half is new
    information. If the opponent half correlates with nothing, the rating is a
    team-quality ranking wearing a matchup costume.
    """
    out = {}
    for col in ("ypc_vs_own", "game_ypc", "rush_yards", "game_aly"):
        if col not in per_game:
            continue
        d = per_game[["run_matchup", col]].dropna()
        out[col] = round(float(np.corrcoef(d.run_matchup, d[col])[0, 1]), 3) if len(d) > 10 else None
    for col in ("own_run_block", "opp_run_stop"):
        if col in per_game:
            d = per_game[[col, "game_ypc", "ypc_vs_own"]].dropna()
            out.setdefault("own_vs_opp", {})[col] = {
                "game_ypc": round(float(np.corrcoef(d[col], d.game_ypc)[0, 1]), 3),
                "ypc_vs_own": round(float(np.corrcoef(d[col], d.ypc_vs_own)[0, 1]), 3)}
    out["per_season_ypc_vs_own"] = {
        int(s): round(float(np.corrcoef(g.run_matchup, g.ypc_vs_own)[0, 1]), 3)
        for s, g in per_game.dropna(subset=["ypc_vs_own"]).groupby("season")
        if len(g) > 10}
    return out
