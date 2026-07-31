"""Opponent-adjusted team power ratings / grades.

A 0-100 grade per team (50 = league average) for offense and defense, built from an
SRS-style opponent adjustment of points scored / allowed. Measured (EXPERIMENTS.md
"opponent-ADJUSTED TEAM grades"): offense is a stable, predictable trait (split-half
r ≈ 0.49); defense is noisier but opponent adjustment ~doubles its stability (0.14→0.26).

Two uses:
  * a preseason prior (last season's grade regressed toward average), and
  * week-to-week grades that blend that prior with in-season results, the prior fading
    as real games accumulate — so early weeks aren't overreactive.

Honest scope: a team's own offense is already reflected in its players' rolling form, and
the opponent-*defense* side of a matchup is a weak per-game lever (~1-4.5% of player FP).
These grades are for interpretability (UI), game context, and as a team-scoring anchor for
matchup-varying season projections — not a big single-game accuracy lever.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# fraction of a team's grade that carries over year-to-year (rest regresses to mean).
_YOY_CARRYOVER = 0.65
# equivalent games of weight the preseason prior gets when blending with in-season.
_PRIOR_GAMES = 6.0
_GRADE_SCALE = 15.0  # 1 SD of rating ≈ 15 grade points


def _game_frame(seasons: list[int], schedule=None) -> pd.DataFrame:
    """One row per team per completed game: points for / against + opponent."""
    if schedule is None:
        import nflreadpy as nfl
        schedule = pd.concat([nfl.load_schedules(seasons=[s]).to_pandas() for s in seasons],
                             ignore_index=True)
    sch = schedule.dropna(subset=["home_score", "away_score"])
    rows = []
    for _, g in sch.iterrows():
        rows.append((g["season"], g["week"], g["home_team"], g["away_team"],
                     g["home_score"], g["away_score"]))
        rows.append((g["season"], g["week"], g["away_team"], g["home_team"],
                     g["away_score"], g["home_score"]))
    return pd.DataFrame(rows, columns=["season", "week", "team", "opp", "pf", "pa"])


def opponent_adjusted(gf: pd.DataFrame, iters: int = 50):
    """SRS-style opponent-adjusted offense & defense ratings (points vs league average
    against an average opponent), centered at 0. Positive offense = scores above average;
    positive defense = allows above average (i.e. worse defense)."""
    teams = gf["team"].unique()
    lg = gf["pf"].mean()
    off = {t: 0.0 for t in teams}
    dff = {t: 0.0 for t in teams}
    for _ in range(iters):
        new_off = {t: np.mean([pf - lg - dff[o]
                               for pf, o in zip(gf.loc[gf.team == t, "pf"],
                                                gf.loc[gf.team == t, "opp"])]) for t in teams}
        new_dff = {t: np.mean([pa - lg - off[o]
                               for pa, o in zip(gf.loc[gf.team == t, "pa"],
                                                gf.loc[gf.team == t, "opp"])]) for t in teams}
        off, dff = new_off, new_dff
    return pd.Series(off, name="off"), pd.Series(dff, name="def"), float(lg)


def to_grade(rating: pd.Series, invert: bool = False) -> pd.Series:
    """Map a mean-0 rating to a 0-100 grade (50 = average). `invert=True` for defense so
    a stingier defense (allows fewer points, negative rating) grades HIGHER."""
    z = rating / (rating.std() or 1.0)
    if invert:
        z = -z
    return (50 + _GRADE_SCALE * z).clip(0, 100)


def league_avg_points(season: int, schedule=None) -> float:
    """Average points per team per game for a season (the matchup baseline)."""
    gf = _game_frame([season], schedule=schedule)
    return float(gf["pf"].mean()) if len(gf) else 22.5


def scoring_multiplier(off_team: float, def_opp: float, league_avg: float,
                       damp: float = 0.75, lo: float = 0.85, hi: float = 1.15) -> float:
    """Matchup scoring multiplier for a team (offense rating `off_team`) facing an
    opponent defense (`def_opp`), relative to that team vs an average defense.

    Ratings are in points (mean 0); def_opp > 0 means the opponent allows more than
    average, so the multiplier > 1. `damp` shrinks toward 1 for the estimation error /
    weak-signal reality of the defense side (EXPERIMENTS.md); clipped to [lo, hi] so no
    single matchup swings a projection more than ~15%."""
    neutral = league_avg + off_team
    if neutral <= 0:
        return 1.0
    raw = (league_avg + off_team + def_opp) / neutral
    return float(min(hi, max(lo, 1.0 + damp * (raw - 1.0))))


def game_environments(season: int, *, grades: pd.DataFrame | None = None,
                      league_avg: float | None = None, schedule=None,
                      damp: float = 0.5, lo: float = 0.85, hi: float = 1.20) -> pd.DataFrame:
    """Per-game scoring-environment multiplier from the expected game total.

    For each matchup, expected points for each side come from the SRS prediction
    (offense vs opponent defense); their sum is the expected game total. A game whose
    expected total exceeds the league-average game gets an environment multiplier > 1
    (shootout — boosts BOTH teams' players), below-average gets < 1. Measured: shootout
    games (51+ total) see ~4x the boom rate and ~60% higher ceilings, so this mainly
    lifts the DISTRIBUTION/ceiling; `damp` keeps the mean shift conservative (predicted
    totals correlate with actuals ~0.2). Returns rows (season, week, team, opponent,
    expected_team_points, expected_total, env_mult).
    """
    if grades is None:
        g = grades_for(season, schedule=schedule)
        grades = g if not g.empty else preseason_prior(season - 1, schedule=schedule)
    if league_avg is None:
        league_avg = league_avg_points(season - 1, schedule=schedule)
    off = grades["off_rating"].to_dict()
    dff = grades["def_rating"].to_dict()
    avg_total = 2.0 * league_avg
    if schedule is None:
        import nflreadpy as nfl
        schedule = nfl.load_schedules(seasons=[season]).to_pandas()
    sch = schedule[schedule["season"] == season] if "season" in schedule.columns else schedule
    rows = []
    for _, gm in sch.iterrows():
        h, a = gm.get("home_team"), gm.get("away_team")
        if pd.isna(h) or pd.isna(a):
            continue
        exp_h = league_avg + off.get(h, 0.0) + dff.get(a, 0.0)
        exp_a = league_avg + off.get(a, 0.0) + dff.get(h, 0.0)
        total = exp_h + exp_a
        env = float(min(hi, max(lo, 1.0 + damp * (total / avg_total - 1.0))))
        for team, opp, exp in [(h, a, exp_h), (a, h, exp_a)]:
            rows.append({"season": season, "week": int(gm["week"]), "team": team,
                         "opponent": opp, "expected_team_points": round(exp, 1),
                         "expected_total": round(total, 1), "env_mult": round(env, 3)})
    return pd.DataFrame(rows)


# alias so game_environments can fall back to in-season grades when available
def grades_for(season, **kw):
    try:
        return grades(season, **kw)
    except Exception:
        return pd.DataFrame()


def preseason_prior(prior_season: int, schedule=None) -> pd.DataFrame:
    """Preseason grades for the season after `prior_season`: last year's opponent-adjusted
    ratings regressed toward the mean (teams keep ~65% of their edge year to year)."""
    gf = _game_frame([prior_season], schedule=schedule)
    off, dff, _ = opponent_adjusted(gf)
    off, dff = off * _YOY_CARRYOVER, dff * _YOY_CARRYOVER
    return pd.DataFrame({
        "off_rating": off, "def_rating": dff,
        "off_grade": to_grade(off).round(1),
        "def_grade": to_grade(dff, invert=True).round(1),
    })


def grades(season: int, through_week: int | None = None, prior=None, schedule=None) -> pd.DataFrame:
    """Team grades for `season` as of `through_week` (all completed games if None),
    blending a preseason prior with in-season results — the prior worth ~`_PRIOR_GAMES`
    games and fading as real games accrue.

    `prior`: a preseason_prior() DataFrame (built from season-1 if omitted)."""
    if prior is None:
        prior = preseason_prior(season - 1, schedule=schedule)
    gf = _game_frame([season], schedule=schedule)
    if through_week is not None:
        gf = gf[gf["week"] <= through_week]

    if gf.empty:  # preseason / week 0 -> pure prior
        out = prior.copy()
        out["games"] = 0
        return out

    off_in, dff_in, _ = opponent_adjusted(gf)
    n = gf.groupby("team").size()
    rows = {}
    for t in set(off_in.index) | set(prior.index):
        g = float(n.get(t, 0))
        po = prior["off_rating"].get(t, 0.0)
        pd_ = prior["def_rating"].get(t, 0.0)
        io = off_in.get(t, 0.0)
        idf = dff_in.get(t, 0.0)
        w = _PRIOR_GAMES
        off_r = (w * po + g * io) / (w + g)
        def_r = (w * pd_ + g * idf) / (w + g)
        rows[t] = (off_r, def_r, int(g))
    df = pd.DataFrame(rows, index=["off_rating", "def_rating", "games"]).T
    df["off_grade"] = to_grade(df["off_rating"]).round(1)
    df["def_grade"] = to_grade(df["def_rating"], invert=True).round(1)
    return df.sort_values("off_grade", ascending=False)
