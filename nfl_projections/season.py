"""Full-season projection assembly.

The weekly model gives a matchup-neutral "expected points given current form" number
(the same every week, since its only inputs are the player's own rolling stats). This
module turns that into a full-season projection that VARIES BY OPPONENT: each of a team's
scheduled games gets a matchup multiplier from the team power ratings (own offense grade
vs the opponent's defense grade), so a player projects for more against a soft defense and
less against a stingy one.

Honest scope (EXPERIMENTS.md): the opponent-defense side of a matchup is a weak, dampened
per-game lever — the base rolling-form projection is the backbone; the matchup nudge is
small and clipped. Rookies come in via the draft-capital prior (predict.py). This is a
preseason / forward-looking season view; regenerate as real games update both the base
form and the grades.
"""
from __future__ import annotations

import pandas as pd

from . import ratings as ratings_mod

_TEAM_COLS = ("team", "recent_team", "team_abbr", "club_code")


def _team_col(df: pd.DataFrame) -> str:
    for c in _TEAM_COLS:
        if c in df.columns:
            return c
    raise KeyError(f"no team column in projections (looked for {_TEAM_COLS})")


def _schedule_opponents(season: int, schedule=None) -> pd.DataFrame:
    """Long form: one row per team per scheduled game -> (season, week, team, opp, home)."""
    if schedule is None:
        import nflreadpy as nfl
        schedule = nfl.load_schedules(seasons=[season]).to_pandas()
    sch = schedule[schedule["season"] == season] if "season" in schedule.columns else schedule
    rows = []
    for _, g in sch.iterrows():
        h, a = g.get("home_team"), g.get("away_team")
        if pd.isna(h) or pd.isna(a):
            continue
        rows.append((season, g["week"], h, a, True))
        rows.append((season, g["week"], a, h, False))
    return pd.DataFrame(rows, columns=["season", "week", "team", "opp", "home"])


def assemble_season(base_projections: pd.DataFrame, season: int, *, grades=None,
                    league_avg=None, schedule=None, damp: float = 0.75) -> pd.DataFrame:
    """Expand matchup-neutral base projections into a per-game season projection.

    Args:
        base_projections: one row per player with a team column and
            ``fanduel_fantasy_points`` (and optionally floor/ceiling).
        grades: a ratings.grades() frame (off_rating/def_rating by team); built from the
            prior season's preseason prior if omitted.
        league_avg: league avg points/team/game; derived from the prior season if omitted.

    Returns a long DataFrame: one row per player per scheduled game with base_projection,
    matchup_multiplier, opponent, and the adjusted projection (+ floor/ceiling if present).
    """
    if grades is None:
        grades = ratings_mod.grades(season, schedule=schedule)
    if league_avg is None:
        league_avg = ratings_mod.league_avg_points(season - 1, schedule=schedule)

    off = grades["off_rating"].to_dict()
    dff = grades["def_rating"].to_dict()
    tcol = _team_col(base_projections)
    sched = _schedule_opponents(season, schedule=schedule)

    have_band = {"floor", "ceiling"} <= set(base_projections.columns)
    out = []
    for _, p in base_projections.iterrows():
        team = p[tcol]
        games = sched[sched["team"] == team]
        base = float(p["fanduel_fantasy_points"])
        for _, g in games.iterrows():
            mult = ratings_mod.scoring_multiplier(
                off.get(team, 0.0), dff.get(g["opp"], 0.0), league_avg, damp=damp)
            row = {
                "player_id": p.get("player_id"),
                "player_name": p.get("player_name") or p.get("player_display_name"),
                "position": p.get("position"),
                "team": team,
                "week": int(g["week"]),
                "opponent": g["opp"],
                "home": bool(g["home"]),
                "base_projection": round(base, 2),
                "matchup_multiplier": round(mult, 3),
                "projection": round(base * mult, 2),
            }
            if have_band:
                row["floor"] = round(float(p["floor"]) * mult, 2)
                row["ceiling"] = round(float(p["ceiling"]) * mult, 2)
            out.append(row)
    return pd.DataFrame(out)


def season_totals(weekly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a per-game season projection to per-player totals + games played."""
    agg = {"projection": "sum", "base_projection": "mean"}
    if "floor" in weekly.columns:
        agg["floor"] = "sum"
        agg["ceiling"] = "sum"
    g = (weekly.groupby(["player_id", "player_name", "position", "team"], dropna=False)
         .agg(**{"proj_total": ("projection", "sum"),
                 "proj_per_game": ("projection", "mean"),
                 "games": ("week", "count")})
         .reset_index()
         .sort_values("proj_total", ascending=False))
    return g


def project_season(service, season: int, *, base_week: int = 1, grades=None,
                   league_avg=None, schedule=None, damp: float = 0.75,
                   positions=None, use_injuries: bool = False) -> pd.DataFrame:
    """Convenience: build matchup-neutral base projections from a ProjectionService
    (projecting `base_week` for current form) and assemble the season. Returns the
    per-game long DataFrame; call season_totals() for per-player totals."""
    base = service.project(season, base_week, positions=positions,
                           use_injuries=use_injuries, as_frame=True, schedule=schedule)
    return assemble_season(base, season, grades=grades, league_avg=league_avg,
                           schedule=schedule, damp=damp)
