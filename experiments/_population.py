"""Which player-weeks a residual test is read on.

    POPULATION=all   every board row (deduplicated) -- the old reading
    POPULATION=top   evaluate.top_slice: per week the top 24 QB / 36 RB /
                     50 WR / 18 TE by trailing-5 form, chosen before the game

The residual tests (blitz_qb, coverage_grades, opportunity_share) call
`restrict` on the board rows they test against, so one switch re-reads each of
them on the players who actually score (EXPERIMENTS.md, 2026-11-19).
"""
import os

import pandas as pd

POPULATION = os.environ.get("POPULATION", "all")
_HIST = None


def _history():
    global _HIST
    if _HIST is None:
        from nfl_projections import config
        _HIST = pd.read_csv(config.DATASET_PATH,
                            usecols=["player_id", "season", "week", "season_type",
                                     "fanduel_fantasy_points"], low_memory=False)
    return _HIST


def restrict(board, population=None):
    """Board rows deduplicated per player-week, cut to the population."""
    from nfl_projections import evaluate

    population = population or POPULATION
    b = board.drop_duplicates(["player_id", "season", "week"]).reset_index(drop=True)
    b = b[pd.to_numeric(b["week"], errors="coerce") <= 18].reset_index(drop=True)
    if population == "all":
        print(f"[population: all, {len(b):,} board player-weeks]")
        return b
    if population != "top":
        raise ValueError(f"POPULATION must be all or top, not {population!r}")
    b["form"] = evaluate.trailing_form(_history(), b)
    out = b[evaluate.top_slice(b, "form")].reset_index(drop=True)
    print(f"[population: top slice, {len(out):,} of {len(b):,} board player-weeks]")
    return out
