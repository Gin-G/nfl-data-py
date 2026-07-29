"""Unit tests for team power ratings / grades (synthetic schedule, no network)."""
import numpy as np
import pandas as pd

from nfl_projections import ratings


def _schedule():
    """4 teams, a full round-robin (home & away) where STRONG >> mid >> WEAK on offense."""
    scores = {  # (home, away): (home_pts, away_pts)
        ("STR", "WEAK"): (35, 10), ("WEAK", "STR"): (13, 31),
        ("STR", "MID"): (28, 20), ("MID", "STR"): (17, 27),
        ("MID", "WEAK"): (24, 14), ("WEAK", "MID"): (16, 23),
    }
    rows = []
    wk = 1
    for (h, a), (hs, as_) in scores.items():
        rows.append(dict(season=2020, week=wk, home_team=h, away_team=a,
                         home_score=hs, away_score=as_))
        wk += 1
    return pd.DataFrame(rows)


def test_grades_rank_offense_correctly():
    g = ratings.grades(2020, prior=_empty_prior(), schedule=_schedule())
    assert g.loc["STR", "off_grade"] > g.loc["MID", "off_grade"] > g.loc["WEAK", "off_grade"]


def test_average_maps_to_50():
    s = pd.Series({"A": 0.0, "B": 3.0, "C": -3.0})
    grade = ratings.to_grade(s)
    assert abs(grade["A"] - 50.0) < 1e-6  # mean rating -> grade 50


def test_defense_invert_rewards_stinginess():
    # negative def_rating = allows fewer points = better defense = higher grade
    s = pd.Series({"stingy": -4.0, "avg": 0.0, "leaky": 4.0})
    g = ratings.to_grade(s, invert=True)
    assert g["stingy"] > g["avg"] > g["leaky"]


def test_grades_in_0_100_range():
    g = ratings.grades(2020, prior=_empty_prior(), schedule=_schedule())
    assert g["off_grade"].between(0, 100).all()
    assert g["def_grade"].between(0, 100).all()


def _empty_prior():
    idx = ["STR", "MID", "WEAK"]
    return pd.DataFrame({"off_rating": 0.0, "def_rating": 0.0,
                         "off_grade": 50.0, "def_grade": 50.0}, index=idx)
