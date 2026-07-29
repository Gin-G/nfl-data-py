"""Unit tests for player grades (synthetic weekly stats, no network)."""
import numpy as np
import pandas as pd

from nfl_projections import player_grades as pg


def _stats():
    """3 WRs over 6 weeks: a high-opportunity star, a mid, and a low-usage player."""
    rows = []
    profiles = {
        "star": dict(wopr=0.75, target_share=0.30, rec_yds=95, rec=8, tds=0.7),
        "mid":  dict(wopr=0.45, target_share=0.18, rec_yds=55, rec=5, tds=0.3),
        "low":  dict(wopr=0.20, target_share=0.08, rec_yds=25, rec=2, tds=0.1),
    }
    for pid, p in profiles.items():
        for wk in range(1, 7):
            rows.append(dict(
                player_id=pid, player_display_name=pid, position="WR", week=wk,
                wopr=p["wopr"], target_share=p["target_share"], carries=0, attempts=0,
                receiving_yards=p["rec_yds"], receptions=p["rec"], receiving_tds=p["tds"],
                receiving_epa=1.0, rushing_epa=0.0, passing_epa=0.0,
                passing_yards=0, passing_tds=0, passing_interceptions=0,
                rushing_yards=0, rushing_tds=0, targets=int(p["target_share"] * 35)))
    return pd.DataFrame(rows)


def test_grade_orders_by_opportunity_and_production():
    g = pg.grades(_stats()).set_index("player_id")
    assert g.loc["star", "grade"] > g.loc["mid", "grade"] > g.loc["low", "grade"]


def test_grades_in_0_100():
    g = pg.grades(_stats())
    assert g["grade"].between(0, 100).all()


def test_average_player_near_50():
    # with a symmetric spread the middle player should grade near the positional average
    g = pg.grades(_stats()).set_index("player_id")
    assert 35 <= g.loc["mid", "grade"] <= 65


def test_through_week_filter():
    g_all = pg.grades(_stats())
    g_early = pg.grades(_stats(), through_week=3)
    # both produce a full set of graded WRs
    assert set(g_all["player_id"]) == set(g_early["player_id"]) == {"star", "mid", "low"}
