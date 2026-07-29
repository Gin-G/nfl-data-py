"""Unit tests for the season-projection assembly (synthetic data, no network)."""
import pandas as pd

from nfl_projections import season as S


def _base():
    return pd.DataFrame([
        {"player_id": "p1", "player_name": "WR One", "position": "WR", "team": "AAA",
         "fanduel_fantasy_points": 12.0, "floor": 6.0, "ceiling": 20.0},
    ])


def _grades():
    return pd.DataFrame({"off_rating": {"AAA": 4.0, "SOFT": 0.0, "STINGY": 0.0},
                         "def_rating": {"AAA": 0.0, "SOFT": 5.0, "STINGY": -5.0}})


def _sched():
    return pd.DataFrame([
        {"season": 2026, "week": 1, "home_team": "AAA", "away_team": "SOFT"},
        {"season": 2026, "week": 2, "home_team": "STINGY", "away_team": "AAA"},
    ])


def test_soft_defense_raises_stingy_lowers():
    wk = S.assemble_season(_base(), 2026, grades=_grades(), league_avg=22.5, schedule=_sched())
    soft = wk[wk.opponent == "SOFT"].iloc[0]
    stingy = wk[wk.opponent == "STINGY"].iloc[0]
    assert soft["projection"] > soft["base_projection"] > stingy["projection"]


def test_multiplier_clipped_to_band():
    # extreme opponent rating must not push the multiplier past the ±15% clip
    grades = pd.DataFrame({"off_rating": {"AAA": 0.0, "X": 0.0}, "def_rating": {"AAA": 0.0, "X": 99.0}})
    sched = pd.DataFrame([{"season": 2026, "week": 1, "home_team": "AAA", "away_team": "X"}])
    wk = S.assemble_season(_base(), 2026, grades=grades, league_avg=22.5, schedule=sched)
    assert wk.iloc[0]["matchup_multiplier"] <= 1.15


def test_band_scales_with_multiplier():
    wk = S.assemble_season(_base(), 2026, grades=_grades(), league_avg=22.5, schedule=_sched())
    soft = wk[wk.opponent == "SOFT"].iloc[0]
    assert soft["ceiling"] > 20.0 and soft["floor"] > 6.0  # both lifted vs a soft D


def test_season_totals_aggregate():
    wk = S.assemble_season(_base(), 2026, grades=_grades(), league_avg=22.5, schedule=_sched())
    tot = S.season_totals(wk)
    assert tot.iloc[0]["games"] == 2
    assert abs(tot.iloc[0]["proj_total"] - wk["projection"].sum()) < 1e-6
