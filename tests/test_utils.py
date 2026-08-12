import pandas as pd
import pytest

from nfl_projections import utils
from nfl_projections.utils import normalize_player_name, parse_weeks, regular_games


def test_normalize_strips_suffixes():
    assert normalize_player_name("Odell Beckham Jr.") == "odell beckham"
    assert normalize_player_name("Robert Griffin III") == "robert griffin"
    assert normalize_player_name("Marvin Harrison Jr") == "marvin harrison"


def test_normalize_handles_empty():
    assert normalize_player_name(None) == ""
    assert normalize_player_name("") == ""
    assert normalize_player_name(float("nan")) == ""


def test_normalize_collapses_spaces():
    assert normalize_player_name("  A.J.   Brown ") == "a.j. brown"


def test_parse_weeks_single():
    assert parse_weeks("3") == [3]


def test_parse_weeks_range():
    assert parse_weeks("1-4") == [1, 2, 3, 4]


def test_parse_weeks_list_and_mixed():
    assert parse_weeks("1,5,3") == [1, 3, 5]
    assert parse_weeks("1-3,7") == [1, 2, 3, 7]


def test_regular_games_drops_avg_rows():
    df = pd.DataFrame({
        "week": [1, 2, "AVG", 3],
        "player_id": ["a", "a", "a", "a"],
    })
    out = regular_games(df)
    assert len(out) == 3
    assert out["week"].tolist() == [1, 2, 3]


class TestRegularGamesExcludesPostseason:
    """The postseason filter was missing for a long time, so the training set,
    the backtest population and board ground truth all quietly included weeks
    19-22 — a sample containing only playoff teams."""

    def _frame(self):
        return pd.DataFrame({
            "player_id": ["p1"] * 5,
            "week": [17, 18, 19, 21, "AVG"],
            "season_type": ["REG", "REG", "POST", "POST", "REG"],
            "fanduel_fantasy_points": [10.0, 12.0, 20.0, 25.0, 11.0],
        })

    def test_postseason_rows_are_dropped(self):
        out = utils.regular_games(self._frame())
        assert out["week"].tolist() == [17, 18]

    def test_avg_rows_still_dropped(self):
        assert "AVG" not in utils.regular_games(self._frame())["week"].tolist()

    def test_season_totals_no_longer_include_playoff_points(self):
        out = utils.regular_games(self._frame())
        assert out["fanduel_fantasy_points"].sum() == pytest.approx(22.0)

    def test_frames_without_season_type_are_untouched(self):
        # older / synthetic frames have no season_type; don't drop everything
        df = pd.DataFrame({"week": [1, 2, "AVG"], "x": [1, 2, 3]})
        assert utils.regular_games(df)["week"].tolist() == [1, 2]

    def test_null_season_type_is_kept(self):
        df = pd.DataFrame({"week": [1, 2], "season_type": [None, "REG"], "x": [1, 2]})
        assert len(utils.regular_games(df)) == 2
