import pandas as pd

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
