"""The slate matcher: the one place a FanDuel name has to find a board row.

Every case here is a real 2026 spelling that missed or could have collided.
"""
import pandas as pd

from nfl_projections import names
from nfl_projections.optimizer import merge_fanduel_salaries


def _board():
    return pd.DataFrame({
        "player_name": ["Tre Harris", "Josh Palmer", "Matthew Hibner", "DJ Moore",
                        "James Cook", "Justin Jefferson", "Justin Jefferson",
                        "Travis Etienne Jr.", "Trevor Etienne", "Bijan Robinson"],
        "team": ["LAC", "BUF", "BAL", "BUF", "BUF", "MIN", "CLE", "NO", "CAR", "ATL"],
        "position": ["WR", "WR", "TE", "WR", "RB", "WR", "WR", "RB", "RB", "RB"],
        "player_id": list("abcdefghij"),
        "fanduel_fantasy_points": [6.0, 5.0, 3.0, 9.0, 17.0, 15.0, 0.1, 6.0, 4.0, 20.0],
    })


def _slate(rows):
    return pd.DataFrame(rows, columns=["Nickname", "Team", "Position", "Salary"])


def test_name_key_drops_suffix_punctuation_and_middle_initials():
    assert names.name_key("Tre' Harris") == names.name_key("Tre Harris")
    assert names.name_key("D.J. Moore") == names.name_key("DJ Moore")
    assert names.name_key("James Cook III") == names.name_key("James Cook")
    assert names.name_key("Travis Etienne Jr.") == "travis etienne"


def test_spellings_fanduel_and_nflverse_disagree_on():
    slate = _slate([("Tre' Harris", "LAC", "WR", 5000), ("Joshua Palmer", "BUF", "WR", 5000),
                    ("Matt Hibner", "BAL", "TE", 4500), ("D.J. Moore", "BUF", "WR", 6700),
                    ("James Cook III", "BUF", "RB", 8400)])
    idx = names.match_slate(slate, _board())
    assert list(idx) == [0, 1, 2, 3, 4]


def test_a_shared_name_goes_by_team_and_never_guesses():
    slate = _slate([("Justin Jefferson", "MIN", "WR", 8100),
                    ("Justin Jefferson", "NYG", "WR", 4500)])
    idx = names.match_slate(slate, _board())
    assert idx.iloc[0] == 5
    assert idx.iloc[1] is None          # two on the board, neither on his team


def test_brothers_on_different_teams_stay_apart():
    slate = _slate([("Travis Etienne", "NO", "RB", 5000), ("Trevor Etienne", "CAR", "RB", 4500)])
    assert list(names.match_slate(slate, _board())) == [7, 8]


def test_a_last_name_alone_needs_one_candidate_and_agreeing_first_names():
    # "Brian Robinson" is not on this board; Bijan must not be taken for him.
    slate = _slate([("Brian Robinson", "ATL", "RB", 5500)])
    assert names.match_slate(slate, _board()).iloc[0] is None


def test_fanduel_team_codes():
    board = pd.DataFrame({"player_name": ["Travis Hunter"], "team": ["JAX"],
                          "position": ["WR"]})
    slate = _slate([("Travis Hunter", "JAC", "WR", 6000)])
    assert names.team_key("JAC") == "JAX" and names.team_key("LAR") == "LA"
    assert names.match_slate(slate, board).iloc[0] == 0


def test_the_optimizer_merge_uses_it():
    slate = _slate([("Joshua Palmer", "BUF", "WR", 5000),
                    ("Justin Jefferson", "MIN", "WR", 8000)])
    merged = merge_fanduel_salaries(slate, _board())
    assert merged.fanduel_fantasy_points.tolist() == [5.0, 15.0]
    assert merged.player_id.tolist() == ["b", "f"]
    assert merged.value.round(2).tolist() == [1.0, 1.88]
