#!/usr/bin/env python3
"""Price-tier grouping of a board against a FanDuel slate."""

import pandas as pd
import pytest

from nfl_projections import dfs_targets as T
from nfl_projections.utils import normalize_player_name


def _slate(rows=None):
    rows = rows or [
        ("Alpha Back",  "RB", "DET", "MIA", "DET@MIA", 10200, 27.3, 7, ""),
        ("Bravo Back",  "RB", "CAR", "DEN", "DEN@CAR",  7400, 16.1, 7, ""),
        ("Cheap Back",  "RB", "ARI", "SEA", "ARI@SEA",  5800,  9.6, 7, ""),
        ("Hurt Back",   "RB", "SF",  "LV",  "LV@SF",    8700, 18.0, 7, "O"),
        ("IR Back",     "RB", "NE",  "GB",  "GB@NE",    5400,  8.0, 7, "IR"),
        ("Quest Back",  "RB", "NYG", "PHI", "NYG@PHI",  6000, 11.0, 7, "Q"),
        ("Alpha Arm",   "QB", "DET", "MIA", "DET@MIA",  7700, 21.7, 7, ""),
        ("Alpha Catch", "WR", "LAR", "WAS", "LAR@WAS",  9200, 18.4, 6, ""),
        ("Rookie Catch","WR", "PHI", "NYG", "NYG@PHI",  5000,  4.0, 2, ""),
        ("A Defense",   "D",  "NYG", "PHI", "NYG@PHI",  4800,  7.0, 8, ""),
    ]
    return pd.DataFrame([{
        "Nickname": n, "Position": p, "Team": t, "Opponent": o, "Game": g,
        "Salary": s, "FPPG": f, "Played": pl, "Injury Indicator": inj,
    } for n, p, t, o, g, s, f, pl, inj in rows])


def _board(rows=None):
    rows = rows or [
        ("Alpha Back",  "RB", "DET", 24.3, 8.2, 22.0, 44.3),
        ("Bravo Back",  "RB", "CAR", 16.6, 4.4, 15.0, 31.4),
        ("Cheap Back",  "RB", "ARI", 14.5, 2.7, 13.0, 27.0),
        ("Hurt Back",   "RB", "SF",  16.9, 4.0, 15.0, 30.0),
        ("IR Back",     "RB", "NE",   5.0, 1.0,  4.0, 11.0),
        ("Quest Back",  "RB", "NYG", 10.2, 2.0,  9.0, 20.0),
        ("Alpha Arm",   "QB", "DET", 18.5, 6.1, 17.0, 33.6),
        ("Alpha Catch", "WR", "LAR", 17.6, 4.5, 16.0, 33.3),
        ("Rookie Catch","WR", "PHI",  5.2, 1.0,  4.0, 12.0),
        ("A Defense",   "D",  "NYG",  7.5, 1.0,  7.0, 16.0),
    ]
    return pd.DataFrame([{
        "player_name": n, "position": p, "team": t, "projected_points": pts,
        "floor": lo, "median": md, "ceiling": hi,
    } for n, p, t, pts, lo, md, hi in rows])


def _build(**kw):
    return T.build(_board(), _slate(), name_key=normalize_player_name, **kw)


class TestTiers:
    def test_the_bands_are_price_not_quality(self):
        assert T.tier_of(9000) == "premium"
        assert T.tier_of(7500) == "premium"      # inclusive floor
        assert T.tier_of(7499) == "mid"
        assert T.tier_of(6000) == "mid"          # inclusive floor of mid
        assert T.tier_of(5999) == "value"
        assert T.tier_of(None) is None

    def test_the_boundaries_move_when_asked(self):
        assert T.tier_of(6500, premium_min=6000, value_max=5000) == "premium"
        f = _build(premium_min=6000, value_max=5000)
        assert set(f[f.slate_position == "RB"].tier) == {"premium", "mid"}

    def test_a_defense_is_found_under_either_spelling(self):
        assert T.canonical_positions(["DEF"]) == ("D",)
        assert T.canonical_positions(["D/ST", "dst", "qb"]) == ("D", "QB")
        f = T.build(_board(), _slate(), name_key=normalize_player_name,
                    positions=T.canonical_positions(["DEF"]))
        assert list(f.Nickname) == ["A Defense"]


class TestWhoIsOffered:
    def test_nobody_who_cannot_be_rostered(self):
        names = set(_build().Nickname)
        assert "Hurt Back" not in names       # FanDuel "O"
        assert "IR Back" not in names         # FanDuel "IR"

    def test_questionable_is_still_offered_but_said_out_loud(self):
        f = _build()
        row = f[f.Nickname == "Quest Back"].iloc[0]
        assert row.injury_flag == "Q"
        assert "questionable" in T.summarize(row)

    def test_a_slate_player_with_no_projection_is_dropped_not_zeroed(self):
        board = _board()
        board = board[board.player_name != "Cheap Back"]
        f = T.build(board, _slate(), name_key=normalize_player_name)
        assert "Cheap Back" not in set(f.Nickname)
        # ...and dropping him must not renumber anyone else's value.
        full = _build()
        for name in ("Alpha Back", "Bravo Back"):
            assert (f[f.Nickname == name].value.iloc[0]
                    == full[full.Nickname == name].value.iloc[0])

    def test_a_board_player_not_on_the_slate_is_not_a_target(self):
        board = pd.concat([_board(), pd.DataFrame([{
            "player_name": "Bye Week Guy", "position": "RB", "team": "PIT",
            "projected_points": 99.0, "floor": 1, "median": 50, "ceiling": 150,
        }])], ignore_index=True)
        f = T.build(board, _slate(), name_key=normalize_player_name)
        assert "Bye Week Guy" not in set(f.Nickname)


class TestOrdering:
    def test_value_is_points_per_thousand(self):
        f = _build()
        row = f[f.Nickname == "Alpha Back"].iloc[0]
        assert row.value == pytest.approx(24.3 / 10.2, abs=1e-3)

    def test_ranks_are_within_position_and_within_tier(self):
        f = _build()
        rbs = f[f.slate_position == "RB"].sort_values("value", ascending=False)
        assert list(rbs.position_rank) == sorted(rbs.position_rank)
        # Cheap Back is the only value RB left once O/IR are gone...
        value_rbs = rbs[rbs.tier == "value"]
        assert set(value_rbs.tier_rank) == set(range(1, len(value_rbs) + 1))

    def test_the_cheap_back_can_outrank_the_expensive_one(self):
        """The whole point of a value column."""
        f = _build()
        cheap = f[f.Nickname == "Cheap Back"].iloc[0]
        alpha = f[f.Nickname == "Alpha Back"].iloc[0]
        assert cheap.projected_points < alpha.projected_points
        assert cheap.value > alpha.value
        assert cheap.position_rank < alpha.position_rank

    def test_nothing_about_the_opponent_changes_the_order(self):
        """Context must not leak into the ranking.

        Built twice with opponents permuted: the ordering is identical,
        because no opponent column is read by `build` at all.
        """
        slate = _slate()
        shuffled = slate.copy()
        shuffled["Opponent"] = shuffled["Opponent"].iloc[::-1].values
        a = T.build(_board(), slate, name_key=normalize_player_name)
        b = T.build(_board(), shuffled, name_key=normalize_player_name)
        assert list(a.Nickname) == list(b.Nickname)
        assert list(a.value) == list(b.value)


class TestTheSummary:
    def test_it_states_the_board_the_price_and_the_rank(self):
        f = _build()
        text = T.summarize(f[f.Nickname == "Alpha Back"].iloc[0])
        assert "24.3" in text and "$10,200" in text
        assert "1st among premium RBs" in text
        assert "Floor 8.2 to ceiling 44.3" in text

    def test_a_thin_sample_is_called_thin_rather_than_averaged(self):
        f = _build()
        text = T.summarize(f[f.Nickname == "Rookie Catch"].iloc[0])
        assert "Only 2 games played" in text
        assert "thin" in text

    def test_context_is_labelled_as_context(self):
        f = _build()
        ctx = {"opponent_fpa": {"grade": 28.0, "rank": 30, "band": "very tough"}}
        text = T.summarize(f[f.Nickname == "Alpha Catch"].iloc[0], ctx)
        assert text.count("Context only:") == 1
        assert "very tough" in text and "30th softest" in text

    def test_no_context_means_no_claim_about_the_opponent(self):
        f = _build()
        text = T.summarize(f[f.Nickname == "Alpha Catch"].iloc[0])
        assert "Context" not in text
        assert "against" not in text

    def test_it_compares_against_the_player_s_own_season(self):
        f = _build()
        text = T.summarize(f[f.Nickname == "Cheap Back"].iloc[0])
        assert "above his own season average" in text

    def test_ordinals(self):
        assert T._ordinal(1) == "1st" and T._ordinal(11) == "11th"
        assert T._ordinal(22) == "22nd" and T._ordinal(13) == "13th"
        assert T._ordinal(None) == "unranked"


class TestNothingHereIsAnOverride:
    """Behavioural, not textual.

    Scanning the source for names or for the word "spread" flags the FanDuel
    column `Injury Indicator` and the floor-to-ceiling spread in the
    docstring, neither of which is what the rule is about. What the rule is
    about is whether output can depend on WHO a player is, or on a line. Both
    are testable directly.
    """

    def test_renaming_every_player_changes_nothing(self):
        """No name-specific behaviour can survive this."""
        board, slate = _board(), _slate()
        alias = {n: f"Player {i}" for i, n in enumerate(slate.Nickname)}
        board2, slate2 = board.copy(), slate.copy()
        board2["player_name"] = board2.player_name.map(alias).fillna(
            board2.player_name)
        slate2["Nickname"] = slate2.Nickname.map(alias)

        a = T.build(board, slate, name_key=normalize_player_name)
        b = T.build(board2, slate2, name_key=normalize_player_name)
        assert [alias[n] for n in a.Nickname] == list(b.Nickname)
        assert list(a.value) == list(b.value)
        assert list(a.tier) == list(b.tier)

    def test_adding_vegas_columns_changes_nothing(self):
        """A line on the slate must not be able to move a ranking."""
        slate = _slate()
        withline = slate.copy()
        withline["Implied Total"] = range(len(withline))
        withline["Spread"] = [-7.5] * len(withline)
        withline["Game Total"] = [52.5] * len(withline)

        a = T.build(_board(), slate, name_key=normalize_player_name)
        b = T.build(_board(), withline, name_key=normalize_player_name)
        assert list(a.Nickname) == list(b.Nickname)
        assert list(a.value) == list(b.value)

    def test_a_summary_quotes_no_line(self):
        f = _build()
        for _, row in f.iterrows():
            text = T.summarize(row).lower()
            for banned in ("implied", "spread of", "favorite", "underdog",
                           "moneyline", "total of"):
                assert banned not in text, (row.Nickname, banned)
