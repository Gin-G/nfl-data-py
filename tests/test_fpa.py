"""Unit tests for opponent-adjusted fantasy points allowed (synthetic)."""
import numpy as np
import pandas as pd
import pytest

from nfl_projections import fpa


def _weekly(rows):
    """rows: (player_id, position, team, opponent, week, points)."""
    return pd.DataFrame(
        [{"player_id": p, "player_name": p, "position": pos, "team": t,
          "opponent_team": o, "season": 2026, "week": w, "fd": pts}
         for p, pos, t, o, w, pts in rows])


def _career(player, position, opponent_by_week, points_by_week, team="OFF"):
    return [(player, position, team, opponent_by_week[i], w, points_by_week[i])
            for i, w in enumerate(range(1, len(points_by_week) + 1))]


class TestBand:
    def test_the_bands_run_tough_to_soft(self):
        assert fpa.band(20) == "very tough"
        assert fpa.band(40) == "tough"
        assert fpa.band(50) == "average"
        assert fpa.band(60) == "soft"
        assert fpa.band(80) == "very soft"

    def test_a_missing_grade_is_not_guessed(self):
        assert fpa.band(None) == "unknown"
        assert fpa.band(np.nan) == "unknown"


class TestDeltas:
    def test_a_week_never_feeds_its_own_baseline(self):
        w = _weekly(_career("a", "TE", ["X"] * 5, [10, 10, 10, 10, 30]))
        d = fpa.deltas(w)
        # weeks 1-3 have no baseline yet (needs 3 prior games)
        assert set(d.week) == {4, 5}
        assert d[d.week == 4].baseline.iloc[0] == 10.0
        assert d[d.week == 5].delta.iloc[0] == pytest.approx(20.0)

    def test_it_refuses_a_frame_with_no_opponent(self):
        w = _weekly(_career("a", "TE", ["X"] * 5, [10] * 5)).drop(columns=["opponent_team"])
        with pytest.raises(ValueError, match="opponent"):
            fpa.deltas(w)

    def test_it_names_what_is_missing(self):
        with pytest.raises(ValueError, match="position"):
            fpa.deltas(pd.DataFrame({"player_id": ["a"], "season": [2026],
                                     "week": [1], "opponent_team": ["X"]}))


class TestScoringFromComponents:
    """The other tests hand `_points` a ready-made `fd`; real frames do not."""

    @staticmethod
    def _stat_weeks(**extra):
        rows = []
        for week in range(1, 6):
            row = {"player_id": "a", "player_name": "a", "position": "WR",
                   "season": 2026, "week": week, "opponent_team": "X",
                   "receiving_yards": 60, "receptions": 6, "receiving_tds": 0}
            row.update(extra)
            rows.append(row)
        return pd.DataFrame(rows)

    def test_it_scores_fanduel_from_stat_columns(self):
        d = fpa.deltas(self._stat_weeks())
        # 60 yards at 0.1 plus 6 catches at 0.5: half-PPR, not the PPR the old
        # FPA table used, which would have made this 12.0.
        assert d["fd"].iloc[0] == pytest.approx(9.0)

    def test_a_frame_with_no_fumbles_column_does_not_crash(self):
        # `weekly.get("rushing_fumbles_lost", 0)` handed back a bare int, which
        # has no .fillna -- so any frame omitting the column raised
        # AttributeError instead of scoring. Several nflverse shapes omit it.
        frame = self._stat_weeks()
        assert "rushing_fumbles_lost" not in frame.columns
        assert len(fpa.deltas(frame)) == 2       # weeks 4 and 5 have baselines

    def test_a_lost_fumble_still_costs_two_points(self):
        d = fpa.deltas(self._stat_weeks(rushing_fumbles_lost=1))
        assert d["fd"].iloc[0] == pytest.approx(7.0)

    def test_interceptions_fall_back_to_the_older_column_name(self):
        rows = [{"player_id": "qb", "player_name": "qb", "position": "QB",
                 "season": 2026, "week": week, "opponent_team": "X",
                 "passing_yards": 250, "passing_tds": 2, "interceptions": 1}
                for week in range(1, 6)]
        d = fpa.deltas(pd.DataFrame(rows))
        assert d["fd"].iloc[0] == pytest.approx(250 * 0.04 + 2 * 4 - 1)


class TestRatings:
    @staticmethod
    def two_defences():
        """HARD faced a star and held him level; SOFT faced a scrub and did not.

        Raw FPA ranks HARD as the softer of the two -- it allowed 20 a week
        against SOFT's 10 -- and that is the ranking this module exists to fix.
        """
        rows = []
        # a star: 20 a week, meets his baseline against HARD
        rows += _career("star", "TE", ["Z", "Z", "Z", "HARD", "HARD"],
                        [20, 20, 20, 20, 20], team="A")
        # a scrub: 5 a week, then doubles it against SOFT
        rows += _career("scrub", "TE", ["Z", "Z", "Z", "SOFT", "SOFT"],
                        [5, 5, 5, 10, 10], team="B")
        return _weekly(rows)

    def test_raw_fpa_and_the_adjustment_disagree(self):
        r = fpa.ratings(self.two_defences(), 2026).set_index("defense")
        assert r.loc["HARD", "allowed"] > r.loc["SOFT", "allowed"]      # raw
        assert r.loc["HARD", "adjusted"] < r.loc["SOFT", "adjusted"]    # adjusted
        assert r.loc["SOFT", "grade"] > r.loc["HARD", "grade"]

    def test_expected_records_the_schedule_faced(self):
        r = fpa.ratings(self.two_defences(), 2026).set_index("defense")
        assert r.loc["HARD", "expected"] == pytest.approx(20.0)
        # The scrub's own baseline climbs after he doubles up in week 4, so the
        # level SOFT faced averages (5.0 + 6.25) / 2 rather than a flat 5.
        assert r.loc["SOFT", "expected"] == pytest.approx(5.625)

    def test_a_defence_that_held_a_star_level_grades_near_zero_adjusted(self):
        r = fpa.ratings(self.two_defences(), 2026).set_index("defense")
        assert abs(r.loc["HARD", "adjusted"]) < 0.01

    def test_only_weeks_before_the_cutoff_count(self):
        w = _weekly(_career("a", "TE", ["X"] * 6, [10, 10, 10, 10, 10, 40]))
        through5 = fpa.ratings(w, 2026, 6).set_index("defense")
        through6 = fpa.ratings(w, 2026, 7).set_index("defense")
        assert through5.loc["X", "adjusted"] == pytest.approx(0.0)
        assert through6.loc["X", "adjusted"] > 1.0

    def test_an_empty_window_returns_the_columns_anyway(self):
        w = _weekly(_career("a", "TE", ["X"] * 5, [10] * 5))
        out = fpa.ratings(w, 2026, 1)
        assert len(out) == 0
        for col in ("defense", "position", "allowed", "adjusted", "grade", "band"):
            assert col in out.columns

    def test_shrinkage_pulls_a_thin_sample_toward_average(self):
        # one player-week of evidence should not produce a big rating
        thin = _weekly(_career("a", "TE", ["X"] * 4, [10, 10, 10, 40]))
        r = fpa.ratings(thin, 2026).set_index("defense")
        assert r.loc["X", "adjusted"] < 30.0 * 1 / (1 + fpa.SHRINK) + 0.01

    def test_grades_are_computed_inside_a_position(self):
        # A tight end's points are not a receiver's, so each position is
        # centred on its own league average.
        rows = (_career("te", "TE", ["X"] * 5, [8, 8, 8, 8, 8], team="A")
                + _career("wr", "WR", ["X"] * 5, [20, 20, 20, 20, 20], team="B"))
        r = fpa.ratings(_weekly(rows), 2026)
        assert set(r.position) == {"TE", "WR"}
        # One team per group and both held level: each sits at its own league
        # average rather than being compared across positions.
        assert (r.grade == 50.0).all()
        assert r[r.position == "TE"].allowed.iloc[0] == pytest.approx(8.0)
        assert r[r.position == "WR"].allowed.iloc[0] == pytest.approx(20.0)


class TestOutliers:
    def test_a_single_distorting_week_is_flagged(self):
        # Four normal weeks and one where the offence did nothing -- the
        # Tampa Bay at Atlanta case, where no pass was completed.
        rows = _career("a", "WR", ["ATL"] * 8,
                       [15, 15, 15, 15, 15, 15, 15, 0])
        r = fpa.ratings(_weekly(rows), 2026).set_index("defense")
        assert r.loc["ATL", "worst_week"] == 8
        assert r.loc["ATL", "worst_week_swing"] > 0   # dropping it raises them

    def test_a_steady_defence_is_not_flagged(self):
        rows = _career("a", "WR", ["X"] * 8, [15] * 8)
        r = fpa.ratings(_weekly(rows), 2026).set_index("defense")
        assert not np.isfinite(pd.to_numeric(r.loc["X", "worst_week"], errors="coerce"))
        assert r.loc["X", "worst_week_swing"] == 0.0

    def test_winsorising_caps_an_extreme_week_without_dropping_it(self):
        # The median was the worst of four estimators for receivers, so the
        # extreme week has to keep its sign and lose only its magnitude.
        mild = _weekly(_career("a", "WR", ["X"] * 5, [10, 10, 10, 10, 18]))
        wild = _weekly(_career("a", "WR", ["X"] * 5, [10, 10, 10, 10, 90]))
        m = fpa.ratings(mild, 2026).set_index("defense").loc["X", "adjusted"]
        w = fpa.ratings(wild, 2026).set_index("defense").loc["X", "adjusted"]
        assert w > m > 0           # still worse, direction kept
        assert w < m * 4           # but nothing like 90 - 10 would imply

    def test_the_report_calls_out_the_outlier_week(self):
        rows = _career("a", "WR", ["ATL"] * 8, [15, 15, 15, 15, 15, 15, 15, 0])
        r = fpa.ratings(_weekly(rows), 2026)
        text = fpa.report(r, "WR")
        assert "ATL" in text
        assert "week 8" in text

    def test_the_report_survives_a_position_it_has_nothing_for(self):
        rows = _career("a", "WR", ["X"] * 5, [15] * 5)
        assert "no QB" in fpa.report(fpa.ratings(_weekly(rows), 2026), "QB")


class TestMatchupsForWeek:
    @staticmethod
    def schedule():
        return pd.DataFrame({"season": [2026], "week": [9],
                             "home_team": ["A"], "away_team": ["HARD"]})

    def test_each_team_gets_the_grade_of_the_defence_it_faces(self):
        w = TestRatings.two_defences()
        rated = fpa.ratings(w, 2026)
        out = fpa.matchups_for_week(rated, self.schedule(), 2026, 9)
        a_te = out[(out.team == "A") & (out.position == "TE")]
        assert len(a_te) == 1
        assert a_te.opponent.iloc[0] == "HARD"
        hard_grade = rated.set_index(["defense", "position"]).loc[("HARD", "TE"), "grade"]
        assert a_te.fpa_grade.iloc[0] == pytest.approx(hard_grade)

    def test_a_week_with_no_games_returns_the_columns(self):
        rated = fpa.ratings(TestRatings.two_defences(), 2026)
        out = fpa.matchups_for_week(rated, self.schedule(), 2026, 14)
        assert len(out) == 0
        assert "fpa_grade" in out.columns

    def test_no_ratings_returns_the_columns(self):
        out = fpa.matchups_for_week(pd.DataFrame(), self.schedule(), 2026, 9)
        assert len(out) == 0
        assert "fpa_band" in out.columns
