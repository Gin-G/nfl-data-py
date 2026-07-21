import pandas as pd

from nfl_projections import dataset


class TestProcessSnapCounts:
    def test_renames_and_totals(self):
        df = pd.DataFrame({
            "player": ["A", "B"],
            "offense_snaps": [50, 20],
            "defense_snaps": [0, 5],
            "st_snaps": [5, 0],
            "offense_pct": [0.8, 0.3],  # decimals -> percentages
        })
        out = dataset.process_snap_counts(df)
        assert "offensive_snaps" in out.columns
        assert out["total_snaps"].tolist() == [55, 25]
        assert out["offensive_snap_pct"].tolist() == [80.0, 30.0]

    def test_percentages_already_scaled(self):
        df = pd.DataFrame({"player": ["A"], "offense_pct": [85.0]})
        out = dataset.process_snap_counts(df)
        assert out["offensive_snap_pct"].tolist() == [85.0]


class TestSeasonAverages:
    def _df(self):
        return pd.DataFrame({
            "player_id": ["a", "a", "b"],
            "player_name": ["A", "A", "B"],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 1],
            "fanduel_fantasy_points": [10.0, 20.0, 5.0],
        })

    def test_adds_avg_rows(self):
        out = dataset.add_season_averages(self._df())
        avg_rows = out[out["week"] == "AVG"]
        assert len(avg_rows) == 2
        a_avg = avg_rows[avg_rows["player_id"] == "a"].iloc[0]
        assert a_avg["fanduel_fantasy_points"] == 15.0

    def test_original_rows_preserved(self):
        df = self._df()
        out = dataset.add_season_averages(df)
        assert len(out[out["week"] != "AVG"]) == len(df)


class TestRollingAverages:
    def test_week1_uses_previous_season_avg(self):
        df = pd.DataFrame({
            "player_id": ["a"] * 4,
            "season": [2023, 2023, 2023, 2024],
            "week": [1, 2, "AVG", 1],
            "fanduel_fantasy_points": [10.0, 20.0, 15.0, 30.0],
        })
        out = dataset.add_rolling_averages(df)
        week1_2024 = out[(out["season"] == 2024) & (out["week"] == 1)].iloc[0]
        assert week1_2024["avg_fppg"] == 15.0

    def test_midseason_uses_rolling_mean(self):
        df = pd.DataFrame({
            "player_id": ["a"] * 3,
            "season": [2024] * 3,
            "week": [1, 2, 3],
            "fanduel_fantasy_points": [10.0, 20.0, 30.0],
        })
        out = dataset.add_rolling_averages(df)
        week3 = out[out["week"] == 3].iloc[0]
        assert week3["avg_fppg"] == 15.0  # mean of weeks 1-2

    def test_week1_no_history_is_zero(self):
        df = pd.DataFrame({
            "player_id": ["a"],
            "season": [2024],
            "week": [1],
            "fanduel_fantasy_points": [10.0],
        })
        out = dataset.add_rolling_averages(df)
        assert out.iloc[0]["avg_fppg"] == 0
