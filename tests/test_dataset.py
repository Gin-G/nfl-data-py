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


class TestLoadThroughLatest:
    @staticmethod
    def _loader(fail_on, message):
        calls = []

        def load(seasons):
            calls.append(list(seasons))
            if fail_on in seasons:
                raise ConnectionError(message)
            return pd.DataFrame({"season": seasons})
        return load, calls

    def test_drops_an_unpublished_newest_season(self):
        load, calls = self._loader(2026, "404 Client Error: Not Found for url")
        frame, loaded = dataset.load_through_latest(load, [2024, 2025, 2026], "stats")
        assert loaded == [2024, 2025]
        assert frame["season"].tolist() == [2024, 2025]
        assert calls == [[2024, 2025, 2026], [2024, 2025]]

    def test_season_range_check_counts_as_unpublished(self):
        load, _ = self._loader(2027, "Season must be between 2012 and 2026")
        _, loaded = dataset.load_through_latest(load, [2026, 2027], "snaps")
        assert loaded == [2026]

    def test_any_other_failure_still_raises(self):
        # A flaky download must not quietly fall back to last season's data.
        import pytest

        load, _ = self._loader(2026, "Connection reset by peer")
        with pytest.raises(ConnectionError):
            dataset.load_through_latest(load, [2025, 2026], "stats")

    def test_only_the_newest_season_may_be_missing(self):
        import pytest

        load, _ = self._loader(2024, "404 Client Error")
        with pytest.raises(ConnectionError):
            dataset.load_through_latest(load, [2024, 2025, 2026], "stats")


class TestMergeSnapsById:
    def _stats(self):
        return pd.DataFrame({
            "player_id": ["wr", "lb", "burden"],
            "player_display_name": ["Justin Jefferson", "Justin Jefferson",
                                    "Luther Burden"],
            "team": ["MIN", "CLE", "CHI"], "season": 2026, "week": 5,
        })

    def _snaps(self):
        return dataset.process_snap_counts(pd.DataFrame({
            "player": ["Justin Jefferson", "Justin Jefferson", "Luther Burden III"],
            "pfr_player_id": ["JeffJu00", "JeffJu01", "BurdLu00"],
            "team": ["MIN", "CLE", "CHI"], "season": 2026, "week": 5,
            "offense_snaps": [63, 0, 40], "offense_pct": [0.89, 0.0, 0.6],
            "defense_snaps": [0, 55, 0], "st_snaps": [0, 10, 2],
        }))

    def test_namesakes_get_their_own_snaps(self):
        xw = pd.DataFrame({"gsis_id": ["wr", "lb", "burden"],
                           "pfr_id": ["JeffJu00", "JeffJu01", "BurdLu00"]})
        out = dataset.merge_snaps(self._stats(), self._snaps(), xw)
        assert len(out) == 3
        assert out.set_index("player_id").offensive_snaps.to_dict() == {
            "wr": 63, "lb": 0, "burden": 40}

    def test_without_a_crosswalk_falls_back_to_name_on_team(self):
        out = dataset.merge_snaps(self._stats(), self._snaps(), None)
        assert len(out) == 3
        snaps = out.set_index("player_id").offensive_snaps
        assert snaps["wr"] == 63 and snaps["lb"] == 0
        assert snaps["burden"] == 40          # the suffix does not lose him


class TestMergeDepth:
    def test_one_row_per_player_week_at_his_best_line(self):
        df = pd.DataFrame({"player_id": ["a"], "season": [2023], "week": [3],
                           "position": ["WR"]})
        depth = pd.DataFrame({"gsis_id": ["a", "a", "a"], "season": 2023, "week": 3,
                              "position": ["WR", "WR", "KR"],
                              "depth_team": ["2", "1", "3"]})
        out = dataset.merge_depth(df, depth)
        assert len(out) == 1
        assert out.depth_team.iloc[0] == "1"
