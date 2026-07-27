import numpy as np
import pandas as pd

from nfl_projections import opponent


def _game(player_id, season, week, defense, position="RB", points=10.0, **stats):
    row = {
        "player_id": player_id,
        "season": season,
        "week": week,
        "team": stats.pop("team", "OFF"),
        "opponent_team": defense,
        "position": position,
        "fanduel_fantasy_points": points,
        "passing_yards": stats.pop("passing_yards", 0),
        "rushing_yards": stats.pop("rushing_yards", 0),
        "receiving_yards": stats.pop("receiving_yards", 0),
        "passing_tds": stats.pop("passing_tds", 0),
        "rushing_tds": stats.pop("rushing_tds", 0),
        "receiving_tds": stats.pop("receiving_tds", 0),
        "receptions": stats.pop("receptions", 0),
        "passing_interceptions": stats.pop("interceptions", 0),
        "rushing_fumbles": stats.pop("rushing_fumbles", 0),
        "receiving_fumbles": stats.pop("receiving_fumbles", 0),
    }
    row.update(stats)
    return row


class TestBuildDefenseForm:
    def _dataset(self):
        # Defense DEF faces one RB each week, escalating fantasy points
        return pd.DataFrame([
            _game("p1", 2024, 1, "DEF", points=10.0, rushing_yards=100, rushing_tds=1, receptions=3),
            _game("p2", 2024, 2, "DEF", points=20.0, rushing_yards=200, rushing_tds=2, receptions=5),
            _game("p3", 2024, 3, "DEF", points=30.0, rushing_yards=300, rushing_tds=3, receptions=7),
        ])

    def test_form_entering_week_excludes_current_and_future(self):
        form = opponent.build_defense_form(self._dataset(), window=6)
        # No form for week 1 (nothing before it)
        assert form[form["week"] == 1].empty
        # Entering week 2 = week 1 only
        wk2 = form[form["week"] == 2].iloc[0]
        assert wk2["opp_fppg_allowed"] == 10.0
        assert wk2["opp_yards_allowed_pg"] == 100.0
        # Entering week 3 = mean of weeks 1 and 2 (no leakage from week 3)
        wk3 = form[form["week"] == 3].iloc[0]
        assert wk3["opp_fppg_allowed"] == 15.0
        assert wk3["opp_yards_allowed_pg"] == 150.0

    def test_window_limits_lookback(self):
        form = opponent.build_defense_form(self._dataset(), window=1)
        # window=1 -> entering week 3 sees only week 2
        wk3 = form[form["week"] == 3].iloc[0]
        assert wk3["opp_fppg_allowed"] == 20.0

    def test_position_specific_yards(self):
        df = pd.DataFrame([
            _game("q1", 2024, 1, "DEF", position="QB", points=18.0, passing_yards=250, passing_tds=2, interceptions=1),
            _game("q2", 2024, 2, "DEF", position="QB", points=18.0, passing_yards=250, passing_tds=2, interceptions=1),
        ])
        form = opponent.build_defense_form(df, window=6)
        wk2 = form[form["week"] == 2].iloc[0]
        assert wk2["opp_yards_allowed_pg"] == 250.0      # passing for QB
        assert wk2["opp_turnovers_forced_pg"] == 1.0     # interceptions for QB
        assert wk2["opp_receptions_allowed_pg"] == 0.0   # QBs don't catch

    def test_defensive_rank(self):
        df = pd.DataFrame([
            _game("a", 2024, 1, "GOOD_D", points=5.0),
            _game("b", 2024, 2, "GOOD_D", points=5.0),
            _game("c", 2024, 1, "BAD_D", points=25.0),
            _game("d", 2024, 2, "BAD_D", points=25.0),
        ])
        form = opponent.build_defense_form(df, window=6)
        wk2 = form[form["week"] == 2].set_index("defense_team")
        assert wk2.loc["GOOD_D", "opp_defensive_rank"] == 1
        assert wk2.loc["BAD_D", "opp_defensive_rank"] == 2


class TestScheduleMap:
    def test_home_and_away_rows(self):
        schedule = pd.DataFrame({
            "season": [2024], "week": [1], "game_type": ["REG"],
            "home_team": ["DAL"], "away_team": ["NYG"],
        })
        smap = opponent.build_schedule_map([2024], schedule=schedule)
        dal = smap[smap["team"] == "DAL"].iloc[0]
        nyg = smap[smap["team"] == "NYG"].iloc[0]
        assert dal["opponent"] == "NYG" and dal["is_home_game"] == 1.0
        assert nyg["opponent"] == "DAL" and nyg["is_home_game"] == 0.0

    def test_drops_non_regular_season(self):
        schedule = pd.DataFrame({
            "season": [2024, 2024], "week": [1, 1], "game_type": ["REG", "POST"],
            "home_team": ["DAL", "KC"], "away_team": ["NYG", "BUF"],
        })
        smap = opponent.build_schedule_map([2024], schedule=schedule)
        assert set(smap["team"]) == {"DAL", "NYG"}


class TestNextGameKeys:
    def test_shift_within_player_season(self):
        df = pd.DataFrame({
            "player_id": ["a", "a", "a"],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 3],
            "team": ["DAL", "DAL", "DAL"],
        })
        out = opponent.add_next_game_keys(df)
        assert out["next_game_week"].tolist() == [2, 3, np.nan] or (
            out["next_game_week"].iloc[0] == 2 and out["next_game_week"].iloc[1] == 3
            and pd.isna(out["next_game_week"].iloc[2])
        )

    def test_no_cross_season_leak(self):
        df = pd.DataFrame({
            "player_id": ["a", "a"],
            "season": [2023, 2024],
            "week": [18, 1],
            "team": ["DAL", "DAL"],
        })
        out = opponent.add_next_game_keys(df).sort_values(["season", "week"])
        assert pd.isna(out["next_game_week"].iloc[0])  # last game of 2023


class TestMatchupAndTraining:
    def _dataset(self):
        return pd.DataFrame([
            _game("p1", 2024, 1, "DEF", team="OFF", points=10.0, rushing_yards=100),
            _game("p1", 2024, 2, "DEF", team="OFF", points=20.0, rushing_yards=200),
            _game("p1", 2024, 3, "DEF", team="OFF", points=30.0, rushing_yards=300),
        ])

    def _schedule(self):
        # OFF plays DEF every week
        return pd.DataFrame({
            "season": [2024, 2024, 2024],
            "week": [1, 2, 3],
            "game_type": ["REG", "REG", "REG"],
            "home_team": ["OFF", "OFF", "OFF"],
            "away_team": ["DEF", "DEF", "DEF"],
        })

    def test_matchup_table_joins_defense_form(self):
        smap = opponent.build_schedule_map([2024], schedule=self._schedule())
        table = opponent.build_matchup_table(self._dataset(), smap)
        off_wk3 = table[(table["team"] == "OFF") & (table["week"] == 3)
                        & (table["position"] == "RB")].iloc[0]
        # OFF's opponent DEF allowed mean 15 fppg entering week 3
        assert off_wk3["opp_fppg_allowed"] == 15.0
        assert off_wk3["is_home_game"] == 1.0

    def test_attach_training_uses_next_game(self):
        smap = opponent.build_schedule_map([2024], schedule=self._schedule())
        table = opponent.build_matchup_table(self._dataset(), smap)

        df = self._dataset().copy()
        df = opponent.add_next_game_keys(df)
        attached = opponent.attach_training_features(df, table)

        # Row for week 1 predicts week 2; opp features describe DEF entering week 2 (=10)
        wk1 = attached[attached["week"] == 1].iloc[0]
        assert wk1["opp_fppg_allowed"] == 10.0
        # Row for week 2 predicts week 3; DEF entering week 3 = 15
        wk2 = attached[attached["week"] == 2].iloc[0]
        assert wk2["opp_fppg_allowed"] == 15.0

    def test_lookup_week_features_neutral_for_missing(self):
        smap = opponent.build_schedule_map([2024], schedule=self._schedule())
        table = opponent.build_matchup_table(self._dataset(), smap)
        wk = opponent.lookup_week_features(table, 2024, 1)
        # Week 1 has no prior defense form -> neutral fallback
        assert wk.loc[("OFF", "RB"), "opp_fppg_allowed"] == opponent.NEUTRAL_VALUES["opp_fppg_allowed"]
