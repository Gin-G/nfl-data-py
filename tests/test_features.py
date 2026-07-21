import numpy as np
import pandas as pd

from nfl_projections import features


def _game_row(player_id, season, week, points=10.0, **extra):
    row = {
        "player_id": player_id,
        "player_name": extra.pop("player_name", f"Player {player_id}"),
        "player_display_name": extra.pop("player_display_name", f"Player {player_id}"),
        "season": season,
        "week": week,
        "position": extra.pop("position", "RB"),
        "fanduel_fantasy_points": points,
    }
    row.update(extra)
    return row


def make_history(rows):
    return pd.DataFrame([_game_row(*r[:4], **r[4]) if len(r) > 4 else _game_row(*r)
                         for r in rows])


class TestIsRookie:
    def test_veteran_with_prior_seasons(self):
        df = make_history([
            ("vet1", 2024, 1), ("vet1", 2024, 2), ("vet1", 2025, 1),
        ])
        assert features.is_rookie("Player vet1", "vet1", df, 2025) is False

    def test_unknown_player_is_rookie(self):
        df = make_history([("vet1", 2024, 1)])
        assert features.is_rookie("Total Unknown", "nobody", df, 2025) is True

    def test_rookie_with_one_game(self):
        df = make_history([("rk1", 2025, 1)])
        assert features.is_rookie("Player rk1", "rk1", df, 2025) is True

    def test_rookie_with_two_games_uses_ml(self):
        # 2+ current-season games -> treated as ML-predictable, not rookie
        df = make_history([("rk1", 2025, 1), ("rk1", 2025, 2)])
        assert features.is_rookie("Player rk1", "rk1", df, 2025) is False

    def test_avg_rows_ignored(self):
        df = make_history([("rk1", 2025, "AVG"), ("rk1", 2025, 1)])
        assert features.is_rookie("Player rk1", "rk1", df, 2025) is True


class TestDerivedFeatures:
    def test_yards_per_carry(self):
        df = pd.DataFrame({
            "carries": [10, 0],
            "rushing_yards": [50, 0],
        })
        out = features.add_derived_features(df)
        assert out["yards_per_carry"].tolist() == [5.0, 0.0]

    def test_snap_features(self):
        df = pd.DataFrame({
            "offensive_snaps": [60, 10],
            "offensive_snap_pct": [90.0, 15.0],
            "fanduel_fantasy_points": [20.0, 2.0],
        })
        out = features.add_derived_features(df)
        assert out["high_snap_count"].tolist() == [1, 0]
        assert out["is_primary_player"].tolist() == [1, 0]
        assert out["reduced_snaps"].tolist() == [0, 1]
        # 20 points / 60 snaps
        assert np.isclose(out["fantasy_per_snap"].iloc[0], 1 / 3)

    def test_position_specific_usage(self):
        df = pd.DataFrame({
            "position": ["QB", "RB", "WR"],
            "attempts": [30, 0, 0],
            "carries": [2, 15, 0],
            "targets": [0, 5, 8],
        })
        out = features.add_derived_features(df)
        assert out["qb_passing_volume"].tolist() == [30, 0, 0]
        assert out["rb_total_touches"].tolist() == [0, 20, 0]
        assert out["wr_te_targets"].tolist() == [0, 0, 8]

    def test_no_crash_on_missing_columns(self):
        out = features.add_derived_features(pd.DataFrame({"anything": [1]}))
        assert len(out) == 1


class TestTargets:
    def test_next_week_shift(self):
        df = pd.DataFrame({
            "player_id": ["a"] * 3,
            "season": [2024] * 3,
            "week": [1, 2, 3],
            "fanduel_fantasy_points": [10.0, 20.0, 30.0],
        })
        out = features.add_next_week_targets(df, ["fanduel_fantasy_points"])
        # Last game of the season has no next week and is dropped
        assert len(out) == 2
        assert out["next_week_fanduel_fantasy_points"].tolist() == [20.0, 30.0]

    def test_shift_does_not_cross_seasons(self):
        df = pd.DataFrame({
            "player_id": ["a"] * 2,
            "season": [2023, 2024],
            "week": [18, 1],
            "fanduel_fantasy_points": [10.0, 20.0],
        })
        out = features.add_next_week_targets(df, ["fanduel_fantasy_points"])
        assert len(out) == 0  # each season's only game has no successor


class TestCleanTrainingData:
    def test_filters_low_activity_and_old_seasons(self):
        rows = []
        # 5 games in 2024 for an active player
        for w in range(1, 6):
            rows.append(("active", 2024, w, {}))
        # 1 game only for a fringe player
        rows.append(("fringe", 2024, 1, {}))
        # active in 2018 (before min_season)
        for w in range(1, 6):
            rows.append(("old", 2018, w, {}))
        df = make_history([(r[0], r[1], r[2], 10.0) for r in rows])

        out = features.clean_training_data(df, min_season=2020, min_games=3)
        assert set(out["player_id"]) == {"active"}


class TestCategoricalCleaning:
    def test_invalid_positions_become_unknown(self):
        df = pd.DataFrame({"position": ["QB", "OL", None]})
        out = features.clean_categorical_features(df, ["position"])
        assert out["position"].tolist() == ["QB", "Unknown", "Unknown"]

    def test_team_nan_becomes_unknown(self):
        df = pd.DataFrame({"recent_team": ["DAL", None]})
        out = features.clean_categorical_features(df, ["recent_team"])
        assert out["recent_team"].tolist() == ["DAL", "Unknown"]
