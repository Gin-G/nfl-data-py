"""Unit tests for depth-chart role -> volume/availability logic (roles.py)."""
import pandas as pd

from nfl_projections import roles


def test_norm_rank_handles_junk():
    assert roles.norm_rank(1.0) == 1
    assert roles.norm_rank("2") == 2
    assert roles.norm_rank("N/A") == roles._UNKNOWN_RANK
    assert roles.norm_rank(None) == roles._UNKNOWN_RANK
    assert roles.norm_rank(float("nan")) == roles._UNKNOWN_RANK
    assert roles.norm_rank(0) == roles._UNKNOWN_RANK  # 0 is not a real rank


def test_deep_backup_no_longer_escapes():
    # The old penalty only fired for role == "backup" (rank 2); deep backups
    # (rank >= 3) got NOTHING. Now every backup rank is discounted, monotonically.
    starter = roles.per_game_role_multiplier("QB", 1)
    backup = roles.per_game_role_multiplier("QB", 2)
    deep = roles.per_game_role_multiplier("QB", 5)
    assert starter == 1.0
    assert deep < backup < starter
    assert deep < 0.2  # a 5th-string QB is not a starter-rate weekly number


def test_no_production_gate():
    # A productive fill-in must still be discounted for being a backup — the old
    # avg_fppg < 8.0 gate (which let the inflated fill-ins through) is gone.
    assert roles.per_game_role_multiplier("RB", 2) < 1.0
    assert roles.per_game_role_multiplier("WR", 3) < 1.0


def test_backup_qb_expected_games_is_tiny():
    assert roles.expected_games("QB", 1) == 17.0
    assert roles.expected_games("QB", 2) <= 2.0   # behind a healthy starter
    assert roles.expected_games("QB", 4) <= 1.0


def test_skill_backups_stay_mostly_available():
    # Skill backups are usually active (volume is trimmed by multiplier/budget,
    # not by sitting) — so games stay high, unlike QBs.
    assert roles.expected_games("RB", 2) >= 14.0
    assert roles.expected_games("WR", 2) >= 14.0


def test_expected_games_capped_at_scheduled():
    assert roles.expected_games("RB", 1, scheduled_games=2) == 2.0


def test_budgets_from_history_beat_default_when_present():
    ds = pd.DataFrame([
        {"season": 2025, "week": 1, "team": "AAA", "position": "RB",
         "fanduel_fantasy_points": 15.0},
        {"season": 2025, "week": 1, "team": "AAA", "position": "RB",
         "fanduel_fantasy_points": 5.0},
        {"season": 2025, "week": 2, "team": "AAA", "position": "RB",
         "fanduel_fantasy_points": 20.0},
    ])
    b = roles.position_budgets(ds, recent_seasons=1)
    # team-game RB sums: wk1 = 20, wk2 = 20 -> mean 20
    assert abs(b["RB"] - 20.0) < 1e-6
    # untouched positions fall back to defaults
    assert b["WR"] == roles.DEFAULT_BUDGETS["WR"]


def test_budgets_default_when_no_dataset():
    assert roles.position_budgets(None) == roles.DEFAULT_BUDGETS


def test_play_weight_matches_expected_games():
    assert roles.play_weight("QB", 1) == 1.0
    assert roles.play_weight("QB", 2) <= 0.12          # ~1.5/17
    assert 0.9 < roles.play_weight("RB", 2) <= 1.0


def test_apply_team_budget_scales_over_budget_group_only():
    frame = pd.DataFrame([
        # DAL RBs sum to 30 > budget 20 -> scaled to 20; ratio preserved
        {"team": "DAL", "position": "RB", "week": 1, "pts": 18.0, "ceil": 36.0},
        {"team": "DAL", "position": "RB", "week": 1, "pts": 12.0, "ceil": 24.0},
        # a lone WR under budget -> untouched
        {"team": "DAL", "position": "WR", "week": 1, "pts": 15.0, "ceil": 30.0},
    ])
    roles.apply_team_budget(frame, {"RB": 20.0, "WR": 34.0}, points_col="pts",
                            group_cols=("team", "position", "week"),
                            scale_cols=["pts", "ceil"])
    rbs = frame[frame.position == "RB"]
    assert abs(rbs["pts"].sum() - 20.0) < 1e-6              # capped to budget
    assert abs(rbs.iloc[0]["pts"] / rbs.iloc[1]["pts"] - 1.5) < 1e-6  # split preserved
    assert abs(rbs.iloc[0]["ceil"] - rbs.iloc[0]["pts"] * 2) < 1e-6   # band scaled too
    assert frame[frame.position == "WR"].iloc[0]["pts"] == 15.0       # under budget: untouched
