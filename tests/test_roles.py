"""Unit tests for depth-chart role -> volume/availability logic (roles.py)."""
import pandas as pd
import pytest

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


# --------------------------------------------------------------------------- promotion

def test_effective_rank_only_counts_absences_above():
    assert roles.effective_rank(2, [1]) == 1
    assert roles.effective_rank(3, [1, 2]) == 1
    assert roles.effective_rank(3, [1]) == 2
    # a backup being out does not promote the starter
    assert roles.effective_rank(1, [2, 3]) == 1
    # and it never goes below 1
    assert roles.effective_rank(2, [1, 1]) == 1


def test_effective_rank_is_identity_with_nobody_out():
    for rank in (1, 2, 3, 4, 9):
        assert roles.effective_rank(rank, ()) == rank


def test_promotion_is_neutral_when_nobody_above_is_out():
    assert roles.promotion_multiplier("RB", 2, ()) == 1.0
    assert roles.promotion_multiplier("RB", 1, [2, 3]) == 1.0


def test_promotion_undoes_the_discount():
    """RB2 with RB1 out is charged the rate of the rank he is filling.

    The 0.80 discount goes and nothing is added on top: the fitted 1.42 was
    withdrawn when it failed to pay in FanDuel points through the trained model
    (experiments/promotion_endtoend.py). Asserted against the constant rather
    than a literal so the test states the composition, not the number.
    """
    promo = roles.promotion_multiplier("RB", 2, [1])
    expected = (1.0 / 0.80) * roles.PROMOTION_FACTOR["RB"]
    assert promo == pytest.approx(expected, rel=1e-6)
    assert roles.PROMOTION_FACTOR["RB"] == 1.0


def test_only_the_quarterback_keeps_an_extra_factor():
    """QB is the one position with a lift beyond the rank change.

    Everywhere else the promotion is parameter-free, which is what the
    end-to-end test preferred at every position it could resolve.
    """
    assert roles.PROMOTION_FACTOR["QB"] > 1.0
    for position in ("RB", "WR", "TE"):
        assert roles.PROMOTION_FACTOR[position] == 1.0
        promo = roles.promotion_multiplier(position, 2, [1])
        static = roles.per_game_role_multiplier(position, 2)
        assert promo == pytest.approx(1.0 / static, rel=1e-6)


def test_promotion_is_larger_for_a_deeper_backup():
    """An RB3 stepping up two spots gains more than an RB2 stepping up one."""
    assert (roles.promotion_multiplier("RB", 3, [1, 2])
            > roles.promotion_multiplier("RB", 2, [1]))


def test_quarterback_promotion_is_the_biggest():
    """A backup QB who plays is the starter; nothing else is that binary."""
    qb = roles.promotion_multiplier("QB", 2, [1])
    assert qb > roles.promotion_multiplier("RB", 2, [1])
    assert qb > roles.promotion_multiplier("WR", 2, [1])


def test_receivers_get_the_discount_back_and_no_more():
    """The fitted extra factor is 1.00 for WR and TE, so only the discount goes."""
    for position in ("WR", "TE"):
        promo = roles.promotion_multiplier(position, 2, [1])
        static = roles.per_game_role_multiplier(position, 2)
        assert promo == pytest.approx(1.0 / static, rel=1e-6)


def test_promotion_handles_unreadable_ranks():
    """Depth charts carry 'N/A' and None; neither may raise here."""
    assert roles.promotion_multiplier("RB", None, [1]) == pytest.approx(
        roles.promotion_multiplier("RB", roles.norm_rank(None), [1]))
    assert roles.effective_rank("N/A", [1]) >= 1
