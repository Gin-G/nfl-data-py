"""Unit tests for the rank-aware backtest metrics."""
import numpy as np
import pandas as pd
import pytest

from nfl_projections import evaluate


def _weekly(players, weeks=10):
    """players: {name: (per_week_projected, per_week_actual)}"""
    rows = []
    for i, (name, (proj, act)) in enumerate(players.items()):
        for w in range(1, weeks + 1):
            rows.append({"player_id": f"p{i}", "player_name": name, "position": "WR",
                         "week": w, "predicted": proj, "actual": act})
    return pd.DataFrame(rows)


class TestSeasonTotals:
    def test_sums_over_the_same_player_weeks(self):
        df = _weekly({"A": (10.0, 12.0)}, weeks=10)
        out = evaluate.season_totals_from_backtest(df, min_weeks=8)
        assert out["weeks"].iloc[0] == 10
        assert out["proj_total"].iloc[0] == pytest.approx(100.0)
        assert out["act_total"].iloc[0] == pytest.approx(120.0)

    def test_min_weeks_filters_thin_samples(self):
        df = pd.concat([_weekly({"A": (10.0, 12.0)}, weeks=10),
                        _weekly({"B": (9.0, 9.0)}, weeks=3)])
        out = evaluate.season_totals_from_backtest(df, min_weeks=8)
        assert out["player_name"].tolist() == ["A"]

    def test_empty_in_empty_out(self):
        assert evaluate.season_totals_from_backtest(pd.DataFrame()).empty


class TestRankMetrics:
    def test_perfect_ordering_scores_one(self):
        players = {f"P{i}": (float(30 - i), float(30 - i)) for i in range(26)}
        out = evaluate.rank_metrics(_weekly(players), positions=["WR"], verbose=False)
        assert out["spearman"].iloc[0] == pytest.approx(1.0)
        assert out["top12"].iloc[0] == pytest.approx(1.0)
        assert out["top24"].iloc[0] == pytest.approx(1.0)

    def test_reversed_ordering_scores_minus_one(self):
        players = {f"P{i}": (float(i + 1), float(30 - i)) for i in range(26)}
        out = evaluate.rank_metrics(_weekly(players), positions=["WR"], verbose=False)
        assert out["spearman"].iloc[0] == pytest.approx(-1.0)
        assert out["top12"].iloc[0] < 0.2

    def test_detects_a_compressed_elite_tail(self):
        # same ORDER as reality, but the projected top is squashed flat:
        # spearman stays perfect while the spread ratio collapses
        act = {f"P{i}": float(30 - i) for i in range(26)}
        squashed = {n: (20.0 - 0.01 * i, a) for i, (n, a) in enumerate(act.items())}
        out = evaluate.rank_metrics(_weekly(squashed), positions=["WR"], verbose=False)
        assert out["spearman"].iloc[0] == pytest.approx(1.0)
        assert out["r_1_5_proj"].iloc[0] < 1.01        # flat
        assert out["r_1_5_act"].iloc[0] > 1.1          # reality is not
        assert out["r_1_24_proj"].iloc[0] < out["r_1_24_act"].iloc[0]

    def test_top12_overlap_is_a_set_not_an_order(self):
        # projected top-12 contains the same players, shuffled within
        act = {f"P{i}": float(30 - i) for i in range(24)}
        shuffled = {}
        for i, (n, a) in enumerate(act.items()):
            proj = float(30 - i) + (2.0 if i % 2 else -2.0)   # jitter within the tier
            shuffled[n] = (proj, a)
        out = evaluate.rank_metrics(_weekly(shuffled), positions=["WR"], verbose=False)
        assert out["top12"].iloc[0] >= 0.9
        assert out["spearman"].iloc[0] < 1.0

    def test_positions_with_too_few_players_are_skipped(self):
        out = evaluate.rank_metrics(_weekly({"A": (10.0, 10.0)}), positions=["WR"],
                                    verbose=False)
        assert out.empty


class TestHelpers:
    def test_spread_ratio_reads_descending_ranks(self):
        assert evaluate._spread_ratio([10, 8, 6, 4, 2], 1, 5) == pytest.approx(5.0)
        assert evaluate._spread_ratio([2, 4, 6, 8, 10], 1, 5) == pytest.approx(5.0)

    def test_spread_ratio_guards_short_and_zero(self):
        assert np.isnan(evaluate._spread_ratio([10, 8], 1, 5))
        assert np.isnan(evaluate._spread_ratio([10, 8, 6, 4, 0], 1, 5))

    def test_spearman_ignores_monotone_rescaling(self):
        a = [1, 2, 3, 4, 5]
        b = [10, 200, 3000, 40000, 500000]
        assert evaluate._spearman(a, b) == pytest.approx(1.0)
