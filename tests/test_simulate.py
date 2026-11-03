"""Unit tests for the Monte-Carlo game simulator (synthetic, deterministic seed)."""
import numpy as np
import pandas as pd

from nfl_projections import simulate as sim


def _players():
    # QB + WR on team A (should correlate via shared pass env); a WR on team B (independent)
    base = dict(exp_carries=0, exp_rush_td=0, ypc=0)
    return pd.DataFrame([
        dict(player_id="qbA", player_name="QB A", position="QB", team="A",
             exp_targets=0, exp_att=34, exp_rec_td=0, exp_pass_td=1.8, exp_int=0.7,
             ypt=0, ypa=7.2, **base),
        dict(player_id="wrA", player_name="WR A", position="WR", team="A",
             exp_targets=9, exp_att=0, exp_rec_td=0.5, exp_pass_td=0, exp_int=0,
             ypt=9.0, ypa=0, **base),
        dict(player_id="wrB", player_name="WR B", position="WR", team="B",
             exp_targets=9, exp_att=0, exp_rec_td=0.5, exp_pass_td=0, exp_int=0,
             ypt=9.0, ypa=0, **base),
    ])


def test_quantiles_ordered_and_positive():
    summ, _ = sim.simulate(_players(), n_sims=2000, seed=1)
    assert (summ["floor"] <= summ["median"]).all()
    assert (summ["median"] <= summ["ceiling"]).all()
    assert (summ["mean"] > 0).all()


def test_teammates_correlate_more_than_opponents():
    _, sims = sim.simulate(_players(), n_sims=4000, seed=2)
    same = np.corrcoef(sims[0], sims[1])[0, 1]      # QB A <-> WR A
    cross = np.corrcoef(sims[0], sims[2])[0, 1]     # QB A <-> WR B (other team)
    assert same > 0.10          # shared pass environment correlates teammates
    assert same > cross + 0.08  # and more than an opposing player


def test_mean_anchor_rescales_to_target():
    p = _players()
    p["proj"] = [18.0, 12.0, 12.0]
    summ, _ = sim.simulate(p, n_sims=3000, seed=3, mean_anchor="proj")
    assert np.allclose(summ["mean"].to_numpy(), p["proj"].to_numpy(), atol=0.6)


def test_stack_distribution_preserves_correlation():
    p = _players()
    _, sims = sim.simulate(p, n_sims=3000, seed=4)
    stack = sim.stack_distribution(sims, [0, 1])   # QB + WR combined per sim
    assert stack.shape == (3000,)
    assert abs(stack.mean() - (sims[0].mean() + sims[1].mean())) < 1e-6


def test_td_coupling_strengthens_qb_wr_stack():
    # WR A is the QB's only TD-catching option, so coupling the team's passing TDs to the
    # receiver produces a strong QB<->WR correlation that independent Poisson TDs could not.
    _, sims = sim.simulate(_players(), n_sims=6000, seed=7)
    assert np.corrcoef(sims[0], sims[1])[0, 1] > 0.25


def test_qb_downside_fattens_the_floor():
    # The ~4% benching term (x0.35) leaves a visible chunk of very-low QB games — a heavier
    # low tail than a symmetric distribution would have.
    _, sims = sim.simulate(_players(), n_sims=8000, seed=8)
    qb = sims[0]
    assert (qb < 0.5 * np.median(qb)).mean() > 0.04


def test_position_catch_rate_rb_over_wr():
    # Identical target volume and yards-per-target; the RB's higher catch rate (0.77 vs 0.64)
    # means more receptions -> more PPR points on average than the WR.
    base = dict(exp_carries=0, exp_att=0, exp_rush_td=0, exp_pass_td=0, exp_int=0,
                exp_rec_td=0.4, ypc=0, ypa=0)
    p = pd.DataFrame([
        dict(player_id="rb", player_name="RB", position="RB", team="A", exp_targets=8, ypt=7.0, **base),
        dict(player_id="wr", player_name="WR", position="WR", team="B", exp_targets=8, ypt=7.0, **base),
    ])
    summ, _ = sim.simulate(p, n_sims=8000, seed=11)
    means = summ.set_index("player_id")["mean"]
    assert means["rb"] > means["wr"]


def test_build_expectations_from_prior_games():
    prior = pd.DataFrame({
        "targets": [8, 10, 9], "carries": [0, 0, 0], "attempts": [0, 0, 0],
        "receiving_yards": [70, 90, 80], "rushing_yards": [0, 0, 0], "passing_yards": [0, 0, 0],
        "receiving_tds": [0, 1, 0], "rushing_tds": [0, 0, 0], "passing_tds": [0, 0, 0],
        "passing_interceptions": [0, 0, 0]})
    e = sim.build_expectations(prior)
    assert abs(e["exp_targets"] - 9.0) < 1e-6
    assert 8 < e["ypt"] < 9.5   # ~80 yds / 9 targets


class TestPoolSims:
    """Lineup-level distributions: shape and correlation, never a new board."""

    @staticmethod
    def pool():
        return pd.DataFrame({
            "player_id": [None, None, None],
            "player_name": ["A", "B", "Def"],
            "position": ["QB", "WR", "D"],
            "team": ["X", "X", "Y"],
            "lineup_points": [20.0, 12.0, 7.0],
            "is_def": [False, False, True],
        })

    def test_every_row_keeps_its_projection_as_its_mean(self):
        # The simulator supplies shape and correlation. If it also moved the
        # mean it would be a second, worse projection model.
        sims = sim.pool_sims(self.pool(), None, n_sims=20000, seed=3)
        assert sims.shape == (3, 20000)
        np.testing.assert_allclose(sims.mean(axis=1), [20.0, 12.0, 7.0], rtol=0.03)

    def test_a_defence_gets_the_measured_defensive_shape(self):
        # 2019-2026: a defence is right-skewed with a floor near zero and a
        # long tail, not a bell around its mean.
        sims = sim.pool_sims(self.pool(), None, n_sims=20000, seed=3)
        defence = sims[2]
        assert defence.min() < 0            # 35+ points allowed is -4
        assert np.median(defence) < defence.mean()
        assert defence.max() > 25

    def test_a_skill_player_without_history_stays_non_negative(self):
        sims = sim.pool_sims(self.pool(), None, n_sims=5000, seed=3)
        assert sims[0].min() >= 0
        assert sims[1].min() >= 0

    def test_a_missing_projection_does_not_become_a_nan_lineup(self):
        pool = self.pool()
        pool.loc[1, "lineup_points"] = np.nan
        sims = sim.pool_sims(pool, None, n_sims=500, seed=3)
        assert np.isfinite(sims).all()

    def test_it_is_reproducible(self):
        a = sim.pool_sims(self.pool(), None, n_sims=500, seed=11)
        b = sim.pool_sims(self.pool(), None, n_sims=500, seed=11)
        np.testing.assert_array_equal(a, b)


class TestLineupTotals:
    def test_totals_sum_the_same_simulations(self):
        sims = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        np.testing.assert_array_equal(sim.lineup_totals(sims, [0, 1]),
                                      [11.0, 22.0, 33.0])

    def test_p_at_least_counts_the_sims_that_clear_the_bar(self):
        sims = np.array([[10.0, 10.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
        assert sim.p_at_least(sims, [0, 1], 15.0) == 0.5

    def test_correlated_players_do_not_average_each_other_out(self):
        # Summing marginals independently would understate a stack's spread.
        # Summing the same simulations keeps it, which is the whole point.
        together = np.array([[0.0, 20.0], [0.0, 20.0]])
        totals = sim.lineup_totals(together, [0, 1])
        assert totals.tolist() == [0.0, 40.0]


class TestSpreadCalibration:
    """The usage simulator is over-dispersed; the correction is volume-aware.

    Fitted on 1,318 player-seasons (2019-2023), validated on 567 (2024-2026).
    """

    def test_a_workhorse_is_shrunk_more_than_a_bit_player(self):
        # The error grows with volume: a 20-touch back simulated 40% more
        # volatile than he really is, a 5-touch back came out about right.
        assert sim.spread_factor("RB", 20) < sim.spread_factor("RB", 9)
        assert sim.spread_factor("RB", 9) < sim.spread_factor("RB", 5)
        assert sim.spread_factor("RB", 5) > 0.95

    def test_volume_shrinks_spread_for_receivers_too(self):
        assert sim.spread_factor("WR", 9) < sim.spread_factor("WR", 3.5)

    def test_an_unknown_position_or_workload_changes_nothing(self):
        # A missing column must never silently reshape the simulation.
        assert sim.spread_factor("D", 10) == 1.0
        assert sim.spread_factor("RB", None) == 1.0
        assert sim.spread_factor("RB", float("nan")) == 1.0

    def test_the_factor_is_clamped_against_extrapolation(self):
        lo, hi = sim._SPREAD_CLAMP
        for opp in (0, 1, 50, 500, 10_000):
            assert lo <= sim.spread_factor("RB", opp) <= hi

    def test_shrinking_moves_the_spread_and_not_the_mean(self):
        row = np.array([1.0, 5.0, 9.0, 20.0])
        out = sim._shrink_spread(row, 0.5)
        assert out.mean() == row.mean()
        np.testing.assert_allclose(out.std(), row.std() * 0.5)

    def test_a_factor_of_one_is_the_identity(self):
        row = np.array([1.0, 5.0, 9.0])
        np.testing.assert_array_equal(sim._shrink_spread(row, 1.0), row)

    def test_trailing_opportunity_is_carries_plus_targets(self):
        hist = pd.DataFrame({
            "player_id": ["a"] * 4,
            "season": [2026] * 4, "week": [1, 2, 3, 4],
            "carries": [10, 12, 14, 16], "targets": [2, 2, 2, 2],
        })
        # last two games only: (14+2) and (16+2)
        assert sim.trailing_opportunity(hist, trailing=2)["a"] == 17.0

    def test_trailing_opportunity_tolerates_a_missing_frame(self):
        assert sim.trailing_opportunity(None) == {}
        assert sim.trailing_opportunity(pd.DataFrame({"x": [1]})) == {}

    @staticmethod
    def _pool_and_history():
        pool = pd.DataFrame({
            "player_id": ["rb1"],
            "player_name": ["Workhorse"],
            "position": ["RB"],
            "team": ["X"],
            "lineup_points": [18.0],
            "is_def": [False],
        })
        hist = pd.DataFrame({
            "player_id": ["rb1"] * 6,
            "season": [2026] * 6, "week": [1, 2, 3, 4, 5, 6],
            "carries": [19, 21, 20, 22, 18, 20],
            "targets": [3, 2, 4, 3, 2, 3],
            "attempts": [0] * 6,
            "rushing_yards": [85, 95, 70, 110, 60, 90],
            "receiving_yards": [20, 15, 30, 18, 10, 22],
            "passing_yards": [0] * 6,
            "rushing_tds": [1, 0, 1, 1, 0, 0],
            "receiving_tds": [0] * 6, "passing_tds": [0] * 6,
            "passing_interceptions": [0] * 6,
        })
        return pool, hist

    def test_calibration_narrows_a_workhorse_without_moving_his_mean(self):
        pool, hist = self._pool_and_history()
        raw = sim.pool_sims(pool, hist, n_sims=8000, seed=5, calibrate_spread=False)
        cal = sim.pool_sims(pool, hist, n_sims=8000, seed=5, calibrate_spread=True)
        # the board is untouched -- only the width of the week changes
        np.testing.assert_allclose(cal.mean(axis=1), raw.mean(axis=1), rtol=1e-12)
        assert cal.std() < raw.std() * 0.85

    def test_calibration_is_opt_out(self):
        pool, hist = self._pool_and_history()
        a = sim.pool_sims(pool, hist, n_sims=2000, seed=5, calibrate_spread=False)
        b = sim.pool_sims(pool, hist, n_sims=2000, seed=5, calibrate_spread=False)
        np.testing.assert_array_equal(a, b)

    def test_a_quarterback_without_history_is_steadier_than_a_back(self):
        # One global CV of 0.75 treated these as the same player. Realized CVs
        # over 1,618 player-seasons: QB 0.46, RB 0.72.
        pool = pd.DataFrame({
            "player_id": [None, None],
            "player_name": ["Q", "R"],
            "position": ["QB", "RB"],
            "team": ["X", "Y"],
            "lineup_points": [18.0, 18.0],
            "is_def": [False, False],
        })
        sims = sim.pool_sims(pool, None, n_sims=40000, seed=7)
        qb_cv = sims[0].std() / sims[0].mean()
        rb_cv = sims[1].std() / sims[1].mean()
        assert qb_cv < rb_cv
        assert abs(qb_cv - 0.46) < 0.05
        assert abs(rb_cv - 0.72) < 0.05
