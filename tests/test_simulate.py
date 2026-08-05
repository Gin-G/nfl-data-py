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
