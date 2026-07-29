"""Usage-based Monte-Carlo game simulator.

The point model gives a matchup-neutral MEAN that we've shown is at its ceiling
(EXPERIMENTS.md). A simulator adds what the mean can't: outcome DISTRIBUTIONS (floor /
ceiling / P(boom)) and PLAYER CORRELATION (QB<->WR stacks) — the things that matter for DFS
lineup construction, not a better point estimate.

Each simulation of a game:
  1. draws a team-level scoring ENVIRONMENT factor (shared by all of a team's players) —
     this single shared draw is what correlates a QB with his receivers (a shootout lifts
     the whole passing game together);
  2. samples each player's opportunity (targets / carries / attempts ~ Poisson * env);
  3. turns opportunity into yards (per-unit efficiency ~ Normal) and TDs (~ Poisson * env),
     TDs being the boom/bust driver a Gaussian projection misses;
  4. scores it as FanDuel points.
Run N times -> per-player distribution + a full sims matrix for joint/stack queries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scoring import fanduel_points

# efficiency dispersion (SD of per-opportunity yards) — game-to-game realism
_YPT_SD, _YPC_SD, _YPA_SD = 4.5, 2.4, 2.3
# team scoring environment split into a shared team factor plus pass-/rush-specific factors.
# The pass factor is shared by the QB and his pass-catchers -> that's what drives the
# QB<->WR stack correlation; the rush factor couples the QB's legs with the backfield.
_ENV_TEAM, _ENV_PASS, _ENV_RUSH = 0.20, 0.30, 0.28
_ENV_IDIO = 0.22  # per-player, per-sim role fluctuation (widens the marginal band)


def build_expectations(prior_games: pd.DataFrame) -> dict | None:
    """Per-game expectations for one player from their recent prior games (leakage-free)."""
    if prior_games.empty:
        return None
    m = prior_games.mean(numeric_only=True)

    def g(c):
        return float(m.get(c, 0.0) or 0.0)

    exp = {
        "exp_targets": g("targets"), "exp_carries": g("carries"), "exp_att": g("attempts"),
        "exp_rec_td": g("receiving_tds"), "exp_rush_td": g("rushing_tds"),
        "exp_pass_td": g("passing_tds"), "exp_int": g("passing_interceptions") or g("interceptions"),
        "ypt": g("receiving_yards") / g("targets") if g("targets") > 0.3 else 0.0,
        "ypc": g("rushing_yards") / g("carries") if g("carries") > 0.3 else 0.0,
        "ypa": g("passing_yards") / g("attempts") if g("attempts") > 0.3 else 0.0,
    }
    return exp


def _env_factor(sigma, n, rng):
    return rng.lognormal(mean=-0.5 * sigma ** 2, sigma=sigma, size=n)  # mean 1


def _simulate_team(exps: list[dict], n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """FanDuel points, shape (n_players, n_sims), for one team. A shared team factor plus
    shared pass-/rush-specific factors (drawn once per sim) correlate the teammates —
    the pass factor is what stacks the QB with his receivers."""
    team_env = _env_factor(_ENV_TEAM, n_sims, rng)
    pass_env = team_env * _env_factor(_ENV_PASS, n_sims, rng)   # QB + pass-catchers
    rush_env = team_env * _env_factor(_ENV_RUSH, n_sims, rng)   # backfield + QB legs
    out = np.zeros((len(exps), n_sims))
    for i, e in enumerate(exps):
        idio = _env_factor(_ENV_IDIO, n_sims, rng)  # this player's own week-to-week swing
        tgt = rng.poisson(np.clip(e["exp_targets"] * pass_env * idio, 0, None))
        car = rng.poisson(np.clip(e["exp_carries"] * rush_env * idio, 0, None))
        att = rng.poisson(np.clip(e["exp_att"] * pass_env * idio, 0, None))
        rec_y = np.maximum(0, tgt * rng.normal(e["ypt"], _YPT_SD, n_sims)) if e["ypt"] else np.zeros(n_sims)
        rush_y = np.maximum(0, car * rng.normal(e["ypc"], _YPC_SD, n_sims)) if e["ypc"] else np.zeros(n_sims)
        pass_y = np.maximum(0, att * rng.normal(e["ypa"], _YPA_SD, n_sims)) if e["ypa"] else np.zeros(n_sims)
        # receptions: a catch rate around ~65% of targets, bounded by targets
        rec = np.minimum(tgt, rng.binomial(np.maximum(tgt, 0), 0.65))
        rec_td = rng.poisson(np.clip(e["exp_rec_td"] * pass_env, 0, None))
        rush_td = rng.poisson(np.clip(e["exp_rush_td"] * rush_env, 0, None))
        pass_td = rng.poisson(np.clip(e["exp_pass_td"] * pass_env, 0, None))
        ints = rng.poisson(np.clip(e["exp_int"] * np.ones(n_sims), 0, None))
        out[i] = fanduel_points(
            passing_yards=pass_y, passing_tds=pass_td, interceptions=ints,
            rushing_yards=rush_y, rushing_tds=rush_td,
            receptions=rec, receiving_yards=rec_y, receiving_tds=rec_td)
    return out


def simulate(players: pd.DataFrame, n_sims: int = 1000, seed: int | None = None,
             mean_anchor: str | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Simulate a slate. `players` needs a ``team`` column and the exp_* / ypt/ypc/ypa
    fields (from build_expectations). Returns (summary_df, sims) where sims is
    (n_players, n_sims) aligned to summary_df rows — use it for stack/correlation queries.

    `mean_anchor`: optional column of externally-projected means (e.g. the point model's
    projection). When given, each player's simulations are rescaled so their mean matches
    it — the model supplies the accurate mean, the simulator supplies the shape and
    correlation around it. This is the recommended production path.
    """
    rng = np.random.default_rng(seed)
    players = players.reset_index(drop=True)
    sims = np.zeros((len(players), n_sims))
    for _, idx in players.groupby("team").groups.items():
        idx = list(idx)
        exps = [players.loc[i, ["exp_targets", "exp_carries", "exp_att", "exp_rec_td",
                                "exp_rush_td", "exp_pass_td", "exp_int", "ypt", "ypc", "ypa"]].to_dict()
                for i in idx]
        team_sims = _simulate_team(exps, n_sims, rng)
        for k, i in enumerate(idx):
            sims[i] = team_sims[k]
    if mean_anchor and mean_anchor in players.columns:
        cur = sims.mean(axis=1, keepdims=True)
        target = players[mean_anchor].to_numpy(dtype=float).reshape(-1, 1)
        scale = np.divide(target, cur, out=np.ones_like(cur), where=cur > 0.5)
        sims = sims * scale
    summary = players[[c for c in ("player_id", "player_name", "position", "team") if c in players.columns]].copy()
    summary["mean"] = sims.mean(axis=1).round(2)
    summary["floor"] = np.percentile(sims, 10, axis=1).round(2)
    summary["median"] = np.percentile(sims, 50, axis=1).round(2)
    summary["ceiling"] = np.percentile(sims, 90, axis=1).round(2)
    summary["p_boom_20"] = (sims >= 20).mean(axis=1).round(3)
    return summary, sims


_STAT_COLS = ["targets", "carries", "attempts", "receiving_yards", "rushing_yards",
              "passing_yards", "receiving_tds", "rushing_tds", "passing_tds",
              "passing_interceptions"]


def project_distributions(proj_frame: pd.DataFrame, history: pd.DataFrame, *, n_sims: int = 1000,
                          seed: int = 0, trailing: int = 6, anchor_col: str = "fanduel_fantasy_points"):
    """Simulator-based distributions for an existing projection frame, anchored to the
    model's mean. For each player, expectations come from their `trailing` most recent
    games in `history`; the simulation is rescaled so its mean matches the model
    projection (`anchor_col`) while its shape + team correlation come from the sim.

    Returns (summary, sims) covering only players with enough history to simulate
    (rookies / no-history players are omitted — keep their model/prior band). `summary`
    has player_id/position/team + mean/floor/median/ceiling/p_boom_20.
    """
    need = [c for c in _STAT_COLS if c in history.columns]
    hist = history.sort_values(["season", "week"]).groupby("player_id")
    exps, keep = [], []
    for _, r in proj_frame.iterrows():
        pid = r["player_id"]
        try:
            pg = hist.get_group(pid).tail(trailing)
        except KeyError:
            continue
        if len(pg) < 2:
            continue
        e = build_expectations(pg[need])
        if e is None or sum(abs(e[k]) for k in ("exp_targets", "exp_carries", "exp_att")) < 0.5:
            continue  # no usable opportunity signal
        e.update(player_id=pid, team=r.get("team"), position=r.get("position"),
                 player_name=r.get("player_name"), proj=float(r[anchor_col]))
        exps.append(e)
        keep.append(pid)
    if not exps:
        return pd.DataFrame(), np.zeros((0, n_sims))
    pl = pd.DataFrame(exps)
    return simulate(pl, n_sims=n_sims, seed=seed, mean_anchor="proj")


def stack_distribution(sims: np.ndarray, idxs: list[int]) -> np.ndarray:
    """Combined per-sim points for a set of players (e.g. a QB + WR stack) — sums the SAME
    simulations so their correlation is preserved."""
    return sims[idxs, :].sum(axis=0)
