"""Rookie draft-capital prior.

Rookies have no game history, so the model can't project them; the Projector's old
fallback was hand-tuned multipliers. Draft capital, however, predicts rookie
early-season production cleanly and monotonically (measured: corr(pick#, wk1-4 ppg)
−0.41…−0.50 by position; EXPERIMENTS.md "Rookie draft-capital prior"). This module
fits, from historical rookies, a smooth per-position expectation of fantasy PPG as a
function of draft pick, plus a floor/ceiling band from the residual spread — a
calibrated *starting picture* for a rookie's first few weeks, to be superseded by real
usage as the season unfolds. It cannot foresee a 7th-round breakout or a 1st-round bust;
it gives the honest expected value given draft capital.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scoring import fanduel_points

SKILL = ["QB", "RB", "WR", "TE"]
_COMPONENTS = ["passing_yards", "passing_tds", "passing_interceptions", "rushing_yards",
               "rushing_tds", "receiving_yards", "receptions", "receiving_tds"]


def _fp_per_game(stats: pd.DataFrame) -> pd.DataFrame:
    """Attach FanDuel points to weekly player-stat rows."""
    stats = stats.copy()
    stats["fp"] = fanduel_points(
        passing_yards=stats.get("passing_yards", 0), passing_tds=stats.get("passing_tds", 0),
        interceptions=stats.get("passing_interceptions", 0),
        rushing_yards=stats.get("rushing_yards", 0), rushing_tds=stats.get("rushing_tds", 0),
        receptions=stats.get("receptions", 0), receiving_yards=stats.get("receiving_yards", 0),
        receiving_tds=stats.get("receiving_tds", 0))
    return stats


def _load_history(max_year: int, min_year: int = 2008):
    """Historical rookie player-weeks (draft season == stat season) with draft capital."""
    import nflreadpy as nfl
    years = list(range(min_year, max_year + 1))
    dp = nfl.load_draft_picks().to_pandas()
    dp = dp[dp["position"].isin(SKILL) & dp["gsis_id"].notna()][
        ["gsis_id", "season", "round", "pick", "position"]].rename(columns={"season": "draft_season"})
    st = pd.concat([nfl.load_player_stats(seasons=[y]).to_pandas() for y in years], ignore_index=True)
    st = _fp_per_game(st).rename(columns={"player_id": "gsis_id"})
    if "position" in st.columns:
        st = st.drop(columns=["position"])
    rk = st.merge(dp, left_on=["gsis_id", "season"], right_on=["gsis_id", "draft_season"])
    return rk[rk["week"] <= 18]


class RookiePrior:
    """Per-position pick→PPG curve (ppg ≈ a + b·log(pick)) with a residual band and
    typical rookie component mix. Fit on rookies drafted <= ``max_year``."""

    # Shrink the fitted pick-curve toward the positional rookie mean. The raw curve is
    # survivorship-inflated at the very top (busts get benched and don't accumulate), so
    # a pick-1 QB / pick-3 RB otherwise out-projects established veterans in wk1. Shrinking
    # regresses genuinely-uncertain rookies toward a realistic below-elite expectation.
    SHRINK = 0.6

    def __init__(self, coeffs: dict, resid: dict, comp_mix: dict, pos_mean: dict | None = None,
                 min_ppg=0.5, max_ppg=26.0):
        self.coeffs = coeffs        # position -> (a, b) for early-season ppg
        self.resid = resid          # position -> (q10, q90) residual offsets
        self.comp_mix = comp_mix    # position -> {component: per-point fraction}
        self.pos_mean = pos_mean or {}  # position -> mean rookie early ppg (shrink target)
        self.min_ppg, self.max_ppg = min_ppg, max_ppg

    @classmethod
    def fit(cls, max_year: int, min_year: int = 2008, early_weeks: int = 4) -> "RookiePrior":
        rk = _load_history(max_year, min_year)
        early = rk[rk["week"] <= early_weeks]
        coeffs, resid, comp_mix, pos_mean = {}, {}, {}, {}
        for pos in SKILL:
            g = (early[early["position"] == pos]
                 .groupby(["gsis_id", "pick"])
                 .agg(ppg=("fp", "mean"), **{c: (c, "mean") for c in _COMPONENTS})
                 .reset_index())
            g = g[g["pick"] >= 1]
            pos_mean[pos] = float(g["ppg"].mean()) if len(g) else 5.0
            if len(g) < 12:
                coeffs[pos], resid[pos] = (g["ppg"].mean() if len(g) else 5.0, 0.0), (-2.0, 2.0)
                comp_mix[pos] = {}
                continue
            x = np.log(g["pick"].values)
            b, a = np.polyfit(x, g["ppg"].values, 1)          # ppg = a + b*log(pick)
            pred = a + b * x
            r = g["ppg"].values - pred
            coeffs[pos] = (float(a), float(b))
            # wide residual band: rookie outcomes generalize worse across years than the
            # in-sample spread implies, so q10/q90 (nominal 80%) lands nearer a real ~60%.
            resid[pos] = (float(np.quantile(r, 0.1)), float(np.quantile(r, 0.9)))
            # component mix: mean per-game component per unit of ppg (for props)
            ppg_mean = max(g["ppg"].mean(), 1e-6)
            comp_mix[pos] = {c: float(g[c].mean() / ppg_mean) for c in _COMPONENTS}
        return cls(coeffs, resid, comp_mix, pos_mean=pos_mean)

    def project(self, position: str, pick: int | None) -> dict | None:
        """Projected rookie fantasy points (+ floor/ceiling + component estimates).
        `pick` None/0 (undrafted) is treated as pick 260 (past the last pick)."""
        if position not in self.coeffs:
            return None
        a, b = self.coeffs[position]
        p = float(pick) if pick and pick > 0 else 260.0
        ppg = a + b * np.log(p)
        # regress toward the positional rookie mean (survivorship-inflated top end)
        mean = self.pos_mean.get(position, ppg)
        ppg = mean + self.SHRINK * (ppg - mean)
        lo, hi = self.resid[position]
        clip = lambda v: float(min(self.max_ppg, max(self.min_ppg, v)))
        proj, floor, ceil = clip(ppg), clip(ppg + lo), clip(ppg + hi)
        out = {
            "fanduel_fantasy_points": round(proj, 1),
            "floor": round(floor, 1),
            "projection_median": round(proj, 1),
            "ceiling": round(ceil, 1),
            "prediction_type": "rookie_prior",
        }
        for c, frac in self.comp_mix.get(position, {}).items():
            out[c] = round(frac * proj, 2)
        return out
