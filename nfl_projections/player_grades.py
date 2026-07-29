"""Player grades (0-100, 50 = positional average).

A per-player quality grade in the same spirit as the team grades (ratings.py), built from
what actually predicts future fantasy production. Measured foundation (EXPERIMENTS.md
"player grade"): OPPORTUNITY metrics (WOPR, target share, carries — who's on the field and
getting the ball) are both the most STABLE and the most PREDICTIVE signals — nearly as
predictive of rest-of-season fantasy PPG as past production itself, and stickier;
EFFICIENCY (EPA/play, RACR, YAC) is noisy for skill positions. So the grade is
opportunity- and production-weighted, efficiency-light.

Like the team grades: a preseason prior (last season's grade regressed toward positional
average) is blended with in-season results, the prior fading as games accrue.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scoring import fanduel_points

POSITIONS = ["QB", "RB", "WR", "TE"]
_YOY_CARRYOVER = 0.60
_PRIOR_GAMES = 5.0
_GRADE_SCALE = 15.0
_MIN_GAMES = 2

# per-position component metrics; weights ≈ opportunity 0.45 / production 0.45 / efficiency 0.10
# (from measured predicts-rest-FP). Opportunity metric(s) differ by position.
_SPEC = {
    "QB": {"opportunity": ["attempts", "carries"], "efficiency": ["epa_per_play"]},
    "RB": {"opportunity": ["carries", "wopr"], "efficiency": ["yards_per_opp"]},
    "WR": {"opportunity": ["wopr"], "efficiency": ["epa_per_play"]},
    "TE": {"opportunity": ["wopr"], "efficiency": ["epa_per_play"]},
}
_WEIGHTS = {"opportunity": 0.45, "production": 0.45, "efficiency": 0.10}


def player_metrics(stats: pd.DataFrame, through_week: int | None = None) -> pd.DataFrame:
    """Per-player per-game averages of the grade inputs (fp + opportunity + efficiency)."""
    st = stats[stats["position"].isin(POSITIONS)].copy()
    if through_week is not None:
        st = st[st["week"] <= through_week]
    st["fp"] = fanduel_points(
        passing_yards=st.get("passing_yards", 0), passing_tds=st.get("passing_tds", 0),
        interceptions=st.get("passing_interceptions", 0),
        rushing_yards=st.get("rushing_yards", 0), rushing_tds=st.get("rushing_tds", 0),
        receptions=st.get("receptions", 0), receiving_yards=st.get("receiving_yards", 0),
        receiving_tds=st.get("receiving_tds", 0))
    opp_ct = (st.get("targets", 0).fillna(0) + st.get("carries", 0).fillna(0)
              + st.get("attempts", 0).fillna(0)).replace(0, np.nan)
    epa = (st.get("receiving_epa", 0).fillna(0) + st.get("rushing_epa", 0).fillna(0)
           + st.get("passing_epa", 0).fillna(0))
    yards = (st.get("receiving_yards", 0).fillna(0) + st.get("rushing_yards", 0).fillna(0))
    st["epa_per_play"] = epa / opp_ct
    st["yards_per_opp"] = yards / opp_ct
    cols = ["fp", "wopr", "target_share", "carries", "attempts", "epa_per_play", "yards_per_opp"]
    cols = [c for c in cols if c in st.columns]
    name_col = "player_display_name" if "player_display_name" in st.columns else "player_name"
    g = (st.groupby(["player_id", "position"])
         .agg(games=("week", "count"), player_name=(name_col, "first"),
              **{c: (c, "mean") for c in cols})
         .reset_index())
    return g[g["games"] >= _MIN_GAMES]


def _z(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0.0


def _component_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    """Composite 0-100 grade per player, position-relative (z-scored within position)."""
    out = []
    for pos, grp in metrics.groupby("position"):
        spec = _SPEC.get(pos)
        if spec is None or len(grp) < 3:
            continue
        g = grp.copy()

        def fam_z(cols):
            cols = [c for c in cols if c in g.columns and g[c].notna().any()]
            if not cols:
                return pd.Series(0.0, index=g.index)
            return pd.concat([_z(g[c].fillna(g[c].mean())) for c in cols], axis=1).mean(axis=1)

        opp = fam_z(spec["opportunity"])
        prod = fam_z(["fp"])
        eff = fam_z(spec["efficiency"])
        composite = (_WEIGHTS["opportunity"] * opp + _WEIGHTS["production"] * prod
                     + _WEIGHTS["efficiency"] * eff)
        g["grade"] = (50 + _GRADE_SCALE * composite).clip(0, 100).round(1)
        g["opportunity_grade"] = (50 + _GRADE_SCALE * opp).clip(0, 100).round(1)
        g["efficiency_grade"] = (50 + _GRADE_SCALE * eff).clip(0, 100).round(1)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def grades(stats: pd.DataFrame, prior_stats: pd.DataFrame | None = None,
           through_week: int | None = None) -> pd.DataFrame:
    """Player grades for a season, blending a preseason prior (last season's grade
    regressed toward positional average, 50) with in-season results — the prior worth
    ~`_PRIOR_GAMES` games and fading as real games accrue.

    `stats`: current season weekly player-stats; `prior_stats`: last season's (optional)."""
    cur = _component_scores(player_metrics(stats, through_week))
    if cur.empty:
        return cur
    prior_grade = {}
    if prior_stats is not None:
        pri = _component_scores(player_metrics(prior_stats))
        # regress last year's grade toward 50 (positional average)
        prior_grade = {pid: 50 + _YOY_CARRYOVER * (gr - 50)
                       for pid, gr in zip(pri["player_id"], pri["grade"])}
    games = dict(zip(cur["player_id"], cur["games"]))
    blended = []
    for _, r in cur.iterrows():
        n = games.get(r["player_id"], 0)
        pg = prior_grade.get(r["player_id"], 50.0)  # unknown players start average
        w = _PRIOR_GAMES
        blended.append(round((w * pg + n * r["grade"]) / (w + n), 1))
    cur["grade"] = blended
    return cur.sort_values("grade", ascending=False).reset_index(drop=True)
