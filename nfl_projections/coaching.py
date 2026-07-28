"""Coach-scheme week-1 prior.

The rolling last-3/last-5 features (features.py) are the accuracy backbone, but they
are useless for *week 1 of a new coaching regime*: a player's trailing form still
reflects the OLD system. Measured foundation (EXPERIMENTS.md, "Coaching"): a new
coach's PRIOR-job tendencies predict his new team's run/pass balance, TE usage,
RB-committee split, and pace better than the outgoing coach's system (~16% closer),
but QB rush share does NOT carry over (it's personnel-driven). So this module nudges a
new-coach team's WEEK-1 (decaying through ~wk2) volume components toward the coach's
prior-job scheme — pass/rush volume, TE target share, RB-committee concentration —
and deliberately leaves QB rushing alone.

Design decisions:
- Adjust VOLUME components only; add just the FanDuel-points DELTA to the model's
  direct-FP headline (using the delta sidesteps the component->FP level bias, #10).
- Apply only to teams with a coaching change, only early season, with linear decay.
- No coach play-calling history (e.g. a defensive HC, or a first-time coordinator) ->
  no scheme nudge (multipliers = 1); the scoring-level fallback (Vegas implied total)
  is left as a separate, optional hook.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import scoring

# ── curated new-coach play-calling history ────────────────────────────────────
# (team, season) pairs where the coach was the primary OFFENSIVE play-caller.
# Only ~5-8 new HCs per year, so a small curated map is accurate and cheap; extend
# each offseason. Defensive HCs / no play-calling history -> omit (no scheme prior).
COACH_PLAYCALL_HISTORY: dict[str, list[tuple[str, int]]] = {
    "Ben Johnson":          [("DET", 2022), ("DET", 2023), ("DET", 2024)],
    "Brian Schottenheimer": [("DAL", 2023), ("DAL", 2024)],
    "Liam Coen":            [("TB", 2024), ("LA", 2022)],
    "Kellen Moore":         [("PHI", 2024), ("LAC", 2023), ("DAL", 2021), ("DAL", 2022)],
    "Pete Carroll":         [("SEA", 2021), ("SEA", 2022), ("SEA", 2023)],
    "Mike Vrabel":          [("TEN", 2021), ("TEN", 2022), ("TEN", 2023)],
    # Aaron Glenn (NYJ) is defensive -> intentionally omitted (no offensive prior).
}

METRICS = ["pass_rate", "top_rb_share", "te_tgt_share", "plays_pg"]  # qb_rush_share excluded on purpose

# how far a coach-team-season is discounted per year of age when averaging a profile
_RECENCY_HALF_LIFE = 3.0


def _neutral(df: pd.DataFrame) -> pd.DataFrame:
    """Neutral game script: downs 1-3, quarters 1-3, win prob 20-80% (strips score-dependence)."""
    return df[df["down"].isin([1, 2, 3]) & df["qtr"].isin([1, 2, 3]) & df["wp"].between(0.2, 0.8)]


def team_tendencies(pbp: pd.DataFrame, team: str, season: int, pos_map: dict) -> dict | None:
    """Neutral-situation offensive tendencies for one team-season, or None if no data."""
    d = pbp[(pbp["posteam"] == team) & (pbp["season"] == season)]
    if d.empty:
        return None
    ngames = d["game_id"].nunique()
    plays = _neutral(d)
    plays = plays[plays["play_type"].isin(["pass", "run"])]
    npass = int((plays["play_type"] == "pass").sum())
    nrush = int((plays["play_type"] == "run").sum())
    if npass + nrush == 0:
        return None
    rush = d[d["play_type"] == "run"].copy()
    rush["rpos"] = rush["rusher_player_id"].map(pos_map)
    rb = rush[rush["rpos"] == "RB"]
    top_rb_share = (rb.groupby("rusher_player_id").size().max() / len(rb)) if len(rb) else np.nan
    tgt = d[d["receiver_player_id"].notna()].copy()
    tgt["tpos"] = tgt["receiver_player_id"].map(pos_map)
    te_tgt_share = (tgt["tpos"] == "TE").mean() if len(tgt) else np.nan
    return {
        "pass_rate": npass / (npass + nrush),
        "top_rb_share": top_rb_share,
        "te_tgt_share": te_tgt_share,
        "plays_pg": len(plays) / ngames if ngames else np.nan,
    }


def league_baseline(pbp: pd.DataFrame, season: int, pos_map: dict) -> dict:
    """Average tendencies across all teams in a season (the 'no information' fallback)."""
    teams = pbp[pbp["season"] == season]["posteam"].dropna().unique()
    vals = [team_tendencies(pbp, t, season, pos_map) for t in teams]
    vals = [v for v in vals if v]
    return {m: float(np.nanmean([v[m] for v in vals])) for m in METRICS}


def coach_profile(coach: str, pbp: pd.DataFrame, pos_map: dict) -> dict | None:
    """Recency-weighted average of a coach's prior-job tendencies, or None if no history."""
    hist = COACH_PLAYCALL_HISTORY.get(coach)
    if not hist:
        return None
    latest = max(s for _, s in hist)
    num = {m: 0.0 for m in METRICS}
    wsum = {m: 0.0 for m in METRICS}
    for team, season in hist:
        t = team_tendencies(pbp, team, season, pos_map)
        if not t:
            continue
        w = 0.5 ** ((latest - season) / _RECENCY_HALF_LIFE)
        for m in METRICS:
            if not np.isnan(t[m]):
                num[m] += w * t[m]
                wsum[m] += w
    prof = {m: (num[m] / wsum[m] if wsum[m] else np.nan) for m in METRICS}
    return prof if any(not np.isnan(v) for v in prof.values()) else None


def _clip(x, lo=0.6, hi=1.6):
    return float(min(hi, max(lo, x)))


def position_multipliers(profile: dict, baseline: dict) -> dict:
    """Per-position volume multipliers from a coach profile vs league baseline.

    Returns factors applied to the model's predicted VOLUME components. Pace scales
    everyone; pass/rush balance splits pass-catchers vs runners; TE-share and
    RB-committee are position-specific overlays. All clipped to a sane band.
    """
    pace = profile["plays_pg"] / baseline["plays_pg"]
    pass_bal = profile["pass_rate"] / baseline["pass_rate"]
    rush_bal = (1 - profile["pass_rate"]) / (1 - baseline["pass_rate"])
    te_over = profile["te_tgt_share"] / baseline["te_tgt_share"]
    rb_conc = profile["top_rb_share"] / baseline["top_rb_share"]  # >1 = more bell-cow

    pass_vol = _clip(pace * pass_bal)   # passing_yards/tds, WR/RB receiving
    rush_vol = _clip(pace * rush_bal)   # rushing_yards/tds
    return {
        "pass_vol": pass_vol,
        "rush_vol": rush_vol,
        "te_recv": _clip(pass_vol * te_over),   # TE receiving gets the TE-share overlay
        "rb_lead_rush": _clip(rush_vol * rb_conc),      # projected lead back
        "rb_other_rush": _clip(rush_vol / max(rb_conc, 1e-6)),  # secondary backs
    }


# volume components scaled per position (QB rushing intentionally absent)
_SCALE = {
    "QB": {"passing_yards": "pass_vol", "passing_tds": "pass_vol"},
    "WR": {"receiving_yards": "pass_vol", "receptions": "pass_vol", "receiving_tds": "pass_vol"},
    "TE": {"receiving_yards": "te_recv", "receptions": "te_recv", "receiving_tds": "te_recv"},
    "RB": {"rushing_yards": None, "rushing_tds": None,  # filled with rb_lead/other at call time
           "receiving_yards": "pass_vol", "receptions": "pass_vol"},
}


# volume components that carry FanDuel points and that we scale (model TARGET_COLS names)
_FP_COMPONENTS = ["passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
                  "receiving_yards", "receptions", "receiving_tds"]


def _fp(comp: dict) -> float:
    """FanDuel points from a component dict (only the volume components we adjust;
    interceptions unchanged so they cancel out of the delta)."""
    return scoring.fanduel_points(
        passing_yards=comp.get("passing_yards", 0), passing_tds=comp.get("passing_tds", 0),
        rushing_yards=comp.get("rushing_yards", 0), rushing_tds=comp.get("rushing_tds", 0),
        receptions=comp.get("receptions", 0), receiving_yards=comp.get("receiving_yards", 0),
        receiving_tds=comp.get("receiving_tds", 0))


def early_season_weight(week: int) -> float:
    """Full nudge in wk1, half in wk2, gone by wk3 (rolling features have taken over)."""
    return {1: 1.0, 2: 0.5}.get(int(week), 0.0)


def apply_prior(result: dict, position: str, week: int, mult: dict,
                is_lead_rb: bool = False) -> dict:
    """Adjust one player's projection in place-ish: scale volume components, then add
    only the FanDuel-points DELTA (decayed by week) to the direct-FP headline.

    `result` must contain the model's predicted TARGET_COLS component values.
    Returns result with an updated ``fanduel_fantasy_points`` and a
    ``coach_prior_delta`` diagnostic.
    """
    w = early_season_weight(week)
    result["coach_prior_delta"] = 0.0
    if w == 0 or position not in _SCALE:
        return result

    scale_map = dict(_SCALE[position])
    if position == "RB":
        rb_key = "rb_lead_rush" if is_lead_rb else "rb_other_rush"
        scale_map["rushing_yards"] = rb_key
        scale_map["rushing_tds"] = rb_key

    before = {c: float(result.get(c, 0.0) or 0.0) for c in _FP_COMPONENTS}
    after = dict(before)
    for comp, factor_key in scale_map.items():
        if factor_key and comp in after:
            after[comp] = before[comp] * mult[factor_key]

    delta = _fp(after) - _fp(before)
    adj = round(w * delta, 2)
    result["coach_prior_delta"] = adj
    result["fanduel_fantasy_points"] = round(
        float(result.get("fanduel_fantasy_points", 0.0)) + adj, 1)
    return result
