"""Volume-share model: predict each player's share of his team's touches for a season.

Rationale (EXPERIMENTS.md): matchup *context* is noise for the per-game MEAN, but *volume*
is the backbone of fantasy scoring, and volume SHARE — carry share, target share — is far more
stable and predictable than per-game points. Coaching's prior-job tendencies (pass rate,
RB-committee concentration, TE usage, pace) were measured to carry over ~16%; they were a wash as
a per-game point nudge, but share ALLOCATION is the application never tested and the natural home
for that signal.

The model predicts, for the upcoming season, each player's share of team carries and team targets
from three signals, in priority order:

  1. depth rank        (from ESPN in production; usage-derived in backtests) -> base share by rank
  2. vacated share     (roster turnover)   -> handled implicitly: a new depth chart re-ranks the
                                              returners/newcomers and shares renormalize within the
                                              team, so a departed starter's touches flow to whoever
                                              now sits at that rank
  3. coaching tendency (nfl_projections.coaching) -> reshape the split: bell-cow vs committee,
                                              pass<->run pool, TE-featured, pace

Shares are FRACTIONS in [0, 1], matching NFL-API's opportunities.py (which reports the historical
actuals as percentages). This module is validated by backtest_shares() BEFORE it feeds projections
— predicted next-season share must beat naive baselines (repeat-last-year, positional-rank mean).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import roles

SKILL = ("QB", "RB", "WR", "TE")
_TEAM_COLS = ("team", "recent_team", "team_abbr", "club_code")
# positions that pool each resource (for team-level normalization)
_CARRY_POS = ("RB", "QB", "WR")
_TARGET_POS = ("WR", "RB", "TE")


def _team_col(df: pd.DataFrame) -> str:
    for c in _TEAM_COLS:
        if c in df.columns:
            return c
    raise KeyError(f"no team column (looked for {_TEAM_COLS})")


def _name_col(df: pd.DataFrame) -> str:
    for c in ("player_display_name", "player_name", "full_name"):
        if c in df.columns:
            return c
    raise KeyError("no player-name column")


def season_share_table(dataset: pd.DataFrame) -> pd.DataFrame:
    """Per (season, team, position, player) actual carry/target share of the team pool, the
    player's season fantasy PPG (the pre-season depth signal), and games played.

    carry_share = player carries / team carries; target_share = player targets / team targets
    (team totals over all skill positions), matching NFL-API opportunities.py."""
    df = dataset.copy()
    if "week" in df.columns:
        df = df[df["week"] != "AVG"]
    tcol = _team_col(df)
    ncol = _name_col(df)
    for c in ("carries", "targets", "fanduel_fantasy_points", "offensive_snap_pct"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["position"].isin(SKILL)]

    player = (df.groupby(["season", tcol, "position", "player_id", ncol], dropna=False)
                .agg(carries=("carries", "sum"), targets=("targets", "sum"),
                     fppg=("fanduel_fantasy_points", "mean"),
                     snap_share=("offensive_snap_pct", "mean"),
                     games=("fanduel_fantasy_points", "size"))
                .reset_index()
                .rename(columns={tcol: "team", ncol: "player_name"}))
    team = (df.groupby(["season", tcol]).agg(team_carries=("carries", "sum"),
                                             team_targets=("targets", "sum"))
              .reset_index().rename(columns={tcol: "team"}))
    out = player.merge(team, on=["season", "team"], how="left")
    out["carry_share"] = np.where(out["team_carries"] > 0, out["carries"] / out["team_carries"], 0.0)
    out["target_share"] = np.where(out["team_targets"] > 0, out["targets"] / out["team_targets"], 0.0)
    out["snap_share"] = out["snap_share"].fillna(0.0) / 100.0
    return out


def _usage_rank(share_table: pd.DataFrame) -> pd.Series:
    """Depth rank within each (season, team, position) by ACTUAL usage (snap share, then
    touches) — defines 'who was the starter' for fitting the share-by-rank prior."""
    st = share_table.copy()
    st["_usage"] = st["snap_share"] + 0.001 * (st["carries"] + st["targets"])
    return (st.groupby(["season", "team", "position"])["_usage"]
              .rank(ascending=False, method="first").astype(int))


def fit_share_prior(share_table: pd.DataFrame, *, max_season: int | None = None,
                    min_games: int = 4) -> pd.DataFrame:
    """Average carry/target/snap share by (position, depth_rank), learned from actual usage.
    Ranks past the explicit tiers collapse to a 'deep' bucket. Fit on seasons <= max_season
    (exclude the test season to keep the backtest out of sample)."""
    st = share_table.copy()
    if max_season is not None:
        st = st[st["season"] <= max_season]
    st = st[st["games"] >= min_games]
    st["rank"] = _usage_rank(st)
    st["rank_b"] = st["rank"].clip(upper=5)  # tier 5 = "deep"
    prior = (st.groupby(["position", "rank_b"])
               .agg(carry_share=("carry_share", "mean"),
                    target_share=("target_share", "mean"),
                    snap_share=("snap_share", "mean"),
                    n=("carry_share", "size"))
               .reset_index().rename(columns={"rank_b": "rank"}))
    return prior


def _prior_lookup(prior: pd.DataFrame) -> dict:
    return {(r["position"], int(r["rank"])): r for _, r in prior.iterrows()}


def predict_shares(roster: pd.DataFrame, prior: pd.DataFrame, *,
                   coaching_mult: dict | None = None, normalize: bool = False) -> pd.DataFrame:
    """Predict carry/target share for a roster with (team, position, depth_rank).

    Each player gets the prior share for his depth rank; a per-team ``coaching_mult`` may
    reshape it (e.g. steepen RB1 for a bell-cow coach). Vacated share flows naturally through
    the rank: when a starter leaves, whoever now sits at rank 1 gets the rank-1 prior.

    ``normalize`` renormalizes each team's pool to sum to 1. It was measured to HURT prediction
    accuracy (the raw rank prior is more robust; backtest_shares), so it defaults off — turn it
    on only when a hard finite-pool constraint is required (e.g. dividing a fixed team budget).
    """
    plut = _prior_lookup(prior)
    r = roster.copy()
    r["depth_rank"] = r["depth_rank"].map(roles.norm_rank)

    def base(row, col):
        rec = plut.get((row["position"], min(int(row["depth_rank"]), 5)))
        return float(rec[col]) if rec is not None else 0.0

    for col, pool_pos in (("carry_share", _CARRY_POS), ("target_share", _TARGET_POS)):
        raw = r.apply(lambda x: base(x, col) if x["position"] in pool_pos else 0.0, axis=1)
        if coaching_mult:
            raw = raw * r.apply(
                lambda x: _reshape(coaching_mult.get(x["team"]), x, col), axis=1)
        if normalize:
            tot = r.assign(_r=raw).groupby("team")["_r"].transform("sum")
            r[col] = np.where(tot > 0, raw / tot, 0.0)
        else:
            r[col] = raw
    return r


def prior_season_shares(share_table: pd.DataFrame, season: int) -> pd.DataFrame:
    """Each player's share in ``season`` on their primary (highest-usage) team — the
    own-history term of the hybrid. Indexed by player_id."""
    pv = share_table[share_table["season"] == season].copy()
    if pv.empty:
        return pv.set_index("player_id") if "player_id" in pv.columns else pv
    pv["_u"] = pv["carry_share"] + pv["target_share"]
    return (pv.sort_values("_u").drop_duplicates("player_id", keep="last")
              .set_index("player_id")[["team", "carry_share", "target_share"]])


def project_shares(roster: pd.DataFrame, dataset: pd.DataFrame, season: int, *,
                   blend: float = 0.6, coaching_mult: dict | None = None) -> pd.DataFrame:
    """Validated share model (backtest_shares): predict each rostered player's carry/target
    share for ``season``. For a player returning to the same team, blend his own prior-year
    share (weight ``blend``) with the depth-rank prior; for a moved/new player, use the rank
    prior alone (which beats repeat-last-year decisively on changed roles). ``roster`` needs
    team / position / player_id / depth_rank."""
    share_table = season_share_table(dataset)
    prior = fit_share_prior(share_table, max_season=season - 1)
    out = roster.drop_duplicates(["player_id", "team"]).copy()
    rank_pred = predict_shares(out, prior, coaching_mult=coaching_mult)
    rc_map = dict(zip(zip(rank_pred["player_id"], rank_pred["team"]), rank_pred["carry_share"]))
    rt_map = dict(zip(zip(rank_pred["player_id"], rank_pred["team"]), rank_pred["target_share"]))
    prev = prior_season_shares(share_table, season - 1)

    cs, ts = [], []
    for _, p in out.iterrows():
        pid, team = p["player_id"], p["team"]
        rc = float(rc_map.get((pid, team), 0.0))
        rt = float(rt_map.get((pid, team), 0.0))
        if pid in prev.index and prev.at[pid, "team"] == team:
            cs.append(blend * float(prev.at[pid, "carry_share"]) + (1 - blend) * rc)
            ts.append(blend * float(prev.at[pid, "target_share"]) + (1 - blend) * rt)
        else:
            cs.append(rc)
            ts.append(rt)
    out["carry_share"] = cs
    out["target_share"] = ts
    return out


def _reshape(mult: dict | None, row, col: str) -> float:
    """Coaching reshape factor for one player's raw share (1.0 if no coaching profile).
    Bell-cow concentration lifts RB1 carries and trims backups; pass tendency lifts the
    target pool for lead pass-catchers."""
    if not mult:
        return 1.0
    rank = int(row["depth_rank"])
    if col == "carry_share":
        conc = mult.get("rb_conc", 1.0)          # >1 = more bell-cow
        return conc if rank == 1 else (1.0 / max(conc, 1e-6))
    lead = mult.get("pass_spread", 1.0)          # >1 = concentrated to top target
    return lead if rank <= 2 else (1.0 / max(lead, 1e-6))


# --------------------------------------------------------------------------- backtest

def _entering_rank(share_table: pd.DataFrame) -> pd.DataFrame:
    """Assign each (season, team, position) player a pre-season depth rank from their PRIOR
    season's fppg (anywhere) — a leakage-free proxy for a preseason depth chart: teams slot by
    recent production; movers carry their number; players with no prior year rank last."""
    st = share_table.sort_values(["player_id", "season"]).copy()
    st["prior_fppg"] = st.groupby("player_id")["fppg"].shift(1)
    st["_key"] = st["prior_fppg"].fillna(-1.0)
    st["entering_rank"] = (st.groupby(["season", "team", "position"])["_key"]
                             .rank(ascending=False, method="first").astype(int))
    return st


def backtest_shares(dataset: pd.DataFrame, *, test_seasons=None, min_games: int = 4,
                    blend: float = 0.6) -> dict:
    """Validate the share model out of sample: predict each player's actual season share from
    his ENTERING depth rank (prior-year production), fitting the prior only on earlier seasons,
    and compare to naive baselines. ``model`` is the shipped hybrid (own prior-year share blended
    with the rank prior for returners; rank prior for movers). ``naiveA`` = repeat last year,
    ``naiveB`` = rank-mean prior alone. Reports MAE (carry+target share) overall and split by
    whether the role CHANGED — the vacated-share cases the model should win."""
    st = _entering_rank(season_share_table(dataset))
    st = st[st["games"] >= min_games]
    seasons = sorted(st["season"].unique())
    test_seasons = test_seasons or seasons[3:]  # need a few years of history to fit

    rows = []
    for Y in test_seasons:
        if Y not in seasons:
            continue
        prior = fit_share_prior(st, max_season=Y - 1, min_games=min_games)
        plut = _prior_lookup(prior)
        cur = st[st["season"] == Y].drop_duplicates(["player_id", "team"]).copy()
        # prior year: one row per player (their primary/highest-usage team)
        pv = st[st["season"] == Y - 1].copy()
        pv["_u"] = pv["carry_share"] + pv["target_share"]
        prev = (pv.sort_values("_u").drop_duplicates("player_id", keep="last")
                  .set_index("player_id")[["team", "carry_share", "target_share"]])

        # model prediction: prior share by entering rank, renormalized within team pool
        roster = cur.rename(columns={"entering_rank": "depth_rank"})
        pred = predict_shares(roster[["team", "position", "player_id", "depth_rank"]], prior)
        pred = pred.set_index(["player_id", "team"])[["carry_share", "target_share"]]

        for _, p in cur.iterrows():
            pid, team = p["player_id"], p["team"]
            actual_c, actual_t = p["carry_share"], p["target_share"]
            # naive A: repeat prior-year share IF same team, else 0 (moved -> lost role)
            same_team = pid in prev.index and prev.at[pid, "team"] == team
            naiveA_c = float(prev.at[pid, "carry_share"]) if same_team else 0.0
            naiveA_t = float(prev.at[pid, "target_share"]) if same_team else 0.0
            # naive B: positional-rank mean = the raw (pre-normalization) prior for the rank
            rec = plut.get((p["position"], min(int(p["entering_rank"]), 5)))
            naiveB_c = float(rec["carry_share"]) if rec is not None else 0.0
            naiveB_t = float(rec["target_share"]) if rec is not None else 0.0
            rank_c, rank_t = (float(pred.at[(pid, team), "carry_share"]),
                              float(pred.at[(pid, team), "target_share"])) \
                if (pid, team) in pred.index else (0.0, 0.0)
            # shipped model: blend own prior-year share for returners, rank prior for movers
            if same_team:
                mc = blend * naiveA_c + (1 - blend) * rank_c
                mt = blend * naiveA_t + (1 - blend) * rank_t
            else:
                mc, mt = rank_c, rank_t
            rows.append({
                # "changed" = not on this team last year (moved / rookie / returning to a new role)
                "season": Y, "position": p["position"], "changed": bool(not same_team),
                "err_model": abs(mc - actual_c) + abs(mt - actual_t),
                "err_naiveA": abs(naiveA_c - actual_c) + abs(naiveA_t - actual_t),
                "err_naiveB": abs(naiveB_c - actual_c) + abs(naiveB_t - actual_t),
            })
    res = pd.DataFrame(rows)
    if res.empty:
        return {"n": 0}
    def mae(sub):
        return {k: round(sub[f"err_{k}"].mean(), 4) for k in ("model", "naiveA", "naiveB")}
    summary = {"n": len(res), "overall": mae(res),
               "stable": mae(res[~res["changed"]]),
               "changed": mae(res[res["changed"]]),
               "by_position": {pos: mae(res[res["position"] == pos]) for pos in SKILL}}
    return summary
