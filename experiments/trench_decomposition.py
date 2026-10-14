"""Split a back's yards into the line's and his own, and see if it predicts.

"Usage decomposition beats trailing rate" (EXPERIMENTS.md) won by taking a rate
apart into the pieces that persist at different speeds. This is the same move
applied to yards per carry, which is the one remaining rate the board leans on
and the thing line-play charting is really about: a back behind a good line and
a back creating yards himself have the same YPC and should not have the same
projection.

Every carry splits exactly four ways, FO's bands with no overlap:

    negative   min(y, 0)          stopped behind the line
    line       clip(y, 0, 4)      the first four yards, blocked for him
    second     clip(y, 4, 10) - 4 the second level, shared
    open       max(y - 10, 0)     the open field, his

Their means sum to his yards per carry exactly, so the baseline and the
treatment see the same information arranged differently — which is what makes
this a test of decomposition rather than of extra data.

Leave-one-season-out, predicting next season's yards per carry, same as the
usage-decomposition and FTN tests so the numbers compare.
"""
import numpy as np
import pandas as pd

from trench_stability import SEASONS, band
from trench_value import carries

MIN_CARRIES = 80
PARTS = ["negative", "line", "second", "open"]


def per_season(plays):
    y = plays.yards_gained
    plays = plays.assign(
        negative=y.clip(upper=0), line=y.clip(lower=0, upper=4),
        second=y.clip(lower=4, upper=10) - 4, open=(y - 10).clip(lower=0))
    agg = {p: (p, "mean") for p in PARTS}
    backs = (plays.groupby(["season", "rusher_player_id", "rusher_player_name"])
             .agg(carries=("yards_gained", "size"), ypc=("yards_gained", "mean"),
                  **agg).reset_index())
    team = (plays.groupby(["season", "posteam"])
            .agg(team_aly=("aly", "mean"), team_carries=("aly", "size")).reset_index())
    # A back's team is where most of his carries came from that season.
    main = (plays.groupby(["season", "rusher_player_id", "posteam"]).size()
            .rename("n").reset_index().sort_values("n")
            .groupby(["season", "rusher_player_id"]).tail(1))
    backs = backs.merge(main[["season", "rusher_player_id", "posteam"]],
                        on=["season", "rusher_player_id"]).merge(
        team, on=["season", "posteam"])
    # The same team line, with this back's own carries taken out of it.
    own = (plays.groupby(["season", "posteam", "rusher_player_id"])
           .agg(own_aly=("aly", "sum"), own_n=("aly", "size")).reset_index())
    tot = (plays.groupby(["season", "posteam"])
           .agg(all_aly=("aly", "sum"), all_n=("aly", "size")).reset_index())
    x = own.merge(tot, on=["season", "posteam"])
    x["team_aly_xback"] = ((x.all_aly - x.own_aly) / (x.all_n - x.own_n)).where(
        x.all_n > x.own_n)
    backs = backs.merge(x[["season", "posteam", "rusher_player_id",
                           "team_aly_xback"]],
                        on=["season", "posteam", "rusher_player_id"], how="left")
    return backs[(backs.carries >= MIN_CARRIES) & backs.team_aly_xback.notna()], team


def pairs(backs, team_aly):
    """One row per back with a qualifying season N and N+1.

    Three versions of "the line he will run behind", because the difference
    between them is the whole question:

        next_team_aly        that team's realised line yards in N+1. Contains
                             the back's own carries, so it partly measures the
                             thing being predicted. Diagnostic only.
        next_team_aly_xback  the same, with his own carries removed. Still from
                             the future, so still not usable in a projection,
                             but it is no longer circular.
        next_team_aly_prior  that team's line yards in season N — known before
                             a ball is kicked, which is what a board can use.
    """
    nxt = backs.assign(season=backs.season - 1)[
        ["season", "rusher_player_id", "ypc", "carries", "posteam",
         "team_aly", "team_aly_xback"]].rename(
        columns={"ypc": "next_ypc", "carries": "next_carries",
                 "posteam": "next_team", "team_aly": "next_team_aly",
                 "team_aly_xback": "next_team_aly_xback"})
    m = backs.merge(nxt, on=["season", "rusher_player_id"])
    prior = team_aly.rename(columns={"posteam": "next_team",
                                     "team_aly": "next_team_aly_prior"})
    m = m.merge(prior[["season", "next_team", "next_team_aly_prior"]],
                on=["season", "next_team"])
    return m[m.next_carries >= MIN_CARRIES]


def loso(m, features, target="next_ypc"):
    """Leave-one-season-out OLS. Returns (MAE, corr)."""
    preds = np.full(len(m), np.nan)
    for season in sorted(m.season.unique()):
        train, test = m.season != season, m.season == season
        X = np.column_stack([np.ones(train.sum())] + [m.loc[train, f] for f in features])
        beta, *_ = np.linalg.lstsq(X, m.loc[train, target], rcond=None)
        Xt = np.column_stack([np.ones(test.sum())] + [m.loc[test, f] for f in features])
        preds[test.to_numpy()] = Xt @ beta
    return (float(np.abs(preds - m[target]).mean()),
            float(np.corrcoef(preds, m[target])[0, 1]))


if __name__ == "__main__":
    plays = pd.concat([carries(s) for s in SEASONS], ignore_index=True)
    plays["aly"] = plays.yards_gained.map(band)
    backs, team_aly = per_season(plays)
    m = pairs(backs, team_aly).reset_index(drop=True)
    print(f"{len(m)} back-season pairs with {MIN_CARRIES}+ carries in both, "
          f"{m.rusher_player_name.nunique()} backs\n")

    flat = float(np.abs(m.next_ypc - m.next_ypc.mean()).mean())
    tests = [
        ("league mean YPC (no skill)", None),
        ("own YPC", ["ypc"]),
        ("own YPC + carries", ["ypc", "carries"]),
        ("decomposed (4 bands)", PARTS),
        ("decomposed + own team's line", PARTS + ["team_aly"]),
        ("+ next team's PRIOR-year line", PARTS + ["next_team_aly_prior"]),
        ("own YPC + next team's PRIOR line", ["ypc", "next_team_aly_prior"]),
        ("next team's PRIOR line alone", ["next_team_aly_prior"]),
        ("[leaky] + next team's realised line", PARTS + ["next_team_aly"]),
        ("[semi-leaky] realised, back removed", PARTS + ["next_team_aly_xback"]),
    ]
    print(f"{'predictor':34s}{'MAE':>8}{'corr':>8}")
    print(f"{'league mean YPC (no skill)':34s}{flat:8.3f}{0.0:8.3f}")
    for label, features in tests[1:]:
        mae, corr = loso(m, features)
        print(f"{label:34s}{mae:8.3f}{corr:8.3f}")

    print("\nStability of each piece, season N -> N+1:")
    nxt = backs.assign(season=backs.season - 1)[
        ["season", "rusher_player_id"] + PARTS + ["ypc"]].rename(
        columns={p: f"n_{p}" for p in PARTS + ["ypc"]})
    s = backs.merge(nxt, on=["season", "rusher_player_id"])
    for p in PARTS + ["ypc"]:
        print(f"  {p:10s}{np.corrcoef(s[p], s[f'n_{p}'])[0, 1]:7.3f}")
