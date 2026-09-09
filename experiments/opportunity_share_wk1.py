"""Week-1 2025 backtest: does opportunity redistribution beat trailing rate?

The protocol is the owner's: train on nothing after 2024, project week 1 of
2025, score against what happened. Week 1 is the hard case on purpose — there
is no current-season usage to lean on, so a model that only knows "what have
you done" is at its weakest, and a role change from the offseason is invisible
to it.

Three projections per player, all built from 2024 and earlier only:

  trailing   the player's own 2024 per-game rate. This is what the shipped
             model effectively reduces to for week 1 — it is the quantity the
             engine's blend is anchored on.
  team_share depth rank -> share of the team's carries -> times the player's
             own efficiency. The redistribution model.
  blend      the average of the two, as a cheap hedge.

Scored on rushing yards for RBs, which is where the Jacksonville case lives.

RESULT (2024 -> week 1 2025, n=73 RBs with a prior season):

    model          MAE     bias    corr
    trailing     23.03   +11.48   0.588
    team_share   19.29    +1.99   0.568
    blend        19.96    +6.74   0.606

The MAE gain is real but the bias is the finding. Trailing over-projects by
11.5 yards a game because it hands every back last year's role and cannot see
the carries that left the building; share-based opportunity is close to
unbiased. For a yardage line that difference is the whole game — a model
running 11 yards high takes overs it should not.

What it does NOT do is pick winners better: correlation is flat to slightly
worse, and among RB1s alone both models correlate poorly (0.27 / 0.18) because
week-1 rushing is genuinely noisy. Read this as "better calibrated", not
"better informed".

Run it:  python experiments/opportunity_share_wk1.py
"""
import pathlib, re
import numpy as np
import pandas as pd

D = pathlib.Path(__file__).parent / "_cache"
D.mkdir(exist_ok=True)
CACHE = pathlib.Path.home() / ".sharp-edge" / "nfl"
WEEKLY = ("https://github.com/nflverse/nflverse-data/releases/download/"
          "stats_player/stats_player_week_{s}.parquet")
DEPTH = ("https://github.com/nflverse/nflverse-data/releases/download/"
         "depth_charts/depth_charts_{s}.parquet")


def weekly(seasons):
    out = []
    for s in seasons:
        p = CACHE / f"week_{s}.parquet"
        out.append(pd.read_parquet(p) if p.exists() else pd.read_parquet(WEEKLY.format(s=s)))
    return pd.concat(out, ignore_index=True)


def week1_depth_ranks(season):
    """RB depth rank per team as of the start of ``season``.

    Taken from the earliest chart published in the season, so it reflects what
    was knowable in week 1 rather than how the year turned out.
    """
    p = D / f"depth_{season}.parquet"
    d = pd.read_parquet(p) if p.exists() else pd.read_parquet(DEPTH.format(s=season))
    if not p.exists():
        d.to_parquet(p)
    rb = d[d.pos_abb.isin(["RB", "HB", "TB"])].copy()
    rb["dt"] = pd.to_datetime(rb.dt, errors="coerce", utc=True)
    first = rb.dt.min()
    rb = rb[rb.dt <= first + pd.Timedelta(days=30)]
    rb = rb.sort_values("dt").drop_duplicates(["team", "gsis_id"], keep="first")
    rb["rank"] = rb.groupby("team").pos_rank.rank(method="first")
    return rb[["team", "gsis_id", "player_name", "rank"]]


def share_curve(hist, through):
    """Median share of team carries by backfield rank, fit on ``through`` and
    earlier. Ranked by carries within a team-season, which is the observable
    proxy for a depth chart in the historical record."""
    h = hist[(hist.season <= through) & (hist.season >= through - 4)
             & (hist.season_type == "REG")]
    rb = h[h.position == "RB"].groupby(["season", "team", "player_id"]).carries.sum().reset_index()
    rb = rb[rb.carries > 0]
    team = h.groupby(["season", "team"]).carries.sum().rename("team_car").reset_index()
    rb = rb.merge(team, on=["season", "team"])
    rb["rank"] = rb.groupby(["season", "team"]).carries.rank(ascending=False, method="first")
    rb["share"] = rb.carries / rb.team_car
    return rb.groupby("rank").share.median().to_dict()


def build(through=2024, target=2025):
    hist = weekly(range(2018, target + 1))
    hist["carries"] = hist.carries.fillna(0)
    hist["rushing_yards"] = hist.rushing_yards.fillna(0)

    prior = hist[(hist.season == through) & (hist.season_type == "REG")]
    curve = share_curve(hist, through)

    # --- what each player did in the prior season ---
    p = prior[prior.position == "RB"].groupby(["player_id", "player_display_name"]).agg(
        g=("week", "count"), car=("carries", "sum"), yds=("rushing_yards", "sum"),
    ).reset_index()
    p["trailing"] = p.yds / p.g
    p["ypc"] = np.where(p.car > 0, p.yds / p.car, np.nan)

    # Efficiency is regressed toward the league mean by carry count: 83 carries
    # of 3.70 is not a 3.70 back, it is a small sample around the mean.
    LG = (prior.rushing_yards.sum() / prior.carries.sum())
    K = 100.0
    p["ypc_reg"] = (p.ypc.fillna(LG) * p.car + LG * K) / (p.car + K)

    # --- team volume, and the new depth chart ---
    team_car = (prior.groupby("team").carries.sum() / 17.0).rename("team_car_pg")
    ranks = week1_depth_ranks(target)

    actual = hist[(hist.season == target) & (hist.week == 1)
                  & (hist.season_type == "REG") & (hist.position == "RB")]
    actual = actual.groupby(["player_id", "player_display_name", "team"]).agg(
        actual=("rushing_yards", "sum"), car=("carries", "sum")).reset_index()

    df = actual.merge(p[["player_id", "trailing", "ypc_reg", "car"]].rename(
        columns={"car": "prior_car"}), on="player_id", how="left")
    df = df.merge(ranks.rename(columns={"gsis_id": "player_id"})[["player_id", "rank"]],
                  on="player_id", how="left")
    df = df.merge(team_car, left_on="team", right_index=True, how="left")

    df["share"] = df["rank"].map(curve)
    df["team_share"] = df.share * df.team_car_pg * df.ypc_reg
    df["blend"] = df[["trailing", "team_share"]].mean(axis=1)
    return df, curve, LG


def score(df, cols, label):
    d = df.dropna(subset=cols + ["actual"])
    print(f"\n{label}  (n={len(d)})")
    print(f"{'model':<12}{'MAE':>8}{'bias':>8}{'corr':>8}")
    for c in cols:
        err = d[c] - d.actual
        print(f"{c:<12}{err.abs().mean():8.2f}{err.mean():+8.2f}"
              f"{np.corrcoef(d[c], d.actual)[0,1]:8.3f}")
    return d


if __name__ == "__main__":
    df, curve, LG = build()
    print("share curve fit on 2020-24:",
          {int(k): round(v, 3) for k, v in sorted(curve.items())[:4]})
    print(f"league ypc {LG:.2f}")

    d = score(df, ["trailing", "team_share", "blend"], "Week 1 2025, all RBs with a prior season")

    # The case that started this: a player whose depth rank moved.
    lead = d[d["rank"] == 1]
    score(lead, ["trailing", "team_share", "blend"], "RB1s only (the promoted/lead backs)")

    print("\nbiggest trailing-model misses among RB1s:")
    lead = lead.assign(miss=(lead.trailing - lead.actual).abs()).sort_values("miss", ascending=False)
    print(f"{'player':24s}{'actual':>8}{'trailing':>10}{'share':>8}{'prior car':>10}")
    for _, r in lead.head(10).iterrows():
        print(f"{r.player_display_name[:23]:24s}{r.actual:8.0f}{r.trailing:10.1f}"
              f"{r.team_share:8.1f}{r.prior_car:10.0f}")
    df.to_parquet(D / "wk1_2025_backtest.parquet")
