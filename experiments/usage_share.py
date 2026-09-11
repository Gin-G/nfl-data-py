"""Usage and role: does opportunity share beat trailing rate for receivers too?

The rushing version already works — depth rank to share of team carries, times
the player's own efficiency, cut week-1 MAE from 23.03 to 19.29 and bias from
+11.48 to +1.99. Receiving is most of the board, so the question is whether the
same shape holds when the opportunity being divided is targets rather than
carries.

Same protocol throughout: nothing after `through`, project week 1 of `target`,
score against actuals.

Three projections per player:

  trailing    the player's own prior-season yards per game. What the shipped
              model effectively reduces to at week 1.
  share       depth rank -> share of the team's targets -> times the player's
              own yards per target, regressed toward the positional mean by
              sample size.
  blend       the mean of the two.

Also tested: whether the share should come from the DEPTH CHART (what the team
says) or from LAST SEASON'S OWN SHARE (what the player did). Those differ
exactly when a role changed, which is the case the whole exercise is about.

RESULT — week 1 of 2022-25, prior season only:

    pos   model      MAE     bias    corr   seasons won
    WR    trailing  24.55   +6.07   0.540
          share     26.98   -2.96   0.249      1/4
          own       23.21   -0.54   0.535      4/4
    TE    trailing  16.41   +6.01   0.472
          share     14.67   -0.92   0.399      4/4
          own       14.99   +1.02   0.474      4/4
          blend     14.57   +2.54   0.492      4/4
    RB    trailing  11.27   +0.90   0.439
          share     11.46   -2.45   0.295      2/4
          own       10.49   -1.51   0.464      4/4

**The depth chart loses to the player's own prior share for receivers** — the
opposite of the rushing result, where depth rank worked. Chart rank is coarse:
two WR2s can see 25% and 15% of targets, and collapsing them to one number
destroys correlation (0.540 -> 0.249). A back's depth rank is closer to binary,
which is why it survives there and not here.

IN-SEASON (2025 weeks 2-6, current-season data once two weeks exist) — the
regime that actually matters from week 2 on:

    pos     n    MAE trail  MAE own   corr trail  corr own   weeks won
    WR    637      21.30     21.38      0.566      0.552       3/5
    TE    291      15.28     14.66      0.521      0.548       4/5
    RB    385      11.05     10.60      0.504      0.527       4/5

**WR does not survive the in-season test.** Its week-1 win was the year-stale
trailing average being bad, not the decomposition being good — once trailing is
current, the two are indistinguishable and correlation is slightly worse. TE and
RB receiving win in both regimes, on MAE and correlation together.

Run it:  python experiments/usage_share.py
"""
import pathlib
import numpy as np
import pandas as pd

D = pathlib.Path(__file__).parent
CACHE = pathlib.Path.home() / ".sharp-edge" / "nfl"
WEEKLY = ("https://github.com/nflverse/nflverse-data/releases/download/"
          "stats_player/stats_player_week_{s}.parquet")
DEPTH = ("https://github.com/nflverse/nflverse-data/releases/download/"
         "depth_charts/depth_charts_{s}.parquet")

# Position -> (the depth abbreviations it appears under, how many ranks matter)
POS_SLOTS = {
    "WR": (("WR", "LWR", "RWR", "SWR"), 5),
    "TE": (("TE",), 3),
    "RB": (("RB", "HB", "TB"), 4),
}


def weekly(seasons):
    out = []
    for s in seasons:
        p = CACHE / f"week_{s}.parquet"
        out.append(pd.read_parquet(p) if p.exists() else pd.read_parquet(WEEKLY.format(s=s)))
    return pd.concat(out, ignore_index=True)


def depth_ranks(season, position):
    """Rank within position per team, from the earliest chart of the season.

    nflverse changed the depth-chart schema for 2025 — the old one is one row
    per player per week with ``club_code``/``depth_team``/``position``, the new
    one is a much larger table keyed on ``team``/``pos_abb``/``pos_rank`` with a
    timestamp. Both are handled rather than pinning to one, because a harness
    that only reads the current format silently loses every earlier season and
    the backtest quietly becomes a one-season test.
    """
    p = D / f"depth_{season}.parquet"
    d = pd.read_parquet(p) if p.exists() else pd.read_parquet(DEPTH.format(s=season))
    if not p.exists():
        d.to_parquet(p)
    slots, _ = POS_SLOTS[position]

    if "pos_abb" in d.columns:                       # 2025+ schema
        d = d[d.pos_abb.isin(slots)].copy()
        d["dt"] = pd.to_datetime(d.dt, errors="coerce", utc=True)
        d = d[d.dt <= d.dt.min() + pd.Timedelta(days=30)]
        d = d.sort_values(["dt", "pos_rank"]).drop_duplicates(
            ["team", "gsis_id"], keep="first")
        d["rank"] = d.groupby("team").pos_rank.rank(method="first")
        return d[["team", "gsis_id", "rank"]].rename(columns={"gsis_id": "player_id"})

    # pre-2025 schema: take the earliest regular-season week available
    d = d[d.position.isin(slots)].copy()
    if "game_type" in d.columns:
        d = d[d.game_type.isin(["REG", None])]
    first_week = d.week.min()
    d = d[d.week == first_week]
    d = d.sort_values("depth_team").drop_duplicates(["club_code", "gsis_id"], keep="first")
    d["rank"] = d.groupby("club_code").depth_team.rank(method="first")
    return d[["club_code", "gsis_id", "rank"]].rename(
        columns={"club_code": "team", "gsis_id": "player_id"})


def share_curve(hist, through, position, window=4):
    """Median share of team targets by rank within position, fit on history.

    Rank is taken by targets within a team-season, which is the observable
    proxy for a depth chart in the historical record.
    """
    h = hist[(hist.season <= through) & (hist.season > through - window)
             & (hist.season_type == "REG")]
    pos = h[h.position == position].groupby(
        ["season", "team", "player_id"]).targets.sum().reset_index()
    pos = pos[pos.targets > 0]
    team = h.groupby(["season", "team"]).targets.sum().rename("team_tgt").reset_index()
    pos = pos.merge(team, on=["season", "team"])
    pos["rank"] = pos.groupby(["season", "team"]).targets.rank(ascending=False, method="first")
    pos["share"] = pos.targets / pos.team_tgt
    return pos.groupby("rank").share.median().to_dict()


def build(through, target, position):
    hist = weekly(range(through - 1, target + 1))
    for c in ("targets", "receiving_yards", "receptions"):
        hist[c] = hist[c].fillna(0)
    prior = hist[(hist.season == through) & (hist.season_type == "REG")]
    curve = share_curve(hist, through, position)

    p = prior[prior.position == position].groupby("player_id").agg(
        g=("week", "count"), tgt=("targets", "sum"),
        yds=("receiving_yards", "sum"), rec=("receptions", "sum")).reset_index()
    p["trailing"] = p.yds / p.g
    p["ypt"] = np.where(p.tgt > 0, p.yds / p.tgt, np.nan)
    p = p[p.g >= 6]

    # Efficiency regressed toward the positional mean by target count.
    LG = prior[prior.position == position].receiving_yards.sum() / max(
        prior[prior.position == position].targets.sum(), 1)
    K = 40.0
    p["ypt_reg"] = (p.ypt.fillna(LG) * p.tgt + LG * K) / (p.tgt + K)

    # What the player's own share was last season — the alternative to the chart.
    tm = prior.groupby("team").targets.sum().rename("team_tgt")
    own = prior[prior.position == position].groupby(
        ["player_id", "team"]).targets.sum().reset_index().merge(
        tm, left_on="team", right_index=True)
    own["own_share"] = own.targets / own.team_tgt
    p = p.merge(own[["player_id", "own_share"]], on="player_id", how="left")

    team_tgt_pg = (prior.groupby("team").targets.sum() / 17.0).rename("team_tgt_pg")
    ranks = depth_ranks(target, position)

    act = hist[(hist.season == target) & (hist.week == 1)
               & (hist.season_type == "REG") & (hist.position == position)]
    act = act.groupby(["player_id", "player_display_name", "team"]).agg(
        actual=("receiving_yards", "sum")).reset_index()
    act["actual"] = act.actual.fillna(0)

    d = act.merge(p[["player_id", "trailing", "ypt_reg", "tgt", "own_share"]],
                  on="player_id", how="inner")
    d = d.merge(ranks, on=["team", "player_id"], how="left")
    d = d.merge(team_tgt_pg, left_on="team", right_index=True, how="left")

    d["chart_share"] = d["rank"].map(curve)
    d["share"] = d.chart_share * d.team_tgt_pg * d.ypt_reg
    d["own"] = d.own_share * d.team_tgt_pg * d.ypt_reg
    d["blend"] = d[["trailing", "share"]].mean(axis=1)
    d["season"] = target
    return d


def report(position):
    frames = [build(t - 1, t, position) for t in (2022, 2023, 2024, 2025)]
    df = pd.concat(frames, ignore_index=True)
    cols = ["trailing", "share", "own", "blend"]
    d = df.dropna(subset=cols + ["actual"])
    print(f"\n=== {position} receiving yards, week 1 of 2022-25  (n={len(d)}) ===")
    print(f"{'model':<10}{'MAE':>8}{'bias':>8}{'corr':>8}{'seasons won':>13}")
    base = (d.trailing - d.actual).abs().mean()
    for c in cols:
        e = d[c] - d.actual
        w = sum(1 for t in (2022, 2023, 2024, 2025)
                if (d[d.season == t][c] - d[d.season == t].actual).abs().mean()
                < (d[d.season == t].trailing - d[d.season == t].actual).abs().mean())
        wins = "" if c == "trailing" else f"{w}/4"
        print(f"{c:<10}{e.abs().mean():8.2f}{e.mean():+8.2f}"
              f"{np.corrcoef(d[c], d.actual)[0,1]:8.3f}{wins:>13}")


if __name__ == "__main__":
    for pos in ("WR", "TE", "RB"):
        report(pos)
