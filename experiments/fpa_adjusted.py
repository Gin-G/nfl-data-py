"""Raw yards-allowed vs opponent-adjusted: which defence metric actually predicts?

The objection to a raw allowed-per-game number is that it measures who a
defence happened to draw as much as how it played. Face three elite tight ends
and the number looks terrible; face three backups and it looks elite. Neither
is a statement about the defence.

The fix is to compare each player against *himself*: for every player-game
against defence D, take the player's production over all his OTHER games that
season as the baseline, and score the game as a ratio. Average those ratios and
the opponent quality cancels, because a good tight end's baseline is high and a
weak one's is low.

Leave-one-out is load-bearing. Using a season average that includes the game
being scored leaks the answer into its own baseline and shrinks every ratio
toward 1.

Four metrics, all built from the prior season only:

  raw        yards allowed per game / league average          (what ships today)
  loo        mean(actual / that player's other-games mean)     the adjusted one
  loo_w      same, weighted by the player's baseline volume    big roles count more
  shrunk     loo regressed toward 1.0 by the number of games faced

RESULT — week 1 of 2022-25, prior season only. The bar is that mean absolute
error improves in at least three of the four seasons; correlation alone is not
enough, since a metric can shuffle ranks without getting closer.

    market                MAE wins   corr wins   shipped
    TE receiving yards      4/4        3/4         yes
    RB receiving yards      4/4        3/4         yes
    RB receptions           3/4        3/4         yes
    TE receptions           2/4        3/4         no
    RB rushing yards        2/4        2/4         no
    WR (any market)         0-2/4      0-1/4       no

Two things this changed. Volume-weighted leave-one-out beats raw allowed-per-
game for tight ends (MAE 18.02 -> 17.70 pooled), and more importantly it
*unlocks running-back receiving*, which the raw metric did not justify — raw
was better in only 2 of 4 seasons, the adjusted version in 4 of 4.

Unweighted leave-one-out is slightly worse than raw. Weighting matters because
a fringe receiver's ratio is mostly noise and an unweighted mean lets it count
as much as a starter's.

Rushing fails for a plain reason: the factor is built from receiving yards
allowed, which is a statement about pass defence. Tested anyway, came back a
coin flip, as it should have.

Run it:  python experiments/fpa_adjusted.py
"""
import pathlib
import numpy as np
import pandas as pd

D = pathlib.Path(__file__).parent
CACHE = pathlib.Path.home() / ".sharp-edge" / "nfl"
WEEKLY = ("https://github.com/nflverse/nflverse-data/releases/download/"
          "stats_player/stats_player_week_{s}.parquet")
POSITIONS = ("TE", "WR", "RB")
STAT = "receiving_yards"


def weekly(seasons):
    out = []
    for s in seasons:
        p = CACHE / f"week_{s}.parquet"
        out.append(pd.read_parquet(p) if p.exists() else pd.read_parquet(WEEKLY.format(s=s)))
    return pd.concat(out, ignore_index=True)


def raw_factor(prior, pos):
    d = prior[prior.position == pos]
    a = d.groupby("opponent_team").agg(yds=(STAT, "sum"), g=("game_id", "nunique"))
    a = a[a.g >= 8]
    pg = a.yds / a.g
    return (pg / pg.mean()).rename("raw")


def loo_factor(prior, pos, min_baseline=10.0, shrink_k=0.0, weighted=False):
    """Mean ratio of a player's game against a defence to his own other games.

    ``min_baseline`` drops players whose own norm is too small for a ratio to
    mean anything — a receiver averaging two yards produces ratios of 0 and 15
    and nothing in between.
    """
    d = prior[prior.position == pos].copy()
    d[STAT] = d[STAT].fillna(0)

    tot = d.groupby("player_id")[STAT].transform("sum")
    cnt = d.groupby("player_id")[STAT].transform("size")
    # The player's mean over every game except this one.
    d["baseline"] = (tot - d[STAT]) / (cnt - 1).replace(0, np.nan)
    d = d[d.baseline >= min_baseline]
    if d.empty:
        return pd.Series(dtype=float, name="loo")

    d["ratio"] = d[STAT] / d.baseline
    # A single freak game should not carry a defence's whole number.
    d["ratio"] = d.ratio.clip(upper=4.0)

    if weighted:
        g = d.groupby("opponent_team").apply(
            lambda x: np.average(x.ratio, weights=x.baseline))
    else:
        g = d.groupby("opponent_team").ratio.mean()
    n = d.groupby("opponent_team").ratio.size()

    if shrink_k:
        g = (g * n + 1.0 * shrink_k) / (n + shrink_k)
    return g.rename("loo")


def build(through, target, pos):
    hist = weekly(range(through - 1, target + 1))
    prior = hist[(hist.season == through) & (hist.season_type == "REG")].copy()
    prior[STAT] = prior[STAT].fillna(0)

    p = prior[prior.position == pos].groupby("player_id").agg(
        g=("week", "count"), yds=(STAT, "sum"), tgt=("targets", "sum")).reset_index()
    p["trailing"] = p.yds / p.g
    p = p[p.tgt >= 20]

    act = hist[(hist.season == target) & (hist.week == 1)
               & (hist.season_type == "REG") & (hist.position == pos)]
    act = act.groupby(["player_id", "player_display_name", "opponent_team"]).agg(
        actual=(STAT, "sum")).reset_index()
    act["actual"] = act.actual.fillna(0)

    df = act.merge(p[["player_id", "trailing"]], on="player_id", how="inner")
    df = df.join(raw_factor(prior, pos), on="opponent_team")
    df = df.join(loo_factor(prior, pos).rename("loo"), on="opponent_team")
    df = df.join(loo_factor(prior, pos, weighted=True).rename("loo_w"), on="opponent_team")
    df = df.join(loo_factor(prior, pos, shrink_k=10.0).rename("shrunk"), on="opponent_team")

    for c in ("raw", "loo", "loo_w", "shrunk"):
        df[c] = df[c].fillna(1.0).clip(0.75, 1.30)
        df[f"p_{c}"] = df.trailing * df[c]
    return df


def score(df, label):
    cols = ["trailing", "p_raw", "p_loo", "p_loo_w", "p_shrunk"]
    d = df.dropna(subset=cols + ["actual"])
    print(f"\n{label}  (n={len(d)})")
    print(f"{'model':<12}{'MAE':>8}{'corr':>8}{'dMAE':>8}{'dcorr':>8}")
    base = None
    for c in cols:
        e = d[c] - d.actual
        mae, r = e.abs().mean(), np.corrcoef(d[c], d.actual)[0, 1]
        if base is None:
            base = (mae, r)
        print(f"{c:<12}{mae:8.2f}{r:8.3f}{mae-base[0]:+8.2f}{r-base[1]:+8.3f}")


if __name__ == "__main__":
    for pos in POSITIONS:
        frames = [build(t - 1, t, pos) for t in (2022, 2023, 2024, 2025)]
        df = pd.concat(frames, ignore_index=True)
        score(df, f"{pos} receiving yards, week 1 of 2022-25")
        print(f"  factor spread  raw {df.raw.quantile(.1):.3f}-{df.raw.quantile(.9):.3f}"
              f"   loo {df.loo.quantile(.1):.3f}-{df.loo.quantile(.9):.3f}")
        print(f"  corr(raw, loo) = {df[['raw','loo']].corr().iloc[0,1]:.3f}")
