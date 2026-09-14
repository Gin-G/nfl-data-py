"""Score a week's PUBLISHED projections against nflverse actuals, before NFL-API can.

NFL-API's scorer (app/scripts/score_projections.py) reads actuals from its own
player_stats table, which the Tuesday db-loader fills — so for a day and a half
after Sunday the prospective record has nothing to say. This reproduces its
rules against nflreadpy directly, read-only, so a week can be judged Monday:

  * the projection is what /projections/ served, not a re-run
  * players with no stat line are not scored (they didn't play)
  * a projection computed on a date after the team's gameday is excluded
  * naive = trailing 5 regular-season games, reaching into the prior season
  * FanDuel half-PPR with -2 per fumble, same as api.utils.fanduel_points

It also reports the projection with the availability weight divided back out
(`cond`): the preseason pipeline multiplies each week by expected games / 17,
which is a season-long rate, while the population scored is players who played.

    python experiments/live_week_accuracy.py 2026 1
"""
import json
import sys
import urllib.request
import warnings

import numpy as np
import pandas as pd
import nflreadpy as nfl

warnings.filterwarnings("ignore")
API = "https://nfl-api.nickknows.net/projections/"


def fanduel(r):
    g = lambda c: float(r[c]) if c in r and pd.notna(r[c]) else 0.0
    py, ry, recy = g("passing_yards"), g("rushing_yards"), g("receiving_yards")
    fum = g("rushing_fumbles") + g("receiving_fumbles") + g("sack_fumbles")
    return (py * .04 + g("passing_tds") * 4 - g("passing_interceptions") + 3 * (py >= 300)
            + ry * .1 + g("rushing_tds") * 6 + 3 * (ry >= 100)
            + g("receptions") * .5 + recy * .1 + g("receiving_tds") * 6 + 3 * (recy >= 100)
            - 2 * fum)


def load(season, week):
    req = urllib.request.Request(f"{API}?season={season}&week={week}&limit=2000",
                                 headers={"User-Agent": "Mozilla/5.0"})
    proj = pd.DataFrame(json.load(urllib.request.urlopen(req))["data"])

    st = nfl.load_player_stats([season - 1, season]).to_pandas()
    st = st[st.season_type == "REG"].copy()
    st["fp"] = st.apply(fanduel, axis=1)
    act = st[(st.season == season) & (st.week == week)].set_index("player_id").fp.rename("actual")
    prior = st[(st.season == season - 1) | ((st.season == season) & (st.week < week))]
    naive = (prior.sort_values(["player_id", "season", "week"]).groupby("player_id")
             .tail(5).groupby("player_id").fp.mean().rename("naive"))

    sch = nfl.load_schedules([season]).to_pandas()
    sch = sch[(sch.week == week) & (sch.game_type == "REG")]
    day = {**dict(zip(sch.home_team, sch.gameday)), **dict(zip(sch.away_team, sch.gameday))}
    played = sch[sch.home_score.notna()]
    done = set(played.home_team) | set(played.away_team)

    df = proj.join(act, on="player_id").join(naive, on="player_id")
    df["after_ko"] = pd.to_datetime(df.computed_at).dt.date.astype(str) > df.team.map(day)
    df["pp"] = df.projected_points
    df["cond"] = df.pp / df.exp_games.fillna(1.0).clip(lower=0.05)
    scored = df[df.team.isin(done) & df.actual.notna() & ~df.after_ko]
    pool = df[df.team.isin(done) & ~df.after_ko].assign(act0=lambda d: d.actual.fillna(0),
                                                       nv0=lambda d: d.naive.fillna(0))
    return df, scored, pool


def report(scored, pool):
    rows = []
    for label, d in [("ALL", scored)] + [(p, scored[scored.position == p]) for p in ["QB", "RB", "WR", "TE"]]:
        nv = d[d.naive.notna()]
        mae = lambda col, x: (x[col] - x.actual).abs().mean()
        rows.append(dict(
            group=label, n=len(d),
            mae_pub=mae("pp", d), bias_pub=(d.pp - d.actual).mean(),
            mae_cond=mae("cond", d), bias_cond=(d.cond - d.actual).mean(),
            n_nv=len(nv), pub_nvset=mae("pp", nv), cond_nvset=mae("cond", nv), naive=mae("naive", nv),
            bias_naive=(nv.naive - nv.actual).mean(),
            band=((d.floor <= d.actual) & (d.actual <= d.ceiling)).mean(),
            above_ceil=(d.actual > d.ceiling).mean(), below_floor=(d.actual < d.floor).mean(),
            rho_model=nv[["pp", "actual"]].corr("spearman").iloc[0, 1],
            rho_naive=nv[["naive", "actual"]].corr("spearman").iloc[0, 1]))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    for pos, n in [("QB", 12), ("RB", 24), ("WR", 36), ("TE", 12)]:
        d = pool[pool.position == pos]
        top = set(d.nlargest(n, "act0").player_id)
        print(f"{pos} top-{n} overlap: model {len(set(d.nlargest(n, 'pp').player_id) & top)}"
              f"  naive {len(set(d.nlargest(n, 'nv0').player_id) & top)}")


if __name__ == "__main__":
    season, week = int(sys.argv[1]), int(sys.argv[2])
    df, scored, pool = load(season, week)
    print(f"{len(df)} projections, {len(scored)} scored, "
          f"{int(df.after_ko.sum())} excluded as computed after kickoff")
    report(scored, pool)
