"""When a regular is out, does the board under-project whoever inherits his work?

Two cases, one mechanism (2026-11-16):

* **The RB.** Kenneth Walker out in 2026 week 10; Emmett Johnson played the
  whole backfield and scored 35.1 off a 6.0 projection. The board's answer is a
  rank multiplier on the backup's own number (x1.25, now x1.25 x 1.42 for the
  RB2 into the RB1 job). The alternative the user put: price the *opportunity*.
  The starter's touches do not vanish; the backup inherits them, at his own
  points per touch.
* **The pass catchers.** The Rams in week 10 with Mumpfield, Atwell and
  Parkinson all hurt. The depth chart re-ranks, but a WR's promotion factor is
  1.0 and the redistributed targets are not priced anywhere.

The feature, per active player in a team-week:

    inherited = vacated opportunities x his share of the active pool
                x his own points per opportunity (shrunk to the position rate)

* vacated: the trailing per-game opportunities (carries + targets for an RB's
  backfield pool, targets for the passing pool) of the team's **regulars who did
  not play** -- the oracle trigger, by gsis id off snap counts. A regular has
  2+ games for this team this season (before this week), averages at least
  MIN_OPP per game in that pool, and played the team's previous game -- a fresh
  absence. (A first pass without that last rule counted every long-term IR
  player as vacated every week, tagging 40% of rows; that is already in form.)
* share: his trailing opportunities over the active, on-board pool members'.
  Carries go to RBs only; targets to WR/TE/RB.
* rate: trailing FanDuel points per opportunity, shrunk toward the position's
  league rate with SHRINK pseudo-opportunities.

Tested the standard way against the production chain's own board (the saved
``promo_e2e_history.parquet`` arms, 2025 + 2026 wk 1-10, with the shipped
RB2-into-RB1 factor applied): a slope stage (board residual on the feature) and a
payment stage (fit on one season, applied to the other, intercept dropped,
paired MAE with a team-week bootstrap). Also the user's literal form for the RB2
into RB1 case: (his own opportunities + the starter's) x his rate, no fitted
parameter.

No named player enters anything here; Johnson is the motivating case only.
"""
import pathlib

import numpy as np
import pandas as pd

from nfl_projections.scoring import fanduel_points_from_stats

CACHE = pathlib.Path(__file__).parent / "_cache"
SEASONS = (2025, 2026)
MIN_GAMES = 2
TRAIL = 6                      # trailing games for usage and rate
MIN_OPP = {"rush": 6.0, "pass": 3.0}
SHRINK = 30.0                  # pseudo-opportunities toward the league rate
BOOT = 2000
SEED = 20261116
SHIPPED_RB2_FACTOR = 1.42      # roles.PROMOTION_INTO_STARTER["RB"], pinned


def weekly(season):
    path = CACHE / f"vo_weekly_{season}.parquet"
    if path.exists() and season != 2026:
        return pd.read_parquet(path)
    import nflreadpy as nfl
    d = nfl.load_player_stats(seasons=[season]).to_pandas()
    d = d[(d.season_type == "REG") & d.position.isin(["RB", "WR", "TE"])].copy()
    d["fp"] = fanduel_points_from_stats(d)
    d = d[["season", "week", "player_id", "player_display_name", "position", "team",
           "carries", "targets", "fp"]].fillna({"carries": 0, "targets": 0})
    d.to_parquet(path)
    return d


def played(season):
    """gsis ids with an offensive snap, and the teams that played, by week."""
    from promotion_endtoend import snaps
    import nflreadpy as nfl
    sn = snaps(season)
    sn = sn[sn.game_type == "REG"]
    xw = nfl.load_players().to_pandas()[["gsis_id", "pfr_id"]].dropna()
    sn = sn.assign(gsis=sn.pfr_player_id.map(dict(zip(xw.pfr_id, xw.gsis_id))))
    on = sn[sn.offense_snaps > 0]
    return ({int(w): set(g.gsis.dropna()) for w, g in on.groupby("week")},
            {int(w): set(g.team) for w, g in sn.groupby("week")})


def usage(st, season, week):
    """Per player on his current team: trailing per-game pool usage and rate."""
    prior = st[(st.season == season) & (st.week < week)]
    prior = prior.sort_values("week").groupby(["player_id", "team"]).tail(TRAIL)
    g = prior.groupby(["player_id", "team"]).agg(
        position=("position", "last"), games=("week", "nunique"),
        carries=("carries", "sum"), targets=("targets", "sum"), fp=("fp", "sum"))
    g["rush_pg"] = (g.carries + g.targets) / g.games      # the backfield pool
    g["pass_pg"] = g.targets / g.games
    g["opp"] = np.where(g.position == "RB", g.carries + g.targets, g.targets)
    return g.reset_index()


def league_rates(st):
    s = st.assign(opp=np.where(st.position == "RB", st.carries + st.targets, st.targets))
    s = s.groupby("position")[["fp", "opp"]].sum()
    return (s.fp / s.opp).to_dict()


def team_week_features(board, st, plays, teams):
    rate0 = league_rates(st)
    out = []
    for (season, week), bw in board.groupby(["season", "week"]):
        u = usage(st, season, week)
        if u.empty or week not in plays[season]:
            continue
        on = plays[season][week]
        u = u[u.games >= MIN_GAMES]
        u["rate"] = ((u.fp + SHRINK * u.position.map(rate0))
                     / (u.opp + SHRINK))
        for team, bt in bw.groupby("team"):
            if team not in teams[season][week]:
                continue
            ut = u[u.team == team]
            # Fresh absences only: he played this team's previous game. A
            # regular out for weeks is already in everyone's form.
            prev = [w for w in plays[season] if w < week and team in teams[season][w]]
            last = plays[season][max(prev)] if prev else set()
            absent = ut[~ut.player_id.isin(on) & ut.player_id.isin(last)]
            vac_rush = absent[(absent.position == "RB")
                              & (absent.rush_pg >= MIN_OPP["rush"])].rush_pg.sum()
            vac_pass = absent[absent.pass_pg >= MIN_OPP["pass"]].pass_pg.sum()
            # RB pass targets are counted in both pools for an absent RB; take
            # them out of the passing pool so they are not handed out twice.
            vac_pass -= absent[(absent.position == "RB")
                               & (absent.rush_pg >= MIN_OPP["rush"])
                               & (absent.pass_pg >= MIN_OPP["pass"])].pass_pg.sum()
            active = ut[ut.player_id.isin(set(bt.player_id))]
            rush_pool = active[active.position == "RB"].rush_pg.sum()
            pass_pool = active.pass_pg.sum()
            starter_out = absent[(absent.position == "RB")
                                 & (absent.rush_pg >= MIN_OPP["rush"])].rush_pg.max()
            for _, p in active.iterrows():
                extra = 0.0
                if p.position == "RB" and rush_pool > 0:
                    extra += vac_rush * p.rush_pg / rush_pool
                elif pass_pool > 0:
                    extra += max(vac_pass, 0) * p.pass_pg / pass_pool
                out.append({"season": season, "week": week, "team": team,
                            "player_id": p.player_id, "rate": p.rate,
                            "own_opp_pg": p.rush_pg if p.position == "RB" else p.pass_pg,
                            "extra_opp": extra, "inherited": extra * p.rate,
                            "starter_opp_pg": starter_out if p.position == "RB" else np.nan,
                            "vac_rush": vac_rush, "vac_pass": max(vac_pass, 0)})
    return pd.DataFrame(out)


def board():
    d = pd.read_parquet(CACHE / "promo_e2e_history.parquet")
    d = d[d.position.isin(["RB", "WR", "TE"])].copy()
    rb2 = (d.position == "RB") & (d["rank"] == 2) & (d.promo_lift != 1.0)
    d["rb2_into_rb1"] = rb2
    # The board as it ships now: the proposed arm plus the RB2-into-RB1 factor.
    d["board"] = np.where(rb2, (d.proposed * SHIPPED_RB2_FACTOR).round(1), d.proposed)
    return d


def ols(x, y):
    x1 = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x1, y, rcond=None)
    resid = y - x1 @ beta
    cov = np.linalg.inv(x1.T @ x1) * (resid @ resid) / (len(y) - 2)
    return beta[1], 1.96 * np.sqrt(cov[1, 1])


def paired(a, b, actual, groups, rng):
    ea, eb = np.abs(a - actual), np.abs(b - actual)
    diff = pd.Series(eb - ea).groupby(groups.values).agg(["sum", "count"])
    keys = np.arange(len(diff))
    draws = []
    for _ in range(BOOT):
        pick = rng.choice(keys, len(keys))
        s = diff.iloc[pick]
        draws.append(s["sum"].sum() / s["count"].sum())
    return eb.mean() - ea.mean(), np.percentile(draws, [5, 95])


def run():
    st = pd.concat([weekly(s) for s in SEASONS], ignore_index=True)
    plays, teams = {}, {}
    for s in SEASONS:
        plays[s], teams[s] = played(s)
    b = board()
    f = team_week_features(b, st, plays, teams)
    d = b.merge(f, on=["season", "week", "team", "player_id"], how="inner")
    d = d[d.actual.notna()]
    d["resid"] = d.actual - d.board
    rng = np.random.default_rng(SEED)
    grp = d.season.astype(str) + "-" + d.week.astype(str) + "-" + d.team
    print(f"rows {len(d)}, with an inheritance {int((d.inherited > 0).sum())}")

    print("\n== slope stage: board residual on inherited points ==")
    for label, m in (("all", d.inherited > 0),
                     ("RB", (d.inherited > 0) & (d.position == "RB")),
                     ("WR/TE", (d.inherited > 0) & d.position.isin(["WR", "TE"])),
                     ("RB2 into RB1", d.rb2_into_rb1 & (d.inherited > 0))):
        s = d[m]
        if len(s) < 10:
            print(f"{label:13s} n={len(s)} too few")
            continue
        slope, ci = ols(s.inherited.values, s.resid.values)
        print(f"{label:13s} n={len(s):4d}  mean inherited {s.inherited.mean():5.2f}  "
              f"resid {s.resid.mean():+5.2f}  slope {slope:+.3f} +- {ci:.3f}")

    print("\n== payment stage: fit on one season, apply to the other ==")
    for label, pos in (("RB", ["RB"]), ("WR/TE", ["WR", "TE"])):
        s = d[d.position.isin(pos)].copy()
        pred = s.board.copy()
        for hold in SEASONS:
            tr = s[(s.season != hold) & (s.inherited > 0)]
            x, y = tr.inherited.values, tr.resid.values
            k = float(x @ y / (x @ x)) if len(tr) else 0.0     # intercept dropped
            te = s.season == hold
            pred[te] = s.board[te] + k * s.inherited[te]
            print(f"  {label} fit without {hold}: k={k:+.3f} (n={len(tr)})")
        for scope, m in (("whole", s.inherited >= 0), ("touched", s.inherited > 0)):
            gain, ci = paired(s.board[m].values, pred[m].values, s.actual[m].values,
                              grp[s.index][m], rng)
            print(f"  {label} {scope:7s} n={int(m.sum()):5d}  board MAE "
                  f"{(s.board[m] - s.actual[m]).abs().mean():.4f}  with feature "
                  f"{(pred[m] - s.actual[m]).abs().mean():.4f}  gain {gain:+.4f} "
                  f"90% [{ci[0]:+.4f}, {ci[1]:+.4f}]")

    print("\n== RB2 into RB1: price the job, not the backup ==")
    r = d[d.rb2_into_rb1 & d.starter_opp_pg.notna()].copy()
    r["job"] = (r.own_opp_pg + r.starter_opp_pg) * r.rate
    r["proposed_x1"] = r.proposed
    for col in ("proposed_x1", "board", "job"):
        print(f"  {col:12s} n={len(r)} mean {r[col].mean():5.2f}  actual "
              f"{r.actual.mean():5.2f}  MAE {(r[col] - r.actual).abs().mean():.3f}  "
              f"corr {np.corrcoef(r[col], r.actual)[0, 1]:.3f}")
    if len(r) >= 10:
        gain, ci = paired(r.board.values, r.job.values, r.actual.values,
                          grp[r.index], rng)
        print(f"  job vs board gain {gain:+.3f} 90% [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    r.to_parquet(CACHE / "vo_rb2.parquet")
    d.to_parquet(CACHE / "vo_rows.parquet")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    run()
