"""The selection tax: does shrinking toward trailing form pay inside the chosen set?

An optimizer maximising a noisy estimate picks the players whose noise ran
positive (2026 week 9: +2.45 bias on the 30 chosen, -0.01 on the 439 not).
Candidate fix: move each projection toward the player's trailing-5 average,
board + w * (trailing5 - board), with w fitted out of sample.

Kill rules, fixed before running:
  slope stage   -- regress (actual - board) on (trailing5 - board) for players
                   projected >= 8, 95% CI clustered by week. CI covering zero:
                   stop.
  payment stage -- w fitted leaving one season out, cash lineups rebuilt on the
                   adjusted board and scored. Top-5 gain under +3 points a
                   lineup, or a CI covering zero: stop.

Data: the 2026 live record (/projections/accuracy rows: frozen pre-game board,
production's own trailing-5 `naive_points`, actuals), and the five cached
backtest seasons 2021-2025 (trailing-5 from the same board's actuals, games
strictly before the week). The backtest board already blends the network with
trailing-5 form (blend.py), so this asks whether MORE form weight helps the
top of the board, not whether form helps at all.

    python experiments/selection_tax.py
"""
import pathlib
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import lineup_depth as depth  # noqa: E402
from nfl_projections import optimizer as opt  # noqa: E402

MIN_PROJ = 8.0
AGGRESSION = 50
N_LINEUPS = 25


def ols_cluster(y, x, groups, intercept=True):
    """Slope and a cluster-robust standard error (clusters = weeks)."""
    y, x, groups = np.asarray(y, float), np.asarray(x, float), np.asarray(groups)
    X = np.column_stack([np.ones_like(x), x]) if intercept else x[:, None]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        s = X[groups == g].T @ resid[groups == g]
        meat += np.outer(s, s)
    k = len(np.unique(groups))
    cov = bread @ meat @ bread * k / max(k - 1, 1)
    return beta[-1], float(np.sqrt(cov[-1, -1])), beta


def live_rows():
    acc = depth.accuracy_rows()
    acc = acc[acc.naive_points.notna() & acc.projected_points.notna()].copy()
    acc["board"], acc["naive"], acc["actual"] = acc.projected_points, acc.naive_points, acc.actual_points
    acc["season_week"] = acc.season.astype(str) + "-" + acc.week.astype(str)
    return acc


def backtest_rows():
    frames = []
    for season in depth.SEASONS:
        b = pd.read_parquet(depth.CACHE / f"blitz_board_{season}_1s60e.parquet")
        b = b[b.position.isin(["QB", "RB", "WR", "TE"])].sort_values(["player_id", "week"])
        b = b.drop_duplicates(["player_id", "week"])   # see lineup_depth.backtest_pools
        b["naive"] = (b.groupby("player_id").actual
                       .transform(lambda s: s.shift().rolling(5, min_periods=1).mean()))
        frames.append(b)
    b = pd.concat(frames, ignore_index=True)
    b = b[b.naive.notna() & b.week.between(4, 18)].copy()
    b["board"] = b.predicted
    b["season_week"] = b.season.astype(str) + "-" + b.week.astype(str)
    return b


def slope_stage(rows, label):
    r = rows[rows.board >= MIN_PROJ]
    y, x = r.actual - r.board, r.naive - r.board
    b, se, beta = ols_cluster(y, x, r.season_week)
    print(f"  {label:<26} n {len(r):>5}  weeks {r.season_week.nunique():>3}  "
          f"slope {b:+.3f}  95% CI [{b - 1.96 * se:+.3f}, {b + 1.96 * se:+.3f}]  "
          f"intercept {beta[0]:+.2f}  sd(naive-board) {x.std():.2f}")
    return b, se


def payment_mae(rows):
    """Leave-one-season-out w, no intercept, on the 8+ population; paired MAE."""
    r = rows[rows.board >= MIN_PROJ].copy()
    r["adj"] = np.nan
    ws = {}
    for season in sorted(r.season.unique()):
        train = r[r.season != season]
        w, _, _ = ols_cluster(train.actual - train.board, train.naive - train.board,
                              train.season_week, intercept=False)
        ws[season] = w
        m = r.season == season
        r.loc[m, "adj"] = r.loc[m, "board"] + w * (r.loc[m, "naive"] - r.loc[m, "board"])
    d = (r.adj - r.actual).abs() - (r.board - r.actual).abs()
    per_week = d.groupby(r.season_week).mean()
    se = per_week.std(ddof=1) / np.sqrt(len(per_week))
    print(f"  w by held-out season: { {k: round(v, 3) for k, v in ws.items()} }")
    print(f"  MAE board {(r.board - r.actual).abs().mean():.3f} -> adjusted "
          f"{(r.adj - r.actual).abs().mean():.3f}; paired change {d.mean():+.3f}, "
          f"95% CI (weeks) [{per_week.mean() - 1.96 * se:+.3f}, {per_week.mean() + 1.96 * se:+.3f}]")
    return ws


def _build(args):
    key, pool, column = args
    settings = opt.contest_settings("cash", AGGRESSION, num_lineups=N_LINEUPS)
    settings.pop("rb_vs_opposing_def", None)
    p = pool.copy()
    p["fanduel_fantasy_points"] = p[column]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lineups = opt.optimize_lineups(
            p[["Nickname", "Position", "Roster Position", "Team", "Salary", "FPPG",
               "fanduel_fantasy_points"]],
            salary_cap=depth.CAP, roster_slots=depth.SKILL_SLOTS,
            rb_vs_opposing_def=False, **settings)
    actual = dict(zip(pool.Nickname, pool.actual))
    board = dict(zip(pool.Nickname, pool.board))
    rows = []
    for lu in lineups:
        names = list(lu["Nickname"].astype(str))
        rows.append((sum(p.set_index("Nickname").loc[names, "fanduel_fantasy_points"]),
                     sum(actual[n] for n in names), sum(board[n] for n in names), names))
    rows.sort(key=lambda t: -t[0])
    return key, column, rows


def payment_lineups(ws):
    """Rebuild the backtest cash builds on the adjusted board; paired by week."""
    model = depth.salary_model(depth.real_pools())
    pools = depth.backtest_pools(model)
    trail = backtest_rows().set_index(["season", "week", "player_id"]).naive
    jobs = []
    for (season, week), pool in pools.items():
        p = pool.copy()
        p["board"] = p.fanduel_fantasy_points
        p["naive"] = [trail.get((season, week, pid), np.nan) for pid in p.Nickname]
        p["naive"] = p.naive.fillna(p.board)
        w = ws[season]
        # Only the top of the board is adjusted: that is the population w was
        # fitted on and the only one a lineup draws from.
        p["adjusted"] = np.where(p.board >= MIN_PROJ, p.board + w * (p.naive - p.board), p.board)
        jobs += [((season, week), p, "board"), ((season, week), p, "adjusted")]
    with ProcessPoolExecutor(max_workers=6) as ex:
        out = list(ex.map(_build, jobs, chunksize=4))
    res = {}
    for key, column, rows in out:
        res.setdefault(key, {})[column] = rows
    recs, sel = [], []
    for key, d in res.items():
        if len(d.get("board", [])) < N_LINEUPS or len(d.get("adjusted", [])) < N_LINEUPS:
            continue
        rec = {"season": key[0], "week": key[1]}
        for col in ("board", "adjusted"):
            a = np.array([r[1] for r in d[col]])
            rec[f"{col}_top5"], rec[f"{col}_all"] = a[:5].mean(), a.mean()
            # Selection tax on the board build: board total minus actual.
            if col == "board":
                rec["tax"] = np.mean([r[2] - r[1] for r in d[col]])
        recs.append(rec)
    df = pd.DataFrame(recs)
    for metric in ("top5", "all"):
        g = df[f"adjusted_{metric}"] - df[f"board_{metric}"]
        se = g.std(ddof=1) / np.sqrt(len(g))
        print(f"  {metric:<5} board {df[f'board_{metric}'].mean():6.2f} -> adjusted "
              f"{df[f'adjusted_{metric}'].mean():6.2f}; paired {g.mean():+.2f}, 95% CI "
              f"[{g.mean() - 1.96 * se:+.2f}, {g.mean() + 1.96 * se:+.2f}], "
              f"adjusted ahead in {int((g > 0).sum())}/{len(g)} weeks")
    print(f"  selection tax on the board build: board total minus actual, "
          f"{df.tax.mean():+.2f} a lineup (8 skill players)")
    df.to_parquet(depth.CACHE / "selection_tax_lineups.parquet")


def main():
    live, bt = live_rows(), backtest_rows()
    print("slope stage: (actual - board) on (trailing5 - board), players projected >= 8")
    slope_stage(live, "2026 live record")
    slope_stage(bt, "backtest 2021-2025")
    for season in depth.SEASONS:
        slope_stage(bt[bt.season == season], f"  {season}")
    print("\npayment stage, per player (backtest, leave one season out):")
    ws = payment_mae(bt)
    print("\npayment stage, lineups (backtest cash builds, 25 lineups):")
    payment_lineups(ws)


if __name__ == "__main__":
    main()
