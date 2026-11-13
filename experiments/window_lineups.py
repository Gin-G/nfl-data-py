"""Does the training window matter for LINEUPS, not just per-player error?

The clean rerun of #39 (2026-11-13) found a model retrained each week on the
current season alone slightly more accurate per player than one trained on all
prior seasons: 4.132 vs 4.158 MAE over 2025 weeks 5-18, 10 of 14 weeks. MAE is
not what the optimizer reads, so this builds cash lineups from each model's
projections for the same weeks and scores what they actually put up.

Pools: every player both models projected that week, priced by the salary
model fitted on the 2026 FanDuel slates (`lineup_depth.salary_model`: salary =
a + b x season-to-date FPPG per position), eight skill slots, $56,000 cap,
cash preset at aggression 50, 25 lineups -- the lineup-depth test's settings,
so the two read on one scale. The lineup a person would play is the builder's
own #1 by projection; top-5 is reported because the depth test found the head
of a cash build is where the value is.

The cash line: the user plays FanDuel 50/50s against ~120 points for a full
nine-man lineup. These builds have no defence, so the bar is 120 minus the
league-average defence (6.21, EXPERIMENTS.md "Defenses"): CASH_LINE - 6.21.

Projections come from `train_window_expanding.py`, which saves them; if the
file is missing this runs that harness first (it trains -- run on the cluster:
POST /experiments/run {"name": "window_lineups", "env": {"SEEDS": "3"}}).

    SEEDS=3 python experiments/window_lineups.py
"""
import os
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
import train_window_expanding as window  # noqa: E402
from nfl_projections import config  # noqa: E402
from nfl_projections import optimizer as opt  # noqa: E402

AGGRESSION = int(os.environ.get("AGGRESSION", 50))
N_LINEUPS = 25
CASH_LINE = float(os.environ.get("CASH_LINE", 120))
DEF_MEAN = 6.21
BAR = CASH_LINE - DEF_MEAN
WORKERS = int(os.environ.get("WORKERS", 4))


def predictions():
    path = window.predictions_path()
    if not path.exists():
        print(f"{path.name} missing: training both windows first")
        window.main()
    return pd.read_parquet(path)


def season_frame(season):
    """Team and season-to-date FPPG (strictly before the week) per player-week."""
    from nfl_projections import dataset
    if os.path.exists(config.DATASET_PATH):
        ds = pd.read_csv(config.DATASET_PATH, low_memory=False,
                         usecols=["player_id", "season", "week", "team",
                                  "fanduel_fantasy_points"])
    else:
        ds = dataset.build_dataset(output_path=config.DATASET_PATH)
    ds = ds[(ds.season == season) & (ds.week != "AVG")].copy()
    ds["week"] = ds.week.astype(int)
    ds = ds.sort_values(["player_id", "week"]).drop_duplicates(["player_id", "week"])
    ds["fppg"] = (ds.groupby("player_id").fanduel_fantasy_points
                  .transform(lambda s: s.shift().expanding().mean()))
    return ds[["player_id", "week", "team", "fppg"]]


def pools(preds, model):
    meta = season_frame(int(preds.season.iloc[0]))
    out = {}
    for (week, arm), g in preds.groupby(["week", "arm"]):
        g = g.merge(meta, on=["player_id", "week"], how="left")
        g = g[g.team.notna()]
        fppg = g.fppg.fillna(g.predicted)
        out[(int(week), arm)] = pd.DataFrame({
            "Nickname": g.player_id, "Position": g.position,
            "Roster Position": g.position, "Team": g.team,
            "Salary": [depth.price(model, p, f) for p, f in zip(g.position, fppg)],
            "FPPG": fppg, "fanduel_fantasy_points": g.predicted, "actual": g.actual,
        }).reset_index(drop=True)
    return out


def build(args):
    key, pool = args
    settings = opt.contest_settings("cash", AGGRESSION, num_lineups=N_LINEUPS)
    settings.pop("rb_vs_opposing_def", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lineups = opt.optimize_lineups(pool.drop(columns=["actual"]), salary_cap=depth.CAP,
                                       roster_slots=depth.SKILL_SLOTS,
                                       rb_vs_opposing_def=False, **settings)
    actual = dict(zip(pool.Nickname, pool.actual))
    proj = dict(zip(pool.Nickname, pool.fanduel_fantasy_points))
    rows = sorted(((sum(proj[n] for n in lu["Nickname"]), sum(actual[n] for n in lu["Nickname"]))
                   for lu in lineups), key=lambda r: -r[0])
    return key, rows


def main():
    preds = predictions()
    model = depth.salary_model(depth.real_pools())
    jobs = list(pools(preds, model).items())
    print(f"{len(jobs)} pools ({preds.week.nunique()} weeks x 2 windows), "
          f"{N_LINEUPS} cash lineups each, aggression {AGGRESSION}; "
          f"cash bar {BAR:.1f} for eight skill players ({CASH_LINE:g} - {DEF_MEAN} defence)")
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        built = dict(ex.map(build, jobs))

    recs = []
    for (week, arm), rows in built.items():
        if len(rows) < 5:
            continue
        a = np.array([r[1] for r in rows])
        recs.append(dict(week=week, arm=arm, top1=a[0], top5=a[:5].mean(), all=a.mean(),
                         top1_cash=a[0] >= BAR, top5_cash=(a[:5] >= BAR).mean()))
    df = pd.DataFrame(recs).pivot(index="week", columns="arm")
    pd.set_option("display.width", 200)
    print("\nper week, actual points of the builds (current = current season only)")
    show = pd.DataFrame({
        "top1 current": df.top1.current, "top1 full": df.top1.full,
        "top5 current": df.top5.current, "top5 full": df.top5.full})
    print(show.round(1).to_string())

    for metric in ("top1", "top5", "all"):
        d = df[metric].current - df[metric].full
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"\n{metric:<5} current {df[metric].current.mean():6.1f}  full "
              f"{df[metric].full.mean():6.1f}  paired {d.mean():+.2f}  95% CI "
              f"[{d.mean() - 1.96 * se:+.2f}, {d.mean() + 1.96 * se:+.2f}]  "
              f"current ahead {int((d > 0).sum())}/{len(d)} weeks")
    print(f"\nclearing the cash bar ({BAR:.1f}):")
    print(f"  the #1 lineup:   current {int(df.top1_cash.current.sum())}/{len(df)} weeks, "
          f"full {int(df.top1_cash.full.sum())}/{len(df)}")
    print(f"  top-5 lineups:   current {df.top5_cash.current.mean():.0%}, "
          f"full {df.top5_cash.full.mean():.0%} of lineups")
    df.to_parquet(HERE / "_cache" / f"window_lineups_{AGGRESSION}.parquet")


if __name__ == "__main__":
    main()
