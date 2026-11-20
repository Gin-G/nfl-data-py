"""Re-score closed model experiments on the players who actually score.

Every model A/B in EXPERIMENTS.md was read on the whole backtest population,
~320 player-weeks a week, a quarter of whom finish under one point. The board
gets that tail right almost for free, so the full pool's MAE and correlation
are padded, and any change whose effect sits among startable players is
diluted roughly 2.5x before it reaches the 0.05 noise bar.

This retrains the baseline and each arm on identical player-weeks and scores
every pair twice: on the full pool (the old reading) and on the top of the
board (`evaluate.top_slice`: per week, the top 24 QB / 36 RB / 50 WR / 18 TE by
trailing-5 form, chosen before the game and the same for every arm).

    ARMS=huber,opponent,no_blend SEASONS=2024,2025 SEEDS=5 python experiments/topslice_rerun.py

Arms (each is evaluate.backtest settings; the baseline is always run):
    huber      loss="huber"                       (#3, "wash, worse booms")
    opponent   use_opponent=True                  (#4/#12, matchup as an input)
    no_blend   blend_form=False                   (does the form blend cost the top?)
    shallow    min_season = test season - 1        (#27, history depth)

Per-row projections are cached to _cache/topslice_<season>_<arm>_<seeds>s.parquet,
so a later slice definition needs no retraining.
"""
import os
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from nfl_projections import config, evaluate
from nfl_projections import dataset as dataset_mod

ARMS = [a for a in os.environ.get("ARMS", "huber,opponent,no_blend").split(",") if a]
SEASONS = [int(s) for s in os.environ.get("SEASONS", "2025").split(",")]
SEEDS = int(os.environ.get("SEEDS", 5))
EPOCHS = int(os.environ.get("EPOCHS", 60))
BOOT = int(os.environ.get("BOOT", 2000))
CACHE = pathlib.Path(__file__).resolve().parent / "_cache"


def arm_kwargs(arm, season):
    return {
        "baseline": {},
        "huber": {"loss": "huber"},
        "opponent": {"use_opponent": True},
        "no_blend": {"blend_form": False},
        "shallow": {"min_season": season - 1},
    }[arm]


def run_arm(df, season, arm):
    path = CACHE / f"topslice_{season}_{arm}_{SEEDS}s{EPOCHS}e.parquet"
    if path.exists():
        print(f"  {arm}: cached {path.name}")
        return pd.read_parquet(path)
    print(f"  {arm}: training ({SEEDS} seeds)...")
    res = evaluate.backtest(df, season, epochs=EPOCHS, n_seeds=SEEDS,
                            **arm_kwargs(arm, season))
    res = res[res["week"] <= 18].drop_duplicates(["player_id", "week"])
    CACHE.mkdir(exist_ok=True)
    res.to_parquet(path)
    return res


def paired(base, arm, mask, boot=BOOT, seed=7):
    """Arm minus baseline MAE on the masked rows, with a week-clustered 95% CI."""
    d = pd.DataFrame({"week": base["season"].astype(str) + "-" + base["week"].astype(str),
                      "diff": (arm["predicted"] - arm["actual"]).abs().values
                              - (base["predicted"] - base["actual"]).abs().values})[mask]
    by = d.groupby("week")["diff"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(by), size=(boot, len(by)))
    s, c = by["sum"].to_numpy(), by["count"].to_numpy()
    draws = s[idx].sum(1) / c[idx].sum(1)
    return d["diff"].mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5), len(d)


def main():
    print(f"Building dataset through {config.CURRENT_SEASON}...")
    df = dataset_mod.build_dataset(output_path=None)
    out = []
    for season in SEASONS:
        print(f"\n=== {season} ===")
        base = run_arm(df, season, "baseline")
        base["form"] = evaluate.trailing_form(df, base)
        for arm in ARMS:
            res = run_arm(df, season, arm)
            m = base[["player_id", "week", "form", "season", "position",
                      "predicted", "actual"]].merge(
                res[["player_id", "week", "predicted"]], on=["player_id", "week"],
                suffixes=("_b", "_a"))
            b = m.rename(columns={"predicted_b": "predicted"})
            a = m.assign(predicted=m["predicted_a"])
            top = evaluate.top_slice(b, "form")
            for pop, mask in (("all", np.ones(len(m), bool)), ("top", top)):
                for pos in ["ALL", *config.POSITIONS]:
                    pm = mask & ((b["position"] == pos).to_numpy() if pos != "ALL" else True)
                    if pm.sum() < 50:
                        continue
                    delta, lo, hi, n = paired(b, a, pm)
                    out.append(dict(season=season, arm=arm, pop=pop, pos=pos, n=n,
                                    base_mae=(b.predicted - b.actual)[pm].abs().mean(),
                                    delta=delta, lo=lo, hi=hi,
                                    base_corr=b.predicted[pm].corr(b.actual[pm]),
                                    arm_corr=a.predicted[pm].corr(a.actual[pm])))
            r = pd.DataFrame([o for o in out if o["season"] == season and o["arm"] == arm
                              and o["pos"] == "ALL"])
            for _, x in r.iterrows():
                print(f"  {arm:9s} {x['pop']:3s} n={x.n:6,} base {x.base_mae:.3f} "
                      f"delta {x.delta:+.3f} [{x.lo:+.3f}, {x.hi:+.3f}] "
                      f"corr {x.base_corr:.3f}->{x.arm_corr:.3f}")
    res = pd.DataFrame(out)
    pd.set_option("display.width", 220)
    print("\n=== all arms, all positions (delta = arm - baseline MAE; negative = arm better) ===")
    print(res.round(3).to_string(index=False))
    flips = res[(res["pos"] == "ALL")].pivot_table(index=["season", "arm"], columns="pop",
                                                   values=["delta", "lo", "hi"])
    print("\n=== full pool vs top slice ===")
    print(flips.round(3).to_string())


if __name__ == "__main__":
    main()
