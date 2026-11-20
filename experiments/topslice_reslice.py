"""Is no_blend's top-slice win real, or manufactured by the slice?

topslice_rerun.py (cluster runs 3-4) found dropping the form blend helps the
top slice at WR/TE in both 2024 and 2025 while it is neutral on the full pool.
But the slice is chosen by trailing-5 form, and trailing-5 form IS the blend's
partner (blend.py: fanduel_fantasy_points_roll5). Selecting players for a high
roll5 selects players whose roll5 overstates them (regression to the mean), so
any arm that leans less on roll5 is flattered on that slice by construction.

This re-scores the cached boards (no training) on slices chosen by keys that
are less and less tied to roll5, all ex-ante:

    form5     trailing 5 games (the current slice; the blend partner)
    form16    trailing 16 regular-season games across seasons
    prior     previous season's points per game (NaN -> ranks last)
    actual    the week's actual points (hindsight; the opposite bias, a bound)

and prints, per key, arm-minus-baseline MAE with the week-clustered CI plus
each side's bias (actual minus predicted) on the slice.

    ARMS=no_blend,huber,opponent,shallow SEASONS=2024,2025 SEEDS=5 python experiments/topslice_reslice.py
"""
import os
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from topslice_rerun import CACHE, EPOCHS, paired                    # noqa: E402
from nfl_projections import config, evaluate                         # noqa: E402
from nfl_projections import dataset as dataset_mod                   # noqa: E402

ARMS = [a for a in os.environ.get("ARMS", "no_blend,huber,opponent,shallow").split(",") if a]
SEASONS = [int(s) for s in os.environ.get("SEASONS", "2024,2025").split(",")]
SEEDS = int(os.environ.get("SEEDS", 5))


def prior_season_ppg(history, results):
    h = evaluate.regular_games(history)[["player_id", "season", "week", "fanduel_fantasy_points"]]
    h = h.dropna().drop_duplicates(["player_id", "season", "week"])
    ppg = h.groupby(["player_id", "season"])["fanduel_fantasy_points"].mean().rename("_p").reset_index()
    ppg["season"] = ppg["season"] + 1
    r = results[["player_id", "season"]].merge(ppg, on=["player_id", "season"], how="left")
    return r["_p"].to_numpy()


def load(season, arm):
    path = CACHE / f"topslice_{season}_{arm}_{SEEDS}s{EPOCHS}e.parquet"
    if not path.exists():
        print(f"  missing {path.name}; run topslice_rerun.py first")
        return None
    return pd.read_parquet(path)


def main():
    df = dataset_mod.build_dataset(output_path=None)
    for season in SEASONS:
        base = load(season, "baseline")
        if base is None:
            continue
        keys = {"form5": evaluate.trailing_form(df, base),
                "form16": evaluate.trailing_form(df, base, n_games=16),
                "prior": prior_season_ppg(df, base),
                "actual": base["actual"].to_numpy()}
        for name, k in keys.items():
            base["_" + name] = k
        print(f"\n=== {season} ({SEEDS} seeds) === delta = arm - baseline MAE "
              "(negative = arm better); bias = mean(actual - predicted)")
        for arm in ARMS:
            res = load(season, arm)
            if res is None:
                continue
            m = base.merge(res[["player_id", "week", "predicted"]],
                           on=["player_id", "week"], suffixes=("", "_a"))
            a = m.assign(predicted=m["predicted_a"])
            for name in keys:
                top = evaluate.top_slice(m, "_" + name)
                for pos in ["ALL", *config.POSITIONS]:
                    pm = top & ((m["position"] == pos).to_numpy() if pos != "ALL" else True)
                    d, lo, hi, n = paired(m, a, pm)
                    bb = (m.actual - m.predicted)[pm].mean()
                    ba = (a.actual - a.predicted)[pm].mean()
                    print(f"  {arm:9s} {name:6s} {pos:3s} n={n:5,} delta {d:+.3f} "
                          f"[{lo:+.3f}, {hi:+.3f}]  bias base {bb:+.2f} arm {ba:+.2f}")


if __name__ == "__main__":
    main()
