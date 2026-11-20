"""#26 re-read on the top of the board: current-season-only vs fully trained.

The live model trains on the current season alone (owner's call, 2026-10-07),
measured at +0.068 MAE retrained weekly -- on the full pool. This re-scores the
per-player projections train_window_expanding.py already saved (no retraining
when the file is there; it runs that script first when it is not) on the
`evaluate.top_slice` population.

    TEST_SEASON=2025 SEEDS=5 python experiments/topslice_window.py
"""
import os
import pathlib
import runpy
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import train_window_expanding as tw                                  # noqa: E402
from topslice_rerun import paired                                   # noqa: E402
from nfl_projections import config, evaluate                         # noqa: E402
from nfl_projections import dataset as dataset_mod                   # noqa: E402


def main():
    path = tw.predictions_path()
    if not path.exists():
        print(f"{path.name} missing; running train_window_expanding first")
        runpy.run_path(str(pathlib.Path(tw.__file__)), run_name="__main__")
    preds = pd.read_parquet(path)
    df = dataset_mod.build_dataset(output_path=None)
    wide = preds.pivot_table(index=["player_id", "season", "week", "position", "actual"],
                             columns="arm", values="predicted").reset_index().dropna()
    wide["form"] = evaluate.trailing_form(df, wide)
    full = wide.assign(predicted=wide["full"])
    cur = wide.assign(predicted=wide["current"])
    top = evaluate.top_slice(full, "form")
    print(f"{tw.TEST_SEASON} weeks {tw.FIRST}-{tw.LAST}, {tw.SEEDS} seed(s): "
          "current-season-only minus fully trained (negative = current better)")
    for pop, mask in (("all", np.ones(len(wide), bool)), ("top", top)):
        for pos in ["ALL", *config.POSITIONS]:
            pm = mask & ((wide["position"] == pos).to_numpy() if pos != "ALL" else True)
            if pm.sum() < 50:
                continue
            d, lo, hi, n = paired(full, cur, pm)
            print(f"  {pop:3s} {pos:3s} n={n:5,} full {(full.predicted - full.actual)[pm].abs().mean():.3f} "
                  f"delta {d:+.3f} [{lo:+.3f}, {hi:+.3f}] "
                  f"corr {full.predicted[pm].corr(full.actual[pm]):.3f}->"
                  f"{cur.predicted[pm].corr(cur.actual[pm]):.3f}")


if __name__ == "__main__":
    main()
