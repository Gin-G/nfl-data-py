"""Ship check for volume.py: one network, scored with the correction on and off.

The gate (opportunity_gate.py, `tier5`) fitted the correction on board
residuals. This scores the code that ships: evaluate.backtest with
volume_correction on vs off, using the same trained ensemble, so the delta is
the correction and nothing else. Every other setting is production (the
blend and its top tier on).

2025 is the honest season: the constants were fitted on 2021-24 boards. 2024
is partly in-sample (its board was in the fit) and is printed for direction only.

    SEASONS=2025,2024 SEEDS=5 python experiments/volume_verify.py
"""
import os
import pathlib
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from topslice_rerun import paired                                    # noqa: E402
from nfl_projections import config, evaluate                         # noqa: E402
from nfl_projections import dataset as dataset_mod                   # noqa: E402
from nfl_projections import model as model_mod                       # noqa: E402

SEASONS = [int(s) for s in os.environ.get("SEASONS", "2025,2024").split(",")]
SEEDS = int(os.environ.get("SEEDS", 5))
EPOCHS = int(os.environ.get("EPOCHS", 60))


def main():
    df = dataset_mod.build_dataset(output_path=None)
    for season in SEASONS:
        trained, _ = model_mod.train_ensemble(
            df[df["season"] < season], n_seeds=SEEDS, epochs=EPOCHS,
            min_season=config.TRAINING_MIN_SEASON, plot_path=None)
        k = ["player_id", "week"]
        on = evaluate.backtest(df, season, trained=trained, volume_correction=True)
        off = evaluate.backtest(df, season, trained=trained, volume_correction=False)
        on = on[on.week <= 18].drop_duplicates(k)
        off = off[off.week <= 18].drop_duplicates(k)
        m = off.merge(on[k + ["predicted"]], on=k, suffixes=("", "_on"))
        a = m.assign(predicted=m["predicted_on"])
        top = evaluate.top_slice(m, evaluate.trailing_form(df, m))
        tag = "out of sample" if season >= 2025 else "constants partly fitted on this season"
        print(f"\n=== {season} ({SEEDS} seeds, {tag}): correction on minus off "
              "(negative = correction better) ===")
        for pop, mask in (("all", np.ones(len(m), bool)), ("top5", top)):
            for pos in ["ALL", *config.POSITIONS]:
                pm = mask & ((m.position == pos).to_numpy() if pos != "ALL" else True)
                d, lo, hi, n = paired(m, a, pm)
                ch = (np.abs(a.predicted - m.predicted)[pm] > 1e-9).mean()
                print(f"  {pop:4s} {pos:3s} n={n:5,} off {(m.predicted - m.actual)[pm].abs().mean():.3f} "
                      f"delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}] changed {ch:.0%}")


if __name__ == "__main__":
    main()
