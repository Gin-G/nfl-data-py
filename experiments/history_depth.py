"""How far back should training reach? TRAINING_MIN_SEASON as a dial.

#26 killed "current season only" (+0.097 MAE). This is the different claim it
did not test: recent football may matter MORE without older football being
worthless. `config.TRAINING_MIN_SEASON` is already the knob — 2020 today.

Tested on finished football, so the windows shift back a year: a model for 2026
week 5 with min_season 2025 has one prior season plus the current four weeks,
which is min_season 2024 tested on 2025. Arms are labelled by how many prior
seasons they keep, with the 2026 equivalent alongside.

Every arm trains on the same shape as production — prior seasons plus the test
season's first four weeks — and is scored on identical player-weeks of weeks
5-18. Five seeds by default: #26 is the cautionary tale, where one seed
understated a 0.1 MAE gap as 0.066.

    python experiments/history_depth.py                 # 2025, 5 seeds
    TEST_SEASON=2024 SEEDS=5 python experiments/history_depth.py
"""
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from nfl_projections import config, evaluate
from nfl_projections import dataset as dataset_mod
from nfl_projections import model as model_mod

TEST_SEASON = int(os.environ.get("TEST_SEASON", 2025))
CUT = int(os.environ.get("CUT_WEEK", 4))
SEEDS = int(os.environ.get("SEEDS", 5))
EPOCHS = int(os.environ.get("EPOCHS", 60))
# Prior seasons kept, newest-first: 1 -> just last season, 6 -> today's default.
DEPTHS = [int(d) for d in os.environ.get("DEPTHS", "1,2,3,6").split(",")]


def main():
    print(f"Building dataset through {config.CURRENT_SEASON}...")
    df = dataset_mod.build_dataset(output_path=None)
    weeks_num = pd.to_numeric(df["week"], errors="coerce")
    train_pool = df[(df["season"] < TEST_SEASON)
                    | ((df["season"] == TEST_SEASON) & (weeks_num <= CUT))]
    weeks = list(range(CUT + 1, 19))

    results = {}
    for depth in DEPTHS:
        min_season = TEST_SEASON - depth
        # What the same window would be called in a 2026 run.
        equivalent = min_season + (2026 - TEST_SEASON)
        label = f"{depth} prior season(s): min_season {min_season} (2026 equiv {equivalent})"
        rows = int(weeks_num[train_pool[train_pool["season"] >= min_season].index].notna().sum())
        print(f"\n=== {label} — {rows:,} game rows, {SEEDS} seed(s) ===")
        trained, _ = model_mod.train_ensemble(
            train_pool, n_seeds=SEEDS, epochs=EPOCHS, min_season=min_season,
            plot_path=None, verbose=0)
        res = evaluate.backtest(df, TEST_SEASON, weeks=weeks, trained=trained)
        results[label] = (rows, res)

    common = None
    for _, res in results.values():
        keys = set(zip(res.player_id, res.week))
        common = keys if common is None else (common & keys)
    print(f"\nScored on {len(common):,} player-weeks common to every arm "
          f"({TEST_SEASON} weeks {weeks[0]}-{weeks[-1]}, {SEEDS} seeds)")

    summary, per_pos = [], []
    for label, (rows, res) in results.items():
        r = res[[k in common for k in zip(res.player_id, res.week)]]
        err = r.predicted - r.actual
        summary.append(dict(window=label, rows=rows, n=len(r), mae=err.abs().mean(),
                            bias=err.mean(),
                            corr=r[["predicted", "actual"]].corr().iloc[0, 1]))
        for pos, g in r.groupby("position"):
            per_pos.append(dict(window=label.split(":")[0], position=pos,
                                mae=(g.predicted - g.actual).abs().mean()))

    out = pd.DataFrame(summary).sort_values("mae")
    best = out.iloc[0]
    out["vs_best"] = out.mae - best.mae
    pd.set_option("display.width", 240)
    print("\n" + out.round(4).to_string(index=False))
    print(f"\nbest: {best.window} at {best.mae:.4f}")
    print("(deltas under ~0.05 are inside the seed lottery even at 5 seeds —"
          " treat a close finish as a tie)")
    print("\nby position (MAE):")
    print(pd.DataFrame(per_pos).pivot(index="position", columns="window", values="mae")
          .round(3).to_string())

    print("\nordering, within position (spearman on weeks 5-18 totals):")
    for label, (_, res) in results.items():
        r = res[[k in common for k in zip(res.player_id, res.week)]]
        metrics = evaluate.rank_metrics(r, verbose=False)
        if metrics is not None and len(metrics):
            print(f"  {label.split(':')[0]}: "
                  f"{metrics.set_index('position')['spearman'].round(3).to_dict()}")


if __name__ == "__main__":
    main()
