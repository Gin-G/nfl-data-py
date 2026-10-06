"""Does a model trained only on the current season match a fully trained one?

The claim worth testing: by week 5 the season's own four weeks carry everything
the historical seasons taught, so training on 2026 alone should be at least as
good for week 5.

It does not need waiting on 2026. 2025 is a finished season, so train on its
first four weeks ALONE and project weeks 5-18, against a model trained on
2020-2024, and against what production actually does (history plus those four
weeks). Identical evaluation rows for all three, model never sees the weeks it
predicts.

What is NOT at stake: the current season reaches every arm through the FEATURES.
A week 5 projection is built from the player's own last-3 and last-5 games,
which in week 5 are current-season football whatever the training window. Only
the learned mapping from form to next week differs between these arms.

    python experiments/train_window.py              # 2025, single seed
    TEST_SEASON=2024 SEEDS=5 python experiments/train_window.py

Set OUT_DIR to keep the per-player-week predictions.
"""
import os
import pathlib
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from nfl_projections import config, evaluate
from nfl_projections import dataset as dataset_mod
from nfl_projections import model as model_mod

TEST_SEASON = int(os.environ.get("TEST_SEASON", 2025))
CUT = int(os.environ.get("CUT_WEEK", 4))        # weeks in hand for a week-5 model
SEEDS = int(os.environ.get("SEEDS", 1))
EPOCHS = int(os.environ.get("EPOCHS", 60))
OUT_DIR = pathlib.Path(os.environ.get("OUT_DIR", "/tmp"))


def arms(df):
    """(name, training rows, min_season) per training window."""
    weeks = pd.to_numeric(df["week"], errors="coerce")
    history = df[df["season"] < TEST_SEASON]
    current = df[(df["season"] == TEST_SEASON) & (weeks <= CUT)]
    return [
        ("history only (2020-%d)" % (TEST_SEASON - 1), history, config.TRAINING_MIN_SEASON),
        ("current season only (%d wk1-%d)" % (TEST_SEASON, CUT), current, TEST_SEASON),
        ("history + current (production)", pd.concat([history, current], ignore_index=True),
         config.TRAINING_MIN_SEASON),
    ]


def main():
    print(f"Building dataset through {config.CURRENT_SEASON}...")
    df = dataset_mod.build_dataset(output_path=None)
    weeks = list(range(CUT + 1, 19))

    results = {}
    for name, train_df, min_season in arms(df):
        rows = len(train_df[pd.to_numeric(train_df["week"], errors="coerce").notna()])
        print(f"\n=== {name}: {rows:,} game rows, {SEEDS} seed(s) ===")
        trained, _ = model_mod.train_ensemble(
            train_df, n_seeds=SEEDS, epochs=EPOCHS, min_season=min_season,
            plot_path=None, verbose=0,
        )
        res = evaluate.backtest(df, TEST_SEASON, weeks=weeks, trained=trained)
        res["arm"] = name
        results[name] = res

    # Score every arm on the player-weeks ALL of them produced, so no arm is
    # flattered by a different population.
    common = None
    for res in results.values():
        keys = set(zip(res.player_id, res.week))
        common = keys if common is None else (common & keys)
    print(f"\nScored on {len(common):,} player-weeks common to every arm "
          f"({TEST_SEASON} weeks {weeks[0]}-{weeks[-1]})")

    summary, per_pos = [], []
    for name, res in results.items():
        r = res[[k in common for k in zip(res.player_id, res.week)]]
        err = r.predicted - r.actual
        summary.append(dict(arm=name, n=len(r), mae=err.abs().mean(), bias=err.mean(),
                            corr=r[["predicted", "actual"]].corr().iloc[0, 1]))
        for pos, g in r.groupby("position"):
            per_pos.append(dict(arm=name, position=pos, n=len(g),
                                mae=(g.predicted - g.actual).abs().mean()))
        if os.environ.get("OUT_DIR"):
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            r.to_csv(OUT_DIR / f"train_window_{name.split()[0]}.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + pd.DataFrame(summary).round(4).to_string(index=False))
    print("\nby position (MAE):")
    print(pd.DataFrame(per_pos).pivot(index="position", columns="arm", values="mae")
          .round(3).to_string())

    print("\nordering, within position (rank_metrics on season totals):")
    for name, res in results.items():
        r = res[[k in common for k in zip(res.player_id, res.week)]]
        metrics = evaluate.rank_metrics(r, verbose=False)
        if metrics is not None and len(metrics):
            m = metrics.set_index("position")["spearman"].round(3).to_dict()
            print(f"  {name}: {m}")


if __name__ == "__main__":
    main()
