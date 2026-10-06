"""The current-season-only proposal as it would actually run, week by week.

`train_window.py` trains once on weeks 1-4 and projects the rest of the season,
which understates the idea: production retrains every week, so by week 12 a
current-season-only model would have eleven weeks behind it, not four. This
retrains for every target week on that season's PRIOR weeks only — the real
shape of "train on 2026 alone" — and scores it against one fully trained model
on identical player-weeks.

Cheap, because that is the whole point of the proposal: a few thousand rows
trains in seconds, where the historical model is most of an hour.

    python experiments/train_window_expanding.py          # 2025
    TEST_SEASON=2024 SEEDS=5 python experiments/train_window_expanding.py
"""
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from nfl_projections import config, evaluate
from nfl_projections import dataset as dataset_mod
from nfl_projections import model as model_mod

TEST_SEASON = int(os.environ.get("TEST_SEASON", 2025))
FIRST = int(os.environ.get("FIRST_WEEK", 5))     # first week a 4-week model exists
LAST = int(os.environ.get("LAST_WEEK", 18))
SEEDS = int(os.environ.get("SEEDS", 1))
EPOCHS = int(os.environ.get("EPOCHS", 60))


def main():
    print(f"Building dataset through {config.CURRENT_SEASON}...")
    df = dataset_mod.build_dataset(output_path=None)
    weeks_num = pd.to_numeric(df["week"], errors="coerce")

    print(f"\n=== fully trained once on seasons < {TEST_SEASON} ===")
    history = df[df["season"] < TEST_SEASON]
    full, _ = model_mod.train_ensemble(
        history, n_seeds=SEEDS, epochs=EPOCHS,
        min_season=config.TRAINING_MIN_SEASON, plot_path=None, verbose=0)

    rows = []
    for week in range(FIRST, LAST + 1):
        current = df[(df["season"] == TEST_SEASON) & (weeks_num < week)]
        n_rows = int(weeks_num[current.index].notna().sum())
        fresh, _ = model_mod.train_ensemble(
            current, n_seeds=SEEDS, epochs=EPOCHS, min_season=TEST_SEASON,
            plot_path=None, verbose=0)

        got = {}
        for name, trained in (("current", fresh), ("full", full)):
            res = evaluate.backtest(df, TEST_SEASON, weeks=[week], trained=trained)
            got[name] = res.set_index("player_id")
        common = got["current"].index.intersection(got["full"].index)
        rec = dict(week=week, train_rows=n_rows, n=len(common))
        for name, res in got.items():
            r = res.loc[common]
            rec[f"mae_{name}"] = (r.predicted - r.actual).abs().mean()
            rec[f"bias_{name}"] = (r.predicted - r.actual).mean()
        rec["delta"] = rec["mae_current"] - rec["mae_full"]
        rows.append(rec)
        print(f"week {week:2d}: {n_rows:5,} current-season rows | "
              f"current {rec['mae_current']:.3f} vs full {rec['mae_full']:.3f} "
              f"({rec['delta']:+.3f}) on {len(common)} players")

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + out.round(3).to_string(index=False))
    n = out["n"].sum()
    pooled_cur = (out.mae_current * out.n).sum() / n
    pooled_full = (out.mae_full * out.n).sum() / n
    print(f"\npooled over weeks {FIRST}-{LAST} ({n:,} player-weeks, {SEEDS} seed(s)): "
          f"current-season-only {pooled_cur:.3f} vs fully trained {pooled_full:.3f} "
          f"({pooled_cur - pooled_full:+.3f})")
    print(f"weeks where current-season-only won: {(out.delta < 0).sum()}/{len(out)}")
    early, late = out[out.week <= 10], out[out.week > 10]
    for label, part in (("weeks 5-10", early), ("weeks 11-18", late)):
        if len(part):
            m = (part.mae_current * part.n).sum() / part.n.sum()
            f = (part.mae_full * part.n).sum() / part.n.sum()
            print(f"  {label}: {m:.3f} vs {f:.3f} ({m - f:+.3f})")


if __name__ == "__main__":
    main()
