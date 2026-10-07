"""History depth, retrained every week — the way production actually runs.

#27 compared min_season depths by training ONCE on prior seasons plus weeks 1-4
and projecting weeks 5-18. That answers "which window is better for week 5" and
nothing about week 12, because no arm was given weeks 5-11 either. Production
retrains weekly, so by week 12 every arm would hold eleven weeks of the current
season — and the question that matters is whether a deep historical window
dilutes those eleven weeks.

This retrains each arm at each target week on everything played before it, so
the only difference between arms is how far back the history reaches. Reports
per week, so a drift either way over the season is visible rather than pooled
away.

    # the shape of the question, cheaply
    DEPTHS=1,6 WEEKS=5,9,13,17 SEEDS=3 python experiments/history_depth_weekly.py

    # once 2026 has weeks on the board
    TEST_SEASON=2026 WEEKS=5,6,7 SEEDS=5 python experiments/history_depth_weekly.py

Each (depth, week) pair is a full training run, so cost is depths x weeks x
seeds. Start narrow.
"""
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from nfl_projections import config, evaluate
from nfl_projections import dataset as dataset_mod
from nfl_projections import model as model_mod

TEST_SEASON = int(os.environ.get("TEST_SEASON", 2025))
SEEDS = int(os.environ.get("SEEDS", 3))
EPOCHS = int(os.environ.get("EPOCHS", 60))
DEPTHS = [int(d) for d in os.environ.get("DEPTHS", "1,6").split(",")]
WEEKS = [int(w) for w in os.environ.get("WEEKS", "5,9,13,17").split(",")]
OUT = os.environ.get("OUT_CSV")


def main():
    print(f"Building dataset through {config.CURRENT_SEASON}...")
    df = dataset_mod.build_dataset(output_path=None)
    weeks_num = pd.to_numeric(df["week"], errors="coerce")

    rows = []
    for week in WEEKS:
        # Everything played before this week, exactly what a run that morning
        # would hold: prior seasons in full, plus this season up to last week.
        played = df[(df["season"] < TEST_SEASON)
                    | ((df["season"] == TEST_SEASON) & (weeks_num < week))]
        current_rows = int(weeks_num[played[played["season"] == TEST_SEASON].index].notna().sum())

        preds = {}
        for depth in DEPTHS:
            min_season = TEST_SEASON - depth
            supplied = int(weeks_num[played[played["season"] >= min_season].index].notna().sum())
            trained, _ = model_mod.train_ensemble(
                played, n_seeds=SEEDS, epochs=EPOCHS, min_season=min_season,
                plot_path=None, verbose=0)
            res = evaluate.backtest(df, TEST_SEASON, weeks=[week], trained=trained)
            preds[depth] = (supplied, res.set_index("player_id"))

        common = None
        for _, res in preds.values():
            common = res.index if common is None else common.intersection(res.index)

        rec = {"week": week, "current_season_rows": current_rows, "n": len(common)}
        for depth, (supplied, res) in preds.items():
            r = res.loc[common]
            rec[f"mae_d{depth}"] = (r.predicted - r.actual).abs().mean()
            rec[f"rows_d{depth}"] = supplied
        if len(DEPTHS) == 2:
            a, b = DEPTHS
            rec["delta"] = rec[f"mae_d{a}"] - rec[f"mae_d{b}"]
        rows.append(rec)
        print("week %2d (%5d current-season rows, %d players): %s" % (
            week, current_rows, len(common),
            "  ".join(f"depth {d}: {rec[f'mae_d{d}']:.3f}" for d in DEPTHS)
            + (f"  delta {rec['delta']:+.3f}" if "delta" in rec else "")))

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\n" + out.round(3).to_string(index=False))

    n = out["n"].sum()
    print(f"\npooled over weeks {WEEKS} ({n:,} player-weeks, {SEEDS} seeds):")
    for depth in DEPTHS:
        pooled = (out[f"mae_d{depth}"] * out.n).sum() / n
        print(f"  {depth} prior season(s) [2026 equiv min_season "
              f"{TEST_SEASON - depth + (2026 - TEST_SEASON)}]: {pooled:.4f}")
    if "delta" in out:
        a, b = DEPTHS
        print(f"  depth {a} beat depth {b} in {(out.delta < 0).sum()}/{len(out)} weeks")
        early = out[out.week <= out.week.median()]
        late = out[out.week > out.week.median()]
        for label, part in (("earlier weeks", early), ("later weeks", late)):
            if len(part):
                da = (part[f"mae_d{a}"] * part.n).sum() / part.n.sum()
                db = (part[f"mae_d{b}"] * part.n).sum() / part.n.sum()
                print(f"  {label}: depth {a} {da:.3f} vs depth {b} {db:.3f} ({da - db:+.3f})")
    if OUT:
        out.to_csv(OUT, index=False)
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
