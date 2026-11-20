"""The gate for opportunity share as a board term: fit on the past, score 2025.

`opportunity_share.py` found the board under-uses volume and that a one-term
correction pays (WR 1.4% MAE on the full pool, 2.8% and a real top-24
reordering on the top slice; EXPERIMENTS.md 2026-11-19). Its payment stage was
leave-one-season-out on one-seed boards, so each season's slope was fitted
partly on later seasons. This is the forward version a shipped term would be:

* the slope per position is fitted on 2021-24 board residuals only (one-seed
  boards, the same cache the residual tests use; trained here if missing);
* it is applied, fixed, to the 2025 board at production settings (five seeds;
  reuses topslice_rerun.py's baseline when it is on the volume);
* scored on the full pool and the top slice, for MAE and for top-N overlap.

Two arms per term. `all`: slope fitted on every player, applied to every player.
`top`: slope fitted on the top slice, applied only to it (the slice is chosen
from trailing form before the game, so a production board could do the same).

    TEST_SEASON=2025 SEEDS=5 python experiments/opportunity_gate.py
"""
import os
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import opportunity_share as osh                                      # noqa: E402
from nfl_projections import config, evaluate                         # noqa: E402
from nfl_projections import dataset as dataset_mod                   # noqa: E402

TEST_SEASON = int(os.environ.get("TEST_SEASON", 2025))
FIT_SEASONS = [int(s) for s in os.environ.get("FIT_SEASONS", "2021,2022,2023,2024").split(",")]
SEEDS = int(os.environ.get("SEEDS", 5))
EPOCHS = int(os.environ.get("EPOCHS", 60))
TERMS = {"share_prior": "opp share", "xfp_prior": "xFP"}
OVERLAP_N = {"QB": 12, "RB": 12, "WR": 24, "TE": 12}
CACHE = HERE / "_cache"
BOOT = 2000


def dataset():
    try:
        return dataset_mod.load_dataset()
    except FileNotFoundError:
        print("building the dataset ...")
        os.makedirs(os.path.dirname(config.DATASET_PATH) or ".", exist_ok=True)
        return dataset_mod.build_dataset(output_path=config.DATASET_PATH)


def board(ds, season, seeds, path):
    if path.exists():
        print(f"  {season}: cached {path.name}")
        return pd.read_parquet(path)
    print(f"  {season}: backtesting ({seeds} seed(s)) ...")
    res = evaluate.backtest(ds, season, epochs=EPOCHS, n_seeds=seeds)
    CACHE.mkdir(exist_ok=True)
    res.to_parquet(path)
    return res


def prepare(b, ds, d):
    b = b.drop_duplicates(["player_id", "season", "week"])
    b = b[b["week"] <= 18].reset_index(drop=True)
    b = b.merge(d[["player_id", "season", "week", "prior_games", *TERMS]]
                .drop_duplicates(["player_id", "season", "week"]),
                on=["player_id", "season", "week"], how="left")
    b["form"] = evaluate.trailing_form(ds, b)
    b["top"] = evaluate.top_slice(b, "form")
    b["resid"] = b["actual"] - b["predicted"]
    b["usable"] = b["prior_games"].fillna(0) >= osh.MIN_PRIOR_GAMES
    return b


def slopes(fit, term, only_top):
    out = {}
    for pos in osh.POSITIONS:
        g = fit[(fit.position == pos) & fit.usable & (fit.top if only_top else True)]
        g = g[["resid", term]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(g) < 300:
            continue
        X = np.column_stack([np.ones(len(g)), g[term].values])
        beta, *_ = np.linalg.lstsq(X, g["resid"].values, rcond=None)
        out[pos] = (beta[1], g[term].mean())      # slope, and the centring
    return out


def apply(test, term, beta, only_top):
    adj = np.zeros(len(test))
    for pos, (slope, centre) in beta.items():
        m = ((test.position == pos) & test.usable & test[term].notna()
             & (test.top if only_top else True)).to_numpy()
        adj[m] = slope * (test.loc[m, term].to_numpy() - centre)
    return test["predicted"].to_numpy() + adj


def _boot(groups, stat, rng):
    idx = rng.integers(0, len(groups), size=(BOOT, len(groups)))
    return np.percentile([stat(np.concatenate([groups[i] for i in row])) for row in idx],
                         [2.5, 97.5])


def score(test, fixed, mask, label):
    """MAE gain (board minus corrected; positive = corrected better), paired CI."""
    e0 = (test.predicted - test.actual).abs().to_numpy()
    e1 = np.abs(fixed - test.actual.to_numpy())
    rows = np.flatnonzero(mask)
    wk = test.week.to_numpy()[rows]
    groups = [rows[wk == w] for w in np.unique(wk)]
    rng = np.random.default_rng(20261119)
    lo, hi = _boot(groups, lambda r: e0[r].mean() - e1[r].mean(), rng)
    return f"{label:5s} n={len(rows):5,} board {e0[rows].mean():.3f} gain {e0[rows].mean() - e1[rows].mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"


def overlap(test, fixed, pos):
    """Top-N overlap with the week's actual best N, over the whole position pool."""
    n = OVERLAP_N[pos]
    t = test.assign(fixed=fixed)
    t = t[t.position == pos]
    rows = []
    for w, g in t.groupby("week"):
        if len(g) < 2 * n:
            continue
        act = set(g.nlargest(n, "actual").index)
        rows.append((len(act & set(g.nlargest(n, "predicted").index)) / n,
                     len(act & set(g.nlargest(n, "fixed").index)) / n))
    o = np.array(rows)
    d = o[:, 1] - o[:, 0]
    rng = np.random.default_rng(7)
    draws = d[rng.integers(0, len(d), size=(BOOT, len(d)))].mean(1)
    return (f"top-{n} overlap {o[:, 0].mean():.4f} -> {o[:, 1].mean():.4f} "
            f"({d.mean():+.4f} [{np.percentile(draws, 2.5):+.4f}, {np.percentile(draws, 97.5):+.4f}], "
            f"{int((d > 0).sum())} up / {int((d < 0).sum())} down of {len(d)} wks)")


def main():
    ds = dataset()
    d = osh.weeks()
    d = osh.priors(d, osh.points_per_opportunity(d))
    print(f"fit on {FIT_SEASONS} (1-seed boards), test {TEST_SEASON} ({SEEDS}-seed board)")
    fit = pd.concat([board(ds, s, 1, CACHE / f"blitz_board_{s}_1s{EPOCHS}e.parquet")
                     for s in FIT_SEASONS], ignore_index=True)
    fit = prepare(fit, ds, d)
    test = prepare(board(ds, TEST_SEASON, SEEDS,
                         CACHE / f"topslice_{TEST_SEASON}_baseline_{SEEDS}s{EPOCHS}e.parquet"),
                   ds, d)
    allm = np.ones(len(test), bool)
    topm = test["top"].to_numpy()
    for term, label in TERMS.items():
        for arm, only_top in (("all", False), ("top", True)):
            beta = slopes(fit, term, only_top)
            fixed = apply(test, term, beta, only_top)
            print(f"\n=== {label}, arm {arm}: slopes "
                  + ", ".join(f"{p} {s:+.2f}" for p, (s, _) in beta.items()))
            for pos in ["ALL", *osh.POSITIONS]:
                pm = (test.position == pos).to_numpy() if pos != "ALL" else allm
                print(f"  {pos:3s} " + score(test, fixed, pm & allm, "all")
                      + " | " + score(test, fixed, pm & topm, "top"))
                if pos != "ALL":
                    print(f"      {overlap(test, fixed, pos)}")


if __name__ == "__main__":
    main()
