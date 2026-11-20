"""Per-position form-blend weights, swept on one raw model (no retraining).

topslice_rerun.py found that dropping the form blend helps WR/TE at the top of
the board and hurts QB (EXPERIMENTS.md, 2026-11-19). The blend weights in
blend.py were fitted in 2025 on the full pool. This sweeps them again.

The raw network output is the cached `no_blend` board. Every weight is applied
to that same output with blend.blend_value, so the comparison carries no
training noise. The blend partner is the dataset's own
`fanduel_fantasy_points_roll5` on each player's latest prior row, the exact
number backtest() blends with.

Each weight is scored on three populations:
    all      every player-week (the standard yardstick)
    top5     top slice chosen by trailing-5 form (biased against the blend)
    top16    top slice chosen by trailing-16 form (not the blend partner)

Fit-on-one/score-on-the-other: the weight picked on 2024 is scored on 2025 and
vice versa, so the reported gain is out of sample.

    SEASONS=2024,2025 SEEDS=5 python experiments/blend_sweep.py
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
from topslice_reslice import prior_season_ppg                        # noqa: E402
from nfl_projections import blend, config, evaluate                  # noqa: E402
from nfl_projections import dataset as dataset_mod                   # noqa: E402

SEASONS = [int(s) for s in os.environ.get("SEASONS", "2024,2025").split(",")]
SEEDS = int(os.environ.get("SEEDS", 5))
GRID = [round(w, 2) for w in np.arange(0.3, 1.01, 0.1)]
POPS = ("all", "top5", "top16", "prior")
TIER = "top16"     # the ex-ante tier the tiered arm re-weights


def roll5(df, board):
    """fanduel_fantasy_points_roll5 on each player's latest row before the week."""
    from nfl_projections import features
    hist = evaluate.regular_games(features.prepare_prediction_base(df))   # as backtest() builds it
    h = hist[["player_id", "season", "week", blend.RECENT_FORM_COL]].dropna(
        subset=[blend.RECENT_FORM_COL]).drop_duplicates(["player_id", "season", "week"]).copy()
    h["_k"] = h["season"].astype(int) * 100 + h["week"].astype(int)
    r = board[["player_id", "season", "week"]].copy()
    r["_k"] = r["season"].astype(int) * 100 + r["week"].astype(int) - 1
    r["_i"] = np.arange(len(r))
    out = pd.merge_asof(r.sort_values("_k"), h[["player_id", "_k", blend.RECENT_FORM_COL]]
                        .sort_values("_k"), on="_k", by="player_id", direction="backward")
    return out.sort_values("_i")[blend.RECENT_FORM_COL].to_numpy()


def blended(raw, form, pos, w):
    """blend_value with this position's model weight set to w."""
    form_w = np.where(np.isnan(form), 0.0, 1.0 - w)
    return (1 - form_w) * raw + form_w * np.nan_to_num(form)


def load(season):
    path = CACHE / f"topslice_{season}_no_blend_{SEEDS}s{EPOCHS}e.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path.name}; run topslice_rerun.py ARMS=no_blend first")
    b = pd.read_parquet(path).drop_duplicates(["player_id", "season", "week"])
    return b[b["week"] <= 18].reset_index(drop=True)


def main():
    df = dataset_mod.build_dataset(output_path=None)
    boards = {}
    for season in SEASONS:
        b = load(season)
        b["raw"] = b["predicted"]
        b["roll5"] = roll5(df, b)
        b["form16"] = evaluate.trailing_form(df, b, n_games=16)
        masks = {"all": np.ones(len(b), bool),
                 "top5": evaluate.top_slice(b, evaluate.trailing_form(df, b)),
                 "top16": evaluate.top_slice(b, evaluate.trailing_form(df, b, n_games=16)),
                 "prior": evaluate.top_slice(b, prior_season_ppg(df, b))}
        boards[season] = (b, masks)

    # MAE per season x position x population x weight
    rows = []
    for season, (b, masks) in boards.items():
        for pos in config.POSITIONS:
            pm = (b["position"] == pos).to_numpy()
            for w in GRID:
                p = blended(b["raw"].to_numpy(), b["roll5"].to_numpy(), pos, w)
                err = np.abs(p - b["actual"].to_numpy())
                for pop in POPS:
                    m = pm & masks[pop]
                    rows.append(dict(season=season, pos=pos, pop=pop, w=w, mae=err[m].mean(), n=m.sum()))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for pop in POPS:
        print(f"\n=== MAE by model weight, population {pop} (current: "
              + ", ".join(f"{p} {w}" for p, w in blend.DEFAULT_MODEL_WEIGHTS.items()) + ") ===")
        print(t[t["pop"] == pop].pivot_table(index=["season", "pos"], columns="w", values="mae")
              .round(3).to_string())

    # out of sample: pick w per position on one season (full pool AND top16
    # separately), score it on the other against the current weights
    if len(SEASONS) < 2:
        return
    print("\n=== TIERED, out of sample: top-tier (trailing-16 form) rows get a model "
          "weight chosen on the other season's tier; every other row keeps the current weight ===")
    for test in SEASONS:
        fit = [s for s in SEASONS if s != test]
        ft = t[(t["season"].isin(fit)) & (t["pop"] == TIER)]
        chosen = ft.groupby(["pos", "w"])["mae"].mean().unstack().idxmin(axis=1).to_dict()
        b, masks = boards[test]
        raw, form, act = b["raw"].to_numpy(), b["roll5"].to_numpy(), b["actual"]
        cur = np.zeros(len(b))
        for pos in config.POSITIONS:
            pm = (b["position"] == pos).to_numpy()
            cur[pm] = blended(raw, form, pos, blend.DEFAULT_MODEL_WEIGHTS[pos])[pm]
        new = cur.copy()
        for pos in config.POSITIONS:
            pm = (b["position"] == pos).to_numpy() & masks[TIER]
            new[pm] = blended(raw, form, pos, chosen[pos])[pm]
        base, arm = b.assign(predicted=cur), b.assign(predicted=new)
        print(f"  tier weights picked on {fit}: "
              + ", ".join(f"{p} {w}" for p, w in chosen.items()) + f"  -> {test}")
        for pop in POPS:
            for pos in ["ALL", *config.POSITIONS]:
                m = masks[pop] & ((b["position"] == pos).to_numpy() if pos != "ALL" else True)
                d, lo, hi, n = paired(base, arm, m)
                print(f"    {pop:5s} {pos:3s} n={n:5,} cur {(base.predicted - act)[m].abs().mean():.3f} "
                      f"delta {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                      f"corr {base.predicted[m].corr(act[m]):.3f}->{arm.predicted[m].corr(act[m]):.3f}")

    # The shippable shape: the tier is a fixed trailing-16 points threshold per
    # position (the fit season's mean weekly N-th best form16), not a rank, so it
    # does not depend on which players the live pool happens to hold.
    print("\n=== TIERED BY THRESHOLD, out of sample (threshold and weights both fit on the other season) ===")
    for test in SEASONS:
        fit = [s for s in SEASONS if s != test]
        thr, wts = {}, {}
        for pos in config.POSITIONS:
            vals, scores = [], {}
            for s in fit:
                fb, _ = boards[s]
                g = fb[fb["position"] == pos]
                vals += [x.nlargest(evaluate.TOP_N[pos]).iloc[-1]
                         for _, x in g.groupby("week")["form16"] if len(x) >= evaluate.TOP_N[pos]]
            thr[pos] = float(np.mean(vals))
            for w in GRID:
                errs = []
                for s in fit:
                    fb, _ = boards[s]
                    m = ((fb["position"] == pos) & (fb["form16"] >= thr[pos])).to_numpy()
                    p = blended(fb["raw"].to_numpy(), fb["roll5"].to_numpy(), pos, w)
                    errs.append(np.abs(p - fb["actual"].to_numpy())[m])
                scores[w] = np.concatenate(errs).mean()
            wts[pos] = min(scores, key=scores.get)
        b, masks = boards[test]
        raw, form, act = b["raw"].to_numpy(), b["roll5"].to_numpy(), b["actual"]
        cur = np.zeros(len(b))
        new = np.zeros(len(b))
        for pos in config.POSITIONS:
            pm = (b["position"] == pos).to_numpy()
            cur[pm] = blended(raw, form, pos, blend.DEFAULT_MODEL_WEIGHTS[pos])[pm]
            hi = pm & (b["form16"] >= thr[pos]).to_numpy()
            new[pm] = cur[pm]
            new[hi] = blended(raw, form, pos, wts[pos])[hi]
        base, arm = b.assign(predicted=cur), b.assign(predicted=new)
        print(f"  fit on {fit}: " + ", ".join(f"{p} >= {thr[p]:.1f} pts -> w {wts[p]}" for p in config.POSITIONS)
              + f"  -> {test}  (tier share {np.mean(b['form16'].to_numpy() >= b['position'].map(thr).to_numpy()):.0%})")
        for pop in POPS:
            for pos in ["ALL", *config.POSITIONS]:
                m = masks[pop] & ((b["position"] == pos).to_numpy() if pos != "ALL" else True)
                d, lo, hi_, n = paired(base, arm, m)
                print(f"    {pop:5s} {pos:3s} n={n:5,} cur {(base.predicted - act)[m].abs().mean():.3f} "
                      f"delta {d:+.3f} [{lo:+.3f}, {hi_:+.3f}]  "
                      f"corr {base.predicted[m].corr(act[m]):.3f}->{arm.predicted[m].corr(act[m]):.3f}")

    print("\n=== out of sample: weight chosen on the other season vs current weights ===")
    for pick_pop in ("all", "top16"):
        for test in SEASONS:
            fit = [s for s in SEASONS if s != test]
            ft = t[(t["season"].isin(fit)) & (t["pop"] == pick_pop)]
            chosen = ft.groupby(["pos", "w"])["mae"].mean().unstack().idxmin(axis=1).to_dict()
            b, masks = boards[test]
            raw, form, act = b["raw"].to_numpy(), b["roll5"].to_numpy(), b["actual"]
            cur = np.zeros(len(b))
            new = np.zeros(len(b))
            for pos in config.POSITIONS:
                pm = (b["position"] == pos).to_numpy()
                cur[pm] = blended(raw, form, pos, blend.DEFAULT_MODEL_WEIGHTS[pos])[pm]
                new[pm] = blended(raw, form, pos, chosen[pos])[pm]
            base = b.assign(predicted=cur)
            arm = b.assign(predicted=new)
            print(f"  picked on {fit} {pick_pop}: "
                  + ", ".join(f"{p} {w}" for p, w in chosen.items()) + f"  -> {test}")
            for pop in POPS:
                for pos in ["ALL", *config.POSITIONS]:
                    m = masks[pop] & ((b["position"] == pos).to_numpy() if pos != "ALL" else True)
                    d, lo, hi, n = paired(base, arm, m)
                    print(f"    {pop:5s} {pos:3s} n={n:5,} cur {(base.predicted - act)[m].abs().mean():.3f} "
                          f"delta {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
