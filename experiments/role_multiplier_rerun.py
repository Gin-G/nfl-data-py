"""#19 rerun on clean data: is the depth-rank multiplier double-counting?

The original (2026-08-11) ran inline on the 2024 backtest built from the
name-joined dataset. This reruns it on the clean 2024 board
(`_cache/blitz_board_2024_1s60e.parquet`, rebuilt 2026-11-13 with snaps and
depth charts joined by gsis id, one row per player-week).

Depth rank is the dataset's `depth_team` for that player-week (the 2024-format
chart). The snap-share arm is a reconstruction -- the original did not record
its exact form -- and uses the plainest version: the player's trailing
five-game offensive snap share, games strictly before the week.

    python experiments/role_multiplier_rerun.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from nfl_projections import config, roles  # noqa: E402


def main():
    board = pd.read_parquet(HERE / "_cache" / "blitz_board_2024_1s60e.parquet")
    board = board[board.position.isin(["QB", "RB", "WR", "TE"])].drop_duplicates(
        ["player_id", "week"])
    ds = pd.read_csv(config.DATASET_PATH, low_memory=False,
                     usecols=["player_id", "season", "week", "depth_team",
                              "offensive_snap_pct"])
    ds = ds[(ds.season == 2024) & (ds.week != "AVG")].copy()
    ds["week"] = ds.week.astype(int)
    ds = ds.sort_values(["player_id", "week"])
    ds["snap_trail"] = (ds.groupby("player_id").offensive_snap_pct
                        .transform(lambda s: s.shift().rolling(5, min_periods=1).mean()))
    d = board.merge(ds[["player_id", "week", "depth_team", "snap_trail"]],
                    on=["player_id", "week"], how="left")
    d["rank"] = pd.to_numeric(d.depth_team, errors="coerce")
    print(f"2024 board rows {len(d):,}; with a depth rank {d['rank'].notna().sum():,}")

    print("\nactual / RAW model prediction, by depth rank (does the model already discount?)")
    rows = []
    for pos in ("RB", "WR", "TE", "QB"):
        for r in (2, 3):
            g = d[(d.position == pos) & (d["rank"] == r)]
            if len(g) < 30:
                continue
            rows.append({"pos": pos, "rank": r, "n": len(g),
                         "actual/predicted": round(g.actual.sum() / g.predicted.sum(), 2),
                         "multiplier applied": roles.per_game_role_multiplier(pos, r)})
    print(pd.DataFrame(rows).to_string(index=False))

    d["arm_none"] = d.predicted
    d["arm_rank"] = d.predicted * [roles.per_game_role_multiplier(p, r if pd.notna(r) else None)
                                   for p, r in zip(d.position, d["rank"])]
    snap = (d.snap_trail.fillna(100) / 100.0).clip(0, 1)
    d["arm_snap"] = d.predicted * snap

    print("\nthree arms on the same 2024 predictions")
    out = []
    for arm in ("arm_none", "arm_rank", "arm_snap"):
        e = d[arm] - d.actual
        sp = {pos: round(g[[arm, "actual"]].corr("spearman").iloc[0, 1], 3)
              for pos, g in d.groupby("position")}
        out.append({"arm": arm, "MAE": round(e.abs().mean(), 3), "bias": round(e.mean(), 3),
                    **{f"sp_{k}": v for k, v in sp.items()}})
    print(pd.DataFrame(out).to_string(index=False))

    # paired, clustered by week
    for a, b in (("arm_rank", "arm_none"), ("arm_snap", "arm_none")):
        diff = (d[a] - d.actual).abs() - (d[b] - d.actual).abs()
        w = diff.groupby(d.week).mean()
        se = w.std(ddof=1) / np.sqrt(len(w))
        print(f"  {a} - {b}: MAE {diff.mean():+.3f}  95% CI by week "
              f"[{w.mean() - 1.96 * se:+.3f}, {w.mean() + 1.96 * se:+.3f}]")


if __name__ == "__main__":
    main()
