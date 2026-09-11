"""The gate: does the usage decomposition beat the PRODUCTION model?

Everything so far has been scored against a trailing average, which the shipped
model is not — it has an ML layer, a form blend, a depth-role multiplier and a
team budget on top. "Beats trailing" was always a proxy, and a weak baseline
makes any method look good.

This runs the real thing. ``evaluate.backtest`` trains on seasons strictly
before the target and predicts each week exactly as production does, with
``include_components`` so the per-stat projections a prop settles on come back
rather than just fantasy points.

Compared on identical player-weeks:

  production   the shipped model's component projection
  trailing     the player's own rate over prior weeks (the old baseline)
  usage        own share of team opportunity x team volume x regressed
               efficiency — the decomposition that beat trailing 4/4 and 4/5
  blend        mean of production and usage

Single-seed to keep the run tractable. The module warns that one seed carries a
+/-0.05 MAE lottery, so deltas below that are not readable — the deltas at stake
here are an order of magnitude larger, but it is worth stating.
"""
import os
import pathlib
import numpy as np
import pandas as pd

D = pathlib.Path(os.environ.get("OUT_DIR", "/tmp"))
DATASET = os.environ.get("NFL_DATASET", "/tmp/nfl_dataset.csv")
SEASON = 2025
WEEKS = list(range(1, 9))

MARKETS = {
    "rushing_yards": ("RB", "carries"),
    "receiving_yards": ("WR", "targets"),
    "receiving_yards_te": ("TE", "targets"),
    "receiving_yards_rb": ("RB", "targets"),
}


def production(dataset):
    """Run the shipped pipeline over the target season."""
    from nfl_projections import evaluate

    cache = D / f"prod_backtest_{SEASON}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    res = evaluate.backtest(dataset, SEASON, weeks=WEEKS, epochs=60, n_seeds=1,
                            include_components=True)
    res.to_parquet(cache)
    return res


def usage_projection(dataset, season, week, position, stat, opp_col):
    """own share of team opportunity x team volume x regressed efficiency.

    Built from everything strictly before ``week`` — current season when there
    are two or more weeks of it, otherwise the prior season.
    """
    past = dataset[((dataset.season == season) & (dataset.week < week))
                   | (dataset.season == season - 1)]
    cur = past[past.season == season]
    use = cur if cur.week.nunique() >= 2 else past
    use = use[use.position == position]
    if use.empty:
        return pd.DataFrame(columns=["player_id", "usage", "trailing"])

    for c in (stat, opp_col):
        use = use.assign(**{c: use[c].fillna(0)})

    team_opp = use.groupby("team")[opp_col].sum()
    team_games = use.groupby("team").week.nunique().clip(lower=1)
    opp_pg = (team_opp / team_games).rename("opp_pg")

    p = use.groupby(["player_id", "team"]).agg(
        g=("week", "nunique"), opp=(opp_col, "sum"), v=(stat, "sum")).reset_index()
    p = p[p.g >= 2]
    if p.empty:
        return pd.DataFrame(columns=["player_id", "usage", "trailing"])

    p["trailing"] = p.v / p.g
    lg = use[stat].sum() / max(use[opp_col].sum(), 1)
    K = 40.0
    eff = np.where(p.opp > 0, p.v / p.opp, lg)
    p["eff_reg"] = (eff * p.opp + lg * K) / (p.opp + K)
    p = p.merge(team_opp.rename("team_opp"), left_on="team", right_index=True)
    p = p.merge(opp_pg, left_on="team", right_index=True)
    p["own_share"] = p.opp / p.team_opp
    p["usage"] = p.own_share * p.opp_pg * p.eff_reg
    return p[["player_id", "usage", "trailing"]]


def compare(prod, dataset, label, position, opp_col):
    """One market: production vs trailing vs usage on identical player-weeks."""
    stat = label.replace("_te", "").replace("_rb", "")
    rows = []
    for wk in WEEKS:
        u = usage_projection(dataset, SEASON, wk, position, stat, opp_col)
        if u.empty:
            continue
        pw = prod[(prod["week"] == wk) & (prod["position"] == position)]
        if pw.empty or stat not in pw.columns:
            continue
        act = dataset[(dataset["season"] == SEASON) & (dataset["week"] == wk)
                      & (dataset["position"] == position)][["player_id", stat]]
        act = act.rename(columns={stat: "actual"})
        # "prod" would collide with DataFrame.prod, the product method, and
        # attribute access resolves to the method rather than the column.
        m = (pw[["player_id", stat]].rename(columns={stat: "pmodel"})
             .merge(u, on="player_id").merge(act, on="player_id"))
        m["week"] = wk
        rows.append(m)

    if not rows:
        print(f"{label:22s}{'no overlap':>8}")
        return
    d = pd.concat(rows, ignore_index=True).dropna(
        subset=["pmodel", "usage", "trailing"])
    d["actual"] = d["actual"].fillna(0)
    # Drop rows nobody would price: no projection and no production.
    d = d[(d["pmodel"] > 3) | (d["actual"] > 3)]
    if len(d) < 60:
        print(f"{label:22s}{len(d):>8}   too few")
        return

    d["blend"] = d[["pmodel", "usage"]].mean(axis=1)
    cols = ("pmodel", "trailing", "usage", "blend")
    mae = {c: (d[c] - d["actual"]).abs().mean() for c in cols}
    corr = {c: np.corrcoef(d[c], d["actual"])[0, 1] for c in cols}
    won = sum(1 for wk in d["week"].unique()
              if (d[d["week"] == wk]["usage"] - d[d["week"] == wk]["actual"]).abs().mean()
              < (d[d["week"] == wk]["pmodel"] - d[d["week"] == wk]["actual"]).abs().mean())
    print(f"{label:22s}{len(d):>6}"
          f"{mae['pmodel']:>8.2f}{mae['trailing']:>8.2f}{mae['usage']:>8.2f}{mae['blend']:>8.2f}"
          f"{mae['usage'] - mae['pmodel']:>+11.2f}"
          f"{corr['pmodel']:>8.3f}{corr['usage']:>8.3f}"
          f"{str(won) + '/' + str(d['week'].nunique()):>9}")


def main():
    if not pathlib.Path(DATASET).exists():
        from nfl_projections import dataset as ds_mod
        print("building dataset from nflreadpy...", flush=True)
        ds_mod.build_dataset().to_csv(DATASET, index=False)
    dataset = pd.read_csv(DATASET, low_memory=False)

    # The dataset carries season-average rows with week == "AVG", so the column
    # is object dtype and any numeric comparison on it raises.
    dataset["week"] = pd.to_numeric(dataset["week"], errors="coerce")
    dataset = dataset[dataset["week"].notna()].copy()
    dataset["week"] = dataset["week"].astype(int)
    if "position" not in dataset.columns and "position_x" in dataset.columns:
        dataset = dataset.rename(columns={"position_x": "position"})

    prod = production(dataset)
    prod["week"] = pd.to_numeric(prod["week"], errors="coerce")
    print(f"\nproduction backtest: {len(prod):,} player-weeks, "
          f"weeks {int(prod['week'].min())}-{int(prod['week'].max())}\n", flush=True)

    print(f"{'market':22s}{'n':>6}{'prod':>8}{'trail':>8}{'usage':>8}{'blend':>8}"
          f"{'usage-prod':>11}{'r prod':>8}{'r use':>8}{'wks won':>9}")
    for label, (position, opp_col) in MARKETS.items():
        try:
            compare(prod, dataset, label, position, opp_col)
        except Exception as e:      # one market must not cost the others
            print(f"{label:22s}   FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
