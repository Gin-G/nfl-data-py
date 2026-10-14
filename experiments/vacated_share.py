"""Is vacated share actually handled by renormalization, or does it need modelling?

`nfl_projections/shares.py` says, in its own design notes, that vacated share is
"handled implicitly: a new depth chart re-ranks the returners/newcomers and
shares renormalize within the team, so a departed starter's touches flow to
whoever now sits at that rank."

That is an assumption, not a measurement, and it is the one assumption in the
share model that has never been tested. It also encodes a specific claim about
how football works: that a vacated role transfers to the next man at that rank.
The alternative is that it does not transfer cleanly — a committee forms, a
free-agent signing takes it, or the returning number two gets a bump smaller
than the departure was worth.

This matters more than the context features that keep failing, because share IS
the quantity that moves projections (EXPERIMENTS.md, repeatedly). A share
feature that works is worth more than any matchup rating.

The test. For every team-season, compute what share left: the prior-year carry
and target share of players who were on the team last year and are not this
year. Then, for returning players, predict this season's share with and without
it, leave-one-season-out. If the implicit handling is sufficient, knowing how
much share was vacated adds nothing.

Share tables are built here from player stats alone, so snap share is absent and
depth rank falls back to the touch-based ordering `_usage_rank` already uses as
its secondary key. That weakens rank slightly for everyone, baseline included.
"""
import pathlib

import numpy as np
import pandas as pd

from nfl_projections import shares
from nfl_projections.scoring import fanduel_points

CACHE = pathlib.Path(__file__).parent / "_cache"
SEASONS = tuple(range(2018, 2026))
BIG_VACANCY = 0.15          # a team losing 15%+ of a pool is a real opening


def weekly(seasons=SEASONS) -> pd.DataFrame:
    """Weekly skill-position stats with FanDuel points, cached."""
    path = CACHE / f"vac_weekly_{seasons[0]}_{seasons[-1]}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    import nflreadpy as nfl
    frames = []
    for season in seasons:
        d = nfl.load_player_stats(seasons=[season]).to_pandas()
        d = d[(d.season_type == "REG") & d.position.isin(shares.SKILL)]
        keep = ["season", "week", "player_id", "player_display_name", "position",
                "team", "carries", "targets", "passing_yards", "passing_tds",
                "passing_interceptions", "rushing_yards", "rushing_tds",
                "receptions", "receiving_yards", "receiving_tds"]
        frames.append(d[[c for c in keep if c in d.columns]])
    d = pd.concat(frames, ignore_index=True)
    d["fanduel_fantasy_points"] = fanduel_points(
        passing_yards=d.get("passing_yards", 0), passing_tds=d.get("passing_tds", 0),
        interceptions=d.get("passing_interceptions", 0),
        rushing_yards=d.get("rushing_yards", 0), rushing_tds=d.get("rushing_tds", 0),
        receptions=d.get("receptions", 0), receiving_yards=d.get("receiving_yards", 0),
        receiving_tds=d.get("receiving_tds", 0))
    d = d.rename(columns={"player_display_name": "player_name"})
    d["offensive_snap_pct"] = np.nan
    CACHE.mkdir(parents=True, exist_ok=True)
    d.to_parquet(path)
    return d


def vacancies(st: pd.DataFrame) -> pd.DataFrame:
    """Per (season, team): the share that left with departing players.

    A player counts as departed if he was on this team last season and is not on
    it this season — which covers retirement, free agency and being cut, and
    does not care which, because the opening is the same either way.
    """
    rows = []
    for season in sorted(st.season.unique())[1:]:
        prev = st[st.season == season - 1]
        now = st[st.season == season]
        present = set(zip(now.team, now.player_id))
        for team, g in prev.groupby("team"):
            gone = g[[(team, pid) not in present for pid in g.player_id]]
            rows.append({"season": season, "team": team,
                         "vacated_carry": float(gone.carry_share.sum()),
                         "vacated_target": float(gone.target_share.sum()),
                         "departures": int(len(gone))})
    return pd.DataFrame(rows)


def returners(st: pd.DataFrame, vac: pd.DataFrame) -> pd.DataFrame:
    """One row per returning player-season: his prior share, rank, and the team's vacancy."""
    st = shares._entering_rank(st)
    rows = []
    for season in sorted(st.season.unique())[1:]:
        prev = st[st.season == season - 1].set_index(["team", "player_id"])
        now = st[st.season == season]
        for _, p in now.iterrows():
            key = (p.team, p.player_id)
            if key not in prev.index:
                continue        # a mover or a rookie: a different problem
            before = prev.loc[key]
            if isinstance(before, pd.DataFrame):
                before = before.iloc[0]
            rows.append({
                "season": season, "team": p.team, "position": p.position,
                "player_id": p.player_id, "player_name": p.player_name,
                "rank": min(int(p.entering_rank), 5),
                "prior_carry": float(before.carry_share),
                "prior_target": float(before.target_share),
                "carry_share": float(p.carry_share),
                "target_share": float(p.target_share),
            })
    out = pd.DataFrame(rows).merge(vac, on=["season", "team"], how="left")
    return out.dropna(subset=["vacated_carry", "vacated_target"]).reset_index(drop=True)


def loso(m, features, target):
    preds = np.full(len(m), np.nan)
    for season in sorted(m.season.unique()):
        train, test = m.season != season, m.season == season
        if train.sum() < 50 or test.sum() == 0:
            continue
        X = np.column_stack([np.ones(train.sum())] + [m.loc[train, f] for f in features])
        beta, *_ = np.linalg.lstsq(X, m.loc[train, target], rcond=None)
        Xt = np.column_stack([np.ones(test.sum())] + [m.loc[test, f] for f in features])
        preds[test.to_numpy()] = Xt @ beta
    ok = ~np.isnan(preds)
    return (float(np.abs(preds[ok] - m[target][ok]).mean()),
            float(np.corrcoef(preds[ok], m[target][ok])[0, 1]), int(ok.sum()))


def coefficient(m, features, target):
    """Pooled OLS coefficients, to read the direction and size of the effect."""
    X = np.column_stack([np.ones(len(m))] + [m[f] for f in features])
    beta, *_ = np.linalg.lstsq(X, m[target], rcond=None)
    return dict(zip(["intercept"] + list(features), np.round(beta, 4)))


if __name__ == "__main__":
    st = shares.season_share_table(weekly())
    st = st[st.games >= 4]
    vac = vacancies(st)
    print(f"{len(st):,} player-seasons, {len(vac)} team-seasons with a prior year")
    print(f"vacated carry share: mean {vac.vacated_carry.mean():.3f}, "
          f"median {vac.vacated_carry.median():.3f}, "
          f"max {vac.vacated_carry.max():.3f}")
    print(f"vacated target share: mean {vac.vacated_target.mean():.3f}, "
          f"median {vac.vacated_target.median():.3f}\n")

    r = returners(st, vac)
    print(f"{len(r):,} returning player-seasons\n")

    for pool, prior_col, vac_col in (("carry_share", "prior_carry", "vacated_carry"),
                                     ("target_share", "prior_target", "vacated_target")):
        print(f"=== {pool}")
        flat = float(np.abs(r[pool] - r[pool].mean()).mean())
        print(f"  league mean alone            MAE {flat:.4f}")
        tests = [
            ("own prior share", [prior_col]),
            ("own prior + rank", [prior_col, "rank"]),
            ("own prior + rank + VACATED", [prior_col, "rank", vac_col]),
            ("+ vacated x prior (interaction)", [prior_col, "rank", vac_col, "_inter"]),
        ]
        r["_inter"] = r[vac_col] * r[prior_col]
        for label, feats in tests:
            mae, corr, n = loso(r, feats, pool)
            print(f"  {label:32s}MAE {mae:.4f}  corr {corr:.3f}  n {n}")
        print("  coefficients:", coefficient(r, [prior_col, "rank", vac_col], pool))

        big = r[r[vac_col] >= BIG_VACANCY]
        small = r[r[vac_col] < BIG_VACANCY]
        print(f"\n  where the opening is real ({BIG_VACANCY:.0%}+ of the pool vacated): "
              f"{len(big):,} rows vs {len(small):,}")
        for label, sub in (("big vacancy", big), ("small vacancy", small)):
            base_mae, _, _ = loso(sub, [prior_col, "rank"], pool)
            vac_mae, _, _ = loso(sub, [prior_col, "rank", vac_col], pool)
            print(f"    {label:16s}baseline {base_mae:.4f} -> with vacated {vac_mae:.4f} "
                  f"({vac_mae - base_mae:+.4f})")
        print()


def destination(st: pd.DataFrame, vac: pd.DataFrame) -> pd.DataFrame:
    """Where a team's vacated share actually ended up.

    Splits each team-season's pool between players who were on the roster last
    year and players who were not, so the question "does the next man up get
    the touches, or does a new body get them?" has a number rather than an
    assumption.
    """
    rows = []
    for season in sorted(st.season.unique())[1:]:
        prev = st[st.season == season - 1]
        now = st[st.season == season]
        prev_by_team = {t: set(g.player_id) for t, g in prev.groupby("team")}
        for team, g in now.groupby("team"):
            was_here = prev_by_team.get(team)
            if not was_here:
                continue
            new = g[[pid not in was_here for pid in g.player_id]]
            old = g[[pid in was_here for pid in g.player_id]]
            rows.append({
                "season": season, "team": team,
                "new_carry": float(new.carry_share.sum()),
                "new_target": float(new.target_share.sum()),
                "ret_carry": float(old.carry_share.sum()),
                "ret_target": float(old.target_share.sum()),
                "newcomers": int(len(new)),
            })
    return pd.DataFrame(rows).merge(vac, on=["season", "team"], how="inner")


if __name__ == "__main__" and True:
    print("\n\n=== WHERE DOES THE VACATED SHARE GO?")
    dest = destination(st, vac)
    for pool, vcol, ncol in (("carries", "vacated_carry", "new_carry"),
                             ("targets", "vacated_target", "new_target")):
        d = dest[dest[vcol] > 0.05]
        absorbed = d[ncol] / d[vcol]
        r = float(np.corrcoef(d[vcol], d[ncol])[0, 1])
        print(f"\n  {pool}: {len(d)} team-seasons that vacated 5%+ of the pool")
        print(f"    share now held by players who were NOT here last year: "
              f"mean {d[ncol].mean():.3f} against vacated {d[vcol].mean():.3f}")
        print(f"    fraction of the vacancy taken by newcomers: "
              f"median {absorbed.median():.2f}, mean {absorbed.mean():.2f}")
        print(f"    correlation, vacated -> newcomer share: {r:.3f}")
        band = pd.cut(d[vcol], [0.05, 0.2, 0.4, 0.6, 1.01],
                      labels=["5-20%", "20-40%", "40-60%", "60%+"])
        print(d.groupby(band, observed=True).agg(
            teams=(ncol, "size"), vacated=(vcol, "mean"),
            newcomer_share=(ncol, "mean")).round(3).to_string())


def movers(st: pd.DataFrame, vac: pd.DataFrame) -> pd.DataFrame:
    """Players on a new team, with what they did on the old one.

    The rookie draft-capital prior already covers arrivals with no NFL history.
    This is the other kind of newcomer — a veteran who arrives carrying a
    measured share from somewhere else — and the share model currently ignores
    that history entirely, giving him only his new depth rank.
    """
    st = shares._entering_rank(st)
    rows = []
    for season in sorted(st.season.unique())[1:]:
        prev = st[st.season == season - 1]
        prev_any = (prev.assign(_u=prev.carry_share + prev.target_share)
                    .sort_values("_u").drop_duplicates("player_id", keep="last")
                    .set_index("player_id"))
        for _, p in st[st.season == season].iterrows():
            if p.player_id not in prev_any.index:
                continue
            before = prev_any.loc[p.player_id]
            if before.team == p.team:
                continue        # a returner: covered above
            rows.append({
                "season": season, "team": p.team, "position": p.position,
                "player_name": p.player_name,
                "rank": min(int(p.entering_rank), 5),
                "old_carry": float(before.carry_share),
                "old_target": float(before.target_share),
                "carry_share": float(p.carry_share),
                "target_share": float(p.target_share),
            })
    return pd.DataFrame(rows).merge(vac, on=["season", "team"], how="left").dropna(
        subset=["vacated_carry"]).reset_index(drop=True)


if __name__ == "__main__" and True:
    print("\n\n=== VETERANS WHO CHANGED TEAMS — does their old share travel?")
    mv = movers(st, vac)
    print(f"{len(mv):,} player-seasons on a new team with a measured prior share\n")
    for pool, old, vcol in (("carry_share", "old_carry", "vacated_carry"),
                            ("target_share", "old_target", "vacated_target")):
        print(f"  {pool}")
        flat = float(np.abs(mv[pool] - mv[pool].mean()).mean())
        print(f"    league mean alone                MAE {flat:.4f}")
        for label, feats in (("new depth rank alone", ["rank"]),
                             ("rank + OLD-TEAM share", ["rank", old]),
                             ("rank + old + team vacancy", ["rank", old, vcol]),
                             ("old-team share alone", [old])):
            mae, corr, n = loso(mv, feats, pool)
            print(f"    {label:32s}MAE {mae:.4f}  corr {corr:.3f}  n {n}")
        print("    coefficients:", coefficient(mv, ["rank", old, vcol], pool), "\n")
