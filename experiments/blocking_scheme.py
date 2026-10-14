"""Does a team's run-direction profile identify its blocking scheme, and does it
predict anything about the backfield?

The coaching breakdowns that prompted this classify every team by blocking
scheme — outside zone, power gap, split zone — attach it to the offensive line
coach, and update it when a team changes (one example given: a team moving from
outside zone to a power-gap scheme after hiring a new line coach). nflverse
publishes no scheme label, but it does publish where every carry went:
`run_location` (left / middle / right) and `run_gap` (guard / tackle / end).

Zone and gap schemes distribute carries differently — zone attacks the edge and
the second level, gap concentrates between the tackles behind pulling blockers —
so the direction profile is a usable proxy for the label.

Two questions, in the order that matters:

1. Is the profile a trait? If a team's run directions do not persist, there is
   no scheme to detect and the rest is moot.
2. Does it predict BACKFIELD CONCENTRATION — what share of a team's carries go
   to its lead back? This is the question worth asking, because every finding in
   EXPERIMENTS.md says projections move on opportunity quantities and not on
   efficiency. If zone teams ride one back and gap teams rotate, that is a share
   feature. If the profile only predicts yards, prior results say it will not
   survive.

Deliberately NOT tested here: scheme change attributed to a coaching hire.
nflverse has no offensive-line-coach field, so the hire cannot be joined; a
change in the profile is detectable, its cause is not.
"""
import pathlib

import numpy as np
import pandas as pd

CACHE = pathlib.Path(__file__).parent / "_cache"
URL = ("https://github.com/nflverse/nflverse-data/releases/download/pbp/"
       "play_by_play_{s}.parquet")
SEASONS = tuple(range(2018, 2026))
COLS = ["season", "week", "season_type", "posteam", "rush_attempt", "qb_scramble",
        "qb_kneel", "two_point_attempt", "yards_gained", "run_location",
        "run_gap", "rusher_player_id"]
# Where a carry went. "middle" with no gap is nflverse's label for a straight
# interior run; the seven buckets below cover every charted carry.
DIRECTIONS = ["left_end", "left_tackle", "left_guard", "middle",
              "right_guard", "right_tackle", "right_end"]
MIN_CARRIES = 250          # a team-season of rushing
MIN_BACK_CARRIES = 30      # before a back counts toward the backfield split


def carries(season):
    path = CACHE / f"pbp_{season}.parquet"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        pd.read_parquet(URL.format(s=season)).to_parquet(path)
    d = pd.read_parquet(path, columns=COLS)
    d = d[(d.season_type == "REG") & (d.rush_attempt == 1) & (d.qb_scramble != 1)
          & (d.qb_kneel != 1) & (d.two_point_attempt != 1)
          & d.posteam.notna() & d.rusher_player_id.notna()
          & d.yards_gained.notna()].copy()
    d["direction"] = _direction(d)
    return d[d.direction.notna()]


def _direction(d):
    loc = d.run_location.fillna("")
    gap = d.run_gap.fillna("")
    out = np.where(loc == "middle", "middle",
                   np.where((loc != "") & (gap != ""), loc + "_" + gap, None))
    return pd.Series(out, index=d.index)


def profiles(d, keys=("season", "posteam")) -> pd.DataFrame:
    """Share of carries by direction, plus the three summary shares.

    `outside_share` is the edge (either end) — the zone signature.
    `interior_share` is guard and middle — the gap signature.
    `balance` is how evenly a team spreads left and right; a scheme that pulls
    one way is not the same as one that runs both.
    """
    counts = (d.groupby(list(keys) + ["direction"]).size()
              .unstack("direction").fillna(0.0))
    for col in DIRECTIONS:
        if col not in counts:
            counts[col] = 0.0
    counts = counts[DIRECTIONS]
    total = counts.sum(axis=1)
    shares = counts.div(total, axis=0)
    shares["carries"] = total
    shares["outside_share"] = shares.left_end + shares.right_end
    shares["interior_share"] = (shares.left_guard + shares.middle
                                + shares.right_guard)
    shares["tackle_share"] = shares.left_tackle + shares.right_tackle
    left = shares.left_end + shares.left_tackle + shares.left_guard
    right = shares.right_end + shares.right_tackle + shares.right_guard
    shares["balance"] = 1.0 - (left - right).abs()
    return shares.reset_index()


def backfield(d, keys=("season", "posteam")) -> pd.DataFrame:
    """How a team divides its carries: lead-back share and concentration.

    `hhi` is the Herfindahl index over every back's share — 1.0 is one back
    taking everything, 0.33 is an even three-way split. It is the measure that
    does not depend on who the lead back happens to be.
    """
    by_back = d.groupby(list(keys) + ["rusher_player_id"]).size().rename("n").reset_index()
    by_back = by_back[by_back.n >= MIN_BACK_CARRIES]
    rows = []
    for key, g in by_back.groupby(list(keys)):
        share = g.n / g.n.sum()
        rows.append({**dict(zip(keys, key if isinstance(key, tuple) else (key,))),
                     "backs": len(g), "lead_share": float(share.max()),
                     "hhi": float((share ** 2).sum())})
    return pd.DataFrame(rows)


def split_half(d, metric, min_carries=120):
    """Correlate a team's direction share between halves of a season."""
    mid = d.groupby("season").week.median()
    halves = d.assign(half=np.where(d.week <= d.season.map(mid), "early", "late"))
    prof = profiles(halves, keys=("season", "posteam", "half"))
    prof = prof[prof.carries >= min_carries]
    wide = prof.pivot_table(index=["season", "posteam"], columns="half", values=metric).dropna()
    if len(wide) < 20:
        return float("nan"), len(wide)
    return float(np.corrcoef(wide["early"], wide["late"])[0, 1]), len(wide)


def year_over_year(prof, metric):
    nxt = prof.assign(season=prof.season - 1)[["season", "posteam", metric]].rename(
        columns={metric: "next"})
    m = prof.merge(nxt, on=["season", "posteam"])
    if len(m) < 20:
        return float("nan"), len(m)
    return float(np.corrcoef(m[metric], m["next"])[0, 1]), len(m)


def loso(m, features, target):
    """Leave-one-season-out OLS, same protocol as the other prediction tests."""
    preds = np.full(len(m), np.nan)
    for season in sorted(m.season.unique()):
        train, test = m.season != season, m.season == season
        if train.sum() < 20 or test.sum() == 0:
            continue
        X = np.column_stack([np.ones(train.sum())] + [m.loc[train, f] for f in features])
        beta, *_ = np.linalg.lstsq(X, m.loc[train, target], rcond=None)
        Xt = np.column_stack([np.ones(test.sum())] + [m.loc[test, f] for f in features])
        preds[test.to_numpy()] = Xt @ beta
    ok = ~np.isnan(preds)
    return (float(np.abs(preds[ok] - m[target][ok]).mean()),
            float(np.corrcoef(preds[ok], m[target][ok])[0, 1]), int(ok.sum()))


if __name__ == "__main__":
    d = pd.concat([carries(s) for s in SEASONS], ignore_index=True)
    print(f"{len(d):,} charted non-QB carries, {SEASONS[0]}-{SEASONS[-1]}\n")

    prof = profiles(d)
    prof = prof[prof.carries >= MIN_CARRIES]
    print("League-average direction mix:")
    for col in DIRECTIONS:
        print(f"  {col:14s}{prof[col].mean() * 100:5.1f}%")

    print("\nIS THE PROFILE A TRAIT?")
    print(f"{'metric':18s}{'split-half r':>14}{'n':>6}{'YoY r':>8}{'n':>6}")
    for metric in ("outside_share", "interior_share", "tackle_share", "balance",
                   "left_end", "middle"):
        sh, shn = split_half(d, metric)
        yoy, yoyn = year_over_year(prof, metric)
        print(f"{metric:18s}{sh:14.3f}{shn:6d}{yoy:8.3f}{yoyn:6d}")

    back = backfield(d)
    m = prof.merge(back, on=["season", "posteam"])
    nxt = m.assign(season=m.season - 1)[["season", "posteam", "lead_share", "hhi", "backs"]]
    nxt = nxt.rename(columns={"lead_share": "next_lead", "hhi": "next_hhi",
                              "backs": "next_backs"})
    pairs = m.merge(nxt, on=["season", "posteam"]).reset_index(drop=True)
    print(f"\n{len(pairs)} team-season pairs with a backfield split in both years")

    print("\nDOES THE PROFILE PREDICT THE BACKFIELD NEXT SEASON?")
    scheme = ["outside_share", "interior_share", "tackle_share", "balance"]
    for target, own in (("next_lead", "lead_share"), ("next_hhi", "hhi")):
        flat = float(np.abs(pairs[target] - pairs[target].mean()).mean())
        print(f"\n  target: {target}   (league mean alone: MAE {flat:.4f})")
        for label, feats in (("own prior value", [own]),
                             ("own + scheme profile", [own] + scheme),
                             ("scheme profile alone", scheme)):
            mae, corr, n = loso(pairs, feats, target)
            print(f"    {label:24s}MAE {mae:.4f}  corr {corr:6.3f}  n {n}")

    print("\nSAME SEASON, for reference — does the mix go with the split at all?")
    for metric in scheme:
        for target in ("lead_share", "hhi"):
            r = float(np.corrcoef(m[metric], m[target])[0, 1])
            print(f"  {metric:16s} vs {target:11s}{r:7.3f}")
