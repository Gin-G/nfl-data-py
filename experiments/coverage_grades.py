"""Per-cornerback coverage grades: are they a trait, and can a board use them?

"Coverage matchup -- a null result, and one narrow win" tested this at the TEAM
level and found wide receiver *actively worse* (MAE 26.73 -> 26.94, correlation
0.472 -> 0.457). The diagnosis written down there was specific:

    "yards allowed to WR" is spread across a defence's whole secondary and a
    team's whole receiving corps, so it says nearly nothing about the matchup
    one receiver faces. The thing that would -- which corner travels with him
    and whether he is any good -- is a shadow-coverage assignment, and
    nflverse has no such field.

Half of that is wrong, and this file is about the half that is. **Pro Football
Reference charts coverage per defender**, weekly, back to 2018: targets,
completions, yards, touchdowns and passer rating allowed, plus how deep he was
tested. So "whether he is any good" IS available. What is still missing is only
the assignment -- who lined up across from whom on a given snap.

That gap decides what can honestly be built. Three questions in order, and the
first one that fails stops the file:

  1. Is a per-corner coverage grade a TRAIT, or 80 targets of noise? Same bar
     as the trench ratings -- split-half within season, year over year.
  2. Does a team coverage grade assembled from its actual corners beat the
     team-level number that already failed? Same receivers, same weeks.
  3. Does the SPREAD of a defence's corner grades predict the gap between the
     receivers it faces? This is the mechanism a "pick on him" read exploits,
     and it is the one testable without knowing the assignment: a team with one
     bad corner has an exploitable side even if its average is fine.

    python experiments/coverage_grades.py
"""
import pathlib

import numpy as np
import pandas as pd

CACHE = pathlib.Path(__file__).parent / "_cache"
PFR = ("https://github.com/nflverse/nflverse-data/releases/download/"
       "pfr_advstats/advstats_week_%s_%d.parquet")
SEASONS = tuple(range(2018, 2027))
SPLIT_SEASONS = (2022, 2023, 2024, 2025)
MIN_TGT = 20        # a corner-half-season with something in it
MIN_TGT_YOY = 40
COVER_POS = ("CB", "DB")


def advstats(kind):
    path = CACHE / f"blitz_pfr_{kind}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frames = []
    for season in SEASONS:
        try:
            frames.append(pd.read_parquet(PFR % (kind, season)))
        except Exception as exc:
            print(f"  {kind} {season}: {type(exc).__name__}")
    out = pd.concat(frames, ignore_index=True)
    CACHE.mkdir(exist_ok=True)
    out.to_parquet(path)
    return out


def positions():
    """pfr_id -> position, to keep corners apart from safeties and linebackers."""
    path = CACHE / "pfr_positions.parquet"
    if path.exists():
        return pd.read_parquet(path)
    import nflreadpy as nfl
    from nfl_projections.utils import to_pandas

    p = to_pandas(nfl.load_players())
    p = p[p.pfr_id.notna()][["pfr_id", "position"]].drop_duplicates("pfr_id")
    p = p.rename(columns={"pfr_id": "pfr_player_id"})
    p.to_parquet(path)
    return p


def corner_weeks():
    """One row per cornerback per week, with what was thrown at him."""
    d = advstats("def")
    d = d[d.game_type == "REG"].merge(positions(), on="pfr_player_id", how="left")
    d = d[d.position.isin(COVER_POS)].copy()
    d["yards"] = d.def_yards_allowed.fillna(0)
    d["comps"] = d.def_completions_allowed.fillna(0)
    d["tds"] = d.def_receiving_td_allowed.fillna(0)
    d["tgt"] = d.def_targets.fillna(0)
    d["air"] = d.def_adot.fillna(0) * d.tgt           # back to a total
    d["yac"] = d.def_yards_after_catch.fillna(0)
    # PFR publishes the rating already averaged, so re-weight by targets before
    # it can be pooled across weeks.
    d["rating_w"] = d.def_passer_rating_allowed.fillna(0) * d.tgt
    return d[d.tgt >= 0]


# metric -> (numerator, denominator, what it is)
METRICS = [
    ("targets per game", "tgt", "games", "role: is he thrown at"),
    ("average depth of target", "air", "tgt", "role: how deep is he tested"),
    ("completion % allowed", "comps", "tgt", "outcome"),
    ("yards allowed per target", "yards", "tgt", "outcome"),
    ("yards after catch per completion", "yac", "comps", "outcome"),
    ("TDs allowed per target", "tds", "tgt", "outcome"),
    ("passer rating allowed", "rating_w", "tgt", "outcome (PFR composite)"),
]


def _rate(x, num, den):
    d = x[den].sum()
    return x[num].sum() / d if d > 0 else np.nan


def split_half(d, metric, weight, min_n=MIN_TGT):
    """A corner against himself, first half of a season versus second."""
    d = d[d.season.isin(SPLIT_SEASONS)]
    mid = d.groupby("season").week.median()
    d = d.assign(half=np.where(d.week <= d.season.map(mid), "early", "late"))
    g = (d.groupby(["season", "half", "pfr_player_id"])
         .apply(lambda x: pd.Series({"val": _rate(x, metric, weight),
                                     "n": x.tgt.sum()}), include_groups=False)
         .reset_index())
    g = g[(g.n >= min_n) & g.val.notna()]
    w = g.pivot_table(index=["season", "pfr_player_id"], columns="half",
                      values="val").dropna()
    if len(w) < 20 or "early" not in w:
        return float("nan"), len(w)
    return float(np.corrcoef(w["early"], w["late"])[0, 1]), len(w)


def year_over_year(d, metric, weight, min_n=MIN_TGT_YOY):
    g = (d.groupby(["season", "pfr_player_id"])
         .apply(lambda x: pd.Series({"val": _rate(x, metric, weight),
                                     "n": x.tgt.sum()}), include_groups=False)
         .reset_index())
    g = g[(g.n >= min_n) & g.val.notna()]
    nxt = g.assign(season=g.season - 1).rename(columns={"val": "next"})
    m = g.merge(nxt[["season", "pfr_player_id", "next"]], on=["season", "pfr_player_id"])
    if len(m) < 20:
        return float("nan"), len(m)
    return float(np.corrcoef(m.val, m["next"])[0, 1]), len(m)


def career_split(d, metric, weight, min_n):
    """Odd weeks against even weeks, pooled over a corner's whole career.

    A season-half split gives a corner 20-40 targets a side, and a rate on 30
    targets is mostly sampling error whether or not a trait is under it. Pooling
    every season he played multiplies the sample without ever letting the same
    game appear on both sides. If a trait exists and the season-half test was
    merely underpowered, it has to show up here.
    """
    d = d.assign(par=np.where(d.week % 2 == 1, "odd", "even"))
    g = (d.groupby(["par", "pfr_player_id"])
         .apply(lambda x: pd.Series({"val": _rate(x, metric, weight),
                                     "n": x.tgt.sum()}), include_groups=False)
         .reset_index())
    g = g[(g.n >= min_n) & g.val.notna()]
    w = g.pivot_table(index="pfr_player_id", columns="par", values="val").dropna()
    if len(w) < 20 or "odd" not in w:
        return float("nan"), len(w)
    return float(np.corrcoef(w["odd"], w["even"])[0, 1]), len(w)


def sample_sweep(d):
    """If the sample is the problem, the correlation climbs with the threshold."""
    print("\nIS IT NOISE OR IS IT NOTHING? raise the minimum and watch")
    print("  A trait that is merely hard to measure gets EASIER to measure with")
    print("  more targets. One that is not there does not.")
    print("  %-30s %s" % ("metric", "  ".join("%8s" % ("min %d" % m)
                                              for m in (20, 30, 40, 60))))
    for label, num, den, what in METRICS:
        if what.startswith("role"):
            continue
        cells = []
        for m in (20, 30, 40, 60):
            r, n = split_half(d, num, den, min_n=m)
            cells.append("-" if np.isnan(r) else "%.3f/%d" % (r, n))
        print("  %-30s %s" % (label, "  ".join("%8s" % c for c in cells)))

    print("\n  ODD vs EVEN WEEKS, pooled over each corner's whole career")
    print("  %-30s %9s %7s %9s" % ("metric", "r", "corners", "median tgt"))
    for label, num, den, what in METRICS:
        for m in (100, 200):
            r, n = career_split(d, num, den, m)
            if m == 100:
                print("  %-30s %9s %7d   (min %d each side)"
                      % (label, "-" if np.isnan(r) else "%.3f" % r, n, m))
            else:
                print("  %-30s %9s %7d   (min %d each side)"
                      % ("", "-" if np.isnan(r) else "%.3f" % r, n, m))


# ------------------------------- does a board-usable grade come out of it

SHRINK_K = 60.0   # targets of league-average coverage mixed into every grade


def team_coverage(d):
    """A defence's coverage grade, assembled from the corners who played it.

    Prior weeks only, within season, carried from the previous season early on,
    and shrunk toward the league completion rate by `SHRINK_K` targets. Only
    completion %% allowed goes in: it is the one outcome metric that survived
    the career split (0.519), and yards allowed per target did not (0.188, and
    negative on the tighter sample), so putting yardage in would be importing
    the noise the stability test just rejected.

    This is a better team number than "yards allowed to WR" in three specific
    ways -- it is built from the defenders who are actually on the field, it is
    weighted by who draws throws, and it excludes safeties and linebackers.
    Whether better is good enough is the next function's problem.
    """
    g = (d.groupby(["season", "week", "team"])
         .agg(tgt=("tgt", "sum"), comps=("comps", "sum"),
              air=("air", "sum")).reset_index()
         .sort_values(["team", "season", "week"]))
    k = g.groupby(["team", "season"])
    for c in ("tgt", "comps", "air"):
        g[f"pre_{c}"] = k[c].cumsum() - g[c]
    prev = (g.groupby(["team", "season"])[["tgt", "comps"]].sum().reset_index())
    prev["season"] = prev.season + 1
    prev = prev.rename(columns={"tgt": "ptgt", "comps": "pcomps"})
    g = g.merge(prev, on=["team", "season"], how="left")
    g[["ptgt", "pcomps"]] = g[["ptgt", "pcomps"]].fillna(0)

    lg = d.comps.sum() / d.tgt.sum()
    # in-season prior weeks, backed by last season, then shrunk to the league
    tgt = g.pre_tgt + 0.5 * g.ptgt
    comp = g.pre_comps + 0.5 * g.pcomps
    g["cover_cmp"] = (comp + SHRINK_K * lg) / (tgt + SHRINK_K)
    g["cover_adot"] = g.pre_air / g.pre_tgt.replace(0, np.nan)
    g["cover_n"] = tgt

    # the exploitable-side measure: how UNEQUAL are this defence's corners
    def spread(x):
        s = x.groupby("pfr_player_id").agg(t=("tgt", "sum"), c=("comps", "sum"))
        s = s[s.t >= 15]
        if len(s) < 2:
            return np.nan
        r = s.c / s.t
        return float(r.max() - r.min())
    sp = (d.sort_values(["team", "season", "week"])
          .groupby(["season", "team"]).apply(spread, include_groups=False)
          .rename("cover_spread").reset_index())
    sp["season"] = sp.season + 1          # last season's spread, strictly prior
    return g.merge(sp, on=["season", "team"], how="left")


def against_the_board(d):
    """The only question that matters: does it explain the board's own error?

    Reuses the production backtest rows cached by `blitz_qb.py` -- same model,
    same 2021-25 window, `opponent=False` as served, so the board carries no
    coverage information of any kind and anything here is new to it.
    """
    from nfl_projections import config

    paths = sorted(CACHE.glob("blitz_board_*_1s60e.parquet"))
    if not paths:
        print("\n(no cached board rows; run experiments/blitz_qb.py with"
              " BLITZ_BOARD=1 first)")
        return
    b = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    opp = pd.read_csv(config.DATASET_PATH,
                      usecols=["player_id", "season", "week", "season_type",
                               "opponent_team", "targets"], low_memory=False)
    opp["week"] = pd.to_numeric(opp.week, errors="coerce")
    opp = opp[(opp.season_type == "REG") & opp.week.notna()]
    opp["week"] = opp.week.astype(int)
    b = b.merge(opp.drop_duplicates(["player_id", "season", "week"]),
                on=["player_id", "season", "week"], how="left")

    cov = team_coverage(d).rename(columns={"team": "opponent_team"})
    b = b.merge(cov[["season", "week", "opponent_team", "cover_cmp",
                     "cover_adot", "cover_spread", "cover_n"]],
                on=["season", "week", "opponent_team"], how="inner")
    b = b[b.cover_n >= 100]
    b["resid"] = b.actual - b.predicted
    for c in ("cover_cmp", "cover_adot", "cover_spread"):
        s = b.groupby("season")[c]
        b[f"z_{c}"] = (b[c] - s.transform("mean")) / s.transform("std")
    print("\nAGAINST THE BOARD'S OWN ERROR (%d player-weeks, 2021-25)" % len(b))
    print("  A positive slope means the board UNDER-rates players facing a")
    print("  defence whose corners allow more completions -- i.e. real")
    print("  information the projection does not have.")
    print("  %-4s %-16s %7s %9s %20s" % ("pos", "grade", "n", "slope", "95% CI"))
    from blitz_qb import allowed_prior, bootstrap, fit   # same estimators, same bar

    # A coverage grade is correlated with simply being a bad defence. The
    # control is the production model's own opponent column: points this
    # defence has already conceded to this very position, prior weeks only.
    ap = allowed_prior()
    b = b.merge(ap.rename(columns={"defteam": "opponent_team"}),
                on=["season", "week", "opponent_team", "position"], how="left")
    s = b.groupby("season").allowed_prior
    b["z_allowed_prior"] = (b.allowed_prior - s.transform("mean")) / s.transform("std")

    print("  %-4s %-20s %7s %9s %20s %9s"
          % ("pos", "grade", "n", "slope", "95% CI", "+quality"))
    for pos in ("WR", "TE", "RB", "QB"):
        sub = b[b.position == pos]
        for lab, col in (("completion % allowed", "z_cover_cmp"),
                         ("corner spread", "z_cover_spread"),
                         ("corner ADOT", "z_cover_adot")):
            a, n = fit(sub, "resid", [col])
            lo, hi = bootstrap(sub, "resid", [col], b=1000)
            c, _ = fit(sub, "resid", [col, "z_allowed_prior"])
            clo, chi = bootstrap(sub, "resid", [col, "z_allowed_prior"], b=1000)
            star = "  <<" if (clo > 0) == (chi > 0) else ""
            print("  %-4s %-20s %7d %9.3f  [%7.3f, %7.3f] %9.3f%s"
                  % (pos, lab, n, a, lo, hi, c, star))
    print("  (`+quality` is the same slope fit beside points already conceded")
    print("   to that position; `<<` marks the ones still clear of zero there)")
    return b


GRADES = {"WR": "z_cover_adot", "TE": "z_cover_spread",
          "RB": "z_cover_cmp", "QB": "z_cover_spread"}


def does_it_pay(b):
    """A clear slope is not a win. Blitz rate had both and paid nothing.

    Each position's best coverage grade is fitted leave-one-season-out and
    applied as one additive term, scored two ways: MAE, and the top-N ordering
    a board is actually read as. The intercept is dropped, because a
    per-position bias shift is #31 and would otherwise be credited here.
    """
    from blitz_qb import _paired

    print("\nDOES IT PAY? leave-one-season-out, slope only")
    print("  %-4s %-16s %7s %10s %11s %11s %6s"
          % ("pos", "grade", "n", "board MAE", "corrected", "gain", "won"))
    keep = {}
    for pos, col in GRADES.items():
        sub = b[(b.position == pos)][["resid", "predicted", "actual", col,
                                      "season", "week"]].replace(
            [np.inf, -np.inf], np.nan).dropna()
        rows = []
        for season in sorted(sub.season.unique()):
            tr, te = sub[sub.season != season], sub[sub.season == season]
            if len(tr) < 500 or len(te) < 100:
                continue
            X = np.column_stack([np.ones(len(tr)), tr[col].values])
            beta, *_ = np.linalg.lstsq(X, tr.resid.values, rcond=None)
            r = te.resid.values
            rows.append(pd.DataFrame({
                "season": te.season.values, "week": te.week.values,
                "e0": np.abs(r), "e2": np.abs(r - beta[1] * te[col].values)}))
        if not rows:
            continue
        e = pd.concat(rows, ignore_index=True)
        wk = e.groupby(["season", "week"])[["e0", "e2"]].mean()
        lo, hi = _paired(e, "e0", "e2")
        gain = e.e0.mean() - e.e2.mean()
        keep[pos] = (gain, lo, hi)
        print("  %-4s %-16s %7d %10.4f %11.4f %+11.4f %3d/%d  [%+.4f, %+.4f]"
              % (pos, col.replace("z_cover_", ""), len(e), e.e0.mean(),
                 e.e2.mean(), gain, int((wk.e0 > wk.e2).sum()), len(wk), lo, hi))

    print("\n  and the ORDER, which is what the page would be read as")
    print("  %-4s %6s %10s %10s %10s %7s" % ("pos", "top-N", "board",
                                             "corrected", "delta", "won"))
    for pos, col in GRADES.items():
        n_top = 24 if pos == "WR" else 12
        sub = b[b.position == pos][["resid", "predicted", "actual", col,
                                    "season", "week"]].replace(
            [np.inf, -np.inf], np.nan).dropna().copy()
        fits = {}
        for season in sorted(sub.season.unique()):
            tr = sub[sub.season != season]
            X = np.column_stack([np.ones(len(tr)), tr[col].values])
            fits[season], *_ = np.linalg.lstsq(X, tr.resid.values, rcond=None)
        sub["fixed"] = sub.predicted + [fits[s][1] * v
                                        for s, v in zip(sub.season, sub[col])]
        rows = []
        for (sn, wkno), wk in sub.groupby(["season", "week"]):
            if len(wk) < n_top * 2:
                continue
            act = set(wk.nlargest(n_top, "actual").index)
            rows.append({"season": sn, "week": wkno,
                         "o0": len(act & set(wk.nlargest(n_top, "predicted").index)) / n_top,
                         "o1": len(act & set(wk.nlargest(n_top, "fixed").index)) / n_top})
        o = pd.DataFrame(rows)
        lo, hi = _paired(o, "o1", "o0")
        print("  %-4s %6d %10.4f %10.4f %+10.4f %4d/%d  [%+.4f, %+.4f]"
              % (pos, n_top, o.o0.mean(), o.o1.mean(), o.o1.mean() - o.o0.mean(),
                 int((o.o1 > o.o0).sum()), len(o), lo, hi))
    return keep


def report():
    d = corner_weeks()
    d["games"] = 1
    print("PFR coverage charting: %d corner-weeks, %d corners, %d-%d"
          % (len(d), d.pfr_player_id.nunique(), d.season.min(), d.season.max()))
    tot = d.groupby(["season", "pfr_player_id"]).tgt.sum()
    print("  targets per corner-season: median %.0f, 25th %.0f, 75th %.0f"
          % (tot.median(), tot.quantile(.25), tot.quantile(.75)))
    print("  -- which is the whole problem: a rate on 60 targets is mostly noise")

    print("\nIS A CORNER'S COVERAGE A TRAIT? (split-half 2022-25, min %d targets"
          " a half; YoY min %d)" % (MIN_TGT, MIN_TGT_YOY))
    print("%-34s %9s %7s %9s %7s   %s"
          % ("metric", "split-half", "n", "YoY", "n", "what it is"))
    for label, num, den, what in METRICS:
        s, ns = split_half(d, num, den)
        y, ny = year_over_year(d, num, den)
        print("%-34s %9s %7d %9s %7d   %s"
              % (label,
                 "-" if np.isnan(s) else "%.3f" % s, ns,
                 "-" if np.isnan(y) else "%.3f" % y, ny, what))
    sample_sweep(d)
    b = against_the_board(d)
    if b is not None:
        does_it_pay(b)


if __name__ == "__main__":
    report()
