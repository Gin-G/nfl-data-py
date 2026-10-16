"""Blitz rate against quarterback output.

The charted-line-play entry left blitz rate faced as the one measurement with no
counterpart in the ratings: a defensive trait at r = 0.659 split-half, the most
stable defensive number anywhere in the trench work, and unused. It also left
the reason not to believe it yet. "Blocking scheme is the most stable trait in
this file and predicts nothing" is two entries further back, and a blitz rate is
the same shape of claim — a coordinator's identity, stable precisely because it
is an identity, with no demonstrated link to anything a projection needs.

So: does the defense's blitz rate move the quarterback it is pointed at?

**No betting lines anywhere in this file.** A sack prop or an implied total
already contains whatever the market knows about the matchup, so conditioning on
one measures what survives after the edge has been taken out, which is the
opposite of the question being asked. A line is a benchmark to score against
later, never a regressor here.

THREE INSTRUMENTS, because one would not be believable

  in-season   the defense's blitz rate over its own prior weeks this season,
              read only once it has 100 dropbacks of it. Closest to what a
              Thursday actually knows; shares a season, a roster and a
              coordinator with the game being predicted.
  last-season the same defense's rate over the whole previous season. Cannot
              share a single play with the outcome, so nothing mechanical can
              couple them; costs the week-1-to-4 teams that changed staff.
  oracle      the rate in the game itself. Not a forecast — it is the ceiling,
              the answer to "if you knew exactly how often they would blitz,
              would it help you?" A null here closes the question for the
              forecastable versions too.

WHAT IS BEING EXPLAINED

Each quarterback-week is scored against his OWN season mean with that week left
out, so the comparison is the same passer in a blitz-heavy matchup versus a
blitz-light one and quarterback quality cancels exactly. Raw slopes are reported
beside it: a schedule is not random, and division opponents repeat.

Then the slope is re-fit next to the two opponent columns the production model
already carries for a quarterback — points allowed to the position, and the
pressure proxy the ratings are built on — because a blitz rate that only works
by standing in for "this defense is good" is not new information.

Errors are a cluster bootstrap over (season, week): a week shares a slate, a
weather map and an injury report. The null is a permutation that reshuffles
which defense each quarterback faced inside its own week, which holds the week
fixed exactly rather than modelling it.

    python experiments/blitz_qb.py
"""
import os
import pathlib

import numpy as np
import pandas as pd

CACHE = pathlib.Path(__file__).parent / "_cache"
PFR = ("https://github.com/nflverse/nflverse-data/releases/download/"
       "pfr_advstats/advstats_week_%s_%d.parquet")
PBP = ("https://github.com/nflverse/nflverse-data/releases/download/pbp/"
       "play_by_play_%d.parquet")
SEASONS = tuple(range(2018, 2027))

MIN_ATT = 10          # a quarterback-week that is a start, not a mop-up drive
MIN_OWN_GAMES = 6     # own-mean needs a season to be a mean of
MIN_PRIOR_DB = 100    # before a defense's in-season blitz rate is read at all
BOOT = 2000
SEED = 20261015

# The production stage trains models, so it is opt-in: BLITZ_BOARD=1
BOARD_SEASONS = (2021, 2022, 2023, 2024, 2025)
BOARD_SEEDS = int(os.environ.get("BLITZ_SEEDS", "1"))
BOARD_EPOCHS = 60


# ---------------------------------------------------------------- loading

def advstats(kind):
    """Weekly PFR charted stats, cached. Its own cache key: `trench_charted`
    stopped at 2025 and this needs the live season too."""
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


def pbp_season(season):
    path = CACHE / f"pbp_{season}.parquet"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        pd.read_parquet(PBP % season).to_parquet(path)
    cols = ["season", "week", "season_type", "posteam", "defteam",
            "qb_dropback", "sack", "qb_hit", "score_differential"]
    d = pd.read_parquet(path, columns=cols)
    return d[(d.season_type == "REG") & d.posteam.notna() & d.defteam.notna()]


def team_weeks():
    """Dropbacks and the shipped pressure proxy, per offense-week.

    The denominator is play-by-play rather than PFR's own `times_pressured_pct`,
    for the reason `trench_charted` gives: that percentage has a zero
    denominator for a backup who threw twice, and both rates have to sit on the
    same plays to be compared at all.
    """
    path = CACHE / "blitz_dropbacks.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frames = []
    for season in SEASONS:
        d = pbp_season(season)
        d = d[d.qb_dropback.fillna(0) == 1].copy()
        d["pres"] = ((d.sack.fillna(0) == 1) | (d.qb_hit.fillna(0) == 1)).astype(float)
        frames.append(d.groupby(["season", "week", "posteam", "defteam"])
                      .agg(dropbacks=("pres", "size"),
                           proxy_pressures=("pres", "sum"),
                           off_lead=("score_differential", "mean"))
                      .reset_index())
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(path)
    return out


def defense_weeks():
    """One row per defense per week: what it did to the passer it faced."""
    p = advstats("pass")
    p = p[p.game_type == "REG"]
    allowed = (p.groupby(["season", "week", "team", "opponent"])
               .agg(blitzed=("times_blitzed", "sum"),
                    hurried=("times_hurried", "sum"),
                    pressured=("times_pressured", "sum"),
                    sacked=("times_sacked", "sum"))
               .reset_index()
               .rename(columns={"team": "posteam", "opponent": "defteam"}))
    out = allowed.merge(team_weeks(), on=["season", "week", "posteam", "defteam"],
                        how="inner")
    out = out[out.dropbacks > 0]
    out["def_lead"] = -out.off_lead   # the scoreboard from the blitzer's side
    # A team plays once a week; the groupby is a guard, not an aggregation.
    g = (out.groupby(["season", "week", "defteam"])
         [["blitzed", "hurried", "pressured", "sacked", "dropbacks",
           "proxy_pressures"]].sum().reset_index())
    lead = (out.groupby(["season", "week", "defteam"]).def_lead.mean()
            .reset_index())
    return g.merge(lead, on=["season", "week", "defteam"])


def qb_weeks():
    """Quarterback starts, with the stat lines a blitz is supposed to move."""
    from nfl_projections import config

    cols = ["player_id", "player_display_name", "position", "season", "week",
            "season_type", "team", "opponent_team", "attempts", "completions",
            "passing_yards", "passing_tds", "passing_interceptions",
            "sacks_suffered", "passing_epa", "passing_cpoe", "passing_air_yards",
            "passing_40", "carries", "rushing_yards", "rushing_tds",
            "fanduel_fantasy_points"]
    d = pd.read_csv(config.DATASET_PATH, usecols=cols, low_memory=False)
    d = d[(d.position == "QB") & (d.season_type == "REG")]
    # The dataset carries 'AVG' summary rows in `week`.
    d["week"] = pd.to_numeric(d.week, errors="coerce")
    d = d[d.week.notna()]
    d["week"] = d.week.astype(int)
    d = d[d.attempts.fillna(0) >= MIN_ATT].copy()
    return d[d.season.isin(SEASONS)]


def allowed_prior():
    """FanDuel points allowed per game to each position, prior weeks only.

    This is the production model's own coarse opponent column —
    `opponent.opp_fppg_allowed` — rebuilt here so the blitz slope can be made
    to survive it. The real one rolls a window rather than the whole season to
    date; this is the same quantity on a slightly longer memory, which if
    anything makes it a *stronger* control than the model's version.
    """
    from nfl_projections import config

    path = CACHE / "blitz_allowed_prior.parquet"
    if path.exists():
        return pd.read_parquet(path)
    d = pd.read_csv(config.DATASET_PATH,
                    usecols=["position", "season", "week", "season_type",
                             "opponent_team", "fanduel_fantasy_points"],
                    low_memory=False)
    d = d[(d.season_type == "REG") & d.position.isin(["QB", "RB", "WR", "TE"])]
    d["week"] = pd.to_numeric(d.week, errors="coerce")
    d = d[d.week.notna() & d.opponent_team.notna()]
    g = (d.groupby(["season", "week", "opponent_team", "position"])
         .fanduel_fantasy_points.sum().rename("conceded").reset_index()
         .rename(columns={"opponent_team": "defteam"}))
    g = g.sort_values(["defteam", "position", "season", "week"])
    k = g.groupby(["defteam", "position", "season"])
    g["games"] = k.cumcount()
    g["allowed_prior"] = ((k.conceded.cumsum() - g.conceded)
                          / g.games.replace(0, np.nan))
    g["week"] = g.week.astype(int)
    out = g[["season", "week", "defteam", "position", "allowed_prior"]]
    out.to_parquet(path)
    return out


# ---------------------------------------------------------- the instruments

def priors(dw):
    """Attach the three blitz-rate instruments to each defense-week."""
    dw = dw.sort_values(["defteam", "season", "week"]).copy()
    g = dw.groupby(["defteam", "season"])
    for col in ("blitzed", "dropbacks", "sacked", "proxy_pressures"):
        dw[f"pre_{col}"] = g[col].cumsum() - dw[col]

    dw["blitz_in"] = np.where(dw.pre_dropbacks >= MIN_PRIOR_DB,
                              dw.pre_blitzed / dw.pre_dropbacks.replace(0, np.nan),
                              np.nan)
    dw["sack_in"] = np.where(dw.pre_dropbacks >= MIN_PRIOR_DB,
                             dw.pre_sacked / dw.pre_dropbacks.replace(0, np.nan),
                             np.nan)
    dw["proxy_in"] = np.where(dw.pre_dropbacks >= MIN_PRIOR_DB,
                              dw.pre_proxy_pressures / dw.pre_dropbacks.replace(0, np.nan),
                              np.nan)
    dw["blitz_now"] = dw.blitzed / dw.dropbacks        # the oracle

    tot = (dw.groupby(["defteam", "season"])[["blitzed", "dropbacks"]].sum()
           .reset_index())
    tot["blitz_prev"] = tot.blitzed / tot.dropbacks
    tot["season"] = tot.season + 1
    return dw.merge(tot[["defteam", "season", "blitz_prev"]],
                    on=["defteam", "season"], how="left")


def zscore(d, col, by="season"):
    """Within-season z-score. Charting volume drifts between seasons and a
    pooled z would read that drift as defensive aggression."""
    g = d.groupby(by)[col]
    return (d[col] - g.transform("mean")) / g.transform("std")


def frame():
    """Quarterback-weeks joined to the defense they faced, own-mean attached."""
    qb, dw = qb_weeks(), priors(defense_weeks())
    d = qb.merge(dw, left_on=["season", "week", "opponent_team"],
                 right_on=["season", "week", "defteam"], how="left")
    matched = d.defteam.notna().mean()
    print("quarterback starts %d, matched to a charted defense-week %.1f%%"
          % (len(d), 100 * matched))
    d = d[d.defteam.notna()].copy()

    d = d.merge(allowed_prior(), on=["season", "week", "defteam", "position"],
                how="left")
    d["fp"] = d.fanduel_fantasy_points
    d["ypa"] = d.passing_yards / d.attempts
    d["sack_rate"] = d.sacks_suffered / (d.attempts + d.sacks_suffered)
    d["int_rate"] = d.passing_interceptions / d.attempts
    d["adot"] = d.passing_air_yards / d.attempts
    d["epa_play"] = d.passing_epa / d.attempts
    d["deep"] = d.passing_40

    # Own-season mean with this week left out, for every outcome.
    g = d.groupby(["player_id", "season"])
    d["own_games"] = g.fp.transform("size")
    for col in OUTCOMES:
        s = g[col].transform("sum")
        d[f"d_{col}"] = d[col] - (s - d[col]) / (d.own_games - 1)
    short = d[d.own_games < MIN_OWN_GAMES]
    if len(short):
        print("dropped %d weeks whose passer has under %d starts that season"
              " (seasons %s)" % (len(short), MIN_OWN_GAMES,
                                 sorted(short.season.unique())))
        print("  -- which costs 2026 entirely: the own-mean needs more games"
              " than the season has played.")
    d = d[d.own_games >= MIN_OWN_GAMES].copy()

    for col in ("blitz_in", "blitz_prev", "blitz_now", "sack_in", "proxy_in",
                "def_lead", "allowed_prior"):
        d[f"z_{col}"] = zscore(d, col)
    return d


OUTCOMES = ["fp", "passing_yards", "passing_tds", "passing_interceptions",
            "sacks_suffered", "attempts", "rushing_yards", "carries", "ypa",
            "sack_rate", "int_rate", "adot", "epa_play", "deep"]


# ------------------------------------------------------------- estimation

def fit(d, ycol, xcols):
    """OLS slope on the first regressor. Rows with any NaN are dropped here
    rather than up front, so each outcome keeps every week it has."""
    sub = d[[ycol, *xcols]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 50:
        return float("nan"), 0
    X = np.column_stack([np.ones(len(sub)), sub[xcols].values])
    beta, *_ = np.linalg.lstsq(X, sub[ycol].values, rcond=None)
    return float(beta[1]), len(sub)


def bootstrap(d, ycol, xcols, b=BOOT, seed=SEED):
    """Cluster bootstrap over (season, week)."""
    sub = d[[ycol, "season", "week", *xcols]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 50:
        return float("nan"), float("nan")
    keys = sub.season * 100 + sub.week
    groups = [g.index.values for _, g in sub.groupby(keys)]
    y = sub[ycol].values
    X = np.column_stack([np.ones(len(sub)), sub[xcols].values])
    pos = {ix: i for i, ix in enumerate(sub.index)}
    groups = [np.array([pos[i] for i in g]) for g in groups]

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(b):
        pick = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[i] for i in pick])
        beta, *_ = np.linalg.lstsq(X[rows], y[rows], rcond=None)
        out.append(beta[1])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def permute(d, ycol, xcol, b=500, seed=SEED):
    """Null: reshuffle which defense each quarterback faced, within its week.

    This holds the week exactly — same slate, same weather, same injury report,
    same league-wide scoring environment — instead of trusting a control for it.
    Returns the two-sided p-value of the observed slope in that null.
    """
    sub = d[[ycol, xcol, "season", "week"]].replace([np.inf, -np.inf], np.nan).dropna()
    obs, _ = fit(sub, ycol, [xcol])
    rng = np.random.default_rng(seed)
    keys = (sub.season * 100 + sub.week).values
    x = sub[xcol].values.copy()
    y = sub[ycol].values
    order = np.argsort(keys, kind="stable")
    bounds = np.flatnonzero(np.diff(keys[order], prepend=-1, append=-1))
    draws = []
    for _ in range(b):
        xs = x.copy()
        for a, z in zip(bounds[:-1], bounds[1:]):
            idx = order[a:z]
            xs[idx] = rng.permutation(x[idx])
        X = np.column_stack([np.ones(len(xs)), xs])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        draws.append(beta[1])
    draws = np.array(draws)
    p = float((np.abs(draws) >= abs(obs)).mean())
    return obs, p, float(draws.std())


# ---------------------------------------------------------------- reports

def spread(d):
    print("\nWHAT A BLITZ RATE ACTUALLY SPANS (defense-seasons)")
    s = (d.groupby(["season", "defteam"])
         .apply(lambda x: x.blitzed.sum() / x.dropbacks.sum(), include_groups=False)
         .rename("rate").reset_index())
    q = s.groupby("season").rate.describe(percentiles=[.1, .5, .9])
    print(q[["count", "mean", "std", "10%", "50%", "90%"]].round(3).to_string())
    print("  one standard deviation of the in-season instrument = %.1f points of"
          " blitz rate" % (100 * d.blitz_in.std()))
    print("  the instruments against the thing itself: in-season r=%.3f,"
          " last-season r=%.3f"
          % (d[["blitz_in", "blitz_now"]].corr().iloc[0, 1],
             d[["blitz_prev", "blitz_now"]].corr().iloc[0, 1]))


def headline(d):
    print("\nFANDUEL POINTS vs BLITZ RATE — slope per +1 sd of blitz rate")
    print("%-34s %8s %8s %18s %9s" % ("", "n", "slope", "95% cluster CI", "perm p"))
    rows = [
        ("raw, in-season instrument", "fp", ["z_blitz_in"]),
        ("own-mean, in-season", "d_fp", ["z_blitz_in"]),
        ("own-mean, last-season", "d_fp", ["z_blitz_prev"]),
        ("own-mean, oracle (same game)", "d_fp", ["z_blitz_now"]),
        ("own-mean, in-season + points allowed", "d_fp",
         ["z_blitz_in", "z_allowed_prior"]),
        ("own-mean, in-season + 3 model cols", "d_fp",
         ["z_blitz_in", "z_allowed_prior", "z_sack_in", "z_proxy_in"]),
        ("own-mean, oracle + 3 model cols", "d_fp",
         ["z_blitz_now", "z_allowed_prior", "z_sack_in", "z_proxy_in"]),
    ]
    for label, y, x in rows:
        slope, n = fit(d, y, x)
        lo, hi = bootstrap(d, y, x)
        _, p, _ = permute(d, y, x[0]) if len(x) == 1 else (0, float("nan"), 0)
        print("%-34s %8d %8.3f   [%6.3f, %6.3f] %9s"
              % (label, n, slope, lo, hi, "-" if np.isnan(p) else "%.3f" % p))


def mechanism(d):
    print("\nWHICH PART OF THE STAT LINE MOVES — own-mean residual per +1 sd")
    print("%-22s %7s %9s %20s %20s" % ("outcome", "n", "mean", "in-season slope",
                                       "oracle slope"))
    for col in OUTCOMES:
        n_ = d[f"d_{col}"].notna().sum()
        a, _ = fit(d, f"d_{col}", ["z_blitz_in"])
        la, ha = bootstrap(d, f"d_{col}", ["z_blitz_in"])
        b, _ = fit(d, f"d_{col}", ["z_blitz_now"])
        lb, hb = bootstrap(d, f"d_{col}", ["z_blitz_now"])
        star = "  <<" if (la > 0) == (ha > 0) or (lb > 0) == (hb > 0) else ""
        print("%-22s %7d %9.3f  %6.3f [%6.3f,%6.3f] %6.3f [%6.3f,%6.3f]%s"
              % (col, n_, d[col].mean(), a, la, ha, b, lb, hb, star))


def oracle_endogeneity(d):
    """Why the oracle reads high, and why that does not rescue it.

    The rate in the game itself is not a forecast and it is not clean either: a
    defense escalates when the offense is moving the ball and stops needing to
    when it is not. The sign of the interception slope is the giveaway — nobody
    believes blitzing *prevents* interceptions, so a strong negative there is
    the feedback loop being read backwards as an effect.
    """
    print("\nWHY THE ORACLE RATE IS NOT A CLEAN MEASUREMENT EITHER")
    print("  blitz rate in the game vs the blitzer's own mean scoreboard: r=%.3f"
          % d[["blitz_now", "def_lead"]].corr().iloc[0, 1])
    print("  a defense that is behind blitzes more, so the rate is partly an"
          " outcome")
    a, n = fit(d, "d_int_rate", ["z_blitz_now"])
    print("  oracle -> interception rate: %+.4f per sd on %d weeks."
          " Nobody thinks a blitz prevents picks." % (a, n))
    for label, x in (("oracle alone", ["z_blitz_now"]),
                     ("oracle, scoreboard held", ["z_blitz_now", "z_def_lead"])):
        s, n = fit(d, "d_fp", x)
        lo, hi = bootstrap(d, "d_fp", x)
        print("  FanDuel points, %-22s %7.3f [%6.3f, %6.3f] on %d"
              % (label, s, lo, hi, n))


def terciles(d):
    print("\nTHE WHOLE DISTRIBUTION, not just its mean (in-season instrument)")
    s = d[d.blitz_in.notna()].copy()
    s["band"] = s.groupby("season").blitz_in.transform(
        lambda x: pd.qcut(x, 3, labels=["low", "mid", "high"], duplicates="drop"))
    g = s.groupby("band", observed=True)
    t = pd.DataFrame({
        "n": g.fp.size(),
        "blitz_rate": g.blitz_in.mean().round(3),
        "fp": g.fp.mean(),
        "own_mean_resid": g.d_fp.mean(),
        "sd_resid": g.d_fp.std(),
        "p10": g.d_fp.quantile(.10),
        "p90": g.d_fp.quantile(.90),
        "share_20plus": g.fp.apply(lambda x: (x >= 20).mean()),
        "share_bust_10": g.fp.apply(lambda x: (x < 10).mean()),
    })
    print(t.round(3).to_string())


def extremes(d):
    """Terciles can hide a cliff. The question a tercile cannot answer is
    whether the defenses that actually blitz half the time do something the
    middle of the league does not."""
    print("\nTHE EXTREMES — deciles of the in-season instrument")
    s = d[d.blitz_in.notna()].copy()
    s["dec"] = s.groupby("season").blitz_in.transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))
    g = s.groupby("dec", observed=True)
    t = pd.DataFrame({"n": g.fp.size(), "blitz_rate": g.blitz_in.mean(),
                      "fp": g.fp.mean(), "resid": g.d_fp.mean(),
                      "sd": g.d_fp.std()})
    print(t.round(3).to_string())
    top, bot = s[s.dec == 9], s[s.dec == 0]
    print("  top decile blitzes %.1f%% of dropbacks, bottom %.1f%%"
          % (100 * top.blitz_in.mean(), 100 * bot.blitz_in.mean()))
    print("  residual gap %.3f FanDuel points on %d vs %d weeks"
          % (top.d_fp.mean() - bot.d_fp.mean(), len(top), len(bot)))


def other_positions(d):
    """The adjacent claim, since the frame is already built.

    A blitz is supposed to push the ball to the check-down and open the seam,
    so if it moves anyone it should move backs and tight ends. Weeks are
    restricted to players on at least 40% of their offense's snaps, which
    conditions mildly on having been available and in the plan; the own-mean
    demeaning handles everything else about who the player is.

    Every slope here is also fit beside `allowed_prior` — points the defense
    has conceded to that very position, prior weeks only. A blitz rate that
    only works by standing in for "this defense gets scored on" is already in
    the model and is not a finding.
    """
    from nfl_projections import config

    print("\nTHE SAME TEST FOR THE SKILL POSITIONS (40%+ snap weeks)")
    cols = ["player_id", "position", "season", "week", "season_type", "team",
            "opponent_team", "offensive_snap_pct", "targets", "receptions",
            "receiving_yards", "carries", "rushing_yards",
            "fanduel_fantasy_points"]
    s = pd.read_csv(config.DATASET_PATH, usecols=cols, low_memory=False)
    s = s[(s.position.isin(["RB", "WR", "TE"])) & (s.season_type == "REG")]
    s["week"] = pd.to_numeric(s.week, errors="coerce")
    s = s[s.week.notna() & (s.offensive_snap_pct >= 40)].copy()
    s["week"] = s.week.astype(int)
    s["fp"] = s.fanduel_fantasy_points

    dw = d[["season", "week", "defteam", "blitz_in", "blitz_prev", "blitz_now",
            "sack_in", "proxy_in", "def_lead"]].drop_duplicates()
    s = s.merge(dw, left_on=["season", "week", "opponent_team"],
                right_on=["season", "week", "defteam"], how="inner")
    s = s.merge(allowed_prior(), on=["season", "week", "defteam", "position"],
                how="left")
    g = s.groupby(["player_id", "season"])
    s["own_games"] = g.fp.transform("size")
    s = s[s.own_games >= MIN_OWN_GAMES].copy()
    g = s.groupby(["player_id", "season"])
    outs = ("fp", "targets", "receiving_yards", "carries", "rushing_yards")
    for col in outs:
        s[f"d_{col}"] = s[col] - (g[col].transform("sum") - s[col]) / (s.own_games - 1)
    for col in ("blitz_in", "blitz_prev", "blitz_now", "sack_in", "proxy_in",
                "allowed_prior", "def_lead"):
        s[f"z_{col}"] = zscore(s, col)

    print("%-4s %-16s %6s %7s %25s %25s"
          % ("pos", "outcome", "n", "mean", "in-season slope", "oracle slope"))
    for pos in ("RB", "WR", "TE"):
        sub = s[s.position == pos]
        for col in outs:
            a, n = fit(sub, f"d_{col}", ["z_blitz_in"])
            la, ha = bootstrap(sub, f"d_{col}", ["z_blitz_in"], b=1000)
            b, _ = fit(sub, f"d_{col}", ["z_blitz_now"])
            lb, hb = bootstrap(sub, f"d_{col}", ["z_blitz_now"], b=1000)
            print("%-4s %-16s %6d %7.2f %7.3f [%7.3f,%7.3f] %7.3f [%7.3f,%7.3f]"
                  % (pos, col, n, sub[col].mean(), a, la, ha, b, lb, hb))

    print("\n  FanDuel points, every instrument, against the model's own column")
    print("  %-4s %-26s %6s %8s %20s %8s"
          % ("pos", "fit", "n", "slope", "95% cluster CI", "perm p"))
    arms = [("in-season alone", ["z_blitz_in"]),
            ("in-season + points allowed", ["z_blitz_in", "z_allowed_prior"]),
            ("in-season + 3 model cols",
             ["z_blitz_in", "z_allowed_prior", "z_sack_in", "z_proxy_in"]),
            ("last-season alone", ["z_blitz_prev"]),
            ("last-season + points allowed", ["z_blitz_prev", "z_allowed_prior"]),
            ("oracle alone", ["z_blitz_now"]),
            ("oracle + scoreboard", ["z_blitz_now", "z_def_lead"])]
    for pos in ("RB", "WR", "TE"):
        sub = s[s.position == pos]
        for label, x in arms:
            a, n = fit(sub, "d_fp", x)
            lo, hi = bootstrap(sub, "d_fp", x, b=1000)
            pv = permute(sub, "d_fp", x[0], b=400)[1] if len(x) == 1 else float("nan")
            print("  %-4s %-26s %6d %8.3f  [%7.3f, %7.3f] %8s"
                  % (pos, label, n, a, lo, hi,
                     "-" if np.isnan(pv) else "%.3f" % pv))
    return s


def reconcile(d, s):
    """The three instruments do not contradict each other; two of them are
    weaker, and by a known amount.

    Each instrument is a noisy reading of the rate in the game: the in-season
    one correlates with it at 0.45, last season's at 0.28. A regression on a
    noisy regressor is attenuated by exactly that ratio, so last season's slope
    should be about 0.63 of the in-season one -- not zero. Checking whether a
    weak instrument's interval contains the ATTENUATED prediction is a
    different and fairer question than whether it clears zero on its own.
    """
    r_in = d[["blitz_in", "blitz_now"]].corr().iloc[0, 1]
    r_pv = d[["blitz_prev", "blitz_now"]].corr().iloc[0, 1]
    print("\nRECONCILING THE INSTRUMENTS (attenuation factor %.3f)" % (r_pv / r_in))
    print("  %-4s %8s %10s %10s %20s %s"
          % ("pos", "in-season", "predicted", "observed", "last-season CI", "ok?"))
    for pos, sub in [("QB", d)] + [(k, s[s.position == k]) for k in ("RB", "WR", "TE")]:
        a, _ = fit(sub, "d_fp", ["z_blitz_in"])
        b, _ = fit(sub, "d_fp", ["z_blitz_prev"])
        lo, hi = bootstrap(sub, "d_fp", ["z_blitz_prev"], b=1000)
        pred = a * r_pv / r_in
        print("  %-4s %8.3f %10.3f %10.3f   [%6.3f, %6.3f]  %s"
              % (pos, a, pred, b, lo, hi, "yes" if lo <= pred <= hi else "NO"))


def pays(s, d):
    """Out of sample, which is a different question from clearing zero.

    Leave one season out: fit the slope on seven seasons, score the eighth.
    The baseline carries the model's own opponent column so the blitz rate has
    to beat what is already available, and the target is the own-mean residual
    rather than the board's own error -- so this says whether the information
    survives being held out, not yet what it is worth in production.
    """
    print("\nLEAVE-ONE-SEASON-OUT: does it survive being held out")
    print("  %-4s %6s %9s %9s %9s %7s"
          % ("pos", "n", "base MAE", "with blitz", "gain", "seasons won"))
    frames = [("QB", d)] + [(k, s[s.position == k]) for k in ("RB", "WR", "TE")]
    for pos, sub in frames:
        cols = ["d_fp", "z_blitz_in", "z_allowed_prior", "season"]
        g = sub[cols].replace([np.inf, -np.inf], np.nan).dropna()
        base_err, plus_err, won = [], [], 0
        for season in sorted(g.season.unique()):
            tr, te = g[g.season != season], g[g.season == season]
            if len(tr) < 200 or len(te) < 50:
                continue
            eb, ep = [], []
            for xcols, bucket in ((["z_allowed_prior"], eb),
                                  (["z_allowed_prior", "z_blitz_in"], ep)):
                X = np.column_stack([np.ones(len(tr)), tr[xcols].values])
                beta, *_ = np.linalg.lstsq(X, tr.d_fp.values, rcond=None)
                Xt = np.column_stack([np.ones(len(te)), te[xcols].values])
                bucket.append(np.abs(te.d_fp.values - Xt @ beta))
            base_err.append(eb[0])
            plus_err.append(ep[0])
            won += int(eb[0].mean() > ep[0].mean())
        if not base_err:
            continue
        b = np.concatenate(base_err).mean()
        pl = np.concatenate(plus_err).mean()
        print("  %-4s %6d %9.4f %9.4f %+9.4f %4d/%d"
              % (pos, sum(len(x) for x in base_err), b, pl, b - pl,
                 won, len(base_err)))
    print("  (gain is MAE points of the own-mean residual, not of the board)")


def mobility(d):
    print("\nDOES A RUNNING QUARTERBACK HANDLE IT DIFFERENTLY")
    s = d[d.blitz_in.notna()].copy()
    g = s.groupby(["player_id", "season"])
    s["own_carries"] = (g.carries.transform("sum") - s.carries) / (s.own_games - 1)
    s = s[s.own_carries.notna()]
    s["legs"] = s.groupby("season").own_carries.transform(
        lambda x: pd.qcut(x, 3, labels=["pocket", "mid", "mobile"], duplicates="drop"))
    for band in ("pocket", "mid", "mobile"):
        sub = s[s.legs == band]
        a, n = fit(sub, "d_fp", ["z_blitz_in"])
        lo, hi = bootstrap(sub, "d_fp", ["z_blitz_in"])
        print("  %-7s n=%5d  own carries/g %.1f  slope %6.3f [%6.3f, %6.3f]"
              % (band, n, sub.own_carries.mean(), a, lo, hi))


def by_season(d):
    print("\nSEASON BY SEASON, own-mean residual on the oracle rate")
    print("  (if a real effect existed in one era and not another, it is here)")
    for season, sub in d.groupby("season"):
        a, n = fit(sub, "d_fp", ["z_blitz_now"])
        lo, hi = bootstrap(sub, "d_fp", ["z_blitz_now"], b=500)
        print("  %d  n=%4d  slope %6.3f [%6.3f, %6.3f]" % (season, n, a, lo, hi))


def stability(d):
    print("\nTHE TRAIT ITSELF, re-measured on this frame")
    s = (d.groupby(["season", "defteam"])
         .apply(lambda x: pd.Series({"blitz": x.blitzed.sum() / x.dropbacks.sum(),
                                     "db": x.dropbacks.sum()}), include_groups=False)
         .reset_index())
    nxt = s.assign(season=s.season - 1).rename(columns={"blitz": "next"})
    m = s.merge(nxt[["season", "defteam", "next"]], on=["season", "defteam"])
    print("  year over year, defense blitz rate: r=%.3f on %d team-seasons"
          % (m.blitz.corr(m["next"]), len(m)))
    print("  so the instrument is sound. That was never the question.")


# ------------------------------------------------- against the actual board

def board_rows():
    """Production backtest rows, cached per season.

    `opponent=False`, which is how the board is actually served -- the model
    carries no opponent column at all (three matchup entries above say why), so
    a blitz rate is not competing with a weaker version of itself. One seed:
    the seed lottery is +/-0.05 MAE and is uncorrelated with any defense's
    blitz rate, so it inflates the residual spread without biasing a slope. It
    would NOT be enough for an MAE A/B, which is why the correction below is
    scored as a paired difference on the same predictions.
    """
    from nfl_projections import dataset as dataset_mod, evaluate

    frames, ds = [], None
    for season in BOARD_SEASONS:
        path = CACHE / f"blitz_board_{season}_{BOARD_SEEDS}s{BOARD_EPOCHS}e.parquet"
        if not path.exists():
            if ds is None:
                ds = dataset_mod.load_dataset()
            print(f"  backtesting {season} ...")
            res = evaluate.backtest(ds, season, epochs=BOARD_EPOCHS,
                                    n_seeds=BOARD_SEEDS, include_components=True)
            res.to_parquet(path)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def _paired(e, a, c, bb=2000, seed=SEED):
    """Cluster bootstrap of a PAIRED error difference over (season, week).

    Paired because both arms score the same predictions off the same trained
    model, so the seed lottery and the week's own scoring environment appear
    identically in each and cancel in the difference.
    """
    groups = [g.index.values for _, g in e.groupby(["season", "week"])]
    d0, d1 = e[a].values, e[c].values
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(bb):
        pick = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[i] for i in pick])
        out.append(d0[rows].mean() - d1[rows].mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def board(d, s):
    """Does the board's own error know about blitz rate, and does fixing it pay?

    This is the step the promotion entry had to come back for: a slope against a
    player's own season mean is not a slope against what production serves. The
    correction is deliberately the crudest possible -- one additive term per
    position, fitted on the other seasons -- because if the simplest form of
    the information cannot pay, a more elaborate one is fitting noise.
    """
    from nfl_projections import config

    print("\nAGAINST THE BOARD ITSELF (production backtest, %d-%d, %d seed)"
          % (BOARD_SEASONS[0], BOARD_SEASONS[-1], BOARD_SEEDS))
    b = board_rows()
    opp = pd.read_csv(config.DATASET_PATH,
                      usecols=["player_id", "season", "week", "season_type",
                               "opponent_team"], low_memory=False)
    opp["week"] = pd.to_numeric(opp.week, errors="coerce")
    opp = opp[(opp.season_type == "REG") & opp.week.notna()]
    opp["week"] = opp.week.astype(int)
    b = b.merge(opp.drop_duplicates(["player_id", "season", "week"]),
                on=["player_id", "season", "week"], how="left")

    dw = pd.concat([d[["season", "week", "defteam", "blitz_in", "blitz_prev",
                       "blitz_now"]],
                    s[["season", "week", "defteam", "blitz_in", "blitz_prev",
                       "blitz_now"]]]).drop_duplicates(["season", "week", "defteam"])
    b = b.merge(dw, left_on=["season", "week", "opponent_team"],
                right_on=["season", "week", "defteam"], how="inner")
    b["resid"] = b.actual - b.predicted
    for col in ("blitz_in", "blitz_prev", "blitz_now"):
        b[f"z_{col}"] = zscore(b, col)
    print("  %d scored player-weeks, board MAE %.4f" % (len(b), b.resid.abs().mean()))

    print("\n  the board's error, regressed on the blitz rate it never saw")
    print("  %-4s %6s %9s %9s %20s %8s"
          % ("pos", "n", "bias", "slope", "95% cluster CI", "perm p"))
    for pos in ("QB", "RB", "WR", "TE"):
        sub = b[b.position == pos]
        a, n = fit(sub, "resid", ["z_blitz_in"])
        lo, hi = bootstrap(sub, "resid", ["z_blitz_in"], b=1000)
        pv = permute(sub, "resid", "z_blitz_in", b=400)[1]
        print("  %-4s %6d %9.3f %9.3f  [%7.3f, %7.3f] %8.3f"
              % (pos, n, sub.resid.mean(), a, lo, hi, pv))

    print("\n  one additive term per position, fitted leave-one-season-out")
    print("  The slope is scored twice. `bias+slope` applies the whole fitted")
    print("  line; `slope only` drops the intercept, because a per-position")
    print("  bias shift is a separate known result -- #31, the bias that should")
    print("  not be corrected -- and would otherwise be charged to the blitz term.")
    print("  %-4s %6s %10s %12s %12s %9s %s"
          % ("pos", "n", "board MAE", "bias+slope", "slope only", "% of MAE",
             "weeks won"))
    g = b[["resid", "actual", "predicted", "z_blitz_in", "position", "season",
           "week"]].replace([np.inf, -np.inf], np.nan).dropna()
    for pos in ("QB", "RB", "WR", "TE"):
        sub = g[g.position == pos]
        rows = []
        for season in sorted(sub.season.unique()):
            tr, te = sub[sub.season != season], sub[sub.season == season]
            if len(tr) < 500 or len(te) < 100:
                continue
            X = np.column_stack([np.ones(len(tr)), tr.z_blitz_in.values])
            beta, *_ = np.linalg.lstsq(X, tr.resid.values, rcond=None)
            r = te.resid.values
            rows.append(pd.DataFrame({
                "season": te.season.values, "week": te.week.values,
                "e0": np.abs(r),
                "e1": np.abs(r - (beta[0] + beta[1] * te.z_blitz_in.values)),
                "e2": np.abs(r - beta[1] * te.z_blitz_in.values)}))
        if not rows:
            continue
        e = pd.concat(rows, ignore_index=True)
        wk = e.groupby(["season", "week"])[["e0", "e2"]].mean()
        print("  %-4s %6d %10.4f %12.4f %12.4f %+8.2f%% %5d/%d"
              % (pos, len(e), e.e0.mean(), e.e1.mean(), e.e2.mean(),
                 100 * (e.e0.mean() - e.e2.mean()) / e.e0.mean(),
                 int((wk.e0 > wk.e2).sum()), len(wk)))
        lo, hi = _paired(e, "e0", "e2")
        print("       slope-only gain %+.4f, cluster bootstrap over weeks"
              " [%+.4f, %+.4f]" % (e.e0.mean() - e.e2.mean(), lo, hi))

    print("\n  and what it does to the ORDER, which is what a board is used for")
    print("  A per-position bias shift cannot change a within-position ranking,")
    print("  so this arm is the slope on its own whichever way it was fitted.")
    print("  %-4s %6s %10s %10s %9s %s"
          % ("pos", "top-N", "board", "corrected", "delta", "weeks won"))
    for pos, n_top in (("QB", 12), ("RB", 12), ("WR", 24), ("TE", 12)):
        sub = g[g.position == pos].copy()
        fits = {}
        for season in sorted(sub.season.unique()):
            tr = sub[sub.season != season]
            X = np.column_stack([np.ones(len(tr)), tr.z_blitz_in.values])
            fits[season], *_ = np.linalg.lstsq(X, tr.resid.values, rcond=None)
        sub["fixed"] = sub.predicted + [
            fits[sn][1] * z for sn, z in zip(sub.season, sub.z_blitz_in)]
        rows = []
        for (sn, wkno), wk in sub.groupby(["season", "week"]):
            if len(wk) < n_top * 2:
                continue
            act = set(wk.nlargest(n_top, "actual").index)
            rows.append({
                "season": sn, "week": wkno,
                "o0": len(act & set(wk.nlargest(n_top, "predicted").index)) / n_top,
                "o1": len(act & set(wk.nlargest(n_top, "fixed").index)) / n_top})
        o = pd.DataFrame(rows)
        lo, hi = _paired(o, "o1", "o0")
        print("  %-4s %6d %10.4f %10.4f %+9.4f %5d/%d  [%+.4f, %+.4f]"
              % (pos, n_top, o.o0.mean(), o.o1.mean(), o.o1.mean() - o.o0.mean(),
                 int((o.o1 > o.o0).sum()), len(o), lo, hi))


def run():
    d = frame()
    print("%d quarterback-weeks, %d-%d, %d passers, %d defense-weeks"
          % (len(d), d.season.min(), d.season.max(), d.player_id.nunique(),
             len(d.groupby(["season", "week", "defteam"]))))
    stability(d)
    spread(d)
    headline(d)
    mechanism(d)
    oracle_endogeneity(d)
    terciles(d)
    extremes(d)
    mobility(d)
    by_season(d)
    s = other_positions(d)
    reconcile(d, s)
    pays(s, d)
    if os.environ.get("BLITZ_BOARD"):
        board(d, s)
    else:
        print("\n(set BLITZ_BOARD=1 for the production-board stage; it trains)")


if __name__ == "__main__":
    run()
