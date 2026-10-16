#!/usr/bin/env python3
"""Opportunity share: does volume beat the board's own projection?

The ask was for the optimizer to prefer players who "have their hands on the
ball a lot ... underpriced wide receivers that get volume, but maybe haven't
scored a lot of touchdowns". That is two claims, and only one of them is
measurable here.

The measurable one: touchdowns are noise and volume is a trait, so a player's
recent FANTASY POINTS are a worse forecast of his next week than his recent
OPPORTUNITIES are. If that holds, the gap between the two -- volume-implied
points minus points actually scored -- is information, and the question becomes
whether the production board already has it. The board carries targets,
carries, target_share, air_yards_share, wopr and an opportunity_score as input
FEATURES, so the honest null is that it does.

The one that is NOT measurable: "underpriced". There are no historical FanDuel
salaries anywhere in either repo and no free source for them (checked
2026-10-07), so nothing here can compare a player to his price. What follows
measures whether the signal beats the BOARD. Underpricing follows only if
salary tracks recent fantasy points, which is an assumption about FanDuel, not
a result in this file.

Definitions, fixed before anything was run:
  opportunity        targets + carries for a skill position -- for a running
                     back that is both, which is what was asked for. For a
                     quarterback it is pass attempts + carries instead, because
                     a quarterback has no targets and fitting points on targets
                     for one produces 5.1 points per target off a column that is
                     almost all zeroes.
  opportunity share  a player's opportunities over his team's that week.
  xFP                opportunities valued at the league's points-per-target and
                     points-per-carry for that position, fitted on OTHER
                     seasons. Deliberately blind to the player's own efficiency
                     and touchdown rate -- that blindness is the whole point.
  gap                xFP per game minus fantasy points per game, from weeks
                     strictly before the one being scored.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")
SEASONS = tuple(range(2018, 2027))
POSITIONS = ("QB", "RB", "WR", "TE")
# A player-week has to be a real role before its share means anything.
MIN_PRIOR_GAMES = 3
MIN_SPLIT_GAMES = 4          # per side of an odd/even split
BOOT = 2000
SEED = 20261016
# The same cached production backtests the blitz and coverage entries scored
# against, so this is the board as served: opponent=False, one seed, 60 epochs.
BOARD_SEASONS = (2021, 2022, 2023, 2024, 2025)
BOARD_SEEDS = 1
BOARD_EPOCHS = 60
TOP_N = {"QB": 12, "RB": 12, "WR": 24, "TE": 12}
# The two volume columns that ARE the opportunity, per position.
VOLUME = {"QB": ("attempts", "carries"), "RB": ("targets", "carries"),
          "WR": ("targets", "carries"), "TE": ("targets", "carries")}

_COLS = ["player_id", "player_name", "position", "season", "week", "season_type",
         "team", "opponent_team", "targets", "carries", "attempts",
         "receptions", "receiving_tds", "rushing_tds", "passing_tds",
         "offensive_snap_pct", "fanduel_fantasy_points"]


# ---------------------------------------------------------------- the frame

def weeks() -> pd.DataFrame:
    """Regular-season player-weeks with volume, team context and points."""
    path = os.path.join(CACHE, "opp_weeks.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    from nfl_projections import config

    d = pd.read_csv(config.DATASET_PATH, usecols=_COLS, low_memory=False)
    d["week"] = pd.to_numeric(d.week, errors="coerce")
    d = d[(d.season_type == "REG") & d.week.notna() & d.season.isin(SEASONS)]
    d["week"] = d.week.astype(int)
    d = d[d.position.isin(POSITIONS)].copy()
    for col in ("targets", "carries", "receptions", "attempts", "receiving_tds",
                "rushing_tds", "passing_tds", "fanduel_fantasy_points"):
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
    # Position-appropriate opportunity: a quarterback's volume is his pass
    # attempts, everyone else's is his targets.
    d["vol_pass"] = np.where(d.position == "QB", d.attempts, d.targets)
    d["opp"] = d.vol_pass + d.carries
    d["tds"] = np.where(d.position == "QB",
                        d.passing_tds + d.rushing_tds,
                        d.receiving_tds + d.rushing_tds)
    # Team opportunity for the share. Summed over the positions that take
    # handoffs and targets, which is what a share is competing for.
    # Team opportunity counts a quarterback's attempts once, as the pass plays
    # they are -- adding them to the targets those same plays produced would
    # double-count every throw and halve every receiver's share.
    skill = d[d.position != "QB"]
    team = (skill.groupby(["season", "week", "team"], as_index=False)
                 .agg(team_opp=("opp", "sum"), team_tgt=("targets", "sum"),
                      team_car=("carries", "sum")))
    d = d.merge(team, on=["season", "week", "team"], how="left")
    d["opp_share"] = d.opp / d.team_opp.replace(0, np.nan)
    d["tgt_share"] = d.targets / d.team_tgt.replace(0, np.nan)
    d["car_share"] = d.carries / d.team_car.replace(0, np.nan)
    d = d.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    os.makedirs(CACHE, exist_ok=True)
    d.to_parquet(path)
    return d


def zscore(frame, col):
    v = pd.to_numeric(frame[col], errors="coerce")
    sd = v.std()
    return (v - v.mean()) / (sd if sd else 1.0)


# ------------------------------------------------------- stage 1: the premise

def _corr_split(d, col, by, min_games):
    """Odd/even correlation of a per-game rate, pooled over `by` groups."""
    g = d.dropna(subset=[col])
    g = g.assign(parity=g.week % 2)
    agg = (g.groupby(by + ["parity"])
            .agg(v=(col, "mean"), n=(col, "size")).reset_index())
    agg = agg[agg.n >= min_games]
    wide = agg.pivot_table(index=by, columns="parity", values="v").dropna()
    if len(wide) < 25:
        return np.nan, len(wide)
    return float(np.corrcoef(wide[0], wide[1])[0, 1]), len(wide)


def stability(d):
    """Is opportunity a trait and scoring a coin flip? The premise, measured.

    Two splits for every number. Within-season odd/even asks whether a role is
    the same role in September and October. Career odd/even pools a player's
    whole history, which separates "there is no trait here" from "one season is
    too small a sample to see it" -- the distinction that decided the coverage
    entry, where yards allowed per target looked weak at a season and vanished
    outright over a career.
    """
    print("IS OPPORTUNITY A TRAIT AND IS SCORING NOISE")
    print("  odd weeks against even weeks, per player")
    print("  %-22s %10s %7s %10s %7s"
          % ("", "in season", "n", "career", "n"))
    rows = [
        ("opportunities / game", "opp"),
        ("opportunity share", "opp_share"),
        ("target share", "tgt_share"),
        ("targets or attempts / game", "vol_pass"),
        ("carries / game", "carries"),
        ("snap share", "offensive_snap_pct"),
        ("FanDuel points / game", "fanduel_fantasy_points"),
        ("touchdowns / game", "tds"),
    ]
    out = {}
    for label, col in rows:
        if col not in d.columns:
            continue
        rs, ns = _corr_split(d, col, ["player_id", "season"], MIN_SPLIT_GAMES)
        rc, nc = _corr_split(d, col, ["player_id"], 2 * MIN_SPLIT_GAMES)
        out[col] = (rs, rc)
        print("  %-22s %10.3f %7d %10.3f %7d" % (label, rs, ns, rc, nc))
    # Touchdowns per opportunity is the part of scoring that volume cannot
    # explain. If it has no trait, then two players with the same volume are
    # the same bet however differently they finished.
    e = d[d.opp >= 2].copy()
    e["td_rate"] = e.tds / e.opp
    rs, ns = _corr_split(e, "td_rate", ["player_id", "season"], MIN_SPLIT_GAMES)
    rc, nc = _corr_split(e, "td_rate", ["player_id"], 2 * MIN_SPLIT_GAMES)
    print("  %-22s %10.3f %7d %10.3f %7d"
          % ("touchdowns / OPPORTUNITY", rs, ns, rc, nc))
    print("\n  by position, in-season opportunity share vs FanDuel points")
    print("  %-5s %10s %10s %8s" % ("pos", "opp share", "points", "n"))
    for pos in POSITIONS:
        sub = d[d.position == pos]
        ro, n = _corr_split(sub, "opp_share", ["player_id", "season"], MIN_SPLIT_GAMES)
        rp, _ = _corr_split(sub, "fanduel_fantasy_points", ["player_id", "season"],
                            MIN_SPLIT_GAMES)
        print("  %-5s %10.3f %10.3f %8d" % (pos, ro, rp, n))
    return out


# ----------------------------------------------------------- stage 2: the xFP

def points_per_opportunity(d):
    """League points per target and per carry, per position, per season.

    Fitted by least squares on (targets, carries) with no intercept, so the
    coefficients are literally what one more of each is worth to a position
    that season. Used leave-one-season-out, so a week's xFP is never valued
    with its own season's rates.
    """
    rows = []
    for (season, pos), g in d.groupby(["season", "position"]):
        X = np.column_stack([g.vol_pass.values, g.carries.values])
        if len(g) < 100 or not X.any():
            continue
        beta, *_ = np.linalg.lstsq(X, g.fanduel_fantasy_points.values, rcond=None)
        rows.append({"season": season, "position": pos,
                     "pts_tgt": beta[0], "pts_car": beta[1], "n": len(g)})
    return pd.DataFrame(rows)


def priors(d, ppo):
    """Per player-week: his own prior weeks of this season, and nothing else.

    Everything is a cumulative mean shifted by one game, so the week being
    scored never contributes to its own forecast. xFP values those prior
    opportunities at rates fitted on OTHER seasons.
    """
    d = d.sort_values(["player_id", "season", "week"]).copy()
    grp = d.groupby(["player_id", "season"])
    for col, name in (("opp", "opp"), ("vol_pass", "tgt"), ("carries", "car"),
                      ("fanduel_fantasy_points", "fp"), ("opp_share", "share"),
                      ("tds", "td")):
        run = grp[col].cumsum() - d[col]
        cnt = grp.cumcount()
        d[f"{name}_prior"] = np.where(cnt > 0, run / cnt.replace(0, np.nan), np.nan)
    d["prior_games"] = grp.cumcount()
    # Last week alone, for the promoted-backup case in `recency`.
    d["share_last"] = grp["opp_share"].shift(1)
    d["opp_last"] = grp["opp"].shift(1)

    # Leave-one-season-out rates.
    rate = ppo.set_index(["season", "position"])
    tot = ppo.groupby("position").agg(t=("pts_tgt", "sum"), c=("pts_car", "sum"),
                                      k=("pts_tgt", "size"))
    pts_t, pts_c = [], []
    for season, pos in zip(d.season, d.position):
        if pos not in tot.index:
            pts_t.append(np.nan); pts_c.append(np.nan); continue
        row = tot.loc[pos]
        own = rate.loc[(season, pos)] if (season, pos) in rate.index else None
        k = row.k - (1 if own is not None else 0)
        if k <= 0:
            pts_t.append(np.nan); pts_c.append(np.nan); continue
        pts_t.append((row.t - (own.pts_tgt if own is not None else 0)) / k)
        pts_c.append((row.c - (own.pts_car if own is not None else 0)) / k)
    d["pts_tgt"] = pts_t
    d["pts_car"] = pts_c
    d["xfp_prior"] = d.tgt_prior * d.pts_tgt + d.car_prior * d.pts_car
    # The signal the ask describes: volume says more than the scoreboard did.
    d["gap"] = d.xfp_prior - d.fp_prior
    return d


def rates_table(ppo):
    print("\nWHAT AN OPPORTUNITY IS WORTH (least squares, no intercept)")
    print("  QB's passing column is ATTEMPTS; everyone else's is TARGETS.")
    print("  %-5s %12s %12s %8s" % ("pos", "pts / pass", "pts / carry", "seasons"))
    for pos in POSITIONS:
        sub = ppo[ppo.position == pos]
        if sub.empty:
            continue
        print("  %-5s %12.3f %12.3f %8d"
              % (pos, sub.pts_tgt.mean(), sub.pts_car.mean(), len(sub)))


# ------------------------------------------- stage 3: which prior forecasts better

def forecasts(d):
    """Prior points, prior xFP, or both -- against next week's actual points.

    Plain out-of-sample MAE on the raw outcome, not on a residual, because the
    question at this stage is only which summary of a player's past is the
    better forecast. The board is not in it yet.
    """
    print("\nWHICH SUMMARY OF THE PAST FORECASTS THE NEXT WEEK")
    print("  leave one season out; MAE of FanDuel points, lower is better")
    print("  %-5s %7s %10s %10s %10s %9s"
          % ("pos", "n", "prior FP", "prior xFP", "both", "best"))
    g = d[(d.prior_games >= MIN_PRIOR_GAMES)].copy()
    g = g[["position", "season", "week", "fanduel_fantasy_points", "fp_prior",
           "xfp_prior", "share_prior"]].replace([np.inf, -np.inf], np.nan).dropna()
    for pos in POSITIONS:
        sub = g[g.position == pos]
        if len(sub) < 500:
            continue
        err = {k: [] for k in ("fp", "xfp", "both")}
        for season in sorted(sub.season.unique()):
            tr, te = sub[sub.season != season], sub[sub.season == season]
            if len(tr) < 500 or len(te) < 100:
                continue
            y = tr.fanduel_fantasy_points.values
            for key, cols in (("fp", ["fp_prior"]), ("xfp", ["xfp_prior"]),
                              ("both", ["fp_prior", "xfp_prior"])):
                X = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in cols])
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                Xt = np.column_stack([np.ones(len(te))] + [te[c].values for c in cols])
                err[key].append(np.abs(te.fanduel_fantasy_points.values - Xt @ beta))
        if not err["fp"]:
            continue
        m = {k: np.concatenate(v).mean() for k, v in err.items()}
        best = min(m, key=m.get)
        print("  %-5s %7d %10.4f %10.4f %10.4f %9s"
              % (pos, sum(len(x) for x in err["fp"]), m["fp"], m["xfp"],
                 m["both"], best))


# ------------------------------------------------------- the board, and paying

def board_rows():
    """The cached production backtests, reused exactly as the blitz entry left
    them. Rebuilding is ~40 minutes a season; nothing here changes the model."""
    from nfl_projections import dataset as dataset_mod, evaluate

    frames, ds = [], None
    for season in BOARD_SEASONS:
        path = os.path.join(CACHE, f"blitz_board_{season}_{BOARD_SEEDS}s{BOARD_EPOCHS}e.parquet")
        if not os.path.exists(path):
            if ds is None:
                ds = dataset_mod.load_dataset()
            print(f"  backtesting {season} ...")
            res = evaluate.backtest(ds, season, epochs=BOARD_EPOCHS,
                                    n_seeds=BOARD_SEEDS, include_components=True)
            res.to_parquet(path)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def fit(sub, y, xcols):
    g = sub[[y] + xcols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(g) < 50:
        return np.nan, len(g)
    X = np.column_stack([np.ones(len(g))] + [g[c].values for c in xcols])
    beta, *_ = np.linalg.lstsq(X, g[y].values, rcond=None)
    return float(beta[1]), len(g)


def bootstrap(sub, y, xcols, b=BOOT, seed=SEED):
    """Cluster bootstrap over (season, week): a slate is one draw, not 300."""
    g = sub[[y, "season", "week"] + xcols].replace([np.inf, -np.inf], np.nan).dropna()
    groups = [gg.index.values for _, gg in g.groupby(["season", "week"])]
    if len(groups) < 10:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(b):
        pick = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[i] for i in pick])
        h = g.loc[rows]
        X = np.column_stack([np.ones(len(h))] + [h[c].values for c in xcols])
        beta, *_ = np.linalg.lstsq(X, h[y].values, rcond=None)
        out.append(beta[1])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def permute(sub, y, x, b=400, seed=SEED):
    """Within-week permutation: holds the slate, the weather and the injury
    report exactly fixed, and asks whether WHICH player got the volume matters."""
    g = sub[[y, x, "season", "week"]].replace([np.inf, -np.inf], np.nan).dropna()
    obs, _ = fit(g, y, [x])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(b):
        h = g.copy()
        h[x] = h.groupby(["season", "week"])[x].transform(
            lambda v: rng.permutation(v.values))
        a, _ = fit(h, y, [x])
        hits += int(abs(a) >= abs(obs))
    return obs, (hits + 1) / (b + 1)


def _paired(e, a, c, b=BOOT, seed=SEED):
    """Paired cluster bootstrap. Both arms score the same predictions off the
    same model, so the seed lottery appears in each and cancels."""
    groups = [g.index.values for _, g in e.groupby(["season", "week"])]
    d0, d1 = e[a].values, e[c].values
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(b):
        pick = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[i] for i in pick])
        out.append(d0[rows].mean() - d1[rows].mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def board(d):
    """Does the board's own error know about opportunity? Then: does fixing it pay.

    The board carries volume as input features, so a null here is the expected
    result and the interesting one. Every slope is reported next to a control on
    the board's own prediction, because a player with a big share is also a
    player the board already projects highly -- without the control this
    rediscovers level and calls it news.
    """
    print("\nAGAINST THE BOARD ITSELF (production backtest %d-%d, %d seed)"
          % (BOARD_SEASONS[0], BOARD_SEASONS[-1], BOARD_SEEDS))
    b = board_rows()
    keep = ["player_id", "season", "week", "share_prior", "share_last",
            "xfp_prior", "fp_prior", "gap", "opp_prior", "prior_games"]
    b = b.merge(d[keep].drop_duplicates(["player_id", "season", "week"]),
                on=["player_id", "season", "week"], how="inner")
    b = b[b.prior_games >= MIN_PRIOR_GAMES].copy()
    b["resid"] = b.actual - b.predicted
    for col in ("share_prior", "xfp_prior", "gap", "opp_prior", "predicted"):
        b[f"z_{col}"] = zscore(b, col)
    print("  %d scored player-weeks, board MAE %.4f"
          % (len(b), b.resid.abs().mean()))

    print("\n  the board's error on each candidate, alone and with its own"
          " projection held")
    print("  %-4s %-12s %7s %9s %20s %8s %12s"
          % ("pos", "term", "n", "slope", "95% cluster CI", "perm p", "+predicted"))
    tested = 0
    for pos in POSITIONS:
        sub = b[b.position == pos]
        for term, label in (("z_gap", "volume-TD gap"),
                            ("z_share_prior", "opp share"),
                            ("z_xfp_prior", "xFP")):
            a, n = fit(sub, "resid", [term])
            lo, hi = bootstrap(sub, "resid", [term], b=1000)
            _, pv = permute(sub, "resid", term, b=300)
            ctl, _ = fit(sub, "resid", [term, "z_predicted"])
            tested += 1
            print("  %-4s %-12s %7d %9.3f  [%7.3f, %7.3f] %8.3f %12.3f"
                  % (pos, label, n, a, lo, hi, pv, ctl))
    print("  %d slopes tested; read the intervals with that in mind." % tested)
    return b


def recency(b):
    """The backup starting because of an injury.

    His season-long share is a bench player's share and last week's is a
    starter's, and the board's volume features are season-rolling. So the test
    is not whether share predicts -- it is whether the JUMP does, over and
    above the level the board already has.
    """
    print("\nTHE PROMOTED BACKUP: last week's share against the season's")
    g = b.dropna(subset=["share_last", "share_prior"]).copy()
    g["jump"] = g.share_last - g.share_prior
    g["z_jump"] = zscore(g, "jump")
    print("  %-4s %7s %9s %20s %8s %12s"
          % ("pos", "n", "slope", "95% cluster CI", "perm p", "+predicted"))
    for pos in POSITIONS:
        sub = g[g.position == pos]
        a, n = fit(sub, "resid", ["z_jump"])
        lo, hi = bootstrap(sub, "resid", ["z_jump"], b=1000)
        _, pv = permute(sub, "resid", "z_jump", b=300)
        ctl, _ = fit(sub, "resid", ["z_jump", "z_predicted"])
        print("  %-4s %7d %9.3f  [%7.3f, %7.3f] %8.3f %12.3f"
              % (pos, n, a, lo, hi, pv, ctl))
    print("\n  and only the clearest promotions: last week's share at least"
          " double the season's")
    print("  %-4s %7s %12s %12s %10s" % ("pos", "n", "board resid", "mean actual",
                                         "mean pred"))
    for pos in POSITIONS:
        sub = g[(g.position == pos) & (g.share_prior > 0)
                & (g.share_last >= 2 * g.share_prior)]
        if len(sub) < 30:
            continue
        print("  %-4s %7d %12.3f %12.3f %10.3f"
              % (pos, len(sub), sub.resid.mean(), sub.actual.mean(),
                 sub.predicted.mean()))
    return g


def matchup(d, b):
    """The other half of the ask: does the opponent move the volume.

    Opponent opportunity allowed, from that defence's prior weeks only. This is
    the one place a matchup term has a measured reason to exist: the blitz entry
    found a defence's blitz rate moves running-back carries and receiver targets
    while leaving points alone, so volume is where an opponent effect was last
    seen alive.
    """
    print("\nDOES THE OPPONENT MOVE THE VOLUME (defence's prior weeks only)")
    allowed = (d.groupby(["season", "week", "opponent_team", "position"],
                         as_index=False)
                .agg(opp_allowed=("opp", "sum")))
    allowed = allowed.sort_values(["opponent_team", "position", "season", "week"])
    grp = allowed.groupby(["opponent_team", "position", "season"])
    run = grp.opp_allowed.cumsum() - allowed.opp_allowed
    cnt = grp.cumcount()
    allowed["allowed_prior"] = np.where(cnt > 0, run / cnt.replace(0, np.nan), np.nan)
    allowed = allowed.rename(columns={"opponent_team": "defteam"})

    m = d.merge(allowed[["season", "week", "defteam", "position", "allowed_prior"]],
                left_on=["season", "week", "opponent_team", "position"],
                right_on=["season", "week", "defteam", "position"], how="inner")
    m = m[m.prior_games >= MIN_PRIOR_GAMES].copy()
    m["z_allowed"] = zscore(m, "allowed_prior")
    # Own prior volume held, so this is the opponent's effect and not the role's.
    m["z_own"] = zscore(m, "opp_prior")
    print("  on a player's OWN opportunities that week, his own prior held")
    print("  %-4s %7s %9s %20s %8s"
          % ("pos", "n", "slope", "95% cluster CI", "perm p"))
    for pos in POSITIONS:
        sub = m[m.position == pos]
        a, n = fit(sub, "opp", ["z_allowed", "z_own"])
        lo, hi = bootstrap(sub, "opp", ["z_allowed", "z_own"], b=1000)
        _, pv = permute(sub, "opp", "z_allowed", b=300)
        print("  %-4s %7d %9.3f  [%7.3f, %7.3f] %8.3f" % (pos, n, a, lo, hi, pv))

    print("\n  and on the BOARD's error, which is the only version that could pay")
    opp = b.merge(d[["player_id", "season", "week", "opponent_team"]]
                  .drop_duplicates(["player_id", "season", "week"]),
                  on=["player_id", "season", "week"], how="left")
    opp = opp.merge(allowed[["season", "week", "defteam", "position", "allowed_prior"]],
                    left_on=["season", "week", "opponent_team", "position"],
                    right_on=["season", "week", "defteam", "position"], how="inner")
    opp["z_allowed"] = zscore(opp, "allowed_prior")
    print("  %-4s %7s %9s %20s %8s %12s"
          % ("pos", "n", "slope", "95% cluster CI", "perm p", "+predicted"))
    for pos in POSITIONS:
        sub = opp[opp.position == pos]
        a, n = fit(sub, "resid", ["z_allowed"])
        lo, hi = bootstrap(sub, "resid", ["z_allowed"], b=1000)
        _, pv = permute(sub, "resid", "z_allowed", b=300)
        c2, _ = fit(sub, "resid", ["z_allowed", "z_predicted"])
        print("  %-4s %7d %9.3f  [%7.3f, %7.3f] %8.3f %12.3f"
              % (pos, n, a, lo, hi, pv, c2))

    # Negative control. The same number joined to the player's OWN defence
    # instead of his opponent's: a defence's volume allowed cannot act on the
    # board's error for its own offence, so anything here is the merge leaking,
    # not football. Reported rather than asserted, because a control nobody
    # prints is a control nobody ran.
    own = b.merge(allowed[["season", "week", "defteam", "position",
                           "allowed_prior"]],
                  left_on=["season", "week", "team", "position"],
                  right_on=["season", "week", "defteam", "position"], how="inner")
    own["z_allowed"] = zscore(own, "allowed_prior")
    print("\n  negative control -- the SAME number joined to his own defence")
    print("  %-4s %7s %9s %20s" % ("pos", "n", "slope", "95% cluster CI"))
    for pos in POSITIONS:
        sub = own[own.position == pos]
        a, n = fit(sub, "resid", ["z_allowed"])
        lo, hi = bootstrap(sub, "resid", ["z_allowed"], b=1000)
        print("  %-4s %7d %9.3f  [%7.3f, %7.3f]" % (pos, n, a, lo, hi))
    return opp


def pays(b, terms):
    """Out of sample, and then in the order a lineup is actually picked from.

    Two bars, because they are different questions. MAE says whether the board
    gets closer; ordering says whether the top of the board changes, and a
    per-position bias shift can move the first without touching the second. The
    optimizer only ever reads the order, so the second bar is the one that
    decides whether any of this belongs in it.
    """
    print("\nDOES IT PAY: leave one season out, applied to the board")
    print("  slope only -- a per-position bias shift is a separate known"
          " result (#31) and would be charged to this term.")
    print("  %-4s %-12s %7s %10s %12s %9s %s"
          % ("pos", "term", "n", "board MAE", "corrected", "% of MAE", "weeks won"))
    for pos in POSITIONS:
        for term, label in terms:
            g = b[(b.position == pos)][["resid", "season", "week", term]] \
                .replace([np.inf, -np.inf], np.nan).dropna()
            rows = []
            for season in sorted(g.season.unique()):
                tr, te = g[g.season != season], g[g.season == season]
                if len(tr) < 500 or len(te) < 100:
                    continue
                X = np.column_stack([np.ones(len(tr)), tr[term].values])
                beta, *_ = np.linalg.lstsq(X, tr.resid.values, rcond=None)
                r = te.resid.values
                rows.append(pd.DataFrame({
                    "season": te.season.values, "week": te.week.values,
                    "e0": np.abs(r),
                    "e1": np.abs(r - beta[1] * te[term].values)}))
            if not rows:
                continue
            e = pd.concat(rows, ignore_index=True)
            wk = e.groupby(["season", "week"])[["e0", "e1"]].mean()
            lo, hi = _paired(e, "e0", "e1")
            print("  %-4s %-12s %7d %10.4f %12.4f %+8.2f%% %4d/%d  [%+.4f, %+.4f]"
                  % (pos, label, len(e), e.e0.mean(), e.e1.mean(),
                     100 * (e.e0.mean() - e.e1.mean()) / e.e0.mean(),
                     int((wk.e0 > wk.e1).sum()), len(wk), lo, hi))

    print("\n  AND WHAT IT DOES TO THE ORDER -- the only thing the optimizer reads")
    print("  top-N overlap with the week's actual best N, per position")
    print("  %-4s %-12s %6s %10s %10s %10s %s"
          % ("pos", "term", "top-N", "board", "corrected", "delta", "weeks won"))
    for pos in POSITIONS:
        n_top = TOP_N[pos]
        for term, label in terms:
            sub = b[b.position == pos][
                ["resid", "actual", "predicted", "season", "week", term]] \
                .replace([np.inf, -np.inf], np.nan).dropna().copy()
            fits = {}
            for season in sorted(sub.season.unique()):
                tr = sub[sub.season != season]
                if len(tr) < 500:
                    continue
                X = np.column_stack([np.ones(len(tr)), tr[term].values])
                fits[season], *_ = np.linalg.lstsq(X, tr.resid.values, rcond=None)
            if not fits:
                continue
            sub = sub[sub.season.isin(fits)]
            sub["fixed"] = sub.predicted + [
                fits[sn][1] * z for sn, z in zip(sub.season, sub[term])]
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
            if len(o) < 10:
                continue
            lo, hi = _paired(o, "o1", "o0")
            print("  %-4s %-12s %6d %10.4f %10.4f %+10.4f %4d/%d  [%+.4f, %+.4f]"
                  % (pos, label, n_top, o.o0.mean(), o.o1.mean(),
                     o.o1.mean() - o.o0.mean(),
                     int((o.o1 > o.o0).sum()), len(o), lo, hi))


def run():
    d = weeks()
    print("%d player-weeks, %d-%d, %d players"
          % (len(d), d.season.min(), d.season.max(), d.player_id.nunique()))
    print("NO HISTORICAL FANDUEL SALARIES EXIST, so nothing here measures")
    print("'underpriced'. Everything below is measured against the BOARD.\n")
    stability(d)
    ppo = points_per_opportunity(d)
    rates_table(ppo)
    d = priors(d, ppo)
    forecasts(d)
    b = board(d)
    recency(b)
    opp = matchup(d, b)
    pays(b, [("z_gap", "volume-TD gap"), ("z_share_prior", "opp share"),
             ("z_xfp_prior", "xFP")])
    # The opponent term produced the single largest slope in this file (QB
    # +1.554), so it faces the same two bars rather than being left as a slope.
    print("\n" + "=" * 70)
    print("THE SAME TWO BARS FOR THE OPPONENT TERM")
    pays(opp, [("z_allowed", "opp volume")])


if __name__ == "__main__":
    run()
