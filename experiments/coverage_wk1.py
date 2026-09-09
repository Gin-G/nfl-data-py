"""Does matchup move receiving-yard projections, or only calibration?

Same protocol as the opportunity-share test: nothing after 2024, project week 1
of 2025, score against what happened. The share model fixed bias but not
correlation — this asks whether the defence a receiver is facing carries the
information that would.

Three things are tested, cheapest first:

  defense    how many receiving yards the opponent allowed to this position,
             relative to league average. The classic matchup adjustment.
  coverage   the receiver's own man-vs-zone production split, applied to how
             much man the opponent actually plays. Individual CB assignments
             are not in nflverse — there is no "who is covering him" field —
             so scheme is the honest substitute.
  both       multiplicative.

RESULT — week 1 of 2022-25, prior season only, scored against actuals:

    position   n     MAE            corr
    TE       183   18.67 -> 17.92   0.371 -> 0.424
    WR       392   26.73 -> 26.94   0.472 -> 0.457
    RB       188   13.59 -> 13.67   0.312 -> 0.321

Only the tight end moves, and it moves both ways that matter — error down and
correlation up, which is what separates a matchup signal from a recalibration.
It holds in three of four seasons (+0.025, +0.134, -0.011, +0.056).

Wide receiver is actively WORSE. "Yards allowed to WR" is spread across a whole
secondary and a whole receiving corps, so it says almost nothing about the
matchup one receiver faces. The thing that would — which corner travels with
him and whether he is any good — is a shadow-coverage assignment, and nflverse
publishes no such field. A tight end draws a far more specific assignment,
usually a linebacker or safety, which is why the team number works for him.

Coverage scheme is a NULL RESULT and is the reason this file exists. Man/zone
is available (49% of snaps classified) and a receiver's own man-vs-zone yards
per target is computable, but regressed for sample size the adjustment spans
0.977 to 1.015 at the 10th and 90th percentiles — a two percent nudge — and it
moved MAE by 0.03 yards over 763 player-weeks. Not wired in. Reviving it needs
per-route matchup data, not a team-level rate.

Run it:  python experiments/coverage_wk1.py
"""
import pathlib
import numpy as np
import pandas as pd

D = pathlib.Path(__file__).parent
CACHE = pathlib.Path.home() / ".sharp-edge" / "nfl"
WEEKLY = ("https://github.com/nflverse/nflverse-data/releases/download/"
          "stats_player/stats_player_week_{s}.parquet")
PART = ("https://github.com/nflverse/nflverse-data/releases/download/"
        "pbp_participation/pbp_participation_{s}.parquet")
PBP = ("https://github.com/nflverse/nflverse-data/releases/download/"
       "pbp/play_by_play_{s}.parquet")

POSITIONS = ("WR", "TE", "RB")


def weekly(seasons):
    out = []
    for s in seasons:
        p = CACHE / f"week_{s}.parquet"
        out.append(pd.read_parquet(p) if p.exists() else pd.read_parquet(WEEKLY.format(s=s)))
    return pd.concat(out, ignore_index=True)


def _cached(url, name, **kw):
    p = D / name
    if p.exists():
        return pd.read_parquet(p)
    df = pd.read_parquet(url, **kw)
    df.to_parquet(p)
    return df


def man_rate_by_defense(seasons):
    """Share of classified coverage snaps each defence plays in man."""
    frames = []
    for s in seasons:
        part = _cached(PART.format(s=s), f"part_{s}.parquet",
                       columns=["nflverse_game_id", "play_id", "defense_man_zone_type"])
        pbp = _cached(PBP.format(s=s), f"pbpmin_{s}.parquet",
                      columns=["game_id", "play_id", "defteam", "season"])
        m = part.rename(columns={"nflverse_game_id": "game_id"}).merge(
            pbp, on=["game_id", "play_id"], how="inner")
        frames.append(m)
    d = pd.concat(frames, ignore_index=True)
    d = d[d.defense_man_zone_type.isin(["MAN_COVERAGE", "ZONE_COVERAGE"])]
    g = d.groupby("defteam").defense_man_zone_type.apply(lambda s: (s == "MAN_COVERAGE").mean())
    return g.rename("man_rate")


def receiver_man_split(seasons):
    """Each receiver's yards per target against man vs zone.

    Needs the play-level receiver, so this joins participation to pbp on the
    play and reads ``receiver_player_id``.
    """
    frames = []
    for s in seasons:
        part = _cached(PART.format(s=s), f"part_{s}.parquet",
                       columns=["nflverse_game_id", "play_id", "defense_man_zone_type"])
        pbp = _cached(PBP.format(s=s), f"pbprec_{s}.parquet",
                      columns=["game_id", "play_id", "receiver_player_id",
                               "receiving_yards", "complete_pass", "pass_attempt"])
        m = part.rename(columns={"nflverse_game_id": "game_id"}).merge(
            pbp, on=["game_id", "play_id"], how="inner")
        frames.append(m)
    d = pd.concat(frames, ignore_index=True)
    d = d[d.receiver_player_id.notna() & (d.pass_attempt == 1)]
    d = d[d.defense_man_zone_type.isin(["MAN_COVERAGE", "ZONE_COVERAGE"])]
    d["yds"] = d.receiving_yards.fillna(0)
    d["is_man"] = d.defense_man_zone_type == "MAN_COVERAGE"

    g = d.groupby(["receiver_player_id", "is_man"]).agg(
        tgt=("yds", "size"), yds=("yds", "sum")).reset_index()
    w = g.pivot(index="receiver_player_id", columns="is_man",
                values=["tgt", "yds"]).fillna(0)
    w.columns = [f"{a}_{'man' if b else 'zone'}" for a, b in w.columns]
    return w


def defense_allowed(prior, seasons):
    """Receiving yards allowed per game to each position, by defence."""
    d = prior[prior.position.isin(POSITIONS)].copy()
    d["receiving_yards"] = d.receiving_yards.fillna(0)
    allowed = d.groupby(["opponent_team", "position"]).agg(
        yds=("receiving_yards", "sum"),
        games=("game_id", "nunique")).reset_index()
    allowed["per_game"] = allowed.yds / allowed.games
    lg = allowed.groupby("position").per_game.mean().rename("lg")
    allowed = allowed.merge(lg, on="position")
    allowed["def_factor"] = allowed.per_game / allowed.lg
    return allowed[["opponent_team", "position", "def_factor"]]


def build(through=2024, target=2025):
    hist = weekly(range(2021, target + 1))
    prior = hist[(hist.season == through) & (hist.season_type == "REG")]

    p = prior[prior.position.isin(POSITIONS)].groupby(
        ["player_id", "player_display_name", "position"]).agg(
        g=("week", "count"), yds=("receiving_yards", "sum"),
        tgt=("targets", "sum")).reset_index()
    p["trailing"] = p.yds / p.g
    p = p[p.tgt >= 20]

    actual = hist[(hist.season == target) & (hist.week == 1)
                  & (hist.season_type == "REG") & hist.position.isin(POSITIONS)]
    actual = actual.groupby(["player_id", "player_display_name", "position",
                             "team", "opponent_team"]).agg(
        actual=("receiving_yards", "sum")).reset_index()
    actual["actual"] = actual.actual.fillna(0)

    df = actual.merge(p[["player_id", "trailing"]], on="player_id", how="inner")

    # --- defence quality ---
    dfac = defense_allowed(prior, [through])
    df = df.merge(dfac, on=["opponent_team", "position"], how="left")
    df["def_factor"] = df.def_factor.fillna(1.0)
    df["defense"] = df.trailing * df.def_factor

    # --- coverage scheme ---
    man = man_rate_by_defense([through])
    split = receiver_man_split([through - 1, through])
    df = df.merge(man, left_on="opponent_team", right_index=True, how="left")
    df = df.merge(split, left_on="player_id", right_index=True, how="left")

    lg_man = float(man.mean())
    ypt_man = df.yds_man / df.tgt_man.replace(0, np.nan)
    ypt_zone = df.yds_zone / df.tgt_zone.replace(0, np.nan)
    # Regress each split toward the player's overall rate by target count.
    K = 25.0
    overall = (df.yds_man + df.yds_zone) / (df.tgt_man + df.tgt_zone).replace(0, np.nan)
    ypt_man = (ypt_man.fillna(overall) * df.tgt_man + overall * K) / (df.tgt_man + K)
    ypt_zone = (ypt_zone.fillna(overall) * df.tgt_zone + overall * K) / (df.tgt_zone + K)

    mr = df.man_rate.fillna(lg_man)
    expected = ypt_man * mr + ypt_zone * (1 - mr)
    baseline = ypt_man * lg_man + ypt_zone * (1 - lg_man)
    df["cov_factor"] = (expected / baseline).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df["coverage"] = df.trailing * df.cov_factor
    df["both"] = df.trailing * df.def_factor * df.cov_factor
    return df


def score(df, cols, label):
    d = df.dropna(subset=cols + ["actual"])
    print(f"\n{label}  (n={len(d)})")
    print(f"{'model':<12}{'MAE':>8}{'bias':>8}{'corr':>8}")
    for c in cols:
        err = d[c] - d.actual
        print(f"{c:<12}{err.abs().mean():8.2f}{err.mean():+8.2f}"
              f"{np.corrcoef(d[c], d.actual)[0,1]:8.3f}")
    return d


def multi(targets=(2022, 2023, 2024, 2025)):
    """Same protocol repeated across several week 1s.

    One week of 48 tight ends is not evidence; a correlation that moves on
    n=48 moves on noise about as often as on signal. This is the check that
    separates the two.
    """
    out = []
    for t in targets:
        d = build(through=t - 1, target=t)
        d["target_season"] = t
        out.append(d)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    df = build()
    cols = ["trailing", "defense", "coverage", "both"]
    score(df, cols, "Week 1 2025, WR/TE/RB receiving yards")
    for pos in POSITIONS:
        score(df[df.position == pos], cols, f"  {pos} only")

    print("\nspread of the adjustments (1.0 = no change):")
    for c in ["def_factor", "cov_factor"]:
        print(f"  {c:12s} p10 {df[c].quantile(.1):.3f}  median {df[c].median():.3f}"
              f"  p90 {df[c].quantile(.9):.3f}")
    df.to_parquet(D / "wk1_2025_coverage.parquet")
