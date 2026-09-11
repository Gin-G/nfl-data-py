"""Does team success-rate-allowed actually predict player production?

It is the most stable defensive trait measured here — split-half r 0.618,
against 0.397 for EPA and 0.06-0.26 for everything in the prior work. Stability
is a precondition, not a result: a trait can persist perfectly and still say
nothing about a player's yards.

Same protocol as everything else. Nothing after `through`, project week 1 of
`target`, score against actuals. Three candidate adjustments, all built from
the prior season only:

  sr      team success rate allowed / league average
  sr_adj  the same, opponent-adjusted — each offence's success against this
          defence relative to that offence's own norm, which is the correction
          that turned raw allowed-yards from useless into useful for RBs
  epa     team epa allowed per play, as a reference point with known lower
          stability (0.397)
"""
import pathlib
import numpy as np
import pandas as pd

D = pathlib.Path(__file__).parent
CACHE = pathlib.Path.home() / ".sharp-edge" / "nfl"
WEEKLY = ("https://github.com/nflverse/nflverse-data/releases/download/"
          "stats_player/stats_player_week_{s}.parquet")
PBP = ("https://github.com/nflverse/nflverse-data/releases/download/pbp/"
       "play_by_play_{s}.parquet")
POSITIONS = ("WR", "TE", "RB", "QB")


def weekly(seasons):
    out = []
    for s in seasons:
        p = CACHE / f"week_{s}.parquet"
        out.append(pd.read_parquet(p) if p.exists() else pd.read_parquet(WEEKLY.format(s=s)))
    return pd.concat(out, ignore_index=True)


def pbp(season):
    p = D / f"pbpfull_{season}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    df = pd.read_parquet(PBP.format(s=season),
                         columns=["game_id", "play_id", "week", "posteam", "defteam",
                                  "epa", "rush_attempt", "pass_attempt", "success",
                                  "yards_gained"])
    df.to_parquet(p)
    return df


def defence_factors(season):
    """Per-defence factors, all relative to the league (1.0 = average)."""
    d = pbp(season)
    d = d[d.epa.notna() & d.defteam.notna() & d.posteam.notna()]

    sr = d.groupby("defteam").success.mean()
    epa = d.groupby("defteam").epa.mean()

    # Opponent-adjusted: each offence's success against this defence relative
    # to that offence's own success elsewhere. Leave-one-out by construction,
    # since the baseline excludes the games against this defence.
    rows = []
    for (off, dfn), g in d.groupby(["posteam", "defteam"]):
        other = d[(d.posteam == off) & (d.defteam != dfn)]
        if len(other) < 200 or len(g) < 30:
            continue
        rows.append({"defteam": dfn, "ratio": g.success.mean() / max(other.success.mean(), 1e-6),
                     "n": len(g)})
    adj = pd.DataFrame(rows)
    sr_adj = (adj.groupby("defteam")
                 .apply(lambda x: np.average(x.ratio, weights=x.n))
              if not adj.empty else pd.Series(dtype=float))

    out = pd.DataFrame({"sr": sr / sr.mean(), "epa_raw": epa})
    # EPA allowed is signed and near zero, so a ratio is meaningless; convert a
    # per-play EPA difference into a multiplicative effect on production.
    out["epa"] = 1.0 + (out.epa_raw - out.epa_raw.mean()) * 2.0
    if len(sr_adj):
        out["sr_adj"] = sr_adj / sr_adj.mean()
    return out


def build(through, target, stat, positions):
    hist = weekly(range(through, target + 1))
    prior = hist[(hist.season == through) & (hist.season_type == "REG")].copy()
    prior[stat] = prior[stat].fillna(0)

    p = prior[prior.position.isin(positions)].groupby(["player_id", "position"]).agg(
        g=("week", "count"), v=(stat, "sum")).reset_index()
    p["trailing"] = p.v / p.g
    p = p[(p.g >= 8) & (p.trailing > 5)]

    act = hist[(hist.season == target) & (hist.week == 1)
               & (hist.season_type == "REG") & hist.position.isin(positions)]
    act = act.groupby(["player_id", "player_display_name", "opponent_team"]).agg(
        actual=(stat, "sum")).reset_index()
    act["actual"] = act.actual.fillna(0)

    fac = defence_factors(through)
    d = act.merge(p[["player_id", "trailing", "position"]], on="player_id", how="inner")
    d = d.join(fac, on="opponent_team")
    for c in ("sr", "sr_adj", "epa"):
        if c in d.columns:
            d[c] = d[c].fillna(1.0).clip(0.80, 1.25)
            d[f"p_{c}"] = d.trailing * d[c]
    d["season"] = target
    return d


def report(stat, positions, label):
    frames = [build(t - 1, t, stat, positions) for t in (2022, 2023, 2024, 2025)]
    df = pd.concat(frames, ignore_index=True)
    cols = ["trailing"] + [c for c in ("p_sr", "p_sr_adj", "p_epa") if c in df.columns]
    d = df.dropna(subset=cols + ["actual"])
    print(f"\n=== {label}  (n={len(d)}) ===")
    print(f"{'model':<12}{'MAE':>8}{'corr':>8}{'dMAE':>8}{'dcorr':>8}{'seasons won':>13}")
    base = None
    for c in cols:
        e = d[c] - d.actual
        mae, r = e.abs().mean(), np.corrcoef(d[c], d.actual)[0, 1]
        if base is None:
            base = (mae, r)
            wins = ""
        else:
            w = sum(1 for t in (2022, 2023, 2024, 2025)
                    if (d[d.season == t][c] - d[d.season == t].actual).abs().mean()
                    < (d[d.season == t].trailing - d[d.season == t].actual).abs().mean())
            wins = f"{w}/4"
        print(f"{c:<12}{mae:8.2f}{r:8.3f}{mae-base[0]:+8.2f}{r-base[1]:+8.3f}{wins:>13}")
    if "sr" in df.columns:
        print(f"  factor spread: sr {df.sr.quantile(.1):.3f}-{df.sr.quantile(.9):.3f}", end="")
        if "sr_adj" in df.columns:
            print(f"   sr_adj {df.sr_adj.quantile(.1):.3f}-{df.sr_adj.quantile(.9):.3f}")
        else:
            print()


if __name__ == "__main__":
    report("receiving_yards", ("WR",), "WR receiving yards")
    report("receiving_yards", ("TE",), "TE receiving yards")
    report("rushing_yards", ("RB",), "RB rushing yards")
    report("receiving_yards", ("RB",), "RB receiving yards")
    report("passing_yards", ("QB",), "QB passing yards")
