"""Charted line play: yards before contact, and real pressure instead of a proxy.

`trenches.py` grades four units off play-by-play alone, and says why in its own
docstring: *"adjusted line yards uses FO's banding because the first four yards
of a carry belong to the line and the eleventh does not, and pressure is
sack-or-hit per dropback because nflverse publishes no pressure field."*

Both of those are workarounds for a measurement that does exist. Pro Football
Reference charts, and nflverse republishes weekly 2018-, the thing the banding
is a proxy FOR:

    rushing_yards_before_contact    where the carry was actually met
    rushing_yards_after_contact     what the back did once it was
    rushing_broken_tackles          ditto, more directly
    times_pressured                 charted pressure, not sack-or-hit
    def_pressures                   the same from the defender's side

The banding is a *positional* proxy: it assumes the line owns yards 0-4 and the
back owns 11+, which is true on average and wrong on any given carry. Yards
before contact is the quantity itself. If the proxy were as good as the
measurement there would be nothing here; the point of this file is to find out,
on the same bar.

`nflreadpy` 0.1.3 has no `load_pfr_advstats` (the R package does), so the
release assets are read directly — one parquet per season per stat type.

Protocol is lifted from `experiments/trench_stability.py` unchanged, so the
numbers drop into the same table as the ones in EXPERIMENTS.md, "The trench
report": split-half within season over 2022-25, min 30 plays a side, and
year-over-year over 2018-25 for the roster-turnover question. QB carries are
excluded from every rushing metric, because a scramble is a protection failure
rather than a run-blocking rep and counting it scores one play twice in
opposite directions.

    python experiments/trench_charted.py
"""
import pathlib

import numpy as np
import pandas as pd

CACHE = pathlib.Path(__file__).parent / "_cache"
RELEASE = ("https://github.com/nflverse/nflverse-data/releases/download/"
           "pfr_advstats/advstats_week_%s_%d.parquet")
SEASONS = tuple(range(2018, 2026))
SPLIT_SEASONS = (2022, 2023, 2024, 2025)   # the window the ledger table uses
MIN_PLAYS = 30
MIN_YOY = 100


def advstats(kind):
    """Weekly PFR charted stats for one stat type, cached."""
    path = CACHE / f"pfr_{kind}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frames = []
    for season in SEASONS:
        try:
            frames.append(pd.read_parquet(RELEASE % (kind, season)))
        except Exception as exc:
            print(f"  {kind} {season}: {type(exc).__name__}")
    out = pd.concat(frames, ignore_index=True)
    CACHE.mkdir(exist_ok=True)
    out.to_parquet(path)
    return out


def positions():
    """pfr_id -> position, so QB carries can be dropped."""
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


def pbp_team_weeks():
    """The play-by-play metrics `trenches.py` ships, aggregated to team-weeks.

    Recomputed here rather than quoted from the ledger entry, because that
    entry's split-half ran over raw plays and this one runs over team-week
    rates. The numbers are close but not identical, and a comparison between a
    new metric measured one way and an old one measured another way is not a
    comparison. These are the apples-to-apples column.
    """
    path = CACHE / "pbp_trench_weeks.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from nfl_projections import trenches

    rush, drop = trenches.plays(list(SEASONS))
    r = (rush.groupby(["season", "week", "posteam", "defteam"])
         .agg(pbp_carries=("aly", "size"), aly=("aly", "sum"),
              stuffs=("stuffed", "sum")).reset_index())
    d = (drop.groupby(["season", "week", "posteam", "defteam"])
         .agg(pbp_dropbacks=("pressured", "size"),
              proxy_pressures=("pressured", "sum")).reset_index())
    out = r.merge(d, on=["season", "week", "posteam", "defteam"], how="outer")
    out.to_parquet(path)
    return out


def dropbacks():
    """Team dropbacks per week, for the pressure-rate denominator.

    From play-by-play rather than PFR's own percentage: `times_pressured_pct`
    has a zero denominator for a backup who threw twice, and the whole point of
    this comparison is that both rates sit on the same plays.
    """
    path = CACHE / "dropbacks.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from nfl_projections.pbp import load_pbp

    pbp = load_pbp(list(SEASONS))
    d = pbp[(pbp.get("season_type", "REG") == "REG") & pbp.posteam.notna()
            & pbp.defteam.notna()]
    d = d[d.qb_dropback.fillna(0) == 1]
    out = (d.groupby(["season", "week", "posteam", "defteam"]).size()
           .rename("dropbacks").reset_index())
    out.to_parquet(path)
    return out


def rush_frame():
    """Team-week rushing charted totals, QB carries removed."""
    r = advstats("rush")
    r = r[r.game_type == "REG"].merge(positions(), on="pfr_player_id", how="left")
    r = r[r.position != "QB"]
    g = (r.groupby(["season", "week", "team", "opponent"])
         .agg(carries=("carries", "sum"),
              ybc=("rushing_yards_before_contact", "sum"),
              yac=("rushing_yards_after_contact", "sum"),
              broken=("rushing_broken_tackles", "sum"))
         .reset_index())
    g = g[g.carries > 0]
    for col in ("ybc", "yac", "broken"):
        g[f"{col}_per_carry"] = g[col] / g.carries
    g = g.rename(columns={"team": "posteam", "opponent": "defteam"})
    pbp = pbp_team_weeks()
    return g.merge(pbp[["season", "week", "posteam", "defteam",
                        "pbp_carries", "aly", "stuffs"]],
                   on=["season", "week", "posteam", "defteam"], how="inner")


def pass_frame():
    """Team-week pressure allowed, and pressure generated, per dropback."""
    p = advstats("pass")
    p = p[p.game_type == "REG"]
    allowed = (p.groupby(["season", "week", "team", "opponent"])
               .agg(pressured=("times_pressured", "sum"),
                    blitzed=("times_blitzed", "sum"),
                    hurried=("times_hurried", "sum"),
                    hit=("times_hit", "sum"),
                    sacked=("times_sacked", "sum"))
               .reset_index()
               .rename(columns={"team": "posteam", "opponent": "defteam"}))

    dfn = advstats("def")
    dfn = dfn[dfn.game_type == "REG"]
    gen = (dfn.groupby(["season", "week", "team", "opponent"])
           .agg(def_pressures=("def_pressures", "sum"),
                def_sacks=("def_sacks", "sum"),
                missed_tackles=("def_missed_tackles", "sum"))
           .reset_index()
           .rename(columns={"team": "defteam", "opponent": "posteam"}))

    db = dropbacks()
    out = allowed.merge(gen, on=["season", "week", "posteam", "defteam"], how="outer")
    out = out.merge(db, on=["season", "week", "posteam", "defteam"], how="inner")
    out = out[out.dropbacks > 0]
    for col in ("pressured", "blitzed", "hurried", "hit", "sacked",
                "def_pressures", "def_sacks"):
        out[f"{col}_rate"] = out[col] / out.dropbacks
    pbp = pbp_team_weeks()
    out = out.merge(pbp[["season", "week", "posteam", "defteam",
                         "pbp_dropbacks", "proxy_pressures"]],
                    on=["season", "week", "posteam", "defteam"], how="inner")
    return out[out.pbp_dropbacks > 0]


def split_half(d, key, metric, weight, min_plays=MIN_PLAYS):
    """Correlate a team's trait between the first and second half of a season.

    Verbatim from trench_stability.split_half except that the metric is a
    weighted rate — a team-week mean of per-carry averages would weight a
    three-carry game like a thirty-carry one, and these frames are already
    aggregated to team-weeks where that harness worked on raw plays.
    """
    d = d[d.season.isin(SPLIT_SEASONS)]
    mid = d.groupby("season").week.median()
    d = d.assign(half=np.where(d.week <= d.season.map(mid), "early", "late"))
    g = (d.groupby(["season", "half", key])
         .apply(lambda x: pd.Series({"val": x[metric].sum() / x[weight].sum(),
                                     "n": x[weight].sum()}), include_groups=False)
         .reset_index())
    g = g[g.n >= min_plays]
    w = g.pivot_table(index=["season", key], columns="half", values="val").dropna()
    if len(w) < 20 or "early" not in w or "late" not in w:
        return float("nan"), len(w)
    return float(np.corrcoef(w["early"], w["late"])[0, 1]), len(w)


def year_over_year(d, key, metric, weight, min_plays=MIN_YOY):
    g = (d.groupby(["season", key])
         .apply(lambda x: pd.Series({"val": x[metric].sum() / x[weight].sum(),
                                     "n": x[weight].sum()}), include_groups=False)
         .reset_index())
    g = g[g.n >= min_plays]
    nxt = g.assign(season=g.season - 1).rename(columns={"val": "next"})
    m = g.merge(nxt[["season", key, "next"]], on=["season", key])
    if len(m) < 20:
        return float("nan"), len(m)
    return float(np.corrcoef(m.val, m["next"])[0, 1]), len(m)


# metric -> (numerator, denominator, what it is meant to measure)
RUSH_METRICS = [
    ("CHARTED yards before contact / carry", "ybc", "carries", "the line"),
    ("CHARTED yards after contact / carry", "yac", "carries", "the back"),
    ("CHARTED broken tackles / carry", "broken", "carries", "the back"),
    ("shipped: adjusted line yards (FO banding)", "aly", "pbp_carries", "the line"),
    ("shipped: stuff rate", "stuffs", "pbp_carries", "the line"),
]
PASS_METRICS = [
    ("CHARTED pressure rate", "pressured", "dropbacks", "protection / rush"),
    ("CHARTED hurry rate", "hurried", "dropbacks", "protection / rush"),
    ("CHARTED blitz rate faced", "blitzed", "dropbacks", "scheme, not the line"),
    ("shipped: pressure proxy (sack or hit)", "proxy_pressures", "pbp_dropbacks",
     "protection / rush"),
    ("shipped: sack rate", "sacked", "dropbacks", "protection / rush"),
]


def report():
    rush, passing = rush_frame(), pass_frame()
    print("PFR charted coverage: %d team-weeks rushing, %d passing, %d-%d"
          % (len(rush), len(passing), rush.season.min(), rush.season.max()))

    print("\nSPLIT-HALF WITHIN SEASON, 2022-25 — same protocol as the ledger table")
    print("%-42s %9s %9s   %s" % ("metric", "offence", "defence", "owns"))
    rows = []
    for label, num, den, owns in RUSH_METRICS:
        o, no = split_half(rush, "posteam", num, den)
        d, nd = split_half(rush, "defteam", num, den)
        rows.append((label, o, d, owns, no, nd))
    for label, num, den, owns in PASS_METRICS:
        o, no = split_half(passing, "posteam", num, den)
        d, nd = split_half(passing, "defteam", num, den)
        rows.append((label, o, d, owns, no, nd))
    o, no = split_half(passing, "defteam", "def_pressures", "dropbacks")
    rows.append(("pressures generated (defender side)", float("nan"), o,
                 "the rush", 0, no))
    for label, o, d, owns, no, nd in rows:
        print("%-42s %9s %9s   %s" % (
            label,
            "-" if np.isnan(o) else "%.3f" % o,
            "-" if np.isnan(d) else "%.3f" % d, owns))

    print("\nYEAR OVER YEAR, 2018-25 — does it survive roster turnover")
    print("%-42s %9s %9s" % ("metric", "offence", "defence"))
    for label, num, den, _ in RUSH_METRICS + PASS_METRICS:
        frame = rush if (label, num, den, _) in RUSH_METRICS else passing
        o, _n1 = year_over_year(frame, "posteam", num, den)
        d, _n2 = year_over_year(frame, "defteam", num, den)
        print("%-42s %9s %9s" % (label,
                                 "-" if np.isnan(o) else "%.3f" % o,
                                 "-" if np.isnan(d) else "%.3f" % d))

    print("\nIS THE CHARTED PRESSURE THE SAME NUMBER AS THE PROXY?")
    p = passing.copy()
    p["proxy"] = (p.sacked + p.hit) / p.dropbacks
    p["charted"] = p.pressured / p.dropbacks
    print("  per team-week: charted mean %.3f, proxy mean %.3f, correlation %.3f"
          % (p.charted.mean(), p.proxy.mean(), p.charted.corr(p.proxy)))
    print("  charted pressure is %.1fx the volume of the proxy, so the same"
          % (p.pressured.sum() / (p.sacked.sum() + p.hit.sum())))
    print("  share of a season's signal rests on more events")

    print("\nHOW MUCH OF A CARRY DOES EACH SIDE OWN (team-season means)")
    s = (rush.groupby("season")[["ybc", "yac", "carries"]].sum())
    s["ybc_per_carry"] = s.ybc / s.carries
    s["yac_per_carry"] = s.yac / s.carries
    s["ybc_share"] = s.ybc / (s.ybc + s.yac)
    print(s[["carries", "ybc_per_carry", "yac_per_carry", "ybc_share"]].round(3).to_string())


if __name__ == "__main__":
    report()
