"""Is the LINE OF SCRIMMAGE a stable trait?

Hand-charted line play is sold as a subscription product: offensive lines vs
defensive fronts, and receivers vs coverages. The coverage half of that is
already settled in EXPERIMENTS.md ("Coverage matchup") — it
works for tight ends and fails for receivers because nflverse publishes no
shadow-coverage assignment. The *trench* half has never been tested here at
all, and it is the one matchup axis with a plausible claim to stability: a
line is five players under contract together, not a weekly assignment.

This is the same split-half protocol as "Granular defensive matchup" so the
numbers sit on the same scale, and the same question is asked first: is it a
trait at all? A metric that does not correlate with itself inside one season
cannot carry a weekly projection.

Metrics, per team, as an offence trait and again as a defence trait:

    adj_line_yards   Football-Outsiders banding of rushing yards — losses count
                     1.2x, 0-4 yards fully, 5-10 half, 11+ not at all. The
                     bands are the whole point: the first four yards are the
                     line's, the eleventh is the back's.
    stuff_rate       carries stopped at or behind the line
    second_level     yards 5-10, which belong to line and back together
    open_field       yards 11+, which belong to the back
    sack_rate        sacks per dropback
    pressure_rate    sacks plus QB hits per dropback (nflverse has no pressure
                     field; a hit is the closest public proxy)

QB runs are excluded from the rushing metrics — a scramble is a failure of
protection, not a run-blocking rep, and FO excludes them for the same reason.

Baselines to beat, from the prior work in EXPERIMENTS.md:
    team defence-vs-position (fantasy pts allowed)   r = 0.06 - 0.25
    team success rate allowed                        r = 0.618
    most stable trait measured anywhere (FTN role)   r = 0.748
"""
import pathlib
import numpy as np
import pandas as pd

CACHE = pathlib.Path(__file__).parent / "_cache"
URL = ("https://github.com/nflverse/nflverse-data/releases/download/pbp/"
       "play_by_play_{s}.parquet")
SEASONS = (2022, 2023, 2024, 2025)
COLS = ["season", "week", "season_type", "posteam", "defteam", "play_type",
        "yards_gained", "rush_attempt", "pass_attempt", "qb_dropback",
        "qb_scramble", "qb_kneel", "sack", "qb_hit", "epa", "success",
        "two_point_attempt", "run_location", "run_gap"]


def load(season):
    path = CACHE / f"pbp_{season}.parquet"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        pd.read_parquet(URL.format(s=season)).to_parquet(path)
    d = pd.read_parquet(path, columns=COLS)
    return d[(d.season_type == "REG") & d.posteam.notna() & d.defteam.notna()]


def band(y):
    """Adjusted-line-yards weight for one carry."""
    if y < 0:
        return 1.2 * y
    if y <= 4:
        return float(y)
    if y <= 10:
        return 4 + 0.5 * (y - 4)
    return 7.0


def prepare(d):
    """Split into a rushing frame and a dropback frame, with metrics attached."""
    rush = d[(d.rush_attempt == 1) & (d.qb_scramble != 1) & (d.qb_kneel != 1)
             & (d.two_point_attempt != 1) & d.yards_gained.notna()].copy()
    y = rush.yards_gained
    rush["adj_line_yards"] = y.map(band)
    rush["stuff_rate"] = (y <= 0).astype(float)
    rush["second_level"] = (y.clip(lower=4, upper=10) - 4).clip(lower=0)
    rush["open_field"] = (y - 10).clip(lower=0)
    rush["gap"] = rush.run_location.fillna("?") + "/" + rush.run_gap.fillna("end")

    drop = d[(d.qb_dropback == 1) & (d.two_point_attempt != 1)].copy()
    drop["sack_rate"] = drop.sack.fillna(0).astype(float)
    drop["pressure_rate"] = ((drop.sack.fillna(0) == 1)
                             | (drop.qb_hit.fillna(0) == 1)).astype(float)
    return rush, drop


def split_half(d, keys, metric, min_plays=30):
    """Correlate a team's trait between the first and second half of a season."""
    mid = d.groupby("season").week.median()
    d = d.assign(half=np.where(d.week <= d.season.map(mid), "early", "late"))
    g = (d.groupby(["season", "half"] + keys)
          .agg(val=(metric, "mean"), n=(metric, "size")).reset_index())
    g = g[g.n >= min_plays]
    w = g.pivot_table(index=["season"] + keys, columns="half", values="val").dropna()
    if len(w) < 20 or "early" not in w or "late" not in w:
        return float("nan"), len(w)
    return float(np.corrcoef(w["early"], w["late"])[0, 1]), len(w)


def year_over_year(d, side, metric, min_plays=100):
    """Does the trait carry from one season to the next? (roster turnover test)"""
    g = (d.groupby(["season", side]).agg(val=(metric, "mean"), n=(metric, "size"))
          .reset_index())
    g = g[g.n >= min_plays]
    nxt = g.assign(season=g.season - 1).rename(columns={"val": "next"})
    m = g.merge(nxt[["season", side, "next"]], on=["season", side])
    if len(m) < 20:
        return float("nan"), len(m)
    return float(np.corrcoef(m.val, m["next"])[0, 1]), len(m)


if __name__ == "__main__":
    d = pd.concat([load(s) for s in SEASONS], ignore_index=True)
    rush, drop = prepare(d)
    print(f"{len(rush):,} non-QB carries and {len(drop):,} dropbacks, "
          f"{SEASONS[0]}-{SEASONS[-1]}\n")

    tests = [
        ("adj_line_yards", rush, 60),
        ("stuff_rate", rush, 60),
        ("second_level", rush, 60),
        ("open_field", rush, 60),
        ("sack_rate", drop, 80),
        ("pressure_rate", drop, 80),
    ]

    print("SPLIT-HALF within a season (is it a trait?)")
    print(f"{'metric':18s}{'offence r':>11}{'n':>6}{'defence r':>11}{'n':>6}")
    for metric, frame, mp in tests:
        o_r, o_n = split_half(frame, ["posteam"], metric, mp)
        d_r, d_n = split_half(frame, ["defteam"], metric, mp)
        print(f"{metric:18s}{o_r:11.3f}{o_n:6d}{d_r:11.3f}{d_n:6d}")

    print("\nYEAR OVER YEAR (does it survive roster turnover?)")
    print(f"{'metric':18s}{'offence r':>11}{'n':>6}{'defence r':>11}{'n':>6}")
    for metric, frame, mp in tests:
        o_r, o_n = year_over_year(frame, "posteam", metric)
        d_r, d_n = year_over_year(frame, "defteam", metric)
        print(f"{metric:18s}{o_r:11.3f}{o_n:6d}{d_r:11.3f}{d_n:6d}")

    print("\nBY GAP — the resolution the charting services work at (offence)")
    print(f"{'metric':18s}{'team r':>9}{'team x gap r':>14}{'pairs':>7}")
    for metric in ("adj_line_yards", "stuff_rate"):
        t_r, _ = split_half(rush, ["posteam"], metric, 60)
        g_r, g_n = split_half(rush, ["posteam", "gap"], metric, 20)
        print(f"{metric:18s}{t_r:9.3f}{g_r:14.3f}{g_n:7d}")

    print("\nFor comparison, from EXPERIMENTS.md:")
    print("  team defence-vs-position (fantasy pts allowed)      0.06 - 0.25")
    print("  team success rate allowed                           0.618")
    print("  most stable trait measured anywhere (FTN role)      0.748")
