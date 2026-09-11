"""Is a defence's behaviour a stable trait at the FORMATION level?

The prior null result (EXPERIMENTS.md, "Why opponent/matchup keeps failing")
tested defence-vs-position at the team level — fantasy points allowed to a
position — and found split-half r of 0.06 to 0.25, i.e. barely a trait at all.

That is one crude aggregate, and it does not rule out the granular version:
that a defence is reliably good against 11 personnel and soft against 12, or
reliably beaten on the ground when it puts six in the box. If *that* persists,
there is an independent edge to model. If it does not, the null result extends
and the honest conclusion is that per-game defensive matchup is mostly noise
whatever resolution you look at.

Same method as the prior test so the numbers are comparable: split each season
into early and late halves by week, compute the trait in each half, correlate
across teams. A trait that does not correlate with itself cannot predict
anything.

Baselines to beat, from the prior work:
    team defence-vs-position      r = 0.06 - 0.25
    opponent-adjusted team SRS    r = 0.264 (defence), 0.489 (offence)
"""
import pathlib, re
import numpy as np
import pandas as pd

D = pathlib.Path(__file__).parent
PART = ("https://github.com/nflverse/nflverse-data/releases/download/"
        "pbp_participation/pbp_participation_{s}.parquet")
PBP = ("https://github.com/nflverse/nflverse-data/releases/download/pbp/"
       "play_by_play_{s}.parquet")
SEASONS = (2022, 2023, 2024, 2025)


def _cached(url, name, **kw):
    p = D / name
    if p.exists():
        return pd.read_parquet(p)
    df = pd.read_parquet(url, **kw)
    df.to_parquet(p)
    return df


def grouping(s):
    if not isinstance(s, str):
        return None
    rb, te = re.search(r"(\d+) RB", s), re.search(r"(\d+) TE", s)
    return f"{rb.group(1)}{te.group(1)}" if rb and te else None


def load(season):
    part = _cached(PART.format(s=season), f"gpart_{season}.parquet",
                   columns=["nflverse_game_id", "play_id", "offense_personnel",
                            "offense_formation", "defenders_in_box",
                            "defense_man_zone_type"])
    pbp = _cached(PBP.format(s=season), f"pbpfull_{season}.parquet",
                  columns=["game_id", "play_id", "week", "posteam", "defteam",
                           "epa", "rush_attempt", "pass_attempt", "success",
                           "yards_gained"])
    d = part.rename(columns={"nflverse_game_id": "game_id"}).merge(
        pbp, on=["game_id", "play_id"], how="inner")
    d["pers"] = d.offense_personnel.map(grouping)
    d["season"] = season
    return d[d.epa.notna()]


def split_half(d, keys, metric, min_plays=30):
    """Correlate a defence's trait between the first and second half of a season.

    Returns (r, n_pairs). ``keys`` is what the trait is measured within — team
    alone reproduces the prior test, team+personnel is the granular version.
    """
    mid = d.groupby("season").week.median()
    d = d.assign(half=np.where(d.week <= d.season.map(mid), "early", "late"))
    g = d.groupby(["season", "half"] + keys).agg(
        val=(metric, "mean"), n=(metric, "size")).reset_index()
    g = g[g.n >= min_plays]
    w = g.pivot_table(index=["season"] + keys, columns="half", values="val")
    w = w.dropna()
    if len(w) < 20:
        return float("nan"), len(w)
    return float(np.corrcoef(w["early"], w["late"])[0, 1]), len(w)


if __name__ == "__main__":
    frames = [load(s) for s in SEASONS]
    d = pd.concat(frames, ignore_index=True)
    print(f"{len(d):,} plays with participation + epa, {SEASONS[0]}-{SEASONS[-1]}\n")

    rush = d[d.rush_attempt == 1]
    pas = d[d.pass_attempt == 1]

    print("Split-half stability of a DEFENCE's trait (higher = more of a real trait):")
    print(f"{'trait':52s}{'r':>8}{'pairs':>8}")

    tests = [
        ("team: epa allowed per play (all)", d, ["defteam"], "epa", 200),
        ("team: epa allowed per rush", rush, ["defteam"], "epa", 100),
        ("team: epa allowed per pass", pas, ["defteam"], "epa", 100),
        ("team: success rate allowed", d, ["defteam"], "success", 200),
        ("team x offensive personnel: epa allowed", d, ["defteam", "pers"], "epa", 30),
        ("team x personnel, rush only: epa allowed", rush, ["defteam", "pers"], "epa", 25),
        ("team x personnel, pass only: epa allowed", pas, ["defteam", "pers"], "epa", 25),
        ("team x offensive formation: epa allowed", d, ["defteam", "offense_formation"], "epa", 30),
        ("team x man/zone: epa allowed", pas, ["defteam", "defense_man_zone_type"], "epa", 30),
    ]
    for label, frame, keys, metric, mp in tests:
        r, n = split_half(frame, keys, metric, min_plays=mp)
        print(f"{label:52s}{r:8.3f}{n:8d}")

    print("\nFor comparison, from the prior work:")
    print("  team defence-vs-position (fantasy pts allowed)      0.06 - 0.25")
    print("  opponent-adjusted team SRS, defence                 0.264")
    print("  opponent-adjusted team SRS, offence                 0.489")
