"""Do Next Gen Stats add anything the model does not already have?

The feature list already carries a lot of what gets called "advanced": EPA by
position, racr, pacr, dakota, target_share, air_yards_share, wopr, snap counts
and snap share, plus derived yards-per-target, epa-per-target and average target
depth. So the question is not "are we using advanced metrics" — we are — but
"which ones are we missing, and do they carry signal the rest does not".

DVOA itself is not obtainable: it is FTN/Football Outsiders' proprietary metric
and is not published in any free feed. The closest public equivalents are the
Next Gen Stats efficiency numbers, which nflverse does publish and which the
model does not touch:

  receiving   avg_separation, avg_cushion, avg_intended_air_yards,
              percent_share_of_intended_air_yards, catch_percentage,
              avg_yac_above_expectation
  rushing     efficiency, rush_yards_over_expected_per_att,
              percent_attempts_gte_eight_defenders, avg_time_to_los

The test is incremental signal, not raw correlation. A metric that correlates
with production but only because it correlates with volume adds nothing. What
matters is whether it predicts the part of next season the player's own
trailing rate gets wrong — the residual.
"""
import pathlib
import numpy as np
import pandas as pd

D = pathlib.Path(__file__).parent
CACHE = pathlib.Path.home() / ".sharp-edge" / "nfl"
WEEKLY = ("https://github.com/nflverse/nflverse-data/releases/download/"
          "stats_player/stats_player_week_{s}.parquet")
NGS = ("https://github.com/nflverse/nflverse-data/releases/download/"
       "nextgen_stats/ngs_{kind}.parquet")

REC_METRICS = ["avg_separation", "avg_cushion", "avg_intended_air_yards",
               "percent_share_of_intended_air_yards", "catch_percentage",
               "avg_yac_above_expectation"]
RUSH_METRICS = ["efficiency", "rush_yards_over_expected_per_att",
                "percent_attempts_gte_eight_defenders", "avg_time_to_los",
                "rush_pct_over_expected"]


def weekly(seasons):
    out = []
    for s in seasons:
        p = CACHE / f"week_{s}.parquet"
        out.append(pd.read_parquet(p) if p.exists() else pd.read_parquet(WEEKLY.format(s=s)))
    return pd.concat(out, ignore_index=True)


def ngs(kind):
    p = D / f"ngs_{kind}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    df = pd.read_parquet(NGS.format(kind=kind))
    df.to_parquet(p)
    return df


def season_ngs(kind, season, metrics):
    """Season-level NGS per player. week == 0 is nflverse's season roll-up."""
    d = ngs(kind)
    d = d[(d.season == season) & (d.season_type == "REG")]
    roll = d[d.week == 0]
    if roll.empty:                       # fall back to averaging the weeks
        roll = d.groupby("player_gsis_id")[metrics].mean().reset_index()
    else:
        roll = roll[["player_gsis_id"] + metrics]
    return roll.rename(columns={"player_gsis_id": "player_id"})


def build(through, target, position, stat, kind, metrics):
    hist = weekly(range(through, target + 1))
    hist[stat] = hist[stat].fillna(0)
    prior = hist[(hist.season == through) & (hist.season_type == "REG")]

    p = prior[prior.position == position].groupby("player_id").agg(
        g=("week", "count"), v=(stat, "sum")).reset_index()
    p["trailing"] = p.v / p.g
    p = p[(p.g >= 8) & (p.trailing > 5)]

    act = hist[(hist.season == target) & (hist.week == 1)
               & (hist.season_type == "REG") & (hist.position == position)]
    act = act.groupby("player_id").agg(actual=(stat, "sum")).reset_index()
    act["actual"] = act.actual.fillna(0)

    d = act.merge(p[["player_id", "trailing"]], on="player_id", how="inner")
    d = d.merge(season_ngs(kind, through, metrics), on="player_id", how="left")
    d["resid"] = d.actual - d.trailing
    d["season"] = target
    return d


def report(position, stat, kind, metrics, label):
    frames = [build(t - 1, t, position, stat, kind, metrics)
              for t in (2022, 2023, 2024, 2025)]
    df = pd.concat(frames, ignore_index=True)
    have = df.dropna(subset=["trailing", "actual"])
    print(f"\n=== {label}  (n={len(have)}, with NGS: "
          f"{have[metrics].notna().all(axis=1).sum()}) ===")
    print(f"{'metric':42s}{'r vs actual':>13}{'r vs residual':>15}")
    for m in metrics:
        s = have.dropna(subset=[m])
        if len(s) < 40:
            print(f"{m:42s}{'too few':>13}{'':>15}")
            continue
        r_act = np.corrcoef(s[m], s.actual)[0, 1]
        r_res = np.corrcoef(s[m], s.resid)[0, 1]
        flag = "  <-- incremental" if abs(r_res) > 0.10 else ""
        print(f"{m:42s}{r_act:13.3f}{r_res:15.3f}{flag}")
    print("  (r vs actual can be volume in disguise; r vs residual is what the"
          " trailing rate misses)")


if __name__ == "__main__":
    report("RB", "rushing_yards", "rushing", RUSH_METRICS, "RB rushing yards")
    report("WR", "receiving_yards", "receiving", REC_METRICS, "WR receiving yards")
    report("TE", "receiving_yards", "receiving", REC_METRICS, "TE receiving yards")
