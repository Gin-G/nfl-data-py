"""Does an opponent's run FRONT move a back's rushing projection?

"Opponent-adjusted defence beats raw allowed-per-game" (EXPERIMENTS.md) closed
tight-end and running-back *receiving* and left one door open in writing:

    Rushing still fails, for a plain reason: the factor is built from receiving
    yards allowed, a statement about pass defence.

So this is the rushing-specific version, built from the trench metrics whose
stability `trench_stability.py` just measured — defensive adjusted line yards
and stuff rate, which are statements about a front rather than a secondary.

Protocol follows the in-season model, not the preseason one: for each back-game
from week 6 on, the baseline is his own trailing production this season, and
the treatment multiplies it by a shrunk opponent factor computed only from
weeks before the game. Nothing in the factor can see the week it predicts.

Bar, same as the adjusted-defence work: MAE must improve in at least three of
the four seasons. Correlation alone can shuffle ranks without getting closer.

EPA per rush is deliberately absent. The factor here is a ratio to the league
mean, and league EPA per rush sits within rounding distance of zero, so the
ratio explodes rather than measuring anything. It needs an additive form, which
is a different experiment.
"""
import pathlib
import numpy as np
import pandas as pd

from trench_stability import CACHE, URL, SEASONS, band

TRAILING_GAMES = 5          # the production naive baseline's window
MIN_GAMES, MIN_CARRIES = 3, 25
COLS = ["season", "week", "season_type", "game_id", "posteam", "defteam",
        "rush_attempt", "qb_scramble", "qb_kneel", "two_point_attempt",
        "yards_gained", "epa", "rusher_player_id", "rusher_player_name"]


def carries(season):
    path = CACHE / f"pbp_{season}.parquet"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        pd.read_parquet(URL.format(s=season)).to_parquet(path)
    d = pd.read_parquet(path, columns=COLS)
    d = d[(d.season_type == "REG") & (d.rush_attempt == 1) & (d.qb_scramble != 1)
          & (d.qb_kneel != 1) & (d.two_point_attempt != 1)
          & d.rusher_player_id.notna() & d.yards_gained.notna()].copy()
    d["aly"] = d.yards_gained.map(band)
    d["stuff"] = (d.yards_gained <= 0).astype(float)
    return d


def factors(plays, season, week, side="defteam"):
    """Each team's to-date run-defence profile, as ratios to the league mean."""
    past = plays[(plays.season == season) & (plays.week < week)]
    if past.empty:
        return {}
    league = {m: past[m].mean() for m in ("aly", "stuff", "yards_gained", "epa")}
    g = past.groupby(side).agg(n=("yards_gained", "size"),
                               **{m: (m, "mean") for m in league})
    out = {}
    for team, row in g.iterrows():
        out[team] = {"n": row.n, **{m: row[m] / league[m] if league[m] else 1.0
                                    for m in league}}
    return out


def shrink(ratio, n, k):
    return 1.0 + (ratio - 1.0) * n / (n + k)


def build(plays):
    """One row per back-game with a usable trailing baseline, plus the factors."""
    per_game = (plays.groupby(["season", "week", "game_id", "rusher_player_id",
                               "rusher_player_name", "posteam", "defteam"])
                .agg(carries=("yards_gained", "size"),
                     rush_yards=("yards_gained", "sum")).reset_index()
                .sort_values(["season", "rusher_player_id", "week"]))

    rows = []
    for season in sorted(per_game.season.unique()):
        season_games = per_game[per_game.season == season]
        for week in sorted(season_games.week.unique()):
            if week < 6:
                continue
            fac = factors(plays, season, week)
            prior = season_games[season_games.week < week]
            trail = (prior.groupby("rusher_player_id")
                     .agg(games=("week", "size"),
                          carries_sum=("carries", "sum")).reset_index())
            recent = (prior.sort_values("week").groupby("rusher_player_id")
                      .tail(TRAILING_GAMES).groupby("rusher_player_id")
                      .rush_yards.mean().rename("baseline").reset_index())
            trail = trail.merge(recent, on="rusher_player_id")
            trail = trail[(trail.games >= MIN_GAMES)
                          & (trail.carries_sum >= MIN_CARRIES)]
            now = season_games[season_games.week == week].merge(
                trail, on="rusher_player_id")
            for _, r in now.iterrows():
                f = fac.get(r.defteam)
                if not f:
                    continue
                rows.append({"season": season, "week": week,
                             "player": r.rusher_player_name,
                             "actual": r.rush_yards, "baseline": r.baseline,
                             "n_def": f["n"],
                             **{f"r_{m}": f[m] for m in ("aly", "stuff",
                                                         "yards_gained", "epa")}})
    return pd.DataFrame(rows)


def score(d, factor, label, results):
    pred = d.baseline * factor
    per = []
    for season, g in d.assign(pred=pred).groupby("season"):
        per.append((float((g.pred - g.actual).abs().mean()),
                    float((g.baseline - g.actual).abs().mean())))
    wins = sum(1 for treat, base in per if treat < base)
    results.append({
        "treatment": label,
        "MAE": float((pred - d.actual).abs().mean()),
        "corr": float(np.corrcoef(pred, d.actual)[0, 1]),
        "MAE wins": f"{wins}/{len(per)}",
    })


if __name__ == "__main__":
    plays = pd.concat([carries(s) for s in SEASONS], ignore_index=True)
    print(f"{len(plays):,} non-QB carries, {SEASONS[0]}-{SEASONS[-1]}")
    d = build(plays)
    print(f"{len(d):,} back-games from week 6 on with a trailing baseline, "
          f"{d.player.nunique()} backs\n")

    base_mae = float((d.baseline - d.actual).abs().mean())
    base_corr = float(np.corrcoef(d.baseline, d.actual)[0, 1])
    print(f"baseline (trailing {TRAILING_GAMES} games): "
          f"MAE {base_mae:.3f}  corr {base_corr:.3f}\n")

    results = []
    for k in (150, 400):
        for metric, name in (("aly", "def adj line yards"),
                             ("stuff", "def stuff rate (inverted)"),
                             ("yards_gained", "def yards per carry")):
            ratio = d[f"r_{metric}"]
            if metric == "stuff":
                ratio = 1.0 / ratio.clip(lower=0.2)   # more stuffs = worse matchup
            score(d, shrink(ratio, d.n_def, k), f"{name}  (k={k})", results)
    out = pd.DataFrame(results)
    out["vs base"] = out.MAE - base_mae
    print(out.to_string(index=False,
                        formatters={"MAE": "{:.3f}".format, "corr": "{:.3f}".format,
                                    "vs base": "{:+.3f}".format}))
    print(f"\nbar: MAE wins 3/4 or better, and MAE below {base_mae:.3f}")
