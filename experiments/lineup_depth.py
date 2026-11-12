"""Does going further down a build cost points? The depth test.

Question: inside one cash build, do the lineups the builder ranks first score
more than the ones it ranks 11th and below -- by enough that a cash entry
should repeat its top few rather than walk down the ladder?

Kill rule, fixed before running: abandon "go shallow" if the mean weekly
top-5 minus ranks-11-25 gap is under 5 points, or if it is positive in fewer
than two thirds of weeks.

Two arms.

* backtest, 2021-2025 weeks 4-18: the cached leakage-free board
  (`evaluate.backtest`, 1 seed, 60 epochs, trained on prior seasons). Salaries
  are a proxy: each position's real 2026 FanDuel salary regressed on FPPG,
  applied to the player's season-to-date average. Caveats: the backtest board
  only holds players who recorded a stat line, so nobody in the pool is
  inactive (an availability oracle that flatters every lineup equally), and
  the proxy price carries none of FanDuel's own information.
* real, 2026 weeks 6-9: the saved FanDuel slate and the board as it stood at
  kickoff, both from production's /optimizer/pool, scored on nflverse's weekly
  stats through the package's own FanDuel scoring. A projected player with no
  stat line scored 0, which is what FanDuel paid him. (Production's frozen
  accuracy rows agree with nflverse on who has a line; they are not used here
  only because the API cannot page out the rows whose error is exactly zero.)

Both arms build the eight skill slots under $56,000 (the cap less a typical
defense), because the backtest board projects no defenses.

    python experiments/lineup_depth.py            # runs and prints
"""
import json
import pathlib
import sys
import urllib.request
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from nfl_projections import optimizer as opt  # noqa: E402

CACHE = pathlib.Path(__file__).resolve().parent / "_cache"
API = "https://nfl-api.nickknows.net"
SEASONS = (2021, 2022, 2023, 2024, 2025)
WEEKS = range(4, 19)
REAL_WEEKS = (6, 7, 8, 9)
N_LINEUPS = 25
CAP = 56000
SKILL_SLOTS = {k: v for k, v in opt.ROSTER_SLOTS.items() if k != "DEF"}
AGGRESSIONS = (0, 50, 100)
HEAD, TAIL_FROM = 5, 11
TEAM_FIX = {"JAC": "JAX", "LAR": "LA", "WSH": "WAS"}


def _get(path):
    req = urllib.request.Request(API + path, headers={"User-Agent": "nfl-projections-research"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _cached_json(name, path):
    f = CACHE / f"{name}.json"
    if not f.exists():
        f.write_text(json.dumps(_get(path)))
    return json.loads(f.read_text())


def real_pools():
    return {w: pd.DataFrame(_cached_json(f"pool_2026_w{w}",
                                         f"/optimizer/pool?season=2026&week={w}")["data"])
            for w in REAL_WEEKS}


def accuracy_rows(season=2026, weeks=range(1, 10)):
    """Every frozen scored row, paged by week/position/direction under the cap."""
    f = CACHE / f"accuracy_{season}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    rows = []
    for w in weeks:
        for pos in ("QB", "RB", "WR", "TE"):
            for d in ("over", "under"):
                got = _get(f"/projections/accuracy/misses?season={season}&week={w}"
                           f"&position={pos}&direction={d}&limit=200")
                data = got.get("data") or []
                assert len(data) < 200, (w, pos, d)   # a full page would mean a cut
                rows += data
    out = pd.DataFrame(rows).drop_duplicates(["season", "week", "player_id"])
    out.to_parquet(f)
    return out


def salary_model(pools):
    """Per position: salary = a + b * FPPG, clipped to the observed range."""
    frame = pd.concat(pools.values(), ignore_index=True)
    frame = frame[frame.position.isin(["QB", "RB", "WR", "TE"]) & (frame.fppg > 0)]
    model = {}
    for pos, g in frame.groupby("position"):
        b, a = np.polyfit(g.fppg, g.salary, 1)
        r = np.corrcoef(g.fppg, g.salary)[0, 1]
        model[pos] = dict(a=a, b=b, lo=g.salary.min(), hi=g.salary.max(), r=r, n=len(g))
    return model


def price(model, pos, fppg):
    m = model[pos]
    return float(np.clip(round((m["a"] + m["b"] * fppg) / 100) * 100, m["lo"], m["hi"]))


def backtest_pools(model):
    """{(season, week): pool} built from the cached boards."""
    out = {}
    for season in SEASONS:
        b = pd.read_parquet(CACHE / f"blitz_board_{season}_1s60e.parquet")
        b = b[b.position.isin(["QB", "RB", "WR", "TE"])].sort_values(["player_id", "week"])
        # The 2021-2024 boards repeat ~1,150 player-weeks a season as exact
        # copies (the dataset's depth-chart merge). Left in, the builder can
        # put one man in two flex slots and the season-to-date average counts
        # his games twice.
        b = b.drop_duplicates(["player_id", "week"])
        # Season to date, strictly before the week: what FanDuel's FPPG would show.
        b["fppg"] = (b.groupby("player_id").actual
                      .transform(lambda s: s.shift().expanding().mean()))
        b["fppg"] = b.fppg.fillna(b.predicted)
        for week in WEEKS:
            w = b[b.week == week].copy()
            if w.empty:
                continue
            w["Salary"] = [price(model, p, f) for p, f in zip(w.position, w.fppg)]
            out[(season, week)] = pd.DataFrame({
                "Nickname": w.player_id, "Position": w.position,
                "Roster Position": w.position, "Team": w.team,
                "Salary": w.Salary, "FPPG": w.fppg,
                "fanduel_fantasy_points": w.predicted, "actual": w.actual,
            }).reset_index(drop=True)
    return out


def nflverse_points(season=2026):
    """{(week, normalized name|team): FanDuel points} from the cached weekly stats."""
    from nfl_projections.opportunity import _points
    s = pd.read_parquet(CACHE / f"stats_{season}_w9.parquet")
    s = s[s.get("season_type", "REG").fillna("REG") == "REG"] if "season_type" in s else s
    s["points"] = _points(s)
    s = s[s.player_display_name.notna() & s.team.notna()]
    return {(int(w), opt.normalize_name(n) + "|" + t): float(pt)
            for w, n, t, pt in zip(s.week, s.player_display_name, s.team, s.points)}


def real_pool_frames(pools, acc):
    out = {}
    for week, p in pools.items():
        p = p[p.position.isin(["QB", "RB", "WR", "TE"])
              & p.projected_points.notna()
              & ~p.injury_flag.fillna("").isin(opt.EXCLUDED_INJURY_STATUSES)].copy()
        p["team"] = p.team.replace(TEAM_FIX)
        actual = {k: v for (w, k), v in acc.items() if w == week}
        keys = p.nickname.map(opt.normalize_name) + "|" + p.team
        out[(2026, week)] = pd.DataFrame({
            "Nickname": p.nickname, "Position": p.position,
            "Roster Position": p.position, "Team": p.team,
            "Salary": p.salary, "FPPG": p.fppg,
            "fanduel_fantasy_points": p.projected_points,
            "actual": [actual.get(k, 0.0) for k in keys],
            "matched": [k in actual for k in keys],
        }).reset_index(drop=True)
    return out


def build_and_score(args):
    key, pool, aggression = args
    settings = opt.contest_settings("cash", aggression, num_lineups=N_LINEUPS)
    settings.pop("rb_vs_opposing_def", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lineups = opt.optimize_lineups(pool.drop(columns=["actual", "matched"], errors="ignore"),
                                       salary_cap=CAP, roster_slots=SKILL_SLOTS,
                                       rb_vs_opposing_def=False, **settings)
    actual = dict(zip(pool.Nickname, pool.actual))
    proj = dict(zip(pool.Nickname, pool.fanduel_fantasy_points))
    rows = []
    for lu in lineups:
        names = list(lu["Nickname"].astype(str))
        rows.append((sum(proj[n] for n in names), sum(actual[n] for n in names)))
    rows.sort(key=lambda r: -r[0])               # the builder's own ranking
    return key, aggression, rows


def summarize(results):
    recs = []
    for (season, week), aggression, rows in results:
        if len(rows) < N_LINEUPS:
            continue
        a = np.array([r[1] for r in rows])
        p = np.array([r[0] for r in rows])
        recs.append(dict(season=season, week=week, aggression=aggression,
                         top1=a[0], top5=a[:HEAD].mean(), mid=a[HEAD:TAIL_FROM - 1].mean(),
                         deep=a[TAIL_FROM - 1:].mean(), all25=a.mean(),
                         proj_top5=p[:HEAD].mean(), proj_tail=p[TAIL_FROM - 1:].mean(),
                         rho=pd.Series(p).corr(pd.Series(a), method="spearman")))
    df = pd.DataFrame(recs)
    df["gap"] = df.top5 - df.deep
    return df


def report(df):
    def block(d, label):
        g = d.gap
        se = g.std(ddof=1) / np.sqrt(len(g))
        print(f"  {label:<22} weeks {len(g):>3}  gap {g.mean():+6.2f}  95% CI "
              f"[{g.mean() - 1.96 * se:+.2f}, {g.mean() + 1.96 * se:+.2f}]  "
              f"positive {int((g > 0).sum())}/{len(g)}  "
              f"top1 {d.top1.mean():6.1f}  top5 {d.top5.mean():6.1f}  "
              f"6-10 {d.mid.mean():6.1f}  11-25 {d.deep.mean():6.1f}  "
              f"proj gap {(d.proj_top5 - d.proj_tail).mean():5.2f}  rho {d.rho.mean():+.3f}")
    for aggression in AGGRESSIONS:
        d = df[df.aggression == aggression]
        print(f"\ncash preset, aggression {aggression} "
              f"(usage cap {opt.contest_settings('cash', aggression)['max_usage_percentage']}%)")
        block(d[d.season < 2026], "backtest 2021-2025")
        for season in SEASONS:
            block(d[d.season == season], f"  {season}")
        block(d[d.season == 2026], "real slates 2026 wk6-9")


def main():
    CACHE.mkdir(exist_ok=True)
    pools26 = real_pools()
    model = salary_model(pools26)
    print("salary model (2026 real slates):")
    for pos, m in model.items():
        print(f"  {pos}: salary = {m['a']:.0f} + {m['b']:.0f} x FPPG, r {m['r']:.3f}, "
              f"n {m['n']}, clipped [{m['lo']:.0f}, {m['hi']:.0f}]")
    acc = nflverse_points()
    real = real_pool_frames(pools26, acc)
    for (s, w), p in real.items():
        print(f"  2026 wk{w}: {len(p)} projected skill players on the slate, "
              f"{int(p.matched.sum())} with a scored row")
    pools = {**backtest_pools(model), **real}
    jobs = [(k, p, a) for k, p in pools.items() for a in AGGRESSIONS]
    with ProcessPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(build_and_score, jobs, chunksize=4))
    pd.to_pickle(results, CACHE / "lineup_depth_raw.pkl")
    df = summarize(results)
    df.to_parquet(CACHE / "lineup_depth_results.parquet")
    report(df)


if __name__ == "__main__":
    main()
