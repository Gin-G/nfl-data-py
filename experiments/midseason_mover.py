"""The player who changed teams in October, and what the weekly path thinks of him.

The season-scale finding (EXPERIMENTS.md, "Vacated share") was that the size of
a role travels with the player: an arriving veteran's share on his old team
carries real information his new depth rank does not, and `shares.py` blends it
in at MOVER_BLEND. That model runs once, in the spring, over players who changed
teams in free agency.

The weekly path has the same case and no term for it. `predict.py` reads a
mover's LAST GAME ROW, whose rolling usage and scoring form are entirely his old
team's, hands the network the new team as a categorical, and then applies
`per_game_role_multiplier` for his new depth rank — one-sided, so only ever
downward. A player traded into a bigger role cannot be revised up; a player
traded into a smaller one keeps a starter's trailing volume and loses only the
rank discount.

Two effects are tangled here and they pull opposite ways:

  settling in      a new playbook in three days. Even into the same size of
                   role, the first week or two on a new team should be smaller
                   than the trailing number suggests.
  role change      the reason most players are traded is that the role is
                   different on the other side, in either direction.

Only the second knows the sign, and only the depth chart knows the second. So
this measures both: a ramp by weeks-since-arrival, and a blend weight on the
old-team trailing usage against the new team's rank baseline.

Identification is from WEEKLY ROSTERS, not stats rows. A traded player who does
not touch the ball still appears on the new roster, and conditioning on having
a stats row would keep only the moves that worked out.
"""
import pathlib

import numpy as np
import pandas as pd

from nfl_projections.utils import normalize_player_name

CACHE = pathlib.Path(__file__).parent / "_cache"
SEASONS = tuple(range(2013, 2026))   # 2013 is the first season of snap counts
POSITIONS = ("RB", "WR", "TE", "QB")
TRAIL_WINDOW = 5            # appearances averaged for "what he was doing"
PLAYED_SNAPS = 1
ARRIVAL_WEEKS = 4           # how far past the move to follow him
TEST_SEASONS = (2023, 2024, 2025)


def _load(name, loader):
    path = CACHE / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    out = pd.concat([loader(s) for s in SEASONS], ignore_index=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out


def rosters():
    def one(season):
        import nflreadpy as nfl
        d = nfl.load_rosters_weekly(seasons=[season]).to_pandas()
        keep = ["season", "week", "team", "position", "full_name", "gsis_id",
                "status", "game_type"]
        return d[[c for c in keep if c in d.columns]]
    d = _load("rosters_2013_2025", one)
    d = d[(d.get("game_type", "REG") == "REG") & d.position.isin(POSITIONS)
          & d.gsis_id.notna()].copy()
    # a claimed player can show on two rosters in one week; the later row is the
    # one he finished the week on
    d = (d.sort_values(["season", "gsis_id", "week"])
          .drop_duplicates(["season", "gsis_id", "week"], keep="last"))
    return d.rename(columns={"gsis_id": "player_id"})


def snaps():
    def one(season):
        import nflreadpy as nfl
        d = nfl.load_snap_counts(seasons=[season]).to_pandas()
        keep = ["season", "week", "team", "player", "position", "offense_snaps",
                "offense_pct", "game_type"]
        d = d[[c for c in keep if c in d.columns]]
        return d[d.get("game_type", "REG") == "REG"]
    d = _load("snaps_2013_2025", one)
    d["key"] = [normalize_player_name(n) for n in d.player]
    return d


def stats():
    def one(season):
        import nflreadpy as nfl
        d = nfl.load_player_stats(seasons=[season]).to_pandas()
        d = d[(d.season_type == "REG") & d.position.isin(POSITIONS)]
        keep = ["season", "week", "team", "player_display_name", "player_id",
                "position", "carries", "targets", "receptions", "rushing_yards",
                "receiving_yards", "rushing_tds", "receiving_tds",
                "passing_yards", "passing_tds", "passing_interceptions", "attempts"]
        return d[[c for c in keep if c in d.columns]]
    d = _load("stats_2013_2025", one)
    d["key"] = [normalize_player_name(n) for n in d.player_display_name]
    return d


def schedules():
    def one(season):
        import nflreadpy as nfl
        d = nfl.load_schedules(seasons=[season]).to_pandas()
        return d[["season", "week", "game_type", "gameday"]]
    return _load("sched_2013_2025", one)


def week_starts():
    """First gameday of each (season, week) — for dating the 2025 chart snapshots."""
    s = schedules()
    s = s[s.game_type == "REG"].copy()
    s["gameday"] = pd.to_datetime(s["gameday"], errors="coerce", utc=True)
    return s.groupby(["season", "week"])["gameday"].min().reset_index()


def charts():
    """One depth-chart rank per (season, week, team, player), 2013-2025.

    This is the instrument that matters, and it has to be independent of the
    player's own usage. A first pass ranked an arrival among his new teammates
    BY TRAILING USAGE, which is circular: a busy player is mechanically slotted
    high, so 'moved up the chart' and 'was productive' became the same variable
    and every rank-based predictor double-counted the trailing number.

    nflverse publishes two formats and they need different handling:
      2013-2024  one chart per week; ``depth_team`` is the rank
      2025+      dated league-wide ESPN snapshots; keep the latest one taken
                 before the week's first kickoff, which is what predict.py's
                 current_depth_chart does in production
    """
    path = CACHE / "charts_2013_2025.parquet"
    if path.exists():
        return pd.read_parquet(path)
    import nflreadpy as nfl
    starts = week_starts()
    frames = []
    for season in SEASONS:
        d = nfl.load_depth_charts(seasons=[season]).to_pandas()
        if "depth_team" in d.columns:                       # weekly format
            d = d[(d.formation == "Offense") & (d.game_type == "REG")
                  & d.position.isin(POSITIONS) & d.gsis_id.notna()]
            out = pd.DataFrame({
                "season": season, "week": pd.to_numeric(d.week, errors="coerce"),
                "team": d.club_code, "player_id": d.gsis_id, "position": d.position,
                "chart_rank": pd.to_numeric(d.depth_team, errors="coerce")})
        else:                                               # dated snapshots
            d = d[d.pos_abb.isin(POSITIONS) & d.gsis_id.notna()].copy()
            d["stamp"] = pd.to_datetime(d.dt, errors="coerce", utc=True)
            rows = []
            for _, wk in starts[starts.season == season].iterrows():
                before = d[d.stamp < wk.gameday]
                if before.empty:
                    continue
                snap = before[before.stamp == before.stamp.max()]
                rows.append(pd.DataFrame({
                    "season": season, "week": int(wk.week), "team": snap.team,
                    "player_id": snap.gsis_id, "position": snap.pos_abb,
                    "chart_rank": pd.to_numeric(snap.pos_rank, errors="coerce")}))
            out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        frames.append(out.dropna(subset=["week", "chart_rank"]))
    c = pd.concat(frames, ignore_index=True)
    c["week"] = c.week.astype(int)
    # ties on the chart (two WRs listed 2nd) and duplicate slot rows: take the best
    c = (c.groupby(["season", "week", "team", "player_id", "position"])
          .chart_rank.min().reset_index())
    CACHE.mkdir(parents=True, exist_ok=True)
    c.to_parquet(path)
    return c


def usage(d):
    """One usage number per position — the volume a projection is really after."""
    carries = d.get("carries", 0).fillna(0)
    targets = d.get("targets", 0).fillna(0)
    attempts = d.get("attempts", 0).fillna(0)
    return np.where(d.position == "QB", attempts,
                    np.where(d.position == "RB", carries + targets, targets))


def moves(ros):
    """Every mid-season team change: (season, player, move_week, old, new).

    ``had_role`` records what he was doing before he moved, so the deep-bench
    churn that makes up most of this list can be separated from the handful of
    moves anyone was deciding about.
    """
    r = ros.copy()
    r["prev_team"] = r.groupby(["season", "player_id"]).team.shift(1)
    r["prev_week"] = r.groupby(["season", "player_id"]).week.shift(1)
    mv = r[r.prev_team.notna() & (r.team != r.prev_team)].copy()
    return mv[["season", "player_id", "full_name", "position", "week", "team",
               "prev_team", "prev_week"]].rename(
        columns={"week": "move_week", "team": "new_team", "prev_team": "old_team"})


def panel(ros=None, sn=None, st=None):
    """Every rostered skill player at every team-week, with what was knowable.

    ``trail`` is the trailing-window mean usage over his most recent appearances
    STRICTLY BEFORE this week, on any team — which is what the weekly path
    carries forward in its rolling features. ``entering_rank`` ranks the team's
    position group by that number, standing in for the depth chart: a mover is
    ranked on his old-team form against his new teammates' own form, which is
    about what an ESPN chart does with him in his first week.
    """
    ros = rosters() if ros is None else ros
    sn = snaps() if sn is None else sn
    st = stats() if st is None else st
    st = st.copy()
    st["usage"] = usage(st)
    st = st.sort_values(["season", "player_id", "week"])
    st["trail_after"] = (st.groupby(["season", "player_id"])["usage"]
                         .transform(lambda s: s.rolling(TRAIL_WINDOW,
                                                        min_periods=1).mean()))

    team_weeks = sn[["season", "week", "team"]].drop_duplicates()
    p = ros.merge(team_weeks, on=["season", "week", "team"], how="inner")

    appear = st[["season", "player_id", "week", "trail_after"]].sort_values("week")
    p = p.sort_values("week")
    p = pd.merge_asof(p, appear, on="week", by=["season", "player_id"],
                      allow_exact_matches=False)
    # no appearance yet this season -> nothing to carry forward
    p["trail"] = p["trail_after"].fillna(0.0)
    p["seen"] = p["trail_after"].notna()
    p["entering_rank"] = (p.groupby(["season", "week", "team", "position"])["trail"]
                          .rank(ascending=False, method="first"))
    p["key"] = [normalize_player_name(n) for n in p.full_name]
    ch = charts().rename(columns={"position": "chart_position"})
    p = p.merge(ch, on=["season", "week", "team", "player_id"], how="left")
    # Ordinal position among the players the chart lists for this room. The raw
    # rank is NOT comparable across nflverse's two chart formats: the 2013-2024
    # weekly charts stop at 3 and pile ties on each number (2.09 players listed
    # "WR1"), so their "WR2" averages 2.66 targets, while a 2025 ESPN snapshot
    # lists exactly one player per rank and its WR2 averages 5.42. Both formats
    # list the same NUMBER of players per room (WR 4.34 vs 4.56), so the ordinal
    # reconciles them — and it is what production reads, since the snapshot rank
    # and the ordinal coincide there.
    listed = p[p.chart_rank.notna()].copy()
    listed["slot"] = (listed.sort_values(["season", "week", "team", "position",
                                         "chart_rank", "player_id"])
                      .groupby(["season", "week", "team", "position"]).cumcount() + 1)
    p = p.merge(listed[["season", "week", "team", "player_id", "slot"]],
                on=["season", "week", "team", "player_id"], how="left")
    p["era"] = np.where(p.season >= 2025, "snapshot", "weekly")

    got = st.set_index(["season", "week", "player_id"])["usage"]
    p["usage"] = [float(got.get((s, w, pid), 0.0))
                  for s, w, pid in zip(p.season, p.week, p.player_id)]
    played = sn[sn.offense_snaps.fillna(0) >= PLAYED_SNAPS]
    avail = set(zip(played.season, played.week, played.team, played.key))
    p["played"] = [(s, w, t, k) in avail
                   for s, w, t, k in zip(p.season, p.week, p.team, p.key)]

    mv = moves(ros)
    first = (mv.sort_values("move_week")
               .drop_duplicates(["season", "player_id", "new_team"], keep="first")
               .set_index(["season", "player_id", "new_team"]))
    keys = list(zip(p.season, p.player_id, p.team))
    p["move_week"] = [float(first.move_week.get(k, np.nan)) for k in keys]
    p["old_team"] = [first.old_team.get(k, None) for k in keys]
    p["last_old_week"] = [float(first.prev_week.get(k, np.nan)) for k in keys]
    p["since_move"] = p["week"] - p["move_week"] + 1
    p["is_mover"] = p["since_move"].notna()

    # The rank he held on the team he left, in his last week there. His trailing
    # usage was earned at THAT rank, which is what makes the new rank's discount
    # the wrong thing to apply to it.
    ranks = (p.drop_duplicates(["season", "player_id", "team", "week"])
              .set_index(["season", "player_id", "team", "week"])["chart_rank"])
    p["old_rank"] = [
        float(ranks.get((s, pid, ot, w), np.nan)) if ot is not None and w == w else np.nan
        for s, pid, ot, w in zip(p.season, p.player_id, p.old_team, p.last_old_week)]

    # The rank production would actually read: the chart where it lists him, and
    # roles._UNKNOWN_RANK where it does not. Requiring a chart row instead threw
    # away almost every FIRST week with a new team, which is the week the whole
    # question is about.
    p["rank"] = p["chart_rank"].fillna(float(roles_unknown_rank()))

    # Trailing usage frozen at his last appearance BEFORE the move — purely the
    # old team. The as-of number above refills with new-team games within two or
    # three appearances, so measuring the arrival effect with it dilutes the
    # effect into the thing being measured.
    st = st.sort_values(["player_id", "season", "week"])
    st["trail_career"] = (st.groupby("player_id")["usage"]
                          .transform(lambda x: x.rolling(TRAIL_WINDOW,
                                                         min_periods=1).mean()))
    last_prior = (st.groupby(["player_id", "season"])["trail_career"].last()
                  .reset_index().sort_values(["player_id", "season"]))
    last_prior["prev_season"] = last_prior.season + 1
    carry = last_prior.set_index(["player_id", "prev_season"])["trail_career"]
    # what the path carries for a player with no game yet this season: the rolling
    # window standing at the end of his last season, which features.py builds
    # across the season boundary
    p["trail_last_season"] = [float(carry.get((pid, s), np.nan))
                              for pid, s in zip(p.player_id, p.season)]

    # his first week on this roster, for spotting a mid-season signing
    firstw = (p.groupby(["season", "player_id", "team"])["week"].min()
              .rename("first_roster_week").reset_index())
    p = p.merge(firstw, on=["season", "player_id", "team"], how="left")

    pre = p.loc[p.is_mover, ["season", "player_id", "move_week"]].drop_duplicates()
    ap = st[["season", "player_id", "week", "trail_after"]].rename(
        columns={"week": "_w", "trail_after": "trail_pre"})
    ap["_w"] = ap["_w"].astype(float)
    ap = ap.sort_values("_w")
    pre = pre.sort_values("move_week")
    pre = pd.merge_asof(pre, ap, left_on="move_week", right_on="_w",
                        by=["season", "player_id"], allow_exact_matches=False)
    pre_lut = pre.set_index(["season", "player_id", "move_week"])["trail_pre"]
    p["trail_pre"] = [float(pre_lut.get((s, pid, mw), np.nan))
                      for s, pid, mw in zip(p.season, p.player_id, p.move_week)]
    return p


def roles_unknown_rank():
    from nfl_projections import roles
    return roles._UNKNOWN_RANK


SLOT_CAP = 5


def rank_prior(p, exclude_season=None, era=None):
    """Mean usage by (position, chart slot) among players who were on the field —
    the baseline for 'the nth man this room lists'.

    Kept per ERA, because the two chart formats are different instruments (see
    panel). ``exclude_season`` holds the test season out of its own baseline.
    """
    d = p[p.played & p.slot.notna()]
    if era is not None:
        d = d[d.era == era]
    if exclude_season is not None:
        d = d[d.season != exclude_season]
    d = d.copy()
    d["slot_b"] = d.slot.clip(upper=SLOT_CAP)
    return (d.groupby(["position", "slot_b"])["usage"].mean()
             .rename("prior").reset_index())


def baseline_table(p, era="snapshot"):
    """The shipped table: usage by position and slot, from the chart format
    production reads, forced monotone because the deep slots are noisy."""
    pri = rank_prior(p, era=era)
    out = {}
    for position, g in pri.groupby("position"):
        g = g.sort_values("slot_b")
        vals, floor = {}, float("inf")
        for r in g.itertuples():
            floor = min(floor, round(float(r.prior), 2))
            vals[int(r.slot_b)] = floor
        out[position] = vals
    return out


def _priced(p, d):
    """Attach everything the weekly path knows about a player-week.

    ``discount`` comes from the RAW chart rank, because that is the number
    production hands roles.per_game_role_multiplier. ``prior`` comes from the
    ordinal slot, fit within the row's own era and excluding its own season.
    """
    from nfl_projections import roles
    luts = {}
    for era in ("weekly", "snapshot"):
        for season in sorted(p.season.unique()):
            pri = rank_prior(p, exclude_season=season, era=era)
            luts[(era, season)] = {(r.position, int(r.slot_b)): r.prior
                                   for r in pri.itertuples()}
    d = d.copy()
    d["prior"] = [luts[(era, season)].get((pos, min(int(sl), SLOT_CAP)), 0.0)
                  for era, season, pos, sl
                  in zip(d.era, d.season, d.position, d.slot)]
    d["discount"] = [roles.per_game_role_multiplier(pos, int(r))
                     for pos, r in zip(d.position, d["rank"])]
    return d


def arrivals(p, max_weeks=ARRIVAL_WEEKS):
    """A mover's first weeks on the new team: he played, he had in-season form
    for the weekly path to carry forward, and the new chart has him.

    All three conditions match production. A player with no appearance yet this
    season takes a different code path (his last game row is last season's), one
    who did not play is zeroed by the injury step, and one the chart does not
    list is handled at ``roles._UNKNOWN_RANK`` — measured separately below,
    because that is a guess rather than a reading.
    """
    a = p[p.is_mover & (p.since_move >= 1) & (p.since_move <= max_weeks)
          & p.played & p.trail_pre.notna() & p.slot.notna()].copy()
    # trail      = the as-of window production actually carries, which refills
    #              with new-team games as he plays
    # trail_pre  = frozen at his last appearance before the move, purely the old
    #              team — the stale number in its undiluted form
    a["trail_asof"] = a["trail"]
    return _priced(p, a)


def settled(p, min_weeks=ARRIVAL_WEEKS + 1):
    """The control: everyone else's player-weeks, same conditions.

    This is the question that decides what the finding is. If the depth-rank
    discount only hurts movers, it is a mover problem and wants a mover's fix.
    If it hurts here too, the discount itself is miscalibrated against a
    trailing average and a carve-out would be hiding that.
    """
    c = p[(~p.is_mover | (p.since_move > min_weeks)) & p.played & p.seen
          & p.slot.notna()]
    return _priced(p, c)


def ramp(a):
    """Actual usage over the trailing number the path carries, and the error of
    each, by weeks since the move."""
    rows = []
    for k in range(1, ARRIVAL_WEEKS + 1):
        w = a[a.since_move == k]
        if len(w) < 20:
            continue
        rows.append({"since_move": k, "n": len(w), "trail": w.trail.mean(),
                     "actual": w.usage.mean(),
                     "ratio": w.usage.mean() / max(w.trail.mean(), 1e-9),
                     "mae_trail": (w.trail - w.usage).abs().mean(),
                     "mae_production": (w.trail * w.discount - w.usage).abs().mean()})
    return pd.DataFrame(rows)


def by_rank_move(a):
    """What the move did to his role on paper, and what he actually did."""
    d = a[a.old_rank.notna() & a.chart_rank.notna()].copy()
    d["rank_move"] = np.where(d.chart_rank < d.old_rank, "up",
                              np.where(d.chart_rank > d.old_rank, "down", "level"))
    return (d.groupby("rank_move")
             .agg(n=("usage", "size"), old_rank=("old_rank", "mean"),
                  new_rank=("chart_rank", "mean"), trail=("trail", "mean"),
                  rank_baseline=("prior", "mean"), actual=("usage", "mean"),
                  production=("discount", lambda x: np.nan))
             .reset_index())


def compare(d, test_seasons=TEST_SEASONS, weight=None, trail_col="trail",
            cap=None):
    """MAE and bias of each candidate on the test seasons."""
    t = d[d.season.isin(test_seasons)]
    tr, act, pr = t[trail_col].values, t.usage.values, t.prior.values
    preds = {"trail": tr, "production": tr * t.discount.values, "rank_baseline": pr}
    if weight is not None:
        preds["blend"] = _blend(tr, pr, weight, cap)
    rows = []
    for name, pred in preds.items():
        rows.append({"predictor": name, "n": len(t),
                     "mae": float(np.abs(pred - act).mean()),
                     "bias": float((pred - act).mean())})
    return pd.DataFrame(rows)


def _mult(trail, prior, weight, cap=None):
    """The multiplier itself — what production can actually apply."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(trail > 0.5, prior / np.maximum(trail, 1e-9), 1.0)
    mult = weight + (1 - weight) * ratio
    return np.clip(mult, 1.0 / cap, cap) if cap is not None else mult


def _blend(trail, prior, weight, cap=None):
    """weight on his own carried-forward form, the rest on the rank baseline.

    Written as a MULTIPLIER on the trailing number rather than as a sum, because
    that is the only shape the weekly path can apply: it has a points projection
    and a stat line, not a usage baseline. ``cap`` bounds the multiplier, which
    matters because the ratio explodes for a player whose trailing usage is
    near zero — a deep backup listed second on his new chart.
    """
    return trail * _mult(trail, prior, weight, cap)


def fit_blend(d, test_seasons=TEST_SEASONS, trail_col="trail", cap=None):
    """Weight on the carried-forward form against the new rank's baseline."""
    train = d[~d.season.isin(test_seasons)]
    tr, act, pr = train[trail_col].values, train.usage.values, train.prior.values
    grid = np.round(np.arange(0.0, 1.01, 0.05), 2)
    errs = [np.abs(_blend(tr, pr, w, cap) - act).mean() for w in grid]
    return float(grid[int(np.argmin(errs))])


def fit_cap(d, weight, test_seasons=TEST_SEASONS, trail_col="trail"):
    train = d[~d.season.isin(test_seasons)]
    tr, act, pr = train[trail_col].values, train.usage.values, train.prior.values
    grid = [1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 1e9]
    errs = [np.abs(_blend(tr, pr, weight, c) - act).mean() for c in grid]
    return float(grid[int(np.argmin(errs))])


def by_week_gain(a, cap=2.5):
    """Where the fix is worth anything, using the trailing window production
    really carries. The effect must decay: once he has played a few games for
    the new team his own form IS new-team form, and there is nothing stale left."""
    rows = []
    for k in range(1, ARRIVAL_WEEKS + 1):
        g = a[a.since_move == k]
        test = g[g.season.isin(TEST_SEASONS)]
        if len(test) < 20:
            continue
        w = fit_blend(a[a.since_move <= k], trail_col="trail", cap=cap)
        tr, act, pr = test.trail.values, test.usage.values, test.prior.values
        rows.append({"since_move": k, "n_test": len(test), "weight": w,
                     "mae_production": float(np.abs(tr * test.discount.values
                                                    - act).mean()),
                     "mae_blend": float(np.abs(_blend(tr, pr, w, cap) - act).mean())})
    out = pd.DataFrame(rows)
    out["gain"] = 1 - out.mae_blend / out.mae_production
    return out


def loso(a, weights=(0.35, 0.45, 0.55, 0.65), caps=(2.0, 3.0, 4.0),
         max_weeks=ARRIVAL_WEEKS):
    """Leave-one-season-out: fit the weight and cap on 12 seasons, score the
    13th, over every arrival week in the file.

    The single train/test split left 21 first-week rows to decide on, which is
    not enough to ship a multiplier. This uses every one of them as a test row
    exactly once.
    """
    rows = []
    for k in range(1, max_weeks + 1):
        g = a[a.since_move <= k]
        err_prod, err_blend, n = [], [], 0
        picks = []
        for season in sorted(g.season.unique()):
            tr_set, te_set = g[g.season != season], g[g.season == season]
            if te_set.empty or len(tr_set) < 50:
                continue
            best, bw, bc = None, None, None
            for w in weights:
                for cap in caps:
                    e = np.abs(_blend(tr_set.trail.values, tr_set.prior.values,
                                      w, cap) - tr_set.usage.values).mean()
                    if best is None or e < best:
                        best, bw, bc = e, w, cap
            tr, act, pr = te_set.trail.values, te_set.usage.values, te_set.prior.values
            err_prod.append(np.abs(tr * te_set.discount.values - act).sum())
            err_blend.append(np.abs(_blend(tr, pr, bw, bc) - act).sum())
            n += len(te_set)
            picks.append((bw, bc))
        if not n:
            continue
        mp, mb = sum(err_prod) / n, sum(err_blend) / n
        rows.append({"weeks": f"1-{k}" if k > 1 else "1", "n": n,
                     "mae_production": mp, "mae_blend": mb,
                     "gain": 1 - mb / mp,
                     "weight": float(np.median([w for w, _ in picks])),
                     "cap": float(np.median([c for _, c in picks]))})
    return pd.DataFrame(rows)


# What blend.py measures the production base's shrinkage to be: predicted spread
# runs 0.33 of actual at QB and ~0.70 at RB/WR/TE. The headline numbers below use
# 0.70, which flatters QB — its own result is noise at n=18 either way.
PRODUCTION_LAMBDA = 0.70


def shrunk_base(d, lam, pos_means=None):
    """A stand-in for what the weekly path actually hands the discount.

    Everything above compares predictors built on a raw trailing average, and the
    depth-rank discount's defence is that it is not applied to one: it is applied
    to the network's output blended 50/50 with form, and the network shrinks hard
    toward the positional mean (blend.py measures predicted spread at 0.33-0.70
    of actual). Shrinkage lifts a low-usage player, which is exactly what a
    discount would then be correcting — so "the discount hurts a trailing
    average" does not by itself say the discount hurts production.

    This rebuilds the base with that shrinkage in it: the network side is
    ``lam * trail + (1 - lam) * positional mean``, and the base is the 50/50
    blend of that with the trailing number. ``lam = 1`` is the raw average.
    """
    if pos_means is None:
        pos_means = d.groupby("position").usage.mean().to_dict()
    pm = np.array([pos_means.get(p, 0.0) for p in d.position])
    net = lam * d.trail.values + (1 - lam) * pm
    return 0.5 * net + 0.5 * d.trail.values


def shrinkage_check(a, c, lams=(1.0, 0.8, 0.6, 0.4),
                    weights=(0.35, 0.45, 0.55, 0.65, 0.75, 0.85), caps=(2.0, 3.0, 4.0)):
    """Does the fix survive once the base carries the network's shrinkage?

    The weight is REFITTED per shrinkage level, leave-one-season-out. A fixed
    weight fitted against a raw trailing average and then applied to a shrunk
    base is not a fair test of the idea: compression toward the positional mean
    already moves a mover part of the way, so the correction left to make is
    smaller and the weight on his own form should rise to meet it. Scoring the
    unrefitted weight makes the term look like it reverses when what reversed
    was the fit.
    """
    pos_means = c.groupby("position").usage.mean().to_dict()
    rows = []
    for label, d in (("arrivals", a), ("control", c)):
        for lam in lams:
            d = d.copy()
            d["_base"] = shrunk_base(d, lam, pos_means)
            e_base = e_disc = e_blend = 0.0
            n = 0
            picks = []
            for season in sorted(d.season.unique()):
                tr_set, te = d[d.season != season], d[d.season == season]
                if te.empty or len(tr_set) < 50:
                    continue
                best = None
                for w in weights:
                    for cap in caps:
                        e = np.abs(tr_set._base.values
                                   * _mult(tr_set.trail.values, tr_set.prior.values,
                                           w, cap) - tr_set.usage.values).mean()
                        if best is None or e < best[0]:
                            best = (e, w, cap)
                _, w, cap = best
                picks.append(w)
                act, base = te.usage.values, te._base.values
                e_base += np.abs(base - act).sum()
                e_disc += np.abs(base * te.discount.values - act).sum()
                e_blend += np.abs(base * _mult(te.trail.values, te.prior.values,
                                               w, cap) - act).sum()
                n += len(te)
            rows.append({
                "group": label, "lam": lam, "n": n,
                "base": e_base / n, "base_x_discount": e_disc / n,
                "blend": e_blend / n,
                "gain_vs_production": 1 - (e_blend / max(e_disc, 1e-9)),
                "weight": float(np.median(picks))})
    return pd.DataFrame(rows)


def off_chart(p, max_weeks=ARRIVAL_WEEKS):
    """Movers the new chart does not list, who get roles._UNKNOWN_RANK = 3."""
    from nfl_projections import roles
    a = p[p.is_mover & (p.since_move >= 1) & (p.since_move <= max_weeks)
          & p.played & p.trail_pre.notna() & p.chart_rank.isna()].copy()
    a["discount"] = [roles.per_game_role_multiplier(pos, None) for pos in a.position]
    return a


def signings(p, max_weeks=ARRIVAL_WEEKS):
    """The other kind of arrival: he joined mid-season with no game yet THIS
    season, so the last game row the path reads is from a prior season.

    Brandin-Cooks-shaped — a free agent or a practice-squad promotion rather
    than a trade. The staleness is measured in months, not weeks, and the main
    measurement above excludes him by construction (it requires in-season form).
    """
    a = p[(~p.is_mover) & (p.first_roster_week > 1)
          & (p.week - p.first_roster_week < max_weeks)
          & (~p.seen) & p.played & p.trail_last_season.notna()
          & p.slot.notna()].copy()
    a["trail"] = a["trail_last_season"]
    return _priced(p, a)


def bootstrap(d, c, n=4000, lam=PRODUCTION_LAMBDA, seed=7):
    """Cluster bootstrap over seasons of the gain from dropping the discount.

    Clustered by season rather than by player-week, because the per-season sign
    flipped 7 times in 13 — the season-level variation is real, and a bootstrap
    over rows would report a confidence interval that ignores it.
    """
    rng = np.random.default_rng(seed)
    pm = c.groupby("position").usage.mean().to_dict()

    def gain(sub):
        base = shrunk_base(sub, lam, pm)
        act, disc = sub.usage.values, sub.discount.values
        return 1 - np.abs(base - act).mean() / np.abs(base * disc - act).mean()

    seasons = sorted(d.season.unique())
    parts = {s: d[d.season == s] for s in seasons}
    draws = []
    for _ in range(n):
        pick = rng.choice(seasons, size=len(seasons), replace=True)
        draws.append(gain(pd.concat([parts[s] for s in pick], ignore_index=True)))
    lo, mid, hi = np.percentile(draws, [5, 50, 95])
    return {"n": len(d), "point": gain(d), "p5": lo, "p50": mid, "p95": hi}


def bias_table(a, c, lam=PRODUCTION_LAMBDA):
    """The part that is not a wash: how hard the discount prices an arrival."""
    pm = c.groupby("position").usage.mean().to_dict()
    rows = []
    for label, d in [("all arrivals", a), ("control", c)] + \
                    [(f"  {pos}", g) for pos, g in a.groupby("position")]:
        base = shrunk_base(d, lam, pm)
        act = d.usage.values
        prod = float((base * d.discount.values - act).mean())
        rows.append({"group": label, "n": len(d), "actual": act.mean(),
                     "bias_production": prod,
                     "share_of_actual": prod / max(act.mean(), 1e-9),
                     "bias_no_discount": float((base - act).mean())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    ros = rosters()
    mv = moves(ros)
    print(f"{len(mv):,} mid-season team changes at skill positions, "
          f"{SEASONS[0]}-{SEASONS[-1]}")
    print(mv.groupby("position").size().to_string(), "\n")

    p = panel(ros)
    a, c = arrivals(p), settled(p)
    print(f"{len(p):,} rostered player-weeks; {int(p.is_mover.sum()):,} on a team "
          f"the player had just joined")
    print(f"{len(a):,} arrival weeks (first {ARRIVAL_WEEKS}) where he played, had "
          f"old-team form to carry, and the new chart listed him")
    print(f"{len(c):,} control player-weeks (not a recent arrival)\n")

    print("THE TWO CHART FORMATS ARE DIFFERENT INSTRUMENTS")
    d = p[p.played & p.chart_rank.notna()].copy()
    d["rank_b"] = d.chart_rank.clip(upper=3)
    d["slot_b"] = d.slot.clip(upper=3)
    for label, key in (("by RAW chart rank", "rank_b"), ("by ORDINAL slot", "slot_b")):
        print(f"  mean usage {label}")
        print(d.assign(era=d.era).pivot_table(index=["position", "era"], columns=key,
                                              values="usage").round(2).to_string())
    print()

    print("WHAT THE MOVE DID TO HIS ROLE ON PAPER (a rank on both sides)")
    print(by_rank_move(a).drop(columns=["production"]).to_string(index=False), "\n")

    print("BY WEEKS SINCE THE MOVE (trailing frozen before the move)")
    print(ramp(a).to_string(index=False), "\n")

    print("A PARAMETERIZED BLEND TOWARD THE NEW SLOT, leave-one-season-out")
    print(loso(a).to_string(index=False))
    for label, sub in (("weeks 2-4", a[a.since_move >= 2]),
                       ("weeks 6-12, the falsification",
                        arrivals(p, max_weeks=12).query("since_move >= 6"))):
        r = loso(sub.assign(since_move=1), max_weeks=1)
        print(f"  {label}: " + r.to_string(index=False).replace("\n", " | "))
    print()

    print("...AND WHAT HAPPENS WHEN THE BASE CARRIES THE NETWORK'S SHRINKAGE")
    print(shrinkage_check(a, c).to_string(index=False), "\n")

    print(f"DROPPING THE DISCOUNT INSTEAD — no fitted parameters, "
          f"lam={PRODUCTION_LAMBDA}")
    print(f"{'group':14s}{'n':>6}{'point':>9}{'p5':>9}{'p50':>9}{'p95':>9}")
    for label, d in [("all arrivals", a)] + [(f"  {pos}", g)
                                             for pos, g in a.groupby("position")]:
        b = bootstrap(d, c)
        print(f"{label:14s}{b['n']:6d}{b['point']:9.3f}{b['p5']:9.3f}"
              f"{b['p50']:9.3f}{b['p95']:9.3f}")
    bc = bootstrap(c.sample(20000, random_state=3), c, n=600)
    print(f"{'control':14s}{bc['n']:6d}{bc['point']:9.3f}{bc['p5']:9.3f}"
          f"{bc['p50']:9.3f}{bc['p95']:9.3f}\n")

    print("THE BIAS, which is real whether or not correcting it pays")
    print(bias_table(a, c).to_string(index=False), "\n")

    oc = off_chart(p)
    print(f"MOVERS THE NEW CHART DOES NOT LIST ({len(oc):,} weeks) — read at "
          f"rank {roles_unknown_rank()} on faith, and that is the right call:")
    print(f"  trailing {oc.trail_pre.mean():.2f}  actual {oc.usage.mean():.2f}  "
          f"production {(oc.trail_pre * oc.discount).mean():.2f}")
    print(f"  MAE no discount {(oc.trail_pre - oc.usage).abs().mean():.3f}  "
          f"production {(oc.trail_pre * oc.discount - oc.usage).abs().mean():.3f}")

    print(f"SLOT BASELINE from the format production reads: {baseline_table(p)}\n")

    sg = signings(p)
    print(f"MID-SEASON SIGNINGS — no game yet this season, so the path reads a "
          f"row from LAST season ({len(sg):,} weeks)")
    print(f"  last season's trailing {sg.trail.mean():.2f}   "
          f"new slot's baseline {sg.prior.mean():.2f}   actual {sg.usage.mean():.2f}")
    pm = c.groupby("position").usage.mean().to_dict()
    base = shrunk_base(sg, PRODUCTION_LAMBDA, pm)
    act = sg.usage.values
    print(f"  MAE production {np.abs(base * sg.discount.values - act).mean():.3f}"
          f"   no discount {np.abs(base - act).mean():.3f}"
          f"   slot baseline alone {np.abs(sg.prior.values - act).mean():.3f}")
    print(f"  bias production {np.mean(base * sg.discount.values - act):+.3f}"
          f"   no discount {np.mean(base - act):+.3f}")
    b = bootstrap(sg, c)
    print(f"  dropping the discount: {b['point']:+.3f} "
          f"[{b['p5']:+.3f}, {b['p95']:+.3f}]")
    print("  by position:")
    for pos, g in sg.groupby("position"):
        if len(g) < 15:
            continue
        gb = shrunk_base(g, PRODUCTION_LAMBDA, pm)
        ga = g.usage.values
        print(f"    {pos:3s} n={len(g):4d}  actual {ga.mean():6.2f}  "
              f"production {np.abs(gb * g.discount.values - ga).mean():6.3f}  "
              f"no discount {np.abs(gb - ga).mean():6.3f}  "
              f"slot alone {np.abs(g.prior.values - ga).mean():6.3f}")
