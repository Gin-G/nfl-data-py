"""The promotion fix, scored the way the board is actually scored.

``EXPERIMENTS.md`` ships ``roles.promotion_multiplier`` with an explicit debt:
"Not yet validated end to end. This is measured against actual usage, not
against the board's own FanDuel-point error through the trained model." This
closes that, and there are two separate gaps in it, not one.

**The currency.** ``starter_out.py`` scored a multiplier against a trailing
usage average. The board is graded in FanDuel points, and the number the
multiplier actually lands on has been through the network (which shrinks hard
toward the positional mean), the recent-form blend, and the static depth-rank
discount. The mid-season mover entry measured exactly what that does to a
multiplicative term: the same gain decayed monotonically as the base carried
more of the network's shrinkage, from +7.3% on a raw average to ~3% at the
level ``blend.py`` measures. A term fitted against trailing usage has not been
tested until it is tested on the base it ships against.

**The trigger.** This one is bigger and was not written down. The factors were
fitted with an *oracle* absence signal — snap counts, which state who played
rather than inferring it. ``predict.py`` fires on the injury report plus roster
status, before the game, and a correct multiplier applied to the wrong player is
a loss, not a wash. So the promotion is run off three absence signals: the one
production has (the Friday injury report plus roster status), the one the fit had
(snap counts, after the fact), and a strictly weaker one that is genuinely
mid-week information (the practice report alone), since production runs Tuesday
and Wednesday.

Harness
-------
One pass over 2025 (weeks 1-18) and 2026 to date, reproducing production's
weekly points path exactly:

    net      = the network's fanduel_fantasy_points for the player's latest
               game strictly before this week
    blended  = blend.blend_value(net, trailing-5 form, position)
    discount = x roles.per_game_role_multiplier(position, chart rank)
    promoted = x roles.promotion_multiplier(position, rank, out_ranks)

Every arm is that same chain with one link changed, so the arms are paired on
identical model output and the seed lottery cancels in the difference.

The model call is batched per week rather than per player — ``_predict_roster_row``
costs 855ms a row, which is hours for this grid. ``verify()`` runs production's
own function on a sample and asserts the batched path reproduces it EXACTLY,
because a harness that merely resembles production measures nothing. It is not
decoration: it caught a real divergence in ``Series.round`` (see ``add_arms``),
and a tolerance of one rounding step would have hidden it.

Leakage, in the three places this harness could have it:

  * The dataset handed to the ``Projector`` is truncated to games strictly
    before the week. Without that, ``_predict_roster_row`` takes each player's
    *latest* row, which in a backtest is week 18, and the trailing form with it.
  * Rosters are truncated the same way. ``dedupe_rosters`` keeps the most
    recent week's entry, which on a full-season frame is the final roster —
    the wrong team and a settled injury status.
  * Depth charts are passed whole and left to ``current_depth_chart``, which
    already picks the latest snapshot on or before the week's first gameday.

The injury report is nflverse's weekly one, published before the games and so
legitimately available — but it is the FINAL report, and production runs
Tuesday/Wednesday off a partial one. nflverse ships one row per player-week with
no timestamp, so the mid-week view cannot be reconstructed; ``practice_out_ranks``
bounds the cost of that instead.

Scope: 2025 and 2026 only, and not by choice. ``DepthChartAnalyzer`` raises
``KeyError: 'pos_abb'`` on a 2024 depth chart, so the shipped role code cannot
run on the older weekly format at all.

    python experiments/promotion_endtoend.py            # primary
    PROMO_WINDOW=current python experiments/promotion_endtoend.py
"""
import os
import pathlib
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE = pathlib.Path(__file__).parent / "_cache"
SEASONS = (2025, 2026)
POSITIONS = ("QB", "RB", "WR", "TE")
EPOCHS = 60
N_SEEDS = int(os.environ.get("PROMO_SEEDS", "5"))
WINDOW = os.environ.get("PROMO_WINDOW", "history")   # history | current
# First week the current-season-only window can be trained at all. Production
# switches to it as soon as the season has one completed week
# (PROJECTION_MIN_CURRENT_WEEKS), but `clean_training_data` keeps only players
# with three or more games, so a window of one or two weeks is an empty frame.
# Four completed weeks is the first point the fit is worth comparing.
MIN_CURRENT_WEEK = 5

# ``roles.PROMOTION_FACTOR`` as it stood when this was measured. Named here, not
# read from the package, so the arms keep their meaning after the constant
# changes: otherwise re-running this would silently redefine "shipped" and the
# numbers in the ledger entry would stop being reproducible. ``run`` prints
# whether the package still agrees with it.
MEASURED_FACTOR = {"RB": 1.42, "QB": 1.29, "WR": 1.0, "TE": 1.0}
PROPOSED_FACTOR = {"RB": 1.0, "QB": 1.29, "WR": 1.0, "TE": 1.0}
BOOT = 2000
SEED = 20261015


# -- data -----------------------------------------------------------------

def _load(name, loader):
    path = CACHE / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from nfl_projections.utils import to_pandas

    frame = to_pandas(loader())
    CACHE.mkdir(exist_ok=True)
    frame.to_parquet(path)
    return frame


def rosters(season):
    import nflreadpy as nfl
    return _load(f"ros_{season}", lambda: nfl.load_rosters_weekly(seasons=[season]))


def charts(season):
    import nflreadpy as nfl
    return _load(f"dc_{season}", lambda: nfl.load_depth_charts(seasons=[season]))


def injuries(season):
    import nflreadpy as nfl
    return _load(f"inj_{season}", lambda: nfl.load_injuries(seasons=[season]))


def snaps(season):
    import nflreadpy as nfl
    return _load(f"snp_{season}", lambda: nfl.load_snap_counts(seasons=[season]))


def schedules(season):
    import nflreadpy as nfl
    return _load(f"sch_{season}", lambda: nfl.load_schedules(seasons=[season]))


def patch_injury_loader(frame):
    """Serve ``fetch_nflverse_injuries`` from a cached frame.

    The Projector fetches the injury report in its constructor, once per week,
    and this harness builds 23 of them. Same rows, no 23 network round trips.
    """
    import nfl_projections.injuries as inj_mod

    def _fetch(season, week, _f=frame):
        return inj_mod.parse_nflverse_injuries(_f, week)

    inj_mod.fetch_nflverse_injuries = _fetch


# -- the oracle absence signal -------------------------------------------

def played_sets(season):
    """(team, normalized name) pairs with an offensive snap, by week.

    This is the signal ``starter_out.py`` fitted against: snap counts state who
    played. Keyed by name because snap counts carry a pro-football-reference id
    and the depth chart carries a gsis one, and the depth analyzer identifies
    players by name anyway.
    """
    from nfl_projections.utils import normalize_player_name

    sn = snaps(season)
    sn = sn[(sn.game_type == "REG") & sn.position.isin(POSITIONS)].copy()
    sn["key"] = sn.player.map(normalize_player_name)
    by_week, teams_by_week = {}, {}
    for week, g in sn.groupby("week"):
        on = g[g.offense_snaps > 0]
        by_week[int(week)] = set(zip(on.team, on.key))
        teams_by_week[int(week)] = set(g.team.unique())
    return by_week, teams_by_week


def practice_out_ranks(projector, inj):
    """``_build_out_ranks`` off the practice report alone — the Wednesday view.

    Production runs Tuesday and Wednesday, and the ``report_status`` this harness
    reads ("Out", "Doubtful") is the Friday game designation. nflverse publishes
    one consolidated row per player-week with no timestamp, so the mid-week view
    cannot be reconstructed exactly. The nearest strictly-weaker signal that IS
    mid-week information is "did not practise": it ignores ``report_status``
    entirely and keeps only ``practice_status`` of DNP, plus roster status, which
    is known all week.

    This is an upper bound on how much the gain can degrade from running early —
    the real Wednesday report sits between this and the Friday one.
    """
    from nfl_projections import config
    from nfl_projections.predict import _extract_team, projectable_status
    from nfl_projections.utils import normalize_player_name

    w = inj[inj.week == projector.week]
    dnp = set(zip(w.team,
                  w.full_name.map(normalize_player_name).where(
                      w.practice_status.astype(str).str.contains(
                          "Did Not Participate", case=False, na=False))))
    dnp.discard((None, None))
    out = {}
    for _, player in projector.rosters.iterrows():
        name = player.get("player_name")
        position = player.get("position")
        if not name or position not in config.POSITIONS:
            continue
        team = _extract_team(player)
        absent = (team, normalize_player_name(name)) in dnp
        if not absent and not projectable_status(player.get("status"),
                                                 player.get("week"), projector.week):
            absent = not projector.injury_analyzer.should_boost_backup(name, team)
        if not absent:
            continue
        role = projector.depth_analyzer.get_player_role(name, team)
        if role and role.get("depth_rank") is not None:
            out.setdefault((team, position), []).append(role["depth_rank"])
    return {k: tuple(v) for k, v in out.items()}


def oracle_out_ranks(projector, played, teams_playing):
    """``_build_out_ranks``, with "did not play" in place of the injury report.

    Mirrors the production builder line for line except for the absence test, so
    the two triggers differ in nothing else. Teams on a bye are skipped: they
    have no snap rows, which would read as the whole roster being out.
    """
    from nfl_projections import config
    from nfl_projections.predict import _extract_team
    from nfl_projections.utils import normalize_player_name

    out = {}
    for _, player in projector.rosters.iterrows():
        name = player.get("player_name")
        position = player.get("position")
        if not name or position not in config.POSITIONS:
            continue
        team = _extract_team(player)
        if team not in teams_playing:
            continue
        if (team, normalize_player_name(name)) in played:
            continue
        role = projector.depth_analyzer.get_player_role(name, team)
        if role and role.get("depth_rank") is not None:
            out.setdefault((team, position), []).append(role["depth_rank"])
    return {k: tuple(v) for k, v in out.items()}


# -- the weekly pass -------------------------------------------------------

def truncate(ds, season, week):
    """Games strictly before (season, week), as production's dataset would be."""
    wk = pd.to_numeric(ds.week, errors="coerce")
    return ds[(ds.season < season) | ((ds.season == season) & (wk < week))]


def candidates(projector):
    """Roster rows production would price this week, and the ones it zeroes.

    Replicates ``predict_position``'s two gates in order: zero out anyone the
    injury analyzer calls absent, then drop anyone off the active roster who is
    not an elevated fill-in.
    """
    from nfl_projections import config
    from nfl_projections.predict import _extract_team, projectable_status

    live, zeroed, seen = [], [], set()
    for _, player in projector.rosters.iterrows():
        name = player.get("player_name")
        position = player.get("position")
        if not name or position not in config.POSITIONS or name in seen:
            continue
        team = _extract_team(player)
        pid = player.get("player_id", player.get("gsis_id", "")) or ""
        if pd.isna(pid):
            pid = ""
        if projector.injury_analyzer.should_zero_out_player(name, team, pid):
            seen.add(name)
            zeroed.append((name, position, team, pid))
            continue
        if (not projectable_status(player.get("status"), player.get("week"),
                                   projector.week)
                and not projector.injury_analyzer.should_boost_backup(name, team)):
            continue
        seen.add(name)
        live.append((name, position, team, pid, player))
    return live, zeroed


def latest_rows(projector, season):
    """Each player's latest history row and recent-game count, batched.

    ``_predict_roster_row`` does this per player with a mask over the whole
    history frame. Same answer: filter to the season-or-prior window, sort, take
    the tail — ``_player_history`` matches on player_id first and falls back to
    the normalised name, so both keys are indexed.
    """
    hist = projector.history
    recent = hist[pd.to_numeric(hist.season, errors="coerce") >= season - 1]
    recent = recent.sort_values(["season", "week"])
    counts_id = recent.groupby("player_id").size()
    counts_nm = recent.groupby("_norm_name").size()
    last_id = recent.groupby("player_id").tail(1).set_index("player_id")
    last_nm = recent.groupby("_norm_name").tail(1).set_index("_norm_name")
    # _player_history returns the id match only when it is non-empty, over the
    # FULL history rather than the recent window, so a player with history but
    # none of it recent must not fall through to the name branch.
    has_id = set(hist.player_id.dropna().unique())
    return counts_id, counts_nm, last_id, last_nm, has_id


def week_frame(ds, trained, season, week, ros, dc, sch, played, teams_playing,
               inj=None):
    """One row per scored player-week, with every multiplier recorded.

    Returns the production chain's pieces rather than its answer, so the arms can
    be composed afterwards without a second model pass.
    """
    from nfl_projections import blend as blend_mod
    from nfl_projections import model as model_mod
    from nfl_projections import predict, roles
    from nfl_projections.utils import normalize_player_name

    ros_w = ros[pd.to_numeric(ros.week, errors="coerce") <= week]
    proj = predict.Projector(truncate(ds, season, week), trained, season, week,
                             use_injuries=True, rosters=ros_w, depth_charts=dc,
                             schedule=sch)
    live, zeroed = candidates(proj)
    out_prod = proj._build_out_ranks()
    out_oracle = oracle_out_ranks(proj, played, teams_playing)
    out_prac = ({} if inj is None
                else practice_out_ranks(proj, inj))
    counts_id, counts_nm, last_id, last_nm, has_id = latest_rows(proj, season)
    preseason = proj._is_preseason()
    weights = blend_mod.weights_for_mode(preseason)

    rows, stat_idx = [], []
    for name, position, team, pid, _player in live:
        if pid and pid in has_id:
            if pid not in last_id.index:
                continue                      # history, but none of it recent
            stat_row, n_games = last_id.loc[pid], int(counts_id[pid])
        else:
            key = normalize_player_name(name)
            if key not in last_nm.index:
                continue
            stat_row, n_games = last_nm.loc[key], int(counts_nm[key])
        if n_games < 2:
            continue                          # routed to the rookie prior
        role = proj.depth_analyzer.get_player_role(name, team)
        rank = role["depth_rank"] if role else None
        rows.append({
            "season": season, "week": week, "player_id": pid, "player_name": name,
            "position": position, "team": team,
            "rank_raw": rank, "rank": roles.norm_rank(rank),
            "on_chart": role is not None,
            "n_form_games": n_games,
            "out_prod": out_prod.get((team, position), ()),
            "out_oracle": out_oracle.get((team, position), ()),
            "out_prac": out_prac.get((team, position), ()),
        })
        stat_idx.append(stat_row)

    zf = pd.DataFrame(zeroed, columns=["player_name", "position", "team", "player_id"])
    zf["season"], zf["week"] = season, week
    if not rows:
        return pd.DataFrame(), zf

    stat_rows = pd.DataFrame(stat_idx).reset_index(drop=True)
    out = pd.DataFrame(rows)
    out["form"] = blend_mod.recent_form_from_rows(stat_rows, preseason=preseason).values
    input_df = model_mod.build_input_rows(trained, stat_rows,
                                          positions=out.position.values,
                                          teams=out.team.values)
    out["net"] = model_mod.predict_batch(trained, input_df)["fanduel_fantasy_points"].values

    out["blended"] = [
        blend_mod.blend_value(n, f, p, weights=weights, n_games=g)
        for n, f, p, g in zip(out.net, out.form, out.position, out.n_form_games)
    ]
    out["role_mult"] = [roles.per_game_role_multiplier(p, r)
                        for p, r in zip(out.position, out.rank_raw)]
    out["role_mult"] = out.role_mult.where(out.role_mult < 1.0, 1.0)
    out["promo"] = [roles.promotion_multiplier(p, r, o)
                    for p, r, o in zip(out.position, out.rank_raw, out.out_prod)]
    out["promo_oracle"] = [roles.promotion_multiplier(p, r, o)
                           for p, r, o in zip(out.position, out.rank_raw, out.out_oracle)]
    out["promo_prac"] = [roles.promotion_multiplier(p, r, o)
                         for p, r, o in zip(out.position, out.rank_raw, out.out_prac)]
    out["promo_below"] = [_promo_below(p, r, o)
                          for p, r, o in zip(out.position, out.rank_raw, out.out_prod)]
    # The parameter-free arms: the rank change alone, no fitted factor. Computed
    # rather than divided back out, so the un-promoted stay at exactly 1.0 —
    # dividing a 1.0 by 1.42 would shrink every RB on the board and turn the
    # comparison into a different question.
    for col, ranks in (("promo_lift", out.out_prod),
                       ("promo_oracle_lift", out.out_oracle),
                       ("promo_prac_lift", out.out_prac)):
        out[col] = [_lift(p, r, o)
                    for p, r, o in zip(out.position, out.rank_raw, ranks)]
    out["preseason"] = preseason
    return out, zf


def _lift(position, rank, out_ranks):
    """The parameter-free half of ``promotion_multiplier``: the rank change alone.

    Charge the player the rate of the rank he is actually filling, and nothing
    beyond it. Written out rather than derived from the package function so it
    does not move when ``PROMOTION_FACTOR`` does.
    """
    from nfl_projections import roles

    r = roles.norm_rank(rank)
    promoted = roles.effective_rank(rank, out_ranks)
    if promoted >= r:
        return 1.0
    static = roles.per_game_role_multiplier(position, r)
    if static <= 0:
        return 1.0
    return roles.per_game_role_multiplier(position, promoted) / static


def _promo_below(position, rank, out_ranks):
    """The placebo multiplier: promote off absences strictly BELOW the player.

    Same injury report, same positions, same arithmetic, through a direction
    ``effective_rank`` ignores by design — an RB2 being out does not promote the
    RB1. A gain that survives here is not the re-ranking.

    Mirrors ``_lift``, not the shipped multiplier, because ``lift`` is the arm
    this entry argues for and a placebo should mirror the thing it is testing.
    It is also, like ``_lift``, free of ``PROMOTION_FACTOR``: reading that
    constant here made the placebo move when the constant did, which the
    reproducibility check caught.
    """
    from nfl_projections import roles

    r = roles.norm_rank(rank)
    below = sum(1 for o in out_ranks if roles.norm_rank(o) > r)
    if not below:
        return 1.0
    promoted = max(1, r - below)
    if promoted >= r:
        return 1.0        # same guard the real term has: nowhere to promote to
    static = roles.per_game_role_multiplier(position, r)
    if static <= 0:
        return 1.0
    return roles.per_game_role_multiplier(position, promoted) / static


# -- arms ------------------------------------------------------------------

ARMS = {
    "net_only": lambda d: d.blended,
    "discount": lambda d: d.blended * d.role_mult,
    "lift": lambda d: d.blended * d.role_mult * d.promo_lift,
    # What the package does right now, whatever PROMOTION_FACTOR says. This is
    # the only arm `verify` can check, because it is the only one production has.
    "shipped": lambda d: d.blended * d.role_mult * d.promo,
    # The two factor settings under comparison, pinned so they stay fixed.
    "as_measured": lambda d: d.blended * d.role_mult * _with(d, "promo_lift",
                                                             MEASURED_FACTOR),
    "proposed": lambda d: d.blended * d.role_mult * _with(d, "promo_lift",
                                                          PROPOSED_FACTOR),
    "oracle_lift": lambda d: d.blended * d.role_mult * d.promo_oracle_lift,
    "oracle_as_measured": lambda d: d.blended * d.role_mult * _with(
        d, "promo_oracle_lift", MEASURED_FACTOR),
    "placebo": lambda d: d.blended * d.role_mult * d.promo_below,
    "practice_lift": lambda d: d.blended * d.role_mult * d.promo_prac_lift,
}


def _with(d, lift_col, factors):
    """A lift column times a per-position factor, applied only where it fired."""
    f = d.position.map(factors).fillna(1.0)
    return d[lift_col] * np.where(d[lift_col] != 1.0, f, 1.0)


def add_arms(d):
    """Compose the arms and round them the way ``_predict_roster_row`` does.

    With Python's ``round``, not ``Series.round``. They disagree on an exact
    decimal tie — ``round(13.95, 1)`` is 13.9 and ``pd.Series([13.95]).round(1)``
    is 14.0, because numpy scales by ten and rints while Python rounds the
    decimal correctly. Ties are vanishingly rare in random data (0 in 2,000,000
    uniform draws) and routine here, because every chain is a 2-decimal network
    output times a short multiplier, so it lands on a clean half constantly.
    Caught by ``verify`` on exactly one player, which is what it is for.
    """
    for name, fn in ARMS.items():
        d[name] = [round(float(v), 1) for v in fn(d)]
    return d


def actuals(ds, seasons):
    """Realised FanDuel points, regular season only, one row per player-week."""
    from nfl_projections.utils import regular_games

    a = ds[ds.week != "AVG"].copy()
    a["week"] = pd.to_numeric(a.week, errors="coerce")
    a = regular_games(a)
    a = a[a.season.isin(seasons) & a.position.isin(POSITIONS)]
    return (a.groupby(["season", "week", "player_id"]).fanduel_fantasy_points
            .sum().rename("actual").reset_index())


# -- reporting -------------------------------------------------------------

MAIN = ["net_only", "discount", "lift", "as_measured"]


def mae_table(d, arms, label="", base_arm="discount"):
    rows = []
    for a in arms:
        err = d[a] - d.actual
        rows.append({"arm": a, "n": len(d), "mae": err.abs().mean(),
                     "bias": err.mean(), "mean_proj": d[a].mean(),
                     "mean_actual": d.actual.mean()})
    t = pd.DataFrame(rows)
    if base_arm in arms:
        base = t.loc[t.arm == base_arm, "mae"].iloc[0]
        t[f"gain_vs_{base_arm}"] = (base - t.mae) / base
    if label:
        print(f"\n{label}")
    print(t.round(4).to_string(index=False))
    return t


def bootstrap(d, a, b, n=BOOT, seed=SEED):
    """Cluster bootstrap over (season, week) of the MAE gain of ``a`` over ``b``.

    Clustered because a week shares an injury report, a slate and a model, so
    the player-weeks inside one are not independent draws — the same reason the
    mid-season mover entry had to resample seasons rather than rows.
    """
    rng = np.random.default_rng(seed)
    keys = sorted(set(zip(d.season, d.week)))
    groups = {k: g for k, g in d.groupby(["season", "week"])}
    pts = []
    for _ in range(n):
        s = pd.concat([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        ma, mb = (s[a] - s.actual).abs().mean(), (s[b] - s.actual).abs().mean()
        pts.append((mb - ma) / mb)
    q = np.percentile(pts, [5, 50, 95])
    base_a = (d[a] - d.actual).abs().mean()
    base_b = (d[b] - d.actual).abs().mean()
    return {"point": (base_b - base_a) / base_b, "p5": q[0], "p50": q[1], "p95": q[2]}


def trigger_report(d):
    fires_p = d.promo != 1.0
    fires_o = d.promo_oracle != 1.0
    both = int((fires_p & fires_o).sum())
    print("\nTHE TWO TRIGGERS, on %d scored player-weeks" % len(d))
    print("  production (injury report + roster status) fires on %5d (%.1f%%)"
          % (fires_p.sum(), 100 * fires_p.mean()))
    print("  oracle     (played no offensive snaps)      fires on %5d (%.1f%%)"
          % (fires_o.sum(), 100 * fires_o.mean()))
    print("  both %d; production only %d; oracle only %d"
          % (both, (fires_p & ~fires_o).sum(), (~fires_p & fires_o).sum()))
    if fires_p.sum():
        print("  production precision %.3f  recall %.3f"
              % (both / fires_p.sum(), both / max(int(fires_o.sum()), 1)))
    agree = d[fires_p & fires_o]
    if len(agree):
        print("  where both fire, mean multiplier: production %.3f oracle %.3f"
              % (agree.promo.mean(), agree.promo_oracle.mean()))
    print("  mean actual: promoted by production %.2f, by oracle %.2f, neither %.2f"
          % (d[fires_p].actual.mean(), d[fires_o].actual.mean(),
             d[~fires_p & ~fires_o].actual.mean()))


def by_position(d, arms, label, base_arm="discount", test_arm="as_measured"):
    rows = []
    for pos in POSITIONS:
        s = d[d.position == pos]
        if len(s) < 10:
            continue
        row = {"position": pos, "n": len(s), "actual": s.actual.mean()}
        for a in arms:
            row[a] = (s[a] - s.actual).abs().mean()
        row["gain"] = (row[base_arm] - row[test_arm]) / row[base_arm]
        rows.append(row)
    t = pd.DataFrame(rows)
    print(f"\n{label}")
    print(t.round(4).to_string(index=False))
    return t


def per_week(d, a, b):
    rows = []
    for (season, week), g in d.groupby(["season", "week"]):
        ma, mb = (g[a] - g.actual).abs().mean(), (g[b] - g.actual).abs().mean()
        rows.append({"season": season, "week": int(week), "n": len(g),
                     b: mb, a: ma, "gain": (mb - ma) / mb})
    return pd.DataFrame(rows)


# -- verification ----------------------------------------------------------

def verify(ds, trained, season, week, ros, dc, sch, n=60):
    """Assert the batched path reproduces ``_predict_roster_row`` exactly.

    The whole harness rests on this. ``_predict_roster_row`` IS production; this
    module is a faster re-expression of it, and a re-expression that quietly
    diverged would produce a very confident number about nothing. The sample is
    weighted to the players the term acts on rather than 60 arbitrary ones.
    """
    from nfl_projections import predict

    ros_w = ros[pd.to_numeric(ros.week, errors="coerce") <= week]
    proj = predict.Projector(truncate(ds, season, week), trained, season, week,
                             use_injuries=True, rosters=ros_w, depth_charts=dc,
                             schedule=sch)
    played, teams = played_sets(season)
    fast, _ = week_frame(ds, trained, season, week, ros, dc, sch,
                         played.get(week, set()), teams.get(week, set()))
    fast = add_arms(fast)
    roster_by_name = {r["player_name"]: r for _, r in proj.rosters.iterrows()}

    promoted = fast[fast.promo != 1.0]
    sample = pd.concat([promoted.head(n // 2),
                        fast.head(n // 2)]).drop_duplicates("player_name")
    checked, bad, worst = 0, [], 0.0
    for _, row in sample.iterrows():
        r = roster_by_name.get(row.player_name)
        if r is None:
            continue
        truth = proj._predict_roster_row(r)
        if not truth or "fanduel_fantasy_points" not in truth:
            continue
        checked += 1
        gap = abs(float(truth["fanduel_fantasy_points"]) - float(row["shipped"]))
        worst = max(worst, gap)
        if gap > 0.051:
            bad.append((row.player_name, float(truth["fanduel_fantasy_points"]),
                        float(row["shipped"]), str(truth.get("role_adjustment"))))
    print("\nVERIFY %s wk%d: %d players against _predict_roster_row, %d over 0.05, "
          "worst %.3f (%d promoted in the week)"
          % (season, week, checked, len(bad), worst, len(promoted)))
    for b in bad[:10]:
        print("   GAP %-24s production %.2f  harness %.2f  (%s)" % b)
    # A single rounding step is 0.05 and cannot be ruled out of a chain that
    # rounds once; anything larger, or more than a couple of players, is the
    # arithmetic diverging and makes every number below meaningless.
    if worst > 0.11 or len(bad) > max(1, checked // 50):
        raise SystemExit("harness diverges from production; nothing below means anything")


# -- driver ----------------------------------------------------------------

def train_for(ds, season, week=None):
    """Train (or reload) the model the week's projections come off.

    Cached on disk because the arms are paired on identical model output, so a
    rerun has to reuse the SAME network or the comparison changes underneath the
    numbers in the ledger.
    """
    from nfl_projections import model as model_mod

    tag = f"{season}_{WINDOW}{'' if week is None else '_w%d' % week}_{N_SEEDS}s{EPOCHS}e"
    cached = CACHE / f"model_{tag}"
    if (cached / "metadata.json").exists():
        print(f"Reusing cached model {cached}")
        return model_mod.TrainedModel.load(str(cached))

    if WINDOW == "current":
        wk = pd.to_numeric(ds.week, errors="coerce")
        tr = ds[(ds.season == season) & (wk < week)]
        tag = f"{season} weeks < {week}"
    else:
        tr = ds[ds.season < season]
        tag = f"seasons < {season}"
    print(f"Training on {tag} ({N_SEEDS} seed(s), {EPOCHS} epochs, {len(tr):,} rows)...")
    trained, _ = model_mod.train_ensemble(tr, n_seeds=N_SEEDS, epochs=EPOCHS,
                                          plot_path=None)
    trained.save(str(cached))
    return trained


def collect(ds, act):
    frames, zeros = [], []
    for season in SEASONS:
        ros, dc, sch = rosters(season), charts(season), schedules(season)
        inj = injuries(season)
        patch_injury_loader(inj)
        played, teams = played_sets(season)
        weeks = [int(w) for w in sorted(act[act.season == season].week.unique())]
        trained = None
        if WINDOW != "current":
            trained = train_for(ds, season)
            verify(ds, trained, season, weeks[len(weeks) // 2], ros, dc, sch)
        verified = WINDOW != "current"
        for week in weeks:
            if WINDOW == "current":
                if week < MIN_CURRENT_WEEK:
                    continue      # production falls back to the history window
                trained = train_for(ds, season, week)
                if not verified:
                    verify(ds, trained, season, week, ros, dc, sch)
                    verified = True
            f, z = week_frame(ds, trained, season, week, ros, dc, sch,
                              played.get(week, set()), teams.get(week, set()),
                              inj=inj)
            if len(f):
                frames.append(f)
            if len(z):
                zeros.append(z)
            print("  %d wk%2d: %3d priced, %3d zeroed, %2d promoted"
                  % (season, week, len(f), len(z),
                     int((f.promo != 1.0).sum()) if len(f) else 0))
    return pd.concat(frames, ignore_index=True), pd.concat(zeros, ignore_index=True)


def run():
    from nfl_projections import dataset, roles

    ds = dataset.load_dataset()
    act = actuals(ds, SEASONS)
    print("dataset %s, %d scorable player-weeks in %s" % (ds.shape, len(act), SEASONS))
    print("window=%s seeds=%d" % (WINDOW, N_SEEDS))
    print("roles.PROMOTION_FACTOR now %s" % roles.PROMOTION_FACTOR)
    print("measured against %s; proposal %s" % (MEASURED_FACTOR, PROPOSED_FACTOR))
    live_arm = ("as_measured" if roles.PROMOTION_FACTOR == MEASURED_FACTOR
                else "proposed" if roles.PROMOTION_FACTOR == PROPOSED_FACTOR
                else "neither")
    print("-> the shipped arm currently equals: %s" % live_arm)

    raw, z = collect(ds, act)
    d = add_arms(raw)
    d = d[d.player_id != ""].merge(act, on=["season", "week", "player_id"], how="inner")
    print("\n%d scored player-weeks over %d weeks"
          % (len(d), d.groupby(["season", "week"]).ngroups))

    zp = z[z.player_id != ""].merge(act, on=["season", "week", "player_id"], how="inner")
    print("%d player-weeks production zeroed that played anyway, mean actual %.2f"
          % (len(zp), zp.actual.mean() if len(zp) else float("nan")))

    trigger_report(d)
    mae_table(d, MAIN + ["placebo"], "FANDUEL MAE, EVERY SCORED PLAYER-WEEK")

    prom = d[d.promo != 1.0]
    mae_table(prom, MAIN, "FANDUEL MAE, THE PLAYERS THE TERM ACTS ON "
                          "(production trigger, n=%d)" % len(prom))
    by_position(prom, MAIN, "...by position")

    print("\nBY ENTERING CHART RANK (production trigger)")
    rows = []
    for (pos, rank), g in prom.groupby(["position", "rank"]):
        if len(g) < 10:
            continue
        rows.append({"position": pos, "rank": rank, "n": len(g),
                     "actual": g.actual.mean(), "promo": g.promo.mean(),
                     "discount": (g.discount - g.actual).abs().mean(),
                     "lift": (g.lift - g.actual).abs().mean(),
                     "as_measured": (g.as_measured - g.actual).abs().mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\nCLUSTER BOOTSTRAP over weeks, MAE gain (positive = first arm better)")
    rows = []
    for arm, base, pop, note in (
            ("as_measured", "discount", d, "all scored"),
            ("as_measured", "discount", prom, "promoted only"),
            ("lift", "discount", prom, "promoted only, parameter-free"),
            ("as_measured", "lift", prom, "promoted, is the fitted factor worth it")):
        rows.append({"arm": arm, "vs": base, "population": note, "n": len(pop),
                     **bootstrap(pop, arm, base)})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\nTHE ORACLE CEILING — the trigger the factors were fitted with")
    oro = d[d.promo_oracle != 1.0]
    mae_table(oro, ["discount", "oracle_lift", "oracle_as_measured"],
              "on the oracle-promoted set (n=%d)" % len(oro))
    for arm in ("oracle_as_measured", "oracle_lift"):
        b = bootstrap(oro, arm, "discount")
        print("  %-15s vs discount: %+.4f [%+.4f, %+.4f]"
              % (arm, b["point"], b["p5"], b["p95"]))
    both = d[(d.promo != 1.0) & (d.promo_oracle != 1.0)]
    mae_table(both, ["discount", "as_measured", "oracle_as_measured"],
              "where the two triggers agree (n=%d)" % len(both))
    only_p = d[(d.promo != 1.0) & (d.promo_oracle == 1.0)]
    if len(only_p) > 10:
        mae_table(only_p, ["discount", "as_measured"],
                  "where production fires and the man above actually PLAYED "
                  "(n=%d)" % len(only_p))

    print("\nTHE WEDNESDAY VIEW — the practice report alone, no Friday designation")
    fp = d.promo_prac != 1.0
    print("  fires on %d player-weeks vs %d for the Friday report; "
          "precision %.3f recall %.3f against the oracle"
          % (fp.sum(), (d.promo != 1.0).sum(),
             (fp & (d.promo_oracle != 1.0)).sum() / max(int(fp.sum()), 1),
             (fp & (d.promo_oracle != 1.0)).sum() / max(int((d.promo_oracle != 1.0).sum()), 1)))
    pr = d[fp]
    if len(pr) > 10:
        mae_table(pr, ["discount", "practice_lift"],
                  "on the practice-promoted set (n=%d)" % len(pr))
        b = bootstrap(pr, "practice_lift", "discount")
        print("  practice_lift vs discount: %+.4f [%+.4f, %+.4f]"
              % (b["point"], b["p5"], b["p95"]))
    # Whole-board effect of degrading the trigger, which is what shipping early costs.
    mae_table(d, ["discount", "practice_lift", "lift"],
              "whole board, Wednesday trigger vs Friday trigger")

    print("\nTHE CHANGE THIS ARGUES FOR — drop PROMOTION_FACTOR at RB, keep it at QB")
    for name, pop in (("every scored player-week", d),
                      ("the promoted set", d[d.promo != 1.0]),
                      ("promoted RBs", d[(d.promo != 1.0) & (d.position == "RB")])):
        mae_table(pop, ["discount", "as_measured", "proposed"],
                  "%s (n=%d)" % (name, len(pop)))
        b = bootstrap(pop, "proposed", "as_measured")
        pwx = per_week(pop, "proposed", "as_measured")
        print("  proposed vs as_measured: %+.4f [%+.4f, %+.4f]; %d of %d weeks favour it"
              % (b["point"], b["p5"], b["p95"], int((pwx.gain > 0).sum()), len(pwx)))

    print("\nORDERING, which MAE structurally cannot see (per position per week,")
    print("over every scored player at the position — the arms differ only on the")
    print("promoted, but ordering is a cross-player property)")
    topn = {"QB": 6, "RB": 12, "WR": 12, "TE": 6}
    rows = []
    for pos in POSITIONS:
        s_pos = d[d.position == pos]
        k = topn[pos]
        for a in ("net_only", "discount", "lift", "as_measured", "proposed"):
            rho, hit, pts = [], [], []
            for _, g in s_pos.groupby(["season", "week"]):
                if len(g) < k + 4:
                    continue
                top = set(g.nlargest(k, "actual").player_id)
                rho.append(g[a].rank().corr(g.actual.rank()))
                pk = g.nlargest(k, a)
                hit.append(len(set(pk.player_id) & top) / k)
                pts.append(pk.actual.mean())
            rows.append({"position": pos, "arm": a, "top_n": k,
                         "spearman": np.mean(rho), "topn_hit": np.mean(hit),
                         "topn_actual_pts": np.mean(pts)})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\nPER POSITION on the promoted set, cluster bootstrap")
    rows = []
    for pos in POSITIONS:
        s_pos = prom[prom.position == pos]
        if len(s_pos) < 20:
            continue
        for a, b_arm in (("lift", "discount"), ("as_measured", "discount"),
                         ("as_measured", "lift")):
            rows.append({"position": pos, "n": len(s_pos), "arm": a, "vs": b_arm,
                         **bootstrap(s_pos, a, b_arm)})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\nPLACEBO — the same absences, counted BELOW the player instead of above")
    pl = d[d.promo_below != 1.0]
    if len(pl) > 10:
        mae_table(pl, ["discount", "lift", "placebo"],
                  "on the placebo-promoted set (n=%d)" % len(pl))
        b = bootstrap(pl, "placebo", "discount")
        print("  placebo vs discount: %+.4f [%+.4f, %+.4f]"
              % (b["point"], b["p5"], b["p95"]))

    pw = per_week(prom, "as_measured", "discount")
    print("\nPER WEEK on the promoted set (%d weeks, %d with a positive gain)"
          % (len(pw), int((pw.gain > 0).sum())))
    print(pw.round(4).to_string(index=False))

    out = CACHE / ("promo_e2e_%s.parquet" % WINDOW)
    d.drop(columns=["out_prod", "out_oracle", "out_prac"]).to_parquet(out)
    print("\nwrote %s" % out)
    return d


if __name__ == "__main__":
    run()
