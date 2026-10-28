"""The starter comes back, and the man who covered for him keeps his rate.

``roles.promotion_multiplier`` handles one direction of a role change: when
the player ahead of you is out, you are charged the rate of the rank you are
filling. Nothing handles the other direction. A fill-in's projection is built
from his own recent games — the network's input row carries his trailing-3
and trailing-5 usage, and the form blend is his trailing-5 points — so in the
weeks after the starter returns, that form still describes the job he no
longer has. The static depth-rank discount then applies to the inflated base.

2026 week 8 is the case that found it: Josh Jacobs back from suspension, and
Pierre Strong, RB3, still projected for 24 rushing yards and 0.22 touchdowns
off two weeks as the lead back. He took 8 snaps and no carries.

The term, parameter-free and the exact mirror of the shipped promotion
---------------------------------------------------------------------
For a player at chart rank r, let S be his teammates at the same position
ranked above him who are playing THIS week. Over his form window (his last
five games, the window ``blend.recent_form_from_rows`` and the rolling
features read), each game g in which some of S did not take an offensive snap
was played at the promoted rank ``effective_rank(r, absent_g)``, i.e. at
``_lift(position, r, absent_g)`` times his static rate. So the window's
average carries an inflation of mean(L_g) over the healthy rate, and

    demote = n / sum(L_g)          (L_g = 1 for games with all of S present)

Only games he played for his current team count — a mid-season mover's old
team is a different question, answered in "The mid-season mover".

Absence in the window is read from snap counts, which is legitimate here:
those games are in the past at prediction time. Who is "playing this week" is
production's own trigger — the injury report plus roster status — so the arm
fires only where production could fire it.

Arms, all paired on identical model output (see ``promotion_endtoend``):

    shipped   production as it stands
    demote    shipped x demote
    placebo   the same arithmetic on absences BELOW the player in his window,
              the direction a demotion has no business caring about

Run:  PROMO_WINDOW=current .venv/bin/python experiments/starter_returns.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("PROMO_WINDOW", "current")
import promotion_endtoend as pe  # noqa: E402

WINDOW_GAMES = 5


def window_games(ds):
    """``{player_id: [(season, week, team), ...]}`` sorted, for lookups."""
    g = ds[ds.week != "AVG"].copy()
    g["week"] = pd.to_numeric(g.week, errors="coerce")
    g = g.dropna(subset=["week"]).sort_values(["season", "week"])
    return {pid: list(zip(x.season.astype(int), x.week.astype(int), x.team))
            for pid, x in g.groupby("player_id")}


def factors(d, games, played_by_season):
    """``demote`` and ``placebo`` multipliers for every scored row."""
    from nfl_projections.utils import normalize_player_name

    d = d.copy()
    d["key"] = d.player_name.map(normalize_player_name)
    demote = np.ones(len(d))
    placebo = np.ones(len(d))
    starter = np.ones(len(d))
    n_inflated = np.zeros(len(d), dtype=int)
    rooms = {k: g for k, g in d.groupby(["season", "week", "team", "position"])}
    for i, r in enumerate(d.itertuples(index=False)):
        room = rooms[(r.season, r.week, r.team, r.position)]
        out_now = {pe.roles_norm(o) for o in r.out_prod} if r.out_prod else set()
        above = [(m.key, m.rank) for m in room.itertuples(index=False)
                 if m.rank < r.rank and m.key != r.key and m.rank not in out_now]
        below = [(m.key, m.rank) for m in room.itertuples(index=False)
                 if m.rank > r.rank and m.key != r.key]
        if not above and not below:
            continue
        hist = [x for x in games.get(r.player_id, [])
                if (x[0], x[1]) < (r.season, r.week) and x[0] >= r.season - 1]
        hist = hist[-WINDOW_GAMES:]
        if not hist:
            continue
        lifts_a, lifts_b, lifts_s, hit = [], [], [], 0
        for s, w, team in hist:
            if team != r.team:
                lifts_a.append(1.0); lifts_b.append(1.0); lifts_s.append(1.0)
                continue
            on = played_by_season.get(s, {}).get(w)
            if on is None:
                lifts_a.append(1.0); lifts_b.append(1.0); lifts_s.append(1.0)
                continue
            gone_a = tuple(rk for k, rk in above if (team, k) not in on)
            gone_b = tuple(rk for k, rk in below if (team, k) not in on)
            la = pe._lift(r.position, r.rank_raw, gone_a) if gone_a else 1.0
            lb = pe._promo_below(r.position, r.rank_raw, gone_b) if gone_b else 1.0
            gone_s = tuple(rk for rk in gone_a if rk == 1)
            ls = pe._lift(r.position, r.rank_raw, gone_s) if gone_s else 1.0
            hit += la != 1.0
            lifts_a.append(la); lifts_b.append(lb); lifts_s.append(ls)
        demote[i] = len(hist) / sum(lifts_a)
        placebo[i] = len(hist) / sum(lifts_b)
        starter[i] = len(hist) / sum(lifts_s)
        n_inflated[i] = hit
    d["demote"], d["placebo_dem"], d["n_inflated"] = demote, placebo, n_inflated
    d["demote_starter"] = starter
    return d


def run():
    from nfl_projections import dataset, roles

    pe.roles_norm = roles.norm_rank
    ds = dataset.load_dataset()
    act = pe.actuals(ds, pe.SEASONS)
    print("dataset %s; window=%s seeds=%d" % (ds.shape, pe.WINDOW, pe.N_SEEDS))
    raw, _ = pe.collect(ds, act)
    d = pe.add_arms(raw)
    played = {}
    for s in sorted({*pe.SEASONS, min(pe.SEASONS) - 1}):
        played[s] = pe.played_sets(s)[0]

    # A player who took offensive snaps and recorded nothing has no stats row,
    # so an inner join drops him — and he is exactly the fill-in this term is
    # about (Pierre Strong, 2026 week 7: 8 snaps, no touch). Count him at zero.
    from nfl_projections.utils import normalize_player_name
    d = d[d.player_id != ""].merge(act, on=["season", "week", "player_id"], how="left")
    on_field = np.array([(t, normalize_player_name(n)) in played.get(s, {}).get(w, set())
                         for s, w, t, n in zip(d.season, d.week, d.team, d.player_name)])
    zero = d.actual.isna() & on_field
    d.loc[zero, "actual"] = 0.0
    d = d[d.actual.notna()]
    print("\n%d scored player-weeks over %d weeks (%d took snaps with no stat line, scored 0)"
          % (len(d), d.groupby(["season", "week"]).ngroups, int(zero.sum())))
    d = factors(d, window_games(ds), played)
    d["demote_arm"] = [round(float(v), 1) for v in d.shipped * d.demote]
    d["placebo_arm"] = [round(float(v), 1) for v in d.shipped * d.placebo_dem]
    d["starter_arm"] = [round(float(v), 1) for v in d.shipped * d.demote_starter]

    fired = d[d.demote != 1.0]
    print("\nTHE TERM FIRES on %d player-weeks (%.1f%%); window games inflated: mean %.2f"
          % (len(fired), 100 * len(fired) / len(d), fired.n_inflated.mean()))
    print("  mean demote %.3f; mean actual %.2f; shipped mean %.2f"
          % (fired.demote.mean(), fired.actual.mean(), fired.shipped.mean()))

    arms = ["shipped", "demote_arm"]
    pe.mae_table(d, arms, "WHOLE BOARD", base_arm="shipped")
    pe.mae_table(fired, arms, "WHERE IT FIRES (n=%d)" % len(fired), base_arm="shipped")
    pe.by_position(fired, arms, "...by position", base_arm="shipped", test_arm="demote_arm")

    print("\nCLUSTER BOOTSTRAP over weeks (positive = demote better)")
    for name, pop in (("whole board", d), ("where it fires", fired)):
        b = pe.bootstrap(pop, "demote_arm", "shipped")
        print("  %-16s %+.4f [%+.4f, %+.4f]" % (name, b["point"], b["p5"], b["p95"]))
    for pos in pe.POSITIONS:
        pop = fired[fired.position == pos]
        if len(pop) >= 20:
            b = pe.bootstrap(pop, "demote_arm", "shipped")
            print("  %-16s %+.4f [%+.4f, %+.4f]  n=%d"
                  % (pos, b["point"], b["p5"], b["p95"], len(pop)))

    st = d[d.demote_starter != 1.0]
    print("\nSTARTER-ONLY VARIANT: fires on %d player-weeks (%.1f%%)"
          % (len(st), 100 * len(st) / len(d)))
    pe.mae_table(st, ["shipped", "starter_arm"], "where it fires", base_arm="shipped")
    for name, pop in [("all", st)] + [(p, st[st.position == p]) for p in pe.POSITIONS]:
        if len(pop) >= 20:
            b = pe.bootstrap(pop, "starter_arm", "shipped")
            print("  %-6s %+.4f [%+.4f, %+.4f]  n=%d"
                  % (name, b["point"], b["p5"], b["p95"], len(pop)))
    b = pe.bootstrap(d, "starter_arm", "shipped")
    print("  whole board %+.4f [%+.4f, %+.4f]" % (b["point"], b["p5"], b["p95"]))

    pw = pe.per_week(fired, "demote_arm", "shipped")
    print("\nweeks favouring it: %d of %d" % (int((pw.gain > 0).sum()), len(pw)))

    pl = d[d.placebo_dem != 1.0]
    if len(pl) > 10:
        pe.mae_table(pl, ["shipped", "placebo_arm"],
                     "PLACEBO, absences below him (n=%d)" % len(pl), base_arm="shipped")
        b = pe.bootstrap(pl, "placebo_arm", "shipped")
        print("  placebo vs shipped: %+.4f [%+.4f, %+.4f]" % (b["point"], b["p5"], b["p95"]))

    print("\nORDERING at RB and WR (the arms differ only where it fires)")
    rows = []
    for pos, k in (("RB", 12), ("WR", 12), ("TE", 6), ("QB", 6)):
        s_pos = d[d.position == pos]
        for a in arms:
            rho, hit = [], []
            for _, g in s_pos.groupby(["season", "week"]):
                if len(g) < k + 4:
                    continue
                top = set(g.nlargest(k, "actual").player_id)
                rho.append(g[a].rank().corr(g.actual.rank()))
                hit.append(len(set(g.nlargest(k, a).player_id) & top) / k)
            rows.append({"position": pos, "arm": a, "spearman": np.mean(rho),
                         "topn_hit": np.mean(hit)})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\nGREEN BAY 2026, weeks 7-8 scored here (Jacobs back):")
    gb = d[(d.season == 2026) & (d.team == "GB") & (d.position == "RB")]
    print(gb[["week", "player_name", "rank", "n_inflated", "demote", "demote_starter",
              "shipped", "demote_arm", "starter_arm", "actual"]].round(3).to_string(index=False))

    out = pe.CACHE / "starter_returns_current.parquet"
    d.drop(columns=["out_prod", "out_oracle", "out_prac"]).to_parquet(out)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    run()
