"""When the starter does not play, what does the backup actually do?

The season-scale finding (EXPERIMENTS.md, "Vacated share") was that a vacated
role transfers to whoever holds it next, and the share model handles that by
re-ranking. The weekly path does not re-rank. `predict.py` zeroes a player who
is OUT and then hands every remaining teammate a multiplier from his **static
depth-chart rank**, so a back listed RB2 keeps the RB2 discount (x0.80) in a
week where he is the only back on the roster. The discount is also one-sided:
`role_mult` is applied only when it is below 1.0, so nothing in the weekly path
can ever adjust a projection upward.

That is the most actionable call in weekly fantasy — the starter is out, the
backup is a play — and if this is wrong it is wrong in the direction that
matters, every week, for the players people are deciding about.

This measures the promotion empirically. Who the starter is entering a week is
set by trailing usage only. Who actually played comes from snap counts, which
state it directly rather than inferring it from an injury report.

    promotion factor = backup's usage in weeks the starter missed
                       -------------------------------------------
                       backup's usage in weeks the starter played
"""
import pathlib

import numpy as np
import pandas as pd

from nfl_projections.utils import normalize_player_name

CACHE = pathlib.Path(__file__).parent / "_cache"
SEASONS = tuple(range(2018, 2026))
POSITIONS = ("RB", "WR", "TE", "QB")
PLAYED_SNAPS = 1            # one offensive snap is enough to count as available
STARTER_SNAP_PCT = 0.45     # what it takes to have been the starter in a week
MIN_WEEKS_EACH = 2          # a backup needs weeks on both sides to compare


def _load(name, loader):
    path = CACHE / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frames = [loader(season) for season in SEASONS]
    out = pd.concat(frames, ignore_index=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out


def snaps():
    def one(season):
        import nflreadpy as nfl
        d = nfl.load_snap_counts(seasons=[season]).to_pandas()
        keep = ["season", "week", "team", "player", "position", "offense_snaps",
                "offense_pct", "game_type"]
        d = d[[c for c in keep if c in d.columns]]
        return d[d.get("game_type", "REG") == "REG"]
    d = _load("snaps_2018_2025", one)
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
    d = _load("stats_2018_2025", one)
    d["key"] = [normalize_player_name(n) for n in d.player_display_name]
    return d


def usage(d):
    """One usage number per position: what a projection is really trying to get."""
    carries = d.get("carries", 0).fillna(0)
    targets = d.get("targets", 0).fillna(0)
    attempts = d.get("attempts", 0).fillna(0)
    return np.where(d.position == "QB", attempts,
                    np.where(d.position == "RB", carries + targets, targets))


def build():
    """A candidate panel: every established player at every later team-week.

    The first version of this built candidates from the week's own stats rows,
    which is circular — a starter who did not play has no stats row, so he could
    never be seen missing. It reported starters missing 2.2% of weeks, which is
    roughly a sixth of the truth.

    A player is a candidate in week W if he has appeared for that team earlier in
    the season, which is also exactly the condition for having a trailing usage
    number. Whether he then played comes from snap counts, and a candidate with
    no snap row in a week his team played did not play.
    """
    sn = snaps()
    st = stats()
    st["usage"] = usage(st)
    st = st.sort_values(["season", "player_id", "week"])

    # Trailing usage as of AFTER each appearance — averaged over appearances, so a
    # player returning from three missed weeks is not mistaken for a backup.
    st["trail_after"] = (st.groupby(["season", "player_id"])["usage"]
                         .transform(lambda s: s.rolling(5, min_periods=1).mean()))

    team_weeks = sn[["season", "week", "team"]].drop_duplicates()
    first = (st.groupby(["season", "team", "position", "player_id", "key"])
             .week.min().rename("first_week").reset_index())
    panel = first.merge(team_weeks, on=["season", "team"])
    panel = panel[panel.week > panel.first_week].copy()

    # trailing value from the most recent appearance strictly before this week
    appear = st[["season", "player_id", "week", "trail_after"]].sort_values("week")
    panel = panel.sort_values("week")
    panel = pd.merge_asof(panel, appear, on="week", by=["season", "player_id"],
                          allow_exact_matches=False)
    panel = panel.dropna(subset=["trail_after"]).rename(columns={"trail_after": "trail"})

    panel["entering_rank"] = (panel.groupby(["season", "week", "team", "position"])["trail"]
                              .rank(ascending=False, method="first"))

    played = sn[sn.offense_snaps.fillna(0) >= PLAYED_SNAPS]
    avail = set(zip(played.season, played.week, played.team, played.key))
    panel["available"] = [
        (s, w, t, k) in avail
        for s, w, t, k in zip(panel.season, panel.week, panel.team, panel.key)]

    # what he actually did that week (no stats row = nothing)
    got = st.set_index(["season", "week", "player_id"])["usage"]
    panel["usage"] = [float(got.get((s, w, pid), 0.0))
                      for s, w, pid in zip(panel.season, panel.week, panel.player_id)]

    leaders = (panel[panel.entering_rank == 1]
               .set_index(["season", "week", "team", "position"])["available"])
    panel["starter_played"] = [
        bool(leaders.get((s, w, t, p), True))
        for s, w, t, p in zip(panel.season, panel.week, panel.team, panel.position)]
    panel["starter_known"] = [
        (s, w, t, p) in leaders.index
        for s, w, t, p in zip(panel.season, panel.week, panel.team, panel.position)]
    return panel[panel.starter_known], played


def promotion(ranked):
    """Each backup's usage with and without the starter, and the ratio."""
    backups = ranked[(ranked.entering_rank >= 2) & ranked.available]
    rows = []
    for (season, pid, position), g in backups.groupby(["season", "player_id", "position"]):
        with_out = g[~g.starter_played]
        with_in = g[g.starter_played]
        if len(with_out) < MIN_WEEKS_EACH or len(with_in) < MIN_WEEKS_EACH:
            continue
        rows.append({"season": season, "player_id": pid, "position": position,
                     "rank": float(g.entering_rank.median()),
                     "weeks_out": len(with_out), "weeks_in": len(with_in),
                     "usage_out": float(with_out.usage.mean()),
                     "usage_in": float(with_in.usage.mean())})
    out = pd.DataFrame(rows)
    out["factor"] = out.usage_out / out.usage_in.replace(0, np.nan)
    return out.dropna(subset=["factor"])


if __name__ == "__main__":
    ranked, _ = build()
    print(f"{len(ranked):,} player-weeks at entering rank 1-4, {SEASONS[0]}-{SEASONS[-1]}")
    share_out = 1 - ranked[ranked.entering_rank == 1].starter_played.mean()
    print(f"the entering rank-1 player missed {share_out:.1%} of team-position-weeks\n")

    print("USAGE BY ENTERING RANK, starter in vs out (per player-week):")
    print(f"{'pos':5s}{'rank':>5}{'starter in':>12}{'starter out':>13}{'factor':>8}{'n out':>7}")
    for position in POSITIONS:
        p = ranked[(ranked.position == position) & ranked.available]
        for rank in (2.0, 3.0, 4.0):
            g = p[p.entering_rank == rank]
            on = g[g.starter_played].usage.mean()
            off = g[~g.starter_played].usage.mean()
            n = int((~g.starter_played).sum())
            if n < 20 or not np.isfinite(on) or on <= 0:
                continue
            print(f"{position:5s}{rank:5.0f}{on:12.2f}{off:13.2f}{off / on:8.2f}{n:7d}")

    print("\nWITHIN-PLAYER promotion factor (same player, both situations):")
    pf = promotion(ranked)
    print(f"{len(pf):,} player-seasons with {MIN_WEEKS_EACH}+ weeks on each side")
    print(f"{'pos':5s}{'n':>6}{'median':>9}{'mean':>8}{'p25':>7}{'p75':>7}")
    for position in POSITIONS:
        g = pf[pf.position == position]
        if len(g) < 10:
            continue
        print(f"{position:5s}{len(g):6d}{g.factor.median():9.2f}{g.factor.mean():8.2f}"
              f"{g.factor.quantile(0.25):7.2f}{g.factor.quantile(0.75):7.2f}")

    print("\nBy entering rank, within-player (RB and WR only — the cases that matter):")
    for position in ("RB", "WR"):
        for rank in (2.0, 3.0):
            g = pf[(pf.position == position) & (pf["rank"] == rank)]
            if len(g) < 10:
                continue
            print(f"  {position} rank {rank:.0f}: n {len(g):4d}  median factor "
                  f"{g.factor.median():.2f}")

    print("\nWhat the weekly path currently applies (roles.per_game_role_multiplier):")
    from nfl_projections import roles
    for position in ("RB", "WR", "TE", "QB"):
        mults = [f"rank {r} x{roles.per_game_role_multiplier(position, r):.2f}"
                 for r in (1, 2, 3)]
        print(f"  {position:4s}{',  '.join(mults)}")


def validate(panel, test_seasons=(2024, 2025)):
    """Out-of-sample: does applying a promotion factor beat ignoring one?

    Fit the factor on the earlier seasons, apply it on the later ones, and
    predict a backup's usage in the weeks his starter was out. The baseline is
    his own trailing usage, which is what the weekly path effectively projects
    before its depth-rank discount makes it smaller still.
    """
    promoted = panel[(panel.entering_rank >= 2) & panel.available
                     & ~panel.starter_played].copy()
    train = promoted[~promoted.season.isin(test_seasons)]
    test = promoted[promoted.season.isin(test_seasons)]
    factors = {}
    for position, g in train.groupby("position"):
        ratio = (g.usage / g.trail.replace(0, np.nan)).dropna()
        factors[position] = float(ratio.median()) if len(ratio) > 30 else 1.0

    print(f"\nfactors fitted on {sorted(set(train.season))}:")
    for position in POSITIONS:
        print(f"  {position:4s}x{factors.get(position, 1.0):.2f}")

    rows = []
    for position, g in test.groupby("position"):
        f = factors.get(position, 1.0)
        base = float((g.trail - g.usage).abs().mean())
        with_f = float((g.trail * f - g.usage).abs().mean())
        static = float((g.trail * g.position.map(
            lambda p: 1.0) - g.usage).abs().mean())
        rows.append({"position": position, "n": len(g),
                     "mean actual": round(float(g.usage.mean()), 2),
                     "trailing only": round(base, 3),
                     "with promotion": round(with_f, 3),
                     "change": round(with_f - base, 3)})
    out = pd.DataFrame(rows)
    print(f"\nMAE predicting a promoted backup's usage, {test_seasons}:")
    print(out.to_string(index=False))
    return factors


if __name__ == "__main__" and True:
    print("\n\n=== OUT-OF-SAMPLE CHECK")
    panel, _ = build()
    validate(panel)
