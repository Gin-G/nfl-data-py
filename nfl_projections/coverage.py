"""Coverage ratings: which corners get beaten, and who is lined up across.

A 0-100 grade per cornerback, 50 = league average and 15 points to a standard
deviation, the same scale as `ratings.py` and `trenches.py` so every number on
the dashboard reads the same way. High is GOOD coverage, so the corners to pick
on are the low ones.

**This module does not feed the projection, and that is a measured decision
rather than an omission.** EXPERIMENTS.md, "Per-corner coverage grades": every
grade here was fitted against the production board's own residual
leave-one-season-out and scored on MAE and on top-N ordering. Nothing paid.
Wide receiver was *harmful* (−0.0032 MAE, interval clear of zero on the wrong
side), tight end and running back were exactly zero, and quarterback was the
only positive and it straddled zero. So these are context numbers for a human
reading a slate, in the same spirit as the trench bands, and `predict.py` never
imports this file. The one coverage adjustment that IS shipped is tight-end
receiving only, in sharp-edge `nfl/matchup.py`, and it comes from the
team-level work in "Coverage matchup — a null result, and one narrow win".

**What is measurable about a cornerback, and what is not.** This is the part
worth knowing before reading any corner ranking anywhere. Pro Football
Reference charts coverage per defender weekly, and the metrics divide sharply
when a corner is correlated against himself (odd weeks vs even, pooled over his
career, 125 corners with 100+ targets a side):

    targets per game                 0.728    how often he is thrown at
    average depth of target          0.733    how deep he is tested
    completion % allowed             0.519    whether he wins the rep
    yards after catch per completion 0.341
    passer rating allowed            0.204
    yards allowed per target         0.188
    TDs allowed per target           0.086

Only the first three survive, and the gap is not a sample-size artifact — the
within-season correlation for yards per target is 0.037 and raising the minimum
target count does not move it, while the tighter career samples turn it
negative. **"Yards allowed" is the number a corner ranking most wants to use
and it is the one that does not persist**, because yards per completion is
carried by a handful of long plays and a corner sees a median of 26 targets a
season. Completion rate is a per-event binary on the same sample, which is why
it survives where the yardage built on top of it does not.

So a grade here is built from completion % allowed alone, heavily shrunk, with
target volume and depth of target shown beside it as the role columns they are.
A corner who is avoided and a corner who is good look identical in an outcome
rate; the volume column is the only thing that tells them apart.

**What is still missing is the assignment.** nflverse has no shadow-coverage
field, so this module can say a defence has one weak corner and cannot say
which receiver draws him. `cover_spread` is the honest version of a "pick on
him" read: how unequal a secondary is, which is the part of that read that does
not need the assignment.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .ratings import to_grade

logger = logging.getLogger(__name__)

# nflreadpy 0.1.3 has no load_pfr_advstats (the R package does), so the
# nflverse-data release assets are read directly, one parquet per season.
_RELEASE = ("https://github.com/nflverse/nflverse-data/releases/download/"
            "pfr_advstats/advstats_week_def_%d.parquet")
# Positions whose coverage is a corner's job. Safeties are charted too and are
# deliberately excluded: a safety's target rate is a scheme artifact.
COVER_POSITIONS = ("CB", "DB")
# Targets of league-average coverage mixed into every corner's grade. Set from
# the measured reliability: completion % allowed correlates 0.519 against
# itself at ~200 career targets a side, so a single season needs most of its
# weight on the mean. 60 puts a 60-target corner at half his own number.
SHRINK_TARGETS = 60.0
# A corner needs this many charted targets before he is listed at all.
MIN_TARGETS = 15
# How much of last season carries into an early-season grade.
PRIOR_WEIGHT = 0.5
# Minimum regular corners before a defence gets a spread number.
MIN_CORNERS = 2

BANDS = [(0, 35, "pick on"), (35, 45, "soft"), (45, 55, "average"),
         (55, 65, "sticky"), (65, 101, "avoid")]


def load_charting(seasons) -> pd.DataFrame:
    """Weekly per-defender coverage charting for the given seasons."""
    frames = []
    for season in sorted({int(s) for s in seasons}):
        try:
            frames.append(pd.read_parquet(_RELEASE % season))
        except Exception as exc:          # a season not yet published
            logger.warning("PFR def charting %d unavailable: %s: %s", season,
                           type(exc).__name__, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _positions() -> pd.DataFrame:
    """pfr_id -> position, to keep corners apart from safeties."""
    import nflreadpy as nfl

    from .utils import to_pandas

    p = to_pandas(nfl.load_players())
    p = p[p.pfr_id.notna()][["pfr_id", "position", "display_name"]]
    return p.drop_duplicates("pfr_id").rename(
        columns={"pfr_id": "pfr_player_id", "display_name": "player_name"})


def corner_weeks(charting: pd.DataFrame, players: pd.DataFrame | None = None
                 ) -> pd.DataFrame:
    """One row per cornerback per week: what was thrown at him and what stuck."""
    if charting.empty:
        return pd.DataFrame()
    d = charting[charting.get("game_type", "REG") == "REG"].copy()
    players = _positions() if players is None else players
    d = d.merge(players, on="pfr_player_id", how="left")
    d = d[d.position.isin(COVER_POSITIONS)].copy()
    d["targets"] = d.def_targets.fillna(0)
    d["completions"] = d.def_completions_allowed.fillna(0)
    d["air"] = d.def_adot.fillna(0) * d.targets
    d["yards"] = d.def_yards_allowed.fillna(0)
    keep = ["season", "week", "team", "opponent", "pfr_player_id",
            "player_name", "pfr_player_name", "position", "targets",
            "completions", "air", "yards"]
    return d[[c for c in keep if c in d.columns]]


def _shrunk_rate(completions, targets, league):
    """A completion rate pulled toward the league mean by SHRINK_TARGETS."""
    return (completions + SHRINK_TARGETS * league) / (targets + SHRINK_TARGETS)


def corner_grades(cw: pd.DataFrame, season: int, week: int | None = None,
                  *, prior_weight: float = PRIOR_WEIGHT) -> pd.DataFrame:
    """Per-corner grades for `season`, using only weeks strictly before `week`.

    Leakage-safe by construction: the week being graded never contributes. Early
    in a season there is not enough charting for a grade, so the previous season
    carries `prior_weight` of the load and fades as targets accrue — the same
    shape of blend the team ratings use.
    """
    if cw.empty:
        return pd.DataFrame()
    cur = cw[(cw.season == season)]
    if week is not None:
        cur = cur[cur.week < int(week)]
    prev = cw[cw.season == season - 1]
    league = ((cur.completions.sum() + prev.completions.sum())
              / max(cur.targets.sum() + prev.targets.sum(), 1))

    def agg(frame):
        if frame.empty:
            return pd.DataFrame(columns=["targets", "completions", "air",
                                         "yards", "games"])
        return frame.groupby("pfr_player_id").agg(
            targets=("targets", "sum"), completions=("completions", "sum"),
            air=("air", "sum"), yards=("yards", "sum"), games=("week", "nunique"))

    a, b = agg(cur), agg(prev)
    idx = a.index.union(b.index)
    a = a.reindex(idx).astype(float).fillna(0.0)
    b = b.reindex(idx).astype(float).fillna(0.0)

    tgt = a.targets + prior_weight * b.targets
    comp = a.completions + prior_weight * b.completions
    rate = _shrunk_rate(comp, tgt, league)

    out = pd.DataFrame({
        "pfr_player_id": idx,
        "targets": a.targets.values,
        "games": a.games.values,
        "targets_weighted": tgt.values,
        "cmp_pct_allowed": np.where(a.targets > 0,
                                    a.completions / a.targets.replace(0, np.nan), np.nan),
        "shrunk_cmp_allowed": rate.values,
        "adot": np.where(a.targets > 0, a.air / a.targets.replace(0, np.nan), np.nan),
        "yards_per_target": np.where(a.targets > 0,
                                     a.yards / a.targets.replace(0, np.nan), np.nan),
    })
    out["targets_per_game"] = np.where(out.games > 0, out.targets / out.games, np.nan)
    # `to_grade` wants a mean-0 rating, so centre on the league completion rate
    # rather than on the mean of whoever happens to be listed. invert: allowing
    # fewer completions is better coverage, so high = good.
    out["cover_grade"] = to_grade(
        pd.Series(out.shrunk_cmp_allowed.values) - league, invert=True).values
    out["league_cmp_allowed"] = league
    out["band"] = [band(g) for g in out.cover_grade]

    names = (cw.sort_values(["season", "week"])
             .drop_duplicates("pfr_player_id", keep="last")
             .set_index("pfr_player_id"))
    for col in ("player_name", "pfr_player_name", "team", "position"):
        if col in names.columns:
            out[col] = out.pfr_player_id.map(names[col])
    if "player_name" in out.columns and "pfr_player_name" in out.columns:
        out["player_name"] = out.player_name.where(out.player_name.notna(),
                                                   out.pfr_player_name)
    out = out[out.targets_weighted >= MIN_TARGETS]
    return out.sort_values("cover_grade").reset_index(drop=True)


def team_coverage(cg: pd.DataFrame) -> pd.DataFrame:
    """A defence's coverage grade, assembled from the corners who played it.

    Target-weighted, because a corner nobody throws at should not swing the
    team number, and `cover_spread` — the gap between the best and worst
    regular corner's completion rate — alongside it. The spread is the
    exploitable-side measure: a secondary can average out fine and still have
    one side worth attacking, and that asymmetry is the part of a "pick on him"
    read that survives not knowing the coverage assignment.
    """
    if cg.empty or "team" not in cg.columns:
        return pd.DataFrame()
    rows = []
    for team, grp in cg.groupby("team"):
        w = grp.targets_weighted
        rate = float((grp.shrunk_cmp_allowed * w).sum() / w.sum()) if w.sum() else np.nan
        regular = grp[grp.targets >= MIN_TARGETS]
        spread = (float(regular.shrunk_cmp_allowed.max()
                        - regular.shrunk_cmp_allowed.min())
                  if len(regular) >= MIN_CORNERS else np.nan)
        # The corner to attack has to be one who plays. Graded off last
        # season's charting alone he may be a healthy scratch, and naming him
        # is the one error this table cannot afford.
        weakest = (regular if len(regular) else grp).nsmallest(1, "cover_grade")
        rows.append({
            "team": team,
            "corners": len(grp),
            "cover_rate": rate,
            "cover_spread": spread,
            "adot": float((grp.adot.fillna(0) * w).sum() / w.sum()) if w.sum() else np.nan,
            "weakest_corner": (weakest.player_name.iloc[0] if len(weakest) else None),
            "weakest_grade": (float(weakest.cover_grade.iloc[0]) if len(weakest) else np.nan),
        })
    out = pd.DataFrame(rows)
    out["cover_grade"] = to_grade(out.cover_rate - out.cover_rate.mean(),
                                  invert=True)
    out["band"] = [band(g) for g in out.cover_grade]
    return out.sort_values("cover_grade").reset_index(drop=True)


def matchups_for_week(tc: pd.DataFrame, schedule: pd.DataFrame, season: int,
                      week: int) -> pd.DataFrame:
    """Every game in a week from the receiving side: who faces which secondary.

    One row per offence, carrying the defence it is pointed at. Attaching the
    actual receivers is left to the caller, which has a roster: this module
    deliberately does not guess at an alignment it cannot see.
    """
    if tc.empty or schedule.empty:
        return pd.DataFrame()
    games = schedule[(schedule.season == season) & (schedule.week == int(week))]
    idx = tc.set_index("team")
    rows = []
    for _, g in games.iterrows():
        h, a = g.get("home_team"), g.get("away_team")
        if pd.isna(h) or pd.isna(a):
            continue
        for off, dfn, home in ((h, a, True), (a, h, False)):
            if dfn not in idx.index:
                continue
            d = idx.loc[dfn]
            rows.append({
                "season": season, "week": int(week), "team": off,
                "opponent": dfn, "home": home,
                "opp_cover_grade": float(d.cover_grade),
                "opp_cover_band": d.band,
                "opp_cover_spread": float(d.cover_spread)
                if pd.notna(d.cover_spread) else np.nan,
                "opp_weakest_corner": d.weakest_corner,
                "opp_weakest_grade": float(d.weakest_grade)
                if pd.notna(d.weakest_grade) else np.nan,
            })
    out = pd.DataFrame(rows)
    return (out.sort_values("opp_cover_grade").reset_index(drop=True)
            if not out.empty else out)


def band(rating: float) -> str:
    """The label for a grade. Read from the defence's side: `pick on` is a
    secondary to attack, `avoid` is one to leave alone."""
    if rating is None or (isinstance(rating, float) and np.isnan(rating)):
        return "unknown"
    for lo, hi, label in BANDS:
        if lo <= rating < hi:
            return label
    return "unknown"


def report(season: int, week: int | None = None, *, seasons=None,
           charting: pd.DataFrame | None = None, schedule=None,
           shells: pd.DataFrame | None = None, with_shells: bool = True):
    """The whole report in one call: corners, secondaries, the week's games, and
    each defence's coverage-shell mix.

    `shells` is the one part that may be stale rather than current — nflverse
    publishes no charting after SHELL_LAST_SEASON — so it is a separate key
    with its own seasons on every row, never merged into the corner grades as
    if it were measured the same week. Pass `with_shells=False` to skip the
    download.
    """
    seasons = seasons or [season - 1, season]
    charting = load_charting(seasons) if charting is None else charting
    cw = corner_weeks(charting)
    cg = corner_grades(cw, season, week)
    tc = team_coverage(cg)
    mu = pd.DataFrame()
    if week is not None:
        if schedule is None:
            import nflreadpy as nfl

            from .utils import to_pandas
            schedule = to_pandas(nfl.load_schedules(seasons=[season]))
        mu = matchups_for_week(tc, schedule, season, week)
    sm = pd.DataFrame()
    if with_shells:
        sp = shell_plays(load_shells(seasons) if shells is None else shells)
        sm = shell_mix(sp, season, week)
    return {"corners": cg, "teams": tc, "matchups": mu, "shells": sm}


# ---------------------------------------------------------------------------
# Coverage shells: what a defence actually plays
# ---------------------------------------------------------------------------
#
# The assignment is what this module says it cannot see (see the module
# docstring). The SHELL is the other half of that gap and nflverse does chart
# it — in the participation release, from the NFL's own charting:
# `defense_coverage_type` (COVER_0..COVER_4, COVER_6, COVER_9, 2_MAN, COMBO)
# and `defense_man_zone_type`.
#
# **It stops at 2025.** nflreadpy refuses a later season outright ("Season must
# be between 2016 and 2025"), and nothing else in nflverse carries the fields:
# not `load_pbp`, not `load_ftn_charting` (pressure and box counts, no
# coverage), not `load_nextgen_stats`. Checked 2026-11-04. So a CURRENT-season
# shell mix cannot be built from nflverse at all, and what this offers for 2026
# is a scheme tendency last measured in 2025. Scheme is a coaching trait and
# travels with a coordinator, which makes a stale mix worth showing — but a
# defence with a new coordinator is described by its predecessor, so every row
# carries the seasons behind it and the plays from each.

# The last season nflverse publishes participation (and therefore shells) for.
SHELL_LAST_SEASON = 2025
# Charted plays a defence needs before its mix is worth reporting. The median
# team-week carries ~38 charted plays, so this is about two games' worth.
MIN_SHELL_PLAYS = 60
# The shells worth their own column, in nflverse's own spelling. COMBO and
# BLOWN are charted too and are left in `other`: a few dozen plays a season,
# and "the chart could not tell" is not a tendency.
SHELLS = ("COVER_0", "COVER_1", "COVER_2", "COVER_3", "COVER_4", "COVER_6",
          "COVER_9", "2_MAN")


def load_shells(seasons) -> pd.DataFrame:
    """Per-play coverage-shell charting for the given seasons.

    Returns an empty frame rather than raising for a season nflverse has no
    participation for, which from 2026 is every season — see the note above.
    """
    import nflreadpy as nfl

    from .utils import to_pandas

    frames = []
    for season in sorted({int(s) for s in seasons}):
        if season > SHELL_LAST_SEASON:
            logger.warning(
                "nflverse publishes no coverage charting for %d (participation "
                "ends %d), so no shell mix can be built for it",
                season, SHELL_LAST_SEASON)
            continue
        try:
            frames.append(to_pandas(nfl.load_participation(seasons=[season])))
        except Exception as exc:
            logger.warning("participation %d unavailable: %s: %s", season,
                           type(exc).__name__, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def shell_plays(part: pd.DataFrame) -> pd.DataFrame:
    """One row per charted play, from the DEFENCE's side.

    Participation names only the offence (`possession_team`), so the defence is
    read off the game id — "2025_09_BAL_JAX" is season, week, away, home — and
    is whichever of the two is not in possession. Doing it from the id rather
    than a schedule join keeps this function offline and testable.
    """
    if part is None or part.empty or "nflverse_game_id" not in part.columns:
        return pd.DataFrame()
    d = part.copy()
    ids = d.nflverse_game_id.astype(str).str.split("_", expand=True)
    if ids.shape[1] < 4:
        return pd.DataFrame()
    d["season"] = pd.to_numeric(ids[0], errors="coerce")
    d["week"] = pd.to_numeric(ids[1], errors="coerce")
    away, home = ids[2], ids[3]
    d["defense"] = home.where(d.possession_team != home, away)
    d["shell"] = d.get("defense_coverage_type")
    mz = d.get("defense_man_zone_type")
    d["man_zone"] = mz.where(mz.isin(["MAN_COVERAGE", "ZONE_COVERAGE"])) \
        if mz is not None else None
    d = d[d.season.notna() & d.week.notna() & d.defense.notna()]
    keep = d[["season", "week", "defense", "shell", "man_zone"]].copy()
    keep["season"] = keep.season.astype(int)
    keep["week"] = keep.week.astype(int)
    return keep.reset_index(drop=True)


def shell_mix(sp: pd.DataFrame, season: int, week: int | None = None, *,
              prior_weight: float = PRIOR_WEIGHT,
              min_plays: int = MIN_SHELL_PLAYS) -> pd.DataFrame:
    """A defence's coverage-shell distribution, one row per defence.

    Windowed exactly like `corner_grades`: only weeks strictly before `week`
    count, and the previous season carries `prior_weight` of the load so an
    early-season mix is not built on two games. A share is a percentage of
    charted plays, so the shell columns sum to 100 with `other`.

    `shell_seasons`, `plays` and `plays_prior` are on every row because the mix
    may be entirely historical: for 2026 nflverse has no charting at all, and a
    row that is 100% last season's scheme has to say so rather than look like
    this week's tendency.
    """
    if sp is None or sp.empty:
        return pd.DataFrame()
    cur = sp[sp.season == int(season)]
    if week is not None:
        cur = cur[cur.week < int(week)]
    prev = sp[sp.season == int(season) - 1]
    if cur.empty and prev.empty:
        return pd.DataFrame()

    def counts(frame, weight):
        """Weighted play counts per (defence, shell) and per man/zone."""
        if frame.empty:
            return {}, {}, {}
        shells = frame[frame.shell.notna()]
        by_shell = {}
        for (team, shell), n in shells.groupby(["defense", "shell"]).size().items():
            by_shell[(team, shell)] = by_shell.get((team, shell), 0.0) + n * weight
        mz = frame[frame.man_zone.notna()]
        man = (mz.man_zone == "MAN_COVERAGE").groupby(mz.defense).sum() * weight
        tot = mz.groupby("defense").size() * weight
        return by_shell, man.to_dict(), tot.to_dict()

    cs, cman, ctot = counts(cur, 1.0)
    ps, pman, ptot = counts(prev, prior_weight)

    teams = sorted({t for t, _ in cs} | {t for t, _ in ps}
                   | set(ctot) | set(ptot))
    rows = []
    for team in teams:
        shell_counts = {}
        for (t, shell), n in list(cs.items()) + list(ps.items()):
            if t == team:
                shell_counts[shell] = shell_counts.get(shell, 0.0) + n
        total = sum(shell_counts.values())
        cur_plays = sum(n for (t, _), n in cs.items() if t == team)
        prior_plays = sum(n for (t, _), n in ps.items() if t == team)
        if total < min_plays:
            continue
        row = {"defense": team,
               "plays": round(cur_plays, 1),
               "plays_prior": round(prior_plays, 1)}
        # `other` absorbs the rounding as well as COMBO/BLOWN, so the
        # columns of a published row add to exactly 100 rather than to 99.8.
        named = 0.0
        for shell in SHELLS:
            share = round(100.0 * shell_counts.get(shell, 0.0) / total, 1)
            row[shell.lower()] = share
            named += share
        row["other"] = round(max(100.0 - named, 0.0), 1)
        man = cman.get(team, 0.0) + pman.get(team, 0.0)
        tot = ctot.get(team, 0.0) + ptot.get(team, 0.0)
        row["man_rate"] = round(100.0 * man / tot, 1) if tot else np.nan
        row["zone_rate"] = round(100.0 - row["man_rate"], 1) if tot else np.nan
        # Which seasons this row is actually made of, newest first.
        seasons = ([int(season)] if cur_plays else []) + \
                  ([int(season) - 1] if prior_plays else [])
        row["shell_seasons"] = ",".join(str(s) for s in seasons)
        row["current_season_plays"] = bool(cur_plays)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("man_rate", ascending=False).reset_index(drop=True)
