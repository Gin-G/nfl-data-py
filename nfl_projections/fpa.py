"""Fantasy points allowed per position, adjusted for who the defence has faced.

Raw FPA -- sum the opposing position's points each week, average the weeks --
cannot tell two very different defences apart. One allowed a lot to tight ends
because it faced four of the best tight ends; the other allowed a lot while
facing replacements. Only the second is a target, and raw FPA ranks them the
same.

So every opposing player is measured against **his own baseline** instead: the
mean FanDuel points he was running at entering that week. A defence's rating is
the average amount the players it faced beat or missed their own level by, which
makes a hard schedule net out on its own. Measured 2019-2026, the two rankings
disagree a lot: Spearman 0.60 for receivers, ~0.76 elsewhere.

**On whether that disagreement is an improvement, the honest answer is: it is a
wash.** Fitted on 2019-2023 and scored on 2024-2026, with raw and adjusted put
on the same percentile scale so the comparison is fair, neither wins
consistently -- adjusted is ahead for receivers in both eras, raw is ahead for
tight ends out of sample, and quarterbacks and backs swap places between the
two halves with every interval overlapping. An earlier claim that the
adjustment roughly doubled the signal came from comparing a percentile rank
against a clipped z-score grade, and did not survive matching the transforms.
What the adjustment reliably buys is interpretability -- `allowed` next to
`expected` says whether a number is a soft defence or a hard schedule, which raw
FPA cannot say at all -- and that is the reason to keep it. The schedule term it
subtracts is not itself a consistent predictor in either direction.

Week to week all of this is small, and meant to be read that way. Against a
player's gap from his own baseline the correlations run 0.01-0.09, and once the
trench and coverage grades are in hand the FPA grade adds nothing that survives
its own interval. This is a tiebreak between comparable players and a sanity
check on a build, not a signal to chase -- the same standing the trench and
coverage work earned. Nothing here enters the optimizer's objective.

**Outliers are clipped on purpose.** In week 8 of 2026, Tampa Bay completed no
passes against Atlanta, ran for 310 yards and won. That week is instructive
about both halves of the problem. nflverse recorded only three Tampa Bay players
for it -- Mayfield 1.4, Gainwell 17.0, Irving 24.0 -- so the WR 0.0 and TE 0.0
that look so alarming never enter either table: with no player-week there is
nothing to average, and Atlanta's receiver numbers run over weeks 1-7 only. What
the game does do is land on the running-back side, where Gainwell's +12.29 gap
against a 4.71 baseline is the kind of row the +/-12 winsor exists for. Taking
the median instead is
over-correction -- it is the worst of four estimators for receivers (0.055
against the mean's 0.072), because a defence that genuinely gets gashed carries
real information. Winsorising keeps the direction and caps the magnitude, and it
is the best or near-best for RB and WR. `ratings` also reports how much the
single most extreme week is worth, so a case like Atlanta's is visible rather
than quietly smoothed away.
"""

import logging

import numpy as np
import pandas as pd

from .ratings import to_grade
from .scoring import fanduel_points_from_stats

logger = logging.getLogger(__name__)

POSITIONS = ("QB", "RB", "WR", "TE")

# A player needs this many prior games before his baseline is worth subtracting.
MIN_PRIOR_GAMES = 3
# Player-observations a defence needs before its own number is trusted over the
# league average. One week against one position is ~2-4 players.
SHRINK = 8.0
# Per-player delta is clipped here before averaging. +/-12 FanDuel points is a
# genuinely extreme week against a baseline, and past it we are reading weather
# and game script rather than the defence.
WINSOR = 12.0
# A week is called out as an outlier when dropping it moves the rating by this.
OUTLIER_SWING = 1.0

# Higher grade = softer = better to target, which is the direction a lineup
# builder reads. The bands match `trenches.OPPONENT_BANDS` in spirit.
BANDS = [(0, 35, "very tough"), (35, 45, "tough"), (45, 55, "average"),
         (55, 65, "soft"), (65, 101, "very soft")]


def band(grade) -> str:
    """Label for a 0-100 FPA grade."""
    if grade is None or not np.isfinite(grade):
        return "unknown"
    for lo, hi, label in BANDS:
        if lo <= grade < hi:
            return label
    return BANDS[-1][2]


def _num(weekly: pd.DataFrame, *names) -> pd.Series:
    """Numeric Series for the first of `names` present, zeros when none is.

    `weekly.get(name, 0)` hands back the bare int for a missing column, which
    then has no `.fillna` -- a crash on any frame that happens not to carry a
    fumbles column, which several nflverse shapes do not.
    """
    for name in names:
        if name in weekly.columns:
            return pd.to_numeric(weekly[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=weekly.index)


def _points(weekly: pd.DataFrame) -> pd.Series:
    """FanDuel points per player-week. Half-PPR, not the PPR the old FPA used."""
    for col in ("fanduel_points", "fd", "actual"):
        if col in weekly.columns:
            return pd.to_numeric(weekly[col], errors="coerce")
    # Lost fumbles and two-point conversions, as FanDuel scores them. This
    # used to skip conversions and sack fumbles.
    return fanduel_points_from_stats(weekly)


def deltas(weekly: pd.DataFrame, *, min_prior_games=MIN_PRIOR_GAMES) -> pd.DataFrame:
    """Per player-week: points, the player's own entering baseline, and the gap.

    The baseline is expanding over the player's prior games in the same season
    and is shifted, so a week never contributes to its own expectation.
    """
    need = {"player_id", "season", "week", "position"}
    missing = need - set(weekly.columns)
    if missing:
        raise ValueError(f"weekly is missing {sorted(missing)}")
    d = weekly.copy()
    d["fd"] = _points(d)
    defence = next((c for c in ("opponent_team", "opponent", "defense") if c in d.columns), None)
    if defence is None:
        raise ValueError("weekly needs an opponent_team / opponent / defense column")
    d["defense"] = d[defence]
    d = d.sort_values(["season", "week"])
    d["baseline"] = (d.groupby(["player_id", "season"], observed=True)["fd"]
                     .transform(lambda s: s.shift(1).expanding(min_prior_games).mean()))
    d = d[d["position"].isin(POSITIONS) & d["defense"].notna()]
    d = d.dropna(subset=["baseline"])
    d["delta"] = d["fd"] - d["baseline"]
    return d[["season", "week", "defense", "position", "player_id",
              "player_name", "fd", "baseline", "delta"]]


def ratings(weekly: pd.DataFrame, season: int, week: int | None = None, *,
            shrink=SHRINK, winsor=WINSOR) -> pd.DataFrame:
    """One row per (defence, position) from games strictly BEFORE `week`.

    Columns: `allowed` (raw FPA per game, what the old table showed), `expected`
    (what the players it faced were running at, which is the schedule it has
    had), `adjusted` (the winsorised mean gap, points per player), `grade`
    (0-100, higher = softer = better to target), `band`, `weeks`, and
    `worst_week` / `worst_week_swing` so a single distorting game is visible.
    """
    d = deltas(weekly)
    d = d[d["season"] == int(season)]
    if week is not None:
        d = d[pd.to_numeric(d["week"], errors="coerce") < int(week)]
    if d.empty:
        return pd.DataFrame(columns=["defense", "position", "allowed", "expected",
                                     "adjusted", "grade", "band", "weeks",
                                     "worst_week", "worst_week_swing"])

    d = d.copy()
    d["clipped"] = d["delta"].clip(-winsor, winsor)
    weekly_sum = (d.groupby(["defense", "position", "week"], observed=True)
                  .agg(allowed=("fd", "sum"), expected=("baseline", "sum"),
                       wk_delta=("clipped", "mean")).reset_index())

    rows = []
    for (defence, position), g in weekly_sum.groupby(["defense", "position"], observed=True):
        obs = int(d[(d.defense == defence) & (d.position == position)].shape[0])
        mean_delta = float(g["wk_delta"].mean())
        # What the single most extreme week is worth: drop each in turn and keep
        # the biggest move. This is the Atlanta check.
        swing, worst = 0.0, None
        if len(g) > 2:
            for _, row in g.iterrows():
                without = float(g.loc[g.week != row.week, "wk_delta"].mean())
                if abs(without - mean_delta) > abs(swing):
                    swing, worst = without - mean_delta, int(row.week)
        rows.append({
            "defense": defence, "position": position,
            "allowed": float(g["allowed"].mean()),
            "expected": float(g["expected"].mean()),
            "adjusted": mean_delta * obs / (obs + shrink),
            "weeks": int(len(g)),
            "worst_week": worst if abs(swing) >= OUTLIER_SWING else None,
            "worst_week_swing": round(swing, 2) if abs(swing) >= OUTLIER_SWING else 0.0,
        })
    out = pd.DataFrame(rows)

    # Graded inside a position: a tight end's points are not a receiver's.
    out["grade"] = np.nan
    for position, g in out.groupby("position", observed=True):
        centred = g["adjusted"] - g["adjusted"].mean()
        spread = centred.std()
        if len(centred) < 2 or not np.isfinite(spread) or spread == 0:
            # One team in the group, or every team identical -- early in a
            # season both happen. `to_grade` divides by the spread and would
            # hand back NaN, which reads as "no data" rather than "average".
            out.loc[g.index, "grade"] = 50.0
        else:
            out.loc[g.index, "grade"] = to_grade(centred).round(1)
    out["band"] = out["grade"].map(band)
    return out.sort_values(["position", "grade"], ascending=[True, False]) \
              .reset_index(drop=True)


def matchups_for_week(rated: pd.DataFrame, schedule: pd.DataFrame, season: int,
                      week: int) -> pd.DataFrame:
    """For every team playing, the grade of the defence it faces, per position.

    One row per (offence team, position), so it joins straight onto a player
    pool by that player's own team and position.
    """
    games = schedule[(schedule["season"] == season) & (schedule["week"] == week)]
    if not len(games) or not len(rated):
        return pd.DataFrame(columns=["team", "opponent", "position", "fpa_grade",
                                     "fpa_band", "fpa_allowed", "fpa_adjusted"])
    look = rated.set_index(["defense", "position"])
    rows = []
    for g in games.itertuples():
        for team, opponent in ((g.home_team, g.away_team), (g.away_team, g.home_team)):
            for position in POSITIONS:
                try:
                    r = look.loc[(opponent, position)]
                except KeyError:
                    continue
                rows.append({
                    "team": team, "opponent": opponent, "position": position,
                    "fpa_grade": float(r["grade"]), "fpa_band": r["band"],
                    "fpa_allowed": float(r["allowed"]),
                    "fpa_adjusted": float(r["adjusted"]),
                    "fpa_weeks": int(r["weeks"]),
                    "fpa_outlier_week": r["worst_week"],
                })
    return pd.DataFrame(rows)


def report(rated: pd.DataFrame, position: str, *, n=10) -> str:
    """The softest and toughest defences for one position, as text."""
    s = rated[rated["position"] == str(position).upper()]
    if not len(s):
        return f"no {position} ratings"
    s = s.sort_values("grade", ascending=False)
    lines = [f"{position.upper()}: softest to toughest "
             f"(grade 50 = league average, higher = target)"]
    head = pd.concat([s.head(n), s.tail(n)]).drop_duplicates("defense")
    for _, r in head.iterrows():
        note = ""
        # None becomes NaN once this is a frame, and NaN is truthy.
        worst = pd.to_numeric(r["worst_week"], errors="coerce")
        if np.isfinite(worst):
            note = (f"   [week {int(worst)} is worth "
                    f"{r['worst_week_swing']:+.1f} of this]")
        lines.append(f"  {r['defense']:<4} grade {r['grade']:>5.1f} {r['band']:<11}"
                     f" allowed {r['allowed']:>5.1f}  faced {r['expected']:>5.1f}"
                     f"  adjusted {r['adjusted']:>+5.2f}{note}")
    return "\n".join(lines)
