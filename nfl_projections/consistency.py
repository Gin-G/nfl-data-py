"""How steady a player has been — and which kind of steadiness is real.

The request this answers is "give me the consistent players, the ones with the
lowest variance around their mean". Measured, that request splits in two, and
only one half of it exists.

**Scoring consistency is not a property of a player.** Split-half on 2025
(odd weeks against even, 210 players with 6+ games and a 5-point mean), the
correlation of a player's fantasy-point standard deviation between his own two
halves is +0.111, CI [-0.025, +0.243]; adjusted for scoring level it is
**-0.047**, CI [-0.181, +0.089]. On 2026 through week 9 the level-adjusted
figure is -0.096, CI [-0.241, +0.054]. The yardstick on the same players is
the MEAN, which self-correlates +0.643. So a player who scored steadily in the
first half of a season is no more likely than anyone else to score steadily in
the second: ranking on it is ranking on noise.

**Raw standard deviation is worse than noise, it is actively misleading**,
because dispersion scales with scoring level: SD correlates +0.647 with the
mean. The ten "most consistent" players of 2025 by lowest SD are Malik
Washington (5.2 points a game), Darius Slayton (5.5), Isiah Pacheco (6.0),
Greg Dulcich (5.5) — a list of players nobody would roster, which is the whole
reason their scores do not move.

**Volume steadiness is real.** A player's opportunities per game — carries plus
targets, or pass attempts for a quarterback — self-correlate +0.943 across
halves, and their DISPERSION self-correlates +0.734, still +0.412 once the
volume level itself is controlled, CI [+0.293, +0.518]. On 2026's nine weeks
that drops to +0.070, CI [-0.080, +0.217]: four or five games a half cannot
measure a dispersion, so in-season the figure is thin rather than absent.

**And it transmits, weakly, to the thing actually wanted**: a player's
opportunity CV in one half predicts his fantasy CV in the other at +0.231, CI
[+0.099, +0.355] (2026: +0.285, CI [+0.142, +0.416]). Steady usage does buy
some steadiness of scoring. It is the only route to it that measures.

The single best predictor of a player's future scoring consistency is, flatly,
his scoring LEVEL: fantasy mean against next-half fantasy CV is -0.338 (2026:
-0.317), beating past CV (+0.235) and opportunity volume (-0.237). The
consistent player is mostly just the good player.

So this module ranks on volume, never on scoring dispersion. The scoring
numbers are carried because they are what a person wants to read, and labelled
as description. See EXPERIMENTS.md "Consistency".
"""

import numpy as np
import pandas as pd

from .opportunity import _points, _volume

#: Below this, a mean and a spread are two or three games, not a tendency.
MIN_GAMES = 4

#: A player touching the ball twice a game can have a tiny CV and is not what
#: anyone means by consistent. Volume is the x-axis of this question, not a
#: filter to apply afterwards — but a floor keeps the ranking honest.
MIN_OPPORTUNITIES = 5.0

#: How many recent games count as "lately", for the trend.
RECENT_GAMES = 3

#: Bars a lineup slot actually cares about clearing.
HIT_BARS = (10.0, 15.0)

POSITIONS = ("QB", "RB", "WR", "TE")


def weekly_table(weekly: pd.DataFrame) -> pd.DataFrame:
    """Player-weeks reduced to the two columns this asks about.

    Reuses `opportunity._volume` and `opportunity._points`, so "opportunity"
    and "a fantasy point" mean exactly what they mean everywhere else — a
    second definition of either is how two screens end up disagreeing.
    """
    if weekly is None or weekly.empty:
        return pd.DataFrame()
    frame = _volume(weekly)
    frame["points"] = _points(frame)
    keep = ["player_id", "player_name", "position", "season", "week", "opp",
            "points"]
    if "player_display_name" in frame.columns:
        frame["player_name"] = frame["player_display_name"].fillna(
            frame.get("player_name"))
    team_col = "_team" if "_team" in frame.columns else "team"
    frame["team"] = frame.get(team_col)
    return frame[[c for c in keep + ["team"] if c in frame.columns]]


def _steadiness(cv: pd.Series) -> pd.Series:
    """A coefficient of variation as a 0-100 score, high meaning steady.

    `100 * (1 - cv)`, floored at 0. Deliberately a restatement rather than a
    fit: a CV of 0.13 becomes 87 and one of 0.50 becomes 50, and anybody can
    check the arithmetic. A percentile would read better and would quietly
    depend on who else happened to be in the cohort.
    """
    return (100.0 * (1.0 - cv.clip(lower=0.0, upper=1.0))).round(1)


def player_consistency(weekly: pd.DataFrame, *, min_games: int = MIN_GAMES,
                       through_week: int | None = None) -> pd.DataFrame:
    """One row per player: how much he gets, how steadily, and what he scored.

    `through_week` keeps only weeks strictly before it, which is what makes
    this usable before a slate rather than after it.
    """
    table = weekly_table(weekly)
    if table.empty:
        return pd.DataFrame()
    if through_week is not None:
        table = table[table.week < int(through_week)]
    table = table[table.position.isin(POSITIONS)]
    if table.empty:
        return pd.DataFrame()

    grouped = table.sort_values("week").groupby("player_id")
    rows = []
    for player_id, games in grouped:
        if len(games) < min_games:
            continue
        points, opp = games.points.astype(float), games.opp.astype(float)
        recent = games.tail(RECENT_GAMES)
        earlier = games.head(max(len(games) - RECENT_GAMES, 0))
        opp_mean = float(opp.mean())
        points_mean = float(points.mean())
        row = {
            "player_id": player_id,
            "player_name": games.player_name.iloc[-1],
            "position": games.position.iloc[-1],
            "team": games.team.iloc[-1] if "team" in games else None,
            "games": int(len(games)),
            "opp_per_game": round(opp_mean, 2),
            "opp_sd": round(float(opp.std(ddof=1)), 2),
            "opp_cv": round(float(opp.std(ddof=1) / opp_mean), 3)
            if opp_mean else None,
            "opp_low": round(float(opp.min()), 1),
            "opp_high": round(float(opp.max()), 1),
            "points_per_game": round(points_mean, 2),
            "points_sd": round(float(points.std(ddof=1)), 2),
            "points_cv": round(float(points.std(ddof=1) / points_mean), 3)
            if points_mean else None,
            "points_low": round(float(points.min()), 1),
            "points_high": round(float(points.max()), 1),
            "points_median": round(float(points.median()), 1),
            # Lately, and against the rest. The trend is what separates a
            # steady 18 touches from a steady 18 that was 24 a month ago.
            "opp_recent": round(float(recent.opp.mean()), 2),
            "opp_trend": round(float(recent.opp.mean() - earlier.opp.mean()), 2)
            if len(earlier) else None,
            "points_recent": round(float(recent.points.mean()), 2),
        }
        for bar in HIT_BARS:
            row[f"hit_rate_{int(bar)}"] = round(float((points >= bar).mean()), 3)
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["volume_steadiness"] = _steadiness(out.opp_cv.fillna(1.0))
    # Ranked on volume steadiness among players who actually get the ball. The
    # scoring columns are never an input to this order: scoring dispersion does
    # not persist, so ordering by it would be ordering by last month's noise.
    ranked = out[out.opp_per_game >= MIN_OPPORTUNITIES].copy()
    ranked["steadiness_rank"] = ranked.volume_steadiness.rank(
        ascending=False, method="min").astype(int)
    ranked["position_rank"] = ranked.groupby("position")[
        "volume_steadiness"].rank(ascending=False, method="min").astype(int)
    out = out.merge(ranked[["player_id", "steadiness_rank", "position_rank"]],
                    on="player_id", how="left")
    return out.sort_values("volume_steadiness", ascending=False,
                           ignore_index=True)


def describe(row) -> str:
    """One sentence about a player's steadiness, in the terms that measure."""
    name_bits = []
    opp = row.get("opp_per_game")
    if opp is not None and not pd.isna(opp):
        name_bits.append(
            f"{opp:.1f} opportunities a game over {int(row['games'])}, "
            f"ranging {row['opp_low']:.0f} to {row['opp_high']:.0f}")
    steady = row.get("volume_steadiness")
    if steady is not None and not pd.isna(steady):
        name_bits.append(f"volume steadiness {steady:.0f}/100")
    trend = row.get("opp_trend")
    if trend is not None and not pd.isna(trend) and abs(trend) >= 2:
        direction = "up" if trend > 0 else "down"
        name_bits.append(
            f"{direction} {abs(trend):.1f} a game lately ({row['opp_recent']:.1f} "
            f"in the last {RECENT_GAMES})")
    sentence = "; ".join(name_bits).capitalize() + "." if name_bits else ""

    points = row.get("points_per_game")
    if points is not None and not pd.isna(points):
        sentence += (f" He has scored {row['points_low']:.1f} to "
                     f"{row['points_high']:.1f}, averaging {points:.1f}")
        hit = row.get("hit_rate_15")
        if hit is not None and not pd.isna(hit):
            sentence += f", clearing 15 in {hit * 100:.0f}% of them"
        sentence += "."
    return sentence.strip()
