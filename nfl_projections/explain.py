"""Why is this player in the lineup?

A projection is one number, and on its own it does not say whether this week's
read is right. This module puts the number back together next to the things a
person would actually check before submitting: where it sits against the
player's recent form, how many opportunities it assumes, how volatile a player
with that workload is, who the opponent front is, and whether the game has
weather in it.

Everything here is **context, never an input to the objective.** That boundary
is measured rather than cautious, and it is the same one `opportunity` already
observes: matchup grades are a tiebreak between comparable players at best
(opponent run grade r +0.166 for backs; coverage null four separate ways; a
soft opponent moves a low-volume player's boom rate 24.6% -> 29.1% and his bust
rate not at all), so letting them move the board would be fitting noise. The
one exception is wind, which is large, position-specific and NOT in the model
-- see `wind_adjustment`, which stays off until someone supplies a reading.
"""

import math

import numpy as np
import pandas as pd

# log(sd) = a + b*log(projection) + c*log1p(opportunities), fitted per position
# over 1,618 player-seasons 2019-2026. R2 0.72-0.74 for RB/WR/TE; QB is 0.11,
# because a quarterback's spread barely moves with his projection at all (the
# log-log slope is 0.22 against ~0.95 for the skill positions).
#
# The sign on `c` is the finding: holding the projection fixed, opportunities
# REDUCE spread. More opportunities is not just more points, it is steadier
# points, and it is the volume rather than the deep-threat archetype that does
# it -- aDOT manages a partial correlation of -0.05 (RB) and +0.14 (WR) against
# volume's -0.34 and -0.27.
_CV_MODEL = {
    "QB": (1.390, 0.182, +0.082),
    "RB": (0.595, 0.922, -0.315),
    "WR": (0.478, 0.989, -0.420),
    "TE": (0.003, 0.942, -0.148),
}
_CV_CLAMP = (0.25, 1.30)

# Measured over 13,304 outdoor player-weeks, 2019-2026, against a trailing
# baseline. Wind is a THRESHOLD and not a slope -- which is why it reads as
# nothing to a linear fit (QB r -0.087) and as a lot to a band:
#
#   mph     0-5     5-10    10-15   15+
#   QB    +0.23   +0.31   -0.37   -2.14
#   WR    -0.10   -0.16   -0.43   -1.37
#   TE    +0.33   +0.50   +0.15   -0.36
#   RB    +0.28   +0.25    0.00   +0.01
#
# The backs are untouched, which is the sensible shape: a team in the wind runs
# more, so the volume moves toward them. A QB plus three receivers in a 15+ mph
# game is about six points of overstatement across a lineup.
WIND_THRESHOLD = 15.0
WIND_PENALTY = {"QB": -2.14, "WR": -1.37, "TE": -0.36, "RB": 0.01}

# nflverse populates `temp`/`wind` only for games it has backfilled -- 45 of 272
# in 2026, none past week 4 -- and never ahead of kickoff, so these columns are
# a measurement tool and not a live feed. `roof` IS complete and is known in
# advance, so a dome game can at least be marked immune.
ROOF_IMMUNE = ("dome", "closed")

_POSITIONS = ("QB", "RB", "WR", "TE")


def player_cv(position, projected, opportunities=np.nan):
    """Predicted coefficient of variation for one player's week."""
    ab = _CV_MODEL.get(str(position or "").upper())
    proj = float(projected) if projected is not None else np.nan
    if ab is None or not np.isfinite(proj) or proj <= 0:
        return np.nan
    a, b, c = ab
    opp = float(opportunities) if opportunities is not None else np.nan
    if not np.isfinite(opp):
        # No workload known: fall back to the position's realized average.
        return {"QB": 0.46, "RB": 0.72, "WR": 0.72, "TE": 0.71}[str(position).upper()]
    sd = np.exp(a + b * np.log(proj) + c * np.log1p(max(opp, 0.0)))
    return float(np.clip(sd / proj, *_CV_CLAMP))


def _gamma_p(k, x):
    """Regularised lower incomplete gamma P(k, x) = gamma(k, x) / Gamma(k).

    Series below the transition point and the Lentz continued fraction above,
    which is the standard split: each converges quickly on one side of x = k+1
    and slowly on the other.
    """
    if x <= 0.0:
        return 0.0
    log_front = -x + k * math.log(x) - math.lgamma(k)
    if x < k + 1.0:
        term = total = 1.0 / k
        for n in range(1, 500):
            term *= x / (k + n)
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return min(1.0, total * math.exp(log_front))
    # continued fraction for Q(k, x), then P = 1 - Q
    tiny = 1e-300
    b = x + 1.0 - k
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - k)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return 1.0 - math.exp(log_front) * h


def _gamma_ratio(cv, q):
    """Quantile `q` of a gamma with mean 1 and the given CV, as a ratio.

    Inverted by bisection on `_gamma_p` rather than with `scipy.stats`, because
    this package cannot assume scipy is importable. CI installs it light and
    does not have it -- a scipy import here took 21 tests down. The API image
    does happen to have scipy, but from its OWN requirements file: this package
    is installed there with --no-deps, so nothing it declares for itself is
    guaranteed to arrive. `evaluate.py` carries its own rank correlation for
    the same reason. The shape stays in [0.59, 16] because `_CV_CLAMP` bounds
    the CV, a range bisection crosses to full double precision well inside its
    iteration budget.
    """
    if not np.isfinite(cv) or cv <= 0 or not 0.0 < q < 1.0:
        return np.nan
    k = 1.0 / cv ** 2
    lo, hi = 0.0, max(10.0, k * 10.0)
    while _gamma_p(k, hi) < q:          # widen if the bracket misses
        hi *= 2.0
        if hi > 1e6:
            return np.nan
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if hi - lo < 1e-12 * max(1.0, hi):
            break
        if _gamma_p(k, mid) < q:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi) / k)   # scale 1/k puts the mean at 1


def typical_week(projected, position, opportunities=np.nan):
    """The MEDIAN week behind a projection, which is not the projection.

    A fantasy score is right-skewed, so the booms pull the mean above what a
    player usually gives you, and how far depends on his volume. Measured:
    a low-volume back's median week is 0.752 of his season mean against a
    workhorse's 0.923 (low-volume WR 0.787, high-volume 0.878), and a gamma
    with the matching CV reproduces those to within about 0.02.
    """
    cv = player_cv(position, projected, opportunities)
    ratio = _gamma_ratio(cv, 0.50)
    return float(projected) * ratio if np.isfinite(ratio) else np.nan


def upside(projected, position, opportunities=np.nan, q=0.85):
    """The `q` quantile of the same distribution -- the tournament number."""
    cv = player_cv(position, projected, opportunities)
    ratio = _gamma_ratio(cv, q)
    return float(projected) * ratio if np.isfinite(ratio) else np.nan


def wind_adjustment(position, wind_mph):
    """Points to add for wind. 0.0 without a reading, so it is inert by default.

    Deliberately not wired into the board: nflverse has no forward-looking wind
    and we have no forecast feed, so this activates only when a caller supplies
    one. The effect is big enough to be worth sourcing -- see WIND_PENALTY.
    """
    if wind_mph is None:
        return 0.0
    try:
        mph = float(wind_mph)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(mph) or mph < WIND_THRESHOLD:
        return 0.0
    return float(WIND_PENALTY.get(str(position or "").upper(), 0.0))


def _notes(row):
    """Short, specific reasons a human would want to see on the card."""
    out = []
    pos = str(row.get("position") or "").upper()
    proj, typ = row.get("projected"), row.get("typical_week")
    if np.isfinite(proj or np.nan) and np.isfinite(typ or np.nan) and proj > 0:
        share = typ / proj
        if share < 0.80:
            out.append(f"boom/bust: typical week is {share:.0%} of the projection")
        elif share > 0.90:
            out.append(f"steady: typical week is {share:.0%} of the projection")

    opp = row.get("opp_per_game")
    if np.isfinite(opp or np.nan):
        if pos == "RB" and opp < 10:
            out.append(f"only {opp:.1f} opportunities/game -- part of a split")
        elif pos in ("WR", "TE") and opp < 4:
            out.append(f"only {opp:.1f} targets/game")

    # Relative, not absolute. `fp_per_game` counts only games played, so for
    # any player worth rostering it sits above a projection that carries
    # regression and a matchup -- a flat 3-point bar fires on all nine and
    # says nothing. A third of his own form is a gap worth a second look.
    form, vs = row.get("trailing_form"), row.get("vs_form")
    if (np.isfinite(vs or np.nan) and np.isfinite(form or np.nan) and form > 0
            and abs(vs) >= 0.33 * form):
        direction = "above" if vs > 0 else "below"
        out.append(f"board is {abs(vs):.1f} {direction} his trailing form "
                   f"({form:.1f})")

    grade, band = row.get("opp_run_grade"), row.get("run_band")
    if pos == "RB" and isinstance(band, str) and band in ("stout", "very stout"):
        out.append(f"opponent front is {band} vs the run ({grade:.0f})")

    # The run band is for backs; a receiver's matchup is the FPA grade.
    fpa_band, fpa_grade = row.get("fpa_band"), row.get("fpa_grade")
    if (pos in ("QB", "WR", "TE") and isinstance(fpa_band, str)
            and fpa_band in ("soft", "very soft", "tough", "very tough")):
        note = (f"opponent is {fpa_band} vs {pos}s "
                f"(adjusted grade {fpa_grade:.0f})")
        outlier = pd.to_numeric(row.get("fpa_outlier_week"), errors="coerce")
        if np.isfinite(outlier):
            note += f", though week {int(outlier)} skews it"
        out.append(note)

    wind = row.get("wind")
    if np.isfinite(wind or np.nan) and wind >= WIND_THRESHOLD:
        pen = WIND_PENALTY.get(pos, 0.0)
        out.append(f"{wind:.0f} mph wind (measured {pen:+.1f} for a {pos})"
                   if pen else f"{wind:.0f} mph wind (backs are unaffected)")
    elif str(row.get("roof") or "").lower() in ROOF_IMMUNE:
        out.append("indoors -- no weather risk")
    return "; ".join(out)


def explain(pool, *, season=None, week=None, opportunity=None, trenches=None,
            fpa=None, schedule=None, history=None, points_col="lineup_points",
            position_col="position", name_col="player_name", team_col="team"):
    """Add the vetting columns to a player pool or a built lineup.

    Every argument but `pool` is optional and each one soft-fails on its own:
    a missing opportunity frame costs the volume columns and nothing else, so
    this can never break a build. Returns a copy of `pool`.
    """
    out = pool.copy()
    # A frame merged from a FanDuel export carries both "Position" and our
    # "position", and a caller who renames one ends up with two columns of the
    # same name -- where `out[col]` is a DataFrame, not a Series. Keep the
    # first of any duplicate label so that can only ever cost a column.
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]
    pos = (out[position_col] if position_col in out.columns
           else pd.Series(index=out.index, dtype=object))
    pos = pos.astype(object)
    # The board projects no defences, so a merged frame's `position` is null
    # for them while FanDuel's `Position` says "D". Prefer whichever is there.
    for alt in ("Position", "Roster Position"):
        if alt in out.columns:
            pos = pos.where(pos.notna(), out[alt])
    pos = pos.where(pos.notna(), "")
    if points_col not in out.columns:
        for alt in ("lineup_points", "fanduel_fantasy_points", "projected_points", "FPPG"):
            if alt in out.columns:
                points_col = alt
                break
    proj = pd.to_numeric(out.get(points_col), errors="coerce")
    out["projected"] = proj
    out["position"] = pos

    # --- opportunity: how much volume the number assumes -------------------
    if opportunity is not None and len(opportunity):
        keys = [c for c in ("player_id",) if c in out.columns and c in opportunity.columns]
        cols = [c for c in ("opp_per_game", "opp_share", "fp_per_game", "prior_games")
                if c in opportunity.columns]
        if keys and cols:
            out = out.merge(opportunity[keys + cols].drop_duplicates(keys),
                            on=keys, how="left")
    for c in ("opp_per_game", "opp_share", "fp_per_game", "prior_games"):
        if c not in out.columns:
            out[c] = np.nan

    # --- where the number sits against recent form -------------------------
    if history is not None and "player_id" in out.columns:
        out["trailing_form"] = _trailing_form(out, history, season, week)
    elif "fp_per_game" in out.columns:
        out["trailing_form"] = pd.to_numeric(out["fp_per_game"], errors="coerce")
    else:
        out["trailing_form"] = np.nan
    out["vs_form"] = out["projected"] - out["trailing_form"]

    # --- the distribution behind the number --------------------------------
    out["volatility"] = [player_cv(p, m, o) for p, m, o
                         in zip(out["position"], out["projected"], out["opp_per_game"])]
    out["typical_week"] = [typical_week(m, p, o) for p, m, o
                           in zip(out["position"], out["projected"], out["opp_per_game"])]
    out["upside"] = [upside(m, p, o) for p, m, o
                     in zip(out["position"], out["projected"], out["opp_per_game"])]

    # --- the opponent front ------------------------------------------------
    out["opp_run_grade"] = np.nan
    out["run_band"] = None
    if trenches is not None and len(trenches) and team_col in out.columns:
        from . import trenches as tr_mod

        # `matchups_for_week` rows are one per OFFENSE, and its `run_stop`
        # column is already the opponent front's own grade -- which is the
        # half of a matchup that carries information, since a team's own line
        # is inside its players' recent form and cannot say anything new.
        grade_col = next((c for c in ("run_stop", "opp_run_stop", "run_matchup")
                          if c in trenches.columns), None)
        if grade_col and "team" in trenches.columns:
            g = trenches.drop_duplicates("team").set_index("team")[grade_col]
            vals = pd.to_numeric(out[team_col].map(g), errors="coerce")
            out["opp_run_grade"] = vals
            out["run_band"] = [tr_mod._band(v, tr_mod.OPPONENT_BANDS)
                               if np.isfinite(v) else None for v in vals]

    # --- the opposing defence against this player's position ---------------
    # The backs get a trench grade; everyone else needs the pass-game
    # equivalent. Raw fantasy points allowed cannot separate a defence that
    # gave up a lot to tight ends because it faced the best ones from a defence
    # that gave up a lot to replacements, so this is the opponent-adjusted
    # version -- see `fpa`.
    out["fpa_grade"], out["fpa_band"] = np.nan, None
    out["fpa_allowed"], out["fpa_outlier_week"] = np.nan, np.nan
    if fpa is not None and len(fpa) and team_col in out.columns:
        cols = [c for c in ("fpa_grade", "fpa_band", "fpa_allowed",
                            "fpa_adjusted", "fpa_weeks", "fpa_outlier_week")
                if c in fpa.columns]
        if {"team", "position"} <= set(fpa.columns) and cols:
            key = out[[team_col]].assign(position=out["position"].astype(str).str.upper())
            key.columns = ["team", "position"]
            look = fpa.assign(position=fpa["position"].astype(str).str.upper()) \
                      .drop_duplicates(["team", "position"]) \
                      .set_index(["team", "position"])[cols]
            joined = key.join(look, on=["team", "position"])
            for c in cols:
                out[c] = joined[c].values

    # --- weather -----------------------------------------------------------
    out["roof"], out["wind"] = None, np.nan
    if schedule is not None and len(schedule) and team_col in out.columns:
        wx = _weather_by_team(schedule, season, week)
        if len(wx):
            out["roof"] = out[team_col].map(wx["roof"])
            out["wind"] = pd.to_numeric(out[team_col].map(wx["wind"]), errors="coerce")
    out["wind_points"] = [wind_adjustment(p, w) for p, w
                          in zip(out["position"], out["wind"])]

    out["why"] = [_notes(r) for _, r in out.iterrows()]
    return out


def _trailing_form(pool, history, season, week, trailing=5):
    """Mean actual FanDuel points over each player's last `trailing` games."""
    from .scoring import fanduel_points

    h = history.copy()
    if season is not None and "season" in h.columns:
        wk = pd.to_numeric(h.get("week"), errors="coerce")
        h = h[(h["season"] < season) | ((h["season"] == season) &
                                        (wk < (week if week is not None else 99)))]
    if "fanduel_points" in h.columns:
        pts = pd.to_numeric(h["fanduel_points"], errors="coerce")
    elif "actual" in h.columns:
        pts = pd.to_numeric(h["actual"], errors="coerce")
    else:
        pts = pd.Series(fanduel_points(
            passing_yards=h.get("passing_yards"), passing_tds=h.get("passing_tds"),
            interceptions=h.get("passing_interceptions", h.get("interceptions")),
            rushing_yards=h.get("rushing_yards"), rushing_tds=h.get("rushing_tds"),
            receptions=h.get("receptions"), receiving_yards=h.get("receiving_yards"),
            receiving_tds=h.get("receiving_tds")), index=h.index)
    order = [c for c in ("season", "week") if c in h.columns]
    h = h.assign(_pts=pts)
    if order:
        h = h.sort_values(order)
    form = h.groupby("player_id")["_pts"].apply(lambda s: s.tail(trailing).mean())
    return pool["player_id"].map(form)


def _weather_by_team(schedule, season, week):
    """roof/wind per team for one week, both sides of every game."""
    s = schedule
    if season is not None and "season" in s.columns:
        s = s[s["season"] == season]
    if week is not None and "week" in s.columns:
        s = s[s["week"] == week]
    if not len(s):
        return pd.DataFrame(columns=["roof", "wind"])
    frames = []
    for side in ("home_team", "away_team"):
        if side not in s.columns:
            continue
        frames.append(s[[side, "roof", "wind"]].rename(columns={side: "team"})
                      if "wind" in s.columns else
                      s[[side, "roof"]].rename(columns={side: "team"}).assign(wind=np.nan))
    if not frames:
        return pd.DataFrame(columns=["roof", "wind"])
    wx = pd.concat(frames, ignore_index=True).dropna(subset=["team"])
    return wx.drop_duplicates("team").set_index("team")[["roof", "wind"]]


def card(row):
    """One player's vetting card as text."""
    name = row.get("player_name") or row.get("Nickname") or "?"
    pos = str(row.get("position") or "")
    parts = [f"{name} ({pos}, {row.get('team') or '?'})"]
    proj = row.get("projected")
    if np.isfinite(proj or np.nan):
        typ, up = row.get("typical_week"), row.get("upside")
        line = f"  projection {proj:.1f}"
        if np.isfinite(typ or np.nan):
            line += f"   typical {typ:.1f}   upside {up:.1f}"
        parts.append(line)
    opp = row.get("opp_per_game")
    if np.isfinite(opp or np.nan):
        share = row.get("opp_share")
        extra = f" ({share:.0%} of his team's)" if np.isfinite(share or np.nan) else ""
        parts.append(f"  {opp:.1f} opportunities/game{extra}")
    why = row.get("why")
    if why:
        parts.append(f"  {why}")
    return "\n".join(parts)


def vet(lineup, **kwargs):
    """`explain` plus a printable report for a built lineup."""
    out = explain(lineup, **kwargs)
    lines = [card(r) for _, r in out.iterrows()]
    total = pd.to_numeric(out["projected"], errors="coerce").sum()
    wind = pd.to_numeric(out["wind_points"], errors="coerce").sum()
    # Deliberately NOT the sum of the typical weeks. Each player's median sits
    # below his mean because his own score is right-skewed, but nine of them
    # added together are very nearly symmetric -- the skew cancels -- so the
    # lineup's median is about its projection, not the sum of nine medians.
    # Summing them understates a real lineup by roughly twenty points.
    lines.append(f"\nlineup projection {total:.1f} (also about its median -- "
                 f"the per-player skew cancels across nine)")
    if wind:
        lines.append(f"unpriced wind exposure {wind:+.1f}")
    return out, "\n".join(lines)
