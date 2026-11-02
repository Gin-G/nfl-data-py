"""Matchup signals, recorded before kickoff so they can be judged later.

This module exists because of a specific failure mode. Every matchup idea in
EXPERIMENTS.md was tested by fitting it to history and reading a correlation,
and almost all of them came back null -- but the few that did not came back
*borderline*, at which point there is no honest way to tell a real effect from
the best of sixteen looks at the same data. The trench report is exactly there:

    RBs projected >= 8, 2026 weeks 3-8, n=134
      correlation with the board's error   +0.166
      bootstrap 95% CI                     [-0.002, +0.324]
      weeks 3-5 / weeks 6-8                +0.263 / +0.099
      "lean up or better" vs the rest      +5.10 against +0.13 (n=25)
      weeks pointing the right way         5 of 5 testable

A sign test on 5 of 5 is about p=0.03, the effect is mechanically the same
story as the running-back/defence pairing, and the 58 threshold is the trench
report's own band rather than one fitted here. All of which makes it the
strongest matchup result on file, and still not enough: 25 treated players, a
confidence interval touching zero, and an effect that halves across the split.

So nothing here adjusts a projection. `snapshot` freezes what the signals said
before a slate; scoring it afterwards accumulates the only sample that can
settle this -- one nobody has looked at yet. Weeks where the answer is "no"
count as much as the others.

The coverage side is already answered and the answer is no. Per-player it
failed three independent ways (team yards allowed, per-corner grades, the
board's own residual; wide receiver was actively harmful). Tested a fourth way
here, at the GAME level -- does a weak secondary flag a game that beats the
board, which is what stacking actually bets on -- it is -0.125 across 87 games,
the wrong sign, where significance needs about 0.21. It is carried in the
snapshot as context so that conclusion keeps being re-checked, not because it
is expected to change.
"""

import numpy as np
import pandas as pd

# The trench report's own bands, not thresholds fitted here.
RUN_BANDS = ((0, 30, "fade"), (30, 42, "lean down"), (42, 58, "neutral"),
             (58, 70, "lean up"), (70, 101, "smash"))
LEAN_UP = 58.0

# Only players a lineup would actually consider. Below this the board's error
# is dominated by whether the man played at all.
ROSTERABLE = 8.0

COLUMNS = ["season", "week", "player_id", "player_name", "position", "team",
           "opponent", "projected", "salary", "value", "run_matchup",
           "pass_matchup", "opp_cover_grade", "run_lean", "rosterable"]


def band(rating, bands=RUN_BANDS):
    """The label for a 0-100 matchup rating."""
    if rating is None or (isinstance(rating, float) and np.isnan(rating)):
        return "unknown"
    for low, high, label in bands:
        if low <= rating < high:
            return label
    return "unknown"


def snapshot(projections, trenches=None, coverage=None, *,
             salaries=None, rosterable=ROSTERABLE):
    """What the matchup signals said about a week, before it was played.

    `projections` needs season/week/player_id/player_name/position/team and a
    `projected` column. `trenches` is one row per team with run_matchup and
    pass_matchup; `coverage` is one row per team with opp_cover_grade. Either
    may be None or empty -- the column comes back as NaN rather than the
    snapshot failing, because a signal that could not be computed this week is
    a fact about the week and not a reason to record nothing.

    `salaries` maps a normalized player name to a FanDuel salary. It is here
    because of a specific failure: the matchup effect was measured between
    players within 1.5 PROJECTED points of each other, where it won 63 of 90
    pairs, and then did nothing when wired into the builder -- which decides in
    VALUE space, so its near-ties are players with similar points per dollar
    rather than similar points. Re-asking the question in value space is the
    test that matters, and without a salary it cannot be asked at all. Only two
    weeks of 2026 had a stored slate to recover salaries from, and on those the
    swap lost about five points a lineup, which is inside one lineup's noise.

    No projection is adjusted. This is a record, not a model.
    """
    frame = pd.DataFrame(projections).copy()
    for column in ("season", "week", "player_id", "player_name", "position",
                   "team", "projected"):
        if column not in frame.columns:
            raise ValueError(f"projections needs a {column!r} column")

    frame["team"] = frame["team"].astype(str).str.upper()
    out = frame[["season", "week", "player_id", "player_name", "position",
                 "team", "projected"]].copy()
    out["projected"] = pd.to_numeric(out["projected"], errors="coerce")

    merged = _lookup(trenches, ("run_matchup", "pass_matchup", "opponent"))
    cover = _lookup(coverage, ("opp_cover_grade",))
    for column in ("run_matchup", "pass_matchup", "opponent"):
        out[column] = out["team"].map(merged.get(column, {}))
    out["opp_cover_grade"] = out["team"].map(cover.get("opp_cover_grade", {}))

    if salaries:
        from .optimizer import normalize_name

        lookup = {normalize_name(str(k)): v for k, v in dict(salaries).items()}
        out["salary"] = [lookup.get(normalize_name(str(n)))
                         for n in out["player_name"]]
    else:
        out["salary"] = np.nan
    out["salary"] = pd.to_numeric(out["salary"], errors="coerce")
    # Points per $1,000: the units the builder actually compares.
    out["value"] = np.where(out["salary"] > 0,
                            out["projected"] / out["salary"] * 1000.0, np.nan)

    out["run_lean"] = [band(v) for v in out["run_matchup"]]
    out["rosterable"] = out["projected"] >= float(rosterable)
    return out[COLUMNS].reset_index(drop=True)


def _lookup(frame, columns):
    """{column: {team: value}} from a per-team frame, tolerating None/empty."""
    if frame is None or len(frame) == 0:
        return {}
    frame = pd.DataFrame(frame).copy()
    if "team" not in frame.columns:
        return {}
    frame["team"] = frame["team"].astype(str).str.upper()
    indexed = frame.drop_duplicates("team").set_index("team")
    return {c: indexed[c].to_dict() for c in columns if c in indexed.columns}


def score(snapshots, actuals):
    """Attach outcomes to a stack of snapshots.

    `actuals` maps player_id -> points scored. A player with no entry did not
    record a stat line, which for this purpose is unknown rather than zero:
    the question is whether the signal predicts PERFORMANCE, and a scratch
    answers a different question (and one the ownership list answers better).
    """
    out = pd.DataFrame(snapshots).copy()
    out["actual"] = out["player_id"].map(dict(actuals))
    out["miss"] = out["actual"] - out["projected"]
    return out


def paired_verdict(scored, *, position="RB", signal="run_matchup",
                   threshold=LEAN_UP, match_on="value", tolerance=0.15):
    """The signal judged between players the builder would call near-tied.

    `match_on` is "value" (points per $1,000, the builder's own units) or
    "projected". The projection-space version is what gave +6.97 a pair; the
    value-space version is the one that corresponds to a decision the builder
    can actually be talked out of, and is the number to watch.
    """
    import itertools

    rows = scored[(scored["position"] == position) & scored["rosterable"]
                  & scored["miss"].notna() & scored[signal].notna()
                  & scored[match_on].notna()]
    wins = losses = 0
    margins = []
    for _, group in rows.groupby(["season", "week"]):
        records = list(group.itertuples())
        for a, b in itertools.combinations(records, 2):
            if abs(getattr(a, match_on) - getattr(b, match_on)) > tolerance:
                continue
            up, down = (a, b) if getattr(a, signal) >= threshold else (b, a)
            if not (getattr(up, signal) >= threshold > getattr(down, signal)):
                continue
            margins.append(float(up.actual - down.actual))
            wins += up.actual > down.actual
            losses += up.actual < down.actual
    if not margins:
        return {"pairs": 0, "verdict": "no comparable pairs recorded yet"}
    return {"pairs": len(margins), "wins": wins, "losses": losses,
            "win_rate": round(wins / max(wins + losses, 1), 3),
            "mean_margin": round(float(np.mean(margins)), 2),
            "match_on": match_on, "tolerance": tolerance,
            "caveat": ("Pairs share players, so they are not independent and "
                       "any interval over them is optimistic.")}


def verdict(scored, *, position="RB", signal="run_matchup", threshold=LEAN_UP):
    """Has the signal paid, on everything recorded so far?

    Reports the per-week sign as well as the pooled gap, because pooling across
    weeks hides the thing that matters: a signal worth using should point the
    right way most weeks, not win on one slate and average out.
    """
    rows = scored[(scored["position"] == position) & scored["rosterable"]
                  & scored["miss"].notna() & scored[signal].notna()]
    if rows.empty:
        return {"n": 0, "weeks": 0, "verdict": "nothing recorded yet"}
    flagged = rows[rows[signal] >= threshold]
    rest = rows[rows[signal] < threshold]
    per_week, right = [], 0
    for (season, week), group in rows.groupby(["season", "week"]):
        up, down = group[group[signal] >= threshold], group[group[signal] < threshold]
        if up.empty or down.empty:
            continue
        gap = float(up["miss"].mean() - down["miss"].mean())
        per_week.append({"season": int(season), "week": int(week),
                         "flagged": len(up), "gap": round(gap, 2)})
        right += gap > 0
    return {
        "n": len(rows), "flagged": len(flagged), "weeks": len(per_week),
        "weeks_right": right,
        "correlation": (round(float(rows[signal].corr(rows["miss"])), 3)
                        if len(rows) > 2 else None),
        "flagged_miss": round(float(flagged["miss"].mean()), 2) if len(flagged) else None,
        "other_miss": round(float(rest["miss"].mean()), 2) if len(rest) else None,
        "gap": (round(float(flagged["miss"].mean() - rest["miss"].mean()), 2)
                if len(flagged) and len(rest) else None),
        "per_week": per_week,
    }
