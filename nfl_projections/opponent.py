"""Opponent defensive-matchup features.

For a next-week projection model, the matchup that matters is the *upcoming*
opponent's defense, described only by games played *before* the week being
projected. This module builds, for every (season, week, team, position), the
opponent that team faces and that opponent's rolling defensive form entering
that week, plus a home/away flag.

The same per-(season, week, team, position) table is joined two ways:

- training: onto each game row's *next* game (the game whose stats are the
  target), so the feature describes the defense the target was scored against.
- prediction / backtest: onto the (season, week, team) being projected.

Because every stat is aggregated strictly from weeks earlier than the target
week, the features are leakage-free.
"""

import logging

import numpy as np
import pandas as pd

from .utils import regular_games, to_pandas

logger = logging.getLogger(__name__)

from .pbp import SCHEME_DEFENSE_FEATURES, SCHEME_DEFENSE_NEUTRAL

# Coarse "points/yards allowed by position" opponent columns
COARSE_OPPONENT_FEATURES = [
    "opp_fppg_allowed",
    "opp_defensive_rank",
    "opp_yards_allowed_pg",
    "opp_tds_allowed_pg",
    "opp_turnovers_forced_pg",
    "opp_receptions_allowed_pg",
    "is_home_game",
]

# Every possible opponent-side column (coarse + pbp scheme splits). Downstream
# code uses this only to (a) neutral-fill and (b) recognize opponent columns;
# a model trained without a given column simply never references it.
OPPONENT_FEATURES = COARSE_OPPONENT_FEATURES + SCHEME_DEFENSE_FEATURES

# Neutral fallbacks for matchups with no prior-week data (e.g. week 1)
NEUTRAL_VALUES = {
    "opp_fppg_allowed": 15.0,
    "opp_defensive_rank": 16.0,
    "opp_yards_allowed_pg": 100.0,
    "opp_tds_allowed_pg": 0.5,
    "opp_turnovers_forced_pg": 0.5,
    "opp_receptions_allowed_pg": 5.0,
    "is_home_game": 0.0,
    **SCHEME_DEFENSE_NEUTRAL,
}

POSITIONS = ["QB", "RB", "WR", "TE"]

# Per-position raw stat aggregates summed each week, then rolled
_SUM_COLS = {
    "fp": "fanduel_fantasy_points",
    "pass_yds": "passing_yards",
    "rush_yds": "rushing_yards",
    "rec_yds": "receiving_yards",
    "pass_td": "passing_tds",
    "rush_td": "rushing_tds",
    "rec_td": "receiving_tds",
    "rec": "receptions",
    "intc": "passing_interceptions",
}


def _position_series(df):
    if "position_x" in df.columns:
        return df["position_x"]
    if "position" in df.columns:
        return df["position"]
    raise KeyError("dataset has no position/position_x column")


def build_defense_form(dataset, window=6):
    """Rolling per-position defensive stats *allowed* entering each week.

    Returns a DataFrame keyed by (season, week, defense_team, position) whose
    stats summarize the games that defense allowed over the ``window`` weeks
    immediately before ``week`` (never including ``week`` itself). Weeks with no
    prior games in the window are absent (callers fall back to neutral values).
    """
    games = regular_games(dataset).copy()
    games["position"] = _position_series(games).values
    games = games[games["position"].isin(POSITIONS)]
    games["opponent_team"] = games["opponent_team"].replace("", np.nan)
    games = games.dropna(subset=["opponent_team", "season", "week"]).reset_index(drop=True)

    def num(col):
        """Numeric column as a zero-filled Series, or zeros if absent."""
        if col in games.columns:
            return pd.to_numeric(games[col], errors="coerce").fillna(0)
        return pd.Series(0.0, index=games.index)

    # Fumbles forced = rushing + receiving fumbles by the offense
    fum = num("rushing_fumbles") + num("receiving_fumbles")
    agg = pd.DataFrame({
        "season": games["season"].values,
        "week": games["week"].astype(int).values,
        "defense_team": games["opponent_team"].values,
        "position": games["position"].values,
        "n": 1.0,
        "fum": fum.values,
    })
    for short, col in _SUM_COLS.items():
        agg[short] = num(col).values

    # Sum per defense-position-week, then roll the window over calendar weeks
    weekly = agg.groupby(["season", "defense_team", "position", "week"]).sum().reset_index()

    rolled = []
    value_cols = ["n", "fum", *_SUM_COLS.keys()]
    for (season, team, position), grp in weekly.groupby(["season", "defense_team", "position"]):
        vals = grp.set_index("week")[value_cols].sort_index()
        full = vals.reindex(range(1, int(vals.index.max()) + 1), fill_value=0.0)
        # Sum of the `window` weeks strictly before each week
        windowed = full.rolling(window, min_periods=1).sum().shift(1)
        windowed = windowed.dropna(how="all")
        windowed = windowed[windowed["n"] > 0]
        if windowed.empty:
            continue
        windowed = windowed.assign(season=season, defense_team=team, position=position)
        rolled.append(windowed.reset_index())

    if not rolled:
        return pd.DataFrame(
            columns=["season", "week", "defense_team", "position", *COARSE_OPPONENT_FEATURES[:-1]]
        )

    form = pd.concat(rolled, ignore_index=True)
    n = form["n"]

    pos = form["position"]
    is_qb = pos == "QB"
    is_rb = pos == "RB"
    is_wrte = pos.isin(["WR", "TE"])

    form["opp_fppg_allowed"] = form["fp"] / n
    form["opp_yards_allowed_pg"] = np.select(
        [is_qb, is_rb, is_wrte],
        [form["pass_yds"] / n, form["rush_yds"] / n, form["rec_yds"] / n],
        default=0.0,
    )
    form["opp_tds_allowed_pg"] = np.select(
        [is_qb, is_rb, is_wrte],
        [form["pass_td"] / n, form["rush_td"] / n, form["rec_td"] / n],
        default=0.0,
    )
    form["opp_turnovers_forced_pg"] = np.where(is_qb, form["intc"] / n, form["fum"] / n)
    form["opp_receptions_allowed_pg"] = np.where(is_qb, 0.0, form["rec"] / n)

    form = _add_defensive_rank(form)
    keep = ["season", "week", "defense_team", "position", *COARSE_OPPONENT_FEATURES[:-1]]
    return form[keep]


def _add_defensive_rank(form):
    """Rank 1 (fewest fantasy points allowed) .. 32 within season/week/position."""
    form["opp_defensive_rank"] = (
        form.groupby(["season", "week", "position"])["opp_fppg_allowed"]
        .rank(method="min", ascending=True)
    )
    return form


def build_schedule_map(seasons, schedule=None):
    """One row per (season, week, team) with its opponent and home/away flag."""
    if schedule is None:
        import nflreadpy as nfl

        schedule = to_pandas(nfl.load_schedules(seasons=list(seasons)))
    schedule = schedule[schedule["game_type"] == "REG"]

    home = pd.DataFrame({
        "season": schedule["season"],
        "week": pd.to_numeric(schedule["week"], errors="coerce"),
        "team": schedule["home_team"],
        "opponent": schedule["away_team"],
        "is_home_game": 1.0,
    })
    away = pd.DataFrame({
        "season": schedule["season"],
        "week": pd.to_numeric(schedule["week"], errors="coerce"),
        "team": schedule["away_team"],
        "opponent": schedule["home_team"],
        "is_home_game": 0.0,
    })
    out = pd.concat([home, away], ignore_index=True).dropna(subset=["week"])
    out["week"] = out["week"].astype(int)
    return out


def build_matchup_table(dataset, schedule_map, window=6, scheme_form=None,
                        include_coarse=True):
    """Per-(season, week, team, position) opponent features for the upcoming game.

    Joins each team's scheduled opponent to that opponent's defensive form
    entering the week. This single table is joined onto next-game keys for
    training and onto the projected week for prediction/backtesting.

    Args:
        include_coarse: attach the coarse points/yards-allowed-by-position form.
        scheme_form: optional pbp scheme-split defense form
            (pbp.build_scheme_defense_form); attached defense-wide.
    """
    # One row per (season, week, team, position) so every slot exists even when
    # the opponent has no prior-week form yet (filled neutral downstream).
    expanded = schedule_map.loc[schedule_map.index.repeat(len(POSITIONS))].copy()
    expanded["position"] = np.tile(POSITIONS, len(schedule_map))
    matchup = expanded

    if include_coarse:
        form = build_defense_form(dataset, window=window)
        matchup = pd.merge(
            matchup, form,
            left_on=["season", "week", "opponent", "position"],
            right_on=["season", "week", "defense_team", "position"],
            how="left",
        ).drop(columns=["defense_team"], errors="ignore")

    if scheme_form is not None:
        # Scheme splits are defense-wide (not per position): merge on team/week
        matchup = pd.merge(
            matchup, scheme_form,
            left_on=["season", "week", "opponent"],
            right_on=["season", "week", "defense_team"],
            how="left",
        ).drop(columns=["defense_team"], errors="ignore")

    return matchup


def _fill_neutral(df):
    for col, val in NEUTRAL_VALUES.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    return df


def add_next_game_keys(df):
    """Attach each row's *next* game keys (team/week it will be scored against).

    Mirrors the next-week target shift: within player-season, the next row is
    the game whose stats become the training target.
    """
    out = df.sort_values(["player_id", "season", "week"]).copy()
    team_col = "team" if "team" in out.columns else "recent_team"
    grp = out.groupby(["player_id", "season"])
    out["next_game_week"] = grp["week"].shift(-1)
    if team_col in out.columns:
        out["next_game_team"] = grp[team_col].shift(-1)
    else:
        out["next_game_team"] = np.nan
    return out


def attach_training_features(df, matchup_table):
    """Merge upcoming-opponent features onto training rows via next-game keys.

    ``df`` must carry next_game_week / next_game_team (see add_next_game_keys)
    and a position column. Rows without a match get neutral values.
    """
    position = _position_series(df)
    keyed = df.copy()
    keyed["_position"] = position.values
    keyed["_next_week"] = pd.to_numeric(keyed["next_game_week"], errors="coerce")

    opp_cols = [c for c in OPPONENT_FEATURES if c in matchup_table.columns]
    feats = matchup_table.rename(
        columns={"week": "_next_week", "team": "next_game_team", "position": "_position"}
    )[["season", "_next_week", "next_game_team", "_position", *opp_cols]]

    merged = pd.merge(
        keyed, feats,
        on=["season", "_next_week", "next_game_team", "_position"],
        how="left",
    )
    merged = merged.drop(columns=["_position", "_next_week"])
    return _fill_neutral(merged)


def lookup_week_features(matchup_table, season, week):
    """Per-(team, position) opponent features for a single projected week.

    Returns a frame indexed by (team, position) so the projector can look up a
    player's matchup by their team and position.
    """
    wk = matchup_table[
        (matchup_table["season"] == season) & (matchup_table["week"] == int(week))
    ].copy()
    wk = _fill_neutral(wk)
    return wk.set_index(["team", "position"])
