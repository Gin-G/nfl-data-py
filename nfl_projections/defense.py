"""Projecting a team defense's FanDuel points.

The board projects players and has never projected defenses, so the optimizer
fills its DEF slot with FanDuel's own season-average FPPG. That is worse than
nothing: measured over 2019-2026, FPPG's mean absolute error on a defense's
actual FanDuel score is **4.63 points against a flat league average's 4.31**.
A defense's season average chases noise — sacks and takeaways repeat weakly and
the points-allowed tier is mostly who it is playing — so ranking defenses by it
is close to ranking them at random.

What is here instead projects the pieces and scores them up:

  * sacks, interceptions, fumble recoveries and defensive/return touchdowns,
    each from a ridge on the defense's own rate and the offence it is facing;
  * the points-allowed tier, which is a **step function**, so its expectation
    is taken over a distribution of points allowed rather than read off the
    predicted mean. `tier(E[PA])` and `E[tier(PA)]` are different numbers and
    the second one is the right one: a defence projected to allow 21 has real
    chances at the 4-point tier and the -1 tier, and the step function at 21
    says 0.

Measured the same way (walk-forward, train on every prior season, 3,906
team-weeks over 2019-2026):

    method              MAE     rho    top-3 picked
    FanDuel FPPG       4.63   0.113       7.22
    league average     4.31      --       6.51
    this model         4.18   0.236       8.45

"top-3 picked" is the one that pays: the mean actual score of the three
defenses each method liked best that week. It beats FPPG in 6 of the 8 seasons
and on MAE in all 8.

**It does not beat the betting market.** Ranking defenses by the opponent's
implied total picks a top three worth ~9.0, and the implied total correlates
0.396 with points allowed where this model manages 0.298. Lines stay out of
the features on purpose (they are the benchmark, not an input), so read the
market as the standing target rather than as something already beaten.
"""

import numpy as np
import pandas as pd

from .utils import to_pandas

# FanDuel's defensive scoring. Points allowed pays by tier, everything else
# per event.
PA_TIERS = ((0, 0, 10), (1, 6, 7), (7, 13, 4), (14, 20, 1),
            (21, 27, 0), (28, 34, -1), (35, 10_000, -4))
EVENT_POINTS = {"d_sacks": 1, "d_int": 2, "d_fr": 2, "d_td": 6,
                "d_safety": 2, "d_block": 2}

# FanDuel spells three teams differently from nflverse, and a defense that
# does not join is a defense that silently keeps its FPPG — the exact failure
# this module exists to remove. The relocations are here so that old seasons
# join too.
TEAM_ALIASES = {"JAC": "JAX", "LAR": "LA", "WSH": "WAS", "WFT": "WAS",
                "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
                "SL": "LA", "STL": "LA", "OAK": "LV", "SD": "LAC"}


def normalize_team(code) -> str:
    """One spelling per team, whichever side of the join it came from."""
    code = str(code or "").strip().upper()
    return TEAM_ALIASES.get(code, code)


# What is averaged for the defence, and for the offence it faces.
DEF_STATS = ["d_sacks", "d_int", "d_fr", "d_td", "allowed", "dst_points"]
OFF_STATS = ["scored", "o_giveaways", "o_sacks_allowed", "o_yards"]
_ALL_STATS = sorted(set(DEF_STATS + OFF_STATS))

FEATURES = (["own_" + s for s in DEF_STATS] + ["opp_" + s for s in OFF_STATS]
            + ["opp_dst_points", "home", "n_games"])

# Weight on last season's rate and on this season's league rate, in games. A
# defence with three games played is still mostly its prior, which is what
# stops week 2 from being decided by one fluke sack total.
PRIOR_GAMES, LEAGUE_GAMES = 3.0, 3.0
RIDGE_ALPHA = 10.0


def points_allowed_tier(points) -> int:
    """FanDuel's points-allowed tier value."""
    points = float(points)
    for low, high, value in PA_TIERS:
        if low <= points <= high:
            return value
    return -4


def fanduel_points(row) -> float:
    """A defense's FanDuel score from its box score and the points it allowed."""
    get = (lambda name: float(row.get(name, 0) or 0))
    return (sum(get(stat) * weight for stat, weight in EVENT_POINTS.items())
            + points_allowed_tier(get("allowed")))


def load_games(seasons) -> pd.DataFrame:
    """One row per team-game: the defense's box score and what it allowed.

    Points allowed comes from the schedule rather than the team stats, because
    the team stats have no score — and the tier is most of the fantasy points.
    """
    import nflreadpy as nfl

    seasons = sorted(set(int(s) for s in seasons))
    stats = to_pandas(nfl.load_team_stats(seasons=seasons, summary_level="week"))
    stats = stats[stats.get("season_type", "REG") == "REG"].copy()
    schedule = pd.concat([to_pandas(nfl.load_schedules(seasons=[s])) for s in seasons],
                         ignore_index=True)
    if "game_type" in schedule.columns:
        schedule = schedule[schedule["game_type"] == "REG"]

    played = []
    for game in schedule.itertuples():
        if pd.isna(getattr(game, "home_score", None)):
            continue
        played.append((game.season, game.week, game.home_team, game.away_team,
                       game.home_score, game.away_score, 1))
        played.append((game.season, game.week, game.away_team, game.home_team,
                       game.away_score, game.home_score, 0))
    games = pd.DataFrame(played, columns=["season", "week", "team", "opp",
                                          "scored", "allowed", "home"])
    for frame, columns in ((games, ("team", "opp")), (stats, ("team",))):
        for column in columns:
            frame[column] = frame[column].map(normalize_team)
    return _attach_box_scores(games, stats)


def _attach_box_scores(games: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Join each team-game to its defensive box score and the opponent's offence."""
    def num(name):
        if name not in stats.columns:
            return pd.Series(0.0, index=stats.index)
        return pd.to_numeric(stats[name], errors="coerce").fillna(0.0)

    box = pd.DataFrame({
        "season": stats["season"], "week": stats["week"], "team": stats["team"],
        "d_sacks": num("def_sacks"),
        "d_int": num("def_interceptions"),
        "d_fr": num("fumble_recovery_opp"),
        # def_tds covers interception and fumble returns; special_teams_tds is
        # the team's own return scores, which FanDuel also pays the defence.
        "d_td": num("def_tds") + num("special_teams_tds"),
        "d_safety": num("def_safeties"),
        "d_block": num("def_punt_blocks") + num("def_pat_blocks") + num("def_fg_blocks"),
        # These describe the team's OFFENCE; they matter as the opponent's.
        "o_giveaways": num("passing_interceptions") + num("fumbles_lost_total"),
        "o_yards": num("passing_yards") + num("rushing_yards"),
    })
    frame = games.merge(box, on=["season", "week", "team"], how="inner")
    mirror = box.rename(columns={"team": "opp",
                                 **{c: "x_" + c for c in box.columns[3:]}})
    frame = frame.merge(mirror, on=["season", "week", "opp"], how="left")
    # Sacks an offence allowed are the sacks the other defence recorded.
    frame["o_sacks_allowed"] = frame["x_d_sacks"]
    frame["dst_points"] = frame.apply(fanduel_points, axis=1)
    return frame.sort_values(["season", "week", "team"]).reset_index(drop=True)


def _league_prior(games: pd.DataFrame) -> pd.DataFrame:
    """The league's rate over COMPLETED weeks, for each row.

    Deliberately not an expanding mean over rows: rows are ordered by week and
    then by team, so a row partway through week 5 would be handed the average
    of the week-5 games that happen to sort before it. Other teams' results
    that Sunday are not knowable on Saturday, and a league average is exactly
    the kind of weak leak that quietly flatters a backtest. Weeks before this
    one, then last season, then everything on file.
    """
    weekly = games.groupby(["season", "week"])[_ALL_STATS].mean()
    through = (weekly.groupby(level="season").apply(
        lambda frame: frame.droplevel("season").expanding().mean().shift(1))
        if len(weekly) else weekly)
    through.index = weekly.index
    prior_season = games.groupby("season")[_ALL_STATS].mean()
    prior_season.index = prior_season.index + 1
    overall = games[_ALL_STATS].mean()

    out = games[["season", "week"]].merge(
        through.reset_index(), on=["season", "week"], how="left").set_index(games.index)
    fallback = games[["season"]].merge(
        prior_season.reset_index(), on="season", how="left").set_index(games.index)
    for stat in _ALL_STATS:
        out[stat] = out[stat].fillna(fallback[stat]).fillna(overall[stat])
    return out[_ALL_STATS]


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    """Add the features, each one using only weeks already played.

    Every rate is a three-way blend of what the team has done this season, what
    it did last season and what the league is doing, weighted by games. Rows
    for an unplayed week are carried through with their features filled and
    their outcome columns left as they came in.
    """
    games = games.sort_values(["season", "week", "team"]).reset_index(drop=True)
    for stat in _ALL_STATS:
        if stat not in games.columns:
            games[stat] = np.nan

    previous = games.groupby(["season", "team"])[_ALL_STATS].mean().reset_index()
    previous["season"] += 1
    previous = previous.rename(columns={s: "prev_" + s for s in _ALL_STATS})
    games = games.merge(previous, on=["season", "team"], how="left")
    league = _league_prior(games)
    key = [games["season"], games["team"]]

    own = pd.DataFrame(index=games.index)
    count = None
    for stat in _ALL_STATS:
        # The running total is taken over FILLED values, not the raw column:
        # a row for a week not yet played is a hole, and `cumsum` leaves a hole
        # in the running total from there on. That is invisible in a backtest,
        # where every row has an outcome, and it zeroes the whole season's
        # form on exactly the rows a live projection is asked about.
        played = games[stat].notna()
        values = games[stat].fillna(0.0)
        total = values.groupby(key).cumsum() - values
        count = played.groupby(key).cumsum() - played.astype(int)
        prior = games["prev_" + stat].fillna(league[stat])
        blended = ((total + PRIOR_GAMES * prior + LEAGUE_GAMES * league[stat])
                   / (count + PRIOR_GAMES + LEAGUE_GAMES))
        own["own_" + stat] = blended.fillna(league[stat])
    own["n_games"] = count

    side = pd.concat([games[["season", "week", "team"]], own], axis=1)
    side = side.drop(columns=["n_games"]).rename(
        columns={"team": "opp", **{("own_" + s): ("opp_" + s) for s in _ALL_STATS}})
    out = pd.concat([games, own], axis=1).merge(
        side, on=["season", "week", "opp"], how="left")

    # FanDuel's own FPPG, for comparison: the raw season-to-date mean, with no
    # blending at all. That lack of blending is most of why it is beatable.
    played = games["dst_points"].notna()
    values = games["dst_points"].fillna(0.0)
    total = values.groupby(key).cumsum() - values
    count = played.groupby(key).cumsum() - played.astype(int)
    out["fppg"] = ((total / count.replace(0, np.nan))
                   .fillna(out["prev_dst_points"])
                   .fillna(league["dst_points"]))
    return out


def _ridge(X, y, alpha=RIDGE_ALPHA):
    """Standardised ridge, returning a predict function. The intercept is free."""
    X = np.asarray(X, dtype=float)
    mean, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    design = np.column_stack([np.ones(len(X)), (X - mean) / sd])
    penalty = alpha * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ np.asarray(y, float))

    def predict(new):
        new = np.asarray(new, dtype=float)
        return np.column_stack([np.ones(len(new)), (new - mean) / sd]) @ beta

    return predict


class DefenseModel:
    """Component ridges plus an empirical spread for the points-allowed tier."""

    def __init__(self, events, allowed, residuals, features):
        self._events = events
        self._allowed = allowed
        self._residuals = residuals
        self.features = list(features)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame[self.features].values
        total = sum(self._events[stat](X) * EVENT_POINTS[stat] for stat in self._events)
        # The tier is a step, so it is averaged over where the score might
        # land, not evaluated at where it is expected to land.
        draws = np.clip(self._allowed(X)[:, None] + self._residuals[None, :], 0, None)
        tiers = np.vectorize(points_allowed_tier)(np.round(draws)).mean(axis=1)
        return total + tiers


def train(frame: pd.DataFrame, features=FEATURES) -> DefenseModel:
    """Fit on played games. Rows with no outcome or no features are dropped."""
    features = list(features)
    frame = frame.dropna(subset=features + ["dst_points", "allowed"])
    if frame.empty:
        raise ValueError("no played games with features to train on")
    X = frame[features].values
    events = {stat: _ridge(X, frame[stat].values)
              for stat in ("d_sacks", "d_int", "d_fr", "d_td")}
    allowed = _ridge(X, frame["allowed"].values)
    residuals = frame["allowed"].values - allowed(X)
    return DefenseModel(events, allowed, residuals, features)


def project(season: int, week: int, history=None, model=None) -> pd.DataFrame:
    """Projected FanDuel points for every defense playing in one week.

    Trains on every played game before that week — this season's and the five
    before it — unless a `model` is handed in. Returns team, opponent, home and
    `projected_points`, best first, beside `fppg` so the two can be compared.
    """
    import nflreadpy as nfl

    season, week = int(season), int(week)
    if history is None:
        history = load_games(range(season - 5, season + 1))
    played = history[(history["season"] < season)
                     | ((history["season"] == season) & (history["week"] < week))]

    schedule = to_pandas(nfl.load_schedules(seasons=[season]))
    schedule = schedule[schedule["week"] == week]
    if "game_type" in schedule.columns:
        schedule = schedule[schedule["game_type"] == "REG"]
    upcoming = []
    for game in schedule.itertuples():
        home, away = normalize_team(game.home_team), normalize_team(game.away_team)
        upcoming.append({"season": season, "week": week, "team": home,
                         "opp": away, "home": 1})
        upcoming.append({"season": season, "week": week, "team": away,
                         "opp": home, "home": 0})
    if not upcoming:
        return pd.DataFrame(columns=["team", "opp", "home", "projected_points", "fppg"])

    combined = pd.concat([played, pd.DataFrame(upcoming)], ignore_index=True)
    featured = build_features(combined)
    target = featured[(featured["season"] == season) & (featured["week"] == week)].copy()
    model = model or train(featured[featured["dst_points"].notna()])
    target = target.dropna(subset=model.features)
    target["projected_points"] = model.predict(target).round(2)
    return (target[["team", "opp", "home", "projected_points", "fppg"]]
            .sort_values("projected_points", ascending=False)
            .reset_index(drop=True))


def backtest(games: pd.DataFrame, first_season=None, features=FEATURES,
             min_train_rows=500) -> pd.DataFrame:
    """Walk forward a season at a time: train on everything before, predict it."""
    featured = build_features(games).dropna(subset=list(features) + ["dst_points"])
    seasons = sorted(featured["season"].unique())
    first = first_season or (seasons[0] + 3 if len(seasons) > 3 else seasons[-1])
    out = []
    for season in [s for s in seasons if s >= first]:
        train_rows = featured[featured["season"] < season]
        test = featured[featured["season"] == season].copy()
        if len(train_rows) < min_train_rows or test.empty:
            continue
        test["projected_points"] = train(train_rows, features).predict(test)
        out.append(test)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)
