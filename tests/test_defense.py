"""The defense projection: scoring, leak-free features, and the tier expectation."""
import numpy as np
import pandas as pd
import pytest

from nfl_projections import defense


TEAMS = tuple(f"T{i:02d}" for i in range(12))


def synthetic_games(seasons=(2020, 2021, 2022, 2023), teams=TEAMS, weeks=14, seed=0):
    """Games where one team's defense is genuinely better than the others'.

    The model has to find that, so the rates differ by team and the outcomes
    are drawn around them rather than being constant.
    """
    rng = np.random.default_rng(seed)
    strength = {team: 0.8 + i * 0.2 for i, team in enumerate(teams)}
    rows = []
    for season in seasons:
        for week in range(1, weeks + 1):
            order = list(teams)
            rng.shuffle(order)
            for home, away in zip(order[::2], order[1::2]):
                for team, opp, at_home in ((home, away, 1), (away, home, 0)):
                    allowed = max(0, int(rng.normal(30 - 4 * strength[team], 7)))
                    rows.append({
                        "season": season, "week": week, "team": team, "opp": opp,
                        "home": at_home, "allowed": allowed,
                        "scored": max(0, int(rng.normal(22, 7))),
                        "d_sacks": rng.poisson(strength[team]),
                        "d_int": rng.poisson(0.3 * strength[team]),
                        "d_fr": rng.poisson(0.2 * strength[team]),
                        "d_td": rng.poisson(0.05),
                        "d_safety": 0, "d_block": 0,
                        "o_giveaways": rng.poisson(1.3),
                        "o_sacks_allowed": rng.poisson(2.3),
                        "o_yards": float(rng.normal(350, 40)),
                    })
    frame = pd.DataFrame(rows)
    frame["dst_points"] = frame.apply(defense.fanduel_points, axis=1)
    return frame


class TestScoring:
    @pytest.mark.parametrize("allowed,value", [
        (0, 10), (3, 7), (6, 7), (7, 4), (13, 4), (14, 1), (20, 1),
        (21, 0), (27, 0), (28, -1), (34, -1), (35, -4), (70, -4)])
    def test_the_points_allowed_tiers(self, allowed, value):
        assert defense.points_allowed_tier(allowed) == value

    def test_a_whole_defensive_line_scores(self):
        # 3 sacks, 2 picks, a fumble, a return score, shutting them out at 6.
        row = {"d_sacks": 3, "d_int": 2, "d_fr": 1, "d_td": 1, "allowed": 6}
        assert defense.fanduel_points(row) == 3 + 4 + 2 + 6 + 7

    def test_a_blowout_loss_can_score_negative(self):
        assert defense.fanduel_points({"allowed": 38}) == -4


class TestFeatures:
    def test_week_one_of_the_first_season_falls_back_to_the_league(self):
        games = synthetic_games()
        featured = defense.build_features(games)
        opening = featured[(featured["week"] == 1)
                           & (featured["season"] == games["season"].min())]
        # Nothing is behind it, so every team gets the same league prior
        # rather than a NaN or a zero.
        assert opening["own_d_sacks"].notna().all()
        assert opening["own_d_sacks"].nunique() == 1

    def test_the_league_prior_does_not_read_the_same_weeks_other_games(self):
        """Other teams' Sunday results are not knowable on Saturday, so a
        league average built by expanding over ROWS is a leak."""
        games = synthetic_games()
        featured = defense.build_features(games)
        louder = games.copy()
        week = (louder["season"] == 2023) & (louder["week"] == 5)
        louder.loc[week, "d_sacks"] += 20
        after = defense.build_features(louder)
        same_week = (after["season"] == 2023) & (after["week"] == 5)
        assert np.allclose(after.loc[same_week, "own_d_sacks"].values,
                           featured.loc[same_week.values, "own_d_sacks"].values)

    def test_doubling_one_teams_outcome_leaves_that_week_alone(self):
        games = synthetic_games()
        featured = defense.build_features(games)
        louder = games.copy()
        mask = (louder["team"] == "T03") & (louder["season"] == 2023) & (louder["week"] == 5)
        louder.loc[mask, "d_sacks"] += 10
        after = defense.build_features(louder)
        row = (after["team"] == "T03") & (after["season"] == 2023) & (after["week"] == 5)
        same = (featured["team"] == "T03") & (featured["season"] == 2023) & (featured["week"] == 5)
        assert np.isclose(after.loc[row, "own_d_sacks"].iloc[0],
                          featured.loc[same, "own_d_sacks"].iloc[0])
        # ... and does reach the week after it, or the feature is inert.
        later = (after["team"] == "T03") & (after["season"] == 2023) & (after["week"] == 6)
        was = (featured["team"] == "T03") & (featured["season"] == 2023) & (featured["week"] == 6)
        assert after.loc[later, "own_d_sacks"].iloc[0] > featured.loc[was, "own_d_sacks"].iloc[0]

    def test_an_unplayed_week_keeps_the_season_behind_it(self):
        """The bug this is here for: a row with no outcome used to zero the
        running total, so a live projection saw a team with no form at all."""
        games = synthetic_games()
        last = int(games["week"].max())
        upcoming = pd.DataFrame([
            {"season": 2023, "week": last + 1, "team": "T03", "opp": "T04", "home": 1},
            {"season": 2023, "week": last + 1, "team": "T04", "opp": "T03", "home": 0}])
        featured = defense.build_features(pd.concat([games, upcoming], ignore_index=True))
        live = featured[(featured["season"] == 2023) & (featured["week"] == last + 1)]
        played = featured[(featured["season"] == 2023) & (featured["week"] == last)
                          & featured["team"].isin(["T03", "T04"])]
        assert live["own_d_sacks"].notna().all()
        # Eight games in, the blend must be near the team's real rate, not zero
        # and not the bare league prior.
        assert live["own_d_sacks"].min() > 0.5
        assert np.isclose(live["n_games"].max(), played["n_games"].max() + 1)
        assert live["fppg"].notna().all()


class TestModel:
    def test_the_tier_is_averaged_over_outcomes_not_read_off_the_mean(self):
        """A defense projected to allow 21 is on the boundary of a step: the
        expectation has to see both sides of it, not score the midpoint."""
        games = synthetic_games()
        featured = defense.build_features(games)
        model = defense.train(featured)
        predictions = model.predict(featured.dropna(subset=model.features))
        # Scoring tier(E[PA]) instead would park a whole band of defenses on
        # exactly the same tier value; the averaged version is continuous.
        assert len(np.unique(np.round(predictions, 3))) > 0.9 * len(predictions)

    def test_it_finds_the_better_defenses(self):
        games = synthetic_games()
        featured = defense.build_features(games)
        model = defense.train(featured[featured["season"] < 2023])
        test = featured[featured["season"] == 2023].dropna(subset=model.features).copy()
        test["pred"] = model.predict(test)
        ranked = test.groupby("team")["pred"].mean()
        # Strength rises with the team's index by construction, so the
        # projection should rank them in roughly that order.
        order = [int(name[1:]) for name in ranked.index]
        rho = np.corrcoef(pd.Series(order).rank(), ranked.rank().values)[0, 1]
        assert rho > 0.8, f"rank correlation with true strength was {rho:.2f}"

    def test_backtest_beats_the_season_average_it_replaces(self):
        games = synthetic_games()
        out = defense.backtest(games, first_season=2022, min_train_rows=100)
        assert not out.empty
        error = lambda col: (out[col] - out["dst_points"]).abs().mean()
        assert error("projected_points") < error("fppg")

    def test_training_on_nothing_is_an_error_not_a_silent_zero(self):
        with pytest.raises(ValueError):
            defense.train(pd.DataFrame(columns=defense.FEATURES + ["dst_points", "allowed"]))


class TestItReachesTheLineup:
    """The point of the model is the DEF slot, which used to take FPPG blind."""

    @staticmethod
    def _slate():
        rows = []

        def add(nick, team, pos, salary, points, fppg=None):
            rows.append({"Nickname": nick, "Team": team, "Roster Position": pos,
                         "Salary": salary, "FPPG": points if fppg is None else fppg,
                         "fanduel_fantasy_points": points, "ceiling": points * 2,
                         "floor": points / 2, "projection_median": points,
                         "Injury Indicator": ""})

        for i, team in enumerate(("AAA", "BBB", "CCC")):
            add(f"QB-{team}", team, "QB", 7000, 18 - i)
            for j in range(3):
                add(f"RB{j}-{team}", team, "RB/FLEX", 6000 - j * 500, 13 - j - i)
                add(f"WR{j}-{team}", team, "WR/FLEX", 6000 - j * 500, 12 - j - i)
            add(f"TE-{team}", team, "TE/FLEX", 4500, 9 - i)
            add(f"TE2-{team}", team, "TE/FLEX", 4000, 7 - i)
        # Two defenses at the same price, and FPPG prefers the wrong one.
        add("D-AAA", "AAA", "DEF", 3500, 0, fppg=12.0)
        add("D-BBB", "BBB", "DEF", 3500, 0, fppg=4.0)
        return pd.DataFrame(rows)

    def test_without_a_projection_the_slot_still_takes_fppg(self):
        from nfl_projections import optimizer
        lineups = optimizer.optimize_lineups(self._slate(), num_lineups=1,
                                             max_per_team=None)
        assert lineups[0][lineups[0]["Roster Position"] == "DEF"].iloc[0]["Nickname"] == "D-AAA"

    def test_a_joined_projection_overrides_it(self):
        from nfl_projections import optimizer
        slate = self._slate()
        slate["defense_points"] = np.where(slate["Nickname"] == "D-BBB", 11.0,
                                           np.where(slate["Nickname"] == "D-AAA", 3.0, np.nan))
        lineups = optimizer.optimize_lineups(slate, num_lineups=1, max_per_team=None)
        assert lineups[0][lineups[0]["Roster Position"] == "DEF"].iloc[0]["Nickname"] == "D-BBB"

    def test_a_defense_with_no_projection_keeps_its_fppg(self):
        from nfl_projections import optimizer
        slate = self._slate()
        slate["defense_points"] = np.nan          # nothing joined for either
        lineups = optimizer.optimize_lineups(slate, num_lineups=1, max_per_team=None)
        assert lineups[0][lineups[0]["Roster Position"] == "DEF"].iloc[0]["Nickname"] == "D-AAA"


class TestTeamSpellings:
    """FanDuel and nflverse disagree on three teams, and a defense that does
    not join is a defense that silently keeps the FPPG this module replaces."""

    @pytest.mark.parametrize("fanduel,nflverse", [
        ("JAC", "JAX"), ("LAR", "LA"), ("WSH", "WAS"), ("OAK", "LV"),
        ("SD", "LAC"), ("STL", "LA"), ("KC", "KC"), ("", "")])
    def test_one_spelling_per_team(self, fanduel, nflverse):
        assert defense.normalize_team(fanduel) == nflverse

    def test_it_is_case_and_space_insensitive(self):
        assert defense.normalize_team(" jac ") == "JAX"
