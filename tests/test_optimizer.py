import warnings
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from nfl_projections import optimizer


def make_merged():
    """A minimal merged salary+projection frame with a fillable roster."""
    rows = []

    def add(nick, pos, salary, mean, ceiling, floor, fppg=None):
        rows.append({
            "Nickname": nick, "Roster Position": pos, "Salary": salary,
            "FPPG": fppg if fppg is not None else mean,
            "fanduel_fantasy_points": mean, "ceiling": ceiling, "floor": floor,
            "projection_median": mean, "Injury Indicator": "",
        })

    add("QB1", "QB", 8000, 20, 30, 12)
    add("RB1", "RB/FLEX", 7000, 15, 28, 6)
    add("RB2", "RB/FLEX", 6000, 12, 20, 5)
    add("RB3", "RB/FLEX", 5000, 10, 18, 4)
    add("WR1", "WR/FLEX", 7000, 14, 26, 5)
    add("WR2", "WR/FLEX", 6000, 12, 22, 4)
    add("WR3", "WR/FLEX", 5500, 11, 20, 4)
    add("WR4", "WR/FLEX", 5000, 10, 19, 3)
    add("TE1", "TE/FLEX", 5000, 9, 16, 3)
    add("TE2", "TE/FLEX", 4500, 8, 14, 2)
    add("DEF1", "DEF", 3500, 0, 0, 0, fppg=8)
    return pd.DataFrame(rows)


class TestMerge:
    def test_merge_carries_projection_and_range(self):
        fanduel = pd.DataFrame({
            "Nickname": ["Bijan Robinson", "Josh Allen"],
            "Salary": [8000, 9000], "Roster Position": ["RB/FLEX", "QB"],
            "FPPG": [18.0, 22.0],
        })
        preds = pd.DataFrame({
            "player_name": ["Bijan Robinson", "Josh Allen"],
            "fanduel_fantasy_points": [15.0, 20.0],
            "floor": [6.0, 9.0], "ceiling": [26.0, 30.0],
        })
        merged = optimizer.merge_fanduel_salaries(fanduel, preds)
        assert {"floor", "ceiling"} <= set(merged.columns)
        # value = points / salary * 1000
        bijan = merged[merged["Nickname"] == "Bijan Robinson"].iloc[0]
        assert round(bijan["value"], 3) == round(15.0 / 8000 * 1000, 3)


class TestObjective:
    def test_objective_column_resolution(self):
        df = make_merged()
        assert optimizer._objective_column(df, "ceiling") == "ceiling"
        assert optimizer._objective_column(df, "mean") == "fanduel_fantasy_points"

    def test_missing_objective_falls_back_to_mean(self):
        df = make_merged().drop(columns=["ceiling"])
        assert optimizer._objective_column(df, "ceiling") == "fanduel_fantasy_points"


class TestOptimizeObjective:
    def _skill_row(self, lineup):
        return lineup[lineup["Roster Position"] != "DEF"].iloc[0]

    def test_ceiling_objective_uses_ceiling_column(self):
        lineups = optimizer.optimize_lineups(
            make_merged(), num_lineups=1, salary_cap=60000, objective="ceiling"
        )
        assert len(lineups) == 1
        row = self._skill_row(lineups[0])
        assert row["lineup_points"] == row["ceiling"]

    def test_mean_objective_uses_mean_column(self):
        lineups = optimizer.optimize_lineups(
            make_merged(), num_lineups=1, salary_cap=60000, objective="mean"
        )
        row = self._skill_row(lineups[0])
        assert row["lineup_points"] == row["fanduel_fantasy_points"]

    def test_def_always_uses_fppg(self):
        lineups = optimizer.optimize_lineups(
            make_merged(), num_lineups=1, salary_cap=60000, objective="ceiling"
        )
        defense = lineups[0][lineups[0]["Roster Position"] == "DEF"].iloc[0]
        assert defense["lineup_points"] == defense["FPPG"] == 8


class TestEligibility:
    """FanDuel's export has used bare positions ("RB") and flex-qualified ones
    ("RB/FLEX"). Matching one exactly built ZERO lineups from the other, with
    nothing in the response to say why — hit live on a real Thu-Mon slate."""

    def test_bare_positions(self):
        assert optimizer.eligible_slots("RB") == {"RB/FLEX", "FLEX"}
        assert optimizer.eligible_slots("WR") == {"WR/FLEX", "FLEX"}
        assert optimizer.eligible_slots("TE") == {"TE/FLEX", "FLEX"}
        assert optimizer.eligible_slots("QB") == {"QB"}

    def test_flex_qualified_positions(self):
        assert optimizer.eligible_slots("RB/FLEX") == {"RB/FLEX", "FLEX"}
        assert optimizer.eligible_slots("TE/FLEX") == {"TE/FLEX", "FLEX"}

    def test_the_many_spellings_of_a_defense(self):
        for spelling in ("D", "DEF", "DST", "D/ST"):
            assert optimizer.eligible_slots(spelling) == {"DEF"}, spelling

    def test_kickers_fill_nothing(self):
        assert optimizer.eligible_slots("K") == set()

    def test_falls_back_to_the_position_column(self):
        # Some exports put something unhelpful in Roster Position.
        assert optimizer.eligible_slots("", "RB") == {"RB/FLEX", "FLEX"}
        assert optimizer.eligible_slots(None, "QB") == {"QB"}
        assert optimizer.eligible_slots(float("nan"), "D") == {"DEF"}

    def test_unknown_values_are_simply_not_eligible(self):
        assert optimizer.eligible_slots("LS", "LS") == set()

    def test_slot_counts_explain_an_impossible_slate(self):
        import pandas as pd

        frame = pd.DataFrame({
            "Roster Position": ["QB", "RB", "RB", "WR", "WR", "WR", "D"],
            "Position": ["QB", "RB", "RB", "WR", "WR", "WR", "D"],
        })
        counts = optimizer.slot_counts(frame)
        assert counts["QB"] == 1 and counts["RB/FLEX"] == 2 and counts["DEF"] == 1
        assert counts["TE/FLEX"] == 0        # the slate has no tight end
        assert counts["FLEX"] == 5


class TestShowdown:
    """A single-game slate: five players, one MVP at 1.5x points and 1.5x salary.

    This is the slate that failed live — run through the classic optimizer it
    produces nothing, because none of its rows name a classic slot.
    """

    @staticmethod
    def _slate():
        # Shaped like the real export: one game, MVP salary column, kickers and
        # defenses in the pool.
        rows = [
            ("CeeDee Lamb", "WR", 12600, 18900, 25.7, 18.0),
            ("Dak Prescott", "QB", 11800, 17700, 21.3, 19.0),
            ("Bucky Irving", "RB", 11200, 16800, 10.2, 12.0),
            ("Mike Evans", "WR", 9800, 14700, 14.1, 13.0),
            ("Jake Ferguson", "TE", 7600, 11400, 9.4, 9.0),
            ("Cheap Flyer", "WR", 4500, 6750, 5.0, 6.0),
            ("Deep Guy", "TE", 3500, 5250, 3.0, 4.0),
            ("Kicker One", "K", 4800, 7200, 8.0, None),
            ("Dallas Cowboys", "D", 4200, 6300, 7.5, None),
        ]
        frame = pd.DataFrame(rows, columns=[
            "Nickname", "Position", "Salary", "MVP 1.5x Salary", "FPPG",
            "fanduel_fantasy_points"])
        frame["Roster Position"] = "MVP - 1.5X Points/AnyFLEX"
        frame["Injury Indicator"] = ""
        return frame

    def test_detected_from_the_mvp_column(self):
        assert optimizer.is_showdown(self._slate())

    def test_detected_from_the_roster_position_alone(self):
        frame = self._slate().drop(columns=["MVP 1.5x Salary"])
        assert optimizer.is_showdown(frame)

    def test_a_classic_slate_is_not_a_showdown(self):
        frame = pd.DataFrame({"Roster Position": ["QB", "RB/FLEX", "DEF"]})
        assert not optimizer.is_showdown(frame)

    def test_builds_five_players_with_one_mvp(self):
        (lineup,) = optimizer.optimize_showdown(self._slate(), num_lineups=1)
        assert len(lineup) == 5
        assert list(lineup["Roster Position"]) == ["MVP", "FLEX", "FLEX", "FLEX", "FLEX"]

    def test_the_mvp_pays_and_scores_the_multiplier(self):
        (lineup,) = optimizer.optimize_showdown(self._slate(), num_lineups=1)
        mvp = lineup.iloc[0]
        source = self._slate().set_index("Nickname").loc[mvp["Nickname"]]
        assert mvp["Salary"] == source["MVP 1.5x Salary"]
        assert mvp["lineup_points"] == pytest.approx(
            max(source["fanduel_fantasy_points"], 0) * 1.5, rel=1e-3)

    def test_stays_under_the_cap_at_mvp_prices(self):
        lineups = optimizer.optimize_showdown(self._slate(), num_lineups=3,
                                              max_usage_percentage=100)
        assert lineups
        for lineup in lineups:
            assert lineup["Salary"].sum() <= 60000

    def test_kickers_and_defenses_are_eligible_and_use_fanduels_points(self):
        lineups = optimizer.optimize_showdown(self._slate(), num_lineups=4,
                                              max_usage_percentage=100)
        names = {n for l in lineups for n in l["Nickname"]}
        assert {"Kicker One", "Dallas Cowboys"} & names, "neither is ever usable"
        for lineup in lineups:
            for _, row in lineup.iterrows():
                if row["Nickname"] == "Kicker One" and row["Roster Position"] != "MVP":
                    assert row["lineup_points"] == 8.0      # FPPG, not a projection

    def test_different_mvps_make_different_lineups(self):
        lineups = optimizer.optimize_showdown(self._slate(), num_lineups=3,
                                              max_usage_percentage=100)
        assert len({l.iloc[0]["Nickname"] for l in lineups}) == len(lineups) > 1

    def test_exclusions_apply(self):
        lineups = optimizer.optimize_showdown(self._slate(), num_lineups=2,
                                              exclude_players=["CeeDee Lamb"],
                                              max_usage_percentage=100)
        names = {n for l in lineups for n in l["Nickname"]}
        assert "CeeDee Lamb" not in names

    def test_a_cap_nothing_fits_returns_nothing(self):
        assert optimizer.optimize_showdown(self._slate(), num_lineups=1, salary_cap=5000) == []


class TestShowdownQuality:
    """The greedy fill left $21k of a $60k cap unspent on a real slate and took
    both kickers; these pin the two fixes."""

    def test_it_spends_the_cap(self):
        slate = TestShowdown._slate()
        (lineup,) = optimizer.optimize_showdown(slate, num_lineups=1)
        spent = lineup["Salary"].sum()
        # Value-greedy filling scored 58.8 on the real slate against 78.7 here;
        # the signature of the old behaviour was leaving a third of the cap.
        assert spent > 0.75 * 60000, f"only spent {spent}"

    def test_it_maximises_points_not_value(self):
        slate = TestShowdown._slate()
        (exact,) = optimizer.optimize_showdown(slate, num_lineups=1)
        # Any other legal 5 from this pool should not beat it.
        import itertools

        best_other = 0.0
        rows = slate.set_index("Nickname")
        for combo in itertools.combinations(rows.index, 5):
            for mvp in combo:
                salary = sum(rows.loc[n, "MVP 1.5x Salary"] if n == mvp
                             else rows.loc[n, "Salary"] for n in combo)
                if salary > 60000:
                    continue
                points = 0.0
                for n in combo:
                    base = rows.loc[n, "fanduel_fantasy_points"]
                    base = rows.loc[n, "FPPG"] if base != base else base
                    points += base * (1.5 if n == mvp else 1)
                best_other = max(best_other, points)
        assert exact["lineup_points"].sum() >= best_other - 0.01

    def test_position_limits_stop_two_kickers(self):
        slate = TestShowdown._slate().copy()
        # Two cheap, efficient kickers: exactly the trap FPPG sets.
        slate.loc[len(slate)] = ["Kicker Two", "K", 4600, 6900, 8.2, None,
                                 "MVP - 1.5X Points/AnyFLEX", ""]
        unlimited = optimizer.optimize_showdown(slate, num_lineups=1)[0]
        limited = optimizer.optimize_showdown(
            slate, num_lineups=1, position_limits={"K": 1})[0]
        assert sum(limited["Position"] == "K") <= 1
        # And the limited lineup is the weaker one, which is the honest trade.
        assert limited["lineup_points"].sum() <= unlimited["lineup_points"].sum()


class TestItBuildsWhateverFits:
    """The greedy builder returned NOTHING on slates where plenty fits.

    Reported from a real slate: "Every slot has players, so the salary cap
    ($60,000) or the exclusions are what nothing fits under. It has to be able
    to build something." It filled slot by slot taking the best points-per-
    dollar player that fit the salary remaining, held nothing back for the slots
    still to come, and abandoned the attempt rather than retrying cheaper when
    one came up empty.
    """

    @staticmethod
    def _trap_slate():
        """Expensive players who ALSO have the best points-per-dollar.

        Greedy spends 48,000 of the 60,000 on two backs and three receivers,
        reaches the tight end slot with 3,000 left, finds nothing, and gives up
        -- while a legal lineup costs 50,000.
        """
        rows = []

        def add(name, pos, salary, pts):
            rows.append({"Nickname": name, "Roster Position": pos, "Position": pos,
                         "Salary": salary, "FPPG": pts, "fanduel_fantasy_points": pts})

        add("QB Star", "QB", 12000, 25.0)
        add("QB Cheap", "QB", 6000, 12.0)
        for i in range(2):
            add(f"RB Good {i}", "RB", 9000, 20.0)
        for i in range(3):
            add(f"WR Good {i}", "WR", 9000, 20.0)
        add("TE Good", "TE", 8000, 17.0)
        for i in range(3):
            add(f"RB Cheap {i}", "RB", 4000, 6.0)
        for i in range(4):
            add(f"WR Cheap {i}", "WR", 4000, 6.0)
        for i in range(2):
            add(f"TE Cheap {i}", "TE", 4000, 5.0)
        add("DEF One", "D", 5000, 8.0)
        add("DEF Two", "D", 5000, 7.5)
        return pd.DataFrame(rows)

    def test_a_lineup_exists_so_one_is_returned(self):
        lineups = optimizer.optimize_lineups(self._trap_slate(), num_lineups=1)
        assert len(lineups) == 1
        assert len(lineups[0]) == 9

    def test_and_it_spends_the_cap_rather_than_banking_it(self):
        """Value-sorting is what left money on the table; points is the score."""
        (lineup,) = optimizer.optimize_lineups(self._trap_slate(), num_lineups=1)
        assert lineup["Salary"].sum() > 55000
        assert lineup["Salary"].sum() <= 60000

    def test_the_cap_is_never_breached(self):
        lineups = optimizer.optimize_lineups(self._trap_slate(), num_lineups=5)
        assert lineups
        for lineup in lineups:
            assert lineup["Salary"].sum() <= 60000

    def test_a_genuinely_impossible_cap_returns_nothing(self):
        """The message the user saw is only honest when this is the case."""
        assert optimizer.optimize_lineups(self._trap_slate(), salary_cap=20000) == []

    def test_the_cheapest_lineup_cost_says_by_how_much(self):
        cheapest = optimizer.cheapest_lineup_cost(self._trap_slate())
        # 6000 QB + 4000x3 RB + 4000x3 WR + 4000 TE + 5000 DEF, the FLEX going
        # to whichever group is cheapest. Nine players, not eight.
        assert cheapest == 39000
        assert optimizer.optimize_lineups(
            self._trap_slate(), salary_cap=cheapest, num_lineups=1)
        assert optimizer.optimize_lineups(
            self._trap_slate(), salary_cap=cheapest - 100, num_lineups=1) == []

    def test_a_slate_missing_a_position_has_no_cheapest_cost(self):
        slate = self._trap_slate()
        assert optimizer.cheapest_lineup_cost(
            slate[slate["Position"] != "D"]) is None


class TestPriceLadder:
    """Several lineups spread across price points, every one under the cap."""

    @staticmethod
    def _slate():
        rows = []
        for pos, n, lo in (("QB", 8, 60), ("RB", 16, 45), ("WR", 22, 45),
                           ("TE", 10, 40), ("D", 6, 30)):
            for i in range(n):
                salary = (lo + i * 2) * 100
                rows.append({
                    "Nickname": f"{pos}{i}", "Roster Position": pos, "Position": pos,
                    "Salary": salary, "FPPG": 4.0 + salary / 900.0,
                    "fanduel_fantasy_points": 4.0 + salary / 900.0})
        return pd.DataFrame(rows)

    def test_each_lineup_is_strictly_cheaper_than_the_last(self):
        lineups = optimizer.optimize_lineups(self._slate(), num_lineups=5)
        assert len(lineups) == 5
        spends = [l["Salary"].sum() for l in lineups]
        assert spends == sorted(spends, reverse=True)
        assert len(set(spends)) == len(spends)

    def test_the_spread_is_wide_enough_to_be_different_pricing(self):
        """A $100 step buys a one-player swap, which is not a price variation."""
        lineups = optimizer.optimize_lineups(self._slate(), num_lineups=5)
        spends = [l["Salary"].sum() for l in lineups]
        assert spends[0] - spends[-1] >= 4 * 600

    def test_every_lineup_is_a_distinct_roster(self):
        lineups = optimizer.optimize_lineups(self._slate(), num_lineups=5)
        rosters = {tuple(sorted(l["Nickname"])) for l in lineups}
        assert len(rosters) == len(lineups)

    def test_points_fall_as_the_budget_does(self):
        """Cheaper is a trade, not a free lunch — worth seeing in the output."""
        lineups = optimizer.optimize_lineups(self._slate(), num_lineups=4,
                                             max_usage_percentage=100)
        points = [l["lineup_points"].sum() for l in lineups]
        assert points == sorted(points, reverse=True)

    def test_price_step_is_adjustable(self):
        tight = optimizer.optimize_lineups(self._slate(), num_lineups=3,
                                           price_step=100)
        wide = optimizer.optimize_lineups(self._slate(), num_lineups=3,
                                          price_step=4000)
        tight_spread = tight[0]["Salary"].sum() - tight[-1]["Salary"].sum()
        wide_spread = wide[0]["Salary"].sum() - wide[-1]["Salary"].sum()
        assert wide_spread > tight_spread

    def test_the_roster_is_legal_at_every_price(self):
        for lineup in optimizer.optimize_lineups(self._slate(), num_lineups=5):
            counts = Counter(lineup["Position"])
            assert len(lineup) == 9
            assert counts["QB"] == 1
            assert counts["D"] == 1
            assert counts["RB"] >= 2 and counts["WR"] >= 3 and counts["TE"] >= 1
            # Exactly one of them carries the extra body: the FLEX.
            assert counts["RB"] + counts["WR"] + counts["TE"] == 7

    def test_nobody_is_rostered_twice(self):
        for lineup in optimizer.optimize_lineups(self._slate(), num_lineups=5):
            assert len(set(lineup["Nickname"])) == len(lineup)


class TestCorrelationRules:
    """The three rules that say how the nine picks may relate to each other.

    Every slate here is built so the *unconstrained* answer breaks the rule
    under test — otherwise the rule passing proves nothing.
    """

    @staticmethod
    def _slate():
        """Three teams, with HOU far and away the best value at every slot.

        Left alone the builder takes five Texans, so each rule below has
        something real to bite on.
        """
        rows = []

        def add(nick, team, pos, salary, points):
            rows.append({
                "Nickname": nick, "Team": team, "Roster Position": pos,
                "Salary": salary, "FPPG": points,
                "fanduel_fantasy_points": points, "ceiling": points * 2,
                "floor": points / 2, "projection_median": points,
                "Injury Indicator": "",
            })

        for team, edge in (("HOU", 6.0), ("DAL", 0.0), ("NYG", -1.0),
                           ("SEA", -2.0), ("MIA", -3.0), ("CHI", -4.0)):
            add(f"QB-{team}", team, "QB", 7000, 18 + edge)
            for j in range(3):
                add(f"RB{j}-{team}", team, "RB/FLEX", 6000 - j * 500, 14 + edge - j)
                add(f"WR{j}-{team}", team, "WR/FLEX", 6000 - j * 500, 13 + edge - j)
            add(f"TE-{team}", team, "TE/FLEX", 4500, 9 + edge)
            add(f"TE2-{team}", team, "TE/FLEX", 4000, 7 + edge)
            add(f"DEF-{team}", team, "DEF", 3500, 8 + edge)
        return pd.DataFrame(rows)

    def _build(self, **kwargs):
        lineups = optimizer.optimize_lineups(self._slate(), num_lineups=1,
                                             salary_cap=60000, **kwargs)
        assert lineups, "no lineup was built"
        return lineups[0]

    def test_without_a_cap_the_builder_stacks_one_team(self):
        # The premise of every test below: left alone it over-concentrates.
        lineup = self._build(max_per_team=None)
        assert Counter(lineup["Team"]).most_common(1)[0][1] >= 5

    @pytest.mark.parametrize("cap", [2, 3, 4])
    def test_team_cap_is_honoured(self, cap):
        lineup = self._build(max_per_team=cap)
        assert max(Counter(lineup["Team"]).values()) <= cap

    def test_the_cap_costs_points_but_still_fills_the_roster(self):
        free = self._build(max_per_team=None)
        capped = self._build(max_per_team=2)
        assert len(capped) == len(free) == 9
        assert capped["lineup_points"].sum() < free["lineup_points"].sum()

    def test_stack_puts_a_pass_catcher_with_the_quarterback(self):
        lineup = self._build(stack_qb=1, max_per_team=None)
        qb_team = lineup[lineup["Roster Position"] == "QB"].iloc[0]["Team"]
        partners = lineup[(lineup["Team"] == qb_team)
                          & (lineup["Roster Position"].isin(["WR/FLEX", "TE/FLEX"]))]
        assert len(partners) >= 1

    def test_a_two_man_stack_is_two_pass_catchers(self):
        lineup = self._build(stack_qb=2, max_per_team=None)
        qb_team = lineup[lineup["Roster Position"] == "QB"].iloc[0]["Team"]
        partners = lineup[(lineup["Team"] == qb_team)
                          & (lineup["Roster Position"].isin(["WR/FLEX", "TE/FLEX"]))]
        assert len(partners) >= 2

    def test_a_running_back_can_be_kept_off_his_own_quarterback(self):
        lineup = self._build(rb_with_qb=False, max_per_team=None)
        qb_team = lineup[lineup["Roster Position"] == "QB"].iloc[0]["Team"]
        assert not ((lineup["Team"] == qb_team)
                    & (lineup["Roster Position"] == "RB/FLEX")).any()

    def test_the_rules_combine(self):
        lineup = self._build(stack_qb=2, rb_with_qb=False, max_per_team=3)
        qb_team = lineup[lineup["Roster Position"] == "QB"].iloc[0]["Team"]
        catchers = lineup[(lineup["Team"] == qb_team)
                          & (lineup["Roster Position"].isin(["WR/FLEX", "TE/FLEX"]))]
        assert len(catchers) >= 2
        assert not ((lineup["Team"] == qb_team)
                    & (lineup["Roster Position"] == "RB/FLEX")).any()
        assert max(Counter(lineup["Team"]).values()) <= 3

    def test_an_impossible_combination_builds_nothing(self):
        # A two-man stack needs three from the quarterback's team; a cap of two
        # forbids it, and no lineup exists rather than an illegal one escaping.
        lineups = optimizer.optimize_lineups(self._slate(), num_lineups=1,
                                             stack_qb=2, max_per_team=2)
        assert lineups == []

    def test_a_cap_the_slate_cannot_reach_builds_nothing(self):
        # Nine roster spots over six teams cannot be done one apiece.
        lineups = optimizer.optimize_lineups(self._slate(), num_lineups=1,
                                             max_per_team=1)
        assert lineups == []

    def test_rules_still_hold_across_a_whole_build(self):
        lineups = optimizer.optimize_lineups(self._slate(), num_lineups=5,
                                             max_usage_percentage=100,
                                             stack_qb=1, rb_with_qb=False,
                                             max_per_team=3)
        assert len(lineups) >= 3
        for lineup in lineups:
            qb_team = lineup[lineup["Roster Position"] == "QB"].iloc[0]["Team"]
            assert ((lineup["Team"] == qb_team)
                    & (lineup["Roster Position"].isin(["WR/FLEX", "TE/FLEX"]))).sum() >= 1
            assert not ((lineup["Team"] == qb_team)
                        & (lineup["Roster Position"] == "RB/FLEX")).any()
            assert max(Counter(lineup["Team"]).values()) <= 3

    def test_a_frame_with_no_team_column_still_builds(self):
        # The default cap cannot be applied to a frame that cannot express
        # teams, so it stands down rather than refusing to build.
        with pytest.warns(RuntimeWarning, match="Team"):
            lineups = optimizer.optimize_lineups(make_merged(), num_lineups=1)
        assert len(lineups) == 1

    def test_but_an_explicit_rule_without_teams_is_an_error(self):
        with pytest.raises(ValueError, match="Team"):
            optimizer.optimize_lineups(make_merged(), num_lineups=1, stack_qb=1)


class TestOwnershipAndTargets:
    """Charging for the field, and aiming at a score instead of a projection."""

    @staticmethod
    def pool():
        # Two interchangeable options at every slot, one chalk and one not.
        rows = []
        for slot, position, n in (("QB", "QB", 2), ("RB", "RB", 3), ("WR", "WR", 4),
                                  ("TE", "TE", 2), ("D", "D", 2)):
            for i in range(n):
                rows.append({
                    "Nickname": f"{position}{i}", "Position": position,
                    "Roster Position": slot, "Salary": 5000 - 200 * i,
                    "FPPG": 10.0 - i, "fanduel_fantasy_points": 10.0 - i,
                    "Team": f"T{i}", "ownership": 40.0 if i == 0 else 1.0,
                })
        return pd.DataFrame(rows)

    def test_charging_for_ownership_swaps_the_chalk_out(self):
        plain = optimizer.optimize_lineups(self.pool(), num_lineups=1, salary_cap=60000,
                                 max_usage_percentage=100, max_per_team=None)
        assert "QB0" in set(plain[0]["Nickname"])
        # QB0 out-projects QB1 by 1.0 but is 39 points of ownership dearer, so
        # at 0.1 points per percent he costs 3.9 and is no longer worth it.
        tilted = optimizer.optimize_lineups(self.pool(), num_lineups=1, salary_cap=60000,
                                  max_usage_percentage=100, max_per_team=None,
                                  ownership_weight=0.1)
        assert "QB0" not in set(tilted[0]["Nickname"])

    def test_a_small_charge_is_not_enough_to_move_anyone(self):
        tilted = optimizer.optimize_lineups(self.pool(), num_lineups=1, salary_cap=60000,
                                  max_usage_percentage=100, max_per_team=None,
                                  ownership_weight=0.001)
        assert "QB0" in set(tilted[0]["Nickname"])

    def test_charging_for_ownership_needs_the_column(self):
        with pytest.raises(ValueError, match="ownership"):
            optimizer.optimize_lineups(self.pool().drop(columns=["ownership"]), num_lineups=1,
                             max_usage_percentage=100, max_per_team=None,
                             ownership_weight=0.1)

    def test_unknown_ownership_is_not_charged_for(self):
        # An unlisted player is a player the field has written off, usually
        # because he is inactive. Reading NaN as 0% would make him the single
        # most attractive name on the board.
        pool = self.pool()
        pool.loc[pool["Nickname"] == "QB0", "ownership"] = np.nan
        tilted = optimizer.optimize_lineups(pool, num_lineups=1, salary_cap=60000,
                                  max_usage_percentage=100, max_per_team=None,
                                  ownership_weight=0.1)
        assert "QB0" in set(tilted[0]["Nickname"])   # charged 0, still the best

    def test_a_target_needs_simulations(self):
        with pytest.raises(ValueError, match="sims"):
            optimizer.optimize_lineups(self.pool(), num_lineups=1, max_usage_percentage=100,
                             max_per_team=None, target=125.0)

    def test_misaligned_simulations_are_refused(self):
        with pytest.raises(ValueError, match="sims has"):
            optimizer.optimize_lineups(self.pool(), num_lineups=1, max_usage_percentage=100,
                             max_per_team=None, target=125.0,
                             sims=np.zeros((3, 100)))

    def test_a_duplicated_index_is_refused(self):
        pool = self.pool()
        pool.index = [0] * len(pool)
        with pytest.raises(ValueError, match="duplicate labels"):
            optimizer.optimize_lineups(pool, num_lineups=1, max_usage_percentage=100,
                             max_per_team=None, target=125.0,
                             sims=np.zeros((len(pool), 100)))

    def test_a_target_ranks_by_the_chance_of_clearing_it(self):
        pool = self.pool()
        # Everyone scores 10 every sim except QB1, who scores 30 half the time.
        sims = np.full((len(pool), 200), 10.0)
        sims[pool.index[pool["Nickname"] == "QB1"][0], :100] = 30.0
        built = optimizer.optimize_lineups(pool, num_lineups=1, salary_cap=60000,
                                 max_usage_percentage=100, max_per_team=None,
                                 target=95.0, sims=sims, candidates=20,
                                 salary_step=200, price_step=200)
        assert "QB1" in set(built[0]["Nickname"])
        assert built[0].attrs["p_target"] == pytest.approx(0.5)

    def test_a_target_reports_what_it_found(self):
        pool = self.pool()
        sims = np.full((len(pool), 50), 10.0)
        built = optimizer.optimize_lineups(pool, num_lineups=1, salary_cap=60000,
                                 max_usage_percentage=100, max_per_team=None,
                                 target=80.0, sims=sims, candidates=4)
        assert built[0].attrs["p_target"] == 1.0
        assert built[0].attrs["sim_mean"] == pytest.approx(90.0)

    def test_a_target_does_not_spread_by_default(self):
        # Five near-identical lineups is the right answer to "clear 125", and
        # the usage cap must not quietly override that.
        pool = self.pool()
        sims = np.full((len(pool), 50), 10.0)
        built = optimizer.optimize_lineups(pool, num_lineups=3, salary_cap=60000,
                                 max_usage_percentage=40, max_per_team=None,
                                 target=80.0, sims=sims, candidates=20,
                                 salary_step=200, price_step=200)
        assert len(built) == 3
        counts = Counter(n for l in built for n in l["Nickname"])
        assert max(counts.values()) == 3          # the best name is in all three

    def test_spreading_skips_a_lineup_that_would_break_the_cap(self):
        # Tested on the selection directly: with nine of thirteen players in
        # every lineup, no fixture can give three genuinely disjoint ones.
        def lineup(names):
            return pd.DataFrame({"Nickname": list(names)})

        ranked = [lineup("ABCDEFGHI"), lineup("ABCDEFGHJ"), lineup("KLMNOPQRS")]
        chosen = optimizer._select_diverse(ranked, 2, max_usage_percentage=50)
        assert [list(l["Nickname"])[0] for l in chosen] == ["A", "K"]

    def test_spreading_falls_back_rather_than_returning_too_few(self):
        # A cap no candidate set can satisfy should cost diversity, not lineups.
        def lineup(names):
            return pd.DataFrame({"Nickname": list(names)})

        ranked = [lineup("ABCDEFGHI"), lineup("ABCDEFGHJ")]
        assert len(optimizer._select_diverse(ranked, 2, max_usage_percentage=50)) == 2

    def test_a_tight_cap_on_a_small_build_is_called_out(self):
        with pytest.warns(RuntimeWarning, match="usage cap"):
            optimizer.optimize_lineups(self.pool(), num_lineups=5, salary_cap=60000,
                             max_usage_percentage=50, max_per_team=None)

    def test_a_big_build_under_the_same_cap_is_quiet(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            optimizer.optimize_lineups(self.pool(), num_lineups=25, salary_cap=60000,
                             max_usage_percentage=50, max_per_team=None)


def upload_lineup(positions, *, ids=True, points=None):
    """A lineup frame shaped for the upload exporter."""
    rows = []
    for i, position in enumerate(positions):
        rows.append({
            "Id": f"135359-{100 + i}" if ids else None,
            "Nickname": f"{position}{i}",
            "Position": position,
            "Roster Position": "DEF" if position == "D" else f"{position}/FLEX",
            "Salary": 5000,
            "lineup_points": (points[i] if points else 10.0 - i),
        })
    return pd.DataFrame(rows)


CLASSIC = ["QB", "RB", "RB", "RB", "WR", "WR", "WR", "TE", "D"]


class TestFanduelUpload:
    """The CSV FanDuel's bulk entry accepts."""

    def test_columns_are_fanduels_own_order(self):
        assert optimizer.FANDUEL_UPLOAD_COLUMNS == (
            "QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DEF")

    def test_the_surplus_running_back_becomes_the_flex(self):
        slots = optimizer.assign_upload_slots(upload_lineup(CLASSIC))
        placed = dict(zip(optimizer.FANDUEL_UPLOAD_COLUMNS,
                          [r["Position"] for r in slots]))
        assert placed["QB"] == "QB"
        assert placed["TE"] == "TE"
        assert placed["DEF"] == "D"
        assert [r["Position"] for r in slots][7] == "RB"

    def test_a_surplus_receiver_becomes_the_flex(self):
        slots = optimizer.assign_upload_slots(
            upload_lineup(["QB", "RB", "RB", "WR", "WR", "WR", "WR", "TE", "D"]))
        assert slots[7]["Position"] == "WR"

    def test_a_surplus_tight_end_becomes_the_flex(self):
        slots = optimizer.assign_upload_slots(
            upload_lineup(["QB", "RB", "RB", "WR", "WR", "WR", "TE", "TE", "D"]))
        assert slots[7]["Position"] == "TE"

    def test_the_flex_is_the_weakest_of_its_group(self):
        # Cosmetic, but deterministic: the named starters should read as the
        # starters rather than depending on row order.
        lineup = upload_lineup(CLASSIC, points=[20, 18, 5, 12, 15, 14, 13, 9, 7])
        slots = optimizer.assign_upload_slots(lineup)
        assert slots[7]["Nickname"] == "RB2"       # the 5.0-point back
        assert {slots[1]["Nickname"], slots[2]["Nickname"]} == {"RB1", "RB3"}

    def test_every_player_appears_exactly_once(self):
        slots = optimizer.assign_upload_slots(upload_lineup(CLASSIC))
        assert len({r["Nickname"] for r in slots}) == 9

    def test_a_short_lineup_is_refused(self):
        with pytest.raises(ValueError, match="nine players"):
            optimizer.assign_upload_slots(upload_lineup(CLASSIC[:8]))

    def test_a_lineup_without_a_flex_is_refused(self):
        with pytest.raises(ValueError, match="cannot fill the FLEX"):
            optimizer.assign_upload_slots(
                upload_lineup(["QB", "QB", "RB", "RB", "WR", "WR", "WR", "TE", "D"]))

    def test_a_spare_defence_cannot_fill_the_flex_either(self):
        with pytest.raises(ValueError, match="cannot fill the FLEX"):
            optimizer.assign_upload_slots(
                upload_lineup(["QB", "RB", "RB", "WR", "WR", "WR", "TE", "D", "D"]))

    def test_an_unplaceable_position_says_so(self):
        with pytest.raises(ValueError, match="upload column"):
            optimizer.assign_upload_slots(
                upload_lineup(["QB", "RB", "RB", "WR", "WR", "WR", "TE", "K", "D"]))

    def test_the_csv_has_a_header_and_one_row_per_lineup(self):
        text = optimizer.write_fanduel_upload([upload_lineup(CLASSIC)] * 2)
        lines = text.strip().split("\r\n")
        assert lines[0] == "QB,RB,RB,WR,WR,WR,TE,FLEX,DEF"
        assert len(lines) == 3
        assert all(len(line.split(",")) == 9 for line in lines)

    def test_cells_carry_the_id_and_the_name(self):
        text = optimizer.write_fanduel_upload([upload_lineup(CLASSIC)])
        assert "135359-100:QB0" in text

    def test_ids_alone_are_also_accepted_by_fanduel(self):
        text = optimizer.write_fanduel_upload([upload_lineup(CLASSIC)],
                                              include_names=False)
        assert "135359-100:" not in text
        assert "135359-100" in text

    def test_the_contest_half_of_the_id_can_be_rewritten(self):
        # An id is "<contest>-<player>". Priced from one contest's list and
        # entered in another, the whole file is rejected.
        text = optimizer.write_fanduel_upload([upload_lineup(CLASSIC)],
                                              contest_id="135859")
        assert "135859-100:QB0" in text
        assert "135359" not in text

    def test_a_player_with_no_fanduel_id_is_refused(self):
        with pytest.raises(ValueError, match="no FanDuel `Id`"):
            optimizer.write_fanduel_upload([upload_lineup(CLASSIC, ids=False)])

    def test_more_than_fanduels_limit_is_refused_not_truncated(self):
        # A silently truncated upload looks exactly like a successful one.
        with pytest.raises(ValueError, match="250 lineups"):
            optimizer.write_fanduel_upload([upload_lineup(CLASSIC)] * 251)

    def test_it_writes_crlf_because_fanduel_reads_it_back(self):
        text = optimizer.write_fanduel_upload([upload_lineup(CLASSIC)])
        assert "\r\n" in text

    def test_it_writes_to_a_path_when_given_one(self, tmp_path):
        out = tmp_path / "upload.csv"
        text = optimizer.write_fanduel_upload([upload_lineup(CLASSIC)], out)
        # read_bytes, not read_text: read_text would translate the CRLF away
        # and the test would pass whatever the file holds.
        assert out.read_bytes().decode() == text


class TestReadFanduelCsv:
    """Both files FanDuel hands out, including the ragged upload template."""

    PLAIN = ("Id,Position,First Name,Nickname,Last Name,FPPG,Played,Salary,Game,"
             "Team,Opponent,Injury Indicator,Injury Details,Tier,,,Roster Position\r\n"
             "135359-1,QB,Josh,Josh Allen,Allen,28.4,6,9400,BAL@BUF,BUF,BAL,,,,,,QB\r\n")

    TEMPLATE = (
        'QB,RB,RB,WR,WR,WR,TE,FLEX,DEF,"","Instructions"\r\n'
        '"","","","","","","","","","","1) Create a lineup"\r\n'
        '"","","","","","","","","",""\r\n'
        '"","","","","","","","","","",Player ID + Player Name,Id,Position,'
        'First Name,Nickname,Last Name,FPPG,Played,Salary,Game,Team,Opponent,'
        'Injury Indicator,Injury Details,Tier,,,Roster Position\r\n'
        '"","","","","","","","","","",135859-1:Josh Allen,135859-1,QB,Josh,'
        'Josh Allen,Allen,28.4,6,9400,BAL@BUF,BUF,BAL,,,,,,QB\r\n')

    def test_reads_the_plain_players_list(self, tmp_path):
        path = tmp_path / "players.csv"
        path.write_text(self.PLAIN, newline="")
        frame = optimizer.read_fanduel_csv(path)
        assert list(frame["Nickname"]) == ["Josh Allen"]
        assert frame["Salary"].iloc[0] == 9400

    def test_reads_the_ragged_upload_template(self, tmp_path):
        # Eleven fields on the instruction rows, twenty-eight on the player
        # rows: read_csv calls that a broken file.
        path = tmp_path / "template.csv"
        path.write_text(self.TEMPLATE, newline="")
        frame = optimizer.read_fanduel_csv(path)
        assert list(frame["Nickname"]) == ["Josh Allen"]
        assert frame["Id"].iloc[0] == "135859-1"

    def test_the_template_carries_the_contest_you_are_entering(self, tmp_path):
        # The whole reason to prefer it over the players list.
        path = tmp_path / "template.csv"
        path.write_text(self.TEMPLATE, newline="")
        frame = optimizer.read_fanduel_csv(path)
        assert frame["Id"].iloc[0].split("-")[0] == "135859"

    def test_numbers_come_back_as_numbers(self, tmp_path):
        path = tmp_path / "template.csv"
        path.write_text(self.TEMPLATE, newline="")
        frame = optimizer.read_fanduel_csv(path)
        assert frame["FPPG"].iloc[0] == pytest.approx(28.4)

    def test_it_accepts_a_file_object(self, tmp_path):
        import io as _io
        frame = optimizer.read_fanduel_csv(_io.StringIO(self.TEMPLATE))
        assert list(frame["Nickname"]) == ["Josh Allen"]

    def test_something_else_entirely_is_refused(self, tmp_path):
        path = tmp_path / "nope.csv"
        path.write_text("a,b,c\r\n1,2,3\r\n", newline="")
        with pytest.raises(ValueError, match="FanDuel player list"):
            optimizer.read_fanduel_csv(path)
