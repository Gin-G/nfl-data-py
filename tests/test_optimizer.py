from collections import Counter

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
