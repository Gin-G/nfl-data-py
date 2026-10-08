import pandas as pd

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
