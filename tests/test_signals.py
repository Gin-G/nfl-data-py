"""Matchup signals recorded before kickoff, so they can be judged later."""

import numpy as np
import pandas as pd
import pytest

from nfl_projections import signals


def projections():
    return pd.DataFrame({
        "season": [2026] * 4,
        "week": [8] * 4,
        "player_id": ["p1", "p2", "p3", "p4"],
        "player_name": ["Back A", "Back B", "Bench C", "Receiver D"],
        "position": ["RB", "RB", "RB", "WR"],
        "team": ["cin", "TEN", "CIN", "TEN"],
        "projected": [14.0, 11.0, 2.0, 12.0],
    })


def trenches():
    return pd.DataFrame({"team": ["CIN", "TEN"],
                         "run_matchup": [66.1, 46.0],
                         "pass_matchup": [47.6, 50.6],
                         "opponent": ["TEN", "CIN"]})


def coverage():
    return pd.DataFrame({"team": ["CIN", "TEN"], "opp_cover_grade": [31.0, 55.0]})


class TestSnapshot:
    def test_it_records_the_signals_against_each_player(self):
        out = signals.snapshot(projections(), trenches(), coverage())
        row = out[out.player_id == "p1"].iloc[0]
        assert row.run_matchup == pytest.approx(66.1)
        assert row.opp_cover_grade == pytest.approx(31.0)
        assert row.opponent == "TEN"

    def test_team_codes_are_matched_case_insensitively(self):
        # The projections frame had "cin"; the trench report has "CIN".
        out = signals.snapshot(projections(), trenches(), coverage())
        assert out[out.player_id == "p1"].iloc[0].run_matchup == pytest.approx(66.1)

    def test_it_never_adjusts_a_projection(self):
        # The whole point: a record, not a model.
        out = signals.snapshot(projections(), trenches(), coverage())
        assert list(out.projected) == [14.0, 11.0, 2.0, 12.0]

    def test_it_marks_who_a_lineup_would_consider(self):
        out = signals.snapshot(projections(), trenches(), coverage())
        assert list(out.rosterable) == [True, True, False, True]

    def test_missing_trenches_leave_the_column_empty_not_the_snapshot(self):
        # A signal that could not be computed is a fact about the week.
        out = signals.snapshot(projections(), None, coverage())
        assert len(out) == 4
        assert out.run_matchup.isna().all()
        assert out.opp_cover_grade.notna().any()

    def test_missing_coverage_is_the_same(self):
        out = signals.snapshot(projections(), trenches(), None)
        assert out.opp_cover_grade.isna().all()
        assert out.run_matchup.notna().any()

    def test_a_projection_frame_without_its_columns_is_refused(self):
        with pytest.raises(ValueError, match="projected"):
            signals.snapshot(projections().drop(columns=["projected"]))

    def test_the_column_order_is_fixed(self):
        # It is written to a table and read back weeks later.
        assert list(signals.snapshot(projections()).columns) == signals.COLUMNS


class TestBands:
    def test_it_uses_the_trench_reports_own_bands(self):
        assert signals.band(66.1) == "lean up"
        assert signals.band(71.0) == "smash"
        assert signals.band(50.0) == "neutral"
        assert signals.band(20.0) == "fade"

    def test_an_unknown_rating_says_so(self):
        assert signals.band(None) == "unknown"
        assert signals.band(float("nan")) == "unknown"


class TestScore:
    def test_it_attaches_outcomes_and_the_miss(self):
        snap = signals.snapshot(projections(), trenches(), coverage())
        out = signals.score(snap, {"p1": 31.9, "p2": 8.0})
        assert out[out.player_id == "p1"].iloc[0]["miss"] == pytest.approx(17.9)
        assert out[out.player_id == "p2"].iloc[0]["miss"] == pytest.approx(-3.0)

    def test_a_player_with_no_stat_line_is_unknown_not_zero(self):
        # Whether the signal predicts PERFORMANCE is a different question from
        # whether a man was active, and the ownership list answers that better.
        snap = signals.snapshot(projections(), trenches(), coverage())
        out = signals.score(snap, {"p1": 31.9})
        assert np.isnan(out[out.player_id == "p2"].iloc[0]["miss"])


class TestVerdict:
    @staticmethod
    def recorded(gaps):
        """One week per entry: a flagged back and an unflagged one."""
        rows = []
        for week, gap in enumerate(gaps, start=1):
            rows += [
                dict(season=2026, week=week, player_id=f"up{week}", player_name="up",
                     position="RB", team="A", opponent="B", projected=12.0,
                     run_matchup=65.0, pass_matchup=50.0, opp_cover_grade=40.0,
                     run_lean="lean up", rosterable=True, actual=12.0 + gap,
                     miss=float(gap)),
                dict(season=2026, week=week, player_id=f"dn{week}", player_name="down",
                     position="RB", team="C", opponent="D", projected=12.0,
                     run_matchup=45.0, pass_matchup=50.0, opp_cover_grade=40.0,
                     run_lean="neutral", rosterable=True, actual=12.0, miss=0.0),
            ]
        return pd.DataFrame(rows)

    def test_nothing_recorded_says_so(self):
        empty = self.recorded([]).reindex(columns=self.recorded([1]).columns)
        assert signals.verdict(empty)["verdict"] == "nothing recorded yet"

    def test_it_counts_the_weeks_that_pointed_the_right_way(self):
        out = signals.verdict(self.recorded([5, 3, -2, 4, 6]))
        assert out["weeks"] == 5
        assert out["weeks_right"] == 4

    def test_it_reports_the_pooled_gap(self):
        out = signals.verdict(self.recorded([4, 4, 4]))
        assert out["gap"] == pytest.approx(4.0)
        assert out["flagged"] == 3

    def test_per_week_detail_is_kept(self):
        # Pooling hides a signal that wins one slate and averages out.
        out = signals.verdict(self.recorded([10, -1]))
        assert [w["gap"] for w in out["per_week"]] == [10.0, -1.0]

    def test_the_deep_bench_is_left_out(self):
        rows = self.recorded([5])
        rows.loc[rows.player_id == "up1", "rosterable"] = False
        assert signals.verdict(rows)["flagged"] == 0

    def test_an_unplayed_player_does_not_count(self):
        rows = self.recorded([5])
        rows.loc[rows.player_id == "up1", "miss"] = np.nan
        assert signals.verdict(rows)["flagged"] == 0
