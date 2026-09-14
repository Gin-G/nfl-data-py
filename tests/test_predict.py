import pytest
import pandas as pd

from nfl_projections.predict import (
    DepthChartAnalyzer,
    Projector,
    InjuryStatusAnalyzer,
    _extract_team,
    current_depth_chart,
    dedupe_rosters,
    projectable_status,
    week_start_date,
)


def make_depth_charts():
    return pd.DataFrame({
        "team": ["DAL", "DAL", "DAL", "PHI"],
        "pos_abb": ["RB", "RB", "QB", "RB"],
        "pos_rank": [1, 2, 1, 1],
        "player_name": ["Star Runner", "Backup Runner", "Franchise QB", "Eagle Back"],
    })


class TestDepthChartAnalyzer:
    def test_roles(self):
        analyzer = DepthChartAnalyzer(make_depth_charts())
        assert analyzer.get_player_role("Star Runner")["role"] == "starter"
        assert analyzer.get_player_role("Backup Runner")["role"] == "backup"
        assert analyzer.get_player_role("Nobody Special") is None

    def test_fuzzy_name_match(self):
        analyzer = DepthChartAnalyzer(make_depth_charts())
        # First initial + last name matches
        assert analyzer.get_player_role("S. Runner")["role"] == "starter"

    def test_rookie_opportunity_starter(self):
        analyzer = DepthChartAnalyzer(make_depth_charts())
        analysis = analyzer.analyze_rookie_opportunity("Star Runner", "DAL", "RB")
        assert analysis["opportunity"] == "high"

    def test_rookie_opportunity_backup_rb(self):
        analyzer = DepthChartAnalyzer(make_depth_charts())
        analysis = analyzer.analyze_rookie_opportunity("Backup Runner", "DAL", "RB")
        assert analysis["opportunity"] == "medium"

    def test_handles_nan_player_names(self):
        # Real depth charts sometimes carry NaN player names; must not crash
        charts = make_depth_charts()
        charts.loc[len(charts)] = {"team": "NYG", "pos_abb": "WR",
                                   "pos_rank": 3, "player_name": float("nan")}
        analyzer = DepthChartAnalyzer(charts)
        assert analyzer.get_player_role("Star Runner")["role"] == "starter"
        assert analyzer.get_player_role(float("nan")) is None


def make_snapshots():
    """nflverse's 2025+ format: dated league-wide snapshots, newest rows first."""
    rows = [
        # September: the QB won the job and the rookie runner took over.
        ("2026-09-14T13:53:31Z", "CLE", "QB", 1, "Deshaun Watson"),
        ("2026-09-14T13:53:31Z", "CLE", "QB", 2, "Shedeur Sanders"),
        ("2026-09-14T13:53:31Z", "CLE", "RB", 1, "Rookie Runner"),
        # August
        ("2026-08-20T12:00:00Z", "CLE", "QB", 1, "Shedeur Sanders"),
        ("2026-08-20T12:00:00Z", "CLE", "QB", 2, "Deshaun Watson"),
        ("2026-08-20T12:00:00Z", "CLE", "RB", 1, "Old Runner"),
        # March, before the draft
        ("2026-03-22T06:38:42Z", "CLE", "QB", 1, "Shedeur Sanders"),
        ("2026-03-22T06:38:42Z", "CLE", "QB", 3, "Deshaun Watson"),
        ("2026-03-22T06:38:42Z", "CLE", "RB", 1, "Old Runner"),
    ]
    return pd.DataFrame(rows, columns=["dt", "team", "pos_abb", "pos_rank", "player_name"])


class TestCurrentDepthChart:
    def test_keeps_only_the_latest_snapshot(self):
        chart = current_depth_chart(make_snapshots())
        assert chart["dt"].unique().tolist() == ["2026-09-14T13:53:31Z"]
        assert len(chart) == 3

    def test_as_of_never_reads_a_later_chart(self):
        chart = current_depth_chart(make_snapshots(), as_of="2026-09-10")
        assert chart["dt"].unique().tolist() == ["2026-08-20T12:00:00Z"]

    def test_as_of_includes_that_days_snapshot(self):
        chart = current_depth_chart(make_snapshots(), as_of="2026-09-14")
        assert chart["dt"].unique().tolist() == ["2026-09-14T13:53:31Z"]

    def test_nothing_before_as_of_is_empty(self):
        assert current_depth_chart(make_snapshots(), as_of="2026-01-01").empty

    def test_weekly_format_passes_through(self):
        charts = make_depth_charts()
        assert current_depth_chart(charts) is charts

    def test_name_fallback_stays_on_the_players_team(self):
        charts = pd.DataFrame({
            "team": ["CAR", "NO"], "pos_abb": ["RB", "RB"], "pos_rank": [4, 1],
            "player_name": ["Trevor Etienne", "Alvin Kamara"],
        })
        analyzer = DepthChartAnalyzer(charts)
        assert analyzer.get_player_role("Travis Etienne", team="NO") is None
        assert analyzer.get_player_role("Travis Etienne")["team"] == "CAR"  # no team: old behaviour

    def test_analyzer_reads_the_current_chart_not_the_oldest(self):
        # The bug: last-write-wins over newest-first rows meant March won.
        analyzer = DepthChartAnalyzer(make_snapshots())
        assert analyzer.get_player_role("Deshaun Watson")["depth_rank"] == 1
        assert analyzer.get_player_role("Shedeur Sanders")["depth_rank"] == 2
        assert analyzer.get_player_role("Old Runner") is None


class TestWeekStartDate:
    def test_first_gameday_of_the_week(self):
        schedule = pd.DataFrame({
            "season": [2026, 2026, 2026, 2026],
            "week": [1, 1, 2, 2],
            "gameday": ["2026-09-09", "2026-09-13", "2026-09-20", "2026-09-17"],
        })
        assert str(week_start_date(2026, 2, schedule)) == "2026-09-17"

    def test_unknown_week_is_none(self):
        schedule = pd.DataFrame({"season": [2026], "week": [1], "gameday": ["2026-09-09"]})
        assert week_start_date(2026, 5, schedule) is None


class TestInjuryNameCollisions:
    """The 2026 week 2 dry run zeroed four healthy players — David Montgomery,
    Travis Etienne, Justice Hill, Barion Brown — because the initial + last-name
    fallback matched someone on ANOTHER team's reserve list."""

    def _analyzer(self):
        overrides = {
            "D.J. Montgomery": {"status": "OUT", "reason": "RES list", "team": "IND",
                                "position": "WR", "player_id": "00-0001"},
            "Trevor Etienne": {"status": "OUT", "reason": "RES list", "team": "NO",
                               "position": "RB", "player_id": "00-0002"},
        }
        return InjuryStatusAnalyzer(overrides, {})

    def test_initial_and_last_name_do_not_cross_teams(self):
        a = self._analyzer()
        assert not a.should_zero_out_player("David Montgomery", team="HOU")

    def test_same_team_different_player_id_is_not_a_match(self):
        # Brothers on one roster: the name fallback lines up, the IDs don't.
        a = self._analyzer()
        assert not a.should_zero_out_player("Travis Etienne", team="NO", player_id="00-0009")

    def test_player_id_match_is_exact(self):
        a = self._analyzer()
        assert a.should_zero_out_player("Dee Jay Montgomery", team="IND", player_id="00-0001")

    def test_name_fallback_still_works_within_a_team(self):
        a = self._analyzer()
        assert a.should_zero_out_player("DJ Montgomery", team="IND")

    def test_unknown_team_keeps_the_old_behaviour(self):
        a = self._analyzer()
        assert a.should_zero_out_player("D.J. Montgomery")


class TestRookieRole:
    """Rookie-prior rows get the depth-rank discount veterans already get."""

    def _projector(self):
        charts = pd.DataFrame({
            "team": ["LA", "LA", "LA", "SEA"], "pos_abb": ["QB", "QB", "QB", "RB"],
            "pos_rank": [1, 2, 3, 1],
            "player_name": ["Matthew Stafford", "Backup Vet", "Ty Simpson", "Jadarian Price"],
        })
        projector = Projector.__new__(Projector)   # just the depth logic, no model
        projector.depth_analyzer = DepthChartAnalyzer(charts)
        return projector

    @staticmethod
    def _prior():
        return {"fanduel_fantasy_points": 11.2, "floor": 3.7, "projection_median": 11.2,
                "ceiling": 19.7, "passing_yards": 180.0, "prediction_type": "rookie_prior"}

    def test_third_string_qb_is_discounted_like_a_veteran(self):
        pred = self._prior()
        self._projector()._apply_rookie_role(pred, "Ty Simpson", "QB", "LA")
        assert pred["fanduel_fantasy_points"] == pytest.approx(11.2 * 0.15, abs=0.01)
        assert pred["ceiling"] == pytest.approx(19.7 * 0.15, abs=0.01)
        assert pred["passing_yards"] == pytest.approx(27.0)

    def test_starter_is_untouched(self):
        pred = self._prior()
        self._projector()._apply_rookie_role(pred, "Jadarian Price", "RB", "SEA")
        assert pred["fanduel_fantasy_points"] == 11.2


class TestProjectableStatus:
    def test_active_players_are_projected(self):
        assert projectable_status("ACT", 2, 2)

    def test_off_roster_statuses_are_not(self):
        for status in ("DEV", "CUT", "EXE", "RES"):
            assert not projectable_status(status, 1, 2)

    def test_last_weeks_gameday_inactive_is_not_this_weeks(self):
        assert projectable_status("INA", 1, 2)

    def test_inactive_for_this_very_week_stays_off(self):
        assert not projectable_status("INA", 2, 2)
        assert not projectable_status("INA", None, 2)


class TestInjuryStatusAnalyzer:
    def _analyzer(self):
        overrides = {
            "Star Runner": {"status": "OUT", "reason": "knee"},
            "Odell Beckham Jr.": {"status": "DOUBTFUL", "reason": "hamstring"},
        }
        backups = {
            "Backup Runner": {"replacing": "Star Runner", "reason": "knee"},
        }
        return InjuryStatusAnalyzer(overrides, backups)

    def test_zero_out_direct_match(self):
        assert self._analyzer().should_zero_out_player("Star Runner") is True

    def test_zero_out_suffix_normalization(self):
        # "Odell Beckham" should match "Odell Beckham Jr."
        assert self._analyzer().should_zero_out_player("Odell Beckham") is True

    def test_healthy_player_not_zeroed(self):
        assert self._analyzer().should_zero_out_player("Healthy Guy") is False

    def test_backup_boost(self):
        analyzer = self._analyzer()
        assert analyzer.should_boost_backup("Backup Runner") is True
        assert analyzer.should_boost_backup("Star Runner") is False

    def test_adjustment_info(self):
        info = self._analyzer().get_adjustment_info("Star Runner")
        assert info["type"] == "injury_zero"
        info = self._analyzer().get_adjustment_info("Backup Runner")
        assert info["type"] == "backup_boost"
        assert info["replacing"] == "Star Runner"


class TestRosterHelpers:
    def test_dedupe_keeps_most_recent_week(self):
        rosters = pd.DataFrame({
            "player_name": ["A", "A", "B"],
            "week": [1, 5, 2],
            "team": ["DAL", "PHI", "NYG"],
        })
        out = dedupe_rosters(rosters)
        assert len(out) == 2
        assert out[out["player_name"] == "A"]["team"].iloc[0] == "PHI"

    def test_dedupe_creates_player_name_from_full_name(self):
        rosters = pd.DataFrame({"full_name": ["A"], "week": [1]})
        out = dedupe_rosters(rosters)
        assert out["player_name"].tolist() == ["A"]

    def test_extract_team_column_variants(self):
        assert _extract_team(pd.Series({"team": "DAL"})) == "DAL"
        assert _extract_team(pd.Series({"club_code": "PHI"})) == "PHI"
        assert _extract_team(pd.Series({"team": None, "recent_team": "NYG"})) == "NYG"
        assert _extract_team(pd.Series({"other": 1})) == "Unknown"
