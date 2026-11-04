import pandas as pd

from nfl_projections import injuries


class TestParseNflverseInjuries:
    def _reports(self):
        return pd.DataFrame({
            "season": [2025] * 4,
            "week": [3, 3, 3, 2],
            "full_name": ["Star Back", "Iffy WR", "Healthy TE", "Wrong Week"],
            "position": ["RB", "WR", "TE", "QB"],
            "team": ["DAL", "PHI", "NYG", "KC"],
            "report_status": ["Out", "Questionable", "", "Out"],
            "practice_status": ["Did Not Participate", "Limited", "Full", "DNP"],
            "report_primary_injury": ["Knee", "Ankle", "Rest", "Shoulder"],
        })

    def test_filters_to_week_and_maps_status(self):
        out = injuries.parse_nflverse_injuries(self._reports(), week=3)
        assert out["Star Back"]["status"] == "OUT"
        assert out["Iffy WR"]["status"] == "QUESTIONABLE"
        assert "Wrong Week" not in out  # different week filtered out

    def test_blank_status_infers_questionable_from_practice(self):
        reports = pd.DataFrame({
            "week": [1],
            "full_name": ["Limited Guy"],
            "position": ["RB"],
            "team": ["DAL"],
            "report_status": [""],
            "practice_status": ["Limited Participation"],
            "report_primary_injury": ["Hamstring"],
        })
        out = injuries.parse_nflverse_injuries(reports, week=1)
        assert out["Limited Guy"]["status"] == "QUESTIONABLE"

    def test_full_practice_no_status_is_dropped(self):
        out = injuries.parse_nflverse_injuries(self._reports(), week=3)
        # Healthy TE: blank status + Full practice -> not an injury
        assert "Healthy TE" not in out


class TestBuildOverrides:
    def _depth_charts(self):
        return pd.DataFrame({
            "team": ["DAL", "DAL", "PHI"],
            "pos_abb": ["RB", "RB", "WR"],
            "pos_rank": [1, 2, 1],
            "player_name": ["Star Back", "Backup Back", "Iffy WR"],
        })

    def _rosters(self):
        return pd.DataFrame({
            "player_name": ["Star Back", "Backup Back", "Iffy WR", "IR Guy"],
            "team": ["DAL", "DAL", "PHI", "NYG"],
            "position": ["RB", "RB", "WR", "TE"],
            "status": ["ACT", "ACT", "ACT", "IR"],
        })

    def test_out_player_zeroed_and_backup_elevated(self):
        injuries_dict = {
            "Star Back": {"status": "OUT", "injury_type": "Knee",
                          "position": "RB", "team": "DAL"},
        }
        overrides, backups = injuries.build_overrides(
            injuries_dict, self._rosters(), self._depth_charts()
        )
        assert overrides["Star Back"]["status"] == "OUT"
        assert "Backup Back" in backups
        assert backups["Backup Back"]["replacing"] == "Star Back"

    def test_ir_players_folded_in_from_rosters(self):
        overrides, _ = injuries.build_overrides(
            {}, self._rosters(), self._depth_charts()
        )
        # IR Guy has roster status IR -> zeroed out even with no report
        assert "IR Guy" in overrides
        assert overrides["IR Guy"]["status"] == "OUT"

    def test_overrides_keep_the_player_id(self):
        # The Projector matches on ID first; a name-only override is what let
        # IND's D.J. Montgomery zero out HOU's David Montgomery.
        rosters = self._rosters().assign(gsis_id=["00-01", "00-02", "00-03", "00-04"])
        overrides, _ = injuries.build_overrides({}, rosters, self._depth_charts())
        assert overrides["IR Guy"]["player_id"] == "00-04"

    def test_report_ids_flow_through(self):
        reports = pd.DataFrame({
            "week": [3], "full_name": ["Star Back"], "gsis_id": ["00-0035685"],
            "position": ["RB"], "team": ["DAL"], "report_status": ["Out"],
            "practice_status": ["DNP"], "report_primary_injury": ["Knee"],
        })
        parsed = injuries.parse_nflverse_injuries(reports, week=3)
        overrides, _ = injuries.build_overrides(parsed, self._rosters(), self._depth_charts())
        assert overrides["Star Back"]["player_id"] == "00-0035685"


class TestInjuryReportCoverage:
    """What the feed HELD, which the parsed output cannot say.

    A week nobody has filed for and a week where every club filed a clean
    report both parse to ``{}``. Clubs do not file until Wednesday, so a
    Tuesday-night board reads an empty frame and projects a healthy league --
    and nothing in the projection afterwards distinguishes that from a league
    that really was healthy.
    """

    def _reports(self):
        return pd.DataFrame({
            "week": [5, 6, 7, 8, 8],
            "full_name": ["a", "b", "c", "d", "e"],
            "report_status": ["Out"] * 5,
        })

    def test_a_week_nobody_filed_for_is_not_available(self):
        cov = injuries.injury_report_coverage(self._reports(), week=9)
        assert cov["report_available"] is False
        assert cov["report_rows_for_week"] == 0
        assert cov["report_latest_week"] == 8      # ...but the feed is not empty
        assert cov["report_weeks"] == [5, 6, 7, 8]

    def test_a_filed_week_is_available(self):
        cov = injuries.injury_report_coverage(self._reports(), week=8)
        assert cov["report_available"] is True
        assert cov["report_rows_for_week"] == 2

    def test_no_feed_at_all(self):
        cov = injuries.injury_report_coverage(None, week=9)
        assert cov == {"report_weeks": [], "report_latest_week": None,
                       "report_rows_for_week": 0, "report_available": False}

    def test_a_frame_without_a_week_column(self):
        cov = injuries.injury_report_coverage(
            pd.DataFrame({"full_name": ["a"]}), week=9)
        assert cov["report_available"] is False


class TestBuildOverridesDetailed:
    def test_counts_separate_the_report_from_the_ir_list(self):
        rosters = pd.DataFrame({
            "player_name": ["Star Back", "Backup Back", "IR Guy"],
            "team": ["DAL", "DAL", "NYG"],
            "position": ["RB", "RB", "TE"],
            "status": ["ACT", "ACT", "IR"],
        })
        depth = pd.DataFrame({
            "team": ["DAL", "DAL"], "pos_abb": ["RB", "RB"], "pos_rank": [1, 2],
            "player_name": ["Star Back", "Backup Back"],
        })
        report = {"Star Back": {"status": "OUT", "injury_type": "Knee",
                                "position": "RB", "team": "DAL", "player_id": "1"}}
        overrides, backups, counts = injuries.build_overrides_detailed(
            report, rosters, depth)
        assert counts == {"report_players": 1, "ir_players": 1, "out": 2,
                          "backups_elevated": 1, "questionable": 0}
        # Two players out, but only one of them because anybody said so.
        assert set(overrides) == {"Star Back", "IR Guy"}
        assert set(backups) == {"Backup Back"}

    def test_nothing_from_either_source_still_counts(self):
        empty_roster = pd.DataFrame(
            columns=["player_name", "team", "position", "status"])
        overrides, backups, counts = injuries.build_overrides_detailed(
            {}, empty_roster, pd.DataFrame(columns=["team", "pos_abb", "pos_rank",
                                                    "player_name"]))
        assert (overrides, backups) == ({}, {})
        assert counts["report_players"] == 0 and counts["ir_players"] == 0

    def test_the_two_tuple_wrapper_still_works(self):
        """build_overrides is called from the projector and the CLI."""
        rosters = pd.DataFrame({"player_name": ["IR Guy"], "team": ["NYG"],
                                "position": ["TE"], "status": ["IR"]})
        depth = pd.DataFrame(columns=["team", "pos_abb", "pos_rank", "player_name"])
        result = injuries.build_overrides({}, rosters, depth)
        assert len(result) == 2
        assert set(result[0]) == {"IR Guy"}
