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
