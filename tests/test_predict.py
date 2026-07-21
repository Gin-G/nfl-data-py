import pandas as pd

from nfl_projections.predict import (
    DepthChartAnalyzer,
    InjuryStatusAnalyzer,
    _extract_team,
    dedupe_rosters,
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
