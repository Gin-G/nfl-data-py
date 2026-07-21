import pandas as pd
import pytest

from nfl_projections import evaluate


class TestLoadExternalProjections:
    def test_fantasypros_style_columns(self, tmp_path):
        path = tmp_path / "fp.csv"
        pd.DataFrame({
            "Player": ["Bijan Robinson", "CeeDee Lamb"],
            "FPTS": [18.5, 16.2],
        }).to_csv(path, index=False)

        out = evaluate.load_external_projections(path)
        assert len(out) == 2
        assert out["external_points"].tolist() == [18.5, 16.2]
        assert out["normalized_name"].tolist() == ["bijan robinson", "ceedee lamb"]

    def test_generic_columns_with_week(self, tmp_path):
        path = tmp_path / "proj.csv"
        pd.DataFrame({
            "player_name": ["A Player"],
            "projected_points": [10.0],
            "week": [3],
        }).to_csv(path, index=False)

        out = evaluate.load_external_projections(path)
        assert out["week"].tolist() == [3]

    def test_unknown_columns_raise(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="player name"):
            evaluate.load_external_projections(path)


class TestCompareToExternal:
    def test_matches_by_normalized_name(self, capsys):
        ours = pd.DataFrame({
            "player_name": ["Odell Beckham Jr.", "Nobody Matches"],
            "predicted": [12.0, 5.0],
            "actual": [14.0, 3.0],
        })
        external = pd.DataFrame({
            "player_name": ["Odell Beckham"],
            "external_points": [11.0],
            "normalized_name": ["odell beckham"],
        })
        merged = evaluate.compare_to_external(ours, external)
        assert len(merged) == 1
        assert merged["external_points"].iloc[0] == 11.0
        out = capsys.readouterr().out
        assert "Ours vs actual" in out

    def test_predictions_csv_column_renamed(self):
        # A predictions CSV uses fanduel_fantasy_points, not "predicted"
        ours = pd.DataFrame({
            "player_name": ["Some Guy"],
            "fanduel_fantasy_points": [9.0],
        })
        external = pd.DataFrame({
            "player_name": ["Some Guy"],
            "external_points": [8.0],
            "normalized_name": ["some guy"],
        })
        merged = evaluate.compare_to_external(ours, external)
        assert merged["predicted"].iloc[0] == 9.0

    def test_no_overlap_returns_empty(self):
        ours = pd.DataFrame({"player_name": ["X"], "predicted": [1.0]})
        external = pd.DataFrame({
            "player_name": ["Y"], "external_points": [2.0], "normalized_name": ["y"],
        })
        assert evaluate.compare_to_external(ours, external).empty


class TestSummarize:
    def test_summary_stats(self, capsys):
        results = pd.DataFrame({
            "player_name": ["A", "B", "C", "D"],
            "position": ["QB", "QB", "RB", "RB"],
            "week": [1, 1, 1, 1],
            "predicted": [20.0, 15.0, 12.0, 8.0],
            "actual": [18.0, 17.0, 10.0, 9.0],
        })
        summary = evaluate.summarize(results)
        assert summary.loc["ALL", "n"] == 4
        assert summary.loc["ALL", "mae"] == 1.75  # (2+2+2+1)/4

    def test_empty_results(self):
        assert evaluate.summarize(pd.DataFrame()).empty
