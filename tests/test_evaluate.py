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


class TestComponentFantasyPoints:
    def test_recomputes_from_components(self):
        # 1 pass TD (4) + 100 pass yds (4) + 50 rush yds (5) + 1 INT (-1) = 12.0
        pred = pd.DataFrame({
            "passing_yards": [100.0], "passing_tds": [1.0],
            "passing_interceptions": [1.0], "rushing_yards": [50.0],
            "rushing_tds": [0.0], "receptions": [0.0],
            "receiving_yards": [0.0], "receiving_tds": [0.0],
        })
        out = evaluate.component_fantasy_points(pred)
        assert round(float(out.iloc[0]), 2) == 12.0

    def test_missing_columns_treated_as_zero(self):
        pred = pd.DataFrame({"receiving_yards": [100.0], "receptions": [8.0]})
        # 100 rec yds (10) + 8 rec (4) + 100-yd bonus (3) = 17.0
        out = evaluate.component_fantasy_points(pred)
        assert round(float(out.iloc[0]), 2) == 17.0

    def test_no_component_columns_keeps_row_count(self):
        # A single-target model predicts only fanduel_fantasy_points: the
        # derived column must still have one row per prediction (all zeros).
        pred = pd.DataFrame({"fanduel_fantasy_points": [12.0, 8.0, 20.0]})
        out = evaluate.component_fantasy_points(pred)
        assert len(out) == 3
        assert out.tolist() == [0.0, 0.0, 0.0]


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
