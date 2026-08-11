"""Unit tests for the recent-form blend."""
import numpy as np
import pandas as pd
import pytest

from nfl_projections import blend


class TestModelWeight:
    def test_known_positions_use_their_fitted_weight(self):
        assert blend.model_weight("QB") == pytest.approx(0.5)
        assert blend.model_weight("TE") == pytest.approx(0.6)

    def test_unknown_position_falls_back(self):
        assert blend.model_weight("K") == pytest.approx(blend.DEFAULT_WEIGHT)

    def test_custom_weights_override(self):
        assert blend.model_weight("QB", {"QB": 0.25}) == pytest.approx(0.25)

    def test_every_weight_is_a_real_blend(self):
        # A weight of 1.0 would be a no-op, 0.0 would discard the model
        for pos, w in blend.DEFAULT_MODEL_WEIGHTS.items():
            assert 0.0 < w < 1.0, pos


class TestBlendValue:
    def test_blends_toward_recent_form(self):
        # QB weight 0.5: 0.5*20 + 0.5*10 = 15
        assert blend.blend_value(20.0, 10.0, "QB") == pytest.approx(15.0)

    def test_form_above_projection_pulls_up(self):
        assert blend.blend_value(10.0, 20.0, "QB") == pytest.approx(15.0)

    def test_missing_form_leaves_the_projection_alone(self):
        # A rookie with no NFL history has no form to blend
        assert blend.blend_value(12.0, None, "RB") == pytest.approx(12.0)
        assert blend.blend_value(12.0, float("nan"), "RB") == pytest.approx(12.0)

    def test_agreement_is_a_no_op(self):
        assert blend.blend_value(15.0, 15.0, "WR") == pytest.approx(15.0)

    def test_separates_players_the_model_flattened(self):
        # The reported symptom: model says 16.95 vs 16.94, but their recent
        # form differs by 5 points. The blend restores a real gap.
        allen = blend.blend_value(16.95, 23.4, "QB")
        goff = blend.blend_value(16.94, 18.9, "QB")
        assert allen - goff > 1.5


class TestPreseasonMode:
    def test_preseason_uses_the_season_average_column(self):
        assert blend.form_col_for_mode(preseason=True) == blend.SEASON_FORM_COL
        assert blend.form_col_for_mode(preseason=False) == blend.RECENT_FORM_COL

    def test_preseason_leans_harder_on_history(self):
        # No current form exists before a season starts, and last season's full
        # average predicts better than its tail
        for pos in blend.PRESEASON_MODEL_WEIGHTS:
            assert (blend.PRESEASON_MODEL_WEIGHTS[pos]
                    <= blend.DEFAULT_MODEL_WEIGHTS[pos])

    def test_preseason_weights_are_real_blends(self):
        for pos, w in blend.PRESEASON_MODEL_WEIGHTS.items():
            assert 0.0 < w < 1.0, pos

    def test_unknown_position_uses_the_matching_mode_default(self):
        w = blend.model_weight("FB", blend.PRESEASON_MODEL_WEIGHTS)
        assert w == pytest.approx(blend.PRESEASON_WEIGHT)

    def test_reads_the_season_average_row(self):
        rows = pd.DataFrame({blend.SEASON_FORM_COL: [14.0],
                             blend.RECENT_FORM_COL: [9.0]})
        assert blend.recent_form_from_rows(rows, preseason=True).iloc[0] == 14.0
        assert blend.recent_form_from_rows(rows, preseason=False).iloc[0] == 9.0


class TestBlendComponents:
    def _rows(self, **cols):
        base = {"passing_yards_roll5": [0.5], "rushing_yards_roll5": [0.1],
                "receiving_yards_roll5": [0.0], "receptions_roll5": [0.0]}
        base.update({k: [v] for k, v in cols.items()})
        return pd.DataFrame(base)

    def test_component_blends_toward_its_own_trailing_average(self):
        # the reported symptom: model says 12 rushing yards, player gains 0.1
        out = blend.blend_components(
            {"fanduel_fantasy_points": 18.0, "rushing_yards": 12.0}, self._rows())
        assert out["rushing_yards"] == pytest.approx(6.05)   # 0.5*12 + 0.5*0.1

    def test_points_are_left_for_the_caller(self):
        out = blend.blend_components(
            {"fanduel_fantasy_points": 18.0, "rushing_yards": 12.0}, self._rows())
        assert out["fanduel_fantasy_points"] == 18.0

    def test_stats_without_a_trailing_column_use_the_fallback_ratio(self):
        # TDs have no rolling feature; they follow the points blend instead
        out = blend.blend_components(
            {"fanduel_fantasy_points": 10.0, "rushing_tds": 0.4},
            self._rows(), fallback_ratio=0.5)
        assert out["rushing_tds"] == pytest.approx(0.2)

    def test_never_returns_a_negative_component(self):
        out = blend.blend_components(
            {"fanduel_fantasy_points": 1.0, "rushing_yards": 0.2},
            self._rows(rushing_yards_roll5=-3.0))
        assert out["rushing_yards"] == 0.0

    def test_missing_trailing_value_leaves_the_component_alone(self):
        out = blend.blend_components(
            {"fanduel_fantasy_points": 10.0, "rushing_yards": 12.0},
            self._rows(rushing_yards_roll5=float("nan")))
        assert out["rushing_yards"] == pytest.approx(12.0)

    def test_a_real_rusher_is_not_dragged_down(self):
        # blending must not flatten everyone: a QB whose trailing rushing is
        # high keeps a high projection
        out = blend.blend_components(
            {"fanduel_fantasy_points": 22.0, "rushing_yards": 34.0},
            self._rows(rushing_yards_roll5=38.0))
        assert out["rushing_yards"] == pytest.approx(36.0)

    def test_does_not_mutate_the_input(self):
        result = {"fanduel_fantasy_points": 18.0, "rushing_yards": 12.0}
        blend.blend_components(result, self._rows())
        assert result["rushing_yards"] == 12.0


class TestRecentFormFromRows:
    def test_reads_the_rolling_column(self):
        rows = pd.DataFrame({blend.RECENT_FORM_COL: [12.5, 8.0]})
        assert blend.recent_form_from_rows(rows).tolist() == [12.5, 8.0]

    def test_missing_column_yields_nan(self):
        out = blend.recent_form_from_rows(pd.DataFrame({"other": [1]}))
        assert out.isna().all()

    def test_non_numeric_becomes_nan(self):
        rows = pd.DataFrame({blend.RECENT_FORM_COL: ["n/a"]})
        assert blend.recent_form_from_rows(rows).isna().all()


class TestBlendFrame:
    def _frame(self):
        return pd.DataFrame({
            "player_name": ["QB A", "RB B", "Rookie C"],
            "position": ["QB", "RB", "WR"],
            "fanduel_fantasy_points": [20.0, 10.0, 8.0],
            "floor": [10.0, 5.0, 4.0],
            "ceiling": [30.0, 15.0, 12.0],
        })

    def test_blends_each_row_by_its_position(self):
        # QB 0.5*20+0.5*10=15, RB 0.5*10+0.5*20=15, rookie has no form
        out = blend.blend_frame(self._frame(), [10.0, 20.0, np.nan])
        assert out["fanduel_fantasy_points"].tolist() == pytest.approx([15.0, 15.0, 8.0])

    def test_scale_cols_travel_with_the_projection(self):
        out = blend.blend_frame(self._frame(), [10.0, 20.0, np.nan],
                                scale_cols=["floor", "ceiling"])
        # QB blended 20 -> 15, ratio 0.75
        assert out["floor"].iloc[0] == pytest.approx(7.5)
        assert out["ceiling"].iloc[0] == pytest.approx(22.5)
        # rookie untouched
        assert out["ceiling"].iloc[2] == pytest.approx(12.0)

    def test_does_not_mutate_the_input(self):
        frame = self._frame()
        blend.blend_frame(frame, [10.0, 20.0, np.nan])
        assert frame["fanduel_fantasy_points"].tolist() == [20.0, 10.0, 8.0]

    def test_zero_projection_does_not_blow_up_the_ratio(self):
        frame = pd.DataFrame({
            "position": ["RB"], "fanduel_fantasy_points": [0.0], "ceiling": [0.0],
        })
        out = blend.blend_frame(frame, [6.0], scale_cols=["ceiling"])
        assert np.isfinite(out["fanduel_fantasy_points"]).all()
        assert np.isfinite(out["ceiling"]).all()

    def test_missing_points_column_is_returned_unchanged(self):
        frame = pd.DataFrame({"position": ["RB"], "something_else": [1.0]})
        assert blend.blend_frame(frame, [5.0]).equals(frame)
