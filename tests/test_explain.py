"""Unit tests for the player vetting layer (synthetic)."""
import numpy as np
import pandas as pd
import pytest

from nfl_projections import explain as ex


class TestPlayerCV:
    def test_volume_lowers_predicted_volatility(self):
        # The finding: at the same projection, the higher-volume player is the
        # steadier one. Volume, not the deep-threat archetype.
        assert ex.player_cv("RB", 12, 20) < ex.player_cv("RB", 12, 6)
        assert ex.player_cv("WR", 12, 9) < ex.player_cv("WR", 12, 3)

    def test_an_unknown_workload_falls_back_to_the_position(self):
        assert ex.player_cv("RB", 12, np.nan) == 0.72
        assert ex.player_cv("QB", 18, None) == 0.46

    def test_an_unknown_position_is_not_guessed(self):
        assert np.isnan(ex.player_cv("D", 8, 0))
        assert np.isnan(ex.player_cv("RB", 0, 10))
        assert np.isnan(ex.player_cv("RB", np.nan, 10))

    def test_it_is_clamped(self):
        lo, hi = ex._CV_CLAMP
        for proj, opp in [(0.5, 0), (1, 50), (60, 1), (40, 40)]:
            cv = ex.player_cv("RB", proj, opp)
            assert lo <= cv <= hi


class TestTypicalWeek:
    def test_the_typical_week_is_below_the_projection(self):
        # A fantasy score is right-skewed, so the booms pull the mean above the
        # median. Measured: low-volume RB 0.752, workhorse 0.923.
        assert ex.typical_week(16, "RB", 20) < 16
        assert ex.typical_week(9, "RB", 5) < 9

    def test_a_workhorse_keeps_more_of_his_projection(self):
        workhorse = ex.typical_week(14, "RB", 21) / 14
        split = ex.typical_week(14, "RB", 6) / 14
        assert workhorse > split
        assert 0.85 < workhorse < 0.97
        assert 0.70 < split < 0.85

    def test_upside_is_above_the_projection_and_ordered(self):
        assert ex.upside(12, "WR", 4) > 12
        # the volatile one has the bigger ceiling off the same projection
        assert ex.upside(12, "WR", 3) > ex.upside(12, "WR", 10)

    def test_quantiles_are_ordered(self):
        assert (ex.typical_week(12, "RB", 12)
                < ex.upside(12, "RB", 12, q=0.70)
                < ex.upside(12, "RB", 12, q=0.95))

    def test_it_does_not_need_scipy(self, monkeypatch):
        """The whole point of hand-rolling the inverse gamma.

        CI keeps its environment light and the API image installs this package
        with --no-deps, so a scipy import here is a module that fails to load
        in two of the three places it runs. It did: 21 tests in this file went
        down with ModuleNotFoundError on the first push.
        """
        import builtins

        real_import = builtins.__import__

        def no_scipy(name, *args, **kwargs):
            if name.split(".")[0] == "scipy":
                raise ImportError("no scipy in this environment")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_scipy)
        assert 0.70 < ex.typical_week(14, "RB", 6) / 14 < 0.85
        assert ex.upside(12, "WR", 4) > 12

    def test_the_quantiles_match_scipy_where_it_is_installed(self):
        """Pins the hand-rolled inverse gamma to the reference implementation."""
        stats = pytest.importorskip("scipy.stats")
        for cv in (0.25, 0.46, 0.72, 1.30):
            for q in (0.05, 0.5, 0.85, 0.99):
                k = 1.0 / cv ** 2
                reference = float(stats.gamma.ppf(q, k, scale=1.0 / k))
                assert ex._gamma_ratio(cv, q) == pytest.approx(reference, abs=1e-9)

    def test_a_degenerate_request_is_nan_not_a_crash(self):
        assert np.isnan(ex._gamma_ratio(float("nan"), 0.5))
        assert np.isnan(ex._gamma_ratio(0.0, 0.5))
        assert np.isnan(ex._gamma_ratio(0.5, 0.0))
        assert np.isnan(ex._gamma_ratio(0.5, 1.0))


class TestWind:
    def test_below_the_threshold_costs_nothing(self):
        assert ex.wind_adjustment("QB", 14.9) == 0.0
        assert ex.wind_adjustment("WR", 0) == 0.0

    def test_the_passing_game_pays_and_the_backs_do_not(self):
        assert ex.wind_adjustment("QB", 20) < -2.0
        assert ex.wind_adjustment("WR", 20) < -1.0
        assert abs(ex.wind_adjustment("RB", 20)) < 0.1

    def test_it_is_inert_without_a_reading(self):
        # nflverse has no forward-looking wind, so this must never guess.
        assert ex.wind_adjustment("QB", None) == 0.0
        assert ex.wind_adjustment("QB", np.nan) == 0.0
        assert ex.wind_adjustment("QB", "breezy") == 0.0


class TestExplain:
    @staticmethod
    def pool():
        return pd.DataFrame({
            "player_id": ["rb1", "rb2", "wr1", "qb1"],
            "player_name": ["Workhorse", "Splitter", "Deep Threat", "Passer"],
            "position": ["RB", "RB", "WR", "QB"],
            "team": ["BUF", "NYJ", "BUF", "BUF"],
            "lineup_points": [16.0, 9.0, 12.0, 19.0],
        })

    @staticmethod
    def opportunity():
        return pd.DataFrame({
            "player_id": ["rb1", "rb2", "wr1", "qb1"],
            "opp_per_game": [21.0, 6.0, 4.0, 3.0],
            "opp_share": [0.62, 0.19, 0.18, 0.08],
            "fp_per_game": [15.0, 8.0, 9.0, 18.0],
            "prior_games": [6, 6, 6, 6],
        })

    def test_it_adds_the_vetting_columns(self):
        out = ex.explain(self.pool(), opportunity=self.opportunity())
        for col in ("projected", "typical_week", "upside", "volatility",
                    "opp_per_game", "vs_form", "why"):
            assert col in out.columns
        assert len(out) == 4

    def test_a_split_back_is_flagged_and_a_workhorse_is_not(self):
        out = ex.explain(self.pool(), opportunity=self.opportunity()).set_index("player_name")
        assert "split" in out.loc["Splitter", "why"]
        assert "split" not in out.loc["Workhorse", "why"]
        assert "steady" in out.loc["Workhorse", "why"]

    def test_it_reports_where_the_number_sits_against_form(self):
        out = ex.explain(self.pool(), opportunity=self.opportunity()).set_index("player_name")
        # 12.0 projected against 9.0 trailing
        assert abs(out.loc["Deep Threat", "vs_form"] - 3.0) < 1e-9
        assert "above his trailing form" in out.loc["Deep Threat", "why"]

    def test_it_survives_with_no_context_at_all(self):
        # A missing frame must cost its own columns and never break a build.
        out = ex.explain(self.pool())
        assert len(out) == 4
        assert out["opp_per_game"].isna().all()
        assert out["typical_week"].notna().all()   # position fallback still works

    def test_weather_marks_a_dome_and_prices_a_gale(self):
        sched = pd.DataFrame({
            "season": [2026, 2026], "week": [9, 9],
            "home_team": ["BUF", "DET"], "away_team": ["NYJ", "GB"],
            "roof": ["outdoors", "dome"], "wind": [22.0, np.nan],
        })
        out = ex.explain(self.pool(), opportunity=self.opportunity(),
                         schedule=sched, season=2026, week=9).set_index("player_name")
        assert out.loc["Passer", "wind"] == 22.0
        assert out.loc["Passer", "wind_points"] < -2.0
        assert "22 mph wind" in out.loc["Passer", "why"]
        # the back in the same game is told he is unaffected
        assert abs(out.loc["Workhorse", "wind_points"]) < 0.1

    def test_a_week_with_no_schedule_row_leaves_weather_blank(self):
        sched = pd.DataFrame({
            "season": [2026], "week": [1], "home_team": ["BUF"],
            "away_team": ["NYJ"], "roof": ["outdoors"], "wind": [30.0],
        })
        out = ex.explain(self.pool(), schedule=sched, season=2026, week=9)
        assert out["wind"].isna().all()
        assert (out["wind_points"] == 0.0).all()

    def test_vet_returns_a_readable_report(self):
        out, text = ex.vet(self.pool(), opportunity=self.opportunity())
        assert "Workhorse" in text
        assert "opportunities/game" in text
        assert "lineup projection" in text
        assert len(out) == 4

    def test_it_does_not_mutate_the_pool(self):
        pool = self.pool()
        before = pool.copy()
        ex.explain(pool, opportunity=self.opportunity())
        pd.testing.assert_frame_equal(pool, before)

    def test_it_reads_an_alternative_points_column(self):
        pool = self.pool().rename(columns={"lineup_points": "fanduel_fantasy_points"})
        out = ex.explain(pool, opportunity=self.opportunity())
        assert out["projected"].notna().all()

    def test_duplicate_column_labels_do_not_break_it(self):
        # A FanDuel-merged frame has both "Position" and "position"; renaming
        # one leaves two columns of the same name, and indexing one then
        # returns a DataFrame rather than a Series.
        pool = self.pool()
        pool = pd.concat([pool, pool[["position"]]], axis=1)   # two "position"
        assert pool.columns.duplicated().any()
        out = ex.explain(pool, opportunity=self.opportunity())
        assert len(out) == 4
        assert out["typical_week"].notna().all()

    def test_a_defence_keeps_the_fanduel_position(self):
        # The board projects no defences, so `position` is null for them.
        pool = pd.DataFrame({
            "player_id": [None], "player_name": ["Vikings"],
            "position": [np.nan], "Position": ["D"], "team": ["MIN"],
            "lineup_points": [8.8],
        })
        out = ex.explain(pool)
        assert out.loc[0, "position"] == "D"

    def test_the_form_flag_is_relative_not_absolute(self):
        # fp_per_game counts only games played, so every rosterable player
        # sits above his projection; a flat bar fires on all nine.
        pool = pd.DataFrame({
            "player_id": ["a", "b"], "player_name": ["Big", "Small"],
            "position": ["RB", "RB"], "team": ["X", "Y"],
            "lineup_points": [20.0, 5.0],
        })
        opp = pd.DataFrame({
            "player_id": ["a", "b"], "opp_per_game": [20.0, 18.0],
            "opp_share": [0.5, 0.5], "fp_per_game": [24.0, 15.0],
            "prior_games": [6, 6],
        })
        out = ex.explain(pool, opportunity=opp).set_index("player_name")
        assert "trailing form" not in out.loc["Big", "why"]      # 4 of 24
        assert "trailing form" in out.loc["Small", "why"]        # 10 of 15

    def test_the_report_does_not_sum_the_medians(self):
        out, text = ex.vet(self.pool(), opportunity=self.opportunity())
        assert "lineup projection" in text
        assert "sum of typical weeks" not in text
        assert "skew cancels" in text

    @staticmethod
    def fpa_frame():
        return pd.DataFrame({
            "team": ["BUF", "NYJ", "BUF", "BUF"],
            "position": ["WR", "RB", "QB", "TE"],
            "fpa_grade": [78.0, 50.0, 30.0, 60.0],
            "fpa_band": ["very soft", "average", "very tough", "soft"],
            "fpa_allowed": [31.0, 20.0, 12.0, 9.0],
            "fpa_adjusted": [1.8, 0.0, -1.3, 0.4],
            "fpa_weeks": [6, 6, 6, 6],
            "fpa_outlier_week": [np.nan, np.nan, 8.0, np.nan],
        })

    def test_a_receiver_gets_a_matchup_band_of_his_own(self):
        # Backs get the trench grade; pass-catchers need the FPA equivalent.
        out = ex.explain(self.pool(), opportunity=self.opportunity(),
                         fpa=self.fpa_frame()).set_index("player_name")
        assert out.loc["Deep Threat", "fpa_grade"] == 78.0
        assert "very soft vs WRs" in out.loc["Deep Threat", "why"]

    def test_a_skewed_rating_says_so(self):
        out = ex.explain(self.pool(), opportunity=self.opportunity(),
                         fpa=self.fpa_frame()).set_index("player_name")
        why = out.loc["Passer", "why"]
        assert "very tough vs QBs" in why
        assert "week 8 skews it" in why

    def test_the_fpa_join_is_keyed_on_position_too(self):
        # BUF's receiver grade must not land on BUF's quarterback.
        out = ex.explain(self.pool(), fpa=self.fpa_frame()).set_index("player_name")
        assert out.loc["Deep Threat", "fpa_grade"] == 78.0
        assert out.loc["Passer", "fpa_grade"] == 30.0
        # the NYJ back gets his own row, the BUF back has none
        assert out.loc["Splitter", "fpa_grade"] == 50.0
        assert np.isnan(out.loc["Workhorse", "fpa_grade"])

    def test_no_fpa_frame_costs_only_those_columns(self):
        out = ex.explain(self.pool(), opportunity=self.opportunity())
        assert out["fpa_grade"].isna().all()
        assert out["typical_week"].notna().all()
