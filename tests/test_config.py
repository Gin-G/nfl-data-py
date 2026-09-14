from datetime import date

from nfl_projections import config


class TestCurrentSeason:
    def test_rolls_over_the_thursday_after_labor_day(self):
        # Labor Day 2026 is September 7, so the season turns on the 10th.
        assert config.current_season(date(2026, 9, 9)) == 2025
        assert config.current_season(date(2026, 9, 10)) == 2026

    def test_labor_day_on_the_first(self):
        # 2025: Labor Day is September 1, season from the 4th.
        assert config.current_season(date(2025, 9, 3)) == 2024
        assert config.current_season(date(2025, 9, 4)) == 2025

    def test_playoffs_and_offseason_belong_to_last_year(self):
        assert config.current_season(date(2027, 2, 7)) == 2026
        assert config.current_season(date(2027, 8, 15)) == 2026

    def test_dataset_seasons_run_through_the_current_season(self):
        assert config.SEASONS[-1] == config.CURRENT_SEASON
        assert config.SEASONS == list(range(2018, config.CURRENT_SEASON + 1))
