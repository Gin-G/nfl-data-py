"""Coverage ratings: the grade scale, the shrinkage, and the leakage guard."""
import numpy as np
import pandas as pd
import pytest

from nfl_projections import coverage


def charting(rows):
    """Minimal PFR-shaped charting frame."""
    return pd.DataFrame([
        {"season": s, "week": w, "game_type": "REG", "team": t,
         "opponent": o, "pfr_player_id": pid, "pfr_player_name": pid,
         "def_targets": tgt, "def_completions_allowed": comp,
         "def_adot": adot, "def_yards_allowed": yds}
        for s, w, t, o, pid, tgt, comp, adot, yds in rows])


def players(ids, position="CB"):
    return pd.DataFrame([{"pfr_player_id": i, "position": position,
                          "player_name": i} for i in ids])


def _frame(n_weeks=6, soft_comp=9, stout_comp=3, season=2026):
    rows = []
    for w in range(1, n_weeks + 1):
        rows.append((season, w, "AAA", "BBB", "soft", 10, soft_comp, 10.0, 120))
        rows.append((season, w, "AAA", "BBB", "stout", 10, stout_comp, 10.0, 40))
        rows.append((season, w, "CCC", "DDD", "mid1", 10, 6, 10.0, 80))
        rows.append((season, w, "CCC", "DDD", "mid2", 10, 6, 10.0, 80))
    return charting(rows)


def grades(frame, season=2026, week=None):
    cw = coverage.corner_weeks(frame, players(["soft", "stout", "mid1", "mid2"]))
    return coverage.corner_grades(cw, season, week)


def test_a_corner_who_allows_more_completions_grades_lower():
    g = grades(_frame()).set_index("pfr_player_id")
    assert g.loc["soft", "cover_grade"] < g.loc["stout", "cover_grade"]
    assert g.loc["soft", "cmp_pct_allowed"] == pytest.approx(0.9)
    assert g.loc["stout", "cmp_pct_allowed"] == pytest.approx(0.3)


def test_fifty_is_average_not_mediocre():
    """Two identical average corners both sit at the middle of the scale."""
    g = grades(_frame()).set_index("pfr_player_id")
    assert g.loc["mid1", "cover_grade"] == pytest.approx(50.0, abs=1e-6)
    assert g.loc["mid2", "cover_grade"] == pytest.approx(50.0, abs=1e-6)


def test_the_grade_is_shrunk_toward_the_league():
    """A small sample cannot reach its own raw rate."""
    g = grades(_frame(n_weeks=2)).set_index("pfr_player_id")
    raw, shrunk = g.loc["soft", "cmp_pct_allowed"], g.loc["soft", "shrunk_cmp_allowed"]
    league = g.loc["soft", "league_cmp_allowed"]
    assert raw > shrunk > league
    # 20 targets against SHRINK_TARGETS=60 keeps at most a quarter of his own
    assert shrunk - league < 0.25 * (raw - league) + 1e-9


def test_more_targets_move_the_grade_further_from_average():
    small = grades(_frame(n_weeks=2)).set_index("pfr_player_id")
    big = grades(_frame(n_weeks=16)).set_index("pfr_player_id")
    assert abs(big.loc["soft", "shrunk_cmp_allowed"] - big.loc["soft", "league_cmp_allowed"]) \
        > abs(small.loc["soft", "shrunk_cmp_allowed"] - small.loc["soft", "league_cmp_allowed"])


def test_the_week_being_graded_never_contributes():
    """Leakage guard: a corner torched in week 6 still grades on weeks 1-5."""
    rows = []
    for w in range(1, 6):
        rows.append((2026, w, "AAA", "BBB", "soft", 10, 5, 10.0, 60))
        rows.append((2026, w, "CCC", "DDD", "mid1", 10, 5, 10.0, 60))
    blown = rows + [(2026, 6, "AAA", "BBB", "soft", 40, 40, 10.0, 600)]
    before = grades(charting(rows), week=6).set_index("pfr_player_id")
    after = grades(charting(blown), week=6).set_index("pfr_player_id")
    assert before.loc["soft", "cover_grade"] == pytest.approx(
        after.loc["soft", "cover_grade"])


def test_last_season_carries_an_early_season_grade():
    prev = _frame(n_weeks=16, season=2025)
    g = grades(prev, season=2026, week=1).set_index("pfr_player_id")
    assert g.loc["soft", "cover_grade"] < g.loc["stout", "cover_grade"]
    assert g.loc["soft", "targets"] == 0        # none of it is this season's


def test_safeties_are_excluded():
    cw = coverage.corner_weeks(_frame(), players(["soft", "stout", "mid1", "mid2"],
                                                 position="S"))
    assert cw.empty


def test_the_weakest_corner_must_actually_play():
    """A corner graded off last season alone is not the one to attack."""
    rows = []
    for w in range(1, 17):          # last season: a dreadful backup
        rows.append((2025, w, "AAA", "BBB", "benched", 10, 10, 10.0, 150))
    for w in range(1, 6):           # this season: only these two play
        rows.append((2026, w, "AAA", "BBB", "soft", 10, 8, 10.0, 100))
        rows.append((2026, w, "AAA", "BBB", "stout", 10, 3, 10.0, 40))
    cw = coverage.corner_weeks(charting(rows),
                               players(["benched", "soft", "stout"]))
    cg = coverage.corner_grades(cw, 2026, 6)
    tc = coverage.team_coverage(cg)
    row = tc[tc.team == "AAA"].iloc[0]
    assert row.weakest_corner == "soft"
    assert row.cover_spread > 0


def test_bands_read_from_the_defences_side():
    assert coverage.band(20.0) == "pick on"
    assert coverage.band(50.0) == "average"
    assert coverage.band(80.0) == "avoid"
    assert coverage.band(float("nan")) == "unknown"


def test_matchups_carry_the_opposing_secondary():
    cg = grades(_frame())
    cg["team"] = cg.pfr_player_id.map(
        {"soft": "AAA", "stout": "AAA", "mid1": "CCC", "mid2": "CCC"})
    tc = coverage.team_coverage(cg)
    sch = pd.DataFrame([{"season": 2026, "week": 6, "home_team": "AAA",
                         "away_team": "CCC"}])
    mu = coverage.matchups_for_week(tc, sch, 2026, 6)
    assert len(mu) == 2
    ccc = mu[mu.team == "CCC"].iloc[0]
    assert ccc.opponent == "AAA"
    assert ccc.opp_weakest_corner == "soft"


def test_the_average_cannot_see_an_exploitable_side():
    """The reason `cover_spread` is reported next to the grade.

    AAA has one excellent corner and one dreadful one; CCC has two average
    ones. Their target-weighted averages are IDENTICAL, so the grade alone
    calls them the same secondary. Only the spread says one of them has a side
    worth attacking -- which is the part of a "pick on him" read that survives
    not knowing who covers whom.
    """
    cg = grades(_frame())
    cg["team"] = cg.pfr_player_id.map(
        {"soft": "AAA", "stout": "AAA", "mid1": "CCC", "mid2": "CCC"})
    tc = coverage.team_coverage(cg).set_index("team")
    assert tc.loc["AAA", "cover_rate"] == pytest.approx(tc.loc["CCC", "cover_rate"])
    assert tc.loc["AAA", "cover_grade"] == pytest.approx(tc.loc["CCC", "cover_grade"])
    assert tc.loc["AAA", "cover_spread"] > 0.2
    assert tc.loc["CCC", "cover_spread"] == pytest.approx(0.0)


def test_empty_charting_degrades_quietly():
    assert coverage.corner_weeks(pd.DataFrame()).empty
    assert coverage.corner_grades(pd.DataFrame(), 2026, 1).empty
    assert coverage.team_coverage(pd.DataFrame()).empty


def test_predict_does_not_import_coverage():
    """The measured decision: nothing here reaches the board."""
    import pathlib
    src = pathlib.Path(__file__).parent.parent / "nfl_projections"
    for name in ("predict.py", "blend.py", "roles.py", "model.py"):
        assert "coverage" not in (src / name).read_text()


class TestShellPlays:
    """The defence is read off the game id, because participation names only
    the offence."""

    def _part(self):
        return pd.DataFrame([
            {"nflverse_game_id": "2025_09_BAL_JAX", "possession_team": "BAL",
             "defense_coverage_type": "COVER_3",
             "defense_man_zone_type": "ZONE_COVERAGE"},
            {"nflverse_game_id": "2025_09_BAL_JAX", "possession_team": "JAX",
             "defense_coverage_type": "COVER_1",
             "defense_man_zone_type": "MAN_COVERAGE"},
            {"nflverse_game_id": "2025_09_BAL_JAX", "possession_team": "JAX",
             "defense_coverage_type": None, "defense_man_zone_type": ""},
        ])

    def test_the_defence_is_whoever_is_not_in_possession(self):
        sp = coverage.shell_plays(self._part())
        assert sp.defense.tolist() == ["JAX", "BAL", "BAL"]
        assert sp.season.tolist() == [2025, 2025, 2025]
        assert sp.week.tolist() == [9, 9, 9]

    def test_a_blank_man_zone_becomes_null_not_a_category(self):
        """Roughly half of charted plays carry an empty man/zone string, which
        would otherwise be counted as its own coverage."""
        sp = coverage.shell_plays(self._part())
        assert sp.man_zone.tolist()[:2] == ["ZONE_COVERAGE", "MAN_COVERAGE"]
        assert pd.isna(sp.man_zone.iloc[2])

    def test_empty_and_malformed_input(self):
        assert coverage.shell_plays(pd.DataFrame()).empty
        assert coverage.shell_plays(None).empty
        assert coverage.shell_plays(
            pd.DataFrame({"nflverse_game_id": ["nope"],
                          "possession_team": ["BAL"]})).empty


def _shells(season, weeks, defense, shell, man_zone, n):
    """`n` identical charted plays, one row each."""
    return pd.DataFrame([{"season": season, "week": w, "defense": defense,
                          "shell": shell, "man_zone": man_zone}
                         for w in weeks for _ in range(n)])


class TestShellMix:
    def test_shares_are_percentages_that_add_to_a_hundred(self):
        sp = pd.concat([
            _shells(2025, [1, 2], "BAL", "COVER_3", "ZONE_COVERAGE", 30),
            _shells(2025, [1, 2], "BAL", "COVER_1", "MAN_COVERAGE", 10),
        ])
        mix = coverage.shell_mix(sp, 2025, week=3)
        row = mix.iloc[0]
        assert row.defense == "BAL"
        assert row.cover_3 == 75.0 and row.cover_1 == 25.0
        cols = [s.lower() for s in coverage.SHELLS] + ["other"]
        assert round(sum(row[c] for c in cols), 1) == 100.0

    def test_man_rate_counts_only_charted_man_zone_plays(self):
        sp = pd.concat([
            _shells(2025, [1], "BAL", "COVER_1", "MAN_COVERAGE", 30),
            _shells(2025, [1], "BAL", "COVER_3", "ZONE_COVERAGE", 30),
            _shells(2025, [1], "BAL", "COVER_2", None, 60),
        ])
        row = coverage.shell_mix(sp, 2025, week=2).iloc[0]
        assert row.man_rate == 50.0 and row.zone_rate == 50.0

    def test_the_graded_week_never_contributes(self):
        """Leakage-safe the same way corner_grades is."""
        sp = pd.concat([
            _shells(2025, [1], "BAL", "COVER_3", "ZONE_COVERAGE", 80),
            _shells(2025, [9], "BAL", "COVER_1", "MAN_COVERAGE", 80),
        ])
        row = coverage.shell_mix(sp, 2025, week=9).iloc[0]
        assert row.cover_3 == 100.0 and row.cover_1 == 0.0

    def test_a_thin_defence_is_left_out(self):
        sp = _shells(2025, [1], "BAL", "COVER_3", "ZONE_COVERAGE", 10)
        assert coverage.shell_mix(sp, 2025, week=2).empty

    def test_the_prior_season_carries_part_of_the_load(self):
        sp = pd.concat([
            _shells(2025, [1], "BAL", "COVER_3", "ZONE_COVERAGE", 100),
            _shells(2024, [1], "BAL", "COVER_1", "MAN_COVERAGE", 100),
        ])
        row = coverage.shell_mix(sp, 2025, week=2,
                                 prior_weight=0.5).iloc[0]
        # 100 current + 50 weighted prior
        assert row.cover_3 == 66.7 and row.cover_1 == 33.3
        assert row.plays == 100.0 and row.plays_prior == 50.0
        assert row.shell_seasons == "2025,2024"
        assert row.current_season_plays


class TestShellMixSaysWhenItIsStale:
    """nflverse publishes no participation after SHELL_LAST_SEASON, so a
    current-season mix cannot be built at all. A row made entirely of last
    season's scheme has to admit it rather than read as this week's tendency.
    """

    def test_a_season_with_no_charting_falls_back_and_flags_it(self):
        sp = _shells(2025, [1, 2], "BAL", "COVER_3", "ZONE_COVERAGE", 100)
        mix = coverage.shell_mix(sp, 2026, week=9)
        row = mix.iloc[0]
        assert not row.current_season_plays
        assert row.plays == 0.0 and row.plays_prior > 0
        assert row.shell_seasons == "2025"
        # The distribution is still usable, just not current.
        assert row.cover_3 == 100.0

    def test_nothing_at_all_is_an_empty_frame(self):
        sp = _shells(2024, [1], "BAL", "COVER_3", "ZONE_COVERAGE", 100)
        assert coverage.shell_mix(sp, 2026, week=9).empty
        assert coverage.shell_mix(pd.DataFrame(), 2026, week=9).empty

    def test_the_cutoff_is_named_so_a_caller_can_check_it(self):
        assert coverage.SHELL_LAST_SEASON == 2025
