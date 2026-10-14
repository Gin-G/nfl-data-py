"""Line-of-scrimmage ratings (synthetic play-by-play, no network)."""
import numpy as np
import pandas as pd
import pytest

from nfl_projections import trenches


# Each team is a known quantity on both sides of the ball, so the opponent
# adjustment has something real to recover and the matchup tests have a weak
# front to point a good line at.
TEAMS = {
    "BLKA": (1.2, 0.0),    # blocks well, average front
    "BLKZ": (-1.2, 0.0),   # blocks badly, average front
    "STOP": (0.0, -1.0),   # average line, stout front
    "SOFT": (0.0, 1.0),    # average line, soft front
    "MIDA": (0.0, 0.0),
    "MIDZ": (0.0, 0.0),
}


def _pbp(seasons=(2024,), weeks=15, seed=0):
    """A league where blocking and run-stopping are known quantities.

    A carry's yardage is its offence's effect plus its opponent's front, so a
    rating that does not adjust for opponent will get the ordering wrong.
    Pressure works the same way: protection lowers it, a rush raises it.
    """
    rng = np.random.default_rng(seed)
    names = list(TEAMS)
    rows = []
    for season in seasons:
        for week in range(1, weeks + 1):
            # A real round-robin: the offset changes each week, so every team
            # faces every other. Rotating a fixed cycle instead would leave
            # each pairing unchanged and nothing could separate line from front.
            offset = 1 + (week - 1) % (len(names) - 1)
            for i, off in enumerate(names):
                dfn = names[(i + offset) % len(names)]
                mean = 4.0 + TEAMS[off][0] + TEAMS[dfn][1]
                for _ in range(26):
                    rows.append({
                        "season": season, "week": week, "season_type": "REG",
                        "posteam": off, "defteam": dfn, "rush_attempt": 1,
                        "qb_dropback": 0, "qb_scramble": 0, "qb_kneel": 0,
                        "two_point_attempt": 0, "sack": 0, "qb_hit": 0,
                        "yards_gained": float(rng.normal(mean, 3.0))})
                p = float(np.clip(0.22 - 0.08 * TEAMS[off][0]
                                  - 0.06 * TEAMS[dfn][1], 0.02, 0.6))
                for hit in rng.random(30) < p:
                    rows.append({
                        "season": season, "week": week, "season_type": "REG",
                        "posteam": off, "defteam": dfn, "rush_attempt": 0,
                        "qb_dropback": 1, "qb_scramble": 0, "qb_kneel": 0,
                        "two_point_attempt": 0, "sack": 0,
                        "qb_hit": 1 if hit else 0, "yards_gained": 6.0})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def frames():
    return trenches.plays(pbp=_pbp())


def test_band_weights_follow_the_bands():
    out = trenches.band(pd.Series([-3.0, 0.0, 4.0, 10.0, 40.0]))
    assert out == pytest.approx([-3.6, 0.0, 4.0, 7.0, 7.0])


def test_band_caps_a_long_run():
    """A 60-yard run must not make a line look better than a 12-yard one."""
    assert trenches.band(pd.Series([12.0]))[0] == trenches.band(pd.Series([60.0]))[0]


def test_qb_runs_are_excluded_from_the_rushing_side():
    pbp = _pbp()
    pbp.loc[pbp.index[:20], ["qb_scramble", "rush_attempt"]] = [1, 1]
    rush, _ = trenches.plays(pbp=pbp)
    assert len(rush) == (pbp.rush_attempt.eq(1) & pbp.qb_scramble.eq(0)).sum()


def test_srs_recovers_the_injected_effects(frames):
    rush, _ = frames
    off, dfn, _ = trenches._srs(rush, "yards_gained")
    assert off["BLKA"] > off["MIDA"] > off["BLKZ"]
    assert dfn["STOP"] < dfn["SOFT"]


def test_run_block_grade_orders_the_lines(frames):
    g = trenches.unit_ratings(*frames)
    assert g.loc["BLKA", "run_block"] > g.loc["MIDA", "run_block"] > g.loc["BLKZ", "run_block"]


def test_run_stop_grade_is_inverted(frames):
    """A front that allows less must grade HIGHER, not lower."""
    g = trenches.unit_ratings(*frames)
    assert g.loc["STOP", "run_stop"] > g.loc["SOFT", "run_stop"]


def test_pass_block_grade_is_inverted(frames):
    g = trenches.unit_ratings(*frames)
    assert g.loc["BLKA", "pass_block"] > g.loc["BLKZ", "pass_block"]


def test_grades_sit_on_a_fifty_centred_scale(frames):
    g = trenches.unit_ratings(*frames)
    for unit in trenches.UNITS:
        assert 0 <= g[unit].min() and g[unit].max() <= 100
        assert abs(g[unit].mean() - 50) < 6


def test_matchup_is_fifty_when_the_units_are_even():
    g = pd.DataFrame({"run_block": [50.0, 50.0], "run_stop": [50.0, 50.0],
                      "pass_block": [50.0, 50.0], "pass_rush": [50.0, 50.0]},
                     index=["A", "B"])
    assert trenches.matchup(g, "A", "B", "run") == 50.0


def test_matchup_rewards_a_good_line_against_a_weak_front():
    g = pd.DataFrame({"run_block": [80.0, 20.0], "run_stop": [80.0, 20.0],
                      "pass_block": [50.0, 50.0], "pass_rush": [50.0, 50.0]},
                     index=["A", "B"])
    assert trenches.matchup(g, "A", "B", "run") > 75
    assert trenches.matchup(g, "B", "A", "run") < 25


def test_matchup_stays_on_the_scale_at_the_extremes():
    g = pd.DataFrame({"run_block": [100.0, 0.0], "run_stop": [0.0, 100.0],
                      "pass_block": [100.0, 0.0], "pass_rush": [0.0, 100.0]},
                     index=["A", "B"])
    for phase in ("run", "pass"):
        assert 0 <= trenches.matchup(g, "A", "B", phase) <= 100
        assert 0 <= trenches.matchup(g, "B", "A", phase) <= 100


def test_matchup_is_nan_for_a_team_with_no_grade():
    g = pd.DataFrame({"run_block": [60.0], "run_stop": [40.0]}, index=["A"])
    assert np.isnan(trenches.matchup(g, "A", "NOBODY", "run"))


def test_regression_pulls_toward_fifty_by_unit():
    prior = pd.DataFrame({u: [70.0] for u in trenches.UNITS}, index=["A"])
    out = trenches.regressed(prior)
    for unit in trenches.UNITS:
        expected = 50 + 20 * trenches.YOY_CARRYOVER[unit]
        assert out.loc["A", unit] == pytest.approx(expected, abs=0.05)
    # Offensive lines hold their form better than defensive fronts do.
    assert out.loc["A", "pass_block"] > out.loc["A", "run_stop"]


def test_ratings_through_cannot_see_the_week_it_rates():
    """The week being rated must not contribute: shove a freak week in and the
    rating for that week stays put while the next week's moves."""
    pbp = _pbp()
    spike = pbp[(pbp.week == 10) & (pbp.posteam == "BLKZ") & (pbp.rush_attempt == 1)].copy()
    spike["yards_gained"] = 40.0
    pbp = pd.concat([pbp[~((pbp.week == 10) & (pbp.posteam == "BLKZ")
                           & (pbp.rush_attempt == 1))], spike], ignore_index=True)
    rush, drop = trenches.plays(pbp=pbp)
    before = trenches.ratings_through(rush, drop, 2024, 10)
    after = trenches.ratings_through(rush, drop, 2024, 11)
    assert after.loc["BLKZ", "run_block"] > before.loc["BLKZ", "run_block"]


def test_ratings_through_falls_back_to_the_prior_when_the_season_is_young(frames):
    rush, drop = frames
    prior = pd.DataFrame({u: [70.0] for u in trenches.UNITS}, index=["BLKA"])
    out = trenches.ratings_through(rush, drop, 2024, 1, prior=prior)
    assert out.basis.iloc[0] == "prior-regressed"


def test_backtest_bands_and_correlations_run_end_to_end(frames):
    rush, drop = frames
    per_game, summary = trenches.backtest(rush, drop, first_week=6)
    assert not per_game.empty and not summary.empty
    assert set(summary.band.dropna()) <= {label for _, _, label in trenches.BANDS}
    # Constructed so the matchup is real, so the rating must track the outcome.
    assert trenches.correlations(per_game)["ypc_vs_own"] > 0.2
