#!/usr/bin/env python3
"""Steadiness, and the one kind of it that measures.

The thing under test is mostly a refusal: the ranking must not be orderable by
scoring dispersion, because scoring dispersion does not persist (split-half
+0.021 once the level is out, CI [-0.114, +0.156]). Ranking on volume is what
is left, and it is what these pin.
"""
import numpy as np
import pandas as pd
import pytest

from nfl_projections import consistency as C


def _weeks(rows):
    """Player-weeks in the shape nflverse hands over."""
    out = []
    for name, position, team, weeks in rows:
        for week, (carries, targets, attempts, rec_yards) in enumerate(weeks, 1):
            out.append({
                "player_id": f"id-{name}", "player_name": name,
                "player_display_name": name, "position": position,
                "team": team, "season": 2026, "week": week,
                "season_type": "REG", "carries": carries, "targets": targets,
                "attempts": attempts, "receptions": targets,
                "receiving_yards": rec_yards, "rushing_yards": carries * 4.0,
                "passing_yards": attempts * 7.0,
            })
    return pd.DataFrame(out)


STEADY = ("Steady Back", "RB", "DET",
          [(18, 3, 0, 20), (19, 2, 0, 18), (17, 3, 0, 22), (18, 3, 0, 19),
           (19, 2, 0, 21), (18, 3, 0, 20)])
SPIKY = ("Spiky Back", "RB", "CAR",
         [(4, 1, 0, 5), (28, 6, 0, 60), (3, 0, 0, 0), (26, 5, 0, 55),
          (2, 1, 0, 8), (30, 4, 0, 50)])
TINY = ("Tiny Role", "WR", "ARI",
        [(0, 2, 0, 10), (0, 2, 0, 11), (0, 2, 0, 9), (0, 2, 0, 10),
         (0, 2, 0, 12), (0, 2, 0, 10)])


class TestItRanksOnVolumeNotOnScoring:
    def test_the_steady_back_outranks_the_spiky_one(self):
        out = C.player_consistency(_weeks([STEADY, SPIKY]))
        assert list(out.player_name) == ["Steady Back", "Spiky Back"]
        steady = out[out.player_name == "Steady Back"].iloc[0]
        spiky = out[out.player_name == "Spiky Back"].iloc[0]
        assert steady.volume_steadiness > spiky.volume_steadiness
        assert steady.steadiness_rank == 1

    def test_a_two_target_role_is_not_the_most_consistent_player(self):
        """His usage never moves, and he is useless. Raw dispersion ranks him
        first; a volume floor is what stops that."""
        out = C.player_consistency(_weeks([STEADY, SPIKY, TINY]))
        tiny = out[out.player_name == "Tiny Role"].iloc[0]
        assert tiny.opp_per_game < C.MIN_OPPORTUNITIES
        assert pd.isna(tiny.steadiness_rank)
        assert out[out.player_name == "Steady Back"].iloc[0].steadiness_rank == 1

    def test_scoring_dispersion_cannot_reorder_anybody(self):
        """The measured reason this module exists. Scoring is description."""
        base = _weeks([STEADY, SPIKY])
        plain = C.player_consistency(base)
        # Wreck the scoring without touching a single opportunity: same
        # carries, same targets, wildly different yards.
        loud = base.copy()
        loud["receiving_yards"] = np.where(
            loud.player_name == "Steady Back",
            np.resize([0, 200, 0, 190, 5, 210], len(loud)), loud.receiving_yards)
        shaken = C.player_consistency(loud)
        assert list(plain.player_name) == list(shaken.player_name)
        for col in ("volume_steadiness", "steadiness_rank", "opp_cv"):
            assert list(plain[col].fillna(-1)) == list(shaken[col].fillna(-1))
        # ... and the scoring columns DID move, so the test is not vacuous.
        assert (plain.points_sd.tolist() != shaken.points_sd.tolist())


class TestWhatARowSays:
    def test_it_carries_the_range_and_the_rate(self):
        out = C.player_consistency(_weeks([STEADY]))
        row = out.iloc[0]
        assert row.games == 6
        assert row.opp_low == 20 and row.opp_high == 21   # carries + targets
        assert 0.0 <= row.hit_rate_15 <= 1.0
        assert row.points_low <= row.points_median <= row.points_high

    def test_steadiness_is_just_one_minus_the_cv(self):
        """No fitting, so anyone can check it."""
        out = C.player_consistency(_weeks([STEADY, SPIKY]))
        for _, r in out.iterrows():
            assert r.volume_steadiness == pytest.approx(
                round(100 * (1 - min(max(r.opp_cv, 0), 1)), 1), abs=0.05)

    def test_a_rising_role_is_reported_as_rising(self):
        rising = ("Rising Back", "RB", "NYJ",
                  [(5, 1, 0, 5), (6, 1, 0, 6), (7, 2, 0, 8),
                   (18, 3, 0, 20), (19, 3, 0, 22), (20, 2, 0, 19)])
        row = C.player_consistency(_weeks([rising])).iloc[0]
        assert row.opp_trend > 5
        assert "up" in C.describe(row)

    def test_the_sentence_leads_with_volume(self):
        row = C.player_consistency(_weeks([STEADY])).iloc[0]
        text = C.describe(row)
        assert "opportunities a game" in text
        assert "volume steadiness" in text
        assert "He has scored" in text


class TestTheEdges:
    def test_too_few_games_is_not_a_tendency(self):
        short = ("Short", "RB", "LV", [(18, 3, 0, 20), (19, 2, 0, 18)])
        assert C.player_consistency(_weeks([short])).empty

    def test_through_week_excludes_the_week_being_projected(self):
        out = C.player_consistency(_weeks([STEADY]), through_week=5)
        assert out.iloc[0].games == 4

    def test_an_empty_frame_is_not_an_error(self):
        assert C.player_consistency(pd.DataFrame()).empty

    def test_a_quarterback_counts_attempts_not_targets(self):
        qb = ("Arm", "QB", "DET",
              [(3, 0, 34, 0), (2, 0, 36, 0), (4, 0, 33, 0),
               (3, 0, 35, 0), (2, 0, 37, 0), (3, 0, 34, 0)])
        row = C.player_consistency(_weeks([qb])).iloc[0]
        assert row.opp_per_game == pytest.approx(38.0, abs=0.5)
        assert row.volume_steadiness > 90
