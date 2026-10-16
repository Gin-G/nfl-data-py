"""Tests for the opportunity-share context.

The guard that matters most is the last one: these columns are measured to be
worth nothing inside the objective, so the test suite has to notice if they
ever get there.
"""
import numpy as np
import pandas as pd
import pytest

from nfl_projections import opportunity, optimizer

SEASON = 2026


def _weekly():
    """Four players over three weeks on one team.

    WR volume is the high-share receiver who has not scored; WR quiet is the
    opposite; RB mixed takes both carries and targets, which is the case the
    request named specifically.
    """
    rows = []
    spec = {
        # player, position, pass volume per week, carries, receiving tds
        ("00-0000001", "WR volume", "WR"): (12, 0, 0),
        ("00-0000002", "WR quiet", "WR"): (3, 0, 1),
        ("00-0000003", "RB mixed", "RB"): (4, 14, 0),
        ("00-0000004", "QB arm", "QB"): (35, 3, 0),
    }
    for (pid, name, pos), (vol, car, rtd) in spec.items():
        for week in (1, 2, 3):
            rows.append({
                "player_id": pid, "player_name": name,
                "player_display_name": name, "position": pos, "team": "AAA",
                "season": SEASON, "week": week, "season_type": "REG",
                "targets": 0 if pos == "QB" else vol,
                "attempts": vol if pos == "QB" else 0,
                "carries": car, "receptions": 0 if pos == "QB" else vol * 0.6,
                "receiving_yards": 0 if pos == "QB" else vol * 8.0,
                "receiving_tds": rtd, "rushing_yards": car * 4.0, "rushing_tds": 0,
                "passing_yards": vol * 7.0 if pos == "QB" else 0,
                "passing_tds": 0, "passing_interceptions": 0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def context():
    return opportunity.opportunity_context(SEASON, 4, _weekly())


class TestOpportunity:
    def test_counts_targets_and_carries_for_a_running_back(self, context):
        """'For RB's it would be both' — the one requirement stated outright."""
        rb = context.set_index("player_id").loc["00-0000003"]
        assert rb.opp_per_game == pytest.approx(18.0)      # 4 targets + 14 carries

    def test_a_quarterbacks_volume_is_his_attempts(self, context):
        """He has no targets; counting them would make him a zero-volume player."""
        qb = context.set_index("player_id").loc["00-0000004"]
        assert qb.opp_per_game == pytest.approx(38.0)      # 35 attempts + 3 carries

    def test_the_quarterback_has_no_share_to_report(self, context):
        """His attempts over the skill positions' targets is about 0.7 for every
        starter alive: it looks like a share and means nothing."""
        qb = context.set_index("player_id").loc["00-0000004"]
        assert pd.isna(qb.opp_share)
        assert not pd.isna(qb.opp_per_game)

    def test_share_is_of_the_team_not_the_league(self, context):
        """The denominator excludes the quarterback's attempts, which are the
        same plays as the targets they produced."""
        c = context.set_index("player_id")
        team_opp = 12 + 3 + 18          # the three skill players
        assert c.loc["00-0000001"].opp_share == pytest.approx(12 / team_opp)
        assert c.loc["00-0000003"].opp_share == pytest.approx(18 / team_opp)

    def test_volume_without_the_scoreboard_reads_positive(self, context):
        """The ask's player: gets the ball, has not found the end zone."""
        c = context.set_index("player_id")
        assert c.loc["00-0000001"].volume_td_gap > 0
        # And the low-volume touchdown scorer is the other sign.
        assert c.loc["00-0000002"].volume_td_gap < c.loc["00-0000001"].volume_td_gap

    def test_the_scored_week_never_contributes_to_its_own_context(self):
        """Leakage would make these columns a description of the week rather
        than something readable before it."""
        weekly = _weekly()
        hot = weekly[(weekly.player_id == "00-0000002") & (weekly.week == 3)].copy()
        hot["week"] = 4
        hot["targets"] = 99
        through_4 = opportunity.opportunity_context(
            SEASON, 4, pd.concat([weekly, hot], ignore_index=True))
        baseline = opportunity.opportunity_context(SEASON, 4, weekly)
        assert (through_4.set_index("player_id").loc["00-0000002"].opp_per_game
                == pytest.approx(
                    baseline.set_index("player_id").loc["00-0000002"].opp_per_game))

    def test_a_thin_sample_is_blank_not_confident(self):
        """Below the floor the columns exist and are empty, rather than carrying
        a per-game rate built from one blowout."""
        one_week = _weekly()
        ctx = opportunity.opportunity_context(SEASON, 2, one_week[one_week.week == 1])
        assert len(ctx) == 4
        assert ctx.opp_share.isna().all()
        assert ctx.volume_td_gap.isna().all()
        assert (ctx.prior_games == 1).all()

    def test_no_weeks_yet_is_an_empty_frame_not_an_error(self):
        ctx = opportunity.opportunity_context(SEASON, 1, _weekly())
        assert ctx.empty
        assert "opp_share" in ctx.columns


class TestAttach:
    def test_missing_context_still_yields_the_columns(self):
        pool = pd.DataFrame({"player_id": ["00-0000001"], "Salary": [7000]})
        out = opportunity.attach(pool, pd.DataFrame())
        for col in opportunity.CONTEXT_COLUMNS:
            assert col in out.columns
            assert out[col].isna().all()

    def test_a_player_with_no_history_joins_as_blank(self, context):
        pool = pd.DataFrame({"player_id": ["00-0000001", "00-9999999"],
                             "Salary": [7000, 4500]})
        out = opportunity.attach(pool, context)
        assert len(out) == 2
        assert not pd.isna(out.set_index("player_id").loc["00-0000001"].opp_per_game)
        assert pd.isna(out.set_index("player_id").loc["00-9999999"].opp_per_game)


class TestItStaysOutOfTheObjective:
    def test_merge_leaves_value_untouched(self, context):
        """The measured decision: context beside the objective, never in it."""
        fd = pd.DataFrame({"Nickname": ["WR volume"], "Salary": [7000],
                           "FPPG": [11.0], "Roster Position": ["WR"]})
        preds = pd.DataFrame({"player_name": ["WR volume"], "player_id": ["00-0000001"],
                              "fanduel_fantasy_points": [14.0]})
        plain = optimizer.merge_fanduel_salaries(fd, preds)
        withctx = optimizer.merge_fanduel_salaries(fd, preds, context)
        assert withctx["value"].iloc[0] == pytest.approx(plain["value"].iloc[0])
        assert withctx["fanduel_fantasy_points"].iloc[0] == pytest.approx(14.0)
        assert not pd.isna(withctx["opp_per_game"].iloc[0])

    def test_the_objective_columns_are_not_opportunity_columns(self):
        """A rename or a 'just try it' edit that routed the objective through
        one of these would be a change the measurement does not support."""
        assert not (set(optimizer.OBJECTIVE_COLUMNS.values())
                    & set(opportunity.CONTEXT_COLUMNS))

    def test_optimizer_does_not_import_opportunity_at_module_scope(self):
        """It is reachable only through the explicit argument, so no code path
        can pick it up by accident."""
        import pathlib
        src = pathlib.Path(optimizer.__file__).read_text()
        head = src.split("def merge_fanduel_salaries")[0]
        assert "opportunity" not in head

    def test_lineups_are_identical_with_and_without_the_context(self, context):
        """The end-to-end form of the same guarantee."""
        fd = pd.DataFrame({
            "Nickname": [f"P{i}" for i in range(12)],
            "Salary": [7000, 6500, 6000, 5500, 5000, 4800, 4600, 4400,
                       4200, 4000, 3800, 3600],
            "FPPG": [18.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
            "Roster Position": ["QB", "RB", "RB", "RB", "WR", "WR", "WR", "WR",
                                "TE", "TE", "D", "D"],
        })
        preds = pd.DataFrame({
            "player_name": fd.Nickname, "player_id": [f"00-000{i:04d}" for i in range(12)],
            "fanduel_fantasy_points": fd.FPPG.values})
        a = optimizer.optimize_lineups(
            optimizer.merge_fanduel_salaries(fd, preds), num_lineups=2)
        b = optimizer.optimize_lineups(
            optimizer.merge_fanduel_salaries(fd, preds, context), num_lineups=2)
        assert len(a) == len(b)
        for la, lb in zip(a, b):
            assert list(la["Nickname"]) == list(lb["Nickname"])
            assert la["lineup_points"].sum() == pytest.approx(lb["lineup_points"].sum())


class TestMidWeek:
    def test_the_jump_uses_his_own_last_game_not_the_slates_last_week(self):
        """Built on a Thursday, the newest week has played one game. Reading the
        slate's max week would leave the jump blank for everyone not in it."""
        weekly = _weekly()
        # Week 4 exists, but only for the one receiver — a Thursday nighter.
        thursday = weekly[(weekly.player_id == "00-0000001")
                          & (weekly.week == 3)].copy()
        thursday["week"] = 4
        thursday["targets"] = 20
        ctx = opportunity.opportunity_context(
            SEASON, 5, pd.concat([weekly, thursday], ignore_index=True))
        c = ctx.set_index("player_id")
        # The receiver who played Thursday has a jump off his week 4.
        assert c.loc["00-0000001"].share_jump > 0
        # And so does everyone else, off their own week 3 rather than nothing.
        assert not pd.isna(c.loc["00-0000002"].share_jump)
        assert not pd.isna(c.loc["00-0000003"].share_jump)
