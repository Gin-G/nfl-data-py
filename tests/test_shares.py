"""Unit tests for the volume-share model (synthetic data, no network/dataset)."""
import pandas as pd

from nfl_projections import shares


def _dataset():
    """Two seasons of a tiny 1-team league: a clear lead RB + backup, two WRs.
    Season 2021 establishes usage; 2022 keeps the same roles."""
    rows = []
    def add(season, pid, name, pos, carries, targets, fppg, snap):
        for wk in range(1, 5):  # 4 games so min_games passes
            rows.append({"season": season, "week": wk, "team": "AAA", "position": pos,
                         "player_id": pid, "player_display_name": name,
                         "carries": carries, "targets": targets,
                         "fanduel_fantasy_points": fppg, "offensive_snap_pct": snap})
    for season in (2021, 2022):
        add(season, "rb1", "Lead Back", "RB", 15, 1, 18.0, 85)
        add(season, "rb2", "Backup Back", "RB", 4, 1, 6.0, 35)
        add(season, "wr1", "Top WR", "WR", 0, 9, 16.0, 90)
        add(season, "wr2", "Slot WR", "WR", 0, 5, 9.0, 70)
    return pd.DataFrame(rows)


def test_share_table_divides_the_team_pool():
    st = shares.season_share_table(_dataset())
    s21 = st[st.season == 2021].set_index("player_id")
    # team RB carries = (15+4)*4; lead back share = 15/19
    assert abs(s21.loc["rb1", "carry_share"] - 15 / 19) < 1e-6
    # target shares sum to ~1 across the team pool
    assert abs(st[st.season == 2021]["target_share"].sum() - 1.0) < 1e-6


def test_prior_is_monotonic_by_rank():
    prior = shares.fit_share_prior(shares.season_share_table(_dataset()))
    rb = prior[prior.position == "RB"].set_index("rank")
    assert rb.loc[1, "carry_share"] > rb.loc[2, "carry_share"]  # RB1 out-carries RB2


def test_predict_ranks_starter_over_backup():
    prior = shares.fit_share_prior(shares.season_share_table(_dataset()))
    roster = pd.DataFrame([
        {"team": "AAA", "position": "RB", "player_id": "x", "depth_rank": 1},
        {"team": "AAA", "position": "RB", "player_id": "y", "depth_rank": 2},
    ])
    pred = shares.predict_shares(roster, prior).set_index("player_id")
    assert pred.loc["x", "carry_share"] > pred.loc["y", "carry_share"]


def test_project_shares_blends_returning_vs_mover():
    ds = _dataset()
    # returning lead back keeps a high carry share; a brand-new rank-1 back gets the rank prior
    roster = pd.DataFrame([
        {"team": "AAA", "position": "RB", "player_id": "rb1", "player_name": "Lead Back",
         "depth_rank": 1},
        {"team": "AAA", "position": "RB", "player_id": "new", "player_name": "New Back",
         "depth_rank": 2},
    ])
    pred = shares.project_shares(roster, ds, 2023).set_index("player_id")
    assert pred.loc["rb1", "carry_share"] > 0.5           # returner keeps lead role
    assert pred.loc["new", "carry_share"] < pred.loc["rb1", "carry_share"]


def test_backtest_model_beats_repeat_baseline():
    # Build 4 seasons so the backtest has history to fit on; stable roles -> the hybrid
    # (own share blended) must not be worse than repeat-last-year.
    frames = []
    base = _dataset()
    for yr in (2021, 2022, 2023, 2024):
        f = base[base.season == 2021].copy()
        f["season"] = yr
        frames.append(f)
    ds = pd.concat(frames, ignore_index=True)
    res = shares.backtest_shares(ds, test_seasons=[2024])
    assert res["n"] > 0
    assert res["overall"]["model"] <= res["overall"]["naiveA"] + 1e-9


def _two_team_dataset():
    """Two teams over two seasons, with one back who changes teams.

    BBB's lead back vacates the role after 2021; AAA's lead back moves there for
    2022 and slots in at rank 1. Both teams' rank-1 slot therefore looks the
    same to a depth chart, while the two players carry very different histories —
    which is the distinction the mover term exists to make.
    """
    rows = []

    def add(season, team, pid, pos, carries, targets, fppg, snap):
        for wk in range(1, 5):
            rows.append({"season": season, "week": wk, "team": team, "position": pos,
                         "player_id": pid, "player_display_name": pid,
                         "carries": carries, "targets": targets,
                         "fanduel_fantasy_points": fppg, "offensive_snap_pct": snap})

    # 2021: a workhorse on AAA, a modest lead back on BBB
    add(2021, "AAA", "workhorse", "RB", 20, 3, 21.0, 90)
    add(2021, "AAA", "aaa_sub", "RB", 3, 1, 5.0, 25)
    add(2021, "AAA", "aaa_wr", "WR", 0, 8, 14.0, 88)
    add(2021, "BBB", "bbb_lead", "RB", 9, 3, 11.0, 55)
    add(2021, "BBB", "bbb_sub", "RB", 8, 3, 10.0, 50)
    add(2021, "BBB", "bbb_wr", "WR", 0, 8, 14.0, 88)
    # 2022: the workhorse is on BBB; bbb_lead is gone
    add(2022, "BBB", "workhorse", "RB", 18, 3, 20.0, 88)
    add(2022, "BBB", "bbb_sub", "RB", 5, 2, 7.0, 35)
    add(2022, "BBB", "bbb_wr", "WR", 0, 8, 14.0, 88)
    add(2022, "AAA", "aaa_sub", "RB", 14, 2, 15.0, 80)
    add(2022, "AAA", "aaa_wr", "WR", 0, 8, 14.0, 88)
    return pd.DataFrame(rows)


def test_mover_carries_his_old_team_share():
    """A rank-1 arrival with a workhorse history must project above the rank prior."""
    data = _two_team_dataset()
    roster = pd.DataFrame([
        {"team": "BBB", "position": "RB", "player_id": "workhorse", "depth_rank": 1},
        {"team": "BBB", "position": "RB", "player_id": "bbb_sub", "depth_rank": 2},
    ])
    with_mover = shares.project_shares(roster, data, 2022).set_index("player_id")
    without = shares.project_shares(roster, data, 2022, mover_blend=0.0).set_index("player_id")
    assert with_mover.loc["workhorse", "carry_share"] > without.loc["workhorse", "carry_share"]


def test_mover_blend_zero_reproduces_the_rank_prior():
    """The old behaviour has to stay reachable, or the change cannot be measured."""
    data = _two_team_dataset()
    roster = pd.DataFrame([
        {"team": "BBB", "position": "RB", "player_id": "workhorse", "depth_rank": 1},
    ])
    prior = shares.fit_share_prior(shares.season_share_table(data), max_season=2021)
    rank_only = shares.predict_shares(roster, prior).set_index("player_id")
    off = shares.project_shares(roster, data, 2022, mover_blend=0.0).set_index("player_id")
    assert abs(off.loc["workhorse", "carry_share"]
               - rank_only.loc["workhorse", "carry_share"]) < 1e-9


def test_returner_is_untouched_by_the_mover_term():
    """A player staying put must blend his own history, not the mover weight."""
    data = _two_team_dataset()
    roster = pd.DataFrame([
        {"team": "BBB", "position": "RB", "player_id": "bbb_sub", "depth_rank": 2},
    ])
    a = shares.project_shares(roster, data, 2022).set_index("player_id")
    b = shares.project_shares(roster, data, 2022, mover_blend=0.0).set_index("player_id")
    assert a.loc["bbb_sub", "carry_share"] == b.loc["bbb_sub", "carry_share"]


def test_backtest_reports_the_mover_split():
    summary = shares.backtest_shares(_two_team_dataset(), test_seasons=[2022], min_games=4)
    if summary.get("n"):
        assert "movers" in summary and "rankonly" in summary["overall"]
