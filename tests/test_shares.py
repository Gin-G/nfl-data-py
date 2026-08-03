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
