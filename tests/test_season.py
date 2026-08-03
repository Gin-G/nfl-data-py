"""Unit tests for the season-projection assembly (synthetic data, no network)."""
import pandas as pd

from nfl_projections import season as S


def _base():
    return pd.DataFrame([
        {"player_id": "p1", "player_name": "WR One", "position": "WR", "team": "AAA",
         "fanduel_fantasy_points": 12.0, "floor": 6.0, "ceiling": 20.0},
    ])


def _grades():
    return pd.DataFrame({"off_rating": {"AAA": 4.0, "SOFT": 0.0, "STINGY": 0.0},
                         "def_rating": {"AAA": 0.0, "SOFT": 5.0, "STINGY": -5.0}})


def _sched():
    return pd.DataFrame([
        {"season": 2026, "week": 1, "home_team": "AAA", "away_team": "SOFT"},
        {"season": 2026, "week": 2, "home_team": "STINGY", "away_team": "AAA"},
    ])


def test_soft_defense_raises_stingy_lowers():
    wk = S.assemble_season(_base(), 2026, grades=_grades(), league_avg=22.5, schedule=_sched())
    soft = wk[wk.opponent == "SOFT"].iloc[0]
    stingy = wk[wk.opponent == "STINGY"].iloc[0]
    assert soft["projection"] > soft["base_projection"] > stingy["projection"]


def test_multiplier_clipped_to_band():
    # extreme opponent rating must not push the multiplier past the ±15% clip
    grades = pd.DataFrame({"off_rating": {"AAA": 0.0, "X": 0.0}, "def_rating": {"AAA": 0.0, "X": 99.0}})
    sched = pd.DataFrame([{"season": 2026, "week": 1, "home_team": "AAA", "away_team": "X"}])
    wk = S.assemble_season(_base(), 2026, grades=grades, league_avg=22.5, schedule=sched)
    assert wk.iloc[0]["matchup_multiplier"] <= 1.15


def test_band_scales_with_multiplier():
    wk = S.assemble_season(_base(), 2026, grades=_grades(), league_avg=22.5, schedule=_sched())
    soft = wk[wk.opponent == "SOFT"].iloc[0]
    assert soft["ceiling"] > 20.0 and soft["floor"] > 6.0  # both lifted vs a soft D


def test_season_totals_aggregate():
    wk = S.assemble_season(_base(), 2026, grades=_grades(), league_avg=22.5, schedule=_sched())
    tot = S.season_totals(wk)
    assert tot.iloc[0]["games"] == 2
    assert abs(tot.iloc[0]["proj_total"] - wk["projection"].sum()) < 1e-6


# --- role/volume corrections (the backup over-projection fix) ---------------

def _full_sched(team="AAA", opp="OPP", weeks=17):
    rows = []
    for w in range(1, weeks + 1):
        rows.append({"season": 2026, "week": w, "home_team": team, "away_team": opp})
    return pd.DataFrame(rows)


def _neutral_grades(teams=("AAA", "OPP")):
    return pd.DataFrame({"off_rating": {t: 0.0 for t in teams},
                         "def_rating": {t: 0.0 for t in teams}})


def test_backup_qb_buried_by_expected_games():
    # Two QBs with identical per-game form; only depth rank differs. The starter
    # should project for a full season, the backup for almost nothing.
    base = pd.DataFrame([
        {"player_id": "s", "player_name": "Starter QB", "position": "QB", "team": "AAA",
         "depth_rank": 1, "fanduel_fantasy_points": 18.0},
        {"player_id": "b", "player_name": "Backup QB", "position": "QB", "team": "AAA",
         "depth_rank": 3, "fanduel_fantasy_points": 18.0},
    ])
    wk = S.assemble_season(base, 2026, grades=_neutral_grades(), league_avg=22.5,
                           schedule=_full_sched())
    tot = S.season_totals(wk).set_index("player_id")
    assert tot.loc["s", "proj_total"] > 250          # ~17 games of a real starter
    assert tot.loc["b", "proj_total"] < 30           # a benched backup is nothing
    assert tot.loc["b", "exp_games"] <= 2.0


def test_same_team_rbs_share_a_finite_pool():
    # Two RBs whose per-game numbers sum well past a team's RB budget must be
    # scaled down to share it — neither can keep a full bell-cow projection.
    base = pd.DataFrame([
        {"player_id": "a", "player_name": "Back A", "position": "RB", "team": "AAA",
         "depth_rank": 1, "fanduel_fantasy_points": 16.0},
        {"player_id": "b", "player_name": "Back B", "position": "RB", "team": "AAA",
         "depth_rank": 1, "fanduel_fantasy_points": 16.0},
    ])
    budgets = {"RB": 20.0, "WR": 34.0, "TE": 11.0, "QB": 22.0}
    wk = S.assemble_season(base, 2026, grades=_neutral_grades(), league_avg=22.5,
                           schedule=_full_sched(), budgets=budgets)
    # per week the two RBs must sum to no more than the 20-pt budget
    per_week = wk.groupby("week")["projection"].sum()
    assert per_week.max() <= 20.0 + 1e-6
    assert (wk["projection"] < 16.0).all()  # each got scaled below its raw rate


def test_use_roles_false_restores_raw_behaviour():
    base = pd.DataFrame([
        {"player_id": "b", "player_name": "Backup QB", "position": "QB", "team": "AAA",
         "depth_rank": 3, "fanduel_fantasy_points": 18.0},
    ])
    wk = S.assemble_season(base, 2026, grades=_neutral_grades(), league_avg=22.5,
                           schedule=_full_sched(), use_roles=False)
    tot = S.season_totals(wk)
    assert tot.iloc[0]["proj_total"] > 250  # raw: 17 full games, no role discount


def test_snap_share_override_halves_a_two_way_player():
    base = pd.DataFrame([
        {"player_id": "h", "player_name": "Travis Hunter", "position": "WR", "team": "AAA",
         "depth_rank": 1, "fanduel_fantasy_points": 14.0},
    ])
    full = S.season_totals(S.assemble_season(
        base, 2026, grades=_neutral_grades(), league_avg=22.5, schedule=_full_sched()))
    capped = S.season_totals(S.assemble_season(
        base, 2026, grades=_neutral_grades(), league_avg=22.5, schedule=_full_sched(),
        snap_share={"Travis Hunter": 0.5}))
    assert abs(capped.iloc[0]["proj_total"] - 0.5 * full.iloc[0]["proj_total"]) < 1.0
