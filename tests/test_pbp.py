import numpy as np
import pandas as pd

from nfl_projections import pbp


def _play(season, week, defteam, posteam, **kw):
    row = {
        "season": season, "week": week, "defteam": defteam, "posteam": posteam,
        "play_type": kw.get("play_type", "run"),
        "rush_attempt": kw.get("rush_attempt", 0),
        "pass_attempt": kw.get("pass_attempt", 0),
        "run_gap": kw.get("run_gap", np.nan),
        "run_location": kw.get("run_location", np.nan),
        "pass_length": kw.get("pass_length", np.nan),
        "air_yards": kw.get("air_yards", np.nan),
        "epa": kw.get("epa", 0.0),
        "rusher_player_id": kw.get("rusher_player_id", np.nan),
        "receiver_player_id": kw.get("receiver_player_id", np.nan),
        "passer_player_id": kw.get("passer_player_id", np.nan),
    }
    return row


def _run(defteam="DEF", outside=True, epa=0.0, week=1, rusher="rb1"):
    return _play(2024, week, defteam, "OFF", play_type="run", rush_attempt=1,
                 run_gap="end" if outside else "guard", epa=epa, rusher_player_id=rusher)


def _pass(defteam="DEF", deep=False, epa=0.0, week=1, air=5, rec="wr1", passer="qb1"):
    return _play(2024, week, defteam, "OFF", play_type="pass", pass_attempt=1,
                 pass_length="deep" if deep else "short", air_yards=air, epa=epa,
                 receiver_player_id=rec, passer_player_id=passer)


class TestDefenseSplits:
    def test_outside_vs_inside_epa(self):
        plays = pd.DataFrame([
            _run(outside=True, epa=1.0), _run(outside=True, epa=1.0),
            _run(outside=False, epa=-0.5), _run(outside=False, epa=-0.5),
        ])
        d = pbp.defense_splits_weekly(plays)
        row = d.iloc[0]
        assert row["rush_epa_outside"] == 1.0
        assert row["rush_epa_inside"] == -0.5

    def test_short_vs_deep_epa(self):
        plays = pd.DataFrame([
            _pass(deep=True, epa=2.0), _pass(deep=False, epa=0.1),
        ])
        d = pbp.defense_splits_weekly(plays)
        row = d.iloc[0]
        assert row["pass_epa_deep"] == 2.0
        assert row["pass_epa_short"] == 0.1

    def test_scheme_form_is_leakage_free(self):
        plays = pd.DataFrame([
            _run(outside=True, epa=1.0, week=1),
            _run(outside=True, epa=3.0, week=2),
        ])
        form = pbp.build_scheme_defense_form(plays, window=6)
        # No form entering week 1; entering week 2 = week 1 only (epa 1.0)
        assert form[form["week"] == 1].empty
        wk2 = form[form["week"] == 2].iloc[0]
        assert wk2["opp_rush_epa_outside"] == 1.0


class TestPlayerTendencies:
    def test_outside_run_share(self):
        plays = pd.DataFrame([
            _run(outside=True, rusher="rb1"), _run(outside=True, rusher="rb1"),
            _run(outside=False, rusher="rb1"), _run(outside=False, rusher="rb1"),
        ])
        ten = pbp.player_tendencies_weekly(plays)
        rb = ten[ten["player_id"] == "rb1"].iloc[0]
        assert rb["ten_outside_run_share"] == 0.5

    def test_receiver_adot_and_deep_rate(self):
        plays = pd.DataFrame([
            _pass(deep=True, air=20, rec="wr1"), _pass(deep=False, air=4, rec="wr1"),
        ])
        ten = pbp.player_tendencies_weekly(plays)
        wr = ten[ten["player_id"] == "wr1"].iloc[0]
        assert wr["ten_adot"] == 12.0        # (20+4)/2
        assert wr["ten_deep_rate"] == 0.5    # 1 of 2 deep

    def test_attach_merges_on_player_week(self):
        dataset = pd.DataFrame({
            "player_id": ["rb1", "rb1"], "season": [2024, 2024],
            "week": [1, "AVG"], "fanduel_fantasy_points": [10.0, 10.0],
        })
        ten = pd.DataFrame({
            "player_id": ["rb1"], "season": [2024], "week": [1],
            "ten_outside_run_share": [0.7], "ten_adot": [np.nan], "ten_deep_rate": [np.nan],
        })
        out = pbp.attach_player_tendencies(dataset, ten)
        wk1 = out[out["week"] == 1].iloc[0]
        assert wk1["ten_outside_run_share"] == 0.7
        # AVG row doesn't match a numeric week -> NaN
        assert pd.isna(out[out["week"] == "AVG"].iloc[0]["ten_outside_run_share"])
