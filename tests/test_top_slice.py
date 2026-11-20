"""The top-of-board slice: chosen before the game, the same for every arm."""
import numpy as np
import pandas as pd

from nfl_projections import evaluate


def _results():
    rows = []
    for wk in (3, 4):
        for i in range(30):
            rows.append({"player_id": f"wr{i}", "season": 2025, "week": wk,
                         "position": "WR", "predicted": 30 - i, "actual": i})
        for i in range(5):
            rows.append({"player_id": f"qb{i}", "season": 2025, "week": wk,
                         "position": "QB", "predicted": 20 - i, "actual": 10})
        rows.append({"player_id": "k0", "season": 2025, "week": wk,
                     "position": "K", "predicted": 9, "actual": 9})
    return pd.DataFrame(rows)


def test_top_n_per_week_and_position():
    r = _results()
    m = evaluate.top_slice(r, "predicted", top_n={"WR": 3, "QB": 2})
    picked = r[m]
    assert len(picked) == 2 * (3 + 2)
    assert set(picked[picked.position == "WR"].player_id) == {"wr0", "wr1", "wr2"}
    assert "K" not in set(picked.position)          # not in top_n: excluded


def test_nan_ranks_last_and_array_keys_work():
    r = _results()
    key = np.where(r.player_id == "wr0", np.nan, r.predicted)
    m = evaluate.top_slice(r, key, top_n={"WR": 1, "QB": 0})
    assert set(r[m].player_id) == {"wr1"}


def test_trailing_form_uses_only_earlier_games_across_seasons():
    hist = pd.DataFrame({
        "player_id": ["a"] * 4, "season": [2024, 2024, 2025, 2025],
        "week": [17, 18, 1, 2], "season_type": "REG",
        "fanduel_fantasy_points": [10.0, 20.0, 30.0, 99.0]})
    res = pd.DataFrame({"player_id": ["a", "a", "b"], "season": [2025, 2025, 2025],
                        "week": [2, 1, 2]})
    f = evaluate.trailing_form(hist, res, n_games=2)
    assert f[0] == 25.0              # weeks 18 and 1 -- never week 2's own 99
    assert f[1] == 15.0              # crosses the season boundary
    assert np.isnan(f[2])            # no history


def test_slice_metrics_shape():
    r = _results()
    out = evaluate.slice_metrics(r, evaluate.top_slice(r, "predicted"))
    assert {"ALL", "QB", "WR"} <= set(out.index)
    assert out.loc["ALL", "n"] == len(r[evaluate.top_slice(r, "predicted")])
