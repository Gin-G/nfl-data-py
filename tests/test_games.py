"""Unit tests for the availability / games-played model (synthetic, no network)."""
import numpy as np
import pandas as pd

from nfl_projections import games


def _dataset():
    """Several seasons where games are noisily related to the prior year (so a shrunk
    slope is the right fit), across a few RBs."""
    rng = np.random.default_rng(0)
    rows = []
    players = {f"rb{i}": 8 + (i % 9) for i in range(40)}  # baseline games per player
    for season in range(2019, 2025):
        for pid, base in players.items():
            g = int(np.clip(base + rng.integers(-3, 4), 1, 17))
            for wk in range(1, g + 1):
                rows.append({"season": season, "week": wk, "player_id": pid,
                             "position": "RB", "fanduel_fantasy_points": 10.0})
    return pd.DataFrame(rows)


def test_history_has_prev_games():
    h = games.games_history(_dataset())
    assert "prev_games" in h.columns
    # a player's 2020 prev_games == their 2019 games
    one = h[h.player_id == "rb1"].set_index("season")
    assert one.loc[2020, "prev_games"] == one.loc[2019, "games"]


def test_model_shrinks_toward_mean():
    m = games.fit_games_model(_dataset(), max_season=2024)
    a, b = m["RB"]
    assert 0.0 < b < 1.0                      # slope shrinks (not repeat-last-year)
    # a durable (17) and a fragile (8) player both regress toward the middle
    hi = games.expected_games_from_history("RB", 17, m)
    lo = games.expected_games_from_history("RB", 8, m)
    assert 1.0 <= lo < hi <= 17.0
    assert hi < 17.0                          # even a 17-game player isn't projected 17


def test_expected_games_uses_history_for_starters_falls_back_for_backups():
    m = games.fit_games_model(_dataset(), max_season=2024)
    # rank-1 with history -> history model
    starter = games.expected_games("RB", 1, 16, m)
    assert abs(starter - games.expected_games_from_history("RB", 16, m)) < 1e-9
    # deep backup -> depth-role baseline regardless of history
    from nfl_projections import roles
    backup = games.expected_games("RB", 5, 16, m)
    assert abs(backup - roles.expected_games("RB", 5)) < 1e-9
    # no history -> depth-role baseline
    nohist = games.expected_games("RB", 1, None, m)
    assert abs(nohist - roles.expected_games("RB", 1)) < 1e-9


def test_backtest_model_beats_flat():
    res = games.backtest_games(_dataset())
    assert res["n"] > 0
    assert res["overall"]["model"] <= res["overall"]["flat"]
