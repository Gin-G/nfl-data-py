"""Positional scarcity and usability (synthetic weekly stats, no network)."""
import numpy as np
import pandas as pd
import pytest

from nfl_projections import scarcity


def _weekly():
    """Two seasons where QB scores high and TE scores low, by construction.

    Each position gets more players than its starter slots so replacement level
    is defined, and the points decline with rank so the ordering is knowable.
    """
    rows = []
    shape = {"QB": (20, 26.0, 0.7), "RB": (34, 18.0, 0.35),
             "WR": (46, 16.0, 0.22), "TE": (22, 11.0, 0.3)}
    for season in (2023, 2024):
        for pos, (count, top, decay) in shape.items():
            for rank in range(count):
                ppg = top - decay * rank
                for week in range(1, 13):
                    rows.append({"season": season, "week": week,
                                 "player_id": f"{pos}{rank}", "player_name": f"{pos} {rank}",
                                 "position": pos, "team": "AAA", "points": ppg})
    return pd.DataFrame(rows)


def test_per_season_ranks_within_position():
    f = scarcity.per_season(_weekly())
    qb = f[(f.season == 2023) & (f.position == "QB")].sort_values("pos_rank_ppg")
    assert qb.player_id.iloc[0] == "QB0"
    assert qb.pos_rank_ppg.tolist() == list(range(1, len(qb) + 1))


def test_replacement_is_below_the_last_starter():
    repl = scarcity.replacement_level(scarcity.per_season(_weekly()))
    for _, row in repl.iterrows():
        assert row.replacement_ppg < row.starter_ppg


def test_replacement_level_follows_positional_shape():
    """QB replacement must sit above TE replacement — that is what scarcity means."""
    repl = scarcity.replacement_level(scarcity.per_season(_weekly()))
    by_pos = repl.groupby("position").replacement_ppg.mean()
    assert by_pos["QB"] > by_pos["RB"] > by_pos["WR"] > by_pos["TE"]


def test_composition_counts_sum_to_n():
    comp = scarcity.top_n_composition(scarcity.per_season(_weekly()))
    for (season, n), g in comp.groupby(["season", "top_n"]):
        assert g.players.sum() == n


def test_usable_rate_is_a_share_of_player_weeks():
    u = scarcity.usable_rate(_weekly())
    assert ((u.usable_rate >= 0) & (u.usable_rate <= 1)).all()
    # Every QB in the fixture averages 13.7-26.0, so the QB rate must beat TE's.
    assert u[u.position == "QB"].usable_rate.mean() > u[u.position == "TE"].usable_rate.mean()


def test_value_over_replacement_subtracts_the_positional_bar():
    bars = {"QB": 18.0, "TE": 7.5}
    proj = pd.DataFrame({"player_id": ["a", "b"], "position": ["QB", "TE"],
                         "fanduel_fantasy_points": [20.0, 12.0]})
    out = scarcity.value_over_replacement(proj, bars)
    assert out.vor.tolist() == [2.0, 4.5]
    # The tight end projects lower and is worth more: the whole point.
    assert out.loc[1, "vor"] > out.loc[0, "vor"]
    assert "vor" not in proj.columns, "must not mutate the caller's frame"


def test_usability_reads_probabilities_off_the_sims():
    sims = np.array([[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]])
    out = scarcity.usability(sims, bar=15.0, bust=5.0)
    assert out["p_usable"].tolist() == [0.75, 0.0]
    assert out["p_bust"].tolist() == [0.0, 1.0]


def test_usability_rejects_a_one_dimensional_matrix():
    with pytest.raises(ValueError):
        scarcity.usability(np.array([1.0, 2.0, 3.0]))
