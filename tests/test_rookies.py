"""Unit tests for the rookie draft-capital prior (no network — coeffs set directly)."""
from nfl_projections.rookies import RookiePrior


def _prior():
    # ppg = a + b*log(pick); b negative so earlier picks project higher
    return RookiePrior(
        coeffs={"RB": (20.0, -3.0), "WR": (16.0, -2.5), "QB": (18.0, -2.8), "TE": (12.0, -2.0)},
        resid={p: (-4.0, 4.0) for p in ("RB", "WR", "QB", "TE")},
        comp_mix={"RB": {"rushing_yards": 4.0, "receiving_yards": 1.5, "receptions": 0.15},
                  "WR": {}, "QB": {}, "TE": {}},
    )


def test_earlier_pick_projects_higher():
    rp = _prior()
    top = rp.project("RB", 3)["fanduel_fantasy_points"]
    mid = rp.project("RB", 45)["fanduel_fantasy_points"]
    late = rp.project("RB", 200)["fanduel_fantasy_points"]
    assert top > mid > late


def test_floor_below_ceiling_and_clipped():
    rp = _prior()
    p = rp.project("WR", 15)
    assert p["floor"] <= p["projection_median"] <= p["ceiling"]
    assert p["floor"] >= 0.5  # min clip


def test_undrafted_treated_as_late_pick():
    rp = _prior()
    undrafted = rp.project("RB", None)["fanduel_fantasy_points"]
    last_pick = rp.project("RB", 259)["fanduel_fantasy_points"]
    assert abs(undrafted - last_pick) < 1.0


def test_component_estimates_scale_with_projection():
    rp = _prior()
    p = rp.project("RB", 5)
    assert p["rushing_yards"] > 0 and p["receptions"] > 0
    # components should be a fraction of the projected points, in a sane range
    assert p["rushing_yards"] < 200


def test_unknown_position_returns_none():
    assert _prior().project("K", 10) is None
