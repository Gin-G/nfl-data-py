# fantasy_calculations.py
import numpy as np
import pandas as pd

@np.vectorize
def calculate_fanduel_fantasy_points(
    passing_yards, passing_tds, interceptions,
    rushing_yards, rushing_tds,
    receptions, receiving_yards, receiving_tds,
    fumbles, return_tds=0, two_point_conversions=0,
    field_goals_0_39=0, field_goals_40_49=0, field_goals_50_plus=0,
    extra_points=0
):
    """
    Calculate FanDuel fantasy points based on player stats
    
    Args:
        passing_yards: Passing yards
        passing_tds: Passing touchdowns
        interceptions: Interceptions thrown
        rushing_yards: Rushing yards
        rushing_tds: Rushing touchdowns
        receptions: Number of receptions
        receiving_yards: Receiving yards
        receiving_tds: Receiving touchdowns
        fumbles: Total fumbles lost
        return_tds: Return touchdowns (default 0)
        two_point_conversions: Two-point conversions (default 0)
        field_goals_0_39: Field goals 0-39 yards (default 0)
        field_goals_40_49: Field goals 40-49 yards (default 0)
        field_goals_50_plus: Field goals 50+ yards (default 0)
        extra_points: Extra points made (default 0)
    
    Returns:
        Float: Total FanDuel fantasy points
    """
    # Handle NaN values by converting to 0
    def safe_value(val):
        return 0 if pd.isna(val) else val
    
    points = (
        safe_value(passing_yards) * 0.04 +
        safe_value(passing_tds) * 4 +
        safe_value(interceptions) * -1 +
        (3 if safe_value(passing_yards) >= 300 else 0) +
        safe_value(rushing_yards) * 0.1 +
        safe_value(rushing_tds) * 6 +
        (3 if safe_value(rushing_yards) >= 100 else 0) +
        safe_value(receptions) * 0.5 +
        safe_value(receiving_yards) * 0.1 +
        safe_value(receiving_tds) * 6 +
        (3 if safe_value(receiving_yards) >= 100 else 0) +
        safe_value(fumbles) * -2 +
        safe_value(return_tds) * 6 +
        safe_value(two_point_conversions) * 2 +
        safe_value(field_goals_0_39) * 3 +
        safe_value(field_goals_40_49) * 4 +
        safe_value(field_goals_50_plus) * 5 +
        safe_value(extra_points) * 1
    )
    return points

def calculate_custom_fantasy_points(stats_dict, scoring_rules):
    """
    Calculate fantasy points using custom scoring rules
    
    Args:
        stats_dict: Dictionary of player stats
        scoring_rules: Dictionary of stat -> points mapping
    
    Returns:
        Float: Total fantasy points
    """
    total_points = 0
    
    for stat, points_per in scoring_rules.items():
        if stat in stats_dict and stats_dict[stat] is not None:
            total_points += stats_dict[stat] * points_per
    
    return total_points