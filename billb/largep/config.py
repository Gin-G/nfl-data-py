# config.py
"""Configuration settings for NFL data processing"""

# Data processing settings
SEASONS = list(range(2018, 2025))
CURRENT_SEASON = 2024
DATA_OUTPUT_PATH = 'data/nfl_dataset_enhanced.csv'

# Fantasy scoring settings
FANDUEL_SCORING = {
    'passing_yards': 0.04,
    'passing_tds': 4,
    'interceptions': -1,
    'passing_bonus_300': 3,  # Bonus for 300+ passing yards
    'rushing_yards': 0.1,
    'rushing_tds': 6,
    'rushing_bonus_100': 3,  # Bonus for 100+ rushing yards
    'receptions': 0.5,
    'receiving_yards': 0.1,
    'receiving_tds': 6,
    'receiving_bonus_100': 3,  # Bonus for 100+ receiving yards
    'fumbles': -2,
    'return_tds': 6,
    'two_point_conversions': 2,
    'field_goals_0_39': 3,
    'field_goals_40_49': 4,
    'field_goals_50_plus': 5,
    'extra_points': 1
}

# Columns to include in averages
AVERAGING_COLUMNS = [
    'passing_yards', 'passing_tds', 'interceptions', 'passing_attempts', 'passing_completions',
    'rushing_yards', 'rushing_tds', 'rushing_attempts', 'rushing_fumbles',
    'receiving_yards', 'receiving_tds', 'receptions', 'targets', 'receiving_fumbles',
    'offensive_snaps', 'defensive_snaps', 'special_teams_snaps', 'total_snaps',
    'offensive_snap_pct', 'defensive_snap_pct', 'special_teams_snap_pct',
    'fanduel_fantasy_points'
]

# Snap count columns
SNAP_COLUMNS = [
    'offensive_snaps', 'defensive_snaps', 'special_teams_snaps', 'total_snaps',
    'offensive_snap_pct', 'defensive_snap_pct', 'special_teams_snap_pct'
]

# Play categorization
SCRIMMAGE_PLAYS = {'run', 'pass', 'no_play'}
SPECIAL_TEAMS_PLAYS = {'kickoff', 'punt', 'field_goal', 'extra_point'}

# Player information columns to preserve
PLAYER_INFO_COLUMNS = ['player_name', 'position', 'recent_team', 'gsis_id']

# Weeks per season (changed in 2021)
def get_weeks_for_season(season):
    """Get the number of weeks for a given season"""
    return range(1, 19) if season >= 2021 else range(1, 18)