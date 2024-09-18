import pandas as pd
import numpy as np
from nfl_data_py import import_weekly_data, import_players, import_rosters, import_schedules
from config import SEASONS, TARGET_COLS, CATEGORICAL_FEATURES, NUMERICAL_FEATURES

def load_and_preprocess_data(seasons=SEASONS):
    """
    Load and preprocess NFL data for the specified seasons.
    """
    # Load weekly data
    player_stats = pd.concat([import_weekly_data(seasons)])
    
    # Load player data
    players = import_players()
    
    # Merge player stats with player info
    df = pd.merge(player_stats, players, left_on='player_id', right_on='gsis_id', how='left')
    df = df.sort_values(['player_id', 'season', 'week'])

    # Create target variables
    for col in TARGET_COLS:
        df[f'next_week_{col}'] = df.groupby(['player_id', 'season'])[col].shift(-1)

    # Remove last week of each season for each player
    df = df.groupby(['player_id', 'season']).apply(lambda x: x.iloc[:-1]).reset_index(drop=True)

    return df

def import_latest_rosters(season):
    """
    Import the latest rosters for the specified season.
    """
    return import_rosters([season])

def import_schedule(season):
    """
    Import the NFL schedule for the specified season.
    """
    return import_schedules([season])

def get_player_data(player_name, df, current_season, current_week):
    """
    Get the latest data for a specific player.
    """
    player_data = df[(df['player_display_name'] == player_name) & 
                     (df['season'] == current_season) & 
                     (df['week'] == current_week)]
    
    if player_data.empty:
        return None
    
    return player_data.iloc[0]

def update_dataset(df, new_data):
    """
    Update the dataset with new weekly data.
    """
    # Remove any existing data for the same week and season
    df = df[~((df['season'] == new_data['season'].iloc[0]) & (df['week'] == new_data['week'].iloc[0]))]
    
    # Append new data
    df = pd.concat([df, new_data], ignore_index=True)
    
    # Sort and recalculate next week stats
    df = df.sort_values(['player_id', 'season', 'week'])
    for col in TARGET_COLS:
        df[f'next_week_{col}'] = df.groupby(['player_id', 'season'])[col].shift(-1)
    
    return df

def get_features(df):
    """
    Extract feature columns from the dataset.
    """
    return df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]

def get_targets(df):
    """
    Extract target columns from the dataset.
    """
    return df[[f'next_week_{col}' for col in TARGET_COLS]]

def filter_active_players(df):
    # Ensure we're not overly restricting the players
    return df  # Adjust this if needed

def get_position_players(df, position):
    # Make sure we're correctly filtering for the position
    return df[df['position_x'] == position]