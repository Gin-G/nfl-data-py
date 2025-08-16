# averaging_utils.py
import pandas as pd
import numpy as np

def add_season_averages(df):
    """
    Add season average rows for each player-season combination
    
    Args:
        df: DataFrame with player stats
    
    Returns:
        DataFrame: Original data plus season average rows
    """
    # Define columns to average (numeric stats that make sense to average)
    avg_columns = [
        'passing_yards', 'passing_tds', 'interceptions', 'passing_attempts', 'passing_completions',
        'rushing_yards', 'rushing_tds', 'rushing_attempts', 'rushing_fumbles',
        'receiving_yards', 'receiving_tds', 'receptions', 'targets', 'receiving_fumbles',
        'offensive_snaps', 'defensive_snaps', 'special_teams_snaps', 'total_snaps',
        'offensive_snap_pct', 'defensive_snap_pct', 'special_teams_snap_pct',
        'fanduel_fantasy_points'
    ]
    
    # Filter to only include regular season weeks (not AVG rows if they already exist)
    regular_weeks = df[df['week'] != 'AVG'].copy()
    
    # Group by player and season to calculate averages
    season_averages = []
    
    for (player_id, season), group in regular_weeks.groupby(['player_id', 'season']):
        if len(group) == 0:
            continue
            
        # Calculate averages for numeric columns
        avg_row = {}
        avg_row['player_id'] = player_id
        avg_row['season'] = season
        avg_row['week'] = 'AVG'  # Special identifier for average rows
        
        # Copy non-numeric columns from first game
        first_game = group.iloc[0]
        for col in ['player_name', 'position', 'recent_team', 'gsis_id']:
            if col in first_game:
                avg_row[col] = first_game[col]
        
        # Calculate averages for numeric columns
        for col in avg_columns:
            if col in group.columns:
                # Use mean for most stats, but use sum for totals if that makes more sense
                if col in ['fanduel_fantasy_points']:
                    avg_row[col] = group[col].mean()
                    avg_row[f'{col}_total'] = group[col].sum()
                else:
                    avg_row[col] = group[col].mean()
        
        # Calculate games played
        avg_row['games_played'] = len(group)
        
        # Calculate avg_fppg for season average rows
        avg_row['avg_fppg'] = avg_row.get('fanduel_fantasy_points', 0)
        
        season_averages.append(avg_row)
    
    # Create DataFrame from season averages
    if season_averages:
        season_avg_df = pd.DataFrame(season_averages)
        
        # Ensure all columns from original DataFrame are present
        for col in df.columns:
            if col not in season_avg_df.columns:
                season_avg_df[col] = np.nan
        
        # Reorder columns to match original DataFrame
        season_avg_df = season_avg_df.reindex(columns=df.columns, fill_value=np.nan)
        
        # Combine original data with season averages
        combined_df = pd.concat([df, season_avg_df], ignore_index=True)
    else:
        combined_df = df.copy()
    
    return combined_df

def add_rolling_averages(df):
    """
    Add rolling average fantasy points per game (avg_fppg) for each player
    Week 1 of each season uses previous season's average
    Subsequent weeks use rolling average of current season
    
    Args:
        df: DataFrame with player stats and season averages
    
    Returns:
        DataFrame: Original data with avg_fppg column added
    """
    # Initialize avg_fppg column
    df['avg_fppg'] = np.nan
    
    # Sort by player, season, week for proper rolling calculation
    df_sorted = df.sort_values(['player_id', 'season', 'week']).copy()
    
    # Process each player separately
    for player_id, player_data in df_sorted.groupby('player_id'):
        player_indices = player_data.index
        
        # Separate regular weeks from AVG rows
        regular_weeks = player_data[player_data['week'] != 'AVG'].copy()
        avg_rows = player_data[player_data['week'] == 'AVG'].copy()
        
        # For AVG rows, avg_fppg is the same as fanduel_fantasy_points
        for idx in avg_rows.index:
            df.at[idx, 'avg_fppg'] = df.at[idx, 'fanduel_fantasy_points']
        
        # Process regular weeks by season
        for season, season_data in regular_weeks.groupby('season'):
            season_indices = season_data.index
            season_weeks = season_data.sort_values('week')
            
            for i, (idx, week_data) in enumerate(season_weeks.iterrows()):
                current_week = week_data['week']
                current_season = week_data['season']
                
                if i == 0:  # First week of season
                    # Use previous season's average if available
                    prev_season_avg = avg_rows[avg_rows['season'] == (current_season - 1)]
                    if not prev_season_avg.empty:
                        df.at[idx, 'avg_fppg'] = prev_season_avg['fanduel_fantasy_points'].iloc[0]
                    else:
                        # No previous season data, use 0 or current game points
                        df.at[idx, 'avg_fppg'] = 0
                else:
                    # Use rolling average of current season up to previous week
                    prev_weeks = season_weeks.iloc[:i]
                    if len(prev_weeks) > 0:
                        avg_fppg = prev_weeks['fanduel_fantasy_points'].mean()
                        df.at[idx, 'avg_fppg'] = avg_fppg
                    else:
                        df.at[idx, 'avg_fppg'] = 0
    
    return df

def calculate_season_totals(df):
    """
    Calculate season totals for each player-season combination
    
    Args:
        df: DataFrame with player stats
    
    Returns:
        DataFrame: Season totals for each player-season
    """
    # Define columns to sum
    sum_columns = [
        'passing_yards', 'passing_tds', 'interceptions', 'passing_attempts', 'passing_completions',
        'rushing_yards', 'rushing_tds', 'rushing_attempts', 'rushing_fumbles',
        'receiving_yards', 'receiving_tds', 'receptions', 'targets', 'receiving_fumbles',
        'fanduel_fantasy_points'
    ]
    
    # Filter to only regular season weeks
    regular_weeks = df[df['week'] != 'AVG'].copy()
    
    # Group by player and season to calculate totals
    season_totals = regular_weeks.groupby(['player_id', 'season']).agg({
        **{col: 'sum' for col in sum_columns if col in regular_weeks.columns},
        'games_played': 'count',
        'player_name': 'first',
        'position': 'first',
        'recent_team': 'first'
    }).reset_index()
    
    return season_totals

def get_player_career_stats(df, player_id):
    """
    Get career statistics for a specific player
    
    Args:
        df: DataFrame with player stats
        player_id: Player ID to get stats for
    
    Returns:
        Dict: Career statistics for the player
    """
    player_data = df[df['player_id'] == player_id]
    
    if player_data.empty:
        return {}
    
    # Get regular season games only
    regular_games = player_data[player_data['week'] != 'AVG']
    
    # Calculate career totals
    career_stats = {
        'games_played': len(regular_games),
        'seasons_played': regular_games['season'].nunique(),
        'career_fantasy_points': regular_games['fanduel_fantasy_points'].sum(),
        'avg_fantasy_points_per_game': regular_games['fanduel_fantasy_points'].mean(),
        'best_season': regular_games.groupby('season')['fanduel_fantasy_points'].sum().idxmax(),
        'best_game': regular_games['fanduel_fantasy_points'].max()
    }
    
    return career_stats