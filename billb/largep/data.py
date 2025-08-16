#!/usr/bin/env python3
"""
Enhanced NFL Data Processing Script
Run with: python data.py

This script processes NFL data and creates an enhanced dataset with:
- All original player stats and snap counts
- Season averages for each player (rows with week='AVG')  
- Rolling weekly averages (avg_fppg column)
- FanDuel fantasy points calculations

Output: data/nfl_dataset.csv (same filename as before, but enhanced)
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict
from nfl_data_py import import_weekly_data, import_players, import_weekly_rosters, import_schedules, import_pbp_data, import_depth_charts

def parse_players(player_string):
    """Parse player string into list of player IDs"""
    if isinstance(player_string, str):
        return player_string.split(';')
    elif isinstance(player_string, list):
        return player_string
    else:
        return []

def categorize_play(play_type):
    """Categorize play types for snap count calculations"""
    if play_type in {'run', 'pass', 'no_play'}:
        return 'scrimmage'
    elif play_type in {'kickoff', 'punt', 'field_goal', 'extra_point'}:
        return 'special_teams'
    else:
        return 'other'

def calculate_player_snap_counts(pbp_data):
    """Calculate snap counts for players from play-by-play data"""
    snap_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    team_snap_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    # Create a copy of the DataFrame to avoid the SettingWithCopyWarning
    pbp_data = pbp_data.copy()
    
    # Pre-compute play categories
    pbp_data['play_category'] = pbp_data['play_type'].map(categorize_play)
    
    # Group by game_id for faster processing
    for game_id, game_data in pbp_data.groupby('game_id'):
        for _, play in game_data.iterrows():
            posteam = play['posteam']
            defteam = play['defteam']
            play_category = play['play_category']
            
            if posteam is None and defteam is None:
                if play_category == 'special_teams':
                    posteam = defteam = 'UNKNOWN_TEAM'
                else:
                    continue
            elif posteam is None:
                posteam = defteam
            elif defteam is None:
                defteam = posteam
            
            # Count team snaps
            if play_category == 'special_teams':
                team_snap_counts[game_id]['special_teams'][posteam] += 1
                team_snap_counts[game_id]['special_teams'][defteam] += 1
            elif play_category == 'scrimmage':
                team_snap_counts[game_id]['offense'][posteam] += 1
                team_snap_counts[game_id]['defense'][defteam] += 1
            
            # Count player snaps
            offense_players = parse_players(play['offense_players'])
            defense_players = parse_players(play['defense_players'])
            
            if play_category == 'special_teams':
                for player in set(offense_players + defense_players):
                    if player:
                        snap_counts[player][game_id]['special_teams'] += 1
            elif play_category == 'scrimmage':
                for player in offense_players:
                    if player:
                        snap_counts[player][game_id]['offense'] += 1
                for player in defense_players:
                    if player:
                        snap_counts[player][game_id]['defense'] += 1
    
    # Calculate percentages and create final dataframe
    snap_count_list = []
    for player_id, games in snap_counts.items():
        for game_id, counts in games.items():
            off_snaps = counts['offense']
            def_snaps = counts['defense']
            st_snaps = counts['special_teams']
            total_snaps = off_snaps + def_snaps + st_snaps
            
            # Find player team
            player_team_mask = (
                (pbp_data['game_id'] == game_id) & 
                ((pbp_data['offense_players'].apply(lambda x: player_id in x if isinstance(x, list) else player_id in str(x))) | 
                 (pbp_data['defense_players'].apply(lambda x: player_id in x if isinstance(x, list) else player_id in str(x))))
            )
            
            if player_team_mask.any():
                player_team = pbp_data[player_team_mask]['posteam'].iloc[0]
            else:
                player_team = 'UNKNOWN'
            
            team_off_snaps = team_snap_counts[game_id]['offense'].get(player_team, 0)
            team_def_snaps = team_snap_counts[game_id]['defense'].get(player_team, 0)
            team_st_snaps = team_snap_counts[game_id]['special_teams'].get(player_team, 0)
            
            snap_count_list.append({
                'player_id': player_id,
                'game_id': game_id,
                'week': pbp_data[pbp_data['game_id'] == game_id]['week'].iloc[0],
                'season': pbp_data[pbp_data['game_id'] == game_id]['season'].iloc[0],
                'offensive_snaps': off_snaps,
                'defensive_snaps': def_snaps,
                'special_teams_snaps': st_snaps,
                'total_snaps': total_snaps,
                'offensive_snap_pct': (off_snaps / team_off_snaps * 100) if team_off_snaps > 0 else 0,
                'defensive_snap_pct': (def_snaps / team_def_snaps * 100) if team_def_snaps > 0 else 0,
                'special_teams_snap_pct': (st_snaps / team_st_snaps * 100) if team_st_snaps > 0 else 0
            })
    
    result_df = pd.DataFrame(snap_count_list)
    return result_df

@np.vectorize
def calculate_fanduel_fantasy_points(
    passing_yards, passing_tds, interceptions,
    rushing_yards, rushing_tds,
    receptions, receiving_yards, receiving_tds,
    fumbles, return_tds=0, two_point_conversions=0,
    field_goals_0_39=0, field_goals_40_49=0, field_goals_50_plus=0,
    extra_points=0
):
    """Calculate FanDuel fantasy points based on player stats"""
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

def add_season_averages(df):
    """Add season average rows for each player-season combination"""
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
                avg_row[col] = group[col].mean()
        
        # Calculate games played and fantasy point totals
        avg_row['games_played'] = len(group)
        avg_row['fanduel_fantasy_points_total'] = group['fanduel_fantasy_points'].sum()
        
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
    """Add rolling average fantasy points per game (avg_fppg) for each player"""
    # Initialize avg_fppg column
    df['avg_fppg'] = np.nan
    
    # Sort by player, season, week for proper rolling calculation
    df_sorted = df.sort_values(['player_id', 'season', 'week']).copy()
    
    # Process each player separately
    for player_id, player_data in df_sorted.groupby('player_id'):
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
                        # No previous season data, use 0
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

def create_dataframe(seasons):
    """Create the enhanced dataframe with all stats and averages"""
    current_season = seasons[len(seasons)-1]
    
    # 1. Fetch data
    print("Importing weekly data...")
    player_stats = pd.concat([import_weekly_data(seasons)])
    print("Importing players data...")
    players = import_players()
    print("Importing rosters data...")
    latest_rosters = import_weekly_rosters([current_season])
    print("Importing schedules data...")
    schedule_2024 = import_schedules([current_season])
    print("Importing play-by-play data...")
    pbp_data = import_pbp_data(seasons)
    print("Importing depth charts...")
    depth_charts = import_depth_charts(seasons)

    # 2. Calculate snap counts for all seasons
    snap_counts_list = []
    for season in seasons:
        weeks = range(1, 19) if season >= 2021 else range(1, 18)
        season_pbp = pbp_data[pbp_data['season'] == season]
        for week in weeks:
            print(f"Processing season {season}, week {week} PBP")
            week_pbp = season_pbp[season_pbp['week'] == week]
            if week_pbp.empty:
                print(f"No data for season {season}, week {week} PBP")
                continue
            week_snap_counts = calculate_player_snap_counts(week_pbp)
            if not week_snap_counts.empty:
                snap_counts_list.append(week_snap_counts)
            else:
                print(f"No snap counts for season {season}, week {week} PBP")

    # Combine snap counts
    all_snap_counts = pd.concat(snap_counts_list, ignore_index=True)

    # 3. Merge player stats with players data
    df = pd.merge(player_stats, players[['gsis_id', 'position']], left_on='player_id', right_on='gsis_id', how='left')
    df = df.sort_values(['player_id', 'season', 'week'])

    # 4. Merge with snap count data
    df = pd.merge(df, all_snap_counts, on=['player_id', 'season', 'week'], how='left')

    # 5. Merge with depth chart data
    print("Merging depth chart data...")
    depth_charts_slim = depth_charts[['gsis_id', 'season', 'week', 'position', 'depth_team']]
    df = pd.merge(df, depth_charts_slim, 
                  left_on=['player_id', 'season', 'week'], 
                  right_on=['gsis_id', 'season', 'week'], 
                  how='left', 
                  suffixes=('', '_depth_chart'))

    # 6. Calculate total_snaps
    df['total_snaps'] = df['offensive_snaps'] + df['defensive_snaps'] + df['special_teams_snaps']
    
    # 7. Fill NaN values in snap count columns with 0
    snap_columns = ['offensive_snaps', 'defensive_snaps', 'special_teams_snaps', 'total_snaps', 
                   'offensive_snap_pct', 'defensive_snap_pct', 'special_teams_snap_pct']
    df[snap_columns] = df[snap_columns].fillna(0)

    # 8. Ensure all percentage columns are between 0 and 100
    percentage_columns = ['offensive_snap_pct', 'defensive_snap_pct', 'special_teams_snap_pct']
    df[percentage_columns] = df[percentage_columns].clip(0, 100)

    # 9. Calculate FanDuel fantasy points
    print("Calculating FanDuel fantasy points...")
    df['fanduel_fantasy_points'] = calculate_fanduel_fantasy_points(
        df['passing_yards'], df['passing_tds'], df['interceptions'],
        df['rushing_yards'], df['rushing_tds'],
        df['receptions'], df['receiving_yards'], df['receiving_tds'],
        df['rushing_fumbles'] + df['receiving_fumbles'] + df.get('sack_fumbles', 0)
    )

    # 10. Add season averages (creates season summary rows)
    print("Adding season averages...")
    df = add_season_averages(df)
    
    # 11. Add rolling averages for weekly data
    print("Adding rolling averages...")
    df = add_rolling_averages(df)

    # 12. Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # 13. Save to CSV (same filename as your original script)
    df.to_csv('data/nfl_dataset.csv', index=False)
    print("Enhanced DataFrame saved to data/nfl_dataset.csv")
    return df

def main():
    """Main execution - same interface as your original script"""
    print("Starting enhanced NFL data processing...")
    
    # Use the same seasons as your original script
    seasons = list(range(2018, 2025))
    
    # Process data
    result = create_dataframe(seasons)
    
    if result is not None:
        print("Data processing completed successfully.")
        print(f"Dataset shape: {result.shape}")
        
        # Show summary of enhancements
        season_avg_rows = result[result['week'] == 'AVG']
        print(f"Season average rows added: {len(season_avg_rows)}")
        
        rolling_avg_sample = result[result['avg_fppg'].notna() & (result['week'] != 'AVG')].head(3)
        if not rolling_avg_sample.empty:
            print("\nSample rolling averages:")
            print(rolling_avg_sample[['player_name', 'season', 'week', 'fanduel_fantasy_points', 'avg_fppg']].to_string(index=False))
        
        print(f"\nEnhanced dataset saved to: data/nfl_dataset.csv")
    else:
        print("Data processing failed. Please check the logs for details.")

if __name__ == "__main__":
    main()