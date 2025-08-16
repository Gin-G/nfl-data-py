#!/usr/bin/env python3
"""
Enhanced NFL Data Processing Script
Run with: python data.py

This script processes NFL data and creates an enhanced dataset with:
- All original player stats and snap counts (using nfl_data_py.import_snap_counts)
- Season averages for each player (rows with week='AVG')  
- Rolling weekly averages (avg_fppg column)
- FanDuel fantasy points calculations

Output: data/nfl_dataset.csv (same filename as before, but enhanced)
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict
from nfl_data_py import import_weekly_data, import_players, import_weekly_rosters, import_schedules, import_snap_counts, import_depth_charts

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

def process_snap_counts(snap_data):
    """Process and clean snap count data from nfl_data_py"""
    print("Processing snap count data...")
    print(f"Available columns in snap data: {list(snap_data.columns)}")
    
    # Create a copy to avoid modifying original
    processed_data = snap_data.copy()
    
    # Check what the actual player ID column is
    player_id_col = None
    possible_player_cols = ['player', 'player_id', 'player_display_name', 'gsis_id']
    for col in possible_player_cols:
        if col in processed_data.columns:
            player_id_col = col
            print(f"Found player ID column: {col}")
            break
    
    if player_id_col is None:
        print(f"Warning: Could not find player ID column in snap data")
        print(f"Available columns: {list(processed_data.columns)}")
        return processed_data
    
    # Rename columns to match existing schema
    column_mapping = {
        player_id_col: 'player_id',  # Use the actual player column found
        'offense_snaps': 'offensive_snaps',
        'offense_pct': 'offensive_snap_pct', 
        'defense_snaps': 'defensive_snaps',
        'defense_pct': 'defensive_snap_pct',
        'st_snaps': 'special_teams_snaps',
        'st_pct': 'special_teams_snap_pct'
    }
    
    # Apply column renaming where columns exist
    for old_col, new_col in column_mapping.items():
        if old_col in processed_data.columns:
            processed_data = processed_data.rename(columns={old_col: new_col})
            print(f"Renamed {old_col} -> {new_col}")
    
    # Print sample of player IDs to debug
    if 'player_id' in processed_data.columns:
        print(f"Sample player IDs: {processed_data['player_id'].head().tolist()}")
    
    # Calculate total snaps
    snap_cols = ['offensive_snaps', 'defensive_snaps', 'special_teams_snaps']
    available_snap_cols = [col for col in snap_cols if col in processed_data.columns]
    
    if available_snap_cols:
        processed_data['total_snaps'] = processed_data[available_snap_cols].fillna(0).sum(axis=1)
        print(f"Calculated total_snaps using columns: {available_snap_cols}")
    
    # Convert percentages from decimals to percentages if needed
    pct_cols = ['offensive_snap_pct', 'defensive_snap_pct', 'special_teams_snap_pct']
    for col in pct_cols:
        if col in processed_data.columns:
            # Check if values are in decimal format (0-1) and convert to percentage
            max_val = processed_data[col].max()
            if max_val <= 1.0:
                processed_data[col] = processed_data[col] * 100
                print(f"Converted {col} from decimal to percentage")
    
    # Fill NaN values with 0 for snap counts and percentages
    numeric_cols = (['offensive_snaps', 'defensive_snaps', 'special_teams_snaps', 'total_snaps'] + 
                   pct_cols)
    for col in numeric_cols:
        if col in processed_data.columns:
            processed_data[col] = processed_data[col].fillna(0)
    
    # Ensure percentages are within valid range
    for col in pct_cols:
        if col in processed_data.columns:
            processed_data[col] = processed_data[col].clip(0, 100)
    
    print(f"Processed {len(processed_data)} snap count records")
    print(f"Final snap data columns: {list(processed_data.columns)}")
    
    # Show sample of processed data
    if len(processed_data) > 0:
        print("Sample processed snap data:")
        sample_cols = ['player_id', 'season', 'week', 'offensive_snaps', 'defensive_snaps', 'special_teams_snaps']
        available_sample_cols = [col for col in sample_cols if col in processed_data.columns]
        print(processed_data[available_sample_cols].head())
    
    return processed_data

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
    print("Importing depth charts...")
    depth_charts = import_depth_charts(seasons)

    # 2. Import snap counts using nfl_data_py (much faster and more reliable)
    print("Importing snap counts...")
    snap_data = import_snap_counts(seasons)
    print(f"Loaded {len(snap_data)} snap count records")
    
    # Process snap count data to match our schema
    snap_data_processed = process_snap_counts(snap_data)

    # 3. Merge player stats with players data
    df = pd.merge(player_stats, players[['gsis_id', 'position']], left_on='player_id', right_on='gsis_id', how='left')
    df = df.sort_values(['player_id', 'season', 'week'])

    # 4. Merge with snap count data
    print("Merging snap count data...")
    
    # Debug information before merge
    print(f"Player stats data shape: {df.shape}")
    print(f"Player stats columns: {list(df.columns)}")
    
    # Check if player_display_name exists
    if 'player_display_name' in df.columns:
        print(f"Found player_display_name in stats data")
        print(f"Player stats player_display_name sample: {df['player_display_name'].head().tolist()}")
        player_name_col = 'player_display_name'
    else:
        print(f"No player_display_name found, using player_name")
        print(f"Player stats player_name sample: {df['player_name'].head().tolist()}")
        player_name_col = 'player_name'
    
    print(f"Snap data player names sample: {snap_data_processed['player_id'].head().tolist()}")
    
    # Since snap data uses player names and stats data uses GSIS IDs,
    # we need to merge on the appropriate name column
    snap_data_for_merge = snap_data_processed.rename(columns={'player_id': player_name_col})
    
    # Check for common players between datasets by name
    common_players = set(df[player_name_col].dropna()) & set(snap_data_for_merge[player_name_col].dropna())
    print(f"Common players between datasets (by {player_name_col}): {len(common_players)}")
    
    if len(common_players) > 0:
        print(f"Sample common players: {list(common_players)[:5]}")
    else:
        # Show some examples to help debug name format differences
        print(f"Sample player names from stats: {df[player_name_col].dropna().head(10).tolist()}")
        print(f"Sample player names from snaps: {snap_data_for_merge[player_name_col].dropna().head(10).tolist()}")
    
    # Check season/week overlap
    if len(common_players) > 0:
        df_keys = set(df[[player_name_col, 'season', 'week']].dropna().apply(tuple, axis=1))
        snap_keys = set(snap_data_for_merge[[player_name_col, 'season', 'week']].dropna().apply(tuple, axis=1))
        common_keys = df_keys & snap_keys
        print(f"Common ({player_name_col}, season, week) combinations: {len(common_keys)}")
    
    # Merge on the appropriate name column, season, week
    df = pd.merge(df, snap_data_for_merge, on=[player_name_col, 'season', 'week'], how='left')
    
    # Report snap count coverage
    snap_coverage = df['offensive_snaps'].notna().sum()
    total_records = len(df)
    coverage_pct = (snap_coverage / total_records) * 100
    print(f"Snap count coverage: {snap_coverage:,}/{total_records:,} ({coverage_pct:.1f}%)")

    # 5. Merge with depth chart data
    print("Merging depth chart data...")
    depth_charts_slim = depth_charts[['gsis_id', 'season', 'week', 'position', 'depth_team']]
    df = pd.merge(df, depth_charts_slim, 
                  left_on=['player_id', 'season', 'week'], 
                  right_on=['gsis_id', 'season', 'week'], 
                  how='left', 
                  suffixes=('', '_depth_chart'))

    # 6. Ensure total_snaps is calculated properly
    if 'total_snaps' not in df.columns or df['total_snaps'].isna().all():
        df['total_snaps'] = df['offensive_snaps'].fillna(0) + df['defensive_snaps'].fillna(0) + df['special_teams_snaps'].fillna(0)
    
    # 7. Fill NaN values in snap count columns with 0
    snap_columns = ['offensive_snaps', 'defensive_snaps', 'special_teams_snaps', 'total_snaps', 
                   'offensive_snap_pct', 'defensive_snap_pct', 'special_teams_snap_pct']
    for col in snap_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 8. Ensure all percentage columns are between 0 and 100
    percentage_columns = ['offensive_snap_pct', 'defensive_snap_pct', 'special_teams_snap_pct']
    for col in percentage_columns:
        if col in df.columns:
            df[col] = df[col].clip(0, 100)

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
        
        # Show snap count statistics
        snap_cols = ['offensive_snaps', 'defensive_snaps', 'special_teams_snaps', 'total_snaps']
        for col in snap_cols:
            if col in result.columns:
                non_zero = (result[col] > 0).sum()
                print(f"{col}: {non_zero:,} non-zero records")
        
        rolling_avg_sample = result[result['avg_fppg'].notna() & (result['week'] != 'AVG')].head(3)
        if not rolling_avg_sample.empty:
            print("\nSample rolling averages:")
            print(rolling_avg_sample[['player_name', 'season', 'week', 'fanduel_fantasy_points', 'avg_fppg']].to_string(index=False))
        
        print(f"\nEnhanced dataset saved to: data/nfl_dataset.csv")
    else:
        print("Data processing failed. Please check the logs for details.")

if __name__ == "__main__":
    main()