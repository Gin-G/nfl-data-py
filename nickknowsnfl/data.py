#!/usr/bin/env python3
"""
Enhanced NFL Data Processing Script
Run with: python data.py

This script processes NFL data and creates an enhanced dataset with:
- All original player stats and snap counts (using nflreadpy.import_snap_counts)
- 2025 data from Sportradar API with proper ID mapping and headshots
- Season averages for each player (rows with week='AVG')  
- Rolling weekly averages (avg_fppg column)
- FanDuel fantasy points calculations

Output: data/nfl_dataset.csv (same filename as before, but enhanced)
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict
import nflreadpy as nfl
from sportradar_nfl_data_collector import collect_2025_data

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
    """Process and clean snap count data from nflreadpy"""
    print("Processing snap count data...")
    
    # Create a copy to avoid modifying original
    processed_data = snap_data.copy()
    
    # Check what the actual player ID column is
    player_id_col = None
    possible_player_cols = ['player', 'player_id', 'player_display_name', 'gsis_id']
    for col in possible_player_cols:
        if col in processed_data.columns:
            player_id_col = col
            break
    
    if player_id_col is None:
        print(f"Warning: Could not find player ID column in snap data")
        return processed_data
    
    # Rename columns to match existing schema
    column_mapping = {
        player_id_col: 'player_id',
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
    
    # Calculate total snaps
    snap_cols = ['offensive_snaps', 'defensive_snaps', 'special_teams_snaps']
    available_snap_cols = [col for col in snap_cols if col in processed_data.columns]
    
    if available_snap_cols:
        processed_data['total_snaps'] = processed_data[available_snap_cols].fillna(0).sum(axis=1)
    
    # Convert percentages from decimals to percentages if needed
    pct_cols = ['offensive_snap_pct', 'defensive_snap_pct', 'special_teams_snap_pct']
    for col in pct_cols:
        if col in processed_data.columns:
            max_val = processed_data[col].max()
            if max_val <= 1.0:
                processed_data[col] = processed_data[col] * 100
    
    # Fill NaN values with 0 for snap counts and percentages
    numeric_cols = (['offensive_snaps', 'defensive_snaps', 'special_teams_snaps', 'total_snaps'] + pct_cols)
    for col in numeric_cols:
        if col in processed_data.columns:
            processed_data[col] = processed_data[col].fillna(0)
    
    # Ensure percentages are within valid range
    for col in pct_cols:
        if col in processed_data.columns:
            processed_data[col] = processed_data[col].clip(0, 100)
    
    return processed_data

def add_season_averages(df):
    """Add season average rows for each player-season combination"""
    avg_columns = [
        'passing_yards', 'passing_tds', 'interceptions', 'attempts', 'completions',
        'rushing_yards', 'rushing_tds', 'carries', 'rushing_fumbles',
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
        avg_row['week'] = 'AVG'
        
        # Copy non-numeric columns from first game
        first_game = group.iloc[0]
        for col in ['player_name', 'player_display_name', 'position', 'recent_team']:
            if col in first_game:
                avg_row[col] = first_game[col]
        
        # Calculate averages for numeric columns
        for col in avg_columns:
            if col in group.columns:
                avg_row[col] = group[col].mean()
        
        # Calculate games played and fantasy point totals
        avg_row['games_played'] = len(group)
        if 'fanduel_fantasy_points' in group.columns:
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
            if 'fanduel_fantasy_points' in df.columns:
                df.at[idx, 'avg_fppg'] = df.at[idx, 'fanduel_fantasy_points']
        
        # Process regular weeks by season
        for season, season_data in regular_weeks.groupby('season'):
            season_weeks = season_data.sort_values('week')
            
            for i, (idx, week_data) in enumerate(season_weeks.iterrows()):
                current_season = week_data['season']
                
                if i == 0:  # First week of season
                    # Use previous season's average if available
                    prev_season_avg = avg_rows[avg_rows['season'] == (current_season - 1)]
                    if not prev_season_avg.empty and 'fanduel_fantasy_points' in prev_season_avg.columns:
                        df.at[idx, 'avg_fppg'] = prev_season_avg['fanduel_fantasy_points'].iloc[0]
                    else:
                        df.at[idx, 'avg_fppg'] = 0
                else:
                    # Use rolling average of current season up to previous week
                    prev_weeks = season_weeks.iloc[:i]
                    if len(prev_weeks) > 0 and 'fanduel_fantasy_points' in prev_weeks.columns:
                        avg_fppg = prev_weeks['fanduel_fantasy_points'].mean()
                        df.at[idx, 'avg_fppg'] = avg_fppg
                    else:
                        df.at[idx, 'avg_fppg'] = 0
    
    return df

def create_dataframe(seasons):
    """Create the enhanced dataframe with all stats and averages"""
    current_season = seasons[-1]
    
    # 1. Fetch historical and current season data
    print("Importing weekly data...")
    historical_seasons = [s for s in seasons if s < 2025]
    current_season_data = []

    # Collect 2025 data from Sportradar if needed
    if 2025 in seasons:
        # Check if cached CSV exists
        cached_2025_file = 'data/sportradar_2025_weeks_1-4.csv'
        
        if os.path.exists(cached_2025_file):
            print(f"Loading cached 2025 data from {cached_2025_file}...")
            df_2025 = pd.read_csv(cached_2025_file)
            print(f"  Loaded {len(df_2025)} records from cache")
        else:
            API_KEY = os.getenv('SPORTRADAR_API_KEY')
            if not API_KEY:
                print("WARNING: SPORTRADAR_API_KEY not set, skipping 2025 data")
                df_2025 = pd.DataFrame()
            else:
                completed_2025_weeks = [1, 2, 3, 4]  # Update as more weeks complete
                print(f"Collecting 2025 weeks: {completed_2025_weeks}")
                df_2025 = collect_2025_data(api_key=API_KEY, weeks=completed_2025_weeks)
        
        if not df_2025.empty:
            # Get the roster data with ID mappings and metadata
                print("Mapping Sportradar IDs to GSIS IDs...")
                rosters_2025 = nfl.load_rosters_weekly(seasons=[2025])
                # Convert from Polars to Pandas if needed
                if hasattr(rosters_2025, 'to_pandas'):
                    rosters_2025 = rosters_2025.to_pandas()
                
                # Debug: print available columns
                print(f"Available roster columns: {rosters_2025.columns.tolist()}")
                
                # Create mapping: sportradar_id -> gsis_id + metadata
                # Check which ID column exists
                id_col = 'gsis_id' if 'gsis_id' in rosters_2025.columns else 'player_id'
                
                required_cols = ['sportradar_id', id_col]
                optional_cols = ['headshot_url', 'position']
                
                # Only include columns that exist
                cols_to_select = required_cols + [c for c in optional_cols if c in rosters_2025.columns]
                
                id_mapping = rosters_2025[cols_to_select].drop_duplicates('sportradar_id')
                
                # Rename to standardize
                if id_col == 'gsis_id':
                    id_mapping = id_mapping.rename(columns={'gsis_id': 'player_id'})
                
                # Drop headshot_url from Sportradar data if it exists (prevents merge conflict)
                df_2025 = df_2025.drop(columns=['headshot_url'], errors='ignore')
                
                # Merge to add GSIS IDs and metadata
                df_2025 = pd.merge(
                    df_2025,
                    id_mapping,
                    left_on='player_id',  # This is sportradar_id from collector
                    right_on='sportradar_id',
                    how='left',
                    suffixes=('_sr', '_nfl')
                )
                
                # Rename for clarity
                df_2025 = df_2025.rename(columns={
                    'player_id_sr': 'player_id_sportradar',
                    'player_id_nfl': 'player_id'
                })
                
                # Use NFL position if available, otherwise use Sportradar position
                if 'position_nfl' in df_2025.columns and 'position_sr' in df_2025.columns:
                    df_2025['position'] = df_2025['position_nfl'].fillna(df_2025['position_sr'])
                    df_2025 = df_2025.drop(columns=['position_sr', 'position_nfl'], errors='ignore')
                
                # Report matching success
                matched = df_2025['player_id'].notna().sum()
                total = len(df_2025)
                print(f"  Matched {matched}/{total} ({matched/total*100:.1f}%) via sportradar_id")
                
                if 'headshot_url' in df_2025.columns:
                    missing_headshots = df_2025['headshot_url'].isna().sum()
                    has_headshots = df_2025['headshot_url'].notna().sum()
                    print(f"  Headshots: {has_headshots} present, {missing_headshots} missing")
                
                current_season_data.append(df_2025)
        else:
            print("  No 2025 data available")

    # Combine historical and current data
    if historical_seasons:
        historical_data = nfl.load_player_stats(seasons=historical_seasons)
        # Convert from Polars to Pandas if needed
        if hasattr(historical_data, 'to_pandas'):
            historical_data = historical_data.to_pandas()
        
        if current_season_data:
            player_stats = pd.concat([historical_data] + current_season_data, ignore_index=True)
        else:
            player_stats = historical_data
    else:
        if current_season_data:
            player_stats = pd.concat(current_season_data, ignore_index=True)
        else:
            raise ValueError("No data to process")
    
    print(f"Total player stat records: {len(player_stats)}")
    
    # 2. Import additional data sources
    print("Importing players data...")
    players = nfl.load_players()
    # Convert from Polars to Pandas if needed
    if hasattr(players, 'to_pandas'):
        players = players.to_pandas()
    
    print("Importing rosters data...")
    latest_rosters = nfl.load_rosters_weekly(seasons=[current_season])
    if hasattr(latest_rosters, 'to_pandas'):
        latest_rosters = latest_rosters.to_pandas()
    
    print("Importing schedules data...")
    schedule = nfl.load_schedules(seasons=[current_season])
    if hasattr(schedule, 'to_pandas'):
        schedule = schedule.to_pandas()
    
    print("Importing depth charts...")
    depth_charts = nfl.load_depth_charts(seasons=seasons)
    if hasattr(depth_charts, 'to_pandas'):
        depth_charts = depth_charts.to_pandas()
    
    print("Importing snap counts...")
    snap_data = nfl.load_snap_counts(seasons=seasons)
    if hasattr(snap_data, 'to_pandas'):
        snap_data = snap_data.to_pandas()
    
    # 3. Process snap count data
    snap_data_processed = process_snap_counts(snap_data)

    # 4. Merge with snap count data
    print("Merging snap count data...")
    
    # Determine the player name column
    if 'player_display_name' in player_stats.columns:
        player_name_col = 'player_display_name'
    else:
        player_name_col = 'player_name'
    
    # Drop conflicting columns from snap data before merge
    snap_data_for_merge = snap_data_processed.rename(columns={'player_id': player_name_col})
    snap_data_for_merge = snap_data_for_merge.drop(
        columns=['position', 'team', 'opponent', 'game_id', 'pfr_game_id', 'game_type'], 
        errors='ignore'
    )
    
    # Merge on player name, season, week
    df = pd.merge(
        player_stats, 
        snap_data_for_merge, 
        on=[player_name_col, 'season', 'week'], 
        how='left'
    )
    
    # Report snap count coverage
    if 'offensive_snaps' in df.columns:
        snap_coverage = df['offensive_snaps'].notna().sum()
        total_records = len(df)
        coverage_pct = (snap_coverage / total_records) * 100
        print(f"Snap count coverage: {snap_coverage:,}/{total_records:,} ({coverage_pct:.1f}%)")

    # 5. Merge with depth chart data
    print("Merging depth chart data...")
    depth_charts_slim = depth_charts[['gsis_id', 'season', 'week', 'position', 'depth_team']].copy()
    df = pd.merge(
        df, 
        depth_charts_slim, 
        left_on=['player_id', 'season', 'week'], 
        right_on=['gsis_id', 'season', 'week'], 
        how='left', 
        suffixes=('', '_depth_chart')
    )
    
    # Drop duplicate gsis_id column if it exists
    if 'gsis_id_depth_chart' in df.columns:
        df = df.drop(columns=['gsis_id_depth_chart'])

    # 6. Ensure snap columns exist and are properly calculated
    snap_columns = ['offensive_snaps', 'defensive_snaps', 'special_teams_snaps', 'total_snaps', 
                   'offensive_snap_pct', 'defensive_snap_pct', 'special_teams_snap_pct']
    
    for col in snap_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Calculate total_snaps if needed
    if df['total_snaps'].isna().all() or (df['total_snaps'] == 0).all():
        df['total_snaps'] = (
            df['offensive_snaps'].fillna(0) + 
            df['defensive_snaps'].fillna(0) + 
            df['special_teams_snaps'].fillna(0)
        )
    
    # Fill NaN values with 0
    for col in snap_columns:
        df[col] = df[col].fillna(0)
    
    # Ensure percentages are within valid range
    percentage_columns = ['offensive_snap_pct', 'defensive_snap_pct', 'special_teams_snap_pct']
    for col in percentage_columns:
        df[col] = df[col].clip(0, 100)

    # 7. Calculate FanDuel fantasy points
    print("Calculating FanDuel fantasy points...")
    fumbles_col = (
        df.get('rushing_fumbles', 0).fillna(0) + 
        df.get('receiving_fumbles', 0).fillna(0) + 
        df.get('sack_fumbles', 0).fillna(0)
    )
    
    df['fanduel_fantasy_points'] = calculate_fanduel_fantasy_points(
        df.get('passing_yards', 0), df.get('passing_tds', 0), df.get('interceptions', 0),
        df.get('rushing_yards', 0), df.get('rushing_tds', 0),
        df.get('receptions', 0), df.get('receiving_yards', 0), df.get('receiving_tds', 0),
        fumbles_col
    )

    # 8. Add season averages
    print("Adding season averages...")
    df = add_season_averages(df)
    
    # 9. Add rolling averages
    print("Adding rolling averages...")
    df = add_rolling_averages(df)

    # 10. Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/nfl_dataset.csv', index=False)
    print("Enhanced DataFrame saved to data/nfl_dataset.csv")
    
    return df

def main():
    """Main execution"""
    print("Starting enhanced NFL data processing...")
    
    # Include all seasons through 2025
    seasons = list(range(2018, 2026))
    
    # Process data
    result = create_dataframe(seasons)
    
    if result is not None:
        print("\nData processing completed successfully.")
        print(f"Dataset shape: {result.shape}")
        
        # Show summary
        season_avg_rows = result[result['week'] == 'AVG']
        print(f"Season average rows added: {len(season_avg_rows)}")
        
        # Show 2025 data summary
        data_2025 = result[result['season'] == 2025]
        if len(data_2025) > 0:
            print(f"\n2025 Season Data:")
            print(f"  Total records: {len(data_2025)}")
            print(f"  Unique players: {data_2025['player_id'].nunique()}")
            weeks_list = sorted([w for w in data_2025['week'].unique() if w != 'AVG'])
            print(f"  Weeks covered: {weeks_list}")
            
            if 'headshot_url' in data_2025.columns:
                headshots = data_2025['headshot_url'].notna().sum()
                print(f"  Headshots: {headshots}/{len(data_2025)} ({headshots/len(data_2025)*100:.1f}%)")
        
        # Show snap count statistics
        snap_cols = ['offensive_snaps', 'defensive_snaps', 'special_teams_snaps', 'total_snaps']
        for col in snap_cols:
            if col in result.columns:
                non_zero = (result[col] > 0).sum()
                print(f"{col}: {non_zero:,} non-zero records")
        
        print(f"\nEnhanced dataset saved to: data/nfl_dataset.csv")
    else:
        print("Data processing failed. Please check the logs for details.")

if __name__ == "__main__":
    main()