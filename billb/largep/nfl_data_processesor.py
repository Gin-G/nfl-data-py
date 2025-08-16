# nfl_data_processor.py
import pandas as pd
import numpy as np
from collections import defaultdict
from nfl_data_py import import_weekly_data, import_players, import_weekly_rosters, import_schedules, import_pbp_data, import_depth_charts

# Import separate modules for better organization
from fantasy_calculations import calculate_fanduel_fantasy_points
from snap_calculations import calculate_player_snap_counts
from averaging_utils import add_season_averages, add_rolling_averages

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

def load_nfl_data():
    """Load all NFL data needed for processing"""
    seasons = list(range(2018, 2025))
    current_season = seasons[len(seasons)-1]
    
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
    
    return {
        'player_stats': player_stats,
        'players': players,
        'latest_rosters': latest_rosters,
        'schedule_2024': schedule_2024,
        'pbp_data': pbp_data,
        'depth_charts': depth_charts,
        'seasons': seasons
    }

def create_dataframe(data_dict):
    """Create the main dataframe with all stats and averages"""
    seasons = data_dict['seasons']
    player_stats = data_dict['player_stats']
    players = data_dict['players']
    pbp_data = data_dict['pbp_data']
    depth_charts = data_dict['depth_charts']
    
    # Calculate snap counts for all seasons
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
            week_snap_counts = calculate_player_snap_counts(week_pbp, categorize_play, parse_players)
            if not week_snap_counts.empty:
                snap_counts_list.append(week_snap_counts)
            else:
                print(f"No snap counts for season {season}, week {week} PBP")

    # Combine snap counts
    all_snap_counts = pd.concat(snap_counts_list, ignore_index=True)

    # Merge player stats with players data
    df = pd.merge(player_stats, players[['gsis_id', 'position']], left_on='player_id', right_on='gsis_id', how='left')
    df = df.sort_values(['player_id', 'season', 'week'])

    # Merge with snap count data
    df = pd.merge(df, all_snap_counts, on=['player_id', 'season', 'week'], how='left')

    # Merge with depth chart data
    print("Merging depth chart data...")
    depth_charts_slim = depth_charts[['gsis_id', 'season', 'week', 'position', 'depth_team']]
    df = pd.merge(df, depth_charts_slim, 
                  left_on=['player_id', 'season', 'week'], 
                  right_on=['gsis_id', 'season', 'week'], 
                  how='left', 
                  suffixes=('', '_depth_chart'))

    # Calculate total_snaps
    df['total_snaps'] = df['offensive_snaps'] + df['defensive_snaps'] + df['special_teams_snaps']
    
    # Fill NaN values in snap count columns with 0
    snap_columns = ['offensive_snaps', 'defensive_snaps', 'special_teams_snaps', 'total_snaps', 
                   'offensive_snap_pct', 'defensive_snap_pct', 'special_teams_snap_pct']
    df[snap_columns] = df[snap_columns].fillna(0)

    # Ensure all percentage columns are between 0 and 100
    percentage_columns = ['offensive_snap_pct', 'defensive_snap_pct', 'special_teams_snap_pct']
    df[percentage_columns] = df[percentage_columns].clip(0, 100)

    # Calculate FanDuel fantasy points
    df['fanduel_fantasy_points'] = calculate_fanduel_fantasy_points(
        df['passing_yards'], df['passing_tds'], df['interceptions'],
        df['rushing_yards'], df['rushing_tds'],
        df['receptions'], df['receiving_yards'], df['receiving_tds'],
        df['rushing_fumbles'] + df['receiving_fumbles'] + df.get('sack_fumbles', 0)
    )

    # Add season averages (creates season summary rows)
    print("Adding season averages...")
    df = add_season_averages(df)
    
    # Add rolling averages for weekly data
    print("Adding rolling averages...")
    df = add_rolling_averages(df)

    # Save to CSV
    df.to_csv('data/nfl_dataset_enhanced.csv', index=False)
    print("Enhanced DataFrame saved to CSV")
    return df

def main():
    """Main execution function"""
    print("Starting enhanced data processing...")
    
    # Load all data
    data_dict = load_nfl_data()
    
    # Process and create dataframe
    result = create_dataframe(data_dict)
    
    if result is not None:
        print("Enhanced data processing completed successfully.")
        print(f"Dataset shape: {result.shape}")
        
        # Show some sample data
        season_avg_rows = result[result['week'] == 'AVG']
        print(f"Season average rows added: {len(season_avg_rows)}")
        
        rolling_avg_sample = result[result['avg_fppg'].notna() & (result['week'] != 'AVG')].head()
        print(f"Sample of rolling averages:\n{rolling_avg_sample[['player_id', 'season', 'week', 'fanduel_fantasy_points', 'avg_fppg']]}")
    else:
        print("Enhanced data processing failed. Please check the logs for details.")

if __name__ == "__main__":
    main()