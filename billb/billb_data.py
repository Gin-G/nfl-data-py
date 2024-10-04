import pandas as pd
import numpy as np
from collections import defaultdict
from nfl_data_py import import_weekly_data, import_players, import_weekly_rosters, import_schedules, import_pbp_data, import_depth_charts

# 1. Fetch data
seasons = list(range(2023, 2025))  # Include 2024
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
pbp_data_2018_2023 = import_pbp_data([2023])
pbp_data_2024 = import_pbp_data([current_season], include_participation=False)
print("Importing depth charts...")
depth_charts = import_depth_charts(seasons)

def parse_players(player_string):
    if isinstance(player_string, str):
        return player_string.split(';')
    elif isinstance(player_string, list):
        return player_string
    else:
        return []

def categorize_play(play_type):
    if play_type in {'run', 'pass', 'no_play'}:
        return 'scrimmage'
    elif play_type in {'kickoff', 'punt', 'field_goal', 'extra_point'}:
        return 'special_teams'
    else:
        return 'other'

def calculate_player_snap_counts(pbp_data):
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
            
            player_team = pbp_data[(pbp_data['game_id'] == game_id) & 
                                   ((pbp_data['offense_players'].apply(lambda x: player_id in x if isinstance(x, list) else player_id in str(x))) | 
                                    (pbp_data['defense_players'].apply(lambda x: player_id in x if isinstance(x, list) else player_id in str(x))))]['posteam'].iloc[0]
            
            team_off_snaps = team_snap_counts[game_id]['offense'][player_team]
            team_def_snaps = team_snap_counts[game_id]['defense'][player_team]
            team_st_snaps = team_snap_counts[game_id]['special_teams'][player_team]
            
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
    points = (
        passing_yards * 0.04 +
        passing_tds * 4 +
        interceptions * -1 +
        (3 if passing_yards >= 300 else 0) +
        rushing_yards * 0.1 +
        rushing_tds * 6 +
        (3 if rushing_yards >= 100 else 0) +
        receptions * 0.5 +
        receiving_yards * 0.1 +
        receiving_tds * 6 +
        (3 if receiving_yards >= 100 else 0) +
        fumbles * -2 +
        return_tds * 6 +
        two_point_conversions * 2 +
        field_goals_0_39 * 3 +
        field_goals_40_49 * 4 +
        field_goals_50_plus * 5 +
        extra_points * 1
    )
    return points

def create_dataframe(seasons):
    snap_counts_list = []
    for season in seasons:
        weeks = range(1, 19) if season >= 2021 else range(1, 18)
        season_pbp = pbp_data_2018_2023[pbp_data_2018_2023['season'] == season]
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

    # Combine snap counts for 2018-2023 seasons
    all_snap_counts = pd.concat(snap_counts_list, ignore_index=True)

    # Merge player stats with players data
    df = pd.merge(player_stats, players[['gsis_id', 'position']], left_on='player_id', right_on='gsis_id', how='left')
    df = df.sort_values(['player_id', 'season', 'week'])

    # Merge with snap count data for 2018-2023
    df = pd.merge(df, all_snap_counts, on=['player_id', 'season', 'week'], how='left')

    # Merge with depth chart data
    print("Merging depth chart data...")
    depth_charts_slim = depth_charts[['gsis_id', 'season', 'week', 'position', 'depth_team']]
    df = pd.merge(df, depth_charts_slim, 
                  left_on=['player_id', 'season', 'week'], 
                  right_on=['gsis_id', 'season', 'week'], 
                  how='left', 
                  suffixes=('', '_depth_chart'))

    # Handle 2024 data
    df_2024 = df[df['season'] == 2024]

    # Update 2024 data in the main dataframe
    df.update(df_2024)

    # Calculate total_snaps
    df['total_snaps'] = df['offensive_snaps'] + df['defensive_snaps'] + df['special_teams_snaps']
    
    # Fill NaN values in snap count columns with 0
    snap_columns = ['offensive_snaps', 'defensive_snaps', 'special_teams_snaps', 'total_snaps', 'offensive_snap_pct', 'defensive_snap_pct', 'special_teams_snap_pct']
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

    df.to_csv('data/nfl_dataset.csv', index=False)
    print("DataFrame saved to CSV")
    return df

# Main execution
print("Starting data processing...")
result = create_dataframe(seasons)
if result is not None:
    print("Data processing completed successfully.")
else:
    print("Data processing failed. Please check the logs for details.")