import pandas as pd
import numpy as np
from collections import defaultdict
from nfl_data_py import import_weekly_data, import_players, import_rosters, import_schedules, import_pbp_data
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
import os

# 1. Fetch data
seasons = list(range(2018, 2024))  # Adjust the range as needed
player_stats = pd.concat([import_weekly_data(seasons)])
players = import_players()
latest_rosters = import_rosters([2024])  # For 2024 predictions
schedule_2024 = import_schedules([2024])
pbp_data = import_pbp_data(seasons)

def parse_players(player_string):
    if isinstance(player_string, str):
        return player_string.split(';')
    elif isinstance(player_string, list):
        return player_string
    else:
        return []

def categorize_play(play_type):
    if play_type in ['run', 'pass', 'no_play']:
        return 'scrimmage'
    elif play_type in ['kickoff', 'punt', 'field_goal', 'extra_point']:
        return 'special_teams'
    else:
        return 'other'

def calculate_player_snap_counts(pbp_data):
    snap_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    team_snap_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for _, play in pbp_data.iterrows():
        game_id = play['game_id']
        posteam = play['posteam']
        defteam = play['defteam']
        play_type = play['play_type']
        
        offense_players = parse_players(play['offense_players'])
        defense_players = parse_players(play['defense_players'])
        
        # Categorize the play
        play_category = categorize_play(play_type)
        
        # Handle None values for posteam and defteam
        if posteam is None and defteam is None:
            # If both are None and play_type is None, skip this play
            if play_type is None:
                continue
            # For special teams plays, we might not have posteam/defteam info
            if play_category == 'special_teams':
                # Use a placeholder team name for special teams plays with no team info
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
        if play_category == 'special_teams':
            for player in offense_players + defense_players:
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
            
            # Determine the player's team for this game
            player_team = pbp_data[(pbp_data['game_id'] == game_id) & 
                                   ((pbp_data['offense_players'].apply(lambda x: player_id in x if isinstance(x, list) else player_id in str(x))) | 
                                    (pbp_data['defense_players'].apply(lambda x: player_id in x if isinstance(x, list) else player_id in str(x))))]['posteam'].iloc[0]
            
            team_off_snaps = team_snap_counts[game_id]['offense'][player_team]
            team_def_snaps = team_snap_counts[game_id]['defense'][player_team]
            team_st_snaps = team_snap_counts[game_id]['special_teams'][player_team]
            
            off_pct = (off_snaps / team_off_snaps * 100) if team_off_snaps > 0 else 0
            def_pct = (def_snaps / team_def_snaps * 100) if team_def_snaps > 0 else 0
            st_pct = (st_snaps / team_st_snaps * 100) if team_st_snaps > 0 else 0
            
            snap_count_list.append({
                'player_id': player_id,
                'game_id': game_id,
                'week': pbp_data[pbp_data['game_id'] == game_id]['week'].iloc[0],
                'season': pbp_data[pbp_data['game_id'] == game_id]['season'].iloc[0],
                'offensive_snaps': off_snaps,
                'defensive_snaps': def_snaps,
                'special_teams_snaps': st_snaps,
                'total_snaps': total_snaps,
                'offensive_snap_pct': off_pct,
                'defensive_snap_pct': def_pct,
                'special_teams_snap_pct': st_pct
            })
    
    return pd.DataFrame(snap_count_list)

# 2. Preprocess data
# Calculate snap counts for each season
snap_counts_list = []
for season in seasons:
    if season < 2021:
        weeks = list(range(1,17))
    else:
        weeks = list(range(1,18))
    for week in weeks:
        week_pbp = pbp_data[pbp_data['week'] == week]
        week_snap_counts = calculate_player_snap_counts(week_pbp)
        snap_counts_list.append(week_snap_counts)

# Combine snap counts for all seasons
all_snap_counts = pd.concat(snap_counts_list, ignore_index=True)

# Merge player stats with players data
df = pd.merge(player_stats, players, left_on='player_id', right_on='gsis_id', how='left')
df = df.sort_values(['player_id', 'season', 'week'])

# Merge with snap count data
df = pd.merge(df, all_snap_counts, on=['player_id', 'season', 'week'], how='left')

# Fill NaN values in snap count columns with 0
snap_columns = ['offensive_snaps', 'defensive_snaps', 'special_teams_snaps', 'total_snaps', 'offensive_snap_pct', 'defensive_snap_pct', 'special_teams_snap_pct']
for col in snap_columns:
    df[col] = df[col].fillna(0)

# If you need a single 'snap_count' column, you can create it like this:
df['snap_count'] = df['offensive_snaps'] + df['defensive_snaps'] + df['special_teams_snaps']

# Ensure all percentage columns are between 0 and 100
percentage_columns = ['offensive_snap_pct', 'defensive_snap_pct', 'special_teams_snap_pct']
for col in percentage_columns:
    df[col] = df[col].clip(0, 100)

def calculate_fanduel_fantasy_points(
    passing_yards, passing_tds, interceptions,
    rushing_yards, rushing_tds,
    receptions, receiving_yards, receiving_tds,
    fumbles, return_tds=0, two_point_conversions=0,
    field_goals_0_39=0, field_goals_40_49=0, field_goals_50_plus=0,
    extra_points=0
):
    points = 0
    
    # Passing
    points += passing_yards * 0.04
    points += passing_tds * 4
    points += interceptions * -1
    points += 3 if passing_yards >= 300 else 0
    
    # Rushing
    points += rushing_yards * 0.1
    points += rushing_tds * 6
    points += 3 if rushing_yards >= 100 else 0
    
    # Receiving
    points += receptions * 0.5
    points += receiving_yards * 0.1
    points += receiving_tds * 6
    points += 3 if receiving_yards >= 100 else 0
    
    # Misc
    points += fumbles * -2
    points += return_tds * 6
    points += two_point_conversions * 2
    
    # Kicking
    points += field_goals_0_39 * 3
    points += field_goals_40_49 * 4
    points += field_goals_50_plus * 5
    points += extra_points * 1
    
    return points

# Function to apply to DataFrame
def fanduel_fantasy_points(row):
    return calculate_fanduel_fantasy_points(
        row['passing_yards'], row['passing_tds'], row['interceptions'],
        row['rushing_yards'], row['rushing_tds'],
        row['receptions'], row['receiving_yards'], row['receiving_tds'],
        row['rushing_fumbles'] + row['receiving_fumbles'] + row.get('sack_fumbles', 0)
    )

# Create target variables
target_cols = [
                'passing_yards', 'attempts', 'completions', 'passing_tds', 'interceptions','sack_yards','sack_fumbles','sack_fumbles_lost',
                'rushing_yards', 'carries', 'rushing_tds', 'rushing_fumbles','rushing_fumbles_lost',
                'receiving_yards', 'targets', 'receptions', 'receiving_tds', 'receiving_fumbles', 'receiving_fumbles_lost', 
                'offensive_snap_pct', 'fanduel_fantasy_points'
            ]

# Assuming you have a DataFrame called 'df' with your data
df['fanduel_fantasy_points'] = df.apply(fanduel_fantasy_points, axis=1)

for col in target_cols:
    df[f'next_week_{col}'] = df.groupby(['player_id', 'season'])[col].shift(-1)

# Remove last week of each season for each player
df = df.groupby(['player_id', 'season']).apply(lambda x: x.iloc[:-1]).reset_index(drop=True)

# Identify features
categorical_features = ['position_x', 'recent_team']
numerical_features = target_cols

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_features)
    ])

# Prepare features
X = df[categorical_features + numerical_features]

# Fit preprocessor and transform data
X_transformed = preprocessor.fit_transform(X)

# Get feature names after preprocessing
feature_names = (numerical_features + 
                 preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist())

# Build and train models
input_dim = X_transformed.shape[1]

def create_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Create and train models for each stat
models = {}
for stat in target_cols:
    y = df[f'next_week_{stat}'].values
    mask = ~np.isnan(y)
    X_stat = X_transformed[mask]
    y_stat = y[mask]
    
    models[stat] = create_model(input_dim)
    models[stat].fit(X_stat, y_stat, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# 4. Prediction function
def predict_2024_week1(player_name):
    try:
        player_matches = latest_rosters[latest_rosters['player_name'].str.contains(player_name, case=False, na=False)]
        
        if player_matches.empty:
            print(f"Error: Player '{player_name}' not found in the latest roster.")
            return None
        
        if len(player_matches) > 1:
            print(f"Multiple matches found for '{player_name}'. Using the first match.")
        
        player_info = player_matches.iloc[0]
        
        player_stats = df[(df['player_id'] == player_info['player_id']) & (df['season'] == 2023)]
        
        if player_stats.empty:
            #print(f"Error: No 2023 stats found for '{player_name}'.")
            return None
        
        last_stats = player_stats.sort_values('week').iloc[-1]
        
        input_data = pd.DataFrame({col: [last_stats[col]] for col in numerical_features})
        input_data['position_x'] = [player_info['position']]
        input_data['recent_team'] = [player_info['team']]
        
        input_transformed = preprocessor.transform(input_data)
        
        predictions = {stat: round(models[stat].predict(input_transformed)[0][0], 2) for stat in target_cols}
        
        return predictions
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def get_all_predictions(position):
    predictions = []
    
    if position in ['QB', 'RB', 'WR', 'TE']:
        players = latest_rosters[latest_rosters['position'] == position]
    else:
        players = latest_rosters[latest_rosters['position'] == position]

    for _, player in players.iterrows():
        pred = predict_2024_week1(player['player_name'])
        if pred:
            if position == 'QB':
                predictions.append({
                    'player_name': player['player_name'],
                    'passing_yards': pred['passing_yards'],
                    'attempts': pred['attempts'],
                    'completions': pred['completions'],
                    'interceptions': pred['interceptions'],
                    'passing_tds': pred['passing_tds'],
                    'sack_yards': pred['sack_yards'],
                    'sack_fumbles': pred['sack_fumbles'],
                    'sack_fumbles_lost': pred['sack_fumbles_lost'],
                    'carries': pred['carries'],
                    'rushing_yards': pred['rushing_yards'],
                    'rushing_tds': pred['rushing_tds'],
                    'offensive_snap_pct': pred['offensive_snap_pct'],
                    'fanduel_fantasy_points': pred['fanduel_fantasy_points']
                })
            elif position == 'RB':
                predictions.append({
                    'player_name': player['player_name'],
                    'carries': pred['carries'],
                    'rushing_yards': pred['rushing_yards'],
                    'rushing_tds': pred['rushing_tds'],
                    'rushing_fumbles': pred['rushing_fumbles'],
                    'rushing_fumbles_lost': pred['rushing_fumbles_lost'],
                    'targets': pred['targets'],
                    'receptions': pred['receptions'],
                    'receiving_yards': pred['receiving_yards'],
                    'receiving_tds': pred['receiving_tds'],
                    'receiving_fumbles': pred['receiving_fumbles'],
                    'receiving_fumbles_lost': pred['receiving_fumbles_lost'],
                    'offensive_snap_pct': pred['offensive_snap_pct'],
                    'fanduel_fantasy_points': pred['fanduel_fantasy_points']
                })
            elif position in ['WR', 'TE']:
                predictions.append({
                    'player_name': player['player_name'],
                    'carries': pred['carries'],
                    'rushing_yards': pred['rushing_yards'],
                    'rushing_tds': pred['rushing_tds'],
                    'rushing_fumbles': pred['rushing_fumbles'],
                    'rushing_fumbles_lost': pred['rushing_fumbles_lost'],
                    'targets': pred['targets'],
                    'receptions': pred['receptions'],
                    'receiving_yards': pred['receiving_yards'],
                    'receiving_tds': pred['receiving_tds'],
                    'receiving_fumbles': pred['receiving_fumbles'],
                    'receiving_fumbles_lost': pred['receiving_fumbles_lost'],
                    'offensive_snap_pct': pred['offensive_snap_pct'],
                    'fanduel_fantasy_points': pred['fanduel_fantasy_points']
                })
    
    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    if not predictions_df.empty:
        # Sort the DataFrame by the first prediction column (assuming it's the most important stat)
        sort_column = predictions_df.columns[1]  # First prediction column after player_name
        predictions_df = predictions_df.sort_values(by=sort_column, ascending=False)

    # Create directory if it doesn't exist
    os.makedirs('predictions', exist_ok=True)

    # Save to CSV
    csv_filename = f'predictions/{position}_predictions_week1_season2024.csv'
    predictions_df.to_csv(csv_filename, index=False)
    print(f"Predictions saved to {csv_filename}")

    # Convert DataFrame back to list of tuples for compatibility with existing code
    predictions_list = predictions_df.values.tolist()
    
    return predictions_list

positions = ['QB', 'RB', 'WR', 'TE']
for position in positions:
    print(f"\nProcessing {position}")
    predictions = get_all_predictions(position)
    print(f"Number of predictions for {position}: {len(predictions)}")
    for pred in predictions[:10]:  # Print top 10 predictions
        print(pred)
