import pandas as pd
import numpy as np
from data_loader import get_player_data, filter_active_players, get_position_players
from models import predict_with_models
from config import TARGET_COLS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
import os

def predict_next_week(player_name, df, models, preprocessor, current_week, current_season):
    """
    Predict stats for a player for the next week.
    """
    print(f"\nDebugging predict_next_week for {player_name}")
    
    if current_week == 1:
        player_data = df[(df['player_display_name'] == player_name) & 
                         (df['season'] == current_season - 1)].sort_values('week').tail(16)
    else:
        player_data = df[(df['player_display_name'] == player_name) & 
                         (df['season'] == current_season) & 
                         (df['week'] < current_week)]

    print(f"Player data shape: {player_data.shape}")
    
    if len(player_data) < 1:
        print(f"Not enough data for {player_name}")
        return None

    position = player_data['position_x'].iloc[0]
    print(f"Player position: {position}")

    # Use the most recent game for prediction
    X = player_data.iloc[-1:] 

    # Add missing numerical columns with zero values
    for feature in NUMERICAL_FEATURES:
        if feature not in X.columns:
            X[feature] = 0

    # Ensure all categorical features are present
    for feature in CATEGORICAL_FEATURES:
        if feature not in X.columns:
            X[feature] = 'Unknown'  # or some appropriate default value

    # Ensure all required features are in the correct order
    X = X[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    X = preprocessor.transform(X)

    # Define target columns based on position
    if position == 'QB':
        target_cols = ['passing_yards', 'attempts', 'completions']
    elif position == 'RB':
        target_cols = ['rushing_yards', 'carries', 'receiving_yards']
    elif position in ['WR', 'TE']:
        target_cols = ['receiving_yards', 'targets', 'receptions']
    else:
        print(f"Unsupported position: {position}")
        return None

    print(f"Available models: {list(models.keys())}")
    predictions = {}
    for target in target_cols:
        model_key = f"{position}_{target}"
        if model_key in models:
            predictions[target] = models[model_key].predict(X)[0]
        else:
            print(f"No model found for {model_key}")
            predictions[target] = 0

    return predictions

def get_all_predictions(position, df, models, preprocessor, current_week, current_season):
    """
    Get predictions for all players in a specific position.
    """
    print(f"\nDebugging get_all_predictions for {position}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Unique seasons: {df['season'].unique()}")
    print(f"Unique weeks: {df['week'].unique()}")
    
    if current_week == 1:
        # For Week 1, use the last week of the previous season
        latest_rosters = df[(df['season'] == current_season - 1) & (df['week'] == df[df['season'] == current_season - 1]['week'].max())]
    else:
        latest_rosters = df[(df['season'] == current_season) & (df['week'] == current_week - 1)]
    
    print(f"Latest rosters shape: {latest_rosters.shape}")
    print(f"Unique positions in latest rosters: {latest_rosters['position_x'].unique()}")
    
    latest_rosters = filter_active_players(latest_rosters)
    print(f"Active players after filtering: {len(latest_rosters)}")

    players = get_position_players(latest_rosters, position)
    print(f"Number of {position} players: {len(players)}")
    print(f"Unique {position} players: {players['player_display_name'].unique()}")

    if position in ['QB', 'RB', 'WR', 'TE']:
        players = get_position_players(latest_rosters, position)
    else:
        players = latest_rosters[latest_rosters['position_x'] == position]

    print(f"Number of {position} players: {len(players)}")

    predictions = []
    for _, player in players.iterrows():
        print(f"Predicting for player: {player['player_display_name']}")
        pred = predict_next_week(player['player_display_name'], df, models, preprocessor, current_week, current_season)
        if pred:
            if position == 'QB':
                predictions.append((player['player_display_name'], pred['passing_yards'], pred['attempts'], pred['completions']))
            elif position == 'RB':
                predictions.append((player['player_display_name'], pred['rushing_yards'], pred['carries'], pred['receiving_yards']))
            elif position in ['WR', 'TE']:
                predictions.append((player['player_display_name'], pred['receiving_yards'], pred['targets'], pred['receptions']))
        else:
            print(f"No valid prediction for {player['player_display_name']}")

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

def store_predictions(predictions, week, season, position):
    """
    Store predictions in a CSV file.
    """
    if position == 'QB':
        columns = ['player_display_name', 'predicted_passing_yards', 'predicted_attempts', 'predicted_completions']
    elif position == 'RB':
        columns = ['player_display_name', 'predicted_rushing_yards', 'predicted_carries', 'predicted_receiving_yards']
    else:  # WR and TE
        columns = ['player_display_name', 'predicted_receiving_yards', 'predicted_targets', 'predicted_receptions']
    
    prediction_df = pd.DataFrame(predictions, columns=columns)
    prediction_df['week'] = week
    prediction_df['season'] = season
    prediction_df['position'] = position
    
    filename = f'predictions_{position}_week_{week}_{season}.csv'
    prediction_df.to_csv('data/' + filename, index=False)
    print(f"Predictions saved to {filename}")

def compare_predictions_to_actual(week, season):
    """
    Compare stored predictions to actual results.
    """
    from data_loader import import_weekly_data
    
    actual_data = import_weekly_data([season])
    actual_data = actual_data[actual_data['week'] == week]
    
    results = []
    
    for position in ['QB', 'RB', 'WR', 'TE']:
        prediction_file = f'predictions_{position}_week_{week}_{season}.csv'
        prediction_file = 'data/' + prediction_file
        if os.path.exists(prediction_file):
            predictions = pd.read_csv(prediction_file)
            
            merged_data = pd.merge(predictions, actual_data, left_on='player_display_name', right_on='player_display_name', how='left')
            
            if position == 'QB':
                merged_data['actual_stat'] = merged_data['passing_yards']
                merged_data['predicted_stat'] = merged_data['predicted_passing_yards']
                stat_name = 'passing yards'
            elif position == 'RB':
                merged_data['actual_stat'] = merged_data['rushing_yards']
                merged_data['predicted_stat'] = merged_data['predicted_rushing_yards']
                stat_name = 'rushing yards'
            else:
                merged_data['actual_stat'] = merged_data['receiving_yards']
                merged_data['predicted_stat'] = merged_data['predicted_receiving_yards']
                stat_name = 'receiving yards'
            
            merged_data['difference'] = merged_data['predicted_stat'] - merged_data['actual_stat']
            
            mse = np.mean(merged_data['difference']**2)
            results.append({
                'position': position,
                'stat': stat_name,
                'mse': mse,
                'data': merged_data[['player_display_name', 'predicted_stat', 'actual_stat', 'difference']]
            })
    
    return results

def generate_fantasy_predictions(df, models, preprocessor, current_week, current_season):
    """
    Generate fantasy point predictions for all relevant players.
    """
    fantasy_predictions = []
    
    for position in ['QB', 'RB', 'WR', 'TE']:
        predictions = get_all_predictions(position, df, models, preprocessor, current_week, current_season)
        for player in predictions:
            fantasy_points = calculate_fantasy_points(player, position)
            fantasy_predictions.append((player[0], position, fantasy_points))
    
    fantasy_predictions.sort(key=lambda x: x[2], reverse=True)
    return fantasy_predictions

def calculate_fantasy_points(player_prediction, position):
    """
    Calculate fantasy points based on predicted stats.
    """
    if position == 'QB':
        return (player_prediction[1] * 0.04 +  # passing yards
                player_prediction[3] * 4 +     # passing TDs (assuming 1 TD per 25 completions)
                player_prediction[2] * 0.1)    # rushing yards (assuming some QB rushes)
    elif position == 'RB':
        return (player_prediction[1] * 0.1 +   # rushing yards
                player_prediction[2] * 0.5 +   # receptions (assuming PPR)
                player_prediction[3] * 0.1)    # receiving yards
    else:  # WR and TE
        return (player_prediction[1] * 0.1 +   # receiving yards
                player_prediction[3] * 0.5)    # receptions (assuming PPR)