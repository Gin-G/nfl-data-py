import pandas as pd
import numpy as np
from data_loader import get_player_data, filter_active_players, get_position_players
from models import predict_with_models
from config import TARGET_COLS
import os

def predict_next_week(player_name, df, models, preprocessor, current_week, current_season):
    """
    Predict next week's stats for a specific player.
    """
    player_data = get_player_data(player_name, df, current_season, current_week)
    
    if player_data is None:
        print(f"Error: No stats found for '{player_name}' in week {current_week}.")
        return None
    
    input_data = player_data[preprocessor.feature_names_in_]
    
    if input_data.isnull().any():
        print(f"Error: Missing data for {player_name}. Null values in: {input_data.columns[input_data.isnull().any()].tolist()}")
        return None
    
    predictions = predict_with_models(models, preprocessor, input_data.to_frame().T)
    
    return {stat: predictions[stat][0] for stat in TARGET_COLS}


def get_all_predictions(position, df, models, preprocessor, current_week, current_season):
    """
    Get predictions for all players in a specific position.
    """
    latest_rosters = filter_active_players(df[(df['season'] == current_season) & (df['week'] == current_week)])
    print(f"Number of active players: {len(latest_rosters)}")

    if position == 'QB':
        players = get_position_players(latest_rosters, position)
    elif position == 'RB':
        players = get_position_players(latest_rosters, position)
    elif position == 'WR':
        players = get_position_players(latest_rosters, position)
    elif position == 'TE':
        players = get_position_players(latest_rosters, position)
    else:
        players = latest_rosters[latest_rosters['position'] == position]

    print(f"Number of {position} players: {len(players)}")

    predictions = []
    for _, player in players.iterrows():
        print(f"Predicting for player: {player['player_name']}")
        pred = predict_next_week(player['player_name'], df, models, preprocessor, current_week, current_season)
        if pred:
            if position == 'QB':
                predictions.append((player['player_name'], pred['passing_yards'], pred['attempts'], pred['completions']))
            elif position == 'RB':
                predictions.append((player['player_name'], pred['rushing_yards'], pred['carries'], pred['receiving_yards']))
            elif position in ['WR', 'TE']:
                predictions.append((player['player_name'], pred['receiving_yards'], pred['targets'], pred['receptions']))
        else:
            print(f"No valid prediction for {player['player_name']}")

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

def store_predictions(predictions, week, season, position):
    """
    Store predictions in a CSV file.
    """
    if position == 'QB':
        columns = ['player_name', 'predicted_passing_yards', 'predicted_attempts', 'predicted_completions']
    elif position == 'RB':
        columns = ['player_name', 'predicted_rushing_yards', 'predicted_carries', 'predicted_receiving_yards']
    else:  # WR and TE
        columns = ['player_name', 'predicted_receiving_yards', 'predicted_targets', 'predicted_receptions']
    
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
            
            merged_data = pd.merge(predictions, actual_data, left_on='player_name', right_on='player_display_name', how='left')
            
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
                'data': merged_data[['player_name', 'predicted_stat', 'actual_stat', 'difference']]
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