from models import create_and_train_models
from nfl_data_py import import_rosters
import pandas as pd

df = pd.read_csv('data/nfl_dataset.csv')

models, preprocessor = create_and_train_models(df)

latest_rosters = import_rosters([2024])

def predict_2024_week1(player_name):
    try:
        # Find the player in the latest roster
        player_matches = latest_rosters[latest_rosters['player_name'].str.contains(player_name, case=False, na=False)]
        
        if player_matches.empty:
            print(f"Error: Player '{player_name}' not found in the latest roster.")
            return None
        
        if len(player_matches) > 1:
            print(f"Multiple matches found for '{player_name}'. Using the first match.")
        
        player_info = player_matches.iloc[0]
        
        # Get the player's stats from the last available week
        player_stats = df[(df['player_id'] == player_info['player_id']) & (df['season'] == 2023)]
        
        if player_stats.empty:
            print(f"Error: No 2023 stats found for '{player_name}'.")
            return None
        
        last_stats = player_stats.sort_values('week').iloc[-1]
        
        # Prepare input data
        input_data = pd.DataFrame({
            'position_x': [player_info['position']],
            'recent_team': [player_info['team']],
            'passing_yards': [last_stats['passing_yards']],
            'rushing_yards': [last_stats['rushing_yards']],
            'receiving_yards': [last_stats['receiving_yards']]
        })
        
        # Transform input data using the same preprocessor
        input_transformed = preprocessor.transform(input_data)
        
        # Make predictions
        predictions = {stat: models[stat].predict(input_transformed)[0][0] for stat in models}
        
        print(f"2024 Week 1 Prediction for {player_name}:")
        print(f"Team: {player_info['team']}")
        print(f"Position: {player_info['position']}")
        
        if player_info['position'] in ['QB']:
            print(f"Predicted passing yards: {predictions['passing_yards']:.2f}")
            print(f"Predicted rushing yards: {predictions['rushing_yards']:.2f}")
        elif player_info['position'] in ['RB']:
            print(f"Predicted rushing yards: {predictions['rushing_yards']:.2f}")
            print(f"Predicted receiving yards: {predictions['receiving_yards']:.2f}")
        elif player_info['position'] in ['WR', 'TE']:
            print(f"Predicted receiving yards: {predictions['receiving_yards']:.2f}")
        
        return predictions
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def get_all_predictions(position):
    predictions = []
    
    if position == 'QB':
        # Filter for QBs who are listed as starters in week 1
        players = latest_rosters[(latest_rosters['position'] == position) & 
                                 (latest_rosters['player_name'].isin(starting_qbs))]
    elif position == 'RB':
        # For RBs, we'll still use depth chart as before
        players = latest_rosters[(latest_rosters['position'] == position) & 
                                 (latest_rosters['depth_chart_position'].isin(['1', '2']))]
    elif position == 'WR':
        players = latest_rosters[(latest_rosters['position'] == position) & 
                                 (latest_rosters['depth_chart_position'].isin(['1', '2', '3']))]
    elif position == 'TE':
        players = latest_rosters[(latest_rosters['position'] == position) & 
                                 (latest_rosters['depth_chart_position'].isin(['1', '2']))]
    else:
        players = latest_rosters[latest_rosters['position'] == position]

    for _, player in players.iterrows():
        pred = predict_2024_week1(player['player_name'])
        if pred:  # Check if prediction was successful
            if position == 'QB':
                predictions.append((player['player_name'], pred['passing_yards']))
            elif position == 'RB':
                predictions.append((player['player_name'], pred['rushing_yards']))
            elif position in ['WR', 'TE']:
                predictions.append((player['player_name'], pred['receiving_yards']))
    
    # Sort predictions in descending order
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

# Example usage
predict_2024_week1("Patrick Mahomes")
predict_2024_week1("Christian McCaffrey")
predict_2024_week1("Travis Kelce")
predict_2024_week1("Justin Jefferson")
