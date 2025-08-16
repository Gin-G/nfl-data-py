import pandas as pd
import numpy as np
from nfl_data_py import import_weekly_data, import_players, import_weekly_rosters, import_schedules
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate, Multiply, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
import logging
import warnings
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the data
logging.info("Loading data from CSV...")
df = pd.read_csv('data/complete_nfl_with_sportradar.csv')
logging.info(f"Data loaded. Shape: {df.shape}")
logging.debug(f"Columns: {df.columns.tolist()}")

# Create target variables
target_cols = [
    'passing_yards', 'attempts', 'completions', 'passing_tds', 'interceptions','sack_yards','sack_fumbles','sack_fumbles_lost',
    'rushing_yards', 'carries', 'rushing_tds', 'rushing_fumbles','rushing_fumbles_lost',
    'receiving_yards', 'targets', 'receptions', 'receiving_tds', 'receiving_fumbles', 'receiving_fumbles_lost', 
    'offensive_snap_pct', 'fanduel_fantasy_points'
]

for col in target_cols:
    df[f'next_week_{col}'] = df.groupby(['player_id', 'season'])[col].shift(-1)

# Improved data preprocessing
def preprocess_data(df):
    # Log-transform highly skewed variables
    skewed_vars = ['passing_yards', 'rushing_yards', 'receiving_yards']
    for var in skewed_vars:
        df[f'{var}_log'] = np.log1p(df[var])
    
    # Create interaction features with error handling
    df['pass_per_attempt'] = np.where(df['attempts'] > 0, df['passing_yards'] / df['attempts'], 0)
    df['rush_per_carry'] = np.where(df['carries'] > 0, df['rushing_yards'] / df['carries'], 0)
    df['rec_per_target'] = np.where(df['targets'] > 0, df['receiving_yards'] / df['targets'], 0)
    
    # Replace infinity and NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    return df

df = preprocess_data(df)

# Identify features
categorical_features = ['position_x', 'recent_team']
numerical_features = target_cols + [f'{var}_log' for var in ['passing_yards', 'rushing_yards', 'receiving_yards']] + ['pass_per_attempt', 'rush_per_carry', 'rec_per_target']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_features)
    ])

# Prepare features
X = df[categorical_features + numerical_features]
logging.info(f"Feature matrix shape: {X.shape}")

# Fit preprocessor and transform data
X_transformed = preprocessor.fit_transform(X)
logging.info(f"Transformed feature matrix shape: {X_transformed.shape}")

# Get feature names after preprocessing
feature_names = (numerical_features + 
                 preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist())

# Improved model architecture
def create_improved_model(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(512, activation='LeakyReLU')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='LeakyReLU')(x)
    x = Dropout(0.3)(x)
    
    # Separate branches for different stat types
    passing_branch = Dense(64, activation='LeakyReLU')(x)
    rushing_branch = Dense(64, activation='LeakyReLU')(x)
    receiving_branch = Dense(64, activation='LeakyReLU')(x)
    
    # Output layers
    passing_output = Dense(8, name='passing')(passing_branch)  # 8 passing stats
    rushing_output = Dense(5, name='rushing')(rushing_branch)  # 5 rushing stats
    receiving_output = Dense(6, name='receiving')(receiving_branch)  # 6 receiving stats
    other_output = Dense(output_dim - 19, name='other')(x)  # remaining stats
    
    # Combine outputs
    outputs = Concatenate()([passing_output, rushing_output, receiving_output, other_output])
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Improved loss function
def improved_custom_loss(y_true, y_pred):
    # Separate losses for different stat types
    mse_passing = mean_squared_error(y_true[:, :8], y_pred[:, :8])
    mse_rushing = mean_squared_error(y_true[:, 8:13], y_pred[:, 8:13])
    mse_receiving = mean_squared_error(y_true[:, 13:19], y_pred[:, 13:19])
    mse_other = mean_squared_error(y_true[:, 19:], y_pred[:, 19:])
    
    # Combine losses with appropriate weighting
    total_loss = mse_passing + mse_rushing + mse_receiving + 0.5 * mse_other
    
    return total_loss

# Add a custom callback to show progress bar for epochs
class EpochProgressBar(TensorBoard):
    def __init__(self, total_epochs, **kwargs):
        super().__init__(**kwargs)
        self.pbar = tqdm(total=total_epochs, desc="Training Epochs", leave=False)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        self.pbar.close()

# Prepare target data
y = df[[f'next_week_{stat}' for stat in target_cols]].values
mask = ~np.isnan(y).any(axis=1)
X_train = X_transformed[mask]
y_train = y[mask]

logging.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

# Create and train the improved model
logging.info("Starting model training...")
input_dim = X_train.shape[1]
output_dim = len(target_cols)
model = create_improved_model(input_dim, output_dim)

# Learning rate scheduler
lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)

# Compile the model
model.compile(optimizer=Adam(lr=1e-3), loss=improved_custom_loss)

# Train the model
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=64, 
    validation_split=0.2, 
    verbose=0,
    callbacks=[EpochProgressBar(total_epochs=100), lr_scheduler]
)

logging.info("Model training completed.")

# Save the model and preprocessor
os.makedirs('models', exist_ok=True)
model.save('models/nfl_projection_model.keras')
joblib.dump(preprocessor, 'models/preprocessor.pkl')
logging.info("Model and preprocessor saved.")

# Plot and save learning curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Learning Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
os.makedirs('predictions/learning_curves', exist_ok=True)
plt.savefig('predictions/learning_curves/learning_curves.png')
plt.close()

# Updated prediction function - now flexible for any season/week
def predict_player_week(player_name, season, week):
    """
    Predict stats for any player in any season/week
    """
    try:
        # Get rosters for the specified season
        rosters = import_weekly_rosters([season])
        player_matches = rosters[rosters['player_name'].str.contains(player_name, case=False, na=False)]
        
        if player_matches.empty:
            logging.warning(f"No roster match found for player: {player_name}")
            return None
        
        player_info = player_matches.iloc[0]
        
        # Get the player's stats from the previous week in the same season
        player_stats = df[(df['player_id'] == player_info['player_id']) & (df['season'] == season) & (df['week'] == week - 1)]
        
        if player_stats.empty:
            # If no stats for the previous week, use the last available stats from the previous season
            logging.info(f"No stats for week {week-1}. Using last available stats from {season-1} for {player_name}")
            player_stats = df[(df['player_id'] == player_info['player_id']) & (df['season'] == season - 1)]
            if player_stats.empty:
                logging.warning(f"No historical stats found for player: {player_name}")
                return None
            last_stats = player_stats.sort_values('week').iloc[-1]
        else:
            last_stats = player_stats.iloc[0]
        
        # Prepare input data with averaged features
        input_data = pd.DataFrame({col: [avg_stats[col]] for col in numerical_features})
        input_data['position_x'] = [player_info['position']]
        input_data['recent_team'] = [player_info['team']]
        
        # Add depth_tier from recent data if available
        if len(player_stats_list) > 0 and 'depth_tier' in player_stats_list[0].index:
            input_data['depth_tier'] = [player_stats_list[0]['depth_tier']]
        else:
            # Default to 'deep_bench' if no depth chart data available
            input_data['depth_tier'] = ['deep_bench']
        
        # Apply preprocessing to the input data
        input_data = preprocess_data(input_data)
        
        input_transformed = preprocessor.transform(input_data)
        
        predictions = model.predict(input_transformed)[0]
        
        # Apply a small threshold to avoid near-zero predictions
        threshold = 0.01
        adjusted_predictions = [max(pred, threshold) if i != target_cols.index('offensive_snap_pct') else pred for i, pred in enumerate(predictions)]
        
        result = {stat: round(pred, 2) for stat, pred in zip(target_cols, adjusted_predictions)}
        result['player_name'] = player_name
        result['season'] = season
        result['week'] = week
        result['position'] = player_info['position']
        result['team'] = player_info['team']
        
        return result
    
    except Exception as e:
        logging.error(f"Error predicting for {player_name}: {str(e)}")
        return None

def get_all_predictions_for_week(position, season, week):
    """
    Generate predictions for all active players of a position for any season/week
    """
    predictions = []
    
    # Get rosters for the specified season
    try:
        rosters = import_weekly_rosters([season])
    except Exception as e:
        logging.error(f"Could not fetch rosters for season {season}: {str(e)}")
        return pd.DataFrame()
    
    # Filter players by position and active status
    active_players = rosters[(rosters['position'] == position) & (rosters['status'] == 'ACT')]

    for _, player in tqdm(active_players.iterrows(), total=len(active_players), desc=f"Predicting {position} for {season} Week {week}"):
        pred = predict_player_week(player['player_name'], season, week)
        if pred:
            predictions.append(pred)
    
    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    if not predictions_df.empty:
        # Filter to likely starters based on season-long snap percentage averages
        if position == 'QB':
            # For QBs, calculate average snap percentage across recent seasons
            starter_candidates = []
            
            for _, row in predictions_df.iterrows():
                player_name = row['player_name']
                
                # Get all games for this player in recent seasons
                player_games = df[
                    (df['player_name'] == player_name) & 
                    (df['season'] >= season - 2) &  # Look at last 2 seasons
                    (df['position_x'] == position) &
                    (df['offensive_snap_pct'] > 0)  # Only count games where they actually played
                ]
                
                # FIRST: Check if player is confirmed starter in depth chart
                is_depth_chart_starter = False
                if len(player_games) > 0:
                    # Check if player has depth=1 (starter) in recent games
                    starter_games = player_games[player_games.get('depth', 99) == 1]
                    if len(starter_games) >= 3:  # If they were depth=1 in 3+ recent games
                        is_depth_chart_starter = True
                        logging.debug(f"QB Depth Chart Starter: {player_name} - depth=1 in {len(starter_games)} games")
                
                if is_depth_chart_starter:
                    starter_candidates.append(row)
                    continue  # Skip snap % analysis if they're confirmed starter
                
                # SECOND: Snap percentage analysis for players without clear depth chart status
                if len(player_games) >= 3:  # Need at least 3 games of data
                    # Strategy 1: Overall season average (for established starters)
                    avg_snap_pct = player_games['offensive_snap_pct'].mean()
                    games_played = len(player_games)
                    
                    # Strategy 2: Recent games average (for mid-season starters/rookies)
                    # Look at last 8 games to see if they became a starter
                    recent_games = player_games.nlargest(8, ['season', 'week'])
                    recent_avg_snap_pct = recent_games['offensive_snap_pct'].mean()
                    
                    # Strategy 3: Peak starter period (for players who had a starting stretch)
                    # Find the best 6-game stretch for this player
                    if len(player_games) >= 6:
                        # Sort by season and week
                        sorted_games = player_games.sort_values(['season', 'week'])
                        best_stretch_avg = 0
                        
                        # Check every 6-game window
                        for i in range(len(sorted_games) - 5):
                            stretch = sorted_games.iloc[i:i+6]
                            stretch_avg = stretch['offensive_snap_pct'].mean()
                            if stretch_avg > best_stretch_avg:
                                best_stretch_avg = stretch_avg
                    else:
                        best_stretch_avg = avg_snap_pct
                    
                    # QB qualifies as starter if ANY of these conditions are met:
                    conditions = [
                        # Established starter: 70%+ average over 8+ games
                        avg_snap_pct > 70 and games_played >= 8,
                        
                        # Recent starter: 80%+ in recent games (rookies who took over)
                        recent_avg_snap_pct > 80 and len(recent_games) >= 4,
                        
                        # Peak starter: Had a 6-game stretch averaging 75%+ (mid-season takeover)
                        best_stretch_avg > 75 and games_played >= 6,
                        
                        # High snap rate starter: 85%+ average even with fewer games (clear starter when healthy)
                        avg_snap_pct > 85 and games_played >= 4
                    ]
                    
                    if any(conditions):
                        starter_candidates.append(row)
                        reason = ""
                        if conditions[0]:
                            reason = f"Established starter: {avg_snap_pct:.1f}% avg over {games_played} games"
                        elif conditions[1]:
                            reason = f"Recent starter: {recent_avg_snap_pct:.1f}% in recent {len(recent_games)} games"
                        elif conditions[2]:
                            reason = f"Peak starter: {best_stretch_avg:.1f}% best stretch over {games_played} games"
                        elif conditions[3]:
                            reason = f"High-snap starter: {avg_snap_pct:.1f}% avg over {games_played} games"
                        
                        logging.debug(f"QB Starter: {player_name} - {reason}")
                    else:
                        logging.debug(f"QB Backup: {player_name} - {avg_snap_pct:.1f}% avg, {recent_avg_snap_pct:.1f}% recent, {best_stretch_avg:.1f}% peak over {games_played} games")
            
            if starter_candidates:
                predictions_df = pd.DataFrame(starter_candidates)
                logging.info(f"Filtered to {len(starter_candidates)} likely starting QBs")
            else:
                # Fallback: keep top 32 by projected fantasy points
                predictions_df = predictions_df.head(32)
                logging.warning("No QBs met starter criteria, keeping top 32 by projection")
        
        elif position in ['RB', 'WR', 'TE']:
            # For skill positions, use lower thresholds since multiple players contribute
            starter_candidates = []
            
            for _, row in predictions_df.iterrows():
                player_name = row['player_name']
                
                # Get all games for this player in recent seasons
                player_games = df[
                    (df['player_name'] == player_name) & 
                    (df['season'] >= season - 2) &
                    (df['position_x'] == position) &
                    (df['offensive_snap_pct'] > 0)
                ]
                
                if len(player_games) >= 3:  # Need at least 3 games of data
                    avg_snap_pct = player_games['offensive_snap_pct'].mean()
                    games_played = len(player_games)
                    
                    # Position-specific thresholds
                    if position == 'RB':
                        # RB1s typically get 40%+ snaps, RB2s get 20%+
                        threshold = 20
                        min_games = 6
                    elif position == 'WR':
                        # WR1/WR2/WR3 can all be fantasy relevant with 25%+ snaps
                        threshold = 25
                        min_games = 6
                    else:  # TE
                        # TE1s typically get 40%+ snaps
                        threshold = 30
                        min_games = 6
                    
                    if avg_snap_pct > threshold and games_played >= min_games:
                        starter_candidates.append(row)
                        logging.debug(f"{position} Starter: {player_name} - {avg_snap_pct:.1f}% avg snaps over {games_played} games")
                    else:
                        logging.debug(f"{position} Backup: {player_name} - {avg_snap_pct:.1f}% avg snaps over {games_played} games")
            
            if starter_candidates:
                predictions_df = pd.DataFrame(starter_candidates)
                logging.info(f"Filtered to {len(starter_candidates)} likely starting {position}s")
            else:
                # Fallback: keep top 60% of players
                keep_count = max(len(predictions_df) * 6 // 10, 10)  # Keep at least 10
                predictions_df = predictions_df.head(keep_count)
                logging.warning(f"No {position}s met starter criteria, keeping top {keep_count} by projection")
        
        # Sort the DataFrame by fantasy points
        predictions_df = predictions_df.sort_values(by='fanduel_fantasy_points', ascending=False)
        
        # Remove duplicate rows based on all columns
        predictions_df = predictions_df.drop_duplicates()

    # Create directory if it doesn't exist
    os.makedirs('predictions', exist_ok=True)

    # Save to CSV
    csv_filename = f'predictions/{position}_predictions_week{week}_season{season}.csv'
    predictions_df.to_csv(csv_filename, index=False)
    logging.info(f"Predictions saved to {csv_filename}")

    return predictions_df

# Example usage - you can change these values
season_to_predict = 2025  # Change this to any season
week_to_predict = 1       # Change this to any week
positions = ['QB', 'RB', 'WR', 'TE']

# Test single player prediction
print("\n=== Single Player Test ===")
test_prediction = predict_player_week('Josh Allen', season_to_predict, week_to_predict)
if test_prediction:
    print(f"Josh Allen Week {week_to_predict} {season_to_predict} prediction:")
    print(f"Fantasy Points: {test_prediction['fanduel_fantasy_points']}")
    print(f"Passing Yards: {test_prediction['passing_yards']}")
    print(f"Passing TDs: {test_prediction['passing_tds']}")

# Generate predictions for all positions
print(f"\n=== Generating All Predictions for {season_to_predict} Week {week_to_predict} ===")
start_time = time.time()

for position in tqdm(positions, desc="Processing positions"):
    predictions_df = get_all_predictions_for_week(position, season_to_predict, week_to_predict)
    if not predictions_df.empty:
        print(f"\n{position} Top 5 Projected Fantasy Points:")
        top_5 = predictions_df.head(5)[['player_name', 'team', 'fanduel_fantasy_points']]
        for idx, row in top_5.iterrows():
            print(f"  {row['player_name']} ({row['team']}): {row['fanduel_fantasy_points']:.1f}")

end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
print(f"\nPredictions saved in 'predictions/' directory for {season_to_predict} Week {week_to_predict}")