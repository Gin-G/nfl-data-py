import pandas as pd
import numpy as np
from nfl_data_py import import_weekly_data, import_players, import_rosters, import_schedules
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import mean_squared_error
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the data
logging.info("Loading data from CSV...")
df = pd.read_csv('data/nfl_dataset.csv')
logging.info(f"Data loaded. Shape: {df.shape}")
logging.debug(f"Columns: {df.columns.tolist()}")
logging.debug(f"Sample data:\n{df.head()}")

# Check for missing values
#missing_values = df.isnull().sum()
#logging.info(f"Missing values in dataset:\n{missing_values[missing_values > 0]}")

seasons = list(range(2018, 2025))
latest_rosters = import_rosters([2024])
schedule_2024 = import_schedules([2024])

# Create target variables
target_cols = [
    'passing_yards', 'attempts', 'completions', 'passing_tds', 'interceptions','sack_yards','sack_fumbles','sack_fumbles_lost',
    'rushing_yards', 'carries', 'rushing_tds', 'rushing_fumbles','rushing_fumbles_lost',
    'receiving_yards', 'targets', 'receptions', 'receiving_tds', 'receiving_fumbles', 'receiving_fumbles_lost', 
    'offensive_snap_pct', 'fanduel_fantasy_points'
]

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
logging.info(f"Feature matrix shape: {X.shape}")

# Fit preprocessor and transform data
X_transformed = preprocessor.fit_transform(X)
logging.info(f"Transformed feature matrix shape: {X_transformed.shape}")

# Get feature names after preprocessing
feature_names = (numerical_features + 
                 preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist())

# Build and train models
input_dim = X_transformed.shape[1]

def create_multi_output_model(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(256, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    
    # Separate outputs for each group of related stats
    passing_output = Dense(5, activation='relu', name='passing')(x)
    rushing_output = Dense(5, activation='relu', name='rushing')(x)
    receiving_output = Dense(5, activation='relu', name='receiving')(x)
    other_output = Dense(output_dim - 15, activation='relu', name='other')(x)
    
    # Create correlation layers
    snap_pct = Dense(1, activation='sigmoid', name='snap_pct')(x)
    
    # Adjust passing stats based on snap percentage
    adjusted_passing = Multiply()([passing_output, snap_pct])
    
    # Adjust rushing stats based on snap percentage and carries
    carries = Dense(1, activation='relu', name='carries')(x)
    adjusted_rushing = Multiply()([rushing_output, snap_pct, carries])
    
    # Adjust receiving stats based on snap percentage and targets
    targets = Dense(1, activation='relu', name='targets')(x)
    adjusted_receiving = Multiply()([receiving_output, snap_pct, targets])
    
    # Concatenate all outputs
    outputs = Concatenate()([adjusted_passing, adjusted_rushing, adjusted_receiving, other_output])
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=custom_loss)
    return model

# Update custom loss function to account for correlations
def custom_loss(y_true, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    # Extract relevant predictions
    snap_pct = y_pred[:, target_cols.index('offensive_snap_pct')]
    passing_yards = y_pred[:, target_cols.index('passing_yards')]
    rushing_yards = y_pred[:, target_cols.index('rushing_yards')]
    receiving_yards = y_pred[:, target_cols.index('receiving_yards')]
    carries = y_pred[:, target_cols.index('carries')]
    targets = y_pred[:, target_cols.index('targets')]
    
    # Penalize inconsistent predictions
    penalty = tf.keras.backend.relu(passing_yards - 400 * snap_pct) + \
              tf.keras.backend.relu(rushing_yards - 200 * carries * snap_pct) + \
              tf.keras.backend.relu(receiving_yards - 200 * targets * snap_pct)
    
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-7
    
    # Add L1 regularization to encourage non-zero predictions
    l1_reg = tf.keras.backend.sum(tf.keras.backend.abs(y_pred + epsilon))
    
    return mse + 0.1 * tf.keras.backend.mean(penalty) + 0.01 * l1_reg

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

# Create and train the multi-output model
logging.info("Starting model training...")
input_dim = X_train.shape[1]
output_dim = len(target_cols)
model = create_multi_output_model(input_dim, output_dim)

callbacks = [EpochProgressBar(total_epochs=100)]

history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2, 
    verbose=0,
    callbacks=callbacks
)

logging.info("Model training completed.")

# Plot and save learning curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Learning Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('predictions/learning_curves/learning_curves.png')
plt.close()

# Update prediction function with error handling and logging
def predict_2024_week(player_name, week):
    try:
        logging.info(f"Predicting for player: {player_name}, week: {week}")
        player_matches = latest_rosters[latest_rosters['player_name'].str.contains(player_name, case=False, na=False)]
        
        if player_matches.empty:
            logging.warning(f"No roster match found for player: {player_name}")
            return None
        
        player_info = player_matches.iloc[0]
        logging.debug(f"Player info: {player_info}")
        
        # Get the player's stats from the previous week
        player_stats = df[(df['player_id'] == player_info['player_id']) & (df['season'] == 2024) & (df['week'] == week - 1)]
        
        if player_stats.empty:
            logging.info(f"No stats for previous week. Using last available stats from 2023 for {player_name}")
            # If no stats for the previous week, use the last available stats from 2023
            player_stats = df[(df['player_id'] == player_info['player_id']) & (df['season'] == 2023)]
            if player_stats.empty:
                logging.warning(f"No historical stats found for player: {player_name}")
                return None
            last_stats = player_stats.sort_values('week').iloc[-1]
        else:
            last_stats = player_stats.iloc[0]
        
        logging.debug(f"Last stats for {player_name}: {last_stats}")
        
        input_data = pd.DataFrame({col: [last_stats[col]] for col in numerical_features})
        input_data['position_x'] = [player_info['position']]
        input_data['recent_team'] = [player_info['team']]
        
        logging.debug(f"Input data for prediction: {input_data}")
        
        input_transformed = preprocessor.transform(input_data)
        
        predictions = model.predict(input_transformed)[0]
        logging.debug(f"Raw predictions for {player_name}: {predictions}")
        
        # Apply a small threshold to avoid near-zero predictions
        threshold = 0.01
        adjusted_predictions = [max(pred, threshold) if i != target_cols.index('offensive_snap_pct') else pred for i, pred in enumerate(predictions)]
        
        result = {stat: round(pred, 2) for stat, pred in zip(target_cols, adjusted_predictions)}
        logging.info(f"Final predictions for {player_name}: {result}")
        return result
    
    except Exception as e:
        logging.error(f"Error predicting for {player_name}: {str(e)}")
        return None

def get_all_predictions(position, week):
    predictions = []
    
    # Filter players by position and active status
    active_players = latest_rosters[(latest_rosters['position'] == position) & (latest_rosters['status'] == 'ACT')]

    logging.info(f"Processing {len(active_players)} active {position} players for Week {week}")
    for _, player in tqdm(active_players.iterrows(), total=len(active_players), desc=f"Predicting {position}"):
        pred = predict_2024_week(player['player_name'], week)
        if pred:
            pred['player_name'] = player['player_name']
            predictions.append(pred)
    
    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    if not predictions_df.empty:
        # Sort the DataFrame by fantasy points
        predictions_df = predictions_df.sort_values(by='fanduel_fantasy_points', ascending=False)

    # Create directory if it doesn't exist
    os.makedirs('predictions', exist_ok=True)

    # Save to CSV
    csv_filename = f'predictions/{position}_predictions_week{week}_season2024.csv'
    predictions_df.to_csv(csv_filename, index=False)
    logging.info(f"Predictions saved to {csv_filename}")

    return predictions_df

# Main execution
positions = ['QB', 'RB', 'WR', 'TE']
week_to_predict = 3  # Set this to the week you want to predict

start_time = time.time()

for position in tqdm(positions, desc="Processing positions"):
    predictions_df = get_all_predictions(position, week_to_predict)
    logging.info(f"Predictions for {position}:\n{predictions_df.head()}")
    logging.info(f"Shape of predictions for {position}: {predictions_df.shape}")

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")