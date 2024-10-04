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

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the data
logging.info("Loading data from CSV...")
df = pd.read_csv('data/nfl_dataset.csv')
logging.info(f"Data loaded. Shape: {df.shape}")
logging.debug(f"Columns: {df.columns.tolist()}")

seasons = list(range(2018, 2025))
latest_rosters = import_weekly_rosters([2024])
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
    passing_output = Dense(5, name='passing')(passing_branch)
    rushing_output = Dense(5, name='rushing')(rushing_branch)
    receiving_output = Dense(5, name='receiving')(receiving_branch)
    other_output = Dense(output_dim - 15, name='other')(x)
    
    # Combine outputs
    outputs = Concatenate()([passing_output, rushing_output, receiving_output, other_output])
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Improved loss function
def improved_custom_loss(y_true, y_pred):
    # Separate losses for different stat types
    mse_passing = mean_squared_error(y_true[:, :5], y_pred[:, :5])
    mse_rushing = mean_squared_error(y_true[:, 5:10], y_pred[:, 5:10])
    mse_receiving = mean_squared_error(y_true[:, 10:15], y_pred[:, 10:15])
    mse_other = mean_squared_error(y_true[:, 15:], y_pred[:, 15:])
    
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
    epochs=200, 
    batch_size=64, 
    validation_split=0.2, 
    verbose=0,
    callbacks=[EpochProgressBar(total_epochs=200), lr_scheduler]
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
        player_matches = latest_rosters[latest_rosters['player_name'].str.contains(player_name, case=False, na=False)]
        
        if player_matches.empty:
            logging.warning(f"No roster match found for player: {player_name}")
            return None
        
        player_info = player_matches.iloc[0]
        
        # Get the player's stats from the previous week
        player_stats = df[(df['player_id'] == player_info['player_id']) & (df['season'] == 2024) & (df['week'] == week - 1)]
        
        if player_stats.empty:
            logging.info(f"No stats for previous week. Using last available stats from 2023 for {player_name}")
            # If no stats for the previous week, use the last available stats from 2023
            player_stats = df[(df['player_id'] == player_info['player_id']) & (df['season'] == max(seasons))]
            if player_stats.empty:
                logging.warning(f"No historical stats found for player: {player_name}")
                return None
            last_stats = player_stats.sort_values('week').iloc[-1]
        else:
            last_stats = player_stats.iloc[0]
        
        # Prepare input data with new features
        input_data = pd.DataFrame({col: [last_stats[col]] for col in numerical_features})
        input_data['position_x'] = [player_info['position']]
        input_data['recent_team'] = [player_info['team']]
        
        # Apply preprocessing to the input data
        input_data = preprocess_data(input_data)
        
        input_transformed = preprocessor.transform(input_data)
        
        predictions = model.predict(input_transformed)[0]
        
        # Apply a small threshold to avoid near-zero predictions
        threshold = 0.01
        adjusted_predictions = [max(pred, threshold) if i != target_cols.index('offensive_snap_pct') else pred for i, pred in enumerate(predictions)]
        
        result = {stat: round(pred, 2) for stat, pred in zip(target_cols, adjusted_predictions)}
        return result
    
    except Exception as e:
        logging.error(f"Error predicting for {player_name}: {str(e)}")
        return None

def get_all_predictions(position, week):
    predictions = []
    
    # Filter players by position and active status
    active_players = latest_rosters[(latest_rosters['position'] == position) & (latest_rosters['status'] == 'ACT')]

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
        
        # Remove duplicate rows based on all columns
        predictions_df = predictions_df.drop_duplicates()

    # Create directory if it doesn't exist
    os.makedirs('predictions', exist_ok=True)

    # Save to CSV
    csv_filename = f'predictions/{position}_predictions_week{week}_season2024.csv'
    predictions_df.to_csv(csv_filename, index=False)
    logging.info(f"Predictions saved to {csv_filename}")

    return predictions_df

#print(predict_2024_week('Marvin Harrison',3))

# Main execution
positions = ['QB', 'RB', 'WR', 'TE']
week_to_predict = 5  # Set this to the week you want to predict

start_time = time.time()

for position in tqdm(positions, desc="Processing positions"):
    predictions_df = get_all_predictions(position, week_to_predict)

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")