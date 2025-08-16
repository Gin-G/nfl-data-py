import pandas as pd
import numpy as np
from nfl_data_py import import_weekly_data, import_players, import_weekly_rosters, import_schedules
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the enhanced data
logging.info("Loading enhanced data from CSV...")
df = pd.read_csv('data/nfl_dataset.csv')
logging.info(f"Data loaded. Shape: {df.shape}")

year = 2025
latest_rosters = import_weekly_rosters([year])
schedule = import_schedules([year])

# Data quality check and cleaning
def analyze_and_clean_data(df):
    """Analyze data quality and clean for better model training"""
    
    print(f"=== DATA QUALITY ANALYSIS ===")
    print(f"Total records: {len(df)}")
    
    # Remove season average rows for training (keep only regular games)
    if 'week' in df.columns:
        regular_games = df[df['week'] != 'AVG'].copy()
        print(f"Regular season games: {len(regular_games)}")
        
        # Convert week to numeric for proper sorting
        regular_games['week'] = pd.to_numeric(regular_games['week'], errors='coerce')
        regular_games = regular_games.dropna(subset=['week'])
    else:
        regular_games = df.copy()
    
    # Remove players with insufficient data (less than 3 games)
    player_game_counts = regular_games.groupby('player_id').size()
    active_players = player_game_counts[player_game_counts >= 3].index
    filtered_games = regular_games[regular_games['player_id'].isin(active_players)]
    print(f"After filtering low-activity players: {len(filtered_games)}")
    
    # Remove extreme outliers in fantasy points (likely data errors)
    q99 = filtered_games['fanduel_fantasy_points'].quantile(0.99)
    q1 = filtered_games['fanduel_fantasy_points'].quantile(0.01)
    clean_games = filtered_games[
        (filtered_games['fanduel_fantasy_points'] >= q1) & 
        (filtered_games['fanduel_fantasy_points'] <= q99)
    ]
    print(f"After removing outliers: {len(clean_games)}")
    
    # Focus on recent seasons for better relevance (2020-2024)
    if 'season' in clean_games.columns:
        recent_games = clean_games[clean_games['season'] >= 2020]
        print(f"Recent seasons (2020+): {len(recent_games)}")
    else:
        recent_games = clean_games
    
    return recent_games

# Clean the data
df_clean = analyze_and_clean_data(df)

# Enhanced target columns - focus on the most important stats
target_cols = [
    'passing_yards', 'passing_tds', 'interceptions',
    'rushing_yards', 'rushing_tds', 
    'receiving_yards', 'receptions', 'receiving_tds',
    'fanduel_fantasy_points'  # This is our primary target
]

# Filter to only columns that exist
target_cols = [col for col in target_cols if col in df_clean.columns]
print(f"Target columns: {target_cols}")

# Create next week targets with better handling
def create_targets_with_validation(df, target_cols):
    """Create next week targets with validation"""
    df_with_targets = df.copy()
    
    # Sort by player, season, week for proper shifting
    df_with_targets = df_with_targets.sort_values(['player_id', 'season', 'week'])
    
    # Create targets
    for col in target_cols:
        if col in df_with_targets.columns:
            df_with_targets[f'next_week_{col}'] = df_with_targets.groupby(['player_id', 'season'])[col].shift(-1)
    
    # Only keep records that have valid targets (not last game of season)
    target_columns = [f'next_week_{col}' for col in target_cols]
    valid_mask = ~df_with_targets[target_columns].isna().all(axis=1)
    df_with_targets = df_with_targets[valid_mask]
    
    print(f"Records with valid targets: {len(df_with_targets)}")
    return df_with_targets

df_with_targets = create_targets_with_validation(df_clean, target_cols)

# Smart feature engineering focused on predictive power
def create_predictive_features(df):
    """Create features that actually predict performance"""
    
    df_features = df.copy()
    
    # Basic efficiency metrics
    if 'attempts' in df_features.columns and 'passing_yards' in df_features.columns:
        df_features['yards_per_attempt'] = np.where(
            df_features['attempts'] > 0, 
            df_features['passing_yards'] / df_features['attempts'], 0
        )
    
    if 'carries' in df_features.columns and 'rushing_yards' in df_features.columns:
        df_features['yards_per_carry'] = np.where(
            df_features['carries'] > 0, 
            df_features['rushing_yards'] / df_features['carries'], 0
        )
    
    if 'targets' in df_features.columns and 'receiving_yards' in df_features.columns:
        df_features['yards_per_target'] = np.where(
            df_features['targets'] > 0, 
            df_features['receiving_yards'] / df_features['targets'], 0
        )
    
    # Position-specific usage metrics
    if 'position' in df_features.columns:
        # QB: Attempts per game indicates passing volume
        if 'attempts' in df_features.columns:
            df_features['qb_passing_volume'] = np.where(
                df_features['position'] == 'QB', df_features['attempts'], 0
            )
        
        # RB: Carries + targets indicates total touches
        if 'carries' in df_features.columns and 'targets' in df_features.columns:
            df_features['rb_total_touches'] = np.where(
                df_features['position'] == 'RB', 
                df_features['carries'].fillna(0) + df_features['targets'].fillna(0), 0
            )
        
        # WR/TE: Target share is crucial
        if 'targets' in df_features.columns:
            df_features['wr_te_targets'] = np.where(
                df_features['position'].isin(['WR', 'TE']), df_features['targets'], 0
            )
    
    # Recent performance trends (if avg_fppg exists)
    if 'avg_fppg' in df_features.columns and 'fanduel_fantasy_points' in df_features.columns:
        df_features['performance_vs_average'] = (
            df_features['fanduel_fantasy_points'] - df_features['avg_fppg']
        )
        
        # Consistency indicator
        df_features['is_consistent_performer'] = (
            (df_features['performance_vs_average'].abs() < 5).astype(int)
        )
    
    # Simple rolling averages (last 3 games)
    for stat in ['fanduel_fantasy_points', 'passing_yards', 'rushing_yards', 'receiving_yards']:
        if stat in df_features.columns:
            df_features[f'{stat}_recent_avg'] = (
                df_features.groupby('player_id')[stat]
                .rolling(3, min_periods=1)
                .mean()
                .reset_index(drop=True)
            )
    
    # Fill NaN values
    df_features = df_features.fillna(0)
    
    return df_features

df_final = create_predictive_features(df_with_targets)

# Feature selection - only use the most predictive features
numerical_features = [
    # Core stats that exist in our data
    'passing_yards', 'passing_tds', 'interceptions',
    'rushing_yards', 'rushing_tds',
    'receiving_yards', 'receptions', 'receiving_tds',
    'offensive_snap_pct',
    
    # Efficiency metrics
    'yards_per_attempt', 'yards_per_carry', 'yards_per_target',
    
    # Position-specific features  
    'qb_passing_volume', 'rb_total_touches', 'wr_te_targets',
    
    # Performance trends
    'avg_fppg', 'performance_vs_average', 'is_consistent_performer',
    
    # Recent averages
    'fanduel_fantasy_points_recent_avg', 'passing_yards_recent_avg',
    'rushing_yards_recent_avg', 'receiving_yards_recent_avg'
]

# Filter to only features that exist
numerical_features = [col for col in numerical_features if col in df_final.columns]
print(f"Numerical features ({len(numerical_features)}): {numerical_features}")

# Categorical features
categorical_features = ['position']
if 'recent_team' in df_final.columns:
    categorical_features.append('recent_team')

print(f"Categorical features: {categorical_features}")

# Clean categorical data
def clean_categorical_features(df, cat_features):
    """Clean categorical features for encoding"""
    df_clean = df.copy()
    for col in cat_features:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).replace(['nan', 'None'], 'Unknown')
    return df_clean

df_final_clean = clean_categorical_features(df_final, categorical_features)

# Prepare training data with proper validation split
all_features = categorical_features + numerical_features
X = df_final_clean[all_features]
target_columns = [f'next_week_{col}' for col in target_cols]
y = df_final_clean[target_columns].values

# Remove any remaining NaN values
valid_mask = ~np.isnan(y).any(axis=1) & ~np.isinf(y).any(axis=1)
X_clean = X[valid_mask]
y_clean = y[valid_mask]

print(f"Final training data: X={X_clean.shape}, y={y_clean.shape}")

# Create preprocessor
preprocessor = ColumnTransformer([
    ('num', RobustScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
])

# Fit and transform
X_transformed = preprocessor.fit_transform(X_clean)
print(f"Transformed features: {X_transformed.shape}")

# Proper train/validation split (not just random - respect temporal order)
# Use earlier games for training, later games for validation
df_with_indices = df_final_clean[valid_mask].copy()
df_with_indices['original_index'] = range(len(df_with_indices))

# Sort by season and week to maintain temporal order
df_sorted = df_with_indices.sort_values(['season', 'week'])

# Use first 80% of temporal data for training
split_idx = int(0.8 * len(df_sorted))
train_indices = df_sorted.iloc[:split_idx]['original_index'].values
val_indices = df_sorted.iloc[split_idx:]['original_index'].values

X_train = X_transformed[train_indices]
X_val = X_transformed[val_indices]
y_train = y_clean[train_indices]
y_val = y_clean[val_indices]

print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")

# Simpler, more focused model architecture
def create_focused_model(input_dim, output_dim):
    """Create a focused model that avoids overfitting"""
    
    inputs = Input(shape=(input_dim,))
    
    # Simpler architecture with proper regularization
    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer - no activation for regression
    outputs = Dense(output_dim, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create and compile model
input_dim = X_train.shape[1]
output_dim = len(target_cols)
model = create_focused_model(input_dim, output_dim)

# Use simpler loss function and better optimizer settings
model.compile(
    optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
    loss='mse',
    metrics=['mae']
)

print("Model architecture:")
model.summary()

# Better callbacks
callbacks = [
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=10, 
        min_lr=1e-6, 
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
]

# Train with validation data
print("Starting model training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

print("Model training completed.")

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

os.makedirs('predictions/learning_curves', exist_ok=True)
plt.tight_layout()
plt.savefig('predictions/learning_curves/improved_model_curves.png', dpi=300)
plt.close()

# Improved prediction function with better data handling
def improved_predict_week(player_name, week):
    """Improved prediction with better feature engineering"""
    try:
        # Find player in roster
        player_matches = latest_rosters[
            latest_rosters['player_name'].str.contains(player_name, case=False, na=False)
        ]
        
        if player_matches.empty:
            print(f"No roster match found for: {player_name}")
            return None
        
        player_info = player_matches.iloc[0]
        
        # Get player's recent performance (last 5 games from 2024)
        player_data = df_final_clean[
            (df_final_clean['player_id'] == player_info['player_id']) & 
            (df_final_clean['season'] >= 2023)  # Recent data only
        ]
        
        if player_data.empty:
            print(f"No recent data for: {player_name}")
            return None
        
        # Use the most recent game or season average
        recent_stats = player_data.sort_values(['season', 'week']).iloc[-1]
        
        # Prepare input features
        input_data = pd.DataFrame()
        
        # Add numerical features
        for col in numerical_features:
            if col in recent_stats.index:
                input_data[col] = [recent_stats[col]]
            else:
                input_data[col] = [0]
        
        # Add categorical features
        input_data['position'] = [str(player_info['position'])]
        if 'recent_team' in categorical_features:
            input_data['recent_team'] = [str(player_info.get('team', 'Unknown'))]
        
        # Transform input
        input_transformed = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_transformed, verbose=0)[0]
        
        # Format results
        result = {}
        for i, stat in enumerate(target_cols):
            # Apply reasonable bounds
            if stat == 'fanduel_fantasy_points':
                result[stat] = max(0, min(60, prediction[i]))  # Fantasy points 0-60
            elif 'yards' in stat:
                result[stat] = max(0, min(500, prediction[i]))  # Yards 0-500
            elif 'tds' in stat or stat == 'receptions':
                result[stat] = max(0, min(6, prediction[i]))   # TDs/receptions 0-6
            else:
                result[stat] = max(0, prediction[i])
            
            result[stat] = round(result[stat], 2)
        
        # Add context
        if 'avg_fppg' in recent_stats.index:
            result['vs_season_avg'] = round(
                result['fanduel_fantasy_points'] - recent_stats['avg_fppg'], 2
            )
            result['season_avg'] = round(recent_stats['avg_fppg'], 2)
        
        return result
        
    except Exception as e:
        print(f"Error predicting for {player_name}: {e}")
        return None

# Generate improved predictions
def get_improved_predictions(position, week):
    """Generate improved predictions for a position"""
    predictions = []
    
    # Get active players for this position
    active_players = latest_rosters[
        (latest_rosters['position'] == position) & 
        (latest_rosters['status'] == 'ACT')
    ]
    
    print(f"Generating predictions for {len(active_players)} {position}s...")
    
    for _, player in tqdm(active_players.iterrows(), 
                         total=len(active_players), 
                         desc=f"Predicting {position}"):
        pred = improved_predict_week(player['player_name'], week)
        if pred:
            pred['player_name'] = player['player_name']
            pred['position'] = position
            predictions.append(pred)
    
    # Convert to DataFrame and sort
    if predictions:
        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df.sort_values('fanduel_fantasy_points', ascending=False)
        predictions_df['rank'] = range(1, len(predictions_df) + 1)
        
        # Save predictions
        os.makedirs('predictions', exist_ok=True)
        filename = f'predictions/{position}_improved_predictions_week{week}_season{year}.csv'
        predictions_df.to_csv(filename, index=False)
        print(f"Saved {len(predictions_df)} {position} predictions to {filename}")
        
        return predictions_df
    else:
        print(f"No valid predictions generated for {position}")
        return pd.DataFrame()

# Generate predictions for all positions
positions = ['QB', 'RB', 'WR', 'TE']
week_to_predict = 1

print("\n" + "="*50)
print("GENERATING IMPROVED PREDICTIONS")
print("="*50)

start_time = time.time()

for position in positions:
    print(f"\n--- {position} PREDICTIONS ---")
    predictions_df = get_improved_predictions(position, week_to_predict)
    
    if not predictions_df.empty:
        print(f"Top 5 {position} predictions:")
        top_5 = predictions_df.head(5)
        for _, player in top_5.iterrows():
            name = player['player_name']
            points = player['fanduel_fantasy_points']
            vs_avg = player.get('vs_season_avg', 'N/A')
            season_avg = player.get('season_avg', 'N/A')
            print(f"  {player['rank']:2d}. {name:20s}: {points:5.1f} pts "
                  f"(vs avg: {vs_avg:+5.1f}, season: {season_avg})")

end_time = time.time()
print(f"\nTotal time: {end_time - start_time:.1f} seconds")
print("\nImproved predictions completed! Check the CSV files in predictions/ folder.")