import pandas as pd
import numpy as np
from nfl_data_py import import_weekly_data, import_players, import_weekly_rosters, import_schedules, import_draft_picks, import_depth_charts
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

print("\n🧹 CLEANING ROSTER DATA TO PREVENT DUPLICATES...")
print(f"Original roster data: {len(latest_rosters)} rows")

# Check for duplicates in roster data
duplicate_players = latest_rosters['player_name'].duplicated().sum()
if duplicate_players > 0:
    print(f"Found {duplicate_players} duplicate player entries in roster data")
    
    # Show examples of duplicates
    dup_names = latest_rosters[latest_rosters['player_name'].duplicated(keep=False)]['player_name'].value_counts().head(5)
    print("Top duplicate players:")
    for name, count in dup_names.items():
        print(f"  {name}: {count} entries")
    
    # Priority order for keeping duplicates: ACT > INACT > other statuses
    status_priority = {'ACT': 1, 'INACT': 2}
    latest_rosters['status_priority'] = latest_rosters['status'].map(status_priority).fillna(3)
    
    # Sort by priority and keep best entry for each player
    latest_rosters = latest_rosters.sort_values(['player_name', 'status_priority', 'week'], 
                                                ascending=[True, True, False])
    latest_rosters = latest_rosters.drop_duplicates(subset=['player_name'], keep='first')
    
    print(f"After deduplication: {len(latest_rosters)} rows")
    print(f"Removed {duplicate_players} duplicate roster entries")
else:
    print("No duplicate players found in roster data")

# Load current depth charts for real-time role detection
print("Loading current depth charts...")
current_depth_charts = import_depth_charts([year])
print(f"Loaded {len(current_depth_charts)} depth chart entries")

# Data quality check and cleaning
def analyze_and_clean_data(df):
    """Analyze data quality and clean for better model training"""
    
    print(f"=== DATA QUALITY ANALYSIS ===")
    print(f"Total records: {len(df)}")
    print(f"Available columns: {len(df.columns)}")
    
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

def validate_predictions_no_duplicates(predictions_df, position):
    """Validate that predictions have no duplicates"""
    if predictions_df.empty:
        return predictions_df
    
    print(f"\n🔍 Validating {position} predictions...")
    
    # Check for exact duplicates (all columns)
    exact_dupes = predictions_df.duplicated().sum()
    if exact_dupes > 0:
        print(f"⚠️  Found {exact_dupes} exact duplicate rows")
        predictions_df = predictions_df.drop_duplicates()
    
    # Check for player name duplicates
    name_dupes = predictions_df['player_name'].duplicated().sum()
    if name_dupes > 0:
        print(f"⚠️  Found {name_dupes} duplicate player names")
        duplicated_players = predictions_df[predictions_df['player_name'].duplicated(keep=False)]
        print("Duplicate players found:")
        for name in duplicated_players['player_name'].unique():
            player_rows = predictions_df[predictions_df['player_name'] == name]
            print(f"  {name}: {len(player_rows)} copies")
            print(f"    Ranks: {player_rows['rank'].tolist()}")
            print(f"    Points: {player_rows['fanduel_fantasy_points'].tolist()}")
        
        # Keep only first occurrence
        predictions_df = predictions_df.drop_duplicates(subset=['player_name'], keep='first')
        print(f"✅ After cleanup: {len(predictions_df)} unique players")
    
    # Validate rank sequence
    expected_ranks = list(range(1, len(predictions_df) + 1))
    actual_ranks = sorted(predictions_df['rank'].tolist())
    
    if actual_ranks != expected_ranks:
        print(f"⚠️  Rank sequence issue!")
        print(f"   Expected: 1 to {len(predictions_df)}")
        print(f"   Actual: {actual_ranks}")
        
        # Fix ranks
        predictions_df = predictions_df.sort_values('fanduel_fantasy_points', ascending=False)
        predictions_df = predictions_df.reset_index(drop=True)
        predictions_df['rank'] = range(1, len(predictions_df) + 1)
        if 'prediction_rank' in predictions_df.columns:
            predictions_df['prediction_rank'] = predictions_df['rank']
        print("✅ Fixed rank sequence")
    
    print(f"✅ Validation complete: {len(predictions_df)} unique players with ranks 1-{len(predictions_df)}")
    return predictions_df

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

# Get comprehensive features based on available columns
def get_comprehensive_features(df):
    """Get comprehensive feature list based on available columns in the enhanced dataset"""
    
    available_cols = set(df.columns)
    print(f"Total columns available: {len(available_cols)}")
    
    # Core offensive stats
    core_offensive = [
        'passing_yards', 'passing_tds', 'interceptions', 'completions', 'attempts',
        'rushing_yards', 'rushing_tds', 'carries', 'rushing_fumbles', 'rushing_fumbles_lost',
        'receiving_yards', 'receiving_tds', 'receptions', 'targets', 'receiving_fumbles', 'receiving_fumbles_lost',
        'sacks', 'sack_yards', 'sack_fumbles', 'sack_fumbles_lost'
    ]
    
    # Advanced passing metrics
    advanced_passing = [
        'passing_air_yards', 'passing_yards_after_catch', 'passing_first_downs', 
        'passing_epa', 'passing_2pt_conversions', 'pacr', 'dakota'
    ]
    
    # Advanced rushing metrics  
    advanced_rushing = [
        'rushing_first_downs', 'rushing_epa', 'rushing_2pt_conversions'
    ]
    
    # Advanced receiving metrics
    advanced_receiving = [
        'receiving_air_yards', 'receiving_yards_after_catch', 'receiving_first_downs',
        'receiving_epa', 'receiving_2pt_conversions', 'racr', 'target_share', 
        'air_yards_share', 'wopr'
    ]
    
    # SNAP COUNT DATA - This is the key enhancement!
    snap_count_features = [
        'offensive_snaps', 'offensive_snap_pct',
        'defensive_snaps', 'defensive_snap_pct', 
        'special_teams_snaps', 'special_teams_snap_pct',
        'total_snaps'
    ]
    
    # Special teams and misc
    special_misc = [
        'special_teams_tds', 'fantasy_points', 'fantasy_points_ppr'
    ]
    
    # Performance tracking (from enhanced data)
    performance_tracking = [
        'avg_fppg'  # Rolling average from data processing
    ]
    
    # Combine all feature categories
    all_potential_features = (
        core_offensive + advanced_passing + advanced_rushing + 
        advanced_receiving + snap_count_features + special_misc + performance_tracking
    )
    
    # Filter to only features that exist in the dataset
    existing_features = [col for col in all_potential_features if col in available_cols]
    
    print(f"Feature categories breakdown:")
    print(f"  Core offensive: {len([f for f in core_offensive if f in available_cols])}")
    print(f"  Advanced passing: {len([f for f in advanced_passing if f in available_cols])}")
    print(f"  Advanced rushing: {len([f for f in advanced_rushing if f in available_cols])}")
    print(f"  Advanced receiving: {len([f for f in advanced_receiving if f in available_cols])}")
    print(f"  🏈 Snap counts: {len([f for f in snap_count_features if f in available_cols])}")
    print(f"  Special/misc: {len([f for f in special_misc if f in available_cols])}")
    print(f"  Performance tracking: {len([f for f in performance_tracking if f in available_cols])}")
    
    return existing_features

# Enhanced feature engineering with comprehensive snap count insights
def create_enhanced_predictive_features(df):
    """Create enhanced features including snap count derived metrics"""
    
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
    
    # SNAP COUNT DERIVED FEATURES - Crucial for opportunity prediction!
    
    # Offensive opportunity metrics
    if 'offensive_snaps' in df_features.columns:
        # High snap count indicates more opportunity
        df_features['high_snap_count'] = (df_features['offensive_snaps'] >= 50).astype(int)
        
        # Snap count efficiency (fantasy points per snap)
        if 'fanduel_fantasy_points' in df_features.columns:
            df_features['fantasy_per_snap'] = np.where(
                df_features['offensive_snaps'] > 0,
                df_features['fanduel_fantasy_points'] / df_features['offensive_snaps'], 0
            )
    
    # Usage rate relative to team
    if 'offensive_snap_pct' in df_features.columns:
        # Snap percentage bins (starter vs backup vs occasional)
        df_features['snap_role'] = pd.cut(
            df_features['offensive_snap_pct'],
            bins=[0, 25, 60, 100],
            labels=[0, 1, 2],  # 0=backup, 1=rotation, 2=starter
            include_lowest=True
        ).astype(float)
        
        # High usage indicator
        df_features['is_primary_player'] = (df_features['offensive_snap_pct'] >= 60).astype(int)
        
        # Snap reduction indicator (injury/reduced role)
        df_features['reduced_snaps'] = (df_features['offensive_snap_pct'] < 50).astype(int)
    
    # Special teams involvement
    if 'special_teams_snaps' in df_features.columns:
        df_features['special_teams_player'] = (df_features['special_teams_snaps'] > 0).astype(int)
    
    # Total opportunity score (combination of touches and snap count)
    if all(col in df_features.columns for col in ['carries', 'targets', 'offensive_snaps']):
        touches = df_features['carries'].fillna(0) + df_features['targets'].fillna(0)
        df_features['opportunity_score'] = (
            touches * 2 +  # Touches are highly valuable
            df_features['offensive_snaps'].fillna(0) * 0.1  # Snaps provide baseline opportunity
        )
    
    # Advanced efficiency metrics
    if 'passing_epa' in df_features.columns and 'attempts' in df_features.columns:
        df_features['epa_per_attempt'] = np.where(
            df_features['attempts'] > 0,
            df_features['passing_epa'] / df_features['attempts'], 0
        )
    
    if 'receiving_epa' in df_features.columns and 'targets' in df_features.columns:
        df_features['epa_per_target'] = np.where(
            df_features['targets'] > 0,
            df_features['receiving_epa'] / df_features['targets'], 0
        )
    
    # Target quality metrics
    if all(col in df_features.columns for col in ['receiving_air_yards', 'targets']):
        df_features['avg_target_depth'] = np.where(
            df_features['targets'] > 0,
            df_features['receiving_air_yards'] / df_features['targets'], 0
        )
    
    # Completion rate and accuracy
    if 'completions' in df_features.columns and 'attempts' in df_features.columns:
        df_features['completion_rate'] = np.where(
            df_features['attempts'] > 0,
            df_features['completions'] / df_features['attempts'], 0
        )
    
    # Catch rate
    if 'receptions' in df_features.columns and 'targets' in df_features.columns:
        df_features['catch_rate'] = np.where(
            df_features['targets'] > 0,
            df_features['receptions'] / df_features['targets'], 0
        )
    
    # Position-specific usage metrics
    position_col = None
    if 'position_x' in df_features.columns:
        position_col = 'position_x'
    elif 'position' in df_features.columns:
        position_col = 'position'
    
    if position_col:
        # QB: Volume and efficiency
        if 'attempts' in df_features.columns:
            df_features['qb_passing_volume'] = np.where(
                df_features[position_col] == 'QB', df_features['attempts'], 0
            )
        
        # RB: Total touches and snap involvement
        if all(col in df_features.columns for col in ['carries', 'targets']):
            df_features['rb_total_touches'] = np.where(
                df_features[position_col] == 'RB', 
                df_features['carries'].fillna(0) + df_features['targets'].fillna(0), 0
            )
        
        # WR/TE: Target volume and snap rate
        if 'targets' in df_features.columns:
            df_features['wr_te_targets'] = np.where(
                df_features[position_col].isin(['WR', 'TE']), df_features['targets'], 0
            )
        
        if 'offensive_snap_pct' in df_features.columns:
            df_features['wr_te_snap_rate'] = np.where(
                df_features[position_col].isin(['WR', 'TE']), 
                df_features['offensive_snap_pct'], 0
            )
    
    # Recent performance trends (enhanced with snap data)
    if 'avg_fppg' in df_features.columns and 'fanduel_fantasy_points' in df_features.columns:
        df_features['performance_vs_average'] = (
            df_features['fanduel_fantasy_points'] - df_features['avg_fppg']
        )
        
        # Consistency indicator
        df_features['is_consistent_performer'] = (
            (df_features['performance_vs_average'].abs() < 3).astype(int)
        )
        
        # Trending up/down
        df_features['above_season_average'] = (
            df_features['fanduel_fantasy_points'] > df_features['avg_fppg']
        ).astype(int)
    
    # Game script indicators
    if all(col in df_features.columns for col in ['attempts', 'carries']):
        # High pass/run ratio might indicate trailing/leading
        total_plays = df_features['attempts'].fillna(0) + df_features['carries'].fillna(0)
        df_features['pass_heavy_script'] = np.where(
            total_plays > 0,
            df_features['attempts'].fillna(0) / total_plays > 0.6,
            0
        ).astype(int)
    
    # Fill NaN values
    df_features = df_features.fillna(0)
    
    return df_features

# Enhanced feature selection
def get_enhanced_numerical_features(df):
    """Get enhanced numerical features including all snap count metrics"""
    
    # Get all available base features
    comprehensive_features = get_comprehensive_features(df)
    
    # Derived features from feature engineering
    derived_features = [
        # Basic efficiency
        'yards_per_attempt', 'yards_per_carry', 'yards_per_target',
        'completion_rate', 'catch_rate',
        
        # Snap count derived metrics (KEY ADDITIONS!)
        'fantasy_per_snap', 'high_snap_count', 'snap_role', 'is_primary_player',
        'special_teams_player', 'opportunity_score', 'reduced_snaps',
        
        # Advanced efficiency
        'epa_per_attempt', 'epa_per_target', 'avg_target_depth',
        
        # Position-specific features  
        'qb_passing_volume', 'rb_total_touches', 'wr_te_targets', 'wr_te_snap_rate',
        
        # Performance trends
        'performance_vs_average', 'is_consistent_performer', 'above_season_average',
        
        # Game script
        'pass_heavy_script'
    ]
    
    return comprehensive_features, derived_features

# Apply enhanced feature engineering
df_final = create_enhanced_predictive_features(df_with_targets)

# Get comprehensive feature lists
base_features, derived_features = get_enhanced_numerical_features(df_final)

# Combine all numerical features
numerical_features = []

# Add base features that exist
for feature in base_features:
    if feature in df_final.columns:
        numerical_features.append(feature)

# Add derived features that exist  
for feature in derived_features:
    if feature in df_final.columns:
        numerical_features.append(feature)

print(f"\n=== FEATURE ANALYSIS ===")
print(f"Total numerical features: {len(numerical_features)}")
print(f"Features by category:")
print(f"  📊 Base stats: {len([f for f in base_features if f in df_final.columns])}")
print(f"  🔧 Derived features: {len([f for f in derived_features if f in df_final.columns])}")

# Show snap count features specifically
snap_features = [f for f in numerical_features if 'snap' in f.lower() or f in ['total_snaps', 'opportunity_score', 'fantasy_per_snap']]
print(f"  🏈 Snap-related features: {len(snap_features)}")

# Enhanced categorical features
categorical_features = []
if 'position_x' in df_final.columns:
    categorical_features.append('position_x')
elif 'position' in df_final.columns:
    categorical_features.append('position')

if 'recent_team' in df_final.columns:
    categorical_features.append('recent_team')

print(f"\nCategorical features: {categorical_features}")

# Clean categorical data
def clean_categorical_features(df, cat_features):
    """Clean categorical features for encoding"""
    df_clean = df.copy()
    for col in cat_features:
        if col in df_clean.columns:
            # Handle position columns
            if 'position' in col:
                # Ensure valid position values
                valid_positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].apply(
                    lambda x: x if x in valid_positions else 'Unknown'
                )
            else:
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

print(f"\n=== TRAINING DATA SUMMARY ===")
print(f"Final training data: X={X_clean.shape}, y={y_clean.shape}")
print(f"Target columns: {target_cols}")

# Create preprocessor with enhanced features
preprocessor = ColumnTransformer([
    ('num', RobustScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
])

# Fit and transform
X_transformed = preprocessor.fit_transform(X_clean)
print(f"Transformed features: {X_transformed.shape}")

# Proper train/validation split (temporal order)
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

# Enhanced model architecture with reduced regularization for more variance
def create_enhanced_model(input_dim, output_dim):
    """Create model with reduced regularization for more prediction variance"""
    
    inputs = Input(shape=(input_dim,))
    
    # Reduced regularization to allow more variance in predictions
    x = Dense(512, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)  # Reduced from 0.3
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)  # Reduced from 0.3
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.15)(x)  # Reduced from 0.3
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)   # Reduced from 0.2
    
    # Output layer
    outputs = Dense(output_dim, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create and compile model
input_dim = X_train.shape[1]
output_dim = len(target_cols)
model = create_enhanced_model(input_dim, output_dim)

# Adjusted compilation for more variance
model.compile(
    optimizer=Adam(learning_rate=0.0015, beta_1=0.9, beta_2=0.999),  # Slightly higher learning rate
    loss='mse',
    metrics=['mae']
)

print(f"\n=== MODEL ARCHITECTURE ===")
print(f"Input features: {input_dim}")
print(f"Output targets: {output_dim}")
model.summary()

# Training callbacks
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

# Train model
print(f"\n=== TRAINING MODEL ===")
print("Starting enhanced model training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

print("Enhanced model training completed.")

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Enhanced Model Loss')
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
plt.savefig('predictions/learning_curves/enhanced_model_curves.png', dpi=300)
plt.close()

# ===== IMPROVED ROOKIE DETECTION =====

def is_actually_rookie(player_name, player_id, historical_df, current_season=2025):
    """Determine if player is actually a rookie (entered NFL in current season)"""
    
    # Check if player has ANY NFL game data
    player_history = historical_df[
        (historical_df['player_id'] == player_id) |
        (historical_df['player_display_name'].str.contains(player_name, case=False, na=False)) |
        (historical_df['player_name'].str.contains(player_name, case=False, na=False))
    ]
    
    if player_history.empty:
        # No NFL history = rookie
        return True
    
    # Check seasons played
    seasons_played = player_history['season'].unique()
    
    # If player only appears in current season or later, they're a rookie
    if all(season >= current_season for season in seasons_played):
        return True
    
    # Manual override for known 2025 rookies (since we're predicting 2025)
    known_2025_rookies = [
        'Cam Ward', 'Ashton Jeanty', 'Travis Hunter', 'Abdul Carter',
        'Shedeur Sanders', 'Tetairoa McMillan', 'Will Campbell', 'Mason Graham',
        'Colston Loveland', 'Luther Burden III', 'Mykel Williams'
    ]
    
    if any(rookie.lower() in player_name.lower() for rookie in known_2025_rookies):
        return True
    
    return False

# ===== ENHANCED DEPTH CHART ANALYZER =====

class EnhancedDepthChartAnalyzer:
    """Enhanced depth chart analyzer with rookie context"""
    
    def __init__(self, depth_chart_data):
        self.depth_charts = depth_chart_data
        self._process_depth_charts()
    
    def _process_depth_charts(self):
        """Process depth chart data with team context"""
        # Focus on offensive skill positions
        offensive_positions = ['QB', 'RB', 'WR', 'TE', 'FB']
        
        self.offensive_depth = self.depth_charts[
            self.depth_charts['pos_abb'].isin(offensive_positions)
        ].copy()
        
        # Enhanced role analysis
        self.offensive_depth['depth_role'] = self.offensive_depth['pos_rank'].apply(
            lambda x: 'starter' if x == 1 else 'backup' if x == 2 else 'deep_backup'
        )
        
        # Calculate team position depth
        self.team_position_depth = {}
        for (team, position), group in self.offensive_depth.groupby(['team', 'pos_abb']):
            sorted_group = group.sort_values('pos_rank')
            self.team_position_depth[(team, position)] = {
                'total_players': len(sorted_group),
                'starter': sorted_group.iloc[0]['player_name'] if len(sorted_group) > 0 else None,
                'backup': sorted_group.iloc[1]['player_name'] if len(sorted_group) > 1 else None,
                'depth_chart': sorted_group[['player_name', 'pos_rank', 'depth_role']].to_dict('records')
            }
        
        print(f"Processed depth charts: {len(self.offensive_depth)} offensive players")
        
        # Create lookup dictionaries
        self.player_roles = {}
        for _, row in self.offensive_depth.iterrows():
            player_name = row['player_name']
            self.player_roles[player_name] = {
                'team': row['team'],
                'position': row['pos_abb'],
                'depth_rank': row['pos_rank'],
                'role': row['depth_role'],
                'pos_name': row['pos_name']
            }
    
    def get_player_role(self, player_name):
        """Get current role for a player from depth charts"""
        # Try exact match
        if player_name in self.player_roles:
            return self.player_roles[player_name]
        
        # Try partial matching for different name formats
        for depth_name, role_info in self.player_roles.items():
            if self._names_similar(player_name, depth_name):
                return role_info
        
        return None
    
    def get_team_competition(self, team, position):
        """Get competition level at a position for a team"""
        key = (team, position)
        if key in self.team_position_depth:
            depth_info = self.team_position_depth[key]
            return {
                'total_players': depth_info['total_players'],
                'starter': depth_info['starter'],
                'backup': depth_info['backup'],
                'competition_level': 'high' if depth_info['total_players'] >= 4 else 'medium' if depth_info['total_players'] >= 3 else 'low'
            }
        return None
    
    def analyze_rookie_opportunity(self, player_name, team, position):
        """Analyze opportunity for a rookie based on team depth chart"""
        team_depth = self.get_team_competition(team, position)
        player_role = self.get_player_role(player_name)
        
        if not team_depth:
            return {'opportunity': 'unknown', 'context': 'No team depth chart data'}
        
        if not player_role:
            return {'opportunity': 'low', 'context': f'Not found on {team} depth chart'}
        
        opportunity_analysis = {
            'depth_rank': player_role['depth_rank'],
            'role': player_role['role'],
            'team_depth': team_depth['total_players'],
            'competition_level': team_depth['competition_level']
        }
        
        # Determine opportunity level
        if player_role['role'] == 'starter':
            opportunity_analysis['opportunity'] = 'high'
            opportunity_analysis['context'] = f"Listed as starter on {team} depth chart"
        elif player_role['role'] == 'backup' and position == 'RB':
            # RB backups can still get significant work
            opportunity_analysis['opportunity'] = 'medium'
            opportunity_analysis['context'] = f"Backup RB, could see rotation work"
        elif player_role['role'] == 'backup':
            opportunity_analysis['opportunity'] = 'low'
            opportunity_analysis['context'] = f"Backup behind {team_depth.get('starter', 'established starter')}"
        else:
            opportunity_analysis['opportunity'] = 'very_low'
            opportunity_analysis['context'] = f"Deep on depth chart"
        
        return opportunity_analysis
    
    def _names_similar(self, name1, name2):
        """Enhanced name similarity check"""
        name1_parts = name1.lower().split()
        name2_parts = name2.lower().split()
        
        if len(name1_parts) >= 2 and len(name2_parts) >= 2:
            # Last name must match
            last_name_match = name1_parts[-1] == name2_parts[-1]
            
            # First name or initial must match
            first_name_match = (
                name1_parts[0] == name2_parts[0] or  # Full first name
                name1_parts[0][0] == name2_parts[0][0]  # First initial
            )
            
            return last_name_match and first_name_match
        
        return False

class InjuryStatusAnalyzer:
    """Analyze player injury status and team depth situations"""
    
    def __init__(self, roster_data, depth_charts):
        self.roster_data = roster_data
        self.depth_charts = depth_charts
        self.injury_overrides = self._setup_injury_overrides()
        self.backup_situations = self._identify_backup_situations()
    
    def _setup_injury_overrides(self):
        """Manual injury/suspension overrides for current season"""
        # Update this list based on current NFL injury reports
        return {
            # Format: 'Player Name': {'status': 'OUT'/'QUESTIONABLE'/'IR', 'replacement': 'Backup Name', 'reason': 'injury reason'}
            'Joe Burrow': {
                'status': 'IR', 
                'replacement': 'Jake Browning', 
                'reason': 'wrist injury',
                'team': 'CIN',
                'position': 'QB'
            },
            # Add other injured players as needed
            # 'Aaron Rodgers': {'status': 'OUT', 'replacement': 'Zach Wilson', 'reason': 'achilles injury', 'team': 'NYJ', 'position': 'QB'},
        }
    
    def _identify_backup_situations(self):
        """Identify backup players who should get starter projections"""
        backup_situations = {}
        
        for injured_player, info in self.injury_overrides.items():
            if info['status'] in ['OUT', 'IR'] and 'replacement' in info:
                replacement = info['replacement']
                backup_situations[replacement] = {
                    'replacing': injured_player,
                    'reason': f"Starting due to {injured_player} {info['reason']}",
                    'team': info['team'],
                    'position': info['position'],
                    'role': 'emergency_starter'
                }
        
        return backup_situations
    
    def check_player_status(self, player_name):
        """Check if player has injury/status issues"""
        # Direct injury match
        if player_name in self.injury_overrides:
            return self.injury_overrides[player_name]
        
        # Check if player is a backup getting starter role
        if player_name in self.backup_situations:
            return self.backup_situations[player_name]
        
        return None
    
    def should_zero_out_player(self, player_name):
        """Determine if player should get 0 points due to injury"""
        status_info = self.check_player_status(player_name)
        if status_info:
            return status_info.get('status') in ['OUT', 'IR', 'SUSPENDED']
        return False
    
    def should_boost_backup(self, player_name):
        """Determine if backup should get starter-level projections"""
        return player_name in self.backup_situations
    
    def get_adjustment_info(self, player_name):
        """Get detailed adjustment information for logging"""
        status_info = self.check_player_status(player_name)
        if not status_info:
            return None
        
        if self.should_zero_out_player(player_name):
            return {
                'type': 'injury_zero',
                'reason': f"OUT due to {status_info.get('reason', 'injury')}",
                'original_role': 'injured_starter'
            }
        elif self.should_boost_backup(player_name):
            return {
                'type': 'backup_boost', 
                'reason': status_info.get('reason', 'replacing injured starter'),
                'replacing': status_info.get('replacing'),
                'original_role': 'backup_now_starting'
            }
        
        return None

# Initialize the injury analyzer
print("\n=== INITIALIZING INJURY STATUS ANALYZER ===")
injury_analyzer = InjuryStatusAnalyzer(latest_rosters, current_depth_charts)

# Report current injury situations
print("Current injury overrides:")
for player, info in injury_analyzer.injury_overrides.items():
    status = info.get('status', 'Unknown')
    reason = info.get('reason', 'Unknown')
    replacement = info.get('replacement', 'None')
    print(f"  {player} ({status}): {reason} -> {replacement}")

print("\nBackup players getting starter roles:")
for backup, info in injury_analyzer.backup_situations.items():
    replacing = info.get('replacing', 'Unknown')
    reason = info.get('reason', 'Unknown')
    print(f"  {backup}: {reason}")

# ===== ENHANCED ROOKIE PREDICTOR =====

class EnhancedRookiePredictor:
    """Enhanced rookie predictor with depth chart integration"""
    
    def __init__(self, historical_df, depth_analyzer, current_season=2025):
        self.historical_df = historical_df
        self.depth_analyzer = depth_analyzer
        self.current_season = current_season
        self.draft_data = self._load_draft_data()
        self.rookie_baselines = self._calculate_baselines()
    
    def _load_draft_data(self):
        """Load draft data for current season"""
        try:
            draft_data = import_draft_picks([self.current_season])
            print(f"Loaded {len(draft_data)} draft picks for {self.current_season}")
            return draft_data
        except Exception as e:
            print(f"Could not load draft data: {e}")
            return pd.DataFrame()
    
    def _calculate_baselines(self):
        """Calculate rookie baselines from historical data"""
        player_first_seasons = self.historical_df.groupby('player_id')['season'].min()
        
        rookie_data = []
        for player_id, first_season in player_first_seasons.items():
            player_rookie_games = self.historical_df[
                (self.historical_df['player_id'] == player_id) &
                (self.historical_df['season'] == first_season) &
                (self.historical_df['week'] != 'AVG')
            ]
            if len(player_rookie_games) >= 3:
                rookie_data.append(player_rookie_games)
        
        if not rookie_data:
            return {}
        
        all_rookies = pd.concat(rookie_data)
        
        baselines = {}
        for pos in ['QB', 'RB', 'WR', 'TE']:
            pos_rookies = all_rookies[all_rookies['position_x'] == pos]
            if len(pos_rookies) > 20:
                baselines[pos] = {
                    'avg_fppg': pos_rookies['fanduel_fantasy_points'].mean(),
                    'avg_snaps': pos_rookies['offensive_snap_pct'].mean(),
                    'sample_size': len(pos_rookies)
                }
                print(f"Rookie baseline {pos}: {baselines[pos]['avg_fppg']:.1f} FPPG")
        
        return baselines
    
    def get_draft_info(self, player_name):
        """Get draft information for a player"""
        if self.draft_data.empty:
            return None
        
        matches = self.draft_data[
            self.draft_data['pfr_player_name'].str.lower() == player_name.lower()
        ]
        
        if matches.empty:
            matches = self.draft_data[
                self.draft_data['pfr_player_name'].str.contains(player_name, case=False, na=False)
            ]
        
        if not matches.empty:
            draft_info = matches.iloc[0]
            return {
                'draft_position': int(draft_info['pick']),
                'round': int(draft_info['round']),
                'team': draft_info['team']
            }
        return None
    
    def predict_rookie(self, player_name, position, team):
        """Generate rookie prediction with depth chart analysis"""
        baseline = self.rookie_baselines.get(position)
        if not baseline:
            return None
        
        base_points = baseline['avg_fppg']
        
        # Get draft information
        draft_info = self.get_draft_info(player_name)
        if draft_info:
            pick = draft_info['draft_position']
            if pick <= 10:
                draft_multiplier = 1
                tier = 'elite'
            elif pick <= 32:
                draft_multiplier = 0.8
                tier = 'high'
            elif pick <= 64:
                draft_multiplier = 0.7
                tier = 'medium'
            elif pick <= 100:
                draft_multiplier = 0.6
                tier = 'late'
            else:
                draft_multiplier = 0.5
                tier = 'very_late'
        else:
            draft_multiplier = 0.3
            tier = 'undrafted'
        
        # CRITICAL: Analyze depth chart opportunity
        opportunity_analysis = self.depth_analyzer.analyze_rookie_opportunity(
            player_name, team, position
        )
        
        # Depth chart multiplier based on opportunity
        if opportunity_analysis['opportunity'] == 'high':
            depth_multiplier = 1.5  # Starter role
        elif opportunity_analysis['opportunity'] == 'medium':
            depth_multiplier = 0.8  # Backup with opportunity
        elif opportunity_analysis['opportunity'] == 'low':
            depth_multiplier = 0.3  # Clear backup
        else:  # very_low
            depth_multiplier = 0.1  # Deep backup
        
        # Position-specific adjustments
        if position == 'QB':
            if opportunity_analysis['opportunity'] == 'high':
                position_multiplier = 1.0  # Starting rookie QB
            else:
                position_multiplier = 0.2  # Backup rookie QB unlikely to play
        elif position == 'RB':
            position_multiplier = 1.2  # RBs can contribute early
        elif position in ['WR', 'TE']:
            if tier in ['elite', 'high']:
                position_multiplier = 1.0
            else:
                position_multiplier = 0.7
        else:
            position_multiplier = 1.0
        
        # Calculate final points
        final_points = base_points * draft_multiplier * depth_multiplier * position_multiplier
        final_points = max(1.0, min(25.0, final_points))
        
        return {
            'fanduel_fantasy_points': round(final_points, 1),
            'prediction_type': 'rookie',
            'draft_tier': tier,
            'draft_info': draft_info,
            'depth_chart_opportunity': opportunity_analysis,
            'baseline_used': baseline['avg_fppg'],
            'draft_multiplier': draft_multiplier,
            'depth_multiplier': depth_multiplier,
            'position_multiplier': position_multiplier
        }

# Initialize enhanced depth chart analyzer
print("\n=== INITIALIZING ENHANCED DEPTH CHART ANALYZER ===")
enhanced_depth_analyzer = EnhancedDepthChartAnalyzer(current_depth_charts)

# Initialize enhanced rookie predictor  
print("\n=== INITIALIZING ENHANCED ROOKIE PREDICTION SYSTEM ===")
enhanced_rookie_predictor = EnhancedRookiePredictor(df_final_clean, enhanced_depth_analyzer, year)

# ===== ENHANCED COMPREHENSIVE PREDICTION FUNCTION =====

def comprehensive_predict_week_enhanced(player_name, week):
    """Enhanced prediction with injury status integration"""
    try:
        # Find player in roster
        player_matches = latest_rosters[
            latest_rosters['player_name'].str.contains(player_name, case=False, na=False)
        ]
        
        if player_matches.empty:
            return None
        
        player_info = player_matches.iloc[0]
        position = player_info['position']
        team = player_info.get('team', 'Unknown')
        player_id = player_info.get('player_id', '')
        
        # CRITICAL: Check injury status FIRST
        injury_status = injury_analyzer.check_player_status(player_name)
        
        if injury_analyzer.should_zero_out_player(player_name):
            # Player is injured/out - return 0 points
            adjustment_info = injury_analyzer.get_adjustment_info(player_name)
            print(f"INJURED: {player_name} -> 0 points ({adjustment_info['reason']})")
            
            return {
                'player_name': player_name,
                'position': position,
                'team': team,
                'fanduel_fantasy_points': 0.0,
                'prediction_type': 'injured_out',
                'injury_status': injury_status.get('status', 'OUT'),
                'injury_reason': injury_status.get('reason', 'injury'),
                'role_adjustment': adjustment_info['reason']
            }
        
        # Check if this is a backup getting starter role
        backup_boost = injury_analyzer.should_boost_backup(player_name)
        
        # Enhanced rookie detection
        is_rookie = is_actually_rookie(player_name, player_id, df_final_clean, year)
        
        if is_rookie:
            # Rookie prediction logic (same as before)
            print(f"ROOKIE: {player_name} ({position}, {team})")
            rookie_pred = enhanced_rookie_predictor.predict_rookie(player_name, position, team)
            if rookie_pred:
                rookie_pred['player_name'] = player_name
                rookie_pred['position'] = position
                rookie_pred['team'] = team
                return rookie_pred
            else:
                depth_role = enhanced_depth_analyzer.get_player_role(player_name)
                fallback_points = 3.0
                if depth_role and depth_role['role'] == 'starter':
                    fallback_points = 8.0
                
                return {
                    'player_name': player_name,
                    'position': position,
                    'team': team,
                    'fanduel_fantasy_points': fallback_points,
                    'prediction_type': 'rookie_fallback'
                }
        
        # Veteran player logic with injury adjustments
        print(f"VETERAN: {player_name} ({position}, {team})")
        
        # Get player data for ML prediction
        player_data = df_final_clean[
            (df_final_clean['player_display_name'].str.contains(player_name, case=False, na=False)) |
            (df_final_clean['player_name'].str.contains(player_name, case=False, na=False))
        ]
        
        if player_data.empty:
            return None
        
        recent_data = player_data[player_data['season'] >= 2023]
        if recent_data.empty:
            return None
        
        recent_stats = recent_data.sort_values(['season', 'week']).iloc[-1]
        
        # ML prediction (same as before)
        input_data = pd.DataFrame()
        
        for col in numerical_features:
            if col in recent_stats.index:
                input_data[col] = [recent_stats[col]]
            else:
                input_data[col] = [0]
        
        for cat_col in categorical_features:
            if 'position' in cat_col:
                input_data[cat_col] = [str(position)]
            elif cat_col == 'recent_team':
                input_data[cat_col] = [str(team)]
            else:
                input_data[cat_col] = ['Unknown']
        
        input_transformed = preprocessor.transform(input_data)
        prediction = model.predict(input_transformed, verbose=0)[0]
        
        result = {}
        for i, stat in enumerate(target_cols):
            if stat == 'fanduel_fantasy_points':
                result[stat] = max(0, min(70, prediction[i]))
            elif 'yards' in stat:
                result[stat] = max(0, min(400, prediction[i]))
            elif 'tds' in stat or stat == 'receptions':
                result[stat] = max(0, min(6, prediction[i]))
            else:
                result[stat] = max(0, prediction[i])
            result[stat] = round(result[stat], 2)
        
        # INJURY/BACKUP ADJUSTMENTS
        original_points = result['fanduel_fantasy_points']
        adjustment = "no adjustment"
        
        if backup_boost:
            # This backup is now starting due to injury
            adjustment_info = injury_analyzer.get_adjustment_info(player_name)
            replacing = adjustment_info.get('replacing', 'injured starter')
            
            if position == 'QB':
                # Backup QBs getting starter role - significant boost
                result['fanduel_fantasy_points'] *= 2.5  # Big boost for emergency starter
                adjustment = f"emergency starter boost (replacing {replacing})"
                print(f"BACKUP BOOST: {player_name} now starting for {replacing}")
            else:
                # Other positions - moderate boost
                result['fanduel_fantasy_points'] *= 1.8
                adjustment = f"injury replacement boost (replacing {replacing})"
        
        # Standard depth chart adjustments (only if not already boosted)
        if not backup_boost:
            depth_role = enhanced_depth_analyzer.get_player_role(player_name)
            
            if depth_role:
                if depth_role['role'] == 'backup' or depth_role['depth_rank'] > 2:
                    if position == 'QB':
                        result['fanduel_fantasy_points'] *= 0.12
                        adjustment = "depth chart backup QB penalty"
                    else:
                        result['fanduel_fantasy_points'] *= 0.3
                        adjustment = f"depth chart backup {position} penalty"
                elif depth_role['role'] == 'deep_backup':
                    result['fanduel_fantasy_points'] *= 0.05
                    adjustment = "deep backup penalty"
                else:
                    # Elite player boosts for starters
                    if position == 'QB':
                        elite_qbs = ['Josh Allen', 'Lamar Jackson', 'Patrick Mahomes']  # Removed Joe Burrow
                        tier1_qbs = ['Dak Prescott', 'Tua Tagovailoa', 'C.J. Stroud', 'Jalen Hurts']
                        
                        if player_name in elite_qbs:
                            result['fanduel_fantasy_points'] *= 1.5
                            adjustment = "elite QB boost"
                        elif player_name in tier1_qbs:
                            result['fanduel_fantasy_points'] *= 1.25
                            adjustment = "tier 1 QB boost"
        
        result['fanduel_fantasy_points'] = round(result['fanduel_fantasy_points'], 1)
        
        result.update({
            'player_name': player_name,
            'position': position,
            'team': team,
            'prediction_type': 'veteran_ml',
            'depth_chart_role': depth_role['role'] if 'depth_role' in locals() and depth_role else 'unknown',
            'depth_rank': depth_role['depth_rank'] if 'depth_role' in locals() and depth_role else 'N/A',
            'role_adjustment': adjustment,
            'original_prediction': round(original_points, 1),
            'injury_replacement': backup_boost
        })
        
        return result
        
    except Exception as e:
        print(f"Error predicting for {player_name}: {e}")
        return None

# Updated prediction generation function
def get_comprehensive_predictions(position, week):
    """Generate predictions for all players at a position with depth chart analysis"""
    predictions = []
    
    active_players = latest_rosters[
        (latest_rosters['position'] == position) & 
        (latest_rosters['status'] == 'ACT')
    ]
    
    print(f"\nGenerating predictions for {len(active_players)} {position}s...")
    
    veteran_count = 0
    rookie_count = 0
    depth_chart_adjustments = 0
    processed_players = set()  # Track processed players to avoid duplicates
    
    for _, player in tqdm(active_players.iterrows(), 
                         total=len(active_players), 
                         desc=f"Predicting {position}"):
        
        player_name = player['player_name']
        
        # CRITICAL: Skip if we already processed this player
        if player_name in processed_players:
            print(f"⚠️  Skipping duplicate player: {player_name}")
            continue
        
        pred = comprehensive_predict_week_enhanced(player_name, week)
        if pred:
            processed_players.add(player_name)  # Mark as processed
            predictions.append(pred)
            
            if pred.get('prediction_type', '').startswith('rookie'):
                rookie_count += 1
            else:
                veteran_count += 1
                if 'depth chart' in pred.get('role_adjustment', ''):
                    depth_chart_adjustments += 1
    
    if predictions:
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # CRITICAL: Remove any remaining duplicates based on player_name
        print(f"Before duplicate removal: {len(predictions_df)} players")
        initial_count = len(predictions_df)
        
        predictions_df = predictions_df.drop_duplicates(subset=['player_name'], keep='first')
        
        final_count = len(predictions_df)
        if initial_count != final_count:
            print(f"⚠️  Removed {initial_count - final_count} duplicates after DataFrame creation")
        
        # Sort by fantasy points
        predictions_df = predictions_df.sort_values('fanduel_fantasy_points', ascending=False)
        
        # RESET INDEX before assigning ranks
        predictions_df = predictions_df.reset_index(drop=True)
        
        # Assign clean sequential ranks
        predictions_df['rank'] = range(1, len(predictions_df) + 1)
        predictions_df['prediction_rank'] = predictions_df['rank']  # Mirror rank to prediction_rank
        
        # Final validation
        duplicate_check = predictions_df['player_name'].duplicated().sum()
        if duplicate_check > 0:
            print(f"❌ ERROR: Still have {duplicate_check} duplicates after cleaning!")
            print("Duplicate players:", predictions_df[predictions_df['player_name'].duplicated()]['player_name'].tolist())
            
            # Emergency cleanup
            predictions_df = predictions_df.drop_duplicates(subset=['player_name'], keep='first').reset_index(drop=True)
            predictions_df['rank'] = range(1, len(predictions_df) + 1)
            predictions_df['prediction_rank'] = predictions_df['rank']
        
        # Save predictions
        os.makedirs('predictions', exist_ok=True)
        filename = f'predictions/{position}_predictions_week{week}_season{year}.csv'
        predictions_df.to_csv(filename, index=False)
        
        print(f"\n📊 {position} Summary:")
        print(f"  • Total: {len(predictions_df)} players")
        print(f"  • Veterans (ML): {veteran_count}")
        print(f"  • Rookies: {rookie_count}")
        print(f"  • Depth chart adjustments: {depth_chart_adjustments}")
        print(f"  • Ranks: 1 to {len(predictions_df)}")
        print(f"  • Saved: {filename}")
        
        return predictions_df
    
    return pd.DataFrame()

# Generate predictions for all positions
positions = ['QB', 'RB', 'WR', 'TE']
week_to_predict = 5

print("\n" + "="*80)
print("COMPREHENSIVE PREDICTIONS WITH ENHANCED DEPTH CHART INTEGRATION")
print("="*80)

start_time = time.time()

for position in positions:
    print(f"\n{'='*20} {position} PREDICTIONS {'='*20}")
    predictions_df = get_comprehensive_predictions(position, week_to_predict)
    
    if not predictions_df.empty:
        print(f"\nTop 10 {position} predictions:")
        top_10 = predictions_df.head(10)
        
        for _, player in top_10.iterrows():
            name = player['player_name']
            points = player['fanduel_fantasy_points']
            pred_type = player.get('prediction_type', 'unknown')
            
            if 'rookie' in pred_type:
                draft_tier = player.get('draft_tier', 'unknown')
                if 'depth_chart_opportunity' in player:
                    depth_opp = player['depth_chart_opportunity']
                    depth_role = depth_opp.get('role', 'unknown')
                    opportunity = depth_opp.get('opportunity', 'unknown')
                    print(f"  {player['rank']:2d}. {name:20s}: {points:5.1f} pts [ROOKIE - {draft_tier}, depth: {depth_role}, opp: {opportunity}]")
                else:
                    print(f"  {player['rank']:2d}. {name:20s}: {points:5.1f} pts [ROOKIE - {draft_tier}]")
            else:
                depth_role = player.get('depth_chart_role', 'N/A')
                depth_rank = player.get('depth_rank', 'N/A')
                adjustment = player.get('role_adjustment', 'none')
                original = player.get('original_prediction', points)
                
                if 'penalty' in adjustment:
                    print(f"  {player['rank']:2d}. {name:20s}: {points:5.1f} pts [VET - depth: {depth_role}#{depth_rank}, was {original:.1f}]")
                elif 'boost' in adjustment:
                    print(f"  {player['rank']:2d}. {name:20s}: {points:5.1f} pts [VET - depth: {depth_role}#{depth_rank}, boosted from {original:.1f}]")
                else:
                    print(f"  {player['rank']:2d}. {name:20s}: {points:5.1f} pts [VET - depth: {depth_role}#{depth_rank}]")
        
        # Show detailed analysis
        rookies = predictions_df[predictions_df['prediction_type'].str.contains('rookie', na=False)]
        veterans = predictions_df[predictions_df['prediction_type'] == 'veteran_ml']
        
        if len(rookies) > 0:
            print(f"\n🎓 Rookie Analysis ({len(rookies)} players):")
            print(f"  • Avg projection: {rookies['fanduel_fantasy_points'].mean():.1f} FPPG")
            print(f"  • Top rookie: {rookies.iloc[0]['player_name']} ({rookies.iloc[0]['fanduel_fantasy_points']:.1f} pts)")
            
            if 'draft_tier' in rookies.columns:
                tier_counts = rookies['draft_tier'].value_counts()
                print(f"  • Draft tiers: {dict(tier_counts)}")
        
        if len(veterans) > 0:
            print(f"\n⭐ Veteran Analysis ({len(veterans)} players):")
            print(f"  • Avg projection: {veterans['fanduel_fantasy_points'].mean():.1f} FPPG")
            
            # Show depth chart role distribution
            if 'depth_chart_role' in veterans.columns:
                role_counts = veterans['depth_chart_role'].value_counts()
                print(f"  • Depth chart roles: {dict(role_counts)}")
            
            # Show players with significant adjustments
            boosted_players = veterans[veterans['role_adjustment'].str.contains('boost', na=False)]
            penalized_players = veterans[veterans['role_adjustment'].str.contains('penalty', na=False)]
            
            if len(boosted_players) > 0:
                print(f"  • Elite players boosted: {len(boosted_players)}")
                print(f"    Examples: {', '.join(boosted_players['player_name'].head(3).tolist())}")
            
            if len(penalized_players) > 0:
                print(f"  • Backup players penalized: {len(penalized_players)}")
                print(f"    Examples: {', '.join(penalized_players['player_name'].head(3).tolist())}")

end_time = time.time()
print(f"\n⏱️  Total prediction time: {end_time - start_time:.1f} seconds")

print(f"\n🎉 COMPREHENSIVE PREDICTION SYSTEM WITH ENHANCED DEPTH CHARTS COMPLETE!")
print(f"✅ Features implemented:")
print(f"   • Enhanced rookie detection (Caleb Williams = veteran, Cam Ward = rookie)")
print(f"   • Real-time depth chart integration with opportunity analysis")
print(f"   • Veteran ML predictions with {len(numerical_features)} features")
print(f"   • Rookie predictions with draft position + depth chart context")
print(f"   • Elite player tier adjustments for more variance")
print(f"   • Backup detection via current depth charts")
print(f"   • Depth chart opportunity scoring for rookies")

print(f"\n📁 Check CSV files in predictions/ folder for detailed results")
print(f"📊 Files include depth_chart_opportunity, role analysis, and comprehensive context")