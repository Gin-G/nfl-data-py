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

# Import injury integration
from injuries import integrate_sportradar_injuries

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging with both file and console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_debug.log', mode='w'),  # Write to file
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger(__name__)

# Load the enhanced data
logging.info("Loading enhanced data from CSV...")
df = pd.read_csv('data/nfl_dataset.csv')
logging.info(f"Data loaded. Shape: {df.shape}")

# Define prediction parameters EARLY (needed for injury loading)
week_to_predict = 5  # UPDATE THIS FOR THE WEEK YOU WANT TO PREDICT
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
    
    # Keep the MOST RECENT week's data (highest week number)
    latest_rosters = latest_rosters.sort_values(['player_name', 'week'], 
                                                ascending=[True, False])
    latest_rosters = latest_rosters.drop_duplicates(subset=['player_name'], keep='first')
    
    print(f"After deduplication: {len(latest_rosters)} rows")
    print(f"Kept most recent week data for each player")
else:
    print("No duplicate players found in roster data")

# Load current depth charts for real-time role detection
print("Loading current depth charts...")
current_depth_charts = import_depth_charts([year])
print(f"Loaded {len(current_depth_charts)} depth chart entries")

# LOAD INJURY DATA FROM SPORTRADAR API
print("\n" + "="*80)
print("LOADING INJURY DATA FROM SPORTRADAR API")
print("="*80)

injury_overrides, backup_situations = integrate_sportradar_injuries(
    week=week_to_predict,
    roster_data=latest_rosters,
    depth_charts=current_depth_charts
)

print(f"\n✅ Injury data loaded:")
print(f"   • {len(injury_overrides)} injured players (OUT/DOUBTFUL)")
print(f"   • {len(backup_situations)} backups elevated to starter roles")

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
    print(f"  Snap counts: {len([f for f in snap_count_features if f in available_cols])}")
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
    
    # SNAP COUNT DERIVED FEATURES
    if 'offensive_snaps' in df_features.columns:
        df_features['high_snap_count'] = (df_features['offensive_snaps'] >= 50).astype(int)
        
        if 'fanduel_fantasy_points' in df_features.columns:
            df_features['fantasy_per_snap'] = np.where(
                df_features['offensive_snaps'] > 0,
                df_features['fanduel_fantasy_points'] / df_features['offensive_snaps'], 0
            )
    
    if 'offensive_snap_pct' in df_features.columns:
        df_features['snap_role'] = pd.cut(
            df_features['offensive_snap_pct'],
            bins=[0, 25, 60, 100],
            labels=[0, 1, 2],
            include_lowest=True
        ).astype(float)
        
        df_features['is_primary_player'] = (df_features['offensive_snap_pct'] >= 60).astype(int)
        df_features['reduced_snaps'] = (df_features['offensive_snap_pct'] < 50).astype(int)
    
    if 'special_teams_snaps' in df_features.columns:
        df_features['special_teams_player'] = (df_features['special_teams_snaps'] > 0).astype(int)
    
    if all(col in df_features.columns for col in ['carries', 'targets', 'offensive_snaps']):
        touches = df_features['carries'].fillna(0) + df_features['targets'].fillna(0)
        df_features['opportunity_score'] = (
            touches * 2 +
            df_features['offensive_snaps'].fillna(0) * 0.1
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
    
    if all(col in df_features.columns for col in ['receiving_air_yards', 'targets']):
        df_features['avg_target_depth'] = np.where(
            df_features['targets'] > 0,
            df_features['receiving_air_yards'] / df_features['targets'], 0
        )
    
    if 'completions' in df_features.columns and 'attempts' in df_features.columns:
        df_features['completion_rate'] = np.where(
            df_features['attempts'] > 0,
            df_features['completions'] / df_features['attempts'], 0
        )
    
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
        if 'attempts' in df_features.columns:
            df_features['qb_passing_volume'] = np.where(
                df_features[position_col] == 'QB', df_features['attempts'], 0
            )
        
        if all(col in df_features.columns for col in ['carries', 'targets']):
            df_features['rb_total_touches'] = np.where(
                df_features[position_col] == 'RB', 
                df_features['carries'].fillna(0) + df_features['targets'].fillna(0), 0
            )
        
        if 'targets' in df_features.columns:
            df_features['wr_te_targets'] = np.where(
                df_features[position_col].isin(['WR', 'TE']), df_features['targets'], 0
            )
        
        if 'offensive_snap_pct' in df_features.columns:
            df_features['wr_te_snap_rate'] = np.where(
                df_features[position_col].isin(['WR', 'TE']), 
                df_features['offensive_snap_pct'], 0
            )
    
    if 'avg_fppg' in df_features.columns and 'fanduel_fantasy_points' in df_features.columns:
        df_features['performance_vs_average'] = (
            df_features['fanduel_fantasy_points'] - df_features['avg_fppg']
        )
        
        df_features['is_consistent_performer'] = (
            (df_features['performance_vs_average'].abs() < 3).astype(int)
        )
        
        df_features['above_season_average'] = (
            df_features['fanduel_fantasy_points'] > df_features['avg_fppg']
        ).astype(int)
    
    if all(col in df_features.columns for col in ['attempts', 'carries']):
        total_plays = df_features['attempts'].fillna(0) + df_features['carries'].fillna(0)
        df_features['pass_heavy_script'] = np.where(
            total_plays > 0,
            df_features['attempts'].fillna(0) / total_plays > 0.6,
            0
        ).astype(int)
    
    df_features = df_features.fillna(0)
    
    return df_features

def get_enhanced_numerical_features(df):
    """Get enhanced numerical features including all snap count metrics"""
    
    comprehensive_features = get_comprehensive_features(df)
    
    derived_features = [
        'yards_per_attempt', 'yards_per_carry', 'yards_per_target',
        'completion_rate', 'catch_rate',
        'fantasy_per_snap', 'high_snap_count', 'snap_role', 'is_primary_player',
        'special_teams_player', 'opportunity_score', 'reduced_snaps',
        'epa_per_attempt', 'epa_per_target', 'avg_target_depth',
        'qb_passing_volume', 'rb_total_touches', 'wr_te_targets', 'wr_te_snap_rate',
        'performance_vs_average', 'is_consistent_performer', 'above_season_average',
        'pass_heavy_script'
    ]
    
    return comprehensive_features, derived_features

# Apply enhanced feature engineering
df_final = create_enhanced_predictive_features(df_with_targets)

# Get comprehensive feature lists
base_features, derived_features = get_enhanced_numerical_features(df_final)

# Combine all numerical features
numerical_features = []

for feature in base_features:
    if feature in df_final.columns:
        numerical_features.append(feature)

for feature in derived_features:
    if feature in df_final.columns:
        numerical_features.append(feature)

print(f"\n=== FEATURE ANALYSIS ===")
print(f"Total numerical features: {len(numerical_features)}")
print(f"Features by category:")
print(f"  Base stats: {len([f for f in base_features if f in df_final.columns])}")
print(f"  Derived features: {len([f for f in derived_features if f in df_final.columns])}")

snap_features = [f for f in numerical_features if 'snap' in f.lower() or f in ['total_snaps', 'opportunity_score', 'fantasy_per_snap']]
print(f"  Snap-related features: {len(snap_features)}")

# Enhanced categorical features
categorical_features = []
if 'position_x' in df_final.columns:
    categorical_features.append('position_x')
elif 'position' in df_final.columns:
    categorical_features.append('position')

if 'recent_team' in df_final.columns:
    categorical_features.append('recent_team')

print(f"\nCategorical features: {categorical_features}")

def clean_categorical_features(df, cat_features):
    """Clean categorical features for encoding"""
    df_clean = df.copy()
    for col in cat_features:
        if col in df_clean.columns:
            if 'position' in col:
                valid_positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].apply(
                    lambda x: x if x in valid_positions else 'Unknown'
                )
            else:
                df_clean[col] = df_clean[col].astype(str).replace(['nan', 'None'], 'Unknown')
    return df_clean

df_final_clean = clean_categorical_features(df_final, categorical_features)

# Prepare training data
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

# Create preprocessor
preprocessor = ColumnTransformer([
    ('num', RobustScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
])

# Fit and transform
X_transformed = preprocessor.fit_transform(X_clean)
print(f"Transformed features: {X_transformed.shape}")

# Train/validation split
df_with_indices = df_final_clean[valid_mask].copy()
df_with_indices['original_index'] = range(len(df_with_indices))

df_sorted = df_with_indices.sort_values(['season', 'week'])

split_idx = int(0.8 * len(df_sorted))
train_indices = df_sorted.iloc[:split_idx]['original_index'].values
val_indices = df_sorted.iloc[split_idx:]['original_index'].values

X_train = X_transformed[train_indices]
X_val = X_transformed[val_indices]
y_train = y_clean[train_indices]
y_val = y_clean[val_indices]

print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")

def create_enhanced_model(input_dim, output_dim):
    """Create model with reduced regularization for more prediction variance"""
    
    inputs = Input(shape=(input_dim,))
    
    x = Dense(512, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.15)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    
    outputs = Dense(output_dim, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create and compile model
input_dim = X_train.shape[1]
output_dim = len(target_cols)
model = create_enhanced_model(input_dim, output_dim)

model.compile(
    optimizer=Adam(learning_rate=0.0015, beta_1=0.9, beta_2=0.999),
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
    
    player_history = historical_df[
        (historical_df['player_id'] == player_id) |
        (historical_df['player_display_name'].str.contains(player_name, case=False, na=False)) |
        (historical_df['player_name'].str.contains(player_name, case=False, na=False))
    ]
    
    if player_history.empty:
        return True
    
    seasons_played = player_history['season'].unique()
    
    if all(season >= current_season for season in seasons_played):
        return True
    
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
        offensive_positions = ['QB', 'RB', 'WR', 'TE', 'FB']
        
        self.offensive_depth = self.depth_charts[
            self.depth_charts['pos_abb'].isin(offensive_positions)
        ].copy()
        
        self.offensive_depth['depth_role'] = self.offensive_depth['pos_rank'].apply(
            lambda x: 'starter' if x == 1 else 'backup' if x == 2 else 'deep_backup'
        )
        
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
        if player_name in self.player_roles:
            return self.player_roles[player_name]
        
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
        
        if player_role['role'] == 'starter':
            opportunity_analysis['opportunity'] = 'high'
            opportunity_analysis['context'] = f"Listed as starter on {team} depth chart"
        elif player_role['role'] == 'backup' and position == 'RB':
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
            last_name_match = name1_parts[-1] == name2_parts[-1]
            first_name_match = (
                name1_parts[0] == name2_parts[0] or
                name1_parts[0][0] == name2_parts[0][0]
            )
            
            return last_name_match and first_name_match
        
        return False

# ===== INJURY STATUS ANALYZER WITH SPORTRADAR DATA =====
class InjuryStatusAnalyzer:
    """Analyze player injury status using Sportradar API data"""
    
    def __init__(self, injury_overrides, backup_situations):
        self.injury_overrides = injury_overrides
        self.backup_situations = backup_situations
        # Create normalized name lookups
        self._create_normalized_lookups()
    
    def _create_normalized_lookups(self):
        """Create normalized name lookups for better matching"""
        self.injury_by_normalized = {}
        for name, info in self.injury_overrides.items():
            normalized = self._normalize_name(name)
            self.injury_by_normalized[normalized] = (name, info)
        
        self.backup_by_normalized = {}
        for name, info in self.backup_situations.items():
            normalized = self._normalize_name(name)
            self.backup_by_normalized[normalized] = (name, info)
    
    def _normalize_name(self, name):
        """Normalize a name for comparison"""
        if not name:
            return ""
        # Remove suffixes, convert to lowercase, remove extra spaces
        name_clean = name.lower().strip()
        # Remove common suffixes
        for suffix in [' jr.', ' sr.', ' iii', ' ii', ' iv', ' jr', ' sr']:
            name_clean = name_clean.replace(suffix, '')
        return ' '.join(name_clean.split())
    
    def check_player_status(self, player_name):
        """Check if player has injury/status issues with fuzzy matching"""
        # Direct match first
        if player_name in self.injury_overrides:
            return self.injury_overrides[player_name]
        
        if player_name in self.backup_situations:
            return self.backup_situations[player_name]
        
        # Try normalized matching
        normalized = self._normalize_name(player_name)
        
        if normalized in self.injury_by_normalized:
            original_name, info = self.injury_by_normalized[normalized]
            return info
        
        if normalized in self.backup_by_normalized:
            original_name, info = self.backup_by_normalized[normalized]
            return info
        
        # Try partial matching on last name
        last_name = normalized.split()[-1] if normalized else ""
        if last_name and len(last_name) > 3:
            for norm_key, (orig_name, info) in self.injury_by_normalized.items():
                if norm_key.endswith(last_name):
                    # Check first initial matches
                    if normalized[0] == norm_key[0]:
                        return info
            
            for norm_key, (orig_name, info) in self.backup_by_normalized.items():
                if norm_key.endswith(last_name):
                    if normalized[0] == norm_key[0]:
                        return info
        
        return None
    
    def should_zero_out_player(self, player_name):
        """Determine if player should get 0 points due to injury"""
        status_info = self.check_player_status(player_name)
        if status_info and 'status' in status_info:
            return status_info['status'] in ['OUT', 'DOUBTFUL']
        return False
    
    def should_boost_backup(self, player_name):
        """Determine if backup should get starter-level projections"""
        status_info = self.check_player_status(player_name)
        # Check if this player is in backup_situations
        if player_name in self.backup_situations:
            return True
        
        # Try normalized matching
        normalized = self._normalize_name(player_name)
        if normalized in self.backup_by_normalized:
            return True
        
        return False
    
    def get_adjustment_info(self, player_name):
        """Get detailed adjustment information"""
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
    
# Initialize the injury analyzer with Sportradar data
logger.info("\n=== INITIALIZING INJURY STATUS ANALYZER ===")
injury_analyzer = InjuryStatusAnalyzer(injury_overrides, backup_situations)

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
        
        draft_info = self.get_draft_info(player_name)
        if draft_info:
            pick = draft_info['draft_position']
            if pick <= 10:
                draft_multiplier = 1.6
                tier = 'elite'
            elif pick <= 32:
                draft_multiplier = 1.3
                tier = 'high'
            elif pick <= 64:
                draft_multiplier = 1.0
                tier = 'medium'
            elif pick <= 100:
                draft_multiplier = 0.8
                tier = 'late'
            else:
                draft_multiplier = 0.6
                tier = 'very_late'
        else:
            draft_multiplier = 0.4
            tier = 'undrafted'
        
        opportunity_analysis = self.depth_analyzer.analyze_rookie_opportunity(
            player_name, team, position
        )
        
        if opportunity_analysis['opportunity'] == 'high':
            depth_multiplier = 1.5
        elif opportunity_analysis['opportunity'] == 'medium':
            depth_multiplier = 0.8
        elif opportunity_analysis['opportunity'] == 'low':
            depth_multiplier = 0.3
        else:
            depth_multiplier = 0.1
        
        if position == 'QB':
            if opportunity_analysis['opportunity'] == 'high':
                position_multiplier = 1.0
            else:
                position_multiplier = 0.2
        elif position == 'RB':
            position_multiplier = 1.2
        elif position in ['WR', 'TE']:
            if tier in ['elite', 'high']:
                position_multiplier = 1.0
            else:
                position_multiplier = 0.7
        else:
            position_multiplier = 1.0
        
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

# ===== COMPREHENSIVE PREDICTION WITH INJURY INTEGRATION =====

def comprehensive_predict_week_enhanced(player_name, week):
    """Enhanced prediction with injury status integration"""
    try:
        player_matches = latest_rosters[
            latest_rosters['player_name'].str.contains(player_name, case=False, na=False)
        ]
        
        if player_matches.empty:
            return None
        
        player_info = player_matches.iloc[0]
        position = player_info['position']
        team = player_info.get('team', 'Unknown')
        player_id = player_info.get('player_id', '')
        
        # CHECK INJURY STATUS FIRST
        injury_status = injury_analyzer.check_player_status(player_name)
        
        if injury_analyzer.should_zero_out_player(player_name):
            adjustment_info = injury_analyzer.get_adjustment_info(player_name)
            
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
        
        backup_boost = injury_analyzer.should_boost_backup(player_name)
        
        is_rookie = is_actually_rookie(player_name, player_id, df_final_clean, year)
        
        if is_rookie:
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
        
        original_points = result['fanduel_fantasy_points']
        adjustment = "no adjustment"
        
        if backup_boost:
            adjustment_info = injury_analyzer.get_adjustment_info(player_name)
            replacing = adjustment_info.get('replacing', 'injured starter')
            
            # Get the backup player's historical production
            avg_fppg = recent_stats.get('avg_fppg', 0) if 'avg_fppg' in recent_stats.index else 0
            games_played = len(player_data[player_data['week'] != 'AVG'])
            
            # Only boost if player has some track record OR is a high-ranked backup
            depth_role = enhanced_depth_analyzer.get_player_role(player_name)
            is_second_string = depth_role and depth_role.get('depth_rank', 99) == 2
            
            # Tiered boost based on player's history
            if games_played >= 5 and avg_fppg >= 5.0:
                # Proven backup with decent production
                if position == 'QB':
                    result['fanduel_fantasy_points'] *= 2.5
                    adjustment = f"proven backup QB replacing {replacing}"
                else:
                    result['fanduel_fantasy_points'] *= 1.8
                    adjustment = f"proven backup replacing {replacing}"
            elif is_second_string and games_played >= 3:
                # Listed #2 on depth chart with some experience
                if position == 'QB':
                    result['fanduel_fantasy_points'] *= 2.0
                    adjustment = f"backup QB replacing {replacing}"
                else:
                    result['fanduel_fantasy_points'] *= 1.4
                    adjustment = f"backup replacing {replacing}"
            elif result['fanduel_fantasy_points'] < 3.0:
                # Deep backup with no history - give them a baseline floor instead of boost
                if position == 'QB':
                    result['fanduel_fantasy_points'] = 12.0  # QB floor
                    adjustment = f"emergency QB replacing {replacing} (baseline)"
                elif position in ['RB', 'WR']:
                    result['fanduel_fantasy_points'] = 6.0  # Skill position floor
                    adjustment = f"emergency starter replacing {replacing} (baseline)"
                else:  # TE
                    result['fanduel_fantasy_points'] = 4.0  # TE floor
                    adjustment = f"emergency TE replacing {replacing} (baseline)"
            else:
                # Has minimal production, apply conservative boost
                result['fanduel_fantasy_points'] *= 1.3
                adjustment = f"backup replacing {replacing} (limited history)"
        
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
                    if position == 'QB':
                        elite_qbs = ['Josh Allen', 'Lamar Jackson', 'Patrick Mahomes']
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
    
def get_comprehensive_predictions(position, week):
    """Generate predictions for all players at a position"""
    predictions = []
    
    # GET ALL PLAYERS, not just active ones - we need to check injuries for everyone
    all_position_players = latest_rosters[
        latest_rosters['position'] == position
    ]
    
    print(f"\nGenerating predictions for {len(all_position_players)} {position}s...")
    print(f"  Status breakdown: {all_position_players['status'].value_counts().to_dict()}")
    
    veteran_count = 0
    rookie_count = 0
    injured_count = 0
    backup_elevated_count = 0
    processed_players = set()
    
    for _, player in tqdm(all_position_players.iterrows(), 
                         total=len(all_position_players), 
                         desc=f"Predicting {position}"):
        
        player_name = player['player_name']
        
        if player_name in processed_players:
            continue
        
        # CHECK INJURY STATUS BEFORE ANYTHING ELSE
        if injury_analyzer.should_zero_out_player(player_name):
            adjustment_info = injury_analyzer.get_adjustment_info(player_name)
            
            processed_players.add(player_name)
            predictions.append({
                'player_name': player_name,
                'position': position,
                'team': player.get('team', 'Unknown'),
                'fanduel_fantasy_points': 0.0,
                'prediction_type': 'injured_out',
                'injury_status': 'OUT',
                'injury_reason': adjustment_info['reason'],
                'role_adjustment': adjustment_info['reason']
            })
            injured_count += 1
            continue
        
        # Skip inactive players who are NOT injured (practice squad, etc)
        if player['status'] != 'ACT' and not injury_analyzer.should_boost_backup(player_name):
            continue
        
        pred = comprehensive_predict_week_enhanced(player_name, week)
        if pred:
            processed_players.add(player_name)
            predictions.append(pred)
            
            pred_type = pred.get('prediction_type', '')
            if 'injured' in pred_type:
                injured_count += 1
            elif pred.get('injury_replacement'):
                backup_elevated_count += 1
            elif pred_type.startswith('rookie'):
                rookie_count += 1
            else:
                veteran_count += 1
    
    if predictions:
        predictions_df = pd.DataFrame(predictions)
        
        print(f"Before duplicate removal: {len(predictions_df)} players")
        predictions_df = predictions_df.drop_duplicates(subset=['player_name'], keep='first')
        
        predictions_df = predictions_df.sort_values('fanduel_fantasy_points', ascending=False)
        predictions_df = predictions_df.reset_index(drop=True)
        predictions_df['rank'] = range(1, len(predictions_df) + 1)
        predictions_df['prediction_rank'] = predictions_df['rank']
        
        os.makedirs('predictions', exist_ok=True)
        filename = f'predictions/{position}_predictions_week{week}_season{year}.csv'
        predictions_df.to_csv(filename, index=False)
        
        print(f"\n{position} Summary:")
        print(f"  Total: {len(predictions_df)} players")
        print(f"  Veterans (ML): {veteran_count}")
        print(f"  Rookies: {rookie_count}")
        print(f"  Injured (zeroed): {injured_count}")
        print(f"  Backups elevated: {backup_elevated_count}")
        print(f"  Saved: {filename}")
        
        return predictions_df
    
    return pd.DataFrame()

# Generate predictions for all positions
positions = ['QB', 'RB', 'WR', 'TE']

print("\n" + "="*80)
print(f"PREDICTIONS WITH INJURY INTEGRATION - WEEK {week_to_predict}")
print("="*80)

start_time = time.time()

for position in positions:
    print(f"\n{'='*20} {position} PREDICTIONS {'='*20}")
    predictions_df = get_comprehensive_predictions(position, week_to_predict)
    
    if not predictions_df.empty:
        print(f"\nTop 10 {position} predictions:")
        
        injured = predictions_df[predictions_df['prediction_type'].str.contains('injured', na=False)]
        if len(injured) > 0:
            print(f"\nInjured {position}s (0 points):")
            for _, player in injured.head(5).iterrows():
                print(f"  {player['player_name']}: {player['injury_status']} - {player['injury_reason']}")
        
        backups = predictions_df[predictions_df.get('injury_replacement', False) == True]
        if len(backups) > 0:
            print(f"\nElevated Backups:")
            for _, player in backups.head(5).iterrows():
                original = player.get('original_prediction', 0)
                print(f"  {player['player_name']}: {player['fanduel_fantasy_points']} pts (boosted from {original})")
        
        healthy = predictions_df[~predictions_df['prediction_type'].str.contains('injured', na=False)]
        print(f"\nTop Healthy Players:")
        for _, player in healthy.head(10).iterrows():
            name = player['player_name']
            points = player['fanduel_fantasy_points']
            pred_type = player.get('prediction_type', 'unknown')
            
            print(f"  {player['rank']:2d}. {name:20s}: {points:5.1f} pts [{pred_type}]")

end_time = time.time()
print(f"\nTotal prediction time: {end_time - start_time:.1f} seconds")

print(f"\nPREDICTION SYSTEM WITH INJURY INTEGRATION COMPLETE!")
print(f"Features:")
print(f"  Real-time injury data from Sportradar API")
print(f"  Injured players automatically zeroed out")
print(f"  Backup players elevated with appropriate boosts")
print(f"  ML predictions with {len(numerical_features)} features")
print(f"  Depth chart integration")

print(f"\nCheck predictions/ folder for CSV files")