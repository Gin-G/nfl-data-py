# Configuration variables

# Seasons to consider for data
SEASONS = list(range(2018, 2024))

# Target columns for prediction
TARGET_COLS = [
    'passing_yards', 'attempts', 'completions', 
    'rushing_yards', 'carries', 
    'receiving_yards', 'targets', 'receptions'
]

# Categorical features for model input
CATEGORICAL_FEATURES = ['position_x', 'recent_team']  

# Numerical features for model input
NUMERICAL_FEATURES = [
    'passing_yards', 'attempts', 'completions', 
    'rushing_yards', 'carries', 
    'receiving_yards', 'targets', 'receptions'
] 

# Model parameters
MODEL_PARAMS = {
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.2
}

# File paths
DATA_DIR = 'data/'
MODEL_DIR = 'saved_models/'
PREDICTIONS_DIR = 'predictions/'

# Plotting parameters
PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']