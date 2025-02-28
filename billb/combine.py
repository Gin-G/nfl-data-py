import pandas as pd
import os
import re
from tqdm import tqdm

def get_available_weeks(predictions_directory):
    weeks = set()
    for file in os.listdir(predictions_directory):
        match = re.search(r'week(\d+)_season2024\.csv$', file)
        if match:
            week = int(match.group(1))
            if 1 <= week <= 7:  # Only include weeks 1 through 6
                weeks.add(week)
    return sorted(list(weeks))

def merge_predictions_with_dataset(original_dataset_path, predictions_directory, output_path):
    """
    Merge prediction files with the original dataset for the 2024 season, weeks 1-6.
    
    Args:
    original_dataset_path (str): Path to the original NFL dataset CSV file.
    predictions_directory (str): Path to the directory containing prediction CSV files.
    output_path (str): Path where the merged dataset will be saved.
    
    Returns:
    pd.DataFrame: The merged dataset with predictions.
    """
    # Get available weeks from prediction files
    available_weeks = get_available_weeks(predictions_directory)
    if not available_weeks:
        print("No valid prediction files found for weeks 1-7 of the 2024 season.")
        return None
    
    print(f"Found prediction files for weeks: {available_weeks}")

    # Load the original dataset
    print("Loading original dataset...")
    df_original = pd.read_csv(original_dataset_path)
    
    # Filter for 2024 season and available weeks
    df_original = df_original[(df_original['season'] == 2024) & (df_original['week'].isin(available_weeks))]
    
    if df_original.empty:
        print("No data available in the original dataset for the selected weeks of the 2024 season.")
        return None
    
    # Initialize a dictionary to store predictions
    all_predictions = {}
    
    # Process prediction files for available weeks
    for week in tqdm(available_weeks, desc="Processing prediction files"):
        for position in ['QB', 'RB', 'WR', 'TE']:
            file_name = f"{position}_predictions_week{week}_season2024.csv"
            file_path = os.path.join(predictions_directory, file_name)
            
            if os.path.exists(file_path):
                try:
                    df_pred = pd.read_csv(file_path)
                    
                    # Create a unique key for each player-week combination
                    df_pred['key'] = df_pred['player_name'].str.lower() + '_' + str(week)
                    
                    # Store predictions in the dictionary
                    all_predictions.update(df_pred.set_index('key').to_dict('index'))
                except Exception as e:
                    print(f"Error processing file {file_name}: {str(e)}")
            else:
                print(f"Warning: Missing prediction file for {position} in week {week}")
    
    if not all_predictions:
        print("No valid prediction data found.")
        return None
    
    print("Merging predictions with original dataset...")
    # Function to get predictions for a row
    def get_predictions(row):
        key = f"{row['player_display_name'].lower()}_{row['week']}"
        pred = all_predictions.get(key, {})
        return pd.Series({f'pred_{col}': pred.get(col, None) for col in target_cols})
    
    # Apply the function to each row of the original dataset
    df_merged = df_original.copy()
    tqdm.pandas(desc="Applying predictions")
    pred_columns = df_merged.progress_apply(get_predictions, axis=1)
    
    # Concatenate the original dataset with the prediction columns
    df_merged = pd.concat([df_merged, pred_columns], axis=1)
    
    # Save the merged dataset
    print(f"Saving merged dataset to {output_path}")
    df_merged.to_csv(output_path, index=False)
    
    print("Merge complete!")
    return df_merged

# List of target columns (make sure this matches your prediction columns)
target_cols = [
    'passing_yards', 'attempts', 'completions', 'passing_tds', 'interceptions', 'sack_yards', 'sack_fumbles', 'sack_fumbles_lost',
    'rushing_yards', 'carries', 'rushing_tds', 'rushing_fumbles', 'rushing_fumbles_lost',
    'receiving_yards', 'targets', 'receptions', 'receiving_tds', 'receiving_fumbles', 'receiving_fumbles_lost', 
    'offensive_snap_pct', 'fanduel_fantasy_points'
]

# Example usage:
df_merged = merge_predictions_with_dataset(
    'data/nfl_dataset.csv',
    'predictions',
    'data/nfl_dataset_with_predictions_2024.csv'
)

if df_merged is not None:
    # Add some debug information
    print("\nDebug Information:")
    print(f"Total rows in merged dataset: {len(df_merged)}")
    print("Sample of merged data:")
    print(df_merged[['player_display_name', 'season', 'week'] + [f'pred_{col}' for col in target_cols]].head())
    print("\nNull counts in prediction columns:")
    print(df_merged[[f'pred_{col}' for col in target_cols]].isnull().sum())
    print("\nDistribution of weeks:")
    print(df_merged['week'].value_counts().sort_index())
else:
    print("Merging process failed or no data was available.")