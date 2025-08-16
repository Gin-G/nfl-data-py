import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm

def get_available_weeks(predictions_directory):
    """Get available weeks from prediction files"""
    weeks = set()
    for file in os.listdir(predictions_directory):
        match = re.search(r'week(\d+)_season2025\.csv$', file)  # Updated for 2025
        if match:
            week = int(match.group(1))
            if 1 <= week <= 18:  # Full season range
                weeks.add(week)
    return sorted(list(weeks))

def analyze_prediction_accuracy(df_merged, target_cols):
    """Analyze prediction accuracy using the new averaging features"""
    print("\n=== PREDICTION ACCURACY ANALYSIS ===")
    
    # Calculate errors for each stat
    for col in target_cols:
        actual_col = col
        pred_col = f'pred_{col}'
        
        if actual_col in df_merged.columns and pred_col in df_merged.columns:
            # Remove rows with missing values
            valid_data = df_merged.dropna(subset=[actual_col, pred_col])
            
            if len(valid_data) > 0:
                # Calculate basic error metrics
                mae = np.mean(np.abs(valid_data[actual_col] - valid_data[pred_col]))
                rmse = np.sqrt(np.mean((valid_data[actual_col] - valid_data[pred_col])**2))
                
                # Calculate accuracy relative to baseline (avg_fppg for fantasy points)
                if col == 'fanduel_fantasy_points' and 'avg_fppg' in valid_data.columns:
                    baseline_mae = np.mean(np.abs(valid_data[actual_col] - valid_data['avg_fppg']))
                    improvement = (baseline_mae - mae) / baseline_mae * 100
                    print(f"{col:25s}: MAE={mae:6.2f}, RMSE={rmse:6.2f}, vs avg_fppg: {improvement:+5.1f}%")
                else:
                    print(f"{col:25s}: MAE={mae:6.2f}, RMSE={rmse:6.2f}")

def merge_predictions_with_enhanced_dataset(original_dataset_path, predictions_directory, output_path):
    """
    Enhanced merge function that utilizes new averaging features
    
    Args:
    original_dataset_path (str): Path to the enhanced NFL dataset CSV file.
    predictions_directory (str): Path to the directory containing prediction CSV files.
    output_path (str): Path where the merged dataset will be saved.
    
    Returns:
    pd.DataFrame: The merged dataset with predictions and analysis.
    """
    # Get available weeks from prediction files
    available_weeks = get_available_weeks(predictions_directory)
    if not available_weeks:
        print("No valid prediction files found.")
        return None
    
    print(f"Found prediction files for weeks: {available_weeks}")

    # Load the enhanced dataset
    print("Loading enhanced dataset...")
    df_original = pd.read_csv(original_dataset_path)
    
    print(f"Original dataset shape: {df_original.shape}")
    print(f"Available columns: {df_original.columns.tolist()}")
    
    # Check for new averaging features
    averaging_features = [col for col in df_original.columns if 'avg' in col.lower() or col in ['games_played']]
    if averaging_features:
        print(f"Detected averaging features: {averaging_features}")
    
    # Filter for specified season and available weeks (exclude season average rows)
    season_filter = 2025  # Updated for current season
    df_filtered = df_original[
        (df_original['season'] == season_filter) & 
        (df_original['week'] != 'AVG') &  # Exclude season average rows
        (df_original['week'].isin(available_weeks))
    ]
    
    if df_filtered.empty:
        print(f"No data available for season {season_filter} and weeks {available_weeks}")
        return None
    
    print(f"Filtered dataset shape: {df_filtered.shape}")
    
    # Initialize prediction storage
    all_predictions = {}
    
    # Process prediction files
    print("Loading prediction files...")
    for week in tqdm(available_weeks, desc="Processing prediction files"):
        for position in ['QB', 'RB', 'WR', 'TE']:
            # Try multiple file naming patterns
            file_patterns = [
                f"{position}_predictions_week{week}_season{season_filter}.csv",
                f"{position}_improved_predictions_week{week}_season{season_filter}.csv",
                f"{position}_enhanced_predictions_week{week}_season{season_filter}.csv",
                f"{position}_robust_enhanced_predictions_week{week}_season{season_filter}.csv"
            ]
            
            file_found = False
            for pattern in file_patterns:
                file_path = os.path.join(predictions_directory, pattern)
                
                if os.path.exists(file_path):
                    try:
                        df_pred = pd.read_csv(file_path)
                        
                        # Create unique keys for matching
                        df_pred['match_key'] = (df_pred['player_name'].str.lower().str.strip() + 
                                              '_' + str(week))
                        
                        # Store predictions with enhanced information
                        for _, row in df_pred.iterrows():
                            key = row['match_key']
                            pred_data = row.to_dict()
                            pred_data['prediction_file'] = pattern
                            pred_data['prediction_rank'] = row.get('rank', None)
                            all_predictions[key] = pred_data
                        
                        file_found = True
                        break
                        
                    except Exception as e:
                        print(f"Error processing {pattern}: {str(e)}")
            
            if not file_found:
                print(f"Warning: No prediction file found for {position} week {week}")
    
    if not all_predictions:
        print("No valid prediction data found.")
        return None
    
    print(f"Loaded predictions for {len(all_predictions)} player-week combinations")
    
    # Enhanced matching function
    def get_enhanced_predictions(row):
        """Get predictions with enhanced matching and analysis"""
        
        # Try multiple matching strategies
        player_name = str(row['player_name']).lower().strip()
        display_name = str(row.get('player_display_name', '')).lower().strip()
        week = row['week']
        
        match_keys = [
            f"{player_name}_{week}",
            f"{display_name}_{week}"
        ]
        
        # Find the best match
        pred_data = None
        for key in match_keys:
            if key in all_predictions:
                pred_data = all_predictions[key]
                break
        
        if pred_data is None:
            # Return empty predictions
            result = {f'pred_{col}': None for col in target_cols}
            result.update({
                'prediction_found': False,
                'prediction_file': None,
                'prediction_rank': None,
                'pred_vs_avg': None,
                'pred_vs_baseline': None
            })
            return pd.Series(result)
        
        # Extract prediction values
        result = {}
        for col in target_cols:
            result[f'pred_{col}'] = pred_data.get(col, None)
        
        # Add metadata
        result['prediction_found'] = True
        result['prediction_file'] = pred_data.get('prediction_file', None)
        result['prediction_rank'] = pred_data.get('rank', None)
        
        # Enhanced analysis using averaging features
        if 'avg_fppg' in row.index and 'fanduel_fantasy_points' in pred_data:
            pred_points = pred_data.get('fanduel_fantasy_points', 0)
            avg_points = row['avg_fppg']
            
            if pd.notna(avg_points) and avg_points > 0:
                result['pred_vs_avg'] = pred_points - avg_points
                result['pred_vs_baseline'] = (pred_points / avg_points - 1) * 100
            else:
                result['pred_vs_avg'] = None
                result['pred_vs_baseline'] = None
        else:
            result['pred_vs_avg'] = None
            result['pred_vs_baseline'] = None
        
        # Add confidence scores if available from predictions
        result['prediction_confidence'] = pred_data.get('confidence_score', None)
        result['predicted_vs_avg'] = pred_data.get('predicted_vs_avg', None)
        
        return pd.Series(result)
    
    # Apply enhanced prediction matching
    print("Merging predictions with enhanced analysis...")
    df_merged = df_filtered.copy()
    
    tqdm.pandas(desc="Applying enhanced predictions")
    pred_results = df_merged.progress_apply(get_enhanced_predictions, axis=1)
    
    # Concatenate results
    df_merged = pd.concat([df_merged, pred_results], axis=1)
    
    # Add derived analysis columns
    print("Adding derived analysis columns...")
    
    # Prediction accuracy metrics
    for col in ['fanduel_fantasy_points', 'passing_yards', 'rushing_yards', 'receiving_yards']:
        if col in df_merged.columns and f'pred_{col}' in df_merged.columns:
            # Absolute error
            df_merged[f'{col}_abs_error'] = np.abs(
                df_merged[col] - df_merged[f'pred_{col}']
            )
            
            # Percentage error
            df_merged[f'{col}_pct_error'] = np.where(
                df_merged[col] > 0,
                (df_merged[f'pred_{col}'] - df_merged[col]) / df_merged[col] * 100,
                None
            )
    
    # Enhanced performance categories using avg_fppg
    if 'avg_fppg' in df_merged.columns and 'fanduel_fantasy_points' in df_merged.columns:
        df_merged['actual_vs_avg_category'] = pd.cut(
            df_merged['fanduel_fantasy_points'] - df_merged['avg_fppg'],
            bins=[-np.inf, -5, -2, 2, 5, np.inf],
            labels=['Well Below Avg', 'Below Avg', 'Average', 'Above Avg', 'Well Above Avg']
        )
        
        if 'pred_fanduel_fantasy_points' in df_merged.columns:
            df_merged['pred_vs_avg_category'] = pd.cut(
                df_merged['pred_fanduel_fantasy_points'] - df_merged['avg_fppg'],
                bins=[-np.inf, -5, -2, 2, 5, np.inf],
                labels=['Well Below Avg', 'Below Avg', 'Average', 'Above Avg', 'Well Above Avg']
            )
    
    # Save the enhanced merged dataset
    print(f"Saving enhanced merged dataset to {output_path}")
    df_merged.to_csv(output_path, index=False)
    
    # Generate analysis report
    analyze_prediction_accuracy(df_merged, target_cols)
    
    # Summary statistics
    print(f"\n=== MERGE SUMMARY ===")
    print(f"Total rows in merged dataset: {len(df_merged)}")
    print(f"Predictions found: {df_merged['prediction_found'].sum():,}")
    print(f"Prediction rate: {df_merged['prediction_found'].mean()*100:.1f}%")
    
    if 'prediction_rank' in df_merged.columns:
        print(f"Average prediction rank: {df_merged['prediction_rank'].mean():.1f}")
    
    # Week distribution
    print(f"\nWeek distribution:")
    print(df_merged['week'].value_counts().sort_index())
    
    # Position distribution (if predictions found)
    if 'position' in df_merged.columns and df_merged['prediction_found'].any():
        print(f"\nPosition distribution (with predictions):")
        pos_dist = df_merged[df_merged['prediction_found']]['position'].value_counts()
        print(pos_dist)
    
    print("\nEnhanced merge complete!")
    return df_merged

# Enhanced target columns to match your current model
target_cols = [
    'passing_yards', 'passing_tds', 'interceptions',
    'rushing_yards', 'rushing_tds', 
    'receiving_yards', 'receptions', 'receiving_tds',
    'fanduel_fantasy_points'
]

# Alternative: Use all original target columns if available
extended_target_cols = [
    'passing_yards', 'attempts', 'completions', 'passing_tds', 'interceptions', 'sack_yards', 'sack_fumbles', 'sack_fumbles_lost',
    'rushing_yards', 'carries', 'rushing_tds', 'rushing_fumbles', 'rushing_fumbles_lost',
    'receiving_yards', 'targets', 'receptions', 'receiving_tds', 'receiving_fumbles', 'receiving_fumbles_lost', 
    'offensive_snap_pct', 'fanduel_fantasy_points'
]

if __name__ == "__main__":
    # Run the enhanced merge
    df_merged = merge_predictions_with_enhanced_dataset(
        'data/nfl_dataset.csv',  # Your enhanced dataset
        'predictions',           # Prediction files directory
        'data/nfl_enhanced_dataset_with_predictions_2025.csv'  # Output file
    )
    
    if df_merged is not None:
        print("\n=== ENHANCED ANALYSIS SAMPLE ===")
        
        # Show sample of enhanced data
        enhanced_cols = [
            'player_name', 'position', 'week', 'fanduel_fantasy_points', 
            'avg_fppg', 'pred_fanduel_fantasy_points', 'pred_vs_avg', 
            'prediction_rank', 'prediction_found'
        ]
        
        available_cols = [col for col in enhanced_cols if col in df_merged.columns]
        sample_data = df_merged[df_merged['prediction_found'] == True][available_cols].head(10)
        
        if not sample_data.empty:
            print("\nSample of enhanced merged data:")
            print(sample_data.to_string(index=False))
        
        # Show performance by position using averaging features
        if 'avg_fppg' in df_merged.columns and 'pred_fanduel_fantasy_points' in df_merged.columns:
            print("\n=== POSITION-LEVEL ANALYSIS ===")
            position_analysis = df_merged[df_merged['prediction_found']].groupby('position').agg({
                'fanduel_fantasy_points': 'mean',
                'avg_fppg': 'mean', 
                'pred_fanduel_fantasy_points': 'mean',
                'pred_vs_avg': 'mean',
                'prediction_rank': 'mean'
            }).round(2)
            
            print("Average performance by position:")
            print(position_analysis)
    else:
        print("Enhanced merging process failed or no data was available.")