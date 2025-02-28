import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from tqdm import tqdm

def load_data(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    print(f"Shape of the dataset: {df.shape}")
    print(f"Columns in the dataset: {df.columns.tolist()}")
    
    # Check for empty columns
    empty_columns = df.columns[df.isna().all()].tolist()
    if empty_columns:
        print("\nWarning: The following columns are entirely empty:")
        for col in empty_columns:
            print(f"  - {col}")
    
    # Check data types
    print("\nColumn data types:")
    print(df.dtypes)
    
    return df

def analyze_predictions(df, position, metrics):
    print(f"\nAnalysis for {position}:")
    position_df = df[df['position'] == position]
    print(f"Number of {position} entries: {len(position_df)}")
    
    for metric in metrics:
        actual = position_df[metric]
        predicted = position_df[f'pred_{metric}']
        
        print(f"\n  {metric}:")
        print(f"    Actual - min: {actual.min()}, max: {actual.max()}, mean: {actual.mean():.2f}")
        print(f"    Predicted - min: {predicted.min()}, max: {predicted.max()}, mean: {predicted.mean():.2f}")
        
        # Check for missing values
        missing_actual = actual.isnull().sum()
        missing_predicted = predicted.isnull().sum()
        print(f"    Missing values - Actual: {missing_actual}, Predicted: {missing_predicted}")
        
        # Remove rows with NaN values
        valid_data = position_df.dropna(subset=[metric, f'pred_{metric}'])
        
        if len(valid_data) > 0:
            actual = valid_data[metric]
            predicted = valid_data[f'pred_{metric}']
            
            # Print some sample data
            print(f"    Sample data (first 5 rows):")
            print(valid_data[[metric, f'pred_{metric}']].head())
            
            try:
                mse = mean_squared_error(actual, predicted)
                mae = mean_absolute_error(actual, predicted)
                print(f"    Mean Squared Error: {mse}")
                print(f"    Mean Absolute Error: {mae}")
            except Exception as e:
                print(f"    Error calculating metrics: {str(e)}")
        else:
            print("    No valid data for analysis after removing NaN values.")

def generate_plots(df, position_metrics):
    print("Generating weekly plots for all positions and metrics")
    
    # Ensure 'week' column exists
    if 'week' not in df.columns:
        print("Error: 'week' column not found in the dataset")
        return

    # Group data by week
    grouped = df.groupby('week')

    for week, week_data in tqdm(grouped, desc="Processing weeks"):
        for position, metrics in position_metrics.items():
            position_data = week_data[week_data['position'] == position]
            
            for metric in metrics:
                plt.figure(figsize=(12, 6))
                
                actual = position_data[metric]
                predicted = position_data[f'pred_{metric}']
                players = position_data['player_name']  # Assuming there's a 'player_name' column
                
                x = range(len(players))
                plt.bar(x, actual, width=0.4, align='center', label='Actual', alpha=0.8)
                plt.bar([i + 0.4 for i in x], predicted, width=0.4, align='center', label='Predicted', alpha=0.8)
                
                plt.title(f'{position} - {metric} (Week {week})')
                plt.xlabel('Players')
                plt.ylabel(metric)
                plt.legend()
                
                plt.xticks([i + 0.2 for i in x], players, rotation=90, ha='right')
                plt.tight_layout()
                
                plt.savefig(f'predictions/plots/week_{week}_{position}_{metric}_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()

def main():
    # Load the dataset
    df = load_data('data/nfl_dataset_with_predictions_2024.csv')

    # Define metrics for each position
    position_metrics = {
        'QB': ['passing_yards', 'attempts', 'completions', 'passing_tds', 'interceptions', 'rushing_yards', 'rushing_tds', 'fanduel_fantasy_points'],
        'RB': ['rushing_yards', 'carries', 'rushing_tds', 'receiving_yards', 'receptions', 'receiving_tds', 'fanduel_fantasy_points'],
        'WR': ['receiving_yards', 'targets', 'receptions', 'receiving_tds', 'fanduel_fantasy_points'],
        'TE': ['receiving_yards', 'targets', 'receptions', 'receiving_tds', 'fanduel_fantasy_points']
    }

    # Analyze predictions for each position
    for position, metrics in position_metrics.items():
        analyze_predictions(df, position, metrics)

    # Generate plots for all positions and metrics
    generate_plots(df, position_metrics)

if __name__ == "__main__":
    main()