import pandas as pd
from data_loader import load_and_preprocess_data, update_dataset
from models import create_and_train_models, load_models, update_models, save_models
from predictions import get_all_predictions, store_predictions, compare_predictions_to_actual
from plots import plot_top_predictions, plot_prediction_vs_actual, plot_weekly_performance
from config import SEASONS, TARGET_COLS
import os

def main():
    # Initial setup
    current_season = 2023
    current_week = 1

    # Load or create initial dataset
    if os.path.exists('data/nfl_dataset.csv'):
        df = pd.read_csv('data/nfl_dataset.csv')
    else:
        df = load_and_preprocess_data(SEASONS)
        df.to_csv('data/nfl_dataset.csv', index=False)

    # Load or train models
    if os.path.exists('saved_models/model_qb.h5'):
        models, preprocessor = load_models()
    else:
        models, preprocessor = create_and_train_models(df)
        save_models(models, preprocessor)

    # Main loop for weekly predictions
    while current_week <= 18:  # Assuming a 18-week season
        print(f"\nProcessing Week {current_week}")

        # Make predictions for each position
        for position in ['QB', 'RB', 'WR', 'TE']:
            print(f"Making predictions for {position}")
            predictions = get_all_predictions(position, df, models, preprocessor, current_week, current_season)
            print(f"Number of predictions for {position}: {len(predictions)}")
            if predictions:
                store_predictions(predictions, current_week, current_season, position)
                plot_top_predictions(predictions, position, TARGET_COLS[0], current_week, current_season)
            else:
                print(f"No valid predictions for {position}")

        # After week is over, update models with actual data
        new_data = load_and_preprocess_data([current_season])
        new_data = new_data[new_data['week'] == current_week]
        
        if not new_data.empty:
            df = update_dataset(df, new_data)
            models = update_models(models, preprocessor, new_data, current_week, current_season)
            
            # Compare predictions to actual results
            results = compare_predictions_to_actual(current_week, current_season)
            for result in results:
                plot_prediction_vs_actual(result['data'], result['position'], result['stat'], current_week, current_season)

        current_week += 1

        # In a real scenario, you would wait here until the next week's data is available
        input("Press Enter to continue to next week...")

    # Plot overall performance
    plot_weekly_performance(results, 'QB', 'passing_yards', current_season)

if __name__ == "__main__":
    main()