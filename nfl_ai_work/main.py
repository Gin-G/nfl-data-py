import pandas as pd
import urllib
from data_loader import load_and_preprocess_data, update_dataset
from models import create_and_train_models, load_models, update_models, save_models
from predictions import get_all_predictions, store_predictions, compare_predictions_to_actual
from plots import plot_top_predictions, plot_prediction_vs_actual, plot_weekly_performance
from config import SEASONS, TARGET_COLS
import os

def main():
    # Initial setup
    current_season = 2024
    current_week = 1  # Start from week 1

    # Load or create initial dataset
    if os.path.exists('data/nfl_dataset.csv'):
        df = pd.read_csv('data/nfl_dataset.csv')
    else:
        df = load_and_preprocess_data(SEASONS)
        df.to_csv('data/nfl_dataset.csv', index=False)

    # Load or train models
    if os.path.exists('saved_models/model_qb.keras'):
        print("Loading models...")
        models, preprocessor = load_models()
        print(f"Loaded models: {list(models.keys())}")
        print(f"Preprocessor features: {preprocessor.get_feature_names_out()}")
    else:
        models, preprocessor = create_and_train_models(df)
        print(f"Loaded models: {list(models.keys())}")
        print(f"Preprocessor features: {preprocessor.get_feature_names_out()}")
        save_models(models, preprocessor)

    print(f"\nProcessing Week {current_week}")

    # Check if we're at the beginning of a new season
    if current_week == 1:
        # Use data from the previous season for initial prediction
        prediction_data = df[df['season'] == current_season - 1]
    else:
        prediction_data = df[df['season'] == current_season]

    # Make predictions for each position
    for position in ['QB', 'RB', 'WR', 'TE']:
        print(f"Making predictions for {position}")
        if current_week == 1:
            prediction_data = df[df['season'] == current_season - 1]
        else:
            prediction_data = df[df['season'] == current_season]

        predictions = get_all_predictions(position, prediction_data, models, preprocessor, current_week, current_season)
        print(f"Number of predictions for {position}: {len(predictions)}")
        if predictions:
            store_predictions(predictions, current_week, current_season, position)
            plot_top_predictions(predictions, position, TARGET_COLS[0], current_week, current_season)
        else:
            print(f"No valid predictions for {position}")

    # Try to load new data, but handle the case when it's not available
    try:
        new_data = load_and_preprocess_data([current_season])
        new_data = new_data[new_data['week'] == current_week]
    except urllib.error.HTTPError as e:
        print(f"Data for Week {current_week} of season {current_season} is not available yet.")
        new_data = pd.DataFrame()

    if not new_data.empty:
        df = update_dataset(df, new_data)
        models = update_models(models, preprocessor, new_data, current_week, current_season)

        try:
            # Compare predictions to actual results
            results = compare_predictions_to_actual(current_week, current_season)
            for result in results:
                plot_prediction_vs_actual(result['data'], result['position'], result['stat'], current_week, current_season)
        except urllib.error.HTTPError as e:
            results = pd.DataFrame()
            print(f"Actual data for Week {current_week} of season {current_season} is not available yet.")

        # Plot overall performance
        plot_weekly_performance(results, 'QB', 'passing_yards', current_season)
    else:
        print("No new data available to update models or compare predictions.")

if __name__ == "__main__":
    main()