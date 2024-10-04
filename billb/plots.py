import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_top_predictions(predictions, position, stat_name, week, season, top_n=10):
    """
    Plot bar chart of top predictions for a given position and stat.
    Accepts either a DataFrame or a path to a CSV file.
    """
    if isinstance(predictions, str):
        # If predictions is a string, assume it's a path to a CSV file
        predictions_df = pd.read_csv(predictions)
    elif isinstance(predictions, pd.DataFrame):
        # If predictions is already a DataFrame, use it as is
        predictions_df = predictions
    else:
        raise ValueError("Input must be either a DataFrame or a path to a CSV file")

    if predictions_df.empty:
        print(f"No predictions to plot for {position}")
        return

    # Ensure there's a 'player_name' column
    if 'player_name' not in predictions_df.columns:
        raise ValueError("The DataFrame must contain a 'player_name' column")

    # Sort the DataFrame by the stat_name column in descending order
    predictions_df = predictions_df.sort_values(by=stat_name, ascending=False)

    # Select the top N predictions
    top_predictions = predictions_df.head(top_n)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='player_name', y=stat_name, data=top_predictions)
    plt.title(f'Top {top_n} Predicted {stat_name} for {position}s (Week {week}, Season {season})')
    plt.xlabel('Player')
    plt.ylabel(f'Predicted {stat_name}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_prediction_vs_actual(merged_data, position, stat_name, week, season):
    """
    Plot scatter plot of predicted vs actual stats.
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='predicted_stat', y='actual_stat', data=merged_data)
    
    max_val = max(merged_data['predicted_stat'].max(), merged_data['actual_stat'].max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    
    plt.title(f'{position} {stat_name}: Predicted vs Actual (Week {week}, Season {season})')
    plt.xlabel(f'Predicted {stat_name}')
    plt.ylabel(f'Actual {stat_name}')
    plt.tight_layout()
    plt.show()

def plot_prediction_error_distribution(merged_data, position, stat_name, week, season):
    """
    Plot histogram of prediction errors.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(merged_data['difference'], kde=True)
    plt.title(f'{position} {stat_name}: Prediction Error Distribution (Week {week}, Season {season})')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plot_weekly_performance(weekly_mse, position, stat_name, season):
    """
    Plot line chart of model performance (MSE) over weeks.
    """
    weeks = list(weekly_mse.keys())
    mse_values = list(weekly_mse.values())
    
    plt.figure(figsize=(12, 6))
    plt.plot(weeks, mse_values, marker='o')
    plt.title(f'{position} {stat_name}: Weekly Model Performance (Season {season})')
    plt.xlabel('Week')
    plt.ylabel('Mean Squared Error')
    plt.xticks(weeks)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_fantasy_points_distribution(fantasy_predictions, week, season):
    """
    Plot box plot of predicted fantasy points by position.
    """
    df = pd.DataFrame(fantasy_predictions, columns=['Player', 'Position', 'Fantasy Points'])
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Position', y='Fantasy Points', data=df)
    plt.title(f'Predicted Fantasy Points Distribution by Position (Week {week}, Season {season})')
    plt.ylabel('Predicted Fantasy Points')
    plt.tight_layout()
    plt.show()

def plot_player_trend(player_name, stat_name, actual_data, predicted_data, season):
    """
    Plot line chart of a player's actual and predicted stats over the season.
    """
    weeks = list(actual_data.keys())
    actual_values = list(actual_data.values())
    predicted_values = [predicted_data.get(week, np.nan) for week in weeks]
    
    plt.figure(figsize=(12, 6))
    plt.plot(weeks, actual_values, marker='o', label='Actual')
    plt.plot(weeks, predicted_values, marker='s', label='Predicted')
    plt.title(f"{player_name}'s {stat_name} Trend (Season {season})")
    plt.xlabel('Week')
    plt.ylabel(stat_name)
    plt.legend()
    plt.xticks(weeks)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, features):
    """
    Plot correlation matrix of selected features.
    """
    corr = df[features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.show()

plot_top_predictions('predictions/QB_predictions_week5_season2024.csv', 'QB', 'fanduel_fantasy_points', '5','2024', 24)
plot_top_predictions('predictions/RB_predictions_week5_season2024.csv', 'RB', 'fanduel_fantasy_points', '5','2024', 36)
plot_top_predictions('predictions/WR_predictions_week5_season2024.csv', 'WR', 'fanduel_fantasy_points', '5','2024', 24)
plot_top_predictions('predictions/TE_predictions_week5_season2024.csv', 'TE', 'fanduel_fantasy_points', '5','2024', 12)