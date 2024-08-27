import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_top_predictions(predictions, position, stat_name, week, season, top_n=10):
    """
    Plot bar chart of top predictions for a given position and stat.
    """
    if not predictions:
        print(f"No predictions to plot for {position}")
        return
    
    top_predictions = predictions[:top_n]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=[p[0] for p in top_predictions], y=[p[1] for p in top_predictions])
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