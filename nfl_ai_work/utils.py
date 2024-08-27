import pandas as pd
import numpy as np

def calculate_fantasy_points(stats, scoring_system='standard'):
    """
    Calculate fantasy points based on player stats.
    
    :param stats: dict, player statistics
    :param scoring_system: str, 'standard' or 'ppr'
    :return: float, fantasy points
    """
    points = 0
    
    # Passing
    points += stats.get('passing_yards', 0) * 0.04  # 1 point per 25 passing yards
    points += stats.get('passing_touchdowns', 0) * 4  # 4 points per passing TD
    points -= stats.get('interceptions', 0) * 2  # -2 points per interception

    # Rushing
    points += stats.get('rushing_yards', 0) * 0.1  # 1 point per 10 rushing yards
    points += stats.get('rushing_touchdowns', 0) * 6  # 6 points per rushing TD

    # Receiving
    points += stats.get('receiving_yards', 0) * 0.1  # 1 point per 10 receiving yards
    points += stats.get('receiving_touchdowns', 0) * 6  # 6 points per receiving TD
    
    if scoring_system == 'ppr':
        points += stats.get('receptions', 0)  # 1 point per reception in PPR

    # Fumbles
    points -= stats.get('fumbles_lost', 0) * 2  # -2 points per fumble lost

    return points

def calculate_accuracy(predictions, actuals):
    """
    Calculate prediction accuracy.
    
    :param predictions: list of predicted values
    :param actuals: list of actual values
    :return: float, accuracy score
    """
    return 1 - np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals)))

def get_player_consistency(player_stats):
    """
    Calculate player consistency based on coefficient of variation.
    
    :param player_stats: list of player's weekly stats
    :return: float, consistency score (lower is more consistent)
    """
    return np.std(player_stats) / np.mean(player_stats)

def identify_sleepers(predictions, adp_data, threshold=20):
    """
    Identify potential sleeper picks based on predictions vs ADP.
    
    :param predictions: dict, predicted fantasy points for players
    :param adp_data: dict, average draft position for players
    :param threshold: int, threshold for ADP difference
    :return: list of potential sleeper picks
    """
    sleepers = []
    for player, points in predictions.items():
        if player in adp_data:
            predicted_rank = sorted(predictions, key=predictions.get, reverse=True).index(player) + 1
            adp_rank = adp_data[player]
            if adp_rank - predicted_rank > threshold:
                sleepers.append((player, predicted_rank, adp_rank))
    return sleepers

def calculate_value_over_replacement(predictions, position):
    """
    Calculate value over replacement for each player in a position.
    
    :param predictions: dict, predicted fantasy points for players
    :param position: str, player position
    :return: dict, value over replacement for each player
    """
    sorted_players = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    replacement_level = sorted_players[12][1] if position in ['QB', 'TE'] else sorted_players[24][1]
    return {player: points - replacement_level for player, points in sorted_players}