import pandas as pd
import numpy as np
from collections import Counter

def get_top_n_by_position(df, position, n=5):
    """Get top N players by projection at a given position"""
    position_df = df[df['Roster Position'] == position].copy()
    return position_df.nlargest(n, 'lineup_points')

def is_lineup_unique(new_lineup, existing_lineups):
    """Check if a lineup is unique compared to existing lineups"""
    for existing in existing_lineups:
        if set(existing['Nickname'].values) == set(new_lineup['Nickname'].values):
            return False
    return True

def calculate_player_usage(lineups):
    """Calculate usage statistics for all players across lineups"""
    total_lineups = len(lineups)
    all_players = []
    
    # Collect all player appearances
    for lineup in lineups:
        all_players.extend(lineup['Nickname'].tolist())
    
    # Count appearances
    player_counts = Counter(all_players)
    
    # Convert to DataFrame with percentages
    usage_df = pd.DataFrame.from_dict(player_counts, orient='index', columns=['Count'])
    usage_df['Usage_Percentage'] = (usage_df['Count'] / total_lineups * 100).round(2)
    usage_df = usage_df.sort_values('Usage_Percentage', ascending=False)
    
    return usage_df

def check_usage_limit(player_nickname, current_lineups, num_lineups, max_usage_percentage=50):
    """Check if adding this player would exceed the usage limit"""
    if not current_lineups:
        return True
        
    current_usage = sum(1 for lineup in current_lineups if player_nickname in lineup['Nickname'].values)
    projected_usage_percentage = (current_usage + 1) / num_lineups * 100
    
    return projected_usage_percentage <= max_usage_percentage

def optimize_lineups(csv_file, num_lineups, salary_cap=60000, exclude_players=None, include_players=None):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Handle injury filtering
    excluded_injury_statuses = ['IR', 'O', 'D']
    df['Injury Indicator'] = df['Injury Indicator'].fillna('')
    df = df[~df['Injury Indicator'].str.upper().isin([x.upper() for x in excluded_injury_statuses])]
    
    # Handle required columns and clean data
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Salary', 'FPPG', 'Roster Position', 'Nickname'])
    
    # Set up lineup points column
    df['lineup_points'] = df['fanduel_fantasy_points']
    df.loc[df['Roster Position'] == 'DEF', 'lineup_points'] = df.loc[df['Roster Position'] == 'DEF', 'FPPG']
    
    all_lineups = []
    if exclude_players is None:
        exclude_players = set()
    
    # Strategy patterns for lineup building
    strategies = [
        ('QB', 5),    # First 5 lineups built around top QBs
        ('RB/FLEX', 5),  # Next 5 built around top RBs
        ('WR/FLEX', 5),  # Next 5 built around top WRs
        ('TE/FLEX', 5),  # Next 5 built around top TEs
    ]

    current_lineup = 0
    attempts = 0
    max_attempts = num_lineups * 10  # Prevent infinite loops

    while current_lineup < num_lineups and attempts < max_attempts:
        attempts += 1
        strategy_index = min(current_lineup // 5, len(strategies) - 1)
        focus_position, n_players = strategies[strategy_index]
        
        # Get top players for the focus position
        top_players = get_top_n_by_position(df, focus_position, n_players)
        
        for _, focus_player in top_players.iterrows():
            if current_lineup >= num_lineups:
                break
                
            # Check usage limit for focus player
            if not check_usage_limit(focus_player['Nickname'], all_lineups, num_lineups):
                continue
                
            # Start with a fresh DataFrame for each attempt
            current_df = df.copy()
            
            # Initialize lineup with the focus player
            lineup = [focus_player]
            remaining_salary = salary_cap - focus_player['Salary']
            
            # Set up positions needed
            positions_needed = {
                'QB': 1,
                'RB/FLEX': 2,
                'WR/FLEX': 3,
                'TE/FLEX': 1,
                'DEF': 1,
                'FLEX': 1
            }
            
            # Reduce count for the position we've filled
            positions_needed[focus_player['Roster Position']] -= 1
            
            # Fill remaining positions
            lineup_complete = True
            for position in ['QB', 'RB/FLEX', 'WR/FLEX', 'TE/FLEX', 'DEF', 'FLEX']:
                if position == focus_player['Roster Position']:
                    continue
                    
                if position == 'FLEX':
                    available_players = current_df[current_df['Roster Position'].isin(['RB/FLEX', 'WR/FLEX', 'TE/FLEX'])]
                else:
                    available_players = current_df[current_df['Roster Position'] == position]
                
                # Sort by value (points per dollar)
                available_players['value'] = available_players['lineup_points'] / available_players['Salary']
                available_players = available_players.sort_values('value', ascending=False)
                
                count = positions_needed[position]
                for _, player in available_players.iterrows():
                    if count > 0 and player['Salary'] <= remaining_salary:
                        # Check usage limit before adding player
                        if check_usage_limit(player['Nickname'], all_lineups, num_lineups):
                            lineup.append(player)
                            remaining_salary -= player['Salary']
                            count -= 1
                            current_df = current_df[current_df['Nickname'] != player['Nickname']]
                    if count == 0:
                        break
                        
                if count > 0:
                    lineup_complete = False
                    break
                positions_needed[position] = count
            
            # Check if we have a valid and unique lineup
            if lineup_complete and sum(positions_needed.values()) == 0:
                lineup_df = pd.DataFrame(lineup)
                if is_lineup_unique(lineup_df, all_lineups):
                    all_lineups.append(lineup_df)
                    current_lineup += 1
                    print(f"Successfully created lineup {current_lineup}")
                    
                    if current_lineup >= num_lineups:
                        break
    
    if current_lineup < num_lineups:
        print(f"\nWarning: Could only generate {current_lineup} valid lineups while maintaining usage limits")
    
    # Calculate and display player usage statistics
    print("\nPlayer Usage Statistics:")
    usage_stats = calculate_player_usage(all_lineups)
    print("\nTop 20 Most Used Players:")
    print(usage_stats.head(20))
    
    # Print usage by position
    print("\nUsage Statistics by Position:")
    for position in ['QB', 'RB/FLEX', 'WR/FLEX', 'TE/FLEX', 'DEF']:
        position_players = set()
        for lineup in all_lineups:
            position_players.update(lineup[lineup['Roster Position'] == position]['Nickname'].tolist())
        print(f"\n{position} Usage:")
        position_usage = usage_stats[usage_stats.index.isin(position_players)]
        print(position_usage)
    
    return all_lineups

def display_lineups(lineups):
    if not lineups:
        print("No valid lineups found.")
        return
    
    for i, lineup in enumerate(lineups, 1):
        print(f"\nLineup {i}:")
        display_columns = ['Roster Position', 'Nickname', 'Salary', 'lineup_points', 'Injury Indicator']
        lineup_display = lineup[display_columns].copy()
        print(lineup_display)
        print(f"Total Salary: ${lineup['Salary'].sum()}")
        total_points = lineup['lineup_points'].sum()
        print(f"Projected Points: {total_points:.2f}")

# Example usage
csv_file = '../predictions/fanduel_value_week5.csv'  # Replace with your CSV file path
num_lineups = 5  # Number of unique lineups to generate
salary_cap = 60000
exclude_players = []  # Replace with actual names
include_players = []  # Replace with actual name

try:
    lineups = optimize_lineups(csv_file, num_lineups, salary_cap, exclude_players, include_players)
    display_lineups(lineups)
except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    traceback.print_exc()