# snap_calculations.py
import pandas as pd
from collections import defaultdict

def calculate_player_snap_counts(pbp_data, categorize_play_func, parse_players_func):
    """
    Calculate snap counts for players from play-by-play data
    
    Args:
        pbp_data: Play-by-play DataFrame
        categorize_play_func: Function to categorize play types
        parse_players_func: Function to parse player strings
    
    Returns:
        DataFrame: Player snap counts with percentages
    """
    snap_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    team_snap_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    # Create a copy of the DataFrame to avoid the SettingWithCopyWarning
    pbp_data = pbp_data.copy()
    
    # Pre-compute play categories
    pbp_data['play_category'] = pbp_data['play_type'].map(categorize_play_func)
    
    # Group by game_id for faster processing
    for game_id, game_data in pbp_data.groupby('game_id'):
        for _, play in game_data.iterrows():
            posteam = play['posteam']
            defteam = play['defteam']
            play_category = play['play_category']
            
            if posteam is None and defteam is None:
                if play_category == 'special_teams':
                    posteam = defteam = 'UNKNOWN_TEAM'
                else:
                    continue
            elif posteam is None:
                posteam = defteam
            elif defteam is None:
                defteam = posteam
            
            # Count team snaps
            if play_category == 'special_teams':
                team_snap_counts[game_id]['special_teams'][posteam] += 1
                team_snap_counts[game_id]['special_teams'][defteam] += 1
            elif play_category == 'scrimmage':
                team_snap_counts[game_id]['offense'][posteam] += 1
                team_snap_counts[game_id]['defense'][defteam] += 1
            
            # Count player snaps
            offense_players = parse_players_func(play['offense_players'])
            defense_players = parse_players_func(play['defense_players'])
            
            if play_category == 'special_teams':
                for player in set(offense_players + defense_players):
                    if player:
                        snap_counts[player][game_id]['special_teams'] += 1
            elif play_category == 'scrimmage':
                for player in offense_players:
                    if player:
                        snap_counts[player][game_id]['offense'] += 1
                for player in defense_players:
                    if player:
                        snap_counts[player][game_id]['defense'] += 1
    
    # Calculate percentages and create final dataframe
    snap_count_list = []
    for player_id, games in snap_counts.items():
        for game_id, counts in games.items():
            off_snaps = counts['offense']
            def_snaps = counts['defense']
            st_snaps = counts['special_teams']
            total_snaps = off_snaps + def_snaps + st_snaps
            
            # Find player team
            player_team_mask = (
                (pbp_data['game_id'] == game_id) & 
                ((pbp_data['offense_players'].apply(lambda x: player_id in x if isinstance(x, list) else player_id in str(x))) | 
                 (pbp_data['defense_players'].apply(lambda x: player_id in x if isinstance(x, list) else player_id in str(x))))
            )
            
            if player_team_mask.any():
                player_team = pbp_data[player_team_mask]['posteam'].iloc[0]
            else:
                player_team = 'UNKNOWN'
            
            team_off_snaps = team_snap_counts[game_id]['offense'].get(player_team, 0)
            team_def_snaps = team_snap_counts[game_id]['defense'].get(player_team, 0)
            team_st_snaps = team_snap_counts[game_id]['special_teams'].get(player_team, 0)
            
            snap_count_list.append({
                'player_id': player_id,
                'game_id': game_id,
                'week': pbp_data[pbp_data['game_id'] == game_id]['week'].iloc[0],
                'season': pbp_data[pbp_data['game_id'] == game_id]['season'].iloc[0],
                'offensive_snaps': off_snaps,
                'defensive_snaps': def_snaps,
                'special_teams_snaps': st_snaps,
                'total_snaps': total_snaps,
                'offensive_snap_pct': (off_snaps / team_off_snaps * 100) if team_off_snaps > 0 else 0,
                'defensive_snap_pct': (def_snaps / team_def_snaps * 100) if team_def_snaps > 0 else 0,
                'special_teams_snap_pct': (st_snaps / team_st_snaps * 100) if team_st_snaps > 0 else 0
            })
    
    result_df = pd.DataFrame(snap_count_list)
    return result_df