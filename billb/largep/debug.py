#!/usr/bin/env python3
"""
Debug specific plays to understand discrepancies between Sportradar and nfl_data_py
"""

import os
import time
import requests
from collections import defaultdict

API_KEY = os.getenv('SPORTRADAR_API_KEY')

def get_game_pbp(game_id):
    """Fetch PBP data for a game"""
    base_url = "https://api.sportradar.com/nfl/official/trial/v7/en"
    url = f"{base_url}/games/{game_id}/pbp.json"
    headers = {
        "accept": "application/json",
        "x-api-key": API_KEY
    }
    
    time.sleep(3)
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None

def analyze_player_plays(game_id, player_name_search):
    """Analyze all plays for a specific player to see what's being counted"""
    
    print(f"\nAnalyzing plays for: {player_name_search}")
    print("="*80)
    
    pbp_data = get_game_pbp(game_id)
    if not pbp_data:
        print("Could not fetch PBP data")
        return
    
    player_plays = []
    
    # Process each quarter
    periods = pbp_data.get('periods', [])
    
    for period_num, period in enumerate(periods, 1):
        pbp = period.get('pbp', [])
        
        for item in pbp:
            if item.get('type') == 'drive':
                events = item.get('events', [])
                
                for event in events:
                    if event.get('type') == 'play':
                        # Check if this play involves our player
                        statistics = event.get('statistics', [])
                        
                        for stat in statistics:
                            player = stat.get('player', {})
                            if not player:
                                continue
                            
                            player_name = player.get('name', '')
                            if player_name_search.lower() in player_name.lower():
                                play_info = {
                                    'quarter': period_num,
                                    'clock': event.get('clock', ''),
                                    'play_type': event.get('play_type', ''),
                                    'official': event.get('official', True),
                                    'description': event.get('description', ''),
                                    'stat_type': stat.get('stat_type', ''),
                                    'player_name': player_name,
                                    'stat_details': {k: v for k, v in stat.items() 
                                                   if k not in ['player', 'team']},
                                    'start_situation': event.get('start_situation', {}),
                                    'end_situation': event.get('end_situation', {})
                                }
                                player_plays.append(play_info)
    
    # Print summary
    print(f"\nFound {len(player_plays)} plays involving {player_name_search}")
    
    # Group by stat type
    by_stat_type = defaultdict(list)
    for play in player_plays:
        by_stat_type[play['stat_type']].append(play)
    
    print(f"\nBreakdown by stat type:")
    for stat_type, plays in by_stat_type.items():
        official_count = sum(1 for p in plays if p['official'])
        unofficial_count = sum(1 for p in plays if not p['official'])
        print(f"  {stat_type}: {len(plays)} total ({official_count} official, {unofficial_count} unofficial)")
    
    # Show all plays in detail
    print(f"\nDetailed Play-by-Play:")
    print("="*80)
    
    for i, play in enumerate(player_plays, 1):
        official_marker = "✓" if play['official'] else "✗"
        print(f"\n{i}. Q{play['quarter']} {play['clock']} - {play['stat_type'].upper()} [{official_marker}]")
        print(f"   Type: {play['play_type']}")
        print(f"   Description: {play['description'][:100]}...")
        print(f"   Stats: {play['stat_details']}")
        
        # Check for penalty indicators
        desc_lower = play['description'].lower()
        if 'penalty' in desc_lower:
            print(f"   ⚠️  PENALTY in description")
        if 'nullified' in desc_lower:
            print(f"   ⚠️  NULLIFIED in description")
        if 'no play' in desc_lower:
            print(f"   ⚠️  NO PLAY in description")
    
    # Count attempts/targets/carries that are official
    print(f"\nOfficial Stat Totals:")
    print("="*80)
    
    for stat_type, plays in by_stat_type.items():
        official_plays = [p for p in plays if p['official']]
        
        if stat_type == 'pass':
            attempts = sum(p['stat_details'].get('attempt', 0) for p in official_plays)
            completions = sum(p['stat_details'].get('complete', 0) for p in official_plays)
            print(f"  Passing: {attempts} attempts, {completions} completions")
        
        elif stat_type == 'rush':
            carries = sum(p['stat_details'].get('attempt', 0) for p in official_plays)
            print(f"  Rushing: {carries} carries")
        
        elif stat_type == 'receive':
            targets = sum(p['stat_details'].get('target', 0) for p in official_plays)
            receptions = sum(p['stat_details'].get('reception', 0) for p in official_plays)
            print(f"  Receiving: {targets} targets, {receptions} receptions")


def compare_seahawks_broncos_game():
    """
    Analyze the SEA @ DEN game where Geno Smith had discrepancy
    Game ID from 2024 Week 1
    """
    
    # This is the SEA @ DEN game from 2024 Week 1
    # Geno Smith: Sportradar=31 attempts, nfl_data_py=25 attempts (6 extra)
    
    game_id = "8d5c502c-49d2-443d-ade6-c2f4e6c0eed3"  # SEA @ DEN
    
    print("\n" + "="*80)
    print("INVESTIGATING: Geno Smith Attempt Discrepancy")
    print("SEA @ DEN - 2024 Week 1")
    print("Sportradar: 31 attempts | nfl_data_py: 25 attempts | Difference: +6")
    print("="*80)
    
    analyze_player_plays(game_id, "Geno Smith")


def compare_chiefs_ravens_game():
    """
    Analyze the KC @ BAL game where Isiah Pacheco had discrepancy
    """
    
    game_id = "ebb5f3ac-bb13-4dc4-ab8b-6c65623eac13"  # KC @ BAL
    
    print("\n" + "="*80)
    print("INVESTIGATING: Isiah Pacheco Carry Discrepancy")
    print("KC @ BAL - 2024 Week 1")
    print("Sportradar: 17 carries | nfl_data_py: 15 carries | Difference: +2")
    print("="*80)
    
    analyze_player_plays(game_id, "Isiah Pacheco")


if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: SPORTRADAR_API_KEY environment variable not set")
        exit(1)
    
    # Analyze the two biggest discrepancies
    compare_chiefs_ravens_game()
    
    print("\n" + "="*80)
    print("\nWaiting 5 seconds before next analysis...")
    time.sleep(5)
    
    compare_seahawks_broncos_game()
    
    print("\n" + "="*80)
    print("\nANALYSIS COMPLETE")
    print("="*80)
    print("\nLook for patterns:")
    print("  1. Are there plays marked 'official: false'?")
    print("  2. Are there penalties in the description?")
    print("  3. Are there 'NO PLAY' or 'NULLIFIED' indicators?")
    print("  4. Do stats accumulate even on penalized plays?")