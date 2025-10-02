#!/usr/bin/env python3
"""
Validation Script: Compare Sportradar API data vs nfl_data_py
Tests the Sportradar collector against known 2024 data to ensure accuracy
"""

import pandas as pd
import numpy as np
from nfl_data_py import import_weekly_data
from sportradar_nfl_data_collector import SportradarNFLCollector
import requests
from collections import defaultdict
import time
import os

# Ensure API key is set
if not os.getenv('SPORTRADAR_API_KEY'):
    raise EnvironmentError(
        "SPORTRADAR_API_KEY environment variable not set.\n"
        "Set it with: export SPORTRADAR_API_KEY='your_api_key_here'"
    )

class SportradarValidator:
    """Validate Sportradar data collection against nfl_data_py baseline"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.collector = SportradarNFLCollector(api_key)
        self.validation_results = []
        
    def get_2024_schedule_game_ids(self, week):
        """Get 2024 game IDs from Sportradar to test against"""
        base_url = "https://api.sportradar.com/nfl/official/trial/v7/en"
        url = f"{base_url}/games/2024/REG/{week:02d}/schedule.json"
        
        headers = {
            "accept": "application/json",
            "x-api-key": self.api_key
        }
        
        print(f"  Fetching schedule (waiting 3 seconds for rate limit)...")
        time.sleep(3)  # Rate limit - trial allows 1 req/sec, use 3 to be safe
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            games = data.get('week', {}).get('games', [])
            game_ids = [g['id'] for g in games if g.get('status') == 'closed']
            return game_ids
        return []
    
    def compare_player_stats(self, sportradar_df, nfl_data_df, week):
        """Compare player stats between both data sources"""
        
        print(f"\n{'='*70}")
        print(f"VALIDATION REPORT - 2024 Week {week}")
        print(f"{'='*70}\n")
        
        # Key stats to validate
        stat_columns = [
            'passing_yards', 'passing_tds', 'interceptions', 'completions', 'attempts',
            'rushing_yards', 'rushing_tds', 'carries',
            'receiving_yards', 'receiving_tds', 'receptions', 'targets'
        ]
        
        # Normalize player names for matching
        sportradar_df['name_normalized'] = sportradar_df['player_name'].str.lower().str.strip()
        nfl_data_df['name_normalized'] = nfl_data_df['player_display_name'].str.lower().str.strip()
        
        # Get common players
        sportradar_players = set(sportradar_df['name_normalized'])
        nfl_data_players = set(nfl_data_df['name_normalized'])
        common_players = sportradar_players & nfl_data_players
        
        print(f"Dataset Coverage:")
        print(f"  Sportradar players: {len(sportradar_players)}")
        print(f"  nfl_data_py players: {len(nfl_data_players)}")
        print(f"  Common players: {len(common_players)}")
        print(f"  Coverage: {len(common_players)/len(nfl_data_players)*100:.1f}%")
        
        if len(common_players) == 0:
            print("\n  WARNING: No common players found! Check name matching logic.")
            return None
        
        # Compare stats for common players
        discrepancies = []
        perfect_matches = 0
        
        for player_name in common_players:
            sr_player = sportradar_df[sportradar_df['name_normalized'] == player_name].iloc[0]
            nfl_player = nfl_data_df[nfl_data_df['name_normalized'] == player_name].iloc[0]
            
            player_discrepancies = []
            is_perfect_match = True
            
            for stat in stat_columns:
                if stat not in sr_player or stat not in nfl_player:
                    continue
                
                sr_val = sr_player[stat] if pd.notna(sr_player[stat]) else 0
                nfl_val = nfl_player[stat] if pd.notna(nfl_player[stat]) else 0
                
                # Allow small floating point differences
                if abs(sr_val - nfl_val) > 0.5:
                    is_perfect_match = False
                    player_discrepancies.append({
                        'stat': stat,
                        'sportradar': sr_val,
                        'nfl_data_py': nfl_val,
                        'diff': sr_val - nfl_val
                    })
            
            if is_perfect_match:
                perfect_matches += 1
            elif player_discrepancies:
                discrepancies.append({
                    'player': sr_player['player_name'],
                    'position': sr_player.get('position', 'UNK'),
                    'team': sr_player.get('recent_team', 'UNK'),
                    'discrepancies': player_discrepancies
                })
        
        # Calculate accuracy
        accuracy = (perfect_matches / len(common_players)) * 100
        
        print(f"\nAccuracy Results:")
        print(f"  Perfect matches: {perfect_matches}/{len(common_players)} ({accuracy:.1f}%)")
        print(f"  Players with discrepancies: {len(discrepancies)}")
        
        # Show top discrepancies
        if discrepancies:
            print(f"\nTop 10 Players with Discrepancies:")
            print(f"{'='*70}")
            
            for i, disc in enumerate(discrepancies[:10], 1):
                print(f"\n{i}. {disc['player']} ({disc['position']}, {disc['team']})")
                for stat_disc in disc['discrepancies']:
                    print(f"   {stat_disc['stat']}:")
                    print(f"     Sportradar: {stat_disc['sportradar']}")
                    print(f"     nfl_data_py: {stat_disc['nfl_data_py']}")
                    print(f"     Difference: {stat_disc['diff']:+.1f}")
        
        # Aggregate statistics
        print(f"\nAggregate Validation:")
        print(f"{'='*70}")
        
        for stat in stat_columns:
            if stat in sportradar_df.columns and stat in nfl_data_df.columns:
                sr_total = sportradar_df[stat].sum()
                nfl_total = nfl_data_df[stat].sum()
                diff_pct = ((sr_total - nfl_total) / max(nfl_total, 1)) * 100
                
                print(f"\n{stat}:")
                print(f"  Sportradar total: {sr_total:,.1f}")
                print(f"  nfl_data_py total: {nfl_total:,.1f}")
                print(f"  Difference: {diff_pct:+.2f}%")
        
        return {
            'week': week,
            'common_players': len(common_players),
            'perfect_matches': perfect_matches,
            'accuracy': accuracy,
            'discrepancies': discrepancies
        }
    
    def detailed_play_analysis(self, game_id, week):
        """Deep dive into a single game to identify specific issues"""
        print(f"\nDetailed Play-by-Play Analysis")
        print(f"{'='*70}")
        print(f"Game ID: {game_id}")
        
        pbp_data = self.collector.get_game_pbp(game_id)
        
        if not pbp_data:
            print("Could not fetch PBP data")
            return
        
        # Analyze play characteristics
        total_plays = 0
        official_plays = 0
        unofficial_plays = 0
        plays_by_type = defaultdict(int)
        unofficial_reasons = defaultdict(int)
        
        periods = pbp_data.get('periods', [])
        
        for period in periods:
            pbp = period.get('pbp', [])
            
            for item in pbp:
                if item.get('type') == 'drive':
                    events = item.get('events', [])
                    
                    for event in events:
                        if event.get('type') == 'play':
                            total_plays += 1
                            play_type = event.get('play_type', 'unknown')
                            plays_by_type[play_type] += 1
                            
                            if event.get('official', True):
                                official_plays += 1
                            else:
                                unofficial_plays += 1
                                # Try to determine why it's unofficial
                                desc = event.get('description', '').lower()
                                if 'penalty' in desc:
                                    unofficial_reasons['penalty'] += 1
                                elif 'nullified' in desc:
                                    unofficial_reasons['nullified'] += 1
                                else:
                                    unofficial_reasons['other'] += 1
        
        print(f"\nPlay Statistics:")
        print(f"  Total plays: {total_plays}")
        print(f"  Official plays: {official_plays} ({official_plays/max(total_plays,1)*100:.1f}%)")
        print(f"  Unofficial plays: {unofficial_plays} ({unofficial_plays/max(total_plays,1)*100:.1f}%)")
        
        print(f"\nPlays by Type:")
        for play_type, count in sorted(plays_by_type.items(), key=lambda x: x[1], reverse=True):
            print(f"  {play_type}: {count}")
        
        if unofficial_reasons:
            print(f"\nUnofficial Play Reasons:")
            for reason, count in unofficial_reasons.items():
                print(f"  {reason}: {count}")
        
        return {
            'total_plays': total_plays,
            'official_plays': official_plays,
            'unofficial_plays': unofficial_plays,
            'plays_by_type': dict(plays_by_type),
            'unofficial_reasons': dict(unofficial_reasons)
        }
    
    def run_validation(self, week=1):
        """Run complete validation for a week"""
        print(f"\n{'#'*70}")
        print(f"# STARTING VALIDATION FOR 2024 WEEK {week}")
        print(f"{'#'*70}\n")
        
        # Add initial delay to avoid hitting rate limit immediately
        print("Waiting 3 seconds before starting API calls...")
        time.sleep(3)
        
        # Step 1: Get baseline data from nfl_data_py
        print("Step 1: Loading baseline data from nfl_data_py...")
        nfl_data = import_weekly_data([2024])
        nfl_data_week = nfl_data[nfl_data['week'] == week].copy()
        print(f"  Loaded {len(nfl_data_week)} player records")
        
        # Step 2: Get game IDs from Sportradar
        print("\nStep 2: Fetching 2024 game IDs from Sportradar...")
        game_ids = self.get_2024_schedule_game_ids(week)
        print(f"  Found {len(game_ids)} completed games")
        
        if not game_ids:
            print("  ERROR: No game IDs found. Cannot proceed.")
            return None
        
        # Step 3: Process games with Sportradar
        print("\nStep 3: Processing games with Sportradar collector...")
        for game_id in game_ids:
            self.collector.process_game(game_id, 2024, week)
        
        sportradar_df = self.collector.convert_to_dataframe()
        print(f"  Collected {len(sportradar_df)} player records")
        
        # Step 4: Detailed analysis on first game
        if game_ids:
            print("\nStep 4: Running detailed play analysis on first game...")
            play_analysis = self.detailed_play_analysis(game_ids[0], week)
        
        # Step 5: Compare datasets
        print("\nStep 5: Comparing datasets...")
        validation_result = self.compare_player_stats(sportradar_df, nfl_data_week, week)
        
        # Step 6: Generate recommendations
        print(f"\n{'='*70}")
        print("RECOMMENDATIONS")
        print(f"{'='*70}\n")
        
        if validation_result:
            if validation_result['accuracy'] >= 95:
                print("  Status: EXCELLENT - Data collection is highly accurate")
                print("  Action: Collector is ready for production use")
            elif validation_result['accuracy'] >= 85:
                print("  Status: GOOD - Minor discrepancies detected")
                print("  Action: Review specific player discrepancies above")
                print("  Possible causes:")
                print("    - Name matching issues")
                print("    - Missing stat types in collector")
                print("    - Timing differences (stats corrections)")
            elif validation_result['accuracy'] >= 70:
                print("  Status: FAIR - Significant discrepancies detected")
                print("  Action: Review collector logic for specific stat types")
                print("  Possible causes:")
                print("    - Missing edge cases in play processing")
                print("    - Incorrect stat type mapping")
                print("    - Official flag not being respected")
            else:
                print("  Status: POOR - Major issues detected")
                print("  Action: Deep debugging required")
                print("  Possible causes:")
                print("    - Fundamental logic error in collector")
                print("    - Wrong API endpoints or data format")
                print("    - Player ID mapping issues")
        
        print(f"\nSpecific Issues to Investigate:")
        if validation_result and validation_result['discrepancies']:
            # Identify patterns in discrepancies
            stat_issues = defaultdict(int)
            position_issues = defaultdict(int)
            
            for disc in validation_result['discrepancies']:
                position_issues[disc['position']] += 1
                for stat_disc in disc['discrepancies']:
                    stat_issues[stat_disc['stat']] += 1
            
            if stat_issues:
                print("\n  Most problematic stats:")
                for stat, count in sorted(stat_issues.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    - {stat}: {count} players affected")
            
            if position_issues:
                print("\n  Most problematic positions:")
                for pos, count in sorted(position_issues.items(), key=lambda x: x[1], reverse=True):
                    print(f"    - {pos}: {count} players affected")
        
        # Save detailed results
        if validation_result:
            output_file = f"validation_results_2024_week{week}.json"
            import json
            with open(output_file, 'w') as f:
                json.dump({
                    'week': week,
                    'accuracy': validation_result['accuracy'],
                    'common_players': validation_result['common_players'],
                    'perfect_matches': validation_result['perfect_matches'],
                    'discrepancy_count': len(validation_result['discrepancies']),
                    'play_analysis': play_analysis if 'play_analysis' in locals() else None
                }, f, indent=2)
            print(f"\n  Detailed results saved to: {output_file}")
        
        return validation_result


def main():
    """Run the validation"""
    API_KEY = os.getenv('SPORTRADAR_API_KEY')
    
    validator = SportradarValidator(API_KEY)
    
    # Test Week 1 of 2024 season
    result = validator.run_validation(week=1)
    
    print(f"\n{'#'*70}")
    print("# VALIDATION COMPLETE")
    print(f"{'#'*70}\n")
    
    if result and result['accuracy'] >= 85:
        print("VERDICT: Sportradar collector is ready for 2025 data collection")
    else:
        print("VERDICT: Sportradar collector needs refinement before production use")


if __name__ == "__main__":
    main()