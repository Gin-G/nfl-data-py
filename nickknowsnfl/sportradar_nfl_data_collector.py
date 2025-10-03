#!/usr/bin/env python3
"""
Sportradar NFL Data Collector
Fetches 2025 NFL data from Sportradar API and formats it to match nfl_data_py schema
"""

import requests
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import time
import json
import os
from tqdm import tqdm

# Ensure API key is set
if not os.getenv('SPORTRADAR_API_KEY'):
    raise EnvironmentError(
        "SPORTRADAR_API_KEY environment variable not set.\n"
        "Set it with: export SPORTRADAR_API_KEY='your_api_key_here'"
    )

class SportradarNFLCollector:
    """Collect and process NFL data from Sportradar API"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.sportradar.com/nfl/official/trial/v7/en"
        self.headers = {
            "accept": "application/json",
            "x-api-key": api_key
        }
        
        # Rate limiting: Trial allows 1 request per second, use 3 seconds to be safe
        self.rate_limit_delay = 3.0
        
        # Player stat accumulators - regular dict
        self.player_stats = {}
        
        # Player metadata cache
        self.player_metadata = {}
        
    def _make_request(self, url):
        """Make API request with rate limiting and error handling"""
        try:
            time.sleep(self.rate_limit_delay)
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
    
    def get_weekly_schedule(self, season, week):
        """Fetch weekly schedule to get game IDs"""
        url = f"{self.base_url}/games/{season}/REG/{week:02d}/schedule.json"
        print(f"Fetching schedule for {season} Week {week}...")
        return self._make_request(url)
    
    def get_game_pbp(self, game_id):
        """Fetch play-by-play data for a specific game"""
        url = f"{self.base_url}/games/{game_id}/pbp.json"
        print(f"  Fetching PBP for game {game_id[:8]}...")
        return self._make_request(url)
    
    def process_play_statistics(self, play, game_info, week):
        """Process a single play's statistics and accumulate player stats"""
        
        # Skip unofficial plays (penalties, etc)
        if not play.get('official', True):
            return
        
        # Check if play was nullified by penalty at the play level
        play_nullified = False
        details = play.get('details', [])
        for detail in details:
            if detail.get('category') == 'penalty':
                penalty = detail.get('penalty', {})
                if penalty.get('no_play'):
                    play_nullified = True
                    break
        
        # Get play metadata
        play_type = play.get('play_type', '')
        
        # Process each player's statistics in the play
        statistics = play.get('statistics', [])
        
        for stat in statistics:
            stat_type = stat.get('stat_type')
            player = stat.get('player', {})
            team = stat.get('team', {})
            
            if not player or not player.get('id'):
                continue
            
            player_id = player['id']
            player_name = player.get('name', '')
            position = player.get('position', '')
            team_alias = team.get('alias', '')
            season = game_info['season']
            
            # Store player metadata
            if player_id not in self.player_metadata:
                self.player_metadata[player_id] = {
                    'player_id': player_id,
                    'player_name': player_name,
                    'player_display_name': player_name,
                    'position': position,
                    'recent_team': team_alias,
                    'sr_id': player.get('sr_id', '')
                }
            
            # Create player-game key
            key = (player_id, season, week, team_alias)
            
            # Initialize stats dict if needed
            if key not in self.player_stats:
                self.player_stats[key] = {}
            
            # Process different stat types
            # Pass both stat and play_nullified flag to stat processors
            if stat_type == 'pass':
                self._process_passing_stats(key, stat, play_nullified)
            elif stat_type == 'rush':
                self._process_rushing_stats(key, stat, play_nullified)
            elif stat_type == 'receive':
                self._process_receiving_stats(key, stat, play_nullified)
            elif stat_type == 'fumble':
                self._process_fumble_stats(key, stat, play_nullified)
            elif stat_type == 'defense':
                self._process_defense_stats(key, stat, play_nullified)
    
    def _process_passing_stats(self, key, stat, play_nullified=False):
        """Process passing statistics"""
        # Skip nullified plays (penalties that negate the play)
        # Check both play-level and stat-level nullification
        if play_nullified or stat.get('nullified', False):
            return
        
        stats = self.player_stats[key]
        
        stats['completions'] = stats.get('completions', 0) + stat.get('complete', 0)
        stats['attempts'] = stats.get('attempts', 0) + stat.get('attempt', 0)
        stats['passing_yards'] = stats.get('passing_yards', 0) + stat.get('yards', 0)
        stats['passing_tds'] = stats.get('passing_tds', 0) + stat.get('touchdown', 0)
        stats['interceptions'] = stats.get('interceptions', 0) + stat.get('interception', 0)
        stats['sacks'] = stats.get('sacks', 0) + stat.get('sack', 0)
        stats['sack_yards'] = stats.get('sack_yards', 0) + stat.get('sack_yards', 0)
        stats['passing_air_yards'] = stats.get('passing_air_yards', 0) + stat.get('att_yards', 0)
        stats['passing_first_downs'] = stats.get('passing_first_downs', 0) + stat.get('firstdown', 0)
        stats['passing_2pt_conversions'] = stats.get('passing_2pt_conversions', 0) + stat.get('two_point_conv', 0)
        
        # Calculate yards after catch
        if stat.get('yards', 0) and stat.get('att_yards', 0):
            yac = stat['yards'] - stat['att_yards']
            stats['passing_yards_after_catch'] = stats.get('passing_yards_after_catch', 0) + yac
    
    def _process_rushing_stats(self, key, stat, play_nullified=False):
        """Process rushing statistics"""
        # Skip nullified plays (penalties that negate the play)
        # Check both play-level and stat-level nullification
        if play_nullified or stat.get('nullified', False):
            return
        
        stats = self.player_stats[key]
        
        stats['carries'] = stats.get('carries', 0) + stat.get('attempt', 0)
        stats['rushing_yards'] = stats.get('rushing_yards', 0) + stat.get('yards', 0)
        stats['rushing_tds'] = stats.get('rushing_tds', 0) + stat.get('touchdown', 0)
        stats['rushing_first_downs'] = stats.get('rushing_first_downs', 0) + stat.get('firstdown', 0)
        stats['rushing_2pt_conversions'] = stats.get('rushing_2pt_conversions', 0) + stat.get('two_point_conv', 0)
    
    def _process_receiving_stats(self, key, stat, play_nullified=False):
        """Process receiving statistics"""
        # Skip nullified plays (penalties that negate the play)
        # Check both play-level and stat-level nullification
        if play_nullified or stat.get('nullified', False):
            return
        
        stats = self.player_stats[key]
        
        stats['receptions'] = stats.get('receptions', 0) + stat.get('reception', 0)
        stats['targets'] = stats.get('targets', 0) + stat.get('target', 0)
        stats['receiving_yards'] = stats.get('receiving_yards', 0) + stat.get('yards', 0)
        stats['receiving_tds'] = stats.get('receiving_tds', 0) + stat.get('touchdown', 0)
        stats['receiving_first_downs'] = stats.get('receiving_first_downs', 0) + stat.get('firstdown', 0)
        stats['receiving_2pt_conversions'] = stats.get('receiving_2pt_conversions', 0) + stat.get('two_point_conv', 0)
        stats['receiving_yards_after_catch'] = stats.get('receiving_yards_after_catch', 0) + stat.get('yards_after_catch', 0)
        
        # Calculate receiving air yards
        if stat.get('yards', 0) and stat.get('yards_after_catch', 0):
            air_yards = stat['yards'] - stat['yards_after_catch']
            stats['receiving_air_yards'] = stats.get('receiving_air_yards', 0) + air_yards
    
    def _process_fumble_stats(self, key, stat, play_nullified=False):
        """Process fumble statistics"""
        # Skip nullified plays (penalties that negate the play)
        # Check both play-level and stat-level nullification
        if play_nullified or stat.get('nullified', False):
            return
        
        stats = self.player_stats[key]
        
        # Determine fumble type based on play context
        if stat.get('lost', 0):
            stats['rushing_fumbles_lost'] = stats.get('rushing_fumbles_lost', 0) + 1
            stats['rushing_fumbles'] = stats.get('rushing_fumbles', 0) + 1
    
    def _process_defense_stats(self, key, stat, play_nullified=False):
        """Process defensive statistics (for reference, not directly used in fantasy)"""
        # Skip nullified plays (penalties that negate the play)
        # Check both play-level and stat-level nullification
        if play_nullified or stat.get('nullified', False):
            return
        
        stats = self.player_stats[key]
        
        # These don't affect offensive fantasy scoring but good to track
        stats['tackles'] = stats.get('tackles', 0) + stat.get('tackle', 0)
        stats['assists'] = stats.get('assists', 0) + stat.get('ast_tackle', 0)
        stats['sacks_made'] = stats.get('sacks_made', 0) + stat.get('sack', 0)
    
    def process_game(self, game_id, season, week):
        """Process all plays from a game"""
        pbp_data = self.get_game_pbp(game_id)
        
        if not pbp_data:
            print(f"  ⚠️  No PBP data for game {game_id[:8]}")
            return
        
        game_info = {
            'game_id': game_id,
            'season': season,
            'week': week
        }
        
        # Process each quarter
        periods = pbp_data.get('periods', [])
        play_count = 0
        
        for period in periods:
            pbp = period.get('pbp', [])
            
            for item in pbp:
                if item.get('type') == 'drive':
                    # Process plays within the drive
                    events = item.get('events', [])
                    for event in events:
                        if event.get('type') == 'play':
                            self.process_play_statistics(event, game_info, week)
                            play_count += 1
        
        print(f"  ✅ Processed {play_count} plays")
    
    def collect_week_data(self, season, week):
        """Collect all data for a specific week"""
        schedule = self.get_weekly_schedule(season, week)
        
        if not schedule:
            print(f"❌ Could not fetch schedule for Week {week}")
            return
        
        games = schedule.get('week', {}).get('games', [])
        print(f"📅 Found {len(games)} games in Week {week}")
        
        for game in tqdm(games, desc=f"Week {week} Games"):
            game_id = game.get('id')
            game_status = game.get('status')
            
            # Only process closed/completed games
            if game_status != 'closed':
                print(f"  ⏭️  Skipping {game_id[:8]} (status: {game_status})")
                continue
            
            self.process_game(game_id, season, week)
    
    def convert_to_dataframe(self):
        """Convert accumulated stats to DataFrame matching nfl_data_py schema"""
        rows = []
        
        for (player_id, season, week, team), stats in self.player_stats.items():
            player_meta = self.player_metadata.get(player_id, {})
            
            row = {
                # Player identifiers
                'player_id': player_id,
                'player_name': player_meta.get('player_name', ''),
                'player_display_name': player_meta.get('player_display_name', ''),
                'position': player_meta.get('position', ''),
                'recent_team': team,
                'season': season,
                'week': week,
                'season_type': 'REG',
                
                # Passing stats
                'completions': stats.get('completions', 0),
                'attempts': stats.get('attempts', 0),
                'passing_yards': stats.get('passing_yards', 0),
                'passing_tds': stats.get('passing_tds', 0),
                'interceptions': stats.get('interceptions', 0),
                'sacks': stats.get('sacks', 0),
                'sack_yards': stats.get('sack_yards', 0),
                'sack_fumbles': stats.get('sack_fumbles', 0),
                'sack_fumbles_lost': stats.get('sack_fumbles_lost', 0),
                'passing_air_yards': stats.get('passing_air_yards', 0),
                'passing_yards_after_catch': stats.get('passing_yards_after_catch', 0),
                'passing_first_downs': stats.get('passing_first_downs', 0),
                'passing_2pt_conversions': stats.get('passing_2pt_conversions', 0),
                
                # Rushing stats
                'carries': stats.get('carries', 0),
                'rushing_yards': stats.get('rushing_yards', 0),
                'rushing_tds': stats.get('rushing_tds', 0),
                'rushing_fumbles': stats.get('rushing_fumbles', 0),
                'rushing_fumbles_lost': stats.get('rushing_fumbles_lost', 0),
                'rushing_first_downs': stats.get('rushing_first_downs', 0),
                'rushing_2pt_conversions': stats.get('rushing_2pt_conversions', 0),
                
                # Receiving stats
                'receptions': stats.get('receptions', 0),
                'targets': stats.get('targets', 0),
                'receiving_yards': stats.get('receiving_yards', 0),
                'receiving_tds': stats.get('receiving_tds', 0),
                'receiving_fumbles': stats.get('receiving_fumbles', 0),
                'receiving_fumbles_lost': stats.get('receiving_fumbles_lost', 0),
                'receiving_air_yards': stats.get('receiving_air_yards', 0),
                'receiving_yards_after_catch': stats.get('receiving_yards_after_catch', 0),
                'receiving_first_downs': stats.get('receiving_first_downs', 0),
                'receiving_2pt_conversions': stats.get('receiving_2pt_conversions', 0),
            }
            
            # Calculate derived metrics
            if row['targets'] > 0:
                row['target_share'] = 0  # Would need team totals
                row['air_yards_share'] = 0  # Would need team totals
                row['wopr'] = 0  # Would need team totals
            
            if row['receiving_yards'] > 0 and row['receiving_air_yards'] > 0:
                row['racr'] = row['receiving_yards'] / max(row['receiving_air_yards'], 1)
            else:
                row['racr'] = 0
            
            if row['passing_yards'] > 0 and row['passing_yards_after_catch'] > 0:
                row['pacr'] = row['passing_yards_after_catch'] / max(row['passing_yards'], 1)
            else:
                row['pacr'] = 0
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Add columns that would come from other sources
        missing_cols = [
            'opponent_team', 'gsis_id', 'position_x', 'position_group', 
            'headshot_url', 'passing_epa', 'rushing_epa', 'receiving_epa',
            'dakota', 'special_teams_tds', 'fantasy_points', 'fantasy_points_ppr',
            'game_id', 'pfr_game_id', 'game_type', 'pfr_player_id'
        ]
        
        for col in missing_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        return df
    
    def save_weekly_data(self, df, season, week, output_dir='data'):
        """Save weekly data to CSV"""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/sportradar_week{week}_{season}.csv"
        df.to_csv(filename, index=False)
        print(f"💾 Saved {len(df)} player records to {filename}")
        return filename


def collect_2025_data(api_key, weeks=[1], output_dir='data'):
    """
    Main function to collect 2025 NFL data
    
    Args:
        api_key: Sportradar API key
        weeks: List of weeks to collect (e.g., [1, 2, 3])
        output_dir: Directory to save CSV files
    
    Returns:
        Combined DataFrame with all weeks
    """
    collector = SportradarNFLCollector(api_key)
    season = 2025
    
    all_weeks_data = []
    
    for week in weeks:
        print(f"\n{'='*60}")
        print(f"COLLECTING WEEK {week} DATA")
        print(f"{'='*60}")
        
        # Collect data for the week
        collector.collect_week_data(season, week)
        
        # Convert to DataFrame
        df_week = collector.convert_to_dataframe()
        
        if not df_week.empty:
            # Save individual week
            collector.save_weekly_data(df_week, season, week, output_dir)
            all_weeks_data.append(df_week)
            
            print(f"\n📊 Week {week} Summary:")
            print(f"  Players: {len(df_week)}")
            print(f"  Teams: {df_week['recent_team'].nunique()}")
            
            # Show top performers
            if 'passing_yards' in df_week.columns:
                top_qb = df_week.nlargest(1, 'passing_yards')
                if not top_qb.empty:
                    qb = top_qb.iloc[0]
                    print(f"  Top QB: {qb['player_name']} ({qb['passing_yards']:.0f} pass yds)")
            
            if 'rushing_yards' in df_week.columns:
                top_rb = df_week.nlargest(1, 'rushing_yards')
                if not top_rb.empty:
                    rb = top_rb.iloc[0]
                    print(f"  Top RB: {rb['player_name']} ({rb['rushing_yards']:.0f} rush yds)")
            
            if 'receiving_yards' in df_week.columns:
                top_wr = df_week.nlargest(1, 'receiving_yards')
                if not top_wr.empty:
                    wr = top_wr.iloc[0]
                    print(f"  Top WR: {wr['player_name']} ({wr['receiving_yards']:.0f} rec yds)")
        
        # Clear stats for next week
        collector.player_stats.clear()
    
    # Combine all weeks
    if all_weeks_data:
        df_combined = pd.concat(all_weeks_data, ignore_index=True)
        combined_file = f"{output_dir}/sportradar_2025_weeks_{min(weeks)}-{max(weeks)}.csv"
        df_combined.to_csv(combined_file, index=False)
        print(f"\n{'='*60}")
        print(f"✅ COLLECTION COMPLETE")
        print(f"{'='*60}")
        print(f"📁 Combined file: {combined_file}")
        print(f"📊 Total records: {len(df_combined)}")
        print(f"📅 Weeks collected: {weeks}")
        
        return df_combined
    
    return pd.DataFrame()


if __name__ == "__main__":
    # Get API key from environment variable
    API_KEY = os.getenv('SPORTRADAR_API_KEY')
    
    if not API_KEY:
        print("ERROR: SPORTRADAR_API_KEY environment variable not set")
        print("Set it with: export SPORTRADAR_API_KEY='your_api_key_here'")
        exit(1)
    
    # Collect Week 1 data
    df = collect_2025_data(
        api_key=API_KEY,
        weeks=[1],  # Add more weeks as they complete: [1, 2, 3]
        output_dir='data'
    )
    
    if not df.empty:
        print("\n🎉 Ready to integrate with your existing pipeline!")
        print("Next steps:")
        print("1. Merge this data with historical data from nfl_data_py")
        print("2. Run your data.py enhancement script")
        print("3. Run model.py for predictions")