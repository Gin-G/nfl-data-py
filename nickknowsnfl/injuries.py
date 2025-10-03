#!/usr/bin/env python3
"""
Enhanced Sportradar Injury API Integration with Debugging
Automatically fetch and process injury data for predictions
"""

import os
import requests
import time
import json
from collections import defaultdict

class SportradarInjuryAnalyzer:
    """Fetch and analyze injury data from Sportradar API"""
    
    def __init__(self, api_key, season=2025):
        self.api_key = api_key
        self.season = season
        self.base_url = "https://api.sportradar.com/nfl/official/trial/v7/en"
        self.headers = {
            "accept": "application/json"
        }
        
    def fetch_weekly_injuries(self, week):
        """Fetch injury report for a specific week"""
        url_patterns = [
            f"{self.base_url}/seasons/{self.season}/REG/{week:02d}/injuries.json?api_key={self.api_key}",
            f"{self.base_url}/seasons/{self.season}/REG/injuries.json?api_key={self.api_key}",
            f"{self.base_url}/league/{self.season}/REG/{week}/injuries.json?api_key={self.api_key}",
        ]
        
        print(f"\n{'='*70}")
        print(f"Fetching injury data for Week {week}...")
        print(f"{'='*70}")
        
        for i, url in enumerate(url_patterns, 1):
            print(f"\nAttempt {i}: Trying URL pattern...")
            safe_url = url.replace(self.api_key, "***API_KEY***")
            print(f"URL: {safe_url}")
            
            time.sleep(1)
            
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                
                print(f"Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    print("✅ Success! Received data")
                    data = response.json()
                    
                    return data
                    
                elif response.status_code == 404:
                    print(f"⚠️  404 Not Found - trying next URL pattern")
                    continue
                    
                elif response.status_code == 401:
                    print(f"❌ 401 Unauthorized - check your API key")
                    print(f"Response: {response.text[:200]}")
                    return None
                    
                elif response.status_code == 403:
                    print(f"❌ 403 Forbidden - API key may not have access to this endpoint")
                    print(f"Response: {response.text[:200]}")
                    return None
                    
                else:
                    print(f"⚠️  Unexpected status code: {response.status_code}")
                    print(f"Response: {response.text[:500]}")
                    continue
                    
            except requests.exceptions.Timeout:
                print(f"❌ Request timed out")
                continue
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Error: {e}")
                continue
                
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON: {e}")
                print(f"Raw response: {response.text[:500]}")
                continue
        
        print(f"\n❌ All URL patterns failed")
        return None
    
    def fetch_ir_players(self, roster_data):
        """Extract IR/PUP players from roster data"""
        print(f"\n{'='*70}")
        print("CHECKING ROSTER FOR IR/PUP PLAYERS")
        print(f"{'='*70}")
        
        ir_pup_players = {}
        
        # Filter for inactive statuses
        inactive_statuses = ['RES', 'IR', 'PUP', 'SUS', 'NON']
        
        if 'status' not in roster_data.columns:
            return ir_pup_players
        
        inactive_roster = roster_data[roster_data['status'].isin(inactive_statuses)]
        
        # Deduplicate by player name (roster has multiple week entries per player)
        inactive_roster = inactive_roster.drop_duplicates(subset=['player_name', 'team', 'position'], keep='first')
        
        # Focus on offensive skill positions
        skill_positions = ['QB', 'RB', 'WR', 'TE']
        skill_inactive = inactive_roster[inactive_roster['position'].isin(skill_positions)]
        
        for _, player in skill_inactive.iterrows():
            player_name = player.get('player_name', player.get('display_name', ''))
            if not player_name:
                continue
                
            ir_pup_players[player_name] = {
                'status': 'OUT',  # IR/PUP = definitely out
                'injury_type': f"{player['status']} list",
                'position': player['position'],
                'team': player.get('team', 'UNK'),
                'player_id': player.get('player_id', ''),
                'sportradar_id': '',
                'practice_status': 'IR/PUP/Reserve'
            }
            
        return ir_pup_players
    
    def merge_injury_sources(self, api_injuries, ir_injuries):
        """Merge injuries from API and roster IR/PUP data"""
        print(f"\n{'='*70}")
        print("MERGING INJURY SOURCES")
        print(f"{'='*70}")
        
        # Start with API injuries (most up to date game status)
        merged = api_injuries.copy()
        
        # Add IR/PUP players not already in API data
        added_count = 0
        for player_name, injury_info in ir_injuries.items():
            if player_name not in merged:
                merged[player_name] = injury_info
                added_count += 1
        
        print(f"API injuries: {len(api_injuries)}")
        print(f"IR/PUP players: {len(ir_injuries)}")
        print(f"Added from IR/PUP: {added_count}")
        print(f"Total merged: {len(merged)}")
        
        return merged
    
    def fetch_season_injuries(self):
        """Fetch all injuries for the season (alternative endpoint)"""
        url = f"{self.base_url}/injuries.json?api_key={self.api_key}"
        
        print(f"\nTrying season-wide injury endpoint...")
        safe_url = url.replace(self.api_key, "***API_KEY***")
        print(f"URL: {safe_url}")
        
        time.sleep(1)
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ Success!")
                data = response.json()
                print(f"Response keys: {list(data.keys())}")
                return data
            else:
                print(f"Response: {response.text[:500]}")
                
        except Exception as e:
            print(f"Error: {e}")
        
        return None
    
    def process_injury_data(self, injury_data):
        """Process injury JSON into structured format"""
        if not injury_data:
            print("❌ No injury data to process")
            return {}
        
        injuries = {}
        
        teams = injury_data.get('teams', [])
        
        if not teams:
            print("⚠️  No 'teams' key in injury data")
            print(f"Available keys: {list(injury_data.keys())}")
            
            if 'week' in injury_data:
                teams = injury_data['week'].get('teams', [])
        
        for team in teams:
            team_alias = team.get('alias', team.get('market', 'UNK'))
            team_name = team.get('name', '')
            players = team.get('players', [])
            
            for player_entry in players:
                
                player_name = (player_entry.get('name') or 
                              player_entry.get('full_name') or 
                              player_entry.get('preferred_name', ''))
                
                position = player_entry.get('position', '')
                player_id = player_entry.get('id', '')
                sr_id = player_entry.get('sr_id', '')
                
                injuries_list = player_entry.get('injuries', [])
                
                if not injuries_list:
                    continue
                
                for injury_info in injuries_list:
                    # Get game status
                    status = (injury_info.get('status') or 
                             injury_info.get('game_status', ''))
                    
                    # Get practice status for inference
                    practice_info = injury_info.get('practice', {})
                    practice_status = practice_info.get('status', '') if isinstance(practice_info, dict) else ''
                    
                    # Infer game status from practice participation if no game status
                    if not status and practice_status:
                        if 'Did Not Participate' in practice_status:
                            status = 'QUESTIONABLE'  # Conservative - might be OUT
                        elif 'Limited' in practice_status:
                            status = 'QUESTIONABLE'
                        # Full participation = likely playing, don't mark as injury
                    
                    primary_injury = (injury_info.get('primary') or 
                                    injury_info.get('description') or
                                    injury_info.get('injury') or
                                    injury_info.get('comment', ''))
                    
                    # Only include if we have a status (game or inferred from practice)
                    if player_name and status:
                        player_key = player_name
                        if player_key in injuries:
                            injuries[player_key]['injury_type'] += f", {primary_injury}"
                        else:
                            injuries[player_key] = {
                                'status': status.upper(),
                                'injury_type': primary_injury or 'Not specified',
                                'position': position or 'UNK',
                                'team': team_alias,
                                'player_id': player_id,
                                'sportradar_id': sr_id,
                                'practice_status': practice_status
                            }
                    
                    break
        
        print(f"\n✅ Processed {len(injuries)} total injured players")
        return injuries
    
    def categorize_injury_impact(self, injuries_dict, roster_data, depth_charts):
        """Categorize injuries by fantasy impact"""
        
        impact_categories = {
            'zero_out': [],
            'boost_backup': [],
            'questionable': []
        }
        
        print(f"\n{'='*70}")
        print("CATEGORIZING INJURY IMPACT")
        print(f"{'='*70}")
        
        # First pass: categorize all injuries and build injured players set
        out_doubtful_players = set()
        
        for player_name, injury_info in injuries_dict.items():
            status = injury_info['status']
            position = injury_info['position']
            team = injury_info['team']
            
            if status in ['OUT', 'DOUBTFUL', 'D']:
                out_doubtful_players.add(player_name.lower())
                impact_categories['zero_out'].append({
                    'player': player_name,
                    'status': status,
                    'injury': injury_info['injury_type'],
                    'position': position,
                    'team': team
                })
            
            elif status in ['QUESTIONABLE', 'Q']:
                impact_categories['questionable'].append({
                    'player': player_name,
                    'injury': injury_info['injury_type'],
                    'position': position,
                    'team': team
                })
        
        # Second pass: find healthy backups for injured players
        for injured_info in impact_categories['zero_out']:
            player_name = injured_info['player']
            position = injured_info['position']
            team = injured_info['team']
            
            backup = self._find_healthy_backup(
                team, position, player_name, roster_data, depth_charts, out_doubtful_players
            )
            
            if backup:
                # VERIFY THE BACKUP IS ON THE SAME TEAM
                backup_team = depth_charts[
                    depth_charts['player_name'].str.contains(backup, case=False, na=False)
                ]['team'].iloc[0] if len(depth_charts[
                    depth_charts['player_name'].str.contains(backup, case=False, na=False)
                ]) > 0 else None
                
                if backup_team == team:  # ONLY add if same team
                    impact_categories['boost_backup'].append({
                        'player': backup,
                        'replacing': player_name,
                        'reason': f"{player_name} {injured_info['status'].lower()} - {injured_info['injury']}",
                        'position': position,
                        'team': team
                    })
                else:
                    pass
                    
        return impact_categories
    
    def _find_healthy_backup(self, team, position, injured_player, roster_data, depth_charts, out_doubtful_players):
        """Find a HEALTHY backup player who should replace injured player"""
        
        team_depth = depth_charts[
            (depth_charts['team'] == team) & 
            (depth_charts['pos_abb'] == position)
        ].sort_values('pos_rank')
        
        if team_depth.empty:
            return None
        
        # Find injured player's rank using partial name matching
        injured_rank = team_depth[
            team_depth['player_name'].str.contains(injured_player, case=False, na=False)
        ]
        
        if injured_rank.empty:
            injured_parts = injured_player.lower().split()
            if len(injured_parts) >= 2:
                last_name = injured_parts[-1]
                injured_rank = team_depth[
                    team_depth['player_name'].str.lower().str.contains(last_name, na=False)
                ]
        
        if injured_rank.empty:
            return None
        
        injured_pos_rank = injured_rank.iloc[0]['pos_rank']
        injured_player_name = injured_rank.iloc[0]['player_name']
        
        # Get all players ranked below the injured player
        potential_backups = team_depth[
            (team_depth['pos_rank'] > injured_pos_rank) &
            (team_depth['player_name'] != injured_player_name)
        ].sort_values('pos_rank')
        
        # Find the first HEALTHY backup
        checked_players = set()
        for _, backup_row in potential_backups.iterrows():
            backup_name = backup_row['player_name']
            backup_lower = backup_name.lower()
            
            if backup_lower in checked_players:
                continue
            checked_players.add(backup_lower)
            
            if backup_lower in out_doubtful_players:
                continue
            
            return backup_name
        
        return None
    
    def create_injury_overrides(self, week, roster_data, depth_charts):
        """Create injury overrides compatible with InjuryStatusAnalyzer"""
        
        # Fetch API injury data
        injury_json = self.fetch_weekly_injuries(week)
        
        if not injury_json:
            print("\nWeekly endpoint failed, trying season-wide endpoint...")
            injury_json = self.fetch_season_injuries()
        
        # Get IR/PUP players from roster
        ir_injuries = self.fetch_ir_players(roster_data)
        
        # Process API injuries
        api_injuries = {}
        if injury_json:
            api_injuries = self.process_injury_data(injury_json)
        else:
            print("\n⚠️  Could not fetch API injury data")
        
        # Merge both sources
        all_injuries = self.merge_injury_sources(api_injuries, ir_injuries)
        
        if not all_injuries:
            print("\n❌ No injuries found from any source")
            return {}, {}
        
        impact = self.categorize_injury_impact(all_injuries, roster_data, depth_charts)
        
        overrides = {}
        backup_situations = {}
        
        for player_info in impact['zero_out']:
            overrides[player_info['player']] = {
                'status': player_info['status'],
                'reason': player_info['injury'],
                'team': player_info['team'],
                'position': player_info['position']
            }
        
        for backup_info in impact['boost_backup']:
            backup_situations[backup_info['player']] = {
                'replacing': backup_info['replacing'],
                'reason': backup_info['reason'],
                'team': backup_info['team'],
                'position': backup_info['position'],
                'role': 'emergency_starter'
            }
            
            injured = backup_info['replacing']
            if injured in overrides:
                overrides[injured]['replacement'] = backup_info['player']
        
        # Report summary
        print(f"\n{'='*70}")
        print("INJURY IMPACT SUMMARY")
        print(f"{'='*70}")
        print(f"Players OUT/Doubtful: {len(impact['zero_out'])}")
        print(f"Backups elevated: {len(impact['boost_backup'])}")
        print(f"Questionable (normal projections): {len(impact['questionable'])}")
        
        return overrides, backup_situations

def integrate_sportradar_injuries(week, roster_data, depth_charts):
    """
    Main integration function - call this from model.py
    
    Returns:
        tuple: (injury_overrides_dict, backup_situations_dict)
    """
    api_key = os.getenv('SPORTRADAR_API_KEY')
    
    if not api_key:
        print("WARNING: SPORTRADAR_API_KEY not set, using empty injury data")
        return {}, {}
    
    analyzer = SportradarInjuryAnalyzer(api_key)
    overrides, backups = analyzer.create_injury_overrides(week, roster_data, depth_charts)
    
    return overrides, backups


# Example usage
if __name__ == "__main__":
    import nflreadpy as nfl
    
    API_KEY = os.getenv('SPORTRADAR_API_KEY')
    
    if not API_KEY:
        print("❌ Set SPORTRADAR_API_KEY environment variable")
        print("\nExample:")
        print("  export SPORTRADAR_API_KEY='your_key_here'")
        exit(1)
    
    print(f"✅ API Key found: {API_KEY[:10]}...")
    
    # Load required data
    print("\n📊 Loading roster and depth chart data...")
    rosters = nfl.load_rosters_weekly(seasons=[2025])
    depth_charts = nfl.load_depth_charts(seasons=[2025])
    
    # Convert from Polars to Pandas if needed
    if hasattr(rosters, 'to_pandas'):
        rosters = rosters.to_pandas()
    if hasattr(depth_charts, 'to_pandas'):
        depth_charts = depth_charts.to_pandas()
    
    print(f"  Rosters: {len(rosters)} players")
    print(f"  Depth charts: {len(depth_charts)} entries")
    
    # Get injury data for Week 5
    overrides, backups = integrate_sportradar_injuries(5, rosters, depth_charts)
    
    print(f"\n{'='*70}")
    print("FINAL OUTPUT")
    print(f"{'='*70}")
    print(f"\nInjury Overrides: {len(overrides)} players")
    print(f"Backup Situations: {len(backups)} players")
    
    if overrides:
        print("\n📋 Sample injury overrides:")
        for player, info in list(overrides.items())[:5]:
            print(f"  {player}: {info}")
    
    if backups:
        print("\n📋 Sample backup situations:")
        for player, info in list(backups.items())[:5]:
            print(f"  {player}: {info}")
    
    if not overrides and not backups:
        print("\n⚠️  No injury data found. Possible reasons:")
        print("  1. Wrong API endpoint URL")
        print("  2. Week 5 data not yet available")
        print("  3. API key doesn't have access to injury data")
        print("  4. Trial API has limited data")
        print("\nCheck Sportradar documentation for correct endpoints:")