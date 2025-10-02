#!/usr/bin/env python3
"""
Sportradar Injury API Integration
Automatically fetch and process injury data for predictions
"""

import os
import requests
import time
from collections import defaultdict

class SportradarInjuryAnalyzer:
    """Fetch and analyze injury data from Sportradar API"""
    
    def __init__(self, api_key, season=2025):
        self.api_key = api_key
        self.season = season
        self.base_url = "https://api.sportradar.com/nfl/official/trial/v7/en"
        self.headers = {
            "accept": "application/json",
            "x-api-key": api_key
        }
        
    def fetch_weekly_injuries(self, week):
        """Fetch injury report for a specific week"""
        url = f"{self.base_url}/seasons/{self.season}/REG/{week:02d}/injuries.json"
        
        print(f"Fetching injury data for Week {week}...")
        time.sleep(3)  # Rate limiting
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching injury data: {e}")
            return None
    
    def process_injury_data(self, injury_data):
        """Process injury JSON into structured format"""
        if not injury_data:
            return {}
        
        injuries = {}
        teams = injury_data.get('teams', [])
        
        for team in teams:
            team_alias = team.get('alias', 'UNK')
            players = team.get('players', [])
            
            for player_entry in players:
                player_info = player_entry.get('player', {})
                injury_info = player_entry.get('injury', {})
                
                player_name = player_info.get('name', '')
                position = player_info.get('position', '')
                status = injury_info.get('status', '')
                primary_injury = injury_info.get('primary', '')
                
                if player_name and status:
                    injuries[player_name] = {
                        'status': status.upper(),
                        'injury_type': primary_injury,
                        'position': position,
                        'team': team_alias,
                        'player_id': player_info.get('id', ''),
                        'sportradar_id': player_info.get('sr_id', '')
                    }
        
        return injuries
    
    def categorize_injury_impact(self, injuries_dict, roster_data, depth_charts):
        """Categorize injuries by fantasy impact"""
        
        impact_categories = {
            'zero_out': [],      # OUT/Doubtful - give 0 points
            'boost_backup': [],  # Backups who should get starter projections
            'questionable': []   # Questionable - proceed normally
        }
        
        for player_name, injury_info in injuries_dict.items():
            status = injury_info['status']
            position = injury_info['position']
            team = injury_info['team']
            
            # Categorize based on status
            if status in ['OUT', 'DOUBTFUL']:
                impact_categories['zero_out'].append({
                    'player': player_name,
                    'status': status,
                    'injury': injury_info['injury_type'],
                    'position': position,
                    'team': team
                })
                
                # Find replacement/backup
                backup = self._find_backup_player(
                    team, position, player_name, roster_data, depth_charts
                )
                
                if backup:
                    impact_categories['boost_backup'].append({
                        'player': backup,
                        'replacing': player_name,
                        'reason': f"{player_name} {status.lower()} - {injury_info['injury_type']}",
                        'position': position,
                        'team': team
                    })
            
            elif status == 'QUESTIONABLE':
                impact_categories['questionable'].append({
                    'player': player_name,
                    'injury': injury_info['injury_type'],
                    'position': position,
                    'team': team
                })
        
        return impact_categories
    
    def _find_backup_player(self, team, position, injured_player, roster_data, depth_charts):
        """Find the backup player who should replace injured player"""
        
        # Look in depth charts for next player at that position/team
        team_depth = depth_charts[
            (depth_charts['team'] == team) & 
            (depth_charts['pos_abb'] == position)
        ].sort_values('pos_rank')
        
        # Find injured player's rank
        injured_rank = team_depth[
            team_depth['player_name'].str.contains(injured_player, case=False, na=False)
        ]
        
        if injured_rank.empty:
            return None
        
        injured_pos_rank = injured_rank.iloc[0]['pos_rank']
        
        # Get next player in depth chart
        backup = team_depth[team_depth['pos_rank'] == injured_pos_rank + 1]
        
        if not backup.empty:
            return backup.iloc[0]['player_name']
        
        return None
    
    def create_injury_overrides(self, week, roster_data, depth_charts):
        """Create injury overrides compatible with InjuryStatusAnalyzer"""
        
        # Fetch and process injury data
        injury_json = self.fetch_weekly_injuries(week)
        injuries = self.process_injury_data(injury_json)
        
        print(f"\nProcessed {len(injuries)} injured players for Week {week}")
        
        # Categorize by impact
        impact = self.categorize_injury_impact(injuries, roster_data, depth_charts)
        
        # Create overrides dict
        overrides = {}
        backup_situations = {}
        
        # Players to zero out
        for player_info in impact['zero_out']:
            overrides[player_info['player']] = {
                'status': player_info['status'],
                'reason': player_info['injury'],
                'team': player_info['team'],
                'position': player_info['position']
            }
        
        # Backups getting starter roles
        for backup_info in impact['boost_backup']:
            backup_situations[backup_info['player']] = {
                'replacing': backup_info['replacing'],
                'reason': backup_info['reason'],
                'team': backup_info['team'],
                'position': backup_info['position'],
                'role': 'emergency_starter'
            }
            
            # Add replacement info to injured player's override
            injured = backup_info['replacing']
            if injured in overrides:
                overrides[injured]['replacement'] = backup_info['player']
        
        # Report summary
        print(f"\nInjury Impact Summary:")
        print(f"  Players OUT/Doubtful: {len(impact['zero_out'])}")
        print(f"  Backups elevated: {len(impact['boost_backup'])}")
        print(f"  Questionable (normal projections): {len(impact['questionable'])}")
        
        if impact['zero_out']:
            print("\nPlayers to zero out:")
            for p in impact['zero_out'][:10]:  # Show first 10
                print(f"  {p['player']} ({p['position']}, {p['team']}): {p['status']} - {p['injury']}")
        
        if impact['boost_backup']:
            print("\nBackups getting starter projections:")
            for b in impact['boost_backup'][:10]:  # Show first 10
                print(f"  {b['player']} replacing {b['replacing']} ({b['position']}, {b['team']})")
        
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
    from nfl_data_py import import_weekly_rosters, import_depth_charts
    
    API_KEY = os.getenv('SPORTRADAR_API_KEY')
    
    if not API_KEY:
        print("Set SPORTRADAR_API_KEY environment variable")
        exit(1)
    
    # Load required data
    print("Loading roster and depth chart data...")
    rosters = import_weekly_rosters([2025])
    depth_charts = import_depth_charts([2025])
    
    # Get injury data for Week 5
    overrides, backups = integrate_sportradar_injuries(5, rosters, depth_charts)
    
    print(f"\n{'='*70}")
    print("FINAL OUTPUT")
    print(f"{'='*70}")
    print(f"\nInjury Overrides: {len(overrides)} players")
    print(f"Backup Situations: {len(backups)} players")
    
    # Show sample
    if overrides:
        print("\nSample injury overrides:")
        for player, info in list(overrides.items())[:5]:
            print(f"  {player}: {info}")
    
    if backups:
        print("\nSample backup situations:")
        for player, info in list(backups.items())[:5]:
            print(f"  {player}: {info}")