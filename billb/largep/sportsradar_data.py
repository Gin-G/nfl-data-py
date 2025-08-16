#!/usr/bin/env python3
"""
Standalone SportRadar Data Export Script
Export all teams and players from SportRadar API to CSV files

Usage:
    python export_sportradar_data.py --api-key YOUR_API_KEY [--output-dir data/]
"""

import requests
import pandas as pd
import argparse
import time
import logging
from typing import Dict, List, Any
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SportRadarExporter:
    """Export SportRadar NFL data to CSV files"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.sportradar.us/nfl/official/trial/v7/en"
        self.session = requests.Session()
        
    def _make_request(self, url: str, max_retries: int = 3) -> requests.Response:
        """Make API request with retry logic"""
        for attempt in range(max_retries):
            try:
                params = {'api_key': self.api_key}
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    logger.warning(f"Rate limited, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise e
    
    def get_teams(self) -> Dict:
        """Get all NFL teams"""
        logger.info("Fetching teams from SportRadar API...")
        url = f"{self.base_url}/league/teams.json"
        response = self._make_request(url)
        data = response.json()
        logger.info(f"Fetched {len(data.get('teams', []))} teams")
        return data
    
    def get_team_roster(self, team_id: str) -> Dict:
        """Get team roster"""
        url = f"{self.base_url}/teams/{team_id}/full_roster.json"
        response = self._make_request(url)
        data = response.json()
        return data
    
    def export_teams(self, output_dir: str) -> str:
        """Export teams to CSV"""
        teams_data = self.get_teams()
        
        teams_list = []
        for team in teams_data.get('teams', []):
            team_record = {
                'sportradar_team_id': team.get('id'),
                'name': team.get('name'),
                'alias': team.get('alias'),
                'market': team.get('market'),
                'conference': team.get('conference'),
                'division': team.get('division'),
                'venue_id': team.get('venue', {}).get('id'),
                'venue_name': team.get('venue', {}).get('name'),
                'venue_city': team.get('venue', {}).get('city'),
                'venue_state': team.get('venue', {}).get('state'),
                'venue_capacity': team.get('venue', {}).get('capacity'),
                'founded': team.get('founded'),
            }
            teams_list.append(team_record)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        teams_file = os.path.join(output_dir, f"sportradar_teams_{timestamp}.csv")
        teams_df = pd.DataFrame(teams_list)
        teams_df.to_csv(teams_file, index=False)
        
        logger.info(f"Exported {len(teams_list)} teams to {teams_file}")
        return teams_file
    
    def export_players(self, output_dir: str) -> str:
        """Export all players to CSV"""
        logger.info("Starting player export...")
        
        # Get teams first
        teams_data = self.get_teams()
        all_players = []
        
        total_teams = len(teams_data.get('teams', []))
        
        for i, team in enumerate(teams_data.get('teams', []), 1):
            team_id = team.get('id')
            team_alias = team.get('alias')
            
            logger.info(f"Processing team {team_alias} ({i}/{total_teams})...")
            
            try:
                roster_data = self.get_team_roster(team_id)
                
                for player in roster_data.get('players', []):
                    player_record = {
                        # Core identification
                        'sportradar_player_id': player.get('id'),
                        'name': player.get('name'),
                        'display_name': player.get('display_name'),
                        'first_name': player.get('first_name'),
                        'last_name': player.get('last_name'),
                        'position': player.get('position'),
                        'jersey': player.get('jersey'),
                        
                        # Physical attributes
                        'height': player.get('height'),
                        'weight': player.get('weight'),
                        'birth_date': player.get('birth_date'),
                        'birth_place': player.get('birth_place'),
                        
                        # Career info
                        'experience': player.get('experience'),
                        'college': player.get('college'),
                        'high_school': player.get('high_school'),
                        
                        # Draft info
                        'draft_year': player.get('draft', {}).get('year') if player.get('draft') else None,
                        'draft_round': player.get('draft', {}).get('round') if player.get('draft') else None,
                        'draft_pick': player.get('draft', {}).get('pick') if player.get('draft') else None,
                        'draft_team': player.get('draft', {}).get('team') if player.get('draft') else None,
                        
                        # Status
                        'status': player.get('status'),
                        'rookie': player.get('rookie'),
                        'years_pro': player.get('years_pro'),
                        
                        # Team info
                        'team_id': team_id,
                        'team_name': team.get('name'),
                        'team_alias': team.get('alias'),
                        'team_market': team.get('market'),
                        
                        # Timestamps
                        'updated': player.get('updated'),
                        'export_timestamp': datetime.now().isoformat()
                    }
                    all_players.append(player_record)
                
                logger.info(f"  Added {len(roster_data.get('players', []))} players from {team_alias}")
                
                # Rate limiting - small delay between teams
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing team {team_alias}: {e}")
                continue
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        players_file = os.path.join(output_dir, f"sportradar_players_{timestamp}.csv")
        players_df = pd.DataFrame(all_players)
        players_df.to_csv(players_file, index=False)
        
        logger.info(f"Exported {len(all_players)} players to {players_file}")
        return players_file
    
    def create_summary(self, teams_file: str, players_file: str) -> None:
        """Create a summary of the exported data"""
        try:
            teams_df = pd.read_csv(teams_file)
            players_df = pd.read_csv(players_file)
            
            print("\n" + "="*60)
            print("SPORTRADAR EXPORT SUMMARY")
            print("="*60)
            
            print(f"\nTEAMS:")
            print(f"  Total: {len(teams_df)}")
            print(f"  Conferences: {teams_df['conference'].value_counts().to_dict()}")
            
            print(f"\nPLAYERS:")
            print(f"  Total: {len(players_df)}")
            print(f"  By Position: {dict(players_df['position'].value_counts().head(10))}")
            print(f"  By Status: {players_df['status'].value_counts().to_dict()}")
            
            print(f"\nFILES CREATED:")
            print(f"  Teams: {teams_file}")
            print(f"  Players: {players_file}")
            
            print(f"\nSAMPLE PLAYERS:")
            sample = players_df[['name', 'position', 'team_alias', 'jersey', 'college']].head(5)
            for _, player in sample.iterrows():
                print(f"  {player['name']} ({player['position']}) - {player['team_alias']} #{player['jersey']} - {player['college']}")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error creating summary: {e}")

def main():
    parser = argparse.ArgumentParser(description='Export SportRadar NFL data to CSV files')
    parser.add_argument('--api-key', required=True, help='SportRadar API key')
    parser.add_argument('--output-dir', default='data', help='Output directory for CSV files')
    parser.add_argument('--teams-only', action='store_true', help='Export only teams (faster)')
    parser.add_argument('--players-only', action='store_true', help='Export only players')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize exporter
    exporter = SportRadarExporter(args.api_key)
    
    teams_file = None
    players_file = None
    
    try:
        if not args.players_only:
            teams_file = exporter.export_teams(args.output_dir)
        
        if not args.teams_only:
            players_file = exporter.export_players(args.output_dir)
        
        # Create summary
        if teams_file and players_file:
            exporter.create_summary(teams_file, players_file)
        
        print(f"\n✅ Export completed successfully!")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())