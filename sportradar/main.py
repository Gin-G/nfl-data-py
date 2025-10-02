import requests
import json
from collections import defaultdict
from typing import Dict, List, Tuple
import time

class NFLTeamSnapAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "accept": "application/json",
            "x-api-key": api_key
        }
        
        # Data storage
        self.team_snap_counts = {}
        self.game_summaries = []
        
        # Play categorization
        self.offensive_stats = {'pass', 'rush', 'receive', 'fumble_recovery'}
        self.defensive_stats = {'defense', 'tackle', 'sack', 'interception', 'fumble_forced'}
        self.special_teams_stats = {'kick', 'punt', 'return', 'block'}
        self.special_teams_plays = {'kickoff', 'punt', 'field_goal', 'extra_point'}

    def get_schedule(self, season: int) -> dict:
        """Get season schedule"""
        url = f"https://api.sportradar.com/nfl/official/trial/v7/en/games/{season}/REG/schedule.json"
        response = requests.get(url, headers=self.headers)
        return response.json()

    def get_play_by_play(self, game_id: str) -> dict:
        """Get play-by-play data for a specific game"""
        url = f"https://api.sportradar.com/nfl/official/trial/v7/en/games/{game_id}/pbp.json"
        response = requests.get(url, headers=self.headers)
        return response.json()

    def categorize_play(self, play: dict) -> str:
        """Determine if play is offensive, defensive, or special teams"""
        play_type = play.get('play_type', '').lower()
        
        if play_type in self.special_teams_plays:
            return 'special_teams'
        elif 'kick' in play_type or 'punt' in play_type:
            return 'special_teams'
        else:
            return 'regular'  # Will be split into offensive/defensive based on stats

    def process_play_statistics(self, play: dict, possession_team_id: str) -> dict:
        """Process a single play's statistics and categorize player participation"""
        play_data = {
            'offensive': defaultdict(list),
            'defensive': defaultdict(list), 
            'special_teams': defaultdict(list)
        }
        
        if 'statistics' not in play:
            return play_data
            
        play_category = self.categorize_play(play)
        
        for stat in play['statistics']:
            if 'player' not in stat or 'team' not in stat:
                continue
                
            player_info = {
                'name': stat['player']['name'],
                'position': stat['player']['position'],
                'jersey': stat['player'].get('jersey', ''),
                'stat_type': stat['stat_type']
            }
            
            team_id = stat['team']['id']
            stat_type = stat['stat_type']
            
            # Categorize based on stat type and play context
            if play_category == 'special_teams':
                play_data['special_teams'][team_id].append(player_info)
            elif stat_type in self.offensive_stats:
                play_data['offensive'][team_id].append(player_info)
            elif stat_type in self.defensive_stats:
                play_data['defensive'][team_id].append(player_info)
            else:
                # Use possession to determine offensive vs defensive
                if team_id == possession_team_id:
                    play_data['offensive'][team_id].append(player_info)
                else:
                    play_data['defensive'][team_id].append(player_info)
        
        return play_data

    def process_game_pbp(self, game_data: dict) -> dict:
        """Process complete game play-by-play data"""
        game_snap_counts = {
            'game_info': {
                'id': game_data.get('id', ''),
                'home_team': game_data.get('summary', {}).get('home', {}),
                'away_team': game_data.get('summary', {}).get('away', {}),
                'final_score': {
                    'home': game_data.get('summary', {}).get('home', {}).get('points', 0),
                    'away': game_data.get('summary', {}).get('away', {}).get('points', 0)
                }
            },
            'teams': {}
        }
        
        # Initialize team data
        if 'summary' in game_data:
            for team_type in ['home', 'away']:
                team = game_data['summary'][team_type]
                team_id = team['id']
                game_snap_counts['teams'][team_id] = {
                    'team_info': team,
                    'offensive_snaps': defaultdict(int),
                    'defensive_snaps': defaultdict(int),
                    'special_teams_snaps': defaultdict(int),
                    'total_plays': {'offensive': 0, 'defensive': 0, 'special_teams': 0}
                }
        
        # Process all plays
        if 'periods' in game_data:
            for period in game_data['periods']:
                if 'pbp' not in period:
                    continue
                    
                for item in period['pbp']:
                    if item.get('type') == 'drive' and 'events' in item:
                        possession_team_id = item.get('offensive_team', {}).get('id', '')
                        
                        for event in item['events']:
                            if event.get('type') == 'play':
                                play_data = self.process_play_statistics(event, possession_team_id)
                                
                                # Update snap counts
                                for phase in ['offensive', 'defensive', 'special_teams']:
                                    for team_id, players in play_data[phase].items():
                                        if team_id in game_snap_counts['teams']:
                                            for player in players:
                                                player_key = f"{player['name']} (#{player['jersey']}) - {player['position']}"
                                                game_snap_counts['teams'][team_id][f'{phase}_snaps'][player_key] += 1
                                            
                                            if players:  # Only count if players were involved
                                                game_snap_counts['teams'][team_id]['total_plays'][phase] += 1
        
        return game_snap_counts

    def calculate_snap_percentages(self, snap_counts: dict, total_plays: int) -> dict:
        """Calculate snap percentages for players"""
        percentages = {}
        for player, count in snap_counts.items():
            percentage = (count / total_plays * 100) if total_plays > 0 else 0
            percentages[player] = {
                'snaps': count,
                'percentage': round(percentage, 1)
            }
        return percentages

    def generate_team_report(self, team_data: dict) -> str:
        """Generate formatted report for a single team"""
        team_info = team_data['team_info']
        report = []
        
        report.append(f"\n{'='*60}")
        report.append(f"{team_info['market']} {team_info['name']} ({team_info['alias']})")
        report.append(f"{'='*60}")
        
        # Offensive snaps
        if team_data['offensive_snaps']:
            total_off_plays = team_data['total_plays']['offensive']
            report.append(f"\n📊 OFFENSIVE SNAP COUNTS (Total Plays: {total_off_plays})")
            report.append("-" * 50)
            
            off_percentages = self.calculate_snap_percentages(
                team_data['offensive_snaps'], total_off_plays
            )
            
            for player, data in sorted(off_percentages.items(), 
                                     key=lambda x: x[1]['snaps'], reverse=True):
                report.append(f"{player:<45} {data['snaps']:>3} ({data['percentage']:>5.1f}%)")
        
        # Defensive snaps
        if team_data['defensive_snaps']:
            total_def_plays = team_data['total_plays']['defensive']
            report.append(f"\n🛡️ DEFENSIVE SNAP COUNTS (Total Plays: {total_def_plays})")
            report.append("-" * 50)
            
            def_percentages = self.calculate_snap_percentages(
                team_data['defensive_snaps'], total_def_plays
            )
            
            for player, data in sorted(def_percentages.items(), 
                                     key=lambda x: x[1]['snaps'], reverse=True):
                report.append(f"{player:<45} {data['snaps']:>3} ({data['percentage']:>5.1f}%)")
        
        # Special teams snaps
        if team_data['special_teams_snaps']:
            total_st_plays = team_data['total_plays']['special_teams']
            report.append(f"\n🏈 SPECIAL TEAMS SNAP COUNTS (Total Plays: {total_st_plays})")
            report.append("-" * 50)
            
            st_percentages = self.calculate_snap_percentages(
                team_data['special_teams_snaps'], total_st_plays
            )
            
            for player, data in sorted(st_percentages.items(), 
                                     key=lambda x: x[1]['snaps'], reverse=True):
                report.append(f"{player:<45} {data['snaps']:>3} ({data['percentage']:>5.1f}%)")
        
        return "\n".join(report)

    def analyze_week(self, season: int, week: int) -> str:
        """Analyze all games in a specific week"""
        print(f"Fetching schedule for {season} season, week {week}...")
        schedule = self.get_schedule(season)
        
        report = []
        report.append(f"NFL SNAP COUNT ANALYSIS - {season} Season, Week {week}")
        report.append("="*80)
        
        # Find the specified week
        target_week = None
        for week_data in schedule.get('weeks', []):
            if week_data['title'] == str(week):
                target_week = week_data
                break
        
        if not target_week:
            return f"Week {week} not found in schedule"
        
        games_processed = 0
        for game in target_week.get('games', []):
            if game.get('status') != 'closed':
                print(f"Skipping {game['away']['name']} @ {game['home']['name']} - Game not completed")
                continue
            
            print(f"\nProcessing: {game['away']['name']} @ {game['home']['name']}")
            
            try:
                # Add delay to respect API rate limits
                time.sleep(1)
                
                pbp_data = self.get_play_by_play(game['id'])
                game_analysis = self.process_game_pbp(pbp_data)
                
                # Add game summary
                game_info = game_analysis['game_info']
                report.append(f"\n🏈 {game_info['away_team']['market']} {game_info['away_team']['name']} {game_info['final_score']['away']} - {game_info['home_team']['market']} {game_info['home_team']['name']} {game_info['final_score']['home']}")
                
                # Generate reports for both teams
                for team_id, team_data in game_analysis['teams'].items():
                    team_report = self.generate_team_report(team_data)
                    report.append(team_report)
                
                games_processed += 1
                
            except Exception as e:
                print(f"Error processing game {game['id']}: {str(e)}")
                continue
        
        report.append(f"\n\nAnalysis completed. {games_processed} games processed.")
        return "\n".join(report)

    def analyze_single_game(self, game_id: str) -> str:
        """Analyze a single game by ID"""
        print(f"Processing game ID: {game_id}")
        
        try:
            pbp_data = self.get_play_by_play(game_id)
            game_analysis = self.process_game_pbp(pbp_data)
            
            report = []
            game_info = game_analysis['game_info']
            report.append(f"NFL SNAP COUNT ANALYSIS")
            report.append(f"🏈 {game_info['away_team']['market']} {game_info['away_team']['name']} {game_info['final_score']['away']} - {game_info['home_team']['market']} {game_info['home_team']['name']} {game_info['final_score']['home']}")
            
            # Generate reports for both teams
            for team_id, team_data in game_analysis['teams'].items():
                team_report = self.generate_team_report(team_data)
                report.append(team_report)
            
            return "\n".join(report)
            
        except Exception as e:
            return f"Error processing game {game_id}: {str(e)}"


# Usage Examples
def main():
    # Initialize analyzer with your API key
    api_key = "MwV55LuOUHtG5pzxvGGjEay1RJsyNvOgpHKyqNVh"
    analyzer = NFLTeamSnapAnalyzer(api_key)
    
    # Example 1: Analyze entire week
    week_report = analyzer.analyze_week(season=2025, week=1)
    print(week_report)
    
    # Example 2: Analyze single game (replace with actual game ID)
    # game_report = analyzer.analyze_single_game("your-game-id-here")
    # print(game_report)
    
    # Example 3: Get just the schedule to see available games
    # schedule = analyzer.get_schedule(2025)
    # for week_data in schedule['weeks']:
    #     if week_data['title'] == '1':
    #         for game in week_data['games']:
    #             print(f"{game['id']}: {game['away']['name']} @ {game['home']['name']} - {game['status']}")

if __name__ == "__main__":
    main()