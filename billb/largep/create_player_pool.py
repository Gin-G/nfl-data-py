import pandas as pd
import json
import os
from tqdm import tqdm
from nfl_data_py import import_schedules, import_weekly_rosters

def load_current_rosters(season):
    """Load current roster data to get up-to-date team assignments"""
    print(f"📋 Loading current rosters for {season}...")
    
    try:
        roster_df = import_weekly_rosters([season])
        
        # Filter for active players only
        active_roster = roster_df[roster_df['status'] == 'ACT'].copy()
        
        if active_roster.empty:
            print(f"   ⚠️ No active players found in {season} roster")
            return {}
        
        print(f"   ✅ Found {len(active_roster)} active players in current rosters")
        
        # Debug: Check the column names
        print(f"   🔍 Roster columns: {list(roster_df.columns)}")
        
        # Create player-to-team mapping
        current_teams = {}
        valid_names = 0
        
        for _, player in active_roster.iterrows():
            # Try different name column variations that might exist
            possible_name_cols = ['display_name', 'full_name', 'player_name', 'name', 'player_display_name']
            player_name = None
            
            for col in possible_name_cols:
                if col in player and pd.notna(player.get(col)) and str(player.get(col)).strip():
                    player_name = str(player.get(col)).strip()
                    break
            
            team = str(player.get('team', '')).strip()
            position = str(player.get('position', '')).strip()
            
            if player_name and team and len(player_name) > 2:
                valid_names += 1
                
                # Store multiple name variations
                names_to_store = [player_name.lower().strip()]
                
                # Add variations if different columns exist
                for col in possible_name_cols:
                    if col in player and pd.notna(player.get(col)):
                        alt_name = str(player.get(col)).lower().strip()
                        if alt_name and alt_name not in names_to_store:
                            names_to_store.append(alt_name)
                
                # Add first + last name combination if available
                first_name = str(player.get('first_name', '')).strip()
                last_name = str(player.get('last_name', '')).strip()
                if first_name and last_name:
                    full_name = f"{first_name} {last_name}".lower().strip()
                    if full_name not in names_to_store:
                        names_to_store.append(full_name)
                
                for name in names_to_store:
                    if name and len(name) > 2:
                        current_teams[name] = {
                            'current_team': team,
                            'position': position,
                            'status': player.get('status', 'ACT'),
                            'jersey_number': player.get('jersey_number', ''),
                            'roster_name': player_name
                        }
        
        print(f"   📊 Found {valid_names} players with valid names")
        print(f"   📊 Created current team mapping for {len(current_teams)} player name variations")
        
        # Debug: Show a few examples
        if current_teams:
            print("   🔍 Sample roster entries:")
            for i, (name, info) in enumerate(list(current_teams.items())[:3]):
                print(f"      {name} -> {info['current_team']} ({info['position']})")
        
        return current_teams
        
    except Exception as e:
        print(f"   ❌ Error loading current rosters: {e}")
        import traceback
        traceback.print_exc()
        return {}

def load_schedule_data(season, week):
    """Load and process schedule data for game matchups"""
    print(f"📅 Loading schedule data for {season} Week {week}...")
    
    try:
        schedule_df = import_schedules([season])
        
        # Filter for the specific week
        week_schedule = schedule_df[
            (schedule_df['season'] == season) & 
            (schedule_df['week'] == week) &
            (schedule_df['game_type'] == 'REG')
        ].copy()
        
        if week_schedule.empty:
            print(f"   ⚠️ No games found for {season} Week {week}")
            return {}
        
        print(f"   ✅ Found {len(week_schedule)} games for Week {week}")
        
        # Create team matchup lookup
        matchup_data = {}
        
        for _, game in week_schedule.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Format game time
            gameday = pd.to_datetime(game['gameday'])
            game_time_formatted = gameday.strftime('%a %m/%d')
            
            # Add time if available
            if pd.notna(game.get('gametime')):
                game_time_formatted += f" {game['gametime']}"
            else:
                game_time_formatted += " TBD"
            
            # Home team info
            matchup_data[home_team] = {
                'opponent_team': away_team,
                'home_away': 'home',
                'game_time': game_time_formatted,
                'gameday': gameday.strftime('%Y-%m-%d'),
                'stadium': game.get('stadium', ''),
                'game_id': game.get('game_id', '')
            }
            
            # Away team info  
            matchup_data[away_team] = {
                'opponent_team': home_team,
                'home_away': 'away',
                'game_time': game_time_formatted,
                'gameday': gameday.strftime('%Y-%m-%d'),
                'stadium': game.get('stadium', ''),
                'game_id': game.get('game_id', '')
            }
        
        return matchup_data
        
    except Exception as e:
        print(f"   ❌ Error loading schedule: {e}")
        return {}

def create_team_mapping():
    """Create mapping between different team name formats"""
    team_mapping = {
        'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Pittsburgh Steelers': 'PIT',
        'Baltimore Ravens': 'BAL', 'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND',
        'Jacksonville Jaguars': 'JAX', 'Tennessee Titans': 'TEN', 'Buffalo Bills': 'BUF',
        'Miami Dolphins': 'MIA', 'New England Patriots': 'NE', 'New York Jets': 'NYJ',
        'Denver Broncos': 'DEN', 'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV',
        'Los Angeles Chargers': 'LAC', 'Dallas Cowboys': 'DAL', 'New York Giants': 'NYG',
        'Philadelphia Eagles': 'PHI', 'Washington Commanders': 'WAS', 'Chicago Bears': 'CHI',
        'Detroit Lions': 'DET', 'Green Bay Packers': 'GB', 'Minnesota Vikings': 'MIN',
        'Atlanta Falcons': 'ATL', 'Carolina Panthers': 'CAR', 'New Orleans Saints': 'NO',
        'Tampa Bay Buccaneers': 'TB', 'Arizona Cardinals': 'ARI', 'Los Angeles Rams': 'LA',
        'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA'
    }
    
    # Add reverse mapping and self-mapping
    reverse_mapping = {v: k for k, v in team_mapping.items()}
    team_mapping.update(reverse_mapping)
    for abbr in list(team_mapping.values()):
        if isinstance(abbr, str) and len(abbr) <= 3:
            team_mapping[abbr] = abbr
    
    return team_mapping

def find_current_team_for_player(player_name, current_teams):
    """Find the current team for a player by checking multiple name variations"""
    if not player_name:
        return None
    
    # Normalize the player name
    name_clean = str(player_name).lower().strip()
    
    # Check exact match first
    if name_clean in current_teams:
        return current_teams[name_clean]['current_team']
    
    # Check for partial matches if exact match fails
    # First try: see if the prediction name is contained in any roster name
    for roster_name, roster_info in current_teams.items():
        if name_clean in roster_name or roster_name in name_clean:
            return roster_info['current_team']
    
    # Second try: Handle common name variations (e.g., "Joe" vs "Joseph")
    name_parts = name_clean.split()
    if len(name_parts) >= 2:
        # Try just first and last name
        simplified_name = f"{name_parts[0]} {name_parts[-1]}"
        if simplified_name in current_teams:
            return current_teams[simplified_name]['current_team']
        
        # Try matching by last name + first letter of first name
        for roster_name, roster_info in current_teams.items():
            roster_parts = roster_name.split()
            if len(roster_parts) >= 2:
                # Check if last names match and first names start with same letter
                if (name_parts[-1] == roster_parts[-1] and 
                    name_parts[0][0] == roster_parts[0][0]):
                    return roster_info['current_team']
    
    return None

def create_frontend_player_pools(season=2025, week=1, 
                                predictions_dir="predictions",
                                output_dir="player_pools"):
    """
    Create player pools in the exact format expected by the Svelte frontend
    Focuses only on the columns needed by the lineup picker
    """
    
    print(f"Creating FRONTEND-READY player pools for {season} Week {week}")
    print("="*70)
    
    # Load current roster and schedule data
    current_teams = load_current_rosters(season)
    matchup_data = load_schedule_data(season, week)
    team_mapping = create_team_mapping()
    
    # Process predictions for each position
    positions = ['QB', 'RB', 'WR', 'TE']
    os.makedirs(output_dir, exist_ok=True)
    total_players_created = 0
    
    for position in positions:
        print(f"\n🏈 Processing {position}...")
        
        # Try multiple prediction file patterns
        pred_files = [
            f"{position}_improved_predictions_week{week}_season{season}.csv",
            f"{position}_predictions_week{week}_season{season}.csv",
            f"{position}_enhanced_predictions_week{week}_season{season}.csv"
        ]
        
        pred_path = None
        for pred_file in pred_files:
            test_path = os.path.join(predictions_dir, pred_file)
            if os.path.exists(test_path):
                pred_path = test_path
                break
        
        if not pred_path:
            print(f"  ❌ No prediction file found for {position}")
            continue
        
        # Load predictions
        df_pred = pd.read_csv(pred_path)
        print(f"  📄 Loaded {len(df_pred)} predictions from {os.path.basename(pred_path)}")
        print(f"  🔍 Prediction columns: {list(df_pred.columns)}")
        
        # Create player pool
        frontend_players = []
        team_corrections = 0
        
        for _, pred in df_pred.iterrows():
            # Get current team assignment
            current_team = find_current_team_for_player(pred['player_name'], current_teams)
            
            if current_team:
                final_team = current_team
                team_corrections += 1
                print(f"    🔄 {pred['player_name']}: assigned to {current_team}")
            else:
                # No current team found, need to handle this
                final_team = ""  # Will show as TBD in frontend
                print(f"    ⚠️ {pred['player_name']}: no current team found")
            
            # Get projected points from prediction data
            projected_points = pred.get('fanduel_fantasy_points', 0) or 0
            
            # Calculate salary (price) based on projected points
            salary = max(4000, round(int(projected_points * 500) / 100) * 100)
            
            # Get opponent and schedule info
            team_key = team_mapping.get(final_team, final_team)
            matchup_info = matchup_data.get(team_key, {})
            
            opponent = matchup_info.get('opponent_team', '')
            home_away = matchup_info.get('home_away', '')
            
            if opponent and home_away:
                opponent_display = f"vs {opponent}" if home_away == 'home' else f"@ {opponent}"
            else:
                opponent_display = "TBD"
            
            # Create frontend-compatible player object
            frontend_player = {
                # Essential IDs
                'id': f"{pred['player_name'].lower().replace(' ', '_')}_{final_team}_{season}",
                'sportradar_player_id': '',  # We don't have this from predictions alone
                
                # Names (frontend uses both)
                'player_name': pred['player_name'],
                'name': pred['player_name'],
                
                # Position
                'position': pred.get('position', position),
                
                # Pricing (frontend expects both)
                'price': salary,
                'salary': salary,
                
                # Fantasy projections (frontend expects both)
                'projected_points': round(projected_points, 1),
                'avg_fppg': round(projected_points, 1),
                
                # Team info (required for display)
                'team': final_team,
                'recent_team': final_team,
                
                # Opponent info (required for matchup display)
                'opponent_team': opponent,
                'opponent_display': opponent_display,
                'home_away': home_away,
                
                # Game timing (required for frontend display)
                'game_time': matchup_info.get('game_time', 'TBD'),
                'gameday': matchup_info.get('gameday', ''),
                'stadium': matchup_info.get('stadium', ''),
                'game_id': matchup_info.get('game_id', ''),
                
                # Player details
                'headshot_url': '',  # We don't have this from predictions
                'jersey_number': '',
                'roster_status': 'ACT',
                
                # Ranking and confidence
                'rank': pred.get('rank', len(frontend_players) + 1),
                'prediction_rank': pred.get('rank', len(frontend_players) + 1),
                'confidence': (pred.get('vs_season_avg', 0) / 100) if pd.notna(pred.get('vs_season_avg')) else 0
            }
            
            frontend_players.append(frontend_player)
        
        # Sort by projected points (descending) and assign ranks
        if frontend_players:
            df_frontend = pd.DataFrame(frontend_players)
            df_frontend = df_frontend.sort_values('projected_points', ascending=False)
            df_frontend['rank'] = range(1, len(df_frontend) + 1)
            df_frontend['prediction_rank'] = df_frontend['rank']
            
            # Save to CSV
            output_file = f"{position}_player_pool_{season}_week{week}.csv"
            output_path = os.path.join(output_dir, output_file)
            df_frontend.to_csv(output_path, index=False)
            
            total_players_created += len(frontend_players)
            
            print(f"  ✅ {position}: {len(frontend_players)} players saved")
            print(f"     Team assignments: {team_corrections}/{len(frontend_players)}")
            print(f"     Price range: ${df_frontend['salary'].min():,} - ${df_frontend['salary'].max():,}")
            
            # Show top 3
            print(f"     Top 3:")
            for _, player in df_frontend.head(3).iterrows():
                team_display = player['team'] if player['team'] else 'UNK'
                matchup = player['opponent_display']
                game_time = player['game_time']
                print(f"       {player['rank']}. {player['player_name']} ({team_display}) - ${player['salary']:,} ({player['projected_points']} pts) - {matchup} {game_time}")
    
    # Create combined file
    create_combined_frontend_file(season, week, output_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ FRONTEND-READY player pools complete!")
    print(f"📊 Total players created: {total_players_created}")
    print(f"📁 Files saved to: {output_dir}/")
    
    return True

def create_combined_frontend_file(season, week, output_dir):
    """Create combined file with all positions in frontend format"""
    positions = ['QB', 'RB', 'WR', 'TE']
    all_players = []
    
    for position in positions:
        file_path = os.path.join(output_dir, f"{position}_player_pool_{season}_week{week}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_players.append(df)
    
    if all_players:
        df_combined = pd.concat(all_players, ignore_index=True)
        
        # Sort by position priority, then by projected points
        position_order = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 4}
        df_combined['position_order'] = df_combined['position'].map(position_order)
        df_combined = df_combined.sort_values(['position_order', 'projected_points'], ascending=[True, False])
        df_combined = df_combined.drop('position_order', axis=1)
        
        # Save combined file
        combined_file = f"all_players_pool_{season}_week{week}.csv"
        combined_path = os.path.join(output_dir, combined_file)
        df_combined.to_csv(combined_path, index=False)
        
        print(f"\n📋 Combined frontend file: {combined_file}")
        print(f"   Total players: {len(df_combined)}")
        
        position_counts = df_combined['position'].value_counts()
        for pos in ['QB', 'RB', 'WR', 'TE']:
            if pos in position_counts:
                print(f"   {pos}: {position_counts[pos]} players")
        
        print(f"   Salary range: ${df_combined['salary'].min():,} - ${df_combined['salary'].max():,}")
        print(f"   Average salary: ${df_combined['salary'].mean():,.0f}")
        
        # Show sample with teams assigned
        players_with_teams = df_combined[df_combined['team'] != '']
        if not players_with_teams.empty:
            print(f"\n📅 Sample players with team assignments:")
            for _, player in players_with_teams.head(3).iterrows():
                print(f"   {player['player_name']} ({player['team']}) {player['opponent_display']} - {player['game_time']}")

if __name__ == "__main__":
    success = create_frontend_player_pools()
    
    if success:
        print("\n" + "="*70)
        print("🎯 SVELTE FRONTEND INTEGRATION READY!")
        print("\n✅ Features included:")
        print("   • All columns expected by lineup picker")
        print("   • Current 2025 team assignments") 
        print("   • Opponent matchups and game times")
        print("   • Calculated salaries based on projections")
        print("   • Proper ranking and confidence scores")
        print("\n🚀 Ready to load directly into your Svelte app!")
        print("📝 Import this CSV into your frontend player pool loader")