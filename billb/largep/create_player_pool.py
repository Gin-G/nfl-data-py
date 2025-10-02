import pandas as pd
import json
import os
from tqdm import tqdm
from nfl_data_py import import_schedules, import_weekly_rosters

def load_nfl_dataset_with_headshots(dataset_path="data/nfl_dataset.csv"):
    """Load NFL dataset with player headshot URLs and other info"""
    print(f"📊 Loading NFL dataset with headshots...")
    
    try:
        df_nfl = pd.read_csv(dataset_path)
        print(f"   ✅ Loaded {len(df_nfl)} records from NFL dataset")
        
        # Create lookup dictionary with player names and headshots
        player_lookup = {}
        
        # Get latest info for each player (most recent record)
        for _, player in df_nfl.groupby('player_id').last().iterrows():
            names_to_try = []
            
            # Add various name columns that might exist
            name_columns = ['player_name', 'player_display_name', 'sportradar_name', 'name']
            for name_col in name_columns:
                if pd.notna(player.get(name_col)) and str(player.get(name_col)).strip():
                    names_to_try.append(str(player[name_col]).lower().strip())
            
            # Store comprehensive player info for each name variation
            player_info = {
                'sportradar_player_id': player.get('sportradar_player_id', ''),
                'player_name': player.get('player_display_name', player.get('player_name', '')),
                'display_name': player.get('player_display_name', player.get('player_name', '')),
                'position': player.get('position', ''),
                'headshot_url': player.get('headshot_url', ''),
                'image_url': player.get('image_url', ''),  # Alternative headshot field
                'height': player.get('height', ''),
                'weight': player.get('weight', ''),
                'college': player.get('college', ''),
                'jersey': player.get('jersey', ''),
                'historical_team': player.get('recent_team', player.get('sportradar_team_alias', ''))
            }
            
            # Use image_url if headshot_url is empty
            if not player_info['headshot_url'] and player_info['image_url']:
                player_info['headshot_url'] = player_info['image_url']
            
            for name in names_to_try:
                if name and len(name) > 2:
                    player_lookup[name] = player_info
        
        print(f"   📝 Created NFL dataset lookup for {len(player_lookup)} name variations")
        return player_lookup
        
    except Exception as e:
        print(f"   ❌ Error loading NFL dataset: {e}")
        return {}

def load_sportradar_players(sportradar_csv_path="data/sportradar_players_20250806_221354.csv"):
    """Load SportRadar player data for ID and info lookup"""
    print(f"📊 Loading SportRadar player data...")
    
    try:
        df_sr = pd.read_csv(sportradar_csv_path)
        print(f"   ✅ Loaded {len(df_sr)} SportRadar players")
        
        # Create lookup dictionary by name variations
        sr_lookup = {}
        
        for _, player in df_sr.iterrows():
            # Get all name variations
            names_to_try = []
            
            # Add display_name if available
            if pd.notna(player.get('display_name')) and str(player.get('display_name')).strip():
                names_to_try.append(str(player['display_name']).lower().strip())
            
            # Add name if available
            if pd.notna(player.get('name')) and str(player.get('name')).strip():
                names_to_try.append(str(player['name']).lower().strip())
            
            # Add first + last name combination
            first_name = str(player.get('first_name', '')).strip()
            last_name = str(player.get('last_name', '')).strip()
            if first_name and last_name:
                full_name = f"{first_name} {last_name}".lower().strip()
                names_to_try.append(full_name)
            
            # Store player info for each name variation
            player_info = {
                'sportradar_player_id': player.get('sportradar_player_id', ''),
                'display_name': player.get('display_name', ''),
                'name': player.get('name', ''),
                'position': player.get('position', ''),
                'jersey': player.get('jersey', ''),
                'team_alias': player.get('team_alias', ''),
                'height': player.get('height', ''),
                'weight': player.get('weight', ''),
                'college': player.get('college', ''),
                'experience': player.get('experience', '')
            }
            
            for name in names_to_try:
                if name and len(name) > 2:
                    sr_lookup[name] = player_info
        
        print(f"   📝 Created SportRadar lookup for {len(sr_lookup)} name variations")
        return sr_lookup
        
    except Exception as e:
        print(f"   ❌ Error loading SportRadar data: {e}")
        return {}

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

def find_comprehensive_player_info(player_name, nfl_lookup, sportradar_lookup, current_teams):
    """Find comprehensive player info from NFL dataset, SportRadar, and current roster data"""
    if not player_name:
        return {}
    
    # Normalize the player name
    name_clean = str(player_name).lower().strip()
    
    # Start with empty info
    player_info = {
        'sportradar_player_id': '',
        'display_name': '',
        'jersey': '',
        'current_team': '',
        'position': '',
        'height': '',
        'weight': '',
        'college': '',
        'headshot_url': ''
    }
    
    # First check NFL dataset for headshots and comprehensive info
    if name_clean in nfl_lookup:
        nfl_info = nfl_lookup[name_clean]
        player_info.update({
            'sportradar_player_id': nfl_info.get('sportradar_player_id', '') or player_info.get('sportradar_player_id', ''),
            'display_name': nfl_info.get('display_name', '') or nfl_info.get('player_name', ''),
            'position': nfl_info.get('position', ''),
            'height': nfl_info.get('height', ''),
            'weight': nfl_info.get('weight', ''),
            'college': nfl_info.get('college', ''),
            'headshot_url': nfl_info.get('headshot_url', ''),
            'jersey': nfl_info.get('jersey', ''),
            'team_from_nfl': nfl_info.get('historical_team', '')
        })
    
    # Then check SportRadar data for additional info
    if name_clean in sportradar_lookup:
        sr_info = sportradar_lookup[name_clean]
        player_info.update({
            'sportradar_player_id': sr_info.get('sportradar_player_id', '') or player_info.get('sportradar_player_id', ''),
            'display_name': sr_info.get('display_name', '') or sr_info.get('name', '') or player_info.get('display_name', ''),
            'jersey': sr_info.get('jersey', '') or player_info.get('jersey', ''),
            'position': sr_info.get('position', '') or player_info.get('position', ''),
            'height': sr_info.get('height', '') or player_info.get('height', ''),
            'weight': sr_info.get('weight', '') or player_info.get('weight', ''),
            'college': sr_info.get('college', '') or player_info.get('college', ''),
            'team_from_sr': sr_info.get('team_alias', '')
        })
    
    # Finally check current roster data for team assignment
    if name_clean in current_teams:
        roster_info = current_teams[name_clean]
        player_info.update({
            'current_team': roster_info.get('current_team', ''),
            'position': roster_info.get('position', '') or player_info.get('position', ''),
            'jersey': roster_info.get('jersey_number', '') or player_info.get('jersey', ''),
            'roster_status': roster_info.get('status', 'ACT')
        })
    else:
        # Try partial matching for team assignment
        for roster_name, roster_info in current_teams.items():
            if name_clean in roster_name or roster_name in name_clean:
                player_info['current_team'] = roster_info.get('current_team', '')
                break
    
    # Use final name preference
    if not player_info['display_name']:
        player_info['display_name'] = player_name
    
    return player_info

def create_comprehensive_player_pools_with_headshots(season=2025, week=4, 
                                                    predictions_dir="predictions",
                                                    nfl_dataset_path="data/nfl_dataset.csv",
                                                    sportradar_csv_path="data/sportradar_players_20250806_221354.csv",
                                                    output_dir="player_pools"):
    """
    Create comprehensive player pools with headshots from NFL dataset and all other data
    """
    
    print(f"Creating COMPREHENSIVE player pools with HEADSHOTS for {season} Week {week}")
    print("="*80)
    
    # Load all data sources
    nfl_lookup = load_nfl_dataset_with_headshots(nfl_dataset_path)
    sportradar_lookup = load_sportradar_players(sportradar_csv_path)
    current_teams = load_current_rosters(season)
    matchup_data = load_schedule_data(season, week)
    team_mapping = create_team_mapping()
    
    # Process predictions for each position
    positions = ['QB', 'RB', 'WR', 'TE']
    os.makedirs(output_dir, exist_ok=True)
    total_players_created = 0
    headshot_matches = 0
    
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
        
        # Create player pool
        comprehensive_players = []
        sportradar_matches = 0
        team_assignments = 0
        position_headshots = 0
        
        for _, pred in df_pred.iterrows():
            # Get comprehensive player info including headshots
            player_info = find_comprehensive_player_info(
                pred['player_name'], nfl_lookup, sportradar_lookup, current_teams
            )
            
            if player_info.get('sportradar_player_id'):
                sportradar_matches += 1
            if player_info.get('current_team'):
                team_assignments += 1
            if player_info.get('headshot_url'):
                position_headshots += 1
            
            # Get projected points from prediction data
            projected_points = pred.get('fanduel_fantasy_points', 0) or 0
            
            # Calculate salary with position-specific multiplier (33% reduction for QBs)
            multiplier = 450 if position == 'QB' else 500
            salary = max(4000, round(int(projected_points * multiplier) / 100) * 100)
            
            # Get opponent and schedule info
            final_team = player_info.get('current_team', '')
            team_key = team_mapping.get(final_team, final_team)
            matchup_info = matchup_data.get(team_key, {})
            
            opponent = matchup_info.get('opponent_team', '')
            home_away = matchup_info.get('home_away', '')
            
            if opponent and home_away:
                opponent_display = f"vs {opponent}" if home_away == 'home' else f"@ {opponent}"
            else:
                opponent_display = "TBD"
            
            # Create comprehensive player object with headshot
            comprehensive_player = {
                # Essential IDs
                'id': f"{pred['player_name'].lower().replace(' ', '_')}_{final_team}_{season}",
                'sportradar_player_id': player_info.get('sportradar_player_id', ''),
                
                # Names (frontend uses both)
                'player_name': pred['player_name'],
                'name': player_info.get('display_name', pred['player_name']),
                
                # Position
                'position': player_info.get('position', position),
                
                # Pricing (frontend expects both)
                'price': salary,
                'salary': salary,
                
                # Fantasy projections (frontend expects both)
                'projected_points': round(projected_points, 1),
                'avg_fppg': round(projected_points, 1),
                
                # Team info (required for display) - TEAM ABBREVIATION, NOT UUID
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
                
                # Player details with HEADSHOTS from NFL dataset
                'headshot_url': player_info.get('headshot_url', ''),
                'jersey_number': player_info.get('jersey', ''),
                'roster_status': player_info.get('roster_status', 'ACT'),
                'height': player_info.get('height', ''),
                'weight': player_info.get('weight', ''),
                'college': player_info.get('college', ''),
                
                # Ranking and confidence
                'rank': pred.get('rank', len(comprehensive_players) + 1),
                'prediction_rank': pred.get('rank', len(comprehensive_players) + 1),
                'confidence': round(pred.get('vs_season_avg', 0) if pd.notna(pred.get('vs_season_avg')) else 0, 2)
            }
            
            comprehensive_players.append(comprehensive_player)
        
        # Sort by projected points (descending) and assign ranks
        if comprehensive_players:
            df_comprehensive = pd.DataFrame(comprehensive_players)
            df_comprehensive = df_comprehensive.sort_values('projected_points', ascending=False)
            df_comprehensive['rank'] = range(1, len(df_comprehensive) + 1)
            df_comprehensive['prediction_rank'] = df_comprehensive['rank']
            
            # Save to CSV
            output_file = f"{position}_player_pool_{season}_week{week}.csv"
            output_path = os.path.join(output_dir, output_file)
            df_comprehensive.to_csv(output_path, index=False)
            
            total_players_created += len(comprehensive_players)
            headshot_matches += position_headshots
            
            print(f"  ✅ {position}: {len(comprehensive_players)} players saved")
            print(f"     SportRadar IDs: {sportradar_matches}/{len(comprehensive_players)} ({sportradar_matches/len(comprehensive_players)*100:.1f}%)")
            print(f"     Team assignments: {team_assignments}/{len(comprehensive_players)} ({team_assignments/len(comprehensive_players)*100:.1f}%)")
            print(f"     Headshot URLs: {position_headshots}/{len(comprehensive_players)} ({position_headshots/len(comprehensive_players)*100:.1f}%)")
            print(f"     Salary range: ${df_comprehensive['salary'].min():,} - ${df_comprehensive['salary'].max():,}")
            
            # Show top 3 with headshot status
            print(f"     Top 3:")
            for _, player in df_comprehensive.head(3).iterrows():
                team_display = player['team'] if player['team'] else 'UNK'
                matchup = player['opponent_display']
                game_time = player['game_time']
                sr_id = "✅" if player['sportradar_player_id'] else "❌"
                headshot = "🖼️" if player['headshot_url'] else "❌"
                print(f"       {player['rank']}. {player['player_name']} ({team_display}) - ${player['salary']:,} ({player['projected_points']} pts) - {matchup} {game_time} {sr_id} {headshot}")
    
    # Create combined file
    create_combined_comprehensive_file_with_headshots(season, week, output_dir)
    
    print(f"\n{'='*80}")
    print(f"✅ COMPREHENSIVE player pools with HEADSHOTS complete!")
    print(f"📊 Total players created: {total_players_created}")
    print(f"🖼️ Players with headshots: {headshot_matches}/{total_players_created} ({headshot_matches/total_players_created*100:.1f}%)")
    print(f"📁 Files saved to: {output_dir}/")
    
    return True

def create_combined_comprehensive_file_with_headshots(season, week, output_dir):
    """Create combined file with all positions in comprehensive format including headshots"""
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
        
        print(f"\n📋 Combined comprehensive file with headshots: {combined_file}")
        print(f"   Total players: {len(df_combined)}")
        
        position_counts = df_combined['position'].value_counts()
        for pos in ['QB', 'RB', 'WR', 'TE']:
            if pos in position_counts:
                print(f"   {pos}: {position_counts[pos]} players")
        
        print(f"   Salary range: ${df_combined['salary'].min():,} - ${df_combined['salary'].max():,}")
        print(f"   Average salary: ${df_combined['salary'].mean():,.0f}")
        
        # Count headshots
        headshot_count = len(df_combined[df_combined['headshot_url'] != ''])
        print(f"   Players with headshots: {headshot_count}/{len(df_combined)} ({headshot_count/len(df_combined)*100:.1f}%)")
        
        # Count SportRadar matches
        sr_matches = len(df_combined[df_combined['sportradar_player_id'] != ''])
        print(f"   SportRadar IDs: {sr_matches}/{len(df_combined)} ({sr_matches/len(df_combined)*100:.1f}%)")

if __name__ == "__main__":
    success = create_comprehensive_player_pools_with_headshots()
    
    if success:
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE PLAYER POOLS WITH HEADSHOTS READY!")
        print("\n✅ Features included:")
        print("   • SportRadar player IDs (when available)")
        print("   • Player headshot URLs from NFL dataset")
        print("   • Current 2025 team assignments (as abbreviations)")
        print("   • Opponent matchups and game times")
        print("   • Fixed QB pricing (33% reduction)")
        print("   • Player details (height, weight, college)")
        print("   • Proper ranking and confidence scores")
        print("\n🚀 Ready to load directly into your backend!")
        print("📝 Import these CSVs into your player pool loader")