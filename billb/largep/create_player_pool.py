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
        
        # Create player-to-team mapping
        current_teams = {}
        
        for _, player in active_roster.iterrows():
            player_name = str(player.get('display_name', player.get('full_name', ''))).strip()
            team = str(player.get('team', '')).strip()
            position = str(player.get('position', '')).strip()
            
            if player_name and team:
                # Store multiple name variations
                names_to_store = [
                    player_name.lower().strip(),
                    str(player.get('full_name', '')).lower().strip(),
                    str(player.get('first_name', '') + ' ' + player.get('last_name', '')).lower().strip()
                ]
                
                for name in names_to_store:
                    if name and len(name) > 2:
                        current_teams[name] = {
                            'current_team': team,
                            'position': position,
                            'status': player.get('status', 'ACT'),
                            'jersey_number': player.get('jersey_number', ''),
                            'roster_name': player_name
                        }
        
        print(f"   📊 Created current team mapping for {len(current_teams)} player name variations")
        return current_teams
        
    except Exception as e:
        print(f"   ❌ Error loading current rosters: {e}")
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

def create_player_pools(season=2025, week=1, 
                       dataset_path="data/complete_nfl_with_sportradar.csv",
                       predictions_dir="predictions",
                       output_dir="player_pools"):
    """Create player pools with current roster data and schedule integration"""
    
    print(f"Creating player pools for {season} Week {week}")
    print("="*60)
    
    # Load current roster and schedule data
    current_teams = load_current_rosters(season)
    matchup_data = load_schedule_data(season, week)
    team_mapping = create_team_mapping()
    
    # Load historical dataset for player info
    df_enhanced = pd.read_csv(dataset_path, low_memory=False)
    print(f"📊 Loaded historical dataset: {df_enhanced.shape}")
    
    # Create player lookup
    player_lookup = {}
    
    # Get latest info for each player from historical data
    for _, player in df_enhanced.groupby('player_id').last().iterrows():
        names_to_try = []
        
        for name_col in ['player_name', 'player_display_name', 'sportradar_name']:
            if pd.notna(player.get(name_col)):
                names_to_try.append(str(player[name_col]).lower().strip())
        
        for name in names_to_try:
            if name:
                player_lookup[name] = {
                    'sportradar_player_id': player.get('sportradar_player_id', ''),
                    'player_name': player.get('player_display_name', player.get('player_name', '')),
                    'name': player.get('player_display_name', player.get('player_name', '')),
                    'position': player.get('position', ''),
                    'headshot_url': player.get('headshot_url', ''),
                    'historical_team': player.get('recent_team', player.get('sportradar_team_alias', ''))
                }
    
    # Update with current roster data
    team_updates = 0
    for name, current_info in current_teams.items():
        if name in player_lookup:
            player_lookup[name]['current_team'] = current_info['current_team']
            player_lookup[name]['position'] = current_info['position'] or player_lookup[name]['position']
            player_lookup[name]['jersey_number'] = current_info['jersey_number']
            player_lookup[name]['roster_status'] = current_info['status']
            team_updates += 1
        else:
            player_lookup[name] = {
                'sportradar_player_id': '',
                'player_name': current_info['roster_name'],
                'name': current_info['roster_name'],
                'position': current_info['position'],
                'headshot_url': '',
                'current_team': current_info['current_team'],
                'jersey_number': current_info['jersey_number'],
                'roster_status': current_info['status'],
                'historical_team': ''
            }
    
    print(f"🎯 Created lookup for {len(player_lookup)} player name variations")
    print(f"🔄 Updated {team_updates} players with current team assignments")
    
    # Add schedule data to player lookup
    for name, player_info in player_lookup.items():
        player_team = player_info.get('current_team') or player_info.get('historical_team', '')
        team_key = team_mapping.get(player_team, player_team)
        matchup_info = matchup_data.get(team_key, {})
        
        opponent = matchup_info.get('opponent_team', '')
        home_away = matchup_info.get('home_away', '')
        
        if opponent and home_away:
            opponent_display = f"vs {opponent}" if home_away == 'home' else f"@ {opponent}"
        else:
            opponent_display = "TBD"
        
        player_info.update({
            'team': player_team,
            'recent_team': player_team,
            'opponent_team': opponent,
            'opponent_display': opponent_display,
            'home_away': home_away,
            'game_time': matchup_info.get('game_time', 'TBD'),
            'gameday': matchup_info.get('gameday', ''),
            'stadium': matchup_info.get('stadium', ''),
            'game_id': matchup_info.get('game_id', '')
        })
    
    if matchup_data:
        print(f"📅 Schedule data loaded for {len(matchup_data)} teams")
    
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
        
        # Create player pool
        clean_players = []
        matched = 0
        with_schedule = 0
        with_current_team = 0
        
        for _, pred in df_pred.iterrows():
            name_clean = str(pred['player_name']).lower().strip()
            player_info = player_lookup.get(name_clean, {})
            
            if player_info.get('sportradar_player_id'):
                matched += 1
            if player_info.get('opponent_team'):
                with_schedule += 1
            if player_info.get('current_team'):
                with_current_team += 1
            
            # Get projected points and calculate price
            projected_points = pred.get('fanduel_fantasy_points', pred.get('projected_points', 0)) or 0
            price = max(4000, round(int(projected_points * 500) / 100) * 100)
            
            clean_player = {
                'sportradar_player_id': player_info.get('sportradar_player_id', ''),
                'player_name': player_info.get('player_name', pred['player_name']),
                'name': player_info.get('name', pred['player_name']),
                'position': player_info.get('position', position),
                'headshot_url': player_info.get('headshot_url', ''),
                'price': price,
                'salary': price,
                'projected_points': round(projected_points, 1),
                'avg_fppg': round(projected_points, 1),
                'team': player_info.get('current_team') or player_info.get('historical_team', pred.get('team', '')),
                'recent_team': player_info.get('current_team') or player_info.get('historical_team', pred.get('team', '')),
                'opponent_team': player_info.get('opponent_team', ''),
                'opponent_display': player_info.get('opponent_display', 'TBD'),
                'home_away': player_info.get('home_away', ''),
                'game_time': player_info.get('game_time', 'TBD'),
                'gameday': player_info.get('gameday', ''),
                'stadium': player_info.get('stadium', ''),
                'game_id': player_info.get('game_id', ''),
                'jersey_number': player_info.get('jersey_number', ''),
                'roster_status': player_info.get('roster_status', 'ACT'),
                'prediction_rank': pred.get('rank', pred.get('prediction_rank', len(clean_players) + 1)),
                'confidence': pred.get('confidence_score', pred.get('vs_season_avg', 0))
            }
            
            clean_players.append(clean_player)
        
        # Sort and save
        if clean_players:
            df_clean = pd.DataFrame(clean_players)
            df_clean = df_clean.sort_values('price', ascending=False)
            df_clean['rank'] = range(1, len(df_clean) + 1)
            
            output_file = f"{position}_player_pool_{season}_week{week}.csv"
            output_path = os.path.join(output_dir, output_file)
            df_clean.to_csv(output_path, index=False)
            
            total_players_created += len(clean_players)
            
            print(f"  ✅ {position}: {len(clean_players)} players saved")
            print(f"     Sportradar IDs: {matched}/{len(clean_players)} ({matched/len(clean_players)*100:.1f}%)")
            print(f"     Current teams: {with_current_team}/{len(clean_players)} ({with_current_team/len(clean_players)*100:.1f}%)")
            print(f"     Schedule data: {with_schedule}/{len(clean_players)} ({with_schedule/len(clean_players)*100:.1f}%)")
            print(f"     Price range: ${df_clean['price'].min():,} - ${df_clean['price'].max():,}")
            
            # Show top 3
            print(f"     Top 3:")
            for _, player in df_clean.head(3).iterrows():
                matchup = player.get('opponent_display', 'TBD')
                game_time = player.get('game_time', 'TBD')
                current_team = player.get('team', 'UNK')
                print(f"       {player['rank']}. {player['player_name']} ({current_team}) - ${player['price']:,} ({player['projected_points']} pts) - {matchup} {game_time}")
        else:
            print(f"  ❌ No valid players created for {position}")
    
    # Create combined file
    create_combined_file(season, week, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ Player pool creation complete!")
    print(f"📊 Total players created: {total_players_created}")
    print(f"📁 Files saved to: {output_dir}/")
    
    return True

def create_combined_file(season, week, output_dir):
    """Create combined file with all positions"""
    positions = ['QB', 'RB', 'WR', 'TE']
    all_players = []
    
    for position in positions:
        file_path = os.path.join(output_dir, f"{position}_player_pool_{season}_week{week}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_players.append(df)
    
    if all_players:
        df_combined = pd.concat(all_players, ignore_index=True)
        
        # Sort by position priority, then by price
        position_order = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 4}
        df_combined['position_order'] = df_combined['position'].map(position_order)
        df_combined = df_combined.sort_values(['position_order', 'price'], ascending=[True, False])
        df_combined = df_combined.drop('position_order', axis=1)
        
        # Save combined file
        combined_file = f"all_players_pool_{season}_week{week}.csv"
        combined_path = os.path.join(output_dir, combined_file)
        df_combined.to_csv(combined_path, index=False)
        
        print(f"\n📋 Combined file: {combined_file}")
        print(f"   Total players: {len(df_combined)}")
        
        position_counts = df_combined['position'].value_counts()
        for pos in ['QB', 'RB', 'WR', 'TE']:
            if pos in position_counts:
                print(f"   {pos}: {position_counts[pos]} players")
        
        print(f"   Price range: ${df_combined['price'].min():,} - ${df_combined['price'].max():,}")
        print(f"   Average price: ${df_combined['price'].mean():,.0f}")
        
        # Show sample matchups
        sample_with_schedule = df_combined[df_combined['opponent_team'].notna()].head(3)
        if not sample_with_schedule.empty:
            print(f"\n📅 Sample matchups with current teams:")
            for _, player in sample_with_schedule.iterrows():
                print(f"   {player['player_name']} ({player['team']}) {player['opponent_display']} - {player['game_time']}")

if __name__ == "__main__":
    success = create_player_pools()
    
    if success:
        print("\n" + "="*60)
        print("🎯 FRONTEND INTEGRATION READY!")
        print("\n✅ Features included:")
        print("   • Sportradar player IDs")
        print("   • Current 2025 team assignments") 
        print("   • Real schedule matchups")
        print("   • Calculated prices")
        print("   • Frontend-compatible format")
        print("\n🚀 Ready to load into your Svelte app!")