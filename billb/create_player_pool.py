import pandas as pd
import os
import argparse
from tqdm import tqdm
import re

def normalize_player_name(name):
    """
    Normalize player names by removing suffixes and standardizing format
    """
    if pd.isna(name) or not name:
        return ""
    
    name = str(name).strip()
    
    # Common suffixes to remove (case insensitive)
    suffixes = [
        r'\s+Jr\.?$',      # Jr, Jr.
        r'\s+Sr\.?$',      # Sr, Sr.
        r'\s+III$',        # III
        r'\s+II$',         # II
        r'\s+IV$',         # IV
        r'\s+V$',          # V
        r'\s+VI$',         # VI
    ]
    
    # Remove suffixes
    for suffix in suffixes:
        name = re.sub(suffix, '', name, flags=re.IGNORECASE)
    
    # Clean up extra spaces and convert to lowercase
    name = re.sub(r'\s+', ' ', name).strip().lower()
    
    return name

def create_player_pool(season, week, main_dataset_path="data/complete_nfl_with_sportradar.csv", 
                      predictions_directory="predictions", output_directory="player_pools"):
    """
    Create clean player pool CSVs by combining predictions with player info
    
    Args:
    season (int): Season year (e.g., 2025)
    week (int): Week number (e.g., 1)
    main_dataset_path (str): Path to complete_nfl_with_sportradar.csv
    predictions_directory (str): Path to predictions folder
    output_directory (str): Where to save the clean player pool CSVs
    """
    
    print(f"Creating player pool for {season} Week {week}")
    print("="*50)
    
    if not os.path.exists(main_dataset_path):
        print(f"Error: Main dataset not found at {main_dataset_path}")
        return False
    
    print("Loading main dataset...")
    df_main = pd.read_csv(main_dataset_path)
    
    # Get unique player info (latest record for each player to get current info)
    print("Processing player information...")
    player_info = df_main.groupby('player_id').last().reset_index()
    
    # Create player lookup by name variations (with suffix normalization)
    player_lookup = {}
    
    for _, player in player_info.iterrows():
        # Create multiple name variations for matching
        names_to_try = []
        
        # Get all possible name fields
        name_fields = ['player_display_name', 'player_name', 'sportradar_display_name']
        
        for field in name_fields:
            if pd.notna(player.get(field)):
                original_name = str(player[field]).strip()
                normalized_name = normalize_player_name(original_name)
                
                # Add both original and normalized versions
                names_to_try.append(original_name.lower().strip())
                if normalized_name and normalized_name != original_name.lower().strip():
                    names_to_try.append(normalized_name)
        
        # Store player info under all name variations
        for name in names_to_try:
            if name:
                player_lookup[name] = {
                    'sportradar_player_id': player.get('sportradar_player_id', ''),
                    'player_name': player.get('player_display_name', player.get('player_name', '')),
                    'position_x': player.get('position_x', ''),
                    'headshot_url': player.get('headshot_url', ''),
                    'recent_team': player.get('recent_team', '')
                }
    
    print(f"Created lookup for {len(player_lookup)} player name variations")
    
    # Process each position
    positions = ['QB', 'RB', 'WR', 'TE']
    
    os.makedirs(output_directory, exist_ok=True)
    
    total_players_processed = 0
    files_created = []
    
    for position in tqdm(positions, desc="Processing positions"):
        # Use standard naming convention: {position}_predictions_week{week}_season{season}.csv
        prediction_file = f"{position}_predictions_week{week}_season{season}.csv"
        prediction_path = os.path.join(predictions_directory, prediction_file)
        
        if not os.path.exists(prediction_path):
            print(f"\nWarning: {prediction_file} not found, skipping {position}")
            continue
        
        print(f"\nProcessing {position}...")
        
        # Load predictions
        try:
            df_pred = pd.read_csv(prediction_path)
        except Exception as e:
            print(f"Error loading {prediction_file}: {e}")
            continue
        
        if df_pred.empty:
            print(f"Warning: {prediction_file} is empty, skipping {position}")
            continue
        
        # Create clean player pool
        clean_players = []
        matched_count = 0
        unmatched_players = []
        
        for _, pred in df_pred.iterrows():
            pred_name_original = str(pred['player_name']).strip()
            pred_name_normalized = normalize_player_name(pred_name_original)
            
            # Try to find player info using multiple name variations
            player_info_found = None
            
            # Try original name first
            player_info_found = player_lookup.get(pred_name_original.lower().strip())
            
            # If not found, try normalized name
            if not player_info_found and pred_name_normalized:
                player_info_found = player_lookup.get(pred_name_normalized)
            
            if player_info_found:
                # Calculate price (projected points * 500, rounded to nearest $100)
                projected_points = pred.get('fanduel_fantasy_points', pred.get('projected_points', 0))
                if pd.isna(projected_points):
                    projected_points = 0
                    
                price = round(int(projected_points * 500) / 100) * 100
                
                clean_player = {
                    'sportradar_player_id': player_info_found['sportradar_player_id'],
                    'player_name': player_info_found['player_name'],
                    'position_x': player_info_found['position_x'],
                    'headshot_url': player_info_found['headshot_url'],
                    'price': price,
                    'projected_points': round(projected_points, 1),
                    'team': player_info_found['recent_team']
                }
                
                clean_players.append(clean_player)
                matched_count += 1
            else:
                # Player not found in main dataset - use prediction data only
                projected_points = pred.get('fanduel_fantasy_points', pred.get('projected_points', 0))
                if pd.isna(projected_points):
                    projected_points = 0
                    
                price = round(int(projected_points * 500) / 100) * 100
                
                clean_player = {
                    'sportradar_player_id': pred.get('sportradar_player_id', ''),
                    'player_name': pred['player_name'],
                    'position_x': pred.get('position', position),
                    'headshot_url': pred.get('headshot_url', ''),
                    'price': price,
                    'projected_points': round(projected_points, 1),
                    'team': pred.get('team', '')
                }
                
                clean_players.append(clean_player)
                unmatched_players.append(pred_name_original)
        
        if not clean_players:
            print(f"  No valid players found for {position}")
            continue
        
        # Create DataFrame and sort by price (highest first)
        df_clean = pd.DataFrame(clean_players)
        df_clean = df_clean.sort_values('price', ascending=False)
        
        # Save clean player pool
        output_file = f"{position}_player_pool_{season}_week{week}.csv"
        output_path = os.path.join(output_directory, output_file)
        df_clean.to_csv(output_path, index=False)
        
        files_created.append(output_file)
        total_players_processed += len(df_clean)
        
        print(f"  Saved {len(df_clean)} {position} players to {output_file}")
        print(f"  Matched {matched_count}/{len(df_pred)} players with main dataset")
        
        # Show unmatched players for debugging
        if unmatched_players:
            print(f"  Unmatched players: {', '.join(unmatched_players[:5])}")
            if len(unmatched_players) > 5:
                print(f"    ... and {len(unmatched_players) - 5} more")
        
        # Show top 5 most expensive
        if not df_clean.empty:
            print(f"  Top 5 most expensive {position}s:")
            for _, player in df_clean.head(5).iterrows():
                print(f"    {player['player_name']} ({player['team']}): ${player['price']:,} ({player['projected_points']} pts)")
    
    return files_created, total_players_processed

def create_combined_player_pool(season, week, output_directory="player_pools"):
    """Create a single combined file with all positions"""
    
    positions = ['QB', 'RB', 'WR', 'TE']
    all_players = []
    
    for position in positions:
        file_path = os.path.join(output_directory, f"{position}_player_pool_{season}_week{week}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_players.append(df)
    
    if all_players:
        df_combined = pd.concat(all_players, ignore_index=True)
        df_combined = df_combined.sort_values(['position_x', 'price'], ascending=[True, False])
        
        combined_file = f"all_players_pool_{season}_week{week}.csv"
        combined_path = os.path.join(output_directory, combined_file)
        df_combined.to_csv(combined_path, index=False)
        
        print(f"\nCombined player pool saved to {combined_file}")
        print(f"Total players: {len(df_combined)}")
        print(f"Players by position: {df_combined['position_x'].value_counts().to_dict()}")
        
        return combined_file
    
    return None

def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description='Create DFS player pools from predictions')
    parser.add_argument('--season', type=int, default=2025, help='Season year (default: 2025)')
    parser.add_argument('--week', type=int, default=1, help='Week number (default: 1)')
    parser.add_argument('--main-dataset', type=str, default='data/complete_nfl_with_sportradar.csv',
                       help='Path to main dataset (default: data/complete_nfl_with_sportradar.csv)')
    parser.add_argument('--predictions-dir', type=str, default='predictions',
                       help='Path to predictions directory (default: predictions)')
    parser.add_argument('--output-dir', type=str, default='player_pools',
                       help='Output directory (default: player_pools)')
    parser.add_argument('--no-combined', action='store_true',
                       help='Skip creating combined file')
    
    args = parser.parse_args()
    
    # Create individual position files
    files_created, total_players = create_player_pool(
        season=args.season,
        week=args.week,
        main_dataset_path=args.main_dataset,
        predictions_directory=args.predictions_dir,
        output_directory=args.output_dir
    )
    
    if not files_created:
        print("No player pool files were created. Check your prediction files.")
        return
    
    # Create combined file unless disabled
    combined_file = None
    if not args.no_combined:
        combined_file = create_combined_player_pool(args.season, args.week, args.output_dir)
    
    # Summary
    print("\n" + "="*50)
    print("Player pool creation complete!")
    print(f"Season: {args.season}, Week: {args.week}")
    print(f"Total players processed: {total_players}")
    print("\nFiles created:")
    for file in files_created:
        print(f"  - {file}")
    if combined_file:
        print(f"  - {combined_file}")
    
    print("\nEach file contains:")
    print("  - sportradar_player_id")
    print("  - player_name") 
    print("  - position_x")
    print("  - headshot_url")
    print("  - price (projected_points * 500, rounded to nearest $100)")
    print("  - projected_points")
    print("  - team")

if __name__ == "__main__":
    main()