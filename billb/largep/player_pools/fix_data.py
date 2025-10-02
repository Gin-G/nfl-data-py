#!/usr/bin/env python3
"""
Quick fix for existing duplicate files
"""
import pandas as pd
import os
import glob

def fix_existing_files():
    """Fix all existing prediction files"""
    
    # Find all prediction CSV files
    file_patterns = [
        "predictions/*_predictions_week*_season*.csv",
        "*_predictions_week*.csv", 
        "*player_pool*.csv"
    ]
    
    files_found = []
    for pattern in file_patterns:
        files_found.extend(glob.glob(pattern))
    
    if not files_found:
        print("No prediction files found to fix")
        return
    
    print(f"Found {len(files_found)} files to check:")
    
    for filepath in files_found:
        print(f"\nProcessing: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            original_count = len(df)
            
            # Check for duplicates (exclude rank columns)
            check_cols = [col for col in df.columns if col not in ['rank', 'prediction_rank', 'confidence']]
            
            if 'player_name' in df.columns:
                # Use player_name as primary duplicate check
                dupes = df['player_name'].duplicated().sum()
                if dupes > 0:
                    print(f"  Found {dupes} duplicate players")
                    
                    # Show examples
                    dup_players = df[df['player_name'].duplicated(keep=False)]['player_name'].unique()
                    for player in dup_players[:3]:
                        player_rows = df[df['player_name'] == player]
                        print(f"    {player}: {len(player_rows)} copies, ranks: {player_rows.get('rank', []).tolist()}")
                    
                    # Fix duplicates
                    df_clean = df.drop_duplicates(subset=['player_name'], keep='first')
                    
                    # Re-sort and re-rank
                    if 'projected_points' in df_clean.columns:
                        df_clean = df_clean.sort_values('projected_points', ascending=False)
                    elif 'fanduel_fantasy_points' in df_clean.columns:
                        df_clean = df_clean.sort_values('fanduel_fantasy_points', ascending=False)
                    
                    df_clean = df_clean.reset_index(drop=True)
                    df_clean['rank'] = range(1, len(df_clean) + 1)
                    
                    if 'prediction_rank' in df_clean.columns:
                        df_clean['prediction_rank'] = df_clean['rank']
                    
                    # Save fixed file
                    backup_file = filepath.replace('.csv', '_backup.csv')
                    df.to_csv(backup_file, index=False)  # Backup original
                    df_clean.to_csv(filepath, index=False)  # Save fixed
                    
                    print(f"  Fixed: {original_count} -> {len(df_clean)} rows")
                    print(f"  Backup saved: {backup_file}")
                else:
                    print("  No duplicates found")
            else:
                print("  No player_name column found")
                
        except Exception as e:
            print(f"  Error processing {filepath}: {e}")

if __name__ == "__main__":
    fix_existing_files()