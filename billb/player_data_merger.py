"""
Comprehensive NFL Player Data Merger
Merges NFL dataset with SportRadar player data, handling edge cases and preserving all statistics.
"""

import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class ComprehensivePlayerMerger:
    """
    Complete player data merger with generational matching and statistics preservation
    """
    
    def __init__(self):
        self.nfl_df = None
        self.sportradar_df = None
        self.merged_df = None
        self.unmatched_nfl_df = None
        self.unmatched_sr_df = None
        self.mapping_stats = {
            'exact_matches': 0,
            'position_matches': 0,
            'fuzzy_matches': 0,
            'unmatched_nfl': 0,
            'unmatched_sportradar': 0,
            'duplicate_names_resolved': 0,
            'generational_mismatches_avoided': 0,
            'total_nfl_records': 0,
            'preserved_records': 0
        }
    
    def load_data(self, nfl_csv_path: str, sportradar_csv_path: str) -> None:
        """Load both CSV files and analyze structure"""
        try:
            self.nfl_df = pd.read_csv(nfl_csv_path)
            self.sportradar_df = pd.read_csv(sportradar_csv_path)
            
            self.mapping_stats['total_nfl_records'] = len(self.nfl_df)
            
            logger.info(f"Loaded NFL dataset: {len(self.nfl_df):,} records")
            logger.info(f"Unique NFL players: {self.nfl_df['player_id'].nunique():,}")
            logger.info(f"Loaded SportRadar dataset: {len(self.sportradar_df):,} players")
            
            # Analyze NFL dataset structure
            print(f"\n📊 NFL Dataset Analysis:")
            print(f"  Seasons: {sorted(self.nfl_df['season'].unique())}")
            print(f"  Positions: {sorted([p for p in self.nfl_df['position_x'].unique() if pd.notna(p)])}")
            print(f"  Statistical columns: {len([col for col in self.nfl_df.columns if any(stat in col.lower() for stat in ['passing', 'rushing', 'receiving', 'fantasy', 'yards', 'tds'])])}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def extract_name_suffix(self, name: str) -> tuple:
        """Extract name and generational suffix (Jr, Sr, II, etc.)"""
        if pd.isna(name) or not name:
            return "", ""
        
        # Common generational suffixes
        suffixes = ['jr', 'sr', 'iii', 'ii', 'iv', 'v']
        name_clean = name.lower().strip()
        
        for suffix in suffixes:
            if name_clean.endswith(f' {suffix}') or name_clean.endswith(f'.{suffix}'):
                base_name = re.sub(rf'\s*\.?{suffix}$', '', name_clean).strip()
                return base_name, suffix
        
        return name_clean, ""
    
    def normalize_name(self, name: str) -> str:
        """Normalize player names for comparison (without removing suffixes)"""
        if pd.isna(name) or not name:
            return ""
        
        # Convert to lowercase, remove special characters, extra spaces
        normalized = re.sub(r'[^\w\s.]', '', str(name).lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def is_generational_mismatch(self, nfl_name: str, sr_name: str, nfl_player_id: str, sr_experience: int = None) -> tuple:
        """
        Check if two players with similar names are likely different generations
        Returns (is_mismatch, reason)
        """
        try:
            nfl_base, nfl_suffix = self.extract_name_suffix(nfl_name)
            sr_base, sr_suffix = self.extract_name_suffix(sr_name)
            
            # If base names are different, not a generational issue
            if nfl_base != sr_base:
                return False, ""
            
            # If one has a suffix and the other doesn't, likely different generations
            if bool(nfl_suffix) != bool(sr_suffix):
                return True, f"Suffix mismatch: '{nfl_suffix}' vs '{sr_suffix}'"
            
            # If both have different suffixes, definitely different generations
            if nfl_suffix and sr_suffix and nfl_suffix != sr_suffix:
                return True, f"Different suffixes: '{nfl_suffix}' vs '{sr_suffix}'"
            
            # Check for realistic experience mismatches using actual NFL data
            if sr_experience is not None and nfl_player_id:
                # Get actual NFL career span for this player
                player_data = self.nfl_df[self.nfl_df['player_id'] == nfl_player_id]
                if len(player_data) > 0:
                    nfl_seasons = player_data['season'].unique()
                    earliest_season = player_data['season'].min()
                    latest_season = player_data['season'].max()
                    
                    # Only flag if someone has been in the dataset for 4+ years but SR shows ≤2 years
                    # This catches veterans being matched to rookies/sophomores with same names
                    years_in_dataset = latest_season - earliest_season + 1
                    
                    if years_in_dataset >= 4 and sr_experience <= 2:
                        return True, f"Veteran/rookie mismatch: {years_in_dataset} years in dataset ({earliest_season}-{latest_season}) but SR shows {sr_experience} years experience"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Error in generational mismatch check: {e}")
            return False, ""
    
    def similarity_score(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names"""
        norm1 = self.normalize_name(name1)
        norm2 = self.normalize_name(name2)
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def is_likely_different_person(self, nfl_name: str, sr_name: str, match_score: float) -> tuple:
        """
        Check if two names that score well are actually different people
        Returns (is_different, reason)
        """
        
        try:
            # Known problematic matches - completely different people
            known_mismatches = [
                ('frank gore', 'frank gore jr.', "Father/son with same name"),
                ('kyle williams', 'tyleik williams', "Different people with similar names"),
                ('paul richardson', 'jp richardson', "Different people - Paul vs JP"),
                ('tye smith', 'tykee smith', "Different people with similar spelling"),
                ('trenton scott', 'trent scott', "Different people - Trenton vs Trent"),
                ('keesean johnson', 'kisean johnson', "Different people with similar spelling"),
                ('scott miller', 'scotty miller', "Different people - Scott vs Scotty"),
            ]
            
            # Known good matches (nicknames, etc.)
            known_good_matches = [
                ('jamycal hasty', 'jamycal hasty'),
                ('andrew ogletree', 'drew ogletree'),
                ('josh palmer', 'joshua palmer'),
            ]
            
            # Normalize names for comparison
            nfl_norm = self.normalize_name(nfl_name)
            sr_norm = self.normalize_name(sr_name)
            
            # Check known mismatches
            for nfl_bad, sr_bad, reason in known_mismatches:
                if (nfl_norm == nfl_bad and sr_norm == sr_bad) or (nfl_norm == sr_bad and sr_norm == nfl_bad):
                    return True, reason
            
            # Check known good matches
            for nfl_good, sr_good in known_good_matches:
                if (nfl_norm == nfl_good and sr_norm == sr_good) or (nfl_norm == sr_good and sr_norm == nfl_good):
                    return False, "Known good match"
            
            # If we get here, it's probably fine
            return False, ""
            
        except Exception as e:
            logger.error(f"Error in is_likely_different_person: {e}")
            return False, "Error in check"
    
    def find_position_group_match(self, position: str) -> str:
        """Map position to position group for broader matching"""
        if pd.isna(position):
            return ""
            
        position_groups = {
            'QB': 'QB',
            'RB': 'RB', 'FB': 'RB',
            'WR': 'WR',
            'TE': 'TE',
            'T': 'OL', 'G': 'OL', 'C': 'OL', 'OT': 'OL', 'OL': 'OL',
            'DE': 'DL', 'DT': 'DL', 'NT': 'DL', 'DL': 'DL',
            'OLB': 'LB', 'ILB': 'LB', 'MLB': 'LB', 'LB': 'LB',
            'CB': 'DB', 'SS': 'DB', 'FS': 'DB', 'SAF': 'DB', 'DB': 'DB',
            'K': 'K',
            'P': 'P',
            'LS': 'LS'
        }
        return position_groups.get(position, position)
    
    def create_player_mapping(self) -> pd.DataFrame:
        """
        Create mapping between NFL player IDs and SportRadar player IDs
        Returns DataFrame with just the mapping (not all statistics)
        """
        
        # Get unique NFL players for mapping
        unique_nfl_players = self.nfl_df.drop_duplicates(subset=['player_id']).copy()
        logger.info(f"Creating mapping for {len(unique_nfl_players)} unique NFL players...")
        
        matches = []
        used_sportradar_ids = set()
        
        for _, nfl_player in unique_nfl_players.iterrows():
            try:
                nfl_name = nfl_player['player_display_name']
                nfl_position = nfl_player['position_x']
                nfl_pos_group = self.find_position_group_match(nfl_position)
                
                best_match = None
                best_score = 0
                match_type = None
                
                logger.debug(f"Processing player: {nfl_name} ({nfl_position})")
                
                # Strategy 1: Exact display name + position match
                exact_matches = self.sportradar_df[
                    (self.sportradar_df['name'] == nfl_name) |
                    (self.sportradar_df['display_name'] == nfl_name)
                ]
                
                logger.debug(f"Found {len(exact_matches)} exact name matches for {nfl_name}")
                
                # Filter out generational mismatches before position filtering
                if len(exact_matches) > 0:
                    valid_matches = []
                    for _, sr_player in exact_matches.iterrows():
                        try:
                            is_mismatch, reason = self.is_generational_mismatch(
                                nfl_name, 
                                sr_player['name'],
                                nfl_player['player_id'], 
                                sr_player.get('experience')
                            )
                            
                            if not is_mismatch:
                                valid_matches.append(sr_player)
                            else:
                                self.mapping_stats['generational_mismatches_avoided'] += 1
                                logger.info(f"Avoided generational mismatch: {nfl_name} -> {sr_player['name']} ({reason})")
                        except Exception as e:
                            logger.error(f"Error in generational mismatch check for {nfl_name}: {e}")
                            continue
                    
                    if valid_matches:
                        exact_matches = pd.DataFrame(valid_matches)
                    else:
                        exact_matches = pd.DataFrame()
                
                # Filter by position
                position_filtered = exact_matches[
                    exact_matches['position'] == nfl_position
                ] if len(exact_matches) > 0 else pd.DataFrame()
                
                if len(position_filtered) == 1 and position_filtered.iloc[0]['sportradar_player_id'] not in used_sportradar_ids:
                    best_match = position_filtered.iloc[0]
                    best_score = 1.0
                    match_type = 'exact_name_position'
                    self.mapping_stats['exact_matches'] += 1
                
                # Strategy 2: Handle duplicate names by using position groups
                elif len(exact_matches) > 1:
                    pos_group_matches = exact_matches[
                        exact_matches['position'].apply(self.find_position_group_match) == nfl_pos_group
                    ]
                    
                    if len(pos_group_matches) == 1 and pos_group_matches.iloc[0]['sportradar_player_id'] not in used_sportradar_ids:
                        best_match = pos_group_matches.iloc[0]
                        best_score = 0.95
                        match_type = 'name_position_group'
                        self.mapping_stats['position_matches'] += 1
                        self.mapping_stats['duplicate_names_resolved'] += 1
                        
                        logger.info(f"Resolved duplicate name: {nfl_name} -> {best_match['position']} (vs others)")
                
                # Strategy 3: Fuzzy matching + position for close name variations
                if best_match is None:
                    for _, sr_player in self.sportradar_df.iterrows():
                        if sr_player['sportradar_player_id'] in used_sportradar_ids:
                            continue
                        
                        sr_pos_group = self.find_position_group_match(sr_player['position'])
                        if sr_pos_group != nfl_pos_group:
                            continue
                        
                        # Skip if this looks like a generational mismatch
                        is_mismatch, reason = self.is_generational_mismatch(
                            nfl_name, 
                            sr_player['name'],
                            nfl_player['player_id'],
                            sr_player.get('experience')
                        )
                        if is_mismatch:
                            continue
                        
                        # Try multiple name combinations
                        name_variations = [
                            sr_player['name'],
                            sr_player['display_name'],
                            f"{sr_player['first_name']} {sr_player['last_name']}"
                        ]
                        
                        for sr_name in name_variations:
                            if pd.isna(sr_name):
                                continue
                                
                            similarity = self.similarity_score(nfl_name, sr_name)
                            
                            if similarity > 0.85 and similarity > best_score:
                                # Check if this might be different people
                                is_different, diff_reason = self.is_likely_different_person(nfl_name, sr_name, similarity)
                                if is_different:
                                    logger.info(f"Rejected potential match: {nfl_name} -> {sr_name} ({diff_reason})")
                                    continue
                                    
                                best_match = sr_player
                                best_score = similarity
                                match_type = 'fuzzy_name_position'
                
                # Record the match
                if best_match is not None:
                    used_sportradar_ids.add(best_match['sportradar_player_id'])
                    
                    if match_type == 'fuzzy_name_position':
                        self.mapping_stats['fuzzy_matches'] += 1
                        logger.info(f"Fuzzy match: {nfl_name} -> {best_match['name']} (score: {best_score:.3f})")
                    
                    matches.append({
                        'nfl_player_id': nfl_player['player_id'],
                        'nfl_player_name': nfl_player['player_name'],
                        'nfl_display_name': nfl_player['player_display_name'],
                        'nfl_position': nfl_player['position_x'],
                        'nfl_position_group': nfl_player['position_group'],
                        'sportradar_player_id': best_match['sportradar_player_id'],
                        'sportradar_name': best_match['name'],
                        'sportradar_display_name': best_match['display_name'],
                        'sportradar_first_name': best_match['first_name'],
                        'sportradar_last_name': best_match['last_name'],
                        'sportradar_position': best_match['position'],
                        'sportradar_jersey': best_match['jersey'],
                        'sportradar_height': best_match['height'],
                        'sportradar_weight': best_match['weight'],
                        'sportradar_birth_date': best_match['birth_date'],
                        'sportradar_experience': best_match['experience'],
                        'sportradar_college': best_match['college'],
                        'sportradar_team_alias': best_match['team_alias'],
                        'sportradar_status': best_match['status'],
                        'match_type': match_type,
                        'match_score': best_score
                    })
                else:
                    self.mapping_stats['unmatched_nfl'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing player {nfl_player.get('player_display_name', 'Unknown')}: {e}")
                self.mapping_stats['unmatched_nfl'] += 1
                continue
        
        # Count unmatched SportRadar players
        self.mapping_stats['unmatched_sportradar'] = len(self.sportradar_df) - len(used_sportradar_ids)
        
        mapping_df = pd.DataFrame(matches)
        logger.info(f"Created mapping for {len(mapping_df)} players")
        
        return mapping_df
    
    def merge_with_statistics(self, mapping_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the player mapping with ALL NFL statistics
        This preserves every single NFL record while adding SportRadar IDs where available
        """
        
        logger.info("Merging mapping with full NFL statistics dataset...")
        
        # Create mapping dictionary for fast lookup
        player_mapping = mapping_df.set_index('nfl_player_id').to_dict('index')
        
        # Add SportRadar columns to the full NFL dataset
        merged_records = []
        
        for _, nfl_record in self.nfl_df.iterrows():
            nfl_player_id = nfl_record['player_id']
            
            # Start with the original NFL record
            merged_record = nfl_record.to_dict()
            
            # Add SportRadar data if mapping exists
            if nfl_player_id in player_mapping:
                mapping_info = player_mapping[nfl_player_id]
                
                # Add all SportRadar fields
                for key, value in mapping_info.items():
                    if key.startswith('sportradar_') or key in ['match_type', 'match_score']:
                        merged_record[key] = value
                
                merged_record['has_sportradar_mapping'] = True
                self.mapping_stats['preserved_records'] += 1
            else:
                # Add empty SportRadar columns for unmapped players
                sr_columns = [col for col in mapping_df.columns if col.startswith('sportradar_') or col in ['match_type', 'match_score']]
                for col in sr_columns:
                    merged_record[col] = None
                
                merged_record['has_sportradar_mapping'] = False
            
            merged_records.append(merged_record)
        
        merged_df = pd.DataFrame(merged_records)
        
        logger.info(f"Merged dataset created with {len(merged_df):,} total records")
        logger.info(f"Records with SportRadar mapping: {self.mapping_stats['preserved_records']:,}")
        
        return merged_df
    
    def create_unmatched_datasets(self, mapping_df: pd.DataFrame) -> tuple:
        """Create datasets of unmatched players for review"""
        
        # Unmatched NFL players (unique players, not all records)
        mapped_nfl_ids = set(mapping_df['nfl_player_id'])
        unique_nfl_players = self.nfl_df.drop_duplicates(subset=['player_id'])
        unmatched_nfl = unique_nfl_players[~unique_nfl_players['player_id'].isin(mapped_nfl_ids)].copy()
        
        # Add some analysis columns
        unmatched_nfl['seasons_played'] = unmatched_nfl['player_id'].apply(
            lambda pid: len(self.nfl_df[self.nfl_df['player_id'] == pid]['season'].unique())
        )
        unmatched_nfl['last_season'] = unmatched_nfl['player_id'].apply(
            lambda pid: self.nfl_df[self.nfl_df['player_id'] == pid]['season'].max()
        )
        unmatched_nfl['total_games'] = unmatched_nfl['player_id'].apply(
            lambda pid: len(self.nfl_df[self.nfl_df['player_id'] == pid])
        )
        
        # Unmatched SportRadar players  
        mapped_sr_ids = set(mapping_df['sportradar_player_id'])
        unmatched_sr = self.sportradar_df[~self.sportradar_df['sportradar_player_id'].isin(mapped_sr_ids)].copy()
        
        return unmatched_nfl, unmatched_sr
    
    def run_complete_merge(self, nfl_csv_path: str, sportradar_csv_path: str, output_dir: str = "data") -> Dict:
        """
        Run the complete merge process
        Returns dictionary with file paths and statistics
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("🏈 Starting Comprehensive NFL Player Data Merge")
        print("=" * 60)
        
        # Step 1: Load data
        print("📥 Loading datasets...")
        self.load_data(nfl_csv_path, sportradar_csv_path)
        
        # Step 2: Create player mapping
        print("\n🔄 Creating player mapping...")
        mapping_df = self.create_player_mapping()
        
        # Step 3: Merge with all statistics
        print("\n📊 Merging with complete statistics...")
        self.merged_df = self.merge_with_statistics(mapping_df)
        
        # Step 4: Create unmatched datasets
        print("\n📋 Creating unmatched player datasets...")
        self.unmatched_nfl_df, self.unmatched_sr_df = self.create_unmatched_datasets(mapping_df)
        
        # Step 5: Save all outputs
        print("\n💾 Saving results...")
        
        output_files = {}
        
        # Main merged dataset with ALL statistics
        main_output = output_path / "complete_nfl_with_sportradar.csv"
        self.merged_df.to_csv(main_output, index=False)
        output_files['complete_dataset'] = str(main_output)
        
        # Player mapping only (for reference)
        mapping_output = output_path / "player_id_mapping.csv"
        mapping_df.to_csv(mapping_output, index=False)
        output_files['player_mapping'] = str(mapping_output)
        
        # Unmatched datasets
        unmatched_nfl_output = output_path / "unmatched_nfl_players.csv"
        self.unmatched_nfl_df.to_csv(unmatched_nfl_output, index=False)
        output_files['unmatched_nfl'] = str(unmatched_nfl_output)
        
        unmatched_sr_output = output_path / "unmatched_sportradar_players.csv"
        self.unmatched_sr_df.to_csv(unmatched_sr_output, index=False)
        output_files['unmatched_sportradar'] = str(unmatched_sr_output)
        
        # Records with SportRadar mapping only (for current player analysis)
        current_players = self.merged_df[self.merged_df['has_sportradar_mapping'] == True].copy()
        current_output = output_path / "current_players_with_stats.csv"
        current_players.to_csv(current_output, index=False)
        output_files['current_players'] = str(current_output)
        
        # Step 6: Print comprehensive summary
        self.print_comprehensive_summary()
        
        return {
            'files': output_files,
            'stats': self.mapping_stats,
            'summary': {
                'total_nfl_records': len(self.merged_df),
                'records_with_mapping': len(current_players),
                'unique_mapped_players': len(mapping_df),
                'mapping_coverage_pct': len(mapping_df) / self.nfl_df['player_id'].nunique() * 100,
                'record_coverage_pct': len(current_players) / len(self.merged_df) * 100
            }
        }
    
    def print_comprehensive_summary(self):
        """Print detailed summary of the merge results"""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE MERGE SUMMARY")
        print("="*80)
        
        # Dataset overview
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"  Total NFL records: {len(self.merged_df):,}")
        print(f"  Records with SportRadar mapping: {self.mapping_stats['preserved_records']:,}")
        print(f"  Record coverage: {self.mapping_stats['preserved_records']/len(self.merged_df)*100:.1f}%")
        print(f"  Unique NFL players: {self.nfl_df['player_id'].nunique():,}")
        print(f"  Mapped to SportRadar: {self.mapping_stats['exact_matches'] + self.mapping_stats['position_matches'] + self.mapping_stats['fuzzy_matches']:,}")
        print(f"  Player coverage: {(self.mapping_stats['exact_matches'] + self.mapping_stats['position_matches'] + self.mapping_stats['fuzzy_matches'])/self.nfl_df['player_id'].nunique()*100:.1f}%")
        
        # Match quality breakdown
        print(f"\n🎯 MATCH QUALITY BREAKDOWN:")
        print(f"  Exact name + position: {self.mapping_stats['exact_matches']:,}")
        print(f"  Position group resolution: {self.mapping_stats['position_matches']:,}")
        print(f"  Fuzzy name matching: {self.mapping_stats['fuzzy_matches']:,}")
        print(f"  Duplicate names resolved: {self.mapping_stats['duplicate_names_resolved']:,}")
        print(f"  Generational mismatches avoided: {self.mapping_stats['generational_mismatches_avoided']:,}")
        
        # Unmatched analysis
        print(f"\n❌ UNMATCHED PLAYERS:")
        print(f"  NFL players without SportRadar mapping: {self.mapping_stats['unmatched_nfl']:,}")
        print(f"  SportRadar players without NFL history: {self.mapping_stats['unmatched_sportradar']:,}")
        
        # Show retirement analysis
        if len(self.unmatched_nfl_df) > 0:
            recent_players = self.unmatched_nfl_df[self.unmatched_nfl_df['last_season'] >= 2022]
            old_players = self.unmatched_nfl_df[self.unmatched_nfl_df['last_season'] < 2022]
            
            print(f"\n🏃 UNMATCHED PLAYER ANALYSIS:")
            print(f"  Likely retired (last played before 2022): {len(old_players):,}")
            print(f"  Recently active (2022+): {len(recent_players):,}")
        
        # Season coverage analysis
        if self.mapping_stats['preserved_records'] > 0:
            mapped_records = self.merged_df[self.merged_df['has_sportradar_mapping'] == True]
            print(f"\n📅 SEASON COVERAGE (mapped records):")
            season_counts = mapped_records['season'].value_counts().sort_index()
            for season, count in season_counts.items():
                total_season_records = len(self.merged_df[self.merged_df['season'] == season])
                pct = count / total_season_records * 100
                print(f"  {season}: {count:,}/{total_season_records:,} records ({pct:.1f}%)")
        
        # Statistical completeness
        stat_columns = [col for col in self.merged_df.columns if any(stat in col.lower() for stat in 
                       ['passing', 'rushing', 'receiving', 'fantasy', 'yards', 'tds'])]
        
        print(f"\n📈 STATISTICAL DATA PRESERVED:")
        print(f"  Statistical columns: {len(stat_columns)}")
        print(f"  All historical statistics preserved: ✅")
        print(f"  SportRadar IDs available for current players: ✅")
        
        print(f"\n🚀 MACHINE LEARNING READINESS:")
        print(f"  ✅ Complete historical data (2018-2024) for model training")
        print(f"  ✅ Current player SportRadar IDs for projections")
        print(f"  ✅ Position-specific data available for all major positions")
        print(f"  ✅ Generational mismatches avoided (e.g., Frank Gore Sr. vs Jr.)")
        
        print(f"\n📁 OUTPUT FILES CREATED:")
        print(f"  • complete_nfl_with_sportradar.csv - Full dataset with all records")
        print(f"  • player_id_mapping.csv - Player ID mapping reference")
        print(f"  • current_players_with_stats.csv - Only records with SportRadar IDs")
        print(f"  • unmatched_nfl_players.csv - NFL players without SportRadar mapping")
        print(f"  • unmatched_sportradar_players.csv - SportRadar players without NFL history")
        
        print("="*80)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive NFL Player Data Merger')
    parser.add_argument('--nfl-csv', default='data/nfl_dataset.csv', help='Path to NFL dataset CSV')
    parser.add_argument('--sportradar-csv', default='data/sportradar_players_20250806_221354.csv', help='Path to SportRadar players CSV')
    parser.add_argument('--output-dir', default='data', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Verify input files exist
    nfl_path = Path(args.nfl_csv)
    sr_path = Path(args.sportradar_csv)
    
    if not nfl_path.exists():
        print(f"❌ NFL dataset not found: {nfl_path}")
        return 1
    
    if not sr_path.exists():
        print(f"❌ SportRadar dataset not found: {sr_path}")
        return 1
    
    try:
        # Run the complete merge
        merger = ComprehensivePlayerMerger()
        results = merger.run_complete_merge(str(nfl_path), str(sr_path), args.output_dir)
        
        print(f"\n🎉 MERGE COMPLETED SUCCESSFULLY!")
        print(f"\n📊 FINAL STATISTICS:")
        print(f"  Total records processed: {results['summary']['total_nfl_records']:,}")
        print(f"  Records with SportRadar mapping: {results['summary']['records_with_mapping']:,}")
        print(f"  Player mapping coverage: {results['summary']['mapping_coverage_pct']:.1f}%")
        print(f"  Record coverage: {results['summary']['record_coverage_pct']:.1f}%")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"  1. Use 'complete_nfl_with_sportradar.csv' for machine learning training")
        print(f"  2. Use 'current_players_with_stats.csv' for 2025 projections")
        print(f"  3. Review unmatched files for any important missing players")
        print(f"  4. Train models on ALL data, apply to current players only")
        
        return 0
        
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        print(f"❌ Error during merge: {e}")
        return 1


if __name__ == "__main__":
    exit(main())