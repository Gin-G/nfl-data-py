# depth_chart_integration.py
import pandas as pd
import requests
import json
from typing import Dict, List, Optional
import logging
import os

class DepthChartManager:
    """Manages depth chart data from SportRadar API and integrates with NFL dataset"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.sportradar.com/nfl/official/trial/v7/en"
        self.depth_chart_cache = {}
    
    def fetch_depth_chart(self, season: int, season_type: str = "REG", week: int = None) -> Dict:
        """Fetch depth chart data from SportRadar API"""
        
        # Determine the correct final week based on season
        if week is None:
            if season >= 2021:
                week = 18  # 18-week season
            else:
                week = 17  # 17-week season
        
        url = f"{self.base_url}/seasons/{season}/{season_type}/{week}/depth_charts.json"
        headers = {
            "accept": "application/json",
            "x-api-key": self.api_key
        }
        
        cache_key = f"{season}_{season_type}_{week}"
        if cache_key in self.depth_chart_cache:
            return self.depth_chart_cache[cache_key]
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            self.depth_chart_cache[cache_key] = data
            return data
        except Exception as e:
            logging.error(f"Failed to fetch depth chart for {season} week {week}: {e}")
            return {}
    
    def parse_depth_charts(self, depth_chart_data: Dict) -> pd.DataFrame:
        """Parse depth chart JSON into a flat DataFrame"""
        records = []
        
        if 'teams' not in depth_chart_data:
            return pd.DataFrame()
        
        for team in depth_chart_data['teams']:
            team_alias = team.get('alias', '')
            team_name = team.get('name', '')
            
            # Parse offense positions
            if 'offense' in team:
                for position_group in team['offense']:
                    position_info = position_group.get('position', {})
                    position_name = position_info.get('name', '')
                    
                    for player in position_info.get('players', []):
                        records.append({
                            'team': team_alias,
                            'team_name': team_name,
                            'player_name': player.get('name', ''),
                            'player_id': player.get('id', ''),
                            'sr_id': player.get('sr_id', ''),
                            'jersey': player.get('jersey', ''),
                            'position': player.get('position', position_name),
                            'depth': player.get('depth', 99),
                            'side': 'offense'
                        })
            
            # Parse defense positions (if needed)
            if 'defense' in team:
                for position_group in team['defense']:
                    position_info = position_group.get('position', {})
                    position_name = position_info.get('name', '')
                    
                    for player in position_info.get('players', []):
                        records.append({
                            'team': team_alias,
                            'team_name': team_name,
                            'player_name': player.get('name', ''),
                            'player_id': player.get('id', ''),
                            'sr_id': player.get('sr_id', ''),
                            'jersey': player.get('jersey', ''),
                            'position': player.get('position', position_name),
                            'depth': player.get('depth', 99),
                            'side': 'defense'
                        })
        
        return pd.DataFrame(records)
    
    def get_historical_depth_charts(self, seasons: List[int]) -> pd.DataFrame:
        """Fetch depth charts for multiple seasons with rate limiting"""
        import time
        all_depth_charts = []
        
        # Try all seasons from 2018 onwards
        available_seasons = [s for s in seasons if s >= 2018]
        
        for i, season in enumerate(available_seasons):
            logging.info(f"Fetching depth chart for {season}")
            depth_data = self.fetch_depth_chart(season)
            
            if depth_data:
                df = self.parse_depth_charts(depth_data)
                df['season'] = season
                all_depth_charts.append(df)
                logging.info(f"Successfully fetched {len(df)} players for {season}")
            else:
                logging.warning(f"No depth chart data for {season}")
            
            # Rate limiting: wait between requests
            if i < len(available_seasons) - 1:  # Don't wait after last request
                logging.info("Waiting 3 seconds to avoid rate limiting...")
                time.sleep(3)
        
        if all_depth_charts:
            combined_df = pd.concat(all_depth_charts, ignore_index=True)
            logging.info(f"Total depth chart records: {len(combined_df)}")
            return combined_df
        else:
            logging.error("No depth chart data retrieved for any season")
            return pd.DataFrame()
    
    def create_player_name_mapping(self, nfl_data: pd.DataFrame) -> Dict[str, str]:
        """Create mapping between SportRadar names and NFL data names"""
        # This helps handle name variations between data sources
        name_mapping = {}
        
        unique_players = nfl_data[['player_name', 'recent_team', 'position_x']].drop_duplicates()
        
        for _, player in unique_players.iterrows():
            # Create variations of the name to match against SportRadar
            name = player['player_name']
            
            # Handle common name variations
            variations = [
                name,
                name.replace('.', ''),
                name.replace(' Jr.', ''),
                name.replace(' Sr.', ''),
                name.replace(' III', ''),
                name.replace(' II', ''),
            ]
            
            for variation in variations:
                name_mapping[variation] = name
        
        return name_mapping
    
    def integrate_depth_charts_with_nfl_data(self, nfl_data: pd.DataFrame, 
                                           depth_charts: pd.DataFrame) -> pd.DataFrame:
        """Add depth chart information to NFL dataset using SportRadar player IDs"""
        
        logging.info("Starting depth chart integration using SportRadar player IDs...")
        logging.info(f"NFL data shape: {nfl_data.shape}")
        logging.info(f"Depth chart data shape: {depth_charts.shape}")
        
        # Check if sportradar_player_id exists in NFL data
        if 'sportradar_player_id' not in nfl_data.columns:
            logging.error("sportradar_player_id column not found in NFL data!")
            logging.info(f"Available columns: {nfl_data.columns.tolist()}")
            return nfl_data
        
        # Debug: Show some sample data
        if not depth_charts.empty:
            logging.info(f"Depth chart seasons: {sorted(depth_charts['season'].unique())}")
            logging.info(f"Sample depth chart player IDs: {depth_charts['player_id'].head().tolist()}")
        
        # Check NFL data SportRadar IDs
        nfl_sr_ids = nfl_data['sportradar_player_id'].dropna().unique()
        depth_sr_ids = depth_charts['player_id'].dropna().unique()
        
        logging.info(f"NFL data has {len(nfl_sr_ids)} unique SportRadar player IDs")
        logging.info(f"Depth chart has {len(depth_sr_ids)} unique SportRadar player IDs")
        
        # Find overlapping IDs
        overlapping_ids = set(nfl_sr_ids).intersection(set(depth_sr_ids))
        logging.info(f"Found {len(overlapping_ids)} overlapping SportRadar player IDs")
        
        if len(overlapping_ids) == 0:
            logging.warning("No overlapping SportRadar player IDs found!")
            logging.info(f"Sample NFL SportRadar IDs: {nfl_sr_ids[:5].tolist()}")
            logging.info(f"Sample depth chart IDs: {depth_sr_ids[:5].tolist()}")
            
            # Add dummy columns and return
            nfl_data['depth'] = 99
            nfl_data['is_starter'] = 0
            nfl_data['depth_tier'] = 'deep_bench'
            return nfl_data
        
        # Create lookup dictionary: (player_id, season) -> depth info
        depth_lookup = {}
        for _, row in depth_charts.iterrows():
            key = (row['player_id'], row['season'])
            depth_lookup[key] = {
                'depth': row['depth'],
                'position': row['position'],
                'team': row['team'],
                'player_name': row['player_name']
            }
        
        logging.info(f"Created depth lookup with {len(depth_lookup)} entries")
        
        # Add depth information to NFL data
        nfl_data['depth'] = 99
        nfl_data['depth_chart_name'] = ''
        nfl_data['depth_chart_team'] = ''
        
        matched_records = 0
        
        for idx, row in nfl_data.iterrows():
            if pd.isna(row['sportradar_player_id']):
                continue
                
            key = (row['sportradar_player_id'], row['season'])
            
            if key in depth_lookup:
                depth_info = depth_lookup[key]
                nfl_data.at[idx, 'depth'] = depth_info['depth']
                nfl_data.at[idx, 'depth_chart_name'] = depth_info['player_name']
                nfl_data.at[idx, 'depth_chart_team'] = depth_info['team']
                matched_records += 1
        
        logging.info(f"Successfully matched {matched_records} records using SportRadar player IDs")
        
        # Create starter flag and depth tier
        nfl_data['is_starter'] = (nfl_data['depth'] == 1).astype(int)
        nfl_data['depth_tier'] = pd.cut(
            nfl_data['depth'],
            bins=[0, 1, 2, 3, float('inf')],
            labels=['starter', 'backup', 'third_string', 'deep_bench']
        )
        
        # Log success stats
        starters_found = nfl_data['is_starter'].sum()
        logging.info(f"Found {starters_found} starter records")
        
        if starters_found > 0:
            starters = nfl_data[nfl_data['is_starter'] == 1]
            logging.info(f"Starters by position: {starters['position_x'].value_counts().to_dict()}")
            
            # Show some examples of successful matches
            sample_starters = starters[['player_name', 'depth_chart_name', 'recent_team', 'season', 'depth']].head(10)
            logging.info("Sample starter matches:")
            for _, starter in sample_starters.iterrows():
                logging.info(f"  {starter['player_name']} ({starter['recent_team']}, {starter['season']}) -> depth {starter['depth']}")
        
        return nfl_data

def update_nfl_dataset_with_depth_charts(csv_path: str, api_key: str, 
                                        output_path: str = None) -> pd.DataFrame:
    """Main function to update NFL dataset with depth chart data"""
    
    # Load existing NFL data
    logging.info(f"Loading NFL data from {csv_path}")
    nfl_data = pd.read_csv(csv_path)
    
    # Get unique seasons from the data
    seasons = sorted(nfl_data['season'].unique())
    logging.info(f"Found seasons: {seasons}")
    
    # Initialize depth chart manager
    depth_manager = DepthChartManager(api_key)
    
    # Fetch depth charts for all seasons
    depth_charts = depth_manager.get_historical_depth_charts(seasons)
    
    if depth_charts.empty:
        logging.warning("No depth chart data found")
        # Add dummy depth chart columns
        nfl_data['depth'] = 99
        nfl_data['is_starter'] = 0
        nfl_data['depth_tier'] = 'unknown'
        return nfl_data
    
    # Integrate depth chart data
    logging.info("Integrating depth chart data with NFL dataset")
    enhanced_data = depth_manager.integrate_depth_charts_with_nfl_data(nfl_data, depth_charts)
    
    # Save enhanced dataset
    if output_path:
        enhanced_data.to_csv(output_path, index=False)
        logging.info(f"Enhanced dataset saved to {output_path}")
    
    return enhanced_data

def predict_2025_starters(depth_charts_2024: pd.DataFrame) -> Dict[str, Dict]:
    """Predict likely 2025 starters based on 2024 depth charts"""
    
    # Get the latest depth chart positions for each team/position
    latest_starters = depth_charts_2024[depth_charts_2024['depth'] == 1].groupby(['team', 'position']).first()
    
    predicted_starters = {}
    
    for (team, position), player_info in latest_starters.iterrows():
        if team not in predicted_starters:
            predicted_starters[team] = {}
        
        predicted_starters[team][position] = {
            'player_name': player_info['player_name'],
            'confidence': 'high',  # Could be more sophisticated
            'reasoning': 'was_2024_starter'
        }
    
    return predicted_starters

# Enhanced prediction function that considers depth charts
def enhanced_predict_player_week(player_name: str, season: int, week: int, 
                               depth_charts: pd.DataFrame = None):
    """Enhanced prediction function that considers starter status"""
    
    # Your existing prediction logic here...
    # (from billb_model3.py)
    
    prediction = predict_player_week(player_name, season, week)  # Your existing function
    
    if prediction and depth_charts is not None:
        # Get player's depth chart info
        player_depth = depth_charts[
            (depth_charts['player_name'] == player_name) & 
            (depth_charts['season'] == season - 1)  # Use previous season's depth chart
        ]
        
        if not player_depth.empty:
            depth = player_depth.iloc[0]['depth']
            
            # Adjust predictions based on depth chart position
            if depth == 1:  # Starter
                prediction['starter_confidence'] = 'high'
                prediction['snap_expectation'] = 'full_game'
            elif depth == 2:  # Backup
                prediction['starter_confidence'] = 'low'
                prediction['snap_expectation'] = 'limited'
                # Scale down projections
                for stat in ['passing_yards', 'rushing_yards', 'receiving_yards', 'fanduel_fantasy_points']:
                    if stat in prediction:
                        prediction[stat] *= 0.3  # Backup gets ~30% of starter stats
            else:  # Deep bench
                prediction['starter_confidence'] = 'very_low'
                prediction['snap_expectation'] = 'minimal'
                # Heavy scaling
                for stat in ['passing_yards', 'rushing_yards', 'receiving_yards', 'fanduel_fantasy_points']:
                    if stat in prediction:
                        prediction[stat] *= 0.1
        else:
            prediction['starter_confidence'] = 'unknown'
            prediction['snap_expectation'] = 'unknown'
    
    return prediction

# Example usage
if __name__ == "__main__":
    # Your SportRadar API key
    API_KEY = os.getenv('SPORTRADAR_API_KEY')
    
    # Update your dataset with depth chart data
    enhanced_data = update_nfl_dataset_with_depth_charts(
        csv_path='data/complete_nfl_with_sportradar.csv',
        api_key=API_KEY,
        output_path='data/nfl_with_depth_charts.csv'
    )
    
    print(f"Enhanced dataset shape: {enhanced_data.shape}")
    print(f"Columns added: depth, is_starter, depth_tier")
    
    # Show starter distribution
    print("\nStarter distribution by position:")
    if 'is_starter' in enhanced_data.columns and not enhanced_data[enhanced_data['is_starter'] == 1].empty:
        starters = enhanced_data[enhanced_data['is_starter'] == 1]
        print(starters['position_x'].value_counts())
    else:
        print("No starter data found - depth charts may not have been available")