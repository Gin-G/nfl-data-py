import logging
import pandas as pd
import re

def normalize_name(name):
    # Remove suffixes like 'Jr.', 'Sr.', 'III', etc.
    name = re.sub(r'\s+(Jr\.|Sr\.|I{2,}|IV)$', '', name)
    # Remove middle initials
    name = re.sub(r'\s+[A-Z]\.\s+', ' ', name)
    # Remove any remaining periods and convert to lowercase
    return name.replace('.', '').lower().strip()

def merge_dataframes(fanduel_df, predictions_df):
    # Normalize names in both dataframes
    fanduel_df['Normalized_Name'] = fanduel_df['Nickname'].apply(normalize_name)
    predictions_df['Normalized_Name'] = predictions_df['player_name'].apply(normalize_name)
    
    # Merge dataframes based on normalized names
    merged_df = pd.merge(fanduel_df, predictions_df, on='Normalized_Name', how='left')
    
    # Drop the temporary normalized name columns
    merged_df = merged_df.drop(columns=['Normalized_Name'])
    
    return merged_df

# Load the data
logging.info("Loading data from CSV...")
qbpredictions_df = pd.read_csv('predictions/QB_predictions_week5_season2024.csv')
rbpredictions_df = pd.read_csv('predictions/RB_predictions_week5_season2024.csv')
tepredictions_df = pd.read_csv('predictions/TE_predictions_week5_season2024.csv')
wrpredictions_df = pd.read_csv('predictions/WR_predictions_week5_season2024.csv')
fanduel_df = pd.read_csv('data/FanDuel-NFL-2024-Week5-MainSlate-players-list.csv')

# Combine all prediction dataframes
all_predictions = pd.concat([qbpredictions_df, rbpredictions_df, tepredictions_df, wrpredictions_df], ignore_index=True)

# Merge FanDuel data with predictions
df = merge_dataframes(fanduel_df, all_predictions)

# Calculate value
df['value'] = (df['fanduel_fantasy_points'] / df['Salary']) * 1000

# Sort and save to CSV
df.sort_values(['value'], ascending=False).to_csv('predictions/fanduel_value_week5.csv', index=False)

# Display the resulting dataframe
print(df)