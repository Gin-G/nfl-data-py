# Quick test to see what's actually in the roster
from nfl_data_py import import_weekly_rosters
rosters = import_weekly_rosters([2025])
print(rosters.columns.tolist())
print(rosters[['player_id', 'sportradar_id', 'player_name']].head())