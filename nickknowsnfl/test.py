#!/usr/bin/env python3
"""
Test script to check nflreadpy roster columns
"""

import nflreadpy as nfl
import pandas as pd

print("Loading rosters from nflreadpy...")
rosters = nfl.load_rosters_weekly(seasons=[2025])

# Convert from Polars to Pandas if needed
if hasattr(rosters, 'to_pandas'):
    rosters = rosters.to_pandas()

print(f"\nRoster data shape: {rosters.shape}")
print(f"\nAvailable columns:")
for col in sorted(rosters.columns):
    print(f"  - {col}")

print(f"\nFirst few rows:")
print(rosters.head())

# Check for player ID related columns
player_id_cols = [col for col in rosters.columns if 'id' in col.lower() or 'player' in col.lower()]
print(f"\nPlayer/ID related columns:")
for col in player_id_cols:
    print(f"  - {col}")
    non_null = rosters[col].notna().sum()
    total = len(rosters)
    print(f"    Coverage: {non_null}/{total} ({non_null/total*100:.1f}%)")