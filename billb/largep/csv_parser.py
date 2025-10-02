#!/usr/bin/env python3
"""
Check the final nfl_dataset.csv to verify 2025 data is present
"""

import pandas as pd

print("Loading nfl_dataset.csv...")
df = pd.read_csv('data/nfl_dataset.csv')

print(f"\n{'='*70}")
print("DATASET OVERVIEW")
print(f"{'='*70}")
print(f"Total records: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Seasons covered: {sorted(df['season'].unique())}")

print(f"\n{'='*70}")
print("2025 DATA CHECK")
print(f"{'='*70}")

df_2025 = df[df['season'] == 2025]

if len(df_2025) == 0:
    print("❌ NO 2025 DATA FOUND IN DATASET")
else:
    print(f"✅ 2025 data present: {len(df_2025):,} records")
    
    # Check weeks
    weeks_2025 = df_2025[df_2025['week'] != 'AVG']['week'].unique()
    print(f"\nWeeks in 2025: {sorted(weeks_2025)}")
    
    # Check unique players
    unique_players = df_2025['player_id'].nunique()
    print(f"Unique players: {unique_players}")
    
    # Check for headshot_url
    if 'headshot_url' in df_2025.columns:
        has_headshot = df_2025['headshot_url'].notna().sum()
        print(f"\nHeadshot URLs:")
        print(f"  Present: {has_headshot}/{len(df_2025)} ({has_headshot/len(df_2025)*100:.1f}%)")
        print(f"  Missing: {df_2025['headshot_url'].isna().sum()}")
    else:
        print(f"\n❌ headshot_url column missing")
    
    # Check sample players
    print(f"\nSample 2025 players (Week 1):")
    sample = df_2025[df_2025['week'] == 1].nlargest(5, 'fanduel_fantasy_points')
    cols_to_show = ['player_name', 'position', 'week', 'fanduel_fantasy_points']
    if 'headshot_url' in sample.columns:
        cols_to_show.append('headshot_url')
    
    for idx, row in sample.iterrows():
        headshot_status = "✓" if pd.notna(row.get('headshot_url')) else "✗"
        print(f"  {headshot_status} {row['player_name']:20s} {row['position']:3s} - {row['fanduel_fantasy_points']:.1f} pts")
    
    # Check for Sportradar ID column
    sportradar_cols = [col for col in df_2025.columns if 'sportradar' in col.lower()]
    if sportradar_cols:
        print(f"\nSportradar ID columns found: {sportradar_cols}")
        for col in sportradar_cols:
            non_null = df_2025[col].notna().sum()
            print(f"  {col}: {non_null} non-null values")
    else:
        print(f"\n⚠️  No Sportradar ID column found")

print(f"\n{'='*70}")
print("COLUMN ANALYSIS")
print(f"{'='*70}")

# Check key columns
key_columns = [
    'player_id', 'player_name', 'player_display_name', 'position', 
    'headshot_url', 'fanduel_fantasy_points', 'avg_fppg',
    'offensive_snaps', 'offensive_snap_pct'
]

print("\nKey column presence in 2025 data:")
for col in key_columns:
    if col in df_2025.columns:
        non_null = df_2025[col].notna().sum()
        pct = non_null / len(df_2025) * 100
        print(f"  ✓ {col:30s} {non_null:5,}/{len(df_2025):5,} ({pct:5.1f}%)")
    else:
        print(f"  ✗ {col:30s} MISSING")

print(f"\n{'='*70}")
print("COMPARISON: 2024 vs 2025")
print(f"{'='*70}")

df_2024 = df[(df['season'] == 2024) & (df['week'] == 1)]
df_2025_w1 = df_2025[df_2025['week'] == 1]

print(f"\nWeek 1 comparison:")
print(f"  2024: {len(df_2024):,} records")
print(f"  2025: {len(df_2025_w1):,} records")

if 'headshot_url' in df.columns:
    headshots_2024 = df_2024['headshot_url'].notna().sum()
    headshots_2025 = df_2025_w1['headshot_url'].notna().sum()
    print(f"\nHeadshot URLs in Week 1:")
    print(f"  2024: {headshots_2024}/{len(df_2024)} ({headshots_2024/len(df_2024)*100:.1f}%)")
    print(f"  2025: {headshots_2025}/{len(df_2025_w1)} ({headshots_2025/len(df_2025_w1)*100:.1f}%)")

if 'offensive_snaps' in df.columns:
    snaps_2024 = (df_2024['offensive_snaps'] > 0).sum()
    snaps_2025 = (df_2025_w1['offensive_snaps'] > 0).sum()
    print(f"\nOffensive snaps in Week 1:")
    print(f"  2024: {snaps_2024}/{len(df_2024)} ({snaps_2024/len(df_2024)*100:.1f}%)")
    print(f"  2025: {snaps_2025}/{len(df_2025_w1)} ({snaps_2025/len(df_2025_w1)*100:.1f}%)")

print(f"\n{'='*70}")
print("VERDICT")
print(f"{'='*70}\n")

if len(df_2025) == 0:
    print("❌ 2025 data is NOT in the dataset")
    print("   The merge likely failed silently")
elif len(df_2025) < 1000:
    print("⚠️  2025 data is present but seems incomplete")
    print(f"   Only {len(df_2025)} records (expected ~4400)")
else:
    print("✅ 2025 data successfully integrated!")
    print(f"   {len(df_2025):,} records across {len(weeks_2025)} weeks")
    
    if 'headshot_url' in df_2025.columns:
        pct = (df_2025['headshot_url'].notna().sum() / len(df_2025)) * 100
        if pct > 80:
            print(f"   ✅ Headshots: {pct:.1f}% coverage")
        else:
            print(f"   ⚠️  Headshots: {pct:.1f}% coverage (lower than expected)")

print()