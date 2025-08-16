import nfl_data_py as nfl
import pandas as pd
picks = nfl.import_draft_picks([2025])

# See all columns
print("All columns:")
print(picks.columns.tolist())

# See first row with all data
pd.set_option('display.max_columns', None)
print("\nFirst row with all columns:")
print(picks.head(1))

# Look for name-related columns specifically
name_cols = [col for col in picks.columns if 'name' in col.lower()]
print(f"\nName-related columns: {name_cols}")

if name_cols:
    print("\nSample name data:")
    print(picks[['season', 'round', 'pick', 'team'] + name_cols].head(10))