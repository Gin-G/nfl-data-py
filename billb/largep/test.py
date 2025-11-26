"""
Test script to see what nflreadpy actually returns
Run this in your Flask app context or Python shell
"""
import nflreadpy as nfl
import pandas as pd

print(nfl.get_current_week())