"""FanDuel lineup optimizer.

Works from a FanDuel main-slate player list CSV merged with our projections
(see merge_fanduel_salaries), building diverse lineups under the salary cap
with a per-player usage limit.
"""

import re
from collections import Counter

import numpy as np
import pandas as pd

ROSTER_SLOTS = {
    "QB": 1,
    "RB/FLEX": 2,
    "WR/FLEX": 3,
    "TE/FLEX": 1,
    "DEF": 1,
    "FLEX": 1,
}
FLEX_ELIGIBLE = ["RB/FLEX", "WR/FLEX", "TE/FLEX"]
EXCLUDED_INJURY_STATUSES = ["IR", "O", "D"]


def normalize_name(name):
    """Normalize FanDuel nicknames for matching against our projections."""
    name = re.sub(r"\s+(Jr\.|Sr\.|I{2,}|IV)$", "", name)
    name = re.sub(r"\s+[A-Z]\.\s+", " ", name)  # middle initials
    return name.replace(".", "").lower().strip()


def merge_fanduel_salaries(fanduel_df, predictions_df):
    """Join a FanDuel salary export with our projections by normalized name.

    Adds a 'value' column (projected points per $1000 of salary).
    """
    fanduel_df = fanduel_df.copy()
    predictions_df = predictions_df.copy()
    fanduel_df["Normalized_Name"] = fanduel_df["Nickname"].apply(normalize_name)
    predictions_df["Normalized_Name"] = predictions_df["player_name"].apply(normalize_name)

    merged = pd.merge(fanduel_df, predictions_df, on="Normalized_Name", how="left")
    merged = merged.drop(columns=["Normalized_Name"])
    merged["value"] = (merged["fanduel_fantasy_points"] / merged["Salary"]) * 1000
    return merged


def _get_top_n_by_position(df, position, n=5):
    return df[df["Roster Position"] == position].nlargest(n, "lineup_points")


def _is_lineup_unique(new_lineup, existing_lineups):
    new_set = set(new_lineup["Nickname"].values)
    return all(set(l["Nickname"].values) != new_set for l in existing_lineups)


def calculate_player_usage(lineups):
    """Usage counts and percentages for players across generated lineups."""
    all_players = []
    for lineup in lineups:
        all_players.extend(lineup["Nickname"].tolist())
    counts = Counter(all_players)

    usage = pd.DataFrame.from_dict(counts, orient="index", columns=["Count"])
    usage["Usage_Percentage"] = (usage["Count"] / len(lineups) * 100).round(2)
    return usage.sort_values("Usage_Percentage", ascending=False)


def _check_usage_limit(nickname, current_lineups, num_lineups, max_usage_percentage):
    if not current_lineups:
        return True
    current = sum(1 for l in current_lineups if nickname in l["Nickname"].values)
    return (current + 1) / num_lineups * 100 <= max_usage_percentage


def optimize_lineups(df, num_lineups=5, salary_cap=60000, exclude_players=None,
                     max_usage_percentage=50):
    """Build diverse FanDuel lineups from a merged salary+projection frame.

    Args:
        df: output of merge_fanduel_salaries (needs Salary, FPPG,
            Roster Position, Nickname, fanduel_fantasy_points columns)
        num_lineups: how many unique lineups to generate
        salary_cap: FanDuel salary cap
        exclude_players: nicknames to leave out entirely
        max_usage_percentage: cap on how often one player appears

    Returns a list of lineup DataFrames.
    """
    df = df.copy()

    df["Injury Indicator"] = df.get("Injury Indicator", pd.Series("", index=df.index)).fillna("")
    df = df[~df["Injury Indicator"].str.upper().isin(EXCLUDED_INJURY_STATUSES)]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Salary", "FPPG", "Roster Position", "Nickname"]
    )
    if exclude_players:
        df = df[~df["Nickname"].isin(set(exclude_players))]

    # DEF has no model projection; use FanDuel's FPPG
    df["lineup_points"] = df["fanduel_fantasy_points"]
    df.loc[df["Roster Position"] == "DEF", "lineup_points"] = df.loc[
        df["Roster Position"] == "DEF", "FPPG"
    ]

    strategies = [("QB", 5), ("RB/FLEX", 5), ("WR/FLEX", 5), ("TE/FLEX", 5)]
    all_lineups = []
    current_lineup = 0
    attempts = 0
    max_attempts = num_lineups * 10

    while current_lineup < num_lineups and attempts < max_attempts:
        attempts += 1
        focus_position, n_players = strategies[min(current_lineup // 5, len(strategies) - 1)]

        for _, focus_player in _get_top_n_by_position(df, focus_position, n_players).iterrows():
            if current_lineup >= num_lineups:
                break
            if not _check_usage_limit(focus_player["Nickname"], all_lineups,
                                      num_lineups, max_usage_percentage):
                continue

            current_df = df.copy()
            lineup = [focus_player]
            remaining_salary = salary_cap - focus_player["Salary"]
            positions_needed = dict(ROSTER_SLOTS)
            positions_needed[focus_player["Roster Position"]] -= 1

            lineup_complete = True
            for position in ROSTER_SLOTS:
                if position == focus_player["Roster Position"]:
                    continue

                if position == "FLEX":
                    available = current_df[current_df["Roster Position"].isin(FLEX_ELIGIBLE)]
                else:
                    available = current_df[current_df["Roster Position"] == position]

                available = available.copy()
                available["value"] = available["lineup_points"] / available["Salary"]
                available = available.sort_values("value", ascending=False)

                count = positions_needed[position]
                for _, player in available.iterrows():
                    if count > 0 and player["Salary"] <= remaining_salary:
                        if _check_usage_limit(player["Nickname"], all_lineups,
                                              num_lineups, max_usage_percentage):
                            lineup.append(player)
                            remaining_salary -= player["Salary"]
                            count -= 1
                            current_df = current_df[current_df["Nickname"] != player["Nickname"]]
                    if count == 0:
                        break
                if count > 0:
                    lineup_complete = False
                    break
                positions_needed[position] = count

            if lineup_complete and sum(positions_needed.values()) == 0:
                lineup_df = pd.DataFrame(lineup)
                if _is_lineup_unique(lineup_df, all_lineups):
                    all_lineups.append(lineup_df)
                    current_lineup += 1
                    print(f"Created lineup {current_lineup}")
                    if current_lineup >= num_lineups:
                        break

    if current_lineup < num_lineups:
        print(f"Warning: only generated {current_lineup}/{num_lineups} valid "
              f"lineups under the usage limits")
    return all_lineups


def optimize_from_csv(csv_file, num_lineups=5, salary_cap=60000, exclude_players=None,
                      max_usage_percentage=50):
    """Optimize lineups from a merged salary+projection CSV file."""
    df = pd.read_csv(csv_file)
    return optimize_lineups(df, num_lineups=num_lineups, salary_cap=salary_cap,
                            exclude_players=exclude_players,
                            max_usage_percentage=max_usage_percentage)


def display_lineups(lineups):
    """Print each lineup with salary and projected point totals."""
    if not lineups:
        print("No valid lineups found.")
        return
    for i, lineup in enumerate(lineups, 1):
        print(f"\nLineup {i}:")
        cols = [c for c in ["Roster Position", "Nickname", "Salary", "lineup_points",
                            "Injury Indicator"] if c in lineup.columns]
        print(lineup[cols])
        print(f"Total Salary: ${lineup['Salary'].sum():,.0f}")
        print(f"Projected Points: {lineup['lineup_points'].sum():.2f}")

    print("\nTop 20 most used players:")
    print(calculate_player_usage(lineups).head(20))
