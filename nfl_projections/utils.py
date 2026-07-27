"""Shared helpers: name normalization, week parsing, nflreadpy conversion."""

import re

import pandas as pd

NAME_SUFFIXES = [
    r"\s+Jr\.?$",
    r"\s+Sr\.?$",
    r"\s+III$",
    r"\s+II$",
    r"\s+IV$",
    r"\s+V$",
    r"\s+VI$",
]


def to_pandas(df):
    """Convert a Polars frame from nflreadpy to pandas (no-op if already pandas)."""
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    return df


def normalize_player_name(name):
    """Lowercase a player name and strip suffixes (Jr., III, ...) for matching."""
    if pd.isna(name) or not name:
        return ""
    name = str(name).strip()
    for suffix in NAME_SUFFIXES:
        name = re.sub(suffix, "", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip().lower()


def parse_weeks(spec):
    """Parse a week spec like '3', '1-17', or '1,2,5' into a sorted list of ints."""
    weeks = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            weeks.update(range(int(start), int(end) + 1))
        else:
            weeks.add(int(part))
    return sorted(weeks)


def regular_games(df):
    """Drop season-average rows (week == 'AVG') and coerce week to numeric."""
    games = df[df["week"] != "AVG"].copy()
    games["week"] = pd.to_numeric(games["week"], errors="coerce")
    return games.dropna(subset=["week"])
