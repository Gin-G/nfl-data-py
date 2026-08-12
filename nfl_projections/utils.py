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
    """REGULAR-SEASON games only: drop season-average rows (week == 'AVG') and
    postseason rows, and coerce week to numeric.

    The postseason filter was missing for a long time, so anything built on this
    — the training set, the backtest population, board ground truth — quietly
    included weeks 19-22 (4.6% of 2025 rows). That is not leakage, but it is a
    biased sample: only playoff teams appear in it, so it over-weights good
    offenses and inflates any season total for a team that went deep. The
    benchmark labelled "2025 weeks 1-18" was really weeks 1-22.
    """
    games = df[df["week"] != "AVG"].copy()
    if "season_type" in games.columns:
        games = games[(games["season_type"] == "REG") | games["season_type"].isna()]
    games["week"] = pd.to_numeric(games["week"], errors="coerce")
    return games.dropna(subset=["week"])
