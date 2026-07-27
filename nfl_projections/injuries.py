"""Injury integration.

Turns weekly injury information into two dicts the projector consumes:

- injury_overrides: players to zero out (OUT/DOUBTFUL)
- backup_situations: healthy backups elevated because a starter is out

The default source is nflverse's weekly injury report via
``nflreadpy.load_injuries`` — free, no API key, no rate limits. IR/PUP/reserve
players come straight from roster ``status``. The Sportradar API path is kept
as an optional fallback for anyone who has a key (SPORTRADAR_API_KEY), but is
no longer required.
"""

import json
import logging
import os
import time

import requests

from .utils import to_pandas

logger = logging.getLogger(__name__)

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]
INACTIVE_STATUSES = ["RES", "IR", "PUP", "SUS", "NON"]

# nflverse report_status -> our normalized game status
_REPORT_STATUS_MAP = {
    "OUT": "OUT",
    "DOUBTFUL": "DOUBTFUL",
    "QUESTIONABLE": "QUESTIONABLE",
    "QUES": "QUESTIONABLE",
}


# -- source-agnostic helpers ------------------------------------------------

def _ensure_player_name(df):
    if "player_name" not in df.columns and "full_name" in df.columns:
        df = df.copy()
        df["player_name"] = df["full_name"]
    return df


def fetch_ir_players(roster_data):
    """IR/PUP/reserve skill players from roster ``status``, marked OUT."""
    ir_pup_players = {}
    roster_data = _ensure_player_name(roster_data)

    if "status" not in roster_data.columns:
        logger.warning("No 'status' column in roster data")
        return ir_pup_players

    inactive = roster_data[roster_data["status"].isin(INACTIVE_STATUSES)]
    inactive = inactive.drop_duplicates(
        subset=["player_name", "team", "position"], keep="first"
    )
    skill_inactive = inactive[inactive["position"].isin(SKILL_POSITIONS)]

    for _, player in skill_inactive.iterrows():
        player_name = player.get("player_name", player.get("full_name", ""))
        if not player_name:
            continue
        ir_pup_players[player_name] = {
            "status": "OUT",
            "injury_type": f"{player['status']} list",
            "position": player["position"],
            "team": player.get("team", "UNK"),
            "player_id": player.get("gsis_id", player.get("player_id", "")),
            "practice_status": "IR/PUP/Reserve",
        }

    print(f"Found {len(ir_pup_players)} skill players on IR/PUP/Reserve")
    return ir_pup_players


def merge_injury_sources(primary, secondary):
    """Union of two {player_name: info} dicts; ``primary`` wins on conflict."""
    merged = primary.copy()
    for player_name, info in secondary.items():
        merged.setdefault(player_name, info)
    return merged


def _find_healthy_backup(team, position, injured_player, depth_charts, out_players):
    """Highest-ranked healthy backup behind an injured player, or None."""
    team_depth = depth_charts[
        (depth_charts["team"] == team) & (depth_charts["pos_abb"] == position)
    ].sort_values("pos_rank")
    if team_depth.empty:
        return None

    injured_rank = team_depth[
        team_depth["player_name"].str.contains(injured_player, case=False, na=False)
    ]
    if injured_rank.empty:
        parts = injured_player.lower().split()
        if len(parts) >= 2:
            injured_rank = team_depth[
                team_depth["player_name"].str.lower().str.contains(parts[-1], na=False)
            ]
    if injured_rank.empty:
        return None

    injured_pos_rank = injured_rank.iloc[0]["pos_rank"]
    injured_name = injured_rank.iloc[0]["player_name"]
    potential_backups = team_depth[
        (team_depth["pos_rank"] > injured_pos_rank)
        & (team_depth["player_name"] != injured_name)
    ].sort_values("pos_rank")

    checked = set()
    for _, backup_row in potential_backups.iterrows():
        backup_name = backup_row["player_name"]
        try:
            backup_lower = backup_name.lower()
        except AttributeError:
            continue
        if backup_lower in checked or backup_lower in out_players:
            checked.add(backup_lower)
            continue
        checked.add(backup_lower)
        return backup_name
    return None


def categorize_injury_impact(injuries_dict, roster_data, depth_charts):
    """Split injuries into zero_out / boost_backup / questionable buckets."""
    impact = {"zero_out": [], "boost_backup": [], "questionable": []}
    roster_data = _ensure_player_name(roster_data)

    out_players = set()
    for player_name, injury_info in injuries_dict.items():
        status = injury_info["status"]
        entry = {
            "player": player_name,
            "injury": injury_info["injury_type"],
            "position": injury_info["position"],
            "team": injury_info["team"],
        }
        if status in ["OUT", "DOUBTFUL", "D"]:
            out_players.add(player_name.lower())
            impact["zero_out"].append({**entry, "status": status})
        elif status in ["QUESTIONABLE", "Q"]:
            impact["questionable"].append(entry)

    for injured in impact["zero_out"]:
        backup = _find_healthy_backup(
            injured["team"], injured["position"], injured["player"],
            depth_charts, out_players,
        )
        if backup:
            impact["boost_backup"].append({
                "player": backup,
                "replacing": injured["player"],
                "reason": f"{injured['player']} {injured['status'].lower()} - {injured['injury']}",
                "position": injured["position"],
                "team": injured["team"],
            })

    return impact


def build_overrides(injuries_dict, roster_data, depth_charts):
    """Turn a flat injuries dict into (overrides, backup_situations)."""
    ir_injuries = fetch_ir_players(roster_data)
    all_injuries = merge_injury_sources(injuries_dict, ir_injuries)

    if not all_injuries:
        print("No injuries found from any source")
        return {}, {}

    impact = categorize_injury_impact(all_injuries, roster_data, depth_charts)

    overrides = {}
    backup_situations = {}
    for player_info in impact["zero_out"]:
        overrides[player_info["player"]] = {
            "status": player_info["status"],
            "reason": player_info["injury"],
            "team": player_info["team"],
            "position": player_info["position"],
        }
    for backup_info in impact["boost_backup"]:
        backup_situations[backup_info["player"]] = {
            "replacing": backup_info["replacing"],
            "reason": backup_info["reason"],
            "team": backup_info["team"],
            "position": backup_info["position"],
            "role": "emergency_starter",
        }
        if backup_info["replacing"] in overrides:
            overrides[backup_info["replacing"]]["replacement"] = backup_info["player"]

    print(f"Injury impact: {len(impact['zero_out'])} OUT/doubtful, "
          f"{len(impact['boost_backup'])} backups elevated, "
          f"{len(impact['questionable'])} questionable")
    return overrides, backup_situations


# -- nflverse source (default) ----------------------------------------------

def _first_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def parse_nflverse_injuries(injuries_df, week):
    """Flatten an nflverse weekly injury-report frame into {name: info}.

    Uses report_status for the game designation, falling back to inferring
    QUESTIONABLE from limited/DNP practice participation when the report
    status is blank (early-week reports).
    """
    df = to_pandas(injuries_df)
    if "week" in df.columns:
        df = df[df["week"] == week]
    if df.empty:
        return {}

    name_col = _first_col(df, ["full_name", "player_name", "player_display_name"])
    status_col = _first_col(df, ["report_status", "game_status", "status"])
    practice_col = _first_col(df, ["practice_status", "practice_primary_injury"])
    injury_col = _first_col(df, ["report_primary_injury", "injury", "primary_injury"])
    team_col = _first_col(df, ["team", "club_code", "team_abbr"])
    pos_col = _first_col(df, ["position", "pos"])

    injuries = {}
    for _, row in df.iterrows():
        player_name = str(row.get(name_col, "") or "").strip()
        if not player_name:
            continue

        raw_status = str(row.get(status_col, "") or "").strip().upper()
        status = _REPORT_STATUS_MAP.get(raw_status, "")

        practice_status = str(row.get(practice_col, "") or "").strip() if practice_col else ""
        if not status and practice_status:
            low = practice_status.lower()
            if "did not participate" in low or "limited" in low or "dnp" in low:
                status = "QUESTIONABLE"

        if not status:
            continue

        injury_type = str(row.get(injury_col, "") or "").strip() or "Not specified"
        injuries[player_name] = {
            "status": status,
            "injury_type": injury_type,
            "position": str(row.get(pos_col, "") or "UNK"),
            "team": str(row.get(team_col, "") or "UNK"),
            "practice_status": practice_status,
        }

    print(f"Processed {len(injuries)} injured players from nflverse injury reports")
    return injuries


def fetch_nflverse_injuries(season, week):
    """Load and flatten the nflverse weekly injury report (no API key needed)."""
    import nflreadpy as nfl

    try:
        injuries_df = to_pandas(nfl.load_injuries(seasons=[season]))
    except Exception as e:  # network / availability issues
        logger.warning("Could not load nflverse injuries: %s", e)
        return {}
    return parse_nflverse_injuries(injuries_df, week)


# -- Sportradar source (optional fallback) ----------------------------------

class SportradarInjuryAnalyzer:
    """Fetch injury data from the Sportradar API (requires a paid/trial key)."""

    def __init__(self, api_key, season=2025):
        self.api_key = api_key
        self.season = season
        self.base_url = "https://api.sportradar.com/nfl/official/trial/v7/en"
        self.headers = {"accept": "application/json"}

    def fetch_weekly_injuries(self, week):
        url_patterns = [
            f"{self.base_url}/seasons/{self.season}/REG/{week:02d}/injuries.json?api_key={self.api_key}",
            f"{self.base_url}/seasons/{self.season}/REG/injuries.json?api_key={self.api_key}",
            f"{self.base_url}/league/{self.season}/REG/{week}/injuries.json?api_key={self.api_key}",
        ]
        print(f"Fetching Sportradar injury data for week {week}...")
        for url in url_patterns:
            time.sleep(1)  # trial API rate limit
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    return response.json()
                if response.status_code == 404:
                    continue
                if response.status_code in (401, 403):
                    logger.error("Sportradar returned %s - check your API key",
                                 response.status_code)
                    return None
                logger.warning("Unexpected status %s from Sportradar", response.status_code)
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                logger.warning("Sportradar request failed: %s", e)
        logger.warning("All Sportradar injury URL patterns failed for week %s", week)
        return None

    def process_injury_data(self, injury_data):
        """Flatten Sportradar injury JSON into {player_name: info}."""
        if not injury_data:
            return {}
        injuries = {}
        teams = injury_data.get("teams", [])
        if not teams and "week" in injury_data:
            teams = injury_data["week"].get("teams", [])

        for team in teams:
            team_alias = team.get("alias", team.get("market", "UNK"))
            for player_entry in team.get("players", []):
                player_name = (
                    player_entry.get("name")
                    or player_entry.get("full_name")
                    or player_entry.get("preferred_name", "")
                )
                position = player_entry.get("position", "")
                for injury_info in player_entry.get("injuries", []):
                    status = injury_info.get("status") or injury_info.get("game_status", "")
                    practice_info = injury_info.get("practice", {})
                    practice_status = (
                        practice_info.get("status", "")
                        if isinstance(practice_info, dict) else ""
                    )
                    if not status and practice_status:
                        if "Did Not Participate" in practice_status or "Limited" in practice_status:
                            status = "QUESTIONABLE"
                    primary_injury = (
                        injury_info.get("primary")
                        or injury_info.get("description")
                        or injury_info.get("injury")
                        or injury_info.get("comment", "")
                    )
                    if player_name and status:
                        injuries[player_name] = {
                            "status": status.upper(),
                            "injury_type": primary_injury or "Not specified",
                            "position": position or "UNK",
                            "team": team_alias,
                            "practice_status": practice_status,
                        }
                    break  # only the first injury entry per player
        print(f"Processed {len(injuries)} injured players from Sportradar API")
        return injuries


def fetch_sportradar_injuries(season, week):
    """Load and flatten Sportradar injuries. Empty dict if no key is set."""
    api_key = os.getenv("SPORTRADAR_API_KEY")
    if not api_key:
        return {}
    analyzer = SportradarInjuryAnalyzer(api_key, season=season)
    return analyzer.process_injury_data(analyzer.fetch_weekly_injuries(week))


# -- entry point ------------------------------------------------------------

def integrate_injuries(week, roster_data, depth_charts, season=2025, source="nflverse"):
    """Fetch injuries and return (injury_overrides, backup_situations).

    Args:
        source: "nflverse" (default, free) or "sportradar" (needs a key).
            "auto" tries nflverse and falls back to Sportradar if it finds
            nothing and a key is present.
    """
    if source in ("sportradar",):
        injuries = fetch_sportradar_injuries(season, week)
    else:
        injuries = fetch_nflverse_injuries(season, week)
        if not injuries and source == "auto":
            injuries = fetch_sportradar_injuries(season, week)

    # IR/PUP from rosters is always folded in by build_overrides
    return build_overrides(injuries, roster_data, depth_charts)


# Backward-compatible alias (older callers / --source sportradar)
def integrate_sportradar_injuries(week, roster_data, depth_charts, season=2025):
    return integrate_injuries(week, roster_data, depth_charts, season=season, source="auto")
