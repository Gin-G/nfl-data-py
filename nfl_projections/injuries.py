"""Sportradar injury integration.

Fetches the weekly injury report from the Sportradar API (requires the
SPORTRADAR_API_KEY environment variable), merges in IR/PUP players from
roster data, and produces:

- injury_overrides: players who should be zeroed out (OUT/DOUBTFUL)
- backup_situations: healthy backups elevated because a starter is out
"""

import json
import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]
INACTIVE_STATUSES = ["RES", "IR", "PUP", "SUS", "NON"]


class SportradarInjuryAnalyzer:
    """Fetch and analyze injury data from the Sportradar API."""

    def __init__(self, api_key, season=2025):
        self.api_key = api_key
        self.season = season
        self.base_url = "https://api.sportradar.com/nfl/official/trial/v7/en"
        self.headers = {"accept": "application/json"}

    def fetch_weekly_injuries(self, week):
        """Fetch the injury report for a specific week, trying known URL formats."""
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
                    logger.error(
                        "Sportradar API returned %s - check your API key/permissions",
                        response.status_code,
                    )
                    return None
                logger.warning("Unexpected status %s from Sportradar", response.status_code)
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                logger.warning("Sportradar request failed: %s", e)

        logger.warning("All Sportradar injury URL patterns failed for week %s", week)
        return None

    def fetch_season_injuries(self):
        """Fallback: season-wide injury endpoint."""
        url = f"{self.base_url}/injuries.json?api_key={self.api_key}"
        time.sleep(1)
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            logger.warning("Season injury endpoint returned %s", response.status_code)
        except requests.exceptions.RequestException as e:
            logger.warning("Season injury request failed: %s", e)
        return None

    def fetch_ir_players(self, roster_data):
        """Extract IR/PUP/reserve skill players from roster data as OUT."""
        ir_pup_players = {}

        if "player_name" not in roster_data.columns and "full_name" in roster_data.columns:
            roster_data = roster_data.copy()
            roster_data["player_name"] = roster_data["full_name"]

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
                "status": "OUT",  # IR/PUP = definitely out
                "injury_type": f"{player['status']} list",
                "position": player["position"],
                "team": player.get("team", "UNK"),
                "player_id": player.get("gsis_id", player.get("player_id", "")),
                "sportradar_id": player.get("sportradar_id", ""),
                "practice_status": "IR/PUP/Reserve",
            }

        print(f"Found {len(ir_pup_players)} skill players on IR/PUP/Reserve")
        return ir_pup_players

    def process_injury_data(self, injury_data):
        """Flatten the injury JSON into {player_name: injury_info}."""
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
                    # No game status: infer conservatively from practice participation
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
                        if player_name in injuries:
                            injuries[player_name]["injury_type"] += f", {primary_injury}"
                        else:
                            injuries[player_name] = {
                                "status": status.upper(),
                                "injury_type": primary_injury or "Not specified",
                                "position": position or "UNK",
                                "team": team_alias,
                                "player_id": player_entry.get("id", ""),
                                "sportradar_id": player_entry.get("sr_id", ""),
                                "practice_status": practice_status,
                            }
                    break  # only the first injury entry per player

        print(f"Processed {len(injuries)} injured players from Sportradar API")
        return injuries

    def merge_injury_sources(self, api_injuries, ir_injuries):
        """Union of API injuries (priority) and roster IR/PUP players."""
        merged = api_injuries.copy()
        for player_name, injury_info in ir_injuries.items():
            if player_name not in merged:
                merged[player_name] = injury_info
        return merged

    def categorize_injury_impact(self, injuries_dict, roster_data, depth_charts):
        """Split injuries into zero_out / boost_backup / questionable buckets."""
        impact = {"zero_out": [], "boost_backup": [], "questionable": []}

        if "player_name" not in roster_data.columns and "full_name" in roster_data.columns:
            roster_data = roster_data.copy()
            roster_data["player_name"] = roster_data["full_name"]

        out_doubtful_players = set()
        for player_name, injury_info in injuries_dict.items():
            status = injury_info["status"]
            entry = {
                "player": player_name,
                "injury": injury_info["injury_type"],
                "position": injury_info["position"],
                "team": injury_info["team"],
            }
            if status in ["OUT", "DOUBTFUL", "D"]:
                out_doubtful_players.add(player_name.lower())
                impact["zero_out"].append({**entry, "status": status})
            elif status in ["QUESTIONABLE", "Q"]:
                impact["questionable"].append(entry)

        for injured in impact["zero_out"]:
            backup = self._find_healthy_backup(
                injured["team"], injured["position"], injured["player"],
                depth_charts, out_doubtful_players,
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

    def _find_healthy_backup(self, team, position, injured_player, depth_charts,
                             out_doubtful_players):
        """Find the highest-ranked healthy backup behind an injured player."""
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
            if backup_lower in checked or backup_lower in out_doubtful_players:
                checked.add(backup_lower)
                continue
            checked.add(backup_lower)
            return backup_name
        return None

    def create_injury_overrides(self, week, roster_data, depth_charts):
        """Full flow: fetch, merge, categorize. Returns (overrides, backups)."""
        injury_json = self.fetch_weekly_injuries(week)
        if not injury_json:
            injury_json = self.fetch_season_injuries()

        ir_injuries = self.fetch_ir_players(roster_data)
        api_injuries = self.process_injury_data(injury_json) if injury_json else {}
        all_injuries = self.merge_injury_sources(api_injuries, ir_injuries)

        if not all_injuries:
            print("No injuries found from any source")
            return {}, {}

        impact = self.categorize_injury_impact(all_injuries, roster_data, depth_charts)

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


def integrate_sportradar_injuries(week, roster_data, depth_charts, season=2025):
    """Main entry point. Returns (injury_overrides, backup_situations).

    Falls back to empty dicts (no injury adjustments) when SPORTRADAR_API_KEY
    is not set.
    """
    api_key = os.getenv("SPORTRADAR_API_KEY")
    if not api_key:
        print("WARNING: SPORTRADAR_API_KEY not set, skipping injury adjustments")
        return {}, {}

    analyzer = SportradarInjuryAnalyzer(api_key, season=season)
    return analyzer.create_injury_overrides(week, roster_data, depth_charts)
