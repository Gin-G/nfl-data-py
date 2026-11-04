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

import pandas as pd
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
            "player_id": injury_info.get("player_id", ""),
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
    return build_overrides_detailed(injuries_dict, roster_data, depth_charts)[:2]


def build_overrides_detailed(injuries_dict, roster_data, depth_charts):
    """``(overrides, backup_situations, counts)``.

    ``counts`` separates what each source contributed, which is the thing a
    projection cannot be read back from afterwards: a board with 24 players
    zeroed looks the same whether the weekly report named them or they were all
    simply sitting on IR. See ``integrate_injuries_detailed``.
    """
    ir_injuries = fetch_ir_players(roster_data)
    all_injuries = merge_injury_sources(injuries_dict, ir_injuries)
    counts = {"report_players": len(injuries_dict), "ir_players": len(ir_injuries),
              "out": 0, "backups_elevated": 0, "questionable": 0}

    if not all_injuries:
        print("No injuries found from any source")
        return {}, {}, counts

    impact = categorize_injury_impact(all_injuries, roster_data, depth_charts)
    counts.update(out=len(impact["zero_out"]),
                  backups_elevated=len(impact["boost_backup"]),
                  questionable=len(impact["questionable"]))

    overrides = {}
    backup_situations = {}
    for player_info in impact["zero_out"]:
        overrides[player_info["player"]] = {
            "status": player_info["status"],
            "reason": player_info["injury"],
            "team": player_info["team"],
            "position": player_info["position"],
            "player_id": player_info.get("player_id", ""),
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
    return overrides, backup_situations, counts


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
    id_col = _first_col(df, ["gsis_id", "player_id"])

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
            "player_id": str(row.get(id_col) or "") if id_col else "",
        }

    print(f"Processed {len(injuries)} injured players from nflverse injury reports")
    return injuries


def injury_report_coverage(injuries_df, week):
    """What the weekly report HOLDS for a week, before anything is parsed out.

    Deliberately separate from parsing. ``parse_nflverse_injuries`` filters to
    the exact week and returns ``{}`` both for a week where every club filed a
    clean report and for a week nobody has filed yet — and teams do not file
    until Wednesday, so a Tuesday run reads an empty frame and projects as if
    the league were healthy. Those two cases have to be told apart after the
    fact, so what the feed covered is measured here and carried out with the
    projection rather than inferred from the absence of overrides.
    """
    df = to_pandas(injuries_df)
    empty = {"report_weeks": [], "report_latest_week": None,
             "report_rows_for_week": 0, "report_available": False}
    if df is None or len(df) == 0 or "week" not in df.columns:
        return empty
    weeks = pd.to_numeric(df["week"], errors="coerce").dropna().astype(int)
    if weeks.empty:
        return empty
    rows = int((weeks == week).sum())
    return {
        "report_weeks": sorted({int(w) for w in weeks}),
        "report_latest_week": int(weeks.max()),
        "report_rows_for_week": rows,
        "report_available": rows > 0,
    }


def _nflverse_injuries_with_coverage(season, week):
    """``(injuries, coverage)`` from the nflverse weekly report."""
    import nflreadpy as nfl

    try:
        injuries_df = to_pandas(nfl.load_injuries(seasons=[season]))
    except Exception as e:  # network / availability issues
        logger.warning("Could not load nflverse injuries: %s", e)
        return {}, dict(injury_report_coverage(None, week), report_error=str(e))
    return (parse_nflverse_injuries(injuries_df, week),
            injury_report_coverage(injuries_df, week))


def fetch_nflverse_injuries(season, week):
    """Load and flatten the nflverse weekly injury report (no API key needed)."""
    return _nflverse_injuries_with_coverage(season, week)[0]


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
    return integrate_injuries_detailed(week, roster_data, depth_charts,
                                       season=season, source=source)[:2]


def integrate_injuries_detailed(week, roster_data, depth_charts, season=2025,
                                source="nflverse"):
    """``(injury_overrides, backup_situations, provenance)``.

    ``provenance`` is what the sources offered for *this* week — which weeks the
    report covers, how many rows it filed for the week being projected, and how
    many players each source accounted for. It is the record that makes a board
    readable back: "nobody was marked out" and "nobody had yet been asked" are
    the same empty dict without it.
    """
    coverage = {}
    if source in ("sportradar",):
        injuries = fetch_sportradar_injuries(season, week)
        used = "sportradar"
    else:
        injuries, coverage = _nflverse_injuries_with_coverage(season, week)
        used = "nflverse"
        if not injuries and source == "auto":
            injuries = fetch_sportradar_injuries(season, week)
            if injuries:
                used = "sportradar"

    # IR/PUP from rosters is always folded in by build_overrides
    overrides, backups, counts = build_overrides_detailed(
        injuries, roster_data, depth_charts)
    provenance = {"source": used, "season": season, "week": week,
                  **injury_report_coverage(None, week), **coverage, **counts}
    return overrides, backups, provenance


# Backward-compatible alias (older callers / --source sportradar)
def integrate_sportradar_injuries(week, roster_data, depth_charts, season=2025):
    return integrate_injuries(week, roster_data, depth_charts, season=season, source="auto")
