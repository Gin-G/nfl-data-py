"""Build DFS player-pool CSVs from weekly prediction files.

Combines predictions with player identity info (sportradar_id, headshot,
team) from the historical dataset and assigns a synthetic price
(projected points * 500, rounded to the nearest $100).
"""

import os

import pandas as pd

from . import config
from .utils import normalize_player_name


def _price_for(projected_points):
    if pd.isna(projected_points):
        projected_points = 0
    return round(int(projected_points * 500) / 100) * 100


def build_player_lookup(df_main):
    """Map normalized player names -> identity info from the dataset."""
    player_info = df_main.groupby("player_id").last().reset_index()

    position_col = "position_x" if "position_x" in player_info.columns else "position"
    team_col = "recent_team" if "recent_team" in player_info.columns else "team"

    lookup = {}
    for _, player in player_info.iterrows():
        names_to_try = []
        for field in ["player_display_name", "player_name"]:
            if field in player.index and pd.notna(player.get(field)):
                original = str(player[field]).strip()
                names_to_try.append(original.lower())
                normalized = normalize_player_name(original)
                if normalized and normalized != original.lower():
                    names_to_try.append(normalized)

        def clean(val):
            return "" if pd.isna(val) else str(val)

        info = {
            "sportradar_id": clean(player.get("sportradar_id", "")),
            "player_name": clean(
                player.get("player_display_name", player.get("player_name", ""))
            ),
            "position": clean(player.get(position_col, "")),
            "headshot_url": clean(player.get("headshot_url", "")),
            "recent_team": clean(player.get(team_col, "")),
        }
        for name in names_to_try:
            if name:
                lookup[name] = info
    return lookup


def create_player_pool(season, week,
                       main_dataset_path=config.DATASET_PATH,
                       predictions_directory=config.PREDICTIONS_DIR,
                       output_directory=config.POOLS_DIR,
                       positions=None):
    """Create per-position player pool CSVs. Returns (files_created, total)."""
    positions = positions or config.POSITIONS

    if not os.path.exists(main_dataset_path):
        raise FileNotFoundError(f"Main dataset not found at {main_dataset_path}")

    print(f"Creating player pool for {season} week {week}")
    df_main = pd.read_csv(main_dataset_path, low_memory=False)
    player_lookup = build_player_lookup(df_main)
    print(f"Created lookup for {len(player_lookup)} player name variations")

    os.makedirs(output_directory, exist_ok=True)
    files_created = []
    total_players = 0

    for position in positions:
        prediction_path = os.path.join(
            predictions_directory,
            f"{position}_predictions_week{week}_season{season}.csv",
        )
        if not os.path.exists(prediction_path):
            print(f"Warning: {prediction_path} not found, skipping {position}")
            continue

        df_pred = pd.read_csv(prediction_path)
        if df_pred.empty:
            continue

        clean_players = []
        unmatched = []
        for _, pred in df_pred.iterrows():
            name_original = str(pred["player_name"]).strip()
            info = (
                player_lookup.get(name_original.lower())
                or player_lookup.get(normalize_player_name(name_original))
            )

            projected = pred.get("fanduel_fantasy_points", pred.get("projected_points", 0))
            if pd.isna(projected):
                projected = 0

            if info:
                clean_players.append({
                    "sportradar_id": info["sportradar_id"],
                    "player_name": info["player_name"],
                    "position": info["position"],
                    "headshot_url": info["headshot_url"],
                    "price": _price_for(projected),
                    "projected_points": round(projected, 1),
                    "team": info["recent_team"],
                })
            else:
                unmatched.append(name_original)
                clean_players.append({
                    "sportradar_id": str(pred.get("sportradar_id", "") or ""),
                    "player_name": name_original,
                    "position": pred.get("position", position),
                    "headshot_url": pred.get("headshot_url", ""),
                    "price": _price_for(projected),
                    "projected_points": round(projected, 1),
                    "team": str(pred.get("team", "") or ""),
                })

        if not clean_players:
            continue

        df_clean = pd.DataFrame(clean_players).sort_values("price", ascending=False)
        output_file = f"{position}_player_pool_{season}_week{week}.csv"
        df_clean.to_csv(os.path.join(output_directory, output_file), index=False)

        files_created.append(output_file)
        total_players += len(df_clean)
        print(f"  {position}: saved {len(df_clean)} players "
              f"({len(df_clean) - len(unmatched)} matched to dataset)")
        if unmatched:
            print(f"    Unmatched: {', '.join(unmatched[:5])}"
                  + (f" ... +{len(unmatched) - 5} more" if len(unmatched) > 5 else ""))

    return files_created, total_players


def create_combined_player_pool(season, week, output_directory=config.POOLS_DIR,
                                positions=None):
    """Concatenate the per-position pools into one all-players CSV."""
    positions = positions or config.POSITIONS

    frames = []
    for position in positions:
        path = os.path.join(
            output_directory, f"{position}_player_pool_{season}_week{week}.csv"
        )
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["position", "price"], ascending=[True, False])

    combined_file = f"all_players_pool_{season}_week{week}.csv"
    df.to_csv(os.path.join(output_directory, combined_file), index=False)
    print(f"Combined pool: {len(df)} players -> {combined_file}")
    return combined_file
