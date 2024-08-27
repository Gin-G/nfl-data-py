import nfl_data_py as nfl
import pandas as pd
import xarray as xr

def get_all_player_ids(year):
    player_list = []
    weekly_data = get_weekly_data(year)
    players = weekly_data['player_id'].tolist()
    for player in players:
        if player not in player_list:
            player_list.append(player)
    return player_list

def get_all_player_names(year):
    player_list = []
    weekly_data = get_weekly_data(year)
    players = weekly_data['player_display_name'].tolist()
    for player in players:
        if player not in player_list:
            player_list.append(player)
    return player_list

def get_game_pbp_data(game_id):
    year = game_id.split('_')[0]
    pbp_data = get_pbp_data(int(year))
    game_pbp_data = pbp_data.loc[pbp_data['game_id'] == game_id]
    return game_pbp_data

def get_pbp_data(year):
    if isinstance(year, list):
        pbp_data = nfl.import_pbp_data(year)
    else:
        pbp_data = nfl.import_pbp_data([year])
    return pbp_data

def get_players():
    player = nfl.import_players()
    return player

def get_playerid_weekly_data(player_id, year):
    weekly_data = get_weekly_data(year)
    playerid_weekly_data = weekly_data.loc[weekly_data['player_id'] == player_id]
    return playerid_weekly_data

def get_player_weekly_data(player_name, year):
    weekly_data = get_weekly_data(year)
    player_weekly_data = weekly_data.loc[weekly_data['player_display_name'] == player_name]
    return player_weekly_data

def get_schedule(year):
    # Variable to link to other data is game_id
    schedule = nfl.import_schedules([year])
    return schedule

def get_seasonal_data(year):
    seasonal_data = nfl.import_seasonal_data([year])
    return seasonal_data

def get_team_list(year):
    team_list = []
    roster_data = get_roster_data(year)
    teams = roster_data['team'].tolist()
    for team in teams:
        if team not in team_list:
            team_list.append(team)
    return team_list

def get_team_schedule(team, year):
    schedule = get_schedule(year)
    home_team_schedule = schedule.loc[schedule['home_team'] == team]
    away_team_schedule = schedule.loc[schedule['away_team'] == team]
    full_schedule = [home_team_schedule, away_team_schedule]
    full_schedule = pd.concat(full_schedule)
    return full_schedule

def get_weekly_data(year):
    # Group player data by player_id
    if isinstance(year, list):
        weekly_data = nfl.import_weekly_data(year)
    else:
        weekly_data = nfl.import_weekly_data([year])
    return weekly_data

def get_roster_data(year, columns=None):
    roster_data = nfl.import_rosters([year], columns)
    return roster_data

def get_team_roster_data(team, year):
    roster_data = get_roster_data(year)
    team_roster_data = roster_data.loc[roster_data['team'] == team]
    return team_roster_data

def get_ftn_data(year):
    ftn_data = nfl.import_ftn_data([year])
    return ftn_data

def get_snap_counts(year):
    snap_counts = nfl.import_snap_counts([year])
    return snap_counts
