import requests

season = 2025
week = 1

def get_schedule(season):

    url = f"https://api.sportradar.com/nfl/official/trial/v7/en/games/{season}/REG/schedule.json"

    headers = {
        "accept": "application/json",
        "x-api-key": "MwV55LuOUHtG5pzxvGGjEay1RJsyNvOgpHKyqNVh"
    }

    response = requests.get(url, headers=headers)

    return response.json()

def get_play_by_play(game_id):
    url = f"https://api.sportradar.com/nfl/official/trial/v7/en/games/{game_id}/pbp.json"

    headers = {
        "accept": "application/json",
        "x-api-key": "MwV55LuOUHtG5pzxvGGjEay1RJsyNvOgpHKyqNVh"
    }

    response = requests.get(url, headers=headers)

    print(response.text)
    return response.json()

season2025 = get_schedule(season)
weeks2025 = season2025['weeks']
for weeks in weeks2025:
    if weeks['title'] == str(week):
        games = weeks['games']
        for game in games:
            print(game['id'])
            print(game['away']['name'] + " @ " + game['home']['name'])
            print(game['scheduled'])
            print(game['status'])
            print("")