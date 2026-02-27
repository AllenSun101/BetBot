import requests
import nflreadpy as nfl
from weather import get_weather_statistics

team_id_map = {
    "DET": "8",
    "SEA": "26",
}

def get_snap_count_stats(player_name: str, season: int):
    snap_counts = nfl.load_snap_counts([season]).to_pandas()
    return snap_counts[snap_counts["player"] == player_name]

def get_player_stats(player_name: str, season: str):
    player_stats = nfl.load_player_stats([season]).to_pandas()
    return player_stats[player_stats["player_display_name"] == player_name]

def get_team_stats(team_name: str, season: int):
    team_stats = nfl.load_team_stats([season]).to_pandas()
    return team_stats[team_stats["team"] == team_name]

def get_defense_stats(team_name: str, season: int):
    team_stats = nfl.load_team_stats([season]).to_pandas()
    team_schedule = team_stats[team_stats["team"] == team_name]

    defense_stats = {}
    for idx, row in team_schedule.iterrows():
        week = row["week"]
        opponent_name = row["opponent_team"]
        opponent_schedule = team_stats[team_stats["team"] == opponent_name]
        opponent_df = opponent_schedule[opponent_schedule["week"] < week]

        # for each game, get yards allowed???
        for oidx, orow in opponent_df.iterrows():
            x = "completions"
            x = "attempts"
            x = "passing_yards"
            x = "passing_tds"
            x = "passing_air_yards"
            x = "passing_yards_after_catch"
            x = "carries"
            x = "rushing_yards"
            x = "rushing_tds"
            x = 0 # find the offensive game

        defense_statistics = {
            "def_tackles_solo": opponent_df["def_tackles_solo"].mean(),
            "def_tackles_with_assist": opponent_df["def_tackles_with_assist"].mean(),
            "def_tackles_for_loss": opponent_df["def_tackles_for_loss"].mean(),
            "def_tackles_for_loss_yards": opponent_df["def_tackles_for_loss_yards"].mean(),
            "def_fumbles_forced": opponent_df["def_fumbles_forced"].mean(),
            "def_sacks": opponent_df["def_sacks"].mean(),
            "def_sack_yards": opponent_df["def_sack_yards"].mean(),
            "def_qb_hits": opponent_df["def_qb_hits"].mean(),
            "def_interceptions": opponent_df["def_interceptions"].mean(),
            "def_interception_yards": opponent_df["def_interception_yards"].mean(),
            "def_pass_defended": opponent_df["def_pass_defended"].mean(),
            "def_tds": opponent_df["def_tds"].mean(),
            "def_safeties": opponent_df["def_safeties"].mean(),
        }
        
        defense_stats[week] = defense_statistics
    return defense_stats

def get_competition_stats(team_name: str, season: int):
    team_id = team_id_map[team_name]

    url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/teams/{team_id}/events?lang=en&region=us"
    response = requests.get(url)
    schedule = response.json()

    competition_statistics = {}
    for event in schedule["items"]:
        event_url = event["$ref"]
        event_data = requests.get(event_url).json()

        date = event_data["date"]
        stadium = event_data["competitions"][0]["venue"]["fullName"]
        city = event_data["competitions"][0]["venue"]["address"]["city"]
        grass = event_data["competitions"][0]["venue"]["grass"]
        indoor = event_data["competitions"][0]["venue"]["indoor"]

        for competitor in event_data["competitions"][0]["competitors"]:
            if competitor["id"] == team_id:
                if competitor["homeAway"] == "home":
                    home = True
                else:
                    home = False

        week_url_parts = event_data["week"]["$ref"].split("/")
        season_idx = week_url_parts.index("types") if "types" in week_url_parts else -1
        week_idx = week_url_parts.index("weeks") if "weeks" in week_url_parts else -1
        season_type = week_url_parts[season_idx + 1] if season_idx != -1 and season_idx + 1 < len(week_url_parts) else None
        week = int(week_url_parts[week_idx + 1].split("?")[0]) if week_idx != -1 and week_idx + 1 < len(week_url_parts) else None

        if season_type == "3":
            week += 18
        if week == 23:
            week = 22

        competition_statistics[week] = {
            "date": date, 
            "stadium": stadium,
            "city": city,
            "home": home,
            "grass": grass,
            "indoor": indoor,
        }

    return competition_statistics

def get_weather_stats(schedule: dict[str, dict]):
    weather_statistics = {}
    for week, game in schedule.items():
        stadium = game["stadium"]
        city = game["city"]
        date = game["date"]
        weather_statistics[week] = get_weather_statistics(f"{stadium} {city}", date)
    
    return weather_statistics

if __name__ == "__main__":
    print(get_player_stats("Cooper Kupp", 2025))