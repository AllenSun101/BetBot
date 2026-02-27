import requests
from weather import get_weather_statistics
import nflreadpy as nfl

# injury data
# get team schedule -> week # to event_id mapping
# we only need per competition data

"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2025/teams/1/depthcharts?lang=en&region=us"
"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/401773016?lang=en&region=us"


def get_player_id(team_id: str, player_name: str) -> str:
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}?enable=roster"
    response = requests.get(url)
    roster = response.json()

    for player in roster["team"]["athletes"]:
        if player["fullName"] == player_name:
            return player["id"]
    
    return ""

def get_player_events(season: str, player_id: str):
    url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/athletes/{player_id}/eventlog"
    response = requests.get(url)
    event_log = response.json()
    events = []
    for event in event_log["events"]["items"]:
        try:
            competition = event["competition"]["$ref"]
            statistics = event["statistics"]["$ref"]
            played = event["played"]
            events.append((competition, statistics, played))
        except Exception as e:
            print(e)
    
    return events

# WR -> play share, run/pass offense types, QB/RB share of runs (percentages of action) -> how to get
# play by play analysis of decision-making, conditionals -> pass more/run more?

# splits -> questions of statistical significance - is there a performance difference?
# splits used for visualization

if __name__ == "__main__":
    player_id = get_player_id("8", "Jared Goff")
    events = get_player_events(2025, player_id)
