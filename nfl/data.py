import nflreadpy as nfl

def get_team_stats(seasons: list[str]):
    team_stats = nfl.load_team_stats(seasons).to_pandas()
    team_stats.to_csv("nfl/team_stats.csv")

def get_player_stats(player_name: str, seasons: list[str]):
    player_stats = nfl.load_player_stats(seasons).to_pandas()
    df = player_stats[player_stats["player_display_name"] == player_name]
    df.to_csv("nfl/player_stats.csv")
