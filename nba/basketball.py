from nba_api.stats.endpoints import playercareerstats, playergamelogs, playerindex

career = playercareerstats.PlayerCareerStats(player_id='203999')
df = career.season_totals_regular_season.get_data_frame()
print(df)

"site.api.espn.com/apis/site/v2/sports/basketball/nba/news"

# minutes * team shots/minute * shots/team shots * points/shot
# how do we model different player combinations

# minutes * pace * shots share * accuracy

# Player stats -> accuracy
# Team stats - Box score -> Pace, shots share
# Injury reports -> minutes, shots share

game_logs = playergamelogs.PlayerGameLogs(player_id_nullable='203999', season_nullable=2025)
df = game_logs.get_data_frames()
print(df)

player_info = playerindex.PlayerIndex(season="2025-26")
df = player_info.get_data_frames()[0]
df.to_csv("players.csv")
# get team roster, consider all others in each game? -> minutes played for each one -> players team field

# vs. all else who played -> share of total minutes when playing -> when one is zero, get impact

