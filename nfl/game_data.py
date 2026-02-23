import nflreadpy as nfl
import pandas as pd

def get_team_data():
    years = [2020, 2021, 2022, 2023, 2024, 2025]

def process_pbp_data():
    years = [2025]
    games_list = []

    for year in years:
        df = nfl.load_pbp(year).to_pandas()
        current_file_name = ""
        current_file_plays = []

        for idx, row in df.iterrows():
            if row["play_id"] == 1.0:
                if current_file_name != "":
                    current_game_df = pd.DataFrame(current_file_plays)
                    current_game_df.to_csv(f"pbp/{current_file_name}.csv")
                    current_file_plays = []
                
                if row["weather"] == None:
                    row["weather"] = "None Temp: None° F, Humidity: None%, Wind: NE None mph"

                games_list.append({
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "season_type": row["season_type"],
                    "season": row["season"],
                    "week": row["week"],
                    "game_date": row["game_date"],
                    "start_time": row["start_time"],
                    "game_id": row["game_id"],
                    "stadium": row["game_stadium"],
                    "roof": row["roof"],
                    "surface": row["surface"],
                    "condition": row["weather"].split("Temp")[0],
                    "temperature": row["weather"].split(":")[1].split("°")[0].strip(),
                    "humidity": row["weather"].split(":")[2].split("%")[0].strip(),
                    "wind": row["weather"].split(":")[3].split("mph")[0].strip(),
                })

                current_file_name = row["game_id"]

            else: 
                filtered_row = row[["posteam", "side_of_field", "yardline_100", "game_seconds_remaining", "drive",
                    "down", "goal_to_go", "yrdln", "ydstogo", "ydsnet", "desc", "play_type",
                    "yards_gained", "shotgun", "no_huddle", "qb_dropback", "qb_kneel", "qb_spike",
                    "qb_scramble", "pass_length", "pass_location", "air_yards", "yards_after_catch",
                    "run_location", "run_gap", "field_goal_result", "kick_distance", "extra_point_result",
                    "two_point_conv_result", "timeout", "timeout_team", "td_team", "td_player_name",
                    "total_home_score", "total_away_score", "score_differential", 
                    "score_differential_post", "punt_blocked", "first_down_rush", "first_down_pass",
                    "first_down_penalty", "third_down_converted", "third_down_failed",
                    "fourth_down_converted", "fourth_down_failed", "incomplete_pass", "touchback",
                    "interception", "punt_inside_twenty", "punt_in_endzone", "punt_out_of_bounds", 
                    "punt_downed", "punt_fair_catch", "kickoff_inside_twenty", "kickoff_in_endzone",
                    "kickoff_out_of_bounds", "kickoff_downed", "kickoff_fair_catch", "fumble_forced",
                    "fumble_not_forced", "fumble_out_of_bounds", "solo_tackle", "safety", "penalty",
                    "tackled_for_loss", "fumble_lost", "own_kickoff_recovery", 
                    "own_kickoff_recovery_td", "qb_hit", "rush_attempt", "pass_attempt", "sack", 
                    "touchdown", "pass_touchdown", "rush_touchdown", "extra_point_attempt", 
                    "two_point_attempt", "field_goal_attempt", "kickoff_attempt", "punt_attempt", 
                    "fumble", "complete_pass", "assist_tackle", "lateral_reception", "lateral_rush",
                    "lateral_return", "lateral_recovery", "passer_player_name", "passing_yards",
                    "receiver_player_name", "receiving_yards", "rusher_player_name", "rushing_yards", 
                    "lateral_receiver_player_name", "lateral_receiving_yards", 
                    "lateral_rusher_player_name", "lateral_rushing_yards",   
                    "punter_player_name", "kicker_player_name", "tackle_with_assist", "return_team", 
                    "return_yards", "penalty_team", "penalty_yards", "penalty_type", "series_result", 
                    "special_teams_play"]]
                
                current_file_plays.append(filtered_row.to_dict())
                
    current_game_df = pd.DataFrame(current_file_plays)
    current_game_df.to_csv(f"pbp/{current_file_name}.csv")
    
    games_df = pd.DataFrame(games_list) 
    games_df.to_csv("pbp/games.csv")

if __name__ == "__main__":
    process_pbp_data()