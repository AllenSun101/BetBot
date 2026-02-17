import nfl.data as data
from nfl.wr import yard_prediction
import pandas as pd
import nfl.pass_attempts as pass_attempts
import nfl.target_share as target_share
import nfl.catch_rate as catch_rate
import nfl.yards_per_catch as yards_per_catch

prediction_map = {
    "wr_yards": yard_prediction
}

def test_model(prediction_type: str):
    df = pd.read_csv("nfl/player_stats.csv")
    last_week = df.iloc[-1]["week"]

    pass_attempts_df = pass_attempts.get_pass_attempts()
    target_share_df = target_share.get_target_share()
    catch_rate_df = catch_rate.get_catch_rate()
    yards_per_catch_df = yards_per_catch.get_yards_per_catch()

    tests = []
    squared_error = 0
    absolute_error = 0

    for week in range(last_week+1):
        target_week = df[(df["season"] == 2025) & (df["week"] == week)]

        if len(target_week) == 0:
            continue
        
        actual_value = target_week.iloc[0]["receiving_yards"]
        prediction = prediction_map[prediction_type](week, pass_attempts_df, target_share_df, 
                                                     catch_rate_df, yards_per_catch_df)
        error = abs(actual_value - prediction)
        absolute_error += error
        squared_error += error * error
        tests.append({"week": week, "predicted": prediction, "actual": actual_value, "error": error})

    return tests, squared_error, absolute_error

if __name__ == "__main__":
    player_name = "Cooper Kupp"
    prediction_type = "wr_yards"
    season = 2025
    lookback_seasons = [2020, 2021, 2022, 2023, 2024, 2025]

    data.get_team_stats([season])
    data.get_player_stats(player_name, lookback_seasons)

    results, mse, mae = test_model(prediction_type)
    results_df = pd.DataFrame(results)
    print(results_df)
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")