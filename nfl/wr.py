import nfl.pass_attempts as pass_attempts
import nfl.target_share as target_share
import nfl.catch_rate as catch_rate
import nfl.yards_per_catch as yards_per_catch
from model import model_predict

def yard_prediction(player_name: str):
    df = pass_attempts.get_pass_attempts(2025)
    X = df[["pass_attempts_last_3", "pass_attempts_last_5", "pass_attempts_last_8"]]
    y = df[["attempts"]]

    pass_attempts_prediction = model_predict(X, y)

    df = target_share.get_target_share(player_name, [2020, 2021, 2022, 2023, 2024, 2025])

    X = df[["target_share_last_3", "target_share_last_5", "target_share_last_8",
            "targets_last_3", "targets_last_5", "targets_last_8",
            "season_type"]]
    y = df[["target_share"]]

    target_share_prediction = model_predict(X, y)

    df = catch_rate.get_catch_rate(player_name, [2020, 2021, 2022, 2023, 2024, 2025])

    X = df[["receptions_last_3", "receptions_last_5", "receptions_last_8",
            "targets_last_3", "targets_last_5", "targets_last_8",
            "season_type"]]
    y = df[["catch_rate"]]

    catch_rate_prediction = model_predict(X, y)

    df = yards_per_catch.get_yards_per_catch(player_name, [2020, 2021, 2022, 2023, 2024, 2025])

    X = df[["receiving_air_yards_last_3", "receiving_air_yards_last_5", "receiving_air_yards_last_8",
            "receiving_yards_after_catch_last_3", "receiving_yards_after_catch_last_5", 
            "receiving_yards_after_catch_last_8", "season_type"]]
    y = df[["yards_per_catch"]]

    yards_per_catch_prediction = model_predict(X, y)

    return pass_attempts_prediction * target_share_prediction * catch_rate_prediction * yards_per_catch_prediction

if __name__ == "__main__":
    print(yard_prediction("Cooper Kupp"))