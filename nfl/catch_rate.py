import numpy as np
import nflreadpy as nfl
from model import model

def get_catch_rate(player_name: str, seasons: list[int]):
    player_stats = nfl.load_player_stats(seasons).to_pandas()
    df = player_stats[player_stats["player_display_name"] == player_name].copy()
    
    df["catch_rate"] = df["receptions"] / df["targets"].replace(0, np.nan)
    df["receptions_last_3"] = (df['receptions'].shift(1).rolling(3, min_periods=1).mean())
    df["receptions_last_5"] = (df['receptions'].shift(1).rolling(5, min_periods=1).mean())
    df["receptions_last_8"] = (df['receptions'].shift(1).rolling(8, min_periods=1).mean())
    df["targets_last_3"] = (df['targets'].shift(1).rolling(3, min_periods=1).mean())
    df["targets_last_5"] = (df['targets'].shift(1).rolling(5, min_periods=1).mean())
    df["targets_last_8"] = (df['targets'].shift(1).rolling(8, min_periods=1).mean())
    df["season_type"] = df["season_type"].astype('category')
    # defensive statistics
    
    return df

if __name__ == "__main__":
    df = get_catch_rate("Cooper Kupp", [2020, 2021, 2022, 2023, 2024, 2025])

    X = df[["receptions_last_3", "receptions_last_5", "receptions_last_8",
            "targets_last_3", "targets_last_5", "targets_last_8",
            "season_type"]]
    y = df[["catch_rate"]]
    
    X_train = X.iloc[:-1]
    X_test = X.iloc[-1:]
    y_train = y.iloc[:-1]
    y_test = y.iloc[-1:]

    print(model(X_train, X_test, y_train, y_test))
