from model import model
import numpy as np
import pandas as pd

def get_yards_per_catch():
    df = pd.read_csv("nfl/player_stats.csv")
    
    df["yards_per_catch"] = df["receiving_yards"] / df["receptions"].replace(0, np.nan)
    df["receiving_air_yards_last_3"] = (df['receiving_air_yards'].shift(1).rolling(3, min_periods=1).mean())
    df["receiving_air_yards_last_5"] = (df['receiving_air_yards'].shift(1).rolling(5, min_periods=1).mean())
    df["receiving_air_yards_last_8"] = (df['receiving_air_yards'].shift(1).rolling(8, min_periods=1).mean())
    df["receiving_yards_after_catch_last_3"] = (df['receiving_yards_after_catch'].shift(1).rolling(3, min_periods=1).mean())
    df["receiving_yards_after_catch_last_5"] = (df['receiving_yards_after_catch'].shift(1).rolling(5, min_periods=1).mean())
    df["receiving_yards_after_catch_last_8"] = (df['receiving_yards_after_catch'].shift(1).rolling(8, min_periods=1).mean())
    df["season_type"] = df["season_type"].astype('category')
    # defensive statistics

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["yards_per_catch"])
    
    return df

if __name__ == "__main__":
    df = get_yards_per_catch()

    X = df[["receiving_air_yards_last_3", "receiving_air_yards_last_5", "receiving_air_yards_last_8",
            "receiving_yards_after_catch_last_3", "receiving_yards_after_catch_last_5", 
            "receiving_yards_after_catch_last_8", "season_type"]]
    y = df[["yards_per_catch"]]
    
    X_train = X.iloc[:-1]
    X_test = X.iloc[-1:]
    y_train = y.iloc[:-1]
    y_test = y.iloc[-1:]

    print(model(X_train, X_test, y_train, y_test))
