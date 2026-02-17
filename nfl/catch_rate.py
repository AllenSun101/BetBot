import numpy as np
import pandas as pd
from model import model

def get_catch_rate():
    df = pd.read_csv("nfl/player_stats.csv")
    
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
    df = get_catch_rate()

    X = df[["receptions_last_3", "receptions_last_5", "receptions_last_8",
            "targets_last_3", "targets_last_5", "targets_last_8",
            "season_type"]]
    y = df[["catch_rate"]]
    
    X_train = X.iloc[:-1]
    X_test = X.iloc[-1:]
    y_train = y.iloc[:-1]
    y_test = y.iloc[-1:]

    print(model(X_train, X_test, y_train, y_test))
