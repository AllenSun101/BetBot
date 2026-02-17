from model import model
import pandas as pd

def get_target_share():
    df = pd.read_csv("nfl/player_stats.csv")

    df["target_share_last_3"] = (df['target_share'].shift(1).rolling(3, min_periods=1).mean())
    df["target_share_last_5"] = (df['target_share'].shift(1).rolling(5, min_periods=1).mean())
    df["target_share_last_8"] = (df['target_share'].shift(1).rolling(8, min_periods=1).mean())
    df["targets_last_3"] = (df['targets'].shift(1).rolling(3, min_periods=1).mean())
    df["targets_last_5"] = (df['targets'].shift(1).rolling(5, min_periods=1).mean())
    df["targets_last_8"] = (df['targets'].shift(1).rolling(8, min_periods=1).mean())
    df["season_type"] = df["season_type"].astype('category')
    # defensive statistics
    
    return df

if __name__ == "__main__":
    df = get_target_share()

    X = df[["target_share_last_3", "target_share_last_5", "target_share_last_8",
            "targets_last_3", "targets_last_5", "targets_last_8",
            "season_type"]]
    y = df[["target_share"]]
    
    X_train = X.iloc[:-1]
    X_test = X.iloc[-1:]
    y_train = y.iloc[:-1]
    y_test = y.iloc[-1:]

    print(model(X_train, X_test, y_train, y_test))
