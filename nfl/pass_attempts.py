import nflreadpy as nfl
from model import model

def get_pass_attempts(season: int):
    team_stats = nfl.load_team_stats([season]).to_pandas()

    team_stats['pass_attempts_last_3'] = (team_stats.groupby('team')['attempts'].shift(1).rolling(3, min_periods=1).mean())
    team_stats['pass_attempts_last_5'] = (team_stats.groupby('team')['attempts'].shift(1).rolling(5, min_periods=1).mean())
    team_stats['pass_attempts_last_8'] = (team_stats.groupby('team')['attempts'].shift(1).rolling(8, min_periods=1).mean())

    return team_stats

if __name__ == "__main__":
    df = get_pass_attempts(2025)
    X = df[["pass_attempts_last_3", "pass_attempts_last_5", "pass_attempts_last_8"]]
    y = df[["attempts"]]
    
    X_train = X.iloc[:-1]
    X_test = X.iloc[-1:]
    y_train = y.iloc[:-1]
    y_test = y.iloc[-1:]

    print(model(X_train, X_test, y_train, y_test))
