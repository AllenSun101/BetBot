"""
Feature Engineering Pipeline — Minutes Prediction (v2)
Target: MIN (minutes played per game)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config import DATA_PATH, ML_DATASET_MIN, X_FEATURES_MIN, Y_TARGET_MIN

def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
    df = df[df['MIN'] > 2].copy()
    print(f"  Rows after MIN > 2 filter: {len(df)}")
    return df

def basic_features(df):
    print("Engineering basic features...")
    df['IS_HOME'] = df['MATCHUP'].str.contains(r'vs\.', na=False).astype(int)
    df['WON'] = (df['WL'] == 'W').astype(int)

    df['USAGE_RATE_PROXY'] = (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / df['MIN'].clip(lower=0.1)

    df['EFFICIENCY_PER_MIN'] = (
        df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK']
        - df['TOV'] - (df['FGA'] - df['FGM'])
    ) / df['MIN'].clip(lower=0.1)

    df['TRUE_SHOOTING'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA'])).clip(lower=0.1)
    df['AST_TO_RATIO'] = df['AST'] / (df['TOV'] + 0.1)

    df['STOCKS'] = df['STL'] + df['BLK']
    df['ACTIONS_PER_MIN'] = (df['FGA'] + df['FTA'] + df['AST'] + df['REB']) / df['MIN'].clip(lower=0.1)

    df['BLOWOUT_FLAG'] = (df['PLUS_MINUS'].abs() > 15).astype(int)
    df['CLOSE_GAME_FLAG'] = (df['PLUS_MINUS'].abs() <= 5).astype(int)

    df['GAME_NUM_IN_SEASON'] = df.groupby('PLAYER_ID').cumcount() + 1
    df['DAY_OF_WEEK'] = df['GAME_DATE'].dt.dayofweek
    df['MONTH'] = df['GAME_DATE'].dt.month
    df['LATE_SEASON'] = df['MONTH'].isin([3, 4]).astype(int)

    return df

def add_rolling_features(df, stats, windows=[3,5,10]):
    grp = df.groupby('PLAYER_ID')
    for w in windows:
        for col in stats:
            df[f'{col}_ROLL{w}'] = grp[col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=max(1, w//2)).mean()
            )
            if col == 'MIN':
                df[f'{col}_ROLL{w}_STD'] = grp[col].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=2).std()
                )
    return df

def add_ewma_features(df):
    for col in ['MIN', 'PTS', 'EFFICIENCY_PER_MIN', 'USAGE_RATE_PROXY']:
        for span in [5, 10]:
            df[f'{col}_EWM{span}'] = df.groupby('PLAYER_ID')[col].transform(
                lambda x: x.shift(1).ewm(span=span, min_periods=2).mean()
            )
    return df

def rolling_features(df):
    print("Engineering rolling features...")

    STATS = ['MIN','PTS','FGA','FG_PCT','FG3_PCT','FT_PCT','REB','AST',
             'STL','BLK','TOV','PF','PLUS_MINUS','EFFICIENCY_PER_MIN',
             'USAGE_RATE_PROXY','TRUE_SHOOTING','STOCKS','AST_TO_RATIO']

    df = add_rolling_features(df, STATS)
    df = add_ewma_features(df)

    df['MIN_ROLL5_MEDIAN'] = df.groupby('PLAYER_ID')['MIN'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).median()
    )

    df['MIN_ROLL10_MEDIAN'] = df.groupby('PLAYER_ID')['MIN'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).median()
    )

    df['MIN_TREND_5G'] = df.groupby('PLAYER_ID')['MIN'].transform(
        lambda x: rolling_slope(x, 5)
    )

    return df

def rolling_slope(series, window=5):
    result = series.copy() * np.nan
    s_shifted = series.shift(1)
    for i in range(window, len(series)):
        y = s_shifted.iloc[i-window:i].values
        mask = ~np.isnan(y)
        if mask.sum() >= 2:
            result.iloc[i] = np.polyfit(np.arange(window)[mask], y[mask], 1)[0]
    return result

def rest_features(df):
    print("Engineering rest/fatigue features...")

    df['DAYS_REST'] = df.groupby('PLAYER_ID')['GAME_DATE'].transform(
        lambda x: x.diff().dt.days.clip(upper=10)
    )

    df['IS_BACK_TO_BACK'] = (df['DAYS_REST'] == 1).astype(int)
    df['IS_3_IN_4'] = (df['DAYS_REST'] <= 1).astype(int)

    df['BTB_ROAD'] = ((df['DAYS_REST'] == 1) & (df['IS_HOME'] == 0)).astype(int)
    df['BTB_HOME'] = ((df['DAYS_REST'] == 1) & (df['IS_HOME'] == 1)).astype(int)

    df['EXTENDED_REST'] = (df['DAYS_REST'] >= 4).astype(int)

    df['CUMULATIVE_MIN_10D'] = df.groupby('PLAYER_ID', group_keys=False).apply(
        lambda g: minutes_in_window_days(g, 10)
    )

    df['GAMES_SINCE_RETURN'] = df.groupby('PLAYER_ID', group_keys=False).apply(
        games_since_long_absence
    )

    df['IRONMAN_STREAK'] = df.groupby('PLAYER_ID')['MIN'].transform(
        ironman_streak
    )

    return df

def minutes_in_window_days(grp, days=10):
    result = []
    for idx, row in grp.iterrows():
        cutoff = row['GAME_DATE'] - pd.Timedelta(days=days)
        past = grp[(grp['GAME_DATE'] >= cutoff) & (grp['GAME_DATE'] < row['GAME_DATE'])]
        result.append(past['MIN'].sum())
    return pd.Series(result, index=grp.index)

def games_since_long_absence(grp, absence_thresh=7):
    """Number of games since the player returned from a long absence (0 if never / normal)."""
    result = pd.Series(0, index=grp.index)
    days_rest = grp['GAME_DATE'].diff().dt.days.fillna(2)
    in_ramp = False
    ramp_count = 0
    for i, (idx, dr) in enumerate(zip(grp.index, days_rest)):
        if in_ramp:
            ramp_count += 1
            result[idx] = ramp_count
            if ramp_count >= 5:
                in_ramp = False
        if dr >= absence_thresh:
            in_ramp = True
            ramp_count = 0
    return result

def ironman_streak(series, threshold=30):
    shifted = series.shift(1)
    result = pd.Series(0, index=series.index)
    streak = 0
    for i in range(len(shifted)):
        v = shifted.iloc[i]
        if pd.notna(v) and v >= threshold:
            streak += 1
        else:
            streak = 0
        result.iloc[i] = streak
    return result

def opponent_features(df):
    print("Engineering opponent features...")

    df['OPPONENT'] = df['MATCHUP'].apply(get_opponent)

    df['OPP_AVG_MIN_ALLOWED'] = df.groupby('OPPONENT')['MIN'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    df['OPP_STRENGTH'] = -df.groupby('OPPONENT')['PLUS_MINUS'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    df['OPP_PACE_PROXY'] = df.groupby('OPPONENT')['FGA'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    df['H2H_MIN_VS_OPP'] = df.groupby(['PLAYER_ID','OPPONENT'])['MIN'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    return df

def get_opponent(matchup):
    if not isinstance(matchup, str): return None
    parts = matchup.split(' ')
    return parts[-1] if len(parts) >= 3 else None

def team_features(df):
    print("Engineering team-relative features...")

    df['MIN_RANK_IN_TEAM'] = df.groupby(['GAME_DATE','TEAM_ABBREVIATION'])['MIN_ROLL5']\
        .rank(ascending=False, method='dense')

    team_total = df.groupby(['GAME_DATE','TEAM_ABBREVIATION'])['MIN'].transform('sum')
    df['TEAM_MIN_SHARE'] = df['MIN'] / team_total.clip(lower=1)

    df['TEAM_MIN_SHARE_ROLL5'] = df.groupby('PLAYER_ID')['TEAM_MIN_SHARE'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean()
    )

    df['TEAM_MIN_STD'] = df.groupby(['GAME_DATE','TEAM_ABBREVIATION'])['MIN'].transform('std')

    df['TEAM_DEPTH_ROLL5'] = df.groupby('PLAYER_ID')['TEAM_MIN_STD'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean()
    )

    return df

def player_role_features(df):
    print("Engineering role/archetype features...")
    for col in ['MIN', 'PTS', 'FGA', 'REB', 'AST', 'STL', 'BLK']:
        df[f'{col}_SZN_AVG'] = df.groupby('PLAYER_ID')[col].transform(
            lambda x: x.shift(1).expanding().mean())
    df['SCORER_SCORE']    = df['PTS_SZN_AVG'] + df['FGA_SZN_AVG'] * 0.5
    df['PLAYMAKER_SCORE'] = df['AST_SZN_AVG'] * 2
    df['DEFENDER_SCORE']  = (df['STL_SZN_AVG'] + df['BLK_SZN_AVG']) * 2
    df['REBOUNDER_SCORE'] = df['REB_SZN_AVG']
    df['MIN_CONSISTENCY'] = df.groupby('PLAYER_ID')['MIN'].transform(
        lambda x: x.shift(1).expanding().std())

    df['CAREER_GAMES'] = df.groupby('PLAYER_ID').cumcount()
    df['IS_ROOKIE_PHASE']   = (df['CAREER_GAMES'] < 82).astype(int)
    df['IS_VETERAN_PHASE']  = (df['CAREER_GAMES'] > 400).astype(int)

    return df

def volatility_and_momentum_features(df):
    print("Engineering volatility and momentum features...")
    df['PTS_ZSCORE_RECENT'] = (
        (df['PTS_ROLL5'] - df['PTS_SZN_AVG']) /
        df.groupby('PLAYER_ID')['PTS'].transform(
            lambda x: x.shift(1).expanding().std()).clip(lower=0.1))
    df['PF_ROLL5'] = df.groupby('PLAYER_ID')['PF'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    df['HIGH_FOUL_RISK'] = (df['PF_ROLL5'] > 3.5).astype(int)
    df['PM_MOMENTUM'] = df.groupby('PLAYER_ID')['PLUS_MINUS'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean())

    df['MIN_VOL_RATIO'] = df['MIN_ROLL5_STD'] / df['MIN_ROLL5'].clip(lower=1)

    df['USAGE_SPIKE'] = (
        (df['USAGE_RATE_PROXY_ROLL5'] - df.groupby('PLAYER_ID')['USAGE_RATE_PROXY'].transform(
            lambda x: x.shift(1).expanding().mean())) /
        df.groupby('PLAYER_ID')['USAGE_RATE_PROXY'].transform(
            lambda x: x.shift(1).expanding().std()).clip(lower=0.01))

    return df

def interaction_features(df):
    print("Engineering interaction features...")
    df['FATIGUE_x_CLOSER']   = df['CUMULATIVE_MIN_10D'] * df['CLOSE_GAME_FLAG']
    df['SCORER_x_HOME']      = df['SCORER_SCORE'] * df['IS_HOME']
    df['EFFICIENCY_x_OPP']   = df['EFFICIENCY_PER_MIN'] * df['OPP_STRENGTH'].fillna(0)
    df['ROLL5_MIN_x_REST']   = df['MIN_ROLL5'] * df['DAYS_REST'].fillna(2)
    df['CONSISTENCY_x_RANK'] = df['MIN_CONSISTENCY'] * (1 / df['MIN_RANK_IN_TEAM'].clip(lower=1))

    df['VETERAN_x_REST']    = df['IS_VETERAN_PHASE'] * df['DAYS_REST'].fillna(2)
    df['EWM5_x_OPP_PACE']  = df['MIN_EWM5'] * df['OPP_PACE_PROXY'].fillna(0)
    df['RAMP_x_ROLE']       = df['GAMES_SINCE_RETURN'] * df['MIN_SZN_AVG']
    df['DEPTH_x_RANK']      = df['TEAM_DEPTH_ROLL5'] * df['MIN_RANK_IN_TEAM']
    df['VOL_RATIO_x_STREAK']= df['MIN_VOL_RATIO'] * df['IRONMAN_STREAK']

    return df

def game_context_features(df):
    print("Engineering game-level context features...")
    df['WIN_PCT_SZN'] = df.groupby('PLAYER_ID')['WON'].transform(
        lambda x: x.shift(1).expanding().mean())
    
    return df

def final_assembly(df):
    print("Assembling final dataset...")
    FEATURE_COLS = [
        # Context
        'GAME_NUM_IN_SEASON','IS_HOME','DAY_OF_WEEK','DAYS_REST','IS_BACK_TO_BACK','BTB_ROAD','BTB_HOME','EXTENDED_REST',
        'IS_3_IN_4','LATE_SEASON','MONTH','BLOWOUT_FLAG','CLOSE_GAME_FLAG',
        # Rolling means
        'MIN_ROLL3','MIN_ROLL5','MIN_ROLL10','MIN_ROLL3_STD','MIN_ROLL5_STD','MIN_ROLL10_STD','MIN_ROLL5_MEDIAN','MIN_ROLL10_MEDIAN','MIN_TREND_5G',
        # EWMA
        'MIN_EWM5','MIN_EWM10','PTS_EWM5','PTS_EWM10','EFFICIENCY_PER_MIN_EWM5','USAGE_RATE_PROXY_EWM5',
        # Scoring/shooting rolls
        'PTS_ROLL3','PTS_ROLL5','PTS_ROLL10','FGA_ROLL5','FG_PCT_ROLL5','FG3_PCT_ROLL5','FT_PCT_ROLL5','REB_ROLL5','AST_ROLL5','STL_ROLL5','BLK_ROLL5',
        'TOV_ROLL5','PF_ROLL5','PLUS_MINUS_ROLL5','EFFICIENCY_PER_MIN_ROLL5','USAGE_RATE_PROXY_ROLL5','TRUE_SHOOTING_ROLL5','STOCKS_ROLL5','AST_TO_RATIO_ROLL5',
        # Season averages
        'MIN_SZN_AVG','PTS_SZN_AVG','FGA_SZN_AVG','REB_SZN_AVG','AST_SZN_AVG','STL_SZN_AVG','BLK_SZN_AVG',
        # Role scores
        'SCORER_SCORE','PLAYMAKER_SCORE','DEFENDER_SCORE','REBOUNDER_SCORE','MIN_CONSISTENCY','MIN_VOL_RATIO',
        # Team context
        'TEAM_MIN_SHARE_ROLL5','MIN_RANK_IN_TEAM','TEAM_DEPTH_ROLL5',
        # Opponent
        'OPP_AVG_MIN_ALLOWED','OPP_STRENGTH','OPP_PACE_PROXY','H2H_MIN_VS_OPP',
        # Load / fatigue
        'CUMULATIVE_MIN_10D','GAMES_SINCE_RETURN','IRONMAN_STREAK',
        # Career phase
        'IS_ROOKIE_PHASE','IS_VETERAN_PHASE',
        # Momentum / volatility
        'PTS_ZSCORE_RECENT','HIGH_FOUL_RISK','PM_MOMENTUM','WIN_PCT_SZN','USAGE_SPIKE',
        # Interactions
        'FATIGUE_x_CLOSER','SCORER_x_HOME','EFFICIENCY_x_OPP','ROLL5_MIN_x_REST','CONSISTENCY_x_RANK','VETERAN_x_REST','EWM5_x_OPP_PACE','RAMP_x_ROLE',
        'DEPTH_x_RANK','VOL_RATIO_x_STREAK',
    ]

    X = df[FEATURE_COLS].copy()
    y = df['MIN'].copy()
    valid_mask = X['MIN_ROLL5'].notna() & X['MIN_ROLL3'].notna()
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    X = X.fillna(X.median(numeric_only=True))

    print(f"\n✓ Final X shape : {X.shape}")
    print(f"✓ Target (MIN)  : mean={y.mean():.1f}  std={y.std():.1f}")

    return X, y

def save_outputs(X, y):
    X.to_csv(X_FEATURES_MIN, index=False)
    y.to_csv(Y_TARGET_MIN,   index=False)
    combined = X.copy(); combined['MIN_TARGET'] = y
    combined.to_csv(ML_DATASET_MIN, index=False)
    print(f"\n✓ Saved to {ML_DATASET_MIN}")

def main():
    df = load_data()
    df = basic_features(df)
    df = rolling_features(df)
    df = rest_features(df)
    df = opponent_features(df)
    df = team_features(df)
    df = player_role_features(df)
    df = volatility_and_momentum_features(df)
    df = interaction_features(df)
    df = game_context_features(df)

    X, y = final_assembly(df)
    save_outputs(X, y)

if __name__ == "__main__":
    main()