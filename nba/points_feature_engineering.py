"""
points_feature_engineering.py — Points Prediction Feature Engineering (v3)
Target: PTS (points scored in a game)

Refactored from a top-level script into a proper module with a main() entry
point, reusable helper functions, and no side-effects on import.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle

from config import (
    DATA_PATH, MIN_MODEL_PKL,
    ML_DATASET_PTS, X_FEATURES_PTS, Y_TARGET_PTS, PTS_PLAYER_NAMES,
)
from injury_features import build_injury_features, merge_injury_features_into_df, _all_injury_feature_cols

# ── Rolling / window helpers ───────────────────────────────────────────────────

def rolling_mean(series: pd.Series, w: int, mp: int | None = None) -> pd.Series:
    mp = mp or max(1, w // 2)
    return series.shift(1).rolling(w, min_periods=mp).mean()


def rolling_slope(series: pd.Series, window: int = 5) -> pd.Series:
    result = series.copy() * np.nan
    s = series.shift(1)
    for i in range(window, len(series)):
        y = s.iloc[i - window:i].values
        mask = ~np.isnan(y)
        if mask.sum() >= 2:
            result.iloc[i] = np.polyfit(np.arange(window)[mask], y[mask], 1)[0]
    return result


def expanding_mean(series: pd.Series) -> pd.Series:
    return series.shift(1).expanding().mean()


# ── Feature-engineering steps ──────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    df = df[df["MIN"] > 2].copy()
    print(f"  Rows: {len(df)}")
    return df


def add_basic_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Game-level binary flags and calendar features."""
    print("Basic flags...")
    df["IS_HOME"] = df["MATCHUP"].str.contains(r"vs\.", na=False).astype(int)
    df["WON"] = (df["WL"] == "W").astype(int)
    df["GAME_NUM_IN_SEASON"] = df.groupby("PLAYER_ID").cumcount() + 1
    df["DAY_OF_WEEK"] = df["GAME_DATE"].dt.dayofweek
    df["MONTH"] = df["GAME_DATE"].dt.month
    df["LATE_SEASON"] = df["MONTH"].isin([3, 4]).astype(int)
    df["BLOWOUT_FLAG"] = (df["PLUS_MINUS"].abs() > 15).astype(int)
    df["CLOSE_GAME_FLAG"] = (df["PLUS_MINUS"].abs() <= 5).astype(int)

    df["DAYS_REST"] = df.groupby("PLAYER_ID")["GAME_DATE"].transform(
        lambda x: x.diff().dt.days.clip(upper=10)
    )
    df["IS_BACK_TO_BACK"] = (df["DAYS_REST"] == 1).astype(int)
    df["IS_3_IN_4"] = (df["DAYS_REST"].fillna(2) <= 1).astype(int)
    df["BTB_ROAD"] = ((df["DAYS_REST"] == 1) & (df["IS_HOME"] == 0)).astype(int)
    df["BTB_HOME"] = ((df["DAYS_REST"] == 1) & (df["IS_HOME"] == 1)).astype(int)
    df["EXTENDED_REST"] = (df["DAYS_REST"] >= 4).astype(int)
    return df


def _cumulative_min_10d(grp: pd.DataFrame) -> pd.Series:
    result = []
    for idx, row in grp.iterrows():
        cutoff = row["GAME_DATE"] - pd.Timedelta(days=10)
        past = grp[(grp["GAME_DATE"] >= cutoff) & (grp["GAME_DATE"] < row["GAME_DATE"])]
        result.append(past["MIN"].sum())
    return pd.Series(result, index=grp.index)


def _games_since_long_absence(grp: pd.DataFrame, absence_thresh: int = 7) -> pd.Series:
    result = pd.Series(0, index=grp.index)
    days_rest = grp["GAME_DATE"].diff().dt.days.fillna(2)
    in_ramp = False
    ramp_count = 0
    for idx, dr in zip(grp.index, days_rest):
        if in_ramp:
            ramp_count += 1
            result[idx] = ramp_count
            if ramp_count >= 5:
                in_ramp = False
        if dr >= absence_thresh:
            in_ramp = True
            ramp_count = 0
    return result


def _ironman_streak(series: pd.Series, threshold: int = 30) -> pd.Series:
    shifted = series.shift(1)
    result = pd.Series(0, index=series.index)
    streak = 0
    for i in range(len(shifted)):
        v = shifted.iloc[i]
        streak = streak + 1 if (pd.notna(v) and v >= threshold) else 0
        result.iloc[i] = streak
    return result


def add_minutes_passthrough_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild all the rolling/EWMA features originally produced by
    minutes_feature_engineering.py so the points model can use them
    as pass-through inputs.
    """
    print("Rebuilding minutes-model pass-through features...")

    grp = df.groupby("PLAYER_ID")

    # Basic derived stats
    df["USAGE_RATE_PROXY"] = (df["FGA"] + 0.44 * df["FTA"] + df["TOV"]) / df["MIN"].clip(lower=0.1)
    df["EFFICIENCY_PER_MIN"] = (
        df["PTS"] + df["REB"] + df["AST"] + df["STL"] + df["BLK"]
        - df["TOV"] - (df["FGA"] - df["FGM"])
    ) / df["MIN"].clip(lower=0.1)
    df["TRUE_SHOOTING"] = df["PTS"] / (2 * (df["FGA"] + 0.44 * df["FTA"])).clip(lower=0.1)
    df["AST_TO_RATIO"] = df["AST"] / (df["TOV"] + 0.1)
    df["STOCKS"] = df["STL"] + df["BLK"]

    # Rolling means for core stats
    for col in ["MIN", "PTS", "FGA", "FG_PCT", "FG3_PCT", "FT_PCT",
                "REB", "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS"]:
        for w in [3, 5, 10]:
            df[f"{col}_ROLL{w}"] = grp[col].transform(lambda x, ww=w: rolling_mean(x, ww))
        df[f"{col}_SZN_AVG"] = grp[col].transform(expanding_mean)

    # Rolling stds for MIN
    for w in [3, 5, 10]:
        df[f"MIN_ROLL{w}_STD"] = grp["MIN"].transform(
            lambda x, ww=w: x.shift(1).rolling(ww, min_periods=2).std()
        )

    # Rolling medians
    df["MIN_ROLL5_MEDIAN"] = grp["MIN"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).median()
    )
    df["MIN_ROLL10_MEDIAN"] = grp["MIN"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).median()
    )

    # EWMA
    for col in ["MIN", "PTS", "EFFICIENCY_PER_MIN", "USAGE_RATE_PROXY"]:
        for span in [5, 10]:
            df[f"{col}_EWM{span}"] = grp[col].transform(
                lambda x, s=span: x.shift(1).ewm(span=s, min_periods=2).mean()
            )

    print("  MIN trend slope...")
    df["MIN_TREND_5G"] = grp["MIN"].transform(lambda x: rolling_slope(x, 5))

    # Derived stat rolls
    for col in ["USAGE_RATE_PROXY", "EFFICIENCY_PER_MIN", "TRUE_SHOOTING", "STOCKS", "AST_TO_RATIO"]:
        df[f"{col}_ROLL5"] = grp[col].transform(lambda x: rolling_mean(x, 5))

    # Opponent features
    def _get_opp(matchup: str) -> str | None:
        if not isinstance(matchup, str):
            return None
        parts = matchup.split(" ")
        return parts[-1] if len(parts) >= 3 else None

    df["OPPONENT"] = df["MATCHUP"].apply(_get_opp)
    df["OPP_AVG_MIN_ALLOWED"] = df.groupby("OPPONENT")["MIN"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["OPP_STRENGTH"] = -df.groupby("OPPONENT")["PLUS_MINUS"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["OPP_PACE_PROXY"] = df.groupby("OPPONENT")["FGA"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["H2H_MIN_VS_OPP"] = df.groupby(["PLAYER_ID", "OPPONENT"])["MIN"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Team features
    team_total_min = df.groupby(["GAME_DATE", "TEAM_ABBREVIATION"])["MIN"].transform("sum")
    df["TEAM_MIN_SHARE"] = df["MIN"] / team_total_min.clip(lower=1)
    df["TEAM_MIN_SHARE_ROLL5"] = grp["TEAM_MIN_SHARE"].transform(lambda x: rolling_mean(x, 5))
    df["MIN_RANK_IN_TEAM"] = df.groupby(["GAME_DATE", "TEAM_ABBREVIATION"])["MIN_ROLL5"].rank(
        ascending=False, method="dense"
    )
    df["MIN_CONSISTENCY"] = grp["MIN"].transform(lambda x: x.shift(1).expanding().std())
    df["MIN_VOL_RATIO"] = df["MIN_ROLL5_STD"] / df["MIN_ROLL5"].clip(lower=1)
    df["TEAM_MIN_STD"] = df.groupby(["GAME_DATE", "TEAM_ABBREVIATION"])["MIN"].transform("std")
    df["TEAM_DEPTH_ROLL5"] = grp["TEAM_MIN_STD"].transform(lambda x: rolling_mean(x, 5))

    # Role scores
    df["SCORER_SCORE"] = df["PTS_SZN_AVG"] + df["FGA_SZN_AVG"] * 0.5
    df["PLAYMAKER_SCORE"] = df["AST_SZN_AVG"] * 2
    df["DEFENDER_SCORE"] = (df["STL_SZN_AVG"] + df["BLK_SZN_AVG"]) * 2
    df["REBOUNDER_SCORE"] = df["REB_SZN_AVG"]
    df["PF_ROLL5"] = grp["PF"].transform(lambda x: rolling_mean(x, 5))
    df["HIGH_FOUL_RISK"] = (df["PF_ROLL5"] > 3.5).astype(int)
    df["PM_MOMENTUM"] = grp["PLUS_MINUS"].transform(lambda x: rolling_mean(x, 5))
    df["WIN_PCT_SZN"] = grp["WON"].transform(expanding_mean)
    df["PTS_ZSCORE_RECENT"] = (
        (df["PTS_ROLL5"] - df["PTS_SZN_AVG"])
        / grp["PTS"].transform(lambda x: x.shift(1).expanding().std()).clip(lower=0.1)
    )

    # Fatigue / career phase
    print("  Cumulative load (10d)...")
    df["CUMULATIVE_MIN_10D"] = df.groupby("PLAYER_ID", group_keys=False).apply(_cumulative_min_10d)
    df["CAREER_GAMES"] = grp.cumcount()
    df["IS_ROOKIE_PHASE"] = (df["CAREER_GAMES"] < 82).astype(int)
    df["IS_VETERAN_PHASE"] = (df["CAREER_GAMES"] > 400).astype(int)

    print("  Injury return ramp...")
    df["GAMES_SINCE_RETURN"] = df.groupby("PLAYER_ID", group_keys=False).apply(
        _games_since_long_absence
    )
    df["IRONMAN_STREAK"] = grp["MIN"].transform(lambda x: _ironman_streak(x, threshold=30))
    df["USAGE_SPIKE"] = (
        (df["USAGE_RATE_PROXY_ROLL5"] - grp["USAGE_RATE_PROXY"].transform(
            lambda x: x.shift(1).expanding().mean()
        ))
        / grp["USAGE_RATE_PROXY"].transform(
            lambda x: x.shift(1).expanding().std()
        ).clip(lower=0.01)
    )

    # Cross-feature interactions (minutes-model family)
    df["FATIGUE_x_CLOSER"] = df["CUMULATIVE_MIN_10D"] * df["CLOSE_GAME_FLAG"]
    df["SCORER_x_HOME"] = df["SCORER_SCORE"] * df["IS_HOME"]
    df["EFFICIENCY_x_OPP"] = df["EFFICIENCY_PER_MIN"] * df["OPP_STRENGTH"].fillna(0)
    df["ROLL5_MIN_x_REST"] = df["MIN_ROLL5"] * df["DAYS_REST"].fillna(2)
    df["CONSISTENCY_x_RANK"] = df["MIN_CONSISTENCY"] * (1 / df["MIN_RANK_IN_TEAM"].clip(lower=1))
    df["VETERAN_x_REST"] = df["IS_VETERAN_PHASE"] * df["DAYS_REST"].fillna(2)
    df["EWM5_x_OPP_PACE"] = df["MIN_EWM5"] * df["OPP_PACE_PROXY"].fillna(0)
    df["RAMP_x_ROLE"] = df["GAMES_SINCE_RETURN"] * df["MIN_SZN_AVG"]
    df["DEPTH_x_RANK"] = df["TEAM_DEPTH_ROLL5"] * df["MIN_RANK_IN_TEAM"]
    df["VOL_RATIO_x_STREAK"] = df["MIN_VOL_RATIO"] * df["IRONMAN_STREAK"]

    return df


def inject_predicted_minutes(df: pd.DataFrame, avail_min_features: list[str]) -> pd.DataFrame:
    """Load the trained minutes model and add PREDICTED_MIN / PREDICTED_MIN_STD."""
    print("Injecting predicted minutes from minutes model...")

    # Import classes so pickle can deserialise them
    import sys
    import minutes_predictor
    sys.modules['__main__'] = minutes_predictor
    from minutes_predictor import MinutesPredictor, LeagueModel, PlayerModel


    with open(MIN_MODEL_PKL, "rb") as f:
        min_predictor = pickle.load(f)

    X_min_raw = df[avail_min_features].copy().fillna(df[avail_min_features].median(numeric_only=True))
    df["PREDICTED_MIN"] = min_predictor.league_model.predict(X_min_raw)

    try:
        df["PREDICTED_MIN_STD"] = min_predictor.league_model.predict_std(X_min_raw)
    except Exception:
        df["PREDICTED_MIN_STD"] = 5.0

    return df


def add_points_features(df: pd.DataFrame) -> pd.DataFrame:
    """Shot profile, scoring load, PTS rolling stats, and opponent defensive features."""
    print("Shot profile & volume features...")
    grp = df.groupby("PLAYER_ID")

    for col, new_col in [("FGA", "FGA_PER_MIN"), ("FG3A", "FG3A_PER_MIN"), ("FTA", "FTA_PER_MIN")]:
        df[new_col] = df[col] / df["MIN"].clip(lower=0.1)
        for w in [3, 5, 10]:
            df[f"{new_col}_ROLL{w}"] = grp[new_col].transform(lambda x, ww=w: rolling_mean(x, ww))
        df[f"{new_col}_EWM5"] = grp[new_col].transform(
            lambda x: x.shift(1).ewm(span=5, min_periods=2).mean()
        )

    df["THREE_PT_RATE"] = df["FG3A"] / (df["FGA"] + 0.01)
    df["FT_RATE"] = df["FTA"] / (df["FGA"] + 0.01)
    df["MID_RANGE_RATE"] = (1 - df["THREE_PT_RATE"] - df["FT_RATE"].clip(0, 1)).clip(0, 1)
    for col in ["THREE_PT_RATE", "FT_RATE", "MID_RANGE_RATE"]:
        df[f"{col}_ROLL5"] = grp[col].transform(lambda x: rolling_mean(x, 5))
        df[f"{col}_SZN"] = grp[col].transform(expanding_mean)

    for eff in ["FG_PCT", "FG3_PCT", "FT_PCT", "TRUE_SHOOTING"]:
        for w in [3, 5, 10]:
            df[f"{eff}_ROLL{w}"] = grp[eff].transform(lambda x, ww=w: rolling_mean(x, ww, mp=1))
        df[f"{eff}_SZN_AVG"] = grp[eff].transform(expanding_mean)
        df[f"{eff}_EWM5"] = grp[eff].transform(
            lambda x: x.shift(1).ewm(span=5, min_periods=2).mean()
        )

    df["PTS_PER_SHOT"] = df["PTS"] / (df["FGA"] + 0.44 * df["FTA"] + 0.01)
    df["PTS_PER_SHOT_ROLL5"] = grp["PTS_PER_SHOT"].transform(lambda x: rolling_mean(x, 5))

    df["CREATION_PROXY"] = df["FGM"] - df["AST_SZN_AVG"] * 0.15
    df["CREATION_PROXY_ROLL5"] = grp["CREATION_PROXY"].transform(lambda x: rolling_mean(x, 5))

    print("Opponent defensive features...")
    df["OPP_PTS_ALLOWED_AVG"] = df.groupby("OPPONENT")["PTS"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["OPP_FGA_ALLOWED_AVG"] = df.groupby("OPPONENT")["FGA"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["OPP_3PT_ALLOWED_AVG"] = df.groupby("OPPONENT")["FG3A"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["OPP_DEF_INTENSITY"] = (
        df.groupby("OPPONENT")
        .apply(lambda g: (g["STL"] + g["BLK"]).shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    print("Scoring load features...")
    team_total_pts = df.groupby(["GAME_DATE", "TEAM_ABBREVIATION"])["PTS"].transform("sum")
    df["TEAM_PTS_SHARE"] = df["PTS"] / team_total_pts.clip(lower=1)
    df["TEAM_PTS_SHARE_ROLL5"] = grp["TEAM_PTS_SHARE"].transform(lambda x: rolling_mean(x, 5))
    df["PTS_RANK_IN_TEAM"] = df.groupby(["GAME_DATE", "TEAM_ABBREVIATION"])["PTS"].rank(
        ascending=False, method="dense"
    )
    df["PTS_RANK_IN_TEAM_ROLL5"] = grp["PTS_RANK_IN_TEAM"].transform(lambda x: rolling_mean(x, 5))

    print("PTS rolling stats...")
    for w in [3, 5, 10]:
        df[f"PTS_ROLL{w}_STD"] = grp["PTS"].transform(
            lambda x, ww=w: x.shift(1).rolling(ww, min_periods=2).std()
        )
    print("  PTS trend slope...")
    df["PTS_TREND_5G"] = grp["PTS"].transform(lambda x: rolling_slope(x, 5))
    df["PTS_EWM3"] = grp["PTS"].transform(lambda x: x.shift(1).ewm(span=3, min_periods=2).mean())
    df["PTS_EWM5"] = grp["PTS"].transform(lambda x: x.shift(1).ewm(span=5, min_periods=2).mean())
    df["PTS_CV_SZN"] = (
        grp["PTS"].transform(lambda x: x.shift(1).expanding().std())
        / df["PTS_SZN_AVG"].clip(lower=0.5)
    )
    df["HOT_STREAK"] = grp["PTS"].transform(
        lambda x: (x.shift(1) > x.shift(1).expanding().mean()).rolling(3, min_periods=2).sum() >= 2
    ).astype(int)
    df["COLD_STREAK"] = grp["PTS"].transform(
        lambda x: (x.shift(1) < x.shift(1).expanding().mean() * 0.7).rolling(3, min_periods=2).sum() >= 2
    ).astype(int)
    df["PTS_MOMENTUM_3G"] = (
        (df["PTS_ROLL3"] - df["PTS_SZN_AVG"])
        / grp["PTS"].transform(lambda x: x.shift(1).expanding().std()).clip(lower=0.1)
    )

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-product interaction features for the points model."""
    print("Interaction features...")
    df["VOL_x_EFF"] = df["FGA_PER_MIN_ROLL5"] * df["TRUE_SHOOTING_ROLL5"]
    df["VOL_x_HOME"] = df["FGA_PER_MIN_ROLL5"] * df["IS_HOME"]
    df["EFF_x_OPP_PTS"] = df["TRUE_SHOOTING_ROLL5"] * df["OPP_PTS_ALLOWED_AVG"].fillna(0)
    df["PREDMIN_x_VOL"] = df["PREDICTED_MIN"] * df["FGA_PER_MIN_ROLL5"]
    df["PREDMIN_x_EFF"] = df["PREDICTED_MIN"] * df["TRUE_SHOOTING_ROLL5"]
    df["3PT_RATE_x_EFF"] = df["THREE_PT_RATE_ROLL5"] * df["FG3_PCT_ROLL5"]
    df["LOAD_x_FATIGUE"] = df["TEAM_PTS_SHARE_ROLL5"] * df["CUMULATIVE_MIN_10D"]
    df["SCORER_x_OPP"] = df["SCORER_SCORE"] * df["OPP_PTS_ALLOWED_AVG"].fillna(0)
    df["HOT_x_HOME"] = df["HOT_STREAK"] * df["IS_HOME"]
    df["CV_x_CLOSE"] = df["PTS_CV_SZN"] * df["CLOSE_GAME_FLAG"]
    df["EWM_PTS_x_OPP_DEF"] = df["PTS_EWM5"] * df["OPP_DEF_INTENSITY"].fillna(0)
    df["CREATION_x_OPP_PACE"] = df["CREATION_PROXY_ROLL5"] * df["OPP_PACE_PROXY"].fillna(0)
    df["PREDMIN_x_CREATION"] = df["PREDICTED_MIN"] * df["CREATION_PROXY_ROLL5"]
    df["COLD_x_AWAY"] = df["COLD_STREAK"] * (1 - df["IS_HOME"])
    df["FT_RATE_x_FOUL_RISK"] = df["FT_RATE_ROLL5"] * (1 - df["HIGH_FOUL_RISK"])
    return df


FEATURE_COLS_PTS = [
    "PREDICTED_MIN", "PREDICTED_MIN_STD",
    "GAME_NUM_IN_SEASON", "IS_HOME", "DAY_OF_WEEK", "DAYS_REST",
    "IS_BACK_TO_BACK", "BTB_ROAD", "BTB_HOME", "EXTENDED_REST",
    "IS_3_IN_4", "LATE_SEASON", "MONTH", "BLOWOUT_FLAG", "CLOSE_GAME_FLAG",
    "MIN_ROLL3", "MIN_ROLL5", "MIN_ROLL10", "MIN_TREND_5G", "MIN_SZN_AVG", "MIN_CONSISTENCY",
    "MIN_EWM5", "MIN_EWM10", "MIN_ROLL5_MEDIAN", "MIN_VOL_RATIO",
    "PTS_ROLL3", "PTS_ROLL5", "PTS_ROLL10",
    "PTS_ROLL3_STD", "PTS_ROLL5_STD", "PTS_ROLL10_STD",
    "PTS_TREND_5G", "PTS_CV_SZN", "HOT_STREAK", "COLD_STREAK", "PTS_MOMENTUM_3G",
    "PTS_EWM3", "PTS_EWM5",
    "FGA_PER_MIN_ROLL3", "FGA_PER_MIN_ROLL5", "FGA_PER_MIN_ROLL10",
    "FGA_PER_MIN_EWM5",
    "FG3A_PER_MIN_ROLL5", "FTA_PER_MIN_ROLL5",
    "THREE_PT_RATE_ROLL5", "THREE_PT_RATE_SZN", "FT_RATE_ROLL5", "FT_RATE_SZN",
    "MID_RANGE_RATE_ROLL5",
    "FG_PCT_ROLL3", "FG_PCT_ROLL5", "FG_PCT_ROLL10", "FG_PCT_SZN_AVG", "FG_PCT_EWM5",
    "FG3_PCT_ROLL5", "FG3_PCT_ROLL10", "FG3_PCT_SZN_AVG", "FG3_PCT_EWM5",
    "FT_PCT_ROLL5", "FT_PCT_SZN_AVG", "FT_PCT_EWM5",
    "TRUE_SHOOTING_ROLL3", "TRUE_SHOOTING_ROLL5", "TRUE_SHOOTING_ROLL10",
    "TRUE_SHOOTING_SZN_AVG", "TRUE_SHOOTING_EWM5",
    "PTS_PER_SHOT_ROLL5", "CREATION_PROXY_ROLL5",
    "TEAM_PTS_SHARE_ROLL5", "PTS_RANK_IN_TEAM_ROLL5",
    "PTS_SZN_AVG", "FGA_SZN_AVG", "AST_SZN_AVG", "STL_SZN_AVG", "BLK_SZN_AVG",
    "SCORER_SCORE", "PLAYMAKER_SCORE",
    "IS_ROOKIE_PHASE", "IS_VETERAN_PHASE",
    "OPP_PTS_ALLOWED_AVG", "OPP_FGA_ALLOWED_AVG", "OPP_3PT_ALLOWED_AVG",
    "OPP_STRENGTH", "OPP_DEF_INTENSITY",
    "CUMULATIVE_MIN_10D", "PTS_ZSCORE_RECENT", "PM_MOMENTUM",
    "WIN_PCT_SZN", "HIGH_FOUL_RISK", "GAMES_SINCE_RETURN",
    "VOL_x_EFF", "VOL_x_HOME", "EFF_x_OPP_PTS", "PREDMIN_x_VOL", "PREDMIN_x_EFF",
    "3PT_RATE_x_EFF", "LOAD_x_FATIGUE", "SCORER_x_OPP", "HOT_x_HOME", "CV_x_CLOSE",
    "EWM_PTS_x_OPP_DEF", "CREATION_x_OPP_PACE", "PREDMIN_x_CREATION",
    "COLD_x_AWAY", "FT_RATE_x_FOUL_RISK",
    # ── Injury features ──────────────────────────────────────────────────────
    # Own injury status
    "PLAYER_STATUS_SCORE",
    "PLAYER_IS_OUT",
    "PLAYER_IS_QUESTIONABLE",
    "PLAYER_INJURY_RISK_ROLL5",
    "DAYS_SINCE_LAST_INJURY",
    "RETURN_FROM_INJURY_FLAG",
    "INJURY_GAMES_MISSED_RECENT",
    # Teammate redistribution — drives minutes and shot volume changes
    "TEAMMATE_MIN_ABSORBED",       # estimated extra minutes from absent teammates
    "TEAMMATE_FGA_ABSORBED",       # estimated extra FGA from absent teammates
    "TEAM_STARS_OUT",
    "LINEUP_DISRUPTION_SCORE",
    "TEAM_INJURY_SEVERITY",
    "TEAM_INJURY_SEVERITY_ROLL5",
    # Opponent
    "OPP_INJURY_SEVERITY",
    "OPP_STARS_OUT",
    # Composite interactions (from injury_features.py)
    "INJURY_MIN_BOOST",            # absorbed_min × minute-consistency
    "INJURY_PTS_BOOST",            # absorbed_fga × true shooting
    "OPP_INJURY_ADVANTAGE",
    # Points-specific injury interactions (computed below in add_injury_features_pts)
    "INJ_BOOST_x_SCORER",          # teammate_fga_absorbed × scorer role
    "INJ_BOOST_x_PREDMIN",         # teammate_fga_absorbed × predicted minutes
    "OPP_DEPLETED_x_EFF",          # opponent depletion × shooting efficiency
    "RETURN_x_VOL",                # return-from-injury × shot volume (restricted)
]

# columns from minutes_feature_engineering needed for the MIN model injection
MIN_FEATURES = [
    "GAME_NUM_IN_SEASON", "IS_HOME", "DAY_OF_WEEK", "DAYS_REST", "IS_BACK_TO_BACK",
    "BTB_ROAD", "BTB_HOME", "EXTENDED_REST",
    "IS_3_IN_4", "LATE_SEASON", "MONTH", "BLOWOUT_FLAG", "CLOSE_GAME_FLAG",
    "MIN_ROLL3", "MIN_ROLL5", "MIN_ROLL10", "MIN_ROLL3_STD", "MIN_ROLL5_STD",
    "MIN_ROLL10_STD", "MIN_ROLL5_MEDIAN", "MIN_ROLL10_MEDIAN", "MIN_TREND_5G",
    "MIN_EWM5", "MIN_EWM10", "PTS_EWM5", "PTS_EWM10",
    "EFFICIENCY_PER_MIN_EWM5", "USAGE_RATE_PROXY_EWM5",
    "PTS_ROLL3", "PTS_ROLL5", "PTS_ROLL10",
    "FGA_ROLL5", "FG_PCT_ROLL5", "FG3_PCT_ROLL5", "FT_PCT_ROLL5",
    "REB_ROLL5", "AST_ROLL5", "STL_ROLL5", "BLK_ROLL5", "TOV_ROLL5", "PF_ROLL5",
    "PLUS_MINUS_ROLL5", "EFFICIENCY_PER_MIN_ROLL5", "USAGE_RATE_PROXY_ROLL5",
    "TRUE_SHOOTING_ROLL5", "STOCKS_ROLL5", "AST_TO_RATIO_ROLL5",
    "MIN_SZN_AVG", "PTS_SZN_AVG", "FGA_SZN_AVG", "REB_SZN_AVG",
    "AST_SZN_AVG", "STL_SZN_AVG", "BLK_SZN_AVG",
    "SCORER_SCORE", "PLAYMAKER_SCORE", "DEFENDER_SCORE", "REBOUNDER_SCORE",
    "MIN_CONSISTENCY", "MIN_VOL_RATIO",
    "TEAM_MIN_SHARE_ROLL5", "MIN_RANK_IN_TEAM", "TEAM_DEPTH_ROLL5",
    "OPP_AVG_MIN_ALLOWED", "OPP_STRENGTH", "OPP_PACE_PROXY", "H2H_MIN_VS_OPP",
    "CUMULATIVE_MIN_10D", "GAMES_SINCE_RETURN", "IRONMAN_STREAK",
    "IS_ROOKIE_PHASE", "IS_VETERAN_PHASE",
    "PTS_ZSCORE_RECENT", "HIGH_FOUL_RISK", "PM_MOMENTUM", "WIN_PCT_SZN", "USAGE_SPIKE",
    "FATIGUE_x_CLOSER", "SCORER_x_HOME", "EFFICIENCY_x_OPP",
    "ROLL5_MIN_x_REST", "CONSISTENCY_x_RANK",
    "VETERAN_x_REST", "EWM5_x_OPP_PACE", "RAMP_x_ROLE",
    "DEPTH_x_RANK", "VOL_RATIO_x_STREAK",
    # Injury features passed through so predicted minutes is injury-aware
    "PLAYER_STATUS_SCORE", "PLAYER_IS_OUT", "PLAYER_IS_QUESTIONABLE",
    "PLAYER_INJURY_RISK_ROLL5", "DAYS_SINCE_LAST_INJURY",
    "RETURN_FROM_INJURY_FLAG", "INJURY_GAMES_MISSED_RECENT",
    "TEAMMATE_MIN_ABSORBED", "TEAMMATE_FGA_ABSORBED",
    "TEAM_INJURY_SEVERITY", "TEAM_STARS_OUT", "LINEUP_DISRUPTION_SCORE",
    "TEAM_INJURY_SEVERITY_ROLL5",
    "OPP_INJURY_SEVERITY", "OPP_STARS_OUT",
    "INJURY_MIN_BOOST", "INJURY_PTS_BOOST", "OPP_INJURY_ADVANTAGE",
]


def add_injury_features_pts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge injury features and add points-specific injury interaction terms.
    Reads data/injury_data.csv — populated by running data_pipeline.py.
    """
    print("Engineering injury features (points pipeline)...")
    inj_features = build_injury_features(player_game_df=df, verbose=True)
    df = merge_injury_features_into_df(df, inj_features)

    # Points-specific interaction terms — built on top of base injury features
    # INJ_BOOST_x_SCORER: a primary scorer benefits more from extra shot attempts
    df["INJ_BOOST_x_SCORER"] = (
        df["TEAMMATE_FGA_ABSORBED"].fillna(0)
        * df.get("SCORER_SCORE", pd.Series(0, index=df.index)).fillna(0)
    )
    # INJ_BOOST_x_PREDMIN: extra FGA opportunity is larger when the player is already
    # projected to play more minutes
    if "PREDICTED_MIN" in df.columns:
        df["INJ_BOOST_x_PREDMIN"] = df["TEAMMATE_FGA_ABSORBED"].fillna(0) * df["PREDICTED_MIN"].fillna(0)
    else:
        df["INJ_BOOST_x_PREDMIN"] = 0.0

    # OPP_DEPLETED_x_EFF: efficient scorers extract more from a depleted defense
    df["OPP_DEPLETED_x_EFF"] = (
        df["OPP_INJURY_ADVANTAGE"].fillna(0)
        * df.get("TRUE_SHOOTING_ROLL5", pd.Series(0, index=df.index)).fillna(0)
    )
    # RETURN_x_VOL: players returning from injury often have restricted shot volume
    df["RETURN_x_VOL"] = (
        df["RETURN_FROM_INJURY_FLAG"].fillna(0)
        * df.get("FGA_PER_MIN_ROLL5", pd.Series(0, index=df.index)).fillna(0)
    )

    print(f"  ✓ {len(_all_injury_feature_cols()) + 4} injury-related features added to points pipeline")
    return df


def assemble_and_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Select features, apply validity mask, fill NAs, and return X, y, player_names."""
    print("Assembling feature matrix...")
    available = [c for c in FEATURE_COLS_PTS if c in df.columns]
    X = df[available].copy()
    y = df["PTS"].copy()
    player_names = df["PLAYER_NAME"].copy()

    valid_mask = (
        X["PTS_ROLL5"].notna()
        & X["FGA_PER_MIN_ROLL5"].notna()
        & X["TRUE_SHOOTING_ROLL5"].notna()
    )
    X            = X[valid_mask].reset_index(drop=True)
    y            = y[valid_mask].reset_index(drop=True)
    player_names = player_names[valid_mask].reset_index(drop=True)

    # Zero-fill injury cols first (missing report dates should be 0, not median)
    injury_cols = [c for c in _all_injury_feature_cols() if c in X.columns]
    X[injury_cols] = X[injury_cols].fillna(0.0)
    # Fill any remaining NaNs in rolling stats with column medians
    X = X.fillna(X.median(numeric_only=True))

    print(f"\n✓ X shape   : {X.shape}")
    print(f"✓ Players   : {player_names.nunique()}")
    print(f"✓ Target PTS: mean={y.mean():.1f}  std={y.std():.1f}")
    print(f"✓ Injury features: {len(injury_cols)}/{len(_all_injury_feature_cols())} present")
    return X, y, player_names


def save_outputs(X: pd.DataFrame, y: pd.Series, player_names: pd.Series) -> None:
    X.to_csv(X_FEATURES_PTS, index=False)
    y.to_csv(Y_TARGET_PTS, index=False)
    player_names.to_csv(PTS_PLAYER_NAMES, index=False)

    # Embed PLAYER_NAME in the combined CSV so points_predictor.py can read it
    # directly without re-deriving it (same pattern as minutes pipeline).
    combined = X.copy()
    combined["PTS_TARGET"]   = y
    combined["PLAYER_NAME"]  = player_names.values
    combined.to_csv(ML_DATASET_PTS, index=False)
    print(f"\n✓ Saved to {ML_DATASET_PTS}  ({len(combined):,} rows, {player_names.nunique()} players)")


def main() -> None:
    df = load_data()
    df = add_basic_flags(df)

    # Reads data/injury_data.csv written by data_pipeline.py.
    # Degrades gracefully to zero-filled columns if the file isn't present.
    try:
        df = add_injury_features_pts(df)
    except Exception as e:
        print(f"  [injury] Skipped injury features: {e}")
        
    df = add_minutes_passthrough_features(df)

    avail_min_features = [c for c in MIN_FEATURES if c in df.columns]
    df = inject_predicted_minutes(df, avail_min_features)

    df = add_points_features(df)
    df = add_interaction_features(df)

    X, y, player_names = assemble_and_filter(df)
    save_outputs(X, y, player_names)


if __name__ == "__main__":
    main()