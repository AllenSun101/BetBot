"""
minutes_predictor.py — Two-Stage NBA Minutes Predictor (v3)

Stage 1 : League-wide stacking ensemble  (HGBR + LightGBM → Ridge)
Stage 2 : Player-level BayesianRidge on residuals

All shared machinery lives in base_predictor.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from base_predictor import BaseLeagueModel, BasePlayerModel, BasePredictor
from config import DATA_PATH, ML_DATASET_MIN, MIN_MODEL_PKL, MIN_FI_CSV, WALKFORWARD_MIN

# ── Feature list ───────────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Context
    "GAME_NUM_IN_SEASON", "IS_HOME", "DAY_OF_WEEK", "DAYS_REST",
    "IS_BACK_TO_BACK", "BTB_ROAD", "BTB_HOME", "EXTENDED_REST",
    "IS_3_IN_4", "LATE_SEASON", "MONTH", "BLOWOUT_FLAG", "CLOSE_GAME_FLAG",
    # Rolling means
    "MIN_ROLL3", "MIN_ROLL5", "MIN_ROLL10",
    "MIN_ROLL3_STD", "MIN_ROLL5_STD", "MIN_ROLL10_STD",
    "MIN_ROLL5_MEDIAN", "MIN_ROLL10_MEDIAN", "MIN_TREND_5G",
    # EWMA
    "MIN_EWM5", "MIN_EWM10", "PTS_EWM5", "PTS_EWM10",
    "EFFICIENCY_PER_MIN_EWM5", "USAGE_RATE_PROXY_EWM5",
    # Scoring / shooting rolls
    "PTS_ROLL3", "PTS_ROLL5", "PTS_ROLL10",
    "FGA_ROLL5", "FG_PCT_ROLL5", "FG3_PCT_ROLL5", "FT_PCT_ROLL5",
    "REB_ROLL5", "AST_ROLL5", "STL_ROLL5", "BLK_ROLL5",
    "TOV_ROLL5", "PF_ROLL5", "PLUS_MINUS_ROLL5",
    "EFFICIENCY_PER_MIN_ROLL5", "USAGE_RATE_PROXY_ROLL5",
    "TRUE_SHOOTING_ROLL5", "STOCKS_ROLL5", "AST_TO_RATIO_ROLL5",
    # Season averages
    "MIN_SZN_AVG", "PTS_SZN_AVG", "FGA_SZN_AVG", "REB_SZN_AVG",
    "AST_SZN_AVG", "STL_SZN_AVG", "BLK_SZN_AVG",
    # Role scores
    "SCORER_SCORE", "PLAYMAKER_SCORE", "DEFENDER_SCORE", "REBOUNDER_SCORE",
    "MIN_CONSISTENCY", "MIN_VOL_RATIO",
    # Team context
    "TEAM_MIN_SHARE_ROLL5", "MIN_RANK_IN_TEAM", "TEAM_DEPTH_ROLL5",
    # Opponent
    "OPP_AVG_MIN_ALLOWED", "OPP_STRENGTH", "OPP_PACE_PROXY", "H2H_MIN_VS_OPP",
    # Load / fatigue
    "CUMULATIVE_MIN_10D", "GAMES_SINCE_RETURN", "IRONMAN_STREAK",
    # Career phase
    "IS_ROOKIE_PHASE", "IS_VETERAN_PHASE",
    # Momentum / volatility
    "PTS_ZSCORE_RECENT", "HIGH_FOUL_RISK", "PM_MOMENTUM", "WIN_PCT_SZN", "USAGE_SPIKE",
    # Interactions
    "FATIGUE_x_CLOSER", "SCORER_x_HOME", "EFFICIENCY_x_OPP",
    "ROLL5_MIN_x_REST", "CONSISTENCY_x_RANK",
    "VETERAN_x_REST", "EWM5_x_OPP_PACE", "RAMP_x_ROLE",
    "DEPTH_x_RANK", "VOL_RATIO_x_STREAK",
    # ── Injury features ──────────────────────────────────────────────────────
    # Own injury status
    "PLAYER_STATUS_SCORE",
    "PLAYER_IS_OUT",
    "PLAYER_IS_QUESTIONABLE",
    "PLAYER_INJURY_RISK_ROLL5",
    "DAYS_SINCE_LAST_INJURY",
    "RETURN_FROM_INJURY_FLAG",
    "INJURY_GAMES_MISSED_RECENT",
    # Teammate redistribution
    "TEAMMATE_MIN_ABSORBED",
    "TEAMMATE_FGA_ABSORBED",
    "TEAM_STARS_OUT",
    "LINEUP_DISRUPTION_SCORE",
    "TEAM_INJURY_SEVERITY",
    "TEAM_INJURY_SEVERITY_ROLL5",
    # Opponent
    "OPP_INJURY_SEVERITY",
    "OPP_STARS_OUT",
    # Composite interactions
    "INJURY_MIN_BOOST",
    "INJURY_PTS_BOOST",
    "OPP_INJURY_ADVANTAGE",
]


# ── Domain-specific subclasses ─────────────────────────────────────────────────

class LeagueModel(BaseLeagueModel):
    FEATURE_COLS = FEATURE_COLS
    hgb_params = dict(
        max_iter=400, learning_rate=0.05, max_depth=6,
        min_samples_leaf=20, l2_regularization=0.1, random_state=42,
    )
    lgb_params = dict(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        num_leaves=50, min_child_samples=20, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    uncertainty_clip = (1.5, 12.0)

    def fit(self, X: pd.DataFrame, y: pd.Series, tscv_splits: int = 5) -> "LeagueModel":
        print("─── Stage 1: Training League Minutes Model ───")
        return super().fit(X, y, tscv_splits)


class PlayerModel(BasePlayerModel):
    default_residual_std = 5.0
    default_residual_clip = 10.0


class MinutesPredictor(BasePredictor):
    LeagueModelClass = LeagueModel
    PlayerModelClass = PlayerModel
    clip_max = 48.0
    target_col = "minutes"

    def save(self, path: Path = None) -> None:  # type: ignore[override]
        super().save(path or MIN_MODEL_PKL)

    @classmethod
    def load(cls, path: Path = None) -> "MinutesPredictor":  # type: ignore[override]
        return super().load(path or MIN_MODEL_PKL)


# ── Entry point ────────────────────────────────────────────────────────────────

def _compute_feature_importances(predictor: MinutesPredictor, X: pd.DataFrame, y: pd.Series):
    try:
        import shap
        cols = predictor._cols
        idx = np.random.choice(len(X), size=min(2000, len(X)), replace=False)
        explainer = shap.TreeExplainer(predictor.league_model.hgb)
        sv = explainer.shap_values(X.iloc[idx][cols])
        fi = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)
        print("\n── Top 15 SHAP feature importances ──")
    except Exception:
        from sklearn.inspection import permutation_importance

        cols = predictor._cols
        idx = np.random.choice(len(X), size=min(3000, len(X)), replace=False)
        perm = permutation_importance(
            predictor.league_model.hgb,
            X.iloc[idx][cols], y.iloc[idx],
            n_repeats=5, random_state=42, n_jobs=-1,
        )
        fi = pd.Series(perm.importances_mean, index=cols).sort_values(ascending=False)
        print("\n── Top 15 permutation feature importances ──")

    print(fi.head(15).to_string())
    fi.to_csv(MIN_FI_CSV)


if __name__ == "__main__":
    df = pd.read_csv(ML_DATASET_MIN)

    # PLAYER_NAME is saved into ml_dataset_minutes.csv by minutes_feature_engineering.py.
    # Reading it here guarantees exact row alignment with X — no recomputation needed.
    if "PLAYER_NAME" not in df.columns:
        raise ValueError(
            "PLAYER_NAME column not found in ml_dataset_minutes.csv.\n"
            "Re-run minutes_feature_engineering.py to regenerate the dataset."
        )

    player_names = df["PLAYER_NAME"].copy()
    X = df.drop(columns=["MIN_TARGET", "PLAYER_NAME"])
    y = df["MIN_TARGET"]

    print(f"Dataset: {X.shape[0]:,} rows | {X.shape[1]} features | {player_names.nunique()} players\n")
    predictor = MinutesPredictor()
    predictor.fit_league(X, y)
    predictor.fit_all_players(X, y, player_names, min_games=10)

    demo = "De'Aaron Fox"
    eval_results = predictor.evaluate_player(demo, X, y, player_names, holdout_last_n=10)
    if eval_results is not None:
        eval_results.to_csv(WALKFORWARD_MIN, index=False)

    predictor.evaluate_global_walkforward(X, y, player_names, n_splits=5)
    _compute_feature_importances(predictor, X, y)
    predictor.save()
    print(f"\n✓ All outputs saved to {MIN_MODEL_PKL.parent}")