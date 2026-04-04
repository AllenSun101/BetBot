"""
points_predictor.py — Two-Stage NBA Points Predictor (v3)

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
from config import ML_DATASET_PTS, PTS_PLAYER_NAMES, PTS_MODEL_PKL, PTS_FI_CSV

# ── Feature list ───────────────────────────────────────────────────────────────

FEATURE_COLS_PTS = [
    # Minutes prediction (injected from Stage 1)
    "PREDICTED_MIN", "PREDICTED_MIN_STD",
    # Context
    "GAME_NUM_IN_SEASON", "IS_HOME", "DAY_OF_WEEK", "DAYS_REST",
    "IS_BACK_TO_BACK", "BTB_ROAD", "BTB_HOME", "EXTENDED_REST",
    "IS_3_IN_4", "LATE_SEASON", "MONTH", "BLOWOUT_FLAG", "CLOSE_GAME_FLAG",
    # Minutes features
    "MIN_ROLL3", "MIN_ROLL5", "MIN_ROLL10", "MIN_TREND_5G",
    "MIN_SZN_AVG", "MIN_CONSISTENCY", "MIN_EWM5", "MIN_EWM10",
    "MIN_ROLL5_MEDIAN", "MIN_VOL_RATIO",
    # PTS rolling
    "PTS_ROLL3", "PTS_ROLL5", "PTS_ROLL10",
    "PTS_ROLL3_STD", "PTS_ROLL5_STD", "PTS_ROLL10_STD",
    "PTS_TREND_5G", "PTS_CV_SZN", "HOT_STREAK", "COLD_STREAK", "PTS_MOMENTUM_3G",
    "PTS_EWM3", "PTS_EWM5",
    # Shot profile / volume
    "FGA_PER_MIN_ROLL3", "FGA_PER_MIN_ROLL5", "FGA_PER_MIN_ROLL10",
    "FGA_PER_MIN_EWM5",
    "FG3A_PER_MIN_ROLL5", "FTA_PER_MIN_ROLL5",
    "THREE_PT_RATE_ROLL5", "THREE_PT_RATE_SZN", "FT_RATE_ROLL5", "FT_RATE_SZN",
    "MID_RANGE_RATE_ROLL5",
    # Shooting efficiency
    "FG_PCT_ROLL3", "FG_PCT_ROLL5", "FG_PCT_ROLL10", "FG_PCT_SZN_AVG", "FG_PCT_EWM5",
    "FG3_PCT_ROLL5", "FG3_PCT_ROLL10", "FG3_PCT_SZN_AVG", "FG3_PCT_EWM5",
    "FT_PCT_ROLL5", "FT_PCT_SZN_AVG", "FT_PCT_EWM5",
    "TRUE_SHOOTING_ROLL3", "TRUE_SHOOTING_ROLL5", "TRUE_SHOOTING_ROLL10",
    "TRUE_SHOOTING_SZN_AVG", "TRUE_SHOOTING_EWM5",
    # Scoring quality
    "PTS_PER_SHOT_ROLL5", "CREATION_PROXY_ROLL5",
    # Team scoring context
    "TEAM_PTS_SHARE_ROLL5", "PTS_RANK_IN_TEAM_ROLL5",
    # Season averages
    "PTS_SZN_AVG", "FGA_SZN_AVG", "AST_SZN_AVG", "STL_SZN_AVG", "BLK_SZN_AVG",
    # Role
    "SCORER_SCORE", "PLAYMAKER_SCORE",
    # Career phase
    "IS_ROOKIE_PHASE", "IS_VETERAN_PHASE",
    # Opponent context
    "OPP_PTS_ALLOWED_AVG", "OPP_FGA_ALLOWED_AVG", "OPP_3PT_ALLOWED_AVG",
    "OPP_STRENGTH", "OPP_DEF_INTENSITY",
    # Load / fatigue / momentum
    "CUMULATIVE_MIN_10D", "PTS_ZSCORE_RECENT", "PM_MOMENTUM",
    "WIN_PCT_SZN", "HIGH_FOUL_RISK", "GAMES_SINCE_RETURN",
    # Interactions (base)
    "VOL_x_EFF", "VOL_x_HOME", "EFF_x_OPP_PTS",
    "PREDMIN_x_VOL", "PREDMIN_x_EFF",
    "3PT_RATE_x_EFF", "LOAD_x_FATIGUE", "SCORER_x_OPP",
    "HOT_x_HOME", "CV_x_CLOSE",
    # Interactions (extended)
    "EWM_PTS_x_OPP_DEF", "CREATION_x_OPP_PACE",
    "PREDMIN_x_CREATION", "COLD_x_AWAY", "FT_RATE_x_FOUL_RISK",
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
    # Composite interactions (from injury_features.py)
    "INJURY_MIN_BOOST",
    "INJURY_PTS_BOOST",
    "OPP_INJURY_ADVANTAGE",
    # Points-specific interactions (from add_injury_features_pts)
    "INJ_BOOST_x_SCORER",       # teammate_fga_absorbed × scorer role
    "INJ_BOOST_x_PREDMIN",      # teammate_fga_absorbed × predicted minutes
    "OPP_DEPLETED_x_EFF",       # opponent depletion × shooting efficiency
    "RETURN_x_VOL",             # return-from-injury × shot volume
]


# ── Domain-specific subclasses ─────────────────────────────────────────────────

class LeaguePointsModel(BaseLeagueModel):
    FEATURE_COLS = FEATURE_COLS_PTS
    hgb_params = dict(
        max_iter=500, learning_rate=0.04, max_depth=6,
        min_samples_leaf=25, l2_regularization=0.15, random_state=42,
    )
    lgb_params = dict(
        n_estimators=500, learning_rate=0.04, max_depth=6,
        num_leaves=50, min_child_samples=25, reg_lambda=0.15,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    uncertainty_clip = (2.0, 15.0)

    def fit(self, X: pd.DataFrame, y: pd.Series, tscv_splits: int = 5) -> "LeaguePointsModel":
        print("─── Stage 1: Training League Points Model ───")
        return super().fit(X, y, tscv_splits)


class PlayerPointsModel(BasePlayerModel):
    default_residual_std = 6.0
    default_residual_clip = 12.0


class PointsPredictor(BasePredictor):
    LeagueModelClass = LeaguePointsModel
    PlayerModelClass = PlayerPointsModel
    clip_max = 100.0
    target_col = "points"

    def save(self, path: Path = None) -> None:  # type: ignore[override]
        super().save(path or PTS_MODEL_PKL)

    @classmethod
    def load(cls, path: Path = None) -> "PointsPredictor":  # type: ignore[override]
        return super().load(path or PTS_MODEL_PKL)


# ── Entry point ────────────────────────────────────────────────────────────────

def _compute_feature_importances(predictor: PointsPredictor, X: pd.DataFrame, y: pd.Series):
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
    fi.to_csv(PTS_FI_CSV)


def _demo_player_walkforward(
    predictor: PointsPredictor,
    X: pd.DataFrame,
    y: pd.Series,
    player_names: pd.Series,
    holdout: int = 10,
):
    for demo in ["De'Aaron Fox", "Nikola Jokić", "Stephen Curry"]:
        try:
            mask = player_names == demo
            X_p = X[mask].reset_index(drop=True)
            y_p = y[mask].reset_index(drop=True)
            n = len(X_p)
            if n <= holdout + 5:
                continue
            tr, te = slice(0, n - holdout), slice(n - holdout, n)
            lp_test = predictor.league_model.predict(X_p.iloc[te])
            pm_eval = PlayerPointsModel(demo)
            pm_eval.fit(
                X_p.iloc[tr], y_p.iloc[tr],
                predictor.league_model.predict(X_p.iloc[tr]),
                predictor._cols,
            )
            final_test = np.clip(lp_test + pm_eval.predict_residual(X_p.iloc[te]), 0, 100)
            y_test = y_p.iloc[te].values
            print(f"\n── {demo} (last {holdout} games) ──")
            print(f"  League MAE   : {np.abs(lp_test - y_test).mean():.2f}")
            print(f"  Two-stage MAE: {np.abs(final_test - y_test).mean():.2f}")
        except Exception as exc:
            print(f"⚠ {demo}: {exc}")


if __name__ == "__main__":
    df = pd.read_csv(ML_DATASET_PTS)

    # PLAYER_NAME is saved into ml_dataset_points.csv by points_feature_engineering.py.
    # Reading it here guarantees exact row alignment with X — no separate file needed.
    if "PLAYER_NAME" not in df.columns:
        raise ValueError(
            "PLAYER_NAME column not found in ml_dataset_points.csv.\n"
            "Re-run points_feature_engineering.py to regenerate the dataset."
        )

    player_names = df["PLAYER_NAME"].copy()
    X = df.drop(columns=["PTS_TARGET", "PLAYER_NAME"])
    y = df["PTS_TARGET"]

    print(f"Dataset: {X.shape[0]:,} rows | {X.shape[1]} features | {player_names.nunique()} players\n")
    predictor = PointsPredictor()
    predictor.fit_league(X, y)
    predictor.fit_all_players(X, y, player_names, min_games=10)

    _demo_player_walkforward(predictor, X, y, player_names)
    predictor.evaluate_global_walkforward(X, y, player_names, n_splits=5)
    _compute_feature_importances(predictor, X, y)
    predictor.save()
    print(f"\n✓ All outputs saved to {PTS_MODEL_PKL.parent}")