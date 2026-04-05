"""
base_predictor.py — Shared base classes for the NBA Props Engine.

Contains:
  - QuantileUncertaintyModel   (used by both minutes & points league models)
  - BaseLeagueModel            (stacking ensemble: HGBR + optional LightGBM → Ridge)
  - BasePlayerModel            (BayesianRidge on league residuals)
  - BasePredictor              (orchestrates league + player models)
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_cols(feature_cols: list[str], X: pd.DataFrame) -> list[str]:
    """Return only the feature columns that exist in X (graceful degradation)."""
    return [c for c in feature_cols if c in X.columns]


def apply_two_stage(
    predictor: "BasePredictor",
    X: pd.DataFrame,
    player_names: pd.Series,
    clip_max: float,
) -> np.ndarray:
    """
    Apply league + player-level corrections to produce final predictions.
    Shared by evaluate_model.py, generate_preds.py, and walkforward loops.
    """
    lp = predictor.league_model.predict(X)
    fp = lp.copy()
    for name in player_names.unique():
        m = player_names == name
        if name in predictor.player_models:
            delta = predictor.player_models[name].predict_residual(X[m])
            fp[m.values] = np.clip(lp[m.values] + delta, 0, clip_max)
    return fp


# ── Uncertainty ────────────────────────────────────────────────────────────────

class QuantileUncertaintyModel:
    """
    Trains upper/lower quantile HGBR regressors to produce prediction intervals.
    std ≈ PI-width / 2.56  (80% PI).
    """

    def __init__(
        self,
        quantile_lo: float = 0.1,
        quantile_hi: float = 0.9,
        std_clip_lo: float = 1.5,
        std_clip_hi: float = 12.0,
    ):
        self.std_clip_lo = std_clip_lo
        self.std_clip_hi = std_clip_hi
        hgb_kw = dict(max_iter=200, learning_rate=0.05, max_depth=4, random_state=0)
        self.lo = HistGradientBoostingRegressor(loss="quantile", quantile=quantile_lo, **hgb_kw)
        self.hi = HistGradientBoostingRegressor(loss="quantile", quantile=quantile_hi, **hgb_kw)
        self.cols_: list[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, cols: list[str]) -> "QuantileUncertaintyModel":
        self.cols_ = cols
        self.lo.fit(X[cols], y)
        self.hi.fit(X[cols], y)
        self.is_fitted = True
        return self

    def predict_std(self, X: pd.DataFrame) -> np.ndarray:
        """Approximate σ from 80% PI width."""
        width = self.hi.predict(X[self.cols_]) - self.lo.predict(X[self.cols_])
        return np.clip(width / 2.56, self.std_clip_lo, self.std_clip_hi)


# ── League model ───────────────────────────────────────────────────────────────

class BaseLeagueModel:
    """
    Stacking ensemble: HistGradientBoosting + optional LightGBM → Ridge meta-learner.
    Uses TimeSeriesSplit OOF to build the meta-features without future leakage.

    Subclasses set:
        FEATURE_COLS   list[str]  — ordered feature list for the domain
        hgb_params     dict       — kwargs for HistGradientBoostingRegressor
        lgb_params     dict       — kwargs for LGBMRegressor
        uncertainty_clip (lo, hi) — std clipping for QuantileUncertaintyModel
    """

    FEATURE_COLS: list[str] = []
    hgb_params: dict = dict(
        max_iter=400, learning_rate=0.05, max_depth=6,
        min_samples_leaf=20, l2_regularization=0.1, random_state=42,
    )
    lgb_params: dict = dict(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        num_leaves=50, min_child_samples=20, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    uncertainty_clip: tuple[float, float] = (1.5, 12.0)

    def __init__(self, use_lgbm: bool = True):
        self.use_lgbm = use_lgbm
        self.hgb = HistGradientBoostingRegressor(**self.hgb_params)
        self._lgbm_available = False
        if use_lgbm:
            try:
                import lightgbm as lgb  # noqa: F401
                self.lgb_model = lgb.LGBMRegressor(**self.lgb_params)
                self._lgbm_available = True
            except ImportError:
                print("  LightGBM not available — using HGB only.")
        self.meta = Ridge(alpha=1.0)
        self.uncertainty = QuantileUncertaintyModel(
            quantile_lo=0.1,
            quantile_hi=0.9,
            std_clip_lo=self.uncertainty_clip[0],
            std_clip_hi=self.uncertainty_clip[1],
        )
        self.cols_: list[str] = []
        self.is_fitted = False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_oof(self, Xf: pd.DataFrame, y: pd.Series, tscv_splits: int) -> np.ndarray:
        """Build out-of-fold predictions for HGB (and optionally LGB)."""
        tscv = TimeSeriesSplit(n_splits=tscv_splits)

        self.hgb.fit(Xf, y)
        hgb_oof = np.zeros(len(y))
        for tr_idx, val_idx in tscv.split(Xf):
            m = HistGradientBoostingRegressor(**self.hgb_params)
            m.fit(Xf.iloc[tr_idx], y.iloc[tr_idx])
            hgb_oof[val_idx] = m.predict(Xf.iloc[val_idx])

        if not self._lgbm_available:
            return hgb_oof.reshape(-1, 1)

        import lightgbm as lgb
        self.lgb_model.fit(Xf, y)
        lgb_oof = np.zeros(len(y))
        for tr_idx, val_idx in tscv.split(Xf):
            m = lgb.LGBMRegressor(**self.lgb_params)
            m.fit(Xf.iloc[tr_idx], y.iloc[tr_idx])
            lgb_oof[val_idx] = m.predict(Xf.iloc[val_idx])

        return np.column_stack([hgb_oof, lgb_oof])

    def _stack_predict(self, X: pd.DataFrame) -> np.ndarray:
        Xf = X[self.cols_]
        preds = [self.hgb.predict(Xf)]
        if self._lgbm_available:
            preds.append(self.lgb_model.predict(Xf))
        return self.meta.predict(np.column_stack(preds))

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series, tscv_splits: int = 5) -> "BaseLeagueModel":
        self.cols_ = safe_cols(self.FEATURE_COLS, X)
        Xf = X[self.cols_]

        oof_stack = self._build_oof(Xf, y, tscv_splits)
        self.meta.fit(oof_stack, y)

        if self._lgbm_available:
            print(
                f"  Stacking weights — HGB: {self.meta.coef_[0]:.3f}"
                f"  LGB: {self.meta.coef_[1]:.3f}"
            )

        oof_pred = self.meta.predict(oof_stack)
        oof_mae = mean_absolute_error(y, oof_pred)
        oof_r2 = 1 - np.sum((y - oof_pred) ** 2) / np.sum((y - y.mean()) ** 2)
        print(f"  OOF MAE : {oof_mae:.3f}")
        print(f"  OOF R²  : {oof_r2:.3f}")

        print("  Training quantile uncertainty model...")
        self.uncertainty.fit(X, y, self.cols_)

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._stack_predict(X)

    def predict_std(self, X: pd.DataFrame) -> np.ndarray:
        return self.uncertainty.predict_std(X)


# ── Player model ───────────────────────────────────────────────────────────────

class BasePlayerModel:
    """
    BayesianRidge model trained on a single player's residuals
    relative to the league model prediction.

    Subclasses may override defaults for residual_std_ / residual_clip_.
    """

    default_residual_std: float = 5.0
    default_residual_clip: float = 10.0

    def __init__(self, player_name: str):
        self.player_name = player_name
        self.model = BayesianRidge(max_iter=500, tol=1e-4)
        self.scaler = StandardScaler()
        self.n_games: int = 0
        self.is_fitted = False
        self.train_mae: Optional[float] = None
        self.alpha_: Optional[float] = None
        self.lambda_: Optional[float] = None
        self.residual_std_ = self.default_residual_std
        self.residual_clip_ = self.default_residual_clip
        self.cols: list[str] = []

    def fit(
        self,
        X_p: pd.DataFrame,
        y_p: pd.Series,
        league_preds: np.ndarray,
        cols: list[str],
    ) -> "BasePlayerModel":
        self.cols = cols
        self.n_games = len(X_p)
        residuals = y_p.values - league_preds

        if len(residuals) > 5:
            self.residual_std_ = float(np.std(residuals))
            self.residual_clip_ = float(np.percentile(np.abs(residuals), 95))
        else:
            self.residual_std_ = self.default_residual_std
            self.residual_clip_ = self.default_residual_clip

        Xs = self.scaler.fit_transform(X_p[cols])
        self.model.fit(Xs, residuals)
        self.train_mae = mean_absolute_error(y_p, league_preds + self.model.predict(Xs))
        self.alpha_ = self.model.alpha_
        self.lambda_ = self.model.lambda_
        self.is_fitted = True
        return self

    def predict_residual(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict(self.scaler.transform(X[self.cols]))
        return np.clip(raw, -self.residual_clip_, self.residual_clip_)

    def predict_residual_std(self, X: pd.DataFrame) -> np.ndarray:
        _, std = self.model.predict(self.scaler.transform(X[self.cols]), return_std=True)
        return np.maximum(std, 1.0)


# ── Base predictor ─────────────────────────────────────────────────────────────

class BasePredictor:
    """
    Orchestrates a two-stage prediction pipeline:
      Stage 1 — BaseLeagueModel  (full population)
      Stage 2 — BasePlayerModel  (per-player residual correction)

    Subclasses must set:
        LeagueModelClass   : type[BaseLeagueModel]
        PlayerModelClass   : type[BasePlayerModel]
        clip_max           : float   (max valid prediction, e.g. 48 or 100)
        target_col         : str     (name used in print messages, e.g. "minutes")
    """

    LeagueModelClass = BaseLeagueModel
    PlayerModelClass = BasePlayerModel
    clip_max: float = 100.0
    target_col: str = "target"

    def __init__(self):
        self.league_model = self.LeagueModelClass()
        self.player_models: dict[str, BasePlayerModel] = {}
        self._cols: list[str] = []

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit_league(self, X: pd.DataFrame, y: pd.Series) -> "BasePredictor":
        self.league_model.fit(X, y)
        self._cols = self.league_model.cols_
        return self

    def fit_all_players(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        player_names: pd.Series,
        min_games: int = 10,
    ) -> "BasePredictor":
        counts = player_names.value_counts()
        eligible = counts[counts >= min_games].index.tolist()
        print(f"\nFitting {len(eligible)} player models (≥{min_games} games)...")
        for name in eligible:
            try:
                mask = player_names == name
                X_p = X[mask].reset_index(drop=True)
                y_p = y[mask].reset_index(drop=True)
                lp = self.league_model.predict(X_p)
                pm = self.PlayerModelClass(name)
                pm.fit(X_p, y_p, lp, self._cols)
                self.player_models[name] = pm
            except Exception as exc:
                print(f"  ⚠ Skipped {name}: {exc}")
        print(f"✓ {len(self.player_models)} player models fitted")
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        player_name: str,
        X_new: pd.DataFrame,
        return_uncertainty: bool = False,
    ) -> dict:
        lp = self.league_model.predict(X_new)
        league_std = self.league_model.predict_std(X_new)

        if player_name not in self.player_models:
            result = {
                "league_pred": lp,
                "player_delta": np.zeros_like(lp),
                "final_pred": np.clip(lp, 0, self.clip_max),
                "n_games_seen": 0,
            }
            if return_uncertainty:
                result["std"] = league_std
            return result

        pm = self.player_models[player_name]
        delta = pm.predict_residual(X_new)
        final = np.clip(lp + delta, 0, self.clip_max)
        result = {
            "league_pred": lp,
            "player_delta": delta,
            "final_pred": final,
            "n_games_seen": pm.n_games,
        }
        if return_uncertainty:
            player_std = pm.predict_residual_std(X_new)
            result["std"] = np.sqrt(league_std**2 + player_std**2)
        return result

    def predict_all(
        self, X: pd.DataFrame, player_names: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Batch prediction over a full dataset.

        Returns
        -------
        league_preds : np.ndarray
        final_preds  : np.ndarray
        stds         : np.ndarray
        """
        league_preds = self.league_model.predict(X)
        final_preds = league_preds.copy()
        default_std = self.league_model.predict_std(X)
        stds = default_std.copy()

        for name, pm in self.player_models.items():
            mask = (player_names == name).values
            if mask.sum() == 0:
                continue
            delta = pm.predict_residual(X[mask])
            std_p = pm.predict_residual_std(X[mask])
            final_preds[mask] = np.clip(league_preds[mask] + delta, 0, self.clip_max)
            stds[mask] = std_p

        return league_preds, final_preds, stds

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate_player(
        self,
        player_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        player_names: pd.Series,
        holdout_last_n: int = 10,
    ) -> Optional[pd.DataFrame]:
        """Walk-forward evaluation for a single player (holdout last N games)."""
        mask = player_names == player_name
        X_p = X[mask].reset_index(drop=True)
        y_p = y[mask].reset_index(drop=True)
        n = len(X_p)
        if n <= holdout_last_n + 5:
            print(f"⚠ Not enough data for {player_name}")
            return None
        tr = slice(0, n - holdout_last_n)
        te = slice(n - holdout_last_n, n)

        league_test = self.league_model.predict(X_p.iloc[te])
        pm_eval = self.PlayerModelClass(player_name)
        pm_eval.fit(X_p.iloc[tr], y_p.iloc[tr], self.league_model.predict(X_p.iloc[tr]), self._cols)
        delta_test = pm_eval.predict_residual(X_p.iloc[te])
        final_test = np.clip(league_test + delta_test, 0, self.clip_max)
        y_test = y_p.iloc[te].values

        results = pd.DataFrame({
            "game": range(1, holdout_last_n + 1),
            f"actual_{self.target_col}": y_test,
            "league_pred": league_test,
            "final_pred": final_test,
            "league_err": np.abs(league_test - y_test),
            "final_err": np.abs(final_test - y_test),
        })
        print(f"\n─── Walk-Forward: {player_name} (last {holdout_last_n} games) ───")
        print(f"  League MAE   : {results['league_err'].mean():.2f} {self.target_col}")
        print(f"  Two-stage MAE: {results['final_err'].mean():.2f} {self.target_col}")
        print(results.to_string(index=False))
        return results

    def evaluate_global_walkforward(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        player_names: pd.Series,
        n_splits: int = 5,
    ) -> pd.DataFrame:
        """Time-series walk-forward evaluation across all data."""
        print(f"\n─── Global Walk-Forward Evaluation ({n_splits} folds) ───")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        records = []
        for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
            pn_tr = player_names.iloc[tr_idx]
            pn_te = player_names.iloc[te_idx]

            fold_predictor = self.__class__()
            fold_predictor.fit_league(X_tr, y_tr)
            fold_predictor.fit_all_players(X_tr, y_tr, pn_tr, min_games=10)

            lp, fp, _ = fold_predictor.predict_all(X_te, pn_te)
            y_arr = y_te.values
            records.append({
                "fold": fold + 1,
                "n_test": int(len(y_te)),
                "league_mae": float(mean_absolute_error(y_te, lp)),
                "final_mae": float(mean_absolute_error(y_te, fp)),
                "league_r2": float(1 - np.sum((y_arr - lp) ** 2) / np.sum((y_arr - y_arr.mean()) ** 2)),
                "final_r2": float(1 - np.sum((y_arr - fp) ** 2) / np.sum((y_arr - y_arr.mean()) ** 2)),
            })
            r = records[-1]
            print(
                f"  Fold {fold+1}: n={r['n_test']:,}  "
                f"League MAE={r['league_mae']:.3f}  "
                f"Final MAE={r['final_mae']:.3f}  "
                f"R²={r['final_r2']:.3f}"
            )

        df_res = pd.DataFrame(records)
        print(
            f"\n  MEAN — League MAE: {df_res['league_mae'].mean():.3f}  "
            f"Final MAE: {df_res['final_mae'].mean():.3f}  "
            f"R²: {df_res['final_r2'].mean():.3f}"
        )
        return df_res

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"✓ Saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "BasePredictor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"✓ Loaded from {path}")
        return obj
