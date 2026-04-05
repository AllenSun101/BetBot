"""
evaluate_model.py — CLI evaluation harness for the NBA Props pipeline.

Usage:
    python evaluate_model.py                       # evaluate both models
    python evaluate_model.py --model pts           # points only
    python evaluate_model.py --model min           # minutes only
    python evaluate_model.py --player "Luka Dončić"
    python evaluate_model.py --compare v1_results.json
"""

from __future__ import annotations

import sys
import argparse
import json
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from config import ML_DATASET_PTS, ML_DATASET_MIN, PTS_MODEL_PKL, MIN_MODEL_PKL, DATA_PATH
from base_predictor import BasePredictor, apply_two_stage
from minutes_predictor import MinutesPredictor
from points_predictor import PointsPredictor

# ── Pretty printing ────────────────────────────────────────────────────────────

WIDTH = 72


def hdr(title: str, char: str = "═") -> None:
    pad = (WIDTH - len(title) - 2) // 2
    print(f"\n{char*pad} {title} {char*(WIDTH - pad - len(title) - 2)}")


def print_metric_row(label: str, v1: float, v2: float | None = None,
                     good_lower: bool = True) -> None:
    if v2 is None:
        print(f"  {label:<30} {v1:.4f}")
        return
    delta = v2 - v1
    arrow = "↓" if delta < 0 else "↑"
    sign = "✓" if (delta < 0) == good_lower else "✗"
    pct = abs(delta) / abs(v1) * 100 if v1 else 0
    print(f"  {label:<30} {v1:.3f} → {v2:.3f}  {arrow}{abs(delta):.3f} ({pct:.1f}%) {sign}")


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_pts_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(ML_DATASET_PTS)
    if "PLAYER_NAME" not in df.columns:
        raise ValueError("PLAYER_NAME missing from dataset — re-run points_feature_engineering.py")
    player_names = df["PLAYER_NAME"].copy()
    X = df.drop(columns=["PTS_TARGET", "PLAYER_NAME"])
    y = df["PTS_TARGET"]
    return X, y, player_names


def load_min_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(ML_DATASET_MIN)
    if "PLAYER_NAME" not in df.columns:
        raise ValueError("PLAYER_NAME missing from dataset — re-run minutes_feature_engineering.py")
    player_names = df["PLAYER_NAME"].copy()
    X = df.drop(columns=["MIN_TARGET", "PLAYER_NAME"])
    y = df["MIN_TARGET"]
    return X, y, player_names


# ── OOS walk-forward evaluation ────────────────────────────────────────────────

def run_oos_walkforward(
    PredictorClass: type,
    X: pd.DataFrame,
    y: pd.Series,
    player_names: pd.Series,
    n_splits: int = 5,
    label: str = "",
) -> tuple[list[dict], dict]:
    """
    Perform time-series walk-forward cross-validation and return per-fold
    records plus an aggregated summary dict.

    Returns (fold_records, summary) where summary contains:
        final_mae_mean, final_mae_std, league_mae_mean, league_mae_std,
        final_r2_mean, final_r2_std, within_2, within_5, n_folds, n_test
    """
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_records = []

    print(f"\n  Running {n_splits}-fold OOS walk-forward for {label}…")
    print(f"  {'Fold':>4}  {'N test':>7}  {'League MAE':>10}  {'Final MAE':>10}  {'Final R²':>9}")
    print(f"  {'-'*50}")

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        fold_pred = PredictorClass()
        fold_pred.fit_league(X.iloc[tr_idx], y.iloc[tr_idx])
        fold_pred.fit_all_players(
            X.iloc[tr_idx], y.iloc[tr_idx],
            player_names.iloc[tr_idx], min_games=10,
        )
        lp, fp, _ = fold_pred.predict_all(X.iloc[te_idx], player_names.iloc[te_idx])
        y_te = y.iloc[te_idx].values
        errs_f = np.abs(y_te - fp)
        r2_f   = float(1 - np.sum((y_te - fp)**2) / np.sum((y_te - y_te.mean())**2))

        rec = {
            "fold":       fold,
            "n_test":     int(len(y_te)),
            "league_mae": float(np.abs(y_te - lp).mean()),
            "final_mae":  float(errs_f.mean()),
            "league_r2":  float(1 - np.sum((y_te-lp)**2)/np.sum((y_te-y_te.mean())**2)),
            "final_r2":   r2_f,
            "within_2":   float((errs_f <= 2).mean()),
            "within_5":   float((errs_f <= 5).mean()),
        }
        fold_records.append(rec)
        print(f"  {fold:>4}  {rec['n_test']:>7,}  {rec['league_mae']:>10.3f}  "
              f"{rec['final_mae']:>10.3f}  {r2_f:>9.3f}")

    # Aggregate
    lmaes = [f["league_mae"] for f in fold_records]
    fmaes = [f["final_mae"]  for f in fold_records]
    fr2s  = [f["final_r2"]   for f in fold_records]
    summary = {
        "league_mae_mean": float(np.mean(lmaes)),
        "league_mae_std":  float(np.std(lmaes)),
        "final_mae_mean":  float(np.mean(fmaes)),
        "final_mae_std":   float(np.std(fmaes)),
        "final_r2_mean":   float(np.mean(fr2s)),
        "final_r2_std":    float(np.std(fr2s)),
        "within_2":        fold_records[-1]["within_2"],
        "within_5":        fold_records[-1]["within_5"],
        "n_folds":         len(fold_records),
        "n_test":          sum(f["n_test"] for f in fold_records),
    }
    print(f"\n  Mean  {'':>7}  {summary['league_mae_mean']:>10.3f}  "
          f"{summary['final_mae_mean']:>10.3f}  {summary['final_r2_mean']:>9.3f}")
    print(f"  Std   {'':>7}  {summary['league_mae_std']:>10.3f}  "
          f"{summary['final_mae_std']:>10.3f}  {summary['final_r2_std']:>9.3f}")

    return fold_records, summary


# ── Evaluation metrics ─────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    errors = np.abs(y_true - y_pred)
    return {
        "mae": float(errors.mean()),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "r2": float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1, None))) * 100),
        "within_2": float((errors <= 2).mean()),
        "within_5": float((errors <= 5).mean()),
        "within_10": float((errors <= 10).mean()),
    }


def print_oos_table(summary: dict, label: str) -> None:
    print(f"\n  {label} OOS Walk-Forward Results  ({summary['n_folds']} folds, "
          f"{summary['n_test']:,} total test rows)")
    print(f"  {'Metric':<28} {'League':>10} {'Two-Stage':>10} {'Δ':>10}")
    print(f"  {'-'*58}")
    rows = [
        ("MAE (mean ± std)", "league_mae_mean", "final_mae_mean", "league_mae_std", "final_mae_std", True),
        ("R² (mean ± std)",  "league_r2_mean",  "final_r2_mean",  None, "final_r2_std", False),
        ("Within 2 (last fold)", None, "within_2",  None, None, False),
        ("Within 5 (last fold)", None, "within_5",  None, None, False),
    ]
    for name, lk, fk, lstd_k, fstd_k, good_lower in rows:
        lv = summary.get(lk) if lk else None
        fv = summary.get(fk)
        fstd = summary.get(fstd_k, 0) if fstd_k else 0
        lv_str = f"{lv:.3f}" if lv is not None else "—"
        fv_str = f"{fv:.3f} ±{fstd:.3f}" if fv is not None else "—"
        if lv is not None and fv is not None:
            delta = fv - lv
            better = (delta < 0) == good_lower
            mark = "✓" if better else "✗"
            delta_str = f"{delta:>+9.4f} {mark}"
        else:
            delta_str = ""
        pct_str = f"{fv:.1%}" if fk in ("within_2","within_5") and fv is not None else fv_str
        print(f"  {name:<28} {lv_str:>10} {pct_str:>10} {delta_str}")


def coverage_stats(predictor: BasePredictor, X: pd.DataFrame, y: pd.Series) -> dict:
    stds = predictor.league_model.predict_std(X)
    errs = np.abs(np.array(y) - predictor.league_model.predict(X))
    return {
        "cov80": float((errs <= 1.28 * stds).mean()),
        "cov95": float((errs <= 1.96 * stds).mean()),
        "mean_std": float(stds.mean()),
    }


# ── Player-level helpers ───────────────────────────────────────────────────────

def player_leaderboard(
    predictor: BasePredictor,
    X: pd.DataFrame,
    y: pd.Series,
    player_names: pd.Series,
    min_games: int = 20,
    top_n: int = 30,
) -> list[dict]:
    from sklearn.metrics import mean_absolute_error
    eligible = player_names.value_counts()[lambda s: s >= min_games].index.tolist()
    records = []
    for name in eligible[:top_n]:
        mask = player_names == name
        X_p, y_p = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
        lp = predictor.league_model.predict(X_p)
        fp = (
            np.clip(lp + predictor.player_models[name].predict_residual(X_p), 0, predictor.clip_max)
            if name in predictor.player_models else np.clip(lp, 0, predictor.clip_max)
        )
        records.append({
            "player":       name,
            "games":        int(mask.sum()),
            "league_mae":   float(mean_absolute_error(y_p, lp)),
            "final_mae":    float(mean_absolute_error(y_p, fp)),
            "residual_std": float(np.std(y_p.values - fp)),
        })
    records.sort(key=lambda r: r["final_mae"])
    return records


def player_deep_dive(
    predictor: BasePredictor,
    X: pd.DataFrame,
    y: pd.Series,
    player_names: pd.Series,
    player_name: str,
    holdout_n: int = 10,
) -> dict | None:
    return predictor.evaluate_player(player_name, X, y, player_names, holdout_last_n=holdout_n)


# ── Per-model evaluation workflow ─────────────────────────────────────────────

def run_model_eval(
    predictor: BasePredictor,
    X: pd.DataFrame,
    y: pd.Series,
    player_names: pd.Series,
    label: str,
    args: argparse.Namespace,
    baseline: dict | None,
    PredictorClass: type,
) -> dict:
    """Full OOS evaluation for one model. Returns results dict."""
    # OOS walk-forward — the primary accuracy numbers
    fold_records, oos_summary = run_oos_walkforward(
        PredictorClass, X, y, player_names, n_splits=5, label=label
    )
    print_oos_table(oos_summary, label)
    results: dict = {"oos": oos_summary, "folds": fold_records}

    # Uncertainty (in-sample, indicative only)
    try:
        cov = coverage_stats(predictor, X, y)
        results["uncertainty"] = cov
        print(
            f"\n  Uncertainty (in-sample) — Mean std: {cov['mean_std']:.2f}  "
            f"80% PI coverage: {cov['cov80']:.1%}  "
            f"95% PI coverage: {cov['cov95']:.1%}"
        )
    except Exception as exc:
        print(f"  ⚠ Uncertainty stats unavailable: {exc}")

    # Leaderboard (in-sample per-player, for reference)
    if args.leaderboard:
        hdr(f"{label} Player Leaderboard (in-sample)", "─")
        lb = player_leaderboard(predictor, X, y, player_names, min_games=20, top_n=25)
        results["leaderboard"] = lb
        print(f"\n  {'Player':<25} {'G':>4} {'LeagMAE':>9} {'FinalMAE':>9} {'Δ':>7}")
        print(f"  {'-' * 56}")
        for p in lb[:15]:
            delta = p["league_mae"] - p["final_mae"]
            print(
                f"  {p['player']:<25} {p['games']:>4} "
                f"{p['league_mae']:>9.2f} {p['final_mae']:>9.2f} {delta:>+7.2f}"
            )

    # Single-player deep-dive
    if args.player:
        hdr(f"Player Deep-Dive: {args.player}", "─")
        pd_res = player_deep_dive(predictor, X, y, player_names, args.player)
        if pd_res is not None:
            results["player_eval"] = {
                "player":     args.player,
                "league_mae": float(pd_res["league_err"].mean()),
                "final_mae":  float(pd_res["final_err"].mean()),
            }

    # Compare to baseline
    if baseline:
        b_key = label.lower()[:3]
        if b_key in baseline:
            hdr(f"{label} vs Baseline", "─")
            b = baseline[b_key]
            boos = b.get("oos", {})
            if boos:
                print(f"  {'Metric':<20} {'Baseline':>12} {'Current':>12} {'Δ':>10}")
                for key, good_lower in [
                    ("final_mae_mean", True), ("final_r2_mean", False)
                ]:
                    bv = boos.get(key)
                    cv = oos_summary.get(key)
                    if bv and cv:
                        print_metric_row(key, bv, cv, good_lower=good_lower)

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="NBA Props — Model Evaluation (OOS Walk-Forward)")
    parser.add_argument("--model",      choices=["pts", "min", "both"], default="both")
    parser.add_argument("--player",     type=str, default=None)
    parser.add_argument("--leaderboard",action="store_true", default=True)
    parser.add_argument("--compare",    type=str, default=None)
    parser.add_argument("--out",        type=str, default="eval_results.json")
    args = parser.parse_args()

    baseline = None
    if args.compare:
        try:
            with open(args.compare) as f:
                baseline = json.load(f)
            print(f"✓ Loaded baseline from {args.compare}")
        except Exception as exc:
            print(f"⚠ Could not load baseline: {exc}")

    all_results: dict = {}

    if args.model in ("pts", "both"):
        hdr("POINTS MODEL — OOS WALK-FORWARD EVALUATION")
        X, y, player_names = load_pts_data()
        predictor = PointsPredictor.load(PTS_MODEL_PKL)
        all_results["pts"] = run_model_eval(
            predictor, X, y, player_names, "Points", args, baseline, PointsPredictor
        )

    if args.model in ("min", "both"):
        hdr("MINUTES MODEL — OOS WALK-FORWARD EVALUATION")
        try:
            X, y, player_names = load_min_data()
            predictor = MinutesPredictor.load(MIN_MODEL_PKL)
            all_results["min"] = run_model_eval(
                predictor, X, y, player_names, "Minutes", args, baseline, MinutesPredictor
            )
        except Exception as exc:
            print(f"  ⚠ Minutes model evaluation failed: {exc}")

    hdr("SAVING RESULTS")
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  ✓ Saved full evaluation to {out_path}")
    print(f"  → Re-run with --compare {out_path} after next training to track changes")


if __name__ == "__main__":
    main()