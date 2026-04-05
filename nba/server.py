"""
NBA Props Engine — Local Dashboard Server (v4)
==============================================
Removed: backtest endpoint and all related code.
Added:   /api/evaluate_model — runs accuracy evaluation against a data file.

Usage
-----
    pip install flask
    python server.py
    → open http://localhost:5000

Endpoints
---------
GET  /                          → serves nba_props_dashboard.html
GET  /api/status                → artifact status
POST /api/retrain               → retrain full pipeline
GET  /api/retrain/status        → poll retrain progress
POST /api/evaluate_model        → evaluate model accuracy on a data file
POST /api/evaluate              → evaluate a single player prop
"""

import sys, json, math, threading, time, traceback, os
from pathlib import Path
from flask import Flask, jsonify, request, send_file

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    ARTIFACTS, ML_DATASET_PTS,
    FINAL_PREDS_NPY, PRED_STDS_NPY,
    MIN_MODEL_PKL, PTS_MODEL_PKL, DATA_PATH,
    ML_DATASET_MIN, MIN_FI_CSV, PTS_FI_CSV,
)


def _safe_val(v):
    import numpy as np
    if isinstance(v, (np.integer,)):      return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, np.ndarray):         return v.tolist()
    return v

def _safe_dict(d):
    return {k: _safe_val(v) for k, v in d.items()}


app = Flask(__name__)

# ── Retrain state ─────────────────────────────────────────────────────────────
_retrain_state = {"running": False, "steps": [], "done": False, "error": None}


# ── Status ────────────────────────────────────────────────────────────────────

def _status_dict():
    files = {
        "min_features": ML_DATASET_MIN,
        "min_model":    MIN_MODEL_PKL,
        "pts_features": ML_DATASET_PTS,
        "pts_model":    PTS_MODEL_PKL,
        "predictions":  FINAL_PREDS_NPY,
    }
    result = {}
    for k, path in files.items():
        p = Path(path)
        if p.exists():
            mtime = os.path.getmtime(p)
            result[k] = {"exists": True, "mtime_str": time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))}
        else:
            result[k] = {"exists": False}
    return result


# ── Model accuracy evaluation ─────────────────────────────────────────────────

def _run_model_evaluation(data_path_str: str | None = None, model: str = "pts") -> dict:
    """
    Run out-of-sample (OOS) walk-forward evaluation using TimeSeriesSplit.
    Returns fold-level metrics so numbers reflect real predictive accuracy,
    not in-sample fit.
    """
    import numpy as np
    import pandas as pd

    results = {}

    def _oos_metrics(fold_records: list[dict]) -> dict:
        """Aggregate fold results into mean ± std summary."""
        keys = ["league_mae", "final_mae", "league_r2", "final_r2"]
        agg = {}
        for k in keys:
            vals = [f[k] for f in fold_records if k in f]
            agg[k + "_mean"] = float(np.mean(vals)) if vals else None
            agg[k + "_std"]  = float(np.std(vals))  if len(vals) > 1 else 0.0
        # Within-N from last fold (most recent held-out data)
        last = fold_records[-1] if fold_records else {}
        agg["within_2"] = last.get("within_2")
        agg["within_5"] = last.get("within_5")
        agg["n_folds"]  = len(fold_records)
        agg["n_test"]   = sum(f.get("n_test", 0) for f in fold_records)
        return agg

    if model in ("pts", "both"):
        if not Path(ML_DATASET_PTS).exists():
            raise FileNotFoundError("Points feature dataset not found. Run the pipeline first.")
        if not Path(PTS_MODEL_PKL).exists():
            raise FileNotFoundError("Points model not found. Run the pipeline first.")

        import points_predictor as _pp
        sys.modules['__main__'] = _pp
        from points_predictor import PointsPredictor
        from sklearn.model_selection import TimeSeriesSplit

        df_pts = pd.read_csv(ML_DATASET_PTS)
        if "PLAYER_NAME" not in df_pts.columns:
            raise ValueError("PLAYER_NAME missing from dataset — re-run points_feature_engineering.py")
        player_names = df_pts["PLAYER_NAME"].copy()
        X = df_pts.drop(columns=["PTS_TARGET", "PLAYER_NAME"])
        y = df_pts["PTS_TARGET"].values

        predictor = PointsPredictor.load(PTS_MODEL_PKL)

        # OOS walk-forward: 5 time-series folds
        tscv = TimeSeriesSplit(n_splits=5)
        fold_records = []
        for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
            fold_pred = PointsPredictor()
            fold_pred.fit_league(X.iloc[tr_idx], pd.Series(y[tr_idx]))
            fold_pred.fit_all_players(
                X.iloc[tr_idx], pd.Series(y[tr_idx]),
                player_names.iloc[tr_idx], min_games=10
            )
            lp, fp, _ = fold_pred.predict_all(X.iloc[te_idx], player_names.iloc[te_idx])
            y_te = y[te_idx]
            errs_l = np.abs(y_te - lp)
            errs_f = np.abs(y_te - fp)
            fold_records.append({
                "fold": fold + 1,
                "n_test": int(len(y_te)),
                "league_mae": float(errs_l.mean()),
                "final_mae":  float(errs_f.mean()),
                "league_r2":  float(1 - np.sum((y_te-lp)**2) / np.sum((y_te-y_te.mean())**2)),
                "final_r2":   float(1 - np.sum((y_te-fp)**2) / np.sum((y_te-y_te.mean())**2)),
                "within_2":   float((errs_f <= 2).mean()),
                "within_5":   float((errs_f <= 5).mean()),
            })

        oos = _oos_metrics(fold_records)

        # In-sample on full dataset for player leaderboard display only
        lp_full, fp_full, stds = predictor.predict_all(X, player_names)
        player_lb = []
        for name in player_names.value_counts()[player_names.value_counts() >= 20].head(30).index:
            mask = (player_names == name).values
            yi, lpi, fpi = y[mask], lp_full[mask], fp_full[mask]
            player_lb.append({
                "player":     name,
                "games":      int(mask.sum()),
                "league_mae": round(float(np.abs(yi - lpi).mean()), 2),
                "final_mae":  round(float(np.abs(yi - fpi).mean()), 2),
            })
        player_lb.sort(key=lambda r: r["final_mae"])

        errs_f_full = np.abs(y - fp_full)
        hist_counts, hist_edges = np.histogram(errs_f_full, bins=[0,2,4,6,8,10,15,100])
        error_hist = [
            {"bin": f"{int(hist_edges[i])}-{int(hist_edges[i+1])}", "count": int(hist_counts[i])}
            for i in range(len(hist_counts))
        ]

        results["pts"] = {
            "oos": oos,
            "folds": fold_records,
            "n": int(len(y)),
            "mean_std": float(stds.mean()),
            "player_leaderboard": player_lb,
            "error_histogram": error_hist,
        }

    if model in ("min", "both"):
        if not Path(ML_DATASET_MIN).exists():
            raise FileNotFoundError("Minutes feature dataset not found. Run the pipeline first.")
        if not Path(MIN_MODEL_PKL).exists():
            raise FileNotFoundError("Minutes model not found. Run the pipeline first.")

        import minutes_predictor as _mp
        sys.modules['__main__'] = _mp
        from minutes_predictor import MinutesPredictor
        from sklearn.model_selection import TimeSeriesSplit

        df_min = pd.read_csv(ML_DATASET_MIN)
        if "PLAYER_NAME" not in df_min.columns:
            raise ValueError("PLAYER_NAME missing from dataset — re-run minutes_feature_engineering.py")
        player_names_m = df_min["PLAYER_NAME"].copy()
        X_m = df_min.drop(columns=["MIN_TARGET", "PLAYER_NAME"])
        y_m = df_min["MIN_TARGET"].values

        pred_m = MinutesPredictor.load(MIN_MODEL_PKL)

        tscv = TimeSeriesSplit(n_splits=5)
        fold_records_m = []
        for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_m)):
            fold_pred = MinutesPredictor()
            fold_pred.fit_league(X_m.iloc[tr_idx], pd.Series(y_m[tr_idx]))
            fold_pred.fit_all_players(
                X_m.iloc[tr_idx], pd.Series(y_m[tr_idx]),
                player_names_m.iloc[tr_idx], min_games=10
            )
            lp, fp, _ = fold_pred.predict_all(X_m.iloc[te_idx], player_names_m.iloc[te_idx])
            y_te = y_m[te_idx]
            errs_f = np.abs(y_te - fp)
            fold_records_m.append({
                "fold": fold + 1,
                "n_test": int(len(y_te)),
                "league_mae": float(np.abs(y_te - lp).mean()),
                "final_mae":  float(errs_f.mean()),
                "league_r2":  float(1 - np.sum((y_te-lp)**2) / np.sum((y_te-y_te.mean())**2)),
                "final_r2":   float(1 - np.sum((y_te-fp)**2) / np.sum((y_te-y_te.mean())**2)),
                "within_2":   float((errs_f <= 2).mean()),
                "within_5":   float((errs_f <= 5).mean()),
            })

        results["min"] = {
            "oos": _oos_metrics(fold_records_m),
            "folds": fold_records_m,
            "n": int(len(y_m)),
        }

    # Feature importances (unchanged)
    fi_rows = []
    fi_path = Path(PTS_FI_CSV) if model in ("pts", "both") else Path(MIN_FI_CSV)
    if fi_path.exists():
        try:
            fi_df = pd.read_csv(fi_path, index_col=0)
            fi_rows = [
                {"feature": str(r.name), "importance": float(r.iloc[0])}
                for _, r in fi_df.head(20).iterrows()
            ]
        except Exception:
            pass

    results["feature_importances"] = fi_rows
    return results


# ── Retrain thread ────────────────────────────────────────────────────────────

def _retrain_thread(data_path=None):
    import subprocess
    global _retrain_state
    scripts = [
        ("1_min_features", ROOT / "minutes_feature_engineering.py"),
        ("2_min_model",    ROOT / "minutes_predictor.py"),
        ("3_pts_features", ROOT / "points_feature_engineering.py"),
        ("4_pts_model",    ROOT / "points_predictor.py"),
        ("5_predictions",  ROOT / "generate_preds.py"),
    ]
    env = None
    if data_path:
        env = {**os.environ, "NBA_DATA_PATH": str(data_path)}

    for label, script in scripts:
        _retrain_state["steps"].append({"step": label, "status": "running"})
        t0 = time.time()
        try:
            r = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True, text=True, timeout=900, env=env,
            )
            elapsed = round(time.time() - t0, 1)
            ok = r.returncode == 0
            _retrain_state["steps"][-1].update({
                "status":  "ok" if ok else "failed",
                "elapsed": elapsed,
                "stderr":  r.stderr[-500:] if not ok else "",
            })
            if not ok:
                _retrain_state["error"] = f"{label} failed: {r.stderr[-300:]}"
                break
        except Exception as e:
            _retrain_state["steps"][-1].update({"status": "failed", "error": str(e)})
            _retrain_state["error"] = str(e)
            break

    _retrain_state["running"] = False
    _retrain_state["done"]    = True


# ── Global error handler ──────────────────────────────────────────────────────

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    html_path = ROOT / "nba_props_dashboard.html"
    if not html_path.exists():
        return "nba_props_dashboard.html not found", 404
    return send_file(html_path)


@app.route("/api/status")
def api_status():
    return jsonify(_status_dict())


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    global _retrain_state
    if _retrain_state["running"]:
        return jsonify({"error": "Retrain already in progress"}), 409
    data      = request.get_json(force=True) or {}
    data_path = data.get("data_path")
    _retrain_state = {"running": True, "steps": [], "done": False, "error": None}
    t = threading.Thread(target=_retrain_thread, args=(data_path,), daemon=True)
    t.start()
    return jsonify({"started": True})


@app.route("/api/retrain/status")
def api_retrain_status():
    return jsonify(_retrain_state)


@app.route("/api/evaluate_model", methods=["POST"])
def api_evaluate_model():
    """
    Evaluate model accuracy (MAE, RMSE, R², within-N) against the trained artifacts.
    Optionally re-runs feature engineering against a custom data file path.

    Body (JSON):
        data_path  : str  (optional) — path to player_data_full.csv on server disk
        model      : "pts" | "min" | "both"  (default: "pts")
    """
    data      = request.get_json(force=True) or {}
    data_path = data.get("data_path")  # optional override
    model     = data.get("model", "pts")

    if model not in ("pts", "min", "both"):
        return jsonify({"error": "model must be 'pts', 'min', or 'both'"}), 400

    try:
        result = _run_model_evaluation(data_path_str=data_path, model=model)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """Evaluate a single player prop."""
    data = request.get_json(force=True) or {}
    required = ["player_name", "opponent", "is_home", "game_date", "line", "side", "american_odds"]
    missing  = [k for k in required if k not in data or data[k] is None or data[k] == ""]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        from prop_evaluator import PropEvaluator
        ev     = PropEvaluator()
        result = ev.evaluate_from_dataset(
            player_name   = str(data["player_name"]),
            opponent      = str(data["opponent"]).upper(),
            is_home       = int(data["is_home"]),
            game_date     = str(data["game_date"]),
            line          = float(data["line"]),
            side          = str(data["side"]).lower(),
            american_odds = int(data["american_odds"]),
        )
        return jsonify(_safe_dict(result))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/evaluate_parlay", methods=["POST"])
def api_evaluate_parlay():
    """
    Evaluate a two-leg points prop parlay.

    Body (JSON):
        leg1                  : dict   — same fields as /api/evaluate
        leg2                  : dict   — same fields as /api/evaluate
        parlay_american_odds  : int    — book payout in American odds  (e.g. +260)
        parlay_multiplier     : float  — book payout as profit multiplier (e.g. 3.5)
        Exactly one of the two payout fields must be supplied.
    """
    data = request.get_json(force=True) or {}

    for leg_key in ("leg1", "leg2"):
        if leg_key not in data or not isinstance(data[leg_key], dict):
            return jsonify({"error": f"Missing or invalid '{leg_key}' object"}), 400
        required = ["player_name", "opponent", "is_home", "game_date", "line", "side", "american_odds"]
        missing = [k for k in required if k not in data[leg_key] or data[leg_key][k] in (None, "")]
        if missing:
            return jsonify({"error": f"{leg_key} missing fields: {', '.join(missing)}"}), 400

    has_american    = data.get("parlay_american_odds") not in (None, "")
    has_multiplier  = data.get("parlay_multiplier")    not in (None, "")
    if not has_american and not has_multiplier:
        return jsonify({"error": "Provide parlay_american_odds or parlay_multiplier"}), 400
    if has_american and has_multiplier:
        return jsonify({"error": "Provide only one of parlay_american_odds or parlay_multiplier"}), 400

    def _coerce_leg(raw: dict) -> dict:
        return dict(
            player_name   = str(raw["player_name"]),
            opponent      = str(raw["opponent"]).upper(),
            is_home       = int(raw["is_home"]),
            game_date     = str(raw["game_date"]),
            line          = float(raw["line"]),
            side          = str(raw["side"]).lower(),
            american_odds = int(raw["american_odds"]),
        )

    try:
        from prop_evaluator import ParlayEvaluator
        ev     = ParlayEvaluator()
        result = ev.evaluate(
            leg1                 = _coerce_leg(data["leg1"]),
            leg2                 = _coerce_leg(data["leg2"]),
            parlay_american_odds = int(data["parlay_american_odds"]) if has_american else None,
            parlay_multiplier    = float(data["parlay_multiplier"])  if has_multiplier else None,
        )
        safe = {
            k: (_safe_dict(v) if isinstance(v, dict) else _safe_val(v))
            for k, v in result.items()
        }
        return jsonify(safe)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/batch_props", methods=["POST"])
def api_batch_props():
    """
    Evaluate a batch of props from a spreadsheet and return all two-leg combos.

    Body (JSON):
        rows      : list of dicts — each with player_name, opponent, is_home,
                                    prop_line, game_date (optional)
        game_date : str  — default date for rows that omit game_date
    """
    data      = request.get_json(force=True) or {}
    rows      = data.get("rows", [])
    game_date = data.get("game_date", "")

    if not rows:
        return jsonify({"error": "No rows provided"}), 400
    if not game_date:
        from datetime import date
        game_date = date.today().isoformat()

    try:
        from prop_evaluator import batch_evaluate_props
        props, combos = batch_evaluate_props(rows, game_date)
        return jsonify({"props": props, "combos": combos})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  NBA Props Engine — Local Dashboard")
    print("  ────────────────────────────────────")
    print(f"  Project folder : {ROOT}")
    print(f"  Data file      : {DATA_PATH}")
    print(f"  Artifacts      : {ARTIFACTS}")
    print()
    for k, v in _status_dict().items():
        icon = "✓" if v["exists"] else "✗"
        ts   = v.get("mtime_str", "not found")
        print(f"  {icon}  {k:<20} {ts}")
    print()
    print("  Open http://localhost:5000 in your browser")
    print()
    app.run(host="127.0.0.1", port=5000, debug=False)