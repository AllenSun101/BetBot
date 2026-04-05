"""
config.py  —  Central path configuration for the NBA Props Engine.

All scripts resolve paths relative to the directory this file lives in,
so the entire project works from any folder you choose to put it in.
Just keep all .py files in the same directory alongside player_data.csv.

Folder layout expected:
  your_project/
  ├── config.py                      ← this file
  ├── player_data.csv                ← your NBA game log data
  ├── minutes_feature_engineering.py
  ├── minutes_predictor.py
  ├── points_feature_engineering.py
  ├── points_predictor.py
  ├── generate_preds.py
  ├── props_backtest.py
  ├── pipeline_runner.py
  ├── nba_props_dashboard.html
  └── artifacts/                     ← auto-created on first run
      ├── ml_dataset_minutes.csv
      ├── ml_dataset_pts.csv
      ├── minutes_predictor.pkl
      ├── pts_predictor.pkl
      ├── final_preds.npy
      ├── pred_stds.npy
      └── backtest_ledger.csv  (etc.)
"""

from pathlib import Path

# Root = the directory this config.py lives in
ROOT = Path(__file__).resolve().parent

# ── Input data ────────────────────────────────────────────────────────────────
# Default: player_data.csv sits in the same folder as the scripts.
# Override: pass --data /path/to/file.csv to pipeline_runner.py, or
#           set the NBA_DATA_PATH environment variable.
_override_file = ROOT / "_data_override.txt"
if _override_file.exists():
    DATA_PATH = Path(_override_file.read_text().strip())
elif "NBA_DATA_PATH" in __import__("os").environ:
    DATA_PATH = Path(__import__("os").environ["NBA_DATA_PATH"])
else:
    DATA_PATH = ROOT / "data/player_data_full.csv"

# ── Artifacts directory (created automatically) ───────────────────────────────
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# ── Intermediate datasets ─────────────────────────────────────────────────────
ML_DATASET_MIN  = ARTIFACTS / "ml_dataset_minutes.csv"
ML_DATASET_PTS  = ARTIFACTS / "ml_dataset_pts.csv"
X_FEATURES_MIN  = ARTIFACTS / "X_min_features.csv"
Y_TARGET_MIN    = ARTIFACTS / "y_min_target.csv"
X_FEATURES_PTS  = ARTIFACTS / "X_pts_features.csv"
Y_TARGET_PTS    = ARTIFACTS / "y_pts_target.csv"
PTS_PLAYER_NAMES= ARTIFACTS / "pts_player_names.csv"

# ── Trained models ────────────────────────────────────────────────────────────
MIN_MODEL_PKL   = ARTIFACTS / "minutes_predictor.pkl"
PTS_MODEL_PKL   = ARTIFACTS / "pts_predictor.pkl"

# ── Prediction arrays ─────────────────────────────────────────────────────────
FINAL_PREDS_NPY = ARTIFACTS / "final_preds.npy"
PRED_STDS_NPY   = ARTIFACTS / "pred_stds.npy"
LEAGUE_PREDS_NPY= ARTIFACTS / "league_preds.npy"

# ── Feature importances ───────────────────────────────────────────────────────
MIN_FI_CSV      = ARTIFACTS / "feature_importances.csv"
PTS_FI_CSV      = ARTIFACTS / "pts_feature_importances.csv"

# ── Walk-forward evaluation ───────────────────────────────────────────────────
WALKFORWARD_MIN = ARTIFACTS / "walkforward_eval.csv"

def str_path(p) -> str:
    """Return string path — convenience for libraries that don't accept Path objects."""
    return str(p)