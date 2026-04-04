"""
data_pipeline.py — NBA data fetcher and normalizer.

Fetches box scores via nba_api and normalizes them into the canonical
column schema used throughout the Props Engine.

New API column  →  Canonical column
────────────────────────────────────
personId        →  PLAYER_ID
firstName+familyName → PLAYER_NAME
minutes (str)   →  MIN  (float)
points          →  PTS
fieldGoalsMade  →  FGM
fieldGoalsAttempted → FGA
fieldGoalsPercentage → FG_PCT
threePointersMade   → FG3M
threePointersAttempted → FG3A
threePointersPercentage → FG3_PCT
freeThrowsMade      → FTM
freeThrowsAttempted → FTA
freeThrowsPercentage → FT_PCT
reboundsTotal       → REB
assists             → AST
steals              → STL
blocks              → BLK
turnovers           → TOV
foulsPersonal       → PF
plusMinusPoints     → PLUS_MINUS
teamTricode         → TEAM_ABBREVIATION
GAME_DATE (from metadata) → GAME_DATE
MATCHUP   (from metadata) → MATCHUP
WL        (from metadata) → WL

Failed game IDs are written to data/failed_game_ids.json so they can be
retried later with:
    python data_pipeline.py --retry
    python data_pipeline.py --retry --clear-on-success
"""

from __future__ import annotations

from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv3
import argparse
import json
import pandas as pd
import time
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR        = Path("data")
MASTER_CSV      = DATA_DIR / "player_data_full.csv"
FAILED_IDS_JSON = DATA_DIR / "failed_game_ids.json"
SEASON          = "2025-26"

SLEEP_SUCCESS = 1.5
SLEEP_RETRY   = 5
MAX_RETRIES   = 2

DATA_DIR.mkdir(exist_ok=True)


# ── Column mapping ─────────────────────────────────────────────────────────────

_COL_MAP = {
    "personId":                 "PLAYER_ID",
    "teamTricode":              "TEAM_ABBREVIATION",
    "minutes":                  "MIN",
    "points":                   "PTS",
    "fieldGoalsMade":           "FGM",
    "fieldGoalsAttempted":      "FGA",
    "fieldGoalsPercentage":     "FG_PCT",
    "threePointersMade":        "FG3M",
    "threePointersAttempted":   "FG3A",
    "threePointersPercentage":  "FG3_PCT",
    "freeThrowsMade":           "FTM",
    "freeThrowsAttempted":      "FTA",
    "freeThrowsPercentage":     "FT_PCT",
    "reboundsTotal":            "REB",
    "assists":                  "AST",
    "steals":                   "STL",
    "blocks":                   "BLK",
    "turnovers":                "TOV",
    "foulsPersonal":            "PF",
    "plusMinusPoints":          "PLUS_MINUS",
}

# ── Failed game ID tracking ────────────────────────────────────────────────────

def _load_failed() -> dict:
    """
    Load the failed game IDs file.
    Returns a dict keyed by game_id with metadata about each failure:
        { "game_id": { "first_failed": ISO timestamp, "attempts": N, "last_error": str } }
    """
    if FAILED_IDS_JSON.exists():
        try:
            return json.loads(FAILED_IDS_JSON.read_text())
        except Exception:
            log.warning("Could not parse failed_game_ids.json — starting fresh.")
    return {}


def _save_failed(failed: dict) -> None:
    FAILED_IDS_JSON.write_text(json.dumps(failed, indent=2))


def _record_failure(failed: dict, game_id: str, error: str) -> dict:
    """Add or update a failure entry for game_id."""
    gid = str(game_id)
    if gid not in failed:
        failed[gid] = {
            "first_failed": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "attempts":     0,
            "last_error":   "",
        }
    failed[gid]["attempts"]   += 1
    failed[gid]["last_error"]  = str(error)
    failed[gid]["last_tried"]  = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return failed


def _remove_success(failed: dict, game_id: str) -> dict:
    """Remove a game from the failed list after a successful retry."""
    failed.pop(str(game_id), None)
    return failed


def list_failed() -> list[str]:
    """Return the current list of failed game IDs."""
    return list(_load_failed().keys())


# ── Schema helpers ─────────────────────────────────────────────────────────────

def parse_minutes(val) -> float:
    """Convert 'mm:ss' string or plain float to decimal minutes."""
    if pd.isna(val) or val == "":
        return 0.0
    s = str(val).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            return int(parts[0]) + int(parts[1]) / 60.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def normalize_boxscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename new API columns to canonical names, parse minutes, build PLAYER_NAME.
    Works with both camelCase (new API) and UPPER_CASE (old) schemas.
    """
    if "firstName" in df.columns and "familyName" in df.columns:
        df["PLAYER_NAME"] = (df["firstName"].fillna("") + " " + df["familyName"].fillna("")).str.strip()
    elif "PLAYER_NAME" not in df.columns:
        df["PLAYER_NAME"] = df.get("playerName", "Unknown")

    rename = {old: new for old, new in _COL_MAP.items() if old in df.columns and new not in df.columns}
    df = df.rename(columns=rename)

    if "MIN" in df.columns:
        df["MIN"] = df["MIN"].apply(parse_minutes)

    return df


# ── API fetch helpers ──────────────────────────────────────────────────────────

def get_game_ids_in_range(start_date: str, end_date: str):
    log.info(f"Fetching all games for season {SEASON}...")
    df_all = leaguegamefinder.LeagueGameFinder(
        season_nullable=SEASON,
        league_id_nullable="00"
    ).get_data_frames()[0]

    df_all["GAME_DATE"] = pd.to_datetime(df_all["GAME_DATE"])
    start_dt = pd.to_datetime(start_date)
    end_dt   = pd.to_datetime(end_date)

    df_filtered = df_all[(df_all["GAME_DATE"] >= start_dt) & (df_all["GAME_DATE"] <= end_dt)]
    game_ids = df_filtered["GAME_ID"].unique().tolist()
    log.info(f"Games in range {start_date} → {end_date}: {len(game_ids)}")
    return game_ids, df_filtered


def fetch_game_boxscore(game_id: str) -> tuple[pd.DataFrame | None, str | None]:
    """
    Fetch one game's box score with retries.
    Returns (DataFrame, None) on success or (None, error_message) on total failure.
    """
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            boxscore = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
            if hasattr(boxscore, "player_stats"):
                df = boxscore.player_stats.get_data_frame()
            else:
                df = boxscore.get_data_frames()[0]
            df["GAME_ID"] = game_id
            time.sleep(SLEEP_SUCCESS)
            return df, None
        except Exception as e:
            last_error = str(e)
            log.warning(f"Game {game_id} attempt {attempt}/{MAX_RETRIES}: {e}")
            time.sleep(SLEEP_RETRY * attempt)

    log.error(f"Game {game_id} failed after {MAX_RETRIES} attempts — recorded to {FAILED_IDS_JSON}")
    return None, last_error


# ── Save helpers ───────────────────────────────────────────────────────────────

def _attach_metadata(df_stats: pd.DataFrame, gid: str, df_games_info: pd.DataFrame) -> pd.DataFrame:
    """Merge leaguegamefinder metadata columns onto a box score DataFrame."""
    meta_rows = df_games_info[df_games_info["GAME_ID"] == gid]
    if not meta_rows.empty:
        meta = meta_rows.iloc[0]
        df_stats["MATCHUP"]           = meta["MATCHUP"]
        df_stats["WL"]                = meta["WL"]
        df_stats["SEASON_ID"]         = meta.get("SEASON_ID", "")
        df_stats["GAME_DATE"]         = pd.to_datetime(meta["GAME_DATE"]).strftime("%Y-%m-%d")
        df_stats["TEAM_ABBREVIATION"] = meta.get("TEAM_ABBREVIATION",
                                                  df_stats.get("TEAM_ABBREVIATION", ""))
    return df_stats


def _merge_and_save(master_df: pd.DataFrame, new_rows: list[pd.DataFrame]) -> pd.DataFrame:
    """Concat new rows onto master, deduplicate, and write to CSV."""
    new_df   = pd.concat(new_rows, ignore_index=True)
    combined = pd.concat([master_df, new_df], ignore_index=True) if not master_df.empty else new_df
    combined["GAME_ID"]   = combined["GAME_ID"].astype(str).str.lstrip("0")
    combined["PLAYER_ID"] = combined["PLAYER_ID"].astype(str)
    combined = combined.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"], keep="last")
    combined.to_csv(MASTER_CSV, index=False)
    log.info(f"Saved CSV: {len(combined):,} total rows")
    return combined


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch all games in the date range, normalize columns, and append to master CSV.
    Failures are recorded to failed_game_ids.json automatically.
    """
    if MASTER_CSV.exists():
        master_df = normalize_boxscore(pd.read_csv(MASTER_CSV))
        log.info(f"Loaded existing master CSV: {len(master_df):,} rows")
    else:
        master_df = pd.DataFrame()

    existing_games = set(master_df["GAME_ID"].astype(str)) if not master_df.empty else set()
    failed         = _load_failed()

    game_ids, df_games_info = get_game_ids_in_range(start_date, end_date)
    game_ids_to_fetch = [gid for gid in game_ids if str(gid) not in existing_games]
    log.info(f"Games to fetch: {len(game_ids_to_fetch)}")

    all_new_rows: list[pd.DataFrame] = []
    n_failed = 0

    for i, gid in enumerate(game_ids_to_fetch, 1):
        df_stats, error = fetch_game_boxscore(gid)

        if df_stats is None or df_stats.empty:
            failed = _record_failure(failed, gid, error or "empty response")
            n_failed += 1
        else:
            df_stats = normalize_boxscore(df_stats)
            df_stats = _attach_metadata(df_stats, gid, df_games_info)
            all_new_rows.append(df_stats)
            # If this game was previously failed and now succeeded, remove it
            failed = _remove_success(failed, gid)

        if i % 50 == 0:
            log.info(f"Progress: {i}/{len(game_ids_to_fetch)} — {n_failed} failures so far")

    # Always persist the failure ledger (even if unchanged, keeps it fresh)
    _save_failed(failed)
    if n_failed:
        log.warning(f"{n_failed} game(s) failed — see {FAILED_IDS_JSON}")

    if all_new_rows:
        master_df = _merge_and_save(master_df, all_new_rows)
    else:
        log.info("No new rows to add")

    return master_df


# ── Retry failed games ────────────────────────────────────────────────────────

def _fetch_game_metadata(game_id: str) -> pd.DataFrame:
    """
    Fetch LeagueGameFinder metadata for a single game ID.
    Returns a DataFrame (may be empty if the game isn't found yet).
    Fetched fresh each call so mid-season additions are always current.
    """
    try:
        df = leaguegamefinder.LeagueGameFinder(
            game_id_nullable=game_id,
            league_id_nullable="00",
        ).get_data_frames()[0]
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        time.sleep(SLEEP_SUCCESS)
        return df
    except Exception as e:
        log.warning(f"  Could not fetch metadata for {game_id}: {e}")
        return pd.DataFrame()


def retry_failed(clear_on_success: bool = True) -> dict:
    """
    Attempt to fetch every game ID currently in failed_game_ids.json.
    Metadata (MATCHUP, WL, GAME_DATE …) is fetched fresh per-game so that
    games added mid-season are found correctly.

    Parameters
    ----------
    clear_on_success : bool
        If True (default), remove a game from the failed list once it fetches
        successfully.  Set to False to keep the entry for inspection.

    Returns
    -------
    dict with keys:
        attempted    : int        — number of IDs tried
        recovered    : list[str]  — IDs that succeeded this time
        still_failed : list[str]  — IDs that failed again
    """
    failed = _load_failed()
    if not failed:
        log.info("No failed game IDs on record — nothing to retry.")
        return {"attempted": 0, "recovered": [], "still_failed": []}

    log.info(f"Retrying {len(failed)} failed game ID(s)...")
    master_df = normalize_boxscore(pd.read_csv(MASTER_CSV)) if MASTER_CSV.exists() else pd.DataFrame()

    recovered:    list[str]          = []
    still_failed: list[str]          = []
    new_rows:     list[pd.DataFrame] = []

    for gid in list(failed.keys()):
        log.info(f"  Retrying {gid} (prev attempts: {failed[gid]['attempts']})...")
        df_stats, error = fetch_game_boxscore(gid)

        if df_stats is None or df_stats.empty:
            failed = _record_failure(failed, gid, error or "empty response")
            still_failed.append(gid)
            log.warning(f"  ✗ {gid} still failing")
            continue

        # Fetch metadata fresh for this specific game — captures any new games
        # added since the last pipeline run without re-pulling the whole season.
        df_meta = _fetch_game_metadata(gid)
        df_stats = normalize_boxscore(df_stats)
        df_stats = _attach_metadata(df_stats, gid, df_meta)
        new_rows.append(df_stats)
        recovered.append(gid)
        log.info(f"  ✓ {gid} recovered")
        if clear_on_success:
            failed = _remove_success(failed, gid)

    _save_failed(failed)

    if new_rows:
        master_df = _merge_and_save(master_df, new_rows)

    summary = {
        "attempted":    len(recovered) + len(still_failed),
        "recovered":    recovered,
        "still_failed": still_failed,
    }
    log.info(
        f"Retry complete — recovered: {len(recovered)}, "
        f"still failed: {len(still_failed)}"
    )
    if still_failed:
        log.warning(f"Remaining failures saved to {FAILED_IDS_JSON}")
    return summary


# ── Load helper (used by feature engineering) ─────────────────────────────────

def load_and_normalize(path: Path | str) -> pd.DataFrame:
    """
    Load a CSV from disk and normalize it to the canonical column schema.
    Use this in feature engineering scripts to handle both old and new data.
    """
    df = pd.read_csv(path)
    df = normalize_boxscore(df)

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    if "MIN" in df.columns:
        df = df[df["MIN"] > 0].copy()

    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NBA Props data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Fetch games for a date range
  python data_pipeline.py --start 2025-10-30 --end 2025-11-15

  # Show which game IDs are currently marked as failed
  python data_pipeline.py --list-failed

  # Retry all failed game IDs (remove from list when successful)
  python data_pipeline.py --retry

  # Retry but keep entries even after success (for manual inspection)
  python data_pipeline.py --retry --keep-on-success
""",
    )
    p.add_argument("--start",           metavar="YYYY-MM-DD", help="Start date for range fetch, inclusive")
    p.add_argument("--end",             metavar="YYYY-MM-DD", help="End date for range fetch, inclusive")
    p.add_argument("--retry",           action="store_true",  help="Retry all failed game IDs")
    p.add_argument("--keep-on-success", action="store_true",
                   help="With --retry: keep game in failed list even after it succeeds")
    p.add_argument("--list-failed",     action="store_true",  help="Print failed game IDs and exit")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.list_failed:
        failed = _load_failed()
        if not failed:
            print("No failed game IDs on record.")
        else:
            print(f"{len(failed)} failed game ID(s):\n")
            print(f"  {'Game ID':<20} {'Attempts':>8}  {'Last error'}")
            print("  " + "-" * 70)
            for gid, info in failed.items():
                print(f"  {gid:<20} {info['attempts']:>8}  {info['last_error'][:60]}")

    elif args.retry:
        summary = retry_failed(clear_on_success=not args.keep_on_success)
        print(f"\nRetry summary:")
        print(f"  Attempted    : {summary['attempted']}")
        print(f"  Recovered    : {len(summary['recovered'])}")
        print(f"  Still failed : {len(summary['still_failed'])}")
        if summary["recovered"]:
            print(f"  Recovered IDs: {', '.join(summary['recovered'])}")
        if summary["still_failed"]:
            print(f"  Still failing: {', '.join(summary['still_failed'])}")

    elif args.start and args.end:
        df = run_pipeline(start_date=args.start, end_date=args.end)
        print(f"\nFinal dataset: {len(df):,} rows")
        print(f"Unique players: {df['PLAYER_ID'].nunique()}")
        print(f"Failed IDs on record: {len(_load_failed())}")

    else:
        _build_parser().print_help()
