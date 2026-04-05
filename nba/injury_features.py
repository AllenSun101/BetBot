"""
injury_features.py — NBA Injury Feature Engineering

Reads injury data from data/injury_data.csv (written by data_pipeline.py)
and engineers features for the minutes and points prediction pipelines.

All API fetching lives in data_pipeline.py. This module is purely a
feature-engineering layer — no network calls happen here.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TEAMMATE INJURIES AFFECT MINUTES AND POINTS — THE MECHANISM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NBA teams dress exactly 13 players per game and play roughly 240 total
player-minutes. When a high-usage player is out, those minutes and shot
attempts don't disappear — they redistribute to the remaining healthy
players, roughly in proportion to each player's existing role.

We model this redistribution explicitly:

1. MINUTES REDISTRIBUTION
   ─────────────────────────────────────────────────────────────────────
   When teammate X (who averaged 30 min/game) is Out, the team still
   needs to fill those 30 minutes. Each healthy player absorbs a share
   proportional to their own minute share of the remaining rotation.

   absorbed_min_i = absent_min * (player_i_min / sum_of_other_healthy_min)

   We approximate this as:
       TEAMMATE_MIN_ABSORBED ≈ sum over Out teammates of:
           teammate_avg_min × (this_player_avg_min / team_avg_min_total)

   This is the key feature for the MINUTES model.

2. USAGE / SHOT REDISTRIBUTION  
   ─────────────────────────────────────────────────────────────────────
   Shot attempts redistribute similarly. If the missing player was a
   primary scorer (high FGA/game), more shot opportunities flow to
   healthy players weighted by their own shot-creation rate.

       TEAMMATE_FGA_ABSORBED ≈ sum over Out scorers of:
           teammate_avg_fga × (this_player_fga_share_of_team)

   This is the key feature for the POINTS model.

3. STAR MULTIPLIER
   ─────────────────────────────────────────────────────────────────────
   A "star" player (≥28 avg min/game) being Out has outsized impact
   because they're often the primary ball-handler or play-finisher —
   their absence disrupts offensive sets beyond just raw minutes.
   We track this separately as TEAM_STARS_OUT.

4. OPPONENT INJURY ADVANTAGE
   ─────────────────────────────────────────────────────────────────────
   When the opposing team's star defender or primary ball-handler is
   out, the offensive player's efficiency and usage increase. We capture
   this via OPP_INJURY_SEVERITY and OPP_STARS_OUT.

5. OWN INJURY HISTORY (return-from-injury ramp-up)
   ─────────────────────────────────────────────────────────────────────
   Players returning from injury typically play restricted minutes for
   3-5 games. RETURN_FROM_INJURY_FLAG and INJURY_GAMES_MISSED_RECENT
   capture this suppression effect.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output columns (all zero-filled when injury data is unavailable):
    PLAYER_ID, GAME_DATE,
    -- own injury status --
    PLAYER_STATUS_SCORE, PLAYER_IS_OUT, PLAYER_IS_QUESTIONABLE,
    PLAYER_INJURY_RISK_ROLL5, DAYS_SINCE_LAST_INJURY,
    RETURN_FROM_INJURY_FLAG, INJURY_GAMES_MISSED_RECENT,
    -- teammate redistribution (the core signal) --
    TEAMMATE_MIN_ABSORBED,       # estimated extra minutes from absent teammates
    TEAMMATE_FGA_ABSORBED,       # estimated extra shot attempts from absent teammates
    TEAM_STARS_OUT,              # number of star (≥28 avg min) teammates Out
    LINEUP_DISRUPTION_SCORE,     # star-weighted disruption index
    TEAM_INJURY_SEVERITY,        # raw severity sum across all listed teammates
    TEAM_INJURY_SEVERITY_ROLL5,  # smoothed 5-game rolling version
    -- opponent injury --
    OPP_INJURY_SEVERITY, OPP_STARS_OUT,
    -- composite interactions --
    INJURY_MIN_BOOST,            # absorbed_min × player's minutes consistency
    INJURY_PTS_BOOST,            # absorbed_fga × player's true shooting
    OPP_INJURY_ADVANTAGE         # opp depletion signal
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_DEFAULT_INJURY_CSV = Path("data") / "injury_data.csv"

# Season-average minutes threshold to classify a player as a "star"
# for lineup disruption and minute-redistribution purposes.
STAR_MIN_THRESHOLD = 28.0


# ── Load injury data from pipeline CSV ────────────────────────────────────────

def load_injury_data(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load the injury CSV written by data_pipeline.py.
    Returns an empty DataFrame if the file doesn't exist.
    """
    csv_path = Path(path) if path else _DEFAULT_INJURY_CSV
    if not csv_path.exists():
        print(f"[injury] {csv_path} not found — run data_pipeline.py first.")
        return pd.DataFrame()
    df = pd.read_csv(csv_path, parse_dates=["GAME_DATE"])
    print(f"[injury] Loaded {len(df):,} injury rows from {csv_path}")
    return df


def _empty_injury_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_all_injury_feature_cols())


# ── Main entry point ──────────────────────────────────────────────────────────

def build_injury_features(
    player_game_df: pd.DataFrame,
    injury_df: Optional[pd.DataFrame] = None,
    injury_csv: Optional[Path | str] = None,
    star_threshold_min: float = STAR_MIN_THRESHOLD,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Produce an injury feature DataFrame aligned to player_game_df.

    Parameters
    ----------
    player_game_df : pd.DataFrame
        Must have: PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION, GAME_DATE,
                   MIN, MATCHUP, FGA (for shot redistribution)
    injury_df : pd.DataFrame or None
        Pre-loaded injury data. If None, loads from injury_csv or the
        default data/injury_data.csv written by data_pipeline.py.
    injury_csv : Path or str or None
        Override path to the injury CSV.
    star_threshold_min : float
        Avg-minutes threshold to classify a player as a "star" for
        lineup-disruption and redistribution calculations.

    Returns
    -------
    pd.DataFrame with one row per (PLAYER_ID, GAME_DATE), all injury
    feature columns filled (NaN → 0).
    """
    df = player_game_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    # ── 1. Load injury data ────────────────────────────────────────────────────
    if injury_df is None:
        injury_df = load_injury_data(injury_csv)

    if injury_df is None or injury_df.empty:
        print("[injury] No injury data — returning zero-filled feature frame.")
        return _zero_injury_features(df)

    injury_df = injury_df.copy()
    injury_df["GAME_DATE"] = pd.to_datetime(injury_df["GAME_DATE"])
    if "TEAM" in injury_df.columns and "TEAM_ABBREVIATION" not in injury_df.columns:
        injury_df = injury_df.rename(columns={"TEAM": "TEAM_ABBREVIATION"})

    # ── 2. Compute per-player season-average MIN and FGA (lagged, no leakage) ─
    # These rolling averages are used to estimate how much of an absent
    # teammate's workload this player would absorb.
    df["_AVG_MIN"] = df.groupby("PLAYER_ID")["MIN"].transform(
        lambda x: x.shift(1).expanding(min_periods=3).mean()
    ).fillna(df["MIN"])

    df["_AVG_FGA"] = df.groupby("PLAYER_ID")["FGA"].transform(
        lambda x: x.shift(1).expanding(min_periods=3).mean()
    ).fillna(df.get("FGA", 0))

    # Team totals (also lagged) — used to compute each player's share
    team_avg_min = df.groupby(["GAME_DATE", "TEAM_ABBREVIATION"])["_AVG_MIN"].transform("sum")
    team_avg_fga = df.groupby(["GAME_DATE", "TEAM_ABBREVIATION"])["_AVG_FGA"].transform("sum")
    df["_MIN_SHARE"]  = df["_AVG_MIN"] / team_avg_min.clip(lower=1)
    df["_FGA_SHARE"]  = df["_AVG_FGA"] / team_avg_fga.clip(lower=1)
    df["_IS_STAR"]    = (df["_AVG_MIN"] >= star_threshold_min).astype(int)

    # ── 3. Own injury features ─────────────────────────────────────────────────
    if verbose:
        print("[injury] Building player-level own-injury features…")
    own_inj = _build_own_injury_features(df, injury_df)

    # ── 4. Teammate redistribution features ───────────────────────────────────
    if verbose:
        print("[injury] Building teammate redistribution features…")
    teammate_feats = _build_teammate_redistribution(df, injury_df, star_threshold_min)

    # ── 5. Opponent injury features ───────────────────────────────────────────
    if verbose:
        print("[injury] Building opponent injury features…")
    opp_feats = _build_opp_injury_features(df, injury_df, star_threshold_min)

    # ── 6. Merge all back onto the game log ───────────────────────────────────
    df = df.merge(own_inj,        on=["PLAYER_ID", "GAME_DATE"], how="left")
    df = df.merge(teammate_feats, on=["PLAYER_ID", "GAME_DATE"], how="left")
    df = df.merge(opp_feats,      on=["PLAYER_ID", "GAME_DATE"], how="left")

    # ── 7. Composite interactions ─────────────────────────────────────────────
    # INJURY_MIN_BOOST: expected extra minutes × player's minute-consistency
    # (consistent players get more of the redistributed minutes)
    min_consistency = df.groupby("PLAYER_ID")["MIN"].transform(
        lambda x: 1.0 / (x.shift(1).rolling(10, min_periods=3).std().clip(lower=0.5))
    ).fillna(1.0)
    df["INJURY_MIN_BOOST"] = df["TEAMMATE_MIN_ABSORBED"].fillna(0) * min_consistency

    # INJURY_PTS_BOOST: extra shot attempts × shooting efficiency
    # (efficient shooters extract more value from the extra attempts)
    true_shooting = df.groupby("PLAYER_ID")["PTS"].transform(
        lambda x: x.shift(1).expanding(min_periods=3).mean()
    ).fillna(10) / (2 * (
        df.groupby("PLAYER_ID")["FGA"].transform(
            lambda x: x.shift(1).expanding(min_periods=3).mean()
        ).fillna(10).clip(lower=1)
    ))
    df["INJURY_PTS_BOOST"] = df["TEAMMATE_FGA_ABSORBED"].fillna(0) * true_shooting.clip(0, 1)

    # OPP_INJURY_ADVANTAGE: combined signal for easier offensive matchup
    df["OPP_INJURY_ADVANTAGE"] = (
        df["OPP_INJURY_SEVERITY"].fillna(0)
        * (1.0 + df["OPP_STARS_OUT"].fillna(0) * 0.2)
    )

    # ── 8. Zero-fill and return only injury columns ────────────────────────────
    inj_cols = _all_injury_feature_cols()
    for c in inj_cols:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(0.0)

    if verbose:
        print(f"[injury] ✓ {len(inj_cols)} injury feature columns ready.")

    return df[["PLAYER_ID", "GAME_DATE"] + inj_cols].copy()


# ── Own injury features ────────────────────────────────────────────────────────

def _build_own_injury_features(
    df: pd.DataFrame,
    injury_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each player-game row, look up that player's own injury status on the
    game date and build rolling injury history features.
    """
    # Match injury report names to PLAYER_IDs via the game log
    name_to_id = (
        df[["PLAYER_ID", "PLAYER_NAME"]]
        .drop_duplicates("PLAYER_ID")
        .assign(NAME_KEY=lambda d: d["PLAYER_NAME"].str.upper().str.strip())
    )
    inj = injury_df.copy()
    inj["NAME_KEY"] = inj["PLAYER_NAME"].str.upper().str.strip()
    inj = inj.merge(name_to_id[["PLAYER_ID", "NAME_KEY"]], on="NAME_KEY", how="left")
    inj = inj.dropna(subset=["PLAYER_ID"])
    inj["PLAYER_ID"] = inj["PLAYER_ID"].astype(df["PLAYER_ID"].dtype)

    # Aggregate to one row per (player, date) — take worst status if multiple entries
    inj_agg = inj.groupby(["PLAYER_ID", "GAME_DATE"]).agg(
        PLAYER_STATUS_SCORE=("SEVERITY_SCORE", "max"),
        PLAYER_IS_OUT=("IS_OUT", "max"),
        PLAYER_IS_QUESTIONABLE=("IS_QUESTIONABLE", "max"),
    ).reset_index()

    merged = (
        df[["PLAYER_ID", "GAME_DATE"]]
        .merge(inj_agg, on=["PLAYER_ID", "GAME_DATE"], how="left")
        .fillna({"PLAYER_STATUS_SCORE": 0.0, "PLAYER_IS_OUT": 0, "PLAYER_IS_QUESTIONABLE": 0})
        .sort_values(["PLAYER_ID", "GAME_DATE"])
    )

    # Rolling injury risk over last 5 games (lagged — no same-game leakage)
    merged["PLAYER_INJURY_RISK_ROLL5"] = merged.groupby("PLAYER_ID")["PLAYER_STATUS_SCORE"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Days since last injury listing
    def _days_since(grp: pd.DataFrame) -> pd.Series:
        result = pd.Series(60.0, index=grp.index)  # default 60 = "long time ago"
        last_date = None
        for idx, row in grp.iterrows():
            if last_date is not None:
                result[idx] = min((row["GAME_DATE"] - last_date).days, 60)
            if row["PLAYER_STATUS_SCORE"] > 0:
                last_date = row["GAME_DATE"]
        return result

    dsi = []
    for _, grp in merged.groupby("PLAYER_ID"):
        dsi.append(_days_since(grp))
    merged["DAYS_SINCE_LAST_INJURY"] = pd.concat(dsi)

    # Return-from-injury flag: 1 if player was Out in the immediately prior game
    merged["RETURN_FROM_INJURY_FLAG"] = merged.groupby("PLAYER_ID")["PLAYER_IS_OUT"].transform(
        lambda x: x.shift(1).fillna(0).astype(int)
    )

    # Estimated games missed in last 30 days based on gaps in game log
    def _games_missed(grp: pd.DataFrame) -> pd.Series:
        result = pd.Series(0.0, index=grp.index)
        for i, (idx, row) in enumerate(grp.iterrows()):
            cutoff = row["GAME_DATE"] - pd.Timedelta(days=30)
            window = grp[(grp["GAME_DATE"] >= cutoff) & (grp["GAME_DATE"] < row["GAME_DATE"])]
            if len(window) >= 2:
                span_days = (row["GAME_DATE"] - window["GAME_DATE"].min()).days
                expected  = span_days / 2.3   # ~3.5 games/week = ~1 game/2.3 days
                result.iloc[i] = max(0, round(expected - len(window)))
        return result

    gmr = []
    for _, grp in merged.groupby("PLAYER_ID"):
        gmr.append(_games_missed(grp))
    merged["INJURY_GAMES_MISSED_RECENT"] = pd.concat(gmr)

    return merged[["PLAYER_ID", "GAME_DATE",
                   "PLAYER_STATUS_SCORE", "PLAYER_IS_OUT", "PLAYER_IS_QUESTIONABLE",
                   "PLAYER_INJURY_RISK_ROLL5", "DAYS_SINCE_LAST_INJURY",
                   "RETURN_FROM_INJURY_FLAG", "INJURY_GAMES_MISSED_RECENT"]]


# ── Teammate redistribution features ─────────────────────────────────────────

def _build_teammate_redistribution(
    df: pd.DataFrame,
    injury_df: pd.DataFrame,
    star_threshold_min: float,
) -> pd.DataFrame:
    """
    For each healthy player-game, compute how many minutes and shot attempts
    they are expected to absorb from injured/absent teammates.

    Core logic:
    -----------
    For each Out teammate T on team K on date D:
        - T's expected contribution = T's season-avg minutes / FGA (lagged)
        - This player P absorbs a share = P's avg_min / sum(other healthy players' avg_min)

    This correctly handles the case where multiple players are out —
    the remaining players share the full absent workload.
    """
    # Build a lookup: for each (team, date), which players are listed Out?
    # Match by PLAYER_NAME since injury report doesn't have PLAYER_ID.
    name_to_stats = (
        df.groupby("PLAYER_NAME")
        .apply(lambda g: g.sort_values("GAME_DATE").set_index("GAME_DATE")[["PLAYER_ID", "TEAM_ABBREVIATION", "_AVG_MIN", "_AVG_FGA", "_IS_STAR"]])
        .reset_index()
    )

    inj_out = injury_df[injury_df["IS_OUT"] == 1][["GAME_DATE", "PLAYER_NAME", "TEAM_ABBREVIATION"]].copy()

    # For each out player, look up their season-average stats at that date
    # We use a snapshot join: find the most recent _AVG_MIN before the game date.
    # Since we already have _AVG_MIN per (PLAYER_ID, GAME_DATE) in df, we look it up directly.
    player_stats_snapshot = df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_DATE", "_AVG_MIN", "_AVG_FGA", "_IS_STAR"]].copy()

    # Join out players to their stats
    inj_out = inj_out.merge(
        player_stats_snapshot.rename(columns={
            "PLAYER_NAME": "PLAYER_NAME",
            "_AVG_MIN":    "ABSENT_AVG_MIN",
            "_AVG_FGA":    "ABSENT_AVG_FGA",
            "_IS_STAR":    "ABSENT_IS_STAR",
        })[["PLAYER_NAME", "GAME_DATE", "TEAM_ABBREVIATION", "ABSENT_AVG_MIN", "ABSENT_AVG_FGA", "ABSENT_IS_STAR"]],
        on=["PLAYER_NAME", "GAME_DATE", "TEAM_ABBREVIATION"],
        how="left",
    )
    inj_out["ABSENT_AVG_MIN"]  = inj_out["ABSENT_AVG_MIN"].fillna(20.0)  # fallback estimate
    inj_out["ABSENT_AVG_FGA"]  = inj_out["ABSENT_AVG_FGA"].fillna(7.0)
    inj_out["ABSENT_IS_STAR"]  = inj_out["ABSENT_IS_STAR"].fillna(0)

    # Aggregate to team-date level: total absent minutes, FGA, and star count
    team_absent = inj_out.groupby(["GAME_DATE", "TEAM_ABBREVIATION"]).agg(
        TOTAL_ABSENT_MIN=("ABSENT_AVG_MIN", "sum"),
        TOTAL_ABSENT_FGA=("ABSENT_AVG_FGA", "sum"),
        TEAM_STARS_OUT=("ABSENT_IS_STAR", "sum"),
        TEAM_INJURY_SEVERITY=("ABSENT_AVG_MIN", "count"),  # count of Out players as proxy
    ).reset_index()

    # Also compute LINEUP_DISRUPTION_SCORE = sum of absent star avg-min
    star_disruption = (
        inj_out[inj_out["ABSENT_IS_STAR"] == 1]
        .groupby(["GAME_DATE", "TEAM_ABBREVIATION"])["ABSENT_AVG_MIN"]
        .sum()
        .reset_index()
        .rename(columns={"ABSENT_AVG_MIN": "LINEUP_DISRUPTION_SCORE"})
    )
    team_absent = team_absent.merge(star_disruption, on=["GAME_DATE", "TEAM_ABBREVIATION"], how="left")
    team_absent["LINEUP_DISRUPTION_SCORE"] = team_absent["LINEUP_DISRUPTION_SCORE"].fillna(0.0)

    # Now compute per-player redistribution:
    # Each player absorbs:  absent_total × (their_min_share_of_healthy_rotation)
    # healthy_rotation_min = team_total_avg_min - total_absent_min
    result = df[["PLAYER_ID", "GAME_DATE", "TEAM_ABBREVIATION",
                 "_AVG_MIN", "_AVG_FGA", "_MIN_SHARE", "_FGA_SHARE"]].merge(
        team_absent, on=["GAME_DATE", "TEAM_ABBREVIATION"], how="left"
    )
    result = result.fillna({
        "TOTAL_ABSENT_MIN":       0.0,
        "TOTAL_ABSENT_FGA":       0.0,
        "TEAM_STARS_OUT":         0.0,
        "TEAM_INJURY_SEVERITY":   0.0,
        "LINEUP_DISRUPTION_SCORE":0.0,
    })

    # Healthy rotation total (denominator for redistribution share)
    team_total_avg_min = df.groupby(["GAME_DATE", "TEAM_ABBREVIATION"])["_AVG_MIN"].transform("sum")
    healthy_rotation = (team_total_avg_min - result["TOTAL_ABSENT_MIN"]).clip(lower=1)
    healthy_share_min = result["_AVG_MIN"] / healthy_rotation
    healthy_share_fga = result["_AVG_FGA"] / (
        df.groupby(["GAME_DATE", "TEAM_ABBREVIATION"])["_AVG_FGA"].transform("sum")
        - result["TOTAL_ABSENT_FGA"]
    ).clip(lower=1)

    result["TEAMMATE_MIN_ABSORBED"] = result["TOTAL_ABSENT_MIN"] * healthy_share_min.clip(0, 1)
    result["TEAMMATE_FGA_ABSORBED"] = result["TOTAL_ABSENT_FGA"] * healthy_share_fga.clip(0, 1)

    # Rolling 5-game smoothed team severity (reduce noise from single-day reports)
    result = result.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    result["TEAM_INJURY_SEVERITY_ROLL5"] = result.groupby("TEAM_ABBREVIATION")["TEAM_INJURY_SEVERITY"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    return result[["PLAYER_ID", "GAME_DATE",
                   "TEAMMATE_MIN_ABSORBED", "TEAMMATE_FGA_ABSORBED",
                   "TEAM_STARS_OUT", "LINEUP_DISRUPTION_SCORE",
                   "TEAM_INJURY_SEVERITY", "TEAM_INJURY_SEVERITY_ROLL5"]]


# ── Opponent injury features ───────────────────────────────────────────────────

def _build_opp_injury_features(
    df: pd.DataFrame,
    injury_df: pd.DataFrame,
    star_threshold_min: float,
) -> pd.DataFrame:
    """
    For each player-game, look up the opposing team's injury severity.
    """
    def _get_opp(matchup: str, own_team: str) -> Optional[str]:
        if not isinstance(matchup, str):
            return None
        parts = matchup.strip().split(" ")
        # format: "LAL vs. GSW" or "LAL @ GSW"
        return parts[-1] if len(parts) >= 3 else None

    tmp = df[["PLAYER_ID", "GAME_DATE", "MATCHUP", "TEAM_ABBREVIATION"]].copy()
    tmp["OPPONENT"] = tmp.apply(
        lambda r: _get_opp(r["MATCHUP"], r["TEAM_ABBREVIATION"]), axis=1
    )

    inj_out = injury_df[injury_df["IS_OUT"] == 1].copy()
    player_stats_snapshot = df[["PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_DATE", "_AVG_MIN"]].copy()
    inj_out = inj_out.merge(
        player_stats_snapshot.rename(columns={"_AVG_MIN": "ABSENT_AVG_MIN"}),
        on=["PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_DATE"],
        how="left",
    )
    inj_out["ABSENT_AVG_MIN"] = inj_out["ABSENT_AVG_MIN"].fillna(20.0)
    inj_out["ABSENT_IS_STAR"] = (inj_out["ABSENT_AVG_MIN"] >= star_threshold_min).astype(int)

    opp_agg = inj_out.groupby(["GAME_DATE", "TEAM_ABBREVIATION"]).agg(
        OPP_INJURY_SEVERITY=("SEVERITY_SCORE", "sum"),
        OPP_STARS_OUT=("ABSENT_IS_STAR", "sum"),
    ).reset_index().rename(columns={"TEAM_ABBREVIATION": "OPPONENT"})

    result = tmp.merge(opp_agg, on=["OPPONENT", "GAME_DATE"], how="left")
    result = result.fillna({"OPP_INJURY_SEVERITY": 0.0, "OPP_STARS_OUT": 0.0})
    return result[["PLAYER_ID", "GAME_DATE", "OPP_INJURY_SEVERITY", "OPP_STARS_OUT"]]


# ── Zero-fill fallback ─────────────────────────────────────────────────────────

def _zero_injury_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df[["PLAYER_ID", "GAME_DATE"]].copy()
    for col in _all_injury_feature_cols():
        result[col] = 0.0
    return result


def _all_injury_feature_cols() -> list[str]:
    return [
        # Own injury status
        "PLAYER_STATUS_SCORE",
        "PLAYER_IS_OUT",
        "PLAYER_IS_QUESTIONABLE",
        "PLAYER_INJURY_RISK_ROLL5",
        "DAYS_SINCE_LAST_INJURY",
        "RETURN_FROM_INJURY_FLAG",
        "INJURY_GAMES_MISSED_RECENT",
        # Teammate redistribution (the core signal)
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


# ── Public merge helper ────────────────────────────────────────────────────────

def merge_injury_features_into_df(
    engineered_df: pd.DataFrame,
    injury_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the output of build_injury_features() into an already-engineered
    feature DataFrame (from minutes_feature_engineering.py or
    points_feature_engineering.py).
    """
    engineered_df = engineered_df.copy()
    engineered_df["GAME_DATE"] = pd.to_datetime(engineered_df["GAME_DATE"])
    injury_features_df["GAME_DATE"] = pd.to_datetime(injury_features_df["GAME_DATE"])

    merged = engineered_df.merge(
        injury_features_df,
        on=["PLAYER_ID", "GAME_DATE"],
        how="left",
    )
    for col in _all_injury_feature_cols():
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)
        else:
            merged[col] = 0.0
    return merged


# ── CLI / smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    inj_df = load_injury_data()
    if inj_df.empty:
        print("No injury data found. Run:  python data_pipeline.py --start YYYY-MM-DD --end YYYY-MM-DD")
    else:
        print(f"\n{len(inj_df):,} injury rows across {inj_df['GAME_DATE'].nunique()} dates")
        print(f"Date range: {inj_df['GAME_DATE'].min().date()} → {inj_df['GAME_DATE'].max().date()}")
        print(f"\nStatus breakdown:\n{inj_df['STATUS'].value_counts().to_string()}")
        print(f"\nSample rows:\n{inj_df.head(10).to_string(index=False)}")