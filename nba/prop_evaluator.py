"""
prop_evaluator.py — NBA Prop Evaluator (v4)

Evaluates a player points over/under prop using the two-stage model.
Updated to use data_pipeline.load_and_normalize() for new column schema.

Usage
-----
    python prop_evaluator.py

Or programmatically:
    from prop_evaluator import PropEvaluator
    ev = PropEvaluator()
    result = ev.evaluate_from_dataset(
        player_name="Stephen Curry",
        opponent="MEM",
        is_home=1,
        game_date="2025-03-29",
        line=28.5,
        side="over",
        american_odds=-110,
    )
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import DATA_PATH, MIN_MODEL_PKL, PTS_MODEL_PKL
from data_pipeline import load_and_normalize, fetch_injury_range, load_injury_data
from injury_features import build_injury_features, _all_injury_feature_cols
from minutes_predictor import MinutesPredictor, FEATURE_COLS as MIN_FEATURES
from points_predictor import PointsPredictor, FEATURE_COLS_PTS as PTS_FEATURES

# Canonical stat columns used by FeatureBuilder
RAW_STAT_COLS = [
    "MIN", "PTS", "FGA", "FGM", "FG_PCT", "FG3A", "FG3M", "FG3_PCT",
    "FTA", "FTM", "FT_PCT", "REB", "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS",
]


# ── Odds utilities ─────────────────────────────────────────────────────────────

def american_to_implied(american: int) -> float:
    return abs(american) / (abs(american) + 100) if american < 0 else 100 / (american + 100)


def american_to_decimal(american: int) -> float:
    """Decimal odds = total returned per $1 staked (includes stake)."""
    return 1 + 100 / abs(american) if american < 0 else 1 + american / 100


def decimal_to_american(decimal: float) -> int:
    """Convert decimal odds to American. decimal must be > 1."""
    if decimal >= 2.0:
        american = (decimal - 1) * 100
    else:
        american = -100 / (decimal - 1)
    return int(round(american / 5) * 5)


def multiplier_to_decimal(multiplier: float) -> float:
    """
    Convert a payout multiplier to decimal odds.
    A 3.5x multiplier means $10 bet returns $35 total.
    """
    return multiplier


def multiplier_to_implied(multiplier: float) -> float:
    """Breakeven probability implied by a payout multiplier."""
    return 1.0 / multiplier_to_decimal(multiplier)


def implied_to_american(prob: float) -> int:
    american = -(prob / (1 - prob)) * 100 if prob >= 0.5 else ((1 - prob) / prob) * 100
    return int(round(american / 5) * 5)


def implied_to_multiplier(prob: float) -> float:
    """Fair payout multiplier (profit per $1) for a given win probability."""
    if prob <= 0:
        return float("inf")
    return round((1.0 / prob) - 1.0, 3)


def resolve_parlay_payout(
    parlay_american_odds,
    parlay_multiplier,
):
    """
    Accept either American odds or a payout multiplier; return
    (book_break_even, book_decimal, book_american, book_multiplier).
    Exactly one of the two inputs must be non-None.
    """
    if parlay_american_odds is not None and parlay_multiplier is not None:
        raise ValueError("Provide either parlay_american_odds or parlay_multiplier, not both.")
    if parlay_american_odds is None and parlay_multiplier is None:
        raise ValueError("Provide one of parlay_american_odds or parlay_multiplier.")

    if parlay_american_odds is not None:
        book_decimal    = american_to_decimal(int(parlay_american_odds))
        book_american   = int(parlay_american_odds)
        book_multiplier = round(book_decimal - 1.0, 4)
    else:
        book_decimal    = multiplier_to_decimal(float(parlay_multiplier))
        book_american   = decimal_to_american(book_decimal)
        book_multiplier = float(parlay_multiplier)

    book_break_even = 1.0 / book_decimal
    return book_break_even, book_decimal, book_american, book_multiplier


# ── Dataset loader ─────────────────────────────────────────────────────────────

class DatasetLoader:
    """
    Loads player history from the master CSV using normalize_boxscore
    so it works with both old UPPER_CASE and new camelCase schemas.
    """

    def __init__(self, data_path: Path | None = None):
        path = data_path or DATA_PATH
        print(f"Loading dataset from {path}...", end=" ", flush=True)
        self._df = (
            load_and_normalize(path)
            .pipe(lambda d: d[d["MIN"] > 2])
            .sort_values(["PLAYER_ID", "GAME_DATE"])
            .reset_index(drop=True)
        )
        print(f"done ({len(self._df):,} rows).")
        # Injury data: loaded once from the cached CSV, keyed by date on demand
        self._injury_df: Optional[pd.DataFrame] = load_injury_data()
        self._injury_cache: dict[str, dict] = {}  # date_str → injury feature dict

    def get_injury_features_for_date(
        self,
        player_name: str,
        team_abbr: str,
        opponent: str,
        game_date: str,
    ) -> dict:
        """
        Return injury feature values for a specific player/game, fetching the
        injury report for game_date from the API if not already in the cached CSV.

        Returns a dict mapping each injury feature name to its value (0.0 default).
        """
        date_str = str(game_date)
        zero = {c: 0.0 for c in _all_injury_feature_cols()}

        # Try to fetch if the date isn't in the cached CSV
        inj_df = self._injury_df
        if inj_df is None or inj_df.empty or (
            not (pd.to_datetime(inj_df["GAME_DATE"]) == pd.to_datetime(date_str)).any()
        ):
            print(f"  [injury] No cached data for {date_str} — fetching from API…")
            try:
                fetched = fetch_injury_range(date_str, date_str)
                if not fetched.empty:
                    if inj_df is not None and not inj_df.empty:
                        inj_df = pd.concat([inj_df, fetched], ignore_index=True).drop_duplicates(
                            subset=["GAME_DATE", "PLAYER_NAME", "TEAM_ABBREVIATION"], keep="last"
                        )
                    else:
                        inj_df = fetched
                    self._injury_df = inj_df
            except Exception as e:
                print(f"  [injury] Fetch failed: {e} — using zeros")
                return zero

        if inj_df is None or inj_df.empty:
            return zero

        # Build injury features using the player game history subset
        cutoff = pd.to_datetime(date_str)
        player_hist = self._df[
            (self._df["PLAYER_NAME"] == player_name) &
            (self._df["GAME_DATE"] < cutoff)
        ].tail(30).copy()  # last 30 games is enough context

        if player_hist.empty:
            return zero

        # Add a synthetic "upcoming game" row to get features for this date
        last = player_hist.iloc[-1].copy()
        last["GAME_DATE"] = cutoff
        last["MATCHUP"] = f"{team_abbr} vs. {opponent}" if last.get("IS_HOME", 1) else f"{team_abbr} @ {opponent}"
        synth_df = pd.concat([player_hist, last.to_frame().T], ignore_index=True)
        synth_df["GAME_DATE"] = pd.to_datetime(synth_df["GAME_DATE"])

        try:
            inj_feats = build_injury_features(
                player_game_df=synth_df,
                injury_df=inj_df,
                verbose=False,
            )
            # Take the last row (the synthetic upcoming game)
            row = inj_feats[inj_feats["GAME_DATE"] == cutoff]
            if row.empty:
                return zero
            result = {}
            for col in _all_injury_feature_cols():
                result[col] = float(row[col].iloc[0]) if col in row.columns else 0.0
            return result
        except Exception as e:
            print(f"  [injury] Feature build failed: {e} — using zeros")
            return zero

    def get_player_games(self, player_name: str, before_date: str | date) -> pd.DataFrame:
        cutoff = pd.to_datetime(before_date)
        mask = (self._df["PLAYER_NAME"] == player_name) & (self._df["GAME_DATE"] < cutoff)
        return self._df[mask].sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

    def get_opponent_stats(self, opp_abbr: str, before_date: str | date) -> dict:
        cutoff = pd.to_datetime(before_date)
        mask = (
            self._df["MATCHUP"].str.contains(opp_abbr, na=False)
            & (self._df["TEAM_ABBREVIATION"] != opp_abbr)
            & (self._df["GAME_DATE"] < cutoff)
        )
        opp_df = self._df[mask]
        if opp_df.empty:
            return {}
        return {
            "avg_pts_allowed": float(opp_df["PTS"].mean()),
            "avg_fga_allowed": float(opp_df["FGA"].mean()),
            "avg_3pt_allowed": float(opp_df["FG3A"].mean()),
            "avg_min_allowed": float(opp_df["MIN"].mean()),
            "strength":        float(-opp_df["PLUS_MINUS"].mean()),
        }

    def derive_context(self, player_name: str, game_date: str | date, is_home: int, opponent: str) -> dict:
        gd = pd.to_datetime(game_date)
        hist = self.get_player_games(player_name, gd)
        if hist.empty:
            return self._default_context(is_home, game_date)

        days_rest = int(min((gd - hist["GAME_DATE"].iloc[0]).days, 10))
        season_start = pd.Timestamp(gd.year if gd.month >= 8 else gd.year - 1, 8, 1)
        szn_games = hist[hist["GAME_DATE"] >= season_start]
        game_num = int(len(szn_games)) + 1
        win_pct = float((szn_games["WL"] == "W").mean()) if len(szn_games) else 0.5
        ten_days_ago = gd - pd.Timedelta(days=10)
        cum_min_10d = float(hist[hist["GAME_DATE"] >= ten_days_ago]["MIN"].sum())
        team_abbr = hist["TEAM_ABBREVIATION"].iloc[0]
        team_min_share, min_rank, pts_rank, team_avg_pts = self._team_context(player_name, team_abbr, gd)

        return {
            "is_home": is_home, "game_date": str(game_date), "game_num": game_num,
            "days_rest": days_rest, "win_pct_szn": win_pct,
            "team_min_share_roll5": team_min_share, "min_rank_in_team": min_rank,
            "pts_rank_in_team": pts_rank, "cumulative_min_10d": cum_min_10d,
            "team_avg_pts": team_avg_pts,
        }

    def _default_context(self, is_home: int, game_date: str | date) -> dict:
        return {
            "is_home": is_home, "game_date": str(game_date), "game_num": 1,
            "days_rest": 2, "win_pct_szn": 0.5, "team_min_share_roll5": 0.14,
            "min_rank_in_team": 1, "pts_rank_in_team": 1.0,
            "cumulative_min_10d": 0.0, "team_avg_pts": 110.0,
        }

    def _team_context(self, player_name: str, team_abbr: str, gd: pd.Timestamp) -> tuple:
        team_mask = (self._df["TEAM_ABBREVIATION"] == team_abbr) & (self._df["GAME_DATE"] < gd)
        team_df = self._df[team_mask]
        recent_dates = sorted(team_df["GAME_DATE"].unique())[-5:]
        if not len(recent_dates):
            return 0.14, 1.0, 1.0, 110.0

        td = team_df[team_df["GAME_DATE"].isin(recent_dates)].copy()

        def _rank(grp, col, rank_col):
            grp = grp.copy()
            grp[rank_col] = grp[col].rank(ascending=False, method="dense")
            return grp

        team_total_min = td.groupby("GAME_DATE")["MIN"].transform("sum")
        td["MIN_SHARE"] = td["MIN"] / team_total_min.clip(lower=1)
        player_td = td[td["PLAYER_NAME"] == player_name]

        team_min_share = float(player_td["MIN_SHARE"].mean()) if not player_td.empty else 0.14
        ranked_min = td.groupby("GAME_DATE", group_keys=False).apply(lambda g: _rank(g, "MIN", "RANK"))
        player_min_ranked = ranked_min[ranked_min["PLAYER_NAME"] == player_name]
        min_rank = float(player_min_ranked["RANK"].mean()) if not player_min_ranked.empty else 1.0
        ranked_pts = td.groupby("GAME_DATE", group_keys=False).apply(lambda g: _rank(g, "PTS", "PTS_RANK"))
        player_pts_ranked = ranked_pts[ranked_pts["PLAYER_NAME"] == player_name]
        pts_rank = float(player_pts_ranked["PTS_RANK"].mean()) if not player_pts_ranked.empty else 1.0
        team_avg_pts = float(td.groupby("GAME_DATE")["PTS"].sum().mean())

        return team_min_share, min_rank, pts_rank, team_avg_pts

    def build_game_rows(self, player_name: str, game_date: str | date, n_recent: int = 10) -> tuple:
        gd = pd.to_datetime(game_date)
        hist = self.get_player_games(player_name, gd)
        season_start = pd.Timestamp(gd.year if gd.month >= 8 else gd.year - 1, 8, 1)
        szn = hist[hist["GAME_DATE"] >= season_start]

        def to_rows(df: pd.DataFrame) -> list[dict]:
            return [
                {c: float(row[c]) if c in row.index else 0.0 for c in RAW_STAT_COLS}
                for _, row in df.iterrows()
            ]

        return to_rows(hist.head(n_recent)), to_rows(szn)


# ── Feature builder ────────────────────────────────────────────────────────────

class FeatureBuilder:
    def __init__(self, games, season_games, context, opp_stats=None, injury_features=None):
        self.games = games
        self.season_games = season_games
        self.context = context
        self.opp_stats = opp_stats or {}
        self.injury_features = injury_features or {}
        self.df   = pd.DataFrame(list(reversed(games)), columns=RAW_STAT_COLS)
        self.s_df = pd.DataFrame(list(reversed(season_games)), columns=RAW_STAT_COLS)
        self._enrich(self.df)
        self._enrich(self.s_df)

    def _enrich(self, df):
        df["EFFICIENCY_PER_MIN"] = (
            df["PTS"] + df["REB"] + df["AST"] + df["STL"] + df["BLK"]
            - df["TOV"] - (df["FGA"] - df["FGM"])
        ) / df["MIN"].clip(lower=0.1)
        df["TRUE_SHOOTING"]    = df["PTS"] / (2 * (df["FGA"] + 0.44 * df["FTA"]).clip(lower=0.1))
        df["USAGE_RATE_PROXY"] = (df["FGA"] + 0.44 * df["FTA"] + df["TOV"]) / df["MIN"].clip(lower=0.1)
        df["AST_TO_RATIO"]     = df["AST"] / (df["TOV"] + 0.1)
        df["STOCKS"]           = df["STL"] + df["BLK"]
        df["FGA_PER_MIN"]      = df["FGA"] / df["MIN"].clip(lower=0.1)
        df["FG3A_PER_MIN"]     = df["FG3A"] / df["MIN"].clip(lower=0.1)
        df["FTA_PER_MIN"]      = df["FTA"] / df["MIN"].clip(lower=0.1)
        df["THREE_PT_RATE"]    = df["FG3A"] / (df["FGA"] + 0.01)
        df["FT_RATE"]          = df["FTA"] / (df["FGA"] + 0.01)
        df["MID_RANGE_RATE"]   = (1 - df["THREE_PT_RATE"] - df["FT_RATE"].clip(0, 1)).clip(0, 1)

    def _roll(self, col, w, df=None):
        d = df if df is not None else self.df
        vals = d[col].dropna().values if col in d.columns else np.array([])
        return float(np.mean(vals[-w:])) if len(vals) else 0.0

    def _roll_std(self, col, w):
        vals = self.df[col].dropna().values if col in self.df.columns else np.array([])
        return float(np.std(vals[-w:], ddof=1)) if len(vals) >= 2 else 0.0

    def _season_mean(self, col):
        return float(self.s_df[col].dropna().mean()) if col in self.s_df.columns else 0.0

    def _season_std(self, col):
        vals = self.s_df[col].dropna().values if col in self.s_df.columns else np.array([])
        return float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0

    def _slope(self, col, w=5):
        vals = self.df[col].dropna().values[-w:] if col in self.df.columns else np.array([])
        if len(vals) < 3:
            return 0.0
        return float(np.polyfit(np.arange(len(vals)), vals, 1)[0])

    def _context_features(self):
        ctx = self.context
        gd_raw = ctx.get("game_date", date.today())
        gd = datetime.strptime(gd_raw, "%Y-%m-%d").date() if isinstance(gd_raw, str) else gd_raw
        dr = int(ctx.get("days_rest", 2))
        is_home = int(bool(ctx.get("is_home", True)))
        month = gd.month
        return {
            "GAME_NUM_IN_SEASON": int(ctx.get("game_num", len(self.season_games) + 1)),
            "IS_HOME": is_home, "DAY_OF_WEEK": gd.weekday(),
            "DAYS_REST": dr, "IS_BACK_TO_BACK": int(dr == 1),
            "IS_3_IN_4": int(dr <= 1), "LATE_SEASON": int(month in [3, 4]),
            "MONTH": month, "BLOWOUT_FLAG": 0, "CLOSE_GAME_FLAG": 0,
            "BTB_ROAD": int(dr == 1 and not is_home),
            "BTB_HOME": int(dr == 1 and bool(is_home)),
            "EXTENDED_REST": int(dr >= 4),
        }

    def build(self) -> dict:
        ctx_feat = self._context_features()
        is_home  = ctx_feat["IS_HOME"]
        dr       = ctx_feat["DAYS_REST"]
        opp      = self.opp_stats

        min_szn = self._season_mean("MIN"); pts_szn = self._season_mean("PTS")
        pts_std_s = self._season_std("PTS"); fga_szn = self._season_mean("FGA")
        reb_szn = self._season_mean("REB"); ast_szn = self._season_mean("AST")
        stl_szn = self._season_mean("STL"); blk_szn = self._season_mean("BLK")
        fg_szn = self._season_mean("FG_PCT"); fg3_szn = self._season_mean("FG3_PCT")
        ft_szn = self._season_mean("FT_PCT"); ts_szn = self._season_mean("TRUE_SHOOTING")
        three_szn = self._season_mean("THREE_PT_RATE"); ft_r_szn = self._season_mean("FT_RATE")
        min_cons = self._season_std("MIN"); pts_cv = pts_std_s / max(pts_szn, 0.5)

        scorer_score = pts_szn + fga_szn * 0.5; playmaker_score = ast_szn * 2
        defender_score = (stl_szn + blk_szn) * 2; rebounder_score = reb_szn

        min_r3, min_r5, min_r10 = self._roll("MIN", 3), self._roll("MIN", 5), self._roll("MIN", 10)
        pts_r3, pts_r5, pts_r10 = self._roll("PTS", 3), self._roll("PTS", 5), self._roll("PTS", 10)
        eff_r5 = self._roll("EFFICIENCY_PER_MIN", 5); ts_r3 = self._roll("TRUE_SHOOTING", 3)
        ts_r5 = self._roll("TRUE_SHOOTING", 5); ts_r10 = self._roll("TRUE_SHOOTING", 10)
        usg_r5 = self._roll("USAGE_RATE_PROXY", 5); pm_r5 = self._roll("PLUS_MINUS", 5)
        pf_r5 = self._roll("PF", 5)
        fga_pm_r3 = self._roll("FGA_PER_MIN", 3); fga_pm_r5 = self._roll("FGA_PER_MIN", 5)
        fga_pm_r10 = self._roll("FGA_PER_MIN", 10); fg3a_pm_r5 = self._roll("FG3A_PER_MIN", 5)
        fta_pm_r5 = self._roll("FTA_PER_MIN", 5); three_r5 = self._roll("THREE_PT_RATE", 5)
        ft_r_r5 = self._roll("FT_RATE", 5); mid_r5 = self._roll("MID_RANGE_RATE", 5)
        fg_r3 = self._roll("FG_PCT", 3); fg_r5 = self._roll("FG_PCT", 5)
        fg_r10 = self._roll("FG_PCT", 10); fg3_r5 = self._roll("FG3_PCT", 5)
        fg3_r10 = self._roll("FG3_PCT", 10); ft_r5 = self._roll("FT_PCT", 5)
        min_r3_std = self._roll_std("MIN", 3); min_r5_std = self._roll_std("MIN", 5)
        min_r10_std = self._roll_std("MIN", 10)
        pts_r3_std = self._roll_std("PTS", 3); pts_r5_std = self._roll_std("PTS", 5)
        pts_r10_std = self._roll_std("PTS", 10)
        pts_zscore = (pts_r5 - pts_szn) / max(pts_std_s, 0.1)
        hot_streak = int(sum(1 for g in self.games[:3] if g["PTS"] > pts_szn) >= 2)
        cold_streak = int(sum(1 for g in self.games[:3] if g["PTS"] < pts_szn * 0.7) >= 2)

        win_pct       = float(self.context.get("win_pct_szn", 0.5))
        team_min_share = float(self.context.get("team_min_share_roll5", 0.14))
        min_rank      = int(self.context.get("min_rank_in_team", 1))
        cum_min_10d   = float(self.context.get("cumulative_min_10d", sum(g["MIN"] for g in self.games[:5])))
        team_pts_share = pts_r5 / max(self.context.get("team_avg_pts", 110) * team_min_share * 5, 0.1)

        opp_avg_min = float(opp.get("avg_min_allowed", min_szn))
        opp_strength = float(opp.get("strength", 0.0))
        opp_pts_all  = float(opp.get("avg_pts_allowed", pts_szn))
        opp_fga_all  = float(opp.get("avg_fga_allowed", fga_szn))
        opp_3pt_all  = float(opp.get("avg_3pt_allowed", self._season_mean("FG3A")))

        feat: dict = {
            **ctx_feat,
            "MIN_ROLL3": min_r3, "MIN_ROLL5": min_r5, "MIN_ROLL10": min_r10,
            "MIN_ROLL3_STD": min_r3_std, "MIN_ROLL5_STD": min_r5_std, "MIN_ROLL10_STD": min_r10_std,
            "MIN_ROLL5_MEDIAN": min_r5, "MIN_ROLL10_MEDIAN": min_r10,
            "MIN_TREND_5G": self._slope("MIN", 5),
            "MIN_EWM5": min_r5, "MIN_EWM10": min_r10,
            "MIN_SZN_AVG": min_szn, "MIN_CONSISTENCY": min_cons,
            "MIN_VOL_RATIO": min_r5_std / max(min_r5, 1),
            "PTS_ROLL3": pts_r3, "PTS_ROLL5": pts_r5, "PTS_ROLL10": pts_r10,
            "PTS_EWM3": pts_r3, "PTS_EWM5": pts_r5,
            "PTS_ROLL3_STD": pts_r3_std, "PTS_ROLL5_STD": pts_r5_std, "PTS_ROLL10_STD": pts_r10_std,
            "PTS_TREND_5G": self._slope("PTS", 5), "PTS_CV_SZN": pts_cv,
            "HOT_STREAK": hot_streak, "COLD_STREAK": cold_streak,
            "PTS_MOMENTUM_3G": (pts_r3 - pts_szn) / max(pts_std_s, 0.1),
            "FGA_ROLL5": self._roll("FGA", 5), "FG_PCT_ROLL5": fg_r5,
            "FG3_PCT_ROLL5": fg3_r5, "FT_PCT_ROLL5": ft_r5,
            "REB_ROLL5": self._roll("REB", 5), "AST_ROLL5": self._roll("AST", 5),
            "STL_ROLL5": self._roll("STL", 5), "BLK_ROLL5": self._roll("BLK", 5),
            "TOV_ROLL5": self._roll("TOV", 5), "PF_ROLL5": pf_r5,
            "PLUS_MINUS_ROLL5": pm_r5,
            "EFFICIENCY_PER_MIN_ROLL5": eff_r5, "EFFICIENCY_PER_MIN_EWM5": eff_r5,
            "USAGE_RATE_PROXY_ROLL5": usg_r5, "USAGE_RATE_PROXY_EWM5": usg_r5,
            "TRUE_SHOOTING_ROLL5": ts_r5, "TRUE_SHOOTING_ROLL3": ts_r3,
            "TRUE_SHOOTING_ROLL10": ts_r10, "TRUE_SHOOTING_SZN_AVG": ts_szn,
            "TRUE_SHOOTING_EWM5": ts_r5,
            "STOCKS_ROLL5": self._roll("STOCKS", 5), "AST_TO_RATIO_ROLL5": self._roll("AST_TO_RATIO", 5),
            "MIN_SZN_AVG": min_szn, "PTS_SZN_AVG": pts_szn, "FGA_SZN_AVG": fga_szn,
            "REB_SZN_AVG": reb_szn, "AST_SZN_AVG": ast_szn, "STL_SZN_AVG": stl_szn,
            "BLK_SZN_AVG": blk_szn, "FG_PCT_SZN_AVG": fg_szn, "FG3_PCT_SZN_AVG": fg3_szn,
            "FT_PCT_SZN_AVG": ft_szn, "FG_PCT_EWM5": fg_r5, "FG3_PCT_EWM5": fg3_r5,
            "FT_PCT_EWM5": ft_r5, "FG_PCT_ROLL3": fg_r3, "FG_PCT_ROLL10": fg_r10,
            "FG3_PCT_ROLL10": fg3_r10,
            "SCORER_SCORE": scorer_score, "PLAYMAKER_SCORE": playmaker_score,
            "DEFENDER_SCORE": defender_score, "REBOUNDER_SCORE": rebounder_score,
            "FGA_PER_MIN_ROLL3": fga_pm_r3, "FGA_PER_MIN_ROLL5": fga_pm_r5,
            "FGA_PER_MIN_ROLL10": fga_pm_r10, "FGA_PER_MIN_EWM5": fga_pm_r5,
            "FG3A_PER_MIN_ROLL5": fg3a_pm_r5, "FTA_PER_MIN_ROLL5": fta_pm_r5,
            "THREE_PT_RATE_ROLL5": three_r5, "THREE_PT_RATE_SZN": three_szn,
            "FT_RATE_ROLL5": ft_r_r5, "FT_RATE_SZN": ft_r_szn, "MID_RANGE_RATE_ROLL5": mid_r5,
            "PTS_PER_SHOT_ROLL5": pts_r5 / max(self._roll("FGA", 5) + 0.44 * fta_pm_r5 + 0.01, 0.01),
            "CREATION_PROXY_ROLL5": self._roll("FGM", 5) - ast_szn * 0.15,
            "TEAM_MIN_SHARE_ROLL5": team_min_share, "MIN_RANK_IN_TEAM": min_rank,
            "TEAM_DEPTH_ROLL5": 0.0, "TEAM_PTS_SHARE_ROLL5": team_pts_share,
            "PTS_RANK_IN_TEAM_ROLL5": float(self.context.get("pts_rank_in_team", 1)),
            "OPP_AVG_MIN_ALLOWED": opp_avg_min, "OPP_STRENGTH": opp_strength,
            "OPP_PACE_PROXY": opp_fga_all, "H2H_MIN_VS_OPP": min_szn,
            "OPP_PTS_ALLOWED_AVG": opp_pts_all, "OPP_FGA_ALLOWED_AVG": opp_fga_all,
            "OPP_3PT_ALLOWED_AVG": opp_3pt_all, "OPP_DEF_INTENSITY": 0.0,
            "CUMULATIVE_MIN_10D": cum_min_10d, "GAMES_SINCE_RETURN": 0,
            "IRONMAN_STREAK": 0, "IS_ROOKIE_PHASE": 0, "IS_VETERAN_PHASE": 0,
            "PTS_ZSCORE_RECENT": pts_zscore, "HIGH_FOUL_RISK": int(pf_r5 > 3.5),
            "PM_MOMENTUM": pm_r5, "WIN_PCT_SZN": win_pct, "USAGE_SPIKE": 0.0,
            "FATIGUE_x_CLOSER": 0.0, "SCORER_x_HOME": scorer_score * is_home,
            "EFFICIENCY_x_OPP": eff_r5 * opp_strength,
            "ROLL5_MIN_x_REST": min_r5 * dr,
            "CONSISTENCY_x_RANK": min_cons * (1 / max(min_rank, 1)),
            "VETERAN_x_REST": 0.0, "EWM5_x_OPP_PACE": min_r5 * opp_fga_all,
            "RAMP_x_ROLE": 0.0, "DEPTH_x_RANK": 0.0, "VOL_RATIO_x_STREAK": 0.0,
            "VOL_x_EFF": fga_pm_r5 * ts_r5, "VOL_x_HOME": fga_pm_r5 * is_home,
            "EFF_x_OPP_PTS": ts_r5 * opp_pts_all,
            "PREDMIN_x_VOL": 0.0, "PREDMIN_x_EFF": 0.0,
            "3PT_RATE_x_EFF": three_r5 * fg3_r5,
            "LOAD_x_FATIGUE": team_pts_share * cum_min_10d,
            "SCORER_x_OPP": scorer_score * opp_pts_all,
            "HOT_x_HOME": hot_streak * is_home, "CV_x_CLOSE": pts_cv * 0,
            "EWM_PTS_x_OPP_DEF": pts_r5 * 0, "CREATION_x_OPP_PACE": 0.0,
            "PREDMIN_x_CREATION": 0.0, "COLD_x_AWAY": cold_streak * (1 - is_home),
            "FT_RATE_x_FOUL_RISK": ft_r_r5 * (1 - int(pf_r5 > 3.5)),
        }
        # Inject all injury features (default 0 if not available)
        from injury_features import _all_injury_feature_cols
        for col in _all_injury_feature_cols():
            feat[col] = float(self.injury_features.get(col, 0.0))

        # Points-specific injury interactions (require some base features already set)
        feat["INJ_BOOST_x_SCORER"]  = feat["TEAMMATE_FGA_ABSORBED"] * feat.get("SCORER_SCORE", 0)
        feat["INJ_BOOST_x_PREDMIN"] = feat["TEAMMATE_FGA_ABSORBED"] * feat.get("PREDICTED_MIN", 0)
        feat["OPP_DEPLETED_x_EFF"]  = feat["OPP_INJURY_ADVANTAGE"]  * feat.get("TRUE_SHOOTING_ROLL5", 0)
        feat["RETURN_x_VOL"]        = feat["RETURN_FROM_INJURY_FLAG"] * feat.get("FGA_PER_MIN_ROLL5", 0)

        return feat


# ── Prop Evaluator ─────────────────────────────────────────────────────────────

class PropEvaluator:
    def __init__(self, data_path: Path | None = None):
        self._loader = DatasetLoader(data_path)
        print("Loading models...", end=" ", flush=True)
        import minutes_predictor
        sys.modules['__main__'] = minutes_predictor
        self.min_model: MinutesPredictor = MinutesPredictor.load(MIN_MODEL_PKL)
        import points_predictor
        sys.modules['__main__'] = points_predictor
        self.pts_model: PointsPredictor  = PointsPredictor.load(PTS_MODEL_PKL)
        print("done.\n")

    def evaluate_from_dataset(
        self,
        player_name: str,
        opponent: str,
        is_home: int,
        game_date: str,
        line: float,
        side: str,
        american_odds: int,
        n_recent: int = 10,
    ) -> dict:
        loader = self._loader
        print(f"\nPulling data for {player_name} vs {opponent} on {game_date}...")
        games, season_games = loader.build_game_rows(player_name, game_date, n_recent)
        if not games:
            raise ValueError(f"No games found for '{player_name}' before {game_date}.")

        context   = loader.derive_context(player_name, game_date, is_home, opponent)
        opp_stats = loader.get_opponent_stats(opponent, game_date)

        # Fetch injury features for this specific game date
        team_abbr = loader.get_player_games(player_name, game_date)["TEAM_ABBREVIATION"].iloc[0] \
            if not loader.get_player_games(player_name, game_date).empty else ""
        injury_features = loader.get_injury_features_for_date(
            player_name=player_name,
            team_abbr=team_abbr,
            opponent=opponent,
            game_date=game_date,
        )

        if opp_stats:
            print(f"  Opponent ({opponent}) avg pts allowed: {opp_stats.get('avg_pts_allowed', 0):.1f}")
        else:
            print(f"  ⚠  No opponent data for '{opponent}' — using league averages.")

        # Log notable injury context
        if injury_features.get("PLAYER_IS_OUT", 0):
            print(f"  ⚠  {player_name} is listed OUT on {game_date}")
        elif injury_features.get("PLAYER_IS_QUESTIONABLE", 0):
            print(f"  ⚠  {player_name} is QUESTIONABLE on {game_date}")
        if injury_features.get("TEAMMATE_MIN_ABSORBED", 0) > 5:
            print(f"  ℹ  Teammate absence: ~{injury_features['TEAMMATE_MIN_ABSORBED']:.1f} min projected boost")

        print(f"  Recent games   : {len(games)}")
        print(f"  Season games   : {len(season_games)}")
        print(f"  Game #         : {context['game_num']}   Days rest: {context['days_rest']}")

        return self.evaluate({
            "player_name": player_name, "line": line, "side": side,
            "american_odds": american_odds, "games": games,
            "season_games": season_games, "context": context,
            "opp_stats": opp_stats, "injury_features": injury_features,
        })

    def evaluate(self, prop: dict) -> dict:
        player = prop["player_name"]
        line   = float(prop["line"])
        side   = prop["side"].lower()
        odds   = int(prop["american_odds"])

        builder = FeatureBuilder(
            prop["games"], prop.get("season_games", prop["games"]),
            prop.get("context", {}), prop.get("opp_stats", {}),
            prop.get("injury_features", {}),
        )
        feat = builder.build()

        # Stage 1: predict minutes
        X_min = pd.DataFrame([{k: feat.get(k, 0) for k in MIN_FEATURES}]).fillna(0)
        min_result = self.min_model.predict(player, X_min, return_uncertainty=True)
        pred_min   = float(min_result["final_pred"][0])
        min_std    = float(min_result.get("std", np.array([5.0]))[0])

        feat["PREDICTED_MIN"]      = pred_min
        feat["PREDICTED_MIN_STD"]  = min_std
        feat["PREDMIN_x_VOL"]      = pred_min * feat.get("FGA_PER_MIN_ROLL5", 0)
        feat["PREDMIN_x_EFF"]      = pred_min * feat.get("TRUE_SHOOTING_ROLL5", 0)
        feat["PREDMIN_x_CREATION"] = pred_min * feat.get("CREATION_PROXY_ROLL5", 0)
        # Update injury interactions that depend on PREDICTED_MIN
        feat["INJ_BOOST_x_PREDMIN"] = feat.get("TEAMMATE_FGA_ABSORBED", 0) * pred_min

        # Stage 2: predict points
        X_pts = pd.DataFrame([{k: feat.get(k, 0) for k in PTS_FEATURES}]).fillna(0)
        pts_result = self.pts_model.predict(player, X_pts, return_uncertainty=True)
        pred_pts   = float(pts_result["final_pred"][0])
        pts_std    = float(pts_result.get("std", np.array([6.5]))[0])

        p_over   = float(1 - stats.norm.cdf(line, loc=pred_pts, scale=pts_std))
        p_under  = 1.0 - p_over
        model_prob = p_over if side == "over" else p_under
        book_prob  = american_to_implied(odds)
        edge       = model_prob - book_prob

        # Injury context summary for display
        inj = prop.get("injury_features", {})
        injury_note = None
        if inj.get("PLAYER_IS_OUT", 0):
            injury_note = "⚠ Player listed OUT"
        elif inj.get("PLAYER_IS_QUESTIONABLE", 0):
            injury_note = "⚠ Player QUESTIONABLE"
        elif inj.get("TEAMMATE_MIN_ABSORBED", 0) > 5:
            injury_note = f"ℹ Teammate absence: +{inj['TEAMMATE_MIN_ABSORBED']:.1f} min projected"
        elif inj.get("OPP_STARS_OUT", 0) >= 1:
            injury_note = f"ℹ Opponent missing {int(inj['OPP_STARS_OUT'])} star(s)"

        return {
            "player": player, "line": line, "side": side, "book_odds": odds,
            "pred_min": round(pred_min, 1), "min_std": round(min_std, 1),
            "pred_pts": round(pred_pts, 1), "pts_std": round(pts_std, 1),
            "p_over": round(p_over, 4), "p_under": round(p_under, 4),
            "model_prob": round(model_prob, 4), "book_prob": round(book_prob, 4),
            "edge": round(edge, 4), "has_edge": edge > 0,
            "model_american": implied_to_american(model_prob),
            "fair_american": implied_to_american(p_over if side == "over" else p_under),
            "player_model": player in self.pts_model.player_models,
            "games_seen": self.pts_model.player_models[player].n_games
                          if player in self.pts_model.player_models else 0,
            "injury_note": injury_note,
            "teammate_min_absorbed": round(float(inj.get("TEAMMATE_MIN_ABSORBED", 0)), 1),
            "opp_stars_out": int(inj.get("OPP_STARS_OUT", 0)),
        }

    def print_result(self, r: dict) -> None:
        w = "═" * 56
        print(f"\n{w}")
        print(f"  {r['player'].upper()}")
        print(f"  {r['side'].upper()} {r['line']} pts  @ {r['book_odds']:+d}")
        print(w)
        print(f"  Predicted minutes : {r['pred_min']} ± {r['min_std']} min")
        print(f"  Predicted points  : {r['pred_pts']} ± {r['pts_std']} pts\n")
        if r.get("injury_note"):
            print(f"  {r['injury_note']}\n")
        print(f"  P(over)           : {r['p_over']:.1%}")
        print(f"  P(under)          : {r['p_under']:.1%}\n")
        print(f"  Model probability : {r['model_prob']:.1%}")
        print(f"  Book implied      : {r['book_prob']:.1%}")
        print(f"  Edge              : {r['edge']:+.1%}\n")
        print(f"  Model odds        : {r['model_american']:+d}")
        print(f"  Fair odds         : {r['fair_american']:+d}")
        print(f"  Book odds         : {r['book_odds']:+d}\n")
        if r["has_edge"]:
            print(f"  ✓  BET {r['side'].upper()}")
        else:
            print("  ✗  NO BET")
        print()
        if not r["player_model"]:
            print("  ⚠  No player-specific model — using league baseline only.")
        else:
            print(f"  Player model trained on {r['games_seen']} games.")
        print(w)


# ── Parlay Evaluator ──────────────────────────────────────────────────────────

class ParlayEvaluator:
    """
    Evaluates a two-leg points prop parlay.

    Each leg is evaluated independently by PropEvaluator, then the combined
    (joint) model probability is compared against the parlay payout's
    breakeven probability to compute edge.

    Legs are assumed independent — a conservative assumption that slightly
    understates true edge when both legs are from the same game (correlated
    outcomes), but is the correct default for players on different teams.

    Usage
    -----
        ev = ParlayEvaluator()
        result = ev.evaluate(leg1_kwargs, leg2_kwargs, parlay_american_odds)
    """

    def __init__(self, data_path: Path | None = None):
        # Share one PropEvaluator instance (and therefore one DatasetLoader +
        # model load) across both legs to avoid loading twice.
        self._prop_ev = PropEvaluator(data_path)

    def evaluate(
        self,
        leg1: dict,
        leg2: dict,
        parlay_american_odds=None,
        parlay_multiplier=None,
    ) -> dict:
        """
        Evaluate a two-leg parlay.

        Supply exactly one of:
          parlay_american_odds : int    e.g. +260
          parlay_multiplier    : float  e.g. 3.5  (means $35 profit on $10 bet)
        """
        print("\n── Evaluating Leg 1 ──")
        r1 = self._prop_ev.evaluate_from_dataset(**leg1)

        print("\n── Evaluating Leg 2 ──")
        r2 = self._prop_ev.evaluate_from_dataset(**leg2)

        model_joint_prob = r1["model_prob"] * r2["model_prob"]

        book_break_even, book_decimal, book_american, book_multiplier = resolve_parlay_payout(
            parlay_american_odds, parlay_multiplier
        )

        fair_american   = implied_to_american(model_joint_prob) if model_joint_prob < 1 else -99999
        fair_multiplier = implied_to_multiplier(model_joint_prob)
        edge     = model_joint_prob - book_break_even
        has_edge = edge > 0

        return {
            "leg1": r1,
            "leg2": r2,
            "leg1_model_prob":          round(r1["model_prob"], 4),
            "leg2_model_prob":          round(r2["model_prob"], 4),
            "model_joint_prob":         round(model_joint_prob, 4),
            "book_break_even":          round(book_break_even, 4),
            "parlay_book_odds":         book_american,
            "parlay_book_multiplier":   round(book_multiplier, 4),
            "parlay_fair_american":     fair_american,
            "parlay_fair_multiplier":   fair_multiplier,
            "edge":                     round(edge, 4),
            "has_edge":                 has_edge,
            "independence_assumed":     True,
        }

    def print_result(self, r: dict, bankroll: float = 10_000) -> None:
        w = "═" * 60
        print(f"\n{w}")
        print("  TWO-LEG PARLAY EVALUATION")
        print(w)

        for i, leg_key in enumerate(("leg1", "leg2"), 1):
            leg = r[leg_key]
            print(f"\n  Leg {i}: {leg['player'].upper()}")
            print(f"    {leg['side'].upper()} {leg['line']} pts  @ {leg['book_odds']:+d}")
            print(f"    Predicted: {leg['pred_pts']} ± {leg['pts_std']} pts "
                  f"({leg['pred_min']} min)")
            print(f"    Model prob: {leg['model_prob']:.1%}  "
                  f"{'✓ has edge' if leg['has_edge'] else '✗ no edge'} on this leg alone")

        print(f"\n  {'─' * 56}")
        print(f"  Joint model probability  : {r['model_joint_prob']:.1%}  (Leg1 × Leg2)")
        print(f"  Book breakeven           : {r['book_break_even']:.1%}")
        print(f"  Book payout              : {r['parlay_book_odds']:+d}  ({r['parlay_book_multiplier']:.2f}x)")
        print(f"  Fair payout              : {r['parlay_fair_american']:+d}  ({r['parlay_fair_multiplier']:.2f}x)")
        print(f"  Parlay edge              : {r['edge']:+.1%}")
        print()

        if r["has_edge"]:
            print(f"  ✅  BET PARLAY — Edge: {r['edge']:+.1%}")
        else:
            print("  ❌  NO BET — combined probability does not beat the book's implied odds")

        print()
        print("  ⚠  Legs treated as independent. Adjust judgment for same-game parlays.")
        print(w)

# ── CLI ────────────────────────────────────────────────────────────────────────

def _ask(prompt, type_fn=str, default=None):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        val = input(f"  {prompt}{suffix}: ").strip()
        if val == "" and default is not None:
            return default
        try:
            return type_fn(val)
        except ValueError:
            print(f"  ✗ Expected {type_fn.__name__}, got '{val}'.")


def _ask_leg(label: str) -> dict:
    """Interactively collect inputs for one prop leg."""
    print(f"\n  ── {label} ──")
    player_name   = _ask("Player name (exact)", str)
    opponent      = _ask("Opponent abbreviation (e.g. LAL)", str).upper()
    is_home       = _ask("Home? (1=yes, 0=no)", int, 1)
    game_date     = _ask("Game date (YYYY-MM-DD)", str, date.today().isoformat())
    line          = _ask("Prop line (e.g. 24.5)", float)
    side = ""
    while side not in ("over", "under"):
        side = input("  Side [over/under]: ").strip().lower()
    american_odds = _ask("Book odds for this leg alone (e.g. -115)", int)
    return dict(player_name=player_name, opponent=opponent, is_home=is_home,
                game_date=game_date, line=line, side=side, american_odds=american_odds)


def run_cli():
    print("\n" + "─" * 56)
    print("  NBA Prop Evaluator")
    print("─" * 56)

    mode = ""
    while mode not in ("single", "parlay"):
        mode = input("  Mode [single / parlay]: ").strip().lower()

    if mode == "single":
        player_name   = _ask("Player name (exact)", str)
        opponent      = _ask("Opponent abbreviation (e.g. LAL)", str).upper()
        is_home       = _ask("Home? (1=yes, 0=no)", int, 1)
        game_date     = _ask("Game date (YYYY-MM-DD)", str, date.today().isoformat())
        line          = _ask("Prop line (e.g. 24.5)", float)
        side = ""
        while side not in ("over", "under"):
            side = input("  Side [over/under]: ").strip().lower()
        american_odds = _ask("Book odds (e.g. -115 or +105)", int)

        evaluator = PropEvaluator()
        result = evaluator.evaluate_from_dataset(
            player_name=player_name, opponent=opponent, is_home=is_home,
            game_date=game_date, line=line, side=side, american_odds=american_odds,
        )
        evaluator.print_result(result)

        out = Path("prop_results.json")
        existing = json.loads(out.read_text()) if out.exists() else []
        existing.append({**result, "mode": "single", "evaluated_at": datetime.now().isoformat()})
        out.write_text(json.dumps(existing, indent=2))
        print(f"\n  Result saved to {out}\n")

    else:  # parlay
        leg1 = _ask_leg("Leg 1")
        leg2 = _ask_leg("Leg 2")
        parlay_odds = _ask("Parlay payout odds (e.g. +260)", int)

        evaluator = ParlayEvaluator()
        result = evaluator.evaluate(leg1, leg2, parlay_odds)
        evaluator.print_result(result)

        out = Path("prop_results.json")
        existing = json.loads(out.read_text()) if out.exists() else []
        existing.append({**result, "mode": "parlay", "evaluated_at": datetime.now().isoformat()})
        out.write_text(json.dumps(existing, indent=2))
        print(f"\n  Result saved to {out}\n")


if __name__ == "__main__":
    run_cli()

# ── Batch evaluator (used by /api/batch_props) ────────────────────────────────

def batch_evaluate_props(rows: list[dict], game_date: str) -> list[dict]:
    """
    Evaluate a list of prop rows (from a spreadsheet upload) and return
    per-prop results plus all two-leg combination joint probabilities.

    Each row must have:
        player_name, opponent, is_home (0/1), prop_line, game_date (optional)

    Returns two lists:
        props     : per-player evaluation results (pred_pts, model_prob for
                    best side, etc.)
        combos    : all C(n,2) two-leg combinations with joint probability,
                    fair multiplier, and fair American odds, sorted by
                    joint probability descending.
    """
    ev = PropEvaluator()
    props = []

    for row in rows:
        player   = str(row["player_name"]).strip()
        opp      = str(row["opponent"]).strip().upper()
        is_home  = int(row.get("is_home", 1))
        line     = float(row["prop_line"])
        gdate    = str(row.get("game_date", game_date)).strip() or game_date
        # use a dummy leg odds — only model_prob matters for batch
        dummy_odds = -110

        try:
            r_over  = ev.evaluate_from_dataset(
                player_name=player, opponent=opp, is_home=is_home,
                game_date=gdate, line=line, side="over",
                american_odds=dummy_odds,
            )
            r_under = ev.evaluate_from_dataset(
                player_name=player, opponent=opp, is_home=is_home,
                game_date=gdate, line=line, side="under",
                american_odds=dummy_odds,
            )
            # Best side = whichever has higher model probability
            best    = r_over if r_over["model_prob"] >= r_under["model_prob"] else r_under
            props.append({
                "player":       player,
                "opponent":     opp,
                "is_home":      is_home,
                "prop_line":    line,
                "game_date":    gdate,
                "pred_pts":     best["pred_pts"],
                "pred_min":     best["pred_min"],
                "pts_std":      best["pts_std"],
                "best_side":    best["side"],
                "model_prob":   best["model_prob"],
                "p_over":       r_over["model_prob"],
                "p_under":      r_under["model_prob"],
                "player_model": best["player_model"],
                "error":        None,
            })
        except Exception as e:
            props.append({
                "player": player, "opponent": opp, "is_home": is_home,
                "prop_line": line, "game_date": gdate,
                "pred_pts": None, "pred_min": None, "pts_std": None,
                "best_side": None, "model_prob": None,
                "p_over": None, "p_under": None, "player_model": False,
                "error": str(e),
            })

    # All two-leg combinations
    combos = []
    valid = [p for p in props if p["model_prob"] is not None]
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            a, b = valid[i], valid[j]
            joint = a["model_prob"] * b["model_prob"]
            combos.append({
                "leg1_player":    a["player"],
                "leg1_side":      a["best_side"],
                "leg1_line":      a["prop_line"],
                "leg1_prob":      round(a["model_prob"], 4),
                "leg2_player":    b["player"],
                "leg2_side":      b["best_side"],
                "leg2_line":      b["prop_line"],
                "leg2_prob":      round(b["model_prob"], 4),
                "joint_prob":     round(joint, 4),
                "fair_multiplier": round((1.0 / joint) - 1.0, 3) if joint > 0 else None,
                "fair_american":   implied_to_american(joint) if joint > 0 else None,
            })

    combos.sort(key=lambda c: c["joint_prob"], reverse=True)
    return props, combos