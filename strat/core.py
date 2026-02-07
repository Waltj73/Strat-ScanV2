# strat/core.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

REQUIRED_COLS = ["Open","High","Low","Close","Volume"]

def _last_two(df: pd.DataFrame):
    if df is None or df.empty or len(df) < 2:
        return None
    return df.iloc[-1], df.iloc[-2]

def is_inside(cur, prev) -> bool:
    return (cur["High"] <= prev["High"]) and (cur["Low"] >= prev["Low"])

def is_2u(cur, prev) -> bool:
    return (cur["High"] > prev["High"]) and (cur["Low"] >= prev["Low"])

def is_2d(cur, prev) -> bool:
    return (cur["Low"] < prev["Low"]) and (cur["High"] <= prev["High"])

def is_3(cur, prev) -> bool:
    return (cur["High"] > prev["High"]) and (cur["Low"] < prev["Low"])

def candle_type_label(df: pd.DataFrame) -> str:
    pair = _last_two(df)
    if not pair:
        return "n/a"
    cur, prev = pair
    if is_inside(cur, prev):
        return "1"
    if is_3(cur, prev):
        return "3"
    if is_2u(cur, prev):
        return "2U"
    if is_2d(cur, prev):
        return "2D"
    return "n/a"

def strat_inside(df: pd.DataFrame) -> bool:
    pair = _last_two(df)
    if not pair:
        return False
    cur, prev = pair
    return is_inside(cur, prev)

def strat_212_up(df: pd.DataFrame) -> bool:
    if df is None or df.empty or len(df) < 4:
        return False
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    prev_a = df.iloc[-4]
    return is_2u(a, prev_a) and is_inside(b, a) and is_2u(c, b)

def strat_212_dn(df: pd.DataFrame) -> bool:
    if df is None or df.empty or len(df) < 4:
        return False
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    prev_a = df.iloc[-4]
    return is_2d(a, prev_a) and is_inside(b, a) and is_2d(c, b)

def compute_flags(d: pd.DataFrame, w: pd.DataFrame, m: pd.DataFrame) -> Dict[str, bool]:
    return {
        "D_Inside": strat_inside(d),
        "W_Inside": strat_inside(w),
        "M_Inside": strat_inside(m),
        "D_212Up": strat_212_up(d),
        "W_212Up": strat_212_up(w),
        "D_212Dn": strat_212_dn(d),
        "W_212Dn": strat_212_dn(w),
    }

def score_regime(flags: Dict[str, bool]) -> Tuple[int, int]:
    # Simple scoring for now; we can weight M/W/D in Step 4.
    bull = 0
    bear = 0
    bull += 2 if flags.get("W_212Up") else 0
    bull += 1 if flags.get("D_212Up") else 0
    bear += 2 if flags.get("W_212Dn") else 0
    bear += 1 if flags.get("D_212Dn") else 0
    return bull, bear

def market_bias_and_strength(market_rows: List[Dict]) -> Tuple[str, int, int]:
    bull_total = sum(int(r.get("BullScore", 0)) for r in market_rows)
    bear_total = sum(int(r.get("BearScore", 0)) for r in market_rows)
    diff = bull_total - bear_total
    strength = int(max(0, min(100, 50 + diff * 8)))
    if diff >= 2:
        return "LONG", strength, diff
    if diff <= -2:
        return "SHORT", strength, diff
    return "MIXED", strength, diff

def best_trigger(bias: str, d: pd.DataFrame, w: pd.DataFrame) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    Weekly-first. Bias-aware.
    LONG: entry = IB high, stop = IB low
    SHORT: entry = IB low, stop = IB high
    """
    def levels(cur):
        hi = float(cur["High"])
        lo = float(cur["Low"])
        if bias == "SHORT":
            return lo, hi
        return hi, lo

    if strat_inside(w) and len(w) >= 2:
        cur = w.iloc[-1]
        entry, stop = levels(cur)
        return "W", entry, stop

    if strat_inside(d) and len(d) >= 2:
        cur = d.iloc[-1]
        entry, stop = levels(cur)
        return "D", entry, stop

    return None, None, None
