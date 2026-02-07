from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pandas as pd

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

def _ok(df: pd.DataFrame, n: int = 2) -> bool:
    return df is not None and not df.empty and len(df) >= n and all(c in df.columns for c in REQUIRED_COLS)

def last_two(df: pd.DataFrame) -> Optional[Tuple[pd.Series, pd.Series]]:
    if not _ok(df, 2):
        return None
    return df.iloc[-1], df.iloc[-2]

def is_inside(cur: pd.Series, prev: pd.Series) -> bool:
    return (cur["High"] <= prev["High"]) and (cur["Low"] >= prev["Low"])

def is_2u(cur: pd.Series, prev: pd.Series) -> bool:
    return (cur["High"] > prev["High"]) and (cur["Low"] >= prev["Low"])

def is_2d(cur: pd.Series, prev: pd.Series) -> bool:
    return (cur["Low"] < prev["Low"]) and (cur["High"] <= prev["High"])

def candle_type(df: pd.DataFrame) -> str:
    pair = last_two(df)
    if not pair:
        return "n/a"
    cur, prev = pair
    if is_inside(cur, prev): return "1"
    if is_2u(cur, prev):     return "2U"
    if is_2d(cur, prev):     return "2D"
    return "3"  # outside (we’re not using 3 for signals yet)

def strat_inside(df: pd.DataFrame) -> bool:
    pair = last_two(df)
    if not pair:
        return False
    cur, prev = pair
    return is_inside(cur, prev)

def strat_212_up(df: pd.DataFrame) -> bool:
    if not _ok(df, 4):
        return False
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    prev_a = df.iloc[-4]
    return is_2u(a, prev_a) and is_inside(b, a) and is_2u(c, b)

def strat_212_dn(df: pd.DataFrame) -> bool:
    if not _ok(df, 4):
        return False
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    prev_a = df.iloc[-4]
    return is_2d(a, prev_a) and is_inside(b, a) and is_2d(c, b)

def best_trigger(bias: str, d: pd.DataFrame, w: pd.DataFrame):
    # Weekly inside triggers first, then daily.
    if bias not in ("LONG", "SHORT"):
        return None, None, None

    if strat_inside(w) and len(w) >= 2:
        cur = w.iloc[-1]
        hi, lo = float(cur["High"]), float(cur["Low"])
        return ("W", hi, lo) if bias == "LONG" else ("W", lo, hi)

    if strat_inside(d) and len(d) >= 2:
        cur = d.iloc[-1]
        hi, lo = float(cur["High"]), float(cur["Low"])
        return ("D", hi, lo) if bias == "LONG" else ("D", lo, hi)

    return None, None, None

def compute_flags(d: pd.DataFrame, w: pd.DataFrame, m: pd.DataFrame) -> Dict[str, bool]:
    return {
        "D_Inside": strat_inside(d),
        "W_Inside": strat_inside(w),
        "M_Inside": strat_inside(m),
        "D_212Up":  strat_212_up(d),
        "W_212Up":  strat_212_up(w),
        "D_212Dn":  strat_212_dn(d),
        "W_212Dn":  strat_212_dn(w),
    }
