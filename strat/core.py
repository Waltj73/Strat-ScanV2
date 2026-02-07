# strat/core.py

import numpy as np
import pandas as pd
# strat/core.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import pandas as pd

def candle_color(cur: pd.Series) -> str:
    if float(cur["Close"]) > float(cur["Open"]):
        return "green"
    if float(cur["Close"]) < float(cur["Open"]):
        return "red"
    return "doji"

def candle_type(cur: pd.Series, prev: pd.Series) -> str:
    hi, lo = float(cur["High"]), float(cur["Low"])
    phi, plo = float(prev["High"]), float(prev["Low"])

    inside = (hi <= phi) and (lo >= plo)
    two_up = (hi > phi) and (lo >= plo)
    two_dn = (lo < plo) and (hi <= phi)
    outside = (hi > phi) and (lo < plo)

    if inside:
        return "1"
    if two_up:
        return "2U"
    if two_dn:
        return "2D"
    if outside:
        return "3"
    return "?"

def candle_type_label(df: pd.DataFrame) -> str:
    if df is None or df.empty or len(df) < 2:
        return "n/a"
    cur = df.iloc[-1]
    prev = df.iloc[-2]
    return candle_type(cur, prev)

def last_two(df: pd.DataFrame) -> Optional[Tuple[pd.Series, pd.Series]]:
    if df is None or df.empty or len(df) < 2:
        return None
    return df.iloc[-1], df.iloc[-2]

def compute_flags(d: pd.DataFrame, w: pd.DataFrame, m: pd.DataFrame) -> Dict[str, bool]:
    def tf_flags(prefix: str, df: pd.DataFrame) -> Dict[str, bool]:
        pair = last_two(df)
        if not pair:
            return {
                f"{prefix}_Bull": False, f"{prefix}_Bear": False, f"{prefix}_Inside": False,
                f"{prefix}_212Up": False, f"{prefix}_212Dn": False,
            }

        cur, prev = pair
        t = candle_type(cur, prev)
        col = candle_color(cur)

        bull = (t == "2U") or (t == "3" and col == "green")
        bear = (t == "2D") or (t == "3" and col == "red")
        inside = (t == "1")

        return {
            f"{prefix}_Bull": bull,
            f"{prefix}_Bear": bear,
            f"{prefix}_Inside": inside,
        }

    flags = {}
    flags.update(tf_flags("D", d))
    flags.update(tf_flags("W", w))
    flags.update(tf_flags("M", m))

    # 2-1-2 patterns (D/W only)
    flags["D_212Up"] = _is_212_up(d)
    flags["W_212Up"] = _is_212_up(w)
    flags["D_212Dn"] = _is_212_dn(d)
    flags["W_212Dn"] = _is_212_dn(w)

    return flags

def _is_212_up(df: pd.DataFrame) -> bool:
    if df is None or df.empty or len(df) < 4:
        return False
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    pa = df.iloc[-4]
    return (candle_type(a, pa) == "2U") and (candle_type(b, a) == "1") and (candle_type(c, b) == "2U")

def _is_212_dn(df: pd.DataFrame) -> bool:
    if df is None or df.empty or len(df) < 4:
        return False
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    pa = df.iloc[-4]
    return (candle_type(a, pa) == "2D") and (candle_type(b, a) == "1") and (candle_type(c, b) == "2D")

def score_regime(flags: Dict[str, bool]) -> Tuple[int, int]:
    # weights: M=3, W=2, D=1; plus 2-1-2 bonuses
    bull = 0
    bear = 0

    bull += 3 if flags.get("M_Bull") else 0
    bull += 2 if flags.get("W_Bull") else 0
    bull += 1 if flags.get("D_Bull") else 0

    bear += 3 if flags.get("M_Bear") else 0
    bear += 2 if flags.get("W_Bear") else 0
    bear += 1 if flags.get("D_Bear") else 0

    bull += 2 if flags.get("W_212Up") else 0
    bull += 1 if flags.get("D_212Up") else 0
    bear += 2 if flags.get("W_212Dn") else 0
    bear += 1 if flags.get("D_212Dn") else 0

    return bull, bear
