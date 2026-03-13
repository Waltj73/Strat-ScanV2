# strat/core.py
from __future__ import annotations

from typing import Dict, Optional, Tuple
import pandas as pd

# -------------------------
# Candle helpers
# -------------------------
def candle_color(cur: pd.Series) -> str:
    o = float(cur["Open"])
    c = float(cur["Close"])
    if c > o:
        return "green"
    if c < o:
        return "red"
    return "doji"


def candle_color_last(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "n/a"
    cur = df.iloc[-1]
    return candle_color(cur)


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
    cur, prev = df.iloc[-1], df.iloc[-2]
    return candle_type(cur, prev)


def candle_type_with_color(df: pd.DataFrame) -> str:
    if df is None or df.empty or len(df) < 2:
        return "n/a"

    cur, prev = df.iloc[-1], df.iloc[-2]
    t = candle_type(cur, prev)
    col = candle_color(cur)

    if col == "green":
        return f"{t} (G)"
    if col == "red":
        return f"{t} (R)"

    return f"{t} (D)"


def last_two(df: pd.DataFrame) -> Optional[Tuple[pd.Series, pd.Series]]:
    if df is None or df.empty or len(df) < 2:
        return None
    return df.iloc[-1], df.iloc[-2]


# -------------------------
# 2-1-2 patterns
# -------------------------
def _is_212_up(df: pd.DataFrame) -> bool:
    if df is None or df.empty or len(df) < 4:
        return False

    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    pa = df.iloc[-4]

    return (
        candle_type(a, pa) == "2U"
        and candle_type(b, a) == "1"
        and candle_type(c, b) == "2U"
    )


def _is_212_dn(df: pd.DataFrame) -> bool:
    if df is None or df.empty or len(df) < 4:
        return False

    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    pa = df.iloc[-4]

    return (
        candle_type(a, pa) == "2D"
        and candle_type(b, a) == "1"
        and candle_type(c, b) == "2D"
    )


# -------------------------
# STRAT flags
# -------------------------
def compute_flags(d: pd.DataFrame, w: pd.DataFrame, m: pd.DataFrame) -> Dict[str, bool]:

    def tf_flags(prefix: str, df: pd.DataFrame) -> Dict[str, bool]:

        pair = last_two(df)

        if not pair:
            return {
                f"{prefix}_Bull": False,
                f"{prefix}_Bear": False,
                f"{prefix}_Inside": False,
                f"{prefix}_IF_Up": False,
                f"{prefix}_IF_Down": False,
            }

        cur, prev = pair
        t = candle_type(cur, prev)

        c = float(cur["Close"])
        phi, plo = float(prev["High"]), float(prev["Low"])

        if_up = c > phi
        if_down = c < plo

        bull = (t == "2U") or (t == "3" and candle_color(cur) == "green")
        bear = (t == "2D") or (t == "3" and candle_color(cur) == "red")
        inside = t == "1"

        return {
            f"{prefix}_Bull": bull,
            f"{prefix}_Bear": bear,
            f"{prefix}_Inside": inside,
            f"{prefix}_IF_Up": bool(if_up),
            f"{prefix}_IF_Down": bool(if_down),
        }

    flags: Dict[str, bool] = {}

    flags.update(tf_flags("D", d))
    flags.update(tf_flags("W", w))
    flags.update(tf_flags("M", m))

    flags["D_212Up"] = _is_212_up(d)
    flags["W_212Up"] = _is_212_up(w)

    flags["D_212Dn"] = _is_212_dn(d)
    flags["W_212Dn"] = _is_212_dn(w)

    return flags


# -------------------------
# STRAT scoring
# -------------------------
def score_regime(flags: Dict[str, bool]) -> Tuple[int, int]:

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


# -------------------------
# Market bias
# -------------------------
def market_bias_and_strength(market_rows: list[dict]):

    bull_total = sum(int(r.get("BullScore", 0)) for r in market_rows)
    bear_total = sum(int(r.get("BearScore", 0)) for r in market_rows)

    diff = bull_total - bear_total

    strength = max(0, min(100, int(50 + diff * 5)))

    if diff >= 3:
        return "LONG", strength, diff

    if diff <= -3:
        return "SHORT", strength, diff

    return "MIXED", strength, diff


# -------------------------
# Trigger detection
# -------------------------
def best_trigger(bias: str, d: pd.DataFrame, w: pd.DataFrame):

    def is_inside(df: pd.DataFrame):

        if df is None or df.empty or len(df) < 2:
            return False

        cur, prev = df.iloc[-1], df.iloc[-2]

        return (
            float(cur["High"]) <= float(prev["High"])
            and float(cur["Low"]) >= float(prev["Low"])
        )

    if is_inside(w):

        cur = w.iloc[-1]

        hi = float(cur["High"])
        lo = float(cur["Low"])

        if bias == "SHORT":
            return "W", lo, hi

        return "W", hi, lo

    if is_inside(d):

        cur = d.iloc[-1]

        hi = float(cur["High"])
        lo = float(cur["Low"])

        if bias == "SHORT":
            return "D", lo, hi

        return "D", hi, lo

    return None, None, None


# -------------------------
# FAILED 2 MONTH SETUP
# -------------------------
def setup_failed2m_2uW_actionableD(d, w, m):

    if d is None or w is None or m is None:
        return False

    if len(m) < 2 or len(w) < 2 or len(d) < 2:
        return False

    m_cur, m_prev = m.iloc[-1], m.iloc[-2]

    m_type = candle_type(m_cur, m_prev)
    m_color = candle_color(m_cur)

    if not (m_type == "2D" and m_color == "green"):
        return False

    w_cur, w_prev = w.iloc[-1], w.iloc[-2]

    if candle_type(w_cur, w_prev) != "2U":
        return False

    d_cur, d_prev = d.iloc[-1], d.iloc[-2]

    d_type = candle_type(d_cur, d_prev)

    inside = d_type == "1"
    continuation = _is_212_up(d)

    return inside or continuation
