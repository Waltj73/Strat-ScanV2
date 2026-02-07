# strat/core.py
from __future__ import annotations

from typing import Dict, Optional, Tuple, List
import pandas as pd


# -------------------------
# Helpers: normalize OHLC
# -------------------------
REQUIRED = ["Open", "High", "Low", "Close"]

def _norm_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has Open/High/Low/Close with correct casing."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # normalize common lowercase variants
    rename = {}
    for c in out.columns:
        if not isinstance(c, str):
            continue
        lc = c.strip().lower()
        if lc == "open":
            rename[c] = "Open"
        elif lc == "high":
            rename[c] = "High"
        elif lc == "low":
            rename[c] = "Low"
        elif lc == "close":
            rename[c] = "Close"
        elif lc in ("adj close", "adj_close", "adjclose"):
            # only map to Close if Close doesn't exist
            if "Close" not in out.columns:
                rename[c] = "Close"

    if rename:
        out = out.rename(columns=rename)

    # If Close still missing, try Adj Close style column
    if "Close" not in out.columns:
        for alt in ["Adj Close", "adj close", "Adj_Close", "AdjClose"]:
            if alt in out.columns:
                out["Close"] = out[alt]
                break

    # Must have OHLC
    if not set(REQUIRED).issubset(set(out.columns)):
        return pd.DataFrame()

    # Coerce numeric
    for c in REQUIRED:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=REQUIRED)
    return out


# -------------------------
# STRAT candle logic
# -------------------------
def candle_color(cur: pd.Series) -> str:
    try:
        o = float(cur["Open"])
        c = float(cur["Close"])
    except Exception:
        return "doji"

    if c > o:
        return "green"
    if c < o:
        return "red"
    return "doji"


def candle_type(cur: pd.Series, prev: pd.Series) -> str:
    try:
        hi, lo = float(cur["High"]), float(cur["Low"])
        phi, plo = float(prev["High"]), float(prev["Low"])
    except Exception:
        return "?"

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


def last_two(df: pd.DataFrame) -> Optional[Tuple[pd.Series, pd.Series]]:
    df = _norm_ohlc(df)
    if df.empty or len(df) < 2:
        return None
    return df.iloc[-1], df.iloc[-2]


def candle_type_label(df: pd.DataFrame) -> str:
    pair = last_two(df)
    if not pair:
        return "n/a"
    cur, prev = pair
    return candle_type(cur, prev)


# -------------------------
# 2-1-2 patterns
# -------------------------
def _is_212_up(df: pd.DataFrame) -> bool:
    df = _norm_ohlc(df)
    if df.empty or len(df) < 4:
        return False
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    pa = df.iloc[-4]
    return (candle_type(a, pa) == "2U") and (candle_type(b, a) == "1") and (candle_type(c, b) == "2U")


def _is_212_dn(df: pd.DataFrame) -> bool:
    df = _norm_ohlc(df)
    if df.empty or len(df) < 4:
        return False
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    pa = df.iloc[-4]
    return (candle_type(a, pa) == "2D") and (candle_type(b, a) == "1") and (candle_type(c, b) == "2D")


# -------------------------
# Flags + scoring
# -------------------------
def compute_flags(d: pd.DataFrame, w: pd.DataFrame, m: pd.DataFrame) -> Dict[str, bool]:
    def tf_flags(prefix: str, df: pd.DataFrame) -> Dict[str, bool]:
        pair = last_two(df)
        if not pair:
            return {
                f"{prefix}_Bull": False,
                f"{prefix}_Bear": False,
                f"{prefix}_Inside": False,
            }

        cur, prev = pair
        t = candle_type(cur, prev).strip().upper()
        col = candle_color(cur)

        bull = (t == "2U") or (t == "3" and col == "green")
        bear = (t == "2D") or (t == "3" and col == "red")
        inside = (t == "1")

        return {
            f"{prefix}_Bull": bool(bull),
            f"{prefix}_Bear": bool(bear),
            f"{prefix}_Inside": bool(inside),
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
# Market bias + triggers
# (Keep your existing versions if you already have them elsewhere)
# -------------------------
def market_bias_and_strength(rows: List[Dict]) -> Tuple[str, int, int]:
    bull_total = sum(int(r.get("BullScore", 0)) for r in rows)
    bear_total = sum(int(r.get("BearScore", 0)) for r in rows)
    diff = bull_total - bear_total

    strength = max(0, min(100, int(50 + diff * 5)))

    if diff >= 3:
        bias = "LONG"
    elif diff <= -3:
        bias = "SHORT"
    else:
        bias = "MIXED"

    return bias, strength, diff


def best_trigger(bias: str, d: pd.DataFrame, w: pd.DataFrame):
    """
    Weekly-first inside bar trigger.
    LONG: entry = IB high, stop = IB low
    SHORT: entry = IB low, stop = IB high
    """
    bias = (bias or "MIXED").upper()
    d = _norm_ohlc(d)
    w = _norm_ohlc(w)

    def _inside_levels(df: pd.DataFrame):
        if df.empty or len(df) < 2:
            return None
        cur, prev = df.iloc[-1], df.iloc[-2]
        if candle_type(cur, prev) != "1":
            return None
        return float(cur["High"]), float(cur["Low"])

    wl = _inside_levels(w)
    if wl:
        hi, lo = wl
        if bias == "SHORT":
            return "W", lo, hi
        return "W", hi, lo

    dl = _inside_levels(d)
    if dl:
        hi, lo = dl
        if bias == "SHORT":
            return "D", lo, hi
        return "D", hi, lo

    return None, None, None
