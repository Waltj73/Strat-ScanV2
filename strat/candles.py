import pandas as pd
from typing import Optional, Tuple

def is_green(bar: pd.Series) -> bool:
    return float(bar["Close"]) > float(bar["Open"])

def is_red(bar: pd.Series) -> bool:
    return float(bar["Close"]) < float(bar["Open"])

def classify_candle(cur: pd.Series, prev: pd.Series) -> str:
    """
    STRAT candle types:
      1  = inside
      2U = two-up
      2D = two-down
      3  = outside (takes both sides)
    """
    ch, cl = float(cur["High"]), float(cur["Low"])
    ph, pl = float(prev["High"]), float(prev["Low"])

    inside = (ch <= ph) and (cl >= pl)
    if inside:
        return "1"

    two_up = (ch > ph) and (cl >= pl)
    if two_up:
        return "2U"

    two_dn = (cl < pl) and (ch <= ph)
    if two_dn:
        return "2D"

    # outside (or weird overlap)
    return "3"

def last_two(df: pd.DataFrame) -> Optional[Tuple[pd.Series, pd.Series]]:
    if df is None or df.empty or len(df) < 2:
        return None
    return df.iloc[-1], df.iloc[-2]

def last_candle_type(df: pd.DataFrame) -> Optional[str]:
    pair = last_two(df)
    if not pair:
        return None
    cur, prev = pair
    return classify_candle(cur, prev)

def last_inside(df: pd.DataFrame) -> bool:
    t = last_candle_type(df)
    return t == "1"

def last_outside(df: pd.DataFrame) -> bool:
    t = last_candle_type(df)
    return t == "3"
