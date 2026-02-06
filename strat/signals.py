import pandas as pd
from typing import Dict

from strat.candles import last_two, classify_candle, is_green, is_red

def strat_bull(df: pd.DataFrame) -> bool:
    pair = last_two(df)
    if not pair:
        return False
    cur, prev = pair
    return classify_candle(cur, prev) == "2U" and is_green(cur)

def strat_bear(df: pd.DataFrame) -> bool:
    pair = last_two(df)
    if not pair:
        return False
    cur, prev = pair
    return classify_candle(cur, prev) == "2D" and is_red(cur)

def strat_inside(df: pd.DataFrame) -> bool:
    pair = last_two(df)
    if not pair:
        return False
    cur, prev = pair
    return classify_candle(cur, prev) == "1"

def strat_212_up(df: pd.DataFrame) -> bool:
    # Needs 4 bars: prev_a, a, b, c
    if df is None or df.empty or len(df) < 4:
        return False
    prev_a = df.iloc[-4]
    a = df.iloc[-3]
    b = df.iloc[-2]
    c = df.iloc[-1]
    return (
        classify_candle(a, prev_a) == "2U"
        and classify_candle(b, a) == "1"
        and classify_candle(c, b) == "2U"
    )

def strat_212_dn(df: pd.DataFrame) -> bool:
    if df is None or df.empty or len(df) < 4:
        return False
    prev_a = df.iloc[-4]
    a = df.iloc[-3]
    b = df.iloc[-2]
    c = df.iloc[-1]
    return (
        classify_candle(a, prev_a) == "2D"
        and classify_candle(b, a) == "1"
        and classify_candle(c, b) == "2D"
    )

def compute_flags(d: pd.DataFrame, w: pd.DataFrame, m: pd.DataFrame) -> Dict[str, bool]:
    return {
        "D_Bull": strat_bull(d),
        "W_Bull": strat_bull(w),
        "M_Bull": strat_bull(m),

        "D_Bear": strat_bear(d),
        "W_Bear": strat_bear(w),
        "M_Bear": strat_bear(m),

        "D_Inside": strat_inside(d),
        "W_Inside": strat_inside(w),
        "M_Inside": strat_inside(m),

        "D_212Up": strat_212_up(d),
        "W_212Up": strat_212_up(w),

        "D_212Dn": strat_212_dn(d),
        "W_212Dn": strat_212_dn(w),
    }
