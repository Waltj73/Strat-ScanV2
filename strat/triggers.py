import pandas as pd
from typing import Optional, Tuple

from strat.signals import strat_inside

def best_trigger(d: pd.DataFrame, w: pd.DataFrame) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    Weekly-first trigger (STRAT standard):
      If Weekly Inside Bar -> use Weekly levels
      else if Daily Inside Bar -> use Daily levels
    Returns:
      (tf, entry, stop) where entry/stop are LONG levels:
        entry = IB high, stop = IB low
    """
    if w is not None and not w.empty and strat_inside(w) and len(w) >= 2:
        cur = w.iloc[-1]
        hi, lo = float(cur["High"]), float(cur["Low"])
        return "W", hi, lo

    if d is not None and not d.empty and strat_inside(d) and len(d) >= 2:
        cur = d.iloc[-1]
        hi, lo = float(cur["High"]), float(cur["Low"])
        return "D", hi, lo

    return None, None, None

def trigger_status(d: pd.DataFrame, w: pd.DataFrame, m: pd.DataFrame) -> str:
    tf, entry, stop = best_trigger(d, w)

    d_ready = bool(strat_inside(d) and tf == "D" and entry is not None and stop is not None)
    w_ready = bool(strat_inside(w) and tf == "W" and entry is not None and stop is not None)
    m_inside = bool(m is not None and not m.empty and strat_inside(m))

    return f"D: {'READY' if d_ready else 'WAIT'} | W: {'READY' if w_ready else 'WAIT'} | M: {'INSIDE' if m_inside else '—'}"
