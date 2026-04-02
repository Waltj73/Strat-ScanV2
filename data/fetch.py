import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

# =========================
# INDEX CLEANING
# =========================
def _ensure_datetime_index(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Force datetime index ALWAYS
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

    # Remove timezone issues
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        try:
            df.index = df.index.tz_convert(None)
        except Exception:
            pass

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    return df


# =========================
# FETCH DATA
# =========================
@st.cache_data(ttl=60 * 20, show_spinner=False)
def get_hist(ticker: str, period: str = "3y"):

    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = _ensure_datetime_index(df)

    # Flatten columns if needed
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).title() for c in df.columns]

    # Ensure Close exists
    if "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    if "Volume" not in df.columns:
        df["Volume"] = 0

    needed = ["Open", "High", "Low", "Close", "Volume"]

    if not set(needed).issubset(df.columns):
        return pd.DataFrame()

    df = df[needed].copy()

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    return df


# =========================
# RESAMPLE (ABSOLUTE FIX)
# =========================
def resample_ohlc(df, rule):

    if df is None or df.empty:
        return pd.DataFrame()

    df = _ensure_datetime_index(df)

    if df.empty:
        return pd.DataFrame()

    # 🔥 FORCE RULE SAFELY (THIS IS THE FIX)
    rule_map = {
        "M": "ME",
        "W": "W-FRI",
        "D": "D",
        "ME": "ME",
        "W-FRI": "W-FRI"
    }

    rule = str(rule).strip().upper()
    rule = rule_map.get(rule, "ME")  # default to safe value

    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    if df.empty:
        return pd.DataFrame()

    try:
        out = df.resample(rule).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        })
    except Exception:
        return pd.DataFrame()

    return out.dropna()
