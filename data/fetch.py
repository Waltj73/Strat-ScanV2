import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

# =========================
# CLEAN INDEX
# =========================
def _ensure_datetime_index(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.sort_index()

    try:
        df.index = df.index.tz_localize(None)
    except:
        pass

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
            progress=False,
        )
    except:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = _ensure_datetime_index(df)

    # Normalize columns
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).title() for c in df.columns]

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

    df = df.dropna()

    return df


# =========================
# RESAMPLE (FINAL FIX)
# =========================
def resample_ohlc(df, rule):

    if df is None or df.empty:
        return pd.DataFrame()

    df = _ensure_datetime_index(df)

    if df.empty:
        return pd.DataFrame()

    # 🔥 FORCE SAFE RULES ONLY
    if rule == "M":
        rule = "ME"
    elif rule == "W":
        rule = "W-FRI"
    elif rule not in ["ME", "W-FRI", "D"]:
        return pd.DataFrame()

    try:
        resampled = df.resample(rule).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        })
    except:
        return pd.DataFrame()

    resampled = resampled.dropna()

    return resampled
