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
    except Exception:
        try:
            df.index = df.index.tz_convert(None)
        except Exception:
            pass

    df = df[~df.index.duplicated(keep="last")]

    return df


# =========================
# CLEAN YFINANCE DATA
# =========================
def _flatten_yf_columns(df, ticker):
    if df is None or df.empty:
        return pd.DataFrame()

    df = _ensure_datetime_index(df)
    if df.empty:
        return pd.DataFrame()

    # Flatten multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df.columns = [str(c).title() for c in df.columns]

    if "Adj Close" in df.columns and "Close" not in df.columns:
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
# FETCH DATA
# =========================
@st.cache_data(ttl=60 * 20, show_spinner=False)
def get_hist(ticker: str, period: str = "3y"):

    try:
        raw = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        return pd.DataFrame()

    df = _flatten_yf_columns(raw, ticker)
    df = _ensure_datetime_index(df)

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

    # 🔥 HARD FIX — NEVER LET "M" HIT PANDAS
    rule = str(rule).strip().upper()

    if rule == "M":
        rule = "ME"
    elif rule == "W":
        rule = "W-FRI"
    elif rule not in ["ME", "W-FRI", "D"]:
        return pd.DataFrame()

    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    if df.empty:
        return pd.DataFrame()

    try:
        resampled = df.resample(rule).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        })
    except Exception:
        return pd.DataFrame()

    return resampled.dropna()
