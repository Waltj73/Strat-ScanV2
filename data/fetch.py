import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

# =========================
# INDEX CLEANING
# =========================
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

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
# COLUMN CLEANING
# =========================
def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()

    return df


# =========================
# YFINANCE CLEANING
# =========================
def _flatten_yf_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = _ensure_datetime_index(df)
    if df.empty:
        return pd.DataFrame()

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
# DATA FETCH
# =========================
@st.cache_data(ttl=60 * 20, show_spinner=False)
def get_hist(ticker: str, period: str = "3y") -> pd.DataFrame:
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
# RESAMPLING (HARD FIX)
# =========================
def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:

    if df is None or df.empty:
        return pd.DataFrame()

    df = _ensure_datetime_index(df)

    if df.empty:
        return pd.DataFrame()

    # 🔥 FORCE VALID RULE (NO EXCEPTIONS)
    if not isinstance(rule, str):
        return pd.DataFrame()

    rule = rule.strip().upper()

    if rule == "M":
        rule = "ME"
    elif rule == "W":
        rule = "W-FRI"
    elif rule not in ["ME", "W-FRI", "D"]:
        return pd.DataFrame()

    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    if df.empty:
        return pd.DataFrame()

    def safe_first(x):
        return x.iloc[0] if len(x) else np.nan

    def safe_last(x):
        return x.iloc[-1] if len(x) else np.nan

    # 🔥 GUARANTEED SAFE RESAMPLE
    try:
        g = df.resample(rule)
    except Exception:
        return pd.DataFrame()

    out = pd.DataFrame({
        "Open": g["Open"].apply(safe_first),
        "High": g["High"].max(),
        "Low": g["Low"].min(),
        "Close": g["Close"].apply(safe_last),
        "Volume": g["Volume"].sum(),
    })

    out = out.dropna()

    return out
