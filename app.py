# app.py (v1.3.2)
# STRAT Scanner — Two Scenarios + Weekly + Monthly Completed vs Current Month Types (Fix)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime, timezone

st.set_page_config(page_title="2D Scan (STRAT) v1.3.2", layout="wide")

CHECK = "✅"
XMARK = "—"

def ck(v: bool) -> str:
    return CHECK if bool(v) else XMARK

st.markdown(
    """
    <style>
      .stDataFrame { padding-top: 0.25rem; }
      h1, h2, h3 { margin-bottom: 0.25rem; }
      .muted { color: rgba(120,120,120,0.9); font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_TOP50 = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","BRK-B","LLY","AVGO",
    "TSLA","JPM","V","UNH","XOM","WMT","JNJ","MA","PG","COST",
    "HD","ABBV","MRK","ORCL","CVX","KO","BAC","NFLX","CRM","TMO",
    "PEP","LIN","WFC","ACN","MCD","CSCO","QCOM","TXN","INTU","DIS",
    "ABT","IBM","GE","AMD","CAT","AMGN","ADBE","NOW","PM","GS"
]

# -----------------------------
# Wikipedia fetch (avoid 403)
# -----------------------------
def _fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()
    return r.text

@st.cache_data(ttl=60 * 60 * 24)
def get_sp500_tickers() -> list[str]:
    html = _fetch_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = pd.read_html(StringIO(html))[0]
    tickers = df["Symbol"].astype(str).str.strip().tolist()
    return sorted(set([t.replace(".", "-") for t in tickers]))

@st.cache_data(ttl=60 * 60 * 24)
def get_nasdaq100_tickers() -> list[str]:
    html = _fetch_html("https://en.wikipedia.org/wiki/Nasdaq-100")
    tables = pd.read_html(StringIO(html))

    best = None
    for t in tables:
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = [" ".join(map(str, c)).strip() for c in t.columns]
        cols = [str(c).lower() for c in t.columns]
        if any("ticker" in c for c in cols) or any("symbol" in c for c in cols):
            best = t
            break
    if best is None:
        raise ValueError("Could not find Nasdaq-100 constituents table.")

    ticker_col = None
    for c in best.columns:
        cl = str(c).lower()
        if "ticker" in cl or "symbol" in cl:
            ticker_col = c
            break
    if ticker_col is None:
        raise ValueError("Could not locate ticker/symbol column on Nasdaq-100 table.")

    tickers = best[ticker_col].astype(str).str.strip().tolist()
    return sorted(set([t.replace(".", "-") for t in tickers]))

@st.cache_data(ttl=60 * 60 * 12)
def get_top50_by_marketcap(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            fi = getattr(tk, "fast_info", {}) or {}
            mcap = fi.get("marketCap", None)
            if mcap is None:
                info = tk.info or {}
                mcap = info.get("marketCap", None)
            if mcap is None:
                continue
            rows.append({"Ticker": t, "MarketCap": float(mcap)})
        except Exception:
            continue
    df = pd.DataFrame(rows).dropna()
    return df.sort_values("MarketCap", ascending=False).head(50).reset_index(drop=True)

# -----------------------------
# Data fetch
# -----------------------------
@st.cache_data(ttl=60 * 15)
def fetch_batch_daily(tickers: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    data = yf.download(
        " ".join(tickers),
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    out: dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.get_level_values(0):
                df = data[t].copy().dropna()
                df = df.rename(columns=str.title)
                keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
                df = df[keep].dropna()
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                if len(df) > 10:
                    out[t] = df
    else:
        df = data.copy().dropna().rename(columns=str.title)
        keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
        df = df[keep].dropna()
        df.index = pd.to_datetime(df.index)
        out[tickers[0]] = df.sort_index()
    return out

@st.cache_data(ttl=60 * 15)
def fetch_single_daily(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)
    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df = df[keep].dropna()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

# -----------------------------
# Resampling
# -----------------------------
def to_monthly_ohlc(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d.index = pd.to_datetime(d.index)
    d = d.sort_index()
    monthly = pd.DataFrame(
        {
            "Open": d["Open"].resample("MS").first(),
            "High": d["High"].resample("MS").max(),
            "Low": d["Low"].resample("MS").min(),
            "Close": d["Close"].resample("MS").last(),
        }
    ).dropna()
    return monthly

def to_weekly_ohlc(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d.index = pd.to_datetime(d.index)
    d = d.sort_index()
    weekly = pd.DataFrame(
        {
            "Open": d["Open"].resample("W-FRI").first(),
            "High": d["High"].resample("W-FRI").max(),
            "Low": d["Low"].resample("W-FRI").min(),
            "Close": d["Close"].resample("W-FRI").last(),
        }
    ).dropna()
    return weekly

# -----------------------------
# STRAT Bar Type (single source of truth)
# -----------------------------
def bar_type(cur_high, cur_low, prev_high, prev_low) -> str:
    if cur_high < prev_high and cur_low > prev_low:
        return "1"
    if cur_high > prev_high and cur_low >= prev_low:
        return "2U"
    if cur_low < prev_low and cur_high <= prev_high:
        return "2D"
    if cur_high > prev_high and cur_low < prev_low:
        return "3"
    return XMARK

def strat_type_latest(df: pd.DataFrame) -> str:
    if df is None or df.empty or len(df) < 2:
        return XMARK
    cur = df.iloc[-1]
    prev = df.iloc[-2]
    return bar_type(float(cur["High"]), float(cur["Low"]), float(prev["High"]), float(prev["Low"]))

def strat_type_at_pos(df: pd.DataFrame, pos: int) -> str:
    # pos refers to df.iloc[pos] vs df.iloc[pos-1]
    if df is None or df.empty or len(df) < 2:
        return XMARK
    if pos == 0 or abs(pos) > len(df) - 1:
        return XMARK
    cur = df.iloc[pos]
    prev = df.iloc[pos - 1]
    return bar_type(float(cur["High"]), float(cur["Low"]), float(prev["High"]), float(prev["Low"]))

def is_current_month_in_progress(monthly: pd.DataFrame) -> bool:
    if monthly is None or monthly.empty:
        return False
    now = pd.Timestamp(datetime.now(timezone.utc).date())
    last_idx = monthly.index[-1]
    return (last_idx.year == now.year) and (last_idx.month == now.month)

def monthly_context(monthly: pd.DataFrame) -> dict:
    """
    Returns:
      - Completed month type (last fully closed month)
      - Current month type (in-progress month) if available
      - Completed month start date & target high
      - Completed month green flag
    """
    if monthly is None or monthly.empty or len(monthly) < 3:
        return {"ok": False}

    in_progress = is_current_month_in_progress(monthly)

    # Completed month index:
    comp_pos = -2 if in_progress else -1
    if len(monthly) < abs(comp_pos) + 2:
        return {"ok": False}

    comp = monthly.iloc[comp_pos]
    comp_prev = monthly.iloc[comp_pos - 1]
    completed_type = bar_type(float(comp["High"]), float(comp["Low"]), float(comp_prev["High"]), float(comp_prev["Low"]))
    completed_green = float(comp["Close"]) > float(comp["Open"])
    completed_start = monthly.index[comp_pos]
    target = float(comp["High"])

    current_type = XMARK
    current_start = None
    if in_progress and len(monthly) >= 2:
        # Current month is last row (partial) compared to completed month
        cur_pos = -1
        cur = monthly.iloc[cur_pos]
        prev = monthly.iloc[cur_pos - 1]
        current_type = bar_type(float(cur["High"]), float(cur["Low"]), float(prev["High"]), float(prev["Low"]))
        current_start = monthly.index[cur_pos]

    return {
        "ok": True,
        "completed_month_start": completed_start,
        "completed_month_type": completed_type,
        "completed_month_green": bool(completed_green),
        "current_month_start": current_start,
        "current_month_type": current_type,
        "target": target,
    }

def daily_latest_block(daily: pd.DataFrame) -> dict:
    if daily is None or daily.empty or len(daily) < 2:
        return {"ok": False}
    d = daily.iloc[-1]
    p = daily.iloc[-2]
    d_type = bar_type(float(d["High"]), float(d["Low"]), float(p["High"]), float(p["Low"]))
    return {
        "ok": True,
        "bar_date": daily.index[-1],
        "last_close": float(d["Close"]),
        "daily_type": d_type,
        "is_inside": d_type == "1",
        "is_2u": d_type == "2U",
        "is_2d": d_type == "2D",
        "is_3": d_type == "3",
    }

# -----------------------------
# Probability (unchanged)
# -----------------------------
def monthly_hit_probability_after_2d_green(daily: pd.DataFrame, years: int = 2) -> dict:
    if daily is None or daily.empty:
        return {"setups": 0, "hit_rate": 0.0, "avg_days": None}

    cutoff = daily.index.max() - pd.DateOffset(years=years)
    d = daily[daily.index >= cutoff].copy()
    if d.empty or len(d) < 60:
        return {"setups": 0, "hit_rate": 0.0, "avg_days": None}

    mdf = to_monthly_ohlc(d)
    if len(mdf) < 3:
        return {"setups": 0, "hit_rate": 0.0, "avg_days": None}

    setups = 0
    hits = 0
    days_to_hit = []

    for i in range(1, len(mdf) - 1):
        m = mdf.iloc[i]
        mp = mdf.iloc[i - 1]
        mtype = bar_type(float(m["High"]), float(m["Low"]), float(mp["High"]), float(mp["Low"]))
        green = float(m["Close"]) > float(m["Open"])
        if not (mtype == "2D" and green):
            continue

        setups += 1
        target = float(m["High"])

        next_start = mdf.index[i + 1]
        next_end = next_start + pd.offsets.MonthEnd(1)
        nm = d[(d.index >= next_start) & (d.index <= next_end)]
        if nm.empty:
            continue

        hit_rows = nm[nm["High"] >= target]
        if not hit_rows.empty:
            hits += 1
            days_to_hit.append((hit_rows.index[0] - nm.index[0]).days)

    hit_rate = (hits / setups) if setups > 0 else 0.0
    avg_days = float(np.mean(days_to_hit)) if days_to_hit else None
    return {"setups": setups, "hit_rate": hit_rate, "avg_days": avg_days}

# -----------------------------
# Scoring
# -----------------------------
def tier(score: float) -> str:
    if score >= 80: return "A"
    if score >= 65: return "B"
    if score >= 50: return "C"
    return "D"

def score_room(room_pct: float) -> float:
    return max(0.0, min(15.0, room_pct))

def score_prob(prob_pct: float) -> float:
    return max(0.0, min(10.0, (prob_pct / 100.0) * 10.0))

def score_s1(month_ok: bool, daily_actionable: bool, room_pct: float, prob_pct: float) -> float:
    s = 0.0
    s += 45.0 if month_ok else 0.0
    s += 30.0 if daily_actionable else 0.0
    s += score_room(room_pct)
    s += score_prob(prob_pct)
    return round(s, 1)

def score_s2(month_ok: bool, daily_actionable: bool, room_pct: float, prob_pct: float) -> float:
    s = 0.0
    s += 40.0 if month_ok else 0.0
    s += 35.0 if daily_actionable else 0.0
    s += score_room(room_pct)
    s += score_prob(prob_pct)
    return round(s, 1)

# -----------------------------
# Scenarios
# -----------------------------
def scan_scenario_1(ticker: str, daily: pd.DataFrame) -> dict:
    if daily is None or daily.empty or len(daily) < 60:
        return {"Ticker": ticker, "Score": 0.0, "Tier": "D", "Status": "No data"}

    mdf = to_monthly_ohlc(daily)
    wdf = to_weekly_ohlc(daily)

    mc = monthly_context(mdf)
    dl = daily_latest_block(daily)

    if not mc.get("ok") or not dl.get("ok"):
        return {"Ticker": ticker, "Score": 0.0, "Tier": "D", "Status": "Insufficient"}

    weekly_type = strat_type_latest(wdf)

    # Scenario 1 uses COMPLETED month context
    completed_type = mc["completed_month_type"]
    completed_green = mc["completed_month_green"]
    month_ok = (completed_type == "2D") and completed_green

    daily_actionable = dl["is_2d"] or dl["is_inside"]

    target = mc["target"]
    last_close = dl["last_close"]
    room_pct = (target - last_close) / last_close * 100.0

    prob = monthly_hit_probability_after_2d_green(daily, years=2)
    prob_pct = prob["hit_rate"] * 100.0

    score = score_s1(month_ok, daily_actionable, room_pct, prob_pct)

    return {
        "Ticker": ticker,
        "Score": score,
        "Tier": tier(score),
        "Weekly": weekly_type,
        "Completed_Month": mc["completed_month_start"].strftime("%Y-%m"),
        "Completed_Month_Type": completed_type,
        "Completed_Month_Green": ck(completed_green),
        "Current_Month": (mc["current_month_start"].strftime("%Y-%m") if mc["current_month_start"] is not None else XMARK),
        "Current_Month_Type": mc["current_month_type"],
        "Actionable_Daily": ck(daily_actionable),
        "Daily_Type": dl["daily_type"],
        "Last_Bar": pd.to_datetime(dl["bar_date"]).strftime("%Y-%m-%d"),
        "Last_Close": round(last_close, 2),
        "Target_PrevMonthHigh": round(target, 2),
        "Room_%": round(room_pct, 2),
        "Prob_Hit_%_2Y": round(prob_pct, 1),
        "Status": "OK",
    }

def scan_scenario_2(ticker: str, daily: pd.DataFrame, allow_daily_2u_as_actionable: bool) -> dict:
    if daily is None or daily.empty or len(daily) < 60:
        return {"Ticker": ticker, "Score": 0.0, "Tier": "D", "Status": "No data"}

    mdf = to_monthly_ohlc(daily)
    wdf = to_weekly_ohlc(daily)

    mc = monthly_context(mdf)
    dl = daily_latest_block(daily)

    if not mc.get("ok") or not dl.get("ok"):
        return {"Ticker": ticker, "Score": 0.0, "Tier": "D", "Status": "Insufficient"}

    weekly_type = strat_type_latest(wdf)

    # Scenario 2 uses COMPLETED month context
    completed_type = mc["completed_month_type"]
    month_ok = (completed_type == "2U")

    daily_actionable = dl["is_inside"] or (allow_daily_2u_as_actionable and dl["is_2u"])

    target = mc["target"]
    last_close = dl["last_close"]
    room_pct = (target - last_close) / last_close * 100.0

    prob = monthly_hit_probability_after_2d_green(daily, years=2)
    prob_pct = prob["hit_rate"] * 100.0

    score = score_s2(month_ok, daily_actionable, room_pct, prob_pct)

    return {
        "Ticker": ticker,
        "Score": score,
        "Tier": tier(score),
        "Weekly": weekly_type,
        "Completed_Month": mc["completed_month_start"].strftime("%Y-%m"),
        "Completed_Month_Type": completed_type,
        "Current_Month": (mc["current_month_start"].strftime("%Y-%m") if mc["current_month_start"] is not None else XMARK),
        "Current_Month_Type": mc["current_month_type"],
        "Actionable_Daily": ck(daily_actionable),
        "Daily_Type": dl["daily_type"],
        "Last_Bar": pd.to_datetime(dl["bar_date"]).strftime("%Y-%m-%d"),
        "Last_Close": round(last_close, 2),
        "Target_PrevMonthHigh": round(target, 2),
        "Room_%": round(room_pct, 2),
        "Prob_Hit_%_2Y": round(prob_pct, 1),
        "Status": "OK",
    }

# -----------------------------
# UI
# -----------------------------
st.title("2D Scan (STRAT) — v1.3.2 (Monthly Context Fix)")
st.markdown(
    '<div class="muted">'
    'Now shows <b>Completed Month Type</b> (last closed) AND <b>Current Month Type</b> (in-progress) '
    'so it matches what you see on TOS.'
    '</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Universe")
    universe = st.selectbox(
        "Select universe",
        ["Top 50 (Mega Caps: S&P500 + Nasdaq-100)", "Fallback Top 50 (manual)", "Custom (paste)"],
        index=0,
    )
    period = st.selectbox("History period", ["6mo", "1y", "2y", "5y"], index=2)

    st.header("Scenario 2 Daily Actionable")
    allow_daily_2u_as_actionable = st.checkbox("Count daily 2U as actionable (optional)", value=False)

    st.header("Ranking Filter")
    min_score = st.slider("Minimum Score", 0, 100, 60)

    run = st.button("Run Scan", type="primary")

tickers: list[str] = []
note = None

try:
    if universe == "Fallback Top 50 (manual)":
        tickers = DEFAULT_TOP50
        note = "Using manual fallback Top 50."
    elif universe.startswith("Top 50"):
        sp = get_sp500_tickers()
        ndx = get_nasdaq100_tickers()
        combined = sorted(set(sp + ndx))
        top50_df = get_top50_by_marketcap(combined)
        if top50_df is None or top50_df.empty:
            tickers = DEFAULT_TOP50
            note = "Top 50 market-cap fetch empty; using fallback Top 50."
        else:
            tickers = top50_df["Ticker"].tolist()
            note = "Using Top 50 mega caps (by market cap)."
    else:
        tickers_text = st.text_area(
            "Tickers (comma/newline separated)",
            value="AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, AVGO, AMD, PLTR",
            height=120,
        )
        tickers = [t.strip().upper() for t in tickers_text.replace("\n", ",").split(",") if t.strip()]
        note = f"Using custom list ({len(tickers)} tickers)."
except Exception as e:
    tickers = DEFAULT_TOP50
    note = f"Universe fetch failed ({e}). Using fallback Top 50."

if note:
    st.info(note)

if not run:
    st.write("Click **Run Scan** to generate results.")
    st.stop()

st.info(f"Downloading daily data (batch) for {len(tickers)} tickers…")
batch = fetch_batch_daily(tickers, period=period)

res1, res2 = [], []
progress = st.progress(0)

for i, t in enumerate(tickers, start=1):
    df = batch.get(t, pd.DataFrame())
    if df is None or df.empty:
        df = fetch_single_daily(t, period=period)

    res1.append(scan_scenario_1(t, df))
    res2.append(scan_scenario_2(t, df, allow_daily_2u_as_actionable))

    progress.progress(i / max(len(tickers), 1))

df1 = pd.DataFrame(res1)
df2 = pd.DataFrame(res2)

if not df1.empty:
    df1 = df1[df1["Score"] >= float(min_score)].copy()
    df1 = df1.sort_values(["Score", "Room_%"], ascending=[False, False]).reset_index(drop=True)

if not df2.empty:
    df2 = df2[df2["Score"] >= float(min_score)].copy()
    df2 = df2.sort_values(["Score", "Room_%"], ascending=[False, False]).reset_index(drop=True)

tab1, tab2 = st.tabs(["Scenario 1: Completed M 2D+Green → Daily 2D/1", "Scenario 2: Completed M 2U → Daily 1 (+ opt 2U)"])

with tab1:
    st.subheader("Scenario 1 — Ranked Results")
    st.dataframe(df1, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Scenario 2 — Ranked Results")
    st.dataframe(df2, use_container_width=True, hide_index=True)
