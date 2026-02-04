# app.py (v1.3.1)
# STRAT Scanner — Two Scenarios + Weekly Display + Correct STRAT Type Derivation

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime, timezone

st.set_page_config(page_title="2D Scan (STRAT) v1.3.1", layout="wide")

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

def pick_eval_month_pos(monthly: pd.DataFrame, use_last_completed_month: bool = True) -> int:
    now = pd.Timestamp(datetime.now(timezone.utc).date())
    last_idx = monthly.index[-1]
    in_progress = (last_idx.year == now.year) and (last_idx.month == now.month)
    return (-2 if in_progress else -1) if use_last_completed_month else -1

def monthly_eval_block(monthly: pd.DataFrame, use_last_completed_month: bool) -> dict:
    if monthly is None or monthly.empty or len(monthly) < 3:
        return {"ok": False}

    pos = pick_eval_month_pos(monthly, use_last_completed_month)
    if len(monthly) < (abs(pos) + 2):
        return {"ok": False}

    eval_df = monthly.iloc[:pos+1] if pos != -1 else monthly
    # Get the evaluated month candle (pos) and its previous (pos-1)
    m = monthly.iloc[pos]
    mp = monthly.iloc[pos-1]

    m_type = bar_type(float(m["High"]), float(m["Low"]), float(mp["High"]), float(mp["Low"]))
    m_green = float(m["Close"]) > float(m["Open"])
    target = float(m["High"])

    return {
        "ok": True,
        "eval_month_start": monthly.index[pos],
        "monthly_type": m_type,
        "monthly_green": bool(m_green),
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
# Scoring (kept simple)
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
    # Monthly (45) + Daily (30) + Room (15) + Prob (10) = 100
    s = 0.0
    s += 45.0 if month_ok else 0.0
    s += 30.0 if daily_actionable else 0.0
    s += score_room(room_pct)
    s += score_prob(prob_pct)
    return round(s, 1)

def score_s2(month_ok: bool, daily_actionable: bool, room_pct: float, prob_pct: float) -> float:
    # Monthly (40) + Daily (35) + Room (15) + Prob (10) = 100
    s = 0.0
    s += 40.0 if month_ok else 0.0
    s += 35.0 if daily_actionable else 0.0
    s += score_room(room_pct)
    s += score_prob(prob_pct)
    return round(s, 1)

# -----------------------------
# Scenario scan functions
# -----------------------------
def scan_scenario_1(ticker: str, daily: pd.DataFrame, use_last_completed_month: bool) -> dict:
    if daily is None or daily.empty or len(daily) < 60:
        return {"Ticker": ticker, "Score": 0.0, "Tier": "D", "Status": "No data"}

    mdf = to_monthly_ohlc(daily)
    wdf = to_weekly_ohlc(daily)

    m = monthly_eval_block(mdf, use_last_completed_month)
    d = daily_latest_block(daily)

    if not m.get("ok") or not d.get("ok"):
        return {"Ticker": ticker, "Score": 0.0, "Tier": "D", "Status": "Insufficient"}

    weekly_type = strat_type_latest(wdf)

    monthly_type = m["monthly_type"]
    monthly_green = m["monthly_green"]

    # Scenario 1 monthly condition: 2D + green
    month_ok = (monthly_type == "2D") and monthly_green

    # Scenario 1 daily actionable: daily 2D OR inside(1)
    daily_actionable = d["is_2d"] or d["is_inside"]

    target = m["target"]
    last_close = d["last_close"]
    room_pct = (target - last_close) / last_close * 100.0

    prob = monthly_hit_probability_after_2d_green(daily, years=2)
    prob_pct = prob["hit_rate"] * 100.0

    score = score_s1(month_ok, daily_actionable, room_pct, prob_pct)

    return {
        "Ticker": ticker,
        "Score": score,
        "Tier": tier(score),
        "Weekly": weekly_type,
        "Monthly_Type": monthly_type,
        "Monthly_Green": ck(monthly_green),
        "Actionable_Daily": ck(daily_actionable),
        "Daily_Type": d["daily_type"],
        "Eval_Month": m["eval_month_start"].strftime("%Y-%m"),
        "Last_Bar": pd.to_datetime(d["bar_date"]).strftime("%Y-%m-%d"),
        "Last_Close": round(last_close, 2),
        "Target_PrevMonthHigh": round(target, 2),
        "Room_%": round(room_pct, 2),
        "Prob_Hit_%_2Y": round(prob_pct, 1),
        "Status": "OK",
    }

def scan_scenario_2(ticker: str, daily: pd.DataFrame, use_last_completed_month: bool, allow_daily_2u_as_actionable: bool) -> dict:
    if daily is None or daily.empty or len(daily) < 60:
        return {"Ticker": ticker, "Score": 0.0, "Tier": "D", "Status": "No data"}

    mdf = to_monthly_ohlc(daily)
    wdf = to_weekly_ohlc(daily)

    m = monthly_eval_block(mdf, use_last_completed_month)
    d = daily_latest_block(daily)

    if not m.get("ok") or not d.get("ok"):
        return {"Ticker": ticker, "Score": 0.0, "Tier": "D", "Status": "Insufficient"}

    weekly_type = strat_type_latest(wdf)
    monthly_type = m["monthly_type"]

    # Scenario 2 monthly condition: 2U
    month_ok = (monthly_type == "2U")

    # Scenario 2 daily actionable:
    # - always inside (1) is actionable (compression breakout)
    # - optionally allow daily 2U to count as actionable (toggle)
    daily_actionable = d["is_inside"] or (allow_daily_2u_as_actionable and d["is_2u"])

    target = m["target"]
    last_close = d["last_close"]
    room_pct = (target - last_close) / last_close * 100.0

    prob = monthly_hit_probability_after_2d_green(daily, years=2)
    prob_pct = prob["hit_rate"] * 100.0

    score = score_s2(month_ok, daily_actionable, room_pct, prob_pct)

    return {
        "Ticker": ticker,
        "Score": score,
        "Tier": tier(score),
        "Weekly": weekly_type,
        "Monthly_Type": monthly_type,
        "Actionable_Daily": ck(daily_actionable),
        "Daily_Type": d["daily_type"],
        "Eval_Month": m["eval_month_start"].strftime("%Y-%m"),
        "Last_Bar": pd.to_datetime(d["bar_date"]).strftime("%Y-%m-%d"),
        "Last_Close": round(last_close, 2),
        "Target_PrevMonthHigh": round(target, 2),
        "Room_%": round(room_pct, 2),
        "Prob_Hit_%_2Y": round(prob_pct, 1),
        "Status": "OK",
    }

# -----------------------------
# UI
# -----------------------------
st.title("2D Scan (STRAT) — v1.3.1 (Fixed Types)")
st.markdown(
    '<div class="muted">'
    'Scenario 1: <b>M 2D+Green</b> → Actionable Daily (<b>2D or 1</b>) • '
    'Scenario 2: <b>M 2U</b> → Actionable Daily (<b>1</b> + optional <b>2U</b>) • '
    'Weekly shown as <b>1/2U/2D/3</b> (context only)'
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

    st.header("Monthly Handling")
    use_last_completed_month = st.checkbox("Use last COMPLETED month (stable)", value=True)

    st.header("Scenario 2 Daily Actionable")
    allow_daily_2u_as_actionable = st.checkbox("Count daily 2U as actionable (optional)", value=False)

    st.header("Ranking Filter")
    min_score = st.slider("Minimum Score", 0, 100, 60)

    run = st.button("Run Scan", type="primary")

# Universe build
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

    res1.append(scan_scenario_1(t, df, use_last_completed_month))
    res2.append(scan_scenario_2(t, df, use_last_completed_month, allow_daily_2u_as_actionable))

    progress.progress(i / max(len(tickers), 1))

df1 = pd.DataFrame(res1)
df2 = pd.DataFrame(res2)

if not df1.empty:
    df1 = df1[df1["Score"] >= float(min_score)].copy()
    df1 = df1.sort_values(["Score", "Room_%"], ascending=[False, False]).reset_index(drop=True)

if not df2.empty:
    df2 = df2[df2["Score"] >= float(min_score)].copy()
    df2 = df2.sort_values(["Score", "Room_%"], ascending=[False, False]).reset_index(drop=True)

tab1, tab2 = st.tabs(["Scenario 1: M 2D+Green → Daily 2D/1", "Scenario 2: M 2U → Daily 1 (+ optional 2U)"])

with tab1:
    st.subheader("Scenario 1 — Ranked Results")
    st.dataframe(df1, use_container_width=True, hide_index=True)
    st.write(f"Shown: **{len(df1)}** / {len(res1)} (min score = {min_score})")
    st.caption("Key columns to verify: Monthly_Type, Monthly_Green, Daily_Type, Actionable_Daily.")

with tab2:
    st.subheader("Scenario 2 — Ranked Results")
    st.dataframe(df2, use_container_width=True, hide_index=True)
    st.write(f"Shown: **{len(df2)}** / {len(res2)} (min score = {min_score})")
    st.caption("Key columns to verify: Monthly_Type, Daily_Type, Actionable_Daily. Weekly is context only.")
