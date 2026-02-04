# app.py (v1.3)
# 2D Scan (STRAT) — Two Scenarios + Weekly Continuity Column + Ranked Output
#
# SCENARIO 1 (Pullback-Continuation):
#   Monthly: 2-Down AND Green
#   Daily actionable: 2D OR Inside
#   Target: evaluated month high (prior month high)
#
# SCENARIO 2 (Trend-Continuation):
#   Monthly: 2-Up
#   Daily actionable: 2U OR Inside
#   Target: evaluated month high (prior month high)
#
# Weekly Continuity (display only):
#   Weekly bar type shown as: 1 / 2U / 2D / 3
#
# Ranking Score (0–100) for each scenario:
#   Monthly context points + Daily trigger points + Room-to-target + Probability bonus
#
# Probability metric (2-year empirical):
#   After a 2-down green month, probability price hits THAT month’s high during the following month.
#   (Used as a context bonus in both scenarios.)
#
# requirements.txt:
#   streamlit
#   yfinance
#   pandas
#   numpy
#   lxml
#   html5lib
#   requests

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime, timezone

st.set_page_config(page_title="2D Scan (STRAT) v1.3", layout="wide")

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

# -----------------------------
# Fallback universe
# -----------------------------
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
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = _fetch_html(url)
    df = pd.read_html(StringIO(html))[0]
    tickers = df["Symbol"].astype(str).str.strip().tolist()
    return sorted(set([t.replace(".", "-") for t in tickers]))

@st.cache_data(ttl=60 * 60 * 24)
def get_nasdaq100_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    html = _fetch_html(url)
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
# Data fetch (batch + fallback)
# -----------------------------
@st.cache_data(ttl=60 * 15)
def fetch_batch_daily(tickers: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    joined = " ".join(tickers)
    data = yf.download(
        joined,
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
                keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                df = df[keep].dropna()
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                if len(df) > 10:
                    out[t] = df
    else:
        df = data.copy().dropna()
        df = df.rename(columns=str.title)
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].dropna()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        out[tickers[0]] = df

    return out

@st.cache_data(ttl=60 * 15)
def fetch_single_daily(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

# -----------------------------
# STRAT helpers
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
            "Volume": d["Volume"].resample("MS").sum() if "Volume" in d.columns else np.nan,
        }
    ).dropna(subset=["Open", "High", "Low", "Close"])
    return monthly

def to_weekly_ohlc(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d.index = pd.to_datetime(d.index)
    d = d.sort_index()

    weekly = pd.DataFrame(
        {
            "Open": d["Open"].resample("W-MON").first(),
            "High": d["High"].resample("W-MON").max(),
            "Low": d["Low"].resample("W-MON").min(),
            "Close": d["Close"].resample("W-MON").last(),
        }
    ).dropna()
    return weekly

def pick_eval_month(monthly: pd.DataFrame, use_last_completed_month: bool = True) -> int:
    now = pd.Timestamp(datetime.now(timezone.utc).date())
    last_idx = monthly.index[-1]
    in_progress = (last_idx.year == now.year) and (last_idx.month == now.month)
    return (-2 if in_progress else -1) if use_last_completed_month else -1

def monthly_flags(monthly: pd.DataFrame, use_last_completed_month: bool = True) -> dict:
    if monthly is None or monthly.empty or len(monthly) < 3:
        return {"ok": False}

    eval_pos = pick_eval_month(monthly, use_last_completed_month)
    if len(monthly) < (abs(eval_pos) + 2):
        return {"ok": False}

    m = monthly.iloc[eval_pos]
    m_prev = monthly.iloc[eval_pos - 1]

    m2d = (m["Low"] < m_prev["Low"]) and (m["High"] <= m_prev["High"])
    m2u = (m["High"] > m_prev["High"]) and (m["Low"] >= m_prev["Low"])
    mgreen = (m["Close"] > m["Open"])
    target = float(m["High"])

    return {
        "ok": True,
        "eval_month_start": monthly.index[eval_pos],
        "m2d": bool(m2d),
        "m2u": bool(m2u),
        "mgreen": bool(mgreen),
        "target": target,
    }

def strat_bar_type(df: pd.DataFrame) -> str:
    if df is None or df.empty or len(df) < 2:
        return XMARK

    cur = df.iloc[-1]
    prev = df.iloc[-2]

    h, l = float(cur["High"]), float(cur["Low"])
    ph, pl = float(prev["High"]), float(prev["Low"])

    if h < ph and l > pl:
        return "1"
    if h > ph and l >= pl:
        return "2U"
    if l < pl and h <= ph:
        return "2D"
    if h > ph and l < pl:
        return "3"
    return XMARK

def daily_bar_types(daily: pd.DataFrame) -> dict:
    if daily is None or daily.empty or len(daily) < 3:
        return {"ok": False}

    d = daily.iloc[-1]
    p = daily.iloc[-2]

    high, low = float(d["High"]), float(d["Low"])
    ph, pl = float(p["High"]), float(p["Low"])

    is_2u = (high > ph) and (low >= pl)
    is_2d = (low < pl) and (high <= ph)
    is_1  = (high < ph) and (low > pl)
    is_3  = (high > ph) and (low < pl)

    return {
        "ok": True,
        "bar_date": daily.index[-1],
        "last_close": float(d["Close"]),
        "is_2u": bool(is_2u),
        "is_2d": bool(is_2d),
        "is_1": bool(is_1),
        "is_3": bool(is_3),
        "inside": bool(is_1),
    }

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
        prev = mdf.iloc[i - 1]
        m = mdf.iloc[i]

        m2d = (m["Low"] < prev["Low"]) and (m["High"] <= prev["High"])
        green = (m["Close"] > m["Open"])
        if not (m2d and green):
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
# Scoring (scenario-specific)
# -----------------------------
def score_s1(m2d: bool, mgreen: bool, daily_2d: bool, inside: bool, room_pct: float, prob_pct: float) -> float:
    score = 0.0
    # Monthly (0–45)
    score += 25.0 if m2d else 0.0
    score += 20.0 if mgreen else 0.0
    # Daily (0–30)
    if daily_2d:
        score += 25.0
    elif inside:
        score += 20.0
    # Room (0–15)
    score += max(0.0, min(15.0, room_pct))
    # Prob bonus (0–10)
    score += max(0.0, min(10.0, (prob_pct / 100.0) * 10.0))
    return round(score, 1)

def score_s2(m2u: bool, daily_2u: bool, inside: bool, room_pct: float, prob_pct: float) -> float:
    score = 0.0
    # Monthly (0–40)
    score += 40.0 if m2u else 0.0
    # Daily (0–35)
    if daily_2u:
        score += 30.0
    elif inside:
        score += 20.0
    # Room (0–15)
    score += max(0.0, min(15.0, room_pct))
    # Prob bonus (0–10)
    score += max(0.0, min(10.0, (prob_pct / 100.0) * 10.0))
    return round(score, 1)

def tier(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    return "D"

# -----------------------------
# Scenario scanners
# -----------------------------
def scan_scenario_1(ticker: str, daily: pd.DataFrame, use_last_completed_month: bool) -> dict:
    if daily is None or daily.empty or len(daily) < 60:
        return {"Ticker": ticker, "Score": 0.0, "Tier": "D", "Status": "No data"}

    mdf = to_monthly_ohlc(daily)
    wdf = to_weekly_ohlc(daily)
    weekly_type = strat_bar_type(wdf)

    mf = monthly_flags(mdf, use_last_completed_month)
    df = daily_bar_types(daily)

    if not mf.get("ok") or not df.get("ok"):
        return {"Ticker": ticker, "Score": 0.0, "Tier": "D", "Status": "Insufficient"}

    m2d = mf["m2d"]
    mgreen = mf["mgreen"]
    daily_2d = df["is_2d"]
    inside = df["inside"]

    target = mf["target"]
    last_close = df["last_close"]
    room_pct = (target - last_close) / last_close * 100.0

    prob = monthly_hit_probability_after_2d_green(daily, years=2)
    prob_pct = prob["hit_rate"] * 100.0

    score = score_s1(m2d, mgreen, daily_2d, inside, room_pct, prob_pct)

    return {
        "Ticker": ticker,
        "Score": score,
        "Tier": tier(score),
        "Weekly": weekly_type,
        "M_2D": ck(m2d),
        "M_Green": ck(mgreen),
        "D_2D": ck(daily_2d),
        "D_Inside": ck(inside),
        "Eval_Month": mf["eval_month_start"].strftime("%Y-%m"),
        "Last_Bar": pd.to_datetime(df["bar_date"]).strftime("%Y-%m-%d"),
        "Last_Close": round(last_close, 2),
        "Target_PrevMonthHigh": round(target, 2),
        "Room_%": round(room_pct, 2),
        "Prob_Hit_%_2Y": round(prob_pct, 1),
        "Setups_2Y": int(prob["setups"]),
        "AvgDays_2Y": (round(prob["avg_days"], 1) if prob["avg_days"] is not None else None),
        "Status": "OK",
    }

def scan_scenario_2(ticker: str, daily: pd.DataFrame, use_last_completed_month: bool) -> dict:
    if daily is None or daily.empty or len(daily) < 60:
        return {"Ticker": ticker, "Score": 0.0, "Tier": "D", "Status": "No data"}

    mdf = to_monthly_ohlc(daily)
    wdf = to_weekly_ohlc(daily)
    weekly_type = strat_bar_type(wdf)

    mf = monthly_flags(mdf, use_last_completed_month)
    df = daily_bar_types(daily)

    if not mf.get("ok") or not df.get("ok"):
        return {"Ticker": ticker, "Score": 0.0, "Tier": "D", "Status": "Insufficient"}

    m2u = mf["m2u"]
    daily_2u = df["is_2u"]
    inside = df["inside"]

    target = mf["target"]
    last_close = df["last_close"]
    room_pct = (target - last_close) / last_close * 100.0

    prob = monthly_hit_probability_after_2d_green(daily, years=2)
    prob_pct = prob["hit_rate"] * 100.0

    score = score_s2(m2u, daily_2u, inside, room_pct, prob_pct)

    return {
        "Ticker": ticker,
        "Score": score,
        "Tier": tier(score),
        "Weekly": weekly_type,
        "M_2U": ck(m2u),
        "D_2U": ck(daily_2u),
        "D_Inside": ck(inside),
        "Eval_Month": mf["eval_month_start"].strftime("%Y-%m"),
        "Last_Bar": pd.to_datetime(df["bar_date"]).strftime("%Y-%m-%d"),
        "Last_Close": round(last_close, 2),
        "Target_PrevMonthHigh": round(target, 2),
        "Room_%": round(room_pct, 2),
        "Prob_Hit_%_2Y": round(prob_pct, 1),
        "Setups_2Y": int(prob["setups"]),
        "AvgDays_2Y": (round(prob["avg_days"], 1) if prob["avg_days"] is not None else None),
        "Status": "OK",
    }

# -----------------------------
# UI
# -----------------------------
st.title("2D Scan (STRAT) — v1.3")
st.markdown(
    '<div class="muted">Tabs: (1) <b>M 2D+Green → D 2D/Inside</b> • (2) <b>M 2U → D 2U/Inside</b> • Weekly shown as <b>1 / 2U / 2D / 3</b> • Output: <b>Ranked</b></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Universe")
    universe = st.selectbox(
        "Select universe",
        [
            "Top 50 (Mega Caps: S&P500 + Nasdaq-100)",
            "Fallback Top 50 (manual)",
            "Custom (paste)",
        ],
        index=0,
    )
    period = st.selectbox("History period (download)", ["6mo", "1y", "2y", "5y"], index=2)

    st.header("Monthly Handling")
    use_last_completed_month = st.checkbox("Use last COMPLETED month (stable)", value=True)

    st.header("Ranking Filter")
    min_score = st.slider("Minimum Score", 0, 100, 60)

    run = st.button("Run Scan", type="primary")

# Universe
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
            "Tickers (comma or newline separated)",
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

if not tickers:
    st.warning("No tickers loaded.")
    st.stop()

# Download
st.info(f"Downloading daily data (batch) for {len(tickers)} tickers…")
batch = fetch_batch_daily(tickers, period=period)

# Scan
res1 = []
res2 = []
progress = st.progress(0)

for i, t in enumerate(tickers, start=1):
    df = batch.get(t, pd.DataFrame())
    if df is None or df.empty:
        df = fetch_single_daily(t, period=period)

    res1.append(scan_scenario_1(t, df, use_last_completed_month))
    res2.append(scan_scenario_2(t, df, use_last_completed_month))

    progress.progress(i / max(len(tickers), 1))

df1 = pd.DataFrame(res1)
df2 = pd.DataFrame(res2)

# Filter + sort
if not df1.empty:
    df1 = df1[df1["Score"] >= float(min_score)].copy()
    df1 = df1.sort_values(["Score", "Room_%"], ascending=[False, False]).reset_index(drop=True)

if not df2.empty:
    df2 = df2[df2["Score"] >= float(min_score)].copy()
    df2 = df2.sort_values(["Score", "Room_%"], ascending=[False, False]).reset_index(drop=True)

tab1, tab2 = st.tabs(
    [
        "Scenario 1: M 2D+Green → D 2D/Inside",
        "Scenario 2: M 2U → D 2U/Inside",
    ]
)

with tab1:
    st.subheader("Scenario 1 — Ranked Results")
    st.dataframe(df1, use_container_width=True, hide_index=True)
    st.write(f"Shown: **{len(df1)}** / {len(res1)} (min score = {min_score})")
    st.caption("Weekly shows the current weekly STRAT type (1 / 2U / 2D / 3). Target is evaluated month high (prior month high).")

with tab2:
    st.subheader("Scenario 2 — Ranked Results")
    st.dataframe(df2, use_container_width=True, hide_index=True)
    st.write(f"Shown: **{len(df2)}** / {len(res2)} (min score = {min_score})")
    st.caption("Weekly shows the current weekly STRAT type (1 / 2U / 2D / 3). Target is evaluated month high (prior month high).")
