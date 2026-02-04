# app.py (v1.1)
# STRAT Scanner Dashboard (Streamlit) — Ranking Engine (0–100)
#
# Setup logic:
#   Monthly context (evaluated month):
#     2-Down: low < prior low AND high <= prior high
#     Green: close > open
#   Daily trigger (latest day):
#     Failed 2 Down: low < prior low AND high > prior high
#     Inside Day: high < prior high AND low > prior low
#   Target: evaluated month high
#
# Probabilities (2-year empirical):
#   After a 2-down green month, probability price hits that month’s high in the FOLLOWING month.
#
# Ranking Score (0–100):
#   Monthly (0–40): 2Down +20, Green +20
#   Daily (0–30): Failed2 +30, Inside +20
#   Room (0–15): 1 point per 1% room (cap 15)
#   Probability (0–15): hit_rate% mapped to 0–15
#
# Universe:
#   Top 50 mega caps (S&P500 + Nasdaq-100) by market cap, with fallback list.
#
# Requirements (requirements.txt):
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

st.set_page_config(page_title="2D Scan (STRAT) v1.1", layout="wide")

# -----------------------------
# Styling (green checkmarks)
# -----------------------------
CHECK = "✅"
XMARK = "—"

st.markdown(
    """
    <style>
      /* make the dataframe area breathe */
      .stDataFrame { padding-top: 0.25rem; }

      /* nicer headings spacing */
      h1, h2, h3 { margin-bottom: 0.25rem; }

      /* green check emoji fallback (emoji already green, but keep consistent) */
      .green-check { color: #16a34a; font-weight: 700; }

      /* small muted caption */
      .muted { color: rgba(120,120,120,0.9); font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

def ck(v: bool) -> str:
    return CHECK if bool(v) else XMARK

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
    tables = pd.read_html(StringIO(html))
    df = tables[0]
    tickers = df["Symbol"].astype(str).str.strip().tolist()
    tickers = [t.replace(".", "-") for t in tickers]
    return sorted(set(tickers))

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
    tickers = [t.replace(".", "-") for t in tickers]
    return sorted(set(tickers))

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
    df = df.sort_values("MarketCap", ascending=False).head(50).reset_index(drop=True)
    return df

# -----------------------------
# Data fetch
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
# STRAT logic
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

def strat_month_flags(monthly: pd.DataFrame, use_last_completed_month: bool = True) -> dict:
    if monthly is None or monthly.empty or len(monthly) < 3:
        return {"ok": False}

    now = pd.Timestamp(datetime.now(timezone.utc).date())
    last_idx = monthly.index[-1]
    in_progress = (last_idx.year == now.year) and (last_idx.month == now.month)

    eval_pos = (-2 if in_progress else -1) if use_last_completed_month else -1
    if len(monthly) < (abs(eval_pos) + 2):
        return {"ok": False}

    m = monthly.iloc[eval_pos]
    m_prev = monthly.iloc[eval_pos - 1]

    month_2down = (m["Low"] < m_prev["Low"]) and (m["High"] <= m_prev["High"])
    month_green = (m["Close"] > m["Open"])
    target = float(m["High"])

    return {
        "ok": True,
        "eval_month_start": monthly.index[eval_pos],
        "month_2down": bool(month_2down),
        "month_green": bool(month_green),
        "target_prior_month_high": target,
    }

def strat_daily_flags(daily: pd.DataFrame) -> dict:
    if daily is None or daily.empty or len(daily) < 3:
        return {"ok": False}

    d = daily.iloc[-1]
    d_prev = daily.iloc[-2]

    failed2_down = (d["Low"] < d_prev["Low"]) and (d["High"] > d_prev["High"])
    inside_day = (d["High"] < d_prev["High"]) and (d["Low"] > d_prev["Low"])
    failed2_down_green = bool(failed2_down and (d["Close"] > d["Open"]))

    return {
        "ok": True,
        "bar_date": daily.index[-1],
        "last_close": float(d["Close"]),
        "failed2_down": bool(failed2_down),
        "failed2_down_green": bool(failed2_down_green),
        "inside_day": bool(inside_day),
    }

def monthly_hit_probability(daily: pd.DataFrame, years: int = 2) -> dict:
    """
    Empirical probability:
    After a 2-down green month, does price hit THAT month’s high during the following month?
    """
    if daily is None or daily.empty:
        return {"setups": 0, "hit_rate": 0.0, "avg_days": None}

    cutoff = daily.index.max() - pd.DateOffset(years=years)
    d = daily[daily.index >= cutoff].copy()
    if d.empty or len(d) < 60:
        return {"setups": 0, "hit_rate": 0.0, "avg_days": None}

    monthly = to_monthly_ohlc(d)
    if len(monthly) < 3:
        return {"setups": 0, "hit_rate": 0.0, "avg_days": None}

    setups = 0
    hits = 0
    days_to_hit = []

    for i in range(1, len(monthly) - 1):
        m_prev = monthly.iloc[i - 1]
        m = monthly.iloc[i]

        month_2down = (m["Low"] < m_prev["Low"]) and (m["High"] <= m_prev["High"])
        month_green = (m["Close"] > m["Open"])
        if not (month_2down and month_green):
            continue

        setups += 1
        target = float(m["High"])

        next_month_start = monthly.index[i + 1]
        next_month_end = next_month_start + pd.offsets.MonthEnd(1)

        nm = d[(d.index >= next_month_start) & (d.index <= next_month_end)]
        if nm.empty:
            continue

        hit_rows = nm[nm["High"] >= target]
        if not hit_rows.empty:
            hits += 1
            first_hit = hit_rows.index[0]
            start = nm.index[0]
            days_to_hit.append((first_hit - start).days)

    hit_rate = (hits / setups) if setups > 0 else 0.0
    avg_days = float(np.mean(days_to_hit)) if days_to_hit else None

    return {"setups": setups, "hit_rate": hit_rate, "avg_days": avg_days}

def score_setup(month2down: bool, monthgreen: bool, failed2: bool, inside: bool, room_pct: float, hit_rate_pct: float) -> float:
    score = 0.0

    # Monthly (0–40)
    score += 20.0 if month2down else 0.0
    score += 20.0 if monthgreen else 0.0

    # Daily (0–30)
    if failed2:
        score += 30.0
    elif inside:
        score += 20.0

    # Room (0–15) (cap at 15%)
    score += max(0.0, min(15.0, room_pct))

    # Probability (0–15)
    score += max(0.0, min(15.0, (hit_rate_pct / 100.0) * 15.0))

    return round(score, 1)

def tier_from_score(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    return "D"

def scan_one(
    ticker: str,
    daily: pd.DataFrame,
    use_last_completed_month: bool,
    require_failed2_green: bool,
) -> dict:
    if daily is None or daily.empty or len(daily) < 60:
        return {"Ticker": ticker, "Score": 0.0, "Tier": "D", "Status": "No/insufficient data"}

    monthly = to_monthly_ohlc(daily)
    m = strat_month_flags(monthly, use_last_completed_month=use_last_completed_month)
    d = strat_daily_flags(daily)

    if not m.get("ok") or not d.get("ok"):
        return {"Ticker": ticker, "Score": 0.0, "Tier": "D", "Status": "Insufficient candles"}

    # Monthly flags
    month2down = m["month_2down"]
    monthgreen = m["month_green"]
    eval_month = m["eval_month_start"].strftime("%Y-%m")
    target = m["target_prior_month_high"]

    # Daily flags
    failed2 = d["failed2_down_green"] if require_failed2_green else d["failed2_down"]
    inside = d["inside_day"]
    last_bar = pd.to_datetime(d["bar_date"]).strftime("%Y-%m-%d")
    last_close = d["last_close"]

    trigger = "FAILED2D" if failed2 else ("INSIDE" if inside else "—")

    # Room
    room_pct = (target - last_close) / last_close * 100.0

    # Probability (2y)
    prob = monthly_hit_probability(daily, years=2)
    hit_rate_pct = prob["hit_rate"] * 100.0

    # Score + tier
    score = score_setup(
        month2down=month2down,
        monthgreen=monthgreen,
        failed2=failed2,
        inside=inside,
        room_pct=room_pct,
        hit_rate_pct=hit_rate_pct,
    )
    tier = tier_from_score(score)

    return {
        "Ticker": ticker,
        "Score": score,
        "Tier": tier,
        "Trigger": trigger,
        "Monthly_2Down": ck(month2down),
        "Monthly_Green": ck(monthgreen),
        "Eval_Month": eval_month,
        "Last_Bar": last_bar,
        "Last_Close": round(last_close, 2),
        "Target_PriorMonthHigh": round(target, 2),
        "Room_To_Target_%": round(room_pct, 2),
        "Setup_Count_2Y": int(prob["setups"]),
        "Hit_Rate_%_2Y": round(hit_rate_pct, 1),
        "Avg_Days_To_Target_2Y": (round(prob["avg_days"], 1) if prob["avg_days"] is not None else None),
        "Status": "OK",
    }

# -----------------------------
# UI
# -----------------------------
st.title("2D Scan (STRAT) — v1.1 (Ranked)")
st.markdown(
    '<div class="muted">Monthly: <b>2-Down + Green</b> • Daily: <b>Failed2D or Inside</b> • Target: <b>prior month high</b> • Probability: <b>2-year hit rate</b> • Output: <b>Score 0–100</b></div>',
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

    st.header("Signal Rules")
    use_last_completed_month = st.checkbox("Use last COMPLETED month (stable)", value=True)
    require_failed2_green = st.checkbox("Require Failed 2 Down to close green", value=False)

    st.header("Ranking Filters")
    min_score = st.slider("Minimum Score", 0, 100, 60)

    run = st.button("Run Scan", type="primary")

# Build tickers
tickers: list[str] = []
universe_note = None

try:
    if universe == "Fallback Top 50 (manual)":
        tickers = DEFAULT_TOP50
        universe_note = "Using manual fallback Top 50."

    elif universe.startswith("Top 50"):
        sp = get_sp500_tickers()
        ndx = get_nasdaq100_tickers()
        combined = sorted(set(sp + ndx))
        top50_df = get_top50_by_marketcap(combined)

        if top50_df is None or top50_df.empty:
            tickers = DEFAULT_TOP50
            universe_note = "Top 50 market-cap fetch returned empty; using fallback Top 50."
        else:
            tickers = top50_df["Ticker"].tolist()
            universe_note = "Using Top 50 mega caps (by market cap)."

    else:
        tickers_text = st.text_area(
            "Tickers (comma or newline separated)",
            value="AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, AVGO, AMD, PLTR",
            height=120,
        )
        tickers = [t.strip().upper() for t in tickers_text.replace("\n", ",").split(",") if t.strip()]
        universe_note = f"Using custom list ({len(tickers)} tickers)."

except Exception as e:
    tickers = DEFAULT_TOP50
    universe_note = f"Universe fetch failed ({e}). Using fallback Top 50."

if universe_note:
    st.info(universe_note)

if not run:
    st.write("Click **Run Scan** to generate ranked results.")
    st.stop()

if not tickers:
    st.warning("No tickers loaded.")
    st.stop()

# Download daily data (batch)
st.info(f"Downloading daily data (batch) for {len(tickers)} tickers…")
batch = fetch_batch_daily(tickers, period=period)

# Scan loop
results = []
progress = st.progress(0)

for i, t in enumerate(tickers, start=1):
    df = batch.get(t, pd.DataFrame())
    if df is None or df.empty:
        df = fetch_single_daily(t, period=period)

    results.append(
        scan_one(
            ticker=t,
            daily=df,
            use_last_completed_month=use_last_completed_month,
            require_failed2_green=require_failed2_green,
        )
    )
    progress.progress(i / max(len(tickers), 1))

res = pd.DataFrame(results)

# Filter + sort
if not res.empty:
    res = res[res["Score"] >= float(min_score)].copy()
    res = res.sort_values(["Score", "Room_To_Target_%"], ascending=[False, False]).reset_index(drop=True)

st.subheader("Ranked Results")
st.dataframe(res, use_container_width=True, hide_index=True)

st.write(f"Shown: **{len(res)}** / {len(results)} (min score = {min_score})")

st.caption(
    "✅ = condition met. Score is a weighted ranking (Monthly + Daily trigger + Room + 2Y hit rate). "
    "This is an opportunity list, not a binary match filter."
)
