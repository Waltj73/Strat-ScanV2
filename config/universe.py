# data/universe.py

MARKET_ETFS = {
    "S&P 500": "SPY",
    "Nasdaq 100": "QQQ",
    "Russell 2000": "IWM",
    "Dow Jones": "DIA",
}

SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Staples": "XLP",
    "Discretionary": "XLY",
    "Comm Services": "XLC",
    "Real Estate": "XLRE",
}

# Keep this list small/curated for speed. Add more later.
SECTOR_TICKERS = {
    "Technology": ["AAPL","MSFT","NVDA","AMD","AVGO","ORCL","ADBE","QCOM","CSCO","INTC"],
    "Financials": ["JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP","SPGI"],
    "Health Care": ["UNH","JNJ","LLY","PFE","MRK","ABBV","TMO","ABT","DHR","ISRG"],
    "Energy": ["XOM","CVX","COP","EOG","SLB","HAL","MPC","VLO","PSX","OXY"],
    "Industrials": ["CAT","DE","HON","GE","LMT","RTX","BA","UNP","UPS","ETN"],
    "Materials": ["LIN","APD","SHW","NUE","DOW","ECL","FCX","NEM","VMC","ALB"],
    "Utilities": ["NEE","DUK","SO","D","AEP","EXC","XEL","SRE","ED","PEG"],
    "Staples": ["PG","KO","PEP","WMT","COST","PM","MO","MDLZ","CL","KMB"],
    "Discretionary": ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","BKNG","TJX","CMG"],
    "Comm Services": ["GOOGL","GOOG","META","NFLX","TMUS","DIS","CMCSA","CHTR","SPOT","ROKU"],
    "Real Estate": ["PLD","AMT","EQIX","PSA","O","WELL","DLR","SPG","CCI","VICI"],
}
