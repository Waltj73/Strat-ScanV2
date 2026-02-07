# config/universe.py

# Major market ETFs
MARKET_ETFS = {
    "S&P 500": "SPY",
    "Nasdaq 100": "QQQ",
    "Russell 2000": "IWM",
    "Dow Jones": "DIA",
}

# Metals ETFs
METALS_ETFS = {
    "Metals - Gold": "GLD",
    "Metals - Silver": "SLV",
    "Metals - Copper": "CPER",
    "Metals - Platinum": "PPLT",
    "Metals - Palladium": "PALL",
}

# Sector ETFs
SECTOR_ETFS = {
    "Energy": "XLE",
    "Comm Services": "XLC",
    "Staples": "XLP",
    "Materials": "XLB",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Discretionary": "XLY",
    "Utilities": "XLU",
    "Financials": "XLF",
    "Technology": "XLK",
    "Health Care": "XLV",
    **METALS_ETFS,
}

# Ticker universe per sector
SECTOR_TICKERS = {
    "Technology": [
        "AAPL","MSFT","NVDA","AVGO","CRM","ORCL",
        "ADBE","AMD","CSCO","INTC","QCOM","TXN","NOW"
    ],
    "Energy": [
        "XOM","CVX","COP","EOG","SLB","PSX","MPC"
    ],
    "Financials": [
        "JPM","BAC","WFC","GS","MS","BLK"
    ],
    "Health Care": [
        "UNH","JNJ","LLY","ABBV","MRK","PFE"
    ],
    "Industrials": [
        "CAT","DE","HON","RTX","GE","UNP"
    ],
    "Discretionary": [
        "AMZN","TSLA","HD","MCD","NKE"
    ],
    "Staples": [
        "PG","KO","PEP","WMT","COST"
    ],
    "Utilities": [
        "NEE","DUK","SO","AEP"
    ],
    "Materials": [
        "LIN","SHW","FCX","NEM"
    ],
    "Real Estate": [
        "PLD","AMT","EQIX"
    ],
    "Comm Services": [
        "GOOGL","META","NFLX","DIS"
    ],

    # Metals groups
    "Metals - Gold": ["GLD"],
    "Metals - Silver": ["SLV"],
    "Metals - Copper": ["CPER"],
    "Metals - Platinum": ["PPLT"],
    "Metals - Palladium": ["PALL"],
}
