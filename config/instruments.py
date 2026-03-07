"""
AlphaDesk — Instrument Universe
eToro instrument IDs and metadata.
Update these IDs from the eToro Metadata/Instruments API endpoint.
"""

# Equity universe — US large caps (eToro instrument IDs)
# NOTE: These are placeholder IDs. Fetch actual IDs from:
# GET https://public-api.etoro.com/api/v1/metadata/instruments
US_EQUITIES = {
    "AAPL": {"etoro_id": None, "sector": "Technology"},
    "MSFT": {"etoro_id": None, "sector": "Technology"},
    "GOOGL": {"etoro_id": None, "sector": "Technology"},
    "AMZN": {"etoro_id": None, "sector": "Consumer Discretionary"},
    "NVDA": {"etoro_id": None, "sector": "Technology"},
    "META": {"etoro_id": None, "sector": "Technology"},
    "TSLA": {"etoro_id": None, "sector": "Consumer Discretionary"},
    "JPM": {"etoro_id": None, "sector": "Financials"},
    "V": {"etoro_id": None, "sector": "Financials"},
    "JNJ": {"etoro_id": None, "sector": "Healthcare"},
    "WMT": {"etoro_id": None, "sector": "Consumer Staples"},
    "PG": {"etoro_id": None, "sector": "Consumer Staples"},
    "XOM": {"etoro_id": None, "sector": "Energy"},
    "UNH": {"etoro_id": None, "sector": "Healthcare"},
    "HD": {"etoro_id": None, "sector": "Consumer Discretionary"},
    "BAC": {"etoro_id": None, "sector": "Financials"},
    "DIS": {"etoro_id": None, "sector": "Communication"},
    "KO": {"etoro_id": None, "sector": "Consumer Staples"},
    "PFE": {"etoro_id": None, "sector": "Healthcare"},
    "NFLX": {"etoro_id": None, "sector": "Communication"},
}

# European equities
EU_EQUITIES = {
    "ASML": {"etoro_id": None, "sector": "Technology", "exchange": "AMS"},
    "LVMH": {"etoro_id": None, "sector": "Consumer Discretionary", "exchange": "EPA"},
    "SAP": {"etoro_id": None, "sector": "Technology", "exchange": "ETR"},
    "NOVO-B": {"etoro_id": None, "sector": "Healthcare", "exchange": "CPH"},
    "NESN": {"etoro_id": None, "sector": "Consumer Staples", "exchange": "SWX"},
    "SHEL": {"etoro_id": None, "sector": "Energy", "exchange": "LON"},
    "AZN": {"etoro_id": None, "sector": "Healthcare", "exchange": "LON"},
    "SIEGY": {"etoro_id": None, "sector": "Industrials", "exchange": "ETR"},
    "TTE": {"etoro_id": None, "sector": "Energy", "exchange": "EPA"},
    "SAN": {"etoro_id": None, "sector": "Financials", "exchange": "BME"},
}

# Forex pairs
FX_PAIRS = {
    "EURUSD": {"etoro_id": None, "base": "EUR", "quote": "USD"},
    "GBPUSD": {"etoro_id": None, "base": "GBP", "quote": "USD"},
    "USDJPY": {"etoro_id": None, "base": "USD", "quote": "JPY"},
    "AUDUSD": {"etoro_id": None, "base": "AUD", "quote": "USD"},
    "USDCHF": {"etoro_id": None, "base": "USD", "quote": "CHF"},
    "EURGBP": {"etoro_id": None, "base": "EUR", "quote": "GBP"},
    "EURJPY": {"etoro_id": None, "base": "EUR", "quote": "JPY"},
    "GBPJPY": {"etoro_id": None, "base": "GBP", "quote": "JPY"},
    "NZDUSD": {"etoro_id": None, "base": "NZD", "quote": "USD"},
    "USDCAD": {"etoro_id": None, "base": "USD", "quote": "CAD"},
}

# All instruments for easy iteration
ALL_INSTRUMENTS = {
    "us_equities": US_EQUITIES,
    "eu_equities": EU_EQUITIES,
    "fx_pairs": FX_PAIRS,
}
