"""
AlphaDesk — Instrument Universe
eToro instrument IDs (immutable, never change even if ticker changes).
Resolved via GET /api/v1/market-data/search?internalSymbolFull=SYMBOL
"""

# Equity universe — US large caps
US_EQUITIES = {
    # ── Technology ──
    "AAPL": {"etoro_id": 1001, "sector": "Technology"},
    "MSFT": {"etoro_id": 1004, "sector": "Technology"},
    "GOOGL": {"etoro_id": 6434, "sector": "Technology"},
    "AMZN": {"etoro_id": 1005, "sector": "Consumer Discretionary"},
    "NVDA": {"etoro_id": 1137, "sector": "Technology"},
    "META": {"etoro_id": 1003, "sector": "Technology"},
    "NFLX": {"etoro_id": 1127, "sector": "Communication"},

    # ── Energy ──
    "XOM": {"etoro_id": 1036, "sector": "Energy"},

    # ── Consumer Discretionary ──
    "HD": {"etoro_id": 1018, "sector": "Consumer Discretionary"},
    "TSLA": {"etoro_id": 1111, "sector": "Consumer Discretionary"},
    "DIS": {"etoro_id": 1016, "sector": "Communication"},

    # ── Financials ──
    "JPM": {"etoro_id": 1023, "sector": "Financials"},
    "BAC": {"etoro_id": 1011, "sector": "Financials"},
    "V": {"etoro_id": 308, "sector": "Financials"},

    # ── Healthcare ──
    "JNJ": {"etoro_id": 1022, "sector": "Healthcare"},
    "UNH": {"etoro_id": 1032, "sector": "Healthcare"},
    "PFE": {"etoro_id": 1028, "sector": "Healthcare"},

    # ── Consumer Staples ──
    "WMT": {"etoro_id": 1035, "sector": "Consumer Staples"},
    "PG": {"etoro_id": 1029, "sector": "Consumer Staples"},
    "KO": {"etoro_id": 1024, "sector": "Consumer Staples"},
}

# ETFs currently in portfolio
ETFS = {
    "XLE": {"etoro_id": 3008, "sector": "Energy"},
    "OIH": {"etoro_id": 3206, "sector": "Energy"},
    "BE": {"etoro_id": 6614, "sector": "Energy"},
}

# Forex pairs
FX_PAIRS = {
    "EURUSD": {"etoro_id": 1, "base": "EUR", "quote": "USD"},
    "GBPUSD": {"etoro_id": 2, "base": "GBP", "quote": "USD"},
    "USDJPY": {"etoro_id": 5, "base": "USD", "quote": "JPY"},
    "AUDUSD": {"etoro_id": 7, "base": "AUD", "quote": "USD"},
    "USDCHF": {"etoro_id": 6, "base": "USD", "quote": "CHF"},
    "EURGBP": {"etoro_id": 8, "base": "EUR", "quote": "GBP"},
}

# All instruments flat mapping: symbol -> etoro_id
ALL_IDS = {}
for _d in (US_EQUITIES, ETFS, FX_PAIRS):
    for _sym, _meta in _d.items():
        ALL_IDS[_sym] = _meta["etoro_id"]


def get_instrument_id(symbol: str) -> int:
    """Get eToro instrument ID for a symbol."""
    clean = symbol.replace("=X", "").replace("/", "")
    return ALL_IDS.get(clean, ALL_IDS.get(symbol))


def get_symbol(instrument_id: int) -> str:
    """Reverse lookup: instrument ID to symbol."""
    for sym, iid in ALL_IDS.items():
        if iid == instrument_id:
            return sym
    return f"ID:{instrument_id}"
