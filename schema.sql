-- AlphaDesk Database Schema

-- Segnali di trading (buy/sell generati dalle strategie)
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    ticker TEXT NOT NULL,
    strategy TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL')),
    strength REAL DEFAULT 0.0,
    metadata TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Trade eseguiti
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL')),
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    commission REAL DEFAULT 0.0,
    strategy TEXT,
    signal_id INTEGER,
    pnl REAL,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (signal_id) REFERENCES signals(id)
);

-- Snapshot giornalieri del portafoglio
CREATE TABLE IF NOT EXISTS daily_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    total_equity REAL NOT NULL,
    cash REAL NOT NULL,
    positions_value REAL NOT NULL,
    daily_pnl REAL DEFAULT 0.0,
    cumulative_pnl REAL DEFAULT 0.0,
    drawdown REAL DEFAULT 0.0,
    num_positions INTEGER DEFAULT 0,
    metadata TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Key-value store per configurazione e stato
CREATE TABLE IF NOT EXISTS kv_store (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indici
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);
CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_daily_snapshots_date ON daily_snapshots(date);
