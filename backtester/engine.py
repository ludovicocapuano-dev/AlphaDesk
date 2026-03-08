"""
AlphaDesk — Backtesting Engine
Event-driven backtester with realistic execution simulation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("alphadesk.backtest")


@dataclass
class BacktestTrade:
    """A completed backtest trade."""
    symbol: str
    strategy: str
    direction: str         # "Buy" or "Sell"
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    amount: float
    stop_loss: float
    take_profit: float
    pnl: float = 0
    pnl_pct: float = 0
    exit_reason: str = ""
    holding_days: int = 0

    def __post_init__(self):
        if self.direction == "Buy":
            self.pnl_pct = (self.exit_price - self.entry_price) / self.entry_price
        else:
            self.pnl_pct = (self.entry_price - self.exit_price) / self.entry_price
        self.pnl = self.amount * self.pnl_pct
        self.holding_days = (self.exit_date - self.entry_date).days


@dataclass
class BacktestPosition:
    """An open backtest position."""
    symbol: str
    strategy: str
    direction: str
    entry_date: datetime
    entry_price: float
    amount: float
    stop_loss: float
    take_profit: float
    instrument_id: int = 0


@dataclass
class BacktestConfig:
    """Backtesting parameters."""
    initial_capital: float = 100_000
    commission_pct: float = 0.001       # 0.1% per trade (spread cost)
    slippage_pct: float = 0.0005        # 0.05% slippage
    max_positions: int = 20
    risk_per_trade: float = 0.05        # 5% max per trade
    start_date: str = "2023-01-01"
    end_date: str = "2025-12-31"


class BacktestEngine:
    """
    Event-driven backtester.

    Simulates strategy execution on historical data with:
    - Realistic commission and slippage
    - Stop loss / take profit execution
    - Portfolio-level risk limits
    - Detailed performance analytics
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.capital = self.config.initial_capital
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.trades: List[BacktestTrade] = []
        self.open_positions: List[BacktestPosition] = []
        self.daily_returns: List[float] = []
        self._peak_equity = self.config.initial_capital

    def run(self, strategy, price_data: Dict[str, pd.DataFrame]) -> 'BacktestResult':
        """
        Run backtest for a single strategy.

        Args:
            strategy: Strategy instance (must implement generate_signals_sync)
            price_data: dict of {symbol: DataFrame with OHLCV + indicators}

        Returns:
            BacktestResult with full analytics
        """
        logger.info(f"Starting backtest: {strategy.name}")
        logger.info(f"Capital: ${self.config.initial_capital:,.0f}")
        logger.info(f"Period: {self.config.start_date} → {self.config.end_date}")

        # Get all dates from all symbols
        all_dates = set()
        for symbol, df in price_data.items():
            all_dates.update(df.index.tolist())
        all_dates = sorted(all_dates)

        # Filter date range (handle tz-aware yfinance data)
        start = pd.Timestamp(self.config.start_date)
        end = pd.Timestamp(self.config.end_date)
        # Match timezone of data if present
        if all_dates and hasattr(all_dates[0], 'tzinfo') and all_dates[0].tzinfo is not None:
            start = start.tz_localize(all_dates[0].tzinfo)
            end = end.tz_localize(all_dates[0].tzinfo)
        all_dates = [d for d in all_dates if start <= d <= end]

        prev_equity = self.capital

        for date in all_dates:
            # 1. Check stops and targets on open positions
            self._check_stops_and_targets(date, price_data)

            # 2. Run strategy signal generation for this date
            signals = self._generate_signals_for_date(strategy, date, price_data)

            # 3. Execute signals
            for signal in signals:
                self._execute_signal(signal, date, price_data)

            # 4. Mark to market
            total_equity = self._mark_to_market(date, price_data)
            self.equity_curve.append((date, total_equity))

            # Daily return
            daily_ret = (total_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            self.daily_returns.append(daily_ret)
            prev_equity = total_equity

            # Peak equity for drawdown
            if total_equity > self._peak_equity:
                self._peak_equity = total_equity

        # Close all remaining positions at end
        self._close_all_positions(all_dates[-1] if all_dates else datetime.now(), price_data)

        return BacktestResult(self)

    def _check_stops_and_targets(self, date, price_data: Dict[str, pd.DataFrame]):
        """Check stop loss and take profit on open positions."""
        closed = []

        for pos in self.open_positions:
            if pos.symbol not in price_data:
                continue

            df = price_data[pos.symbol]
            if date not in df.index:
                continue

            bar = df.loc[date]
            high = bar["high"]
            low = bar["low"]
            close = bar["close"]

            exit_price = None
            exit_reason = ""

            if pos.direction == "Buy":
                if low <= pos.stop_loss:
                    exit_price = pos.stop_loss
                    exit_reason = "Stop Loss"
                elif high >= pos.take_profit:
                    exit_price = pos.take_profit
                    exit_reason = "Take Profit"
            else:  # Sell/Short
                if high >= pos.stop_loss:
                    exit_price = pos.stop_loss
                    exit_reason = "Stop Loss"
                elif low <= pos.take_profit:
                    exit_price = pos.take_profit
                    exit_reason = "Take Profit"

            if exit_price:
                # Apply slippage
                exit_price *= (1 - self.config.slippage_pct) if pos.direction == "Buy" else (1 + self.config.slippage_pct)

                trade = BacktestTrade(
                    symbol=pos.symbol,
                    strategy=pos.strategy,
                    direction=pos.direction,
                    entry_date=pos.entry_date,
                    exit_date=date,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    amount=pos.amount,
                    stop_loss=pos.stop_loss,
                    take_profit=pos.take_profit,
                    exit_reason=exit_reason,
                )
                self.trades.append(trade)
                self.capital += trade.pnl - (pos.amount * self.config.commission_pct)
                closed.append(pos)

        for pos in closed:
            self.open_positions.remove(pos)

    def _generate_signals_for_date(self, strategy, date, price_data) -> list:
        """Generate signals using data available up to 'date' (no look-ahead bias)."""
        signals = []

        for symbol, df in price_data.items():
            if date not in df.index:
                continue

            # Only use data up to current date (prevent look-ahead)
            historical = df.loc[:date]
            if len(historical) < 60:
                continue

            # Check if strategy has a sync signal method for backtesting
            if hasattr(strategy, 'generate_signal_sync'):
                signal = strategy.generate_signal_sync(symbol, 0, historical)
                if signal:
                    signals.append(signal)
            else:
                # Fallback: use the _evaluate methods directly
                signal = self._try_evaluate(strategy, symbol, historical)
                if signal:
                    signals.append(signal)

        return signals

    def _try_evaluate(self, strategy, symbol: str, df: pd.DataFrame):
        """Try to call strategy's internal evaluation method."""
        from core.data_engine import DataEngine

        df_with_indicators = DataEngine.compute_indicators(df)

        if hasattr(strategy, '_evaluate_momentum'):
            return strategy._evaluate_momentum(symbol, 0, df_with_indicators)
        elif hasattr(strategy, '_evaluate_zscore'):
            return strategy._evaluate_zscore(symbol, 0, df_with_indicators)
        elif hasattr(strategy, '_score_pair'):
            return None  # Pairs trading needs special handling
        return None

    def _execute_signal(self, signal, date, price_data):
        """Execute a signal if risk limits allow."""
        # Check position limits
        if len(self.open_positions) >= self.config.max_positions:
            return

        # Check if already in this symbol
        if any(p.symbol == signal.symbol for p in self.open_positions):
            return

        # Position sizing
        risk_amount = self.capital * min(signal.suggested_size_pct, self.config.risk_per_trade)
        if risk_amount < 100:  # Minimum trade size
            return

        # Apply slippage and commission
        entry_price = signal.entry_price
        if signal.direction == "Buy":
            entry_price *= (1 + self.config.slippage_pct)
        else:
            entry_price *= (1 - self.config.slippage_pct)

        commission = risk_amount * self.config.commission_pct
        self.capital -= commission

        pos = BacktestPosition(
            symbol=signal.symbol,
            strategy=signal.strategy_name,
            direction=signal.direction,
            entry_date=date,
            entry_price=entry_price,
            amount=risk_amount,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )
        self.open_positions.append(pos)

    def _mark_to_market(self, date, price_data) -> float:
        """Calculate total equity (cash + unrealized P&L)."""
        unrealized = 0
        for pos in self.open_positions:
            if pos.symbol in price_data and date in price_data[pos.symbol].index:
                current_price = price_data[pos.symbol].loc[date, "close"]
                if pos.direction == "Buy":
                    unrealized += pos.amount * ((current_price - pos.entry_price) / pos.entry_price)
                else:
                    unrealized += pos.amount * ((pos.entry_price - current_price) / pos.entry_price)

        return self.capital + unrealized

    def _close_all_positions(self, date, price_data):
        """Close all positions at end of backtest."""
        for pos in list(self.open_positions):
            if pos.symbol in price_data and date in price_data[pos.symbol].index:
                close_price = price_data[pos.symbol].loc[date, "close"]
            else:
                close_price = pos.entry_price  # Flat if no data

            trade = BacktestTrade(
                symbol=pos.symbol,
                strategy=pos.strategy,
                direction=pos.direction,
                entry_date=pos.entry_date,
                exit_date=date,
                entry_price=pos.entry_price,
                exit_price=close_price,
                amount=pos.amount,
                stop_loss=pos.stop_loss,
                take_profit=pos.take_profit,
                exit_reason="End of Backtest",
            )
            self.trades.append(trade)
            self.capital += trade.pnl

        self.open_positions.clear()


class BacktestResult:
    """Comprehensive backtest analytics."""

    def __init__(self, engine: BacktestEngine):
        self.initial_capital = engine.config.initial_capital
        self.final_capital = engine.capital
        self.equity_curve = engine.equity_curve
        self.trades = engine.trades
        self.daily_returns = np.array(engine.daily_returns)
        self._compute_metrics()

    def _compute_metrics(self):
        """Compute all performance metrics."""
        # Basic P&L
        self.total_return = (self.final_capital - self.initial_capital) / self.initial_capital
        self.total_pnl = self.final_capital - self.initial_capital

        # Trade statistics
        self.num_trades = len(self.trades)
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]
        self.num_winners = len(winning)
        self.num_losers = len(losing)
        self.win_rate = self.num_winners / self.num_trades if self.num_trades > 0 else 0

        self.avg_win = np.mean([t.pnl_pct for t in winning]) if winning else 0
        self.avg_loss = np.mean([t.pnl_pct for t in losing]) if losing else 0
        self.avg_win_dollars = np.mean([t.pnl for t in winning]) if winning else 0
        self.avg_loss_dollars = np.mean([t.pnl for t in losing]) if losing else 0

        self.largest_win = max([t.pnl for t in self.trades], default=0)
        self.largest_loss = min([t.pnl for t in self.trades], default=0)

        self.profit_factor = (
            abs(sum(t.pnl for t in winning)) / abs(sum(t.pnl for t in losing))
            if losing and sum(t.pnl for t in losing) != 0 else float('inf')
        )

        self.avg_holding_days = np.mean([t.holding_days for t in self.trades]) if self.trades else 0

        # Risk metrics
        if len(self.daily_returns) > 1:
            self.sharpe_ratio = (
                np.mean(self.daily_returns) / np.std(self.daily_returns) * np.sqrt(252)
                if np.std(self.daily_returns) > 0 else 0
            )
            self.sortino_ratio = self._compute_sortino()
            self.annual_volatility = np.std(self.daily_returns) * np.sqrt(252)
            self.calmar_ratio = (
                (self.total_return / abs(self.max_drawdown))
                if hasattr(self, 'max_drawdown') and self.max_drawdown != 0 else 0
            )
        else:
            self.sharpe_ratio = 0
            self.sortino_ratio = 0
            self.annual_volatility = 0
            self.calmar_ratio = 0

        # Drawdown analysis
        self._compute_drawdowns()

        # By exit reason
        self.exit_reasons = {}
        for t in self.trades:
            reason = t.exit_reason
            if reason not in self.exit_reasons:
                self.exit_reasons[reason] = {"count": 0, "pnl": 0}
            self.exit_reasons[reason]["count"] += 1
            self.exit_reasons[reason]["pnl"] += t.pnl

    def _compute_sortino(self) -> float:
        """Sortino ratio (penalizes only downside volatility)."""
        downside = self.daily_returns[self.daily_returns < 0]
        if len(downside) == 0:
            return float('inf')
        downside_std = np.std(downside)
        if downside_std == 0:
            return 0
        return np.mean(self.daily_returns) / downside_std * np.sqrt(252)

    def _compute_drawdowns(self):
        """Compute max drawdown and drawdown duration."""
        if not self.equity_curve:
            self.max_drawdown = 0
            self.max_drawdown_duration = 0
            return

        equities = [e[1] for e in self.equity_curve]
        peak = equities[0]
        max_dd = 0
        dd_start = 0
        max_dd_duration = 0
        current_dd_start = 0

        for i, eq in enumerate(equities):
            if eq > peak:
                peak = eq
                duration = i - current_dd_start
                if duration > max_dd_duration:
                    max_dd_duration = duration
                current_dd_start = i

            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        self.max_drawdown = max_dd
        self.max_drawdown_duration = max_dd_duration

    def summary(self) -> str:
        """Generate formatted summary report."""
        years = len(self.daily_returns) / 252 if len(self.daily_returns) > 0 else 1
        annual_return = (1 + self.total_return) ** (1 / years) - 1 if years > 0 else 0

        report = f"""
╔══════════════════════════════════════════════════════════╗
║              BACKTEST RESULTS                            ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  RETURNS                                                 ║
║  ────────────────────────────────                        ║
║  Total Return:       {self.total_return:>+8.2%}                         ║
║  Annual Return:      {annual_return:>+8.2%}                         ║
║  Total P&L:          ${self.total_pnl:>+12,.2f}                    ║
║  Final Capital:      ${self.final_capital:>12,.2f}                    ║
║                                                          ║
║  RISK METRICS                                            ║
║  ────────────────────────────────                        ║
║  Sharpe Ratio:       {self.sharpe_ratio:>8.2f}                         ║
║  Sortino Ratio:      {self.sortino_ratio:>8.2f}                         ║
║  Max Drawdown:       {self.max_drawdown:>8.2%}                         ║
║  Annual Volatility:  {self.annual_volatility:>8.2%}                         ║
║  Calmar Ratio:       {self.calmar_ratio:>8.2f}                         ║
║                                                          ║
║  TRADE STATISTICS                                        ║
║  ────────────────────────────────                        ║
║  Total Trades:       {self.num_trades:>8d}                         ║
║  Win Rate:           {self.win_rate:>8.1%}                         ║
║  Winners/Losers:     {self.num_winners:>4d} / {self.num_losers:<4d}                        ║
║  Avg Win:            {self.avg_win:>+8.2%}  (${self.avg_win_dollars:>+8,.0f})         ║
║  Avg Loss:           {self.avg_loss:>+8.2%}  (${self.avg_loss_dollars:>+8,.0f})         ║
║  Largest Win:        ${self.largest_win:>+12,.2f}                    ║
║  Largest Loss:       ${self.largest_loss:>+12,.2f}                    ║
║  Profit Factor:      {self.profit_factor:>8.2f}                         ║
║  Avg Holding Days:   {self.avg_holding_days:>8.1f}                         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
        return report

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame for analysis."""
        return pd.DataFrame([
            {
                "symbol": t.symbol,
                "strategy": t.strategy,
                "direction": t.direction,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "amount": t.amount,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "holding_days": t.holding_days,
                "exit_reason": t.exit_reason,
            }
            for t in self.trades
        ])
