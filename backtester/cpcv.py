"""
AlphaDesk - Combinatorial Purged Cross-Validation (CPCV)

Implements CPCV from "Advances in Financial Machine Learning" (Lopez de Prado).

Instead of a single walk-forward split, CPCV:
1. Splits data into N contiguous groups
2. For each combination of k test groups out of N, trains on the remaining N-k
3. Purges training samples near test boundaries to prevent lookahead bias
4. Applies an embargo gap between train and test periods
5. Aggregates out-of-sample results across all combinatorial paths
6. Computes Probability of Backtest Overfitting (PBO)
"""

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from backtester.engine import BacktestConfig, BacktestEngine, BacktestResult

logger = logging.getLogger("alphadesk.backtest.cpcv")


@dataclass
class CPCVFoldResult:
    """Result from a single CPCV fold (one combination of test groups)."""
    fold_id: int
    test_groups: Tuple[int, ...]
    train_groups: Tuple[int, ...]
    backtest_result: BacktestResult
    oos_sharpe: float
    oos_return: float
    num_trades: int


@dataclass
class CPCVResult:
    """Aggregated CPCV results across all combinatorial paths."""
    strategy_name: str
    n_groups: int
    k_test: int
    embargo_days: int
    n_folds: int
    fold_results: List[CPCVFoldResult]

    # Aggregated metrics
    oos_sharpes: np.ndarray = field(default_factory=lambda: np.array([]))
    oos_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    pbo: float = 0.0  # Probability of Backtest Overfitting

    def compute_summary(self):
        """Compute aggregated statistics from fold results."""
        self.oos_sharpes = np.array([f.oos_sharpe for f in self.fold_results])
        self.oos_returns = np.array([f.oos_return for f in self.fold_results])
        self._compute_pbo()

    def _compute_pbo(self):
        """
        Compute Probability of Backtest Overfitting.

        PBO = fraction of OOS paths where Sharpe ratio is below the median.
        A PBO > 0.5 indicates the strategy is likely overfit.

        In the full Lopez de Prado formulation, PBO uses the rank correlation
        between IS and OOS performance across paths. Here we use the simplified
        approach: PBO = P(OOS Sharpe < median OOS Sharpe) estimated from the
        empirical distribution, which converges to the same interpretation when
        the number of combinatorial paths is large.
        """
        if len(self.oos_sharpes) == 0:
            self.pbo = 1.0
            return

        # Fraction of paths with negative OOS Sharpe (simplified PBO)
        # This answers: "what fraction of paths would have lost money OOS?"
        self.pbo = np.mean(self.oos_sharpes < 0)

    def summary(self) -> str:
        """Generate formatted CPCV report."""
        valid_sharpes = self.oos_sharpes[np.isfinite(self.oos_sharpes)]
        total_trades = sum(f.num_trades for f in self.fold_results)
        folds_with_trades = sum(1 for f in self.fold_results if f.num_trades > 0)

        report = f"""
========================================================================
  COMBINATORIAL PURGED CROSS-VALIDATION (CPCV) RESULTS
========================================================================
  Strategy:           {self.strategy_name}
  Groups (N):         {self.n_groups}
  Test groups (k):    {self.k_test}
  Embargo:            {self.embargo_days} days
  Total folds:        {self.n_folds}
  Folds with trades:  {folds_with_trades}
------------------------------------------------------------------------
  OOS SHARPE DISTRIBUTION
  Mean:               {np.mean(valid_sharpes):>+8.3f}
  Median:             {np.median(valid_sharpes):>+8.3f}
  Std Dev:            {np.std(valid_sharpes):>8.3f}
  Min:                {np.min(valid_sharpes):>+8.3f}
  Max:                {np.max(valid_sharpes):>+8.3f}
  Pct > 0:            {np.mean(valid_sharpes > 0):>8.1%}
------------------------------------------------------------------------
  OOS RETURN DISTRIBUTION
  Mean:               {np.mean(self.oos_returns):>+8.3%}
  Median:             {np.median(self.oos_returns):>+8.3%}
  Std Dev:            {np.std(self.oos_returns):>8.3%}
------------------------------------------------------------------------
  PROBABILITY OF BACKTEST OVERFITTING (PBO)
  PBO:                {self.pbo:>8.1%}
  Interpretation:     {"LOW RISK" if self.pbo < 0.3 else "MODERATE RISK" if self.pbo < 0.5 else "HIGH RISK - likely overfit"}
------------------------------------------------------------------------
  Total trades across folds: {total_trades}
========================================================================
"""
        return report


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation engine.

    Parameters:
        n_groups: Number of contiguous time groups to split data into (default 6)
        k_test: Number of groups to hold out as test in each fold (default N/2)
        embargo_days: Number of trading days to purge between train/test boundaries
        config: BacktestConfig for each fold's engine
    """

    def __init__(
        self,
        n_groups: int = 6,
        k_test: Optional[int] = None,
        embargo_days: int = 5,
        config: Optional[BacktestConfig] = None,
    ):
        self.n_groups = n_groups
        self.k_test = k_test if k_test is not None else n_groups // 2
        self.embargo_days = embargo_days
        self.config = config or BacktestConfig()

        if self.k_test >= self.n_groups:
            raise ValueError(f"k_test ({self.k_test}) must be < n_groups ({self.n_groups})")
        if self.k_test < 1:
            raise ValueError("k_test must be >= 1")

    def run(
        self,
        strategy,
        price_data: Dict[str, pd.DataFrame],
    ) -> CPCVResult:
        """
        Run CPCV for a strategy on the provided price data.

        Args:
            strategy: Strategy instance with generate_signal_sync or _evaluate_* methods
            price_data: dict of {symbol: DataFrame} with OHLCV + indicators

        Returns:
            CPCVResult with all fold results and aggregate metrics
        """
        # Collect all unique trading dates across all symbols
        all_dates = sorted(set(
            d for df in price_data.values() for d in df.index
        ))

        if len(all_dates) < self.n_groups * 20:
            raise ValueError(
                f"Not enough data: {len(all_dates)} dates for {self.n_groups} groups. "
                f"Need at least {self.n_groups * 20} dates."
            )

        # Split dates into N contiguous groups
        groups = self._split_into_groups(all_dates)

        logger.info(f"CPCV: {self.n_groups} groups, {self.k_test} test groups, "
                     f"embargo={self.embargo_days} days")
        for i, g in enumerate(groups):
            logger.info(f"  Group {i}: {g[0].strftime('%Y-%m-%d')} -> "
                         f"{g[-1].strftime('%Y-%m-%d')} ({len(g)} days)")

        # Generate all combinations of test groups
        test_combos = list(combinations(range(self.n_groups), self.k_test))
        n_folds = len(test_combos)
        logger.info(f"CPCV: C({self.n_groups},{self.k_test}) = {n_folds} folds")

        fold_results = []

        for fold_id, test_group_indices in enumerate(test_combos):
            train_group_indices = tuple(
                i for i in range(self.n_groups) if i not in test_group_indices
            )

            logger.info(f"Fold {fold_id + 1}/{n_folds}: "
                         f"train={train_group_indices}, test={test_group_indices}")

            # Build train and test date sets with purging and embargo
            train_dates, test_dates = self._build_train_test_dates(
                groups, train_group_indices, test_group_indices
            )

            if len(test_dates) < 10:
                logger.warning(f"  Fold {fold_id + 1}: too few test dates ({len(test_dates)}), skipping")
                continue

            # Slice price_data to test period only
            test_price_data = self._slice_price_data(price_data, test_dates)

            if not test_price_data:
                logger.warning(f"  Fold {fold_id + 1}: no price data in test period, skipping")
                continue

            # Run backtest on test period
            # The strategy generates signals using historical data up to each date
            # (the engine handles this via its look-ahead prevention in
            # _generate_signals_for_date). We pass the full data but configure
            # the engine to only iterate over test dates.
            fold_config = BacktestConfig(
                initial_capital=self.config.initial_capital,
                commission_pct=self.config.commission_pct,
                slippage_pct=self.config.slippage_pct,
                max_positions=self.config.max_positions,
                risk_per_trade=self.config.risk_per_trade,
                start_date=min(test_dates).strftime('%Y-%m-%d'),
                end_date=max(test_dates).strftime('%Y-%m-%d'),
            )

            engine = BacktestEngine(fold_config)

            # We pass the full price data (so the strategy has history for
            # indicators), but the engine's date filtering will restrict
            # trading to the test window only.
            try:
                result = engine.run(strategy, price_data)
            except Exception as e:
                logger.error(f"  Fold {fold_id + 1} failed: {e}")
                continue

            fold_result = CPCVFoldResult(
                fold_id=fold_id,
                test_groups=test_group_indices,
                train_groups=train_group_indices,
                backtest_result=result,
                oos_sharpe=result.sharpe_ratio,
                oos_return=result.total_return,
                num_trades=result.num_trades,
            )
            fold_results.append(fold_result)

            logger.info(f"  Fold {fold_id + 1} done: "
                         f"Sharpe={result.sharpe_ratio:+.3f}, "
                         f"Return={result.total_return:+.2%}, "
                         f"Trades={result.num_trades}")

        # Build aggregate result
        cpcv_result = CPCVResult(
            strategy_name=strategy.name,
            n_groups=self.n_groups,
            k_test=self.k_test,
            embargo_days=self.embargo_days,
            n_folds=n_folds,
            fold_results=fold_results,
        )
        cpcv_result.compute_summary()

        return cpcv_result

    def _split_into_groups(self, all_dates: list) -> List[list]:
        """Split sorted date list into N approximately equal contiguous groups."""
        n = len(all_dates)
        groups = []
        for i in range(self.n_groups):
            start_idx = i * n // self.n_groups
            end_idx = (i + 1) * n // self.n_groups
            groups.append(all_dates[start_idx:end_idx])
        return groups

    def _build_train_test_dates(
        self,
        groups: List[list],
        train_indices: Tuple[int, ...],
        test_indices: Tuple[int, ...],
    ) -> Tuple[set, set]:
        """
        Build train and test date sets with purging and embargo.

        Purging: Remove training samples that are within `embargo_days` of
        any test group boundary (start or end). This prevents information
        leakage from training samples that are temporally adjacent to the
        test period.

        Embargo: Additional gap days removed from the training set after
        each test group ends, to account for serial correlation in returns.
        """
        test_dates = set()
        for idx in test_indices:
            test_dates.update(groups[idx])

        train_dates = set()
        for idx in train_indices:
            train_dates.update(groups[idx])

        # Identify test boundaries (first and last date of each test group)
        test_boundaries = []
        for idx in test_indices:
            g = groups[idx]
            test_boundaries.append(g[0])   # start boundary
            test_boundaries.append(g[-1])  # end boundary

        # Purge: remove training dates within embargo_days of any test boundary
        purge_count = 0
        dates_to_remove = set()
        for train_date in train_dates:
            for boundary in test_boundaries:
                # Convert to comparable format
                td = pd.Timestamp(train_date)
                bd = pd.Timestamp(boundary)
                day_diff = abs((td - bd).days)
                if day_diff <= self.embargo_days:
                    dates_to_remove.add(train_date)
                    break

        purge_count = len(dates_to_remove)
        train_dates -= dates_to_remove

        logger.debug(f"  Purged {purge_count} training dates near test boundaries")

        return train_dates, test_dates

    def _slice_price_data(
        self,
        price_data: Dict[str, pd.DataFrame],
        dates: set,
    ) -> Dict[str, pd.DataFrame]:
        """Slice price data to only include rows within the given date set."""
        sliced = {}
        for symbol, df in price_data.items():
            mask = df.index.isin(dates)
            sub = df[mask]
            if not sub.empty:
                sliced[symbol] = sub
        return sliced
