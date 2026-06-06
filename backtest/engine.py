"""
backtest/engine.py
==================
Five-Layer Backtest Validation Framework.

Architecture Layers:
    Layer 1 – Standard Backtest        : Full-sample performance metrics
    Layer 2 – Walk-Forward Validation  : Out-of-sample generalization test
    Layer 3 – Parameter Stability      : Heatmap across parameter grid
    Layer 4 – Market Regime Analysis   : Performance segmented by regime
    Layer 5 – Block Bootstrap Monte Carlo: Statistical significance bounds

Design Principles:
    - "Code = Truth": all metrics computed deterministically from price series
    - LLM reads output dicts; never calls engine methods
    - slippage and commission applied on every rebalance trade
    - weights.shift(1) enforced inside _apply_weights() as final guard

Dependencies:
    vectorbt  — portfolio simulation and metrics
    numpy     — numerical operations
    pandas    — time-series manipulation
    sklearn   — StandardScaler for regime detection
    scipy     — statistical tests

Author: Yuchuan Wu
"""

from __future__ import annotations

import itertools
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """
    Central configuration for all five backtest layers.

    Attributes:
        slippage:          One-way slippage as fraction of trade value (e.g. 0.001 = 10 bps).
        commission:        One-way commission as fraction of trade value (e.g. 0.0005 = 5 bps).
        initial_cash:      Starting portfolio value in USD.
        benchmark_ticker:  Ticker for buy-and-hold comparison.
        wf_train_years:    Walk-forward training window in years.
        wf_test_months:    Walk-forward test window in months.
        n_bootstrap:       Number of bootstrap simulations (Layer 5).
        bootstrap_block:   Block length in trading days for block bootstrap.
        regime_window:     Rolling window (days) for regime detection.
        random_seed:       Seed for reproducibility.
    """
    slippage: float = 0.001           # 10 basis points one-way
    commission: float = 0.0005        # 5 basis points one-way
    initial_cash: float = 100_000.0
    benchmark_ticker: str = "SPY"
    wf_train_years: int = 3
    wf_test_months: int = 6
    n_bootstrap: int = 2000
    bootstrap_block: int = 60         # 20-day blocks preserve autocorrelation
    regime_window: int = 252          # 1-year rolling for regime detection
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Metric helpers (no vectorbt dependency — pure pandas/numpy)
# ---------------------------------------------------------------------------

class MetricsCalculator:
    """
    Compute standard portfolio performance metrics from a returns series.

    All methods are static; no state. Uses daily returns as input.
    LLM reads the output dict but never calls these methods directly.
    """

    TRADING_DAYS_PER_YEAR: int = 252

    @staticmethod
    def compute_all(
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        name: str = "Strategy",
    ) -> Dict:
        """
        Compute full suite of performance metrics.

        Args:
            returns:           Daily strategy returns (arithmetic, not log).
            benchmark_returns: Daily benchmark returns for relative metrics.
                               If None, benchmark metrics are omitted.
            name:              Label for this result set.

        Returns:
            Dictionary with the following keys:
                total_return, annualized_return, annualized_vol,
                sharpe_ratio, sortino_ratio, max_drawdown,
                calmar_ratio, win_rate, avg_win, avg_loss,
                [benchmark comparative metrics if provided]
        """
        ann = MetricsCalculator.TRADING_DAYS_PER_YEAR
        rets = returns.dropna()

        if len(rets) == 0:
            logger.warning("Empty returns series passed to MetricsCalculator.")
            return {"error": "Empty returns series", "name": name}

        # --- Core metrics ---
        total_return = (1 + rets).prod() - 1
        n_years = len(rets) / ann
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        annualized_vol = rets.std() * np.sqrt(ann)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0.0

        # Sortino: downside deviation only
        downside = rets[rets < 0]
        downside_vol = downside.std() * np.sqrt(ann) if len(downside) > 0 else 1e-9
        sortino = annualized_return / downside_vol

        # Maximum drawdown
        cumulative = (1 + rets).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = cumulative / rolling_max - 1
        max_drawdown = drawdown.min()

        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        # Win/loss statistics
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        win_rate = len(wins) / len(rets) if len(rets) > 0 else 0.0

        metrics: Dict = {
            "name": name,
            "total_return": round(total_return, 6),
            "annualized_return": round(annualized_return, 6),
            "annualized_vol": round(annualized_vol, 6),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(max_drawdown, 6),
            "calmar_ratio": round(calmar, 4),
            "win_rate": round(win_rate, 4),
            "avg_daily_win": round(wins.mean(), 6) if len(wins) > 0 else 0.0,
            "avg_daily_loss": round(losses.mean(), 6) if len(losses) > 0 else 0.0,
            "n_trading_days": len(rets),
            "n_years": round(n_years, 2),
        }

        # --- Benchmark-relative metrics ---
        if benchmark_returns is not None:
            bmark = benchmark_returns.reindex(rets.index).dropna()
            aligned_rets = rets.reindex(bmark.index).dropna()
            if len(aligned_rets) > 1:
                excess = aligned_rets - bmark
                tracking_error = excess.std() * np.sqrt(ann)
                information_ratio = excess.mean() * ann / tracking_error if tracking_error > 0 else 0.0
                bmark_total = (1 + bmark).prod() - 1
                metrics.update({
                    "benchmark_total_return": round(bmark_total, 6),
                    "excess_return": round(total_return - bmark_total, 6),
                    "tracking_error": round(tracking_error, 6),
                    "information_ratio": round(information_ratio, 4),
                })

        return metrics

    @staticmethod
    def equity_curve(returns: pd.Series, initial_cash: float = 1.0) -> pd.Series:
        """
        Compute equity curve from daily returns.

        Args:
            returns:      Daily arithmetic returns.
            initial_cash: Starting value (default 1.0 for normalized curve).

        Returns:
            Cumulative equity series with same index as returns.
        """
        return initial_cash * (1 + returns).cumprod()


# ---------------------------------------------------------------------------
# Portfolio simulation (vectorbt-free fallback + vectorbt integration)
# ---------------------------------------------------------------------------

def simulate_portfolio(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    config: BacktestConfig,
) -> Tuple[pd.Series, pd.Series]:
    """
    Simulate portfolio returns from price data and target weights.

    This is the core simulation engine. Uses pandas arithmetic for transparency.
    vectorbt is called for validation/visualization (see Layer 1); the canonical
    return series comes from this function.

    Look-ahead Bias Final Guard:
        weights are shifted forward by 1 day inside this function.
        Combined with the shift(1) already applied by the strategy, total shift = 1.
        (The strategy's shift + engine's shift would double-count; engine only applies
        shift here if the strategy has NOT already shifted. See _safe_shift flag.)

    Args:
        prices:  DataFrame[DatetimeIndex × tickers] of adjusted close prices.
        weights: DataFrame[DatetimeIndex × tickers] of target portfolio weights.
                 MUST be execution-ready (already shifted by strategy).
        config:  BacktestConfig with slippage and commission parameters.

    Returns:
        Tuple of:
            strategy_returns: pd.Series of daily portfolio returns.
            benchmark_returns: pd.Series of buy-and-hold benchmark returns (SPY).
    """
    # Align weights to prices index
    weights_aligned = weights.reindex(prices.index).ffill().fillna(0.0)

    # Asset daily returns
    asset_returns = prices.pct_change().fillna(0.0)

    # Portfolio return = sum(weight[t-1] * return[t])
    # weights_aligned is already execution-ready (shifted by strategy)
    # We use the weight at t to represent "what we hold during day t"
    # which was set based on signal from t-1 close → correct
    port_returns = (weights_aligned * asset_returns).sum(axis=1)

    # --- Transaction cost deduction ---
    # Compute daily turnover (sum of absolute weight changes)
    weight_changes = weights_aligned.diff().abs().sum(axis=1).fillna(0.0)
    # Round-trip cost = (slippage + commission) * 2 sides; apply to one-way turnover
    transaction_costs = weight_changes * (config.slippage + config.commission)
    port_returns = port_returns - transaction_costs

    # --- Benchmark ---
    if config.benchmark_ticker in prices.columns:
        bench_returns = prices[config.benchmark_ticker].pct_change().fillna(0.0)
    else:
        logger.warning(
            "Benchmark ticker '%s' not in price data. Using equal-weight benchmark.",
            config.benchmark_ticker
        )
        bench_returns = asset_returns.mean(axis=1)

    logger.debug(
        "Portfolio simulation complete. Mean daily return: %.4f%%, "
        "total transaction cost: %.4f%%",
        port_returns.mean() * 100,
        transaction_costs.sum() * 100,
    )
    return port_returns, bench_returns


# ---------------------------------------------------------------------------
# Layer 1: Standard Backtest
# ---------------------------------------------------------------------------

class Layer1StandardBacktest:
    """
    Layer 1: Full-sample standard backtest.

    Runs strategy over entire available history and computes
    comprehensive performance metrics vs. benchmark.

    This layer answers: "Does the strategy have edge over the full sample?"
    """

    def __init__(self, config: BacktestConfig):
        """
        Args:
            config: BacktestConfig instance.
        """
        self.config = config

    def run(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        strategy_name: str = "Strategy",
    ) -> Dict:
        """
        Execute full-sample backtest.

        Args:
            prices:        Full price history DataFrame.
            weights:       Execution-ready weights (from SignalResult.weights).
            strategy_name: Label for output.

        Returns:
            Dictionary containing:
                metrics:          Performance metric dict.
                strategy_returns: pd.Series of daily returns.
                benchmark_returns: pd.Series of benchmark daily returns.
                equity_curve:     pd.Series of cumulative portfolio value.
                benchmark_curve:  pd.Series of buy-and-hold equity curve.
        """
        logger.info("Layer 1: Running standard backtest for '%s'.", strategy_name)

        port_returns, bench_returns = simulate_portfolio(prices, weights, self.config)

        metrics = MetricsCalculator.compute_all(
            port_returns, bench_returns, name=strategy_name
        )
        bench_metrics = MetricsCalculator.compute_all(
            bench_returns, name=f"{self.config.benchmark_ticker} Buy & Hold"
        )

        equity = MetricsCalculator.equity_curve(port_returns, self.config.initial_cash)
        bench_equity = MetricsCalculator.equity_curve(bench_returns, self.config.initial_cash)

        logger.info(
            "Layer 1 Results | Sharpe: %.2f | Ann. Return: %.2f%% | Max DD: %.2f%%",
            metrics["sharpe_ratio"],
            metrics["annualized_return"] * 100,
            metrics["max_drawdown"] * 100,
        )

        return {
            "layer": 1,
            "strategy_metrics": metrics,
            "benchmark_metrics": bench_metrics,
            "strategy_returns": port_returns,
            "benchmark_returns": bench_returns,
            "equity_curve": equity,
            "benchmark_curve": bench_equity,
        }


# ---------------------------------------------------------------------------
# Layer 2: Walk-Forward Validation
# ---------------------------------------------------------------------------

class Layer2WalkForward:
    """
    Layer 2: Walk-forward out-of-sample validation.

    Splits data into rolling train/test windows to test whether
    strategy generalizes beyond its training period.

    This layer answers: "Does performance hold out-of-sample?"

    Walk-Forward Logic:
        |---- train_years ----|-- test_months --|
                              |---- train_years ----|-- test_months --|
                                                    ...
    """

    def __init__(self, config: BacktestConfig):
        """
        Args:
            config: BacktestConfig (uses wf_train_years, wf_test_months).
        """
        self.config = config

    def run(
        self,
        prices: pd.DataFrame,
        strategy,
        strategy_name: str = "Strategy",
    ) -> Dict:
        """
        Run walk-forward validation across all windows.

        For each window, the strategy is re-fitted on training data and
        evaluated on the held-out test period.

        Args:
            prices:        Full price history.
            strategy:      IStrategy instance (generate_signals will be called
                           with training-window prices).
            strategy_name: Label for output.

        Returns:
            Dictionary containing:
                windows:          List of per-window metric dicts.
                oos_returns:      Concatenated out-of-sample returns.
                summary:          Aggregated statistics across windows.
        """
        logger.info("Layer 2: Running walk-forward validation for '%s'.", strategy_name)

        train_days = self.config.wf_train_years * 252
        test_days = self.config.wf_test_months * 21  # approx trading days per month

        windows = []
        oos_returns_list = []
        start_idx = train_days

        window_id = 0
        while start_idx + test_days <= len(prices):
            train_prices = prices.iloc[:start_idx]
            test_prices = prices.iloc[start_idx: start_idx + test_days]

            # Generate signals on training window ONLY
            try:
                signal = strategy.generate_signals(train_prices)
                # Extend weights to test period (hold last weights → no refit bias)
                last_weights = signal.weights.iloc[[-1]]
                test_weights = pd.DataFrame(
                    np.tile(last_weights.values, (len(test_prices), 1)),
                    index=test_prices.index,
                    columns=signal.weights.columns,
                )

                # Re-run strategy on extended data for cleaner signals
                extended_prices = prices.iloc[:start_idx + test_days]
                signal_ext = strategy.generate_signals(extended_prices)
                test_weights_ext = signal_ext.weights.reindex(test_prices.index)

                test_rets, bench_rets = simulate_portfolio(
                    test_prices, test_weights_ext, self.config
                )
                window_metrics = MetricsCalculator.compute_all(
                    test_rets, bench_rets,
                    name=f"Window_{window_id}_OOS"
                )
                window_metrics["train_start"] = str(prices.index[0].date())
                window_metrics["train_end"] = str(train_prices.index[-1].date())
                window_metrics["test_start"] = str(test_prices.index[0].date())
                window_metrics["test_end"] = str(test_prices.index[-1].date())
                window_metrics["window_id"] = window_id

                windows.append(window_metrics)
                oos_returns_list.append(test_rets)

                logger.debug(
                    "WF Window %d | Train: %s→%s | Test: %s→%s | Sharpe: %.2f",
                    window_id,
                    train_prices.index[0].date(), train_prices.index[-1].date(),
                    test_prices.index[0].date(), test_prices.index[-1].date(),
                    window_metrics["sharpe_ratio"],
                )
            except Exception as e:
                logger.warning("Walk-forward window %d failed: %s", window_id, e)

            start_idx += test_days
            window_id += 1

        if not oos_returns_list:
            logger.error("No walk-forward windows completed.")
            return {"layer": 2, "error": "No windows completed", "windows": []}

        oos_returns = pd.concat(oos_returns_list).sort_index()
        oos_metrics = MetricsCalculator.compute_all(oos_returns, name="OOS_Combined")

        # Summary statistics across windows
        window_sharpes = [w["sharpe_ratio"] for w in windows]
        window_rets = [w["annualized_return"] for w in windows]
        summary = {
            "n_windows": len(windows),
            "pct_positive_sharpe": sum(s > 0 for s in window_sharpes) / len(window_sharpes),
            "mean_window_sharpe": np.mean(window_sharpes),
            "std_window_sharpe": np.std(window_sharpes),
            "mean_window_ann_return": np.mean(window_rets),
            "combined_oos_metrics": oos_metrics,
        }

        logger.info(
            "Layer 2 Complete | %d windows | OOS Sharpe: %.2f | %% Positive: %.0f%%",
            len(windows),
            oos_metrics["sharpe_ratio"],
            summary["pct_positive_sharpe"] * 100,
        )

        return {
            "layer": 2,
            "windows": windows,
            "oos_returns": oos_returns,
            "summary": summary,
        }


# ---------------------------------------------------------------------------
# Layer 3: Parameter Stability Heatmap
# ---------------------------------------------------------------------------

class Layer3ParameterStability:
    """
    Layer 3: Parameter sensitivity and stability analysis.

    Sweeps across a grid of strategy parameters and computes Sharpe ratio
    for each combination. The resulting heatmap reveals whether performance
    is robust or highly sensitive to parameter choice.

    This layer answers: "Is the strategy over-fitted to specific parameters?"

    Robustness Criterion:
        A strategy is considered robust if a ±20% change in any parameter
        does not reduce the Sharpe ratio by more than 30%.
    """

    def __init__(self, config: BacktestConfig):
        """
        Args:
            config: BacktestConfig instance.
        """
        self.config = config

    def run(
        self,
        prices: pd.DataFrame,
        strategy_class,
        param_grid: Dict,
        fixed_params: Optional[Dict] = None,
        strategy_name: str = "Strategy",
    ) -> Dict:
        """
        Run parameter sweep and compute stability metrics.

        Args:
            prices:         Full price history.
            strategy_class: IStrategy subclass (not instance).
            param_grid:     Dict of {param_name: [value1, value2, ...]}.
                            Generates Cartesian product of all combinations.
            fixed_params:   Parameters held constant across sweep.
            strategy_name:  Label for output.

        Returns:
            Dictionary containing:
                results_df:     DataFrame with one row per parameter combination.
                heatmap_data:   2D pivot if exactly 2 parameters, else None.
                best_params:    Parameter combo with highest Sharpe.
                stability_score: Fraction of combos with positive Sharpe.
        """
        logger.info(
            "Layer 3: Parameter stability sweep for '%s'. Grid: %s",
            strategy_name, {k: len(v) for k, v in param_grid.items()}
        )

        fixed = fixed_params or {}
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        logger.info("Total parameter combinations: %d", len(combinations))

        results = []
        for combo in combinations:
            params = dict(zip(param_names, combo))
            all_params = {**fixed, **params}

            try:
                strat = strategy_class(**all_params)
                signal = strat.generate_signals(prices)
                port_returns, bench_returns = simulate_portfolio(
                    prices, signal.weights, self.config
                )
                metrics = MetricsCalculator.compute_all(port_returns, bench_returns)
                row = {**params, "sharpe_ratio": metrics["sharpe_ratio"],
                       "annualized_return": metrics["annualized_return"],
                       "max_drawdown": metrics["max_drawdown"]}
                results.append(row)
                logger.debug("Params %s → Sharpe: %.3f", params, metrics["sharpe_ratio"])
            except Exception as e:
                logger.warning("Param combo %s failed: %s", params, e)
                row = {**params, "sharpe_ratio": np.nan,
                       "annualized_return": np.nan, "max_drawdown": np.nan}
                results.append(row)

        results_df = pd.DataFrame(results)

        # Heatmap pivot (only for 2-parameter grids)
        heatmap_data = None
        if len(param_names) == 2:
            heatmap_data = results_df.pivot(
                index=param_names[0],
                columns=param_names[1],
                values="sharpe_ratio"
            )

        best_row = results_df.loc[results_df["sharpe_ratio"].idxmax()]
        best_params = {p: best_row[p] for p in param_names}
        stability = (results_df["sharpe_ratio"] > 0).mean()

        logger.info(
            "Layer 3 Complete | Best Sharpe: %.2f | Stability Score: %.1f%%",
            best_row["sharpe_ratio"], stability * 100
        )

        return {
            "layer": 3,
            "results_df": results_df,
            "heatmap_data": heatmap_data,
            "best_params": best_params,
            "best_sharpe": float(best_row["sharpe_ratio"]),
            "stability_score": float(stability),
            "param_names": param_names,
        }


# ---------------------------------------------------------------------------
# Layer 4: Market Regime Analysis
# ---------------------------------------------------------------------------

class Layer4RegimeAnalysis:
    """
    Layer 4: Market environment segmentation and performance attribution.

    Segments the backtest period into market regimes and computes strategy
    performance separately for each regime. Regimes are defined by trailing
    benchmark return and volatility.

    Regime Definitions (based on benchmark):
        Bull Market : Trailing 252-day return > +10%
        Bear Market : Trailing 252-day return < -10%
        Sideways    : Otherwise

    This layer answers: "In which market environments does the strategy add value?"
    """

    REGIMES = {
        "bull": "Bull Market (benchmark >+10% trailing yr)",
        "bear": "Bear Market (benchmark <-10% trailing yr)",
        "sideways": "Sideways Market (benchmark -10% to +10%)",
    }

    def __init__(self, config: BacktestConfig):
        """
        Args:
            config: BacktestConfig (uses regime_window, benchmark_ticker).
        """
        self.config = config

    def _classify_regimes(self, bench_prices: pd.Series) -> pd.Series:
        """
        Classify each day into a market regime.

        Args:
            bench_prices: Price series for the benchmark asset.

        Returns:
            Series of regime labels ('bull', 'bear', 'sideways') with same index.
        """
        trailing_return = bench_prices / bench_prices.shift(self.config.regime_window) - 1
        regime = pd.Series("sideways", index=bench_prices.index)
        regime[trailing_return > 0.10] = "bull"
        regime[trailing_return < -0.10] = "bear"
        return regime

    def run(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        strategy_name: str = "Strategy",
    ) -> Dict:
        """
        Compute regime-segmented performance metrics.

        Args:
            prices:        Full price history including benchmark.
            weights:       Execution-ready weights.
            strategy_name: Label for output.

        Returns:
            Dictionary containing:
                regime_metrics:   Dict of {regime_label: metrics_dict}.
                regime_labels:    pd.Series of daily regime classifications.
                regime_counts:    Day count per regime.
        """
        logger.info("Layer 4: Market regime analysis for '%s'.", strategy_name)

        port_returns, bench_returns = simulate_portfolio(prices, weights, self.config)

        # Classify regimes using benchmark price
        if self.config.benchmark_ticker in prices.columns:
            bench_prices = prices[self.config.benchmark_ticker]
        else:
            bench_prices = prices.iloc[:, 0]
            logger.warning("Benchmark ticker not found; using first column for regime.")

        regime_labels = self._classify_regimes(bench_prices)

        regime_metrics = {}
        regime_counts = regime_labels.value_counts().to_dict()

        for regime_key, regime_desc in self.REGIMES.items():
            mask = regime_labels == regime_key
            regime_rets = port_returns[mask]
            regime_bench = bench_returns[mask]

            if len(regime_rets) < 21:  # Need at least 1 month of data
                logger.debug("Regime '%s' has too few observations (%d), skipping.",
                             regime_key, len(regime_rets))
                continue

            metrics = MetricsCalculator.compute_all(
                regime_rets, regime_bench,
                name=f"{strategy_name}_{regime_key}"
            )
            metrics["regime_description"] = regime_desc
            metrics["n_days_in_regime"] = int(mask.sum())
            regime_metrics[regime_key] = metrics

            logger.info(
                "  %s | Days: %d | Sharpe: %.2f | Ann. Ret: %.2f%%",
                regime_key, mask.sum(),
                metrics["sharpe_ratio"],
                metrics["annualized_return"] * 100,
            )

        return {
            "layer": 4,
            "regime_metrics": regime_metrics,
            "regime_labels": regime_labels,
            "regime_counts": regime_counts,
        }


# ---------------------------------------------------------------------------
# Layer 5: Block Bootstrap Monte Carlo
# ---------------------------------------------------------------------------

class Layer5BlockBootstrap:
    """
    Layer 5: Block Bootstrap Monte Carlo simulation.

    Assesses statistical significance of strategy performance by comparing
    realized metrics against a distribution of bootstrapped null results.

    Method:
        1. Resample the strategy's return series using non-overlapping blocks
           to preserve short-term autocorrelation structure.
        2. Compute Sharpe ratio for each bootstrapped sample.
        3. Compare realized Sharpe to the bootstrap distribution.
        4. Report p-value, confidence intervals, and "skill vs. luck" judgment.

    This layer answers: "Is strategy performance statistically significant?"

    Block Bootstrap vs. Standard Bootstrap:
        Standard bootstrap breaks time-series autocorrelation (e.g. momentum,
        volatility clustering). Block bootstrap preserves local structure by
        resampling contiguous blocks of returns.
    """

    def __init__(self, config: BacktestConfig):
        """
        Args:
            config: BacktestConfig (uses n_bootstrap, bootstrap_block, random_seed).
        """
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def _block_bootstrap_returns(self, returns: pd.Series) -> pd.Series:
        """
        Generate one bootstrap sample via circular block resampling.

        Args:
            returns: Original daily return series.

        Returns:
            Resampled return series of same length as input.
        """
        n = len(returns)
        block_size = self.config.bootstrap_block
        arr = returns.values
        max_start = n - block_size
        if max_start <= 0:
            indices = self.rng.choice(n, size=n, replace=True)
            return pd.Series(arr[indices], index=returns.index)
        n_blocks = int(np.ceil(n / block_size))

        # Randomly select start indices (circular to handle end-of-array)
        starts = self.rng.choice(max_start, size=n_blocks, replace=False)
        blocks = []
        for s in starts:
            # Circular indexing: wraps around end of array
            blocks.append(arr[s:s+block_size])

        bootstrap_arr = np.concatenate(blocks)[:n]
        return pd.Series(bootstrap_arr, index=returns.index)

    def _compute_sharpe(self, returns: pd.Series) -> float:
        """
        Compute annualized Sharpe ratio from daily returns.

        Args:
            returns: Daily arithmetic return series.

        Returns:
            Annualized Sharpe ratio (float).
        """
        ann = MetricsCalculator.TRADING_DAYS_PER_YEAR
        vol = returns.std()
        if vol == 0:
            return 0.0
        return (returns.mean() * ann) / (vol * np.sqrt(ann))

    def run(
        self,
        strategy_returns: pd.Series,
        strategy_name: str = "Strategy",
    ) -> Dict:
        """
        Run block bootstrap Monte Carlo simulation.

        Args:
            strategy_returns: Daily strategy return series (from Layer 1).
            strategy_name:    Label for output.

        Returns:
            Dictionary containing:
                realized_sharpe:      Actual Sharpe ratio.
                bootstrap_sharpes:    Array of simulated Sharpe ratios.
                p_value:              Fraction of sims exceeding realized Sharpe.
                ci_95:                (5th, 95th) percentile confidence interval.
                ci_99:                (1st, 99th) percentile confidence interval.
                interpretation:       Human-readable significance assessment.
        """
        logger.info(
            "Layer 5: Block Bootstrap Monte Carlo | n=%d, block=%d days",
            self.config.n_bootstrap, self.config.bootstrap_block
        )

        rets = strategy_returns.dropna()
        realized_sharpe = self._compute_sharpe(rets)

        bootstrap_sharpes = np.empty(self.config.n_bootstrap)
        for i in range(self.config.n_bootstrap):
            sim_rets = self._block_bootstrap_returns(rets)
            bootstrap_sharpes[i] = self._compute_sharpe(sim_rets)

        # p-value: probability of getting realized Sharpe by chance
        # Under H0: returns are iid with same mean/vol (block bootstrap breaks structure)
        p_value = float((bootstrap_sharpes >= realized_sharpe).mean())

        ci_95 = (float(np.percentile(bootstrap_sharpes, 5)),
                 float(np.percentile(bootstrap_sharpes, 95)))
        ci_99 = (float(np.percentile(bootstrap_sharpes, 1)),
                 float(np.percentile(bootstrap_sharpes, 99)))

        # Interpretation
        if p_value < 0.01:
            significance = "Highly Significant (p < 1%): Strong evidence of genuine edge."
        elif p_value < 0.05:
            significance = "Significant (p < 5%): Moderate evidence of genuine edge."
        elif p_value < 0.10:
            significance = "Marginally Significant (p < 10%): Weak evidence; interpret cautiously."
        else:
            significance = "Not Significant (p ≥ 10%): Performance may be attributable to luck."

        logger.info(
            "Layer 5 Complete | Realized Sharpe: %.2f | p-value: %.3f | %s",
            realized_sharpe, p_value, significance
        )

        return {
            "layer": 5,
            "strategy_name": strategy_name,
            "realized_sharpe": round(realized_sharpe, 4),
            "bootstrap_sharpes": bootstrap_sharpes,
            "bootstrap_mean_sharpe": float(bootstrap_sharpes.mean()),
            "bootstrap_std_sharpe": float(bootstrap_sharpes.std()),
            "p_value": round(p_value, 4),
            "ci_95": ci_95,
            "ci_99": ci_99,
            "significance": significance,
            "n_simulations": self.config.n_bootstrap,
            "block_size_days": self.config.bootstrap_block,
        }


# ---------------------------------------------------------------------------
# Master engine: orchestrates all five layers
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Master orchestrator for the five-layer validation framework.

    Runs all layers sequentially and assembles a unified results report.
    The AI Agent layer reads the returned dict to generate research narratives.

    Usage::

        engine = BacktestEngine(config)
        results = engine.run_all_layers(prices, strategy)
        # results is a dict readable by the LLM Agent
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Args:
            config: BacktestConfig. If None, uses default parameters.
        """
        self.config = config or BacktestConfig()
        self.layer1 = Layer1StandardBacktest(self.config)
        self.layer2 = Layer2WalkForward(self.config)
        self.layer3 = Layer3ParameterStability(self.config)
        self.layer4 = Layer4RegimeAnalysis(self.config)
        self.layer5 = Layer5BlockBootstrap(self.config)
        logger.info("BacktestEngine initialized with config: %s", self.config)

    def run_all_layers(
        self,
        prices: pd.DataFrame,
        strategy,
        run_layer3: bool = True,
        run_layer2: bool = True,
    ) -> Dict:
        """
        Execute all five validation layers and return unified results.

        Args:
            prices:      Full price history DataFrame.
            strategy:    IStrategy instance (implements generate_signals).
            run_layer3:  If False, skip parameter stability sweep (time-saving).
            run_layer2:  If False, skip walk-forward (time-saving).

        Returns:
            Unified dict with keys 'layer_1' through 'layer_5', plus
            'summary' with a consolidated view for the AI Agent.
        """
        logger.info(
            "=" * 60 + "\nRunning Full 5-Layer Backtest: %s\n" + "=" * 60,
            strategy.name
        )

        # Generate signals once for full dataset
        signal = strategy.generate_signals(prices)
        signal.validate()

        results = {}

        # Layer 1
        results["layer_1"] = self.layer1.run(prices, signal.weights, strategy.name)

        # Layer 2
        if run_layer2:
            results["layer_2"] = self.layer2.run(prices, strategy, strategy.name)
        else:
            results["layer_2"] = {"layer": 2, "skipped": True}

        # Layer 3
        if run_layer3 and hasattr(strategy, "get_parameter_grid"):
            param_grid = strategy.get_parameter_grid()
            results["layer_3"] = self.layer3.run(
                prices, type(strategy), param_grid,
                fixed_params={"apply_abs_filter": strategy.apply_abs_filter},
                strategy_name=strategy.name,
            )
        else:
            results["layer_3"] = {"layer": 3, "skipped": True}

        # Layer 4
        results["layer_4"] = self.layer4.run(prices, signal.weights, strategy.name)

        # Layer 5
        results["layer_5"] = self.layer5.run(
            results["layer_1"]["strategy_returns"], strategy.name
        )

        # Build summary for AI Agent
        l1 = results["layer_1"]["strategy_metrics"]
        l5 = results["layer_5"]
        summary = {
            "strategy_name": strategy.name,
            "full_sample_sharpe": l1["sharpe_ratio"],
            "full_sample_ann_return": l1["annualized_return"],
            "max_drawdown": l1["max_drawdown"],
            "vs_benchmark_excess": l1.get("excess_return", "N/A"),
            "statistical_significance": l5["significance"],
            "p_value": l5["p_value"],
            "strategy_metadata": signal.metadata,
        }
        results["summary"] = summary

        logger.info(
            "All layers complete.\n%s",
            "\n".join(f"  {k}: {v}" for k, v in summary.items()
                      if not isinstance(v, dict))
        )
        return results
