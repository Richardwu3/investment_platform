"""
backtest/run_first_backtest.py
==============================
Entry point for the AI-Augmented Investment Research Platform.

This script:
    1. Downloads historical price data via yfinance
    2. Runs the GTAA strategy through the 5-layer validation engine
    3. Prints a formatted performance report to console
    4. Saves results to reports/ directory (JSON + CSV)

Usage:
    python backtest/run_first_backtest.py

    # Skip slow layers for rapid iteration:
    python backtest/run_first_backtest.py --fast

    # Custom date range:
    python backtest/run_first_backtest.py --start 2015-01-01 --end 2024-01-01

Requirements:
    pip install yfinance pandas numpy scikit-learn

Author: Yuchuan Wu
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup: allow imports from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.gtaa import GTAAStrategy
from backtest.engine import BacktestEngine, BacktestConfig

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

def download_prices(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    cache_dir: str = "data",
) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance.

    Uses a local CSV cache to avoid redundant API calls in repeated runs.
    Cache is invalidated if end date or tickers change.

    Args:
        tickers:   List of asset tickers (e.g. ['SPY', 'QQQ', 'TLT', 'GLD', 'DBC']).
        start:     Start date string 'YYYY-MM-DD'.
        end:       End date string 'YYYY-MM-DD'. If None, uses today.
        cache_dir: Directory for CSV cache files.

    Returns:
        DataFrame[DatetimeIndex × tickers] of adjusted close prices.
        Index is a clean DatetimeIndex with no time component.
        Rows with ALL NaN are dropped; remaining NaN filled forward then backward.

    Raises:
        ValueError: If download fails for all tickers.
        RuntimeError: If price data has fewer than 300 rows after cleaning.
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    cache_path = Path(cache_dir) / f"prices_{'_'.join(sorted(tickers))}_{start}_{end}.csv"

    # Try cache first
    if cache_path.exists():
        logger.info("Loading prices from cache: %s", cache_path)
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        prices.index = pd.DatetimeIndex(prices.index).normalize()
        logger.info("Cached data: %d rows, %s to %s", len(prices),
                    prices.index[0].date(), prices.index[-1].date())
        return prices

    logger.info("Downloading prices from Yahoo Finance: %s [%s → %s]", tickers, start, end)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    try:
        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=True,     # Returns adjusted prices directly in 'Close'
            progress=False,
            threads=True,
        )
    except Exception as e:
        raise ValueError(f"yfinance download failed: {e}") from e

    # Extract adjusted close
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][tickers].copy()
    else:
        # Single ticker case
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Clean index
    prices.index = pd.DatetimeIndex(prices.index).normalize()
    prices.index.name = "Date"

    # Drop rows where ALL assets are NaN (market holidays with no data)
    prices = prices.dropna(how="all")

    # Forward-fill then backward-fill remaining NaN (handles staggered listing dates)
    prices = prices.ffill().bfill()

    if len(prices) < 300:
        raise RuntimeError(
            f"Only {len(prices)} rows downloaded. "
            "Check tickers, date range, or internet connection."
        )

    logger.info(
        "Download complete: %d rows, %d assets, %s to %s",
        len(prices), len(prices.columns),
        prices.index[0].date(), prices.index[-1].date()
    )

    # Save to cache
    prices.to_csv(cache_path)
    logger.info("Cached to: %s", cache_path)

    return prices


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_pct(value: float | str, decimals: int = 2) -> str:
    """Format a float as a percentage string."""
    if isinstance(value, str):
        return value
    return f"{value * 100:.{decimals}f}%"


def format_ratio(value: float | str, decimals: int = 2) -> str:
    """Format a float as a ratio string."""
    if isinstance(value, str):
        return value
    return f"{value:.{decimals}f}"


def print_layer1_report(layer1: dict, config: BacktestConfig) -> None:
    """
    Print formatted Layer 1 (Standard Backtest) results to console.

    Args:
        layer1: Results dict from Layer1StandardBacktest.run().
        config: BacktestConfig for benchmark label.
    """
    sm = layer1["strategy_metrics"]
    bm = layer1["benchmark_metrics"]
    bmark_label = config.benchmark_ticker

    print("\n" + "=" * 65)
    print("  LAYER 1: STANDARD BACKTEST RESULTS")
    print("=" * 65)
    print(f"{'Metric':<30} {'Strategy':>15} {bmark_label + ' B&H':>15}")
    print("-" * 65)

    rows = [
        ("Total Return",          format_pct(sm["total_return"]),
                                  format_pct(bm["total_return"])),
        ("Annualized Return",     format_pct(sm["annualized_return"]),
                                  format_pct(bm["annualized_return"])),
        ("Annualized Volatility", format_pct(sm["annualized_vol"]),
                                  format_pct(bm["annualized_vol"])),
        ("Sharpe Ratio",          format_ratio(sm["sharpe_ratio"]),
                                  format_ratio(bm["sharpe_ratio"])),
        ("Sortino Ratio",         format_ratio(sm["sortino_ratio"]),
                                  format_ratio(bm["sortino_ratio"])),
        ("Max Drawdown",          format_pct(sm["max_drawdown"]),
                                  format_pct(bm["max_drawdown"])),
        ("Calmar Ratio",          format_ratio(sm["calmar_ratio"]),
                                  format_ratio(bm["calmar_ratio"])),
        ("Win Rate (daily)",      format_pct(sm["win_rate"]),
                                  format_pct(bm["win_rate"])),
        ("Period (years)",        format_ratio(sm["n_years"], 1),
                                  format_ratio(bm["n_years"], 1)),
    ]

    for label, strat_val, bench_val in rows:
        print(f"  {label:<28} {strat_val:>15} {bench_val:>15}")

    print("-" * 65)
    excess = sm.get("excess_return", "N/A")
    ir = sm.get("information_ratio", "N/A")
    print(f"  {'Excess Return vs Benchmark':<28} {format_pct(excess):>15}")
    print(f"  {'Information Ratio':<28} {format_ratio(ir) if isinstance(ir, float) else ir:>15}")
    print("=" * 65)


def print_layer2_report(layer2: dict) -> None:
    """
    Print Layer 2 (Walk-Forward) summary.

    Args:
        layer2: Results dict from Layer2WalkForward.run().
    """
    if layer2.get("skipped"):
        print("\n  [Layer 2: Walk-Forward — SKIPPED (--fast mode)]")
        return

    summary = layer2.get("summary", {})
    oos = summary.get("combined_oos_metrics", {})

    print("\n" + "=" * 65)
    print("  LAYER 2: WALK-FORWARD VALIDATION")
    print("=" * 65)
    print(f"  Windows run:                    {summary.get('n_windows', 'N/A'):>10}")
    print(f"  % Windows with positive Sharpe: {format_pct(summary.get('pct_positive_sharpe', 0)):>10}")
    print(f"  Mean window Sharpe:             {format_ratio(summary.get('mean_window_sharpe', 0)):>10}")
    print(f"  Std window Sharpe:              {format_ratio(summary.get('std_window_sharpe', 0)):>10}")
    print(f"  Combined OOS Ann. Return:       {format_pct(oos.get('annualized_return', 0)):>10}")
    print(f"  Combined OOS Sharpe:            {format_ratio(oos.get('sharpe_ratio', 0)):>10}")
    print(f"  Combined OOS Max Drawdown:      {format_pct(oos.get('max_drawdown', 0)):>10}")
    print("=" * 65)


def print_layer3_report(layer3: dict) -> None:
    """
    Print Layer 3 (Parameter Stability) summary.

    Args:
        layer3: Results dict from Layer3ParameterStability.run().
    """
    if layer3.get("skipped"):
        print("\n  [Layer 3: Parameter Stability — SKIPPED (--fast mode)]")
        return

    print("\n" + "=" * 65)
    print("  LAYER 3: PARAMETER STABILITY ANALYSIS")
    print("=" * 65)
    print(f"  Parameter grid tested:          {layer3.get('param_names', [])}")
    print(f"  Best Sharpe found:              {format_ratio(layer3.get('best_sharpe', 0)):>10}")
    print(f"  Best parameters:                {layer3.get('best_params', {})}")
    print(f"  Stability score (% positive):   {format_pct(layer3.get('stability_score', 0)):>10}")

    if layer3.get("heatmap_data") is not None:
        print("\n  Sharpe Ratio Heatmap:")
        print(layer3["heatmap_data"].round(2).to_string())
    print("=" * 65)


def print_layer4_report(layer4: dict) -> None:
    """
    Print Layer 4 (Regime Analysis) results.

    Args:
        layer4: Results dict from Layer4RegimeAnalysis.run().
    """
    print("\n" + "=" * 65)
    print("  LAYER 4: MARKET REGIME ANALYSIS")
    print("=" * 65)
    counts = layer4.get("regime_counts", {})
    for regime_key, metrics in layer4.get("regime_metrics", {}).items():
        n_days = counts.get(regime_key, 0)
        print(f"\n  [{regime_key.upper()} | {n_days} days]")
        print(f"    {metrics.get('regime_description', '')}")
        print(f"    Ann. Return:  {format_pct(metrics.get('annualized_return', 0))}")
        print(f"    Sharpe:       {format_ratio(metrics.get('sharpe_ratio', 0))}")
        print(f"    Max Drawdown: {format_pct(metrics.get('max_drawdown', 0))}")
    print("=" * 65)


def print_layer5_report(layer5: dict) -> None:
    """
    Print Layer 5 (Bootstrap Monte Carlo) results.

    Args:
        layer5: Results dict from Layer5BlockBootstrap.run().
    """
    print("\n" + "=" * 65)
    print("  LAYER 5: BLOCK BOOTSTRAP MONTE CARLO")
    print("=" * 65)
    print(f"  Simulations run:        {layer5.get('n_simulations', 'N/A'):>10}")
    print(f"  Block size:             {str(layer5.get('block_size_days', '')) + ' days':>10}")
    print(f"  Realized Sharpe:        {format_ratio(layer5.get('realized_sharpe', 0)):>10}")
    print(f"  Bootstrap mean Sharpe:  {format_ratio(layer5.get('bootstrap_mean_sharpe', 0)):>10}")
    print(f"  Bootstrap std Sharpe:   {format_ratio(layer5.get('bootstrap_std_sharpe', 0)):>10}")
    ci95 = layer5.get("ci_95", (0, 0))
    ci99 = layer5.get("ci_99", (0, 0))
    print(f"  95% CI:                 [{format_ratio(ci95[0])}, {format_ratio(ci95[1])}]")
    print(f"  99% CI:                 [{format_ratio(ci99[0])}, {format_ratio(ci99[1])}]")
    print(f"  p-value:                {layer5.get('p_value', 'N/A'):>10}")
    print(f"\n  ★ {layer5.get('significance', 'N/A')}")
    print("=" * 65)


def save_results(results: dict, output_dir: str = "reports") -> None:
    """
    Save backtest results to disk (JSON summary + CSV equity curves).

    Args:
        results:    Full results dict from BacktestEngine.run_all_layers().
        output_dir: Directory to save output files.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_name = results["summary"]["strategy_name"].replace(" ", "_")

    # Save equity curves as CSV
    equity_csv = Path(output_dir) / f"{strategy_name}_{timestamp}_equity.csv"
    equity_df = pd.DataFrame({
        "strategy_equity": results["layer_1"]["equity_curve"],
        "benchmark_equity": results["layer_1"]["benchmark_curve"],
        "strategy_returns": results["layer_1"]["strategy_returns"],
        "benchmark_returns": results["layer_1"]["benchmark_returns"],
    })
    equity_df.to_csv(equity_csv)
    logger.info("Equity curves saved: %s", equity_csv)

    # Save JSON summary (exclude non-serializable objects)
    def make_serializable(obj):
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, (pd.Timestamp,)):
            return str(obj)
        return str(obj)

    summary_json = Path(output_dir) / f"{strategy_name}_{timestamp}_summary.json"
    json_safe = json.loads(json.dumps(results["summary"], default=make_serializable))
    with open(summary_json, "w") as f:
        json.dump(json_safe, f, indent=2)
    logger.info("Summary JSON saved: %s", summary_json)

    # Save Layer 3 heatmap if available
    l3 = results.get("layer_3", {})
    if not l3.get("skipped") and l3.get("heatmap_data") is not None:
        heatmap_csv = Path(output_dir) / f"{strategy_name}_{timestamp}_heatmap.csv"
        l3["heatmap_data"].to_csv(heatmap_csv)
        logger.info("Parameter heatmap saved: %s", heatmap_csv)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="GTAA Strategy — 5-Layer Backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start", default="2010-01-01", help="Data start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Data end date YYYY-MM-DD (default: today)")
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip Layer 2 (walk-forward) and Layer 3 (param sweep) for speed"
    )
    parser.add_argument("--lookback", type=int, default=126, help="Momentum lookback days")
    parser.add_argument("--top-n", type=int, default=2, help="Top-N assets to hold")
    parser.add_argument("--no-abs-filter", action="store_true",
                        help="Disable absolute momentum filter (never go to cash)")
    parser.add_argument("--bootstrap-n", type=int, default=500,
                        help="Number of bootstrap simulations")
    parser.add_argument("--output-dir", default="reports", help="Output directory for reports")
    return parser.parse_args()


def main() -> None:
    """
    Main execution function.

    Orchestrates data download, strategy initialization, engine run,
    and report printing. All layers run sequentially with progress logging.
    """
    args = parse_args()

    print("\n" + "=" * 65)
    print("  AI-AUGMENTED INVESTMENT RESEARCH PLATFORM")
    print("  Columbia MAFN | Phase 1: GTAA Backtest")
    print("=" * 65)
    print(f"  Start:     {args.start}")
    print(f"  End:       {args.end or 'Today'}")
    print(f"  Lookback:  {args.lookback} trading days (~{args.lookback // 21} months)")
    print(f"  Top-N:     {args.top_n} assets")
    print(f"  Abs Filter:{not args.no_abs_filter}")
    print(f"  Fast Mode: {args.fast}")
    print("=" * 65 + "\n")

    # --- Step 1: Download data ---
    UNIVERSE = ["SPY", "QQQ", "TLT", "GLD", "DBC"]
    try:
        prices = download_prices(
            tickers=UNIVERSE,
            start=args.start,
            end=args.end,
            cache_dir="data",
        )
    except Exception as e:
        logger.error("Data download failed: %s", e)
        sys.exit(1)

    # --- Step 2: Initialize strategy ---
    strategy = GTAAStrategy(
        lookback_days=args.lookback,
        top_n=args.top_n,
        apply_abs_filter=not args.no_abs_filter,
        universe=UNIVERSE,
    )

    # --- Step 3: Configure and run engine ---
    config = BacktestConfig(
        slippage=0.001,
        commission=0.0005,
        initial_cash=100_000.0,
        benchmark_ticker="SPY",
        n_bootstrap=args.bootstrap_n,
        bootstrap_block=20,
        wf_train_years=3,
        wf_test_months=6,
    )

    engine = BacktestEngine(config)

    try:
        results = engine.run_all_layers(
            prices=prices,
            strategy=strategy,
            run_layer2=not args.fast,
            run_layer3=not args.fast,
        )
    except Exception as e:
        logger.error("Backtest engine failed: %s", e, exc_info=True)
        sys.exit(1)

    # --- Step 4: Print formatted report ---
    print_layer1_report(results["layer_1"], config)
    print_layer2_report(results["layer_2"])
    print_layer3_report(results["layer_3"])
    print_layer4_report(results["layer_4"])
    print_layer5_report(results["layer_5"])

    # --- Step 5: Save to disk ---
    save_results(results, output_dir=args.output_dir)

    # --- Step 6: AI Agent handoff summary ---
    print("\n" + "=" * 65)
    print("  AI AGENT HANDOFF — Ready for Narrative Generation")
    print("=" * 65)
    summary = results["summary"]
    print(f"  Strategy:          {summary['strategy_name']}")
    print(f"  Full-Sample Sharpe:{format_ratio(summary['full_sample_sharpe']):>10}")
    print(f"  Ann. Return:       {format_pct(summary['full_sample_ann_return']):>10}")
    print(f"  Max Drawdown:      {format_pct(summary['max_drawdown']):>10}")
    print(f"  vs. SPY Excess:    {format_pct(summary['vs_benchmark_excess']) if isinstance(summary['vs_benchmark_excess'], float) else summary['vs_benchmark_excess']:>10}")
    print(f"  Statistical Test:  {summary['statistical_significance']}")
    print("\n  → Pass results['summary'] and results['layer_1']['strategy_metadata']")
    print("    to AI Agent Layer for report generation.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
