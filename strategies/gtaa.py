"""
strategies/gtaa.py
==================
Global Tactical Asset Allocation (GTAA) Strategy.

Reference:
    Faber, M. (2007). "A Quantitative Approach to Tactical Asset Allocation."
    Journal of Wealth Management.

Strategy Rules:
    Universe  : SPY (US Equity), QQQ (Tech), TLT (Long Bond), GLD (Gold), DBC (Commodities)
    Signal    : 126-trading-day momentum (≈ 6-month return)
    Selection : Top-2 assets by momentum score
    Weights   : Equal weight (50% each) among selected assets
    Rebalance : Signal generated at month-end close → executed at next open (month+1 day 1)
    Cash Rule : If fewer than 2 assets have positive momentum, go 100% cash

Look-ahead Bias Controls:
    1. prices.shift(1) inside compute_momentum() → signal at t uses data through t-1
    2. weights.shift(1) in engine → weight generated at month-end applied next trading day
    3. No future data injected; engine passes full slice to generate_signals()

Author: Yuchuan Wu
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.base import IStrategy, SignalResult, compute_momentum, equal_weight_top_n, is_month_end

logger = logging.getLogger(__name__)


class GTAAStrategy(IStrategy):
    """
    Global Tactical Asset Allocation strategy.

    A momentum-based tactical allocation that rotates among five asset classes.
    Selects the top-N assets by trailing momentum and holds them equally weighted.
    Moves to cash when momentum is negative for selected assets (optional filter).

    Attributes:
        lookback_days:    Trading days for momentum lookback (default 126 ≈ 6 months).
        top_n:            Number of top assets to hold (default 2).
        apply_abs_filter: If True, assets with negative momentum receive 0 weight
                          (strategy goes partially/fully to cash). Default True.
        universe:         List of asset tickers.
    """

    DEFAULT_UNIVERSE = ["SPY", "QQQ", "TLT", "GLD", "DBC"]

    def __init__(
        self,
        lookback_days: int = 126,
        top_n: int = 2,
        apply_abs_filter: bool = True,
        universe: Optional[List[str]] = None,
    ):
        """
        Initialize GTAA strategy.

        Args:
            lookback_days:    Momentum lookback in trading days. 126 ≈ 6 calendar months.
            top_n:            Number of assets selected each rebalance. Must be < len(universe).
            apply_abs_filter: If True, exclude assets with negative momentum (go to cash).
            universe:         Asset tickers. Defaults to [SPY, QQQ, TLT, GLD, DBC].
        """
        _universe = universe or self.DEFAULT_UNIVERSE
        super().__init__(name=f"GTAA_{lookback_days}d_Top{top_n}", universe=_universe)
        self.lookback_days = lookback_days
        self.top_n = top_n
        self.apply_abs_filter = apply_abs_filter

        if top_n >= len(_universe):
            raise ValueError(
                f"top_n ({top_n}) must be less than universe size ({len(_universe)})."
            )
        logger.info(
            "GTAAStrategy config: lookback=%d, top_n=%d, abs_filter=%s, universe=%s",
            lookback_days, top_n, apply_abs_filter, _universe,
        )

    # ------------------------------------------------------------------
    # Core signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, prices: pd.DataFrame) -> SignalResult:
        """
        Generate portfolio weights from historical price data.

        Algorithm:
            1. Compute 126-day momentum for each asset (using shifted prices).
            2. Identify month-end trading days.
            3. On each month-end, rank assets by momentum and select top-N.
            4. Optionally zero out assets with negative momentum.
            5. Forward-fill weights to daily frequency.
            6. Shift weights forward by 1 day (signal at close t → execution at open t+1).

        Args:
            prices: DataFrame[DatetimeIndex × tickers] of adjusted close prices.
                    Must contain all tickers in self.universe.
                    Assumed daily frequency; missing dates already handled by caller.

        Returns:
            SignalResult with:
                weights: Daily weights shifted by 1 (execution-ready).
                metadata: Momentum scores and selected assets at each rebalance.
                rebalance_dates: Month-end dates where portfolio was recalculated.

        Raises:
            KeyError: If any universe ticker is missing from prices columns.
            ValueError: If prices has fewer rows than lookback_days + 1.
        """
        # --- Validation ---
        missing = set(self.universe) - set(prices.columns)
        if missing:
            raise KeyError(f"Tickers missing from price data: {missing}")

        prices = prices[self.universe].copy()
        min_rows = self.lookback_days + 1
        if len(prices) < min_rows:
            raise ValueError(
                f"Need at least {min_rows} rows; got {len(prices)}. "
                "Extend the data download window."
            )

        logger.info(
            "Generating GTAA signals: %d rows, %s to %s",
            len(prices), prices.index[0].date(), prices.index[-1].date()
        )

        # --- Step 1: Momentum scores (shift(1) inside compute_momentum) ---
        # momentum[t] = lagged_price[t] / lagged_price[t - lookback] - 1
        # where lagged_price = prices.shift(1)
        # → signal at t uses prices through close of t-1 (conservative)
        momentum_scores = compute_momentum(prices, self.lookback_days)

        # --- Step 2: Identify month-end trading days ---
        month_end_mask = is_month_end(prices.index)
        rebalance_dates = prices.index[month_end_mask].tolist()

        # --- Step 3 & 4: Compute weights on each rebalance date ---
        # Initialize with NaN; ffill will propagate last valid weight
        raw_weights = pd.DataFrame(np.nan, index=prices.index, columns=self.universe)
        metadata_log: Dict = {"rebalance_records": []}

        # Before enough history, set to zero (warm-up period)
        warmup_end = prices.index[self.lookback_days]
        raw_weights.loc[prices.index < warmup_end] = 0.0

        for rebal_date in rebalance_dates:
            if rebal_date < warmup_end:
                raw_weights.loc[rebal_date] = 0.0
                continue

            scores_today = momentum_scores.loc[rebal_date]

            # Apply absolute momentum filter (optional)
            if self.apply_abs_filter:
                scores_filtered = scores_today.copy()
                scores_filtered[scores_filtered <= 0] = np.nan
            else:
                scores_filtered = scores_today.copy()

            # Equal-weight top-N
            w = equal_weight_top_n(scores_filtered, self.top_n)
            raw_weights.loc[rebal_date] = w.values

            # Log for AI narrative layer
            selected = w[w > 0].index.tolist()
            metadata_log["rebalance_records"].append({
                "date": str(rebal_date.date()),
                "selected_assets": selected,
                "momentum_scores": scores_today.round(4).to_dict(),
                "weights": w.round(4).to_dict(),
            })
            logger.debug(
                "Rebalance %s: selected=%s, scores=%s",
                rebal_date.date(), selected,
                {k: round(v, 3) for k, v in scores_today.items()}
            )

        # --- Step 5: Forward-fill weights to daily frequency ---
        # ffill carries last rebalance weight until next month-end signal
        raw_weights = raw_weights.ffill()

        # Any remaining NaN at start → zero (insufficient history)
        raw_weights = raw_weights.fillna(0.0)

        # --- Step 6: Shift forward by 1 trading day ---
        # CRITICAL: This is the primary look-ahead bias guard at the engine boundary.
        # Signal generated at close of month-end t → applied at open of day t+1.
        # The engine also applies an additional shift(1) to be safe; see engine.py.
        execution_weights = raw_weights.shift(1).fillna(0.0)

        # --- Build result ---
        result = SignalResult(
            weights=execution_weights,
            metadata={
                "strategy_config": {
                    "lookback_days": self.lookback_days,
                    "top_n": self.top_n,
                    "apply_abs_filter": self.apply_abs_filter,
                    "universe": self.universe,
                },
                **metadata_log,
            },
            rebalance_dates=rebalance_dates,
            strategy_name=self.name,
        )
        result.validate()

        logger.info(
            "Signal generation complete. %d rebalance events. "
            "Invested days: %d / %d (%.1f%%)",
            len(rebalance_dates),
            (execution_weights.sum(axis=1) > 0).sum(),
            len(execution_weights),
            100 * (execution_weights.sum(axis=1) > 0).mean(),
        )
        return result

    # ------------------------------------------------------------------
    # Metadata for AI narrative layer
    # ------------------------------------------------------------------

    def get_metadata(self) -> Dict:
        """
        Return strategy parameters for LLM narrative generation.

        The AI Agent reads this to construct plain-English explanations of
        strategy logic. This method never modifies strategy state.

        Returns:
            Dictionary with strategy description, parameters, and rationale.
        """
        return {
            "strategy_name": self.name,
            "strategy_type": "Momentum-based Tactical Asset Allocation",
            "universe": self.universe,
            "parameters": {
                "lookback_days": self.lookback_days,
                "lookback_description": f"~{self.lookback_days // 21} months of trading days",
                "top_n": self.top_n,
                "apply_abs_filter": self.apply_abs_filter,
            },
            "rebalance_frequency": "Monthly (last trading day of month)",
            "execution_assumption": "Signal at month-end close, executed at next open",
            "cash_rule": (
                "Allocates to cash when fewer than top_n assets have positive momentum"
                if self.apply_abs_filter else "Always fully invested in top_n assets"
            ),
            "reference": "Faber (2007), Journal of Wealth Management",
            "look_ahead_controls": [
                "prices.shift(1) in momentum computation",
                "weights.shift(1) before execution",
                "No survivorship bias: universe fixed at initialization",
            ],
        }

    def get_parameter_grid(self) -> Dict:
        """
        Return parameter ranges for sensitivity analysis (Layer 3 heatmap).

        Used by the backtest engine's parameter stability sweep.

        Returns:
            Dictionary of parameter names → lists of candidate values.
        """
        return {
            "lookback_days": [63, 84, 105, 126, 147, 168, 189],   # 3–9 months
            "top_n": [1, 2, 3],
        }
