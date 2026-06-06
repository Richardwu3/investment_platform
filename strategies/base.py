"""
strategies/base.py
==================
Unified strategy interface for the AI-Augmented Investment Research Platform.

Design Principles:
    - IStrategy defines the contract; implementations own the logic
    - generate_signals() is the ONLY method that touches price data
    - Weights must be forward-safe (no look-ahead): signal at t → execution at t+1
    - LLM layer reads metadata; never calls compute methods directly

Author: Yuchuan Wu
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data contract: what a strategy must return
# ---------------------------------------------------------------------------

@dataclass
class SignalResult:
    """
    Structured container for strategy output.

    Attributes:
        weights:    DataFrame[date × asset] with portfolio weights (rows sum ≤ 1.0).
                    Index must be DatetimeIndex aligned to price data.
        metadata:   Arbitrary key-value store surfaced to the AI Agent layer for
                    narrative generation (e.g. momentum scores, ranked assets).
        rebalance_dates: List of dates on which the portfolio actually changed.
        strategy_name: Human-readable label passed through to reports.
    """
    weights: pd.DataFrame
    metadata: Dict = field(default_factory=dict)
    rebalance_dates: List[pd.Timestamp] = field(default_factory=list)
    strategy_name: str = "UnnamedStrategy"

    def validate(self) -> None:
        """
        Assert basic sanity checks on generated weights.

        Raises:
            ValueError: If weights contain NaN, negative values, or row sums > 1.0
                        (allow small floating-point tolerance of 1e-6).
        """
        if self.weights.isnull().any().any():
            raise ValueError("SignalResult.weights contains NaN values.")
        if (self.weights < -1e-9).any().any():
            raise ValueError("SignalResult.weights contains negative values.")
        row_sums = self.weights.sum(axis=1)
        if (row_sums > 1.0 + 1e-6).any():
            bad_dates = row_sums[row_sums > 1.0 + 1e-6].index.tolist()
            raise ValueError(
                f"Weights sum > 1.0 on dates: {bad_dates[:5]} (showing first 5)"
            )
        logger.debug("SignalResult validation passed for '%s'.", self.strategy_name)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class IStrategy(abc.ABC):
    """
    Abstract base class for all investment strategies.

    Contract:
        Subclasses MUST implement generate_signals().
        All numerical computation lives here; the AI Agent layer is read-only
        with respect to strategy internals.

    Look-ahead Bias Prevention:
        The engine calls generate_signals() once with the FULL price history.
        Strategies are responsible for using shift(1) or equivalent to ensure
        that the signal generated on date t uses only information available
        through close of date t, and is EXECUTED at open of date t+1.

    Example::

        class MyStrategy(IStrategy):
            def generate_signals(self, prices: pd.DataFrame) -> SignalResult:
                ...
                return SignalResult(weights=weights_df, strategy_name=self.name)
    """

    def __init__(self, name: str, universe: Optional[List[str]] = None):
        """
        Initialize strategy.

        Args:
            name:     Human-readable identifier used in reports and logs.
            universe: Asset ticker list. If None, inferred from prices columns.
        """
        self.name = name
        self.universe = universe or []
        self._fitted = False
        logger.info("Strategy '%s' initialized. Universe: %s", name, universe)

    @abc.abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> SignalResult:
        """
        Compute portfolio weights from historical price data.

        THIS IS THE CORE METHOD. All subclasses must implement it.

        Args:
            prices: DataFrame with DatetimeIndex and one column per asset ticker.
                    Contains adjusted close prices. Assumed daily frequency.
                    No future data will be present — the engine slices before calling.

        Returns:
            SignalResult with weights aligned to prices.index.
            Weight at index[t] represents the SIGNAL computed at close of day t,
            to be EXECUTED at open of day t+1.
            weights.sum(axis=1) must be in [0, 1.0] (cash = 1 - sum).

        Implementation Notes:
            - Use prices.shift(1) when computing indicators to avoid look-ahead.
            - For monthly rebalancing: compute on month-end, carry forward with ffill.
            - Return zero weights for dates before strategy "warm-up" is complete.
        """
        raise NotImplementedError

    def get_metadata(self) -> Dict:
        """
        Return strategy configuration for AI narrative generation.

        Override to expose strategy parameters for the LLM explanation layer.
        This data is read-only — LLM never writes back to strategy state.

        Returns:
            Dictionary of strategy parameters and human-readable descriptions.
        """
        return {
            "strategy_name": self.name,
            "universe": self.universe,
            "description": self.__doc__ or "No description provided.",
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', universe={self.universe})"


# ---------------------------------------------------------------------------
# Utility functions shared across strategies
# ---------------------------------------------------------------------------

def compute_momentum(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    Compute total return momentum over a rolling lookback window.

    Uses prices.shift(1) to ensure the momentum score at date t reflects
    only price data through close of date t-1 (conservative, avoids same-day
    look-ahead on the signal computation date itself).

    Args:
        prices:   DataFrame of adjusted close prices (DatetimeIndex × tickers).
        lookback: Number of calendar days for momentum window.

    Returns:
        DataFrame of momentum scores (same shape as prices), NaN where
        insufficient history exists.

    Notes:
        Momentum = P(t) / P(t - lookback) - 1, computed on lagged prices.
        This is equivalent to the total return over the lookback period.
    """
    # Shift by 1 so signal at t uses data through t-1
    lagged = prices.shift(1)
    momentum = lagged / lagged.shift(lookback) - 1
    return momentum


def equal_weight_top_n(scores: pd.Series, n: int) -> pd.Series:
    """
    Assign equal weights to top-N assets by score; zero to the rest.

    Args:
        scores: Series of ranking scores (higher = better). NaN scores
                are excluded from ranking and receive zero weight.
        n:      Number of top assets to select.

    Returns:
        Series of weights summing to 1.0 (or 0.0 if all scores are NaN).
    """
    weights = pd.Series(0.0, index=scores.index)
    valid = scores.dropna()
    if valid.empty:
        return weights
    top_n = valid.nlargest(n).index
    weights[top_n] = 1.0 / n
    return weights


def is_month_end(dt_index: pd.DatetimeIndex) -> pd.Series:
    """
    Return a boolean Series marking month-end dates in the DatetimeIndex.

    Args:
        dt_index: DatetimeIndex of a price DataFrame.

    Returns:
        Boolean Series (index=dt_index) where True = last trading day of month.
    """
    # A date is month-end if the next date is in a different month
    idx = pd.Series(dt_index, index=dt_index)
    next_month = idx.shift(-1).dt.month
    return (idx.dt.month != next_month) | idx.index.isin([dt_index[-1]])
