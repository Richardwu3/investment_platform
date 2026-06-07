"""
decisions/confidence.py
=======================
Confidence score computation bridging Phase 1 signals → Phase 2 journal.

This module solves the "where does ai_confidence come from?" problem
identified in the design review. GTAA produces momentum scores, not
confidence values. This module derives a principled confidence metric
from those scores.

Three methods are provided (all deterministic — no LLM involvement):

    Method 1: momentum_spread
        confidence = (top1_score - top2_score) / abs(top1_score)
        High spread → model is "more sure" about the top pick.
        Normalized to [0, 1] via sigmoid-like mapping.

    Method 2: rank_separation
        confidence = (top_n_avg - rest_avg) / pooled_std
        Measures how cleanly the top-N separates from the rest.
        Equivalent to a one-tailed t-statistic.

    Method 3: absolute_momentum_strength
        confidence = mean(top_n_momentum_scores) / historical_vol_of_scores
        How strong the signal is relative to typical signal magnitude.
        Requires historical scores for baseline.

Usage::

    from decisions.confidence import ConfidenceCalculator
    scores = {"SPY": 0.089, "QQQ": 0.072, "TLT": -0.031, "GLD": 0.012, "DBC": -0.044}
    calc = ConfidenceCalculator(method="momentum_spread")
    result = calc.compute(scores, top_n=2)
    # result.confidence = 0.72, result.method = "momentum_spread"

Author: Yuchuan Wu — Phase 2
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

MethodType = Literal["momentum_spread", "rank_separation", "absolute_momentum_strength"]


@dataclass
class ConfidenceResult:
    """
    Output of confidence computation.

    Attributes:
        confidence:    Float in [0.0, 1.0]. 0 = no confidence, 1 = maximum confidence.
        method:        Which method was used.
        selected:      Assets selected by the strategy (top-N by momentum).
        raw_score:     Pre-normalized raw metric (for transparency/debugging).
        interpretation: Human-readable explanation of the score.
    """
    confidence: float
    method: str
    selected: List[str]
    raw_score: float
    interpretation: str


class ConfidenceCalculator:
    """
    Computes principled confidence scores from GTAA momentum signals.

    All methods are deterministic and produce values in [0.0, 1.0].
    The score is passed to DecisionRepository.log_decision() alongside
    the momentum scores for full auditability.
    """

    def __init__(
        self,
        method: MethodType = "momentum_spread",
        historical_scores: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        """
        Args:
            method:            Which scoring method to use.
            historical_scores: Required for 'absolute_momentum_strength' only.
                               List of past momentum score dicts.
        """
        self.method = method
        self.historical_scores = historical_scores or []
        logger.debug("ConfidenceCalculator initialized: method=%s", method)

    def compute(
        self,
        momentum_scores: Dict[str, float],
        top_n: int = 2,
    ) -> ConfidenceResult:
        """
        Compute confidence from a single period's momentum scores.

        Args:
            momentum_scores: Dict {ticker: momentum_value} for all universe assets.
            top_n:           Number of top assets selected by the strategy.

        Returns:
            ConfidenceResult with confidence score and metadata.

        Raises:
            ValueError: If momentum_scores is empty or top_n >= len(scores).
        """
        if not momentum_scores:
            raise ValueError("momentum_scores cannot be empty.")
        if top_n >= len(momentum_scores):
            raise ValueError(
                f"top_n ({top_n}) must be less than number of assets ({len(momentum_scores)})."
            )

        # Sort by score descending; filter NaN
        valid = {k: v for k, v in momentum_scores.items() if v == v}  # NaN check
        ranked = sorted(valid.items(), key=lambda x: x[1], reverse=True)
        selected = [ticker for ticker, _ in ranked[:top_n]]

        dispatch = {
            "momentum_spread":             self._momentum_spread,
            "rank_separation":             self._rank_separation,
            "absolute_momentum_strength":  self._absolute_momentum_strength,
        }

        handler = dispatch.get(self.method)
        if handler is None:
            raise ValueError(f"Unknown method: '{self.method}'. Choose from {list(dispatch.keys())}")

        confidence, raw_score = handler(ranked, top_n)
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

        interpretation = self._interpret(confidence)
        result = ConfidenceResult(
            confidence=round(confidence, 4),
            method=self.method,
            selected=selected,
            raw_score=round(raw_score, 6),
            interpretation=interpretation,
        )
        logger.debug(
            "Confidence: %.3f (method=%s, raw=%.4f, selected=%s)",
            confidence, self.method, raw_score, selected
        )
        return result

    # ------------------------------------------------------------------
    # Method implementations
    # ------------------------------------------------------------------

    def _momentum_spread(
        self, ranked: List[tuple], top_n: int
    ) -> tuple[float, float]:
        """
        Method 1: Spread between top-1 and top-(top_n+1) momentum scores.

        Intuition: If top pick has much higher momentum than the next
        alternative, the model is more "sure" about the selection.

        Raw score: (top1 - next_out) / (|top1| + |next_out| + epsilon)
        Maps to [0, 1] via sigmoid(5 * raw_score).
        """
        top1_score = ranked[0][1]
        next_out_score = ranked[top_n][1] if len(ranked) > top_n else 0.0

        epsilon = 1e-8
        raw_spread = (top1_score - next_out_score) / (
            abs(top1_score) + abs(next_out_score) + epsilon
        )
        # Sigmoid mapping: raw_spread ∈ [-1, 1] → confidence ∈ [0, 1]
        confidence = 1.0 / (1.0 + math.exp(-5.0 * raw_spread))
        return confidence, raw_spread

    def _rank_separation(
        self, ranked: List[tuple], top_n: int
    ) -> tuple[float, float]:
        """
        Method 2: Standardized separation between top-N and the rest.

        Intuition: If top-N assets have consistently higher scores than
        the remaining assets with low within-group variance, the signal
        is cleaner.

        Raw score: (mean_top - mean_rest) / pooled_std
        Maps to [0, 1] via tanh(raw_score / 2) * 0.5 + 0.5.
        """
        top_scores = [s for _, s in ranked[:top_n]]
        rest_scores = [s for _, s in ranked[top_n:]]

        mean_top  = sum(top_scores) / len(top_scores)
        mean_rest = sum(rest_scores) / len(rest_scores) if rest_scores else 0.0

        all_scores = top_scores + rest_scores
        var = sum((x - (sum(all_scores) / len(all_scores))) ** 2
                  for x in all_scores) / len(all_scores)
        pooled_std = max(math.sqrt(var), 1e-8)

        raw_separation = (mean_top - mean_rest) / pooled_std
        confidence = math.tanh(raw_separation / 2.0) * 0.5 + 0.5
        return confidence, raw_separation

    def _absolute_momentum_strength(
        self, ranked: List[tuple], top_n: int
    ) -> tuple[float, float]:
        """
        Method 3: Selected assets' momentum vs. historical baseline.

        Intuition: A momentum score of 8% in a volatile market means less
        than 8% in a calm market. Normalize by historical signal dispersion.

        Requires historical_scores to be populated at initialization.
        Falls back to momentum_spread if no history available.
        """
        if not self.historical_scores:
            logger.warning(
                "absolute_momentum_strength requires historical_scores; "
                "falling back to momentum_spread."
            )
            return self._momentum_spread(ranked, top_n)

        top_scores = [s for _, s in ranked[:top_n]]
        mean_top_current = sum(top_scores) / len(top_scores)

        # Compute historical distribution of top-N mean scores
        historical_means = []
        for hist_period in self.historical_scores:
            hist_ranked = sorted(hist_period.values(), reverse=True)
            if len(hist_ranked) >= top_n:
                historical_means.append(sum(hist_ranked[:top_n]) / top_n)

        if not historical_means:
            return self._momentum_spread(ranked, top_n)

        hist_mean = sum(historical_means) / len(historical_means)
        hist_var = sum((x - hist_mean) ** 2 for x in historical_means) / len(historical_means)
        hist_std = max(math.sqrt(hist_var), 1e-8)

        # z-score of current signal strength
        z_score = (mean_top_current - hist_mean) / hist_std
        confidence = 1.0 / (1.0 + math.exp(-z_score))  # sigmoid
        return confidence, z_score

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _interpret(confidence: float) -> str:
        """
        Return a human-readable interpretation of a confidence score.

        Args:
            confidence: Float in [0, 1].

        Returns:
            Interpretation string for the AI Agent / CLI display.
        """
        if confidence >= 0.80:
            return "High confidence — strong momentum separation, clear signal."
        elif confidence >= 0.65:
            return "Moderate-high confidence — signal present, minor ambiguity."
        elif confidence >= 0.50:
            return "Moderate confidence — consider additional validation."
        elif confidence >= 0.35:
            return "Low confidence — weak signal, higher risk of false positive."
        else:
            return "Very low confidence — momentum nearly indistinguishable, consider cash."

    @classmethod
    def from_signal_result(
        cls,
        signal_metadata: dict,
        method: MethodType = "momentum_spread",
    ) -> Optional["ConfidenceResult"]:
        """
        Convenience constructor: compute confidence from a Phase 1 SignalResult.metadata dict.

        Args:
            signal_metadata: The 'metadata' dict from SignalResult (from gtaa.generate_signals).
            method:          Confidence method to use.

        Returns:
            ConfidenceResult, or None if metadata lacks momentum scores.
        """
        records = signal_metadata.get("rebalance_records", [])
        if not records:
            logger.warning("No rebalance_records found in signal metadata.")
            return None

        latest = records[-1]
        scores = latest.get("momentum_scores")
        if not scores:
            logger.warning("No momentum_scores in latest rebalance record.")
            return None

        top_n = len(latest.get("selected_assets", []))
        if top_n == 0:
            logger.warning("No selected_assets in latest rebalance record.")
            return None

        calc = cls(method=method)
        return calc.compute(scores, top_n=top_n)
