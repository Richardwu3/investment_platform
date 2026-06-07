"""
decisions/analyzer.py
=====================
Analytics engine for the Decision Journal.

Computes adoption rates, AI accuracy metrics, human value-add statistics,
and regime-conditional performance. This module is the "Feedback Loop" layer
of the 7-layer platform architecture — it closes the loop between AI output
and decision quality over time.

Design: Code = Truth
    All numbers computed here in Python/pandas.
    Results dict passed to LLM for narrative generation (not computed by LLM).

Metrics Defined:
    adoption_rate       : % of AI recommendations fully approved by human
    modification_rate   : % of decisions where human changed weights
    rejection_rate      : % of decisions rejected outright
    ai_accuracy         : % of decisions where ai_correct == 'direction_correct'
                          (excludes 'inconclusive' from denominator)
    human_value_add     : mean(actual_return - ai_only_return) across modify decisions
                          Positive = human modifications improved outcomes
    confidence_calibration: whether higher ai_confidence predicts better outcomes

Author: Yuchuan Wu — Phase 2
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional pandas — degrade gracefully for minimal installs
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas not available; some analytics features will be limited.")


class DecisionAnalyzer:
    """
    Computes all analytics metrics from a list of decision+outcome records.

    All inputs are plain Python dicts (from DecisionRepository.get_all_outcomes()).
    No database calls inside this class — separation of concerns.

    Usage::

        repo = DecisionRepository(db)
        rows = repo.get_all_outcomes()
        analyzer = DecisionAnalyzer(rows)
        report = analyzer.full_report()
    """

    # Below this absolute return, we call the signal 'inconclusive'
    INCONCLUSIVE_THRESHOLD: float = 0.005

    def __init__(self, records: List[Dict]) -> None:
        """
        Args:
            records: List of merged decision+outcome dicts from DecisionRepository.
                     Decisions without outcomes are included but excluded from
                     accuracy metrics.
        """
        self.records = records
        self.total = len(records)
        self._with_outcomes = [r for r in records if r.get("actual_return_30d") is not None]
        logger.debug(
            "Analyzer initialized: %d total decisions, %d with outcomes.",
            self.total, len(self._with_outcomes)
        )

    # ------------------------------------------------------------------
    # Adoption / Decision Type Breakdown
    # ------------------------------------------------------------------

    def adoption_rates(self) -> Dict:
        """
        Compute breakdown of human decision types.

        Returns:
            Dict with keys:
                total_decisions    : int
                approved           : int
                modified           : int
                rejected           : int
                adoption_rate      : float [0,1] — fraction approved
                modification_rate  : float [0,1] — fraction modified
                rejection_rate     : float [0,1] — fraction rejected
        """
        if self.total == 0:
            return {
                "total_decisions": 0,
                "approved": 0, "modified": 0, "rejected": 0,
                "adoption_rate": None, "modification_rate": None, "rejection_rate": None,
            }

        counts = {"approve": 0, "modify": 0, "reject": 0}
        for r in self.records:
            decision = r.get("human_decision", "")
            if decision in counts:
                counts[decision] += 1

        return {
            "total_decisions": self.total,
            "approved": counts["approve"],
            "modified": counts["modify"],
            "rejected": counts["reject"],
            "adoption_rate": round(counts["approve"] / self.total, 4),
            "modification_rate": round(counts["modify"] / self.total, 4),
            "rejection_rate": round(counts["reject"] / self.total, 4),
        }

    # ------------------------------------------------------------------
    # AI Accuracy
    # ------------------------------------------------------------------

    def ai_accuracy(self) -> Dict:
        """
        Compute AI directional accuracy from recorded outcomes.

        Definition:
            Numerator   : decisions where ai_correct == 'direction_correct'
            Denominator : all decisions with outcomes EXCLUDING 'inconclusive'
            Rationale   : inconclusive = return too small to attribute to signal

        Returns:
            Dict with keys:
                n_with_outcomes      : int — outcomes recorded so far
                n_conclusive         : int — excludes inconclusive
                n_correct            : int
                n_wrong              : int
                n_inconclusive       : int
                ai_accuracy_rate     : float | None — None if no conclusive outcomes
                mean_return_correct  : float — avg return on correct calls
                mean_return_wrong    : float — avg return on wrong calls
        """
        with_outcomes = self._with_outcomes
        if not with_outcomes:
            return {
                "n_with_outcomes": 0, "n_conclusive": 0, "n_correct": 0,
                "n_wrong": 0, "n_inconclusive": 0, "ai_accuracy_rate": None,
                "mean_return_correct": None, "mean_return_wrong": None,
            }

        correct = [r for r in with_outcomes if r.get("ai_correct") == "direction_correct"]
        wrong   = [r for r in with_outcomes if r.get("ai_correct") == "direction_wrong"]
        incon   = [r for r in with_outcomes if r.get("ai_correct") == "inconclusive"]
        conclusive = correct + wrong

        accuracy = len(correct) / len(conclusive) if conclusive else None

        mean_ret_correct = (
            np.mean([r["actual_return_30d"] for r in correct]) if correct else None
        )
        mean_ret_wrong = (
            np.mean([r["actual_return_30d"] for r in wrong]) if wrong else None
        )

        return {
            "n_with_outcomes": len(with_outcomes),
            "n_conclusive": len(conclusive),
            "n_correct": len(correct),
            "n_wrong": len(wrong),
            "n_inconclusive": len(incon),
            "ai_accuracy_rate": round(accuracy, 4) if accuracy is not None else None,
            "mean_return_correct": round(mean_ret_correct, 6) if mean_ret_correct is not None else None,
            "mean_return_wrong": round(mean_ret_wrong, 6) if mean_ret_wrong is not None else None,
        }

    # ------------------------------------------------------------------
    # Human Value-Add
    # ------------------------------------------------------------------

    def human_value_add(self) -> Dict:
        """
        Measure whether human modifications improved outcomes vs. pure AI signals.

        human_value_add = actual_return (with human mods) - ai_only_return (pure AI)
        Positive = human added value; Negative = human detracted value.

        Only computed for decisions where human_decision == 'modify' AND
        outcomes have been recorded AND ai_only_return_30d is available.

        Returns:
            Dict with keys:
                n_modify_with_outcomes : int
                mean_human_value_add   : float | None
                pct_modifications_helpful : float | None — fraction with positive value add
                total_human_alpha      : float | None — sum of value adds
        """
        modify_outcomes = [
            r for r in self._with_outcomes
            if r.get("human_decision") == "modify"
            and r.get("human_value_add") is not None
        ]

        if not modify_outcomes:
            return {
                "n_modify_with_outcomes": 0,
                "mean_human_value_add": None,
                "pct_modifications_helpful": None,
                "total_human_alpha": None,
            }

        value_adds = [r["human_value_add"] for r in modify_outcomes]
        return {
            "n_modify_with_outcomes": len(modify_outcomes),
            "mean_human_value_add": round(np.mean(value_adds), 6),
            "pct_modifications_helpful": round(
                sum(v > 0 for v in value_adds) / len(value_adds), 4
            ),
            "total_human_alpha": round(sum(value_adds), 6),
        }

    # ------------------------------------------------------------------
    # Confidence Calibration
    # ------------------------------------------------------------------

    def confidence_calibration(self) -> Dict:
        """
        Test whether higher AI confidence predicts better outcomes.

        Method:
            Split decisions into terciles by ai_confidence.
            Compute AI accuracy rate within each tercile.
            Well-calibrated model: accuracy increases monotonically with confidence.

        Returns:
            Dict with keys:
                has_confidence_data : bool
                tercile_analysis    : list of dicts [{confidence_range, accuracy, n}]
                correlation         : float | None — Pearson r between confidence and return
                interpretation      : str
        """
        scored = [
            r for r in self._with_outcomes
            if r.get("ai_confidence") is not None
            and r.get("actual_return_30d") is not None
        ]

        if len(scored) < 6:
            return {
                "has_confidence_data": False,
                "tercile_analysis": [],
                "correlation": None,
                "interpretation": "Insufficient data (need ≥ 6 outcomes with confidence scores).",
            }

        confidences = [r["ai_confidence"] for r in scored]
        returns = [r["actual_return_30d"] for r in scored]

        # Pearson correlation between confidence and realized return
        corr = float(np.corrcoef(confidences, returns)[0, 1])

        # Tercile split
        thresholds = np.percentile(confidences, [33, 67])
        terciles = []
        labels = ["Low (bottom 33%)", "Mid (33–67%)", "High (top 33%)"]
        bounds = [
            (0.0, thresholds[0]),
            (thresholds[0], thresholds[1]),
            (thresholds[1], 1.0),
        ]
        for label, (lo, hi) in zip(labels, bounds):
            subset = [r for r in scored if lo <= r["ai_confidence"] <= hi]
            conclusive = [
                r for r in subset if r.get("ai_correct") in ("direction_correct", "direction_wrong")
            ]
            correct = [r for r in conclusive if r.get("ai_correct") == "direction_correct"]
            accuracy = len(correct) / len(conclusive) if conclusive else None
            terciles.append({
                "label": label,
                "confidence_range": (round(lo, 3), round(hi, 3)),
                "n": len(subset),
                "accuracy": round(accuracy, 4) if accuracy is not None else None,
                "mean_return": round(np.mean([r["actual_return_30d"] for r in subset]), 6),
            })

        if corr > 0.3:
            interpretation = "Confidence is positively correlated with returns — model is well-calibrated."
        elif corr > 0:
            interpretation = "Weak positive correlation — confidence has limited predictive power."
        elif corr > -0.3:
            interpretation = "Near-zero or negative correlation — confidence scores may be poorly calibrated."
        else:
            interpretation = "Negative correlation — higher confidence predicts worse outcomes. Investigate."

        return {
            "has_confidence_data": True,
            "tercile_analysis": terciles,
            "correlation": round(corr, 4),
            "interpretation": interpretation,
        }

    # ------------------------------------------------------------------
    # Strategy-level breakdown
    # ------------------------------------------------------------------

    def by_strategy(self) -> Dict[str, Dict]:
        """
        Break down key metrics by strategy name.

        Useful when multiple strategies share the same journal.

        Returns:
            Dict of {strategy_name: {adoption_rate, accuracy, n_decisions}}.
        """
        strategies: Dict[str, List] = {}
        for r in self.records:
            strat = r.get("strategy", "Unknown")
            strategies.setdefault(strat, []).append(r)

        result = {}
        for strat, rows in strategies.items():
            sub_analyzer = DecisionAnalyzer(rows)
            result[strat] = {
                "n_decisions": len(rows),
                "adoption_rates": sub_analyzer.adoption_rates(),
                "ai_accuracy": sub_analyzer.ai_accuracy(),
            }
        return result

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(self) -> Dict:
        """
        Compute and return all analytics metrics as a single structured dict.

        This is the primary output consumed by:
            1. The CLI `analyze` command (prints to terminal)
            2. The AI Agent layer (for narrative generation)

        Returns:
            Dict with keys:
                adoption_rates, ai_accuracy, human_value_add,
                confidence_calibration, by_strategy,
                pending_outcomes_count, generated_at
        """
        from datetime import datetime

        # Count pending outcomes (decisions old enough but not yet recorded)
        pending = sum(
            1 for r in self.records
            if r.get("actual_return_30d") is None
            and r.get("human_decision") != "reject"
        )

        report = {
            "generated_at": datetime.now().isoformat(),
            "adoption_rates": self.adoption_rates(),
            "ai_accuracy": self.ai_accuracy(),
            "human_value_add": self.human_value_add(),
            "confidence_calibration": self.confidence_calibration(),
            "by_strategy": self.by_strategy(),
            "pending_outcomes_count": pending,
        }

        logger.info(
            "Full report generated | %d decisions | adoption=%.0f%% | accuracy=%s",
            self.total,
            (report["adoption_rates"]["adoption_rate"] or 0) * 100,
            report["ai_accuracy"]["ai_accuracy_rate"],
        )
        return report

    # ------------------------------------------------------------------
    # DataFrame export (requires pandas)
    # ------------------------------------------------------------------

    def to_dataframe(self):  # -> Optional[pd.DataFrame]
        """
        Export all records to a pandas DataFrame for further analysis.

        Returns:
            pd.DataFrame if pandas is available, else None.
        """
        if not HAS_PANDAS:
            logger.warning("pandas not installed; cannot export DataFrame.")
            return None

        rows = []
        for r in self.records:
            row = {
                "id": r.get("id"),
                "date": r.get("date"),
                "strategy": r.get("strategy"),
                "ai_confidence": r.get("ai_confidence"),
                "human_decision": r.get("human_decision"),
                "actual_return_30d": r.get("actual_return_30d"),
                "benchmark_return_30d": r.get("benchmark_return_30d"),
                "ai_correct": r.get("ai_correct"),
                "human_value_add": r.get("human_value_add"),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
