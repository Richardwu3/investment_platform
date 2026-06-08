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

New in Phase 2.2:
    calculate_human_value_add() — single-decision value-add from outcome fields.
                                  Deterministic, used by review_copilot.
    per_decision_value_add()    — time-series of value-add across all decisions.
                                  Returns list[dict] for charting and trend analysis.

Metrics Defined:
    adoption_rate            : % of AI recommendations fully approved
    modification_rate        : % of decisions where human changed weights
    rejection_rate           : % of decisions rejected outright
    ai_accuracy              : % correct direction calls (excludes inconclusive)
    human_value_add (agg)    : mean(actual - ai_only) across modify decisions
    confidence_calibration   : whether higher ai_confidence predicts better outcomes

Author: Yuchuan Wu — Phase 2
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas not available; to_dataframe() will return None.")


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

    INCONCLUSIVE_THRESHOLD: float = 0.005

    def __init__(self, records: List[Dict]) -> None:
        """
        Args:
            records: List of merged decision+outcome dicts from
                     DecisionRepository.get_all_outcomes(). Decisions without
                     outcomes are included but excluded from accuracy metrics.
        """
        self.records = records
        self.total = len(records)
        self._with_outcomes = [
            r for r in records if r.get("actual_return_30d") is not None
        ]
        logger.debug(
            "Analyzer initialized: %d total decisions, %d with outcomes.",
            self.total, len(self._with_outcomes),
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
            d = r.get("human_decision", "")
            if d in counts:
                counts[d] += 1

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
                n_with_outcomes      : int
                n_conclusive         : int
                n_correct            : int
                n_wrong              : int
                n_inconclusive       : int
                ai_accuracy_rate     : float | None
                mean_return_correct  : float | None
                mean_return_wrong    : float | None
        """
        wo = self._with_outcomes
        if not wo:
            return {
                "n_with_outcomes": 0, "n_conclusive": 0, "n_correct": 0,
                "n_wrong": 0, "n_inconclusive": 0, "ai_accuracy_rate": None,
                "mean_return_correct": None, "mean_return_wrong": None,
            }

        correct    = [r for r in wo if r.get("ai_correct") == "direction_correct"]
        wrong      = [r for r in wo if r.get("ai_correct") == "direction_wrong"]
        incon      = [r for r in wo if r.get("ai_correct") == "inconclusive"]
        conclusive = correct + wrong
        accuracy   = len(correct) / len(conclusive) if conclusive else None

        mean_correct = np.mean([r["actual_return_30d"] for r in correct]) if correct else None
        mean_wrong   = np.mean([r["actual_return_30d"] for r in wrong])   if wrong   else None

        return {
            "n_with_outcomes": len(wo),
            "n_conclusive": len(conclusive),
            "n_correct": len(correct),
            "n_wrong": len(wrong),
            "n_inconclusive": len(incon),
            "ai_accuracy_rate": round(accuracy, 4) if accuracy is not None else None,
            "mean_return_correct": round(mean_correct, 6) if mean_correct is not None else None,
            "mean_return_wrong": round(mean_wrong, 6) if mean_wrong is not None else None,
        }

    # ------------------------------------------------------------------
    # Human Value-Add  (aggregate statistics)
    # ------------------------------------------------------------------

    def human_value_add(self) -> Dict:
        """
        Measure whether human modifications improved outcomes vs. pure AI signals.

        Aggregated across all 'modify' decisions that have outcomes AND
        ai_only_return_30d recorded. Single-decision calculation is in
        calculate_human_value_add() below.

        Returns:
            Dict with keys:
                n_modify_with_outcomes    : int
                mean_human_value_add      : float | None  (mean across modify decisions)
                pct_modifications_helpful : float | None  (fraction > 0)
                total_human_alpha         : float | None  (sum of value adds)
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
            "mean_human_value_add": round(float(np.mean(value_adds)), 6),
            "pct_modifications_helpful": round(
                sum(v > 0 for v in value_adds) / len(value_adds), 4
            ),
            "total_human_alpha": round(sum(value_adds), 6),
        }

    # ------------------------------------------------------------------
    # Human Value-Add  (single-decision, used by review_copilot)
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_human_value_add(outcome: Dict) -> Optional[float]:
        """
        Compute the human value-add for a single decision's outcome record.

        Human value-add = actual_return_30d - ai_only_return_30d

        This is the per-decision version of the aggregate human_value_add() metric.
        Positive = human modification improved the outcome vs. pure AI signal.
        Negative = human modification hurt the outcome.
        None     = ai_only_return_30d was not recorded (counterfactual unavailable).

        Used by review_copilot.py to populate the human-vs-AI comparison section
        of the Markdown report without needing the full records list.

        Args:
            outcome: A single outcome dict (from DecisionRepository or full_trace).
                     Must contain 'actual_return_30d' and optionally 'ai_only_return_30d'.

        Returns:
            Float difference in returns, or None if counterfactual is unavailable.

        Example::

            outcome = {"actual_return_30d": 0.028, "ai_only_return_30d": 0.021}
            val = DecisionAnalyzer.calculate_human_value_add(outcome)
            # val == 0.007  (+0.7%, human added value)
        """
        actual = outcome.get("actual_return_30d")
        ai_only = outcome.get("ai_only_return_30d")
        if actual is None or ai_only is None:
            return None
        return round(actual - ai_only, 6)

    # ------------------------------------------------------------------
    # Per-decision value-add time series
    # ------------------------------------------------------------------

    def per_decision_value_add(self) -> List[Dict]:
        """
        Return a time-series of human value-add, one entry per modify decision
        that has both an outcome and a recorded ai_only_return_30d.

        Suitable for trend charts and month-by-month attribution analysis.
        More useful than the aggregate scalar for AI PM portfolio storytelling.

        Returns:
            List of dicts ordered by date ascending:
                {
                    decision_id   : int,
                    date          : str (YYYY-MM-DD),
                    strategy      : str,
                    actual_return : float,
                    ai_return     : float,
                    value_add     : float,
                    helpful       : bool,   — True if value_add > 0
                }

        Example::

            series = analyzer.per_decision_value_add()
            total_alpha = sum(r["value_add"] for r in series)
        """
        result = []
        for r in self.records:
            if r.get("human_decision") != "modify":
                continue
            actual = r.get("actual_return_30d")
            ai_only = r.get("human_value_add")  # stored as (actual - ai_only)
            # We need to reconstruct ai_only from (human_value_add = actual - ai_only)
            # But we can also call calculate_human_value_add if the field is present
            va = r.get("human_value_add")
            if actual is None or va is None:
                continue
            ai_only_reconstructed = actual - va
            result.append({
                "decision_id": r.get("id"),
                "date": r.get("date"),
                "strategy": r.get("strategy"),
                "actual_return": round(actual, 6),
                "ai_return": round(ai_only_reconstructed, 6),
                "value_add": round(va, 6),
                "helpful": va > 0,
            })

        result.sort(key=lambda x: x["date"] or "")
        return result

    # ------------------------------------------------------------------
    # Confidence Calibration
    # ------------------------------------------------------------------

    def confidence_calibration(self) -> Dict:
        """
        Test whether higher AI confidence predicts better outcomes.

        Splits decisions into terciles by ai_confidence and computes accuracy
        within each tercile. Also reports Pearson correlation between
        ai_confidence and actual_return_30d.

        Returns:
            Dict with keys:
                has_confidence_data : bool
                tercile_analysis    : list[dict]
                correlation         : float | None
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
        corr = float(np.corrcoef(confidences, returns)[0, 1])

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
            conclusive = [r for r in subset if r.get("ai_correct") in
                          ("direction_correct", "direction_wrong")]
            correct = [r for r in conclusive if r.get("ai_correct") == "direction_correct"]
            accuracy = len(correct) / len(conclusive) if conclusive else None
            terciles.append({
                "label": label,
                "confidence_range": (round(lo, 3), round(hi, 3)),
                "n": len(subset),
                "accuracy": round(accuracy, 4) if accuracy is not None else None,
                "mean_return": round(float(np.mean([r["actual_return_30d"] for r in subset])), 6),
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
    # Strategy breakdown
    # ------------------------------------------------------------------

    def by_strategy(self) -> Dict[str, Dict]:
        """
        Break down key metrics by strategy name.

        Returns:
            Dict of {strategy_name: {n_decisions, adoption_rates, ai_accuracy}}.
        """
        strategies: Dict[str, List] = {}
        for r in self.records:
            strat = r.get("strategy", "Unknown")
            strategies.setdefault(strat, []).append(r)

        result = {}
        for strat, rows in strategies.items():
            sub = DecisionAnalyzer(rows)
            result[strat] = {
                "n_decisions": len(rows),
                "adoption_rates": sub.adoption_rates(),
                "ai_accuracy": sub.ai_accuracy(),
            }
        return result

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(self) -> Dict:
        """
        Compute and return all analytics metrics as a single structured dict.

        Consumed by:
            1. CLI `analyze` command (prints to terminal)
            2. AI Agent layer (narrative generation context)

        Returns:
            Dict with keys:
                adoption_rates, ai_accuracy, human_value_add,
                per_decision_value_add, confidence_calibration,
                by_strategy, pending_outcomes_count, generated_at
        """
        from datetime import datetime

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
            "per_decision_value_add": self.per_decision_value_add(),
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

    def to_dataframe(self):
        """
        Export all records to a pandas DataFrame for further analysis.

        Returns:
            pd.DataFrame if pandas is available, else None.
        """
        if not HAS_PANDAS:
            logger.warning("pandas not installed; cannot export DataFrame.")
            return None

        rows = [{
            "id": r.get("id"),
            "date": r.get("date"),
            "strategy": r.get("strategy"),
            "ai_confidence": r.get("ai_confidence"),
            "human_decision": r.get("human_decision"),
            "actual_return_30d": r.get("actual_return_30d"),
            "benchmark_return_30d": r.get("benchmark_return_30d"),
            "ai_correct": r.get("ai_correct"),
            "human_value_add": r.get("human_value_add"),
        } for r in self.records]

        df = pd.DataFrame(rows)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
