"""
tests/test_decisions.py
========================
Unit tests for Phase 2: Decision Journal.

Run with:
    python -m pytest tests/ -v
    # or without pytest:
    python tests/test_decisions.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decisions.db import Database, DecisionRepository
from decisions.analyzer import DecisionAnalyzer
from decisions.confidence import ConfidenceCalculator


def make_test_repo() -> tuple[Database, DecisionRepository, str]:
    """Create a temp DB for testing. Returns (db, repo, tmp_path)."""
    tmp = tempfile.mktemp(suffix=".db")
    db = Database(tmp)
    repo = DecisionRepository(db)
    return db, repo, tmp


class TestDatabase(unittest.TestCase):
    """Tests for database connection and schema initialization."""

    def setUp(self):
        _, self.repo, self.tmp = make_test_repo()

    def tearDown(self):
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def test_schema_creates_tables(self):
        """All three tables should exist after initialization."""
        db = Database(self.tmp)
        with db.connect() as conn:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        self.assertIn("decisions", tables)
        self.assertIn("outcomes", tables)
        self.assertIn("rebalance_cycles", tables)

    def test_schema_idempotent(self):
        """Calling init twice should not raise."""
        db = Database(self.tmp)
        db._initialize_schema()  # Should not raise


class TestDecisionRepository(unittest.TestCase):
    """Tests for CRUD operations on decisions and outcomes."""

    def setUp(self):
        _, self.repo, self.tmp = make_test_repo()

    def tearDown(self):
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def _make_cycle(self) -> int:
        return self.repo.create_cycle("2024-01-31", "GTAA_126d_Top2")

    def test_log_approve_decision(self):
        cycle_id = self._make_cycle()
        did = self.repo.log_decision(
            date="2024-01-31",
            strategy="GTAA_126d_Top2",
            ai_signal={"SPY": 0.5, "TLT": 0.5},
            human_decision="approve",
            ai_confidence=0.75,
            cycle_id=cycle_id,
        )
        self.assertIsInstance(did, int)
        self.assertGreater(did, 0)

    def test_log_modify_decision_requires_weights(self):
        """Modify without human_weights should raise ValueError."""
        self._make_cycle()
        with self.assertRaises(ValueError):
            self.repo.log_decision(
                date="2024-01-31",
                strategy="GTAA_126d_Top2",
                ai_signal={"SPY": 0.5, "TLT": 0.5},
                human_decision="modify",
                # missing human_weights
            )

    def test_log_modify_decision_with_weights(self):
        cycle_id = self._make_cycle()
        did = self.repo.log_decision(
            date="2024-01-31",
            strategy="GTAA_126d_Top2",
            ai_signal={"SPY": 0.5, "TLT": 0.5},
            human_decision="modify",
            human_weights={"SPY": 0.7, "GLD": 0.3},
            human_reason="Rebalancing risk.",
            cycle_id=cycle_id,
        )
        self.assertGreater(did, 0)

    def test_invalid_human_decision_raises(self):
        with self.assertRaises(ValueError):
            self.repo.log_decision(
                date="2024-01-31",
                strategy="GTAA_126d_Top2",
                ai_signal={"SPY": 0.5},
                human_decision="maybe",  # invalid
            )

    def test_get_decision_returns_deserialized(self):
        cycle_id = self._make_cycle()
        did = self.repo.log_decision(
            date="2024-01-31",
            strategy="GTAA",
            ai_signal={"SPY": 0.5, "TLT": 0.5},
            human_decision="approve",
            ai_momentum_scores={"SPY": 0.08, "TLT": 0.04},
            cycle_id=cycle_id,
        )
        record = self.repo.get_decision(did)
        self.assertIsInstance(record["ai_signal"], dict)
        self.assertIsInstance(record["ai_momentum_scores"], dict)
        self.assertEqual(record["ai_signal"]["SPY"], 0.5)

    def test_get_nonexistent_decision_returns_none(self):
        result = self.repo.get_decision(99999)
        self.assertIsNone(result)

    def test_list_decisions_empty(self):
        decisions = self.repo.list_decisions()
        self.assertEqual(decisions, [])

    def test_list_decisions_filter(self):
        cycle_id = self._make_cycle()
        self.repo.log_decision("2024-01-31", "GTAA", {"SPY": 0.5}, "approve", cycle_id=cycle_id)
        cycle_id2 = self.repo.create_cycle("2024-02-28", "GTAA")
        self.repo.log_decision("2024-02-28", "GTAA", {"TLT": 0.5}, "reject",
                               human_reason="test", cycle_id=cycle_id2)
        approvals = self.repo.list_decisions(human_decision="approve")
        self.assertEqual(len(approvals), 1)
        self.assertEqual(approvals[0]["human_decision"], "approve")

    def test_log_outcome(self):
        cycle_id = self._make_cycle()
        did = self.repo.log_decision(
            date="2024-01-31",
            strategy="GTAA",
            ai_signal={"SPY": 0.5, "TLT": 0.5},
            human_decision="approve",
            cycle_id=cycle_id,
        )
        oid = self.repo.log_outcome(did, actual_return_30d=0.035, benchmark_return_30d=0.028)
        self.assertGreater(oid, 0)

    def test_outcome_ai_correct_classification(self):
        """Test that ai_correct is classified correctly from return magnitude."""
        cycle_id = self._make_cycle()
        did = self.repo.log_decision("2024-01-31", "GTAA", {"SPY": 0.5}, "approve", cycle_id=cycle_id)
        self.repo.log_outcome(did, actual_return_30d=0.04)  # positive → direction_correct

        record = self.repo.get_all_outcomes()
        outcome_record = next((r for r in record if r["id"] == did), None)
        self.assertIsNotNone(outcome_record)
        self.assertEqual(outcome_record["ai_correct"], "direction_correct")

    def test_outcome_inconclusive_classification(self):
        """Return < 0.5% should be 'inconclusive'."""
        cycle_id = self._make_cycle()
        did = self.repo.log_decision("2024-01-31", "GTAA", {"SPY": 0.5}, "approve", cycle_id=cycle_id)
        self.repo.log_outcome(did, actual_return_30d=0.002)  # < 0.5% threshold
        records = self.repo.get_all_outcomes()
        record = next((r for r in records if r["id"] == did), None)
        self.assertEqual(record["ai_correct"], "inconclusive")

    def test_duplicate_outcome_raises(self):
        cycle_id = self._make_cycle()
        did = self.repo.log_decision("2024-01-31", "GTAA", {"SPY": 0.5}, "approve", cycle_id=cycle_id)
        self.repo.log_outcome(did, actual_return_30d=0.03)
        with self.assertRaises(ValueError):
            self.repo.log_outcome(did, actual_return_30d=0.04)

    def test_update_execution_price(self):
        cycle_id = self._make_cycle()
        did = self.repo.log_decision("2024-01-31", "GTAA", {"SPY": 0.5}, "approve", cycle_id=cycle_id)
        self.repo.update_execution_price(did, {"SPY": 412.5})
        record = self.repo.get_decision(did)
        self.assertIsInstance(record["execution_price"], dict)
        self.assertEqual(record["execution_price"]["SPY"], 412.5)

    def test_pending_outcomes_returns_old_decisions(self):
        cycle_id = self._make_cycle()
        did = self.repo.log_decision("2020-01-01", "GTAA", {"SPY": 0.5}, "approve", cycle_id=cycle_id)
        pending = self.repo.get_pending_outcomes(days_threshold=30)
        ids = [r["id"] for r in pending]
        self.assertIn(did, ids)

    def test_rejected_decisions_not_in_pending(self):
        cycle_id = self._make_cycle()
        self.repo.log_decision("2020-01-01", "GTAA", {"SPY": 0.5}, "reject",
                               human_reason="test", cycle_id=cycle_id)
        pending = self.repo.get_pending_outcomes(days_threshold=30)
        self.assertEqual(len(pending), 0)


class TestDecisionAnalyzer(unittest.TestCase):
    """Tests for analytics metrics."""

    def _make_records(self):
        """Generate synthetic records for testing."""
        return [
            {"id": 1, "strategy": "GTAA", "human_decision": "approve",
             "ai_confidence": 0.8, "actual_return_30d": 0.04, "benchmark_return_30d": 0.03,
             "ai_correct": "direction_correct", "human_value_add": None},
            {"id": 2, "strategy": "GTAA", "human_decision": "modify",
             "ai_confidence": 0.65, "actual_return_30d": 0.02, "benchmark_return_30d": 0.025,
             "ai_correct": "direction_correct", "human_value_add": 0.005},
            {"id": 3, "strategy": "GTAA", "human_decision": "approve",
             "ai_confidence": 0.55, "actual_return_30d": -0.015, "benchmark_return_30d": -0.01,
             "ai_correct": "direction_wrong", "human_value_add": None},
            {"id": 4, "strategy": "GTAA", "human_decision": "reject",
             "ai_confidence": 0.40, "actual_return_30d": None, "benchmark_return_30d": None,
             "ai_correct": None, "human_value_add": None},
            {"id": 5, "strategy": "GTAA", "human_decision": "approve",
             "ai_confidence": 0.72, "actual_return_30d": 0.003, "benchmark_return_30d": 0.008,
             "ai_correct": "inconclusive", "human_value_add": None},
        ]

    def test_adoption_rates(self):
        analyzer = DecisionAnalyzer(self._make_records())
        rates = analyzer.adoption_rates()
        self.assertEqual(rates["total_decisions"], 5)
        self.assertEqual(rates["approved"], 3)
        self.assertEqual(rates["modified"], 1)
        self.assertEqual(rates["rejected"], 1)
        self.assertAlmostEqual(rates["adoption_rate"], 0.6, places=2)

    def test_ai_accuracy_excludes_inconclusive(self):
        analyzer = DecisionAnalyzer(self._make_records())
        acc = analyzer.ai_accuracy()
        # Conclusive: direction_correct (×2) + direction_wrong (×1) = 3
        self.assertEqual(acc["n_conclusive"], 3)
        self.assertEqual(acc["n_correct"], 2)
        self.assertEqual(acc["n_wrong"], 1)
        self.assertAlmostEqual(acc["ai_accuracy_rate"], 2/3, places=4)

    def test_human_value_add(self):
        analyzer = DecisionAnalyzer(self._make_records())
        hva = analyzer.human_value_add()
        # Only 1 modify record with human_value_add
        self.assertEqual(hva["n_modify_with_outcomes"], 1)
        self.assertAlmostEqual(hva["mean_human_value_add"], 0.005, places=6)

    def test_empty_records(self):
        analyzer = DecisionAnalyzer([])
        self.assertEqual(analyzer.adoption_rates()["total_decisions"], 0)
        self.assertIsNone(analyzer.ai_accuracy()["ai_accuracy_rate"])

    def test_full_report_structure(self):
        analyzer = DecisionAnalyzer(self._make_records())
        report = analyzer.full_report()
        required_keys = ["adoption_rates", "ai_accuracy", "human_value_add",
                         "confidence_calibration", "by_strategy", "pending_outcomes_count"]
        for key in required_keys:
            self.assertIn(key, report)

    def test_confidence_calibration_requires_min_data(self):
        """With < 6 outcomes, calibration should return has_confidence_data=False."""
        analyzer = DecisionAnalyzer(self._make_records())
        calib = analyzer.confidence_calibration()
        # We have 4 records with both confidence and return, but < 6
        self.assertFalse(calib["has_confidence_data"])


class TestConfidenceCalculator(unittest.TestCase):
    """Tests for confidence score computation."""

    SCORES = {"SPY": 0.089, "QQQ": 0.072, "TLT": -0.031, "GLD": 0.012, "DBC": -0.044}

    def test_momentum_spread_returns_valid_range(self):
        calc = ConfidenceCalculator(method="momentum_spread")
        result = calc.compute(self.SCORES, top_n=2)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_rank_separation_returns_valid_range(self):
        calc = ConfidenceCalculator(method="rank_separation")
        result = calc.compute(self.SCORES, top_n=2)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_selected_assets_are_top_n(self):
        calc = ConfidenceCalculator(method="momentum_spread")
        result = calc.compute(self.SCORES, top_n=2)
        self.assertEqual(set(result.selected), {"SPY", "QQQ"})

    def test_high_spread_yields_high_confidence(self):
        """When top asset dominates, confidence should be high."""
        scores = {"SPY": 0.20, "QQQ": 0.001, "TLT": -0.05, "GLD": -0.02, "DBC": -0.03}
        calc = ConfidenceCalculator(method="momentum_spread")
        result = calc.compute(scores, top_n=1)
        self.assertGreater(result.confidence, 0.7)

    def test_top_n_gte_n_assets_raises(self):
        calc = ConfidenceCalculator()
        with self.assertRaises(ValueError):
            calc.compute({"SPY": 0.1, "TLT": 0.05}, top_n=2)

    def test_absolute_strength_falls_back_without_history(self):
        """Without historical_scores, falls back to momentum_spread."""
        calc = ConfidenceCalculator(method="absolute_momentum_strength")
        result = calc.compute(self.SCORES, top_n=2)
        self.assertGreaterEqual(result.confidence, 0.0)

    def test_from_signal_result_metadata(self):
        """Test convenience constructor from Phase 1 metadata format."""
        metadata = {
            "rebalance_records": [{
                "date": "2024-01-31",
                "selected_assets": ["SPY", "QQQ"],
                "momentum_scores": {"SPY": 0.089, "QQQ": 0.072, "TLT": -0.031},
                "weights": {"SPY": 0.5, "QQQ": 0.5},
            }]
        }
        result = ConfidenceCalculator.from_signal_result(metadata)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.confidence, 0.0)

    def test_interpretation_is_string(self):
        calc = ConfidenceCalculator()
        result = calc.compute(self.SCORES, top_n=2)
        self.assertIsInstance(result.interpretation, str)
        self.assertGreater(len(result.interpretation), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
