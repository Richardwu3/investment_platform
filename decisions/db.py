"""
decisions/db.py
===============
Database layer for the Decision Journal.

Manages SQLite schema creation, connection lifecycle, and all
data access operations. This module is the ONLY place that touches
the database — CLI and analyzer import from here, never write SQL directly.

Schema Design Rationale:
    decisions table  : Captures the state at decision time (what AI recommended,
                       what human decided, why). Immutable after creation except
                       for execution_price which may be filled post-approval.
    outcomes table   : Captures results 30 days later. Separate because the data
                       arrives at a different point in time — filling it at
                       decision time would require a time machine.
    rebalance_cycles : Groups multiple decisions made in the same monthly cycle,
                       enabling cycle-level analysis (not just decision-level).

ai_correct Definition (fixes the schema ambiguity in original spec):
    We define "AI correct" as a 3-way enum, not a boolean:
        'direction_correct'  — AI picked an asset that had positive return
        'direction_wrong'    — AI picked an asset that had negative return
        'inconclusive'       — Return too small to distinguish signal from noise (<0.5%)
    This is stored as TEXT in SQLite. The analyzer maps it to numeric scores.

Author: Yuchuan Wu — Phase 2
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
-- ──────────────────────────────────────────────────────────────────────────
-- rebalance_cycles: one row per monthly rebalance event
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rebalance_cycles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_date      TEXT    NOT NULL,          -- YYYY-MM-DD, month-end signal date
    strategy        TEXT    NOT NULL,          -- e.g. 'GTAA_126d_Top2'
    market_regime   TEXT,                      -- 'bull' | 'bear' | 'sideways' (from Layer 4)
    notes           TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────────────────────────────
-- decisions: one row per asset position in a rebalance cycle
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS decisions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Linkage
    cycle_id                INTEGER REFERENCES rebalance_cycles(id),
    date                    TEXT    NOT NULL,   -- YYYY-MM-DD signal date

    -- Strategy identification
    strategy                TEXT    NOT NULL,

    -- AI recommendation (what the model said)
    ai_signal               TEXT    NOT NULL,   -- JSON: {"SPY": 0.5, "TLT": 0.5}
    ai_confidence           REAL,              -- 0.0–1.0, derived from momentum spread
    ai_confidence_method    TEXT,              -- how confidence was computed
    ai_momentum_scores      TEXT,              -- JSON: {"SPY": 0.08, "QQQ": 0.04, ...}
    ai_selected_assets      TEXT,              -- JSON array: ["SPY", "TLT"]

    -- Human decision
    human_decision          TEXT    NOT NULL   -- 'approve' | 'modify' | 'reject'
                            CHECK (human_decision IN ('approve', 'modify', 'reject')),
    human_weights           TEXT,              -- JSON: actual weights after human review
                                               -- NULL if approved (use ai_signal) or rejected
    human_reason            TEXT,              -- free text: why modified or rejected

    -- Execution (filled after order placed, may be NULL at decision time)
    execution_price         TEXT,              -- JSON: {"SPY": 412.5, "TLT": 98.2}
    executed_at             TEXT,              -- ISO datetime of execution

    -- Metadata
    created_at              TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at              TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────────────────────────────
-- outcomes: one row per decision, filled ~30 days later
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS outcomes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id         INTEGER NOT NULL UNIQUE REFERENCES decisions(id),

    -- Realized performance
    actual_return_30d   REAL,              -- portfolio return 30 calendar days after decision
    benchmark_return_30d REAL,            -- SPY return over same window (for excess return)
    asset_returns       TEXT,             -- JSON: {"SPY": 0.03, "TLT": -0.01, ...}

    -- AI accuracy assessment (3-way enum, not boolean — see module docstring)
    ai_correct          TEXT              -- 'direction_correct' | 'direction_wrong' | 'inconclusive'
                        CHECK (ai_correct IN ('direction_correct', 'direction_wrong',
                                              'inconclusive', NULL)),

    -- Counterfactual: what would have happened with pure AI weights
    ai_only_return_30d  REAL,            -- return using ai_signal weights (ignoring human mod)
    human_value_add     REAL,            -- actual_return_30d - ai_only_return_30d

    -- Metadata
    outcome_date        TEXT,            -- date the outcome was recorded
    notes               TEXT,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────────────────────────────
-- Indexes for common query patterns
-- ──────────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_decisions_date     ON decisions(date);
CREATE INDEX IF NOT EXISTS idx_decisions_strategy ON decisions(strategy);
CREATE INDEX IF NOT EXISTS idx_outcomes_decision  ON outcomes(decision_id);
"""


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------

class Database:
    """
    SQLite database manager for the Decision Journal.

    Handles connection lifecycle, schema initialization, and provides
    a context-managed connection for safe transaction handling.

    All methods enforce:
        - Parameterized queries (no string formatting with user input)
        - Explicit commit/rollback
        - Structured logging on every write operation

    Usage::

        db = Database("decisions/journal.db")
        with db.connect() as conn:
            conn.execute("SELECT * FROM decisions")
    """

    def __init__(self, db_path: str = "decisions/journal.db") -> None:
        """
        Initialize database, creating file and schema if needed.

        Args:
            db_path: Path to SQLite file. Parent directory created if absent.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()
        logger.info("Database ready at: %s", self.db_path.resolve())

    def _initialize_schema(self) -> None:
        """
        Create tables and indexes if they don't exist.

        Idempotent — safe to call on every startup.
        Uses IF NOT EXISTS throughout.
        """
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
        logger.debug("Schema initialized.")

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager yielding an open, configured SQLite connection.

        Commits on clean exit, rolls back on exception.
        Sets row_factory to sqlite3.Row for dict-like access.

        Yields:
            sqlite3.Connection with row_factory set.

        Raises:
            sqlite3.Error: Re-raised after rollback on any database error.

        Example::

            with db.connect() as conn:
                rows = conn.execute("SELECT * FROM decisions").fetchall()
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # Better concurrent read performance
        conn.execute("PRAGMA foreign_keys=ON")    # Enforce FK constraints
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            logger.error("Database error, rolled back: %s", e)
            raise
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Data Access Objects
# ---------------------------------------------------------------------------

class DecisionRepository:
    """
    All read/write operations for the decisions and outcomes tables.

    This is the single source of truth for database interactions.
    CLI and Analyzer call methods here; they never write raw SQL.
    """

    def __init__(self, db: Database) -> None:
        """
        Args:
            db: Initialized Database instance.
        """
        self.db = db

    # ------------------------------------------------------------------
    # Rebalance cycle operations
    # ------------------------------------------------------------------

    def create_cycle(
        self,
        cycle_date: str,
        strategy: str,
        market_regime: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        """
        Create a new rebalance cycle record.

        Args:
            cycle_date:    Month-end signal date (YYYY-MM-DD).
            strategy:      Strategy name (e.g. 'GTAA_126d_Top2').
            market_regime: Optional regime label from Layer 4.
            notes:         Free-text notes.

        Returns:
            Integer ID of the newly created cycle.
        """
        sql = """
            INSERT INTO rebalance_cycles (cycle_date, strategy, market_regime, notes)
            VALUES (?, ?, ?, ?)
        """
        with self.db.connect() as conn:
            cursor = conn.execute(sql, (cycle_date, strategy, market_regime, notes))
            cycle_id = cursor.lastrowid

        logger.info("Created rebalance cycle id=%d for %s on %s", cycle_id, strategy, cycle_date)
        return cycle_id

    # ------------------------------------------------------------------
    # Decision operations
    # ------------------------------------------------------------------

    def log_decision(
        self,
        date: str,
        strategy: str,
        ai_signal: Dict[str, float],
        human_decision: str,
        ai_confidence: Optional[float] = None,
        ai_confidence_method: Optional[str] = None,
        ai_momentum_scores: Optional[Dict[str, float]] = None,
        ai_selected_assets: Optional[List[str]] = None,
        human_weights: Optional[Dict[str, float]] = None,
        human_reason: Optional[str] = None,
        execution_price: Optional[Dict[str, float]] = None,
        cycle_id: Optional[int] = None,
    ) -> int:
        """
        Record a single asset-allocation decision with AI and human components.

        Args:
            date:                  Signal date (YYYY-MM-DD).
            strategy:              Strategy identifier.
            ai_signal:             AI-recommended weights dict {ticker: weight}.
            human_decision:        One of 'approve' | 'modify' | 'reject'.
            ai_confidence:         Float [0,1] representing model confidence.
            ai_confidence_method:  How confidence was computed (for reproducibility).
            ai_momentum_scores:    Raw momentum scores per asset.
            ai_selected_assets:    List of assets AI chose.
            human_weights:         Modified weights if human_decision == 'modify'.
            human_reason:          Explanation of modification or rejection.
            execution_price:       Actual fill prices {ticker: price}, if known.
            cycle_id:              FK to rebalance_cycles. Optional.

        Returns:
            Integer ID of the newly created decision record.

        Raises:
            ValueError: If human_decision is invalid or human_weights missing
                        when human_decision == 'modify'.
            sqlite3.Error: On database write failure.
        """
        # --- Validation ---
        valid_decisions = {"approve", "modify", "reject"}
        if human_decision not in valid_decisions:
            raise ValueError(
                f"human_decision must be one of {valid_decisions}, got '{human_decision}'"
            )
        if human_decision == "modify" and human_weights is None:
            raise ValueError(
                "human_weights must be provided when human_decision == 'modify'. "
                "If rejecting, use human_decision='reject' instead."
            )

        # --- Serialize JSON fields ---
        sql = """
            INSERT INTO decisions (
                cycle_id, date, strategy,
                ai_signal, ai_confidence, ai_confidence_method,
                ai_momentum_scores, ai_selected_assets,
                human_decision, human_weights, human_reason,
                execution_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            cycle_id,
            date,
            strategy,
            json.dumps(ai_signal),
            ai_confidence,
            ai_confidence_method,
            json.dumps(ai_momentum_scores) if ai_momentum_scores else None,
            json.dumps(ai_selected_assets) if ai_selected_assets else None,
            human_decision,
            json.dumps(human_weights) if human_weights else None,
            human_reason,
            json.dumps(execution_price) if execution_price else None,
        )

        with self.db.connect() as conn:
            cursor = conn.execute(sql, params)
            decision_id = cursor.lastrowid

        logger.info(
            "Logged decision id=%d | date=%s | strategy=%s | human=%s",
            decision_id, date, strategy, human_decision
        )
        return decision_id

    def update_execution_price(
        self,
        decision_id: int,
        execution_price: Dict[str, float],
        executed_at: Optional[str] = None,
    ) -> None:
        """
        Fill in execution prices after order placement.

        These are not known at decision time in a Human-in-the-Loop workflow,
        so this method exists to update the record post-execution.

        Args:
            decision_id:     ID of the decision to update.
            execution_price: {ticker: fill_price} dict.
            executed_at:     ISO datetime string. Defaults to now.
        """
        executed_at = executed_at or datetime.now().isoformat()
        sql = """
            UPDATE decisions
            SET execution_price = ?, executed_at = ?, updated_at = datetime('now')
            WHERE id = ?
        """
        with self.db.connect() as conn:
            conn.execute(sql, (json.dumps(execution_price), executed_at, decision_id))
        logger.info("Updated execution prices for decision id=%d", decision_id)

    def get_decision(self, decision_id: int) -> Optional[Dict]:
        """
        Fetch a single decision by ID.

        Args:
            decision_id: Primary key of the decision.

        Returns:
            Dict with all decision fields (JSON fields deserialized), or None if not found.
        """
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM decisions WHERE id = ?", (decision_id,)
            ).fetchone()
        if row is None:
            return None
        return self._deserialize_decision(dict(row))

    def list_decisions(
        self,
        strategy: Optional[str] = None,
        human_decision: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict]:
        """
        List decisions with optional filters.

        Args:
            strategy:       Filter by strategy name. None = all strategies.
            human_decision: Filter by human decision type. None = all types.
            limit:          Max rows to return.
            offset:         Rows to skip (for pagination).

        Returns:
            List of decision dicts, ordered by date descending.
        """
        conditions: List[str] = []
        params: List[Any] = []

        if strategy:
            conditions.append("strategy = ?")
            params.append(strategy)
        if human_decision:
            conditions.append("human_decision = ?")
            params.append(human_decision)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT d.*, o.actual_return_30d, o.ai_correct
            FROM decisions d
            LEFT JOIN outcomes o ON o.decision_id = d.id
            {where}
            ORDER BY d.date DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self.db.connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [self._deserialize_decision(dict(r)) for r in rows]

    def get_pending_outcomes(self, days_threshold: int = 30) -> List[Dict]:
        """
        Find decisions that are old enough for outcome recording but not yet recorded.

        Args:
            days_threshold: Minimum age in calendar days before outcome is due.

        Returns:
            List of decision dicts without a corresponding outcomes row.
        """
        sql = """
            SELECT d.*
            FROM decisions d
            LEFT JOIN outcomes o ON o.decision_id = d.id
            WHERE o.id IS NULL
              AND d.human_decision != 'reject'
              AND julianday('now') - julianday(d.date) >= ?
            ORDER BY d.date ASC
        """
        with self.db.connect() as conn:
            rows = conn.execute(sql, (days_threshold,)).fetchall()
        return [self._deserialize_decision(dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # Outcome operations
    # ------------------------------------------------------------------

    def log_outcome(
        self,
        decision_id: int,
        actual_return_30d: float,
        benchmark_return_30d: Optional[float] = None,
        asset_returns: Optional[Dict[str, float]] = None,
        ai_only_return_30d: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> int:
        """
        Record the realized outcome for a past decision.

        Automatically computes:
            - ai_correct: based on sign of actual_return_30d vs threshold
            - human_value_add: actual_return - ai_only_return (if both provided)

        Args:
            decision_id:          FK to decisions table.
            actual_return_30d:    Realized portfolio return over 30 days.
            benchmark_return_30d: SPY return over same window.
            asset_returns:        Per-asset return dict for attribution.
            ai_only_return_30d:   Counterfactual: return with pure AI weights.
            notes:                Free-text notes.

        Returns:
            Integer ID of the outcomes record.

        Raises:
            ValueError:     If decision_id not found or outcome already exists.
            sqlite3.Error:  On database error.
        """
        # Verify decision exists
        decision = self.get_decision(decision_id)
        if decision is None:
            raise ValueError(f"Decision id={decision_id} not found.")

        # Check for duplicate
        with self.db.connect() as conn:
            existing = conn.execute(
                "SELECT id FROM outcomes WHERE decision_id = ?", (decision_id,)
            ).fetchone()
        if existing:
            raise ValueError(
                f"Outcome already recorded for decision id={decision_id}. "
                "Use update_outcome() to modify."
            )

        # Compute ai_correct based on defined threshold
        INCONCLUSIVE_THRESHOLD = 0.005  # 0.5% — below this, signal is noise
        if abs(actual_return_30d) < INCONCLUSIVE_THRESHOLD:
            ai_correct = "inconclusive"
        elif actual_return_30d > 0:
            ai_correct = "direction_correct"
        else:
            ai_correct = "direction_wrong"

        human_value_add = None
        if ai_only_return_30d is not None:
            human_value_add = actual_return_30d - ai_only_return_30d

        sql = """
            INSERT INTO outcomes (
                decision_id, actual_return_30d, benchmark_return_30d,
                asset_returns, ai_correct, ai_only_return_30d,
                human_value_add, outcome_date, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, date('now'), ?)
        """
        params = (
            decision_id,
            actual_return_30d,
            benchmark_return_30d,
            json.dumps(asset_returns) if asset_returns else None,
            ai_correct,
            ai_only_return_30d,
            human_value_add,
            notes,
        )

        with self.db.connect() as conn:
            cursor = conn.execute(sql, params)
            outcome_id = cursor.lastrowid

        logger.info(
            "Logged outcome id=%d for decision id=%d | return=%.2f%% | ai_correct=%s",
            outcome_id, decision_id, actual_return_30d * 100, ai_correct
        )
        return outcome_id

    def get_all_outcomes(self) -> List[Dict]:
        """
        Fetch all decisions with their outcomes (LEFT JOIN).

        Returns:
            List of merged decision+outcome dicts.
            Decisions without outcomes have None for outcome fields.
        """
        sql = """
            SELECT
                d.id, d.date, d.strategy,
                d.ai_signal, d.ai_confidence, d.ai_selected_assets,
                d.human_decision, d.human_weights, d.human_reason,
                o.actual_return_30d, o.benchmark_return_30d,
                o.ai_correct, o.human_value_add, o.outcome_date
            FROM decisions d
            LEFT JOIN outcomes o ON o.decision_id = d.id
            ORDER BY d.date DESC
        """
        with self.db.connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [self._deserialize_decision(dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deserialize_decision(row: Dict) -> Dict:
        """
        Deserialize JSON string fields in a database row dict.

        Args:
            row: Raw dict from sqlite3.Row conversion.

        Returns:
            Dict with JSON string fields parsed to Python objects.
        """
        json_fields = [
            "ai_signal", "ai_momentum_scores", "ai_selected_assets",
            "human_weights", "execution_price", "asset_returns"
        ]
        for field in json_fields:
            if field in row and isinstance(row[field], str):
                try:
                    row[field] = json.loads(row[field])
                except (json.JSONDecodeError, TypeError):
                    pass  # Leave as-is if not valid JSON
        return row
