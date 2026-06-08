"""
decisions/db.py
===============
Database layer for the Decision Journal.

Manages SQLite schema creation, connection lifecycle, and all
data access operations. This module is the ONLY place that touches
the database — CLI and analyzer import from here, never write SQL directly.

Schema Design Rationale:
    rebalance_cycles : Groups decisions by monthly event. One cycle = one rebalance.
    decisions        : One row per allocation decision (AI signal + human review).
                       Immutable after creation except execution_price (post-fill).
    outcomes         : Realized results recorded ~30 days later. Separated because
                       results arrive at a different time than the decision.
    executions       : Individual order fills. One decision → many execution legs.
                       Tracks broker-level detail: symbol, side, qty, fill price,
                       commission. Separate from decisions because:
                         (a) execution is post-approval, not concurrent with decision
                         (b) one allocation decision may generate multiple fills
                         (c) fills may arrive over minutes/hours (partial fills)

ai_correct Definition:
    'direction_correct' — realized return > +0.5%
    'direction_wrong'   — realized return < -0.5%
    'inconclusive'      — |return| < 0.5%, noise floor

commission_type in executions:
    'flat' — fixed dollar amount per fill (e.g. $1.00)
    'bps'  — basis points of notional (e.g. 0.5 bps = 0.00005 × notional)

Author: Yuchuan Wu — Phase 2
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL — append-only; existing tables unchanged by new additions
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
-- ──────────────────────────────────────────────────────────────────────────
-- rebalance_cycles: one row per monthly rebalance event
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rebalance_cycles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_date      TEXT    NOT NULL,
    strategy        TEXT    NOT NULL,
    market_regime   TEXT,
    notes           TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────────────────────────────
-- decisions: one row per asset-allocation decision
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS decisions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id                INTEGER REFERENCES rebalance_cycles(id),
    date                    TEXT    NOT NULL,
    strategy                TEXT    NOT NULL,
    ai_signal               TEXT    NOT NULL,
    ai_confidence           REAL,
    ai_confidence_method    TEXT,
    ai_momentum_scores      TEXT,
    ai_selected_assets      TEXT,
    human_decision          TEXT    NOT NULL
                            CHECK (human_decision IN ('approve', 'modify', 'reject')),
    human_weights           TEXT,
    human_reason            TEXT,
    execution_price         TEXT,
    executed_at             TEXT,
    created_at              TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at              TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────────────────────────────
-- outcomes: one row per decision, filled ~30 days later
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS outcomes (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id          INTEGER NOT NULL UNIQUE REFERENCES decisions(id),
    actual_return_30d    REAL,
    benchmark_return_30d REAL,
    asset_returns        TEXT,
    ai_correct           TEXT
                         CHECK (ai_correct IN ('direction_correct', 'direction_wrong',
                                               'inconclusive', NULL)),
    ai_only_return_30d   REAL,
    human_value_add      REAL,
    outcome_date         TEXT,
    notes                TEXT,
    created_at           TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────────────────────────────
-- executions: broker-level fill records linked to a decision
--
-- One decision → one or many executions (one per symbol leg, or partial fills).
-- Only non-rejected decisions should have executions — enforced at app layer
-- because SQLite CHECK can't join across tables.
--
-- Fields:
--   symbol          : ticker symbol (e.g. 'SPY')
--   side            : 'buy' | 'sell' | 'sell_short'
--   quantity        : number of shares / units (REAL to support fractional)
--   price           : fill price per unit in USD
--   net_amount      : quantity × price ± commission (signed, negative = cash out)
--   commission      : absolute commission cost in USD (always positive)
--   commission_type : 'flat' (fixed $) | 'bps' (basis points of notional)
--   broker          : broker identifier (e.g. 'alpaca', 'ibkr', 'paper')
--   order_id        : broker-assigned order reference for reconciliation
--   execution_time  : ISO datetime of fill
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS executions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id     INTEGER NOT NULL REFERENCES decisions(id),
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL
                    CHECK (side IN ('buy', 'sell', 'sell_short')),
    quantity        REAL    NOT NULL CHECK (quantity > 0),
    price           REAL    NOT NULL CHECK (price > 0),
    net_amount      REAL,
    commission      REAL    NOT NULL DEFAULT 0.0 CHECK (commission >= 0),
    commission_type TEXT    NOT NULL DEFAULT 'flat'
                    CHECK (commission_type IN ('flat', 'bps')),
    broker          TEXT    NOT NULL DEFAULT 'paper',
    order_id        TEXT,
    execution_time  TEXT    NOT NULL,
    notes           TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────────────────────────────
-- Indexes
-- ──────────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_decisions_date      ON decisions(date);
CREATE INDEX IF NOT EXISTS idx_decisions_strategy  ON decisions(strategy);
CREATE INDEX IF NOT EXISTS idx_outcomes_decision   ON outcomes(decision_id);
CREATE INDEX IF NOT EXISTS idx_executions_decision ON executions(decision_id);
CREATE INDEX IF NOT EXISTS idx_executions_symbol   ON executions(symbol);
CREATE INDEX IF NOT EXISTS idx_executions_time     ON executions(execution_time);
"""


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------

class Database:
    """
    SQLite database manager for the Decision Journal.

    Handles connection lifecycle, schema initialization, and provides
    a context-managed connection for safe transaction handling.

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
        Create tables and indexes if they don't exist. Idempotent.
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
            sqlite3.Error: Re-raised after rollback.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
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
# Data Access Object
# ---------------------------------------------------------------------------

class DecisionRepository:
    """
    All read/write operations for decisions, outcomes, and executions.

    Single source of truth for database interactions. CLI, analyzer,
    and review_copilot import from here; they never write raw SQL.
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
            market_regime: Optional regime label from Layer 4 ('bull'|'bear'|'sideways').
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
        logger.info("Created cycle id=%d for %s on %s", cycle_id, strategy, cycle_date)
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
            date:                 Signal date (YYYY-MM-DD).
            strategy:             Strategy identifier.
            ai_signal:            AI-recommended weights dict {ticker: weight}.
            human_decision:       One of 'approve' | 'modify' | 'reject'.
            ai_confidence:        Float [0,1] representing model confidence.
            ai_confidence_method: How confidence was computed.
            ai_momentum_scores:   Raw momentum scores per asset.
            ai_selected_assets:   List of assets AI chose.
            human_weights:        Modified weights if human_decision == 'modify'.
            human_reason:         Explanation of modification or rejection.
            execution_price:      Actual fill prices {ticker: price}, if known.
            cycle_id:             FK to rebalance_cycles. Optional.

        Returns:
            Integer ID of the newly created decision record.

        Raises:
            ValueError: If human_decision is invalid or human_weights missing
                        when human_decision == 'modify'.
            sqlite3.Error: On database write failure.
        """
        valid = {"approve", "modify", "reject"}
        if human_decision not in valid:
            raise ValueError(f"human_decision must be one of {valid}, got '{human_decision}'")
        if human_decision == "modify" and human_weights is None:
            raise ValueError(
                "human_weights must be provided when human_decision == 'modify'."
            )

        sql = """
            INSERT INTO decisions (
                cycle_id, date, strategy,
                ai_signal, ai_confidence, ai_confidence_method,
                ai_momentum_scores, ai_selected_assets,
                human_decision, human_weights, human_reason, execution_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            cycle_id, date, strategy,
            json.dumps(ai_signal),
            ai_confidence, ai_confidence_method,
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
            decision_id, date, strategy, human_decision,
        )
        return decision_id

    def update_execution_price(
        self,
        decision_id: int,
        execution_price: Dict[str, float],
        executed_at: Optional[str] = None,
    ) -> None:
        """
        Fill in execution prices after order placement (post-approval update).

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
        Fetch a single decision by ID with deserialized JSON fields.

        Args:
            decision_id: Primary key of the decision.

        Returns:
            Dict with all fields (JSON fields deserialized), or None if not found.
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
        List decisions with optional filters, joined with outcome summary.

        Args:
            strategy:       Filter by strategy name. None = all.
            human_decision: Filter by decision type. None = all.
            limit:          Max rows to return.
            offset:         Rows to skip (pagination).

        Returns:
            List of dicts ordered by date descending.
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
        Find decisions old enough for outcome recording but not yet recorded.

        Uses ISO date comparison (safe for YYYY-MM-DD strings in SQLite).

        Args:
            days_threshold: Minimum age in calendar days.

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
            ai_correct      : 3-way classification based on return magnitude
            human_value_add : actual_return - ai_only_return (if counterfactual provided)

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
            ValueError:    If decision_id not found or outcome already exists.
            sqlite3.Error: On database error.
        """
        decision = self.get_decision(decision_id)
        if decision is None:
            raise ValueError(f"Decision id={decision_id} not found.")

        with self.db.connect() as conn:
            existing = conn.execute(
                "SELECT id FROM outcomes WHERE decision_id = ?", (decision_id,)
            ).fetchone()
        if existing:
            raise ValueError(
                f"Outcome already recorded for decision id={decision_id}."
            )

        INCONCLUSIVE_THRESHOLD = 0.005
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
        with self.db.connect() as conn:
            cursor = conn.execute(sql, (
                decision_id, actual_return_30d, benchmark_return_30d,
                json.dumps(asset_returns) if asset_returns else None,
                ai_correct, ai_only_return_30d, human_value_add, notes,
            ))
            outcome_id = cursor.lastrowid
        logger.info(
            "Logged outcome id=%d for decision id=%d | return=%.2f%% | ai_correct=%s",
            outcome_id, decision_id, actual_return_30d * 100, ai_correct,
        )
        return outcome_id

    def get_all_outcomes(self) -> List[Dict]:
        """
        Fetch all decisions with their outcomes (LEFT JOIN).

        Returns:
            List of merged dicts ordered by date descending.
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
    # Execution operations  (NEW in Phase 2.2)
    # ------------------------------------------------------------------

    def add_execution(
        self,
        decision_id: int,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        execution_time: str,
        commission: float = 0.0,
        commission_type: str = "flat",
        broker: str = "paper",
        order_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        """
        Record a single broker fill associated with a decision.

        One decision may generate multiple execution records (one per symbol leg,
        or multiple records per symbol for partial fills). The net_amount is
        computed automatically as (quantity × price) + commission, signed by side:
        buys are cash-out (negative net_amount from cash perspective),
        sells are cash-in (positive net_amount).

        Args:
            decision_id:     FK to decisions table. Must exist and be non-rejected.
            symbol:          Ticker symbol (e.g. 'SPY').
            side:            Trade direction: 'buy' | 'sell' | 'sell_short'.
            quantity:        Number of shares/units (must be > 0).
            price:           Fill price per unit in USD (must be > 0).
            execution_time:  ISO datetime of fill (e.g. '2024-01-31T09:31:00').
            commission:      Commission cost in USD (default 0.0).
            commission_type: 'flat' (fixed $) or 'bps' (basis points of notional).
            broker:          Broker identifier (default 'paper').
            order_id:        Broker-assigned reference string for reconciliation.
            notes:           Free-text notes.

        Returns:
            Integer ID of the newly created execution record.

        Raises:
            ValueError: If decision_id not found, decision is rejected,
                        side is invalid, quantity or price are non-positive.
            sqlite3.Error: On database write failure.
        """
        valid_sides = {"buy", "sell", "sell_short"}
        if side not in valid_sides:
            raise ValueError(f"side must be one of {valid_sides}, got '{side}'")
        if quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {quantity}")
        if price <= 0:
            raise ValueError(f"price must be > 0, got {price}")

        decision = self.get_decision(decision_id)
        if decision is None:
            raise ValueError(f"Decision id={decision_id} not found.")
        if decision["human_decision"] == "reject":
            raise ValueError(
                f"Decision id={decision_id} was rejected — cannot log executions "
                "for a rejected decision."
            )

        # Compute net_amount: buys cost cash (negative), sells return cash (positive)
        notional = quantity * price
        if commission_type == "bps":
            # commission argument is interpreted as basis points, convert to dollars
            commission_dollars = notional * (commission / 10_000)
        else:
            commission_dollars = commission

        if side == "buy":
            net_amount = -(notional + commission_dollars)
        else:  # sell or sell_short
            net_amount = notional - commission_dollars

        sql = """
            INSERT INTO executions (
                decision_id, symbol, side, quantity, price,
                net_amount, commission, commission_type,
                broker, order_id, execution_time, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.connect() as conn:
            cursor = conn.execute(sql, (
                decision_id, symbol, side, quantity, price,
                net_amount, commission_dollars, commission_type,
                broker, order_id, execution_time, notes,
            ))
            execution_id = cursor.lastrowid

        logger.info(
            "Logged execution id=%d | decision=%d | %s %s %.2f @ %.2f | net=%.2f",
            execution_id, decision_id, side.upper(), symbol, quantity, price, net_amount,
        )
        return execution_id

    def get_executions_by_decision(self, decision_id: int) -> List[Dict]:
        """
        Fetch all execution records for a given decision, ordered by time.

        Args:
            decision_id: FK to decisions table.

        Returns:
            List of execution dicts (may be empty if no fills recorded yet).
            Each dict has keys: id, decision_id, symbol, side, quantity, price,
            net_amount, commission, commission_type, broker, order_id,
            execution_time, notes, created_at.
        """
        sql = """
            SELECT *
            FROM executions
            WHERE decision_id = ?
            ORDER BY execution_time ASC, id ASC
        """
        with self.db.connect() as conn:
            rows = conn.execute(sql, (decision_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_full_trace(self, decision_id: int) -> Optional[Dict]:
        """
        Return the complete lifecycle of a decision as a single structured dict.

        Assembles: decision → rebalance_cycle → executions → outcome
        into a nested dict optimized for display (CLI trace command) and
        narrative generation (review_copilot).

        Args:
            decision_id: Primary key of the decision.

        Returns:
            Dict with keys:
                decision        : full decision dict (deserialized)
                cycle           : rebalance_cycle dict, or None
                executions      : list of execution dicts (may be empty)
                outcome         : outcome dict, or None
                execution_summary : {
                    total_legs      : int,
                    total_notional  : float,
                    total_commission: float,
                    symbols_traded  : list[str],
                    brokers_used    : list[str],
                    first_fill_time : str | None,
                    last_fill_time  : str | None,
                }
            Returns None if decision_id not found.

        Example::

            trace = repo.get_full_trace(3)
            # trace["decision"]["human_decision"] == "modify"
            # trace["executions"][0]["symbol"] == "SPY"
            # trace["outcome"]["actual_return_30d"] == 0.034
        """
        decision = self.get_decision(decision_id)
        if decision is None:
            return None

        # Fetch cycle if linked
        cycle = None
        if decision.get("cycle_id"):
            with self.db.connect() as conn:
                row = conn.execute(
                    "SELECT * FROM rebalance_cycles WHERE id = ?",
                    (decision["cycle_id"],),
                ).fetchone()
            if row:
                cycle = dict(row)

        # Fetch executions
        executions = self.get_executions_by_decision(decision_id)

        # Fetch outcome
        outcome = None
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM outcomes WHERE decision_id = ?", (decision_id,)
            ).fetchone()
        if row:
            outcome = self._deserialize_decision(dict(row))

        # Build execution summary
        total_notional = sum(abs(e.get("net_amount") or 0.0) for e in executions)
        total_commission = sum(e.get("commission") or 0.0 for e in executions)
        symbols = list(dict.fromkeys(e["symbol"] for e in executions))  # order-preserving unique
        brokers = list(dict.fromkeys(e["broker"] for e in executions))
        fill_times = [e["execution_time"] for e in executions if e.get("execution_time")]

        execution_summary = {
            "total_legs": len(executions),
            "total_notional": round(total_notional, 2),
            "total_commission": round(total_commission, 4),
            "symbols_traded": symbols,
            "brokers_used": brokers,
            "first_fill_time": min(fill_times) if fill_times else None,
            "last_fill_time": max(fill_times) if fill_times else None,
        }

        logger.debug("Full trace assembled for decision id=%d", decision_id)
        return {
            "decision": decision,
            "cycle": cycle,
            "executions": executions,
            "outcome": outcome,
            "execution_summary": execution_summary,
        }

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
            "human_weights", "execution_price", "asset_returns",
        ]
        for field in json_fields:
            if field in row and isinstance(row[field], str):
                try:
                    row[field] = json.loads(row[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return row
