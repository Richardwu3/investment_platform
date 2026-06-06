# FinPilot -- AI-Augmented Investment Research Platform
**Phase 1: GTAA Backtest Engine**

## Architecture

```
Data Layer → Strategy Layer → Backtest Engine (5 Layers) → AI Agent → Human Review → Decision Logger → Feedback Loop
```

### Design Principles
| Principle | Implementation |
|-----------|---------------|
| Code = Truth | All metrics computed deterministically in Python |
| LLM = Interface | AI Agent reads output dicts; never modifies strategy state |
| Human-in-the-Loop | AI generates report, human approves before any order |
| Look-ahead Free | `shift(1)` at momentum computation + weight execution boundary |

---

## Phase 1 File Structure

```
investment_platform/
├── strategies/
│   ├── base.py          # IStrategy abstract class, SignalResult, shared utilities
│   └── gtaa.py          # GTAA implementation (SPY/QQQ/TLT/GLD/DBC, 126d momentum, Top2)
├── backtest/
│   ├── engine.py        # 5-layer validation framework
│   └── run_first_backtest.py  # Entry point
├── data/                # Auto-created, CSV price cache
├── reports/             # Auto-created, output JSON + CSV
└── requirements.txt
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full backtest (all 5 layers)
python backtest/run_first_backtest.py

# Fast mode (skip walk-forward and param sweep)
python backtest/run_first_backtest.py --fast

# Custom parameters
python backtest/run_first_backtest.py --start 2015-01-01 --lookback 189 --top-n 3
```

---

## The 5-Layer Validation Framework

| Layer | Name | Question Answered |
|-------|------|------------------|
| 1 | Standard Backtest | Does the strategy have edge over the full sample? |
| 2 | Walk-Forward | Does performance generalize out-of-sample? |
| 3 | Parameter Stability | Is performance robust to parameter variation? |
| 4 | Market Regime | In which environments does the strategy add value? |
| 5 | Block Bootstrap | Is performance statistically significant vs. luck? |

---

## Look-ahead Bias Architecture

```
prices[t-1] ──→ compute_momentum() ──→ scores[t]  ← shift(1) inside function
                                            │
                                    equal_weight_top_n()
                                            │
                                      weights_raw[t]
                                            │
                                       .shift(1)       ← applied by strategy
                                            │
                                  execution_weights[t]  ← "hold during day t"
                                            │
                              price_return[t] × weight[t]  ← correct!
```

Signal computed at month-end close of day t-1 → applied to returns of day t.

---

## GTAA Strategy Details

- **Universe**: SPY (US Equity), QQQ (Tech/Growth), TLT (Long Duration Bond), GLD (Gold), DBC (Commodities)
- **Signal**: 126 trailing trading day momentum = P(t-1) / P(t-1-126) - 1
- **Selection**: Top 2 assets by momentum score
- **Weights**: Equal weight (50/50)
- **Rebalance**: Month-end signal → month+1 day 1 execution
- **Cash Rule**: If selected asset has negative momentum → go to cash (configurable)
- **Costs**: Slippage 10bps + Commission 5bps (one-way, applied on turnover)
- **Reference**: Faber (2007), "A Quantitative Approach to Tactical Asset Allocation"

---

## AI Agent Integration (Phase 2)

The engine outputs a structured `results` dict:
```python
results["summary"]          # High-level metrics for LLM prompt injection
results["layer_1"]          # Full metrics + equity curves
results["layer_5"]          # Statistical significance for narrative
signal.metadata             # Selected assets, momentum scores per rebalance date
strategy.get_metadata()     # Strategy description for LLM to explain logic
```

The LLM reads these dicts to generate plain-English research reports.
It **never** calls `generate_signals()` or modifies numerical results.
