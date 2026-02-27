# 📈 Kesha: Quantitative Trading Automation System

An advanced, production-grade automated trading system utilizing the **OpenClaw framework**. This project bridges the gap between Pine Script-based algorithmic signal generation and Python-driven execution logic. Designed for high-frequency precision, it focuses on strict risk management, dynamic capital allocation, and real-time market validation.

---

## ⚙️ Core Architecture & Features

| Component | Module | Description |
|-----------|--------|-------------|
| **Multi-Layer Validation Engine** | `src/validation/engine.py` | Dynamically filters market noise, validates BPU/BSU retest systems, and enforces strict spread/slippage thresholds before execution. |
| **Risk & Capital Management** | `src/risk/manager.py` | ATR-adjusted position sizing and institutional-grade drawdown controls. |
| **Signal Generation** | `src/signals/generator.py` | Detects false breakouts and multi-candle confirmations from raw OHLCV data. |
| **Order Execution** | `src/execution/executor.py` | Submits and manages orders via the OpenClaw REST API. |
| **Market Data** | `src/data/market_data.py` | Abstracts OHLCV and quote retrieval behind a provider interface. |
| **Notifications** | `src/notifications/telegram_bot.py` | Real-time trade alerts and system-status messages via Telegram Bot API. |

### Validation Layers (applied in order)
1. **Spread check** – reject if the current bid/ask spread exceeds the configured threshold.
2. **Slippage check** – reject if the estimated slippage is too large.
3. **Noise filter** – IQR-based outlier test on recent bar ranges.
4. **BPU/BSU retest** – confirm that price has retested the key level within tolerance.
5. **Multi-candle confirmation** – require N consecutive confirming candles.

---

## 🗂️ Project Structure

```
quant-trading-bot-kesha/
├── config/
│   └── settings.py          # All configuration (loaded from environment variables)
├── src/
│   ├── main.py              # Orchestrator / entry point
│   ├── data/
│   │   └── market_data.py   # Market-data provider (OpenClaw-backed)
│   ├── signals/
│   │   └── generator.py     # Signal generation (false breakout + momentum)
│   ├── validation/
│   │   └── engine.py        # Multi-layer validation engine
│   ├── risk/
│   │   └── manager.py       # Risk & capital management
│   ├── execution/
│   │   └── executor.py      # OpenClaw order executor
│   └── notifications/
│       └── telegram_bot.py  # Telegram Bot notifications
├── pine_scripts/
│   └── kesha_signals.pine   # Pine Script v5 indicator (TradingView)
├── tests/
│   ├── test_validation.py
│   ├── test_risk.py
│   └── test_signals.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## 🛠️ Tech Stack

- **Languages:** Python 3.12, Pine Script v5, C++ (OpenClaw internals)
- **Frameworks & Infrastructure:** OpenClaw, Docker, Telegram Bot API
- **Data & Analytics:** Pandas, NumPy, XGBoost (feature pipeline integration)

---

## 🚀 Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys and settings
```

### 2. Run locally

```bash
pip install -r requirements.txt
python -m src.main
```

### 3. Run with Docker

```bash
docker-compose up -d
```

### 4. Run tests

```bash
pytest tests/
```

---

## 📡 Pine Script Indicator

The `pine_scripts/kesha_signals.pine` script can be added to TradingView as a custom indicator. It detects the same BPU/BSU retest and false-breakout patterns as the Python engine and can fire **webhook alerts** in the JSON format expected by the Kesha bot.

---

## ⚠️ Disclaimer

*This repository contains the architectural outline and non-sensitive logic of the proprietary "Kesha" trading system. Core execution keys and proprietary alpha-generating models are kept private.*

**Trading financial instruments involves significant risk of loss. This software is provided for educational and research purposes only.**
