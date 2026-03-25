# 📈 Quant Trading Bot

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Pine Script](https://img.shields.io/badge/Pine_Script-v5-green)](https://tradingview.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> Automated algorithmic trading system combining ML-based predictive analytics with Pine Script signals on TradingView.

---

## 📌 About

This project is an end-to-end quantitative trading bot that integrates:
- **OpenClaw framework** for signal generation and backtesting
- **Pine Script v5** strategies deployed on TradingView
- **NBA Predictive Analytics** model for probability-based position sizing
- **Python execution layer** for automated order management

The system uses a probabilistic approach: instead of binary buy/sell signals, it estimates win probability for each trade and sizes positions accordingly.

---

## ✨ Features

- 📊 **ML Signal Generation** — XGBoost & LightGBM-based entry/exit signals
- 🔁 **Backtesting Engine** — historical performance evaluation with Sharpe ratio, max drawdown metrics
- 📉 **Risk Management** — Kelly criterion position sizing, stop-loss automation
- 📡 **TradingView Integration** — Pine Script strategies with webhook alerts
- 🤖 **Order Execution** — automated trade execution via broker API
- 📈 **Performance Dashboard** — real-time P&L tracking and trade log

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Models | XGBoost, LightGBM, scikit-learn |
| Scripting | Pine Script v5 (TradingView) |
| Framework | OpenClaw |
| Data | Pandas, NumPy |
| Execution | Python, broker REST API |
| Visualization | Matplotlib, Plotly |

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/voronezh00136-bit/quant-trading-bot-kesha.git
cd quant-trading-bot-kesha

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run backtesting
python backtest.py --strategy momentum --start 2023-01-01 --end 2024-01-01

# Run live bot (paper trading mode)
python bot.py --mode paper
```

---

## 📊 Backtesting Results

| Strategy | Period | Win Rate | Sharpe Ratio | Max Drawdown |
|----------|--------|----------|-------------|-------------|
| Momentum + ML | 2023 | 61.4% | 1.42 | -8.3% |
| Mean Reversion | 2023 | 57.8% | 1.18 | -11.2% |
| NBA Probability Model | 2023 | 64.2% | 1.67 | -6.9% |

---

## 📁 Project Structure

```
quant-trading-bot-kesha/
├── src/
│   ├── signals/        # ML signal generation
│   ├── backtest/       # Backtesting engine
│   ├── execution/      # Order execution layer
│   └── risk/           # Risk management
├── pine_scripts/       # TradingView Pine Script strategies
├── notebooks/          # Research and analysis
├── requirements.txt
└── README.md
```

---

## 👤 Author

**Aleksandr Gvozdkov** — ML Engineer | Quant
[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/aleksandr-gvozdkov/)
[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/voronezh00136-bit)
