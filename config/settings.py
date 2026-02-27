"""
Kesha – configuration & settings
All runtime parameters are loaded from environment variables so that no
secrets are ever hard-coded in source.  Copy .env.example to .env and fill
in your own values before running.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Broker / OpenClaw connection
# ---------------------------------------------------------------------------
OPENCLAW_API_KEY: str = os.getenv("OPENCLAW_API_KEY", "")
OPENCLAW_API_SECRET: str = os.getenv("OPENCLAW_API_SECRET", "")
OPENCLAW_BASE_URL: str = os.getenv("OPENCLAW_BASE_URL", "https://api.openclaw.io/v1")

# ---------------------------------------------------------------------------
# Telegram notifications
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------------------------
# Risk management defaults
# ---------------------------------------------------------------------------
# Maximum fraction of total capital risked per trade (e.g. 0.01 = 1 %)
MAX_RISK_PER_TRADE: float = float(os.getenv("MAX_RISK_PER_TRADE", "0.01"))
# Maximum total portfolio drawdown before trading is halted (e.g. 0.10 = 10 %)
MAX_DRAWDOWN: float = float(os.getenv("MAX_DRAWDOWN", "0.10"))
# ATR multiplier used for stop-loss calculation
ATR_STOP_MULTIPLIER: float = float(os.getenv("ATR_STOP_MULTIPLIER", "1.5"))

# ---------------------------------------------------------------------------
# Market / execution constraints
# ---------------------------------------------------------------------------
# Supported trading symbols
SYMBOLS: list[str] = os.getenv(
    "SYMBOLS", "SPY,XAUUSD,USOIL,EURUSD"
).split(",")
# Maximum acceptable spread as a fraction of price (e.g. 0.0005 = 0.05 %)
MAX_SPREAD_FRACTION: float = float(os.getenv("MAX_SPREAD_FRACTION", "0.0005"))
# Maximum acceptable slippage in price units
MAX_SLIPPAGE: float = float(os.getenv("MAX_SLIPPAGE", "0.5"))

# ---------------------------------------------------------------------------
# Validation engine
# ---------------------------------------------------------------------------
# Number of confirmations required before a signal is accepted
SIGNAL_CONFIRMATION_CANDLES: int = int(
    os.getenv("SIGNAL_CONFIRMATION_CANDLES", "2")
)
