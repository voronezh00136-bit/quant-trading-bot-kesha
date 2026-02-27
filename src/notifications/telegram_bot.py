"""
Telegram Bot Notifications
===========================
Sends real-time trade alerts and system status messages via the Telegram Bot
API using the ``python-telegram-bot`` library.
"""
from __future__ import annotations

import logging
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Thin wrapper around the Telegram Bot API ``sendMessage`` endpoint.

    Uses the synchronous ``requests``-based send to keep the caller free of
    async boilerplate; for high-throughput deployments this can be swapped for
    the async version from ``python-telegram-bot``.

    Parameters
    ----------
    token:
        Telegram bot token (from BotFather).
    chat_id:
        Target chat / channel ID.
    session:
        Optional ``requests.Session`` for testing.
    """

    BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(
        self,
        token: str = settings.TELEGRAM_BOT_TOKEN,
        chat_id: str = settings.TELEGRAM_CHAT_ID,
        session: Any = None,
    ) -> None:
        self.token = token
        self.chat_id = chat_id
        if session is None:
            import requests
            self._session = requests.Session()
        else:
            self._session = session

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def send(self, message: str, parse_mode: str = "Markdown") -> bool:
        """
        Send *message* to the configured chat.

        Returns ``True`` on success, ``False`` on failure.
        """
        if not self.token or not self.chat_id:
            logger.debug("Telegram not configured – message suppressed: %s", message)
            return False
        url = self.BASE_URL.format(token=self.token)
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode,
        }
        try:
            response = self._session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.debug("Telegram message sent successfully")
            return True
        except Exception as exc:
            logger.error("Failed to send Telegram message: %s", exc)
            return False

    def notify_signal(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        confidence: float,
        source: str,
    ) -> bool:
        """Send a formatted signal notification."""
        emoji = "🟢" if direction == "long" else "🔴"
        message = (
            f"{emoji} *Signal detected*\n"
            f"Symbol: `{symbol}`\n"
            f"Direction: `{direction.upper()}`\n"
            f"Entry: `{entry_price:.4f}`\n"
            f"Stop-loss: `{stop_loss:.4f}`\n"
            f"Confidence: `{confidence:.1%}`\n"
            f"Source: `{source}`"
        )
        return self.send(message)

    def notify_order(
        self,
        symbol: str,
        direction: str,
        size: float,
        entry_price: float,
        broker_order_id: str,
    ) -> bool:
        """Send a formatted order-submitted notification."""
        emoji = "📥"
        message = (
            f"{emoji} *Order submitted*\n"
            f"Symbol: `{symbol}`\n"
            f"Direction: `{direction.upper()}`\n"
            f"Size: `{size:.4f}`\n"
            f"Entry: `{entry_price:.4f}`\n"
            f"Broker ID: `{broker_order_id}`"
        )
        return self.send(message)

    def notify_drawdown_halt(self, drawdown: float) -> bool:
        """Send a critical drawdown-halt alert."""
        message = (
            f"🚨 *TRADING HALTED*\n"
            f"Portfolio drawdown reached `{drawdown:.1%}` – all new entries suspended."
        )
        return self.send(message)
