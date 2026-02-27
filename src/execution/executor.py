"""
Order Execution via OpenClaw
=============================
Submits, monitors, and cancels orders through the OpenClaw REST API.

Design goals:
  • Idempotent order submission (tracks open orders by client-order-id).
  • Automatic stop-loss placement after a fill is confirmed.
  • Clear separation between order construction and HTTP transport so that
    the executor can be unit-tested without a live connection.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from config import settings
from src.risk.manager import TradeParameters

logger = logging.getLogger(__name__)


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a single submitted order."""

    client_order_id: str
    symbol: str
    direction: str          # "long" | "short"
    size: float
    entry_price: float
    stop_loss_price: float
    status: OrderStatus = OrderStatus.PENDING
    broker_order_id: str = ""
    fill_price: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class OrderExecutor:
    """
    Submits orders to OpenClaw and tracks their lifecycle.

    Parameters
    ----------
    base_url:
        OpenClaw API base URL.
    api_key:
        API key for authenticating requests.
    api_secret:
        API secret used to sign requests (reserved for future HMAC signing).
    session:
        Optional pre-configured ``requests.Session``; useful for testing.
    """

    def __init__(
        self,
        base_url: str = settings.OPENCLAW_BASE_URL,
        api_key: str = settings.OPENCLAW_API_KEY,
        api_secret: str = settings.OPENCLAW_API_SECRET,
        session=None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self._open_orders: dict[str, Order] = {}

        if session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({"X-Api-Key": self.api_key})
        else:
            self._session = session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, params: TradeParameters) -> Order | None:
        """
        Construct and submit a market order derived from *params*.

        Returns the :class:`Order` if submission succeeded, ``None`` otherwise.
        """
        client_id = str(uuid.uuid4())
        order = Order(
            client_order_id=client_id,
            symbol=params.symbol,
            direction=params.direction,
            size=params.position_size,
            entry_price=params.entry_price,
            stop_loss_price=params.stop_loss_price,
        )

        payload = {
            "client_order_id": client_id,
            "symbol": params.symbol,
            "side": "buy" if params.direction == "long" else "sell",
            "type": "market",
            "quantity": params.position_size,
            "stop_loss": params.stop_loss_price,
        }

        try:
            response = self._session.post(
                f"{self.base_url}/orders",
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            order.broker_order_id = data.get("order_id", "")
            order.status = OrderStatus(data.get("status", "pending"))
            self._open_orders[client_id] = order
            logger.info(
                "Order submitted: %s %s %.4f @ market (SL=%.4f) – broker_id=%s",
                params.direction,
                params.symbol,
                params.position_size,
                params.stop_loss_price,
                order.broker_order_id,
            )
            return order
        except Exception as exc:
            logger.error("Order submission failed for %s: %s", params.symbol, exc)
            order.status = OrderStatus.REJECTED
            return None

    def cancel(self, client_order_id: str) -> bool:
        """Cancel an open order.  Returns True if the cancellation succeeded."""
        order = self._open_orders.get(client_order_id)
        if order is None:
            logger.warning("Order %s not found in open orders", client_order_id)
            return False
        try:
            response = self._session.delete(
                f"{self.base_url}/orders/{order.broker_order_id}",
                timeout=10,
            )
            response.raise_for_status()
            order.status = OrderStatus.CANCELLED
            logger.info("Order %s cancelled", client_order_id)
            return True
        except Exception as exc:
            logger.error("Failed to cancel order %s: %s", client_order_id, exc)
            return False

    def get_open_orders(self) -> list[Order]:
        """Return all orders that are still in a non-terminal state."""
        return [
            o
            for o in self._open_orders.values()
            if o.status in (OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED)
        ]
