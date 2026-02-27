"""
Risk & Capital Management
==========================
Implements ATR-adjusted position sizing and institutional-grade drawdown
controls.

Key responsibilities:
  • Compute the maximum allowable position size for any given trade.
  • Track running P&L and halt trading when the drawdown limit is breached.
  • Provide stop-loss levels derived from the current ATR.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class TradeParameters:
    """Parameters computed for a single prospective trade."""

    symbol: str
    direction: str          # "long" | "short"
    entry_price: float
    stop_loss_price: float
    position_size: float    # in units / contracts
    risk_amount: float      # dollar amount at risk


class RiskManager:
    """
    Manages per-trade risk and portfolio-level drawdown controls.

    Parameters
    ----------
    total_capital:
        Current total portfolio value in account currency.
    max_risk_per_trade:
        Fraction of ``total_capital`` to risk on a single trade (default 1 %).
    max_drawdown:
        Maximum allowed portfolio drawdown fraction before trading is halted
        (default 10 %).
    atr_stop_multiplier:
        Multiplier applied to the ATR to derive the stop-loss distance.
    """

    def __init__(
        self,
        total_capital: float,
        max_risk_per_trade: float = settings.MAX_RISK_PER_TRADE,
        max_drawdown: float = settings.MAX_DRAWDOWN,
        atr_stop_multiplier: float = settings.ATR_STOP_MULTIPLIER,
    ) -> None:
        if total_capital <= 0:
            raise ValueError("total_capital must be positive")
        self.total_capital = total_capital
        self.peak_capital = total_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_drawdown = max_drawdown
        self.atr_stop_multiplier = atr_stop_multiplier
        self._trading_halted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def trading_halted(self) -> bool:
        return self._trading_halted

    def current_drawdown(self) -> float:
        """Return the current drawdown as a positive fraction (0.0 – 1.0)."""
        if self.peak_capital <= 0:
            return 0.0
        return max(0.0, (self.peak_capital - self.total_capital) / self.peak_capital)

    def update_capital(self, new_capital: float) -> None:
        """
        Update the current portfolio value after a trade is closed.
        Automatically halts trading when the drawdown limit is exceeded.
        """
        if new_capital <= 0:
            raise ValueError("new_capital must be positive")
        self.total_capital = new_capital
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
        drawdown = self.current_drawdown()
        if drawdown >= self.max_drawdown:
            self._trading_halted = True
            logger.warning(
                "Trading HALTED – drawdown %.2f%% reached/exceeded limit %.2f%%",
                drawdown * 100,
                self.max_drawdown * 100,
            )
        else:
            self._trading_halted = False

    def compute_trade_parameters(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        atr: float,
    ) -> TradeParameters | None:
        """
        Compute ATR-adjusted position size and stop-loss for a proposed trade.

        Returns ``None`` if trading is currently halted or if inputs are invalid.
        """
        if self._trading_halted:
            logger.warning("Cannot open trade on %s – trading is halted", symbol)
            return None
        if entry_price <= 0 or atr <= 0:
            logger.error("Invalid entry_price or ATR for %s", symbol)
            return None
        if direction not in ("long", "short"):
            logger.error("Unknown direction %r for %s", direction, symbol)
            return None

        stop_distance = self.atr_stop_multiplier * atr
        if direction == "long":
            stop_loss_price = entry_price - stop_distance
        else:
            stop_loss_price = entry_price + stop_distance

        risk_amount = self.total_capital * self.max_risk_per_trade
        # Position size = dollar risk / per-unit risk
        position_size = risk_amount / stop_distance if stop_distance > 0 else 0.0

        params = TradeParameters(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            position_size=round(position_size, 4),
            risk_amount=round(risk_amount, 2),
        )
        logger.info(
            "Trade parameters for %s %s: size=%.4f, SL=%.4f, risk=$%.2f",
            direction,
            symbol,
            position_size,
            stop_loss_price,
            risk_amount,
        )
        return params
