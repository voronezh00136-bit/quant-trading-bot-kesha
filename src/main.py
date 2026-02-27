"""
Kesha – Main Entry Point
=========================
Orchestrates the full signal-to-execution pipeline:

  1. Fetch latest market data for each configured symbol.
  2. Run the signal generator.
  3. Validate each signal through the multi-layer validation engine.
  4. Compute ATR-adjusted trade parameters via the risk manager.
  5. Submit the order to OpenClaw.
  6. Send a Telegram notification.

Run once via cron / scheduler or as a continuous loop depending on the
deployment model.
"""
from __future__ import annotations

import logging
import time

from config import settings
from src.data.market_data import OpenClawDataProvider
from src.execution.executor import OrderExecutor
from src.notifications.telegram_bot import TelegramNotifier
from src.risk.manager import RiskManager
from src.signals.generator import SignalGenerator, SignalType
from src.validation.engine import MarketSnapshot, ValidationEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("kesha.main")

# ---------------------------------------------------------------------------
# Component registry – instantiated once and reused across iterations
# ---------------------------------------------------------------------------
_data_provider = OpenClawDataProvider(
    base_url=settings.OPENCLAW_BASE_URL,
    api_key=settings.OPENCLAW_API_KEY,
)
_signal_generator = SignalGenerator()
_validation_engine = ValidationEngine()
_executor = OrderExecutor()
_notifier = TelegramNotifier()


def run_once(risk_manager: RiskManager, key_levels: dict[str, float]) -> None:
    """
    Execute one full pipeline cycle across all configured symbols.

    Parameters
    ----------
    risk_manager:
        Shared risk manager tracking the current portfolio state.
    key_levels:
        Mapping of symbol → key BPU/BSU price level to test.
    """
    if risk_manager.trading_halted:
        logger.warning("Trading is halted – skipping cycle")
        return

    for symbol in settings.SYMBOLS:
        logger.info("Processing symbol: %s", symbol)

        # 1. Fetch data
        candles = _data_provider.get_candles(symbol, timeframe="1h", count=100)
        if candles.empty:
            logger.warning("No candle data for %s – skipping", symbol)
            continue

        quote = _data_provider.get_quote(symbol)
        key_level = key_levels.get(symbol, float("nan"))
        if key_level != key_level:  # NaN check
            logger.warning("No key level configured for %s – skipping", symbol)
            continue

        # 2. Generate signal
        signal = _signal_generator.generate(symbol, candles, key_level)
        if signal is None or signal.signal_type == SignalType.FLAT:
            logger.info("No actionable signal for %s", symbol)
            continue

        # 3. Validate signal
        snapshot = MarketSnapshot(
            symbol=symbol,
            bid=quote.get("bid", 0.0),
            ask=quote.get("ask", 0.0),
            last_price=quote.get("last", 0.0),
            candles=candles,
            atr=signal.atr,
            key_level=key_level,
            signal_direction=signal.signal_type.value,
        )
        validation = _validation_engine.validate(snapshot)
        if not validation.passed:
            logger.info(
                "Signal for %s rejected by validation: %s",
                symbol,
                "; ".join(validation.reasons),
            )
            continue

        # Notify about valid signal
        _notifier.notify_signal(
            symbol=symbol,
            direction=signal.signal_type.value,
            entry_price=quote.get("last", 0.0),
            stop_loss=0.0,  # filled in after risk calc
            confidence=signal.confidence,
            source=signal.source,
        )

        # 4. Compute trade parameters
        trade_params = risk_manager.compute_trade_parameters(
            symbol=symbol,
            direction=signal.signal_type.value,
            entry_price=quote.get("last", 0.0),
            atr=signal.atr,
        )
        if trade_params is None:
            logger.warning("Risk manager declined trade for %s", symbol)
            continue

        # 5. Submit order
        order = _executor.submit(trade_params)
        if order is None:
            logger.error("Order submission failed for %s", symbol)
            continue

        # 6. Notify order
        _notifier.notify_order(
            symbol=symbol,
            direction=order.direction,
            size=order.size,
            entry_price=order.entry_price,
            broker_order_id=order.broker_order_id,
        )
        logger.info("Cycle complete for %s – order %s", symbol, order.broker_order_id)


def main(
    total_capital: float = 100_000.0,
    key_levels: dict[str, float] | None = None,
    loop_interval_seconds: int = 3600,
    max_iterations: int | None = None,
) -> None:
    """
    Start the trading bot loop.

    Parameters
    ----------
    total_capital:
        Starting portfolio value in account currency.
    key_levels:
        BPU/BSU levels per symbol.  When omitted a default empty mapping is used.
    loop_interval_seconds:
        Seconds to sleep between iterations (default: 3600 = 1 hour).
    max_iterations:
        Stop after this many iterations.  ``None`` means run indefinitely.
    """
    if key_levels is None:
        key_levels = {}

    risk_manager = RiskManager(total_capital=total_capital)
    logger.info(
        "Kesha trading system started | capital=%.2f | symbols=%s",
        total_capital,
        settings.SYMBOLS,
    )

    iteration = 0
    while True:
        try:
            run_once(risk_manager, key_levels)
        except Exception as exc:
            logger.exception("Unexpected error in trading cycle: %s", exc)

        iteration += 1
        if max_iterations is not None and iteration >= max_iterations:
            logger.info("Reached max_iterations=%d – shutting down", max_iterations)
            break

        logger.info("Sleeping %d seconds until next cycle …", loop_interval_seconds)
        time.sleep(loop_interval_seconds)


if __name__ == "__main__":
    main()
