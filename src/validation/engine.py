"""
Multi-Layer Validation Engine
==============================
Filters market noise, validates BPU/BSU retest systems, and enforces strict
spread/slippage thresholds before a signal is forwarded to the executor.

Validation layers (applied in order):
  1. Spread check  – reject if the current bid/ask spread is too wide.
  2. Slippage check – reject if estimated slippage exceeds the threshold.
  3. Noise filter   – reject if recent volatility is abnormally high (IQR-based).
  4. BPU/BSU retest – validate that price has confirmed a retest of the key level.
  5. Multi-candle confirmation – require N consecutive confirming candles.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    """Lightweight container for the current market state."""

    symbol: str
    bid: float
    ask: float
    last_price: float
    # OHLCV candles, most-recent last (DataFrame with columns open/high/low/close/volume)
    candles: pd.DataFrame
    # Pre-computed ATR value for the current bar
    atr: float
    # Key support/resistance level being tested (BPU upper or BSL lower boundary)
    key_level: float
    signal_direction: str  # "long" | "short"


@dataclass
class ValidationResult:
    """Outcome of one full validation pass."""

    passed: bool
    symbol: str
    reasons: list[str] = field(default_factory=list)

    def reject(self, reason: str) -> "ValidationResult":
        self.passed = False
        self.reasons.append(reason)
        return self


class ValidationEngine:
    """
    Runs all validation layers against a :class:`MarketSnapshot` and returns
    a :class:`ValidationResult`.
    """

    def __init__(
        self,
        max_spread_fraction: float = settings.MAX_SPREAD_FRACTION,
        max_slippage: float = settings.MAX_SLIPPAGE,
        confirmation_candles: int = settings.SIGNAL_CONFIRMATION_CANDLES,
        noise_iqr_multiplier: float = 3.0,
        retest_tolerance: float = 0.002,
    ) -> None:
        self.max_spread_fraction = max_spread_fraction
        self.max_slippage = max_slippage
        self.confirmation_candles = confirmation_candles
        self.noise_iqr_multiplier = noise_iqr_multiplier
        self.retest_tolerance = retest_tolerance  # fraction of price

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, snapshot: MarketSnapshot) -> ValidationResult:
        result = ValidationResult(passed=True, symbol=snapshot.symbol)
        checks = [
            self._check_spread,
            self._check_slippage,
            self._check_noise,
            self._check_bpu_bsu_retest,
            self._check_multi_candle_confirmation,
        ]
        for check in checks:
            check(snapshot, result)
            if not result.passed:
                logger.warning(
                    "Validation failed for %s: %s",
                    snapshot.symbol,
                    result.reasons[-1],
                )
                return result
        logger.info("Validation passed for %s", snapshot.symbol)
        return result

    # ------------------------------------------------------------------
    # Individual validation layers
    # ------------------------------------------------------------------

    def _check_spread(self, snap: MarketSnapshot, result: ValidationResult) -> None:
        if snap.ask <= 0:
            result.reject("Ask price must be positive")
            return
        spread_fraction = (snap.ask - snap.bid) / snap.ask
        if spread_fraction > self.max_spread_fraction:
            result.reject(
                f"Spread too wide: {spread_fraction:.5f} > {self.max_spread_fraction:.5f}"
            )

    def _check_slippage(self, snap: MarketSnapshot, result: ValidationResult) -> None:
        mid = (snap.bid + snap.ask) / 2.0
        slippage = abs(snap.last_price - mid)
        if slippage > self.max_slippage:
            result.reject(
                f"Slippage too high: {slippage:.4f} > {self.max_slippage:.4f}"
            )

    def _check_noise(self, snap: MarketSnapshot, result: ValidationResult) -> None:
        """Reject if recent bar ranges are abnormally large (IQR-based outlier test)."""
        if snap.candles.empty or len(snap.candles) < 10:
            return  # not enough history – pass through
        ranges = snap.candles["high"] - snap.candles["low"]
        q1, q3 = float(np.percentile(ranges, 25)), float(np.percentile(ranges, 75))
        iqr = q3 - q1
        upper_fence = q3 + self.noise_iqr_multiplier * iqr
        last_range = float(ranges.iloc[-1])
        if last_range > upper_fence:
            result.reject(
                f"Market noise too high: last range {last_range:.4f} > fence {upper_fence:.4f}"
            )

    def _check_bpu_bsu_retest(
        self, snap: MarketSnapshot, result: ValidationResult
    ) -> None:
        """
        Confirm that the most-recent close has retested the key BPU/BSU level
        within the tolerance band.
        """
        if snap.candles.empty:
            result.reject("No candle data for BPU/BSU retest check")
            return
        last_close = float(snap.candles["close"].iloc[-1])
        tolerance = snap.key_level * self.retest_tolerance
        distance = abs(last_close - snap.key_level)
        if distance > tolerance:
            result.reject(
                f"BPU/BSU retest not confirmed: distance {distance:.4f} > tolerance {tolerance:.4f}"
            )

    def _check_multi_candle_confirmation(
        self, snap: MarketSnapshot, result: ValidationResult
    ) -> None:
        """
        Require `confirmation_candles` consecutive bullish (long) or bearish
        (short) closing candles above/below the key level.
        """
        if len(snap.candles) < self.confirmation_candles:
            result.reject(
                f"Insufficient candle history for multi-candle confirmation "
                f"(need {self.confirmation_candles})"
            )
            return
        recent = snap.candles.tail(self.confirmation_candles)
        if snap.signal_direction == "long":
            confirmed = all(recent["close"] > recent["open"])
        elif snap.signal_direction == "short":
            confirmed = all(recent["close"] < recent["open"])
        else:
            result.reject(f"Unknown signal direction: {snap.signal_direction!r}")
            return
        if not confirmed:
            result.reject(
                f"Multi-candle confirmation failed for {snap.signal_direction} "
                f"over last {self.confirmation_candles} candles"
            )
