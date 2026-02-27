"""
Signal Generation
==================
Translates raw OHLCV candle data into structured trading signals.

Detects:
  • False breakouts (price briefly breaches a level but closes back inside).
  • Multi-candle momentum confirmation patterns.
  • ATR-normalised trend direction.

The module is intentionally broker-agnostic; it only analyses data and emits
:class:`Signal` objects.  Downstream consumers (validation engine, executor)
decide whether to act on them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Signal:
    """A single trading signal emitted by the generator."""

    symbol: str
    signal_type: SignalType
    key_level: float        # BPU upper / BSU lower boundary being tested
    atr: float              # ATR at signal time
    confidence: float       # 0.0 – 1.0 composite score
    source: str             # human-readable description of the pattern matched


class SignalGenerator:
    """
    Generates :class:`Signal` objects from OHLCV candle DataFrames.

    Parameters
    ----------
    atr_period:
        Number of bars used to compute the ATR (default 14).
    false_breakout_atr_threshold:
        How many ATR units a close must retrace back inside a level to qualify
        as a confirmed false breakout (default 0.5).
    momentum_candles:
        Minimum number of consecutive same-direction candles required for a
        momentum signal (default 3).
    """

    def __init__(
        self,
        atr_period: int = 14,
        false_breakout_atr_threshold: float = 0.5,
        momentum_candles: int = 3,
    ) -> None:
        self.atr_period = atr_period
        self.false_breakout_atr_threshold = false_breakout_atr_threshold
        self.momentum_candles = momentum_candles

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        symbol: str,
        candles: pd.DataFrame,
        key_level: float,
    ) -> Signal | None:
        """
        Analyse *candles* and return the strongest :class:`Signal`, or
        ``None`` if no pattern qualifies.

        Parameters
        ----------
        symbol:
            Ticker/instrument identifier.
        candles:
            DataFrame with columns ``open``, ``high``, ``low``, ``close``,
            ``volume``.  Most-recent bar must be last.
        key_level:
            Relevant BPU/BSU price level to test against.
        """
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(candles.columns):
            logger.error("Candle DataFrame missing required columns for %s", symbol)
            return None
        if len(candles) < self.atr_period + self.momentum_candles:
            logger.warning("Insufficient candle history for %s", symbol)
            return None

        candles = candles.copy()
        candles["atr"] = self._compute_atr(candles)
        current_atr = float(candles["atr"].iloc[-1])
        if np.isnan(current_atr) or current_atr <= 0:
            logger.warning("ATR unavailable for %s", symbol)
            return None

        # Candidate signals (collect all that fire, pick best confidence)
        candidates: list[Signal] = []

        fb = self._detect_false_breakout(symbol, candles, key_level, current_atr)
        if fb:
            candidates.append(fb)

        mo = self._detect_momentum(symbol, candles, key_level, current_atr)
        if mo:
            candidates.append(mo)

        if not candidates:
            return None

        best = max(candidates, key=lambda s: s.confidence)
        logger.info(
            "Signal generated for %s: %s (confidence=%.2f, source=%s)",
            symbol,
            best.signal_type.value,
            best.confidence,
            best.source,
        )
        return best

    # ------------------------------------------------------------------
    # ATR computation
    # ------------------------------------------------------------------

    def _compute_atr(self, candles: pd.DataFrame) -> pd.Series:
        high = candles["high"]
        low = candles["low"]
        prev_close = candles["close"].shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(self.atr_period).mean()

    # ------------------------------------------------------------------
    # Pattern detectors
    # ------------------------------------------------------------------

    def _detect_false_breakout(
        self,
        symbol: str,
        candles: pd.DataFrame,
        key_level: float,
        atr: float,
    ) -> Signal | None:
        """
        A false breakout occurs when the previous bar's high (for short) or
        low (for long) breaches *key_level*, but the bar closes back on the
        opposite side, and the current bar confirms by closing further away.
        """
        if len(candles) < 2:
            return None
        prev = candles.iloc[-2]
        curr = candles.iloc[-1]
        threshold = self.false_breakout_atr_threshold * atr

        # Bullish false breakout (BSU retest): prev low dipped below level,
        # closed above; current bar also closes above.
        if (
            prev["low"] < key_level
            and prev["close"] > key_level
            and curr["close"] > key_level
            and (curr["close"] - key_level) > threshold
        ):
            confidence = min(
                1.0, (curr["close"] - key_level) / (atr if atr > 0 else 1)
            )
            return Signal(
                symbol=symbol,
                signal_type=SignalType.LONG,
                key_level=key_level,
                atr=atr,
                confidence=round(confidence, 3),
                source="false_breakout_bullish",
            )

        # Bearish false breakout (BPU retest): prev high exceeded level,
        # closed below; current bar also closes below.
        if (
            prev["high"] > key_level
            and prev["close"] < key_level
            and curr["close"] < key_level
            and (key_level - curr["close"]) > threshold
        ):
            confidence = min(
                1.0, (key_level - curr["close"]) / (atr if atr > 0 else 1)
            )
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SHORT,
                key_level=key_level,
                atr=atr,
                confidence=round(confidence, 3),
                source="false_breakout_bearish",
            )
        return None

    def _detect_momentum(
        self,
        symbol: str,
        candles: pd.DataFrame,
        key_level: float,
        atr: float,
    ) -> Signal | None:
        """
        Detects N consecutive bullish/bearish candles closing on the correct
        side of *key_level*.
        """
        recent = candles.tail(self.momentum_candles)
        all_bullish = all(recent["close"] > recent["open"])
        all_bearish = all(recent["close"] < recent["open"])
        last_close = float(candles["close"].iloc[-1])

        if all_bullish and last_close > key_level:
            strength = (last_close - key_level) / (atr if atr > 0 else 1)
            confidence = min(1.0, 0.5 + 0.1 * self.momentum_candles + strength * 0.2)
            return Signal(
                symbol=symbol,
                signal_type=SignalType.LONG,
                key_level=key_level,
                atr=atr,
                confidence=round(confidence, 3),
                source="momentum_bullish",
            )

        if all_bearish and last_close < key_level:
            strength = (key_level - last_close) / (atr if atr > 0 else 1)
            confidence = min(1.0, 0.5 + 0.1 * self.momentum_candles + strength * 0.2)
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SHORT,
                key_level=key_level,
                atr=atr,
                confidence=round(confidence, 3),
                source="momentum_bearish",
            )
        return None
