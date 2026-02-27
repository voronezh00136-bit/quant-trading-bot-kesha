"""
Tests for the Signal Generation module.
"""
import numpy as np
import pandas as pd
import pytest

from src.signals.generator import SignalGenerator, SignalType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candles(n: int = 30, bullish: bool = True, base: float = 100.0) -> pd.DataFrame:
    rows = []
    price = base
    for _ in range(n):
        if bullish:
            o, c = price, price + 0.5
            h, l = c + 0.2, o - 0.2
        else:
            o, c = price + 0.5, price
            h, l = o + 0.2, c - 0.2
        rows.append({"open": o, "high": h, "low": l, "close": c, "volume": 1000})
        price = c
    return pd.DataFrame(rows)


class TestSignalGeneratorBasic:
    def test_returns_none_for_empty_candles(self):
        gen = SignalGenerator()
        result = gen.generate("TEST", pd.DataFrame(), key_level=100.0)
        assert result is None

    def test_returns_none_for_insufficient_history(self):
        gen = SignalGenerator(atr_period=14, momentum_candles=3)
        candles = _make_candles(5)
        result = gen.generate("TEST", candles, key_level=100.0)
        assert result is None

    def test_returns_none_for_missing_columns(self):
        gen = SignalGenerator()
        candles = pd.DataFrame({"open": [1, 2, 3], "close": [2, 3, 4]})
        result = gen.generate("TEST", candles, key_level=2.0)
        assert result is None


class TestFalseBreakoutDetection:
    def test_bullish_false_breakout(self):
        gen = SignalGenerator(atr_period=5, false_breakout_atr_threshold=0.1, momentum_candles=1)
        rows = [{"open": 100, "high": 101, "low": 99.5, "close": 100.5, "volume": 1000}] * 10
        # Previous bar: low dipped below key_level=100, closed above
        rows.append({"open": 100.5, "high": 101.0, "low": 99.8, "close": 100.6, "volume": 1000})
        # Current bar: closes above key_level by more than threshold
        rows.append({"open": 100.6, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 1000})
        candles = pd.DataFrame(rows)
        result = gen.generate("TEST", candles, key_level=100.0)
        assert result is not None
        assert result.signal_type == SignalType.LONG
        assert result.source == "false_breakout_bullish"

    def test_bearish_false_breakout(self):
        gen = SignalGenerator(atr_period=5, false_breakout_atr_threshold=0.1, momentum_candles=1)
        rows = [{"open": 100, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000}] * 10
        # Previous bar: high exceeded key_level=101, closed below
        rows.append({"open": 100.5, "high": 101.2, "low": 100.0, "close": 100.4, "volume": 1000})
        # Current bar: closes below key_level by more than threshold
        rows.append({"open": 100.4, "high": 100.8, "low": 99.0, "close": 99.5, "volume": 1000})
        candles = pd.DataFrame(rows)
        result = gen.generate("TEST", candles, key_level=101.0)
        assert result is not None
        assert result.signal_type == SignalType.SHORT
        assert result.source == "false_breakout_bearish"


class TestMomentumDetection:
    def test_bullish_momentum(self):
        gen = SignalGenerator(atr_period=5, false_breakout_atr_threshold=100, momentum_candles=3)
        candles = _make_candles(30, bullish=True, base=100.0)
        key_level = 95.0  # well below last close
        result = gen.generate("TEST", candles, key_level=key_level)
        assert result is not None
        assert result.signal_type == SignalType.LONG

    def test_bearish_momentum(self):
        gen = SignalGenerator(atr_period=5, false_breakout_atr_threshold=100, momentum_candles=3)
        candles = _make_candles(30, bullish=False, base=100.0)
        key_level = 110.0  # well above last close
        result = gen.generate("TEST", candles, key_level=key_level)
        assert result is not None
        assert result.signal_type == SignalType.SHORT


class TestConfidenceRange:
    def test_confidence_within_bounds(self):
        gen = SignalGenerator(atr_period=5, momentum_candles=3)
        candles = _make_candles(30, bullish=True)
        result = gen.generate("TEST", candles, key_level=95.0)
        if result is not None:
            assert 0.0 <= result.confidence <= 1.0
