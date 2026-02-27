"""
Tests for the Multi-Layer Validation Engine.
"""
import numpy as np
import pandas as pd
import pytest

from src.validation.engine import MarketSnapshot, ValidationEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candles(n: int = 20, bullish: bool = True) -> pd.DataFrame:
    """Return a simple DataFrame of synthetic OHLCV candles."""
    rows = []
    price = 100.0
    for _ in range(n):
        if bullish:
            o, c = price, price + 0.5
        else:
            o, c = price + 0.5, price
        rows.append({"open": o, "high": c + 0.2, "low": o - 0.2, "close": c, "volume": 1000})
        price = c
    return pd.DataFrame(rows)


def _make_snapshot(
    bid: float = 100.0,
    ask: float = 100.05,
    last: float = 100.02,
    atr: float = 1.0,
    key_level: float = 100.0,
    direction: str = "long",
    candles: pd.DataFrame | None = None,
) -> MarketSnapshot:
    if candles is None:
        candles = _make_candles(bullish=(direction == "long"))
    return MarketSnapshot(
        symbol="TEST",
        bid=bid,
        ask=ask,
        last_price=last,
        candles=candles,
        atr=atr,
        key_level=key_level,
        signal_direction=direction,
    )


# ---------------------------------------------------------------------------
# Spread check
# ---------------------------------------------------------------------------

class TestSpreadCheck:
    def test_passes_tight_spread(self):
        # Use a loose retest tolerance so only the spread layer is exercised.
        engine = ValidationEngine(max_spread_fraction=0.001, retest_tolerance=0.5)
        snap = _make_snapshot(bid=100.0, ask=100.05)  # 0.05 % spread
        result = engine.validate(snap)
        assert result.passed

    def test_fails_wide_spread(self):
        engine = ValidationEngine(max_spread_fraction=0.0001)
        snap = _make_snapshot(bid=100.0, ask=101.0)  # 1 % spread
        result = engine.validate(snap)
        assert not result.passed
        assert any("Spread" in r for r in result.reasons)

    def test_fails_zero_ask(self):
        engine = ValidationEngine()
        snap = _make_snapshot(bid=0.0, ask=0.0, last=0.0)
        result = engine.validate(snap)
        assert not result.passed


# ---------------------------------------------------------------------------
# Slippage check
# ---------------------------------------------------------------------------

class TestSlippageCheck:
    def test_passes_low_slippage(self):
        # Use a loose retest tolerance so only the slippage layer is exercised.
        engine = ValidationEngine(max_spread_fraction=0.01, max_slippage=1.0, retest_tolerance=0.5)
        snap = _make_snapshot(bid=100.0, ask=100.10, last=100.05)
        result = engine.validate(snap)
        assert result.passed

    def test_fails_high_slippage(self):
        engine = ValidationEngine(max_spread_fraction=0.01, max_slippage=0.01)
        # mid = 100.05, last = 102.0  →  slippage = 1.95
        snap = _make_snapshot(bid=100.0, ask=100.10, last=102.0)
        result = engine.validate(snap)
        assert not result.passed
        assert any("Slippage" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# Noise filter
# ---------------------------------------------------------------------------

class TestNoiseFilter:
    def test_passes_normal_volatility(self):
        engine = ValidationEngine(
            max_spread_fraction=0.01,
            max_slippage=100,
        )
        candles = _make_candles(30)
        snap = _make_snapshot(candles=candles)
        result = engine.validate(snap)
        # Should not fail on noise with uniform candles
        noise_rejected = any("noise" in r.lower() for r in result.reasons)
        assert not noise_rejected

    def test_fails_extreme_last_range(self):
        engine = ValidationEngine(
            max_spread_fraction=0.01,
            max_slippage=100,
            noise_iqr_multiplier=1.0,
        )
        rows = [{"open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000}] * 19
        # Last bar has a hugely abnormal range
        rows.append({"open": 100, "high": 200, "low": 50, "close": 101, "volume": 1000})
        candles = pd.DataFrame(rows)
        snap = _make_snapshot(candles=candles)
        result = engine.validate(snap)
        assert not result.passed
        assert any("noise" in r.lower() for r in result.reasons)


# ---------------------------------------------------------------------------
# BPU/BSU retest check
# ---------------------------------------------------------------------------

class TestBpuBsuRetest:
    def test_passes_within_tolerance(self):
        engine = ValidationEngine(
            max_spread_fraction=0.01,
            max_slippage=100,
            noise_iqr_multiplier=10,
            retest_tolerance=0.01,  # 1 %
        )
        candles = _make_candles(20, bullish=True)
        last_close = float(candles["close"].iloc[-1])
        # Key level very close to last close
        key_level = last_close * 1.001
        snap = _make_snapshot(key_level=key_level, candles=candles)
        result = engine.validate(snap)
        retest_rejected = any("BPU" in r or "BSU" in r for r in result.reasons)
        assert not retest_rejected

    def test_fails_far_from_level(self):
        engine = ValidationEngine(
            max_spread_fraction=0.01,
            max_slippage=100,
            noise_iqr_multiplier=10,
            retest_tolerance=0.0001,  # very tight
        )
        candles = _make_candles(20, bullish=True)
        last_close = float(candles["close"].iloc[-1])
        # Key level far from last close
        key_level = last_close * 1.10
        snap = _make_snapshot(key_level=key_level, candles=candles)
        result = engine.validate(snap)
        assert not result.passed
        assert any("BPU" in r or "BSU" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# Multi-candle confirmation
# ---------------------------------------------------------------------------

class TestMultiCandleConfirmation:
    def test_passes_bullish_confirmation(self):
        engine = ValidationEngine(
            max_spread_fraction=0.01,
            max_slippage=100,
            noise_iqr_multiplier=10,
            retest_tolerance=0.05,
            confirmation_candles=2,
        )
        candles = _make_candles(20, bullish=True)
        snap = _make_snapshot(direction="long", candles=candles)
        result = engine.validate(snap)
        conf_rejected = any("Multi-candle" in r for r in result.reasons)
        assert not conf_rejected

    def test_fails_bearish_when_expecting_long(self):
        engine = ValidationEngine(
            max_spread_fraction=0.01,
            max_slippage=100,
            noise_iqr_multiplier=10,
            retest_tolerance=0.05,
            confirmation_candles=2,
        )
        candles = _make_candles(20, bullish=False)
        snap = _make_snapshot(direction="long", candles=candles)
        result = engine.validate(snap)
        assert not result.passed
        assert any("Multi-candle" in r for r in result.reasons)

    def test_fails_insufficient_candles(self):
        engine = ValidationEngine(
            max_spread_fraction=0.01,
            max_slippage=100,
            noise_iqr_multiplier=10,
            retest_tolerance=0.05,
            confirmation_candles=5,
        )
        candles = _make_candles(3)
        snap = _make_snapshot(candles=candles)
        result = engine.validate(snap)
        assert not result.passed

    def test_fails_unknown_direction(self):
        engine = ValidationEngine(
            max_spread_fraction=0.01,
            max_slippage=100,
            noise_iqr_multiplier=10,
            retest_tolerance=0.05,
            confirmation_candles=2,
        )
        candles = _make_candles(20, bullish=True)
        snap = _make_snapshot(direction="sideways", candles=candles)
        result = engine.validate(snap)
        assert not result.passed
