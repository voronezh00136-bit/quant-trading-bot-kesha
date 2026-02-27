"""
Market Data Module
===================
Provides OHLCV candle data to the rest of the system.

In production this module talks to a real market-data API (e.g. through the
OpenClaw connection).  The interface is intentionally kept thin so that the
concrete data source can be swapped without touching the rest of the code.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)

# Column contract expected throughout the system
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


class MarketDataProvider(ABC):
    """Abstract base class for market-data providers."""

    @abstractmethod
    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int,
    ) -> pd.DataFrame:
        """
        Return the most recent *count* completed OHLCV candles for *symbol*.

        Parameters
        ----------
        symbol:
            Instrument identifier (e.g. ``"SPY"``, ``"XAUUSD"``).
        timeframe:
            Candle timeframe (e.g. ``"1h"``, ``"4h"``, ``"1d"``).
        count:
            Number of candles to retrieve.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by datetime with columns
            ``open``, ``high``, ``low``, ``close``, ``volume``.
        """

    @abstractmethod
    def get_quote(self, symbol: str) -> dict[str, float]:
        """
        Return the latest bid/ask/last quote for *symbol*.

        Returns
        -------
        dict with keys ``bid``, ``ask``, ``last``.
        """

    def validate_candles(self, candles: pd.DataFrame) -> bool:
        """Return True if *candles* has all required columns and is non-empty."""
        if candles.empty:
            logger.warning("Empty candle DataFrame received")
            return False
        missing = set(REQUIRED_COLUMNS) - set(candles.columns)
        if missing:
            logger.error("Candle DataFrame missing columns: %s", missing)
            return False
        return True


class OpenClawDataProvider(MarketDataProvider):
    """
    Market-data provider backed by the OpenClaw REST API.

    The actual HTTP calls are delegated to ``requests`` sessions so they can
    be easily mocked in tests.
    """

    def __init__(self, base_url: str, api_key: str, session=None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        if session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({"X-Api-Key": self.api_key})
        else:
            self._session = session

    def get_candles(
        self,
        symbol: str,
        timeframe: str = "1h",
        count: int = 100,
    ) -> pd.DataFrame:
        url = f"{self.base_url}/candles"
        params = {"symbol": symbol, "timeframe": timeframe, "count": count}
        try:
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data, columns=["timestamp"] + REQUIRED_COLUMNS)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as exc:
            logger.error("Failed to fetch candles for %s: %s", symbol, exc)
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

    def get_quote(self, symbol: str) -> dict[str, float]:
        url = f"{self.base_url}/quote"
        try:
            response = self._session.get(url, params={"symbol": symbol}, timeout=10)
            response.raise_for_status()
            return response.json()  # expected: {"bid": ..., "ask": ..., "last": ...}
        except Exception as exc:
            logger.error("Failed to fetch quote for %s: %s", symbol, exc)
            return {"bid": 0.0, "ask": 0.0, "last": 0.0}
