"""
Quant Trading Bot - ML Signal Generation
Generates trading signals using XGBoost and LightGBM models.
Author: Aleksandr Gvozdkov
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logger = logging.getLogger(__name__)


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    signal: Signal
    confidence: float
    win_probability: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None


class MLSignalGenerator:
    """
    ML-based trading signal generator using XGBoost and LightGBM.
    Uses a probabilistic approach with Kelly criterion for position sizing.
    """
    
    def __init__(self, model_path: str = 'models/best_model.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = StandardScaler()
        self.min_confidence = 0.60  # Minimum win probability to trade
        self.kelly_fraction = 0.25   # Fractional Kelly for risk management
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical features for signal generation."""
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Momentum indicators
        df['rsi'] = self._compute_rsi(df['close'], 14)
        df['macd'], df['macd_signal'] = self._compute_macd(df['close'])
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Volatility
        df['atr'] = self._compute_atr(df['high'], df['low'], df['close'], 14)
        df['volatility_20'] = df['returns'].rolling(20).std()
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Trend strength
        df['adx'] = self._compute_adx(df['high'], df['low'], df['close'], 14)
        
        # Price position
        df['price_position_20'] = (df['close'] - df['close'].rolling(20).min()) / (
            df['close'].rolling(20).max() - df['close'].rolling(20).min()
        )
        
        return df.fillna(0)
    
    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        """Generate a trading signal for the latest bar."""
        df = self.compute_features(df.copy())
        
        feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        X = df[feature_cols].iloc[-1:].values
        
        # Get win probability
        win_prob = self.model.predict_proba(X)[0][1]
        lose_prob = 1 - win_prob
        
        # Kelly criterion position sizing
        kelly = (win_prob - lose_prob / (2.0 - 1)) / 1.0  # Assuming 1:1 reward/risk initially
        position_size = max(0, kelly * self.kelly_fraction)
        
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        # Generate signal based on win probability
        if win_prob >= self.min_confidence:
            signal = Signal.BUY
            stop_loss = current_price - (2 * atr)
            take_profit = current_price + (3 * atr)
            
            # Refine Kelly with actual R:R
            rr_ratio = (take_profit - current_price) / (current_price - stop_loss)
            kelly = (win_prob - lose_prob / rr_ratio) / rr_ratio
            position_size = max(0, kelly * self.kelly_fraction)
            
        elif win_prob <= (1 - self.min_confidence):
            signal = Signal.SELL
            stop_loss = current_price + (2 * atr)
            take_profit = current_price - (3 * atr)
            position_size = max(0, kelly * self.kelly_fraction)
        else:
            signal = Signal.HOLD
            stop_loss = None
            take_profit = None
            position_size = 0.0
        
        return TradingSignal(
            signal=signal,
            confidence=max(win_prob, 1 - win_prob),
            win_probability=win_prob,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=min(position_size, 0.20)  # Max 20% of portfolio
        )
    
    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line
    
    @staticmethod
    def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    @staticmethod
    def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = MLSignalGenerator._compute_atr(high, low, close, 1)
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
        minus_di = 100 * (minus_dm.abs().rolling(period).mean() / tr.rolling(period).mean())
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(period).mean()
