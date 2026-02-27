"""
Tests for the Risk & Capital Management module.
"""
import pytest

from src.risk.manager import RiskManager, TradeParameters


class TestRiskManagerInit:
    def test_valid_init(self):
        rm = RiskManager(total_capital=10_000)
        assert rm.total_capital == 10_000
        assert not rm.trading_halted

    def test_invalid_capital_raises(self):
        with pytest.raises(ValueError):
            RiskManager(total_capital=0)
        with pytest.raises(ValueError):
            RiskManager(total_capital=-1)


class TestCurrentDrawdown:
    def test_no_drawdown_initially(self):
        rm = RiskManager(total_capital=10_000)
        assert rm.current_drawdown() == 0.0

    def test_drawdown_after_loss(self):
        rm = RiskManager(total_capital=10_000)
        rm.update_capital(9_000)
        assert abs(rm.current_drawdown() - 0.10) < 1e-9

    def test_drawdown_resets_on_new_high(self):
        rm = RiskManager(total_capital=10_000)
        rm.update_capital(9_000)
        rm.update_capital(11_000)
        assert rm.current_drawdown() == 0.0


class TestUpdateCapital:
    def test_halts_at_max_drawdown(self):
        rm = RiskManager(total_capital=10_000, max_drawdown=0.10)
        rm.update_capital(9_000)
        assert rm.trading_halted

    def test_does_not_halt_below_limit(self):
        rm = RiskManager(total_capital=10_000, max_drawdown=0.15)
        rm.update_capital(9_000)  # 10 % drawdown < 15 % limit
        assert not rm.trading_halted

    def test_resumes_after_recovery(self):
        rm = RiskManager(total_capital=10_000, max_drawdown=0.10)
        rm.update_capital(9_000)  # halted
        rm.update_capital(10_000)  # recovered – no longer halted
        assert not rm.trading_halted

    def test_invalid_capital_raises(self):
        rm = RiskManager(total_capital=10_000)
        with pytest.raises(ValueError):
            rm.update_capital(0)


class TestComputeTradeParameters:
    def test_returns_params_for_valid_long(self):
        rm = RiskManager(
            total_capital=10_000,
            max_risk_per_trade=0.01,
            atr_stop_multiplier=1.5,
        )
        params = rm.compute_trade_parameters("SPY", "long", entry_price=400.0, atr=2.0)
        assert params is not None
        assert params.symbol == "SPY"
        assert params.direction == "long"
        # stop_distance = 1.5 * 2.0 = 3.0
        assert abs(params.stop_loss_price - 397.0) < 1e-6
        # risk_amount = 10_000 * 0.01 = 100
        assert abs(params.risk_amount - 100.0) < 1e-2
        # position_size = 100 / 3.0 ≈ 33.3333
        assert abs(params.position_size - 33.3333) < 0.001

    def test_returns_params_for_valid_short(self):
        rm = RiskManager(total_capital=10_000, max_risk_per_trade=0.01, atr_stop_multiplier=1.5)
        params = rm.compute_trade_parameters("XAUUSD", "short", entry_price=2000.0, atr=5.0)
        assert params is not None
        assert params.stop_loss_price == pytest.approx(2000.0 + 1.5 * 5.0)

    def test_returns_none_when_halted(self):
        rm = RiskManager(total_capital=10_000, max_drawdown=0.10)
        rm.update_capital(9_000)  # trigger halt
        params = rm.compute_trade_parameters("SPY", "long", entry_price=400.0, atr=2.0)
        assert params is None

    def test_returns_none_for_invalid_entry_price(self):
        rm = RiskManager(total_capital=10_000)
        assert rm.compute_trade_parameters("SPY", "long", entry_price=0.0, atr=1.0) is None

    def test_returns_none_for_invalid_atr(self):
        rm = RiskManager(total_capital=10_000)
        assert rm.compute_trade_parameters("SPY", "long", entry_price=400.0, atr=0.0) is None

    def test_returns_none_for_unknown_direction(self):
        rm = RiskManager(total_capital=10_000)
        assert rm.compute_trade_parameters("SPY", "sideways", entry_price=400.0, atr=1.0) is None
