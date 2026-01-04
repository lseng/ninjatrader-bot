"""
Performance Metrics

Calculate trading performance metrics from backtest results.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from .engine import BacktestResult, Trade, Side


@dataclass
class PerformanceMetrics:
    """Comprehensive trading performance metrics."""

    # Return metrics
    total_return: float
    total_return_pct: float
    annual_return: float
    monthly_return: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # in bars
    volatility: float

    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    largest_win: float
    largest_loss: float

    # Risk-adjusted
    avg_win_loss_ratio: float
    expectancy: float
    expected_value_per_trade: float

    # Position metrics
    avg_bars_in_trade: float
    avg_bars_winning: float
    avg_bars_losing: float

    # Long/Short breakdown
    long_trades: int
    short_trades: int
    long_win_rate: float
    short_win_rate: float

    @classmethod
    def from_backtest(
        cls,
        result: BacktestResult,
        risk_free_rate: float = 0.05,
        trading_days_per_year: int = 252
    ) -> "PerformanceMetrics":
        """
        Calculate all metrics from a backtest result.

        Args:
            result: BacktestResult from backtest engine
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            trading_days_per_year: Trading days per year
        """
        trades = result.trades
        equity = result.equity_curve
        initial_capital = result.config.initial_capital

        if len(trades) == 0:
            return cls._empty_metrics()

        # Returns
        total_return = equity.iloc[-1] - initial_capital
        total_return_pct = total_return / initial_capital

        # Estimate annual return
        days = (result.end_date - result.start_date).days
        years = max(days / 365, 1/365)
        annual_return = (1 + total_return_pct) ** (1 / years) - 1
        monthly_return = (1 + annual_return) ** (1/12) - 1

        # Daily returns for risk metrics
        returns = equity.pct_change().dropna()
        volatility = returns.std() * np.sqrt(trading_days_per_year)

        # Sharpe Ratio
        excess_returns = returns.mean() * trading_days_per_year - risk_free_rate
        sharpe = excess_returns / volatility if volatility > 0 else 0

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(trading_days_per_year)
        sortino = excess_returns / downside_std if downside_std > 0 else 0

        # Max Drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_dd = abs(drawdown.min())

        # Drawdown duration
        in_drawdown = drawdown < 0
        dd_groups = (~in_drawdown).cumsum()
        dd_durations = in_drawdown.groupby(dd_groups).sum()
        max_dd_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0

        # Calmar Ratio
        calmar = annual_return / max_dd if max_dd > 0 else 0

        # Trade analysis
        pnls = [t.pnl for t in trades]
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]

        total_trades = len(trades)
        winning_trades = len(winning)
        losing_trades = len(losing)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
        avg_trade = sum(pnls) / total_trades if total_trades > 0 else 0

        largest_win = max(pnls) if pnls else 0
        largest_loss = min(pnls) if pnls else 0

        # Risk ratios
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        ev_per_trade = sum(pnls) / total_trades if total_trades > 0 else 0

        # Duration analysis
        bars = [t.bars_held for t in trades]
        winning_bars = [t.bars_held for t in winning]
        losing_bars = [t.bars_held for t in losing]

        avg_bars = np.mean(bars) if bars else 0
        avg_bars_win = np.mean(winning_bars) if winning_bars else 0
        avg_bars_lose = np.mean(losing_bars) if losing_bars else 0

        # Long/Short breakdown
        long_trades_list = [t for t in trades if t.side == Side.LONG]
        short_trades_list = [t for t in trades if t.side == Side.SHORT]

        long_count = len(long_trades_list)
        short_count = len(short_trades_list)

        long_winners = len([t for t in long_trades_list if t.pnl > 0])
        short_winners = len([t for t in short_trades_list if t.pnl > 0])

        long_wr = long_winners / long_count if long_count > 0 else 0
        short_wr = short_winners / short_count if short_count > 0 else 0

        return cls(
            total_return=total_return,
            total_return_pct=total_return_pct,
            annual_return=annual_return,
            monthly_return=monthly_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            volatility=volatility,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_win_loss_ratio=win_loss_ratio,
            expectancy=expectancy,
            expected_value_per_trade=ev_per_trade,
            avg_bars_in_trade=avg_bars,
            avg_bars_winning=avg_bars_win,
            avg_bars_losing=avg_bars_lose,
            long_trades=long_count,
            short_trades=short_count,
            long_win_rate=long_wr,
            short_win_rate=short_wr
        )

    @classmethod
    def _empty_metrics(cls) -> "PerformanceMetrics":
        """Return empty metrics when no trades."""
        return cls(
            total_return=0, total_return_pct=0, annual_return=0, monthly_return=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, max_drawdown=0,
            max_drawdown_duration=0, volatility=0, total_trades=0,
            winning_trades=0, losing_trades=0, win_rate=0, profit_factor=0,
            avg_win=0, avg_loss=0, avg_trade=0, largest_win=0, largest_loss=0,
            avg_win_loss_ratio=0, expectancy=0, expected_value_per_trade=0,
            avg_bars_in_trade=0, avg_bars_winning=0, avg_bars_losing=0,
            long_trades=0, short_trades=0, long_win_rate=0, short_win_rate=0
        )

    def summary(self) -> str:
        """Generate a formatted summary string."""
        return f"""
=== BACKTEST PERFORMANCE SUMMARY ===

RETURNS
  Total Return:      ${self.total_return:,.2f} ({self.total_return_pct:.2%})
  Annual Return:     {self.annual_return:.2%}
  Monthly Return:    {self.monthly_return:.2%}

RISK METRICS
  Sharpe Ratio:      {self.sharpe_ratio:.2f}
  Sortino Ratio:     {self.sortino_ratio:.2f}
  Calmar Ratio:      {self.calmar_ratio:.2f}
  Max Drawdown:      {self.max_drawdown:.2%}
  DD Duration:       {self.max_drawdown_duration} bars
  Volatility:        {self.volatility:.2%}

TRADE STATISTICS
  Total Trades:      {self.total_trades}
  Win Rate:          {self.win_rate:.2%}
  Profit Factor:     {self.profit_factor:.2f}
  Avg Win:           ${self.avg_win:,.2f}
  Avg Loss:          ${self.avg_loss:,.2f}
  Avg Trade:         ${self.avg_trade:,.2f}
  Largest Win:       ${self.largest_win:,.2f}
  Largest Loss:      ${self.largest_loss:,.2f}

EXPECTANCY
  Win/Loss Ratio:    {self.avg_win_loss_ratio:.2f}
  Expectancy:        ${self.expectancy:,.2f}
  EV per Trade:      ${self.expected_value_per_trade:,.2f}

POSITION ANALYSIS
  Avg Bars/Trade:    {self.avg_bars_in_trade:.1f}
  Avg Bars (Win):    {self.avg_bars_winning:.1f}
  Avg Bars (Loss):   {self.avg_bars_losing:.1f}

LONG/SHORT BREAKDOWN
  Long Trades:       {self.long_trades} ({self.long_win_rate:.2%} win rate)
  Short Trades:      {self.short_trades} ({self.short_win_rate:.2%} win rate)
"""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "annual_return": self.annual_return,
            "monthly_return": self.monthly_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "volatility": self.volatility,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "avg_trade": self.avg_trade,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "avg_win_loss_ratio": self.avg_win_loss_ratio,
            "expectancy": self.expectancy,
            "expected_value_per_trade": self.expected_value_per_trade,
            "avg_bars_in_trade": self.avg_bars_in_trade,
            "avg_bars_winning": self.avg_bars_winning,
            "avg_bars_losing": self.avg_bars_losing,
            "long_trades": self.long_trades,
            "short_trades": self.short_trades,
            "long_win_rate": self.long_win_rate,
            "short_win_rate": self.short_win_rate
        }
