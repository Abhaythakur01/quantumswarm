"""
Performance metrics for backtesting evaluation.
Calculates standard financial metrics for strategy assessment.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PerformanceMetrics:
    """
    Calculate and store performance metrics.

    Metrics include:
    - Returns: total, annualized, cumulative
    - Risk: volatility, max drawdown, VaR, CVaR
    - Risk-adjusted: Sharpe, Sortino, Calmar
    - Trade: win rate, profit factor, expectancy
    """

    returns: np.ndarray
    equity_curve: np.ndarray
    risk_free_rate: float = 0.02
    trading_days: int = 252

    def __post_init__(self):
        """Compute all metrics on initialization."""
        self.returns = np.array(self.returns)
        self.equity_curve = np.array(self.equity_curve)

        # Handle edge cases
        if len(self.returns) == 0:
            self.returns = np.array([0.0])
        if len(self.equity_curve) == 0:
            self.equity_curve = np.array([1.0])

    @property
    def total_return(self) -> float:
        """Total cumulative return."""
        if len(self.equity_curve) < 2:
            return 0.0
        return (self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0]

    @property
    def annualized_return(self) -> float:
        """Annualized return (CAGR)."""
        if len(self.equity_curve) < 2:
            return 0.0

        n_periods = len(self.equity_curve)
        years = n_periods / self.trading_days

        if years <= 0:
            return 0.0

        total = self.total_return
        if total <= -1:
            return -1.0

        return (1 + total) ** (1 / years) - 1

    @property
    def volatility(self) -> float:
        """Annualized volatility (standard deviation of returns)."""
        if len(self.returns) < 2:
            return 0.0
        return np.std(self.returns, ddof=1) * np.sqrt(self.trading_days)

    @property
    def downside_volatility(self) -> float:
        """Annualized downside volatility (for Sortino ratio)."""
        negative_returns = self.returns[self.returns < 0]
        if len(negative_returns) < 2:
            return 0.0
        return np.std(negative_returns, ddof=1) * np.sqrt(self.trading_days)

    @property
    def sharpe_ratio(self) -> float:
        """Sharpe ratio (risk-adjusted return)."""
        excess_return = self.annualized_return - self.risk_free_rate
        vol = self.volatility

        if vol == 0:
            return 0.0

        return excess_return / vol

    @property
    def sortino_ratio(self) -> float:
        """Sortino ratio (downside risk-adjusted return)."""
        excess_return = self.annualized_return - self.risk_free_rate
        downside_vol = self.downside_volatility

        if downside_vol == 0:
            return 0.0 if excess_return <= 0 else float("inf")

        return excess_return / downside_vol

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown (peak to trough decline)."""
        if len(self.equity_curve) < 2:
            return 0.0

        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (self.equity_curve - peak) / peak
        return abs(np.min(drawdown))

    @property
    def calmar_ratio(self) -> float:
        """Calmar ratio (return / max drawdown)."""
        mdd = self.max_drawdown
        if mdd == 0:
            return 0.0 if self.annualized_return <= 0 else float("inf")

        return self.annualized_return / mdd

    @property
    def var_95(self) -> float:
        """Value at Risk at 95% confidence level."""
        if len(self.returns) < 2:
            return 0.0
        return np.percentile(self.returns, 5)

    @property
    def cvar_95(self) -> float:
        """Conditional VaR (Expected Shortfall) at 95%."""
        if len(self.returns) < 2:
            return 0.0
        var = self.var_95
        return np.mean(self.returns[self.returns <= var])

    @property
    def skewness(self) -> float:
        """Skewness of returns distribution."""
        if len(self.returns) < 3:
            return 0.0
        return stats.skew(self.returns)

    @property
    def kurtosis(self) -> float:
        """Excess kurtosis of returns distribution."""
        if len(self.returns) < 4:
            return 0.0
        return stats.kurtosis(self.returns)

    @property
    def positive_days(self) -> int:
        """Number of days with positive returns."""
        return np.sum(self.returns > 0)

    @property
    def negative_days(self) -> int:
        """Number of days with negative returns."""
        return np.sum(self.returns < 0)

    @property
    def win_rate(self) -> float:
        """Percentage of days with positive returns."""
        total = self.positive_days + self.negative_days
        if total == 0:
            return 0.0
        return self.positive_days / total

    @property
    def best_day(self) -> float:
        """Best single day return."""
        if len(self.returns) == 0:
            return 0.0
        return np.max(self.returns)

    @property
    def worst_day(self) -> float:
        """Worst single day return."""
        if len(self.returns) == 0:
            return 0.0
        return np.min(self.returns)

    @property
    def avg_daily_return(self) -> float:
        """Average daily return."""
        if len(self.returns) == 0:
            return 0.0
        return np.mean(self.returns)

    def profit_factor(self, trades: list) -> float:
        """
        Profit factor (gross profit / gross loss).

        Args:
            trades: List of Trade objects

        Returns:
            Profit factor
        """
        if not trades:
            return 0.0

        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def get_drawdown_series(self) -> np.ndarray:
        """Get the drawdown series."""
        peak = np.maximum.accumulate(self.equity_curve)
        return (self.equity_curve - peak) / peak

    def get_underwater_periods(self) -> list[dict]:
        """Get periods where portfolio was underwater (below peak)."""
        drawdowns = self.get_drawdown_series()
        periods = []

        in_drawdown = False
        start_idx = 0

        for i, dd in enumerate(drawdowns):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                periods.append({
                    "start": start_idx,
                    "end": i,
                    "duration": i - start_idx,
                    "max_drawdown": np.min(drawdowns[start_idx:i]),
                })

        # Handle ongoing drawdown
        if in_drawdown:
            periods.append({
                "start": start_idx,
                "end": len(drawdowns) - 1,
                "duration": len(drawdowns) - start_idx,
                "max_drawdown": np.min(drawdowns[start_idx:]),
            })

        return periods

    def rolling_sharpe(self, window: int = 60) -> np.ndarray:
        """Calculate rolling Sharpe ratio."""
        if len(self.returns) < window:
            return np.array([])

        rolling_mean = pd.Series(self.returns).rolling(window).mean()
        rolling_std = pd.Series(self.returns).rolling(window).std()

        daily_rf = self.risk_free_rate / self.trading_days
        rolling_sharpe = (rolling_mean - daily_rf) / rolling_std * np.sqrt(self.trading_days)

        return rolling_sharpe.values

    def to_dict(self) -> dict:
        """Export all metrics as dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "win_rate": self.win_rate,
            "best_day": self.best_day,
            "worst_day": self.worst_day,
            "avg_daily_return": self.avg_daily_return,
            "positive_days": self.positive_days,
            "negative_days": self.negative_days,
        }

    def summary(self) -> str:
        """Generate a text summary of metrics."""
        lines = [
            "=" * 50,
            "PERFORMANCE SUMMARY",
            "=" * 50,
            "",
            "Returns:",
            f"  Total Return:      {self.total_return:>10.2%}",
            f"  Annualized Return: {self.annualized_return:>10.2%}",
            f"  Avg Daily Return:  {self.avg_daily_return:>10.4%}",
            "",
            "Risk:",
            f"  Volatility:        {self.volatility:>10.2%}",
            f"  Max Drawdown:      {self.max_drawdown:>10.2%}",
            f"  VaR (95%):         {self.var_95:>10.4%}",
            f"  CVaR (95%):        {self.cvar_95:>10.4%}",
            "",
            "Risk-Adjusted:",
            f"  Sharpe Ratio:      {self.sharpe_ratio:>10.2f}",
            f"  Sortino Ratio:     {self.sortino_ratio:>10.2f}",
            f"  Calmar Ratio:      {self.calmar_ratio:>10.2f}",
            "",
            "Trading Days:",
            f"  Positive Days:     {self.positive_days:>10d}",
            f"  Negative Days:     {self.negative_days:>10d}",
            f"  Win Rate:          {self.win_rate:>10.2%}",
            f"  Best Day:          {self.best_day:>10.2%}",
            f"  Worst Day:         {self.worst_day:>10.2%}",
            "",
            "Distribution:",
            f"  Skewness:          {self.skewness:>10.2f}",
            f"  Kurtosis:          {self.kurtosis:>10.2f}",
            "=" * 50,
        ]

        return "\n".join(lines)


class TradeMetrics:
    """
    Calculate trade-level metrics.
    """

    def __init__(self, trades: list):
        """
        Initialize with list of Trade objects.

        Args:
            trades: List of closed trades
        """
        self.trades = [t for t in trades if t.is_closed]
        self.winning = [t for t in self.trades if t.pnl > 0]
        self.losing = [t for t in self.trades if t.pnl <= 0]

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return len(self.winning) / self.total_trades

    @property
    def avg_win(self) -> float:
        if not self.winning:
            return 0.0
        return np.mean([t.pnl for t in self.winning])

    @property
    def avg_loss(self) -> float:
        if not self.losing:
            return 0.0
        return np.mean([t.pnl for t in self.losing])

    @property
    def largest_win(self) -> float:
        if not self.winning:
            return 0.0
        return max(t.pnl for t in self.winning)

    @property
    def largest_loss(self) -> float:
        if not self.losing:
            return 0.0
        return min(t.pnl for t in self.losing)

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.winning)
        gross_loss = abs(sum(t.pnl for t in self.losing))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    @property
    def expectancy(self) -> float:
        """Expected value per trade."""
        if self.total_trades == 0:
            return 0.0
        return sum(t.pnl for t in self.trades) / self.total_trades

    @property
    def payoff_ratio(self) -> float:
        """Average win / average loss."""
        if self.avg_loss == 0:
            return float("inf") if self.avg_win > 0 else 0.0
        return abs(self.avg_win / self.avg_loss)

    @property
    def avg_holding_period(self) -> float:
        if not self.trades:
            return 0.0
        return np.mean([t.holding_period for t in self.trades])

    @property
    def max_consecutive_wins(self) -> int:
        return self._max_consecutive(lambda t: t.pnl > 0)

    @property
    def max_consecutive_losses(self) -> int:
        return self._max_consecutive(lambda t: t.pnl <= 0)

    def _max_consecutive(self, condition) -> int:
        if not self.trades:
            return 0

        max_streak = 0
        current_streak = 0

        for trade in self.trades:
            if condition(trade):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "winning_trades": len(self.winning),
            "losing_trades": len(self.losing),
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "payoff_ratio": self.payoff_ratio,
            "avg_holding_period": self.avg_holding_period,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
        }

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "TRADE STATISTICS",
            "=" * 50,
            "",
            f"Total Trades:          {self.total_trades:>10d}",
            f"Winning Trades:        {len(self.winning):>10d}",
            f"Losing Trades:         {len(self.losing):>10d}",
            f"Win Rate:              {self.win_rate:>10.2%}",
            "",
            f"Average Win:           ${self.avg_win:>10,.2f}",
            f"Average Loss:          ${self.avg_loss:>10,.2f}",
            f"Largest Win:           ${self.largest_win:>10,.2f}",
            f"Largest Loss:          ${self.largest_loss:>10,.2f}",
            "",
            f"Profit Factor:         {self.profit_factor:>10.2f}",
            f"Expectancy:            ${self.expectancy:>10,.2f}",
            f"Payoff Ratio:          {self.payoff_ratio:>10.2f}",
            "",
            f"Avg Holding Period:    {self.avg_holding_period:>10.1f} bars",
            f"Max Consecutive Wins:  {self.max_consecutive_wins:>10d}",
            f"Max Consecutive Losses:{self.max_consecutive_losses:>10d}",
            "=" * 50,
        ]

        return "\n".join(lines)


def compare_strategies(results: dict[str, dict]) -> pd.DataFrame:
    """
    Compare multiple backtest results.

    Args:
        results: Dictionary mapping strategy name to backtest results

    Returns:
        DataFrame with comparison metrics
    """
    comparison = []

    for name, result in results.items():
        metrics = {
            "Strategy": name,
            "Total Return": result.get("total_return", 0),
            "Annualized Return": result.get("annualized_return", 0),
            "Sharpe Ratio": result.get("sharpe_ratio", 0),
            "Sortino Ratio": result.get("sortino_ratio", 0),
            "Max Drawdown": result.get("max_drawdown", 0),
            "Volatility": result.get("volatility", 0),
            "Win Rate": result.get("win_rate", 0),
            "Total Trades": result.get("total_trades", 0),
            "Profit Factor": result.get("profit_factor", 0),
        }
        comparison.append(metrics)

    df = pd.DataFrame(comparison)
    df = df.set_index("Strategy")

    return df


if __name__ == "__main__":
    # Test metrics
    print("Testing Performance Metrics...")
    print("=" * 50)

    # Create synthetic returns
    np.random.seed(42)
    returns = np.random.randn(252) * 0.02 + 0.0005  # Daily returns

    # Create equity curve
    equity = 100000 * np.cumprod(1 + returns)

    # Calculate metrics
    metrics = PerformanceMetrics(returns, equity)

    print(metrics.summary())

    # Test trade metrics
    print("\nTesting Trade Metrics...")

    from backtest.engine import Trade

    trades = [
        Trade("TEST", "buy", 100, 100.0, datetime.now(), 110.0, datetime.now(), 1000, 0.1, 10, 5),
        Trade("TEST", "buy", 100, 100.0, datetime.now(), 95.0, datetime.now(), -500, -0.05, 10, 3),
        Trade("TEST", "buy", 100, 100.0, datetime.now(), 105.0, datetime.now(), 500, 0.05, 10, 4),
    ]

    from datetime import datetime
    trade_metrics = TradeMetrics(trades)
    print(trade_metrics.summary())

    print("\nMetrics tests passed!")
