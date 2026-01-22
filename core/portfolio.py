"""
Portfolio management and risk tracking.
Handles position sizing, risk limits, and portfolio analytics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger
from collections import deque


@dataclass
class RiskLimits:
    """Risk management parameters."""
    max_position_size: float = 0.10  # Max 10% per position
    max_sector_exposure: float = 0.30  # Max 30% per sector
    max_portfolio_drawdown: float = 0.15  # 15% drawdown limit
    max_daily_loss: float = 0.03  # 3% daily loss limit
    max_correlation: float = 0.70  # Max correlation between positions
    min_cash_reserve: float = 0.05  # Keep 5% in cash
    max_leverage: float = 1.0  # No leverage by default
    stop_loss_pct: float = 0.05  # 5% stop loss per position


@dataclass
class Trade:
    """Record of a single trade."""
    symbol: str
    side: str  # 'buy' or 'sell'
    shares: float
    price: float
    timestamp: datetime
    fees: float = 0.0
    slippage: float = 0.0
    order_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        """Total cost including fees and slippage."""
        if self.side == "buy":
            return self.shares * self.price + self.fees + self.slippage
        return self.shares * self.price - self.fees - self.slippage

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "shares": self.shares,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "fees": self.fees,
            "total_cost": self.total_cost,
        }


@dataclass
class PositionInfo:
    """Detailed position information."""
    symbol: str
    shares: float
    avg_entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float
    sector: Optional[str] = None
    weight: float = 0.0  # Portfolio weight


class PortfolioManager:
    """
    Manages portfolio state, executes trades, and tracks risk metrics.

    Features:
    - Position tracking and P&L calculation
    - Risk limit enforcement
    - Trade execution with fees and slippage
    - Portfolio analytics and reporting
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        risk_limits: Optional[RiskLimits] = None,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
    ):
        """
        Initialize portfolio manager.

        Args:
            initial_capital: Starting cash
            risk_limits: Risk management parameters
            transaction_cost: Transaction cost as fraction
            slippage: Slippage as fraction
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.risk_limits = risk_limits or RiskLimits()
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        # Position tracking
        self.positions: dict[str, dict] = {}  # symbol -> {shares, avg_price, cost_basis}
        self.realized_pnl: dict[str, float] = {}  # symbol -> realized pnl

        # Trade history
        self.trades: list[Trade] = []

        # Portfolio value history for analytics
        self.value_history: deque = deque(maxlen=252 * 5)  # ~5 years of daily data
        self.daily_returns: deque = deque(maxlen=252)

        # Risk tracking
        self.peak_value = initial_capital
        self.daily_starting_value = initial_capital

        # Sector mapping (can be updated externally)
        self.sector_map: dict[str, str] = {}

        self.logger = logger.bind(component="portfolio")

    def update_prices(self, prices: dict[str, float]):
        """
        Update current prices for all positions.

        Args:
            prices: Dictionary of symbol -> current price
        """
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos["current_price"] = prices[symbol]

    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get detailed position information."""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        market_value = pos["shares"] * pos.get("current_price", pos["avg_price"])
        cost_basis = pos["cost_basis"]
        unrealized_pnl = market_value - cost_basis
        unrealized_pnl_pct = unrealized_pnl / cost_basis if cost_basis > 0 else 0

        return PositionInfo(
            symbol=symbol,
            shares=pos["shares"],
            avg_entry_price=pos["avg_price"],
            current_price=pos.get("current_price", pos["avg_price"]),
            market_value=market_value,
            cost_basis=cost_basis,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            realized_pnl=self.realized_pnl.get(symbol, 0.0),
            sector=self.sector_map.get(symbol),
            weight=market_value / self.total_value if self.total_value > 0 else 0,
        )

    @property
    def position_value(self) -> float:
        """Total value of all positions."""
        return sum(
            pos["shares"] * pos.get("current_price", pos["avg_price"])
            for pos in self.positions.values()
        )

    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        return self.cash + self.position_value

    @property
    def total_return(self) -> float:
        """Total return since inception."""
        return (self.total_value - self.initial_capital) / self.initial_capital

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak."""
        if self.peak_value == 0:
            return 0.0
        return (self.peak_value - self.total_value) / self.peak_value

    def can_trade(self, symbol: str, side: str, shares: float, price: float) -> tuple[bool, str]:
        """
        Check if trade is allowed under risk limits.

        Args:
            symbol: Asset symbol
            side: 'buy' or 'sell'
            shares: Number of shares
            price: Trade price

        Returns:
            (allowed, reason) tuple
        """
        trade_value = shares * price

        if side == "buy":
            # Check cash availability
            total_cost = trade_value * (1 + self.transaction_cost + self.slippage)
            min_cash = self.total_value * self.risk_limits.min_cash_reserve

            if total_cost > self.cash - min_cash:
                return False, f"Insufficient cash (need ${total_cost:,.2f}, have ${self.cash - min_cash:,.2f})"

            # Check position size limit
            new_position_value = trade_value
            if symbol in self.positions:
                current_value = self.positions[symbol]["shares"] * price
                new_position_value += current_value

            max_position_value = self.total_value * self.risk_limits.max_position_size
            if new_position_value > max_position_value:
                return False, f"Position too large (${new_position_value:,.2f} > ${max_position_value:,.2f})"

            # Check sector exposure
            sector = self.sector_map.get(symbol)
            if sector:
                sector_exposure = self._get_sector_exposure(sector) + trade_value
                max_sector = self.total_value * self.risk_limits.max_sector_exposure
                if sector_exposure > max_sector:
                    return False, f"Sector exposure too high ({sector})"

            # Check drawdown limit
            if self.drawdown > self.risk_limits.max_portfolio_drawdown:
                return False, f"Portfolio drawdown limit exceeded ({self.drawdown:.1%})"

            # Check daily loss limit
            daily_return = (self.total_value - self.daily_starting_value) / self.daily_starting_value
            if daily_return < -self.risk_limits.max_daily_loss:
                return False, f"Daily loss limit exceeded ({daily_return:.1%})"

        elif side == "sell":
            # Check if we have the position
            if symbol not in self.positions:
                return False, f"No position in {symbol}"

            if shares > self.positions[symbol]["shares"]:
                return False, f"Insufficient shares (have {self.positions[symbol]['shares']:.2f})"

        return True, "OK"

    def execute_trade(
        self,
        symbol: str,
        side: str,
        shares: float,
        price: float,
        timestamp: Optional[datetime] = None,
        force: bool = False,
    ) -> Optional[Trade]:
        """
        Execute a trade.

        Args:
            symbol: Asset symbol
            side: 'buy' or 'sell'
            shares: Number of shares
            price: Trade price
            timestamp: Trade timestamp
            force: Skip risk checks

        Returns:
            Trade object if successful, None otherwise
        """
        if not force:
            allowed, reason = self.can_trade(symbol, side, shares, price)
            if not allowed:
                self.logger.warning(f"Trade rejected: {reason}")
                return None

        timestamp = timestamp or datetime.now()

        # Calculate fees
        trade_value = shares * price
        fees = trade_value * self.transaction_cost
        slippage_cost = trade_value * self.slippage

        if side == "buy":
            # Adjust price for slippage (worse execution)
            effective_price = price * (1 + self.slippage)
            total_cost = shares * effective_price + fees

            self.cash -= total_cost

            if symbol in self.positions:
                # Add to existing position
                pos = self.positions[symbol]
                total_shares = pos["shares"] + shares
                total_cost_basis = pos["cost_basis"] + (shares * effective_price)
                pos["shares"] = total_shares
                pos["cost_basis"] = total_cost_basis
                pos["avg_price"] = total_cost_basis / total_shares
                pos["current_price"] = price
            else:
                # New position
                self.positions[symbol] = {
                    "shares": shares,
                    "avg_price": effective_price,
                    "cost_basis": shares * effective_price,
                    "current_price": price,
                }

        else:  # sell
            effective_price = price * (1 - self.slippage)
            proceeds = shares * effective_price - fees

            pos = self.positions[symbol]

            # Calculate realized P&L
            cost_per_share = pos["cost_basis"] / pos["shares"]
            realized = (effective_price - cost_per_share) * shares

            self.cash += proceeds
            self.realized_pnl[symbol] = self.realized_pnl.get(symbol, 0) + realized

            # Update position
            pos["shares"] -= shares
            pos["cost_basis"] -= cost_per_share * shares

            if pos["shares"] <= 0.0001:  # Close position if negligible
                del self.positions[symbol]

        # Record trade
        trade = Trade(
            symbol=symbol,
            side=side,
            shares=shares,
            price=price,
            timestamp=timestamp,
            fees=fees,
            slippage=slippage_cost,
        )
        self.trades.append(trade)

        # Update peak value
        self.peak_value = max(self.peak_value, self.total_value)

        self.logger.info(
            f"Executed {side.upper()} {shares:.2f} {symbol} @ ${price:.2f} "
            f"(fees: ${fees:.2f})"
        )

        return trade

    def _get_sector_exposure(self, sector: str) -> float:
        """Calculate total exposure to a sector."""
        exposure = 0.0
        for symbol, pos in self.positions.items():
            if self.sector_map.get(symbol) == sector:
                exposure += pos["shares"] * pos.get("current_price", pos["avg_price"])
        return exposure

    def check_stop_losses(self, prices: dict[str, float]) -> list[str]:
        """
        Check positions against stop loss limits.

        Args:
            prices: Current prices

        Returns:
            List of symbols that hit stop loss
        """
        stopped_out = []

        for symbol, pos in list(self.positions.items()):
            if symbol not in prices:
                continue

            current_price = prices[symbol]
            pnl_pct = (current_price - pos["avg_price"]) / pos["avg_price"]

            if pnl_pct < -self.risk_limits.stop_loss_pct:
                stopped_out.append(symbol)
                self.logger.warning(
                    f"Stop loss triggered for {symbol}: {pnl_pct:.1%} "
                    f"(limit: {-self.risk_limits.stop_loss_pct:.1%})"
                )

        return stopped_out

    def record_daily_value(self):
        """Record end-of-day portfolio value."""
        value = self.total_value
        self.value_history.append({
            "timestamp": datetime.now(),
            "value": value,
        })

        if len(self.value_history) > 1:
            prev_value = self.value_history[-2]["value"]
            daily_return = (value - prev_value) / prev_value if prev_value > 0 else 0
            self.daily_returns.append(daily_return)

        # Reset daily starting value
        self.daily_starting_value = value

    def get_analytics(self) -> dict:
        """Calculate portfolio analytics."""
        returns = np.array(list(self.daily_returns)) if self.daily_returns else np.array([0])

        # Calculate metrics
        total_return = self.total_return
        sharpe = self._calculate_sharpe(returns)
        sortino = self._calculate_sortino(returns)
        max_dd = self._calculate_max_drawdown()
        win_rate = self._calculate_win_rate()
        profit_factor = self._calculate_profit_factor()

        return {
            "total_value": self.total_value,
            "cash": self.cash,
            "position_value": self.position_value,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "current_drawdown": self.drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(self.trades),
            "num_positions": len(self.positions),
            "total_realized_pnl": sum(self.realized_pnl.values()),
        }

    def _calculate_sharpe(self, returns: np.ndarray, risk_free: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        excess_returns = returns - risk_free / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def _calculate_sortino(self, returns: np.ndarray, risk_free: float = 0.02) -> float:
        """Calculate annualized Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free / 252
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0 if excess_returns.mean() <= 0 else float("inf")
        return np.sqrt(252) * excess_returns.mean() / downside.std()

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from value history."""
        if len(self.value_history) < 2:
            return 0.0

        values = [v["value"] for v in self.value_history]
        peak = values[0]
        max_dd = 0.0

        for value in values[1:]:
            peak = max(peak, value)
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

    def _calculate_win_rate(self) -> float:
        """Calculate win rate from closed trades."""
        if not self.realized_pnl:
            return 0.0

        wins = sum(1 for pnl in self.realized_pnl.values() if pnl > 0)
        return wins / len(self.realized_pnl)

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if not self.realized_pnl:
            return 0.0

        gross_profit = sum(pnl for pnl in self.realized_pnl.values() if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in self.realized_pnl.values() if pnl < 0))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def get_positions_summary(self) -> pd.DataFrame:
        """Get summary of all positions as DataFrame."""
        if not self.positions:
            return pd.DataFrame()

        records = []
        for symbol in self.positions:
            pos_info = self.get_position(symbol)
            if pos_info:
                records.append({
                    "symbol": pos_info.symbol,
                    "shares": pos_info.shares,
                    "avg_price": pos_info.avg_entry_price,
                    "current_price": pos_info.current_price,
                    "market_value": pos_info.market_value,
                    "unrealized_pnl": pos_info.unrealized_pnl,
                    "unrealized_pnl_pct": pos_info.unrealized_pnl_pct,
                    "weight": pos_info.weight,
                    "sector": pos_info.sector,
                })

        return pd.DataFrame(records)

    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.realized_pnl = {}
        self.trades = []
        self.value_history.clear()
        self.daily_returns.clear()
        self.peak_value = self.initial_capital
        self.daily_starting_value = self.initial_capital
        self.logger.info("Portfolio reset")


class RiskManager:
    """
    Standalone risk management component.
    Can be used independently or with PortfolioManager.
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.logger = logger.bind(component="risk")

    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss_price: float,
        risk_per_trade: float = 0.01,
    ) -> float:
        """
        Calculate position size using risk-based sizing.

        Args:
            portfolio_value: Total portfolio value
            entry_price: Expected entry price
            stop_loss_price: Stop loss price
            risk_per_trade: Maximum risk per trade as fraction

        Returns:
            Number of shares to trade
        """
        risk_amount = portfolio_value * risk_per_trade
        risk_per_share = abs(entry_price - stop_loss_price)

        if risk_per_share == 0:
            return 0.0

        shares = risk_amount / risk_per_share

        # Apply position size limit
        max_shares = (portfolio_value * self.limits.max_position_size) / entry_price
        shares = min(shares, max_shares)

        return shares

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.25,
    ) -> float:
        """
        Calculate Kelly criterion position size.

        Args:
            win_rate: Historical win rate
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
            kelly_fraction: Fraction of Kelly to use (default 0.25 = quarter Kelly)

        Returns:
            Recommended position size as fraction of portfolio
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0

        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - p
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p

        kelly = (b * p - q) / b

        # Apply fraction and limits
        size = kelly * kelly_fraction
        size = max(0, min(size, self.limits.max_position_size))

        return size

    def check_correlation_risk(
        self,
        returns: pd.DataFrame,
        new_symbol: str,
        existing_symbols: list[str],
    ) -> tuple[bool, float]:
        """
        Check if adding a new position would violate correlation limits.

        Args:
            returns: DataFrame of historical returns
            new_symbol: Symbol to add
            existing_symbols: Current portfolio symbols

        Returns:
            (allowed, max_correlation) tuple
        """
        if new_symbol not in returns.columns:
            return True, 0.0

        if not existing_symbols:
            return True, 0.0

        existing = [s for s in existing_symbols if s in returns.columns]
        if not existing:
            return True, 0.0

        correlations = returns[existing].corrwith(returns[new_symbol])
        max_corr = correlations.abs().max()

        return max_corr <= self.limits.max_correlation, max_corr
